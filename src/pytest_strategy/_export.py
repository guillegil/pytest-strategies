"""
The strategy export, schema 1: the document ``export_strategies()`` returns, and
the fragments ``Parameter.to_dict()``, ``TestArg.to_dict()`` and
``RNGType.to_dict()`` write.

The document::

    {"schema": 1, "kind": "strategies",
     "generator": {"name": "pytest-strategies", "version": "4.0.0"},
     "seed": 1, "nsamples": 10,
     "strategies": [
       {"name": "burst",
        "origin": {"folder": "tests/dma", "file": "tests/dma/strategies.py",
                   "qualname": "burst", "line": 12},
        "context": "3f2a9c1e",
        "parameter": {"schema": 1, "arguments": [...], ...}},
       {"name": "broken", "origin": {...}, "context": null,
        "error": {"type": "RuntimeError", "message": "boom"}}]}

Every registration is an entry, sorted by name and folder. An entry has a
``parameter``; an ``error``, with a ``note`` when the plugin adds one (why ``ctx``
was None, for example); or, for a factory that declares ``ctx`` in a folder whose
``conftest.py`` was not loaded, ``unavailable``. Values are written in the
encoding of ``_encode.py``.

Evolution: readers ignore the keys, and the values of string enums (``kind``,
``source``, ``VectorInfo.kind``), they do not know; 4.x may add both within
schema 1 (``source: "dependent"`` is reserved for 4.1), and an unknown
``$``-tagged value is read like a ``$repr``. Removing, renaming or retyping a key
bumps the schema.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._context import below, unloaded_conftests
from ._encode import encode
from ._factory import FactoryError, FactoryInputs, analyse, call_factory
from ._ids import names_id
from ._registry import Factory, display_path, factory_source, registry
from ._runtime import runtime
from ._streams import INSTALLED_FOLDERS, StreamKey, path_part, seed_part
from .parameters import _ParameterSet
from .rng import (
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    _Stream,
)

if TYPE_CHECKING:
    import pytest

    from .parameters import Parameter
    from .test_args import TestArg

# The schema version of the documents and of Parameter.to_dict()
SCHEMA = 1

# The kind of document export_strategies() returns
KIND = "strategies"


# ---------------------------------------------------------------------------
# RNG types, arguments and Parameters
# ---------------------------------------------------------------------------


def _predicate(rng_type: Any) -> dict[str, Any]:
    """Whether a predicate filters the draws (the function itself is not written)."""
    return {"predicate": rng_type.predicate is not None}


def _bounds(rng_type: RNGInteger | RNGFloat) -> dict[str, Any]:
    return {"min": encode(rng_type.min), "max": encode(rng_type.max), **_predicate(rng_type)}


def _ranges(rng_type: RNGWeightedInteger | RNGWeightedFloat) -> dict[str, Any]:
    ranges = [
        {"min": encode(low), "max": encode(high), "weight": encode(weight)}
        for (low, high), weight in rng_type.ranges.items()
    ]
    return {"ranges": ranges, **_predicate(rng_type)}


def _enum(rng_type: RNGEnum[Any]) -> dict[str, Any]:
    weights = rng_type.weights
    return {
        "enum": rng_type.enum_class.__qualname__,
        "members": [encode(member) for member in rng_type.enum_class],
        "weights": (
            None
            if weights is None
            else [
                {"member": encode(member), "weight": encode(weight)}
                for member, weight in weights.items()
            ]
        ),
        **_predicate(rng_type),
    }


def _sequence(rng_type: RNGSequence[Any] | Series[Any]) -> dict[str, Any]:
    # A predicate was applied when the sequence was built: these are the values left
    return {
        "sequence": [encode(value) for value in rng_type.sequence],
        "skip_if_empty": rng_type.skip_if_empty,
    }


def _string(rng_type: RNGString) -> dict[str, Any]:
    return {
        "length": rng_type.length,
        "min_length": rng_type.min_length,
        "max_length": rng_type.max_length,
        "charset": rng_type.charset,
    }


# The typed fields of each built-in RNG type, by exact class: a subclass is a
# custom type, written by its public attributes
_TYPED: dict[type, Callable[[Any], dict[str, Any]]] = {
    RNGInteger: _bounds,
    RNGFloat: _bounds,
    RNGWeightedInteger: _ranges,
    RNGWeightedFloat: _ranges,
    RNGBoolean: lambda rng_type: {"true_probability": encode(rng_type.true_probability)},
    RNGChoice: lambda rng_type: {"choices": [encode(value) for value in rng_type.choices]},
    RNGEnum: _enum,
    RNGSequence: _sequence,
    Series: _sequence,
    RNGString: _string,
}


def _public_attributes(obj: Any) -> Iterable[tuple[str, Any]]:
    """The public instance attributes of ``obj``: its ``__dict__``'s, then its slots'."""
    seen: set[str] = set()
    for name, value in getattr(obj, "__dict__", {}).items():
        seen.add(name)
        if not name.startswith("_"):
            yield name, value
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            try:
                value = getattr(obj, name)
            except AttributeError:
                # A slot that was never set
                continue
            yield name, value


def rng_type_dict(rng_type: Any) -> dict[str, Any]:
    """
    Describe an RNG type (``RNGType.to_dict()``): ``{"type": <class qualname>, ...}``
    with the typed fields of a built-in type, or for any other class (a subclass of
    a built-in one included) ``attributes``: its public instance attributes, each
    in the value encoding.
    """
    data: dict[str, Any] = {"type": type(rng_type).__qualname__}
    typed = _TYPED.get(type(rng_type))
    if typed is not None:
        data.update(typed(rng_type))
    else:
        data["attributes"] = {name: encode(value) for name, value in _public_attributes(rng_type)}
    return data


def _python_type(arg: TestArg) -> str | None:
    """The qualified name of the type of an argument's values, or None when unknown."""
    try:
        python_type = arg.type
    except Exception:
        # An RNGType subclass without its own python_type raises NotImplementedError
        return None
    if python_type is Any or not isinstance(python_type, type):
        return None
    return python_type.__qualname__


def argument_dict(arg: TestArg) -> dict[str, Any]:
    """
    Describe a test argument (``TestArg.to_dict()``): its name, description,
    ``python_type``, whether a validator is set, and its ``source``: ``"value"``
    with the ``value``, or ``"rng"`` with what its RNG type's ``to_dict()`` returns.
    """
    data: dict[str, Any] = {
        "name": arg.name,
        "description": arg.description,
        "python_type": _python_type(arg),
        "validator": arg._validator is not None,
    }
    if arg.is_static:
        data["source"] = "value"
        data["value"] = encode(arg._value)
    else:
        data["source"] = "rng"
        rng_type = arg.rng_type
        # An RNGType's to_dict(), which a subclass may override, or for an object
        # that only has a generate() method, its attributes
        data["rng"] = (
            rng_type.to_dict() if isinstance(rng_type, RNGType) else rng_type_dict(rng_type)
        )
    return data


def _vectors(param: Parameter, kind: str) -> list[dict[str, Any]]:
    """The directed or the test vectors, in order: name, names-format ID and values."""
    stored = param._directed_vectors if kind == "directed" else param._test_vectors
    names = param.arg_names
    vectors = []
    for name, vector in stored.items():
        # A pytest.param(...) vector's values; its marks are not written
        values = vector.values if isinstance(vector, _ParameterSet) else vector
        vectors.append(
            {
                "name": name,
                "id": names_id(kind, name, None, ()),
                "values": {arg: encode(value) for arg, value in zip(names, values, strict=True)},
            }
        )
    return vectors


def parameter_dict(param: Parameter, constraints_off: Iterable[str] = ()) -> dict[str, Any]:
    """
    Describe a Parameter (``Parameter.to_dict()``), with ``"schema": 1``.

    Args:
        param: The Parameter
        constraints_off: The names of the constraints turned off for its strategy
            in this run, which get ``"enabled": false``
    """
    off = frozenset(constraints_off)
    return {
        "schema": SCHEMA,
        "arguments": [argument_dict(arg) for arg in param.test_args],
        "directed_vectors": _vectors(param, "directed"),
        "test_vectors": _vectors(param, "test"),
        "constraints": [
            {"name": name, "enabled": name not in off} for name in param.vector_constraints
        ],
        "always_include_directed": bool(param.always_include_directed),
        "max_retries": param.max_retries,
        "nsamples": param.nsamples,
        "per_sequence_samples": param.per_sequence_samples,
        "max_exhaustive": param.max_exhaustive,
        "skip_reason": param.skip_reason,
    }


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def document() -> dict[str, Any]:
    """
    Return the export of every registration (see ``export_strategies()``), after
    loading every strategy file of the active session.
    """
    from . import __version__

    runtime.load_all_strategy_files()
    config = runtime.current.config if runtime.current is not None else None
    entries = [
        _entry(name, registration.factory, config)
        for name in registry.names()
        for registration in registry.registrations(name)
    ]
    entries.sort(key=lambda entry: (entry["name"], entry["origin"]["folder"] or ""))
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "generator": {"name": "pytest-strategies", "version": __version__},
        "seed": encode(runtime.run_seed()),
        "nsamples": runtime.session_options(config).base.nsamples,
        "strategies": entries,
    }


def _entry(name: str, factory: Factory, config: pytest.Config | None) -> dict[str, Any]:
    """Export one registration: call its factory as collection does, and describe it."""
    # Imported here: the plugin imports the package, which imports this module
    from ._resolver import check_factory_result
    from .plugin import definition_part

    rootpath = getattr(config, "rootpath", None)
    entry: dict[str, Any] = {"name": name, "origin": origin(factory, rootpath), "context": None}
    # The factory's module's name, or for a strategy file, a test module or a
    # conftest.py its folder, as its file system spells it (see source_part, as for
    # a fixture's key)
    folder = definition_part(factory, config, folder=True)
    stream = StreamKey.root(seed_part(runtime.run_seed()), "export", name, folder)
    # The context of the folder the factory is registered in (or the rootdir's),
    # computed only if the factory asks for it
    where = context_folder(factory, rootpath)
    context = runtime.path_context(where)
    received: list[bool] = []

    def ctx() -> Any:
        value = context()
        received.append(True)
        return value

    described: dict[str, Any]
    try:
        if where is not None and "ctx" in analyse(factory).declares:
            # A test there would get a context that may come from these
            unloaded = unloaded_conftests(config, where)
            if unloaded:
                entry["unavailable"] = not_loaded(unloaded, rootpath)
                return entry
        # The session's options for this strategy, the instance collection uses,
        # and the random stream root(S, "export", name, folder) (streams v1)
        options = runtime.strategy_options(name, config)
        with _Stream(stream) as rng:
            inputs = FactoryInputs(options=options, rng=rng, ctx=ctx, why_no_ctx=context.why_none)
            result = call_factory(name, factory, inputs, rootpath=rootpath)
        param = check_factory_result(name, factory, result)
        described = {"parameter": parameter_dict(param, options.constraints_off)}
        # An RNG type's own to_dict() may return what JSON cannot hold: then this
        # entry, not the whole export, reports it
        json.dumps(described, allow_nan=False)
    except Exception as e:
        described = {"error": _error(e)}
    if received:
        # The fingerprint the context got when its implementation returned it
        entry["context"] = context.answer().fingerprint
    entry.update(described)
    return entry


def _error(error: Exception) -> dict[str, str]:
    """
    Describe why a registration could not be exported: what its factory raised,
    with the plugin's note when it adds one, or the plugin's own error (a signature
    it cannot call, a result that is not a Parameter, a context hook that raised).
    """
    if isinstance(error, FactoryError):
        data = {"type": type(error.error).__qualname__, "message": str(error.error)}
        if error.note:
            data["note"] = error.note
        return data
    return {"type": type(error).__qualname__, "message": str(error)}


def _posix(path: str | os.PathLike[str], rootpath: str | os.PathLike[str] | None) -> str:
    """A real path relative to the rootdir, or absolute outside it, in posix form."""
    return Path(display_path(path, rootpath)).as_posix()


def origin(factory: Factory, rootpath: str | os.PathLike[str] | None) -> dict[str, Any]:
    """
    Return where a factory is defined: its file's ``folder`` and ``file``, relative
    to the rootdir in posix form (absolute outside it; None when the code has no
    file), its ``qualname`` and its first ``line``.
    """
    filename, qualname, line = factory_source(factory)
    folder = file = None
    # exec'd code has a made-up file name ("<string>")
    if filename and os.path.isfile(filename):
        folder = _posix(os.path.dirname(os.path.abspath(filename)), rootpath)
        file = _posix(filename, rootpath)
    return {"folder": folder, "file": file, "qualname": qualname, "line": line}


def context_folder(
    factory: Factory, rootpath: str | os.PathLike[str] | None
) -> str | os.PathLike[str] | None:
    """
    Return the folder whose context ``export_strategies()`` gives a factory, the
    context a test there gets: its file's folder, when that is inside the rootdir
    (as it is spelled, or by its real path) and not in an installed package (a
    ``site-packages`` or ``dist-packages`` folder below the rootdir, such as a
    virtualenv's, which pytest does not collect; one above the rootdir, which a
    checkout may be in, does not count); otherwise the rootdir. None without a
    rootdir (outside a session).
    """
    if rootpath is None:
        return None
    source = factory_source(factory)[0]
    if not source or not os.path.isfile(source):
        return rootpath
    folder = os.path.dirname(os.path.abspath(source))
    parts = below(folder, rootpath)
    if parts is None or not INSTALLED_FOLDERS.isdisjoint(parts):
        return rootpath
    return folder


def not_loaded(conftests: list[str], rootpath: str | os.PathLike[str] | None) -> str:
    """Say which ``conftest.py`` files were not loaded, relative to the rootdir."""
    names = [path_part(conftest, rootpath) for conftest in conftests]
    if len(names) == 1:
        return f"{names[0]} was not loaded in this session"
    return f"{', '.join(names[:-1])} and {names[-1]} were not loaded in this session"
