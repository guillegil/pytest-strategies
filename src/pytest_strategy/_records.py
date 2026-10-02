"""
Record mode: a test that receives a strategy's row as one object.

A test receives the row as one record when (1) neither the test nor any fixture
it uses asks for one of the strategy's argument names, and (2) exactly one test
parameter is annotated with a record type whose fields are exactly those names.
Otherwise the strategy passes its arguments by name, one test parameter each.

Four kinds of record types are recognized, and their field sets are fixed, so a
later release that builds more of them cannot change which parameter matches:

- dataclasses, pydantic dataclasses included: the ``init=True`` fields. They are
  the only kind built;
- NamedTuple: ``_fields``;
- TypedDict: the required and the optional keys;
- pydantic v2 models: the ``model_fields`` names (not aliases). pydantic is found
  through ``sys.modules``, never imported.

Unions and ``Optional``, unresolvable annotations, pydantic v1 models, attrs
classes and ``Vector`` are not record types.

Pure functions — no pytest runtime dependency beyond inspect/dataclasses.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import types
import typing
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal, NamedTuple

from ._introspection import PYTEST_FIXTURES, lazy_signature
from ._vector import Vector

RecordKind = Literal["dataclass", "namedtuple", "typeddict", "pydantic"]

# How messages name each kind of record type
_KIND_NAMES: dict[str, str] = {
    "dataclass": "a dataclass",
    "namedtuple": "a NamedTuple",
    "typeddict": "a TypedDict",
    "pydantic": "a pydantic model",
}

# An annotation that names something the test module cannot resolve
_UNRESOLVED: Any = object()


class RecordParam(NamedTuple):
    """The test parameter that receives a strategy's row as one record."""

    name: str
    record_type: type
    kind: RecordKind
    fields: frozenset[str]


def record_class(annotation: object) -> type | None:
    """
    Return the class an annotation names, after stripping ``Annotated[...]`` and
    generic arguments (``P[int]`` becomes ``P``), or None for anything else, such as
    a union.
    """
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    origin = typing.get_origin(annotation)
    if origin is not None:
        if _is_union_origin(origin):
            return None
        annotation = origin
    return annotation if isinstance(annotation, type) else None


def record_kind(cls: type) -> RecordKind | None:
    """Say which kind of record type ``cls`` is, or None when it is not one."""
    if issubclass(cls, Vector):
        # A Vector is a namedtuple, but it is the row itself, not a record of it
        return None
    if dataclasses.is_dataclass(cls):
        return "dataclass"
    if issubclass(cls, tuple) and isinstance(getattr(cls, "_fields", None), tuple):
        return "namedtuple"
    if (
        issubclass(cls, dict)
        and hasattr(cls, "__required_keys__")
        and hasattr(cls, "__optional_keys__")
    ):
        # typing's and typing_extensions' TypedDicts alike
        return "typeddict"
    if _is_pydantic_model(cls):
        return "pydantic"
    return None


def record_fields(cls: type, kind: RecordKind) -> frozenset[str]:
    """Return the field names of a record type, which must equal the argument names."""
    record: Any = cls
    if kind == "dataclass":
        return frozenset(f.name for f in dataclasses.fields(record) if f.init)
    if kind == "namedtuple":
        return frozenset(record._fields)
    if kind == "typeddict":
        return frozenset(record.__required_keys__ | record.__optional_keys__)
    # The field names, not their aliases; computed fields are not in model_fields
    return frozenset(record.model_fields)


def _is_pydantic_model(cls: type) -> bool:
    """Return True for a pydantic v2 model class, without importing pydantic."""
    pydantic = sys.modules.get("pydantic")
    base = getattr(pydantic, "BaseModel", None)
    if not isinstance(base, type) or not issubclass(cls, base):
        # pydantic.v1 models do not subclass pydantic.BaseModel
        return False
    # pydantic v1's BaseModel has no model_fields
    return isinstance(getattr(cls, "model_fields", None), dict)


def _is_union_origin(origin: object) -> bool:
    """Return True for the origin of ``Union[...]``, ``Optional[...]`` or ``X | Y``."""
    return origin is typing.Union or origin is types.UnionType


def detect_record_param(
    test_fn: Callable[..., Any],
    argnames: Sequence[str],
    fixturenames: Collection[str] | None,
    *,
    pytest_fixtures: frozenset[str] | set[str] = PYTEST_FIXTURES,
) -> RecordParam | None:
    """
    Return the test parameter that receives the strategy's row as one record.

    The test is in record mode when no name in *fixturenames* is one of the
    *argnames* and exactly one test parameter is annotated with a record type whose
    fields are exactly the *argnames*. A test parameter is one that pytest fills: no
    default, not ``*args``, ``**kwargs`` or positional-only, and not ``self``,
    ``cls`` or a name in *pytest_fixtures*. When no annotation matches exactly but a
    single parameter has a record type, the missing and extra fields are reported:
    a dataclass parameter is still returned, so that building the records reports
    them, as 3.0 did, and any other kind fails here.

    String annotations (``from __future__ import annotations`` or quoted forward
    references) are resolved in the test module's globals, so the record type must
    be defined at module level.

    Args:
        test_fn: The test function
        argnames: The strategy's argument names
        fixturenames: The names the test asks for: its parameters, its usefixtures,
            the autouse fixtures and what they ask for in turn
            (``metafunc.fixturenames``). None reads the test's own parameters.
        pytest_fixtures: Built-in fixture names, which are never record parameters

    Returns:
        The record parameter, or None when the strategy passes its arguments by name

    Raises:
        ValueError: When two parameters match exactly, or when the record type is a
            NamedTuple, TypedDict or pydantic model, which are not built yet (listing
            the missing and extra fields of one that does not match)
    """
    asked = set(lazy_signature(test_fn).parameters) if fixturenames is None else fixturenames
    if any(name in asked for name in argnames):
        return None

    candidates = _candidates(test_fn, pytest_fixtures)
    wanted = set(argnames)
    exact = [c for c in candidates if c.fields == wanted]
    if len(exact) > 1:
        listed = _join([f"'{c.name}' ({c.record_type.__name__})" for c in exact])
        raise ValueError(
            f"parameters {listed} are each annotated with a record type whose fields are "
            f"the strategy's arguments ({', '.join(argnames)}), so it is not clear which "
            "one receives the row. Annotate only one of them with a record type, or take "
            "the arguments as parameters."
        )
    if exact:
        record = exact[0]
    elif len(candidates) == 1:
        record = candidates[0]
        if record.kind != "dataclass":
            raise ValueError(_mismatch(record, argnames))
        # convert_to_dataclass lists the missing and extra fields
        return record
    else:
        return None
    if record.kind != "dataclass":
        raise ValueError(
            f"parameter '{record.name}' is annotated with {record.record_type.__name__}, "
            f"{_KIND_NAMES[record.kind]}; record mode supports dataclasses (NamedTuple, "
            "TypedDict and pydantic models are not supported yet). Take the arguments as "
            "parameters or use a dataclass."
        )
    return record


def _mismatch(record: RecordParam, argnames: Sequence[str]) -> str:
    """Describe a record type that is not a dataclass and whose fields are not the arguments."""
    cls = record.record_type.__name__
    missing = [name for name in argnames if name not in record.fields]
    extra = sorted(record.fields - set(argnames))
    problems = []
    if missing:
        problems.append(f"missing {_join([repr(name) for name in missing])}")
    if extra:
        problems.append(f"extra {_join([repr(name) for name in extra])}")
    return (
        f"parameter '{record.name}' is annotated with {cls}, {_KIND_NAMES[record.kind]}, "
        f"whose fields do not match the strategy's arguments ({', '.join(argnames)}): "
        f"{'; '.join(problems)}. Record mode supports dataclasses (NamedTuple, TypedDict "
        "and pydantic models are not supported yet). Take the arguments as parameters or "
        "use a dataclass with those fields."
    )


def record_hints(
    test_fn: Callable[..., Any],
    argnames: Sequence[str],
    fixturenames: Collection[str] | None,
    *,
    pytest_fixtures: frozenset[str] | set[str] = PYTEST_FIXTURES,
) -> list[str]:
    """
    Say why no parameter of the test receives the row as a record.

    For the signature error of a test that does not take every argument by name:
    a fixture asks for an argument, so the strategy passes its arguments by name
    although a parameter has a record type; a parameter annotated with a record
    type whose fields are the arguments has a default, so pytest does not fill it,
    or is annotated with a union of it; or a parameter's annotation cannot be
    resolved, in a test that looks written for record mode. Other parameters, such
    as a fixture annotated with a type imported under TYPE_CHECKING, get no hint.
    """
    hints: list[str] = []
    params = lazy_signature(test_fn).parameters
    wanted = set(argnames)
    exact = [c for c in _candidates(test_fn, pytest_fixtures) if c.fields == wanted]
    asked = [
        name
        for name in argnames
        if fixturenames is not None and name in fixturenames and name not in params
    ]
    if asked and exact:
        record = exact[0]
        hints.append(
            f"A fixture of the test asks for {_join([repr(a) for a in asked])}, so the "
            f"strategy passes its arguments by name: parameter '{record.name}' "
            f"({record.record_type.__name__}) receives the row as a record only when "
            "no fixture asks for an argument."
        )
    # An annotation that cannot be resolved can only be the missing record type when
    # the test takes no argument by name and no parameter has the record type already
    record_mode_test = not exact and not any(name in params for name in argnames)
    for param, raw, annotation in _annotated_parameters(test_fn, pytest_fixtures):
        if annotation is _UNRESOLVED:
            if record_mode_test:
                hints.append(
                    f"Parameter '{param.name}' is annotated with {_annotation_text(raw)!r}, "
                    "which cannot be resolved in the test module's globals, so it is not a "
                    "record type: define the class at module level."
                )
        elif _is_union_origin(typing.get_origin(annotation)):
            if any(_has_fields(arg, wanted) for arg in typing.get_args(annotation)):
                hints.append(
                    f"Parameter '{param.name}' is annotated with {_annotation_text(raw)}, "
                    "a union, which is not a record type."
                )
        elif param.default is not param.empty and _has_fields(annotation, wanted):
            hints.append(
                f"Parameter '{param.name}' has a default, so pytest does not fill it and "
                "it does not receive the row as a record."
            )
    return hints


def matching_record_params(
    test_fn: Callable[..., Any],
    argnames: Sequence[str],
    *,
    pytest_fixtures: frozenset[str] | set[str] = PYTEST_FIXTURES,
) -> list[RecordParam]:
    """
    Return the test parameters annotated with a record type whose fields are
    exactly *argnames*, in signature order.

    In named mode (a fixture asks for an argument) such a parameter receives
    nothing from the strategy: a fixture or a parametrization must give it a value.
    """
    wanted = set(argnames)
    return [
        c
        for c in _candidates(test_fn, pytest_fixtures)
        if c.fields == wanted and c.name not in wanted
    ]


def _has_fields(annotation: object, fields: set[str]) -> bool:
    """Return True when an annotation names a record type whose fields are ``fields``."""
    cls = record_class(annotation)
    kind = record_kind(cls) if cls is not None else None
    return cls is not None and kind is not None and record_fields(cls, kind) == fields


def _candidates(
    test_fn: Callable[..., Any], pytest_fixtures: frozenset[str] | set[str]
) -> list[RecordParam]:
    """
    Return the test parameters annotated with a record type, in signature order.

    A test parameter is one that pytest fills (see :func:`detect_record_param`).
    """
    candidates = []
    for param, _, annotation in _annotated_parameters(test_fn, pytest_fixtures):
        cls = record_class(annotation)
        kind = record_kind(cls) if cls is not None else None
        if cls is not None and kind is not None and param.default is param.empty:
            candidates.append(RecordParam(param.name, cls, kind, record_fields(cls, kind)))
    return candidates


def _annotated_parameters(
    test_fn: Callable[..., Any], pytest_fixtures: frozenset[str] | set[str]
) -> list[tuple[inspect.Parameter, Any, Any]]:
    """
    Return the test's annotated parameters with the annotation as written and as
    resolved (``_UNRESOLVED`` when it cannot be).

    ``*args``, ``**kwargs``, positional-only parameters, ``self``, ``cls`` and the
    built-in fixtures are left out; parameters with a default are kept.
    """
    sig = lazy_signature(test_fn)
    try:
        hints = typing.get_type_hints(test_fn, include_extras=True)
    except Exception:
        # One unresolvable annotation (e.g. a fixture type imported under
        # TYPE_CHECKING) must not disable record mode: fall back to the raw
        # annotations and resolve them one by one below.
        hints = {}

    annotated = []
    for name, param in sig.parameters.items():
        if (
            param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            or param.annotation is param.empty
            or name in ("self", "cls")
            or name in pytest_fixtures
        ):
            continue
        annotation = hints.get(name, param.annotation)
        if isinstance(annotation, str):
            annotation = _eval_annotation(test_fn, annotation)
        elif isinstance(annotation, typing.ForwardRef):
            # Python 3.14 keeps a name it cannot resolve as a ForwardRef
            annotation = _UNRESOLVED
        annotated.append((param, param.annotation, annotation))
    return annotated


def _eval_annotation(test_fn: Callable[..., Any], annotation: str) -> Any:
    """Evaluate a string annotation in *test_fn*'s module globals (``_UNRESOLVED`` on failure)."""
    module_globals = getattr(inspect.unwrap(test_fn), "__globals__", {})
    try:
        return eval(annotation, module_globals)
    except Exception:
        return _UNRESOLVED


def _annotation_text(annotation: object) -> str:
    """Return an annotation as it is written."""
    if isinstance(annotation, typing.ForwardRef):
        return annotation.__forward_arg__
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _join(items: Sequence[str]) -> str:
    """Join ``['a', 'b', 'c']`` as ``a, b and c``."""
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def convert_to_dataclass(
    samples: Sequence[tuple[Any, ...]],
    argnames: Sequence[str],
    dataclass_type: type,
) -> list[Any]:
    """
    Convert a sequence of tuple samples to dataclass instances.

    The function validates that the dataclass ``__init__`` fields (fields with
    ``init=False`` are skipped) exactly match the strategy *argnames*, then
    builds each instance with keyword arguments, so argname order and
    keyword-only fields do not matter. A hand-written ``__init__`` that does not
    take the field names as keywords (other names, or positional-only) gets the
    values positionally, in field declaration order.

    Args:
        samples: Sequence of value tuples from the strategy.
        argnames: Argument names from the strategy (must match dataclass fields).
        dataclass_type: The dataclass type to instantiate.

    Returns:
        List of dataclass instances.

    Raises:
        ValueError: When the dataclass fields do not match *argnames*.
    """
    dc_fields = {f.name for f in dataclasses.fields(dataclass_type) if f.init}
    strategy_fields = set(argnames)

    if dc_fields != strategy_fields:
        missing = strategy_fields - dc_fields
        extra = dc_fields - strategy_fields

        error_msg = "Dataclass fields don't match strategy parameters!\n"
        error_msg += f"  Strategy provides: {list(argnames)}\n"
        error_msg += f"  Dataclass expects: {list(dc_fields)}\n"

        if missing:
            error_msg += f"  Missing in dataclass: {list(missing)}\n"
        if extra:
            error_msg += f"  Extra in dataclass: {list(extra)}\n"

        raise ValueError(error_msg)

    try:
        lazy_signature(dataclass_type).bind(**dict.fromkeys(argnames))
    except TypeError:
        # Hand-written __init__ with other or positional-only parameter names
        order = [f.name for f in dataclasses.fields(dataclass_type) if f.init]
        return [
            dataclass_type(*(values[name] for name in order))
            for values in (dict(zip(argnames, sample)) for sample in samples)
        ]
    except ValueError:
        # No inspectable signature: keep the keyword construction
        pass

    return [dataclass_type(**dict(zip(argnames, sample))) for sample in samples]
