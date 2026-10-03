"""
The fingerprint of a testbench context: a short hash of the object a
``pytest_strategies_context`` implementation returned, so two runs (CI and a local
rerun, or two pytest-xdist workers) can tell whether their factories received the
same context.

It is the first 8 hex characters of the SHA-256 of a canonical JSON encoding of the
object, which depends only on what the object holds: not on ``PYTHONHASHSEED``,
memory addresses, ``--import-mode`` or the folder the checkout is in.

- None, bool, int and str are written as they are; a float as its repr, so NaN and
  the infinities work; an Enum member as its type's qualified name and its name;
  bytes as hex.
- A path inside the rootdir is written relative to it, in posix form, so two
  checkouts agree; another path in posix form. A date, time, datetime, timedelta,
  Decimal, UUID or complex is a tagged string.
- A pydantic v2 model, recognized by its ``model_fields`` and ``model_dump``
  without importing pydantic, is written as ``model_dump(mode="python")``, which
  leaves out its ``Field(exclude=True)`` fields and keeps a ``SecretStr`` masked.
- Dataclasses and NamedTuples are written field by field, mappings as their
  (key, value) pairs in their order, lists and tuples as arrays, and sets with
  their elements sorted by their encoding. A container met again inside itself is
  written as a reference to it, so a cycle ends.
- A class is written as its qualified name. Anything else is written as its repr,
  without the memory addresses in it (`` at 0x7f...``). An object whose type keeps
  the default repr (``<Plain object at 0x...>``) is written as its type's name
  alone: its state is not in the fingerprint, which says so by listing the type as
  *partial*.

Type names are qualified names, never module names, which depend on
``--import-mode``. An object that cannot be encoded (its repr raises, or it nests
deeper than the recursion limit) gives the fingerprint ``unavailable``: the run
never fails for it.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Set
from pathlib import Path, PurePath
from typing import Any

# The fingerprint of a context that could not be encoded
UNAVAILABLE = "unavailable"

# The hex characters of the SHA-256 that make the fingerprint
_LENGTH = 8

# A memory address in a repr: "<Plain object at 0x7f3a2b1c>", "<function f at 0x...>"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

# The types written as a tagged string: (type, tag, the string), datetime before
# date, which it subclasses
_TAGGED: tuple[tuple[type, str, Callable[[Any], str]], ...] = (
    (datetime.datetime, "datetime", datetime.datetime.isoformat),
    (datetime.date, "date", datetime.date.isoformat),
    (datetime.time, "time", datetime.time.isoformat),
    (datetime.timedelta, "timedelta", str),
    (decimal.Decimal, "decimal", str),
    (uuid.UUID, "uuid", str),
    (complex, "complex", repr),
)


def fingerprint(
    value: Any, rootpath: str | os.PathLike[str] | None = None
) -> tuple[str, tuple[str, ...]]:
    """
    Return the fingerprint of a context (see the module docstring).

    Args:
        value: The context
        rootpath: The rootdir, which paths inside it are written relative to

    Returns:
        The first 8 hex characters of the SHA-256 of the canonical encoding, and the
        qualified names of the types whose objects are in it by their type alone
        (sorted); ``("unavailable", ())`` when the object cannot be encoded
    """
    encoder = _Encoder(rootpath)
    try:
        text = _dumps(encoder.encode(value))
    except Exception:
        # A repr or a model_dump() that raised, a RecursionError
        return UNAVAILABLE, ()
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:_LENGTH], tuple(
        sorted(encoder.partial)
    )


def _dumps(encoded: Any) -> str:
    """Return the canonical JSON text of an encoded object (ASCII, no spaces)."""
    return json.dumps(encoded, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class _Encoder:
    """
    Encode an object into JSON values, for one fingerprint. Every value that JSON
    does not hold as it is becomes an object with one key, its tag, and a mapping
    becomes a list of pairs, so two kinds of values never encode alike.
    """

    def __init__(self, rootpath: str | os.PathLike[str] | None) -> None:
        self.partial: set[str] = set()
        self._roots = (
            (Path(rootpath), Path(os.path.realpath(rootpath))) if rootpath is not None else ()
        )
        # The containers being encoded, by id, with their depth: one met again is
        # inside itself
        self._open: dict[int, int] = {}

    def encode(self, value: Any) -> Any:
        """Return the JSON value of ``value``."""
        if value is None:
            return None
        # Before int and str: IntEnum and StrEnum members are ints and strs
        if isinstance(value, enum.Enum):
            name = value.name
            # A Flag value that is not a member, such as Perm(0), has no name
            shown = name if isinstance(name, str) else self.encode(value.value)
            return {"enum": [type(value).__qualname__, shown]}
        if isinstance(value, bool):
            return bool(value)
        # By their values: an int or str subclass's own __int__ or __str__ is not called
        if isinstance(value, int):
            return int.__int__(value)
        if isinstance(value, str):
            return str.__str__(value)
        if isinstance(value, float):
            return {"float": float.__repr__(value)}
        if isinstance(value, (bytes, bytearray)):
            return {"bytes": bytes(value).hex()}
        if isinstance(value, PurePath):
            return {"path": self._path(value)}
        for kind, tag, text in _TAGGED:
            if isinstance(value, kind):
                return {tag: text(value)}
        if isinstance(value, type):
            return {"type": value.__qualname__}
        key = id(value)
        depth = self._open.get(key)
        if depth is not None:
            # How many containers up the cycle goes back
            return {"cycle": len(self._open) - depth}
        self._open[key] = len(self._open)
        try:
            return self._container(value)
        finally:
            del self._open[key]

    def _container(self, value: Any) -> Any:
        """Return the JSON value of a model, a record, a mapping, a list or a set."""
        kind = type(value)
        name = kind.__qualname__
        # Looked up on the type, so an object whose __getattr__ answers any name (a
        # mock) is not taken for a model
        if isinstance(getattr(kind, "model_fields", None), dict) and callable(
            getattr(kind, "model_dump", None)
        ):
            return {"model": [name, self.encode(value.model_dump(mode="python"))]}
        if dataclasses.is_dataclass(value):
            return {
                "dataclass": [name, [self._field(value, f.name) for f in dataclasses.fields(value)]]
            }
        if isinstance(value, tuple) and isinstance(getattr(kind, "_fields", None), tuple):
            pairs = zip(kind._fields, value, strict=False)
            return {"namedtuple": [name, [[field, self.encode(item)] for field, item in pairs]]}
        if isinstance(value, Mapping):
            return {"map": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        if isinstance(value, Set):
            return {"set": sorted((self.encode(item) for item in value), key=_dumps)}
        if getattr(kind, "__repr__", None) is object.__repr__:
            # The default repr shows only the type and the address
            self.partial.add(name)
            return {"object": name}
        return {"repr": [name, _ADDRESS.sub("", repr(value))]}

    def _field(self, value: Any, name: str) -> list[Any]:
        """Return a dataclass field as [name, value], or [name] when it is not set."""
        try:
            item = getattr(value, name)
        except AttributeError:
            # An init=False field without a default, never assigned
            return [name]
        return [name, self.encode(item)]

    def _path(self, value: PurePath) -> str:
        """
        Return a path in posix form: relative to the rootdir when it is inside it, as
        it is spelled or as its real path (through a link), else as it is.
        """
        if self._roots and isinstance(value, Path) and value.is_absolute():
            spelled, real = self._roots
            for path, root in ((value, spelled), (Path(os.path.realpath(value)), real)):
                try:
                    return path.relative_to(root).as_posix()
                except ValueError:
                    pass
        return value.as_posix()
