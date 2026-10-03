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
- Dataclasses, attrs classes (recognized by their ``__attrs_attrs__``, without
  importing attrs) and NamedTuples are written field by field,
  ``types.SimpleNamespace`` and ``argparse.Namespace`` objects as their attributes
  in their order, mappings as their (key, value) pairs in their order, lists and
  tuples as arrays, and sets with their elements sorted by their encoding. A
  container met again inside itself is written as a reference to it, so a cycle
  ends.
- A class is written as its qualified name. Anything else is written as its repr,
  without the memory addresses in it (`` at 0x7f...``, and a mock's
  ``id='140...'``). An object whose type keeps the default repr (``<Plain object
  at 0x...>``) is written as its type's name alone: its state is not in the
  fingerprint, which says so by listing the type as *partial*. So is an object
  whose repr may show a set in hash order, which ``PYTHONHASHSEED`` changes: one
  with a set of strings, Enum members or objects among its attributes, or in the
  containers and objects they hold (:func:`_shows_hash_order`).

Type names are qualified names, never module names, which depend on
``--import-mode``. An object that cannot be encoded (its repr raises, or it nests
deeper than the recursion limit) gives the fingerprint ``unavailable``: the run
never fails for it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import decimal
import enum
import hashlib
import json
import os
import re
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Set
from pathlib import Path, PurePath
from types import (
    BuiltinFunctionType,
    FunctionType,
    MemberDescriptorType,
    ModuleType,
    SimpleNamespace,
)
from typing import Any

# The fingerprint of a context that could not be encoded
UNAVAILABLE = "unavailable"

# The hex characters of the SHA-256 that make the fingerprint
_LENGTH = 8

# A memory address in a repr: "<Plain object at 0x7f3a2b1c>", "<function f at 0x...>",
# and a mock's, which ends its repr: "<MagicMock name='dut' id='140139121477264'>"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+| id='[0-9]+'(?=>)")

# How many objects _shows_hash_order() looks at, at most, for a set an object's
# repr may show in hash order
_ATTRIBUTE_LIMIT = 10_000

# The objects _shows_hash_order() does not look into: values that hold no set, and
# those whose attributes are not what their repr shows
_LEAVES = (
    str,
    bytes,
    bytearray,
    int,
    float,
    complex,
    type(None),
    PurePath,
    type,
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
)

# The types whose hash does not depend on PYTHONHASHSEED or on memory addresses,
# so a set of them iterates in the same order in every process
_HASHED_ALIKE = (bool, int, float, complex, type(None))

# The types JSON holds as they are, written as they are when a value is of one of
# them exactly (a subclass is written by its value, below)
_AS_IS = frozenset({str, int, bool, type(None)})

# What _Encoder._scalar() returns for a value it does not encode
_CONTAINER = object()

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
        text = encoder.text(value)
    except Exception:
        # A repr or a model_dump() that raised, a RecursionError
        return UNAVAILABLE, ()
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:_LENGTH], tuple(
        sorted(encoder.partial)
    )


def canonical(rootpath: str | os.PathLike[str] | None = None) -> Callable[[Any], str]:
    """
    Return a function that gives the canonical JSON text of a value, the text the
    fingerprint hashes (see the module docstring), for the rootdir ``rootpath``. It
    raises what encoding the value raised (a repr, a ``RecursionError``).
    """
    return _Encoder(rootpath).text


# The canonical JSON text of an encoded object: ASCII, no spaces
_JSON = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _dumps(encoded: Any) -> str:
    """Return the canonical JSON text of an encoded object (ASCII, no spaces)."""
    return _JSON.encode(encoded)


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

    def text(self, value: Any) -> str:
        """Return the canonical JSON text of ``value``."""
        return _dumps(self.encode(value))

    def encode(self, value: Any) -> Any:
        """Return the JSON value of ``value``."""
        kind = type(value)
        # The most common values first, by their exact type
        if kind in _AS_IS:
            return value
        if kind is float:
            return {"float": float.__repr__(value)}
        if kind is not list and kind is not tuple:
            scalar = self._scalar(value)
            if scalar is not _CONTAINER:
                return scalar
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

    def _scalar(self, value: Any) -> Any:
        """
        Return the JSON value of an Enum member, a number, a string, bytes, a path, a
        type or a tagged type, or ``_CONTAINER`` for any other value.
        """
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
        return _CONTAINER

    def _container(self, value: Any) -> Any:
        """Return the JSON value of a model, a record, a mapping, a list or a set."""
        kind = type(value)
        if kind is list or kind is tuple:
            return [self.encode(item) for item in value]
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
        # An attrs class, without importing attrs: its fields, on the type
        fields = getattr(kind, "__attrs_attrs__", None)
        if isinstance(fields, tuple) and all(
            isinstance(getattr(f, "name", None), str) for f in fields
        ):
            return {"attrs": [name, [self._field(value, f.name) for f in fields]]}
        if isinstance(value, tuple) and isinstance(getattr(kind, "_fields", None), tuple):
            pairs = zip(kind._fields, value, strict=False)
            return {"namedtuple": [name, [[field, self.encode(item)] for field, item in pairs]]}
        if isinstance(value, (SimpleNamespace, argparse.Namespace)):
            attributes = vars(value).items()
            return {"namespace": [name, [[field, self.encode(item)] for field, item in attributes]]}
        if isinstance(value, Mapping):
            return {"map": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        if isinstance(value, Set):
            return {"set": sorted((self.encode(item) for item in value), key=_dumps)}
        if getattr(kind, "__repr__", None) is object.__repr__ or _shows_hash_order(value):
            # The default repr shows only the type and the address; another may show
            # a set in the order of this process's PYTHONHASHSEED
            self.partial.add(name)
            return {"object": name}
        return {"repr": [name, _ADDRESS.sub("", repr(value))]}

    def _field(self, value: Any, name: str) -> list[Any]:
        """
        Return a dataclass or attrs field as [name, value], or [name] when it is not
        set.
        """
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


def _shows_hash_order(value: Any) -> bool:
    """
    Whether the repr of ``value`` may show a set in the order of this process's
    ``PYTHONHASHSEED``: a set of two or more elements, one of them hashed by its
    text or its address (a str, bytes, an Enum member, an object), among its
    attributes or in the containers and objects they hold. It looks at
    ``_ATTRIBUTE_LIMIT`` objects at most, and answers no when it found none among
    them.
    """
    seen: set[int] = set()
    members: dict[type, list[MemberDescriptorType]] = {}
    pending = [value]
    while pending and len(seen) < _ATTRIBUTE_LIMIT:
        item = pending.pop()
        if isinstance(item, _LEAVES) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (set, frozenset)):
            if len(item) > 1 and not all(_hashed_alike(element) for element in item):
                return True
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, deque)):
            pending.extend(item)
        else:
            pending.extend(_attributes(item, members))
    return False


def _hashed_alike(value: Any) -> bool:
    """Whether the hash of ``value`` is the same in every process."""
    if type(value) in (tuple, frozenset):
        return all(_hashed_alike(item) for item in value)
    return type(value) in _HASHED_ALIKE


def _attributes(value: Any, members: dict[type, list[MemberDescriptorType]]) -> list[Any]:
    """
    Return the values of an object's attributes: those in its ``__dict__``, and its
    slots that are set. They are read without calling a ``__getattr__`` (a mock's
    makes up any name).

    Args:
        value: The object
        members: The slots of each type met so far, filled in as types are met
    """
    found: list[Any] = []
    # No __dict__, or one that is not a mapping
    with contextlib.suppress(Exception):
        found.extend(object.__getattribute__(value, "__dict__").values())
    kind = type(value)
    slots = members.get(kind)
    if slots is None:
        slots = members[kind] = [
            member
            for klass in kind.__mro__
            for member in vars(klass).values()
            if isinstance(member, MemberDescriptorType)
        ]
    for member in slots:
        # A slot never set
        with contextlib.suppress(AttributeError):
            found.append(member.__get__(value, kind))
    return found
