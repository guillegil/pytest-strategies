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
  without the memory addresses in it: `` at 0x7f...`` inside a ``<...>`` repr
  (``<function f at 0x7f...>``, ``<weakref at 0x...; to 'A' at 0x...>``), and a
  mock's ``id='140...'``. An address that a repr of its own shows outside
  ``<...>`` (``Periph('uart0' at 0x40001000)``) is kept. The sets the repr shows
  as Python does (``{'b', 'a'}``) are written with their items sorted, so a repr
  that shows a set in hash order, which ``PYTHONHASHSEED`` changes, gives one
  text in every process (:func:`_sorted_sets`). Both read the repr's quoted
  strings and brackets once, so a ``'->'`` or an ``O'Brien`` in it changes
  nothing, and the time grows with its length (:func:`_scan`). An object whose
  type keeps the default repr (``<Plain object at 0x...>``) is written as its
  type's name alone: its state is not in the fingerprint, which says so by
  listing the type as *partial*.

Type names are qualified names, never module names, which depend on
``--import-mode``. An object that cannot be encoded (its repr raises, or it nests
deeper than the recursion limit) gives the fingerprint ``unavailable``: the run
never fails for it.
"""

from __future__ import annotations

import argparse
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
from types import SimpleNamespace
from typing import Any

# The fingerprint of a context that could not be encoded
UNAVAILABLE = "unavailable"

# The hex characters of the SHA-256 that make the fingerprint
_LENGTH = 8

# A memory address in a <...> repr: "<Plain object at 0x7f3a2b1c>", "<function f at
# 0x...>" (see _without_addresses)
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

# A mock's id, which ends its <...> repr: "<MagicMock name='dut' id='140139121477264'>"
_MOCK_ID = re.compile(r" id='[0-9]+'\Z")

# The brackets _scan() pairs up in a repr: each opening one and its closing one. "<"
# opens a <...> repr (<Color.RED: 1>) only when a ">" closes it.
_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}

# What _scan() looks at in a repr: quotes, brackets, commas and colons
_SPECIAL = re.compile(r"""['"()\[\]{}<>,:]""")

# A quoted string in a repr, from its opening quote
_QUOTED = {
    "'": re.compile(r"'(?:[^'\\]|\\.)*'", re.DOTALL),
    '"': re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL),
}

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
        if getattr(kind, "__repr__", None) is object.__repr__:
            # The default repr shows only the type and the address
            self.partial.add(name)
            return {"object": name}
        return {"repr": [name, _shown(repr(value))]}

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


def _shown(text: str) -> str:
    """
    Return a repr as the fingerprint writes it: without its memory addresses
    (:func:`_without_addresses`), and with its sets sorted (:func:`_sorted_sets`).
    """
    if "{" not in text and " at 0x" not in text and " id='" not in text:
        return text
    tokens, pairs, unpaired = _scan(text)
    return _render(text, tokens, pairs, sort=not unpaired, strip=True)


def _without_addresses(text: str) -> str:
    """
    Return a repr without the memory addresses in it: `` at 0x...`` inside a
    ``<...>`` repr (``<function f at 0x7f...>``, ``<code object f at 0x..., file
    ...>``, ``<weakref at 0x...; to 'A' at 0x...>``), and a mock's ``id='...'``,
    which ends its repr. A ``<...>`` repr is one whose ``<`` a ``>`` closes
    (:func:`_scan`), so an address after a comparison's ``<`` or in a quoted
    string is not in one. An address a repr of its own shows outside ``<...>``,
    such as a register's (``Periph('uart0' at 0x40001000)``), is kept.
    """
    if " at 0x" not in text and " id='" not in text:
        return text
    tokens, pairs, _ = _scan(text)
    return _render(text, tokens, pairs, sort=False, strip=True)


def _sorted_sets(text: str) -> str:
    """
    Return a repr with the items of each set it shows as Python does sorted:
    ``{...}`` with two or more items and no ``:`` between them (a dict's), inside
    ``frozenset(...)`` too, nested sets first. So ``Bench({'beta', 'alpha'})``
    becomes ``Bench({'alpha', 'beta'})``, whatever order this process's
    ``PYTHONHASHSEED`` gave the set. Quoted strings are read as they are, so a
    brace or a comma inside one is not a set's (:func:`_scan`). A repr whose
    brackets do not pair up is returned as it is, and so is a set shown in
    another form (``",".join(tags)``).
    """
    if "{" not in text:
        return text
    tokens, pairs, unpaired = _scan(text)
    if unpaired:
        return text
    return _render(text, tokens, pairs, sort=True, strip=False)


def _opens_string(text: str, index: int) -> bool:
    """
    Whether the quote at ``index`` in a repr can open a string. A string a repr
    shows is not part of a word, so a quote right after a letter, a digit or an
    underscore opens none (the apostrophe in ``O'Brien`` or ``it's``), unless that
    letter is the ``b`` of ``b'...'`` and follows none of them itself.
    """
    before = text[index - 1] if index > 0 else " "
    if before in "bB":
        before = text[index - 2] if index > 1 else " "
    return not (before.isalnum() or before == "_")


def _scan(text: str) -> tuple[list[tuple[int, int]], dict[int, int], bool]:
    """
    Read a repr once, from left to right: its quoted strings and its brackets.

    A quote opens a string where a repr can show one (:func:`_opens_string`) and a
    matching quote closes it; quotes, brackets, commas and colons inside it are
    the string's. Brackets pair up as Python's reprs nest them: a closing bracket
    closes the innermost one open. A ``<`` that a ``>`` closes opens a ``<...>``
    repr; one that nothing closes before a closing bracket of an enclosing group,
    or before the end, is a comparison's, and so is a ``>`` that closes no ``<``
    (an arrow's). A closing bracket that closes nothing, or an opening one other
    than ``<`` that nothing closes, leaves the brackets unpaired. Each bracket is
    looked at once, so the time grows with the length of the repr, whatever its
    comparisons.

    Returns:
        The special characters outside strings (:data:`_SPECIAL`), and each
        string, as (start, end) in order; the index of the closing bracket of
        each opening one that pairs up, by the opening one's index; and whether
        some brackets do not pair up
    """
    tokens: list[tuple[int, int]] = []
    pairs: dict[int, int] = {}
    # The opening brackets not closed yet, innermost last
    open_: list[int] = []
    unpaired = False
    position = 0
    while True:
        match = _SPECIAL.search(text, position)
        if match is None:
            break
        index = match.start()
        char = text[index]
        position = index + 1
        if char in _QUOTED:
            quoted = _QUOTED[char].match(text, index) if _opens_string(text, index) else None
            if quoted is None:
                # A quote that opens no string is text
                continue
            position = quoted.end()
        elif char in _BRACKETS:
            open_.append(index)
        elif char == ">":
            if open_ and text[open_[-1]] == "<":
                pairs[open_.pop()] = index
        elif char in ")]}":
            # A "<" that no ">" closed inside the group is a comparison's
            while open_ and text[open_[-1]] == "<":
                open_.pop()
            if open_ and _BRACKETS[text[open_[-1]]] == char:
                pairs[open_.pop()] = index
            else:
                unpaired = True
        tokens.append((index, position))
    return tokens, pairs, unpaired or any(text[index] != "<" for index in open_)


class _Group:
    """A bracket group being written by :func:`_render`."""

    __slots__ = ("opening", "end", "items", "part", "keyed")

    def __init__(self, opening: str, end: int) -> None:
        # Its opening bracket, and the index of its closing one
        self.opening = opening
        self.end = end
        # A {...}'s items written so far, and the item being written
        self.items: list[str] = []
        self.part: list[str] = []
        # A colon between the items: a dict
        self.keyed = False


def _render(
    text: str, tokens: list[tuple[int, int]], pairs: dict[int, int], *, sort: bool, strip: bool
) -> str:
    """
    Write a repr read by :func:`_scan` again, with the sets it shows sorted
    (``sort``, :func:`_sorted_sets`) and without the memory addresses in its
    ``<...>`` reprs (``strip``, :func:`_without_addresses`). Brackets that do not
    pair up are written as they are.
    """
    groups = [_Group("", len(text))]
    # How many <...> reprs are open
    angled = 0
    position = 0
    for start, end in tokens:
        group = groups[-1]
        plain = text[position:start]
        if angled and strip:
            plain = _ADDRESS.sub("", plain)
        group.part.append(plain)
        position = end
        char = text[start]
        if start in pairs:
            groups.append(_Group(char, pairs[start]))
            angled += char == "<"
        elif start == group.end:
            groups.pop()
            group.items.append("".join(group.part))
            if sort and group.opening == "{" and not group.keyed and len(group.items) > 1:
                inner = ", ".join(sorted(item.strip() for item in group.items))
            else:
                inner = ",".join(group.items)
            if group.opening == "<":
                angled -= 1
                if strip:
                    inner = _MOCK_ID.sub("", inner)
            groups[-1].part.append(group.opening + inner + char)
        elif char == "," and group.opening == "{":
            group.items.append("".join(group.part))
            group.part = []
        else:
            group.keyed = group.keyed or char == ":"
            group.part.append(text[start:end])
    groups[-1].part.append(text[position:])
    return "".join(groups[-1].part)
