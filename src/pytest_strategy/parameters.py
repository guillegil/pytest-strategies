# parameter.py

from __future__ import annotations

import _random
import contextlib
import dataclasses
import difflib
import functools
import inspect
import itertools
import keyword
import math
import numbers
import operator
import os
import random
import re
import reprlib
import warnings
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast

import pytest

from ._ids import ID_FORMATS
from ._streams import StreamKey, encode, seed_part
from ._vector import RowKind as _RowKind
from ._vector import Vector, VectorInfo
from ._vector import vector_type as _vector_type
from ._warnings import PytestStrategiesWarning
from .rng import RNG, RNGValueError, SequenceLike, Series
from .test_args import TestArg

# pytest.param() returns a ParameterSet (a NamedTuple), which pytest does not export
_ParameterSet = type(pytest.param())

# What Parameter(ids=...) takes: None for the strategies_ids ini option, one of its
# formats, or a function that returns a row's test ID (None for the default)
_Ids = Literal["names", "values"] | Callable[[VectorInfo], str | None] | None


def _check_ids(ids: object) -> None:
    """
    Check a value of ``Parameter(ids=...)``.

    Raises:
        ValueError: For anything but None, "names", "values" or a callable
    """
    if ids is None or callable(ids) or (isinstance(ids, str) and ids in ID_FORMATS):
        return
    formats = ", ".join(f'"{f}"' for f in ID_FORMATS)
    raise ValueError(f"ids must be None, {formats} or a callable, got {ids!r}")


def _normalize_vector(kind: str, name: object, raw: object, arg_names: tuple[str, ...]) -> Vector:
    """
    Turn a directed or test vector, as given, into the row it stands for.

    The rules, in order:

    - a ``pytest.param(...)`` keeps its marks, and its values become the Vector:
      when its only value is a Mapping, that Mapping is a named vector, otherwise
      its values are taken by position. An ``id=`` fails: the vector's name is its
      test ID;
    - a Mapping is a named vector: its keys are the argument names, in any order;
    - a namedtuple (a Vector too) is placed by its field names, which must be the
      argument names;
    - a dataclass or pydantic model instance fails: records as vectors are not
      supported yet (a pydantic model is iterable, so this comes before the
      iterable rule);
    - a str, bytes, bytearray or a value that is not iterable fails, with a hint
      for one-argument strategies;
    - any other iterable (tuple, list, range) is taken by position.

    Args:
        kind: "directed" or "test", for the messages
        name: The vector's name, which must be a non-empty str
        raw: The vector as given
        arg_names: The strategy's argument names, in declaration order

    Returns:
        A Vector with the values in declaration order, or for a pytest.param(...)
        vector the ParameterSet with that Vector as its values, typed as a Vector

    Raises:
        RNGValueError: If the name is not a non-empty str, or the vector does not
            give exactly one value per argument
    """
    label = f"{kind.capitalize()} vector"
    if not isinstance(name, str) or not name:
        raise RNGValueError(f"{label} names must be non-empty strings, got {name!r}")
    where = f"{label} {name!r}"
    row_type = _vector_type(arg_names)

    if isinstance(raw, _ParameterSet):
        # A ParameterSet is itself a namedtuple of (values, marks, id), so it comes first
        if raw.id is not None:
            raise RNGValueError(
                f"{where} is a pytest.param with id={raw.id!r}, but the vector's name is "
                f"its ID ({kind}-{name}). Remove id=, and name the vector after the ID it "
                "should have, or build IDs with Parameter(ids=...)"
            )
        given = raw.values
        if len(given) == 1 and isinstance(given[0], Mapping):
            row = _vector_by_name(where, given[0], "dict", arg_names, row_type, in_param=True)
        else:
            row = _vector_by_position(where, given, row_type)
        # Typed as a Vector, like every row: the API types its rows as Vectors, as
        # 3.0 typed them as tuples, so typed code that unpacks a row or reads its
        # fields type-checks. The ParameterSet's .values is the Vector.
        return cast(Vector, raw._replace(values=row))

    if isinstance(raw, Mapping):
        return _vector_by_name(where, raw, "dict", arg_names, row_type)

    fields = getattr(type(raw), "_fields", None)
    if isinstance(raw, tuple) and isinstance(fields, tuple):
        if type(raw) is row_type:
            return raw
        # 3.0 placed a namedtuple by position; its fields say which value is which
        where = f"{where} ({type(raw).__name__}, a namedtuple placed by its field names)"
        return _vector_by_name(where, dict(zip(fields, raw)), "namedtuple", arg_names, row_type)

    record = _record_kind(raw)
    if record is not None:
        example = ", ".join(f"{arg!r}: ..." for arg in arg_names)
        raise RNGValueError(
            f"{where} is an instance of {type(raw).__name__}, {record}: record instances "
            f"as vectors are not supported yet; use a dict, such as {{{example}}}"
        )

    # A string is iterable, but it is one value: 3.0 split "a" into ("a",) and
    # b"\x00" into (0,)
    values: Iterable[Any] | None = None
    if not isinstance(raw, (str, bytes, bytearray)):
        with contextlib.suppress(TypeError):
            values = iter(raw)  # type: ignore[call-overload]
    if values is None:
        shown = reprlib.repr(raw)
        if len(arg_names) == 1:
            hint = f"For a one-argument strategy write ({shown},) or {{{arg_names[0]!r}: {shown}}}"
        else:
            hint = (
                f"Give one value per argument ({', '.join(arg_names)}), as a tuple or a "
                "dict of argument names to values"
            )
        raise RNGValueError(
            f"{where} is {shown} ({type(raw).__name__}), not a tuple of values. {hint}"
        )
    return _vector_by_position(where, tuple(values), row_type)


def _record_kind(value: object) -> str | None:
    """Say which kind of record instance ``value`` is, or None when it is not one."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "a dataclass"
    # pydantic v2 is recognized without importing it. model_fields is read from the
    # class: pydantic 2.11 deprecates reading it from an instance.
    if not isinstance(value, type) and isinstance(getattr(type(value), "model_fields", None), dict):
        return "a pydantic model"
    return None


def _vector_by_position(where: str, values: Sequence[Any], row_type: type[Vector]) -> Vector:
    """Build the row of a vector whose values are given in declaration order."""
    expected = len(row_type._fields)
    if len(values) != expected:
        raise RNGValueError(f"{where} has {len(values)} values, expected {expected}")
    return tuple.__new__(row_type, values)


def _vector_by_name(
    where: str,
    values: Mapping[Any, Any],
    what: str,
    arg_names: tuple[str, ...],
    row_type: type[Vector],
    *,
    in_param: bool = False,
) -> Vector:
    """
    Build the row of a vector whose values are given by argument name.

    Args:
        where: The vector, for the messages ("Directed vector 'zeros'")
        values: The values by argument name, in any order
        what: "dict" or "namedtuple", for the messages
        arg_names: The strategy's argument names, in declaration order
        row_type: The class of the row
        in_param: Whether the dict is the only value of a pytest.param(...), for
            the hint of a one-argument strategy
    """
    part = "keys" if what == "dict" else "fields"
    expected = f"The {part} of a {what} vector are the strategy's arguments: {', '.join(arg_names)}"
    if what == "dict" and len(arg_names) == 1:
        # A dict is always a named vector, also for a one-argument strategy
        shown = reprlib.repr(dict(values))
        named = f"{{{arg_names[0]!r}: {shown}}}"
        if in_param:
            # pytest.param((d,)) would be read by position and pass the tuple (d,)
            expected += (
                f". In a pytest.param, a dict value for the one argument is written "
                f"pytest.param({named}, marks=...)"
            )
        else:
            expected += f". A dict value for the one argument is written ({shown},) or {named}"
    for key in values:
        if not isinstance(key, str):
            raise RNGValueError(f"{where} has the key {key!r}, which is not a str. {expected}")
    unknown = [key for key in values if key not in arg_names]
    missing = [arg for arg in arg_names if arg not in values]
    problems = []
    if unknown:
        described = []
        for key in unknown:
            close = difflib.get_close_matches(key, missing, n=1)
            described.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
        plural = "s" if len(unknown) > 1 else ""
        problems.append(f"has unknown argument{plural} {', '.join(described)}")
    if missing:
        problems.append("is missing " + ", ".join(repr(arg) for arg in missing))
    if problems:
        raise RNGValueError(f"{where} {' and '.join(problems)}. {expected}")
    return tuple.__new__(row_type, [values[arg] for arg in arg_names])


def _vector_values(vector: object) -> Sequence[Any]:
    """Return the values of a stored directed or test vector (a pytest.param's values)."""
    if isinstance(vector, _ParameterSet):
        return vector.values
    return cast(Vector, vector)


def _check_arg_name(name: Any) -> None:
    """
    Check that an argument name can be a field of the strategy's Vector rows.

    Raises:
        RNGValueError: If the name is not an identifier, is a keyword or starts
            with "_", which a namedtuple field cannot be. Soft keywords (``type``,
            ``match``) and non-ASCII identifiers are fine.
    """
    if not isinstance(name, str) or not name.isidentifier():
        problem = "is not a Python identifier"
    elif keyword.iskeyword(name):
        problem = "is a Python keyword"
    elif name.startswith("_"):
        problem = "starts with '_'"
    else:
        return
    raise RNGValueError(
        f"Parameter argument {name!r} {problem}. Argument names are the field names of "
        "the strategy's rows (Vector), so each must be an identifier that is not a "
        "keyword and does not start with '_'"
    )


# Label text taken from a str value is at most this many characters
_LABEL_WIDTH = 40

# Characters label text cannot contain: "=" and "~" are the label's own syntax
# (ARG=TEXT~m), "[" and "]" enclose the parameters of a node ID, and whitespace
# (str.isspace(), as \s matches it) cannot be typed in a -k expression
_LABEL_EXCLUDED = re.compile(r"[\s=~\[\]]")

# What a text two different values share stands for: neither of them
_SHARED = object()


class _Key(NamedTuple):
    """The two keys of one position in an enumerated argument's sequence (_position_keys)."""

    # What the test ID shows: "ch=2", "ch=1~1" for a repeated value, "ch3" by position
    label: str
    # What the row's random streams are keyed by: "i:2", "i:1~1", "#3"
    token: str


def _label_text(text: str, width: int | None = _LABEL_WIDTH) -> str | None:
    """
    Return ``text`` when a label can show it: non-empty, at most ``width``
    characters (any length for None), printable, and without whitespace, ``=``,
    ``~``, ``[`` or ``]``. Otherwise None.
    """
    if (
        text
        and (width is None or len(text) <= width)
        and text.isprintable()
        and _LABEL_EXCLUDED.search(text) is None
    ):
        return text
    return None


def _escaped(text: str) -> str:
    """Double each ``~`` of a token's text, so that text never ends like a repeat (``~1``)."""
    return text.replace("~", "~~") if "~" in text else text


def _value_keys(value: object) -> tuple[str | None, str | None]:
    """
    Return the token of a value, before any repeat suffix, and its label text, in
    D2's type order. The token is None for a value keyed by its position, and the
    text None for a value a label cannot show.
    """
    # The common exact types first: none of them can be a type that comes earlier
    exact = type(value)
    if exact is int:
        number: object = value
    elif exact is str:
        text = cast(str, value)
        return f"s:{_escaped(text)}", _label_text(text)
    elif value is None:
        return "n", "None"
    elif exact is bool:
        return f"b:{value}", str(value)
    # Before int and str: an IntEnum or StrEnum member is keyed as a member
    elif isinstance(value, Enum):
        name = value.name
        if not isinstance(name, str):
            # A Flag value that is no member has no name
            return None, None
        return f"e:{_escaped(type(value).__qualname__)}.{_escaped(name)}", _label_text(name, None)
    elif isinstance(value, numbers.Integral):
        try:
            number = operator.index(value)
        except (TypeError, ValueError):
            return None, None
    elif isinstance(value, numbers.Real):
        try:
            real = float(value)
        except (TypeError, ValueError, OverflowError):
            return None, None
        return f"f:{real.hex()}", repr(real)
    elif isinstance(value, str):
        return f"s:{_escaped(value)}", _label_text(value)
    elif isinstance(value, bytes):
        return f"y:{value.hex()}", None
    # Anything else is keyed by its position; a class or a function shows its name
    elif isinstance(value, type) or inspect.isfunction(value) or inspect.isbuiltin(value):
        called = getattr(value, "__name__", None)
        return None, _label_text(called, None) if isinstance(called, str) else None
    else:
        return None, None
    try:
        digits = int.__repr__(cast(int, number))
    except ValueError:
        # More digits than int-to-str conversion allows
        return None, None
    return f"i:{digits}", digits


def _position_keys(arg: str, sequence: Sequence[Any]) -> tuple[_Key, ...]:
    """
    Return the label and the token of each position of an enumerated argument's
    sequence: the sequence it declares, after its predicate. For None, bools, Enum
    members, numbers and strs they depend only on the value and how often it came
    before, so a row keeps its ID and its values when a scalar is added elsewhere in
    the sequence; other values are keyed by their position.

    The type of each value is decided in this order, with isinstance: None, bool,
    Enum member (IntEnum and StrEnum members too), int or another
    numbers.Integral, float or another numbers.Real, str, bytes, anything else.

    - The token keys the row's random streams: ``n``, ``b:True``,
      ``e:<Enum qualname>.<member>``, ``i:42``, ``f:<float.hex()>``, ``s:fast``,
      ``y:<bytes hex>``, and ``#<position>`` for anything else. A ``~`` in the text
      of an ``s:`` or ``e:`` token is doubled.
    - The label is what the test ID shows: ``ARG=TEXT``, where TEXT is ``str(v)``
      for None and a bool, the member's name for an Enum member, ``repr()`` of the
      int or float for a number, the ``__name__`` of a class or function, and the
      str itself when it is non-empty, at most 40 characters, printable, and has no
      whitespace, ``=``, ``~``, ``[`` or ``]``. A value without such text (a tuple,
      bytes, another object or str) is labeled by its position: ``ARG<position>``
      (``ch3``).
    - A value already seen m times gets ``~m`` in its token and label (``i:1~1``,
      ``ch=1~1``), so a value listed twice still keys two rows; a value keyed by its
      position needs no suffix in its token. When two different values would get the
      same text (``1`` and ``"1"``), every position of those values gets the
      positional label.

    Args:
        arg: The argument's name
        sequence: The argument's sequence of values

    Returns:
        One _Key per position, in order
    """
    bases = [_value_keys(value) for value in sequence]
    # Two positions hold the same value when their tokens are equal, or, for values
    # keyed by their position, when they hold the same object
    seen: dict[object, int] = {}
    repeats = []
    # The value each text stands for, or _SHARED when two different values have it
    meanings: dict[str, object] = {}
    for (token, text), value in zip(bases, sequence):
        same: object = token if token is not None else id(value)
        repeat = seen.get(same, 0)
        seen[same] = repeat + 1
        repeats.append(repeat)
        if text is not None and meanings.setdefault(text, same) != same:
            meanings[text] = _SHARED
    keys = []
    for position, ((token, text), repeat) in enumerate(zip(bases, repeats)):
        suffix = f"~{repeat}" if repeat else ""
        if text is not None and meanings[text] is not _SHARED:
            label = f"{arg}={text}{suffix}"
        else:
            label = f"{arg}{position}"
        keys.append(_Key(label, f"{token}{suffix}" if token is not None else f"#{position}"))
    return tuple(keys)


def _auto_order(rng_type: SequenceLike[Any]) -> list[tuple[int, Any]]:
    """
    Return the values of a sequence argument in its ``--nsamples=auto`` order (a
    Series in declaration order, an RNGSequence permuted), each with its position in
    the declared sequence, which keys and labels its rows whatever the order.

    Series and RNGSequence give the positions (``_auto_positions()``). For a
    subclass with its own ``_get_auto_sequence()``, each value it returns is matched
    to the first position not taken yet that holds the same object, else an equal
    value; values that are the same object take their positions in the order they
    come.

    Raises:
        RNGValueError: If such a ``_get_auto_sequence()`` returned a value that is
            not in the sequence (a subclass that builds new values)
    """
    declared = rng_type.sequence
    if type(rng_type)._get_auto_sequence is SequenceLike._get_auto_sequence:
        return [(position, declared[position]) for position in rng_type._auto_positions()]
    by_object: dict[int, deque[int]] = {}
    for position, value in enumerate(declared):
        by_object.setdefault(id(value), deque()).append(position)
    taken: set[int] = set()
    order = []
    for value in rng_type._get_auto_sequence():
        positions = by_object.get(id(value), deque())
        while positions and positions[0] in taken:
            positions.popleft()
        found = positions.popleft() if positions else _equal_position(declared, value, taken)
        if found is None:
            raise RNGValueError(
                f"{type(rng_type).__name__}._get_auto_sequence() returned {value!r}, which "
                "is not one of its sequence's values (or more often than the sequence has it)"
            )
        taken.add(found)
        order.append((found, value))
    return order


def _equal_position(declared: Sequence[Any], value: object, taken: set[int]) -> int | None:
    """Return the first position not taken whose value equals ``value``, else None."""
    for position, candidate in enumerate(declared):
        if position in taken:
            continue
        try:
            if candidate == value:
                return position
        except Exception:
            # A comparison that raises (or gives an ambiguous truth value) is no match
            continue
    return None


def _describe_row(kind: str, index: int | None, labels: tuple[str, ...] = ()) -> str:
    """
    Name a row in a message: ``random row 3``, ``random row 1 (ch=2)`` or
    ``exhaustive row 5 (ch=2, dev=b)``; ``the row`` for a row without a number.
    """
    if index is None:
        return "the row"
    described = f"{kind} row {index}"
    return f"{described} ({', '.join(labels)})" if labels else described


@dataclasses.dataclass(slots=True)
class _Row:
    """
    One row of a generation call, with its identity (D2).

    Every generation path returns these (``Parameter._generate_rows``), so the row
    index is defined in one place: the plugin builds the row's test ID, its random
    streams and its metadata from the same row. Rows are not changed once built.

    Attributes:
        kind: "directed", "test", "random", "exhaustive", or "skipped" for the one
            row of a strategy whose skip_if_empty sequence has no values
        name: The directed or test vector's name, else None
        index: The number the messages show: the vector's position in
            directed_vectors (the number --vector-index takes) or test_vectors; j
            for a random row; the row's position in the product of the enumerated
            arguments' declared sequences, in declaration order, for an exhaustive
            row; None for "skipped"
        pos: The enumerated arguments as (argument name, token) pairs, sorted by
            argument name (see _position_keys); empty when the row enumerates none.
            A finite run enumerates the Series arguments, per_sequence_samples=True
            the Series and RNGSequence arguments, and --nsamples=auto every Series
            and RNGSequence argument.
        j: The row's number within its combination of enumerated values, from 0: k
            for the k-th plain random row; the cycle for a finite Series row (a
            combination the constraints skip leaves a gap); the row's number for
            per_sequence_samples=True; 0 for an exhaustive row. None for the other
            kinds.
        values: The row, a Vector (Nones for "skipped")
        labels: The labels of the enumerated arguments, in declaration order
        param: The pytest.param(...) a directed or test vector was given as, whose
            values are ``values``; None otherwise
    """

    kind: _RowKind
    name: str | None
    index: int | None
    pos: tuple[tuple[str, str], ...]
    j: int | None
    values: Vector
    labels: tuple[str, ...] = ()
    param: Any = None

    @property
    def sample(self) -> Vector:
        """The row as generate_vectors() returns it: ``param``, or else ``values``."""
        return cast(Vector, self.param) if self.param is not None else self.values


class _Enumeration:
    """
    The enumerated arguments of a generation path and the keys of their positions.
    A combination gives one position per argument, in declaration order.
    """

    __slots__ = ("_parts", "_by_name")

    def __init__(self, names: Sequence[str], sequences: Sequence[Sequence[Any]]) -> None:
        # A combination's index in the declaration-order product of the sequences
        strides = [math.prod(len(s) for s in sequences[i + 1 :]) for i in range(len(names))]
        # Per argument and position: its label, its (name, token) pair and its share
        # of the index
        self._parts = [
            [(key.label, (name, key.token), position * stride) for position, key in enumerate(keys)]
            for name, keys, stride in zip(
                names, [_position_keys(n, s) for n, s in zip(names, sequences)], strides
            )
        ]
        # pos lists the arguments sorted by name, so reordering them keeps a row's keys
        self._by_name = sorted(range(len(names)), key=list(names).__getitem__)

    def identify(
        self, combination: Sequence[int]
    ) -> tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]]:
        """
        Return a combination's index (its position in the declaration-order
        product), its labels in declaration order and its pos (the (name, token)
        pairs sorted by name).
        """
        if len(self._parts) == 1:
            label, pair, index = self._parts[0][combination[0]]
            return index, (label,), (pair,)
        parts = [arg_parts[p] for arg_parts, p in zip(self._parts, combination)]
        return (
            sum([part[2] for part in parts]),
            tuple([part[0] for part in parts]),
            tuple([parts[a][1] for a in self._by_name]),
        )


def _direct_key() -> StreamKey:
    """
    Return the key of a generation call made outside the plugin (generate_vectors()
    and the other generators called directly): ``RNG.get_seed()`` and 128 bits of the
    current generator, so consecutive calls differ and ``RNG.seed(s)`` repeats them.
    """
    return StreamKey.root(seed_part(RNG.get_seed()), "direct", RNG._generator.getrandbits(128))


class _ArgRandom(random.Random):
    """
    The generator of one argument's row streams (``_RowStreams``).

    It is reseeded with a 128-bit int at the start of every row, by ``reseed()``,
    which seeds it as ``random.Random.seed()`` seeds it with an int (the seeding in
    C, then no cached ``gauss()`` value) without that method's checks of the seed's
    type: about 0.6 us of the 8 us a seed costs.
    """

    def reseed(self, seed: int) -> None:
        """Seed the generator as ``self.seed(seed)`` would, for an int seed."""
        _random.Random.seed(self, seed)
        self.gauss_next = None


class _RowStreams:
    """
    The random streams of a generation call's rows (D5): one ``random.Random`` per
    argument that draws, reseeded at the start of each row from the key
    ``T/"row"/pos/j/argname``, where T is the call's key.

    A drawn argument's values therefore depend only on T, the row's identity (pos
    and j) and the argument's name: not on the other rows, on the other arguments,
    or on what was drawn before. The generators continue across the attempts of
    one row, so a constraint that rejects a row changes only that row.
    """

    __slots__ = ("_rows", "_names", "_generators", "_pos")

    def __init__(self, key: StreamKey, args: Sequence[TestArg]) -> None:
        self._rows = key.child("row")
        # Each argument's part of its streams' keys, encoded once
        self._names = [encode(arg.name) for arg in args]
        # A static value= argument draws nothing, so it gets no generator
        self._generators: list[_ArgRandom | None] = [
            None if arg.is_static else _ArgRandom(0) for arg in args
        ]
        # The encoded parts of each pos seen, which the rows of a combination share
        self._pos: dict[tuple[tuple[str, str], ...], bytes] = {}

    def start(
        self,
        pos: tuple[tuple[str, str], ...],
        j: int,
        drawn: Sequence[int],
        ambient: random.Random,
    ) -> list[tuple[int, random.Random]]:
        """
        Reseed the generators of the ``drawn`` positions for the row (pos, j), and
        return each position with the generator its argument draws from: its own,
        or ``ambient`` for a static argument, which draws nothing.
        """
        # pos is flattened into the key: its parts are strs and j is the first int,
        # so the path stays unambiguous
        encoded = self._pos.get(pos)
        if encoded is None:
            encoded = self._pos[pos] = encode(*[part for pair in pos for part in pair])
        row = encoded + encode(j)
        rows, names, generators = self._rows, self._names, self._generators
        started: list[tuple[int, random.Random]] = []
        for i in drawn:
            generator = generators[i]
            if generator is None:
                started.append((i, ambient))
            else:
                # The seed of T/"row"/pos/j/argname
                generator.reseed(rows.child_seed_int(row + names[i]))
                started.append((i, generator))
        return started


def _auto_order_from(key: StreamKey, rng_type: SequenceLike[Any]) -> list[tuple[int, Any]]:
    """
    Return ``_auto_order(rng_type)`` with ``RNG.generator()`` on the stream of
    ``key`` (``T/"order"/argname``), so an RNGSequence's permutation under
    ``--nsamples=auto`` depends on nothing but its argument.
    """
    # Under the lock, as a row's generators are (Parameter._build_row)
    with RNG._ambient._lock:
        ambient = RNG._generator
        RNG._generator = random.Random(key.seed_int())
        try:
            return _auto_order(rng_type)
        finally:
            RNG._generator = ambient


# The constraints a generation call evaluates, as (name, function) pairs in order
_Constraints = tuple[tuple[str, Callable[[Vector], object]], ...]

# Characters a constraint name cannot contain: they separate the items of
# --strategy-constraint-off ("S:a,b") and of the diagnostics ("a=3, b=1")
_NAME_SEPARATORS = (":", ",", "=")

# Example rows in the diagnostics are cut to about this many characters
_EXAMPLE_WIDTH = 120


def _check_constraint_name(name: object) -> str:
    """
    Check a constraint name given as a dict key or ``add_constraint(name=...)``.

    Raises:
        RNGValueError: If the name is not a non-empty str, or contains whitespace,
            ``:``, ``,`` or ``=``. ``.`` is allowed.
    """
    if (
        isinstance(name, str)
        and name
        and not any(c.isspace() or c in _NAME_SEPARATORS for c in name)
    ):
        return name
    raise RNGValueError(
        f"Constraint name {name!r} is not valid: a constraint name is a non-empty str "
        "without whitespace, ':', ',' or '=', which separate the items of "
        "--strategy-constraint-off and of the diagnostics"
    )


def _function_name(fn: Callable[..., object]) -> str | None:
    """
    Return the name a constraint gets from its function: ``__name__`` for a
    function, a bound method or a builtin. None for a lambda, a partial or a
    callable object, which get ``constraint_<i>``.
    """
    target = fn.__func__ if inspect.ismethod(fn) else fn
    if inspect.isfunction(target) or inspect.isbuiltin(target):
        name = getattr(fn, "__name__", None)
        # A lambda's name is "<lambda>"
        if isinstance(name, str) and name.isidentifier():
            return name
    return None


def _constraint_origin(fn: Callable[..., object]) -> str:
    """Say where a constraint comes from, for messages: ``lambda at strategies.py:42``."""
    if isinstance(fn, functools.partial):
        return f"partial of {_constraint_origin(fn.func)}"
    target = fn.__func__ if inspect.ismethod(fn) else fn
    code = getattr(target, "__code__", None)
    if inspect.isfunction(target) and code is not None:
        what = "lambda" if target.__name__ == "<lambda>" else target.__qualname__
        return f"{what} at {os.path.basename(code.co_filename)}:{code.co_firstlineno}"
    if inspect.isclass(fn) or inspect.isbuiltin(fn):
        return str(getattr(fn, "__qualname__", fn))
    return f"{type(fn).__qualname__} instance"


def _short_repr(row: object) -> str:
    """Return the repr of an example row, cut to about _EXAMPLE_WIDTH characters."""
    text = repr(row)
    return text if len(text) <= _EXAMPLE_WIDTH else text[: _EXAMPLE_WIDTH - 3] + "..."


class _Rejections:
    """
    The draws the constraints rejected in a run of attempts (one row, one Series
    visit, a whole cycle), counted by the name of the first failing constraint,
    with the first row each constraint rejected.

    ``total``, when given, also counts every rejection of the generation call, for
    the -v summary; it is not cleared with the rest.
    """

    __slots__ = ("counts", "first", "none_only", "total")

    def __init__(self, total: Counter[str] | None = None) -> None:
        self.counts: Counter[str] = Counter()
        self.first: dict[str, Vector] = {}
        # Whether every rejection by the constraint was a None result (missing return?)
        self.none_only: dict[str, bool] = {}
        self.total = total

    def add(self, name: str, row: Vector, result: object) -> None:
        """Count one draw that the constraint ``name`` rejected with ``result``."""
        self.counts[name] += 1
        if self.total is not None:
            self.total[name] += 1
        if name in self.first:
            if result is not None:
                self.none_only[name] = False
        else:
            self.first[name] = row
            self.none_only[name] = result is None

    def merge(self, other: _Rejections) -> None:
        """Add ``other``'s counts, keeping the earlier first rows (``total`` is untouched)."""
        self.counts.update(other.counts)
        for name, row in other.first.items():
            if name in self.first:
                self.none_only[name] = self.none_only[name] and other.none_only[name]
            else:
                self.first[name] = row
                self.none_only[name] = other.none_only[name]

    def copy(self) -> _Rejections:
        """Return a copy of the counts and first rows, without ``total``."""
        copied = _Rejections()
        copied.merge(self)
        return copied

    def clear(self) -> None:
        """Start a new run of attempts (``total`` keeps counting)."""
        self.counts.clear()
        self.first.clear()
        self.none_only.clear()


class _GenerationStats:
    """What the constraints did in a generation call, for the plugin's -v summary."""

    __slots__ = ("rejected", "left_out")

    def __init__(self) -> None:
        # Draws rejected, by the name of the first failing constraint
        self.rejected: Counter[str] = Counter()
        # Combinations generate_exhaustive() dropped because the constraints rejected them
        self.left_out = 0


class _ConstraintsExhausted(ValueError):
    """
    The constraints rejected every draw a row (or every combination) was allowed.

    The message ends with advice that does not know the strategy; the resolver
    replaces it with advice that names ``--strategy-constraint-off`` for the
    strictest constraint.

    Attributes:
        detail: The message without the advice
        strictest: The constraint with the most rejections (the first in
            evaluation order on a tie), or None without rejections
        retries: Whether raising max_retries could help (the row has random
            arguments to redraw)
    """

    def __init__(self, detail: str, strictest: str | None, retries: bool) -> None:
        self.detail = detail
        self.strictest = strictest
        self.retries = retries
        if retries:
            advice = "Raise Parameter(max_retries=...) or relax a constraint."
        else:
            advice = "Relax a constraint."
        super().__init__(f"{detail} {advice}")


# The attribute of a constraint's exception that holds its _ConstraintFailure
_CONSTRAINT_FAILURE = "_pytest_strategies_constraint_failure"

# The last exception a constraint raised that takes no attribute, with its
# _ConstraintFailure (one entry at most): _constraint_failure() finds the failure
# here, and the resolver drops it once reported (_forget_constraint_failure())
_unattached: list[tuple[BaseException, _ConstraintFailure]] = []


class _ConstraintFailure:
    """
    Which constraint raised, and on which row.

    The generators re-raise the constraint's own exception, so a caller that
    catches its type keeps working, with this attached to it (``_CONSTRAINT_FAILURE``,
    or ``_unattached`` for an exception that takes no attribute) and described in a
    note. The resolver builds the collection error from it, chained to that
    exception, so the user's frame is shown.

    Attributes:
        label: The constraint, as the messages show it ("'ratio'")
        row: Where it raised: "random row 3, Vector(...)", "exhaustive row 5 (ch=2),
            Vector(...)", or "the row Vector(...)" for a row without a number
        detail: The message without the note on the constraints turned off
        off_before: The constraints before it in the mapping that the generation
            call turned off (one of them may have guarded it)
        added_note: The note :meth:`attach` added to the exception, or None
    """

    __slots__ = ("label", "row", "detail", "off_before", "added_note")

    def __init__(
        self,
        label: str,
        row: str,
        error: Exception,
        off_before: Sequence[str] = (),
    ) -> None:
        self.label = label
        self.row = row
        # Not the exception itself, which holds this object
        self.detail = f"Constraint {label} raised {type(error).__name__} on {self.row}: {error}"
        self.off_before = tuple(off_before)
        self.added_note: str | None = None

    def note(self, turned_off_by: str) -> str:
        """Return the note the exception gets: ``Raised by constraint 'ratio' on ...``."""
        return self._with_off_before(
            f"Raised by constraint {self.label} on {self.row}", turned_off_by
        )

    def attach(self, error: BaseException, turned_off_by: str) -> None:
        """
        Attach this failure to the exception the constraint raised, with its note.

        The note of the failure attached before, if any, is replaced in place: the
        resolver attaches the failure again to name ``--strategy-constraint-off``
        instead of ``constraints_off``, and a constraint may raise one exception
        object again (a module-level instance), so the notes do not pile up.

        It never replaces the exception: the note and the failure are written with
        ``object.__setattr__``, past the ``__setattr__`` of a frozen dataclass or
        attrs exception. An exception that rejects even that gets no note (nor does
        one whose ``__notes__`` is not a list), and its failure goes to
        ``_unattached``.
        """
        note = self.note(turned_off_by)
        previous = _constraint_failure(error)
        replaced = previous.added_note if previous is not None else None
        notes = getattr(error, "__notes__", None)
        self.added_note = None
        with contextlib.suppress(AttributeError, TypeError):
            if isinstance(notes, list):
                if replaced is not None and replaced in notes:
                    notes[notes.index(replaced)] = note
                else:
                    notes.append(note)
                self.added_note = note
            elif notes is None:
                object.__setattr__(error, "__notes__", [note])
                self.added_note = note
            # else: __notes__ that is not a list, which add_note() rejects too
        try:
            object.__setattr__(error, _CONSTRAINT_FAILURE, self)
        except (AttributeError, TypeError):
            _unattached[:] = [(error, self)]

    def message(self, turned_off_by: str) -> str:
        """
        Return the message: ``Constraint 'ratio' raised ZeroDivisionError on random row
        3, Vector(...): division by zero``, with a note naming the constraints before
        this one that ``turned_off_by`` turned off. The resolver names
        ``--strategy-constraint-off``.
        """
        return self._with_off_before(self.detail, turned_off_by)

    def _with_off_before(self, text: str, turned_off_by: str) -> str:
        """Add the constraints turned off before this one to ``text``, if any."""
        if not self.off_before:
            return text
        names = ", ".join(repr(name) for name in self.off_before)
        if len(self.off_before) == 1:
            note = f"constraint {names} before it is turned off"
        else:
            note = f"constraints {names} before it are turned off"
        return f"{text} ({note} by {turned_off_by})"


def _constraint_failure(error: BaseException) -> _ConstraintFailure | None:
    """Return the _ConstraintFailure of an exception a constraint raised, else None."""
    failure = getattr(error, _CONSTRAINT_FAILURE, None)
    if isinstance(failure, _ConstraintFailure):
        return failure
    return next((kept for raised, kept in _unattached if raised is error), None)


def _forget_constraint_failure(error: BaseException) -> None:
    """Drop the failure ``_unattached`` keeps for ``error``, if any."""
    _unattached[:] = [entry for entry in _unattached if entry[0] is not error]


class Parameter:
    """
    Manages a collection of TestArg instances and generates parameter vectors.

    Supports:
    - Multiple test arguments
    - Directed test vectors (named edge cases)
    - Vector-level constraints
    - CLI-based filtering
    """

    def __init__(
        self,
        *test_args: TestArg,
        directed_vectors: Mapping[str, Iterable[Any]] | None = None,
        test_vectors: Mapping[str, Iterable[Any]] | None = None,
        always_include_directed: bool = True,
        vector_constraints: (
            Mapping[str, Callable[[Vector], object]] | Iterable[Callable[[Vector], object]] | None
        ) = None,
        max_retries: int = 100,
        nsamples: int | str | None = None,
        per_sequence_samples: bool = False,
        max_exhaustive: int | None = None,
        ids: _Ids = None,
    ) -> None:
        """
        Initialize a Parameter container.

        Args:
            *test_args: Variable number of TestArg instances
            directed_vectors: Mapping of vector names (non-empty strings) to vectors.
                A vector gives one value per argument: a tuple or list in declaration
                order, a dict of argument names to values in any order, or a
                namedtuple whose fields are the argument names. A
                pytest.param(*values, marks=...) of one of these keeps its marks on its
                row; it cannot have an id=, because the vector's name is its test ID.
                Each is stored as a Vector; a dict value for a one-argument strategy is
                written ({"a": 1},) or {"cfg": {"a": 1}}, and in a pytest.param only
                pytest.param({"cfg": {"a": 1}}, marks=...).
            test_vectors: Mapping of test vector names to vectors (for test mode), in
                the forms directed_vectors takes
            always_include_directed: If True, directed vectors are included in "mixed" mode
            vector_constraints: Functions that validate entire random rows, as a dict of
                names to functions or as a list. A list names each function after its
                ``__name__``, and a lambda, partial or callable object
                ``constraint_<i>``, i being its position. Each receives the row as a
                Vector (``v.addr`` or ``v[0]``); they run in order, and the first
                falsy result rejects the row. A name is a non-empty str without
                whitespace, ":", "," or "=".
            max_retries: Maximum attempts to satisfy vector_constraints before raising (>= 1).
                In finite mode with Series args, a Series combination whose random args
                exhaust max_retries is skipped with a PytestStrategiesWarning; the call
                raises only when a whole cycle of combinations yields no vector. With
                per_sequence_samples=True, such a combination gets fewer than n rows (with
                a warning), and the call raises only when no combination yields a row.
            nsamples: Default number of random samples for this strategy (None, "auto"
                for exhaustive generation, or an int >= 0)
            per_sequence_samples: If True, generate the n random samples once for every
                combination of the Series/RNGSequence args (in declaration order) instead
                of n in total. Two devices with the default n=10 give 20 rows. A value
                listed twice counts twice.
            max_exhaustive: The most rows --nsamples=auto (or per_sequence_samples=True)
                may generate for this strategy; the plugin fails the test's collection
                above it, before generating. None uses the strategies_max_exhaustive ini
                option (100,000 by default).
            ids: The test IDs of this strategy's rows. None follows the
                strategies_ids ini option; "names" (directed-zeros, rand-3, ch=2-rand-1)
                or "values" (the 3.0 IDs, built from the values) overrides it for this
                strategy. A callable is called with each row's VectorInfo while the
                tests are collected (not for the skipped row of an empty skip_if_empty
                sequence), whose id is the row's ID in the ini option's format: it
                returns the ID to use, or None to keep that one. IDs that come out
                the same are suffixed as pytest suffixes them (odd0, odd1).

        Raises:
            ValueError: If nsamples, max_retries or max_exhaustive is not a valid count,
                per_sequence_samples is not a bool, or ids is not None, "names",
                "values" or a callable
            RNGValueError: If two test args have the same name, or a name is not an
                identifier, is a keyword or starts with "_"
            RNGValueError: If a directed or test vector's name is not a non-empty str,
                or the vector does not give exactly one value per argument (a str,
                bytes or scalar vector, a dict with missing or unknown names, a
                dataclass or pydantic model instance)
            RNGValueError: If a constraint name is not valid, two constraints have
                one name (pass a dict to name them), or one function is given twice
            TypeError: If vector_constraints is a single callable or a str instead
                of a dict or list, or holds something that is not callable

        Examples:
            # Simple parameter with 2 args
            param = Parameter(
                TestArg("x", rng_type=RNGInteger(0, 10)),
                TestArg("y", rng_type=RNGInteger(0, 10))
            )

            # With directed vectors, by position or by name
            param = Parameter(
                TestArg("x", rng_type=RNGInteger(0, 10)),
                TestArg("y", rng_type=RNGInteger(0, 10)),
                directed_vectors={
                    "origin": (0, 0),
                    "max": {"y": 10, "x": 10},
                }
            )
        """
        # Validate counts up front (bool is an int subclass, so reject it explicitly).
        # "auto" is the value factories receive for --nsamples=auto and pass through.
        if (
            nsamples is not None
            and nsamples != "auto"
            and (not isinstance(nsamples, int) or isinstance(nsamples, bool) or nsamples < 0)
        ):
            raise ValueError(f'nsamples must be None or an int >= 0 (or "auto"), got {nsamples!r}')
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError(f"max_retries must be an int >= 1, got {max_retries!r}")
        if not isinstance(per_sequence_samples, bool):
            raise ValueError(f"per_sequence_samples must be a bool, got {per_sequence_samples!r}")
        if max_exhaustive is not None and (
            not isinstance(max_exhaustive, int)
            or isinstance(max_exhaustive, bool)
            or max_exhaustive < 1
        ):
            raise ValueError(f"max_exhaustive must be None or an int >= 1, got {max_exhaustive!r}")
        _check_ids(ids)
        seen: set[str] = set()
        for arg in test_args:
            _check_arg_name(arg.name)
            if arg.name in seen:
                raise RNGValueError(f"Parameter has two test args named {arg.name!r}")
            seen.add(arg.name)

        # Copy the caller's containers so add_*/remove_* never mutate shared objects
        self.test_args = list(test_args)
        arg_names = self.arg_names
        self._directed_vectors = {
            k: _normalize_vector("directed", k, v, arg_names)
            for k, v in (directed_vectors or {}).items()
        }
        self._test_vectors = {
            k: _normalize_vector("test", k, v, arg_names) for k, v in (test_vectors or {}).items()
        }
        self.always_include_directed = always_include_directed
        # The constraints by name, in evaluation order, and the names the Parameter
        # made up (constraint_<i>), which the messages show with the constraint's origin
        self._constraints: dict[str, Callable[[Vector], object]] = {}
        self._unnamed: set[str] = set()
        self._set_constraints(vector_constraints)
        self.max_retries = max_retries
        self.nsamples = nsamples
        self.per_sequence_samples = per_sequence_samples
        self.max_exhaustive = max_exhaustive
        self.ids: _Ids = ids

    @property
    def directed_vectors(self) -> Mapping[str, Vector]:
        """
        The directed vectors by name, as Vectors (a pytest.param(...) vector keeps its
        marks, with a Vector as its values; it is typed as a Vector, as every row is).

        Read-only: add_directed_vector() and remove_directed_vector() change it.
        """
        return MappingProxyType(self._directed_vectors)

    @property
    def test_vectors(self) -> Mapping[str, Vector]:
        """
        The test vectors by name, as for directed_vectors.

        Read-only: add_test_vector() and remove_test_vector() change it.
        """
        return MappingProxyType(self._test_vectors)

    @property
    def vector_constraints(self) -> Mapping[str, Callable[[Vector], object]]:
        """
        The constraints by name, in the order they are evaluated.

        Iterating gives the names. Read-only: add_constraint(), remove_constraint()
        and clear_constraints() change it.
        """
        return MappingProxyType(self._constraints)

    def _set_constraints(
        self,
        constraints: (
            Mapping[str, Callable[[Vector], object]] | Iterable[Callable[[Vector], object]] | None
        ),
    ) -> None:
        """Add the constraints given to the constructor, as a dict or as a list."""
        if constraints is None:
            return
        if isinstance(constraints, Mapping):
            for key, fn in constraints.items():
                self._insert(_check_constraint_name(key), fn, f"vector_constraints[{key!r}]")
            return
        expected = "vector_constraints must be a dict of names to constraints or a list of them"
        if isinstance(constraints, (str, bytes)):
            raise TypeError(f"{expected}, not a {type(constraints).__name__} ({constraints!r})")
        if callable(constraints):
            raise TypeError(
                f"{expected}, not a single callable ({_constraint_origin(constraints)}): "
                "put it in a list"
            )
        try:
            given = list(constraints)
        except TypeError:
            raise TypeError(f"{expected}, not {type(constraints).__name__}") from None
        # The function names first, so a constraint_<i> never takes the name of a
        # function later in the list
        names = [_function_name(fn) if callable(fn) else None for fn in given]
        taken = {name for name in names if name is not None}
        for i, fn in enumerate(given):
            name = names[i]
            if name is None:
                name = self._free_name(i, taken)
                taken.add(name)
            self._insert(name, fn, f"vector_constraints[{i}]", unnamed=names[i] is None)

    def _free_name(self, position: int, taken: Iterable[str] = ()) -> str:
        """Return constraint_<i> for an unnamed constraint, i its position or the next free."""
        used = set(taken) | set(self._constraints)
        while f"constraint_{position}" in used:
            position += 1
        return f"constraint_{position}"

    def _insert(self, name: str, fn: object, where: str, *, unnamed: bool = False) -> None:
        """
        Add one constraint at the end of the evaluation order.

        Args:
            name: Its name, already checked
            fn: The constraint
            where: Where it was given, for the messages ("vector_constraints[0]")
            unnamed: Whether the Parameter made the name up (constraint_<i>)

        Raises:
            TypeError: If fn is not callable
            RNGValueError: If the function is already a constraint, or the name is taken
        """
        if not callable(fn):
            raise TypeError(f"{where} is {fn!r} ({type(fn).__name__}), not a callable")
        for existing_name, existing in self._constraints.items():
            # A bound method is a new object on each access, but compares equal
            if existing is fn or (inspect.ismethod(fn) and existing == fn):
                raise RNGValueError(
                    f"The constraint {_constraint_origin(fn)} is given twice (it is "
                    f"already the constraint {existing_name!r}); give each constraint once"
                )
        if name in self._constraints:
            first = _constraint_origin(self._constraints[name])
            raise RNGValueError(
                f"Two constraints are named {name!r}: {first} and {_constraint_origin(fn)}. "
                "A constraint is named after its function, so functions with one name, "
                "such as closures made by one helper, need names of their own: pass a "
                f"dict, such as vector_constraints={{'{name}_a': ..., '{name}_b': ...}}, "
                "or add_constraint(fn, name=...)"
            )
        self._constraints[name] = fn
        if unnamed:
            self._unnamed.add(name)

    def _unnamed_origin(self, name: str) -> str | None:
        """Return where a constraint_<i> comes from ("lambda at strategies.py:42"), else None."""
        if name in self._unnamed and name in self._constraints:
            return _constraint_origin(self._constraints[name])
        return None

    def _evaluated(self, constraints_off: Iterable[str] = ()) -> _Constraints:
        """
        Return the constraints a generation call evaluates, in order: all of them
        but the ones named in ``constraints_off``. The Parameter is not changed.

        A name the Parameter does not have turns nothing off, as a bare
        ``--strategy-constraint-off`` name in a strategy without that constraint, so
        a factory can pass ``options.constraints_off`` on as it is.

        Raises:
            TypeError: If constraints_off is a str instead of a collection of names
        """
        if isinstance(constraints_off, (str, bytes)):
            raise TypeError(
                "constraints_off must be a collection of constraint names, not a "
                f"{type(constraints_off).__name__} ({constraints_off!r})"
            )
        off = set(constraints_off)
        return tuple((name, fn) for name, fn in self._constraints.items() if name not in off)

    def _validate_vector(
        self,
        vector: Vector,
        constraints: _Constraints,
        rejections: _Rejections | None = None,
        kind: _RowKind = "random",
        index: int | None = None,
        labels: tuple[str, ...] = (),
    ) -> bool:
        """
        Validate a vector against the constraints, in order.

        Args:
            vector: Parameter vector to validate
            constraints: The constraints to evaluate (``_evaluated()``)
            rejections: If given, the first constraint that rejects the vector is
                counted in it, by name
            kind, index, labels: The row being drawn, for the message of a
                constraint that raises (``_describe_row()``); an index of None for a
                row without a number (generate_vector())

        Returns:
            True if all constraints pass, False otherwise

        Raises:
            Exception: The exception a constraint raised, unchanged but for a note
                naming the constraint and the row, and its _ConstraintFailure
        """
        for name, constraint in constraints:
            try:
                result = constraint(vector)
                rejected = not result
            except Exception as e:
                origin = self._unnamed_origin(name)
                label = f"{name!r} ({origin})" if origin else repr(name)
                # "random row 3, Vector(...)", or "the row Vector(...)" without a number
                separator = " " if index is None else ", "
                row = f"{_describe_row(kind, index, labels)}{separator}{_short_repr(vector)}"
                failure = _ConstraintFailure(label, row, e, self._off_before(name, constraints))
                failure.attach(e, "constraints_off")
                raise
            if rejected:
                if rejections is not None:
                    rejections.add(name, vector, result)
                return False
        return True

    def _off_before(self, name: str, constraints: _Constraints) -> list[str]:
        """Return the constraints before ``name`` in the mapping that ``constraints`` leaves out."""
        evaluated = {evaluated_name for evaluated_name, _ in constraints}
        before = itertools.takewhile(lambda other: other != name, self._constraints)
        return [other for other in before if other not in evaluated]

    def _describe_rejections(self, rejections: _Rejections) -> str:
        """
        Say which constraints rejected the draws, for a message: the count per name
        of the first failing constraint, and the first row each one rejected.
        """
        counts = rejections.counts
        names = [name for name in self._constraints if counts[name]]
        if not names:
            return ""
        listed = ", ".join(f"{name}={counts[name]}" for name in names)
        examples = []
        for name in names:
            origin = self._unnamed_origin(name)
            notes = [origin] if origin else []
            if rejections.none_only[name]:
                notes.append("returned None, missing return?")
            label = f"{name} ({'; '.join(notes)})" if notes else name
            examples.append(f"{label}: {_short_repr(rejections.first[name])}")
        return (
            f"Rejected by (first failing constraint per draw): {listed}. "
            f"First rows rejected: {'; '.join(examples)}."
        )

    def _exhausted(
        self, what: str, rejections: _Rejections, *, retries: bool
    ) -> _ConstraintsExhausted:
        """
        Build the error for constraints that rejected every draw.

        Args:
            what: What could not be generated, the first sentence without its period
            rejections: The rejections of those draws
            retries: Whether the rows had random arguments to redraw
        """
        counts = rejections.counts
        # The most rejections; max() keeps the first, so a tie goes to the constraint
        # evaluated first
        strictest = max(self._constraints, key=counts.__getitem__) if counts else None
        detail = f"{what}. {self._describe_rejections(rejections)}".rstrip()
        return _ConstraintsExhausted(detail, strictest, retries)

    # ====
    # Vector Management
    # ====

    def add_directed_vector(self, name: str, values: Iterable[Any]) -> None:
        """
        Add a named directed test vector, or replace the one with that name.

        Args:
            name: Unique name for the vector, a non-empty str
            values: The vector, in any form directed_vectors takes: one value per
                argument as a tuple or list, a dict of argument names to values, a
                namedtuple with the argument names as fields, or a
                pytest.param(...) of one of these

        Raises:
            RNGValueError: If the name is not a non-empty str, or the vector does
                not give exactly one value per argument

        Example:
            param.add_directed_vector("edge_case", (0, 100, "fast"))
            param.add_directed_vector("zeros", {"addr": 0, "len": 0})
        """
        self._directed_vectors[name] = _normalize_vector("directed", name, values, self.arg_names)

    def remove_directed_vector(self, name: str) -> None:
        """
        Remove a directed vector by name.

        Args:
            name: Name of the vector to remove

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self._directed_vectors:
            raise KeyError(f"No directed vector named '{name}'")
        del self._directed_vectors[name]

    def add_test_vector(self, name: str, values: Iterable[Any]) -> None:
        """
        Add a named test vector, or replace the one with that name.

        Args:
            name: Unique name for the vector, a non-empty str
            values: The vector, in any form add_directed_vector() takes

        Raises:
            RNGValueError: If the name is not a non-empty str, or the vector does
                not give exactly one value per argument

        Example:
            param.add_test_vector("test_case_1", (0, 100, "fast"))
        """
        self._test_vectors[name] = _normalize_vector("test", name, values, self.arg_names)

    def remove_test_vector(self, name: str) -> None:
        """
        Remove a test vector by name.

        Args:
            name: Name of the vector to remove

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self._test_vectors:
            raise KeyError(f"No test vector named '{name}'")
        del self._test_vectors[name]

    def get_test_vector(self, name: str) -> Vector:
        """
        Get a specific test vector by name.

        Args:
            name: Name of the vector

        Returns:
            The test vector: a Vector, or the pytest.param(...) whose values are one
            (typed as a Vector, as every row is)

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self._test_vectors:
            raise KeyError(f"No test vector named '{name}'")
        return self._test_vectors[name]

    def get_directed_vector(self, name: str) -> Vector:
        """
        Get a specific directed vector by name.

        Args:
            name: Name of the vector

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are
            one (typed as a Vector, as every row is)

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self._directed_vectors:
            raise KeyError(f"No directed vector named '{name}'")
        return self._directed_vectors[name]

    # ====
    # Sample Generation
    # ====

    @property
    def skip_reason(self) -> str | None:
        """
        Why this Parameter's tests are skipped, or None.

        Set when a Series/RNGSequence arg created with skip_if_empty has no values
        (the first such arg, in declaration order). The vector generators then return
        no vectors, and the strategy's tests are reported as skipped with this reason.
        """
        for arg in self.test_args:
            if isinstance(arg.rng_type, SequenceLike) and arg.rng_type.skip_reason is not None:
                return arg.rng_type.skip_reason
        return None

    @property
    def vector_type(self) -> type[Vector]:
        """
        The class of this Parameter's rows: a Vector whose fields are arg_names.

        Parameters with the same argument names share the class.
        """
        return _vector_type(self.arg_names)

    def _build_row(
        self,
        row_type: type[Vector],
        values: list[Any],
        drawn: Sequence[int],
        attempts: int,
        rejections: _Rejections,
        constraints: _Constraints,
        streams: _RowStreams,
        pos: tuple[tuple[str, str], ...],
        j: int,
        kind: _RowKind = "random",
        index: int | None = None,
        labels: tuple[str, ...] = (),
    ) -> Vector | None:
        """
        Fill one row in declaration order and return it once the constraints accept it.

        Every generation path builds its rows here. Each drawn argument draws from
        its own stream for the row (``streams``): its generator is reseeded from the
        row's key first, and is ``RNG.generator()`` only while that argument's RNG
        type and validator run. The constraints run with the generator that was
        there before, which is restored also when something raises.

        Args:
            row_type: The class of the rows (``self.vector_type``)
            values: One slot per argument. The enumerated (Series/sequence) positions
                already hold their validated values; the others are overwritten.
            drawn: The positions to generate, in declaration order: each gets
                ``arg.generate()`` on every attempt
            attempts: How many draws of the drawn positions to try. Each argument's
                stream continues from one attempt to the next.
            rejections: Counts the first constraint that rejects each attempt
            constraints: The constraints to evaluate (``_evaluated()``)
            streams: The generation call's row streams
            pos, j: The row's identity, which keys its streams (see ``_Row``)
            kind, index, labels: The row being drawn, for the message of a
                constraint that raises (an index of None for generate_vector())

        Returns:
            The first attempt's Vector that every constraint accepts, or None when
            all attempts were rejected
        """
        args = self.test_args
        # RNG._generator is the process's: under the ambient generator's lock, which
        # entering and ending a stream hold too, another thread that draws a row or
        # enters a stream meanwhile cannot take an argument's generator for the one
        # to put back (see _Ambient)
        with RNG._ambient._lock:
            ambient = RNG._generator
            generators = streams.start(pos, j, drawn, ambient)
            try:
                for _ in range(attempts):
                    for i, generator in generators:
                        RNG._generator = generator
                        values[i] = args[i].generate()
                    RNG._generator = ambient
                    # The whole row goes to the constraints, as a Vector
                    row = tuple.__new__(row_type, values)
                    if self._validate_vector(row, constraints, rejections, kind, index, labels):
                        return row
            finally:
                RNG._generator = ambient
        return None

    def generate_vector(self, *, _key: StreamKey | None = None) -> Vector:
        """
        Generate a single random parameter vector.

        Args:
            _key: Private: the key of the row's streams. Without it, the key comes
                from the seed and 128 bits of ``RNG.generator()``, so consecutive
                calls differ and ``RNG.seed(s)`` repeats them.

        Returns:
            A Vector of generated values, one per TestArg, every argument drawn (a
            Series or RNGSequence argument too, from its own stream). Unless the
            Parameter has a Series argument, or per_sequence_samples=True with a
            Series or RNGSequence argument, this is the first random row of
            generate_vectors() with the same key.

        Raises:
            ValueError: If the constraints reject max_retries draws; the message
                counts the rejections per constraint name
            Exception: The exception a constraint raised, with a note naming the
                constraint and the row

        Example:
            vector = param.generate_vector()  # e.g., Vector(x=5, y=3.14, mode="fast")
        """
        streams = _RowStreams(_key if _key is not None else _direct_key(), self.test_args)
        return self._random_row(self.vector_type, _Rejections(), self._evaluated(), streams)

    def _random_row(
        self,
        row_type: type[Vector],
        rejections: _Rejections,
        constraints: _Constraints,
        streams: _RowStreams,
        index: int | None = None,
    ) -> Vector:
        """
        Draw every argument of one row, redrawing up to max_retries times (the plain
        path and generate_vector).

        Args:
            row_type: The class of the rows
            rejections: Counts this row's rejected draws (cleared first)
            constraints: The constraints to evaluate (``_evaluated()``)
            streams: The generation call's row streams
            index: The row's number among the random rows (its j), or None for a
                single row, which draws as row 0
        """
        if rejections.counts:
            rejections.clear()
        width = len(self.test_args)
        row = self._build_row(
            row_type,
            [None] * width,
            range(width),
            self.max_retries,
            rejections,
            constraints,
            streams,
            (),
            0 if index is None else index,
            "random",
            index,
        )
        if row is None:
            what = "a random row" if index is None else f"random row {index}"
            raise self._exhausted(
                f"Could not generate {what} after max_retries={self.max_retries} draws",
                rejections,
                retries=True,
            )
        return row

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the parameter metadata to a dictionary.
        """
        return {
            "arguments": [arg.to_dict() for arg in self.test_args],
            "directed_vectors": {
                name: [str(v) for v in _vector_values(vector)]
                for name, vector in self.directed_vectors.items()
            },
            "test_vectors": {
                name: [str(v) for v in _vector_values(vector)]
                for name, vector in self.test_vectors.items()
            },
            "always_include_directed": self.always_include_directed,
            "has_constraints": bool(self.vector_constraints),
            "nsamples": self.nsamples,
            "per_sequence_samples": self.per_sequence_samples,
            "skip_reason": self.skip_reason,
        }

    def generate_vectors(
        self,
        n: int,
        *,
        mode: str = "all",
        filter_by_name: str | None = None,
        filter_by_index: int | None = None,
        constraints_off: Iterable[str] = (),
        _stats: _GenerationStats | None = None,
        _key: StreamKey | None = None,
    ) -> list[Vector]:
        """
        Generate parameter vectors.

        Args:
            n: Number of random samples to generate (an int >= 0). With
                per_sequence_samples=True, this many per combination of the
                Series/RNGSequence args.
            mode: Sampling mode
                - "all": All directed vectors + n random samples (default)
                - "random_only": Only n random samples, no directed
                - "directed_only": Only directed vectors, ignore n
                - "mixed": Directed (if always_include_directed=True) + n random
                - "test": Only test vectors, ignore n and directed
            filter_by_name: Only return this directed vector (for -vn CLI)
            filter_by_index: Only return directed vector at index (for -vi CLI)
            constraints_off: The names of constraints not to evaluate in this call
                (``--strategy-constraint-off``); names the Parameter does not have
                are ignored. The Parameter keeps its constraints.
            _stats: Private: counts the rejections per constraint
            _key: Private: the key of the random rows' streams (the plugin's is the
                test's). Without it, the key comes from the seed and 128 bits of
                ``RNG.generator()``, so consecutive calls differ and ``RNG.seed(s)``
                repeats them.

        Each random row draws each argument from a stream of its own, keyed by the
        row and the argument's name: the first n rows are the same for any larger n,
        and an argument's values do not change when another argument or a
        constraint that accepts the row is added.

        The arguments after n are keyword-only. The plugin does not call this
        method: it generates a test's rows with the same code, so a subclass that
        overrides it does not change the tests' rows.

        Returns:
            List of parameter vectors, each a Vector, or for a pytest.param(...)
            directed or test vector the ParameterSet whose values are a Vector. Every
            row is typed as a Vector, so typed code can unpack the rows and read their
            fields; code that keeps pytest.param rows tells them apart with
            isinstance(row, Vector). Empty when skip_reason is set.

        Raises:
            KeyError / IndexError: If filter_by_name / filter_by_index names no
                directed vector (also when skip_reason is set)
            ValueError: If n is not an int >= 0 in a mode that generates samples
            TypeError: If constraints_off is a str instead of a collection of names
            ValueError: If the vector constraints reject every draw of a random row
                (or every combination); the message counts the rejections by the
                name of the first failing constraint and shows the first row each
                one rejected
            Exception: The exception a constraint raised, with a note naming the
                constraint and the row

        Warns:
            PytestStrategiesWarning: If a Series combination is skipped, or with
                per_sequence_samples=True a combination gets fewer than n rows, because
                its random args did not satisfy the constraints within max_retries
                draws. The warning counts the rejections by constraint name.

        Examples:
            # All directed + 10 random
            samples = param.generate_vectors(10, mode="all")

            # Only random
            samples = param.generate_vectors(10, mode="random_only")

            # Only directed
            samples = param.generate_vectors(0, mode="directed_only")

            # Get specific vector by name
            samples = param.generate_vectors(0, filter_by_name="edge_case")

            # Get specific vector by index
            samples = param.generate_vectors(0, filter_by_index=0)
        """
        rows = self._generate_rows(
            n,
            mode=mode,
            filter_by_name=filter_by_name,
            filter_by_index=filter_by_index,
            constraints_off=constraints_off,
            stats=_stats,
            key=_key,
        )
        if rows and rows[0].kind == "skipped":
            return []
        return [row.values if row.param is None else row.param for row in rows]

    def _generate_rows(
        self,
        n: int,
        *,
        exhaustive: bool = False,
        mode: str = "all",
        filter_by_name: str | None = None,
        filter_by_index: int | None = None,
        constraints_off: Iterable[str] = (),
        stats: _GenerationStats | None = None,
        key: StreamKey | None = None,
    ) -> list[_Row]:
        """
        Generate the rows of a run with their identity: the one entry point of
        generation. generate_vectors() and generate_exhaustive() return the rows'
        values, and the plugin builds each row's test ID and metadata from its kind,
        name, index, pos and j.

        A random or exhaustive row draws each argument from the stream
        ``key/"row"/pos/j/argname`` (``_RowStreams``), and under ``--nsamples=auto``
        an RNGSequence's order comes from ``key/"order"/argname``.

        Args:
            n: Number of random rows (per combination with per_sequence_samples=True),
                as generate_vectors() takes it; not used when exhaustive is set
            exhaustive: Enumerate the Series/RNGSequence args (``--nsamples=auto``)
                instead of drawing n random rows. Vector filters, modes and directed
                vectors apply as they do for a count.
            mode: The sampling mode, as for generate_vectors()
            filter_by_name: Only the directed vector with this name
            filter_by_index: Only the directed vector at this index
            constraints_off: The names of constraints not to evaluate in this call
            stats: Counts the rejections per constraint, and the combinations the
                exhaustive rows left out
            key: The key of the rows' streams: the test's key T for the plugin.
                Without it, the key comes from the seed and 128 bits of
                ``RNG.generator()`` (``_direct_key()``), drawn only when the call
                generates random or exhaustive rows (not for n=0).

        Returns:
            The rows in order: the directed or test vectors the mode and the filters
            select, then the random or exhaustive rows. A Parameter whose
            skip_if_empty sequence has no values gives one "skipped" row instead.

        Raises:
            As generate_vectors(), and with exhaustive set as generate_exhaustive()
        """
        constraints = self._evaluated(constraints_off)

        # Handle CLI filters first (override mode). A missing vector raises even when
        # the Parameter is skipped, so callers can still tell whether a filter matched.
        selected: int | None = None
        if filter_by_name is not None:
            self.get_vector_by_name(filter_by_name)
            selected = list(self._directed_vectors).index(filter_by_name)
        elif filter_by_index is not None:
            self.get_vector_by_index(filter_by_index)
            selected = filter_by_index
        if selected is not None:
            if self.skip_reason is not None:
                return [self._skipped_row()]
            return [self._vector_rows("directed")[selected]]

        # Validate mode
        valid_modes = ["all", "random_only", "directed_only", "mixed", "test"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        # Mode: test - only test vectors; directed_only - only directed vectors
        if mode in ("test", "directed_only"):
            if self.skip_reason is not None:
                return [self._skipped_row()]
            return self._vector_rows("test" if mode == "test" else "directed")

        # The remaining modes generate n random samples. A non-int n would never equal
        # the Series row count (bool is an int subclass, so reject it explicitly).
        if not exhaustive:
            if not isinstance(n, int) or isinstance(n, bool):
                raise ValueError(f"n must be an int, got {n!r}")
            if n < 0:
                raise ValueError(f"n must be >= 0, got {n}")

        # An empty skip_if_empty sequence: nothing to generate, in any mode
        if self.skip_reason is not None:
            return [self._skipped_row()]

        rows: list[_Row] = []
        # Mode: all - always include all directed vectors; random_only - none
        if mode == "all" or mode == "mixed" and self.always_include_directed:
            rows.extend(self._vector_rows("directed"))

        if not exhaustive and n == 0:
            # No random rows: the direct key's bits are not drawn either
            return rows
        if exhaustive:
            # Before the direct key's bits are drawn: a call that fails draws nothing
            self._check_exhaustive()
        if key is None:
            key = _direct_key()
        if exhaustive:
            rows.extend(self._exhaustive_rows(constraints, key, stats))
            return rows

        # Every rejected draw is also counted here, for the plugin's -v summary
        total = stats.rejected if stats is not None else None
        if self.per_sequence_samples and self._sequence_indices():
            rows.extend(self._per_sequence_rows(n, constraints, key, total))
        elif any(isinstance(arg.rng_type, Series) for arg in self.test_args):
            rows.extend(self._series_rows(n, constraints, key, total))
        else:
            row_type = self.vector_type
            rejections = _Rejections(total)
            streams = _RowStreams(key, self.test_args)
            for k in range(n):
                row = self._random_row(row_type, rejections, constraints, streams, k)
                rows.append(_Row("random", None, k, (), k, row))
        return rows

    def _vector_rows(self, kind: Literal["directed", "test"]) -> list[_Row]:
        """Return the directed or the test vectors as rows, in order."""
        stored: Mapping[str, object] = (
            self._directed_vectors if kind == "directed" else self._test_vectors
        )
        rows = []
        for index, (name, vector) in enumerate(stored.items()):
            if isinstance(vector, _ParameterSet):
                # The pytest.param(...) keeps its marks; its values are the Vector
                values = cast(Vector, vector.values)
                rows.append(_Row(kind, name, index, (), None, values, param=vector))
            else:
                rows.append(_Row(kind, name, index, (), None, cast(Vector, vector)))
        return rows

    def _skipped_row(self) -> _Row:
        """Return the one row of a Parameter whose skip_if_empty sequence has no values."""
        values = tuple.__new__(self.vector_type, [None] * len(self.test_args))
        return _Row("skipped", None, None, (), None, values)

    def _generate_row(
        self,
        key: StreamKey,
        pos: Iterable[tuple[str, str]],
        j: int,
        *,
        constraints_off: Iterable[str] = (),
    ) -> Vector:
        """
        Compute one random or exhaustive row on its own: the values the row (pos, j)
        gets in a generation call whose key is ``key``, whatever the call's other
        rows, n or mode (D5).

        The arguments that pos names hold the values their tokens name
        (``_position_keys``), validated; every other argument is drawn from the
        row's streams, with the retries the generation paths give it.

        Args:
            key: The generation call's key: the test's key T, or the ``_key`` of a
                direct call
            pos: The row's enumerated arguments as (argument name, token) pairs, in
                any order; empty for a plain random row
            j: The row's number within its combination (0 for an exhaustive row)
            constraints_off: The names of constraints not to evaluate

        Returns:
            The row, a Vector

        Raises:
            KeyError: If pos names an argument twice, an argument that is not a
                Series or RNGSequence argument, or a token its sequence does not have
            ValueError: If an enumerated value fails its argument's validator, or the
                constraints reject every attempt
        """
        args = self.test_args
        pairs = tuple(sorted(pos))
        tokens = dict(pairs)
        if len(tokens) != len(pairs):
            raise KeyError(f"pos names an argument twice: {pairs!r}")
        values: list[Any] = [None] * len(args)
        labels = []
        drawn = []
        for i, arg in enumerate(args):
            token = tokens.pop(arg.name, None)
            if token is None:
                drawn.append(i)
                continue
            if not isinstance(arg.rng_type, SequenceLike):
                raise KeyError(f"{arg.name!r} is not a Series or RNGSequence argument")
            sequence = arg.rng_type.sequence
            keys = _position_keys(arg.name, sequence)
            position = next((p for p, k in enumerate(keys) if k.token == token), None)
            if position is None:
                raise KeyError(f"{arg.name!r} has no value with the token {token!r}")
            values[i] = arg._validate(sequence[position])
            labels.append(keys[position].label)
        if tokens:
            raise KeyError(f"No argument named {next(iter(tokens))!r}")
        rejections = _Rejections()
        # Redrawing only helps when there are other positions to change
        attempts = self.max_retries if drawn else 1
        row = self._build_row(
            self.vector_type,
            values,
            drawn,
            attempts,
            rejections,
            self._evaluated(constraints_off),
            _RowStreams(key, args),
            pairs,
            j,
            "random",
            j,
            tuple(labels),
        )
        if row is None:
            what = _describe_row("random", j, tuple(labels))
            if drawn:
                what += f" after max_retries={attempts} draws"
            else:
                what += ": the constraints reject its values"
            raise self._exhausted(f"Could not generate {what}", rejections, retries=bool(drawn))
        return row

    def _series_rows(
        self,
        n: int,
        constraints: _Constraints,
        key: StreamKey,
        total: Counter[str] | None = None,
    ) -> list[_Row]:
        """
        Generate n rows that cycle through the combinations of the Series args (the
        finite mode of a Parameter with Series args).

        The Series args are enumerated: their combinations follow declaration order
        (the leftmost arg is the slowest counter), and the visits cycle through them
        until n rows are made. The other args are drawn for every visit, and redrawn
        up to max_retries times when the constraints reject the row. A row's j is the
        cycle of its visit, so a combination the constraints skip leaves a gap.

        Args:
            n: Number of rows
            constraints: The constraints to evaluate (``_evaluated()``)
            key: The key of the rows' streams
            total: If given, counts every rejected draw by constraint name

        Raises:
            ValueError: If a Series value fails its argument's validator
            ValueError: If a whole cycle of combinations in a row produced no row

        Warns:
            PytestStrategiesWarning: Once for each combination skipped because its
                random args did not satisfy the constraints within max_retries draws
        """
        args = self.test_args
        row_type = self.vector_type
        series_indices = [
            i for i, a in enumerate(args) if a.rng_type and isinstance(a.rng_type, Series)
        ]
        series_seqs = [args[i].rng_type.sequence for i in series_indices]
        random_indices = [i for i in range(len(args)) if i not in series_indices]
        # Redrawing only helps when there are non-Series positions to change
        attempts = self.max_retries if random_indices else 1
        num_combos = math.prod(len(seq) for seq in series_seqs)
        enumeration = _Enumeration([args[i].name for i in series_indices], series_seqs)
        streams = _RowStreams(key, args)
        rows: list[_Row] = []
        misses = 0
        # Combinations skipped after redrawing their random args, keyed by their
        # index in the product (Series values need not be hashable), with the
        # rejections of their first skipped visit
        skipped: dict[int, tuple[tuple[Any, ...], _Rejections]] = {}
        # The rejections of one visit, and of the visits since the last row
        visit = _Rejections(total)
        misses_rejections = _Rejections()
        # A combination is a tuple of positions in the Series' sequences
        combinations = itertools.product(*(range(len(seq)) for seq in series_seqs))
        for k, combination in enumerate(itertools.cycle(combinations)):
            if len(rows) >= n:
                break
            combo = tuple(seq[p] for seq, p in zip(series_seqs, combination))
            vec: list[Any] = [None] * len(args)
            # Series values skip arg.generate(), so apply the arg's validator here
            for idx, value in zip(series_indices, combo):
                vec[idx] = args[idx]._validate(value)
            # The visit's cycle: a skipped visit leaves its number unused
            j = k // num_combos
            _, labels, pos = enumeration.identify(combination)
            # Try fresh random values for the non-Series positions
            if visit.counts:
                visit.clear()
            row = self._build_row(
                row_type,
                vec,
                random_indices,
                attempts,
                visit,
                constraints,
                streams,
                pos,
                j,
                "random",
                j,
                labels,
            )
            if row is not None:
                rows.append(_Row("random", None, j, pos, j, row, labels))
                misses = 0
                if misses_rejections.counts:
                    misses_rejections.clear()
            else:
                # Skip a combination the constraints reject and move on to the next
                # one, unless a whole cycle in a row has produced nothing
                misses += 1
                misses_rejections.merge(visit)
                if misses == num_combos:
                    raise self._exhausted(
                        "Could not generate valid vector: none of the "
                        f"{num_combos} Series combinations satisfied the vector "
                        f"constraints ({attempts} attempt(s) each)",
                        misses_rejections,
                        retries=bool(random_indices),
                    )
                # With random args the rejection may just be unlucky draws of a valid
                # combination, so the skip must not go unnoticed
                if random_indices and k % num_combos not in skipped:
                    skipped[k % num_combos] = (combo, visit.copy())
        # Warn once per skipped combination, and only when no error was raised
        for combo, rejected in skipped.values():
            values = ", ".join(
                f"{args[idx].name}={value!r}" for idx, value in zip(series_indices, combo)
            )
            warnings.warn(
                f"Series combination ({values}) skipped: the vector constraints "
                f"rejected max_retries={attempts} draws of the non-Series args. "
                f"{self._describe_rejections(rejected)} "
                "Raise max_retries, or relax the constraints if this combination "
                "should be tested.",
                PytestStrategiesWarning,
                # The caller of generate_vectors()
                stacklevel=4,
            )
        return rows

    def _sequence_indices(self) -> list[int]:
        """Return the positions of the Series/RNGSequence args."""
        return [
            i
            for i, a in enumerate(self.test_args)
            if a.rng_type and isinstance(a.rng_type, SequenceLike)
        ]

    def _check_exhaustive(self) -> None:
        """
        Raise the error of an exhaustive call on a Parameter without Series or
        RNGSequence args.

        Raises:
            ValueError: If no sequence arguments are present
        """
        if not self._sequence_indices():
            raise ValueError("No sequence arguments found for exhaustive generation")

    def _per_sequence_rows(
        self,
        n: int,
        constraints: _Constraints,
        key: StreamKey,
        total: Counter[str] | None = None,
    ) -> list[_Row]:
        """
        Generate n random rows for every combination of the sequence args.

        Combinations follow declaration order (leftmost arg is the slowest counter), for
        RNGSequence as well as Series. The non-sequence args are drawn fresh for every
        row and redrawn up to max_retries times when the constraints reject the vector.
        A row's j is its number within its combination.

        Args:
            n: Number of rows per combination
            constraints: The constraints to evaluate (``_evaluated()``)
            key: The key of the rows' streams
            total: If given, counts every rejected draw by constraint name

        Returns:
            The rows, grouped by combination

        Raises:
            ValueError: If a sequence value fails its argument's validator
            ValueError: If the vector constraints reject every combination

        Warns:
            PytestStrategiesWarning: If a combination produced fewer than n rows because
                its random args did not satisfy the constraints within max_retries draws
        """
        # No rows asked for: don't walk (and validate) the whole product
        if n == 0:
            return []

        args = self.test_args
        sequence_indices = self._sequence_indices()
        sequences = [args[i].rng_type.sequence for i in sequence_indices]
        random_indices = [i for i in range(len(args)) if i not in sequence_indices]
        # Redrawing only helps when there are non-sequence positions to change
        attempts = self.max_retries if random_indices else 1

        row_type = self.vector_type
        enumeration = _Enumeration([args[i].name for i in sequence_indices], sequences)
        streams = _RowStreams(key, args)
        rows: list[_Row] = []
        # Combinations cut short after redrawing their random args, with their row
        # count and the rejections of the row they could not fill
        short: list[tuple[tuple[Any, ...], int, _Rejections]] = []
        # The rejections of one row, and of every row that could not be filled
        visit = _Rejections(total)
        failed = _Rejections()
        for combination in itertools.product(*(range(len(seq)) for seq in sequences)):
            combo = tuple(seq[p] for seq, p in zip(sequences, combination))
            vec: list[Any] = [None] * len(args)
            # Sequence values skip arg.generate(), so apply the arg's validator here
            for idx, value in zip(sequence_indices, combo):
                vec[idx] = args[idx]._validate(value)
            _, labels, pos = enumeration.identify(combination)

            made = 0
            while made < n:
                if visit.counts:
                    visit.clear()
                row = self._build_row(
                    row_type,
                    vec,
                    random_indices,
                    attempts,
                    visit,
                    constraints,
                    streams,
                    pos,
                    made,
                    "random",
                    made,
                    labels,
                )
                if row is None:
                    # Further rows of this combination would most likely fail too
                    failed.merge(visit)
                    if random_indices:
                        short.append((combo, made, visit.copy()))
                    break
                rows.append(_Row("random", None, made, pos, made, row, labels))
                made += 1

        num_combos = math.prod(len(seq) for seq in sequences)
        if n and num_combos and not rows:
            raise self._exhausted(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each)",
                failed,
                retries=bool(random_indices),
            )

        for combo, made, rejected in short:
            values = ", ".join(
                f"{args[idx].name}={value!r}" for idx, value in zip(sequence_indices, combo)
            )
            warnings.warn(
                f"Sequence combination ({values}) produced {made} of {n} rows: the vector "
                f"constraints rejected max_retries={attempts} draws of the other args. "
                f"{self._describe_rejections(rejected)} "
                "Raise max_retries, or relax the constraints if this combination "
                "should be tested.",
                PytestStrategiesWarning,
                # The caller of generate_vectors()
                stacklevel=4,
            )

        return rows

    def generate_exhaustive(
        self,
        *,
        constraints_off: Iterable[str] = (),
        _stats: _GenerationStats | None = None,
        _key: StreamKey | None = None,
    ) -> list[Vector]:
        """
        Generate all combinations of sequence arguments (Cartesian product).
        For non-sequence arguments, generate a random value for each combination.

        A combination the constraints still reject after max_retries draws of its
        random arguments is left out. The plugin does not call this method, as for
        generate_vectors().

        Args:
            constraints_off: The names of constraints not to evaluate in this call
                (``--strategy-constraint-off``); names the Parameter does not have
                are ignored. The Parameter keeps its constraints.
            _stats: Private: counts the rejections per constraint and the
                combinations left out
            _key: Private: the key of the rows' streams, as for generate_vectors()

        Returns:
            List of Vectors. Empty when skip_reason is set.

        Raises:
            TypeError: If constraints_off is a str instead of a collection of names
            ValueError: If no sequence arguments are present
            ValueError: If a sequence value fails its argument's validator
            ValueError: If the vector constraints reject every combination
            Exception: The exception a constraint raised, with a note naming the
                constraint and the row
        """
        constraints = self._evaluated(constraints_off)

        # An empty skip_if_empty sequence has no combinations to enumerate
        if self.skip_reason is not None:
            return []
        # Before the direct key's bits are drawn: a call that fails draws nothing
        self._check_exhaustive()
        key = _key if _key is not None else _direct_key()
        return [row.values for row in self._exhaustive_rows(constraints, key, _stats)]

    def _exhaustive_rows(
        self,
        constraints: _Constraints,
        key: StreamKey,
        stats: _GenerationStats | None = None,
    ) -> list[_Row]:
        """
        Generate one row for every combination of the sequence args (Cartesian
        product), drawing the other args for each.

        The combinations follow each arg's ``--nsamples=auto`` order: a Series in
        declaration order, an RNGSequence in a random permutation drawn from its
        own stream (``key/"order"/argname``). A row's index is its combination's
        position in the declaration-order product, whatever that order, and its j
        is 0.

        Args:
            constraints: The constraints to evaluate (``_evaluated()``)
            key: The key of the rows' streams and of the sequences' orders
            stats: If given, counts the rejections per constraint and the
                combinations left out

        Raises:
            ValueError: If no sequence arguments are present (callers that draw
                the direct key check it first, with ``_check_exhaustive()``)
            ValueError: If a sequence value fails its argument's validator
            ValueError: If the vector constraints reject every combination
        """
        args = self.test_args
        # Identify sequence args and their indices, with their values in auto order
        sequence_indices = []
        orders = []
        for i, arg in enumerate(args):
            if arg.rng_type and isinstance(arg.rng_type, SequenceLike):
                sequence_indices.append(i)
                orders.append(_auto_order_from(key.child("order", arg.name), arg.rng_type))

        if not orders:
            self._check_exhaustive()

        random_indices = [i for i in range(len(args)) if i not in sequence_indices]
        # Redrawing only helps when there are non-sequence positions to change
        attempts = self.max_retries if random_indices else 1

        # Generate Cartesian product
        row_type = self.vector_type
        enumeration = _Enumeration(
            [args[i].name for i in sequence_indices],
            [args[i].rng_type.sequence for i in sequence_indices],
        )
        streams = _RowStreams(key, args)
        rows: list[_Row] = []
        # The rejections of one combination, and of the combinations left out
        visit = _Rejections(stats.rejected if stats is not None else None)
        dropped = _Rejections()
        for combination in itertools.product(*orders):
            positions = [position for position, _ in combination]
            # Create a mutable vector (list) to fill in
            vector: list[Any] = [None] * len(args)

            # Fill in sequence values (they skip arg.generate(), so validate them here)
            for idx, (_, value) in zip(sequence_indices, combination):
                vector[idx] = args[idx]._validate(value)

            # Fill in non-sequence values with random generation, redrawing them if the
            # constraints reject the vector. A combination that still fails (e.g. its
            # sequence values alone break a constraint) is dropped.
            index, labels, pos = enumeration.identify(positions)
            if visit.counts:
                visit.clear()
            row = self._build_row(
                row_type,
                vector,
                random_indices,
                attempts,
                visit,
                constraints,
                streams,
                pos,
                0,
                "exhaustive",
                index,
                labels,
            )
            if row is not None:
                rows.append(_Row("exhaustive", None, index, pos, 0, row, labels))
            else:
                dropped.merge(visit)

        # No rows from a non-empty product means the constraints rejected every
        # combination. Fail like finite mode does instead of yielding an empty parameter
        # set, which pytest would silently skip.
        num_combos = math.prod(len(order) for order in orders)
        if num_combos and not rows:
            raise self._exhausted(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each)",
                dropped,
                retries=bool(random_indices),
            )
        if stats is not None:
            stats.left_out += num_combos - len(rows)

        return rows

    # ====
    # CLI Support
    # ====

    def get_vector_by_name(self, name: str) -> Vector:
        """
        Get directed vector by name (for -vn CLI argument).

        Args:
            name: Name of the directed vector

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are
            one (typed as a Vector, as every row is)

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self.directed_vectors:
            available = ", ".join(self.directed_vectors.keys())
            raise KeyError(f"No directed vector named '{name}'. " f"Available: {available}")
        return self.directed_vectors[name]

    def get_vector_by_index(self, index: int) -> Vector:
        """
        Get directed vector by index (for -vi CLI argument).

        Args:
            index: Index of the directed vector (0-based)

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are
            one (typed as a Vector, as every row is)

        Raises:
            IndexError: If index is out of range
        """
        names = list(self.directed_vectors.keys())
        if not names:
            raise IndexError(f"Vector index {index} out of range: there are no directed vectors")
        if index < 0 or index >= len(names):
            raise IndexError(
                f"Vector index {index} out of range. " f"Valid range: 0-{len(names)-1}"
            )
        return self.directed_vectors[names[index]]

    def list_vector_names(self) -> list[str]:
        """
        List all directed vector names.

        Returns:
            List of vector names in order
        """
        return list(self.directed_vectors.keys())

    # ====
    # Constraint Management
    # ====

    def add_constraint(self, fn: Callable[[Vector], object], *, name: str | None = None) -> str:
        """
        Add a constraint that validates entire parameter vectors, evaluated after
        the ones already there.

        Args:
            fn: Function that takes the row as a Vector and returns a truth value
                (a falsy result rejects the row)
            name: The constraint's name. Without it, a function or bound method is
                named after its ``__name__``, and a lambda, partial or callable
                object ``constraint_<i>``, i being its position. A name is a
                non-empty str without whitespace, ":", "," or "=".

        The arguments after fn are keyword-only.

        Returns:
            The name the constraint got

        Raises:
            TypeError: If fn is not callable
            RNGValueError: If the name is not valid or already taken, or fn is
                already a constraint

        Example:
            # Ensure first arg < second arg
            param.add_constraint(lambda v: v.lo < v.hi, name="ordered")
        """
        if not callable(fn):
            raise TypeError(f"add_constraint() takes a callable, got {fn!r} ({type(fn).__name__})")
        unnamed = False
        if name is not None:
            name = _check_constraint_name(name)
        else:
            name = _function_name(fn)
            if name is None:
                name = self._free_name(len(self._constraints))
                unnamed = True
        self._insert(name, fn, "add_constraint()'s fn", unnamed=unnamed)
        return name

    def remove_constraint(self, name: str) -> None:
        """
        Remove a constraint by name.

        Args:
            name: Name of the constraint to remove

        Raises:
            KeyError: If no constraint has that name; the message lists the names
        """
        if name not in self._constraints:
            names = ", ".join(self._constraints) or "none"
            raise KeyError(f"No constraint named {name!r}. Constraints: {names}")
        del self._constraints[name]
        self._unnamed.discard(name)

    def clear_constraints(self) -> None:
        """Remove all vector constraints."""
        self._constraints.clear()
        self._unnamed.clear()

    # ====
    # Introspection
    # ====

    @property
    def arg_names(self) -> tuple[str, ...]:
        """Get tuple of argument names."""
        return tuple(arg.name for arg in self.test_args)

    @property
    def arg_types(self) -> tuple[type, ...]:
        """Get tuple of argument types."""
        return tuple(arg.type for arg in self.test_args)

    @property
    def vector_names(self) -> list[str]:
        """Get list of directed vector names."""
        return list(self.directed_vectors.keys())

    @property
    def num_args(self) -> int:
        """Get number of test arguments."""
        return len(self.test_args)

    @property
    def num_directed_vectors(self) -> int:
        """Get number of directed vectors."""
        return len(self.directed_vectors)

    def get_arg(self, name: str) -> TestArg:
        """
        Get TestArg by name.

        Args:
            name: Name of the argument

        Returns:
            The TestArg instance

        Raises:
            KeyError: If argument name doesn't exist
        """
        for arg in self.test_args:
            if arg.name == name:
                return arg
        raise KeyError(f"No argument named '{name}'")

    # ====
    # String Representation
    # ====

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Parameter(args={self.num_args}, " f"directed_vectors={self.num_directed_vectors})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        args_str = ", ".join(self.arg_names)
        vectors_str = ", ".join(self.vector_names) if self.vector_names else "none"
        return f"Parameter({args_str})\n" f"  Directed vectors: {vectors_str}"
