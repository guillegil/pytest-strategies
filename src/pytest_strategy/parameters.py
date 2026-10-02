# parameter.py

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import functools
import inspect
import itertools
import keyword
import math
import os
import reprlib
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

from ._vector import Vector
from ._vector import vector_type as _vector_type
from ._warnings import PytestStrategiesWarning
from .rng import RNGValueError, SequenceLike, Series
from .test_args import TestArg

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

# pytest.param() returns a ParameterSet (a NamedTuple), which pytest does not export
_ParameterSet = type(pytest.param())


def _normalize_vector(
    kind: str, name: object, raw: object, arg_names: tuple[str, ...]
) -> Vector | ParameterSet:
    """
    Turn a directed or test vector, as given, into the row it stands for.

    The rules, in order:

    - a ``pytest.param(...)`` keeps its marks and id, and its values become the
      Vector: when its only value is a Mapping, that Mapping is a named vector,
      otherwise its values are taken by position;
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
        vector the ParameterSet with that Vector as its values

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
        given = raw.values
        if len(given) == 1 and isinstance(given[0], Mapping):
            row = _vector_by_name(where, given[0], "dict", arg_names, row_type)
        else:
            row = _vector_by_position(where, given, row_type)
        return raw._replace(values=row)

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
) -> Vector:
    """
    Build the row of a vector whose values are given by argument name.

    Args:
        where: The vector, for the messages ("Directed vector 'zeros'")
        values: The values by argument name, in any order
        what: "dict" or "namedtuple", for the messages
        arg_names: The strategy's argument names, in declaration order
        row_type: The class of the row
    """
    part = "keys" if what == "dict" else "fields"
    expected = f"The {part} of a {what} vector are the strategy's arguments: {', '.join(arg_names)}"
    if what == "dict" and len(arg_names) == 1:
        # A dict is always a named vector, also for a one-argument strategy
        shown = reprlib.repr(dict(values))
        expected += (
            f". A dict value for the one argument is written ({shown},) or "
            f"{{{arg_names[0]!r}: {shown}}}"
        )
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


def _vector_values(vector: Vector | ParameterSet) -> Sequence[Any]:
    """Return the values of a stored directed or test vector (a pytest.param's values)."""
    return vector.values if isinstance(vector, _ParameterSet) else vector


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


class _ConstraintError(ValueError):
    """
    A constraint raised: names the constraint and the row.

    It is raised from the constraint's own exception, which the resolver chains
    the collection error to, so the user's frame is shown.

    Attributes:
        detail: The message without the note on the constraints turned off
        off_before: The constraints before it in the mapping that the generation
            call turned off (one of them may have guarded it)
    """

    def __init__(
        self,
        label: str,
        row: Vector,
        index: int | None,
        error: Exception,
        off_before: Sequence[str] = (),
    ) -> None:
        where = "the row" if index is None else f"random row {index},"
        self.detail = (
            f"Constraint {label} raised {type(error).__name__} on {where} "
            f"{_short_repr(row)}: {error}"
        )
        self.off_before = tuple(off_before)
        super().__init__(self.message("constraints_off"))

    def message(self, turned_off_by: str) -> str:
        """
        Return the message, with a note naming the constraints before this one that
        ``turned_off_by`` turned off. The resolver names ``--strategy-constraint-off``.
        """
        if not self.off_before:
            return self.detail
        names = ", ".join(repr(name) for name in self.off_before)
        if len(self.off_before) == 1:
            note = f"constraint {names} before it is turned off"
        else:
            note = f"constraints {names} before it are turned off"
        return f"{self.detail} ({note} by {turned_off_by})"


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
    ) -> None:
        """
        Initialize a Parameter container.

        Args:
            *test_args: Variable number of TestArg instances
            directed_vectors: Mapping of vector names (non-empty strings) to vectors.
                A vector gives one value per argument: a tuple or list in declaration
                order, a dict of argument names to values in any order, or a
                namedtuple whose fields are the argument names. A
                pytest.param(*values, marks=...) of one of these keeps its marks (and
                id) on its row. Each is stored as a Vector; a dict value for a
                one-argument strategy is written ({"a": 1},) or {"cfg": {"a": 1}}.
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

        Raises:
            ValueError: If nsamples, max_retries or max_exhaustive is not a valid count,
                or per_sequence_samples is not a bool
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

    @property
    def directed_vectors(self) -> Mapping[str, Vector | ParameterSet]:
        """
        The directed vectors by name, as Vectors (a pytest.param(...) vector keeps its
        marks, with a Vector as its values).

        Read-only: add_directed_vector() and remove_directed_vector() change it.
        """
        return MappingProxyType(self._directed_vectors)

    @property
    def test_vectors(self) -> Mapping[str, Vector | ParameterSet]:
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

        Raises:
            TypeError: If constraints_off is a str instead of a collection of names
            ValueError: If constraints_off names a constraint this Parameter does
                not have
        """
        if isinstance(constraints_off, (str, bytes)):
            raise TypeError(
                "constraints_off must be a collection of constraint names, not a "
                f"{type(constraints_off).__name__} ({constraints_off!r})"
            )
        off = dict.fromkeys(constraints_off)
        unknown = [name for name in off if name not in self._constraints]
        if unknown:
            names = ", ".join(self._constraints) or "none"
            raise ValueError(
                f"constraints_off names no constraint {', '.join(map(repr, unknown))}. "
                f"Constraints: {names}"
            )
        return tuple((name, fn) for name, fn in self._constraints.items() if name not in off)

    def _validate_vector(
        self,
        vector: Vector,
        constraints: _Constraints,
        rejections: _Rejections | None = None,
        index: int | None = None,
    ) -> bool:
        """
        Validate a vector against the constraints, in order.

        Args:
            vector: Parameter vector to validate
            constraints: The constraints to evaluate (``_evaluated()``)
            rejections: If given, the first constraint that rejects the vector is
                counted in it, by name
            index: The number of the random row being drawn, for the message of a
                constraint that raises (None when it has no such number)

        Returns:
            True if all constraints pass, False otherwise

        Raises:
            _ConstraintError: If a constraint raises, chained to its exception
        """
        for name, constraint in constraints:
            try:
                result = constraint(vector)
                rejected = not result
            except Exception as e:
                origin = self._unnamed_origin(name)
                label = f"{name!r} ({origin})" if origin else repr(name)
                raise _ConstraintError(
                    label, vector, index, e, self._off_before(name, constraints)
                ) from e
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

    def get_test_vector(self, name: str) -> Vector | ParameterSet:
        """
        Get a specific test vector by name.

        Args:
            name: Name of the vector

        Returns:
            The test vector: a Vector, or the pytest.param(...) whose values are one

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self._test_vectors:
            raise KeyError(f"No test vector named '{name}'")
        return self._test_vectors[name]

    def get_directed_vector(self, name: str) -> Vector | ParameterSet:
        """
        Get a specific directed vector by name.

        Args:
            name: Name of the vector

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are one

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
        index: int | None = None,
    ) -> Vector | None:
        """
        Fill one row in declaration order and return it once the constraints accept it.

        Every generation path builds its rows here.

        Args:
            row_type: The class of the rows (``self.vector_type``)
            values: One slot per argument. The enumerated (Series/sequence) positions
                already hold their validated values; the others are overwritten.
            drawn: The positions to generate, in declaration order: each gets
                ``arg.generate()`` on every attempt
            attempts: How many draws of the drawn positions to try
            rejections: Counts the first constraint that rejects each attempt
            constraints: The constraints to evaluate (``_evaluated()``)
            index: The number of the random row, for the message of a constraint
                that raises

        Returns:
            The first attempt's Vector that every constraint accepts, or None when
            all attempts were rejected
        """
        args = self.test_args
        for _ in range(attempts):
            for i in drawn:
                values[i] = args[i].generate()
            # The whole row goes to the constraints, as a Vector
            row = tuple.__new__(row_type, values)
            if self._validate_vector(row, constraints, rejections, index):
                return row
        return None

    def generate_vector(self) -> Vector:
        """
        Generate a single random parameter vector.

        Returns:
            A Vector of generated values, one per TestArg

        Raises:
            ValueError: If the constraints reject max_retries draws; the message
                counts the rejections per constraint name
            ValueError: If a constraint raises, naming the constraint and the row

        Example:
            vector = param.generate_vector()  # e.g., Vector(x=5, y=3.14, mode="fast")
        """
        return self._random_row(self.vector_type, _Rejections(), self._evaluated())

    def _random_row(
        self,
        row_type: type[Vector],
        rejections: _Rejections,
        constraints: _Constraints,
        index: int | None = None,
    ) -> Vector:
        """
        Draw every argument of one row, redrawing up to max_retries times (the plain
        path and generate_vector).

        Args:
            row_type: The class of the rows
            rejections: Counts this row's rejected draws (cleared first)
            constraints: The constraints to evaluate (``_evaluated()``)
            index: The row's number among the random rows, or None for a single row
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
    ) -> list[Vector | ParameterSet]:
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
                (``--strategy-constraint-off``). The Parameter keeps them.
            _stats: Private: counts the rejections per constraint, for the plugin

        The arguments after n are keyword-only.

        Returns:
            List of parameter vectors, each a Vector, or for a pytest.param(...)
            directed or test vector the ParameterSet whose values are a Vector. Empty
            when skip_reason is set.

        Raises:
            KeyError / IndexError: If filter_by_name / filter_by_index names no
                directed vector (also when skip_reason is set)
            ValueError: If n is not an int >= 0 in a mode that generates samples
            ValueError: If constraints_off names a constraint the Parameter does not
                have (TypeError if it is a str)
            ValueError: If the vector constraints reject every draw of a random row
                (or every combination); the message counts the rejections by the
                name of the first failing constraint and shows the first row each
                one rejected
            ValueError: If a constraint raises, naming the constraint and the row
                (chained to the constraint's exception)

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
        samples: list[Vector | ParameterSet] = []
        constraints = self._evaluated(constraints_off)

        # Handle CLI filters first (override mode). A missing vector raises even when
        # the Parameter is skipped, so callers can still tell whether a filter matched.
        if filter_by_name is not None:
            vector = self.get_vector_by_name(filter_by_name)
            return [] if self.skip_reason is not None else [vector]

        if filter_by_index is not None:
            vector = self.get_vector_by_index(filter_by_index)
            return [] if self.skip_reason is not None else [vector]

        # Validate mode
        valid_modes = ["all", "random_only", "directed_only", "mixed", "test"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        # Mode: test - only test vectors
        if mode == "test":
            return [] if self.skip_reason is not None else list(self.test_vectors.values())

        # Mode: directed_only
        if mode == "directed_only":
            return [] if self.skip_reason is not None else list(self.directed_vectors.values())

        # The remaining modes generate n random samples. A non-int n would never equal
        # the Series row count below (bool is an int subclass, so reject it explicitly).
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError(f"n must be an int, got {n!r}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")

        # An empty skip_if_empty sequence: nothing to generate, in any mode
        if self.skip_reason is not None:
            return []

        # Mode: all - always include all directed vectors
        if mode == "all" or mode == "mixed" and self.always_include_directed:
            samples.extend(self.directed_vectors.values())

        # Mode: random_only - skip directed vectors entirely
        # (no action needed, samples stays empty)

        # Every rejected draw is also counted here, for the plugin's -v summary
        total = _stats.rejected if _stats is not None else None

        # Generate samples (for all modes except directed_only)
        if mode != "directed_only" and self.per_sequence_samples and self._sequence_indices():
            samples.extend(self._generate_per_sequence(n, constraints, total))
        elif mode != "directed_only":
            row_type = self.vector_type
            # Series-aware branch: if any arg uses Series, produce ordered/cycling rows
            series_indices = [
                i
                for i, a in enumerate(self.test_args)
                if a.rng_type and isinstance(a.rng_type, Series)
            ]
            if series_indices:
                series_seqs = [self.test_args[i].rng_type.sequence for i in series_indices]
                random_indices = [i for i in range(len(self.test_args)) if i not in series_indices]
                # Redrawing only helps when there are non-Series positions to change
                attempts = self.max_retries if random_indices else 1
                num_combos = math.prod(len(seq) for seq in series_seqs)
                series_rows = 0
                misses = 0
                # Combinations skipped after redrawing their random args, keyed by their
                # index in the product (Series values need not be hashable), with the
                # rejections of their first skipped visit
                skipped: dict[int, tuple[tuple[Any, ...], _Rejections]] = {}
                # The rejections of one visit, and of the visits since the last row
                visit = _Rejections(total)
                misses_rejections = _Rejections()
                for k, combo in enumerate(itertools.cycle(itertools.product(*series_seqs))):
                    if series_rows >= n:
                        break
                    vec: list[Any] = [None] * len(self.test_args)
                    # Series values skip arg.generate(), so apply the arg's validator here
                    for pos, idx in enumerate(series_indices):
                        vec[idx] = self.test_args[idx]._validate(combo[pos])
                    # Try fresh random values for the non-Series positions
                    if visit.counts:
                        visit.clear()
                    row = self._build_row(
                        row_type, vec, random_indices, attempts, visit, constraints
                    )
                    if row is not None:
                        samples.append(row)
                        series_rows += 1
                        misses = 0
                        if misses_rejections.counts:
                            misses_rejections.clear()
                    else:
                        # Skip a combination the constraints reject and move on to the
                        # next one, unless a whole cycle in a row has produced nothing
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
                        # With random args the rejection may just be unlucky draws of a
                        # valid combination, so the skip must not go unnoticed
                        if random_indices and k % num_combos not in skipped:
                            skipped[k % num_combos] = (combo, visit.copy())
                # Warn once per skipped combination, and only when no error was raised
                for combo, rejected in skipped.values():
                    values = ", ".join(
                        f"{self.test_args[idx].name}={value!r}"
                        for idx, value in zip(series_indices, combo)
                    )
                    warnings.warn(
                        f"Series combination ({values}) skipped: the vector constraints "
                        f"rejected max_retries={attempts} draws of the non-Series args. "
                        f"{self._describe_rejections(rejected)} "
                        "Raise max_retries, or relax the constraints if this combination "
                        "should be tested.",
                        PytestStrategiesWarning,
                        stacklevel=2,
                    )
            else:
                rejections = _Rejections(total)
                for k in range(n):
                    samples.append(self._random_row(row_type, rejections, constraints, k))

        return samples

    def _sequence_indices(self) -> list[int]:
        """Return the positions of the Series/RNGSequence args."""
        return [
            i
            for i, a in enumerate(self.test_args)
            if a.rng_type and isinstance(a.rng_type, SequenceLike)
        ]

    def _generate_per_sequence(
        self, n: int, constraints: _Constraints, total: Counter[str] | None = None
    ) -> list[Vector]:
        """
        Generate n random rows for every combination of the sequence args.

        Combinations follow declaration order (leftmost arg is the slowest counter), for
        RNGSequence as well as Series. The non-sequence args are drawn fresh for every
        row and redrawn up to max_retries times when the constraints reject the vector.

        Args:
            n: Number of rows per combination
            constraints: The constraints to evaluate (``_evaluated()``)
            total: If given, counts every rejected draw by constraint name

        Returns:
            List of Vectors, grouped by combination

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

        sequence_indices = self._sequence_indices()
        sequences = [self.test_args[i].rng_type.sequence for i in sequence_indices]
        random_indices = [i for i in range(len(self.test_args)) if i not in sequence_indices]
        # Redrawing only helps when there are non-sequence positions to change
        attempts = self.max_retries if random_indices else 1

        row_type = self.vector_type
        samples: list[Vector] = []
        # Combinations cut short after redrawing their random args, with their row
        # count and the rejections of the row they could not fill
        short: list[tuple[tuple[Any, ...], int, _Rejections]] = []
        # The rejections of one row, and of every row that could not be filled
        visit = _Rejections(total)
        failed = _Rejections()
        for combo in itertools.product(*sequences):
            vec: list[Any] = [None] * len(self.test_args)
            # Sequence values skip arg.generate(), so apply the arg's validator here
            for idx, value in zip(sequence_indices, combo):
                vec[idx] = self.test_args[idx]._validate(value)

            rows = 0
            while rows < n:
                if visit.counts:
                    visit.clear()
                row = self._build_row(row_type, vec, random_indices, attempts, visit, constraints)
                if row is None:
                    # Further rows of this combination would most likely fail too
                    failed.merge(visit)
                    if random_indices:
                        short.append((combo, rows, visit.copy()))
                    break
                samples.append(row)
                rows += 1

        num_combos = math.prod(len(seq) for seq in sequences)
        if n and num_combos and not samples:
            raise self._exhausted(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each)",
                failed,
                retries=bool(random_indices),
            )

        for combo, rows, rejected in short:
            values = ", ".join(
                f"{self.test_args[idx].name}={value!r}"
                for idx, value in zip(sequence_indices, combo)
            )
            warnings.warn(
                f"Sequence combination ({values}) produced {rows} of {n} rows: the vector "
                f"constraints rejected max_retries={attempts} draws of the other args. "
                f"{self._describe_rejections(rejected)} "
                "Raise max_retries, or relax the constraints if this combination "
                "should be tested.",
                PytestStrategiesWarning,
                stacklevel=3,
            )

        return samples

    def generate_exhaustive(
        self, *, constraints_off: Iterable[str] = (), _stats: _GenerationStats | None = None
    ) -> list[Vector]:
        """
        Generate all combinations of sequence arguments (Cartesian product).
        For non-sequence arguments, generate a random value for each combination.

        A combination the constraints still reject after max_retries draws of its
        random arguments is left out.

        Args:
            constraints_off: The names of constraints not to evaluate in this call
                (``--strategy-constraint-off``). The Parameter keeps them.
            _stats: Private: counts the rejections per constraint and the
                combinations left out, for the plugin

        Returns:
            List of Vectors. Empty when skip_reason is set.

        Raises:
            ValueError: If constraints_off names a constraint the Parameter does not
                have (TypeError if it is a str)
            ValueError: If no sequence arguments are present
            ValueError: If a sequence value fails its argument's validator
            ValueError: If the vector constraints reject every combination
            ValueError: If a constraint raises, naming the constraint and the row
        """
        constraints = self._evaluated(constraints_off)

        # An empty skip_if_empty sequence has no combinations to enumerate
        if self.skip_reason is not None:
            return []

        # Identify sequence args and their indices
        sequence_indices = []
        sequences = []

        for i, arg in enumerate(self.test_args):
            if arg.rng_type and isinstance(arg.rng_type, SequenceLike):
                sequence_indices.append(i)
                sequences.append(arg.rng_type._get_auto_sequence())

        if not sequences:
            # If no sequences, fallback to a single random sample?
            # Or raise error? The plan implies this is for "auto" mode with sequences.
            # If "auto" is used without sequences, maybe default to 10 random samples?
            # For now, let's raise error or return empty, but strategy should handle fallback.
            # Let's return a single random sample to be safe if called directly,
            # but Strategy should probably check this.
            # Actually, let's raise ValueError as per docstring.
            raise ValueError("No sequence arguments found for exhaustive generation")

        random_indices = [i for i in range(len(self.test_args)) if i not in sequence_indices]
        # Redrawing only helps when there are non-sequence positions to change
        attempts = self.max_retries if random_indices else 1

        # Generate Cartesian product
        row_type = self.vector_type
        samples: list[Vector] = []
        # The rejections of one combination, and of the combinations left out
        visit = _Rejections(_stats.rejected if _stats is not None else None)
        dropped = _Rejections()
        for combination in itertools.product(*sequences):
            # Create a mutable vector (list) to fill in
            vector: list[Any] = [None] * len(self.test_args)

            # Fill in sequence values (they skip arg.generate(), so validate them here)
            for idx, value in zip(sequence_indices, combination):
                vector[idx] = self.test_args[idx]._validate(value)

            # Fill in non-sequence values with random generation, redrawing them if the
            # constraints reject the vector. A combination that still fails (e.g. its
            # sequence values alone break a constraint) is dropped.
            if visit.counts:
                visit.clear()
            row = self._build_row(row_type, vector, random_indices, attempts, visit, constraints)
            if row is not None:
                samples.append(row)
            else:
                dropped.merge(visit)

        # No samples from a non-empty product means the constraints rejected every
        # combination. Fail like finite mode does instead of yielding an empty parameter
        # set, which pytest would silently skip.
        num_combos = math.prod(len(seq) for seq in sequences)
        if num_combos and not samples:
            raise self._exhausted(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each)",
                dropped,
                retries=bool(random_indices),
            )
        if _stats is not None:
            _stats.left_out += num_combos - len(samples)

        return samples

    # ====
    # CLI Support
    # ====

    def get_vector_by_name(self, name: str) -> Vector | ParameterSet:
        """
        Get directed vector by name (for -vn CLI argument).

        Args:
            name: Name of the directed vector

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are one

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self.directed_vectors:
            available = ", ".join(self.directed_vectors.keys())
            raise KeyError(f"No directed vector named '{name}'. " f"Available: {available}")
        return self.directed_vectors[name]

    def get_vector_by_index(self, index: int) -> Vector | ParameterSet:
        """
        Get directed vector by index (for -vi CLI argument).

        Args:
            index: Index of the directed vector (0-based)

        Returns:
            The directed vector: a Vector, or the pytest.param(...) whose values are one

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
