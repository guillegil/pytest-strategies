# parameter.py

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import itertools
import keyword
import math
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
        vector_constraints: Sequence[Callable[[Vector], object]] | None = None,
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
            vector_constraints: List of functions that validate entire parameter vectors.
                Each receives the row as a Vector (``v.addr`` or ``v[0]``), and a falsy
                result rejects it.
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
        self.vector_constraints = list(vector_constraints or [])
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

    def _validate_vector(self, vector: Vector, rejections: Counter[int] | None = None) -> bool:
        """
        Validate a vector against all constraints.

        Args:
            vector: Parameter vector to validate
            rejections: If given, the index of the first constraint that rejects the
                vector is counted in it

        Returns:
            True if all constraints pass, False otherwise
        """
        for index, constraint in enumerate(self.vector_constraints):
            if not constraint(vector):
                if rejections is not None:
                    rejections[index] += 1
                return False
        return True

    def _describe_rejections(self, rejections: Counter[int]) -> str:
        """Name the constraints that rejected vectors, with their counts, for an error."""
        if not rejections:
            return ""
        parts = []
        for index, count in sorted(rejections.items()):
            constraint = self.vector_constraints[index]
            name = getattr(constraint, "__name__", None)
            if name is None or name == "<lambda>":
                kind = "lambda" if name == "<lambda>" else type(constraint).__name__
                label = f"constraint #{index} ({kind})"
            else:
                label = f"{name} (constraint #{index})"
            parts.append(f"{label} rejected {count}")
        return " Rejected by: " + ", ".join(parts) + "."

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
        rejections: Counter[int],
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
            if self._validate_vector(row, rejections):
                return row
        return None

    def generate_vector(self) -> Vector:
        """
        Generate a single random parameter vector.

        Returns:
            A Vector of generated values, one per TestArg

        Raises:
            ValueError: If generated vector fails constraints

        Example:
            vector = param.generate_vector()  # e.g., Vector(x=5, y=3.14, mode="fast")
        """
        return self._random_row(self.vector_type)

    def _random_row(self, row_type: type[Vector]) -> Vector:
        """Draw every argument of one row, redrawing up to max_retries times (generate_vector)."""
        rejections: Counter[int] = Counter()
        width = len(self.test_args)
        row = self._build_row(row_type, [None] * width, range(width), self.max_retries, rejections)
        if row is None:
            raise ValueError(
                f"Could not generate valid vector after {self.max_retries} attempts. "
                "Check your constraints." + self._describe_rejections(rejections)
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

        The arguments after n are keyword-only.

        Returns:
            List of parameter vectors, each a Vector, or for a pytest.param(...)
            directed or test vector the ParameterSet whose values are a Vector. Empty
            when skip_reason is set.

        Raises:
            KeyError / IndexError: If filter_by_name / filter_by_index names no
                directed vector (also when skip_reason is set)
            ValueError: If n is not an int >= 0 in a mode that generates samples
            ValueError: If the vector constraints reject every generated vector

        Warns:
            PytestStrategiesWarning: If a Series combination is skipped, or with
                per_sequence_samples=True a combination gets fewer than n rows, because
                its random args did not satisfy the constraints within max_retries draws

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

        # Generate samples (for all modes except directed_only)
        if mode != "directed_only" and self.per_sequence_samples and self._sequence_indices():
            samples.extend(self._generate_per_sequence(n))
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
                # index in the product (Series values need not be hashable)
                skipped: dict[int, tuple[Any, ...]] = {}
                rejections: Counter[int] = Counter()
                for k, combo in enumerate(itertools.cycle(itertools.product(*series_seqs))):
                    if series_rows >= n:
                        break
                    vec: list[Any] = [None] * len(self.test_args)
                    # Series values skip arg.generate(), so apply the arg's validator here
                    for pos, idx in enumerate(series_indices):
                        vec[idx] = self.test_args[idx]._validate(combo[pos])
                    # Try fresh random values for the non-Series positions
                    row = self._build_row(row_type, vec, random_indices, attempts, rejections)
                    if row is not None:
                        samples.append(row)
                        series_rows += 1
                        misses = 0
                    else:
                        # Skip a combination the constraints reject and move on to the
                        # next one, unless a whole cycle in a row has produced nothing
                        misses += 1
                        if misses == num_combos:
                            raise ValueError(
                                "Could not generate valid vector: none of the "
                                f"{num_combos} Series combinations satisfied the vector "
                                f"constraints ({attempts} attempt(s) each). "
                                "Check your constraints." + self._describe_rejections(rejections)
                            )
                        # With random args the rejection may just be unlucky draws of a
                        # valid combination, so the skip must not go unnoticed
                        if random_indices:
                            skipped.setdefault(k % num_combos, combo)
                # Warn once per skipped combination, and only when no error was raised
                for combo in skipped.values():
                    values = ", ".join(
                        f"{self.test_args[idx].name}={value!r}"
                        for idx, value in zip(series_indices, combo)
                    )
                    warnings.warn(
                        f"Series combination ({values}) skipped: the vector constraints "
                        f"rejected max_retries={attempts} draws of the non-Series args. "
                        "Raise max_retries, or relax the constraints if this combination "
                        "should be tested.",
                        PytestStrategiesWarning,
                        stacklevel=2,
                    )
            else:
                for _ in range(n):
                    samples.append(self._random_row(row_type))

        return samples

    def _sequence_indices(self) -> list[int]:
        """Return the positions of the Series/RNGSequence args."""
        return [
            i
            for i, a in enumerate(self.test_args)
            if a.rng_type and isinstance(a.rng_type, SequenceLike)
        ]

    def _generate_per_sequence(self, n: int) -> list[Vector]:
        """
        Generate n random rows for every combination of the sequence args.

        Combinations follow declaration order (leftmost arg is the slowest counter), for
        RNGSequence as well as Series. The non-sequence args are drawn fresh for every
        row and redrawn up to max_retries times when the constraints reject the vector.

        Args:
            n: Number of rows per combination

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
        # Combinations cut short after redrawing their random args, with their row count
        short: list[tuple[tuple[Any, ...], int]] = []
        rejections: Counter[int] = Counter()
        for combo in itertools.product(*sequences):
            vec: list[Any] = [None] * len(self.test_args)
            # Sequence values skip arg.generate(), so apply the arg's validator here
            for idx, value in zip(sequence_indices, combo):
                vec[idx] = self.test_args[idx]._validate(value)

            rows = 0
            while rows < n:
                row = self._build_row(row_type, vec, random_indices, attempts, rejections)
                if row is None:
                    # Further rows of this combination would most likely fail too
                    if random_indices:
                        short.append((combo, rows))
                    break
                samples.append(row)
                rows += 1

        num_combos = math.prod(len(seq) for seq in sequences)
        if n and num_combos and not samples:
            raise ValueError(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each). "
                "Check your constraints." + self._describe_rejections(rejections)
            )

        for combo, rows in short:
            values = ", ".join(
                f"{self.test_args[idx].name}={value!r}"
                for idx, value in zip(sequence_indices, combo)
            )
            warnings.warn(
                f"Sequence combination ({values}) produced {rows} of {n} rows: the vector "
                f"constraints rejected max_retries={attempts} draws of the other args. "
                "Raise max_retries, or relax the constraints if this combination "
                "should be tested.",
                PytestStrategiesWarning,
                stacklevel=3,
            )

        return samples

    def generate_exhaustive(self) -> list[Vector]:
        """
        Generate all combinations of sequence arguments (Cartesian product).
        For non-sequence arguments, generate a random value for each combination.

        Returns:
            List of Vectors. Empty when skip_reason is set.

        Raises:
            ValueError: If no sequence arguments are present
            ValueError: If a sequence value fails its argument's validator
            ValueError: If the vector constraints reject every combination
        """
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
        rejections: Counter[int] = Counter()
        for combination in itertools.product(*sequences):
            # Create a mutable vector (list) to fill in
            vector: list[Any] = [None] * len(self.test_args)

            # Fill in sequence values (they skip arg.generate(), so validate them here)
            for idx, value in zip(sequence_indices, combination):
                vector[idx] = self.test_args[idx]._validate(value)

            # Fill in non-sequence values with random generation, redrawing them if the
            # constraints reject the vector. A combination that still fails (e.g. its
            # sequence values alone break a constraint) is dropped.
            row = self._build_row(row_type, vector, random_indices, attempts, rejections)
            if row is not None:
                samples.append(row)

        # No samples from a non-empty product means the constraints rejected every
        # combination. Fail like finite mode does instead of yielding an empty parameter
        # set, which pytest would silently skip.
        num_combos = math.prod(len(seq) for seq in sequences)
        if num_combos and not samples:
            raise ValueError(
                "Could not generate valid vector: none of the "
                f"{num_combos} sequence combinations satisfied the vector "
                f"constraints ({attempts} attempt(s) each). "
                "Check your constraints." + self._describe_rejections(rejections)
            )

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

    def add_constraint(self, constraint: Callable[[Vector], object]) -> None:
        """
        Add a constraint that validates entire parameter vectors.

        Args:
            constraint: Function that takes the row as a Vector and returns a truth
                value (False rejects the row)

        Example:
            # Ensure first arg < second arg
            param.add_constraint(lambda v: v.lo < v.hi)  # or v[0] < v[1]
        """
        self.vector_constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Remove all vector constraints."""
        self.vector_constraints = []

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
