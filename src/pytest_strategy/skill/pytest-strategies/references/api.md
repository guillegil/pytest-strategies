# pytest-strategies API reference

Read the section you need; SKILL.md has the overview and the common traps. The
signature blocks (the `python` blocks whose functions end in `...`) match the code;
a test of the package compares them with `inspect.signature`.

1. [Imports](#1-imports)
2. [register, strategy and export_strategies](#2-register-strategy-and-export_strategies)
3. [Factories](#3-factories)
4. [Parameter](#4-parameter)
5. [TestArg](#5-testarg)
6. [RNG types](#6-rng-types)
7. [Series, RNGSequence and exhaustive mode](#7-series-rngsequence-and-exhaustive-mode)
8. [The RNG helpers and seeding](#8-the-rng-helpers-and-seeding)
9. [Sample counts and vector modes](#9-sample-counts-and-vector-modes)
10. [Command-line and ini options](#10-command-line-and-ini-options)
11. [Test IDs and selecting rows](#11-test-ids-and-selecting-rows)
12. [Strategies files and name lookup](#12-strategies-files-and-name-lookup)
13. [The context hook, strategies_ctx and get_context](#13-the-context-hook-strategies_ctx-and-get_context)
14. [Record mode](#14-record-mode)
15. [Per-test metadata](#15-per-test-metadata)
16. [Reproducing failures](#16-reproducing-failures)
17. [Random streams and pytest-xdist](#17-random-streams-and-pytest-xdist)
18. [Errors and what to do](#18-errors-and-what-to-do)
19. [Upgrading from 3.x and 2.x](#19-upgrading-from-3x-and-2x)
20. [The skill installer](#20-the-skill-installer)

## 1. Imports

Everything public is importable from the package root:

```python
from pytest_strategy import (
    register, strategy, Strategy,           # decorators; Strategy holds the aliases
    export_strategies,                      # every registered strategy as JSON
    Parameter, TestArg,
    StrategyOptions,                        # what a factory receives as options
    Vector,                                 # the class of every generated row
    VectorInfo, VECTOR_KEY, VECTORS_KEY,    # a test item's row: item.stash[VECTOR_KEY]
    get_context,                            # a folder's pytest_strategies_context object
    RNG, RNGType,
    RNGInteger, RNGFloat, RNGBoolean, RNGChoice, RNGEnum, RNGString,
    RNGWeightedInteger, RNGWeightedFloat,
    Series, RNGSequence, SequenceLike,
    RNGValueError, PytestStrategiesWarning,
)
```

`pytest_strategy.rng` also exposes the RNG classes. `pytest_strategy.strategy` is the
`strategy` function (`from pytest_strategy.strategy import Strategy,
PytestStrategiesWarning` keeps working for old code). Use
`pytest_strategy.PytestStrategiesWarning` in `filterwarnings`.

## 2. register, strategy and export_strategies

```python
def register(name: str) -> Callable[[F], F]: ...
def strategy(
    name: str | Callable[..., Parameter], *, validate_signature: bool = True
) -> Callable[[F], F]: ...
def export_strategies(*, format: str = "json") -> str: ...
```

- `@register("name")` records the factory under `name`, scoped to the folder of the
  file that defines it (see section 12). It returns the function unchanged, with its
  type.
- `@strategy("name")` looks the name up from the test's folder upward when the test is
  collected. `@strategy(factory)` uses that factory directly; it does not need to be
  registered, and the node IDs are the same as with the name.
- `@strategy` only marks the test (a `strategy` marker holding the name or factory).
  The factory runs, and the test is parametrized, in `pytest_generate_tests`, so a
  missing name or a signature mismatch is reported at collection.
- `validate_signature=False` skips the check that the test takes every argument name.
  It is keyword-only: `@strategy("name", False)` raises `TypeError`.
- `Strategy.register is register` and `Strategy.strategy is strategy`: the old
  spellings are aliases, with no warning. `Strategy.export_strategies` is
  `export_strategies`.
- Apply `@strategy` to test functions and methods, not to classes or modules.
- `export_strategies(format="json")` returns a JSON string describing every
  registered strategy (arguments, RNG types, vectors). It loads every strategies file
  and calls each factory as at collection: the `ctx` of the factory's own folder,
  and the session's `nsamples` (10 without `--nsamples`) and `options`.
  The document has schema 1: `{"schema": 1, "kind": "strategies", "generator": {...},
  "seed": ..., "nsamples": ..., "strategies": [...]}`, one entry per registration
  sorted by name and folder, each with `name`, `origin` (`folder`, `file`,
  `qualname`, `line`), `context` (a fingerprint or null) and one of `parameter`
  (`Parameter.to_dict()`), `error` (`{"type": "RuntimeError", "message": "boom"}`)
  or `unavailable` (section 13). In `Parameter.to_dict()` (`"schema": 1`) each
  argument has `source` `"value"` with `value`, or `"rng"` with `rng`
  (`RNGType.to_dict()`: `{"type": "RNGInteger", "min": 0, "max": 255, "predicate":
  false}`); vectors are `{"name", "id", "values"}` lists and constraints
  `{"name", "enabled"}`. Values that JSON does not hold are tagged:
  `{"$float": "nan"}`, `{"$enum": "Color", "member": "RED"}`,
  `{"$repr": "b'\\x00'", "$type": "bytes"}`. Ignore keys and enum values you do not
  know: 4.x adds them within schema 1, and read an unknown `$`-tagged object like
  `$repr`.

## 3. Factories

```python
@register("name")
def factory(nsamples, ctx, rng, options):   # each one optional, any order, by name
    return Parameter(...)
```

- Called once per test that uses it, at collection, with `metafunc.config` available.
- It receives by name exactly the inputs it declares, like fixtures:
  - `nsamples`: the `--nsamples` integer, 10 when the option is not given, or the
    string `"auto"` under `--nsamples=auto`. A factory returning a `Parameter` can
    ignore it; the plugin applies the count.
  - `ctx`: the result of the `pytest_strategies_context` hook for the test's folder
    (section 13). Only factories that declare a `ctx` parameter trigger the hook.
  - `rng`: the plugin's `random.Random`; `rng is RNG.generator()` during the call.
  - `options`: this strategy's `StrategyOptions`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class StrategyOptions:
    strategy: str                                # the resolved name, as in the -v summary
    nsamples: int | Literal["auto"] = 10
    nsamples_source: str = "default"             # "--nsamples" or "default"
    mode: str = "all"                            # the --vector-mode value
    vector_name: str | None = None               # --vector-name
    vector_index: int | None = None              # --vector-index
    constraints_off: frozenset[str] = frozenset()  # names turned off in this strategy

    @property
    def filtered(self) -> bool: ...              # --vector-name or --vector-index given
```

- Any other parameter must have a default, which it keeps, and `*args`/`**kwargs`
  receive nothing. `def factory(n)` or `def factory(n, /)` fails collection with
  `has a parameter 'n', which the plugin does not provide ... Did you mean 'nsamples'?`.
  `base`, `config` and `request` are reserved, even with a default. Later 4.x
  releases add inputs only as `StrategyOptions` fields or under the reserved names.
- A `functools.partial`, a bound method, a classmethod, a staticmethod, a class, a
  callable object or a `functools.wraps` decorated function works as a factory. A
  decorator without `functools.wraps` hides the signature, so the factory is called
  with no arguments. `mock.patch` mocks must be the first parameters. An `async def`
  factory fails.
- Return a `Parameter`. Anything else fails the collection of the tests that use
  it: `Strategy 'name' must return a Parameter, got NoneType (did the factory forget
  to return?)`.
- An exception in the factory fails the collection of the tests that use it, with
  `Error calling strategy factory '<name>' (nsamples=...)`.
- Values the factory draws itself are reproducible only when drawn through `rng` or
  the plugin's generators (section 8), not through plain `random`. A factory wrapped
  in `functools.cache` runs once, for the first test that uses it, so it must not
  draw: a test run alone could get other values.

## 4. Parameter

```python
class Parameter:
    def __init__(
        self,
        *test_args: TestArg,
        directed_vectors: Mapping[str, Iterable[Any]] | None = None,
        test_vectors: Mapping[str, Iterable[Any]] | None = None,
        always_include_directed: bool = True,
        vector_constraints: (
            Mapping[str, Callable[[Vector], object]]
            | Iterable[Callable[[Vector], object]]
            | None
        ) = None,
        max_retries: int = 100,
        nsamples: int | str | None = None,
        per_sequence_samples: bool = False,
        max_exhaustive: int | None = None,
        ids: (
            Literal["names", "values"] | Callable[[VectorInfo], str | None] | None
        ) = None,
    ): ...
```

| Argument | Meaning |
| --- | --- |
| `*test_args` | the arguments, in order; names must be unique identifiers that are not keywords and do not start with `_` (`RNGValueError` otherwise) |
| `directed_vectors` | named rows, placed before the random rows: a dict of argument names to values, one value per argument in order (a tuple or list), or a namedtuple with the argument names as fields; names are non-empty strings |
| `test_vectors` | named rows used only by `--vector-mode=test`, in the same forms |
| `always_include_directed` | whether `--vector-mode=mixed` includes the directed vectors |
| `vector_constraints` | functions taking the row as a `Vector` (`v.lo` or `v[0]`), as a dict of names to functions or a list (named by each function's `__name__`, `constraint_<i>` for a lambda, a partial or a callable object); they run in order and a random row that fails one is redrawn |
| `max_retries` | redraws per row before giving up (int >= 1) |
| `nsamples` | this strategy's default count: `None`, an int >= 0 or `"auto"`; an integer `--nsamples` overrides it |
| `per_sequence_samples` | count rows per combination of the `Series`/`RNGSequence` arguments (section 7) |
| `max_exhaustive` | this strategy's limit on exhaustive combinations (section 7) |
| `ids` | this strategy's test IDs (section 11): `None` follows the `strategies_ids` ini option, `"names"` or `"values"` overrides it, and a function receives each row's `VectorInfo` and returns the ID or `None` to keep it |

```python
class Parameter:
    # Directed and test vectors
    def add_directed_vector(self, name: str, values: Iterable[Any]) -> None: ...
    def remove_directed_vector(self, name: str) -> None: ...
    def add_test_vector(self, name: str, values: Iterable[Any]) -> None: ...
    def remove_test_vector(self, name: str) -> None: ...
    def get_directed_vector(self, name: str) -> Vector: ...
    def get_test_vector(self, name: str) -> Vector: ...
    def get_vector_by_name(self, name: str) -> Vector: ...   # a directed vector
    def get_vector_by_index(self, index: int) -> Vector: ...  # a directed vector, from 0
    def list_vector_names(self) -> list[str]: ...             # the directed vectors' names

    # Constraints
    def add_constraint(
        self, fn: Callable[[Vector], object], *, name: str | None = None
    ) -> str: ...                                             # returns the name
    def remove_constraint(self, name: str) -> None: ...      # KeyError lists the names
    def clear_constraints(self) -> None: ...

    # Generation outside the plugin, and the export
    def generate_vectors(
        self,
        n: int,
        *,
        mode: str = "all",
        filter_by_name: str | None = None,
        filter_by_index: int | None = None,
        constraints_off: Iterable[str] = (),
    ) -> list[Vector]: ...
    def generate_vector(self) -> Vector: ...
    def generate_exhaustive(self, *, constraints_off: Iterable[str] = ()) -> list[Vector]: ...
    def get_arg(self, name: str) -> TestArg: ...
    def to_dict(self) -> dict[str, Any]: ...

    @property
    def directed_vectors(self) -> Mapping[str, Vector]: ...    # read-only
    @property
    def test_vectors(self) -> Mapping[str, Vector]: ...        # read-only
    @property
    def vector_constraints(self) -> Mapping[str, Callable[[Vector], object]]: ...
    @property
    def vector_type(self) -> type[Vector]: ...
    @property
    def arg_names(self) -> tuple[str, ...]: ...
    @property
    def arg_types(self) -> tuple[type, ...]: ...
    @property
    def num_args(self) -> int: ...
    @property
    def vector_names(self) -> list[str]: ...                   # the directed vectors' names
    @property
    def num_directed_vectors(self) -> int: ...
    @property
    def skip_reason(self) -> str | None: ...                   # see skip_if_empty
```

- Directed and test vectors must give one value per argument (`RNGValueError`
  otherwise, naming the vector: a wrong length, a missing or unknown dict key, with
  "did you mean"). A dict is always a named vector, so a one-argument strategy's
  vector is `{"n": -2}` (or the tuple `(-2,)`). A str, bytes or scalar vector fails
  with `For a one-argument strategy write (-2,) or {'n': -2}`, and a dict value for
  the one argument is written `{"cfg": {"a": 1}}` (in a `pytest.param`, only
  `pytest.param({"cfg": {"a": 1}}, marks=...)`). Dataclass and pydantic model
  instances are not supported as vectors yet; use a dict. `pytest.param(...,
  marks=...)` wraps any of these forms (an `id=` fails: the vector's name is its
  test ID; build other IDs with `ids=`). Vectors are not checked by predicates,
  constraints or validators.
- The `Parameter` stores each vector as a `Vector` (a `pytest.param` keeps its
  marks, with a `Vector` as its values). `directed_vectors` and `test_vectors`
  are read-only mappings; change them with the `add_*` and `remove_*` methods.
  `test_args` holds the `TestArg`s.
- `vector_constraints` is a read-only mapping of names to functions (iterating gives
  the names). A constraint name is a non-empty string without whitespace, `:`, `,`
  or `=`; two constraints with one name (closures from one helper) fail, so name
  them with a dict. Constraints run in order and the first falsy result rejects the
  draw, so a constraint may rely on the ones before it.
- `constraints_off` names constraints that call does not evaluate (names it does not
  have are ignored); the `Parameter` keeps them. A plugin run never changes a
  factory's `Parameter`.
- The generated rows are `Vector`s: tuples whose fields are the argument names
  (`v.lo`), equal to the plain tuple of their values, with the repr
  `Vector(lo=0, hi=5)`. `type(v) is tuple` is false. A misspelled field raises
  `AttributeError: Vector has no argument 'hgh'; its arguments are lo, hi`. Fields
  named `count` or `index` hide the tuple methods (use `tuple.index(v, x)`). Under
  mypy a field is `Any`, so a constraint annotated `-> bool` returns `bool(...)`.
  The `get_*_vector()` methods return a `pytest.param(...)` vector as the
  `pytest.param` whose `.values` is the Vector.
- `generate_vectors()` and the other generators called directly (outside a test)
  draw from streams keyed by the seed and 128 bits of `RNG.generator()`, so
  consecutive calls differ and `RNG.seed(s)` repeats them.

## 5. TestArg

```python
class TestArg:
    def __init__(
        self,
        name: str,
        rng_type: RNGType | None = None,
        *,
        value: Any = None,
        validator: Callable[[Any], bool] | None = None,
        description: str = "",
    ): ...
    def generate(self) -> Any: ...                       # one value, validated
    def generate_samples(self, n: int) -> list[Any]: ... # [value], or n draws
    def to_dict(self) -> dict[str, Any]: ...

    @property
    def name(self) -> str: ...
    @property
    def rng_type(self) -> RNGType | None: ...
    @property
    def description(self) -> str: ...
    @property
    def type(self) -> type: ...                          # the values' Python type
    @property
    def is_static(self) -> bool: ...                     # True with value=
```

- `name` must match the test's parameter (or a record field, section 14).
- Give `rng_type` for generated values, or `value` for a fixed one; with neither,
  `ValueError: TestArg 'x' must have a value or an rng_type`. `value`, `validator`
  and `description` are keyword-only: `TestArg("x", None, 5)` raises `TypeError`;
  write `TestArg("x", value=5)`.
- `rng_type` must be an RNG type (or an object with a `generate()` method). A bare
  function or lambda raises `TypeError`; subclass `RNGType` instead (section 6). So
  does the class itself: write `RNGBoolean()`, not `RNGBoolean`.
- `validator` runs on generated and fixed values. A value that fails it stops
  collection with `ValueError` and is not redrawn, so filter with a `predicate` on the
  RNG type instead, and use `validator` only as an assertion.
- A `TestArg` holds no directed or test vectors: give them to the `Parameter`, as
  `directed_vectors`, `test_vectors` and `always_include_directed` (section 19 lists
  the 3.x options that are gone).

## 6. RNG types

All draw from the generator the plugin gives them, so their values follow the seed.
Each checks its arguments when built and raises `RNGValueError` (a `ValueError`).

```python
class RNGInteger(RNGType[int]):
    def __init__(self, min: int | None = None, max: int | None = None,
                 predicate: Callable[[int], bool] | None = None): ...
class RNGFloat(RNGType[float]):
    def __init__(self, min: float | None = None, max: float | None = None,
                 predicate: Callable[[float], bool] | None = None): ...
class RNGBoolean(RNGType[bool]):
    def __init__(self, true_probability: float = 0.5): ...
class RNGChoice(RNGType[T]):
    def __init__(self, choices: list[T]): ...
class RNGEnum(RNGType[E]):
    def __init__(self, enum_class: type[E], weights: dict[E, float] | None = None,
                 predicate: Callable[[E], bool] | None = None): ...
class RNGString(RNGType[str]):
    def __init__(self, length: int | None = None, min_length: int = 1,
                 max_length: int = 20, charset: str = "abcdefghijklmnopqrstuvwxyz"): ...
class RNGWeightedInteger(RNGType[int]):
    def __init__(self, ranges: dict[tuple[int, int], float],
                 predicate: Callable[[int], bool] | None = None): ...
class RNGWeightedFloat(RNGType[float]):
    def __init__(self, ranges: dict[tuple[float, float], float],
                 predicate: Callable[[float], bool] | None = None): ...
class Series(SequenceLike[T]):
    def __init__(self, sequence: Sequence[T], predicate: Callable[[T], bool] | None = None,
                 *, skip_if_empty: str | None = None): ...
class RNGSequence(SequenceLike[T]):
    def __init__(self, sequence: Sequence[T], predicate: Callable[[T], bool] | None = None,
                 *, skip_if_empty: str | None = None): ...
```

| Type | Rejected at construction |
| --- | --- |
| `RNGInteger` | `min > max`, also when only one bound is given (the bounds default to `-2**31` and `2**31 - 1`) |
| `RNGFloat` | `min > max` (the bounds default to 0.0 and 1.0, so `RNGFloat(min=5.0)` fails), infinite or NaN bounds |
| `RNGChoice` | an empty list |
| `RNGEnum` | not an Enum class, no members, weights that are empty, negative, non-finite, all zero or not members, a predicate no member satisfies |
| `RNGString` | negative lengths, `min_length > max_length`, an empty charset with a length above 0 |
| `RNGWeightedInteger` | bad weights, a range that is not a `(lo, hi)` tuple with `lo <= hi` |
| `RNGWeightedFloat` | as above, plus infinite or NaN bounds |
| `Series`, `RNGSequence` | see section 7 |

- Bounds are inclusive.
- `predicate` filters draws. For numbers a draw that fails is retried (up to 100
  times, then `RNGValueError`); `RNGEnum` and the sequences filter their members up
  front, so a draw never fails while a valid member exists.
- `RNGEnum` weights do not need to sum to 1. With weights, only the weighted members
  are drawn. With a predicate, the accepted members keep their relative weights.
- `RNGWeighted*` choose a range by weight, then a value in it.
- Custom types subclass `RNGType` and implement `generate()` and `python_type`
  (the base class raises `NotImplementedError`):

```python
class RNGType(Generic[T]):
    def generate(self) -> T: ...             # draw one value
    @property
    def python_type(self) -> type[T]: ...    # the values' type
    def to_dict(self) -> dict[str, Any]: ...
```

- `generate()` draws from `RNG.generator()` or the `RNG.*` helpers called inside it,
  never from a generator kept from earlier (such as the factory's `rng`), and keeps
  no state between calls (a counter makes row k depend on the rows before it). Each
  argument of a random row then draws from a stream of its own.
- `to_dict()` (for the export) lists a custom type's public attributes as
  `{"type": "Walk", "attributes": {...}}`; override it to choose the fields.

## 7. Series, RNGSequence and exhaustive mode

| | finite `--nsamples=K` | `--nsamples=auto` |
| --- | --- | --- |
| `Series` | cycles through the combinations in order (K >= len), or the first K (K < len) | every combination of the `Series`/`RNGSequence` arguments, in declaration order; leftmost argument is the slowest counter |
| `RNGSequence` | K random picks, like `RNGChoice` | a random permutation, each value once |

- Arguments that are not sequences get a fresh random value for each combination.
- The sequence is copied into a list. A `set` or `frozenset` is rejected because its
  order is not reproducible; pass `sorted(...)` or a list. `None` is rejected.
- `predicate` removes values up front.
- An empty sequence (or one the predicate empties) raises `RNGValueError`, unless
  `skip_if_empty="<reason>"` is given (keyword-only, a non-empty string). Then every
  test using the strategy becomes one skipped row with ID `skipped` and that reason
  (`pytest -rs` shows it), in every vector mode and under `auto`. A test with other
  parametrization is skipped once per combination of it. The test's signature (or
  record) is still checked. A reason passed positionally lands in `predicate` and
  raises "did you mean skip_if_empty=...?".
- **Constraints with `Series` in finite mode:** a combination the constraints reject is
  skipped with a `PytestStrategiesWarning` (after `max_retries` redraws of its random
  arguments), and the cycle continues. Collection fails only when a whole cycle yields
  nothing.
- **`per_sequence_samples=True`:** a finite K counts per combination of the sequence
  arguments, walked in declaration order for both types. Two devices and K=10 give 20
  rows (`device=devA-rand-0` to `device=devB-rand-9`); several sequence arguments
  multiply. Directed vectors still come first. A combination whose random arguments
  keep failing the constraints gets fewer rows and a warning; collection fails only
  when no combination yields a row. Ignored under `auto`, and a `Parameter` without
  sequence arguments behaves as if it were `False`.
- **Under `auto`:** combinations the constraints reject are dropped (after
  `max_retries` redraws of the random arguments); if all are dropped, collection
  fails. Directed vectors are added per `--vector-mode`. A strategy with no sequence
  argument falls back to its own `nsamples` or 10.
- **Size guard:** before generating exhaustive rows (the combinations under `auto`,
  or combinations x K with `per_sequence_samples=True`), the plugin computes their
  count without building them. Above the limit (100,000 rows by default) collection
  fails with the strategy name, the count and how to raise the limit:
  `Parameter(max_exhaustive=...)` for one strategy, or the `strategies_max_exhaustive`
  ini option for the project.

## 8. The RNG helpers and seeding

```python
class RNG:
    @staticmethod
    def get_seed() -> int: ...                      # the run's seed
    @staticmethod
    def generator() -> random.Random: ...           # what the RNG types draw from
    @staticmethod
    def integer(min: int = -2147483648, max: int = 2147483647,
                predicate: Callable[[int], bool] | None = None) -> int: ...
    @staticmethod
    def float(min: float = 0.0, max: float = 1.0,
              predicate: Callable[[float], bool] | None = None) -> float: ...
    @staticmethod
    def boolean(true_probability: float = 0.5) -> bool: ...
    @staticmethod
    def choice(items: list[T]) -> T: ...
    @staticmethod
    def string(length: int | None = None, min_length: int = 1, max_length: int = 20,
               charset: str = "abcdefghijklmnopqrstuvwxyz") -> str: ...
    @staticmethod
    def winteger(ranges: dict[tuple[int, int], float],
                 predicate: Callable[[int], bool] | None = None) -> int: ...
    @staticmethod
    def wfloat(ranges: dict[tuple[float, float], float],
               predicate: Callable[[float], bool] | None = None) -> float: ...
    @staticmethod
    def seed(seed: int | None = None) -> None: ...  # restarts the current stream
    @staticmethod
    def refresh_seed(key: str | int | None = None) -> None: ...  # a stream of its own
```

- The helpers draw from a generator the plugin owns, never from the global `random`.
  In a factory, a strategies file while the plugin imports it, a test module's top
  level, a fixture's setup and each phase of a test, that generator follows the
  seed on a stream of its own: a test body's `RNG.integer()` is the same alone, in
  the suite and under xdist, and an `RNG.seed()` call changes only the rest of its
  stream. `RNG.refresh_seed(key=...)` starts a stream derived from the seed and the
  key, whatever ran before it.
- Draws when a `conftest.py` is imported do not follow the seed, nor do those of a
  strategies file that a `conftest.py` imports at its top. A helper module that test
  modules or strategy files import draws on the stream of the first module that
  imports it, so its values change when one file runs alone. Draw in a fixture, the
  context hook or a strategies file instead.
- The plugin never calls `random.seed()`. Plain `random` calls (in factories,
  strategies files, conftest or test bodies) are not reproduced by `--rng-seed`.
  To seed plain `random` from the run's seed, do it yourself, for example per test
  with an autouse fixture: `random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")`.

## 9. Sample counts and vector modes

Random rows per strategy and test:

| `--nsamples` | `Parameter(nsamples=)` | Random rows |
| --- | --- | --- |
| not given | unset | 10 |
| not given | 25 | 25 |
| `5` | 25 | 5 (the command line wins) |
| `auto` | any | every `Series`/`RNGSequence` combination; without sequence arguments, the strategy's own int count or 10 |
| not given | `"auto"` | as `--nsamples=auto` |

With `per_sequence_samples=True` a finite count applies to each combination. A row
keeps its values when the count grows: `--nsamples=50` gives the rows of
`--nsamples=10`, with the same values, and 40 more.

`--vector-mode`:

| Mode | Directed vectors | Random rows | Test vectors |
| --- | --- | --- | --- |
| `all` (default) | yes, first | yes | no |
| `random_only` | no | yes | no |
| `directed_only` | yes | no | no |
| `mixed` | if `always_include_directed` (default `True`) | yes | no |
| `test` | no | no | yes |

`--vector-name=NAME` and `--vector-index=I` (0-based, in definition order) take
precedence over the mode and `--nsamples`: each strategy yields only that directed
vector. Strategies without it give an empty parameter set, so their tests are skipped
("got empty parameter set"); so do `directed_only` and `test` for a strategy without
such vectors. When no strategy in the run has the named vector, pytest stops with a
usage error listing each strategy's directed vectors.

## 10. Command-line and ini options

| Option | Meaning |
| --- | --- |
| `--rng-seed=S` | integer seed; without it a new seed is chosen each run, except that `--lf` and `--sw` reuse the failed run's (section 16) |
| `--nsamples=N` or `--nsamples=auto` | random rows per strategy (int >= 0), or enumerate sequences; anything else is a usage error |
| `--vector-mode=MODE` | `all`, `random_only`, `directed_only`, `mixed`, `test` |
| `--vector-name=NAME` | only the directed vector named NAME |
| `--vector-index=I` | only the directed vector at index I |
| `--strategy-constraint-off=[S:]NAME[,...]` | turn the constraint NAME off for this run, in every strategy or only in strategy S (its resolved name, as in the `-v` summary); repeatable. An item that matches no constraint of a resolved strategy is a usage error in a whole-suite run (with "did you mean" and the constraints by strategy), and a red line in a run narrowed by paths, node IDs, `--lf`, `--sw`, `--ignore` or a start below the rootdir, or with a module that was skipped or failed to collect |
| `--list-strategies` | load every strategies file, list the registered names (without calling any factory) and exit |

| ini option | Meaning |
| --- | --- |
| `strategies_max_exhaustive` | the project's limit on exhaustive combinations (default 100000) |
| `strategies_ids` | `names` (default) or `values` (the 3.0 IDs built from the values); anything else is a usage error (section 11) |

`-o NAME=VALUE` sets an ini option for one run (`-o strategies_ids=values`). In a
native TOML configuration (pytest 9's `pytest.toml`, or `[tool.pytest]`), write the
values as strings: `strategies_max_exhaustive = "500000"`. New command-line options
start with `--strategy-` and new ini options with `strategies_`.

Output:

- The header shows `pytest-strategies: RNG seed = S` (hidden by `-q` and
  `--no-header`), then `pytest-strategies: constraints off: aligned,
  dma_burst:no_4k_cross` (the items as given) when constraints are turned off, and
  the reused-seed line of `--lf` (section 16).
- After the collection (also with `-q`): `pytest-strategies: context 976bcfdf` when
  a context was computed (section 13).
- A run with failures prints the reproduce line and the failed rows (section 16).
- `-v` adds a "Strategy Summary" with, per strategy, the number of tests, the
  counts of directed and random rows (and of test, exhaustive and skipped rows when
  there are some), where `nsamples` came from, `rejected: aligned=412,
  no_4k_cross=37` and `off: ...` for constraints, and `left out: N combinations`
  under `auto`; then a "Contexts" block with each context's fingerprint and number
  of tests.
- Warnings raised while generating a strategy's rows are reported at the test,
  prefixed with `Strategy '<name>' (<test>): `.

## 11. Test IDs and selecting rows

Each row's test ID names it; the same for every seed in the default `names` format:

| Row | ID |
| --- | --- |
| directed vector `zeros` | `directed-zeros` |
| test vector `max` | `test-max` |
| random row 3, nothing enumerated | `rand-3` |
| random row j of an enumerated combination | `ch=2-rand-1`, `ch=0-dev=a-rand-1` |
| a row of `--nsamples=auto` | `ch=2`, `ch=0-dev=b` |
| the row of an empty `skip_if_empty` sequence | `skipped` |

- An argument's value is in the ID exactly when the row enumerates it: the `Series`
  arguments in finite mode, the `Series` and `RNGSequence` arguments with
  `per_sequence_samples=True`, and every sequence argument under `auto`. Several
  are joined with `-` in declaration order.
- A label is `ARG=TEXT`: `str(v)` for a bool or None, the member name of an Enum,
  `repr()` of an int or float, `__name__` of a class or function, and a string as it
  is when it is non-empty, at most 40 characters, printable and without whitespace,
  `=`, `~`, `[` or `]`. A value listed twice gets `~1` the second time (`ch=1~1`). A
  value without text (a tuple, bytes, another string), and two values with the same
  text (`1` and `"1"`), are labeled by position: `cfg0`, `cfg1`.
- Under `auto` every seed gives the same set of IDs as long as the constraints drop
  no combination; only the order of `RNGSequence` rows changes.
- A test with stacked `@strategy` or `@pytest.mark.parametrize` decorators gets the
  parts joined with `-`, the decorator closest to the `def` first:
  `test_two[directed-zeros-fast]`.
- **`-k`:** `-k zeros` selects `directed-zeros`, `-k directed` every directed row,
  `-k "not rand"` no random row. `-k "rand-3"` also matches `rand-30`: use
  `-k "test_w[rand-3]"` or the node ID. `-k` cannot contain `=`, so select a
  sequence value's rows by node ID (`pytest "tests/test_esm.py::test_esm[ch=2-rand-1]"`).
  A name in it can hold only letters, digits and `_ - . : / + \ [ ]` (not
  whitespace, parentheses, commas or quotes), and a non-ASCII name needs
  pytest's escaped form (`-k 'caf\xe9'`): give vectors identifier-like names.
- **A node ID** with the run's seed reruns that row with the values it had, for any
  `--nsamples` that still generates it (`rand-12` with 10 rows gives pytest's "not
  found"; give the same `--nsamples`).
- **`strategies_ids = values`** (ini option, or `-o strategies_ids=values`) gives the
  3.0 IDs built from the values (`addr=0,len=1`, a record's `init` fields), which
  change with the seed.
- **`Parameter(ids=...)`:** `"names"` or `"values"` for one strategy, or a function
  called once per row at collection (not for `skipped`) with the row's `VectorInfo`,
  whose `id` is the row's ID in the effective format. It returns the ID, or `None` to
  keep that one. Duplicates get pytest's suffixes (`odd0`, `odd1`), and
  `item.stash[VECTOR_KEY].id` is the final ID. Anything but a non-empty str or
  None, or an exception, fails collection: `In test_write: Strategy 'burst': ids=
  returned 42 for row rand-3; return a str or None`. Build IDs from what does not
  depend on the seed (`kind`, `name`, `index`, the enumerated values), so that `-k`
  and `--deselect` keep working.
- IDs never change the generated rows. The `RNG` draws of test phases and
  function-scoped fixtures are keyed by the node ID, so they follow it.

## 12. Strategies files and name lookup

**File names:** `strategies.py`, `strategy.py`, `*_strategies.py`, `*_strategy.py`,
`test_strategies.py`. A file is loaded only if its text contains `@register("`,
`@Strategy.register(` or `@<module>.register("` (for example `@ps.register("` after
`import pytest_strategy as ps`); outside `@Strategy.register(`, the name must be
written as a string literal. A `conftest.py` or a test module may also register
strategies.

**Lazy loading:** when pytest collects a test module, the plugin first loads the
strategies files in the module's folder and then in each parent folder up to the
rootdir (or the `testpaths` entry that contains it), closest first, each file at most
once per session. Files in unrelated folders are not imported by that run, unless a
test asks for a name that no folder on its path registers: then every strategies file
is loaded to find it.

**Import:** files are imported with pytest's importer, using the session's
`--import-mode` and rootdir. The module gets the name a test importing it would get,
so `from .strategies import default` in a test returns the module the plugin loaded
(the same Enum classes, no second execution), and relative imports inside a
strategies file work in package folders. `test_strategies.py` keeps assertion
rewriting.

**Import-time draws:** each strategies file is imported with its own random stream,
derived from the seed and the file's path relative to the rootdir (`../shared/...`
outside it), so values drawn at import time through the plugin's generators do not
depend on which folders load first.

**Lookup:**

- `register()` records the folder of the file that defines the factory.
- Lookup starts at the test module's folder and walks up to the `testpaths` entry
  (or command-line folder) that contains it, or to the rootdir without `testpaths`;
  the first folder that has the name wins. Strategies files above that point are not
  loaded (register shared strategies in the rootdir `conftest.py` instead).
- A name that no folder on that path has, but that is registered only once elsewhere,
  is found too. When several folders off the path register it, a registration from
  outside the rootdir (an installed package, a shared plugin) wins; otherwise the
  lookup fails with an error that names them.
- Two different factories with the same name in one folder are a usage error (exit
  code 4, or 2 under pytest-xdist) naming both files. The same function registered
  again is silent.
- `@strategy(factory)` skips the lookup entirely.

**Load failures:** a strategies file that raises while importing is reported and does
not stop the run; a file that calls `pytest.skip(..., allow_module_level=True)` or
`pytest.importorskip()` at module level is skipped. "Strategy 'x' not found" lists the
names visible from the test, the closest match ("did you mean"), and the files that
failed to load or were skipped.

## 13. The context hook, strategies_ctx and get_context

```python
def pytest_strategies_context(config: pytest.Config) -> Any: ...   # the hookspec
def get_context(config: pytest.Config, path: str | os.PathLike[str]) -> Any: ...
```

```python
# conftest.py at the rootdir, or in the folder it is for (or a plugin)
import pytest

@pytest.hookimpl(optionalhook=True)
def pytest_strategies_context(config):
    return Testbench.parse_config(config.getoption("--tb-config"))
```

- A `firstresult` hook; `config` is the pytest config. Return any object.
- `optionalhook=True` keeps the `conftest.py` usable when the plugin is not loaded
  (without it pytest stops with "unknown hook").
- Each test gets the context of its own folder. The plugin asks the
  implementations the folder sees in this order: `tryfirst` ones, the
  `conftest.py` files from the test's folder upward, the other plugins (last
  registered first), `trylast` ones; the first that is not `None` answers. A
  `wrapper=True` implementation can change the answer by returning a new object
  (`{**ctx, "extra": 1}`), or return the object it receives as it is (one context
  with the folders without it); changing that object fails its folders, since
  other folders get it too. Its code before `yield` runs after
  the implementations it wraps. A factory registered in another folder gets the
  test's folder's context.
- Each implementation that is not a wrapper is called at most once per session,
  the first time a factory with a `ctx` parameter (or `strategies_ctx` or
  `get_context()`) needs it; the result (or the exception) is reused for every
  later factory, and folders that end at the same implementation share one
  object. A wrapper runs once per answering implementation and set of wrappers.
- When no implementation returns a value, `ctx` keeps its default (or a value bound
  with `functools.partial`), else `None`.
- If an implementation raises, each test that uses a factory with `ctx` in a
  folder that asks it fails collection with
  `Strategy factory '<name>' has a 'ctx' parameter, but the pytest_strategies_context
  hook raised <error>`. `pytest.fail()` is reported as is, and
  `pytest.skip(..., allow_module_level=True)` skips those tests.
- When a factory fails with `ctx` None while a `conftest.py` in another folder
  implements the hook, the error says where: `ctx is None for tests/b: no
  pytest_strategies_context implementation in this folder or above answered
  (implemented in tests/a/conftest.py; move it to a common parent conftest)`.
- Random draws in the hook come from a stream of their own, derived from the seed and
  started anew for each implementation, and do not shift any test's rows.
- `export_strategies()` gives a factory the context of its own file's folder (the
  rootdir's for a file outside the rootdir or in an installed package). A `ctx`
  factory in a folder whose `conftest.py` the run did not load (`pytest tests/a`
  leaves `tests/b/conftest.py` out) is not called: its entry is
  `{"unavailable": "tests/b/conftest.py was not loaded in this session"}`.

**The fingerprint.** After the collection (also with `-q` and `--collect-only`) the
plugin prints a fingerprint of each context it computed, the first 8 hex characters
of a SHA-256 taken when the hook returned the object: `pytest-strategies: context
976bcfdf`, or with several, `pytest-strategies: contexts conftest.py 976bcfdf,
tests/tb_a/conftest.py b1e1b237`. Nothing is printed when no context was computed or
every one is `None`; under pytest-xdist the controller prints the workers' line at
the end of the run.

- The encoding does not depend on `PYTHONHASHSEED` (sets are sorted), the checkout
  folder (rootdir paths are relative) or `--import-mode` (types by qualified name).
  A pydantic v2 model is its `model_dump()`: `Field(exclude=True)` fields are left
  out and a `SecretStr` stays masked. Dataclasses, attrs classes and NamedTuples
  count field by field, `SimpleNamespace` and `argparse.Namespace` by attribute.
  Other objects are their repr without memory addresses (` at 0x...` inside
  `<...>`, a mock's `id='...'`), with the sets it shows as `{...}` sorted; one
  with the default repr counts by its type alone, shown as `(partial: Plain)`. A
  repr that shows a set another way (`",".join(tags)`) should sort it. One that
  cannot be encoded is `unavailable`. Keep volatile values (temporary paths, times)
  out of the context, or exclude them.
- The reproduce line of a failed run ends with the contexts the failed tests'
  factories received (`(context 976bcfdf)`, or `(contexts conftest.py 976bcfdf,
  tests/tb_a/conftest.py b1e1b237)`; a failed setup or call counts, an error in
  teardown alone does not), `-v` lists each context with its number of
  tests, and `item.stash[VECTOR_KEY].context` is the fingerprint for the rows of a
  factory that received `ctx` (else `None`). A factory or test that changes the
  object changes no fingerprint.
- Under pytest-xdist every worker calls the hook; it must return the same
  configuration in each. A run whose workers computed different fingerprints for one
  context fails (section 17).

**The same object in fixtures.** Use the hook for data the rows depend on; the live
objects (a testbench connection) stay fixtures, built on the same object:

```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def tb(strategies_ctx):                  # the object the factories received
    return Testbench(strategies_ctx)

# tests/tb_a/conftest.py, a folder with its own pytest_strategies_context
from pytest_strategy import get_context

@pytest.fixture(scope="session")
def tb_a(request):
    return Testbench(get_context(request.config, __file__))
```

- `strategies_ctx` is a session-scoped fixture of the plugin: the object the
  factories of the tests that use it received, computed if no factory needed it
  yet. Those tests must share one context. When they are in folders whose contexts
  come from different implementations, each fails with `strategies_ctx is a session
  fixture, but the tests that use it have different contexts (conftest.py: ...;
  tests/tb_a/conftest.py: ...). In a folder with its own pytest_strategies_context,
  use pytest_strategy.get_context(request.config, __file__) in that folder's
  conftest.py fixtures.` Deselecting one folder's tests also makes the run pass.
  The tests of a folder whose `conftest.py` defines its own `strategies_ctx` (one
  that does not request the plugin's) do not count. When no test requests it, a
  `request.getfixturevalue("strategies_ctx")` counts every test of the run. When
  some do, a test that asks for it that way gets their context, and fails with the
  message after its setup or its body when its own folder's context is another
  one. So does a test that gets the value (or error) a fixture cached when its
  setup asked for it that way, whichever test that setup ran for; a fixture that
  asks only for some tests leaves the others alone. A raised error keeps its
  traceback, with the message in a `pytest-strategies` report section below it.
- `get_context(config, path)` returns the context of the folder of `path` (a file
  or a folder), the object a test there gets. A folder whose `conftest.py` pytest
  did not load (no test there collected) gets the nearest loaded one's above. Call
  it from fixtures or hooks; in `pytest_configure` it sees only the `conftest.py`
  files loaded so far. A config of no running session raises `RuntimeError`.
- Both raise the implementation's exception as it is: a `pytest.skip` in the hook
  skips the tests that use them.

## 14. Record mode

```python
@dataclass
class Point:
    x: int
    y: int

@strategy("points")          # a strategy with TestArg("x", ...) and TestArg("y", ...)
def test_scaled(point: Point, scale):   # scale is a fixture
    ...
```

A test receives the row as one record when (1) neither the test nor any fixture it
uses asks for one of the strategy's argument names, and (2) exactly one test
parameter is annotated with a record type whose fields are exactly those names.
Otherwise the strategy passes its arguments by name, one test parameter each.

- "Asks for" covers the test's parameters, `usefixtures`, autouse fixtures and what
  they ask for in turn. A fixture that takes `x` and `y` receives them;
  `validate_signature` does not change the choice, and the named-mode check counts
  the names a fixture asks for.
- A test parameter is one that pytest fills: no default, not `*args`/`**kwargs`,
  not `self`, `cls` or a built-in fixture.
- Record types: dataclasses (pydantic dataclasses too), after stripping
  `Annotated[...]` and generic arguments. Fields with `init=False` are not counted
  and not shown in IDs, `kw_only` fields work, and one-argument strategies work.
- NamedTuple, TypedDict and pydantic models are recognized (fields: `_fields`, the
  required and optional keys, the `model_fields` names) and fail with "not
  supported yet". Unions, pydantic v1 models, attrs classes, `Vector` and
  unresolvable annotations are not record types.
- Two matching parameters fail and are both named. A single record parameter whose
  fields do not match fails with the missing and extra names.

String annotations (`from __future__ import annotations`) work if the dataclass is
defined at module level. The IDs are those of section 11 (`rand-3`); under
`strategies_ids = values` they list the fields, `x=1,y=2`.

## 15. Per-test metadata

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class VectorInfo:
    strategy: str                     # the resolved name, as in the -v summary
    origin: str | None                # the factory, "tests/dma/strategies.py:12"
    kind: str                         # "directed", "test", "random", "exhaustive", "skipped"
    name: str | None                  # the directed or test vector's name
    index: int | None                 # vector position, or the row's number
    enumerated: tuple[str, ...]       # the arguments whose values are in the ID
    values: Vector                    # the row; Nones for "skipped"
    id: str                           # this strategy's part of the ID, final
    seed: int                         # the run's seed
    context: str | None               # the context fingerprint, when ctx was received
    constraints_off: tuple[str, ...]  # constraints turned off in this strategy
    streams: int = 1                  # the version of the random streams
    def to_dict(self) -> dict[str, Any]: ...   # {"schema": 1, ...}, export encoding
```

- `item.stash[VECTOR_KEY]` holds the `VectorInfo` of a strategy row's item, and
  `item.stash[VECTORS_KEY]` the tuple of every strategy's, in node-ID order, for a
  test with stacked `@strategy` decorators (`VECTOR_KEY` holds the first). Items
  without a strategy, and pytest's item for an empty parameter set, have none: use
  `item.stash.get(VECTOR_KEY, None)` (in a fixture, `request.node.stash`).
- The plugin stores it while collecting, before any `pytest_collection_modifyitems`
  hook runs, so hooks, fixtures and reports can read it.
- `index` is the vector's position in `directed_vectors` (what `--vector-index`
  takes) or `test_vectors`, the row's number within its combination for a random
  row, its position among the combinations for an exhaustive row, and `None` for
  `skipped`.
- The infos also travel as the argument of a `strategy` mark on each row, which adds
  no `-k` keyword. `item.iter_markers("strategy")` yields the test's own `@strategy`
  mark first and the row's after it, and `item.get_closest_marker("strategy")`
  returns the test's own: read the stash instead.
- Read it; do not build it. Fields that later releases add come with defaults.

## 16. Reproducing failures

- A run with failures prints, after the tracebacks and also under `-q`,
  `pytest-strategies: reproduce with --rng-seed=S`, followed by `(context 976bcfdf)`
  when the failed tests' factories received a context (section 13). A passing run
  does not print it.
- Then `pytest-strategies: failed rows:` lists, for each strategy row whose setup
  or call failed, the command that reruns it with the same values, run from the
  same folder, and what the row is (`  pytest 'tests/t.py::test_w[rand-3]'
  --rng-seed=S  # burst random 3`): at most 10 below `-v` (`... and N more`),
  none under `-qq`. The command adds the run's `--nsamples`, `--vector-mode`,
  `--vector-name`, `--vector-index`, `-o strategies_*`, `-c`, `--rootdir`, and
  the constraints turned off in the row's strategies (`STRATEGY:NAME`).
- Each failed row also gets a `pytest-strategies` section under its traceback:

  ```text
  ------------------------------ pytest-strategies -------------------------------
  strategy  burst (tests/dma/strategies.py:12)
  vector    rand-3 (random row 3)
  values    addr=4096
            len=17
  seed      1763926297314361000
  context   3f2a9c1e
  rerun     pytest 'tests/dma/test_write.py::test_write[rand-3]' --rng-seed=1763926297314361000
  ```

  `vector` reads `directed-zeros (directed vector 'zeros', #0)`, `test-max (test
  vector 'max', #1)` or `ch=2 (exhaustive row 5)` for the other kinds; `values` are
  reprs, one argument per line, cut at 4,000 characters below `-vv`; `context`
  appears only when the factory received `ctx`. A test with stacked `@strategy`
  decorators gets one block per strategy and one `rerun` line. A failure reported
  as plain text (an XPASS of a strict xfail) gets no section.
- A test file outside the rootdir (`-c ci/pytest.ini` with `tests/`) gets a command
  that starts from the run's paths (with its `--ignore` and `--ignore-glob`) and
  selects the row with `-k`
  (`pytest . --rng-seed=S -c ci/pytest.ini -k 'test_write[rand-3]'`),
  since its node ID depends on them; when no `-k` expression selects only that
  row, a `note` line says to pass a `--rootdir` that contains the tests.
- What shapes the context (an environment variable, an option of your own) is not in
  the command: compare the `context` lines. Compare the `rootdir:` lines too.
- With `--junitxml`, each failed row's failure text ends with that section, and
  the test suite gets the properties `pytest_strategies.seed` and
  `pytest_strategies.failed.<i>` (the failed rows' commands, from 0). With
  `junit_family = xunit1` or `legacy`, a failed row's test case also gets
  `pytest_strategies.strategy`, `.kind`, `.name`, `.index`, `.id`,
  `.value.<argument>`, `.seed`, `.context`, `.constraints_off` and `.command` (run
  from the rootdir), as `pytest_strategies.<i>.*` for stacked strategies; the
  default `xunit2` gets none per test case.
- `--lf`, `--sw` and `--sw-skip` without `--rng-seed` reuse the seed of the newest
  failed row they rerun among those the run collects (the paths and node IDs
  given, else the testpaths or the current folder; not `-k` or `-m`), recorded in
  pytest's cache under `pytest-strategies/failed-seeds` with the options of its
  rerun command, so the rows fail with the same values. The header adds
  `pytest-strategies: seed reused from the failed run for --lf (--rng-seed
  overrides)`, ending with `; recorded with --nsamples=13` when the recorded
  options differ (they are not applied). Failed rows recorded under another seed
  are deselected (pytest keeps them in its last-failed set) and the run ends with
  `pytest-strategies: deselected 2 failed rows recorded under another seed; run
  them with:` and `  pytest --lf --rng-seed=S1 tests/a/test_dma.py  # 2 rows`
  (their files, or their node IDs when a file holds other failed tests). A row
  leaves the record when it passes under its seed and options. `--rng-seed` wins;
  `--ff`, `--nf` and `--sw-reset` draw a new seed.
- Once a failure is understood, keep its values as a directed vector
  (`"bug_1234": {"addr": 4096, "len": 17}`): it runs under every seed, and
  `-k bug_1234` or `--vector-name=bug_1234` runs it alone.

## 17. Random streams and pytest-xdist

- Each argument of a random row draws from its own `random.Random`, keyed by the
  seed, the strategy name, the test's node ID without its parameters (its path
  relative to the rootdir, its class and its name), the row and the argument's name.
  A test gets the same rows (and node IDs) whether you run the whole suite, one file
  or one test, in any order and with any `--import-mode`. Two tests sharing a
  strategy get different random rows. A row keeps its values with more rows, when
  its node ID runs alone and when another argument is added or changed; a
  constraint redraws only the rows it rejects, so turning one off keeps the rows it
  never rejected. The derivation is versioned (`VectorInfo.streams`, now 1) and
  changes only in a major release.
- Renaming an argument changes its values; renaming the strategy, the test, its
  class or file, or changing the rootdir changes every row. Adding or removing an
  enumerated argument, or changing whether an argument is enumerated, changes every
  row. For `Series` and `per_sequence_samples`, which rows exist depends on the count
  and the sequence lengths. A new Python minor version may change what `randint`,
  `choice` and `sample` return for a seed.
- A constraint that draws (`RNG.integer()`) and an RNG type that draws from a
  generator kept from the factory draw outside those streams: their values change
  when other rows or tests change. Each test whose rows do so gets one
  `PytestStrategiesWarning` naming the strategy and the test. Move the draw into an
  RNG type's `generate()`, and keep constraints to reading the row.
- Keep the same seed, the same rootdir and the same plugin version. The rootdir is
  the directory of the ini file (`pytest.ini`, or `pyproject.toml` with
  `[tool.pytest.ini_options]`); without one it depends on the folder pytest is run
  from, so compare the `rootdir:` line of both runs. Values for a seed may differ
  between major versions (4.0.0 changed the random rows).
- pytest-xdist works with or without `--rng-seed`: the controller sends its seed to
  the workers. Each worker sends back the fingerprint of each context it computed
  and a digest of each strategy's node IDs and values (in the fingerprint's
  encoding, so sets are sorted); when two workers differ,
  the run fails with exit code 4 (when it would have passed or collected nothing)
  and prints, before the reproduce line:

  ```text
  pytest-strategies: the xdist workers generated different vectors:
    context tests/tb_a/conftest.py: gw0 1a2b3c4d, gw1 9f8e7d6c
    values of strategy dma_burst: gw0 5e6f7a8b, gw1 0c1d2e3f
  ```

  Names in the test IDs keep xdist from noticing different values itself. The
  causes: a factory drawing from Python's global `random` (use `rng`), a list
  built from a set of strings (sort it: its order follows `PYTHONHASHSEED`), a
  context holding temporary paths, process IDs or times. A context or strategy
  only one worker computed is not compared, nor is a folder's context that a
  wrapper built from an object a test had changed. When the values change the IDs (a
  `Series` from the context), xdist's "Different tests were collected" comes
  first.

## 18. Errors and what to do

| Message | Cause and fix |
| --- | --- |
| `Strategy 'x' not found` | Not visible from the test's folder. Check the spelling (see "did you mean"), move the strategy to a parent folder, pass the factory with `@strategy(factory)`, or fix a strategies file that failed to load (listed). |
| usage error naming two files for one name | Two factories with one name in the same folder. Rename one, or move it to its own folder. |
| `Could not generate random row K after max_retries=N draws` (or `Could not generate valid vector` for combinations) | The constraints rejected every draw; the message counts the rejections by the name of the first failing constraint and shows the first row each rejected. Relax the constraint, narrow the ranges, raise `Parameter(max_retries=)`, or turn one off for a run with the `--strategy-constraint-off` item it names. |
| `Constraint 'x' raised ...` | A constraint raised on the row shown (its frame follows). Fix the constraint; one before it in the mapping can guard it (`{"nonzero": ..., "ratio": ...}`), and `(constraint 'nonzero' before it is turned off by --strategy-constraint-off)` says that guard was turned off. |
| `--strategy-constraint-off=x matched no constraint` | No strategy the run resolved has a constraint named x (aimed items: in that strategy). Use a name from "Constraints by strategy". |
| `Two constraints are named 'x'` | Two functions with one name in a constraint list. Pass a dict of names to functions. |
| `No valid value found after N attempts` | A number predicate rejected every draw. Narrow the range. |
| `Strategy 'x' (test_y): something drew from the plugin's generator while the rows were generated` warning | A constraint calls `RNG.*`, or an RNG type draws from a generator kept from the factory (`rng`). Draw only inside an RNG type's `generate()`, from `RNG.generator()` or the `RNG.*` helpers. Under `filterwarnings = error` it fails collection as `Error generating samples for strategy 'x': ...`. |
| `the xdist workers generated different vectors` (exit code 4) | Two pytest-xdist workers built different contexts or generated different values under the same IDs. Remove what differs between processes: global `random` draws in factories (use `rng`), lists built from sets (sort them), temporary paths, process IDs or times in the context (or pydantic `Field(exclude=True)`). |
| `Series combination (...) skipped` warning | One combination's random arguments failed the constraints `max_retries` times. Raise `max_retries` or relax the constraint. |
| `... would generate N rows ..., more than the limit of ...` | Too many exhaustive combinations. Reduce them or raise `max_exhaustive` / `strategies_max_exhaustive`. |
| `Directed vector 'x' has N values, expected M` | A directed or test vector does not have one value per argument. |
| `Directed vector 'x' has unknown argument 'lenght' (did you mean 'len'?)`, `... is missing 'len'` | A dict vector's keys must be exactly the argument names. |
| `Directed vector 'x' is -2 (int), not a tuple of values. For a one-argument strategy write (-2,) or {'n': -2}` | A bare value as a vector. Write a dict (or a tuple) of the values. |
| `... is an instance of Burst, a dataclass: record instances as vectors are not supported yet` | Use a dict of the argument names to values. |
| an `id=` on a `pytest.param` vector fails | The vector's name is its test ID. Rename the vector, or build IDs with `Parameter(ids=...)`. |
| `RNGValueError` for an argument name | Argument names are identifiers that are not keywords and do not start with `_`. |
| `TestArg 'x' rng_type must be an RNGType or have a generate() method, got ...` (`TypeError`) | A lambda or a class was given as `rng_type`. Pass an instance (`RNGBoolean()`) or subclass `RNGType`. |
| `TestArg.__init__() takes from 2 to 3 positional arguments but 4 were given` | Options after `rng_type` are keyword-only: `TestArg("x", value=5)`. Likewise `strategy("x", validate_signature=False)` and `export_strategies(format="json")`. |
| signature mismatch at collection | The test does not take every argument name (and record mode does not apply). Add the parameters or a matching dataclass. Lines below it say when a fixture asks for an argument, or a parameter's annotation is unresolvable, a union or has a default. |
| `parameters 'p' (Point) and 'q' (Other) are each annotated with a record type ...` | Two parameters match the arguments. Annotate only one with a record type. |
| `parameter 'txn' is annotated with BusTxn, a pydantic model; ... not supported yet` | Only dataclasses are built in 4.0. Take the arguments as parameters or use a dataclass. |
| `got empty parameter set` skip | `--vector-name`/`--vector-index` selected a vector this strategy lacks, or `--vector-mode=directed_only`/`test` ran a strategy with no directed/test vectors. |
| `RNGValueError` when building a type | Bad RNG arguments (section 6). |
| `Strategy factory 'x' (...) has a parameter 'n', which the plugin does not provide` | Factories receive `nsamples`, `ctx`, `rng` and `options` by name. Rename the parameter, or give it a default. |
| `has a parameter 'config', a name the plugin reserves` | `base`, `config` and `request` are reserved. Rename the parameter; pass settings through `ctx`. |
| `Strategy 'x': ids= returned 42 for row rand-3; return a str or None` | The `ids=` function returned something else, `""`, or raised. |
| `strategies_ids must be 'names' or 'values'` (usage error) | Fix the ini option or the `-o` value. |
| `strategies_ctx is a session fixture, but the tests that use it have different contexts (...)` | The tests that use `strategies_ctx` (directly or through a fixture such as `tb`) are in folders whose `pytest_strategies_context` answers come from different implementations, named in the message. In a folder with its own implementation, build its fixtures on `get_context(request.config, __file__)`, or run the folders separately. |

## 19. Upgrading from 3.x and 2.x

### Removed in 4.0

Deprecated in 3.0 and removed in 4.0, with what a 4.0 run shows:

| Removed | What you see | Use instead |
| --- | --- | --- |
| factories returning `(argnames, samples)` | collection fails: `Strategy 'name' returned an (argnames, samples) tuple ... no longer supported in 4.0` | return a `Parameter`, with one `TestArg` per argument and the fixed rows as `directed_vectors` (a fixed table without random arguments fits `@pytest.mark.parametrize`) |
| `TestArg(directed_values=..., test_values=..., always_include_directed=...)` | `TypeError: ... got an unexpected keyword argument 'directed_values'`; `TestArg("x")` with neither `value` nor `rng_type` raises `ValueError` | `Parameter(directed_vectors=..., test_vectors=..., always_include_directed=...)` |
| the `TestArg` properties `directed_values`, `test_values`, `has_directed_values` | `AttributeError` | `param.directed_vectors`, `param.test_vectors` |
| `RNG.set_max_retries(n)` | `AttributeError` | `Parameter(max_retries=n)`; predicate retries stay at 100 |
| `pytest_strategy.configure()` | `ImportError` or `AttributeError` | delete the call (it never did anything) |
| `Strategy.set_config()` | `AttributeError` | delete the call; the plugin reads the config itself |

### Other changes from 3.x

- **Test IDs** name the rows (`test_write[rand-3]`, `[directed-zeros]`) instead of
  showing their values: redo `-k` expressions, `--deselect` lists and CI history
  that matched values, or keep the 3.0 format for a while with
  `strategies_ids = values` (section 11).
- **Values.** For the same seed, random rows, factory draws, strategy-file and
  test-module import draws, context-hook draws, fixture and test-body draws and
  `RNG.refresh_seed()` streams give other values than 3.x. Directed and test vectors
  and `Series` values are the same. Record seeds again; turn important 3.x rows into
  directed vectors before upgrading. Inherited test methods in two subclasses get
  rows of their own.
- **Factories** receive their inputs by name (section 3): `def f(n)` fails, rename
  it `nsamples`. `rng` and `options` now receive values, `base`, `config` and
  `request` are reserved, `mock.patch` mocks come first, and a wrapper without
  `functools.wraps` is called with no arguments.
- **Keyword-only options:** `TestArg`'s after `rng_type`, `strategy(...,
  validate_signature=)`, `export_strategies(format=)` and `generate_vectors(n, ...)`.
  An `rng_type` without `generate()` (a lambda) is a `TypeError`.
- **Rows** are `Vector`s (`type(v) is tuple` is false); `vector_constraints`,
  `directed_vectors` and `test_vectors` are read-only mappings (iterating
  `vector_constraints` gives names); two constraints with one name fail; argument
  names must be identifiers; a namedtuple vector is placed by its field names; a
  bare value as a vector and an `id=` on a `pytest.param` vector fail.
- **Context per folder:** a conftest below the rootdir answers only for its folder,
  a conftest answers before a plugin, the hook can run once per implementation, and
  `export_strategies()` uses each factory's own folder. Under pytest-xdist, workers
  with different contexts or values fail the run (exit code 4).
- **Record mode:** a fixture that asks for a strategy argument by name now forces
  named mode, and with `validate_signature=False` a record parameter whose
  same-named fixture consumes the arguments gets that fixture's value.
- **Export:** schema 1 with a `strategies` list (section 2) instead of
  `{name: ...}`; factories get the session's `nsamples` (10 by default), not 1.
- **Output:** the repro section, the failed-row commands, JUnit properties, the
  context line and the `--lf` seed reuse (section 16). Each row carries a
  `strategy` mark holding its `VectorInfo`.
- **Collection is slower:** about 8 µs per drawing argument per random row.

### Upgrading from 2.x

- **Plain `random` is no longer seeded by `--rng-seed`.** Draw through the RNG types
  or `RNG.*` helpers, or seed `random` yourself from `RNG.get_seed()` (section 8).
- **Strategies files load later and only when needed.** In 2.x every strategies file
  was imported when the session started. Now a file is imported when pytest collects
  a test module in its folder (or below). Do not rely on a strategies file being
  imported by another folder's run, and move import-time side effects into fixtures
  or conftest.
- **A same-folder name clash is an error.** 2.x warned and the last registration won.
  Rename one of the strategies, or move it to another folder.
- `register`/`strategy` are plain functions; `Strategy.register`/`Strategy.strategy`
  are the same objects. `@strategy(factory)` accepts the factory itself. Names are
  scoped by folder; the same name in two folders is fine. Strategies files can use
  relative imports, and tests importing them get the loaded module. `RNGValueError`
  is a `ValueError`. `import pytest_strategy.strategy as m` gives the `strategy`
  function.

## 20. The skill installer

`pytest-strategies skill install` (or `python -m pytest_strategy skill install`)
copies this skill from the installed package:

| Flag | Destination |
| --- | --- |
| `--claude` | `.claude/skills/pytest-strategies/` |
| `--agents` (alias `--generic`) | `.agents/skills/pytest-strategies/` |
| `--all` (the default) | both |
| `--global` | the home folders instead of the current directory: `$CLAUDE_CONFIG_DIR/skills` (or `~/.claude/skills`) and `~/.agents/skills` |

Reinstalling replaces only the `pytest-strategies` folder, so run it again after
upgrading the library to keep the skill in step with the installed version.
`pytest-strategies --version` prints the library version.
