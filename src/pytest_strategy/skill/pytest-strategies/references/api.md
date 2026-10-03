# pytest-strategies 3.0.0 API reference

Read the section you need; SKILL.md has the overview and the common traps.

1. [Imports](#1-imports)
2. [register and strategy](#2-register-and-strategy)
3. [Factories](#3-factories)
4. [Parameter](#4-parameter)
5. [TestArg](#5-testarg)
6. [RNG types](#6-rng-types)
7. [Series, RNGSequence and exhaustive mode](#7-series-rngsequence-and-exhaustive-mode)
8. [The RNG helpers and seeding](#8-the-rng-helpers-and-seeding)
9. [Sample counts and vector modes](#9-sample-counts-and-vector-modes)
10. [Command-line and ini options](#10-command-line-and-ini-options)
11. [Strategies files and name lookup](#11-strategies-files-and-name-lookup)
12. [The pytest_strategies_context hook](#12-the-pytest_strategies_context-hook)
13. [Dataclass mode](#13-dataclass-mode)
14. [Reproducibility and pytest-xdist](#14-reproducibility-and-pytest-xdist)
15. [Errors and what to do](#15-errors-and-what-to-do)
16. [Deprecations and upgrading from 2.x](#16-deprecations-and-upgrading-from-2x)
17. [The skill installer](#17-the-skill-installer)

## 1. Imports

Everything public is importable from the package root:

```python
from pytest_strategy import (
    register, strategy, Strategy,           # decorators; Strategy holds the aliases
    Parameter, TestArg,
    StrategyOptions,                        # what a factory receives as options
    Vector,                                 # the class of every generated row
    VectorInfo, VECTOR_KEY, VECTORS_KEY,    # a test item's row: item.stash[VECTOR_KEY]
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

## 2. register and strategy

```python
def register(name: str): ...                                    # decorator for a factory
def strategy(name_or_factory, *, validate_signature=True): ...  # decorator for a test
```

- `@register("name")` records the factory under `name`, scoped to the folder of the
  file that defines it (see section 11). It returns the function unchanged, with its
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
  spellings are aliases, with no warning.
- `Strategy.export_strategies(format="json")` returns a JSON string describing every
  registered strategy (arguments, RNG types, vectors). It loads every strategies file
  and calls each factory as at collection: the same `ctx` as the hook, and the
  session's `nsamples` (10 without `--nsamples`) and `options`.
- Apply `@strategy` to test functions and methods, not to classes or modules.

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
  - `ctx`: the result of the `pytest_strategies_context` hook (section 12). Only
    factories that declare a `ctx` parameter trigger the hook.
  - `rng`: the plugin's `random.Random`; `rng is RNG.generator()` during the call.
  - `options`: a frozen `StrategyOptions` (keyword-only fields `strategy`, `nsamples`,
    `nsamples_source`, `mode`, `vector_name`, `vector_index`, `constraints_off`, and
    the `filtered` property).
- Any other parameter must have a default, which it keeps, and `*args`/`**kwargs`
  receive nothing. `def factory(n)` or `def factory(n, /)` fails collection with
  `has a parameter 'n', which the plugin does not provide ... Did you mean 'nsamples'?`.
  `base`, `config` and `request` are reserved, even with a default.
- A `functools.partial`, a bound method, a classmethod, a staticmethod, a class, a
  callable object or a `functools.wraps` decorated function works as a factory. A
  decorator without `functools.wraps` hides the signature, so the factory is called
  with no arguments. `mock.patch` mocks must be the first parameters. An `async def`
  factory fails.
- Return a `Parameter`. Since 4.0 an `(argnames, samples)` tuple fails the
  collection of the tests that use it.
- An exception in the factory fails the collection of the tests that use it, with
  `Error calling strategy factory '<name>' (nsamples=...)`.
- Values the factory draws itself are reproducible only when drawn through the
  plugin's generators (section 8), not through plain `random`.

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
| `directed_vectors` | named rows, placed before the random rows: one value per argument in order (a tuple or list), or by name (a dict of argument names to values, or a namedtuple with those fields); names are non-empty strings |
| `test_vectors` | named rows used only by `--vector-mode=test`, in the same forms |
| `always_include_directed` | whether `--vector-mode=mixed` includes the directed vectors |
| `vector_constraints` | functions taking the row as a `Vector` (`v.lo` or `v[0]`), as a dict of names to functions or a list (named by each function's `__name__`, `constraint_<i>` for a lambda); they run in order and a random row that fails one is redrawn |
| `max_retries` | redraws per row before giving up (int >= 1) |
| `nsamples` | this strategy's default count: `None`, an int >= 0 or `"auto"`; an integer `--nsamples` overrides it |
| `per_sequence_samples` | count rows per combination of the `Series`/`RNGSequence` arguments (section 7) |
| `max_exhaustive` | this strategy's limit on exhaustive combinations (section 7) |
| `ids` | this strategy's test IDs: `None` follows the `strategies_ids` ini option, `"names"` or `"values"` overrides it, and a function receives each row's `VectorInfo` (its `id` in the ini option's format) and returns the ID or `None` to keep it; not called for the skipped row. Duplicates get pytest's suffixes (`odd0`, `odd1`); anything but a non-empty str or None, or an exception, fails collection (`Strategy 'burst': ids= returned 42 for row rand-3; return a str or None`). IDs never change the generated rows; the `RNG` draws of test phases and function-scoped fixtures are keyed by the node ID, so they follow it |

- Directed and test vectors must give one value per argument (`RNGValueError`
  otherwise, naming the vector: a wrong length, a missing or unknown dict key,
  with "did you mean"). A str, bytes or scalar vector fails with
  `For a one-argument strategy write ('a',) or {'x': 'a'}`, and a dict value for
  one argument is written `({"a": 1},)` or `{"cfg": {"a": 1}}` (in a `pytest.param`,
  only `pytest.param({"cfg": {"a": 1}}, marks=...)`). Dataclass and
  pydantic model instances are not supported as vectors yet; use a dict.
  `pytest.param(..., marks=...)` wraps any of these forms (an `id=` fails: the
  vector's name is its test ID; build other IDs with `ids=`). Vectors are not
  checked by predicates, constraints or validators.
- The `Parameter` stores each vector as a `Vector` (a `pytest.param` keeps its
  marks, with a `Vector` as its values). `directed_vectors` and `test_vectors`
  are read-only mappings; change them with the `add_*` and `remove_*` methods.
- The `Parameter` copies the dicts and lists it is given. `vector_constraints` is
  a read-only mapping of names to functions (iterating gives the names). A
  constraint name is a non-empty string without whitespace, `:`, `,` or `=`; two
  constraints with one name (closures from one helper) fail, so name them with a
  dict.
- Methods: `add_directed_vector(name, values)`, `remove_directed_vector(name)`,
  `add_test_vector(name, values)`, `remove_test_vector(name)`,
  `get_directed_vector(name)`, `get_test_vector(name)`, `get_vector_by_name(name)`,
  `get_vector_by_index(i)`, `list_vector_names()`,
  `add_constraint(fn, *, name=None)` (returns the name), `remove_constraint(name)`,
  `clear_constraints()`, `get_arg(name)`,
  `generate_vectors(n, *, mode="all", filter_by_name=None, filter_by_index=None, constraints_off=())`,
  `generate_exhaustive(*, constraints_off=())`, `to_dict()`. `constraints_off` names
  constraints that call does not evaluate (names it does not have are ignored); the
  `Parameter` keeps them.
- Properties: `arg_names`, `arg_types`, `vector_names`, `num_args`,
  `num_directed_vectors`, `skip_reason`, `vector_type`.
- The generated rows are `Vector`s: tuples whose fields are the argument names
  (`v.lo`), equal to the plain tuple of their values, with the repr
  `Vector(lo=0, hi=5)`. A misspelled field raises
  `AttributeError: Vector has no argument 'hgh'; its arguments are lo, hi`. Under
  mypy a field is `Any`, so a constraint annotated `-> bool` returns `bool(...)`.
  The generators and the `get_*_vector()` methods are typed as returning `Vector`s;
  a `pytest.param(...)` vector comes back as the `pytest.param` whose `.values` is
  the Vector.

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
```

- `name` must match the test's parameter (or a dataclass field).
- Give `rng_type` for generated values, or `value` for a fixed one. `value`,
  `validator` and `description` are keyword-only: `TestArg("x", None, 5)` raises
  `TypeError`; write `TestArg("x", value=5)`.
- `rng_type` must be an RNG type (or an object with a `generate()` method). A bare
  function or lambda raises `TypeError`; subclass `RNGType` instead (section 6). So
  does the class itself: write `RNGBoolean()`, not `RNGBoolean`.
- `validator` runs on generated and fixed values. A value that fails it stops
  collection with `ValueError` and is not redrawn, so filter with a `predicate` on the
  RNG type instead, and use `validator` only as an assertion.
- `directed_values=`, `test_values=` and `always_include_directed=` were removed in
  4.0: they never produced rows. Use the `Parameter`'s `directed_vectors`,
  `test_vectors` and `always_include_directed`.

## 6. RNG types

All draw from the generator the plugin gives them, so their values follow the seed.
Each checks its arguments when built and raises `RNGValueError` (a `ValueError`).

| Type | Signature | Rejected at construction |
| --- | --- | --- |
| `RNGInteger` | `(min=-2**31, max=2**31-1, predicate=None)` | `min > max`, also when only one bound is given |
| `RNGFloat` | `(min=0.0, max=1.0, predicate=None)` | `min > max` (`RNGFloat(min=5.0)` fails: `max` defaults to 1.0), infinite or NaN bounds |
| `RNGBoolean` | `(true_probability=0.5)` | |
| `RNGChoice` | `(choices: list)` | an empty list |
| `RNGEnum` | `(enum_class, weights=None, predicate=None)` | not an Enum class, no members, weights that are empty, negative, non-finite, all zero or not members, a predicate no member satisfies |
| `RNGString` | `(length=None, min_length=1, max_length=20, charset="abcdefghijklmnopqrstuvwxyz")` | negative lengths, `min_length > max_length`, an empty charset with a length above 0 |
| `RNGWeightedInteger` | `(ranges: dict[(lo, hi), weight], predicate=None)` | bad weights, a range that is not a `(lo, hi)` tuple with `lo <= hi` |
| `RNGWeightedFloat` | `(ranges: dict[(lo, hi), weight], predicate=None)` | as above, plus infinite or NaN bounds |
| `Series` | `(sequence, predicate=None, *, skip_if_empty=None)` | see section 7 |
| `RNGSequence` | `(sequence, predicate=None, *, skip_if_empty=None)` | see section 7 |

- Bounds are inclusive.
- `predicate` filters draws. For numbers a draw that fails is retried (up to 100
  times, then `RNGValueError`); `RNGEnum` and the sequences filter their members up
  front, so a draw never fails while a valid member exists.
- `RNGEnum` weights do not need to sum to 1. With weights, only the weighted members
  are drawn. With a predicate, the accepted members keep their relative weights.
- `RNGWeighted*` choose a range by weight, then a value in it.
- Custom types subclass `RNGType` and implement `generate()` and `python_type`.
  `generate()` draws from `RNG.generator()` or the `RNG.*` helpers called inside
  it, never from a generator kept from earlier (such as the factory's `rng`), and
  keeps no state between calls (a counter makes row k depend on the rows before
  it). Each argument of a random row then draws from a stream of its own.

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
  dataclass) is still checked. A reason passed positionally lands in `predicate` and
  raises "did you mean skip_if_empty=...?".
- **Constraints with `Series` in finite mode:** a combination the constraints reject is
  skipped with a `PytestStrategiesWarning` (after `max_retries` redraws of its random
  arguments), and the cycle continues. Collection fails only when a whole cycle yields
  nothing.
- **`per_sequence_samples=True`:** a finite K counts per combination of the sequence
  arguments, walked in declaration order for both types. Two devices and K=10 give 20
  rows; several sequence arguments multiply. Directed vectors still come first. A
  combination whose random arguments keep failing the constraints gets fewer rows and
  a warning; collection fails only when no combination yields a row. Ignored under
  `auto`, and a `Parameter` without sequence arguments behaves as if it were `False`.
- **Under `auto`:** combinations the constraints reject are dropped (after
  `max_retries` redraws of the random arguments); if all are dropped, collection
  fails. Directed vectors are added per `--vector-mode`. A strategy with no sequence
  argument falls back to its own `nsamples` or 10.
- **Size guard:** before generating exhaustive rows (the combinations under `auto`,
  or combinations x K with `per_sequence_samples=True`), the plugin computes their
  count without building them. Above the limit
  (100,000 rows by default) collection fails with the strategy name, the count and how
  to raise the limit: `Parameter(max_exhaustive=...)` for one strategy, or the
  `strategies_max_exhaustive` ini option for the project.

## 8. The RNG helpers and seeding

```python
RNG.get_seed()                         # the run's seed
RNG.generator()                        # the random.Random the RNG types draw from
RNG.integer(min, max, predicate=None)
RNG.float(min=0.0, max=1.0, predicate=None)
RNG.boolean(true_probability=0.5)
RNG.choice(items)
RNG.string(length=None, min_length=1, max_length=20, charset="abc...z")
RNG.winteger(ranges, predicate=None)    # ranges: {(lo, hi): weight, ...}
RNG.wfloat(ranges, predicate=None)
```

- The helpers draw from a generator the plugin owns, never from the global `random`.
  In a factory, a strategies file while the plugin imports it, a test module's top
  level, a fixture's setup and each phase of a test, that generator follows the
  seed on a stream of its own: a test body's `RNG.integer()` is the same alone, in
  the suite and under xdist, and an `RNG.seed()` call changes only the rest of its
  stream. Draws when a `conftest.py` is imported do not follow the seed, and a
  helper module that test modules or strategy files import draws on the stream of
  the first module that imports it, shifting that module's draws, so its values
  change when one file runs alone: draw in a fixture, the context hook or a
  strategies file instead.
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

With `per_sequence_samples=True` a finite count applies to each combination.

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
("got empty parameter set"). When no strategy in the run has it, pytest stops with a
usage error listing each strategy's directed vectors.

## 10. Command-line and ini options

| Option | Meaning |
| --- | --- |
| `--rng-seed=S` | integer seed; without it a new seed is chosen each run |
| `--nsamples=N` or `--nsamples=auto` | random rows per strategy (int >= 0), or enumerate sequences; anything else is a usage error |
| `--vector-mode=MODE` | `all`, `random_only`, `directed_only`, `mixed`, `test` |
| `--vector-name=NAME` | only the directed vector named NAME |
| `--vector-index=I` | only the directed vector at index I |
| `--strategy-constraint-off=[S:]NAME[,...]` | turn the constraint NAME off for this run, in every strategy or only in strategy S; repeatable. The header lists what is off and `-v` adds `off: NAME`. An item that matches no constraint of a resolved strategy is a usage error in a whole-suite run (with "did you mean" and the constraints by strategy), and a red line in a run narrowed by paths, node IDs, `--lf`, `--sw`, `--ignore` or a start below the rootdir, or with a module that was skipped or failed to collect |
| `--list-strategies` | load every strategies file, list the registered names and exit |

| ini option | Meaning |
| --- | --- |
| `strategies_max_exhaustive` | the project's limit on exhaustive combinations (default 100000) |
| `strategies_ids` | `names` (default: `directed-zeros`, `test-max`, `rand-3`, `ch=2-rand-1`, `ch=0-dev=b`, `skipped`; the same for every seed) or `values` (the 3.0 IDs built from the values); anything else is a usage error |

Reporting:

- The header shows `pytest-strategies: RNG seed = S` (hidden by `-q` and `--no-header`).
- A run with failures prints "reproduce with `--rng-seed=S`" after the tracebacks,
  also under `-q`.
  A passing run does not print it.
- `-v` adds, per strategy, the counts of directed and random rows (and of test,
  exhaustive and skipped rows when there are some) and where `nsamples` came from.
- Warnings raised while generating a strategy's rows are reported at the test, prefixed
  with `Strategy '<name>' (<test>): `.

## 11. Strategies files and name lookup

**File names:** `strategies.py`, `strategy.py`, `*_strategies.py`, `*_strategy.py`,
`test_strategies.py`. A file is loaded only if its text contains `@register("`,
`@Strategy.register(` or `@<module>.register("` (for example `@ps.register("` after
`import pytest_strategy as ps`); outside `@Strategy.register(`, the name must be
written as a string literal. A `conftest.py` may also register strategies.

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
  code 4, or 2 under pytest-xdist) naming both files. The same function registered again is silent.
- `@strategy(factory)` skips the lookup entirely.

**Load failures:** a strategies file that raises while importing is reported and does
not stop the run; a file that calls `pytest.skip(..., allow_module_level=True)` or
`pytest.importorskip()` at module level is skipped. "Strategy 'x' not found" lists the
names visible from the test, the closest match ("did you mean"), and the files that
failed to load or were skipped.

## 12. The pytest_strategies_context hook

```python
# conftest.py at the rootdir (or a plugin)
import pytest

@pytest.hookimpl(optionalhook=True)
def pytest_strategies_context(config):
    return Testbench.parse_config(config.getoption("--tb-config"))
```

- A `firstresult` hook; `config` is the pytest config. Return any object.
- `optionalhook=True` keeps the `conftest.py` usable when the plugin is not loaded
  (without it pytest stops with "unknown hook").
- Called at most once per session, the first time a factory with a `ctx` parameter
  runs; the result (or the exception) is reused for every later factory.
- When no implementation returns a value, `ctx` keeps its default (or a value bound
  with `functools.partial`), else `None`.
- If the hook raises, each test that uses a factory with `ctx` fails collection with
  `Strategy factory '<name>' has a 'ctx' parameter, but the pytest_strategies_context
  hook raised <error>`. `pytest.fail()` is reported as is, and
  `pytest.skip(..., allow_module_level=True)` skips those tests.
- Implement it in the rootdir `conftest.py`: a deeper `conftest.py` is loaded only
  when pytest reaches its folder, which may be too late, and its result then applies
  to the whole session.
- Random draws in the hook come from a stream of their own, derived from the seed, and
  do not shift any test's rows.
- Under pytest-xdist every worker calls it; it must return the same configuration in
  each, or xdist reports "Different tests were collected".
- Use it for data the rows depend on; the live objects (a testbench connection) stay
  fixtures.

## 13. Dataclass mode

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
  supported yet". Unions, pydantic v1 models, attrs classes and unresolvable
  annotations are not record types.
- Two matching parameters fail and are both named. A single record parameter whose
  fields do not match fails with the missing and extra names.

String annotations (`from __future__ import annotations`) work if the dataclass is
defined at module level. IDs look like `x=1,y=2`.

## 14. Reproducibility and pytest-xdist

- Each argument of a random row draws from its own `random.Random`, keyed by the
  seed, the strategy name, the test's node ID without its parameters (its path
  relative to the rootdir, its class and its name), the row and the argument's name.
  A test gets the same rows (and node IDs) whether you run the whole suite, one file
  or one test, in any order and with any `--import-mode`. Two tests sharing a
  strategy get different random rows. A row keeps its values with more rows, when
  its node ID runs alone and when another argument is added or changed; a
  constraint redraws only the rows it rejects.
- A constraint that draws (`RNG.integer()`) and an RNG type that draws from a
  generator kept from the factory draw outside those streams: their values change
  when other rows or tests change. Each test whose rows do so gets one
  `PytestStrategiesWarning` naming the strategy and the test. Move the draw into an
  RNG type's `generate()`, and keep constraints to reading the row.
- Keep the same seed, the same rootdir and the same plugin version. The rootdir is
  the directory of the ini file (`pytest.ini`, or `pyproject.toml` with
  `[tool.pytest.ini_options]`); without one it depends on the folder pytest is run
  from, so compare the `rootdir:` line of both runs. Values for a seed may differ between major versions; check the
  CHANGELOG before comparing with a 3.x run (4.0.0 changed the random rows).
- pytest-xdist works with or without `--rng-seed`: the controller sends its seed to
  the workers.

## 15. Errors and what to do

| Message | Cause and fix |
| --- | --- |
| `Strategy 'x' not found` | Not visible from the test's folder. Check the spelling (see "did you mean"), move the strategy to a parent folder, pass the factory with `@strategy(factory)`, or fix a strategies file that failed to load (listed). |
| usage error naming two files for one name | Two factories with one name in the same folder. Rename one, or move it to its own folder. |
| `Could not generate random row K after max_retries=N draws` (or `Could not generate valid vector` for combinations) | The constraints rejected every draw; the message counts the rejections by the name of the first failing constraint and shows the first row each rejected. Relax the constraint, narrow the ranges or raise `Parameter(max_retries=)`. |
| `Constraint 'x' raised ...` | A constraint raised on the row shown (its frame follows). Fix the constraint; one before it in the mapping can guard it (`{"nonzero": ..., "ratio": ...}`), and `(constraint 'nonzero' before it is turned off by --strategy-constraint-off)` says that guard was turned off. |
| `--strategy-constraint-off=x matched no constraint` | No strategy the run resolved has a constraint named x (aimed items: in that strategy). Use a name from "Constraints by strategy". |
| `Two constraints are named 'x'` | Two functions with one name in a constraint list. Pass a dict of names to functions. |
| `No valid value found after N attempts` | A number predicate rejected every draw. Narrow the range. |
| `Strategy 'x' (test_y): something drew from the plugin's generator while the rows were generated` warning | A constraint calls `RNG.*`, or an RNG type draws from a generator kept from the factory (`rng`). Draw only inside an RNG type's `generate()`, from `RNG.generator()` or the `RNG.*` helpers. Under `filterwarnings = error` it fails collection as `Error generating samples for strategy 'x': ...`. |
| `Series combination (...) skipped` warning | One combination's random arguments failed the constraints `max_retries` times. Raise `max_retries` or relax the constraint. |
| `... would generate N rows ..., more than the limit of ...` | Too many exhaustive combinations. Reduce them or raise `max_exhaustive` / `strategies_max_exhaustive`. |
| `Directed vector 'x' has N values, expected M` | A directed or test vector does not have one value per argument. |
| signature mismatch at collection | The test does not take every argument name (and dataclass mode does not apply). Add the parameters or a matching dataclass. Lines below it say when a fixture asks for an argument, or a parameter's annotation is unresolvable, a union or has a default. |
| `parameters 'p' (Point) and 'q' (Other) are each annotated with a record type ...` | Two parameters match the arguments. Annotate only one with a record type. |
| `parameter 'txn' is annotated with BusTxn, a pydantic model; ... not supported yet` | Only dataclasses are built in 4.0. Take the arguments as parameters or use a dataclass. |
| `got empty parameter set` skip | `--vector-name`/`--vector-index` selected a vector this strategy lacks, or `--vector-mode=directed_only`/`test` ran a strategy with no directed/test vectors. |
| `RNGValueError` when building a type | Bad RNG arguments (section 6). |
| `Strategy factory 'x' (...) has a parameter 'n', which the plugin does not provide` | Factories receive `nsamples`, `ctx`, `rng` and `options` by name. Rename the parameter, or give it a default. |
| `has a parameter 'config', a name the plugin reserves` | `base`, `config` and `request` are reserved. Rename the parameter; pass settings through `ctx`. |

## 16. Deprecations and upgrading from 2.x

Deprecated in 3.0 and removed in 4.0:

| Removed | Use instead |
| --- | --- |
| factories returning `(argnames, samples)` | return a `Parameter` |
| `TestArg(directed_values=..., test_values=..., always_include_directed=...)` | `Parameter(directed_vectors=..., test_vectors=..., always_include_directed=...)` |
| `RNG.set_max_retries(n)` | `Parameter(max_retries=n)` |
| `pytest_strategy.configure()` | nothing (it never did anything) |
| `Strategy.set_config()` | nothing; the plugin reads the config itself |

Breaking changes from 2.x, and what to do:

- **Plain `random` is no longer seeded by `--rng-seed`.** Draw through the RNG types
  or `RNG.*` helpers, or seed `random` yourself from `RNG.get_seed()` (section 8).
- **Strategies files load later and only when needed.** In 2.x every strategies file
  was imported when the session started. Now a file is imported when pytest collects
  a test module in its folder (or below). Do not rely on a strategies file being imported by
  another folder's run, and move import-time side effects into fixtures or conftest.
- **A same-folder name clash is an error.** 2.x warned and the last registration won.
  Rename one of the strategies, or move it to another folder.

Other changes worth knowing when editing 2.x-era code:

- `register`/`strategy` are plain functions; `Strategy.register`/`Strategy.strategy`
  are the same objects.
- `@strategy(factory)` accepts the factory itself.
- Names are scoped by folder; the same name in two folders is fine.
- Strategies files can use relative imports, and tests importing them get the loaded
  module.
- `RNGValueError` is a `ValueError`.
- `import pytest_strategy.strategy as m` now gives the `strategy` function.

## 17. The skill installer

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
