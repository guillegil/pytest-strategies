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
def register(name: str): ...                                 # decorator for a factory
def strategy(name_or_factory, validate_signature=True): ...  # decorator for a test
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
- `Strategy.register is register` and `Strategy.strategy is strategy`: the old
  spellings are aliases, with no warning.
- `Strategy.export_strategies(format="json")` returns a JSON string describing every
  registered strategy (arguments, RNG types, vectors). It loads every strategies file
  and passes the same `ctx` as the hook.
- Apply `@strategy` to test functions and methods, not to classes or modules.

## 3. Factories

```python
@register("name")
def factory(nsamples, ctx):   # both parameters optional, any order, keyword or positional
    return Parameter(...)
```

- Called once per test that uses it, at collection, with `metafunc.config` available.
- `nsamples`: the `--nsamples` integer, 10 when the option is not given, or the string
  `"auto"` under `--nsamples=auto`. A factory returning a `Parameter` can ignore it;
  the plugin applies the count.
- `ctx`: the result of the `pytest_strategies_context` hook (section 12). Only
  factories that declare a `ctx` parameter trigger the hook.
- A `functools.partial`, a decorated function or a callable object works as a factory.
- Return a `Parameter`. An `(argnames, samples)` tuple still works but emits a
  `DeprecationWarning` pointing at the factory, and CLI vector options do not apply
  to it.
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
        directed_vectors: dict[str, tuple] | None = None,
        test_vectors: dict[str, tuple] | None = None,
        always_include_directed: bool = True,
        vector_constraints: list[Callable[[tuple], bool]] | None = None,
        max_retries: int = 100,
        nsamples: int | str | None = None,
        per_sequence_samples: bool = False,
        max_exhaustive: int | None = None,
    ): ...
```

| Argument | Meaning |
| --- | --- |
| `*test_args` | the arguments, in order; names must be unique (`RNGValueError` otherwise) |
| `directed_vectors` | named rows, one value per argument in order (a tuple or list), placed before the random rows |
| `test_vectors` | named rows used only by `--vector-mode=test` |
| `always_include_directed` | whether `--vector-mode=mixed` includes the directed vectors |
| `vector_constraints` | functions taking the row tuple; a random row that fails one is redrawn |
| `max_retries` | redraws per row before giving up (int >= 1) |
| `nsamples` | this strategy's default count: `None`, an int >= 0 or `"auto"`; an integer `--nsamples` overrides it |
| `per_sequence_samples` | count rows per combination of the `Series`/`RNGSequence` arguments (section 7) |
| `max_exhaustive` | this strategy's limit on exhaustive combinations (section 7) |

- Directed and test vectors must have one value per argument (`ValueError`
  otherwise). They are not checked by predicates, constraints or validators.
- The `Parameter` copies the dicts and lists it is given.
- Methods: `add_directed_vector(name, values)`, `remove_directed_vector(name)`,
  `add_test_vector(name, values)`, `remove_test_vector(name)`,
  `get_directed_vector(name)`, `get_test_vector(name)`, `get_vector_by_name(name)`,
  `get_vector_by_index(i)`, `list_vector_names()`, `add_constraint(fn)`,
  `clear_constraints()`, `get_arg(name)`, `generate_vectors(n, mode=...)`,
  `generate_exhaustive()`, `to_dict()`.
- Properties: `arg_names`, `arg_types`, `vector_names`, `num_args`,
  `num_directed_vectors`, `skip_reason`.

## 5. TestArg

```python
class TestArg:
    def __init__(
        self,
        name: str,
        rng_type: RNGType | None = None,
        value: Any = None,
        validator: Callable[[Any], bool] | None = None,
        description: str = "",
    ): ...
```

- `name` must match the test's parameter (or a dataclass field).
- Give `rng_type` for generated values, or `value` for a fixed one.
- `validator` runs on generated and fixed values. A value that fails it stops
  collection with `ValueError` and is not redrawn, so filter with a `predicate` on the
  RNG type instead, and use `validator` only as an assertion.
- `directed_values=` and `test_values=` are deprecated: they never produced rows. Use
  the `Parameter`'s `directed_vectors` and `test_vectors`.
- Pass `validator` and `description` by keyword: the 4th and 5th positional
  parameters are the deprecated `directed_values` and `test_values`.

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
  Inside a factory, and in a strategies file while the plugin imports it, that
  generator follows the seed.
- The plugin never calls `random.seed()`. Plain `random` calls (in factories,
  strategies files, conftest or test bodies) are not reproduced by `--rng-seed`.
  To seed plain `random` from the run's seed, do it yourself, for example per test
  with an autouse fixture: `random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")`.
- Draws inside test bodies are not part of the parametrization. Seed them in the test
  (as above) if they must be reproducible.

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
| `--list-strategies` | load every strategies file, list the registered names and exit |

| ini option | Meaning |
| --- | --- |
| `strategies_max_exhaustive` | the project's limit on exhaustive combinations (default 100000) |

Reporting:

- The header shows `pytest-strategies: RNG seed = S` (hidden by `-q` and `--no-header`).
- A run with failures prints "reproduce with `--rng-seed=S`" after the tracebacks,
  also under `-q`.
  A passing run does not print it.
- `-v` adds, per strategy, the counts of directed, random and test rows and where
  `nsamples` came from.
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
derived from the seed and the file's path relative to the rootdir, so values drawn at
import time through the plugin's generators do not depend on which folders load first.

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

Used when all of these hold:

- the strategy has at least two arguments;
- none of the argument names is a parameter of the test;
- exactly one test parameter is annotated with a dataclass whose `__init__` fields are
  the argument names.

Fields with `init=False` are not counted, `kw_only` fields work, and `self`, `cls` and
fixtures can be anywhere in the signature. String annotations
(`from __future__ import annotations`) work if the dataclass is defined at module
level. A dataclass whose fields do not match fails collection with the missing and
extra names. IDs look like `x=1,y=2`.

## 14. Reproducibility and pytest-xdist

- Each (strategy, test) pair draws from its own `random.Random`, keyed by the seed,
  the strategy name, the test file's path relative to the rootdir and the test's
  qualified name. A test gets the same rows (and node IDs) whether you run the whole
  suite, one file or one test, in any order and with any `--import-mode`. Two tests
  sharing a strategy get different random rows.
- Keep the same seed, the same rootdir and the same plugin version. The rootdir is
  the directory of the ini file (`pytest.ini`, or `pyproject.toml` with
  `[tool.pytest.ini_options]`); without one it depends on the folder pytest is run
  from, so compare the `rootdir:` line of both runs. Values for a seed may differ between major versions; check the
  CHANGELOG before comparing with a 2.x run.
- pytest-xdist works with or without `--rng-seed`: the controller sends its seed to
  the workers.

## 15. Errors and what to do

| Message | Cause and fix |
| --- | --- |
| `Strategy 'x' not found` | Not visible from the test's folder. Check the spelling (see "did you mean"), move the strategy to a parent folder, pass the factory with `@strategy(factory)`, or fix a strategies file that failed to load (listed). |
| usage error naming two files for one name | Two factories with one name in the same folder. Rename one, or move it to its own folder. |
| `Could not generate valid vector` | The constraints rejected every draw within `max_retries`; the message names each constraint and its rejection count. Relax the constraint, narrow the ranges or raise `Parameter(max_retries=)`. |
| `No valid value found after N attempts` | A number predicate rejected every draw. Narrow the range. |
| `Series combination (...) skipped` warning | One combination's random arguments failed the constraints `max_retries` times. Raise `max_retries` or relax the constraint. |
| `... would generate N rows ..., more than the limit of ...` | Too many exhaustive combinations. Reduce them or raise `max_exhaustive` / `strategies_max_exhaustive`. |
| `Directed vector 'x' has N values, expected M` | A directed or test vector does not have one value per argument. |
| signature mismatch at collection | The test does not take every argument name (and dataclass mode does not apply). Add the parameters or a matching dataclass. |
| `got empty parameter set` skip | `--vector-name`/`--vector-index` selected a vector this strategy lacks, or `--vector-mode=directed_only`/`test` ran a strategy with no directed/test vectors. |
| `RNGValueError` when building a type | Bad RNG arguments (section 6). |
| `Factory should accept an 'nsamples' parameter (or no parameters)` | The factory's signature cannot receive `nsamples`. Accept `nsamples`, `**kwargs` or nothing. |

## 16. Deprecations and upgrading from 2.x

Deprecated in 3.0 (a `DeprecationWarning` at your line; removed in 4.0):

| Deprecated | Use instead |
| --- | --- |
| factories returning `(argnames, samples)` | return a `Parameter` |
| `TestArg(directed_values=..., test_values=...)` | `Parameter(directed_vectors=..., test_vectors=...)` |
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
