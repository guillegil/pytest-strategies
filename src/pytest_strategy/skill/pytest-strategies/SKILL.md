---
name: pytest-strategies
description: Write, fix and debug pytest tests that use the pytest-strategies plugin (import name pytest_strategy) for constrained-random and directed test parametrization. Use this skill whenever code imports pytest_strategy or uses register/strategy, Parameter, TestArg, Vector, RNGInteger, RNGChoice, RNGEnum, Series, RNGSequence or another RNG type; when creating or editing strategies.py or *_strategies.py files; when adding directed or test vectors, named constraints or config-driven strategies (the pytest_strategies_context hook, ctx, strategies_ctx, get_context, skip_if_empty); when choosing --nsamples, --vector-mode, --strategy-constraint-off or -k on row IDs such as test_x[rand-3]; and when a failed generated test has to be reproduced from its pytest-strategies section, --rng-seed or --lf. Use it even when the user only says randomized tests, test vectors or strategies in a project that depends on pytest-strategies.
---

# pytest-strategies

Documents pytest-strategies 4.0.0.

pytest-strategies (import name `pytest_strategy`) parametrizes pytest tests from
*strategies*: factory functions that return a `Parameter` describing each test
argument (a random generator, a fixed value or an ordered series) plus named edge
cases. Rows are generated at collection time from a per-run seed, so each run
explores new values. Every row has a name in its test ID that is the same for
every seed, and a failed row prints the command that reruns it with its values.

This file covers the patterns and traps behind most tasks. For full signatures,
every option, the error messages and what changed from 3.x, read
[references/api.md](references/api.md).

## First, look at the project

- Check the installed version against the one above:
  `python -c "import pytest_strategy as p; print(p.__version__)"`. Before 4.0,
  test IDs showed the values (`test_x[addr=0,len=1]`) instead of the row's name
  and directed vectors were tuples only; 2.x also lacks the plain
  `register`/`strategy` functions. The reference's section 19 says what differs.
- List what already exists: `pytest --list-strategies` prints every registered
  name and exits. Reuse or extend a strategy before writing a new one.
- Look for strategies files next to the tests and in parent folders, and for a
  `pytest_strategies_context` hook in the `conftest.py` files (each folder can
  have its own).

## The basic pattern

```python
# tests/dma/strategies.py
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("dma_burst")
def dma_burst():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 0xFFFF)),
        TestArg("length", rng_type=RNGInteger(1, 256)),
        directed_vectors={                          # always run, before the random rows
            "zeros": {"addr": 0, "length": 1},      # by argument name
            "page_end": (4092, 4),                  # or one value per TestArg, in order
        },
        vector_constraints={                        # named; each gets the row (a Vector)
            "aligned": lambda v: v.addr % 4 == 0,
            "no_4k_cross": lambda v: v.addr % 4096 + v.length <= 4096,
        },
    )
```

```python
# tests/dma/test_dma.py
from pytest_strategy import strategy

@strategy("dma_burst")                     # looked up from this folder upward
def test_burst(addr, length, dma):         # dma is an ordinary fixture
    assert dma.transfer(addr, length).ok
```

`pytest tests/dma` runs `test_burst[directed-zeros]`, `test_burst[directed-page_end]`
and then `test_burst[rand-0]` to `test_burst[rand-9]`, with new random values each
run.

A test can also take the factory itself, `@strategy(dma_burst)`: it then needs no
registration and no name lookup, and editors can jump to it. To pass a factory from
a strategies file, import it (in a package folder: `from .strategies import
dma_burst`); the test gets the same module object the plugin loaded.

## How a factory is called

- Once for each test that uses it, while pytest collects that test (in
  `pytest_generate_tests`), not when the file is imported.
- It receives by name, in any order, only the inputs it declares, like a test
  with fixtures: `nsamples` (an int, or `"auto"` under `--nsamples=auto`), `ctx`
  (the testbench context of the test's folder, see below), `rng` (the plugin's
  `random.Random`) and `options` (a frozen `StrategyOptions`: the run's mode,
  vector filters and constraints turned off). Most factories need none of them:
  return a `Parameter` and the plugin generates the rows.
- Any other parameter needs a default: `def f(n)` fails collection (rename it
  `nsamples`, or drop it). `*args`/`**kwargs` receive nothing, and `base`, `config`
  and `request` are reserved. Fixtures are never factory inputs.
- It returns a `Parameter`; anything else (a tuple, `None`) fails collection.
- The test takes every `TestArg` name as a parameter (checked at collection;
  `@strategy(..., validate_signature=False)` turns the check off), or the whole row
  as one record (see Traps). Its other parameters are fixtures, as usual, and a
  fixture can take strategy arguments too: `def user(name, age)` gets each row's.

## Test IDs and selecting rows

Each row is named in the test ID, the same for every seed:

| Row | ID |
| --- | --- |
| directed vector `zeros` | `test_burst[directed-zeros]` |
| test vector `max` (only with `--vector-mode=test`) | `test_burst[test-max]` |
| random row 3 | `test_burst[rand-3]` |
| random row 1 of the `Series` value `ch=2` | `test_esm[ch=2-rand-1]` |
| a row of `--nsamples=auto` | `test_esm[ch=2-dev=b]` |
| the row of an empty `skip_if_empty` sequence | `test_esm[skipped]` |

- `-k zeros` runs the directed vector `zeros` (so does `--vector-name=zeros`),
  `-k directed-` every directed row and `-k "not rand-"` no random row. Keep the
  `-`: `-k` matches substrings of the whole node name, so without it a test,
  module or vector whose name holds the word (`test_random_io`, `operand_max`)
  matches too.
- `-k "rand-3"` also matches `rand-30`: for one row use `-k "test_burst[rand-3]"`
  or the node ID. `-k` cannot contain `=`, so select a sequence value's rows by
  node ID: `pytest "tests/test_esm.py::test_esm[ch=2-rand-1]"`.
- Name vectors like identifiers (`bug_1234`, `page_end`): a name in `-k` can
  hold only letters, digits and `_ - . : / + \ [ ]`, not spaces, `=`,
  parentheses or commas.
- The IDs show no drawn values. `-o strategies_ids=values` (or the ini option)
  brings back the 3.0 IDs built from the values, which change with the seed;
  `Parameter(ids="values")` does it for one strategy, and `ids=fn` builds IDs from
  each row's `VectorInfo` (return `None` to keep one). An `id=` on a `pytest.param`
  vector fails: the vector's name is its ID.
- With stacked `@strategy` or `@pytest.mark.parametrize` decorators the parts are
  joined with `-`, the decorator nearest the `def` first:
  `test_two[directed-zeros-fast]`.

## Where strategies live

A strategies file is a file named `strategies.py`, `strategy.py`,
`*_strategies.py`, `*_strategy.py` or `test_strategies.py` that contains a
register decorator with the name as a string literal (`@register("name")`,
`@Strategy.register(...)`, `@<module>.register("name")`); `register("x")(fn)` or
`@register(NAME)` alone is not detected. A `conftest.py` or a test module can
register strategies too.

Names are scoped by folder, like fixtures:

- A name is visible in the folder that registers it and in every folder below.
- The plugin walks from the test's folder up to the `testpaths` entry (or
  command-line folder) that contains it, or to the rootdir when `testpaths` is not
  set, and the nearest folder that registers the name wins. `tests/esm/strategies.py`
  and `tests/dma/strategies.py` can both define `"default"`, and a
  `tests/strategies.py` `"default"` serves folders that have none. A strategies
  file above that point (at the project root with `testpaths = ["tests"]`) is never
  loaded: put shared strategies in `tests/strategies.py`, or register them in the
  rootdir `conftest.py`.
- Files with the same name in several folders are fine for registration. To import
  from them in tests, the folders need `__init__.py` (`from .strategies import X`)
  or the files need distinct names (`esm_strategies.py`): a plain
  `from strategies import X` gets whichever folder's file was imported first.
- A name that no folder on that path defines, but that is registered only once
  elsewhere (a sibling folder, an installed package), is found too. When several
  such folders register it, the lookup fails and names them.
- Two different factories with one name in one folder are a usage error naming
  both files. "Strategy 'x' not found" lists the names visible from that test, a
  "did you mean" suggestion and the files that failed to load.

Strategies files are loaded lazily through pytest's own importer: when pytest
collects a test module in that folder or below, closest folder first, each file once
per session. So `pytest tests/esm` does not import `tests/dma/strategies.py`, unless
a test asks for a name that no folder on its path registers. Relative imports work
inside strategies files in package folders, and `--list-strategies` loads every
file. Do not rely on side effects of importing a strategies file.

## Building a Parameter

```python
from pytest_strategy import Parameter, RNGInteger, TestArg

def lo_below_hi(v):
    return v.lo <= v.hi                     # v[0] <= v[1] works too

def ranges():
    return Parameter(
        TestArg("lo", rng_type=RNGInteger(0, 100)),
        TestArg("hi", rng_type=RNGInteger(0, 100)),
        TestArg("mode", value="fast"),                  # fixed value
        directed_vectors={"empty": {"lo": 5, "hi": 5, "mode": "fast"}},
        test_vectors={"bug_1234": (0, 100, "fast")},    # only with --vector-mode=test
        vector_constraints=[lo_below_hi],               # named after the function
        nsamples=25,                                    # --nsamples=N overrides it
        max_retries=100,                                # redraws per row before failing
    )
```

- A vector gives one value per argument: a dict of argument names to values (any
  order; a missing or misspelled key fails with "did you mean"), a tuple or list in
  argument order, or a namedtuple whose fields are the argument names. Tuple and
  dict vectors can be mixed. For a one-argument strategy write `{"n": -2}`; a bare
  `-2` or `"a"` fails with a hint. A dict value for that one argument is written
  `{"cfg": {"a": 1}}`.
- `pytest.param({"n": -2}, marks=pytest.mark.xfail)` marks one row (no `id=`).
- Vectors are used as written: predicates, constraints and validators do not
  check them, so keep them valid yourself.
- A predicate on an RNG type filters one argument
  (`RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)`); `vector_constraints`
  relate arguments. A constraint gets the row as a `Vector`, a tuple whose fields
  are the argument names (`v.lo`), and returns whether to keep it. They run in
  order, so one can rely on those before it. A constraint only reads the row: one
  that draws (`RNG.integer()`) gets a warning.
- Constraints are named: by the keys of a dict (`{"ordered": lambda v: v.lo <= v.hi}`),
  or in a list by each function's name (a lambda becomes `constraint_0`, so use a
  dict for lambdas). Two functions with one name, such as closures from one
  helper, fail: give them names with a dict. `--strategy-constraint-off=lo_below_hi`
  (or `ranges:lo_below_hi`, in that strategy only) turns one off for a run. When
  retries run out, the error counts the rejections per name and shows the first
  row each constraint rejected.
- Argument names must be unique identifiers that are not keywords and do not start
  with `_`.

RNG types (all importable from `pytest_strategy`):

| Type | Draws | Example |
| --- | --- | --- |
| `RNGInteger` / `RNGFloat` | a number in `[min, max]` | `RNGInteger(0, 255)` |
| `RNGBoolean` | `True` with a probability | `RNGBoolean(true_probability=0.8)` |
| `RNGChoice` | one of a list | `RNGChoice(["a", "b"])` |
| `RNGEnum` | an Enum member, optional weights and predicate | `RNGEnum(Mode, weights={Mode.A: 3, Mode.B: 1})` |
| `RNGString` | a string | `RNGString(min_length=1, max_length=8, charset="ab")` |
| `RNGWeightedInteger` / `RNGWeightedFloat` | a range chosen by weight | `RNGWeightedInteger({(0, 9): 0.9, (10, 99): 0.1})` |
| `Series` | values in declaration order | `Series([1, 2, 4, 8])` |
| `RNGSequence` | values in random order | `RNGSequence(channels)` |

Bad arguments (`min > max`, empty choices, all-zero weights, an empty sequence)
raise `RNGValueError` (a `ValueError`) when the type is built, so a broken
strategy fails at collection. A custom type subclasses `RNGType` (see Traps).

## Series vs RNGSequence

Both walk a fixed list of values; `RNG` in the name means random order.

| | `--nsamples=K` | `--nsamples=auto` |
| --- | --- | --- |
| `Series` | cycles through the values in order (first K when K < len) | every combination, in order (leftmost argument slowest) |
| `RNGSequence` | K random picks, like `RNGChoice` | a random permutation, each value once |

- Other arguments get fresh random values for each combination.
- `per_sequence_samples=True` on the `Parameter` gives K random rows for each
  combination of the `Series`/`RNGSequence` arguments (2 devices x K) instead of
  K in total, in declaration order for both types. It has no effect under `auto`.
- Pass a list or tuple, never a `set` (its order is not reproducible and is
  rejected): `Series(sorted(names))`.
- Exhaustive generation above 100,000 rows fails collection before any row is
  built: the combinations under `--nsamples=auto`, or combinations x K with
  `per_sequence_samples=True`. Raise the limit with `Parameter(max_exhaustive=...)`
  or the `strategies_max_exhaustive` ini option, or reduce the combinations.

## Values that come from configuration: ctx, strategies_ctx and get_context

Factories run at collection, before any fixture exists, so they cannot use
fixtures. When the vectors depend on configuration (a testbench file named on the
command line, say), return it from the context hook in a `conftest.py` and give the
factory a `ctx` parameter:

```python
# conftest.py (rootdir)
import pytest

def pytest_addoption(parser):
    parser.addoption("--tb-config", default="testbench.json")

@pytest.hookimpl(optionalhook=True)   # keeps conftest.py working without the plugin
def pytest_strategies_context(config):
    return load_testbench(config.getoption("--tb-config"))  # your project's loader
```

```python
# tests/esm_strategies.py
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

@register("esm_rw")
def esm_rw(ctx):
    channels = [p.channel for p in ctx.peripherals if p.kind == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=Series(channels, skip_if_empty="no Esm channel")),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        per_sequence_samples=True,     # nsamples writes for each channel
    )
```

- **A context per folder.** Each test gets the context of its own folder: the
  nearest `conftest.py` whose hook returns something other than `None` answers (a
  plugin's hook only where no `conftest.py` does). So `tests/board_a/conftest.py`
  and `tests/board_b/conftest.py` can each return their own board's
  configuration, and one shared strategy (in `tests/`) gives each folder's tests
  that folder's channels. A factory registered in another folder gets the test's
  folder's context.
- **The same object in fixtures.** Build the live testbench on the object the
  factories got instead of parsing the configuration again:

  ```python
  # conftest.py, when the whole run has one context
  @pytest.fixture(scope="session")
  def tb(strategies_ctx):              # a session fixture of the plugin
      return Testbench(strategies_ctx)
  ```

  ```python
  # tests/board_a/conftest.py, a folder with its own pytest_strategies_context
  from pytest_strategy import get_context

  @pytest.fixture(scope="session")
  def tb(request):
      return Testbench(get_context(request.config, __file__))
  ```

  `strategies_ctx` has one value per session, so it fails each test that uses it
  when those tests are in folders with different contexts: with a context per
  folder, define `tb` in each folder's `conftest.py` with `get_context()`.
  `get_context(config, path)` returns the context of the folder of `path` (a file
  or folder); call it from fixtures or hooks.
- Each implementation runs at most once per session, the first time a factory
  with `ctx` (or `strategies_ctx` or `get_context()`) needs it. Factories without
  `ctx` never trigger it. With no hook result, `ctx` keeps its default, or is
  `None`. A `wrapper=True` implementation must return a new object
  (`{**ctx, "extra": 1}`), never change the one it receives.
- After the collection the plugin prints `pytest-strategies: context 976bcfdf`
  (`contexts conftest.py ..., tests/board_a/conftest.py ...` with several), a hash
  of each context taken when the hook returned it (sets sorted, rootdir paths
  relative, pydantic `Field(exclude=True)` left out), so two runs can tell whether
  they built the same one. Keep volatile values (temp paths, times) out of it.
- Under pytest-xdist every worker calls the hook, so it must return the same data
  in each: a run whose workers built different contexts, or whose factories drew
  different values (global `random`, `list()` of a set), fails with exit code 4
  and names the context or strategy.
- `skip_if_empty="reason"` (keyword-only) turns an empty `Series`/`RNGSequence`
  into one skipped row (test ID `[skipped]`) instead of a collection error.

## Running, selecting and reproducing

| Option | Effect |
| --- | --- |
| `--rng-seed=S` | use seed S (otherwise a new seed each run, but `--lf` and `--sw` reuse the failed run's) |
| `--nsamples=N` / `--nsamples=auto` | random rows per strategy (overrides `Parameter(nsamples=)`), or enumerate `Series`/`RNGSequence` |
| `--vector-mode=MODE` | `all` (default), `random_only`, `directed_only`, `mixed`, `test` |
| `--vector-name=NAME` / `--vector-index=I` | only that directed vector |
| `--strategy-constraint-off=[STRATEGY:]NAME` | turn a named constraint off for this run (comma-separated, repeatable) |
| `-k EXPR`, node IDs | select rows by name (see "Test IDs and selecting rows") |
| `--list-strategies` | list the registered names and exit |

`-v` also prints per-strategy row counts, where `nsamples` came from, the
rejections per constraint and the constraints turned off. With
`--strategy-constraint-off` the header says `pytest-strategies: constraints off:
...`; a name that matches no constraint stops a whole-suite run with "did you mean".

To reproduce a failure:

1. Read the failed row's `pytest-strategies` section, under its traceback in the
   log: `strategy`, `vector` (`rand-3 (random row 3)`), `values` (one argument per
   line), `seed`, `context` (when the factory took `ctx`) and `rerun`, the command.
   `pytest-strategies: failed rows:`, after `reproduce with --rng-seed=S`, lists
   the same commands (also under `-q`).
2. Run that command from the folder the failing run started in, as it is:
   `pytest 'tests/dma/test_dma.py::test_burst[rand-3]' --rng-seed=S`, plus the
   run's `--nsamples`, `--vector-mode`, `-c`, `--rootdir`, `-o strategies_*` and
   `--strategy-constraint-off` when it had them. The node ID names the row, and the
   seed and options give its values, whether the row runs alone or in the suite.
3. If the values differ, compare the `rootdir:` line of both logs (the test's path
   relative to the rootdir is part of its random stream; without an ini file the
   rootdir depends on where pytest is run from) and the `context` lines when the
   factories take `ctx`. The plugin version must match too.
4. Locally, `pytest --lf` (or `--sw`) without `--rng-seed` reruns the failed rows
   with the seed they failed under: a second header line says
   `seed reused from the failed run for --lf`, ending with
   `recorded with --nsamples=13` when the failed run had options this one lacks
   (add them). Failed rows recorded under another seed are deselected, and the run
   ends with the `pytest --lf --rng-seed=S ...` command that reruns them. The paths
   you give (not `-k`) choose which failed rows' seed is reused.
5. Once fixed, add the failing values as a directed vector
   (`"bug_1234": {"addr": 4096, "length": 17}`) so they run every time, under any
   seed: `-k bug_1234` or `--vector-name=bug_1234` runs it alone, as
   `test_burst[directed-bug_1234]`.

Each item also carries its row: `item.stash[VECTOR_KEY]` (from `pytest_strategy`)
is a `VectorInfo` with `strategy`, `kind`, `name`, `index`, `values` (a `Vector`),
`id`, `seed` and `context`, for hooks, fixtures and reports
(`request.node.stash.get(VECTOR_KEY, None)`).

## Traps

- **Fixtures cannot reach factories.** Factories run at collection. Use the
  `ctx` hook for configuration, and keep runtime objects (connections, devices) as
  fixtures built on the same object: `strategies_ctx`, or
  `get_context(request.config, __file__)` in a folder with its own hook.
- **Select rows by name, not by value.** IDs carry the row's name, so `-k zeros`,
  `-k "not rand-"`, `--deselect` and node IDs keep working from one seed to the
  next, but no drawn value appears in an ID: to find a failing value, read the
  row's `pytest-strategies` section. `-k` cannot use `=`; select `ch=2` rows by
  node ID.
- **Plain `random` is not seeded by the plugin.** `random.randint()` in a factory,
  at the top of a strategies file or in a test body is not reproduced by
  `--rng-seed`. Draw through the RNG types, the `RNG.*` helpers
  (`RNG.integer(0, 9)`), the factory's `rng` or `RNG.generator()`. Those follow
  the seed on streams of their own: in a factory, a strategies file, a test
  module's top level, a fixture and each phase of a test. To seed plain `random`
  per test, add an autouse fixture in `conftest.py`:

  ```python
  import random
  import pytest
  from pytest_strategy import RNG

  @pytest.fixture(autouse=True)
  def _seed_random(request):
      random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")
  ```

- **Draw only inside an RNG type's `generate()`.** A custom `RNGType` implements
  `generate()` and `python_type`, draws inside `generate()` from
  `RNG.generator()` or the `RNG.*` helpers, never from the factory's `rng` kept on
  the instance, and keeps no state between calls (a counter makes row k depend on
  the rows before it). A test whose rows draw outside the arguments' streams (that,
  or a constraint that draws) gets a `PytestStrategiesWarning` "something drew from
  the plugin's generator while the rows were generated".
- **Draws that do not follow the seed:** a `functools.cache` factory runs once,
  for the first test that uses it, so it must not draw; code run when a
  `conftest.py` is imported, and a strategies file that a `conftest.py` imports at
  its top, draw before the seed is set. Move such draws into the context hook, a
  fixture or a strategies file loaded by the plugin.
- **Record mode.** A test can take the row as one dataclass instead of one
  parameter per argument: `def test_p(point: Point)`. It applies when neither the
  test nor any fixture it uses asks for an argument name, and exactly one
  parameter is annotated with a dataclass whose `__init__` fields are exactly the
  argument names. A fixture that takes the arguments and builds the object keeps
  the strategy in named mode. Mismatched fields fail collection with the missing
  and extra names; NamedTuple, TypedDict and pydantic models fail with "not
  supported yet". With string annotations, define the dataclass at module level.
- **Shadowing.** A subfolder that registers a name hides the parent's strategy of
  that name for its tests. Pass the factory itself when a test needs a specific one.
- **Over-constrained strategies** fail with "Could not generate random row K after
  max_retries=N draws" ("Could not generate valid vector" for `Series`
  combinations), followed by the rejections per constraint name and the first row
  each rejected. Narrow the ranges rather than filtering most draws.
- **`--vector-name`/`--vector-index` select directed vectors only.** Strategies
  without that vector yield an empty parameter set (their tests are skipped). If no
  strategy has it, the run stops with a usage error. `--vector-mode=directed_only`
  or `test` skips the tests of a strategy without such vectors the same way
  ("got empty parameter set").
- **Warnings as errors.** Projects with `filterwarnings = error` turn a
  `PytestStrategiesWarning` (for example a skipped `Series` combination) into a
  failure. Import it from `pytest_strategy`.
- **Renames change values.** A row's values depend on the seed, the strategy name,
  the test's node ID (its path relative to the rootdir, class and name), the row and
  the argument's name. Renaming any of them gives other values for the same seed;
  turn values worth keeping into directed vectors first.
- Do not call a factory or `Parameter.generate_vectors()` from a test to get rows;
  decorate the test with `@strategy` so seeding, IDs and CLI options apply.

## Upgrading from 3.x

- **Removed in 4.0:** tuple-returning factories (return a `Parameter`),
  `TestArg(directed_values=..., test_values=...)` (use `Parameter(directed_vectors=,
  test_vectors=)`), `RNG.set_max_retries()` (use `Parameter(max_retries=)`),
  `configure()` and `Strategy.set_config()` (delete the call). The full table is in
  references/api.md, section 19.
- Test IDs name the rows instead of showing their values, and the same seed gives
  other random values than 3.x: update `-k` and `--deselect` lists, and record
  seeds again. Factory parameters other than `nsamples`, `ctx`, `rng` and `options`
  need a default, and `TestArg`'s options after `rng_type` are keyword-only.
- `Strategy.register`/`Strategy.strategy` still work as aliases; write new code with
  `register`/`strategy`.

## Checking your work

Run the tests twice without a seed (new rows each time), with `--nsamples=auto` when
there are `Series`/`RNGSequence` arguments, and with `--vector-mode=directed_only`
when the strategy has directed vectors (without any, that mode leaves the test
skipped with "got empty parameter set"). Run one row by node ID with the seed of the
first run to check that it reproduces, and run with `-n 2` when the project uses
pytest-xdist. After upgrading the library, refresh this skill with
`pytest-strategies skill install`.
