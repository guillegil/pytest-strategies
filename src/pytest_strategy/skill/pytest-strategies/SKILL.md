---
name: pytest-strategies
description: Write, fix and debug pytest tests that use the pytest-strategies plugin (import name pytest_strategy) for constrained-random and directed test parametrization. Use this skill whenever code imports pytest_strategy or uses register/strategy, Strategy.register/Strategy.strategy, Parameter, TestArg, RNGInteger, RNGChoice, RNGEnum, Series, RNGSequence or another RNG type; when creating or editing strategies.py or *_strategies.py files; when adding directed or test vectors, constraints or config-driven strategies (the pytest_strategies_context hook, ctx, skip_if_empty, per_sequence_samples); when choosing --nsamples, --vector-mode, --vector-name or --vector-index; and when a test with generated parameters fails and has to be reproduced with --rng-seed. Use it even when the user only says randomized tests, test vectors or strategies in a project that depends on pytest-strategies.
---

# pytest-strategies

Documents pytest-strategies 3.0.0.

pytest-strategies (import name `pytest_strategy`) parametrizes pytest tests from
*strategies*: factory functions that return a `Parameter` describing each test
argument (a random generator, a fixed value or an ordered series) plus named edge
cases. Rows are generated at collection time from a per-run seed, so each run
explores new values and any run can be replayed exactly.

This file covers the patterns and traps behind most tasks. For full signatures,
every option, the error messages and upgrade notes from 2.x, read
[references/api.md](references/api.md).

## First, look at the project

- Check the version: `python -c "import pytest_strategy as p; print(p.__version__)"`.
  On 2.x there are no plain `register`/`strategy` functions, no folder-scoped
  names and no factory references: use `@Strategy.register`/`@Strategy.strategy`,
  and see "Deprecations and upgrading from 2.x" in the reference for what differs.
- List what already exists: `pytest --list-strategies` prints every registered
  name and exits. Reuse or extend a strategy before writing a new one.
- Look for strategies files next to the tests and in parent folders, and for a
  `pytest_strategies_context` hook in the `conftest.py` files (each folder can
  have its own).

## The basic pattern

```python
# tests/payments/strategies.py
from pytest_strategy import Parameter, RNGChoice, RNGInteger, TestArg, register

@register("payment")
def payment(nsamples):
    return Parameter(
        TestArg("amount", rng_type=RNGInteger(1, 10_000)),
        TestArg("currency", rng_type=RNGChoice(["EUR", "USD", "JPY"])),
        directed_vectors={            # always run first, one value per TestArg, in order
            "minimum": (1, "EUR"),
            "maximum": (10_000, "JPY"),
        },
    )
```

```python
# tests/payments/test_charge.py
from pytest_strategy import strategy

@strategy("payment")                         # looked up from this folder upward
def test_charge(amount, currency, gateway):  # gateway is an ordinary fixture
    assert gateway.charge(amount, currency).ok
```

A factory can also be passed directly. It then needs no registration and no name
lookup, and editors can jump to it:

```python
from pytest_strategy import Parameter, RNGInteger, TestArg, strategy

def digits():                                # no parameters is fine
    return Parameter(TestArg("n", rng_type=RNGInteger(0, 9)))

@strategy(digits)
def test_digit(n):
    assert 0 <= n <= 9
```

To pass a factory from a strategies file, import it (in a package folder:
`from .strategies import payment`, then `@strategy(payment)`). The test gets the
same module object the plugin loaded.

## How a factory is called

- Once for each test that uses it, while pytest collects that test (in
  `pytest_generate_tests`), not when the file is imported.
- It receives by name, in any order, the inputs it declares: `nsamples` (an int,
  or `"auto"` under `--nsamples=auto`), `ctx` (the result of the context hook, see
  below), `rng` (`RNG.generator()`) and `options` (a frozen `StrategyOptions`).
  All are optional. It rarely needs `nsamples`: return a `Parameter` and the
  plugin generates the rows. Any other parameter needs a default (`def f(n)`
  fails), and `base`, `config` and `request` are reserved.
- It returns a `Parameter`; anything else fails collection (see Upgrading from 3.x).
- The test must take every `TestArg` name as a parameter (checked at collection;
  `@strategy(..., validate_signature=False)` turns the check off). Its other
  parameters are fixtures, as usual. Or use dataclass mode (see Traps).
- Rows: the directed vectors, then `nsamples` random rows (10 by default). Test IDs
  name the row, the same for every seed: `test_charge[directed-zeros]`,
  `test_charge[rand-3]`, and `ch=2-rand-1` when a `Series` value is enumerated.
  Select a directed vector with `-k zeros` or `--vector-name=zeros`.
  `Parameter(ids="values")` gives one strategy the 3.0 IDs built from the values,
  and `ids=fn` builds them from each row's `VectorInfo` (return None to keep one).

## Where strategies live

A strategies file is a file named `strategies.py`, `strategy.py`,
`*_strategies.py`, `*_strategy.py` or `test_strategies.py` that contains a
register decorator (`@register("name")`, `@Strategy.register(...)` or
`@<module>.register("name")`, the name written as a string literal). A
`conftest.py` can register strategies too. A strategies file that only
registers with a plain call (`register("x")(fn)`) or a name held in a variable
(`@register(NAME)`) is not loaded.

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
- Two different factories with the same name in the same folder (two files, or a
  file and `conftest.py`) stop the run with a usage error (exit code 4, or 2 under
  pytest-xdist) that names
  both files. Registering the same function again is fine.
- "Strategy 'x' not found" lists the names visible from that test, a "did you
  mean" suggestion and the files that failed to load.

Strategies files are loaded lazily through pytest's own importer: when pytest
collects a test module in that folder or below, closest folder first, each file once
per session. So `pytest tests/esm` does not import `tests/dma/strategies.py`, unless
a test asks for a name that no folder on its path registers (then every strategies
file is loaded to find it). Relative imports work inside strategies files in package
folders, and `--list-strategies` loads every file. Do not rely on side effects of
importing a strategies file.

## Building a Parameter

```python
from pytest_strategy import Parameter, RNGInteger, TestArg

def ranges(nsamples):
    return Parameter(
        TestArg("lo", rng_type=RNGInteger(0, 100)),
        TestArg("hi", rng_type=RNGInteger(0, 100)),
        TestArg("mode", value="fast"),                  # fixed value
        directed_vectors={"empty": (5, 5, "fast")},     # run in the default mode; one arg: (5,)
        test_vectors={"bug_1234": (0, 100, "fast")},    # only with --vector-mode=test
        vector_constraints=[lo_below_hi],               # each gets the row as a Vector
        nsamples=25,                                    # default count; --nsamples=N overrides it
        max_retries=100,                                # redraws per row before failing
    )

def lo_below_hi(v):
    return v.lo <= v.hi                                 # v[0] <= v[1] works too
```

- A predicate on an RNG type filters one argument
  (`RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)`); `vector_constraints`
  relate arguments. A constraint gets the row as a `Vector`, a tuple whose fields
  are the argument names. Constraints are named: by a function's name in a list
  (above: `lo_below_hi`), or by the keys of a dict
  (`{"ordered": lambda v: v.lo <= v.hi}`). When retries run out, the error counts
  by name how often each constraint rejected a row, and
  `--strategy-constraint-off=lo_below_hi` turns one off for a run. A constraint
  only reads the row: one that draws (`RNG.integer()`) gets a warning.
- Directed and test vectors are used as written. Predicates, constraints and
  validators do not check them, so keep them valid yourself.
- Argument names must be unique within a `Parameter`.

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
strategy fails at collection.

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

## Values that come from configuration: ctx and skip_if_empty

Factories run at collection, before any fixture exists, so they cannot use
fixtures. When the vectors depend on configuration (a testbench file named on the
command line, say), return it from the context hook in a `conftest.py` (the
rootdir's for the whole project) and give the factory a `ctx` parameter:

```python
# conftest.py (rootdir)
import pytest

def pytest_addoption(parser):
    parser.addoption("--tb-config", default="testbench.yaml")

@pytest.hookimpl(optionalhook=True)   # keeps conftest.py working without the plugin
def pytest_strategies_context(config):
    return load_testbench(config.getoption("--tb-config"))  # your project's loader
```

```python
# tests/esm/strategies.py
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

@register("esm_rw")
def esm_rw(nsamples, ctx):
    channels = [p.channel for p in ctx.peripherals if p.kind == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=Series(channels, skip_if_empty="no Esm peripheral")),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        per_sequence_samples=True,     # nsamples writes for each channel
    )
```

- Each test gets the context of its own folder: the nearest `conftest.py` whose
  hook returns something other than `None` answers (a plugin's hook only where no
  `conftest.py` does), so `tests/a/conftest.py` can give `tests/a` another
  testbench than the rootdir's gives the rest. A factory from another folder gets
  the test's folder's context.
- Each implementation runs at most once per session, the first time a factory
  with `ctx` (or `strategies_ctx` or `get_context()`, below) needs it. A
  `wrapper=True` one must return a new object (`{**ctx, "extra": 1}`), never
  change the one it receives. Factories
  without `ctx` never trigger it. With no hook result, `ctx` keeps its default,
  or is `None`.
- Fixtures get the same object: `def tb(strategies_ctx)` (a session fixture of
  the plugin) for the tests of one context, or in a folder with its own hook
  `get_context(request.config, __file__)` (from `pytest_strategy`) in that
  folder's `conftest.py`. `strategies_ctx` fails each test that uses it when
  those tests are in folders with different contexts.
- Under pytest-xdist every worker calls it, so it must return the same data in
  each: a run whose workers built different contexts, or whose factories drew
  different values (global `random`, `list()` of a set), fails with exit code 4
  and names the context or strategy.
- After the collection the plugin prints `pytest-strategies: context 976bcfdf`,
  a hash of the context taken when the hook returned it (sets sorted, rootdir
  paths relative, pydantic `Field(exclude=True)` left out), so two runs can tell
  whether they built the same one. Keep volatile values (temp paths, times) out
  of it. The reproduce line of a failed run ends with `(context 976bcfdf)`, and
  `item.stash[VECTOR_KEY].context` holds it for factories that received `ctx`.
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
| `--list-strategies` | list the registered names and exit |

`-v` also prints per-strategy row counts and where `nsamples` came from.

To reproduce a failure:

1. Take the command from the failing run: each failed row's `pytest-strategies`
   section, under its traceback, shows the row, its values, the seed and a
   `rerun` line, and `pytest-strategies: failed rows:` after
   `reproduce with --rng-seed=S` lists the same commands (also under `-q`). The
   seed alone is in that line and in the header (`pytest-strategies: RNG seed = S`).
2. Run the command from the folder the failing run started in. It is
   `pytest "tests/payments/test_charge.py::test_charge[rand-3]" --rng-seed=S`, plus
   the run's `--nsamples`, `--vector-mode`, `-c`, `--rootdir` and
   `--strategy-constraint-off` when it had them: the node ID names the row, and
   the seed and options give its values.
3. The same seed, rootdir and plugin version give the same rows whether you run the
   suite, one file or one test, in any order, with or without xdist. The test's path
   relative to the rootdir is part of its random stream: compare the `rootdir:` line
   of the CI log with yours, and the `pytest-strategies: context` line when the
   factories take `ctx`. Without an ini file the rootdir depends on where pytest
   is run from, so run from the same folder as CI (or add a `pytest.ini` or
   `[tool.pytest.ini_options]`).
4. Locally, `pytest --lf` (or `--sw`) without `--rng-seed` reruns the failed rows
   with the seed they failed under: a second header line says
   `seed reused from the failed run for --lf`, and ends with
   `recorded with --nsamples=13` when the failed run had options this one lacks
   (add them). Failed rows recorded under another seed are deselected, and the run
   ends with the `pytest --lf --rng-seed=S <files>` command that reruns them. The
   paths you give (not `-k`) choose which failed rows' seed is reused. When the
   reused rows are not collected (renamed, or other options), the end of the run
   gives their command too.
5. Once fixed, add the failing values as a directed vector (`"bug_1234": (17, "EUR")`;
   one argument needs a trailing comma, `(-2,)`) so they run every time, not only
   under that seed. Run it with `--vector-name=bug_1234`, which needs no seed. Its ID
   is `directed-bug_1234`.

## Traps

- **Fixtures cannot reach factories.** Factories run at collection. Use the
  `ctx` hook for configuration, and keep runtime objects (connections, devices) as
  fixtures of the test. A fixture that needs the same configuration takes the
  hook's object from `strategies_ctx`, or from `get_context(request.config,
  __file__)` in a folder with its own hook, instead of parsing it again.
- **Plain `random` is not seeded by the plugin.** Since 3.0.0 the plugin never
  calls `random.seed()`. `random.randint()` in a factory, at the top of a
  strategies file or in a test body is not reproduced by `--rng-seed`. Draw
  through the RNG types, the `RNG.*` helpers (`RNG.integer(0, 9)`) or the
  generator from `RNG.generator()`. To seed plain `random` per test, add an
  autouse fixture in `conftest.py`:

  ```python
  import random
  import pytest
  from pytest_strategy import RNG

  @pytest.fixture(autouse=True)
  def _seed_random(request):
      random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")
  ```

- **Dataclass mode.** A test can take the row as one dataclass instead of one
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
  combinations), followed by the rejections per constraint name. Narrow the ranges
  rather than filtering most draws.
- **`--vector-name`/`--vector-index` select directed vectors only.** Strategies
  without that vector yield an empty parameter set (their tests are skipped). If no
  strategy has it, the run stops with a usage error.
- **Warnings as errors.** Projects with `filterwarnings = error` turn a
  `PytestStrategiesWarning` (for example a skipped `Series` combination) into a
  failure. Import it from `pytest_strategy`.
- **Draw only inside an RNG type's `generate()`.** A custom `RNGType` draws from
  `RNG.generator()` or the `RNG.*` helpers there, never from the factory's `rng`
  kept on the instance, and keeps no state between calls. A test whose rows draw
  outside the arguments' streams (that, or a constraint that draws) gets a
  `PytestStrategiesWarning` "something drew from the plugin's generator while the
  rows were generated".
- Do not call a factory or `Parameter.generate_vectors()` from a test to get rows;
  decorate the test with `@strategy` so seeding, IDs and CLI options apply.

## Upgrading from 3.x

- **Removed in 4.0:** tuple-returning factories (return a `Parameter`),
  `TestArg(directed_values=..., test_values=...)` (use `Parameter(directed_vectors=,
  test_vectors=)`), `RNG.set_max_retries()` (use `Parameter(max_retries=)`),
  `configure()` and `Strategy.set_config()` (delete the call). The full table is in
  references/api.md, section 16.
- `Strategy.register`/`Strategy.strategy` still work as aliases; write new code with
  `register`/`strategy`.

## Checking your work

Run the tests twice without a seed (new rows each time), with `--nsamples=auto` when
there are `Series`/`RNGSequence` arguments, and with `--vector-mode=directed_only`
when the strategy has directed vectors (without any, that mode leaves the test
skipped with "got empty parameter set").
After upgrading the library, refresh this skill with `pytest-strategies skill install`.
