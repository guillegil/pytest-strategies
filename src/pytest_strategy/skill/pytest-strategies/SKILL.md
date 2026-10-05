---
name: pytest-strategies
description: Write, fix and debug pytest tests that use the pytest-strategies plugin (import name pytest_strategy) for constrained-random and directed test parametrization. Use this skill whenever code imports pytest_strategy or uses register/strategy, Parameter, TestArg, Vector, RNGInteger, RNGChoice, RNGEnum, Series, RNGSequence or another RNG type; when creating or editing strategies.py or *_strategies.py files; when reusing or extending a strategy (Parameter.extend); when adding directed or test vectors, named constraints or config-driven strategies (the pytest_strategies_context hook, ctx, strategies_ctx, get_context, skip_if_empty); when choosing --nsamples, --vector-mode, --strategy-constraint-off, --strategy-details or -k on row IDs such as test_x[rand-3]; and when a failed generated test has to be reproduced from its rerun command, --rng-seed or --lf. Use it even when the user only says randomized tests, test vectors or strategies in a project that depends on pytest-strategies.
---

# pytest-strategies

Documents pytest-strategies 4.1.1.

pytest-strategies (import name `pytest_strategy`) parametrizes pytest tests from
*strategies*: factories that return a `Parameter` describing each test argument (a
random generator, a fixed value or an ordered series) plus named edge cases. Rows
are generated at collection from a per-run seed, so each run explores new values;
each row's test ID is a name that is the same for every seed, and a failed run
prints the command that reruns each failed row with its values.

This file covers the patterns and the traps. For full signatures, every option,
the error messages and what changed from 3.x, read
[references/api.md](references/api.md) (its table of contents lists 20 sections).

## Before writing anything

- Check the installed version: `python -c "import pytest_strategy as p; print(p.__version__)"`.
  `Parameter.extend()` and `--strategy-details` need 4.1. Before 4.0, test IDs
  showed values (`test_x[addr=0,len=1]`) and factories received `nsamples` by
  position; references/api.md, section 19 lists the differences.
- Run `pytest --list-strategies` to see the registered names. It collects the tests
  first, so a broken factory shows as a collection error (exit code 2).
- Look for strategies files next to the tests and in parent folders, and for a
  `pytest_strategies_context` hook in `conftest.py` files. Reuse or extend an
  existing strategy before writing a new one.

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
            "zeros": {"addr": 0, "length": 1},
            "page_end": {"addr": 4092, "length": 4},
        },
        vector_constraints={                        # each gets the row as a Vector
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

`pytest tests/dma` runs `test_burst[directed-zeros]`, `test_burst[directed-page_end]`,
then `test_burst[rand-0]` to `test_burst[rand-9]`, with new random values each run.

`@strategy(dma_burst)` passes the factory itself: no registration or name lookup
is needed. Import it relatively in a package folder (`from .strategies import
dma_burst`); the test gets the module the plugin loaded. Its random rows are keyed
by a name its own file registers it under, else by its qualified name
(references/api.md, section 2).

## How a factory is called

- Once per test that uses it, at collection (`pytest_generate_tests`), never at
  import. It returns a `Parameter`; anything else fails collection.
- It receives, by name and only if it declares them: `nsamples` (an int, or `"auto"`
  under `--nsamples=auto`), `ctx` (the test folder's context, see below), `rng`
  (the plugin's `random.Random`) and `options` (a frozen `StrategyOptions`). Most
  factories declare none.
- Any other parameter needs a default (`def f(n)` fails: rename it `nsamples`).
  `*args`/`**kwargs` receive nothing; `base`, `config` and `request` are reserved.
  Fixtures are never factory inputs.
- The test takes every `TestArg` name as a parameter (checked at collection; turn
  off with `@strategy(..., validate_signature=False)`), or the whole row as one
  dataclass (record mode, see Traps). Its other parameters are fixtures, and a
  fixture may take strategy arguments too: `def mask(width, signed)` gets each row's.

## Building a Parameter

```python
from pytest_strategy import Parameter, RNGInteger, TestArg

def lo_below_hi(v):
    return v.lo <= v.hi

def ranges():
    return Parameter(
        TestArg("lo", rng_type=RNGInteger(0, 100)),
        TestArg("hi", rng_type=RNGInteger(0, 100)),
        TestArg("mode", value="fast"),                  # fixed value
        directed_vectors={"empty": {"lo": 5, "hi": 5, "mode": "fast"}},
        test_vectors={"bug_1234": (0, 100, "fast")},    # only with --vector-mode=test
        vector_constraints=[lo_below_hi],               # a list: named after each function
        nsamples=25,                                    # an integer --nsamples overrides it
        max_retries=100,                                # draws per row before failing
    )
```

- **Vectors** give one value per argument: a dict by name (preferred; a typo fails
  with "did you mean"), a tuple in argument order, or a namedtuple with the argument
  names as fields. A one-argument vector is `{"n": -2}`; a bare `-2` fails.
  `pytest.param({"n": -2}, marks=pytest.mark.xfail)` marks one row; it takes no
  `id=` (the vector's name is its ID), and a tuple inside it is one value.
- **Vectors are used as written**: predicates, constraints and validators never
  check them.
- **Predicates vs constraints.** A predicate filters one argument
  (`RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)`). A constraint relates
  arguments: it receives the row as a `Vector` (`v.lo`, or `v[0]`), runs in order
  after the ones before it, and returns whether to keep the row. Constraints only
  read the row; one that draws gets a warning.
- **Constraint names** come from dict keys or, in a list, from each function's
  `__name__`; a lambda in a list becomes `constraint_<i>`, so put lambdas in a dict.
  Two functions with one name (closures from one helper) fail: name them with a dict.
- Argument names are unique identifiers, not keywords, not starting with `_`.

RNG types (all importable from `pytest_strategy`; bad arguments such as `min > max`
or empty choices raise `RNGValueError` when the type is built):

| Type | Draws | Example |
| --- | --- | --- |
| `RNGInteger` / `RNGFloat` | a number in `[min, max]` | `RNGInteger(0, 255)` |
| `RNGBoolean` | `True` with a probability | `RNGBoolean(true_probability=0.8)` |
| `RNGChoice` | one of a list | `RNGChoice(["a", "b"])` |
| `RNGEnum` | an Enum member, optional weights | `RNGEnum(Mode, weights={Mode.A: 3, Mode.B: 1})` |
| `RNGString` | a string | `RNGString(min_length=1, max_length=8, charset="ab")` |
| `RNGWeightedInteger` / `RNGWeightedFloat` | a range chosen by weight | `RNGWeightedInteger({(0, 9): 0.9, (10, 99): 0.1})` |
| `Series` | values in declaration order | `Series([1, 2, 4, 8])` |
| `RNGSequence` | values in random order | `RNGSequence(channels)` |

### Series and RNGSequence

| | `--nsamples=K` | `--nsamples=auto` |
| --- | --- | --- |
| `Series` | cycles through the values in order | every combination, in order (leftmost argument slowest) |
| `RNGSequence` | K random picks | a random permutation, each value once |

- Other arguments get fresh random values for each combination.
- `per_sequence_samples=True` gives K random rows per combination of the
  `Series`/`RNGSequence` arguments instead of K in total (no effect under `auto`).
- Pass a list or tuple, never a `set` (rejected: its order is not reproducible).
- `skip_if_empty="reason"` (keyword-only) turns an empty sequence into one skipped
  row (`[skipped]`) instead of a collection error.
- Above 100,000 exhaustive rows collection fails before generating; raise it with
  `Parameter(max_exhaustive=...)` or the `strategies_max_exhaustive` ini option.

## Reusing a strategy

Prefer reuse to copying. In order of preference:

1. **By name** from any folder at or below the one that registers it.
2. **By the factory**: `@strategy(dma_burst)`, which also avoids shadowing.
3. **By extension** (4.1): call the base factory and `extend()` its `Parameter`.
   `extend()` returns a new `Parameter`; the base is not changed.

```python
from pytest_strategy import RNGChoice, RNGInteger, TestArg, register
from .strategies import dma_burst

@register("dma_short")
def dma_short():
    return dma_burst().extend(TestArg("length", rng_type=RNGInteger(1, 8)), nsamples=5)

@register("dma_prio")
def dma_prio():
    return dma_burst().extend(
        TestArg("prio", rng_type=RNGChoice([0, 1, 2])),   # new name: added last
        defaults={"prio": 0},                              # its value in the kept vectors
        directed_vectors={"urgent": {"addr": 0, "length": 4, "prio": 2}},
        vector_constraints={"no_4k_cross": None},          # None removes by name
    )
```

- A `TestArg` with a known name replaces that argument in place; vectors keep their
  values. A new name is appended, and every kept vector needs its value from
  `defaults` or the `TestArg`'s fixed `value=`; otherwise `extend()` raises and
  names the vector.
- Vectors and constraints given as dicts are added (new name), replaced in place
  (known name) or removed (`None`; an unknown name raises). A list of constraints
  is appended. Unlisted settings (`nsamples`, `max_retries`, `ids`, ...) are kept.
- A base factory with inputs gets them passed on: `def b(ctx): return a(ctx).extend(...)`.
- The new strategy has its own name, so its random rows differ from the base's.
  Stacking `@strategy("a")` and `@strategy("b")` on one test instead crosses their
  rows (`test_x[directed-zeros-rand-1]`).

## Where strategies live

A strategies file is named `strategies.py`, `strategy.py`, `*_strategies.py`,
`*_strategy.py` or `test_strategies.py` and registers with a literal name:
`@register("name")`. `register("x")(fn)` or `@register(NAME)` alone is not detected.
A `conftest.py` or test module may register too.

- **Names are scoped like fixtures.** A name is visible in its folder and below.
  The lookup walks from the test's folder up to the `testpaths` entry (or rootdir)
  that contains it; the nearest registration wins. A strategies file above that
  point is never loaded: put shared strategies in `tests/strategies.py` or the
  rootdir `conftest.py`.
- A name no folder on the path defines, registered exactly once elsewhere, is found
  too; registered in several such folders, the lookup fails and lists them. Two
  factories with one name in one folder are a usage error.
- **Imports.** Several folders with `strategies.py` need `__init__.py` files and
  relative imports (`from .strategies import X`), or distinct file names: a plain
  `from strategies import X` gets whichever folder's file was imported first.
- Files load lazily, once per session, when pytest collects a test module in that
  folder or below. Do not rely on import side effects.

## Values from configuration: ctx

Factories run before fixtures exist. When vectors depend on configuration, return it
from the context hook and declare `ctx`:

```python
# conftest.py (rootdir)
import pytest

def pytest_addoption(parser):
    parser.addoption("--tb-config", default="testbench.yaml")

@pytest.hookimpl(optionalhook=True)   # keeps conftest.py working without the plugin
def pytest_strategies_context(config):
    return load_testbench(config.getoption("--tb-config"))  # your project's loader

@pytest.fixture(scope="session")
def tb(strategies_ctx):               # the same object the factories received
    return Testbench(strategies_ctx)
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
        per_sequence_samples=True,
    )
```

- **One context per folder.** The nearest `conftest.py` whose hook returns non-`None`
  answers for a test, so `tests/board_a/conftest.py` and `tests/board_b/conftest.py`
  can return different boards to one shared strategy.
- **Fixtures.** `strategies_ctx` has one value per session and fails when its tests
  have different contexts; then build the fixture per folder with
  `get_context(request.config, __file__)`.
- Each hook runs at most once per session, the first time something needs it. A
  `wrapper=True` hook returns a new object, never mutates the one it received.
- After collection the run prints `pytest-strategies: context 976bcfdf`, a hash
  that tells two runs' contexts apart; keep volatile values (temp paths, times) out.
- Under pytest-xdist every worker calls the hook and must get the same data, or the
  run fails with exit code 4 naming the context or strategy.

## Running, selecting and reproducing

| Option | Effect |
| --- | --- |
| `--rng-seed=S` | use seed S (default: new each run; `--lf`/`--sw` reuse the failed run's) |
| `--nsamples=N` / `--nsamples=auto` | random rows per strategy, or enumerate sequences |
| `--vector-mode=MODE` | `all` (default), `random_only`, `directed_only`, `mixed`, `test` |
| `--vector-name=NAME` / `--vector-index=I` | only that directed vector |
| `--strategy-constraint-off=[STRATEGY:]NAME` | turn a named constraint off for this run |
| `--strategy-details` | show each failed row's values, seed and rerun command under its traceback (4.1) |
| `--list-strategies` | list the registered names and exit |
| `-v` | per-strategy row counts, `nsamples` source, rejections per constraint |

**Test IDs** name rows, the same for every seed: `[directed-zeros]`, `[test-max]`
(only under `--vector-mode=test`), `[rand-3]`, `[ch=2-rand-1]` (a `Series` value),
`[skipped]`. Stacked decorators join with `-`, nearest the `def` first.

- `-k zeros` selects a vector by name, `-k directed-` every directed row,
  `-k "not rand-"` no random row. Keep the `-`: `-k` matches substrings of the whole
  node ID (`-k rand` also matches `test_random_io`).
- `-k "rand-3"` also matches `rand-30`, and `-k` cannot contain `=`: select one row
  by node ID, `pytest "tests/test_esm.py::test_esm[ch=2-rand-1]"`.
- Name vectors like identifiers (`bug_1234`): `-k` accepts only letters, digits and
  `_ - . : / + \ [ ]`.
- IDs hold no values. `-o strategies_ids=values` or `Parameter(ids="values")`
  restores value-based IDs, which change with the seed.

**To reproduce a failure:**

1. A failed run prints `pytest-strategies: reproduce with --rng-seed=S` and, under
   `failed rows:`, one rerun command per failed row (10 below `-v`). Run a command
   as printed, from the folder the run started in; it carries the seed and every
   option that shapes the rows.
2. Locally, `pytest --lf` (or `--sw`) without `--rng-seed` reruns the failed rows
   with the seed they failed under.
3. To see the values, add `--strategy-details`: each failed row then gets a
   `pytest-strategies` section (strategy, vector, values, seed, context, rerun).
   It prints one block per failed row, so narrow first:
   `pytest --lf --strategy-details -x`.
4. If a rerun gives other values, compare the `rootdir:` lines (the test's path
   relative to the rootdir is part of its stream), the `context` fingerprints and
   the plugin version.
5. Once fixed, pin the values as a directed vector (`"bug_1234": {"addr": 4096,
   "length": 17}`) so they run under every seed.

`item.stash[VECTOR_KEY]` (from `pytest_strategy`) is each item's `VectorInfo`:
`strategy`, `kind`, `name`, `index`, `values`, `id`, `seed`, `context`.

## Traps

- **Fixtures cannot reach factories.** Factories run at collection. Put
  configuration in the `ctx` hook and build runtime objects as fixtures on the same
  object: `strategies_ctx`, or `get_context(request.config, __file__)` in a folder
  with its own hook.
- **Select rows by name, not by value.** `-k zeros`, `-k "not rand-"`, `--deselect`
  and node IDs keep working across seeds; no drawn value appears in an ID.
- **Plain `random` does not follow `--rng-seed`.** Draw through the RNG types, the
  `RNG.*` helpers (`RNG.integer(0, 9)`), the factory's `rng` or `RNG.generator()`.
  To seed plain `random` per test, add an autouse fixture:

  ```python
  import random
  import pytest
  from pytest_strategy import RNG

  @pytest.fixture(autouse=True)
  def _seed_random(request):
      random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")
  ```

- **Custom RNG types draw only inside `generate()`**, from `RNG.generator()` or the
  `RNG.*` helpers, never from a stored `rng`, and keep no state between calls.
- **Draws that escape the seed:** a `functools.cache` factory (runs once, so must
  not draw), code at the top of a `conftest.py`, and a strategies file that a
  `conftest.py` imports at its top all draw before the seed is set. Move such draws
  into the context hook, a fixture or a plugin-loaded strategies file.
- **Record mode.** `def test_p(point: Point)` takes the row as one dataclass when no
  parameter or fixture of the test asks for an argument name and exactly one
  parameter is annotated with a dataclass whose `__init__` fields are exactly the
  argument names. NamedTuple, TypedDict and pydantic models fail ("not supported yet").
- **Shadowing.** A subfolder registering a name hides the parent's for its tests;
  pass the factory itself when a test needs a specific one.
- **Over-constrained strategies** fail with "Could not generate random row K after
  max_retries=N draws" plus rejections per constraint. Narrow the ranges rather
  than filtering most draws.
- **`--vector-name`, `--vector-index`, `--vector-mode=directed_only` or `test`**
  leave strategies without such vectors with an empty parameter set (skipped);
  a `--vector-name` no strategy has is a usage error.
- **`filterwarnings = error`** turns a `PytestStrategiesWarning` (a skipped `Series`
  combination, a drawing constraint) into a failure.
- **Renames change values.** A row's values depend on the seed, the strategy name,
  the test's node ID, the row and the argument name. Pin values worth keeping as
  directed vectors before renaming.
- **Never call a factory or `generate_vectors()` from a test** to get rows; decorate
  it with `@strategy` so seeding, IDs and options apply. Calling a factory inside
  another factory, to `extend()` it, is fine.

## Upgrading from 3.x

- **Removed in 4.0:** tuple-returning factories (return a `Parameter`),
  `TestArg(directed_values=..., test_values=...)` (use `Parameter(directed_vectors=,
  test_vectors=)`), `RNG.set_max_retries()` (use `Parameter(max_retries=)`),
  `configure()` and `Strategy.set_config()` (delete the call). The full table is in
  references/api.md, section 19.
- IDs name rows instead of showing values, and the same seed gives other random
  values than 3.x: update `-k` and `--deselect` lists and record seeds again.
  Factory parameters other than `nsamples`, `ctx`, `rng` and `options` need a
  default, and `TestArg` options after `rng_type` are keyword-only.
- `Strategy.register`/`Strategy.strategy` remain as aliases; write
  `register`/`strategy`.
- 4.1 only adds: `Parameter.extend()`, and `--strategy-details`, since failed rows
  no longer print a section each by default (put it in `addopts` for the 4.0 output).

## Checking your work

Run the tests twice without a seed (new rows each time); with `--nsamples=auto` when
there are `Series`/`RNGSequence` arguments; with `--vector-mode=directed_only` when
the strategy has directed vectors; one row by node ID with the first run's seed, to
check it reproduces; and with `-n 2` when the project uses pytest-xdist. After
upgrading the library, refresh this skill with `pytest-strategies skill install`.
