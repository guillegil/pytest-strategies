# pytest-strategies 🧪

**Powerful, constrained-randomized test generation for pytest.**

`pytest-strategies` extends pytest with a robust framework for defining test strategies that combine **random generation**, **directed edge cases**, and **constraints**. It bridges the gap between simple parametrization and property-based testing, giving you full control over your test data.

[![Tests](https://github.com/guillegil/pytest-strategies/actions/workflows/tests.yml/badge.svg)](https://github.com/guillegil/pytest-strategies/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **Hybrid Generation**: seamlessly mix **randomly generated** data with **directed** (hardcoded) edge cases.
- **Sequence Testing**: Use `Series` for **deterministic, ordered** value sequences (Cartesian product) or `RNGSequence` for a **randomized permutation** of the same values.
- **Type-Safe RNG**: Built-in generators for Integers, Floats, Booleans, Strings, Choices, Sequences, and **Enums**.
- **Weighted Probabilities**: Define custom distributions for Enums, Integers, and Floats.
- **Constraints & Predicates**: Filter generated values using simple lambda predicates or complex vector constraints.
- **Fixture Integration**: Works out-of-the-box with standard pytest fixtures (custom, parametrized, or built-in).
- **Reproducibility**: Deterministic generation via seed control for debugging failures.
- **CLI Control**: Filter strategies, change generation modes, or increase sample sizes directly from the command line.
- **Per-Strategy Sample Count**: Let a strategy declare its own vector count as a soft default that an integer `--nsamples` can still override.
- **Folder-Scoped Names**: Strategy names are looked up like fixtures, nearest folder first, and strategy files are imported only for the folders being tested.
- **Typed**: The package ships type hints (`py.typed`), and the decorators keep the decorated function's type.
- **Agent Skill**: `pytest-strategies skill install` gives Claude Code, Codex and other coding agents a guide to writing strategies.

## 📦 Installation

`pytest-strategies` needs Python 3.11 or later and pytest 8.4.2 or later. It is not published on PyPI yet. Install a release from GitHub:

```bash
pip install "pytest-strategies @ git+https://github.com/guillegil/pytest-strategies.git@v3.0.0"
```

or from a local clone with `pip install -e .`.

Upgrading from 2.x? The [CHANGELOG](CHANGELOG.md) lists what changed in 3.0.0 and what to do about it.

### Agent skill

The package ships a skill that teaches coding agents (Claude Code, Codex and others that read `SKILL.md` files) how to write strategies and tests with this plugin. Install it into your project:

```bash
pytest-strategies skill install            # .claude/skills/ and .agents/skills/
pytest-strategies skill install --claude   # only .claude/skills/
pytest-strategies skill install --agents   # only .agents/skills/ (--generic works too)
pytest-strategies skill install --global   # in your home folder instead
```

`python -m pytest_strategy skill install` does the same. A global Claude install honors `CLAUDE_CONFIG_DIR`. Run the command again after upgrading the package to update the skill.

## ⚡ Quick Start

Define a strategy and apply it to your test:

```python
from pytest_strategy import Parameter, RNGChoice, RNGInteger, TestArg, register, strategy

# 1. Register a strategy
@register("user_age_strategy")
def create_user_strategy(nsamples: int) -> Parameter:
    return Parameter(
        # Randomly generate ages between 0 and 100
        TestArg("age", rng_type=RNGInteger(0, 100)),
        
        # Randomly choose a user type
        TestArg("user_type", rng_type=RNGChoice(["admin", "user", "guest"])),
        
        # Always include these specific edge cases
        directed_vectors={
            "newborn": (0, "guest"),
            "centenarian": (100, "user"),
            "admin_edge": (18, "admin"),
        }
    )

# 2. Use the strategy in your test
@strategy("user_age_strategy")
def test_user_validation(age, user_type):
    assert 0 <= age <= 100
    assert user_type in ["admin", "user", "guest"]
```

Run it:
```bash
pytest test_users.py
```

Strategies can be registered in the test module that uses them, as here, or in separate strategy files that the plugin imports for you (see [Strategy Files and Scoped Names](#10-strategy-files-and-scoped-names)). A test can also take the factory itself, `@strategy(create_user_strategy)`, which needs no name lookup.

`register` and `strategy` are also available as `Strategy.register` and `Strategy.strategy`, the 2.x spelling, which keeps working.

## 📖 Core Concepts

### 1. Strategies & Parameters
A **Strategy** is a factory function that returns a `Parameter` object. The `Parameter` defines the shape of your test data using `TestArg` definitions.

```python
@register("math_ops")
def math_strategy(nsamples: int):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 100)),
        TestArg("y", rng_type=RNGInteger(1, 100)), # Avoid 0 for division
        directed_vectors={
            "identity": (1, 1),
            "large_nums": (100, 100)
        }
    )
```

The factory is called once for each test that uses the strategy, when pytest collects that test. Like a test with fixtures, it receives by name the inputs its parameters ask for, in any order, and nothing else:
- `nsamples`: the run's sample count, an integer (the `--nsamples` value, or 10 when the option is not given), or the string `"auto"` under `--nsamples=auto`. A factory that returns a `Parameter` does not need it, because the plugin generates the vectors itself.
- `ctx`: the testbench context (see [Configuration-Dependent Strategies](#11-configuration-dependent-strategies-new-in-v200)).
- `rng`: the plugin's `random.Random` (`RNG.generator()`), for draws the RNG types do not cover.
- `options`: a frozen `StrategyOptions` with this strategy's name and the run's `--nsamples`, `--vector-mode`, `--vector-name` and `--vector-index` values.

Any other parameter needs a default, which it keeps; `*args` and `**kwargs` receive nothing. A parameter without a default that is not one of these names fails collection with a message naming it (since 4.0 `def factory(n)` no longer receives `nsamples` by position), and `base`, `config` and `request` are reserved names. A factory decorated without `functools.wraps` is called with no arguments, and `mock.patch` mocks must be the first parameters.

A factory must return a `Parameter`. Returning an `(argnames, samples)` tuple, the 1.x form deprecated in 3.0, fails the collection of the tests that use the strategy since 4.0: give the `Parameter` one `TestArg` per argument and the fixed rows as `directed_vectors`.

Strategy names are scoped by folder, like fixtures in `conftest.py` files (see [Strategy Files and Scoped Names](#10-strategy-files-and-scoped-names)). A name registered twice in the same folder is an error.

`PytestStrategiesWarning` is a `UserWarning` subclass that you can import from `pytest_strategy`. The plugin emits it for problems that do not stop a run, such as a `Series` combination skipped because the constraints reject it. To turn it into an error, add `error::pytest_strategy.PytestStrategiesWarning` to your `filterwarnings` setting.

### 2. RNG Types
`pytest-strategies` provides rich, type-safe generators:

| Type           | Description               | Example                                            |
| -------------- | ------------------------- | -------------------------------------------------- |
| `RNGInteger`   | Integers in range         | `RNGInteger(0, 10)`                                |
| `RNGFloat`     | Floats in range           | `RNGFloat(0.0, 1.0)`                               |
| `RNGBoolean`   | Booleans with probability | `RNGBoolean(true_probability=0.8)`                 |
| `RNGString`    | Random strings            | `RNGString(min_length=5, max_length=10)`           |
| `Series`       | Ordered, deterministic sequence | `Series([1, 2, 3])`                          |
| `RNGSequence`  | Randomized sequence (permutation) | `RNGSequence([1, 2, 3])`                   |
| `RNGChoice`    | Choice from list          | `RNGChoice(["a", "b", "c"])`                       |
| `RNGEnum`      | Python Enum members       | `RNGEnum(MyEnum)`                                  |
| `RNGWeighted*` | Weighted ranges           | `RNGWeightedInteger({(0,10): 0.9, (11,100): 0.1})` |

RNG types check their arguments when they are constructed. A misconfigured strategy fails at collection with an `RNGValueError` (from `pytest_strategy.rng`) instead of drawing wrong values later. These cases are rejected:
- `RNGInteger` or `RNGFloat` with `min > max`. This includes a single bound that crosses the other bound's default: `RNGFloat(min=5.0)` fails because `max` defaults to 1.0.
- `RNGFloat` or `RNGWeightedFloat` bounds that are infinite or NaN.
- Weights of `RNGWeightedInteger`, `RNGWeightedFloat` or `RNGEnum` that are empty, negative, not finite or all zero, or whose total overflows. Some zero weights are allowed.
- A range of `RNGWeightedInteger` or `RNGWeightedFloat` that is not a `(min, max)` tuple with `min <= max`.
- `RNGEnum` given something that is not an `Enum` class, an `Enum` with no members, or a predicate that no member satisfies.
- `RNGString` with an empty `charset` (unless the length is 0), a negative length, or `min_length > max_length`.
- A `set` or `frozenset` passed to `Series` or `RNGSequence`. Their iteration order is not reproducible, so pass `sorted(...)` or a list instead.
- An empty `Series` or `RNGSequence`, or one whose predicate rejects every value, unless it has `skip_if_empty` (see [Skipping when a sequence is empty](#skipping-when-a-sequence-is-empty)).

`Parameter` raises `ValueError` when `nsamples` is not `None`, `"auto"` or an integer >= 0, when `max_retries` is not an integer >= 1, when `max_exhaustive` is not `None` or an integer >= 1, when `per_sequence_samples` is not a bool, or when two of its arguments have the same name.

### 3. Enums & Weighted Generation
The `RNGEnum` class supports standard Python Enums, including weighted selection and predicates.

```python
from enum import Enum
class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"

# Weights: 70% SUCCESS, 10% FAILED, 20% PENDING.
# The predicate excludes PENDING, so SUCCESS and FAILED are drawn 7:1.
arg = TestArg("status", rng_type=RNGEnum(
    Status,
    weights={Status.SUCCESS: 0.7, Status.FAILED: 0.1, Status.PENDING: 0.2},
    predicate=lambda s: s != Status.PENDING
))
```
With a predicate, each draw picks among the members that the predicate accepts, and they keep their relative weights.

### 4. Constraints
You can enforce rules on generated data:

**Per-Argument Predicates:**
```python
# Only even numbers
RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)
```

**Cross-Argument Constraints:**
```python
Parameter(
    TestArg("min", rng_type=RNGInteger(0, 10)),
    TestArg("max", rng_type=RNGInteger(0, 10)),
    vector_constraints={
        "ordered": lambda v: v.min < v.max,  # Ensure min < max
    }
)
```
A constraint receives the row as a `Vector`, a tuple whose fields are the argument names, so `v.min` and `v[0]` both work. Constraints are named: by the keys of a dict, or, in a list, by each function's name (`constraint_<i>` for a lambda). They run in order, and the first falsy result rejects the row. A random vector that fails a constraint is drawn again, up to `max_retries` times (a `Parameter` argument, default 100). If no valid vector turns up, collecting the test fails with "Could not generate random row K after max_retries=N draws", and the message counts the draws each constraint rejected first, by name, with the first row each one rejected, so you can tell which one is too strict. With `-v`, the Strategy Summary lists the rejections per constraint for each strategy.

### 5. Sequence Testing & Exhaustive Generation (New in v2.0.0)

There are two sequence types. Both walk a fixed set of values, but they differ in **ordering**:

| Type          | `--nsamples=auto`                                   | `--nsamples=K` (finite)                                              |
| ------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `Series`      | Values in **declaration order** (Cartesian product) | Cycles through the product in order (`K >= len`) or takes the first `K` (`K < len`), skipping combinations that the constraints reject |
| `RNGSequence` | A **random permutation** (each value once)          | Random picks, like `RNGChoice`                                      |

> **Rule of thumb:** the `RNG` prefix means random. `Series` is deterministic and ordered; `RNGSequence` is randomized.

**Deterministic Strategy (`Series`):**
```python
from pytest_strategy import Series

@register("matrix_test")
def matrix_strategy(nsamples):
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"]))
    )
```
Running with `pytest --nsamples=auto` generates 6 tests in order: `(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')`. The leftmost arg is the slowest counter.

**Randomized Strategy (`RNGSequence`):**
```python
from pytest_strategy import RNGSequence

TestArg("x", rng_type=RNGSequence([1, 2, 3]))
```
Under `--nsamples=auto`, this yields a permutation of `[1, 2, 3]` (each value exactly once, random order). Under a finite `--nsamples=K`, it picks `K` random values.

**Mixed Mode (Sequence + Random):**
If you mix a sequence type with random types (e.g., `RNGInteger`), the random values are regenerated for *each* sequence combination.

```python
@register("mixed_test")
def mixed_strategy(nsamples):
    return Parameter(
        # Deterministic: iterate through all user roles in order
        TestArg("role", rng_type=Series(["admin", "user", "guest"])),
        # Random: generate a fresh random ID for each role
        TestArg("id", rng_type=RNGInteger(1, 1000))
    )
```
Running `pytest --nsamples=auto` generates 3 tests (one per role), each with a random ID.

**Filtering with Predicates:**
Both sequence types accept a `predicate` argument to exclude values before generation.

```python
# Iterate only even numbers from 0-9, in order
TestArg("evens", rng_type=Series(range(10), predicate=lambda x: x % 2 == 0))
```

**What `--nsamples=auto` does:**
`auto` replaces a strategy's random samples with the combinations of its sequence arguments. Everything else works as it does with a finite count:
- Only `Series` and `RNGSequence` arguments are enumerated. Every other argument (`RNGInteger`, `RNGEnum`, `RNGChoice`, `RNGBoolean`, ...) gets a fresh random value for each combination.
- Combinations that `vector_constraints` reject are left out. Random arguments are redrawn up to `max_retries` times before a combination is dropped. If the constraints reject every combination, collecting the test fails.
- Directed vectors are placed before the combinations in the default `all` mode. In `mixed` mode they are included when `always_include_directed` is set (the default). `random_only` gives only the combinations.
- `--vector-mode=test`, `--vector-mode=directed_only`, `--vector-name` and `--vector-index` take precedence: they select only those vectors, and no combinations are generated.
- A strategy with no `Series` or `RNGSequence` argument has nothing to enumerate. It falls back to its own `nsamples` (see [Per-Strategy Sample Count](#8-per-strategy-sample-count-new-in-v200)), or 10 random samples.
- The number of combinations is checked before any is generated. More than 100,000 rows fails collection with a message that shows the product (`a=1,000 x b=1,000`). Raise the limit for one strategy with `Parameter(max_exhaustive=...)`, or for the project with the `strategies_max_exhaustive` ini option. The same limit applies to `per_sequence_samples=True`, where each combination counts `K` rows.

**Constraints on `Series` in finite mode:**
With a finite `--nsamples`, a `Series` combination that the constraints reject is skipped, and the cycle continues with the next combination. If the strategy also has random arguments, they are redrawn up to `max_retries` times before the combination is skipped. Each such skip emits a `PytestStrategiesWarning`: raise `max_retries`, or relax the constraint if that combination should be tested. If a whole cycle of combinations yields no valid vector, collecting the test fails.

**`n` samples per sequence combination (`per_sequence_samples`):**
By default a finite `--nsamples=K` gives `K` rows in total, shared among the sequence values. Pass `per_sequence_samples=True` to get `K` random rows for *each* combination of the `Series`/`RNGSequence` arguments instead:

```python
@register("per_device")
def per_device(nsamples):
    return Parameter(
        TestArg("device", rng_type=Series(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        per_sequence_samples=True,
    )
```
With the default of 10 samples this runs 20 tests: 10 for `devA`, then 10 for `devB`, each with a fresh random `width`. Several sequence arguments multiply (2 devices × 3 modes × `K`), and a value listed twice counts twice. Combinations follow declaration order for `RNGSequence` too, and directed vectors are still placed first. Under `--nsamples=auto` the flag has no effect: every combination runs once. A combination whose random arguments the constraints reject `max_retries` times in a row gets fewer rows and a `PytestStrategiesWarning`; collecting the test fails only if no combination yields a row. The warning names the strategy and the test.

#### Skipping when a sequence is empty
When the values come from configuration, there may be none: a testbench without any Esm peripheral, for example. An empty `Series` or `RNGSequence` normally fails collection. Give it a `skip_if_empty` reason instead (a keyword argument), and the strategy contributes a single skipped row with that reason, so each test that uses it is skipped:

```python
from pytest_strategy import Parameter, RNGInteger, RNGSequence, TestArg, register, strategy

# From your testbench configuration, e.g.
# [p.channel for p in config.peripherals.values() if p.type == "Esm"]
ESM_CHANNELS = []

@register("esm_rw")
def esm_rw(nsamples):
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(ESM_CHANNELS, skip_if_empty="no Esm peripheral in this testbench config")),
        TestArg("wdata", rng_type=RNGInteger(min=0, max=255)),
        per_sequence_samples=True,
    )

@strategy("esm_rw")
def test_rw(channel, wdata):
    ...
```
With channels, the option changes nothing. Without any, `pytest -rs` shows `SKIPPED [1] test_esm.py:<line>: no Esm peripheral in this testbench config`, and the test's ID is `test_rw[skipped]`. A test with other parametrization (a stacked `@pytest.mark.parametrize`, a parametrized fixture) is skipped once per combination of it. The skip applies in every `--vector-mode` and with `--nsamples=auto`, and directed and test vectors are skipped too. A `--vector-name` or `--vector-index` that names one of the strategy's directed vectors also gives the skipped test. The reason must be a non-empty string. It also applies when the predicate rejects every value. The test's signature (or dataclass) is still checked against the strategy, so a mismatch fails collection even on a configuration without values. `skip_if_empty` works for strategies that return a `Parameter`.

### 6. Metadata Export (New in v1.0.0)

You can export all registered strategies and their metadata (parameters, RNG types, constraints) to JSON for analysis or integration with other tools.

```python
from pytest_strategy import export_strategies

# Export as JSON string
json_data = export_strategies(format="json")
print(json_data)
```

In a pytest session every strategy file is imported first. A name registered in several folders is exported once, for the registration made last. Each factory gets the inputs it declares, as at collection, with the session's options: `nsamples` is the `--nsamples` value, `"auto"`, or 10 without the option.

### 7. Test Values (New in v2.0.0)

You can define test-specific vectors that only run when using `--vector-mode=test`. This is useful for defining specific test scenarios that you want to verify independently from random or directed vectors.

```python
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("api_test")
def api_test_strategy(nsamples):
    return Parameter(
        TestArg("status_code", rng_type=RNGInteger(200, 500)),
        # Test vectors: specific scenarios to verify
        test_vectors={
            "success": (200,),
            "not_found": (404,),
            "server_error": (500,)
        }
    )
```
Running with `pytest --vector-mode=test` runs only the test vectors, ignoring random and directed vectors.

A directed or test vector holds one value per argument, in argument order. It can be a tuple or a list (for example read from a YAML file), so `[404]` works like `(404,)`, or a dict of argument names to values in any order, such as `{"status_code": 404}`. A namedtuple is placed by its field names, which must be the argument names. A bare value such as `404` or `"a"` is an error, and a dict value for a one-argument strategy is written `({"a": 1},)` or `{"cfg": {"a": 1}}`. Vector names are non-empty strings. `directed_vectors` and `test_vectors` are read-only mappings: change them with `add_directed_vector()`, `remove_directed_vector()` and the `test` equivalents.

### 8. Per-Strategy Sample Count (New in v2.0.0)

By default the number of generated vectors is controlled globally by `--nsamples` (10 when unset). A strategy can declare its own count by passing `nsamples` to its `Parameter`:

```python
@register("edge_heavy")
def edge_heavy_strategy(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 100)),
        nsamples=25,  # this strategy generates 25 vectors by default
    )
```

This is a **soft default**: an explicit integer `--nsamples` on the command line still wins, so you can always scale a whole run from the CLI.

| `--nsamples` (CLI) | `Parameter(nsamples=...)` | Random vectors generated |
| ------------------ | ------------------------- | ------------------------ |
| not passed         | `25`                      | 25 (strategy value)      |
| not passed         | unset                     | 10 (global default)      |
| `--nsamples=5`     | `25`                      | 5 (CLI overrides)        |
| `--nsamples=auto`  | any                       | Every `Series`/`RNGSequence` combination. A strategy without such arguments uses its own value (here 25), or 10 when unset |
| not passed         | `"auto"`                  | Same as `--nsamples=auto` |

Directed vectors are added on top of these, according to `--vector-mode`. With `Parameter(per_sequence_samples=True)`, a finite count applies to each `Series`/`RNGSequence` combination instead of the whole strategy (see [`per_sequence_samples`](#5-sequence-testing--exhaustive-generation-new-in-v200)).

### 9. Dataclass Parameters

Instead of one test parameter per strategy argument, a test can take a single dataclass whose fields are the strategy's arguments:

```python
from dataclasses import dataclass

import pytest

from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy


@dataclass
class Point:
    x: int
    y: int


@register("points")
def points_strategy(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 10)),
        TestArg("y", rng_type=RNGInteger(0, 10)),
    )


@pytest.fixture
def scale():
    return 2


class TestPoints:
    @strategy("points")
    def test_scaled(self, point: Point, scale):
        assert 0 <= point.x * scale <= 20
```

Dataclass mode is used when all of these are true:
- the strategy has at least two arguments;
- none of the arguments is a parameter of the test;
- exactly one test parameter is annotated with a dataclass whose `__init__` fields are the strategy's argument names.

Fields with `init=False` are not counted, and `kw_only` fields work. The other parameters (`self`, `cls`, fixtures) are left alone, in any position. String annotations (`from __future__ import annotations` or quoted names) work too, as long as the dataclass is defined at module level. If a single parameter has a dataclass annotation but its fields do not match, collection fails with a message that lists the missing and extra fields.

### 10. Strategy Files and Scoped Names

Strategies do not have to be registered in the test module. A **strategy file** is a file named `strategies.py`, `strategy.py`, `*_strategies.py` or `*_strategy.py` that registers a strategy with a decorator: `@register("name")`, `@Strategy.register(...)`, or an alias such as `@S.register("name")`. The file's text is checked before it is imported, so write the name as a string literal in the `@register(...)` forms. A plain call such as `register("x")(fn)`, or a name held in a variable (`@register(NAME)`), is not recognized, and a file that only registers that way is not imported.

**Loading.** Before pytest imports a test module, the plugin imports the strategy files in the module's folder and in each folder above it, up to the rootdir (or the `testpaths` entry or command-line directory that contains the module). Each file is imported once per session, closest folder first and in name order within a folder. Strategy files in folders that no collected test module is in or below are not imported, unless a test asks for a name that is not registered on its path (see below) or you run `pytest --list-strategies`.

**Names are scoped by folder.** A strategy name is looked up like a fixture in a `conftest.py` file: in the test's folder first, then in each folder above it, and the nearest registration wins. A strategy registered in a test module or a `conftest.py` belongs to that file's folder. So with this layout, the tests under `tests/esm/` get the `default` strategy of `tests/esm/strategies.py`, and every other test gets the one of `tests/strategies.py`:

```text
tests/
├── strategies.py          # @register("default")
├── test_dma.py            # @strategy("default") -> tests/strategies.py
└── esm/
    ├── strategies.py      # @register("default")
    └── test_esm.py        # @strategy("default") -> tests/esm/strategies.py
```

When no folder on the test's path registers the name, the plugin imports every strategy file it can find and uses a registration from another place if there is exactly one: a sibling folder, or a module installed outside the rootdir (a registration outside the rootdir wins over the ones inside it). This keeps layouts like a shared `tests/strategies/` folder working. If several folders register the name, collecting the test fails with an error that lists them; register the strategy in a folder above the test, or pass the factory itself.

**Passing the factory.** `@strategy(my_factory)` uses that function, with no name lookup, so it cannot pick up another folder's strategy. The factory does not need to be registered. When it is, the test gets the same vectors and IDs as it would with the name.

**A name registered twice in one folder** (in one file or in two) is an error: the plugin emits a `PytestStrategiesWarning` naming both functions, and pytest stops with a usage error after collection. Running the same registration again (the same file executed twice) is silent. Decorated factories count as the function they decorate. Factories built by one shared function or class (closures, `functools.partial` objects, instances) count as that function or class, so registering two of them under one name is not reported.

**Where files are searched.** The search for all strategy files covers the `testpaths` directories from your pytest configuration, expanding glob patterns such as `pkgs/*/tests`, or the rootdir when `testpaths` is not set, and the directory of each path given on the command line. Below these directories it skips what pytest's collection skips: hidden directories (names starting with `.`), `__pycache__`, directories matching `norecursedirs` (by default these include `build`, `dist`, `venv` and `node_modules`; a pattern with a `/`, such as `tests/data`, is matched against the path), and virtual environments (a directory containing `pyvenv.cfg`, or a conda environment containing `conda-meta/history`). Like pytest, it follows symlinked directories.

**Imports.** A strategy file is imported the way pytest imports a test module in its folder, following `--import-mode`. In the default `prepend` mode its folder (or the root of its package) is put on `sys.path`, so it can import modules next to it. A test module or `conftest.py` that imports a strategy file (`from strategies import Mode`, for example to use an `Enum` it defines) gets the module the plugin loaded, so the file is not executed a second time and the tests compare against the same classes the strategy uses. When two folders without `__init__.py` both have a `strategies.py`, the second one is imported under a unique module name instead. A plain `from strategies import Mode` in that second folder still returns the first folder's module (Python caches modules by name), so its tests cannot import from it: give such files distinct names (`esm_strategies.py`), or add `__init__.py` files and use `from .strategies import Mode`.

Values that a strategy file draws when it is imported come from a random stream of their own, derived from the seed and the file's path, so `--rng-seed` reproduces them whatever else was collected first.

**Failures.** A strategy file that fails to import does not stop the run. The plugin prints `pytest-strategies: Warning - Failed to load <path>: <error>`. A file that calls `pytest.skip(..., allow_module_level=True)` or `pytest.importorskip()` at module level is skipped, and that is reported with `-v`. A "Strategy 'name' not found" error names the test, suggests close names, and lists the files that failed to load or were skipped, and the files with a strategy file name that mention `register` but were not imported because they have no registration decorator. `pytest -vv` prints each loaded file, and `pytest --list-strategies` lists the registered strategy names (with the file of each registration when a name is registered in several folders) and exits.

### 11. Configuration-Dependent Strategies (New in v2.0.0)

Some vectors depend on configuration that is only known when the session runs, such as a testbench description whose file is named on the command line. Strategy factories run before any fixture exists, so they cannot use one. Instead, implement the `pytest_strategies_context` hook in the rootdir's `conftest.py`: what it returns is passed as `ctx` to every factory that has a `ctx` parameter.

```python
# conftest.py
import pytest

from testbench import Testbench  # your project's code

def pytest_addoption(parser):
    parser.addoption("--tb-config", default="testbench.yaml")

@pytest.hookimpl(optionalhook=True)
def pytest_strategies_context(config):
    return Testbench.parse_config(config.getoption("--tb-config"))
```

`optionalhook=True` keeps the `conftest.py` usable when pytest-strategies is not loaded (not installed, or `-p no:pytest_strategy`): without it, pytest stops with "unknown hook".

```python
# esm_strategies.py
from pytest_strategy import Parameter, RNGInteger, RNGSequence, TestArg, register

@register("esm_rw")
def esm_rw(nsamples, ctx):
    channels = [p.channel for p in ctx.peripherals.values() if p.type == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(channels, skip_if_empty="no Esm peripheral")),
        TestArg("wdata", rng_type=RNGInteger(min=0, max=255)),
        per_sequence_samples=True,
    )
```

Tests use the strategy as usual, with `@strategy("esm_rw")`. With two Esm channels in the configuration they run 10 writes per channel; with none they are skipped with the reason.

- The hook is called at most once per session, the first time a factory with a `ctx` parameter runs, and its result is reused for the others. Factories without `ctx` are called as before and never trigger it.
- When no implementation returns a value, a `ctx` parameter keeps its default (or a value bound with `functools.partial`), and is `None` without one.
- If the hook raises, each test module that uses a factory with `ctx` fails collection with `Strategy factory '<name>' has a 'ctx' parameter, but the pytest_strategies_context hook raised <error>`. `pytest.fail()` in the hook is reported as it is. The hook can also call `pytest.skip(..., allow_module_level=True)` to skip those modules.
- Implement it in the rootdir's `conftest.py` or in a plugin. The result is shared by the whole session, and factories run while test modules are collected: a `conftest.py` further down is only loaded when pytest reaches its directory, so the hook there may be called too late, and once loaded its result also applies to modules outside that directory.
- Random draws in the hook come from a stream of their own, derived from the seed, so they are reproduced by `--rng-seed` and do not change any test's vectors.
- Under pytest-xdist every worker calls the hook, so it must return the same configuration in each, or the workers collect different tests.
- `export_strategies()` passes the same `ctx`.

Your testbench fixture does not change: the hook only has to describe the configuration the vectors depend on.

## 🔌 Fixture Integration

Strategies work seamlessly with standard pytest fixtures. You don't need any special configuration; just add the fixture to your test signature.

```python
@pytest.fixture
def database():
    return MockDB()

@strategy("user_strategy")
def test_db_insert(username, age, database): # 'database' is a fixture
    # 'username' and 'age' come from the strategy
    user = database.create_user(username, age)
    assert user.id is not None
```

## 🎛️ CLI Options

Control test generation directly from the command line:

| Option              | Description                                                             | Example                                     |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `--nsamples`        | Number of random samples per strategy (default 10), or per `Series`/`RNGSequence` combination for a strategy with `per_sequence_samples=True`. An integer overrides a strategy's own `nsamples`. `auto` enumerates the `Series`/`RNGSequence` arguments (see [What `--nsamples=auto` does](#5-sequence-testing--exhaustive-generation-new-in-v200)). Any other value is a usage error. | `pytest --nsamples=50` or `--nsamples=auto` |
| `--vector-mode`     | Generation mode: `all`, `random_only`, `directed_only`, `mixed`, `test` | `pytest --vector-mode=test`                 |
| `--vector-name`     | Run only the directed vector with this name                             | `pytest --vector-name=edge_case_1`          |
| `--vector-index`    | Run only the directed vector at this index (0-based, in definition order) | `pytest --vector-index=0`                 |
| `--rng-seed`        | Set seed for reproducibility                                            | `pytest --rng-seed=42`                      |
| `--list-strategies` | List the registered strategy names and exit                             | `pytest --list-strategies`                  |

The ini option `strategies_max_exhaustive` (default `100000`) sets the most rows `--nsamples=auto` or `per_sequence_samples=True` may generate for one strategy.

`--vector-name` and `--vector-index` take precedence over `--vector-mode` and `--nsamples`. A strategy without the requested directed vector yields no vectors, so the tests that use it are skipped ("got empty parameter set"). If no strategy in the run has the vector, for example because of a typo or an index that is out of range everywhere, pytest stops with a usage error that lists each strategy's directed vectors. It does not skip every test.

## 🔄 Reproducibility

The RNG seed of each run is printed in the pytest report header:
```text
pytest-strategies: RNG seed = 1763926297314361000
```
When tests fail, the plugin also prints how to rerun them with the same vectors, after the failure tracebacks and before the short test summary, even with `-q`:
```text
pytest-strategies: reproduce with --rng-seed=1763926297314361000
```

With `-v`, a "Strategy Summary" section lists each strategy with the number of tests that use it, their directed, random and test rows, and where the sample count came from (`--nsamples`, `Parameter(nsamples=)` or the default).

Each strategy and test pair draws from its own random stream. The stream is derived from the seed, the strategy name, the test's file path relative to the rootdir, and the test's qualified name. As a result:
- A test gets the same vectors and node IDs whether you run the whole suite, one file or one test, in any collection order and with any `--import-mode`.
- Two tests that use the same strategy get different random vectors.
- Values that a strategy factory draws itself are reproduced too. A factory that needs other random operations (`shuffle`, `gauss`) can draw from `RNG.generator()`, the generator the RNG types use.

The plugin draws from a `random.Random` instance of its own and never seeds Python's global `random` module. So your own use of `random` does not change the generated vectors, and `--rng-seed` does not reproduce what `random.random()` returns in your code, including in a factory: draw from `RNG.generator()` there. For the same seed, 3.0.0 generates the same values as 2.0.0 for strategies that draw through the RNG types, except values a strategy file draws when it is imported (3.0.0 gives each file a stream of its own); values differ from those of 1.x, so a seed recorded with 1.x does not reproduce that run.

The seed reproduces the generated test parameters, not random draws made inside test bodies. A test body that draws from `RNG` gets whatever state the generator is in when the test runs, which depends on the tests collected and run before it. To make plain `random` draws in test bodies follow the seed, seed it for each test in your `conftest.py`:

```python
# conftest.py
import random

import pytest

from pytest_strategy import RNG


@pytest.fixture(autouse=True)
def _seed_random(request):
    # Each test's random draws follow --rng-seed, whatever ran before it
    random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")
```

Values that a strategy file draws when it is imported are reproduced by the seed (see [Strategy Files and Scoped Names](#10-strategy-files-and-scoped-names)). A draw at module level in a test module (for example `BASE = RNG.integer(0, 1000)`) follows the seed too, but it depends on the test modules collected before it, so running one file alone gives it another value. Move such draws into the factory or a strategy file.

Keep the same rootdir, because the test's path relative to the rootdir is part of the stream. pytest uses the directory of your ini file (such as `pytest.ini`) as the rootdir when there is one.

**pytest-xdist:** runs with `-n` work with or without `--rng-seed`. The controller sends its seed to the workers, so they all generate the same tests.

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
