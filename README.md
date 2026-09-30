# pytest-strategies 🧪

**Powerful, constrained-randomized test generation for pytest.**

`pytest-strategies` extends pytest with a robust framework for defining test strategies that combine **random generation**, **directed edge cases**, and **constraints**. It bridges the gap between simple parametrization and property-based testing, giving you full control over your test data.

[![Tests](https://github.com/guillegil/pytest-strategies/actions/workflows/tests.yml/badge.svg)](https://github.com/guillegil/pytest-strategies/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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

## 📦 Installation

`pytest-strategies` is not published on PyPI yet. Install it from GitHub:

```bash
pip install git+https://github.com/guillegil/pytest-strategies.git
```

or from a local clone with `pip install -e .`.

## ⚡ Quick Start

Define a strategy and apply it to your test:

```python
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNGChoice

# 1. Register a strategy
@Strategy.register("user_age_strategy")
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
@Strategy.strategy("user_age_strategy")
def test_user_validation(age, user_type):
    assert 0 <= age <= 100
    assert user_type in ["admin", "user", "guest"]
```

Run it:
```bash
pytest test_users.py
```

Strategies can be registered in the test module that uses them, as here, or in separate strategy files that the plugin imports for you (see [Strategy Files](#10-strategy-files)).

## 📖 Core Concepts

### 1. Strategies & Parameters
A **Strategy** is a factory function that returns a `Parameter` object. The `Parameter` defines the shape of your test data using `TestArg` definitions.

```python
@Strategy.register("math_ops")
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

The factory is called once for each test that uses the strategy, when pytest collects that test. It receives the run's sample count as `nsamples`: an integer (the `--nsamples` value, or 10 when the option is not given), or the string `"auto"` under `--nsamples=auto`. The factory may take it as a keyword or positional parameter, or take no parameters at all; the plugin reads the signature to decide how to call it. A factory that returns a `Parameter` does not need to use `nsamples`, because the plugin generates the vectors itself.

Strategy names share one registry. If a different function registers a name that is already taken, the plugin emits a `PytestStrategiesWarning` naming both functions, and the last registration wins. Running the same function's registration again (for example, the same file imported twice) is silent.

`PytestStrategiesWarning` is a `UserWarning` subclass that you can import from `pytest_strategy.strategy`. To turn it into an error, add `error::pytest_strategy.strategy.PytestStrategiesWarning` to your `filterwarnings` setting.

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
- Weights of `RNGWeightedInteger`, `RNGWeightedFloat` or `RNGEnum` that are empty, negative, not finite or all zero. Some zero weights are allowed.
- `RNGEnum` given something that is not an `Enum` class, an `Enum` with no members, or a predicate that no member satisfies.
- `RNGString` with an empty `charset` (unless the length is 0), a negative length, or `min_length > max_length`.
- A `set` or `frozenset` passed to `Series` or `RNGSequence`. Their iteration order is not reproducible, so pass `sorted(...)` or a list instead.
- An empty `Series` or `RNGSequence`, or one whose predicate rejects every value, unless it has `skip_if_empty` (see [Skipping when a sequence is empty](#skipping-when-a-sequence-is-empty)).

`Parameter` raises `ValueError` when `nsamples` is not `None`, `"auto"` or an integer >= 0, or when `max_retries` is not an integer >= 1.

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
    vector_constraints=[
        lambda v: v[0] < v[1]  # Ensure min < max
    ]
)
```
A random vector that fails a constraint is drawn again, up to `max_retries` times (a `Parameter` argument, default 100). If no valid vector turns up, collecting the test fails with "Could not generate valid vector".

### 5. Sequence Testing & Exhaustive Generation (New in v1.1.0)

There are two sequence types. Both walk a fixed set of values, but they differ in **ordering**:

| Type          | `--nsamples=auto`                                   | `--nsamples=K` (finite)                                              |
| ------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `Series`      | Values in **declaration order** (Cartesian product) | Cycles through the product in order (`K >= len`) or takes the first `K` (`K < len`), skipping combinations that the constraints reject |
| `RNGSequence` | A **random permutation** (each value once)          | Random picks, like `RNGChoice`                                      |

> **Rule of thumb:** the `RNG` prefix means random. `Series` is deterministic and ordered; `RNGSequence` is randomized.

**Deterministic Strategy (`Series`):**
```python
from pytest_strategy import Series

@Strategy.register("matrix_test")
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
@Strategy.register("mixed_test")
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
- A strategy with no `Series` or `RNGSequence` argument has nothing to enumerate. It falls back to its own `nsamples` (see [Per-Strategy Sample Count](#8-per-strategy-sample-count-new-in-v110)), or 10 random samples.

**Constraints on `Series` in finite mode:**
With a finite `--nsamples`, a `Series` combination that the constraints reject is skipped, and the cycle continues with the next combination. If the strategy also has random arguments, they are redrawn up to `max_retries` times before the combination is skipped. Each such skip emits a `PytestStrategiesWarning`: raise `max_retries`, or relax the constraint if that combination should be tested. If a whole cycle of combinations yields no valid vector, collecting the test fails.

**`n` samples per sequence value (`per_sequence_samples`):**
By default a finite `--nsamples=K` gives `K` rows in total, shared among the sequence values. Pass `per_sequence_samples=True` to get `K` random rows for *each* combination of the `Series`/`RNGSequence` arguments instead:

```python
@Strategy.register("per_device")
def per_device(nsamples):
    return Parameter(
        TestArg("device", rng_type=Series(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        per_sequence_samples=True,
    )
```
With the default of 10 samples this runs 20 tests: 10 for `devA`, then 10 for `devB`, each with a fresh random `width`. Several sequence arguments multiply (2 devices × 3 modes × `K`). Combinations follow declaration order for `RNGSequence` too, and directed vectors are still placed first. Under `--nsamples=auto` the flag has no effect: every combination runs once. A combination whose random arguments the constraints reject `max_retries` times in a row gets fewer rows and a `PytestStrategiesWarning`; collecting the test fails only if no combination yields a row.

#### Skipping when a sequence is empty
When the values come from configuration, there may be none: a testbench without any Esm peripheral, for example. An empty `Series` or `RNGSequence` normally fails collection. Give it a `skip_if_empty` reason instead, and every test that uses the strategy is reported as one skipped test with that reason:

```python
@Strategy.register("esm_rw")
def esm_rw(nsamples):
    channels = [p.channel for p in CONFIG.peripherals.values() if p.type == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(channels, skip_if_empty="no Esm peripheral in this testbench config")),
        TestArg("wdata", rng_type=RNGInteger(min=0, max=255)),
        per_sequence_samples=True,
    )
```
With channels, the option changes nothing. Without any, `pytest -rs` shows `SKIPPED [1] test_esm.py:12: no Esm peripheral in this testbench config`, and the test's ID is `test_rw[skipped]`. The skip applies in every `--vector-mode` and with `--nsamples=auto`, and directed and test vectors are skipped too. A `--vector-name` or `--vector-index` that names one of the strategy's vectors also gives the skipped test. The reason must be a non-empty string. It also applies when the predicate rejects every value.

### 6. Metadata Export (New in v1.0.0)

You can export all registered strategies and their metadata (parameters, RNG types, constraints) to JSON for analysis or integration with other tools.

```python
from pytest_strategy import Strategy

# Export as JSON string
json_data = Strategy.export_strategies(format="json")
print(json_data)
```

### 7. Test Values (New in v1.1.0)

You can define test-specific vectors that only run when using `--vector-mode=test`. This is useful for defining specific test scenarios that you want to verify independently from random or directed vectors.

```python
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("api_test")
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

### 8. Per-Strategy Sample Count (New in v1.1.0)

By default the number of generated vectors is controlled globally by `--nsamples` (10 when unset). A strategy can declare its own count by passing `nsamples` to its `Parameter`:

```python
@Strategy.register("edge_heavy")
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

Directed vectors are added on top of these, according to `--vector-mode`.

### 9. Dataclass Parameters

Instead of one test parameter per strategy argument, a test can take a single dataclass whose fields are the strategy's arguments:

```python
from dataclasses import dataclass

import pytest

from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger


@dataclass
class Point:
    x: int
    y: int


@Strategy.register("points")
def points_strategy(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 10)),
        TestArg("y", rng_type=RNGInteger(0, 10)),
    )


@pytest.fixture
def scale():
    return 2


class TestPoints:
    @Strategy.strategy("points")
    def test_scaled(self, point: Point, scale):
        assert 0 <= point.x * scale <= 20
```

Dataclass mode is used when all of these are true:
- the strategy has at least two arguments;
- none of the arguments is a parameter of the test;
- exactly one test parameter is annotated with a dataclass whose `__init__` fields are the strategy's argument names.

Fields with `init=False` are not counted, and `kw_only` fields work. The other parameters (`self`, `cls`, fixtures) are left alone, in any position. String annotations (`from __future__ import annotations` or quoted names) work too, as long as the dataclass is defined at module level. If a single parameter has a dataclass annotation but its fields do not match, collection fails with a message that lists the missing and extra fields.

### 10. Strategy Files

Strategies do not have to be registered in the test module. The plugin imports strategy files when the session starts, before any test module is collected. A file is loaded when both of these are true:
- its name is `strategies.py`, `strategy.py`, `*_strategies.py` or `*_strategy.py`, and
- it contains the text `@Strategy.register`.

The plugin searches the `testpaths` directories from your pytest configuration, expanding glob patterns such as `pkgs/*/tests`. When `testpaths` is not set, it searches the rootdir. It also searches the directory of each path given on the command line. Below these directories it skips what pytest's collection skips: hidden directories (names starting with `.`), `__pycache__`, directories matching `norecursedirs` (by default these include `build`, `dist`, `venv` and `node_modules`), and virtual environments (any directory containing a `pyvenv.cfg` file). A directory named on the command line is always searched. Files are loaded in sorted path order, so the same file wins a duplicate strategy name on every machine.

Each strategy file is imported as a standalone module, not as part of a package. Relative imports do not work in it, and a sibling module can only be imported if its directory is on `sys.path`.

A strategy file that fails to import does not stop the run. The plugin prints `pytest-strategies: Warning - Failed to load <path>: <error>` when the session starts. A file that calls `pytest.skip(..., allow_module_level=True)` or `pytest.importorskip()` at module level is skipped, and that is reported with `-v`. Any "Strategy 'name' not found" error lists the files that failed to load or were skipped. `pytest -vv` prints each loaded file, and `pytest --list-strategies` lists the registered strategy names and exits.

## 🔌 Fixture Integration

Strategies work seamlessly with standard pytest fixtures. You don't need any special configuration; just add the fixture to your test signature.

```python
@pytest.fixture
def database():
    return MockDB()

@Strategy.strategy("user_strategy")
def test_db_insert(username, age, database): # 'database' is a fixture
    # 'username' and 'age' come from the strategy
    user = database.create_user(username, age)
    assert user.id is not None
```

## 🎛️ CLI Options

Control test generation directly from the command line:

| Option              | Description                                                             | Example                                     |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `--nsamples`        | Number of random samples per strategy (default 10). An integer overrides a strategy's own `nsamples`. `auto` enumerates the `Series`/`RNGSequence` arguments (see [What `--nsamples=auto` does](#5-sequence-testing--exhaustive-generation-new-in-v110)). Any other value is a usage error. | `pytest --nsamples=50` or `--nsamples=auto` |
| `--vector-mode`     | Generation mode: `all`, `random_only`, `directed_only`, `mixed`, `test` | `pytest --vector-mode=test`                 |
| `--vector-name`     | Run only the directed vector with this name                             | `pytest --vector-name=edge_case_1`          |
| `--vector-index`    | Run only the directed vector at this index (0-based, in definition order) | `pytest --vector-index=0`                 |
| `--rng-seed`        | Set seed for reproducibility                                            | `pytest --rng-seed=42`                      |
| `--list-strategies` | List the registered strategy names and exit                             | `pytest --list-strategies`                  |

`--vector-name` and `--vector-index` take precedence over `--vector-mode` and `--nsamples`. A strategy without the requested directed vector yields no vectors, so the tests that use it are skipped ("got empty parameter set"). If no strategy in the run has the vector, for example because of a typo or an index that is out of range everywhere, pytest stops with a usage error that lists each strategy's directed vectors. It does not skip every test.

## 🔄 Reproducibility

The RNG seed of each run is printed in the pytest report header:
```text
pytest-strategies: RNG seed = 1763926297314361000
```
pytest does not show the header with `-q` or `--no-header`, so drop those flags (or pass your own `--rng-seed`) when you need the seed, e.g. in CI.

If a test fails, pass this seed to generate the same test vectors again:
```bash
pytest --rng-seed=1763926297314361000
```

Each strategy and test pair draws from its own random stream. The stream is derived from the seed, the strategy name, the test's file path relative to the rootdir, and the test's qualified name. As a result:
- A test gets the same vectors and node IDs whether you run the whole suite, one file or one test, in any collection order and with any `--import-mode`.
- Two tests that use the same strategy get different random vectors.
- Values that a strategy factory draws itself are reproduced too.

The seed reproduces the generated test parameters, not random draws made inside test bodies. A test body that draws from `RNG` or `random` gets whatever state the global generator is in when the test runs. That state depends on the tests collected and run before it, so it changes when you rerun a single test or run under pytest-xdist. Seed such draws in the test itself (see [docs/dev.md](docs/dev.md#reproducibility)).

For the same seed, the generated values differ from those of 1.1.0a2 and earlier, so a seed recorded with an older version does not reproduce that run. Also keep the same rootdir, because the test's path relative to the rootdir is part of the stream. pytest uses the directory of your ini file (such as `pytest.ini`) as the rootdir when there is one.

**pytest-xdist:** runs with `-n` work with or without `--rng-seed`. The controller sends its seed to the workers, so they all generate the same tests.

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
