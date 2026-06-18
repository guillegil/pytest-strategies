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
- **Per-Strategy Sample Count**: Let a strategy declare its own vector count as a soft default that `--nsamples` can still override.

## 📦 Installation

```bash
pip install pytest-strategies
```

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

### 3. Enums & Weighted Generation
The `RNGEnum` class supports standard Python Enums, including weighted selection and predicates.

```python
from enum import Enum
class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"

# 70% Success, 20% Pending, 10% Failed
# Exclude PENDING state entirely via predicate
arg = TestArg("status", rng_type=RNGEnum(
    Status,
    weights={Status.SUCCESS: 0.7, Status.FAILED: 0.1, Status.PENDING: 0.2},
    predicate=lambda s: s != Status.PENDING
))
```

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
    ]
)
```

### 5. Sequence Testing & Exhaustive Generation (New in v1.1.0)

There are two sequence types. Both walk a fixed set of values, but they differ in **ordering**:

| Type          | `--nsamples=auto`                                   | `--nsamples=K` (finite)                                              |
| ------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `Series`      | Values in **declaration order** (Cartesian product) | Cycles through the product in order (`K >= len`) or takes the first `K` (`K < len`) |
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

This is a **soft default**: an explicit `--nsamples` on the command line still wins, so you can always scale a whole run from the CLI.

| `--nsamples` (CLI) | `Parameter(nsamples=...)` | Vectors generated              |
| ------------------ | ------------------------- | ------------------------------ |
| not passed         | `25`                      | 25 (strategy value)            |
| not passed         | unset                     | 10 (global default)            |
| `--nsamples=5`     | `25`                      | 5 (CLI overrides)              |
| `--nsamples=auto`  | any                       | exhaustive (auto always wins)  |

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

| Option           | Description                                                             | Example                                     |
| ---------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `--nsamples`     | Number of samples (default 10), or "auto" for exhaustive. Overrides a strategy's own `nsamples`. | `pytest --nsamples=50` or `--nsamples=auto` |
| `--vector-mode`  | Generation mode: `all`, `random_only`, `directed_only`, `mixed`, `test` | `pytest --vector-mode=test`                 |  |
| `--vector-name`  | Run only a specific directed vector by name                             | `pytest --vector-name=edge_case_1`          |
| `--vector-index` | Run only a specific sample by index                                     | `pytest --vector-index=0`                   |
| `--rng-seed`     | Set seed for reproducibility                                            | `pytest --rng-seed=42`                      |

## 🔄 Reproducibility

Every test run prints the RNG seed used:
```text
pytest-strategies: RNG seed = 1763926297314361000
```
If a test fails, you can reproduce the exact same data sequence by passing this seed:
```bash
pytest --rng-seed=1763926297314361000
```

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
