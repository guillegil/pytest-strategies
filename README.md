# pytest-strategies 🧪

**Powerful, constrained-randomized test generation for pytest.**

`pytest-strategies` extends pytest with a robust framework for defining test strategies that combine **random generation**, **directed edge cases**, and **constraints**. It bridges the gap between simple parametrization and property-based testing, giving you full control over your test data.

[![Tests](https://github.com/guillegil/pytest-strategies/actions/workflows/tests.yml/badge.svg)](https://github.com/guillegil/pytest-strategies/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **Hybrid Generation**: seamlessly mix **randomly generated** data with **directed** (hardcoded) edge cases.
- **Type-Safe RNG**: Built-in generators for Integers, Floats, Booleans, Strings, Choices, and **Enums**.
- **Weighted Probabilities**: Define custom distributions for Enums, Integers, and Floats.
- **Constraints & Predicates**: Filter generated values using simple lambda predicates or complex vector constraints.
- **Fixture Integration**: Works out-of-the-box with standard pytest fixtures (custom, parametrized, or built-in).
- **Reproducibility**: Deterministic generation via seed control for debugging failures.
- **CLI Control**: Filter strategies, change generation modes, or increase sample sizes directly from the command line.

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
```

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

| Option           | Description                                            | Example                              |
| ---------------- | ------------------------------------------------------ | ------------------------------------ |
| `--nsamples`     | Number of random samples to generate (default: 10)     | `pytest --nsamples=50`               |
| `--vector-mode`  | Generation mode: `all`, `random_only`, `directed_only` | `pytest --vector-mode=directed_only` |
| `--vector-name`  | Run only a specific directed vector by name            | `pytest --vector-name=edge_case_1`   |
| `--vector-index` | Run only a specific sample by index                    | `pytest --vector-index=0`            |
| `--rng-seed`     | Set seed for reproducibility                           | `pytest --rng-seed=42`               |

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
