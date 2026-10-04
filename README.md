# pytest-strategies 🧪

**Powerful, constrained-randomized test generation for pytest.**

`pytest-strategies` extends pytest with a robust framework for defining test strategies that combine **random generation**, **directed edge cases**, and **constraints**. It bridges the gap between simple parametrization and property-based testing, giving you full control over your test data.

[![Tests](https://github.com/guillegil/pytest-strategies/actions/workflows/tests.yml/badge.svg)](https://github.com/guillegil/pytest-strategies/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **Hybrid Generation**: mix **randomly generated** rows with **directed** (hardcoded) edge cases and **test** vectors that run on request.
- **Stable Test IDs**: rows are named, not spelled out by their values: `test_write[directed-zeros]`, `test_write[rand-3]`. The IDs are the same for every seed, so `-k zeros`, `--deselect`, CI history and node-ID reruns keep working.
- **Reproducible Failures**: each failed row shows its values, the seed and the command that runs it alone, in the terminal and in JUnit XML, and `--lf` reruns the failed rows with the values they failed with.
- **Row-Stable Random Streams**: a row's values depend on the seed, the test and the row, not on `--nsamples`, the other rows, the collection order or pytest-xdist.
- **Sequence Testing**: use `Series` for **deterministic, ordered** value sequences (Cartesian product) or `RNGSequence` for a **randomized permutation** of the same values.
- **Type-Safe RNG**: built-in generators for Integers, Floats, Booleans, Strings, Choices, Sequences, and **Enums**.
- **Weighted Probabilities**: define custom distributions for Enums, Integers, and Floats.
- **Named Constraints**: filter values with predicates and rows with named constraints that read the row by argument name (`v.addr`); turn one off for a run with `--strategy-constraint-off`.
- **Testbench Context**: build vectors from configuration that is only known when the session runs, with a context per folder, the same object in your fixtures, and a fingerprint that tells two runs' configurations apart.
- **Fixture Integration**: works out-of-the-box with standard pytest fixtures (custom, parametrized, or built-in), and a test can take the whole row as one dataclass.
- **CLI Control**: filter vectors, change generation modes, or increase sample sizes directly from the command line.
- **Per-Strategy Sample Count**: let a strategy declare its own vector count as a soft default that an integer `--nsamples` can still override.
- **Folder-Scoped Names**: strategy names are looked up like fixtures, nearest folder first, and strategy files are imported only for the folders being tested.
- **Metadata**: `item.stash[VECTOR_KEY]` describes the row of each test, and `export_strategies()` writes every strategy as versioned JSON.
- **Typed**: the package ships type hints (`py.typed`), and the decorators keep the decorated function's type.
- **Agent Skill**: `pytest-strategies skill install` gives Claude Code, Codex and other coding agents a guide to writing strategies.

## 📦 Installation

`pytest-strategies` needs Python 3.11 or later and pytest 8.4.2 or later. It is not published on PyPI yet. Install a release from GitHub:

```bash
pip install "pytest-strategies @ git+https://github.com/guillegil/pytest-strategies.git@v4.0.0"
```

or from a local clone with `pip install -e .`.

Upgrading from 3.x? Test IDs and random values change, and the APIs that 3.0 deprecated are gone: see [Upgrading to 4.0](#-upgrading-to-40).

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
def create_user_strategy() -> Parameter:
    return Parameter(
        # Randomly generate ages between 0 and 100
        TestArg("age", rng_type=RNGInteger(0, 100)),
        # Randomly choose a user type
        TestArg("user_type", rng_type=RNGChoice(["admin", "user", "guest"])),
        # Always include these specific edge cases, by position or by name
        directed_vectors={
            "newborn": (0, "guest"),
            "centenarian": (100, "user"),
            "admin_edge": {"age": 18, "user_type": "admin"},
        },
    )

# 2. Use the strategy in your test
@strategy("user_age_strategy")
def test_user_validation(age, user_type):
    assert 0 <= age <= 100
    assert user_type in ["admin", "user", "guest"]
```

Run it:
```bash
pytest test_users.py -v
```
```text
pytest-strategies: RNG seed = 1763926297314361000
...
test_users.py::test_user_validation[directed-newborn] PASSED
test_users.py::test_user_validation[directed-centenarian] PASSED
test_users.py::test_user_validation[directed-admin_edge] PASSED
test_users.py::test_user_validation[rand-0] PASSED
...
test_users.py::test_user_validation[rand-9] PASSED
```

The directed vectors run first, then 10 random rows (`--nsamples=50` asks for 50). Each row has a name, so `pytest test_users.py -k newborn` runs one edge case, and `pytest "test_users.py::test_user_validation[rand-3]" --rng-seed=1763926297314361000` runs one random row again with the values it had (see [Test IDs and Selecting Rows](#-test-ids-and-selecting-rows)).

Strategies can be registered in the test module that uses them, as here, or in separate strategy files that the plugin imports for you (see [Strategy Files and Scoped Names](#9-strategy-files-and-scoped-names)). A test can also take the factory itself, `@strategy(create_user_strategy)`, which needs no name lookup.

`register` and `strategy` are also available as `Strategy.register` and `Strategy.strategy`, the 2.x spelling, which keeps working.

## 📖 Core Concepts

### 1. Strategies and Factories
A **strategy** is a factory function that returns a `Parameter` object. The `Parameter` defines the shape of your test data using `TestArg` definitions.

```python
@register("math_ops")
def math_strategy():
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
- `ctx`: the testbench context of the test's folder (see [Configuration-Dependent Strategies](#10-configuration-dependent-strategies)).
- `rng`: the plugin's `random.Random` (`RNG.generator()`), for draws the RNG types do not cover. Its draws follow `--rng-seed` (see [Reproducibility](#-reproducibility)).
- `options`: a frozen `StrategyOptions`, exported from `pytest_strategy`, with the run's options for this strategy:

| Field             | Value                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------- |
| `strategy`        | The name the strategy was resolved under, as in the `-v` summary                        |
| `nsamples`        | The `--nsamples` value, `"auto"`, or 10 without the option                              |
| `nsamples_source` | `"--nsamples"` or `"default"`                                                           |
| `mode`            | The `--vector-mode` value (`"all"` by default)                                          |
| `vector_name`     | The `--vector-name` value, or `None`                                                    |
| `vector_index`    | The `--vector-index` value, or `None`. `options.filtered` is true when either is set    |
| `constraints_off` | The constraint names `--strategy-constraint-off` turns off for this strategy, a `frozenset` |

```python
import random

from pytest_strategy import Parameter, RNGChoice, RNGInteger, StrategyOptions, TestArg, register

from vectors import load_vectors  # your project's code


@register("alu")
def alu(rng: random.Random, options: StrategyOptions) -> Parameter:
    # The factory's own draws follow --rng-seed too
    ops = rng.sample(["add", "sub", "mul", "div", "shl", "shr"], k=3)
    # Reading the golden vectors is only worth it when they run
    golden = {} if options.mode == "random_only" else load_vectors("alu.yaml")
    return Parameter(
        TestArg("op", rng_type=RNGChoice(ops)),
        TestArg("a", rng_type=RNGInteger(0, 255)),
        directed_vectors=golden,
    )
```

Any other parameter needs a default, which it keeps; `*args` and `**kwargs` receive nothing. A parameter without a default that is not one of these names fails collection with a message naming it, and suggests the closest input (`def factory(n)` no longer receives `nsamples` by position). `base`, `config` and `request` are reserved names, an error even with a default. A factory decorated without `functools.wraps` is called with no arguments, and `mock.patch` mocks must be the first parameters. Later 4.x releases add inputs only as new `StrategyOptions` fields or under the reserved names, so a factory written for 4.0 keeps receiving exactly what it receives now.

A factory must return a `Parameter`. Anything else fails the collection of the tests that use the strategy, with a message that names the strategy and what it returned.

Strategy names are scoped by folder, like fixtures in `conftest.py` files (see [Strategy Files and Scoped Names](#9-strategy-files-and-scoped-names)). A name registered twice in the same folder is an error.

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

`TestArg`'s `rng_type` must be an RNG type (or an object with a `generate()` method): a bare lambda, or a class instead of an instance (`RNGBoolean` for `RNGBoolean()`), raises `TypeError`. The options after `rng_type` are keyword-only: `TestArg(name, rng_type=None, *, value=None, validator=None, description="")`. A `TestArg` with `value=` passes that value in every random row.

`Parameter` raises `ValueError` when `nsamples` is not `None`, `"auto"` or an integer >= 0, when `max_retries` is not an integer >= 1, when `max_exhaustive` is not `None` or an integer >= 1, when `per_sequence_samples` is not a bool, when `ids` is not `None`, `"names"`, `"values"` or a callable, or when two of its arguments have the same name. An argument name must be a valid Python identifier that is not a keyword and does not start with `_`, because it names a field of the row (see [Rows and Constraints](#4-rows-and-constraints)).

**Custom RNG types.** Subclass `RNGType` and implement `generate()` and the `python_type` property. `generate()` is called once per draw, while `RNG.generator()` is that argument's own random stream for the row, so draw there, from `RNG.generator()` or the `RNG.*` helpers, and keep no state between calls:

```python
from pytest_strategy import RNG, RNGType


class RNGAligned(RNGType[int]):
    """An address below `top`, aligned to `align` bytes."""

    def __init__(self, top: int, align: int) -> None:
        self.top = top
        self.align = align

    def generate(self) -> int:
        # Drawn here, from this argument's stream for this row
        return RNG.integer(0, self.top // self.align - 1) * self.align

    @property
    def python_type(self) -> type[int]:
        return int
```

A type that draws from a generator kept from earlier (such as the factory's `rng`), or that counts its calls (a walking-ones pattern), makes a row's values depend on the rows drawn before it; see [Reproducibility](#-reproducibility).

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

### 4. Rows and Constraints
Every row a `Parameter` generates is a `Vector`: a namedtuple, exported from `pytest_strategy`, whose fields are the strategy's argument names in declaration order. `v.addr` and `v[0]` both work, so do unpacking and `len(v)`, and a Vector compares and hashes like the plain tuple of its values (`v == (0, 1)`), although `type(v) is tuple` is false. A misspelled field raises `AttributeError: Vector has no argument 'lenght'; its arguments are addr, len`. Fields named `count` or `index` hide the tuple methods of the same name: use `tuple.index(v, x)` then. `Parameter.vector_type` returns the strategy's Vector class, and `generate_vector()`, `generate_vectors()`, `generate_exhaustive()` and the `get_*_vector()` methods return Vectors. Tests receive the arguments one by one (or as a record, see [Record Parameters](#8-record-parameters)), never the Vector itself.

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

A constraint receives the row as a `Vector` and returns whether to keep it. Constraints are named: by the keys of a dict, or, in a list, by each function's name (`constraint_<i>` for a lambda, a `functools.partial` or a callable object, i being its position). A name is a non-empty string without whitespace, `:`, `,` or `=`. Two constraints with one name (typically two closures made by one helper) fail when the `Parameter` is built: pass a dict to name them. `param.vector_constraints` is a read-only mapping of names to functions, in the order they run; change it with `add_constraint(fn, *, name=None)`, which returns the name, `remove_constraint(name)` and `clear_constraints()`.

Constraints run in order, and the first falsy result rejects the row, so a constraint may rely on the ones before it. A constraint should only read the row: one that draws random values gets a warning (see [Reproducibility](#-reproducibility)). A constraint that raises fails collection with an error that names it and the row, followed by its own traceback. Directed and test vectors are never checked by constraints.

A random row that fails a constraint is drawn again, up to `max_retries` times (a `Parameter` argument, default 100). If no valid row turns up, collecting the test fails, and the message counts the rejected draws by the name of the first constraint that rejected each one, with the first row each constraint rejected, so you can tell which one is too strict:

```text
Could not generate random row 3 after max_retries=100 draws. Rejected by (first failing constraint per draw): aligned=97, no_4k_cross=3. First rows rejected: aligned: Vector(addr=4097, len=16); no_4k_cross: Vector(addr=4090, len=16). Raise Parameter(max_retries=...), relax a constraint, or turn one off for this run with --strategy-constraint-off=dma_burst:aligned.
```

With `-v`, the Strategy Summary lists the rejections per constraint for each strategy.

**Turning a constraint off for one run.** Name it with `--strategy-constraint-off`: `--strategy-constraint-off=ordered` turns off every constraint named `ordered`, and `--strategy-constraint-off=my_strategy:ordered` only the one in that strategy (the name the strategy was resolved under, as in the `-v` summary). Items are separated by commas and the option can be repeated, so `--strategy-constraint-off=my_strategy:ordered,small` turns `ordered` off in `my_strategy` and `small` in every strategy. The `Parameter` keeps its constraints, and the rows that the constraint never rejected keep their values. The report header lists what is off, and `-v` adds `off: ordered` to the strategy's summary line. A factory sees the names in `options.constraints_off`.

An item that matches no constraint of any strategy the run resolved stops a run of the whole suite with a usage error that suggests the closest name and lists the constraints by strategy. A run narrowed to some tests (paths or node IDs, `--lf`, `--sw`, `--ignore`, or no path from a folder below the rootdir, where pytest collects only that folder) resolves only some strategies, and so does a run in which a test module is skipped or fails to collect: it prints that message in red and goes on, so the option can be kept in `addopts`.

### 5. Directed and Test Vectors

**Directed vectors** are the fixed rows a strategy always tests: edge cases, known bugs. **Test vectors** are scenarios that run only with `--vector-mode=test`, to verify them on their own:

```python
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("api_test")
def api_test_strategy():
    return Parameter(
        TestArg("status_code", rng_type=RNGInteger(200, 500)),
        directed_vectors={"ok": (200,)},
        # Test vectors: specific scenarios to verify
        test_vectors={
            "success": (200,),
            "not_found": (404,),
            "server_error": (500,)
        }
    )
```

`--vector-mode` chooses which rows run:

| Mode                | Rows                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `all` (the default) | The directed vectors, then the random rows                                                 |
| `random_only`       | The random rows                                                                            |
| `directed_only`     | The directed vectors                                                                       |
| `mixed`             | The random rows, and the directed vectors before them when `Parameter(always_include_directed=True)` (the default) |
| `test`              | The test vectors                                                                           |

`--vector-name=NAME` and `--vector-index=I` run a single directed vector, by its name or its position in `directed_vectors` (from 0), and take precedence over `--vector-mode` and `--nsamples`. A strategy without the requested directed vector yields no vectors, so the tests that use it are skipped ("got empty parameter set"). If no strategy in the run has the vector, for example because of a typo or an index that is out of range everywhere, pytest stops with a usage error that lists each strategy's directed vectors. It does not skip every test.

A directed or test vector holds one value per argument:
- a tuple or a list (for example read from a YAML file), in argument order: `(404,)` or `[404]`;
- a dict of argument names to values, in any order: `{"status_code": 404}`. Its keys must be the argument names, and a missing or unknown key fails with a message that names it (`Directed vector 'zeros' has unknown argument 'lenght' (did you mean 'len'?)`). Tuple and dict vectors can be mixed in one `Parameter`;
- a namedtuple, placed by its field names, which must be the argument names;
- `pytest.param(..., marks=...)` with one of these, to mark a row: `pytest.param(0, 1, marks=pytest.mark.xfail)` or `pytest.param({"addr": 0, "len": 1}, marks=...)`. Its values are checked like the others. It cannot have an `id=`, because the vector's name is its ID.

A bare value such as `404` or `"a"` is an error, with a hint: for a one-argument strategy write `(404,)` or `{"status_code": 404}`. A dict is always a named vector, so a dict value for a one-argument strategy is written `({"a": 1},)` or `{"cfg": {"a": 1}}`, and in a `pytest.param` only `pytest.param({"cfg": {"a": 1}}, marks=...)`. A dataclass or pydantic instance as a vector is not supported yet; use a dict. Vector names are non-empty strings; identifier-like names are easiest to select with `-k` (see [Test IDs and Selecting Rows](#-test-ids-and-selecting-rows)).

`directed_vectors` and `test_vectors` are read-only mappings of names to Vectors (a `pytest.param` vector stays a `pytest.param` whose values are a Vector): change them with `add_directed_vector()`, `remove_directed_vector()` and the `test` equivalents.

### 6. Sequence Testing & Exhaustive Generation

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
def matrix_strategy():
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"]))
    )
```
Running with `pytest --nsamples=auto` generates 6 tests in order: `(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')`, named `x=1-y=a`, `x=1-y=b` and so on. The leftmost arg is the slowest counter.

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
def mixed_strategy():
    return Parameter(
        # Deterministic: iterate through all user roles in order
        TestArg("role", rng_type=Series(["admin", "user", "guest"])),
        # Random: generate a fresh random ID for each role
        TestArg("id", rng_type=RNGInteger(1, 1000))
    )
```
Running `pytest --nsamples=auto` generates 3 tests (one per role, `role=admin`, `role=user` and `role=guest`), each with a random ID.

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
- A strategy with no `Series` or `RNGSequence` argument has nothing to enumerate. It falls back to its own `nsamples` (see [Per-Strategy Sample Count](#7-per-strategy-sample-count)), or 10 random samples.
- The number of combinations is checked before any is generated. More than 100,000 rows fails collection with a message that shows the product (`a=1,000 x b=1,000`). Raise the limit for one strategy with `Parameter(max_exhaustive=...)`, or for the project with the `strategies_max_exhaustive` ini option. The same limit applies to `per_sequence_samples=True`, where each combination counts `K` rows.

**Constraints on `Series` in finite mode:**
With a finite `--nsamples`, a `Series` combination that the constraints reject is skipped, and the cycle continues with the next combination. If the strategy also has random arguments, they are redrawn up to `max_retries` times before the combination is skipped. Each such skip emits a `PytestStrategiesWarning`: raise `max_retries`, or relax the constraint if that combination should be tested. If a whole cycle of combinations yields no valid vector, collecting the test fails.

**`n` samples per sequence combination (`per_sequence_samples`):**
By default a finite `--nsamples=K` gives `K` rows in total, shared among the sequence values. Pass `per_sequence_samples=True` to get `K` random rows for *each* combination of the `Series`/`RNGSequence` arguments instead:

```python
@register("per_device")
def per_device():
    return Parameter(
        TestArg("device", rng_type=Series(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        per_sequence_samples=True,
    )
```
With the default of 10 samples this runs 20 tests: 10 for `devA` (`device=devA-rand-0` to `device=devA-rand-9`), then 10 for `devB`, each with a fresh random `width`. Several sequence arguments multiply (2 devices × 3 modes × `K`), and a value listed twice counts twice. Combinations follow declaration order for `RNGSequence` too, and directed vectors are still placed first. Under `--nsamples=auto` the flag has no effect: every combination runs once. A combination whose random arguments the constraints reject `max_retries` times in a row gets fewer rows and a `PytestStrategiesWarning`; collecting the test fails only if no combination yields a row. The warning names the strategy and the test.

#### Skipping when a sequence is empty
When the values come from configuration, there may be none: a testbench without any Esm peripheral, for example. An empty `Series` or `RNGSequence` normally fails collection. Give it a `skip_if_empty` reason instead (a keyword argument), and the strategy contributes a single skipped row with that reason, so each test that uses it is skipped:

```python
from pytest_strategy import Parameter, RNGInteger, RNGSequence, TestArg, register, strategy

# From your testbench configuration, e.g.
# [p.channel for p in config.peripherals.values() if p.type == "Esm"]
ESM_CHANNELS = []

@register("esm_rw")
def esm_rw():
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(ESM_CHANNELS, skip_if_empty="no Esm peripheral in this testbench config")),
        TestArg("wdata", rng_type=RNGInteger(min=0, max=255)),
        per_sequence_samples=True,
    )

@strategy("esm_rw")
def test_rw(channel, wdata):
    ...
```
With channels, the option changes nothing. Without any, `pytest -rs` shows `SKIPPED [1] test_esm.py:<line>: no Esm peripheral in this testbench config`, and the test's ID is `test_rw[skipped]`. A test with other parametrization (a stacked `@pytest.mark.parametrize`, a parametrized fixture) is skipped once per combination of it. The skip applies in every `--vector-mode` and with `--nsamples=auto`, and directed and test vectors are skipped too. A `--vector-name` or `--vector-index` that names one of the strategy's directed vectors also gives the skipped test. The reason must be a non-empty string. It also applies when the predicate rejects every value. The test's signature (or record parameter) is still checked against the strategy, so a mismatch fails collection even on a configuration without values. `skip_if_empty` works for strategies that return a `Parameter`.

### 7. Per-Strategy Sample Count

By default the number of generated vectors is controlled globally by `--nsamples` (10 when unset). A strategy can declare its own count by passing `nsamples` to its `Parameter`:

```python
@register("edge_heavy")
def edge_heavy_strategy():
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

Directed vectors are added on top of these, according to `--vector-mode`. With `Parameter(per_sequence_samples=True)`, a finite count applies to each `Series`/`RNGSequence` combination instead of the whole strategy (see [`per_sequence_samples`](#6-sequence-testing--exhaustive-generation)). A row keeps its values when the count grows: `--nsamples=50` gives the rows of `--nsamples=10`, with the same values, and 40 more.

### 8. Record Parameters

Instead of one test parameter per strategy argument, a test can take a single record, a dataclass whose fields are the strategy's arguments:

```python
from dataclasses import dataclass

import pytest

from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy


@dataclass
class Point:
    x: int
    y: int


@register("points")
def points_strategy():
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

The rule: **a test receives the row as one record when (1) neither the test nor any fixture it uses asks for one of the strategy's argument names, and (2) exactly one test parameter is annotated with a record type whose fields are exactly those names. Otherwise the strategy passes its arguments by name, one test parameter each.**

- "Asks for" covers the test's parameters, `usefixtures`, autouse fixtures and what those fixtures ask for in turn. So a fixture that takes `x` and `y` and builds the object itself receives the arguments, with or without `validate_signature`, which does not change the choice.
- A test parameter is one that pytest fills: no default, not `*args` or `**kwargs`, and not `self`, `cls` or a built-in fixture. The other parameters are left alone, in any position.
- The record types are dataclasses, pydantic dataclasses included, once `Annotated[...]` and generic arguments are stripped (`Pair[int]` is `Pair`). Their fields are the `init=True` fields: fields with `init=False` are not counted (nor shown in test IDs), and `kw_only` fields work. A strategy with one argument can use a record too.
- NamedTuple, TypedDict and pydantic models are recognized but not supported yet: a test whose record would be one fails collection with "not supported yet". Their fields are `_fields`, the required and optional keys, and the `model_fields` names (not aliases). Because they are recognized now, supporting them later cannot change which parameter an existing test uses.
- `Optional[...]` and other unions, pydantic v1 models, attrs classes, `Vector` and annotations that cannot be resolved are not record types. String annotations (`from __future__ import annotations` or quoted names) work, as long as the class is defined at module level.
- Two parameters that both match fail collection, naming both. If a single parameter has a record type but its fields do not match, collection fails with a message that lists the missing and extra fields.

### 9. Strategy Files and Scoped Names

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

Values that a strategy file draws when it is imported come from a random stream of their own, derived from the seed and the file's path relative to the rootdir, so `--rng-seed` reproduces them whatever else was collected first. A file outside the rootdir is keyed by its relative path too (`../shared/strategies.py`), so two checkouts in different folders get the same values. An `RNG.seed()` call in a strategy file changes only the rest of that file's draws.

**Failures.** A strategy file that fails to import does not stop the run. The plugin prints `pytest-strategies: Warning - Failed to load <path>: <error>`. A file that calls `pytest.skip(..., allow_module_level=True)` or `pytest.importorskip()` at module level is skipped, and that is reported with `-v`. A "Strategy 'name' not found" error names the test, suggests close names, and lists the files that failed to load or were skipped, and the files with a strategy file name that mention `register` but were not imported because they have no registration decorator. `pytest -vv` prints each loaded file, and `pytest --list-strategies` lists the registered strategy names (with the file of each registration when a name is registered in several folders) and exits, without calling any factory.

### 10. Configuration-Dependent Strategies

Some vectors depend on configuration that is only known when the session runs, such as a testbench description whose file is named on the command line. Strategy factories run before any fixture exists, so they cannot use one. Instead, implement the `pytest_strategies_context` hook in a `conftest.py` (or a plugin): what it returns is passed as `ctx` to every factory with a `ctx` parameter that a test in that `conftest.py`'s folder, or below it, uses.

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
def esm_rw(ctx):
    channels = [p.channel for p in ctx.peripherals.values() if p.type == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(channels, skip_if_empty="no Esm peripheral")),
        TestArg("wdata", rng_type=RNGInteger(min=0, max=255)),
        per_sequence_samples=True,
    )
```

Tests use the strategy as usual, with `@strategy("esm_rw")`. With two Esm channels in the configuration they run 10 writes per channel; with none they are skipped with the reason.

#### A context per folder

- Each test gets the context of its own folder, so one test tree can hold two testbench configurations. The plugin calls the implementations it can see from there in this order: `tryfirst` ones, then the `conftest.py` files from the test's folder upward, then the other plugins (last registered first), then `trylast` ones. The first that returns something other than `None` answers: the nearest `conftest.py` wins, one that returns `None` defers to the folder above, and a plugin answers only where no `conftest.py` does, whatever order they were registered in.
- A `wrapper=True` implementation runs around the one that answered and can change its answer by returning a new object (`{**ctx, "extra": 1}`); a child folder whose `conftest.py` returns `None` shares its parent's wrapped object. A wrapper must leave the object it receives as it is, because the folders that do not see the wrapper get that object too: when the object's fingerprint (below) changed while the wrappers ran, the tests of the folders that see the wrapper fail with an error that says so. A wrapper that returns the very object it received (one that only checks or logs it) gives its folders the context of the folders without it, with the same label. Each implementation runs once, for the first folder that asks, so a wrapper's code before its `yield` runs after the implementations it wraps.
- A factory registered in another folder gets the context of the test's folder, the way a fixture override follows the test that requests it.
- Each implementation that is not a wrapper is called at most once per session, the first time a factory with a `ctx` parameter (or `strategies_ctx` or `get_context()`, below) needs it, and its result is reused: tests in folders that end at the same implementation share one object. A wrapper runs once for each implementation that answers under it, and again for each other set of wrappers it is in. Factories without `ctx` are called as before and never trigger it.
- When no implementation returns a value, a `ctx` parameter keeps its default (or a value bound with `functools.partial`), and is `None` without one.
- If an implementation raises, each test module that uses a factory with `ctx` in a folder that consults it fails collection with `Strategy factory '<name>' has a 'ctx' parameter, but the pytest_strategies_context hook raised <error>`. `pytest.fail()` in the hook is reported as it is. The hook can also call `pytest.skip(..., allow_module_level=True)` to skip those modules.
- When a factory fails with `ctx` set to `None` while a `conftest.py` in another folder implements the hook, the error adds which folders implement it, so that the hook can be moved to a common parent `conftest.py`.
- Random draws in the hook come from a stream of their own, derived from the seed and started anew for each implementation, so they are reproduced by `--rng-seed`, do not depend on which tests run, and do not change any test's vectors.
- Under pytest-xdist every worker calls the hook, so it must return the same configuration in each: a run whose workers built different contexts fails (see [pytest-xdist](#pytest-xdist) below).
- `export_strategies()`, which has no test, gives a factory the context of the folder its file is in, as a test there gets it, or the rootdir's for a file outside the rootdir or in an installed package inside it (a `site-packages` or `dist-packages` folder below the rootdir). When pytest has not loaded a `conftest.py` of that folder, or of a folder between it and the rootdir, because the run collected no test there (`pytest tests/a` leaves `tests/b/conftest.py` out), that context cannot be known: a factory with `ctx` there is not called, and its entry reads `{"unavailable": "tests/b/conftest.py was not loaded in this session"}`.

#### The context in fixtures: `strategies_ctx` and `get_context()`

The hook only has to describe the configuration the vectors depend on. Your testbench fixture can build on the same object instead of parsing the configuration a second time:

```python
# conftest.py
import pytest

from testbench import Testbench  # your project's code

@pytest.fixture(scope="session")
def tb(strategies_ctx):
    return Testbench(strategies_ctx)
```

```python
# tests/tb_a/conftest.py: a folder with its own pytest_strategies_context
import pytest

from pytest_strategy import get_context
from testbench import Testbench  # your project's code

@pytest.fixture(scope="session")
def tb_a(request):
    return Testbench(get_context(request.config, __file__))
```

- `strategies_ctx` is a session fixture of the plugin. It returns the object the factories of the tests that use it received, and computes it when no factory needed it yet.
- A session fixture has one value, so the tests that use it must share one context. When they are in folders whose contexts come from different implementations, each of them fails with `strategies_ctx is a session fixture, but the tests that use it have different contexts (conftest.py: ...; tests/tb_a/conftest.py: ...)`, which names each context's first test. Deselecting one folder's tests makes the run pass; a folder with its own implementation uses `get_context()` in its `conftest.py` instead. The tests that use it are those that request it, directly or through their fixtures, except the tests of a folder whose `conftest.py` defines its own `strategies_ctx` fixture, which does not request the plugin's. When none does and a test asks for it through `request.getfixturevalue()`, every test of the run counts. When some do, a test that asks for it that way gets their context, whichever test asks first, and fails with the message after its setup or its body when its own folder's context is another one. So does a test that gets a fixture whose setup asked for it that way, directly or through other fixtures, when the test gets the value (or the error) that setup cached, whichever test it ran for; a function-scoped fixture that asks only for some tests leaves the others alone. An error the test raised keeps its traceback, and the message follows it in a `pytest-strategies` section of that test's report only, also when other tests get the same error from a fixture's cache.
- `pytest_strategy.get_context(config, path)` returns the context of the folder of `path` (a file or a folder): the object a test there gets. A folder whose `conftest.py` pytest has not loaded, because no test there was collected, gets the context of the nearest loaded `conftest.py` above it. Call it from fixtures or hooks: in `pytest_configure`, or while `conftest.py` files are imported, it sees only those loaded so far. It raises `RuntimeError` for a config that no running session uses.
- Both raise what the implementation raised, as it is: a `pytest.skip` in the hook skips the tests that use them.

#### The context fingerprint

After the collection, the plugin prints a fingerprint of the contexts the run computed, so you can tell whether two runs (a CI job and your rerun) built the same configuration. Several contexts are named by the `conftest.py` (or plugin) that answered:

```text
pytest-strategies: context 976bcfdf
pytest-strategies: contexts conftest.py 976bcfdf, tests/tb_a/conftest.py b1e1b237
```

- The fingerprint is the first 8 hex characters of the SHA-256 of a canonical encoding of the object. It is computed when the hook returns the object, before any factory or fixture receives it, so a factory or a test that changes the object later changes no fingerprint. Nothing is printed when no factory with `ctx` ran, or when every context is `None`; the line appears with `-q` and `--collect-only` too. Under pytest-xdist the controller collects no tests, so it prints the line the workers printed after their collection at the end of the run, in its terminal summary.
- The encoding depends only on what the object holds. Sets are sorted, so the hash order of `PYTHONHASHSEED` does not matter. Paths inside the rootdir are written relative to it, so two checkouts agree. A pydantic v2 model is written as its `model_dump()`, which leaves out `Field(exclude=True)` fields and keeps a `SecretStr` masked. Dataclasses, attrs classes and NamedTuples are written field by field, `SimpleNamespace` and `argparse.Namespace` objects by their attributes, mappings as their pairs in order, floats, dates, `Decimal` and `UUID` as text, and classes and Enum members by their qualified names, never their modules. Anything else is written as its repr, without memory addresses (` at 0x7f...` inside a `<...>` repr, and a mock's `id='140...'`; an address a repr of its own shows, such as `Periph('uart0' at 0x40001000)`, is kept), and with the items of the sets it shows as Python does (`{'b', 'a'}`) sorted. An object that keeps the default repr (`<Plain object at 0x...>`) is in the fingerprint by its type alone, and the line says so: `context 976bcfdf (partial: Plain)`. A repr that shows a set in another form (`",".join(tags)`) should sort it. An object that cannot be encoded (its repr raises) gives `unavailable`, and never fails the run.
- Leave volatile values out of the context (temporary paths, process IDs, times), or mark them `Field(exclude=True)` in a pydantic model, so that the fingerprint stays the same from one run to the next.
- When tests fail, the line that says how to reproduce the run ends with the contexts their factories received: `pytest-strategies: reproduce with --rng-seed=S (context 976bcfdf)`, or `(contexts conftest.py 976bcfdf, tests/tb_a/conftest.py b1e1b237)`. Under pytest-xdist the workers send them to the controller. Tests whose setup or call failed count; an error in a test's teardown alone adds no context. The `pytest-strategies` section of each failed row shows the fingerprint of its factory's context in its `context` line (see [Reproducing a failure](#reproducing-a-failure)).
- `-v` adds a "Contexts" block to the Strategy Summary, with each context's label, fingerprint and number of tests whose factories received it. `item.stash[VECTOR_KEY].context` holds the fingerprint for the rows of a factory that received `ctx`, and `None` for the others (see [Per-Test Metadata](#-per-test-metadata)).

### 11. Metadata Export

`export_strategies()` returns every registered strategy as a JSON document, for analysis or integration with other tools:

```python
from pytest_strategy import export_strategies

# "json" is the only format; the argument is keyword-only
json_data = export_strategies(format="json")
print(json_data)
```

For a `tests/dma/strategies.py` that registers `burst` with two arguments, a directed vector and a named constraint, and a run with `--rng-seed=1` (shortened):

```json
{
  "schema": 1,
  "kind": "strategies",
  "generator": {"name": "pytest-strategies", "version": "4.0.0"},
  "seed": 1,
  "nsamples": 10,
  "strategies": [
    {
      "name": "burst",
      "origin": {"folder": "tests/dma", "file": "tests/dma/strategies.py", "qualname": "burst", "line": 4},
      "context": null,
      "parameter": {
        "schema": 1,
        "arguments": [
          {"name": "addr", "description": "", "python_type": "int", "validator": false,
           "source": "rng", "rng": {"type": "RNGInteger", "min": 0, "max": 4095, "predicate": false}},
          {"name": "len", "description": "", "python_type": "int", "validator": false,
           "source": "rng", "rng": {"type": "RNGInteger", "min": 1, "max": 64, "predicate": false}}
        ],
        "directed_vectors": [{"name": "zeros", "id": "directed-zeros", "values": {"addr": 0, "len": 1}}],
        "test_vectors": [],
        "constraints": [{"name": "no_4k_cross", "enabled": true}],
        "always_include_directed": true,
        "max_retries": 100,
        "nsamples": null,
        "per_sequence_samples": false,
        "max_exhaustive": null,
        "skip_reason": null
      }
    }
  ]
}
```

- `seed` and `nsamples` are the run's seed and `--nsamples` value (10 without the option).
- `strategies` holds one entry per registration, a name registered in several folders once per folder, sorted by name and folder. An entry has the strategy's `name`, its `origin` (`folder` and `file`, relative to the rootdir in posix form or absolute outside it, `qualname` and `line`), its `context` (the fingerprint of the context its factory received, or `null`) and one of: `parameter`, the factory's `Parameter.to_dict()`; `error`, such as `{"type": "RuntimeError", "message": "boom"}` for a factory that raised; or `unavailable` (see [A context per folder](#a-context-per-folder)).
- In `Parameter.to_dict()`, each argument has its `source`: `"rng"` with `rng`, its RNG type's `to_dict()` (a custom type lists its public attributes), or `"value"` with its `value`. Each constraint has `enabled`, `false` for one that `--strategy-constraint-off` turns off in this run.
- Values keep their type: JSON for None, bools, ints, strings and finite floats, `{"$float": "nan"}`, `{"$enum": "Color", "member": "RED"}`, or `{"$repr": "b'\\x00'", "$type": "bytes"}` for anything else, so the text has no NaN and a strict `json.loads` reads it. `VectorInfo.to_dict()` (see [Per-Test Metadata](#-per-test-metadata)) uses the same encoding.
- **How the schema evolves.** A reader should ignore the keys it does not know, and the values of `kind` and `source` it does not know, and read an unknown `$`-tagged object like `$repr`: 4.x releases may add them within schema 1. Removing, renaming or retyping a key changes the schema number.

In a pytest session every strategy file is imported first. Each factory gets the inputs it declares, as at collection, with the session's options: `nsamples` is the `--nsamples` value, `"auto"`, or 10 without the option, and `ctx` the context of the factory's own folder (see [A context per folder](#a-context-per-folder)).

Its random draws come from a stream derived from the seed, the strategy name and the factory's module name, or the folder of its file for a factory in a strategy file, a test module or a `conftest.py`, so every call in a session, and in every environment and checkout, exports the same values. A file counts as a strategy file, a test module or a `conftest.py` when it has such a name and is inside the rootdir, below a `testpaths` entry, or imported by its path in the session (a test module collected from a folder named on the command line) and, when `consider_namespace_packages` is off (pytest's default), not a module of a regular package that Python holds under its package name (`acme.strategies` for `acme/strategies.py` next to `acme/__init__.py`), the name pytest then imports it under in every import mode. A module named like one elsewhere, such as a library's `extacme/strategies.py` on `sys.path`, is keyed by its module's name, so every checkout folder exports the same values.

A factory in a package module named like one inside the rootdir (`src/acme/strategies.py`, `src/acme/test_utils.py`) is therefore keyed by its folder in a checkout or an editable install and by its module's name when installed, so the two export different values; rename the module (`acme/catalog.py`) to avoid that. Outside the rootdir and the `testpaths`, these are keyed by their folder in a run that collects that folder (one that names it on the command line) and by their module's name in a run that does not: (a) a helper named like one, not in a regular package, imported by its module's name; (b) a regular package's module that Python holds only under a longer namespace-package name (`ns.acme.strategies`); (c) with `consider_namespace_packages = true`, also a regular package's module imported by its name. List such a folder in `testpaths` to key them by their folder in every run.

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

A fixture can take strategy arguments too: with a fixture `def user(username, age)`, a test `def test_login(user)` decorated with `@strategy("user_strategy")` gets a user built from each row. The testbench context reaches fixtures through `strategies_ctx` and `get_context()` (see [The context in fixtures](#the-context-in-fixtures-strategies_ctx-and-get_context)). A fixture's random draws follow `--rng-seed` (see [Draws outside the rows](#draws-outside-the-rows)).

## 🔖 Test IDs and Selecting Rows

Each row's test ID names the row instead of showing its values, so a row keeps its node ID for every seed:

| Row                                                       | ID                                    |
| --------------------------------------------------------- | ------------------------------------- |
| Directed vector `zeros`                                   | `test_write[directed-zeros]`          |
| Test vector `max`                                         | `test_write[test-max]`                |
| Random row 3                                              | `test_write[rand-3]`                  |
| Random row 1 of a finite run with the `Series` value 2    | `test_esm[ch=2-rand-1]`               |
| A row of `--nsamples=auto`                                | `test_esm[ch=0-dev=b]`                |
| The row of an empty `skip_if_empty` sequence              | `test_esm[skipped]`                   |

- Random rows are numbered from 0. An argument's value is in the ID exactly when the row enumerates it: the `Series` arguments in finite mode, the `Series` and `RNGSequence` arguments with `per_sequence_samples=True`, and every `Series` and `RNGSequence` argument under `--nsamples=auto`. Several of them are joined with `-` in declaration order. Rows of `--nsamples=auto` have no `rand-` part, because there is one per combination.
- An enumerated value is written `ARG=TEXT`. The text of a bool or `None` is `str(v)`, of an Enum member its name, of an int or a float its `repr`, of a class or function its `__name__`, and of a string the string itself, when it is non-empty, at most 40 characters, printable, and has no whitespace, `=`, `~`, `[` or `]`. A value listed twice gets `~1` the second time (`Series([1, 1, 2])` gives `ch=1`, `ch=1~1` and `ch=2`). A value without text (a tuple, bytes, another string) is written by its position, `ARG<position>`: `cfg0`, `cfg1`. So are two values that would get the same text (`1` and `"1"`).
- Under `--nsamples=auto`, every seed gives the same set of IDs as long as the constraints drop no combination; only the order of `RNGSequence` rows changes. A `Series`-only strategy with more rows than combinations repeats values, as `ch=0-rand-0` and `ch=0-rand-1`.
- A test with a stacked `@pytest.mark.parametrize`, or several `@strategy` decorators, gets IDs joined with `-` in node-ID order, the decorator closest to the `def` first: `test_two[directed-zeros-fast]`.

**Selecting rows with `-k`.** pytest matches `-k` against the parts of the ID:
- `-k zeros` selects the directed vector `zeros`, `-k directed` every directed row, and `-k "not rand"` leaves out the random rows.
- `-k "rand-3"` also matches `rand-30` to `rand-39`. To select one random row, use `-k "test_write[rand-3]"` or its node ID.
- `-k` cannot contain `=` (pytest's grammar), so select the rows of a sequence value by node ID: `pytest "tests/test_esm.py::test_esm[ch=2-rand-1]"`.
- Directed and test vector names can be any non-empty string, but a name in `-k` can hold only letters, digits and `_ - . : / + \ [ ]` (not whitespace, `=`, parentheses, commas or quotes), and a non-ASCII name needs pytest's escaped form (`-k 'caf\xe9'`): identifier-like names are easiest to select.

**Rerunning one row.** `pytest "tests/test_dma.py::test_write[rand-3]" --rng-seed=S` runs that row with the values it had in the run with seed `S`, for any `--nsamples` that still generates it: rows keep their values when there are more rows or fewer. A row the run does not generate (`rand-12` with the default of 10 rows) gives pytest's own "not found" error, so give the same `--nsamples` as the run. When a test fails, the plugin prints this command for each failed row (see [Reproducing a failure](#reproducing-a-failure)), and `--lf` reruns the failed rows with their seed (see [Rerunning failures with `--lf` and `--sw`](#rerunning-failures-with---lf-and---sw)).

**`strategies_ids = values`.** The ini option `strategies_ids` sets the format of the IDs: `names`, the default, or `values`, the 3.0 IDs built from the values (`addr=0,len=1`), which change with the seed. A run can switch with `-o strategies_ids=values`. Any other value is a usage error.

**`Parameter(ids=...)`** sets the test IDs of one strategy: `"names"` or `"values"` overrides the ini option, and a function builds them. The function is called once per row while the tests are collected, with the row's `VectorInfo` (see [Per-Test Metadata](#-per-test-metadata)), whose `id` is the row's ID in the ini option's format. It returns the ID to use, or `None` to keep that one:

```python
def by_mode(info):
    # (8, "le") has no text of its own: the default ID is mode0-rand-1
    if info.kind == "random":
        width, order = info.values.mode
        return f"w{width}{order}-rand-{info.index}"
    return None  # keep directed-zeros

Parameter(
    TestArg("mode", rng_type=Series([(8, "le"), (16, "be")])),
    TestArg("addr", rng_type=RNGInteger(0, 4095)),
    directed_vectors={"zeros": ((8, "le"), 0)},
    ids=by_mode,  # test_write[directed-zeros], test_write[w8le-rand-0], test_write[w16be-rand-0], ...
)
```

Build IDs from what does not depend on the seed: the row's `kind`, `name` and `index`, and the values of its `enumerated` arguments. An ID built from a drawn value changes with the seed, as in 3.0, so `-k` expressions, `--deselect` lists and CI history lose track of the row from one run to the next (the plugin's rerun commands, which carry the seed, still work).

Rows that get the same ID are suffixed the way pytest suffixes duplicate IDs (`len=17_0` and `len=17_1`, `odd0` and `odd1`), so they also pass under pytest's `strict_parametrization_ids`, and `item.stash[VECTOR_KEY].id` is the suffixed ID. Anything but a non-empty `str` or `None`, or an exception, fails the collection of the test: `In test_write: Strategy 'burst': ids= returned 42 for row rand-3; return a str or None`. The skipped row of an empty `skip_if_empty` sequence keeps `skipped` without a call. A `pytest.param` vector cannot have an `id=`: the vector's name is its ID, so rename the vector or use `ids=`.

`ids=` and `strategies_ids` never change the generated rows. The `RNG` draws of a test's setup, body and teardown, and of its function-scoped fixtures, come from streams of the test's node ID (see [Draws outside the rows](#draws-outside-the-rows)), which contains the ID, so they change with it: reproduce a run with the same IDs.

## 🧾 Per-Test Metadata

The item of each strategy row carries a `VectorInfo` that describes the row, in `item.stash[VECTOR_KEY]`. `VectorInfo`, `VECTOR_KEY` and `VECTORS_KEY` are exported from `pytest_strategy`. The plugin stores it while the tests are collected, before any `pytest_collection_modifyitems` hook runs, so your hooks, fixtures and reports can read it:

```python
# conftest.py
import pytest

from pytest_strategy import VECTOR_KEY


def pytest_collection_modifyitems(items):
    # Run every directed vector before the random rows
    def rank(item):
        info = item.stash.get(VECTOR_KEY, None)
        return 0 if info is not None and info.kind == "directed" else 1

    items.sort(key=rank)


@pytest.fixture(autouse=True)
def _log_row(request):
    info = request.node.stash.get(VECTOR_KEY, None)
    if info is not None:
        print(f"{info.strategy} {info.id}: {info.values}")
```

| Field             | Value                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| `strategy`        | The strategy's resolved name, as in the `-v` summary                                                    |
| `origin`          | Where the factory is defined, `"tests/dma/strategies.py:12"` (relative to the rootdir), or `None`        |
| `kind`            | `"directed"`, `"test"`, `"random"`, `"exhaustive"` (a row of `--nsamples=auto`) or `"skipped"`          |
| `name`            | The directed or test vector's name, else `None`                                                         |
| `index`           | The vector's position in `directed_vectors` (what `--vector-index` takes) or `test_vectors`; the row's number within its combination for a random row; its position among the combinations for an exhaustive row; `None` for `"skipped"` |
| `enumerated`      | The arguments the row enumerates, whose values are in the ID                                            |
| `values`          | The row, a `Vector` (`info.values.addr`); Nones for `"skipped"`                                         |
| `id`              | This strategy's part of the test ID, after `ids=` and duplicate suffixes, before pytest escapes it      |
| `seed`            | The run's seed                                                                                          |
| `context`         | The fingerprint of the context the factory received, or `None` when it declares no `ctx`               |
| `constraints_off` | The constraints turned off in this strategy                                                             |
| `streams`         | The version of the random streams the values come from (1)                                              |

- `info.to_dict()` gives the row as JSON-ready data with `"schema": 1`, its values in the export's encoding (see [Metadata Export](#11-metadata-export)).
- A test with several `@strategy` decorators has one `VectorInfo` per strategy in `item.stash[VECTORS_KEY]`, in the order of the node ID; `VECTOR_KEY` holds the first.
- Items without a strategy, and the item pytest makes for an empty parameter set, have none: use `item.stash.get(VECTOR_KEY, None)`.
- The infos also travel as the argument of a `strategy` mark on each row, which adds no `-k` keyword and no marker. `item.iter_markers("strategy")` yields the test's own `@strategy` mark first and the row's marks after it, and `item.get_closest_marker("strategy")` returns the test's own mark: read the stash instead.
- `VectorInfo` is frozen, and you read it rather than build it. Fields that later releases add come with defaults.

## 🎛️ Options

Control test generation directly from the command line:

| Option              | Description                                                             | Example                                     |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `--nsamples`        | Number of random samples per strategy (default 10), or per `Series`/`RNGSequence` combination for a strategy with `per_sequence_samples=True`. An integer overrides a strategy's own `nsamples`. `auto` enumerates the `Series`/`RNGSequence` arguments (see [What `--nsamples=auto` does](#6-sequence-testing--exhaustive-generation)). Any other value is a usage error. | `pytest --nsamples=50` or `--nsamples=auto` |
| `--vector-mode`     | Generation mode: `all`, `random_only`, `directed_only`, `mixed`, `test` (see [Directed and Test Vectors](#5-directed-and-test-vectors)) | `pytest --vector-mode=test`                 |
| `--vector-name`     | Run only the directed vector with this name                             | `pytest --vector-name=edge_case_1`          |
| `--vector-index`    | Run only the directed vector at this index (0-based, in definition order) | `pytest --vector-index=0`                 |
| `--rng-seed`        | Set seed for reproducibility. Without it, `--lf` and `--sw` reuse the seed of the failed run (see [Reproducibility](#-reproducibility)) | `pytest --rng-seed=42`                      |
| `--strategy-constraint-off` | Turn named constraints off for this run: `NAME` in every strategy, `STRATEGY:NAME` in one; comma-separated, repeatable (see [Rows and Constraints](#4-rows-and-constraints)) | `pytest --strategy-constraint-off=dma_burst:aligned` |
| `--list-strategies` | List the registered strategy names and exit                             | `pytest --list-strategies`                  |

Ini options go in your pytest configuration, and `-o NAME=VALUE` sets one for a single run:

| Ini option                  | Default  | Description                                                                                         |
| --------------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| `strategies_ids`            | `names`  | The format of the test IDs: `names` (`rand-3`) or `values` (`addr=0,len=1`, the 3.0 format); see [Test IDs and Selecting Rows](#-test-ids-and-selecting-rows). Any other value is a usage error |
| `strategies_max_exhaustive` | `100000` | The most rows `--nsamples=auto` or `per_sequence_samples=True` may generate for one strategy; `Parameter(max_exhaustive=...)` overrides it |

In a native TOML configuration (pytest 9's `pytest.toml`, or `[tool.pytest]` in `pyproject.toml`), write the values as strings: `strategies_max_exhaustive = "500000"`.

The plugin also provides the session fixture `strategies_ctx` (see [The context in fixtures](#the-context-in-fixtures-strategies_ctx-and-get_context)) and the `strategy` marker that `@strategy` adds. Command-line options added in later releases start with `--strategy-` and ini options with `strategies_`; there are no command-line copies of ini options, since `-o` sets those.

With `-v`, a "Strategy Summary" section lists each strategy with the number of tests that use it, their directed and random rows (and their test rows, the exhaustive rows of `--nsamples=auto` and the skipped row of an empty `skip_if_empty` sequence, when there are some), and where the sample count came from (`--nsamples`, `Parameter(nsamples=)` or the default), the rejections per constraint and the constraints turned off, followed by the contexts' fingerprints.

## 🔄 Reproducibility

The RNG seed of each run is printed in the pytest report header:
```text
pytest-strategies: RNG seed = 1763926297314361000
```

### Reproducing a failure

When tests fail, the plugin also prints how to rerun them with the same vectors, after the failure tracebacks and before the short test summary, even with `-q`:
```text
pytest-strategies: reproduce with --rng-seed=1763926297314361000
pytest-strategies: failed rows:
  pytest 'tests/dma/test_write.py::test_write[rand-3]' --rng-seed=1763926297314361000  # burst random 3
  pytest 'tests/dma/test_write.py::test_write[directed-zeros]' --rng-seed=1763926297314361000  # burst directed zeros
```
When the factories of the failed tests received a context, the first line ends with its fingerprint, such as `(context 976bcfdf)` (see [The context fingerprint](#the-context-fingerprint)). Then comes, for each strategy row whose setup or call failed, the command that runs that row alone with the same values, from the folder pytest was started in, and what the row is. Below `-v` at most 10 rows are listed, followed by `... and 4 more`; `-qq` prints only the first line. An error in a test's teardown alone lists nothing.

Under its traceback, each failed strategy row also gets a `pytest-strategies` section that says what the row is and how to run it again:
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
- `strategy` names the strategy and where its factory is defined, and `vector` the row: `directed-zeros (directed vector 'zeros', #0)`, `test-max (test vector 'max', #1)`, `ch=2 (exhaustive row 5)`. `values` shows each argument's value by its repr (by its type's name when the repr shows a memory address, and as `<Reg: repr() raised RuntimeError>` when it raises), cut at 4,000 characters below `-vv`. The `context` line appears only when the factory received `ctx`. A test with several `@strategy` decorators gets one block per strategy and one `rerun` line.
- The command adds every option that decides which rows exist and what they hold: `--nsamples`, `--vector-mode`, `--vector-name` and `--vector-index` when the run had them (from the command line, `addopts` or a conftest), the `-o strategies_*` overrides, `-c` and `--rootdir` as given, and the constraints the run turned off in the row's own strategies, as `--strategy-constraint-off=STRATEGY:NAME` items, so that rerunning one module never names a constraint only another module has. Arguments are quoted for the platform's shell. What shapes the context (an environment variable, an option of your own) is not added: compare the `context` lines.
- pytest names a test file outside the rootdir (`pytest -c ci/pytest.ini` with the tests in `tests/`) by the path the run started from (given on the command line, or the folder pytest runs in), so a rerun of its node ID would give the row another node ID and other values. Its command starts from the run's own paths instead, with the run's `--ignore` and `--ignore-glob` options, and selects the row with `-k`: `pytest . --rng-seed=S -c ci/pytest.ini -k 'test_write[rand-3]'`, with the module's name too (`test_dma.py and test_write[rand-3]`) when another test has the same name. When no `-k` expression selects only that row (a name with `=`, or two modules of the same name), the command names the node ID, its section ends with a `note` line, and its row in the list with `(outside the rootdir)`: run with a `--rootdir` that contains the tests, such as `--rootdir=.`, to get a command that reproduces them.
- A failure that pytest reports as plain text, such as an XPASS of a `strict` xfail, is listed but gets no section. An error reported without a traceback, such as a missing fixture, gets the section below its report. Under pytest-xdist the workers send the rows to the controller.

A `--junitxml` report holds the same information, under pytest-xdist too:
- The failure (or error) text of each failed strategy row ends with its `pytest-strategies` section.
- The test suite gets the property `pytest_strategies.seed`, and `pytest_strategies.failed.0`, `pytest_strategies.failed.1` and so on, the commands of the list of failed rows, in the same order:
  ```xml
  <properties>
    <property name="pytest_strategies.seed" value="1763926297314361000"/>
    <property name="pytest_strategies.failed.0" value="pytest 'tests/dma/test_write.py::test_write[rand-3]' --rng-seed=1763926297314361000"/>
  </properties>
  ```
- With `junit_family = xunit1` or `legacy`, each failed strategy row's test case also gets string properties: `pytest_strategies.strategy`, `.kind`, `.name` (for a directed or test vector), `.index`, `.id`, `.value.<argument>` for each value (as in the section), `.seed`, `.context` and `.constraints_off` when set, and `.command`, the command to run from the rootdir. With several `@strategy` decorators they are numbered per strategy, in the order of the node ID: `pytest_strategies.0.strategy`, `pytest_strategies.1.strategy`. pytest's default family, `xunit2`, has no properties per test case in its schema, so there the test cases get none.

### Rerunning failures with `--lf` and `--sw`

`--lf` and `--sw` rerun the failed rows with the values they failed with. The plugin records the seed of each failed strategy row, with the options its rerun command adds, in pytest's cache (`pytest-strategies/failed-seeds`). A run with `--lf`, `--sw` or `--sw-skip` and no `--rng-seed` reuses the seed of the newest of the failed rows it reruns, among those whose file still exists and that the run collects: those the paths and node IDs on the command line select or, without any, those in the `testpaths` or in the folder pytest runs in. A test file outside the rootdir (`-c ci/pytest.ini` with the tests in `tests/`) counts by the name pytest gives it, its path from the path the run started from, so run `--lf` from the same paths as the failed run. `-k` and `-m` apply once the tests are collected, after the seed is chosen, so they do not choose it: give a path or a node ID instead. A line under the seed line says so:
```text
pytest-strategies: RNG seed = 1763926297314361000
pytest-strategies: seed reused from the failed run for --lf (--rng-seed overrides)
```
- The recorded options are not applied. When they differ from the run's, the line ends with them (`; recorded with --nsamples=13`, or `without --vector-mode=test`): add them to get the same rows. A recorded `-c` or `--rootdir` is kept relative to the rootdir, so it names the same file from any folder (an ini file outside the rootdir, as in `-c /dev/null`, by its absolute path), and agrees with a run that uses that ini file or rootdir without giving it (`pytest --lf` in `ci` after `pytest -c ci/pytest.ini`). A `--rootdir` the run gives agrees with rows recorded without it.
- Failed rows recorded under another seed are deselected, so pytest keeps them in its last-failed set, and the end of the run gives the command that reruns them, one per seed (and per set of options), even with `-qq`. It names their files, or their node IDs when a file also holds other failed tests, so it reruns no other failed test with values it did not fail with:
  ```text
  pytest-strategies: deselected 2 failed rows recorded under another seed; run them with:
    pytest --lf --rng-seed=1763926297314360000 tests/dma/test_write.py  # 2 rows
  ```
  For rows in files outside the rootdir, whose node IDs a path would change, the command starts from the run's own paths instead (`pytest --lf --rng-seed=S -c ci/pytest.ini .`), and leaves out the other failed tests these paths collect with `--deselect`, or with `--ignore` for a file inside the rootdir: pytest's `--lf` skips the files outside the rootdir once it collected such a file.
- When the run collects none, or not all, of the failed rows it reused the seed of (their test was deleted or renamed, or the run's options do not generate them), the end of the run says so and gives the command that reruns them, with their recorded options:
  ```text
  pytest-strategies: 1 failed row recorded under the reused seed was not collected; unless deleted or renamed, run it with:
    pytest --lf --rng-seed=1763926297314360000 --nsamples=13 tests/dma/test_write.py  # 1 row
  ```
- `--rng-seed` always wins and deselects nothing, and so does a `config.option.rng_seed` that a `conftest.py` sets in its `pytest_configure`. An `RNG.seed()` call there does not change the reused seed, as it does not change `--rng-seed`. `--ff` and `--nf` draw a new seed, because they run every test, and so does `--sw-reset`.
- A row leaves the record once it passes under its recorded seed and options. Under pytest-xdist the controller reads and writes the record and sends the seed to the workers. Without pytest's cache plugin (`-p no:cacheprovider`) there is no `--lf` or `--sw`, and nothing is recorded.

### Random streams

Each strategy and test pair draws from its own random streams ("streams v1"). Each argument of a random row draws from a stream derived from the seed, the strategy name, the test's node ID without its parameters (its file path relative to the rootdir, its class and its name), the row (the values it enumerates and its number) and the argument's name. The factory draws from a stream derived from the seed, the strategy name and the test's node ID without its parameters, so its draws do not change the rows. A random row that a constraint rejects is drawn again from the same streams, so the row's values are those of the first draw every constraint accepts. The derivation is versioned (`VectorInfo.streams`) and changes only in a major release.

What this gives you:
- A test gets the same vectors and node IDs whether you run the whole suite, one file or one test, in any collection order, with any `--import-mode`, on any OS, with any `PYTHONHASHSEED`, and under pytest-xdist.
- Two tests that use the same strategy get different random vectors, and so does a test method that two classes inherit.
- A row keeps its values when there are more rows (`--nsamples=50` keeps the rows of `--nsamples=10`), when it runs alone by its node ID, under any `--vector-mode`, `--vector-name` or `--vector-index`, and when directed or test vectors are added. Adding, removing or changing another drawn argument leaves this argument's values as they are, and reordering the enumerated arguments keeps every row's values. A constraint changes only the rows it rejects, so turning one off keeps the rows it never rejected, and appending a `Series` value keeps the other combinations' values.
- Values that a strategy factory draws itself are reproduced too, through `rng` or `RNG.generator()`, the generator the RNG types use (also for other random operations, such as `shuffle` or `gauss`).

What does not hold:
- Renaming an argument changes its values. Renaming the strategy, the test, its class or its file, or changing the rootdir, changes every row: keep the same rootdir, because the test's path relative to the rootdir is part of the stream (pytest uses the directory of your ini file, such as `pytest.ini`, as the rootdir when there is one).
- Adding or removing an enumerated argument, or changing whether an argument is enumerated (a `Series` against an `RNGSequence` in finite mode, `per_sequence_samples`, `--nsamples=auto`), changes every row.
- For `Series` and `per_sequence_samples`, which rows exist depends on the count and the sequence lengths, and on the seed when a combination runs out of retries; only each row's values are stable.
- Inserting a `Series` value that has no text of its own (a tuple, say) shifts the positions of the later values, and adding an `RNGSequence` value reorders its rows under `--nsamples=auto`.
- A new Python minor version may change what `randint`, `choice` and `sample` return for a seed, because Python only guarantees `random()`.

These hold when a row's random values come from its RNG types:
- A custom RNG type (a subclass of `RNGType`) should draw inside its `generate()`, from `RNG.generator()` or the `RNG.*` helpers called there, and keep no state between calls: a counter that walks a pattern makes row k depend on the rows drawn before it.
- A constraint that draws (`RNG.integer()`), or an RNG type that draws from a generator kept from the factory (its `rng`), draws from the plugin's generator outside the rows' streams: those values repeat for the same seed, tests and options, but change when other rows or tests change. The plugin then emits one `PytestStrategiesWarning` per test, naming the strategy and the test; under `filterwarnings = error` it fails the collection.
- A factory wrapped in `functools.cache` runs once, for the first test that uses it, so its draws come from that test's stream: it should not draw, because a test run alone could get other values.

### Draws outside the rows

The plugin draws from a `random.Random` instance of its own and never seeds Python's global `random` module. So your own use of `random` does not change the generated vectors, and `--rng-seed` does not reproduce what `random.random()` returns in your code, including in a factory: draw from `rng` or `RNG.generator()` there.

The seed also reproduces the `RNG` draws made while the tests run. Each phase of a test (setup, call and teardown) draws from a stream derived from the seed, the test's node ID and the phase, and each fixture's setup from a stream derived from the seed, the node ID of the test, class, module or package the fixture is set up for (none for a session fixture), its name, its parameter index and where it is defined (its module, or its file for a `conftest.py` or a test module, its function, and the folder, module or class it is registered for), so that a fixture overriding another of the same name, or one fixture function imported into two folders' `conftest.py` files, gets values of its own. So a test body's `RNG.integer()` and a fixture's draws are the same whether the test runs alone, in the suite, in another order or under pytest-xdist, also when a module- or session-scoped fixture is set up during that test's setup. A fixture's teardown (the code after its `yield`, or a finalizer) runs in the teardown of the test that ends the fixture's scope, and draws from that test's teardown stream. An `RNG.seed()` call in a test, a fixture, a factory or a strategy file changes only the rest of that stream's draws, and `RNG.refresh_seed(key=...)` gives a stream derived from the seed and the key, whatever ran before it.

To make plain `random` draws in test bodies follow the seed too, seed it for each test in your `conftest.py`:

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

Values that a strategy file draws when it is imported are reproduced by the seed (see [Strategy Files and Scoped Names](#9-strategy-files-and-scoped-names)). A draw at module level in a test module (for example `BASE = RNG.integer(0, 1000)`) comes from a stream derived from the seed and the module's path, so running one file alone gives it the same value.

Draws made when a `conftest.py` is imported do not follow the seed: pytest imports the conftest files of the rootdir and of the folders on the command line before the seed is set, so a node-ID rerun would import them at another time. Move such draws into the context hook, a fixture or a strategy file. For the same reason, a strategy file that a `conftest.py` imports at its top is executed with that `conftest.py`, not on its own stream, and its import-time draws do not follow `--rng-seed` either; import it inside the fixture or hook that needs it. A helper module that test modules or strategy files import (a `helpers.py` that is not a test module, a `conftest.py` or a strategy file) runs on the stream of the first module that imports it, so its import-time draws, and the draws that this module makes after importing it, change with what was collected before: running one file alone can give them other values. Move such draws into a fixture, the context hook or a strategy file as well.

### pytest-xdist

Runs with `-n` work with or without `--rng-seed`. The controller sends its seed to the workers, so they all generate the same tests, as long as the strategy factories and the context hook give the same result in every worker.

Test IDs name the rows instead of showing their values, so pytest-xdist does not notice workers that generate different values under the same IDs. The plugin checks it: when its session ends, each worker sends the fingerprint of each context it computed (by its label) and a digest of each strategy's values (the first 8 hex characters of the SHA-256 of its tests' node IDs and values, in the fingerprint's encoding, taken after the collection). When two workers differ on a context or a strategy, the controller names them in red at the end of the run, before the line that says how to reproduce it, and a run that would have passed (or collected no tests) fails with exit code 4:

```text
pytest-strategies: the xdist workers generated different vectors:
  context tests/tb_a/conftest.py: gw0 1a2b3c4d, gw1 9f8e7d6c
  values of strategy dma_burst: gw0 5e6f7a8b, gw1 0c1d2e3f
Make pytest_strategies_context and the strategy factories give the same result in every worker:
no temporary paths, process IDs, times, unseeded random values or lists built from sets
(leave them out, or use pydantic Field(exclude=True) in a context).
```

- Typical causes are a factory that draws from Python's global `random` (draw from `rng` or `RNG.generator()`), or that builds a list from a set of strings, whose order follows each worker's `PYTHONHASHSEED` (sort it), and a context holding a temporary path, a process ID or the time.
- A context differs even when the values agree, because tests can read it through `strategies_ctx`. The fingerprint is taken when the hook returns, so a test that changes the object on one worker does not count. Nor does the context of a folder whose `wrapper=True` implementation built it on that worker from an object a test had already changed (a folder first asked while the tests run): its fingerprint shows the change, so it is not compared.
- A context or a strategy that only one worker computed is not compared: a test that runs on one worker only can compute its folder's context through `strategies_ctx` or `get_context()`. A worker that crashed sent nothing and is left out.
- When the values change the test IDs (a `Series` built from the context), pytest-xdist itself stops with "Different tests were collected", and the plugin's message follows it, naming the cause.

## 🧭 Upgrading to 4.0

4.0.0 changes test IDs once and random values once more, and removes what 3.0 deprecated. The [CHANGELOG](CHANGELOG.md)'s 4.0.0 section starts with "Migrating from 3.x", a checklist with what you see for each change and what to do. In short:

- **Test IDs** name the rows (`test_write[rand-3]`) instead of showing their values. Update `-k` expressions, `--deselect` lists and anything else that names value IDs, or keep the 3.0 IDs for a while with `strategies_ids = values`.
- **Random values.** For the same seed, 4.0.0 gives random rows, and the values that factories, strategy files and the context hook draw themselves, other values than 3.x, so a seed recorded with 3.x does not reproduce that run's random rows. Directed and test vectors and `Series` values are the same, unless a factory draws them.
- **Factory inputs by name.** A factory parameter named anything but `nsamples`, `ctx`, `rng` or `options` needs a default: rename `def factory(n)` to `def factory(nsamples)`, or drop the parameter.
- **Tuple factories.** A factory that returns an `(argnames, samples)` tuple fails the collection of the tests that use it: return a `Parameter`, with one `TestArg` per argument and the fixed rows as `directed_vectors`. A fixed table with no random arguments fits `@pytest.mark.parametrize` better.
- **Removed APIs.** `TestArg(directed_values=..., test_values=..., always_include_directed=...)` raises `TypeError`: give the `Parameter` `directed_vectors=`, `test_vectors=` and `always_include_directed=` instead. The `TestArg` properties `directed_values`, `test_values` and `has_directed_values` are gone. `RNG.set_max_retries()` is gone: use `Parameter(max_retries=...)`. `configure()` and `Strategy.set_config()` did nothing: delete the calls.
- **Keyword-only options.** `TestArg`'s options after `rng_type`, `strategy(..., validate_signature=...)`, `export_strategies(format=...)` and the options of `generate_vectors()` must be passed by keyword.
- **Rows** are `Vector`s, tuples whose `type(v) is tuple` is false, and a namedtuple directed or test vector is placed by its field names, no longer by position. A bare value such as `404` as a vector is an error, and so is an `id=` on a `pytest.param` vector.
- **Record parameters.** A fixture that asks for a strategy argument's name makes the strategy pass its arguments by name, so a test that also takes a record of them fails collection; with `validate_signature=False`, a record parameter whose same-named fixture consumes the arguments now gets that fixture's value.
- **Export.** `export_strategies()` returns schema 1, with a `strategies` list instead of a mapping by name.

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
