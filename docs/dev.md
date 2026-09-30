# pytest_strategy

A pytest plugin for constrained-randomized test parametrization with directed testing support.

## Overview

`pytest_strategy` enables you to write powerful parametrized tests that combine:
- **Constrained random generation** - Generate test inputs with specific constraints
- **Directed testing** - Define specific test cases (edge cases, known bugs, etc.)
- **Reproducibility** - Seed-based random generation for consistent test runs
- **CLI control** - Run specific test vectors via command-line arguments

## Project Goal

The main goal is to make pytest tests more powerful by allowing you to:

1. **Generate randomized test parameters** with constraints (e.g., "integers between 1-100 that are even")
2. **Mix random and directed tests** seamlessly (e.g., "always test edge cases + 100 random cases")
3. **Control test execution** via CLI (e.g., "run only the 'edge_zero' test vector")
4. **Reproduce failures** using seed values

## Quick Start

### Basic Usage

```python
from pytest_strategy import Strategy, Parameter, TestArg
from pytest_strategy.rng import RNGInteger, RNGFloat, RNGChoice

# Define a strategy
@Strategy.register("test_addition_strategy")
def create_addition_samples(nsamples):
    # Return a Parameter; the plugin generates the vectors from it
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        directed_vectors={
            "zeros": (0, 0),
            "max": (100, 100),
        }
    )

# Use the strategy in a test
@Strategy.strategy("test_addition_strategy")
def test_addition(a, b):
    result = a + b
    assert result >= 0
    assert result == b + a  # commutative
```

### Running Tests

```bash
# Run with default 10 samples
pytest test_example.py

# Run with 50 random samples
pytest test_example.py --nsamples 50

# Enumerate Series/RNGSequence args (strategies without any use their own
# nsamples, or 10)
pytest test_example.py --nsamples auto

# Run only directed vectors
pytest test_example.py --vector-mode directed_only

# Run specific vector by name
pytest test_example.py --vector-name "zeros"

# Run the directed vector at index 0
pytest test_example.py --vector-index 0

# Set seed for reproducibility
pytest test_example.py --rng-seed 42
```

## Architecture

### File Structure

```
src/pytest_strategy/
├── __init__.py          # Package initialization
├── plugin.py            # Pytest plugin hooks and CLI options
├── strategy.py          # Strategy decorator and registry
├── parameters.py        # Parameter class (vector container)
├── test_args.py         # TestArg class (single argument definition)
├── rng.py               # Random number generation and RNG types
├── py.typed             # PEP 561 marker: type checkers use the package's annotations
└── _*.py                # Internal helpers (resolver, introspection, test IDs, dataclasses,
                         # runtime state, warning categories)
```

## Core Components

### 1. `rng.py` - Random Number Generation

The foundation of randomized testing. Provides:

#### RNG Class (Static Methods)
Core random generation with seed management:

```python
from pytest_strategy.rng import RNG

# Seed management
RNG.seed(42)              # Set the seed and reseed the global random state
RNG.get_seed()            # Get current seed
RNG.refresh_seed()        # Restart the global random state from the current seed

# Basic generators
RNG.integer(min=0, max=100)                    # Random integer
RNG.float(min=0.0, max=1.0)                    # Random float
RNG.boolean(true_probability=0.5)              # Random boolean
RNG.choice(['a', 'b', 'c'])                    # Random choice
RNG.string(length=10, charset="abc")           # Random string

# With constraints
RNG.integer(0, 100, predicate=lambda x: x % 2 == 0)  # Even numbers only

# Weighted generators
RNG.winteger({
    (0, 20): 0.8,      # 80% from 0-20
    (21, 100): 0.2     # 20% from 21-100
})

RNG.wfloat({
    (0.0, 1.0): 0.7,
    (1.0, 10.0): 0.3
})
```

#### RNG Type Classes
Object-oriented approach for defining argument types:

```python
from pytest_strategy.rng import (
    RNGInteger, RNGFloat, RNGBoolean, RNGChoice, 
    RNGString, RNGWeightedInteger, RNGWeightedFloat
)

# Basic types
int_type = RNGInteger(min=1, max=100)
float_type = RNGFloat(min=0.0, max=1.0)
bool_type = RNGBoolean(true_probability=0.8)
choice_type = RNGChoice(choices=['fast', 'slow', 'medium'])
string_type = RNGString(length=10)

# Weighted types
weighted_int = RNGWeightedInteger(
    ranges={(0, 20): 0.8, (21, 100): 0.2}
)

# Generate values
value = int_type.generate()
python_type = int_type.python_type  # Returns: int
```

**Key Features:**
- Seed-based reproducibility
- Constraint support via predicates
- Weighted range generation
- Configurable retry logic
- Type-safe generation
- Arguments checked at construction: the RNG type classes raise `RNGValueError`
  for `min > max`, empty, negative, non-finite or all-zero weights, a non-Enum
  class or an unsatisfiable `RNGEnum` predicate, an empty `RNGString` charset,
  and a `set`/`frozenset` passed to `Series` or `RNGSequence`

With a predicate, `RNGWeightedInteger`/`RNGWeightedFloat` choose a new range for
every retry, and `RNGEnum` draws only among the members the predicate accepts
(keeping their relative weights).

---

### 2. `test_args.py` - Test Argument Definition

Defines a single test argument with its type and generation rules.

```python
from pytest_strategy import TestArg
from pytest_strategy.rng import RNGInteger, RNGChoice

# Pure random generation
arg1 = TestArg(
    name="count",
    rng_type=RNGInteger(min=1, max=100),
    description="Random count between 1-100"
)

# Static value (directed test)
arg2 = TestArg(
    name="count",
    value=0,
    description="Edge case: zero"
)

# Mixed: directed + random
arg3 = TestArg(
    name="count",
    rng_type=RNGInteger(min=1, max=100),
    directed_values=[0, 1, 99, 100],  # Included by generate_samples()
    description="Count with edge cases"
)

# With validation
arg4 = TestArg(
    name="port",
    rng_type=RNGInteger(min=1024, max=65535),
    validator=lambda x: x > 0,
    description="Valid port number"
)

# Generate values
value = arg1.generate()                    # Single value
samples = arg3.generate_samples(10)        # 10 samples (+ directed if configured)
```

**Key Features:**
- Three modes: static, random, mixed
- Optional validation
- Directed values support
- Type introspection
- Integration with RNG types

**Properties:**
- `name` - Argument name
- `description` - Human-readable description
- `type` - Python type
- `is_static` - Whether it has a fixed value
- `has_directed_values` - Whether it has directed values

**Inside a `Parameter`:** a strategy uses each argument's `rng_type` (or static
`value`) and its `validator`. The argument-level `directed_values` and
`test_values` are only used by `TestArg.generate_samples()`. A `Parameter` does
not turn them into vectors, so define edge cases as the `Parameter`'s
`directed_vectors` and `test_vectors`. The validator runs on random draws,
static values and `Series`/`RNGSequence` values, but not on directed or test
vectors. A value that fails it stops collection with a `ValueError` and is not
redrawn, so use a predicate on the RNG type to filter values instead.

---

### 3. `parameters.py` - Parameter Vector Container

Groups multiple `TestArg` instances into parameter vectors (tuples).

```python
from pytest_strategy import Parameter, TestArg
from pytest_strategy.rng import RNGInteger, RNGFloat, RNGChoice

# Create parameter with multiple arguments
param = Parameter(
    TestArg("count", rng_type=RNGInteger(0, 100)),
    TestArg("timeout", rng_type=RNGFloat(0.1, 10.0)),
    TestArg("mode", rng_type=RNGChoice(["fast", "slow"])),
    directed_vectors={
        "edge_zero": (0, 0.1, "fast"),
        "edge_max": (100, 10.0, "slow"),
        "typical": (50, 5.0, "fast"),
    },
    always_include_directed=True
)

# Generate samples with different modes
samples = param.generate_vectors(10, mode="all")           # 3 directed + 10 random
samples = param.generate_vectors(10, mode="random_only")   # 10 random only
samples = param.generate_vectors(0, mode="directed_only")  # 3 directed only
samples = param.generate_vectors(10, mode="mixed")         # Respects always_include_directed

# CLI support
vector = param.get_vector_by_name("edge_zero")    # Get specific vector
vector = param.get_vector_by_index(0)             # Get by index
names = param.list_vector_names()                 # List all vector names

# Add vectors dynamically
param.add_directed_vector("custom", (25, 2.5, "slow"))

# Add constraints (cross-parameter validation)
param.add_constraint(lambda v: v[0] < v[1])  # Ensure first < second
```

**Sampling Modes:**

| Mode            | Directed Vectors | Random Samples | Use Case                        |
| --------------- | ---------------- | -------------- | ------------------------------- |
| `all`           | ✅ All            | ✅ n samples    | Comprehensive testing (default) |
| `random_only`   | ❌ None           | ✅ n samples    | Pure randomized testing         |
| `directed_only` | ✅ All            | ❌ None         | Only known test cases           |
| `mixed`         | ⚠️ Conditional*   | ✅ n samples    | Flexible (respects flag)        |
| `test`          | ❌ None           | ❌ None         | Only the `test_vectors`         |

*Respects `always_include_directed` initialization flag

`filter_by_name` / `filter_by_index` (the `--vector-name` / `--vector-index`
options) take precedence over the mode and return that one directed vector, or
raise `KeyError` / `IndexError` when it does not exist. `n` must be an int >= 0
in the modes that generate samples.

**Constraints:** a random vector that fails `vector_constraints` is redrawn up to
`max_retries` times (default 100) before `generate_vectors` raises. With `Series`
args, `n` rows are taken by cycling through the `Series` combinations. A
combination the constraints reject is skipped, after its random args have been
redrawn up to `max_retries` times, and each such skip with random args emits a
`PytestStrategiesWarning`. The call raises only when a whole cycle yields no
vector. `generate_exhaustive()` (used for `--nsamples=auto`) builds the
Cartesian product of the `Series`/`RNGSequence` args, drops combinations the
constraints reject after the same redraws, and raises when it drops all of them.

With `Parameter(per_sequence_samples=True)`, `n` counts per combination of the
`Series`/`RNGSequence` args: `generate_vectors(n)` walks their Cartesian product
in declaration order (for `RNGSequence` too) and draws `n` rows of fresh random
args for each one. A combination whose random args are rejected `max_retries`
times in a row stops short with a `PytestStrategiesWarning`; the call raises when
no combination yields a row. `generate_exhaustive()` ignores the flag, and a
`Parameter` without sequence args behaves as if it were `False`.

The `Parameter` copies the `directed_vectors`, `test_vectors` and
`vector_constraints` it is given, so `add_*`/`remove_*` never change the
caller's dicts and lists.

**Key Features:**
- Vector management (add, remove, get)
- Multiple sampling modes
- CLI integration (by name/index)
- Cross-parameter constraints
- Introspection (arg names, types, etc.)

**Properties:**
- `arg_names` - Tuple of argument names
- `arg_types` - Tuple of argument types
- `vector_names` - List of directed vector names
- `num_args` - Number of arguments
- `num_directed_vectors` - Number of directed vectors

---

### 4. `strategy.py` - Strategy Registry & Decorator

Manages strategy registration and applies parametrization to tests.

```python
from pytest_strategy import Strategy, Parameter, TestArg
from pytest_strategy.rng import RNGInteger

# Register a strategy
@Strategy.register("my_strategy")
def create_samples(nsamples):
    # Return the Parameter itself; the plugin generates the vectors from it,
    # which is what lets CLI options such as --vector-mode apply to it
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 10)),
        TestArg("y", rng_type=RNGInteger(0, 10)),
        directed_vectors={
            "origin": (0, 0),
            "max": (10, 10),
        }
    )

# Apply strategy to test
@Strategy.strategy("my_strategy")
def test_coordinates(x, y):
    assert x >= 0
    assert y >= 0
    assert x + y <= 20
```

**How It Works:**
1. `@Strategy.register()` stores factory functions in a global registry. If a
   different function registers a name that is already taken, it emits a
   `PytestStrategiesWarning` (from `pytest_strategy.strategy`), and the last
   registration wins.
2. `@Strategy.strategy()` runs when the test module is imported. It reseeds the
   random state for this strategy and test (see [Reproducibility](#reproducibility)),
   then calls the factory once. It passes `nsamples` by keyword, positionally, or
   not at all, depending on the factory's signature.
3. It generates the vectors from the returned `Parameter` according to the CLI
   options, and applies `pytest.mark.parametrize()` to them.
4. It creates readable test IDs from the values.

A test parameter that is not one of the strategy's argument names is left to
pytest as a fixture. A test can also take the vector as one dataclass instance:
see "Dataclass Parameters" in the README.

**Key Features:**
- Global strategy registry
- Automatic pytest parametrization
- A random stream of its own for each strategy and test
- CLI option integration
- Readable test IDs. A value whose repr contains a memory address is shown by
  its type name, and set elements are sorted, so IDs are the same on every run
  and on every xdist worker.

---

### 5. `plugin.py` - Pytest Plugin Integration

Provides pytest hooks and CLI options.

**CLI Options:**

```bash
--nsamples N|auto         # Number of random samples (default: 10), or auto
--rng-seed SEED           # Random seed for reproducibility
--vector-mode MODE        # Sampling mode: all, random_only, directed_only, mixed, test
--vector-name NAME        # Run specific directed vector by name
--vector-index INDEX      # Run specific directed vector by index
--list-strategies         # List the registered strategies and exit
```

`--nsamples` is checked when the command line is parsed: anything other than an
integer >= 0 or `auto` (in any case) is a usage error. Under `auto`, directed
vectors are still added per `--vector-mode`, and `--vector-mode=test`,
`--vector-mode=directed_only`, `--vector-name` and `--vector-index` select only
those vectors. A strategy without `Series`/`RNGSequence` args falls back to its
own `nsamples`, or 10. A strategy that lacks the vector requested by
`--vector-name` or `--vector-index` gets an empty parameter set (its tests are
skipped). If no strategy has it, the run stops with a usage error.

**Pytest Hooks:**
- `pytest_addoption` - Adds CLI options
- `pytest_configure` - Sets the config and the run's seed. An explicit
  `--rng-seed` also seeds the global random state; a pytest-xdist worker without
  one takes the controller's seed.
- `pytest_configure_node` - (pytest-xdist only) sends the controller's seed to
  each worker
- `pytest_sessionstart` - Discovers and imports strategy files
- `pytest_report_header` - Prints the seed and the number of strategies and
  strategy files
- `pytest_collection_modifyitems` - Fails the run when `--vector-name` or
  `--vector-index` matched no strategy
- `pytest_collection_finish` - Handles `--list-strategies`

**Strategy file discovery:** at session start the plugin imports every file
named `strategies.py`, `strategy.py`, `*_strategies.py` or `*_strategy.py` that
contains `@Strategy.register`. It searches the `testpaths` directories (glob
patterns expanded), or the rootdir without `testpaths`, plus the directory of
each path given on the command line. Below those directories it skips hidden
directories, `__pycache__`, `norecursedirs` matches and virtual environments
(directories containing `pyvenv.cfg`). Files are loaded in sorted path order,
each as a standalone module (no relative imports). A file that raises while
loading is reported with a `pytest-strategies: Warning - Failed to load ...`
line. A file that calls `pytest.skip()`/`pytest.importorskip()` at module level
is skipped (reported with `-v`). Both kinds are listed in any
"Strategy 'name' not found" error.

---

## Complete Example

```python
# test_math_operations.py

from pytest_strategy import Strategy, Parameter, TestArg
from pytest_strategy.rng import RNGInteger, RNGWeightedInteger

@Strategy.register("division_strategy")
def create_division_samples(nsamples):
    param = Parameter(
        TestArg(
            name="dividend",
            rng_type=RNGInteger(min=-1000, max=1000),
            description="Number to be divided"
        ),
        TestArg(
            name="divisor",
            rng_type=RNGWeightedInteger(
                ranges={
                    (1, 10): 0.7,      # Small divisors 70%
                    (11, 100): 0.3     # Larger divisors 30%
                }
            ),
            description="Number to divide by (never zero)"
        ),
        directed_vectors={
            "simple": (10, 2),
            "negative_dividend": (-10, 2),
            "large_divisor": (100, 50),
            "one_divisor": (42, 1),
        },
        always_include_directed=True
    )

    # Add constraint: divisor must not be zero
    param.add_constraint(lambda v: v[1] != 0)

    return param

@Strategy.strategy("division_strategy")
def test_division(dividend, divisor):
    result = dividend / divisor

    # Basic properties
    assert result * divisor == dividend or abs(result * divisor - dividend) < 0.0001

    # Sign rules
    if dividend > 0 and divisor > 0:
        assert result > 0
    elif dividend < 0 and divisor < 0:
        assert result > 0
    elif (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
        assert result < 0

@Strategy.register("string_concat_strategy")
def create_string_samples(nsamples):
    from pytest_strategy.rng import RNGString, RNGChoice

    return Parameter(
        TestArg("str1", rng_type=RNGString(min_length=0, max_length=20)),
        TestArg("str2", rng_type=RNGString(min_length=0, max_length=20)),
        TestArg("separator", rng_type=RNGChoice(choices=["", " ", "-", "_"])),
        directed_vectors={
            "empty_strings": ("", "", ""),
            "no_separator": ("hello", "world", ""),
            "with_space": ("hello", "world", " "),
        }
    )

@Strategy.strategy("string_concat_strategy")
def test_string_concatenation(str1, str2, separator):
    result = str1 + separator + str2

    assert len(result) == len(str1) + len(separator) + len(str2)
    assert result.startswith(str1)
    assert result.endswith(str2)
    if separator:
        assert separator in result
```

**Run the tests:**

```bash
# Default: directed vectors + 10 random
# (4 + 10 = 14 cases for test_division, 3 + 10 = 13 for test_string_concatenation)
pytest test_math_operations.py

# More random samples
pytest test_math_operations.py --nsamples 100

# Only directed tests
pytest test_math_operations.py --vector-mode directed_only

# Only random tests
pytest test_math_operations.py --nsamples 50 --vector-mode random_only

# Run specific vector
pytest test_math_operations.py --vector-name "simple"

# Reproducible run
pytest test_math_operations.py --rng-seed 42

# Verbose output
pytest test_math_operations.py -v
```

## Advanced Features

### Cross-Parameter Constraints

```python
param = Parameter(
    TestArg("min_val", rng_type=RNGInteger(0, 100)),
    TestArg("max_val", rng_type=RNGInteger(0, 100)),
)

# Ensure min < max
param.add_constraint(lambda v: v[0] < v[1])

# Multiple constraints
param.add_constraint(lambda v: v[1] - v[0] >= 10)  # At least 10 apart
```

### Weighted Distributions

```python
# Test edge cases more frequently
port_arg = TestArg(
    name="port",
    rng_type=RNGWeightedInteger(
        ranges={
            (1, 1023): 0.1,        # System ports (10%)
            (1024, 49151): 0.8,    # User ports (80%)
            (49152, 65535): 0.1    # Dynamic ports (10%)
        }
    )
)
```

### Validation

```python
# Argument-level validation
arg = TestArg(
    name="percentage",
    rng_type=RNGFloat(0.0, 100.0),
    validator=lambda x: 0 <= x <= 100
)

# Vector-level validation
param.add_constraint(lambda v: v[0] + v[1] <= 100)
```

### Reproducibility

Parametrized values are generated when pytest collects the tests, so the seed
has to be set before collection. Use the CLI option:

```bash
pytest --rng-seed 42
```

The seed of every run is shown in the pytest report header
(`pytest-strategies: RNG seed = ...`), which pytest hides under `-q` or
`--no-header`. A run without `--rng-seed` picks a seed from the clock, and
passing that printed seed reproduces the run.

**Per-test streams:** before calling a strategy's factory for a test, the plugin
reseeds the global random state from the run seed and a key made of the strategy
name, the test file's path relative to the rootdir and the test's qualified name
(`RNG.refresh_seed(key=...)`). Each strategy and test pair therefore gets its own
stream. The vectors of a test do not depend on which other tests are collected,
on the collection order or on `--import-mode`. Two tests that share a strategy
get different vectors. For the same seed, the values differ from those of
1.1.0a2 and earlier.

**Global random state:** with `--rng-seed`, the global `random` state is seeded
when pytest is configured. Without it, the plugin does not touch the state at
startup. Right before it loads strategy files, it starts the state from the
seed, so values drawn while the files are imported are reproducible too. After
that, the state is reseeded for each strategy and test as described above. When
a session ends (including an in-process `pytester` run), the seed, the global
random state and the strategy registry are restored to what they were when it
began.

**pytest-xdist:** the controller sends its seed to the workers, so `-n` works
with or without `--rng-seed` and every worker generates the same tests.

**Test bodies:** the seed reproduces the parameters, not random values drawn
inside a test body. Those come from the global random state as the earlier
collection and tests left it, so they change when a single test is rerun or
tests are scheduled differently under xdist. Calling `RNG.seed()` inside a test
body does not change the test's parameters, which are already fixed by then. To
make a body's own draws reproducible, reseed in the body. A stream keyed by the
node ID still follows `--rng-seed`:

```python
from pytest_strategy.rng import RNG

def test_something(request):
    RNG.refresh_seed(key=request.node.nodeid)
    value = RNG.integer(0, 100)  # Same value for the same --rng-seed

def test_fixed():
    RNG.seed(42)  # Also changes the run's seed (RNG.get_seed()) from here on
    value = RNG.integer(0, 100)  # Same value on every run
```

## Best Practices

### 1. Always Include Edge Cases

```python
directed_vectors={
    "zero": (0,),
    "negative": (-1,),
    "max": (sys.maxsize,),
    "min": (-sys.maxsize,),
}
```

### 2. Use Constraints for Valid Inputs

```python
# Instead of hoping random generation produces valid inputs
param.add_constraint(lambda v: v[0] < v[1])  # min < max
```

### 3. Use Weighted Distributions for Important Cases

```python
# Test edge cases more frequently
RNGWeightedInteger({
    (0, 10): 0.5,      # Small numbers 50%
    (11, 100): 0.3,    # Medium numbers 30%
    (101, 1000): 0.2   # Large numbers 20%
})
```

### 4. Name Directed Vectors Descriptively

```python
directed_vectors={
    "edge_zero": (0, 0),
    "edge_max": (100, 100),
    "typical_case": (50, 50),
    "bug_12345": (42, 17),  # Regression test
}
```

### 5. Use Validation for Complex Constraints

```python
def is_valid_config(config_tuple):
    timeout, retries, mode = config_tuple
    if mode == "fast":
        return timeout < 1.0 and retries <= 3
    return True

param.add_constraint(is_valid_config)
```

## Troubleshooting

### "No valid value found after N attempts"

Your constraints are too restrictive. Either:
- Relax the constraints
- Increase retry limit: `RNG.set_max_retries(1000)`
- Use directed values instead

### "Could not generate valid vector ..."

The `vector_constraints` rejected every draw (or, with `Series`/`RNGSequence`
args, every combination). Relax them, or raise `Parameter(max_retries=...)`.
The related `PytestStrategiesWarning` "Series combination (...) skipped" means
one combination was skipped after `max_retries` redraws of its random args.

### "Strategy not found"

Make sure you:
1. Registered the strategy with `@Strategy.register("name")`
2. Used the exact same name in `@Strategy.strategy("name")`
3. Registered it in the test module itself, in an imported module, or in a
   strategy file that discovery finds (see "Strategy file discovery" above;
   `pytest -vv` lists the loaded files)

The error message lists strategy files that failed to load or skipped
themselves; the load failures are also printed when the session starts.

### Tests not reproducible

- Pass the same `--rng-seed` value (a run's seed is shown in the report header); calling `RNG.seed()` inside a test body does not change its parametrized values
- Use the same pytest-strategies version and the same rootdir: both change the generated values
- Random values drawn inside a test body are not covered by the seed; reseed in the body (see [Reproducibility](#reproducibility))

## Future Enhancements

- [ ] Vector groups (categorize directed vectors)
- [ ] Combinatorial mode (all combinations of directed values)
- [ ] Replay support (save/load generated vectors)
- [ ] Statistics tracking (which vectors found bugs)
- [ ] Partial vector support (None = generate random)
- [ ] Vector inheritance/templates
- [ ] Integration with hypothesis
- [ ] Custom RNG types (user-defined)

## Contributing

Contributions welcome! Areas of interest:
- Additional RNG types
- Better CLI integration
- Performance optimizations
- Documentation improvements

## License

[Your License Here]

---

**pytest_strategy** - Making randomized testing easy and powerful! 🎲✨