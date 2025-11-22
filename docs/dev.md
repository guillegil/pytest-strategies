# pytest_strategies

A pytest plugin for constrained-randomized test parametrization with directed testing support.

## Overview

`pytest_strategies` enables you to write powerful parametrized tests that combine:
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
from pytest_strategies import Strategy, Parameter, TestArg
from pytest_strategies.rng import RNGInteger, RNGFloat, RNGChoice

# Define a strategy
@Strategy.register("test_addition_strategy")
def create_addition_samples(nsamples):
    # Create parameter with test arguments
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        directed_vectors={
            "zeros": (0, 0),
            "max": (100, 100),
        }
    )

    # Generate samples
    samples = param.generate_samples(nsamples, mode="all")

    # Return (argnames, samples) tuple
    return param.arg_names, samples

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

# Run only directed vectors
pytest test_example.py --vector-mode directed_only

# Run specific vector by name
pytest test_example.py --vector-name "zeros"

# Run specific vector by index
pytest test_example.py --vector-index 0

# Set seed for reproducibility
pytest test_example.py --seed 42
```

## Architecture

### File Structure

```
pytest_strategies/
├── __init__.py          # Package initialization
├── plugin.py            # Pytest plugin hooks and CLI options
├── strategies.py        # Strategy decorator and registry
├── parameter.py         # Parameter class (vector container)
├── test_args.py         # TestArg class (single argument definition)
├── rng.py              # Random number generation and RNG types
└── pyproject.toml      # Project configuration
```

## Core Components

### 1. `rng.py` - Random Number Generation

The foundation of randomized testing. Provides:

#### RNG Class (Static Methods)
Core random generation with seed management:

```python
from pytest_strategies.rng import RNG

# Seed management
RNG.seed(42)              # Set seed for reproducibility
RNG.get_seed()            # Get current seed
RNG.refresh_seed()        # Refresh random state

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
from pytest_strategies.rng import (
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

---

### 2. `test_args.py` - Test Argument Definition

Defines a single test argument with its type and generation rules.

```python
from pytest_strategies import TestArg
from pytest_strategies.rng import RNGInteger, RNGChoice

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
    directed_values=[0, 1, 99, 100],  # Always test these
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

---

### 3. `parameter.py` - Parameter Vector Container

Groups multiple `TestArg` instances into parameter vectors (tuples).

```python
from pytest_strategies import Parameter, TestArg
from pytest_strategies.rng import RNGInteger, RNGFloat, RNGChoice

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
samples = param.generate_samples(10, mode="all")           # 3 directed + 10 random
samples = param.generate_samples(10, mode="random_only")   # 10 random only
samples = param.generate_samples(0, mode="directed_only")  # 3 directed only
samples = param.generate_samples(10, mode="mixed")         # Respects always_include_directed

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

| Mode | Directed Vectors | Random Samples | Use Case |
|------|-----------------|----------------|----------|
| `all` | ✅ All | ✅ n samples | Comprehensive testing (default) |
| `random_only` | ❌ None | ✅ n samples | Pure randomized testing |
| `directed_only` | ✅ All | ❌ None | Only known test cases |
| `mixed` | ⚠️ Conditional* | ✅ n samples | Flexible (respects flag) |

*Respects `always_include_directed` initialization flag

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

### 4. `strategies.py` - Strategy Registry & Decorator

Manages strategy registration and applies parametrization to tests.

```python
from pytest_strategies import Strategy, Parameter, TestArg
from pytest_strategies.rng import RNGInteger

# Register a strategy
@Strategy.register("my_strategy")
def create_samples(nsamples):
    param = Parameter(
        TestArg("x", rng_type=RNGInteger(0, 10)),
        TestArg("y", rng_type=RNGInteger(0, 10)),
        directed_vectors={
            "origin": (0, 0),
            "max": (10, 10),
        }
    )

    samples = param.generate_samples(nsamples, mode="all")

    # Must return (argnames, samples) tuple
    return param.arg_names, samples

# Apply strategy to test
@Strategy.strategy("my_strategy")
def test_coordinates(x, y):
    assert x >= 0
    assert y >= 0
    assert x + y <= 20
```

**How It Works:**
1. `@Strategy.register()` stores factory functions in a global registry
2. `@Strategy.strategy()` retrieves the factory and generates samples
3. Applies `pytest.mark.parametrize()` with generated samples
4. Creates test IDs for better output

**Key Features:**
- Global strategy registry
- Automatic pytest parametrization
- Seed refresh before generation
- CLI option integration
- Readable test IDs

---

### 5. `plugin.py` - Pytest Plugin Integration

Provides pytest hooks and CLI options.

**CLI Options:**

```bash
--nsamples N              # Number of random samples (default: 10)
--seed SEED               # Random seed for reproducibility
--vector-mode MODE        # Sampling mode: all, random_only, directed_only, mixed
--vector-name NAME        # Run specific directed vector by name
--vector-index INDEX      # Run specific directed vector by index
```

**Pytest Hooks:**
- `pytest_addoption` - Adds CLI options
- `pytest_configure` - Initializes plugin and sets config
- `pytest_collection_modifyitems` - Can modify test collection (future use)

---

## Complete Example

```python
# test_math_operations.py

from pytest_strategies import Strategy, Parameter, TestArg
from pytest_strategies.rng import RNGInteger, RNGWeightedInteger

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

    samples = param.generate_samples(nsamples, mode="all")
    return param.arg_names, samples

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
    from pytest_strategies.rng import RNGString, RNGChoice

    param = Parameter(
        TestArg("str1", rng_type=RNGString(min_length=0, max_length=20)),
        TestArg("str2", rng_type=RNGString(min_length=0, max_length=20)),
        TestArg("separator", rng_type=RNGChoice(choices=["", " ", "-", "_"])),
        directed_vectors={
            "empty_strings": ("", "", ""),
            "no_separator": ("hello", "world", ""),
            "with_space": ("hello", "world", " "),
        }
    )

    samples = param.generate_samples(nsamples, mode="all")
    return param.arg_names, samples

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
# Default: 4 directed + 10 random = 14 test cases per test
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
pytest test_math_operations.py --seed 42

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

```python
# Set seed in test or via CLI
from pytest_strategies.rng import RNG

def test_something():
    RNG.seed(42)
    # Test will always generate same random values
```

Or via CLI:
```bash
pytest --seed 42
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

### "Strategy not found"

Make sure you:
1. Registered the strategy with `@Strategy.register("name")`
2. Used the exact same name in `@Strategy.strategy("name")`
3. Imported the module containing the registration

### Tests not reproducible

- Use `--seed` CLI option or `RNG.seed()` in code
- Ensure no other randomness sources (use RNG class only)

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

**pytest_strategies** - Making randomized testing easy and powerful! 🎲✨