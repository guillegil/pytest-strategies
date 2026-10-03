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
from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

# Define a strategy
@register("test_addition_strategy")
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

# Use the strategy in a test, by name or by passing the factory:
# @strategy(create_addition_samples)
@strategy("test_addition_strategy")
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
├── __init__.py          # Package initialization and public names
├── __main__.py          # python -m pytest_strategy (the pytest-strategies command)
├── plugin.py            # Pytest plugin hooks, CLI options, strategy file loading
├── _api.py              # register(), strategy(), export_strategies() and the Strategy facade
├── _registry.py         # The folder-scoped strategy registry
├── _resolver.py         # Turns a strategy and a test into a parametrization
├── _cli.py              # pytest-strategies skill install
├── strategy.py          # 2.x compatibility module (Strategy, PytestStrategiesWarning)
├── parameters.py        # Parameter class (vector container)
├── _vector.py           # Vector, the class of the generated rows (a namedtuple per argument names),
│                        # and VectorInfo, each row's metadata in item.stash[VECTOR_KEY]
├── test_args.py         # TestArg class (single argument definition)
├── rng.py               # Random number generation and RNG types
├── hookspecs.py         # Hooks the plugin adds (pytest_strategies_context)
├── skill/               # The agent skill that pytest-strategies skill install copies
├── py.typed             # PEP 561 marker: type checkers use the package's annotations
└── _*.py                # Other internal helpers (introspection, test IDs, record mode,
                         # StrategyOptions, factory calls, runtime state, warning categories,
                         # the value encoding of schema 1 documents, the random stream keys)
```

## Core Components

### 1. `rng.py` - Random Number Generation

The foundation of randomized testing. Provides:

#### RNG Class (Static Methods)
Core random generation with seed management:

```python
from pytest_strategy.rng import RNG

# Seed management
RNG.seed(42)              # Set the seed and restart the generator from it
RNG.get_seed()            # Get current seed
RNG.refresh_seed()        # Restart the generator from the current seed
RNG.refresh_seed(key="x") # Restart it on the stream of the seed and a key
RNG.generator()           # The random.Random instance all draws come from

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
- Seed-based reproducibility, from a generator of the plugin's own: Python's
  global `random` state is never seeded or used
- Constraint support via predicates (each draw gets 100 attempts)
- Weighted range generation
- Type-safe generation
- Arguments checked at construction: the RNG type classes raise `RNGValueError`
  for `min > max` (also in an `RNGWeighted*` range), infinite or NaN float
  bounds, empty, negative, non-finite or all-zero weights (or a total that
  overflows), a non-Enum class or an unsatisfiable `RNGEnum` predicate, an
  empty `RNGString` charset, and a `set`/`frozenset` passed to `Series` or
  `RNGSequence`
- `Series`/`RNGSequence` raise on an empty sequence (also after the predicate)
  unless created with `skip_if_empty="<reason>"`. With the reason, the empty
  sequence is kept, its `skip_reason` is the reason, and `generate()` raises

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

# With validation
arg3 = TestArg(
    name="port",
    rng_type=RNGInteger(min=1024, max=65535),
    validator=lambda x: x > 0,
    description="Valid port number"
)

# Generate values
value = arg1.generate()                    # Single value
samples = arg1.generate_samples(10)        # 10 draws ([value] for a static argument)
```

**Key Features:**
- Two modes: static and random
- Optional validation
- Type introspection
- Integration with RNG types

`value`, `validator` and `description` are keyword-only. `rng_type` must be an
`RNGType` or have a callable `generate()`; anything else, such as a bare lambda or
a class instead of an instance (`RNGBoolean` for `RNGBoolean()`), raises
`TypeError` when the `TestArg` is built. An object without `python_type` gives
the type `Any`.

**Properties:**
- `name` - Argument name
- `description` - Human-readable description
- `type` - Python type
- `is_static` - Whether it has a fixed value

**Inside a `Parameter`:** a strategy uses each argument's `rng_type` (or static
`value`) and its `validator`. Define edge cases as the `Parameter`'s
`directed_vectors` and `test_vectors`. (The argument-level `directed_values`,
`test_values` and `always_include_directed`, deprecated in 3.0, are removed in 4.0:
a `Parameter` never turned them into vectors.) The validator runs on random draws,
static values and `Series`/`RNGSequence` values, but not on directed or test
vectors. A value that fails it stops collection with a `ValueError` and is not
redrawn, so use a predicate on the RNG type to filter values instead.

---

### 3. `parameters.py` - Parameter Vector Container

Groups multiple `TestArg` instances into parameter vectors (tuples). Every row
it generates is a `Vector` (`_vector.py`): `vector_type(names)` builds one
namedtuple class per tuple of argument names, which also subclasses `Vector`, and
caches it, so `Parameter.vector_type` is shared by Parameters with the same names
and a prefix of the names gets a class of its own. One private `_build_row` fills
a row in declaration order (the enumerated values are already in place, the other
arguments are drawn), builds the Vector and hands it to the constraints, for the
plain, Series, `per_sequence_samples` and exhaustive paths alike. Each drawn
argument draws from a generator of its own, reseeded for each row from the row's
key (`_RowStreams`, see [Reproducibility](#reproducibility)); the generator is
installed as `RNG._generator` only while the argument's `generate()` and
validator run, and a retry continues it. `_generate_row(key, pos, j)` builds one
row on its own, with the values of the full list. Argument names must therefore
be identifiers that are not keywords and do not start with `_`.

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

**Rows:** every generation path goes through one entry point,
`Parameter._generate_rows(n, *, exhaustive=False, mode=..., filter_by_name=...,
filter_by_index=..., constraints_off=..., stats=None)`, which returns the rows as
`_Row`s in their 3.0 order: `kind` (`directed`, `test`, `random`, `exhaustive`,
or `skipped` for the one row of an empty `skip_if_empty` sequence), `name` (the
vector's name, else None), `index` (the number `VectorInfo.index` and the
messages show: the vector's position, `j` for a random row, the position in the
declaration-order product of the enumerated sequences for an exhaustive row,
None for `skipped`), `pos` (the enumerated arguments as `(name, token)` pairs,
sorted by name), `j` (the row's number within its combination: k for a plain
random row, the cycle for a finite Series row, so a combination the constraints
skip leaves a gap, the row's number with `per_sequence_samples`, 0 for an
exhaustive row), `values` (a `Vector`), `labels` (the enumerated arguments'
labels, in declaration order) and `param` (the `pytest.param` a vector was given
as). `generate_vectors()` and `generate_exhaustive()` return the rows' values.
The resolver calls `_generate_rows()` itself, with `exhaustive=True` under
`--nsamples=auto`, and builds each row's test ID (`names_id()`) and `VectorInfo`
from the same `_Row`, so the row index is defined once; a subclass that
overrides `generate_vectors()` does not change a test's rows.
`_position_keys(arg, sequence)` gives each position of an enumerated argument's
sequence (after its predicate) a `_Key(label, token)`: the label the ID shows
(`ch=2`, `ch=1~1` for a repeat, `ch3` by position) and a typed token (`n`,
`b:True`, `e:<qualname>.<member>`, `i:42`, `f:<float.hex()>`, `s:fast`,
`y:<hex>`, `#<position>`). `_value_keys()` decides a value's type in the order
None, bool, Enum member, `numbers.Integral`, `numbers.Real`, str, bytes,
anything else, and keys a value it cannot convert (a Flag value that is no
member, an int with too many digits for `str()`) by its position. `_auto_order()`
gives a sequence's `--nsamples=auto` order as positions
(`SequenceLike._auto_positions()`), and matches the values of a subclass that
overrides `_get_auto_sequence()` back to their positions (`RNGValueError` for a
value its sequence does not have).

**Constraints:** `vector_constraints` is a read-only mapping of names to
functions (the private `_constraints` dict), in evaluation order; a list given
to the constructor is named by each function's `__name__`, or `constraint_<i>`
for lambdas, partials and callable objects (`_unnamed` keeps those names, so the
messages add the constraint's origin). Duplicate names, a function given twice
and names with whitespace, `:`, `,` or `=` fail when the `Parameter` is built. A
random vector that fails a constraint is redrawn up to `max_retries` times
(default 100) before `generate_vectors` raises `_ConstraintsExhausted`, whose
message counts the draws by the name of the first failing constraint
(`_Rejections`) and shows the first row each one rejected; the resolver replaces
its last sentence with advice that names the strategy and its strictest
constraint. The exception of a constraint that raises goes up unchanged, so a
caller that catches its type keeps working, with a note naming the constraint
and the row and a `_ConstraintFailure` in its `_CONSTRAINT_FAILURE` attribute
(`_ConstraintFailure.attach()`, which replaces the note of an earlier failure on
the same exception object, so a re-raised instance keeps one note); the resolver
attaches it again so the note names `--strategy-constraint-off`, builds the
collection error from it and chains it to the user's exception. `attach()`
writes the note and the attribute with `object.__setattr__`, so a frozen
dataclass or attrs exception is not replaced by a `FrozenInstanceError` (an
exception that rejects even that gets no note, and its failure waits in
`_unattached` until the resolver reports it); for the same reason the resolver's
`_attributed_warnings` is a class, since a `contextlib.contextmanager` assigns
the exception's `__traceback__` on the way out. The resolver collects the
counts (and the combinations `--nsamples=auto` left out) for the `-v` summary in
a `_GenerationStats` it passes as the `stats` keyword of `_generate_rows()`; the
private `_stats` keyword of `generate_vectors()` and `generate_exhaustive()`
passes one on.
Their keyword-only `constraints_off` names constraints the call does not
evaluate (`_evaluated()` builds the list once per call); the `Parameter` keeps
them, so a cached factory's `Parameter` is never changed. The resolver passes
the names of `--strategy-constraint-off` that the `Parameter` has
(`StrategyOptions.constraints_off` intersected with its constraint names) and
records every resolved strategy's constraint names on the session
(`SessionState.constraint_names`), whether or not the run evaluates them, so the
plugin can check the items once collection ends. A raising constraint's error
names the constraints before it that were turned off (`_ConstraintFailure.off_before`).
With `Series`
args, `n` rows are taken by cycling through the `Series` combinations. A
combination the constraints reject is skipped, after its random args have been
redrawn up to `max_retries` times, and each such skip with random args emits a
`PytestStrategiesWarning`. The call raises only when a whole cycle yields no
vector. `generate_exhaustive()` (the rows of `--nsamples=auto`) builds the
Cartesian product of the `Series`/`RNGSequence` args, drops combinations the
constraints reject after the same redraws, and raises when it drops all of them.

**Size guard:** before generating `--nsamples=auto` combinations, or
`per_sequence_samples=True` rows, the resolver multiplies the sequence lengths
(times the rows per combination) and fails collection when the product is above
the limit: `Parameter(max_exhaustive=...)`, else the `strategies_max_exhaustive`
ini option, else 100,000.

With `Parameter(per_sequence_samples=True)`, `n` counts per combination of the
`Series`/`RNGSequence` args: `generate_vectors(n)` walks their Cartesian product
in declaration order (for `RNGSequence` too) and draws `n` rows of fresh random
args for each one. A combination whose random args are rejected `max_retries`
times in a row stops short with a `PytestStrategiesWarning`; the call raises when
no combination yields a row. `generate_exhaustive()` ignores the flag, and a
`Parameter` without sequence args behaves as if it were `False`, and `n=0`
returns `[]` without walking the combinations.

The resolver re-emits the `PytestStrategiesWarning`s raised while generating
vectors at the test function, prefixed with `Strategy '<name>' (<test>): `, so
the warnings summary says which strategy and test they concern.

`Parameter.skip_reason` is the reason of the first `Series`/`RNGSequence` arg
created with `skip_if_empty` that has no values, or `None`. When it is set,
`generate_vectors` and `generate_exhaustive` return `[]` in every mode, without
drawing random values (an invalid `n` still raises). `filter_by_name`/`filter_by_index` still raise
`KeyError`/`IndexError` for a missing vector, so the CLI can tell whether a
filter matched. The resolver then parametrizes the test with a single
`pytest.param(None, ..., marks=pytest.mark.skip(reason=...))` row, whose ID is `skipped`,
also in dataclass mode, where no instance is built but the dataclass fields are
still checked against the strategy's arguments. `skip_if_empty` is keyword-only,
and a non-callable `predicate` (such as a reason passed positionally) raises.

The `Parameter` copies the `directed_vectors`, `test_vectors` and
`vector_constraints` it is given, so `add_*`/`remove_*` never change the
caller's dicts and lists. Every directed and test vector goes through
`_normalize_vector()` (in `__init__` and the `add_*` methods), which stores it
as a `Vector` in declaration order: a dict or a namedtuple by name, a tuple, list
or other iterable by position, and a `pytest.param` rebuilt around the `Vector`
with its marks. Strings, bytes, scalars, record instances and a `pytest.param`
with an `id=` (the vector's name is its ID) fail there, and so do names that are
not non-empty strings. `directed_vectors` and
`test_vectors` are read-only `MappingProxyType` views of the private dicts.

**Key Features:**
- Vector management (add, remove, get)
- Multiple sampling modes
- CLI integration (by name/index)
- Cross-parameter constraints
- Introspection (arg names, types, etc.)

Two args with the same name raise `RNGValueError`.

**Properties:**
- `arg_names` - Tuple of argument names
- `arg_types` - Tuple of argument types
- `vector_names` - List of directed vector names
- `num_args` - Number of arguments
- `num_directed_vectors` - Number of directed vectors

---

### 4. `_api.py` and `_registry.py` - Registration & Decorator

`register()` stores factories, and `strategy()` marks the tests that use them.
Both are exported by the package, and also available as `Strategy.register` and
`Strategy.strategy`.

```python
from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

# Register a strategy
@register("my_strategy")
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
@strategy("my_strategy")
def test_coordinates(x, y):
    assert x >= 0
    assert y >= 0
    assert x + y <= 20
```

**How It Works:**
1. `@register(name)` adds the factory to the registry under the name and the
   folder of the file that defines it. Each folder has at most one registration
   per name. Registering a name again in the same folder replaces the factory,
   and when the function is a different one it emits a `PytestStrategiesWarning`
   and records a clash that fails the run with a usage error after collection
   (on a pytest-xdist worker, the worker runs nothing and the controller reports
   it). A function is identified by its file, qualified name and first line,
   looking through `functools.wraps` decorators, so re-running the same file is
   silent.
2. `@strategy(name_or_factory)` only adds a `strategy` marker to the test; it
   runs nothing when the module is imported.
3. In `pytest_generate_tests`, the plugin resolves each `strategy` marker. A
   name is looked up from the test's folder upward, nearest first
   (`registry.nearest`). When no folder on that path registers it, the plugin
   loads every strategy file and uses the only registration elsewhere,
   preferring one outside the rootdir; several candidates are an error. A
   factory passed directly is used as it is.
4. `build_parametrization()` calls the factory once through
   `_factory.call_factory()`, on the factory's own stream (see
   [Reproducibility](#reproducibility)). The rows draw from streams keyed by
   the run seed, the strategy name and the test's node ID. The factory gets, by
   name, the inputs it declares: `nsamples`, `ctx`, `rng` and `options` (see
   "Factory inputs" below).
5. It generates the vectors from the returned `Parameter` according to the CLI
   options, with readable test IDs. The plugin inserts a
   `pytest.mark.parametrize` marker right after the `strategy` marker, so
   pytest applies it together with the test's own `parametrize` markers, in
   decorator order.

Errors while resolving (an unknown name, a signature mismatch, a factory that
raises `ValueError`) fail the test's collection like pytest's own parametrize
errors: `In test_x: <message>`, with the factory's own frames and without the
plugin's. `--full-trace` shows the full traceback.

**Factory inputs:** `_factory.analyse()` reads the factory's signature into a
`CallPlan` without calling anything, and `call_factory()` follows it. The
signature of the callable that is called decides; when it has none or only
`*args`/`**kwargs`, the `__wrapped__` chain is followed one level at a time
(also behind a partial, a bound method, a callable object's `__call__` or a
class's `__init__`) and the first signature that names its parameters decides,
so with stacked decorators it is the outermost wrapper's that does. Without
one the factory is called with no arguments. Each parameter named `nsamples`,
`ctx`, `rng` or `options` gets that input (positional-only ones by position,
the others by keyword, or all by position when a `functools.wraps` wrapper
passed through has `*args` but no `**kwargs`); any other parameter keeps its
default, `*args` and `**kwargs` receive nothing, and the mocks of a
`mock.patch` passed through take the first parameters.
A parameter without a default that is not an input, a reserved name (`base`,
`config`, `request`) and an `async def` factory fail with a `ValueError` before
the factory or the context hook runs. `ctx` is computed only for a factory that
declares it, and is left out when it is None and the parameter has a default.
The rule for 4.x: a new factory input arrives as a new `StrategyOptions` field
with a default, or under a reserved name. The plugin never starts passing a
value to a name that 4.0 accepted.

A test parameter that is not one of the strategy's argument names is left to
pytest as a fixture. A test can also take the vector as one dataclass instance:
see "Dataclass Parameters" in the README. `_records.py` holds that rule:
`detect_record_param(test_fn, argnames, fixturenames)` reads
`metafunc.fixturenames`, so an argument that the test or any of its fixtures asks
for keeps the strategy in named mode, and the named-mode signature check counts
those names as taken. It recognizes dataclasses, NamedTuple, TypedDict and
pydantic v2 models (through `sys.modules`, never importing pydantic) with fixed
field sets, and builds only dataclasses. The resolver asserts that no argument
name is in `fixturenames` in record mode: only the record parameter is
parametrized, so a fixture asking for an argument would find nothing.

**Key Features:**
- Folder-scoped strategy names
- Automatic pytest parametrization
- Random streams of their own for each strategy, test, row and argument
- CLI option integration
- Test IDs that name the row (`directed-zeros`, `rand-3`, `ch=2-rand-1`, built by
  `names_id()` from the row's kind, name, j and labels), the same for every seed.
  The ini option `strategies_ids=values` gives the 3.0 IDs built from the values
  instead: there a value whose repr contains a memory address is shown by
  its type name, and set elements are sorted (also inside tuples, lists,
  dicts, and dataclass and namedtuple values that keep their generated repr),
  so IDs are the same on every run and on every xdist worker.
  `Parameter(ids="names" | "values")` overrides the ini option for one strategy,
  and a callable `ids=` (`_custom_ids()`) receives each row's `VectorInfo` with
  the ID of the effective format and returns a str or None (keep it); the
  skipped row gets no call. `_unique_ids()` then suffixes duplicates, and the
  `VectorInfo` of each row carries the final ID.

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
--strategy-constraint-off [STRATEGY:]NAME[,...]  # Turn constraints off (repeatable)
--list-strategies         # List the registered strategies and exit
```

**Ini options:**

```ini
strategies_max_exhaustive = 100000  # Most rows --nsamples=auto (or per_sequence_samples) may generate per strategy
strategies_ids = names              # Test IDs of strategy rows: names, or values (the 3.0 format)
```

`--nsamples` is checked when the command line is parsed: anything other than an
integer >= 0 or `auto` (in any case) is a usage error. Under `auto`, directed
vectors are still added per `--vector-mode`, and `--vector-mode=test`,
`--vector-mode=directed_only`, `--vector-name` and `--vector-index` select only
those vectors. A strategy without `Series`/`RNGSequence` args falls back to its
own `nsamples`, or 10. A strategy that lacks the vector requested by
`--vector-name` or `--vector-index` gets an empty parameter set (its tests are
skipped). If no strategy has it, the run stops with a usage error.

`--strategy-constraint-off` items are checked when the command line is parsed
(`_options.parse_constraint_off`: whitespace, empty items, `:x` and `x:` are
usage errors) and split at their last `:` into `(strategy, name)` pairs on
`SessionOptions`. Once collection ends, an item that matches no constraint of a
strategy the run resolved is a usage error when the run collected the whole
suite, and otherwise a red line after the collection report (on a pytest-xdist
worker, sent with the `-v` summary and printed by the controller). A run counts
as the whole suite (`_collects_whole_suite`) when it was given no paths or node
IDs (`config.args_source` is not `ARGS`), started in the rootdir (pytest's own
rule in `Config._decide_args`: a run without arguments from a folder below it
collects only that folder), and got none of `--lf`, `--sw`, `--ignore` or
`--ignore-glob`, and when no test module or class was skipped or failed to
collect (`pytest_collectreport` sets `SessionState.collectors_incomplete`: the
strategies of its tests may not have been resolved). It is not checked under
`--list-strategies` or when no `Parameter` strategy was resolved.

**Pytest Hooks:**
- `pytest_addhooks` - Adds the `pytest_strategies_context` hook (see below)
- `pytest_addoption` - Adds CLI options
- `pytest_configure` - Checks the `strategies_ids` ini option (any value but
  `names` or `values` is a `UsageError`, exit code 4), opens the session's state
  and sets the run's seed, restarting the plugin's generator from it; a
  pytest-xdist worker without `--rng-seed` takes the controller's seed
- `pytest_configure_node` - (pytest-xdist only) sends the controller's seed to
  each worker
- `pytest_plugin_registered` - Records a `conftest.py` that pytest imported by
  its path (a module registered under its path) in
  `SessionState.imported_files`, which keys its fixtures and factories by its
  path (see [Reproducibility](#reproducibility)). The hook is historic, so the
  conftest files pytest loaded before the plugin registered count too
- `pytest_collectstart` - Before a test module is imported, loads the strategy
  files of its folder and the folders above it
- `pytest_generate_tests` - Resolves the test's `strategy` markers into
  `parametrize` markers. A second, module-level implementation runs last: it
  fails a test written for record mode whose fixtures take every argument by
  name when no fixture or parametrization gives its record parameter a value
- `pytest_itemcollected` - (tryfirst) Stores the `VectorInfo` of each strategy
  row an item runs, read from the row's `strategy` mark: the first in node-ID
  order under `VECTOR_KEY`, all of them under `VECTORS_KEY`, before any
  `pytest_collection_modifyitems` hook
- `pytest_collectreport` - Notes a test module or class that was skipped or
  failed to collect: its strategies may be unresolved, so the run counts as
  narrowed for `--strategy-constraint-off`
- `pytest_collection_modifyitems` - Fails the run on a name registered twice in
  one folder, when `--vector-name` or `--vector-index` matched no strategy, and
  when a `--strategy-constraint-off` item matched no constraint in a run of the
  whole suite
- `pytest_collection_finish` - Handles `--list-strategies`, and prints the
  unmatched `--strategy-constraint-off` items of a narrowed run
- `pytest_report_header` - Prints the seed, and the `--strategy-constraint-off`
  items when given
- `pytest_terminal_summary` - After a failed run, prints
  `pytest-strategies: reproduce with --rng-seed=S`; with `-v`, a Strategy
  Summary (tests and the rows of each kind per strategy, and where `nsamples`
  came from)

**Strategy files:** a strategy file is named `strategies.py`, `strategy.py`,
`*_strategies.py` or `*_strategy.py` and contains a registration decorator
(`@register("...")`, `@Strategy.register(`, or an alias such as
`@S.register("...")`, with the name as a string literal in the `register`
forms; searched in the file's bytes, so any source encoding works). When pytest starts collecting a test module, the plugin loads the
strategy files of the module's folder and of each folder above it, up to the
rootdir or the search path (see below) that contains it, closest first. Each
folder and file is loaded once per session.

The search paths are the `testpaths` directories (glob patterns expanded), or
the rootdir without `testpaths`, plus the directory of each path given on the
command line. Every strategy file below them is loaded when a name is not
registered on a test's path, for `--list-strategies` and for
`export_strategies()`. Below those directories the search skips hidden
directories, `__pycache__`, `norecursedirs` matches (matched as pytest does, so
`tests/data` works) and virtual environments (directories containing
`pyvenv.cfg` or `conda-meta/history`), and it follows symlinked directories.

Files are imported with pytest's `import_path()`, with the session's
`--import-mode`, `rootdir` and `consider_namespace_packages`, so a strategy file
gets the module name a test module importing it would use. When that name is
taken by another file (`ImportPathMismatchError`, or a module with another
`__file__` under importlib mode), the file is imported under a unique
`pytest_strategies_discovered.*` name instead, and a meta path finder
(`_StrategyFileFinder`, placed after pytest's assertion rewriting hook) hands
that module out when a test module or `conftest.py` imports the file, so the
file is not executed again. A file that a `conftest.py` or test module imported
before the plugin reached it, with its strategies registered, is used as it is.
A file is imported on a stream of its own, keyed by its path relative to the
rootdir (see [Reproducibility](#reproducibility)).

A file that raises while loading is reported with a
`pytest-strategies: Warning - Failed to load ...` line. A file that calls
`pytest.skip()`/`pytest.importorskip()` at module level is skipped (reported
with `-v`). Both kinds are listed in any "Strategy 'name' not found" error,
together with the files matching a pattern that mention `register` but have no
registration decorator (so were not imported).

**`pytest_strategies_context(config)`:** a `firstresult` hook the plugin adds.
A factory with a `ctx` parameter gets its result as `ctx` (when no
implementation returns a value, `ctx` keeps its default, or a value bound with
`functools.partial`, and is `None` without one); other factories never trigger
it. `_factory.call_factory` calls it through `runtime.strategy_context()` the
first time a factory needs it, and the session keeps the result, or the
exception it raised, for every later factory. Each (nested) session and each pytest-xdist worker
calls it once. See the README for an example.

---

## Complete Example

```python
# test_math_operations.py

from pytest_strategy import Parameter, RNGInteger, RNGWeightedInteger, TestArg, register, strategy

@register("division_strategy")
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

@strategy("division_strategy")
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

@register("string_concat_strategy")
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

@strategy("string_concat_strategy")
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
`--no-header`. A failed run also prints
`pytest-strategies: reproduce with --rng-seed=...` after the failure tracebacks,
also under `-q`. A run
without `--rng-seed` picks a seed from the clock, and passing that printed seed
reproduces the run.

**One generator of the plugin's own:** every draw (the RNG types and the `RNG.*`
helpers) comes from `RNG.generator()`, a `random.Random` instance. The plugin
never seeds or draws from Python's global `random` state, so code that uses
`random` directly is not reproduced by `--rng-seed` and does not change the
generated vectors. A factory that needs `shuffle` or `gauss` should call them
on `RNG.generator()`.

**Plugin streams (streams v1, `_streams.py`):** every random stream is keyed by
a `StreamKey` under the run seed S (`runtime.run_seed()`, never the mutable
`RNG._seed`), except the user and direct streams, keyed under
`RNG.get_seed()`:

| Key | What draws from it |
|---|---|
| `T = root(S, "test", strategy, nodeid)` | everything for one strategy on one test |
| `T/"factory"` | the factory call: `rng`, `RNG.*`, `RNG.generator()` |
| `T/"row"/pos/j/name`, `T/"order"/name` | the rows (below) |
| `root(S, "file", path)` | a strategy file's import; `path` is the file's (`_streams.file_part()`) |
| `root(S, "module", path)` | a test module's import (a `pytest_make_collect_report` wrapper for a `Module`); `path` is the module's (`_streams.file_part()`) |
| `root(S, "ctx")` | the `pytest_strategies_context` call |
| `root(S, "fixture", scope, name, param_index, where, qualname, base)` | a fixture's setup (a `pytest_fixture_setup` wrapper); `scope` is the node ID of the node it is set up for, `""` for the session and the rootdir's node (`plugin._node_part()`); `where` and `qualname` are the fixture function's module or file and its qualified name (`plugin._fixture_definition()`), `base` the node ID pytest registered the fixture for (`plugin._fixture_base()`) |
| `root(S, "body", nodeid, phase)` | one phase of a test, `setup`, `call` or `teardown` (wrappers around `pytest_runtest_setup`, `_call` and `_teardown`) |
| `root(RNG.get_seed(), "user", key)` | `RNG.refresh_seed(key=...)` |
| `root(S, "export", name, folder)` | a factory call from `export_strategies()`; `folder` is the factory's module or its file's folder (`_registry.source_part()`), relative to the rootdir; outside a session, the module's name for a module `sys.modules` has under it, else the folder's absolute path |
| `root(RNG.get_seed(), "direct", n)` | `generate_vectors()` and the other generators called directly |

A path in a key is relative to the rootdir in posix form
(`_streams.path_part()`): the real path's when that is in the rootdir, else the
path's as pytest spells it when that is in the rootdir (a folder linked into the
checkout from a place that does not move with it, which every checkout that
links it then keys alike), else the real path's, also outside the rootdir
(`../shared/strategies.py`); it is absolute only on another Windows drive. A
test module or strategy file of an installed package (a file in a
`site-packages` or `dist-packages` folder, a test `--pyargs` runs) has its path
below that folder instead (`_streams.installed_part()`), which does not depend
on where the environment is. A fixture's `where` and an exported factory's
`folder` come from `_registry.source_part()` (through `plugin.definition_part()`).
They are the file, or its folder for an export, for a `conftest.py`, a test
module that `python_files` matches or a strategy file that pytest or the plugin
imports by its path, under a module name that depends on `--import-mode` and on
the `__init__.py` files. A file with such a name counts as one when any of these
holds (`_registry._imported_by_path()`):

1. It is inside the rootdir, by its real path or as it is spelled (a folder
   linked into the checkout counts), whatever the testpaths, `norecursedirs` and
   the folders named on the command line say.
2. It is below a `testpaths` entry (glob patterns expanded), which is static
   configuration: with `testpaths = ../shared`, `../shared/test_x.py` keeps its
   path.
3. pytest or the plugin imported that very file by its path in this session: a
   test module pytest collected (the `pytest_make_collect_report` wrapper sees
   each `Module`), a `conftest.py` it loaded (pytest registers it as a plugin
   under its path; `pytest_plugin_registered` is historic, so the conftest files
   loaded before the plugin registers are seen too), or a strategy file the
   plugin loaded. The session records their real paths in
   `SessionState.imported_files` as they are imported; outside a session the set
   is empty. A key reads the set when it is built: a fixture's when it first
   draws, after collection; an export's when `export_strategies()` runs, which
   during collection sees only the files imported so far. When the session's
   `consider_namespace_packages` is false (pytest's default), rule 3 leaves out a
   module of a regular package that `sys.modules` holds under its package name
   (`_registry._held_under_package_name()`): the dotted name of the chain of
   folders with an `__init__.py` above it, ending at the module
   (`acme.test_utils` for `acme/test_utils.py`; the chain stops where
   `_pytest.pathlib.resolve_package_path()` stops it, at a folder without one or
   whose name is not an identifier). Without the option, pytest 8.4 and 9 import
   such a module under that name in every import mode, the name another module's
   `import acme.test_utils` gives it, so the rules below key it alike whether or
   not the run collects its folder. With `consider_namespace_packages = true`,
   pytest names it from `sys.path` instead (`ns.acme.test_utils` where the folder
   above a namespace folder `ns/` is on `sys.path`, `acme.test_utils` elsewhere,
   or a name importlib makes from the path), which depends on the launcher (the
   `pytest` script puts its own folder on `sys.path`, `python -m pytest` the
   working directory), the working directory and `--import-mode`, so rule 3
   applies to every file it recorded. `plugin.definition_part()` reads the option
   (`True` without a config) and passes it to `source_part()`.

Every run of one checkout therefore keys the files of rules 1 and 2 alike: a full
run, the run of a folder outside the testpaths (`pytest tests/integration` with
`testpaths = tests/unit`, `pytest examples/` here, or a folder outside the rootdir
with `-c` or `--rootdir`), the run of one node ID, every `--import-mode`, pytest 8
and 9. (That is the file's part of a key. pytest names a test outside the rootdir
from the path named on the command line, `test_a.py::test_a` in the run of its
folder and `::test_a` in the run of its node ID, so the keys that hold node IDs, a
fixture's `scope` and `base` and a test's `T` and body streams, differ between
those two runs whatever the file's part.) A file with such a name for which none
of them holds is a module of a library on `sys.path` (an editable install's `.pth`
entry, `PYTHONPATH`, a `pip install` target folder), such as
`extacme/strategies.py` or `extacme/test_helpers.py`: its path relative to the
rootdir would change with the folder the checkout is in (`../../libs/extacme` in
one, `../../../../libs/extacme` in another), so the rules for any other module
apply to it. They apply as well to a regular package's module that rule 3 leaves
out, such as `acme/test_utils.py` next to a rootdir in `tests/` (a flat layout).

They are the module's name for a module imported by its name: an installed
package's (in a `site-packages` or `dist-packages` folder, also when it is named
like a test module), an editable install's, a plugin's or a helper module's, so a
package's fixture draws the same installed, installed in editable mode or checked
out next to the tests (also next to a rootdir in the checkout's `tests/` folder),
and for code with no file (`exec`'d code, whose `"<string>"` would resolve against
the working directory). They are the file for a module that `sys.modules` does
not have under its name, and for a module whose name begins with the rootdir's
folder or a folder above it (a rootdir with an `__init__.py`, or `proj.util` for
the tests of a package checkout in `proj/tests`), whose name depends on the
folders the checkout is in.

These limitations remain. A package module inside the rootdir whose file name
matches `python_files` or a strategy file pattern (`src/acme/test_utils.py`,
`src/acme/strategies.py`) keeps its path there and has its module's name
installed, so its fixtures and exported factories draw other values from the
checkout or an editable install of it than from the installed package. Renaming
the module (`src/acme/testing.py`) avoids that. Telling such a module from a test
module by what the session collects would make the key depend on the run: pytest
collects the same file in one run and not in another (a bare `pytest` without
testpaths collects `src/acme/test_utils.py`), and a file it collects has a module
name that depends on `--import-mode`.

And rule 3 keys these files by their paths in a run that collects their folder
and by their module's names in a run that does not, so the two runs draw other
values:

- (a) a helper named like a test module or a strategy file, outside the rootdir
  and the testpaths, not in a regular package (its folder has no `__init__.py`),
  imported by its name: `pytest -c pytest.ini ../other`, run in `proj/`, collects
  `../other/test_b.py` and keys it by its path; the run of one node ID in
  `../other/test_a.py`, which does `from test_b import port`, does not collect it
  and keys it by its module's name;
- (b) a regular package's module that `sys.modules` holds only under a longer
  namespace-package name, such as a strategy file `ns/acme/strategies.py` that a
  `conftest.py` imports as `ns.acme.strategies`;
- (c) with `consider_namespace_packages = true`, (a) also holds for a regular
  package's module outside the rootdir and the testpaths: `acme/test_utils.py`
  next to a rootdir in `tests/` is keyed by its path in `pytest -c
  tests/pytest.ini` from the project, which collects `acme/`, and by its name in
  `pytest` from `tests/`.

Listing such a folder in `testpaths` keys its files by their paths in every run.

A fixture's definition and base are in its key because pytest sets up several
fixtures of one name for the same scope node: an override that requests the
fixture it overrides (`def x(x)`), the session fixtures of one name in two
sibling folders' `conftest.py` files, and one fixture function that two
`conftest.py` files import (`from helpers.fixtures import port`), which only
`base` tells apart: the folder of the `conftest.py`, the test module or class
the fixture is registered for, or `""` for a plugin and the rootdir's
`conftest.py` (`FixtureDef.node` on pytest 9, whose node ID is `"."` there, and
`FixtureDef.baseid` on pytest 8). `plugin._node_part()` turns `"."` into `""`
for `base` and for `scope`: when the rootdir has an `__init__.py`, pytest 9 sets
a package-scoped fixture of its `conftest.py` up for the rootdir's `Package`
(`"."`), and pytest 8 for the session.

Each non-row stream runs in an `rng._Stream` block: in the block,
`RNG._ambient` (the plugin's generator, an `rng._Ambient`) draws from the key's
stream as if reseeded in place from it, and is installed as `RNG._generator`;
when the block ends, also on an exception, the generator's state,
`RNG._generator` and `RNG._seed` are put back. Streams nest. Reseeding in place
keeps a generator taken earlier from `RNG.generator()` on the current stream,
and installing the ambient generator lets a strategy file imported from inside
a constraint (when `RNG._generator` is an argument's) draw from its file
stream. An `RNG.seed()` call inside a stream changes only the rest of that
stream, and `RNG.get_seed()` is S everywhere else. `_Stream` is a class with
`__enter__` and `__exit__`, not a `contextlib.contextmanager`, so that a frozen
dataclass exception passes through unchanged.

The seeding is lazy. Entering a block records its key (or a function that
builds it) as pending; `_Ambient` seeds itself from it at the first `random()`
or `getrandbits()`, or when `getstate()` reads its state, and only then saves
the state in use for the block to put back. `seed()` and `setstate()` replace
the state, so they skip the pending seeding but still save that state.
`random.Random` draws only through `random()` and `getrandbits()` (its other
methods call them, and `_randbelow` stays the `getrandbits()` one), so the
values are those of a generator seeded when the block starts; a pending block
also clears `gauss_next`, as seeding does, and puts it back. Seeding a pending
block, and entering and ending one, hold the generator's lock
(`_Ambient._lock`, an `RLock`): a thread that draws while another enters or
ends a block (a stimulus thread that a fixture starts) never seeds the
generator from a block that has ended or saves the state on another block, and
a block stays pending until it is seeded, so such a draw waits for the seeding
instead of drawing from the state the seeding replaces. Its draws still come
from whichever block is in use, so they are not reproducible. Blocks that two
threads enter may end in any order (`export_strategies()` in a thread while a
test phase runs): a block that ends before a block entered after it hands that
block what it would have put back. A draw from inside the seeding (a signal
handler, a garbage collector callback) draws from the state in use, without
seeding the block again or saving the state twice. Drawing a row holds the lock
too (`Parameter._build_row()`), because the argument generators it installs as
`RNG._generator` are process-wide: another thread that draws a row, or enters
or ends a block, meanwhile cannot take one of them for the generator to put
back, which would leave it installed after the rows. Another thread's `RNG.*`
draw meanwhile still draws from it. An `os.register_at_fork` hook takes the
lock before a fork and gives the child a new one, so a child forked while
another thread seeds a block can draw. The plugin passes the fixture, phase and
test module streams a function that builds the key, so a block that draws
nothing costs about 2 µs, and one that draws about 20 µs more, mostly the
seeding; each draw from the ambient generator costs about 0.1 µs more than from
a plain `random.Random`. Most test phases and fixtures draw nothing: a run of
20,000 trivial tests takes about 5% longer than without these streams, most of
it in the three hook wrappers per test.

**Row streams (streams v1, `_streams.py`):** the rows draw from streams keyed
under `T = StreamKey.root(seed, "test", strategy, nodeid)`, where `nodeid` is the
test's node ID without its parameters (`metafunc.definition.nodeid`). Each drawn
argument of a row draws from `T/"row"/pos/j/name`: `pos` is the row's enumerated
position, its `(name, token)` pairs sorted by name and flattened (the tokens of
`_position_keys`), `j` is the row's index within that position, and `name` is
the argument's name. The order of an `RNGSequence` under `--nsamples=auto` comes
from `T/"order"/name`. A row's values therefore depend only on the seed, the
strategy, the test, the row and the argument: more rows keep the first ones, a
node ID run alone gets the values of the full run, adding, reordering or
changing another argument leaves an argument's values alone, and a constraint
redraws only the rows it rejects, continuing the same streams. Static `value=`
arguments draw nothing. Direct calls (`generate_vectors()`, `generate_vector()`,
`generate_exhaustive()` outside the plugin) use the key
`root(RNG.get_seed(), "direct", n)`, where `n` is 128 bits drawn from
`RNG._generator` once per call. The vectors of a test do not depend on which
other tests are collected, on the collection order or on `--import-mode`. Two
tests that share a strategy get different vectors, and so do two classes that
inherit one test method. For the same seed, directed and test vectors and
`Series` values are those of 3.x, unless the factory draws them; random rows
and the values that factories, strategy files and the context hook draw differ
from 3.x's.

**Guard on draws outside the row streams:** these properties hold when every
random value of a row comes from its arguments' RNG types, drawn inside
`generate()` from `RNG.generator()` or the `RNG.*` helpers, with no state kept
between calls (the `RNGType` docstring says so). Constraints run with the
ambient generator installed, and a factory's `rng` is the ambient generator
itself, so a constraint that calls `RNG.*` or an RNG type that draws from a
generator kept from the factory draws from `RNG._ambient`: its values depend on
what was drawn before (other rows, other tests of the module), not only on the
row. `build_parametrization` compares `RNG._ambient._position()` before and
after `_generate_rows()`: the pending key while nothing has used the current
stream, which costs nothing, or else the generator's state (about 20 µs per
test). For a test function the current stream is its module's, so the check is
free unless the module drew when it was imported; pytest collects a class in a
collect report of its own, outside the module's stream, so a test method always
reads the state. When it changed, it emits one
`PytestStrategiesWarning` inside `_attributed_warnings`, so it is prefixed with
`Strategy '<name>' (<test>): ` and points at the test. It is a warning, not an
error, because the values still repeat for the same seed, tests and options;
under `filterwarnings = error` it becomes the collection error
`Error generating samples for strategy ...`. A stream that runs inside a
constraint (a strategy file imported there) puts the ambient generator back and
does not trigger it. A generator an RNG type creates for itself is not the
plugin's, so the guard cannot see it.

**Golden values:** `tests/golden/seed1.json` holds the values of streams v1 for
`--rng-seed=1`: every row of a small project that draws with every built-in RNG
type, with predicates and under a constraint, in `Series` and
`per_sequence_samples` rows and in an `RNGSequence` permutation under
`--nsamples=auto`, with values that a factory, a strategy file and the context
hook drew. `tests/integration/test_golden_values_integration.py` checks them on
every CI cell (Linux and Windows, Python 3.11 to 3.14, pytest 8 and 9), under
two `PYTHONHASHSEED` values. A change that fails it gives every recorded seed
other values, so it belongs in a major release, with a new `_streams.VERSION`.
The values also rest on `random.Random`'s `randint`, `choice`, `choices` and
`sample`, which CPython may change; a CPython that does gets rows of its own in
the file, keyed by its version. A built-in RNG type added later gets a strategy
and a test of its own in the project, which adds rows and changes none.

**Import time:** `pytest_configure` restarts the generator from the seed. A
strategy file is imported on its file stream and a test module on its module
stream, so their import-time draws do not depend on what was collected before,
and a module collected alone gets the values of the full run. Draws when a
`conftest.py` is imported are not keyed: pytest imports the initial conftests
(the rootdir's and those of the folders on the command line) before
`pytest_configure` seeds, and a node-ID rerun makes another conftest an initial
one, so a key would give the full run and the rerun different values. The
others are imported during collection, outside any stream. Any other module (a
helper that test modules or strategy files import) runs on the stream of the
first module that imports it, and shifts that module's later draws, so both
depend on what was collected before; the README says to move such draws into a
fixture, the context hook or a strategy file. A strategy file
that a `conftest.py` imports at its top is reused as it is, so its import-time
draws do not follow the seed either; the README says to import it inside a
fixture or hook. When a session ends (including an in-process `pytester` run),
the seed, the ambient generator's state, the installed generator and the
strategy registry are restored to what they were when it began.

**pytest-xdist:** the controller sends its seed to the workers, so `-n` works
with or without `--rng-seed` and every worker generates the same tests.

**Test bodies and fixtures:** each phase of a test (setup, call and teardown)
runs on its body stream and each fixture's setup on its fixture stream, so a
test body's `RNG` draws and a fixture's are the same whether the test runs
alone, in the suite, in another order or under xdist. A fixture's stream is
also keyed by its definition and where pytest registered it, so a fixture that
overrides another of the same name and requests it, two sibling folders'
session fixtures of one name, or one fixture function imported into two
folders' `conftest.py` files, draw different values. A module- or
session-scoped fixture is set up during the setup of whichever test needs it
first; with a stream of its own, neither its draws nor that test's depend on
which test that is. A fixture's teardown and a finalizer run in the teardown of
the test that ends the fixture's scope, and draw from that test's teardown
stream. The pseudo-fixtures of direct parametrization draw nothing and get no
stream. `RNG.refresh_seed(key=request.node.nodeid)` is no longer needed in a
test body; it still gives the stream of the seed and the key, whatever ran
before it.

For plain `random` calls, seed the global state per test from the run's seed,
for example in an autouse fixture:
`random.seed(f"{RNG.get_seed()}:{request.node.nodeid}")` (see the README).

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

A predicate on an RNG type rejected 100 draws in a row. Either:
- Relax the predicate, or narrow the range so most values pass it
- Move the rule to `vector_constraints`, whose redraws `Parameter(max_retries=...)` controls
- Use directed vectors instead

### "Could not generate random row K ..." or "Could not generate valid vector ..."

The `vector_constraints` rejected every draw (or, with `Series`/`RNGSequence`
args, every combination). The message counts the draws by the name of the first
constraint that rejected each one, and shows the first row each rejected, so you
can see which one is too strict. Relax it, raise `Parameter(max_retries=...)`,
or turn it off for one run with `--strategy-constraint-off=STRATEGY:NAME`.
The related `PytestStrategiesWarning` "Series combination (...) skipped" means
one combination was skipped after `max_retries` redraws of its random args.

### "Strategy not found"

Make sure you:
1. Registered the strategy with `@register("name")`
2. Used the exact same name in `@strategy("name")` (the error suggests close
   names)
3. Registered it in the test's folder or a folder above it: in the test module,
   a `conftest.py`, or a strategy file (see "Strategy files" above;
   `pytest -vv` lists the loaded files)

The error message lists the names the test can use, the strategy files that
failed to load or skipped themselves (the load failures are also printed when
they happen), and files named like strategy files that were not imported
because they have no registration decorator.

### "Strategy 'name' is not registered in ... and several other folders register it"

The test's folder and the folders above it do not register the name, and
several other folders do. Register the strategy in a folder above the test, or
pass the factory itself: `@strategy(factory)`.

### "... would generate N rows ..., more than the limit"

`--nsamples=auto` or `per_sequence_samples=True` would generate more rows than
the size guard allows. Use fewer sequence values, or raise the limit with
`Parameter(max_exhaustive=...)` or the `strategies_max_exhaustive` ini option.

### Tests not reproducible

- Pass the same `--rng-seed` value (a run's seed is shown in the report header, and after a failed run); calling `RNG.seed()` inside a test body does not change its parametrized values
- Use the same rootdir and a pytest-strategies version that generates the same values (2.0.0 and 3.0.0 do, except values strategy files draw when they are imported; 1.x and 4.0.0 do not)
- Draw from the RNG types or `RNG.generator()` in factories: plain `random` calls are not seeded by the plugin
- Draws made when a `conftest.py` is imported are not covered by the seed, and those of a helper module that test modules import depend on which module imports it first; move them into the context hook, a fixture or a strategy file (see [Reproducibility](#reproducibility))

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

Set up and check a change the way CI does:

```bash
pip install -e ".[dev]"
python -m pytest -n auto                  # The suite (warnings are errors)
python -m pytest examples/*.py --nsamples=auto
ruff check src/ tests/ benchmarks/ && black --check src/ tests/ benchmarks/
mypy --strict src/pytest_strategy/ tests/unittests/test_typing.py
```

`benchmarks/bench.py` times row generation through
`Parameter.generate_vectors()`: 10,000 rows of 5 arguments, 100,000 rows of 4,
and 5,000 rows of 2 arguments whose constraints reject about half the draws.
`--sweep` adds that last case at about 0%, 50% and 90% rejection, per accepted
row, and `--memory` the collection of 100,000 exhaustive rows in a new
interpreter (time and peak memory). CI runs `python benchmarks/bench.py --sweep
--memory` as an informational step of the examples job, which never fails it.
Timings compare only on one machine: to compare two versions, run the script
once with each, the other version's `src` on `PYTHONPATH`. On Python 3.11
(Linux), 4.0 generates 10,000 rows of 5 arguments in about 0.47 s (3.0: 0.05 s),
about 8.5 µs more per drawn argument per row, almost all of it seeding the
argument's generator. A rejected row continues its arguments' streams instead
of reseeding them, so the cost per accepted row does not grow with the rejection
rate: 16 to 17 µs more than 3.0 at 0%, 50% and 90%. Collecting 100,000
exhaustive rows takes about 10 s and 417 MiB at peak (3.0: 6 s and 328 MiB).

Options that a 4.x release adds to a public callable go after a `*`, so that
they are keyword-only and no existing positional call changes meaning. Since 4.0,
`TestArg`'s options after `rng_type`, `strategy()`'s `validate_signature`,
`export_strategies()`'s `format` and `Parameter.generate_vectors()`'s options
after `n` are keyword-only.

The suite runs with `filterwarnings = error`, `--strict-markers`,
`--strict-config` and `empty_parameter_set_mark = fail_at_collect`, and an
autouse fixture in `tests/conftest.py` restores the registry, the seed and the
random state after each test. Inner `pytester` runs that expect a warning use
`runpytest_subprocess`, so the outer `error` filter does not apply to them.

## Releasing

1. Set the version in `pyproject.toml` and `__version__` in
   `src/pytest_strategy/__init__.py` (a test checks they match), and turn
   `## [Unreleased]` in `CHANGELOG.md` into `## [X.Y.Z] - <date>` with a new empty
   `[Unreleased]` above it and a comparison link at the bottom.
2. Merge into `main` once CI is green.
3. Tag that commit and push the tag:
   `git tag -a vX.Y.Z -m "pytest-strategies X.Y.Z" && git push origin vX.Y.Z`.
   Alternatively, run the Release workflow by hand on `main` with the version;
   it creates the tag itself.
4. The Release workflow (`.github/workflows/release.yml`) checks that the tag
   matches `pyproject.toml`, builds the sdist and wheel, runs the examples
   against the wheel and publishes a GitHub Release with the CHANGELOG section
   as notes and both files attached.

## License

MIT License. See [LICENSE](../LICENSE) for details.

---

**pytest_strategy** - Making randomized testing easy and powerful! 🎲✨