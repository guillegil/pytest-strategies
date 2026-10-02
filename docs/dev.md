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
├── _vector.py           # Vector, the class of the generated rows (a namedtuple per argument names)
├── test_args.py         # TestArg class (single argument definition)
├── rng.py               # Random number generation and RNG types
├── hookspecs.py         # Hooks the plugin adds (pytest_strategies_context)
├── skill/               # The agent skill that pytest-strategies skill install copies
├── py.typed             # PEP 561 marker: type checkers use the package's annotations
└── _*.py                # Other internal helpers (introspection, test IDs, record mode,
                         # StrategyOptions, factory calls, runtime state, warning categories)
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
plain, Series, `per_sequence_samples` and exhaustive paths alike. Argument names
must therefore be identifiers that are not keywords and do not start with `_`.

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
the exception's `__traceback__` on the way out. The private
`_stats` keyword of `generate_vectors()` and `generate_exhaustive()` collects the
counts (and the combinations `--nsamples=auto` left out) for the `-v` summary.
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
vector. `generate_exhaustive()` (used for `--nsamples=auto`) builds the
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
`pytest.param(None, ..., marks=pytest.mark.skip(reason=...), id="skipped")` row,
also in dataclass mode, where no instance is built but the dataclass fields are
still checked against the strategy's arguments. `skip_if_empty` is keyword-only,
and a non-callable `predicate` (such as a reason passed positionally) raises.

The `Parameter` copies the `directed_vectors`, `test_vectors` and
`vector_constraints` it is given, so `add_*`/`remove_*` never change the
caller's dicts and lists. Every directed and test vector goes through
`_normalize_vector()` (in `__init__` and the `add_*` methods), which stores it
as a `Vector` in declaration order: a dict or a namedtuple by name, a tuple, list
or other iterable by position, and a `pytest.param` rebuilt around the `Vector`
with its marks and id. Strings, bytes, scalars and record instances fail there,
and so do names that are not non-empty strings. `directed_vectors` and
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
4. `build_parametrization()` restarts the generator on the strategy and test's
   own stream (see [Reproducibility](#reproducibility)), then calls the factory
   once through `_factory.call_factory()`. The factory gets, by name, the inputs
   it declares: `nsamples`, `ctx`, `rng` and `options` (see "Factory inputs"
   below).
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
- A random stream of its own for each strategy and test
- CLI option integration
- Readable test IDs. A value whose repr contains a memory address is shown by
  its type name, and set elements are sorted (also inside tuples, lists,
  dicts, and dataclass and namedtuple values that keep their generated repr),
  so IDs are the same on every run and on every xdist worker.

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
- `pytest_configure` - Opens the session's state and sets the run's seed,
  restarting the plugin's generator from it; a pytest-xdist worker without
  `--rng-seed` takes the controller's seed
- `pytest_configure_node` - (pytest-xdist only) sends the controller's seed to
  each worker
- `pytest_collectstart` - Before a test module is imported, loads the strategy
  files of its folder and the folders above it
- `pytest_generate_tests` - Resolves the test's `strategy` markers into
  `parametrize` markers. A second, module-level implementation runs last: it
  fails a test written for record mode whose fixtures take every argument by
  name when no fixture or parametrization gives its record parameter a value
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
  Summary (tests and directed, random and test rows per strategy, and where
  `nsamples` came from)

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
The generator is restarted on a stream keyed by the file's path relative to the
rootdir while a file is imported, and restored afterwards.

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

**Per-test streams:** before calling a strategy's factory for a test, the plugin
restarts the generator from the run seed and a key made of the strategy name,
the test file's path relative to the rootdir and the test's qualified name
(`RNG.refresh_seed(key=...)`). Each strategy and test pair therefore gets its
own stream. The vectors of a test do not depend on which other tests are
collected, on the collection order or on `--import-mode`. Two tests that share
a strategy get different vectors. For the same seed, the values are those of
2.0.0, and differ from those of 1.x (1.0.0 and the 1.1.0 pre-releases).

**Import time:** `pytest_configure` restarts the generator from the seed. A
strategy file is imported on a stream of its own (keyed by its path), and the
generator's state is restored afterwards, so its import-time draws do not
depend on what was collected before. Draws at module level in a test module
come from the generator as the tests collected before it left it: they follow
the seed but change when the module is collected alone. When a session ends
(including an in-process `pytester` run), the seed, the generator's state and
the strategy registry are restored to what they were when it began.

**pytest-xdist:** the controller sends its seed to the workers, so `-n` works
with or without `--rng-seed` and every worker generates the same tests.

**Test bodies:** the seed reproduces the parameters, not random values drawn
inside a test body. `RNG` draws there come from the generator as the earlier
collection and tests left it, so they change when a single test is rerun or
tests are scheduled differently under xdist. Calling `RNG.seed()` inside a test
body does not change the test's parameters, which are already fixed by then. To
make a body's own draws reproducible, reseed in the body. A stream keyed by the
node ID still follows `--rng-seed`:

```python
import random

from pytest_strategy import RNG

def test_something(request):
    RNG.refresh_seed(key=request.node.nodeid)
    value = RNG.integer(0, 100)  # Same value for the same --rng-seed

def test_fixed():
    value = random.Random(42).randint(0, 100)  # Same value on every run
```

Do not call `RNG.seed()` in a test body: it restarts the plugin's generator from
another seed, so the tests after it that reseed from `RNG.get_seed()` no longer
follow `--rng-seed`.

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
- Use the same rootdir and a pytest-strategies version that generates the same values (2.0.0 and 3.0.0 do, except values strategy files draw when they are imported; 1.x does not)
- Draw from the RNG types or `RNG.generator()` in factories: plain `random` calls are not seeded by the plugin
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

Set up and check a change the way CI does:

```bash
pip install -e ".[dev]"
python -m pytest -n auto                  # The suite (warnings are errors)
python -m pytest examples/*.py --nsamples=auto
ruff check src/ tests/ && black --check src/ tests/
mypy --strict src/pytest_strategy/ tests/unittests/test_typing.py
```

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