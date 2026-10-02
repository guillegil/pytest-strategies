# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `StrategyOptions`, exported from `pytest_strategy`: the run's options for one strategy. Its fields are `strategy` (the resolved name), `nsamples` (the `--nsamples` value, `"auto"`, or 10 without the option), `nsamples_source` (`"--nsamples"` or `"default"`), `mode` (the `--vector-mode` value), `vector_name`, `vector_index` and `constraints_off`, and its `filtered` property is true under `--vector-name` or `--vector-index`. It is a frozen dataclass whose fields are keyword-only, so later 4.x releases can add fields with defaults. The plugin reads the options once per session and builds one instance per strategy name, and a factory receives its strategy's instance as `options`.
- Factories can declare `rng` and `options` (see Changed): `rng` is the plugin's `random.Random`, the same object as `RNG.generator()` during the call, and `options` is the strategy's `StrategyOptions`.
- `Vector`, exported from `pytest_strategy`: the rows a `Parameter` generates are instances of a namedtuple over the strategy's argument names that subclasses `Vector`, so a constraint can read `v.addr` as well as `v[0]`. `Parameter.vector_type` returns the class, and a misspelled field raises `AttributeError: Vector has no argument 'lenght'; its arguments are addr, len`. A Vector compares and hashes like the plain tuple of its values, can be pickled and copied, and under `mypy --strict` its fields read as `Any`, so a constraint annotated `-> bool` returns `bool(...)`.
- Named constraints: `vector_constraints={"aligned": aligned, "no_4k_cross": no_4k_cross}` names each constraint. A list still works and names each constraint after its function's `__name__`, and a lambda, `functools.partial` or callable object `constraint_<i>`, i being its position. A name is a non-empty string without whitespace, `:`, `,` or `=` (`.` is allowed). `add_constraint(fn, *, name=None)` returns the name it used, and the new `remove_constraint(name)` removes one (`KeyError` listing the names otherwise).
- Constraint diagnostics. When the draws of a row run out, the error counts them by the name of the first constraint that rejected each draw, and shows the first row each constraint rejected: `Could not generate random row 3 after max_retries=100 draws. Rejected by (first failing constraint per draw): aligned=97, no_4k_cross=3. First rows rejected: aligned: Vector(addr=4097, len=16); no_4k_cross: Vector(addr=4090, len=16). Raise Parameter(max_retries=...), relax a constraint, or turn one off for this run with --strategy-constraint-off=dma_burst:aligned.` The last sentence names the constraint with the most rejections. A lambda appears as `constraint_1 (lambda at strategies.py:42)`, a constraint that returned None for every draw it rejected gets `returned None, missing return?`, and the rows are cut to about 120 characters. The "Series combination (...) skipped" and "Sequence combination (...) produced k of n rows" warnings and the errors for constraints that reject every combination count the rejections by name too. With `-v`, the Strategy Summary adds `rejected: aligned=412, no_4k_cross=37` to each strategy with constraints, and `left out: N combinations` to those generated with `--nsamples=auto`.
- A constraint that raises fails the collection with an error that names it and the row, followed by the constraint's own frames: `Constraint 'ratio' raised ZeroDivisionError on random row 3, Vector(addr=4, len=0): division by zero`. A predicate that rejects every draw names its argument: `Argument 'width' could not draw a value its predicate accepts: No valid value found after 100 attempts`.
- `--strategy-constraint-off=[STRATEGY:]NAME[,...]` turns named constraints off for one run, like SystemVerilog's `constraint_mode(0)`: `NAME` in every strategy that has a constraint with that name, `STRATEGY:NAME` only in the strategy resolved under that name. Items are separated by commas and the option can be repeated, so `--strategy-constraint-off=dma_burst:aligned,no_4k_cross` turns `aligned` off in `dma_burst` and `no_4k_cross` everywhere. Each item is split at its last `:`, so a strategy name may contain `:`; whitespace, empty items, `:x` and `x:` are usage errors when the command line is parsed. The `Parameter` keeps its constraints (a cached factory's too), directed and test vectors do not change, and a factory sees the names in `options.constraints_off`. The header gets `pytest-strategies: constraints off: dma_burst:aligned, no_4k_cross`, and `-v` adds `off: aligned` to the strategy's summary line. An item that matches no constraint of any strategy the run resolved stops a run of the whole suite with a usage error (exit code 4, or 2 under pytest-xdist): `--strategy-constraint-off=no_4k_crss matched no constraint. Did you mean 'no_4k_cross'? Constraints by strategy: dma_burst: aligned, no_4k_cross; esm: none`. A run given paths or node IDs, `--lf`, `--sw`, `--ignore` or `--ignore-glob` resolves only some strategies, so it prints the message in red and goes on. Nothing is checked under `--list-strategies` or when no `Parameter` strategy was resolved. When a raising constraint comes after one that is turned off, its error ends with `(constraint 'nonzero' before it is turned off by --strategy-constraint-off)`. `Parameter.generate_vectors()` and `generate_exhaustive()` take the names as the keyword-only `constraints_off` (a name the `Parameter` does not have is a `ValueError`).
- Directed and test vectors by name: `directed_vectors={"zeros": {"addr": 0, "len": 0}}` gives the values in any order, and the keys must be the strategy's argument names (`Directed vector 'zeros' has unknown argument 'lenght' (did you mean 'len'?) and is missing 'len'. ...`). Tuple, list and dict vectors can be mixed in one `Parameter`, `add_directed_vector()` and `add_test_vector()` take dicts too, and `pytest.param({"addr": 0, "len": 0}, marks=...)` is a named vector with marks. A dict is always a named vector, so a dict value for a one-argument strategy is written `({"a": 1},)` or `{"cfg": {"a": 1}}`.

- Dataclass mode works for one-argument strategies, generic dataclasses (`def test_p(p: Pair[int])`) and dataclasses with an `init=False` field that `__init__` does not set. 3.0 required two arguments, ignored generic aliases, and failed building the test IDs of the last.

### Changed
- Factories receive their inputs by name, like fixtures: each parameter named `nsamples`, `ctx`, `rng` or `options` gets that value, in any order, positional-only or keyword-only, and nothing else is passed. What changes from 3.0:
  - A parameter without a default whose name is not one of these fails the collection of the tests that use the strategy, before the factory or the context hook runs: `Strategy factory 'burst' (tests/strategies.py:12:burst) has a parameter 'n', which the plugin does not provide. Factories receive arguments by name: nsamples, ctx, rng, options. Did you mean 'nsamples'? ...`. 3.0 passed `nsamples` by position to `def factory(n)` and `def factory(n, /)`: rename the parameter to `nsamples`. A parameter with a default keeps it (3.0 passed `nsamples` to `def factory(n=5)`).
  - `*args` and `**kwargs` receive nothing (3.0 passed `nsamples` to them).
  - Parameters named `rng` or `options` now receive values, and `base`, `config` and `request` are reserved: a factory that declares one fails, even with a default (`has a parameter 'config', a name the plugin reserves: ...`).
  - With `mock.patch`, the mock parameters must come first (`def factory(getcwd, nsamples)`). A factory whose mocks come after the inputs fails with a message that says so.
  - A decorator without `functools.wraps` hides the factory's signature, so the factory is called with no arguments, and a `TypeError` it raises gets a hint to add `functools.wraps`. 3.0 tried `factory(nsamples=...)` and then `factory(nsamples)`. A `functools.wraps` wrapper with only `*args` gets the inputs by position.
  - The factory is called exactly once: the 3.0 retry after a `TypeError` is gone.
  - An `async def` factory fails with `async factories are not supported` without being called, and a factory that returns a coroutine fails after the coroutine is closed, so no "coroutine was never awaited" warning follows. The check looks through partials and `functools.wraps` decorators, so an `async def` function behind a plain decorator that runs it to completion (`return asyncio.run(fn(*args, **kwargs))` in a `functools.wraps` wrapper), which 3.0 called, fails too: make the factory a plain function.
- `export_strategies()` calls each factory with the inputs it declares, as at collection, and with the session's options for its strategy: `nsamples` is the `--nsamples` value, `"auto"`, or 10 without the option (3.0 passed 1), and `options` is the instance the strategy's tests get. A factory without a `ctx` parameter still never runs the context hook.
- Options are keyword-only: `TestArg(name, rng_type=None, *, value=None, validator=None, description="")`, `strategy(name, *, validate_signature=True)`, `export_strategies(*, format="json")` and `Parameter.generate_vectors(n, *, mode="all", filter_by_name=None, filter_by_index=None, constraints_off=())`. Passing one by position raises Python's `TypeError` (for example `strategy() takes 1 positional argument but 2 were given`), and mypy reports the call: write `TestArg("x", value=5)` and `@strategy("name", validate_signature=False)`. `pytest --markers` and the error for a malformed `strategy` marker show `strategy(name_or_factory, *, validate_signature=True)`. Options that later 4.x releases add will be keyword-only too.
- Constraints receive each row as a `Vector`, and `generate_vector()`, `generate_vectors()`, `generate_exhaustive()` and the `get_directed_vector()`, `get_test_vector()`, `get_vector_by_name()` and `get_vector_by_index()` methods return Vectors, whose repr is `Vector(addr=0, len=16)`. A `pytest.param` vector comes back as a new `pytest.param` with the same marks and id and a Vector as its values, not as the object it was given. `isinstance(v, tuple)`, indexing, unpacking and `==` with a tuple keep working, but `type(v) is tuple` is false, and an argument named `count` or `index` shadows the tuple method of that name (`tuple.index(v, x)` still works). `vector_constraints` takes a `Mapping[str, Callable[[Vector], object]]` or an iterable of `Callable[[Vector], object]`, and `add_constraint()` takes a `Callable[[Vector], object]`, so constraints annotated for tuples still type-check.
- Directed and test vectors are checked and stored as Vectors when the `Parameter` is built or `add_directed_vector()`/`add_test_vector()` is called. The errors are `RNGValueError`, a `ValueError` subclass:
  - A namedtuple vector is placed by its field names, which must be the strategy's argument names (`BusTxn(len=4, addr=0)` gives `Vector(addr=0, len=4)`). 3.0 placed it by position.
  - A `str`, `bytes`, `bytearray` or non-iterable vector fails: `Directed vector 'v' is 'a' (str), not a tuple of values. For a one-argument strategy write ('a',) or {'x': 'a'}`. 3.0 split a string into its characters, so `"a"` happened to work in a one-argument strategy and `b"\x00"` gave `0`.
  - A dataclass or pydantic model instance fails with `record instances as vectors are not supported yet; use a dict, ...`.
  - Vector names must be non-empty strings: `Directed vector names must be non-empty strings, got 0`. 3.0 accepted any key, including `""`.
  - A vector of the wrong length given to `add_directed_vector()` or `add_test_vector()` reads `Directed vector 'x' has 3 values, expected 2`, as in the constructor, instead of `Vector must have 2 values, got 3`.
- `vector_constraints` is a read-only mapping of names to constraints, in the order they run: iterating it gives the names, not the functions, and `param.vector_constraints.append(f)` or `param.vector_constraints["x"] = f` fails. Use `add_constraint()`, `remove_constraint()` and `clear_constraints()`; `len()` works as before. A `Parameter` whose constraints break the naming rules fails when it is built (`RNGValueError`, or `TypeError` for the wrong kind of value):
  - Two constraints with one name, typically closures made by one helper: `Two constraints are named 'check': within.<locals>.check at strategies.py:30 and within.<locals>.check at strategies.py:30. ...`. Name them with a dict.
  - The same function given twice, a constraint that is not callable, an invalid name, and a single function or a string instead of a list or dict.
  - `add_constraint()`'s argument is now called `fn`, and `name` is keyword-only.
- The messages for constraints that reject every draw change their text (see Added): `Could not generate valid vector after N attempts. Check your constraints. Rejected by: constraint #0 (lambda) rejected ...` is now `Could not generate random row K after max_retries=N draws. Rejected by (first failing constraint per draw): ...` (`a random row` for `generate_vector()`), and the errors for constraints that reject every Series or sequence combination end with the counts by name.
- `directed_vectors` and `test_vectors` are read-only mappings: `param.directed_vectors["x"] = (1,)` raises `TypeError`, and assigning the attribute raises `AttributeError`. Use `add_directed_vector()`, `remove_directed_vector()`, `add_test_vector()` and `remove_test_vector()`. They still compare equal to dicts, so `param.directed_vectors == {"zeros": (0, 0)}` keeps passing.
- Argument names must be valid `Vector` fields: a `Parameter` with a `TestArg` whose name is not a Python identifier, is a keyword or starts with `_` raises `RNGValueError: Parameter argument '_x' starts with '_'. ...`. Soft keywords such as `type` and non-ASCII identifiers work. 3.0 accepted these names, and a test could take an argument named like `_x`: rename it, in the `TestArg` and in the test.
- `TestArg`'s `rng_type` must be an `RNGType` or an object with a `generate()` method. Anything else, such as a bare lambda or a class instead of an instance (`rng_type=RNGBoolean` for `RNGBoolean()`), raises `TypeError: TestArg 'x' rng_type must be an RNGType or have a generate() method, got ...` when the `TestArg` is built, also when a `value` is given. 3.0 accepted them, and failed only when it drew a value. An object with `generate()` and no `python_type` gives `TestArg.type` `Any`.

- Dataclass (record) mode follows one rule: a test receives the row as one record when neither the test nor any fixture it uses asks for one of the strategy's argument names, and exactly one test parameter that pytest fills (no default, not `self`, `cls` or a built-in fixture) is annotated with a record type whose fields are exactly those names. Otherwise the strategy passes its arguments by name. What changes from 3.0:
  - A fixture that asks for an argument name, directly, through `usefixtures` or as an autouse fixture, keeps the strategy in named mode, so a test written for dataclass mode next to such a fixture fails collection. The signature error names the argument: `A fixture of the test asks for 'x', so the strategy passes its arguments by name: parameter 'p' (Point) receives the row as a record only when no fixture asks for an argument.`
  - `validate_signature` no longer changes the choice, and the named-mode check counts the names a fixture asks for, so a fixture that takes the arguments and builds the object no longer needs `validate_signature=False`. With `validate_signature=False`, a sole dataclass parameter whose same-named fixture asks for the arguments now gets that fixture's value: 3.0 parametrized the parameter with an instance it built and never ran the fixture. Nothing reports this change.
  - NamedTuple, TypedDict and pydantic v2 models are record types that 4.0 recognizes but does not build: a test whose record would be one fails with `parameter 'txn' is annotated with BusTxn, a pydantic model; record mode supports dataclasses (NamedTuple, TypedDict and pydantic models are not supported yet). Take the arguments as parameters or use a dataclass.` 3.0 used named mode. Their fields are `_fields`, the required and optional keys, and the `model_fields` names (not aliases). pydantic is recognized without being imported. Unions, pydantic v1 models and attrs classes are not record types.
  - Two parameters that both match fail with `parameters 'p' (Point) and 'q' (Other) are each annotated with a record type whose fields are the strategy's arguments (x, y), so it is not clear which one receives the row. ...`. 3.0 used named mode.
  - A parameter with a default (`def test_p(p: Point = None)`) is not a record parameter. 3.0 chose it, and pytest then refused to parametrize an argument with a default.
  - Dataclass-mode test IDs leave out `init=False` fields.
  - The signature error says why no parameter receives the row as a record, when a parameter's annotation cannot be resolved (such as a class defined inside a test class, under `from __future__ import annotations`), is a union, or has a default. Its missing parameters are listed in argument order.

### Removed
The APIs deprecated in 3.0, without warnings left behind:
- Factories that return an `(argnames, samples)` tuple. The tests that use such a strategy fail collection with `Strategy 'name' returned an (argnames, samples) tuple (factory: file:line:name)`, which says what to return instead: a `Parameter`, with one `TestArg` per argument and the fixed rows as `directed_vectors`. Unlike a tuple's rows, which 3.0 gave in every `--vector-mode`, directed vectors are left out under `--vector-mode=random_only` and `test`. A factory that returns anything else that is not a `Parameter` fails with `must return a Parameter, got NoneType (did the factory forget to return?)`, and `export_strategies()` reports both as errors instead of a `legacy_tuple` entry.
- `RNG.set_max_retries()`. Use `Parameter(max_retries=...)`; the draws a `predicate=` gets stay fixed at 100.
- `configure()`, which did nothing. Delete the call.
- `Strategy.set_config()`. Delete the call: the plugin uses the config of the session that collects each test.
- `TestArg(directed_values=..., test_values=..., always_include_directed=...)`, the `directed_values`, `test_values` and `has_directed_values` properties, and the `has_directed_values`, `has_test_values` and `always_include_directed` keys of `TestArg.to_dict()`. Passing them is a `TypeError`, and `TestArg("x")` without a value or an rng_type raises `ValueError: TestArg 'x' must have a value or an rng_type`. Use `Parameter(directed_vectors=..., test_vectors=..., always_include_directed=...)`. `TestArg.generate_samples(n)` stays: `[value]` for a static argument, n draws otherwise.

### Fixed
- A dict directed or test vector is a named vector (see Added). 3.0 turned it into the tuple of its keys and passed the argument names to the test as values.
- A `pytest.param(*values, marks=...)` directed or test vector is checked by the number of its values instead of the length of the `pytest.param` itself, which is always 3. Marked rows such as `pytest.param(1, 2, marks=pytest.mark.xfail)` now work in a strategy with two arguments (or one) and keep their marks, `pytest.param(1, 2)` in a strategy with three arguments fails when the `Parameter` is built, naming the vector, instead of with pytest's parametrize error, and `export_strategies()` lists the vector's values.
- A dataclass parameter next to other fixtures with `validate_signature=False` failed collection (`function uses no argument 'x'`), because 3.0 then passed the arguments by name. It now receives the row as a record.

## [3.0.0] - 2026-09-30

For the same `--rng-seed`, 3.0.0 generates the same values and test IDs as 2.0.0 for strategies that draw through the RNG types, so recorded seeds and node IDs keep working. The exception is values a strategy file draws when it is imported, which now come from a stream of the file's own (see Changed). What changes is where strategies are looked up, when strategy files run, and who owns the random state.

### Breaking changes and how to upgrade from 2.0.0
- **The plugin no longer seeds Python's global `random` module.** It draws from a `random.Random` instance of its own. A factory that calls `random.randint()` and similar directly is no longer reproduced by `--rng-seed`: call the same methods on `RNG.generator()`, or use the RNG types. Test bodies that relied on the plugin's seeding can seed `random` per test from `RNG.get_seed()` (the README shows an autouse fixture).
- **Strategy files are imported later, and only where needed.** A folder's strategy files are imported when pytest first collects a test module in that folder or below, not at session start (files above the `testpaths` entry that contains the test are not searched), and files in folders with no collected tests are not imported unless a test asks for a name that no folder above it registers. They are imported the way pytest imports a test module (following `--import-mode`) instead of as standalone modules. A strategy file whose side effects must happen at startup should be imported from `conftest.py`.
- **Strategy names are scoped by folder, and a name registered twice in one folder is an error.** The nearest folder's registration wins, like fixtures. Two different functions registering one name in the same folder now stop the run with a usage error (exit 4, or 2 under pytest-xdist) after collection, instead of a warning with the last registration winning. Rename one, or move it to a subfolder.
- **`pytest_strategy.strategy` is now the `strategy` decorator.** `from pytest_strategy.strategy import Strategy, PytestStrategiesWarning` and warning filters that name `pytest_strategy.strategy.PytestStrategiesWarning` keep working, but `import pytest_strategy.strategy as m` now gives the decorator. Import from `pytest_strategy` instead.
- **Very large exhaustive runs fail at collection.** `--nsamples=auto` (and `per_sequence_samples=True`) is limited to 100,000 rows per strategy. Raise the limit with `Parameter(max_exhaustive=...)` or the `strategies_max_exhaustive` ini option.
- **Two arguments with the same name in one `Parameter`** raise `RNGValueError` when the `Parameter` is built.

### Added
- `register` and `strategy` as plain functions: `from pytest_strategy import register, strategy`, then `@register("name")` and `@strategy("name")`. `Strategy.register` and `Strategy.strategy` are the same functions. `export_strategies` is exported too.
- `@strategy(factory)`: pass the factory itself instead of its name. It needs no lookup and no registration; a registered factory gives the same values and IDs as its name.
- Folder-scoped strategy names. A test uses the registration in its own folder or the nearest folder above it. When none of them registers the name, a single registration elsewhere (a sibling folder, or an installed package outside the rootdir) is used, and several are an error that lists them.
- `pytest-strategies skill install [--claude | --agents | --all] [--global]` (also `python -m pytest_strategy`): installs an agent skill that teaches Claude Code, Codex and other agents to write strategies, into `.claude/skills/` and `.agents/skills/` of the project or the home folder.
- `RNG.generator()`: the `random.Random` instance every draw comes from, for factories that need `shuffle`, `gauss` and the like.
- Size guard: `Parameter(max_exhaustive=...)` and the `strategies_max_exhaustive` ini option (default 100,000).
- A failed run prints `pytest-strategies: reproduce with --rng-seed=S` after the failure tracebacks, before the short test summary, also under `-q`. With `-v`, the Strategy Summary lists each strategy with its tests, its directed, random and test rows, and where `nsamples` came from.
- Error messages: a misspelled strategy name gets "Did you mean ...?", and "Could not generate valid vector" says how many draws each constraint rejected, by name.
- Type hints for the whole package, checked with `mypy --strict`. `register` and `strategy` keep the decorated function's type, and the RNG types are generic (`RNGEnum(Color).generate()` is a `Color`).
- `PytestStrategiesWarning` can be imported from `pytest_strategy`.
- `--list-strategies` shows the file of each registration when a name is registered in several folders.

### Changed
- Strategies are resolved when pytest collects the test (`pytest_generate_tests`), not when the test module is imported. `@strategy` only adds a `strategy` marker. A strategy can be registered after the tests that use it, and resolution errors are collection errors in the style of pytest's own parametrize errors (`In test_x: ...`), showing the factory's frames but not the plugin's; `--full-trace` shows everything.
- `RNGValueError` is a `ValueError` subclass, so `except ValueError` catches it.
- Strategy files are recognized by `@register("...")` and aliased decorators such as `@S.register("...")`, not only `@Strategy.register`. A strategy file can import modules next to it under the default `--import-mode=prepend`, and a test module that imports it gets the same module (in a package, or when its module name is unique). A strategy file that a test module imports from another folder is loaded by the plugin first, so its import-time draws come from its own stream too.
- Values that a strategy file draws when it is imported come from a stream keyed by the file's path, so they no longer depend on which files or test modules were loaded first.
- The "Strategy 'name' not found" error names the test and lists only the strategies that test can use.
- The report header shows only the seed.

### Deprecated
These emit a `DeprecationWarning` that points at your code, keep working in 3.x, and will be removed in 4.0:
- Factories that return an `(argnames, samples)` tuple. Return a `Parameter`.
- `RNG.set_max_retries()`. Use `Parameter(max_retries=...)` with `vector_constraints`, or a predicate that accepts more values.
- `configure()`, which did nothing.
- `Strategy.set_config()`: the plugin uses the config of the session that collects each test.
- `TestArg(directed_values=..., test_values=...)`, which a `Parameter` never used. Use `Parameter(directed_vectors=..., test_vectors=...)`.

### Removed
- The private `Strategy` helpers (`_validate_signature`, `_is_dataclass_mode`, `_convert_to_dataclass`, `_generate_test_ids`, `_generate_dataclass_ids`), `_print_import_message`, and the plugin's empty session hooks.
- The demo that ran with `python -m pytest_strategy.rng`. It is now `examples/rng_example.py`.
- `run_tests.py`. CI runs pytest directly, on pytest 8.4.2 and 9, Linux and Windows, with and without pytest-xdist.

### Fixed
- Without `--rng-seed`, draws at module level in a test module came from an unseeded generator, so the printed seed did not reproduce them and pytest-xdist workers could collect different tests. They now follow the seed.
- A strategy file that a test module imported before the plugin loaded it no longer shifts the values that other strategy files draw at import time.

## [2.0.0] - 2026-09-30

2.0.0 is the first release after 1.0.0. 1.1.0a1 and 1.1.0a2 were pre-releases that were never tagged; their changes are part of it.

### Breaking changes and how to upgrade from 1.0.0
- **Python 3.11 or later and pytest 8.4.2 or later** are required. Python 3.10 and pytest 7 are no longer supported.
- **Generated values and test IDs change.** The same `--rng-seed` gives different vectors than 1.x, and test IDs are built differently, so node IDs saved for `--lf`, `--deselect` or CI filters must be recorded again.
- **Invalid RNG arguments fail when the strategy is built**, with `RNGValueError` (for example `RNGInteger(10, 0)`, or an empty `Series`). `RNGValueError` is not a `ValueError` subclass, so `except ValueError` does not catch it. Pass `skip_if_empty=` to a sequence that can be empty.
- **Command-line mistakes are usage errors** (exit code 4): a negative `--nsamples` (1.0.0 accepted it), and a `--vector-name` or `--vector-index` that no strategy has. Such a `--vector-name` used to skip every test, and such a `--vector-index` used to fail collection.
- **`RNGSequence` under `--nsamples=auto`** yields a random permutation of its values. Use `Series` where the order matters.
- **Dataclass mode** is chosen by annotation: the parameter annotated with a dataclass whose fields are the strategy's arguments receives the vector, and every other parameter is treated as a fixture.
- **License**: the project is MIT licensed.

### Added
- `PytestStrategiesWarning` (a `UserWarning` subclass, importable from `pytest_strategy.strategy`). It is emitted when two different functions register the same strategy name, and when finite mode skips a `Series` combination. With `error::pytest_strategy.strategy.PytestStrategiesWarning` in `filterwarnings`, a name registered twice in strategy files stops the run when it starts, with a usage error.
- `py.typed` marker: type checkers now use the package's annotations, which can surface new type errors in code that uses it.
- pytest-xdist support without `--rng-seed`: the controller sends its seed to the workers, so they generate the same tests.
- `Strategy.export_strategies()` reports `enum_class`, and `has_predicate` for the RNG types that store a predicate (`RNGInteger`, `RNGFloat`, `RNGEnum`, `RNGWeighted*`), in each argument's `rng_details`.
- `Series(..., skip_if_empty="<reason>")` and `RNGSequence(..., skip_if_empty="<reason>")` (keyword-only): an empty sequence, for example from a configuration without such devices, skips the strategy's tests with that reason instead of failing collection. `Parameter.skip_reason`, `to_dict()["skip_reason"]` and the argument's `rng_details["skip_if_empty"]` report it.
- `Parameter(per_sequence_samples=True)`: a finite `nsamples` gives that many random rows for each combination of the `Series`/`RNGSequence` args instead of in total, so two devices with the default 10 samples run 20 tests. `--nsamples=auto` is unaffected.
- `pytest_strategies_context(config)` hook: implement it in `conftest.py` and a strategy factory with a `ctx` parameter gets its result, for example a testbench configuration read from a file named on the command line. The hook is called at most once per session, only when a factory needs `ctx`, with a random stream of its own. When no implementation answers, `ctx` keeps its default. Factories without a `ctx` parameter are unaffected.

### Changed
- A strategy factory parameter named `ctx` receives the `pytest_strategies_context` result, or keeps its default when there is none. A factory that used `ctx` for something else, for example with no default or as its `nsamples`, needs to rename it.
- **Generated values**: each (strategy, test) pair draws from its own random stream, derived from the seed, the strategy name, the test's file path relative to the rootdir and its qualified name. The same `--rng-seed` gives different vectors than 1.1.0a2, so a seed recorded with an older version does not reproduce that run. Tests that share a strategy no longer get identical vectors, and the values no longer depend on collection order or `--import-mode`.
- **Generated values**: with a predicate, `RNGEnum` draws only among the members the predicate accepts, and `RNGWeightedInteger`/`RNGWeightedFloat` choose a new range on every retry. Seeded values with a predicate differ as a result.
- **Generated values**: under `--nsamples=auto`, the random args of a combination that the constraints reject are redrawn up to `max_retries` times before the combination is dropped.
- **Test IDs**: a value whose repr contains a memory address is shown by its type name (`codec=Codec0`), set elements are sorted, and a single-argument tuple value keeps its full ID (`pt=(1, 2)` instead of `pt=1_0`). Node IDs recorded before change accordingly.
- `--nsamples=auto`: directed vectors are included in the `all` and `mixed` modes, and `--vector-mode=test`/`directed_only`, `--vector-name` and `--vector-index` select only those vectors. A strategy without `Series`/`RNGSequence` args falls back to its own `nsamples` (or 10) instead of failing collection. A strategy whose constraints reject every combination now fails collection instead of running no tests.
- `--nsamples` is checked when the command line is parsed: anything but an integer >= 0 or `auto` (in any case) is a usage error (exit 4).
- `--vector-index` out of range for a strategy skips its tests, as `--vector-name` does, instead of failing collection. A `--vector-name` or `--vector-index` that no strategy matches is a usage error (exit 4, or 2 under pytest-xdist) instead of skipping every test.
- `Series` in finite mode skips combinations that `vector_constraints` reject, with a `PytestStrategiesWarning` when their random args were redrawn `max_retries` times. Collection fails only when a whole cycle yields no vector.
- Invalid arguments raise at construction: `RNGValueError` for `RNGInteger`/`RNGFloat` with `min > max`, infinite or NaN `RNGFloat`/`RNGWeightedFloat` bounds, a weighted range that is not a `(min, max)` tuple with `min <= max`, empty, negative, non-finite or all-zero weights or weights whose total overflows, a non-Enum, member-less or unsatisfiable `RNGEnum`, bad `RNGString` lengths or an empty charset, and a set passed to `Series`/`RNGSequence`. `ValueError` for `Parameter(nsamples=...)`, `Parameter(max_retries=...)` and a bad `n` for `generate_vectors`.
- A `TestArg` validator also checks `Series` and `RNGSequence` values.
- `PytestStrategiesWarning`s raised while generating vectors start with the strategy and the test (`Strategy 'name' (test_fn): ...`) and point at the test function instead of the plugin's code.
- `Series` and `RNGSequence` reject `None` and a predicate that is not callable with `RNGValueError`.
- A "Strategy not found" error also lists files named like strategy files that mention `register` but were not imported because they lack the literal `@Strategy.register` (an alias or a plain call), and notes that strategy files are imported when the session starts.
- The plugin makes repeated test IDs unique itself, with pytest's own suffixes, so strategies whose rows repeat (for example with `per_sequence_samples=True`) also collect under pytest 9's `strict_parametrization_ids`. IDs in normal runs are unchanged.
- Strategy factories are called exactly once, with `nsamples` passed by keyword, positionally or not at all, depending on their signature. An exception raised inside a factory is reported as `Error calling strategy factory '<name>' (nsamples=<n>): <Type>: <message>`.
- Strategy file discovery searches the `testpaths` entries (glob patterns expanded; the rootdir without `testpaths`) and the directories of command-line paths. Below them it skips hidden directories, `norecursedirs` matches (path patterns such as `tests/data` included) and virtual and conda environments, and follows symlinked directories, as pytest's collection does. It loads the files one search directory at a time, in sorted path order within each, and a file reached through two paths only once.
- Strategy files that fail to load are always reported, and a "Strategy not found" error lists the files that failed to load or skipped themselves.
- The global `random` state is started from the seed right before strategy files are loaded, so values drawn while they are imported are reproduced by the printed seed. A session that ends restores the seed, the global random state and the strategy registry it started with.
- A test module or `conftest.py` that imports a strategy file gets the module the plugin loaded, instead of executing the file again. Before, the second execution redrew its import-time values from a random state that depended on the tests collected before, silently replaced its strategies, and gave the test a second copy of its classes. A strategy file that a `conftest.py` imports before the session starts is used as it is, not loaded again.
- Dataclass mode is chosen by annotation: the parameter annotated with a dataclass whose fields are the strategy's arguments receives the vector, and every other parameter is treated as a fixture. Custom fixture names no longer need to be added to `Strategy.PYTEST_FIXTURES`.
- `Parameter` copies the `directed_vectors`, `test_vectors` and `vector_constraints` it is given.
- `--list-strategies` exits with code 2 when collection had errors.
- **License**: the project is MIT licensed. Before, the `LICENSE` file said GPL-3.0 and the package metadata said Apache-2.0; the metadata now uses the SPDX `license` field.
- **Requirements**: Python 3.11 or later (Python 3.10 is no longer supported) and pytest 8.4.2 or later. Python 3.14 is supported.
- Package metadata: the `docs` extra (Sphinx, never configured) is removed.

### Fixed
- **Python 3.14**: a test, factory or dataclass annotated with a name that is not defined yet (an import under `TYPE_CHECKING`, a class defined later) no longer fails with `NameError` when the strategy is applied.
- Strategy files in a source encoding other than UTF-8 are found (the `@Strategy.register` marker is searched in the file's bytes).
- **RNG**: weighted generators with a predicate no longer fail when only some of their ranges can satisfy it.
- **RNG**: `RNGEnum` with a predicate no longer fails while a valid member exists.
- **RNG**: `RNGFloat` with a single bound no longer draws outside that bound, and `RNG.string` rejects bad lengths and an empty charset up front.
- **RNG**: `RNGFloat` and `RNGWeightedFloat` ranges wider than the largest float (such as `RNGFloat(-sys.float_info.max, sys.float_info.max)`) no longer draw `inf`.
- **Parameters**: a `Series` strategy with `vector_constraints` no longer fails collection in finite mode, including the default run.
- **Parameters**: `add_*`/`remove_*` no longer change the dicts and lists the caller passed to `Parameter`.
- **Parameters**: a directed or test vector given as a list gives a single-argument test its element, not the whole list.
- **CLI**: an empty `--vector-name=` is a usage error, like any name that no strategy has, instead of running every vector.
- **Factories**: zero-argument factories work, also behind a `functools.wraps` decorator or `functools.cache`, and keyword-only and `**kwargs` factories work in `Strategy.export_strategies`.
- **Factories**: a `TypeError` raised inside a factory is no longer reported as a signature problem, and the factory is no longer called twice.
- **Legacy tuple strategies**: `"x,y"` argnames, generator samples and `pytest.param` samples (with their marks and ids) work.
- **Dataclass mode**: works in test classes, next to fixtures in any position, with string annotations, with `kw_only` and `init=False` fields, and with `pytest.param` samples.
- **Test IDs**: no memory addresses, set elements in a stable order (also inside dataclass and namedtuple values), and no collisions between single-argument tuple values.
- **Signature validation**: strategy arguments named like built-in fixtures (`cache`, `tmpdir`, ...) no longer fail validation.
- **Reproducibility**: an unseeded run can be reproduced with its printed seed (except random values drawn at module level in a test module; see the README).
- **Reproducibility**: `--rng-seed` in a nested in-process session (pytester) no longer leaks into the enclosing session.
- **pytest-xdist**: unseeded runs no longer abort with "Different tests were collected" (with the same exception), and `--list-strategies` no longer crashes with an INTERNALERROR.
- **Discovery**: projects below a dot directory and `testpaths` entries with `..` are searched.
- **Discovery**: strategy files inside a virtual or conda environment (such as the plugin's own `strategy.py`) are no longer imported.
- **Discovery**: a directory that cannot be entered no longer aborts the run with an INTERNALERROR (Python 3.11 to 3.13).
- **Discovery**: strategy files in symlinked directories are found.
- **Discovery**: files with a lowercase `@strategy.register` (such as a `functools.singledispatch` function named `strategy`) are no longer imported.
- **Discovery**: `pytest.skip()`, `pytest.importorskip()` or `pytest.fail()` in a strategy file no longer crash the session.
- **Discovery**: `-vv` no longer reports a strategy file outside the rootdir as failed to load.
- **Docs and examples**: the README, docs/dev.md and the package docstring no longer use the removed `generate_samples` API or a nonexistent `--seed` flag, and `examples/enum_example.py` no longer fails for some seeds.

## [1.1.0a2] - 2026-06-17 (pre-release, not tagged)

### BREAKING CHANGE
- `RNGSequence` under `nsamples="auto"` now produces a **permutation** (each element once, random order) instead of the previously ordered Cartesian product. Tests asserting exact order on `RNGSequence` auto results must be updated to use membership/count assertions, or migrated to `Series` for deterministic ordering. Note: when `Series` values are combined with constraint predicates, the constraint applies per row — if the constraint depends on Series position, behavior is tied to the cycling order.

### Added
- **Per-Strategy Sample Count**: `Parameter(nsamples=N)` lets a strategy declare its own vector count as a soft default. An explicit `--nsamples` on the CLI overrides it, `--nsamples=auto` always wins, otherwise the strategy value is used (falling back to 10).
- `Series([...])`: new deterministic ordered sequence type. In auto mode, produces values in exact declaration order. In finite mode, cycles (K >= len) or truncates to first K (K < len). Multiple `Series` args produce the full Cartesian product in `itertools.product` order (leftmost arg is slowest counter).
- `SequenceLike`: shared base class for `Series` and `RNGSequence`. Code keying on sequence arguments should use `isinstance(x, SequenceLike)`.
- `RNGSequence` and `SequenceLike` promoted to `pytest_strategy.__all__`.

## [1.1.0a1] - 2025-11-24 (pre-release, not tagged)

### Added
- **Sequence Testing**: New `RNGSequence` class for defining deterministic sequences of values.
- **Exhaustive Generation**: Support for `nsamples="auto"` to generate the Cartesian product of all sequence arguments.
- **Mixed Mode Generation**: Automatically handles mixing sequence arguments with random arguments (random values are regenerated for each sequence combination).
- **CLI Update**: Updated `--nsamples` to accept "auto" or an integer.
- **Test Values**: New `test_values` (TestArg) and `test_vectors` (Parameter) arguments for defining specific test scenarios.
- **Test Vector Mode**: New `--vector-mode=test` CLI option to execute only test vectors.

## [1.0.0] - 2025-11-23

### Added
- **New `Parameter` Class**: A robust container for test arguments that supports directed vectors, constraints, and metadata.
- **Metadata Export**: New `Strategy.export_strategies(format="json")` method to export strategy definitions for external tooling.
- **`to_dict()` Methods**: Added serialization support to `TestArg` and `Parameter` classes.
- **RNG Types**:
    - `RNGEnum`: Support for Python Enums with weighted selection and predicates.
    - `RNGInteger`, `RNGFloat`, `RNGBoolean`, `RNGString`, `RNGChoice`.
- **CLI Options**:
    - `--vector-mode`: Control generation mode (`all`, `random_only`, `directed_only`, `mixed`).
    - `--vector-name`: Filter specific directed vectors by name.
    - `--vector-index`: Filter specific vectors by index.
    - `--nsamples`: Configure the number of random samples generated.
- **Constraints**: Support for vector-level constraints (e.g., `lambda v: v[0] < v[1]`).
- **Directed Testing**: First-class support for named edge cases ("directed vectors").

### Changed
- **Breaking Change**: `Strategy` factories now return `Parameter` instances instead of tuples. (Backward compatibility for tuples is maintained but deprecated).
- **Python Support**: Dropped support for Python 3.9. Now requires Python 3.10+.
- **Type Hints**: Updated codebase to use modern Python 3.10+ type hinting syntax (`|`, `list[]`, etc.).

### Fixed
- Fixed `AttributeError: 'Parameter' object has no attribute 'generate_samples'` by renaming method to `generate_vectors`.
- Improved error handling in `Strategy` decorator.

[Unreleased]: https://github.com/guillegil/pytest-strategies/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/guillegil/pytest-strategies/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/guillegil/pytest-strategies/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/guillegil/pytest-strategies/releases/tag/v1.0.0
