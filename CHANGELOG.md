# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PytestStrategiesWarning` (a `UserWarning` subclass, importable from `pytest_strategy.strategy`). It is emitted when two different functions register the same strategy name, and when finite mode skips a `Series` combination.
- `py.typed` marker: type checkers now use the package's annotations, which can surface new type errors in code that uses it.
- pytest-xdist support without `--rng-seed`: the controller sends its seed to the workers, so they generate the same tests.
- `Strategy.export_strategies()` reports `enum_class` and `has_predicate` in each argument's `rng_details`.
- `Series(..., skip_if_empty="<reason>")` and `RNGSequence(..., skip_if_empty="<reason>")`: an empty sequence, for example from a configuration without such devices, skips the strategy's tests with that reason instead of failing collection. `Parameter.skip_reason` and `to_dict()["skip_reason"]` report it.
- `Parameter(per_sequence_samples=True)`: a finite `nsamples` gives that many random rows for each combination of the `Series`/`RNGSequence` args instead of in total, so two devices with the default 10 samples run 20 tests. `--nsamples=auto` is unaffected.

### Changed
- **Generated values**: each (strategy, test) pair draws from its own random stream, derived from the seed, the strategy name, the test's file path relative to the rootdir and its qualified name. The same `--rng-seed` gives different vectors than 1.1.0a2, so a seed recorded with an older version does not reproduce that run. Tests that share a strategy no longer get identical vectors, and the values no longer depend on collection order or `--import-mode`.
- **Generated values**: with a predicate, `RNGEnum` draws only among the members the predicate accepts, and `RNGWeightedInteger`/`RNGWeightedFloat` choose a new range on every retry. Seeded values with a predicate differ as a result.
- **Generated values**: under `--nsamples=auto`, the random args of a combination that the constraints reject are redrawn up to `max_retries` times before the combination is dropped.
- **Test IDs**: a value whose repr contains a memory address is shown by its type name (`codec=Codec0`), set elements are sorted, and a single-argument tuple value keeps its full ID (`pt=(1, 2)` instead of `pt=1_0`). Node IDs recorded before change accordingly.
- `--nsamples=auto`: directed vectors are included in the `all` and `mixed` modes, and `--vector-mode=test`/`directed_only`, `--vector-name` and `--vector-index` select only those vectors. A strategy without `Series`/`RNGSequence` args falls back to its own `nsamples` (or 10) instead of failing collection. A strategy whose constraints reject every combination now fails collection instead of running no tests.
- `--nsamples` is checked when the command line is parsed: anything but an integer >= 0 or `auto` (in any case) is a usage error (exit 4).
- `--vector-index` out of range for a strategy skips its tests, as `--vector-name` does, instead of failing collection. A `--vector-name` or `--vector-index` that no strategy matches is a usage error (exit 4, or 2 under pytest-xdist) instead of skipping every test.
- `Series` in finite mode skips combinations that `vector_constraints` reject, with a `PytestStrategiesWarning` when their random args were redrawn `max_retries` times. Collection fails only when a whole cycle yields no vector.
- Invalid arguments raise at construction: `RNGValueError` for `RNGInteger`/`RNGFloat` with `min > max`, empty, negative, non-finite or all-zero weights, a non-Enum, member-less or unsatisfiable `RNGEnum`, bad `RNGString` lengths or an empty charset, and a set passed to `Series`/`RNGSequence`. `ValueError` for `Parameter(nsamples=...)`, `Parameter(max_retries=...)` and a bad `n` for `generate_vectors`.
- A `TestArg` validator also checks `Series` and `RNGSequence` values.
- Strategy factories are called exactly once, with `nsamples` passed by keyword, positionally or not at all, depending on their signature. An exception raised inside a factory is reported as `Error calling strategy factory '<name>' (nsamples=<n>): <Type>: <message>`.
- Strategy file discovery searches the `testpaths` entries (glob patterns expanded; the rootdir without `testpaths`) and the directories of command-line paths. Below them it skips hidden directories, `norecursedirs` matches and virtual environments, and it loads the files in sorted path order.
- Strategy files that fail to load are always reported, and a "Strategy not found" error lists the files that failed to load or skipped themselves.
- The global `random` state is started from the seed right before strategy files are loaded, so values drawn while they are imported are reproduced by the printed seed. A session that ends restores the seed, the global random state and the strategy registry it started with.
- Dataclass mode is chosen by annotation: the parameter annotated with a dataclass whose fields are the strategy's arguments receives the vector, and every other parameter is treated as a fixture. Custom fixture names no longer need to be added to `Strategy.PYTEST_FIXTURES`.
- `Parameter` copies the `directed_vectors`, `test_vectors` and `vector_constraints` it is given.
- `--list-strategies` exits with code 2 when collection had errors.
- Package metadata: Python 3.13 added and 3.9 removed from the classifiers; development status is Alpha.

### Fixed
- **RNG**: weighted generators with a predicate no longer fail when only some of their ranges can satisfy it.
- **RNG**: `RNGEnum` with a predicate no longer fails while a valid member exists.
- **RNG**: `RNGFloat` with a single bound no longer draws outside that bound, and `RNG.string` rejects bad lengths and an empty charset up front.
- **Parameters**: a `Series` strategy with `vector_constraints` no longer fails collection in finite mode, including the default run.
- **Parameters**: `add_*`/`remove_*` no longer change the dicts and lists the caller passed to `Parameter`.
- **Factories**: zero-argument factories work, and keyword-only and `**kwargs` factories work in `Strategy.export_strategies`.
- **Factories**: a `TypeError` raised inside a factory is no longer reported as a signature problem, and the factory is no longer called twice.
- **Legacy tuple strategies**: `"x,y"` argnames, generator samples and `pytest.param` samples (with their marks and ids) work.
- **Dataclass mode**: works in test classes, next to fixtures in any position, with string annotations, with `kw_only` and `init=False` fields, and with `pytest.param` samples.
- **Test IDs**: no memory addresses, set elements in a stable order, and no collisions between single-argument tuple values.
- **Signature validation**: strategy arguments named like built-in fixtures (`cache`, `tmpdir`, ...) no longer fail validation.
- **Reproducibility**: an unseeded run can be reproduced with its printed seed.
- **Reproducibility**: `--rng-seed` in a nested in-process session (pytester) no longer leaks into the enclosing session.
- **pytest-xdist**: unseeded runs no longer abort with "Different tests were collected", and `--list-strategies` no longer crashes with an INTERNALERROR.
- **Discovery**: projects below a dot directory and `testpaths` entries with `..` are searched.
- **Discovery**: strategy files inside a virtual environment (such as the plugin's own `strategy.py`) are no longer imported.
- **Discovery**: `pytest.skip()`, `pytest.importorskip()` or `pytest.fail()` in a strategy file no longer crash the session.
- **Discovery**: `-vv` no longer reports a strategy file outside the rootdir as failed to load.
- **Docs and examples**: the README, docs/dev.md and the package docstring no longer use the removed `generate_samples` API or a nonexistent `--seed` flag, and `examples/enum_example.py` no longer fails for some seeds.

## [1.1.0a2] - 2026-06-17

### BREAKING CHANGE
- `RNGSequence` under `nsamples="auto"` now produces a **permutation** (each element once, random order) instead of the previously ordered Cartesian product. Tests asserting exact order on `RNGSequence` auto results must be updated to use membership/count assertions, or migrated to `Series` for deterministic ordering. Note: when `Series` values are combined with constraint predicates, the constraint applies per row — if the constraint depends on Series position, behavior is tied to the cycling order.

### Added
- **Per-Strategy Sample Count**: `Parameter(nsamples=N)` lets a strategy declare its own vector count as a soft default. An explicit `--nsamples` on the CLI overrides it, `--nsamples=auto` always wins, otherwise the strategy value is used (falling back to 10).
- `Series([...])`: new deterministic ordered sequence type. In auto mode, produces values in exact declaration order. In finite mode, cycles (K >= len) or truncates to first K (K < len). Multiple `Series` args produce the full Cartesian product in `itertools.product` order (leftmost arg is slowest counter).
- `SequenceLike`: shared base class for `Series` and `RNGSequence`. Code keying on sequence arguments should use `isinstance(x, SequenceLike)`.
- `RNGSequence` and `SequenceLike` promoted to `pytest_strategy.__all__`.

## [1.1.0a1] - 2025-11-24

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
