# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
