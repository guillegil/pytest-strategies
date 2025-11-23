# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
