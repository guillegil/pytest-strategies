import functools
import inspect
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids
from ._introspection import PYTEST_FIXTURES as _PYTEST_FIXTURES
from ._introspection import detect_dataclass_mode, validate_signature
from ._resolver import call_factory, resolve_and_parametrize
from ._runtime import runtime
from ._warnings import PytestStrategiesWarning
from .parameters import Parameter


def _factory_origin(fn: Callable[..., Any]) -> tuple[str | None, str | None, int | None]:
    """
    Identify a factory by its source file, qualified name and first line.

    The module name is deliberately not used: the plugin executes each strategies
    file under a fresh module name in every (possibly nested) pytest session.
    The file path is normalized, so one file reached through different path
    strings (``proj/../shared/x.py`` and ``shared/x.py``, a symlink) is the same.
    The first line tells apart two functions of the same name in one file.

    ``functools.wraps`` decorators and ``functools.cache`` are looked through, so
    the decorated function counts, not the decorator's wrapper. A
    ``functools.partial`` counts as the function it wraps, and a class or any
    other callable object as its class. So factories built by one function or
    class (closures, partials, instances) cannot be told apart.
    """
    fn = _unwrap(fn)
    while isinstance(fn, functools.partial):
        fn = _unwrap(fn.func)
    code = getattr(fn, "__code__", None)
    if code is not None:
        # The module's __file__ is set by the import system from the real location;
        # co_filename can be stale (pytest's rewritten pyc after a checkout moved)
        filename = getattr(fn, "__globals__", {}).get("__file__") or code.co_filename
        return (_normalize(filename), getattr(fn, "__qualname__", None), code.co_firstlineno)
    cls = fn if isinstance(fn, type) else type(fn)
    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    return (
        _normalize(getattr(module, "__file__", None)),
        cls.__qualname__,
        # Python 3.13+ records where a class statement starts
        getattr(cls, "__firstlineno__", None),
    )


def _unwrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """inspect.unwrap, or fn itself when its __wrapped__ chain loops."""
    try:
        unwrapped: Callable[..., Any] = inspect.unwrap(fn)
    except ValueError:
        return fn
    return unwrapped


def _normalize(filename: str | None) -> str | None:
    """Return the real, normalized form of a path, or None."""
    return os.path.normcase(os.path.realpath(filename)) if filename else None


def _describe_factory(fn: Callable[..., Any]) -> str:
    """Return a readable 'file:line:qualname' description of a factory for messages."""
    filename, qualname, line = _factory_origin(fn)
    where = filename or "<unknown>"
    if line is not None:
        where = f"{where}:{line}"
    return f"{where}:{qualname or repr(fn)}"


class Strategy:
    """
    Decorator-based strategy system for pytest parametrization.
    Supports both Parameter-based and legacy tuple-based strategies.
    """

    # Factories are user functions called as factory(nsamples=...) and may return
    # a Parameter or a legacy (argnames, samples) tuple — kept as Any by design.
    _registry: dict[str, Callable[..., Any]] = {}

    # Common pytest fixtures to exclude from signature validation
    PYTEST_FIXTURES: set[str] = set(_PYTEST_FIXTURES)

    @staticmethod
    def export_strategies(format: str = "json") -> str:
        """
        Export all registered strategies metadata.

        Args:
            format: Export format (currently only "json" is supported)

        Returns:
            Serialized string representation of all strategies
        """
        import json

        strategies_data = {}

        for name, factory in Strategy._registry.items():
            try:
                # Instantiate parameter with dummy count to get metadata
                # We handle both tuple-returning and Parameter-returning factories
                result = call_factory(name, factory, 1)

                if isinstance(result, Parameter):
                    strategies_data[name] = result.to_dict()
                else:
                    # Legacy tuple support (argnames, values)
                    argnames, _ = result
                    strategies_data[name] = {"type": "legacy_tuple", "argnames": argnames}
            except Exception as e:
                strategies_data[name] = {"error": f"Failed to inspect strategy: {str(e)}"}

        if format == "json":
            return json.dumps(strategies_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def set_config(config: "pytest.Config") -> None:
        runtime.config = config

    # ------------------------------------------------------------------ #
    # Private helpers — kept as thin delegators so any test that calls    #
    # them directly (via Strategy._validate_signature etc.) still works.  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_signature(test_fn, argnames: Sequence[str], strategy_name: str) -> None:
        """Thin delegator → _introspection.validate_signature."""
        validate_signature(
            test_fn,
            argnames,
            strategy_name,
            pytest_fixtures=Strategy.PYTEST_FIXTURES,
        )

    @staticmethod
    def _is_dataclass_mode(test_fn, argnames: Sequence[str]) -> tuple[bool, type | None]:
        """Thin delegator → _introspection.detect_dataclass_mode."""
        return detect_dataclass_mode(test_fn, argnames, pytest_fixtures=Strategy.PYTEST_FIXTURES)

    @staticmethod
    def _convert_to_dataclass(
        samples: Sequence[tuple], argnames: Sequence[str], dataclass_type: type
    ) -> list:
        """Thin delegator → _dataclass.convert_to_dataclass."""
        return convert_to_dataclass(samples, argnames, dataclass_type)

    @staticmethod
    def _generate_test_ids(
        argnames: Sequence[str], samples: "Sequence[Any]", max_length: int = 80
    ) -> list[str]:
        """Thin delegator → _ids.generate_test_ids."""
        return generate_test_ids(argnames, samples, max_length)

    @staticmethod
    def _generate_dataclass_ids(
        dataclass_samples: list, dc_type: type, max_length: int = 80
    ) -> list[str]:
        """Thin delegator → _ids.generate_dataclass_ids."""
        return generate_dataclass_ids(dataclass_samples, dc_type, max_length)

    @staticmethod
    def register(name: str):
        """
        Decorator to register a strategy factory function in the global registry.

        Args:
            name: Unique identifier for the strategy

        Returns:
            Decorator function that registers the factory function

        Usage:
            @Strategy.register("my_strategy")
            def create_samples(nsamples):
                return ("param_name",), [sample1, sample2, ...]
        """

        def decorate(fn: Callable[[int | str], tuple[Sequence[str], Sequence[Any]]]):
            # Warn when a different function takes over the name. Re-registering the
            # same function (e.g. a strategies file re-executed in a nested session)
            # stays silent. The last registration wins either way: it is stored
            # before warning, in case the warning is turned into an error.
            existing = Strategy._registry.get(name)
            Strategy._registry[name] = fn
            if existing is not None and _factory_origin(existing) != _factory_origin(fn):
                warnings.warn(
                    f"Strategy '{name}' is registered more than once: "
                    f"{_describe_factory(fn)} replaces {_describe_factory(existing)}",
                    PytestStrategiesWarning,
                    stacklevel=2,
                )
            return fn

        return decorate

    @staticmethod
    def strategy(name: str, validate_signature: bool = True):
        """
        Decorator to apply a registered strategy to a test function.
        Automatically parametrizes the test with generated samples.

        Supports two factory return types:
        1. Parameter instance (recommended): Enables full CLI support
        2. Tuple (argnames, samples): Backward compatibility

        Supports two test modes:
        1. Named parameters: test_fn(a, b, c)
        2. Dataclass parameter: test_fn(params: MyDataclass)

        Args:
            name: Name of the registered strategy to use
            validate_signature: Whether to validate test function signature (default: True)

        Returns:
            Decorator function that parametrizes the test

        Usage:
            # With Parameter (recommended)
            @Strategy.register("my_strategy")
            def create_strategy(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 100)),
                    TestArg("y", rng_type=RNGInteger(0, 100)),
                    directed_vectors={"origin": (0, 0)}
                )

            @Strategy.strategy("my_strategy")
            def test_function(x, y):
                # Test implementation

            # With tuple (backward compatibility)
            @Strategy.register("legacy_strategy")
            def create_samples(nsamples):
                return ("x", "y"), [(1, 2), (3, 4)]

            @Strategy.strategy("legacy_strategy")
            def test_function(x, y):
                # Test implementation
        """

        def decorate(test_fn):
            # Validate that the strategy exists in the registry
            if name not in Strategy._registry:
                available = list(Strategy._registry.keys())
                message = (
                    f"Strategy '{name}' not found. "
                    f"Available strategies: {available if available else 'none'}"
                )
                # Strategy files that failed to load are the likely cause
                if runtime.load_errors:
                    message += "\nStrategy files that failed to load:"
                    for path, error in runtime.load_errors:
                        message += f"\n  {path}: {error}"
                    # A common cause: an import that only works once pytest has
                    # collected a test module (e.g. added its directory to sys.path)
                    message += (
                        "\nStrategy files are imported when the test session starts, before "
                        "test modules are collected, so an import that only works later fails."
                    )
                # So are strategy files that skipped themselves (pytest.importorskip)
                if runtime.skipped_files:
                    message += "\nStrategy files that were skipped:"
                    for path, reason in runtime.skipped_files:
                        message += f"\n  {path}: {reason}"
                # And files named like strategy files that register another way
                # (an alias of Strategy, a plain call): discovery never imported them
                if runtime.unimported_files:
                    message += (
                        "\nFiles matching a strategy file name that were not imported because "
                        "they contain no '@Strategy.register' (use that decorator form):"
                    )
                    for path in runtime.unimported_files:
                        message += f"\n  {path}"
                raise ValueError(message)

            return resolve_and_parametrize(
                name,
                test_fn,
                registry=Strategy._registry,
                config=runtime.config,
                pytest_fixtures=Strategy.PYTEST_FIXTURES,
                validate=validate_signature,
            )

        return decorate
