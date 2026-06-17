from collections.abc import Sequence
from typing import Any, Callable

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids
from ._introspection import PYTEST_FIXTURES as _PYTEST_FIXTURES
from ._introspection import detect_dataclass_mode, validate_signature
from ._resolver import resolve_and_parametrize
from ._runtime import runtime
from .parameters import Parameter


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
                result = factory(1)

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
            Strategy._registry[name] = fn
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
                raise ValueError(
                    f"Strategy '{name}' not found. "
                    f"Available strategies: {available if available else 'none'}"
                )

            return resolve_and_parametrize(
                name,
                test_fn,
                registry=Strategy._registry,
                config=runtime.config,
                pytest_fixtures=Strategy.PYTEST_FIXTURES,
                validate=validate_signature,
            )

        return decorate
