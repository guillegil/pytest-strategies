import inspect
from collections.abc import Sequence
from typing import Any, Callable

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids
from ._introspection import PYTEST_FIXTURES as _PYTEST_FIXTURES
from ._introspection import detect_dataclass_mode, validate_signature
from .parameters import Parameter
from .rng import RNG


class Strategy:
    """
    Decorator-based strategy system for pytest parametrization.
    Supports both Parameter-based and legacy tuple-based strategies.
    """

    # Factories are user functions called as factory(nsamples=...) and may return
    # a Parameter or a legacy (argnames, samples) tuple — kept as Any by design.
    _registry: dict[str, Callable[..., Any]] = {}

    # Global placeholder for the pytest Config object
    # This will be set during pytest_configure hook to access CLI options
    _pytest_config: "pytest.Config | None" = None

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
        Strategy._pytest_config = config

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

            # Get CLI options
            cfg = Strategy._pytest_config
            nsamples = cfg.getoption("nsamples") if cfg else 10
            vector_mode = cfg.getoption("vector_mode") if cfg else "all"
            vector_name = cfg.getoption("vector_name") if cfg else None
            vector_index = cfg.getoption("vector_index") if cfg else None

            # Get the factory function for this strategy
            factory = Strategy._registry[name]

            # Refresh the random number generator seed
            RNG.refresh_seed()

            # Call factory function with keyword argument
            try:
                result = factory(nsamples=nsamples)
            except TypeError as e:
                # Try positional for backward compatibility
                try:
                    result = factory(nsamples)
                except Exception as inner_e:
                    raise ValueError(
                        f"Error calling strategy factory '{name}': {e}. "
                        f"Factory should accept 'nsamples' parameter."
                    ) from inner_e

            # Detect if result is a Parameter instance or tuple
            if isinstance(result, Parameter):
                # NEW MODE: Parameter-based strategy
                param = result

                # Generate samples using Parameter's generate_vectors with CLI options
                try:
                    if nsamples == "auto":
                        samples = param.generate_exhaustive()
                    else:
                        samples = param.generate_vectors(
                            n=int(nsamples),
                            mode=vector_mode,
                            filter_by_name=vector_name,
                            filter_by_index=vector_index,
                        )
                except KeyError as e:
                    # If filtering by name/index and vector doesn't exist, return empty samples
                    # This allows CLI filtering to work gracefully across multiple strategies
                    if vector_name or vector_index is not None:
                        samples = []
                    else:
                        raise ValueError(
                            f"Error generating samples for strategy '{name}': {e}"
                        ) from e
                except Exception as e:
                    raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e

                # Get argument names from Parameter
                argnames = param.arg_names

            else:
                # LEGACY MODE: Tuple-based strategy (backward compatibility)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        f"Strategy '{name}' must return either a Parameter instance "
                        f"or a tuple (argnames, samples), got {type(result).__name__}"
                    )

                argnames, samples = result

                # Convert single string argname to tuple for consistency
                if isinstance(argnames, str):
                    argnames = (argnames,)

            # Detect dataclass mode
            is_dc_mode, dc_type = detect_dataclass_mode(
                test_fn, argnames, pytest_fixtures=Strategy.PYTEST_FIXTURES
            )

            if is_dc_mode:
                # DATACLASS MODE: Convert samples to dataclass instances
                assert dc_type is not None  # guaranteed when is_dc_mode is True
                try:
                    dataclass_samples = convert_to_dataclass(samples, argnames, dc_type)
                except Exception as e:
                    raise ValueError(
                        f"Error converting samples to dataclass for strategy '{name}': {e}"
                    ) from e

                # Get the single parameter name
                sig = inspect.signature(test_fn)
                test_params = [p for p in sig.parameters if p not in Strategy.PYTEST_FIXTURES]
                param_name = test_params[0]

                # Generate test IDs for dataclass mode
                ids = generate_dataclass_ids(dataclass_samples, dc_type)

                # Apply pytest parametrize with single dataclass parameter
                return pytest.mark.parametrize(param_name, dataclass_samples, ids=ids)(test_fn)

            else:
                # NAMED PARAMETERS MODE: Standard behavior

                # Validate signature if requested
                if validate_signature:
                    try:
                        Strategy._validate_signature(test_fn, argnames, name)
                    except ValueError as e:
                        raise ValueError(
                            f"Signature validation failed for strategy '{name}': {e}"
                        ) from e

                # Create comma-separated string of parameter names for pytest.mark.parametrize
                argstr = ",".join(argnames)

                # For single parameters, unwrap the tuples
                if len(argnames) == 1:
                    samples = [s[0] if isinstance(s, tuple) else s for s in samples]

                # Generate test IDs for better test output readability
                ids = generate_test_ids(argnames, samples)

                # Apply pytest parametrize decorator to the test function
                return pytest.mark.parametrize(argstr, samples, ids=ids)(test_fn)

        return decorate
