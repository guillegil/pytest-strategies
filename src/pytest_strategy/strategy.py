import inspect
import pytest
from dataclasses import is_dataclass, fields
from typing import Callable, Sequence, Any, Tuple

from .rng import RNG
from .parameters import Parameter
from .test_args import TestArg


class Strategy:
    """
    Decorator-based strategy system for pytest parametrization.
    Supports both Parameter-based and legacy tuple-based strategies.
    """

    _registry: dict[str, Callable[[int | str], Tuple[Sequence[str], Sequence[Any]]]] = {}

    # Global placeholder for the pytest Config object
    # This will be set during pytest_configure hook to access CLI options
    _pytest_config = None

    # Common pytest fixtures to exclude from signature validation
    PYTEST_FIXTURES = {
        'request', 'tmp_path', 'tmp_path_factory', 'tmpdir', 'tmpdir_factory',
        'capsys', 'capfd', 'caplog', 'monkeypatch', 'pytestconfig',
        'cache', 'doctest_namespace', 'recwarn', 'record_property',
        'record_testsuite_property', 'record_xml_attribute'
    }

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
                    strategies_data[name] = {
                        "type": "legacy_tuple",
                        "argnames": argnames
                    }
            except Exception as e:
                strategies_data[name] = {
                    "error": f"Failed to inspect strategy: {str(e)}"
                }
                
        if format == "json":
            return json.dumps(strategies_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


    @staticmethod
    def set_config(config: dict):
        Strategy._pytest_config = config

    @staticmethod
    def _validate_signature(test_fn, argnames: Sequence[str], strategy_name: str) -> None:
        """
        Validate that test function signature matches strategy argnames.
        
        Automatically excludes pytest fixtures from validation by checking if
        parameters have fixture markers or are in the known fixtures list.

        Args:
            test_fn: The test function to validate
            argnames: Expected argument names from strategy
            strategy_name: Name of the strategy (for error messages)

        Raises:
            ValueError: If signature doesn't match
        """
        sig = inspect.signature(test_fn)
        test_params = list(sig.parameters.keys())

        # Remove pytest fixtures from comparison
        # A parameter is considered a fixture if:
        # 1. It's in the common pytest fixtures list, OR
        # 2. It's not in the strategy argnames (assumed to be a custom fixture)
        actual_params = []
        for p in test_params:
            if p in Strategy.PYTEST_FIXTURES:
                continue  # Skip known pytest fixtures
            # If parameter is not in strategy argnames, assume it's a custom fixture
            if p not in argnames:
                continue  # Skip custom fixtures
            actual_params.append(p)
        
        expected_params = list(argnames)

        # Check for mismatch
        if set(expected_params) != set(actual_params):
            missing = set(expected_params) - set(actual_params)
            extra = set(actual_params) - set(expected_params)

            error_msg = f"Test function signature mismatch for strategy '{strategy_name}'!\n"
            error_msg += f"  Strategy provides: {expected_params}\n"
            error_msg += f"  Test function expects: {actual_params}\n"

            if missing:
                error_msg += f"  Missing parameters: {list(missing)}\n"
            if extra:
                error_msg += f"  Extra parameters: {list(extra)}\n"

            raise ValueError(error_msg)

    @staticmethod
    def _is_dataclass_mode(test_fn, argnames: Sequence[str]) -> Tuple[bool, type | None]:
        """
        Detect if test function expects a single dataclass parameter.

        Returns:
            Tuple of (is_dataclass_mode, dataclass_type)
        """
        sig = inspect.signature(test_fn)
        test_params = list(sig.parameters.keys())

        # Remove fixtures
        actual_params = [p for p in test_params if p not in Strategy.PYTEST_FIXTURES]

        # Check if single parameter and multiple argnames (dataclass mode)
        if len(actual_params) == 1 and len(argnames) > 1:
            # Check if parameter has dataclass type hint
            param = sig.parameters[actual_params[0]]
            if param.annotation != inspect.Parameter.empty:
                if is_dataclass(param.annotation):
                    return True, param.annotation

        return False, None

    @staticmethod
    def _convert_to_dataclass(samples: Sequence[tuple], argnames: Sequence[str], dataclass_type: type) -> list:
        """
        Convert tuple samples to dataclass instances.

        Args:
            samples: List of tuples
            argnames: Argument names from strategy
            dataclass_type: The dataclass type to instantiate

        Returns:
            List of dataclass instances

        Raises:
            ValueError: If dataclass fields don't match argnames
        """
        # Get dataclass field names
        dc_fields = {f.name for f in fields(dataclass_type)}
        strategy_fields = set(argnames)

        # Validate fields match
        if dc_fields != strategy_fields:
            missing = strategy_fields - dc_fields
            extra = dc_fields - strategy_fields

            error_msg = f"Dataclass fields don't match strategy parameters!\n"
            error_msg += f"  Strategy provides: {list(argnames)}\n"
            error_msg += f"  Dataclass expects: {list(dc_fields)}\n"

            if missing:
                error_msg += f"  Missing in dataclass: {list(missing)}\n"
            if extra:
                error_msg += f"  Extra in dataclass: {list(extra)}\n"

            raise ValueError(error_msg)

        # Get field order from dataclass
        dc_field_names = [f.name for f in fields(dataclass_type)]
        
        # Convert samples to dataclass instances
        dataclass_samples = []
        for sample in samples:
            # Create dict mapping argnames to values
            value_dict = dict(zip(argnames, sample))
            # Reorder values to match dataclass field order
            ordered_values = [value_dict[name] for name in dc_field_names]
            # Create dataclass instance with ordered values
            dataclass_samples.append(dataclass_type(*ordered_values))

        return dataclass_samples

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
        def decorate(fn: Callable[[int | str], Tuple[Sequence[str], Sequence[Any]]]):
            # Store the factory function in the global registry
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
                            filter_by_index=vector_index
                        )
                except KeyError as e:
                    # If filtering by name/index and vector doesn't exist, return empty samples
                    # This allows CLI filtering to work gracefully across multiple strategies
                    if vector_name or vector_index is not None:
                        # Return empty list - pytest will skip this test
                        samples = []
                    else:
                        raise ValueError(
                            f"Error generating samples for strategy '{name}': {e}"
                        ) from e
                except Exception as e:
                    raise ValueError(
                        f"Error generating samples for strategy '{name}': {e}"
                    ) from e

                # Get argument names from Parameter
                argnames = param.arg_names

            else:
                # LEGACY MODE: Tuple-based strategy (backward compatibility)
                # Validate that factory returns the expected tuple format
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        f"Strategy '{name}' must return either a Parameter instance "
                        f"or a tuple (argnames, samples), got {type(result).__name__}"
                    )

                # Unpack the result into parameter names and sample values
                argnames, samples = result

                # Convert single string argname to tuple for consistency
                if isinstance(argnames, str):
                    argnames = (argnames,)

            # Detect dataclass mode
            is_dc_mode, dc_type = Strategy._is_dataclass_mode(test_fn, argnames)

            if is_dc_mode:
                # DATACLASS MODE: Convert samples to dataclass instances
                try:
                    dataclass_samples = Strategy._convert_to_dataclass(samples, argnames, dc_type)
                except Exception as e:
                    raise ValueError(
                        f"Error converting samples to dataclass for strategy '{name}': {e}"
                    ) from e

                # Get the single parameter name
                sig = inspect.signature(test_fn)
                test_params = [p for p in sig.parameters.keys() if p not in Strategy.PYTEST_FIXTURES]
                param_name = test_params[0]

                # Generate test IDs for dataclass mode
                ids = Strategy._generate_dataclass_ids(dataclass_samples, dc_type)

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
                ids = Strategy._generate_test_ids(argnames, samples)

                # Apply pytest parametrize decorator to the test function
                # This will create multiple test instances, one for each sample
                return pytest.mark.parametrize(argstr, samples, ids=ids)(test_fn)

        return decorate

    @staticmethod
    def _generate_test_ids(argnames: Sequence[str], samples: Sequence[Any], max_length: int = 80) -> list[str]:
        """
        Generate concise test IDs from argument names and sample values.

        Args:
            argnames: Sequence of argument names
            samples: Sequence of sample values (tuples or single values)
            max_length: Maximum length for test ID (default: 80)

        Returns:
            List of test ID strings
        """
        ids = []

        for sample in samples:
            if len(argnames) == 1:
                # Single parameter: format as "param_name=value"
                value = sample if not isinstance(sample, tuple) else sample[0]
                val_str = repr(value)
                if len(val_str) > max_length - len(argnames[0]) - 1:
                    val_str = val_str[:max_length - len(argnames[0]) - 4] + "..."
                ids.append(f"{argnames[0]}={val_str}")
            else:
                # Multiple parameters: format as "param1=value1,param2=value2"
                parts = []
                for arg_name, value in zip(argnames, sample):
                    val_str = repr(value)
                    # Limit individual value length
                    if len(val_str) > 20:
                        val_str = val_str[:17] + "..."
                    parts.append(f"{arg_name}={val_str}")

                full_id = ",".join(parts)
                # Truncate if too long
                if len(full_id) > max_length:
                    full_id = full_id[:max_length - 3] + "..."
                ids.append(full_id)

        return ids

    @staticmethod
    def _generate_dataclass_ids(dataclass_samples: list, dc_type: type, max_length: int = 80) -> list[str]:
        """
        Generate test IDs for dataclass mode.

        Args:
            dataclass_samples: List of dataclass instances
            dc_type: Dataclass type
            max_length: Maximum length for test ID

        Returns:
            List of test ID strings
        """
        ids = []
        for dc_instance in dataclass_samples:
            # Create readable ID from dataclass fields
            field_strs = []
            for f in fields(dc_type):
                val = getattr(dc_instance, f.name)
                val_str = repr(val)
                if len(val_str) > 20:
                    val_str = val_str[:17] + "..."
                field_strs.append(f"{f.name}={val_str}")

            full_id = ",".join(field_strs)
            if len(full_id) > max_length:
                full_id = full_id[:max_length - 3] + "..."
            ids.append(full_id)

        return ids