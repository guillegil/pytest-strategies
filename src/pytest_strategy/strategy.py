from typing import Callable, Sequence, Any, Tuple
import pytest
import inspect
import functools
from dataclasses import is_dataclass, fields

from .rng import RNG


class Strategy:

    # Global registry to store strategy factory functions
    # Key: strategy name (string), Value: factory function that takes sample count and returns (argnames, samples)
    _registry: dict[str, Callable[[int], Tuple[Sequence[str], Sequence[Any]]]] = {}

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
    def set_config(config: dict):
        Strategy._pytest_config = config

    @staticmethod
    def _validate_signature(test_fn, argnames: Sequence[str], strategy_name: str) -> None:
        """
        Validate that test function signature matches strategy argnames.

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
        actual_params = [p for p in test_params if p not in Strategy.PYTEST_FIXTURES]
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

        # Convert samples to dataclass instances
        # Need to create dict with correct field order
        dataclass_samples = []
        for sample in samples:
            # Create dict mapping argnames to values
            kwargs = dict(zip(argnames, sample))
            dataclass_samples.append(dataclass_type(**kwargs))

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
        def decorate(fn: Callable[[int], Tuple[Sequence[str], Sequence[Any]]]):
            # Store the factory function in the global registry
            Strategy._registry[name] = fn
            return fn
        return decorate

    @staticmethod
    def strategy(name: str, validate_signature: bool = True):
        """
        Decorator to apply a registered strategy to a test function.
        Automatically parametrizes the test with generated samples.

        Supports two modes:
        1. Named parameters: test_fn(a, b, c)
        2. Dataclass parameter: test_fn(params: MyDataclass)

        Args:
            name: Name of the registered strategy to use
            validate_signature: Whether to validate test function signature (default: True)

        Returns:
            Decorator function that parametrizes the test

        Usage:
            # Named parameters
            @Strategy.strategy("my_strategy")
            def test_function(param1, param2):
                # Test implementation

            # Dataclass parameter
            @Strategy.strategy("my_strategy")
            def test_function(params: MyDataclass):
                # Test implementation using params.param1, params.param2
        """
        def decorate(test_fn):
            # Validate that the strategy exists in the registry
            if name not in Strategy._registry:
                raise ValueError(f"No strategy registered under {name!r}")

            # Retrieve sample count from CLI options (default to 10 if no config available)
            cfg = Strategy._pytest_config
            ns = cfg.getoption("nsamples") if cfg else 10

            # Get the factory function for this strategy
            factory = Strategy._registry[name]

            # Refresh the random number generator seed
            RNG.refresh_seed()

            # Try to call factory with positional argument first
            # If that fails (TypeError), try with keyword argument 'nsamples'
            try:
                result = factory(ns)
            except TypeError:
                result = factory(nsamples=ns)

            # Validate that factory returns the expected tuple format
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    f"Strategy {name!r} must return a tuple (argnames, samples), got {result!r}"
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
                dataclass_samples = Strategy._convert_to_dataclass(samples, argnames, dc_type)

                # Get the single parameter name
                sig = inspect.signature(test_fn)
                test_params = [p for p in sig.parameters.keys() if p not in Strategy.PYTEST_FIXTURES]
                param_name = test_params[0]

                # Generate test IDs for dataclass mode
                ids = []
                for i, dc_instance in enumerate(dataclass_samples):
                    # Create readable ID from dataclass fields
                    field_strs = [f"{f.name}={getattr(dc_instance, f.name)!r}" for f in fields(dc_type)]
                    ids.append(",".join(field_strs))

                # Apply pytest parametrize with single dataclass parameter
                return pytest.mark.parametrize(param_name, dataclass_samples, ids=ids)(test_fn)

            else:
                # NAMED PARAMETERS MODE: Standard behavior

                # Validate signature if requested
                if validate_signature:
                    Strategy._validate_signature(test_fn, argnames, name)

                # Create comma-separated string of parameter names for pytest.mark.parametrize
                argstr = ",".join(argnames)

                # Generate test IDs for better test output readability
                if len(argnames) == 1:
                    # Single parameter: format as "param_name=value"
                    ids = [f"{argnames[0]}={v!r}" for v in samples]
                else:
                    # Multiple parameters: format as "param1=value1,param2=value2"
                    ids = [
                        ",".join(f"{n}={v!r}" for n, v in zip(argnames, vals))
                        for vals in samples
                    ]

                # Apply pytest parametrize decorator to the test function
                # This will create multiple test instances, one for each sample
                return pytest.mark.parametrize(argstr, samples, ids=ids)(test_fn)

        return decorate