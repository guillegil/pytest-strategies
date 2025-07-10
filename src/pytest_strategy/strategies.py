
from typing import Callable, Sequence, Any, Tuple
import pytest

from .rng import RNG


class Strategy:

    # Global registry to store strategy factory functions
    # Key: strategy name (string), Value: factory function that takes sample count and returns (argnames, samples)
    _registry: dict[str, Callable[[int], Tuple[Sequence[str], Sequence[Any]]]] = {}

    # Global placeholder for the pytest Config object
    # This will be set during pytest_configure hook to access CLI options
    _pytest_config = None


    @staticmethod
    def set_config(config: dict):
        Strategy._pytest_config = config

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
    def strategy(name: str):
        """
        Decorator to apply a registered strategy to a test function.
        Automatically parametrizes the test with generated samples.
        
        Args:
            name: Name of the registered strategy to use
            
        Returns:
            Decorator function that parametrizes the test
            
        Usage:
            @Strategy.strategy("my_strategy")
            def test_function(param_name):
                # Test implementation
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