"""
pytest_strategies - A pytest plugin for constrained-randomized test parametrization.

This plugin enables powerful parametrized tests that combine:
- Constrained random generation
- Directed testing (edge cases, known bugs)
- Reproducibility via seed-based random generation
- CLI control for test execution

Main Components:
- Strategy: Decorator for registering and applying test strategies
- Parameter: Container for multiple test arguments (parameter vectors)
- TestArg: Single test argument definition with type and generation rules
- RNG: Random number generation with seed management
- RNGType classes: Type-safe random generators (RNGInteger, RNGFloat, etc.)

Example Usage:
    from pytest_strategies import Strategy, Parameter, TestArg
    from pytest_strategies.rng import RNGInteger

    @Strategy.register("addition_strategy")
    def create_samples(nsamples):
        param = Parameter(
            TestArg("a", rng_type=RNGInteger(0, 100)),
            TestArg("b", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "zeros": (0, 0),
                "max": (100, 100),
            }
        )
        return param.arg_names, param.generate_samples(nsamples)

    @Strategy.strategy("addition_strategy")
    def test_addition(a, b):
        assert a + b >= 0

Dataclass Support:
    from dataclasses import dataclass

    @dataclass
    class MathParams:
        a: int
        b: int

    @Strategy.strategy("addition_strategy")
    def test_addition(params: MathParams):
        assert params.a + params.b >= 0

CLI Options:
    pytest --nsamples 50              # Generate 50 random samples
    pytest --seed 42                  # Set random seed for reproducibility
    pytest --vector-mode directed_only # Run only directed test vectors
    pytest --vector-name "zeros"      # Run specific directed vector
"""

__version__ = "0.1.0"
__author__ = "Guillermo Gil"
__email__ = "guillegil@proton.me"

# Core components
from .strategy import Strategy
from .parameters import Parameter
from .test_args import TestArg

# RNG components
from .rng import (
    RNG,
    RNGValueError,
    RNGType,
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGString,
    RNGWeightedInteger,
    RNGWeightedFloat,
)

# Plugin is automatically loaded via entry point
# No need to import plugin module directly

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Core classes
    "Strategy",
    "Parameter",
    "TestArg",

    # RNG classes
    "RNG",
    "RNGType",
    "RNGInteger",
    "RNGFloat",
    "RNGBoolean",
    "RNGChoice",
    "RNGEnum",
    "RNGString",
    "RNGWeightedInteger",
    "RNGWeightedFloat",
]


# Convenience function for quick access to common fixtures
PYTEST_FIXTURES = Strategy.PYTEST_FIXTURES


def get_version():
    """Get the current version of pytest_strategies."""
    return __version__


def list_strategies():
    """
    List all registered strategies.

    Returns:
        list: Names of all registered strategies
    """
    return list(Strategy._registry.keys())


def get_strategy_info(name: str):
    """
    Get information about a registered strategy.

    Args:
        name: Strategy name

    Returns:
        dict: Strategy information

    Raises:
        ValueError: If strategy doesn't exist
    """
    if name not in Strategy._registry:
        raise ValueError(f"No strategy registered under {name!r}")

    return {
        "name": name,
        "factory": Strategy._registry[name],
        "registered": True,
    }


# Module-level configuration
def configure(
    validate_signatures: bool = True,
    default_nsamples: int = 10,
):
    """
    Configure global pytest_strategies settings.

    Args:
        validate_signatures: Enable/disable signature validation globally
        default_nsamples: Default number of samples when not specified via CLI
    """
    # This could be expanded to store global config
    # For now, it's a placeholder for future configuration options
    pass


# Print helpful message on import (optional - can be removed if too verbose)
def _print_import_message():
    """Print helpful message when module is imported (for debugging)."""
    import sys
    if '--help' not in sys.argv and '-h' not in sys.argv:
        # Only print in verbose mode or when explicitly requested
        pass


# Uncomment to enable import message
# _print_import_message()