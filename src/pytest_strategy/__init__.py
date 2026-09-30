"""
pytest_strategies - A pytest plugin for constrained-randomized test parametrization.

This plugin enables powerful parametrized tests that combine:
- Constrained random generation
- Directed testing (edge cases, known bugs)
- Reproducibility via seed-based random generation
- CLI control for test execution

Main Components:
- register / strategy: Decorators that register a strategy factory and apply it to a test
- Parameter: Container for multiple test arguments (parameter vectors)
- TestArg: Single test argument definition with type and generation rules
- RNG: Random number generation with seed management
- RNGType classes: Type-safe random generators (RNGInteger, RNGFloat, etc.)

Strategies can be registered in the test module, as in the example below, in a
conftest.py, or in files named strategies.py, strategy.py, *_strategies.py or
*_strategy.py. A name is visible to the tests in that file's directory and
below, and the nearest registration wins. The plugin loads a directory's
strategies files when a test there first needs a strategy.

Example Usage:
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    @register("addition_strategy")
    def addition_strategy(nsamples):
        return Parameter(
            TestArg("a", rng_type=RNGInteger(0, 100)),
            TestArg("b", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "zeros": (0, 0),
                "max": (100, 100),
            }
        )

    @strategy("addition_strategy")    # or @strategy(addition_strategy)
    def test_addition(a, b):
        assert a + b >= 0

Dataclass Support:
    from dataclasses import dataclass

    @dataclass
    class MathParams:
        a: int
        b: int

    @strategy("addition_strategy")
    def test_addition(params: MathParams):
        assert params.a + params.b >= 0

CLI Options:
    pytest --nsamples 50              # Generate 50 random samples
    pytest --nsamples auto            # Enumerate Series/RNGSequence args
    pytest --rng-seed 42              # Set random seed for reproducibility
    pytest --vector-mode directed_only # Run only directed test vectors
    pytest --vector-name "zeros"      # Run specific directed vector
    pytest --vector-index 0           # Run the directed vector at index 0
"""

__version__ = "2.0.0"
__author__ = "Guillermo Gil"
__email__ = "guillegil@proton.me"

# Core components
# The 2.x module first: binding the strategy decorator below replaces the package
# attribute that importing it sets, while `from pytest_strategy.strategy import ...`
# keeps finding the module
from . import strategy as _strategy_module  # noqa: F401
from ._api import Strategy, export_strategies, register, strategy
from ._warnings import PytestStrategiesWarning
from .parameters import Parameter

# RNG components
from .rng import (
    RNG,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGValueError,
    RNGWeightedFloat,
    RNGWeightedInteger,
    SequenceLike,
    Series,
)
from .test_args import TestArg

# Plugin is automatically loaded via entry point
# No need to import plugin module directly

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Decorators
    "register",
    "strategy",
    "export_strategies",
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
    "RNGSequence",
    "SequenceLike",
    "Series",
    # Errors and warnings
    "RNGValueError",
    "PytestStrategiesWarning",
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


def configure(
    validate_signatures: bool = True,  # noqa: ARG001 - kept for 2.x callers
    default_nsamples: int = 10,  # noqa: ARG001
) -> None:
    """Deprecated: this function never did anything, and will be removed in 4.0."""
    import warnings

    warnings.warn(
        "pytest_strategy.configure() does nothing and will be removed in 4.0",
        DeprecationWarning,
        stacklevel=2,
    )
