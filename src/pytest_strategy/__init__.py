"""
pytest_strategies - A pytest plugin for constrained-randomized test parametrization.

This plugin enables powerful parametrized tests that combine:
- Constrained random generation
- Directed testing (edge cases, known bugs)
- Reproducibility via seed-based random generation, with row-stable streams
- CLI control for test execution

Main Components:
- register / strategy: Decorators that register a strategy factory and apply it to a test
- Parameter: Container for multiple test arguments (parameter vectors)
- TestArg: Single test argument definition with type and generation rules
- Vector: One generated row, a tuple whose fields are the argument names (v.a)
- StrategyOptions: The run's options, for a factory that declares options
- VectorInfo: The row a test item runs, item.stash[VECTOR_KEY]
- get_context: A folder's testbench context (pytest_strategies_context), for fixtures
- RNG: Random number generation with seed management
- RNGType classes: Type-safe random generators (RNGInteger, RNGFloat, etc.)

Strategies can be registered in the test module, as in the example below, in a
conftest.py, or in files named strategies.py, strategy.py, *_strategies.py or
*_strategy.py. A name is visible to the tests in that file's directory and
below, and the nearest registration wins. The plugin loads a directory's
strategy files when pytest first collects a test module there or below; files
above the testpaths entry that contains the test are not searched.

A factory receives, by name, the inputs it declares: nsamples, ctx, rng and
options. The example's factory declares none.

Example Usage:
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    @register("addition_strategy")
    def addition_strategy():
        return Parameter(
            TestArg("a", rng_type=RNGInteger(0, 100)),
            TestArg("b", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "zeros": (0, 0),
                "max": {"a": 100, "b": 100},
            },
            vector_constraints={"bounded": lambda v: v.a + v.b <= 200},
        )

    @strategy("addition_strategy")    # or @strategy(addition_strategy)
    def test_addition(a, b):
        assert a + b >= 0

Record Parameters:
    from dataclasses import dataclass

    @dataclass
    class MathParams:
        a: int
        b: int

    @strategy("addition_strategy")
    def test_addition(params: MathParams):
        assert params.a + params.b >= 0

Test IDs:
    Each row is named, the same for every seed: test_addition[directed-zeros],
    test_addition[directed-max], then test_addition[rand-0] to
    test_addition[rand-9]. So -k selects rows by name, and a node ID with the
    seed runs one row again with the values it had:
    pytest "tests/test_math.py::test_addition[rand-3]" --rng-seed=42.
    A failed row's report ends with a pytest-strategies section that shows its
    values, the seed and that command, and --lf reruns the failed rows with the
    seed they failed with.

CLI Options:
    pytest --nsamples 50              # Generate 50 random samples
    pytest --nsamples auto            # Enumerate Series/RNGSequence args
    pytest --rng-seed 42              # Set random seed for reproducibility
    pytest --vector-mode directed_only # Run only directed test vectors
    pytest --vector-name "zeros"      # Run specific directed vector
    pytest --vector-index 0           # Run the directed vector at index 0
    pytest -k zeros                   # Select the rows named zeros
    pytest -k "not rand"              # Leave out the random rows
    pytest --strategy-constraint-off=addition_strategy:bounded  # Turn a constraint off
"""

__version__ = "3.0.0"
__author__ = "Guillermo Gil"
__email__ = "guillegil@proton.me"


# Core components
# The 2.x module first: binding the strategy decorator below replaces the package
# attribute that importing it sets, while `from pytest_strategy.strategy import ...`
# keeps finding the module
from . import strategy as _strategy_module  # noqa: F401
from ._api import Strategy, export_strategies, get_context, register, strategy
from ._options import StrategyOptions
from ._vector import VECTOR_KEY, VECTORS_KEY, Vector, VectorInfo
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
    # The testbench context
    "get_context",
    # Core classes
    "Strategy",
    "Parameter",
    "TestArg",
    "StrategyOptions",
    "Vector",
    "VectorInfo",
    # Per-test metadata
    "VECTOR_KEY",
    "VECTORS_KEY",
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


def get_version() -> str:
    """Get the current version of pytest_strategies."""
    return __version__


def list_strategies() -> list[str]:
    """
    List all registered strategies.

    During a pytest session this loads every strategy file first, like
    ``--list-strategies``, so the names of folders pytest has not reached yet
    are included.

    Returns:
        list: Names of all registered strategies
    """
    from ._runtime import runtime

    runtime.load_all_strategy_files()
    return list(Strategy._registry.keys())


def get_strategy_info(name: str) -> dict[str, object]:
    """
    Get information about a registered strategy.

    Args:
        name: Strategy name

    Returns:
        dict: Strategy information

    Raises:
        ValueError: If strategy doesn't exist
    """
    from ._runtime import runtime

    runtime.load_all_strategy_files()
    if name not in Strategy._registry:
        raise ValueError(f"No strategy registered under {name!r}")

    return {
        "name": name,
        "factory": Strategy._registry[name],
        "registered": True,
    }
