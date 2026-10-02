"""
Example of Parameter-based strategies.

A strategy is a factory that returns a Parameter. Register it with @register
and apply it to tests with @strategy, by name or by passing the factory itself.
All CLI options (--nsamples, --vector-mode, --vector-name, --vector-index)
apply to it.
"""

from pytest_strategy import (
    Parameter,
    RNGChoice,
    RNGFloat,
    RNGInteger,
    TestArg,
    register,
    strategy,
)


# ============================================================================
# EXAMPLE 1: Basic Parameter-Based Strategy
# ============================================================================

@register("addition_strategy")
def create_addition_strategy(nsamples):
    """
    Strategy using Parameter class (NEW RECOMMENDED WAY).
    
    This enables:
    - CLI options: --vector-mode, --vector-name, --vector-index
    - Directed test vectors
    - Constraints
    - All Parameter features
    """
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        directed_vectors={
            "zeros": (0, 0),
            "ones": (1, 1),
            "max": (100, 100),
        }
    )


@strategy("addition_strategy")
def test_addition(a, b):
    """Test addition with Parameter-based strategy."""
    result = a + b
    assert result >= 0
    assert result == a + b


# ============================================================================
# EXAMPLE 2: Strategy with Constraints
# ============================================================================

@register("range_strategy")
def create_range_strategy(nsamples):
    """Strategy with constraints ensuring min < max."""
    return Parameter(
        TestArg("min_val", rng_type=RNGInteger(0, 50)),
        TestArg("max_val", rng_type=RNGInteger(50, 100)),
        vector_constraints=[
            lambda v: v.min_val < v.max_val  # The row is a Vector: v.min_val is v[0]
        ],
        directed_vectors={
            "edge_case": (0, 100),
            "narrow": (49, 51),
        }
    )


@strategy("range_strategy")
def test_range_validation(min_val, max_val):
    """Test range validation with constraints."""
    assert min_val < max_val
    assert 0 <= min_val <= 50
    assert 50 <= max_val <= 100


# ============================================================================
# EXAMPLE 3: Complex Strategy with Multiple Types
# ============================================================================

@register("api_endpoint_strategy")
def create_api_test_strategy(nsamples):
    """Strategy simulating API endpoint testing."""
    return Parameter(
        TestArg("user_id", rng_type=RNGInteger(1, 10000)),
        TestArg("timeout", rng_type=RNGFloat(0.1, 5.0)),
        TestArg("method", rng_type=RNGChoice(["GET", "POST", "PUT", "DELETE"])),
        directed_vectors={
            "admin_user": (1, 1.0, "GET"),
            "regular_user": (5000, 2.0, "POST"),
            "slow_request": (100, 5.0, "GET"),
        }
    )


@strategy("api_endpoint_strategy")
def test_api_endpoint(user_id, timeout, method):
    """Test API endpoint with various parameters."""
    assert 1 <= user_id <= 10000
    assert 0.1 <= timeout <= 5.0
    assert method in ["GET", "POST", "PUT", "DELETE"]


# ============================================================================
# EXAMPLE 4: Passing the Factory Instead of a Name
# ============================================================================

def create_doubling_strategy(nsamples):
    """
    A factory that is not registered: the test passes the function itself.

    No name is looked up, so the test cannot pick up another folder's strategy
    by mistake. A registered factory can be passed the same way.
    """
    return Parameter(
        TestArg("x", rng_type=RNGInteger(-1000, 1000)),
        directed_vectors={"zero": (0,)},
    )


@strategy(create_doubling_strategy)
def test_doubling(x):
    """Test with a strategy passed by reference."""
    assert 2 * x == x + x


# ============================================================================
# CLI USAGE EXAMPLES
# ============================================================================

"""
Run tests with CLI options:

# Use all directed vectors + random samples (default)
pytest examples/strategy_example.py --nsamples=20

# Use only directed vectors
pytest examples/strategy_example.py --vector-mode=directed_only

# Use only random samples
pytest examples/strategy_example.py --vector-mode=random_only

# Run specific directed vector by name
pytest examples/strategy_example.py --vector-name=zeros

# Run specific directed vector by index
pytest examples/strategy_example.py --vector-index=0

# Set RNG seed for reproducibility
pytest examples/strategy_example.py --rng-seed=42

# Combine options
pytest examples/strategy_example.py --nsamples=50 --vector-mode=all --rng-seed=42
"""


if __name__ == "__main__":
    print("This is an example file demonstrating Parameter-based strategies.")
    print("Run with pytest to see the strategies in action.")
    print()
    print("Example commands:")
    print("  pytest examples/strategy_example.py -v")
    print("  pytest examples/strategy_example.py --vector-mode=directed_only -v")
    print("  pytest examples/strategy_example.py --vector-name=zeros -v")
