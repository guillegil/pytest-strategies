"""
Example of Sequence Testing with pytest-strategies.

This example demonstrates how to use `RNGSequence` to test deterministic sequences
of values, and how to combine them with random generation.

To run this example with exhaustive sequence generation:
    pytest examples/sequence_example.py --nsamples=auto -v

To run with random sampling (normal mode):
    pytest examples/sequence_example.py --nsamples=5 -v
"""

from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNGFloat
from pytest_strategy.rng import RNGSequence

# 1. Basic Sequence Strategy
# This strategy iterates through a list of user roles.
@Strategy.register("user_roles")
def user_roles_strategy(nsamples):
    return Parameter(
        TestArg("role", rng_type=RNGSequence(["admin", "editor", "viewer", "guest"])),
        TestArg("active", rng_type=RNGSequence([True, False]))
    )

@Strategy.strategy("user_roles")
def test_permissions(role, active):
    """
    Test permissions for different user roles and states.
    
    With --nsamples=auto, this will run 8 tests (4 roles * 2 states).
    """
    print(f"Testing role: {role}, active: {active}")
    assert role in ["admin", "editor", "viewer", "guest"]
    assert isinstance(active, bool)


# 2. Mixed Sequence and Random Strategy
# This strategy combines a deterministic sequence (endpoints) with random data (payloads).
@Strategy.register("api_endpoints")
def api_endpoints_strategy(nsamples):
    return Parameter(
        # Deterministic: We want to test ALL these endpoints
        TestArg("endpoint", rng_type=RNGSequence(["/users", "/products", "/orders"])),
        # Random: We want random IDs and payloads for each endpoint
        TestArg("id", rng_type=RNGInteger(1, 1000)),
        TestArg("load_factor", rng_type=RNGFloat(0.0, 1.0))
    )

@Strategy.strategy("api_endpoints")
def test_api_stability(endpoint, id, load_factor):
    """
    Test API stability across endpoints with random load.
    
    With --nsamples=auto, this will run 3 tests (one for each endpoint),
    generating a fresh random id and load_factor for each one.
    """
    print(f"Testing {endpoint} with ID {id} and load {load_factor:.2f}")
    assert endpoint.startswith("/")
    assert 1 <= id <= 1000
    assert 0.0 <= load_factor <= 1.0


# 3. Single Sequence with Constraints
@Strategy.register("constrained_sequence")
def constrained_sequence_strategy(nsamples):
    return Parameter(
        TestArg("a", rng_type=RNGSequence([1, 2, 3, 4])),
        TestArg("b", rng_type=RNGSequence([1, 2, 3, 4])),
        # Only test pairs where a < b
        vector_constraints=[lambda v: v[0] < v[1]]
    )

@Strategy.strategy("constrained_sequence")
def test_pairs(a, b):
    """
    Test pairs where a < b.
    
    With --nsamples=auto, this generates all 16 combinations,
    but filters down to only those satisfying a < b (6 tests).
    """
    print(f"Testing pair: {a} < {b}")
    assert a < b
