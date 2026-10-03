"""
Example of Sequence Testing with pytest-strategies.

This example demonstrates how to use `RNGSequence` to test every value of a
sequence, and how to combine it with random generation. With --nsamples=auto
each RNGSequence is walked in a random order, and the RNGSequence args of one
strategy form a Cartesian product: every combination of their values that
passes the vector constraints runs once. A single RNGSequence arg therefore
uses each value exactly once, while with two or more args a value appears once
for each combination of the other args' values. Other args get a fresh random
value for each combination. Use `Series` instead when the order must be fixed.

With a finite --nsamples, RNGSequence args are drawn at random, like RNGChoice.

To run this example with exhaustive sequence generation:
    pytest examples/sequence_example.py --nsamples=auto -v

To run with random sampling (normal mode):
    pytest examples/sequence_example.py --nsamples=5 -v

Under --nsamples=auto each row's test ID shows the values it enumerates, the
same for every seed: test_pairs[a=1-b=2], test_permissions[role=admin-active=True].
-k cannot contain "=", so run one such row by its node ID:
    pytest "examples/sequence_example.py::test_pairs[a=1-b=2]" --nsamples=auto -v

With a finite --nsamples the RNGSequence values are drawn, and the rows are
named test_pairs[rand-0] to test_pairs[rand-4]:
    pytest examples/sequence_example.py --nsamples=5 -k "test_pairs" -v
"""

from pytest_strategy import Parameter, RNGFloat, RNGInteger, RNGSequence, TestArg, register, strategy

# 1. Basic Sequence Strategy
# This strategy iterates through a list of user roles.
@register("user_roles")
def user_roles_strategy():
    return Parameter(
        TestArg("role", rng_type=RNGSequence(["admin", "editor", "viewer", "guest"])),
        TestArg("active", rng_type=RNGSequence([True, False]))
    )

@strategy("user_roles")
def test_permissions(role, active):
    """
    Test permissions for different user roles and states.
    
    With --nsamples=auto, this will run 8 tests (4 roles * 2 states).
    """
    print(f"Testing role: {role}, active: {active}")
    assert role in ["admin", "editor", "viewer", "guest"]
    assert isinstance(active, bool)


# 2. Mixed Sequence and Random Strategy
# This strategy combines a sequence (endpoints) with random data (payloads).
@register("api_endpoints")
def api_endpoints_strategy():
    return Parameter(
        # Exhaustive: We want to test ALL these endpoints
        TestArg("endpoint", rng_type=RNGSequence(["/users", "/products", "/orders"])),
        # Random: We want random IDs and payloads for each endpoint
        TestArg("id", rng_type=RNGInteger(1, 1000)),
        TestArg("load_factor", rng_type=RNGFloat(0.0, 1.0))
    )

@strategy("api_endpoints")
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


# 3. Sequences with Constraints
@register("constrained_sequence")
def constrained_sequence_strategy():
    return Parameter(
        TestArg("a", rng_type=RNGSequence([1, 2, 3, 4])),
        TestArg("b", rng_type=RNGSequence([1, 2, 3, 4])),
        # Only test pairs where a < b
        vector_constraints={"a_below_b": lambda v: v.a < v.b}
    )

@strategy("constrained_sequence")
def test_pairs(a, b):
    """
    Test pairs where a < b.
    
    With --nsamples=auto, this generates all 16 combinations,
    but filters down to only those satisfying a < b (6 tests).
    """
    print(f"Testing pair: {a} < {b}")
    assert a < b


# 4. Filtered Sequence (Predicate Support)
@register("filtered_sequence")
def filtered_sequence_strategy():
    return Parameter(
        # Use a predicate to filter the sequence during initialization
        # Here we take range(20) but keep only multiples of 3
        TestArg("val", rng_type=RNGSequence(range(20), predicate=lambda x: x % 3 == 0)),
        TestArg("label", rng_type=RNGSequence(["A", "B"]))
    )

@strategy("filtered_sequence")
def test_filtered(val, label):
    """
    Test with a filtered sequence.
    
    The sequence range(20) filtered by (x % 3 == 0) yields:
    [0, 3, 6, 9, 12, 15, 18] (7 items)
    
    With --nsamples=auto, this generates 7 * 2 = 14 tests.
    """
    print(f"Testing filtered value: {val} with label {label}")
    assert val % 3 == 0
    assert val < 20
    assert label in ["A", "B"]
