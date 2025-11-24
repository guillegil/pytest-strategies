"""
Example of Test Values feature with pytest-strategies.

This example demonstrates how to use `test_values` and `test_vectors` to define
test-specific values that only run when using --vector-mode=test.

To run this example with test vectors only:
    pytest examples/test_values_example.py --vector-mode=test -v

To run with all vectors (directed + random):
    pytest examples/test_values_example.py --nsamples=5 -v
"""

from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNGChoice
from pytest_strategy.rng import RNGSequence

# 1. Basic Test Vectors
# This strategy defines specific test cases that should always be verified.
@Strategy.register("api_test_cases")
def api_test_cases_strategy(nsamples):
    return Parameter(
        TestArg("endpoint", rng_type=RNGChoice(["/users", "/products", "/orders"])),
        TestArg("status_code", rng_type=RNGInteger(200, 500)),
        # Test vectors: specific scenarios we want to test
        test_vectors={
            "success_case": ("/users", 200),
            "not_found": ("/products", 404),
            "server_error": ("/orders", 500),
            "unauthorized": ("/users", 401)
        }
    )

@Strategy.strategy("api_test_cases")
def test_api_responses(endpoint, status_code):
    """
    Test API responses.
    
    With --vector-mode=test, this will run 4 tests (the test vectors).
    With default mode, this will run test vectors + directed + random samples.
    """
    print(f"Testing {endpoint} with status {status_code}")
    assert endpoint.startswith("/")
    assert 200 <= status_code <= 599


# 2. Test Vectors with Directed Vectors
# You can combine test vectors with directed vectors for different testing modes.
@Strategy.register("user_validation")
def user_validation_strategy(nsamples):
    return Parameter(
        TestArg("age", rng_type=RNGInteger(0, 120)),
        TestArg("role", rng_type=RNGChoice(["admin", "user", "guest"])),
        # Directed vectors: edge cases to always include
        directed_vectors={
            "newborn": (0, "guest"),
            "senior": (100, "user")
        },
        # Test vectors: specific test scenarios
        test_vectors={
            "admin_test": (30, "admin"),
            "guest_test": (25, "guest"),
            "boundary_test": (18, "user")
        }
    )

@Strategy.strategy("user_validation")
def test_user_permissions(age, role):
    """
    Test user permissions.
    
    With --vector-mode=test, runs only test vectors (3 tests).
    With --vector-mode=directed_only, runs only directed vectors (2 tests).
    With --vector-mode=all, runs directed + test + random samples.
    """
    print(f"Testing user: age={age}, role={role}")
    assert 0 <= age <= 120
    assert role in ["admin", "user", "guest"]


# 3. Test Values in TestArg
# You can also define test values at the TestArg level.
@Strategy.register("payment_test")
def payment_test_strategy(nsamples):
    return Parameter(
        # Test values: specific amounts to test
        TestArg("amount", rng_type=RNGInteger(1, 10000), test_values=[0, 1, 100, 9999]),
        TestArg("currency", rng_type=RNGChoice(["USD", "EUR", "GBP"]), test_values=["USD"]),
        # Test vectors combine the test values
        test_vectors={
            "zero_amount": (0, "USD"),
            "large_amount": (9999, "EUR")
        }
    )

@Strategy.strategy("payment_test")
def test_payment_processing(amount, currency):
    """
    Test payment processing.
    
    With --vector-mode=test, runs only test vectors (2 tests).
    """
    print(f"Testing payment: {amount} {currency}")
    assert amount >= 0
    assert currency in ["USD", "EUR", "GBP"]


# 4. Complex Test Scenario
# Combine test vectors with constraints for complex testing scenarios.
@Strategy.register("date_range_test")
def date_range_test_strategy(nsamples):
    return Parameter(
        TestArg("start_day", rng_type=RNGInteger(1, 31)),
        TestArg("end_day", rng_type=RNGInteger(1, 31)),
        # Test vectors: specific date ranges to verify
        test_vectors={
            "same_day": (15, 15),
            "one_week": (1, 7),
            "month_start": (1, 1),
            "month_end": (31, 31),
            "full_month": (1, 31)
        },
        # Constraint: start <= end
        vector_constraints=[lambda v: v[0] <= v[1]]
    )

@Strategy.strategy("date_range_test")
def test_date_ranges(start_day, end_day):
    """
    Test date range validation.
    
    With --vector-mode=test, runs only test vectors (5 tests).
    All test vectors satisfy the constraint start_day <= end_day.
    """
    print(f"Testing date range: {start_day} to {end_day}")
    assert 1 <= start_day <= 31
    assert 1 <= end_day <= 31
    assert start_day <= end_day
