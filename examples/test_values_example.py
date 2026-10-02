"""
Example of Test Values feature with pytest-strategies.

This example demonstrates how to use `test_vectors` to define test-specific
vectors that only run when using --vector-mode=test.

To run this example with test vectors only:
    pytest examples/test_values_example.py --vector-mode=test -v

To run with all vectors (directed + random):
    pytest examples/test_values_example.py --nsamples=5 -v

To turn the date range constraint off for one run, by its name:
    pytest examples/test_values_example.py --strategy-constraint-off=date_range_test:ordered -v
"""

import pytest

from pytest_strategy import Parameter, RNGChoice, RNGInteger, RNGSequence, TestArg, register, strategy

# 1. Basic Test Vectors
# This strategy defines specific test cases that should always be verified.
@register("api_test_cases")
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

@strategy("api_test_cases")
def test_api_responses(endpoint, status_code):
    """
    Test API responses.
    
    With --vector-mode=test, this will run 4 tests (the test vectors).
    With default mode (--vector-mode=all), this runs only random samples (this
    strategy has no directed vectors); test vectors are not included.
    """
    print(f"Testing {endpoint} with status {status_code}")
    assert endpoint.startswith("/")
    assert 200 <= status_code <= 599


# 2. Test Vectors with Directed Vectors
# You can combine test vectors with directed vectors for different testing modes.
@register("user_validation")
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

@strategy("user_validation")
def test_user_permissions(age, role):
    """
    Test user permissions.
    
    With --vector-mode=test, runs only test vectors (3 tests).
    With --vector-mode=directed_only, runs only directed vectors (2 tests).
    With --vector-mode=all (default), runs directed + random samples (test vectors
    are not included).
    """
    print(f"Testing user: age={age}, role={role}")
    assert 0 <= age <= 120
    assert role in ["admin", "user", "guest"]


# 3. Test Vectors Next to Directed Vectors
# Directed vectors run in the default mode, test vectors only with
# --vector-mode=test.
@register("payment_test")
def payment_test_strategy(nsamples):
    return Parameter(
        TestArg("amount", rng_type=RNGInteger(1, 10000)),
        TestArg("currency", rng_type=RNGChoice(["USD", "EUR", "GBP"])),
        directed_vectors={
            "smallest": (1, "USD"),
        },
        # Test vectors: the scenarios that run with --vector-mode=test
        test_vectors={
            "zero_amount": (0, "USD"),
            "large_amount": (9999, "EUR")
        }
    )

@strategy("payment_test")
def test_payment_processing(amount, currency):
    """
    Test payment processing.
    
    With --vector-mode=test, runs only test vectors (2 tests).
    """
    print(f"Testing payment: {amount} {currency}")
    assert amount >= 0
    assert currency in ["USD", "EUR", "GBP"]


# 4. Complex Test Scenario
# Combine test vectors with constraints for complex testing scenarios. The
# constraint has a name, so a run can turn it off to also draw ranges that end
# before they start, which the code under test must reject.
@register("date_range_test")
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
        # A named constraint: start <= end, reading the row's arguments by name
        vector_constraints={"ordered": lambda v: v.start_day <= v.end_day}
    )

def days_in_range(start_day, end_day):
    """The code under test: the number of days from start_day to end_day."""
    if end_day < start_day:
        raise ValueError(f"The range ends on day {end_day}, before day {start_day}")
    return end_day - start_day + 1

@strategy("date_range_test")
def test_date_ranges(start_day, end_day):
    """
    Test date range validation.
    
    With --vector-mode=test, runs only test vectors (5 tests).
    All test vectors satisfy the constraint start_day <= end_day.
    With --strategy-constraint-off=date_range_test:ordered, the random ranges
    can also end before they start, and days_in_range must reject those.
    """
    print(f"Testing date range: {start_day} to {end_day}")
    assert 1 <= start_day <= 31
    assert 1 <= end_day <= 31
    if start_day <= end_day:
        assert 1 <= days_in_range(start_day, end_day) <= 31
    else:
        # Only drawn when the "ordered" constraint is turned off
        with pytest.raises(ValueError):
            days_in_range(start_day, end_day)
