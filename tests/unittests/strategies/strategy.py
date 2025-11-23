"""
Strategy definitions for unit tests.

This file contains all strategy registrations used by the test suite.
Each strategy defines parameter generation rules for specific test scenarios.
"""

from pytest_strategies import Strategy, Parameter, TestArg
from pytest_strategies.rng import (
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGString,
    RNGWeightedInteger,
)


# ============================================================================
# SIGNATURE VALIDATION STRATEGIES
# ============================================================================

@Strategy.register("correct_sig_strategy")
def create_correct_sig_samples(nsamples):
    """Strategy with two integer parameters for signature validation tests."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("missing_param_strategy")
def create_missing_param_samples(nsamples):
    """Strategy with three parameters to test missing parameter detection."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        TestArg("c", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("extra_param_strategy")
def create_extra_param_samples(nsamples):
    """Strategy with one parameter to test extra parameter detection."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("fixture_strategy")
def create_fixture_samples(nsamples):
    """Strategy for testing with pytest fixtures."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("no_validation_strategy")
def create_no_validation_samples(nsamples):
    """Strategy for testing disabled validation."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


# ============================================================================
# DATACLASS MODE STRATEGIES
# ============================================================================

@Strategy.register("basic_dataclass_strategy")
def create_basic_dataclass_samples(nsamples):
    """Basic strategy for dataclass mode with two integer parameters."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("complex_dataclass_strategy")
def create_complex_dataclass_samples(nsamples):
    """Complex strategy with multiple types for dataclass mode."""
    param = Parameter(
        TestArg("count", rng_type=RNGInteger(1, 100)),
        TestArg("timeout", rng_type=RNGFloat(0.1, 10.0)),
        TestArg("mode", rng_type=RNGChoice(["fast", "slow"])),
        TestArg("enabled", rng_type=RNGBoolean()),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("mismatch_strategy")
def create_mismatch_samples(nsamples):
    """Strategy for testing dataclass field mismatch detection."""
    param = Parameter(
        TestArg("x", rng_type=RNGInteger(0, 100)),
        TestArg("y", rng_type=RNGInteger(0, 100)),
        # Missing 'z' that dataclass expects
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("directed_dataclass_strategy")
def create_directed_dataclass_samples(nsamples):
    """Strategy with directed vectors for dataclass mode."""
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        TestArg("operation", rng_type=RNGChoice(["+", "-", "*"])),
        directed_vectors={
            "zeros": (0, 0, "+"),
            "max": (100, 100, "*"),
            "subtract": (50, 25, "-"),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


# ============================================================================
# MIXED MODE STRATEGIES
# ============================================================================

@Strategy.register("mixed_strategy")
def create_mixed_samples(nsamples):
    """Strategy for testing mixed mode with fixtures."""
    param = Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("config_strategy")
def create_config_samples(nsamples):
    """Strategy for server configuration testing."""
    param = Parameter(
        TestArg("host", rng_type=RNGChoice(["localhost", "127.0.0.1"])),
        TestArg("port", rng_type=RNGInteger(1024, 65535)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


# ============================================================================
# EDGE CASE STRATEGIES
# ============================================================================

@Strategy.register("single_param_strategy")
def create_single_param_samples(nsamples):
    """Strategy with single parameter (not dataclass mode)."""
    param = Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


@Strategy.register("single_field_strategy")
def create_single_field_samples(nsamples):
    """Strategy with single field for dataclass edge case."""
    param = Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
    )
    return param.arg_names, param.generate_samples(nsamples, mode="random_only")


# ============================================================================
# INTEGRATION TEST STRATEGIES
# ============================================================================

@Strategy.register("server_test_strategy")
def create_server_test_samples(nsamples):
    """
    Real-world server configuration strategy.

    Tests various server configurations with directed vectors for
    common scenarios (default, secure, minimal).
    """
    param = Parameter(
        TestArg("host", rng_type=RNGChoice(["localhost", "127.0.0.1", "0.0.0.0"])),
        TestArg("port", rng_type=RNGInteger(1024, 65535)),
        TestArg("timeout", rng_type=RNGFloat(0.1, 30.0)),
        TestArg("retries", rng_type=RNGInteger(1, 10)),
        TestArg("ssl_enabled", rng_type=RNGBoolean()),
        directed_vectors={
            "default": ("localhost", 8080, 5.0, 3, False),
            "secure": ("0.0.0.0", 443, 10.0, 5, True),
            "minimal": ("127.0.0.1", 1024, 0.1, 1, False),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


@Strategy.register("addition_strategy")
def create_addition_samples(nsamples):
    """
    Simple addition strategy for basic examples.

    Generates two integers with edge cases for zero and maximum values.
    """
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        directed_vectors={
            "zeros": (0, 0),
            "max": (100, 100),
            "one_zero": (0, 50),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


@Strategy.register("math_strategy")
def create_math_samples(nsamples):
    """
    Mathematical operations strategy.

    Tests various math operations with different operands.
    """
    param = Parameter(
        TestArg("a", rng_type=RNGInteger(0, 100)),
        TestArg("b", rng_type=RNGInteger(0, 100)),
        TestArg("operation", rng_type=RNGChoice(["+", "-", "*", "/"])),
        directed_vectors={
            "add_zeros": (0, 0, "+"),
            "divide_by_one": (100, 1, "/"),
            "multiply_by_zero": (50, 0, "*"),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


# ============================================================================
# ADVANCED STRATEGIES
# ============================================================================

@Strategy.register("weighted_port_strategy")
def create_weighted_port_samples(nsamples):
    """
    Weighted port number strategy.

    Generates port numbers with realistic distribution:
    - 90% user ports (1024-49151)
    - 10% dynamic ports (49152-65535)
    """
    param = Parameter(
        TestArg(
            "port",
            rng_type=RNGWeightedInteger(
                ranges={
                    (1024, 49151): 0.9,   # User ports (90%)
                    (49152, 65535): 0.1   # Dynamic ports (10%)
                }
            )
        ),
        directed_vectors={
            "http": (80,),
            "https": (443,),
            "custom": (8080,),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


@Strategy.register("string_test_strategy")
def create_string_test_samples(nsamples):
    """
    String generation strategy.

    Tests various string scenarios including empty, short, and long strings.
    """
    param = Parameter(
        TestArg("text", rng_type=RNGString(min_length=1, max_length=50)),
        TestArg("length", rng_type=RNGInteger(1, 50)),
        directed_vectors={
            "empty": ("", 0),
            "single": ("a", 1),
            "long": ("a" * 100, 100),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


@Strategy.register("constrained_range_strategy")
def create_constrained_range_samples(nsamples):
    """
    Constrained range strategy.

    Ensures min < max constraint on generated ranges.
    """
    param = Parameter(
        TestArg("min_val", rng_type=RNGInteger(0, 100)),
        TestArg("max_val", rng_type=RNGInteger(0, 100)),
        vector_constraints=[
            lambda v: v[0] < v[1],  # min < max
        ],
        directed_vectors={
            "zero_to_hundred": (0, 100),
            "narrow": (45, 55),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


@Strategy.register("api_request_strategy")
def create_api_request_samples(nsamples):
    """
    API request strategy.

    Simulates various API request scenarios with different methods,
    status codes, and response times.
    """
    param = Parameter(
        TestArg("method", rng_type=RNGChoice(["GET", "POST", "PUT", "DELETE"])),
        TestArg("status_code", rng_type=RNGChoice([200, 201, 400, 404, 500])),
        TestArg("response_time", rng_type=RNGFloat(0.01, 5.0)),
        TestArg("retry_count", rng_type=RNGInteger(0, 3)),
        directed_vectors={
            "success": ("GET", 200, 0.1, 0),
            "not_found": ("GET", 404, 0.05, 1),
            "server_error": ("POST", 500, 2.0, 3),
            "created": ("POST", 201, 0.5, 0),
        }
    )
    return param.arg_names, param.generate_samples(nsamples, mode="all")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_all_strategies():
    """List all registered strategies in this module."""
    return [
        # Signature validation
        "correct_sig_strategy",
        "missing_param_strategy",
        "extra_param_strategy",
        "fixture_strategy",
        "no_validation_strategy",

        # Dataclass mode
        "basic_dataclass_strategy",
        "complex_dataclass_strategy",
        "mismatch_strategy",
        "directed_dataclass_strategy",

        # Mixed mode
        "mixed_strategy",
        "config_strategy",

        # Edge cases
        "single_param_strategy",
        "single_field_strategy",

        # Integration
        "server_test_strategy",
        "addition_strategy",
        "math_strategy",

        # Advanced
        "weighted_port_strategy",
        "string_test_strategy",
        "constrained_range_strategy",
        "api_request_strategy",
    ]


def get_strategy_description(name: str) -> str:
    """Get description of a strategy by name."""
    descriptions = {
        "correct_sig_strategy": "Two integer parameters for signature validation",
        "missing_param_strategy": "Three parameters to test missing parameter detection",
        "extra_param_strategy": "One parameter to test extra parameter detection",
        "fixture_strategy": "Testing with pytest fixtures",
        "no_validation_strategy": "Testing disabled validation",

        "basic_dataclass_strategy": "Basic dataclass with two integers",
        "complex_dataclass_strategy": "Complex dataclass with multiple types",
        "mismatch_strategy": "Dataclass field mismatch detection",
        "directed_dataclass_strategy": "Dataclass with directed vectors",

        "mixed_strategy": "Mixed mode with fixtures",
        "config_strategy": "Server configuration testing",

        "single_param_strategy": "Single parameter (not dataclass)",
        "single_field_strategy": "Single field dataclass",

        "server_test_strategy": "Real-world server configuration",
        "addition_strategy": "Simple addition with edge cases",
        "math_strategy": "Mathematical operations",

        "weighted_port_strategy": "Weighted port number distribution",
        "string_test_strategy": "String generation scenarios",
        "constrained_range_strategy": "Constrained min/max ranges",
        "api_request_strategy": "API request simulation",
    }
    return descriptions.get(name, "No description available")


if __name__ == "__main__":
    print("=== Registered Test Strategies ===\n")

    for strategy_name in list_all_strategies():
        description = get_strategy_description(strategy_name)
        print(f"✓ {strategy_name}")
        print(f"  {description}\n")

    print(f"Total strategies: {len(list_all_strategies())}")