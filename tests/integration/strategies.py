"""
Strategy definitions for integration tests.

This file demonstrates the complete pytest-strategies workflow:
- Strategy registration with @Strategy.register
- Parameter creation with TestArg instances
- RNG configuration for random value generation
- Directed test vectors for edge cases
- Constraints for valid parameter combinations
"""

from typing import Tuple, Sequence, Any
from pytest_strategy import (
    Strategy,
    Parameter,
    TestArg,
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGString,
    RNGWeightedInteger,
)


# ============================================================================
# BASIC STRATEGIES
# ============================================================================

@Strategy.register("simple_integer_strategy")
def create_simple_integer_strategy(nsamples: int) -> Parameter:
    """Simple strategy with single integer parameter."""
    return Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
        directed_vectors={
            "zero": (0,),
            "fifty": (50,),
            "max": (100,),
        }
    )


@Strategy.register("multi_param_strategy")
def create_multi_param_strategy(nsamples: int) -> Parameter:
    """Strategy with multiple parameters of different types."""
    return Parameter(
        TestArg("count", rng_type=RNGInteger(1, 100)),
        TestArg("ratio", rng_type=RNGFloat(0.0, 1.0)),
        TestArg("enabled", rng_type=RNGBoolean(0.5)),
        directed_vectors={
            "baseline": (10, 0.5, True),
            "disabled": (1, 0.0, False),
            "max": (100, 1.0, True),
        }
    )


# ============================================================================
# STRATEGIES WITH CONSTRAINTS
# ============================================================================

@Strategy.register("constrained_range_strategy")
def create_constrained_range_strategy(nsamples: int) -> Parameter:
    """Strategy with constraints ensuring min < max."""
    return Parameter(
        TestArg("min_val", rng_type=RNGInteger(0, 50)),
        TestArg("max_val", rng_type=RNGInteger(50, 100)),
        vector_constraints=[
            lambda v: v[0] < v[1],  # min < max
        ],
        directed_vectors={
            "full_range": (0, 100),
            "narrow": (49, 51),
            "mid_range": (25, 75),
        }
    )


@Strategy.register("complex_constraint_strategy")
def create_complex_constraint_strategy(nsamples: int) -> Parameter:
    """Strategy with multiple constraints."""
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)),
        TestArg("y", rng_type=RNGInteger(0, 100, predicate=lambda y: y % 5 == 0)),
        vector_constraints=[
            lambda v: v[0] + v[1] <= 100,  # sum <= 100
            lambda v: v[0] >= v[1] // 2,   # x >= y/2
        ],
        directed_vectors={
            "zeros": (0, 0),
            "balanced": (50, 50),
        }
    )


# ============================================================================
# REAL-WORLD SIMULATION STRATEGIES
# ============================================================================

@Strategy.register("api_request_strategy")
def create_api_request_strategy(nsamples: int) -> Parameter:
    """Strategy simulating API request parameters."""
    return Parameter(
        TestArg("user_id", rng_type=RNGInteger(1, 10000)),
        TestArg("page_size", rng_type=RNGWeightedInteger(
            ranges={
                (10, 50): 0.7,    # Common page sizes
                (100, 500): 0.3,  # Large pages
            }
        )),
        TestArg("timeout", rng_type=RNGFloat(0.1, 5.0)),
        TestArg("method", rng_type=RNGChoice(["GET", "POST", "PUT", "DELETE"])),
        TestArg("include_metadata", rng_type=RNGBoolean(0.3)),
        vector_constraints=[
            lambda v: v[1] <= 500,  # page_size <= 500
        ],
        directed_vectors={
            "admin_request": (1, 50, 1.0, "GET", True),
            "user_request": (5000, 25, 2.0, "POST", False),
            "bulk_request": (100, 500, 5.0, "GET", True),
        }
    )


@Strategy.register("database_query_strategy")
def create_database_query_strategy(nsamples: int) -> Parameter:
    """Strategy simulating database query parameters."""
    return Parameter(
        TestArg("offset", rng_type=RNGInteger(0, 1000)),
        TestArg("limit", rng_type=RNGInteger(1, 100)),
        TestArg("sort_field", rng_type=RNGChoice(["id", "name", "created_at", "updated_at"])),
        TestArg("sort_order", rng_type=RNGChoice(["asc", "desc"])),
        vector_constraints=[
            lambda v: v[0] + v[1] <= 1000,  # offset + limit <= 1000
        ],
        directed_vectors={
            "first_page": (0, 10, "id", "asc"),
            "last_page": (990, 10, "id", "desc"),
            "large_page": (0, 100, "created_at", "desc"),
        }
    )


# ============================================================================
# VALIDATION STRATEGIES
# ============================================================================

@Strategy.register("validated_strategy")
def create_validated_strategy(nsamples: int) -> Parameter:
    """Strategy with validators on TestArgs."""
    return Parameter(
        TestArg(
            "positive",
            rng_type=RNGInteger(1, 100, predicate=lambda x: x > 0),
            validator=lambda x: x > 0
        ),
        TestArg(
            "even",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0),
            validator=lambda x: x % 2 == 0
        ),
        TestArg(
            "in_range",
            rng_type=RNGFloat(0.0, 1.0),
            validator=lambda x: 0.0 <= x <= 1.0
        ),
        directed_vectors={
            "valid_case": (10, 20, 0.5),
            "edge_case": (1, 0, 0.0),
        }
    )


# ============================================================================
# STRING GENERATION STRATEGIES
# ============================================================================

@Strategy.register("string_strategy")
def create_string_strategy(nsamples: int) -> Parameter:
    """Strategy with string generation."""
    return Parameter(
        TestArg("username", rng_type=RNGString(min_length=5, max_length=15)),
        TestArg("code", rng_type=RNGString(length=6, charset="0123456789ABCDEF")),
        TestArg("category", rng_type=RNGChoice(["admin", "user", "guest"])),
        directed_vectors={
            "admin_user": ("admin_user", "ABC123", "admin"),
            "guest_user": ("guest", "000000", "guest"),
        }
    )


# ============================================================================
# MIXED MODE STRATEGIES
# ============================================================================

@Strategy.register("mixed_static_random_strategy")
def create_mixed_static_random_strategy(nsamples: int) -> Parameter:
    """Strategy mixing static values, directed values, and random generation."""
    return Parameter(
        TestArg("static_val", value=42),  # Static value
        TestArg("random_val", rng_type=RNGInteger(0, 100)),  # Random
        TestArg("directed_val", rng_type=RNGChoice([1, 2, 3])),  # Use RNGChoice for directed-like behavior
        TestArg("mixed_val", 
                rng_type=RNGInteger(0, 100),
                directed_values=[0, 50, 100],
                always_include_directed=True),  # Mixed
        directed_vectors={
            "all_zeros": (42, 0, 1, 0),
            "all_max": (42, 100, 3, 100),
        }
    )


# ============================================================================
# LEGACY TUPLE-BASED STRATEGY (BACKWARD COMPATIBILITY)
# ============================================================================

@Strategy.register("legacy_tuple_strategy")
def create_legacy_tuple_strategy(nsamples: int) -> Tuple[Tuple[str, ...], Sequence[Tuple[int, ...]]]:
    """Legacy tuple-based strategy for backward compatibility testing."""
    argnames = ("x", "y", "z")
    samples = [
        (i, i * 2, i * 3)
        for i in range(nsamples)
    ]
    return argnames, samples
