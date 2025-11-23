"""
Comprehensive integration tests for pytest-strategies.

These tests demonstrate the complete workflow integrating:
- Strategy registration and application
- Parameter creation and configuration
- TestArg with various RNG types
- Directed test vectors
- Constraints and validation
- CLI options support
- Backward compatibility

The strategies are defined in strategies.py and auto-discovered by the plugin.
"""

import pytest
from typing import Tuple
from pytest_strategy import Strategy, RNG


# ============================================================================
# BASIC STRATEGY TESTS
# ============================================================================

@Strategy.strategy("simple_integer_strategy")
def test_simple_integer(value):
    """Test simple integer strategy."""
    assert isinstance(value, int)
    assert 0 <= value <= 100


@Strategy.strategy("multi_param_strategy")
def test_multi_param(count, ratio, enabled):
    """Test strategy with multiple parameters."""
    assert isinstance(count, int)
    assert 1 <= count <= 100
    assert isinstance(ratio, float)
    assert 0.0 <= ratio <= 1.0
    assert isinstance(enabled, bool)


# ============================================================================
# CONSTRAINT TESTS
# ============================================================================

@Strategy.strategy("constrained_range_strategy")
def test_constrained_range(min_val, max_val):
    """Test that constraints are enforced."""
    assert isinstance(min_val, int)
    assert isinstance(max_val, int)
    assert min_val < max_val  # Constraint
    assert 0 <= min_val <= 50
    assert 50 <= max_val <= 100


@Strategy.strategy("complex_constraint_strategy")
def test_complex_constraints(x, y):
    """Test multiple constraints."""
    # RNG predicates
    assert x % 2 == 0  # x is even
    assert y % 5 == 0  # y is multiple of 5
    
    # Parameter constraints
    assert x + y <= 100  # sum <= 100
    assert x >= y // 2   # x >= y/2


# ============================================================================
# REAL-WORLD SIMULATION TESTS
# ============================================================================

@Strategy.strategy("api_request_strategy")
def test_api_request(user_id, page_size, timeout, method, include_metadata):
    """Test API request parameter generation."""
    assert 1 <= user_id <= 10000
    assert 10 <= page_size <= 500
    assert 0.1 <= timeout <= 5.0
    assert method in ["GET", "POST", "PUT", "DELETE"]
    assert isinstance(include_metadata, bool)


@Strategy.strategy("database_query_strategy")
def test_database_query(offset, limit, sort_field, sort_order):
    """Test database query parameter generation."""
    assert 0 <= offset <= 1000
    assert 1 <= limit <= 100
    assert offset + limit <= 1000  # Constraint
    assert sort_field in ["id", "name", "created_at", "updated_at"]
    assert sort_order in ["asc", "desc"]


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@Strategy.strategy("validated_strategy")
def test_validated_params(positive, even, in_range):
    """Test that validators are enforced."""
    assert positive > 0
    assert even % 2 == 0
    assert 0.0 <= in_range <= 1.0


# ============================================================================
# STRING GENERATION TESTS
# ============================================================================

@Strategy.strategy("string_strategy")
def test_string_generation(username, code, category):
    """Test string parameter generation."""
    assert isinstance(username, str)
    assert 5 <= len(username) <= 15
    
    assert isinstance(code, str)
    assert len(code) == 6
    assert all(c in "0123456789ABCDEF" for c in code)
    
    assert category in ["admin", "user", "guest"]


# ============================================================================
# MIXED MODE TESTS
# ============================================================================

@Strategy.strategy("mixed_static_random_strategy")
def test_mixed_mode(static_val, random_val, directed_val, mixed_val):
    """Test mixed static/directed/random strategy."""
    assert static_val == 42  # Always static
    assert 0 <= random_val <= 100  # Random
    assert directed_val in [1, 2, 3]  # From directed values
    assert 0 <= mixed_val <= 100  # Can be directed or random


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================

@Strategy.strategy("legacy_tuple_strategy")
def test_legacy_tuple(x, y, z):
    """Test legacy tuple-based strategy."""
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(z, int)
    assert y == x * 2
    assert z == x * 3


# ============================================================================
# RNG SEED REPRODUCIBILITY TESTS
# ============================================================================

class TestReproducibility:
    """Test RNG seed reproducibility across strategies."""

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        # Create parameter
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 1000)),
            TestArg("y", rng_type=RNGInteger(0, 1000))
        )
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = param.generate_vectors(10, mode="random_only")
        
        # Generate with same seed
        RNG.seed(42)
        samples2 = param.generate_vectors(10, mode="random_only")
        
        # Should be identical
        assert samples1 == samples2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("value", rng_type=RNGInteger(0, 1000))
        )
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = param.generate_vectors(20, mode="random_only")
        
        # Generate with different seed
        RNG.seed(99)
        samples2 = param.generate_vectors(20, mode="random_only")
        
        # Should be different
        assert samples1 != samples2


# ============================================================================
# DIRECTED VECTOR TESTS
# ============================================================================

class TestDirectedVectors:
    """Test directed vector functionality."""

    def test_directed_vectors_included(self):
        """Test that directed vectors are included in samples."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            TestArg("y", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "origin": (0, 0),
                "max": (100, 100),
            }
        )
        
        # Generate in 'all' mode
        samples = param.generate_vectors(5, mode="all")
        
        # Should include directed vectors
        assert (0, 0) in samples
        assert (100, 100) in samples
        assert len(samples) == 7  # 2 directed + 5 random

    def test_directed_only_mode(self):
        """Test directed_only mode."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("value", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "zero": (0,),
                "fifty": (50,),
                "max": (100,),
            }
        )
        
        samples = param.generate_vectors(100, mode="directed_only")
        
        # Should only have directed vectors
        assert len(samples) == 3
        assert (0,) in samples
        assert (50,) in samples
        assert (100,) in samples


# ============================================================================
# CLI OPTIONS INTEGRATION TESTS
# ============================================================================

class TestCLIIntegration:
    """Test CLI options integration (these test the underlying functionality)."""

    def test_vector_mode_random_only(self):
        """Test random_only mode."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("value", rng_type=RNGInteger(0, 100)),
            directed_vectors={"zero": (0,)}
        )
        
        samples = param.generate_vectors(10, mode="random_only")
        
        # Should not include directed vector
        assert (0,) not in samples
        assert len(samples) == 10

    def test_filter_by_name(self):
        """Test filtering by vector name."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            TestArg("y", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "origin": (0, 0),
                "center": (50, 50),
                "max": (100, 100),
            }
        )
        
        samples = param.generate_vectors(0, filter_by_name="center")
        
        # Should only have the named vector
        assert len(samples) == 1
        assert samples[0] == (50, 50)

    def test_filter_by_index(self):
        """Test filtering by vector index."""
        from pytest_strategy import Parameter, TestArg, RNGInteger
        
        param = Parameter(
            TestArg("value", rng_type=RNGInteger(0, 100)),
            directed_vectors={
                "first": (10,),
                "second": (20,),
                "third": (30,),
            }
        )
        
        samples = param.generate_vectors(0, filter_by_index=1)
        
        # Should only have the indexed vector
        assert len(samples) == 1
        assert samples[0] == (20,)


# ============================================================================
# COMPLETE WORKFLOW TEST
# ============================================================================

class TestCompleteWorkflow:
    """Test the complete pytest-strategies workflow."""

    def test_end_to_end_workflow(self):
        """
        Test complete workflow:
        1. Create RNG types
        2. Create TestArgs with RNG types
        3. Create Parameter with TestArgs
        4. Add directed vectors
        5. Add constraints
        6. Generate samples
        7. Verify all features work together
        """
        from pytest_strategy import (
            Parameter,
            TestArg,
            RNG,
            RNGInteger,
            RNGFloat,
            RNGChoice,
        )
        
        # Set seed for reproducibility
        RNG.seed(42)
        
        # Create TestArgs with various RNG types
        arg1 = TestArg(
            "count",
            rng_type=RNGInteger(1, 100, predicate=lambda x: x > 0),
            validator=lambda x: x > 0
        )
        arg2 = TestArg(
            "ratio",
            rng_type=RNGFloat(0.0, 1.0),
            validator=lambda x: 0.0 <= x <= 1.0
        )
        arg3 = TestArg(
            "mode",
            rng_type=RNGChoice(["fast", "slow", "medium"])
        )
        
        # Create Parameter with constraints
        param = Parameter(
            arg1, arg2, arg3,
            vector_constraints=[
                lambda v: v[0] <= 100,  # count <= 100
                lambda v: v[1] >= 0.0,  # ratio >= 0
            ],
            directed_vectors={
                "baseline": (10, 0.5, "medium"),
                "fast_mode": (1, 0.1, "fast"),
                "slow_mode": (100, 1.0, "slow"),
            }
        )
        
        # Test different generation modes
        all_samples = param.generate_vectors(5, mode="all")
        assert len(all_samples) == 8  # 3 directed + 5 random
        
        random_samples = param.generate_vectors(10, mode="random_only")
        assert len(random_samples) == 10
        
        directed_samples = param.generate_vectors(0, mode="directed_only")
        assert len(directed_samples) == 3
        
        # Verify all samples satisfy constraints
        for sample in all_samples:
            assert 1 <= sample[0] <= 100
            assert 0.0 <= sample[1] <= 1.0
            assert sample[2] in ["fast", "slow", "medium"]
        
        # Test CLI filtering
        filtered = param.generate_vectors(0, filter_by_name="baseline")
        assert len(filtered) == 1
        assert filtered[0] == (10, 0.5, "medium")
        
        # Test introspection
        assert param.arg_names == ("count", "ratio", "mode")
        assert param.num_args == 3
        assert param.num_directed_vectors == 3
        assert "baseline" in param.vector_names
