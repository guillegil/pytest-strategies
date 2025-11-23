"""
Integration tests for TestArg, RNG, and Parameter classes.

These tests demonstrate how the three core components work together
in realistic scenarios, testing the full workflow from RNG configuration
through TestArg creation to Parameter sample generation.
"""

import pytest
from pytest_strategy import (
    RNG,
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGString,
    RNGWeightedInteger,
    RNGWeightedFloat,
    TestArg,
    Parameter,
)


# ============================================================================
# BASIC INTEGRATION TESTS
# ============================================================================

class TestBasicIntegration:
    """Test basic integration of RNG, TestArg, and Parameter."""

    def test_single_arg_workflow(self):
        """Test complete workflow with a single argument."""
        # Set seed for reproducibility
        RNG.seed(42)
        
        # Create RNG type
        rng_type = RNGInteger(0, 100)
        
        # Create TestArg
        arg = TestArg("value", rng_type=rng_type)
        
        # Create Parameter
        param = Parameter(arg)
        
        # Generate samples
        samples = param.generate_samples(10, mode="random_only")
        
        # Verify
        assert len(samples) == 10
        assert all(isinstance(s, tuple) for s in samples)
        assert all(len(s) == 1 for s in samples)
        assert all(0 <= s[0] <= 100 for s in samples)

    def test_multiple_args_workflow(self):
        """Test complete workflow with multiple arguments."""
        RNG.seed(42)
        
        # Create multiple TestArgs with different RNG types
        arg1 = TestArg("count", rng_type=RNGInteger(1, 100))
        arg2 = TestArg("ratio", rng_type=RNGFloat(0.0, 1.0))
        arg3 = TestArg("enabled", rng_type=RNGBoolean(0.7))
        
        # Create Parameter
        param = Parameter(arg1, arg2, arg3)
        
        # Generate samples
        samples = param.generate_samples(20, mode="random_only")
        
        # Verify
        assert len(samples) == 20
        assert all(len(s) == 3 for s in samples)
        assert all(1 <= s[0] <= 100 for s in samples)
        assert all(0.0 <= s[1] <= 1.0 for s in samples)
        assert all(isinstance(s[2], bool) for s in samples)

    def test_directed_and_random_integration(self):
        """Test integration of directed values with random generation."""
        RNG.seed(42)
        
        # Create TestArgs with directed values
        arg1 = TestArg(
            "x",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 50, 100]
        )
        arg2 = TestArg(
            "y",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 50, 100]
        )
        
        # Create Parameter with directed vectors
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "origin": (0, 0),
                "center": (50, 50),
                "max": (100, 100),
            }
        )
        
        # Generate samples in 'all' mode
        samples = param.generate_samples(5, mode="all")
        
        # Should have 3 directed vectors + 5 random = 8 total
        assert len(samples) == 8
        
        # First 3 should be directed vectors
        assert (0, 0) in samples
        assert (50, 50) in samples
        assert (100, 100) in samples


# ============================================================================
# RNG SEED REPRODUCIBILITY TESTS
# ============================================================================

class TestRNGReproducibility:
    """Test that RNG seed ensures reproducible parameter generation."""

    def test_seed_reproducibility_single_arg(self):
        """Test reproducibility with single argument."""
        arg = TestArg("value", rng_type=RNGInteger(0, 1000))
        param = Parameter(arg)
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = param.generate_samples(20, mode="random_only")
        
        # Generate with same seed
        RNG.seed(42)
        samples2 = param.generate_samples(20, mode="random_only")
        
        # Should be identical
        assert samples1 == samples2

    def test_seed_reproducibility_multiple_args(self):
        """Test reproducibility with multiple arguments."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 100))
        arg2 = TestArg("y", rng_type=RNGFloat(0.0, 10.0))
        arg3 = TestArg("mode", rng_type=RNGChoice(["fast", "slow", "medium"]))
        
        param = Parameter(arg1, arg2, arg3)
        
        # Generate with seed 123
        RNG.seed(123)
        samples1 = param.generate_samples(15, mode="random_only")
        
        # Generate with same seed
        RNG.seed(123)
        samples2 = param.generate_samples(15, mode="random_only")
        
        # Should be identical
        assert samples1 == samples2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        arg = TestArg("value", rng_type=RNGInteger(0, 1000))
        param = Parameter(arg)
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = param.generate_samples(20, mode="random_only")
        
        # Generate with different seed
        RNG.seed(99)
        samples2 = param.generate_samples(20, mode="random_only")
        
        # Should be different
        assert samples1 != samples2


# ============================================================================
# CONSTRAINT INTEGRATION TESTS
# ============================================================================

class TestConstraintIntegration:
    """Test integration of RNG predicates and Parameter constraints."""

    def test_rng_predicate_with_parameter(self):
        """Test RNG predicates work with Parameter generation."""
        RNG.seed(42)
        
        # Create TestArgs with RNG predicates
        arg1 = TestArg(
            "even",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)
        )
        arg2 = TestArg(
            "odd",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 1)
        )
        
        param = Parameter(arg1, arg2)
        samples = param.generate_samples(20, mode="random_only")
        
        # Verify predicates were respected
        assert all(s[0] % 2 == 0 for s in samples)  # even
        assert all(s[1] % 2 == 1 for s in samples)  # odd

    def test_parameter_constraints_with_rng(self):
        """Test Parameter-level constraints with RNG generation."""
        RNG.seed(42)
        
        arg1 = TestArg("min", rng_type=RNGInteger(0, 50))
        arg2 = TestArg("max", rng_type=RNGInteger(50, 100))
        
        # Add constraint: min < max
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        
        samples = param.generate_samples(30, mode="random_only")
        
        # Verify constraint is satisfied
        assert all(s[0] < s[1] for s in samples)

    def test_combined_predicates_and_constraints(self):
        """Test combination of RNG predicates and Parameter constraints."""
        RNG.seed(42)
        
        # Even numbers only
        arg1 = TestArg(
            "x",
            rng_type=RNGInteger(0, 50, predicate=lambda x: x % 2 == 0)
        )
        # Odd numbers only
        arg2 = TestArg(
            "y",
            rng_type=RNGInteger(0, 50, predicate=lambda x: x % 2 == 1)
        )
        
        # Constraint: x + y < 60
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] + v[1] < 60]
        )
        
        samples = param.generate_samples(25, mode="random_only")
        
        # Verify both predicates and constraints
        assert all(s[0] % 2 == 0 for s in samples)  # x is even
        assert all(s[1] % 2 == 1 for s in samples)  # y is odd
        assert all(s[0] + s[1] < 60 for s in samples)  # sum < 60


# ============================================================================
# WEIGHTED RNG INTEGRATION TESTS
# ============================================================================

class TestWeightedRNGIntegration:
    """Test integration of weighted RNG types with Parameters."""

    def test_weighted_integer_distribution(self):
        """Test weighted integer generation in Parameter context."""
        RNG.seed(42)
        
        # 80% low values, 20% high values
        arg = TestArg(
            "value",
            rng_type=RNGWeightedInteger(
                ranges={
                    (0, 10): 0.8,
                    (90, 100): 0.2,
                }
            )
        )
        
        param = Parameter(arg)
        samples = param.generate_samples(100, mode="random_only")
        
        # Count distribution
        low_count = sum(1 for s in samples if 0 <= s[0] <= 10)
        high_count = sum(1 for s in samples if 90 <= s[0] <= 100)
        
        # Should be roughly 80/20 split (allow variance)
        assert 60 <= low_count <= 95
        assert 5 <= high_count <= 40

    def test_weighted_float_with_constraints(self):
        """Test weighted float generation with constraints."""
        RNG.seed(42)
        
        arg1 = TestArg(
            "threshold",
            rng_type=RNGWeightedFloat(
                ranges={
                    (0.0, 0.3): 0.5,
                    (0.7, 1.0): 0.5,
                }
            )
        )
        arg2 = TestArg("value", rng_type=RNGFloat(0.0, 1.0))
        
        # Constraint: value must be within threshold
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[1] <= v[0] or v[1] >= 0.7]
        )
        
        samples = param.generate_samples(50, mode="random_only")
        
        # Verify constraint
        assert all(s[1] <= s[0] or s[1] >= 0.7 for s in samples)


# ============================================================================
# COMPLEX TYPE INTEGRATION TESTS
# ============================================================================

class TestComplexTypeIntegration:
    """Test integration with various RNG types."""

    def test_all_rng_types_together(self):
        """Test Parameter with all different RNG types."""
        RNG.seed(42)
        
        args = [
            TestArg("int_val", rng_type=RNGInteger(0, 100)),
            TestArg("float_val", rng_type=RNGFloat(0.0, 1.0)),
            TestArg("bool_val", rng_type=RNGBoolean(0.5)),
            TestArg("choice_val", rng_type=RNGChoice(["a", "b", "c"])),
            TestArg("string_val", rng_type=RNGString(length=5)),
            TestArg("weighted_int", rng_type=RNGWeightedInteger({(0, 10): 1.0})),
            TestArg("weighted_float", rng_type=RNGWeightedFloat({(0.0, 1.0): 1.0})),
        ]
        
        param = Parameter(*args)
        samples = param.generate_samples(10, mode="random_only")
        
        # Verify all samples have correct structure
        assert len(samples) == 10
        assert all(len(s) == 7 for s in samples)
        
        # Verify types
        for sample in samples:
            assert isinstance(sample[0], int)
            assert isinstance(sample[1], float)
            assert isinstance(sample[2], bool)
            assert sample[3] in ["a", "b", "c"]
            assert isinstance(sample[4], str) and len(sample[4]) == 5
            assert isinstance(sample[5], int)
            assert isinstance(sample[6], float)

    def test_string_generation_integration(self):
        """Test string generation in Parameter context."""
        RNG.seed(42)
        
        arg1 = TestArg("name", rng_type=RNGString(min_length=5, max_length=10))
        arg2 = TestArg("code", rng_type=RNGString(length=4, charset="0123456789"))
        
        param = Parameter(arg1, arg2)
        samples = param.generate_samples(15, mode="random_only")
        
        # Verify string constraints
        assert all(5 <= len(s[0]) <= 10 for s in samples)
        assert all(len(s[1]) == 4 for s in samples)
        assert all(all(c in "0123456789" for c in s[1]) for s in samples)


# ============================================================================
# REAL-WORLD SCENARIO TESTS
# ============================================================================

class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_api_endpoint_testing_scenario(self):
        """Simulate testing an API endpoint with various parameters."""
        RNG.seed(42)
        
        # Simulate API parameters
        arg_user_id = TestArg(
            "user_id",
            rng_type=RNGInteger(1, 10000),
            directed_values=[1, 999, 10000]  # Edge cases
        )
        arg_page_size = TestArg(
            "page_size",
            rng_type=RNGWeightedInteger(
                ranges={
                    (10, 50): 0.7,    # Common sizes
                    (100, 500): 0.3,  # Large pages
                }
            ),
            directed_values=[1, 10, 100]  # Edge cases
        )
        arg_include_deleted = TestArg(
            "include_deleted",
            rng_type=RNGBoolean(0.2)  # 20% true
        )
        
        param = Parameter(
            arg_user_id,
            arg_page_size,
            arg_include_deleted,
            directed_vectors={
                "admin_user": (1, 50, True),
                "regular_user": (5000, 25, False),
            }
        )
        
        # Generate test cases
        samples = param.generate_samples(20, mode="all")
        
        # Should include directed vectors
        assert (1, 50, True) in samples
        assert (5000, 25, False) in samples
        
        # Verify all samples are valid
        assert all(1 <= s[0] <= 10000 for s in samples)
        assert all(1 <= s[1] <= 500 for s in samples)
        assert all(isinstance(s[2], bool) for s in samples)

    def test_database_query_testing_scenario(self):
        """Simulate testing database queries with constraints."""
        RNG.seed(42)
        
        arg_offset = TestArg(
            "offset",
            rng_type=RNGInteger(0, 1000),
            directed_values=[0]  # Always test from beginning
        )
        arg_limit = TestArg(
            "limit",
            rng_type=RNGInteger(1, 100),
            directed_values=[1, 10, 100]  # Common limits
        )
        arg_sort_order = TestArg(
            "sort_order",
            rng_type=RNGChoice(["asc", "desc"])
        )
        
        # Constraint: offset + limit <= 1000 (max result set)
        param = Parameter(
            arg_offset,
            arg_limit,
            arg_sort_order,
            vector_constraints=[lambda v: v[0] + v[1] <= 1000]
        )
        
        samples = param.generate_samples(30, mode="all")
        
        # Verify constraint
        assert all(s[0] + s[1] <= 1000 for s in samples)
        assert all(s[2] in ["asc", "desc"] for s in samples)

    def test_configuration_testing_scenario(self):
        """Simulate testing application configuration."""
        RNG.seed(42)
        
        arg_timeout = TestArg(
            "timeout_ms",
            rng_type=RNGWeightedInteger(
                ranges={
                    (100, 1000): 0.6,      # Fast
                    (1000, 5000): 0.3,     # Medium
                    (5000, 30000): 0.1,    # Slow
                }
            ),
            directed_values=[100, 30000]  # Min/max
        )
        arg_retry_count = TestArg(
            "retry_count",
            rng_type=RNGInteger(0, 5),
            directed_values=[0, 1, 5]
        )
        arg_enable_cache = TestArg(
            "enable_cache",
            rng_type=RNGBoolean(0.8)  # Usually enabled
        )
        arg_log_level = TestArg(
            "log_level",
            rng_type=RNGChoice(["DEBUG", "INFO", "WARNING", "ERROR"])
        )
        
        param = Parameter(
            arg_timeout,
            arg_retry_count,
            arg_enable_cache,
            arg_log_level,
            directed_vectors={
                "production": (1000, 3, True, "WARNING"),
                "development": (5000, 1, False, "DEBUG"),
            }
        )
        
        samples = param.generate_samples(25, mode="all")
        
        # Should include directed vectors
        assert (1000, 3, True, "WARNING") in samples
        assert (5000, 1, False, "DEBUG") in samples
        
        # Verify all samples are valid
        assert all(100 <= s[0] <= 30000 for s in samples)
        assert all(0 <= s[1] <= 5 for s in samples)
        assert all(isinstance(s[2], bool) for s in samples)
        assert all(s[3] in ["DEBUG", "INFO", "WARNING", "ERROR"] for s in samples)


# ============================================================================
# MODE INTEGRATION TESTS
# ============================================================================

class TestModeIntegration:
    """Test different generation modes with full integration."""

    def test_all_mode_integration(self):
        """Test 'all' mode with directed values at multiple levels."""
        RNG.seed(42)
        
        arg1 = TestArg(
            "x",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 100],
            always_include_directed=True
        )
        arg2 = TestArg(
            "y",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 100],
            always_include_directed=True
        )
        
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "origin": (0, 0),
                "corner": (100, 100),
            }
        )
        
        samples = param.generate_samples(5, mode="all")
        
        # Should have directed vectors + random samples
        assert len(samples) == 7  # 2 directed + 5 random
        assert (0, 0) in samples
        assert (100, 100) in samples

    def test_mixed_mode_integration(self):
        """Test 'mixed' mode respects always_include_directed flag."""
        RNG.seed(42)
        
        arg = TestArg("value", rng_type=RNGInteger(0, 100))
        
        # With always_include_directed=True
        param_true = Parameter(
            arg,
            directed_vectors={"zero": (0,), "max": (100,)},
            always_include_directed=True
        )
        samples_true = param_true.generate_samples(5, mode="mixed")
        assert len(samples_true) == 7  # 2 directed + 5 random
        
        # With always_include_directed=False
        param_false = Parameter(
            arg,
            directed_vectors={"zero": (0,), "max": (100,)},
            always_include_directed=False
        )
        samples_false = param_false.generate_samples(5, mode="mixed")
        assert len(samples_false) == 5  # Only random

    def test_directed_only_mode_integration(self):
        """Test 'directed_only' mode ignores random generation."""
        RNG.seed(42)
        
        arg1 = TestArg("x", rng_type=RNGInteger(0, 100))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 100))
        
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "v1": (10, 20),
                "v2": (30, 40),
                "v3": (50, 60),
            }
        )
        
        samples = param.generate_samples(100, mode="directed_only")
        
        # Should only have directed vectors, ignore n=100
        assert len(samples) == 3
        assert (10, 20) in samples
        assert (30, 40) in samples
        assert (50, 60) in samples


# ============================================================================
# VALIDATION INTEGRATION TESTS
# ============================================================================

class TestValidationIntegration:
    """Test validation across RNG, TestArg, and Parameter."""

    def test_testarg_validator_integration(self):
        """Test TestArg validators work with Parameter generation."""
        RNG.seed(42)
        
        arg1 = TestArg(
            "positive",
            rng_type=RNGInteger(1, 100, predicate=lambda x: x > 0),
            validator=lambda x: x > 0
        )
        arg2 = TestArg(
            "even",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0),
            validator=lambda x: x % 2 == 0
        )
        
        param = Parameter(arg1, arg2)
        samples = param.generate_samples(20, mode="random_only")
        
        # Validators should be satisfied
        assert all(s[0] > 0 for s in samples)
        assert all(s[1] % 2 == 0 for s in samples)

    def test_multi_level_constraints(self):
        """Test constraints at RNG, TestArg, and Parameter levels."""
        RNG.seed(42)
        
        # RNG level: only multiples of 5
        arg1 = TestArg(
            "x",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 5 == 0),
            validator=lambda x: x % 5 == 0  # TestArg level
        )
        # RNG level: only multiples of 10
        arg2 = TestArg(
            "y",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 10 == 0),
            validator=lambda x: x % 10 == 0  # TestArg level
        )
        
        # Parameter level: x + y <= 100
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] + v[1] <= 100]
        )
        
        samples = param.generate_samples(15, mode="random_only")
        
        # All constraints should be satisfied
        assert all(s[0] % 5 == 0 for s in samples)  # RNG + TestArg
        assert all(s[1] % 10 == 0 for s in samples)  # RNG + TestArg
        assert all(s[0] + s[1] <= 100 for s in samples)  # Parameter
