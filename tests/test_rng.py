"""
Unit tests for RNG module.

Tests all RNG functions, methods, and RNG type classes to ensure
proper random value generation, seed management, and constraint handling.
"""

import pytest
import random
from pytest_strategy import (
    RNG,
    RNGValueError,
    RNGType,
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGString,
    RNGWeightedInteger,
    RNGWeightedFloat,
)


# ============================================================================
# SEED MANAGEMENT TESTS
# ============================================================================

class TestRNGSeedManagement:
    """Test RNG seed management functionality."""

    def test_seed_initialization(self):
        """Test that RNG initializes with a seed."""
        seed = RNG.get_seed()
        assert seed is not None
        assert isinstance(seed, int)

    def test_seed_setting(self):
        """Test setting a specific seed."""
        test_seed = 42
        RNG.seed(test_seed)
        assert RNG.get_seed() == test_seed

    def test_seed_none_keeps_current(self):
        """Test that seed(None) keeps the current seed."""
        RNG.seed(12345)
        current_seed = RNG.get_seed()
        RNG.seed(None)
        assert RNG.get_seed() == current_seed

    def test_seed_reproducibility(self):
        """Test that same seed produces same sequence."""
        RNG.seed(42)
        values1 = [RNG.integer(0, 100) for _ in range(10)]

        RNG.seed(42)
        values2 = [RNG.integer(0, 100) for _ in range(10)]

        assert values1 == values2

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different sequences."""
        RNG.seed(42)
        values1 = [RNG.integer(0, 100) for _ in range(10)]

        RNG.seed(99)
        values2 = [RNG.integer(0, 100) for _ in range(10)]

        assert values1 != values2

    def test_refresh_seed(self):
        """Test that refresh_seed resets to current seed."""
        RNG.seed(42)
        values1 = [RNG.integer(0, 100) for _ in range(5)]

        RNG.refresh_seed()
        values2 = [RNG.integer(0, 100) for _ in range(5)]

        assert values1 == values2

    def test_set_max_retries(self):
        """Test setting max retries for constrained generation."""
        RNG.set_max_retries(50)
        assert RNG._max_retries == 50

        # Reset to default
        RNG.set_max_retries(100)


# ============================================================================
# INTEGER GENERATION TESTS
# ============================================================================

class TestRNGInteger:
    """Test RNG.integer() method."""

    def test_integer_default_range(self):
        """Test integer generation with default range."""
        RNG.seed(42)
        value = RNG.integer()
        assert isinstance(value, int)
        assert -2**31 <= value <= 2**31 - 1

    def test_integer_custom_range(self):
        """Test integer generation with custom range."""
        RNG.seed(42)
        for _ in range(100):
            value = RNG.integer(0, 10)
            assert 0 <= value <= 10

    def test_integer_single_value_range(self):
        """Test integer generation when min == max."""
        value = RNG.integer(5, 5)
        assert value == 5

    def test_integer_negative_range(self):
        """Test integer generation with negative range."""
        RNG.seed(42)
        for _ in range(100):
            value = RNG.integer(-100, -50)
            assert -100 <= value <= -50

    def test_integer_with_predicate(self):
        """Test integer generation with predicate constraint."""
        RNG.seed(42)
        # Generate only even numbers
        value = RNG.integer(0, 100, predicate=lambda x: x % 2 == 0)
        assert value % 2 == 0

    def test_integer_predicate_multiple_values(self):
        """Test that predicate works consistently."""
        RNG.seed(42)
        for _ in range(20):
            value = RNG.integer(0, 100, predicate=lambda x: x % 5 == 0)
            assert value % 5 == 0

    def test_integer_impossible_predicate_raises_error(self):
        """Test that impossible predicate raises RNGValueError."""
        RNG.seed(42)
        RNG.set_max_retries(10)

        with pytest.raises(RNGValueError, match="No valid value found"):
            # Impossible: number between 0-10 that's > 100
            RNG.integer(0, 10, predicate=lambda x: x > 100)

        # Reset
        RNG.set_max_retries(100)

    def test_integer_distribution(self):
        """Test that integer generation has reasonable distribution."""
        RNG.seed(42)
        values = [RNG.integer(0, 10) for _ in range(1000)]

        # Check that we got a variety of values
        unique_values = set(values)
        assert len(unique_values) >= 8  # Should have most values in range


# ============================================================================
# FLOAT GENERATION TESTS
# ============================================================================

class TestRNGFloat:
    """Test RNG.float() method."""

    def test_float_default_range(self):
        """Test float generation with default range."""
        RNG.seed(42)
        value = RNG.float()
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_float_custom_range(self):
        """Test float generation with custom range."""
        RNG.seed(42)
        for _ in range(100):
            value = RNG.float(0.0, 10.0)
            assert 0.0 <= value <= 10.0

    def test_float_negative_range(self):
        """Test float generation with negative range."""
        RNG.seed(42)
        for _ in range(100):
            value = RNG.float(-10.0, -1.0)
            assert -10.0 <= value <= -1.0

    def test_float_with_predicate(self):
        """Test float generation with predicate constraint."""
        RNG.seed(42)
        # Generate only values > 0.5
        value = RNG.float(0.0, 1.0, predicate=lambda x: x > 0.5)
        assert value > 0.5

    def test_float_predicate_multiple_values(self):
        """Test that predicate works consistently."""
        RNG.seed(42)
        for _ in range(20):
            value = RNG.float(0.0, 10.0, predicate=lambda x: x > 5.0)
            assert value > 5.0

    def test_float_impossible_predicate_raises_error(self):
        """Test that impossible predicate raises RNGValueError."""
        RNG.seed(42)
        RNG.set_max_retries(10)

        with pytest.raises(RNGValueError, match="No valid value found"):
            # Impossible: float between 0-1 that's > 10
            RNG.float(0.0, 1.0, predicate=lambda x: x > 10.0)

        # Reset
        RNG.set_max_retries(100)

    def test_float_precision(self):
        """Test that float values have proper precision."""
        RNG.seed(42)
        values = [RNG.float(0.0, 1.0) for _ in range(100)]

        # Check that we have variety (not all same value)
        unique_values = set(values)
        assert len(unique_values) > 50  # Should have many unique values


# ============================================================================
# BOOLEAN GENERATION TESTS
# ============================================================================

class TestRNGBoolean:
    """Test RNG.boolean() method."""

    def test_boolean_default_probability(self):
        """Test boolean generation with default 50/50 probability."""
        RNG.seed(42)
        values = [RNG.boolean() for _ in range(1000)]

        true_count = sum(values)
        true_ratio = true_count / len(values)

        # Should be roughly 50% (allow 10% variance)
        assert 0.4 <= true_ratio <= 0.6

    def test_boolean_custom_probability(self):
        """Test boolean generation with custom probability."""
        RNG.seed(42)
        values = [RNG.boolean(true_probability=0.8) for _ in range(1000)]

        true_count = sum(values)
        true_ratio = true_count / len(values)

        # Should be roughly 80% (allow 10% variance)
        assert 0.7 <= true_ratio <= 0.9

    def test_boolean_always_true(self):
        """Test boolean with probability 1.0 always returns True."""
        RNG.seed(42)
        values = [RNG.boolean(true_probability=1.0) for _ in range(100)]
        assert all(values)

    def test_boolean_always_false(self):
        """Test boolean with probability 0.0 always returns False."""
        RNG.seed(42)
        values = [RNG.boolean(true_probability=0.0) for _ in range(100)]
        assert not any(values)

    def test_boolean_returns_bool_type(self):
        """Test that boolean returns actual bool type."""
        RNG.seed(42)
        value = RNG.boolean()
        assert isinstance(value, bool)


# ============================================================================
# CHOICE GENERATION TESTS
# ============================================================================

class TestRNGChoice:
    """Test RNG.choice() method."""

    def test_choice_from_list(self):
        """Test choosing from a list of items."""
        RNG.seed(42)
        choices = ['a', 'b', 'c', 'd']
        value = RNG.choice(choices)
        assert value in choices

    def test_choice_all_items_possible(self):
        """Test that all items can be chosen."""
        RNG.seed(42)
        choices = ['a', 'b', 'c']
        values = [RNG.choice(choices) for _ in range(100)]

        # All choices should appear at least once
        assert set(values) == set(choices)

    def test_choice_single_item(self):
        """Test choosing from single-item list."""
        value = RNG.choice(['only'])
        assert value == 'only'

    def test_choice_empty_list_raises_error(self):
        """Test that empty list raises RNGValueError."""
        with pytest.raises(RNGValueError, match="cannot be empty"):
            RNG.choice([])

    def test_choice_different_types(self):
        """Test choosing from list with different types."""
        RNG.seed(42)
        choices = [1, 'two', 3.0, True, None]
        value = RNG.choice(choices)
        assert value in choices

    def test_choice_distribution(self):
        """Test that choice has reasonable distribution."""
        RNG.seed(42)
        choices = ['a', 'b', 'c']
        values = [RNG.choice(choices) for _ in range(300)]

        # Each choice should appear roughly 1/3 of the time (allow variance)
        for choice in choices:
            count = values.count(choice)
            ratio = count / len(values)
            assert 0.2 <= ratio <= 0.45  # Roughly 1/3 with variance


# ============================================================================
# STRING GENERATION TESTS
# ============================================================================

class TestRNGString:
    """Test RNG.string() method."""

    def test_string_fixed_length(self):
        """Test string generation with fixed length."""
        RNG.seed(42)
        value = RNG.string(length=10)
        assert isinstance(value, str)
        assert len(value) == 10

    def test_string_variable_length(self):
        """Test string generation with variable length."""
        RNG.seed(42)
        for _ in range(20):
            value = RNG.string(min_length=5, max_length=15)
            assert 5 <= len(value) <= 15

    def test_string_default_charset(self):
        """Test string uses default lowercase charset."""
        RNG.seed(42)
        value = RNG.string(length=100)
        assert all(c in "abcdefghijklmnopqrstuvwxyz" for c in value)

    def test_string_custom_charset(self):
        """Test string generation with custom charset."""
        RNG.seed(42)
        value = RNG.string(length=50, charset="0123456789")
        assert all(c in "0123456789" for c in value)

    def test_string_empty_length(self):
        """Test string generation with length 0."""
        value = RNG.string(length=0)
        assert value == ""

    def test_string_single_char_charset(self):
        """Test string with single character charset."""
        value = RNG.string(length=10, charset="a")
        assert value == "aaaaaaaaaa"

    def test_string_variety(self):
        """Test that strings have variety in characters."""
        RNG.seed(42)
        value = RNG.string(length=100)
        unique_chars = set(value)

        # Should have multiple different characters
        assert len(unique_chars) > 5


# ============================================================================
# WEIGHTED INTEGER TESTS
# ============================================================================

class TestRNGWeightedInteger:
    """Test RNG.winteger() method."""

    def test_winteger_single_range(self):
        """Test weighted integer with single range."""
        RNG.seed(42)
        ranges = {(0, 10): 1.0}
        for _ in range(20):
            value = RNG.winteger(ranges)
            assert 0 <= value <= 10

    def test_winteger_multiple_ranges(self):
        """Test weighted integer with multiple ranges."""
        RNG.seed(42)
        ranges = {
            (0, 10): 0.5,
            (20, 30): 0.5,
        }
        for _ in range(20):
            value = RNG.winteger(ranges)
            assert (0 <= value <= 10) or (20 <= value <= 30)

    def test_winteger_weight_distribution(self):
        """Test that weights affect distribution."""
        RNG.seed(42)
        ranges = {
            (0, 10): 0.9,    # 90% weight
            (20, 30): 0.1,   # 10% weight
        }

        values = [RNG.winteger(ranges) for _ in range(1000)]

        low_range_count = sum(1 for v in values if 0 <= v <= 10)
        high_range_count = sum(1 for v in values if 20 <= v <= 30)

        # Low range should have ~90% of values
        low_ratio = low_range_count / len(values)
        assert 0.8 <= low_ratio <= 1.0  # Allow some variance

    def test_winteger_with_predicate(self):
        """Test weighted integer with predicate."""
        RNG.seed(42)
        ranges = {(0, 100): 1.0}
        value = RNG.winteger(ranges, predicate=lambda x: x % 2 == 0)
        assert value % 2 == 0

    def test_winteger_unnormalized_weights(self):
        """Test that weights don't need to sum to 1.0."""
        RNG.seed(42)
        ranges = {
            (0, 10): 8,    # 80%
            (20, 30): 2,   # 20%
        }

        # Should work without error
        values = [RNG.winteger(ranges) for _ in range(100)]
        assert len(values) == 100


# ============================================================================
# WEIGHTED FLOAT TESTS
# ============================================================================

class TestRNGWeightedFloat:
    """Test RNG.wfloat() method."""

    def test_wfloat_single_range(self):
        """Test weighted float with single range."""
        RNG.seed(42)
        ranges = {(0.0, 10.0): 1.0}
        for _ in range(20):
            value = RNG.wfloat(ranges)
            assert 0.0 <= value <= 10.0

    def test_wfloat_multiple_ranges(self):
        """Test weighted float with multiple ranges."""
        RNG.seed(42)
        ranges = {
            (0.0, 1.0): 0.5,
            (10.0, 20.0): 0.5,
        }
        for _ in range(20):
            value = RNG.wfloat(ranges)
            assert (0.0 <= value <= 1.0) or (10.0 <= value <= 20.0)

    def test_wfloat_weight_distribution(self):
        """Test that weights affect distribution."""
        RNG.seed(42)
        ranges = {
            (0.0, 1.0): 0.9,     # 90% weight
            (10.0, 20.0): 0.1,   # 10% weight
        }

        values = [RNG.wfloat(ranges) for _ in range(1000)]

        low_range_count = sum(1 for v in values if 0.0 <= v <= 1.0)
        high_range_count = sum(1 for v in values if 10.0 <= v <= 20.0)

        # Low range should have ~90% of values
        low_ratio = low_range_count / len(values)
        assert 0.8 <= low_ratio <= 1.0  # Allow some variance

    def test_wfloat_with_predicate(self):
        """Test weighted float with predicate."""
        RNG.seed(42)
        ranges = {(0.0, 10.0): 1.0}
        value = RNG.wfloat(ranges, predicate=lambda x: x > 5.0)
        assert value > 5.0


# ============================================================================
# RNG TYPE CLASS TESTS
# ============================================================================

class TestRNGIntegerType:
    """Test RNGInteger type class."""

    def test_rng_integer_type_creation(self):
        """Test creating RNGInteger type."""
        rng_type = RNGInteger(0, 100)
        assert rng_type.min == 0
        assert rng_type.max == 100

    def test_rng_integer_type_default_range(self):
        """Test RNGInteger with default range."""
        rng_type = RNGInteger()
        assert rng_type.min == -2**31
        assert rng_type.max == 2**31 - 1

    def test_rng_integer_type_generate(self):
        """Test RNGInteger.generate() method."""
        RNG.seed(42)
        rng_type = RNGInteger(0, 10)
        for _ in range(20):
            value = rng_type.generate()
            assert 0 <= value <= 10

    def test_rng_integer_type_python_type(self):
        """Test RNGInteger.python_type property."""
        rng_type = RNGInteger(0, 100)
        assert rng_type.python_type == int

    def test_rng_integer_type_with_predicate(self):
        """Test RNGInteger with predicate."""
        RNG.seed(42)
        rng_type = RNGInteger(0, 100, predicate=lambda x: x % 2 == 0)
        value = rng_type.generate()
        assert value % 2 == 0


class TestRNGFloatType:
    """Test RNGFloat type class."""

    def test_rng_float_type_creation(self):
        """Test creating RNGFloat type."""
        rng_type = RNGFloat(0.0, 10.0)
        assert rng_type.min == 0.0
        assert rng_type.max == 10.0

    def test_rng_float_type_default_range(self):
        """Test RNGFloat with default range."""
        rng_type = RNGFloat()
        assert rng_type.min == 0.0
        assert rng_type.max == 1.0

    def test_rng_float_type_generate(self):
        """Test RNGFloat.generate() method."""
        RNG.seed(42)
        rng_type = RNGFloat(0.0, 10.0)
        for _ in range(20):
            value = rng_type.generate()
            assert 0.0 <= value <= 10.0

    def test_rng_float_type_python_type(self):
        """Test RNGFloat.python_type property."""
        rng_type = RNGFloat(0.0, 1.0)
        assert rng_type.python_type == float

    def test_rng_float_type_with_predicate(self):
        """Test RNGFloat with predicate."""
        RNG.seed(42)
        rng_type = RNGFloat(0.0, 10.0, predicate=lambda x: x > 5.0)
        value = rng_type.generate()
        assert value > 5.0


class TestRNGBooleanType:
    """Test RNGBoolean type class."""

    def test_rng_boolean_type_creation(self):
        """Test creating RNGBoolean type."""
        rng_type = RNGBoolean(0.7)
        assert rng_type.true_probability == 0.7

    def test_rng_boolean_type_default_probability(self):
        """Test RNGBoolean with default probability."""
        rng_type = RNGBoolean()
        assert rng_type.true_probability == 0.5

    def test_rng_boolean_type_generate(self):
        """Test RNGBoolean.generate() method."""
        RNG.seed(42)
        rng_type = RNGBoolean(0.8)
        values = [rng_type.generate() for _ in range(1000)]

        true_ratio = sum(values) / len(values)
        assert 0.7 <= true_ratio <= 0.9

    def test_rng_boolean_type_python_type(self):
        """Test RNGBoolean.python_type property."""
        rng_type = RNGBoolean()
        assert rng_type.python_type == bool


class TestRNGChoiceType:
    """Test RNGChoice type class."""

    def test_rng_choice_type_creation(self):
        """Test creating RNGChoice type."""
        choices = ['a', 'b', 'c']
        rng_type = RNGChoice(choices)
        assert rng_type.choices == choices

    def test_rng_choice_type_empty_raises_error(self):
        """Test that empty choices raises error."""
        with pytest.raises(RNGValueError, match="cannot be empty"):
            RNGChoice([])

    def test_rng_choice_type_generate(self):
        """Test RNGChoice.generate() method."""
        RNG.seed(42)
        choices = ['a', 'b', 'c']
        rng_type = RNGChoice(choices)

        for _ in range(20):
            value = rng_type.generate()
            assert value in choices

    def test_rng_choice_type_python_type(self):
        """Test RNGChoice.python_type property."""
        rng_type = RNGChoice(['a', 'b', 'c'])
        assert rng_type.python_type == str

        rng_type_int = RNGChoice([1, 2, 3])
        assert rng_type_int.python_type == int


class TestRNGStringType:
    """Test RNGString type class."""

    def test_rng_string_type_creation(self):
        """Test creating RNGString type."""
        rng_type = RNGString(length=10)
        assert rng_type.length == 10

    def test_rng_string_type_default_params(self):
        """Test RNGString with default parameters."""
        rng_type = RNGString()
        assert rng_type.length is None
        assert rng_type.min_length == 1
        assert rng_type.max_length == 20
        assert rng_type.charset == "abcdefghijklmnopqrstuvwxyz"

    def test_rng_string_type_generate_fixed_length(self):
        """Test RNGString.generate() with fixed length."""
        RNG.seed(42)
        rng_type = RNGString(length=10)
        value = rng_type.generate()
        assert len(value) == 10

    def test_rng_string_type_generate_variable_length(self):
        """Test RNGString.generate() with variable length."""
        RNG.seed(42)
        rng_type = RNGString(min_length=5, max_length=15)
        for _ in range(20):
            value = rng_type.generate()
            assert 5 <= len(value) <= 15

    def test_rng_string_type_python_type(self):
        """Test RNGString.python_type property."""
        rng_type = RNGString()
        assert rng_type.python_type == str


class TestRNGWeightedIntegerType:
    """Test RNGWeightedInteger type class."""

    def test_rng_weighted_integer_type_creation(self):
        """Test creating RNGWeightedInteger type."""
        ranges = {(0, 10): 0.8, (20, 30): 0.2}
        rng_type = RNGWeightedInteger(ranges)
        assert rng_type.ranges == ranges

    def test_rng_weighted_integer_type_generate(self):
        """Test RNGWeightedInteger.generate() method."""
        RNG.seed(42)
        ranges = {(0, 10): 0.8, (20, 30): 0.2}
        rng_type = RNGWeightedInteger(ranges)

        for _ in range(20):
            value = rng_type.generate()
            assert (0 <= value <= 10) or (20 <= value <= 30)

    def test_rng_weighted_integer_type_python_type(self):
        """Test RNGWeightedInteger.python_type property."""
        ranges = {(0, 10): 1.0}
        rng_type = RNGWeightedInteger(ranges)
        assert rng_type.python_type == int


class TestRNGWeightedFloatType:
    """Test RNGWeightedFloat type class."""

    def test_rng_weighted_float_type_creation(self):
        """Test creating RNGWeightedFloat type."""
        ranges = {(0.0, 1.0): 0.8, (10.0, 20.0): 0.2}
        rng_type = RNGWeightedFloat(ranges)
        assert rng_type.ranges == ranges

    def test_rng_weighted_float_type_generate(self):
        """Test RNGWeightedFloat.generate() method."""
        RNG.seed(42)
        ranges = {(0.0, 1.0): 0.8, (10.0, 20.0): 0.2}
        rng_type = RNGWeightedFloat(ranges)

        for _ in range(20):
            value = rng_type.generate()
            assert (0.0 <= value <= 1.0) or (10.0 <= value <= 20.0)

    def test_rng_weighted_float_type_python_type(self):
        """Test RNGWeightedFloat.python_type property."""
        ranges = {(0.0, 1.0): 1.0}
        rng_type = RNGWeightedFloat(ranges)
        assert rng_type.python_type == float


# ============================================================================
# BASE CLASS TESTS
# ============================================================================

class TestRNGTypeBase:
    """Test RNGType base class."""

    def test_rng_type_base_not_implemented(self):
        """Test that RNGType base class raises NotImplementedError."""
        rng_type = RNGType()

        with pytest.raises(NotImplementedError):
            rng_type.generate()

        with pytest.raises(NotImplementedError):
            _ = rng_type.python_type


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestRNGEdgeCases:
    """Test edge cases and error handling."""

    def test_integer_min_greater_than_max(self):
        """Test that min > max still works (random.randint handles it)."""
        # Python's random.randint will raise ValueError
        with pytest.raises(ValueError):
            RNG.integer(10, 5)

    def test_float_min_greater_than_max(self):
        """Test float with min > max."""
        # random.uniform handles this by swapping
        RNG.seed(42)
        value = RNG.float(10.0, 5.0)
        assert 5.0 <= value <= 10.0

    def test_boolean_invalid_probability(self):
        """Test boolean with invalid probability values."""
        # Values outside 0-1 should still work (just always True or False)
        RNG.seed(42)

        # Probability > 1 should always be True
        assert RNG.boolean(true_probability=2.0) is True

        # Probability < 0 should always be False
        assert RNG.boolean(true_probability=-1.0) is False

    def test_string_negative_length(self):
        """Test string with negative length."""
        # Should raise ValueError from range()
        with pytest.raises(ValueError):
            RNG.string(length=-5)

    def test_weighted_empty_ranges(self):
        """Test weighted generation with empty ranges dict."""
        # Should raise IndexError when trying to choose
        with pytest.raises(IndexError):
            RNG.winteger({})

    def test_large_seed_value(self):
        """Test setting very large seed value."""
        large_seed = 2**63 - 1
        RNG.seed(large_seed)
        assert RNG.get_seed() == large_seed

        # Should still generate values
        value = RNG.integer(0, 100)
        assert 0 <= value <= 100

    def test_zero_max_retries(self):
        """Test behavior with zero max retries."""
        RNG.set_max_retries(0)

        # Should fail immediately with impossible predicate
        with pytest.raises(RNGValueError, match="No valid value found after 0 attempts"):
            RNG.integer(0, 10, predicate=lambda x: x > 100)

        # Reset
        RNG.set_max_retries(100)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRNGIntegration:
    """Integration tests combining multiple RNG features."""

    def test_multiple_rng_types_together(self):
        """Test using multiple RNG types in sequence."""
        RNG.seed(42)

        int_val = RNG.integer(0, 100)
        float_val = RNG.float(0.0, 1.0)
        bool_val = RNG.boolean()
        choice_val = RNG.choice(['a', 'b', 'c'])
        string_val = RNG.string(length=10)

        assert isinstance(int_val, int)
        assert isinstance(float_val, float)
        assert isinstance(bool_val, bool)
        assert isinstance(choice_val, str)
        assert isinstance(string_val, str)

    def test_rng_type_classes_together(self):
        """Test using multiple RNG type classes together."""
        RNG.seed(42)

        types = [
            RNGInteger(0, 100),
            RNGFloat(0.0, 1.0),
            RNGBoolean(),
            RNGChoice(['x', 'y', 'z']),
            RNGString(length=5),
        ]

        for rng_type in types:
            value = rng_type.generate()
            assert value is not None

    def test_seed_consistency_across_types(self):
        """Test that seed affects all RNG types consistently."""
        RNG.seed(42)
        values1 = [
            RNG.integer(0, 100),
            RNG.float(0.0, 1.0),
            RNG.boolean(),
        ]

        RNG.seed(42)
        values2 = [
            RNG.integer(0, 100),
            RNG.float(0.0, 1.0),
            RNG.boolean(),
        ]

        assert values1 == values2

    def test_complex_predicate_chain(self):
        """Test complex predicates with multiple conditions."""
        RNG.seed(42)

        # Generate even number divisible by 5
        value = RNG.integer(
            0, 100,
            predicate=lambda x: x % 2 == 0 and x % 5 == 0
        )

        assert value % 2 == 0
        assert value % 5 == 0

    def test_weighted_with_complex_ranges(self):
        """Test weighted generation with many ranges."""
        RNG.seed(42)

        ranges = {
            (0, 10): 0.4,
            (20, 30): 0.3,
            (40, 50): 0.2,
            (60, 70): 0.1,
        }

        values = [RNG.winteger(ranges) for _ in range(100)]

        # All values should be in one of the ranges
        for value in values:
            assert any(
                low <= value <= high
                for (low, high) in ranges.keys()
            )


# ============================================================================
# PERFORMANCE TESTS (Optional)
# ============================================================================

class TestRNGPerformance:
    """Performance tests to ensure RNG is reasonably fast."""

    def test_integer_generation_performance(self):
        """Test that integer generation is fast."""
        RNG.seed(42)
        import time

        start = time.time()
        values = [RNG.integer(0, 1000) for _ in range(10000)]
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        assert len(values) == 10000

    def test_predicate_performance(self):
        """Test that predicate generation is reasonably fast."""
        RNG.seed(42)
        import time

        start = time.time()
        # Generate 1000 even numbers (should be fast)
        values = [RNG.integer(0, 1000, predicate=lambda x: x % 2 == 0) 
                  for _ in range(1000)]
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        assert all(v % 2 == 0 for v in values)