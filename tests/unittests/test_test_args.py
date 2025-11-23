"""
Unit tests for TestArg module.

Tests the TestArg class to ensure proper initialization, value generation,
validation, and handling of static, random, and mixed test argument modes.
"""

import pytest
from pytest_strategy import (
    RNGInteger,
    RNGFloat,
    RNGBoolean,
    RNGChoice,
    RNGString,
)
from pytest_strategy.test_args import TestArg


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestTestArgInitialization:
    """Test TestArg initialization in various modes."""

    def test_static_value_initialization(self):
        """Test creating TestArg with static value."""
        arg = TestArg("count", value=42)
        assert arg.name == "count"
        assert arg.is_static
        assert not arg.has_directed_values

    def test_random_initialization(self):
        """Test creating TestArg with RNG type."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100))
        assert arg.name == "count"
        assert not arg.is_static
        assert arg.rng_type is not None

    def test_directed_values_initialization(self):
        """Test creating TestArg with directed values."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100), directed_values=[0, 1, 99])
        assert arg.name == "count"
        assert arg.has_directed_values
        assert len(arg.directed_values) == 3

    def test_mixed_mode_initialization(self):
        """Test creating TestArg with both directed values and RNG type."""
        arg = TestArg(
            "count",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 1],
            description="Test count"
        )
        assert arg.name == "count"
        assert arg.has_directed_values
        assert arg.rng_type is not None
        assert arg.description == "Test count"

    def test_initialization_with_validator(self):
        """Test creating TestArg with validator function."""
        validator = lambda x: x > 0
        arg = TestArg("positive", rng_type=RNGInteger(1, 100), validator=validator)
        assert arg._validator is validator

    def test_initialization_no_value_raises_error(self):
        """Test that initialization without value, rng_type, or directed_values raises error."""
        with pytest.raises(ValueError, match="must have either a value, rng_type, or directed_values"):
            TestArg("invalid")

    def test_initialization_with_description(self):
        """Test initialization with description."""
        arg = TestArg("count", value=10, description="Number of items")
        assert arg.description == "Number of items"


# ============================================================================
# VALUE GENERATION TESTS
# ============================================================================

class TestTestArgGeneration:
    """Test TestArg value generation methods."""

    def test_generate_static_value(self):
        """Test generating from static value."""
        arg = TestArg("count", value=42)
        assert arg.generate() == 42
        assert arg.generate() == 42  # Should always return same value

    def test_generate_random_value(self):
        """Test generating random value."""
        arg = TestArg("count", rng_type=RNGInteger(0, 10))
        value = arg.generate()
        assert isinstance(value, int)
        assert 0 <= value <= 10

    def test_generate_with_validator(self):
        """Test that generate respects validator."""
        # Use RNG predicate to ensure even numbers are generated
        arg = TestArg(
            "even",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0),
            validator=lambda x: x % 2 == 0
        )
        # Generate should produce even numbers that pass validation
        for _ in range(10):
            value = arg.generate()
            assert value % 2 == 0

    def test_generate_without_rng_type_raises_error(self):
        """Test that generate raises error when no rng_type and no static value."""
        arg = TestArg("count", directed_values=[1, 2, 3])
        with pytest.raises(ValueError, match="Cannot generate value"):
            arg.generate()

    def test_generate_samples_static_value(self):
        """Test generating samples from static value."""
        arg = TestArg("count", value=42)
        samples = arg.generate_samples(10)
        assert samples == [42]  # Static value returns single item

    def test_generate_samples_random_only(self):
        """Test generating random samples without directed values."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100))
        samples = arg.generate_samples(5)
        assert len(samples) == 5
        assert all(isinstance(v, int) for v in samples)
        assert all(0 <= v <= 100 for v in samples)

    def test_generate_samples_with_directed_values(self):
        """Test generating samples includes directed values."""
        arg = TestArg(
            "count",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 1, 99],
            always_include_directed=True
        )
        samples = arg.generate_samples(5)
        # Should have 3 directed + 5 random = 8 total
        assert len(samples) == 8
        assert samples[0] == 0
        assert samples[1] == 1
        assert samples[2] == 99

    def test_generate_samples_directed_not_included(self):
        """Test generating samples without directed values when flag is False."""
        arg = TestArg(
            "count",
            rng_type=RNGInteger(0, 100),
            directed_values=[0, 1, 99],
            always_include_directed=False
        )
        samples = arg.generate_samples(5)
        # Should have only 5 random samples
        assert len(samples) == 5

    def test_generate_samples_static_with_directed(self):
        """Test static value with directed values."""
        arg = TestArg(
            "count",
            value=42,
            directed_values=[0, 1],
            always_include_directed=True
        )
        samples = arg.generate_samples(10)
        # Should return only directed values, not the static value
        assert samples == [0, 1]

    def test_generate_samples_zero_count(self):
        """Test generating zero samples."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100))
        samples = arg.generate_samples(0)
        assert samples == []


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestTestArgValidation:
    """Test TestArg validation functionality."""

    def test_validator_passes(self):
        """Test that valid values pass validation."""
        arg = TestArg(
            "positive",
            value=10,
            validator=lambda x: x > 0
        )
        assert arg.generate() == 10

    def test_validator_fails(self):
        """Test that invalid values fail validation."""
        arg = TestArg(
            "positive",
            value=-5,
            validator=lambda x: x > 0
        )
        with pytest.raises(ValueError, match="failed validation"):
            arg.generate()

    def test_validator_with_custom_message(self):
        """Test validation error message includes argument name."""
        arg = TestArg(
            "my_arg",
            value=0,
            validator=lambda x: x > 0
        )
        with pytest.raises(ValueError, match="my_arg"):
            arg.generate()

    def test_no_validator_always_passes(self):
        """Test that without validator, all values are accepted."""
        arg = TestArg("count", value=-100)
        assert arg.generate() == -100


# ============================================================================
# PROPERTY TESTS
# ============================================================================

class TestTestArgProperties:
    """Test TestArg properties and introspection."""

    def test_name_property(self):
        """Test name property."""
        arg = TestArg("my_arg", value=10)
        assert arg.name == "my_arg"

    def test_description_property(self):
        """Test description property."""
        arg = TestArg("count", value=10, description="Item count")
        assert arg.description == "Item count"

    def test_type_property_from_rng(self):
        """Test type property from RNG type."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100))
        assert arg.type == int

    def test_type_property_from_value(self):
        """Test type property from static value."""
        arg = TestArg("count", value=42)
        assert arg.type == int

        arg_float = TestArg("ratio", value=3.14)
        assert arg_float.type == float

    def test_type_property_from_directed_values(self):
        """Test type property from directed values."""
        arg = TestArg("count", directed_values=[1, 2, 3])
        assert arg.type == int

    def test_is_static_property(self):
        """Test is_static property."""
        static_arg = TestArg("count", value=42)
        assert static_arg.is_static

        random_arg = TestArg("count", rng_type=RNGInteger(0, 100))
        assert not random_arg.is_static

    def test_has_directed_values_property(self):
        """Test has_directed_values property."""
        arg_with = TestArg("count", rng_type=RNGInteger(0, 100), directed_values=[0, 1])
        assert arg_with.has_directed_values

        arg_without = TestArg("count", rng_type=RNGInteger(0, 100))
        assert not arg_without.has_directed_values

    def test_rng_type_property(self):
        """Test rng_type property."""
        rng = RNGInteger(0, 100)
        arg = TestArg("count", rng_type=rng)
        assert arg.rng_type is rng

    def test_directed_values_property(self):
        """Test directed_values property."""
        values = [0, 1, 99, 100]
        arg = TestArg("count", rng_type=RNGInteger(0, 100), directed_values=values)
        assert arg.directed_values == values


# ============================================================================
# STRING REPRESENTATION TESTS
# ============================================================================

class TestTestArgStringRepresentation:
    """Test TestArg string representations."""

    def test_repr_static_value(self):
        """Test __repr__ for static value."""
        arg = TestArg("count", value=42)
        repr_str = repr(arg)
        assert "count" in repr_str
        assert "42" in repr_str

    def test_repr_random_type(self):
        """Test __repr__ for random type."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100))
        repr_str = repr(arg)
        assert "count" in repr_str
        assert "int" in repr_str

    def test_repr_with_directed_values(self):
        """Test __repr__ with directed values."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100), directed_values=[0, 1, 2])
        repr_str = repr(arg)
        assert "count" in repr_str
        assert "directed=3" in repr_str

    def test_str_with_description(self):
        """Test __str__ with description."""
        arg = TestArg("count", value=10, description="Number of items")
        str_repr = str(arg)
        assert "count" in str_repr
        assert "Number of items" in str_repr

    def test_str_without_description(self):
        """Test __str__ without description."""
        arg = TestArg("count", value=10)
        str_repr = str(arg)
        assert str_repr == "count"


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestTestArgEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directed_values_list(self):
        """Test with empty directed values list."""
        arg = TestArg("count", rng_type=RNGInteger(0, 100), directed_values=[])
        assert not arg.has_directed_values
        samples = arg.generate_samples(5)
        assert len(samples) == 5

    def test_different_rng_types(self):
        """Test with different RNG types."""
        # Integer
        arg_int = TestArg("count", rng_type=RNGInteger(0, 10))
        assert arg_int.type == int

        # Float
        arg_float = TestArg("ratio", rng_type=RNGFloat(0.0, 1.0))
        assert arg_float.type == float

        # Boolean
        arg_bool = TestArg("flag", rng_type=RNGBoolean())
        assert arg_bool.type == bool

        # Choice
        arg_choice = TestArg("mode", rng_type=RNGChoice(["fast", "slow"]))
        assert arg_choice.type == str

        # String
        arg_string = TestArg("name", rng_type=RNGString(length=10))
        assert arg_string.type == str

    def test_none_value_with_directed_values(self):
        """Test that None value with directed values works."""
        arg = TestArg("count", directed_values=[1, 2, 3])
        assert arg.has_directed_values
        samples = arg.generate_samples(0)
        assert samples == [1, 2, 3]

    def test_mixed_type_directed_values(self):
        """Test directed values with mixed types."""
        arg = TestArg("value", directed_values=[1, 2.5, "three"])
        # Type should be from first element
        assert arg.type == int

    def test_large_sample_generation(self):
        """Test generating large number of samples."""
        arg = TestArg("count", rng_type=RNGInteger(0, 1000))
        samples = arg.generate_samples(1000)
        assert len(samples) == 1000
        assert all(isinstance(v, int) for v in samples)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTestArgIntegration:
    """Integration tests for TestArg with various scenarios."""

    def test_complete_workflow_static(self):
        """Test complete workflow with static value."""
        arg = TestArg(
            "timeout",
            value=30,
            description="Connection timeout in seconds"
        )
        
        assert arg.name == "timeout"
        assert arg.is_static
        assert arg.type == int
        assert arg.generate() == 30
        assert arg.generate_samples(10) == [30]

    def test_complete_workflow_random(self):
        """Test complete workflow with random generation."""
        arg = TestArg(
            "port",
            rng_type=RNGInteger(1024, 65535),
            description="Network port"
        )
        
        assert arg.name == "port"
        assert not arg.is_static
        assert arg.type == int
        
        value = arg.generate()
        assert 1024 <= value <= 65535
        
        samples = arg.generate_samples(10)
        assert len(samples) == 10
        assert all(1024 <= v <= 65535 for v in samples)

    def test_complete_workflow_mixed(self):
        """Test complete workflow with mixed mode."""
        arg = TestArg(
            "count",
            rng_type=RNGInteger(1, 100),
            directed_values=[0, 1, 100],
            description="Item count with edge cases",
            always_include_directed=True
        )
        
        assert arg.name == "count"
        assert not arg.is_static
        assert arg.has_directed_values
        assert arg.type == int
        
        samples = arg.generate_samples(5)
        assert len(samples) == 8  # 3 directed + 5 random
        assert samples[0] == 0
        assert samples[1] == 1
        assert samples[2] == 100

    def test_workflow_with_validation(self):
        """Test workflow with validation."""
        arg = TestArg(
            "even_number",
            rng_type=RNGInteger(0, 100, predicate=lambda x: x % 2 == 0),
            validator=lambda x: x % 2 == 0,
            description="Even numbers only"
        )
        
        # Generate should produce even numbers
        for _ in range(10):
            value = arg.generate()
            assert value % 2 == 0
        
        samples = arg.generate_samples(10)
        assert all(v % 2 == 0 for v in samples)
