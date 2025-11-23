"""
Unit tests for RNGEnum class.

Tests cover:
- Basic enum generation
- Weighted enum selection
- Predicate filtering
- Combined weights and predicates
- Error handling
- Edge cases
"""

import pytest
from enum import Enum
from pytest_strategy import RNG, RNGEnum, RNGValueError


# Test Enums
class Status(Enum):
    """Example enum for status values"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class Priority(Enum):
    """Example enum for priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Color(Enum):
    """Example enum for colors"""
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"


class TestRNGEnumBasic:
    """Test basic RNGEnum functionality"""

    def test_initialization_with_enum_class(self):
        """Test that RNGEnum can be initialized with an Enum class"""
        rng_enum = RNGEnum(Status)
        assert rng_enum.enum_class == Status
        assert rng_enum.weights is None
        assert rng_enum.predicate is None

    def test_initialization_with_non_enum_raises_error(self):
        """Test that initializing with non-Enum class raises error"""
        with pytest.raises(RNGValueError, match="is not an Enum class"):
            RNGEnum(str)

    def test_python_type_property(self):
        """Test that python_type returns the enum class"""
        rng_enum = RNGEnum(Priority)
        assert rng_enum.python_type == Priority

    def test_generate_returns_enum_member(self):
        """Test that generate returns a member of the enum"""
        RNG.seed(42)
        rng_enum = RNGEnum(Status)
        value = rng_enum.generate()
        assert isinstance(value, Status)
        assert value in Status

    def test_generate_uniform_distribution(self):
        """Test that uniform generation covers all members"""
        RNG.seed(42)
        rng_enum = RNGEnum(Status)
        
        # Generate many samples
        samples = [rng_enum.generate() for _ in range(100)]
        
        # Should have all members represented
        unique_values = set(samples)
        assert len(unique_values) >= 3  # At least 3 out of 4 members


class TestRNGEnumWeighted:
    """Test weighted enum selection"""

    def test_weighted_selection(self):
        """Test that weighted selection works"""
        RNG.seed(42)
        rng_enum = RNGEnum(Status, weights={
            Status.SUCCESS: 0.9,
            Status.FAILED: 0.1
        })
        
        # Generate many samples
        samples = [rng_enum.generate() for _ in range(100)]
        
        # Should only contain weighted members
        unique_values = set(samples)
        assert unique_values.issubset({Status.SUCCESS, Status.FAILED})
        
        # SUCCESS should be much more common
        success_count = samples.count(Status.SUCCESS)
        assert success_count > 70  # Should be around 90

    def test_weighted_with_all_members(self):
        """Test weighted selection with all enum members"""
        RNG.seed(42)
        rng_enum = RNGEnum(Priority, weights={
            Priority.LOW: 0.1,
            Priority.MEDIUM: 0.2,
            Priority.HIGH: 0.4,
            Priority.CRITICAL: 0.3
        })
        
        samples = [rng_enum.generate() for _ in range(200)]
        unique_values = set(samples)
        
        # Should have all members
        assert len(unique_values) == 4

    def test_weights_dont_need_to_sum_to_one(self):
        """Test that weights are normalized automatically"""
        RNG.seed(42)
        rng_enum = RNGEnum(Status, weights={
            Status.SUCCESS: 90,
            Status.FAILED: 10
        })
        
        # Should work fine (weights will be normalized)
        value = rng_enum.generate()
        assert value in {Status.SUCCESS, Status.FAILED}

    def test_invalid_weight_member_raises_error(self):
        """Test that invalid enum member in weights raises error"""
        with pytest.raises(RNGValueError, match="is not a member of"):
            RNGEnum(Status, weights={
                Status.SUCCESS: 0.5,
                Priority.HIGH: 0.5  # Wrong enum!
            })


class TestRNGEnumPredicate:
    """Test predicate filtering"""

    def test_predicate_filtering(self):
        """Test that predicate filters values correctly"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Status,
            predicate=lambda s: s != Status.ERROR
        )
        
        # Generate many samples
        samples = [rng_enum.generate() for _ in range(50)]
        
        # Should never contain ERROR
        assert Status.ERROR not in samples
        assert all(s != Status.ERROR for s in samples)

    def test_predicate_with_multiple_exclusions(self):
        """Test predicate that excludes multiple values"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Priority,
            predicate=lambda p: p not in {Priority.LOW, Priority.CRITICAL}
        )
        
        samples = [rng_enum.generate() for _ in range(50)]
        
        # Should only contain MEDIUM and HIGH
        unique_values = set(samples)
        assert unique_values.issubset({Priority.MEDIUM, Priority.HIGH})

    def test_predicate_impossible_raises_error(self):
        """Test that impossible predicate raises error after retries"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Status,
            predicate=lambda s: False  # Impossible condition
        )
        
        with pytest.raises(RNGValueError, match="No valid value found"):
            rng_enum.generate()


class TestRNGEnumWeightedWithPredicate:
    """Test combined weighted selection and predicate filtering"""

    def test_weighted_with_predicate(self):
        """Test that weights and predicate work together"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Priority,
            weights={
                Priority.HIGH: 0.6,
                Priority.MEDIUM: 0.3,
                Priority.LOW: 0.1
            },
            predicate=lambda p: p != Priority.LOW
        )
        
        samples = [rng_enum.generate() for _ in range(100)]
        
        # Should not contain LOW (filtered by predicate)
        assert Priority.LOW not in samples
        
        # Should contain HIGH and MEDIUM
        unique_values = set(samples)
        assert unique_values.issubset({Priority.HIGH, Priority.MEDIUM})
        
        # HIGH should be more common than MEDIUM
        high_count = samples.count(Priority.HIGH)
        medium_count = samples.count(Priority.MEDIUM)
        assert high_count > medium_count

    def test_weighted_predicate_filters_weighted_members(self):
        """Test predicate can filter out weighted members"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Status,
            weights={
                Status.SUCCESS: 0.5,
                Status.FAILED: 0.3,
                Status.ERROR: 0.2
            },
            predicate=lambda s: s != Status.ERROR
        )
        
        samples = [rng_enum.generate() for _ in range(50)]
        
        # ERROR should be filtered out despite being in weights
        assert Status.ERROR not in samples
        assert set(samples).issubset({Status.SUCCESS, Status.FAILED})


class TestRNGEnumReproducibility:
    """Test seed-based reproducibility"""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results"""
        rng_enum = RNGEnum(Status)
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = [rng_enum.generate() for _ in range(20)]
        
        # Generate with same seed
        RNG.seed(42)
        samples2 = [rng_enum.generate() for _ in range(20)]
        
        # Should be identical
        assert samples1 == samples2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        rng_enum = RNGEnum(Priority)
        
        # Generate with seed 42
        RNG.seed(42)
        samples1 = [rng_enum.generate() for _ in range(20)]
        
        # Generate with different seed
        RNG.seed(99)
        samples2 = [rng_enum.generate() for _ in range(20)]
        
        # Should be different
        assert samples1 != samples2


class TestRNGEnumEdgeCases:
    """Test edge cases and special scenarios"""

    def test_single_member_enum(self):
        """Test enum with only one member"""
        class SingleValue(Enum):
            ONLY = "only"
        
        rng_enum = RNGEnum(SingleValue)
        value = rng_enum.generate()
        assert value == SingleValue.ONLY

    def test_enum_with_mixed_value_types(self):
        """Test enum with different value types"""
        class MixedEnum(Enum):
            INT_VAL = 1
            STR_VAL = "string"
            FLOAT_VAL = 3.14
        
        RNG.seed(42)
        rng_enum = RNGEnum(MixedEnum)
        
        samples = [rng_enum.generate() for _ in range(30)]
        unique_values = set(samples)
        
        # Should cover all members
        assert len(unique_values) >= 2

    def test_weighted_single_member(self):
        """Test weighted selection with single member"""
        rng_enum = RNGEnum(Status, weights={Status.SUCCESS: 1.0})
        
        value = rng_enum.generate()
        assert value == Status.SUCCESS

    def test_predicate_allows_single_value(self):
        """Test predicate that allows only one value"""
        RNG.seed(42)
        rng_enum = RNGEnum(
            Priority,
            predicate=lambda p: p == Priority.HIGH
        )
        
        samples = [rng_enum.generate() for _ in range(10)]
        assert all(s == Priority.HIGH for s in samples)


class TestRNGEnumIntegration:
    """Test RNGEnum integration with other components"""

    def test_with_test_arg(self):
        """Test RNGEnum works with TestArg"""
        from pytest_strategy import TestArg
        
        arg = TestArg("status", rng_type=RNGEnum(Status))
        
        RNG.seed(42)
        value = arg.generate()
        assert isinstance(value, Status)

    def test_with_parameter(self):
        """Test RNGEnum works with Parameter"""
        from pytest_strategy import Parameter, TestArg
        
        param = Parameter(
            TestArg("status", rng_type=RNGEnum(Status)),
            TestArg("priority", rng_type=RNGEnum(Priority))
        )
        
        RNG.seed(42)
        samples = param.generate_samples(10, mode="random_only")
        
        assert len(samples) == 10
        for status, priority in samples:
            assert isinstance(status, Status)
            assert isinstance(priority, Priority)

    def test_weighted_enum_with_parameter(self):
        """Test weighted RNGEnum in Parameter"""
        from pytest_strategy import Parameter, TestArg
        
        param = Parameter(
            TestArg("status", rng_type=RNGEnum(
                Status,
                weights={Status.SUCCESS: 0.8, Status.FAILED: 0.2}
            )),
            TestArg("priority", rng_type=RNGEnum(
                Priority,
                predicate=lambda p: p != Priority.LOW
            ))
        )
        
        RNG.seed(42)
        samples = param.generate_samples(20, mode="random_only")
        
        assert len(samples) == 20
        for status, priority in samples:
            assert status in {Status.SUCCESS, Status.FAILED}
            assert priority != Priority.LOW
