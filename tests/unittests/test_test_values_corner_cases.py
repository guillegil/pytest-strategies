"""
Corner case tests for test_values feature.
"""

import pytest
from pytest_strategy import Parameter, TestArg, RNGInteger


class TestTestValuesCornerCases:
    """Test corner cases for test_values feature."""

    def test_empty_test_vectors(self):
        """Test that empty test_vectors dict works."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={}
        )
        vectors = param.generate_vectors(n=5, mode="test")
        assert len(vectors) == 0

    def test_test_vectors_with_constraints(self):
        """Test that test vectors respect constraints."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "valid": (5, 5),
                "invalid": (10, 1)  # This violates x < y constraint
            },
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        # Test vectors are returned as-is, constraints are not applied in generate_vectors
        # Constraints are only applied during generate_vector() for random generation
        vectors = param.generate_vectors(n=0, mode="test")
        assert len(vectors) == 2

    def test_test_values_only_no_rng_type(self):
        """Test TestArg with only test_values and no rng_type."""
        arg = TestArg("x", test_values=[1, 2, 3])
        assert arg.test_values == [1, 2, 3]
        assert arg.rng_type is None
        
        # Should raise error when trying to generate since no rng_type
        with pytest.raises(ValueError, match="Cannot generate value"):
            arg.generate()

    def test_mixed_test_and_directed_vectors(self):
        """Test Parameter with both test and directed vectors."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            directed_vectors={"dir1": (1,), "dir2": (2,)},
            test_vectors={"test1": (99,), "test2": (100,)}
        )
        
        # Test mode should only return test vectors
        test_vectors = param.generate_vectors(n=10, mode="test")
        assert len(test_vectors) == 2
        assert (99,) in test_vectors
        assert (100,) in test_vectors
        assert (1,) not in test_vectors
        assert (2,) not in test_vectors
        
        # Directed mode should only return directed vectors
        directed_vectors = param.generate_vectors(n=10, mode="directed_only")
        assert len(directed_vectors) == 2
        assert (1,) in directed_vectors
        assert (2,) in directed_vectors
        assert (99,) not in directed_vectors
        assert (100,) not in directed_vectors

    def test_test_vector_duplicate_names(self):
        """Test that duplicate test vector names overwrite."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={"test1": (5,)}
        )
        
        # Add another vector with same name
        param.add_test_vector("test1", (10,))
        
        # Should have only one vector with the new value
        assert len(param.test_vectors) == 1
        assert param.test_vectors["test1"] == (10,)

    def test_test_vectors_with_none_values(self):
        """Test that test vectors can contain None values."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "with_none": (None, 5),
                "both_none": (None, None)
            }
        )
        vectors = param.generate_vectors(n=0, mode="test")
        assert len(vectors) == 2
        assert (None, 5) in vectors
        assert (None, None) in vectors

    def test_large_number_of_test_vectors(self):
        """Test with a large number of test vectors."""
        test_vectors = {f"test_{i}": (i,) for i in range(1000)}
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors=test_vectors
        )
        vectors = param.generate_vectors(n=0, mode="test")
        assert len(vectors) == 1000

    def test_test_vectors_with_special_characters_in_names(self):
        """Test that test vector names can have special characters."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "test-with-dashes": (1,),
                "test_with_underscores": (2,),
                "test.with.dots": (3,),
                "test with spaces": (4,)
            }
        )
        assert len(param.test_vectors) == 4
        assert param.get_test_vector("test-with-dashes") == (1,)
        assert param.get_test_vector("test with spaces") == (4,)

    def test_test_vectors_order_preservation(self):
        """Test that test vectors maintain insertion order (Python 3.7+)."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "first": (1,),
                "second": (2,),
                "third": (3,)
            }
        )
        # Dict maintains insertion order in Python 3.7+
        vector_names = list(param.test_vectors.keys())
        assert vector_names == ["first", "second", "third"]

    def test_single_arg_multiple_test_vectors(self):
        """Test single argument with multiple test vectors."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "min": (0,),
                "mid": (5,),
                "max": (10,)
            }
        )
        vectors = param.generate_vectors(n=0, mode="test")
        assert len(vectors) == 3
        assert (0,) in vectors
        assert (5,) in vectors
        assert (10,) in vectors

    def test_test_vectors_with_complex_types(self):
        """Test test vectors with complex Python types."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "with_list": ([1, 2, 3], 5),
                "with_dict": ({"key": "value"}, 10),
                "with_tuple": ((1, 2), 15)
            }
        )
        vectors = param.generate_vectors(n=0, mode="test")
        assert len(vectors) == 3
        assert ([1, 2, 3], 5) in vectors
        assert ({"key": "value"}, 10) in vectors
        assert ((1, 2), 15) in vectors
