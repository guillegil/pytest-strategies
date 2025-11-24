"""
Unit tests for test_values feature.
"""

import pytest
from pytest_strategy import Parameter, TestArg, RNGInteger
from pytest_strategy.rng import RNG

class TestTestArgWithTestValues:
    """Test TestArg with test_values parameter."""

    def test_initialization_with_test_values(self):
        """Test initialization with test values."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10), test_values=[5, 10, 15])
        assert arg.test_values == [5, 10, 15]

    def test_initialization_only_test_values(self):
        """Test initialization with only test values (no rng_type)."""
        arg = TestArg("x", test_values=[1, 2, 3])
        assert arg.test_values == [1, 2, 3]
        assert arg.rng_type is None

    def test_to_dict_includes_test_values(self):
        """Test that to_dict includes test values."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10), test_values=[5, 10])
        data = arg.to_dict()
        assert data["has_test_values"] is True

    def test_to_dict_no_test_values(self):
        """Test that to_dict correctly reports no test values."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        data = arg.to_dict()
        assert data["has_test_values"] is False


class TestParameterWithTestVectors:
    """Test Parameter with test_vectors parameter."""

    def test_initialization_with_test_vectors(self):
        """Test initialization with test vectors."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "test1": (5, 5),
                "test2": (10, 10)
            }
        )
        assert len(param.test_vectors) == 2
        assert param.test_vectors["test1"] == (5, 5)

    def test_validate_test_vectors_wrong_length(self):
        """Test that validation catches wrong length test vectors."""
        with pytest.raises(ValueError, match="Test vector 'bad' has 1 values, expected 2"):
            Parameter(
                TestArg("x", rng_type=RNGInteger(0, 10)),
                TestArg("y", rng_type=RNGInteger(0, 10)),
                test_vectors={"bad": (5,)}
            )

    def test_generate_vectors_test_mode(self):
        """Test generate_vectors with mode='test'."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            test_vectors={
                "test1": (5, 5),
                "test2": (10, 10)
            }
        )
        vectors = param.generate_vectors(n=100, mode="test")
        assert len(vectors) == 2
        assert (5, 5) in vectors
        assert (10, 10) in vectors

    def test_generate_vectors_test_mode_ignores_directed(self):
        """Test that test mode ignores directed vectors."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            directed_vectors={"dir1": (1, 1)},
            test_vectors={"test1": (5, 5)}
        )
        vectors = param.generate_vectors(n=100, mode="test")
        assert len(vectors) == 1
        assert (5, 5) in vectors
        assert (1, 1) not in vectors

    def test_add_test_vector(self):
        """Test adding a test vector."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10))
        )
        param.add_test_vector("test1", (5, 5))
        assert param.test_vectors["test1"] == (5, 5)

    def test_add_test_vector_wrong_length(self):
        """Test that adding wrong length test vector raises error."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10))
        )
        with pytest.raises(ValueError, match="Vector must have 2 values, got 1"):
            param.add_test_vector("bad", (5,))

    def test_remove_test_vector(self):
        """Test removing a test vector."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={"test1": (5,)}
        )
        param.remove_test_vector("test1")
        assert "test1" not in param.test_vectors

    def test_remove_test_vector_not_found(self):
        """Test removing non-existent test vector raises error."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10))
        )
        with pytest.raises(KeyError, match="No test vector named 'missing'"):
            param.remove_test_vector("missing")

    def test_get_test_vector(self):
        """Test getting a test vector."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={"test1": (5,)}
        )
        vector = param.get_test_vector("test1")
        assert vector == (5,)

    def test_get_test_vector_not_found(self):
        """Test getting non-existent test vector raises error."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10))
        )
        with pytest.raises(KeyError, match="No test vector named 'missing'"):
            param.get_test_vector("missing")

    def test_to_dict_includes_test_vectors(self):
        """Test that to_dict includes test vectors."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            test_vectors={"test1": (5,)}
        )
        data = param.to_dict()
        assert "test_vectors" in data
        assert "test1" in data["test_vectors"]
