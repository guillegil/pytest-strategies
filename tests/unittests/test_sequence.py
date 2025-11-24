"""
Unit tests for Sequence Testing feature.
"""

import pytest
from pytest_strategy import Parameter, TestArg, RNGInteger
from pytest_strategy.rng import RNGSequence, RNGValueError, RNG

class TestRNGSequence:
    """Test RNGSequence class."""

    def test_initialization(self):
        """Test initialization with sequence."""
        seq = RNGSequence([1, 2, 3])
        assert seq.sequence == [1, 2, 3]
        assert seq.python_type == int

    def test_initialization_empty_raises_error(self):
        """Test that empty sequence raises error."""
        with pytest.raises(RNGValueError, match="Sequence cannot be empty"):
            RNGSequence([])

    def test_initialization_with_predicate(self):
        """Test initialization with predicate filtering."""
        seq = RNGSequence([1, 2, 3, 4, 5], predicate=lambda x: x % 2 == 0)
        assert seq.sequence == [2, 4]
        
    def test_initialization_all_filtered_raises_error(self):
        """Test that error is raised if predicate filters all items."""
        with pytest.raises(RNGValueError, match="filtered by predicate"):
            RNGSequence([1, 3, 5], predicate=lambda x: x % 2 == 0)

    def test_generate_random(self):
        """Test random generation (normal mode)."""
        RNG.seed(42)
        seq = RNGSequence([1, 2, 3])
        value = seq.generate()
        assert value in [1, 2, 3]

class TestParameterExhaustive:
    """Test Parameter.generate_exhaustive method."""

    def test_single_sequence(self):
        """Test exhaustive generation with single sequence."""
        param = Parameter(
            TestArg("item", rng_type=RNGSequence([1, 2, 3]))
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 3
        assert samples == [(1,), (2,), (3,)]

    def test_multiple_sequences_cartesian_product(self):
        """Test Cartesian product of multiple sequences."""
        param = Parameter(
            TestArg("a", rng_type=RNGSequence([1, 2])),
            TestArg("b", rng_type=RNGSequence(["x", "y"]))
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 4
        expected = [(1, "x"), (1, "y"), (2, "x"), (2, "y")]
        assert set(samples) == set(expected)

    def test_mixed_sequence_and_random(self):
        """Test mixed sequence and random arguments."""
        RNG.seed(42)
        param = Parameter(
            TestArg("seq", rng_type=RNGSequence([1, 2])),
            TestArg("rnd", rng_type=RNGInteger(10, 20))
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 2
        
        # Check sequence values
        assert samples[0][0] == 1
        assert samples[1][0] == 2
        
        # Check random values
        assert 10 <= samples[0][1] <= 20
        assert 10 <= samples[1][1] <= 20

    def test_no_sequence_raises_error(self):
        """Test that exhaustive generation raises error if no sequence args."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10))
        )
        with pytest.raises(ValueError, match="No sequence arguments found"):
            param.generate_exhaustive()

    def test_with_constraints(self):
        """Test exhaustive generation with constraints."""
        param = Parameter(
            TestArg("a", rng_type=RNGSequence([1, 2, 3])),
            TestArg("b", rng_type=RNGSequence([1, 2, 3])),
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        samples = param.generate_exhaustive()
        # Expected: (1,2), (1,3), (2,3)
        assert len(samples) == 3
        assert all(s[0] < s[1] for s in samples)

    def test_exhaustive_with_predicate_sequence(self):
        """Test exhaustive generation with filtered sequence."""
        param = Parameter(
            TestArg("x", rng_type=RNGSequence([1, 2, 3, 4], predicate=lambda x: x % 2 == 0)),
            TestArg("y", rng_type=RNGSequence(["a", "b"]))
        )
        samples = param.generate_exhaustive()
        # x should only be [2, 4]
        # y is ["a", "b"]
        # Expected: (2, "a"), (2, "b"), (4, "a"), (4, "b")
        assert len(samples) == 4
        assert set(s[0] for s in samples) == {2, 4}
