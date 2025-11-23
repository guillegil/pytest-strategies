"""
Unit tests for Parameter module.

Tests the Parameter class to ensure proper initialization, directed vector management,
sample generation with different modes, CLI support, constraint handling, and introspection.
"""

import pytest
from pytest_strategy import RNGInteger, RNGFloat, RNGChoice, RNGBoolean
from pytest_strategy.test_args import TestArg
from pytest_strategy.parameters import Parameter


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestParameterInitialization:
    """Test Parameter initialization."""

    def test_basic_initialization(self):
        """Test creating Parameter with test args."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        assert param.num_args == 2
        assert param.arg_names == ("x", "y")

    def test_initialization_with_directed_vectors(self):
        """Test creating Parameter with directed vectors."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "origin": (0, 0),
                "max": (10, 10),
            }
        )
        
        assert param.num_directed_vectors == 2
        assert "origin" in param.vector_names
        assert "max" in param.vector_names

    def test_initialization_with_constraints(self):
        """Test creating Parameter with vector constraints."""
        arg1 = TestArg("min", rng_type=RNGInteger(0, 100))
        arg2 = TestArg("max", rng_type=RNGInteger(0, 100))
        
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        
        assert len(param.vector_constraints) == 1

    def test_initialization_always_include_directed_flag(self):
        """Test always_include_directed flag."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        
        param_true = Parameter(arg, always_include_directed=True)
        assert param_true.always_include_directed is True
        
        param_false = Parameter(arg, always_include_directed=False)
        assert param_false.always_include_directed is False

    def test_initialization_invalid_directed_vector_length(self):
        """Test that directed vectors with wrong length raise error."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        
        with pytest.raises(ValueError, match="has 3 values, expected 2"):
            Parameter(
                arg1, arg2,
                directed_vectors={"invalid": (1, 2, 3)}
            )

    def test_initialization_empty_args(self):
        """Test creating Parameter with no args."""
        param = Parameter()
        assert param.num_args == 0
        assert param.arg_names == ()


# ============================================================================
# DIRECTED VECTOR MANAGEMENT TESTS
# ============================================================================

class TestParameterDirectedVectors:
    """Test directed vector management methods."""

    def test_add_directed_vector(self):
        """Test adding a directed vector."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        param.add_directed_vector("custom", (5, 5))
        assert param.num_directed_vectors == 1
        assert param.get_directed_vector("custom") == (5, 5)

    def test_add_directed_vector_wrong_length(self):
        """Test adding directed vector with wrong length raises error."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        with pytest.raises(ValueError, match="must have 1 values, got 2"):
            param.add_directed_vector("invalid", (1, 2))

    def test_remove_directed_vector(self):
        """Test removing a directed vector."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"vec1": (1,), "vec2": (2,)}
        )
        
        param.remove_directed_vector("vec1")
        assert param.num_directed_vectors == 1
        assert "vec1" not in param.vector_names

    def test_remove_nonexistent_vector_raises_error(self):
        """Test removing non-existent vector raises KeyError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        with pytest.raises(KeyError, match="No directed vector named"):
            param.remove_directed_vector("nonexistent")

    def test_get_directed_vector(self):
        """Test getting a directed vector by name."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg1, arg2,
            directed_vectors={"origin": (0, 0)}
        )
        
        vector = param.get_directed_vector("origin")
        assert vector == (0, 0)

    def test_get_nonexistent_vector_raises_error(self):
        """Test getting non-existent vector raises KeyError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        with pytest.raises(KeyError, match="No directed vector named"):
            param.get_directed_vector("nonexistent")


# ============================================================================
# VECTOR GENERATION TESTS
# ============================================================================

class TestParameterVectorGeneration:
    """Test parameter vector generation."""

    def test_generate_single_vector(self):
        """Test generating a single random vector."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        vector = param.generate_vector()
        assert isinstance(vector, tuple)
        assert len(vector) == 2
        assert all(0 <= v <= 10 for v in vector)

    def test_generate_vector_with_constraints(self):
        """Test that generated vectors satisfy constraints."""
        arg1 = TestArg("min", rng_type=RNGInteger(0, 50))
        arg2 = TestArg("max", rng_type=RNGInteger(50, 100))
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        
        for _ in range(10):
            vector = param.generate_vector()
            assert vector[0] < vector[1]

    def test_generate_vector_impossible_constraints_raises_error(self):
        """Test that impossible constraints raise error."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            vector_constraints=[lambda v: v[0] > 100]  # Impossible
        )
        
        with pytest.raises(ValueError, match="Could not generate valid vector"):
            param.generate_vector()


# ============================================================================
# SAMPLE GENERATION TESTS
# ============================================================================

class TestParameterSampleGeneration:
    """Test parameter sample generation with different modes."""

    def test_generate_samples_all_mode(self):
        """Test 'all' mode includes all directed vectors + random samples."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "five": (5,), "ten": (10,)}
        )
        
        samples = param.generate_samples(5, mode="all")
        assert len(samples) == 8  # 3 directed + 5 random

    def test_generate_samples_random_only_mode(self):
        """Test 'random_only' mode excludes directed vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)}
        )
        
        samples = param.generate_samples(5, mode="random_only")
        assert len(samples) == 5  # Only random, no directed

    def test_generate_samples_directed_only_mode(self):
        """Test 'directed_only' mode returns only directed vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "five": (5,), "ten": (10,)}
        )
        
        samples = param.generate_samples(100, mode="directed_only")
        assert len(samples) == 3  # Only directed, ignores n

    def test_generate_samples_mixed_mode_with_flag_true(self):
        """Test 'mixed' mode with always_include_directed=True."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)},
            always_include_directed=True
        )
        
        samples = param.generate_samples(5, mode="mixed")
        assert len(samples) == 7  # 2 directed + 5 random

    def test_generate_samples_mixed_mode_with_flag_false(self):
        """Test 'mixed' mode with always_include_directed=False."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)},
            always_include_directed=False
        )
        
        samples = param.generate_samples(5, mode="mixed")
        assert len(samples) == 5  # Only random, no directed

    def test_generate_samples_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        with pytest.raises(ValueError, match="Invalid mode"):
            param.generate_samples(5, mode="invalid_mode")

    def test_generate_samples_zero_count(self):
        """Test generating zero samples."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        samples = param.generate_samples(0, mode="random_only")
        assert samples == []


# ============================================================================
# CLI SUPPORT TESTS
# ============================================================================

class TestParameterCLISupport:
    """Test CLI support methods (filter by name/index)."""

    def test_filter_by_name(self):
        """Test filtering samples by vector name."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={
                "zero": (0,),
                "five": (5,),
                "ten": (10,)
            }
        )
        
        samples = param.generate_samples(100, filter_by_name="five")
        assert len(samples) == 1
        assert samples[0] == (5,)

    def test_filter_by_name_nonexistent_raises_error(self):
        """Test filtering by non-existent name raises KeyError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,)}
        )
        
        with pytest.raises(KeyError, match="No directed vector named"):
            param.generate_samples(5, filter_by_name="nonexistent")

    def test_filter_by_index(self):
        """Test filtering samples by vector index."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={
                "zero": (0,),
                "five": (5,),
                "ten": (10,)
            }
        )
        
        samples = param.generate_samples(100, filter_by_index=1)
        assert len(samples) == 1
        # Index 1 should be "five" (second in order)
        assert samples[0] == (5,)

    def test_filter_by_index_out_of_range_raises_error(self):
        """Test filtering by out-of-range index raises IndexError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)}
        )
        
        with pytest.raises(IndexError, match="out of range"):
            param.generate_samples(5, filter_by_index=5)

    def test_get_vector_by_name(self):
        """Test get_vector_by_name method."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)}
        )
        
        vector = param.get_vector_by_name("zero")
        assert vector == (0,)

    def test_get_vector_by_index(self):
        """Test get_vector_by_index method."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "ten": (10,)}
        )
        
        vector = param.get_vector_by_index(0)
        assert vector == (0,)

    def test_list_vector_names(self):
        """Test list_vector_names method."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"zero": (0,), "five": (5,), "ten": (10,)}
        )
        
        names = param.list_vector_names()
        assert names == ["zero", "five", "ten"]


# ============================================================================
# CONSTRAINT MANAGEMENT TESTS
# ============================================================================

class TestParameterConstraints:
    """Test constraint management methods."""

    def test_add_constraint(self):
        """Test adding a constraint."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        param.add_constraint(lambda v: v[0] < v[1])
        assert len(param.vector_constraints) == 1

    def test_multiple_constraints(self):
        """Test adding multiple constraints."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        param.add_constraint(lambda v: v[0] < v[1])
        param.add_constraint(lambda v: v[0] + v[1] <= 15)
        
        assert len(param.vector_constraints) == 2
        
        # Generate vectors should satisfy both constraints
        for _ in range(10):
            vector = param.generate_vector()
            assert vector[0] < vector[1]
            assert vector[0] + vector[1] <= 15

    def test_clear_constraints(self):
        """Test clearing all constraints."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            vector_constraints=[lambda v: v[0] > 5]
        )
        
        assert len(param.vector_constraints) == 1
        param.clear_constraints()
        assert len(param.vector_constraints) == 0


# ============================================================================
# INTROSPECTION TESTS
# ============================================================================

class TestParameterIntrospection:
    """Test Parameter introspection properties and methods."""

    def test_arg_names_property(self):
        """Test arg_names property."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        arg3 = TestArg("z", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2, arg3)
        
        assert param.arg_names == ("x", "y", "z")

    def test_arg_types_property(self):
        """Test arg_types property."""
        arg1 = TestArg("count", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("ratio", rng_type=RNGFloat(0.0, 1.0))
        arg3 = TestArg("flag", rng_type=RNGBoolean())
        param = Parameter(arg1, arg2, arg3)
        
        assert param.arg_types == (int, float, bool)

    def test_vector_names_property(self):
        """Test vector_names property."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"a": (1,), "b": (2,), "c": (3,)}
        )
        
        assert param.vector_names == ["a", "b", "c"]

    def test_num_args_property(self):
        """Test num_args property."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        assert param.num_args == 2

    def test_num_directed_vectors_property(self):
        """Test num_directed_vectors property."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg,
            directed_vectors={"a": (1,), "b": (2,)}
        )
        
        assert param.num_directed_vectors == 2

    def test_get_arg_by_name(self):
        """Test getting TestArg by name."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(arg1, arg2)
        
        retrieved = param.get_arg("x")
        assert retrieved is arg1

    def test_get_arg_nonexistent_raises_error(self):
        """Test getting non-existent arg raises KeyError."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        with pytest.raises(KeyError, match="No argument named"):
            param.get_arg("nonexistent")


# ============================================================================
# STRING REPRESENTATION TESTS
# ============================================================================

class TestParameterStringRepresentation:
    """Test Parameter string representations."""

    def test_repr(self):
        """Test __repr__ method."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg1, arg2,
            directed_vectors={"origin": (0, 0)}
        )
        
        repr_str = repr(param)
        assert "args=2" in repr_str
        assert "directed_vectors=1" in repr_str

    def test_str(self):
        """Test __str__ method."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        param = Parameter(
            arg1, arg2,
            directed_vectors={"origin": (0, 0), "max": (10, 10)}
        )
        
        str_repr = str(param)
        assert "x, y" in str_repr
        assert "origin" in str_repr
        assert "max" in str_repr


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestParameterEdgeCases:
    """Test edge cases and error handling."""

    def test_single_arg_parameter(self):
        """Test Parameter with single argument."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        assert param.num_args == 1
        vector = param.generate_vector()
        assert isinstance(vector, tuple)
        assert len(vector) == 1

    def test_many_args_parameter(self):
        """Test Parameter with many arguments."""
        args = [TestArg(f"arg{i}", rng_type=RNGInteger(0, 10)) for i in range(10)]
        param = Parameter(*args)
        
        assert param.num_args == 10
        vector = param.generate_vector()
        assert len(vector) == 10

    def test_no_directed_vectors(self):
        """Test Parameter with no directed vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        assert param.num_directed_vectors == 0
        assert param.vector_names == []
        
        samples = param.generate_samples(5, mode="directed_only")
        assert samples == []

    def test_many_directed_vectors(self):
        """Test Parameter with many directed vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 100))
        vectors = {f"vec{i}": (i,) for i in range(50)}
        param = Parameter(arg, directed_vectors=vectors)
        
        assert param.num_directed_vectors == 50
        samples = param.generate_samples(0, mode="directed_only")
        assert len(samples) == 50

    def test_complex_constraints(self):
        """Test complex constraint scenarios."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        arg3 = TestArg("z", rng_type=RNGInteger(0, 10))
        
        param = Parameter(
            arg1, arg2, arg3,
            vector_constraints=[
                lambda v: v[0] < v[1],
                lambda v: v[1] < v[2],
                lambda v: v[0] + v[1] + v[2] <= 20
            ]
        )
        
        for _ in range(10):
            vector = param.generate_vector()
            assert vector[0] < vector[1] < vector[2]
            assert sum(vector) <= 20


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestParameterIntegration:
    """Integration tests for Parameter with various scenarios."""

    def test_complete_workflow_basic(self):
        """Test complete workflow with basic parameter."""
        arg1 = TestArg("count", rng_type=RNGInteger(1, 100))
        arg2 = TestArg("timeout", rng_type=RNGFloat(0.1, 10.0))
        
        param = Parameter(arg1, arg2)
        
        # Generate single vector
        vector = param.generate_vector()
        assert len(vector) == 2
        assert isinstance(vector[0], int)
        assert isinstance(vector[1], float)
        
        # Generate samples
        samples = param.generate_samples(10, mode="random_only")
        assert len(samples) == 10

    def test_complete_workflow_with_directed_vectors(self):
        """Test complete workflow with directed vectors."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 100))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 100))
        
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "origin": (0, 0),
                "max": (100, 100),
                "mid": (50, 50)
            }
        )
        
        # All mode
        samples_all = param.generate_samples(5, mode="all")
        assert len(samples_all) == 8  # 3 directed + 5 random
        
        # Directed only
        samples_directed = param.generate_samples(0, mode="directed_only")
        assert len(samples_directed) == 3
        
        # Random only
        samples_random = param.generate_samples(5, mode="random_only")
        assert len(samples_random) == 5

    def test_complete_workflow_with_constraints(self):
        """Test complete workflow with constraints."""
        arg1 = TestArg("min", rng_type=RNGInteger(0, 50))
        arg2 = TestArg("max", rng_type=RNGInteger(50, 100))
        
        param = Parameter(
            arg1, arg2,
            vector_constraints=[lambda v: v[0] < v[1]]
        )
        
        samples = param.generate_samples(20, mode="random_only")
        assert len(samples) == 20
        assert all(s[0] < s[1] for s in samples)

    def test_cli_workflow(self):
        """Test CLI filtering workflow."""
        arg1 = TestArg("x", rng_type=RNGInteger(0, 10))
        arg2 = TestArg("y", rng_type=RNGInteger(0, 10))
        
        param = Parameter(
            arg1, arg2,
            directed_vectors={
                "test1": (1, 1),
                "test2": (2, 2),
                "test3": (3, 3)
            }
        )
        
        # Filter by name
        samples_name = param.generate_samples(0, filter_by_name="test2")
        assert len(samples_name) == 1
        assert samples_name[0] == (2, 2)
        
        # Filter by index
        samples_index = param.generate_samples(0, filter_by_index=0)
        assert len(samples_index) == 1
        assert samples_index[0] == (1, 1)
        
        # List names
        names = param.list_vector_names()
        assert names == ["test1", "test2", "test3"]

    def test_dynamic_modification_workflow(self):
        """Test dynamically modifying parameter."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        
        # Add directed vectors
        param.add_directed_vector("vec1", (1,))
        param.add_directed_vector("vec2", (2,))
        assert param.num_directed_vectors == 2
        
        # Add constraints
        param.add_constraint(lambda v: v[0] > 5)
        
        # Generate with constraint
        for _ in range(10):
            vector = param.generate_vector()
            assert vector[0] > 5
        
        # Remove vector
        param.remove_directed_vector("vec1")
        assert param.num_directed_vectors == 1
        
        # Clear constraints
        param.clear_constraints()
        assert len(param.vector_constraints) == 0
