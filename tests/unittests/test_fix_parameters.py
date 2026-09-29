"""
Regression tests for Parameter bug fixes.

Each test class covers one fixed defect in pytest_strategy/parameters.py.
"""

import pytest

from pytest_strategy import RNGInteger, Series
from pytest_strategy.parameters import Parameter
from pytest_strategy.test_args import TestArg


class TestCallerContainersNotMutated:
    """Parameter copies directed_vectors, test_vectors and vector_constraints."""

    def test_add_methods_do_not_mutate_caller_objects(self):
        """add_* on one Parameter must not leak into the caller's list/dicts."""
        constraints = [lambda v: v[0] >= 0]
        directed = {"zero": (0,)}
        test_vectors = {"one": (1,)}

        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            directed_vectors=directed,
            test_vectors=test_vectors,
            vector_constraints=constraints,
        )
        param.add_constraint(lambda v: v[0] > 5)
        param.add_directed_vector("ten", (10,))
        param.add_test_vector("two", (2,))

        assert len(constraints) == 1
        assert directed == {"zero": (0,)}
        assert test_vectors == {"one": (1,)}
        # The Parameter itself still sees its own additions
        assert len(param.vector_constraints) == 2
        assert param.directed_vectors == {"zero": (0,), "ten": (10,)}
        assert param.test_vectors == {"one": (1,), "two": (2,)}

    def test_remove_methods_do_not_mutate_caller_objects(self):
        """remove_* on one Parameter must not delete entries from the caller's dicts."""
        directed = {"zero": (0,)}
        test_vectors = {"one": (1,)}

        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            directed_vectors=directed,
            test_vectors=test_vectors,
        )
        param.remove_directed_vector("zero")
        param.remove_test_vector("one")

        assert directed == {"zero": (0,)}
        assert test_vectors == {"one": (1,)}

    def test_shared_containers_do_not_leak_between_parameters(self):
        """A factory reusing module-level containers must not grow them per call."""
        common = [lambda v: v[0] >= 0]
        edges = {"zero": (0,)}

        def strat_a():
            param = Parameter(
                TestArg("x", rng_type=RNGInteger(0, 100)),
                vector_constraints=common,
                directed_vectors=edges,
            )
            param.add_constraint(lambda v: v[0] > 5)
            param.add_directed_vector("ten", (10,))
            return param

        for _ in range(3):
            strat_a()

        strat_b = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            vector_constraints=common,
            directed_vectors=edges,
        )
        assert len(common) == 1
        assert len(strat_b.vector_constraints) == 1
        assert strat_b.directed_vectors == {"zero": (0,)}


class TestCountValidation:
    """Parameter(nsamples=..., max_retries=...) and generate_vectors(n) reject bad counts."""

    @pytest.mark.parametrize("nsamples", ["5", -1, 2.5, True, False])
    def test_invalid_nsamples_raises(self, nsamples):
        with pytest.raises(ValueError, match="nsamples must be None or an int >= 0"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), nsamples=nsamples)

    @pytest.mark.parametrize("nsamples", [None, 0, 1, 25])
    def test_valid_nsamples_accepted(self, nsamples):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), nsamples=nsamples)
        assert param.nsamples == nsamples

    @pytest.mark.parametrize("max_retries", [0, -3, "100", 1.5, True])
    def test_invalid_max_retries_raises(self, max_retries):
        with pytest.raises(ValueError, match="max_retries must be an int >= 1"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), max_retries=max_retries)

    def test_max_retries_one_accepted(self):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), max_retries=1)
        assert len(param.generate_vectors(3, mode="random_only")) == 3

    @pytest.mark.parametrize(
        "rng_type",
        [RNGInteger(0, 10), Series([1, 2, 3])],
        ids=["random", "series"],
    )
    @pytest.mark.parametrize("mode", ["all", "random_only", "mixed"])
    def test_negative_n_raises_for_every_generating_path(self, rng_type, mode):
        param = Parameter(TestArg("x", rng_type=rng_type))
        with pytest.raises(ValueError, match="n must be >= 0, got -1"):
            param.generate_vectors(-1, mode=mode)

    def test_zero_n_returns_only_directed(self):
        param = Parameter(
            TestArg("x", rng_type=Series([1, 2, 3])),
            directed_vectors={"edge": (0,)},
        )
        assert param.generate_vectors(0) == [(0,)]
