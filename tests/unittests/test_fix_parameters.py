"""
Regression tests for Parameter bug fixes.

Each test class covers one fixed defect in pytest_strategy/parameters.py.
"""

from pytest_strategy import RNGInteger
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
