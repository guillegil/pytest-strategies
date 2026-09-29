"""
Regression tests for Parameter bug fixes.

Each test class covers one fixed defect in pytest_strategy/parameters.py.
"""

import pytest

from pytest_strategy import RNGInteger, Series
from pytest_strategy.parameters import Parameter
from pytest_strategy.rng import RNG, RNGSequence
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


class TestSeriesFiniteConstraintsSkipCombinations:
    """Finite mode skips Series combinations that the vector constraints reject."""

    @staticmethod
    def _ordered_pairs(**kwargs):
        return Parameter(
            TestArg("lo", rng_type=Series([1, 2, 3])),
            TestArg("hi", rng_type=Series([1, 2, 3])),
            vector_constraints=[lambda v: v[0] < v[1]],
            **kwargs,
        )

    def test_all_series_cycles_over_valid_combinations(self):
        """n larger than the valid set cycles through it in product order."""
        samples = self._ordered_pairs().generate_vectors(10)
        valid = [(1, 2), (1, 3), (2, 3)]
        assert samples == (valid * 4)[:10]

    def test_all_series_truncates_to_first_valid_combinations(self):
        """n smaller than the valid set takes the first n valid rows."""
        assert self._ordered_pairs().generate_vectors(2) == [(1, 2), (1, 3)]

    def test_finite_matches_exhaustive_filtering(self):
        """Finite mode with n == number of valid rows equals auto mode."""
        param = self._ordered_pairs()
        exhaustive = param.generate_exhaustive()
        assert param.generate_vectors(len(exhaustive), mode="random_only") == exhaustive

    def test_single_series_value_excluding_constraint(self):
        param = Parameter(
            TestArg("x", rng_type=Series([1, 2, 3])),
            vector_constraints=[lambda v: v[0] != 2],
        )
        assert param.generate_vectors(3, mode="random_only") == [(1,), (3,), (1,)]

    def test_series_plus_random_constraint_on_series_value_only(self):
        """Redrawing the random arg cannot fix a rejected Series value; it is skipped."""
        RNG.seed(0)
        param = Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            TestArg("i", rng_type=RNGInteger(0, 5)),
            vector_constraints=[lambda v: v[0] != 1],
        )
        samples = param.generate_vectors(4)
        assert [s[0] for s in samples] == [2, 3, 2, 3]
        assert all(0 <= s[1] <= 5 for s in samples)

    def test_series_plus_random_constraint_on_random_value_keeps_every_combination(self):
        """A constraint on the random arg is met by redrawing, so no combination is lost."""
        RNG.seed(0)
        param = Parameter(
            TestArg("role", rng_type=Series(["admin", "user", "guest"])),
            TestArg("uid", rng_type=RNGInteger(1, 1000)),
            vector_constraints=[lambda v: v[1] > 500],
        )
        samples = param.generate_vectors(6)
        assert [s[0] for s in samples] == ["admin", "user", "guest"] * 2
        assert all(s[1] > 500 for s in samples)

    def test_directed_vectors_still_prepended(self):
        param = self._ordered_pairs(directed_vectors={"same": (2, 2)})
        assert param.generate_vectors(3) == [(2, 2), (1, 2), (1, 3), (2, 3)]

    def test_all_series_unsatisfiable_raises_after_one_cycle(self):
        """Every combination is checked once; nothing is pointlessly retried."""
        calls = []

        def never(v):
            calls.append(v)
            return False

        param = Parameter(
            TestArg("a", rng_type=Series([1, 2])),
            TestArg("b", rng_type=Series(["x", "y", "z"])),
            vector_constraints=[never],
        )
        with pytest.raises(ValueError, match="Could not generate valid vector"):
            param.generate_vectors(5)
        assert calls == [(1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z")]

    def test_series_plus_random_unsatisfiable_raises_after_one_cycle(self):
        """With random args, each combination gets max_retries attempts, then it stops."""
        calls = []

        def never(v):
            calls.append(v)
            return False

        param = Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            TestArg("i", rng_type=RNGInteger(0, 100)),
            vector_constraints=[never],
            max_retries=4,
        )
        with pytest.raises(ValueError, match="none of the 3 Series combinations"):
            param.generate_vectors(10)
        assert [v[0] for v in calls] == [1] * 4 + [2] * 4 + [3] * 4


class TestExhaustiveRetriesRandomPositions:
    """Auto mode redraws random args instead of dropping a combination on one bad draw."""

    @pytest.mark.parametrize("seed", range(50))
    def test_constraint_on_random_arg_keeps_every_combination(self, seed):
        RNG.seed(seed)
        param = Parameter(
            TestArg("role", rng_type=Series(["admin", "user", "guest"])),
            TestArg("uid", rng_type=RNGInteger(1, 1000)),
            vector_constraints=[lambda v: v[1] > 500],
        )
        samples = param.generate_exhaustive()
        assert [s[0] for s in samples] == ["admin", "user", "guest"]
        assert all(s[1] > 500 for s in samples)

    def test_sequence_only_constraint_still_filters_with_random_arg(self):
        """Combinations whose sequence values break a constraint are still dropped."""
        RNG.seed(0)
        param = Parameter(
            TestArg("lo", rng_type=Series([1, 2, 3])),
            TestArg("hi", rng_type=Series([1, 2, 3])),
            TestArg("pad", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda v: v[0] < v[1]],
        )
        samples = param.generate_exhaustive()
        assert [s[:2] for s in samples] == [(1, 2), (1, 3), (2, 3)]

    def test_unsatisfiable_random_constraint_drops_after_max_retries(self):
        calls = []

        def never(v):
            calls.append(v)
            return False

        param = Parameter(
            TestArg("s", rng_type=Series([1, 2])),
            TestArg("i", rng_type=RNGInteger(0, 9)),
            vector_constraints=[never],
            max_retries=3,
        )
        assert param.generate_exhaustive() == []
        assert [v[0] for v in calls] == [1, 1, 1, 2, 2, 2]

    def test_no_random_args_checks_each_combination_once(self):
        calls = []

        def odd(v):
            calls.append(v)
            return v[0] % 2 == 1

        param = Parameter(TestArg("s", rng_type=Series([1, 2, 3])), vector_constraints=[odd])
        assert param.generate_exhaustive() == [(1,), (3,)]
        assert calls == [(1,), (2,), (3,)]


class TestValidatorAppliedToSequenceValues:
    """A TestArg validator also checks Series/sequence values placed by Parameter."""

    @staticmethod
    def _positive(values, rng_cls=Series):
        return Parameter(TestArg("x", rng_type=rng_cls(values), validator=lambda x: x > 0))

    def test_series_finite_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="Value -1 failed validation for argument 'x'"):
            self._positive([1, -1]).generate_vectors(4, mode="random_only")

    def test_series_exhaustive_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="Value -1 failed validation for argument 'x'"):
            self._positive([1, -1]).generate_exhaustive()

    def test_rngsequence_exhaustive_rejects_invalid_value(self):
        """Matches finite mode, where RNGSequence values already go through generate()."""
        with pytest.raises(ValueError, match="Value -1 failed validation for argument 'x'"):
            self._positive([1, -1], rng_cls=RNGSequence).generate_exhaustive()

    def test_valid_values_pass_through(self):
        param = self._positive([1, 2])
        assert param.generate_vectors(3, mode="random_only") == [(1,), (2,), (1,)]
        assert param.generate_exhaustive() == [(1,), (2,)]

    def test_validator_sees_each_series_value(self):
        seen = []

        def record(value):
            seen.append(value)
            return True

        param = Parameter(
            TestArg("a", rng_type=Series([1, 2]), validator=record),
            TestArg("b", rng_type=Series(["x"])),
        )
        param.generate_vectors(3, mode="random_only")
        assert seen == [1, 2, 1]
        seen.clear()
        param.generate_exhaustive()
        assert seen == [1, 2]
