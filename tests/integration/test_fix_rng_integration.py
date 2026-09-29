"""
Integration tests for fixed RNG bugs, run through real pytest sessions.

These tests use pytester to check that:
- Strategies with a selective predicate on weighted types or RNGEnum collect
  for every --rng-seed (they used to fail collection for most seeds)
- Misconfigured RNG types fail collection with a clear error naming the problem
  (they used to run with wrong values, or fail with an unrelated message)

Distinct strategy filenames are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import pytest

from pytest_strategy import Strategy

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _restore_registry():
    """Drop what each in-process run registers, so reruns don't trip the duplicate-name warning."""
    registry = dict(Strategy._registry)
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)


class TestSelectivePredicatesCollect:
    """A predicate that only some ranges/members satisfy no longer breaks collection."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_weighted_and_enum_predicates_collect_for_every_seed(self, pytester, seed):
        """Weighted ranges and a 60-member enum with a selective predicate, 10 samples each."""
        pytester.makepyfile(**{f"fixrng_pred{seed}_strategies": """
            from enum import Enum

            from pytest_strategy import (
                Parameter,
                RNGEnum,
                RNGWeightedFloat,
                RNGWeightedInteger,
                Strategy,
                TestArg,
            )

            Big = Enum("Big", [f"M{i}" for i in range(60)])

            @Strategy.register("fixrng_weighted")
            def weighted(nsamples):
                return Parameter(
                    TestArg(
                        "x",
                        rng_type=RNGWeightedInteger(
                            {(0, 10): 0.7, (100, 110): 0.3}, predicate=lambda v: v > 50
                        ),
                    ),
                    TestArg(
                        "y",
                        rng_type=RNGWeightedFloat(
                            {(0.0, 1.0): 0.7, (10.0, 11.0): 0.3}, predicate=lambda v: v > 5.0
                        ),
                    ),
                )

            @Strategy.register("fixrng_enum")
            def enum_strategy(nsamples):
                return Parameter(
                    TestArg("m", rng_type=RNGEnum(Big, predicate=lambda m: m.name == "M7"))
                )
            """})
        pytester.makepyfile(**{f"test_fixrng_pred{seed}": """
            from pytest_strategy import Strategy

            @Strategy.strategy("fixrng_weighted")
            def test_weighted(x, y):
                assert 100 <= x <= 110
                assert 10.0 <= y <= 11.0

            @Strategy.strategy("fixrng_enum")
            def test_enum(m):
                assert m.name == "M7"
            """})

        result = pytester.runpytest_inprocess("--nsamples=10", f"--rng-seed={seed}")

        result.assert_outcomes(passed=20)


class TestMisconfiguredTypesFailClearly:
    """A bad RNG type configuration fails collection with an error that names it."""

    @pytest.mark.parametrize(
        "rng_type, expected",
        [
            ("RNGFloat(min=5.0)", "*RNGFloat min (5.0) must be <= max (1.0)*defaults to 1.0*"),
            (
                "RNGWeightedInteger({(0, 9): 1.0, (10, 19): -5.0, (20, 29): 5.0})",
                "*RNGWeightedInteger weight for (10, 19) must be a finite number >= 0*",
            ),
            ("RNGEnum(Color.RED)", "*Color.RED* is not an Enum class*"),
            ("Series({'a', 'b', 'c'})", "*Series requires an ordered sequence, got a set*"),
            (
                "RNGString(min_length=0, max_length=3, charset='')",
                "*RNGString charset cannot be empty*",
            ),
        ],
        ids=["float_bounds", "negative_weight", "enum_member", "series_set", "empty_charset"],
    )
    def test_collection_error_names_the_problem(self, pytester, request, rng_type, expected):
        """Each misconfiguration is reported by the RNG type itself, for any seed."""
        name = request.node.callspec.id
        pytester.makepyfile(**{f"fixrng_bad_{name}_strategies": f"""
            from enum import Enum

            from pytest_strategy import (
                Parameter,
                RNGEnum,
                RNGFloat,
                RNGString,
                RNGWeightedInteger,
                Series,
                Strategy,
                TestArg,
            )

            class Color(Enum):
                RED = 1
                BLUE = 2

            @Strategy.register("fixrng_bad_{name}")
            def bad(nsamples):
                return Parameter(TestArg("v", rng_type={rng_type}))
            """})
        pytester.makepyfile(**{f"test_fixrng_bad_{name}": f"""
            from pytest_strategy import Strategy

            @Strategy.strategy("fixrng_bad_{name}")
            def test_bad(v):
                pass
            """})

        result = pytester.runpytest_inprocess("--nsamples=10", "--rng-seed=1")

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines([expected])
