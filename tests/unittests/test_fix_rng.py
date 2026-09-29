"""
Regression tests for fixed bugs in the RNG module.

Tests cover:
- Weighted generators with a predicate re-choosing the range on every retry
- RNGInteger/RNGFloat rejecting min > max at construction
- Weights of weighted types and RNGEnum validated at construction
"""

import math
import random
from enum import Enum

import pytest

from pytest_strategy import (
    RNG,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGValueError,
    RNGWeightedFloat,
    RNGWeightedInteger,
)


class Shade(Enum):
    """Small enum used by the RNGEnum tests"""

    LIGHT = 1
    MEDIUM = 2
    DARK = 3


class TestWeightedPredicateRedrawsRange:
    """A predicate on a weighted generator filters the whole weighted domain"""

    def test_winteger_predicate_satisfiable_in_one_range(self):
        """Values are found even when most of the weight is on a range the predicate rejects"""
        RNG.seed(0)
        ranges = {(0, 10): 0.7, (100, 110): 0.3}

        values = [RNG.winteger(ranges, predicate=lambda x: x > 50) for _ in range(500)]

        assert all(100 <= v <= 110 for v in values)

    def test_wfloat_predicate_satisfiable_in_one_range(self):
        """Same as the integer case, for floats"""
        RNG.seed(0)
        ranges = {(0.0, 1.0): 0.7, (10.0, 11.0): 0.3}

        values = [RNG.wfloat(ranges, predicate=lambda x: x > 5.0) for _ in range(500)]

        assert all(10.0 <= v <= 11.0 for v in values)

    def test_weighted_types_predicate_satisfiable_in_one_range(self):
        """RNGWeightedInteger and RNGWeightedFloat inherit the fix"""
        RNG.seed(0)
        int_type = RNGWeightedInteger({(0, 10): 0.7, (100, 110): 0.3}, predicate=lambda x: x > 50)
        float_type = RNGWeightedFloat(
            {(0.0, 1.0): 0.7, (10.0, 11.0): 0.3}, predicate=lambda x: x > 5.0
        )

        assert all(int_type.generate() >= 100 for _ in range(500))
        assert all(float_type.generate() >= 10.0 for _ in range(500))

    def test_predicate_gives_conditional_distribution(self):
        """Accepted values follow the weights adjusted for rejections, like rejection sampling"""
        RNG.seed(1)
        ranges = {(0, 9): 0.5, (100, 109): 0.5}

        # Only 0 is accepted in the low range, so it should hold about 1/11 of the values
        values = [RNG.winteger(ranges, predicate=lambda x: x == 0 or x >= 100) for _ in range(2000)]
        low_share = sum(1 for v in values if v < 100) / len(values)

        assert low_share < 0.2

    @pytest.mark.parametrize("seed", [0, 7, 2024])
    def test_winteger_stream_unchanged_without_predicate(self, seed):
        """Without a predicate the seeded sequence is the same as one choices+randint per draw"""
        ranges = {(0, 10): 0.7, (100, 200): 0.2, (-50, -40): 0.1}

        random.seed(seed)
        expected = []
        for _ in range(50):
            lo, hi = random.choices(list(ranges), weights=list(ranges.values()), k=1)[0]
            expected.append(random.randint(lo, hi))

        RNG.seed(seed)
        assert [RNG.winteger(ranges) for _ in range(50)] == expected

    @pytest.mark.parametrize("seed", [0, 7, 2024])
    def test_wfloat_stream_unchanged_without_predicate(self, seed):
        """Without a predicate the seeded sequence is the same as one choices+uniform per draw"""
        ranges = {(0.0, 1.0): 3, (10.0, 20.0): 1}

        random.seed(seed)
        expected = []
        for _ in range(50):
            lo, hi = random.choices(list(ranges), weights=list(ranges.values()), k=1)[0]
            expected.append(random.uniform(lo, hi))

        RNG.seed(seed)
        assert [RNG.wfloat(ranges) for _ in range(50)] == expected


class TestRNGTypeBounds:
    """RNGInteger/RNGFloat reject min > max when constructed, after filling defaults"""

    def test_float_min_only_above_default_max(self):
        """RNGFloat(min=5.0) used to draw from [1.0, 5.0], below the requested min"""
        with pytest.raises(
            RNGValueError,
            match=r"RNGFloat min \(5\.0\) must be <= max \(1\.0\) "
            r"\(max was not given and defaults to 1\.0\)",
        ):
            RNGFloat(min=5.0)

    def test_float_max_only_below_default_min(self):
        """RNGFloat(max=-2.0) used to draw from [-2.0, 0.0], above the requested max"""
        with pytest.raises(
            RNGValueError,
            match=r"RNGFloat min \(0\.0\) must be <= max \(-2\.0\) "
            r"\(min was not given and defaults to 0\.0\)",
        ):
            RNGFloat(max=-2.0)

    def test_float_explicit_reversed_bounds(self):
        """Explicit reversed bounds are rejected instead of silently swapped"""
        with pytest.raises(RNGValueError, match=r"RNGFloat min \(10\.0\) must be <= max \(0\.0\)$"):
            RNGFloat(10.0, 0.0)

    def test_integer_min_only_above_default_max(self):
        """RNGInteger(min=2**31) fails at construction, not with 'empty range' at generate()"""
        with pytest.raises(
            RNGValueError,
            match=r"RNGInteger min \(2147483648\) must be <= max \(2147483647\) "
            r"\(max was not given",
        ):
            RNGInteger(min=2**31)

    def test_integer_max_only_below_default_min(self):
        """RNGInteger(max=-(2**31) - 1) fails at construction"""
        with pytest.raises(RNGValueError, match="min was not given and defaults to -2147483648"):
            RNGInteger(max=-(2**31) - 1)

    def test_integer_explicit_reversed_bounds(self):
        """Explicit reversed bounds are rejected at construction"""
        with pytest.raises(RNGValueError, match=r"RNGInteger min \(10\) must be <= max \(5\)$"):
            RNGInteger(10, 5)

    def test_valid_single_bound_still_works(self):
        """A single bound on the right side of the default still builds and generates"""
        RNG.seed(0)
        float_type = RNGFloat(min=0.5)
        int_type = RNGInteger(max=0)

        assert (float_type.min, float_type.max) == (0.5, 1.0)
        assert all(0.5 <= float_type.generate() <= 1.0 for _ in range(100))
        assert all(-(2**31) <= int_type.generate() <= 0 for _ in range(100))

    def test_equal_bounds_allowed(self):
        """min == max is a valid one-value range"""
        assert RNGInteger(5, 5).generate() == 5
        assert RNGFloat(2.5, 2.5).generate() == 2.5


class TestWeightsValidation:
    """Weights are checked when the RNG type is built, not when a value is generated"""

    @pytest.mark.parametrize("rng_cls", [RNGWeightedInteger, RNGWeightedFloat])
    def test_negative_weight_rejected(self, rng_cls):
        """A negative weight used to skew random.choices so positive-weight ranges never came up"""
        with pytest.raises(
            RNGValueError,
            match=rf"{rng_cls.__name__} weight for \(1, 1\) must be a finite number >= 0, got -5",
        ):
            rng_cls({(0, 0): 1, (1, 1): -5, (2, 2): 5})

    @pytest.mark.parametrize("rng_cls", [RNGWeightedInteger, RNGWeightedFloat])
    def test_all_zero_weights_rejected(self, rng_cls):
        """All-zero weights used to build fine and fail with a plain ValueError at generate()"""
        with pytest.raises(RNGValueError, match="weights cannot all be zero"):
            rng_cls({(0, 1): 0, (2, 3): 0.0})

    @pytest.mark.parametrize("rng_cls", [RNGWeightedInteger, RNGWeightedFloat])
    def test_empty_ranges_rejected(self, rng_cls):
        """An empty ranges dict is rejected with RNGValueError instead of IndexError"""
        with pytest.raises(RNGValueError, match=f"{rng_cls.__name__} weights cannot be empty"):
            rng_cls({})

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_weight_rejected(self, bad):
        """NaN or infinite weights are rejected at construction"""
        with pytest.raises(RNGValueError, match="must be a finite number >= 0"):
            RNGWeightedInteger({(0, 1): 1, (2, 3): bad})

    def test_some_zero_weights_allowed(self):
        """A zero weight on some entries is a valid way to exclude them"""
        RNG.seed(0)
        int_type = RNGWeightedInteger({(0, 0): 0, (5, 5): 1})
        float_type = RNGWeightedFloat({(0.0, 0.0): 0, (5.0, 5.0): 2})

        assert {int_type.generate() for _ in range(50)} == {5}
        assert {float_type.generate() for _ in range(50)} == {5.0}

    def test_enum_negative_weight_rejected(self):
        """RNGEnum with a negative weight used to never pick members with a positive weight"""
        with pytest.raises(RNGValueError, match="RNGEnum weight for .*MEDIUM.* got -1.0"):
            RNGEnum(Shade, weights={Shade.LIGHT: 1.0, Shade.MEDIUM: -1.0, Shade.DARK: 5.0})

    def test_enum_all_zero_weights_rejected(self):
        """RNGEnum with all-zero weights fails at construction"""
        with pytest.raises(RNGValueError, match="RNGEnum weights cannot all be zero"):
            RNGEnum(Shade, weights={Shade.LIGHT: 0, Shade.MEDIUM: 0})

    def test_enum_empty_weights_rejected(self):
        """weights={} used to fall back to uniform selection over all members"""
        with pytest.raises(RNGValueError, match="RNGEnum weights cannot be empty"):
            RNGEnum(Shade, weights={})

    def test_enum_some_zero_weights_allowed(self):
        """A zero weight on some members is a valid way to exclude them"""
        RNG.seed(0)
        rng_enum = RNGEnum(Shade, weights={Shade.LIGHT: 0, Shade.DARK: 1})

        assert {rng_enum.generate() for _ in range(50)} == {Shade.DARK}
