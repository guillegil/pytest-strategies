"""
Regression tests for fixed bugs in the RNG module.

Tests cover:
- Weighted generators with a predicate re-choosing the range on every retry
"""

import random

import pytest

from pytest_strategy import RNG, RNGWeightedFloat, RNGWeightedInteger


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
