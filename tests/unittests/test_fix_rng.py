"""
Regression tests for fixed bugs in the RNG module.

Tests cover:
- Weighted generators with a predicate re-choosing the range on every retry
- RNGInteger/RNGFloat rejecting min > max at construction
- Weights of weighted types and RNGEnum validated at construction
- RNGEnum rejecting non-Enum arguments and member-less Enums with RNGValueError
- RNGEnum filtering members by the predicate up front instead of retrying draws
- Series/RNGSequence rejecting unordered sets
- RNG.string/RNGString validating length bounds and charset
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
    RNGSequence,
    RNGString,
    RNGValueError,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
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


class TestRNGEnumClassValidation:
    """RNGEnum raises RNGValueError, not TypeError/IndexError, for a bad enum_class"""

    @pytest.mark.parametrize("bad", [Shade.LIGHT, "Shade", 42, None])
    def test_non_class_rejected(self, bad):
        """issubclass() used to raise TypeError before the RNGValueError branch could run"""
        with pytest.raises(RNGValueError, match="is not an Enum class"):
            RNGEnum(bad)

    def test_non_enum_class_still_rejected(self):
        """A class that is not an Enum keeps raising RNGValueError"""
        with pytest.raises(RNGValueError, match="is not an Enum class"):
            RNGEnum(int)

    def test_enum_without_members_rejected(self):
        """A member-less Enum used to build fine and fail with IndexError at generate()"""

        class Empty(Enum):
            pass

        with pytest.raises(RNGValueError, match="Empty has no members"):
            RNGEnum(Empty)


class TestRNGEnumPredicateFiltersMembers:
    """RNGEnum applies the predicate to its finite set of members instead of retrying draws"""

    def test_selective_predicate_on_large_enum(self):
        """One valid member out of 60 used to fail about 19% of calls after 100 retries"""
        big = Enum("Big", [f"M{i}" for i in range(60)])
        RNG.seed(0)
        rng_enum = RNGEnum(big, predicate=lambda m: m is big.M7)

        assert {rng_enum.generate() for _ in range(1000)} == {big.M7}

    def test_selective_predicate_on_low_weight_members(self):
        """Excluding the member that holds 99% of the weight used to fail about 37% of calls"""
        RNG.seed(0)
        rng_enum = RNGEnum(
            Shade,
            weights={Shade.LIGHT: 0.99, Shade.MEDIUM: 0.005, Shade.DARK: 0.005},
            predicate=lambda s: s is not Shade.LIGHT,
        )

        values = [rng_enum.generate() for _ in range(500)]

        assert set(values) == {Shade.MEDIUM, Shade.DARK}

    def test_filtered_members_keep_relative_weights(self):
        """The result matches rejection sampling: weights renormalized over accepted members"""
        RNG.seed(3)
        rng_enum = RNGEnum(
            Shade,
            weights={Shade.LIGHT: 0.6, Shade.MEDIUM: 0.3, Shade.DARK: 0.1},
            predicate=lambda s: s is not Shade.DARK,
        )

        values = [rng_enum.generate() for _ in range(3000)]
        light_share = values.count(Shade.LIGHT) / len(values)

        assert Shade.DARK not in values
        assert 0.62 < light_share < 0.72  # expected 2/3

    def test_impossible_predicate_raises_at_construction(self):
        """No member satisfies the predicate: fail when built, not after 100 draws"""
        with pytest.raises(RNGValueError, match="No valid value found: no member of Shade"):
            RNGEnum(Shade, predicate=lambda s: False)

    def test_predicate_accepting_only_unweighted_members_raises(self):
        """Only weighted members are candidates, so accepting an unweighted one is not enough"""
        with pytest.raises(RNGValueError, match="No valid value found"):
            RNGEnum(Shade, weights={Shade.LIGHT: 1.0}, predicate=lambda s: s is Shade.DARK)

    def test_predicate_accepting_only_zero_weight_members_raises(self):
        """Accepted members whose weights are all zero can never be drawn"""
        with pytest.raises(RNGValueError, match="with a positive weight"):
            RNGEnum(
                Shade,
                weights={Shade.LIGHT: 1.0, Shade.MEDIUM: 0.0},
                predicate=lambda s: s is Shade.MEDIUM,
            )

    @pytest.mark.parametrize("seed", [0, 42, 2024])
    def test_stream_unchanged_without_predicate(self, seed):
        """Without a predicate the seeded sequence is the same single random call per draw"""
        weights = {Shade.LIGHT: 0.5, Shade.DARK: 0.3, Shade.MEDIUM: 0.2}

        random.seed(seed)
        expected_uniform = [random.choice(list(Shade)) for _ in range(50)]
        expected_weighted = [
            random.choices(list(weights), weights=list(weights.values()), k=1)[0] for _ in range(50)
        ]

        RNG.seed(seed)
        uniform = RNGEnum(Shade)
        weighted = RNGEnum(Shade, weights=weights)
        assert [uniform.generate() for _ in range(50)] == expected_uniform
        assert [weighted.generate() for _ in range(50)] == expected_weighted


class TestSequenceLikeRejectsSets:
    """Series/RNGSequence need an ordered input for seeded runs to be reproducible"""

    @pytest.mark.parametrize("seq_cls", [Series, RNGSequence])
    @pytest.mark.parametrize("container", [set, frozenset])
    def test_set_rejected(self, seq_cls, container):
        """A set of strings iterates in an order that changes with PYTHONHASHSEED"""
        with pytest.raises(
            RNGValueError,
            match=rf"{seq_cls.__name__} requires an ordered sequence, got a "
            rf"{container.__name__} .* use sorted\(\.\.\.\) or a list",
        ):
            seq_cls(container({"alpha", "beta", "gamma", "delta"}))

    def test_sorted_set_accepted(self):
        """The suggested fix works"""
        series = Series(sorted({"beta", "alpha", "gamma"}))

        assert series._get_auto_sequence() == ["alpha", "beta", "gamma"]

    @pytest.mark.parametrize(
        "sequence",
        [
            ["b", "a", "c"],
            ("b", "a", "c"),
            {"b": 1, "a": 2, "c": 3}.keys(),
            (x for x in "bac"),
        ],
        ids=["list", "tuple", "dict_keys", "generator"],
    )
    def test_ordered_iterables_still_accepted(self, sequence):
        """dict keys, generators and other ordered iterables keep working"""
        assert Series(sequence)._get_auto_sequence() == ["b", "a", "c"]


class TestStringArgsValidation:
    """RNG.string (ValueError) and RNGString (RNGValueError at construction) check their args"""

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"min_length": -3, "max_length": -1}, r"min_length cannot be negative"),
            ({"min_length": -10, "max_length": 2}, r"min_length cannot be negative"),
            ({"min_length": 5, "max_length": 2}, r"min_length \(5\) must be <= max_length \(2\)"),
            ({"charset": ""}, r"charset cannot be empty unless the length is 0"),
            ({"min_length": 0, "max_length": 3, "charset": ""}, r"charset cannot be empty"),
            ({"length": 4, "charset": ""}, r"charset cannot be empty"),
        ],
        ids=[
            "negative_range",
            "negative_min",
            "min_above_max",
            "empty_charset",
            "empty_charset_min_0",
            "empty_charset_fixed_length",
        ],
    )
    def test_rng_string_rejects_bad_args(self, kwargs, message):
        """Negative min_length used to collapse to '', an empty charset failed only for some seeds"""
        with pytest.raises(ValueError, match=f"^String {message}"):
            RNG.string(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"length": -2}, r"length cannot be negative \(got length=-2\)"),
            ({"min_length": -10, "max_length": 2}, r"min_length cannot be negative"),
            ({"min_length": 5, "max_length": 2}, r"min_length \(5\) must be <= max_length \(2\)"),
            ({"charset": ""}, r"charset cannot be empty unless the length is 0"),
            ({"min_length": 0, "max_length": 3, "charset": ""}, r"charset cannot be empty"),
        ],
        ids=[
            "negative_length",
            "negative_min",
            "min_above_max",
            "empty_charset",
            "empty_charset_min_0",
        ],
    )
    def test_rng_string_type_rejects_bad_args_at_construction(self, kwargs, message):
        """RNGString fails when defined instead of at generate() for some seeds"""
        with pytest.raises(RNGValueError, match=f"^RNGString {message}"):
            RNGString(**kwargs)

    def test_negative_fixed_length_keeps_value_error(self):
        """RNG.string(length=-1) still raises ValueError, as documented"""
        with pytest.raises(ValueError, match="String length cannot be negative"):
            RNG.string(length=-1)

    def test_empty_charset_allowed_for_zero_length(self):
        """An empty charset is fine when the string is always empty"""
        assert RNG.string(length=0, charset="") == ""
        assert RNG.string(min_length=0, max_length=0, charset="") == ""
        assert RNGString(length=0, charset="").generate() == ""

    def test_min_max_ignored_with_fixed_length(self):
        """min_length/max_length are not checked when a fixed length makes them unused"""
        RNG.seed(0)
        assert len(RNGString(length=3, min_length=10, max_length=2).generate()) == 3
        assert len(RNG.string(length=3, min_length=-1)) == 3

    def test_zero_min_length_uniform_lengths(self):
        """A valid 0..2 range gives each length about a third of the time"""
        RNG.seed(12345)
        rng_type = RNGString(min_length=0, max_length=2)

        lengths = [len(rng_type.generate()) for _ in range(3000)]

        assert all(0.25 < lengths.count(n) / len(lengths) < 0.42 for n in (0, 1, 2))
