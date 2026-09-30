"""
Tests for Parameter(per_sequence_samples=True): n random rows per combination of the
Series/RNGSequence args instead of n rows in total.
"""

import warnings
from collections import Counter

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    PytestStrategiesWarning,
    RNGInteger,
    RNGSequence,
    Series,
    TestArg,
)


@pytest.fixture(autouse=True)
def _seed():
    RNG.seed(1234)


def device_width(seq_type, **kwargs):
    return Parameter(
        TestArg("device", rng_type=seq_type(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        **kwargs,
    )


class TestRowsPerCombination:
    @pytest.mark.parametrize("seq_type", [Series, RNGSequence])
    def test_n_rows_for_each_sequence_value(self, seq_type):
        samples = device_width(seq_type, per_sequence_samples=True).generate_vectors(10)

        assert len(samples) == 20
        assert Counter(s[0] for s in samples) == {"devA": 10, "devB": 10}
        assert all(1 <= s[1] <= 64 for s in samples)

    @pytest.mark.parametrize("seq_type", [Series, RNGSequence])
    def test_rows_are_grouped_in_declaration_order(self, seq_type):
        samples = device_width(seq_type, per_sequence_samples=True).generate_vectors(3)

        assert [s[0] for s in samples] == ["devA"] * 3 + ["devB"] * 3

    def test_random_args_are_drawn_per_row(self):
        samples = device_width(Series, per_sequence_samples=True).generate_vectors(10)

        assert len({s[1] for s in samples if s[0] == "devA"}) > 1

    def test_multiple_sequences_use_their_product(self):
        param = Parameter(
            TestArg("device", rng_type=Series(["devA", "devB"])),
            TestArg("mode", rng_type=RNGSequence(["fast", "slow", "idle"])),
            TestArg("width", rng_type=RNGInteger(1, 64)),
            per_sequence_samples=True,
        )

        samples = param.generate_vectors(2)

        assert len(samples) == 2 * 3 * 2
        assert Counter((s[0], s[1]) for s in samples) == {
            (d, m): 2 for d in ("devA", "devB") for m in ("fast", "slow", "idle")
        }

    def test_zero_samples_gives_no_rows(self):
        assert device_width(Series, per_sequence_samples=True).generate_vectors(0) == []

    def test_zero_samples_does_not_validate_the_sequence(self):
        param = Parameter(
            TestArg("device", rng_type=Series(["devA", "bad"]), validator=lambda d: d != "bad"),
            TestArg("width", rng_type=RNGInteger(1, 64)),
            per_sequence_samples=True,
        )

        assert param.generate_vectors(0) == []

    def test_without_sequence_args_the_flag_has_no_effect(self):
        param = Parameter(TestArg("width", rng_type=RNGInteger(1, 64)), per_sequence_samples=True)

        assert len(param.generate_vectors(10)) == 10


class TestModesAndVectors:
    def test_directed_vectors_are_prepended(self):
        param = device_width(
            Series, per_sequence_samples=True, directed_vectors={"narrow": ("devA", 1)}
        )

        samples = param.generate_vectors(4, mode="all")

        assert samples[0] == ("devA", 1)
        assert len(samples) == 1 + 2 * 4

    def test_random_only_has_no_directed_vectors(self):
        param = device_width(
            Series, per_sequence_samples=True, directed_vectors={"narrow": ("devA", 1)}
        )

        assert len(param.generate_vectors(4, mode="random_only")) == 8

    def test_directed_only_ignores_the_flag(self):
        param = device_width(
            Series, per_sequence_samples=True, directed_vectors={"narrow": ("devA", 1)}
        )

        assert param.generate_vectors(10, mode="directed_only") == [("devA", 1)]

    def test_exhaustive_is_unchanged(self):
        param = device_width(Series, per_sequence_samples=True)

        assert [s[0] for s in param.generate_exhaustive()] == ["devA", "devB"]


class TestConstraintsAndValidation:
    def test_constraint_on_random_arg_is_redrawn(self):
        param = device_width(
            Series,
            per_sequence_samples=True,
            vector_constraints=[lambda v: v[1] % 2 == 0],
        )

        samples = param.generate_vectors(5)

        assert len(samples) == 10
        assert all(s[1] % 2 == 0 for s in samples)

    def test_unsatisfiable_combination_is_skipped_with_a_warning(self):
        param = device_width(
            Series,
            per_sequence_samples=True,
            vector_constraints=[lambda v: v[0] == "devA" or v[1] > 100],
            max_retries=5,
        )

        with pytest.warns(PytestStrategiesWarning, match=r"device='devB'.*0 of 3 rows"):
            samples = param.generate_vectors(3)

        assert [s[0] for s in samples] == ["devA"] * 3

    def test_sequence_only_constraint_skips_silently(self):
        param = Parameter(
            TestArg("device", rng_type=Series(["devA", "devB"])),
            per_sequence_samples=True,
            vector_constraints=[lambda v: v[0] != "devB"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = param.generate_vectors(2)

        assert samples == [("devA",), ("devA",)]

    def test_every_combination_rejected_raises(self):
        param = device_width(
            Series,
            per_sequence_samples=True,
            vector_constraints=[lambda v: False],
            max_retries=3,
        )

        with pytest.raises(ValueError, match="none of the 2 sequence combinations"):
            param.generate_vectors(3)

    def test_validator_applies_to_sequence_values(self):
        param = Parameter(
            TestArg(
                "device",
                rng_type=Series(["devA", "bad"]),
                validator=lambda d: d.startswith("dev"),
            ),
            per_sequence_samples=True,
        )

        with pytest.raises(ValueError, match="failed validation"):
            param.generate_vectors(1)


class TestDefaultsAndExport:
    def test_flag_defaults_to_false_and_keeps_total_count(self):
        param = device_width(Series)

        assert param.per_sequence_samples is False
        assert len(param.generate_vectors(10)) == 10

    def test_default_random_stream_is_unchanged(self):
        RNG.seed(99)
        baseline = device_width(RNGSequence).generate_vectors(10)
        RNG.seed(99)
        again = device_width(RNGSequence, per_sequence_samples=False).generate_vectors(10)

        assert baseline == again

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_non_bool_flag_is_rejected(self, value):
        with pytest.raises(ValueError, match="per_sequence_samples must be a bool"):
            device_width(Series, per_sequence_samples=value)

    def test_to_dict_reports_the_flag(self):
        assert device_width(Series, per_sequence_samples=True).to_dict()["per_sequence_samples"]
