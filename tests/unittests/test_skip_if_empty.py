"""
Tests for Series/RNGSequence(skip_if_empty=...): an empty sequence skips the strategy's
tests with a reason instead of failing at construction.
"""

import json

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    RNGSequence,
    RNGValueError,
    Series,
    Strategy,
    TestArg,
)

REASON = "no Esm peripheral in this testbench config"


@pytest.fixture(autouse=True)
def _seed():
    RNG.seed(1234)


def channel_param(channels, seq_type=RNGSequence, **kwargs):
    return Parameter(
        TestArg("channel", rng_type=seq_type(channels, skip_if_empty=REASON)),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        **kwargs,
    )


@pytest.mark.parametrize("seq_type", [Series, RNGSequence])
class TestSequenceType:
    def test_empty_sequence_is_allowed(self, seq_type):
        seq = seq_type([], skip_if_empty=REASON)

        assert seq.sequence == []
        assert seq.skip_reason == REASON

    def test_predicate_that_filters_everything_is_allowed(self, seq_type):
        seq = seq_type([1, 3], predicate=lambda x: x % 2 == 0, skip_if_empty=REASON)

        assert seq.skip_reason == REASON

    def test_non_empty_sequence_has_no_skip_reason(self, seq_type):
        seq = seq_type([3, 7], skip_if_empty=REASON)

        assert seq.skip_reason is None
        assert seq.generate() in (3, 7)

    def test_empty_without_the_option_still_raises_and_names_it(self, seq_type):
        with pytest.raises(RNGValueError, match="skip_if_empty"):
            seq_type([])

    @pytest.mark.parametrize("reason", ["", "   ", True, 1])
    def test_reason_must_be_a_non_empty_string(self, seq_type, reason):
        with pytest.raises(RNGValueError, match="skip_if_empty must be a non-empty"):
            seq_type([1, 2], skip_if_empty=reason)

    def test_drawing_from_an_empty_sequence_names_the_reason(self, seq_type):
        with pytest.raises(RNGValueError, match="no values to draw from"):
            seq_type([], skip_if_empty=REASON).generate()

    def test_set_is_still_rejected(self, seq_type):
        with pytest.raises(RNGValueError, match="ordered sequence"):
            seq_type(set(), skip_if_empty=REASON)


class TestParameter:
    @pytest.mark.parametrize("seq_type", [Series, RNGSequence])
    def test_skip_reason_comes_from_the_empty_sequence(self, seq_type):
        assert channel_param([], seq_type).skip_reason == REASON
        assert channel_param([3, 7], seq_type).skip_reason is None

    def test_first_empty_sequence_in_declaration_order_wins(self):
        param = Parameter(
            TestArg("device", rng_type=Series([], skip_if_empty="no device")),
            TestArg("channel", rng_type=RNGSequence([], skip_if_empty="no channel")),
        )

        assert param.skip_reason == "no device"

    def test_parameter_without_sequences_has_no_skip_reason(self):
        assert Parameter(TestArg("wdata", rng_type=RNGInteger(0, 255))).skip_reason is None

    @pytest.mark.parametrize("seq_type", [Series, RNGSequence])
    @pytest.mark.parametrize("mode", ["all", "random_only", "mixed", "directed_only", "test"])
    @pytest.mark.parametrize("per_sequence_samples", [False, True])
    def test_every_mode_generates_nothing(self, seq_type, mode, per_sequence_samples):
        param = channel_param(
            [],
            seq_type,
            directed_vectors={"zero": (3, 0)},
            test_vectors={"smoke": (3, 1)},
            per_sequence_samples=per_sequence_samples,
        )

        assert param.generate_vectors(10, mode=mode) == []

    def test_exhaustive_generates_nothing(self):
        assert channel_param([], Series).generate_exhaustive() == []

    def test_filters_still_report_a_missing_vector(self):
        param = channel_param([], directed_vectors={"zero": (3, 0)})

        assert param.generate_vectors(0, filter_by_name="zero") == []
        assert param.generate_vectors(0, filter_by_index=0) == []
        with pytest.raises(KeyError):
            param.generate_vectors(0, filter_by_name="missing")
        with pytest.raises(IndexError):
            param.generate_vectors(0, filter_by_index=5)

    def test_skipped_parameter_draws_no_random_values(self):
        RNG.seed(7)
        channel_param([]).generate_vectors(10)
        after_skip = RNG.integer(0, 10**9)
        RNG.seed(7)

        assert RNG.integer(0, 10**9) == after_skip

    def test_non_empty_sequence_generates_as_usual(self):
        param = channel_param([3, 7], per_sequence_samples=True)

        assert len(param.generate_vectors(5)) == 10

    def test_to_dict_reports_the_reason(self):
        assert channel_param([]).to_dict()["skip_reason"] == REASON
        assert channel_param([3]).to_dict()["skip_reason"] is None


def test_export_strategies_handles_a_skipped_strategy():
    registry = dict(Strategy._registry)
    try:

        @Strategy.register("skip_if_empty_export")
        def _factory(nsamples):
            return channel_param([])

        data = json.loads(Strategy.export_strategies())["skip_if_empty_export"]
    finally:
        Strategy._registry.clear()
        Strategy._registry.update(registry)

    assert data["skip_reason"] == REASON
    assert data["arguments"][0]["rng_details"]["skip_if_empty"] == REASON
