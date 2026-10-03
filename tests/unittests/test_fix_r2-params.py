"""
Regression tests for Parameter, RNGEnum and TestArg.to_dict review fixes.

Covers:
- Parameter(nsamples="auto") accepted again (factories pass --nsamples=auto through)
- generate_vectors rejecting a non-int n instead of looping forever on Series args
- Finite-mode Series warning when a combination is skipped after random redraws
- generate_exhaustive raising when the constraints reject every combination
- RNGEnum accepting a Flag whose iteration is empty when weights name members
- RNGEnum reading predicate and weights when drawing, not only at construction
- TestArg.to_dict exporting the Enum class of an RNGEnum and whether a predicate is set
"""

import json
import random
import warnings
from enum import Enum, Flag
from unittest.mock import MagicMock

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGEnum,
    RNGInteger,
    RNGSequence,
    RNGValueError,
    Series,
    Strategy,
    TestArg,
)
from pytest_strategy._resolver import build_parametrization
from pytest_strategy.strategy import PytestStrategiesWarning


class Color(Enum):
    """Small enum used by the RNGEnum tests"""

    RED = 1
    GREEN = 2
    BLUE = 3


class Mode(Flag):
    """A Flag whose members are all zero-valued or multi-bit"""

    NONE = 0
    BOTH = 3


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo global changes: the strategy registry, the RNG seed and the random state."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


def _strategy_skip_warnings(record):
    return [w for w in record if issubclass(w.category, PytestStrategiesWarning)]


class TestNsamplesAutoAccepted:
    """Parameter(nsamples="auto") works again, so factories can forward --nsamples=auto."""

    def test_auto_accepted(self):
        param = Parameter(TestArg("x", rng_type=Series([1, 2, 3])), nsamples="auto")
        assert param.nsamples == "auto"
        assert param.to_dict()["nsamples"] == "auto"

    @pytest.mark.parametrize("nsamples", ["AUTO", "all", "10", "", 1.0])
    def test_other_values_still_rejected(self, nsamples):
        with pytest.raises(
            ValueError, match='nsamples must be None or an int >= 0 \\(or "auto"\\)'
        ):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), nsamples=nsamples)

    @staticmethod
    def _resolve(factory, cli_nsamples):
        config = MagicMock()
        config.getoption.side_effect = lambda opt, default=None: {
            "nsamples": cli_nsamples,
            "vector_mode": "all",
            "vector_name": None,
            "vector_index": None,
        }.get(opt, default)

        def test_fn(x):
            pass

        return build_parametrization(
            "fix_r2_auto", factory, test_fn, config=config, pytest_fixtures=set()
        ).values

    def test_factory_forwarding_cli_auto_is_exhaustive(self):
        """--nsamples=auto reaches the factory as "auto"; forwarding it used to raise."""

        def forward(nsamples):
            return Parameter(TestArg("x", rng_type=Series([1, 2, 3])), nsamples=nsamples)

        assert self._resolve(forward, "auto") == [1, 2, 3]
        assert self._resolve(forward, "5") == [1, 2, 3, 1, 2]

    def test_per_strategy_auto_default_is_exhaustive(self):
        """A strategy defaulting to "auto" enumerates its Series without any CLI flag."""

        def auto_default(nsamples):
            return Parameter(TestArg("x", rng_type=Series(["a", "b"])), nsamples="auto")

        assert self._resolve(auto_default, None) == ["a", "b"]


class TestGenerateVectorsRejectsNonIntN:
    """A non-int n used to make the Series loop spin forever with growing memory."""

    @staticmethod
    def _param_with_loop_guard(*extra_args):
        calls = []

        def guard(vector):
            calls.append(vector)
            if len(calls) > 1000:
                raise RuntimeError("generate_vectors kept looping")
            return True

        return Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            *extra_args,
            vector_constraints=[guard],
        )

    @pytest.mark.parametrize("n", [2.5, 3.0, "3", None, True, False])
    @pytest.mark.parametrize("mode", ["all", "random_only", "mixed"])
    def test_series_non_int_n_raises(self, n, mode):
        param = self._param_with_loop_guard(TestArg("r", rng_type=RNGInteger(0, 9)))
        with pytest.raises(ValueError, match="n must be an int, got"):
            param.generate_vectors(n, mode=mode)

    def test_series_only_non_int_n_raises(self):
        param = self._param_with_loop_guard()
        with pytest.raises(ValueError, match="n must be an int, got 2.5"):
            param.generate_vectors(2.5, mode="random_only")

    @pytest.mark.parametrize("n", [2.5, 3.0, True])
    def test_random_path_raises_the_same_error(self, n):
        """Both generation paths reject a non-int n the same way."""
        param = Parameter(TestArg("r", rng_type=RNGInteger(0, 9)))
        with pytest.raises(ValueError, match="n must be an int, got"):
            param.generate_vectors(n, mode="random_only")

    def test_int_n_still_works(self):
        param = self._param_with_loop_guard(TestArg("r", rng_type=RNGInteger(0, 9)))
        assert [v[0] for v in param.generate_vectors(4, mode="random_only")] == [1, 2, 3, 1]


class TestSeriesSkipAfterRedrawsWarns:
    """A Series combination skipped after max_retries unlucky redraws is reported."""

    @staticmethod
    def _rare_constraint_param():
        return Parameter(
            TestArg("mode", rng_type=Series(["a", "b", "c"])),
            TestArg("x", rng_type=RNGInteger(0, 99)),
            vector_constraints=[lambda v: v[1] > 97],
        )

    def test_wrong_series_rows_never_silent(self):
        """Over 200 seeds, 68 used to return wrong Series rows with no error or warning."""
        warned_seeds = 0
        for seed in range(200):
            RNG.seed(seed)
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                try:
                    rows = [v[0] for v in self._rare_constraint_param().generate_vectors(3)]
                except ValueError:
                    continue
            messages = [str(w.message) for w in _strategy_skip_warnings(record)]
            if rows != ["a", "b", "c"]:
                assert messages, f"seed {seed}: rows {rows} without a warning"
                warned_seeds += 1
            # Every Series value left untested is named by a warning
            for value in {"a", "b", "c"} - set(rows):
                assert any(f"(mode={value!r}) skipped" in m for m in messages), seed
        assert warned_seeds > 0

    def test_one_warning_per_combination_per_call(self):
        """A combination skipped on every cycle is reported once, naming it and max_retries."""
        RNG.seed(0)
        param = Parameter(
            TestArg("mode", rng_type=Series(["a", "b", "c"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda v: v[0] != "a"],
            max_retries=5,
        )
        with pytest.warns(PytestStrategiesWarning) as record:
            samples = param.generate_vectors(6)
        assert [s[0] for s in samples] == ["b", "c"] * 3
        messages = [str(w.message) for w in _strategy_skip_warnings(record)]
        assert len(messages) == 1
        assert "Series combination (mode='a') skipped" in messages[0]
        assert "max_retries=5" in messages[0]

        # A second call warns again
        with pytest.warns(PytestStrategiesWarning, match=r"\(mode='a'\)"):
            param.generate_vectors(2)

    def test_warning_names_every_series_arg(self):
        RNG.seed(0)
        param = Parameter(
            TestArg("lo", rng_type=Series([1, 2])),
            TestArg("pad", rng_type=RNGInteger(0, 9)),
            TestArg("hi", rng_type=Series(["x", "y"])),
            vector_constraints=[lambda v: (v[0], v[2]) != (2, "x")],
        )
        with pytest.warns(PytestStrategiesWarning, match=r"\(lo=2, hi='x'\) skipped"):
            param.generate_vectors(3)

    def test_series_only_constraint_skips_silently(self):
        """Without random args no redraw happens, so the skip stays silent."""
        param = Parameter(
            TestArg("lo", rng_type=Series([1, 2, 3])),
            TestArg("hi", rng_type=Series([1, 2, 3])),
            vector_constraints=[lambda v: v[0] < v[1]],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert param.generate_vectors(3) == [(1, 2), (1, 3), (2, 3)]

    def test_no_warning_when_every_combination_fails(self):
        """The ValueError already explains an unsatisfiable constraint."""
        param = Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            TestArg("i", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda v: False],
            max_retries=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="none of the 3 Series combinations"):
                param.generate_vectors(4)

    def test_no_warning_when_redraws_succeed(self):
        RNG.seed(0)
        param = Parameter(
            TestArg("role", rng_type=Series(["admin", "user", "guest"])),
            TestArg("uid", rng_type=RNGInteger(1, 1000)),
            vector_constraints=[lambda v: v[1] > 500],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = param.generate_vectors(6)
        assert [s[0] for s in samples] == ["admin", "user", "guest"] * 2


class TestExhaustiveUnsatisfiableRaises:
    """--nsamples=auto used to yield an empty parameter set that pytest silently skipped."""

    def test_all_series_unsatisfiable_raises(self):
        param = Parameter(
            TestArg("x", rng_type=Series([1, 2, 3])),
            TestArg("y", rng_type=Series([1, 2])),
            vector_constraints=[lambda v: v[0] + v[1] > 100],
        )
        with pytest.raises(
            ValueError,
            match=r"none of the 6 sequence combinations satisfied the vector "
            r"constraints \(1 attempt\(s\) each\)",
        ):
            param.generate_exhaustive()

    def test_sequence_plus_random_unsatisfiable_raises(self):
        param = Parameter(
            TestArg("s", rng_type=RNGSequence([1, 2, 3])),
            TestArg("i", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda v: v[1] > 50],
            max_retries=4,
        )
        with pytest.raises(ValueError, match=r"none of the 3 sequence combinations.*4 attempt"):
            param.generate_exhaustive()

    def test_partially_satisfiable_still_filters(self):
        param = Parameter(
            TestArg("x", rng_type=Series([1, 2, 3])),
            vector_constraints=[lambda v: v[0] == 3],
        )
        assert param.generate_exhaustive() == [(3,)]


class TestRNGEnumFlagWithoutCanonicalMembers:
    """Iterating such a Flag yields nothing, but weights can still name members."""

    def test_weights_naming_members_accepted(self):
        RNG.seed(0)
        rng_enum = RNGEnum(Mode, weights={Mode.BOTH: 1, Mode.NONE: 1})
        assert {rng_enum.generate() for _ in range(50)} == {Mode.BOTH, Mode.NONE}

    def test_weights_with_predicate_accepted(self):
        RNG.seed(0)
        rng_enum = RNGEnum(Mode, weights={Mode.BOTH: 1, Mode.NONE: 1}, predicate=bool)
        assert {rng_enum.generate() for _ in range(50)} == {Mode.BOTH}

    @pytest.mark.parametrize("predicate", [None, bool])
    def test_without_weights_rejected(self, predicate):
        with pytest.raises(RNGValueError, match="Mode has no members to choose from"):
            RNGEnum(Mode, predicate=predicate)

    def test_empty_enum_still_rejected_with_weights(self):
        class Empty(Enum):
            pass

        with pytest.raises(RNGValueError, match="RNGEnum weights cannot be empty"):
            RNGEnum(Empty, weights={})


class TestRNGEnumReadsPredicateAndWeightsLive:
    """predicate/weights set after construction used to crash generate() or be ignored."""

    def test_predicate_set_after_construction(self):
        """This used to raise IndexError: the members were only filtered in __init__."""
        RNG.seed(0)
        rng_enum = RNGEnum(Color)
        rng_enum.predicate = lambda c: c is not Color.RED
        assert {rng_enum.generate() for _ in range(100)} == {Color.GREEN, Color.BLUE}

    def test_predicate_replaced_after_construction(self):
        RNG.seed(0)
        rng_enum = RNGEnum(Color, predicate=lambda c: c is not Color.RED)
        rng_enum.predicate = lambda c: c is Color.RED
        assert {rng_enum.generate() for _ in range(50)} == {Color.RED}

    def test_weights_replaced_after_construction(self):
        RNG.seed(0)
        rng_enum = RNGEnum(Color, predicate=lambda c: c is not Color.RED)
        rng_enum.weights = {Color.BLUE: 1.0}
        assert {rng_enum.generate() for _ in range(100)} == {Color.BLUE}

    def test_weights_changed_in_place(self):
        RNG.seed(0)
        weights = {Color.RED: 1.0, Color.GREEN: 1.0, Color.BLUE: 1.0}
        rng_enum = RNGEnum(Color, weights=weights, predicate=lambda c: c is not Color.RED)
        weights[Color.GREEN] = 0.0
        assert {rng_enum.generate() for _ in range(100)} == {Color.BLUE}

    def test_unsatisfiable_predicate_set_later_raises_rng_error(self):
        rng_enum = RNGEnum(Color)
        rng_enum.predicate = lambda c: False
        with pytest.raises(RNGValueError, match="No valid value found: no member of Color"):
            rng_enum.generate()

    def test_unsatisfiable_predicate_still_raises_at_construction(self):
        with pytest.raises(RNGValueError, match="No valid value found"):
            RNGEnum(Color, predicate=lambda c: False)

    @pytest.mark.parametrize("seed", [0, 42, 2024])
    def test_stream_unchanged_with_predicate(self, seed):
        """Each draw is still one random call over the accepted members, in member order."""
        weights = {Color.RED: 0.5, Color.GREEN: 0.3, Color.BLUE: 0.2}
        accepted = [Color.GREEN, Color.BLUE]

        random.seed(seed)
        expected_uniform = [random.choice(accepted) for _ in range(50)]
        expected_weighted = [
            random.choices(accepted, weights=[0.3, 0.2], k=1)[0] for _ in range(50)
        ]

        RNG.seed(seed)
        uniform = RNGEnum(Color, predicate=lambda c: c is not Color.RED)
        weighted = RNGEnum(Color, weights=weights, predicate=lambda c: c is not Color.RED)
        assert [uniform.generate() for _ in range(50)] == expected_uniform
        assert [weighted.generate() for _ in range(50)] == expected_weighted


class TestToDictDescribesRNGConfiguration:
    """The export used to drop the Enum class and report a set predicate as no predicate."""

    def test_rng_enum_without_predicate(self):
        rng = TestArg("color", rng_type=RNGEnum(Color)).to_dict()["rng"]
        assert rng == {
            "type": "RNGEnum",
            "enum": "Color",
            "members": [{"$enum": "Color", "member": member.name} for member in Color],
            "weights": None,
            "predicate": False,
        }

    def test_rng_enum_with_predicate(self):
        rng_type = RNGEnum(Color, predicate=lambda c: c is not Color.RED)
        rng = TestArg("color", rng_type=rng_type).to_dict()["rng"]
        assert rng["enum"] == "Color"
        assert rng["predicate"] is True

    def test_rng_integer_keys_kept(self):
        plain = TestArg("n", rng_type=RNGInteger(0, 10)).to_dict()["rng"]
        filtered = TestArg("n", rng_type=RNGInteger(0, 10, predicate=bool)).to_dict()["rng"]
        assert plain == {"type": "RNGInteger", "min": 0, "max": 10, "predicate": False}
        assert filtered == {"type": "RNGInteger", "min": 0, "max": 10, "predicate": True}

    def test_rng_without_predicate_attribute_has_no_flag(self):
        rng = TestArg("s", rng_type=Series([1, 2])).to_dict()["rng"]
        assert "predicate" not in rng

    def test_export_strategies_is_json_with_details(self):
        @Strategy.register("fix_r2_export")
        def factory(nsamples):
            return Parameter(
                TestArg("color", rng_type=RNGEnum(Color, predicate=lambda c: c is not Color.RED)),
            )

        data = json.loads(Strategy.export_strategies())
        [entry] = [entry for entry in data["strategies"] if entry["name"] == "fix_r2_export"]
        rng = entry["parameter"]["arguments"][0]["rng"]
        assert rng["enum"] == "Color"
        assert rng["predicate"] is True
