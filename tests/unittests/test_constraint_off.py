"""
Unit tests for turning constraints off for a run (4.0): the
--strategy-constraint-off items, how the session's options split them into the
names each strategy turns off, the generators' keyword-only ``constraints_off``,
which leaves the Parameter's constraints in place, and the resolver, which turns
off only the names a strategy's Parameter has.
"""

import argparse
import functools
import inspect
from unittest.mock import MagicMock

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    RNGSequence,
    Series,
    StrategyOptions,
    TestArg,
)
from pytest_strategy._options import (
    SessionOptions,
    constraint_off_item,
    parse_constraint_off,
    parse_session_options,
)
from pytest_strategy._resolver import _exhausted_message, build_parametrization
from pytest_strategy._runtime import Resolution, StrategyRuntime
from pytest_strategy.parameters import (
    _constraint_failure,
    _ConstraintsExhausted,
    _GenerationStats,
)
from pytest_strategy.plugin import _constraint_off_type, _summary_lines


def never(v):
    return False


def _recording(name, calls, result=True):
    """Return a constraint that records the rows it is called with under ``name``."""

    def constraint(v):
        calls.append((name, tuple(v)))
        return result

    return constraint


def _mock_config(**options):
    """Return a stand-in config whose getoption() serves the given CLI options."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    return config


class TestParseItems:
    @pytest.mark.parametrize(
        ("value", "pairs"),
        [
            ("never", ((None, "never"),)),
            ("dma_burst:aligned", (("dma_burst", "aligned"),)),
            ("dma_burst:aligned,no_4k_cross", (("dma_burst", "aligned"), (None, "no_4k_cross"))),
            ("a,b", ((None, "a"), (None, "b"))),
            # Split at the last ":", so a strategy name may contain ":"
            ("ns:dma_burst:aligned", (("ns:dma_burst", "aligned"),)),
            ("model.addr", ((None, "model.addr"),)),
        ],
    )
    def test_items(self, value, pairs):
        assert parse_constraint_off(value) == pairs

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (":x", "the item ':x' has no strategy name before ':'"),
            ("S:a,:x", "the item ':x' has no strategy name before ':'"),
            ("x:", "the item 'x:' has no constraint name"),
            ("a b", "'a b' contains whitespace"),
            ("a, b", "'a, b' contains whitespace"),
            ("a\tb", r"'a\\tb' contains whitespace"),
            ("a,,b", "'a,,b' has an empty item"),
            ("a,", "'a,' has an empty item"),
            ("", "'' has an empty item"),
        ],
    )
    def test_malformed_items_fail(self, value, message):
        with pytest.raises(ValueError, match=f"^{message}. Expected ITEM"):
            parse_constraint_off(value)

    def test_the_option_type_raises_an_argparse_error(self):
        with pytest.raises(argparse.ArgumentTypeError, match="no strategy name"):
            _constraint_off_type(":x")

    def test_the_option_type_keeps_the_value(self):
        assert _constraint_off_type("S:a,b") == "S:a,b"

    def test_items_are_written_back(self):
        assert constraint_off_item(None, "never") == "never"
        assert constraint_off_item("ns:dma", "aligned") == "ns:dma:aligned"
        for value in ("never", "dma_burst:aligned", "ns:dma:aligned"):
            assert constraint_off_item(*parse_constraint_off(value)[0]) == value

    @pytest.mark.parametrize("strategy", ["a,b", "my burst", "tab\tbed", ""])
    def test_a_strategy_an_item_cannot_name_gets_the_bare_name(self, strategy):
        assert constraint_off_item(strategy, "aligned") == "aligned"

    @pytest.mark.parametrize(
        ("strategy", "item"),
        [("dma_burst", "dma_burst:aligned"), ("a,b", "aligned"), ("my burst", "aligned")],
    )
    def test_the_exhausted_advice_names_an_item_the_option_accepts(self, strategy, item):
        error = _ConstraintsExhausted("Could not generate random row 0.", "aligned", True)

        message = _exhausted_message(strategy, error)

        assert message.endswith(f"turn one off for this run with --strategy-constraint-off={item}.")
        parse_constraint_off(item)


class TestSessionOptions:
    def test_repeated_values_are_split_into_pairs_in_order(self):
        config = _mock_config(strategy_constraint_off=["S:a,b", "never", "T:c"])

        session = parse_session_options(config)

        assert session.constraints_off == (("S", "a"), (None, "b"), (None, "never"), ("T", "c"))

    def test_a_stand_ins_single_value(self):
        session = parse_session_options(_mock_config(strategy_constraint_off="S:a"))

        assert session.constraints_off == (("S", "a"),)

    def test_without_the_option_nothing_is_off(self):
        assert parse_session_options(_mock_config()).constraints_off == ()
        assert parse_session_options(None).constraints_off == ()

    def test_an_item_aimed_at_one_strategy_and_a_bare_one(self):
        # S:a,b turns a off in S only and b everywhere
        session = parse_session_options(_mock_config(strategy_constraint_off=["S:a,b"]))

        assert session.for_strategy("S").constraints_off == {"a", "b"}
        assert session.for_strategy("T").constraints_off == {"b"}

    def test_the_items_as_written_once_each(self):
        session = SessionOptions(
            base=StrategyOptions(strategy=""),
            constraints_off=(("S", "a"), (None, "b"), ("S", "a"), ("ns:S", "c")),
        )

        assert session.constraints_off_items == ["S:a", "b", "ns:S:c"]

    def test_the_session_options_are_read_once(self):
        rt = StrategyRuntime()
        config = _mock_config(strategy_constraint_off=["a"])
        rt.push(config)
        try:
            session = rt.session_options(config)

            assert rt.session_options(config) is session
            assert rt.strategy_options("S", config).constraints_off == {"a"}
            assert rt.current.options is session
        finally:
            rt.pop()

    def test_another_config_is_read_but_not_cached(self):
        rt = StrategyRuntime()
        rt.push(_mock_config())
        try:
            stand_in = _mock_config(strategy_constraint_off=["a"])

            assert rt.session_options(stand_in).constraints_off == ((None, "a"),)
            assert rt.current.options is None
        finally:
            rt.pop()


class TestGenerators:
    def test_a_constraint_turned_off_is_not_called(self):
        calls = []
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={
                "first": _recording("first", calls),
                "never": never,
                "last": _recording("last", calls),
            },
        )

        rows = param.generate_vectors(3, constraints_off={"never"})

        assert len(rows) == 3
        assert [name for name, _ in calls] == ["first", "last"] * 3

    def test_the_parameter_keeps_its_constraints(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "small": lambda v: v.x < 5},
        )
        before = dict(param.vector_constraints)

        param.generate_vectors(4, constraints_off=["never"])

        assert dict(param.vector_constraints) == before
        # And a call without constraints_off evaluates them all again
        with pytest.raises(ValueError, match="Rejected by .*: never=100"):
            param.generate_vectors(1)

    def test_the_other_constraints_still_reject(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "small": lambda v: v.x < 5},
        )

        rows = param.generate_vectors(20, constraints_off=("never",))

        assert all(x < 5 for (x,) in rows)

    def test_every_constraint_can_be_turned_off(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"a": never, "b": lambda v: False},
        )

        assert len(param.generate_vectors(3, constraints_off=["a", "b"])) == 3

    def test_without_constraints_off_the_values_are_unchanged(self):
        def make():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                vector_constraints={"even": lambda v: v.x % 2 == 0},
            )

        RNG.seed(7)
        before = make().generate_vectors(10)
        RNG.seed(7)
        after = make().generate_vectors(10, constraints_off=())

        assert before == after

    def test_directed_and_test_vectors_are_the_same(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"odd": (1,), "even": (2,)},
            test_vectors={"three": (3,)},
            vector_constraints={"even": lambda v: v.x % 2 == 0},
        )

        on = param.generate_vectors(0)
        off = param.generate_vectors(0, constraints_off=["even"])

        assert on == off == [(1,), (2,)]
        assert param.generate_vectors(0, mode="test", constraints_off=["even"]) == [(3,)]

    @pytest.mark.parametrize("how", ["series", "per_sequence", "exhaustive"])
    def test_every_generation_path_leaves_it_out(self, how):
        calls = []
        sequence = Series([0, 1]) if how == "series" else RNGSequence([0, 1])
        param = Parameter(
            TestArg("ch", rng_type=sequence),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "seen": _recording("seen", calls)},
            per_sequence_samples=how == "per_sequence",
        )

        if how == "exhaustive":
            rows = param.generate_exhaustive(constraints_off=["never"])
        else:
            rows = param.generate_vectors(2, constraints_off=["never"])

        assert rows and len(calls) == len(rows)
        assert {name for name, _ in calls} == {"seen"}

    def test_the_exhausted_error_counts_the_constraints_still_on(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "big": lambda v: v.x > 9},
            max_retries=5,
        )

        with pytest.raises(ValueError) as excinfo:
            param.generate_vectors(1, constraints_off=["never"])

        assert "Rejected by (first failing constraint per draw): big=5. " in str(excinfo.value)
        assert excinfo.value.strictest == "big"

    def test_stats_count_only_the_constraints_still_on(self):
        stats = _GenerationStats()
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "odd": lambda v: v.x % 2 == 1},
        )

        param.generate_vectors(5, constraints_off=["never"], _stats=stats)

        assert set(stats.rejected) <= {"odd"}

    def test_an_unknown_name_fails_listing_the_constraints(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never, "small": lambda v: v.x < 5},
        )

        message = "constraints_off names no constraint 'nevr', 'big'. Constraints: never, small"
        with pytest.raises(ValueError, match=f"^{message}$"):
            param.generate_vectors(1, constraints_off=["nevr", "never", "big"])
        with pytest.raises(ValueError, match="^constraints_off names no constraint 'x'"):
            param.generate_vectors(0, filter_by_name="zeros", constraints_off=["x"])
        with pytest.raises(ValueError, match=r"Constraints: none$"):
            Parameter(TestArg("x", value=1)).generate_vectors(1, constraints_off=["x"])

    def test_a_str_fails(self):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), vector_constraints=[never])

        with pytest.raises(TypeError, match="collection of constraint names, not a str"):
            param.generate_vectors(1, constraints_off="never")
        with pytest.raises(TypeError, match="collection of constraint names, not a str"):
            param.generate_exhaustive(constraints_off="never")

    def test_generate_exhaustive_checks_the_names_too(self):
        param = Parameter(TestArg("ch", rng_type=Series([0, 1])), vector_constraints=[never])

        with pytest.raises(ValueError, match="constraints_off names no constraint 'nevr'"):
            param.generate_exhaustive(constraints_off=["nevr"])
        assert param.generate_exhaustive(constraints_off=["never"]) == [(0,), (1,)]

    @pytest.mark.parametrize("method", ["generate_vectors", "generate_exhaustive"])
    def test_constraints_off_is_keyword_only(self, method):
        parameter = inspect.signature(getattr(Parameter, method)).parameters["constraints_off"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == ()


class TestRaisingConstraintNote:
    def _ratio(self):
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 9)),
            TestArg("len", value=0),
            vector_constraints={
                "nonzero": lambda v: v.len != 0,
                "aligned": lambda v: v.addr % 1 == 0,
                "ratio": lambda v: 64 / v.len > 4,
                "last": never,
            },
        )

    def test_a_constraint_turned_off_before_it_is_named(self):
        with pytest.raises(ZeroDivisionError) as excinfo:
            self._ratio().generate_vectors(1, constraints_off=["nonzero", "last"])

        [note] = excinfo.value.__notes__
        assert note.startswith("Raised by constraint 'ratio' on random row 0, Vector(addr=")
        assert note.endswith(
            "len=0) (constraint 'nonzero' before it is turned off by constraints_off)"
        )
        assert _constraint_failure(excinfo.value).off_before == ("nonzero",)

    def test_the_resolver_names_the_option(self):
        with pytest.raises(ZeroDivisionError) as excinfo:
            self._ratio().generate_vectors(1, constraints_off=["nonzero", "aligned"])

        message = _constraint_failure(excinfo.value).message("--strategy-constraint-off")
        assert message.startswith(
            "Constraint 'ratio' raised ZeroDivisionError on random row 0, Vector(addr="
        )
        assert message.endswith(
            "len=0): division by zero (constraints 'nonzero', 'aligned' before it are "
            "turned off by --strategy-constraint-off)"
        )

    def test_no_note_when_none_before_it_is_off(self):
        with pytest.raises(ZeroDivisionError) as excinfo:
            Parameter(
                TestArg("len", value=0),
                vector_constraints={"ratio": lambda v: 64 / v.len > 4, "after": never},
            ).generate_vectors(1, constraints_off=["after"])

        assert excinfo.value.__notes__ == [
            "Raised by constraint 'ratio' on random row 0, Vector(len=0)"
        ]
        assert _constraint_failure(excinfo.value).off_before == ()


class TestResolver:
    def _factory(self, calls):
        def factory():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                directed_vectors={"odd": (1,)},
                vector_constraints={"never": never, "seen": _recording("seen", calls)},
            )

        return factory

    def test_the_names_the_parameter_has_are_turned_off(self):
        calls = []

        def test_fn(x):
            pass

        parametrization = build_parametrization(
            "co_unit_burst",
            self._factory(calls),
            test_fn,
            config=_mock_config(nsamples=3, strategy_constraint_off=["co_unit_burst:never"]),
            pytest_fixtures=set(),
        )

        assert parametrization.values[0] == 1
        assert len(parametrization.values) == 4
        assert len(calls) == 3

    def test_names_the_parameter_does_not_have_are_ignored(self):
        calls = []

        def test_fn(x):
            pass

        # "other" is no constraint of this strategy, and "never" is aimed at another
        config = _mock_config(nsamples=2, strategy_constraint_off=["other,elsewhere:never"])
        with pytest.raises(ValueError, match="Rejected by .*: never=100"):
            build_parametrization(
                "co_unit_other",
                self._factory(calls),
                test_fn,
                config=config,
                pytest_fixtures=set(),
            )

    def test_a_raising_constraint_names_the_option(self):
        def factory():
            return Parameter(
                TestArg("len", value=0),
                vector_constraints={"nonzero": lambda v: v.len != 0, "ratio": lambda v: 1 / v.len},
            )

        def test_fn(len):
            pass

        with pytest.raises(ValueError) as excinfo:
            build_parametrization(
                "co_unit_ratio",
                factory,
                test_fn,
                config=_mock_config(strategy_constraint_off=["nonzero"]),
                pytest_fixtures=set(),
            )

        assert str(excinfo.value) == (
            "Error generating samples for strategy 'co_unit_ratio': Constraint 'ratio' raised "
            "ZeroDivisionError on random row 0, Vector(len=0): division by zero (constraint "
            "'nonzero' before it is turned off by --strategy-constraint-off)"
        )
        assert isinstance(excinfo.value.__cause__, ZeroDivisionError)

    def test_a_cached_factorys_parameter_keeps_its_constraints(self):
        @functools.cache
        def factory():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                vector_constraints={"never": never, "small": lambda v: v.x < 5},
            )

        def test_fn(x):
            pass

        build_parametrization(
            "co_unit_cached",
            factory,
            test_fn,
            config=_mock_config(nsamples=2, strategy_constraint_off=["never"]),
            pytest_fixtures=set(),
        )

        assert list(factory().vector_constraints) == ["never", "small"]


class TestSummary:
    def test_off_lists_the_constraints_turned_off_and_rejected_the_others(self):
        resolution = Resolution(
            strategy="dma_burst",
            where="strategies.py",
            random=10,
            nsamples=10,
            source="default",
            constraints=("aligned", "never", "no_4k_cross"),
            rejected={"aligned": 4},
            constraints_off=("never",),
        )

        assert _summary_lines([resolution]) == [
            "dma_burst (strategies.py): 1 test(s), 0 directed, 10 random rows; nsamples=10 "
            "from default; rejected: aligned=4, no_4k_cross=0; off: never"
        ]

    def test_every_constraint_off(self):
        resolution = Resolution(
            strategy="s",
            where="strategies.py",
            random=3,
            constraints=("a", "b"),
            constraints_off=("a", "b"),
        )

        assert _summary_lines([resolution]) == [
            "s (strategies.py): 1 test(s), 0 directed, 3 random rows; off: a, b"
        ]
