"""
Unit tests for StrategyOptions: the run's options as a strategy factory sees them,
read from the config once per session and cached per resolved strategy name.
"""

import dataclasses
import inspect
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest

import pytest_strategy
from pytest_strategy import Parameter, RNGInteger, StrategyOptions, TestArg
from pytest_strategy._options import SessionOptions, VectorMode, parse_session_options
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._runtime import StrategyRuntime, runtime


class CountingConfig:
    """A stand-in config whose getoption() serves the given CLI options and counts calls."""

    def __init__(self, **options):
        self.values = {
            "nsamples": None,
            "vector_mode": "all",
            "vector_name": None,
            "vector_index": None,
            **options,
        }
        self.calls = []

    def getoption(self, name, default=None):
        self.calls.append(name)
        return self.values.get(name, default)


def _mock_config(**options):
    """Return a MagicMock config like the resolver tests' (getoption serves the options)."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    return config


def _fields(options):
    """Return the options' fields and the filtered property, for one comparison."""
    return {**dataclasses.asdict(options), "filtered": options.filtered}


DEFAULTS = {
    "nsamples": 10,
    "nsamples_source": "default",
    "mode": "all",
    "vector_name": None,
    "vector_index": None,
    "constraints_off": frozenset(),
    "filtered": False,
}


class TestStrategyOptions:
    def test_defaults(self):
        assert _fields(StrategyOptions(strategy="s")) == {"strategy": "s", **DEFAULTS}

    @pytest.mark.parametrize(
        ("options", "filtered"),
        [
            ({"vector_name": "zeros"}, True),
            ({"vector_index": 0}, True),
            ({"vector_name": "zeros", "vector_index": 1}, True),
            ({"mode": "directed_only"}, False),
        ],
    )
    def test_filtered_by_vector_name_or_index(self, options, filtered):
        assert StrategyOptions(strategy="s", **options).filtered is filtered

    @pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(StrategyOptions)])
    def test_assignment_raises_frozen_instance_error(self, field):
        options = StrategyOptions(strategy="s")

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(options, field, None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(options, field)

    def test_positional_construction_raises_type_error(self):
        with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
            StrategyOptions("s")

    def test_every_field_is_keyword_only(self):
        parameters = list(inspect.signature(StrategyOptions).parameters.values())

        assert [p.name for p in parameters] == [f.name for f in dataclasses.fields(StrategyOptions)]
        assert {p.kind for p in parameters} == {inspect.Parameter.KEYWORD_ONLY}

    def test_strategy_is_required(self):
        with pytest.raises(TypeError, match="'strategy'"):
            StrategyOptions()

    def test_slots_and_hashable(self):
        options = StrategyOptions(strategy="s", constraints_off=frozenset({"aligned"}))

        assert not hasattr(options, "__dict__")
        assert options == StrategyOptions(strategy="s", constraints_off=frozenset({"aligned"}))
        assert hash(options) == hash(
            StrategyOptions(strategy="s", constraints_off=frozenset({"aligned"}))
        )

    def test_exported_from_the_package(self):
        assert "StrategyOptions" in pytest_strategy.__all__
        assert pytest_strategy.StrategyOptions is StrategyOptions

    def test_vector_mode_values(self):
        assert get_args(VectorMode) == ("all", "random_only", "directed_only", "mixed", "test")


class TestParseSessionOptions:
    def test_no_config_gives_the_defaults(self):
        assert _fields(parse_session_options(None).for_strategy("s")) == {
            "strategy": "s",
            **DEFAULTS,
        }

    def test_simple_namespace_config_gives_the_defaults(self):
        # The stand-in configs of the ctx tests have a hook and nothing else
        config = SimpleNamespace(hook=SimpleNamespace(pytest_strategies_context=lambda config: 1))

        assert parse_session_options(config).for_strategy("s") == StrategyOptions(strategy="s")

    def test_options_the_config_does_not_have_give_the_defaults(self):
        config = MagicMock()
        config.getoption.side_effect = lambda opt, default=None: default

        assert parse_session_options(config).for_strategy("s") == StrategyOptions(strategy="s")

    def test_directed_only_with_a_vector_name_and_nsamples(self):
        config = _mock_config(nsamples=3, vector_mode="directed_only", vector_name="zeros")

        assert _fields(parse_session_options(config).for_strategy("dma_burst")) == {
            "strategy": "dma_burst",
            "nsamples": 3,
            "nsamples_source": "--nsamples",
            "mode": "directed_only",
            "vector_name": "zeros",
            "vector_index": None,
            "constraints_off": frozenset(),
            "filtered": True,
        }

    def test_nsamples_auto(self):
        options = parse_session_options(_mock_config(nsamples="auto")).for_strategy("s")

        assert (options.nsamples, options.nsamples_source) == ("auto", "--nsamples")

    def test_an_explicit_nsamples_equal_to_the_default_is_from_the_option(self):
        # --nsamples=10 overrides Parameter(nsamples=), so it is not the default
        options = parse_session_options(_mock_config(nsamples=10)).for_strategy("s")

        assert (options.nsamples, options.nsamples_source) == (10, "--nsamples")

    def test_nsamples_given_as_text_by_a_stand_in(self):
        options = parse_session_options(_mock_config(nsamples="5")).for_strategy("s")

        assert (options.nsamples, options.nsamples_source) == (5, "--nsamples")

    def test_vector_index(self):
        options = parse_session_options(_mock_config(vector_index=0)).for_strategy("s")

        assert (options.vector_index, options.filtered) == (0, True)

    def test_a_stand_ins_unknown_mode_is_kept(self):
        # generate_vectors() reports it, as before
        options = parse_session_options(_mock_config(vector_mode="bad")).for_strategy("s")

        assert options.mode == "bad"


class TestForStrategy:
    def test_each_strategy_gets_its_name_and_the_shared_options(self):
        session = SessionOptions(
            base=StrategyOptions(strategy="", nsamples=3, nsamples_source="--nsamples")
        )

        burst = session.for_strategy("dma_burst")

        assert burst == StrategyOptions(
            strategy="dma_burst", nsamples=3, nsamples_source="--nsamples"
        )
        assert session.base.strategy == ""

    def test_constraints_off_are_the_bare_names_and_those_aimed_at_the_strategy(self):
        session = SessionOptions(
            base=StrategyOptions(strategy=""),
            constraints_off=(
                ("dma_burst", "aligned"),
                (None, "no_4k_cross"),
                ("esm", "nonzero"),
                ("ns:dma_burst", "within"),
            ),
        )

        assert session.for_strategy("dma_burst").constraints_off == {"aligned", "no_4k_cross"}
        assert session.for_strategy("esm").constraints_off == {"nonzero", "no_4k_cross"}
        assert session.for_strategy("ns:dma_burst").constraints_off == {"within", "no_4k_cross"}
        assert session.for_strategy("other").constraints_off == {"no_4k_cross"}


class TestSessionCache:
    def test_the_session_part_is_read_once_and_each_name_gets_one_instance(self):
        rt = StrategyRuntime()
        config = CountingConfig(nsamples=3, vector_name="zeros")
        rt.push(config)
        try:
            first = rt.strategy_options("dma_burst", config)
            reads = list(config.calls)
            again = rt.strategy_options("dma_burst", config)
            other = rt.strategy_options("esm", config)

            assert again is first
            assert other is not first
            assert (first.strategy, other.strategy) == ("dma_burst", "esm")
            assert (other.nsamples, other.vector_name) == (3, "zeros")
            # Read once for the session: the second name reads nothing
            assert sorted(reads) == ["nsamples", "vector_index", "vector_mode", "vector_name"]
            assert config.calls == reads
            assert rt.current.strategy_options == {"dma_burst": first, "esm": other}
        finally:
            rt.pop()

    def test_another_config_is_read_but_not_cached(self):
        rt = StrategyRuntime()
        session_config = CountingConfig()
        rt.push(session_config)
        try:
            stand_in = CountingConfig(nsamples=7)
            first = rt.strategy_options("s", stand_in)
            second = rt.strategy_options("s", stand_in)

            assert first == second and first is not second
            assert first.nsamples == 7
            assert rt.current.options is None
            assert rt.current.strategy_options == {}
            assert session_config.calls == []
        finally:
            rt.pop()

    def test_none_and_no_session_give_new_instances(self):
        rt = StrategyRuntime()
        config = CountingConfig(nsamples=4)

        assert rt.strategy_options("s", config).nsamples == 4
        assert rt.strategy_options("s", None) == StrategyOptions(strategy="s")

        rt.push(None)
        try:
            assert rt.strategy_options("s", None) is not rt.strategy_options("s", None)
            assert rt.current.strategy_options == {}
        finally:
            rt.pop()

    def test_nested_sessions_have_their_own_options(self):
        rt = StrategyRuntime()
        outer_config = CountingConfig(nsamples=3)
        inner_config = CountingConfig(nsamples=5)
        rt.push(outer_config)
        try:
            outer = rt.strategy_options("s", outer_config)
            rt.push(inner_config)
            try:
                assert rt.strategy_options("s", inner_config).nsamples == 5
            finally:
                rt.pop()

            assert rt.strategy_options("s", outer_config) is outer
            assert outer.nsamples == 3
        finally:
            rt.pop()


class TestResolverUsesTheOptions:
    def test_a_stand_in_config_leaves_the_running_sessions_cache_alone(self):
        cached = dict(runtime.current.strategy_options)

        def factory(nsamples):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        def test_fn(x):
            pass

        parametrization = build_parametrization(
            "so_unit_stand_in",
            factory,
            test_fn,
            config=_mock_config(nsamples=4),
            pytest_fixtures=set(),
        )

        assert len(parametrization.values) == 4
        assert runtime.current.strategy_options == cached
