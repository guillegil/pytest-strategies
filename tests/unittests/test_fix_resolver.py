"""
Regression tests for resolver and strategy registration fixes.

Most tests mock the pytest.Config and registry to drive resolve_and_parametrize
directly and inspect the pytest.mark.parametrize it applies.
"""

import inspect
import json
import textwrap
import warnings
from unittest.mock import MagicMock

import pytest

from pytest_strategy import RNGInteger, Strategy
from pytest_strategy._resolver import call_factory, resolve_and_parametrize
from pytest_strategy.parameters import Parameter
from pytest_strategy.rng import RNGChoice, Series
from pytest_strategy.strategy import PytestStrategiesWarning
from pytest_strategy.test_args import TestArg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**options):
    """Return a mock pytest.Config whose getoption() serves the given CLI options."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    return config


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    _fn.__signature__ = inspect.Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


def _resolve(factory, argnames, *, validate=False, **options):
    """Resolve ``factory`` for a test taking ``argnames`` under the given CLI options.

    Returns the (argstr, samples, ids) passed to pytest.mark.parametrize.
    """
    marked = resolve_and_parametrize(
        "strat",
        _make_test_fn(argnames),
        registry={"strat": factory},
        config=_make_config(**options),
        pytest_fixtures=set(),
        validate=validate,
    )
    mark = marked.pytestmark[-1]
    return mark.args[0], list(mark.args[1]), mark.kwargs["ids"]


def _series_param(**kwargs):
    """Two Series args (3x2 product) plus one directed and one test vector."""
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"])),
        directed_vectors={"corner": (99, "z")},
        test_vectors={"tv": (42, "t")},
        **kwargs,
    )


def _random_param(**kwargs):
    """A single random (non-sequence) arg plus one directed and one test vector."""
    return Parameter(
        TestArg("code", rng_type=RNGInteger(200, 400)),
        directed_vectors={"corner": (500,)},
        test_vectors={"ok": (200,)},
        **kwargs,
    )


PRODUCT = [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")]


# ---------------------------------------------------------------------------
# --nsamples=auto honours --vector-mode, --vector-name and --vector-index
# ---------------------------------------------------------------------------


class TestAutoModeHonoursVectorOptions:
    """'auto' replaces only the random part of generation; modes and filters still apply."""

    def test_all_mode_prepends_directed_vectors(self):
        _, samples, _ = _resolve(lambda nsamples: _series_param(), ["x", "y"], nsamples="auto")
        assert samples == [(99, "z")] + PRODUCT

    def test_mixed_mode_prepends_directed_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_mode="mixed"
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_mixed_mode_without_always_include_directed(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(always_include_directed=False),
            ["x", "y"],
            nsamples="auto",
            vector_mode="mixed",
        )
        assert samples == PRODUCT

    def test_random_only_mode_is_exhaustive_rows_only(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="random_only",
        )
        assert samples == PRODUCT

    def test_test_mode_returns_test_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_mode="test"
        )
        assert samples == [(42, "t")]

    def test_directed_only_mode_returns_directed_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="directed_only",
        )
        assert samples == [(99, "z")]

    def test_vector_name_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_name="corner"
        )
        assert samples == [(99, "z")]

    def test_vector_index_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_index=0
        )
        assert samples == [(99, "z")]

    def test_vector_name_filter_takes_precedence_over_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="test",
            vector_name="corner",
        )
        assert samples == [(99, "z")]

    def test_missing_vector_name_gives_empty_samples(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_name="nope"
        )
        assert samples == []


class TestAutoModeWithoutSequenceArgs:
    """A Parameter with no Series/RNGSequence arg falls back to finite generation."""

    def test_falls_back_to_param_nsamples(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(nsamples=4), ["code"], nsamples="auto"
        )
        assert samples[0] == 500
        assert len(samples) == 1 + 4
        assert all(200 <= s <= 400 for s in samples[1:])

    def test_falls_back_to_default_of_10(self):
        _, samples, _ = _resolve(lambda nsamples: _random_param(), ["code"], nsamples="auto")
        assert samples[0] == 500
        assert len(samples) == 1 + 10

    def test_fallback_honours_random_only_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(nsamples=4),
            ["code"],
            nsamples="auto",
            vector_mode="random_only",
        )
        assert len(samples) == 4
        assert all(200 <= s <= 400 for s in samples)

    def test_test_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_mode="test"
        )
        assert samples == [200]

    def test_directed_only_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(),
            ["code"],
            nsamples="auto",
            vector_mode="directed_only",
        )
        assert samples == [500]

    def test_vector_name_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_name="corner"
        )
        assert samples == [500]

    def test_invalid_mode_still_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _resolve(lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_mode="bad")


# ---------------------------------------------------------------------------
# --vector-index out of range behaves like a missing --vector-name
# ---------------------------------------------------------------------------


class TestVectorIndexOutOfRange:
    """A strategy without the requested index gets an empty sample list, not an error."""

    def test_index_beyond_directed_vectors(self):
        _, samples, _ = _resolve(lambda nsamples: _random_param(), ["code"], vector_index=1)
        assert samples == []

    def test_strategy_without_directed_vectors(self):
        param = Parameter(TestArg("w", rng_type=RNGInteger(0, 9)))
        _, samples, _ = _resolve(lambda nsamples: param, ["w"], vector_index=0)
        assert samples == []

    def test_index_beyond_directed_vectors_under_auto(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_index=5
        )
        assert samples == []

    def test_index_in_range_still_selects_vector(self):
        _, samples, _ = _resolve(lambda nsamples: _random_param(), ["code"], vector_index=0)
        assert samples == [500]

    def test_index_error_without_filter_still_raises(self):
        class BrokenParameter(Parameter):
            def generate_vectors(self, *args, **kwargs):
                raise IndexError("boom")

        param = BrokenParameter(TestArg("w", rng_type=RNGInteger(0, 9)))
        with pytest.raises(ValueError, match="Error generating samples for strategy 'strat'"):
            _resolve(lambda nsamples: param, ["w"])


# ---------------------------------------------------------------------------
# Factory calling: chosen from the signature, called once, real errors surfaced
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_registry():
    """Restore Strategy._registry after a test that registers strategies."""
    saved = dict(Strategy._registry)
    yield
    Strategy._registry.clear()
    Strategy._registry.update(saved)


class TestCallFactory:
    """call_factory passes nsamples the way the factory's signature accepts it."""

    def test_keyword_only_parameter(self):
        def factory(*, nsamples):
            return nsamples

        assert call_factory("s", factory, 3) == 3

    def test_var_keyword(self):
        def factory(**kwargs):
            return kwargs

        assert call_factory("s", factory, 3) == {"nsamples": 3}

    def test_positional_parameter_with_another_name(self):
        def factory(n):
            return n

        assert call_factory("s", factory, 3) == 3

    def test_positional_only_parameter(self):
        def factory(nsamples, /):
            return nsamples

        assert call_factory("s", factory, 3) == 3

    def test_var_positional(self):
        def factory(*args):
            return args

        assert call_factory("s", factory, 3) == (3,)

    def test_zero_argument_factory(self):
        def factory():
            return "called"

        assert call_factory("s", factory, 3) == "called"

    def test_unsupported_signature_is_not_called(self):
        calls = []

        def factory(a, b):
            calls.append((a, b))

        with pytest.raises(ValueError, match="Strategy factory 's'.*nsamples"):
            call_factory("s", factory, 3)
        assert calls == []

    def test_type_error_in_body_is_reported_once(self):
        calls = []

        def factory(nsamples):
            calls.append(nsamples)
            return None + 1

        with pytest.raises(ValueError) as exc_info:
            call_factory("buggy", factory, 10)

        assert calls == [10]
        message = str(exc_info.value)
        assert "'buggy'" in message
        assert "TypeError" in message
        assert "unsupported operand" in message
        assert "should accept" not in message
        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_transient_error_is_not_retried(self):
        calls = []

        def factory(nsamples):
            calls.append(nsamples)
            if len(calls) == 1:
                raise TypeError("transient")
            return "second call"

        with pytest.raises(ValueError, match="transient"):
            call_factory("s", factory, 10)
        assert calls == [10]


class TestResolverFactoryCalling:
    """resolve_and_parametrize goes through call_factory."""

    def test_type_error_in_body_is_reported_once(self):
        calls = []

        def factory(nsamples):
            calls.append(nsamples)
            return (0, 1) + None

        with pytest.raises(ValueError) as exc_info:
            _resolve(factory, ["x"])

        assert calls == [10]
        assert "TypeError" in str(exc_info.value)
        assert "should accept" not in str(exc_info.value)

    def test_zero_argument_factory(self):
        def factory():
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=3)

        _, samples, _ = _resolve(factory, ["x"])
        assert len(samples) == 3

    def test_factory_failing_under_auto_reports_real_error(self):
        calls = []

        def factory(nsamples):
            calls.append(nsamples)
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                directed_vectors={f"row_{i}": (i,) for i in range(nsamples)},
                nsamples=0,
            )

        with pytest.raises(ValueError) as exc_info:
            _resolve(factory, ["x"], nsamples="auto")

        assert calls == ["auto"]
        message = str(exc_info.value)
        assert "nsamples='auto'" in message
        assert "TypeError" in message
        assert "should accept" not in message


class TestExportStrategiesFactoryCalling:
    """export_strategies calls factories the same way the resolver does."""

    def test_keyword_only_var_keyword_and_zero_arg_factories(self, restore_registry):
        received = []

        @Strategy.register("fix_export_kwonly")
        def kwonly(*, nsamples):
            received.append(nsamples)
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        @Strategy.register("fix_export_varkw")
        def varkw(**kwargs):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        @Strategy.register("fix_export_noargs")
        def noargs():
            return Parameter(TestArg("x", value=1), nsamples=1)

        data = json.loads(Strategy.export_strategies())

        assert received == [1]
        assert data["fix_export_kwonly"]["arguments"][0]["name"] == "x"
        assert data["fix_export_varkw"]["arguments"][0]["name"] == "x"
        assert data["fix_export_noargs"]["arguments"][0]["name"] == "x"

    def test_factory_error_is_still_recorded(self, restore_registry):
        @Strategy.register("fix_export_broken")
        def broken(nsamples):
            raise RuntimeError("boom")

        data = json.loads(Strategy.export_strategies())

        assert "RuntimeError: boom" in data["fix_export_broken"]["error"]


# ---------------------------------------------------------------------------
# Single-argument strategies: IDs show the whole value passed to the test
# ---------------------------------------------------------------------------


class TestSingleArgumentIds:
    """IDs are built from the values passed to parametrize, unwrapped exactly once."""

    def test_tuple_valued_directed_vectors(self):
        param = Parameter(
            TestArg("pt", rng_type=RNGChoice([(1, 2)])),
            directed_vectors={"a": ((1, 2),), "b": ((1, 3),)},
        )
        _, samples, ids = _resolve(lambda nsamples: param, ["pt"], vector_mode="directed_only")
        assert samples == [(1, 2), (1, 3)]
        assert ids == ["pt=(1, 2)", "pt=(1, 3)"]

    def test_tuple_valued_random_samples(self):
        param = Parameter(TestArg("pt", rng_type=RNGChoice([(7, 8)])))
        _, samples, ids = _resolve(
            lambda nsamples: param, ["pt"], nsamples=2, vector_mode="random_only"
        )
        assert samples == [(7, 8), (7, 8)]
        # Repeated rows are suffixed as pytest would, so strict IDs accept them
        assert ids == ["pt=(7, 8)0", "pt=(7, 8)1"]


# ---------------------------------------------------------------------------
# pytest.param vectors: marks and ids are kept, IDs are built from the values
# ---------------------------------------------------------------------------


class TestPytestParamVectors:
    """A pytest.param vector reaches parametrize like a pytest.param row would."""

    def test_pytest_param_single_arg_is_not_unwrapped(self):
        xfail = pytest.mark.xfail(strict=True)
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={
                "one": pytest.param(1, marks=xfail),
                "five": pytest.param(5, id="five"),
                "two": (2,),
            },
            nsamples=0,
        )
        _, samples, ids = _resolve(lambda nsamples: param, ["x"])
        assert samples[0] == pytest.param(1, marks=xfail)
        assert samples[1] == pytest.param(5, id="five")
        assert samples[2] == 2
        assert ids == ["x=1", "x=5", "x=2"]

    def test_pytest_param_multi_arg_ids_use_values(self):
        xfail = pytest.mark.xfail(strict=True)
        param = Parameter(
            TestArg("a", rng_type=RNGInteger(0, 9)),
            TestArg("b", rng_type=RNGInteger(0, 9)),
            directed_vectors={"unequal": pytest.param(1, 2, marks=xfail), "equal": (3, 3)},
            nsamples=0,
        )
        _, samples, ids = _resolve(lambda nsamples: param, ["a", "b"])
        assert samples == [pytest.param(1, 2, marks=xfail), (3, 3)]
        assert ids == ["a=1,b=2", "a=3,b=3"]

    def test_pytest_param_id_repeating_another_row_is_suffixed(self):
        xfail = pytest.mark.xfail(strict=True)
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"one": (1,), "two": pytest.param(2, id="x=1", marks=xfail)},
            nsamples=0,
        )
        _, samples, ids = _resolve(lambda nsamples: param, ["x"])
        # The row keeps its marks; its ids entry, which pytest ignores, is unchanged
        assert samples == [1, pytest.param(2, id="x=1_1", marks=xfail)]
        assert ids == ["x=1_0", "x=2"]


# ---------------------------------------------------------------------------
# Duplicate strategy names: warn when a different function takes over a name
# ---------------------------------------------------------------------------

STRATEGY_SOURCE = textwrap.dedent("""
    from pytest_strategy import Parameter, Strategy, TestArg

    @Strategy.register("fix_dup_source")
    def factory(nsamples):
        return Parameter(TestArg("x", value=1), nsamples=1)
    """)


class TestDuplicateRegistration:
    """Strategy.register warns on a real name clash and stays silent on a re-run."""

    def test_different_function_warns_and_last_wins(self, restore_registry):
        @Strategy.register("fix_dup")
        def first(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        with pytest.warns(PytestStrategiesWarning, match="'fix_dup'") as record:

            @Strategy.register("fix_dup")
            def second(nsamples):
                return Parameter(TestArg("y", value=2), nsamples=1)

        assert Strategy._registry["fix_dup"] is second
        assert len(record) == 1
        message = str(record[0].message)
        assert "first" in message
        assert "second" in message
        # stacklevel points at the registering code, not at the plugin
        assert record[0].filename == __file__

    def test_warning_is_a_user_warning(self):
        assert issubclass(PytestStrategiesWarning, UserWarning)

    def test_same_function_again_is_silent(self, restore_registry):
        def factory(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        Strategy.register("fix_dup_same")(factory)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Strategy.register("fix_dup_same")(factory)

    def test_reexecuted_source_file_is_silent(self, restore_registry):
        # The plugin re-executes a strategies file under a new module name per session
        code = compile(STRATEGY_SOURCE, "/virtual/tests/strategies.py", "exec")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exec(code, {"__name__": "strategies_0"})
            exec(code, {"__name__": "strategies_1"})

    def test_same_name_from_another_file_in_the_folder_warns(self, restore_registry):
        exec(compile(STRATEGY_SOURCE, "/virtual/users/strategies.py", "exec"), {})
        with pytest.warns(PytestStrategiesWarning, match="users.strategies.py"):
            exec(compile(STRATEGY_SOURCE, "/virtual/users/more_strategies.py", "exec"), {})

    def test_same_name_from_another_folder_is_kept_silently(self, restore_registry):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exec(compile(STRATEGY_SOURCE, "/virtual/users/strategies.py", "exec"), {})
            exec(compile(STRATEGY_SOURCE, "/virtual/billing/strategies.py", "exec"), {})
