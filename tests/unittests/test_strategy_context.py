"""Unit tests for the ``ctx`` that factories get from the pytest_strategies_context hook."""

import functools
import random
from types import SimpleNamespace

import pytest

from pytest_strategy import RNG, Parameter, RNGInteger, StrategyOptions, TestArg
from pytest_strategy._factory import FactoryInputs, call_factory
from pytest_strategy._runtime import StrategyRuntime, runtime


class HookCalls:
    """A stand-in config whose pytest_strategies_context returns (or raises) a value."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.config = SimpleNamespace(hook=SimpleNamespace(pytest_strategies_context=self.hook))

    def hook(self, config):
        assert config is self.config
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def hook():
    """Run the test inside a runtime session whose hook returns a testbench config."""
    calls = HookCalls(result={"channels": [3, 5]})
    runtime.push(calls.config)
    try:
        yield calls
    finally:
        runtime.pop()


def _parameter():
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 1)))


def _call(factory, nsamples=3, name="s"):
    """Call ``factory`` as the resolver does: ctx comes from the session's hook."""
    inputs = FactoryInputs(
        options=StrategyOptions(strategy=name, nsamples=nsamples),
        rng=RNG.generator(),
        ctx=runtime.strategy_context,
    )
    return call_factory(name, factory, inputs)


class TestFactoriesWithCtx:
    def test_ctx_is_passed_by_keyword(self, hook):
        seen = {}

        def factory(nsamples, ctx):
            seen.update(nsamples=nsamples, ctx=ctx)

        _call(factory)

        assert seen == {"nsamples": 3, "ctx": {"channels": [3, 5]}}

    def test_keyword_only_ctx(self, hook):
        def factory(nsamples, *, ctx):
            return ctx

        assert _call(factory) == {"channels": [3, 5]}

    def test_ctx_without_nsamples(self, hook):
        def factory(ctx):
            return ctx

        assert _call(factory) == {"channels": [3, 5]}

    def test_positional_only_nsamples(self, hook):
        def factory(nsamples, /, ctx):
            return nsamples, ctx

        assert _call(factory) == (3, {"channels": [3, 5]})

    def test_ctx_with_a_default(self, hook):
        def factory(nsamples, ctx=None):
            return ctx

        assert _call(factory) == {"channels": [3, 5]}

    def test_wrapper_without_signature_passes_ctx_to_the_wrapped_function(self, hook):
        def factory(nsamples, ctx):
            return nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            return factory(*args, **kwargs)

        assert _call(wrapper) == (3, {"channels": [3, 5]})

    def test_hook_is_called_once_per_session(self, hook):
        for _ in range(3):
            _call(lambda nsamples, ctx: _parameter())

        assert hook.calls == 1


class TestFactoriesWithoutCtx:
    def test_hook_is_not_called(self, hook):
        def factory(nsamples):
            return nsamples

        assert _call(factory) == 3
        assert hook.calls == 0

    def test_kwargs_factory_does_not_get_ctx(self, hook):
        def factory(**kwargs):
            return kwargs

        assert _call(factory) == {}
        assert hook.calls == 0

    def test_rejected_signature_does_not_call_the_hook(self, hook):
        def factory(a, b, ctx):
            pass

        with pytest.raises(ValueError, match="has a parameter 'a'"):
            _call(factory)
        assert hook.calls == 0


class TestWithoutHookResult:
    """No implementation answered: a ctx default is kept, and ctx is None without one."""

    @pytest.fixture
    def no_answer(self):
        calls = HookCalls(result=None)
        runtime.push(calls.config)
        try:
            yield calls
        finally:
            runtime.pop()

    def test_default_is_kept(self, no_answer):
        def factory(nsamples, ctx="default bench"):
            return ctx

        assert _call(factory) == "default bench"

    def test_ctx_bound_by_partial_is_kept(self, no_answer):
        def factory(nsamples, ctx):
            return ctx

        assert _call(functools.partial(factory, ctx="bench A")) == "bench A"

    def test_args_only_wrapper_around_a_default_ctx(self, no_answer):
        def factory(nsamples, ctx=None):
            return nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args):
            return factory(*args)

        assert _call(wrapper) == (3, None)

    def test_without_a_default_ctx_is_none(self, no_answer):
        assert _call(lambda nsamples, ctx: ctx) is None

    def test_ctx_is_none_without_a_session(self):
        assert StrategyRuntime().strategy_context() is None

    def test_ctx_is_none_when_no_implementation_answers(self):
        calls = HookCalls(result=None)
        runtime.push(calls.config)
        try:
            assert _call(lambda nsamples, ctx: ctx) is None
            assert _call(lambda nsamples, ctx: ctx) is None
        finally:
            runtime.pop()
        assert calls.calls == 1

    def test_each_session_calls_its_own_hook(self):
        outer = HookCalls(result="outer")
        inner = HookCalls(result="inner")
        runtime.push(outer.config)
        try:
            assert _call(lambda ctx: ctx, 1) == "outer"
            runtime.push(inner.config)
            try:
                assert _call(lambda ctx: ctx, 1) == "inner"
            finally:
                runtime.pop()
            assert _call(lambda ctx: ctx, 1) == "outer"
        finally:
            runtime.pop()
        assert (outer.calls, inner.calls) == (1, 1)


class TestHookRandomStream:
    def test_draws_in_the_hook_leave_the_callers_random_state_alone(self):
        def draw_in_hook(config):
            return RNG.generator().random()

        config = SimpleNamespace(hook=SimpleNamespace(pytest_strategies_context=draw_in_hook))
        runtime.push(config)
        try:
            RNG.generator().seed("caller stream")
            expected = random.Random("caller stream").random()
            ctx = _call(lambda ctx: ctx)
            assert RNG.generator().random() == expected
        finally:
            runtime.pop()

        # The hook's own draws come from a stream derived from the seed
        assert ctx == random.Random(f"{RNG.get_seed()}:pytest_strategies_context").random()


class TestHookErrors:
    def test_error_names_the_strategy_and_is_raised_for_every_factory(self):
        calls = HookCalls(error=FileNotFoundError("tb.yaml"))
        runtime.push(calls.config)
        try:
            for name in ("first", "second"):
                with pytest.raises(ValueError) as excinfo:
                    _call(lambda nsamples, ctx: ctx, name=name)
                assert str(excinfo.value) == (
                    f"Strategy factory '{name}' has a 'ctx' parameter, but the "
                    "pytest_strategies_context hook raised FileNotFoundError: tb.yaml"
                )
                assert isinstance(excinfo.value.__cause__, FileNotFoundError)
        finally:
            runtime.pop()
        assert calls.calls == 1

    def test_skip_is_propagated_as_is(self):
        calls = HookCalls(error=pytest.skip.Exception("no testbench", allow_module_level=True))
        runtime.push(calls.config)
        try:
            for _ in range(2):
                with pytest.raises(pytest.skip.Exception, match="no testbench"):
                    _call(lambda ctx: ctx)
        finally:
            runtime.pop()
        assert calls.calls == 1

    def test_repeated_errors_keep_the_original_traceback(self):
        calls = HookCalls(error=RuntimeError("boom"))
        runtime.push(calls.config)
        try:
            depths = []
            for _ in range(3):
                with pytest.raises(ValueError) as excinfo:
                    _call(lambda ctx: ctx)
                tb, depth = excinfo.value.__cause__.__traceback__, 0
                while tb is not None:
                    tb, depth = tb.tb_next, depth + 1
                depths.append(depth)
        finally:
            runtime.pop()
        assert depths[0] == depths[1] == depths[2]
