"""
Unit tests for the factory contract: a factory declares any of nsamples, ctx, rng
and options and receives exactly those, by name (_factory.py).

The last classes run small pytester projects: factories of every callable kind,
the messages a test's collection fails with, and async factories under
filterwarnings = error.
"""

import functools
import inspect
import itertools
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pytest_strategy import RNG, Parameter, RNGInteger, StrategyOptions, TestArg
from pytest_strategy._factory import INPUTS, FactoryInputs, analyse, call_factory
from pytest_strategy._resolver import build_parametrization

pytest_plugins = ["pytester"]

HERE = Path(__file__).parent
BENCH = SimpleNamespace(name="bench")


class Hook:
    """A ctx provider that counts its calls and returns ``result``."""

    def __init__(self, result=BENCH):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


def _inputs(hook=None, nsamples=3):
    """Return the inputs of strategy 's' with ``nsamples``, whose ctx comes from ``hook``."""
    return FactoryInputs(
        options=StrategyOptions(strategy="s", nsamples=nsamples, nsamples_source="--nsamples"),
        rng=RNG.generator(),
        ctx=hook if hook is not None else Hook(),
    )


def _call(factory, hook=None, nsamples=3):
    """Call ``factory`` for strategy 's', with messages relative to this folder."""
    return call_factory("s", factory, _inputs(hook, nsamples), rootpath=HERE)


def _parameter():
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))


def _factory(params, names):
    """Build ``def factory(<params>)`` returning the values of ``names`` by name."""
    namespace = {}
    body = ", ".join(f"{name!r}: {name}" for name in names)
    exec(f"def factory({params}):\n    return {{{body}}}\n", namespace)
    return namespace["factory"]


# ---------------------------------------------------------------------------
# The four inputs
# ---------------------------------------------------------------------------


class TestInputs:
    @pytest.mark.parametrize(
        "template",
        ["{}", "*, {}", "{}, /"],
        ids=["positional-or-keyword", "keyword-only", "positional-only"],
    )
    def test_any_order(self, template):
        inputs = _inputs()
        for order in itertools.permutations(INPUTS):
            factory = _factory(template.format(", ".join(order)), order)
            assert call_factory("s", factory, inputs) == {
                "nsamples": 3,
                "ctx": BENCH,
                "rng": RNG.generator(),
                "options": inputs.options,
            }, order

    def test_mixed_kinds(self):
        def factory(options, /, rng, *, nsamples, ctx):
            return options.strategy, rng, nsamples, ctx

        assert _call(factory) == ("s", RNG.generator(), 3, BENCH)

    def test_only_declared_inputs_are_passed(self):
        def factory(rng):
            return rng

        assert _call(factory) is RNG.generator()

    def test_nsamples_is_the_options_value(self):
        def factory(nsamples, options):
            return nsamples, options.nsamples

        assert _call(factory, nsamples="auto") == ("auto", "auto")

    def test_zero_argument_factory(self):
        hook = Hook()

        def factory():
            return "called"

        assert _call(factory, hook) == "called"
        assert hook.calls == 0

    def test_the_hook_is_called_only_for_ctx(self):
        hook = Hook()
        _call(lambda nsamples, rng, options: None, hook)
        assert hook.calls == 0
        _call(lambda ctx: None, hook)
        assert hook.calls == 1

    def test_var_args_receive_nothing(self):
        def factory(*args):
            return args

        assert _call(factory) == ()

    def test_var_kwargs_receive_nothing(self):
        def factory(**kwargs):
            return kwargs

        assert _call(factory) == {}

    def test_var_args_next_to_inputs_receive_nothing(self):
        def factory(nsamples, *args, ctx, **kwargs):
            return nsamples, args, ctx, kwargs

        assert _call(factory) == (3, (), BENCH, {})

    def test_other_parameters_with_defaults_keep_them(self):
        def factory(n=5, *, width=16, nsamples):
            return n, width, nsamples

        assert _call(factory) == (5, 16, 3)

    def test_defaulted_positional_only_parameters_before_an_input(self):
        def factory(width=16, nsamples=0, /, depth=4):
            return width, nsamples, depth

        assert _call(factory) == (16, 3, 4)

    def test_a_positional_only_ctx_keeps_its_default_when_the_hook_gives_none(self):
        def factory(ctx="default bench", nsamples=0, /):
            return ctx, nsamples

        assert _call(factory, Hook(None)) == ("default bench", 3)
        assert _call(factory, Hook("A")) == ("A", 3)

    def test_ctx_is_none_without_a_default(self):
        assert _call(lambda ctx: ctx, Hook(None)) is None

    def test_factory_is_called_once(self):
        calls = []

        def factory(nsamples):
            calls.append(nsamples)
            raise TypeError("inside the factory")

        with pytest.raises(ValueError, match="TypeError: inside the factory") as excinfo:
            _call(factory)
        assert calls == [3]
        assert "functools.wraps" not in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, TypeError)

    def test_a_raising_factory_keeps_the_3_0_message(self):
        def factory(nsamples):
            raise RuntimeError("boom")

        with pytest.raises(ValueError) as excinfo:
            _call(factory, nsamples="auto")
        assert str(excinfo.value) == (
            "Error calling strategy factory 's' (nsamples='auto'): RuntimeError: boom"
        )


class TestThroughTheResolver:
    """build_parametrization passes the plugin's generator and the strategy's options."""

    @staticmethod
    def _build(factory, config=None):
        def test_fn(x):
            pass

        return build_parametrization(
            "s", factory, test_fn, config=config, pytest_fixtures=set(), validate=True
        )

    def test_rng_is_the_plugin_generator(self):
        seen = []

        def factory(rng):
            seen.append(rng is RNG.generator())
            return _parameter()

        self._build(factory)
        assert seen == [True]

    def test_options_are_the_strategys(self):
        seen = []

        def factory(options, nsamples):
            seen.append((options, nsamples))
            return _parameter()

        self._build(factory)
        [(options, nsamples)] = seen
        assert isinstance(options, StrategyOptions)
        assert (options.strategy, options.nsamples, nsamples) == ("s", 10, 10)

    def test_rows_do_not_depend_on_which_inputs_are_declared(self):
        """rng is the generator the rows draw from, positioned as before."""

        def plain(nsamples):
            return _parameter()

        def with_all(options, rng, nsamples):
            return _parameter()

        RNG.seed(7)
        first = self._build(plain).values
        RNG.seed(7)
        assert self._build(with_all).values == first


# ---------------------------------------------------------------------------
# Names the plugin does not provide
# ---------------------------------------------------------------------------


class TestUnknownNames:
    @pytest.mark.parametrize("params", ["n", "n, /", "n, ctx"])
    def test_nsamples_by_position_is_gone(self, params):
        calls = []
        namespace = {"calls": calls}
        exec(f"def burst({params}):\n    calls.append(1)\n", namespace)
        hook = Hook()

        with pytest.raises(ValueError) as excinfo:
            _call(namespace["burst"], hook)

        message = str(excinfo.value)
        assert "has a parameter 'n', which the plugin does not provide" in message
        assert "Did you mean 'nsamples'?" in message
        assert (calls, hook.calls) == ([], 0)

    def test_message(self):
        def burst(n):
            pass

        line = burst.__code__.co_firstlineno
        with pytest.raises(ValueError) as excinfo:
            call_factory("burst", burst, _inputs(), rootpath=HERE)

        assert str(excinfo.value) == (
            f"Strategy factory 'burst' (test_factory_inputs.py:{line}:"
            "TestUnknownNames.test_message.<locals>.burst) has a parameter 'n', which the "
            "plugin does not provide. Factories receive arguments by name: nsamples, ctx, "
            "rng, options. Did you mean 'nsamples'? Since 4.0 the plugin no longer passes "
            "nsamples by position to a parameter with another name: rename it, or give it "
            "a default if the plugin should leave it alone."
        )

    def test_the_path_is_absolute_outside_the_rootdir(self, tmp_path):
        def burst(n):
            pass

        with pytest.raises(ValueError, match="has a parameter 'n'") as excinfo:
            call_factory("burst", burst, _inputs(), rootpath=tmp_path)
        assert f"({os.path.realpath(__file__)}:" in str(excinfo.value)

    def test_close_name(self):
        with pytest.raises(ValueError) as excinfo:
            _call(lambda nsamples, option: None)
        message = str(excinfo.value)
        assert "has a parameter 'option'" in message
        assert "Did you mean 'options'? Rename it, or give it a default" in message
        assert "Since 4.0" not in message

    def test_a_second_unknown_name_after_nsamples(self):
        with pytest.raises(ValueError) as excinfo:
            _call(lambda nsamples, a: None)
        message = str(excinfo.value)
        assert "has a parameter 'a'" in message
        assert "Did you mean" not in message

    @pytest.mark.parametrize("name", ["self", "cls"])
    def test_self_and_cls(self, name):
        with pytest.raises(ValueError, match=f"'{name}' is not passed either: register a bound"):
            _call(_factory(f"{name}, nsamples", ["nsamples"]))

    def test_fixture_name(self):
        with pytest.raises(ValueError) as excinfo:
            _call(lambda nsamples, tmp_path: None)
        assert "factories run at collection, before fixtures exist; use ctx" in str(excinfo.value)

    def test_the_hook_is_not_called_for_a_rejected_ctx_factory(self):
        hook = Hook()
        with pytest.raises(ValueError, match="has a parameter 'a'"):
            _call(lambda a, b, ctx: None, hook)
        assert hook.calls == 0


class TestReservedNames:
    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("base", "a later 4.x release passes it the strategy the factory extends"),
            ("config", "factories do not receive the pytest config, because"),
            ("request", "factories run when pytest collects the tests, before any test's"),
        ],
    )
    @pytest.mark.parametrize("params", ["nsamples, {}", "nsamples, {}=None", "*, ctx, {}=1"])
    def test_reserved_even_with_a_default(self, name, reason, params):
        calls = []
        namespace = {"calls": calls}
        exec(f"def factory({params.format(name)}):\n    calls.append(1)\n", namespace)
        hook = Hook()

        with pytest.raises(ValueError) as excinfo:
            _call(namespace["factory"], hook)

        message = str(excinfo.value)
        assert f"has a parameter '{name}', a name the plugin reserves: {reason}" in message
        assert message.endswith("Rename it.")
        assert (calls, hook.calls) == ([], 0)


# ---------------------------------------------------------------------------
# partial, decorators, mock.patch and cached factories
# ---------------------------------------------------------------------------


class TestPartials:
    def test_bound_ctx_is_kept_when_the_hook_gives_none(self):
        def factory(nsamples, ctx):
            return ctx

        assert _call(functools.partial(factory, ctx="A"), Hook(None)) == "A"
        assert _call(functools.partial(factory, ctx="A"), Hook("B")) == "B"

    def test_other_bound_keyword_is_kept(self):
        def factory(nsamples, width):
            return nsamples, width

        assert _call(functools.partial(factory, width=16)) == (3, 16)

    def test_positionally_bound_nsamples_is_not_passed(self):
        def factory(nsamples, ctx=None):
            return nsamples, ctx

        assert _call(functools.partial(factory, 7)) == (7, BENCH)

    @pytest.mark.parametrize("name", ["nsamples", "rng", "options"])
    def test_bound_input_is_overridden(self, name):
        def factory(nsamples, rng, options):
            return {"nsamples": nsamples, "rng": rng, "options": options}

        inputs = _inputs()
        received = call_factory("s", functools.partial(factory, **{name: "bound"}), inputs)
        assert received == {"nsamples": 3, "rng": RNG.generator(), "options": inputs.options}

    def test_partial_of_a_wraps_wrapper(self):
        # The partial has no __wrapped__ of its own: the wrapped signature is its func's
        def factory(nsamples, width=8):
            return nsamples, width

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            return factory(*args, **kwargs)

        assert _call(functools.partial(wrapper, width=16)) == (3, 16)

    def test_partial_of_a_wraps_wrapper_with_only_args(self):
        def factory(width, nsamples, ctx=None):
            return width, nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args):
            return factory(*args)

        assert _call(functools.partial(wrapper, 16), Hook("bench")) == (16, 3, "bench")

    def test_partial_of_a_cached_factory(self):
        @functools.cache
        def factory(nsamples, width):
            return nsamples, width

        assert _call(functools.partial(factory, width=16)) == (3, 16)


class TestWrappers:
    def test_wraps_wrapper_with_args_and_kwargs_gets_keywords(self):
        received = []

        def factory(*, nsamples, ctx):
            return nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            received.append((args, kwargs))
            return factory(*args, **kwargs)

        assert _call(wrapper) == (3, BENCH)
        assert received == [((), {"nsamples": 3, "ctx": BENCH})]

    def test_wraps_wrapper_with_only_args_gets_positions(self):
        received = []

        def factory(nsamples, ctx=None):
            return nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args):
            received.append(args)
            return factory(*args)

        assert _call(wrapper, Hook("bench")) == (3, "bench")
        assert _call(wrapper, Hook(None)) == (3, None)
        assert received == [(3, "bench"), (3, None)]

    def test_wraps_wrapper_with_only_args_fills_gaps_with_defaults(self):
        def factory(width=16, nsamples=0, depth=4):
            return width, nsamples, depth

        @functools.wraps(factory)
        def wrapper(*args):
            assert args == (16, 3)
            return factory(*args)

        assert _call(wrapper) == (16, 3, 4)

    def test_wraps_wrapper_with_only_args_cannot_pass_a_keyword_only_input(self):
        def factory(nsamples, *, ctx):
            return nsamples, ctx

        @functools.wraps(factory)
        def wrapper(*args):
            return factory(*args)

        hook = Hook()
        with pytest.raises(ValueError, match="give the wrapper \\*\\*kwargs too") as excinfo:
            _call(wrapper, hook)
        assert "keyword-only parameter 'ctx'" in str(excinfo.value)
        assert hook.calls == 0

    def test_wrapper_signature_decides_when_it_names_parameters(self):
        def factory(nsamples, extra):
            return nsamples, extra

        @functools.wraps(factory)
        def wrapper(nsamples):
            return factory(nsamples, "added")

        assert _call(wrapper) == (3, "added")

    def test_opaque_wrapper_is_called_once_with_no_arguments(self):
        calls = []

        def factory(nsamples):
            return nsamples

        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return factory(*args, **kwargs)

        with pytest.raises(ValueError) as excinfo:
            _call(wrapper)

        assert calls == [((), {})]
        message = str(excinfo.value)
        assert message.startswith("Error calling strategy factory 's' (nsamples=3): TypeError:")
        assert "missing 1 required positional argument: 'nsamples'" in message
        assert "decorate the wrapper with @functools.wraps(factory)" in message
        assert isinstance(excinfo.value.__cause__, TypeError)

    def test_opaque_wrapper_around_a_zero_argument_function(self):
        def factory():
            return "made"

        def wrapper(*args, **kwargs):
            return factory(*args, **kwargs)

        assert _call(wrapper) == "made"

    def test_opaque_wrapper_without_a_type_error_gets_no_hint(self):
        def wrapper(*args, **kwargs):
            raise RuntimeError("boom")

        with pytest.raises(ValueError) as excinfo:
            _call(wrapper)
        assert "RuntimeError: boom" in str(excinfo.value)
        assert "functools.wraps" not in str(excinfo.value)

    def test_callable_without_a_signature_is_called_with_no_arguments(self):
        class NoSignature:
            # inspect.signature() raises TypeError for this
            __signature__ = "not a signature"

            def __call__(self, *args, **kwargs):
                return args, kwargs

        assert analyse(NoSignature()).opaque is True
        assert _call(NoSignature()) == ((), {})


class TestCachedFactories:
    def test_cache(self):
        calls = []

        @functools.cache
        def factory(nsamples, ctx):
            calls.append(nsamples)
            return nsamples, ctx

        assert _call(factory, Hook("bench")) == (3, "bench")
        assert _call(factory, Hook("bench")) == (3, "bench")
        assert calls == [3]

    def test_zero_argument_cache(self):
        @functools.cache
        def factory():
            return "made"

        assert _call(factory) == "made"


class TestMockPatch:
    def test_mocks_first(self):
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(getcwd, nsamples, ctx=None):
            return nsamples, ctx, os.getcwd(), getcwd.return_value

        assert _call(factory) == (3, BENCH, "/fake", "/fake")

    def test_two_mocks_first(self):
        @mock.patch("os.getcwd", return_value="/fake")
        @mock.patch("os.getpid", return_value=7)
        def factory(getpid, getcwd, *, nsamples):
            return nsamples, os.getpid(), os.getcwd()

        assert _call(factory) == (3, 7, "/fake")

    def test_patch_with_new_passes_no_mock(self):
        @mock.patch("os.getcwd", new=lambda: "/new")
        def factory(nsamples):
            return nsamples, os.getcwd()

        assert _call(factory) == (3, "/new")

    @pytest.mark.parametrize("params", ["nsamples, getcwd", "nsamples, getcwd=None"])
    def test_mocks_last(self, params):
        calls = []
        namespace = {"calls": calls}
        exec(f"def factory({params}):\n    calls.append(1)\n", namespace)
        factory = mock.patch("os.getcwd", return_value="/fake")(namespace["factory"])

        with pytest.raises(ValueError) as excinfo:
            _call(factory)

        message = str(excinfo.value)
        assert "is decorated with mock.patch, which passes its mocks to its first" in message
        assert "so 'nsamples' would receive a mock. Put the mock parameters first" in message
        assert calls == []

    def test_patch_multiple_passes_mocks_by_keyword(self):
        @mock.patch.multiple("os", getcwd=mock.DEFAULT)
        def factory(nsamples, getcwd=None):
            return nsamples, isinstance(getcwd, mock.MagicMock)

        assert _call(factory) == (3, True)

    def test_patch_multiple_with_a_patch(self):
        @mock.patch.multiple("os", getcwd=mock.DEFAULT)
        @mock.patch("os.getpid", return_value=7)
        def factory(getpid, nsamples, getcwd=None):
            return nsamples, os.getpid(), isinstance(getcwd, mock.MagicMock)

        assert _call(factory) == (3, 7, True)

    def test_partial_of_a_patched_factory(self):
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(getcwd, nsamples, width=8):
            return nsamples, width, os.getcwd()

        assert _call(functools.partial(factory, width=16)) == (3, 16, "/fake")

    def test_patched_call_and_init(self):
        class Factory:
            @mock.patch("os.getcwd", return_value="/fake")
            def __call__(self, getcwd, nsamples):
                return nsamples, os.getcwd()

        class Made:
            @mock.patch("os.getcwd", return_value="/fake")
            def __init__(self, getcwd, nsamples):
                self.made = nsamples, os.getcwd()

        assert _call(Factory()) == (3, "/fake")
        assert _call(Made).made == (3, "/fake")

    def test_unknown_name_in_a_patched_factory(self):
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(getcwd, other, nsamples):
            pass

        with pytest.raises(ValueError) as excinfo:
            _call(factory)
        message = str(excinfo.value)
        assert "has a parameter 'other'" in message
        assert message.endswith(
            "Rename it, or give it a default if the plugin should leave it alone. The factory "
            "is decorated with mock.patch, which passes its mocks to its first parameters "
            "('getcwd')."
        )

    def test_nsamples_by_position_in_a_patched_factory(self):
        # The mocks come first already: the hint is the one for 3.0's nsamples
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(getcwd, n):
            pass

        with pytest.raises(ValueError) as excinfo:
            _call(factory)
        message = str(excinfo.value)
        assert "has a parameter 'n'" in message
        assert "Did you mean 'nsamples'? Since 4.0" in message
        assert "passes its mocks to its first parameters ('getcwd')." in message
        assert "put the mock parameters first" not in message

    def test_unknown_name_after_an_input_in_a_patched_factory(self):
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(width, nsamples, getcwd):
            pass

        with pytest.raises(ValueError) as excinfo:
            _call(factory)
        message = str(excinfo.value)
        assert "has a parameter 'getcwd'" in message
        assert message.endswith(
            "passes its mocks to its first parameters ('width'): if 'getcwd' is a mock "
            "parameter, put the mock parameters first, before the ones the plugin passes."
        )

    def test_positional_only_input_after_a_mock(self):
        @mock.patch("os.getcwd", return_value="/fake")
        def factory(getcwd, nsamples, /):
            pass

        with pytest.raises(ValueError, match="'nsamples' cannot be passed by position"):
            _call(factory)


class TestCallableKinds:
    def test_callable_object(self):
        class Factory:
            def __call__(self, rng, nsamples):
                return rng, nsamples

        assert _call(Factory()) == (RNG.generator(), 3)

    def test_methods(self):
        class Strategies:
            def bound(self, nsamples):
                return "bound", nsamples

            @classmethod
            def from_class(cls, nsamples):
                return cls.__name__, nsamples

            @staticmethod
            def static(nsamples):
                return "static", nsamples

        assert _call(Strategies().bound) == ("bound", 3)
        assert _call(Strategies.from_class) == ("Strategies", 3)
        assert _call(Strategies.static) == ("static", 3)
        assert _call(Strategies.__dict__["static"]) == ("static", 3)

    def test_class(self):
        class Bus(Parameter):
            def __init__(self, nsamples, options):
                super().__init__(TestArg("x", value=(nsamples, options.strategy)))

        assert isinstance(_call(Bus), Bus)

    def test_wraps_decorated_call_and_init(self):
        def passthrough(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        class Factory:
            @passthrough
            def __call__(self, nsamples, ctx):
                return "callable", nsamples, ctx

        class Made:
            @passthrough
            def __init__(self, nsamples):
                self.nsamples = nsamples

        assert _call(Factory()) == ("callable", 3, BENCH)
        assert _call(Made).nsamples == 3


# ---------------------------------------------------------------------------
# async factories
# ---------------------------------------------------------------------------


class TestAsyncFactories:
    def test_async_def_fails_before_it_is_called(self):
        calls = []

        async def factory(nsamples):
            calls.append(nsamples)

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            calls.append("wrapper")
            return factory(*args, **kwargs)

        candidates = (factory, wrapper, functools.partial(factory, 3), functools.partial(wrapper))
        for candidate in candidates:
            with pytest.raises(ValueError, match="async factories are not supported"):
                _call(candidate)
        assert calls == []

    def test_a_returned_coroutine_is_closed(self):
        coroutines = []

        async def make():
            return _parameter()

        def opaque(*args, **kwargs):
            coroutines.append(make())
            return coroutines[-1]

        with pytest.raises(ValueError) as excinfo:
            _call(opaque)

        assert "returned a coroutine: async factories are not supported" in str(excinfo.value)
        assert [inspect.getcoroutinestate(c) for c in coroutines] == ["CORO_CLOSED"]


# ---------------------------------------------------------------------------
# pytester projects
# ---------------------------------------------------------------------------

KINDS_STRATEGIES = """
    import functools

    from pytest_strategy import Parameter, RNG, RNGInteger, TestArg, register
    from pytest_strategy._runtime import runtime


    def _param(tag, nsamples):
        return Parameter(TestArg("x", value=(tag, nsamples)))


    class Callable:
        def __call__(self, nsamples, rng):
            assert rng is RNG.generator()
            return _param("callable", nsamples)


    class Factories:
        def bound(self, options):
            # The strategy's instance, cached for the session
            config = runtime.current.config
            assert options is runtime.strategy_options("kinds_bound", config)
            return _param("bound", options.nsamples)

        @classmethod
        def from_class(cls, nsamples):
            return _param(cls.__name__, nsamples)

        @staticmethod
        def static(nsamples):
            return _param("static", nsamples)

        @register("kinds_static_in_body")
        @staticmethod
        def static_in_body(*, nsamples):
            return _param("static_in_body", nsamples)


    class Bus(Parameter):
        def __init__(self, nsamples, options):
            super().__init__(TestArg("x", value=(options.strategy, nsamples)))


    def passthrough(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper


    @passthrough
    def decorated(nsamples, width=8):
        return _param(("decorated", width), nsamples)


    register("kinds_callable")(Callable())
    register("kinds_bound")(Factories().bound)
    register("kinds_classmethod")(Factories.from_class)
    register("kinds_staticmethod")(Factories.static)
    register("kinds_class")(Bus)
    register("kinds_cached")(functools.cache(lambda nsamples, rng: _param("cached", nsamples)))
    register("kinds_partial")(functools.partial(decorated, width=16))
    """

KINDS_TESTS = """
    from pytest_strategy import strategy

    @strategy("kinds_callable")
    def test_callable(x):
        assert x == ("callable", 2)

    @strategy("kinds_bound")
    def test_bound(x):
        assert x == ("bound", 2)

    @strategy("kinds_classmethod")
    def test_classmethod(x):
        assert x == ("Factories", 2)

    @strategy("kinds_staticmethod")
    def test_staticmethod(x):
        assert x == ("static", 2)

    @strategy("kinds_static_in_body")
    def test_static_in_body(x):
        assert x == ("static_in_body", 2)

    @strategy("kinds_class")
    def test_class(x):
        assert x == ("kinds_class", 2)

    @strategy("kinds_cached")
    def test_cached(x):
        assert x == ("cached", 2)

    @strategy("kinds_partial")
    def test_partial(x):
        assert x == (("decorated", 16), 2)
    """


class TestProjects:
    def test_callable_kinds_collect(self, pytester):
        pytester.makepyfile(kinds_strategies=KINDS_STRATEGIES, test_kinds=KINDS_TESTS)

        result = pytester.runpytest_inprocess("--nsamples=2")

        result.assert_outcomes(passed=8 * 2)

    def test_collection_error_names_the_factory_relative_to_the_rootdir(self, pytester):
        pytester.makepyfile(
            burst_strategies="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("burst")
            def burst(n):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
            """,
            test_burst="""
            from pytest_strategy import strategy

            @strategy("burst")
            def test_burst(x):
                pass
            """,
        )

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "*In test_burst: Strategy factory 'burst' (burst_strategies.py:3:burst) has a "
                "parameter 'n', which the plugin does not provide.*Did you mean 'nsamples'?*"
            ]
        )

    def test_async_factories_fail_without_a_runtime_warning(self, pytester):
        pytester.makeini("""
            [pytest]
            filterwarnings = error
            """)
        pytester.makepyfile(
            async_strategies="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("async_def")
            async def async_def(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

            def _opaque(fn):
                def wrapper(*args, **kwargs):
                    return fn()
                return wrapper

            @register("returns_coroutine")
            @_opaque
            async def returns_coroutine():
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
            """,
            test_async_a="""
            from pytest_strategy import strategy

            @strategy("async_def")
            def test_a(x):
                pass
            """,
            test_async_b="""
            from pytest_strategy import strategy

            @strategy("returns_coroutine")
            def test_b(x):
                pass
            """,
        )

        result = pytester.runpytest_subprocess()

        result.assert_outcomes(errors=2)
        result.stdout.fnmatch_lines(
            [
                "*In test_a: Strategy factory 'async_def' (*) is an async function: async "
                "factories are not supported*",
                "*In test_b: Strategy factory 'returns_coroutine' (*) returned a coroutine: async "
                "factories are not supported*",
            ]
        )
        output = result.stdout.str() + result.stderr.str()
        assert "RuntimeWarning" not in output
        assert "never awaited" not in output
