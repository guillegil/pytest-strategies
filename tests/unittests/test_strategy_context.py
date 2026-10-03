"""
Unit tests for the ``ctx`` that factories get from the pytest_strategies_context
hook: which implementations a folder sees, the order they are asked in, the store
that calls each one at most once per session, and ``get_context()``, which gives a
folder's context to fixtures.

The sessions here have a real plugin manager with the plugin's hook, and no pytest
session: a plugin registered under a name that ends with ``conftest.py`` is that
folder's conftest.py, as pytest registers one.
"""

import functools
import random
from types import SimpleNamespace

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    StrategyOptions,
    TestArg,
    get_context,
    hookspecs,
)
from pytest_strategy._context import (
    NO_ANSWER,
    ContextStore,
    FolderContext,
    call_order,
    visible_from,
)
from pytest_strategy._factory import FactoryInputs, call_factory
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy._streams import StreamKey
from pytest_strategy.plugin import _ctx_scopes_message

# The run seed of the sessions below
SEED = 77


class Plugin:
    """A plugin whose pytest_strategies_context returns ``value``, or calls ``fn``."""

    def __init__(self, value=None, fn=None):
        self.value = value
        self.fn = fn
        self.calls = 0

    def pytest_strategies_context(self, config):
        self.calls += 1
        if self.fn is not None:
            return self.fn(config)
        return self.value


class Session:
    """A stand-in config with a real plugin manager, for one runtime session."""

    def __init__(self, rootpath):
        self.rootpath = rootpath
        self.pluginmanager = pytest.PytestPluginManager()
        self.pluginmanager.add_hookspecs(hookspecs)
        self.config = SimpleNamespace(pluginmanager=self.pluginmanager, rootpath=rootpath)

    def conftest(self, folder, plugin):
        """Register ``plugin`` as the conftest.py of ``folder`` (relative to the rootdir)."""
        (self.rootpath / folder).mkdir(parents=True, exist_ok=True)
        self.pluginmanager.register(plugin, str(self.rootpath / folder / "conftest.py"))
        return plugin

    def plugin(self, plugin, name=None):
        """Register ``plugin`` as a plugin that is not a conftest.py."""
        self.pluginmanager.register(plugin, name)
        return plugin

    def impls(self):
        return self.pluginmanager.hook.pytest_strategies_context.get_hookimpls()

    def names(self, impls):
        """The plugins of ``impls``, by their labels."""
        store = ContextStore(self.config)
        return [store.label(impl) for impl in impls]

    def context(self, folder):
        """The context a test in ``folder`` (relative to the rootdir) gets."""
        return runtime.strategy_context(self.rootpath / folder)


@pytest.fixture
def session(tmp_path):
    """Run the test inside a runtime session with a plugin manager and run seed 77."""
    bench = Session(tmp_path)
    state = runtime.push(bench.config)
    state.run_seed = SEED
    try:
        yield bench
    finally:
        runtime.pop()


@pytest.fixture
def hook(session):
    """A session whose rootdir conftest.py returns a testbench config."""
    return session.conftest(".", Plugin({"channels": [3, 5]}))


def _parameter():
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 1)))


def _call(factory, nsamples=3, name="s", context=None):
    """
    Call ``factory`` as the resolver does, with the context of ``context`` (a
    FolderContext), or of the rootdir.
    """
    context = context if context is not None else runtime.path_context()
    inputs = FactoryInputs(
        options=StrategyOptions(strategy=name, nsamples=nsamples),
        rng=RNG.generator(),
        ctx=context,
        why_no_ctx=context.why_none,
    )
    return call_factory(name, factory, inputs)


def _ctx_draw():
    """The first integer a draw on the stream root(S, "ctx") gives."""
    return random.Random(StreamKey.root(SEED, "ctx").seed_int()).randint(0, 10**9)


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

    def test_the_implementation_is_called_once_per_session(self, hook):
        for _ in range(3):
            _call(lambda nsamples, ctx: _parameter())

        assert hook.calls == 1

    def test_every_factory_gets_the_same_object(self, hook):
        first = _call(lambda ctx: ctx)

        assert _call(lambda ctx: ctx) is first


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

    def test_a_test_s_folder_context_calls_nothing_until_asked(self, session):
        plugin = session.conftest(".", Plugin("bench"))
        node = SimpleNamespace(
            path=session.rootpath / "test_x.py", ihook=session.pluginmanager.hook
        )

        context = runtime.test_context(node)

        assert plugin.calls == 0
        assert context() == "bench"
        assert plugin.calls == 1


class TestWithoutHookResult:
    """No implementation answered: a ctx default is kept, and ctx is None without one."""

    @pytest.fixture
    def no_answer(self, session):
        return session.conftest(".", Plugin(None))

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

    def test_ctx_is_none_without_a_plugin_manager(self, tmp_path):
        runtime.push(SimpleNamespace(rootpath=tmp_path))
        try:
            assert _call(lambda ctx: ctx) is None
        finally:
            runtime.pop()

    def test_ctx_is_none_when_no_implementation_answers(self, no_answer):
        assert _call(lambda nsamples, ctx: ctx) is None
        assert _call(lambda nsamples, ctx: ctx) is None
        assert no_answer.calls == 1

    def test_without_any_implementation_nothing_is_called(self, session):
        assert runtime.path_context().answer() is NO_ANSWER


class TestCallOrder:
    """The order the plugin asks the implementations a folder sees in (D7)."""

    def test_conftests_from_the_folder_up_then_plugins_last_registered_first(self, session):
        session.plugin(Plugin("early"), "early")
        session.conftest("tests/a/deep", Plugin())
        session.conftest(".", Plugin())
        session.plugin(Plugin("late"), "late")
        session.conftest("tests/a", Plugin())

        order = session.names(call_order(session.impls()))

        assert order == [
            "tests/a/deep/conftest.py",
            "tests/a/conftest.py",
            "conftest.py",
            "late",
            "early",
        ]

    def test_tryfirst_and_trylast_groups_put_conftests_first_too(self, session):
        class First(Plugin):
            @pytest.hookimpl(tryfirst=True)
            def pytest_strategies_context(self, config):
                return super().pytest_strategies_context(config)

        class Last(Plugin):
            @pytest.hookimpl(trylast=True)
            def pytest_strategies_context(self, config):
                return super().pytest_strategies_context(config)

        session.conftest(".", Last())
        session.plugin(First(), "first plugin")
        session.conftest("tests", Plugin())
        session.plugin(Last(), "last plugin")
        session.conftest("tests/a", First())
        session.plugin(Plugin(), "plugin")

        order = session.names(call_order(session.impls()))

        assert order == [
            "tests/a/conftest.py",
            "first plugin",
            "tests/conftest.py",
            "plugin",
            "conftest.py",
            "last plugin",
        ]

    def test_wrappers_come_first(self, session):
        class Wrapper(Plugin):
            @pytest.hookimpl(wrapper=True)
            def pytest_strategies_context(self, config):
                return (yield)

        session.conftest(".", Wrapper())
        session.conftest("tests", Plugin())
        session.plugin(Wrapper(), "wrapping plugin")
        session.conftest("tests/a", Wrapper())

        order = session.names(call_order(session.impls()))

        assert order == [
            "tests/a/conftest.py",
            "conftest.py",
            "wrapping plugin",
            "tests/conftest.py",
        ]


class TestVisibleFrom:
    """The path caller: the implementations the folder of a path sees."""

    def test_conftests_of_other_folders_are_left_out(self, session):
        session.conftest(".", Plugin())
        session.conftest("tests/a", Plugin())
        session.conftest("tests/a/deep", Plugin())
        session.conftest("tests/b", Plugin())
        session.plugin(Plugin(), "plugin")

        def seen(path):
            return sorted(session.names(visible_from(session.config, session.rootpath / path)))

        assert seen("tests/a/deep") == [
            "conftest.py",
            "plugin",
            "tests/a/conftest.py",
            "tests/a/deep/conftest.py",
        ]
        assert seen("tests/a") == ["conftest.py", "plugin", "tests/a/conftest.py"]
        assert seen("tests/b") == ["conftest.py", "plugin", "tests/b/conftest.py"]
        # A folder without a conftest.py of its own, and one pytest never collected
        assert seen("tests/c") == ["conftest.py", "plugin"]
        assert seen("tests/ab") == ["conftest.py", "plugin"]

    def test_a_file_counts_as_its_folder(self, session):
        session.conftest("tests/a", Plugin())
        (session.rootpath / "tests/a/test_x.py").write_text("")

        file = visible_from(session.config, session.rootpath / "tests/a/test_x.py")
        folder = visible_from(session.config, session.rootpath / "tests/a")

        assert session.names(file) == session.names(folder) == ["tests/a/conftest.py"]

    def test_pluggy_order_is_kept(self, session):
        session.conftest(".", Plugin())
        session.plugin(Plugin(), "plugin")
        session.conftest("tests", Plugin())

        impls = visible_from(session.config, session.rootpath / "tests")

        assert impls == session.impls()

    def test_without_a_plugin_manager_none(self, tmp_path):
        assert visible_from(SimpleNamespace(rootpath=tmp_path), tmp_path) == []


class TestFolderContexts:
    """What a folder's tests get, per D7's rule."""

    def test_the_nearest_conftest_that_answers_wins(self, session):
        session.conftest(".", Plugin({"name": "root"}))
        session.conftest("tests/a", Plugin({"name": "A"}))
        session.plugin(Plugin({"name": "plugin"}), "plugin")

        assert session.context("tests/a")["name"] == "A"
        assert session.context("tests/a/deep")["name"] == "A"
        assert session.context("tests/b")["name"] == "root"
        assert session.context(".")["name"] == "root"

    def test_a_conftest_returning_none_shares_the_parent_s_object_and_call(self, session):
        root = session.conftest(".", Plugin({"name": "root"}))
        a = session.conftest("tests/a", Plugin({"name": "A"}))
        deep = session.conftest("tests/a/deep", Plugin(None))

        in_a = runtime.path_context(session.rootpath / "tests/a").answer()
        in_deep = runtime.path_context(session.rootpath / "tests/a/deep").answer()

        assert in_deep.value is in_a.value
        assert in_deep.label == in_a.label == "tests/a/conftest.py"
        assert (root.calls, a.calls, deep.calls) == (0, 1, 1)

    def test_two_answering_scopes_give_two_calls(self, session):
        root = session.conftest(".", Plugin({"name": "root"}))
        a = session.conftest("tests/a", Plugin({"name": "A"}))

        for folder in ("tests/a", "tests/b", "tests/a/deep", "tests/c", "."):
            session.context(folder)

        assert (root.calls, a.calls) == (1, 1)
        assert session.context("tests/b") is session.context("tests/c")

    def test_a_plugin_answers_only_where_no_conftest_does(self, session):
        session.conftest("tests/a", Plugin("A"))
        session.conftest("tests/a/deep", Plugin(None))
        session.plugin(Plugin("plugin"), "plugin")

        assert session.context("tests/a/deep") == "A"
        assert session.context("tests/b") == "plugin"

    def test_a_tryfirst_rootdir_implementation_wins_everywhere(self, session):
        class First(Plugin):
            @pytest.hookimpl(tryfirst=True)
            def pytest_strategies_context(self, config):
                return super().pytest_strategies_context(config)

        session.conftest(".", First("root"))
        session.conftest("tests/a", Plugin("A"))

        assert session.context("tests/a") == "root"
        assert session.context("tests/b") == "root"

    def test_labels(self, session):
        session.conftest("tests/a", Plugin("A"))
        session.plugin(Plugin("named"), "named plugin")

        assert runtime.path_context(session.rootpath / "tests/a").answer().label == (
            "tests/a/conftest.py"
        )
        assert runtime.path_context(session.rootpath / "tests/b").answer().label == ("named plugin")

    def test_a_plugin_without_a_name_is_labeled_by_its_type(self, session):
        session.plugin(Plugin("unnamed"))

        assert runtime.path_context().answer().label == "Plugin"

    def test_nothing_answering_is_labeled_none(self, session):
        session.conftest(".", Plugin(None))

        assert runtime.path_context().answer().label == "none"


class TestWrappers:
    def test_a_wrapper_extends_the_answer(self, session):
        class Extend(Plugin):
            @pytest.hookimpl(wrapper=True)
            def pytest_strategies_context(self, config):
                self.calls += 1
                ctx = yield
                return {**ctx, "extended": True}

        root = session.conftest(".", Plugin({"name": "root"}))
        a = session.conftest("tests/a", Plugin({"name": "A"}))
        wrapper = session.conftest("tests/a/w", Extend())

        assert session.context("tests/a/w") == {"name": "A", "extended": True}
        assert session.context("tests/a") == {"name": "A"}
        answer = runtime.path_context(session.rootpath / "tests/a/w").answer()
        assert answer.label == "tests/a/w/conftest.py"
        # Each implementation once, the wrapper once for its list of implementations
        session.context("tests/a/w/deeper")
        assert (root.calls, a.calls, wrapper.calls) == (0, 1, 1)

    def test_an_old_style_hookwrapper_can_replace_the_answer(self, session):
        class Replace(Plugin):
            @pytest.hookimpl(hookwrapper=True)
            def pytest_strategies_context(self, config):
                outcome = yield
                outcome.force_result(("replaced", outcome.get_result()))

        session.conftest(".", Plugin("root"))
        session.plugin(Replace(), "replacing plugin")

        assert session.context("tests") == ("replaced", "root")

    def test_an_implementation_s_error_goes_through_the_wrapper(self, session):
        class Catch(Plugin):
            @pytest.hookimpl(wrapper=True)
            def pytest_strategies_context(self, config):
                try:
                    return (yield)
                except RuntimeError as e:
                    return f"caught {e}"

        def broken(config):
            raise RuntimeError("x")

        session.conftest(".", Plugin(fn=broken))
        session.conftest("tests", Catch())

        assert session.context("tests") == "caught x"
        with pytest.raises(RuntimeError, match="x"):
            session.context(".")

    def test_a_wrapper_that_raises_fails_its_folders(self, session):
        class Broken(Plugin):
            @pytest.hookimpl(wrapper=True)
            def pytest_strategies_context(self, config):
                yield
                raise LookupError("wrapper")

        session.conftest(".", Plugin("root"))
        session.conftest("tests/a", Broken())

        with pytest.raises(LookupError, match="wrapper"):
            session.context("tests/a")
        assert session.context("tests/b") == "root"


class TestHookRandomStream:
    def test_draws_in_the_hook_leave_the_callers_random_state_alone(self, session):
        session.conftest(".", Plugin(fn=lambda config: RNG.generator().random()))

        RNG.generator().seed("caller stream")
        expected = random.Random("caller stream").random()
        ctx = _call(lambda ctx: ctx)
        assert RNG.generator().random() == expected

        # The hook's own draws come from the stream root(S, "ctx") of the run seed
        assert ctx == random.Random(StreamKey.root(SEED, "ctx").seed_int()).random()

    def test_the_hook_draws_the_same_whatever_drew_before(self, tmp_path):
        contexts = []
        for seed in (1, 2):
            bench = Session(tmp_path)
            bench.conftest(".", Plugin(fn=lambda config: RNG.integer(0, 10**9)))
            runtime.push(bench.config).run_seed = SEED
            try:
                RNG.seed(seed)  # The factory's stream, which differs per test
                RNG.generator().random()
                contexts.append(_call(lambda ctx: ctx))
            finally:
                runtime.pop()

        assert contexts == [_ctx_draw(), _ctx_draw()]

    def test_each_implementation_starts_the_stream_anew(self, session):
        def draw(name):
            return lambda config: (name, RNG.integer(0, 10**9))

        session.conftest(".", Plugin(fn=draw("root")))
        session.conftest("tests/a", Plugin(fn=draw("A")))

        # Whichever is called first, each draws the first value of root(S, "ctx")
        assert session.context("tests/b") == ("root", _ctx_draw())
        assert session.context("tests/a") == ("A", _ctx_draw())

    def test_a_reseed_in_the_hook_stays_in_the_hook(self, session):
        def reseeding_hook(config):
            RNG.seed(5)
            return RNG.get_seed()

        session.conftest(".", Plugin(fn=reseeding_hook))

        RNG.seed(3)
        assert _call(lambda ctx: ctx) == 5
        assert RNG.get_seed() == 3


class TestHookErrors:
    def test_error_names_the_strategy_and_is_raised_for_every_factory(self, session):
        def broken(config):
            raise FileNotFoundError("tb.yaml")

        plugin = session.conftest(".", Plugin(fn=broken))

        for name in ("first", "second"):
            with pytest.raises(ValueError) as excinfo:
                _call(lambda nsamples, ctx: ctx, name=name)
            assert str(excinfo.value) == (
                f"Strategy factory '{name}' has a 'ctx' parameter, but the "
                "pytest_strategies_context hook raised FileNotFoundError: tb.yaml"
            )
            assert isinstance(excinfo.value.__cause__, FileNotFoundError)
        assert plugin.calls == 1

    def test_skip_is_propagated_as_is(self, session):
        def skip(config):
            pytest.skip("no testbench", allow_module_level=True)

        plugin = session.conftest(".", Plugin(fn=skip))

        for _ in range(2):
            with pytest.raises(pytest.skip.Exception, match="no testbench"):
                _call(lambda ctx: ctx)
        assert plugin.calls == 1

    def test_repeated_errors_keep_the_original_traceback(self, session):
        def broken(config):
            raise RuntimeError("boom")

        session.conftest(".", Plugin(fn=broken))

        depths = []
        for _ in range(3):
            with pytest.raises(ValueError) as excinfo:
                _call(lambda ctx: ctx)
            tb, depth = excinfo.value.__cause__.__traceback__, 0
            while tb is not None:
                tb, depth = tb.tb_next, depth + 1
            depths.append(depth)
        assert depths[0] == depths[1] == depths[2]

    def test_an_error_affects_only_the_folders_that_ask_the_implementation(self, session):
        def broken(config):
            raise RuntimeError("A is broken")

        session.conftest(".", Plugin("root"))
        session.conftest("tests/a", Plugin(fn=broken))
        session.conftest("tests/a/deep", Plugin("deep"))

        with pytest.raises(RuntimeError, match="A is broken"):
            session.context("tests/a")
        assert session.context("tests/a/deep") == "deep"
        assert session.context("tests/b") == "root"
        answer = runtime.path_context(session.rootpath / "tests/a").answer()
        assert answer.label == "tests/a/conftest.py"


class TestMigrationHint:
    """A factory that fails with ctx None is told where the hook is implemented."""

    def test_the_error_names_the_conftest_that_implements_the_hook(self, session):
        session.conftest("tests/a", Plugin({"x": 1}))
        session.conftest("tests/c", Plugin({"x": 3}))
        context = runtime.path_context(session.rootpath / "tests/b/test_b.py")

        with pytest.raises(ValueError) as excinfo:
            _call(lambda ctx: ctx["x"], context=context)

        assert str(excinfo.value) == (
            "Error calling strategy factory 's' (nsamples=3): TypeError: 'NoneType' object "
            "is not subscriptable. ctx is None for tests/b: no pytest_strategies_context "
            "implementation in this folder or above answered (implemented in "
            "tests/a/conftest.py, tests/c/conftest.py; move it to a common parent conftest)"
        )

    def test_not_when_no_conftest_implements_the_hook(self, session):
        context = runtime.path_context(session.rootpath / "tests/b")

        with pytest.raises(ValueError) as excinfo:
            _call(lambda ctx: ctx["x"], context=context)

        assert "ctx is None" not in str(excinfo.value)

    def test_not_when_the_factory_kept_a_default_that_is_not_none(self, session):
        session.conftest("tests/a", Plugin({"x": 1}))
        context = runtime.path_context(session.rootpath / "tests/b")

        def factory(ctx={}):  # noqa: B006
            return ctx["x"]

        with pytest.raises(ValueError) as excinfo:
            _call(factory, context=context)

        assert "ctx is None" not in str(excinfo.value)

    def test_when_the_factory_kept_a_default_of_none(self, session):
        session.conftest("tests/a", Plugin({"x": 1}))
        context = runtime.path_context(session.rootpath / "tests/b/test_b.py")

        def factory(ctx=None):
            return ctx["x"]

        with pytest.raises(ValueError, match="ctx is None for tests/b: .*tests/a/conftest.py"):
            _call(factory, context=context)

    def test_not_when_the_folder_has_a_context(self, session):
        session.conftest(".", Plugin({"y": 1}))
        session.conftest("tests/a", Plugin({"x": 1}))
        context = runtime.path_context(session.rootpath / "tests/b")

        with pytest.raises(ValueError) as excinfo:
            _call(lambda ctx: ctx["x"], context=context)

        assert "ctx is None" not in str(excinfo.value)

    def test_the_rootdir_folder_is_named_dot(self, session):
        session.conftest("tests/a", Plugin({"x": 1}))

        hint = runtime.path_context(session.rootpath).why_none()

        assert hint is not None and hint.startswith("ctx is None for .: ")

    def test_without_a_store_no_hint(self):
        assert FolderContext(None, tuple, lambda: None, lambda: 1).why_none() is None


class TestSessions:
    def test_each_session_calls_its_own_implementations(self, tmp_path):
        outer, inner = Session(tmp_path / "outer"), Session(tmp_path / "inner")
        outer_plugin = outer.conftest(".", Plugin("outer"))
        inner_plugin = inner.conftest(".", Plugin("inner"))
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
        assert (outer_plugin.calls, inner_plugin.calls) == (1, 1)

    def test_a_session_on_the_same_config_starts_a_store_of_its_own(self, session):
        plugin = session.conftest(".", Plugin("bench"))
        session.context(".")

        state = runtime.push(session.config)
        try:
            assert state.contexts is not runtime._stack[-2].contexts
            session.context(".")
        finally:
            runtime.pop()

        assert plugin.calls == 2


class TestGetContext:
    """get_context(config, path): a folder's context, for its conftest.py fixtures."""

    def test_returns_the_object_a_factory_there_receives(self, session):
        session.conftest(".", Plugin({"name": "root"}))
        session.conftest("tests/a", Plugin({"name": "A"}))
        context = runtime.path_context(session.rootpath / "tests/a/test_a.py")

        received = _call(lambda ctx: ctx, context=context)

        # The folder's conftest.py (__file__), the folder, and a file in it
        assert get_context(session.config, session.rootpath / "tests/a/conftest.py") is received
        assert get_context(session.config, str(session.rootpath / "tests/a")) is received
        assert get_context(session.config, session.rootpath / "tests/a/test_x.py") is received
        assert get_context(session.config, session.rootpath / "tests/b") == {"name": "root"}

    def test_a_folder_gets_the_nearest_answer_above_it(self, session):
        tests = session.conftest("tests", Plugin("tests"))
        session.conftest("tests/a/deep", Plugin(None))

        assert get_context(session.config, session.rootpath / "tests/a/deep") == "tests"
        assert get_context(session.config, session.rootpath / "tests/a") == "tests"
        assert get_context(session.config, session.rootpath) is None
        assert tests.calls == 1

    def test_the_implementation_s_exception_is_raised_as_it_is(self, session):
        def broken(config):
            raise LookupError("tb.yaml")

        plugin = session.conftest(".", Plugin(fn=broken))

        errors = []
        for _ in range(2):
            with pytest.raises(LookupError, match="tb.yaml") as excinfo:
                get_context(session.config, session.rootpath)
            errors.append(excinfo.value)
            # The implementation's own frame, where it raised, ends the traceback
            assert excinfo.traceback[-1].name == "broken"
        assert errors[0] is errors[1]
        assert plugin.calls == 1

    def test_a_skip_is_raised_as_it_is(self, session):
        def skip(config):
            pytest.skip("no testbench")

        session.conftest(".", Plugin(fn=skip))

        with pytest.raises(pytest.skip.Exception, match="no testbench"):
            get_context(session.config, session.rootpath)

    def test_an_unknown_config_raises_runtime_error(self, session):
        session.conftest(".", Plugin("bench"))
        other = SimpleNamespace(pluginmanager=session.pluginmanager, rootpath=session.rootpath)

        with pytest.raises(RuntimeError, match="config of no running pytest-strategies session"):
            get_context(other, session.rootpath)
        with pytest.raises(RuntimeError):
            get_context(None, session.rootpath)

    def test_the_config_of_a_session_that_ended_raises_runtime_error(self, tmp_path):
        bench = Session(tmp_path)
        bench.conftest(".", Plugin("bench"))
        runtime.push(bench.config)
        try:
            assert get_context(bench.config, tmp_path) == "bench"
        finally:
            runtime.pop()

        with pytest.raises(RuntimeError):
            get_context(bench.config, tmp_path)

    def test_an_outer_session_s_config_gets_the_outer_context_and_seed(self, tmp_path):
        outer, inner = Session(tmp_path / "outer"), Session(tmp_path / "inner")
        outer.conftest(".", Plugin(fn=lambda config: RNG.integer(0, 10**9)))
        inner.conftest(".", Plugin("inner"))
        runtime.push(outer.config).run_seed = SEED
        try:
            runtime.push(inner.config).run_seed = SEED + 1
            try:
                assert get_context(outer.config, outer.rootpath) == _ctx_draw()
                assert get_context(inner.config, inner.rootpath) == "inner"
            finally:
                runtime.pop()
        finally:
            runtime.pop()

    def test_it_is_public(self):
        import pytest_strategy

        assert "get_context" in pytest_strategy.__all__
        assert pytest_strategy.get_context is get_context


class TestStrategiesCtxMessage:
    """The message of the strategies_ctx fixture whose tests span two contexts."""

    def test_labels_sorted_with_each_one_s_first_test(self):
        message = _ctx_scopes_message(
            {
                "none": dict.fromkeys(["tests/b/test_b.py::test_b"]),
                "tests/tb_a/conftest.py": dict.fromkeys(
                    ["tests/tb_a/test_a.py::test_1", "tests/tb_a/test_a.py::test_2", "t::t3"]
                ),
                "conftest.py": dict.fromkeys(["tests/test_x.py::test_x"]),
            }
        )

        assert message == (
            "strategies_ctx is a session fixture, but the tests that use it have different "
            "contexts (conftest.py: tests/test_x.py::test_x; none: tests/b/test_b.py::test_b; "
            "tests/tb_a/conftest.py: tests/tb_a/test_a.py::test_1 and 2 more). In a folder with "
            "its own pytest_strategies_context, use pytest_strategy.get_context(request.config, "
            "__file__) in that folder's conftest.py fixtures."
        )
