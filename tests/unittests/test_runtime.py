"""Unit tests for the per-session StrategyRuntime stack."""

from pytest_strategy._runtime import StrategyRuntime


class TestStrategyRuntimeStack:
    """The runtime keeps a stack of per-session state."""

    def test_push_and_pop_balance(self):
        rt = StrategyRuntime()
        assert rt.current is None
        rt.push("cfg1")
        assert rt.current.config == "cfg1"
        rt.pop()
        assert rt.current is None

    def test_pop_on_empty_stack_is_safe(self):
        rt = StrategyRuntime()
        rt.pop()  # must not raise
        assert rt.current is None

    def test_nested_sessions_have_independent_state(self):
        rt = StrategyRuntime()
        rt.push("outer")
        rt.strategies_loaded = True
        rt.push("inner")
        # Inner session starts fresh and shadows the outer config.
        assert rt.strategies_loaded is False
        assert rt.current.config == "inner"
        rt.pop()
        # Outer state is restored intact.
        assert rt.strategies_loaded is True
        assert rt.current.config == "outer"

    def test_strategies_loaded_setter_with_empty_stack_is_noop(self):
        rt = StrategyRuntime()
        rt.strategies_loaded = True
        assert rt.strategies_loaded is False

    def test_record_discovered_file_targets_active_session(self):
        rt = StrategyRuntime()
        rt.record_discovered_file("ghost")  # no session: dropped, no error
        assert rt.discovered_files == []
        rt.push("cfg")
        rt.record_discovered_file("real")
        assert rt.discovered_files == ["real"]
        rt.pop()
        assert rt.discovered_files == []

    def test_each_session_has_a_context_store_of_its_own(self):
        rt = StrategyRuntime()
        outer = rt.push("outer").contexts
        assert outer.config == "outer"
        inner = rt.push("inner").contexts
        assert inner is not outer
        assert inner.config == "inner"
        rt.pop()
        # The outer session's store, with what its implementations answered, is back
        assert rt.current.contexts is outer
        rt.pop()
        assert rt.current is None
