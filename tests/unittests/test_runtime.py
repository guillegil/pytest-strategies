"""Unit tests for the per-session StrategyRuntime stack."""

from pytest_strategy._runtime import StrategyRuntime


class TestStrategyRuntimeStack:
    """The runtime keeps a stack of per-session state."""

    def test_push_and_pop_balance(self):
        rt = StrategyRuntime()
        assert rt.current is None
        rt.push("cfg1")
        assert rt.config == "cfg1"
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
        assert rt.config == "inner"
        rt.pop()
        # Outer state is restored intact.
        assert rt.strategies_loaded is True
        assert rt.config == "outer"

    def test_config_setter_with_empty_stack_does_not_leak(self):
        rt = StrategyRuntime()
        rt.config = "stray"  # must NOT push an unpoppable state
        assert rt.current is None
        assert rt.config is None

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
