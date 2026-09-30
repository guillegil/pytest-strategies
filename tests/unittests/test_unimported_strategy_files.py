"""Unit tests for the files named like strategy files that discovery does not import,
and for the "Strategy not found" error that lists them."""

from pathlib import Path

import pytest

from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy.plugin import PytestStrategyPlugin

DECORATED = """
from pytest_strategy import Strategy

@Strategy.register("unimp_strat")
def strat(nsamples):
    return ("x",), [(1,)]
"""

ALIASED = """
from pytest_strategy import Strategy as S

@S.register("unimp_strat")
def strat(nsamples):
    return ("x",), [(1,)]
"""

# Registers without a decorator, so discovery does not import it
PLAIN_CALL = """
from pytest_strategy import register

def strat(nsamples):
    return ("x",), [(1,)]

register("unimp_strat")(strat)
"""

HINT = (
    "\nFiles matching a strategy file name that were not imported because "
    "they contain no registration decorator (use @register(...)):"
)
LAZY_LOAD_NOTE = (
    "\nStrategy files are imported when pytest first collects a test module in their "
    "folder or below, so an import that only works after other test modules are collected fails."
)


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.fixture
def session():
    """Run the test inside a fresh runtime session."""
    runtime.push("unimported-unit")
    try:
        yield
    finally:
        runtime.pop()


def _not_found_message(name="unimp_never_registered"):
    with pytest.raises(ValueError) as excinfo:
        PytestStrategyPlugin().resolve(name, Path("/project/tests/test_x.py"), None)
    return str(excinfo.value)


class TestDiscoveryCollectsUnimportedFiles:
    """Files matching a pattern without a registration decorator are handed back, not returned."""

    def test_files_without_the_decorator_are_collected_in_order(self, tmp_path):
        decorated = _write(tmp_path / "strategies.py", DECORATED)
        plain = _write(tmp_path / "b" / "plain_strategy.py", PLAIN_CALL)
        unrelated = _write(tmp_path / "a" / "order_strategies.py", "X = 1\n")
        _write(tmp_path / "helpers.py", DECORATED)  # no strategy file name

        unimported: list[Path] = []
        found = PytestStrategyPlugin()._discover_strategy_files([tmp_path], (), unimported)

        assert found == [decorated]
        # A file that never mentions register is an unrelated module, not a candidate
        assert unimported == [plain]
        assert unrelated not in found

    @pytest.mark.parametrize(
        "decorator",
        [
            '@Strategy.register("x")',
            "@register('x')",
            '@S.register("x")',
            '@pytest_strategy.register(\n    "x"\n)',
            "@pytest_strategy.Strategy.register(NAME)",
        ],
    )
    def test_each_decorator_form_is_imported(self, tmp_path, decorator):
        path = _write(tmp_path / "strategies.py", f"{decorator}\ndef f(nsamples):\n    pass\n")

        found = PytestStrategyPlugin()._discover_strategy_files([tmp_path], ())

        assert found == [path]

    def test_test_modules_that_only_use_strategies_are_not_collected(self, tmp_path):
        _write(
            tmp_path / "test_strategy.py",
            "from pytest_strategy import Strategy\n\n"
            "@Strategy.strategy('x')\ndef test_x(a):\n    pass\n",
        )

        unimported: list[Path] = []
        PytestStrategyPlugin()._discover_strategy_files([tmp_path], (), unimported)

        assert unimported == []

    def test_file_in_another_source_encoding_is_imported(self, tmp_path):
        path = tmp_path / "latin_strategies.py"
        path.write_bytes(
            b"# -*- coding: latin-1 -*-\n# caf\xe9\n"
            b"from pytest_strategy import Strategy\n\n@Strategy.register('lat')\n"
            b"def lat(nsamples):\n    return ('x', [1])\n"
        )

        unimported: list[Path] = []
        found = PytestStrategyPlugin()._discover_strategy_files([tmp_path], (), unimported)

        assert found == [path]
        assert unimported == []

    def test_skipped_directories_are_not_collected(self, tmp_path):
        _write(tmp_path / ".hidden" / "strategies.py", PLAIN_CALL)
        _write(tmp_path / "build" / "strategies.py", PLAIN_CALL)
        _write(tmp_path / "__pycache__" / "strategies.py", PLAIN_CALL)
        _write(tmp_path / "myenv" / "pyvenv.cfg", "home = /usr/bin\n")
        _write(tmp_path / "myenv" / "lib" / "strategies.py", PLAIN_CALL)
        _write(tmp_path / ".plain_strategies.py", PLAIN_CALL)

        unimported: list[Path] = []
        PytestStrategyPlugin()._discover_strategy_files([tmp_path], ["build"], unimported)

        assert unimported == []

    def test_file_under_two_search_paths_is_collected_once(self, tmp_path):
        plain = _write(tmp_path / "tests" / "plain_strategies.py", PLAIN_CALL)

        unimported: list[Path] = []
        PytestStrategyPlugin()._discover_strategy_files(
            [tmp_path, tmp_path / "tests"], (), unimported
        )

        assert unimported == [plain]


class TestUnimportedFilesBelongToTheirSession:
    def test_recorded_only_in_the_active_session(self):
        rt = StrategyRuntime()
        rt.record_unimported_file(Path("ghost_strategies.py"))  # no session: dropped
        assert rt.unimported_files == []
        rt.push("outer")
        rt.push("inner")
        rt.record_unimported_file(Path("inner_strategies.py"))
        assert rt.unimported_files == [Path("inner_strategies.py")]
        rt.pop()
        assert rt.unimported_files == []
        rt.pop()


class TestNotFoundErrorListsUnimportedFiles:
    def test_unimported_files_are_listed_with_the_hint(self, session):
        first = Path("/project/tests/plain_strategies.py")
        second = Path("/project/tests/sub/strategy.py")
        runtime.record_unimported_file(first)
        runtime.record_unimported_file(second)

        message = _not_found_message()

        assert message.startswith("Strategy 'unimp_never_registered' not found.")
        assert message.endswith(f"{HINT}\n  {first}\n  {second}")

    def test_nothing_is_listed_without_unimported_files(self, session):
        message = _not_found_message()

        assert "were not imported" not in message
        assert LAZY_LOAD_NOTE not in message

    def test_failed_files_are_followed_by_the_session_start_note(self, session):
        broken = Path("/project/tests/strategies.py")
        error = "ModuleNotFoundError: No module named 'helpers'"
        runtime.record_load_error(broken, error)

        message = _not_found_message()

        expected = f"\nStrategy files that failed to load:\n  {broken}: {error}{LAZY_LOAD_NOTE}"
        assert expected in message
