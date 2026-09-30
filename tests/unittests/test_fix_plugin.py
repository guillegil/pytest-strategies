"""Unit tests for plugin fixes: discovery, seeding, xdist, strategy file loading and options."""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pytest_strategy import RNG
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy.plugin import PytestStrategyPlugin

STRATEGY_SOURCE = """
from pytest_strategy import Strategy

@Strategy.register("s")
def s(nsamples):
    return ("x",), [(1,)]
"""


def _write_strategy_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STRATEGY_SOURCE)
    return path


class TestDiscoveryHiddenDirectories:
    """Only directories below the search path are filtered as hidden."""

    def test_project_under_dot_directory_is_searched(self, tmp_path):
        """An ancestor of the search path starting with '.' must not hide its files."""
        project = tmp_path / ".hidden" / "proj"
        strategy_file = _write_strategy_file(project / "strategies.py")

        found = PytestStrategyPlugin()._discover_strategy_files([project])

        assert found == [strategy_file]

    def test_hidden_and_cache_directories_inside_project_are_skipped(self, tmp_path):
        """Hidden and __pycache__ directories below the search path stay excluded."""
        project = tmp_path / ".hidden" / "proj"
        strategy_file = _write_strategy_file(project / "strategies.py")
        _write_strategy_file(project / ".venv" / "lib_strategies.py")
        _write_strategy_file(project / "__pycache__" / "strategies.py")

        found = PytestStrategyPlugin()._discover_strategy_files([project])

        assert found == [strategy_file]

    def test_search_path_going_up_a_level_is_searched(self, tmp_path):
        """A testpaths entry such as '../shared' keeps a literal '..' in the path."""
        (tmp_path / "proj").mkdir()
        _write_strategy_file(tmp_path / "shared" / "strategies.py")

        search_path = tmp_path / "proj" / ".." / "shared"

        found = PytestStrategyPlugin()._discover_strategy_files([search_path])

        assert [p.name for p in found] == ["strategies.py"]


class TestSeedNone:
    """RNG.seed(None) keeps the seed and restarts the generator from it."""

    def test_seed_none_keeps_seed_and_restarts_the_generator(self):
        RNG.seed(1234)
        expected = [RNG.generator().random() for _ in range(3)]

        RNG.generator().random()
        RNG.seed(None)
        assert RNG.get_seed() == 1234
        assert [RNG.generator().random() for _ in range(3)] == expected

        RNG.generator().random()
        RNG.refresh_seed()
        assert [RNG.generator().random() for _ in range(3)] == expected

    def test_seeding_leaves_the_global_random_state_alone(self):
        state = random.getstate()
        RNG.seed(1234)
        RNG.refresh_seed(key="s:mod.test_a")
        RNG.integer(0, 100)
        assert random.getstate() == state


class TestXdistConfigureNode:
    """The controller hands its seed to every xdist worker."""

    def test_configure_node_sends_current_seed(self):
        RNG.seed(98765)
        node = SimpleNamespace(workerinput={})

        PytestStrategyPlugin().pytest_configure_node(node)

        assert node.workerinput["pytest_strategies_seed"] == 98765


class TestKeyedRefreshSeed:
    """RNG.refresh_seed(key) gives each key its own reproducible stream."""

    @staticmethod
    def _draw(key=None):
        RNG.refresh_seed(key=key)
        return [RNG.generator().random() for _ in range(3)]

    def test_same_key_repeats_its_stream(self):
        RNG.seed(42)
        assert self._draw("s:mod.test_a") == self._draw("s:mod.test_a")

    def test_different_keys_get_different_streams(self):
        RNG.seed(42)
        assert self._draw("s:mod.test_a") != self._draw("s:mod.test_b")
        assert self._draw("s:mod.test_a") != self._draw("t:mod.test_a")

    def test_keyed_stream_follows_the_seed(self):
        RNG.seed(42)
        first = self._draw("s:mod.test_a")
        RNG.seed(43)
        assert self._draw("s:mod.test_a") != first

    def test_without_key_the_run_seed_is_used(self):
        RNG.seed(42)
        # The same values as the global random state seeded with it (2.x behavior)
        stdlib = random.Random(42)
        expected = [stdlib.random() for _ in range(3)]
        assert self._draw() == expected

    def test_keyed_stream_is_stable_across_processes(self):
        """The key must not go through hash(), which is salted per process."""
        code = (
            "from pytest_strategy import RNG\n"
            "RNG.seed(42)\n"
            "RNG.refresh_seed(key='s:mod.test_a')\n"
            "print(RNG.generator().random())\n"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for hash_seed in ("1", "2")
        }
        assert len(outputs) == 1


class TestRuntimeRestoresSeed:
    """Ending a session restores the RNG seed it began with."""

    def test_pop_restores_seed_of_enclosing_session(self):
        rt = StrategyRuntime()
        RNG.seed(7)
        rt.push("outer")
        rt.push("inner")
        RNG.seed(42)  # the inner session's --rng-seed

        rt.pop()
        assert RNG.get_seed() == 7

        rt.pop()
        assert RNG.get_seed() == 7


class _TerminalReporterStub:
    """Collects the lines written through the terminal reporter."""

    def __init__(self):
        self.lines = []

    def write_line(self, line, **markup):
        self.lines.append(line)


@pytest.fixture
def load_session(tmp_path):
    """A fresh runtime session and a minimal config for _load_strategy_files.

    ``config.terminal.lines`` holds what the plugin wrote to the terminal.
    """
    terminal = _TerminalReporterStub()
    config = SimpleNamespace(
        option=SimpleNamespace(verbose=0),
        rootpath=tmp_path,
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: terminal if name == "terminalreporter" else None
        ),
        terminal=terminal,
    )
    runtime.push(config)
    try:
        yield config
    finally:
        runtime.pop()


def _discovered_modules():
    return {name for name in sys.modules if name.startswith("pytest_strategies_discovered.")}


class TestLoadStrategyFilesOutcomes:
    """Skipped or failing strategy files are dropped cleanly."""

    @pytest.mark.parametrize(
        "body",
        [
            "import pytest\npytest.skip('no gpu', allow_module_level=True)\n",
            "import pytest\npytest.importorskip('pytest_strategies_missing_module')\n",
            "import pytest\npytest.fail('broken setup')\n",
            "raise RuntimeError('boom')\n",
        ],
        ids=["skip", "importorskip", "fail", "error"],
    )
    def test_file_is_not_loaded_and_leaves_no_module_behind(self, tmp_path, load_session, body):
        strategy_file = tmp_path / "strategies.py"
        strategy_file.write_text(body + STRATEGY_SOURCE)
        before = _discovered_modules()

        try:
            PytestStrategyPlugin()._load_strategy_files([strategy_file], load_session)
        except pytest.skip.Exception as e:
            # Escaping, it would mark this test skipped instead of failing it.
            pytest.fail(f"Skipped escaped _load_strategy_files: {e}")

        assert runtime.discovered_files == []
        assert _discovered_modules() == before


class TestLoadErrorsAreReported:
    """A strategy file that fails to load is recorded and reported at any verbosity."""

    def test_load_error_is_recorded_and_written_without_verbose(self, tmp_path, load_session):
        strategy_file = tmp_path / "strategies.py"
        strategy_file.write_text(
            "from pytest_strategies_missing_helper import x\n" + STRATEGY_SOURCE
        )

        PytestStrategyPlugin()._load_strategy_files([strategy_file], load_session)

        error = "ModuleNotFoundError: No module named 'pytest_strategies_missing_helper'"
        assert runtime.load_errors == [(strategy_file, error)]
        assert load_session.terminal.lines == [
            f"pytest-strategies: Warning - Failed to load {strategy_file}: {error}"
        ]

    def test_skipped_file_is_not_a_load_error(self, tmp_path, load_session):
        strategy_file = tmp_path / "strategies.py"
        strategy_file.write_text(
            "import pytest\npytest.skip('no gpu', allow_module_level=True)\n" + STRATEGY_SOURCE
        )

        PytestStrategyPlugin()._load_strategy_files([strategy_file], load_session)

        assert runtime.load_errors == []
        assert load_session.terminal.lines == []

    def test_strategy_not_found_error_lists_failed_files(self, tmp_path, load_session):

        broken = tmp_path / "strategies.py"
        runtime.record_load_error(broken, "SyntaxError: expected ':' (strategies.py, line 3)")

        with pytest.raises(ValueError) as excinfo:
            PytestStrategyPlugin().resolve(
                "pytest_strategies_never_registered", Path("/project/tests/test_x.py"), None
            )

        message = str(excinfo.value)
        assert message.startswith("Strategy 'pytest_strategies_never_registered' not found.")
        assert "Strategy files that failed to load:" in message
        assert f"{broken}: SyntaxError: expected ':' (strategies.py, line 3)" in message

    def test_load_errors_belong_to_their_session(self):
        rt = StrategyRuntime()
        rt.record_load_error("ghost", "Error: x")  # no session: dropped
        assert rt.load_errors == []
        rt.push("outer")
        rt.push("inner")
        rt.record_load_error("inner.py", "Error: x")
        rt.pop()
        assert rt.load_errors == []
        rt.pop()


class TestNsamplesOptionType:
    """--nsamples accepts 'auto' (any case) or an integer >= 0, nothing else."""

    @staticmethod
    def _parse(value):
        from pytest_strategy.plugin import _nsamples_type

        return _nsamples_type(value)

    @pytest.mark.parametrize(
        "value, expected",
        [("auto", "auto"), ("AUTO", "auto"), (" Auto ", "auto"), ("0", 0), ("7", 7)],
    )
    def test_valid_values_are_parsed(self, value, expected):
        assert self._parse(value) == expected

    @pytest.mark.parametrize("value", ["-1", "abc", "2.5", ""])
    def test_invalid_values_raise_argument_type_error(self, value):
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            self._parse(value)
        assert str(excinfo.value) == f"expected an integer >= 0 or 'auto', got {value!r}"
