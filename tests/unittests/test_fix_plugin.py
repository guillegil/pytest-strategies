"""Unit tests for plugin fixes: discovery, seeding and xdist support."""

import random
from types import SimpleNamespace

from pytest_strategy import RNG
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
    """RNG.seed(None) keeps the seed but still refreshes the random state."""

    def test_seed_none_reseeds_random_from_current_seed(self):
        RNG.seed(1234)
        expected = [random.random() for _ in range(3)]

        random.seed()  # state from OS entropy, as in a fresh unseeded process
        RNG.seed(None)

        assert RNG.get_seed() == 1234
        assert [random.random() for _ in range(3)] == expected


class TestXdistConfigureNode:
    """The controller hands its seed to every xdist worker."""

    def test_configure_node_sends_current_seed(self):
        RNG.seed(98765)
        node = SimpleNamespace(workerinput={})

        PytestStrategyPlugin().pytest_configure_node(node)

        assert node.workerinput["pytest_strategies_seed"] == 98765
