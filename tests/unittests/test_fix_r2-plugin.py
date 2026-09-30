"""Unit tests for plugin fixes: global random state, nested sessions, strategy file
loading without the terminal plugin, and the scope and order of discovery."""

import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from pytest_strategy import RNG, Strategy
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy.plugin import PytestStrategyPlugin

STRATEGY_SOURCE = """
from pytest_strategy import Strategy

@Strategy.register("r2_unit_strat")
def r2_unit_strat(nsamples):
    return ("x",), [(1,)]
"""

MISSING_MODULE = "pytest_strategies_missing_module"
SKIP_REASON = f"could not import '{MISSING_MODULE}': No module named '{MISSING_MODULE}'"


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what these tests change globally: the seed, random state and registry."""
    seed = RNG.get_seed()
    state = random.getstate()
    registry = dict(Strategy._registry)
    yield
    RNG._seed = seed
    random.setstate(state)
    Strategy._registry.clear()
    Strategy._registry.update(registry)


def _write(path, source=STRATEGY_SOURCE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_virtualenv(path):
    """Create a virtualenv-like directory holding a file that looks like a strategy file."""
    (path / "lib").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return _write(path / "lib" / "site-packages" / "pytest_strategy" / "strategy.py")


def _config(rootpath, option=None):
    """A minimal config for _load_strategy_files, without a terminal reporter."""
    return SimpleNamespace(
        option=option if option is not None else SimpleNamespace(verbose=0),
        rootpath=rootpath,
        pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
    )


@pytest.fixture
def session():
    """Run the test inside a fresh runtime session."""
    runtime.push("r2-unit")
    try:
        yield
    finally:
        runtime.pop()


class TestSeedLeavesGlobalRandomAlone:
    """RNG.seed(None) keeps the seed and does not reseed the global random module."""

    def test_seed_none_keeps_seed_and_random_state(self):
        RNG.seed(1234)
        random.seed(0)  # e.g. a conftest pinning its own randomness
        state = random.getstate()

        RNG.seed(None)

        assert RNG.get_seed() == 1234
        assert random.getstate() == state

    def test_explicit_seed_leaves_random_alone(self):
        random.seed(0)
        state = random.getstate()

        RNG.seed(99)

        assert RNG.get_seed() == 99
        assert random.getstate() == state
        # The generator draws what the global random state seeded with it would
        assert RNG.generator().random() == random.Random(99).random()


class TestRuntimeRestoresGlobalState:
    """Ending a session restores the random state and registry it began with."""

    def test_pop_continues_the_generator_stream_instead_of_restarting_it(self):
        rt = StrategyRuntime()
        RNG.seed(123)
        expected = [RNG.generator().random() for _ in range(3)]

        RNG.seed(123)
        got = [RNG.generator().random()]
        rt.push("inner")
        RNG.seed(42)  # the inner session's --rng-seed
        RNG.refresh_seed(key="s:mod.test")
        RNG.generator().random()
        rt.pop()
        got += [RNG.generator().random(), RNG.generator().random()]

        assert got == expected

    def test_pop_restores_seed_without_touching_random(self):
        rt = StrategyRuntime()
        RNG.seed(7)
        rt.push("inner")
        RNG.seed(42)
        random.seed(5)
        rt.push("innermost")
        state = random.getstate()
        rt.pop()

        assert RNG.get_seed() == 42
        assert random.getstate() == state
        rt.pop()
        assert RNG.get_seed() == 7

    def test_pop_restores_the_registry_in_place(self):
        rt = StrategyRuntime()
        registry = Strategy._registry

        def outer(nsamples):
            return ("x",), [(1,)]

        def inner(nsamples):
            return ("x",), [(2,)]

        Strategy.register("r2_outer_strat")(outer)
        rt.push("inner")
        Strategy.register("r2_inner_strat")(inner)
        del Strategy._registry["r2_outer_strat"]
        rt.pop()

        assert Strategy._registry is registry
        assert "r2_inner_strat" not in Strategy._registry
        assert Strategy._registry["r2_outer_strat"] is outer


class TestLoadWithoutTerminalPlugin:
    """With -p no:terminal, config.option has no ``verbose`` and there is no reporter."""

    def test_successful_load_is_not_a_load_error(self, tmp_path, session):
        strategy_file = _write(tmp_path / "strategies.py")

        PytestStrategyPlugin()._load_strategy_files(
            [strategy_file], _config(tmp_path, option=SimpleNamespace())
        )

        assert runtime.load_errors == []
        assert runtime.discovered_files == [strategy_file]
        assert "r2_unit_strat" in Strategy._registry

    def test_skipped_file_does_not_raise(self, tmp_path, session):
        strategy_file = _write(
            tmp_path / "strategies.py",
            f"import pytest\npytest.importorskip({MISSING_MODULE!r})\n" + STRATEGY_SOURCE,
        )

        PytestStrategyPlugin()._load_strategy_files(
            [strategy_file], _config(tmp_path, option=SimpleNamespace())
        )

        assert runtime.load_errors == []
        assert runtime.skipped_files == [(strategy_file, SKIP_REASON)]


class TestSkippedFilesAreNamed:
    """A strategy file that skipped itself is named in the "not found" error."""

    def test_skipped_file_is_recorded(self, tmp_path, session):
        strategy_file = _write(
            tmp_path / "strategies.py",
            "import pytest\npytest.skip('no gpu', allow_module_level=True)\n" + STRATEGY_SOURCE,
        )

        PytestStrategyPlugin()._load_strategy_files([strategy_file], _config(tmp_path))

        assert runtime.skipped_files == [(strategy_file, "no gpu")]
        assert runtime.load_errors == []
        assert runtime.discovered_files == []

    def test_strategy_not_found_error_lists_skipped_files(self, session):
        skipped = Path("/project/tests/strategies.py")
        runtime.record_skipped_file(skipped, SKIP_REASON)

        with pytest.raises(ValueError) as excinfo:
            PytestStrategyPlugin().resolve(
                "r2_never_registered", Path("/project/tests/test_x.py"), None
            )

        message = str(excinfo.value)
        assert message.startswith("Strategy 'r2_never_registered' not found.")
        assert f"\nStrategy files that were skipped:\n  {skipped}: {SKIP_REASON}" in message

    def test_skipped_files_belong_to_their_session(self):
        rt = StrategyRuntime()
        rt.record_skipped_file("ghost.py", "reason")  # no session: dropped
        assert rt.skipped_files == []
        rt.push("outer")
        rt.push("inner")
        rt.record_skipped_file("inner.py", "reason")
        rt.pop()
        assert rt.skipped_files == []
        rt.pop()


class TestVerboseLoadOutsideRootdir:
    """-vv reports a file outside rootdir as loaded, by its absolute path."""

    def test_file_outside_rootdir_is_loaded(self, tmp_path, session):
        strategy_file = _write(tmp_path / "shared" / "strategies.py")
        (tmp_path / "proj").mkdir()
        lines = []
        terminal = SimpleNamespace(write_line=lambda line, **markup: lines.append(line))
        config = _config(tmp_path / "proj", option=SimpleNamespace(verbose=2))
        config.pluginmanager = SimpleNamespace(
            get_plugin=lambda name: terminal if name == "terminalreporter" else None
        )

        PytestStrategyPlugin()._load_strategy_files([strategy_file], config)

        assert runtime.load_errors == []
        assert runtime.discovered_files == [strategy_file]
        assert lines == [f"pytest-strategies: Loaded {strategy_file}"]


class TestDiscoverySkipsEnvironmentsAndIgnoredDirectories:
    """Discovery prunes virtualenvs and norecursedirs, like pytest's own collection."""

    def test_virtualenv_of_any_name_is_skipped(self, tmp_path):
        project = tmp_path / "proj"
        strategy_file = _write(project / "strategies.py")
        _make_virtualenv(project / "venv")
        _make_virtualenv(project / "myenv")

        found = PytestStrategyPlugin()._discover_strategy_files([project])

        assert found == [strategy_file]

    def test_norecursedirs_patterns_are_skipped(self, tmp_path):
        project = tmp_path / "proj"
        strategy_file = _write(project / "tests" / "strategies.py")
        _write(project / "build" / "strategies.py")
        _write(project / "tests" / "old.egg" / "strategies.py")

        found = PytestStrategyPlugin()._discover_strategy_files([project], ["build", "*.egg"])

        assert found == [strategy_file]

    def test_skipped_directory_given_as_search_path_is_searched(self, tmp_path):
        strategy_file = _write(tmp_path / "build" / "strategies.py")
        venv_file = _make_virtualenv(tmp_path / "venv")

        plugin = PytestStrategyPlugin()

        assert plugin._discover_strategy_files([tmp_path / "build"], ["build"]) == [strategy_file]
        assert plugin._discover_strategy_files([tmp_path / "venv"]) == [venv_file]


class _ReversedScandir:
    """os.scandir with the entries in reverse name order, as some filesystems list them."""

    real_scandir = os.scandir

    def __init__(self, path="."):
        with self.real_scandir(path) as entries:
            self._entries = iter(sorted(entries, key=lambda e: e.name, reverse=True))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def close(self):
        pass


class TestDiscoveryOrder:
    """Strategy files are loaded in the same order on every filesystem."""

    def test_files_are_sorted_by_path_below_the_search_path(self, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        files = [
            _write(project / rel)
            for rel in ("a/strategies.py", "b/strategies.py", "b_strategies.py", "c/z_strategy.py")
        ]
        monkeypatch.setattr(os, "scandir", _ReversedScandir)

        found = PytestStrategyPlugin()._discover_strategy_files([project])

        assert found == files


def _search_config(rootdir, testpaths=(), args=(), invocation_dir=None):
    """A minimal config for _search_paths."""
    ini = {"testpaths": list(testpaths)}
    return SimpleNamespace(
        rootpath=rootdir,
        getini=ini.__getitem__,
        args=list(args),
        invocation_params=SimpleNamespace(dir=invocation_dir or rootdir),
    )


class TestSearchPaths:
    """Search paths follow testpaths globs and the paths given on the command line."""

    def test_testpaths_glob_patterns_are_expanded(self, tmp_path):
        for name in ("beta", "alpha"):
            (tmp_path / "pkgs" / name / "tests").mkdir(parents=True)

        config = _search_config(tmp_path, testpaths=["pkgs/*/tests"])
        paths = PytestStrategyPlugin()._search_paths(config)

        assert paths == [
            tmp_path / "pkgs" / "alpha" / "tests",
            tmp_path / "pkgs" / "beta" / "tests",
        ]

    def test_literal_testpaths_are_kept(self, tmp_path):
        (tmp_path / "tests").mkdir()

        config = _search_config(tmp_path, testpaths=["tests", "../shared"])
        paths = PytestStrategyPlugin()._search_paths(config)

        assert paths == [tmp_path / "tests", tmp_path / "../shared"]

    def test_command_line_paths_outside_the_search_paths_are_added(self, tmp_path):
        (tmp_path / "tests").mkdir()
        _write(tmp_path / "integration" / "test_integ.py", "def test_integ():\n    pass\n")
        (tmp_path / "e2e").mkdir()

        config = _search_config(
            tmp_path,
            testpaths=["tests"],
            args=["integration/test_integ.py::test_integ", str(tmp_path / "e2e"), "tests"],
        )
        paths = PytestStrategyPlugin()._search_paths(config)

        assert paths == [tmp_path / "tests", tmp_path / "integration", tmp_path / "e2e"]

    def test_command_line_paths_are_relative_to_the_invocation_dir(self, tmp_path):
        (tmp_path / "tests" / "sub").mkdir(parents=True)
        (tmp_path / "other").mkdir()

        config = _search_config(
            tmp_path,
            testpaths=["tests"],
            args=["sub", "../other", "not.a.path"],
            invocation_dir=tmp_path / "tests",
        )
        paths = PytestStrategyPlugin()._search_paths(config)

        assert paths == [tmp_path / "tests", tmp_path / "other"]

    def test_command_line_path_in_a_skipped_directory_is_added(self, tmp_path):
        (tmp_path / "build" / "tests").mkdir(parents=True)
        (tmp_path / ".hidden").mkdir()

        config = _search_config(tmp_path, args=["build/tests", ".hidden"])
        paths = PytestStrategyPlugin()._search_paths(config, ["build"])

        assert paths == [tmp_path, tmp_path / "build" / "tests", tmp_path / ".hidden"]
