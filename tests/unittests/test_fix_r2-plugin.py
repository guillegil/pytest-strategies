"""Unit tests for plugin fixes: global random state, nested sessions and strategy
file loading without the terminal plugin."""

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

    def test_explicit_seed_still_seeds_random(self):
        random.seed(99)
        expected = [random.random() for _ in range(3)]

        random.seed(0)
        RNG.seed(99)

        assert RNG.get_seed() == 99
        assert [random.random() for _ in range(3)] == expected


class TestRuntimeRestoresGlobalState:
    """Ending a session restores the random state and registry it began with."""

    def test_pop_continues_the_random_stream_instead_of_restarting_it(self):
        rt = StrategyRuntime()
        random.seed(123)
        expected = [random.random() for _ in range(3)]

        random.seed(123)
        got = [random.random()]
        rt.push("inner")
        RNG.seed(42)  # the inner session's --rng-seed
        RNG.refresh_seed(key="s:mod.test")
        random.random()
        rt.pop()
        got += [random.random(), random.random()]

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
            Strategy.strategy("r2_never_registered")(lambda x: None)

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

    def test_file_outside_rootdir_is_loaded(self, tmp_path, session, capsys):
        strategy_file = _write(tmp_path / "shared" / "strategies.py")
        (tmp_path / "proj").mkdir()

        PytestStrategyPlugin()._load_strategy_files(
            [strategy_file], _config(tmp_path / "proj", option=SimpleNamespace(verbose=2))
        )

        assert runtime.load_errors == []
        assert runtime.discovered_files == [strategy_file]
        assert f"pytest-strategies: Loaded {strategy_file}" in capsys.readouterr().out
