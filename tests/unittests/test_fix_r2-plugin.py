"""Unit tests for plugin fixes: global random state and nested sessions."""

import random

import pytest

from pytest_strategy import RNG, Strategy
from pytest_strategy._runtime import StrategyRuntime


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
