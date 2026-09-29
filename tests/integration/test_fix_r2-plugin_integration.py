"""
End-to-end tests for plugin fixes, run through pytester.

Subprocess runs are used unless a test is about in-process sessions. Strategy
names are unique to this module, so they do not clash with other tests.
"""

import random

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]

STRATEGIES = """
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("r2_ints")
def r2_ints(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)
"""

TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("r2_ints")
def test_ints(x):
    assert 0 <= x <= 10**9
"""


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


class TestGlobalRandomState:
    """Without --rng-seed the plugin leaves the global random module alone."""

    @pytest.mark.parametrize("args", [(), ("-n", "1")], ids=["plain", "xdist"])
    def test_conftest_seed_survives_an_unseeded_run(self, pytester, args):
        if args:
            pytest.importorskip("xdist")
        expected = random.Random(0).random()
        pytester.makeconftest("import random\n\nrandom.seed(0)\n")
        pytester.makepyfile(test_seeded=f"""
            import random

            def test_draw():
                assert random.random() == {expected!r}
            """)

        result = pytester.runpytest_subprocess(*args)

        result.assert_outcomes(passed=1)

    def test_xdist_workers_do_not_share_random_state(self, pytester):
        """A project without strategy files: each worker keeps its own entropy."""
        pytest.importorskip("xdist")
        pytester.makepyfile(test_ports="""
            import os
            import random
            from pathlib import Path

            import pytest

            @pytest.mark.parametrize("i", range(4))
            def test_port(i):
                worker = os.environ["PYTEST_XDIST_WORKER"]
                port = random.randint(20000, 2**62)
                Path(__file__).with_name(f"{worker}-{i}.port").write_text(str(port))
            """)

        result = pytester.runpytest_subprocess("-n", "2")

        result.assert_outcomes(passed=4)
        ports = {}
        for path in pytester.path.glob("gw*-*.port"):
            ports.setdefault(path.name.split("-")[0], set()).add(path.read_text())
        assert sorted(ports) == ["gw0", "gw1"]
        assert not ports["gw0"] & ports["gw1"]


class TestNestedSessionGlobalState:
    """An in-process session leaves no random state or registrations behind."""

    def test_inner_runs_do_not_restart_the_outer_random_stream(self, pytester):
        pytester.makepyfile(r2_nested_strategies=STRATEGIES)
        pytester.makepyfile(test_nested=TESTS)
        seed = RNG.get_seed()
        random.seed(123)
        expected = [random.random() for _ in range(3)]

        random.seed(123)
        got = [random.random()]
        pytester.runpytest_inprocess().assert_outcomes(passed=3)
        got.append(random.random())
        pytester.runpytest_inprocess("--rng-seed=5").assert_outcomes(passed=3)
        got.append(random.random())

        assert got == expected
        assert RNG.get_seed() == seed
        assert "r2_ints" not in Strategy._registry

    def test_sibling_sessions_do_not_warn_about_each_others_strategies(self, pytester):
        """The same strategy from a file of another name, as in a sibling pytester test."""
        source = STRATEGIES + TESTS.replace("from pytest_strategy import Strategy\n", "")
        error = ("-W", "error::pytest_strategy.strategy.PytestStrategiesWarning")

        first = pytester.makepyfile(test_first=source)
        pytester.runpytest_inprocess(*error).assert_outcomes(passed=3)
        assert "r2_ints" not in Strategy._registry

        first.unlink()
        pytester.makepyfile(test_second=source)
        pytester.runpytest_inprocess(*error).assert_outcomes(passed=3, warnings=0)
