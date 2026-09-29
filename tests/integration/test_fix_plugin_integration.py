"""
End-to-end tests for plugin fixes, run through pytester.

Subprocess runs are used where a test needs a fresh process (its own RNG seed,
its own strategy registry) or has to exit the inner session.
"""

import re

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("big_ints")
def big_ints(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)
"""

TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("big_ints")
def test_first(x):
    assert 0 <= x <= 10**9

@Strategy.strategy("big_ints")
def test_second(x):
    assert 0 <= x <= 10**9
"""


class TestDiscoveryUnderDotDirectory:
    """A project whose path contains a dot directory must still be discovered."""

    def test_project_under_dot_directory_runs(self, pytester, monkeypatch):
        project = pytester.path / ".hidden" / "proj"
        project.mkdir(parents=True)
        (project / "pytest.ini").write_text("[pytest]\n")
        (project / "strategies.py").write_text(STRATEGIES)
        (project / "test_dot.py").write_text(TESTS)
        # A strategy file inside a hidden directory of the project stays ignored.
        (project / ".venvlike").mkdir()
        (project / ".venvlike" / "other_strategies.py").write_text(
            STRATEGIES.replace('"big_ints"', '"hidden_strat"')
        )
        monkeypatch.chdir(project)

        result = pytester.runpytest_subprocess()
        result.assert_outcomes(passed=6)

        listing = pytester.runpytest_subprocess("--list-strategies")
        listing.stdout.fnmatch_lines(["*big_ints*"])
        listing.stdout.no_fnmatch_line("*hidden_strat*")


def _seed_from_header(result):
    """Return the seed printed in the 'pytest-strategies: RNG seed = N' header."""
    match = re.search(r"RNG seed = (\d+)", result.stdout.str())
    assert match, result.stdout.str()
    return int(match.group(1))


def _test_ids(result):
    """Return the parametrized test IDs (``test_x[...]``) found in a run's output."""
    return re.findall(r"test_\w+\[[^\]]*\]", result.stdout.str())


class TestUnseededRunReproducibility:
    """The seed printed by an unseeded run must reproduce it exactly."""

    def test_printed_seed_reproduces_module_level_draws(self, pytester):
        """Randomness drawn before the first factory runs must also be reproduced.

        The strategy file draws an offset at import time, before any factory
        runs. Without a seed the global random state used to come from OS
        entropy, so rerunning with the printed seed gave different vectors.
        """
        pytester.makepyfile(offset_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNG

            OFFSET = RNG.integer(0, 10**9)

            @Strategy.register("offset_strat")
            def offset(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(OFFSET, OFFSET + 10**6)), nsamples=3
                )
            """)
        pytester.makepyfile(test_offset="""
            from pytest_strategy import Strategy

            @Strategy.strategy("offset_strat")
            def test_offset(x):
                pass
            """)

        unseeded = pytester.runpytest_subprocess("--collect-only")
        seed = _seed_from_header(unseeded)
        reproduced = pytester.runpytest_subprocess("--collect-only", f"--rng-seed={seed}")

        unseeded_ids = _test_ids(unseeded)
        assert len(unseeded_ids) == 3
        assert _test_ids(reproduced) == unseeded_ids


class TestXdistSeedSharing:
    """pytest-xdist workers must all use the controller's seed."""

    def test_unseeded_xdist_run_collects_the_same_tests_on_every_worker(self, pytester):
        pytest.importorskip("xdist")
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_xdist=TESTS)

        result = pytester.runpytest_subprocess("-n", "2")

        result.stdout.no_fnmatch_line("*Different tests were collected*")
        result.assert_outcomes(passed=6)

    def test_worker_uses_seed_from_workerinput(self, pytester):
        """Without --rng-seed, a worker takes the seed the controller sent."""
        pytester.makeconftest("""
            import pytest

            @pytest.hookimpl(tryfirst=True)
            def pytest_configure(config):
                # Stand in for an xdist worker (workerinput is set before configure).
                config.workerinput = {"pytest_strategies_seed": 4242}
            """)
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_worker=TESTS)

        worker = pytester.runpytest_subprocess("-p", "no:xdist", "--collect-only")
        seeded = pytester.runpytest_subprocess(
            "-p", "no:xdist", "--collect-only", "--rng-seed=4242"
        )

        assert _seed_from_header(worker) == 4242
        assert _test_ids(worker) == _test_ids(seeded)

    def test_plugin_works_without_xdist(self, pytester):
        """The optional xdist hook must not break a run where xdist is absent."""
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_no_xdist=TESTS)

        result = pytester.runpytest_subprocess("-p", "no:xdist")

        result.assert_outcomes(passed=6)
