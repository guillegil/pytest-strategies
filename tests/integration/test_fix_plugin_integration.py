"""
End-to-end tests for plugin fixes, run through pytester.

Subprocess runs are used where a test needs a fresh process (its own RNG seed,
its own strategy registry) or has to exit the inner session.
"""

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
