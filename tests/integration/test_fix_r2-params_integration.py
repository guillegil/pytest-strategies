"""
End-to-end regression tests for the Parameter review fixes.

These tests use pytester to run isolated in-process pytest sessions.
Distinct strategy filenames are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import random

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what the in-process runs change globally: the registry and the RNG seed."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


class TestNsamplesAutoPassThrough:
    """A factory may pass the nsamples it receives (or "auto") on to Parameter."""

    def _make_strategies(self, pytester, prefix):
        pytester.makepyfile(**{f"{prefix}_strategies": f"""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("{prefix}_passthru")
            def passthru(nsamples):
                return Parameter(TestArg("x", rng_type=Series([1, 2, 3])), nsamples=nsamples)

            @Strategy.register("{prefix}_auto_default")
            def auto_default(nsamples):
                return Parameter(TestArg("y", rng_type=Series(["a", "b"])), nsamples="auto")
            """})
        pytester.makepyfile(**{f"test_{prefix}": f"""
            from pytest_strategy import Strategy

            @Strategy.strategy("{prefix}_passthru")
            def test_pass(x):
                assert x in (1, 2, 3)

            @Strategy.strategy("{prefix}_auto_default")
            def test_auto_default(y):
                assert y in ("a", "b")
            """})

    def test_cli_auto_forwarded_to_parameter(self, pytester):
        """--nsamples=auto used to fail collection: Parameter rejected "auto"."""
        self._make_strategies(pytester, "fixr2auto_a")
        result = pytester.runpytest_inprocess("--nsamples=auto", "--rng-seed=1")
        result.assert_outcomes(passed=5)

    def test_per_strategy_auto_default(self, pytester):
        """A strategy defaulting to "auto" used to fail every run, even without a flag."""
        self._make_strategies(pytester, "fixr2auto_b")
        result = pytester.runpytest_inprocess("--collect-only", "-q", "--rng-seed=1")
        result.stdout.fnmatch_lines(
            ["*test_auto_default[[]y='a'[]]*", "*test_auto_default[[]y='b'[]]*"]
        )
        result.stdout.fnmatch_lines(["*12 tests collected*"])


class TestSeriesSkipWarningShown:
    """A Series combination skipped after random redraws shows up in the warnings summary."""

    def test_skip_reported_in_warnings_summary(self, pytester):
        pytester.makepyfile(fixr2skip_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, Series, RNGInteger

            @Strategy.register("fixr2skip_modes")
            def factory(nsamples):
                return Parameter(
                    TestArg("mode", rng_type=Series(["a", "b", "c"])),
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    vector_constraints=[lambda v: v[0] != "a"],
                    max_retries=5,
                )
            """)
        pytester.makepyfile(test_fixr2skip="""
            from pytest_strategy import Strategy

            @Strategy.strategy("fixr2skip_modes")
            def test_modes(mode, x):
                assert mode != "a"
            """)
        # A subprocess keeps this suite's filterwarnings=error out of the inner run
        result = pytester.runpytest_subprocess("--nsamples=4", "--rng-seed=1")
        result.assert_outcomes(passed=4, warnings=1)
        result.stdout.fnmatch_lines(
            [
                "*/test_fixr2skip.py:3: PytestStrategiesWarning: Strategy 'fixr2skip_modes' "
                "(test_modes): Series combination (mode='a') skipped*max_retries=5*"
            ]
        )


class TestAutoUnsatisfiableConstraintFails:
    """--nsamples=auto used to skip the test (exit 0) when no combination was valid."""

    @pytest.mark.parametrize("nsamples", [None, "auto"])
    def test_collection_error_in_both_modes(self, pytester, nsamples):
        prefix = f"fixr2unsat_{nsamples or 'finite'}"
        pytester.makepyfile(**{f"{prefix}_strategies": f"""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("{prefix}")
            def factory(nsamples):
                return Parameter(
                    TestArg("x", rng_type=Series([1, 2, 3])),
                    TestArg("y", rng_type=Series([1, 2])),
                    vector_constraints=[lambda v: v[0] + v[1] > 100],
                )
            """})
        pytester.makepyfile(**{f"test_{prefix}": f"""
            from pytest_strategy import Strategy

            @Strategy.strategy("{prefix}")
            def test_sum(x, y):
                pass
            """})
        args = ["-rs"] + ([f"--nsamples={nsamples}"] if nsamples else [])
        result = pytester.runpytest_inprocess(*args)
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines(["*none of the 6 * combinations satisfied the vector*"])
        result.stdout.no_fnmatch_line("*empty parameter set*")
