"""
End-to-end regression tests for Parameter bug fixes.

These tests use pytester to run isolated in-process pytest sessions.
Distinct strategy filenames are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

pytest_plugins = ["pytester"]


class TestSeriesConstraintsFiniteMode:
    """A Series strategy with a vector constraint collects in finite mode too."""

    def _make_ordered_pairs(self, pytester, prefix):
        pytester.makepyfile(**{f"{prefix}_strategies": f"""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("{prefix}_pairs")
            def factory(nsamples):
                return Parameter(
                    TestArg("lo", rng_type=Series([1, 2, 3])),
                    TestArg("hi", rng_type=Series([1, 2, 3])),
                    vector_constraints=[lambda v: v[0] < v[1]],
                )
            """})
        pytester.makepyfile(**{f"test_{prefix}": f"""
            from pytest_strategy import Strategy

            @Strategy.strategy("{prefix}_pairs")
            def test_pairs(lo, hi):
                assert lo < hi
            """})

    def test_default_nsamples_skips_rejected_combinations(self, pytester):
        """Default run (nsamples 10) used to fail collection on the first rejected row."""
        self._make_ordered_pairs(pytester, "fixser_a")
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=10)

    def test_small_nsamples_takes_first_valid_rows(self, pytester):
        self._make_ordered_pairs(pytester, "fixser_b")
        result = pytester.runpytest_inprocess("--nsamples=2", "--collect-only", "-q")
        result.stdout.fnmatch_lines(["*test_pairs[[]lo=1,hi=2[]]*", "*test_pairs[[]lo=1,hi=3[]]*"])
        result.stdout.no_fnmatch_line("*lo=2,hi=3*")

    def test_auto_still_filters(self, pytester):
        self._make_ordered_pairs(pytester, "fixser_c")
        result = pytester.runpytest_inprocess("--nsamples=auto")
        result.assert_outcomes(passed=3)
