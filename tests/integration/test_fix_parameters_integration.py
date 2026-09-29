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


class TestExhaustiveRandomConstraint:
    """--nsamples=auto covers every Series value when the constraint is on a random arg."""

    def test_auto_keeps_every_role_for_every_seed(self, pytester):
        pytester.makepyfile(fixexh_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, Series, RNGInteger

            @Strategy.register("fixexh_roles")
            def factory(nsamples):
                return Parameter(
                    TestArg("role", rng_type=Series(["admin", "user", "guest"])),
                    TestArg("uid", rng_type=RNGInteger(1, 1000)),
                    vector_constraints=[lambda v: v[1] > 500],
                )
            """)
        pytester.makepyfile(test_fixexh="""
            from pytest_strategy import Strategy

            @Strategy.strategy("fixexh_roles")
            def test_roles(role, uid):
                assert uid > 500
            """)
        # Seed 4 used to produce an empty parameter set (0 tests)
        for seed in range(6):
            result = pytester.runpytest_inprocess("--nsamples=auto", f"--rng-seed={seed}")
            result.assert_outcomes(passed=3)


class TestValidatorOnSeriesValues:
    """An invalid Series value is reported at collection instead of reaching the test."""

    def _make_ports(self, pytester, prefix):
        pytester.makepyfile(**{f"{prefix}_strategies": f"""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("{prefix}_ports")
            def factory(nsamples):
                return Parameter(
                    TestArg(
                        "port",
                        rng_type=Series([80, 443, -1]),
                        validator=lambda p: 0 < p < 65536,
                    ),
                )
            """})
        pytester.makepyfile(**{f"test_{prefix}": f"""
            from pytest_strategy import Strategy

            @Strategy.strategy("{prefix}_ports")
            def test_port(port):
                assert 0 < port < 65536
            """})

    def test_finite_mode_reports_invalid_value(self, pytester):
        self._make_ports(pytester, "fixval_a")
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*Value -1 failed validation for argument 'port'*"])

    def test_auto_mode_reports_invalid_value(self, pytester):
        self._make_ports(pytester, "fixval_b")
        result = pytester.runpytest_inprocess("--nsamples=auto")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*Value -1 failed validation for argument 'port'*"])
