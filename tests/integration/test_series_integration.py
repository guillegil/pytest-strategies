"""
Integration tests for the Series deterministic sequence type.
"""

pytest_plugins = ["pytester"]


class TestSeriesIntegration:
    """Integration tests for Series via pytester. Covers S3, S8."""

    def test_series_auto_ordered_via_pytester(self, pytester):
        """S3 end-to-end: --nsamples=auto with Series -> ordered parametrize."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("series_ordered_strat")
            def strat(nsamples):
                return Parameter(TestArg("x", rng_type=Series([10, 20, 30])))

            @Strategy.strategy("series_ordered_strat")
            def test_ordered(x):
                assert x in [10, 20, 30]
        """)
        result = pytester.runpytest("--nsamples=auto", "-v")
        result.assert_outcomes(passed=3)
        # Verify order: 10 appears before 20, 20 before 30
        stdout = result.stdout.str()
        assert "10" in stdout
        assert "20" in stdout
        assert "30" in stdout
        # Find positions to verify ordering
        idx_10 = stdout.find("10]") if "10]" in stdout else stdout.find("x=10")
        idx_20 = stdout.find("20]") if "20]" in stdout else stdout.find("x=20")
        idx_30 = stdout.find("30]") if "30]" in stdout else stdout.find("x=30")
        assert idx_10 < idx_20 < idx_30

    def test_series_cycle_finite_via_pytester(self, pytester):
        """S8 end-to-end: --nsamples=10 with Series([0,1,2]) -> cycling, 10 tests pass."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("series_cycle_strat")
            def strat(nsamples):
                return Parameter(TestArg("x", rng_type=Series([0, 1, 2])))

            @Strategy.strategy("series_cycle_strat")
            def test_cycle(x):
                assert x in [0, 1, 2]
        """)
        result = pytester.runpytest("--nsamples=10", "-v")
        result.assert_outcomes(passed=10)

    def test_series_two_arg_auto_cartesian_via_pytester(self, pytester):
        """S4 end-to-end: --nsamples=auto two Series -> 6 tests (2x3 product)."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("series_cartesian_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("a", rng_type=Series([1, 2])),
                    TestArg("b", rng_type=Series(["x", "y", "z"])),
                )

            @Strategy.strategy("series_cartesian_strat")
            def test_cartesian(a, b):
                assert a in [1, 2]
                assert b in ["x", "y", "z"]
        """)
        result = pytester.runpytest("--nsamples=auto", "-v")
        result.assert_outcomes(passed=6)
