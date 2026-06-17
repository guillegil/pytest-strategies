"""
Integration tests for the --list-strategies CLI option.
"""

pytest_plugins = ["pytester"]


class TestListStrategiesIntegration:
    """Integration tests for --list-strategies."""

    def test_list_strategies_lists_registered_and_exits_clean(self, pytester):
        """--list-strategies must list registered strategies and exit without error."""
        pytester.makepyfile(strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("alpha_strat")
            def alpha(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 10)))

            @Strategy.register("beta_strat")
            def beta(nsamples):
                return Parameter(TestArg("y", rng_type=RNGInteger(0, 10)))
            """)

        # Subprocess gives a clean process: the module-level plugin singleton
        # keeps `strategies_loaded`/`Strategy._registry` across in-process runs,
        # which would skip auto-discovery and leak the outer session's registry.
        result = pytester.runpytest_subprocess("--list-strategies")

        # Must not crash (the import bug raised INTERNALERROR / non-zero).
        assert result.ret == 0
        result.stdout.fnmatch_lines(["*alpha_strat*"])
        result.stdout.fnmatch_lines(["*beta_strat*"])
