"""
Integration tests for per-strategy nsamples override feature.

These tests use pytester to spin up isolated in-process pytest sessions and
verify the full end-to-end behavior of the nsamples precedence rules.

Distinct strategy filenames are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

pytest_plugins = ["pytester"]


class TestNsamplesPerStrategyIntegration:
    """End-to-end vector count per scenario via real pytest sessions."""

    def test_strategy_nsamples_used_when_cli_absent(self, pytester):
        """Scenario 2: no CLI --nsamples + strategy nsamples=15 → 15 test cases."""
        pytester.makepyfile(ns_a_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("ns_strat_a")
            def factory(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)), nsamples=15)
            """)
        pytester.makepyfile(test_ns_a="""
            from pytest_strategy import Strategy

            @Strategy.strategy("ns_strat_a")
            def test_ns_a(x):
                assert isinstance(x, int)
            """)
        # No --nsamples passed; strategy declares nsamples=15
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=15)

    def test_fallback_10_when_neither_set(self, pytester):
        """Scenario 3: no CLI --nsamples + no param nsamples → 10 test cases (fallback)."""
        pytester.makepyfile(ns_b_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("ns_strat_b")
            def factory(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)))
            """)
        pytester.makepyfile(test_ns_b="""
            from pytest_strategy import Strategy

            @Strategy.strategy("ns_strat_b")
            def test_ns_b(x):
                assert isinstance(x, int)
            """)
        # No --nsamples, no param nsamples → fallback 10
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=10)

    def test_cli_overrides_strategy_nsamples(self, pytester):
        """Scenario 4: --nsamples=5 overrides strategy nsamples=15 → 5 test cases."""
        pytester.makepyfile(ns_c_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("ns_strat_c")
            def factory(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)), nsamples=15)
            """)
        pytester.makepyfile(test_ns_c="""
            from pytest_strategy import Strategy

            @Strategy.strategy("ns_strat_c")
            def test_ns_c(x):
                assert isinstance(x, int)
            """)
        result = pytester.runpytest_inprocess("--nsamples=5")
        result.assert_outcomes(passed=5)

    def test_auto_wins_over_strategy_nsamples(self, pytester):
        """Scenario 5: --nsamples=auto + strategy nsamples=15 → exhaustive (not 15)."""
        pytester.makepyfile(ns_d_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger
            from pytest_strategy.rng import RNGSequence

            @Strategy.register("ns_strat_d")
            def factory(nsamples):
                seq = RNGSequence([10, 20, 30])
                return Parameter(TestArg("x", rng_type=seq), nsamples=15)
            """)
        pytester.makepyfile(test_ns_d="""
            from pytest_strategy import Strategy

            @Strategy.strategy("ns_strat_d")
            def test_ns_d(x):
                assert x in [10, 20, 30]
            """)
        # auto → exhaustive; 3-element sequence → 3 test cases (not 15)
        result = pytester.runpytest_inprocess("--nsamples=auto")
        result.assert_outcomes(passed=3)

    def test_legacy_path_no_typeerror_when_cli_absent(self, pytester):
        """Scenario 6: legacy tuple factory + no CLI --nsamples → no TypeError, 10 cases."""
        pytester.makepyfile(ns_e_strategies="""
            from pytest_strategy import Strategy

            @Strategy.register("ns_strat_e")
            def factory(nsamples):
                # Legacy path: return a tuple (argnames, samples)
                samples = [(i,) for i in range(nsamples)]
                return ("x", samples)
            """)
        pytester.makepyfile(test_ns_e="""
            from pytest_strategy import Strategy

            @Strategy.strategy("ns_strat_e")
            def test_ns_e(x):
                assert isinstance(x, int)
            """)
        # No --nsamples; legacy path must receive 10 (int), not None
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=10)
