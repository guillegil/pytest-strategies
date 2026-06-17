"""
Integration tests proving per-session isolation of plugin runtime state.

Before the runtime refactor, the auto-discovery flag lived on a module-level
plugin singleton and never reset, so an in-process pytest session nested inside
another (the common ``pytester`` case) skipped discovery entirely and leaked the
outer session's registry. The runtime now keeps a stack of per-session state, so
each nested session discovers its own strategy files.

Distinct strategy filenames are used per run on purpose: reusing one filename
would collide in CPython's import / bytecode caches (an artifact unrelated to the
plugin's own state), which is not what these tests exercise.
"""

pytest_plugins = ["pytester"]


class TestSessionIsolation:
    """Nested in-process sessions must each run their own discovery."""

    def test_inprocess_session_discovers_its_own_strategy(self, pytester):
        """A run nested in the outer suite must discover its own strategy file.

        Under the old global flag this failed: the outer session had already set
        ``strategies_loaded`` and the inner run skipped discovery, so the
        strategy was never registered.
        """
        pytester.makepyfile(alpha_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("alpha_strat")
            def a(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 5)))
            """)
        pytester.makepyfile(test_alpha="""
            from pytest_strategy import Strategy

            @Strategy.strategy("alpha_strat")
            def test_alpha(x):
                assert 0 <= x <= 5
            """)
        pytester.runpytest_inprocess("--nsamples=2").assert_outcomes(passed=2)

    def test_second_inprocess_run_is_not_blocked_by_the_first(self, pytester):
        """A second nested run must also discover its (distinct) strategy file."""
        pytester.makepyfile(beta_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("beta_strat")
            def b(nsamples):
                return Parameter(TestArg("y", rng_type=RNGInteger(0, 5)))
            """)
        pytester.makepyfile(test_beta="""
            from pytest_strategy import Strategy

            @Strategy.strategy("beta_strat")
            def test_beta(y):
                assert 0 <= y <= 5
            """)
        # Run twice in-process within the same outer session; the second run
        # must not be blocked by leftover discovery state from the first.
        pytester.runpytest_inprocess("--nsamples=2").assert_outcomes(passed=2)
        pytester.runpytest_inprocess("--nsamples=3").assert_outcomes(passed=3)
