"""
Integration tests for test_values feature with --vector-mode=test.
"""

import pytest

pytest_plugins = ["pytester"]


class TestTestModeIntegration:
    """Integration tests for --vector-mode=test."""

    def test_test_mode_runs_only_test_vectors(self, pytester):
        """Verify that test mode runs only test vectors."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("test_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 100)),
                    TestArg("y", rng_type=RNGInteger(0, 100)),
                    directed_vectors={"dir1": (1, 1)},
                    test_vectors={
                        "test1": (10, 10),
                        "test2": (20, 20)
                    }
                )

            @Strategy.strategy("test_strat")
            def test_gen(x, y):
                # In test mode, should only see (10, 10) and (20, 20)
                assert (x, y) in [(10, 10), (20, 20)]
        """)
        
        # Run with --vector-mode=test
        result = pytester.runpytest("--vector-mode=test")
        
        # Should have 2 tests (only test vectors)
        result.assert_outcomes(passed=2)

    def test_test_mode_ignores_directed_and_random(self, pytester):
        """Verify that test mode ignores directed vectors and random samples."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("mixed_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("val", rng_type=RNGInteger(0, 5)),
                    directed_vectors={"dir1": (1,), "dir2": (2,)},
                    test_vectors={"test1": (99,)}
                )

            @Strategy.strategy("mixed_strat")
            def test_gen(val):
                # In test mode, should only see 99
                assert val == 99
        """)
        
        # Run with --vector-mode=test and nsamples=10
        result = pytester.runpytest("--vector-mode=test", "--nsamples=10")
        
        # Should have 1 test (only test vector, ignoring directed and random)
        result.assert_outcomes(passed=1)

    def test_test_mode_with_no_test_vectors(self, pytester):
        """Verify that test mode with no test vectors runs no tests."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("no_test_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 10)),
                    directed_vectors={"dir1": (5,)}
                )

            @Strategy.strategy("no_test_strat")
            def test_gen(x):
                pass
        """)
        
        # Run with --vector-mode=test
        result = pytester.runpytest("--vector-mode=test")
        
        # Should have 0 passed tests and 1 skipped (no test vectors defined)
        result.assert_outcomes(skipped=1)

