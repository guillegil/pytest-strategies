"""
Integration tests for Sequence Testing feature.
"""

import pytest
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger
from pytest_strategy.rng import RNGSequence

pytest_plugins = ["pytester"]

# Register strategies for testing
@Strategy.register("sequence_strategy")
def create_sequence_strategy(nsamples):
    return Parameter(
        TestArg("a", rng_type=RNGSequence([1, 2])),
        TestArg("b", rng_type=RNGSequence(["x", "y"]))
    )

@Strategy.register("mixed_sequence_strategy")
def create_mixed_strategy(nsamples):
    return Parameter(
        TestArg("seq", rng_type=RNGSequence([1, 2])),
        TestArg("rnd", rng_type=RNGInteger(10, 20))
    )

@Strategy.register("no_sequence_strategy")
def create_no_sequence_strategy(nsamples):
    return Parameter(
        TestArg("rnd", rng_type=RNGInteger(0, 10))
    )

class TestSequenceIntegration:
    """Integration tests for nsamples='auto'."""

    @Strategy.strategy("sequence_strategy")
    def test_auto_mode(self, a, b):
        """Test that auto mode generates all combinations."""
        # This test runs multiple times, once for each combination.
        # We can't easily assert the total count here without a side effect,
        # but we can verify the values are valid.
        assert a in [1, 2]
        assert b in ["x", "y"]

    def test_auto_mode_generation_count(self, pytester):
        """Verify the number of tests generated in auto mode."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg
            from pytest_strategy.rng import RNGSequence

            @Strategy.register("seq_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("a", rng_type=RNGSequence([1, 2])),
                    TestArg("b", rng_type=RNGSequence(["x", "y"]))
                )

            @Strategy.strategy("seq_strat")
            def test_gen(a, b):
                pass
        """)
        
        # Run with nsamples="auto"
        result = pytester.runpytest("--nsamples=auto")
        
        # Should have 4 tests (2 * 2)
        result.assert_outcomes(passed=4)

    def test_auto_mode_mixed_generation_count(self, pytester):
        """Verify the number of tests generated in mixed auto mode."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger
            from pytest_strategy.rng import RNGSequence

            @Strategy.register("mixed_strat")
            def strat(nsamples):
                return Parameter(
                    TestArg("seq", rng_type=RNGSequence([1, 2, 3])),
                    TestArg("rnd", rng_type=RNGInteger(0, 10))
                )

            @Strategy.strategy("mixed_strat")
            def test_gen(seq, rnd):
                pass
        """)
        
        # Run with nsamples="auto"
        result = pytester.runpytest("--nsamples=auto")
        
        # Should have 3 tests (length of sequence)
        result.assert_outcomes(passed=3)

    def test_auto_mode_no_sequence_fails(self, pytester):
        """Verify that auto mode fails if no sequence is present."""
        pytester.makepyfile("""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("no_seq")
            def strat(nsamples):
                return Parameter(
                    TestArg("rnd", rng_type=RNGInteger(0, 10))
                )

            @Strategy.strategy("no_seq")
            def test_gen(rnd):
                pass
        """)
        
        # Run with nsamples="auto"
        result = pytester.runpytest("--nsamples=auto")
        
        # Should fail collection or execution
        result.stdout.fnmatch_lines(["*ValueError: No sequence arguments found*"])
