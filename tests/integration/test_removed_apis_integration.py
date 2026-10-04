"""
End-to-end tests for the APIs that 3.0 deprecated and 4.0 removes, run through
pytester: a factory that returns an (argnames, samples) tuple, or anything else
that is not a Parameter, fails the collection of the tests that use it.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import os

import pytest

pytest_plugins = ["pytester"]


class TestFactoryResults:
    def test_tuple_factory_fails_collection(self, pytester):
        path = pytester.makepyfile(test_rm_tuple="""
            from pytest_strategy import register, strategy

            @register("rm_pairs")
            def pairs(nsamples):
                return ("x", "y"), [(1, 2), (3, 4)]

            @strategy("rm_pairs")
            def test_pairs(x, y):
                pass
            """)
        result = pytester.runpytest("-W", "error::DeprecationWarning")

        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.assert_outcomes(errors=1)
        factory = f"{os.path.realpath(path)}:3:pairs"
        result.stdout.fnmatch_lines(
            [
                "In test_pairs: Strategy 'rm_pairs' returned an (argnames, samples) tuple "
                f"(factory: {factory}). Returning a tuple was deprecated in 3.0 and is "
                "no longer supported in 4.0: return a Parameter, *",
            ]
        )
        assert "warnings summary" not in result.stdout.str()

    def test_factory_returning_none_fails_collection(self, pytester):
        pytester.makepyfile(test_rm_none="""
            from pytest_strategy import register, strategy

            @register("rm_nothing")
            def nothing(nsamples):
                rows = [(1,), (2,)]

            @strategy("rm_nothing")
            def test_nothing(x):
                pass
            """)
        result = pytester.runpytest()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "In test_nothing: Strategy 'rm_nothing' must return a Parameter, got NoneType "
                "(did the factory forget to return?)",
            ]
        )

    def test_only_the_tests_of_that_strategy_fail(self, pytester):
        pytester.makepyfile(
            test_rm_other="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @register("rm_fine")
            def fine(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=2)

            @strategy("rm_fine")
            def test_fine(x):
                pass
            """,
            test_rm_list="""
            from pytest_strategy import register, strategy

            @register("rm_list")
            def rows(nsamples):
                return [("x",), [(1,)]]

            @strategy("rm_list")
            def test_rows(x):
                pass
            """,
        )
        result = pytester.runpytest("--continue-on-collection-errors")

        result.assert_outcomes(passed=2, errors=1)
        result.stdout.fnmatch_lines(
            ["In test_rows: Strategy 'rm_list' must return a Parameter, got list"]
        )
