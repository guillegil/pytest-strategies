"""
End-to-end tests for the keyword-only options and the rng_type check of 4.0, run
through pytester: what a user sees for a positional option, a raw strategy marker
with a positional flag, and an rng_type that is a bare lambda.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import pytest

pytest_plugins = ["pytester"]


class TestMarker:
    def test_marker_help_shows_the_keyword_only_flag(self, pytester):
        result = pytester.runpytest("--markers")

        assert result.ret == pytest.ExitCode.OK
        result.stdout.fnmatch_lines(
            [
                "@pytest.mark.strategy(name_or_factory, [*], validate_signature=True): "
                "parametrize the test with a strategy (added by @strategy)",
            ]
        )

    def test_raw_marker_with_a_positional_flag_fails_collection(self, pytester):
        pytester.makepyfile(test_ko_raw_marker="""
            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("ko_raw")
            def ko_raw(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

            @pytest.mark.strategy("ko_raw", False)
            def test_raw(x):
                pass
            """)
        result = pytester.runpytest()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "In test_raw: invalid strategy marker ('ko_raw', False) {}; "
                "use @strategy(name_or_factory, [*], validate_signature=True)",
            ]
        )

    def test_raw_marker_with_the_flag_by_keyword_collects(self, pytester):
        pytester.makepyfile(test_ko_raw_keyword="""
            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("ko_raw_kw")
            def ko_raw_kw(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

            @pytest.mark.strategy("ko_raw_kw", validate_signature=False)
            def test_raw(x):
                assert 0 <= x <= 9
            """)
        result = pytester.runpytest("--nsamples=3")

        result.assert_outcomes(passed=3)


class TestPositionalOptions:
    def test_strategy_flag_by_position_fails_the_module(self, pytester):
        pytester.makepyfile(test_ko_flag="""
            from pytest_strategy import strategy

            @strategy("ko_flag", False)
            def test_flag(x):
                pass
            """)
        result = pytester.runpytest()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            ["E   TypeError: strategy() takes 1 positional argument but 2 were given"]
        )

    def test_testarg_value_by_position_fails_collection(self, pytester):
        pytester.makepyfile(test_ko_value="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @register("ko_value")
            def ko_value(nsamples):
                return Parameter(TestArg("x", RNGInteger(0, 1), 5))

            @strategy("ko_value")
            def test_value(x):
                pass
            """)
        result = pytester.runpytest()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "In test_value: Error calling strategy factory 'ko_value' (nsamples=10): "
                "TypeError: TestArg.__init__() takes from 2 to 3 positional arguments "
                "but 4 were given",
                '*return Parameter(TestArg("x", RNGInteger(0, 1), 5))',
            ]
        )


class TestRngTypeCheck:
    def test_lambda_rng_type_fails_collection(self, pytester):
        pytester.makepyfile(test_ko_lambda="""
            from pytest_strategy import Parameter, TestArg, register, strategy

            @register("ko_lambda")
            def ko_lambda(nsamples):
                return Parameter(TestArg("x", rng_type=lambda v: v + 1))

            @strategy("ko_lambda")
            def test_lambda(x):
                pass
            """)
        result = pytester.runpytest()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "In test_lambda: Error calling strategy factory 'ko_lambda' (nsamples=10): "
                "TypeError: TestArg 'x' rng_type must be an RNGType or have a generate() "
                "method, got <function ko_lambda.<locals>.<lambda> at 0x*>",
                '*return Parameter(TestArg("x", rng_type=lambda v: v + 1))',
            ]
        )

    def test_object_with_generate_collects(self, pytester):
        pytester.makepyfile(test_ko_duck="""
            from pytest_strategy import Parameter, TestArg, register, strategy

            class Fives:
                python_type = int

                def generate(self):
                    return 5

            @register("ko_duck")
            def ko_duck(nsamples):
                return Parameter(TestArg("x", rng_type=Fives()))

            @strategy("ko_duck")
            def test_duck(x):
                assert x == 5
            """)
        result = pytester.runpytest("--nsamples=2")

        result.assert_outcomes(passed=2)
