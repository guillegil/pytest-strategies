"""
End-to-end tests for the "Strategy not found" error naming the strategy files that
were not imported or failed to load, run through pytester.

Subprocess runs give each test a fresh process (its own strategy registry and
sys.path). Strategy names are unique to this module.
"""

import pytest

pytest_plugins = ["pytester"]

HINT = (
    "*Files matching a strategy file name that were not imported because "
    "they contain no '@Strategy.register' (use that decorator form):"
)

DECORATOR_REGISTRATION = """
from pytest_strategy import Strategy

@Strategy.register("ns_strat")
def strat(nsamples):
    return ("x",), [(1,)]
"""

ALIAS_REGISTRATION = """
from pytest_strategy import Strategy as S

@S.register("ns_strat")
def strat(nsamples):
    return ("x",), [(1,)]
"""

CALL_REGISTRATION = """
from pytest_strategy import Strategy

def strat(nsamples):
    return ("x",), [(1,)]

Strategy.register("ns_strat")(strat)
"""

TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("ns_strat")
def test_strat(x):
    pass
"""


class TestUnimportedFileIsListed:
    """A strategy file that registers without '@Strategy.register' is named when a lookup fails."""

    @pytest.mark.parametrize(
        "source", [ALIAS_REGISTRATION, CALL_REGISTRATION], ids=["alias", "plain-call"]
    )
    def test_file_is_listed_with_the_hint(self, pytester, source):
        pytester.makepyfile(alias_strategies=source, test_strat=TESTS)

        result = pytester.runpytest_subprocess()

        result.stdout.fnmatch_lines(
            [
                "*ValueError: Strategy 'ns_strat' not found. Available strategies: none",
                HINT,
                f"*  {pytester.path / 'alias_strategies.py'}",
            ]
        )
        assert result.ret == pytest.ExitCode.INTERRUPTED

    def test_file_with_the_decorator_is_not_listed(self, pytester):
        pytester.makepyfile(
            strategies=DECORATOR_REGISTRATION,
            alias_strategies=ALIAS_REGISTRATION.replace('"ns_strat"', '"ns_other"'),
            test_strat=TESTS.replace('"ns_strat"', '"ns_other"'),
        )

        result = pytester.runpytest_subprocess()

        output = result.stdout.str()
        # Not fnmatch: the brackets would be a character class
        assert "Strategy 'ns_other' not found. Available strategies: ['ns_strat']" in output
        result.stdout.fnmatch_lines([HINT, f"*  {pytester.path / 'alias_strategies.py'}"])
        assert str(pytester.path / "strategies.py") not in output

    def test_file_is_not_reported_when_every_lookup_succeeds(self, pytester):
        # An unrelated module that happens to have a strategy file name
        pytester.makepyfile(
            order_strategy="def place_order():\n    pass\n",
            strategies=DECORATOR_REGISTRATION,
            test_strat=TESTS,
        )

        result = pytester.runpytest_subprocess("-vv")

        result.assert_outcomes(passed=1)
        assert "order_strategy" not in result.stdout.str() + result.stderr.str()


class TestExcludedFileIsNotListed:
    """Files in directories that discovery skips are not listed either."""

    @pytest.mark.parametrize("directory", [".hidden", "build", "myenv", "vendored"])
    def test_file_in_skipped_directory_is_not_listed(self, pytester, directory):
        pytester.makeini("[pytest]\nnorecursedirs = .* build vendored\n")
        pytester.makepyfile(**{f"{directory}/alias_strategies": ALIAS_REGISTRATION})
        if directory == "myenv":
            # A virtual environment, whatever its name
            (pytester.path / directory / "pyvenv.cfg").write_text("home = /usr/bin\n")
        pytester.makepyfile(test_strat=TESTS)

        result = pytester.runpytest_subprocess()

        result.stdout.fnmatch_lines(
            ["*ValueError: Strategy 'ns_strat' not found. Available strategies: none"]
        )
        assert "alias_strategies.py" not in result.stdout.str()
        assert "were not imported" not in result.stdout.str()


class TestFailedFileIsListed:
    """A strategy file importing what only becomes importable during collection is listed."""

    def test_file_is_listed_with_the_session_start_note(self, pytester):
        pytester.makepyfile(
            **{
                # Importable from sub/ only once pytest puts sub/ on sys.path to
                # import the test module next to it
                "sub/ns_late_helpers": "LIMIT = 5\n",
                "sub/strategies": """
                    from ns_late_helpers import LIMIT

                    from pytest_strategy import Strategy

                    @Strategy.register("ns_late")
                    def late(nsamples):
                        return ("x",), [(LIMIT,)]
                    """,
                "sub/test_late": """
                    from ns_late_helpers import LIMIT

                    from pytest_strategy import Strategy

                    @Strategy.strategy("ns_late")
                    def test_late(x):
                        assert x == LIMIT
                    """,
            }
        )

        result = pytester.runpytest_subprocess()

        result.stdout.fnmatch_lines(
            [
                "*ValueError: Strategy 'ns_late' not found. Available strategies: none",
                "*Strategy files that failed to load:",
                "*strategies.py: ModuleNotFoundError: No module named 'ns_late_helpers'",
                "*Strategy files are imported when the test session starts, before test modules "
                "are collected, so an import that only works later fails.",
            ]
        )
        assert result.ret == pytest.ExitCode.INTERRUPTED
