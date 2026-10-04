"""
Integration tests for the --list-strategies CLI option.
"""

import pytest

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


# A strategy in two folders, and one whose factory raises; no collected test uses
# lst_broken. Written for 3.0's API too (Strategy.register, factories taking
# nsamples), so the listing could be compared with 3.0.0's.
LISTED_STRATEGIES = {
    "tests/a/list_strategies.py": """
from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg


@Strategy.register("lst_shared")
def shared_a(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))


@Strategy.register("lst_alpha")
def alpha(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))


@Strategy.register("lst_broken")
def broken(nsamples):
    raise RuntimeError("boom")
""",
    "tests/b/list_strategies.py": """
from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg


@Strategy.register("lst_shared")
def shared_b(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
""",
    "tests/a/test_lst.py": """
from pytest_strategy import Strategy


@Strategy.strategy("lst_shared")
def test_shared(x):
    pass
""",
}


def _write(pytester, files):
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class TestTheListing:
    """--list-strategies keeps 3.0's text and calls no factory (D19)."""

    @pytest.fixture(autouse=True)
    def utf8_output(self, monkeypatch):
        # Windows pipes default to cp1252, where pytest writes the check mark as \u2713
        monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    def test_the_text_is_3_0_s(self, pytester):
        pytester.makeini("[pytest]\n")
        _write(pytester, LISTED_STRATEGIES)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--list-strategies")

        # The raising factory, which no collected test uses, is listed: the listing
        # does not call it
        assert result.ret == pytest.ExitCode.OK
        lines = result.stdout.lines
        start = next(i for i, line in enumerate(lines) if "Registered Strategies" in line)
        # pytest-strategies 3.0.0 printed these lines for this project
        assert lines[start + 1 : start + 9] == [
            "",
            "Found 4 registered strategies:",
            "",
            "  ✓ lst_alpha",
            "  ✓ lst_broken",
            "  ✓ lst_shared (tests/a/list_strategies.py)",
            "  ✓ lst_shared (tests/b/list_strategies.py)",
            "",
        ]
        result.stdout.no_fnmatch_line("*boom*")

    def test_the_listing_calls_no_factory(self, pytester):
        pytester.makeini("[pytest]\n")
        _write(pytester, LISTED_STRATEGIES)
        # A factory that no test uses, and that writes when it is called
        (pytester.path / "tests/b/list_strategies.py").write_text(
            "from pathlib import Path\n"
            "from pytest_strategy import register\n\n"
            '@register("lst_writes")\n'
            "def writes():\n"
            '    Path(__file__).with_name("called.txt").write_text("called")\n',
            encoding="utf-8",
        )

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--list-strategies")

        assert result.ret == pytest.ExitCode.OK
        result.stdout.fnmatch_lines(["  ✓ lst_writes"])
        assert not (pytester.path / "tests/b/called.txt").exists()

    def test_a_path_after_the_option_is_a_path(self, pytester):
        """
        --list-strategies takes no value: in pytest --list-strategies tests/, tests/
        is what pytest collects, not the testpaths (other/).
        """
        pytester.makeini("[pytest]\ntestpaths = other\n")
        _write(
            pytester,
            {
                "other/test_other.py": "def test_other():\n    pass\n",
                "tests/test_listed.py": (
                    "from pytest_strategy import Parameter, TestArg, register\n\n"
                    '@register("lst_in_a_test_module")\n'
                    "def in_test_module():\n"
                    '    return Parameter(TestArg("x", value=1))\n\n'
                    "def test_listed():\n"
                    "    pass\n"
                ),
            },
        )

        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", "--list-strategies", "tests/"
        )

        # Registered by a test module, which pytest imports only when it collects it
        assert result.ret == pytest.ExitCode.OK
        result.stdout.fnmatch_lines(["collected 1 item", "*Found 1 registered strategies:*"])
        result.stdout.fnmatch_lines(["  ✓ lst_in_a_test_module"])
