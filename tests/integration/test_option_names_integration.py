"""
End-to-end tests for the plugin's option names (D20): pytest --help lists the
command-line options in the pytest-strategies group, each one of the six names
kept from 3.0 or a --strategy-<x> option read as config.option.strategy_<x>; the
plugin adds no option outside that group, and its ini options are string options
named strategies_<x>; -o strategies_ids=values takes effect, on pytest-xdist
workers too, and a bad strategies_ids value stops the run with exit code 4,
listing the valid ones.

What the plugin adds is read as the difference between --help with the plugin and
--help without it (-p no:pytest_strategy), whatever other plugins are installed.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import re

import pytest

pytest_plugins = ["pytester"]

# The 3.0 options that kept their names; every other option is --strategy-<x>
KEPT = {
    "--rng-seed",
    "--nsamples",
    "--vector-mode",
    "--vector-name",
    "--vector-index",
    "--list-strategies",
}
GROUP = "Pytest Strategies Plugin Options:"
# The plugin's name ([project.entry-points.pytest11] in pyproject.toml)
NO_PLUGIN = ("-p", "no:pytest_strategy")
# The first help line of an ini option: "  strategies_ids (string):"
INI_OPTION = re.compile(r"  (\w+) \((\w+)\):")

ON_TESTS = """
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    @register("on_burst")
    def on_burst():
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 4095)),
            directed_vectors={"zeros": (0,)},
            nsamples=2,
        )

    @strategy("on_burst")
    def test_on_burst(addr):
        pass
"""


def _help(pytester, *args):
    """Return the lines of ``pytest --help``."""
    result = pytester.runpytest("-p", "no:cacheprovider", "--help", *args)
    assert result.ret == pytest.ExitCode.OK, result.stderr.str()
    return result.outlines


def _options(lines):
    """
    Return the options of the help lines that start an option's entry: two spaces,
    then the option strings (``-n numprocesses, --numprocesses=numprocesses``),
    while the help text goes on in a column of its own.
    """
    options = []
    for line in lines:
        if line.startswith("  -"):
            column = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
            for part in column.split(", "):
                match = re.match(r"-[^\s=]+", part)
                assert match is not None, line
                options.append(match.group())
    return options


def _group(lines):
    """Return the lines of the pytest-strategies group, from its title to the blank line."""
    start = lines.index(GROUP) + 1
    return lines[start : lines.index("", start)]


def _ini_options(lines):
    """Return the ini options --help lists, as {name: type}."""
    return dict(match.groups() for line in lines if (match := INI_OPTION.match(line)))


class TestHelp:
    def test_the_group_lists_strategy_constraint_off(self, pytester):
        assert "--strategy-constraint-off" in _options(_group(_help(pytester)))

    def test_every_option_of_the_group_is_a_kept_name_or_a_strategy_option(self, pytester):
        options = _options(_group(_help(pytester)))

        assert set(options) >= KEPT
        assert [o for o in options if o not in KEPT and not o.startswith("--strategy-")] == []

    def test_the_plugin_adds_options_only_to_its_group(self, pytester):
        with_plugin = _help(pytester)
        without_plugin = _help(pytester, *NO_PLUGIN)

        assert GROUP not in without_plugin
        added = set(_options(with_plugin)) - set(_options(without_plugin))
        assert added == set(_options(_group(with_plugin)))

    def test_each_option_is_read_under_its_own_name(self, pytester):
        """--strategy-<x> is config.option.strategy_<x>, as --rng-seed is rng_seed."""
        options = _options(_group(_help(pytester)))

        dests = vars(pytester.parseconfig().option)

        assert [o for o in options if o[2:].replace("-", "_") not in dests] == []

    def test_every_ini_option_of_the_plugin_is_a_strategies_string_option(self, pytester):
        with_plugin = _ini_options(_help(pytester))
        without_plugin = _ini_options(_help(pytester, *NO_PLUGIN))

        added = {name: kind for name, kind in with_plugin.items() if name not in without_plugin}

        assert {"strategies_ids", "strategies_max_exhaustive"} <= added.keys()
        assert [name for name in added if not name.startswith("strategies_")] == []
        assert set(added.values()) == {"string"}

    def test_an_abbreviated_option_is_not_accepted(self, pytester):
        """pytest turns argparse's abbreviations off, so no prefix of an option works."""
        result = pytester.runpytest("-p", "no:cacheprovider", "--strategy-constraint=x")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*unrecognized arguments: --strategy-constraint=x*"])

    def test_help_is_shown_with_a_bad_strategies_ids(self, pytester):
        """As for pytest's own ini options, --help does not check the value."""
        lines = _help(pytester, "-o", "strategies_ids=bad")

        assert "--strategy-constraint-off" in _options(_group(lines))


@pytest.fixture
def on_project(pytester):
    pytester.makepyfile(test_on_burst=ON_TESTS)
    return pytester


def _row_ids(result):
    """Return the IDs of the test_on_burst rows the run's output names."""
    return set(re.findall(r"::test_on_burst\[([^\]]*)\]", result.stdout.str()))


class TestStrategiesIds:
    def test_o_values_gives_the_values_ids(self, on_project):
        args = ("-p", "no:cacheprovider", "--rng-seed=1", "--collect-only", "-q")

        names = _row_ids(on_project.runpytest(*args))
        values = _row_ids(on_project.runpytest(*args, "-o", "strategies_ids=values"))

        assert names == {"directed-zeros", "rand-0", "rand-1"}
        assert len(values) == 3
        assert "addr=0" in values
        assert all(re.fullmatch(r"addr=\d+(_\d+)?", row_id) for row_id in values)

    def test_o_values_takes_effect_on_the_xdist_workers(self, on_project):
        pytest.importorskip("xdist")
        args = ("-p", "no:cacheprovider", "--rng-seed=1", "-o", "strategies_ids=values")
        collected = _row_ids(on_project.runpytest(*args, "--collect-only", "-q"))

        result = on_project.runpytest_subprocess(*args, "-n", "2", "-rA")

        result.assert_outcomes(passed=3)
        assert _row_ids(result) == collected

    def test_a_bad_value_under_xdist_stops_the_run_listing_the_valid_ones(self, on_project):
        pytest.importorskip("xdist")

        result = on_project.runpytest_subprocess(
            "-p", "no:cacheprovider", "-n", "2", "-o", "strategies_ids=value"
        )

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            ["ERROR: strategies_ids must be 'names' or 'values', got 'value'"]
        )
        assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()

    @pytest.mark.skipif(pytest.version_tuple < (9,), reason="pytest.toml needs pytest 9")
    @pytest.mark.parametrize(
        ("value", "kind"), [("1", "int"), ("true", "bool"), ('["names"]', "list")]
    )
    def test_a_toml_value_that_is_not_a_string_stops_the_run(self, on_project, value, kind):
        """pytest 9 refuses to read it as a string; the plugin turns that into its message."""
        on_project.makefile(".toml", pytest=f"[pytest]\nstrategies_ids = {value}\n")

        result = on_project.runpytest("-p", "no:cacheprovider")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [f"ERROR: strategies_ids must be 'names' or 'values'; *pytest.toml*got {kind}*"]
        )
        assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
