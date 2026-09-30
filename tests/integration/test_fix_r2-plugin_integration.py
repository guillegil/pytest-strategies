"""
End-to-end tests for plugin fixes, run through pytester.

Subprocess runs are used unless a test is about in-process sessions. Strategy
names are unique to this module, so they do not clash with other tests.
"""

import random
import re

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]

STRATEGIES = """
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("r2_ints")
def r2_ints(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)
"""

TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("r2_ints")
def test_ints(x):
    assert 0 <= x <= 10**9
"""

MISSING_MODULE = "pytest_strategies_missing_module"


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what these tests change globally: the seed, random state and registry."""
    seed = RNG.get_seed()
    state = random.getstate()
    registry = dict(Strategy._registry)
    yield
    RNG._seed = seed
    random.setstate(state)
    Strategy._registry.clear()
    Strategy._registry.update(registry)


class TestGlobalRandomState:
    """Without --rng-seed the plugin leaves the global random module alone."""

    @pytest.mark.parametrize("args", [(), ("-n", "1")], ids=["plain", "xdist"])
    def test_conftest_seed_survives_an_unseeded_run(self, pytester, args):
        if args:
            pytest.importorskip("xdist")
        expected = random.Random(0).random()
        pytester.makeconftest("import random\n\nrandom.seed(0)\n")
        pytester.makepyfile(test_seeded=f"""
            import random

            def test_draw():
                assert random.random() == {expected!r}
            """)

        result = pytester.runpytest_subprocess(*args)

        result.assert_outcomes(passed=1)

    def test_xdist_workers_do_not_share_random_state(self, pytester):
        """A project without strategy files: each worker keeps its own entropy."""
        pytest.importorskip("xdist")
        pytester.makepyfile(test_ports="""
            import os
            import random
            from pathlib import Path

            import pytest

            @pytest.mark.parametrize("i", range(4))
            def test_port(i):
                worker = os.environ["PYTEST_XDIST_WORKER"]
                port = random.randint(20000, 2**62)
                Path(__file__).with_name(f"{worker}-{i}.port").write_text(str(port))
            """)

        result = pytester.runpytest_subprocess("-n", "2")

        result.assert_outcomes(passed=4)
        ports = {}
        for path in pytester.path.glob("gw*-*.port"):
            ports.setdefault(path.name.split("-")[0], set()).add(path.read_text())
        assert sorted(ports) == ["gw0", "gw1"]
        assert not ports["gw0"] & ports["gw1"]


class TestNestedSessionGlobalState:
    """An in-process session leaves no random state or registrations behind."""

    def test_inner_runs_do_not_restart_the_outer_random_stream(self, pytester):
        pytester.makepyfile(r2_nested_strategies=STRATEGIES)
        pytester.makepyfile(test_nested=TESTS)
        seed = RNG.get_seed()
        random.seed(123)
        expected = [random.random() for _ in range(3)]

        random.seed(123)
        got = [random.random()]
        pytester.runpytest_inprocess().assert_outcomes(passed=3)
        got.append(random.random())
        pytester.runpytest_inprocess("--rng-seed=5").assert_outcomes(passed=3)
        got.append(random.random())

        assert got == expected
        assert RNG.get_seed() == seed
        assert "r2_ints" not in Strategy._registry

    def test_sibling_sessions_do_not_warn_about_each_others_strategies(self, pytester):
        """The same strategy from a file of another name, as in a sibling pytester test."""
        source = STRATEGIES + TESTS.replace("from pytest_strategy import Strategy\n", "")
        error = ("-W", "error::pytest_strategy.strategy.PytestStrategiesWarning")

        first = pytester.makepyfile(test_first=source)
        pytester.runpytest_inprocess(*error).assert_outcomes(passed=3)
        assert "r2_ints" not in Strategy._registry

        first.unlink()
        pytester.makepyfile(test_second=source)
        pytester.runpytest_inprocess(*error).assert_outcomes(passed=3, warnings=0)


class TestWithoutTerminalPlugin:
    """-p no:terminal removes the -v option and the terminal reporter."""

    def test_importorskip_in_strategy_file_does_not_crash(self, pytester):
        pytester.makepyfile(strategies=f"""
            import pytest

            pytest.importorskip({MISSING_MODULE!r})

            from pytest_strategy import Strategy

            @Strategy.register("r2_optional")
            def optional(nsamples):
                return ("x",), [(1,)]
            """)
        pytester.makepyfile(test_other="def test_other():\n    pass\n")

        result = pytester.runpytest_subprocess("-p", "no:terminal")

        assert "INTERNALERROR" not in result.stderr.str()
        assert result.ret == pytest.ExitCode.OK

    def test_loaded_file_is_not_reported_as_failed(self, pytester):
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_typo=TESTS.replace('"r2_ints"', '"r2_intz"'))

        result = pytester.runpytest_subprocess("-p", "no:terminal", "--junitxml=report.xml")

        assert result.ret == pytest.ExitCode.INTERRUPTED
        report = (pytester.path / "report.xml").read_text()
        assert "Strategy 'r2_intz' not found" in report
        assert "failed to load" not in report


class TestSkippedStrategyFileIsNamed:
    """Tests that need a skipped file's strategy are told the file was skipped."""

    def test_not_found_error_names_the_skipped_file(self, pytester):
        pytester.makepyfile(strategies=f"""
            import pytest

            pytest.importorskip({MISSING_MODULE!r})

            from pytest_strategy import Strategy

            @Strategy.register("r2_arr")
            def arr(nsamples):
                return ("x",), [(1,)]
            """)
        pytester.makepyfile(test_arr="""
            from pytest_strategy import Strategy

            @Strategy.strategy("r2_arr")
            def test_arr(x):
                pass
            """)

        result = pytester.runpytest_subprocess()

        result.stdout.fnmatch_lines(
            [
                "In test_*: Strategy 'r2_arr' not found. Available strategies: none",
                "*Strategy files that were skipped:",
                f"*strategies.py: could not import '{MISSING_MODULE}': *",
            ]
        )


class TestVerboseLoadOutsideRootdir:
    """-vv reports a strategy file outside rootdir as loaded, not as failed."""

    def test_absolute_testpaths_outside_rootdir(self, pytester, monkeypatch):
        shared = pytester.path / "shared"
        pytester.makepyfile(**{"shared/strategies": STRATEGIES, "shared/test_ints": TESTS})
        project = pytester.mkdir("proj")
        (project / "pytest.ini").write_text(f"[pytest]\ntestpaths = {shared}\n")
        monkeypatch.chdir(project)

        result = pytester.runpytest_subprocess("-vv")

        result.stdout.no_fnmatch_line("*Failed to load*")
        result.stdout.fnmatch_lines([f"pytest-strategies: Loaded {shared / 'strategies.py'}"])
        result.assert_outcomes(passed=3)


class TestDiscoveryScope:
    """Where strategy files are searched for."""

    PLUGIN_LOOKALIKE = '''
        """Usage:

            @Strategy.register("name")
        """
        from ._missing_sibling import something
        '''

    def _make_virtualenv(self, pytester, name):
        """A virtualenv with pytest-strategies installed: its strategy.py fails to load."""
        pytester.makepyfile(
            **{f"{name}/lib/site-packages/pytest_strategy/strategy": self.PLUGIN_LOOKALIKE}
        )
        (pytester.path / name / "pyvenv.cfg").write_text("home = /usr/bin\n")

    def test_virtualenv_in_the_project_is_not_searched(self, pytester):
        self._make_virtualenv(pytester, "venv")
        self._make_virtualenv(pytester, "myenv")
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_ints=TESTS)

        result = pytester.runpytest_subprocess()

        result.stdout.no_fnmatch_line("*Failed to load*")
        result.assert_outcomes(passed=3)

    def test_norecursedirs_is_honoured(self, pytester):
        pytester.makeini("[pytest]\nnorecursedirs = legacy\n")
        pytester.makepyfile(
            **{"legacy/strategies": "# @Strategy.register\nraise RuntimeError('old code')\n"}
        )
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_ints=TESTS)

        result = pytester.runpytest_subprocess()

        result.stdout.no_fnmatch_line("*Failed to load*")
        result.assert_outcomes(passed=3)

    def test_skipped_directory_given_on_the_command_line_is_searched(self, pytester):
        pytester.makepyfile(**{".checks/strategies": STRATEGIES, ".checks/test_ints": TESTS})

        result = pytester.runpytest_subprocess(".checks")

        result.assert_outcomes(passed=3)

    def test_testpaths_glob_is_expanded(self, pytester):
        pytester.makeini("[pytest]\ntestpaths = pkgs/*/tests\n")
        pytester.makepyfile(
            **{"pkgs/alpha/tests/strategies": STRATEGIES, "pkgs/alpha/tests/test_ints": TESTS}
        )

        result = pytester.runpytest_subprocess()

        result.assert_outcomes(passed=3)

    def test_strategies_next_to_a_command_line_path_are_loaded(self, pytester):
        pytester.makeini("[pytest]\ntestpaths = tests\n")
        pytester.makepyfile(**{"tests/test_plain": "def test_plain():\n    pass\n"})
        pytester.makepyfile(
            **{"integration/strategies": STRATEGIES, "integration/test_integ": TESTS}
        )

        result = pytester.runpytest_subprocess("integration/test_integ.py::test_ints")

        result.assert_outcomes(passed=3)


class TestListStrategiesUnderXdist:
    """--list-strategies lists in-process when pytest-xdist distribution is on."""

    def test_list_strategies_with_workers(self, pytester):
        pytest.importorskip("xdist")
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_ints=TESTS)

        result = pytester.runpytest_subprocess("--list-strategies", "-n", "2")

        result.stdout.no_fnmatch_line("*INTERNALERROR*")
        result.stdout.fnmatch_lines(["*Found 1 registered strategies:*", "*r2_ints*"])
        assert result.ret == pytest.ExitCode.OK


VECTOR_STRATEGIES = """
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("r2_ages")
def ages(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 9)),
        directed_vectors={"newborn": (0,), "old": (9,)},
    )

@Strategy.register("r2_other")
def other(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), directed_vectors={"other": (1,)})
"""

VECTOR_TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("r2_ages")
def test_ages(x):
    pass

@Strategy.strategy("r2_other")
def test_other(x):
    pass

def test_plain():
    pass
"""

NO_MATCH = (
    "matched no directed vector in any strategy. "
    "Directed vectors by strategy: r2_ages: newborn, old; r2_other: other"
)


class TestVectorFilterMatchingNothing:
    """--vector-name/--vector-index that no strategy has is an error, not an all-skip run."""

    @pytest.fixture(autouse=True)
    def _project(self, pytester):
        pytester.makepyfile(strategies=VECTOR_STRATEGIES)
        pytester.makepyfile(test_vectors=VECTOR_TESTS)

    @pytest.mark.parametrize("option", ["--vector-name=newbron", "--vector-index=99"])
    def test_filter_matching_no_strategy_is_a_usage_error(self, pytester, option):
        result = pytester.runpytest_subprocess(option)

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines([f"ERROR: {option} {NO_MATCH}"])

    def test_filter_matching_some_strategies_skips_the_others(self, pytester):
        result = pytester.runpytest_subprocess("--vector-name=newborn")

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=2, skipped=1)

    def test_filter_without_parameter_strategies_is_not_checked(self, pytester):
        pytester.makepyfile(strategies="""
            from pytest_strategy import Strategy

            @Strategy.register("r2_ages")
            def ages(nsamples):
                return ("x",), [(1,)]

            @Strategy.register("r2_other")
            def other(nsamples):
                return ("x",), [(2,)]
            """)

        result = pytester.runpytest_subprocess("--vector-name=newbron")

        result.assert_outcomes(passed=3)

    def test_filter_matching_no_strategy_under_xdist(self, pytester):
        pytest.importorskip("xdist")

        result = pytester.runpytest_subprocess("--vector-name=newbron", "-n", "2")

        output = result.stdout.str()
        assert result.ret != pytest.ExitCode.OK
        assert "INTERNALERROR" not in output
        assert f"--vector-name=newbron {NO_MATCH}" in re.sub(r"\s+", " ", output)
        result.assert_outcomes()
