"""
End-to-end tests for the seed reuse of ``--lf`` and ``--sw`` (D4): a rerun of the
failed strategy rows without ``--rng-seed`` reuses the seed of the run that
recorded them, so they run with the values they failed with; rows recorded under
another seed are deselected, with the command that reruns them; and the
failed-seeds map in pytest's cache keeps an entry until its row passes under its
seed and options.

Every run is a subprocess with pytest's cache plugin, as a user runs pytest. The
project's context hook draws, and its conftest.py computes the context in
``pytest_configure``, before the plugin seeds the RNG, so a rerun shows the same
context only when it reused the seed from the start. The conftest.py writes the
node ID of every test it sets up to ``ran.txt`` in the rootdir.
"""

import json
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest

from pytest_strategy._repro import quote

pytest_plugins = ["pytester"]

S1 = 5
S2 = 77

CONFTEST = """
from pytest_strategy import RNG, get_context

def pytest_strategies_context(config):
    return {"top": RNG.integer(1000, 10**6)}

def pytest_configure(config):
    # Computed before the plugin seeds the RNG: it draws from the run's seed
    get_context(config, config.rootpath)

def pytest_runtest_setup(item):
    with open(item.config.rootpath / "ran.txt", "a", encoding="utf-8") as ran:
        ran.write(item.nodeid + "\\n")
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("burst")
def burst(ctx):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, ctx["top"])),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        nsamples=6,
    )
"""

FAILING = ("test_w[rand-1]", "test_w[rand-4]")


def module(failing=FAILING):
    """A test module whose rows named in ``failing`` fail, showing their values."""
    return f"""
from pytest_strategy import strategy

FAILING = {tuple(failing)!r}

@strategy("burst")
def test_w(request, addr, len):
    assert request.node.name not in FAILING, f"{{request.node.name}}: addr={{addr}} len={{len}}"
"""


def project(pytester, folders=("tests",), failing=FAILING):
    """The conftest.py, tests/strategies.py and a test module in each folder."""
    pytester.makeini("[pytest]\n")
    pytester.makeconftest(CONFTEST)
    (pytester.path / "tests").mkdir(exist_ok=True)
    (pytester.path / "tests" / "strategies.py").write_text(STRATEGIES, encoding="utf-8")
    for folder in folders:
        path = pytester.path / folder
        path.mkdir(parents=True, exist_ok=True)
        name = f"test_lfr_{path.name}.py"
        (path / name).write_text(module(failing), encoding="utf-8")


def run(pytester, *args, rootdir="."):
    """Run pytest in a subprocess, with pytest's cache, from the top folder."""
    (pytester.path / rootdir / "ran.txt").unlink(missing_ok=True)
    return pytester.runpytest_subprocess(*args)


def ran(pytester, rootdir="."):
    """The node IDs the last run set up, sorted."""
    path = pytester.path / rootdir / "ran.txt"
    return sorted(path.read_text(encoding="utf-8").splitlines()) if path.exists() else []


def values(result):
    """The values the failed rows showed (their assertion messages), sorted."""
    return sorted(line for line in result.stdout.lines if "AssertionError: test_w[" in line)


def sections(result):
    """The failed rows' section lines but their rerun lines (context included), sorted."""
    keep = ("strategy  ", "vector    ", "values    ", "          ", "seed      ", "context   ")
    return sorted(line for line in result.stdout.lines if line.startswith(keep))


def header(result):
    """The pytest-strategies lines of the report header."""
    lines = result.stdout.lines
    end = next(i for i, line in enumerate(lines) if line.startswith("rootdir: "))
    return [line for line in lines[:end] if line.startswith("pytest-strategies: ")]


def failed_seeds(pytester, rootdir="."):
    """The failed-seeds map in the project's cache."""
    path = pytester.path / rootdir / ".pytest_cache" / "v" / "pytest-strategies" / "failed-seeds"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def lastfailed(pytester, rootdir="."):
    """pytest's own last-failed set."""
    path = pytester.path / rootdir / ".pytest_cache" / "v" / "cache" / "lastfailed"
    return set(json.loads(path.read_text(encoding="utf-8")))


def nodeids(folder="tests", failing=FAILING):
    """The node IDs of the failing rows of the test module in ``folder``."""
    name = folder.rpartition("/")[2]
    return sorted(f"{folder}/test_lfr_{name}.py::{row}" for row in failing)


def printed(result, heading="recorded under another seed"):
    """The arguments of the commands the run printed for the rows it set aside."""
    lines = result.stdout.lines
    start = next(i for i, line in enumerate(lines) if heading in line)
    commands = []
    for line in lines[start + 1 :]:
        if not line.startswith("  pytest "):
            break
        commands.append(split(line.partition("  # ")[0])[1:])
    return commands


def split(command):
    """
    The arguments of a printed command, split as the platform's shell splits them:
    on Windows, double quotes and backslashes that separate the parts of a path.
    """
    if os.name != "nt":
        return shlex.split(command)
    args = shlex.split(command, posix=False)
    return [arg[1:-1] if len(arg) > 1 and arg[0] == arg[-1] == '"' else arg for arg in args]


REUSED = "pytest-strategies: seed reused from the failed run for {} (--rng-seed overrides)"

UNCOLLECTED = (
    "{} recorded under the reused seed {} not collected; unless deleted or renamed, run {} with:"
)


class TestLastFailed:
    def test_lf_reruns_the_failed_rows_with_their_values(self, pytester):
        project(pytester)
        first = run(pytester, f"--rng-seed={S1}")
        first.assert_outcomes(failed=2, passed=4)

        given = run(pytester, "--lf", f"--rng-seed={S1}")
        reused = run(pytester, "--lf")

        given.assert_outcomes(failed=2)
        assert values(given) == values(first) != []
        assert sections(given) == sections(first)
        reused.assert_outcomes(failed=2)
        assert ran(pytester) == nodeids()
        assert values(reused) == values(first)
        assert sections(reused) == sections(first)
        # The seed line, which scripts parse, is unchanged; a line of its own follows
        assert header(given) == [f"pytest-strategies: RNG seed = {S1}"]
        assert header(reused) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        reused.stdout.fnmatch_lines([f"pytest-strategies: reproduce with --rng-seed={S1}*"])

    def test_the_map_holds_the_failed_rows(self, pytester):
        project(pytester)

        run(pytester, f"--rng-seed={S1}", "--nsamples=6")

        assert failed_seeds(pytester) == {
            nodeid: {"seed": S1, "options": ["--nsamples=6"]} for nodeid in nodeids()
        }

    def test_ff_draws_a_new_seed(self, pytester):
        project(pytester)
        run(pytester, f"--rng-seed={S1}")

        result = run(pytester, "--ff")

        (seed_line,) = header(result)
        assert re.fullmatch(r"pytest-strategies: RNG seed = \d+", seed_line)
        assert seed_line != f"pytest-strategies: RNG seed = {S1}"
        result.assert_outcomes(failed=2, passed=4)

    def test_the_reuse_line_names_the_recorded_options(self, pytester):
        failing = ("test_w[rand-12]",)
        project(pytester, failing=failing)
        first = run(pytester, f"--rng-seed={S1}", "--nsamples=13")
        first.assert_outcomes(failed=1, passed=12)

        without = run(pytester, "--lf")
        same = run(pytester, "--lf", "--nsamples=13")

        assert header(without)[1] == REUSED.format("--lf") + "; recorded with --nsamples=13"
        # The options are not applied: 6 rows have no rand-12, and pytest runs the
        # tests it collected when none of them failed before
        without.assert_outcomes(passed=6)
        without.stdout.fnmatch_lines(["run-last-failure: 1 known failures not in selected tests*"])
        # The end of the run gives the command, with the options
        without.stdout.fnmatch_lines(
            [
                f"pytest-strategies: {UNCOLLECTED.format('1 failed row', 'was', 'it')}",
                f"  pytest --lf --rng-seed={S1} --nsamples=13 tests/test_lfr_tests.py  # 1 row",
            ]
        )
        (command,) = printed(without, "under the reused seed")
        again = run(pytester, *command)
        again.assert_outcomes(failed=1)
        assert values(again) == values(first)
        assert header(same)[1] == REUSED.format("--lf")
        same.assert_outcomes(failed=1)
        assert values(same) == values(first)

    def test_the_reuse_line_names_a_constraint_the_run_leaves_on(self, pytester):
        project(pytester)
        run(pytester, f"--rng-seed={S1}", "--strategy-constraint-off=aligned")

        same = run(pytester, "--lf", "--strategy-constraint-off=aligned")
        on = run(pytester, "--lf")

        assert header(same)[:2] == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        assert header(on)[1] == (
            REUSED.format("--lf") + "; recorded with --strategy-constraint-off=burst:aligned"
        )

    def test_rng_seed_wins(self, pytester):
        project(pytester, folders=("tests/a", "tests/b"))
        run(pytester, f"--rng-seed={S1}", "tests/a")
        run(pytester, f"--rng-seed={S2}", "tests/b")

        result = run(pytester, "--lf", "--rng-seed=9")

        assert header(result) == ["pytest-strategies: RNG seed = 9"]
        # Every failed row runs, none is deselected
        assert ran(pytester) == nodeids("tests/a") + nodeids("tests/b")
        result.stdout.no_fnmatch_line("*deselected*")

    def test_a_conftest_s_rng_seed_call_does_not_change_the_reused_seed(self, pytester):
        project(pytester)
        conftest = CONFTEST.replace(
            "def pytest_configure(config):\n", "def pytest_configure(config):\n    RNG.seed(1234)\n"
        )
        pytester.makeconftest(conftest)
        first = run(pytester, f"--rng-seed={S1}")

        result = run(pytester, "--lf")

        # The reused seed acts as --rng-seed, which RNG.seed() does not change either
        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        assert values(result) == values(first) != []
        assert sections(result) == sections(first)

    def test_a_conftest_that_sets_the_option_wins(self, pytester):
        project(pytester, folders=("tests/a", "tests/b"))
        seeding = "    if config.option.rng_seed is None:\n        config.option.rng_seed = 1234\n"
        conftest = CONFTEST.replace(
            "def pytest_configure(config):\n", "def pytest_configure(config):\n" + seeding
        )
        pytester.makeconftest(conftest)
        run(pytester, f"--rng-seed={S1}", "tests/a")
        run(pytester, f"--rng-seed={S2}", "tests/b")

        result = run(pytester, "--lf")

        # As --rng-seed does: no reuse, every failed row runs
        assert header(result) == ["pytest-strategies: RNG seed = 1234"]
        result.assert_outcomes(failed=4)
        assert ran(pytester) == nodeids("tests/a") + nodeids("tests/b")
        result.stdout.no_fnmatch_line("*deselected*")


class TestTwoSeeds:
    def record(self, pytester):
        """Fail two rows of tests/a under S1, then two rows of tests/b under S2."""
        project(pytester, folders=("tests/a", "tests/b"))
        a = run(pytester, f"--rng-seed={S1}", "tests/a")
        b = run(pytester, f"--rng-seed={S2}", "tests/b")
        a.assert_outcomes(failed=2, passed=4)
        b.assert_outcomes(failed=2, passed=4)
        return a, b

    def test_lf_runs_the_newest_seed_s_rows_and_deselects_the_others(self, pytester):
        a, b = self.record(pytester)

        result = run(pytester, "--lf")

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        result.assert_outcomes(failed=2, deselected=2)
        assert ran(pytester) == nodeids("tests/b")
        assert values(result) == values(b)
        assert sections(result) == sections(b)
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: deselected 2 failed rows recorded under another seed; "
                "run them with:",
                f"  pytest --lf --rng-seed={S1} tests/a/test_lfr_a.py  # 2 rows",
            ]
        )
        # pytest keeps the deselected rows in its last-failed set
        assert lastfailed(pytester) == set(nodeids("tests/a") + nodeids("tests/b"))

        (command,) = printed(result)
        again = run(pytester, *command)

        # The printed command reruns only the rows recorded under S1, with their values
        again.assert_outcomes(failed=2)
        assert ran(pytester) == nodeids("tests/a")
        assert values(again) == values(a)
        assert sections(again) == sections(a)

    def test_the_printed_command_leaves_the_other_seed_s_rows_alone(self, pytester):
        # tests/b fails only with the values it got under S2: under S1 it would pass,
        # and pytest would drop it from its last-failed set, with the bug unfixed
        a, b = self.record(pytester)
        failing = [line.rpartition("AssertionError: ")[2] for line in values(b)]
        fixed_unless = f"""
from pytest_strategy import strategy

@strategy("burst")
def test_w(request, addr, len):
    shown = f"{{request.node.name}}: addr={{addr}} len={{len}}"
    assert shown not in {failing!r}, shown
"""
        (pytester.path / "tests" / "b" / "test_lfr_b.py").write_text(fixed_unless, "utf-8")
        result = run(pytester, "--lf")
        result.assert_outcomes(failed=2, deselected=2)

        (command,) = printed(result)
        run(pytester, *command).assert_outcomes(failed=2)

        assert ran(pytester) == nodeids("tests/a")
        assert lastfailed(pytester) == set(nodeids("tests/a") + nodeids("tests/b"))
        again = run(pytester, "--lf")
        # tests/a failed again under S1, so it is the newest: tests/b is set aside
        assert header(again)[0] == f"pytest-strategies: RNG seed = {S1}"
        assert printed(again) == [["--lf", f"--rng-seed={S2}", "tests/b/test_lfr_b.py"]]

    def test_rows_of_a_file_with_other_failures_are_named_by_node_id(self, pytester):
        project(pytester)
        first, second = (f"tests/test_lfr_tests.py::{row}" for row in FAILING)
        run(pytester, f"--rng-seed={S1}", first).assert_outcomes(failed=1)
        run(pytester, f"--rng-seed={S2}", second).assert_outcomes(failed=1)

        result = run(pytester, "--lf")

        assert ran(pytester) == [second]
        assert printed(result) == [["--lf", f"--rng-seed={S1}", first]]
        run(pytester, "--lf", f"--rng-seed={S1}", first).assert_outcomes(failed=1)
        assert ran(pytester) == [first]

    def test_the_paths_on_the_command_line_choose_the_seed(self, pytester):
        a, _ = self.record(pytester)

        result = run(pytester, "--lf", "tests/a")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        assert ran(pytester) == nodeids("tests/a")
        assert values(result) == values(a)
        result.stdout.no_fnmatch_line("*deselected*")

    def test_run_from_a_subfolder_it_collects_choose_the_seed(self, pytester, monkeypatch):
        a, _ = self.record(pytester)
        monkeypatch.chdir(pytester.path / "tests" / "a")

        result = run(pytester, "--lf")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        result.assert_outcomes(failed=2)
        assert ran(pytester) == nodeids("tests/a")
        # The same values; the summary names the tests relative to tests/a
        assert sections(result) == sections(a)
        result.stdout.no_fnmatch_line("*deselected*")

    def test_the_testpaths_choose_the_seed(self, pytester):
        a, _ = self.record(pytester)
        # The strategy file must be in the testpaths to be found
        strategies = pytester.path / "tests" / "strategies.py"
        for folder in ("a", "b"):
            (pytester.path / "tests" / folder / "strategies.py").write_text(STRATEGIES, "utf-8")
        strategies.unlink()
        pytester.makeini("[pytest]\ntestpaths = tests/a\n")

        result = run(pytester, "--lf")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        result.assert_outcomes(failed=2)
        assert ran(pytester) == nodeids("tests/a")
        assert values(result) == values(a)

    def test_k_does_not_choose_the_seed(self, pytester):
        a, _ = self.record(pytester)

        result = run(pytester, "--lf", "-k", "test_lfr_a")

        # The seed is chosen before the tests are collected: the newest row's, of
        # tests/b, whose rows -k deselects. The printed command reruns tests/a's.
        assert header(result)[0] == f"pytest-strategies: RNG seed = {S2}"
        assert ran(pytester) == []
        (command,) = printed(result)
        again = run(pytester, *command)
        assert ran(pytester) == nodeids("tests/a")
        assert values(again) == values(a)

    def test_rows_whose_ids_show_their_values_get_their_command(self, pytester):
        # Under another seed, their node IDs are not collected at all
        project(pytester, folders=("tests/a", "tests/b"))
        by_index = """
from pytest_strategy import VECTOR_KEY, strategy

@strategy("burst")
def test_w(request, addr, len):
    assert request.node.stash[VECTOR_KEY].index not in (1, 4), f"{request.node.name}"
"""
        for folder in ("a", "b"):
            (pytester.path / "tests" / folder / f"test_lfr_{folder}.py").write_text(
                by_index, "utf-8"
            )
        ids = ("-o", "strategies_ids=values")
        a = run(pytester, f"--rng-seed={S1}", *ids, "tests/a")
        run(pytester, f"--rng-seed={S2}", *ids, "tests/b").assert_outcomes(failed=2, passed=4)

        result = run(pytester, "--lf", *ids)

        assert header(result)[0] == f"pytest-strategies: RNG seed = {S2}"
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: deselected 2 failed rows recorded under another seed; "
                "run them with:",
                f"  pytest --lf --rng-seed={S1} -o strategies_ids=values tests/a/test_lfr_a.py"
                "  # 2 rows",
            ]
        )
        (command,) = printed(result)
        again = run(pytester, *command)
        again.assert_outcomes(failed=2)
        assert values(again) == values(a)

    @pytest.mark.parametrize("xdist", [False, True], ids=["one_process", "xdist"])
    def test_the_reused_seed_s_rows_that_are_not_collected_get_their_command(self, pytester, xdist):
        # The newest rows' test was deleted, in a file that still exists: every other
        # failed row is deselected, so no test runs, and the end of the run says why
        a, _ = self.record(pytester)
        (pytester.path / "tests" / "b" / "test_lfr_b.py").write_text(
            "def test_kept():\n    pass\n", encoding="utf-8"
        )
        args = ["-n", "2"] if xdist else []
        if xdist:
            pytest.importorskip("xdist")

        result = run(pytester, "--lf", *args)

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        assert ran(pytester) == []
        assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
        result.stdout.fnmatch_lines(
            [
                f"pytest-strategies: {UNCOLLECTED.format('2 failed rows', 'were', 'them')}",
                f"  pytest --lf --rng-seed={S2} tests/b/test_lfr_b.py  # 2 rows",
                "pytest-strategies: deselected 2 failed rows recorded under another seed; "
                "run them with:",
                f"  pytest --lf --rng-seed={S1} tests/a/test_lfr_a.py  # 2 rows",
            ]
        )
        assert lastfailed(pytester) == set(nodeids("tests/a") + nodeids("tests/b"))
        again = run(pytester, *printed(result)[0])
        assert values(again) == values(a)

    def test_no_word_on_the_reused_rows_when_they_are_collected(self, pytester):
        self.record(pytester)

        result = run(pytester, "--lf")

        result.stdout.no_fnmatch_line("*under the reused seed*")

    def test_no_word_on_the_reused_rows_when_collection_failed(self, pytester):
        # The session stops before it runs a test: the rows were not left out
        self.record(pytester)
        (pytester.path / "tests" / "b" / "test_lfr_b.py").write_text(
            "import not_a_module\n", encoding="utf-8"
        )

        result = run(pytester, "--lf")

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines(["*ERROR collecting tests/b/test_lfr_b.py*"])
        result.stdout.no_fnmatch_line("*under the reused seed*")

    def test_an_entry_whose_file_was_deleted_is_ignored(self, pytester):
        a, _ = self.record(pytester)
        (pytester.path / "tests" / "b" / "test_lfr_b.py").unlink()

        result = run(pytester, "--lf")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        assert ran(pytester) == nodeids("tests/a")
        assert values(result) == values(a)
        result.stdout.no_fnmatch_line("*deselected*")

    def test_under_xdist_the_workers_get_the_reused_seed(self, pytester):
        pytest.importorskip("xdist")
        _, b = self.record(pytester)

        result = run(pytester, "--lf", "-n", "2")

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        result.assert_outcomes(failed=2)
        assert ran(pytester) == nodeids("tests/b")
        assert values(result) == values(b)
        assert sections(result) == sections(b)
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: deselected 2 failed rows recorded under another seed; "
                "run them with:",
                f"  pytest --lf --rng-seed={S1} tests/a/test_lfr_a.py  # 2 rows",
            ]
        )
        # The workers collected the reused rows, and said so to the controller
        result.stdout.no_fnmatch_line("*under the reused seed*")
        assert set(failed_seeds(pytester)) == set(nodeids("tests/a") + nodeids("tests/b"))


class TestOutsideTheRootdir:
    """
    ``-c ci/pytest.ini`` makes ``ci`` the rootdir: pytest names a test file outside
    it from the path the run started from that contains it (here the folder it runs
    in), ``tests/a/test_lfr_a.py``, and the rerun keeps that name.
    """

    CI = ("-c", "ci/pytest.ini")

    def record(self, pytester, folders=("tests/a", "tests/b"), seeds=(S1, S2)):
        """
        Write ci/pytest.ini, and the conftest.py and strategies.py of tests (and of
        ci when it holds a test module), with a test module in each folder; then
        fail two rows of each test module under its seed, selected with -k.
        """
        files = {"ci/pytest.ini": "[pytest]\n"}
        for folder in folders:
            top = folder.partition("/")[0]
            files[f"{top}/conftest.py"] = CONFTEST
            files[f"{top}/strategies.py"] = STRATEGIES
            files[f"{folder}/test_lfr_{folder.rpartition('/')[2]}.py"] = module()
        for name, text in files.items():
            (pytester.path / name).parent.mkdir(parents=True, exist_ok=True)
            (pytester.path / name).write_text(text, encoding="utf-8")
        found = []
        for folder, seed in zip(folders, seeds):
            name = f"test_lfr_{folder.rpartition('/')[2]}"
            result = run(pytester, *self.CI, f"--rng-seed={seed}", "-k", name, rootdir="ci")
            result.assert_outcomes(failed=2, passed=4, deselected=6 * (len(folders) - 1))
            found.append(result)
        return found

    def test_lf_reruns_the_failed_rows_with_their_values(self, pytester):
        (first,) = self.record(pytester, folders=("tests/a",), seeds=(S1,))
        assert list(failed_seeds(pytester, "ci")) == nodeids("tests/a")

        result = run(pytester, *self.CI, "--lf", rootdir="ci")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        result.assert_outcomes(failed=2, deselected=4)
        assert ran(pytester, "ci") == nodeids("tests/a")
        assert values(result) == values(first) != []
        assert sections(result) == sections(first)

    def test_the_command_for_rows_of_another_seed_starts_from_the_run_s_path(self, pytester):
        a, b = self.record(pytester)

        result = run(pytester, *self.CI, "--lf", rootdir="ci")

        assert header(result)[0] == f"pytest-strategies: RNG seed = {S2}"
        result.assert_outcomes(failed=2, deselected=10)
        assert values(result) == values(b)
        deselect = [f"--deselect {quote(nodeid)}" for nodeid in nodeids("tests/b")]
        # The map holds -c relative to the rootdir; the command gives it from the
        # folder pytest runs in, in the platform's form
        ini = quote(str(Path("ci", "pytest.ini")))
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: deselected 2 failed rows recorded under another seed; "
                "run them with:",
                f"  pytest --lf --rng-seed={S1} -c {ini} . {' '.join(deselect)}  # 2 rows",
            ]
        )

        (command,) = printed(result)
        again = run(pytester, *command, rootdir="ci")

        # Only the rows recorded under S1 run, with their values, and the others
        # stay in pytest's last-failed set
        assert ran(pytester, "ci") == nodeids("tests/a")
        assert values(again) == values(a)
        assert lastfailed(pytester, "ci") == set(nodeids("tests/a") + nodeids("tests/b"))

    def test_a_file_inside_the_rootdir_with_a_failed_test_is_ignored(self, pytester):
        # pytest's --lf collects no file outside the rootdir once it collected one
        # inside it that holds a failed test
        a, ci = self.record(pytester, folders=("tests/a", "ci"))

        result = run(pytester, *self.CI, "--lf", rootdir="ci")

        assert header(result)[0] == f"pytest-strategies: RNG seed = {S2}"
        assert values(result) == values(ci)
        (command,) = printed(result)
        assert command == [
            "--lf",
            f"--rng-seed={S1}",
            "-c",
            str(Path("ci", "pytest.ini")),
            ".",
            "--ignore",
            str(Path("ci", "test_lfr_ci.py")),
        ]
        again = run(pytester, *command, rootdir="ci")
        assert ran(pytester, "ci") == nodeids("tests/a")
        assert values(again) == values(a)


class TestFromAnotherFolder:
    """
    The rows are recorded from the top folder with ``-c ci/pytest.ini``, which makes
    ``ci`` the rootdir, and rerun from ``ci``, where pytest finds that file itself:
    the map holds ``-c`` relative to the rootdir.
    """

    def record(self, pytester):
        """Fail two rows of ci/tests/a under S1, then two rows of ci/tests/b under S2."""
        files = {
            "ci/pytest.ini": "[pytest]\n",
            "ci/conftest.py": CONFTEST,
            "ci/tests/strategies.py": STRATEGIES,
            "ci/tests/a/test_lfr_a.py": module(),
            "ci/tests/b/test_lfr_b.py": module(),
        }
        for name, text in files.items():
            (pytester.path / name).parent.mkdir(parents=True, exist_ok=True)
            (pytester.path / name).write_text(text, encoding="utf-8")
        found = []
        for folder, seed in (("a", S1), ("b", S2)):
            path = str(Path("ci", "tests", folder))
            result = run(pytester, "-c", "ci/pytest.ini", path, f"--rng-seed={seed}", rootdir="ci")
            result.assert_outcomes(failed=2, passed=4)
            found.append(result)
        assert {entry["options"][1] for entry in failed_seeds(pytester, "ci").values()} == {
            "pytest.ini"
        }
        return found

    @staticmethod
    def errors(result):
        """The failed rows' assertion messages, without the short summary's paths."""
        return [line for line in values(result) if line.startswith("E ")]

    def test_the_command_for_rows_of_another_seed_runs_from_there(self, pytester, monkeypatch):
        a, b = self.record(pytester)
        monkeypatch.chdir(pytester.path / "ci")

        result = run(pytester, "--lf", rootdir="ci")

        # The run uses the ini file the rows were recorded with: no difference
        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        assert self.errors(result) == self.errors(b) != []
        (command,) = printed(result)
        assert command == [
            "--lf",
            f"--rng-seed={S1}",
            "-c",
            "pytest.ini",
            "tests/a/test_lfr_a.py",
        ]
        again = run(pytester, *command, rootdir="ci")
        again.assert_outcomes(failed=2)
        assert ran(pytester, "ci") == nodeids("tests/a")
        assert self.errors(again) == self.errors(a)

    def test_a_row_that_passes_there_under_its_seed_is_removed(self, pytester, monkeypatch):
        self.record(pytester)
        monkeypatch.chdir(pytester.path / "ci")
        (pytester.path / "ci" / "tests" / "a" / "test_lfr_a.py").write_text(
            module(()), encoding="utf-8"
        )

        result = run(pytester, "--lf", f"--rng-seed={S1}", str(Path("tests", "a")), rootdir="ci")

        result.assert_outcomes(passed=2)
        assert list(failed_seeds(pytester, "ci")) == nodeids("tests/b")


class TestAnIniFileOutsideTheRootdir:
    """
    The rows are recorded with ``-c`` naming an ini file outside the rootdir and
    ``--rootdir=.`` (as ``pytest -c /dev/null --rootdir=.``), then the project moves,
    with its cache, one folder deeper: the map holds the ini file's absolute path,
    which a chain of ``..`` from the rootdir would not keep.
    """

    def test_a_project_moved_deeper_still_agrees(self, pytester, monkeypatch):
        ini = pytester.path / "shared.ini"
        ini.write_text("[pytest]\n", encoding="utf-8")
        files = {
            "conftest.py": CONFTEST,
            "tests/strategies.py": STRATEGIES,
            "tests/a/test_lfr_a.py": module(),
            "tests/b/test_lfr_b.py": module(),
        }
        for name, text in files.items():
            (pytester.path / "project" / name).parent.mkdir(parents=True, exist_ok=True)
            (pytester.path / "project" / name).write_text(text, encoding="utf-8")
        monkeypatch.chdir(pytester.path / "project")
        for folder, seed in (("a", S1), ("b", S2)):
            args = ["-c", str(ini), "--rootdir=.", str(Path("tests", folder)), f"--rng-seed={seed}"]
            run(pytester, *args, rootdir="project").assert_outcomes(failed=2, passed=4)
        options = {tuple(entry["options"]) for entry in failed_seeds(pytester, "project").values()}
        assert options == {("-c", str(ini), "--rootdir=.")}
        shutil.copytree(pytester.path / "project", pytester.path / "moved" / "project")
        monkeypatch.chdir(pytester.path / "moved" / "project")
        # tests/b is fixed
        Path("tests", "b", "test_lfr_b.py").write_text(module(()), encoding="utf-8")

        result = run(pytester, "--lf", "-c", str(ini), "--rootdir=.", rootdir="moved/project")

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        result.assert_outcomes(passed=2, deselected=2)
        assert printed(result) == [
            ["--lf", f"--rng-seed={S1}", "-c", str(ini), "--rootdir=.", "tests/a/test_lfr_a.py"]
        ]
        assert list(failed_seeds(pytester, "moved/project")) == nodeids("tests/a")


class TestTheMap:
    def test_a_row_that_passes_under_its_seed_is_removed(self, pytester):
        project(pytester, folders=("tests/a", "tests/b"))
        run(pytester, f"--rng-seed={S1}", "tests/a")
        run(pytester, f"--rng-seed={S2}", "tests/b")
        # tests/b is fixed
        (pytester.path / "tests" / "b" / "test_lfr_b.py").write_text(module(()), encoding="utf-8")

        result = run(pytester, "--lf")

        result.assert_outcomes(passed=2, deselected=2)
        assert list(failed_seeds(pytester)) == nodeids("tests/a")

    @pytest.mark.parametrize(
        ("recorded", "given"),
        [([], ["--rootdir=."]), (["--rootdir=."], [])],
        ids=["given_by_the_run_only", "recorded_only"],
    )
    def test_a_rootdir_given_on_one_side_only_agrees(self, pytester, recorded, given):
        # The rows were recorded under the rootdir whose cache holds them, the run's
        project(pytester, folders=("tests/a", "tests/b"))
        run(pytester, f"--rng-seed={S1}", "tests/a", *recorded)
        run(pytester, f"--rng-seed={S2}", "tests/b", *recorded)
        (pytester.path / "tests" / "b" / "test_lfr_b.py").write_text(module(()), encoding="utf-8")

        result = run(pytester, "--lf", *given)

        assert header(result) == [f"pytest-strategies: RNG seed = {S2}", REUSED.format("--lf")]
        result.assert_outcomes(passed=2, deselected=2)
        assert list(failed_seeds(pytester)) == nodeids("tests/a")

    def test_under_xdist_the_controller_removes_it(self, pytester):
        pytest.importorskip("xdist")
        project(pytester)
        run(pytester, f"--rng-seed={S1}")
        (pytester.path / "tests" / "test_lfr_tests.py").write_text(module(()), encoding="utf-8")

        result = run(pytester, "--lf", "-n", "2")

        result.assert_outcomes(passed=2)
        assert failed_seeds(pytester) == {}

    @pytest.mark.parametrize(
        "args",
        [[f"--rng-seed={S1}"], ["--lf", f"--rng-seed={S1}", "tests/a/test_lfr_a.py"]],
        ids=["rng_seed", "printed_command"],
    )
    def test_a_row_that_passes_under_its_seed_without_reuse_is_removed(self, pytester, args):
        # The seed is given, as in the command printed for rows set aside: no reuse
        project(pytester, folders=("tests/a",))
        run(pytester, f"--rng-seed={S1}")
        (pytester.path / "tests" / "a" / "test_lfr_a.py").write_text(module(()), "utf-8")

        result = run(pytester, *args)

        result.assert_outcomes(passed=6 if "--lf" not in args else 2)
        assert header(result) == [f"pytest-strategies: RNG seed = {S1}"]
        assert failed_seeds(pytester) == {}

    @pytest.mark.parametrize(
        "args",
        [[f"--rng-seed={S2}"], ["--lf", "--nsamples=6"]],
        ids=["other_seed", "other_options"],
    )
    def test_a_row_that_passes_otherwise_stays(self, pytester, args):
        project(pytester)
        run(pytester, f"--rng-seed={S1}")
        before = failed_seeds(pytester)
        (pytester.path / "tests" / "test_lfr_tests.py").write_text(module(()), encoding="utf-8")

        run(pytester, *args).assert_outcomes(passed=6 if "--lf" not in args else 2)

        assert failed_seeds(pytester) == before

    def test_a_row_that_fails_again_is_recorded_as_the_newest(self, pytester):
        project(pytester, folders=("tests/a", "tests/b"))
        run(pytester, f"--rng-seed={S1}", "tests/a")
        run(pytester, f"--rng-seed={S2}", "tests/b")

        run(pytester, "--lf", f"--rng-seed={S1}")

        assert list(failed_seeds(pytester).items())[-4:] == [
            (nodeid, {"seed": S1, "options": []})
            for nodeid in nodeids("tests/a") + nodeids("tests/b")
        ]

    def test_without_the_cache_plugin_nothing_is_read_or_written(self, pytester):
        project(pytester)
        run(pytester, f"--rng-seed={S1}")
        path = pytester.path / ".pytest_cache" / "v" / "pytest-strategies" / "failed-seeds"
        before = path.read_bytes(), path.stat().st_mtime_ns

        first = run(pytester, "-p", "no:cacheprovider")
        second = run(pytester, "-p", "no:cacheprovider")

        (one,) = header(first)
        (two,) = header(second)
        # A new seed each time, as before
        assert one != two
        assert f"= {S1}" not in one + two
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    def test_without_the_cache_plugin_no_cache_is_created(self, pytester):
        project(pytester)

        run(pytester, "-p", "no:cacheprovider").assert_outcomes(failed=2, passed=4)

        assert not (pytester.path / ".pytest_cache").exists()

    def test_a_passing_run_writes_no_map(self, pytester):
        project(pytester, failing=())

        run(pytester).assert_outcomes(passed=6)

        assert failed_seeds(pytester) is None


class TestStepwise:
    @pytest.mark.parametrize("option", ["--sw", "--sw-skip"])
    def test_sw_reruns_the_failed_row_with_its_values(self, pytester, option):
        project(pytester)
        first = run(pytester, "--sw", f"--rng-seed={S1}")
        first.assert_outcomes(failed=1, passed=1)

        result = run(pytester, option)

        assert header(result)[:2] == [f"pytest-strategies: RNG seed = {S1}", REUSED.format(option)]
        if option == "--sw":
            # It resumes from the failed row, which fails again with the same values
            result.assert_outcomes(failed=1)
            assert values(result) == values(first)
            assert sections(result) == sections(first)
        else:
            # It skips the failed row's failure and stops at the next one
            result.assert_outcomes(failed=2, passed=2)
            assert values(first)[0] in values(result)

    def test_sw_reset_starts_over_with_a_new_seed(self, pytester):
        project(pytester)
        run(pytester, "--sw", f"--rng-seed={S1}")

        result = run(pytester, "--sw-reset")

        (seed_line,) = header(result)
        assert seed_line != f"pytest-strategies: RNG seed = {S1}"
