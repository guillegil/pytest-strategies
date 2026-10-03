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
import re

import pytest

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


def run(pytester, *args):
    """Run pytest in a subprocess, with pytest's cache, from the rootdir."""
    (pytester.path / "ran.txt").unlink(missing_ok=True)
    return pytester.runpytest_subprocess(*args)


def ran(pytester):
    """The node IDs the last run set up, sorted."""
    path = pytester.path / "ran.txt"
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


def failed_seeds(pytester):
    """The failed-seeds map in the project's cache."""
    path = pytester.path / ".pytest_cache" / "v" / "pytest-strategies" / "failed-seeds"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def lastfailed(pytester):
    """pytest's own last-failed set."""
    path = pytester.path / ".pytest_cache" / "v" / "cache" / "lastfailed"
    return set(json.loads(path.read_text(encoding="utf-8")))


def nodeids(folder="tests", failing=FAILING):
    """The node IDs of the failing rows of the test module in ``folder``."""
    name = folder.rpartition("/")[2]
    return sorted(f"{folder}/test_lfr_{name}.py::{row}" for row in failing)


REUSED = "pytest-strategies: seed reused from the failed run for {} (--rng-seed overrides)"


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
                f"  pytest --lf --rng-seed={S1}  # 2 rows",
            ]
        )
        # pytest keeps the deselected rows in its last-failed set
        assert lastfailed(pytester) == set(nodeids("tests/a") + nodeids("tests/b"))

        again = run(pytester, "--lf", f"--rng-seed={S1}")

        assert ran(pytester) == nodeids("tests/a") + nodeids("tests/b")
        assert set(values(a)) <= set(values(again))

    def test_the_paths_on_the_command_line_choose_the_seed(self, pytester):
        a, _ = self.record(pytester)

        result = run(pytester, "--lf", "tests/a")

        assert header(result) == [f"pytest-strategies: RNG seed = {S1}", REUSED.format("--lf")]
        assert ran(pytester) == nodeids("tests/a")
        assert values(result) == values(a)
        result.stdout.no_fnmatch_line("*deselected*")

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
                f"  pytest --lf --rng-seed={S1}  # 2 rows",
            ]
        )
        assert set(failed_seeds(pytester)) == set(nodeids("tests/a") + nodeids("tests/b"))


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
