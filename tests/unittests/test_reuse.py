"""
Unit tests for the seed reuse of ``--lf`` and ``--sw`` (D4): the failed-seeds map
in pytest's cache, how a run updates it, which entry a ``--lf`` or ``--sw`` run
reuses the seed of and which it deselects, the header note on the options, and
the commands that rerun the deselected rows.
"""

import json
from argparse import Namespace
from functools import partial
from pathlib import Path

import pytest

from pytest_strategy._repro import quote
from pytest_strategy._reuse import (
    KEY,
    LASTFAILED,
    STEPWISE,
    Entry,
    Reuse,
    cache_of,
    commands,
    differences,
    flag,
    from_row,
    plan,
    read,
    updated,
    write,
)
from pytest_strategy._runtime import SessionState, runtime
from pytest_strategy.plugin import (
    _deselected_lines,
    _plugin_instance,
    _read_failed_seeds,
    _record_failed_seeds,
)

NODE_A = "tests/a/test_w.py::test_w[rand-1]"
NODE_B = "tests/b/test_w.py::test_w[rand-2]"
NODE_C = "tests/b/test_w.py::test_w[rand-4]"


class Cache:
    """A stand-in for pytest's cache: values go through JSON, as on disk."""

    def __init__(self, values=None):
        self.values = {key: json.dumps(value) for key, value in (values or {}).items()}
        self.writes = 0

    def get(self, key, default):
        return json.loads(self.values[key]) if key in self.values else default

    def set(self, key, value):
        self.writes += 1
        self.values[key] = json.dumps(value)


class PluginManager:
    def __init__(self, cacheprovider=True):
        self.cacheprovider = cacheprovider

    def has_plugin(self, name):
        return name == "cacheprovider" and self.cacheprovider


def config(root, cache=None, args=(), invocation=None, testpaths=(), **options):
    """
    A stand-in config for a run in ``root`` with ``options`` and ``args`` on the
    command line. Without ``args``, pytest starts from the ``testpaths`` (when it
    runs in the rootdir) or from the folder it runs in, as ``config.args`` holds.
    """
    defaults = {
        "lf": False,
        "stepwise": False,
        "stepwise_skip": False,
        "stepwise_reset": False,
        "cacheclear": False,
    }
    source = pytest.Config.ArgsSource
    invocation = invocation or root
    if args:
        started = (list(args), source.ARGS)
    elif testpaths and invocation == root:
        started = (list(testpaths), source.TESTPATHS)
    else:
        started = ([str(invocation)], source.INVOCATION_DIR)
    return Namespace(
        option=Namespace(**{**defaults, **options}),
        cache=cache,
        rootpath=root,
        args=started[0],
        args_source=started[1],
        invocation_params=Namespace(dir=invocation),
        pluginmanager=PluginManager(),
    )


def files(root, *nodeids):
    """Create the test files of ``nodeids`` below ``root``."""
    for nodeid in nodeids:
        path = root / nodeid.partition("::")[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def entries(*pairs):
    """The map {nodeid: {"seed": S, "options": [...]}} of (nodeid, seed, options) triples."""
    return {nodeid: {"seed": seed, "options": list(options)} for nodeid, seed, options in pairs}


def row(seed, options=()):
    """A failed report's ``pytest_strategies`` attribute."""
    return {
        "command": "pytest ...",
        "row": "burst random 1",
        "seed": str(seed),
        "options": json.dumps(list(options)),
    }


class TestTheMap:
    def test_an_empty_cache_has_no_entries(self):
        assert read(Cache()) == {}

    def test_written_entries_are_read_back_in_order(self):
        cache = Cache()

        write(cache, {NODE_B: Entry(2, ("--nsamples=13",)), NODE_A: Entry(1, ())})

        assert list(read(cache).items()) == [
            (NODE_B, Entry(2, ("--nsamples=13",))),
            (NODE_A, Entry(1, ())),
        ]
        assert json.loads(cache.values[KEY]) == entries(
            (NODE_B, 2, ["--nsamples=13"]), (NODE_A, 1, [])
        )

    @pytest.mark.parametrize(
        "value",
        [
            "x",
            {"seed": "1", "options": []},
            {"seed": True, "options": []},
            {"seed": 1.5, "options": []},
            {"seed": 1},
            {"seed": 1, "options": "--nsamples=13"},
            {"seed": 1, "options": [13]},
        ],
        ids=[
            "text",
            "seed_text",
            "seed_bool",
            "seed_float",
            "no_options",
            "options_text",
            "option_int",
        ],
    )
    def test_entries_of_another_shape_are_left_out(self, value):
        cache = Cache({KEY: {NODE_A: value, NODE_B: {"seed": 2, "options": []}}})

        assert read(cache) == {NODE_B: Entry(2, ())}

    def test_a_map_of_another_shape_has_no_entries(self):
        assert read(Cache({KEY: [NODE_A]})) == {}


class TestFromRow:
    def test_the_seed_and_the_options(self):
        assert from_row(row(21, ["--nsamples=13", "-c", "ci/pytest.ini"])) == Entry(
            21, ("--nsamples=13", "-c", "ci/pytest.ini")
        )

    @pytest.mark.parametrize(
        "attribute",
        [
            {"seed": "21"},
            {"options": "[]"},
            {"seed": "x", "options": "[]"},
            {"seed": "21", "options": "not json"},
            {"seed": "21", "options": '"--nsamples=13"'},
        ],
        ids=["no_options", "no_seed", "seed_not_int", "options_not_json", "options_not_list"],
    )
    def test_none_without_a_usable_seed_or_options(self, attribute):
        assert from_row(attribute) is None


class TestUpdated:
    def test_failed_rows_are_added_as_the_newest_in_failure_order(self):
        before = {NODE_A: Entry(1, ())}

        after = updated(before, 2, {}, {NODE_C: row(2), NODE_B: row(2, ["--nsamples=13"])})

        assert list(after.items()) == [
            (NODE_A, Entry(1, ())),
            (NODE_C, Entry(2, ())),
            (NODE_B, Entry(2, ("--nsamples=13",))),
        ]

    def test_a_row_that_fails_again_moves_to_the_end_with_its_new_seed(self):
        before = {NODE_A: Entry(1, ()), NODE_B: Entry(1, ())}

        after = updated(before, 3, {}, {NODE_A: row(3)})

        assert list(after.items()) == [(NODE_B, Entry(1, ())), (NODE_A, Entry(3, ()))]

    def test_a_row_that_passed_under_its_seed_and_options_is_removed(self):
        before = {NODE_A: Entry(1, ("--nsamples=13",)), NODE_B: Entry(1, ())}

        after = updated(before, 1, {NODE_A: ["--nsamples=13"]}, {})

        assert after == {NODE_B: Entry(1, ())}

    @pytest.mark.parametrize(
        ("seed", "options"),
        [(2, ["--nsamples=13"]), (1, []), (1, ["--nsamples=13", "--vector-mode=test"])],
        ids=["other_seed", "other_options", "more_options"],
    )
    def test_a_row_that_passed_otherwise_stays(self, seed, options):
        before = {NODE_A: Entry(1, ("--nsamples=13",))}

        assert updated(before, seed, {NODE_A: options}, {}) == before

    def test_a_row_that_passed_and_then_failed_stays_as_the_newest(self):
        before = {NODE_A: Entry(1, ()), NODE_B: Entry(1, ())}

        after = updated(before, 1, {NODE_A: []}, {NODE_A: row(1)})

        assert list(after) == [NODE_B, NODE_A]

    def test_a_failed_row_without_a_usable_attribute_is_not_recorded(self):
        assert updated({}, 1, {}, {NODE_A: {"command": "pytest ..."}}) == {}

    def test_the_given_map_is_not_changed(self):
        before = {NODE_A: Entry(1, ())}

        updated(before, 1, {NODE_A: []}, {NODE_B: row(1)})

        assert before == {NODE_A: Entry(1, ())}


class TestFlag:
    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({}, None),
            ({"lf": True}, "--lf"),
            ({"stepwise": True}, "--sw"),
            ({"stepwise_skip": True}, "--sw-skip"),
            ({"stepwise": True, "stepwise_skip": True}, "--sw-skip"),
            ({"stepwise": True, "stepwise_reset": True}, None),
            ({"stepwise_reset": True}, None),
            ({"lf": True, "stepwise": True}, "--lf"),
            ({"lf": True, "stepwise_reset": True}, "--lf"),
        ],
        ids=[
            "none",
            "lf",
            "sw",
            "sw_skip",
            "sw_and_skip",
            "sw_reset",
            "reset_only",
            "lf_sw",
            "lf_reset",
        ],
    )
    def test_the_option_that_reruns_failed_tests(self, tmp_path, options, expected):
        assert flag(config(tmp_path, **options)) == expected

    def test_without_pytest_s_cache_plugin_none(self, tmp_path):
        stand_in = config(tmp_path)
        stand_in.option = Namespace()

        assert flag(stand_in) is None


class TestCacheOf:
    def test_the_config_s_cache(self, tmp_path):
        cache = Cache()

        assert cache_of(config(tmp_path, cache)) is cache

    def test_none_without_pytest_s_cache_plugin(self, tmp_path):
        stand_in = config(tmp_path)
        stand_in.pluginmanager = PluginManager(cacheprovider=False)

        assert cache_of(stand_in) is None

    def test_none_under_cache_clear(self, tmp_path):
        assert cache_of(config(tmp_path, cacheclear=True)) is None


class TestPlan:
    def lf(self, root, values, *args, **options):
        """The plan of a --lf run in ``root`` whose cache holds ``values``."""
        return plan(config(root, Cache(values), args, lf=True, **options))

    def test_the_newest_failed_row_s_seed(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B, NODE_C)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, ["--nsamples=13"]), (NODE_C, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True, NODE_C: True},
        }

        assert self.lf(tmp_path, values) == Reuse(
            flag="--lf",
            seed=2,
            rows={NODE_B: Entry(2, ("--nsamples=13",)), NODE_C: Entry(2, ())},
            others={NODE_A: Entry(1, ())},
            failed=frozenset({NODE_A, NODE_B, NODE_C}),
        )

    def test_entries_pytest_no_longer_lists_as_failed_are_left_out(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True},
        }

        reuse = self.lf(tmp_path, values)

        assert (reuse.seed, list(reuse.rows), reuse.others) == (1, [NODE_A], {})

    def test_an_entry_whose_file_was_deleted_is_ignored(self, tmp_path):
        files(tmp_path, NODE_A)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        reuse = self.lf(tmp_path, values)

        assert (reuse.seed, list(reuse.rows), reuse.others) == (1, [NODE_A], {})

    @pytest.mark.parametrize(
        "values",
        [
            {},
            {LASTFAILED: {NODE_A: True}},
            {KEY: entries((NODE_A, 1, []))},
            {KEY: entries((NODE_A, 1, [])), LASTFAILED: {NODE_B: True}},
            {KEY: entries((NODE_A, 1, [])), LASTFAILED: [NODE_A]},
        ],
        ids=["empty", "no_map", "no_lastfailed", "other_tests_failed", "lastfailed_list"],
    )
    def test_none_without_an_entry_to_reuse(self, tmp_path, values):
        files(tmp_path, NODE_A, NODE_B)

        assert self.lf(tmp_path, values) is None

    def test_none_without_lf_or_sw(self, tmp_path):
        files(tmp_path, NODE_A)
        values = {KEY: entries((NODE_A, 1, [])), LASTFAILED: {NODE_A: True}}

        assert plan(config(tmp_path, Cache(values), ff=True)) is None

    def test_none_without_a_cache(self, tmp_path):
        stand_in = config(tmp_path, lf=True)
        stand_in.pluginmanager = PluginManager(cacheprovider=False)

        assert plan(stand_in) is None

    @pytest.mark.parametrize(
        "stepwise",
        [{"last_failed": NODE_A, "last_test_count": 3, "last_cache_date_str": "x"}, NODE_A],
        ids=["dict", "text"],
    )
    @pytest.mark.parametrize("option", ["stepwise", "stepwise_skip"])
    def test_sw_reuses_the_seed_of_the_test_it_resumes_from(self, tmp_path, stepwise, option):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
            STEPWISE: stepwise,
        }

        reuse = plan(config(tmp_path, Cache(values), **{option: True}))

        assert (reuse.seed, list(reuse.rows), reuse.others) == (1, [NODE_A], {})
        assert reuse.flag == ("--sw" if option == "stepwise" else "--sw-skip")

    @pytest.mark.parametrize(
        "stepwise",
        [None, {"last_failed": None, "last_test_count": 3, "last_cache_date_str": "x"}, 3],
        ids=["none", "no_last_failed", "other"],
    )
    def test_sw_without_a_test_to_resume_from(self, tmp_path, stepwise):
        files(tmp_path, NODE_A)
        values = {KEY: entries((NODE_A, 1, [])), STEPWISE: stepwise}

        assert plan(config(tmp_path, Cache(values), stepwise=True)) is None

    @pytest.mark.parametrize(
        ("arg", "seed"),
        [
            ("tests/a", 1),
            ("tests/a/test_w.py", 1),
            ("tests/a/test_w.py::test_w", 1),
            ("tests/a/test_w.py::test_w[rand-1]", 1),
            ("tests/b", 2),
            ("tests", 2),
            (".", 2),
        ],
        ids=["folder", "file", "test", "row", "other_folder", "parent", "rootdir"],
    )
    def test_only_the_rows_the_command_line_selects(self, tmp_path, arg, seed):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        reuse = self.lf(tmp_path, values, arg)

        assert reuse.seed == seed
        assert reuse.others == ({} if arg not in ("tests", ".") else {NODE_A: Entry(1, ())})

    def test_a_run_without_paths_selects_the_folder_it_runs_in(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        below = plan(config(tmp_path, Cache(values), invocation=tmp_path / "tests" / "a", lf=True))
        root = plan(config(tmp_path, Cache(values), lf=True))

        assert (below.seed, below.others) == (1, {})
        assert (root.seed, root.others) == (2, {NODE_A: Entry(1, ())})

    def test_a_run_without_paths_selects_the_testpaths(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        reuse = plan(config(tmp_path, Cache(values), testpaths=["tests/a"], lf=True))

        assert (reuse.seed, reuse.others) == (1, {})

    def test_paths_are_relative_to_the_invocation_folder(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        reuse = plan(config(tmp_path, Cache(values), ["a"], tmp_path / "tests", lf=True))

        assert reuse.seed == 1

    def test_a_test_name_selects_only_that_test(self, tmp_path):
        other = "tests/a/test_w.py::test_w_more[rand-0]"
        files(tmp_path, NODE_A, other)
        values = {
            KEY: entries((NODE_A, 1, []), (other, 2, [])),
            LASTFAILED: {NODE_A: True, other: True},
        }

        assert self.lf(tmp_path, values, "tests/a/test_w.py::test_w").seed == 1

    def test_no_selection_when_an_argument_is_not_a_path(self, tmp_path):
        files(tmp_path, NODE_A, NODE_B)
        values = {
            KEY: entries((NODE_A, 1, []), (NODE_B, 2, [])),
            LASTFAILED: {NODE_A: True, NODE_B: True},
        }

        # A --pyargs module name, say
        assert self.lf(tmp_path, values, "tests/a", "acme.tests").seed == 2

    def test_none_when_the_command_line_selects_no_entry(self, tmp_path):
        files(tmp_path, NODE_A)
        (tmp_path / "other").mkdir()
        values = {KEY: entries((NODE_A, 1, [])), LASTFAILED: {NODE_A: True}}

        assert self.lf(tmp_path, values, "other") is None


class TestDifferences:
    def rows(self, *options):
        return [Entry(1, tuple(o)) for o in options]

    def test_none_when_the_options_agree(self):
        recorded = ["--nsamples=13", "-o", "strategies_ids=values", "-c", "ci/pytest.ini"]

        assert differences(self.rows(recorded), recorded, ()) is None

    def test_the_recorded_options_the_run_lacks(self):
        assert differences(self.rows(["--nsamples=13"]), [], ()) == "with --nsamples=13"

    def test_the_run_s_options_that_were_not_recorded(self):
        assert differences(self.rows([]), ["--vector-mode=test"], ()) == (
            "without --vector-mode=test"
        )

    def test_both(self):
        text = differences(
            self.rows(["--nsamples=13", "-o", "strategies_ids=values"]),
            ["--nsamples=13", "-o", "strategies_max_exhaustive=5"],
            (),
        )

        assert text == ("with -o strategies_ids=values, without -o strategies_max_exhaustive=5")

    def test_the_newest_row_s_options(self):
        rows = self.rows(["--nsamples=13"], ["--nsamples=20"])

        assert differences(rows, ["--nsamples=13"], ()) == (
            "with --nsamples=20, without --nsamples=13"
        )

    def test_values_are_quoted(self):
        text = differences(self.rows(["-c", "my ci/pytest.ini"]), [], ())

        assert text in ("with -c 'my ci/pytest.ini'", 'with -c "my ci/pytest.ini"')

    @pytest.mark.parametrize(
        "current_off",
        [[("burst", "aligned")], [(None, "aligned")], [(None, "aligned"), (None, "other")]],
        ids=["for_the_strategy", "everywhere", "and_more"],
    )
    def test_a_constraint_the_run_turns_off_too(self, current_off):
        rows = self.rows(["--strategy-constraint-off=burst:aligned"])

        assert differences(rows, [], current_off) is None

    def test_a_constraint_the_run_leaves_on(self):
        rows = self.rows(["--strategy-constraint-off=burst:aligned,chan:fast"])

        assert differences(rows, [], [("chan", "fast"), ("other", "aligned")]) == (
            "with --strategy-constraint-off=burst:aligned"
        )

    def test_the_constraints_of_every_reused_row(self):
        rows = self.rows(
            ["--strategy-constraint-off=burst:aligned"], ["--strategy-constraint-off=chan:fast"]
        )

        # quote(): Windows quotes the comma, which PowerShell reads as an array
        assert differences(rows, [], ()) == (
            "with " + quote("--strategy-constraint-off=burst:aligned,chan:fast")
        )

    def test_the_run_s_own_constraints_are_not_compared(self):
        assert differences(self.rows([]), [], [(None, "aligned")]) is None


class TestCommands:
    def test_one_command_per_seed_with_its_count(self):
        rows = [
            (NODE_A, Entry(1, ())),
            (NODE_B, Entry(2, ("--nsamples=13",))),
            (NODE_C, Entry(1, ())),
        ]

        assert commands("--lf", rows) == [
            ("pytest --lf --rng-seed=1 tests/a/test_w.py tests/b/test_w.py", 2),
            ("pytest --lf --rng-seed=2 --nsamples=13 tests/b/test_w.py", 1),
        ]

    def test_rows_of_one_seed_with_other_options_get_their_own(self):
        rows = [(NODE_A, Entry(1, ())), (NODE_B, Entry(1, ("--nsamples=13",)))]

        assert commands("--lf", rows) == [
            ("pytest --lf --rng-seed=1 tests/a/test_w.py", 1),
            ("pytest --lf --rng-seed=1 --nsamples=13 tests/b/test_w.py", 1),
        ]

    def test_the_constraints_turned_off_are_joined(self):
        rows = [
            (NODE_B, Entry(1, ("--nsamples=13", "--strategy-constraint-off=burst:aligned"))),
            (
                NODE_C,
                Entry(1, ("--nsamples=13", "--strategy-constraint-off=chan:fast,burst:aligned")),
            ),
        ]

        assert commands("--lf", rows) == [
            (
                "pytest --lf --rng-seed=1 --nsamples=13 "
                + quote("--strategy-constraint-off=burst:aligned,chan:fast")
                + " tests/b/test_w.py",
                2,
            )
        ]

    def test_the_flag_and_quoting(self):
        (found,) = commands(
            "--sw", [("my tests/test_w.py::t", Entry(1, ("-c", "my ci/pytest.ini")))]
        )

        assert found[0] in (
            "pytest --sw --rng-seed=1 -c 'my ci/pytest.ini' 'my tests/test_w.py'",
            'pytest --sw --rng-seed=1 -c "my ci/pytest.ini" "my tests/test_w.py"',
        )

    @pytest.mark.parametrize(
        "other",
        [NODE_C, "tests/b/test_w.py::test_plain", "tests/b/test_w.py"],
        ids=["other_seed", "not_a_strategy_row", "collection_error"],
    )
    def test_a_file_with_other_failed_tests_is_named_by_node_ids(self, other):
        rows = [(NODE_A, Entry(1, ())), (NODE_B, Entry(1, ()))]

        (found,) = commands("--lf", rows, {NODE_A, NODE_B, other})

        assert found == (f"pytest --lf --rng-seed=1 tests/a/test_w.py {quote(NODE_B)}", 2)

    def test_a_file_with_only_its_rows_is_named_once(self):
        rows = [(NODE_B, Entry(1, ())), (NODE_A, Entry(1, ())), (NODE_C, Entry(1, ()))]

        assert commands("--lf", rows, {NODE_A, NODE_B, NODE_C}) == [
            ("pytest --lf --rng-seed=1 tests/b/test_w.py tests/a/test_w.py", 3)
        ]

    def test_paths_are_written_as_given(self):
        rows = [(NODE_A, Entry(1, ())), (NODE_B, Entry(1, ()))]

        (found,) = commands("--lf", rows, {NODE_A, NODE_B, NODE_C}, lambda path: f"../{path}")

        assert found == (
            f"pytest --lf --rng-seed=1 ../tests/a/test_w.py {quote('../' + NODE_B)}",
            2,
        )


class TestDeselectedLines:
    def state(self):
        state = SessionState()
        state.reuse = Reuse(
            flag="--lf",
            seed=3,
            rows={},
            others={NODE_A: Entry(1, ()), NODE_B: Entry(2, ()), NODE_C: Entry(1, ())},
            failed=frozenset({NODE_A, NODE_B, NODE_C}),
        )
        return state

    def config(self, invocation="."):
        root = Path("/project")
        stand_in = Namespace(rootpath=root, invocation_params=Namespace(dir=root / invocation))
        stand_in.cwd_relative_nodeid = partial(pytest.Config.cwd_relative_nodeid, stand_in)
        return stand_in

    def test_a_command_per_seed_of_the_rows_set_aside(self):
        lines = _deselected_lines(self.state(), self.config())

        assert lines == [
            "pytest-strategies: deselected 3 failed rows recorded under other seeds; run them with:",
            f"  pytest --lf --rng-seed=1 tests/a/test_w.py {quote(NODE_C)}  # 2 rows",
            f"  pytest --lf --rng-seed=2 {quote(NODE_B)}  # 1 row",
        ]

    def test_one_row(self):
        state = self.state()
        state.reuse = Reuse("--lf", 3, {}, {NODE_B: Entry(2, ())}, frozenset({NODE_B}))

        assert _deselected_lines(state, self.config()) == [
            "pytest-strategies: deselected 1 failed row recorded under another seed; run it with:",
            "  pytest --lf --rng-seed=2 tests/b/test_w.py  # 1 row",
        ]

    def test_paths_are_relative_to_the_folder_pytest_runs_in(self):
        lines = _deselected_lines(self.state(), self.config("tests/b"))

        assert lines[1:] == [
            # The path is the platform's, as pytest's own bestrelpath writes it
            f"  pytest --lf --rng-seed=1 {quote(str(Path('..', 'a', 'test_w.py')))} "
            f"{quote('test_w.py::test_w[rand-4]')}  # 2 rows",
            f"  pytest --lf --rng-seed=2 {quote('test_w.py::test_w[rand-2]')}  # 1 row",
        ]

    def test_nothing_without_rows_under_another_seed(self):
        state = self.state()
        state.reuse = Reuse("--lf", 3, {NODE_A: Entry(3, ())}, {})

        assert _deselected_lines(state, self.config()) == []

    def test_nothing_without_reuse(self):
        assert _deselected_lines(SessionState(), self.config()) == []


class TestPluginState:
    def test_a_worker_gets_the_rows_from_the_controller(self, tmp_path):
        stand_in = config(tmp_path, Cache({KEY: entries((NODE_A, 7, []))}))
        stand_in.workerinput = {
            "pytest_strategies_deselect": [NODE_B],
            "pytest_strategies_recorded": [NODE_C],
        }
        state = SessionState()
        state.run_seed = 7

        _read_failed_seeds(stand_in, state)

        assert (state.deselect, state.recorded) == ({NODE_B}, {NODE_C})

    def test_the_rows_recorded_under_the_run_s_seed(self, tmp_path):
        cache = Cache({KEY: entries((NODE_A, 7, []), (NODE_B, 8, []), (NODE_C, 7, ["-x"]))})
        state = SessionState()
        state.run_seed = 7
        state.reuse = Reuse("--lf", 7, {}, {NODE_B: Entry(8, ())})

        _read_failed_seeds(config(tmp_path, cache), state)

        assert (state.deselect, state.recorded) == ({NODE_B}, {NODE_A, NODE_C})

    def test_nothing_without_a_cache(self, tmp_path):
        state = SessionState()

        _read_failed_seeds(config(tmp_path), state)

        assert (state.deselect, state.recorded) == (set(), set())

    def test_the_map_is_updated_with_the_workers_rows(self, tmp_path):
        cache = Cache({KEY: entries((NODE_A, 7, []), (NODE_B, 7, ["--nsamples=13"]))})
        state = SessionState()
        state.run_seed = 7
        state.failed_rows = {NODE_C: row(7)}
        state.worker_reuse = {
            "gw0": {"passed": {NODE_A: []}},
            "gw1": {"passed": {NODE_B: [], "x": "not a list"}},
        }

        _record_failed_seeds(config(tmp_path, cache), state)

        assert json.loads(cache.values[KEY]) == entries(
            (NODE_B, 7, ["--nsamples=13"]), (NODE_C, 7, [])
        )

    def test_the_map_is_written_only_when_it_changed(self, tmp_path):
        cache = Cache({KEY: entries((NODE_A, 7, []))})
        state = SessionState()
        state.run_seed = 7
        state.passed_rows = {NODE_A: ("--nsamples=13",)}

        _record_failed_seeds(config(tmp_path, cache), state)
        state.passed_rows = {}
        _record_failed_seeds(config(tmp_path, cache), state)

        assert cache.writes == 0

    def test_a_run_without_failures_writes_no_map(self, tmp_path):
        cache = Cache()
        state = SessionState()

        _record_failed_seeds(config(tmp_path, cache), state)

        assert KEY not in cache.values

    def test_nothing_is_written_without_a_cache(self, tmp_path):
        state = SessionState()
        state.failed_rows = {NODE_A: row(7)}

        assert _record_failed_seeds(config(tmp_path), state) is None

    def test_a_worker_sends_its_rows_and_writes_no_map(self, tmp_path):
        cache = Cache({KEY: entries((NODE_B, 7, []))})
        stand_in = config(tmp_path, cache)
        stand_in.workeroutput = {}
        state = runtime.push(stand_in)
        try:
            state.run_seed = 7
            state.failed_rows = {NODE_A: row(7)}
            state.passed_rows = {NODE_B: ()}

            _plugin_instance.pytest_sessionfinish(
                Namespace(config=stand_in, items=[], exitstatus=0)
            )
        finally:
            runtime.pop()

        assert stand_in.workeroutput["pytest_strategies_reuse"] == {"passed": {NODE_B: []}}
        assert (cache.writes, json.loads(cache.values[KEY])) == (0, entries((NODE_B, 7, [])))
