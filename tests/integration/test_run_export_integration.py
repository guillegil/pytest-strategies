"""
The run export (--strategy-export, the strategies_export_dir ini option and the
pytest_strategies_export_dir hook) and Parameter(pin_nsamples=True), end to end.
"""

import json
import os
import shlex
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
    from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

    @register("rw")
    def rw(device="esm", width=8):
        return Parameter(
            TestArg("channel", rng_type=Series([0, 1])),
            TestArg("data", rng_type=RNGInteger(0, 2**width - 1)),
            directed_vectors={"zeros": (0, 0)},
            test_vectors={"max": (1, 2**width - 1)},
        )

    @register("pinned")
    def pinned():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("v", rng_type=RNGInteger(0, 9)),
            nsamples="auto",
            pin_nsamples=True,
        )
    """

TESTS = """
    import pytest
    from pytest_strategy import strategy

    @strategy("rw", device="bpam", width=4)
    def test_rw(channel, data, request):
        assert "zeros" not in request.node.name

    @strategy("pinned")
    def test_pinned(ch, v):
        pass

    @pytest.mark.xfail(reason="known")
    @strategy("pinned")
    def test_xfail(ch, v):
        assert False

    def test_plain():
        pass
    """


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_test(document):
    return {test["nodeid"]: test["strategies"] for test in document["tests"]}


@pytest.fixture
def suite(pytester):
    pytester.makepyfile(rx_strategies=STRATEGIES)
    pytester.makepyfile(test_rx=TESTS)
    return pytester


class TestFiles:
    def test_both_files_are_written_with_definitions_and_rows(self, suite):
        result = suite.runpytest_inprocess("--strategy-export=out", "--nsamples=4", "--rng-seed=1")
        result.assert_outcomes(passed=8, failed=1, xfailed=3)
        result.stdout.fnmatch_lines(
            ["pytest-strategies: wrote strategies.json and strategies.vectors.json to *out"]
        )
        definitions = _load(suite.path / "out" / "strategies.json")
        vectors = _load(suite.path / "out" / "strategies.vectors.json")
        assert definitions["schema"] == vectors["schema"] == 1
        assert definitions["kind"] == "strategy_definitions"
        assert vectors["kind"] == "strategy_vectors"
        assert definitions["seed"] == 1 and definitions["nsamples"] == 4
        # Only the tests with a strategy, in collection order
        assert list(_by_test(definitions)) == [
            "test_rx.py::test_rw",
            "test_rx.py::test_pinned",
            "test_rx.py::test_xfail",
        ]
        (rw,) = _by_test(definitions)["test_rx.py::test_rw"]
        assert rw["name"] == "rw"
        assert rw["origin"]["file"] == "rx_strategies.py"
        assert rw["arguments"] == {"device": "bpam", "width": 4}
        assert rw["nsamples"] == {"value": 4, "source": "--nsamples"}
        assert "rows" not in rw
        parameter = rw["parameter"]
        # The definition as the test resolved it: width=4 gives max 15
        assert parameter["arguments"][1]["rng"] == {
            "type": "RNGInteger",
            "min": 0,
            "max": 15,
            "predicate": False,
        }
        assert parameter["directed_vectors"] == [
            {"name": "zeros", "id": "directed-zeros", "values": {"channel": 0, "data": 0}}
        ]
        assert parameter["test_vectors"] == [
            {"name": "max", "id": "test-max", "values": {"channel": 1, "data": 15}}
        ]
        # The vectors file is the same document with the rows
        (rw_rows,) = _by_test(vectors)["test_rx.py::test_rw"]
        assert {k: v for k, v in rw_rows.items() if k != "rows"} == rw
        rows = rw_rows["rows"]
        assert [row["id"] for row in rows] == [
            "directed-zeros",
            "channel=0-rand-0",
            "channel=1-rand-0",
            "channel=0-rand-1",
            "channel=1-rand-1",
        ]
        assert rows[0]["kind"] == "directed" and rows[0]["name"] == "zeros"
        assert rows[0]["items"] == [
            {
                "nodeid": "test_rx.py::test_rw[directed-zeros]",
                "outcome": "failed",
                "rerun": {
                    "args": ["test_rx.py::test_rw[directed-zeros]", "--rng-seed=1", "--nsamples=4"]
                },
            }
        ]
        assert {row["kind"] for row in rows[1:]} == {"random"}
        for row in rows[1:]:
            assert 0 <= row["values"]["data"] <= 15
            assert [item["outcome"] for item in row["items"]] == ["passed"]

    def test_pinned_auto_and_xfail_outcomes(self, suite):
        suite.runpytest_inprocess("--strategy-export=out", "--nsamples=4")
        vectors = _by_test(_load(suite.path / "out" / "strategies.vectors.json"))
        (pinned,) = vectors["test_rx.py::test_pinned"]
        assert pinned["nsamples"] == {"value": "auto", "source": "Parameter(nsamples=, pinned)"}
        assert pinned["parameter"]["pin_nsamples"] is True
        assert [row["id"] for row in pinned["rows"]] == ["ch=0", "ch=1", "ch=2"]
        assert {row["kind"] for row in pinned["rows"]} == {"exhaustive"}
        (xfail,) = vectors["test_rx.py::test_xfail"]
        outcomes = [item["outcome"] for row in xfail["rows"] for item in row["items"]]
        assert outcomes == ["xfailed"] * 3

    def test_test_vectors_run_in_test_mode(self, suite):
        suite.runpytest_inprocess("--strategy-export=out", "--vector-mode=test")
        vectors = _by_test(_load(suite.path / "out" / "strategies.vectors.json"))
        (rw,) = vectors["test_rx.py::test_rw"]
        assert [(row["id"], row["kind"], row["values"]) for row in rw["rows"]] == [
            ("test-max", "test", {"channel": 1, "data": 15})
        ]

    def test_only_selected_items_and_no_outcome_without_a_run(self, suite):
        suite.runpytest_inprocess(
            "--strategy-export=out", "--collect-only", "-k", "zeros", "--rng-seed=3"
        )
        vectors = _load(suite.path / "out" / "strategies.vectors.json")
        (test,) = vectors["tests"]
        assert test["nodeid"] == "test_rx.py::test_rw"
        (row,) = test["strategies"][0]["rows"]
        assert row["items"] == [
            {
                "nodeid": "test_rx.py::test_rw[directed-zeros]",
                "outcome": None,
                # --collect-only and -k are left out of a command that names the
                # row and runs it
                "rerun": {"args": ["test_rx.py::test_rw[directed-zeros]", "--rng-seed=3"]},
            }
        ]

    def test_nothing_is_written_without_the_option(self, suite):
        result = suite.runpytest_inprocess()
        assert "strategies.json" not in result.stdout.str()
        assert not list(suite.path.rglob("strategies.json"))

    def test_an_error_writing_is_reported(self, suite):
        (suite.path / "blocked").write_text("a file, not a folder")
        result = suite.runpytest_inprocess("--strategy-export=blocked/out")
        result.stdout.fnmatch_lines(
            ["pytest-strategies: could not write the strategy export to *blocked*out: *"]
        )


class TestWhere:
    def test_ini_option_is_relative_to_the_rootdir(self, suite):
        suite.makeini("[pytest]\nstrategies_export_dir = reports/strategies\n")
        sub = suite.mkdir("sub")
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(sub)
            suite.runpytest_inprocess(str(suite.path))
        assert not (sub / "reports").exists()
        assert (suite.path / "reports" / "strategies" / "strategies.json").is_file()

    def test_hook_from_conftest_wins_over_the_ini_option(self, suite):
        suite.makeini("[pytest]\nstrategies_export_dir = from_ini\n")
        suite.makeconftest("""
            import pytest

            @pytest.hookimpl(optionalhook=True)
            def pytest_strategies_export_dir(config):
                return "from_hook"
            """)
        suite.runpytest_inprocess()
        assert (suite.path / "from_hook" / "strategies.vectors.json").is_file()
        assert not (suite.path / "from_ini").exists()

    def test_the_option_wins_over_the_hook(self, suite):
        suite.makeconftest("""
            import pytest

            @pytest.hookimpl(optionalhook=True)
            def pytest_strategies_export_dir(config):
                return "from_hook"
            """)
        suite.runpytest_inprocess("--strategy-export=from_option")
        assert (suite.path / "from_option" / "strategies.json").is_file()
        assert not (suite.path / "from_hook").exists()

    def test_conftest_can_set_the_option(self, suite):
        suite.makeconftest("""
            def pytest_configure(config):
                config.option.strategy_export = "from_configure"
            """)
        suite.runpytest_inprocess()
        assert (suite.path / "from_configure" / "strategies.json").is_file()


class TestXdist:
    def test_the_controller_writes_the_workers_rows_with_outcomes(self, suite):
        pytest.importorskip("xdist")
        local = suite.runpytest_inprocess("--strategy-export=local", "--rng-seed=7")
        local.assert_outcomes(passed=14, failed=1, xfailed=3)
        result = suite.runpytest_subprocess(
            "--strategy-export=dist", "--rng-seed=7", "-n", "2", "-p", "no:cacheprovider"
        )
        result.assert_outcomes(passed=14, failed=1, xfailed=3)
        dist = _load(suite.path / "dist" / "strategies.vectors.json")
        same = _load(suite.path / "local" / "strategies.vectors.json")
        # The workers made the commands, with the options the controller was started
        # with (theirs are only the paths): those apart, they are the local run's
        carried = _carried(dist, "-n", "2", "-p", "no:cacheprovider")
        assert same["rerun"]["options"] == ["--rng-seed=7"]
        assert dist["rerun"]["options"] == ["--rng-seed=7", *carried]
        local_tests = json.loads(json.dumps(same["tests"]))
        for test in local_tests:
            reruns = [test["rerun"]]
            for used in test["strategies"]:
                reruns += [item["rerun"] for row in used["rows"] for item in row["items"]]
            for rerun in reruns:
                rerun["args"] += carried
        assert dist["tests"] == local_tests
        assert dist["rerun"]["cwd"] == str(suite.path)
        assert (
            dist["rerun"]["args"][-5:]
            == [
                "--strategy-export=dist",
                "--rng-seed=7",
                "-n",
                "2",
                "-p",
                "no:cacheprovider",
            ][-5:]
        )
        assert dist["rerun"]["args"].count("--rng-seed=7") == 1
        definitions = _load(suite.path / "dist" / "strategies.json")
        assert definitions["rerun"] == dist["rerun"]
        assert [test["rerun"] for test in definitions["tests"]] == [
            {"args": [f"test_rx.py::{name}", "--rng-seed=7", *carried]}
            for name in ("test_rw", "test_pinned", "test_xfail")
        ]


RERUN_STRATEGIES = """
    from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

    @register("word")
    def word():
        return Parameter(
            TestArg("data", rng_type=RNGInteger(0, 10**9)),
            directed_vectors={"zeros": (0,)},
            vector_constraints={"even": lambda v: v.data % 2 == 0},
        )

    @register("lane")
    def lane():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("gain", rng_type=RNGInteger(0, 10**9)),
            nsamples="auto",
            pin_nsamples=True,
        )
    """

RERUN_TESTS = """
    from pytest_strategy import strategy

    @strategy("word")
    def test_word(data):
        assert False

    @strategy("lane")
    @strategy("word")
    def test_both(data, ch, gain):
        assert False

    class TestGroup:
        @strategy("lane")
        def test_lane(self, ch, gain):
            pass
    """


def _values(document):
    """Each row's values, by test, strategy and row ID."""
    return {
        (test["nodeid"], strategy["name"], row["id"]): row["values"]
        for test in document["tests"]
        for strategy in test["strategies"]
        for row in strategy["rows"]
    }


def _items(document):
    """Every item of the export, by node ID (the same for each strategy of its test)."""
    return {
        item["nodeid"]: item
        for test in document["tests"]
        for strategy in test["strategies"]
        for row in strategy["rows"]
        for item in row["items"]
    }


def _tests(document):
    return {test["nodeid"]: test for test in document["tests"]}


def _printed_commands(result):
    """The rerun commands of the list of failed rows, as the terminal printed them."""
    lines = result.stdout.lines
    start = lines.index("pytest-strategies: failed rows:")
    found = []
    for line in lines[start + 1 :]:
        if not line.startswith("  pytest "):
            break
        found.append(line.strip().rpartition("  # ")[0])
    return found


def _carried(document, *options):
    """
    The options an exported command for chosen rows repeats from the run's own:
    the ``--basetemp`` pytester gives a subprocess, then ``options``.
    """
    basetemp = [arg for arg in document["rerun"]["args"] if arg.startswith("--basetemp=")]
    return [*basetemp, *options]


def _pytest(pytester, cwd, args):
    """Run ``pytest <args>`` from ``cwd`` in a subprocess, as a reader of the export does."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return pytester.run(sys.executable, "-m", "pytest", *args)
    finally:
        os.chdir(previous)


@pytest.fixture
def rerun_suite(pytester):
    pytester.makepyfile(rr_strategies=RERUN_STRATEGIES)
    pytester.makepyfile(test_rr=RERUN_TESTS)
    return pytester


posix_quoting = pytest.mark.skipif(os.name == "nt", reason="shlex splits POSIX commands")


class TestRerun:
    def test_the_run_s_command_and_options_are_in_both_files(self, rerun_suite):
        rerun_suite.runpytest_subprocess(
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=2",
            "--strategy-constraint-off=word:even",
            "-p",
            "no:cacheprovider",
        )
        definitions = _load(rerun_suite.path / "out" / "strategies.json")
        vectors = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        rerun = vectors["rerun"]
        assert definitions["rerun"] == rerun
        assert rerun["cwd"] == str(rerun_suite.path)
        assert os.path.isabs(rerun["cwd"])
        # The arguments as given: the seed is there already, and is not added again
        assert rerun["args"][-6:] == [
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=2",
            "--strategy-constraint-off=word:even",
            "-p",
            "no:cacheprovider",
        ]
        assert [arg for arg in rerun["args"] if "rng-seed" in arg] == ["--rng-seed=5"]
        # The seed first, then the generation options, the constraints by strategy,
        # then the run's other options: not its --strategy-export, which would
        # write over this export
        carried = _carried(vectors, "-p", "no:cacheprovider")
        assert rerun["options"] == [
            "--rng-seed=5",
            "--nsamples=2",
            "--strategy-constraint-off=word:even",
            *carried,
        ]
        assert "note" not in rerun
        # A test selected as a whole is run again by its function's node ID
        assert {nodeid: test["rerun"] for nodeid, test in _tests(vectors).items()} == {
            "test_rr.py::test_word": {
                "args": ["test_rr.py::test_word", *rerun["options"]],
            },
            "test_rr.py::test_both": {
                "args": ["test_rr.py::test_both", *rerun["options"]],
            },
            # lane has no constraint named even
            "test_rr.py::TestGroup::test_lane": {
                "args": [
                    "test_rr.py::TestGroup::test_lane",
                    "--rng-seed=5",
                    "--nsamples=2",
                    *carried,
                ],
            },
        }
        assert {n: t["rerun"] for n, t in _tests(definitions).items()} == {
            n: t["rerun"] for n, t in _tests(vectors).items()
        }

    @posix_quoting
    def test_an_item_s_args_are_the_printed_command_s_and_the_run_s_options(self, rerun_suite):
        sub = rerun_suite.mkdir("sub")
        os.chdir(sub)
        result = rerun_suite.runpytest_subprocess(
            "..",
            "--rootdir=..",
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=2",
            "--vector-mode=all",
            "--strategy-constraint-off=even",
            "-v",
            "-p",
            "no:cacheprovider",
        )
        result.assert_outcomes(failed=9, passed=2)
        vectors = _load(sub / "out" / "strategies.vectors.json")
        items = _items(vectors)
        failed = [item for item in items.values() if item["outcome"] == "failed"]
        printed = [shlex.split(command) for command in _printed_commands(result)]
        assert len(printed) == len(failed) == 9
        assert all(command[0] == "pytest" for command in printed)
        # The path the run started from is replaced by the row, and what the
        # command gives itself is not repeated (--vector-mode=all is the default)
        carried = _carried(vectors, "-v", "-p", "no:cacheprovider")
        assert sorted([*command[1:], *carried] for command in printed) == sorted(
            item["rerun"]["args"] for item in failed
        )
        # Relative to the folder pytest was started in, with the row's own constraints
        assert items["test_rr.py::test_both[rand-1-ch=0]"]["rerun"] == {
            "args": [
                "../test_rr.py::test_both[rand-1-ch=0]",
                "--rng-seed=5",
                "--nsamples=2",
                "--rootdir=..",
                "--strategy-constraint-off=word:even",
                *carried,
            ]
        }
        assert items["test_rr.py::TestGroup::test_lane[ch=1]"]["rerun"] == {
            "args": [
                "../test_rr.py::TestGroup::test_lane[ch=1]",
                "--rng-seed=5",
                "--nsamples=2",
                "--rootdir=..",
                *carried,
            ]
        }
        assert vectors["rerun"]["cwd"] == str(sub)
        assert vectors["rerun"]["options"] == [
            "--rng-seed=5",
            "--nsamples=2",
            "--rootdir=..",
            "--strategy-constraint-off=word:even",
            *carried,
        ]

    def test_deselection_leaves_only_the_selected_rows_in_a_test_s_args(self, rerun_suite):
        rerun_suite.runpytest_subprocess(
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=3",
            "-k",
            "test_word and (rand-0 or rand-2) or test_lane",
            "-p",
            "no:cacheprovider",
        )
        vectors = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        tests = _tests(vectors)
        # The run's -k is not repeated: the commands name the rows it selected
        carried = _carried(vectors, "-p", "no:cacheprovider")
        assert vectors["rerun"]["options"] == ["--rng-seed=5", "--nsamples=3", *carried]
        assert list(tests) == ["test_rr.py::test_word", "test_rr.py::TestGroup::test_lane"]
        assert tests["test_rr.py::test_word"]["rerun"] == {
            "args": [
                "test_rr.py::test_word[rand-0]",
                "test_rr.py::test_word[rand-2]",
                "--rng-seed=5",
                "--nsamples=3",
                *carried,
            ]
        }
        assert tests["test_rr.py::TestGroup::test_lane"]["rerun"] == {
            "args": ["test_rr.py::TestGroup::test_lane", "--rng-seed=5", "--nsamples=3", *carried]
        }

    def test_rows_chosen_by_node_id_are_named_by_item(self, rerun_suite):
        rerun_suite.runpytest_subprocess(
            "test_rr.py::test_both[rand-0-ch=0]",
            "test_rr.py::test_both[rand-1-ch=1]",
            "test_rr.py::TestGroup",
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=2",
            "-p",
            "no:cacheprovider",
        )
        vectors = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        tests = _tests(vectors)
        # The node IDs the run started from are not among the options it repeats
        carried = _carried(vectors, "-p", "no:cacheprovider")
        assert vectors["rerun"]["options"] == ["--rng-seed=5", "--nsamples=2", *carried]
        # Every row of both strategies ran, but not every item of the function
        assert tests["test_rr.py::test_both"]["rerun"] == {
            "args": [
                "test_rr.py::test_both[rand-0-ch=0]",
                "test_rr.py::test_both[rand-1-ch=1]",
                "--rng-seed=5",
                "--nsamples=2",
                *carried,
            ]
        }
        assert tests["test_rr.py::TestGroup::test_lane"]["rerun"] == {
            "args": ["test_rr.py::TestGroup::test_lane", "--rng-seed=5", "--nsamples=2", *carried]
        }

    def test_collect_only_has_the_commands(self, rerun_suite):
        rerun_suite.runpytest_subprocess(
            "--collect-only", "--strategy-export=out", "--rng-seed=5", "-p", "no:cacheprovider"
        )
        vectors = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        assert "--collect-only" in vectors["rerun"]["args"]
        # A command for chosen rows runs them: it leaves --collect-only out
        carried = _carried(vectors, "-p", "no:cacheprovider")
        item = _items(vectors)["test_rr.py::test_word[rand-3]"]
        assert item == {
            "nodeid": "test_rr.py::test_word[rand-3]",
            "outcome": None,
            "rerun": {"args": ["test_rr.py::test_word[rand-3]", "--rng-seed=5", *carried]},
        }
        assert _tests(vectors)["test_rr.py::test_word"]["rerun"] == {
            "args": ["test_rr.py::test_word", "--rng-seed=5", *carried]
        }

    @pytest.mark.parametrize("which", ["item", "test"])
    def test_the_args_give_the_same_values(self, rerun_suite, which):
        # No seed: the run chose one, which the commands carry
        rerun_suite.runpytest_subprocess(
            "--strategy-export=out", "--nsamples=3", "-k", "not rand-1", "-p", "no:cacheprovider"
        )
        first = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        if which == "item":
            nodeid = "test_rr.py::test_both[rand-2-ch=1]"
            args = _items(first)[nodeid]["rerun"]["args"]
            ran = [nodeid]
        else:
            args = _tests(first)["test_rr.py::test_word"]["rerun"]["args"]
            ran = [
                f"test_rr.py::test_word[{row}]" for row in ("directed-zeros", "rand-0", "rand-2")
            ]
        _pytest(rerun_suite, first["rerun"]["cwd"], [*args, "--strategy-export=again"])
        again = _load(rerun_suite.path / "again" / "strategies.vectors.json")
        assert again["seed"] == first["seed"]
        assert sorted(_items(again)) == sorted(ran)
        values = _values(again)
        assert values
        assert values == {key: _values(first)[key] for key in values}

    @pytest.mark.parametrize(
        "conftest",
        [
            None,
            "def pytest_configure(config):\n    config.option.rng_seed = 99\n",
            "from pytest_strategy import RNG\n\ndef pytest_configure(config):\n    RNG.seed(1234)\n",
        ],
        ids=["no_seed", "conftest_option", "conftest_rng_seed"],
    )
    def test_the_run_s_args_give_the_same_rows(self, rerun_suite, conftest):
        if conftest is not None:
            rerun_suite.makeconftest(conftest)
        sub = rerun_suite.mkdir("sub")
        os.chdir(sub)
        rerun_suite.runpytest_subprocess(
            "..", "--rootdir=..", "--strategy-export=out", "--nsamples=3", "-p", "no:cacheprovider"
        )
        first = _load(sub / "out" / "strategies.vectors.json")
        rerun = first["rerun"]
        seed = first["seed"]
        if conftest is not None:
            assert seed == (99 if "rng_seed" in conftest else 1234)
        # The run did not give the seed: it is added, once
        assert rerun["args"][-1] == f"--rng-seed={seed}"
        assert [arg for arg in rerun["args"] if "rng-seed" in arg] == [f"--rng-seed={seed}"]
        os.chdir(rerun_suite.path)
        # A later --strategy-export wins, so the first export stays to compare with
        _pytest(rerun_suite, rerun["cwd"], [*rerun["args"], "--strategy-export=again"])
        again = _load(sub / "again" / "strategies.vectors.json")
        assert again["seed"] == seed
        assert _values(again) == _values(first)
        assert len(_values(first)) == 4 + 4 + 2 + 2
        assert {n: i["rerun"] for n, i in _items(again).items()} == {
            n: i["rerun"] for n, i in _items(first).items()
        }
        assert again["rerun"]["args"] == [*rerun["args"], "--strategy-export=again"]

    def test_a_seed_reused_for_lf_is_given_explicitly(self, rerun_suite):
        rerun_suite.runpytest_subprocess("--rng-seed=41", "--nsamples=2")
        rerun_suite.runpytest_subprocess("--lf", "--strategy-export=out", "--nsamples=2")
        vectors = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        assert vectors["seed"] == 41
        assert vectors["rerun"]["args"][-4:] == [
            "--lf",
            "--strategy-export=out",
            "--nsamples=2",
            "--rng-seed=41",
        ]
        # Commands for chosen rows name them: --lf, which selects by the cache, is
        # left out of what they repeat, and the run's own command says what it does
        assert vectors["rerun"]["options"] == ["--rng-seed=41", "--nsamples=2", *_carried(vectors)]
        assert vectors["rerun"]["note"].startswith(
            "--lf selects tests by what the cache holds when the command runs"
        )

    def test_the_run_s_command_has_the_environment_s_options(self, rerun_suite, monkeypatch):
        # A CI job's PYTEST_ADDOPTS: without it, the command read elsewhere would
        # run the default number of rows
        monkeypatch.setenv("PYTEST_ADDOPTS", "--nsamples=6 -p no:cacheprovider")
        rerun_suite.runpytest_subprocess("--strategy-export=out", "-k", "test_word")
        monkeypatch.delenv("PYTEST_ADDOPTS")
        first = _load(rerun_suite.path / "out" / "strategies.vectors.json")
        rerun = first["rerun"]
        assert rerun["args"][:3] == ["--nsamples=6", "-p", "no:cacheprovider"]
        assert rerun["args"][-1] == f"--rng-seed={first['seed']}"
        assert len(_items(first)) == 7
        _pytest(rerun_suite, rerun["cwd"], [*rerun["args"], "--strategy-export=again"])
        again = _load(rerun_suite.path / "again" / "strategies.vectors.json")
        assert _values(again) == _values(first)
        assert sorted(_items(again)) == sorted(_items(first))

    @pytest.mark.parametrize("which", ["item", "test", "options"])
    def test_an_option_that_shapes_the_context_is_repeated(self, pytester, which):
        # The factory draws from a range the project's own option sets: a command
        # without the option would run the row with other values
        pytester.makeconftest("""
            def pytest_addoption(parser):
                parser.addoption("--width", type=int, default=8)

            def pytest_strategies_context(config):
                return {"width": config.getoption("--width")}
            """)
        pytester.makepyfile(cx_strategies="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("sized")
            def sized(ctx):
                return Parameter(TestArg("data", rng_type=RNGInteger(0, (1 << ctx["width"]) - 1)))
            """)
        pytester.makepyfile(test_cx="""
            from pytest_strategy import strategy

            @strategy("sized")
            def test_sized(data):
                pass
            """)
        pytester.runpytest_subprocess(
            "test_cx.py",
            "--width",
            "40",
            "--strategy-export=out",
            "--nsamples=4",
            "-p",
            "no:cacheprovider",
        )
        first = _load(pytester.path / "out" / "strategies.vectors.json")
        carried = _carried(first, "--width", "40", "-p", "no:cacheprovider")
        assert first["rerun"]["options"] == [
            f"--rng-seed={first['seed']}",
            "--nsamples=4",
            *carried,
        ]
        nodeid = "test_cx.py::test_sized[rand-2]"
        if which == "item":
            args = _items(first)[nodeid]["rerun"]["args"]
            assert args == [nodeid, *first["rerun"]["options"]]
            ran = 1
        elif which == "test":
            args = _tests(first)["test_cx.py::test_sized"]["rerun"]["args"]
            assert args == ["test_cx.py::test_sized", *first["rerun"]["options"]]
            ran = 4
        else:
            # As a reader builds a command for items it chose
            args = [nodeid, "test_cx.py::test_sized[rand-0]", *first["rerun"]["options"]]
            ran = 2
        _pytest(pytester, first["rerun"]["cwd"], [*args, "--strategy-export=again"])
        again = _load(pytester.path / "again" / "strategies.vectors.json")
        values = _values(again)
        assert len(values) == ran
        assert values == {key: _values(first)[key] for key in values}
        # Values the default width cannot give
        assert max(row["data"] for row in _values(first).values()) > 255


class TestRerunOutsideTheRootdir:
    """A test file outside the rootdir (``-c ci/pytest.ini`` makes ``ci`` the rootdir)."""

    @staticmethod
    def project(pytester, extra=""):
        (pytester.path / "ci").mkdir()
        (pytester.path / "ci" / "pytest.ini").write_text(f"[pytest]\n{extra}", encoding="utf-8")
        tests = pytester.mkdir("tests")
        (tests / "strategies.py").write_text(
            "from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register\n\n"
            "@register('word')\n"
            "def word():\n"
            "    return Parameter(TestArg('data', rng_type=RNGInteger(0, 10**9)))\n\n"
            "@register('lane')\n"
            "def lane():\n"
            "    return Parameter(TestArg('ch', rng_type=Series([0, 1])), nsamples='auto',"
            " pin_nsamples=True)\n",
            encoding="utf-8",
        )
        (tests / "test_out.py").write_text(
            "from pytest_strategy import strategy\n\n"
            "@strategy('word')\n"
            "def test_word(data):\n"
            "    assert False\n\n"
            "@strategy('lane')\n"
            "def test_lane(ch):\n"
            "    assert False\n",
            encoding="utf-8",
        )
        return tests

    @posix_quoting
    def test_the_k_form_as_printed_and_the_note(self, pytester):
        self.project(pytester)
        result = pytester.runpytest_subprocess(
            "-c",
            "ci/pytest.ini",
            "--strategy-export=out",
            "--rng-seed=5",
            "--nsamples=2",
            "-p",
            "no:cacheprovider",
        )
        result.assert_outcomes(failed=4)
        vectors = _load(pytester.path / "out" / "strategies.vectors.json")
        items = _items(vectors)
        carried = _carried(vectors, "-p", "no:cacheprovider")
        printed = [shlex.split(command)[1:] for command in _printed_commands(result)]
        # The printed command, with the run's other options before its -k (or last)
        expected = []
        for command in printed:
            at = command.index("-k") if "-k" in command else len(command)
            expected.append([*command[:at], *carried, *command[at:]])
        assert sorted(expected) == sorted(item["rerun"]["args"] for item in items.values())
        options = ["--rng-seed=5", "--nsamples=2", "-c", "ci/pytest.ini", *carried]
        assert vectors["rerun"]["options"] == options
        assert items["tests/test_out.py::test_word[rand-1]"]["rerun"] == {
            "args": [".", *options, "-k", "test_word[rand-1]"]
        }
        # ch=0 is not a name -k can hold: the node ID command, which does not reproduce
        lane = items["tests/test_out.py::test_lane[ch=0]"]["rerun"]
        assert lane["args"] == [str(Path("tests", "test_out.py")) + "::test_lane[ch=0]", *options]
        assert lane["note"].startswith("the test is outside the rootdir (ci), so pytest names it")
        tests = _tests(vectors)
        assert tests["tests/test_out.py::test_word"]["rerun"] == {
            "args": [".", *options, "-k", "test_word[rand-0] or test_word[rand-1]"]
        }
        whole = tests["tests/test_out.py::test_lane"]["rerun"]
        assert whole["args"] == [str(Path("tests", "test_out.py")) + "::test_lane", *options]
        assert whole["note"] == lane["note"]

    def test_the_k_form_gives_the_same_values(self, pytester):
        self.project(pytester)
        pytester.runpytest_subprocess(
            "-c", "ci/pytest.ini", "--strategy-export=out", "--nsamples=3", "-p", "no:cacheprovider"
        )
        first = _load(pytester.path / "out" / "strategies.vectors.json")
        for args, ran in (
            (_items(first)["tests/test_out.py::test_word[rand-2]"]["rerun"]["args"], 1),
            (_tests(first)["tests/test_out.py::test_word"]["rerun"]["args"], 3),
            (first["rerun"]["args"], 5),
        ):
            _pytest(pytester, first["rerun"]["cwd"], [*args, "--strategy-export=again"])
            again = _load(pytester.path / "again" / "strategies.vectors.json")
            assert len(_items(again)) == ran
            values = _values(again)
            assert values == {key: _values(first)[key] for key in values}


class TestPinNsamplesValidation:
    def test_needs_a_count(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        with pytest.raises(ValueError, match="pin_nsamples=True needs nsamples"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), pin_nsamples=True)

    def test_must_be_a_bool(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        with pytest.raises(ValueError, match="pin_nsamples must be a bool"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), nsamples=2, pin_nsamples=1)

    def test_extend_keeps_it(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        base = Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), nsamples=2, pin_nsamples=True)
        assert base.extend(nsamples=5).pin_nsamples is True
        assert base.extend(pin_nsamples=False).pin_nsamples is False
        assert base.to_dict()["pin_nsamples"] is True


class TestPinNsamples:
    STRATEGY = """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register

        @register("{name}")
        def factory():
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), {settings})
        """
    TEST = """
        from pytest_strategy import strategy

        @strategy("{name}")
        def test_x(x):
            pass
        """

    def _suite(self, pytester, name, settings):
        pytester.makepyfile(
            **{f"{name}_strategies": self.STRATEGY.format(name=name, settings=settings)}
        )
        pytester.makepyfile(**{f"test_{name}": self.TEST.format(name=name)})

    def test_a_pinned_count_ignores_the_option(self, pytester):
        self._suite(pytester, "pin_a", "nsamples=3, pin_nsamples=True")
        pytester.runpytest_inprocess("--nsamples=50").assert_outcomes(passed=3)
        pytester.runpytest_inprocess("--nsamples=auto").assert_outcomes(passed=3)

    def test_an_unpinned_count_follows_the_option(self, pytester):
        self._suite(pytester, "pin_b", "nsamples=3")
        pytester.runpytest_inprocess("--nsamples=5").assert_outcomes(passed=5)

    def test_pinned_auto_stays_auto_under_an_integer_option(self, pytester):
        pytester.makepyfile(pin_c_strategies="""
            from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

            @register("pin_c")
            def factory():
                return Parameter(
                    TestArg("ch", rng_type=Series([0, 1])),
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    nsamples="auto",
                    pin_nsamples=True,
                )
            """)
        pytester.makepyfile(test_pin_c="""
            from pytest_strategy import strategy

            @strategy("pin_c")
            def test_x(ch, x):
                pass
            """)
        result = pytester.runpytest_inprocess("--nsamples=50", "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*test_x?ch=0? PASSED*", "*test_x?ch=1? PASSED*"])
        result.stdout.fnmatch_lines(["*Parameter(nsamples=, pinned)*"])

    def test_vector_name_still_applies(self, pytester):
        self._suite(
            pytester, "pin_d", 'nsamples=3, pin_nsamples=True, directed_vectors={"z": (0,)}'
        )
        result = pytester.runpytest_inprocess("--vector-name=z")
        result.assert_outcomes(passed=1)


class TestIniValue:
    def test_a_toml_value_that_is_not_a_string_stops_the_run(self, pytester):
        pytester.makefile(".toml", pytest="[pytest]\nstrategies_export_dir = 1\n")
        result = pytester.runpytest("-p", "no:cacheprovider")
        if result.ret != pytest.ExitCode.USAGE_ERROR:
            pytest.skip("this pytest reads a TOML number as a string option")
        # pytest 9.2 rejects it itself, with its own message
        result.stderr.fnmatch_lines(["ERROR: *strategies_export_dir*"])
