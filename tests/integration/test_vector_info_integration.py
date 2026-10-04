"""
End-to-end tests for the per-test metadata (D17), run through pytester: every
strategy row's item has its VectorInfo in ``item.stash[VECTOR_KEY]`` by the time a
conftest's pytest_collection_modifyitems runs, and in fixtures; ``VECTORS_KEY``
follows the node ID for stacked strategies; items without a strategy row have none;
the row's ``strategy`` mark adds no marker; and the -v summary counts rows by kind.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
    import enum

    import pytest
    from pytest_strategy import Parameter, RNGInteger, RNGSequence, Series, TestArg, register


    class Color(enum.Enum):
        RED = 1


    @register("vi_burst")
    def burst():
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            TestArg("len", rng_type=RNGInteger(1, 16)),
            directed_vectors={
                "zeros": (0, 1),
                "marked": pytest.param(1, 2, marks=pytest.mark.xfail(strict=False)),
            },
            test_vectors={"max": (63, 16)},
            nsamples=2,
        )


    @register("vi_esm")
    def esm():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            test_vectors={"first": (0, "a", 0)},
            nsamples=4,
        )


    @register("vi_flag")
    def flag():
        return Parameter(
            TestArg("on", rng_type=RNGInteger(0, 1)),
            test_vectors={"on": (1,)},
            nsamples=1,
        )


    @register("vi_odd")
    def odd():
        return Parameter(
            TestArg("color", value=Color.RED),
            TestArg("ratio", value=float("nan")),
            TestArg("raw", value=b"\\x00"),
            test_vectors={"same": (Color.RED, float("nan"), b"\\x00")},
            nsamples=1,
        )


    @register("vi_empty")
    def empty():
        return Parameter(
            TestArg("ch", rng_type=Series([], skip_if_empty="no channels")),
            TestArg("x", rng_type=RNGInteger(0, 9)),
        )
    """

CONFTEST = """
    import json

    import pytest
    from pytest_strategy import VECTOR_KEY, VECTORS_KEY, VectorInfo

    GENERATE = {}


    def pytest_generate_tests(metafunc):
        # The marks of the test itself: never a row's
        GENERATE[metafunc.definition.name] = [
            type(mark.args[0]).__name__ for mark in metafunc.definition.iter_markers("strategy")
        ]


    def pytest_collection_modifyitems(config, items):
        seen = {}
        for item in items:
            info = item.stash.get(VECTOR_KEY, None)
            infos = item.stash.get(VECTORS_KEY, None)
            closest = item.get_closest_marker("strategy")
            seen[item.nodeid] = {
                "info": None if info is None else info.to_dict(),
                "ids": None if infos is None else [i.id for i in infos],
                "first_is_info": infos is not None and infos[0] is info,
                "marks": [
                    type(mark.args[0]).__name__ if mark.args else None
                    for mark in item.iter_markers("strategy")
                ],
                "closest": None if closest is None else type(closest.args[0]).__name__,
            }
        path = config.rootpath / "collected.json"
        path.write_text(json.dumps({"items": seen, "generate": GENERATE}), encoding="utf-8")


    @pytest.fixture
    def info(request):
        return request.node.stash.get(VECTOR_KEY, None)
    """

TESTS = """
    import json
    from dataclasses import dataclass

    import pytest
    from pytest_strategy import VECTOR_KEY, VECTORS_KEY, strategy


    @dataclass
    class Burst:
        addr: int
        len: int


    @strategy("vi_burst")
    def test_named(addr, len, info):
        assert info.strategy == "vi_burst" and info.values == (addr, len)
        assert info.values._fields == ("addr", "len")


    @strategy("vi_burst")
    def test_record(burst: Burst, info):
        assert info.values == (burst.addr, burst.len)


    @strategy("vi_esm")
    def test_esm(ch, dev, x, info):
        assert info.values == (ch, dev, x)


    @strategy("vi_flag")
    @pytest.mark.parametrize("m", [7])
    @strategy("vi_burst")
    def test_stacked(addr, len, m, on, request):
        burst, flag = request.node.stash[VECTORS_KEY]
        assert request.node.stash[VECTOR_KEY] is burst
        assert (burst.strategy, flag.strategy) == ("vi_burst", "vi_flag")
        assert burst.values == (addr, len) and flag.values == (on,)


    @strategy("vi_flag")
    def test_flag(on, info):
        assert info.values == (on,)


    @strategy("vi_odd")
    def test_odd(color, ratio, raw, info):
        data = json.loads(json.dumps(info.to_dict(), allow_nan=False))
        assert data["schema"] == 1
        assert data["values"]["color"] == {"$enum": "Color", "member": "RED"}
        assert data["values"]["ratio"] == {"$float": "nan"}
        assert data["values"]["raw"] == {"$repr": "b'\\\\x00'", "$type": "bytes"}


    @strategy("vi_empty")
    def test_empty(ch, x):
        pass


    def test_plain(info):
        assert info is None
    """


def _project(pytester):
    pytester.makepyfile(vi_strategies=STRATEGIES)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_vi=TESTS)


def _collected(pytester):
    return json.loads((pytester.path / "collected.json").read_text(encoding="utf-8"))


def _by_test(items):
    """Group the collected items by test name: {name: {row id: entry}}."""
    grouped = {}
    for nodeid, entry in items.items():
        name, _, row = nodeid.partition("::")[2].partition("[")
        grouped.setdefault(name, {})[row.rstrip("]")] = entry
    return grouped


class TestStash:
    def test_every_row_has_its_info_before_modifyitems_and_in_fixtures(self, pytester):
        _project(pytester)

        result = pytester.runpytest("--rng-seed=1")

        result.assert_outcomes(passed=16, skipped=1, xpassed=3)
        tests = _by_test(_collected(pytester)["items"])
        named = tests["test_named"]
        assert list(named) == ["directed-zeros", "directed-marked", "rand-0", "rand-1"]
        zeros = named["directed-zeros"]["info"]
        assert zeros["strategy"] == "vi_burst" and zeros["origin"] == "vi_strategies.py:11"
        assert (zeros["kind"], zeros["name"], zeros["index"], zeros["id"]) == (
            "directed",
            "zeros",
            0,
            "directed-zeros",
        )
        assert zeros["values"] == {"addr": 0, "len": 1} and zeros["seed"] == 1
        marked = named["directed-marked"]["info"]
        assert (marked["kind"], marked["name"], marked["index"]) == ("directed", "marked", 1)
        assert [named[f"rand-{j}"]["info"]["index"] for j in (0, 1)] == [0, 1]
        for entry in named.values():
            assert entry["ids"] == [entry["info"]["id"]] and entry["first_is_info"]

        # Record mode: the same rows, by argument name
        record = tests["test_record"]
        assert list(record) == list(named)
        assert record["directed-zeros"]["info"]["values"] == {"addr": 0, "len": 1}

        # Under the finite count, the Series argument is enumerated
        esm = tests["test_esm"]
        assert list(esm) == ["ch=0-rand-0", "ch=1-rand-0", "ch=0-rand-1", "ch=1-rand-1"]
        assert {tuple(e["info"]["enumerated"]) for e in esm.values()} == {("ch",)}

        [skipped] = tests["test_empty"].values()
        assert skipped["info"]["kind"] == "skipped" and skipped["info"]["index"] is None
        assert skipped["info"]["values"] == {"ch": None, "x": None}
        assert tests["test_plain"][""] == {
            "info": None,
            "ids": None,
            "first_is_info": False,
            "marks": [],
            "closest": None,
        }

    def test_test_and_exhaustive_rows(self, pytester):
        _project(pytester)

        result = pytester.runpytest("--rng-seed=1", "--vector-mode=test")
        result.assert_outcomes(passed=7, skipped=1)
        tests = _by_test(_collected(pytester)["items"])
        [max_row] = tests["test_named"].values()
        assert (max_row["info"]["kind"], max_row["info"]["name"], max_row["info"]["id"]) == (
            "test",
            "max",
            "test-max",
        )

        result = pytester.runpytest("--rng-seed=1", "--nsamples=auto")
        result.assert_outcomes(passed=16, skipped=1, xpassed=3)
        tests = _by_test(_collected(pytester)["items"])
        esm = {row: entry["info"] for row, entry in tests["test_esm"].items()}
        assert sorted(esm) == ["ch=0-dev=a", "ch=0-dev=b", "ch=1-dev=a", "ch=1-dev=b"]
        assert {(i["kind"], tuple(i["enumerated"])) for i in esm.values()} == {
            ("exhaustive", ("ch", "dev"))
        }
        assert {row: i["index"] for row, i in esm.items()} == {
            "ch=0-dev=a": 0,
            "ch=0-dev=b": 1,
            "ch=1-dev=a": 2,
            "ch=1-dev=b": 3,
        }

    def test_stacked_strategies_follow_the_node_id(self, pytester):
        _project(pytester)

        result = pytester.runpytest("--rng-seed=1", "-k", "stacked")

        result.assert_outcomes(passed=3, xpassed=1)
        stacked = _by_test(_collected(pytester)["items"])["test_stacked"]
        # The decorator closest to the def comes first in the node ID
        assert list(stacked) == [
            "directed-zeros-7-rand-0",
            "directed-marked-7-rand-0",
            "rand-0-7-rand-0",
            "rand-1-7-rand-0",
        ]
        for row, entry in stacked.items():
            burst_id = row.partition("-7-")[0]
            assert entry["ids"] == [burst_id, "rand-0"]
            assert entry["info"]["strategy"] == "vi_burst" and entry["info"]["id"] == burst_id
            assert entry["first_is_info"]

    def test_the_empty_parameter_set_item_has_none(self, pytester):
        """vi_flag has no directed vector named zeros, so pytest makes one empty item."""
        _project(pytester)

        result = pytester.runpytest(
            "--rng-seed=1", "--vector-name=zeros", "-k", "test_flag or test_plain"
        )

        result.assert_outcomes(passed=1, skipped=1)
        items = _collected(pytester)["items"]
        [empty] = [nodeid for nodeid in items if "test_flag" in nodeid]
        assert items[empty]["info"] is None and items[empty]["ids"] is None
        assert items[empty]["marks"] == ["str"]

    def test_under_xdist_the_workers_fill_it(self, pytester):
        pytest.importorskip("xdist")
        _project(pytester)

        result = pytester.runpytest_subprocess("-n", "2", "--rng-seed=1", "-p", "no:cacheprovider")

        result.assert_outcomes(passed=16, skipped=1, xpassed=3)


class TestMarks:
    def test_the_row_marks_come_after_the_tests_own(self, pytester):
        """
        pytest puts a test's own marks before its row's, so get_closest_marker() still
        returns the test's own strategy mark; the stash keys are the API.
        """
        _project(pytester)

        pytester.runpytest("--rng-seed=1", "-k", "named or stacked")

        tests = _by_test(_collected(pytester)["items"])
        for entry in tests["test_named"].values():
            assert entry["marks"] == ["str", "VectorInfo"] and entry["closest"] == "str"
        for entry in tests["test_stacked"].values():
            assert entry["marks"] == ["str", "str", "VectorInfo", "VectorInfo"]

    def test_generate_tests_sees_only_the_tests_own_marks(self, pytester):
        _project(pytester)

        pytester.runpytest("--rng-seed=1")

        generate = _collected(pytester)["generate"]
        assert generate["test_named"] == ["str"]
        assert generate["test_stacked"] == ["str", "str"]
        assert generate["test_plain"] == []

    def test_strict_markers_pass(self, pytester):
        _project(pytester)

        result = pytester.runpytest("--rng-seed=1", "--strict-markers", "-W", "error")

        result.assert_outcomes(passed=16, skipped=1, xpassed=3)

    def test_no_new_marker(self, pytester):
        def markers(*args):
            result = pytester.runpytest("--markers", *args)
            assert result.ret == pytest.ExitCode.OK
            return {
                line.split(":", 1)[0]
                for line in result.stdout.lines
                if line.startswith("@pytest.mark.")
            }

        with_plugin = markers()
        assert with_plugin - markers("-p", "no:pytest_strategy") == {
            "@pytest.mark.strategy(name_or_factory, *, validate_signature=True)"
        }

    def test_k_matches_no_new_keyword(self, pytester):
        """The row's mark is named strategy, as the test's: -k vector selects no row."""
        _project(pytester)

        result = pytester.runpytest("--rng-seed=1", "--collect-only", "-q", "-k", "vector")

        assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


class TestVerboseSummary:
    @pytest.mark.parametrize(
        ("args", "lines"),
        [
            (
                [],
                [
                    r"  vi_burst \(vi_strategies\.py\): 3 test\(s\), 6 directed, 6 random rows; "
                    r"nsamples=2 from Parameter\(nsamples=\)$",
                    r"  vi_empty \(vi_strategies\.py\): 1 test\(s\), 0 directed, 0 random, "
                    r"1 skipped rows; nsamples=10 from default$",
                ],
            ),
            (
                ["--vector-mode=test"],
                [
                    r"  vi_burst \(vi_strategies\.py\): 3 test\(s\), 0 directed, 0 random, "
                    r"3 test rows; .*",
                ],
            ),
            (
                ["--nsamples=auto"],
                [
                    r"  vi_burst \(vi_strategies\.py\): 3 test\(s\), 6 directed, 6 random rows; "
                    r"nsamples=2 from Parameter\(nsamples=\), no Series/RNGSequence for auto$",
                    r"  vi_esm \(vi_strategies\.py\): 1 test\(s\), 0 directed, 0 random, "
                    r"4 exhaustive rows; nsamples=auto from --nsamples$",
                ],
            ),
        ],
        ids=["default", "test", "auto"],
    )
    def test_rows_are_counted_by_kind(self, pytester, args, lines):
        _project(pytester)

        result = pytester.runpytest("-v", "--rng-seed=1", *args)

        assert result.ret == pytest.ExitCode.OK
        result.stdout.re_match_lines(lines)
