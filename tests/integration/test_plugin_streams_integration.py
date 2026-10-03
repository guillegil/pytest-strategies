"""
End-to-end tests for the plugin's random streams other than the rows (streams v1,
D5), run through pytester: a factory call, a strategy file's import, a test module's
import, the context hook, each fixture's setup, each phase of a test and an export
call draw from streams of their own, keyed by what they are. Their draws are the same
whichever tests run, in any order, alone or under pytest-xdist, and an RNG.seed()
call in one of them changes nothing outside it.

Each project checks its draws against the key the plugin is meant to use, and the
runs are compared with each other. Distinct module and strategy names are used per
project on purpose (see test_session_isolation_integration.py for rationale).
"""

import ast
import importlib.util
import json
import os
import random
from textwrap import dedent

import pytest

from pytest_strategy import RNG
from pytest_strategy._streams import StreamKey

pytest_plugins = ["pytester"]

SEED = 7


def randint(key):
    """The first RNG.integer(0, 10**9) of the stream of ``key``."""
    return random.Random(key.seed_int()).randint(0, 10**9)


# A module the projects import (pythonpath = .) to record what their tests draw,
# one file per process, so that pytest-xdist workers do not write to the same file
RECORD = """
    import json
    import os
    import pathlib

    DIRECTORY = pathlib.Path(__file__).parent / "records"

    def record(label, value):
        DIRECTORY.mkdir(exist_ok=True)
        with open(DIRECTORY / f"{os.getpid()}.jsonl", "a") as out:
            out.write(json.dumps([label, value]) + "\\n")
"""


def read_records(pytester):
    """Return the values each label recorded in the last run (one per label)."""
    found = {}
    for path in sorted((pytester.path / "records").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            label, value = json.loads(line)
            found.setdefault(label, set()).add(value)
    # A fixture or a module that two pytest-xdist workers set up draws the same
    assert all(len(values) == 1 for values in found.values()), found
    return {label: values.pop() for label, values in found.items()}


def run_and_read(pytester, *args, passed):
    """Run pytest in a subprocess, check the outcome, and return the records."""
    records = pytester.path / "records"
    for path in records.glob("*.jsonl") if records.exists() else ():
        path.unlink()
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}", *args)
    result.assert_outcomes(passed=passed)
    return read_records(pytester)


# ---------------------------------------------------------------------------
# Factories: T/"factory"
# ---------------------------------------------------------------------------


FACTORY_STRATEGIES = """
    import os

    from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, register

    @register("ps_factory")
    def factory(rng):
        assert rng is RNG.generator()
        if os.environ.get("PS_EXTRA_DRAW"):
            rng.random()
        return Parameter(
            TestArg("f", value=(rng.random(), RNG.integer(0, 10**9))),
            TestArg("x", rng_type=RNGInteger(0, 10**9)),
            nsamples=3,
        )

    @register("ps_ctx")
    def with_ctx(ctx):
        return Parameter(TestArg("c", value=ctx["draw"]), nsamples=1)
"""

FACTORY_TESTS = """
    from pytest_strategy import strategy

    @strategy("ps_factory")
    def test_a(f, x):
        pass

    @strategy("ps_factory")
    def test_b(f, x):
        pass

    @strategy("ps_ctx")
    def test_ctx_a(c):
        pass

    @strategy("ps_ctx")
    def test_ctx_b(c):
        pass
"""

CONTEXT_HOOK = """
from pytest_strategy import RNG

def pytest_strategies_context(config):
    return {"draw": RNG.integer(0, 10**9)}
"""


@pytest.fixture
def factory_project(pytester, values_dump):
    pytester.makepyfile(strategies=FACTORY_STRATEGIES, test_ps_factory=FACTORY_TESTS)
    pytester.makeconftest(values_dump.conftest + CONTEXT_HOOK)
    return values_dump


def by_test(rows, name):
    """Return the values of the argument ``name`` by test name, row by row."""
    values = {}
    for nodeid, params in rows:
        params = ast.literal_eval(params)
        if name in params:
            test = nodeid.split("::")[-1].split("[")[0]
            values.setdefault(test, []).append(params[name])
    return values


class TestFactoryStreams:
    def test_draws_are_the_same_for_the_file_one_test_and_any_count(self, factory_project):
        def draws(*args):
            rows = factory_project.collect("--collect-only", f"--rng-seed={SEED}", *args)
            return by_test(rows, "f")

        whole = draws()
        selected = draws("-k", "test_b")
        alone = draws("test_ps_factory.py::test_b")
        five = draws("--nsamples=5")

        assert selected["test_b"] == whole["test_b"]
        assert alone == {"test_b": whole["test_b"]}
        assert five == {test: values[:1] * 5 for test, values in whole.items()}

    def test_two_tests_get_draws_of_their_own(self, factory_project):
        draws = by_test(factory_project.collect("--collect-only", f"--rng-seed={SEED}"), "f")

        for test in ("test_a", "test_b"):
            key = StreamKey.root(SEED, "test", "ps_factory", f"test_ps_factory.py::{test}")
            generator = random.Random(key.child("factory").seed_int())
            expected = (generator.random(), generator.randint(0, 10**9))
            assert draws[test] == [expected] * 3
        assert draws["test_a"] != draws["test_b"]

    def test_an_extra_draw_in_the_factory_leaves_the_rows(self, factory_project, monkeypatch):
        base = factory_project.collect("--collect-only", f"--rng-seed={SEED}")
        monkeypatch.setenv("PS_EXTRA_DRAW", "1")
        extra = factory_project.collect("--collect-only", f"--rng-seed={SEED}")

        assert by_test(extra, "x") == by_test(base, "x")
        assert by_test(extra, "f") != by_test(base, "f")

    def test_the_context_hook_draws_from_its_own_stream(self, factory_project):
        whole = by_test(factory_project.collect("--collect-only", f"--rng-seed={SEED}"), "c")
        alone = by_test(
            factory_project.collect(
                "--collect-only", f"--rng-seed={SEED}", "test_ps_factory.py::test_ctx_b"
            ),
            "c",
        )

        expected = randint(StreamKey.root(SEED, "ctx"))
        assert whole == {"test_ctx_a": [expected], "test_ctx_b": [expected]}
        assert alone == {"test_ctx_b": [expected]}


# ---------------------------------------------------------------------------
# Strategy files: root(S, "file", path)
# ---------------------------------------------------------------------------


FILE_A = """
    from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, register

    DRAW = RNG.integer(0, 10**9)

    @register("ps_file_a")
    def file_a():
        return Parameter(
            TestArg("d", value=DRAW), TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=2
        )
"""

# A strategy file that reseeds the RNG when it is imported
FILE_B = """
    from pytest_strategy import RNG, Parameter, TestArg, register

    RNG.seed(5)
    DRAW = RNG.integer(0, 10**9)

    @register("ps_file_b")
    def file_b():
        return Parameter(TestArg("d", value=DRAW), nsamples=1)
"""

TEST_A = """
    import random

    from pytest_strategy import RNG, strategy
    from pytest_strategy._streams import StreamKey

    @strategy("ps_file_a")
    def test_a(d, x):
        pass

    def test_user_stream(request):
        assert RNG.get_seed() == request.config.getoption("--rng-seed")
        RNG.refresh_seed(key=request.node.nodeid)
        key = StreamKey.root(RNG.get_seed(), "user", request.node.nodeid)
        assert RNG.integer(0, 10**9) == random.Random(key.seed_int()).randint(0, 10**9)
"""

TEST_B = """
    from pytest_strategy import strategy

    @strategy("ps_file_b")
    def test_b(d):
        pass
"""


@pytest.fixture
def file_project(pytester, values_dump):
    pytester.makeconftest(values_dump.conftest)
    for folder, strategies, tests in (("a", FILE_A, TEST_A), ("b", FILE_B, TEST_B)):
        directory = pytester.path / "tests" / folder
        directory.mkdir(parents=True)
        (directory / "strategies.py").write_text(dedent(strategies))
        (directory / f"test_{folder}.py").write_text(dedent(tests))
    return values_dump


class TestFileStreams:
    def test_draws_are_the_same_for_full_subset_and_reversed_runs(self, file_project):
        runs = {
            "full": file_project.collect(f"--rng-seed={SEED}"),
            "subset": file_project.collect(f"--rng-seed={SEED}", "tests/a"),
            "reversed": file_project.collect(f"--rng-seed={SEED}", "tests/b", "tests/a"),
        }

        rows_of_a = {
            run: sorted(row for row in rows if "test_a.py::test_a[" in row[0])
            for run, rows in runs.items()
        }
        assert rows_of_a["full"] == rows_of_a["subset"] == rows_of_a["reversed"]
        expected = randint(StreamKey.root(SEED, "file", "tests/a/strategies.py"))
        assert by_test(rows_of_a["full"], "d") == {"test_a": [expected, expected]}

    def test_a_reseed_in_a_file_changes_only_that_file(self, file_project):
        rows = file_project.collect(f"--rng-seed={SEED}", "tests/b", "tests/a")

        # The test bodies check RNG.get_seed() and their refresh_seed stream
        assert by_test(rows, "d")["test_b"] == [random.Random(5).randint(0, 10**9)]


SHARED_STRATEGIES = """
    import pathlib

    from pytest_strategy import RNG, Parameter, TestArg, register

    DRAW = RNG.integer(0, 10**9)
    pathlib.Path(__file__).with_name("draw.txt").write_text(str(DRAW))

    @register("ps_shared")
    def shared():
        return Parameter(TestArg("d", value=DRAW), nsamples=1)
"""

SHARED_TESTS = """
    from pytest_strategy import strategy

    @strategy("ps_shared")
    def test_shared(d):
        pass
"""


def test_a_file_outside_the_rootdir_draws_the_same_in_two_checkouts(pytester, monkeypatch):
    draws = []
    for base in ("one", "deeper/two"):
        proj = pytester.path / base / "proj"
        shared = pytester.path / base / "shared"
        proj.mkdir(parents=True)
        shared.mkdir()
        (proj / "pytest.ini").write_text("[pytest]\ntestpaths = ../shared\n")
        (shared / "x_strategies.py").write_text(dedent(SHARED_STRATEGIES))
        (shared / "test_x.py").write_text(dedent(SHARED_TESTS))
        monkeypatch.chdir(proj)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=1)
        draws.append(int((shared / "draw.txt").read_text()))

    assert draws == [randint(StreamKey.root(SEED, "file", "../shared/x_strategies.py"))] * 2


# A strategy file a constraint imports while the rows are drawn
LATE_STRATEGIES = """
    from pytest_strategy import RNG, Parameter, TestArg, register

    DRAW = RNG.integer(0, 10**9)

    @register("ps_late")
    def late():
        return Parameter(TestArg("d", value=DRAW), nsamples=1)
"""

LATE_TESTS = """
    import random

    from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, strategy
    from pytest_strategy._streams import StreamKey

    SEEN = []

    def imports_late(v):
        import late_strategies

        SEEN.append(late_strategies.DRAW)
        return True

    def factory():
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)), vector_constraints=[imports_late], nsamples=2
        )

    @strategy(factory)
    def test_x(x):
        pass

    def test_the_file_drew_from_its_stream():
        key = StreamKey.root(RNG.get_seed(), "file", "lib/late_strategies.py")
        assert set(SEEN) == {random.Random(key.seed_int()).randint(0, 10**9)}
"""


def test_a_strategy_file_imported_by_a_constraint_draws_from_its_stream(pytester):
    pytester.makeini("[pytest]\npythonpath = lib\n")
    pytester.mkdir("lib")
    (pytester.path / "lib" / "late_strategies.py").write_text(dedent(LATE_STRATEGIES))
    pytester.makepyfile(test_late=LATE_TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

    result.assert_outcomes(passed=3)


# ---------------------------------------------------------------------------
# Test modules, fixtures and test phases
# ---------------------------------------------------------------------------


RUN_CONFTEST = """
    import os

    import pytest

    from pytest_strategy import RNG
    from record import record

    @pytest.fixture(scope="session")
    def sess():
        value = RNG.integer(0, 10**9)
        record("sess", value)
        return value

    def pytest_collection_modifyitems(items):
        if os.environ.get("PS_REVERSE"):
            items.reverse()
"""

RUN_BODY = """
    import pytest

    from pytest_strategy import RNG
    from record import record

    record("module", RNG.integer(0, 10**9))

    @pytest.fixture(scope="module")
    def mod():
        value = RNG.integer(0, 10**9)
        record("mod", value)
        yield value
        record("mod teardown", RNG.integer(0, 10**9))

    @pytest.fixture(params=["p", "q"])
    def par(request):
        value = RNG.integer(0, 10**9)
        record(f"par {request.param}", value)
        return value

    def test_first():
        record("test_first", RNG.integer(0, 10**9))

    def test_uses(sess, mod):
        record("test_uses", RNG.integer(0, 10**9))

    def test_reseeds():
        RNG.seed(5)
        record("test_reseeds", RNG.integer(0, 10**9))

    def test_par(par):
        pass

    def test_last():
        record("test_last", RNG.integer(0, 10**9))
        record("seed", RNG.get_seed())
"""

RUN_EARLY = """
    from pytest_strategy import RNG
    from record import record

    def test_early(sess):
        record("test_early", RNG.integer(0, 10**9))
"""


@pytest.fixture
def run_project(pytester):
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(record=RECORD)
    pytester.makeconftest(RUN_CONFTEST)
    pytester.mkdir("tests")
    (pytester.path / "tests" / "test_body.py").write_text(dedent(RUN_BODY))
    (pytester.path / "tests" / "test_early.py").write_text(dedent(RUN_EARLY))
    return pytester


BODY = "tests/test_body.py"


class TestBodyAndFixtureStreams:
    def test_each_draw_comes_from_its_key(self, run_project):
        records = run_and_read(run_project, passed=7)

        def body(test, phase="call"):
            return randint(StreamKey.root(SEED, "body", f"{BODY}::{test}", phase))

        def fixture(scope, name, param_index, where, base):
            # The scope node's ID, the name and the parameter index, then where the
            # fixture is defined (its file and its function's qualified name) and the
            # node ID pytest registered it for ("" for the rootdir's conftest.py)
            return randint(
                StreamKey.root(SEED, "fixture", scope, name, param_index, where, name, base)
            )

        assert records == {
            "module": randint(StreamKey.root(SEED, "module", BODY)),
            "test_first": body("test_first"),
            "sess": fixture("", "sess", 0, "conftest.py", ""),
            "mod": fixture(BODY, "mod", 0, BODY, BODY),
            "test_uses": body("test_uses"),
            "test_reseeds": random.Random(5).randint(0, 10**9),
            "par p": fixture(f"{BODY}::test_par[p]", "par", 0, BODY, BODY),
            "par q": fixture(f"{BODY}::test_par[q]", "par", 1, BODY, BODY),
            "test_last": body("test_last"),
            # test_reseeds' RNG.seed(5) ended with its call phase
            "seed": SEED,
            # A module fixture's teardown runs in the teardown of the module's last test
            "mod teardown": body("test_last", "teardown"),
            "test_early": randint(
                StreamKey.root(SEED, "body", "tests/test_early.py::test_early", "call")
            ),
        }

    def test_draws_are_the_same_alone_in_the_suite_and_reversed(self, run_project, monkeypatch):
        suite = run_and_read(run_project, passed=7)
        alone = run_and_read(run_project, f"{BODY}::test_uses", passed=1)
        early = run_and_read(run_project, "tests/test_early.py", passed=1)
        monkeypatch.setenv("PS_REVERSE", "1")
        reversed_ = run_and_read(run_project, passed=7)

        # Alone, test_uses sets up both fixtures; reversed, test_early sets up sess
        for label in ("module", "sess", "mod", "test_uses"):
            assert alone[label] == suite[label], label
        assert early["sess"] == suite["sess"]
        assert {k: v for k, v in reversed_.items() if k != "mod teardown"} == {
            k: v for k, v in suite.items() if k != "mod teardown"
        }

    def test_draws_are_the_same_under_xdist(self, run_project):
        pytest.importorskip("xdist")
        suite = run_and_read(run_project, passed=7)

        distributed = run_and_read(run_project, "-n", "2", passed=7)

        # Which test tears the module fixture down depends on the scheduling
        suite.pop("mod teardown")
        distributed.pop("mod teardown")
        assert distributed == suite


# A conftest.py below the initial ones is imported while its folder is collected,
# outside every stream, so its RNG.seed(5) stays for the rest of the session: the
# test module, fixture and test-phase streams are keyed by the run's seed, not
# RNG.get_seed(). A rerun of the node ID makes the conftest an initial one, which
# pytest_configure's seeding then overrides

RESEEDING_CONFTEST = """
    import pytest

    from pytest_strategy import RNG
    from record import record

    RNG.seed(5)

    @pytest.fixture
    def sub():
        value = RNG.integer(0, 10**9)
        record("sub", value)
        return value
"""

RESEEDING_TESTS = """
    from pytest_strategy import RNG
    from record import record

    record("module", RNG.integer(0, 10**9))

    def test_sub(sub):
        record("test_sub", RNG.integer(0, 10**9))
        record("seed", RNG.get_seed())
"""


def test_a_conftest_that_reseeds_leaves_the_module_fixture_and_body_streams(pytester):
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(record=RECORD)
    sub = pytester.mkdir("tests") / "sub"
    sub.mkdir()
    (sub / "conftest.py").write_text(dedent(RESEEDING_CONFTEST))
    (sub / "test_sub.py").write_text(dedent(RESEEDING_TESTS))

    # The whole project, so that tests/sub/conftest.py is not an initial conftest
    records = run_and_read(pytester, passed=1)

    test = "tests/sub/test_sub.py::test_sub"
    assert records == {
        "module": randint(StreamKey.root(SEED, "module", "tests/sub/test_sub.py")),
        "sub": randint(
            StreamKey.root(
                SEED, "fixture", test, "sub", 0, "tests/sub/conftest.py", "sub", "tests/sub"
            )
        ),
        "test_sub": randint(StreamKey.root(SEED, "body", test, "call")),
        # The conftest's seed is what RNG.get_seed() returns in the test
        "seed": 5,
    }
    # Run alone, the conftest is an initial one, which the session's seed overrides
    assert run_and_read(pytester, test, passed=1) == {**records, "seed": SEED}


# The test IDs never change the rows, but a test's phases draw from streams of its
# node ID, which contains the ID

IDS_TESTS = """
    from pytest_strategy import RNG, VECTOR_KEY, Parameter, RNGInteger, TestArg, strategy
    from record import record

    def ps_ids_rows():
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=2)

    @strategy(ps_ids_rows)
    def test_ids(x, request):
        index = request.node.stash[VECTOR_KEY].index
        record(f"row {index}", x)
        record(f"nodeid {index}", request.node.nodeid)
        record(f"body {index}", RNG.integer(0, 10**9))
"""


def test_the_id_format_changes_the_body_draws_not_the_rows(pytester):
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(record=RECORD, test_ids=IDS_TESTS)

    names = run_and_read(pytester, passed=2)
    values = run_and_read(pytester, "-o", "strategies_ids=values", passed=2)

    for index in (0, 1):
        assert values[f"row {index}"] == names[f"row {index}"]
        assert values[f"nodeid {index}"] != names[f"nodeid {index}"]
        for records in (names, values):
            nodeid = records[f"nodeid {index}"]
            assert records[f"body {index}"] == randint(StreamKey.root(SEED, "body", nodeid, "call"))
        assert values[f"body {index}"] != names[f"body {index}"]


# Fixtures that pytest sets up for the same scope node under one name: an override
# that requests the fixture it overrides, session fixtures of one name in two
# sibling folders' conftest.py files, and one session fixture function that both
# folders' conftest.py files import

SAME_NAME_CONFTEST = """
    import pytest

    from pytest_strategy import RNG
    from record import record

    @pytest.fixture
    def value():
        drawn = RNG.integer(0, 10**9)
        record("value", drawn)
        return drawn

    @pytest.fixture(scope="module")
    def modvalue():
        drawn = RNG.integer(0, 10**9)
        record("modvalue", drawn)
        return drawn
"""

SAME_NAME_OVERRIDE = """
    import pytest

    from pytest_strategy import RNG
    from record import record

    @pytest.fixture
    def value(value):
        drawn = RNG.integer(0, 10**9)
        record("value override", drawn)
        return drawn

    @pytest.fixture(scope="module")
    def modvalue(modvalue):
        drawn = RNG.integer(0, 10**9)
        record("modvalue override", drawn)
        return drawn

    def test_override(value, modvalue):
        pass
"""

SAME_NAME_SIBLING = """
    import pytest

    from pytest_strategy import RNG
    from ps_shared_fixtures import port
    from record import record

    @pytest.fixture(scope="session")
    def resource():
        drawn = RNG.integer(0, 10**9)
        record("resource {folder}", drawn)
        return drawn
"""

SAME_NAME_SHARED = """
    import pytest

    from pytest_strategy import RNG

    @pytest.fixture(scope="session")
    def port():
        return RNG.integer(0, 10**9)
"""


@pytest.fixture
def same_name_project(pytester):
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(record=RECORD)
    pytester.makeconftest(SAME_NAME_CONFTEST)
    pytester.makepyfile(ps_shared_fixtures=SAME_NAME_SHARED)
    tests = pytester.mkdir("tests")
    (tests / "test_override.py").write_text(dedent(SAME_NAME_OVERRIDE))
    for folder in ("a", "b"):
        (tests / folder).mkdir()
        (tests / folder / "conftest.py").write_text(
            dedent(SAME_NAME_SIBLING).replace("{folder}", folder)
        )
        (tests / folder / f"test_{folder}.py").write_text(
            "from record import record\n\n"
            f"def test_{folder}(resource, port):\n"
            f"    record('port {folder}', port)\n"
        )
    return pytester


class TestFixturesOfOneName:
    def test_each_definition_draws_from_a_stream_of_its_own(self, same_name_project):
        records = run_and_read(same_name_project, passed=3)

        def fixture(scope, name, where, base):
            return randint(StreamKey.root(SEED, "fixture", scope, name, 0, where, name, base))

        override = "tests/test_override.py"
        test = f"{override}::test_override"
        assert records == {
            "value": fixture(test, "value", "conftest.py", ""),
            "value override": fixture(test, "value", override, override),
            "modvalue": fixture(override, "modvalue", "conftest.py", ""),
            "modvalue override": fixture(override, "modvalue", override, override),
            "resource a": fixture("", "resource", "tests/a/conftest.py", "tests/a"),
            "resource b": fixture("", "resource", "tests/b/conftest.py", "tests/b"),
            # One function, of a module imported by its name: the folder pytest
            # registered it for tells them apart
            "port a": fixture("", "port", "ps_shared_fixtures", "tests/a"),
            "port b": fixture("", "port", "ps_shared_fixtures", "tests/b"),
        }
        # 4.0's first key had no definition, and its second no base: these pairs
        # drew the same values
        assert records["value"] != records["value override"]
        assert records["modvalue"] != records["modvalue override"]
        assert records["resource a"] != records["resource b"]
        assert records["port a"] != records["port b"]

    def test_draws_are_the_same_alone_in_the_suite_and_under_xdist(self, same_name_project):
        suite = run_and_read(same_name_project, passed=3)
        alone = {
            **run_and_read(same_name_project, "tests/test_override.py", passed=1),
            **run_and_read(same_name_project, "tests/b", passed=1),
            **run_and_read(same_name_project, "tests/a/test_a.py::test_a", passed=1),
        }

        assert alone == suite
        if importlib.util.find_spec("xdist") is not None:
            assert run_and_read(same_name_project, "-n", "2", passed=3) == suite


# ---------------------------------------------------------------------------
# Export: root(S, "export", name, folder)
# ---------------------------------------------------------------------------


EXPORT_STRATEGIES = """
    import json
    import pathlib

    from pytest_strategy import Parameter, TestArg, register

    OUT = pathlib.Path(__file__).parent / "export_draws.jsonl"

    def record(name, rng):
        with open(OUT, "a") as f:
            f.write(json.dumps([name, rng.random()]) + "\\n")
        return Parameter(TestArg("e", value=1), nsamples=1)

    @register("ps_export")
    def exported(rng):
        return record("ps_export", rng)

    # A factory whose code has no file: its co_filename is "<string>"
    namespace = {"record": record}
    exec("def made(rng):\\n    return record('ps_exec', rng)\\n", namespace)
    register("ps_exec")(namespace["made"])
"""

EXPORT_TESTS = """
    from pytest_strategy import RNG, export_strategies

    def test_export(monkeypatch):
        export_strategies()
        # Neither the seed RNG.seed() sets nor the working directory changes the streams
        RNG.seed(5)
        monkeypatch.chdir("tests")
        export_strategies()
"""


def test_export_calls_draw_from_the_strategys_folder_stream(pytester):
    pytester.mkdir("tests")
    (pytester.path / "tests" / "a").mkdir()
    (pytester.path / "tests" / "a" / "strategies.py").write_text(dedent(EXPORT_STRATEGIES))
    (pytester.path / "tests" / "a" / "test_export.py").write_text(dedent(EXPORT_TESTS))

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

    result.assert_outcomes(passed=1)
    lines = (pytester.path / "tests" / "a" / "export_draws.jsonl").read_text().splitlines()

    def draw(name, folder):
        key = StreamKey.root(SEED, "export", name, folder)
        return [name, random.Random(key.seed_int()).random()]

    # The folder of a factory without a file is "", not the working directory
    assert [json.loads(line) for line in lines] == [
        draw("ps_export", "tests/a"),
        draw("ps_exec", ""),
    ] * 2


INSTALLED_STRATEGIES = """
    from pytest_strategy import Parameter, TestArg, register

    @register("ps_installed")
    def installed(rng):
        return Parameter(TestArg("e", value=rng.randint(0, 10**9)), nsamples=1)
"""

INSTALLED_TESTS = """
    import pathlib

    from pytest_strategy import export_strategies

    def test_export():
        (pathlib.Path(__file__).parent / "export.json").write_text(export_strategies())
"""


def test_an_installed_packages_factory_exports_the_same_from_any_environment(pytester):
    """Its folder's path depends on where the package is installed; its module does not."""
    pytester.makeconftest("import ps_installed_strategies")
    pytester.makepyfile(test_export=INSTALLED_TESTS)
    exported = []
    for env, folder in (("env_a", "site-packages"), ("env_b", "dist-packages")):
        package = pytester.path / env / "lib" / "python3" / folder / "ps_installed_strategies"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(dedent(INSTALLED_STRATEGIES))
        # pytest splits ini paths with shlex, which drops Windows backslashes
        pytester.makeini(f"[pytest]\npythonpath = {package.parent.as_posix()}\n")

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=1)
        data = json.loads((pytester.path / "export.json").read_text())
        exported.append(data["ps_installed"]["arguments"][0]["static_value"])

    key = StreamKey.root(SEED, "export", "ps_installed", "ps_installed_strategies")
    assert exported == [str(randint(key))] * 2


# ---------------------------------------------------------------------------
# Keys that do not depend on the environment or the checkout
# ---------------------------------------------------------------------------


# Writes what a project draws to draws.jsonl in the working directory
DRAWS = """
    import json
    import pathlib

    def record(label, value):
        with open(pathlib.Path.cwd() / "draws.jsonl", "a") as out:
            out.write(json.dumps([label, value]) + "\\n")
"""


def read_draws(path):
    """Return the values each label recorded in ``path``/draws.jsonl, then remove it."""
    found = {}
    for line in (path / "draws.jsonl").read_text().splitlines():
        label, value = json.loads(line)
        found.setdefault(label, set()).add(value)
    (path / "draws.jsonl").unlink()
    assert all(len(values) == 1 for values in found.values()), found
    return {label: values.pop() for label, values in found.items()}


ROOT_PACKAGE_CONFTEST = """
    import pytest

    from pytest_strategy import RNG
    from ps_draws import record

    @pytest.fixture(scope="package")
    def root_pkg():
        value = RNG.integer(0, 10**9)
        record("root_pkg", value)
        return value
"""


def test_a_package_fixture_of_a_rootdir_that_is_a_package(pytester, monkeypatch):
    """
    pytest 9 sets it up for the rootdir's Package, whose node ID is ".", and pytest
    8 for the session, whose node ID is "": both are keyed by "", so the fixture
    draws the same on both.
    """
    tests = pytester.mkdir("tests")
    (tests / "pytest.ini").write_text(f"[pytest]\npythonpath = {pytester.path.as_posix()}\n")
    (tests / "__init__.py").write_text("")
    (tests / "conftest.py").write_text(dedent(ROOT_PACKAGE_CONFTEST))
    (tests / "test_a.py").write_text("def test_a(root_pkg):\n    pass\n")
    (tests / "sub").mkdir()
    (tests / "sub" / "__init__.py").write_text("")
    (tests / "sub" / "test_b.py").write_text("def test_b(root_pkg):\n    pass\n")
    pytester.makepyfile(ps_draws=DRAWS)
    monkeypatch.chdir(tests)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

    result.assert_outcomes(passed=2)
    key = StreamKey.root(SEED, "fixture", "", "root_pkg", 0, "conftest.py", "root_pkg", "")
    assert read_draws(tests) == {"root_pkg": randint(key)}


PACKAGE_FIXTURES = """
    import pytest

    from pytest_strategy import RNG, Parameter, TestArg, register

    @pytest.fixture(scope="session")
    def device():
        return RNG.integer(0, 10**9)

    @register("ps_device")
    def device_rows(rng):
        return Parameter(TestArg("d", value=rng.randint(0, 10**9)), nsamples=1)
"""

PACKAGE_TESTS = """
    import json

    from pytest_strategy import export_strategies
    from ps_draws import record

    def test_device(device):
        record("device", device)
        exported = json.loads(export_strategies())["ps_device"]
        record("export", exported["arguments"][0]["static_value"])
"""


def test_a_packages_fixture_draws_the_same_installed_or_from_its_source(pytester):
    """
    A package that ships fixtures and strategies, run installed (tox, CI) and in
    editable mode from its checkout's src/ folder: they are keyed by their module's
    name, so a seed recorded in one reruns in the other.
    """
    pytester.makeconftest("from acme_ps.testing import device")
    pytester.makepyfile(ps_draws=DRAWS, test_device=PACKAGE_TESTS)
    draws = []
    for folder in ("src", ".tox/py/lib/python3/site-packages"):
        package = pytester.path / folder / "acme_ps"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "testing.py").write_text(dedent(PACKAGE_FIXTURES))
        pythonpath = f"{package.parent.as_posix()} {pytester.path.as_posix()}"
        pytester.makeini(f"[pytest]\npythonpath = {pythonpath}\n")

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=1)
        draws.append(read_draws(pytester.path))
        (package / "testing.py").unlink()

    device = StreamKey.root(SEED, "fixture", "", "device", 0, "acme_ps.testing", "device", "")
    export = StreamKey.root(SEED, "export", "ps_device", "acme_ps.testing")
    assert draws == [{"device": randint(device), "export": str(randint(export))}] * 2


LINKED_CONFTEST = """
    import pytest

    from pytest_strategy import RNG
    from ps_draws import record

    @pytest.fixture(scope="session")
    def shared_fix():
        value = RNG.integer(0, 10**9)
        record("fixture", value)
        return value
"""

LINKED_STRATEGIES = """
    from pytest_strategy import RNG, Parameter, TestArg, register

    DRAW = RNG.integer(0, 10**9)

    @register("ps_linked")
    def linked():
        return Parameter(TestArg("d", value=DRAW), nsamples=1)
"""

LINKED_TESTS = """
    from pytest_strategy import RNG, strategy
    from ps_draws import record

    record("module", RNG.integer(0, 10**9))

    @strategy("ps_linked")
    def test_linked(d, shared_fix):
        record("file", d)
"""


def test_a_folder_linked_into_two_checkouts_draws_the_same(pytester, monkeypatch):
    """
    A test folder linked into checkouts at different depths from a place that does
    not move with them: its module, strategy file and fixture streams are keyed by
    the folder as linked, as pytest spells its node IDs, not by where it really is.
    """
    shared = pytester.mkdir("shared")
    (shared / "conftest.py").write_text(dedent(LINKED_CONFTEST))
    (shared / "x_strategies.py").write_text(dedent(LINKED_STRATEGIES))
    (shared / "test_linked.py").write_text(dedent(LINKED_TESTS))
    pytester.makepyfile(ps_draws=DRAWS)
    draws = []
    for base in ("one", "deeper/x/two"):
        proj = pytester.path / base / "proj"
        proj.mkdir(parents=True)
        try:
            os.symlink(shared, proj / "tests_shared", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")
        (proj / "pytest.ini").write_text(f"[pytest]\npythonpath = {pytester.path.as_posix()}\n")
        monkeypatch.chdir(proj)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=1)
        draws.append(read_draws(proj))

    folder = "tests_shared"
    fixture = ("shared_fix", 0, f"{folder}/conftest.py", "shared_fix", folder)
    expected = {
        "module": randint(StreamKey.root(SEED, "module", f"{folder}/test_linked.py")),
        "file": randint(StreamKey.root(SEED, "file", f"{folder}/x_strategies.py")),
        "fixture": randint(StreamKey.root(SEED, "fixture", "", *fixture)),
    }
    assert draws == [expected, expected]


SHIPPED_TESTS = """
    from pytest_strategy import RNG
    from ps_draws import record

    record("module", RNG.integer(0, 10**9))

    def test_shipped():
        record("body", RNG.integer(0, 10**9))
"""


def test_tests_an_installed_package_ships_draw_the_same_from_any_environment(pytester, monkeypatch):
    """--pyargs: a test module's stream is keyed by its path below site-packages."""
    pytester.makepyfile(ps_draws=DRAWS)
    run = pytester.mkdir("run")
    monkeypatch.chdir(run)
    draws = []
    for env in ("venv_a", "deeper/venv_b"):
        tests = pytester.path / env / "lib" / "python3" / "site-packages" / "ps_shipped" / "tests"
        tests.mkdir(parents=True)
        (tests.parent / "__init__.py").write_text("")
        (tests / "__init__.py").write_text("")
        (tests / "test_shipped.py").write_text(dedent(SHIPPED_TESTS))
        pythonpath = f"{tests.parent.parent.as_posix()} {pytester.path.as_posix()}"
        (run / "pytest.ini").write_text(f"[pytest]\npythonpath = {pythonpath}\n")

        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", f"--rng-seed={SEED}", "--pyargs", "ps_shipped.tests"
        )

        result.assert_outcomes(passed=1)
        draws.append(read_draws(run))
        (tests / "test_shipped.py").unlink()

    module = StreamKey.root(SEED, "module", "ps_shipped/tests/test_shipped.py")
    assert draws[0] == draws[1]
    assert draws[0]["module"] == randint(module)


# ---------------------------------------------------------------------------
# Sessions in one process
# ---------------------------------------------------------------------------


NESTED_STRATEGIES = """
    from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, register

    RNG.seed(11)
    DRAW = RNG.integer(0, 10**9)

    @register("ps_nested")
    def nested(rng):
        RNG.seed(12)
        return Parameter(
            TestArg("d", value=(DRAW, rng.random())),
            TestArg("x", rng_type=RNGInteger(0, 10**9)),
            nsamples=3,
        )
"""

NESTED_TESTS = """
    from pytest_strategy import RNG, strategy

    @strategy("ps_nested")
    def test_nested(d, x):
        RNG.seed(13)
        RNG.integer(0, 9)
"""


class TestSessionsInOneProcess:
    @pytest.fixture
    def project(self, pytester, values_dump):
        pytester.makepyfile(strategies=NESTED_STRATEGIES, test_ps_nested=NESTED_TESTS)
        pytester.makeconftest(values_dump.conftest)
        return values_dump

    @pytest.mark.parametrize("args", [[], [f"--rng-seed={SEED}"]], ids=["seed-kept", "rng-seed"])
    def test_a_nested_session_leaves_the_outer_state(self, project, args):
        RNG.seed(1234)
        RNG.generator().random()
        before = (RNG.get_seed(), RNG._ambient.getstate())

        project.run(*args, subprocess=False)

        assert (RNG.get_seed(), RNG._ambient.getstate()) == before
        assert RNG.generator() is RNG._ambient

    def test_two_sessions_in_one_process_give_the_same_rows(self, project):
        RNG.seed(1234)

        first = project.collect(subprocess=False)
        second = project.collect(subprocess=False)

        assert len(first) == 3
        assert second == first
