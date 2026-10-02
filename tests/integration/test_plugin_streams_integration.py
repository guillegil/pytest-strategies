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

        def fixture(scope, name, param_index, where):
            # The scope node's ID, the name and the parameter index, then where the
            # fixture is defined: its file and its function's qualified name
            return randint(StreamKey.root(SEED, "fixture", scope, name, param_index, where, name))

        assert records == {
            "module": randint(StreamKey.root(SEED, "module", BODY)),
            "test_first": body("test_first"),
            "sess": fixture("", "sess", 0, "conftest.py"),
            "mod": fixture(BODY, "mod", 0, BODY),
            "test_uses": body("test_uses"),
            "test_reseeds": random.Random(5).randint(0, 10**9),
            "par p": fixture(f"{BODY}::test_par[p]", "par", 0, BODY),
            "par q": fixture(f"{BODY}::test_par[q]", "par", 1, BODY),
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


# Fixtures that pytest sets up for the same scope node under one name: an override
# that requests the fixture it overrides, and session fixtures of one name in two
# sibling folders' conftest.py files

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
    from record import record

    @pytest.fixture(scope="session")
    def resource():
        drawn = RNG.integer(0, 10**9)
        record("resource {folder}", drawn)
        return drawn
"""


@pytest.fixture
def same_name_project(pytester):
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(record=RECORD)
    pytester.makeconftest(SAME_NAME_CONFTEST)
    tests = pytester.mkdir("tests")
    (tests / "test_override.py").write_text(dedent(SAME_NAME_OVERRIDE))
    for folder in ("a", "b"):
        (tests / folder).mkdir()
        (tests / folder / "conftest.py").write_text(
            dedent(SAME_NAME_SIBLING).replace("{folder}", folder)
        )
        (tests / folder / f"test_{folder}.py").write_text(
            f"def test_{folder}(resource):\n    pass\n"
        )
    return pytester


class TestFixturesOfOneName:
    def test_each_definition_draws_from_a_stream_of_its_own(self, same_name_project):
        records = run_and_read(same_name_project, passed=3)

        def fixture(scope, name, where):
            return randint(StreamKey.root(SEED, "fixture", scope, name, 0, where, name))

        override = "tests/test_override.py"
        assert records == {
            "value": fixture(f"{override}::test_override", "value", "conftest.py"),
            "value override": fixture(f"{override}::test_override", "value", override),
            "modvalue": fixture(override, "modvalue", "conftest.py"),
            "modvalue override": fixture(override, "modvalue", override),
            "resource a": fixture("", "resource", "tests/a/conftest.py"),
            "resource b": fixture("", "resource", "tests/b/conftest.py"),
        }
        # 4.0's first key had no definition: these pairs drew the same values
        assert records["value"] != records["value override"]
        assert records["modvalue"] != records["modvalue override"]
        assert records["resource a"] != records["resource b"]

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

    @register("ps_export")
    def exported(rng):
        out = pathlib.Path(__file__).parent / "export_draws.jsonl"
        with open(out, "a") as f:
            f.write(json.dumps(rng.random()) + "\\n")
        return Parameter(TestArg("e", value=1), nsamples=1)
"""

EXPORT_TESTS = """
    from pytest_strategy import export_strategies

    def test_export():
        export_strategies()
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
    key = StreamKey.root(SEED, "export", "ps_export", "tests/a")
    assert [json.loads(line) for line in lines] == [random.Random(key.seed_int()).random()] * 2


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
