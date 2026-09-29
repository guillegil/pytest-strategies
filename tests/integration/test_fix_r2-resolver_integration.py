"""
End-to-end regression tests for the second round of resolver fixes, run through pytester.

Each test here failed before its fix. Runs that depend on the process (environment,
working directory, pyc cache, xdist) use a subprocess; the others run in-process.
"""

import random
import re
import shutil
import textwrap

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what the in-process runs change globally: the registry, the RNG seed and random state."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


def _ids(result, test_name):
    """The parametrization IDs of ``test_name`` in a --collect-only -q run."""
    return re.findall(rf"{test_name}\[(.*)\]", result.stdout.str())


# ---------------------------------------------------------------------------
# Decorated factories
# ---------------------------------------------------------------------------


class TestDecoratedFactories:
    """Factories behind functools.wraps, mock.patch or *args wrappers are called correctly."""

    def test_decorated_factories_run(self, pytester):
        pytester.makepyfile(deco_strategies="""
            import functools
            import os
            from unittest import mock

            from pytest_strategy import Strategy

            def with_rng(fn):
                @functools.wraps(fn)
                def wrapper(nsamples):
                    return fn(nsamples, 7)
                return wrapper

            def adapt(fn):
                @functools.wraps(fn)
                def wrapper(nsamples):
                    return fn()
                return wrapper

            def logged(fn):
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return wrapper

            @Strategy.register("r2_injected")
            @with_rng
            def injected(nsamples, rng):
                return ("x",), [(rng,)] * nsamples

            @Strategy.register("r2_patched")
            @mock.patch("os.getcwd", return_value="/fake")
            def patched(nsamples, getcwd):
                return ("x",), [(os.getcwd(),)] * nsamples

            @Strategy.register("r2_adapted")
            @adapt
            def adapted():
                return ("x",), [(1,), (2,)]

            @Strategy.register("r2_logged")
            @logged
            def logged_factory(n):
                return ("x",), [(i,) for i in range(n)]
            """)
        pytester.makepyfile(test_deco="""
            from pytest_strategy import Strategy

            @Strategy.strategy("r2_injected")
            def test_injected(x):
                assert x == 7

            @Strategy.strategy("r2_patched")
            def test_patched(x):
                assert x == "/fake"

            @Strategy.strategy("r2_adapted")
            def test_adapted(x):
                assert x in (1, 2)

            @Strategy.strategy("r2_logged")
            def test_logged(x):
                assert 0 <= x < 3
            """)

        result = pytester.runpytest_inprocess("--nsamples=3")

        result.assert_outcomes(passed=3 + 3 + 2 + 3)


# ---------------------------------------------------------------------------
# Duplicate registration of one file reached through two path strings
# ---------------------------------------------------------------------------


class TestDuplicateRegistrationSameFile:
    """No warning, even as an error, when the same file is registered twice."""

    def test_testpaths_outside_rootdir(self, pytester, monkeypatch):
        """The plugin loads proj/../shared/x.py, the test module imports shared/x.py."""
        (pytester.path / "proj").mkdir()
        (pytester.path / "proj" / "pytest.ini").write_text("[pytest]\ntestpaths = ../shared\n")
        shared = pytester.mkdir("shared")
        (shared / "api_strategies.py").write_text(
            "from dataclasses import dataclass\n"
            "from pytest_strategy import Strategy\n"
            "\n"
            "@dataclass\n"
            "class Req:\n"
            "    method: str\n"
            "    path: str\n"
            "\n"
            '@Strategy.register("r2_reqs")\n'
            "def make(nsamples):\n"
            '    return ("method", "path"), [("GET", "/a"), ("POST", "/b")]\n'
        )
        (shared / "test_api.py").write_text(
            "from api_strategies import Req\n"
            "from pytest_strategy import Strategy\n"
            "\n"
            '@Strategy.strategy("r2_reqs")\n'
            "def test_req(r: Req):\n"
            "    assert isinstance(r, Req)\n"
        )
        monkeypatch.chdir(pytester.path / "proj")

        result = pytester.runpytest_subprocess("-W", "error::UserWarning")

        result.assert_outcomes(passed=2)

    def test_moved_checkout_with_stale_pyc(self, pytester, monkeypatch):
        """pytest's rewritten pyc keeps the co_filename of the checkout it was cached in."""
        original = pytester.mkdir("original")
        tests = original / "tests"
        tests.mkdir()
        (tests / "test_strategies.py").write_text(
            "from pytest_strategy import Strategy\n"
            "\n"
            '@Strategy.register("r2_inline")\n'
            "def make(nsamples):\n"
            '    return ("x",), [(1,), (2,)]\n'
            "\n"
            '@Strategy.strategy("r2_inline")\n'
            "def test_inline(x):\n"
            "    pass\n"
        )
        monkeypatch.chdir(original)
        pytester.runpytest_subprocess("-W", "error::UserWarning").assert_outcomes(passed=2)
        assert list((tests / "__pycache__").glob("test_strategies*pytest*.pyc"))

        moved = pytester.path / "moved"
        shutil.copytree(original, moved)  # copy2 keeps mtimes, so the pyc stays valid
        monkeypatch.chdir(moved)

        result = pytester.runpytest_subprocess("-W", "error::UserWarning")

        result.assert_outcomes(passed=2)


# ---------------------------------------------------------------------------
# Dataclass-typed fixture consuming the argnames (validate_signature=False)
# ---------------------------------------------------------------------------

HOSTPORT_STRATEGIES = """
    from pytest_strategy import Strategy

    @Strategy.register("r2_hostport")
    def hostport(nsamples):
        return ("host", "port"), [("localhost", 8000), ("127.0.0.1", 9000)]
    """

SERVER_FIXTURES = textwrap.dedent("""
    from dataclasses import dataclass

    import pytest

    @dataclass
    class Server:
        host: str
        port: int
        {extra_field}

    STARTED = []

    @pytest.fixture
    def server(host, port):
        STARTED.append((host, port))
        yield Server(host, port)

    @pytest.fixture
    def client():
        return "client"
    """)

SERVER_TEST = textwrap.dedent("""
    from pytest_strategy import Strategy

    @Strategy.strategy("r2_hostport", validate_signature=False)
    def test_server(server: Server, client):
        assert (server.host, server.port) in STARTED
    """)


class TestDataclassTypedFixture:
    """The fixture receives the argnames and runs; the test is not in dataclass mode."""

    @pytest.mark.parametrize("extra_field", ["", "started: bool = False"])
    def test_fixture_in_test_module(self, pytester, extra_field):
        pytester.makepyfile(hostport_strategies=HOSTPORT_STRATEGIES)
        fixtures = SERVER_FIXTURES.replace("{extra_field}", extra_field)
        pytester.makepyfile(test_server=fixtures + SERVER_TEST)

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(passed=2)

    def test_fixture_in_conftest(self, pytester):
        pytester.makepyfile(hostport_strategies=HOSTPORT_STRATEGIES)
        pytester.makeconftest(SERVER_FIXTURES.replace("{extra_field}", ""))
        pytester.makepyfile(test_server="from conftest import STARTED, Server\n" + SERVER_TEST)

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(passed=2)


# ---------------------------------------------------------------------------
# Test IDs: strings containing " at 0x", and sets
# ---------------------------------------------------------------------------


class TestIdsInRealRuns:
    """IDs keep string data and do not depend on PYTHONHASHSEED."""

    def test_strings_containing_at_0x_keep_their_value(self, pytester):
        pytester.makepyfile(fault_strategies="""
            from pytest_strategy import Strategy

            @Strategy.register("r2_lines")
            def lines(nsamples):
                return ("line",), [("segfault at 0x0",), ("jump at 0x401000",)]

            @Strategy.register("r2_pairs")
            def pairs(nsamples):
                return ("msg", "code"), [("fault at 0x10", 1), ("fault at 0x20", 2)]
            """)
        pytester.makepyfile(test_faults="""
            from pytest_strategy import Strategy

            @Strategy.strategy("r2_lines")
            def test_parse(line):
                pass

            @Strategy.strategy("r2_pairs")
            def test_pairs(msg, code):
                pass
            """)

        result = pytester.runpytest_inprocess("--collect-only", "-q")

        assert _ids(result, "test_parse") == ["line='segfault at 0x0'", "line='jump at 0x401000'"]
        assert _ids(result, "test_pairs") == [
            "msg='fault at 0x10',code=1",
            "msg='fault at 0x20',code=2",
        ]

    SET_STRATEGIES = """
        from pytest_strategy import Parameter, RNGChoice, Strategy, TestArg

        @Strategy.register("r2_perms")
        def perms(nsamples):
            return Parameter(
                TestArg(
                    "perms",
                    rng_type=RNGChoice(
                        [frozenset({"read", "write", "admin", "exec"}), frozenset({"read"})]
                    ),
                ),
                TestArg("user", rng_type=RNGChoice(["alice", "bob"])),
                directed_vectors={"all": ({"read", "write", "admin", "exec", "delete"}, "root")},
            )
        """
    SET_TESTS = """
        from pytest_strategy import Strategy

        @Strategy.strategy("r2_perms")
        def test_perms(perms, user):
            pass
        """

    def test_set_ids_do_not_depend_on_the_hash_seed(self, pytester, monkeypatch):
        pytester.makepyfile(set_strategies=self.SET_STRATEGIES)
        pytester.makepyfile(test_sets=self.SET_TESTS)

        collected = []
        for hash_seed in ("1", "2", "3"):
            monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
            result = pytester.runpytest_subprocess("--collect-only", "-q", "--rng-seed=1")
            collected.append(_ids(result, "test_perms"))

        assert len(collected[0]) == 11
        assert collected[0][0] == "perms={'admin', 'delete...,user='root'"
        assert collected[1] == collected[0]
        assert collected[2] == collected[0]

    def test_xdist_workers_collect_the_same_set_ids(self, pytester, monkeypatch):
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        pytester.makepyfile(set_strategies=self.SET_STRATEGIES)
        pytester.makepyfile(test_sets=self.SET_TESTS)

        result = pytester.runpytest_subprocess("-n", "3", "--rng-seed=1")

        result.stdout.no_fnmatch_line("*Different tests were collected*")
        result.assert_outcomes(passed=11)
