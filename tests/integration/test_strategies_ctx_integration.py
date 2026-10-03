"""
End-to-end tests for the ``strategies_ctx`` fixture and ``get_context()`` (D8):
tests and fixtures get the object the factories received, a session fixture whose
tests span two contexts fails each of them, and a folder's conftest.py gets its
own folder's context with ``get_context(request.config, __file__)``.
"""

import random
from collections import Counter
from textwrap import dedent

import pytest

from pytest_strategy._streams import StreamKey

pytest_plugins = ["pytester"]

# Writes each call of an implementation to calls.txt; SEEN keeps the object each
# factory call received
CALLS = """
from pathlib import Path

SEEN = []

def answer(config, name):
    with open(Path(config.rootpath, "calls.txt"), "a") as calls:
        calls.write(name + "\\n")
    return {"name": name}

def seen(name):
    \"\"\"The object the factories received in the folders answered by ``name``.\"\"\"
    found = {id(ctx): ctx for ctx in SEEN if ctx is not None and ctx["name"] == name}
    assert len(found) == 1, SEEN
    return next(iter(found.values()))

# Every Testbench the tb fixture built
TBS = []
"""

STRATEGIES = """
import ctx_calls
from pytest_strategy import Parameter, TestArg, register

@register("bench")
def bench(ctx):
    ctx_calls.SEEN.append(ctx)
    return Parameter(TestArg("name", value=None if ctx is None else ctx["name"]), nsamples=1)
"""


def conftest(name, extra=""):
    """
    A conftest.py whose implementation answers ``{"name": name}``, or returns None
    (its call recorded as "none") when ``name`` is None, followed by ``extra``.
    """
    body = "return None" if name is None else f"return answer(config, {name!r})"
    record = "" if name is not None else '    answer(config, "none")\n'
    return f"""
import pytest
from ctx_calls import answer
from pytest_strategy import get_context

def pytest_strategies_context(config):
{record}    {body}
{extra}"""


# A session-scoped testbench built on the context
TB_FIXTURE = """
import ctx_calls

class Testbench:
    def __init__(self, ctx):
        self.ctx = ctx
        ctx_calls.TBS.append(self)

@pytest.fixture(scope="session")
def tb(strategies_ctx):
    return Testbench(strategies_ctx)
"""


def tb_tests(name):
    """A test module whose strategy test and tb test get the context ``name``."""
    return f"""
import ctx_calls
from pytest_strategy import strategy

@strategy("bench")
def test_factory_and_fixture(name, strategies_ctx):
    assert name == {name!r}
    assert strategies_ctx is ctx_calls.seen({name!r})

def test_tb(tb):
    assert tb.ctx is ctx_calls.seen({name!r})
"""


def write(pytester, files):
    """Write ``files`` ({relative path: text}) below the pytester project."""
    pytester.makeini(f"[pytest]\npythonpath = {pytester.path.as_posix()}\n")
    for name, text in {"ctx_calls.py": CALLS, **files}.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))


def calls(pytester):
    """The implementations the last run called, with how often."""
    path = pytester.path / "calls.txt"
    found = Counter(path.read_text().split()) if path.exists() else Counter()
    path.unlink(missing_ok=True)
    return found


# The guard's message for the two-scope project below
GUARD = (
    "strategies_ctx is a session fixture, but the tests that use it have different contexts "
    "(conftest.py: tests/test_root.py::test_root; tests/tb_a/conftest.py: "
    "tests/tb_a/test_a.py::test_a and 1 more). In a folder with its own "
    "pytest_strategies_context, use pytest_strategy.get_context(request.config, __file__) "
    "in that folder's conftest.py fixtures."
)


class TestOneScope:
    @staticmethod
    def one_scope_project(pytester):
        """
        A rootdir conftest.py answering "root" with a session ``tb`` fixture, tests in
        tests/ (the parent folder) and in tests/sub, whose conftest.py returns None.
        """
        tbs = TB_FIXTURE + """
def pytest_sessionfinish(session):
    (session.config.rootpath / "tbs.txt").write_text(str(len(ctx_calls.TBS)))
"""
        write(
            pytester,
            {
                "conftest.py": conftest("root", tbs),
                "tests/strategies.py": STRATEGIES,
                "tests/test_top.py": tb_tests("root"),
                "tests/sub/conftest.py": conftest(None),
                "tests/sub/test_sub.py": tb_tests("root"),
            },
        )

    def test_the_fixture_is_the_object_the_factories_received(self, pytester):
        self.one_scope_project(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=4)
        # tests/sub's implementation defers to the rootdir's, which is called once
        assert calls(pytester) == {"root": 1, "none": 1}
        # One tb for the session, in the parent folder and the child folder
        assert (pytester.path / "tbs.txt").read_text() == "1"

    def test_a_wrapper_keeps_one_object_for_the_parent_and_the_child(self, pytester):
        wrapper = """

@pytest.hookimpl(wrapper=True, specname="pytest_strategies_context")
def pytest_wrap_context(config):
    ctx = yield
    with open(config.rootpath / "calls.txt", "a") as calls:
        calls.write("wrap\\n")
    return {**ctx, "wrapped": True}
"""
        self.one_scope_project(pytester)
        conftest_py = pytester.path / "conftest.py"
        conftest_py.write_text(conftest_py.read_text() + wrapper)

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=4)
        # The wrapper runs once around the rootdir's answer, for both folders
        assert calls(pytester) == {"root": 1, "none": 1, "wrap": 1}
        assert (pytester.path / "tbs.txt").read_text() == "1"
        result.stdout.fnmatch_lines(["pytest-strategies: context ????????"])

    def test_works_under_xdist(self, pytester):
        pytest.importorskip("xdist")
        self.one_scope_project(pytester)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-n", "2")

        result.assert_outcomes(passed=4)

    def test_the_fixture_computes_the_context_when_no_factory_did(self, pytester):
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "test_plain.py": """
def test_plain(strategies_ctx):
    assert strategies_ctx == {"name": "root"}

def test_other(strategies_ctx):
    assert strategies_ctx == {"name": "root"}
""",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=2)
        assert calls(pytester) == {"root": 1}

    def test_the_hook_is_not_called_when_no_test_uses_the_fixture(self, pytester):
        write(
            pytester,
            {"conftest.py": conftest("root"), "test_plain.py": "def test_x():\n    pass\n"},
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)
        assert calls(pytester) == {}

    def test_none_when_nothing_answers(self, pytester):
        write(
            pytester,
            {
                "conftest.py": conftest(None),
                "test_plain.py": "def test_x(strategies_ctx):\n    assert strategies_ctx is None\n",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)

    def test_a_dynamic_request_gets_the_object(self, pytester):
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "test_plain.py": """
def test_x(request):
    assert request.getfixturevalue("strategies_ctx") == {"name": "root"}
""",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)


class TestTwoScopes:
    @staticmethod
    def two_scope_project(pytester):
        """
        A rootdir conftest.py answering "root" with a session ``tb`` fixture, and
        tests/tb_a/conftest.py answering "A": tb's tests are in both scopes.
        """
        write(
            pytester,
            {
                "conftest.py": conftest("root", TB_FIXTURE),
                "tests/tb_a/conftest.py": conftest("A"),
                "tests/test_root.py": """
def test_root(tb):
    assert tb.ctx == {"name": "root"}

def test_plain():
    pass
""",
                "tests/tb_a/test_a.py": """
def test_a(tb):
    assert tb.ctx == {"name": "A"}

def test_a2(tb):
    assert tb.ctx == {"name": "A"}

def test_a_plain():
    pass
""",
            },
        )

    def test_every_test_that_uses_it_fails_with_the_labels_and_get_context(self, pytester):
        self.two_scope_project(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2, errors=3)
        # The message is the whole of each error's lines (pytrace=False); the short
        # test summary may repeat it (under CI, or with wide terminals)
        assert result.stdout.lines.count(GUARD) == 3, result.stdout.str()
        for test in ("test_root", "test_a", "test_a2"):
            result.stdout.fnmatch_lines([f"*ERROR at setup of {test} *"])

    def test_under_xdist_every_worker_reports_the_same_tests(self, pytester):
        pytest.importorskip("xdist")
        self.two_scope_project(pytester)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-n", "2")

        result.assert_outcomes(passed=2, errors=3)
        assert result.stdout.lines.count(GUARD) == 3, result.stdout.str()

    @pytest.mark.parametrize(
        "args",
        [
            ("--deselect=tests/tb_a/test_a.py::test_a", "--deselect=tests/tb_a/test_a.py::test_a2"),
            ("--deselect=tests/test_root.py::test_root",),
            ("-k", "not test_root"),
            ("tests/tb_a",),
        ],
        ids=["without_a", "without_root", "k", "one_folder"],
    )
    def test_deselecting_one_scope_s_tests_makes_the_run_pass(self, pytester, args):
        self.two_scope_project(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider", *args)

        assert result.ret == 0, result.stdout.str()
        assert result.parseoutcomes().get("errors", 0) == 0

    def test_a_dynamic_request_counts_every_test(self, pytester):
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "tests/tb_a/conftest.py": conftest("A"),
                "tests/test_root.py": """
def test_dynamic(request):
    request.getfixturevalue("strategies_ctx")
""",
                "tests/tb_a/test_a.py": "def test_a_plain():\n    pass\n",
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        # test_a_plain does not ask for it, but any test might
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(
            [
                "*have different contexts (conftest.py: tests/test_root.py::test_dynamic; "
                "tests/tb_a/conftest.py: tests/tb_a/test_a.py::test_a_plain)*"
            ]
        )
        pytester.runpytest("-p", "no:cacheprovider", "tests/test_root.py").assert_outcomes(passed=1)

    # A folder's own strategies_ctx, which does not build on the plugin's
    OVERRIDE = """
@pytest.fixture(scope="session")
def strategies_ctx(request):
    return get_context(request.config, __file__)
"""

    @pytest.mark.parametrize("dynamic", [False, True], ids=["requested", "dynamic"])
    def test_a_folder_that_overrides_it_does_not_use_it(self, pytester, dynamic):
        root = (
            'def test_root(request):\n    ctx = request.getfixturevalue("strategies_ctx")\n'
            if dynamic
            else "def test_root(strategies_ctx):\n    ctx = strategies_ctx\n"
        )
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "tests/test_root.py": root + '    assert ctx == {"name": "root"}\n',
                "tests/tb_a/conftest.py": conftest("A", self.OVERRIDE),
                "tests/tb_a/test_a.py": """
def test_a(strategies_ctx):
    assert strategies_ctx == {"name": "A"}
""",
                "tests/tb_b/test_b.py": """
import pytest

@pytest.mark.parametrize("strategies_ctx", [{"name": "B"}])
def test_b(strategies_ctx):
    assert strategies_ctx == {"name": "B"}
""",
                "tests/tb_b/conftest.py": conftest("B"),
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=3)

    def test_an_override_that_builds_on_it_uses_it(self, pytester):
        override = """
@pytest.fixture(scope="session")
def strategies_ctx(strategies_ctx):
    return {**strategies_ctx, "extended": True}
"""
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "tests/test_root.py": "def test_root(strategies_ctx):\n    pass\n",
                "tests/tb_a/conftest.py": conftest("A", override),
                "tests/tb_a/test_a.py": "def test_a(strategies_ctx):\n    pass\n",
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(errors=2)
        result.stdout.fnmatch_lines(
            [
                "*have different contexts (conftest.py: tests/test_root.py::test_root; "
                "tests/tb_a/conftest.py: tests/tb_a/test_a.py::test_a)*"
            ]
        )


class TestGetContext:
    def test_a_folder_s_conftest_gets_its_folder_s_object(self, pytester):
        fixture = """
@pytest.fixture(scope="session")
def tb_{name}(request):
    return get_context(request.config, __file__)
"""
        write(
            pytester,
            {
                "conftest.py": conftest("root", fixture.format(name="root")),
                "tests/strategies.py": STRATEGIES,
                "tests/tb_a/conftest.py": conftest("A", fixture.format(name="a")),
                "tests/test_root.py": """
import ctx_calls
from pytest_strategy import strategy

@strategy("bench")
def test_root(name, tb_root):
    assert tb_root is ctx_calls.seen("root")
""",
                "tests/tb_a/test_a.py": """
import ctx_calls
from pytest_strategy import get_context, strategy

@strategy("bench")
def test_a(name, tb_a, tb_root, request):
    assert tb_a is ctx_calls.seen("A")
    assert tb_root is ctx_calls.seen("root")
    rootpath = request.config.rootpath
    # A folder, a file in it, and a folder below it without a conftest.py
    assert get_context(request.config, rootpath / "tests" / "tb_a") is tb_a
    assert get_context(request.config, str(rootpath / "tests" / "tb_a" / "x.py")) is tb_a
    assert get_context(request.config, rootpath / "tests" / "tb_a" / "deep") is tb_a
    assert get_context(request.config, rootpath / "tests") is tb_root
""",
                "tests/tb_a/deep/data.txt": "",
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert calls(pytester) == {"root": 1, "A": 1}

    def test_an_uncollected_folder_gets_the_nearest_loaded_conftest_s_context(self, pytester):
        write(
            pytester,
            {
                "conftest.py": conftest("root"),
                "tests/a/test_a.py": """
from pytest_strategy import get_context

def test_b_context(request):
    rootpath = request.config.rootpath
    ctx = get_context(request.config, rootpath / "tests" / "b" / "test_b.py")
    (rootpath / "b.txt").write_text(ctx["name"])
""",
                "tests/b/conftest.py": conftest("B"),
                "tests/b/test_b.py": "def test_b():\n    pass\n",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=2)
        assert (pytester.path / "b.txt").read_text() == "B"
        assert calls(pytester) == {"B": 1}

        # pytest has not loaded tests/b/conftest.py: the rootdir's answers
        pytester.runpytest("-p", "no:cacheprovider", "tests/a").assert_outcomes(passed=1)
        assert (pytester.path / "b.txt").read_text() == "root"
        assert calls(pytester) == {"root": 1}

    def test_in_a_conftest_s_pytest_configure_it_sees_the_conftests_loaded_so_far(self, pytester):
        configure = """
import ctx_calls

def pytest_configure(config):
    ctx_calls.TBS.append(get_context(config, __file__))
    ctx_calls.TBS.append(get_context(config, config.rootpath / "tests" / "b"))
"""
        write(
            pytester,
            {
                "conftest.py": conftest("root", configure),
                "tests/test_root.py": """
import ctx_calls

def test_root(strategies_ctx):
    assert ctx_calls.TBS == [{"name": "root"}, {"name": "root"}]
    assert strategies_ctx is ctx_calls.TBS[0]
""",
                "tests/b/conftest.py": conftest("B"),
                "tests/b/test_b.py": """
from pytest_strategy import get_context

def test_b(request):
    assert get_context(request.config, __file__) == {"name": "B"}
""",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=2)
        assert calls(pytester) == {"root": 1, "B": 1}

    @pytest.mark.parametrize("early", [True, False], ids=["pytest_configure", "test"])
    def test_in_pytest_configure_it_draws_from_the_seed_rng_seed_gives(self, pytester, early):
        """
        The plugin seeds after the conftest.py files' pytest_configure, but a
        context computed there already uses the seed of --rng-seed.
        """
        configure = "get_context(config, __file__)" if early else "pass"
        write(
            pytester,
            {
                "conftest.py": f"""
from pytest_strategy import RNG, get_context

def pytest_strategies_context(config):
    return {{"draw": RNG.integer(0, 10**9)}}

def pytest_configure(config):
    {configure}
""",
                "test_draw.py": """
def test_draw(strategies_ctx, request):
    (request.config.rootpath / "draw.txt").write_text(str(strategies_ctx["draw"]))
""",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=7").assert_outcomes(passed=1)

        expected = random.Random(StreamKey.root(7, "ctx").seed_int()).randint(0, 10**9)
        assert (pytester.path / "draw.txt").read_text() == str(expected)


class TestHookErrors:
    TESTS = """
from pytest_strategy import get_context

def test_fixture(strategies_ctx):
    pass

def test_get_context(request):
    get_context(request.config, __file__)

def test_plain():
    pass
"""

    def test_the_hook_s_exception_is_raised_as_it_is(self, pytester):
        write(
            pytester,
            {
                "conftest.py": """
from ctx_calls import answer

def pytest_strategies_context(config):
    answer(config, "raised")
    raise LookupError("no bench file")
""",
                "test_ctx_errors.py": self.TESTS,
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=1, failed=1, errors=1)
        output = result.stdout.str()
        # Not wrapped in a message about the hook, and its frame is shown
        assert output.count("E       LookupError: no bench file") == 2, output
        assert 'raise LookupError("no bench file")' in output
        assert "pytest_strategies_context hook raised" not in output
        assert calls(pytester) == {"raised": 1}

    def test_a_skip_in_the_hook_skips_the_tests_that_use_it(self, pytester):
        write(
            pytester,
            {
                "conftest.py": """
import pytest

def pytest_strategies_context(config):
    pytest.skip("no bench configured", allow_module_level=True)
""",
                "test_ctx_errors.py": self.TESTS,
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", "-rs")

        result.assert_outcomes(passed=1, skipped=2)
        result.stdout.fnmatch_lines(["*no bench configured*"])


def test_a_config_of_a_session_that_ended_raises_runtime_error(pytester):
    pytester.makepyfile(
        test_ended="""
import pytest
from pytest_strategy import get_context

def test_ended(pytester):
    pytester.makepyfile(test_inner="def test_x(): pass")
    configs = []

    class Keep:
        def pytest_configure(self, config):
            configs.append(config)

    pytester.inline_run("-p", "no:cacheprovider", plugins=[Keep()])
    with pytest.raises(RuntimeError, match="no running pytest-strategies session"):
        get_context(configs[0], __file__)
""",
        conftest="pytest_plugins = ['pytester']\n",
    )

    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)


def test_the_fixture_is_listed_with_its_docstring(pytester):
    result = pytester.runpytest("-p", "no:cacheprovider", "--fixtures")

    result.stdout.fnmatch_lines(["strategies_ctx [[]session scope[]]*", "*testbench context*"])
