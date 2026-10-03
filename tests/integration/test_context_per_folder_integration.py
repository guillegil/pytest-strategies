"""
End-to-end tests for the context per folder (D7): a test's factories get the
context of the test's folder, from the pytest_strategies_context implementations
that folder sees, asked in the plugin's own order (tryfirst, the conftest.py files
from the test's folder upward, the other plugins, trylast), each at most once per
session.

The main project has a rootdir conftest.py answering "root", tests/a/conftest.py
answering "A", tests/a/deep/conftest.py answering None, and one factory in
tests/strategies.py that every test uses. In 3.0 the hook was one pluggy call per
session: tests/b got "A" in a full run (tests/a's conftest.py was registered last),
and the result depended on which folders the command line named. Each run here
must give the same node IDs and values, whichever folders it collects.
"""

import json
from collections import Counter
from pathlib import Path
from textwrap import dedent

pytest_plugins = ["pytester"]

SEED = 3

# Writes each call of an implementation to calls.txt, and builds what it returns
# from a draw on the hook's random stream, so a run's values show which
# implementation answered and with which draw
CALLS = """
from pathlib import Path

from pytest_strategy import RNG

# The object each factory call received
SEEN = []

def answer(config, name):
    with open(Path(config.rootpath, "calls.txt"), "a") as calls:
        calls.write(name + "\\n")
    return {"name": name, "draw": RNG.integer(0, 10**9)}
"""

ROOT_CONFTEST = """
from ctx_calls import answer

def pytest_strategies_context(config):
    return answer(config, "root")
"""

A_CONFTEST = """
from ctx_calls import answer

def pytest_strategies_context(config):
    return answer(config, "A")
"""

DEEP_CONFTEST = """
from ctx_calls import answer

def pytest_strategies_context(config):
    answer(config, "deep")
    return None
"""

# Records the object each factory call received
STRATEGIES = """
import ctx_calls
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("shared")
def shared(ctx):
    ctx_calls.SEEN.append(ctx)
    return Parameter(
        TestArg("name", value=ctx["name"]),
        TestArg("draw", value=ctx["draw"]),
        TestArg("x", rng_type=RNGInteger(0, 10**6)),
        nsamples=2,
    )
"""


def ctx_test_module(expected):
    return f"""
from pytest_strategy import strategy

@strategy("shared")
def test_ctx(name, draw, x):
    assert name == {expected!r}
"""


def write(pytester, files):
    """Write ``files`` ({relative path: text}) below the pytester project."""
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))


def project(pytester, values_dump, *, root=ROOT_CONFTEST, b="root", extra=None):
    """
    Write the main project: the rootdir conftest.py ``root`` (with the values dump),
    tests/a answering "A", tests/a/deep answering None, and a test in each folder,
    tests/b's expecting ``b``.
    """
    pytester.makeini(f"[pytest]\npythonpath = {pytester.path.as_posix()}\n")
    write(
        pytester,
        {
            "ctx_calls.py": CALLS,
            "conftest.py": root + values_dump.conftest,
            "tests/strategies.py": STRATEGIES,
            "tests/a/conftest.py": A_CONFTEST,
            "tests/a/deep/conftest.py": DEEP_CONFTEST,
            "tests/a/test_a.py": ctx_test_module("A"),
            "tests/a/deep/test_deep.py": ctx_test_module("A"),
            "tests/b/test_b.py": ctx_test_module(b),
            **(extra or {}),
        },
    )


def calls(pytester):
    """The implementations the last run called, with how often."""
    path = pytester.path / "calls.txt"
    found = Counter(path.read_text().split()) if path.exists() else Counter()
    path.unlink(missing_ok=True)
    return found


# The runs of the regression test: a full run, then runs that collect some folders
RUNS = [(), ("tests/b",), ("tests/a/deep/test_deep.py",), ("tests/b", "tests/a")]


def check_runs(pytester, values_dump, *args):
    """
    Run pytest for each of RUNS and check that every run gives the full run's node
    IDs and values; return the full run's rows by node ID.
    """
    full = None
    for paths in RUNS:
        rows = dict(
            values_dump.collect(
                "-p", "no:cacheprovider", f"--rng-seed={SEED}", *args, *paths, subprocess=False
            )
        )
        calls(pytester)
        if full is None:
            full = rows
            assert len(full) == 6
        else:
            assert rows, paths
            assert {node: full.get(node) for node in rows} == rows, paths
    assert full is not None
    return full


def names(rows):
    """The context name each test file's rows got."""
    found = {}
    for node, params in rows.items():
        name = params.split("'name': ")[1].split(",")[0].strip("'")
        found.setdefault(node.split("::")[0], set()).add(name)
    return {file: names.pop() for file, names in found.items() if len(names) == 1}


class TestEveryRunGetsTheFoldersContexts:
    """The regression test for both 3.0 failures, in the four runs."""

    def test_nearest_conftest_wins_in_every_run(self, pytester, values_dump):
        project(pytester, values_dump)

        full = check_runs(pytester, values_dump)

        assert names(full) == {
            "tests/a/test_a.py": "A",
            "tests/a/deep/test_deep.py": "A",
            "tests/b/test_b.py": "root",
        }

    def test_a_plugin_the_rootdir_conftest_registers_does_not_answer_over_it(
        self, pytester, values_dump
    ):
        plugin = """
class TbPlugin:
    def pytest_strategies_context(self, config):
        return answer(config, "plugin")

def pytest_configure(config):
    config.pluginmanager.register(TbPlugin(), "tb_plugin")
"""
        project(pytester, values_dump, root=ROOT_CONFTEST + plugin)

        full = check_runs(pytester, values_dump)

        assert names(full) == {
            "tests/a/test_a.py": "A",
            "tests/a/deep/test_deep.py": "A",
            "tests/b/test_b.py": "root",
        }

    def test_the_plugin_answers_where_no_conftest_implements_the_hook(self, pytester, values_dump):
        root = """
from ctx_calls import answer

class TbPlugin:
    def pytest_strategies_context(self, config):
        return answer(config, "plugin")

def pytest_configure(config):
    config.pluginmanager.register(TbPlugin(), "tb_plugin")
"""
        project(pytester, values_dump, root=root, b="plugin")

        full = check_runs(pytester, values_dump)

        assert names(full) == {
            "tests/a/test_a.py": "A",
            "tests/a/deep/test_deep.py": "A",
            "tests/b/test_b.py": "plugin",
        }

    def test_draws_in_each_implementation_are_the_same_in_every_run(self, pytester, values_dump):
        project(pytester, values_dump)

        full = check_runs(pytester, values_dump)

        # Each implementation draws the first value of root(S, "ctx"), whichever
        # folder asked first: the full run asks tests/a/deep first, the last run tests/b
        draws = {params.split("'draw': ")[1].split(",")[0] for params in full.values()}
        assert len(draws) == 1


class TestCalls:
    def test_each_implementation_is_called_once_and_none_shares_the_parent(
        self, pytester, values_dump
    ):
        project(pytester, values_dump)
        pytester.makeconftest(ROOT_CONFTEST + """
import json

from pytest_strategy._runtime import runtime

def pytest_collection_finish(session):
    from ctx_calls import SEEN as ctx

    rootpath = session.config.rootpath
    labels = {
        folder: runtime.path_context(rootpath / folder).answer().label
        for folder in ("tests/a", "tests/a/deep", "tests/b")
    }
    (rootpath / "seen.json").write_text(json.dumps({
        "objects": len({id(c) for c in ctx}), "factories": len(ctx), "labels": labels,
    }))
""")

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=6)
        # The rootdir's for tests/b, tests/a's for tests/a and tests/a/deep, and
        # tests/a/deep's once, which defers to tests/a's without calling it again
        assert calls(pytester) == {"root": 1, "A": 1, "deep": 1}
        seen = json.loads((pytester.path / "seen.json").read_text())
        # Three factory calls, two objects: tests/a and tests/a/deep share one
        assert (seen["factories"], seen["objects"]) == (3, 2)
        assert seen["labels"] == {
            "tests/a": "tests/a/conftest.py",
            "tests/a/deep": "tests/a/conftest.py",
            "tests/b": "conftest.py",
        }

    def test_a_folder_that_needs_one_implementation_calls_only_it(self, pytester, values_dump):
        project(pytester, values_dump)

        pytester.runpytest("-p", "no:cacheprovider", "tests/b").assert_outcomes(passed=2)
        assert calls(pytester) == {"root": 1}
        pytester.runpytest("-p", "no:cacheprovider", "tests/a/test_a.py").assert_outcomes(passed=2)
        assert calls(pytester) == {"A": 1}


class TestOrder:
    def test_a_wrapper_extends_the_answer(self, pytester, values_dump):
        wrapper = """
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_strategies_context(config):
    ctx = yield
    return {**ctx, "name": ctx["name"] + "+w"}
"""
        project(
            pytester,
            values_dump,
            extra={"tests/a/w/conftest.py": wrapper, "tests/a/w/test_w.py": ctx_test_module("A+w")},
        )

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=8)
        assert calls(pytester) == {"root": 1, "A": 1, "deep": 1}

    def test_a_wrapper_that_changes_the_object_it_receives_fails_its_folders(
        self, pytester, values_dump
    ):
        """
        The object it would change is the one tests/b gets without the wrapper, so
        tests/b's values would depend on whether tests/w was collected.
        """
        wrapper = """
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_strategies_context(config):
    ctx = yield
    ctx["name"] += "+w"
    return ctx
"""
        project(
            pytester,
            values_dump,
            extra={"tests/w/conftest.py": wrapper, "tests/w/test_w.py": ctx_test_module("root+w")},
        )
        message = (
            "*pytest_strategies_context hook raised RuntimeError: a pytest_strategies_context "
            "wrapper (tests/w/conftest.py) changed the object conftest.py returned. A wrapper "
            "must return a new object, such as {**ctx, ...}, and leave the one it receives as "
            "it is: the folders that do not see the wrapper get that object too"
        )

        for paths in [(), ("tests/w",)]:
            result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}", *paths)
            result.assert_outcomes(errors=1)
            result.stdout.fnmatch_lines(["*ERROR collecting tests/w/test_w.py*", message])

        rows = values_dump.collect("-p", "no:cacheprovider", f"--rng-seed={SEED}", "tests/b")
        assert names(dict(rows)) == {"tests/b/test_b.py": "root"}

    def test_a_tryfirst_rootdir_implementation_wins_everywhere(self, pytester, values_dump):
        root = """
import pytest
from ctx_calls import answer

@pytest.hookimpl(tryfirst=True)
def pytest_strategies_context(config):
    return answer(config, "root")
"""
        project(pytester, values_dump, root=root)
        write(
            pytester,
            {
                "tests/a/test_a.py": ctx_test_module("root"),
                "tests/a/deep/test_deep.py": ctx_test_module("root"),
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=6)
        assert calls(pytester) == {"root": 1}

    def test_a_p_plugin_answers_only_where_no_conftest_does(self, pytester, values_dump):
        root = "from ctx_calls import answer\n"
        plugin = """
from ctx_calls import answer

def pytest_strategies_context(config):
    return answer(config, "plugin")
"""
        project(pytester, values_dump, root=root, b="plugin", extra={"ctx_plugin.py": plugin})
        pytester.syspathinsert()

        result = pytester.runpytest("-p", "no:cacheprovider", "-p", "ctx_plugin")

        result.assert_outcomes(passed=6)
        assert calls(pytester) == {"A": 1, "deep": 1, "plugin": 1}

    def test_a_factory_registered_elsewhere_gets_the_test_s_folder_context(self, pytester):
        write(
            pytester,
            {
                "conftest.py": "def pytest_strategies_context(config):\n    return 'root'\n",
                "tests/a/conftest.py": "def pytest_strategies_context(config):\n    return 'A'\n",
                "tests/b/conftest.py": "def pytest_strategies_context(config):\n    return 'B'\n",
                "tests/a/a_strategies.py": """
from pytest_strategy import Parameter, TestArg, register

@register("from_a")
def from_a(ctx):
    return Parameter(TestArg("name", value=ctx), nsamples=1)
""",
                "tests/a/test_in_a.py": """
from pytest_strategy import strategy

@strategy("from_a")
def test_in_a(name):
    assert name == "A"
""",
                "tests/b/test_in_b.py": """
from pytest_strategy import strategy

@strategy("from_a")
def test_in_b(name):
    assert name == "B"
""",
            },
        )

        pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=2)
        pytester.runpytest("-p", "no:cacheprovider", "tests/b").assert_outcomes(passed=1)


class TestErrors:
    @staticmethod
    def broken_project(pytester, values_dump, raise_line):
        project(
            pytester,
            values_dump,
            extra={
                "tests/a/conftest.py": f"""
import pytest

def pytest_strategies_context(config):
    {raise_line}
""",
                "tests/a/test_plain.py": """
from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

@register("plain_a")
def plain_a():
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=2)

@strategy("plain_a")
def test_plain(x):
    pass

def test_without_strategy():
    pass
""",
            },
        )

    def test_a_raising_hook_fails_only_its_folder_s_ctx_modules(self, pytester, values_dump):
        self.broken_project(pytester, values_dump, 'raise RuntimeError("A is broken")')

        result = pytester.runpytest("-p", "no:cacheprovider", "--continue-on-collection-errors")

        # tests/a and tests/a/deep (which asks tests/a's before any answer) fail;
        # tests/a/test_plain.py and tests/b pass
        result.assert_outcomes(passed=5, errors=2)
        for line in (
            "*ERROR collecting tests/a/deep/test_deep.py*",
            "*ERROR collecting tests/a/test_a.py*",
            "*pytest_strategies_context hook raised RuntimeError: A is broken*",
        ):
            result.stdout.fnmatch_lines([line])

    def test_a_skipping_hook_skips_only_its_folder_s_ctx_modules(self, pytester, values_dump):
        self.broken_project(
            pytester, values_dump, 'pytest.skip("no bench for A", allow_module_level=True)'
        )

        result = pytester.runpytest("-p", "no:cacheprovider", "-rs")

        result.assert_outcomes(passed=5, skipped=2)
        result.stdout.fnmatch_lines(["*no bench for A*"])

    def test_the_migration_hint_names_the_conftest_elsewhere(self, pytester):
        write(
            pytester,
            {
                "tests/a/conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'name': 'A'}\n",
                "tests/strategies.py": """
from pytest_strategy import Parameter, TestArg, register

@register("bench")
def bench(ctx):
    return Parameter(TestArg("name", value=ctx["name"]), nsamples=1)
""",
                "tests/a/test_a.py": ctx_test_module("A")
                .replace('"shared"', '"bench"')
                .replace("name, draw, x", "name"),
                "tests/b/test_b.py": ctx_test_module("B")
                .replace('"shared"', '"bench"')
                .replace("name, draw, x", "name"),
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", "--continue-on-collection-errors")

        result.assert_outcomes(passed=1, errors=1)
        result.stdout.fnmatch_lines(
            [
                "*In test_ctx: Error calling strategy factory 'bench' (nsamples=10): "
                "TypeError: 'NoneType' object is not subscriptable. ctx is None for "
                "tests/b: no pytest_strategies_context implementation in this folder or "
                "above answered (implemented in tests/a/conftest.py; move it to a common "
                "parent conftest)*"
            ]
        )


# Five conftest.py files with every kind of implementation, a plugin and a -p plugin
LISTING_CONFTEST = """
import json

import pytest

from pytest_strategy._context import call_order, visible_from

class Plugin:
    def pytest_strategies_context(self, config):
        return None

def pytest_configure(config):
    config.pluginmanager.register(Plugin(), "registered_plugin")

@pytest.hookimpl(trylast=True)
def pytest_strategies_context(config):
    return None

def pytest_collection_modifyitems(config, items):
    found = {}
    for item in items:
        ihook = call_order(item.ihook.pytest_strategies_context.get_hookimpls())
        path = call_order(visible_from(config, item.path))
        found[item.nodeid] = [[i.plugin_name for i in ihook], [i.plugin_name for i in path]]
    (config.rootpath / "listing.json").write_text(json.dumps(found))
"""

LISTING_KINDS = {
    "tests/conftest.py": "",
    "tests/a/conftest.py": "@pytest.hookimpl(tryfirst=True)\n",
    "tests/a/deep/conftest.py": "@pytest.hookimpl(wrapper=True)\n",
    "tests/b/conftest.py": "",
}


class TestPathCaller:
    def test_lists_what_each_item_s_hook_relay_lists(self, pytester):
        files = {
            "conftest.py": LISTING_CONFTEST,
            "listing_plugin.py": "def pytest_strategies_context(config):\n    return None\n",
        }
        for conftest, decorator in LISTING_KINDS.items():
            body = "return (yield)" if "wrapper" in decorator else "return None"
            files[conftest] = (
                f"import pytest\n\n{decorator}def pytest_strategies_context(config):\n    {body}\n"
            )
        for folder in ("tests", "tests/a", "tests/a/deep", "tests/b", "tests/c"):
            name = folder.replace("/", "_")
            files[f"{folder}/test_{name}.py"] = "def test_x():\n    pass\n"
        write(pytester, files)
        pytester.syspathinsert()

        for args in ((), ("tests/b",), ("tests/a/deep", "tests/c")):
            listing = pytester.path / "listing.json"
            listing.unlink(missing_ok=True)
            result = pytester.runpytest("-p", "no:cacheprovider", "-p", "listing_plugin", *args)
            assert result.ret == 0, args
            found = json.loads(listing.read_text())
            assert found, args
            for node, (ihook, path) in found.items():
                assert ihook == path, node
            # The deepest tree sees all of them
            if not args:
                deep = found["tests/a/deep/test_tests_a_deep.py::test_x"][0]
                assert len(deep) == 6
                assert Path(deep[0]).parent.name == "deep"


class TestNestedSessions:
    def test_an_inner_session_keeps_its_own_context(self, pytester):
        write(
            pytester,
            {
                "conftest.py": """
pytest_plugins = ["pytester"]

def pytest_strategies_context(config):
    return {"name": "outer"}
""",
                "outer_strategies.py": """
from pytest_strategy import Parameter, TestArg, register

SEEN = []

@register("outer_bench")
def outer_bench(ctx):
    SEEN.append(ctx)
    return Parameter(TestArg("name", value=ctx["name"]), nsamples=1)
""",
                "test_outer.py": """
from pytest_strategy import strategy
from pytest_strategy._runtime import runtime

@strategy("outer_bench")
def test_outer(name):
    assert name == "outer"

def test_inner(pytester):
    pytester.makeconftest(
        "def pytest_strategies_context(config):\\n    return {'name': 'inner'}\\n"
    )
    pytester.makepyfile(
        inner_strategies='''
from pytest_strategy import Parameter, TestArg, register

@register("inner_bench")
def inner_bench(ctx):
    return Parameter(TestArg("name", value=ctx["name"]), nsamples=1)
''',
        test_inner='''
from pytest_strategy import strategy

@strategy("inner_bench")
def test_in(name):
    assert name == "inner"
''',
    )
    pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)

    from outer_strategies import SEEN

    assert runtime.strategy_context(__file__) is SEEN[0]
""",
            },
        )

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
