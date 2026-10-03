"""
End-to-end tests for the context fingerprint (D9): the line after the collection,
the context on the line that says how to reproduce a failed run, the -v summary's
Contexts block and ``VectorInfo.context``. Each fingerprint is computed when the
hook returns, so a factory or a test that changes the object later changes none of
them.

The main project has a rootdir conftest.py answering ROOT and, in the variants with
two scopes, tests/tb_a/conftest.py answering A. The expected fingerprints are
computed here from the same objects.
"""

import json
from pathlib import Path
from textwrap import dedent

import pytest

from pytest_strategy._fingerprint import fingerprint

pytest_plugins = ["pytester"]

SEED = 21

ROOT = {"name": "root", "limit": 5}
A = {"name": "A", "limit": 3}

# The rootdir conftest.py: its implementation, and the VectorInfo.context of every
# collected item's strategies, written to infos.json
ROOT_CONFTEST = """
import json

from pytest_strategy import VECTORS_KEY

def pytest_strategies_context(config):
    return {root!r}

def pytest_collection_finish(session):
    infos = {{
        item.nodeid: [info.context for info in item.stash.get(VECTORS_KEY, ())]
        for item in session.items
    }}
    (session.config.rootpath / "infos.json").write_text(json.dumps(infos))
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("bounded")
def bounded(ctx):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, ctx["limit"])), nsamples=2)

@register("plain")
def plain(nsamples):
    return Parameter(TestArg("y", rng_type=RNGInteger(0, 9)), nsamples=2)
"""


def module(strategy, condition="True", name="test_x"):
    """A test module whose test uses ``strategy`` and asserts ``condition``."""
    argument = "y" if strategy == "plain" else "x"
    return f"""
from pytest_strategy import strategy

@strategy({strategy!r})
def {name}({argument}):
    assert {condition}
"""


def fp(value, pytester):
    """The fingerprint of ``value`` in the pytester project."""
    return fingerprint(value, pytester.path)[0]


def write(pytester, files, root=ROOT):
    """Write the main project's rootdir conftest.py and strategies, and ``files``."""
    files = {"conftest.py": ROOT_CONFTEST.format(root=root), "strategies.py": STRATEGIES, **files}
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))


def two_scopes(pytester, root_condition="True", a_condition="True"):
    """The main project with tests/tb_a answering A: one test above it, two in it."""
    write(
        pytester,
        {
            "tests/test_root.py": module("bounded", root_condition, "test_root"),
            "tests/tb_a/conftest.py": f"def pytest_strategies_context(config):\n"
            f"    return {A!r}\n",
            "tests/tb_a/test_a.py": module("bounded", a_condition, "test_a")
            + module("bounded", "True", "test_a_too"),
        },
    )


def infos(pytester):
    """The VectorInfo.context of each item's strategies, by node ID."""
    return json.loads((pytester.path / "infos.json").read_text())


def context_lines(result):
    """The lines that start like the context line."""
    return [line for line in result.stdout.lines if line.startswith("pytest-strategies: context")]


def repro_line(result):
    """The line that says how to reproduce the run."""
    (line,) = [line for line in result.stdout.lines if "reproduce with" in line]
    return line


class TestContextLine:
    @pytest.mark.parametrize(
        "args", [[], ["-q"], ["--collect-only"], ["--collect-only", "-q"]], ids=" ".join
    )
    def test_after_the_collection(self, pytester, args):
        write(pytester, {"tests/test_x.py": module("bounded")})

        result = pytester.runpytest("-p", "no:cacheprovider", *args)

        line = f"pytest-strategies: context {fp(ROOT, pytester)}"
        assert context_lines(result) == [line]
        if "-q" not in args:
            result.stdout.fnmatch_lines(["collected 2 items", line])
        assert result.ret == 0

    def test_absent_when_no_factory_declares_ctx(self, pytester):
        write(pytester, {"tests/test_x.py": module("plain")})

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert context_lines(result) == []

    def test_absent_when_the_hook_returns_none(self, pytester):
        factory = STRATEGIES + """
@register("optional")
def optional(ctx):
    assert ctx is None
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=2)
"""
        write(pytester, {"tests/test_x.py": module("optional")}, root=None)
        (pytester.path / "strategies.py").write_text(dedent(factory))

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert context_lines(result) == []

    def test_several_scopes_with_their_labels(self, pytester):
        two_scopes(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider", "-q")

        result.assert_outcomes(passed=6)
        assert context_lines(result) == [
            f"pytest-strategies: contexts conftest.py {fp(ROOT, pytester)}, "
            f"tests/tb_a/conftest.py {fp(A, pytester)}"
        ]

    def test_only_the_scopes_the_collection_computed(self, pytester):
        two_scopes(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider", "tests/tb_a")

        assert context_lines(result) == [f"pytest-strategies: context {fp(A, pytester)}"]

    def test_partial_types(self, pytester):
        conftest = (
            "class Handle:\n"
            "    pass\n\n"
            "def pytest_strategies_context(config):\n"
            "    return {'limit': 5, 'handle': Handle()}\n"
        )
        write(pytester, {"tests/conftest.py": conftest, "tests/test_x.py": module("bounded")})

        result = pytester.runpytest("-p", "no:cacheprovider")

        (line,) = context_lines(result)
        assert line.endswith(" (partial: Handle)")

    def test_an_implementation_that_raised_is_left_out(self, pytester):
        two_scopes(pytester)
        (pytester.path / "tests/tb_a/conftest.py").write_text(
            "def pytest_strategies_context(config):\n    raise RuntimeError('no bench')\n"
        )

        result = pytester.runpytest("-p", "no:cacheprovider", "--continue-on-collection-errors")

        result.assert_outcomes(passed=2, errors=1)
        assert context_lines(result) == [f"pytest-strategies: context {fp(ROOT, pytester)}"]


class TestReproduceLine:
    def test_ends_with_the_context(self, pytester):
        write(pytester, {"tests/test_x.py": module("bounded", "x < 0")})

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(failed=2)
        assert repro_line(result) == (
            f"pytest-strategies: reproduce with --rng-seed={SEED} (context {fp(ROOT, pytester)})"
        )

    def test_the_3_0_line_when_the_failed_tests_received_no_context(self, pytester):
        write(
            pytester,
            {
                "tests/test_x.py": module("bounded"),
                "tests/test_y.py": module("plain", "y < 0", "test_y"),
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=2, failed=2)
        assert context_lines(result) != []
        assert repro_line(result) == f"pytest-strategies: reproduce with --rng-seed={SEED}"

    def test_the_contexts_the_failed_tests_received(self, pytester):
        two_scopes(pytester, root_condition="True", a_condition="x < 0")

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=4, failed=2)
        assert repro_line(result) == (
            f"pytest-strategies: reproduce with --rng-seed={SEED} (context {fp(A, pytester)})"
        )

    def test_several_with_their_labels(self, pytester):
        two_scopes(pytester, root_condition="x < 0", a_condition="x < 0")

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}", "-q")

        result.assert_outcomes(passed=2, failed=4)
        assert repro_line(result) == (
            f"pytest-strategies: reproduce with --rng-seed={SEED} (contexts conftest.py "
            f"{fp(ROOT, pytester)}, tests/tb_a/conftest.py {fp(A, pytester)})"
        )

    def test_a_setup_error_counts(self, pytester):
        write(
            pytester,
            {
                "tests/test_x.py": module("bounded")
                + "\nimport pytest\n\n@pytest.fixture(autouse=True)\n"
                "def broken():\n    raise RuntimeError('setup')\n"
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(errors=2)
        assert repro_line(result).endswith(f" (context {fp(ROOT, pytester)})")

    def test_a_teardown_error_alone_adds_no_context(self, pytester):
        # As the failed rows of the repro section (D18): setup and call failures only
        write(
            pytester,
            {
                "tests/test_x.py": module("bounded")
                + "\nimport pytest\n\n@pytest.fixture(autouse=True)\n"
                "def broken():\n    yield\n    raise RuntimeError('teardown')\n"
            },
        )

        result = pytester.runpytest("-p", "no:cacheprovider", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=2, errors=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert repro_line(result) == f"pytest-strategies: reproduce with --rng-seed={SEED}"


class TestVerboseSummary:
    def test_the_contexts_block(self, pytester):
        two_scopes(pytester)

        result = pytester.runpytest("-p", "no:cacheprovider", "-v")

        result.stdout.fnmatch_lines(
            [
                "*Strategy Summary*",
                "Registered strategies: *",
                "  bounded (strategies.py): 3 test(s), 0 directed, 6 random rows; "
                "nsamples=2 from Parameter(nsamples=)",
                "Contexts: 2",
                f"  conftest.py: {fp(ROOT, pytester)}, 1 test(s)",
                f"  tests/tb_a/conftest.py: {fp(A, pytester)}, 2 test(s)",
            ]
        )

    def test_no_block_without_a_context(self, pytester):
        write(pytester, {"tests/test_x.py": module("plain")})

        result = pytester.runpytest("-p", "no:cacheprovider", "-v")

        result.stdout.fnmatch_lines(["*Strategy Summary*"])
        result.stdout.no_fnmatch_line("Contexts*")


class TestVectorInfo:
    def test_only_the_strategies_whose_factories_received_ctx(self, pytester):
        write(
            pytester,
            {"tests/test_x.py": """
from pytest_strategy import strategy

@strategy("bounded")
@strategy("plain")
def test_both(x, y):
    pass

@strategy("plain")
def test_plain(y):
    pass
"""},
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=6)
        found = infos(pytester)
        expected = fp(ROOT, pytester)
        both = [contexts for node, contexts in found.items() if "test_both" in node]
        plain = [contexts for node, contexts in found.items() if "test_plain" in node]
        # In node ID order: plain (the decorator nearer the function) first
        assert both == [[None, expected]] * 4
        assert plain == [[None]] * 2


class TestChangedContext:
    def test_a_changed_context_changes_no_fingerprint(self, pytester):
        root = {"name": "root", "limit": 5, "runs": 0}
        counting = STRATEGIES + """
@register("counting")
def counting(ctx):
    ctx["runs"] += 1
    return Parameter(TestArg("x", value=ctx["runs"]), nsamples=1)
"""
        changing = """
from pytest_strategy import strategy

def test_changes(strategies_ctx):
    strategies_ctx["name"] = "changed"
    strategies_ctx["runs"] = 99

@strategy("counting")
def test_second(x):
    assert x < 0
"""
        write(
            pytester,
            {
                "tests/test_1.py": module("counting", "x == 1", "test_first"),
                "tests/test_2.py": changing,
            },
            root=root,
        )
        (pytester.path / "strategies.py").write_text(dedent(counting))

        result = pytester.runpytest("-p", "no:cacheprovider", "-v", f"--rng-seed={SEED}")

        result.assert_outcomes(passed=2, failed=1)
        expected = fp(root, pytester)
        # The second factory saw runs == 1, and its rows still carry the first fingerprint
        assert infos(pytester) == {
            "tests/test_1.py::test_first[rand-0]": [expected],
            "tests/test_2.py::test_changes": [],
            "tests/test_2.py::test_second[rand-0]": [expected],
        }
        assert context_lines(result) == [f"pytest-strategies: context {expected}"]
        assert repro_line(result).endswith(f" (context {expected})")
        result.stdout.fnmatch_lines([f"  conftest.py: {expected}, 2 test(s)"])


class TestAcrossRuns:
    CONFTEST = """
def pytest_strategies_context(config):
    return {
        "bench": config.rootpath / "tb" / "bench.yaml",
        "lanes": {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"},
    }
"""

    def test_one_line_in_two_checkouts_and_every_hash_seed(self, pytester, monkeypatch):
        for checkout in ("ci", "home/me/work"):
            folder = pytester.path / checkout
            folder.mkdir(parents=True)
            (folder / "pytest.ini").write_text("[pytest]\n")
            (folder / "conftest.py").write_text(self.CONFTEST)
            (folder / "strategies.py").write_text(dedent(STRATEGIES))
            (folder / "test_x.py").write_text(dedent(module("bounded", "x < 0")))
            # Lists built from the set would differ; the bound is the set's size
            text = (
                (folder / "strategies.py").read_text().replace('ctx["limit"]', 'len(ctx["lanes"])')
            )
            (folder / "strategies.py").write_text(text)

        lines = set()
        for checkout, hash_seed in (("ci", "1"), ("ci", "2"), ("ci", "3"), ("home/me/work", "1")):
            monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
            result = pytester.runpytest_subprocess(
                "-p", "no:cacheprovider", f"--rng-seed={SEED}", checkout
            )
            result.assert_outcomes(failed=2)
            lines.update(context_lines(result))
            lines.add(repro_line(result))

        expected = fingerprint(
            {
                "bench": Path("tb/bench.yaml"),
                "lanes": {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"},
            }
        )[0]
        assert lines == {
            f"pytest-strategies: context {expected}",
            f"pytest-strategies: reproduce with --rng-seed={SEED} (context {expected})",
        }
