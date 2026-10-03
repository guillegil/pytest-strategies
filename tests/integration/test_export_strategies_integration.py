"""
End-to-end tests for export_strategies() in a pytest session, run through
pytester: each factory is called with the inputs it declares and the session's
options, and the context hook runs only for a factory that declares ctx, which
gets the context of the folder it is registered in, or none when pytest did not
load a conftest.py of that folder.

Distinct strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json
from collections import Counter
from textwrap import dedent

import pytest

pytest_plugins = ["pytester"]

# Counts the hook's calls in hook_calls.txt
CONFTEST = """
    from pathlib import Path

    def pytest_strategies_context(config):
        with open(Path(config.rootpath, "hook_calls.txt"), "a") as calls:
            calls.write("called\\n")
        return "bench"
    """

# Each factory appends what it received to received.jsonl. "ex_{case}_rows" is
# also used by a collected test, so it is called at collection and at export.
STRATEGIES = """
    import json
    from pathlib import Path

    from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, register

    def record(factory, **received):
        with open(Path(__file__).parent / "received.jsonl", "a") as out:
            out.write(json.dumps({{"factory": factory, **received}}) + "\\n")

    def describe(options):
        return {{
            "id": id(options),
            "strategy": options.strategy,
            "nsamples": options.nsamples,
            "nsamples_source": options.nsamples_source,
        }}

    def param():
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

    @register("ex_{case}_rows")
    def rows(rng, options):
        record("rows", rng_is_generator=rng is RNG.generator(), **describe(options))
        return param()

    @register("ex_{case}_count")
    def count(nsamples):
        record("count", nsamples=nsamples)
        return param()

    @register("ex_{case}_none")
    def none():
        record("none")
        return param()
    """

# A factory that declares ctx, added to show that the hook is reached at all
CTX_STRATEGIES = """
    import json
    from pathlib import Path

    from pytest_strategy import Parameter, RNGInteger, TestArg, register

    @register("ex_{case}_ctx")
    def with_ctx(ctx):
        with open(Path(__file__).parent / "received.jsonl", "a") as out:
            out.write(json.dumps({{"factory": "ctx", "ctx": ctx}}) + "\\n")
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
    """

# One collected test of the strategy, then the export, written to exported.json
TESTS = """
    import json
    from pathlib import Path

    from pytest_strategy import export_strategies, strategy

    @strategy("ex_{case}_rows")
    def test_rows(x):
        pass

    def test_export():
        Path("exported.json").write_text(export_strategies())
    """


def _received(pytester):
    lines = (pytester.path / "received.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _hook_calls(pytester):
    calls = pytester.path / "hook_calls.txt"
    return len(calls.read_text().splitlines()) if calls.exists() else 0


@pytest.mark.parametrize(
    ("case", "args", "nsamples", "source"),
    [
        ("default", [], 10, "default"),
        ("count", ["--nsamples=7"], 7, "--nsamples"),
        ("auto", ["--nsamples=auto"], "auto", "--nsamples"),
    ],
)
def test_factories_get_the_sessions_options(pytester, case, args, nsamples, source):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        **{
            f"ex_{case}_strategies": STRATEGIES.format(case=case),
            f"test_ex_{case}": TESTS.format(case=case),
        }
    )

    result = pytester.runpytest("-p", "no:cacheprovider", *args)

    # test_rows has the session's count of rows ("auto" falls back to 10
    # without a Series), and test_export passes
    result.assert_outcomes(passed=(10 if nsamples == "auto" else nsamples) + 1)
    exported = json.loads((pytester.path / "exported.json").read_text())
    for name in ("rows", "count", "none"):
        assert "error" not in exported[f"ex_{case}_{name}"], exported[f"ex_{case}_{name}"]
    collected, *at_export = _received(pytester)
    rows = {
        "factory": "rows",
        "rng_is_generator": True,
        # The same instance at collection and at export
        "id": collected["id"],
        "strategy": f"ex_{case}_rows",
        "nsamples": nsamples,
        "nsamples_source": source,
    }
    assert collected == rows
    assert sorted(at_export, key=lambda r: r["factory"]) == [
        {"factory": "count", "nsamples": nsamples},
        {"factory": "none"},
        rows,
    ]
    # No factory declares ctx, so the hook never ran
    assert _hook_calls(pytester) == 0


def test_a_rejected_factory_is_an_error_entry_relative_to_the_rootdir(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.mkdir("sub")
    pytester.makepyfile(
        **{
            "sub/ex_bad_strategies": """
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("ex_bad_bad")
            def bad(n, ctx):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
            """,
            "test_ex_bad": """
            from pathlib import Path

            from pytest_strategy import export_strategies

            def test_export():
                Path("exported.json").write_text(export_strategies())
            """,
        }
    )

    result = pytester.runpytest("-p", "no:cacheprovider")

    # The export reports the factory instead of failing
    result.assert_outcomes(passed=1)
    exported = json.loads((pytester.path / "exported.json").read_text())
    assert set(exported["ex_bad_bad"]) == {"error"}
    error = exported["ex_bad_bad"]["error"]
    where = "(sub/ex_bad_strategies.py:3:bad)"
    assert f"Strategy factory 'ex_bad_bad' {where} has a parameter 'n'" in error
    assert "Did you mean 'nsamples'?" in error
    # Rejected before the hook runs, although the factory declares ctx
    assert _hook_calls(pytester) == 0


def test_only_a_ctx_factory_runs_the_hook(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        ex_ctx_strategies=STRATEGIES.format(case="ctx"),
        ex_ctx_more_strategies=CTX_STRATEGIES.format(case="ctx"),
        test_ex_ctx=TESTS.format(case="ctx"),
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "--nsamples=2")

    result.assert_outcomes(passed=3)
    exported = json.loads((pytester.path / "exported.json").read_text())
    assert "error" not in exported["ex_ctx_ctx"], exported["ex_ctx_ctx"]
    assert {"factory": "ctx", "ctx": "bench"} in _received(pytester)
    assert _hook_calls(pytester) == 1


# The context of each registration's folder (D7): a project in root/, with a
# rootdir conftest.py answering "root", tests/a/conftest.py answering "A", a
# factory in tests/a and one in tests/b, and one in a library outside the rootdir
# (shared/, on the pythonpath), which tests/a's test module imports. Each hook call
# is written to calls.txt and each factory call to factories.txt.

FOLDER_HOOK = """
    from pathlib import Path

    def pytest_strategies_context(config):
        with open(Path(config.rootpath, "calls.txt"), "a") as calls:
            calls.write("{name}\\n")
        return {value!r}
    """

FOLDER_STRATEGIES = """
    from pathlib import Path

    from pytest_strategy import Parameter, TestArg, register

    @register("ex_fold_{name}")
    def factory(ctx):
        with open(Path(__file__).parents[{up}] / "factories.txt", "a") as out:
            out.write("ex_fold_{name}\\n")
        return Parameter(TestArg("name", value=ctx["name"] if ctx else None), nsamples=1)
    """

# Exports each entry's context name (the "name" argument's value), or what replaces
# its Parameter
FOLDER_EXPORT = """
    import json
    from pathlib import Path

    import ext_fold_lib  # noqa: F401  (a factory outside the rootdir)

    from pytest_strategy import export_strategies

    def test_export(request):
        exported = json.loads(export_strategies())
        found = {
            name: entry["arguments"][0]["static_value"] if "arguments" in entry else entry
            for name, entry in exported.items()
            if name.startswith("ex_fold_")
        }
        Path(request.config.rootpath, "export.json").write_text(json.dumps(found))
    """


def folder_project(pytester, monkeypatch, extra=None):
    """Write the project, and go to its rootdir."""
    files = {
        "root/pytest.ini": "[pytest]\npythonpath = ../shared\n",
        "root/conftest.py": FOLDER_HOOK.format(name="root", value={"name": "root"}),
        "root/tests/a/conftest.py": FOLDER_HOOK.format(name="A", value={"name": "A"}),
        "root/tests/a/a_strategies.py": FOLDER_STRATEGIES.format(name="a", up=2),
        "root/tests/a/test_export.py": FOLDER_EXPORT,
        "root/tests/b/b_strategies.py": FOLDER_STRATEGIES.format(name="b", up=2),
        "root/tests/b/test_b.py": "def test_b():\n    pass\n",
        "shared/ext_fold_lib.py": FOLDER_STRATEGIES.format(name="outside", up=1).replace(
            'Path(__file__).parents[1] / "factories.txt"', '"factories.txt"'
        ),
        **(extra or {}),
    }
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))
    root = pytester.path / "root"
    monkeypatch.chdir(root)
    return root


def folder_run(root, pytester, *args):
    """
    Run pytest in the project, and return the exported contexts, the hook calls and
    the factory calls (with how often).
    """
    for name in ("calls.txt", "factories.txt", "export.json"):
        (root / name).unlink(missing_ok=True)
    result = pytester.runpytest("-p", "no:cacheprovider", *args)
    result.assert_outcomes(passed=1 + (not args))

    def counted(name):
        path = root / name
        return Counter(path.read_text().split()) if path.exists() else Counter()

    return (
        json.loads((root / "export.json").read_text()),
        counted("calls.txt"),
        counted("factories.txt"),
    )


class TestEachRegistrationsFolderContext:
    """export_strategies() gives a factory the context of the folder it is registered in."""

    def test_a_factory_gets_its_folder_s_context_and_one_outside_the_rootdir_the_rootdir_s(
        self, pytester, monkeypatch
    ):
        root = folder_project(pytester, monkeypatch)

        for args in ((), ("tests/a",)):
            exported, calls, factories = folder_run(root, pytester, *args)

            # tests/b has no conftest.py of its own: the rootdir's answers there,
            # also in a run that does not collect it
            assert exported == {"ex_fold_a": "A", "ex_fold_b": "root", "ex_fold_outside": "root"}
            assert calls == {"root": 1, "A": 1}, args
            assert factories == {"ex_fold_a": 1, "ex_fold_b": 1, "ex_fold_outside": 1}

    def test_a_folder_whose_conftest_was_not_loaded_is_unavailable(self, pytester, monkeypatch):
        """
        After pytest tests/a, tests/b's conftest.py, which implements the hook, was
        never loaded: the context of tests/b cannot be known, so its ctx factory is
        not called. Its factory without ctx is exported as usual.
        """
        plain = """
            from pytest_strategy import Parameter, TestArg, register

            @register("ex_fold_b_plain")
            def plain():
                return Parameter(TestArg("name", value="plain"), nsamples=1)
            """
        root = folder_project(
            pytester,
            monkeypatch,
            extra={
                "root/tests/b/conftest.py": FOLDER_HOOK.format(name="B", value={"name": "B"}),
                "root/tests/b/plain_strategies.py": plain,
            },
        )

        exported, calls, factories = folder_run(root, pytester, "tests/a")

        assert exported == {
            "ex_fold_a": "A",
            "ex_fold_b": {"unavailable": "tests/b/conftest.py was not loaded in this session"},
            "ex_fold_b_plain": "plain",
            "ex_fold_outside": "root",
        }
        assert calls == {"root": 1, "A": 1}
        assert factories == {"ex_fold_a": 1, "ex_fold_outside": 1}

        # A run that collects tests/b loads it
        exported, calls, factories = folder_run(root, pytester)

        assert exported["ex_fold_b"] == "B"
        assert calls == {"root": 1, "A": 1, "B": 1}
        assert factories["ex_fold_b"] == 1

    def test_ctx_none_in_the_folder_gets_the_migration_hint(self, pytester, monkeypatch):
        root = folder_project(pytester, monkeypatch)
        (root / "conftest.py").write_text("")
        (root / "tests/b/b_strategies.py").write_text(
            dedent(FOLDER_STRATEGIES.format(name="b", up=2)).replace(
                'ctx["name"] if ctx else None', 'ctx["name"]'
            )
        )

        exported, calls, factories = folder_run(root, pytester)

        error = exported["ex_fold_b"]["error"]
        assert "TypeError: 'NoneType' object is not subscriptable" in error
        assert (
            "ctx is None for tests/b: no pytest_strategies_context implementation in this "
            "folder or above answered (implemented in tests/a/conftest.py; move it to a common "
            "parent conftest)"
        ) in error
        assert calls == {"A": 1}
