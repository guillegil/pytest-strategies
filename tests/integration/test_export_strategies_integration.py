"""
End-to-end tests for export_strategies() in a pytest session, run through
pytester: each factory is called with the inputs it declares and the session's
options, and the context hook runs only for a factory that declares ctx.

Distinct strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json

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
