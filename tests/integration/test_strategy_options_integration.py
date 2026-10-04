"""
End-to-end tests for StrategyOptions, run through pytester: the options each
strategy gets from the command line, cached per session under its resolved name.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json

import pytest

pytest_plugins = ["pytester"]

# Writes every strategy's cached options to options.json once collection ends,
# after checking that the builder hands back the cached instance
CONFTEST = """
    import dataclasses
    import json

    from pytest_strategy._runtime import runtime

    def pytest_collection_finish(session):
        state = runtime.current
        assert state.options is not None
        dumped = {}
        for name, options in state.strategy_options.items():
            assert runtime.strategy_options(name, session.config) is options
            assert state.options.for_strategy(name) == options
            fields = dataclasses.asdict(options)
            fields["constraints_off"] = sorted(fields["constraints_off"])
            fields["filtered"] = options.filtered
            dumped[name] = fields
        (session.config.rootpath / "options.json").write_text(json.dumps(dumped))
    """

# Two strategies, one used by two tests (once through its factory, which
# resolves to the registered name)
TESTS = """
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    @register("so_{case}_burst")
    def make_burst(nsamples):
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)), directed_vectors={{"zeros": (0,)}}
        )

    @register("so_{case}_other")
    def make_other(nsamples):
        return Parameter(
            TestArg("y", rng_type=RNGInteger(0, 9)), directed_vectors={{"zeros": (0,)}}
        )

    @strategy(make_burst)
    def test_by_factory(x):
        pass

    @strategy("so_{case}_burst")
    def test_by_name(x):
        pass

    @strategy("so_{case}_other")
    def test_other(y):
        pass
    """


@pytest.mark.parametrize(
    ("case", "args", "expected"),
    [
        (
            "none",
            [],
            {
                "nsamples": 10,
                "nsamples_source": "default",
                "mode": "all",
                "vector_name": None,
                "vector_index": None,
                "filtered": False,
            },
        ),
        (
            "directed",
            ["--vector-mode=directed_only", "--vector-name=zeros", "--nsamples=3"],
            {
                "nsamples": 3,
                "nsamples_source": "--nsamples",
                "mode": "directed_only",
                "vector_name": "zeros",
                "vector_index": None,
                "filtered": True,
            },
        ),
        (
            "ten",
            ["--nsamples=10"],
            {
                "nsamples": 10,
                "nsamples_source": "--nsamples",
                "mode": "all",
                "vector_name": None,
                "vector_index": None,
                "filtered": False,
            },
        ),
        (
            "auto",
            ["--nsamples=auto"],
            {
                "nsamples": "auto",
                "nsamples_source": "--nsamples",
                "mode": "all",
                "vector_name": None,
                "vector_index": None,
                "filtered": False,
            },
        ),
        (
            "index",
            ["--vector-index=0", "--vector-mode=random_only"],
            {
                "nsamples": 10,
                "nsamples_source": "default",
                "mode": "random_only",
                "vector_name": None,
                "vector_index": 0,
                "filtered": True,
            },
        ),
    ],
)
def test_each_strategy_gets_the_runs_options_under_its_resolved_name(
    pytester, case, args, expected
):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(**{f"test_so_{case}": TESTS.format(case=case)})

    result = pytester.runpytest(*args)

    assert result.ret == pytest.ExitCode.OK
    options = json.loads((pytester.path / "options.json").read_text())
    names = [f"so_{case}_burst", f"so_{case}_other"]
    assert options == {
        name: {"strategy": name, **expected, "constraints_off": []} for name in names
    }


def test_vector_mode_choices_are_the_vector_mode_values(pytester):
    result = pytester.runpytest("--vector-mode=bogus")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        [
            "*--vector-mode: invalid choice: 'bogus' "
            "(choose from *all*, *random_only*, *directed_only*, *mixed*, *test*)"
        ]
    )
