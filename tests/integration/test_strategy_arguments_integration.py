"""
End-to-end tests for ``@strategy(name, **arguments)``: the extra keywords are
passed to the factory by keyword, after the inputs the plugin passes, so one
factory serves several tests. The names are checked against the factory's
signature when the test is collected, and the random streams do not depend on
the arguments.
"""

import json
from textwrap import dedent

import pytest

pytest_plugins = ["pytester"]


CONFTEST = """
import json

import pytest


@pytest.hookimpl(optionalhook=True)
def pytest_strategies_context(config):
    return {1: "ESM", 13: "BPAM", 14: "BPAM"}


def pytest_collection_modifyitems(config, items):
    from pytest_strategy import VECTOR_KEY

    rows = {
        item.nodeid: [item.stash[VECTOR_KEY].strategy, repr(item.callspec.params)]
        for item in items
        if VECTOR_KEY in item.stash
    }
    (config.rootpath / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register


def get_channels(ctx, device):
    return [k for k, v in ctx.items() if v.lower() == device.lower()]


@register("channels")
def channels(ctx, device):
    return Parameter(
        TestArg("channel", rng_type=Series(get_channels(ctx, device),
                                           skip_if_empty=f"no {device} channel")),
        per_sequence_samples=True,
    )


@register("rw")
def rw(ctx, device="esm", *, width=8):
    return channels(ctx, device).extend(
        TestArg("data", rng_type=RNGInteger(0, 2**width - 1)), nsamples=2
    )
"""


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(port_strategies=STRATEGIES)
    return pytester


def rows(pytester):
    return json.loads((pytester.path / "rows.json").read_text(encoding="utf-8"))


class TestPassing:
    def test_each_test_gets_the_rows_of_its_arguments(self, project):
        project.makepyfile(test_rw=dedent("""
                from pytest_strategy import strategy

                @strategy("rw", device="esm")
                def test_esm(channel, data):
                    assert channel == 1

                @strategy("rw", device="bpam", width=16)
                def test_bpam(channel, data):
                    assert channel in (13, 14)

                @strategy("rw")
                def test_default(channel, data):
                    assert channel == 1 and data < 256
                """))
        result = project.runpytest("-v", "--rng-seed=1")
        result.assert_outcomes(passed=8)
        result.stdout.fnmatch_lines(
            [
                "*test_esm?channel=1-rand-0? PASSED*",
                "*test_bpam?channel=13-rand-1? PASSED*",
                "*test_bpam?channel=14-rand-1? PASSED*",
            ]
        )

    def test_a_parameter_without_a_default_is_filled_by_the_test(self, project):
        project.makepyfile(test_ch=dedent("""
                from pytest_strategy import strategy

                @strategy("channels", device="BPAM")
                def test_bpam(channel):
                    assert channel in (13, 14)
                """))
        project.runpytest("--nsamples=1").assert_outcomes(passed=2)

    def test_a_factory_passed_by_object_takes_them_too(self, project):
        project.makepyfile(test_obj=dedent("""
                from pytest_strategy import strategy
                from port_strategies import rw

                @strategy(rw, device="bpam")
                def test_bpam(channel, data):
                    assert channel in (13, 14)
                """))
        project.runpytest().assert_outcomes(passed=4)
        assert {strategy for strategy, _ in rows(project).values()} == {"rw"}

    def test_the_skip_of_an_empty_sequence_follows_the_argument(self, project):
        project.makepyfile(test_none=dedent("""
                from pytest_strategy import strategy

                @strategy("rw", device="dma")
                def test_dma(channel, data):
                    pass
                """))
        result = project.runpytest("-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*no dma channel*"])

    def test_a_factory_with_kwargs_receives_any_name(self, pytester):
        pytester.makepyfile(test_kw=dedent("""
                from pytest_strategy import Parameter, Series, TestArg, register, strategy

                @register("kw")
                def kw(**options):
                    return Parameter(TestArg("x", rng_type=Series([options["start"]])))

                @strategy("kw", start=7)
                def test_x(x):
                    assert x == 7
                """))
        pytester.runpytest("--nsamples=1").assert_outcomes(passed=1)

    def test_the_marker_takes_them_as_keywords(self, project):
        project.makepyfile(test_mark=dedent("""
                import pytest

                @pytest.mark.strategy("rw", device="bpam")
                def test_bpam(channel, data):
                    assert channel in (13, 14)
                """))
        project.runpytest().assert_outcomes(passed=4)


class TestStreams:
    def test_the_values_do_not_depend_on_the_arguments(self, project):
        # Same test, same strategy: only the arguments differ, and the rows the
        # two runs share (channel 1) have the same values
        test = """
            from pytest_strategy import strategy

            @strategy("rw"{arguments})
            def test_w(channel, data):
                pass
            """
        project.makepyfile(test_w=dedent(test.format(arguments="")))
        project.runpytest("--rng-seed=5").assert_outcomes(passed=2)
        without = rows(project)
        project.makepyfile(test_w=dedent(test.format(arguments=', device="ESM", width=8')))
        project.runpytest("--rng-seed=5").assert_outcomes(passed=2)
        assert rows(project) == without


class TestErrors:
    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (
                'devce="esm"',
                "*@strategy('rw', ...) cannot pass 'devce' to port_strategies.py:*: it has no "
                "such parameter (it takes: device, width). Did you mean 'device'?",
            ),
            (
                "ctx={}",
                "*cannot pass 'ctx' to *: it is an input the plugin passes itself.*",
            ),
            (
                "config=1",
                "*cannot pass 'config' to *: the plugin reserves that name;*",
            ),
        ],
    )
    def test_a_name_the_factory_cannot_take_fails_collection(self, project, arguments, message):
        project.makepyfile(test_bad=dedent(f"""
                from pytest_strategy import strategy

                @strategy("rw", {arguments})
                def test_w(channel, data):
                    pass
                """))
        result = project.runpytest()
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines([message])

    def test_a_positional_only_parameter_cannot_be_passed(self, pytester):
        pytester.makepyfile(test_po=dedent("""
                from pytest_strategy import Parameter, Series, TestArg, register, strategy

                @register("po")
                def po(device="esm", /):
                    return Parameter(TestArg("x", rng_type=Series([1])))

                @strategy("po", device="esm")
                def test_x(x):
                    pass
                """))
        result = pytester.runpytest()
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            ["*cannot pass 'device' to *: that parameter is positional-only*"]
        )

    def test_a_missing_argument_names_the_keyword(self, project):
        project.makepyfile(test_missing=dedent("""
                from pytest_strategy import strategy

                @strategy("channels")
                def test_c(channel):
                    pass
                """))
        result = project.runpytest()
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "*has a parameter 'device', which the plugin does not provide.*"
                "or pass it from the test: @strategy(..., device=...).*"
            ]
        )
