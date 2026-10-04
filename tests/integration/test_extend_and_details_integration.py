"""
End-to-end tests for 4.1: a strategy built on another with Parameter.extend(),
and the per-row pytest-strategies section, which a failed row gets only with
--strategy-details.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
import pytest
from pytest_strategy import Parameter, RNGChoice, RNGInteger, TestArg, register


@register("xb_burst")
def xb_burst():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 0xFFFF)),
        TestArg("length", rng_type=RNGInteger(1, 256)),
        directed_vectors={
            "zeros": {"addr": 0, "length": 1},
            "known": pytest.param({"addr": 4, "length": 4}, marks=pytest.mark.xfail(strict=True)),
        },
        vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        nsamples=5,
    )


@register("xb_short")
def xb_short():
    return xb_burst().extend(TestArg("length", rng_type=RNGInteger(1, 8)), nsamples=3)


@register("xb_prio")
def xb_prio():
    return xb_burst().extend(
        TestArg("prio", rng_type=RNGChoice([0, 1, 2])),
        defaults={"prio": 0},
        directed_vectors={"high": {"addr": 8, "length": 4, "prio": 2}},
    )
"""

TESTS = """
from pytest_strategy import strategy


@strategy("xb_short")
def test_short(addr, length):
    assert addr % 4 == 0 and 1 <= length <= 8
    assert (addr, length) != (4, 4)


@strategy("xb_prio")
def test_prio(addr, length, prio):
    assert prio in (0, 1, 2)
    assert (addr, length) != (4, 4)
"""


class TestExtend:
    def test_extended_strategies_run_with_the_base_s_vectors_and_constraints(self, pytester):
        pytester.makepyfile(strategies=STRATEGIES, test_xb=TESTS)

        result = pytester.runpytest("-p", "no:cacheprovider", "-v", "--rng-seed=3")

        result.assert_outcomes(passed=11, xfailed=2)
        result.stdout.fnmatch_lines(
            [
                "*test_short?directed-zeros? PASSED*",
                "*test_short?directed-known? XFAIL*",
                "*test_short?rand-2? PASSED*",
                "*test_prio?directed-high? PASSED*",
                "*test_prio?rand-4? PASSED*",
            ]
        )
        assert "test_short[rand-3]" not in result.stdout.str()

    def test_an_added_argument_without_a_value_fails_collection(self, pytester):
        pytester.makepyfile(
            strategies=STRATEGIES + """

@register("xb_bad")
def xb_bad():
    return xb_burst().extend(TestArg("prio", rng_type=RNGChoice([0, 1])))
""",
            test_xb_bad=(
                "from pytest_strategy import strategy\n\n"
                '@strategy("xb_bad")\ndef test_bad(addr, length, prio):\n    pass\n'
            ),
        )

        result = pytester.runpytest("-p", "no:cacheprovider")

        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines(
            ["*directed vector 'zeros' has no value for the added argument 'prio'*"]
        )


FAILING = """
from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy


@register("xd_many")
def xd_many():
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=12)


@strategy("xd_many")
def test_fails(x):
    assert x < 0
"""


def _sections(text):
    return text.count("- pytest-strategies -")


class TestDetails:
    def test_by_default_no_section_and_a_capped_list_of_rerun_commands(self, pytester):
        pytester.makepyfile(test_xd=FAILING)

        result = pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=5")

        result.assert_outcomes(failed=12)
        out = result.stdout.str()
        assert _sections(out) == 0
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: reproduce with --rng-seed=5",
                "pytest-strategies: failed rows:",
                "  pytest 'test_xd.py::test_fails[rand-0]' --rng-seed=5  # xd_many random 0",
                "  ... and 2 more",
                "  (--strategy-details shows each row's values below its traceback)",
            ]
        )

    @pytest.mark.parametrize("addopts", [False, True], ids=["command_line", "addopts"])
    def test_details_give_each_failed_row_its_section(self, pytester, addopts):
        pytester.makepyfile(test_xd=FAILING)
        args = ("--rng-seed=5",)
        if addopts:
            pytester.makeini("[pytest]\naddopts = --strategy-details\n")
        else:
            args += ("--strategy-details",)

        result = pytester.runpytest("-p", "no:cacheprovider", *args)

        result.assert_outcomes(failed=12)
        out = result.stdout.str()
        assert _sections(out) == 12
        assert "--strategy-details shows" not in out
        result.stdout.fnmatch_lines(
            ["rerun     pytest 'test_xd.py::test_fails[rand-0]' --rng-seed=5"]
        )

    def test_the_details_option_is_not_part_of_the_rerun_command(self, pytester):
        pytester.makepyfile(test_xd=FAILING)

        result = pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=5", "--strategy-details")

        assert "--rng-seed=5 --strategy-details" not in result.stdout.str()

    def test_qq_prints_no_list_and_no_hint(self, pytester):
        pytester.makepyfile(test_xd=FAILING)

        result = pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=5", "-qq")

        assert "--strategy-details shows" not in result.stdout.str()
