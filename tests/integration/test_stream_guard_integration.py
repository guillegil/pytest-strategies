"""
End-to-end tests for the guard on draws outside the rows' streams (D5), run through
pytester: a test whose rows draw from the plugin's generator outside the arguments'
own streams (an RNG type that draws from the factory's rng, a constraint that calls
RNG.*) gets one PytestStrategiesWarning, at the test, naming the strategy and the
test, and under filterwarnings = error its collection fails. Strategies that draw
only on their streams get none.

The runs are subprocesses, which keep this suite's filterwarnings = error out of
them. Distinct module and strategy names are used per project on purpose (see
test_session_isolation_integration.py for rationale).
"""

import pytest

pytest_plugins = ["pytester"]

MESSAGE = "something drew from the plugin's generator while the rows were generated"

GUARD_STRATEGIES = """
from pytest_strategy import RNG, Parameter, RNGInteger, RNGType, TestArg, register


class Kept(RNGType[int]):
    # Draws from the generator it was given instead of RNG.generator()
    def __init__(self, generator):
        self.generator = generator

    def generate(self):
        return self.generator.randint(0, 10**9)

    @property
    def python_type(self):
        return int


@register("guard_kept")
def guard_kept(rng):
    return Parameter(
        TestArg("a", rng_type=Kept(rng)), TestArg("b", rng_type=RNGInteger(0, 10**9))
    )


@register("guard_coin")
def guard_coin():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9)),
        vector_constraints={"coin": lambda v: RNG.integer(0, 1) >= 0},
    )
"""

GUARD_TESTS = """
from pytest_strategy import strategy


@strategy("guard_kept")
def test_kept(a, b):
    pass


@strategy("guard_coin")
def test_coin(a):
    pass


class TestCoin:
    @strategy("guard_coin")
    def test_method(self, a):
        pass
"""


def run(pytester, *args):
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--rng-seed=1", *args)


def test_each_test_gets_one_warning(pytester):
    pytester.makepyfile(guard_strategies=GUARD_STRATEGIES, test_guard=GUARD_TESTS)

    result = run(pytester)

    result.assert_outcomes(passed=30, warnings=3)
    lines = [line for line in result.outlines if "PytestStrategiesWarning: " in line]
    assert len(lines) == 3
    # Each at its test's first line (the decorator), in the test file
    for strategy, test, line in [
        ("guard_kept", "test_kept", 4),
        ("guard_coin", "test_coin", 9),
        ("guard_coin", "TestCoin.test_method", 15),
    ]:
        found = [
            text
            for text in lines
            if f"test_guard.py:{line}: PytestStrategiesWarning: "
            f"Strategy '{strategy}' ({test}): {MESSAGE}" in text
        ]
        assert len(found) == 1, lines


def test_under_filterwarnings_error_the_collection_fails(pytester):
    pytester.makepyfile(guard_strategies=GUARD_STRATEGIES, test_guard=GUARD_TESTS)
    pytester.makeini("[pytest]\nfilterwarnings = error\n")

    result = run(pytester)

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(
        [
            "*In test_kept: Error generating samples for strategy 'guard_kept': "
            f"Strategy 'guard_kept' (test_kept): {MESSAGE}*"
        ]
    )


def test_the_values_repeat_for_the_same_run(pytester):
    # A warning, not an error: for one configuration the values are reproducible
    pytester.makepyfile(guard_strategies=GUARD_STRATEGIES, test_guard=GUARD_TESTS)

    first = run(pytester, "--collect-only", "-q", "-o", "strategies_ids=values")
    again = run(pytester, "--collect-only", "-q", "-o", "strategies_ids=values")

    ids = [line for line in first.outlines if "::" in line]
    assert len(ids) == 30
    assert [line for line in again.outlines if "::" in line] == ids


CLEAN_STRATEGIES = """
from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    RNGSequence,
    RNGType,
    Series,
    TestArg,
    register,
)

FILE_DRAW = RNG.integer(0, 9)


class AtDrawTime(RNGType[int]):
    # Calls RNG.generator() when it draws
    def generate(self):
        return RNG.generator().randint(0, 10**9)

    @property
    def python_type(self):
        return int


@register("guard_clean")
def guard_clean(rng):
    rng.random()
    RNG.integer(0, 9)
    return Parameter(
        TestArg("ch", rng_type=Series([0, 1])),
        TestArg("dev", rng_type=RNGSequence(["x", "y"])),
        TestArg("a", rng_type=AtDrawTime()),
        TestArg("b", rng_type=RNGInteger(0, 10**9, lambda x: x % 2 == 0)),
        TestArg("c", rng_type=RNGInteger(0, 9), validator=lambda x: RNG.boolean() or True),
        TestArg("d", value=rng.random()),
        vector_constraints={"ordered": lambda v: v.a != v.b},
        nsamples=3,
    )


@register("guard_clean_per_sequence")
def guard_clean_per_sequence():
    return Parameter(
        TestArg("dev", rng_type=RNGSequence(["x", "y"])),
        TestArg("a", rng_type=AtDrawTime()),
        nsamples=2,
        per_sequence_samples=True,
    )
"""

CLEAN_TESTS = """
import pytest

from pytest_strategy import RNG, strategy

MODULE_DRAW = RNG.integer(0, 9)


@pytest.fixture
def drawn():
    return RNG.integer(0, 9)


@strategy("guard_clean")
def test_clean(ch, dev, a, b, c, d, drawn):
    RNG.integer(0, 9)


@strategy("guard_clean_per_sequence")
def test_per_sequence(dev, a):
    pass
"""


@pytest.mark.parametrize(
    ("args", "passed"),
    [((), 3 + 2 * 2), (("--nsamples=auto",), 4 + 2), (("--nsamples=25",), 25 + 2 * 25)],
    ids=["finite", "auto", "n25"],
)
def test_draws_on_the_streams_get_no_warning(pytester, args, passed):
    pytester.makepyfile(guard_clean_strategies=CLEAN_STRATEGIES, test_guard_clean=CLEAN_TESTS)
    pytester.makeini("[pytest]\nfilterwarnings = error\n")

    result = run(pytester, *args)

    result.assert_outcomes(passed=passed, warnings=0)
