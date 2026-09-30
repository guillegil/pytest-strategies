"""
End-to-end tests for Series/RNGSequence(skip_if_empty=...) through real pytest sessions.
"""

import random

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what the in-process runs change globally: the registry and the RNG seed."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, RNGSequence, Series, Strategy, TestArg

def esm(channels, seq_type=RNGSequence):
    return Parameter(
        TestArg(
            "channel",
            rng_type=seq_type(channels, skip_if_empty="no Esm peripheral in this config"),
        ),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        directed_vectors={"zero": (3, 0)},
        test_vectors={"smoke": (3, 1)},
    )

@Strategy.register("with_esm")
def with_esm(nsamples):
    return esm([3, 7])

@Strategy.register("no_esm")
def no_esm(nsamples):
    return esm([])

@Strategy.register("no_esm_series")
def no_esm_series(nsamples):
    return esm([], Series)

@Strategy.register("one_arg_empty")
def one_arg_empty(nsamples):
    return Parameter(TestArg("channel", rng_type=Series([], skip_if_empty="nothing to do")))
"""

TEST_MODULE = """
from dataclasses import dataclass

from pytest_strategy import Strategy


@dataclass
class Access:
    channel: int
    wdata: int

    def __post_init__(self):
        assert self.channel is not None, "must not be built from the skipped row"


@Strategy.strategy("with_esm")
def test_with(channel, wdata):
    assert channel in (3, 7)


@Strategy.strategy("no_esm")
def test_without(channel, wdata):
    raise AssertionError("must be skipped")


@Strategy.strategy("no_esm_series")
def test_without_series(channel, wdata):
    raise AssertionError("must be skipped")


@Strategy.strategy("no_esm")
def test_without_dataclass(access: Access):
    raise AssertionError("must be skipped")


@Strategy.strategy("one_arg_empty")
def test_one_arg(channel):
    raise AssertionError("must be skipped")


class TestInClass:
    @Strategy.strategy("no_esm")
    def test_method(self, channel, wdata):
        raise AssertionError("must be skipped")
"""

SKIPPED = 5


@pytest.fixture
def project(pytester):
    pytester.makepyfile(esm_strategies=STRATEGIES, test_esm=TEST_MODULE)
    return pytester


def run(project, *args):
    return project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "-rs", *args)


def test_empty_sequences_skip_with_their_reason(project):
    result = run(project)

    result.assert_outcomes(passed=11, skipped=SKIPPED)
    result.stdout.fnmatch_lines(["SKIPPED [[]*[]] test_esm.py:*: no Esm peripheral in this config"])
    result.stdout.fnmatch_lines(["SKIPPED [[]*[]] test_esm.py:*: nothing to do"])


def test_skipped_tests_have_one_readable_id(project):
    result = run(project, "--collect-only", "-q")

    result.stdout.fnmatch_lines(
        [
            "test_esm.py::test_without[[]skipped[]]",
            "test_esm.py::test_without_dataclass[[]skipped[]]",
            "test_esm.py::test_one_arg[[]skipped[]]",
            "test_esm.py::TestInClass::test_method[[]skipped[]]",
        ]
    )


@pytest.mark.parametrize(
    "args, passed",
    [
        (["--nsamples=auto"], 3),
        (["--nsamples=0"], 1),
        (["--vector-mode=directed_only"], 1),
        (["--vector-mode=random_only"], 10),
        (["--vector-mode=test"], 1),
    ],
)
def test_every_mode_skips(project, args, passed):
    result = run(project, *args)

    result.assert_outcomes(passed=passed, skipped=SKIPPED)


@pytest.mark.parametrize("args", [["--vector-name=zero"], ["--vector-index=0"]])
def test_matching_vector_filter_still_skips_with_the_reason(project, args):
    result = run(project, *args)

    # one_arg_empty has no directed vectors, so the filter leaves it an empty set
    result.assert_outcomes(passed=1, skipped=SKIPPED)
    result.stdout.fnmatch_lines(["*got empty parameter set*"])
    assert result.stdout.str().count("no Esm peripheral in this config") == 4


def test_vector_filter_matching_nothing_is_still_a_usage_error(project):
    result = run(project, "--vector-name=missing")

    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_skip_works_under_xdist(project):
    pytest.importorskip("xdist")

    result = project.runpytest("-p", "no:cacheprovider", "-n", "2")

    result.assert_outcomes(passed=11, skipped=SKIPPED)


def test_list_strategies_shows_the_skipped_strategy(project):
    result = project.runpytest("-p", "no:cacheprovider", "--list-strategies")

    assert result.ret == 0
    result.stdout.fnmatch_lines(["*no_esm*"])


def test_dataclass_mismatch_is_reported_even_when_skipped(pytester):
    pytester.makepyfile(
        esm_strategies=STRATEGIES,
        test_mismatch="""
from dataclasses import dataclass

from pytest_strategy import Strategy


@dataclass
class Access:
    channel: int
    data: int


@Strategy.strategy("no_esm")
def test_rw(access: Access):
    pass
""",
    )

    result = pytester.runpytest("-p", "no:cacheprovider")

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*Error converting samples to dataclass for strategy 'no_esm'*"])
