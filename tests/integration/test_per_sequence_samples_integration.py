"""
End-to-end tests for Parameter(per_sequence_samples=True) through real pytest sessions.
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
from pytest_strategy import Parameter, RNGInteger, RNGSequence, Strategy, TestArg

@Strategy.register("per_device")
def per_device(nsamples):
    return Parameter(
        TestArg("device", rng_type=RNGSequence(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        directed_vectors={"narrow": ("devA", 1)},
        per_sequence_samples=True,
    )
"""

TEST_MODULE = """
from pytest_strategy import Strategy

@Strategy.strategy("per_device")
def test_width(device, width):
    assert device in ("devA", "devB")
    assert 1 <= width <= 64
"""


@pytest.fixture
def project(pytester):
    pytester.makepyfile(persample_strategies=STRATEGIES, test_persample=TEST_MODULE)
    return pytester


def test_default_count_runs_ten_per_device(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1")

    # 1 directed vector + 10 rows for each of the 2 devices
    result.assert_outcomes(passed=21)


def test_cli_nsamples_is_per_device(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "--nsamples=3")

    result.assert_outcomes(passed=7)


def test_directed_only_mode_is_unaffected(project):
    result = project.runpytest(
        "-p", "no:cacheprovider", "--rng-seed=1", "--vector-mode=directed_only"
    )

    result.assert_outcomes(passed=1)


def test_auto_stays_one_row_per_device(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "--nsamples=auto")

    # Directed vector + one exhaustive row per device
    result.assert_outcomes(passed=3)


def test_each_device_gets_the_requested_rows(project):
    result = project.runpytest(
        "-p", "no:cacheprovider", "--rng-seed=1", "--nsamples=4", "--collect-only", "-q"
    )

    ids = [line for line in result.outlines if "::test_width[" in line]
    assert sum("devA" in i for i in ids) == 1 + 4
    assert sum("devB" in i for i in ids) == 4


def test_short_combination_warning_names_strategy_and_test(pytester):
    pytester.makepyfile(
        starved_strategies="""
from pytest_strategy import Parameter, RNGInteger, Series, Strategy, TestArg

@Strategy.register("starved")
def starved(nsamples):
    return Parameter(
        TestArg("device", rng_type=Series(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(1, 64)),
        vector_constraints=[lambda v: v[0] == "devA" or v[1] > 100],
        max_retries=3,
        per_sequence_samples=True,
    )
""",
        test_starved="""
from pytest_strategy import Strategy

@Strategy.strategy("starved")
def test_width(device, width):
    pass
""",
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "--nsamples=2")

    result.assert_outcomes(passed=2, warnings=1)
    result.stdout.fnmatch_lines(
        [
            "*/test_starved.py:3: PytestStrategiesWarning: Strategy 'starved' (test_width): "
            "Sequence combination (device='devB') produced 0 of 2 rows*"
        ]
    )
