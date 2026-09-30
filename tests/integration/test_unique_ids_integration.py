"""
End-to-end tests for the test IDs of repeated rows, with and without pytest's
strict_parametrization_ids.
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
from dataclasses import dataclass

import pytest

from pytest_strategy import Parameter, RNGInteger, RNGSequence, Strategy, TestArg


@dataclass
class Point:
    x: int
    y: int


@Strategy.register("repeated")
def repeated(nsamples):
    # A sequence-only Parameter repeats its single value
    return Parameter(TestArg("x", rng_type=RNGSequence([1])), nsamples=3)


@Strategy.register("per_device")
def per_device(nsamples):
    return Parameter(
        TestArg("width", rng_type=RNGInteger(8, 8)),
        TestArg("device", rng_type=RNGSequence(["devA", "devB"])),
        per_sequence_samples=True,
        nsamples=2,
    )


@Strategy.register("points")
def points(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGSequence([0])),
        TestArg("y", rng_type=RNGInteger(2, 2)),
        nsamples=2,
    )


@Strategy.register("legacy")
def legacy(nsamples):
    # Explicit IDs override the generated ones, and "n=7_0" is already taken
    return "n", [
        7,
        pytest.param(5, id="n=7", marks=pytest.mark.skip(reason="marks are kept")),
        7,
        pytest.param(9, id="n=7_0"),
    ]


@Strategy.register("legacy_points")
def legacy_points(nsamples):
    return ("x", "y"), [pytest.param(0, 2, id="p"), pytest.param(1, 2, id="p")]


@Strategy.register("accents")
def accents(nsamples):
    return "w", [pytest.param("a", id="café"), pytest.param("b", id="café")]
"""

TEST_MODULE = """
from dup_strategies import Point

from pytest_strategy import Strategy


@Strategy.strategy("repeated")
def test_repeated(x):
    assert x == 1


@Strategy.strategy("per_device")
def test_per_device(width, device):
    assert width == 8


@Strategy.strategy("points")
def test_points(p: Point):
    assert p.y == 2


@Strategy.strategy("legacy_points")
def test_legacy_points(p: Point):
    assert p.y == 2


@Strategy.strategy("legacy")
def test_legacy(n):
    assert n in (7, 9)


@Strategy.strategy("accents")
def test_accents(w):
    assert w in ("a", "b")
"""

# pytest.mark.parametrize given the same rows and the IDs the resolver generates
# for them, before they are made unique
PLAIN_TWINS = """
import pytest


@pytest.mark.parametrize("x", [1, 1, 1], ids=["x=1"] * 3)
def test_repeated(x):
    pass


@pytest.mark.parametrize(
    "n",
    [7, pytest.param(5, id="n=7"), 7, pytest.param(9, id="n=7_0")],
    ids=["n=7", "n=5", "n=7", "n=9"],
)
def test_legacy(n):
    pass


@pytest.mark.parametrize(
    "w", [pytest.param("a", id="café"), pytest.param("b", id="café")]
)
def test_accents(w):
    pass
"""

# The IDs pytest gave these rows when it suffixed the duplicates itself
EXPECTED_IDS = [
    "test_repeated[x=1_0]",
    "test_repeated[x=1_1]",
    "test_repeated[x=1_2]",
    "test_per_device[width=8,device='devA'0]",
    "test_per_device[width=8,device='devA'1]",
    "test_per_device[width=8,device='devB'0]",
    "test_per_device[width=8,device='devB'1]",
    "test_points[x=0,y=2_0]",
    "test_points[x=0,y=2_1]",
    "test_legacy_points[p0]",
    "test_legacy_points[p1]",
    "test_legacy[n=7_1]",
    "test_legacy[n=7_2]",
    "test_legacy[n=7_3]",
    "test_legacy[n=7_0]",
    "test_accents[caf\\xe9_0]",
    "test_accents[caf\\xe9_1]",
]


@pytest.fixture
def project(pytester):
    pytester.makepyfile(dup_strategies=STRATEGIES, test_dup=TEST_MODULE)
    return pytester


def _collected_ids(result, module):
    """Return the IDs, without the module, of the tests collected from ``module``."""
    prefix = f"{module}.py::"
    return [line[len(prefix) :] for line in result.outlines if line.startswith(prefix)]


def test_strict_parametrization_ids_accept_repeated_rows(project, pytestconfig):
    try:
        pytestconfig.getini("strict_parametrization_ids")
    except ValueError:
        pytest.skip("this pytest has no strict_parametrization_ids option")

    result = project.runpytest("-p", "no:cacheprovider", "-o", "strict_parametrization_ids=true")

    result.assert_outcomes(passed=16, skipped=1)


def test_ids_are_unchanged_without_strict_ids(project):
    result = project.runpytest("-p", "no:cacheprovider", "--collect-only", "-q")

    assert _collected_ids(result, "test_dup") == EXPECTED_IDS


def test_ids_match_pytest_own_suffixes(project):
    """pytest suffixes the same duplicate IDs identically in a plain parametrize."""
    project.makepyfile(test_plain=PLAIN_TWINS)

    result = project.runpytest("-p", "no:cacheprovider", "--collect-only", "-q")

    plain = _collected_ids(result, "test_plain")
    twins = {"test_repeated", "test_legacy", "test_accents"}
    ids = [i for i in _collected_ids(result, "test_dup") if i.split("[")[0] in twins]
    assert len(plain) == 9
    assert ids == plain
