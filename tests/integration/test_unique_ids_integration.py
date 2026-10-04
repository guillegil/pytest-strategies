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

from pytest_strategy import Parameter, RNGChoice, RNGInteger, RNGSequence, Series, Strategy, TestArg


@dataclass
class Point:
    x: int
    y: int


# A value whose ID in the values format is the suffixed ID of 7
class Taken:
    def __repr__(self):
        return "7_0"


# A value whose ID in the values format ends in a non-ASCII character
class Cafe:
    def __repr__(self):
        return "café"


@Strategy.register("repeated")
def repeated(nsamples):
    # A sequence-only Parameter repeats its single value
    return Parameter(TestArg("x", rng_type=RNGSequence([1])), nsamples=3)


@Strategy.register("channels")
def channels(nsamples):
    # A Series-only Parameter with more rows than values repeats them
    return Parameter(TestArg("ch", rng_type=Series([0, 1])), nsamples=3)


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


@Strategy.register("twin_points")
def twin_points(nsamples):
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 1)),
        TestArg("y", rng_type=RNGInteger(2, 2)),
        directed_vectors={"first": pytest.param({"x": 0, "y": 2}), "second": {"y": 2, "x": 0}},
        nsamples=0,
    )


@Strategy.register("sevens")
def sevens(nsamples):
    # In the values format, "n=7_0" is already taken, and the marks are kept
    return Parameter(
        TestArg("n", rng_type=RNGInteger(0, 9)),
        directed_vectors={
            "seven": (7,),
            "seven_skipped": pytest.param(7, marks=pytest.mark.skip(reason="marks are kept")),
            "seven_again": (7,),
            "taken": (Taken(),),
        },
        nsamples=0,
    )


@Strategy.register("accents")
def accents(nsamples):
    return Parameter(
        TestArg("w", rng_type=RNGChoice(["a", "b"])),
        directed_vectors={"café": (Cafe(),), "cafe": (Cafe(),)},
        nsamples=0,
    )
"""

TEST_MODULE = """
from dup_strategies import Cafe, Point, Taken

from pytest_strategy import Strategy


@Strategy.strategy("repeated")
def test_repeated(x):
    assert x == 1


@Strategy.strategy("channels")
def test_channels(ch):
    assert ch in (0, 1)


@Strategy.strategy("per_device")
def test_per_device(width, device):
    assert width == 8


@Strategy.strategy("points")
def test_points(p: Point):
    assert p.y == 2


@Strategy.strategy("twin_points")
def test_twin_points(p: Point):
    assert p.y == 2


@Strategy.strategy("sevens")
def test_sevens(n):
    assert n == 7 or isinstance(n, Taken)


@Strategy.strategy("accents")
def test_accents(w):
    assert isinstance(w, Cafe)
"""

# The names format: every row of a strategy has its own ID, whatever its values
NAMES_IDS = [
    "test_repeated[rand-0]",
    "test_repeated[rand-1]",
    "test_repeated[rand-2]",
    "test_channels[ch=0-rand-0]",
    "test_channels[ch=1-rand-0]",
    "test_channels[ch=0-rand-1]",
    "test_per_device[device=devA-rand-0]",
    "test_per_device[device=devA-rand-1]",
    "test_per_device[device=devB-rand-0]",
    "test_per_device[device=devB-rand-1]",
    "test_points[rand-0]",
    "test_points[rand-1]",
    "test_twin_points[directed-first]",
    "test_twin_points[directed-second]",
    "test_sevens[directed-seven]",
    "test_sevens[directed-seven_skipped]",
    "test_sevens[directed-seven_again]",
    "test_sevens[directed-taken]",
    "test_accents[directed-caf\\xe9]",
    "test_accents[directed-cafe]",
]

# pytest.mark.parametrize given the same rows and the IDs the resolver generates
# for them in the values format, before they are made unique
PLAIN_TWINS = """
import pytest


@pytest.mark.parametrize("x", [1, 1, 1], ids=["x=1"] * 3)
def test_repeated(x):
    pass


@pytest.mark.parametrize("n", [7, 7, 7, 70], ids=["n=7", "n=7", "n=7", "n=7_0"])
def test_sevens(n):
    pass


@pytest.mark.parametrize("w", ["a", "b"], ids=["w=café", "w=café"])
def test_accents(w):
    pass
"""

# The values format: the IDs pytest gave these rows when it suffixed the duplicates
# itself
VALUES_IDS = [
    "test_repeated[x=1_0]",
    "test_repeated[x=1_1]",
    "test_repeated[x=1_2]",
    "test_channels[ch=0_0]",
    "test_channels[ch=1]",
    "test_channels[ch=0_1]",
    "test_per_device[width=8,device='devA'0]",
    "test_per_device[width=8,device='devA'1]",
    "test_per_device[width=8,device='devB'0]",
    "test_per_device[width=8,device='devB'1]",
    "test_points[x=0,y=2_0]",
    "test_points[x=0,y=2_1]",
    "test_twin_points[x=0,y=2_0]",
    "test_twin_points[x=0,y=2_1]",
    "test_sevens[n=7_1]",
    "test_sevens[n=7_2]",
    "test_sevens[n=7_3]",
    "test_sevens[n=7_0]",
    "test_accents[w=caf\\xe9_0]",
    "test_accents[w=caf\\xe9_1]",
]

VALUES = ("-o", "strategies_ids=values")


@pytest.fixture
def project(pytester):
    pytester.makepyfile(dup_strategies=STRATEGIES, test_dup=TEST_MODULE)
    return pytester


def _collected_ids(result, module):
    """Return the IDs, without the module, of the tests collected from ``module``."""
    prefix = f"{module}.py::"
    return [line[len(prefix) :] for line in result.outlines if line.startswith(prefix)]


@pytest.mark.parametrize("ids", [(), VALUES], ids=["names", "values"])
def test_strict_parametrization_ids_accept_repeated_rows(project, pytestconfig, ids):
    try:
        pytestconfig.getini("strict_parametrization_ids")
    except ValueError:
        pytest.skip("this pytest has no strict_parametrization_ids option")

    result = project.runpytest(
        "-p", "no:cacheprovider", "-o", "strict_parametrization_ids=true", *ids
    )

    result.assert_outcomes(passed=19, skipped=1)


def test_names_ids_need_no_suffix(project):
    result = project.runpytest("-p", "no:cacheprovider", "--collect-only", "-q")

    assert _collected_ids(result, "test_dup") == NAMES_IDS


def test_ids_are_unchanged_without_strict_ids(project):
    result = project.runpytest("-p", "no:cacheprovider", "--collect-only", "-q", *VALUES)

    assert _collected_ids(result, "test_dup") == VALUES_IDS


def test_ids_match_pytest_own_suffixes(project):
    """pytest suffixes the same duplicate IDs identically in a plain parametrize."""
    project.makepyfile(test_plain=PLAIN_TWINS)

    result = project.runpytest("-p", "no:cacheprovider", "--collect-only", "-q", *VALUES)

    plain = _collected_ids(result, "test_plain")
    twins = {"test_repeated", "test_sevens", "test_accents"}
    ids = [i for i in _collected_ids(result, "test_dup") if i.split("[")[0] in twins]
    assert len(plain) == 9
    assert ids == plain


@pytest.mark.parametrize("ids", [(), VALUES], ids=["names", "values"])
def test_a_non_ascii_vector_name_is_selected_by_its_raw_name(project, ids):
    """pytest escapes the node ID; --vector-name takes the name as it is written."""
    result = project.runpytest(
        "-p", "no:cacheprovider", "--collect-only", "-q", "--vector-name=café", *ids
    )

    # The other strategies have no such vector: their tests get an empty parameter set
    selected = [i for i in _collected_ids(result, "test_dup") if i.startswith("test_accents[")]
    assert selected == [
        "test_accents[directed-caf\\xe9]" if not ids else "test_accents[w=caf\\xe9]"
    ]
