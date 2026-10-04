"""
The golden values of random streams v1 (D5). tests/golden/seed1.json holds, for
--rng-seed=1, every row of the pytester project below, by run and in collection
order: its node ID, strategy, kind, name and index, and its values in the schema 1
encoding of VectorInfo.to_dict(). The project draws with every built-in RNG type,
with predicates, under a constraint that rejects rows, in finite Series and
per_sequence_samples rows and in an RNGSequence's permutation under
--nsamples=auto, and its rows carry what a factory, a strategy file and the
context hook drew.

These are the values a recorded seed reruns, so they change only in a major
release, with a new _streams.VERSION. If this test fails after a change to the
plugin, that change gives every recorded seed other values: find what moved a key
or a draw instead of writing the file again. The values also rest on
random.Random's randint, choice, choices and sample, which CPython does not
promise to keep from one version to the next (it does for random() after
seeding). If a new CPython changes them, the rows for that version are added to
the file, keyed by the Python version (D5), and these stay.

The project keeps its tests and its strategy file in a folder, so the node IDs
and the strategy file's key contain a separator, and each run is a subprocess
under two PYTHONHASHSEED values. CI runs this on Linux and Windows, on Python 3.11
to 3.14 and on pytest 8 and 9.

To write the file in a major release, run the project with --collect-only -q
--rng-seed=1, then also with --nsamples=auto, and put the rows each run writes to
rows.json under the run's key in "runs", without their "seed" and "streams".
"""

import json
from pathlib import Path

import pytest

import pytest_strategy
from pytest_strategy import RNGType, SequenceLike

pytest_plugins = ["pytester"]

GOLDEN = Path(__file__).parent.parent / "golden" / "seed1.json"

# The project's runs, by their key in the golden file
RUNS = {"default": (), "--nsamples=auto": ("--nsamples=auto",)}

# tests/golden_strategies.py. Its import draws from its stream, keyed by its path
# relative to the rootdir (tests/golden_strategies.py, also on Windows).
STRATEGIES = """\
from enum import Enum

from pytest_strategy import (
    RNG,
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    TestArg,
    register,
)

FILE_DRAW = RNG.integer(0, 10**9)


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


@register("golden_types")
def golden_types():
    # Each built-in RNG type that draws. Under --nsamples=auto the RNGSequence is
    # enumerated instead, in the order its own stream draws.
    return Parameter(
        TestArg("integer", rng_type=RNGInteger(-1000, 1000)),
        TestArg("wide", rng_type=RNGInteger()),
        TestArg("real", rng_type=RNGFloat(-1.0, 1.0)),
        TestArg("flag", rng_type=RNGBoolean(0.3)),
        TestArg("op", rng_type=RNGChoice(["rd", "wr", "rmw"])),
        TestArg("color", rng_type=RNGEnum(Color)),
        TestArg("weighted_color", rng_type=RNGEnum(Color, weights={Color.RED: 1, Color.BLUE: 3})),
        TestArg("text", rng_type=RNGString(min_length=1, max_length=8)),
        TestArg("hex", rng_type=RNGString(length=4, charset="0123456789abcdef")),
        TestArg("wint", rng_type=RNGWeightedInteger({(0, 9): 1, (100, 109): 3})),
        TestArg("wfloat", rng_type=RNGWeightedFloat({(0.0, 1.0): 1, (10.0, 11.0): 1})),
        TestArg("pick", rng_type=RNGSequence(["x", "y", "z"])),
        nsamples=4,
    )


@register("golden_predicates")
def golden_predicates():
    # A predicate draws again from the argument's own stream
    return Parameter(
        TestArg("multiple", rng_type=RNGInteger(0, 1000, predicate=lambda x: x % 7 == 0)),
        TestArg("high", rng_type=RNGFloat(0.0, 1.0, predicate=lambda x: x > 0.9)),
        TestArg("not_red", rng_type=RNGEnum(Color, predicate=lambda c: c is not Color.RED)),
        TestArg(
            "even",
            rng_type=RNGWeightedInteger({(0, 9): 1, (100, 109): 3}, predicate=lambda x: x % 2 == 0),
        ),
        nsamples=4,
    )


@register("golden_constraint")
def golden_constraint():
    # The constraint rejects about three rows in four, and each argument's stream
    # continues from one attempt of a row to the next
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 2**32 - 1)),
        TestArg("size", rng_type=RNGChoice([1, 2, 4])),
        directed_vectors={"zeros": (0, 1)},
        vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        nsamples=4,
    )


@register("golden_series")
def golden_series():
    # The rows enumerate the Series, whose tokens key them; 3 is listed twice
    return Parameter(
        TestArg("lane", rng_type=Series([0, 3, 3])),
        TestArg("dev", rng_type=Series(["a", "b"])),
        TestArg("data", rng_type=RNGInteger(0, 255)),
        nsamples=7,
    )


@register("golden_per_sequence")
def golden_per_sequence():
    return Parameter(
        TestArg("dev", rng_type=RNGSequence(["a", "b"])),
        TestArg("data", rng_type=RNGInteger(0, 255)),
        per_sequence_samples=True,
        nsamples=2,
    )


@register("golden_auto")
def golden_auto():
    # Under --nsamples=auto the rows follow the RNGSequence's permutation
    return Parameter(
        TestArg("lane", rng_type=Series([0, 1])),
        TestArg("dev", rng_type=RNGSequence(["p", "q", "r", "s", "t"])),
        TestArg("data", rng_type=RNGInteger(0, 255)),
        nsamples=3,
    )


@register("golden_draws")
def golden_draws(rng, ctx):
    # The factory's draws (rng and RNG.*), which differ for each test, and the
    # strategy file's and the context hook's
    return Parameter(
        TestArg("factory_rng", value=rng.random()),
        TestArg("factory_helper", value=RNG.integer(0, 10**9)),
        TestArg("file", value=FILE_DRAW),
        TestArg("context", value=ctx["draw"]),
        TestArg("data", rng_type=RNGInteger(0, 255)),
        nsamples=2,
    )
"""

# tests/test_golden.py
TESTS = """\
from pytest_strategy import strategy


@strategy("golden_types")
def test_types(integer, wide, real, flag, op, color, weighted_color, text, hex, wint, wfloat, pick):
    pass


@strategy("golden_predicates")
def test_predicates(multiple, high, not_red, even):
    pass


@strategy("golden_constraint")
def test_constraint(addr, size):
    pass


@strategy("golden_series")
def test_series(lane, dev, data):
    pass


@strategy("golden_per_sequence")
def test_per_sequence(dev, data):
    pass


@strategy("golden_auto")
def test_auto(lane, dev, data):
    pass


@strategy("golden_draws")
def test_draws(factory_rng, factory_helper, file, context, data):
    pass


class TestDraws:
    @strategy("golden_draws")
    def test_draws(self, factory_rng, factory_helper, file, context, data):
        pass
"""

# The rootdir's conftest.py: the context hook, which draws from its stream, and a
# hook that writes every row to rows.json
CONFTEST = """\
import json

from pytest_strategy import RNG, VECTOR_KEY

KEYS = ("strategy", "kind", "name", "index", "values", "seed", "streams")


def pytest_strategies_context(config):
    return {"draw": RNG.integer(0, 10**9)}


def pytest_collection_modifyitems(config, items):
    rows = []
    for item in items:
        info = item.stash[VECTOR_KEY].to_dict()
        rows.append({"nodeid": item.nodeid, **{key: info[key] for key in KEYS}})
    (config.rootpath / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
"""


@pytest.fixture
def golden():
    """The golden file's contents."""
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture
def project(pytester):
    # The ini file makes the project's folder the rootdir, and no warning may occur
    pytester.makeini("[pytest]\nfilterwarnings = error\n")
    pytester.makeconftest(CONFTEST)
    tests = pytester.mkdir("tests")
    (tests / "golden_strategies.py").write_text(STRATEGIES, encoding="utf-8")
    (tests / "test_golden.py").write_text(TESTS, encoding="utf-8")
    return pytester


def collect_rows(pytester, *args):
    """Collect the project in a subprocess and return the rows it wrote."""
    rows = pytester.path / "rows.json"
    rows.unlink(missing_ok=True)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--collect-only", "-q", *args)
    assert result.ret == 0, f"{result.stdout.str()}\n{result.stderr.str()}"
    found: list[dict[str, object]] = json.loads(rows.read_text(encoding="utf-8"))
    return found


@pytest.mark.parametrize("hash_seed", ["0", "1"])
@pytest.mark.parametrize("run", list(RUNS))
def test_rows_match_the_golden_values(project, golden, monkeypatch, run, hash_seed):
    monkeypatch.setenv("PYTHONHASHSEED", hash_seed)

    rows = collect_rows(project, f"--rng-seed={golden['seed']}", *RUNS[run])

    # Every row comes from the golden's seed and streams version
    versions = {(row.pop("seed"), row.pop("streams")) for row in rows}
    assert versions == {(golden["seed"], golden["streams"])}
    expected = golden["runs"][run]
    assert [row["nodeid"] for row in rows] == [row["nodeid"] for row in expected]
    for row, expected_row in zip(rows, expected, strict=True):
        assert row == expected_row, f"{run} run: {row['nodeid']}"


def test_the_project_uses_every_built_in_rng_type():
    """
    A built-in RNG type added later is added to the project, with a strategy and a
    test of its own: that adds rows to the golden file and changes none.
    """
    rng_types = {
        name
        for name in pytest_strategy.__all__
        if isinstance(getattr(pytest_strategy, name), type)
        and issubclass(getattr(pytest_strategy, name), RNGType)
        and getattr(pytest_strategy, name) not in (RNGType, SequenceLike)
    }

    assert {"RNGInteger", "RNGWeightedFloat", "RNGSequence", "Series"} <= rng_types
    assert {name for name in rng_types if f"{name}(" in STRATEGIES} == rng_types
