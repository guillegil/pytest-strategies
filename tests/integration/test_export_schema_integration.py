"""
The strategy export's schema 1 (D19), end to end. tests/golden/export-schema1.json
holds what export_strategies() returns for the pytester project below, run with
--rng-seed=1 and one constraint turned off: every built-in RNG type, a custom
RNGType, a value argument, Enum, NaN, infinite, bytes, set and tuple values,
named constraints, directed and test vectors, a raising factory, and one name
registered in two folders, whose contexts differ.

The golden's generator.version is "*": the version is compared with the
package's instead, so a release's version bump does not change the file. Any other
difference is a change of the export's format, which schema 1 allows only by adding
keys or enum values (see _export.py): update the file only for such an addition.

To write the file again, run the project with --rng-seed=1
--strategy-constraint-off=ex_types:small, and put the export.json it writes, with
the version replaced by "*", in the file.
"""

import json
from pathlib import Path

import pytest

import pytest_strategy

pytest_plugins = ["pytester"]

GOLDEN = Path(__file__).parent.parent / "golden" / "export-schema1.json"

ARGS = ("--rng-seed=1", "--strategy-constraint-off=ex_types:small")

# The rootdir's conftest.py, and tests/a's: two contexts
ROOT_CONFTEST = """\
def pytest_strategies_context(config):
    return {"bench": "root", "lanes": 4}
"""

A_CONFTEST = """\
def pytest_strategies_context(config):
    return {"bench": "a", "lanes": 8}
"""

# tests/a/bus_strategies.py
A_STRATEGIES = """\
import math
from enum import Enum

import pytest

from pytest_strategy import (
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    TestArg,
    register,
)


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Walk(RNGType[int]):
    # A custom type: written by its public attributes, a set of strings in a
    # stable order; it has no python_type
    def __init__(self, width):
        self.width = width
        self.pattern = (1, 2, 4)
        self.tags = {"walk", "ones", "bus"}
        self._position = 0

    def generate(self):
        return 1 << (self.width - 1)


@register("ex_types")
def ex_types(ctx):
    return Parameter(
        TestArg("integer", rng_type=RNGInteger(0, 255, predicate=lambda v: v % 2 == 0)),
        TestArg("real", rng_type=RNGFloat(-1.5, 2.5), description="a float"),
        TestArg("flag", rng_type=RNGBoolean(0.25)),
        TestArg("data", rng_type=RNGChoice(["rd", b"\\x00\\xff", (1, 2)])),
        TestArg("color", rng_type=RNGEnum(Color, weights={Color.RED: 3, Color.BLUE: 1})),
        TestArg("hue", rng_type=RNGEnum(Color, predicate=lambda c: c is not Color.RED)),
        TestArg("text", rng_type=RNGString(min_length=2, max_length=4, charset="ab")),
        TestArg("wint", rng_type=RNGWeightedInteger({(0, 9): 3, (100, 109): 1})),
        TestArg("wfloat", rng_type=RNGWeightedFloat({(0.0, 1.0): 0.5, (10.0, 11.0): 0.5})),
        TestArg("pick", rng_type=RNGSequence(["x", "y"], skip_if_empty="no picks")),
        TestArg("lane", rng_type=Series([0, 3, 3])),
        TestArg("walk", rng_type=Walk(8)),
        TestArg("bench", value=ctx["bench"]),
        TestArg("mode", value="fast", description="a value", validator=lambda v: v == "fast"),
        directed_vectors={
            "zeros": {
                "integer": 0, "real": math.nan, "flag": False, "data": b"\\x00",
                "color": Color.RED, "hue": Color.GREEN, "text": "ab", "wint": 0,
                "wfloat": 0.0, "pick": "x", "lane": 0, "walk": 1, "bench": "a",
                "mode": "fast",
            },
            "marked": pytest.param(
                {
                    "integer": 2, "real": 1.0, "flag": True, "data": (1, 2),
                    "color": Color.BLUE, "hue": Color.BLUE, "text": "ba", "wint": 9,
                    "wfloat": 1.0, "pick": "y", "lane": 3, "walk": 2, "bench": "a",
                    "mode": "fast",
                },
                marks=pytest.mark.skip(reason="marked"),
            ),
        },
        test_vectors={
            "edges": (
                254, math.inf, True, {"b", "a"}, Color.GREEN, Color.BLUE, "bb", 109,
                -math.inf, "y", 3, frozenset({3, 1}), "a", "fast",
            ),
        },
        vector_constraints={
            "even": lambda v: v.integer % 2 == 0,
            "small": lambda v: v.integer < 100,
        },
        max_retries=50,
        nsamples=4,
        max_exhaustive=1000,
    )


@register("ex_bus")
def ex_bus():
    return Parameter(TestArg("addr", rng_type=RNGInteger(0, 15)))


@register("ex_broken")
def ex_broken(nsamples):
    raise RuntimeError("boom")
"""

# tests/b/bus_strategies.py: "ex_bus" again, in a folder without a conftest.py, so
# its factory gets the rootdir's context
B_STRATEGIES = """\
from pytest_strategy import Parameter, RNGInteger, TestArg, register


@register("ex_bus")
def ex_bus(ctx):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 15)),
        TestArg("bench", value=ctx["bench"]),
        per_sequence_samples=True,
        always_include_directed=False,
        nsamples="auto",
    )
"""

# tests/a/test_bus.py: the export, written to export.json in the rootdir
A_TESTS = """\
from pytest_strategy import export_strategies, strategy


@strategy("ex_types")
def test_types(integer, real, flag, data, color, hue, text, wint, wfloat, pick, lane, walk, bench, mode):
    assert integer < 100 or integer % 2 == 0


def test_export(request):
    (request.config.rootpath / "export.json").write_text(export_strategies(), encoding="utf-8")
"""

B_TESTS = """\
from pytest_strategy import strategy


@strategy("ex_bus")
def test_bus(addr, bench):
    assert bench == "root"
"""


def _no_constant(name):
    raise ValueError(f"{name} in the export")


def loads(text):
    """Read JSON strictly: NaN and infinity are errors."""
    return json.loads(text, parse_constant=_no_constant)


@pytest.fixture
def project(pytester):
    pytester.makeini("[pytest]\nfilterwarnings = error\n")
    pytester.makeconftest(ROOT_CONFTEST)
    files = {
        "tests/a/conftest.py": A_CONFTEST,
        "tests/a/bus_strategies.py": A_STRATEGIES,
        "tests/a/test_bus.py": A_TESTS,
        "tests/b/bus_strategies.py": B_STRATEGIES,
        "tests/b/test_bus_b.py": B_TESTS,
    }
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return pytester


def export(pytester, *args):
    """Run the project in a subprocess and return the text of its export."""
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", *args)
    assert result.ret == 0, f"{result.stdout.str()}\n{result.stderr.str()}"
    return (pytester.path / "export.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("hash_seed", ["0", "1"])
def test_the_export_matches_the_golden_document(project, monkeypatch, hash_seed):
    monkeypatch.setenv("PYTHONHASHSEED", hash_seed)

    exported = loads(export(project, *ARGS))

    assert exported["generator"]["version"] == pytest_strategy.__version__
    exported["generator"]["version"] = "*"
    assert exported == loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_export_is_schema_1_with_every_registration(project):
    exported = loads(export(project, *ARGS))

    assert (exported["schema"], exported["kind"]) == (1, "strategies")
    assert exported["generator"]["name"] == "pytest-strategies"
    assert (exported["seed"], exported["nsamples"]) == (1, 10)
    entries = {(e["name"], e["origin"]["folder"]): e for e in exported["strategies"]}
    # Sorted by name and folder, the name registered twice included
    assert list(entries) == [
        ("ex_broken", "tests/a"),
        ("ex_bus", "tests/a"),
        ("ex_bus", "tests/b"),
        ("ex_types", "tests/a"),
    ]
    assert entries["ex_broken", "tests/a"]["error"] == {"type": "RuntimeError", "message": "boom"}
    types = entries["ex_types", "tests/a"]["parameter"]
    assert types["schema"] == 1
    arguments = {argument["name"]: argument for argument in types["arguments"]}
    assert {key: arguments["mode"][key] for key in ("source", "value", "validator")} == {
        "source": "value",
        "value": "fast",
        "validator": True,
    }
    assert "rng" not in arguments["mode"]
    assert arguments["integer"]["source"] == "rng"
    assert "value" not in arguments["integer"]
    assert types["constraints"] == [
        {"name": "even", "enabled": True},
        {"name": "small", "enabled": False},
    ]
    zeros, marked = types["directed_vectors"]
    assert (zeros["name"], zeros["id"], zeros["values"]["real"]) == (
        "zeros",
        "directed-zeros",
        {"$float": "nan"},
    )
    assert marked["values"]["data"] == {"$repr": "(1, 2)", "$type": "tuple"}
    # The two registrations of ex_bus: one's context is tests/a's, the other's the
    # rootdir's; a factory that does not declare ctx has none
    assert entries["ex_bus", "tests/a"]["context"] is None
    assert entries["ex_bus", "tests/b"]["context"] is not None
    assert entries["ex_types", "tests/a"]["context"] is not None
    assert entries["ex_types", "tests/a"]["context"] != entries["ex_bus", "tests/b"]["context"]


# Writes the context each collected strategy row received, by the row's folder
CONTEXTS_HOOK = """

import json

from pytest_strategy import VECTOR_KEY


def pytest_collection_modifyitems(config, items):
    found = {}
    for item in items:
        info = item.stash.get(VECTOR_KEY, None)
        if info is not None:
            folder = item.path.parent.relative_to(config.rootpath).as_posix()
            found.setdefault(folder, set()).add(info.context)
    contexts = {folder: sorted(values) for folder, values in found.items()}
    (config.rootpath / "contexts.json").write_text(json.dumps(contexts), encoding="utf-8")
"""


@pytest.mark.parametrize(
    ("args", "nsamples"), [((), 10), (("--nsamples=7",), 7), (("--nsamples=auto",), "auto")]
)
def test_each_entry_has_its_folders_context_and_the_sessions_nsamples(project, args, nsamples):
    (project.path / "conftest.py").write_text(ROOT_CONFTEST + CONTEXTS_HOOK, encoding="utf-8")

    exported = loads(export(project, "--strategy-constraint-off=ex_types:small", *args))

    assert exported["nsamples"] == nsamples
    # The tests of each folder received one context, the one export gives the
    # factory registered there; the tests in tests/a and tests/b differ
    contexts = json.loads((project.path / "contexts.json").read_text(encoding="utf-8"))
    assert {folder: len(values) for folder, values in contexts.items()} == {
        "tests/a": 1,
        "tests/b": 1,
    }
    entries = {(e["name"], e["origin"]["folder"]): e for e in exported["strategies"]}
    assert entries["ex_types", "tests/a"]["context"] == contexts["tests/a"][0]
    assert entries["ex_bus", "tests/b"]["context"] == contexts["tests/b"][0]
    assert contexts["tests/a"] != contexts["tests/b"]
