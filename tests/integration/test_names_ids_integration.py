"""
End-to-end tests for the test IDs of strategy rows: the names format (the default)
against a checked-in golden list, selecting rows with -k and --vector-name, the
values format against the IDs pytest-strategies 3.0 gave the same rows, and the
strategies_ids ini option.

The golden lists are in tests/golden: names-ids.json is the --collect-only -q output
of NAMES_STRATEGIES and NAMES_TESTS (--nsamples=auto sorted, as its order depends on
the seed), and values-ids-3.0.json the output of pytest-strategies 3.0.0 for
VALUES_STRATEGIES and VALUES_TESTS, written there with Strategy.register,
Strategy.strategy and factories taking nsamples.
"""

import json
import random
from pathlib import Path

import pytest

from pytest_strategy import RNG, Strategy

pytest_plugins = ["pytester"]

GOLDEN = Path(__file__).parent.parent / "golden"


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


# Every kind of row; no constraint draws, so none can run out of retries. Every
# strategy has a test vector, so --vector-mode=test gives no empty parameter set
# (whose ID depends on the pytest version).
NAMES_STRATEGIES = """
from dataclasses import dataclass, field
from enum import Enum

from pytest_strategy import Parameter, RNGInteger, RNGSequence, Series, TestArg, register


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class Burst:
    addr: int
    len: int
    checked: bool = field(init=False, default=False)


@register("burst")
def burst():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 4095)),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={"zeros": {"addr": 0, "len": 1}, "max": (4095, 64)},
        test_vectors={"max": (4095, 64), "mid": {"len": 32, "addr": 2048}},
        nsamples=3,
    )


@register("esm")
def esm():
    # The Series arguments are enumerated: their values are in the IDs. 2 is listed
    # twice, so its second position is ch=2~1.
    return Parameter(
        TestArg("ch", rng_type=Series([0, 2, 2])),
        TestArg("dev", rng_type=Series(["a", "b"])),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        test_vectors={"broadcast": (0, "a", 255)},
        nsamples=8,
    )


@register("pairs")
def pairs():
    # The combinations the constraint rejects leave gaps in j. They have no random
    # argument, so which ones does not depend on the seed.
    return Parameter(
        TestArg("lo", rng_type=Series([1, 2])),
        TestArg("hi", rng_type=Series([1, 2])),
        vector_constraints={"ordered": lambda v: v.lo < v.hi},
        test_vectors={"wide": {"lo": 1, "hi": 2}},
        nsamples=2,
    )


@register("per_device")
def per_device():
    return Parameter(
        TestArg("device", rng_type=RNGSequence(["devA", "devB"])),
        TestArg("width", rng_type=RNGInteger(8, 32)),
        directed_vectors={"narrow": ("devA", 8)},
        test_vectors={"wide": ("devB", 32)},
        per_sequence_samples=True,
        nsamples=2,
    )


@register("drawn")
def drawn():
    # An RNGSequence is drawn in a finite run, and enumerated under --nsamples=auto
    return Parameter(
        TestArg("device", rng_type=RNGSequence(["devA", "devB", "devC"])),
        test_vectors={"first": ("devA",)},
        nsamples=2,
    )


@register("labels")
def labels():
    # One label per type; a tuple has no text, so it shows its position
    return Parameter(
        TestArg("v", rng_type=Series([None, True, 1.5, -3, "fast", Mode.SLOW, (1, 2), Burst])),
        test_vectors={"none": (None,)},
        nsamples=8,
    )


@register("empty")
def empty():
    return Parameter(
        TestArg("ch", rng_type=Series([], skip_if_empty="no channel in this config")),
        TestArg("x", rng_type=RNGInteger(0, 9)),
    )
"""

NAMES_TESTS = """
import pytest
from row_ids_strategies import Burst

from pytest_strategy import strategy


@strategy("burst")
def test_burst(addr, len):
    pass


@strategy("burst")
def test_record(b: Burst):
    assert not b.checked


@strategy("burst")
def test_vector_sizes(addr, len):
    pass


@strategy("esm")
def test_esm(ch, dev, wdata):
    pass


@strategy("pairs")
def test_pairs(lo, hi):
    assert lo < hi


@strategy("per_device")
def test_per_device(device, width):
    pass


@strategy("drawn")
def test_drawn(device):
    pass


@strategy("labels")
def test_labels(v):
    pass


@strategy("empty")
def test_empty(ch, x):
    pass


@pytest.mark.parametrize("flag", [True])
@strategy("pairs")
@strategy("burst")
def test_stacked(addr, len, lo, hi, flag):
    pass
"""

# Rows whose values do not depend on the seed, so 3.0 gave them the same values.
# Every strategy has a test vector, as for NAMES_STRATEGIES.
VALUES_STRATEGIES = """
from dataclasses import dataclass, field

from pytest_strategy import Parameter, RNGChoice, RNGInteger, Series, TestArg, register


@dataclass
class Burst:
    addr: object
    len: int
    checked: bool = field(init=False, default=False)


@register("v_burst")
def v_burst():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 4095)),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={
            "zeros": (0, 1),
            "max": (4095, 64),
            "named": ("a long register name, cut at 20", 8),
        },
        test_vectors={"mid": (2048, 32)},
        nsamples=0,
    )


@register("v_series")
def v_series():
    return Parameter(
        TestArg("ch", rng_type=Series([0, 2, 2])),
        TestArg("dev", rng_type=Series(["a", "b"])),
        test_vectors={"one": (0, "a")},
        nsamples=8,
    )


@register("v_point")
def v_point():
    return Parameter(
        TestArg("pt", rng_type=RNGChoice([(1, 2)])),
        directed_vectors={"origin": ((0, 0),), "far": (("x" * 90, 1),), "thing": (object(),)},
        test_vectors={"unit": ((1, 1),)},
        nsamples=0,
    )
"""

VALUES_TESTS = """
from value_ids_strategies import Burst

from pytest_strategy import strategy


@strategy("v_burst")
def test_named(addr, len):
    pass


@strategy("v_burst")
def test_record(b: Burst):
    pass


@strategy("v_series")
def test_series(ch, dev):
    pass


@strategy("v_point")
def test_point(pt):
    pass
"""

COLLECT = ("-p", "no:cacheprovider", "--collect-only", "-q")


def _golden(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def _collected(result):
    """Return the node IDs of the parametrized tests the run collected, in order."""
    return [line for line in result.outlines if "::" in line and "[" in line]


@pytest.fixture
def names_project(pytester):
    pytester.makepyfile(row_ids_strategies=NAMES_STRATEGIES, test_row_ids=NAMES_TESTS)
    return pytester


@pytest.fixture
def values_project(pytester):
    pytester.makepyfile(value_ids_strategies=VALUES_STRATEGIES, test_value_ids=VALUES_TESTS)
    return pytester


class TestGoldenNames:
    """The names format gives the same IDs for every seed."""

    @pytest.mark.parametrize("seed", [1, 2])
    @pytest.mark.parametrize("run", ["default", "--vector-mode=test"])
    def test_ids_match_the_golden_list(self, names_project, seed, run):
        args = () if run == "default" else (run,)

        result = names_project.runpytest(*COLLECT, f"--rng-seed={seed}", *args)

        assert _collected(result) == _golden("names-ids.json")[run]

    @pytest.mark.parametrize("seed", [1, 2])
    def test_auto_ids_match_the_golden_set(self, names_project, seed):
        """Only the order of the RNGSequence rows depends on the seed."""
        result = names_project.runpytest(*COLLECT, f"--rng-seed={seed}", "--nsamples=auto")

        assert sorted(_collected(result)) == _golden("names-ids.json")["--nsamples=auto"]


class TestSelectingRows:
    """-k and --vector-name select rows by the names in their IDs."""

    def test_k_selects_the_directed_rows_by_name(self, names_project):
        full = _collected(names_project.runpytest(*COLLECT, "--rng-seed=1"))

        result = names_project.runpytest(*COLLECT, "--rng-seed=1", "-k", "zeros")

        expected = [node_id for node_id in full if "[directed-zeros" in node_id]
        assert len(expected) == 5
        assert _collected(result) == expected

    def test_k_matches_test_names_not_row_ids(self, names_project):
        """No row ID contains "vector", whatever its values."""
        full = _collected(names_project.runpytest(*COLLECT, "--rng-seed=1"))

        result = names_project.runpytest(*COLLECT, "--rng-seed=1", "-k", "vector")

        expected = [node_id for node_id in full if "::test_vector_sizes[" in node_id]
        assert len(expected) == 5
        assert _collected(result) == expected

    def test_the_documented_k_recipes_select_rows_by_kind(self, pytester):
        """
        -k "directed-" selects exactly the directed rows and -k "not rand-" leaves out
        exactly the random rows. The bare words, which the docs first gave, also match
        a test or a vector whose name contains them.
        """
        pytester.makepyfile(
            kinds_strategies="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register


            @register("ops")
            def ops():
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    directed_vectors={"zeros": {"x": 0}, "operand_max": {"x": 9}},
                    nsamples=2,
                )
            """,
            test_kinds="""
            from pytest_strategy import strategy


            @strategy("ops")
            def test_write(x):
                pass


            def test_random_access():
                pass


            def test_directed_mode():
                pass
            """,
        )

        def selected(expression):
            result = pytester.runpytest(*COLLECT, "--rng-seed=1", "-k", expression)
            return [line.split("::")[1] for line in result.outlines if "::" in line]

        directed = ["test_write[directed-zeros]", "test_write[directed-operand_max]"]
        others = ["test_random_access", "test_directed_mode"]
        assert selected("directed-") == directed
        assert selected("not rand-") == [*directed, *others]
        # Substrings of the whole node name
        assert selected("directed") == [*directed, "test_directed_mode"]
        assert selected("not rand") == ["test_write[directed-zeros]", "test_directed_mode"]

    def test_vector_name_keeps_the_rows_id(self, names_project):
        full = _collected(names_project.runpytest(*COLLECT, "--rng-seed=1"))

        result = names_project.runpytest(*COLLECT, "--rng-seed=1", "--vector-name=max")

        # The other strategies have no such vector: their tests get an empty parameter
        # set, whose ID depends on the pytest version
        burst = ("::test_burst[", "::test_record[", "::test_vector_sizes[")
        selected = [i for i in _collected(result) if any(test in i for test in burst)]
        assert selected == [
            "test_row_ids.py::test_burst[directed-max]",
            "test_row_ids.py::test_record[directed-max]",
            "test_row_ids.py::test_vector_sizes[directed-max]",
        ]
        assert set(selected) <= set(full)


class TestValuesFormat:
    """-o strategies_ids=values gives the 3.0 IDs."""

    @pytest.mark.parametrize("run", ["default", "--vector-mode=test", "--nsamples=auto"])
    def test_ids_match_3_0_without_init_false_fields(self, values_project, run):
        args = () if run == "default" else (run,)

        result = values_project.runpytest(*COLLECT, "-o", "strategies_ids=values", *args)

        golden = _golden("values-ids-3.0.json")[run]
        # 3.0 also listed the record's init=False field
        assert any(",checked=False" in node_id for node_id in golden)
        assert _collected(result) == [i.replace(",checked=False", "") for i in golden]

    def test_names_is_the_default(self, values_project):
        result = values_project.runpytest(*COLLECT, "-o", "strategies_ids=names")

        assert _collected(result) == _collected(values_project.runpytest(*COLLECT))
        assert "test_value_ids.py::test_named[directed-zeros]" in _collected(result)


@pytest.mark.parametrize("value", ["foo", "Names", ""])
def test_an_unknown_ids_format_is_a_usage_error(names_project, value):
    result = names_project.runpytest("-p", "no:cacheprovider", "-o", f"strategies_ids={value}")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        [f"ERROR: strategies_ids must be 'names' or 'values', got '{value}'"]
    )


def test_an_unknown_ids_format_in_the_ini_file_is_a_usage_error(names_project):
    names_project.makeini("[pytest]\nstrategies_ids = value\n")

    result = names_project.runpytest("-p", "no:cacheprovider")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["ERROR: strategies_ids must be *, got 'value'"])
