"""
End-to-end tests for the check that every pytest-xdist worker generated the same
vectors (D9), on two workers (``-n 2``).

Each worker sends the fingerprint of each context it computed and a digest of each
strategy's values; when two workers differ, the run fails with exit code 4 (if it
would have passed) and names the context or the strategy. Test IDs name the rows, so
xdist itself no longer notices workers that generated different values under the
same IDs. The controller also prints the context line and the reproduce line's
contexts from what the workers sent, since it collects nothing itself.

The projects' rootdir conftest.py files write, on each worker, the canonical JSON
text of every item's node ID and values (the fingerprint's encoding) and their
reprs to ``values-<worker>.json``, from which the expected digests are computed
here.
"""

import hashlib
import json
import os
import subprocess
import sys
from textwrap import dedent

import pytest

from pytest_strategy._fingerprint import fingerprint

pytest_plugins = ["pytester"]

SEED = 5

# Writes, on each worker, the canonical text of every item's node ID and strategy
# values, and the reprs of those values
DUMP = """
import json
import os

from pytest_strategy import VECTORS_KEY
from pytest_strategy._fingerprint import canonical

def pytest_collection_finish(session):
    text = canonical(session.config.rootpath)
    rows = [
        [
            info.strategy,
            text([item.nodeid, list(info.values)]),
            [repr(value) for value in info.values],
        ]
        for item in session.items
        for info in item.stash.get(VECTORS_KEY, ())
    ]
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    (session.config.rootpath / f"values-{worker}.json").write_text(json.dumps(rows))
"""

# Starts each worker with a hash seed of its own (1, then 2), as two machines would
PER_WORKER_HASH_SEED = """
import os

def pytest_xdist_setupnodes(config, specs):
    os.environ["PYTHONHASHSEED"] = "1"

def pytest_xdist_newgateway(gateway):
    os.environ["PYTHONHASHSEED"] = str(int(os.environ["PYTHONHASHSEED"]) + 1)
"""

# A context that differs between the workers, and is used only as a bound
PER_WORKER_LIMIT = """
import os

def pytest_strategies_context(config):
    return {"limit": 1 if os.environ["PYTEST_XDIST_WORKER"] == "gw0" else 2}
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("bounded")
def bounded(ctx):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, ctx["limit"])), nsamples=12)

@register("plain")
def plain():
    return Parameter(TestArg("y", rng_type=RNGInteger(0, 9)), nsamples=4)
"""

TESTS = """
from pytest_strategy import strategy

@strategy("bounded")
def test_bounded(x):
    assert {condition}

@strategy("plain")
def test_plain(y):
    pass
"""

HINT = [
    "Make pytest_strategies_context and the strategy factories give the same result "
    "in every worker:",
    "no temporary paths, process IDs, times, unseeded random values or lists built from sets",
    "(leave them out, or use pydantic Field(exclude=True) in a context).",
]


def write(pytester, files):
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))


def run(pytester, *args):
    """Run the project in a new process on two pytest-xdist workers."""
    pytest.importorskip("xdist")
    return pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", "-n", "2", f"--rng-seed={SEED}", *args
    )


def digests(pytester, worker):
    """The digest of each strategy's values on ``worker``, computed from its dump."""
    rows = json.loads((pytester.path / f"values-{worker}.json").read_text())
    hashes = {}
    for strategy, text, _ in rows:
        hashes.setdefault(strategy, hashlib.sha256()).update(text.encode() + b"\n")
    return {strategy: digest.hexdigest()[:8] for strategy, digest in hashes.items()}


def reprs(pytester, worker):
    """The reprs of every row's values on ``worker``, from its dump."""
    rows = json.loads((pytester.path / f"values-{worker}.json").read_text())
    return [shown for _, _, shown in rows]


def assert_two_hash_orders():
    """Check that hash seeds 1 and 2 give list({"a", "b", "c"}) two orders."""
    orders = {
        subprocess.run(
            [sys.executable, "-c", 'print(list({"a", "b", "c"}))'],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(orders) == 2, "hash seeds 1 and 2 give one order: pick two others"


def fp(value, pytester):
    return fingerprint(value, pytester.path)[0]


def fp_of_repr(name, text):
    """The fingerprint of an object of the class ``name`` whose repr is ``text``."""
    shown = type(name, (), {"__repr__": lambda self: text})
    return fingerprint(shown())[0]


def section(result):
    """The lines of the check's message, from its first line to its last."""
    lines = result.stdout.lines
    start = lines.index("pytest-strategies: the xdist workers generated different vectors:")
    return lines[start : lines.index(HINT[-1], start) + 1]


class TestWorkersThatDiffer:
    def test_a_context_used_only_as_a_bound_exits_4(self, pytester):
        write(
            pytester,
            {
                "conftest.py": DUMP + PER_WORKER_LIMIT,
                "strategies.py": STRATEGIES,
                "test_x.py": TESTS.format(condition="True"),
            },
        )

        result = run(pytester)

        # The IDs are the same on both workers, so xdist runs the tests
        result.assert_outcomes(passed=16)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        gw0, gw1 = digests(pytester, "gw0"), digests(pytester, "gw1")
        assert gw0["bounded"] != gw1["bounded"]
        assert gw0["plain"] == gw1["plain"]
        assert section(result) == [
            "pytest-strategies: the xdist workers generated different vectors:",
            f"  context conftest.py: gw0 {fp({'limit': 1}, pytester)}, "
            f"gw1 {fp({'limit': 2}, pytester)}",
            f"  values of strategy bounded: gw0 {gw0['bounded']}, gw1 {gw1['bounded']}",
            *HINT,
        ]

    def test_a_failed_run_keeps_its_exit_code_and_the_message_comes_first(self, pytester):
        write(
            pytester,
            {
                "conftest.py": DUMP + PER_WORKER_LIMIT,
                "strategies.py": STRATEGIES,
                "test_x.py": TESTS.format(condition="x < 0"),
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=4, failed=12)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: the xdist workers generated different vectors:",
                "  context conftest.py: gw0 *, gw1 *",
                "  values of strategy bounded: gw0 *, gw1 *",
                *HINT,
                f"pytest-strategies: reproduce with --rng-seed={SEED} (context *)",
            ]
        )

    def test_a_list_built_from_a_set_under_two_hash_seeds_exits_4(self, pytester):
        assert_two_hash_orders()
        strategies = """
from pytest_strategy import Parameter, RNGChoice, TestArg, register

@register("lanes")
def lanes():
    return Parameter(TestArg("lane", rng_type=RNGChoice(list({"a", "b", "c"}))), nsamples=8)
"""
        write(
            pytester,
            {
                "conftest.py": DUMP + PER_WORKER_HASH_SEED,
                "strategies.py": strategies,
                "test_x.py": "from pytest_strategy import strategy\n\n"
                "@strategy('lanes')\ndef test_lane(lane):\n    pass\n",
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=8)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        gw0, gw1 = digests(pytester, "gw0"), digests(pytester, "gw1")
        assert section(result) == [
            "pytest-strategies: the xdist workers generated different vectors:",
            f"  values of strategy lanes: gw0 {gw0['lanes']}, gw1 {gw1['lanes']}",
            *HINT,
        ]

    def test_a_factory_that_draws_from_the_global_random_exits_4(self, pytester):
        strategies = """
import random

from pytest_strategy import Parameter, TestArg, register

@register("unseeded")
def unseeded():
    return Parameter(TestArg("x", value=random.random()), nsamples=2)
"""
        write(
            pytester,
            {
                "conftest.py": DUMP,
                "strategies.py": strategies,
                "test_x.py": "from pytest_strategy import strategy\n\n"
                "@strategy('unseeded')\ndef test_x(x):\n    pass\n",
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=2)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: the xdist workers generated different vectors:",
                "  values of strategy unseeded: gw0 *, gw1 *",
                *HINT,
            ]
        )

    def test_objects_whose_reprs_do_not_show_the_sets_they_hold_exit_4(self, pytester):
        """
        A context and values with reprs of their own that hold sets of strings they
        do not show (pytest's config, a register's chip) keep their state.
        """
        conftest = """
import os

class Testbench:
    def __init__(self, config, channels):
        self.config = config
        self.channels = channels

    def __repr__(self):
        return f"Testbench(channels={self.channels})"

def pytest_strategies_context(config):
    return Testbench(config, 4 if os.environ["PYTEST_XDIST_WORKER"] == "gw0" else 8)
"""
        strategies = """
import os

from pytest_strategy import Parameter, RNGChoice, TestArg, register

class Reg:
    def __init__(self, chip, name):
        self.chip = chip
        self.name = name

    def __repr__(self):
        return f"Reg({self.name!r})"

class Chip:
    def __init__(self):
        self.tags = {"ro", "rw"}
        self.regs = [Reg(self, f"r{i}") for i in range(8)]

CHIP = Chip()

@register("regs")
def regs():
    first = 0 if os.environ["PYTEST_XDIST_WORKER"] == "gw0" else 4
    return Parameter(TestArg("reg", rng_type=RNGChoice(CHIP.regs[first : first + 4])), nsamples=3)
"""
        tests = """
from pytest_strategy import strategy

@strategy("regs")
def test_reg(reg, strategies_ctx):
    pass
"""
        write(
            pytester,
            {"conftest.py": DUMP + conftest, "strategies.py": strategies, "test_x.py": tests},
        )

        result = run(pytester)

        result.assert_outcomes(passed=3)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        gw0, gw1 = digests(pytester, "gw0"), digests(pytester, "gw1")
        four, eight = (fp_of_repr("Testbench", f"Testbench(channels={n})") for n in (4, 8))
        assert section(result) == [
            "pytest-strategies: the xdist workers generated different vectors:",
            f"  context conftest.py: gw0 {four}, gw1 {eight}",
            f"  values of strategy regs: gw0 {gw0['regs']}, gw1 {gw1['regs']}",
            *HINT,
        ]

    def test_series_values_per_worker_follow_xdist_s_own_message(self, pytester):
        conftest = """
import os

def pytest_strategies_context(config):
    return {"channels": [0, 1] if os.environ["PYTEST_XDIST_WORKER"] == "gw0" else [0, 2]}
"""
        strategies = """
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

@register("chans")
def chans(ctx):
    return Parameter(
        TestArg("ch", rng_type=Series(ctx["channels"])),
        TestArg("x", rng_type=RNGInteger(0, 9)),
        nsamples=4,
    )
"""
        write(
            pytester,
            {
                "conftest.py": conftest,
                "strategies.py": strategies,
                "test_x.py": "from pytest_strategy import strategy\n\n"
                "@strategy('chans')\ndef test_x(ch, x):\n    pass\n",
            },
        )

        result = run(pytester)

        # xdist runs nothing and reports a collection error, and that stays the result
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.stdout.fnmatch_lines(
            [
                # Between the worker that collected first and the other one
                "Different tests were collected between gw? and gw?. The difference is:",
                "pytest-strategies: the xdist workers generated different vectors:",
                f"  context conftest.py: gw0 {fp({'channels': [0, 1]}, pytester)}, "
                f"gw1 {fp({'channels': [0, 2]}, pytester)}",
                "  values of strategy chans: gw0 *, gw1 *",
                *HINT,
            ]
        )


class TestWorkersThatAgree:
    def test_a_test_that_changes_the_context_on_one_worker_exits_0(self, pytester):
        conftest = """
def pytest_strategies_context(config):
    return {"limit": 5, "runs": []}
"""
        tests = """
from pytest_strategy import strategy

def test_changes(strategies_ctx):
    strategies_ctx["limit"] = 99
    strategies_ctx["runs"].append(1)

@strategy("bounded")
def test_bounded(x, strategies_ctx):
    pass
"""
        write(
            pytester,
            {"conftest.py": DUMP + conftest, "strategies.py": STRATEGIES, "test_x.py": tests},
        )

        result = run(pytester)

        result.assert_outcomes(passed=13)
        assert result.ret == pytest.ExitCode.OK
        result.stdout.no_fnmatch_line("*different vectors*")
        result.stdout.fnmatch_lines(
            [f"pytest-strategies: context {fp({'limit': 5, 'runs': []}, pytester)}"]
        )

    def test_a_change_before_a_wrapper_folder_asks_on_one_worker_exits_0(self, pytester):
        # On gw0 a session fixture changes the rootdir's object before tests/w first
        # asks for its context, which tests/w's wrapper builds from that object; gw1
        # builds it from the unchanged object
        conftest = """
import os

import pytest
from pytest_strategy import get_context

def pytest_strategies_context(config):
    return {"seen": []}

@pytest.fixture(autouse=True, scope="session")
def _first(request):
    if os.environ["PYTEST_XDIST_WORKER"] == "gw0":
        get_context(request.config, __file__)["seen"].append("gw0")
    get_context(request.config, request.config.rootpath / "tests/w")
"""
        wrapper = """
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_strategies_context(config):
    return {**(yield), "w": 1}
"""
        tests = """
import pytest
from pytest_strategy import get_context

@pytest.mark.parametrize("i", range(8))
def test_w(request, i):
    assert get_context(request.config, __file__)["w"] == 1
"""
        write(
            pytester,
            {"conftest.py": conftest, "tests/w/conftest.py": wrapper, "tests/w/test_w.py": tests},
        )

        result = run(pytester)

        result.assert_outcomes(passed=8)
        assert result.ret == pytest.ExitCode.OK
        result.stdout.no_fnmatch_line("*different vectors*")

    def test_values_and_a_context_whose_reprs_show_sets_in_hash_order_exit_0(self, pytester):
        pytest.importorskip("attrs")
        pytest.importorskip("pydantic")
        assert_two_hash_orders()
        conftest = """
import attrs

NAMES = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}

@attrs.frozen
class Bench:
    lanes: frozenset
    width: int

def pytest_strategies_context(config):
    return Bench(frozenset(NAMES), 8)
"""
        strategies = """
import attrs
import pydantic
from conftest import NAMES
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@attrs.define
class Lanes:
    names: frozenset

class Txn(pydantic.BaseModel):
    lanes: set[str]

class Tags:
    def __init__(self, names):
        self.names = names

    def __repr__(self):
        return f"Tags({self.names!r})"

@register("benches")
def benches(ctx):
    return Parameter(
        TestArg("width", rng_type=RNGInteger(1, ctx.width)),
        TestArg("lanes", value=Lanes(frozenset(NAMES))),
        TestArg("txn", value=Txn(lanes=NAMES)),
        TestArg("tags", value=Tags(NAMES)),
        nsamples=4,
    )
"""
        write(
            pytester,
            {
                "conftest.py": DUMP + PER_WORKER_HASH_SEED + conftest,
                "strategies.py": strategies,
                "test_x.py": "from pytest_strategy import strategy\n\n"
                "@strategy('benches')\ndef test_bench(width, lanes, txn, tags):\n    pass\n",
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=4)
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        result.stdout.no_fnmatch_line("*different vectors*")
        result.stdout.fnmatch_lines(["pytest-strategies: context ????????"])
        # The reprs show the sets in each worker's order, the digests do not
        assert reprs(pytester, "gw0") != reprs(pytester, "gw1")
        assert digests(pytester, "gw0") == digests(pytester, "gw1")

    def test_the_controller_prints_the_context_line(self, pytester):
        write(
            pytester,
            {
                "conftest.py": DUMP + "def pytest_strategies_context(config):\n"
                "    return {'limit': 3}\n",
                "strategies.py": STRATEGIES,
                "test_x.py": TESTS.format(condition="True"),
                "tests/tb_a/conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'limit': 7}\n",
                "tests/tb_a/test_a.py": TESTS.format(condition="True"),
            },
        )

        result = run(pytester, "-q")

        result.assert_outcomes(passed=32)
        assert result.ret == pytest.ExitCode.OK
        assert digests(pytester, "gw0") == digests(pytester, "gw1")
        lines = [line for line in result.stdout.lines if line.startswith("pytest-strategies: con")]
        assert lines == [
            f"pytest-strategies: contexts conftest.py {fp({'limit': 3}, pytester)}, "
            f"tests/tb_a/conftest.py {fp({'limit': 7}, pytester)}"
        ]
        result.stdout.no_fnmatch_line("*different vectors*")

    def test_the_context_line_lists_the_contexts_the_collection_computed(self, pytester):
        # A test of tests/tb_b computes its folder's context while it runs: the
        # controller lists only the contexts the workers computed while they
        # collected, as a run in one process does after its collection
        write(
            pytester,
            {
                "conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'limit': 3}\n",
                "strategies.py": STRATEGIES,
                "test_x.py": TESTS.format(condition="True"),
                "tests/tb_b/conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'limit': 7}\n",
                "tests/tb_b/test_b.py": """
from pytest_strategy import get_context

def test_b(request):
    assert get_context(request.config, __file__) == {"limit": 7}
""",
            },
        )

        one_process = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", f"--rng-seed={SEED}", "-q"
        )
        distributed = run(pytester, "-q")

        for result in (one_process, distributed):
            result.assert_outcomes(passed=17)
            assert [
                line for line in result.stdout.lines if line.startswith("pytest-strategies: con")
            ] == [f"pytest-strategies: context {fp({'limit': 3}, pytester)}"]

    def test_the_reproduce_line_has_the_failed_tests_contexts(self, pytester):
        write(
            pytester,
            {
                "conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'limit': 3}\n",
                "strategies.py": STRATEGIES,
                "test_x.py": TESTS.format(condition="True"),
                "tests/tb_a/conftest.py": "def pytest_strategies_context(config):\n"
                "    return {'limit': 7}\n",
                "tests/tb_a/test_a.py": TESTS.format(condition="x < 0"),
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=20, failed=12)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.stdout.no_fnmatch_line("*different vectors*")
        reproduce = [line for line in result.stdout.lines if "reproduce with" in line]
        assert reproduce == [
            f"pytest-strategies: reproduce with --rng-seed={SEED} "
            f"(context {fp({'limit': 7}, pytester)})"
        ]

    def test_without_a_context_nothing_is_printed(self, pytester):
        write(
            pytester,
            {
                "strategies.py": STRATEGIES,
                "test_x.py": "from pytest_strategy import strategy\n\n"
                "@strategy('plain')\ndef test_plain(y):\n    pass\n",
            },
        )

        result = run(pytester)

        result.assert_outcomes(passed=4)
        assert result.ret == pytest.ExitCode.OK
        result.stdout.no_fnmatch_line("pytest-strategies: context*")
        result.stdout.no_fnmatch_line("*different vectors*")
