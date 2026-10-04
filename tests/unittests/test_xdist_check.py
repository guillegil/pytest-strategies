"""
Unit tests for the check that every pytest-xdist worker generated the same vectors
(D9): what a worker sends when its session finishes (the fingerprint of each
context it computed and the digest of each strategy's values), and how the
controller compares what the workers sent, with fake worker nodes and sessions.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pytest_strategy import VECTORS_KEY, VectorInfo
from pytest_strategy._context import NO_ANSWER, Answer
from pytest_strategy._fingerprint import canonical
from pytest_strategy._runtime import runtime
from pytest_strategy._vector import vector_type
from pytest_strategy.plugin import (
    _DIFFERENT_VECTORS,
    _DIFFERENT_VECTORS_HINT,
    _check,
    _differences,
    _plugin_instance,
    _value_digests,
)

CHECK = "pytest_strategies_check"
SUMMARY = "pytest_strategies_summary"

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml"


def info(strategy, *values, names=("a", "b")):
    """The VectorInfo of a random row of ``strategy`` with ``values``."""
    return VectorInfo(
        strategy=strategy,
        origin="strategies.py:3",
        kind="random",
        name=None,
        index=0,
        enumerated=(),
        values=vector_type(names[: len(values)])(*values),
        id="rand-0",
        seed=1,
        context=None,
        constraints_off=(),
    )


def item(nodeid, *infos):
    """A collected item whose strategies' rows are ``infos``."""
    return SimpleNamespace(nodeid=nodeid, stash={VECTORS_KEY: infos} if infos else {})


def expected_digest(*rows):
    """
    The digest of rows given as (node ID, values): the SHA-256 of the canonical
    JSON text of each row, the encoding of the context fingerprint.
    """
    digest = hashlib.sha256()
    text = canonical()
    for nodeid, values in rows:
        digest.update(text([nodeid, list(values)]).encode() + b"\n")
    return digest.hexdigest()[:8]


# Prints the value digests of rows that hold a pydantic model and an attrs instance
# with sets of strings, and the reprs of those values, one JSON object
DIGESTS = """
import json
from types import SimpleNamespace

import attrs
import pydantic

from pytest_strategy import VECTORS_KEY, VectorInfo
from pytest_strategy.plugin import _value_digests

NAMES = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}

@attrs.define
class Bench:
    lanes: frozenset

class Txn(pydantic.BaseModel):
    lanes: set[str]

class Tags:
    def __init__(self, names):
        self.names = names

    def __repr__(self):
        return f"Tags({self.names!r})"

values = (Bench(frozenset(NAMES)), Txn(lanes=NAMES), SimpleNamespace(lanes=NAMES), Tags(NAMES))
row = VectorInfo(
    strategy="benches", origin="strategies.py:3", kind="random", name=None, index=0,
    enumerated=(), values=values, id="rand-0", seed=1, context=None,
    constraints_off=(),
)
items = [SimpleNamespace(nodeid="t.py::test_a[rand-0]", stash={VECTORS_KEY: (row,)})]
print(json.dumps({"digests": _value_digests(items), "reprs": [repr(v) for v in values]}))
"""


def _digests_under(hash_seed):
    """Run DIGESTS in a new interpreter under PYTHONHASHSEED ``hash_seed``."""
    output = subprocess.run(
        [sys.executable, "-c", DIGESTS],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


def node(worker, contexts=None, values=None, **summary):
    """A worker node that finished, with its part of the check and its summary."""
    output = {
        CHECK: {"contexts": contexts or {}, "values": values or {}},
        SUMMARY: {"count": 0, "lines": [], "names": [], "contexts": [], **summary},
    }
    return SimpleNamespace(gateway=SimpleNamespace(id=worker), workeroutput=output)


def crashed(worker):
    """A worker node that crashed: it sent nothing."""
    return SimpleNamespace(gateway=SimpleNamespace(id=worker))


@pytest.fixture
def controller():
    """A runtime session of a pytest-xdist controller (a config without workerinput)."""
    state = runtime.push(SimpleNamespace(option=SimpleNamespace(dist="load", verbose=0)))
    state.run_seed = 9
    try:
        yield state
    finally:
        runtime.pop()


def finish(*nodes, exitstatus=pytest.ExitCode.OK):
    """Report ``nodes`` down, finish the session, and return its exit status."""
    for each in nodes:
        _plugin_instance.pytest_testnodedown(node=each, error=None)
    session = SimpleNamespace(config=runtime.current.config, items=[], exitstatus=exitstatus)
    _plugin_instance.pytest_sessionfinish(session)
    return session.exitstatus


class Terminal:
    """A terminal reporter that keeps the lines written, and whether each was red."""

    def __init__(self):
        self.lines = []

    def write_line(self, line, **markup):
        self.lines.append((line, bool(markup.get("red"))))

    def section(self, title):
        self.lines.append((f"== {title} ==", False))


class TestDifferences:
    def test_only_keys_with_different_values_are_reported(self):
        checks = {
            "gw0": {
                "contexts": {"conftest.py": "1a2b3c4d", "tests/a/conftest.py": "5e6f7a8b"},
                "values": {"burst": "00000001", "esm": "00000002", "gw0_only": "0000000a"},
            },
            "gw1": {
                "contexts": {
                    "conftest.py": "1a2b3c4d",
                    "tests/a/conftest.py": "9f8e7d6c",
                    "tests/b/conftest.py": "0c1d2e3f",
                },
                "values": {"burst": "00000001", "esm": "00000003"},
            },
        }

        assert _differences(checks) == [
            "  context tests/a/conftest.py: gw0 5e6f7a8b, gw1 9f8e7d6c",
            "  values of strategy esm: gw0 00000002, gw1 00000003",
        ]

    def test_workers_in_the_order_of_their_numbers_each_with_its_value(self):
        checks = {
            worker: {"contexts": {}, "values": {"burst": value}}
            for worker, value in (("gw10", "b"), ("gw2", "a"), ("gw1", "a"))
        }

        assert _differences(checks) == ["  values of strategy burst: gw1 a, gw2 a, gw10 b"]

    def test_none_and_errors_are_compared_as_values(self):
        checks = {
            "gw0": {"contexts": {"none": "none", "tests/a/conftest.py": "error: OSError"}},
            "gw1": {"contexts": {"none": "none", "tests/a/conftest.py": "1a2b3c4d"}},
        }

        assert _differences(checks) == [
            "  context tests/a/conftest.py: gw0 error: OSError, gw1 1a2b3c4d"
        ]

    def test_one_worker_has_nothing_to_compare(self):
        assert _differences({"gw0": {"contexts": {"conftest.py": "1"}, "values": {}}}) == []
        assert _differences({}) == []


class TestController:
    def test_a_crashed_worker_and_keys_on_one_worker_only_are_left_out(self, controller):
        status = finish(
            node("gw0", {"conftest.py": "1a2b3c4d", "tests/b/conftest.py": "77777777"}),
            crashed("gw1"),
            node("gw2", {"conftest.py": "9f8e7d6c"}, {"burst": "0c1d2e3f"}),
        )

        assert status == pytest.ExitCode.USAGE_ERROR
        assert controller.worker_differences == [
            "  context conftest.py: gw0 1a2b3c4d, gw2 9f8e7d6c"
        ]
        assert sorted(controller.worker_checks) == ["gw0", "gw2"]

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            (pytest.ExitCode.OK, pytest.ExitCode.USAGE_ERROR),
            (pytest.ExitCode.NO_TESTS_COLLECTED, pytest.ExitCode.USAGE_ERROR),
            (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.TESTS_FAILED),
            (pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERRUPTED),
            (pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERNAL_ERROR),
            (pytest.ExitCode.USAGE_ERROR, pytest.ExitCode.USAGE_ERROR),
        ],
        ids=lambda code: code.name,
    )
    def test_the_exit_status_changes_only_from_0_or_5(self, controller, before, after):
        status = finish(
            node("gw0", values={"burst": "00000001"}),
            node("gw1", values={"burst": "00000002"}),
            exitstatus=before,
        )

        assert status == after

    def test_workers_that_agree_leave_the_exit_status(self, controller):
        same = {"conftest.py": "1a2b3c4d"}, {"burst": "00000001"}

        assert finish(node("gw0", *same), node("gw1", *same), crashed("gw2")) == 0
        assert finish(exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED) == 5
        assert controller.worker_differences == []

    def test_the_terminal_summary(self, controller):
        finish(
            node(
                "gw1",
                {"conftest.py": "9f8e7d6c"},
                collection_contexts={"conftest.py": "9f8e7d6c"},
                failed_contexts={"conftest.py": "9f8e7d6c"},
            ),
            node(
                "gw0",
                {"conftest.py": "1a2b3c4d"},
                collection_contexts={"conftest.py": "1a2b3c4d", "tests/a/conftest.py": "5e6f7a8b"},
                failed_contexts={"tests/a/conftest.py": "5e6f7a8b"},
            ),
        )
        terminal = Terminal()

        _plugin_instance.pytest_terminal_summary(
            terminal, pytest.ExitCode.TESTS_FAILED, controller.config
        )

        # The contexts as gw0, the first worker, printed them after its collection,
        # then the differences in red, then the contexts the failed tests received on
        # each worker
        assert terminal.lines == [
            (
                "pytest-strategies: contexts conftest.py 1a2b3c4d, " "tests/a/conftest.py 5e6f7a8b",
                False,
            ),
            (_DIFFERENT_VECTORS, True),
            ("  context conftest.py: gw0 1a2b3c4d, gw1 9f8e7d6c", True),
            *[(line, True) for line in _DIFFERENT_VECTORS_HINT],
            (
                "pytest-strategies: reproduce with --rng-seed=9 (contexts conftest.py "
                "9f8e7d6c, tests/a/conftest.py 5e6f7a8b)",
                False,
            ),
        ]

    def test_the_message_matches_the_plan(self):
        assert [_DIFFERENT_VECTORS, *_DIFFERENT_VECTORS_HINT] == [
            "pytest-strategies: the xdist workers generated different vectors:",
            "Make pytest_strategies_context and the strategy factories give the same result "
            "in every worker:",
            "no temporary paths, process IDs, times, unseeded random values or lists built "
            "from sets",
            "(leave them out, or use pydantic Field(exclude=True) in a context).",
        ]


class TestValueDigests:
    def test_one_digest_per_strategy_over_its_items_in_collection_order(self):
        items = [
            item("t.py::test_a[rand-0]", info("burst", 1, "x")),
            item("t.py::test_plain"),
            item("t.py::test_b[rand-0-rand-0]", info("esm", 2.5), info("burst", 3, None)),
        ]

        assert _value_digests(items) == {
            "burst": expected_digest(
                ("t.py::test_a[rand-0]", (1, "x")), ("t.py::test_b[rand-0-rand-0]", (3, None))
            ),
            "esm": expected_digest(("t.py::test_b[rand-0-rand-0]", (2.5,))),
        }
        assert list(_value_digests(items[::-1])) == ["burst", "esm"]
        assert _value_digests(items[::-1])["burst"] != _value_digests(items)["burst"]

    def test_the_node_ids_count(self):
        one = _value_digests([item("t.py::test_a[rand-0]", info("burst", 1))])
        other = _value_digests([item("t.py::test_a[rand-1]", info("burst", 1))])

        assert one != other

    def test_values_by_their_canonical_encoding(self):
        class Handle:
            pass

        values = ({"beta", "alpha"}, Handle())

        assert _value_digests([item("t.py::test_a[rand-0]", info("burst", *values))]) == {
            "burst": expected_digest(("t.py::test_a[rand-0]", values))
        }
        # Sets in sorted order, an object with the default repr by its type's name
        assert json.loads(canonical()(list(values))) == [
            {"set": ["alpha", "beta"]},
            {"object": Handle.__qualname__},
        ]

    def test_paths_inside_the_rootdir_by_their_relative_path(self, tmp_path):
        def digest(root):
            values = info("burst", root / "data" / "a.bin")
            return _value_digests([item("t.py::test_a[rand-0]", values)], root)

        assert digest(tmp_path / "one") == digest(tmp_path / "two")

    def test_one_digest_under_every_hash_seed(self):
        pytest.importorskip("attrs")
        pytest.importorskip("pydantic")
        runs = [_digests_under(seed) for seed in ("1", "2", "3")]

        # The values' reprs show their sets in other orders
        assert len({tuple(run["reprs"]) for run in runs}) > 1
        assert len({json.dumps(run["digests"]) for run in runs}) == 1
        assert "unavailable" not in runs[0]["digests"].values()

    def test_a_value_whose_repr_raises_makes_its_strategy_unavailable(self):
        class Broken:
            def __repr__(self):
                raise RuntimeError("no repr")

        items = [
            item("t.py::test_a[rand-0]", info("burst", 1)),
            item("t.py::test_a[rand-1]", info("burst", Broken())),
            item("t.py::test_b[rand-0]", info("esm", 2)),
            # A later row of the strategy is not encoded
            item("t.py::test_a[rand-2]", info("burst", 3)),
        ]

        digests = _value_digests(items)

        assert digests["burst"] == "unavailable"
        assert digests["esm"] == expected_digest(("t.py::test_b[rand-0]", (2,)))

    def test_values_that_hold_a_large_object_their_repr_does_not_show(self):
        """
        Registers that hold their chip, which holds a set of strings and every
        register: each is written by its repr, without looking into the chip.
        """

        class Chip:
            def __init__(self, size):
                self.tags = {"ro", "rw"}
                self.regs = [Reg(self, f"r{i}") for i in range(size)]

        class Reg:
            def __init__(self, chip, name):
                self.chip = chip
                self.name = name
                self.fields = {"enable": 0, "mode": 1}

            def __repr__(self):
                return f"Reg({self.name!r}, {self.fields})"

        regs = Chip(20_000).regs
        items = [
            item(f"t.py::test_a[rand-{i}]", info("regs", regs[i % 5000])) for i in range(2_000)
        ]

        start = time.perf_counter()
        digests = _value_digests(items)
        # Well under a second; looking into the chip for each row took about 30 s
        assert time.perf_counter() - start < 2
        assert digests["regs"] != _value_digests(items[1:])["regs"]
        assert json.loads(canonical()(regs[0]))["repr"][1] == "Reg('r0', {'enable': 0, 'mode': 1})"


class TestWorker:
    @pytest.fixture
    def worker(self):
        """A runtime session of a pytest-xdist worker (a config with workerinput)."""
        config = SimpleNamespace(workerinput={"workerid": "gw0"}, workeroutput={})
        state = runtime.push(config)
        state.run_seed = 9
        try:
            yield state
        finally:
            runtime.pop()

    def test_the_check_has_strings_only(self):
        state = SimpleNamespace(
            contexts=SimpleNamespace(
                scopes=lambda: {
                    "conftest.py": Answer({"a": 1}, "conftest.py", fingerprint="1a2b3c4d"),
                    "none": NO_ANSWER,
                    "tests/a/conftest.py": Answer(
                        None, "tests/a/conftest.py", OSError("no bench"), None
                    ),
                    "tests/b/conftest.py": Answer(
                        None, "tests/b/conftest.py", pytest.skip.Exception("off"), None
                    ),
                }
            ),
            value_digests={"burst": "00000001"},
        )

        assert _check(state) == {
            "contexts": {
                "conftest.py": "1a2b3c4d",
                "none": "none",
                "tests/a/conftest.py": "error: OSError",
                "tests/b/conftest.py": "error: Skipped",
            },
            "values": {"burst": "00000001"},
        }

    def test_the_values_are_read_when_the_collection_finishes(self, worker):
        values = [1, 2]
        items = [item("t.py::test_a[rand-0]", info("burst", values))]
        session = SimpleNamespace(config=worker.config, items=items, exitstatus=0)
        _plugin_instance.pytest_collection_finish(session)
        # A test changes the list it received; the digest is the one of the collection
        values.append(3)

        _plugin_instance.pytest_sessionfinish(session)

        output = worker.config.workeroutput
        assert output[CHECK] == {
            "contexts": {},
            "values": {"burst": expected_digest(("t.py::test_a[rand-0]", ([1, 2],)))},
        }
        assert output[SUMMARY]["collection_contexts"] == {}
        assert output[SUMMARY]["failed_contexts"] == {}
        # A worker compares nothing, and its exit status is its own
        assert session.exitstatus == 0


class TestContinuousIntegration:
    def test_the_xdist_job_runs_the_projects_on_two_workers(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        # The examples job, which runs the examples on two workers
        job = re.search(r"\n  examples:\n(.*?)\n  \w+:\n", workflow, re.S)
        assert job is not None
        steps = re.split(r"\n      - ", job.group(1))

        (step,) = [step for step in steps if "test_xdist_check_integration.py" in step]
        assert "run: python -m pytest tests/integration/test_xdist_check_integration.py" in step
        assert "continue-on-error" not in step
        assert any("-n 2" in step for step in steps if "examples/" in step)
