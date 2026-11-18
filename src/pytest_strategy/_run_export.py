"""
The run export: two JSON files that describe the strategies a run used, written
when the session ends under ``--strategy-export=DIR`` (or the
``strategies_export_dir`` ini option, or a ``pytest_strategies_export_dir``
hook).

``strategies.json`` (``"kind": "strategy_definitions"``) has one entry per test
function, with each strategy it uses, as the test resolved it: the factory's
origin, the ``@strategy`` arguments, the context fingerprint, the effective
``nsamples`` with its source, and ``Parameter.to_dict()`` (arguments, directed
and test vectors, constraints, settings). No drawn values::

    {"schema": 1, "kind": "strategy_definitions",
     "generator": {"name": "pytest-strategies", "version": "4.3.0"},
     "seed": 1, "nsamples": 10,
     "tests": [
       {"nodeid": "tests/test_rw.py::test_rw",
        "strategies": [
          {"name": "rw", "origin": {...}, "arguments": {"device": "esm"},
           "context": null, "nsamples": {"value": 10, "source": "default"},
           "parameter": {"schema": 1, "arguments": [...], ...}}]}]}

``strategies.vectors.json`` (``"kind": "strategy_vectors"``) is the same
document, with the ``rows`` each strategy gave the test in this run (directed,
test, random, exhaustive and skipped rows), each with the test items that ran it
and their outcome::

    {"id": "rand-3", "kind": "random", "name": null, "index": 3,
     "enumerated": [], "values": {"addr": 812, "len": 7},
     "items": [{"nodeid": "tests/test_rw.py::test_rw[rand-3]", "outcome": "passed"}]}

Only the tests the run selected are written (after ``-k``, ``--deselect`` and
the plugins' own deselection). An outcome is ``passed``, ``failed``,
``skipped``, ``xfailed``, ``xpassed``, ``error`` (setup or teardown failed) or
null when the item did not run (``--collect-only``, ``-x``). Values use the
encoding of ``_encode.py``, as ``export_strategies()``.

Under pytest-xdist every worker collects every test: the workers send the
collected part, the controller adds the outcomes and writes the files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._encode import encode
from ._export import parameter_dict
from ._vector import VECTORS_KEY, VectorInfo

if TYPE_CHECKING:
    import pytest

# The schema version of both documents
SCHEMA = 1

DEFINITIONS_KIND = "strategy_definitions"
VECTORS_KIND = "strategy_vectors"
DEFINITIONS_FILE = "strategies.json"
VECTORS_FILE = "strategies.vectors.json"

# What a pytest-xdist worker sends the controller (workeroutput): the collected
# part of the export, as JSON text
WORKER_OUTPUT = "pytest_strategies_export"
# What the controller sends each worker (workerinput): whether to record it
WORKER_INPUT = "pytest_strategies_export"

# The keys of a VectorInfo.to_dict() a row keeps (the others are the strategy's)
_ROW_KEYS = ("id", "kind", "name", "index", "enumerated", "values")


def check_ini(config: pytest.Config) -> None:
    """
    Check the ``strategies_export_dir`` ini option.

    Raises:
        pytest.UsageError: For a value that pytest cannot read as a string (a
            number in a native TOML configuration file)
    """
    import pytest

    try:
        config.getini("strategies_export_dir")
    except TypeError as error:
        raise pytest.UsageError(f"strategies_export_dir must be a string; {error}") from None


def target(config: pytest.Config) -> Path | None:
    """
    Return the folder the run export goes to, or None when it is off.

    ``--strategy-export=DIR`` wins, relative to the folder pytest was invoked
    from; then the first ``pytest_strategies_export_dir`` implementation that
    returns a path; then the ``strategies_export_dir`` ini option. The last two
    are relative to the rootdir.
    """
    option = config.getoption("strategy_export", None)
    if option:
        return Path(config.invocation_params.dir, option)
    hooked = config.hook.pytest_strategies_export_dir(config=config)
    if hooked is not None and str(hooked) != "":
        return Path(config.rootpath, hooked)
    ini = config.getini("strategies_export_dir")
    if ini:
        return Path(config.rootpath, str(ini))
    return None


def definition(
    *,
    name: str,
    origin: dict[str, Any],
    arguments: Mapping[str, Any] | None,
    context: str | None,
    nsamples: int | str,
    source: str,
    parameter: Any,
    constraints_off: Iterable[str],
) -> dict[str, Any]:
    """
    Describe one strategy as one test resolved it: what ``strategies.json`` lists
    for it. A ``Parameter.to_dict()`` that JSON cannot hold (a custom RNG type's
    own ``to_dict()``) gives an ``error`` instead of a ``parameter``.
    """
    entry: dict[str, Any] = {
        "name": name,
        "origin": origin,
        "arguments": {key: encode(value) for key, value in (arguments or {}).items()},
        "context": context,
        "nsamples": {"value": nsamples, "source": source},
    }
    try:
        described = parameter_dict(parameter, constraints_off)
        json.dumps(described, allow_nan=False)
        entry["parameter"] = described
    except Exception as e:
        entry["error"] = {"type": type(e).__qualname__, "message": str(e)}
    return entry


def _row(info: VectorInfo) -> dict[str, Any]:
    data = info.to_dict()
    row = {key: data[key] for key in _ROW_KEYS}
    row["items"] = []
    return row


def _outcome(previous: str | None, report: pytest.TestReport) -> str | None:
    """The item's outcome after one more of its reports (a rerun starts again)."""
    xfail = hasattr(report, "wasxfail")
    if report.when == "setup":
        if report.failed:
            return "error"
        if report.skipped:
            return "xfailed" if xfail else "skipped"
        return None
    if report.when == "call":
        if report.failed:
            return "failed"
        if report.skipped:
            return "xfailed" if xfail else "skipped"
        return "xpassed" if xfail else "passed"
    # Teardown: a failure after a failed call keeps "failed", as pytest counts it
    if report.failed and previous != "failed":
        return "error"
    return previous


class RunExport:
    """What one session records for the run export."""

    def __init__(self, folder: Path | None) -> None:
        # The folder to write to; None in a pytest-xdist worker, which only records
        self.folder = folder
        # One entry per test and strategy, in the order the tests were resolved:
        # (test node ID without parameters, definition)
        self._uses: list[tuple[str, dict[str, Any]]] = []
        # The use of each row's VectorInfo, by identity; the infos are kept alive
        self._use_of: dict[int, int] = {}
        self._infos: list[VectorInfo] = []
        # The tests the run selected, with their rows: set when collection finishes
        # (or from a worker's output)
        self.collected: list[dict[str, Any]] | None = None
        # Each item's outcome, by node ID
        self.outcomes: dict[str, str | None] = {}
        # The files written, or why they were not, for the terminal summary
        self.written: list[Path] = []
        self.error: str | None = None

    def add_use(self, test: str, entry: dict[str, Any], infos: Iterable[VectorInfo]) -> None:
        """Record how a test resolved one strategy, and the rows it gave it."""
        index = len(self._uses)
        self._uses.append((test, entry))
        for info in infos:
            self._use_of[id(info)] = index
            self._infos.append(info)

    def collect(self, items: Iterable[pytest.Item]) -> None:
        """Group the selected items' rows by test and strategy, in item order."""
        tests: dict[str, list[int]] = {}
        rows: dict[int, dict[int, dict[str, Any]]] = {}
        for item in items:
            for info in item.stash.get(VECTORS_KEY, ()):
                use = self._use_of.get(id(info))
                if use is None:
                    continue
                test = self._uses[use][0]
                uses = tests.setdefault(test, [])
                if use not in uses:
                    uses.append(use)
                row = rows.setdefault(use, {}).setdefault(id(info), _row(info))
                row["items"].append(item.nodeid)
        self.collected = [
            {
                "nodeid": test,
                "strategies": [
                    {**self._uses[use][1], "rows": list(rows[use].values())} for use in uses
                ],
            }
            for test, uses in tests.items()
        ]

    def record(self, report: pytest.TestReport) -> None:
        """Note one report of an item."""
        self.outcomes[report.nodeid] = _outcome(self.outcomes.get(report.nodeid), report)

    def documents(self, seed: Any, nsamples: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the two documents: the definitions, and the same with the rows."""
        from . import __version__

        head = {
            "generator": {"name": "pytest-strategies", "version": __version__},
            "seed": encode(seed),
            "nsamples": nsamples,
        }
        definitions: list[dict[str, Any]] = []
        vectors: list[dict[str, Any]] = []
        for test in self.collected or []:
            defined, ran = [], []
            for strategy in test["strategies"]:
                defined.append({k: v for k, v in strategy.items() if k != "rows"})
                ran.append(
                    {
                        **strategy,
                        "rows": [
                            {
                                **row,
                                "items": [
                                    {"nodeid": nodeid, "outcome": self.outcomes.get(nodeid)}
                                    for nodeid in row["items"]
                                ],
                            }
                            for row in strategy["rows"]
                        ],
                    }
                )
            definitions.append({"nodeid": test["nodeid"], "strategies": defined})
            vectors.append({"nodeid": test["nodeid"], "strategies": ran})
        return (
            {"schema": SCHEMA, "kind": DEFINITIONS_KIND, **head, "tests": definitions},
            {"schema": SCHEMA, "kind": VECTORS_KIND, **head, "tests": vectors},
        )

    def write(self, seed: Any, nsamples: Any) -> None:
        """Write both files into the folder, each replaced in one step."""
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            for name, document in zip(
                (DEFINITIONS_FILE, VECTORS_FILE), self.documents(seed, nsamples), strict=True
            ):
                path = self.folder / name
                partial = path.with_name(f".{name}.{os.getpid()}.tmp")
                partial.write_text(
                    json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                os.replace(partial, path)
                self.written.append(path)
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"

    def worker_output(self) -> str:
        """The collected part, as a pytest-xdist worker sends it."""
        return json.dumps(self.collected or [])

    def take_worker_output(self, text: object) -> None:
        """Use the first worker's collected part (every worker collects every test)."""
        if self.collected is None and isinstance(text, str):
            try:
                collected = json.loads(text)
            except ValueError:
                return
            if isinstance(collected, list):
                self.collected = collected
