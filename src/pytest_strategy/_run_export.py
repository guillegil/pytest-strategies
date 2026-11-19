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
     "generator": {"name": "pytest-strategies", "version": "4.4.0"},
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

Both documents also say how to run their rows again with the same values, in
``rerun`` objects whose ``args`` are the unquoted arguments of a pytest command
run from the top-level ``rerun.cwd`` (4.4)::

    "rerun": {"cwd": "/home/me/project",
              "args": ["tests", "--nsamples=10", "--rng-seed=1"],
              "options": ["--rng-seed=1", "--nsamples=10"]}

At the top level ``args`` runs the whole run again (``_repro.run_args``: the
arguments the run was started with, with its seed) and ``options`` is what
follows node IDs chosen from the run (the seed and ``_repro.rerun_options``: the
generation options, with every constraint the run turned off, then the run's
other options, such as one of the project's own that shapes the context). Each
test has ``rerun.args`` for the rows the run selected (``_repro.rows_rerun``),
and each entry of a row's ``items`` those of the command the terminal prints for
a failed row, with the run's other options (``_repro.row_rerun``). A ``rerun``
has a ``note`` when its command is known not to reproduce the rows, and the
top-level one when the run's command selects tests by the cache
(``_repro.run_note``).

Only the tests the run selected are written (after ``-k``, ``--deselect`` and
the plugins' own deselection). An outcome is ``passed``, ``failed``,
``skipped``, ``xfailed``, ``xpassed``, ``error`` (setup or teardown failed) or
null when the item did not run (``--collect-only``, ``-x``). Values use the
encoding of ``_encode.py``, as ``export_strategies()``.

Under pytest-xdist every worker collects every test: the workers send the
collected part, with the commands of the tests and items and the options (made
from the arguments the controller was started with, which it sends them), the
controller adds the outcomes and the run's own command, and writes the files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _repro
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


def _rerun(rerun: _repro.Rerun) -> dict[str, Any]:
    """A command as the export writes it: its unquoted arguments, and its note."""
    entry: dict[str, Any] = {"args": list(rerun.args)}
    if rerun.note is not None:
        entry["note"] = rerun.note
    return entry


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
        # The number of items each test function was given (every parametrization
        # applied), by its node ID; without an entry it is not known
        self._calls: dict[str, int] = {}
        # The tests the run selected, with their rows: set when collection finishes
        # (or from a worker's output)
        self.collected: list[dict[str, Any]] | None = None
        # What follows node IDs in a command that reruns chosen items of the run,
        # set with ``collected``
        self.options: list[str] = []
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

    def add_calls(self, test: str, count: int) -> None:
        """Record how many items a test function was given."""
        self._calls[test] = count

    def collect(
        self,
        items: Iterable[pytest.Item],
        config: pytest.Config | None = None,
        seed: int | None = None,
        deselected: Iterable[pytest.Item] = (),
    ) -> None:
        """
        Group the selected items' rows by test and strategy, in item order, with
        the commands that run them again (``rerun``): each item's, as the terminal
        prints it for a failed row (``_repro.row_rerun``), and each test's, for the
        rows the run selected (``_repro.rows_rerun``).

        Args:
            items: The items the run selected
            config: The run's config, for the options of the run's commands
            seed: The run's seed; without it no command is recorded
            deselected: The items the run deselected, which decide whether a test
                function was selected as a whole when its number of items is not
                known
        """
        tests: dict[str, list[int]] = {}
        rows: dict[int, dict[int, dict[str, Any]]] = {}
        # Each test's selected items with their commands, and its rows' infos
        chosen: dict[str, list[tuple[pytest.Item, _repro.Rerun]]] = {}
        used: dict[str, dict[int, VectorInfo]] = {}
        for item in items:
            infos = item.stash.get(VECTORS_KEY, ())
            known = [info for info in infos if id(info) in self._use_of]
            if not known:
                continue
            entry: dict[str, Any] = {"nodeid": item.nodeid}
            if seed is not None:
                rerun = _repro.row_rerun(item, infos, carry=True)
                entry["rerun"] = _rerun(rerun)
                test = self._uses[self._use_of[id(known[0])]][0]
                chosen.setdefault(test, []).append((item, rerun))
            for info in known:
                use = self._use_of[id(info)]
                test = self._uses[use][0]
                uses = tests.setdefault(test, [])
                if use not in uses:
                    uses.append(use)
                used.setdefault(test, {})[id(info)] = info
                row = rows.setdefault(use, {}).setdefault(id(info), _row(info))
                row["items"].append(entry)
        partial = {self._test_of(item) for item in deselected}
        self.collected = []
        every: dict[int, VectorInfo] = {}
        for test, uses in tests.items():
            described: dict[str, Any] = {"nodeid": test}
            if seed is not None and test in chosen:
                count = self._calls.get(test)
                if count is not None:
                    whole = count == len(chosen[test])
                else:
                    seen = sum(1 for info in self._infos if id(info) in used[test])
                    total = sum(1 for info in self._infos if self._test_of_info(info) == test)
                    whole = test not in partial and seen == total
                described["rerun"] = _rerun(
                    _repro.rows_rerun(
                        [item for item, _ in chosen[test]],
                        [rerun for _, rerun in chosen[test]],
                        list(used[test].values()),
                        seed,
                        whole=whole,
                    )
                )
            described["strategies"] = [
                {**self._uses[use][1], "rows": list(rows[use].values())} for use in uses
            ]
            self.collected.append(described)
            every.update(used[test])
        self.options = []
        if seed is not None and config is not None:
            self.options = [
                f"--rng-seed={seed}",
                *_repro.rerun_options(config, list(every.values())),
            ]

    def _test_of_info(self, info: VectorInfo) -> str | None:
        use = self._use_of.get(id(info))
        return None if use is None else self._uses[use][0]

    def _test_of(self, item: pytest.Item) -> str | None:
        """The test function (its node ID) an item is a row of, or None."""
        for info in item.stash.get(VECTORS_KEY, ()):
            test = self._test_of_info(info)
            if test is not None:
                return test
        return None

    def record(self, report: pytest.TestReport) -> None:
        """Note one report of an item."""
        self.outcomes[report.nodeid] = _outcome(self.outcomes.get(report.nodeid), report)

    def documents(
        self, seed: Any, nsamples: Any, rerun: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Return the two documents: the definitions, and the same with the rows.

        Args:
            seed: The run's seed
            nsamples: The run's default number of random rows
            rerun: The folder pytest was started in (``cwd``) and the arguments of
                the command that runs the whole run again (``args``); the options
                of a command for chosen items are added to it
        """
        from . import __version__

        head: dict[str, Any] = {
            "generator": {"name": "pytest-strategies", "version": __version__},
            "seed": encode(seed),
            "nsamples": nsamples,
        }
        if rerun is not None:
            head["rerun"] = {
                **{key: value for key, value in rerun.items() if key != "note"},
                "options": list(self.options),
                **({"note": rerun["note"]} if "note" in rerun else {}),
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
                                    {
                                        "nodeid": item["nodeid"],
                                        "outcome": self.outcomes.get(item["nodeid"]),
                                        **{k: v for k, v in item.items() if k != "nodeid"},
                                    }
                                    for item in row["items"]
                                ],
                            }
                            for row in strategy["rows"]
                        ],
                    }
                )
            about = {k: v for k, v in test.items() if k != "strategies"}
            definitions.append({**about, "strategies": defined})
            vectors.append({**about, "strategies": ran})
        return (
            {"schema": SCHEMA, "kind": DEFINITIONS_KIND, **head, "tests": definitions},
            {"schema": SCHEMA, "kind": VECTORS_KIND, **head, "tests": vectors},
        )

    def write(self, seed: Any, nsamples: Any, rerun: Mapping[str, Any] | None = None) -> None:
        """Write both files into the folder, each replaced in one step."""
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            for name, document in zip(
                (DEFINITIONS_FILE, VECTORS_FILE), self.documents(seed, nsamples, rerun), strict=True
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
        """The collected part and the options, as a pytest-xdist worker sends them."""
        return json.dumps({"tests": self.collected or [], "options": self.options})

    def take_worker_output(self, text: object) -> None:
        """Use the first worker's collected part (every worker collects every test)."""
        if self.collected is None and isinstance(text, str):
            try:
                sent = json.loads(text)
            except ValueError:
                return
            if isinstance(sent, dict) and isinstance(sent.get("tests"), list):
                self.collected = sent["tests"]
                options = sent.get("options")
                if isinstance(options, list):
                    self.options = [str(option) for option in options]
