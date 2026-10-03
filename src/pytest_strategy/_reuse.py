"""
Rerunning failed strategy rows with the values they failed with: ``--lf`` and
``--sw`` reuse the seed of the run that recorded the failures.

Every run with pytest's cache keeps, under the cache key
``pytest-strategies/failed-seeds``, the seed and the options of each strategy row
whose setup or call failed, by node ID, newest last::

    {"tests/test_dma.py::test_write[rand-3]": {"seed": 21, "options": ["--nsamples=13"]}}

The options are those of the row's rerun command after ``--rng-seed``
(``_repro.generation_options``), unquoted. The process that reports the run (the
pytest-xdist controller, not a worker) writes the map when the session finishes:
it records the rows that failed, as the newest, and removes an entry only when its
row passed under its seed and its options.

With ``--lf`` or ``--sw`` (``--sw-skip`` too) and no ``--rng-seed``, the run reads
pytest's own last-failed set (``cache/lastfailed``) or the test ``--sw`` resumes
from (``cache/stepwise``), keeps the entries of those node IDs whose file still
exists and that the run collects (the paths and node IDs on the command line, or
else the testpaths or the folder pytest was started in), and seeds from the newest
one: its rows run with the values they failed with. ``-k`` and ``-m`` are applied
only once the tests are collected, after the seed is chosen. The rows recorded
under another seed are deselected, which leaves them in pytest's last-failed set,
and the terminal summary gives a ``pytest --lf --rng-seed=S ...`` command that
reruns only them. The options are not applied: the header line that says the seed
was reused names the recorded ones that differ from the run's.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ._options import constraint_off_item, parse_constraint_off
from ._repro import quote

# The cache key of the map
KEY = "pytest-strategies/failed-seeds"

# pytest's own cache keys: the last-failed set (--lf) and the stepwise state (--sw)
LASTFAILED = "cache/lastfailed"
STEPWISE = "cache/stepwise"

# The options of a rerun command that take their value as the next argument
_PAIRED = ("-o", "-c")

# The --strategy-constraint-off option of a rerun command, as one argument
_OFF = "--strategy-constraint-off="


@dataclass(frozen=True)
class Entry:
    """A failed row's entry in the map: the seed and the options it failed with."""

    seed: int
    options: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """The entry as the map stores it."""
        return {"seed": self.seed, "options": list(self.options)}


@dataclass(frozen=True)
class Reuse:
    """
    What a ``--lf`` or ``--sw`` run reuses.

    Attributes:
        flag: The option that asked for it, ``--lf``, ``--sw`` or ``--sw-skip``
        seed: The seed of the newest failed row, the run's seed
        rows: The failed rows recorded under that seed, by node ID
        others: The failed rows recorded under other seeds, which the run deselects
        failed: The node IDs of the failed tests the run reruns: pytest's
            last-failed set, or the test ``--sw`` resumes from
    """

    flag: str
    seed: int
    rows: dict[str, Entry]
    others: dict[str, Entry]
    failed: frozenset[str] = field(default_factory=frozenset)


def _entry(value: object) -> Entry | None:
    """Read an entry of the map, or None for one of another shape."""
    if not isinstance(value, dict):
        return None
    seed, options = value.get("seed"), value.get("options")
    if type(seed) is not int or not isinstance(options, list):
        return None
    if not all(isinstance(option, str) for option in options):
        return None
    return Entry(seed, tuple(options))


def read(cache: pytest.Cache) -> dict[str, Entry]:
    """Return the map in ``cache``, newest last, leaving out entries of another shape."""
    stored = cache.get(KEY, None)
    entries: dict[str, Entry] = {}
    if isinstance(stored, dict):
        for nodeid, value in stored.items():
            entry = _entry(value)
            if isinstance(nodeid, str) and entry is not None:
                entries[nodeid] = entry
    return entries


def write(cache: pytest.Cache, entries: Mapping[str, Entry]) -> None:
    """Store the map in ``cache``."""
    cache.set(KEY, {nodeid: entry.to_json() for nodeid, entry in entries.items()})


def from_row(row: Mapping[str, str]) -> Entry | None:
    """
    Return the entry of a failed row from its report's ``pytest_strategies``
    attribute (``_repro.failure``), or None when it has no usable seed or options.
    """
    try:
        seed = int(row["seed"])
        options = json.loads(row["options"])
    except (KeyError, TypeError, ValueError):
        return None
    return _entry({"seed": seed, "options": options})


def updated(
    entries: Mapping[str, Entry],
    seed: int,
    passed: Mapping[str, Sequence[str]],
    failed: Mapping[str, Mapping[str, str]],
) -> dict[str, Entry]:
    """
    Return the map after a run: without the entries whose rows passed under their
    seed and options, and with the rows that failed, as the newest, in the order
    they failed.

    Args:
        entries: The map, newest last
        seed: The run's seed
        passed: The options of the rows whose call passed in the run, by node ID
        failed: The ``pytest_strategies`` attributes of the rows whose setup or
            call failed, by node ID, in the order they failed
    """
    result = dict(entries)
    for nodeid, options in passed.items():
        entry = result.get(nodeid)
        if entry is not None and entry.seed == seed and entry.options == tuple(options):
            del result[nodeid]
    for nodeid, row in failed.items():
        new = from_row(row)
        if new is not None:
            result.pop(nodeid, None)
            result[nodeid] = new
    return result


def flag(config: pytest.Config) -> str | None:
    """
    Return the option that reruns failed tests in this run, ``--lf``, ``--sw`` or
    ``--sw-skip``, or None (also for ``--sw-reset``, which starts ``--sw`` over).
    ``--sw-skip`` and ``--sw-reset`` turn ``--sw`` on in pytest's own
    ``pytest_configure``, which may not have run yet.
    """
    option = config.option
    if getattr(option, "lf", False):
        return "--lf"
    if getattr(option, "stepwise_reset", False):
        return None
    if getattr(option, "stepwise_skip", False):
        return "--sw-skip"
    if getattr(option, "stepwise", False):
        return "--sw"
    return None


def cache_of(config: pytest.Config) -> pytest.Cache | None:
    """
    Return the run's cache: ``config.cache``, or before pytest's cache plugin set it
    (its ``pytest_configure`` runs after the plugin's first one) the same cache read
    from its folder. None without the cache plugin (``-p no:cacheprovider``), and
    under ``--cache-clear``, which leaves nothing to read.
    """
    cache: pytest.Cache | None = getattr(config, "cache", None)
    if cache is not None:
        return cache
    if not config.pluginmanager.has_plugin("cacheprovider"):
        return None
    if getattr(config.option, "cacheclear", False):
        return None
    try:
        from _pytest.cacheprovider import Cache

        return Cache.for_config(config, _ispytest=True)
    except (ImportError, AttributeError, TypeError):
        # A pytest that moved it: no seed is reused
        return None


def _failed_ids(cache: pytest.Cache, flag: str) -> set[str]:
    """
    Return the node IDs the run reruns: pytest's last-failed set for ``--lf``, the
    test ``--sw`` resumes from (a dict with ``last_failed``) otherwise.
    """
    if flag == "--lf":
        lastfailed = cache.get(LASTFAILED, {})
        return (
            {nodeid for nodeid in lastfailed if isinstance(nodeid, str)}
            if isinstance(lastfailed, dict)
            else set()
        )
    stepwise = cache.get(STEPWISE, None)
    last = stepwise.get("last_failed") if isinstance(stepwise, dict) else stepwise
    return {last} if isinstance(last, str) and last else set()


def _exists(rootpath: Path, nodeid: str) -> bool:
    """Whether the file of a node ID (relative to the rootdir) still exists."""
    return (rootpath / nodeid.partition("::")[0]).exists()


def _selection(config: pytest.Config) -> Callable[[str], bool] | None:
    """
    Return a test of whether the run collects a node ID: whether one of the paths
    and node IDs it starts from (``config.args``: those on the command line, or
    else the testpaths or the folder pytest was started in) selects it. None when
    one of them is not a path (``--pyargs``).
    """
    targets: list[tuple[str, str]] = []
    for arg in config.args:
        text, _, names = arg.partition("::")
        path = os.path.abspath(config.invocation_params.dir / text)
        if not os.path.exists(path):
            return None
        targets.append((os.path.normcase(path), names))
    rootpath = config.rootpath

    def selects(nodeid: str) -> bool:
        text, _, own = nodeid.partition("::")
        file = os.path.normcase(os.path.abspath(rootpath / text))
        for path, names in targets:
            if file != path and not file.startswith(path.rstrip(os.sep) + os.sep):
                continue
            if not names or own == names or own.startswith((names + "::", names + "[")):
                return True
        return False

    return selects


def plan(config: pytest.Config) -> Reuse | None:
    """
    Return what a run with ``--lf`` or ``--sw`` and no seed of its own reuses: the
    seed of the newest entry of the map among the failed tests it reruns whose file
    still exists and that the command line selects, the rows recorded under it, and
    those recorded under other seeds. None without such an entry, or without the
    option or the cache.
    """
    option = flag(config)
    if option is None:
        return None
    cache = cache_of(config)
    if cache is None:
        return None
    failed = _failed_ids(cache, option)
    if not failed:
        return None
    selects = _selection(config)
    kept = {
        nodeid: entry
        for nodeid, entry in read(cache).items()
        if nodeid in failed
        and _exists(config.rootpath, nodeid)
        and (selects is None or selects(nodeid))
    }
    if not kept:
        return None
    seed = list(kept.values())[-1].seed
    return Reuse(
        flag=option,
        seed=seed,
        rows={nodeid: entry for nodeid, entry in kept.items() if entry.seed == seed},
        others={nodeid: entry for nodeid, entry in kept.items() if entry.seed != seed},
        failed=frozenset(failed),
    )


def _units(options: Sequence[str]) -> list[tuple[str, ...]]:
    """Group a command's options into units: ``-o``/``-c`` with their values, each other alone."""
    units: list[tuple[str, ...]] = []
    position = 0
    while position < len(options):
        if options[position] in _PAIRED and position + 1 < len(options):
            units.append((options[position], options[position + 1]))
            position += 2
        else:
            units.append((options[position],))
            position += 1
    return units


def _split(options: Sequence[str]) -> tuple[list[tuple[str, ...]], list[tuple[str | None, str]]]:
    """
    Split a command's options into its units without ``--strategy-constraint-off``
    and that option's ``(strategy, name)`` items.
    """
    units: list[tuple[str, ...]] = []
    off: list[tuple[str | None, str]] = []
    for unit in _units(options):
        if len(unit) == 1 and unit[0].startswith(_OFF):
            try:
                off.extend(parse_constraint_off(unit[0][len(_OFF) :]))
                continue
            except ValueError:
                pass
        units.append(unit)
    return units, off


def _text(units: Iterable[tuple[str, ...]]) -> str:
    """Write units as on a command line, quoted for the platform's shell."""
    return " ".join(quote(arg) for unit in units for arg in unit)


def _off_unit(items: Iterable[tuple[str | None, str]]) -> tuple[str, ...]:
    """Write ``(strategy, name)`` items as one ``--strategy-constraint-off`` unit."""
    return (_OFF + ",".join(constraint_off_item(*item) for item in items),)


def differences(
    rows: Sequence[Entry], current: Sequence[str], current_off: Sequence[tuple[str | None, str]]
) -> str | None:
    """
    Name the recorded options that differ from the run's: ``with --nsamples=13``
    for those the newest row was recorded with and the run does not have, ``without
    --vector-mode=test`` for those the run has and it was not recorded with, or None
    when they agree.

    A constraint the rows had turned off counts as off in the run when the run
    turns it off in their strategy or everywhere. The run's own items are not
    compared: which strategies a bare name reaches is known only once the tests
    are collected.

    Args:
        rows: The reused rows' entries, newest last
        current: The run's options, as ``_repro.generation_options`` writes them
            without the constraints
        current_off: The run's ``--strategy-constraint-off`` items
    """
    recorded, _ = _split(rows[-1].options)
    off = dict.fromkeys(item for row in rows for item in _split(row.options)[1])
    units, _ = _split(current)
    missing = [
        (strategy, name)
        for strategy, name in off
        if (strategy, name) not in current_off and (None, name) not in current_off
    ]
    extra = [unit for unit in recorded if unit not in units]
    if missing:
        extra.append(_off_unit(missing))
    absent = [unit for unit in units if unit not in recorded]
    parts = []
    if extra:
        parts.append(f"with {_text(extra)}")
    if absent:
        parts.append(f"without {_text(absent)}")
    return ", ".join(parts) or None


def _targets(nodeids: Sequence[str], failed: Collection[str]) -> list[str]:
    """
    Return what selects the rows ``nodeids`` on a command line, in their order: the
    file of the rows whose file holds no other failed test (of ``failed``), and the
    node ID of each of the others.
    """
    mine = set(nodeids)
    shared = {nodeid.partition("::")[0] for nodeid in failed if nodeid not in mine}
    targets: dict[str, None] = {}
    for nodeid in nodeids:
        file = nodeid.partition("::")[0]
        targets[nodeid if file in shared else file] = None
    return list(targets)


def commands(
    flag: str,
    rows: Iterable[tuple[str, Entry]],
    failed: Collection[str] = (),
    where: Callable[[str], str] = str,
) -> list[tuple[str, int]]:
    """
    Return the commands that rerun failed rows recorded under other seeds, each
    with the number of rows it covers: ``pytest --lf --rng-seed=S``, the rows'
    options, and the files or node IDs that select the rows, one command per seed
    (more when rows of one seed were recorded with other options), in the order of
    the map. The constraints the rows turned off are joined into one
    ``--strategy-constraint-off``; its items name their strategy, so each row gets
    its own.

    A command names a row's file when every failed test of that file is one of its
    rows, and the row's node ID otherwise, so it reruns no failed test under a seed
    that test was not recorded under: one that passed with those other values would
    leave pytest's last-failed set, and its failure would be lost.

    Args:
        flag: The option that reruns failed tests, ``--lf``, ``--sw`` or ``--sw-skip``
        rows: The rows' node IDs (relative to the rootdir) and entries, in the
            order of the map
        failed: The node IDs of the failed tests the run reruns (``Reuse.failed``)
        where: Write a node ID or a file relative to the rootdir as the command
            gives it (relative to the folder pytest was started in)
    """
    groups: dict[tuple[int, tuple[tuple[str, ...], ...]], dict[tuple[str | None, str], None]] = {}
    nodeids: dict[tuple[int, tuple[tuple[str, ...], ...]], list[str]] = {}
    for nodeid, row in rows:
        row_units, row_off = _split(row.options)
        key = (row.seed, tuple(row_units))
        groups.setdefault(key, {}).update(dict.fromkeys(row_off))
        nodeids.setdefault(key, []).append(nodeid)
    found = []
    for (seed, units), off in groups.items():
        options = list(units) + ([_off_unit(off)] if off else [])
        targets = [quote(where(target)) for target in _targets(nodeids[(seed, units)], failed)]
        parts = ["pytest", flag, f"--rng-seed={seed}", _text(options), *targets]
        found.append((" ".join(part for part in parts if part), len(nodeids[(seed, units)])))
    return found
