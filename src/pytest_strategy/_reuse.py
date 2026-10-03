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
one: its rows run with the values they failed with. A node ID names its file as
pytest does: from the rootdir, or for a file outside the rootdir (``-c
ci/pytest.ini`` makes ``ci`` the rootdir), from the path the run started from that
contains it. ``-k`` and ``-m`` are applied only once the tests are collected, after
the seed is chosen. The rows recorded under another seed are deselected, which
leaves them in pytest's last-failed set, and the terminal summary gives a ``pytest
--lf --rng-seed=S ...`` command that reruns only them. The options are not
applied: the header line that says the seed was reused names the recorded ones
that differ from the run's.
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
from ._repro import quote, start_args

# The cache key of the map
KEY = "pytest-strategies/failed-seeds"

# pytest's own cache keys: the last-failed set (--lf) and the stepwise state (--sw)
LASTFAILED = "cache/lastfailed"
STEPWISE = "cache/stepwise"

# The options of a rerun command that take their value as the next argument
_PAIRED = ("-o", "-c")

# The --strategy-constraint-off option of a rerun command, as one argument
_OFF = "--strategy-constraint-off="

# What gives failed rows a command of their own: their seed, the units of their
# options but --strategy-constraint-off, and whether they are outside the rootdir
_Group = tuple[int, tuple[tuple[str, ...], ...], bool]


@dataclass(frozen=True)
class Entry:
    """A failed row's entry in the map: the seed and the options it failed with."""

    seed: int
    options: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """The entry as the map stores it."""
        return {"seed": self.seed, "options": list(self.options)}


@dataclass(frozen=True)
class Outside:
    """
    Where a failed row in a file outside the rootdir is collected from: pytest names
    such a file from the path the run started from that contains it, so a command
    that reruns the row starts from that path too, and leaves out the other failed
    tests it collects.

    Attributes:
        start: That path, as a command run from the folder pytest was started in
            gives it
        outside: The node IDs of the failed tests in files outside the rootdir
            that it collects
        inside: The node IDs of those in files inside the rootdir
    """

    start: str
    outside: frozenset[str] = frozenset()
    inside: frozenset[str] = frozenset()


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
        outside: Where the rows of ``rows`` and ``others`` that are in a file
            outside the rootdir are collected from, by node ID
    """

    flag: str
    seed: int
    rows: dict[str, Entry]
    others: dict[str, Entry]
    failed: frozenset[str] = field(default_factory=frozenset)
    outside: dict[str, Outside] = field(default_factory=dict)


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


@dataclass(frozen=True)
class _Start:
    """
    A path or node ID the run starts from (``config.args``).

    Attributes:
        path: Its path, absolute
        names: Its test names (``test_w`` of ``tests/test_w.py::test_w``), or ""
        arg: The argument, as a command run from the folder pytest was started
            in gives it
    """

    path: Path
    names: str
    arg: str


def _starts(config: pytest.Config) -> list[_Start] | None:
    """
    Return the paths and node IDs the run starts from (``config.args``: those on
    the command line, or else the testpaths or the folder pytest was started in).
    None when one of them is not a path (``--pyargs``).
    """
    args = start_args(config, config.invocation_params.dir)
    if args is None:
        return None
    starts = []
    for given, arg in zip(config.args, args):
        text, _, names = given.partition("::")
        path = Path(os.path.abspath(config.invocation_params.dir / text))
        starts.append(_Start(path, names, arg))
    return starts


def _below(path: Path, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or below it."""
    path_text, folder_text = os.path.normcase(path), os.path.normcase(folder)
    return path_text == folder_text or path_text.startswith(folder_text.rstrip(os.sep) + os.sep)


def _names_select(names: str, own: str) -> bool:
    """Whether a start's test names select those of a node ID (``test_w[rand-1]``)."""
    return not names or own == names or own.startswith((names + "::", names + "["))


def _collecting(config: pytest.Config, starts: Sequence[_Start], nodeid: str) -> dict[int, bool]:
    """
    Return the starts that collect the file of a node ID, when it exists, by their
    index in ``starts``, with True for the one pytest names the file from. pytest
    names a file in the rootdir by its path from the rootdir, and a file outside it
    by its path from the start that contains it (``tests/test_dma.py`` from ``.``
    under ``-c ci/pytest.ini``), or by nothing when it is a start itself
    (``::test_write[rand-3]``).
    """
    text, _, own = nodeid.partition("::")
    rootpath = Path(os.path.abspath(config.rootpath))
    found: dict[int, bool] = {}
    if text:
        file = Path(os.path.abspath(rootpath / text))
        if file.exists():
            for index, start in enumerate(starts):
                if _below(file, start.path) and _names_select(start.names, own):
                    found[index] = False
    for index, start in enumerate(starts):
        file = Path(os.path.abspath(start.path / text)) if text else start.path
        if _below(file, rootpath) or not _names_select(start.names, own):
            continue
        if file.exists():
            found[index] = True
    return found


def _outside(
    config: pytest.Config, starts: Sequence[_Start], index: int, failed: Iterable[str]
) -> Outside:
    """Return the start ``starts[index]`` with the failed tests it collects."""
    outside: set[str] = set()
    inside: set[str] = set()
    for nodeid in failed:
        names = _collecting(config, starts, nodeid).get(index)
        if names is not None:
            (outside if names else inside).add(nodeid)
    return Outside(starts[index].arg, frozenset(outside), frozenset(inside))


def plan(config: pytest.Config) -> Reuse | None:
    """
    Return what a run with ``--lf`` or ``--sw`` and no seed of its own reuses: the
    seed of the newest entry of the map among the failed tests it reruns whose file
    still exists and that the run collects (``_collecting``), the rows recorded
    under it, and those recorded under other seeds. None without such an entry, or
    without the option or the cache.
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
    starts = _starts(config)
    kept: dict[str, Entry] = {}
    outside: dict[str, Outside] = {}
    collected: dict[int, Outside] = {}
    for nodeid, entry in read(cache).items():
        if nodeid not in failed:
            continue
        if starts is None:
            # A --pyargs run: whatever it collects, the file must exist
            if _exists(config.rootpath, nodeid):
                kept[nodeid] = entry
            continue
        found = _collecting(config, starts, nodeid)
        if not found:
            continue
        kept[nodeid] = entry
        named = [index for index, names in found.items() if names]
        if named:
            index = named[0]
            if index not in collected:
                collected[index] = _outside(config, starts, index, failed)
            outside[nodeid] = collected[index]
    if not kept:
        return None
    seed = list(kept.values())[-1].seed
    return Reuse(
        flag=option,
        seed=seed,
        rows={nodeid: entry for nodeid, entry in kept.items() if entry.seed == seed},
        others={nodeid: entry for nodeid, entry in kept.items() if entry.seed != seed},
        failed=frozenset(failed),
        outside=outside,
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


def _outside_targets(
    nodeids: Sequence[str], outside: Mapping[str, Outside], where: Callable[[str], str]
) -> list[str]:
    """
    Return what selects the rows ``nodeids`` of files outside the rootdir on a
    command line: the paths the run started from that pytest named their files
    from, so that the rows keep their node IDs, and what leaves out the other
    failed tests those paths collect: ``--ignore`` for each file inside the rootdir
    that holds one, as pytest's ``--lf`` collects no file outside the rootdir once
    it collected such a file (it finds the failed tests' files from the rootdir),
    and ``--deselect`` for each one outside it. A failed test whose node ID begins
    one of the rows' (``tests/test_w.py::test_w`` begins
    ``tests/test_w.py::test_w[rand-1]``) is not deselected: ``--deselect`` matches
    the start of a node ID, and would leave the row out too. All quoted.
    """
    mine = set(nodeids)
    starts: dict[str, None] = {}
    ignore: dict[str, None] = {}
    deselect: set[str] = set()
    for nodeid in nodeids:
        found = outside[nodeid]
        starts[found.start] = None
        ignore.update(dict.fromkeys(sorted(n.partition("::")[0] for n in found.inside)))
        deselect.update(other for other in found.outside if other not in mine)
    return [
        *(quote(start) for start in starts),
        *(f"--ignore {quote(where(file))}" for file in ignore),
        *(
            f"--deselect {quote(other)}"
            for other in sorted(deselect)
            if not any(row.startswith(other) for row in mine)
        ),
    ]


def commands(
    flag: str,
    rows: Iterable[tuple[str, Entry]],
    failed: Collection[str] = (),
    where: Callable[[str], str] = str,
    outside: Mapping[str, Outside] | None = None,
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
    leave pytest's last-failed set, and its failure would be lost. Rows in files
    outside the rootdir get a command of their own, which names neither, as either
    would give them other node IDs: it starts from the paths the run started from
    that pytest named them from, and leaves out the other failed tests those paths
    collect (``_outside_targets``).

    Args:
        flag: The option that reruns failed tests, ``--lf``, ``--sw`` or ``--sw-skip``
        rows: The rows' node IDs (relative to the rootdir) and entries, in the
            order of the map
        failed: The node IDs of the failed tests the run reruns (``Reuse.failed``)
        where: Write a node ID or a file relative to the rootdir as the command
            gives it (relative to the folder pytest was started in)
        outside: Where the rows in files outside the rootdir are collected from
            (``Reuse.outside``)
    """
    outside = outside or {}
    groups: dict[_Group, dict[tuple[str | None, str], None]] = {}
    nodeids: dict[_Group, list[str]] = {}
    for nodeid, row in rows:
        row_units, row_off = _split(row.options)
        key = (row.seed, tuple(row_units), nodeid in outside)
        groups.setdefault(key, {}).update(dict.fromkeys(row_off))
        nodeids.setdefault(key, []).append(nodeid)
    found = []
    for key, off in groups.items():
        seed, units, beyond = key
        options = list(units) + ([_off_unit(off)] if off else [])
        group = nodeids[key]
        if beyond:
            targets = _outside_targets(group, outside, where)
        else:
            targets = [quote(where(target)) for target in _targets(group, failed)]
        parts = ["pytest", flag, f"--rng-seed={seed}", _text(options), *targets]
        found.append((" ".join(part for part in parts if part), len(group)))
    return found
