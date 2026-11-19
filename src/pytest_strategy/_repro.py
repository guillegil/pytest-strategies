"""
What a failed strategy row reports: the ``pytest-strategies`` section under its
traceback, and the command that runs it again with the same values.

The section shows each strategy's row (its strategy and where the factory is,
its kind, name and index, every value, the seed and, when the factory received
one, the context's fingerprint) and the rerun command::

    strategy  burst (tests/dma/strategies.py:12)
    vector    rand-3 (random row 3)
    values    addr=4096
              len=17
    seed      1763926297314361000
    context   3f2a9c1e
    rerun     pytest 'tests/dma/test_write.py::test_write[rand-3]' --rng-seed=1763926297314361000

The rerun command is ``pytest <node id> --rng-seed=S``, with the node ID relative
to the folder pytest was started in, followed by every option that decides which
rows exist and what they hold: the generation options the run had at a value of
its own (``--nsamples``, ``--vector-mode``, ``--vector-name``, ``--vector-index``),
its ``-o strategies_*`` overrides, its ``-c`` and ``--rootdir`` (node IDs and the
random streams are relative to the rootdir), and the constraints turned off in
the row's strategies, as ``STRATEGY:NAME`` items, so a rerun never names a
constraint that only another module's strategy has. Its ``-o
empty_parameter_set_mark`` override comes too: the rerun collects the row's whole
module, where under ``--vector-name`` a strategy without that vector gets an
empty parameter set, which ``fail_at_collect`` turns into a collection error.
Options that shape the context cannot be known; its fingerprint shows when the
rerun got another one. The run export's commands, which are read without the
run's own command at hand, therefore repeat the run's other options too
(``carried_options``).

A test file outside the rootdir (``-c ci/pytest.ini`` makes ``ci`` the rootdir) is
named by pytest relative to the path the run started from that contains it, so a
run of its node ID would give the row another node ID, and with it other random
values. Its command starts from the run's own paths instead, leaves out what the
run's ``--ignore`` and ``--ignore-glob`` left out, and selects the row with ``-k``:
``pytest . --rng-seed=S -c ci/pytest.ini -k 'test_write[rand-3]'``. When no ``-k``
expression selects only that row, the node ID command is given, and the section
says that it does not reproduce the row (``note``).

The section and the list of failed rows are written so that the terminal can
encode them (``encodable``): pytest writes a text that holds a character it
cannot encode escaped as a whole, which would put a section on one line and
double the backslashes of the escapes pytest puts in a node ID.

A ``--junitxml`` report gets the section in the failure text, the seed and each
failed row's command as properties of the test suite (``suite_properties``), and,
with the ``xunit1`` and ``legacy`` families, whose schema has properties per test
case, what the row is as properties of its test case (``testcase_properties``),
with a command run from the rootdir.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.pathlib import bestrelpath

from ._ids import _value_repr
from ._options import constraint_off_item
from ._runtime import runtime

if TYPE_CHECKING:
    from ._vector import VectorInfo

# The title of the section
SECTION = "pytest-strategies"

# The prefix of the JUnit XML properties
PROPERTY = "pytest_strategies"

# The characters a value's repr is cut at in the section, below -vv
VALUE_LIMIT = 4000

# The width of the section's labels ("strategy  ")
_LABEL = 10

# A name in a -k expression (pytest's match-expression grammar), and the words that
# are its operators
_KEYWORD_NAME = re.compile(r"[\w:+\-.\[\]\\/]+")
_KEYWORD_OPERATORS = frozenset({"and", "or", "not"})

# pytest's ini option that decides what an empty parameter set gives, such as the one
# --vector-name gives a strategy without that vector: a rerun command carries its
# override, which decides no row's values (_reuse compares the others)
EMPTY_PARAMETER_SET_MARK = "empty_parameter_set_mark"

# The key of the arguments the run was started with, in a pytest-xdist worker's input
INVOCATION = "pytest_strategies_invocation"

# The options a command for chosen rows gives itself (--rng-seed and
# ``generation_options``), --strategy-export, which would write over the run's
# export, and --deselect, which selects tests (as -k and -m, see ``_carried``): they
# are not taken over from the run's arguments (``carried_options``)
_OWN_OPTIONS = frozenset(
    {
        "--deselect",
        "--rng-seed",
        "--nsamples",
        "--vector-mode",
        "--vector-name",
        "--vector-index",
        "--strategy-constraint-off",
        "--strategy-export",
        "--rootdir",
        "--config-file",
    }
)

# The options that select tests by what the cache holds when the command runs, and
# those that keep the tests from running: a command for chosen rows names its rows
# and runs them, so it leaves them out
CACHE_OPTIONS = frozenset({"--lf", "--last-failed", "--sw", "--stepwise"})
_LEFT_OUT_FLAGS = CACHE_OPTIONS | {
    "--sw-skip",
    "--stepwise-skip",
    "--sw-reset",
    "--stepwise-reset",
    "--cache-clear",
    "--co",
    "--collect-only",
    "--list-strategies",
}

# The length of the pieces of the tests' names that ``_Matchers`` indexes
_GRAM = 6

# The arguments cmd.exe and PowerShell leave as they are (no ",", which PowerShell
# reads as an array, no "%", "@", "&", quotes or brackets)
_WINDOWS_SAFE = re.compile(r"[A-Za-z0-9_+=:./\\-]+")


class SectionedRepr(TerminalRepr):
    """
    A failure's report that takes no sections, such as a missing fixture's
    (``FixtureLookupErrorRepr``), followed by sections, which pytest prints as it
    prints those of a traceback (``ExceptionRepr.addsection``). Its other
    attributes are the report's.
    """

    def __init__(self, inner: TerminalRepr) -> None:
        self.inner = inner
        self.sections: list[tuple[str, str, str]] = []

    def addsection(self, name: str, content: str, sep: str = "-") -> None:
        """Add a section below the report."""
        self.sections.append((name, content, sep))

    def toterminal(self, tw: TerminalWriter) -> None:
        """Write the report, then its sections."""
        self.inner.toterminal(tw)
        for name, content, sep in self.sections:
            tw.sep(sep, name)
            tw.line(content)

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            # Not set yet (a copy being made)
            raise AttributeError(name)
        return getattr(self.inner, name)


def quote(arg: str, *, windows: bool | None = None) -> str:
    """
    Quote a command-line argument for the shell: ``shlex.quote`` on POSIX, double
    quotes on Windows, where the C runtime's rules apply (a ``"`` in the argument
    becomes ``\\"``, and the backslashes before it are doubled). An argument that
    needs no quotes is left as it is.

    Args:
        windows: Quote for Windows; the platform's rule when None
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return shlex.quote(arg)
    if _WINDOWS_SAFE.fullmatch(arg):
        return arg
    quoted = []
    backslashes = 0
    for char in arg:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            quoted.append("\\" * (2 * backslashes + 1) + '"')
        else:
            quoted.append("\\" * backslashes + char)
        backslashes = 0
    # Backslashes before the closing quote are doubled, so it stays a quote
    quoted.append("\\" * (2 * backslashes))
    return '"' + "".join(quoted) + '"'


def encodable(text: str, encoding: str | None) -> str:
    """
    Return ``text`` with each character that ``encoding`` cannot encode written as
    a backslash escape (``\\u03a9``), as pytest writes the non-ASCII characters of a
    test ID, or ``text`` itself when there is none or the encoding is not known.

    pytest's TerminalWriter writes a text that holds such a character escaped as a
    whole (``unicode-escape``): its line breaks as ``\\n``, and each backslash
    doubled, those of the escapes in a node ID too, so a section would print as one
    line and its rerun command would name another test.
    """
    if not encoding:
        return text
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        return text.encode(encoding, "backslashreplace").decode(encoding)
    except LookupError:
        # An encoding Python does not know
        return text
    return text


def terminal_encoding(reporter: object) -> str | None:
    """
    Return the encoding of the stream pytest's terminal reporter ``reporter``
    writes to (its TerminalWriter's private ``_file``, by default ``sys.stdout``),
    or None without the terminal reporter (``-p no:terminal``) or a known encoding.
    """
    if reporter is None:
        return None
    stream = getattr(getattr(reporter, "_tw", None), "_file", sys.stdout)
    encoding = getattr(stream, "encoding", None)
    return encoding if isinstance(encoding, str) else None


def path_from(folder: Path, path: Path) -> str:
    """
    Write ``path`` relative to ``folder`` when it is ``folder`` or below it, and
    absolute otherwise: a file elsewhere (``-c /dev/null``) would be named by a
    chain of ``..`` that depends on how deep ``folder`` is.
    """
    try:
        path.relative_to(folder)
    except ValueError:
        return str(path)
    return bestrelpath(folder, path)


def generation_options(
    config: pytest.Config, infos: Sequence[VectorInfo], *, start: Path | None = None
) -> list[str]:
    """
    Return the options of a row's rerun command after ``--rng-seed``, unquoted:
    ``--nsamples``, ``--vector-mode``, ``--vector-name`` and ``--vector-index`` when
    the run gave them (read from the run's options, so a value from the ini file's
    addopts or a conftest counts too), each ``-o strategies_*`` override and the
    ``-o empty_parameter_set_mark`` override (the last of each name, as pytest
    applies it), ``-c`` and ``--rootdir`` as given, and the constraints the row's
    strategies turned off (``--strategy-constraint-off=STRATEGY:NAME,...``).

    Args:
        config: The run's config
        infos: The row's VectorInfo for each of the item's strategies
        start: The folder the command runs from, when it is not the one pytest was
            started in: ``-c`` and ``--rootdir`` are then relative to it, but an
            ini file that is not below it is absolute (``path_from``)
    """
    base = runtime.session_options(config).base
    args = []
    if base.nsamples_source == "--nsamples":
        args.append(f"--nsamples={base.nsamples}")
    if base.mode != "all":
        args.append(f"--vector-mode={base.mode}")
    if base.vector_name is not None:
        args.append(f"--vector-name={base.vector_name}")
    if base.vector_index is not None:
        args.append(f"--vector-index={base.vector_index}")
    # The last value of each strategies_* ini option that -o overrides (pytest's rule),
    # and of empty_parameter_set_mark: a node ID's rerun collects the whole module,
    # where a strategy without the selected vectors gets an empty parameter set
    overrides: dict[str, str] = {}
    for override in getattr(config.option, "override_ini", None) or ():
        name = override.partition("=")[0].strip()
        if name.startswith("strategies_") or name == EMPTY_PARAMETER_SET_MARK:
            overrides.pop(name, None)
            overrides[name] = override
    for override in overrides.values():
        args += ["-o", override]
    inifile = getattr(config.option, "inifilename", None)
    if inifile:
        if start is not None and config.inipath is not None:
            # The path pytest resolved from the folder it was started in
            inifile = path_from(start, config.inipath)
        args += ["-c", str(inifile)]
    rootdir = getattr(config.option, "rootdir", None)
    if rootdir:
        if start is not None:
            rootdir = bestrelpath(start, config.rootpath)
        args.append(f"--rootdir={rootdir}")
    items = dict.fromkeys(
        constraint_off_item(info.strategy, name) for info in infos for name in info.constraints_off
    )
    if items:
        args.append(f"--strategy-constraint-off={','.join(items)}")
    return args


def outside_rootdir(item: pytest.Item) -> bool:
    """
    Whether an item's file is outside the rootdir, where pytest names it relative to
    the path on the command line that contains it instead (pytest's own test of
    ``path.relative_to(rootpath)``).
    """
    try:
        item.path.relative_to(item.config.rootpath)
    except ValueError:
        return True
    return False


def cwd_relative_nodeid(config: pytest.Config, nodeid: str) -> str:
    """
    Return a node ID (relative to the rootdir) relative to the folder pytest was
    started in, as pytest's ``Config.cwd_relative_nodeid`` did for a string before
    pytest 9.2, which takes a structured ``NodeId`` instead.
    """
    if config.invocation_params.dir == config.rootpath:
        return nodeid
    path, sep, rest = nodeid.partition("::")
    relative = bestrelpath(config.invocation_params.dir, config.rootpath / path)
    return f"{relative}{sep}{rest}"


def rerun_nodeid(item: pytest.Item) -> str:
    """
    Return an item's node ID relative to the folder pytest was started in
    (``cwd_relative_nodeid``). For a file outside the rootdir, whose node ID
    is not relative to the rootdir, the file's own path is used instead.
    """
    config = item.config
    if not outside_rootdir(item):
        return cwd_relative_nodeid(config, item.nodeid)
    names = item.nodeid.partition("::")[2]
    return f"{bestrelpath(config.invocation_params.dir, item.path)}::{names}"


def rerun_function_nodeid(item: pytest.Item) -> str:
    """
    Return the node ID of an item's test function (the item's without its
    parameters) relative to the folder pytest was started in, as ``rerun_nodeid``.
    """
    nodeid = rerun_nodeid(item)
    name = getattr(item, "originalname", None)
    if isinstance(name, str) and name != item.name and nodeid.endswith(item.name):
        return nodeid[: len(nodeid) - len(item.name)] + name
    return nodeid


def rootdir_nodeid(item: pytest.Item) -> str:
    """
    Return an item's node ID relative to the rootdir: its node ID, or for a file
    outside the rootdir, the file's path from the rootdir.
    """
    if not outside_rootdir(item):
        return item.nodeid
    names = item.nodeid.partition("::")[2]
    return f"{bestrelpath(item.config.rootpath, item.path)}::{names}"


def command_line(args: Sequence[str], *, windows: bool | None = None) -> str:
    """
    Return the pytest command of the arguments ``args``, each quoted for the
    platform's shell (``quote``): on POSIX, ``shlex.split`` of it gives
    ``["pytest", *args]``.

    Args:
        windows: Quote for Windows; the platform's rule when None
    """
    return " ".join(["pytest", *(quote(arg, windows=windows) for arg in args)])


def rerun_args(nodeids: Sequence[str], seed: int, options: Sequence[str]) -> list[str]:
    """
    Return the arguments of the command that runs rows again, unquoted: their node
    IDs, ``--rng-seed=S`` and ``options``.
    """
    return [*nodeids, f"--rng-seed={seed}", *options]


def rerun_command(nodeid: str, seed: int, options: Sequence[str]) -> str:
    """
    Return the command that runs a row again, ``pytest <node id> --rng-seed=S``
    followed by ``options``, quoted for the platform's shell.
    """
    return command_line(rerun_args([nodeid], seed, options))


def given_seed(args: Sequence[str]) -> int | None:
    """
    Return the seed the last ``--rng-seed`` of the command-line arguments ``args``
    gives (``--rng-seed=S`` or ``--rng-seed S``, before a ``--``), or None when
    none does or its value is not an integer.
    """
    found: str | None = None
    for position, arg in enumerate(args):
        if arg == "--":
            break
        if arg.startswith("--rng-seed="):
            found = arg.partition("=")[2]
        elif arg == "--rng-seed" and position + 1 < len(args):
            found = args[position + 1]
    if found is None:
        return None
    try:
        return int(found)
    except ValueError:
        return None


def environment_args() -> list[str]:
    """
    Return the arguments the ``PYTEST_ADDOPTS`` environment variable gives, which
    pytest puts before those of the command line, or none when it cannot be split.
    """
    try:
        return shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))
    except ValueError:
        return []


def invocation_args(config: pytest.Config) -> list[str]:
    """
    Return the arguments the run was started with, unquoted: those of the
    ``PYTEST_ADDOPTS`` environment variable, then those pytest was called with
    (``config.invocation_params.args``). The ini file's addopts are not among
    them: a command run from the same folder reads them again.

    They are read once, when the plugin is configured, so that a test that changes
    the environment does not change them. A pytest-xdist worker, which was started
    with other arguments, gets the controller's (``INVOCATION``).
    """
    state = runtime.session_of(config)
    if state is not None and state.invocation is not None:
        return list(state.invocation)
    sent = getattr(config, "workerinput", {}).get(INVOCATION)
    if isinstance(sent, list):
        args = [str(arg) for arg in sent]
    else:
        args = [*environment_args(), *(os.fspath(arg) for arg in config.invocation_params.args)]
    if state is not None:
        state.invocation = list(args)
    return args


def _positionals(config: pytest.Config, args: Sequence[str]) -> list[str] | None:
    """
    Return the paths and node IDs among ``args``, as pytest's own parser finds them
    (it knows which options take a value), or None when it cannot say.
    """
    parser = getattr(config, "_parser", None)
    both = getattr(parser, "parse_known_and_unknown_args", None)
    parse = getattr(parser, "parse_known_args", None)
    try:
        if both is not None:
            # Before pytest 9 a path after an option is among the arguments the
            # parser did not take, which pytest then adds to the paths itself
            known, unknown = both(list(args))
            found = getattr(known, "file_or_dir", None)
            if isinstance(found, list):
                found = [*found, *(arg for arg in unknown if not arg.startswith("-"))]
        elif parse is not None:
            found = getattr(parse(list(args)), "file_or_dir", None)
        else:
            return None
    except (Exception, SystemExit):
        # An argument only the run's start could read (a plugin's own parsing)
        return None
    return list(found) if isinstance(found, list) else None


def _options_of(config: pytest.Config, args: Sequence[str]) -> list[str] | None:
    """
    Return ``args`` without the paths and node IDs the run started from, and
    without what follows a ``--`` (paths too), or None when they cannot be told
    from the options' values.

    An argument is a path when pytest's parser lists it as one and no longer does
    once it is taken out; an option's value that reads the same stays.
    """
    head = list(args[: args.index("--")] if "--" in args else args)
    paths = _positionals(config, head)
    if paths is None:
        return None
    options = []
    left = list(paths)
    for position, arg in enumerate(head):
        if arg in left:
            expected = list(paths)
            expected.remove(arg)
            if _positionals(config, head[:position] + head[position + 1 :]) == expected:
                left.remove(arg)
                continue
        options.append(arg)
    return options


def _covered_override(override: str) -> bool:
    """Whether ``generation_options`` gives an ``-o`` override itself."""
    name = override.partition("=")[0].strip()
    return name.startswith("strategies_") or name == EMPTY_PARAMETER_SET_MARK


def _carried(options: Sequence[str]) -> list[str]:
    """
    Return the options a command for chosen rows takes over from the run's
    ``options`` (its arguments without the paths): all but those the command gives
    itself (``_OWN_OPTIONS``, ``-c``, the ``-o`` overrides of
    ``generation_options``) and those it leaves out (``_LEFT_OUT_FLAGS``, and
    ``-k``, ``-m`` and ``--deselect``: the command names its rows, and a reader may
    put the options after a test the run deselected).
    """
    kept: list[str] = []
    position = 0
    while position < len(options):
        arg = options[position]
        name, equals, value = arg.partition("=")
        position += 1
        if arg in _LEFT_OUT_FLAGS:
            continue
        if name in _OWN_OPTIONS:
            # "--nsamples=4", or "--nsamples 4"
            position += 0 if equals else 1
        elif arg in ("-c", "-k", "-m"):
            position += 1
        elif arg[:2] in ("-c", "-k", "-m") and not arg.startswith("--"):
            # "-kexpression"
            continue
        elif arg in ("-o", "--override-ini"):
            if position < len(options):
                if not _covered_override(options[position]):
                    kept += [arg, options[position]]
                position += 1
        elif name == "--override-ini" and equals:
            if not _covered_override(value):
                kept.append(arg)
        elif arg.startswith("-o") and not arg.startswith("--"):
            if not _covered_override(arg[2:]):
                kept.append(arg)
        else:
            kept.append(arg)
    return kept


def carried_options(config: pytest.Config) -> list[str]:
    """
    Return the options of the run that a command for chosen rows repeats after its
    own, unquoted and as they were given: the arguments the run was started with
    (``invocation_args``) without its paths and node IDs, which the command
    replaces by its rows, without the options the command gives itself (the seed,
    ``generation_options``) and without those that would not run the rows or would
    write over the run's export (``--lf``, ``--sw``, ``--collect-only``,
    ``--cache-clear``, ``--strategy-export``), nor those that select tests (``-k``,
    ``-m``, ``--deselect``): the command names its rows.

    An option of the project's own that shapes the context the factories receive
    (``--width=12``), or one the collection needs (``--import-mode=importlib``,
    ``-p``), is thus given again, so the rows get the same values. None are
    repeated when pytest's parser cannot tell the paths from the options.
    """
    state = runtime.session_of(config)
    if state is not None and state.carried is not None:
        return list(state.carried)
    options = _options_of(config, invocation_args(config))
    carried = [] if options is None else _carried(options)
    if state is not None:
        state.carried = list(carried)
    return carried


def rerun_options(config: pytest.Config, infos: Sequence[VectorInfo]) -> list[str]:
    """
    Return what follows ``--rng-seed`` in a command that runs chosen rows again
    from the folder pytest was started in, unquoted: the options that decide which
    rows exist and what they hold (``generation_options``), then the run's other
    options (``carried_options``).
    """
    return [*generation_options(config, infos), *carried_options(config)]


def run_args(config: pytest.Config, seed: int) -> list[str]:
    """
    Return the arguments of the command that runs the whole run again from the
    folder pytest was started in, with the same vectors, unquoted: the arguments
    the run was started with (``invocation_args``: those of ``PYTEST_ADDOPTS``,
    then those pytest was called with), as given, and ``--rng-seed=S`` when they do
    not give the run's seed (a seed pytest-strategies chose, a conftest.py set or a
    ``--lf`` run reused), placed before a ``--``.

    Nothing is taken out: an option whose effect depends on what is on disk
    (``--lf``, ``--sw``, ``--strategy-export``) is repeated as it was given
    (``run_note`` says so for those that select tests by the cache).
    """
    args = invocation_args(config)
    if given_seed(args) == seed:
        return args
    option = f"--rng-seed={seed}"
    if "--" in args:
        args.insert(args.index("--"), option)
    else:
        args.append(option)
    return args


def run_note(args: Sequence[str]) -> str | None:
    """
    Return what is said of the command of the arguments ``args`` that runs a whole
    run again, or None when it runs the same tests: with ``--lf`` or ``--sw`` it
    selects tests by what the cache holds when it runs.
    """
    head = args[: args.index("--")] if "--" in args else args
    found = [arg for arg in dict.fromkeys(head) if arg in CACHE_OPTIONS]
    if not found:
        return None
    return (
        f"{' and '.join(found)} selects tests by what the cache holds when the command "
        "runs, so it may select other tests than this run did. Each test's own command "
        "names the rows this run selected."
    )


def start_args(config: pytest.Config, start: Path) -> list[str] | None:
    """
    Return the paths and node IDs the run started from (``config.args``: those on
    the command line, or else the testpaths or the folder pytest was started in),
    relative to the folder ``start``, or None when one of them is not an existing
    path (a ``--pyargs`` module).
    """
    found = []
    for arg in config.args:
        text, separator, names = arg.partition("::")
        path = Path(os.path.abspath(config.invocation_params.dir / text))
        if not path.exists():
            return None
        found.append(bestrelpath(start, path) + separator + names)
    return found


def ignore_args(config: pytest.Config, start: Path) -> list[str]:
    """
    Return the run's ``--ignore`` and ``--ignore-glob`` options (from the command
    line or the ini file's addopts), unquoted, each value relative to the folder
    ``start``: the files they left out were not collected, so a command that starts
    from the run's paths must leave them out too.
    """
    args: dict[tuple[str, str], None] = {}
    for name, dest in (("--ignore", "ignore"), ("--ignore-glob", "ignore_glob")):
        for value in getattr(config.option, dest, None) or ():
            # pytest makes them absolute from the folder it was started in
            path = Path(os.path.abspath(config.invocation_params.dir / str(value)))
            args[name, bestrelpath(start, path)] = None
    return [arg for pair in args for arg in pair]


class _Matchers:
    """
    pytest's ``-k`` matcher of each test the run collected, with an index of the
    names they match by, so that the tests a name can select are found without
    asking every matcher: an export gives every row outside the rootdir its
    command, which asking every matcher for every row made quadratic.

    A matcher selects a test by a name when the name is part of one of the test's
    names, whatever the case (``KeywordMatcher.__call__``). The index maps each
    piece of ``_GRAM`` characters of each lowercased name to the names that hold
    it; a name to look for is held only by names that hold all its pieces. Without
    the matchers' private ``_names``, every test is a candidate.
    """

    def __init__(self, tests: Sequence[pytest.Item], matcher_of: Any) -> None:
        self.tests = list(tests)
        self.matchers = [matcher_of(test) for test in self.tests]
        self._tests_of: dict[str, list[int]] | None = {}
        self._names_of: dict[str, list[str]] = {}
        for position, matcher in enumerate(self.matchers):
            names = getattr(matcher, "_names", None)
            if not isinstance(names, Iterable) or isinstance(names, str):
                self._tests_of = None
                return
            for name in {str(name).lower() for name in names}:
                self._tests_of.setdefault(name, []).append(position)
        for name in self._tests_of:
            for gram in {name[at : at + _GRAM] for at in range(len(name) - _GRAM + 1)}:
                self._names_of.setdefault(gram, []).append(name)

    def candidates(self, name: str) -> list[int]:
        """The positions of the tests an expression that requires ``name`` can select."""
        if self._tests_of is None:
            return list(range(len(self.tests)))
        name = name.lower()
        names: Sequence[str]
        if len(name) < _GRAM:
            names = list(self._tests_of)
        else:
            names = min(
                (
                    self._names_of.get(name[at : at + _GRAM], ())
                    for at in range(len(name) - _GRAM + 1)
                ),
                key=len,
            )
        found: set[int] = set()
        for holder in names:
            if name in holder:
                found.update(self._tests_of[holder])
        return sorted(found)


def keyword(item: pytest.Item) -> str | None:
    """
    Return a ``-k`` expression that selects only ``item`` among the tests the run
    collected, those it deselected included (those its ``--ignore`` and
    ``--ignore-glob`` left out, which it never saw, stay out of the command through
    ``ignore_args``): the item's name
    (``test_write[rand-3]``), else with its module's (``test_dma.py and
    test_write[rand-3]``), else with its classes' too. None when there is no such
    expression, or a name does not fit ``-k``'s grammar (``test_esm[ch=2-rand-1]``).

    The tests are matched with pytest's own ``-k`` matcher; without it, None. Only
    the tests the item's name can select are asked (``_Matchers``): every
    expression requires it.
    """
    state = runtime.session_of(item.config)
    if state is None:
        return None
    try:
        from _pytest.mark import KeywordMatcher
        from _pytest.mark.expression import Expression
    except ImportError:
        # A pytest that moved them
        return None
    chain = item.listchain()
    module = [node.name for node in chain if isinstance(node, pytest.Module)]
    classes = [node.name for node in chain if isinstance(node, pytest.Class)]
    if state.keyword_matchers is None:
        tests = [*getattr(item.session, "items", ()), *state.deselected_items]
        state.keyword_matchers = _Matchers(tests, KeywordMatcher.from_item)
    matchers: _Matchers = state.keyword_matchers
    candidates: list[int] | None = None
    for names in ([item.name], [*module, item.name], [*module, *classes, item.name]):
        if not all(_KEYWORD_NAME.fullmatch(n) and n not in _KEYWORD_OPERATORS for n in names):
            continue
        text = " and ".join(names)
        expression = Expression.compile(text)
        if candidates is None:
            candidates = matchers.candidates(item.name)
        matched = [
            matchers.tests[position]
            for position in candidates
            if expression.evaluate(matchers.matchers[position])
        ]
        if matched == [item]:
            return text
    return None


def keyword_args(
    args: Sequence[str], seed: int, options: Sequence[str], expression: str
) -> list[str]:
    """
    Return the arguments of the command that runs rows outside the rootdir again,
    unquoted: ``args`` (the paths to start from), ``--rng-seed=S``, ``options`` and
    ``-k <expression>``.
    """
    return [*args, f"--rng-seed={seed}", *options, "-k", expression]


def keyword_command(args: Sequence[str], seed: int, options: Sequence[str], expression: str) -> str:
    """
    Return the command that runs a row outside the rootdir again, ``pytest <args>
    --rng-seed=S`` followed by ``options`` and ``-k <expression>``, quoted for the
    platform's shell.
    """
    return command_line(keyword_args(args, seed, options, expression))


def outside_args(
    config: pytest.Config,
    expressions: Sequence[str | None],
    seed: int,
    options: Sequence[str],
    start: Path,
) -> list[str] | None:
    """
    Return the arguments of the command that runs rows outside the rootdir again
    from the folder ``start``: the paths the run started from, so that pytest gives
    them the same node IDs, without the files the run's ``--ignore`` and
    ``--ignore-glob`` left out (``ignore_args``), selected with ``-k`` (each row's
    expression, as ``keyword`` gives it, joined with ``or``). None when there is no
    such command: a row without an expression, or a run that started from a
    ``--pyargs`` module.
    """
    chosen = [expression for expression in expressions if expression is not None]
    if not chosen or len(chosen) != len(expressions):
        return None
    args = start_args(config, start)
    if args is None:
        return None
    return keyword_args([*args, *ignore_args(config, start)], seed, options, " or ".join(chosen))


def outside_note(config: pytest.Config) -> str:
    """
    Return what is said of a node ID command for a test outside the rootdir, which
    does not reproduce its rows.
    """
    rootdir = bestrelpath(config.invocation_params.dir, config.rootpath)
    return (
        f"the test is outside the rootdir ({rootdir}), so pytest names it by the "
        "path it was collected from: this command gives the row another node ID "
        "and other values. A run with a --rootdir that contains the test prints a "
        "command that reproduces it."
    )


class Rerun(NamedTuple):
    """The command that runs rows again from the folder pytest was started in."""

    # The command's arguments, unquoted, without "pytest"
    args: list[str]
    # Why the command does not reproduce the rows, or None when it does
    note: str | None = None
    # For a row outside the rootdir: the -k expression that selects only it
    expression: str | None = None

    @property
    def command(self) -> str:
        """The command, quoted for the platform's shell."""
        return command_line(self.args)


def row_rerun(item: pytest.Item, infos: Sequence[VectorInfo], *, carry: bool = False) -> Rerun:
    """
    Return the command that runs an item's row again from the folder pytest was
    started in: what the section's ``rerun`` line gives and, with the run's other
    options after it (``carry``), the run export's ``rerun.args``.

    The command names the item's node ID; for a file outside the rootdir it starts
    from the run's own paths and selects the row with ``-k`` (``outside_args``),
    and when no ``-k`` expression selects only that row it is the node ID command,
    with a note that says it does not reproduce the row.

    Args:
        item: The row's item
        infos: The row's VectorInfo for each of the item's strategies
        carry: Repeat the run's other options (``carried_options``), as a reader
            of the export needs them: the terminal prints the command under the
            run's own, whose options its reader knows
    """
    config = item.config
    seed = infos[0].seed
    options = rerun_options(config, infos) if carry else generation_options(config, infos)
    if not outside_rootdir(item):
        return Rerun(rerun_args([rerun_nodeid(item)], seed, options))
    expression = keyword(item)
    args = outside_args(config, [expression], seed, options, config.invocation_params.dir)
    if args is not None:
        return Rerun(args, None, expression)
    return Rerun(rerun_args([rerun_nodeid(item)], seed, options), outside_note(config))


def rows_rerun(
    items: Sequence[pytest.Item],
    reruns: Sequence[Rerun],
    infos: Sequence[VectorInfo],
    seed: int,
    *,
    whole: bool,
) -> Rerun:
    """
    Return the command that runs several rows of one test function again from the
    folder pytest was started in, with the same values, for the run export: the
    run's other options come after the command's own (``rerun_options``).

    Args:
        items: The rows' items, all of one test function
        reruns: Each item's own command (``row_rerun`` with ``carry``)
        infos: The VectorInfo of the rows, for the constraints turned off
        seed: The run's seed
        whole: Whether they are all the items of the function: the command then
            names the function instead of each item
    """
    config = items[0].config
    options = rerun_options(config, infos)
    if outside_rootdir(items[0]):
        args = outside_args(
            config,
            [rerun.expression for rerun in reruns],
            seed,
            options,
            config.invocation_params.dir,
        )
        if args is not None:
            return Rerun(args)
        note: str | None = outside_note(config)
    else:
        note = None
    nodeids = [rerun_function_nodeid(items[0])] if whole else [rerun_nodeid(item) for item in items]
    return Rerun(rerun_args(nodeids, seed, options), note)


def _outside_command(
    item: pytest.Item, seed: int, options: Sequence[str], start: Path
) -> str | None:
    """
    Return the command that runs a row outside the rootdir again from the folder
    ``start`` (``outside_args``), or None when there is no such command.
    """
    args = outside_args(item.config, [keyword(item)], seed, options, start)
    return None if args is None else command_line(args)


def describe(infos: Sequence[VectorInfo]) -> str:
    """
    Name a row's strategies and rows for the list of failed rows: ``burst random
    3``, ``burst directed zeros``, joined with ``, `` for stacked strategies.
    """
    parts = []
    for info in infos:
        if info.kind in ("directed", "test"):
            parts.append(f"{info.strategy} {info.kind} {info.name}")
        elif info.index is not None:
            parts.append(f"{info.strategy} {info.kind} {info.index}")
        else:
            parts.append(f"{info.strategy} {info.kind}")
    return ", ".join(parts)


def _vector_line(info: VectorInfo) -> str:
    """
    Describe a row by its ID and its kind: ``rand-3 (random row 3)``,
    ``directed-zeros (directed vector 'zeros', #0)``, ``ch=2 (exhaustive row 5)``.
    """
    if info.kind in ("directed", "test"):
        return f"{info.id} ({info.kind} vector {info.name!r}, #{info.index})"
    if info.index is None:
        return f"{info.id} ({info.kind} row)"
    return f"{info.id} ({info.kind} row {info.index})"


def _value_shown(value: Any, verbosity: int) -> str:
    """
    Return a value's stable repr, cut at ``VALUE_LIMIT`` characters below -vv. A
    value whose repr raises is shown by its type and the exception's
    (``<Reg: repr() raised RuntimeError>``), as the report of its failure must
    still be made: an exception here would stop the session with an INTERNALERROR.
    """
    try:
        text = _value_repr(value)
    except Exception as error:
        text = f"<{type(value).__name__}: repr() raised {type(error).__name__}>"
    if verbosity < 2 and len(text) > VALUE_LIMIT:
        text = f"{text[:VALUE_LIMIT]}... ({len(text)} characters; -vv shows all)"
    return text


def _value_text(name: str, value: Any, verbosity: int) -> str:
    """
    Show one value as ``name=repr``, with its stable repr, cut at ``VALUE_LIMIT``
    characters below -vv; the lines of a repr that has several are aligned.
    """
    return f"{name}={_value_shown(value, verbosity)}".replace("\n", "\n" + " " * _LABEL)


def section(
    infos: Sequence[VectorInfo], command: str, verbosity: int, note: str | None = None
) -> str:
    """
    Return the text of a failed row's section: a block for each of the item's
    strategies (its strategy and origin, the row, its values, the seed and the
    context's fingerprint when the factory received one), then the rerun command,
    and ``note`` after it.
    """
    lines = []
    for block, info in enumerate(infos):
        if block:
            lines.append("")
        origin = f" ({info.origin})" if info.origin is not None else ""
        lines.append(f"{'strategy':<{_LABEL}}{info.strategy}{origin}")
        lines.append(f"{'vector':<{_LABEL}}{_vector_line(info)}")
        names = type(info.values)._fields
        values = [
            _value_text(name, value, verbosity)
            for name, value in zip(names, info.values, strict=True)
        ]
        for position, text in enumerate(values or ["(none)"]):
            label = "values" if position == 0 else ""
            lines.append(f"{label:<{_LABEL}}{text}")
        lines.append(f"{'seed':<{_LABEL}}{info.seed}")
        if info.context is not None:
            lines.append(f"{'context':<{_LABEL}}{info.context}")
    lines.append(f"{'rerun':<{_LABEL}}{command}")
    if note is not None:
        lines.append(f"{'note':<{_LABEL}}{note}")
    return "\n".join(lines)


def failure(item: pytest.Item, infos: Sequence[VectorInfo]) -> tuple[str, dict[str, str]]:
    """
    Return what a failed setup or call of a strategy row reports: the text of its
    section, and the ``pytest_strategies`` attribute of its report, from which the
    list of failed rows is printed (strings only, so pytest-xdist carries it to the
    controller):

    - ``command``: the rerun command, with the node ID relative to the folder
      pytest was started in (for a file outside the rootdir, the paths the run
      started from and ``-k``);
    - ``row``: the row's strategies and rows, as ``describe()`` names them, and
      ``(outside the rootdir)`` for a file outside the rootdir that the command
      does not reproduce;
    - ``seed``: the run's seed;
    - ``options``: the arguments after ``--rng-seed`` of the command run from the
      rootdir, unquoted, as a JSON list (``["--nsamples=13", "-c", "pytest.ini"]``
      for ``-c ci/pytest.ini``): what the failed-seeds map records (``_reuse``).
    """
    config = item.config
    seed = infos[0].seed
    recorded = generation_options(config, infos, start=config.rootpath)
    verbosity = int(getattr(config.option, "verbose", 0))
    row = describe(infos)
    rerun = row_rerun(item, infos)
    if rerun.note is not None:
        row += " (outside the rootdir)"
    command = rerun.command
    attribute = {"command": command, "row": row, "seed": str(seed), "options": json.dumps(recorded)}
    return section(infos, command, verbosity, rerun.note), attribute


def testcase_properties(item: pytest.Item, infos: Sequence[VectorInfo]) -> list[tuple[str, str]]:
    """
    Return the JUnit XML properties of a failed strategy row's test case, all
    strings: ``pytest_strategies.strategy``, ``.kind``, ``.name`` (a directed or
    test vector's), ``.index`` (unless None), ``.id``, ``.value.<argument>`` (each
    value's stable repr, cut as in the section), ``.seed``, ``.context`` and
    ``.constraints_off`` (when set; the names, joined with ``,``), and ``.command``,
    the rerun command run from the rootdir. A test with several strategies gets
    them for each one, as ``pytest_strategies.<i>.strategy`` and so on, in the
    order of the node ID.

    Args:
        item: The failed row's item
        infos: Its VectorInfo for each of its strategies
    """
    config = item.config
    verbosity = int(getattr(config.option, "verbose", 0))
    seed = infos[0].seed
    options = generation_options(config, infos, start=config.rootpath)
    command = None
    if outside_rootdir(item):
        command = _outside_command(item, seed, options, config.rootpath)
    if command is None:
        command = rerun_command(rootdir_nodeid(item), seed, options)
    properties = []
    for position, info in enumerate(infos):
        prefix = f"{PROPERTY}." if len(infos) == 1 else f"{PROPERTY}.{position}."
        fields = [("strategy", info.strategy), ("kind", info.kind)]
        if info.name is not None:
            fields.append(("name", info.name))
        if info.index is not None:
            fields.append(("index", str(info.index)))
        fields.append(("id", info.id))
        names = type(info.values)._fields
        fields += [
            (f"value.{name}", _value_shown(value, verbosity))
            for name, value in zip(names, info.values, strict=True)
        ]
        fields.append(("seed", str(info.seed)))
        if info.context is not None:
            fields.append(("context", info.context))
        if info.constraints_off:
            fields.append(("constraints_off", ",".join(info.constraints_off)))
        fields.append(("command", command))
        properties += [(prefix + key, value) for key, value in fields]
    return properties


def suite_properties(seed: int, rows: Iterable[Mapping[str, str]]) -> list[tuple[str, str]]:
    """
    Return the JUnit XML properties of the test suite: ``pytest_strategies.seed``,
    the run's seed, and ``pytest_strategies.failed.<i>``, the rerun command of each
    failed strategy row (as in the list of failed rows), from 0 in the order they
    failed.

    Args:
        seed: The run's seed
        rows: The failed rows' ``pytest_strategies`` report attributes
    """
    commands = [row["command"] for row in rows if "command" in row]
    return [(f"{PROPERTY}.seed", str(seed))] + [
        (f"{PROPERTY}.failed.{i}", command) for i, command in enumerate(commands)
    ]
