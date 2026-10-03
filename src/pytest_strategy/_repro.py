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
constraint that only another module's strategy has. Options that shape the
context cannot be known; its fingerprint shows when the rerun got another one.

A test file outside the rootdir (``-c ci/pytest.ini`` makes ``ci`` the rootdir) is
named by pytest relative to the path given on the command line that contains it,
so a run of its node ID gives the row another node ID, and with it other random
values. Its section says so (``note``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from _pytest.pathlib import bestrelpath

from ._ids import _value_repr
from ._options import constraint_off_item
from ._runtime import runtime

if TYPE_CHECKING:
    import pytest

    from ._vector import VectorInfo

# The title of the section
SECTION = "pytest-strategies"

# The characters a value's repr is cut at in the section, below -vv
VALUE_LIMIT = 4000

# The width of the section's labels ("strategy  ")
_LABEL = 10

# The arguments cmd.exe and PowerShell leave as they are (no ",", which PowerShell
# reads as an array, no "%", "@", "&", quotes or brackets)
_WINDOWS_SAFE = re.compile(r"[A-Za-z0-9_+=:./\\-]+")


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


def generation_options(config: pytest.Config, infos: Sequence[VectorInfo]) -> list[str]:
    """
    Return the options of a row's rerun command after ``--rng-seed``, unquoted:
    ``--nsamples``, ``--vector-mode``, ``--vector-name`` and ``--vector-index`` when
    the run gave them (read from the run's options, so a value from the ini file's
    addopts or a conftest counts too), each ``-o strategies_*`` override, ``-c``
    and ``--rootdir`` as given, and the constraints the row's strategies turned off
    (``--strategy-constraint-off=STRATEGY:NAME,...``).

    Args:
        config: The run's config
        infos: The row's VectorInfo for each of the item's strategies
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
    # The last value of each strategies_* ini option that -o overrides (pytest's rule)
    overrides: dict[str, str] = {}
    for override in getattr(config.option, "override_ini", None) or ():
        name = override.partition("=")[0].strip()
        if name.startswith("strategies_"):
            overrides.pop(name, None)
            overrides[name] = override
    for override in overrides.values():
        args += ["-o", override]
    inifile = getattr(config.option, "inifilename", None)
    if inifile:
        args += ["-c", str(inifile)]
    rootdir = getattr(config.option, "rootdir", None)
    if rootdir:
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


def rerun_nodeid(item: pytest.Item) -> str:
    """
    Return an item's node ID relative to the folder pytest was started in
    (``config.cwd_relative_nodeid``). For a file outside the rootdir, whose node ID
    is not relative to the rootdir, the file's own path is used instead.
    """
    config = item.config
    if not outside_rootdir(item):
        return config.cwd_relative_nodeid(item.nodeid)
    names = item.nodeid.partition("::")[2]
    return f"{bestrelpath(config.invocation_params.dir, item.path)}::{names}"


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


def _value_text(name: str, value: Any, verbosity: int) -> str:
    """
    Show one value as ``name=repr``, with its stable repr, cut at ``VALUE_LIMIT``
    characters below -vv; the lines of a repr that has several are aligned.
    """
    text = _value_repr(value)
    if verbosity < 2 and len(text) > VALUE_LIMIT:
        text = f"{text[:VALUE_LIMIT]}... ({len(text)} characters; -vv shows all)"
    return f"{name}={text}".replace("\n", "\n" + " " * _LABEL)


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
      pytest was started in;
    - ``row``: the row's strategies and rows, as ``describe()`` names them, and
      ``(outside the rootdir)`` for a file whose node ID depends on the paths on
      the command line;
    - ``seed``: the run's seed;
    - ``options``: the command's arguments after ``--rng-seed``, unquoted, as a
      JSON list (``["--nsamples=13", "-c", "ci/pytest.ini"]``).
    """
    config = item.config
    seed = infos[0].seed
    options = generation_options(config, infos)
    command = " ".join(
        ["pytest", quote(rerun_nodeid(item)), f"--rng-seed={seed}", *map(quote, options)]
    )
    verbosity = int(getattr(config.option, "verbose", 0))
    row = describe(infos)
    note = None
    if outside_rootdir(item):
        row += " (outside the rootdir)"
        rootdir = bestrelpath(config.invocation_params.dir, config.rootpath)
        note = (
            f"the test is outside the rootdir ({rootdir}), so pytest names it by the path "
            "on the command line: this command gives the row another node ID and other "
            "values. A run with a --rootdir that contains the test prints a command that "
            "reproduces it."
        )
    attribute = {"command": command, "row": row, "seed": str(seed), "options": json.dumps(options)}
    return section(infos, command, verbosity, note), attribute
