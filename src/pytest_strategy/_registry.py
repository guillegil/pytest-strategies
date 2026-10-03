"""
The strategy registry: the factories registered under each name, scoped by folder.

A name can be registered once per directory. A test finds a name the way pytest
finds a fixture in ``conftest.py`` files: the registration in the test's own
directory or the nearest directory above it wins. A registration that is not on
that path (a sibling folder, an installed package) is used only when it is the
only one with that name.
"""

from __future__ import annotations

import fnmatch
import functools
import inspect
import os
import sys
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from ._streams import INSTALLED_FOLDERS, path_part

# Factories are user callables that return a Parameter. They receive the inputs
# they declare by name (nsamples, ctx, rng, options; see _factory.py)
Factory = Callable[..., Any]

Origin = tuple[str | None, str | None, int | None]

# Strategy file names; the plugin imports such a file only if it also contains a
# registration
STRATEGY_FILE_PATTERNS = ("strategies.py", "strategy.py", "*_strategies.py", "*_strategy.py")

# The test module names pytest collects by default (its python_files ini option)
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def _unwrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """inspect.unwrap, or fn itself when its __wrapped__ chain loops."""
    try:
        unwrapped: Callable[..., Any] = inspect.unwrap(fn)
    except ValueError:
        return fn
    return unwrapped


def _normalize(filename: str | None) -> str | None:
    """Return the real, normalized form of a path, or None."""
    return os.path.normcase(os.path.realpath(filename)) if filename else None


def factory_source(fn: Callable[..., Any]) -> Origin:
    """
    Return the file, qualified name and first line of a factory, with the file
    path as the file system spells it.

    ``functools.wraps`` decorators and ``functools.cache`` are looked through, so
    the decorated function counts, not the decorator's wrapper. A
    ``functools.partial`` counts as the function it wraps, and a class or any
    other callable object as its class.
    """
    fn = _unwrap(fn)
    while isinstance(fn, functools.partial):
        fn = _unwrap(fn.func)
    code = getattr(fn, "__code__", None)
    if code is not None:
        # The module's __file__ is set by the import system from the real location;
        # co_filename can be stale (pytest's rewritten pyc after a checkout moved)
        filename = getattr(fn, "__globals__", {}).get("__file__") or code.co_filename
        return (filename, getattr(fn, "__qualname__", None), code.co_firstlineno)
    cls = fn if isinstance(fn, type) else type(fn)
    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    return (
        getattr(module, "__file__", None),
        cls.__qualname__,
        # Python 3.13+ records where a class statement starts
        getattr(cls, "__firstlineno__", None),
    )


def test_file_patterns(config: Any) -> Sequence[str]:
    """Return a pytest config's ``python_files`` patterns, or pytest's default ones."""
    try:
        patterns = config.getini("python_files") if config is not None else None
    except (AttributeError, ValueError):
        # Not a full pytest config (a unit test's stand-in)
        patterns = None
    return patterns if isinstance(patterns, list) else TEST_FILE_PATTERNS


def source_part(
    fn: Callable[..., Any],
    rootpath: str | os.PathLike[str] | None,
    *,
    folder: bool = False,
    test_files: Sequence[str] = TEST_FILE_PATTERNS,
    searched: Callable[[str], bool] | None = None,
) -> str:
    """
    Return where a function or a factory is defined, as a part of a random stream's
    key that is the same wherever the code is installed or checked out. The
    fixture and export streams use it.

    - The name of its module, for a module imported by that name: an installed
      package's (in a site-packages or dist-packages folder), an editable install's
      (whose file is in a source tree, also next to a rootdir that is a subfolder
      of the checkout), a plugin's or a helper module's.
    - Otherwise the file :func:`factory_source` finds, or with ``folder`` its
      folder, relative to the rootdir in posix form (``_streams.path_part()``): for
      a file that pytest or the plugin imports by its path, whose module name
      depends on ``--import-mode`` and on the folders' ``__init__.py`` files (a
      ``conftest.py``, and a test module, matched by ``test_files``, pytest's
      ``python_files``, or a strategy file in a folder the session searches for
      them, ``searched``); for a module that ``sys.modules`` does not have under
      its name; and for one whose name begins with the rootdir's own folder or a
      folder above it (a rootdir with an ``__init__.py``), which another checkout
      may not have.
    - The name of its module when its code has no file (``"<string>"`` for
      ``exec``'d code, which would resolve against the working directory); ``""``
      without a module either.

    Args:
        fn: The function, factory, partial or callable object
        rootpath: The session's rootdir, or None outside a session
        folder: Return the folder of the file instead of the file
        test_files: The ``python_files`` patterns of test modules
        searched: Whether the session searches the folder of a file for test
            modules and strategy files (``plugin._searched``): a package module
            named like one in another folder (``src/acme/test_utils.py`` with
            ``testpaths = tests``) is imported by its name only. None outside a
            session: every file named like one counts as one.
    """
    source = factory_source(fn)[0]
    # The module of what factory_source() read, through wrappers and partials (a
    # class's, for a callable object)
    fn = _unwrap(fn)
    while isinstance(fn, functools.partial):
        fn = _unwrap(fn.func)
    module = getattr(fn, "__module__", None) or ""
    if (
        source
        and os.path.isfile(source)
        and INSTALLED_FOLDERS.isdisjoint(PurePath(source).parts)
        and not _imported_by_name(source, module, rootpath, test_files, searched)
    ):
        return path_part(os.path.dirname(source) if folder else source, rootpath)
    return module


def _imported_by_name(
    source: str,
    module: str,
    rootpath: str | os.PathLike[str] | None,
    test_files: Sequence[str],
    searched: Callable[[str], bool] | None,
) -> bool:
    """
    Whether the module ``module`` of the file ``source`` is keyed by its name (see
    :func:`source_part`).
    """
    name = os.path.basename(source)
    if name == "conftest.py":
        return False
    if any(
        matches_pattern(pattern, source) for pattern in (*STRATEGY_FILE_PATTERNS, *test_files)
    ) and (searched is None or searched(source)):
        return False
    loaded = sys.modules.get(module) if module else None
    file = getattr(loaded, "__file__", None)
    if not file or (file != source and _normalize(file) != _normalize(source)):
        return False
    if rootpath is None:
        return True
    # The folder of the name's first part: the file's, up one folder per further
    # part. A top-level module's name has none, and a name with more parts than the
    # path has folders does not come from them.
    parents = PurePath(os.path.realpath(source)).parents
    depth = module.count(".") + (name == "__init__.py")
    if depth == 0 or depth >= len(parents):
        return True
    first = os.path.normcase(str(parents[depth - 1]))
    root = os.path.normcase(os.path.realpath(rootpath))
    # The name holds the rootdir folder's name, or the name of a folder above it,
    # when its first part is one of them; a package next to the rootdir (a flat
    # layout with the rootdir in tests/) is not
    return not _contains(first, root)


def matches_pattern(pattern: str, path: str) -> bool:
    """
    Match a file name pattern as pytest matches ``python_files`` and
    ``norecursedirs``: one without a path separator against the name, one with a
    separator (``tests/*.py``) against the end of the path
    (``_pytest.pathlib.fnmatch_ex``).
    """
    if os.sep != "/" and os.sep not in pattern and "/" in pattern:
        pattern = pattern.replace("/", os.sep)
    if os.sep not in pattern:
        return fnmatch.fnmatch(os.path.basename(path), pattern)
    if PurePath(path).is_absolute() and not os.path.isabs(pattern):
        pattern = f"*{os.sep}{pattern}"
    return fnmatch.fnmatch(str(path), pattern)


def _factory_origin(fn: Callable[..., Any]) -> Origin:
    """
    Identify a factory by its source file, qualified name and first line.

    The module name is deliberately not used: a strategies file can be loaded
    under different module names (import modes, nested sessions). The file path
    is normalized, so one file reached through different path strings
    (``proj/../shared/x.py`` and ``shared/x.py``, a symlink) is the same. The
    first line tells apart two functions of the same name in one file.

    Factories built by one function or class (closures, partials, instances)
    cannot be told apart (see :func:`factory_source`).
    """
    filename, qualname, line = factory_source(fn)
    return (_normalize(filename), qualname, line)


def display_path(filename: str | os.PathLike[str], rootpath: str | os.PathLike[str] | None) -> str:
    """
    Return a real path relative to ``rootpath`` in posix form, or the whole real
    path outside it, keeping the file system's spelling (``_normalize``
    lowercases it on Windows).
    """
    real = os.path.realpath(filename)
    if rootpath is not None:
        root = os.path.realpath(rootpath)
        if _contains(os.path.normcase(root), os.path.normcase(real)):
            return Path(os.path.relpath(real, root)).as_posix()
    return real


def _describe_factory(
    fn: Callable[..., Any], *, rootpath: str | os.PathLike[str] | None = None
) -> str:
    """
    Return a readable 'file:line:qualname' description of a factory for messages.

    With ``rootpath``, a file inside it is shown relative to it (see :func:`display_path`).
    """
    filename, qualname, line = factory_source(fn)
    if not filename:
        where = "<unknown>"
    elif rootpath is not None:
        where = display_path(filename, rootpath)
    else:
        where = os.path.realpath(filename)
    if line is not None:
        where = f"{where}:{line}"
    return f"{where}:{qualname or repr(fn)}"


def _contains(directory: str, path: str) -> bool:
    """Return True if ``path`` is ``directory`` or lies below it (both normalized)."""
    try:
        return os.path.commonpath([directory, path]) == directory
    except ValueError:
        # Different drives on Windows
        return False


@dataclass(frozen=True)
class Registration:
    """One factory registered under a name."""

    name: str
    factory: Factory
    origin: Origin
    # Normalized real path of the directory of the file that defines the factory,
    # or None when that file is unknown
    directory: str | None

    @property
    def file(self) -> str | None:
        """Normalized real path of the file that defines the factory, if known."""
        return self.origin[0]


class StrategyRegistry:
    """
    The registered factories, by name and directory.

    Each name keeps its registrations in the order they were made, at most one
    per directory. Registering a name again in the same directory replaces the
    earlier registration and moves it to the end.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[Registration]] = {}

    def add(self, name: str, factory: Factory) -> Registration | None:
        """
        Register ``factory`` under ``name`` in the directory of its file.

        Returns:
            The registration it replaced (same name, same directory), or None
        """
        origin = _factory_origin(factory)
        directory = os.path.dirname(origin[0]) if origin[0] else None
        registration = Registration(name, factory, origin, directory)
        entries = self._entries.setdefault(name, [])
        replaced = None
        for index, existing in enumerate(entries):
            if existing.directory == directory:
                replaced = entries.pop(index)
                break
        entries.append(registration)
        return replaced

    def registrations(self, name: str) -> list[Registration]:
        """Return the registrations of a name, oldest first."""
        return list(self._entries.get(name, ()))

    def names(self) -> list[str]:
        """Return the registered names, in the order they were first registered."""
        return list(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def nearest(self, name: str, directory: str | None) -> Registration | None:
        """
        Return the registration of ``name`` in ``directory`` or the nearest directory
        above it, or None if there is none on that path.

        Args:
            name: Strategy name
            directory: Normalized real path of the test's directory
        """
        if directory is None:
            return None
        best: Registration | None = None
        for registration in self._entries.get(name, ()):
            if registration.directory is None or not _contains(registration.directory, directory):
                continue
            if best is None or len(registration.directory) > len(best.directory or ""):
                best = registration
        return best

    def names_of(self, factory: Factory) -> list[str]:
        """Return the names ``factory`` is registered under."""
        return [
            name
            for name, entries in self._entries.items()
            if any(registration.factory is factory for registration in entries)
        ]

    def remove(self, name: str) -> None:
        """Remove every registration of a name."""
        self._entries.pop(name, None)

    def clear(self) -> None:
        """Remove every registration."""
        self._entries.clear()

    def snapshot(self) -> dict[str, list[Registration]]:
        """Return a copy of the registrations, for :meth:`restore`."""
        return {name: list(entries) for name, entries in self._entries.items()}

    def restore(self, snapshot: dict[str, list[Registration]]) -> None:
        """Put back registrations saved by :meth:`snapshot`."""
        self._entries.clear()
        self._entries.update({name: list(entries) for name, entries in snapshot.items()})


class RegistryView(MutableMapping[str, Factory]):
    """
    ``Strategy._registry``: each name mapped to the factory registered last under it.

    Kept for code written against 2.x, which read and restored the registry as a
    dict. Setting an item registers the factory in the directory of its file.
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry

    def __getitem__(self, name: str) -> Factory:
        registrations = self._registry.registrations(name)
        if not registrations:
            raise KeyError(name)
        return registrations[-1].factory

    def __setitem__(self, name: str, factory: Factory) -> None:
        self._registry.add(name, factory)

    def __delitem__(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(name)
        self._registry.remove(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry.names())

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def clear(self) -> None:
        self._registry.clear()

    def __repr__(self) -> str:
        return f"RegistryView({dict(self)!r})"


# The process-wide registry. Registrations happen when modules are imported, so
# it cannot be per session; a session restores it when it ends (see _runtime).
registry = StrategyRegistry()
