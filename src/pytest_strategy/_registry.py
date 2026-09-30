"""
The strategy registry: the factories registered under each name, scoped by folder.

A name can be registered once per directory. A test finds a name the way pytest
finds a fixture in ``conftest.py`` files: the registration in the test's own
directory or the nearest directory above it wins. A registration that is not on
that path (a sibling folder, an installed package) is used only when it is the
only one with that name.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any

# Factories are user functions called as factory(nsamples=...) that return a
# Parameter or a legacy (argnames, samples) tuple
Factory = Callable[..., Any]

Origin = tuple[str | None, str | None, int | None]


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


def _factory_origin(fn: Callable[..., Any]) -> Origin:
    """
    Identify a factory by its source file, qualified name and first line.

    The module name is deliberately not used: a strategies file can be loaded
    under different module names (import modes, nested sessions). The file path
    is normalized, so one file reached through different path strings
    (``proj/../shared/x.py`` and ``shared/x.py``, a symlink) is the same. The
    first line tells apart two functions of the same name in one file.

    ``functools.wraps`` decorators and ``functools.cache`` are looked through, so
    the decorated function counts, not the decorator's wrapper. A
    ``functools.partial`` counts as the function it wraps, and a class or any
    other callable object as its class. So factories built by one function or
    class (closures, partials, instances) cannot be told apart.
    """
    fn = _unwrap(fn)
    while isinstance(fn, functools.partial):
        fn = _unwrap(fn.func)
    code = getattr(fn, "__code__", None)
    if code is not None:
        # The module's __file__ is set by the import system from the real location;
        # co_filename can be stale (pytest's rewritten pyc after a checkout moved)
        filename = getattr(fn, "__globals__", {}).get("__file__") or code.co_filename
        return (_normalize(filename), getattr(fn, "__qualname__", None), code.co_firstlineno)
    cls = fn if isinstance(fn, type) else type(fn)
    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    return (
        _normalize(getattr(module, "__file__", None)),
        cls.__qualname__,
        # Python 3.13+ records where a class statement starts
        getattr(cls, "__firstlineno__", None),
    )


def _describe_factory(fn: Callable[..., Any]) -> str:
    """Return a readable 'file:line:qualname' description of a factory for messages."""
    filename, qualname, line = _factory_origin(fn)
    where = filename or "<unknown>"
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
