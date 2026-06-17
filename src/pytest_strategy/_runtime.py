"""Per-session runtime state for the strategy plugin.

The strategy registry is a process-global catalog populated at import time by
``@Strategy.register``, so it cannot be made per-session. The active pytest
``Config`` and the auto-discovery bookkeeping, however, ARE per-session.

These are held on a STACK rather than a single slot because pytest sessions can
nest: running pytest in-process (e.g. via ``pytester``) starts an inner session
inside the outer one. A single global flag would let the outer session's
"already discovered" state leak into the inner run and skip its discovery. A
stack gives each session its own state and restores the parent's on exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class SessionState:
    """Mutable state scoped to a single (possibly nested) pytest session."""

    def __init__(self, config: pytest.Config | None = None) -> None:
        self.config: pytest.Config | None = config
        self.strategies_loaded: bool = False
        self.discovered_files: list[Path] = []


class StrategyRuntime:
    """A stack of session states; the top is the currently active session."""

    def __init__(self) -> None:
        self._stack: list[SessionState] = []

    def push(self, config: pytest.Config | None = None) -> SessionState:
        """Begin a session (``pytest_configure``)."""
        state = SessionState(config)
        self._stack.append(state)
        return state

    def pop(self) -> None:
        """End a session (``pytest_unconfigure``)."""
        if self._stack:
            self._stack.pop()

    @property
    def current(self) -> SessionState | None:
        return self._stack[-1] if self._stack else None

    # Convenience accessors that operate on the active session ----------- #

    @property
    def config(self) -> pytest.Config | None:
        return self.current.config if self.current else None

    @config.setter
    def config(self, value: pytest.Config | None) -> None:
        # No-op when there is no active session. The session's config is already
        # captured by push(config); a standalone set_config() with an empty stack
        # must not push (it would never be popped — a leak). Guarded like
        # strategies_loaded below.
        if self.current is not None:
            self.current.config = value

    @property
    def strategies_loaded(self) -> bool:
        return self.current.strategies_loaded if self.current else False

    @strategies_loaded.setter
    def strategies_loaded(self, value: bool) -> None:
        if self.current is not None:
            self.current.strategies_loaded = value

    @property
    def discovered_files(self) -> list[Path]:
        return self.current.discovered_files if self.current else []

    def record_discovered_file(self, path: Path) -> None:
        """Append a discovered file to the active session (no-op if none)."""
        if self.current is not None:
            self.current.discovered_files.append(path)


# Process-wide stack; each session pushes on configure and pops on unconfigure.
runtime = StrategyRuntime()
