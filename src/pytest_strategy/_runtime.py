"""Per-session runtime state for the strategy plugin.

The strategy registry is a process-global catalog populated at import time by
``@Strategy.register``, so it cannot be made per-session (it is only restored
when a session ends, see below). The active pytest ``Config``, the
auto-discovery bookkeeping and the RNG seed, however, ARE per-session.

These are held on a STACK rather than a single slot because pytest sessions can
nest: running pytest in-process (e.g. via ``pytester``) starts an inner session
inside the outer one. A single global flag would let the outer session's
"already discovered" state leak into the inner run and skip its discovery. A
stack gives each session its own state and restores the parent's on exit.

The process-global state a session changes (the RNG seed, the global ``random``
state and the strategy registry) is snapshotted when the session begins and put
back when it ends, so an inner session leaves no trace in the enclosing session
or in later sibling sessions.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .rng import RNG

if TYPE_CHECKING:
    import pytest


class SessionState:
    """Mutable state scoped to a single (possibly nested) pytest session."""

    def __init__(self, config: pytest.Config | None = None) -> None:
        self.config: pytest.Config | None = config
        self.strategies_loaded: bool = False
        self.discovered_files: list[Path] = []
        # Strategy files that raised while loading, as (path, "ErrorType: message").
        self.load_errors: list[tuple[Path, str]] = []
        # Strategy files that called pytest.skip()/importorskip(), as (path, reason).
        self.skipped_files: list[tuple[Path, str]] = []
        # --vector-name/--vector-index bookkeeping: whether a Parameter strategy
        # was resolved with the filter, whether any of them had the vector, and
        # the directed vector names of those that did not (strategy -> names).
        self.vector_filter_resolved: bool = False
        self.vector_filter_matched: bool = False
        self.vector_filter_misses: dict[str, list[str]] = {}
        # Process-global state in effect when this session began, restored on pop:
        # a nested session's --rng-seed, random draws and strategy registrations
        # must not leak into the enclosing session or later sibling sessions.
        self.prev_seed: int | None = None
        self.prev_random_state: Any = None
        self.prev_registry: dict[str, Callable[..., Any]] | None = None


class StrategyRuntime:
    """A stack of session states; the top is the currently active session."""

    def __init__(self) -> None:
        self._stack: list[SessionState] = []

    def push(self, config: pytest.Config | None = None) -> SessionState:
        """Begin a session (``pytest_configure``)."""
        # Imported here: strategy.py imports this module.
        from .strategy import Strategy

        state = SessionState(config)
        state.prev_seed = RNG.get_seed()
        state.prev_random_state = random.getstate()
        state.prev_registry = dict(Strategy._registry)
        self._stack.append(state)
        return state

    def pop(self) -> None:
        """End a session (``pytest_unconfigure``) and restore the state it began with."""
        if self._stack:
            from .strategy import Strategy

            state = self._stack.pop()
            # Restore the seed without reseeding: the random state is restored
            # as it was, not restarted from the beginning of the seed's stream.
            if state.prev_seed is not None:
                RNG._seed = state.prev_seed
            if state.prev_random_state is not None:
                random.setstate(state.prev_random_state)
            if state.prev_registry is not None:
                # In place: resolve_and_parametrize holds the registry by reference.
                Strategy._registry.clear()
                Strategy._registry.update(state.prev_registry)

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

    @property
    def load_errors(self) -> list[tuple[Path, str]]:
        return self.current.load_errors if self.current else []

    def record_load_error(self, path: Path, error: str) -> None:
        """Record a strategy file that failed to load (no-op if no session)."""
        if self.current is not None:
            self.current.load_errors.append((path, error))

    @property
    def skipped_files(self) -> list[tuple[Path, str]]:
        return self.current.skipped_files if self.current else []

    def record_skipped_file(self, path: Path, reason: str) -> None:
        """Record a strategy file that skipped itself (no-op if no session)."""
        if self.current is not None:
            self.current.skipped_files.append((path, reason))

    def record_vector_filter(
        self, strategy: str, matched: bool, vector_names: list[str] | None = None
    ) -> None:
        """
        Record whether a strategy resolved under --vector-name/--vector-index had
        the requested directed vector (no-op if no session).

        Args:
            strategy: Name of the resolved strategy
            matched: True if the strategy has the requested vector
            vector_names: The strategy's directed vector names, for the error
                reported when no strategy matched
        """
        if self.current is not None:
            self.current.vector_filter_resolved = True
            if matched:
                self.current.vector_filter_matched = True
            else:
                self.current.vector_filter_misses[strategy] = list(vector_names or [])


# Process-wide stack; each session pushes on configure and pops on unconfigure.
runtime = StrategyRuntime()
