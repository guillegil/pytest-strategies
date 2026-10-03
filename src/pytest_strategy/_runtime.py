"""Per-session runtime state for the strategy plugin.

The strategy registry is a process-global catalog populated at import time by
``@register``, so it cannot be made per-session (it is only restored when a
session ends, see below). ``register`` runs while a module is imported, with no
pytest object at hand, so the active session is found here. The active pytest
``Config``, the discovery bookkeeping and the RNG seed are per-session.

These are held on a STACK rather than a single slot because pytest sessions can
nest: running pytest in-process (e.g. via ``pytester``) starts an inner session
inside the outer one. A single global flag would let the outer session's
"already discovered" state leak into the inner run and skip its discovery. A
stack gives each session its own state and restores the parent's on exit.

The process-global state a session changes (the RNG seed, the ambient
generator's state, and the strategy registry) is snapshotted when the session
begins and put back when it ends, so an inner session leaves no trace in the
enclosing session or in later sibling sessions.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ._context import HOOK, ContextStore, FolderContext, folder_of, visible_from
from ._options import SessionOptions, StrategyOptions, parse_session_options
from ._registry import Registration, registry
from .rng import RNG

if TYPE_CHECKING:
    import pytest


@dataclass
class Resolution:
    """How one test's strategy was resolved, for the -v summary."""

    strategy: str
    # Where the factory is defined, for the summary ("tests/payments/strategies.py")
    where: str
    # The rows of each kind
    directed: int = 0
    random: int = 0
    test: int = 0
    exhaustive: int = 0
    skipped: int = 0
    # The count the factory's random rows were generated with, and where it came from
    nsamples: int | str | None = None
    source: str = ""
    # The Parameter's constraint names, in order, and the draws each rejected
    constraints: tuple[str, ...] = ()
    rejected: dict[str, int] = field(default_factory=dict)
    # The constraints --strategy-constraint-off turned off, in order
    constraints_off: tuple[str, ...] = ()
    # Under --nsamples=auto: the combinations the constraints left out
    left_out: int | None = None


class SessionState:
    """Mutable state scoped to a single (possibly nested) pytest session."""

    def __init__(self, config: pytest.Config | None = None) -> None:
        self.config: pytest.Config | None = config
        # The seed this session started from: the header, the reproduce line and
        # xdist workers use it even if a test body reseeds the RNG later.
        self.run_seed: int | None = None
        self.strategies_loaded: bool = False
        self.discovered_files: list[Path] = []
        # Module of each loaded strategy file, by real path: an import of the file
        # reuses it instead of running the file again (see plugin._StrategyFileFinder).
        self.strategy_modules: dict[str, ModuleType] = {}
        # Strategy files that raised while loading, as (path, "ErrorType: message").
        self.load_errors: list[tuple[Path, str]] = []
        # Strategy files that called pytest.skip()/importorskip(), as (path, reason).
        self.skipped_files: list[tuple[Path, str]] = []
        # Files named like strategy files that discovery did not import because
        # they contain no registration decorator (e.g. they register through an alias).
        self.unimported_files: list[Path] = []
        # Directories whose strategy files were loaded (normalized real paths), and
        # whether every strategy file of the session was
        self.loaded_dirs: set[str] = set()
        self.all_loaded: bool = False
        # Strategy files imported (or attempted), by normalized real path
        self.loaded_files: set[str] = set()
        # The files pytest or the plugin imported by their paths in this session, by
        # normalized real path: the test modules pytest collected, the conftest.py
        # files it loaded and the strategy files the plugin loaded. A file named like
        # one outside the rootdir and the testpaths is keyed by its path in a fixture's
        # or an export's stream only when it is here and is not, without
        # consider_namespace_packages, a regular package's module held under its
        # package name (_registry.source_part).
        self.imported_files: set[str] = set()
        # Where a lookup stops going up: the rootdir and the search paths
        self.ceilings: set[str] | None = None
        # A name registered twice in the same directory: the messages, reported as
        # a usage error once collection ends
        self.clashes: list[str] = []
        # One entry per test and strategy, for the -v summary
        self.resolutions: list[Resolution] = []
        # On the pytest-xdist controller, which collects nothing: the -v summary
        # of the first worker that finished (every worker collects all the tests)
        self.worker_summary: dict[str, Any] | None = None
        # --vector-name/--vector-index bookkeeping: whether a Parameter strategy
        # was resolved with the filter, whether any of them had the vector, and
        # the directed vector names of those that did not (strategy -> names).
        self.vector_filter_resolved: bool = False
        self.vector_filter_matched: bool = False
        self.vector_filter_misses: dict[str, list[str]] = {}
        # --strategy-constraint-off bookkeeping: the constraint names of every
        # Parameter strategy resolved in this session, by strategy name, in
        # evaluation order (a name resolved in two folders gets both sets), and
        # the lines reporting the items that matched none of them in a run that
        # did not collect the whole suite. A collector that was skipped or failed
        # leaves its tests' strategies unresolved, so the run counts as narrowed.
        self.constraint_names: dict[str, dict[str, None]] = {}
        self.unmatched_constraints_off: list[str] = []
        self.collectors_incomplete: bool = False
        # The options factories receive: the session-wide part, read from the
        # config on first use, and each strategy's instance, by resolved name
        self.options: SessionOptions | None = None
        self.strategy_options: dict[str, StrategyOptions] = {}
        # The pytest_strategies_context implementations' answers: each one is called
        # when a folder first needs it, and its result, or the exception it raised,
        # is kept for the rest of the session (see _context.ContextStore)
        self.contexts: ContextStore = ContextStore(config)
        # Where each fixture that drew is defined and registered
        # (plugin._fixture_definition and _fixture_base), the last parts of its
        # random stream's key, by FixtureDef
        self.fixture_definitions: dict[Any, tuple[str, str, str]] = {}
        # Process-global state in effect when this session began, restored on pop:
        # a nested session's --rng-seed, random draws and strategy registrations
        # must not leak into the enclosing session or later sibling sessions.
        self.prev_seed: int | None = None
        self.prev_rng_state: Any = None
        self.prev_generator: random.Random | None = None
        self.prev_registry: dict[str, list[Registration]] | None = None

    def seed(self) -> int:
        """The seed this session started from, or the RNG's seed before it started."""
        return self.run_seed if self.run_seed is not None else RNG.get_seed()

    def test_context(self, node: Any) -> FolderContext:
        """
        Return the context of a test's folder in this session, computed when it is
        first asked for.

        The implementations it consults are those the test's folder-scoped hook
        relay (``node.ihook``) sees, as for ``pytest_generate_tests``: every plugin's,
        and those of the conftest.py files in the test's folder and above it.

        Args:
            node: The test's node (``metafunc.definition``, or an item)
        """
        if self.config is None:
            return _no_context()
        return FolderContext(
            self.contexts,
            lambda: getattr(node.ihook, HOOK).get_hookimpls(),
            lambda: node.path.parent,
            self.seed,
        )

    def path_context(self, path: str | os.PathLike[str] | None = None) -> FolderContext:
        """
        Return the context of the folder of ``path`` (a file, or a folder) in this
        session, computed when it is first asked for: the context a test in that
        folder gets.

        The implementations it consults are every plugin's, and those of the loaded
        conftest.py files in that folder and above it (``_context.visible_from``).
        Without ``path``, the folder is the rootdir.
        """
        config = self.config
        if config is None:
            return _no_context()
        target = path if path is not None else getattr(config, "rootpath", None)
        if target is None:
            return _no_context()
        return FolderContext(
            self.contexts,
            lambda: visible_from(config, target),
            lambda: folder_of(target),
            self.seed,
        )


def _no_folder() -> None:
    """The folder of a context without a session: none."""
    return None


def _no_context() -> FolderContext:
    """The context without a session or a config: None, and no hook to call."""
    return FolderContext(None, tuple, _no_folder, RNG.get_seed)


class StrategyRuntime:
    """A stack of session states; the top is the currently active session."""

    def __init__(self) -> None:
        self._stack: list[SessionState] = []

    def push(self, config: pytest.Config | None = None) -> SessionState:
        """Begin a session (``pytest_configure``)."""
        state = SessionState(config)
        state.prev_seed = RNG.get_seed()
        state.prev_rng_state = RNG._ambient.getstate()
        state.prev_generator = RNG._generator
        state.prev_registry = registry.snapshot()
        self._stack.append(state)
        return state

    def pop(self) -> None:
        """End a session (``pytest_unconfigure``) and restore the state it began with."""
        if self._stack:
            state = self._stack.pop()
            # Restore the seed without reseeding: the generator is restored as it
            # was, not restarted from the beginning of the seed's stream.
            if state.prev_seed is not None:
                RNG._seed = state.prev_seed
            if state.prev_rng_state is not None:
                RNG._ambient.setstate(state.prev_rng_state)
            if state.prev_generator is not None:
                RNG._generator = state.prev_generator
            # Only a nested session's registrations are removed. The modules that
            # registered stay in sys.modules and are not run again, so a second
            # pytest.main() in the same process needs the registrations.
            if state.prev_registry is not None and self._stack:
                registry.restore(state.prev_registry)

    @property
    def current(self) -> SessionState | None:
        return self._stack[-1] if self._stack else None

    # Convenience accessors that operate on the active session ----------- #

    def run_seed(self) -> int:
        """The seed the active session started from, even if a test reseeded the RNG."""
        state = self.current
        return state.seed() if state is not None else RNG.get_seed()

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

    @property
    def unimported_files(self) -> list[Path]:
        return self.current.unimported_files if self.current else []

    def record_unimported_file(self, path: Path) -> None:
        """Record a file named like a strategy file that was not imported (no-op if no session)."""
        if self.current is not None:
            self.current.unimported_files.append(path)

    def record_clash(self, message: str) -> None:
        """Record a name registered twice in one directory (no-op if no session)."""
        if self.current is not None:
            self.current.clashes.append(message)

    def record_resolution(self, resolution: Resolution) -> None:
        """Record how a test's strategy was resolved (no-op if no session)."""
        if self.current is not None:
            self.current.resolutions.append(resolution)

    def load_all_strategy_files(self) -> None:
        """Load every strategy file of the active session (no-op if none)."""
        state = self.current
        if state is None or state.config is None or state.all_loaded:
            return
        # Imported here: the plugin imports this module
        from .plugin import _plugin_instance

        _plugin_instance.load_all_strategy_files(state.config)

    def session_of(self, config: object) -> SessionState | None:
        """
        Return the session that runs with ``config``, the innermost one if several
        do, or None when none does (a config of a session that ended, or of a run
        without the plugin).
        """
        if config is None:
            return None
        for state in reversed(self._stack):
            if state.config is config:
                return state
        return None

    def test_context(self, node: Any) -> FolderContext:
        """
        Return the context of a test's folder in the active session, computed when a
        factory asks for it (see :meth:`SessionState.test_context`).

        Args:
            node: The test's node (``metafunc.definition``, or an item)
        """
        state = self.current
        return state.test_context(node) if state is not None else _no_context()

    def path_context(self, path: str | os.PathLike[str] | None = None) -> FolderContext:
        """
        Return the context of the folder of ``path`` (a file, or a folder; the rootdir
        without one) in the active session, computed when it is first asked for: the
        context a test in that folder gets (see :meth:`SessionState.path_context`).
        """
        state = self.current
        return state.path_context(path) if state is not None else _no_context()

    def strategy_context(self, path: str | os.PathLike[str] | None = None) -> Any:
        """
        Return the context of the folder of ``path`` (a file, or a folder; the rootdir
        without one) in the active session (see :meth:`path_context`).

        Each ``pytest_strategies_context`` implementation is called when a folder
        first needs it, and its result is reused afterwards. Without a session or a
        config there is no hook to call, and the result is ``None``.

        Raises:
            Whatever the implementation that answered for the folder raised (an
            exception, ``pytest.skip``, ``pytest.fail``), again on every later call
            in the session.
        """
        return self.path_context(path)()

    def session_options(self, config: pytest.Config | None) -> SessionOptions:
        """
        Return the session-wide options read from ``config``.

        For the active session's config they are read once and reused for the
        rest of the session. Any other config (a unit test's stand-in) or None is
        read again, or gives the defaults, and leaves the session's cache alone.
        """
        state = self.current
        if config is None or state is None or state.config is not config:
            return parse_session_options(config)
        if state.options is None:
            state.options = parse_session_options(config)
        return state.options

    def strategy_options(self, name: str, config: pytest.Config | None) -> StrategyOptions:
        """
        Return the options of the strategy resolved under ``name``.

        For the active session's config, the session-wide part is read once and
        each name gets one instance, reused for the rest of the session. Any other
        config (a unit test's stand-in) or None gets a new instance, read from it
        or with the defaults, and leaves the session's cache alone.
        """
        state = self.current
        if config is None or state is None or state.config is not config:
            return parse_session_options(config).for_strategy(name)
        options = state.strategy_options.get(name)
        if options is None:
            options = self.session_options(config).for_strategy(name)
            state.strategy_options[name] = options
        return options

    def record_constraints(self, strategy: str, names: tuple[str, ...]) -> None:
        """
        Record the constraint names of a Parameter strategy resolved under
        ``strategy``, for the --strategy-constraint-off check (no-op if no session).
        """
        if self.current is not None:
            known = self.current.constraint_names.setdefault(strategy, {})
            known.update(dict.fromkeys(names))

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
