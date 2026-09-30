"""
Pytest plugin for pytest-strategies.

``@strategy`` marks a test; the plugin resolves the strategy when pytest
generates the test's parametrization (``pytest_generate_tests``). A strategy
named by a string is looked up from the test's directory upward, and the
strategy files of those directories are loaded the first time a test there
needs one.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fnmatch
import glob
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import os
import re
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any

import pytest
from _pytest.pathlib import ImportPathMismatchError, import_path
from pytest import Config, Session

from ._registry import Registration, _contains, _describe_factory, registry
from ._runtime import runtime
from .rng import RNG

# This package's folder, whose frames are left out of the errors shown for a factory
_PACKAGE_DIR = os.path.normcase(os.path.realpath(os.path.dirname(__file__)))

# Strategy file names; a file is imported only if it also contains a registration
_STRATEGY_FILE_PATTERNS = ("strategies.py", "strategy.py", "*_strategies.py", "*_strategy.py")

# A registration decorator: @Strategy.register(...), or @register("name") and
# @<module>.register("name") with a string literal (functools.singledispatch's
# @f.register(int) is not one)
_REGISTRATION = re.compile(
    rb"@[ \t]*(?:(?:[A-Za-z_][\w.]*\.)?Strategy\.register[ \t]*\("
    rb"|(?:[A-Za-z_][\w.]*\.)?register[ \t]*\(\s*[rRuU]?[\"'])"
)

# Monotonic counter so every load gets a unique module name. Without this, two
# strategy files sharing a relative path (e.g. re-running pytest in-process with
# a rewritten strategies.py) would collide in sys.modules and the second file's
# registrations would be silently dropped by the import cache.
_load_counter = itertools.count()


def _file_key(path: str | os.PathLike[str]) -> str:
    """Identify a file by its real, normalized path."""
    return os.path.normcase(os.path.realpath(path))


def _matches_norecursedirs(pattern: str, path: Path) -> bool:
    """
    Match a norecursedirs pattern against a directory as pytest does.

    A pattern without a path separator is matched against the directory name,
    one with a separator (e.g. ``tests/data``) against the end of the path
    (pytest's ``_pytest.pathlib.fnmatch_ex``).
    """
    if os.sep != "/" and os.sep not in pattern and "/" in pattern:
        pattern = pattern.replace("/", os.sep)
    if os.sep not in pattern:
        return fnmatch.fnmatch(path.name, pattern)
    if PurePath(path).is_absolute() and not os.path.isabs(pattern):
        pattern = f"*{os.sep}{pattern}"
    return fnmatch.fnmatch(str(path), pattern)


class _LoadedModuleLoader(importlib.abc.Loader):
    """Loader that hands out a module the plugin already loaded, without running it."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.module_spec = module.__spec__

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self.module

    def exec_module(self, module: ModuleType) -> None:
        # module_from_spec replaced __spec__: keep the module as it was loaded
        module.__spec__ = self.module_spec


class _StrategyFileFinder(importlib.abc.MetaPathFinder):
    """
    Make ``import my_strategies`` reuse the module the plugin loaded from that file.

    The plugin imports a strategy file under the name a test module importing
    it would use, so this finder is only needed when that was not possible (the
    name was taken by another file) and the file was loaded under a name of its
    own. Running the file a second time would redraw the values it draws at
    import time and give the test a second copy of its classes. The file is
    matched by its real path, so another file with the same name is imported as
    usual. pytest's assertion rewriting hook comes first, so a test module that
    is also a strategy file is still rewritten and collected.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        state = runtime.current
        if state is None or not state.strategy_modules:
            return None
        # Cheap check first: only a name ending like a loaded file can match
        filename = os.path.normcase(fullname.rpartition(".")[2] + ".py")
        if not any(os.path.basename(key) == filename for key in state.strategy_modules):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            return None
        module = state.strategy_modules.get(_file_key(spec.origin))
        if module is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _LoadedModuleLoader(module), origin=spec.origin
        )


_strategy_file_finder = _StrategyFileFinder()


def _install_strategy_file_finder() -> None:
    """Put the finder before the path finder (after pytest's rewriting hook)."""
    if _strategy_file_finder in sys.meta_path:
        return
    try:
        index = sys.meta_path.index(importlib.machinery.PathFinder)
    except ValueError:
        index = len(sys.meta_path)
    sys.meta_path.insert(index, _strategy_file_finder)


class PytestStrategyPlugin:
    """
    Pytest plugin for strategies.

    Per-session state (the active config, the loaded strategy files, what the
    run reports) lives in the shared ``runtime`` object, so each nested session
    has its own and it is dropped on ``pytest_unconfigure``.
    """

    # ==== CONFIGURATION HOOKS ====

    @pytest.hookimpl
    def pytest_configure(self, config: Config) -> None:
        """Seed the RNG and register the strategy marker."""
        rng_seed = config.getoption("--rng-seed", None)
        workerinput = getattr(config, "workerinput", None)
        worker_seed = workerinput.get("pytest_strategies_seed") if workerinput else None
        if rng_seed is None and worker_seed is not None:
            # A pytest-xdist worker without --rng-seed uses the controller's seed
            # (see pytest_configure_node). Workers must generate identical vectors,
            # or xdist aborts with "Different tests were collected".
            RNG.seed(worker_seed)
        else:
            # Without --rng-seed the seed chosen for this process is kept. Either
            # way the generator restarts from it, so values drawn when test
            # modules are imported follow the printed seed. The global random
            # state is not touched.
            RNG.seed(rng_seed)

        config.addinivalue_line(
            "markers",
            "strategy(name_or_factory, validate_signature=True): parametrize the test with "
            "a strategy (added by @strategy)",
        )

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """
        Send the controller's RNG seed to a pytest-xdist worker.

        Optional hook: only called when pytest-xdist is installed.
        """
        node.workerinput["pytest_strategies_seed"] = RNG.get_seed()

    # ==== COLLECTION HOOKS ====

    @pytest.hookimpl
    def pytest_collectstart(self, collector: pytest.Collector) -> None:
        """
        Load the strategy files of a test module's folder and the folders above it,
        before the module is imported.

        So the plugin, not a test module's own ``import``, runs each strategy file
        first, with its own random stream: values a strategy file draws when it is
        imported do not depend on which test modules were collected before.
        """
        if isinstance(collector, pytest.Module) and runtime.current is not None:
            self._load_directories(collector.config, _file_key(collector.path.parent))

    @pytest.hookimpl(tryfirst=True)
    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        """
        Resolve the test's ``strategy`` markers into ``parametrize`` markers.

        Each resolved strategy becomes a ``parametrize`` marker placed right after
        its ``strategy`` marker, so pytest's own hook, which runs next, applies
        them together with the test's ``@pytest.mark.parametrize`` markers in
        decorator order: test IDs are ordered as the decorators are.
        """
        own_markers = metafunc.definition.own_markers
        if not any(mark.name == "strategy" for mark in own_markers):
            return

        from ._api import Strategy
        from ._resolver import build_parametrization

        markers: list[pytest.Mark] = []
        for mark in own_markers:
            markers.append(mark)
            if mark.name != "strategy":
                continue
            error: str | None = None
            try:
                ref, validate = _marker_arguments(mark)
                name, factory = self.resolve(ref, metafunc.definition.path, metafunc.config)
                parametrization = build_parametrization(
                    name,
                    factory,
                    metafunc.function,
                    config=metafunc.config,
                    pytest_fixtures=Strategy.PYTEST_FIXTURES,
                    validate=validate,
                )
            except ValueError as e:
                if metafunc.config.getoption("fulltrace", False):
                    raise
                error = f"In {metafunc.function.__name__}: {e}{_user_traceback(e)}"
            if error is not None:
                # Reported like pytest's own parametrize errors: the message, and the
                # frames of a factory that raised, without the plugin's
                pytest.fail(error, pytrace=False)
            markers.append(
                pytest.mark.parametrize(
                    parametrization.argnames, parametrization.values, ids=parametrization.ids
                ).mark
            )
        own_markers[:] = markers

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, session: Session, config: Config, items: list[pytest.Item]
    ) -> None:
        """
        Fail the run on a strategy name registered twice in one directory, and when
        --vector-name/--vector-index matched no strategy at all.

        A strategy without the requested directed vector gets an empty parameter
        set, so its tests are skipped. That is intended when another strategy has
        the vector, but when none has it (a typo, an index out of range) every
        test would be skipped and the run would still pass.
        """
        message = self._clash_error() or self._vector_filter_error(config)
        if message is None:
            return
        if getattr(config, "workerinput", None) is None:
            raise pytest.UsageError(message)
        # A pytest-xdist worker: an exception here kills the worker, and the
        # controller fails with an INTERNALERROR that hides the message. Run
        # nothing instead; the controller stops the session with this message.
        items.clear()
        session.shouldfail = message

    @staticmethod
    def _clash_error() -> str | None:
        """Describe the strategy names registered twice in one directory, if any."""
        state = runtime.current
        if state is None or not state.clashes:
            return None
        return "\n".join(dict.fromkeys(state.clashes))

    def _vector_filter_error(self, config: Config) -> str | None:
        """
        Describe a --vector-name/--vector-index filter that matched no strategy.

        Args:
            config: Pytest config object

        Returns:
            The error message, or None if no filter was given, no Parameter
            strategy was resolved with it, or at least one of them matched
        """
        state = runtime.current
        if state is None or config.option.list_strategies:
            return None
        if not state.vector_filter_resolved or state.vector_filter_matched:
            return None

        vector_name = config.getoption("vector_name")
        if vector_name is not None:
            option = f"--vector-name={vector_name}"
        else:
            option = f"--vector-index={config.getoption('vector_index')}"
        available = "; ".join(
            f"{name}: {', '.join(names) if names else 'none'}"
            for name, names in sorted(state.vector_filter_misses.items())
        )
        return (
            f"{option} matched no directed vector in any strategy. "
            f"Directed vectors by strategy: {available}"
        )

    # ==== STRATEGY LOOKUP ====

    def resolve(
        self, ref: Any, test_path: Path, config: Config | None
    ) -> tuple[str, Callable[..., Any]]:
        """
        Find the factory a test's ``@strategy(...)`` refers to.

        Args:
            ref: A strategy name, or a factory
            test_path: The test module's path
            config: The session's config

        Returns:
            The strategy's name (for messages and the test's random stream) and
            its factory

        Raises:
            ValueError: If no strategy has that name for the test, or several
                directories that are not above the test register it
        """
        if not isinstance(ref, str):
            # A factory: its registered name keeps the test's values the same as
            # when the test names it
            names = registry.names_of(ref)
            name = names[0] if names else getattr(ref, "__qualname__", None) or repr(ref)
            return name, ref

        directory = _file_key(test_path.parent)
        if config is not None:
            self._load_directories(config, directory)
        found = registry.nearest(ref, directory)
        if found is not None:
            return ref, found.factory

        # Not registered in the test's directory or above it. A registration
        # elsewhere (a sibling folder, an installed package) is used when it is
        # the only one; the other folders' files are needed to know that.
        if config is not None:
            self.load_all_strategy_files(config)
            found = registry.nearest(ref, directory)
            if found is not None:
                return ref, found.factory
        others = [r for r in registry.registrations(ref) if not _on_path(r, directory)]
        rootpath = _file_key(config.rootpath) if config is not None else None
        outside = [r for r in others if not _inside(r, rootpath)]
        candidates = outside or others
        if len(candidates) == 1:
            return ref, candidates[0].factory
        if candidates:
            raise ValueError(_ambiguous_message(ref, test_path, candidates))
        raise ValueError(strategy_not_found_message(ref, directory, rootpath))

    def _load_directories(self, config: Config, directory: str) -> None:
        """Load the strategy files of ``directory`` and each directory above it, closest first."""
        state = runtime.current
        if state is None:
            return
        for current in self._directories_up(config, directory):
            if current in state.loaded_dirs:
                continue
            state.loaded_dirs.add(current)
            files = self._strategy_files_in(Path(current))
            if files:
                self._load_strategy_files(files, config)

    def _directories_up(self, config: Config, directory: str) -> Iterator[str]:
        """
        Yield ``directory`` and the directories above it, up to the rootdir or the
        search path that contains it.

        A directory outside all of them (a test given by an absolute path
        elsewhere) yields only itself.
        """
        state = runtime.current
        if state is not None and state.ceilings is not None:
            ceilings = state.ceilings
        else:
            ceilings = {_file_key(config.rootpath)}
            ceilings.update(_file_key(p) for p in self._search_paths(config))
            if state is not None:
                state.ceilings = ceilings
        if not any(_contains(ceiling, directory) for ceiling in ceilings):
            yield directory
            return
        current = directory
        while True:
            yield current
            if current in ceilings:
                return
            parent = os.path.dirname(current)
            if parent == current:
                return
            current = parent

    def _strategy_files_in(self, directory: Path) -> list[Path]:
        """
        Return the strategy files directly in ``directory``, sorted by name.

        A file named like a strategy file that registers nothing with a decorator
        but mentions ``register`` is recorded as not imported, for the "not
        found" error.
        """
        try:
            names = sorted(
                entry.name
                for entry in os.scandir(directory)
                if not entry.name.startswith(".")
                and any(fnmatch.fnmatch(entry.name, pattern) for pattern in _STRATEGY_FILE_PATTERNS)
                and entry.is_file()
            )
        except OSError:
            return []
        files: list[Path] = []
        for name in names:
            file_path = directory / name
            if self._contains_strategy_registration(file_path):
                files.append(file_path)
            elif b"register" in self._read_bytes(file_path):
                runtime.record_unimported_file(file_path)
        return files

    def load_all_strategy_files(self, config: Config) -> None:
        """Load every strategy file below the search paths (``--list-strategies``, export)."""
        state = runtime.current
        if state is None or state.all_loaded:
            return
        state.all_loaded = True
        norecursedirs = config.getini("norecursedirs")
        unimported: list[Path] = []
        files = self._discover_strategy_files(
            self._search_paths(config, norecursedirs), norecursedirs, unimported
        )
        for file_path in unimported:
            if file_path not in state.unimported_files:
                runtime.record_unimported_file(file_path)
        self._load_strategy_files(files, config)

    # ==== REPORTING HOOKS ====

    @pytest.hookimpl
    def pytest_report_header(self, config: Config, start_path: Path) -> list[str]:
        """Add the RNG seed to the test report header."""
        return [f"pytest-strategies: RNG seed = {RNG.get_seed()}"]

    @pytest.hookimpl
    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: int, config: Config
    ) -> None:
        """Say how to reproduce a failed run, and summarize the strategies with -v."""
        state = runtime.current
        failed = exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERRUPTED)
        distributed = getattr(config.option, "dist", "no") != "no"
        if failed and state is not None and (state.resolutions or distributed):
            terminalreporter.write_line(
                f"pytest-strategies: reproduce with --rng-seed={RNG.get_seed()}"
            )

        if self._verbosity(config) < 1:
            return
        terminalreporter.section("Strategy Summary")
        if not registry:
            terminalreporter.write_line("No strategies registered")
            return
        terminalreporter.write_line(f"Registered strategies: {_registration_count()}")
        if state is not None and state.resolutions:
            for line in _summary_lines(state.resolutions):
                terminalreporter.write_line(f"  {line}")
        if self._verbosity(config) >= 2:
            # Show all strategy names in very verbose mode
            for name in sorted(registry.names()):
                terminalreporter.write_line(f"  - {name}")

    # ==== HELPER METHODS ====

    def _search_paths(self, config: Config, norecursedirs: Sequence[str] = ()) -> list[Path]:
        """
        Determine the directories to search for strategy files.

        These are the testpaths ini entries, with glob patterns expanded as pytest
        does (or the rootdir without testpaths), plus the directory of every path
        given on the command line that they do not already cover.

        Args:
            config: Pytest config object
            norecursedirs: Directory patterns the search does not descend into

        Returns:
            List of directories to search
        """
        rootdir = Path(config.rootpath)
        testpaths = config.getini("testpaths")

        search_paths: list[Path] = []
        for entry in testpaths:
            if any(char in entry for char in "*?["):
                # pytest expands wildcards in testpaths (e.g. "pkgs/*/tests")
                matches = sorted(glob.glob(entry, root_dir=rootdir, recursive=True))
                search_paths.extend(rootdir / match for match in matches)
            else:
                search_paths.append(rootdir / entry)
        if not testpaths:
            search_paths.append(rootdir)

        # Paths named on the command line (or the testpaths pytest collects) may
        # lie outside the search paths, or inside a directory the search skips.
        invocation_dir = config.invocation_params.dir
        for arg in config.args:
            path = Path(os.path.abspath(invocation_dir / arg.split("::")[0]))
            if not path.exists():
                # Not a path, e.g. a module name given with --pyargs
                continue
            directory = path if path.is_dir() else path.parent
            if not any(self._covers(base, directory, norecursedirs) for base in search_paths):
                search_paths.append(directory)

        return search_paths

    def _covers(self, search_path: Path, directory: Path, norecursedirs: Sequence[str]) -> bool:
        """
        Check whether searching search_path also searches directory.

        Args:
            search_path: A directory that is searched
            directory: An absolute, normalized directory
            norecursedirs: Directory patterns the search does not descend into

        Returns:
            True if directory is search_path or a directory the search enters
        """
        current = Path(os.path.abspath(search_path))
        try:
            rel_parts = directory.relative_to(current).parts
        except ValueError:
            return False
        for part in rel_parts:
            current = current / part
            if self._skip_directory(current, norecursedirs):
                return False
        return True

    def _skip_directory(self, path: Path, norecursedirs: Sequence[str]) -> bool:
        """
        Check whether discovery skips a directory below a search path.

        Skipped, as in pytest's own collection: hidden and __pycache__ directories,
        directories matching the norecursedirs ini patterns (e.g. venv, build,
        node_modules, or a path pattern such as tests/data) and virtual
        environments (a directory containing pyvenv.cfg or conda-meta/history,
        whatever its name). An installed pytest-strategies has files matching the
        patterns in a virtual environment.

        Args:
            path: The directory
            norecursedirs: Directory patterns to skip

        Returns:
            True if the directory is skipped
        """
        name = path.name
        return (
            name.startswith(".")
            or name == "__pycache__"
            or any(_matches_norecursedirs(pattern, path) for pattern in norecursedirs)
            or self._is_environment(path)
        )

    @staticmethod
    def _is_environment(path: Path) -> bool:
        """Detect a virtual or conda environment, as pytest's collection does."""
        # os.path.isfile is False on any OSError. Path.is_file raises
        # PermissionError for a directory that cannot be entered before Python
        # 3.14, which would abort the whole run.
        return os.path.isfile(path / "pyvenv.cfg") or os.path.isfile(
            path / "conda-meta" / "history"
        )

    def _discover_strategy_files(
        self,
        search_paths: list[Path],
        norecursedirs: Sequence[str] = (),
        unimported: list[Path] | None = None,
    ) -> list[Path]:
        """
        Discover strategy definition files in the test directory.

        Looks for files matching these patterns:
        - **/strategies.py
        - **/strategy.py
        - **/test_strategies.py (only if it contains @Strategy.register)
        - **/*_strategies.py
        - **/*_strategy.py

        Only files that contain a registration decorator (@register(...),
        @Strategy.register(...)) are returned, so that unrelated modules that
        happen to have such a name are not imported.
        Directories below a search path are skipped as described in
        _skip_directory; symlinked directories are followed, as pytest's
        collection does. Files are returned in a deterministic order: one search
        path after the other, sorted by path within each, and a file reached
        through two paths only once.

        Args:
            search_paths: List of paths to search
            norecursedirs: Directory patterns not to descend into
            unimported: If given, the files matching a pattern that are not
                returned because they contain no registration decorator, but do
                mention register (a plain call), are appended to it, in the same
                order

        Returns:
            List of discovered strategy file paths
        """
        strategy_files: list[Path] = []
        without_registration: list[Path] = []
        seen_files: set[str] = set()

        patterns = _STRATEGY_FILE_PATTERNS

        for search_path in search_paths:
            if not search_path.is_dir():
                continue

            found: list[Path] = []
            # Each real directory is walked once, so a symlink loop ends
            visited: set[tuple[int, int]] = set()
            self._first_visit(search_path, visited)
            for dirpath, dirnames, filenames in os.walk(search_path, followlinks=True):
                directory = Path(dirpath)
                # Prune in place so the walk never enters a skipped directory. Only
                # the directories below the search path are filtered: a project that
                # lives under a dot directory (~/.cache/proj, testpaths = ../shared)
                # or a path given explicitly is searched. Sorted, so the same path
                # reaches a directory linked twice on every filesystem.
                dirnames[:] = [
                    d
                    for d in sorted(dirnames)
                    if not self._skip_directory(directory / d, norecursedirs)
                    and self._first_visit(directory / d, visited)
                ]
                found.extend(
                    directory / name
                    for name in filenames
                    if not name.startswith(".")
                    and any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
                )

            # Load in a fixed order, not the filesystem's: import-time draws and
            # which of two registrations of a name wins depend on the order.
            for file_path in sorted(found, key=lambda p: p.relative_to(search_path).as_posix()):
                # Skip if already found (another search path, or a symlink)
                key = _file_key(file_path)
                if key in seen_files:
                    continue
                seen_files.add(key)

                # Check if file contains strategy registrations
                if self._contains_strategy_registration(file_path):
                    strategy_files.append(file_path)
                # Test modules such as test_strategy.py match the patterns too;
                # only a file that registers some other way is worth reporting
                elif b"register" in self._read_bytes(file_path):
                    without_registration.append(file_path)

        if unimported is not None:
            unimported.extend(without_registration)
        return strategy_files

    @staticmethod
    def _first_visit(directory: Path, visited: set[tuple[int, int]]) -> bool:
        """Record a directory by its device and inode; False if already visited."""
        try:
            stat = os.stat(directory)
        except OSError:
            return False
        key = (stat.st_dev, stat.st_ino)
        if key in visited:
            return False
        visited.add(key)
        return True

    def _contains_strategy_registration(self, file_path: Path) -> bool:
        """
        Check if a file contains a registration decorator.

        ``@register(``, ``@Strategy.register(`` and ``@<module>.register(`` count.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file contains strategy registrations
        """
        # Bytes, so a file in another source encoding (a PEP 263 coding line) is found
        return _REGISTRATION.search(self._read_bytes(file_path)) is not None

    @staticmethod
    def _read_bytes(file_path: Path) -> bytes:
        """Return the file's content, or nothing when it cannot be read."""
        try:
            return file_path.read_bytes()
        except OSError:
            return b""

    def _load_strategy_files(self, strategy_files: list[Path], config: Config) -> None:
        """
        Load strategy definition files by importing them.

        Each file is imported at most once per session, through pytest's own
        importer with the session's --import-mode, so it gets the module name a
        test module importing it would use. Values a file draws from the RNG when
        it is imported come from a stream of its own, derived from the seed and
        its path, so they do not depend on which files were loaded before.

        Args:
            strategy_files: List of strategy file paths to load
            config: Pytest config object
        """
        state = runtime.current
        for file_path in strategy_files:
            key = _file_key(file_path)
            if state is not None:
                if key in state.loaded_files:
                    continue
                state.loaded_files.add(key)

            imported = self._imported_module(file_path)
            if imported is not None:
                # Already imported, with its strategies registered (e.g. a test
                # module ran "from .strategies import Mode"): running it again
                # would make a second copy of its classes, and the tests would get
                # values of the other copy.
                self._record_loaded(file_path, imported, config)
                continue

            rng_state = RNG.generator().getstate()
            RNG.refresh_seed(key=f"file:{_relative(file_path, config)}")
            try:
                module = self._import(file_path, config)

            except pytest.skip.Exception as e:
                # pytest.skip() / pytest.importorskip() at module level: the file
                # opted out of this run. pytest's outcome exceptions derive from
                # BaseException; one escaping this hook aborts the whole session.
                _forget(file_path)
                # Its strategies are missing: the "Strategy not found" error lists it
                runtime.record_skipped_file(file_path, str(e))
                if self._verbosity(config) >= 1:
                    self._write_line(config, f"Skipped {file_path}: {e}")

            except (Exception, pytest.fail.Exception) as e:
                # Don't fail the test session, but always report the error: its
                # strategies are missing, and the "Strategy not found" error lists
                # it again. A PytestStrategiesWarning turned into an error (a name
                # registered twice in one folder) also fails the run as a usage error.
                _forget(file_path)
                error = f"{type(e).__name__}: {e}"
                runtime.record_load_error(file_path, error)
                self._write_line(
                    config, f"Warning - Failed to load {file_path}: {error}", yellow=True
                )

            else:
                # Outside the try: a problem reporting a successful load must not
                # turn it into a load error.
                self._record_loaded(file_path, module, config)
                _install_strategy_file_finder()

            finally:
                RNG.generator().setstate(rng_state)

    def _import(self, file_path: Path, config: Config) -> ModuleType:
        """
        Import a strategy file with pytest's importer, or under a name of its own
        when its natural module name is taken by another file.
        """
        try:
            mode = config.getoption("importmode")
            namespace_packages = config.getini("consider_namespace_packages")
        except (AttributeError, ValueError):
            # Not a full pytest config (a unit test's stand-in)
            return self._import_unique(file_path, config)
        try:
            module = import_path(
                file_path,
                mode=mode,
                root=Path(config.rootpath),
                consider_namespace_packages=namespace_packages,
            )
        except ImportPathMismatchError:
            # Another file with the same module name was imported first (two
            # folders without __init__.py, each with a strategies.py)
            return self._import_unique(file_path, config)
        module_file = getattr(module, "__file__", None)
        if not module_file or _file_key(module_file) != _file_key(file_path):
            # --import-mode=importlib returns a module already imported under
            # that name without checking its file
            return self._import_unique(file_path, config)
        return module

    def _import_unique(self, file_path: Path, config: Config) -> ModuleType:
        """Import a strategy file under a module name no other file uses."""
        module_name = self._create_module_name(file_path, config)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _record_loaded(self, file_path: Path, module: ModuleType, config: Config) -> None:
        """Record a loaded strategy file, so an import of it reuses the module."""
        state = runtime.current
        if state is not None:
            state.strategy_modules[_file_key(file_path)] = module
        runtime.record_discovered_file(file_path)

        # Optionally log in verbose mode
        if self._verbosity(config) >= 2:
            self._write_line(config, f"Loaded {_relative(file_path, config)}")

    @staticmethod
    def _imported_module(file_path: Path) -> ModuleType | None:
        """
        Return a module already imported from this file whose strategies are registered.

        A module left in sys.modules by an earlier in-process session, whose
        registrations that session removed when it ended, does not count.
        """
        from ._registry import _unwrap

        key = None
        for name in registry.names():
            for registration in registry.registrations(name):
                namespace = getattr(_unwrap(registration.factory), "__globals__", None)
                if not namespace:
                    continue
                filename = namespace.get("__file__")
                if not filename or os.path.basename(filename) != file_path.name:
                    continue
                module = sys.modules.get(namespace.get("__name__", ""))
                if module is None or module.__dict__ is not namespace:
                    continue
                key = key or _file_key(file_path)
                if _file_key(filename) == key:
                    return module
        return None

    @staticmethod
    def _verbosity(config: Config) -> int:
        """
        Return the -v count.

        The terminal plugin adds -v, so config.option has no ``verbose`` with
        ``-p no:terminal``; that counts as 0.
        """
        return int(getattr(config.option, "verbose", 0))

    def _write_line(self, config: Config, message: str, **markup: bool) -> None:
        """
        Write a pytest-strategies message through the terminal reporter.

        Args:
            config: Pytest config object
            message: Text to write after the "pytest-strategies: " prefix
            markup: Terminal markup such as ``yellow=True``
        """
        terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter is None:
            return
        # Strategy files are loaded while pytest collects a test module, when
        # output is captured into the module's report
        capture = config.pluginmanager.get_plugin("capturemanager")
        disabled = (
            capture.global_and_fixture_disabled()
            if capture is not None
            else contextlib.nullcontext()
        )
        with disabled:
            # Start a line of its own after pytest's "collecting ..." progress
            writer = getattr(terminalreporter, "_tw", None)
            if writer is not None and bool(getattr(writer, "width_of_current_line", 0)):
                writer.line()
            terminalreporter.write_line(f"pytest-strategies: {message}", **markup)

    def _create_module_name(self, file_path: Path, config: Config) -> str:
        """
        Create a unique module name for a strategy file.

        Args:
            file_path: Path to the strategy file
            config: Pytest config object

        Returns:
            Module name string
        """
        # Unique per load so re-loading the same relative path never collides
        # in sys.modules (which would skip the new file's registrations).
        unique = next(_load_counter)
        try:
            # Try to create relative path from rootpath
            rel_path = file_path.relative_to(config.rootpath)
            # Convert path to module name
            module_name = str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
            return f"pytest_strategies_discovered.{module_name}_{unique}"
        except ValueError:
            # If relative path fails, use absolute path stem
            return f"pytest_strategies_discovered.{file_path.stem}_{unique}"


def _marker_arguments(mark: pytest.Mark) -> tuple[Any, bool]:
    """Return the strategy reference and the validate_signature flag of a ``strategy`` marker."""
    args = list(mark.args)
    kwargs = dict(mark.kwargs)
    if args:
        ref = args.pop(0)
    elif "name" in kwargs:
        ref = kwargs.pop("name")
    else:
        raise ValueError(
            "the strategy marker needs a strategy name or factory, e.g. @strategy('name')"
        )
    validate = kwargs.pop("validate_signature", True)
    if args or kwargs or not (isinstance(ref, str) or callable(ref)):
        raise ValueError(
            f"invalid strategy marker {mark.args!r} {mark.kwargs!r}; use "
            "@strategy(name_or_factory, validate_signature=True)"
        )
    return ref, bool(validate)


def _user_traceback(error: BaseException) -> str:
    """
    Format where the user's code raised the exception behind ``error``, if it did.

    A factory that raised is reported with its own frames only: the frames of
    pluggy, pytest and this package say nothing about the factory.
    """
    cause = error.__cause__
    if cause is None or cause.__traceback__ is None:
        return ""
    frames = [
        frame
        for frame in traceback.extract_tb(cause.__traceback__)
        if not _contains(_PACKAGE_DIR, _file_key(frame.filename))
    ]
    if not frames:
        return ""
    lines = "".join(traceback.format_list(frames)).rstrip()
    return f"\n{lines}\n{type(cause).__name__}: {cause}"


def _on_path(registration: Registration, directory: str) -> bool:
    """Return True if a registration is in ``directory`` or a directory above it."""
    return registration.directory is not None and _contains(registration.directory, directory)


def _inside(registration: Registration, rootpath: str | None) -> bool:
    """Return True if a registration's file is inside the rootdir."""
    return (
        rootpath is not None
        and registration.directory is not None
        and _contains(rootpath, registration.directory)
    )


def _relative(file_path: Path | str, config: Config | None) -> str:
    """Return a path relative to the rootdir in posix form, or as it is outside it."""
    rootpath = getattr(config, "rootpath", None)
    if rootpath is not None:
        try:
            return Path(_file_key(file_path)).relative_to(_file_key(rootpath)).as_posix()
        except ValueError:
            pass
    return str(file_path)


def _forget(file_path: Path) -> None:
    """Remove the modules a failed import of ``file_path`` left in sys.modules."""
    key = _file_key(file_path)
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and os.path.basename(filename) == file_path.name and _file_key(filename) == key:
            del sys.modules[name]


def _ambiguous_message(name: str, test_path: Path, candidates: list[Registration]) -> str:
    """Describe a name that several folders register, none of them above the test."""
    where = "\n".join(f"  {_describe_factory(r.factory)}" for r in candidates)
    return (
        f"Strategy '{name}' is not registered in {test_path.parent} or a folder above it, "
        f"and several other folders register it:\n{where}\n"
        "Register it in a folder above the test, or pass the factory itself: "
        "@strategy(factory)."
    )


def visible_names(directory: str, rootpath: str | None) -> list[str]:
    """Return the strategy names a test in ``directory`` can use, sorted."""
    visible = []
    for name in registry.names():
        registrations = registry.registrations(name)
        if any(_on_path(r, directory) for r in registrations):
            visible.append(name)
            continue
        outside = [r for r in registrations if not _inside(r, rootpath)]
        if len(outside or registrations) == 1:
            visible.append(name)
    return sorted(visible)


def strategy_not_found_message(name: str, directory: str, rootpath: str | None) -> str:
    """
    Describe a strategy name no registration answers, for a test in ``directory``.

    Lists the names the test can use, the closest of them, and the strategy
    files that failed to load, skipped themselves or were not imported.
    """
    available = visible_names(directory, rootpath)
    message = (
        f"Strategy '{name}' not found. "
        f"Available strategies: {available if available else 'none'}"
    )
    close = difflib.get_close_matches(name, available, n=3)
    if close:
        message += ". Did you mean " + " or ".join(repr(c) for c in close) + "?"
    # Strategy files that failed to load are the likely cause
    if runtime.load_errors:
        message += "\nStrategy files that failed to load:"
        for path, error in runtime.load_errors:
            message += f"\n  {path}: {error}"
        # A common cause: an import that only works once pytest has collected
        # another test module (e.g. added its directory to sys.path)
        message += (
            "\nStrategy files are imported when pytest first collects a test module in "
            "their folder or below, so an import that only works after other test "
            "modules are collected fails."
        )
    # So are strategy files that skipped themselves (pytest.importorskip)
    if runtime.skipped_files:
        message += "\nStrategy files that were skipped:"
        for path, reason in runtime.skipped_files:
            message += f"\n  {path}: {reason}"
    # And files named like strategy files that register another way (a plain
    # call): discovery never imported them
    if runtime.unimported_files:
        message += (
            "\nFiles matching a strategy file name that were not imported because "
            "they contain no registration decorator (use @register(...)):"
        )
        for path in runtime.unimported_files:
            message += f"\n  {path}"
    return message


def _registration_count() -> int:
    """Count the registrations: a name registered in two folders counts twice."""
    return sum(len(registry.registrations(name)) for name in registry.names())


def _summary_lines(resolutions: list[Any]) -> list[str]:
    """Summarize the resolved strategies for -v: tests and rows per strategy."""
    by_strategy: dict[tuple[str, str], list[Any]] = {}
    for resolution in resolutions:
        by_strategy.setdefault((resolution.strategy, resolution.where), []).append(resolution)
    lines = []
    for (name, where), entries in sorted(by_strategy.items()):
        directed = sum(e.directed for e in entries)
        random_rows = sum(e.random for e in entries)
        test_rows = sum(e.test for e in entries)
        rows = f"{directed} directed, {random_rows} random"
        if test_rows:
            rows += f", {test_rows} test"
        sources = sorted({f"{e.nsamples} from {e.source}" for e in entries if e.source})
        line = f"{name} ({where}): {len(entries)} test(s), {rows} rows"
        if sources:
            line += "; nsamples=" + ", ".join(sources)
        lines.append(line)
    return lines


# Plugin instance
_plugin_instance = PytestStrategyPlugin()


def _nsamples_type(value: str) -> int | str:
    """
    Parse the --nsamples value: "auto" (any case) or an integer >= 0.

    Args:
        value: Raw command-line value

    Returns:
        "auto" or the number of samples

    Raises:
        argparse.ArgumentTypeError: For any other value, so pytest reports a
            usage error up front instead of a collection error in every module
            that uses a strategy (or silently skipping every test for n < 0)
    """
    if value.strip().lower() == "auto":
        return "auto"

    error = f"expected an integer >= 0 or 'auto', got {value!r}"
    try:
        nsamples = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(error) from None
    if nsamples < 0:
        raise argparse.ArgumentTypeError(error)
    return nsamples


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    """Add the plugin's hooks (``pytest_strategies_context``)."""
    from . import hookspecs

    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for the plugin."""
    group = parser.getgroup("pytest-strategies", "Pytest Strategies Plugin Options")

    group.addoption(
        "--rng-seed",
        type=int,
        default=None,
        help="Set the random seed for reproducible test generation",
    )

    group.addoption(
        "--nsamples",
        action="store",
        type=_nsamples_type,
        default=None,
        help=(
            "Number of random samples to generate per strategy (per Series/RNGSequence "
            "combination for a strategy with per_sequence_samples=True), or 'auto' to "
            "enumerate Series/RNGSequence combinations (strategies without them use their "
            "own count)"
        ),
    )

    group.addoption(
        "--vector-mode",
        action="store",
        type=str,
        default="all",
        choices=["all", "random_only", "directed_only", "mixed", "test"],
        help="Vector generation mode: all, random_only, directed_only, mixed, or test",
    )

    group.addoption(
        "--vector-name",
        action="store",
        type=str,
        default=None,
        help="Run only the directed vector with this name",
    )

    group.addoption(
        "--vector-index",
        action="store",
        type=int,
        default=None,
        help="Run only the directed vector at this index",
    )

    group.addoption(
        "--list-strategies",
        action="store_true",
        default=False,
        help="List all registered strategies and exit",
    )

    parser.addini(
        "strategies_max_exhaustive",
        "Most rows --nsamples=auto (or per_sequence_samples=True) may generate for one "
        "strategy (default: 100000); Parameter(max_exhaustive=...) overrides it",
        default="100000",
    )


def pytest_configure(config: Config) -> None:
    """Register the plugin instance and open a runtime session for this config."""
    # --list-strategies prints from pytest_collection_finish and exits. Under
    # pytest-xdist only the workers collect, and a worker's exit crashes the
    # controller (INTERNALERROR), so list in-process instead, as xdist does for
    # --collect-only. This runs before xdist's trylast pytest_configure starts
    # the distributed session.
    if config.option.list_strategies and getattr(config.option, "dist", "no") != "no":
        config.option.dist = "no"
        if hasattr(config.option, "tx"):
            config.option.tx = []

    if not hasattr(config, "_strategy_plugin_instance"):
        # Push the session state BEFORE registering, so the instance's hooks
        # have a current state.
        runtime.push(config)
        # Config does not declare this attribute, so the type checker needs setattr
        setattr(config, "_strategy_plugin_instance", _plugin_instance)  # noqa: B010
        config.pluginmanager.register(_plugin_instance, "pytest-strategies")


def pytest_unconfigure(config: Config) -> None:
    """Unregister the plugin instance and close this config's runtime session."""
    if hasattr(config, "_strategy_plugin_instance"):
        config.pluginmanager.unregister(_plugin_instance, "pytest-strategies")
        delattr(config, "_strategy_plugin_instance")
        runtime.pop()
        if runtime.current is None and _strategy_file_finder in sys.meta_path:
            sys.meta_path.remove(_strategy_file_finder)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: Session) -> None:
    """
    Handle --list-strategies option to list all registered strategies and exit.
    """
    config = session.config

    if config.option.list_strategies:
        _plugin_instance.load_all_strategy_files(config)
        terminalreporter = config.pluginmanager.get_plugin("terminalreporter")

        if terminalreporter:
            terminalreporter.section("Registered Strategies")

            if registry:
                terminalreporter.write_line(
                    f"\nFound {_registration_count()} registered strategies:\n"
                )

                for name in sorted(registry.names()):
                    registrations = registry.registrations(name)
                    if len(registrations) == 1:
                        terminalreporter.write_line(f"  ✓ {name}")
                        continue
                    # The same name in several folders: say where each one is
                    places = sorted(
                        _relative(r.file, config) if r.file else "<unknown>" for r in registrations
                    )
                    for where in places:
                        terminalreporter.write_line(f"  ✓ {name} ({where})")

                terminalreporter.write_line("")
            else:
                terminalreporter.write_line("\nNo strategies registered.\n")

        # Exit pytest without running tests. Like --collect-only, report
        # collection errors (e.g. a factory that raised) with a non-zero code.
        returncode = pytest.ExitCode.INTERRUPTED if session.testsfailed else pytest.ExitCode.OK
        pytest.exit("Strategy listing complete", returncode=returncode)
