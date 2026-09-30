"""
Pytest plugin for pytest-strategies with auto-discovery of strategy definitions.
"""

import argparse
import fnmatch
import glob
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import os
import sys
from collections.abc import Sequence
from pathlib import Path, PurePath
from types import ModuleType

import pytest
from pytest import Config, Session

from ._runtime import runtime
from ._warnings import PytestStrategiesWarning

# Monotonic counter so every load gets a unique module name. Without this, two
# strategy files sharing a relative path (e.g. re-running pytest in-process with
# a rewritten strategies.py) would collide in sys.modules and the second file's
# registrations would be silently dropped by the import cache.
_load_counter = itertools.count()


def _file_key(path: "str | os.PathLike[str]") -> str:
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

    Running a strategy file a second time (e.g. a test module imports it for an
    Enum) would redraw the values it draws at import time, from a random state
    that depends on the tests collected before, and would silently replace its
    registrations; the test would also get a second copy of its classes. The
    file is matched by its real path, so another file with the same name is
    imported as usual. pytest's assertion rewriting hook comes first, so a test
    module that is also a strategy file is still rewritten and collected.
    """

    def find_spec(self, fullname, path, target=None):
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
    Pytest plugin for strategies with auto-discovery.

    This plugin automatically discovers and loads strategy definition files
    from the test directory before tests run.

    Per-session state (discovery flag, discovered files, active config) lives in
    the shared ``runtime`` object so it can be reset on ``pytest_unconfigure``.
    """

    # ==== CONFIGURATION HOOKS ====

    @pytest.hookimpl
    def pytest_configure(self, config: Config) -> None:
        """
        Configure the plugin and set up Strategy class with config.
        """
        from .rng import RNG
        from .strategy import Strategy

        # Get CLI options
        rng_seed = config.getoption("--rng-seed", None)

        # Configure Strategy and RNG
        Strategy.set_config(config)

        workerinput = getattr(config, "workerinput", None)
        worker_seed = workerinput.get("pytest_strategies_seed") if workerinput else None
        if rng_seed is None and worker_seed is not None:
            # A pytest-xdist worker without --rng-seed uses the controller's seed
            # (see pytest_configure_node). Workers must generate identical vectors,
            # or xdist aborts with "Different tests were collected". Like an
            # unseeded run, it leaves the global random state alone: workers of a
            # project without strategy files keep their own entropy.
            RNG._seed = worker_seed
        else:
            # An explicit --rng-seed also seeds the global random state. Without
            # one, the seed is kept and the random state is left untouched, so a
            # project that seeds random itself (e.g. in a conftest) is unaffected;
            # pytest_sessionstart starts it from the seed before loading strategy
            # files.
            RNG.seed(rng_seed)

        # Register custom markers
        config.addinivalue_line("markers", "strategy(name): mark test to use a specific strategy")

    @pytest.hookimpl
    def pytest_unconfigure(self, config: Config) -> None:
        """Clean up when pytest is unconfiguring."""
        pass

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node) -> None:
        """
        Send the controller's RNG seed to a pytest-xdist worker.

        Optional hook: only called when pytest-xdist is installed.
        """
        from .rng import RNG

        node.workerinput["pytest_strategies_seed"] = RNG.get_seed()

    # ==== SESSION HOOKS ====

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session: Session) -> None:
        """
        Auto-discover and load strategy definition files before tests run.

        This hook runs once at the start of the test session and discovers
        all strategy definition files in the test directory.
        """
        if runtime.strategies_loaded:
            return

        from .rng import RNG

        config = session.config
        norecursedirs = config.getini("norecursedirs")

        # Discover and load strategy files
        search_paths = self._search_paths(config, norecursedirs)
        unimported: list[Path] = []
        strategy_files = self._discover_strategy_files(search_paths, norecursedirs, unimported)
        # Not reported now: most such files are unrelated modules. Strategy.strategy
        # lists them in any "Strategy not found" error.
        for file_path in unimported:
            runtime.record_unimported_file(file_path)

        if strategy_files:
            # Start the global random state from the seed, so draws made when a
            # strategy file is imported (e.g. OFFSET = RNG.integer(...)) are
            # reproduced by --rng-seed=<printed seed>. Only here: a run without
            # strategy files keeps the random state it had.
            RNG.refresh_seed()
            self._load_strategy_files(strategy_files, config)
            runtime.strategies_loaded = True

    @pytest.hookimpl
    def pytest_sessionfinish(self, session: Session, exitstatus: int) -> None:
        """Called at the end of the test session."""
        pass

    # ==== COLLECTION HOOKS ====

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session: Session, config: Config, items: list) -> None:
        """
        Fail the run when --vector-name/--vector-index matched no strategy at all.

        A strategy without the requested directed vector gets an empty parameter
        set, so its tests are skipped. That is intended when another strategy has
        the vector, but when none has it (a typo, an index out of range) every
        test would be skipped and the run would still pass.
        """
        message = self._vector_filter_error(config)
        if message is None:
            return
        if getattr(config, "workerinput", None) is None:
            raise pytest.UsageError(message)
        # A pytest-xdist worker: an exception here kills the worker, and the
        # controller fails with an INTERNALERROR that hides the message. Run
        # nothing instead; the controller stops the session with this message.
        items.clear()
        session.shouldfail = message

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

    # ==== REPORTING HOOKS ====

    @pytest.hookimpl
    def pytest_report_header(self, config: Config, start_path: Path) -> list[str]:
        """Add information to the test report header."""
        from .strategy import Strategy

        lines = []

        # Show RNG seed
        from .rng import RNG

        seed = RNG.get_seed()
        lines.append(f"pytest-strategies: RNG seed = {seed}")

        # Show number of registered strategies
        num_strategies = len(Strategy._registry)
        if num_strategies > 0:
            lines.append(f"pytest-strategies: {num_strategies} strategies registered")

            # Show discovered strategy files
            if runtime.discovered_files:
                lines.append(
                    f"pytest-strategies: Loaded {len(runtime.discovered_files)} strategy file(s)"
                )

        return lines

    @pytest.hookimpl
    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: Config) -> None:
        """Add a section to the terminal summary reporting."""
        from .strategy import Strategy

        # Only show if verbose mode
        if self._verbosity(config) >= 1:
            terminalreporter.section("Strategy Summary")

            if Strategy._registry:
                terminalreporter.write_line(f"Registered strategies: {len(Strategy._registry)}")

                if self._verbosity(config) >= 2:
                    # Show all strategy names in very verbose mode
                    for name in sorted(Strategy._registry.keys()):
                        terminalreporter.write_line(f"  - {name}")
            else:
                terminalreporter.write_line("No strategies registered")

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

        Only files that contain @Strategy.register are returned, so that
        unrelated modules that happen to have such a name are not imported.
        Directories below a search path are skipped as described in
        _skip_directory; symlinked directories are followed, as pytest's
        collection does. Files are returned in a deterministic order: one search
        path after the other, sorted by path within each, and a file reached
        through two paths only once.

        Args:
            search_paths: List of paths to search
            norecursedirs: Directory patterns not to descend into
            unimported: If given, the files matching a pattern that are not
                returned because they contain no @Strategy.register, but do
                mention register (an alias or a plain call), are appended to
                it, in the same order

        Returns:
            List of discovered strategy file paths
        """
        strategy_files: list[Path] = []
        without_registration: list[Path] = []
        seen_files: set[str] = set()

        patterns = [
            "strategies.py",
            "strategy.py",
            "*_strategies.py",
            "*_strategy.py",
        ]

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
        Check if a file contains @Strategy.register decorators.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file contains strategy registrations
        """
        # Bytes, so a file in another source encoding (a PEP 263 coding line) is found
        return b"@Strategy.register" in self._read_bytes(file_path)

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

        Args:
            strategy_files: List of strategy file paths to load
            config: Pytest config object
        """

        _install_strategy_file_finder()
        for file_path in strategy_files:
            imported = self._imported_module(file_path)
            if imported is not None:
                # Already imported under its own name, with its strategies
                # registered (e.g. conftest.py ran "from my_strategies import Mode"):
                # running it again would make a second copy of its classes, and the
                # tests would get values of the other copy.
                self._record_loaded(file_path, imported, config)
                continue

            # Create a module name from the file path
            module_name = self._create_module_name(file_path, config)

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            except PytestStrategiesWarning as e:
                # Only raised when filterwarnings turns the warning into an error
                # (a strategy name registered twice): fail the run, as a test
                # module does, rather than report the file as not loaded while
                # its replacing registration stays in effect.
                sys.modules.pop(module_name, None)
                raise pytest.UsageError(
                    f"pytest-strategies: {file_path}: {type(e).__name__}: {e}"
                ) from e

            except pytest.skip.Exception as e:
                # pytest.skip() / pytest.importorskip() at module level: the file
                # opted out of this run. pytest's outcome exceptions derive from
                # BaseException; one escaping this hook aborts the whole session.
                sys.modules.pop(module_name, None)
                # Its strategies are missing: Strategy.strategy lists it in any
                # "Strategy not found" error.
                runtime.record_skipped_file(file_path, str(e))
                if self._verbosity(config) >= 1:
                    self._write_line(config, f"Skipped {file_path}: {e}")

            except (Exception, pytest.fail.Exception) as e:
                # Don't fail the test session, but always report the error: its
                # strategies are missing, and Strategy.strategy lists it again in
                # any "Strategy not found" error.
                sys.modules.pop(module_name, None)
                error = f"{type(e).__name__}: {e}"
                runtime.record_load_error(file_path, error)
                self._write_line(
                    config, f"Warning - Failed to load {file_path}: {error}", yellow=True
                )

            else:
                # Outside the try: a problem reporting a successful load must not
                # turn it into a load error.
                self._record_loaded(file_path, module, config)

    def _record_loaded(self, file_path: Path, module: ModuleType, config: Config) -> None:
        """Record a loaded strategy file, so an import of it reuses the module."""
        state = runtime.current
        if state is not None:
            state.strategy_modules[_file_key(file_path)] = module
        runtime.record_discovered_file(file_path)

        # Optionally log in verbose mode
        if self._verbosity(config) >= 2:
            try:
                shown = file_path.relative_to(config.rootpath)
            except ValueError:
                # Outside rootdir, e.g. an absolute testpaths entry
                shown = file_path
            print(f"pytest-strategies: Loaded {shown}")

    @staticmethod
    def _imported_module(file_path: Path) -> ModuleType | None:
        """
        Return a module already imported from this file whose strategies are registered.

        A module left in sys.modules by an earlier in-process session, whose
        registrations that session removed when it ended, does not count.
        """
        from .strategy import Strategy, _unwrap

        key = None
        for factory in Strategy._registry.values():
            namespace = getattr(_unwrap(factory), "__globals__", None)
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
        if terminalreporter is not None:
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


def pytest_addhooks(pluginmanager) -> None:
    """Add the plugin's hooks (``pytest_strategies_context``)."""
    from . import hookspecs

    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser) -> None:
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


def pytest_configure(config):
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
        # Push the session state BEFORE registering, so the instance's
        # pytest_configure (which calls Strategy.set_config) has a current state.
        runtime.push(config)
        config._strategy_plugin_instance = _plugin_instance
        config.pluginmanager.register(_plugin_instance, "pytest-strategies")


def pytest_unconfigure(config):
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
        from .strategy import Strategy

        terminalreporter = config.pluginmanager.get_plugin("terminalreporter")

        if terminalreporter:
            terminalreporter.section("Registered Strategies")

            if Strategy._registry:
                terminalreporter.write_line(
                    f"\nFound {len(Strategy._registry)} registered strategies:\n"
                )

                for name in sorted(Strategy._registry.keys()):
                    terminalreporter.write_line(f"  ✓ {name}")

                terminalreporter.write_line("")
            else:
                terminalreporter.write_line("\nNo strategies registered.\n")

        # Exit pytest without running tests. Like --collect-only, report
        # collection errors (e.g. a factory that raised) with a non-zero code.
        returncode = pytest.ExitCode.INTERRUPTED if session.testsfailed else pytest.ExitCode.OK
        pytest.exit("Strategy listing complete", returncode=returncode)
