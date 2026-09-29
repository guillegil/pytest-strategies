"""
Pytest plugin for pytest-strategies with auto-discovery of strategy definitions.
"""

import argparse
import fnmatch
import glob
import importlib.util
import itertools
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pytest import Config, Session

from ._runtime import runtime

# Monotonic counter so every load gets a unique module name. Without this, two
# strategy files sharing a relative path (e.g. re-running pytest in-process with
# a rewritten strategies.py) would collide in sys.modules and the second file's
# registrations would be silently dropped by the import cache.
_load_counter = itertools.count()


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
        strategy_files = self._discover_strategy_files(search_paths, norecursedirs)

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

    @pytest.hookimpl
    def pytest_collection_modifyitems(self, config: Config, items: list) -> None:
        """Modify collected test items if needed."""
        pass

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
            norecursedirs: Directory name patterns the search does not descend into

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
            norecursedirs: Directory name patterns the search does not descend into

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
        names matching the norecursedirs ini patterns (e.g. venv, build,
        node_modules) and virtual environments (a directory containing pyvenv.cfg,
        whatever its name). An installed pytest-strategies has files matching the
        patterns in a virtual environment.

        Args:
            path: The directory
            norecursedirs: Directory name patterns to skip

        Returns:
            True if the directory is skipped
        """
        name = path.name
        return (
            name.startswith(".")
            or name == "__pycache__"
            or any(fnmatch.fnmatch(name, pattern) for pattern in norecursedirs)
            or (path / "pyvenv.cfg").is_file()
        )

    def _discover_strategy_files(
        self, search_paths: list[Path], norecursedirs: Sequence[str] = ()
    ) -> list[Path]:
        """
        Discover strategy definition files in the test directory.

        Looks for files matching these patterns:
        - **/strategies.py
        - **/strategy.py
        - **/test_strategies.py (only if it contains @Strategy.register)
        - **/*_strategies.py
        - **/*_strategy.py

        Directories below a search path are skipped as described in
        _skip_directory. Files are returned in a deterministic order.

        Args:
            search_paths: List of paths to search
            norecursedirs: Directory name patterns not to descend into

        Returns:
            List of discovered strategy file paths
        """
        strategy_files: list[Path] = []

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
            for dirpath, dirnames, filenames in os.walk(search_path):
                directory = Path(dirpath)
                # Prune in place so the walk never enters a skipped directory. Only
                # the directories below the search path are filtered: a project that
                # lives under a dot directory (~/.cache/proj, testpaths = ../shared)
                # or a path given explicitly is searched.
                dirnames[:] = [
                    d for d in dirnames if not self._skip_directory(directory / d, norecursedirs)
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
                # Skip if already found
                if file_path in strategy_files:
                    continue

                # Check if file contains strategy registrations
                if self._contains_strategy_registration(file_path):
                    strategy_files.append(file_path)

        return strategy_files

    def _contains_strategy_registration(self, file_path: Path) -> bool:
        """
        Check if a file contains @Strategy.register decorators.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file contains strategy registrations
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            # Look for @Strategy.register pattern
            return "@Strategy.register" in content or "@strategy.register" in content
        except Exception:
            return False

    def _load_strategy_files(self, strategy_files: list[Path], config: Config) -> None:
        """
        Load strategy definition files by importing them.

        Args:
            strategy_files: List of strategy file paths to load
            config: Pytest config object
        """

        for file_path in strategy_files:
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
        help="Number of random samples to generate per strategy (or 'auto' for exhaustive)",
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
