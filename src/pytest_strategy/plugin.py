"""
Pytest plugin for pytest-strategies with auto-discovery of strategy definitions.
"""

import importlib.util
import itertools
import sys
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
        RNG.seed(rng_seed)

        # Register custom markers
        config.addinivalue_line("markers", "strategy(name): mark test to use a specific strategy")

    @pytest.hookimpl
    def pytest_unconfigure(self, config: Config) -> None:
        """Clean up when pytest is unconfiguring."""
        pass

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

        config = session.config

        # Get the root directory for tests
        rootdir = Path(config.rootpath)
        testpaths = config.getini("testpaths")

        # Determine search paths
        search_paths = [rootdir / path for path in testpaths] if testpaths else [rootdir]

        # Discover and load strategy files
        strategy_files = self._discover_strategy_files(search_paths)

        if strategy_files:
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
        if config.option.verbose >= 1:
            terminalreporter.section("Strategy Summary")

            if Strategy._registry:
                terminalreporter.write_line(f"Registered strategies: {len(Strategy._registry)}")

                if config.option.verbose >= 2:
                    # Show all strategy names in very verbose mode
                    for name in sorted(Strategy._registry.keys()):
                        terminalreporter.write_line(f"  - {name}")
            else:
                terminalreporter.write_line("No strategies registered")

    # ==== HELPER METHODS ====

    def _discover_strategy_files(self, search_paths: list[Path]) -> list[Path]:
        """
        Discover strategy definition files in the test directory.

        Looks for files matching these patterns:
        - **/strategies.py
        - **/strategy.py
        - **/test_strategies.py (only if it contains @Strategy.register)
        - **/*_strategies.py
        - **/*_strategy.py

        Args:
            search_paths: List of paths to search

        Returns:
            List of discovered strategy file paths
        """
        strategy_files = []

        patterns = [
            "**/strategies.py",
            "**/strategy.py",
            "**/*_strategies.py",
            "**/*_strategy.py",
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for pattern in patterns:
                for file_path in search_path.glob(pattern):
                    # Skip __pycache__ and hidden directories
                    if any(
                        part.startswith(".") or part == "__pycache__" for part in file_path.parts
                    ):
                        continue

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
            try:
                # Create a module name from the file path
                module_name = self._create_module_name(file_path, config)

                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    runtime.record_discovered_file(file_path)

                    # Optionally log in verbose mode
                    if config.option.verbose >= 2:
                        print(f"pytest-strategies: Loaded {file_path.relative_to(config.rootpath)}")

            except Exception as e:
                # Log error but don't fail the test session
                if config.option.verbose >= 1:
                    print(f"pytest-strategies: Warning - Failed to load {file_path}: {e}")

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
        type=str,
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

        # Exit pytest without running tests
        pytest.exit("Strategy listing complete", returncode=0)
