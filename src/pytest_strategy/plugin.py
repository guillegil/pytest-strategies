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
import functools
import glob
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import os
import re
import sys
import traceback
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, get_args

import _pytest.python
import pytest
from _pytest.pathlib import ImportPathMismatchError, import_path
from pytest import Config, Session

from ._context import Answer
from ._fingerprint import UNAVAILABLE, canonical
from ._options import VectorMode, constraint_off_item, parse_constraint_off
from ._registry import (
    STRATEGY_FILE_PATTERNS,
    Registration,
    _contains,
    _describe_factory,
    display_path,
    factory_source,
    matches_pattern,
    registry,
    source_part,
    test_file_patterns,
)
from ._runtime import CtxFixture, runtime
from ._streams import StreamKey, file_part, seed_part
from ._vector import VECTOR_KEY, VECTORS_KEY, VectorInfo
from .rng import RNG, _Stream

# This package's folder, whose frames are left out of the errors shown for a factory
_PACKAGE_DIR = os.path.normcase(os.path.realpath(os.path.dirname(__file__)))

# A test's record parameters that its strategies, in named mode, leave to fixtures:
# (parameter, error) pairs, checked once the test is parametrized
_UNFILLED_RECORDS = pytest.StashKey[list[tuple[str, str]]]()

# The function of the fixture pytest makes for each argument of a parametrization
# (private API; None if a pytest moves it): it returns the argument's value and runs
# no user code, so it gets no random stream
_DIRECT_PARAM_FIXTURE: Any = getattr(_pytest.python, "get_direct_param_fixture_func", None)

# Strategy file names; a file is imported only if it also contains a registration
_STRATEGY_FILE_PATTERNS = STRATEGY_FILE_PATTERNS

# A registration decorator: @Strategy.register(...), or @register("name") and
# @<module>.register("name") with a string literal, also as name="..."
# (functools.singledispatch's @f.register(int) is not one)
_REGISTRATION = re.compile(
    rb"@[ \t]*(?:(?:[A-Za-z_][\w.]*\.)?Strategy\.register[ \t]*\("
    rb"|(?:[A-Za-z_][\w.]*\.)?register[ \t]*\(\s*(?:name\s*=\s*)?[rRuU]?[\"'])"
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
    return matches_pattern(pattern, str(path))


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
    Make an import of a strategy file go through the plugin.

    A test module (or a strategy file) that imports a strategy file the plugin
    has not loaded yet, such as another folder's, gets it loaded by the plugin,
    with the file's own random stream: its import-time draws are then the same
    whichever tests are collected. An import of a file the plugin already
    loaded reuses that module, even when it was loaded under a name of its own
    (its natural name was taken by another file): running the file a second
    time would redraw its import-time values and give the test a second copy of
    its classes. Files are matched by their real path, so another file with the
    same name is imported as usual. pytest's assertion rewriting hook comes
    first, so a test module that is also a strategy file is still rewritten and
    collected.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        state = runtime.current
        if state is None:
            return None
        # Cheap check first: only a module named like a strategy file can be one
        filename = fullname.rpartition(".")[2] + ".py"
        if not any(fnmatch.fnmatch(filename, pattern) for pattern in _STRATEGY_FILE_PATTERNS):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            return None
        key = _file_key(spec.origin)
        module = state.strategy_modules.get(key)
        if module is None and key not in state.loaded_files and state.config is not None:
            # Not loaded, nor being loaded right now (the plugin's own import of
            # the file comes through here too), nor failed: load it now
            file_path = Path(spec.origin)
            if _plugin_instance.loads_on_import(file_path, state.config):
                _plugin_instance.load_strategy_file(file_path, state.config)
                module = state.strategy_modules.get(key)
                if module is not None and sys.modules.get(fullname) is module:
                    # The plugin imported it under this very name. The import
                    # system would then use the module's own spec and run the file
                    # a second time; without the entry it takes the spec below,
                    # which hands back this module and puts the entry back.
                    del sys.modules[fullname]
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
        """
        Seed the RNG and register the strategy marker.

        The module's ``pytest_configure`` registers the instance, so this runs
        after the pytest_configure of the initial conftest.py files and of the
        plugins registered after the module: one that seeds there
        (``RNG.seed(1234)``, or ``config.option.rng_seed = 99``) sets the run's
        seed, as in 3.0.
        """
        # A pytest-xdist worker without --rng-seed uses the controller's seed (see
        # pytest_configure_node): workers must generate identical vectors.
        # Without either, the seed chosen for this process is kept. Either way the
        # generator restarts from it, so values drawn when test modules are
        # imported follow the printed seed. The global random state is not touched.
        RNG.seed(_given_seed(config))
        state = runtime.current
        if state is not None:
            state.run_seed = RNG.get_seed()
        # From now on, a test module that imports a strategy file the plugin has
        # not loaded yet gets it loaded by the plugin (see _StrategyFileFinder)
        _install_strategy_file_finder()

        config.addinivalue_line(
            "markers",
            "strategy(name_or_factory, *, validate_signature=True): parametrize the test with "
            "a strategy (added by @strategy)",
        )

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """
        Send the controller's RNG seed to a pytest-xdist worker.

        Optional hook: only called when pytest-xdist is installed.
        """
        node.workerinput["pytest_strategies_seed"] = _run_seed()

    @pytest.hookimpl
    def pytest_plugin_registered(self, plugin: object, plugin_name: str, manager: Any) -> None:
        """
        Record a ``conftest.py`` that pytest imported by its path in this session, so
        its fixtures and factories are keyed by its path (``definition_part``), also
        outside the rootdir and the testpaths.

        pytest registers a conftest module under its path. The hook is historic:
        when the plugin registers, in ``pytest_configure``, it is called for every
        plugin registered before, the initial conftest.py files among them.
        """
        state = runtime.current
        if state is None or state.config is None or manager is not state.config.pluginmanager:
            return
        file = getattr(plugin, "__file__", None)
        if (
            isinstance(plugin, ModuleType)
            and isinstance(file, str)
            and os.path.basename(file) == "conftest.py"
            and os.path.isabs(plugin_name)
            and _file_key(plugin_name) == _file_key(file)
        ):
            state.imported_files.add(_file_key(file))

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
            self._load_directories(collector.config, collector.path.parent)

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(
        self, collector: pytest.Collector
    ) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
        """
        Collect a test module on a random stream of its own, root(S, "module", path)
        (streams v1), where path is the module's path relative to the rootdir, or
        below its site-packages folder for an installed package's (``file_part``).

        Values the module draws when it is imported (``BASE = RNG.integer(0, 9)`` at
        module level) are then the same whether it is collected alone or with other
        modules, in any order. The module is imported here, after pytest_collectstart
        loaded its folder's strategy files. The key is built only if the module draws.

        The module's path is recorded as one pytest imported by its path, so its
        fixtures are keyed by it (``definition_part``).
        """
        state = runtime.current
        if not isinstance(collector, pytest.Module) or state is None:
            return (yield)
        # pytest imports the module by its path: its fixtures are keyed by it
        # (definition_part), also outside the rootdir and the testpaths
        state.imported_files.add(_file_key(collector.path))
        seed = seed_part(_run_seed())
        path, rootpath = collector.path, collector.config.rootpath
        with _Stream(lambda: StreamKey.root(seed, "module", file_part(path, rootpath))):
            return (yield)

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
        unfilled: list[tuple[str, str]] = []
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
                    fixturenames=metafunc.fixturenames,
                    test_key=metafunc.definition.nodeid,
                    context=runtime.test_context(metafunc.definition),
                )
            except ValueError as e:
                if metafunc.config.getoption("fulltrace", False):
                    raise
                error = f"In {metafunc.function.__name__}: {e}{_user_traceback(e)}"
            except pytest.skip.Exception as e:
                # From the factory or the pytest_strategies_context hook. Escaping
                # this hook would skip the whole module, tests without a strategy
                # included, so that needs allow_module_level=True, as at import time.
                if e.allow_module_level:
                    raise
                error = (
                    f"In {metafunc.function.__name__}: pytest.skip({e.msg!r}) was called "
                    "while the strategy was resolved, which would skip the entire module. "
                    "If that is the intention, pass allow_module_level=True. To skip only "
                    "some tests, use @pytest.mark.skip or @pytest.mark.skipif."
                )
            if error is not None:
                # Reported like pytest's own parametrize errors: the message, and the
                # frames of a factory that raised, without the plugin's
                pytest.fail(error, pytrace=False)
            markers.append(
                pytest.mark.parametrize(parametrization.argnames, parametrization.params()).mark
            )
            unfilled.extend(parametrization.unfilled)
        own_markers[:] = markers
        if unfilled:
            metafunc.definition.stash[_UNFILLED_RECORDS] = unfilled

    @pytest.hookimpl(tryfirst=True)
    def pytest_itemcollected(self, item: pytest.Item) -> None:
        """
        Store the VectorInfo of the strategy rows an item runs: the first one in
        the node ID under ``VECTOR_KEY``, every one under ``VECTORS_KEY``.

        Each row carries a ``strategy`` mark holding its VectorInfo (the test's own
        ``strategy`` marks hold a strategy name or factory), and pytest copies a
        row's marks onto its item in the order of the node ID. This runs as each
        item is collected, before any ``pytest_collection_modifyitems`` hook.
        """
        infos = tuple(
            mark.args[0]
            for mark in item.iter_markers("strategy")
            if mark.args and isinstance(mark.args[0], VectorInfo)
        )
        if infos:
            item.stash[VECTOR_KEY] = infos[0]
            item.stash[VECTORS_KEY] = infos

    @pytest.hookimpl
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """
        Note a collector (a test module, a class) that was skipped, such as a module
        that calls ``pytest.importorskip()``, or failed, such as a module that does
        not import or a strategy whose factory raised: the strategies of its tests
        may not have been resolved.
        """
        state = runtime.current
        if state is not None and (report.skipped or report.failed):
            state.collectors_incomplete = True

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, session: Session, config: Config, items: list[pytest.Item]
    ) -> None:
        """
        Fail the run on a strategy name registered twice in one directory, when
        --vector-name/--vector-index matched no strategy at all, and when a
        --strategy-constraint-off item matched no constraint in a run that
        collected the whole suite.

        A strategy without the requested directed vector gets an empty parameter
        set, so its tests are skipped. That is intended when another strategy has
        the vector, but when none has it (a typo, an index out of range) every
        test would be skipped and the run would still pass. Likewise, a misspelled
        constraint name would leave the constraint on while the user believes it is
        off. A run that collected only some tests resolved only some strategies, and
        so did a run in which a test module was skipped or failed to collect, so
        there the unmatched items are reported in red and the run goes on.
        """
        message = self._clash_error() or self._vector_filter_error(config)
        unmatched = self._constraint_off_error(config)
        if unmatched is not None:
            state = runtime.current
            complete = state is not None and not state.collectors_incomplete
            if complete and _collects_whole_suite(config):
                message = message or unmatched
            elif state is not None:
                # Printed once pytest has reported the collection (on a pytest-xdist
                # worker, by the controller: see pytest_terminal_summary)
                state.unmatched_constraints_off = unmatched.splitlines()
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

    def _constraint_off_error(self, config: Config) -> str | None:
        """
        Describe the --strategy-constraint-off items that matched no constraint of a
        strategy this run resolved.

        An item matches when a resolved strategy it applies to (every strategy for a
        bare name) has a constraint with its name, whether or not the run evaluates
        the constraints.

        Returns:
            The message, or None if every item matched, the option was not given,
            no Parameter strategy was resolved, or under --list-strategies
        """
        state = runtime.current
        if state is None or config.option.list_strategies or not state.constraint_names:
            return None
        known = state.constraint_names
        lines = []
        for target, name in dict.fromkeys(runtime.session_options(config).constraints_off):
            if any(
                name in names
                for strategy, names in known.items()
                if target is None or target == strategy
            ):
                continue
            item = constraint_off_item(target, name)
            # A bare name is compared with the names, an aimed one with the aimed items
            if target is None:
                candidates = sorted({c for names in known.values() for c in names})
            else:
                candidates = sorted({f"{s}:{c}" for s, names in known.items() for c in names})
            line = f"--strategy-constraint-off={item} matched no constraint."
            close = difflib.get_close_matches(item, candidates, n=3)
            if close:
                line += " Did you mean " + " or ".join(repr(c) for c in close) + "?"
            lines.append(line)
        if not lines:
            return None
        by_strategy = "; ".join(
            f"{strategy}: {', '.join(names) if names else 'none'}"
            for strategy, names in sorted(known.items())
        )
        return "\n".join(lines) + f" Constraints by strategy: {by_strategy}"

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session: Session) -> None:
        """
        Print the --strategy-constraint-off items that matched no constraint in a
        run that did not collect the whole suite, in red, after the collection report.

        A pytest-xdist worker's output is not shown: the worker sends the lines
        with its -v summary, and the controller prints them in its terminal summary.
        The worker also notes the digest of each strategy's values, before any test
        can change a value, and the contexts the collection computed, for the
        controller (see ``pytest_sessionfinish``).
        """
        state = runtime.current
        if state is None:
            return
        if getattr(session.config, "workerinput", None) is not None:
            state.value_digests = _value_digests(
                session.items, getattr(session.config, "rootpath", None)
            )
            state.collection_contexts = {
                label: _fingerprint_text(answer)
                for label, answer in _computed_contexts(state).items()
            }
            return
        for line in state.unmatched_constraints_off:
            self._write_line(session.config, line, red=True)

    # ==== RANDOM STREAMS OF FIXTURES AND TESTS ====

    @pytest.hookimpl(wrapper=True)
    def pytest_fixture_setup(
        self, fixturedef: pytest.FixtureDef[Any], request: pytest.FixtureRequest
    ) -> Generator[None, Any, Any]:
        """
        Set a fixture up on a random stream of its own (streams v1): root(S,
        "fixture", scope, name, param_index, where, qualname, base), where scope is
        the node ID of the fixture's scope node ("" for the session and the rootdir's
        package, ``_node_part``), and where, qualname and base tell the fixture from
        another of the same name (``_fixture_definition``, ``_fixture_base``).

        A module- or session-scoped fixture is set up during the setup of whichever
        test needs it first; with its own stream, its draws, and that test's, do
        not depend on which test that is. Its teardown runs in the teardown of a
        test, and draws from that test's teardown stream. The key is built only if
        the fixture draws.
        """
        state = runtime.current
        if fixturedef.func is _DIRECT_PARAM_FIXTURE or state is None:
            return (yield)
        seed = seed_part(_run_seed())
        scope = _node_part(request.node.nodeid)
        param_index = getattr(request, "param_index", 0)

        def key() -> StreamKey:
            definition = state.fixture_definitions.get(fixturedef)
            if definition is None:
                definition = (
                    *_fixture_definition(fixturedef.func, request.config),
                    _fixture_base(fixturedef),
                )
                state.fixture_definitions[fixturedef] = definition
            return StreamKey.root(
                seed, "fixture", scope, fixturedef.argname, param_index, *definition
            )

        # The fixtures the test's request resolved (private API): strategies_ctx
        # resolved during this setup was asked for through request.getfixturevalue(),
        # so a later test that gets this fixture from its cache uses it too
        resolved = getattr(request, "_fixture_defs", None)
        if not isinstance(resolved, Mapping):
            resolved = {_CTX_FIXTURE: None}
        before = _CTX_FIXTURE in resolved
        with _Stream(key):
            result = yield
        if not before and _CTX_FIXTURE in resolved and state.ctx_fixture is not None:
            state.ctx_fixture.requesters.add(fixturedef)
        return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> Generator[None, None, None]:
        """
        Run a test's setup on its own random stream (see ``_phase_stream``), and
        check the context it got from ``strategies_ctx`` (see ``_check_ctx``).
        """
        with _phase_stream(item, "setup"):
            try:
                result = yield
            except BaseException as error:
                _check_ctx(item, error)
                raise
            _check_ctx(item, None)
            return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        """
        Run a test's body on its own random stream (see ``_phase_stream``), and
        check the context it got from ``strategies_ctx`` (see ``_check_ctx``).
        """
        with _phase_stream(item, "call"):
            try:
                result = yield
            except BaseException as error:
                _check_ctx(item, error)
                raise
            _check_ctx(item, None)
            return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> Generator[None, None, None]:
        """Run a test's teardown on its own random stream (see ``_phase_stream``)."""
        with _phase_stream(item, "teardown"):
            return (yield)

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
            # when the test names it. Otherwise the function's (or the class's)
            # qualified name: never a repr with a memory address, which would
            # change the test's random stream from run to run.
            names = registry.names_of(ref)
            name = names[0] if names else factory_source(ref)[1] or type(ref).__qualname__
            return name, ref

        directory = _file_key(test_path.parent)
        if config is not None:
            self._load_directories(config, test_path.parent)
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

    def _load_directories(self, config: Config, folder: Path) -> None:
        """
        Load the strategy files of a test module's folder and each directory above
        it, closest first.

        The directories are its real path and those above it, as the file system
        spells them, not ``_file_key``s: files are imported and reported under their
        own spelling (on Windows, ``_file_key`` lowercases the path). The random
        streams of the folder's own files are keyed by its path as pytest spells it
        (``folder``), which is the same in every checkout for a folder linked into
        the rootdir from a place that does not move with it (``path_part``).
        """
        state = runtime.current
        if state is None:
            return
        directory = os.path.realpath(folder)
        for current in self._directories_up(config, directory):
            key = os.path.normcase(current)
            if key in state.loaded_dirs:
                continue
            state.loaded_dirs.add(key)
            files = self._strategy_files_in(Path(current))
            if files:
                self._load_strategy_files(
                    files, config, spelled=folder if current == directory else None
                )

    def _directories_up(self, config: Config, directory: str) -> Iterator[str]:
        """
        Yield ``directory`` and the directories above it, up to the rootdir or the
        search path that contains it.

        A directory outside all of them (a test given by an absolute path
        elsewhere) yields only itself. The ceilings are ``_file_key``s, so each
        directory is compared by its key and yielded as it is spelled.
        """
        state = runtime.current
        if state is not None and state.ceilings is not None:
            ceilings = state.ceilings
        else:
            ceilings = {_file_key(config.rootpath)}
            ceilings.update(_file_key(p) for p in self._search_paths(config))
            if state is not None:
                state.ceilings = ceilings
        if not any(_contains(ceiling, os.path.normcase(directory)) for ceiling in ceilings):
            yield directory
            return
        current = directory
        while True:
            yield current
            if os.path.normcase(current) in ceilings:
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

    def loads_on_import(self, file_path: Path, config: Config) -> bool:
        """
        Check whether an import of ``file_path`` loads it as a strategy file: it has
        a strategy file name and a registration decorator, in a folder the search
        for strategy files covers (not an installed package, not a virtualenv).
        """
        if not any(fnmatch.fnmatch(file_path.name, pattern) for pattern in _STRATEGY_FILE_PATTERNS):
            return False
        try:
            norecursedirs = config.getini("norecursedirs")
            search_paths = self._search_paths(config, norecursedirs)
        except (AttributeError, ValueError):
            # Not a full pytest config (a unit test's stand-in)
            return False
        parent = file_path.parent
        directories = {Path(os.path.abspath(parent)), Path(os.path.realpath(parent))}
        if not any(
            self._covers(search_path, directory, norecursedirs)
            for search_path in search_paths
            for directory in directories
        ):
            return False
        return self._contains_strategy_registration(file_path)

    def load_strategy_file(self, file_path: Path, config: Config) -> None:
        """Load one strategy file, as when pytest collects a test module next to it."""
        self._load_strategy_files([file_path], config)

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
        """Add the RNG seed, and the constraints turned off, to the test report header."""
        lines = [f"pytest-strategies: RNG seed = {_run_seed()}"]
        items = runtime.session_options(config).constraints_off_items
        if items:
            lines.append(f"pytest-strategies: constraints off: {', '.join(items)}")
        return lines

    @pytest.hookimpl
    def pytest_report_collectionfinish(self, config: Config) -> list[str]:
        """
        Print the fingerprints of the testbench contexts the collection computed,
        after the collection report (also under -q and --collect-only): ``context
        976bcfdf``, or ``contexts conftest.py 976bcfdf, tests/a/conftest.py b1e1b237``
        with each one's label, for a context with several. Nothing when no context
        was computed, or none was other than None.

        A pytest-xdist controller collects nothing, so this does not run there.
        """
        state = runtime.session_of(config)
        if state is None:
            return []
        shown = {label: _fingerprint_text(a) for label, a in _computed_contexts(state).items()}
        text = _contexts_text(shown)
        return [f"pytest-strategies: {text}"] if text is not None else []

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Note a test whose setup or call failed, for the line that reproduces the run."""
        if report.failed and report.when != "teardown":
            state = runtime.current
            if state is not None:
                state.failed_tests.add(report.nodeid)

    @pytest.hookimpl
    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: int, config: Config
    ) -> None:
        """
        Say how to reproduce a failed run, with the contexts the failed tests'
        factories received, and summarize the strategies and the contexts with -v.

        On the pytest-xdist controller, which collects nothing, first print what the
        workers printed after their collection (the contexts, the unmatched
        --strategy-constraint-off items), and then, in red, what the workers
        generated differently (see ``pytest_sessionfinish``).
        """
        state = runtime.current
        summary = state.worker_summary if state is not None else None
        if state is not None and state.worker_summaries:
            # The line the workers printed after their collection
            collected = _merged(
                s.get("collection_contexts", {}) for s in _by_worker(state.worker_summaries)
            )
            text = _contexts_text(collected)
            if text is not None:
                terminalreporter.write_line(f"pytest-strategies: {text}")
        # Unmatched --strategy-constraint-off items a pytest-xdist worker reported
        for line in (summary or {}).get("unmatched_constraints_off", []):
            terminalreporter.write_line(f"pytest-strategies: {line}", red=True)
        if state is not None and state.worker_differences:
            for line in [_DIFFERENT_VECTORS, *state.worker_differences, *_DIFFERENT_VECTORS_HINT]:
                terminalreporter.write_line(line, red=True)
        failed = exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERRUPTED)
        distributed = getattr(config.option, "dist", "no") != "no"
        if failed and state is not None and (state.resolutions or distributed):
            # Exactly the 3.0 line when no failed test's factory received a context
            contexts = _contexts_text(state.failed_contexts)
            suffix = f" ({contexts})" if contexts is not None else ""
            terminalreporter.write_line(
                f"pytest-strategies: reproduce with --rng-seed={_run_seed()}{suffix}"
            )

        if self._verbosity(config) < 1:
            return
        terminalreporter.section("Strategy Summary")
        if summary is None:
            summary = _summary(state)
        if not summary["count"]:
            terminalreporter.write_line("No strategies registered")
        else:
            terminalreporter.write_line(f"Registered strategies: {summary['count']}")
            for line in summary["lines"]:
                terminalreporter.write_line(f"  {line}")
            if self._verbosity(config) >= 2:
                # Show all strategy names in very verbose mode
                for name in summary["names"]:
                    terminalreporter.write_line(f"  - {name}")
        entries: dict[str, str] = summary.get("contexts", {})
        if state is not None and state.worker_summaries:
            # Every worker's: a context computed only when a test ran (through
            # strategies_ctx or get_context()) is in the summary of the worker
            # that ran it
            entries = _merged(s.get("contexts", {}) for s in _by_worker(state.worker_summaries))
        if entries:
            terminalreporter.write_line(f"Contexts: {len(entries)}")
            for label, text in sorted(entries.items()):
                terminalreporter.write_line(f"  {label}: {text}")

    @pytest.hookimpl
    def pytest_sessionfinish(self, session: Session) -> None:
        """
        Note the contexts the failed tests' strategy factories received.

        A pytest-xdist worker sends the controller its -v summary and its part of the
        check that every worker generated the same vectors: the fingerprint of each
        context it computed, by label, and the digest of each strategy's values. The
        controller compares them, and when two workers differ on a context or a
        strategy, the run fails with exit code 4 if it would have passed (or
        collected nothing), and the terminal summary names them. Every worker
        collects every test, so their contexts and values must be the same; names
        in the test IDs no longer make pytest-xdist notice different values.
        """
        state = runtime.current
        if state is not None and state.worker_summaries:
            # The controller: its failed tests ran in the workers
            failed = _merged(
                s.get("failed_contexts", {}) for s in _by_worker(state.worker_summaries)
            )
            state.failed_contexts = dict(sorted(failed.items()))
        elif state is not None and state.failed_tests:
            state.failed_contexts = _failed_contexts(state, session.items)
        workeroutput = getattr(session.config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput["pytest_strategies_summary"] = _summary(state)
            workeroutput[_CHECK] = _check(state)
        if state is None:
            return
        state.worker_differences = _differences(state.worker_checks)
        if state.worker_differences and session.exitstatus in (
            pytest.ExitCode.OK,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        ):
            session.exitstatus = pytest.ExitCode.USAGE_ERROR

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        """
        Keep what a pytest-xdist worker sent when its session finished, by worker ID:
        its summary (the -v summary shown is the first worker's that finished, with
        the Contexts of every worker) and
        its part of the check that the workers generated the same vectors. A worker
        that crashed sent nothing.

        The controller collects nothing, and every worker collects all the tests.
        Optional hook: only called when pytest-xdist is installed.
        """
        state = runtime.current
        output = getattr(node, "workeroutput", None)
        if state is None or not isinstance(output, dict):
            return
        worker = str(node.gateway.id)
        summary = output.get("pytest_strategies_summary")
        if isinstance(summary, dict):
            if state.worker_summary is None:
                state.worker_summary = summary
            state.worker_summaries[worker] = summary
        check = output.get(_CHECK)
        if isinstance(check, dict):
            state.worker_checks[worker] = check

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
        search_paths = _testpaths(config) if config.getini("testpaths") else [Path(config.rootpath)]

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

    def _load_strategy_files(
        self, strategy_files: list[Path], config: Config, spelled: Path | None = None
    ) -> None:
        """
        Load strategy definition files by importing them.

        Each file is imported at most once per session, through pytest's own
        importer with the session's --import-mode, so it gets the module name a
        test module importing it would use. Values a file draws from the RNG when
        it is imported come from a stream of its own, derived from the run's seed
        and its path relative to the rootdir (or below its site-packages folder,
        ``file_part``), so they do not depend on which files were loaded before,
        nor on where the checkout is.

        Args:
            strategy_files: List of strategy file paths to load
            config: Pytest config object
            spelled: The files' folder as pytest spells it, when the files are
                spelled by its real path (see ``_load_directories``)
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

            # Imported on its own random stream, root(S, "file", path) (streams v1)
            stream = _Stream(
                StreamKey.root(
                    seed_part(_run_seed()),
                    "file",
                    file_part(
                        file_path if spelled is None else spelled / file_path.name,
                        getattr(config, "rootpath", None),
                    ),
                )
            )
            try:
                with stream:
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
            # A strategy file of the session: its factories are keyed by its path
            # (definition_part), also outside the rootdir and the testpaths
            state.imported_files.add(_file_key(file_path))
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
            "@strategy(name_or_factory, *, validate_signature=True)"
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
    """
    Return a path relative to the rootdir in posix form, or as it is outside it,
    for messages. The file system's spelling is kept, also on Windows.
    """
    rootpath = getattr(config, "rootpath", None)
    if rootpath is not None:
        shown = display_path(file_path, rootpath)
        if not os.path.isabs(shown):
            return shown
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


def _run_seed() -> int:
    """The seed the current session started from, even if a test reseeded the RNG."""
    return runtime.run_seed()


def _fixture_definition(func: Callable[..., Any], config: Config | None) -> tuple[str, str]:
    """
    Return where a fixture is defined, as two parts of its stream key: its module's
    name, or its file relative to the rootdir in posix form for a conftest.py, a
    test module or a strategy file (``source_part()``), and its function's
    qualified name (``TestDb.conn`` for one defined in a class).

    pytest sets up a fixture that overrides another of the same name (``def
    x(x)`` in a test module, over the conftest's ``x``), and the session fixtures
    of one name in two sibling folders' conftest.py files, for the same scope
    node: their definitions give them streams of their own. A fixture that a
    package defines (a plugin's, or a helper module's) is named by its module, so
    it draws the same whether the package is installed, installed in editable
    mode or checked out next to the tests, unless the module's file is named like
    a test module or a strategy file (``test_utils.py``) and is inside the rootdir,
    below a testpaths entry or imported by its path in the session, which keys it
    by its path (``definition_part``); without ``consider_namespace_packages``, a
    regular package's module that pytest imported under its package name
    (``acme.test_utils``) keeps that name. One whose code has no file (``exec``'d
    code) is named by its module too: its file would resolve against the working
    directory.
    """
    return (definition_part(func, config), factory_source(func)[1] or "")


def definition_part(fn: Callable[..., Any], config: Config | None, *, folder: bool = False) -> str:
    """
    Return ``source_part()`` of a fixture or a factory for the session of ``config``:
    relative to its rootdir, with its ``python_files`` patterns, its testpaths, its
    ``consider_namespace_packages`` value (True without a config) and, for the
    active session's config, the files pytest and the plugin imported by their
    paths in it (``SessionState.imported_files``). A ``conftest.py``, a test module
    or a strategy file is keyed by its path inside the rootdir, below a testpaths
    entry, or when this session imported it by its path, unless, with
    ``consider_namespace_packages`` false, it is a regular package's module that
    ``sys.modules`` holds under its package name; elsewhere (a library on
    ``sys.path`` with ``acme/strategies.py``) by the rules of any other module,
    which give its module's name when ``sys.modules`` has it under that name. The
    limitations this leaves are in ``source_part()``.
    """
    state = runtime.current
    imported = (
        state.imported_files
        if state is not None and config is not None and state.config is config
        else frozenset()
    )
    return source_part(
        fn,
        getattr(config, "rootpath", None),
        folder=folder,
        test_files=test_file_patterns(config),
        testpaths=_ini_testpaths(config),
        imported=imported,
        namespace_packages=_namespace_packages(config),
    )


def _namespace_packages(config: Config | None) -> bool:
    """
    The ``consider_namespace_packages`` ini value of ``config``, or True without a
    config, or for one that does not have it (a unit test's stand-in).
    """
    try:
        value = config.getini("consider_namespace_packages") if config is not None else True
    except (AttributeError, ValueError):
        return True
    return value if isinstance(value, bool) else True


def _testpaths(config: Config) -> list[Path]:
    """
    Return the testpaths ini entries as folders, relative to the rootdir, with glob
    patterns expanded as pytest does (``pkgs/*/tests``).
    """
    rootdir = Path(config.rootpath)
    folders: list[Path] = []
    for entry in config.getini("testpaths"):
        if any(char in entry for char in "*?["):
            # pytest expands wildcards in testpaths (e.g. "pkgs/*/tests")
            matches = sorted(glob.glob(entry, root_dir=rootdir, recursive=True))
            folders.extend(rootdir / match for match in matches)
        else:
            folders.append(rootdir / entry)
    return folders


def _ini_testpaths(config: Config | None) -> list[Path]:
    """``_testpaths()``, or no folders for a config without them (a unit test's stand-in)."""
    try:
        if config is None or not isinstance(config.getini("testpaths"), list):
            return []
        return _testpaths(config)
    except (AttributeError, TypeError, ValueError):
        return []


def _fixture_base(fixturedef: pytest.FixtureDef[Any]) -> str:
    """
    Return where pytest registered a fixture, as a part of its stream key: the
    node ID below which it is visible (``FixtureDef.baseid``), the folder of its
    conftest.py, its test module or class, or ``""`` for a plugin's fixture and
    the rootdir's conftest.py (``_node_part``).

    One fixture function that two conftest.py files import (``from
    helpers.fixtures import port``) is registered twice, and pytest sets both up
    for the session: their bases give them streams of their own.
    """
    # pytest 9 registers a fixture for a node, and derives baseid from it
    node = getattr(fixturedef, "node", None)
    return _node_part(str(node.nodeid if node is not None else getattr(fixturedef, "baseid", "")))


def _node_part(nodeid: str) -> str:
    """
    Return a node ID as a part of a fixture's stream key: ``""`` for the rootdir's
    node (``"."``, a ``Dir``, or a ``Package`` when the rootdir has an
    ``__init__.py``), as for the session, which covers the same tests.

    pytest 9 gives ``"."`` where pytest 8 gives ``""``: as the base of a fixture of
    the rootdir's conftest.py, and as the scope node of a package-scoped fixture
    there when the rootdir is a package (pytest 8 then sets it up for the session).
    """
    return "" if nodeid == "." else nodeid


def _phase_stream(item: pytest.Item, phase: str) -> contextlib.AbstractContextManager[Any]:
    """
    Return the random stream of one phase of a test (``setup``, ``call`` or
    ``teardown``): root(S, "body", nodeid, phase) (streams v1).

    A test body's ``RNG`` draws are then the same alone, in the whole suite, in
    any order and on any pytest-xdist worker, and an ``RNG.seed()`` call in it
    changes nothing after the phase. Without a session there is no stream. The key
    is built only if the phase draws.
    """
    if runtime.current is None:
        return contextlib.nullcontext()
    return _Stream(
        functools.partial(StreamKey.root, seed_part(_run_seed()), "body", item.nodeid, phase)
    )


def _summary(state: Any) -> dict[str, Any]:
    """
    Return what the -v Strategy Summary shows, in types pytest-xdist can send, and
    what the controller prints for a worker: the contexts its collection computed
    and those its failed tests' factories received (label -> fingerprint).
    """
    return {
        "count": _registration_count(),
        "lines": _summary_lines(state.resolutions) if state is not None else [],
        "names": sorted(registry.names()),
        "unmatched_constraints_off": (
            list(state.unmatched_constraints_off) if state is not None else []
        ),
        "contexts": _context_entries(state) if state is not None else {},
        "collection_contexts": dict(state.collection_contexts) if state is not None else {},
        "failed_contexts": dict(state.failed_contexts) if state is not None else {},
    }


def _computed_contexts(state: Any) -> dict[str, Answer]:
    """
    Return the contexts the session computed that are not None (their answers by
    label, sorted by label): what the folders that asked got, errors left out.
    """
    return {
        label: answer
        for label, answer in state.contexts.scopes().items()
        if answer.fingerprint is not None
    }


def _fingerprint_text(answer: Answer) -> str:
    """Return a context's fingerprint, and the types it has by name alone: ``(partial: Plain)``."""
    if not answer.partial:
        return str(answer.fingerprint)
    return f"{answer.fingerprint} (partial: {', '.join(answer.partial)})"


def _contexts_text(fingerprints: Mapping[str, str]) -> str | None:
    """
    Describe contexts given as label -> fingerprint: ``context 976bcfdf`` for one,
    ``contexts conftest.py 976bcfdf, tests/a/conftest.py b1e1b237`` for several
    (sorted by label), None for none.
    """
    if not fingerprints:
        return None
    if len(fingerprints) == 1:
        return f"context {next(iter(fingerprints.values()))}"
    return "contexts " + ", ".join(f"{label} {fp}" for label, fp in sorted(fingerprints.items()))


def _context_entries(state: Any) -> dict[str, str]:
    """
    Describe each context the session computed for the -v summary, by label,
    sorted: its fingerprint and the number of tests whose strategy factories
    received it.
    """
    return {
        label: f"{_fingerprint_text(answer)}, {len(state.context_tests.get(label, ()))} test(s)"
        for label, answer in _computed_contexts(state).items()
    }


def _test_key(item: pytest.Item) -> str | None:
    """
    Return the node ID of an item's test without its parameters (the
    ``metafunc.definition.nodeid`` its strategies were resolved with), or None for
    an item that is not a test function's.
    """
    original = getattr(item, "originalname", None)
    if not isinstance(original, str) or not item.nodeid.endswith(item.name):
        return None
    return item.nodeid[: len(item.nodeid) - len(item.name)] + original


def _failed_contexts(state: Any, items: Sequence[pytest.Item]) -> dict[str, str]:
    """
    Return the contexts the strategy factories of the failed tests among ``items``
    received, as label -> fingerprint, sorted by label.
    """
    labels = {test: label for label, tests in state.context_tests.items() for test in tests}
    found: dict[str, str] = {}
    for item in items:
        if item.nodeid not in state.failed_tests:
            continue
        fingerprint = next(
            (info.context for info in item.stash.get(VECTORS_KEY, ()) if info.context),
            None,
        )
        test = _test_key(item)
        label = labels.get(test) if test is not None else None
        if fingerprint is not None and label is not None:
            found[label] = fingerprint
    return dict(sorted(found.items()))


# What a pytest-xdist worker sends the controller for the check that every worker
# generated the same vectors (workeroutput, strings only)
_CHECK = "pytest_strategies_check"

# The message of that check, before and after the lines that name what differs
_DIFFERENT_VECTORS = "pytest-strategies: the xdist workers generated different vectors:"
_DIFFERENT_VECTORS_HINT = (
    "Make pytest_strategies_context and the strategy factories give the same result "
    "in every worker:",
    "no temporary paths, process IDs, times, unseeded random values or lists built from sets",
    "(leave them out, or use pydantic Field(exclude=True) in a context).",
)


def _value_digests(
    items: Sequence[pytest.Item], rootpath: str | os.PathLike[str] | None = None
) -> dict[str, str]:
    """
    Return the digest of each strategy's values, by strategy name, sorted: the
    first 8 hex characters of the SHA-256 of the canonical JSON text of each of its
    items' node ID and values, in collection order. That is the encoding of the
    context fingerprint (``_fingerprint``), which sorts sets, writes models,
    dataclasses and attrs classes field by field, and leaves out memory addresses,
    so the digest does not depend on a worker's ``PYTHONHASHSEED``. A strategy
    with a value that cannot be encoded (its repr raises) gets ``unavailable``.
    """
    text = canonical(rootpath)
    hashes: dict[str, Any] = {}
    for item in items:
        for info in item.stash.get(VECTORS_KEY, ()):
            digest = hashes.get(info.strategy)
            if digest is None:
                digest = hashes[info.strategy] = hashlib.sha256()
            elif isinstance(digest, str):
                continue
            try:
                row = text([item.nodeid, list(info.values)])
            except Exception:
                hashes[info.strategy] = UNAVAILABLE
                continue
            digest.update(row.encode("ascii") + b"\n")
    return {
        name: digest if isinstance(digest, str) else digest.hexdigest()[:8]
        for name, digest in sorted(hashes.items())
    }


def _check(state: Any) -> dict[str, dict[str, str]]:
    """
    Return a pytest-xdist worker's part of the check that the workers generated the
    same vectors: ``contexts``, each context the session computed by its label (its
    fingerprint, ``none`` where nothing answered, or ``error: <type>`` for an
    implementation that raised), and ``values``, each strategy's digest.
    """
    if state is None:
        return {"contexts": {}, "values": {}}
    contexts = {}
    for label, answer in state.contexts.scopes().items():
        if answer.error is not None:
            contexts[label] = f"error: {type(answer.error).__name__}"
        else:
            contexts[label] = answer.fingerprint if answer.fingerprint is not None else "none"
    return {"contexts": contexts, "values": dict(state.value_digests)}


def _worker_order(worker: str) -> tuple[str, int]:
    """Sort pytest-xdist worker IDs by their number: gw2 before gw10."""
    match = re.fullmatch(r"(.*?)(\d+)", worker)
    return (match.group(1), int(match.group(2))) if match else (worker, -1)


def _by_worker(found: Mapping[str, Any]) -> list[Any]:
    """Return the values of a mapping by pytest-xdist worker ID, in worker order."""
    return [found[worker] for worker in sorted(found, key=_worker_order)]


def _merged(mappings: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Merge mappings: a key keeps the value of the first mapping that has it."""
    merged: dict[str, str] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            merged.setdefault(key, value)
    return merged


def _differences(checks: Mapping[str, Mapping[str, Mapping[str, str]]]) -> list[str]:
    """
    Return a line for each context (by label) and each strategy whose fingerprint or
    digest differs between the pytest-xdist workers that have it, with each one's:
    ``  context tests/a/conftest.py: gw0 1a2b3c4d, gw1 9f8e7d6c``, ``  values of
    strategy burst: gw0 5e6f7a8b, gw1 0c1d2e3f``. A context or a strategy only one
    worker has is not compared: a test that runs on one worker only may compute a
    context through ``strategies_ctx``.

    Args:
        checks: What each worker sent (``_check``), by worker ID
    """
    workers = sorted(checks, key=_worker_order)
    lines = []
    for kind, title in (("contexts", "context"), ("values", "values of strategy")):
        found = {worker: checks[worker].get(kind) or {} for worker in workers}
        for key in sorted({key for values in found.values() for key in values}):
            seen = [(worker, found[worker][key]) for worker in workers if key in found[worker]]
            if len({value for _, value in seen}) > 1:
                each = ", ".join(f"{worker} {value}" for worker, value in seen)
                lines.append(f"  {title} {key}: {each}")
    return lines


def _registration_count() -> int:
    """Count the registrations: a name registered in two folders counts twice."""
    return sum(len(registry.registrations(name)) for name in registry.names())


# The row kinds of the -v summary, in the order it lists them
_SUMMARY_KINDS = ("directed", "random", "test", "exhaustive", "skipped")


def _summary_lines(resolutions: list[Any]) -> list[str]:
    """
    Summarize the resolved strategies for -v: tests and rows of each kind per
    strategy, and for a strategy with constraints the draws each rejected (and, under
    --nsamples=auto, the combinations left out) and the constraints
    --strategy-constraint-off turned off.
    """
    by_strategy: dict[tuple[str, str], list[Any]] = {}
    for resolution in resolutions:
        by_strategy.setdefault((resolution.strategy, resolution.where), []).append(resolution)
    lines = []
    for (name, where), entries in sorted(by_strategy.items()):
        counts = {kind: sum(getattr(e, kind) for e in entries) for kind in _SUMMARY_KINDS}
        # Directed and random rows always, the other kinds when there are some
        rows = ", ".join(
            f"{count} {kind}"
            for kind, count in counts.items()
            if count or kind in ("directed", "random")
        )
        sources = sorted({f"{e.nsamples} from {e.source}" for e in entries if e.source})
        line = f"{name} ({where}): {len(entries)} test(s), {rows} rows"
        if sources:
            line += "; nsamples=" + ", ".join(sources)
        # The constraints turned off, which reject nothing
        off = list(dict.fromkeys(c for e in entries for c in e.constraints_off))
        # The draws each constraint rejected first, over the strategy's tests
        rejected: dict[str, int] = {}
        for e in entries:
            for constraint in e.constraints:
                if constraint not in off:
                    count = e.rejected.get(constraint, 0)
                    rejected[constraint] = rejected.get(constraint, 0) + count
        if rejected:
            line += "; rejected: " + ", ".join(f"{c}={count}" for c, count in rejected.items())
            left_out = [e.left_out for e in entries if e.left_out is not None]
            if left_out:
                combinations = "combination" if sum(left_out) == 1 else "combinations"
                line += f"; left out: {sum(left_out)} {combinations}"
        if off:
            line += "; off: " + ", ".join(off)
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


def _constraint_off_type(value: str) -> str:
    """
    Check one --strategy-constraint-off value: ``ITEM[,ITEM...]`` with
    ``ITEM = [STRATEGY:]NAME``.

    Returns:
        The value as given; the session's options split it into items

    Raises:
        argparse.ArgumentTypeError: For whitespace, an empty item, or an empty
            strategy or name (``:x``, ``x:``), so pytest reports a usage error
            when it parses the command line
    """
    try:
        parse_constraint_off(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return value


def _collects_whole_suite(config: Config) -> bool:
    """
    Check whether the run collected every test, by what it was asked to collect:
    no paths or node IDs on the command line, started from the rootdir (pytest
    collects only the current folder when run without arguments from a folder
    below it), and none of --lf, --sw, --ignore or --ignore-glob.

    A run that asked for every test can still leave strategies unresolved, when a
    test module was skipped or failed to collect: pytest_collectreport notes that.
    """
    source = config.args_source
    if source is Config.ArgsSource.ARGS:
        return False
    # pytest's own rule (Config._decide_args): without arguments, a run started in
    # the rootdir collects testpaths (or the rootdir), and any other run the
    # current folder
    if (
        source is Config.ArgsSource.INVOCATION_DIR
        and config.invocation_params.dir != config.rootpath
    ):
        return False
    narrowing = ("lf", "stepwise", "stepwise_skip", "ignore", "ignore_glob")
    return not any(config.getoption(dest, None) for dest in narrowing)


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Fail the collection of a test written for record mode whose fixtures ask for
    every strategy argument, so the strategy passes them by name, when nothing
    gives its record parameter a value.

    Runs once every other pytest_generate_tests hook has parametrized the test, so
    a parametrization of the parameter counts. pytest would otherwise fail each
    row at setup with "fixture 'p' not found", which says nothing of record mode.
    """
    unfilled = metafunc.definition.stash.get(_UNFILLED_RECORDS, None)
    # The names pytest finds a fixture for at setup, parametrized names included. A
    # pytest without this private attribute leaves the error to setup.
    provided = getattr(metafunc, "_arg2fixturedefs", None)
    if not unfilled or not isinstance(provided, Mapping):
        return
    for param, message in unfilled:
        if param not in provided:
            pytest.fail(f"In {metafunc.function.__name__}: {message}", pytrace=False)


# The name of the plugin's fixture of the testbench context
_CTX_FIXTURE = "strategies_ctx"


@pytest.fixture(scope="session")
def strategies_ctx(request: pytest.FixtureRequest) -> Any:
    """
    The testbench context of the tests that use this fixture: the object their
    strategy factories receive as ``ctx`` from ``pytest_strategies_context``,
    computed if no factory needed it yet, or None when nothing answers.

    It is a session fixture, so a session-scoped ``tb`` fixture can build on it. A
    session fixture has one value, so the tests that use it must share one context:
    when they are in folders whose contexts come from different implementations,
    each of them fails, and a folder with its own pytest_strategies_context uses
    ``pytest_strategy.get_context(request.config, __file__)`` in its conftest.py
    fixtures instead. The tests that use it are those that request it, directly or
    through other fixtures, except those in a folder whose conftest.py defines a
    fixture of that name that does not request this one. When no test requests it
    and one asks for it through ``request.getfixturevalue()``, every test counts.
    When some do, a test that asks for it that way gets their context, and fails
    after its setup or its call when its own folder's context is another one. What
    the implementation raised is raised again as it is, so a ``pytest.skip`` there
    skips the tests that use it.
    """
    state = runtime.session_of(request.config)
    if state is None:
        return None
    # The test that asked first (private API, as the session-scoped request's node
    # is the session), and this fixture's definition (private API)
    item = getattr(request, "_pyfuncitem", None)
    own = getattr(request, "_fixturedef", None)
    items = list(request.session.items)
    consumers = [
        i
        for i in items
        if _CTX_FIXTURE in getattr(i, "fixturenames", ()) and _gets_this_fixture(i, own, False)
    ]
    if item is not None and _CTX_FIXTURE in getattr(item, "fixturenames", ()):
        consumers.append(item)
    elif not consumers:
        # Only asked for through request.getfixturevalue(): any test may use it
        consumers = [i for i in items if _gets_this_fixture(i, own, True)]
        if item is not None:
            consumers.append(item)
    # Otherwise a test asked for it through request.getfixturevalue() while others
    # request it: it gets their context, and is checked after its phase
    # (_ctx_mismatch), whichever of them asked first
    if not consumers:
        return state.path_context()()
    # The tests in one file share their folder's context
    answers: dict[Path, Answer] = {}
    scopes: dict[str, dict[str, None]] = {}
    for consumer in consumers:
        answer = answers.get(consumer.path)
        if answer is None:
            answer = answers[consumer.path] = state.test_context(consumer).answer()
        scopes.setdefault(answer.label, {})[consumer.nodeid] = None
    if len(scopes) > 1:
        pytest.fail(_ctx_scopes_message(scopes), pytrace=False)
    answer = answers[consumers[-1].path]
    state.ctx_fixture = CtxFixture(own, answer.label, scopes)
    return answer.get()


def _gets_this_fixture(item: pytest.Item, own: Any, dynamic: bool) -> bool:
    """
    Whether ``item`` gets the plugin's ``strategies_ctx``, whose definition is
    ``own``: the last definition of the name its folder sees, or one that a
    definition it gets requests in turn (an override in a conftest.py that builds
    on it). A test in a folder whose conftest.py overrides the fixture without
    requesting it gets the override only. Unknown (private API: the item's fixture
    closure, and for a dynamic request the fixture manager's lookup) counts as yes.
    """
    if own is None:
        return True
    info = getattr(item, "_fixtureinfo", None)
    definitions = getattr(info, "name2fixturedefs", {}).get(_CTX_FIXTURE)
    if definitions is None and dynamic:
        manager = getattr(item.session, "_fixturemanager", None)
        try:
            definitions = manager.getfixturedefs(_CTX_FIXTURE, item) if manager else None
        except Exception:
            definitions = None
    if not definitions:
        return True
    for definition in reversed(definitions):
        if definition is own:
            return True
        if _CTX_FIXTURE not in getattr(definition, "argnames", ()):
            return False
    return False


def _ctx_mismatch(item: pytest.Item) -> str | None:
    """
    Return the guard's message when ``item`` used the plugin's ``strategies_ctx``
    without being one of the tests it counted when it was set up, through
    ``request.getfixturevalue()`` (its own, or that of a fixture whose setup asked
    for it), and its folder's context is not the one the fixture returned; None
    otherwise. Private API: the fixtures the test's request resolved
    (``item._request._fixture_defs``).
    """
    state = runtime.session_of(item.config)
    fixture = state.ctx_fixture if state is not None else None
    if state is None or fixture is None:
        return None
    resolved = getattr(getattr(item, "_request", None), "_fixture_defs", None)
    if not isinstance(resolved, Mapping) or any(
        item.nodeid in tests for tests in fixture.consumers.values()
    ):
        return None
    if not (
        _CTX_FIXTURE in resolved and _gets_this_fixture(item, fixture.definition, True)
    ) and not any(definition in fixture.requesters for definition in resolved.values()):
        return None
    label = state.test_context(item).answer().label
    if label == fixture.label:
        return None
    scopes = {other: dict(tests) for other, tests in fixture.consumers.items()}
    scopes.setdefault(label, {})[item.nodeid] = None
    return _ctx_scopes_message(scopes)


def _check_ctx(item: pytest.Item, error: BaseException | None) -> None:
    """
    Fail a test's setup or call that used ``strategies_ctx`` with another folder's
    context (``_ctx_mismatch``): when the phase passed or skipped, with the guard's
    message; when it raised an error, which that context may have caused, the
    message is added to it as a note.
    """
    if isinstance(error, pytest.exit.Exception) or (
        error is not None and not isinstance(error, (Exception, pytest.skip.Exception))
    ):
        # pytest.exit(), a KeyboardInterrupt, or a failure that pytest.fail() or
        # pytest.xfail() reported
        return
    message = _ctx_mismatch(item)
    if message is None:
        return
    if error is None or isinstance(error, pytest.skip.Exception):
        raise pytest.fail.Exception(message, pytrace=False) from None
    if isinstance(error, Exception) and message not in getattr(error, "__notes__", ()):
        error.add_note(message)


def _ctx_scopes_message(scopes: Mapping[str, Mapping[str, None]]) -> str:
    """
    Describe the contexts of the tests that use ``strategies_ctx``: their tests by
    context label, in collection order.
    """
    parts = []
    for label in sorted(scopes):
        first, *others = scopes[label]
        parts.append(f"{label}: {first}" + (f" and {len(others)} more" if others else ""))
    return (
        f"{_CTX_FIXTURE} is a session fixture, but the tests that use it have different "
        f"contexts ({'; '.join(parts)}). In a folder with its own pytest_strategies_context, "
        "use pytest_strategy.get_context(request.config, __file__) in that folder's "
        "conftest.py fixtures."
    )


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
        choices=list(get_args(VectorMode)),
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
        "--strategy-constraint-off",
        action="append",
        type=_constraint_off_type,
        default=None,
        dest="strategy_constraint_off",
        metavar="[STRATEGY:]NAME[,...]",
        help=(
            "Turn a named vector constraint off for this run: NAME in every strategy, "
            "STRATEGY:NAME in that strategy only. Items are separated by commas, and the "
            "option can be repeated"
        ),
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

    parser.addini(
        "strategies_ids",
        "Test IDs of strategy rows: names (the default: directed-zeros, rand-3, "
        "ch=2-rand-1, the same for every seed) or values (the 3.0 format, built from "
        "the row's values); Parameter(ids=...) overrides it",
        type="string",
        default="names",
    )


@pytest.hookimpl(tryfirst=True, specname="pytest_configure")
def pytest_configure_session(config: Config) -> None:
    """
    Open a runtime session for this config.

    It runs before the other plugins' and the conftest.py files' pytest_configure,
    so ``get_context()`` works there (with the conftest.py files loaded so far).
    Until the plugin instance seeds the RNG (``pytest_configure`` below), the
    session's seed is the one ``--rng-seed`` or the pytest-xdist controller gives,
    so a context computed there follows it.
    """
    # A bad ini value stops the run with a usage error (exit code 4), before the
    # session starts
    from ._resolver import check_ids_format

    check_ids_format(config)

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
        state = runtime.push(config)
        state.run_seed = _given_seed(config)
        # Config does not declare this attribute, so the type checker needs setattr
        setattr(config, "_strategy_plugin_instance", _plugin_instance)  # noqa: B010


def pytest_configure(config: Config) -> None:
    """
    Register the plugin instance, which runs its pytest_configure at once (the hook
    is historic): it seeds the RNG.

    It is not tryfirst, so the initial conftest.py files and the plugins registered
    after this module run their pytest_configure before it, as in 3.0: one that
    seeds there (``RNG.seed(1234)``, or ``config.option.rng_seed = 99``) sets the
    run's seed.
    """
    if hasattr(config, "_strategy_plugin_instance") and not config.pluginmanager.is_registered(
        _plugin_instance
    ):
        config.pluginmanager.register(_plugin_instance, "pytest-strategies")


def _given_seed(config: Config) -> int | None:
    """
    Return the run's seed that ``--rng-seed`` gives, or on a pytest-xdist worker
    without it the controller's (``pytest_configure_node``); None when neither does.
    """
    rng_seed: int | None = config.getoption("--rng-seed", None)
    if rng_seed is not None:
        return rng_seed
    workerinput = getattr(config, "workerinput", None)
    worker_seed: int | None = workerinput.get("pytest_strategies_seed") if workerinput else None
    return worker_seed


def pytest_unconfigure(config: Config) -> None:
    """Unregister the plugin instance and close this config's runtime session."""
    if hasattr(config, "_strategy_plugin_instance"):
        if config.pluginmanager.is_registered(_plugin_instance):
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
                        _relative(source, config) if source else "<unknown>"
                        for source in (factory_source(r.factory)[0] for r in registrations)
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
