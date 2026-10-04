"""
The public decorators, ``register`` and ``strategy``, and the ``Strategy`` facade.

``@register("name")`` records a factory in the registry. ``@strategy(...)`` only
marks a test: the plugin resolves the strategy when pytest generates the test's
parametrization (``pytest_generate_tests``), with the session's options at hand.
``export_strategies()`` and ``get_context()`` read the running session.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from ._export import document
from ._introspection import PYTEST_FIXTURES as _PYTEST_FIXTURES
from ._registry import Factory, RegistryView, _describe_factory, registry
from ._runtime import runtime
from ._warnings import PytestStrategiesWarning

_F = TypeVar("_F", bound=Callable[..., Any])


def register(name: str) -> Callable[[_F], _F]:
    """
    Register a strategy factory under ``name``.

    The name is visible to the tests in the directory of the file that defines
    the factory and below it; the nearest registration wins, as for fixtures in
    ``conftest.py`` files. A test elsewhere finds it when no other directory
    registers the same name.

    Registering a name twice in the same directory replaces the first factory,
    warns with :class:`PytestStrategiesWarning`, and fails the run with a usage
    error when it happens while pytest collects tests.

    Usage::

        @register("my_strategy")
        def my_strategy(nsamples):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)))
    """
    if not isinstance(name, str):
        raise TypeError(f"register() takes a strategy name, got {name!r}")

    def decorate(fn: _F) -> _F:
        replaced = registry.add(name, fn)
        if replaced is not None and replaced.origin != registry.registrations(name)[-1].origin:
            message = (
                f"Strategy '{name}' is registered twice in the same folder: "
                f"{_describe_factory(fn)} replaces {_describe_factory(replaced.factory)}"
            )
            # Recorded first: the warning may be turned into an error
            runtime.record_clash(message)
            warnings.warn(message, PytestStrategiesWarning, stacklevel=2)
        return fn

    return decorate


def strategy(name: str | Factory, *, validate_signature: bool = True) -> Callable[[_F], _F]:
    """
    Parametrize a test with a strategy.

    Args:
        name: The name a factory was registered under, or the factory itself
            (registered or not)
        validate_signature: Check that the test, or a fixture it uses, takes the
            strategy's arguments (keyword-only). It does not change whether the test
            receives the row as a dataclass.

    The strategy is resolved when pytest collects the test, and its factory must
    return a :class:`Parameter`. The test takes the strategy's arguments by name,
    or, when neither the test nor a fixture it uses asks for one of them, one
    parameter annotated with a dataclass whose fields are those arguments; any
    other parameter is a fixture.

    Usage::

        @strategy("my_strategy")
        def test_x(x):
            ...

        @strategy(my_strategy)
        def test_y(x):
            ...
    """
    if not isinstance(name, str) and not callable(name):
        raise TypeError(f"strategy() takes a strategy name or a factory, got {name!r}")
    # with_args: a callable given as the only argument must be stored, not decorated
    mark = pytest.mark.strategy.with_args(name, validate_signature=validate_signature)

    def decorate(test_fn: _F) -> _F:
        marked: _F = mark(test_fn)
        return marked

    return decorate


def export_strategies(*, format: str = "json") -> str:
    """
    Export every registered strategy as a JSON document of schema 1.

    The document is ``{"schema": 1, "kind": "strategies", "generator": {"name":
    "pytest-strategies", "version": ...}, "seed": ..., "nsamples": ...,
    "strategies": [...]}``: the run's seed and ``--nsamples`` value (10 without it),
    and one entry per registration, a name registered in two folders included,
    sorted by name and folder. An entry has the strategy's ``name``, its ``origin``
    (``folder`` and ``file`` relative to the rootdir in posix form, or absolute
    outside it; ``qualname``; ``line``), its ``context`` and one of:

    - ``parameter``: the factory's ``Parameter.to_dict()``, with
      ``"enabled": false`` for the constraints the run turns off in it;
    - ``error``: ``{"type": "RuntimeError", "message": "boom"}``, what the factory
      raised, or the plugin's own error (a signature it cannot call, a result that
      is not a Parameter), with a ``note`` when the plugin adds one;
    - ``unavailable``: why its folder's context cannot be known (below).

    Values keep their type: JSON for None, bool, int, str and finite floats,
    ``{"$float": "nan"}``, ``{"$enum": "Color", "member": "RED"}``, and
    ``{"$repr": ..., "$type": ...}`` for anything else. The text has no NaN or
    infinity, so ``json.loads`` reads it strictly. Readers ignore keys, and values
    of ``kind`` and ``source``, they do not know: 4.x may add them within schema 1.

    In a pytest session, every strategies file is loaded first. Each factory is
    called once with the inputs it declares, as at collection, and gets the
    session's options for its strategy: its ``nsamples`` is the ``--nsamples``
    value, ``"auto"``, or 10 without the option or outside a session. The context
    hook runs only for a factory that declares ``ctx``, which receives the context
    of the folder its file is in, the one a test there gets (the rootdir's for a
    file outside the rootdir or in an installed package); its entry's ``context``
    is that context's fingerprint (null for a factory without ``ctx``, or a context
    of None). When pytest has not loaded a ``conftest.py`` of that folder or of a
    folder between it and the rootdir (no test there was collected), that context
    cannot be known: such a factory is not called, and its entry has
    ``"unavailable": "tests/b/conftest.py was not loaded in this session"``.
    Each call draws from a random stream of its own, keyed by the run's seed, the
    strategy's name and where its factory is defined: the name of the factory's
    module, the same wherever a package is installed or checked out, or for a
    factory in a strategy file, a test module or a ``conftest.py`` that pytest or
    the plugin imports by its path, the file's folder relative to the rootdir.
    Such a file is one inside the rootdir or below a ``testpaths`` entry, whatever
    the command line names, or one the session imported by its path (a test
    module collected from a folder named on the command line), unless, with
    ``consider_namespace_packages`` false (pytest's default), it is a module of a
    regular package that ``sys.modules`` holds under its package name
    (``acme.strategies``, the name pytest then imports it under in every import
    mode).
    A module named like one elsewhere (a library's ``extacme/strategies.py`` on
    ``sys.path``) is keyed by its module's name, and so is every module outside a
    session, unless ``sys.modules`` does not have it under that name (then its
    folder's absolute path). ``""`` when the factory's code has neither a file nor
    a module.

    Limitations follow (docs/dev.md): a package module named like a strategy file
    or a test module inside the rootdir (``src/acme/strategies.py``) is keyed by
    its folder in a checkout and by its module's name installed, so the two draw
    other values (renaming it avoids that). And outside the rootdir and the
    testpaths, these are keyed by their folder in a run that collects it and by
    their module's name in a run that does not: (a) a helper named like one, not
    in a regular package, imported by its name; (b) a regular package's module
    that ``sys.modules`` holds only under a longer namespace-package name
    (``ns.acme.strategies``); (c) with ``consider_namespace_packages = true``,
    also a regular package's module imported by its name. Listing the folder in
    ``testpaths`` keys them by their folder in every run.

    Args:
        format: Export format (currently only "json" is supported), keyword-only

    Returns:
        The document, as JSON text
    """
    if format != "json":
        raise ValueError(f"Unsupported format: {format}")
    return json.dumps(document(), indent=2, allow_nan=False)


def get_context(config: pytest.Config, path: str | os.PathLike[str]) -> Any:
    """
    Return the testbench context of the folder of ``path`` (a file, or a folder):
    the object a test there gets, the one its strategy factories receive as ``ctx``.

    It is computed from the ``pytest_strategies_context`` implementations that folder
    sees, as for a test there, and kept for the session: every plugin's, and those of
    the loaded ``conftest.py`` files in the folder and above it. A folder whose
    ``conftest.py`` pytest has not loaded (no test there was collected) gets the
    context of the nearest loaded one above it. A relative path is taken from the
    current directory. Typical use, in a folder's ``conftest.py``::

        @pytest.fixture(scope="session")
        def tb(request):
            return Testbench(get_context(request.config, __file__))

    Called while ``conftest.py`` files are imported, or in ``pytest_configure``, it
    sees only the ones loaded so far: call it from fixtures or hooks.

    Args:
        config: The running session's config (``request.config`` in a fixture, or
            a hook's ``config``)
        path: A file or a folder, such as ``__file__``

    Returns:
        The context, or None when no implementation answers for the folder

    Raises:
        RuntimeError: When no pytest-strategies session runs with ``config``
        Whatever the implementation that answered for the folder raised (an
        exception, ``pytest.skip``, ``pytest.fail``), as it is, again on every call
    """
    state = runtime.session_of(config)
    if state is None:
        raise RuntimeError(
            "get_context() was given the config of no running pytest-strategies session: "
            "pass the config of the running session (request.config in a fixture, or a "
            "hook's config), with the pytest-strategies plugin loaded"
        )
    return state.path_context(path)()


class Strategy:
    """
    The 2.x entry point, kept as a namespace.

    ``Strategy.register`` and ``Strategy.strategy`` are :func:`register` and
    :func:`strategy`.
    """

    # name -> the factory registered last under it (2.x compatibility)
    _registry: RegistryView = RegistryView(registry)

    # Common pytest fixtures to exclude from signature validation
    PYTEST_FIXTURES: set[str] = set(_PYTEST_FIXTURES)

    register = staticmethod(register)
    strategy = staticmethod(strategy)
    export_strategies = staticmethod(export_strategies)
