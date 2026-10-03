"""
The public decorators, ``register`` and ``strategy``, and the ``Strategy`` facade.

``@register("name")`` records a factory in the registry. ``@strategy(...)`` only
marks a test: the plugin resolves the strategy when pytest generates the test's
parametrization (``pytest_generate_tests``), with the session's options at hand.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

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
    Export the registered strategies' metadata.

    In a pytest session, every strategies file is loaded first. A name
    registered in several directories is reported for the one registered last.
    Each factory is called once with the inputs it declares, as at collection,
    and gets the session's options for its strategy: its ``nsamples`` is the
    ``--nsamples`` value, ``"auto"``, or 10 without the option or outside a
    session. The context hook runs only for a factory that declares ``ctx``.
    Each call draws from a random stream of its own, keyed by the run's seed, the
    strategy's name and where its factory is defined: the name of the factory's
    module, the same wherever a package is installed or checked out, or for a
    factory in a strategy file, a test module or a ``conftest.py`` that pytest or
    the plugin imports by its path, the file's folder relative to the rootdir.
    Such a file is one inside the rootdir or below a ``testpaths`` entry, whatever
    the command line names, or one the session imported by its path (a test
    module collected from a folder named on the command line), unless it is a
    module of a regular package that ``sys.modules`` holds under its package name
    (``acme.strategies``, the name pytest imports it under in every import mode).
    A module named like one elsewhere (a library's ``extacme/strategies.py`` on
    ``sys.path``) is keyed by its module's name, and so is every module outside a
    session, unless ``sys.modules`` does not have it under that name (then its
    folder's absolute path). ``""`` when the factory's code has neither a file nor
    a module.

    Two limitations follow (docs/dev.md): a package module named like a strategy
    file or a test module inside the rootdir (``src/acme/strategies.py``) is keyed
    by its folder in a checkout and by its module's name installed, so the two
    draw other values (renaming it avoids that); and a helper module named like
    one outside the rootdir and the testpaths, not in a regular package and
    imported by its name, is keyed by its folder in a run that collects its folder
    and by its module's name in a run that does not.

    Args:
        format: Export format (currently only "json" is supported), keyword-only

    Returns:
        Serialized string representation of all strategies
    """
    import json

    from ._factory import FactoryInputs, call_factory
    from ._resolver import check_factory_result
    from ._streams import StreamKey, seed_part
    from .plugin import definition_part
    from .rng import _Stream

    if format != "json":
        raise ValueError(f"Unsupported format: {format}")

    runtime.load_all_strategy_files()
    config = runtime.current.config if runtime.current is not None else None
    rootpath = getattr(config, "rootpath", None)
    strategies_data = {}
    for name in registry.names():
        factory = registry.registrations(name)[-1].factory
        # The factory's module's name, or for a strategy file, a test module or a
        # conftest.py its folder, as its file system spells it (see source_part, as
        # for a fixture's key)
        folder = definition_part(factory, config, folder=True)
        stream = StreamKey.root(seed_part(runtime.run_seed()), "export", name, folder)
        try:
            # The session's options for this strategy, the instance collection uses,
            # and the random stream root(S, "export", name, folder) (streams v1)
            with _Stream(stream) as rng:
                inputs = FactoryInputs(
                    options=runtime.strategy_options(name, config),
                    rng=rng,
                    ctx=runtime.strategy_context,
                )
                result = call_factory(name, factory, inputs, rootpath=rootpath)
            param = check_factory_result(name, factory, result)
            strategies_data[name] = param.to_dict()
        except Exception as e:
            strategies_data[name] = {"error": f"Failed to inspect strategy: {str(e)}"}

    return json.dumps(strategies_data, indent=2)


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
