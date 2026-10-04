"""
Hooks that pytest-strategies adds to pytest.

Implement them in a ``conftest.py`` or a plugin, like any pytest hook.
"""

from typing import Any

import pytest


@pytest.hookspec(firstresult=True)
def pytest_strategies_context(config: pytest.Config) -> Any:
    """
    Return the object passed as ``ctx`` to strategy factories that declare it.

    Use it to build strategies from configuration that is only known when the
    session runs, such as a testbench description read from a file named on the
    command line::

        # conftest.py
        def pytest_addoption(parser):
            parser.addoption("--tb-config")

        @pytest.hookimpl(optionalhook=True)
        def pytest_strategies_context(config):
            return Testbench.parse_config(config.getoption("--tb-config"))

        # strategies.py
        @register("channels")
        def channels(ctx):
            return Parameter(TestArg("channel", rng_type=RNGSequence(ctx.channels)))

        # conftest.py: the testbench fixture builds on the same object
        @pytest.fixture(scope="session")
        def tb(strategies_ctx):
            return Testbench(strategies_ctx)

    ``optionalhook=True`` keeps the ``conftest.py`` usable when pytest-strategies
    is not loaded; without it, pytest stops with "unknown hook".

    Each folder has its own context. A test's factories get the context of the
    test's folder, from the implementations that folder sees: every plugin's that
    is not a ``conftest.py``, and those of the ``conftest.py`` files in the folder
    and in each folder above it. The plugin asks them in this order, and the first
    value that is not None answers:

    1. ``tryfirst`` implementations;
    2. the ``conftest.py`` files, from the test's folder upward;
    3. the other plugins, last registered first;
    4. ``trylast`` implementations.

    Within the ``tryfirst`` and ``trylast`` groups the ``conftest.py`` files come
    first too, from the test's folder upward. So the nearest ``conftest.py`` that
    answers wins, one that returns None defers to the one above, and a plugin
    answers only where no ``conftest.py`` does, whatever order pytest loaded them
    in. One test tree can hold two testbench configurations, one per folder. A
    ``wrapper=True`` (or ``hookwrapper=True``) implementation that a folder sees
    runs around the one that answered, and can change its answer by returning a
    new object (``{**ctx, "extra": 1}``); folders that see the same wrappers and
    get their answer from the same implementation share the wrapped result, and a
    wrapper that returns the very object it received shares the context of the
    folders without it. A wrapper must leave the object it receives as it is,
    since the folders that do not see the wrapper get that object too: when its
    fingerprint (below) changed while the wrappers ran, the folders that see the
    wrapper fail with an error that says so. Each implementation runs once, for
    the first folder that asks, so a wrapper's code before its ``yield`` runs after
    the implementations it wraps, not before them as in other hooks.

    A factory registered in one folder and used by a test in another gets the
    context of the test's folder; ``export_strategies()``, which has no test, gives
    it the context of the folder it is registered in (an entry reads
    ``unavailable`` when a ``conftest.py`` of that folder or above it was not
    loaded in the session). Tests and fixtures get the
    same object: the ``strategies_ctx`` session fixture gives the context of the
    tests that use it (which must share one), and
    ``pytest_strategy.get_context(config, path)`` the context of a folder, for the
    fixtures of that folder's ``conftest.py``.

    Each implementation that is not a wrapper is called at most once per session,
    the first time a folder that needs it asks, and only for a factory with a
    ``ctx`` parameter, ``strategies_ctx`` or ``get_context()``: the folders that
    end at the same implementation share its result. A wrapper runs once for each
    implementation that answers under it, and again for each other set of
    wrappers it is in (a folder whose ``conftest.py`` adds a wrapper). Factories
    without a ``ctx`` parameter never trigger it. When nothing answers, ``ctx``
    keeps its default (or a value bound with ``functools.partial``), and is
    ``None`` without one. Random draws in an
    implementation come from a stream derived from the seed, started anew for
    each implementation, and do not change any test's vectors. Under
    pytest-xdist every worker calls the implementations it needs, and the results
    must be the same in all of them: when two workers' fingerprints of one
    context (below) differ, the run fails with exit code 4, naming the context's
    label (the ``conftest.py`` or plugin that answered).

    When an implementation returns an object, the plugin takes its fingerprint,
    a hash of what the object holds (sets sorted, paths inside the rootdir
    relative to it, a pydantic model's ``Field(exclude=True)`` fields left out).
    It prints it after the collection when the collection computed the context
    (``pytest-strategies: context 976bcfdf``; under pytest-xdist, at the end of
    the run); a context first computed later, by ``strategies_ctx`` or a
    fixture's ``get_context()`` call, appears only in the ``-v`` Contexts block.
    The fingerprint also ends the reproduce line of a failed run whose failed
    tests' factories received the context, and is in ``VectorInfo.context``, so
    two runs can tell whether they received the same context. Keep volatile
    values (temporary paths, process IDs, times) out of the object, or exclude
    them, so that it stays the same from run to run.

    An exception raised by an implementation fails the collection of each module
    whose folder asks it before any other implementation answers, and that uses
    a factory with ``ctx``, with a message naming the strategy, so the module's
    other tests do not run either. For a method of a test class, the class fails
    collection instead, and the module's other tests run. ``pytest.fail()`` is
    reported as it is, and ``pytest.skip(..., allow_module_level=True)`` skips
    those modules or classes. ``strategies_ctx`` and ``get_context()`` raise it
    again as it is, so a skip there skips the tests that use them.

    Args:
        config: The pytest config object.

    Returns:
        Any object, or ``None`` to let the next implementation answer.
    """
