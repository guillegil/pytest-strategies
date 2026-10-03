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
        @Strategy.register("channels")
        def channels(nsamples, ctx):
            return Parameter(TestArg("channel", rng_type=RNGSequence(ctx.channels)))

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
    runs around the others, and can change their answer.

    A factory registered in one folder and used by a test in another gets the
    context of the test's folder.

    Each implementation is called at most once per session, the first time a
    folder that needs it asks, and only for a factory with a ``ctx`` parameter:
    the folders that end at the same implementation share its result. Factories
    without a ``ctx`` parameter never trigger it. When nothing answers, ``ctx``
    keeps its default (or a value bound with ``functools.partial``), and is
    ``None`` without one. Random draws in an implementation come from a stream
    derived from the seed, started anew for each implementation, and do not
    change any test's vectors. Under pytest-xdist every worker calls the
    implementations it needs, and the results must be the same in all of them.

    An exception raised by an implementation fails the collection of each module
    whose folder asks it before any other implementation answers, and that uses
    a factory with ``ctx``, with a message naming the strategy; ``pytest.fail()``
    is reported as it is, and ``pytest.skip(..., allow_module_level=True)``
    skips those modules.

    Args:
        config: The pytest config object.

    Returns:
        Any object, or ``None`` to let the next implementation answer.
    """
