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

    The hook is called at most once per session, when the first factory with a
    ``ctx`` parameter is called, and its result is reused for every other one.
    Factories without a ``ctx`` parameter never trigger it, and they are called
    as before. When no implementation returns a value, ``ctx`` keeps its default
    (or a value bound with ``functools.partial``), and is ``None`` without one.
    Random draws in the hook come from a stream derived from the seed, and do
    not change any test's vectors.

    The result is shared by the whole session, and factories run while test
    modules are collected. Implement the hook in the rootdir's ``conftest.py``
    (or a plugin): a ``conftest.py`` further down is only loaded when pytest
    reaches its directory, so it may be too late, and once loaded its result
    also applies to modules elsewhere. Under pytest-xdist every worker calls the
    hook, and the result must be the same in all of them, or the workers collect
    different tests.

    An exception raised by the hook fails the collection of each module that
    uses a factory with ``ctx``, with a message naming the strategy;
    ``pytest.fail()`` is reported as it is, and ``pytest.skip(...,
    allow_module_level=True)`` skips those modules.

    Args:
        config: The pytest config object.

    Returns:
        Any object, or ``None`` to let the next implementation answer.
    """
