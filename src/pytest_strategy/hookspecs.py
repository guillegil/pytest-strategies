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

        def pytest_strategies_context(config):
            return Testbench.parse_config(config.getoption("--tb-config"))

        # strategies.py
        @Strategy.register("channels")
        def channels(nsamples, ctx):
            return Parameter(TestArg("channel", rng_type=RNGSequence(ctx.channels)))

    The hook is called at most once per session, when the first factory with a
    ``ctx`` parameter is called, and its result is reused for every other one.
    Factories without a ``ctx`` parameter never trigger it, and they are called
    as before. When no implementation returns a value, ``ctx`` is ``None``.

    Factories run while test modules are collected, so implement the hook in the
    rootdir's ``conftest.py`` (or a plugin): a ``conftest.py`` further down may not
    be loaded yet when the hook is called. Under pytest-xdist every worker calls
    it, and the result must be the same in all of them, or the workers collect
    different tests.

    Args:
        config: The pytest config object.

    Returns:
        Any object, or ``None`` to let the next implementation answer.
    """
