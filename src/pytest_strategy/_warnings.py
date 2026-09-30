"""
Warning categories for pytest-strategies.

Kept in a module of its own so that low-level modules (e.g. parameters.py) can
emit them without importing strategy.py, which itself imports those modules.
"""


class PytestStrategiesWarning(UserWarning):
    """
    Warning category for pytest-strategies.

    Emitted e.g. when a strategy name is registered twice, or when finite mode skips
    a Series combination whose random args never satisfied the vector constraints.
    """
