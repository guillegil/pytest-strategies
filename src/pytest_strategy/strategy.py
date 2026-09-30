"""
Compatibility module for code written against pytest-strategies 2.x.

``from pytest_strategy.strategy import Strategy, PytestStrategiesWarning`` and the
warning filter ``error::pytest_strategy.strategy.PytestStrategiesWarning`` keep
working. The package attribute ``pytest_strategy.strategy`` is the ``strategy``
decorator, so ``import pytest_strategy.strategy as m`` gives the decorator, not
this module; import names from it with ``from ... import`` instead.
"""

from ._api import Strategy
from ._registry import _describe_factory as _describe_factory
from ._registry import _factory_origin as _factory_origin
from ._registry import _normalize as _normalize
from ._registry import _unwrap as _unwrap
from ._warnings import PytestStrategiesWarning

__all__ = ["Strategy", "PytestStrategiesWarning"]
