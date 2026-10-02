"""Shared fixtures for the pytest-strategies test suite."""

import random

import pytest

from pytest_strategy import RNG
from pytest_strategy._registry import registry


@pytest.fixture(autouse=True)
def _restore_strategy_state():
    """
    Give every test the registry, seed and random states it started with.

    Tests register strategies, reseed the RNG and draw values; without this, what
    one test leaves behind changes the next one's results, and the suite passes
    or fails depending on the order it runs in.
    """
    saved_registry = registry.snapshot()
    seed = RNG.get_seed()
    generator_state = RNG.generator().getstate()
    random_state = random.getstate()
    try:
        yield
    finally:
        registry.restore(saved_registry)
        RNG._seed = seed
        RNG.generator().setstate(generator_state)
        random.setstate(random_state)
