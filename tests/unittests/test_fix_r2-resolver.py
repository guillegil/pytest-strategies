"""
Regression tests for the second round of resolver, introspection, dataclass and ID fixes.

Each test here failed before its fix, except those whose docstring says they guard
behaviour that already worked. Most drive resolve_and_parametrize directly with a
mocked pytest.Config and inspect the pytest.mark.parametrize it applies.
"""

import inspect
import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pytest_strategy import RNG, RNGInteger, Strategy
from pytest_strategy._resolver import resolve_and_parametrize
from pytest_strategy.parameters import Parameter
from pytest_strategy.rng import Series
from pytest_strategy.test_args import TestArg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what these tests change globally: the registry, the RNG seed and random state."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


def _make_config(rootpath=None, **options):
    """Return a mock pytest.Config serving the given CLI options (and rootpath, if given)."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    if rootpath is not None:
        config.rootpath = Path(rootpath)
    return config


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    _fn.__signature__ = inspect.Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


def _parametrize(factory, test_fn, *, validate=True, config=None):
    """Resolve ``factory`` for ``test_fn``; return the mark's (argnames, values, ids)."""
    marked = resolve_and_parametrize(
        "strat",
        test_fn,
        registry={"strat": factory},
        config=config if config is not None else _make_config(),
        pytest_fixtures=set(),
        validate=validate,
    )
    mark = marked.pytestmark[-1]
    return mark.args[0], list(mark.args[1]), mark.kwargs["ids"]


def _resolve(factory, argnames, **options):
    """Resolve ``factory`` for a test taking ``argnames`` under the given CLI options."""
    return _parametrize(
        factory, _make_test_fn(argnames), validate=False, config=_make_config(**options)
    )


def _with_nsamples(param, nsamples):
    """Set ``param.nsamples`` as ``Parameter(..., nsamples=nsamples)`` would.

    It is assigned after construction so that these tests exercise only the
    resolver, whatever Parameter's own validation of the value is.
    """
    param.nsamples = nsamples
    return param


def _series_param():
    """Two Series args (3x2 product) plus one directed vector."""
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"])),
        directed_vectors={"corner": (99, "z")},
    )


def _random_param():
    """A single random (non-sequence) arg plus one directed vector."""
    return Parameter(
        TestArg("code", rng_type=RNGInteger(200, 400)),
        directed_vectors={"corner": (500,)},
    )


PRODUCT = [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")]


# ---------------------------------------------------------------------------
# nsamples="auto" forwarded into Parameter.nsamples
# ---------------------------------------------------------------------------


class TestParameterNsamplesAuto:
    """A Parameter whose own nsamples is "auto" is exhaustive, or falls back to 10."""

    def test_passthrough_with_series_under_auto_is_exhaustive(self):
        """Guards behaviour that already worked in the resolver."""
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_series_param(), nsamples), ["x", "y"], nsamples="auto"
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_auto_default_without_cli_flag_is_exhaustive(self):
        """Guards behaviour that already worked in the resolver."""
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_series_param(), "auto"), ["x", "y"]
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_random_only_passthrough_under_auto_falls_back_to_10(self):
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_random_param(), nsamples), ["code"], nsamples="auto"
        )
        assert samples[0] == 500
        assert len(samples) == 1 + 10
        assert all(200 <= s <= 400 for s in samples[1:])

    def test_random_only_auto_default_without_cli_flag_falls_back_to_10(self):
        _, samples, _ = _resolve(lambda nsamples: _with_nsamples(_random_param(), "auto"), ["code"])
        assert len(samples) == 1 + 10

    def test_int_param_nsamples_still_used_as_fallback(self):
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_random_param(), 3), ["code"], nsamples="auto"
        )
        assert len(samples) == 1 + 3
