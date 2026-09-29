"""
Regression tests for resolver fixes.

These tests mock the pytest.Config and registry to drive resolve_and_parametrize
directly and inspect the pytest.mark.parametrize it applies.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from pytest_strategy import RNGInteger
from pytest_strategy._resolver import resolve_and_parametrize
from pytest_strategy.parameters import Parameter
from pytest_strategy.rng import Series
from pytest_strategy.test_args import TestArg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**options):
    """Return a mock pytest.Config whose getoption() serves the given CLI options."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    return config


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    _fn.__signature__ = inspect.Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


def _resolve(factory, argnames, **options):
    """Resolve ``factory`` under the given CLI options.

    Returns the (argstr, samples, ids) passed to pytest.mark.parametrize.
    """
    marked = resolve_and_parametrize(
        "strat",
        _make_test_fn(argnames),
        registry={"strat": factory},
        config=_make_config(**options),
        pytest_fixtures=set(),
        validate=False,
    )
    mark = marked.pytestmark[-1]
    return mark.args[0], list(mark.args[1]), mark.kwargs["ids"]


def _series_param(**kwargs):
    """Two Series args (3x2 product) plus one directed and one test vector."""
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"])),
        directed_vectors={"corner": (99, "z")},
        test_vectors={"tv": (42, "t")},
        **kwargs,
    )


def _random_param(**kwargs):
    """A single random (non-sequence) arg plus one directed and one test vector."""
    return Parameter(
        TestArg("code", rng_type=RNGInteger(200, 400)),
        directed_vectors={"corner": (500,)},
        test_vectors={"ok": (200,)},
        **kwargs,
    )


PRODUCT = [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")]


# ---------------------------------------------------------------------------
# --nsamples=auto honours --vector-mode, --vector-name and --vector-index
# ---------------------------------------------------------------------------


class TestAutoModeHonoursVectorOptions:
    """'auto' replaces only the random part of generation; modes and filters still apply."""

    def test_all_mode_prepends_directed_vectors(self):
        _, samples, _ = _resolve(lambda nsamples: _series_param(), ["x", "y"], nsamples="auto")
        assert samples == [(99, "z")] + PRODUCT

    def test_mixed_mode_prepends_directed_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_mode="mixed"
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_mixed_mode_without_always_include_directed(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(always_include_directed=False),
            ["x", "y"],
            nsamples="auto",
            vector_mode="mixed",
        )
        assert samples == PRODUCT

    def test_random_only_mode_is_exhaustive_rows_only(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="random_only",
        )
        assert samples == PRODUCT

    def test_test_mode_returns_test_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_mode="test"
        )
        assert samples == [(42, "t")]

    def test_directed_only_mode_returns_directed_vectors(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="directed_only",
        )
        assert samples == [(99, "z")]

    def test_vector_name_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_name="corner"
        )
        assert samples == [(99, "z")]

    def test_vector_index_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_index=0
        )
        assert samples == [(99, "z")]

    def test_vector_name_filter_takes_precedence_over_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(),
            ["x", "y"],
            nsamples="auto",
            vector_mode="test",
            vector_name="corner",
        )
        assert samples == [(99, "z")]

    def test_missing_vector_name_gives_empty_samples(self):
        _, samples, _ = _resolve(
            lambda nsamples: _series_param(), ["x", "y"], nsamples="auto", vector_name="nope"
        )
        assert samples == []


class TestAutoModeWithoutSequenceArgs:
    """A Parameter with no Series/RNGSequence arg falls back to finite generation."""

    def test_falls_back_to_param_nsamples(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(nsamples=4), ["code"], nsamples="auto"
        )
        assert samples[0] == 500
        assert len(samples) == 1 + 4
        assert all(200 <= s <= 400 for s in samples[1:])

    def test_falls_back_to_default_of_10(self):
        _, samples, _ = _resolve(lambda nsamples: _random_param(), ["code"], nsamples="auto")
        assert samples[0] == 500
        assert len(samples) == 1 + 10

    def test_fallback_honours_random_only_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(nsamples=4),
            ["code"],
            nsamples="auto",
            vector_mode="random_only",
        )
        assert len(samples) == 4
        assert all(200 <= s <= 400 for s in samples)

    def test_test_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_mode="test"
        )
        assert samples == [200]

    def test_directed_only_mode(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(),
            ["code"],
            nsamples="auto",
            vector_mode="directed_only",
        )
        assert samples == [500]

    def test_vector_name_filter(self):
        _, samples, _ = _resolve(
            lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_name="corner"
        )
        assert samples == [500]

    def test_invalid_mode_still_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _resolve(lambda nsamples: _random_param(), ["code"], nsamples="auto", vector_mode="bad")
