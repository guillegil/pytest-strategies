"""
Unit tests for resolver effective-nsamples precedence logic (FR-3 through FR-8).

These tests mock the pytest.Config and registry to exercise the precedence rules
inside resolve_and_parametrize without running a real pytest session.
"""

from unittest.mock import MagicMock

from pytest_strategy import RNGInteger
from pytest_strategy._resolver import resolve_and_parametrize
from pytest_strategy.parameters import Parameter
from pytest_strategy.test_args import TestArg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(nsamples_value):
    """Return a mock pytest.Config that returns nsamples_value for getoption('nsamples')."""
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: {
        "nsamples": nsamples_value,
        "vector_mode": "all",
        "vector_name": None,
        "vector_index": None,
    }.get(opt, default)
    return config


def _make_registry(param: Parameter):
    """Return a minimal registry whose factory always returns param."""
    return {"strat": lambda nsamples: param}


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    # Build signature with the expected parameter names
    params = [
        __import__("inspect").Parameter(
            name,
            __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD,
        )
        for name in argnames
    ]
    _fn.__signature__ = __import__("inspect").Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


# ---------------------------------------------------------------------------
# Precedence tests (FR-3 through FR-8)
# ---------------------------------------------------------------------------


class TestResolverNsamplesPrecedence:
    """Resolver must compute the correct effective nsamples for each scenario."""

    def test_cli_none_param_nsamples_set_uses_param(self):
        """FR-4: CLI absent (None) + Parameter.nsamples=15 → 15 vectors generated."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg, nsamples=15)
        config = _make_config(None)
        registry = _make_registry(param)
        test_fn = _make_test_fn(["x"])

        marked = resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        # Extract the parametrize args list from the mark
        mark = marked.pytestmark[0]
        samples = mark.args[1]
        assert len(samples) == 15

    def test_cli_none_param_nsamples_none_uses_fallback(self):
        """FR-5: CLI absent (None) + Parameter.nsamples=None → 10 vectors (fallback)."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)  # nsamples not set
        config = _make_config(None)
        registry = _make_registry(param)
        test_fn = _make_test_fn(["x"])

        marked = resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        mark = marked.pytestmark[0]
        samples = mark.args[1]
        assert len(samples) == 10

    def test_cli_explicit_overrides_param_nsamples(self):
        """FR-6: CLI='5' overrides Parameter.nsamples=15 → 5 vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg, nsamples=15)
        config = _make_config("5")
        registry = _make_registry(param)
        test_fn = _make_test_fn(["x"])

        marked = resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        mark = marked.pytestmark[0]
        samples = mark.args[1]
        assert len(samples) == 5

    def test_cli_explicit_no_param_nsamples(self):
        """FR-6: CLI='7' + Parameter has no nsamples → 7 vectors."""
        arg = TestArg("x", rng_type=RNGInteger(0, 10))
        param = Parameter(arg)
        config = _make_config("7")
        registry = _make_registry(param)
        test_fn = _make_test_fn(["x"])

        marked = resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        mark = marked.pytestmark[0]
        samples = mark.args[1]
        assert len(samples) == 7

    def test_cli_auto_wins_over_param_nsamples(self):
        """FR-7: CLI='auto' always triggers exhaustive mode, ignoring param.nsamples."""
        from pytest_strategy.rng import RNGSequence

        seq = RNGSequence([1, 2, 3])
        arg = TestArg("x", rng_type=seq)
        param = Parameter(arg, nsamples=15)
        config = _make_config("auto")
        registry = _make_registry(param)
        test_fn = _make_test_fn(["x"])

        marked = resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        mark = marked.pytestmark[0]
        samples = mark.args[1]
        # Exhaustive: 3 elements in sequence → 3 samples (not 15)
        assert len(samples) == 3

    def test_legacy_path_never_receives_none(self):
        """FR-8: Legacy tuple factory must always receive an int, never None.

        When CLI is None and the factory returns a legacy tuple, the nsamples
        passed to the factory must be the integer 10 (fallback), not None.
        """
        received = []

        def legacy_factory(nsamples):
            received.append(nsamples)
            # Return a legacy tuple (argnames, samples)
            return ("x", [(i,) for i in range(nsamples)])

        config = _make_config(None)
        registry = {"strat": legacy_factory}
        test_fn = _make_test_fn(["x"])

        resolve_and_parametrize(
            "strat",
            test_fn,
            registry=registry,
            config=config,
            pytest_fixtures=set(),
            validate=False,
        )

        # The factory must have been called with an int, never with None
        assert len(received) >= 1
        for val in received:
            assert isinstance(val, int), f"Expected int, got {type(val).__name__}: {val!r}"
        # Default fallback is 10
        assert received[0] == 10
