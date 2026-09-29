"""Resolve a registered strategy into a parametrized test function.

This is the orchestration that the ``Strategy.strategy`` decorator delegates to:
read CLI options, call the factory, generate vectors, and apply
``pytest.mark.parametrize`` for either dataclass or named-parameter mode.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids
from ._introspection import detect_dataclass_param, validate_signature
from .parameters import Parameter
from .rng import RNG, SequenceLike

if TYPE_CHECKING:
    from collections.abc import Sequence

# pytest.param() returns a ParameterSet (a NamedTuple), which pytest does not export
_ParameterSet = type(pytest.param())


def _id_row(sample: Any, single: bool) -> Any:
    """Return the row generate_test_ids expects for a sample passed to parametrize."""
    if isinstance(sample, _ParameterSet):
        # Build the ID from the values; pytest still prefers an explicit id
        return sample.values
    # generate_test_ids unwraps a single-argument row once, so re-wrap the value:
    # a tuple value then keeps its full ID
    return (sample,) if single else sample


def _accepts(sig: inspect.Signature, *args: Any, **kwargs: Any) -> bool:
    """Return True if a callable with signature ``sig`` can be called with these arguments."""
    try:
        sig.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def _only_var_args(sig: inspect.Signature) -> bool:
    """Return True if ``sig`` has parameters, all of them ``*args``/``**kwargs``.

    Such a signature (e.g. a decorator without ``functools.wraps``, or
    ``unittest.mock.patch``) does not say how to pass ``nsamples``.
    """
    params = sig.parameters.values()
    return bool(params) and all(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params
    )


def _factory_error(name: str, nsamples: int | str, error: Exception) -> ValueError:
    """Return the error reported when a strategy factory raises ``error``."""
    return ValueError(
        f"Error calling strategy factory '{name}' (nsamples={nsamples!r}): "
        f"{type(error).__name__}: {error}"
    )


def call_factory(name: str, factory: Callable[..., Any], nsamples: int | str) -> Any:
    """
    Call a strategy factory, passing ``nsamples`` the way it accepts it.

    The calling convention is chosen from the signature of the callable that is
    actually called (a decorator's wrapper, not the function it wraps), never by
    retrying after a failure, so the factory is called exactly once:
    ``factory(nsamples=...)`` when it accepts that keyword (a named parameter or
    ``**kwargs``), ``factory(nsamples)`` when it has a positional parameter, and
    ``factory()`` when it takes no arguments.

    The exception is a signature with only ``*args``/``**kwargs`` (e.g. a decorator
    without ``functools.wraps``), or none at all, which does not say how to pass
    ``nsamples``. Such a factory is called as ``factory(nsamples=...)`` and, if that
    raises ``TypeError``, once more as ``factory(nsamples)``.

    Args:
        name: Name of the strategy (for error messages)
        factory: The registered factory function
        nsamples: Value to pass as the factory's ``nsamples``

    Returns:
        Whatever the factory returns

    Raises:
        ValueError: If the signature cannot accept any of these calls, or if the
            factory itself raises (chained to the original exception)
    """
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {"nsamples": nsamples}
    try:
        # follow_wrapped=False: a wraps() decorator's __wrapped__ describes the inner
        # function, not the wrapper that is called
        sig: inspect.Signature | None = inspect.signature(factory, follow_wrapped=False)
    except (TypeError, ValueError):
        # No introspectable signature (e.g. some builtins)
        sig = None

    if sig is None or _only_var_args(sig):
        try:
            return factory(nsamples=nsamples)
        except TypeError as e:
            # The keyword may not be accepted: retry positionally, and report the
            # original error if that fails too
            try:
                return factory(nsamples)
            except Exception as retry_error:
                raise _factory_error(name, nsamples, e) from retry_error
        except Exception as e:
            raise _factory_error(name, nsamples, e) from e

    if not _accepts(sig, nsamples=nsamples):
        if _accepts(sig, nsamples):
            args, kwargs = (nsamples,), {}
        elif _accepts(sig):
            kwargs = {}
        else:
            raise ValueError(
                f"Strategy factory '{name}' cannot be called with its signature {sig}. "
                f"Factory should accept an 'nsamples' parameter (or no parameters)."
            )

    try:
        return factory(*args, **kwargs)
    except Exception as e:
        raise _factory_error(name, nsamples, e) from e


def resolve_and_parametrize(
    name: str,
    test_fn: Callable,
    *,
    registry: dict[str, Callable[..., Any]],
    config: pytest.Config | None,
    pytest_fixtures: set[str],
    validate: bool = True,
) -> Callable:
    """Build and apply the pytest parametrization for a registered strategy."""
    # Get CLI options
    cli_nsamples = config.getoption("nsamples") if config else None
    vector_mode = config.getoption("vector_mode") if config else "all"
    vector_name = config.getoption("vector_name") if config else None
    vector_index = config.getoption("vector_index") if config else None

    # Compute the legacy-safe nsamples to pass to the factory.
    # The factory must always receive an int (FR-8: never None), or the string
    # "auto" when --nsamples=auto is given.
    # At this point we don't yet know whether the factory returns a Parameter
    # or a legacy tuple, so we use the CLI value if explicit, otherwise 10 as
    # a safe sentinel. The real per-strategy override is applied AFTER the
    # factory returns and we confirm it's a Parameter instance.
    if cli_nsamples == "auto":
        factory_nsamples: int | str = "auto"
    elif cli_nsamples is not None:
        factory_nsamples = int(cli_nsamples)
    else:
        # CLI absent: pass 10 to the factory so legacy paths never see None.
        # For Parameter-based strategies the true effective count is resolved
        # below once we have access to param.nsamples.
        factory_nsamples = 10

    factory = registry[name]

    # Refresh the random number generator seed
    RNG.refresh_seed(key=f"{name}:{test_fn.__module__}.{test_fn.__qualname__}")

    # Call the factory function the way its signature accepts nsamples
    result = call_factory(name, factory, factory_nsamples)

    # Detect if result is a Parameter instance or tuple
    if isinstance(result, Parameter):
        # NEW MODE: Parameter-based strategy
        param = result

        # Resolve the effective nsamples with full precedence (FR-3):
        #   1. CLI "auto" → exhaustive (already handled below)
        #   2. CLI explicit int → use it
        #   3. param.nsamples set → use it
        #   4. fallback → 10
        if cli_nsamples == "auto":
            effective_nsamples: int | str = "auto"
        elif cli_nsamples is not None:
            effective_nsamples = int(cli_nsamples)
        elif param.nsamples is not None:
            effective_nsamples = param.nsamples
        else:
            effective_nsamples = 10

        # "auto" enumerates the Series/RNGSequence args. A strategy without any has
        # nothing to enumerate, so it falls back to the finite count instead of failing.
        # param.nsamples may itself be "auto" (e.g. a factory forwarding its nsamples),
        # so only an int count is used; anything else falls back to 10.
        if effective_nsamples == "auto" and not any(
            isinstance(arg.rng_type, SequenceLike) for arg in param.test_args
        ):
            effective_nsamples = param.nsamples if isinstance(param.nsamples, int) else 10

        # Generate samples using Parameter's generate_vectors with CLI options
        try:
            if effective_nsamples == "auto":
                # "auto" replaces only the random samples: CLI filters, vector modes
                # and directed vectors apply exactly as they do for a finite count.
                samples = param.generate_vectors(
                    n=0,
                    mode=vector_mode,
                    filter_by_name=vector_name,
                    filter_by_index=vector_index,
                )
                if (
                    not vector_name
                    and vector_index is None
                    and vector_mode not in ("test", "directed_only")
                ):
                    samples.extend(param.generate_exhaustive())
            else:
                assert isinstance(effective_nsamples, int)
                samples = param.generate_vectors(
                    n=effective_nsamples,
                    mode=vector_mode,
                    filter_by_name=vector_name,
                    filter_by_index=vector_index,
                )
        except (KeyError, IndexError) as e:
            # If filtering by name (KeyError) or index (IndexError) and the vector
            # doesn't exist, return empty samples
            # This allows CLI filtering to work gracefully across multiple strategies
            if vector_name or vector_index is not None:
                samples = []
            else:
                raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e

        # Get argument names from Parameter
        argnames: Sequence[str] = param.arg_names

    else:
        # LEGACY MODE: Tuple-based strategy (backward compatibility)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                f"Strategy '{name}' must return either a Parameter instance "
                f"or a tuple (argnames, samples), got {type(result).__name__}"
            )

        argnames, samples = result

        # Materialize the samples: a generator would otherwise be consumed by ID
        # generation before pytest.mark.parametrize sees it
        samples = list(samples)

        # Convert a string of argnames to a tuple, split on commas as pytest does
        if isinstance(argnames, str):
            argnames = tuple(n.strip() for n in argnames.split(",") if n.strip())

    # Detect dataclass mode. Without signature validation, parameters other than the
    # dataclass one may be fixtures consuming the argnames, so the dataclass
    # parameter must then be the test's only one.
    is_dc_mode, dc_type, dc_param = detect_dataclass_param(
        test_fn, argnames, pytest_fixtures=pytest_fixtures, allow_fixtures=validate
    )

    if is_dc_mode:
        # DATACLASS MODE: Convert samples to dataclass instances
        assert dc_type is not None and dc_param is not None  # guaranteed when is_dc_mode is True
        try:
            dataclass_samples = convert_to_dataclass(samples, argnames, dc_type)
        except Exception as e:
            raise ValueError(
                f"Error converting samples to dataclass for strategy '{name}': {e}"
            ) from e

        # Generate test IDs for dataclass mode
        ids = generate_dataclass_ids(dataclass_samples, dc_type)

        # Apply pytest parametrize to the dataclass parameter chosen by detection
        return cast(
            Callable, pytest.mark.parametrize(dc_param, dataclass_samples, ids=ids)(test_fn)
        )

    else:
        # NAMED PARAMETERS MODE: Standard behavior

        # Validate signature if requested
        if validate:
            try:
                validate_signature(test_fn, argnames, name, pytest_fixtures=pytest_fixtures)
            except ValueError as e:
                raise ValueError(f"Signature validation failed for strategy '{name}': {e}") from e

        # Create comma-separated string of parameter names for pytest.mark.parametrize
        argstr = ",".join(argnames)

        # For single parameters, unwrap the tuples. A pytest.param() sample is a tuple
        # too (ParameterSet); it is passed through unchanged to keep its marks and id.
        if len(argnames) == 1:
            samples = [
                s[0] if isinstance(s, tuple) and not isinstance(s, _ParameterSet) else s
                for s in samples
            ]

        # Generate test IDs for better test output readability, from the same values
        # passed to parametrize
        ids = generate_test_ids(argnames, [_id_row(s, len(argnames) == 1) for s in samples])

        # Apply pytest parametrize decorator to the test function
        return cast(Callable, pytest.mark.parametrize(argstr, samples, ids=ids)(test_fn))
