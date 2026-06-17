"""Resolve a registered strategy into a parametrized test function.

This is the orchestration that the ``Strategy.strategy`` decorator delegates to:
read CLI options, call the factory, generate vectors, and apply
``pytest.mark.parametrize`` for either dataclass or named-parameter mode.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, cast

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids
from ._introspection import detect_dataclass_mode, validate_signature
from .parameters import Parameter
from .rng import RNG

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    # The factory must always receive an int (FR-8: never None).
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
    RNG.refresh_seed()

    # Call factory function with keyword argument
    try:
        result = factory(nsamples=factory_nsamples)
    except TypeError as e:
        # Try positional for backward compatibility
        try:
            result = factory(factory_nsamples)
        except Exception as inner_e:
            raise ValueError(
                f"Error calling strategy factory '{name}': {e}. "
                f"Factory should accept 'nsamples' parameter."
            ) from inner_e

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

        # Generate samples using Parameter's generate_vectors with CLI options
        try:
            if effective_nsamples == "auto":
                samples = param.generate_exhaustive()
            else:
                assert isinstance(effective_nsamples, int)
                samples = param.generate_vectors(
                    n=effective_nsamples,
                    mode=vector_mode,
                    filter_by_name=vector_name,
                    filter_by_index=vector_index,
                )
        except KeyError as e:
            # If filtering by name/index and vector doesn't exist, return empty samples
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

        # Convert single string argname to tuple for consistency
        if isinstance(argnames, str):
            argnames = (argnames,)

    # Detect dataclass mode
    is_dc_mode, dc_type = detect_dataclass_mode(test_fn, argnames, pytest_fixtures=pytest_fixtures)

    if is_dc_mode:
        # DATACLASS MODE: Convert samples to dataclass instances
        assert dc_type is not None  # guaranteed when is_dc_mode is True
        try:
            dataclass_samples = convert_to_dataclass(samples, argnames, dc_type)
        except Exception as e:
            raise ValueError(
                f"Error converting samples to dataclass for strategy '{name}': {e}"
            ) from e

        # Get the single parameter name
        sig = inspect.signature(test_fn)
        test_params = [p for p in sig.parameters if p not in pytest_fixtures]
        param_name = test_params[0]

        # Generate test IDs for dataclass mode
        ids = generate_dataclass_ids(dataclass_samples, dc_type)

        # Apply pytest parametrize with single dataclass parameter
        return cast(
            Callable, pytest.mark.parametrize(param_name, dataclass_samples, ids=ids)(test_fn)
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

        # For single parameters, unwrap the tuples
        if len(argnames) == 1:
            samples = [s[0] if isinstance(s, tuple) else s for s in samples]

        # Generate test IDs for better test output readability
        ids = generate_test_ids(argnames, samples)

        # Apply pytest parametrize decorator to the test function
        return cast(Callable, pytest.mark.parametrize(argstr, samples, ids=ids)(test_fn))
