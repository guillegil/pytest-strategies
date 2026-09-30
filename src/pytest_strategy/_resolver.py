"""Resolve a strategy into the parametrization of a test.

This is the orchestration the plugin runs for each test marked with
``@strategy`` (from ``pytest_generate_tests``): read CLI options, call the
factory, generate vectors, and build the arguments of ``pytest.mark.parametrize``
for either dataclass or named-parameter mode.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import pytest

from ._dataclass import convert_to_dataclass
from ._ids import generate_dataclass_ids, generate_test_ids, make_unique_ids
from ._introspection import detect_dataclass_param, lazy_signature, validate_signature
from ._registry import _describe_factory, display_path, factory_source
from ._runtime import Resolution, runtime
from ._warnings import PytestStrategiesWarning
from .parameters import Parameter
from .rng import RNG, SequenceLike

if TYPE_CHECKING:
    from collections.abc import Sequence

# pytest.param() returns a ParameterSet (a NamedTuple), which pytest does not export
_ParameterSet = type(pytest.param())

# Default for the strategies_max_exhaustive ini option
DEFAULT_MAX_EXHAUSTIVE = 100_000


class Parametrization(NamedTuple):
    """The arguments of ``pytest.mark.parametrize`` for one test and strategy."""

    argnames: str
    values: list[Any]
    ids: list[str]


def _id_row(sample: Any, single: bool) -> Any:
    """Return the row generate_test_ids expects for a sample passed to parametrize."""
    if isinstance(sample, _ParameterSet):
        # Build the ID from the values; pytest still prefers an explicit id
        return sample.values
    # generate_test_ids unwraps a single-argument row once, so re-wrap the value:
    # a tuple value then keeps its full ID
    return (sample,) if single else sample


def _skipped_param(reason: str, width: int) -> Any:
    """Return the single skipped row that stands in for a strategy without values."""
    return pytest.param(*([None] * width), marks=pytest.mark.skip(reason=reason), id="skipped")


@contextlib.contextmanager
def _attributed_warnings(name: str, test_fn: Callable[..., Any]) -> Iterator[None]:
    """
    Re-emit the PytestStrategiesWarnings raised in the block at the test function,
    prefixed with the strategy and the test.

    Raised during vector generation, they would otherwise point into this module and
    not say which strategy or test they are about. Other warnings pass unchanged.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PytestStrategiesWarning)
        yield
    fn = inspect.unwrap(test_fn)
    code = getattr(fn, "__code__", None)
    for w in caught:
        if issubclass(w.category, PytestStrategiesWarning) and code is not None:
            # co_filename can be stale (pytest's rewritten pyc after a checkout moved)
            filename: str = getattr(fn, "__globals__", {}).get("__file__") or code.co_filename
            warnings.warn_explicit(
                f"Strategy '{name}' ({test_fn.__qualname__}): {w.message}",
                w.category,
                filename,
                code.co_firstlineno,
            )
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)


def _unique_ids(
    rows: list[Any], ids: list[str], config: pytest.Config | None
) -> tuple[list[Any], list[str]]:
    """
    Return the rows and ids to parametrize with, giving every row a unique test ID.

    Rows can repeat, and pytest's strict_parametrization_ids makes duplicate IDs a
    collection error instead of suffixing them. The ID of a ``pytest.param(...,
    id=...)`` row overrides its ``ids`` entry, so duplicates are suffixed among these
    effective IDs, and such a row is rebuilt with its new ID.
    """
    effective = [
        row.id if isinstance(row, _ParameterSet) and row.id is not None else row_id
        for row, row_id in zip(rows, ids)
    ]
    escape = config is None or not config.getini(
        "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"
    )
    unique_rows: list[Any] = []
    unique_ids: list[str] = []
    for row, row_id, unique_id in zip(rows, ids, make_unique_ids(effective, escape=escape)):
        if isinstance(row, _ParameterSet) and row.id is not None:
            if unique_id != row.id:
                row = row._replace(id=unique_id)
            # pytest ignores the ids entry of this row: keep the one built from its values
            unique_id = row_id
        unique_rows.append(row)
        unique_ids.append(unique_id)
    return unique_rows, unique_ids


def _test_location(test_fn: Callable[..., Any], config: pytest.Config | None) -> str:
    """
    Return the test's file path relative to the rootdir, in posix form.

    Used in the test's random stream key instead of ``__module__``, which depends
    on ``--import-mode``. Falls back to ``__module__`` without a config or when
    the file is outside the rootdir.
    """
    rootpath = getattr(config, "rootpath", None) if config is not None else None
    if rootpath is not None:
        try:
            fn = inspect.unwrap(test_fn)
            # The module's __file__ comes from the import system; co_filename can be
            # stale (a rewritten pyc cached before the checkout was moved)
            source = getattr(fn, "__globals__", {}).get("__file__") or inspect.getsourcefile(fn)
            if source:
                path = Path(os.path.realpath(source))
                return path.relative_to(os.path.realpath(rootpath)).as_posix()
        except (TypeError, ValueError):
            pass
    return test_fn.__module__


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


def _declares_ctx(sig: inspect.Signature | None) -> bool:
    """Return True if a signature has a ``ctx`` parameter that can be passed by keyword."""
    if sig is None:
        return False
    param = sig.parameters.get("ctx")
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _ctx_kwargs(name: str, sig: inspect.Signature | None) -> dict[str, Any]:
    """
    Return ``{"ctx": <the session's pytest_strategies_context result>}`` for a
    factory with a ``ctx`` parameter, and ``{}`` for any other.

    When no hook implementation answers and ``ctx`` has a default (including a
    value bound by ``functools.partial``), ``{}`` too, so the factory keeps it.
    """
    if not _declares_ctx(sig):
        return {}
    try:
        ctx = runtime.strategy_context()
    except Exception as e:
        raise ValueError(
            f"Strategy factory '{name}' has a 'ctx' parameter, but the "
            f"pytest_strategies_context hook raised {type(e).__name__}: {e}"
        ) from e
    assert sig is not None
    if ctx is None and sig.parameters["ctx"].default is not inspect.Parameter.empty:
        return {}
    return {"ctx": ctx}


def call_factory(name: str, factory: Callable[..., Any], nsamples: int | str) -> Any:
    """
    Call a strategy factory, passing ``nsamples`` the way it accepts it.

    The calling convention is chosen from the signature of the callable that is
    actually called (a decorator's wrapper, not the function it wraps), never by
    retrying after a failure, so the factory is called exactly once:
    ``factory(nsamples=...)`` when it accepts that keyword (a named parameter or
    ``**kwargs``), ``factory(nsamples)`` when it has a positional parameter, and
    ``factory()`` when it takes no arguments.

    The exception is a signature with only ``*args``/``**kwargs`` (e.g. a
    ``functools.wraps`` decorator's wrapper), or none at all (``functools.cache``),
    which does not say how to pass ``nsamples``. Then the signature of the
    function it wraps decides, the same way. When that does not tell either (a
    decorator without ``functools.wraps``, ``unittest.mock.patch``), the factory
    is called as ``factory(nsamples=...)`` and, if that raises ``TypeError``, once
    more as ``factory(nsamples)``.

    A factory with a ``ctx`` parameter also gets ``ctx=`` the session's
    ``pytest_strategies_context`` result. When no implementation answers, ``ctx``
    keeps its default, or is ``None`` without one. For a wrapper with only
    ``*args``/``**kwargs``, the function it wraps decides.

    Args:
        name: Name of the strategy (for error messages)
        factory: The registered factory function
        nsamples: Value to pass as the factory's ``nsamples``

    Returns:
        Whatever the factory returns

    Raises:
        ValueError: If the signature cannot accept any of these calls, if the
            factory itself raises (chained to the original exception), or if the
            pytest_strategies_context hook raised for a factory that needs ``ctx``
    """
    args: tuple[Any, ...] = ()
    try:
        # follow_wrapped=False: a wraps() decorator's __wrapped__ describes the inner
        # function, not the wrapper that is called
        sig: inspect.Signature | None = lazy_signature(factory, follow_wrapped=False)
    except (TypeError, ValueError):
        # No introspectable signature (e.g. some builtins)
        sig = None

    if sig is None or _only_var_args(sig):
        # The wrapper does not say whether ctx is wanted; the wrapped function does
        try:
            wrapped: inspect.Signature | None = lazy_signature(factory)
        except (TypeError, ValueError):
            wrapped = None
        extra = _ctx_kwargs(name, wrapped)
        # mock.patch passes its mocks as extra arguments the wrapped signature lists
        if (
            wrapped is not None
            and not _only_var_args(wrapped)
            and not hasattr(factory, "patchings")
        ):
            # Checked with a ctx, so nsamples is never bound to it positionally
            bind_extra = {"ctx": None} if _declares_ctx(wrapped) else {}
            calls: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
                ((), {"nsamples": nsamples}),
                ((nsamples,), {}),
                ((), {}),
            )
            for call_args, call_kwargs in calls:
                if _accepts(wrapped, *call_args, **call_kwargs, **bind_extra) and (
                    sig is None or _accepts(sig, *call_args, **call_kwargs, **extra)
                ):
                    try:
                        return factory(*call_args, **call_kwargs, **extra)
                    except Exception as e:
                        raise _factory_error(name, nsamples, e) from e
        try:
            return factory(nsamples=nsamples, **extra)
        except TypeError as e:
            # The keyword may not be accepted: retry positionally, and report the
            # original error if that fails too
            try:
                return factory(nsamples, **extra)
            except Exception as retry_error:
                raise _factory_error(name, nsamples, e) from retry_error
        except Exception as e:
            raise _factory_error(name, nsamples, e) from e

    extra = {"ctx": None} if _declares_ctx(sig) else {}
    kwargs: dict[str, Any] = {"nsamples": nsamples, **extra}
    if not _accepts(sig, nsamples=nsamples, **extra):
        if _accepts(sig, nsamples, **extra):
            args, kwargs = (nsamples,), dict(extra)
        elif _accepts(sig, **extra):
            kwargs = dict(extra)
        else:
            raise ValueError(
                f"Strategy factory '{name}' cannot be called with its signature {sig}. "
                f"Factory should accept an 'nsamples' parameter (or no parameters), "
                f"and optionally 'ctx'."
            )
    if extra:
        # Only now: a factory whose signature is rejected never triggers the hook
        del kwargs["ctx"]
        kwargs.update(_ctx_kwargs(name, sig))

    try:
        return factory(*args, **kwargs)
    except Exception as e:
        raise _factory_error(name, nsamples, e) from e


def _max_exhaustive(param: Parameter, config: pytest.Config | None) -> int:
    """Return the most rows the exhaustive generation of ``param`` may produce."""
    if param.max_exhaustive is not None:
        return param.max_exhaustive
    if config is None:
        return DEFAULT_MAX_EXHAUSTIVE
    try:
        raw = config.getini("strategies_max_exhaustive")
    except ValueError:
        # The plugin's ini options are not registered (a config without the plugin)
        return DEFAULT_MAX_EXHAUSTIVE
    if not isinstance(raw, (str, int)):
        # Not a real config (a test's stand-in)
        return DEFAULT_MAX_EXHAUSTIVE
    try:
        limit = int(str(raw).strip())
    except ValueError:
        limit = -1
    if limit < 1:
        raise pytest.UsageError(f"strategies_max_exhaustive must be an integer >= 1, got {raw!r}")
    return limit


def _check_size(
    name: str, param: Parameter, rows_per_combination: int, how: str, limit: int
) -> None:
    """
    Fail before generating when the Series/RNGSequence combinations exceed ``limit`` rows.

    Args:
        name: Strategy name, for the message
        param: The strategy's Parameter
        rows_per_combination: Rows generated for each combination
        how: What asked for every combination, for the message
        limit: The most rows allowed
    """
    sizes = [
        (arg.name, len(arg.rng_type.sequence))
        for arg in param.test_args
        if isinstance(arg.rng_type, SequenceLike)
    ]
    rows = rows_per_combination
    for _, size in sizes:
        rows *= size
    if rows <= limit:
        return
    product = " x ".join(f"{arg}={size:,}" for arg, size in sizes)
    if rows_per_combination != 1:
        product += f" x {rows_per_combination:,} rows each"
    raise ValueError(
        f"Strategy '{name}': {how} would generate {rows:,} rows ({product}), more than "
        f"the limit of {limit:,}. Raise it with Parameter(max_exhaustive=...) or the "
        "strategies_max_exhaustive ini option, or use fewer values."
    )


def _warn_legacy(name: str, factory: Callable[..., Any]) -> None:
    """Warn that a factory returned the deprecated (argnames, samples) tuple, at the factory."""
    # The path as spelled, not normalized: warnings report and filter by it
    filename, _, line = factory_source(factory)
    message = (
        f"Strategy '{name}' returns an (argnames, samples) tuple, which is deprecated and "
        "will stop working in 4.0; return a Parameter instead "
        f"(factory: {_describe_factory(factory)})"
    )
    if filename is None or line is None:
        warnings.warn(message, DeprecationWarning, stacklevel=3)
    else:
        warnings.warn_explicit(message, DeprecationWarning, filename, line)


def build_parametrization(
    name: str,
    factory: Callable[..., Any],
    test_fn: Callable[..., Any],
    *,
    config: pytest.Config | None,
    pytest_fixtures: set[str],
    validate: bool = True,
) -> Parametrization:
    """
    Call a strategy's factory and build the parametrization of a test.

    Args:
        name: The strategy's name, used in messages and in the test's random
            stream key
        factory: The factory to call
        test_fn: The test function
        config: The session's config (CLI options), or None
        pytest_fixtures: Fixture names the signature check ignores
        validate: Check that the test takes the strategy's arguments

    Raises:
        ValueError: With a message naming the strategy when the factory, the
            generation or the signature check fails
    """
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

    # Restart the RNG generator on this strategy and test's own stream
    RNG.refresh_seed(key=f"{name}:{_test_location(test_fn, config)}::{test_fn.__qualname__}")

    # Call the factory function the way its signature accepts nsamples
    result = call_factory(name, factory, factory_nsamples)

    resolution = Resolution(strategy=name, where=_where(factory, config))

    # Set when a skip_if_empty sequence has no values: the test then runs as one
    # skipped row with this reason instead of the generated vectors
    skip_reason: str | None = None

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
            source = "--nsamples"
        elif cli_nsamples is not None:
            effective_nsamples = int(cli_nsamples)
            source = "--nsamples"
        elif param.nsamples is not None:
            effective_nsamples = param.nsamples
            source = "Parameter(nsamples=)"
        else:
            effective_nsamples = 10
            source = "default"

        # "auto" enumerates the Series/RNGSequence args. A strategy without any has
        # nothing to enumerate, so it falls back to the finite count instead of failing.
        # param.nsamples may itself be "auto" (e.g. a factory forwarding its nsamples),
        # so only an int count is used; anything else falls back to 10.
        if effective_nsamples == "auto" and not any(
            isinstance(arg.rng_type, SequenceLike) for arg in param.test_args
        ):
            if isinstance(param.nsamples, int):
                effective_nsamples, source = param.nsamples, "Parameter(nsamples=)"
            else:
                effective_nsamples, source = 10, "default"
            source += ", no Series/RNGSequence for auto"
        resolution.nsamples, resolution.source = effective_nsamples, source

        filtered = vector_name is not None or vector_index is not None
        if not filtered and vector_mode not in ("test", "directed_only"):
            if effective_nsamples == "auto":
                _check_size(name, param, 1, "--nsamples=auto", _max_exhaustive(param, config))
            elif param.per_sequence_samples and isinstance(effective_nsamples, int):
                _check_size(
                    name,
                    param,
                    effective_nsamples,
                    "per_sequence_samples=True",
                    _max_exhaustive(param, config),
                )

        # Generate samples using Parameter's generate_vectors with CLI options
        try:
            # Warnings raised while generating name the strategy and the test
            with _attributed_warnings(name, test_fn):
                if effective_nsamples == "auto":
                    # "auto" replaces only the random samples: CLI filters, vector modes
                    # and directed vectors apply exactly as they do for a finite count.
                    samples = param.generate_vectors(
                        n=0,
                        mode=vector_mode,
                        filter_by_name=vector_name,
                        filter_by_index=vector_index,
                    )
                    if not filtered and vector_mode not in ("test", "directed_only"):
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
            if filtered:
                samples = []
                # The plugin reports a filter that no strategy in the run matches
                runtime.record_vector_filter(name, False, list(param.directed_vectors))
            else:
                raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e
        else:
            if filtered:
                runtime.record_vector_filter(name, True)
            # The generators return nothing for a skipped Parameter. A vector filter
            # that names none of its vectors keeps the empty set, as for any strategy.
            skip_reason = param.skip_reason

        _count_rows(resolution, param, samples, vector_mode, filtered)

        # Get argument names from Parameter
        argnames: Sequence[str] = param.arg_names

    else:
        # LEGACY MODE: Tuple-based strategy (backward compatibility)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                f"Strategy '{name}' must return either a Parameter instance "
                f"or a tuple (argnames, samples), got {type(result).__name__}"
            )
        _warn_legacy(name, factory)

        argnames, samples = result

        # Materialize the samples: a generator would otherwise be consumed by ID
        # generation before pytest.mark.parametrize sees it
        samples = list(samples)
        resolution.random = len(samples)
        resolution.nsamples, resolution.source = factory_nsamples, "legacy tuple"

        # Convert a string of argnames to a tuple, split on commas as pytest does
        if isinstance(argnames, str):
            argnames = tuple(n.strip() for n in argnames.split(",") if n.strip())

    runtime.record_resolution(resolution)

    # Detect dataclass mode. Without signature validation, parameters other than the
    # dataclass one may be fixtures consuming the argnames, so the dataclass
    # parameter must then be the test's only one.
    is_dc_mode, dc_type, dc_param = detect_dataclass_param(
        test_fn, argnames, pytest_fixtures=pytest_fixtures, allow_fixtures=validate
    )

    if is_dc_mode:
        # DATACLASS MODE: Convert samples to dataclass instances
        assert dc_type is not None and dc_param is not None  # guaranteed when is_dc_mode is True
        if skip_reason is not None:
            # No values to build an instance from, but the dataclass must still match
            # the strategy, as the signature check does in named mode
            try:
                convert_to_dataclass([], argnames, dc_type)
            except Exception as e:
                raise ValueError(
                    f"Error converting samples to dataclass for strategy '{name}': {e}"
                ) from e
            return Parametrization(dc_param, [_skipped_param(skip_reason, 1)], ["skipped"])
        try:
            # A pytest.param() sample is converted from its values; its marks and id
            # are re-attached to the instance below
            dataclass_samples = convert_to_dataclass(
                [tuple(s.values) if isinstance(s, _ParameterSet) else s for s in samples],
                argnames,
                dc_type,
            )
        except Exception as e:
            raise ValueError(
                f"Error converting samples to dataclass for strategy '{name}': {e}"
            ) from e

        # Generate test IDs for dataclass mode
        ids = generate_dataclass_ids(dataclass_samples, dc_type)

        params = [
            pytest.param(inst, marks=s.marks, id=s.id) if isinstance(s, _ParameterSet) else inst
            for s, inst in zip(samples, dataclass_samples)
        ]
        params, ids = _unique_ids(params, ids, config)

        # Parametrize the dataclass parameter chosen by detection
        return Parametrization(dc_param, params, ids)

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

        if skip_reason is not None:
            samples = [_skipped_param(skip_reason, len(argnames))]

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
        samples, ids = _unique_ids(samples, ids, config)

        return Parametrization(argstr, samples, ids)


def _where(factory: Callable[..., Any], config: pytest.Config | None) -> str:
    """Return the factory's file, relative to the rootdir when inside it, for the summary."""
    filename = factory_source(factory)[0]
    if filename is None:
        return "<unknown>"
    rootpath = getattr(config, "rootpath", None) if config is not None else None
    return display_path(filename, rootpath)


def _count_rows(
    resolution: Resolution, param: Parameter, samples: list[Any], mode: str, filtered: bool
) -> None:
    """Split a Parameter's generated rows into directed, random and test rows."""
    total = len(samples)
    if filtered or mode == "directed_only":
        resolution.directed = total
    elif mode == "test":
        resolution.test = total
    else:
        if mode == "all" or (mode == "mixed" and param.always_include_directed):
            resolution.directed = min(len(param.directed_vectors), total)
        resolution.random = total - resolution.directed


def resolve_and_parametrize(
    name: str,
    test_fn: Callable[..., Any],
    *,
    registry: Mapping[str, Callable[..., Any]],
    config: pytest.Config | None,
    pytest_fixtures: set[str],
    validate: bool = True,
) -> Callable[..., Any]:
    """Build the parametrization for a registered strategy and apply it to ``test_fn``."""
    parametrization = build_parametrization(
        name,
        registry[name],
        test_fn,
        config=config,
        pytest_fixtures=pytest_fixtures,
        validate=validate,
    )
    mark = pytest.mark.parametrize(
        parametrization.argnames, parametrization.values, ids=parametrization.ids
    )
    return cast(Callable[..., Any], mark(test_fn))
