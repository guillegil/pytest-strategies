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
from typing import Any, NamedTuple, cast

import pytest

from ._dataclass import convert_to_dataclass
from ._factory import FactoryInputs, call_factory
from ._ids import generate_dataclass_ids, generate_test_ids, make_unique_ids
from ._introspection import detect_dataclass_param, validate_signature
from ._registry import _describe_factory, display_path, factory_source
from ._runtime import Resolution, runtime
from ._warnings import PytestStrategiesWarning
from .parameters import (
    Parameter,
    _ConstraintError,
    _ConstraintsExhausted,
    _GenerationStats,
    _ParameterSet,
)
from .rng import RNG, SequenceLike

# Default for the strategies_max_exhaustive ini option
DEFAULT_MAX_EXHAUSTIVE = 100_000


class Parametrization(NamedTuple):
    """The arguments of ``pytest.mark.parametrize`` for one test and strategy."""

    argnames: str
    values: list[Any]
    ids: list[str]


def _id_row(sample: Any) -> Any:
    """Return the values generate_test_ids builds the ID of a generated row from."""
    if isinstance(sample, _ParameterSet):
        # Build the ID from the values; pytest still prefers an explicit id
        return sample.values
    return sample


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


def _exhausted_message(name: str, error: _ConstraintsExhausted) -> str:
    """
    Return the message of constraints that rejected every draw, ending with advice
    that turns the strictest constraint off for this strategy.
    """
    if error.retries:
        advice = "Raise Parameter(max_retries=...), relax a constraint"
    else:
        advice = "Relax a constraint"
    if error.strictest is None:
        return f"{error.detail} {advice}."
    # --strategy-constraint-off splits items on ",", so a strategy whose name has
    # one cannot be targeted: the bare name turns the constraint off everywhere
    item = error.strictest if "," in name else f"{name}:{error.strictest}"
    return (
        f"{error.detail} {advice}, or turn one off for this run with "
        f"--strategy-constraint-off={item}."
    )


def check_factory_result(name: str, factory: Callable[..., Any], result: Any) -> Parameter:
    """
    Return what a strategy factory returned, which must be a :class:`Parameter`.

    Raises:
        ValueError: For anything else. An ``(argnames, samples)`` tuple, the form
            3.0 deprecated, gets a message that says what to return instead.
    """
    if isinstance(result, Parameter):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        raise ValueError(
            f"Strategy '{name}' returned an (argnames, samples) tuple "
            f"(factory: {_describe_factory(factory)}). Returning a tuple was deprecated in "
            "3.0 and is no longer supported in 4.0: return a Parameter, with one TestArg "
            "per argument and fixed rows as directed_vectors; a fixed table with no random "
            "arguments fits @pytest.mark.parametrize better."
        )
    hint = " (did the factory forget to return?)" if result is None else ""
    raise ValueError(
        f"Strategy '{name}' must return a Parameter, got {type(result).__name__}{hint}"
    )


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
    # The CLI options, read once per session
    options = runtime.strategy_options(name, config)
    vector_mode = options.mode
    vector_name = options.vector_name
    vector_index = options.vector_index

    # Restart the RNG generator on this strategy and test's own stream
    RNG.refresh_seed(key=f"{name}:{_test_location(test_fn, config)}::{test_fn.__qualname__}")

    # Call the factory with the inputs it declares by name. Its nsamples is the
    # --nsamples value, "auto", or 10 without the option, never None (FR-8). The
    # count the rows use is resolved below, once the Parameter's own nsamples is
    # known.
    inputs = FactoryInputs(options=options, rng=RNG.generator(), ctx=runtime.strategy_context)
    result = call_factory(name, factory, inputs, rootpath=_rootpath(config))
    param = check_factory_result(name, factory, result)

    resolution = Resolution(strategy=name, where=_where(factory, config))

    # Set when a skip_if_empty sequence has no values: the test then runs as one
    # skipped row with this reason instead of the generated vectors
    skip_reason: str | None = None

    # Resolve the effective nsamples with full precedence (FR-3):
    #   1. CLI "auto" → exhaustive (already handled below)
    #   2. CLI explicit int → use it
    #   3. param.nsamples set → use it
    #   4. fallback → 10
    if options.nsamples_source == "--nsamples":
        effective_nsamples: int | str = options.nsamples
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

    filtered = options.filtered
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
    samples: list[Any]
    # The rejections per constraint (and the combinations "auto" left out), for -v
    stats = _GenerationStats()
    exhaustive = False
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
                    _stats=stats,
                )
                if not filtered and vector_mode not in ("test", "directed_only"):
                    exhaustive = True
                    samples.extend(param.generate_exhaustive(_stats=stats))
            else:
                assert isinstance(effective_nsamples, int)
                samples = param.generate_vectors(
                    n=effective_nsamples,
                    mode=vector_mode,
                    filter_by_name=vector_name,
                    filter_by_index=vector_index,
                    _stats=stats,
                )
    except _ConstraintsExhausted as e:
        raise ValueError(
            f"Error generating samples for strategy '{name}': {_exhausted_message(name, e)}"
        ) from e
    except _ConstraintError as e:
        # Chained to the constraint's own exception, so the error shows the user's frame
        raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e.__cause__
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
    resolution.constraints = tuple(param.vector_constraints)
    resolution.rejected = dict(stats.rejected)
    if exhaustive:
        resolution.left_out = stats.left_out

    argnames = param.arg_names

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

        # Generate test IDs for better test output readability, from the rows' values.
        # generate_test_ids unwraps a single-argument row once, so a tuple value
        # keeps its full ID.
        ids = generate_test_ids(argnames, [_id_row(s) for s in samples])

        # For single parameters, unwrap the tuples. A pytest.param() sample is a tuple
        # too (ParameterSet); it is passed through unchanged to keep its marks and id.
        if len(argnames) == 1:
            samples = [
                s[0] if isinstance(s, tuple) and not isinstance(s, _ParameterSet) else s
                for s in samples
            ]
        samples, ids = _unique_ids(samples, ids, config)

        return Parametrization(argstr, samples, ids)


def _rootpath(config: pytest.Config | None) -> Path | None:
    """Return the config's rootdir, or None without a config."""
    return getattr(config, "rootpath", None) if config is not None else None


def _where(factory: Callable[..., Any], config: pytest.Config | None) -> str:
    """Return the factory's file, relative to the rootdir when inside it, for the summary."""
    filename = factory_source(factory)[0]
    if filename is None:
        return "<unknown>"
    return display_path(filename, _rootpath(config))


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
