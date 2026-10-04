"""Resolve a strategy into the parametrization of a test.

This is the orchestration the plugin runs for each test marked with
``@strategy`` (from ``pytest_generate_tests``): read CLI options, call the
factory, generate vectors, and build the arguments of ``pytest.mark.parametrize``
for either record (dataclass) or named-parameter mode.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import reprlib
import warnings
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, cast

import pytest

from ._context import Answer, FolderContext
from ._factory import FactoryInputs, call_factory
from ._ids import (
    ID_FORMATS,
    generate_dataclass_ids,
    generate_test_ids,
    make_unique_ids,
    names_id,
)
from ._introspection import validate_signature
from ._options import constraint_off_item
from ._records import (
    RecordParam,
    convert_to_dataclass,
    detect_record_param,
    matching_record_params,
    record_hints,
)
from ._registry import _describe_factory, display_path, factory_source
from ._runtime import Resolution, runtime
from ._streams import StreamKey, seed_part
from ._vector import VectorInfo
from ._warnings import PytestStrategiesWarning
from .parameters import (
    Parameter,
    _check_ids,
    _constraint_failure,
    _ConstraintsExhausted,
    _forget_constraint_failure,
    _GenerationStats,
    _ParameterSet,
    _Row,
)
from .rng import RNG, SequenceLike, _Stream

# Default for the strategies_max_exhaustive ini option
DEFAULT_MAX_EXHAUSTIVE = 100_000

# The guard's warning (D5), emitted once per strategy and test when something drew
# from the ambient generator while the test's rows were generated. The re-emitted
# warning starts with "Strategy 'name' (test): ".
_OUTSIDE_STREAMS = (
    "something drew from the plugin's generator while the rows were generated, outside "
    "the arguments' own streams: a vector constraint that calls RNG.*, or an RNG type "
    "that draws from a generator kept from the factory (rng). Those draws depend on what "
    "was drawn before them, so the rows they affect change when other rows or tests "
    "change. Draw only in an RNG type's generate(), from RNG.generator() or the RNG.* "
    "helpers."
)


class Parametrization(NamedTuple):
    """
    The parametrization of one test by one strategy: its argument names, the rows
    the test receives (a vector's ``pytest.param`` where it has marks), their IDs
    and the VectorInfo of each, and in named mode the test parameters annotated
    with the strategy's record type, with the error to report when no fixture or
    parametrization gives one a value.
    """

    argnames: str
    values: list[Any]
    ids: list[str]
    unfilled: tuple[tuple[str, str], ...] = ()
    infos: tuple[VectorInfo, ...] = ()

    def params(self) -> list[Any]:
        """
        Return the rows as ``pytest.mark.parametrize`` receives them:
        ``pytest.param(*values, id=..., marks=[*the vector's marks, strategy mark])``,
        whose ``strategy`` mark holds the row's VectorInfo (the plugin stores it on
        the item, see ``pytest_itemcollected``).
        """
        single = "," not in self.argnames
        strategy_mark = pytest.mark.strategy
        params = []
        for value, row_id, info in zip(self.values, self.ids, self.infos, strict=True):
            if isinstance(value, _ParameterSet):
                values, marks = tuple(value.values), list(value.marks)
            else:
                # As pytest reads a row: one value for a single argument, else a tuple
                values, marks = ((value,) if single else tuple(value)), []
            marks.append(strategy_mark.with_args(info))
            params.append(pytest.param(*values, id=row_id, marks=marks))
        return params


def _skipped_param(reason: str, width: int) -> Any:
    """Return the single skipped row that stands in for a strategy without values."""
    return pytest.param(*([None] * width), marks=pytest.mark.skip(reason=reason))


class _attributed_warnings:
    """
    Re-emit the PytestStrategiesWarnings raised in the block at the test function,
    prefixed with the strategy and the test.

    Raised during vector generation, they would otherwise point into this module and
    not say which strategy or test they are about. Other warnings pass unchanged. An
    exception raised in the block goes up unchanged, and the warnings are dropped.

    A class (named like warnings.catch_warnings): a contextlib.contextmanager assigns
    the exception's ``__traceback__`` on the way out, which a frozen dataclass
    exception rejects, so a constraint's exception would become a FrozenInstanceError.
    """

    def __init__(self, name: str, test_fn: Callable[..., Any]) -> None:
        self._name = name
        self._test_fn = test_fn
        self._catcher = warnings.catch_warnings(record=True)
        self._caught: list[warnings.WarningMessage] = []

    def __enter__(self) -> None:
        self._caught = self._catcher.__enter__()
        warnings.simplefilter("always", PytestStrategiesWarning)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._catcher.__exit__(exc_type, exc, tb)
        if exc is None:
            self._reemit()

    def _reemit(self) -> None:
        """Re-emit the warnings caught in the block, in order."""
        name, test_fn = self._name, self._test_fn
        fn = inspect.unwrap(test_fn)
        code = getattr(fn, "__code__", None)
        for w in self._caught:
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


def _unique_ids(ids: list[str], config: pytest.Config | None) -> list[str]:
    """
    Return the test IDs to parametrize with, each duplicate suffixed.

    The rows of one strategy have unique IDs in the names format, but the values
    format repeats the ID of a repeated row, an ids= callable may give several rows
    one ID, and pytest's strict_parametrization_ids makes duplicate IDs a collection
    error instead of suffixing them.
    """
    escape = config is None or not config.getini(
        "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"
    )
    return make_unique_ids(ids, escape=escape)


def ids_format(config: pytest.Config | None) -> str:
    """
    Return the format of the test IDs: the strategies_ids ini option, "names" or
    "values". The plugin checks the option in pytest_configure.

    Without a config, or for a config without the plugin's ini options (a test's
    stand-in), it is "names".
    """
    if config is None:
        return "names"
    try:
        raw = config.getini("strategies_ids")
    except ValueError:
        # The plugin's ini options are not registered (a config without the plugin)
        return "names"
    # Anything but "values" is "names": a real config's other values were rejected
    return "values" if isinstance(raw, str) and raw.strip() == "values" else "names"


def check_ids_format(config: pytest.Config) -> None:
    """
    Check the strategies_ids ini option.

    Raises:
        pytest.UsageError: For a value other than "names" or "values", or one that
            pytest cannot read as a string
    """
    formats = " or ".join(repr(f) for f in ID_FORMATS)
    try:
        raw = config.getini("strategies_ids")
    except TypeError as error:
        # A value that is not a string in a native TOML configuration file
        # (pytest 9's pytest.toml or [tool.pytest]), which pytest refuses to read
        raise pytest.UsageError(f"strategies_ids must be {formats}; {error}") from None
    if not isinstance(raw, str) or raw.strip() not in ID_FORMATS:
        raise pytest.UsageError(f"strategies_ids must be {formats}, got {raw!r}")


def _row_ids(rows: list[_Row]) -> list[str]:
    """Return the rows' test IDs in the names format."""
    return [names_id(row.kind, row.name, row.j, row.labels) for row in rows]


def _custom_ids(
    name: str, ids: Callable[[VectorInfo], object], infos: Sequence[VectorInfo]
) -> list[str]:
    """
    Return the rows' test IDs from a ``Parameter(ids=...)`` callable, before
    duplicates are suffixed.

    The callable is called once per row with the row's VectorInfo, whose id is the
    row's ID in the effective format, and returns the ID to use, or None to keep
    that one. The skipped row of an empty skip_if_empty sequence keeps its ID
    without a call.

    Raises:
        ValueError: When the callable raises (chained to its exception, so the
            error shows the callable's frames), or returns anything but a
            non-empty str or None
    """
    result = []
    for info in infos:
        if info.kind == "skipped":
            result.append(info.id)
            continue
        try:
            row_id = ids(info)
        except Exception as e:
            raise ValueError(
                f"Strategy '{name}': ids= raised {type(e).__name__} for row {info.id}: {e}"
            ) from e
        if row_id is None:
            result.append(info.id)
        elif isinstance(row_id, str) and row_id:
            result.append(row_id)
        else:
            expected = "a non-empty str" if isinstance(row_id, str) else "a str"
            raise ValueError(
                f"Strategy '{name}': ids= returned {reprlib.repr(row_id)} for row {info.id}; "
                f"return {expected} or None"
            )
    return result


def _vector_infos(
    rows: list[_Row],
    ids: Sequence[str],
    *,
    strategy: str,
    origin: str | None,
    arg_names: Sequence[str],
    context: str | None,
    constraints_off: tuple[str, ...],
) -> tuple[VectorInfo, ...]:
    """
    Return the VectorInfo of each row, whose final test ID is in ``ids``. ``context``
    is the fingerprint of the context the factory received, or None.
    """
    seed = runtime.run_seed()
    order = {arg: position for position, arg in enumerate(arg_names)}
    # The enumerated arguments in declaration order, per set of enumerated arguments
    enumerated: dict[tuple[str, ...], tuple[str, ...]] = {}
    infos = []
    for row, row_id in zip(rows, ids, strict=True):
        names = tuple(arg for arg, _ in row.pos)
        declared = enumerated.get(names)
        if declared is None:
            declared = enumerated[names] = tuple(sorted(names, key=order.__getitem__))
        infos.append(
            VectorInfo(
                strategy=strategy,
                origin=origin,
                kind=row.kind,
                name=row.name,
                index=row.index,
                enumerated=declared,
                values=row.values,
                id=row_id,
                seed=seed,
                context=context,
                constraints_off=constraints_off,
            )
        )
    return tuple(infos)


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


def _test_file(test_fn: Callable[..., Any]) -> str | None:
    """Return the test's file, or None when it is unknown."""
    fn = inspect.unwrap(test_fn)
    file: str | None = getattr(fn, "__globals__", {}).get("__file__")
    if file:
        return file
    try:
        return inspect.getsourcefile(fn)
    except TypeError:
        return None


def _fallback_test_key(test_fn: Callable[..., Any], config: pytest.Config | None) -> str:
    """
    Return the test's key for its random streams without a node ID (a call that
    has no metafunc): the test's location and its qualified name, with ``::``
    for the dots, as pytest writes a node ID.
    """
    return f"{_test_location(test_fn, config)}::{test_fn.__qualname__.replace('.', '::')}"


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
    # A strategy whose name has a "," or whitespace cannot be targeted: the item is
    # then the bare name, which turns the constraint off everywhere
    return (
        f"{error.detail} {advice}, or turn one off for this run with "
        f"--strategy-constraint-off={constraint_off_item(name, error.strictest)}."
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
    fixturenames: Collection[str] | None = None,
    test_key: str | None = None,
    context: FolderContext | None = None,
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
        fixturenames: The names the test and its fixtures ask for
            (``metafunc.fixturenames``), or None to read only the test's parameters
        test_key: The test's part of its random stream key: its node ID without
            parameters (``metafunc.definition.nodeid``), so inherited methods in
            two subclasses get rows of their own. None uses the test's location
            and qualified name.
        context: The context of the test's folder, which a factory that declares
            ``ctx`` receives (``runtime.test_context(metafunc.definition)``). None
            uses the context of the test's file's folder (``runtime.path_context``).

    Raises:
        ValueError: With a message naming the strategy when the factory, the
            generation or the signature check fails
    """
    # The CLI options, read once per session
    options = runtime.strategy_options(name, config)
    vector_mode = options.mode
    vector_name = options.vector_name
    vector_index = options.vector_index

    # The key T of this strategy and test (streams v1, D5). The factory draws from
    # T/"factory", and every random and exhaustive row draws each argument from a
    # stream below T/"row", so the factory's draws, the other rows and the other
    # arguments do not move a row's values.
    if test_key is None:
        test_key = _fallback_test_key(test_fn, config)
    stream_key = StreamKey.root(seed_part(runtime.run_seed()), "test", name, test_key)

    # Call the factory with the inputs it declares by name, on its own stream: rng
    # is RNG.generator() during the call. Its nsamples is the --nsamples value,
    # "auto", or 10 without the option, never None (FR-8). The count the rows use is
    # resolved below, once the Parameter's own nsamples is known.
    folder = context if context is not None else runtime.path_context(_test_file(test_fn))
    # The folder's answer, once the factory asked for ctx
    asked: list[Answer] = []

    def ctx() -> Any:
        answer = folder.answer()
        asked.append(answer)
        return answer.get()

    with _Stream(stream_key.child("factory")) as rng:
        inputs = FactoryInputs(options=options, rng=rng, ctx=ctx, why_no_ctx=folder.why_none)
        result = call_factory(name, factory, inputs, rootpath=_rootpath(config))
    param = check_factory_result(name, factory, result)
    # The fingerprint of the context the factory received (D9): only a factory that
    # declares ctx asks for it, and only an answer that is not None has one
    fingerprint = asked[0].fingerprint if asked else None
    if fingerprint is not None:
        runtime.record_context(asked[0].label, test_key)

    # --strategy-constraint-off: the names this strategy turns off are the ones its
    # Parameter has, in evaluation order. The plugin checks once collection ends that
    # each item matched a constraint of some resolved strategy, so the names are
    # recorded before anything can fail, whatever the run evaluates.
    constraint_names = tuple(param.vector_constraints)
    runtime.record_constraints(name, constraint_names)
    constraints_off = tuple(c for c in constraint_names if c in options.constraints_off)

    # The format of the test IDs: Parameter(ids="names" or "values"), else the ini
    # option, which a callable ids= starts from too
    try:
        _check_ids(param.ids)
    except ValueError as e:
        raise ValueError(f"Strategy '{name}': {e}") from None
    ids_option = param.ids
    id_format = ids_option if isinstance(ids_option, str) else ids_format(config)

    resolution = Resolution(
        strategy=name, where=_where(factory, config), constraints_off=constraints_off
    )

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

    # Generate the rows with the CLI options. "auto" replaces only the random rows
    # with the exhaustive ones: CLI filters, vector modes and directed vectors apply
    # exactly as they do for a finite count.
    # The rejections per constraint (and the combinations "auto" left out), for -v
    stats = _GenerationStats()
    auto = effective_nsamples == "auto"
    exhaustive = auto and not filtered and vector_mode not in ("test", "directed_only")
    # An RNG.seed() call while the rows are drawn changes the seed for nothing after
    # them, as in the plugin's other streams
    seed = RNG._seed
    # The guard (D5): each drawn argument of a row draws from a generator of its
    # own, so the ambient generator changes while the rows are generated only when
    # something else draws from it: a constraint that calls RNG.*, or an RNG type
    # that draws from a generator kept from the factory, whose rng is this object.
    # Its position is the pending key of a stream nothing has drawn from yet, which
    # costs nothing to read, or else its state. For a test function that stream is
    # its module's; a class is collected outside it, so a test method reads the state.
    ambient = RNG._ambient
    position = ambient._position()
    try:
        # Warnings raised while generating name the strategy and the test
        with _attributed_warnings(name, test_fn):
            # A count is passed as it is: a Parameter.nsamples reassigned to a
            # non-int after construction fails here, naming the value
            rows = param._generate_rows(
                0 if auto else cast(int, effective_nsamples),
                exhaustive=auto,
                mode=vector_mode,
                filter_by_name=vector_name,
                filter_by_index=vector_index,
                constraints_off=constraints_off,
                stats=stats,
                key=stream_key,
            )
            if ambient._position() != position:
                warnings.warn(_OUTSIDE_STREAMS, PytestStrategiesWarning, stacklevel=1)
    except _ConstraintsExhausted as e:
        raise ValueError(
            f"Error generating samples for strategy '{name}': {_exhausted_message(name, e)}"
        ) from e
    except Exception as e:
        failure = _constraint_failure(e)
        if failure is not None:
            # A constraint raised (a KeyError too): chained to its own exception, so
            # the error shows the user's frame, whose note names the option too
            failure.attach(e, "--strategy-constraint-off")
            _forget_constraint_failure(e)
            raise ValueError(
                f"Error generating samples for strategy '{name}': "
                f"{failure.message('--strategy-constraint-off')}"
            ) from e
        if not (filtered and isinstance(e, (KeyError, IndexError))):
            raise ValueError(f"Error generating samples for strategy '{name}': {e}") from e
        # Filtering by name (KeyError) or index (IndexError) for a vector this
        # strategy doesn't have gives it no rows, so CLI filtering works across
        # strategies. The plugin reports a filter that no strategy in the run matches.
        rows = []
        runtime.record_vector_filter(name, False, list(param.directed_vectors))
    else:
        if filtered:
            runtime.record_vector_filter(name, True)
        # A skipped Parameter gives one "skipped" row, which stands in for its values
        # below. A vector filter that names none of its vectors keeps the empty set,
        # as for any strategy.
        skip_reason = param.skip_reason
    finally:
        RNG._seed = seed

    _count_rows(resolution, rows)
    resolution.constraints = constraint_names
    resolution.rejected = dict(stats.rejected)
    if exhaustive:
        resolution.left_out = stats.left_out

    argnames = param.arg_names

    runtime.record_resolution(resolution)

    def row_infos(ids: list[str]) -> tuple[VectorInfo, ...]:
        """The rows' VectorInfos, given their IDs."""
        return _vector_infos(
            rows,
            ids,
            strategy=name,
            origin=_origin(factory, config),
            arg_names=argnames,
            context=fingerprint,
            constraints_off=constraints_off,
        )

    def final_ids(ids: list[str]) -> tuple[list[str], tuple[VectorInfo, ...]]:
        """
        Return the rows' final test IDs and their VectorInfos, given their IDs in
        the effective format: a callable ids= replaces them first, then each
        duplicate is suffixed. The VectorInfos carry the final IDs.
        """
        if not callable(ids_option):
            ids = _unique_ids(ids, config)
            return ids, row_infos(ids)
        # The callable sees each row's VectorInfo with its ID in the effective format
        infos = row_infos(ids)
        ids = _unique_ids(_custom_ids(name, ids_option, infos), config)
        return ids, tuple(
            info if info.id == row_id else dataclasses.replace(info, id=row_id)
            for info, row_id in zip(infos, ids, strict=True)
        )

    # Record mode: the test receives the row as one record when neither the test nor
    # any fixture it uses asks for an argument by name, and exactly one parameter is
    # annotated with a record type whose fields are the arguments. validate_signature
    # does not change the choice.
    try:
        record = detect_record_param(
            test_fn, argnames, fixturenames, pytest_fixtures=pytest_fixtures
        )
    except ValueError as e:
        raise ValueError(f"Strategy '{name}': {e}") from None

    if record is not None:
        # RECORD MODE: Convert samples to dataclass instances. A fixture that asks for
        # an argument by name would get nothing, since only the record parameter is
        # parametrized: the rule's first condition rules that out.
        assert fixturenames is None or not set(argnames) & set(fixturenames)
        dc_type, dc_param = record.record_type, record.name
        if skip_reason is not None:
            # No values to build an instance from, but the dataclass must still match
            # the strategy, as the signature check does in named mode
            try:
                convert_to_dataclass([], argnames, dc_type)
            except Exception as e:
                raise ValueError(
                    f"Error converting samples to dataclass for strategy '{name}': {e}"
                ) from e
            ids, infos = final_ids(["skipped"])
            return Parametrization(dc_param, [_skipped_param(skip_reason, 1)], ids, infos=infos)
        try:
            dataclass_samples = convert_to_dataclass(
                [row.values for row in rows], argnames, dc_type
            )
        except Exception as e:
            raise ValueError(
                f"Error converting samples to dataclass for strategy '{name}': {e}"
            ) from e

        if id_format == "names":
            ids = _row_ids(rows)
        else:
            # The 3.0 format: the record's init=True fields and their values
            ids = generate_dataclass_ids(dataclass_samples, dc_type)

        # A pytest.param() vector's marks go on its instance
        params = [
            pytest.param(inst, marks=row.param.marks) if row.param is not None else inst
            for row, inst in zip(rows, dataclass_samples)
        ]

        # Parametrize the dataclass parameter chosen by detection
        ids, infos = final_ids(ids)
        return Parametrization(dc_param, params, ids, infos=infos)

    else:
        # NAMED PARAMETERS MODE: Standard behavior

        # Validate signature if requested. An argument that a fixture of the test asks
        # for is taken by that fixture.
        unfilled: tuple[tuple[str, str], ...] = ()
        if validate:
            try:
                validate_signature(
                    test_fn,
                    argnames,
                    name,
                    pytest_fixtures=pytest_fixtures,
                    fixturenames=fixturenames,
                )
            except ValueError as e:
                # Say why no parameter receives the row as a record, if one could
                hints = "".join(
                    f"  {hint}\n"
                    for hint in record_hints(
                        test_fn, argnames, fixturenames, pytest_fixtures=pytest_fixtures
                    )
                )
                raise ValueError(
                    f"Signature validation failed for strategy '{name}': {e}{hints}"
                ) from e
            # A test written for record mode whose fixtures ask for every argument
            # passes the check above, but its record parameter gets nothing from the
            # strategy. The plugin reports it once the test is parametrized, unless a
            # fixture or a parametrization gives it a value.
            records = matching_record_params(test_fn, argnames, pytest_fixtures=pytest_fixtures)
            if records:
                why = record_hints(test_fn, argnames, fixturenames, pytest_fixtures=pytest_fixtures)
                unfilled = tuple(
                    (record.name, _unfilled_message(name, record, why)) for record in records
                )

        # Create comma-separated string of parameter names for pytest.mark.parametrize
        argstr = ",".join(argnames)

        samples: list[Any]
        if skip_reason is not None:
            samples = [_skipped_param(skip_reason, len(argnames))]
            ids = ["skipped"]
        else:
            if id_format == "names":
                ids = _row_ids(rows)
            else:
                # The 3.0 format, from the rows' values. generate_test_ids unwraps a
                # single-argument row once, so a tuple value keeps its full ID.
                ids = generate_test_ids(argnames, [row.values for row in rows])
            # A pytest.param() vector is passed as it is, to keep its marks. For a
            # single parameter, each other row passes its one value.
            if len(argnames) == 1:
                samples = [row.values[0] if row.param is None else row.param for row in rows]
            else:
                samples = [row.sample for row in rows]

        ids, infos = final_ids(ids)
        return Parametrization(argstr, samples, ids, unfilled, infos)


def _unfilled_message(name: str, record: RecordParam, hints: list[str]) -> str:
    """
    Describe a record parameter that no fixture or parametrization gives a value,
    in a test whose fixtures take the strategy's arguments by name.
    """
    return (
        f"Signature validation failed for strategy '{name}': parameter '{record.name}' "
        f"({record.record_type.__name__}) gets no value: it is not one of the strategy's "
        "arguments, and no fixture or parametrization provides it.\n"
        + "".join(f"  {hint}\n" for hint in hints)
    )


def _rootpath(config: pytest.Config | None) -> Path | None:
    """Return the config's rootdir, or None without a config."""
    return getattr(config, "rootpath", None) if config is not None else None


def _where(factory: Callable[..., Any], config: pytest.Config | None) -> str:
    """Return the factory's file, relative to the rootdir when inside it, for the summary."""
    filename = factory_source(factory)[0]
    if filename is None:
        return "<unknown>"
    return display_path(filename, _rootpath(config))


def _origin(factory: Callable[..., Any], config: pytest.Config | None) -> str | None:
    """
    Return where the factory is defined, as ``file:line`` with the file relative
    to the rootdir when inside it, or None when its file is unknown.
    """
    filename, _, line = factory_source(factory)
    if filename is None:
        return None
    where = display_path(filename, _rootpath(config))
    return where if line is None else f"{where}:{line}"


def _count_rows(resolution: Resolution, rows: list[_Row]) -> None:
    """Count a strategy's rows by kind, for the -v summary."""
    kinds = Counter(row.kind for row in rows)
    resolution.directed = kinds["directed"]
    resolution.test = kinds["test"]
    resolution.random = kinds["random"]
    resolution.exhaustive = kinds["exhaustive"]
    resolution.skipped = kinds["skipped"]


def resolve_and_parametrize(
    name: str,
    test_fn: Callable[..., Any],
    *,
    registry: Mapping[str, Callable[..., Any]],
    config: pytest.Config | None,
    pytest_fixtures: set[str],
    validate: bool = True,
    fixturenames: Collection[str] | None = None,
    test_key: str | None = None,
) -> Callable[..., Any]:
    """Build the parametrization for a registered strategy and apply it to ``test_fn``."""
    parametrization = build_parametrization(
        name,
        registry[name],
        test_fn,
        config=config,
        pytest_fixtures=pytest_fixtures,
        validate=validate,
        fixturenames=fixturenames,
        test_key=test_key,
    )
    mark = pytest.mark.parametrize(parametrization.argnames, parametrization.params())
    return cast(Callable[..., Any], mark(test_fn))
