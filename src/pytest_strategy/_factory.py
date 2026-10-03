"""
Call a strategy factory with the inputs it declares, by name.

A factory declares any of ``nsamples``, ``ctx``, ``rng`` and ``options`` and
receives exactly those, the way a test receives fixtures. :func:`analyse` reads
the factory's signature into a :class:`CallPlan` without calling anything, and
:func:`call_factory` follows that plan to call the factory once.

The set of names is closed: ``*args`` and ``**kwargs`` receive nothing, and a
4.x release adds a factory input as a ``StrategyOptions`` field or under one of
the reserved names, never by passing a value to a name 4.0 left alone.
"""

from __future__ import annotations

import difflib
import functools
import inspect
import os
import random
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._introspection import PYTEST_FIXTURES, lazy_signature
from ._options import StrategyOptions
from ._registry import _describe_factory, _unwrap

# The names a factory receives a value for, in the order the messages list them
INPUTS = ("nsamples", "ctx", "rng", "options")

# Names a factory may not declare, even with a default, and why
RESERVED = {
    "base": "a later 4.x release passes it the strategy the factory extends",
    "config": "factories do not receive the pytest config, because the settings a factory "
    "would read from it are not part of the context's fingerprint or of the command that "
    "reruns a failure; pass them through ctx (pytest_strategies_context)",
    "request": "factories run when pytest collects the tests, before any test's request exists",
}

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True, kw_only=True)
class FactoryInputs:
    """
    The values a factory can receive, by the name of the parameter that declares them.

    Attributes:
        options: The strategy's options; ``nsamples`` is ``options.nsamples``
        rng: The generator the factory's draws come from
        ctx: Returns the context (``pytest_strategies_context``). It is called
            only for a factory that declares ``ctx``, after its signature passed
            the checks.
        why_no_ctx: Returns why the context is None, or None: added to the error
            of a factory that raised with ``ctx`` None (passed, or its default)
    """

    options: StrategyOptions
    rng: random.Random
    ctx: Callable[[], Any]
    why_no_ctx: Callable[[], str | None] = lambda: None


@dataclass(frozen=True)
class Slot:
    """
    One value of a call: an input, or a parameter's default passed explicitly.

    Attributes:
        name: The input's name, or None for a parameter that gets its default
        default: The parameter's default, ``inspect.Parameter.empty`` without one
    """

    name: str | None
    default: Any


@dataclass(frozen=True)
class CallPlan:
    """
    How to call a factory.

    Attributes:
        declares: The input names the factory declares, in signature order
        positional: The values passed by position, in order
        keyword: The inputs passed by keyword
        opaque: True when no signature says what the factory takes, so it is
            called with no arguments
        problem: Why the factory cannot be called, as the end of a message that
            starts with "Strategy factory 'name' (where) ", or None
    """

    declares: tuple[str, ...] = ()
    positional: tuple[Slot, ...] = ()
    keyword: tuple[Slot, ...] = ()
    opaque: bool = False
    problem: str | None = None


def _signature(fn: Any, *, follow_wrapped: bool) -> inspect.Signature | None:
    """Return ``fn``'s signature, or None when it has none (some builtins)."""
    try:
        return lazy_signature(fn, follow_wrapped=follow_wrapped)
    except (TypeError, ValueError):
        return None


def _only_var_args(sig: inspect.Signature) -> bool:
    """Return True if ``sig`` has parameters, all of them ``*args``/``**kwargs``.

    Such a signature (a decorator's wrapper, ``unittest.mock.patch``) does not
    say what the factory takes.
    """
    params = sig.parameters.values()
    return bool(params) and all(p.kind in _VARIADIC for p in params)


def _has_kind(sig: inspect.Signature | None, kind: Any) -> bool:
    """Return True if ``sig`` has a parameter of ``kind``, or is unknown (None)."""
    return sig is None or any(p.kind == kind for p in sig.parameters.values())


def _is_async(factory: Callable[..., Any]) -> bool:
    """Return True for an ``async def`` factory, looking through partials and ``__wrapped__``."""
    if inspect.iscoroutinefunction(factory):
        return True
    fn = _unwrap(factory)
    while isinstance(fn, functools.partial):
        fn = _unwrap(fn.func)
    return inspect.iscoroutinefunction(fn)


def _constructor(cls: Any) -> Any:
    """Return the method a class is called through: ``__init__``, or ``__new__`` without one."""
    return cls.__init__ if cls.__init__ is not object.__init__ else cls.__new__


def _inner(fn: Any) -> Any:
    """
    Return the callable one wrapper below ``fn``, or None at the end of the chain.

    That is ``fn.__wrapped__``. A bound method, a partial, a class (its
    ``__init__``, or its ``__new__`` without one) and a callable object (its
    ``__call__``) step through the function they call, and keep what they add to
    its signature: the bound first argument, the partial's arguments.
    """
    if isinstance(fn, types.MethodType):
        inner = _inner(fn.__func__)
        return None if inner is None else types.MethodType(inner, fn.__self__)
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None:
        return wrapped
    if isinstance(fn, functools.partial):
        inner = _inner(fn.func)
        return None if inner is None else functools.partial(inner, *fn.args, **fn.keywords)
    if inspect.isclass(fn):
        return _inner(types.MethodType(_constructor(fn), fn))
    if callable(fn) and not inspect.isroutine(fn):
        return _inner(fn.__call__)
    return None


def _patchings(fn: Any) -> list[Any] | None:
    """
    Return the ``patchings`` list of ``unittest.mock.patch`` that ``fn`` carries, or None.

    The patched function has it, and a ``functools.wraps`` wrapper or an
    ``update_wrapper`` instance above it carries the same list (copied with its
    ``__dict__``). A partial is looked through, and a class or a callable object
    without a list of its own carries its constructor's or ``__call__``'s.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    patchings = getattr(fn, "patchings", None)
    if patchings is None and inspect.isclass(fn):
        patchings = getattr(_constructor(fn), "patchings", None)
    elif patchings is None and callable(fn) and not inspect.isroutine(fn):
        patchings = getattr(type(fn).__call__, "patchings", None)
    return patchings if isinstance(patchings, list) else None


def _mock_count(passed: list[Any], decider: Any) -> int:
    """
    Return how many mocks ``unittest.mock.patch`` decorators pass the function
    whose signature decides (``decider``), below the wrappers in ``passed``.

    pytest's own rule for test functions: each patch without ``new=`` (and not
    ``patch.multiple``) appends one mock to the positional arguments. Only the
    patches of the wrappers passed through count. A list that ``decider``
    carries too belongs to a function below it, which gets those mocks itself.
    """
    own = _patchings(decider)
    lists: list[list[Any]] = []
    for wrapper in passed:
        patchings = _patchings(wrapper)
        if patchings and patchings is not own and all(patchings is not p for p in lists):
            lists.append(patchings)
    sentinels = [
        getattr(sys.modules.get(module), "DEFAULT", None) for module in ("mock", "unittest.mock")
    ]
    return sum(
        1
        for patchings in lists
        for p in patchings
        if not getattr(p, "attribute_name", None)
        and any(s is not None and getattr(p, "new", None) is s for s in sentinels)
    )


def _unknown(
    param: inspect.Parameter,
    *,
    first_positional: bool,
    after_input: bool,
    mocked: tuple[str, ...],
    has_nsamples: bool,
) -> str:
    """
    Describe a parameter without a default that the plugin does not provide.

    Args:
        param: The parameter
        first_positional: Whether it is the first positional parameter after
            the ones that receive mocks, which 3.0 passed nsamples to
        after_input: Whether it is positional and comes after an input, where
            a mock parameter is out of place
        mocked: The parameters mock.patch passes its mocks to
        has_nsamples: Whether the factory declares nsamples
    """
    name = param.name
    message = (
        f"has a parameter '{name}', which the plugin does not provide. Factories receive "
        f"arguments by name: {', '.join(INPUTS)}."
    )
    if name in ("self", "cls"):
        return message + (
            f" '{name}' is not passed either: register a bound method (instance.method, or "
            "Class.method for a classmethod) or a staticmethod, not the function inside the "
            "class body."
        )
    if name in PYTEST_FIXTURES:
        return message + (
            f" '{name}' is a pytest fixture, but factories run at collection, before fixtures "
            "exist; use ctx (the pytest_strategies_context hook's result) for configuration."
        )
    if first_positional and not has_nsamples:
        message += (
            " Did you mean 'nsamples'? Since 4.0 the plugin no longer passes nsamples by "
            "position to a parameter with another name: rename it, or give it a default if "
            "the plugin should leave it alone."
        )
    else:
        close = difflib.get_close_matches(name, INPUTS, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        message += hint + " Rename it, or give it a default if the plugin should leave it alone."
    if mocked:
        names = ", ".join(f"'{m}'" for m in mocked)
        message += (
            " The factory is decorated with mock.patch, which passes its mocks to its first "
            f"parameters ({names})"
        )
        message += (
            f": if '{name}' is a mock parameter, put the mock parameters first, before the "
            "ones the plugin passes."
            if after_input
            else "."
        )
    return message


def analyse(factory: Callable[..., Any]) -> CallPlan:
    """
    Read how to call ``factory`` from its signature, without calling it.

    The signature of the callable that is called decides (a decorator's wrapper,
    not the function it wraps). When it has none, or only ``*args``/``**kwargs``,
    the chain of wrapped functions (``__wrapped__``, also behind a partial, a
    bound method, a callable object's ``__call__`` or a class's ``__init__``) is
    followed one level at a time, and the first signature that says what it
    takes decides: with stacked decorators, that of the outermost wrapper that
    names its parameters. When no level has one, the factory is opaque and is
    called with no arguments.

    Each parameter named after an input gets that input. Any other parameter
    with a default keeps it, and ``*args``/``**kwargs`` receive nothing.
    Positional-only inputs are passed by position (with the defaults of
    positional-only parameters before them), the others by keyword. When a
    wrapper passed through has ``*args`` and no ``**kwargs``, every input goes
    by position, in the deciding signature's order. ``unittest.mock.patch``
    passes its mocks to the first parameters, which are skipped.

    Returns:
        The plan. Its ``problem`` is set for a factory that cannot be called:
        an ``async def`` factory, a reserved name, a parameter without a default
        that the plugin does not provide, a mock on an input's position, or an
        input that the wrapper cannot pass.
    """
    if _is_async(factory):
        return CallPlan(
            problem="is an async function: async factories are not supported. Return the "
            "Parameter from a plain function."
        )
    # Down the wrapper chain to the first signature that says what it takes
    level: Any = factory
    sig = _signature(factory, follow_wrapped=False)
    passed: list[tuple[Any, inspect.Signature | None]] = []
    while sig is None or _only_var_args(sig):
        passed.append((level, sig))
        level = _inner(level)
        if level is None or len(passed) >= sys.getrecursionlimit():
            # The chain ends (or loops) without one
            return CallPlan(opaque=True)
        sig = _signature(level, follow_wrapped=False)

    params = list(sig.parameters.values())
    mocks = _mock_count([wrapper for wrapper, _ in passed], level)
    taken: list[inspect.Parameter] = []
    if mocks:
        taken = [p for p in params if p.kind in _POSITIONAL][:mocks]
        for p in taken:
            if p.name in INPUTS or p.name in RESERVED:
                names = ", ".join(f"'{t.name}'" for t in taken)
                return CallPlan(
                    problem=f"is decorated with mock.patch, which passes its mocks to its first "
                    f"parameters ({names}), so '{p.name}' would receive a mock. Put the mock "
                    "parameters first, before the ones the plugin passes."
                )
        params = [p for p in params if all(p is not t for t in taken)]

    declares = tuple(p.name for p in params if p.name in INPUTS and p.kind not in _VARIADIC)
    first_positional = next((p for p in params if p.kind in _POSITIONAL), None)
    unknown: str | None = None
    after_input = False
    for p in params:
        if p.kind in _VARIADIC:
            continue
        if p.name in RESERVED:
            return CallPlan(
                problem=f"has a parameter '{p.name}', a name the plugin reserves: "
                f"{RESERVED[p.name]}. Rename it."
            )
        if p.name not in INPUTS and p.default is _EMPTY and unknown is None:
            unknown = _unknown(
                p,
                first_positional=p is first_positional,
                after_input=after_input and p.kind in _POSITIONAL,
                mocked=tuple(t.name for t in taken),
                has_nsamples="nsamples" in declares,
            )
        after_input = after_input or p.name in INPUTS
    if unknown is not None:
        return CallPlan(problem=unknown)

    # A wrapper passed through with *args and no **kwargs can pass only positions
    by_position = any(not _has_kind(s, inspect.Parameter.VAR_KEYWORD) for _, s in passed)
    if by_position:
        for p in params:
            if p.kind == p.KEYWORD_ONLY and p.name in INPUTS:
                return CallPlan(
                    problem=f"has a keyword-only parameter '{p.name}', but the wrapper that "
                    "calls it takes only *args, so the plugin can pass it nothing by keyword: "
                    "give the wrapper **kwargs too."
                )
        positional = [p for p in params if p.kind in _POSITIONAL]
        keyword: list[inspect.Parameter] = []
    else:
        positional = [p for p in params if p.kind == p.POSITIONAL_ONLY]
        keyword = [p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    # Positions stop at the last input; the parameters before it keep their defaults
    last = max((i for i, p in enumerate(positional) if p.name in INPUTS), default=-1)
    if mocks and last >= 0:
        # mock.patch appends its mocks after the positional arguments it is given
        first = next(p.name for p in positional if p.name in INPUTS)
        return CallPlan(
            problem=f"is decorated with mock.patch, which passes its mocks after the "
            f"positional arguments, so '{first}' cannot be passed by position. Make it a "
            "parameter that can be passed by keyword."
        )
    return CallPlan(
        declares=declares,
        positional=tuple(
            Slot(p.name if p.name in INPUTS else None, p.default) for p in positional[: last + 1]
        ),
        keyword=tuple(Slot(p.name, p.default) for p in keyword if p.name in INPUTS),
    )


def _factory_error(
    name: str, nsamples: int | str, error: Exception, plan: CallPlan, note: str | None = None
) -> ValueError:
    """
    Return the error reported when a strategy factory raises ``error``, ending with
    ``note`` when there is one.
    """
    message = (
        f"Error calling strategy factory '{name}' (nsamples={nsamples!r}): "
        f"{type(error).__name__}: {error}"
    )
    if note:
        message += f". {note}"
    if plan.opaque and isinstance(error, TypeError):
        message += (
            ". The plugin called it with no arguments, because its signature takes only "
            "*args/**kwargs or cannot be read: if a decorator's wrapper hides the factory's "
            "parameters, decorate the wrapper with @functools.wraps(factory)."
        )
    return ValueError(message)


def call_factory(
    name: str,
    factory: Callable[..., Any],
    inputs: FactoryInputs,
    *,
    rootpath: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Call a strategy factory once, passing it the inputs it declares (see :func:`analyse`).

    ``ctx`` is computed only for a factory that declares it. When it is None and
    the parameter has a default (declared, or bound with ``functools.partial``),
    it is not passed, so the factory keeps the default.

    Args:
        name: Name of the strategy, for messages
        factory: The factory to call
        inputs: The values the factory can receive
        rootpath: The rootdir, so messages show the factory's file relative to it

    Returns:
        Whatever the factory returns

    Raises:
        ValueError: If the signature cannot be called (before the factory and
            the context hook run), if the context hook raised for a factory that
            declares ``ctx``, or if the factory raised (chained to its exception)
    """
    plan = analyse(factory)
    if plan.problem is not None:
        where = _describe_factory(factory, rootpath=rootpath)
        raise ValueError(f"Strategy factory '{name}' ({where}) {plan.problem}")
    values: dict[str, Any] = {
        "nsamples": inputs.options.nsamples,
        "rng": inputs.rng,
        "options": inputs.options,
    }
    if "ctx" in plan.declares:
        try:
            values["ctx"] = inputs.ctx()
        except Exception as e:
            raise ValueError(
                f"Strategy factory '{name}' has a 'ctx' parameter, but the "
                f"pytest_strategies_context hook raised {type(e).__name__}: {e}"
            ) from e

    def keeps_default(slot: Slot) -> bool:
        return slot.name == "ctx" and values["ctx"] is None and slot.default is not _EMPTY

    args = [
        slot.default if slot.name is None or keeps_default(slot) else values[slot.name]
        for slot in plan.positional
    ]
    kwargs = {
        slot.name: values[slot.name]
        for slot in plan.keyword
        if slot.name is not None and not keeps_default(slot)
    }
    try:
        result = factory(*args, **kwargs)
    except Exception as e:
        # A factory whose ctx was None (passed, or a default of None that it kept)
        # is told why, when a conftest.py elsewhere implements the hook (the
        # folder's own conftest.py files and above do not)
        received_none = any(
            slot.name == "ctx"
            and values["ctx"] is None
            and (slot.default is _EMPTY or slot.default is None)
            for slot in (*plan.positional, *plan.keyword)
        )
        note = inputs.why_no_ctx() if received_none else None
        raise _factory_error(name, inputs.options.nsamples, e, plan, note) from e
    if inspect.iscoroutine(result):
        # Closed, so no "coroutine was never awaited" warning follows the error
        result.close()
        where = _describe_factory(factory, rootpath=rootpath)
        raise ValueError(
            f"Strategy factory '{name}' ({where}) returned a coroutine: async factories are "
            "not supported. Return the Parameter from a plain function."
        )
    return result
