"""
Signature introspection and dataclass detection utilities.

Pure functions — no pytest runtime dependency beyond inspect/dataclasses.
"""

import inspect
import sys
import typing
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

# Common pytest fixtures to exclude from signature validation.
# Mirrors Strategy.PYTEST_FIXTURES (kept in sync; Strategy re-exports this set).
PYTEST_FIXTURES: frozenset[str] = frozenset(
    {
        "request",
        "tmp_path",
        "tmp_path_factory",
        "tmpdir",
        "tmpdir_factory",
        "capsys",
        "capfd",
        "caplog",
        "monkeypatch",
        "pytestconfig",
        "cache",
        "doctest_namespace",
        "recwarn",
        "record_property",
        "record_testsuite_property",
        "record_xml_attribute",
    }
)


def lazy_signature(fn: Any, **kwargs: Any) -> inspect.Signature:
    """
    ``inspect.signature`` that never evaluates annotations.

    Python 3.14 (PEP 649) evaluates annotations lazily, so a test or factory may
    annotate with a name that is not defined yet (a ``TYPE_CHECKING`` import, a
    class defined later). ``inspect.signature`` evaluates them by default and
    raises ``NameError``; ``Format.FORWARDREF`` keeps unresolvable ones as
    ``ForwardRef`` objects instead. Older versions already evaluated them at
    ``def`` time.
    """
    if sys.version_info >= (3, 14):
        from annotationlib import Format

        return inspect.signature(fn, annotation_format=Format.FORWARDREF, **kwargs)
    return inspect.signature(fn, **kwargs)


def validate_signature(
    test_fn,
    argnames: Sequence[str],
    strategy_name: str,
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,  # noqa: ARG001
) -> None:
    """
    Validate that a test function's signature matches the strategy argnames.

    Parameters whose name does not appear in *argnames* are excluded from
    validation (built-in or custom fixtures). A parameter named in *argnames*
    is always a strategy parameter, even when it shares its name with a
    built-in fixture: pytest's parametrize overrides that fixture.

    Args:
        test_fn: The test function to inspect.
        argnames: Expected argument names from the strategy.
        strategy_name: Name of the strategy (used in error messages).
        pytest_fixtures: Built-in fixture names. Kept for backward
            compatibility; every name outside *argnames* is ignored anyway.

    Raises:
        ValueError: When the test-function signature does not match *argnames*.
    """
    sig = lazy_signature(test_fn)
    test_params = list(sig.parameters.keys())

    actual_params = []
    for p in test_params:
        if p not in argnames:
            continue
        actual_params.append(p)

    expected_params = list(argnames)

    if set(expected_params) != set(actual_params):
        missing = set(expected_params) - set(actual_params)
        extra = set(actual_params) - set(expected_params)

        error_msg = f"Test function signature mismatch for strategy '{strategy_name}'!\n"
        error_msg += f"  Strategy provides: {expected_params}\n"
        error_msg += f"  Test function expects: {actual_params}\n"

        if missing:
            error_msg += f"  Missing parameters: {list(missing)}\n"
        if extra:
            error_msg += f"  Extra parameters: {list(extra)}\n"

        raise ValueError(error_msg)


def detect_dataclass_param(
    test_fn,
    argnames: Sequence[str],
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,
    allow_fixtures: bool = True,
) -> tuple[bool, type | None, str | None]:
    """
    Detect dataclass mode and the test parameter that receives the dataclass.

    The test is in dataclass mode when the strategy provides several *argnames*,
    none of them is itself a test parameter, and exactly one parameter is
    annotated with a dataclass whose ``__init__`` fields equal *argnames*.
    ``self``/``cls``, the names in *pytest_fixtures* and every other parameter
    are fixtures and are left alone. When no annotation matches the fields
    exactly but a single parameter is dataclass-annotated, that parameter is
    still chosen so that the field mismatch is reported by
    ``convert_to_dataclass``.

    With ``allow_fixtures=False`` (used when signature validation is off), the
    dataclass parameter must be the only parameter besides ``self``/``cls`` and
    *pytest_fixtures*. A dataclass-typed parameter next to other parameters is
    then treated as a fixture, which may itself consume the *argnames*.

    String annotations (``from __future__ import annotations`` or quoted
    forward references) are resolved in the test module's globals, so the
    dataclass must be defined at module level before the decorator runs.

    Returns:
        ``(True, dataclass_type, param_name)`` in dataclass mode;
        ``(False, None, None)`` otherwise.
    """
    if len(argnames) < 2:
        return False, None, None

    sig = lazy_signature(test_fn)
    if any(name in sig.parameters for name in argnames):
        return False, None, None

    try:
        hints = typing.get_type_hints(test_fn)
    except Exception:
        # One unresolvable annotation (e.g. a fixture type imported under
        # TYPE_CHECKING) must not disable dataclass mode: fall back to the raw
        # annotations and resolve the candidates one by one below.
        hints = {}

    others = [n for n in sig.parameters if n not in ("self", "cls") and n not in pytest_fixtures]
    if not allow_fixtures and len(others) != 1:
        return False, None, None

    candidates: list[tuple[str, type]] = []
    for name in others:
        param = sig.parameters[name]
        annotation = hints.get(name, param.annotation)
        if isinstance(annotation, str):
            annotation = _eval_annotation(test_fn, annotation)
        if isinstance(annotation, type) and is_dataclass(annotation):
            candidates.append((name, annotation))

    wanted = set(argnames)
    exact = [c for c in candidates if {f.name for f in fields(c[1]) if f.init} == wanted]
    if len(exact) == 1:
        param_name, dc_type = exact[0]
    elif not exact and len(candidates) == 1:
        param_name, dc_type = candidates[0]
    else:
        return False, None, None

    return True, dc_type, param_name


def detect_dataclass_mode(
    test_fn,
    argnames: Sequence[str],
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,
) -> tuple[bool, type | None]:
    """
    Detect whether a test function expects a single dataclass parameter.

    Backward-compatible wrapper around :func:`detect_dataclass_param` that
    omits the parameter name.

    Returns:
        ``(True, dataclass_type)`` in dataclass mode; ``(False, None)`` otherwise.
    """
    is_dc_mode, dc_type, _ = detect_dataclass_param(test_fn, argnames, pytest_fixtures)
    return is_dc_mode, dc_type


def _eval_annotation(test_fn, annotation: str) -> Any:
    """Evaluate a string annotation in *test_fn*'s module globals (``None`` on failure)."""
    module_globals = getattr(inspect.unwrap(test_fn), "__globals__", {})
    try:
        return eval(annotation, module_globals)
    except Exception:
        return None
