"""
Signature introspection utilities. Record mode's detection is in ``_records``.

Pure functions — no pytest runtime dependency beyond inspect.
"""

import inspect
import sys
from collections.abc import Callable, Collection, Sequence
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
    test_fn: Callable[..., Any],
    argnames: Sequence[str],
    strategy_name: str,
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,  # noqa: ARG001
    fixturenames: Collection[str] | None = None,
) -> None:
    """
    Validate that a test function's signature matches the strategy argnames.

    Parameters whose name does not appear in *argnames* are excluded from
    validation (built-in or custom fixtures). A parameter named in *argnames*
    is always a strategy parameter, even when it shares its name with a
    built-in fixture: pytest's parametrize overrides that fixture. An argname
    that a fixture of the test asks for (it is in *fixturenames* but not a
    parameter) is taken too: that fixture receives it.

    Args:
        test_fn: The test function to inspect.
        argnames: Expected argument names from the strategy.
        strategy_name: Name of the strategy (used in error messages).
        pytest_fixtures: Built-in fixture names. Kept for backward
            compatibility; every name outside *argnames* is ignored anyway.
        fixturenames: The names the test asks for, through its fixtures too
            (``metafunc.fixturenames``), or None to read only its parameters.

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
    asked = [
        name
        for name in expected_params
        if fixturenames is not None and name in fixturenames and name not in actual_params
    ]

    if set(expected_params) != set(actual_params) | set(asked):
        missing = [p for p in expected_params if p not in actual_params and p not in asked]
        extra = set(actual_params) - set(expected_params)

        error_msg = f"Test function signature mismatch for strategy '{strategy_name}'!\n"
        error_msg += f"  Strategy provides: {expected_params}\n"
        error_msg += f"  Test function expects: {actual_params}\n"
        if asked:
            error_msg += f"  Fixtures of the test ask for: {asked}\n"

        if missing:
            error_msg += f"  Missing parameters: {missing}\n"
        if extra:
            error_msg += f"  Extra parameters: {list(extra)}\n"

        raise ValueError(error_msg)


def detect_dataclass_mode(
    test_fn: Callable[..., Any],
    argnames: Sequence[str],
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,
) -> tuple[bool, type | None]:
    """
    Detect whether a test function expects a single dataclass parameter.

    Backward-compatible wrapper around ``_records.detect_record_param``, reading
    only the test's own parameters. It never raises: two matching parameters, or a
    record type that is not a dataclass, give ``(False, None)``.

    Returns:
        ``(True, dataclass_type)`` in dataclass mode; ``(False, None)`` otherwise.
    """
    # _records imports this module
    from ._records import detect_record_param

    try:
        record = detect_record_param(test_fn, argnames, None, pytest_fixtures=pytest_fixtures)
    except ValueError:
        return False, None
    if record is None:
        return False, None
    return True, record.record_type
