"""
Signature introspection and dataclass detection utilities.

Pure functions — no pytest runtime dependency beyond inspect/dataclasses.
"""

import inspect
from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import cast

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


def validate_signature(
    test_fn,
    argnames: Sequence[str],
    strategy_name: str,
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,
) -> None:
    """
    Validate that a test function's signature matches the strategy argnames.

    Parameters excluded from validation:
    - Parameters listed in *pytest_fixtures* (built-in fixtures)
    - Parameters whose name does not appear in *argnames* (assumed custom fixtures)

    Args:
        test_fn: The test function to inspect.
        argnames: Expected argument names from the strategy.
        strategy_name: Name of the strategy (used in error messages).
        pytest_fixtures: Set of fixture names to ignore.

    Raises:
        ValueError: When the test-function signature does not match *argnames*.
    """
    sig = inspect.signature(test_fn)
    test_params = list(sig.parameters.keys())

    actual_params = []
    for p in test_params:
        if p in pytest_fixtures:
            continue
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


def detect_dataclass_mode(
    test_fn,
    argnames: Sequence[str],
    pytest_fixtures: "frozenset[str] | set[str]" = PYTEST_FIXTURES,
) -> tuple[bool, type | None]:
    """
    Detect whether a test function expects a single dataclass parameter.

    Returns:
        ``(True, dataclass_type)`` when the function has a single non-fixture
        parameter with a dataclass type annotation that covers multiple
        *argnames*; ``(False, None)`` otherwise.
    """
    sig = inspect.signature(test_fn)
    test_params = list(sig.parameters.keys())

    actual_params = [p for p in test_params if p not in pytest_fixtures]

    if len(actual_params) == 1 and len(argnames) > 1:
        param = sig.parameters[actual_params[0]]
        if param.annotation != inspect.Parameter.empty and is_dataclass(param.annotation):
            return True, cast(type, param.annotation)

    return False, None
