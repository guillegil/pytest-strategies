"""
Unit tests for _introspection module.

Tests validate_signature and detect_dataclass_mode as pure functions,
independent of the Strategy class.
"""

from dataclasses import dataclass

import pytest

from pytest_strategy._introspection import (
    PYTEST_FIXTURES,
    detect_dataclass_mode,
    validate_signature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Single:
    value: str


# ---------------------------------------------------------------------------
# PYTEST_FIXTURES constant
# ---------------------------------------------------------------------------


class TestPytestFixturesConstant:
    def test_is_frozenset(self):
        assert isinstance(PYTEST_FIXTURES, frozenset)

    def test_contains_expected_names(self):
        expected = {"request", "tmp_path", "capsys", "monkeypatch", "caplog"}
        assert expected.issubset(PYTEST_FIXTURES)

    def test_does_not_contain_arbitrary_name(self):
        assert "my_custom_fixture" not in PYTEST_FIXTURES


# ---------------------------------------------------------------------------
# validate_signature
# ---------------------------------------------------------------------------


class TestValidateSignature:
    def test_exact_match_passes(self):
        def test_fn(x, y):
            pass

        validate_signature(test_fn, ["x", "y"], "my_strategy")  # no exception

    def test_extra_fixture_param_ignored(self):
        def test_fn(x, y, request):
            pass

        validate_signature(test_fn, ["x", "y"], "my_strategy")  # no exception

    def test_custom_fixture_ignored(self):
        """Parameters not in argnames are treated as custom fixtures and skipped."""

        def test_fn(x, y, my_db_fixture):
            pass

        validate_signature(test_fn, ["x", "y"], "my_strategy")  # no exception

    def test_missing_param_raises(self):
        def test_fn(x):
            pass

        with pytest.raises(ValueError, match="signature mismatch"):
            validate_signature(test_fn, ["x", "y"], "my_strategy")

    def test_error_message_includes_strategy_name(self):
        def test_fn(x):
            pass

        with pytest.raises(ValueError, match="my_strategy"):
            validate_signature(test_fn, ["x", "y"], "my_strategy")

    def test_error_message_includes_missing_params(self):
        def test_fn(x):
            pass

        with pytest.raises(ValueError, match="Missing parameters"):
            validate_signature(test_fn, ["x", "y"], "my_strategy")

    def test_custom_fixture_set_respected(self):
        """Custom pytest_fixtures set is honoured."""

        def test_fn(x, session):
            pass

        validate_signature(test_fn, ["x"], "s", pytest_fixtures={"session"})  # no exception

    def test_single_param_match(self):
        def test_fn(value):
            pass

        validate_signature(test_fn, ["value"], "s")  # no exception

    def test_order_does_not_matter(self):
        def test_fn(b, a):
            pass

        validate_signature(test_fn, ["a", "b"], "s")  # no exception


# ---------------------------------------------------------------------------
# detect_dataclass_mode
# ---------------------------------------------------------------------------


class TestDetectDataclassMode:
    def test_single_param_dataclass_multiple_argnames(self):
        def test_fn(p: Point):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["x", "y"])
        assert is_dc is True
        assert dc_type is Point

    def test_single_param_not_dataclass(self):
        def test_fn(p: int):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["p"])
        assert is_dc is False
        assert dc_type is None

    def test_multiple_params_not_dc_mode(self):
        def test_fn(x: int, y: int):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["x", "y"])
        assert is_dc is False

    def test_single_argname_single_param_not_dc_mode(self):
        """Only one argname — not dataclass mode even with a dataclass type hint."""

        def test_fn(p: Point):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["p"])
        assert is_dc is False

    def test_no_annotation_not_dc_mode(self):
        def test_fn(p):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["x", "y"])
        assert is_dc is False

    def test_fixture_params_excluded(self):
        """request fixture should be excluded from param count."""

        def test_fn(p: Point, request):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["x", "y"])
        assert is_dc is True
        assert dc_type is Point

    def test_custom_fixture_set_respected(self):
        def test_fn(p: Point, session):
            pass

        is_dc, dc_type = detect_dataclass_mode(test_fn, ["x", "y"], pytest_fixtures={"session"})
        assert is_dc is True

    def test_returns_correct_dataclass_type(self):
        def test_fn(s: Single):
            pass

        # Single dataclass but only one argname → not DC mode
        is_dc, dc_type = detect_dataclass_mode(test_fn, ["value", "extra"])
        assert is_dc is True
        assert dc_type is Single
