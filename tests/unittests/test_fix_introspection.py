"""
Regression tests for fixes in the introspection, dataclass and ID helpers.

Each test here failed before its fix: dataclass mode detection with ``self``,
custom fixtures and string annotations; conversion of keyword-only and
``init=False`` dataclasses.
"""

from dataclasses import KW_ONLY, dataclass, field

import pytest

from pytest_strategy._dataclass import convert_to_dataclass
from pytest_strategy._introspection import detect_dataclass_mode, detect_dataclass_param

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Config:
    host: str
    port: int


@dataclass
class Computed:
    x: int
    y: int
    total: int = field(init=False, default=0)

    def __post_init__(self):
        self.total = self.x + self.y


@dataclass(kw_only=True)
class KwOnlyPoint:
    x: int
    y: int


@dataclass
class KwOnlyField:
    x: int
    y: int = field(kw_only=True)


@dataclass
class KwOnlySentinel:
    x: int
    _: KW_ONLY
    y: int


# ---------------------------------------------------------------------------
# detect_dataclass_param / detect_dataclass_mode
# ---------------------------------------------------------------------------


class TestDetectDataclassParam:
    def test_method_self_is_ignored(self):
        class TestPoints:
            def test_point(self, p: Point):
                pass

        assert detect_dataclass_param(TestPoints.test_point, ["x", "y"]) == (True, Point, "p")

    def test_classmethod_style_cls_is_ignored(self):
        def test_point(cls, p: Point):
            pass

        assert detect_dataclass_param(test_point, ["x", "y"]) == (True, Point, "p")

    def test_custom_fixture_after_dataclass_param(self):
        def test_fn(p: Point, db):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_custom_fixture_before_dataclass_param(self):
        """The dataclass parameter is chosen by annotation, not by position."""

        def test_fn(db, p: Point):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_dataclass_typed_fixture_is_left_alone(self):
        """Only the dataclass whose init fields equal the argnames is parametrized."""

        def test_fn(cfg: Config, p: Point):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_string_annotation_is_resolved(self):
        def test_fn(p: "Point"):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_unresolvable_fixture_annotation_does_not_disable_detection(self):
        """get_type_hints fails on the fixture; the dataclass is still resolved."""

        def test_fn(p: "Point", db: "NotImportedAtRuntime"):  # noqa: F821
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_init_false_field_excluded_from_matching(self):
        def test_fn(cfg: Config, c: Computed):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Computed, "c")

    def test_argname_in_signature_is_named_mode(self):
        def test_fn(x, p: Point):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (False, None, None)

    def test_two_matching_dataclass_params_is_ambiguous(self):
        def test_fn(p: Point, q: Point):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (False, None, None)

    def test_detect_dataclass_mode_keeps_two_tuple(self):
        def test_fn(db, p: Point):
            pass

        assert detect_dataclass_mode(test_fn, ["x", "y"]) == (True, Point)


# ---------------------------------------------------------------------------
# convert_to_dataclass
# ---------------------------------------------------------------------------


class TestConvertToDataclassKeywords:
    @pytest.mark.parametrize("dc_type", [KwOnlyPoint, KwOnlyField, KwOnlySentinel])
    def test_keyword_only_fields(self, dc_type):
        result = convert_to_dataclass([(1, 2), (3, 4)], ["x", "y"], dc_type)
        assert result == [dc_type(x=1, y=2), dc_type(x=3, y=4)]

    def test_keyword_only_with_reordered_argnames(self):
        result = convert_to_dataclass([(2, 1)], ["y", "x"], KwOnlyPoint)
        assert result == [KwOnlyPoint(x=1, y=2)]

    def test_init_false_field_not_required(self):
        result = convert_to_dataclass([(1, 2)], ["x", "y"], Computed)
        assert result == [Computed(x=1, y=2)]
        assert result[0].total == 3

    def test_init_false_field_supplied_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="Missing in dataclass: \\['total'\\]"):
            convert_to_dataclass([(1, 2, 3)], ["x", "y", "total"], Computed)
