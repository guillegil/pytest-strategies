"""
Regression tests for fixes in the introspection, dataclass and ID helpers.

Each test here failed before its fix: dataclass mode detection with ``self``,
custom fixtures and string annotations.
"""

from dataclasses import dataclass, field

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
