"""
Unit tests for record mode: which annotations are record types, their field
sets, which test parameter receives the row, and the errors and hints when none
or several do.

A test receives a strategy's row as one record when neither the test nor any
fixture it uses asks for one of the strategy's argument names, and exactly one
test parameter is annotated with a record type whose fields are exactly those
names.
"""

import collections
import os
import subprocess
import sys
import typing
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Annotated, Generic, NamedTuple, NotRequired, Optional, TypedDict, TypeVar
from unittest.mock import MagicMock

import pytest

import pytest_strategy
from pytest_strategy import Parameter, TestArg, Vector
from pytest_strategy import _resolver as resolver_module
from pytest_strategy._ids import generate_dataclass_ids
from pytest_strategy._introspection import detect_dataclass_mode, validate_signature
from pytest_strategy._records import (
    RecordParam,
    detect_record_param,
    record_class,
    record_fields,
    record_hints,
    record_kind,
)
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._vector import vector_type

T = TypeVar("T")


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Other:
    y: int
    x: int


@dataclass
class Point3:
    x: int
    y: int
    z: int = 0


@dataclass
class Pair(Generic[T]):
    x: T
    y: T


@dataclass
class Sum:
    x: int
    y: int
    total: int = field(init=False)


@dataclass
class WithInitVar:
    x: int
    scale: InitVar[int] = 1
    y: int = 0


@dataclass
class Width:
    width: int


class PointNT(NamedTuple):
    x: int
    y: int


PointTuple = collections.namedtuple("PointTuple", ["x", "y"])


class PointTD(TypedDict):
    x: int
    y: NotRequired[int]


class PartialTD(TypedDict, total=False):
    x: int
    y: int


class Plain:
    x: int
    y: int


XY = ["x", "y"]


def _record(test_fn, argnames=XY, fixturenames=None):
    """The (parameter, record type) of record mode, or None in named mode."""
    record = detect_record_param(test_fn, argnames, fixturenames)
    return None if record is None else (record.name, record.record_type)


# ---------------------------------------------------------------------------
# Record types and their fields
# ---------------------------------------------------------------------------


class TestRecordClass:
    @pytest.mark.parametrize(
        "annotation, expected",
        [
            (Point, Point),
            (Annotated[Point, "meta"], Point),
            (Annotated[Annotated[Point, "a"], "b"], Point),
            (Pair[int], Pair),
            (Annotated[Pair[int], "meta"], Pair),
            (list[int], list),
            (Optional[Point], None),  # noqa: UP045
            (Point | None, None),
            (typing.Union[Point, Other], None),  # noqa: UP007
            ("Point", None),
            (None, None),
        ],
    )
    def test_annotations(self, annotation, expected):
        assert record_class(annotation) is expected


class TestRecordKind:
    @pytest.mark.parametrize(
        "cls, kind",
        [
            (Point, "dataclass"),
            (Pair, "dataclass"),
            (PointNT, "namedtuple"),
            (PointTuple, "namedtuple"),
            (PointTD, "typeddict"),
            (PartialTD, "typeddict"),
            (Plain, None),
            (int, None),
            (tuple, None),
            (dict, None),
        ],
    )
    def test_kinds(self, cls, kind):
        assert record_kind(cls) == kind

    def test_vector_is_not_a_record_type(self):
        assert record_kind(Vector) is None
        assert record_kind(vector_type(("x", "y"))) is None

    def test_typing_extensions_typeddict(self):
        typing_extensions = pytest.importorskip("typing_extensions")

        class TD(typing_extensions.TypedDict):
            x: int
            y: int

        assert record_kind(TD) == "typeddict"
        assert record_fields(TD, "typeddict") == {"x", "y"}

    def test_pydantic_model_and_dataclass(self):
        pydantic = pytest.importorskip("pydantic")

        class Model(pydantic.BaseModel):
            x: int
            y: int

        @pydantic.dataclasses.dataclass
        class Checked:
            x: int
            y: int

        assert record_kind(Model) == "pydantic"
        assert record_kind(Checked) == "dataclass"

    def test_pydantic_v1_model_is_not_a_record_type(self):
        pytest.importorskip("pydantic")
        import warnings

        with warnings.catch_warnings():
            # pydantic.v1 warns on Python 3.14
            warnings.simplefilter("ignore")
            from pydantic import v1

        class V1Model(v1.BaseModel):
            x: int
            y: int

        assert record_kind(V1Model) is None

    def test_attrs_class_is_not_a_record_type(self):
        attrs = pytest.importorskip("attrs")

        @attrs.define
        class APoint:
            x: int
            y: int

        assert record_kind(APoint) is None

    def test_pydantic_is_found_through_sys_modules(self, monkeypatch):
        """A model class is not recognized once pydantic is not imported."""
        pydantic = pytest.importorskip("pydantic")

        class Model(pydantic.BaseModel):
            x: int

        monkeypatch.delitem(sys.modules, "pydantic")
        assert record_kind(Model) is None

    def test_detection_never_imports_pydantic(self):
        pytest.importorskip("pydantic")
        code = (
            "import sys\n"
            "from dataclasses import dataclass\n"
            "from pytest_strategy._records import detect_record_param\n"
            "@dataclass\n"
            "class P:\n"
            "    x: int\n"
            "class Q:\n"
            "    x: int\n"
            "def test(p: P, q: Q): pass\n"
            "assert detect_record_param(test, ['x'], None).name == 'p'\n"
            "assert 'pydantic' not in sys.modules\n"
        )
        package_root = str(Path(pytest_strategy.__file__).resolve().parents[1])
        python_path = os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))
        subprocess.run(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": python_path}, check=True
        )


class TestRecordFields:
    def test_dataclass_init_fields(self):
        assert record_fields(Point, "dataclass") == {"x", "y"}
        assert record_fields(Sum, "dataclass") == {"x", "y"}
        assert record_fields(Point3, "dataclass") == {"x", "y", "z"}

    def test_dataclass_init_var_is_not_a_field(self):
        assert record_fields(WithInitVar, "dataclass") == {"x", "y"}

    def test_namedtuple_fields(self):
        assert record_fields(PointNT, "namedtuple") == {"x", "y"}
        assert record_fields(PointTuple, "namedtuple") == {"x", "y"}

    def test_typeddict_required_and_optional_keys(self):
        assert record_fields(PointTD, "typeddict") == {"x", "y"}
        assert record_fields(PartialTD, "typeddict") == {"x", "y"}

    def test_pydantic_field_names_not_aliases_nor_computed_fields(self):
        pydantic = pytest.importorskip("pydantic")

        class Model(pydantic.BaseModel):
            addr: int = pydantic.Field(alias="address")
            len: int

            @pydantic.computed_field
            @property
            def end(self) -> int:
                return self.addr + self.len

        assert record_fields(Model, "pydantic") == {"addr", "len"}


# ---------------------------------------------------------------------------
# Which parameter receives the row
# ---------------------------------------------------------------------------


class TestDetectRecordParam:
    def test_exact_dataclass(self):
        def test_fn(p: Point):
            pass

        record = detect_record_param(test_fn, XY, ["p"])
        assert record == RecordParam("p", Point, "dataclass", frozenset(XY))

    def test_one_argument_strategy(self):
        """3.0 required two arguments."""

        def test_fn(w: Width):
            pass

        assert _record(test_fn, ["width"], ["w"]) == ("w", Width)

    def test_generic_dataclass(self):
        def test_fn(p: Pair[int]):
            pass

        assert _record(test_fn) == ("p", Pair)

    def test_annotated(self):
        def test_fn(p: Annotated[Point, "meta"]):
            pass

        assert _record(test_fn) == ("p", Point)

    def test_string_annotations(self):
        def test_fn(p: "Annotated[Pair[int], 'meta']"):
            pass

        assert _record(test_fn) == ("p", Pair)

    def test_argname_asked_by_a_fixture_is_named_mode(self):
        def test_fn(p: Point, helper):
            pass

        assert _record(test_fn, XY, ["p", "helper", "x"]) is None

    def test_unrelated_fixture_names_keep_record_mode(self):
        def test_fn(p: Point, helper):
            pass

        assert _record(test_fn, XY, ["p", "helper", "db", "request"]) == ("p", Point)

    def test_without_fixturenames_the_parameters_are_read(self):
        def test_fn(x, p: Point):
            pass

        assert _record(test_fn) is None

    @pytest.mark.parametrize(
        "source",
        [
            "def test_fn(p: Point = None): pass",
            "def test_fn(p: Point, /): pass",
            "def test_fn(*p: Point): pass",
            "def test_fn(**p: Point): pass",
            "def test_fn(self: Point): pass",
            "def test_fn(cls: Point): pass",
            "def test_fn(request: Point): pass",
        ],
    )
    def test_parameters_pytest_does_not_fill_are_not_records(self, source):
        namespace = {"Point": Point}
        exec(source, namespace)

        assert _record(namespace["test_fn"]) is None

    def test_keyword_only_parameter(self):
        def test_fn(db, *, p: Point):
            pass

        assert _record(test_fn) == ("p", Point)

    def test_lone_mismatched_dataclass_is_returned(self):
        """Building the records then lists the missing and extra fields."""

        def test_fn(p: Point3):
            pass

        assert _record(test_fn) == ("p", Point3)

    def test_mismatched_dataclasses_are_fixtures(self):
        def test_fn(p: Point3, s: Width):
            pass

        assert _record(test_fn) is None

    def test_exact_match_wins_over_mismatched(self):
        def test_fn(cfg: Point3, p: Point):
            pass

        assert _record(test_fn) == ("p", Point)

    @pytest.mark.parametrize("annotation", [Plain, int, Optional[Point], Vector])  # noqa: UP045
    def test_other_annotations_are_not_records(self, annotation):
        def test_fn(p):
            pass

        test_fn.__annotations__ = {"p": annotation}
        assert _record(test_fn) is None

    def test_unresolvable_annotation_is_not_a_record(self):
        def test_fn(p: "Undefined"):  # noqa: F821
            pass

        assert _record(test_fn) is None


class TestAmbiguity:
    def test_two_exact_dataclasses(self):
        def test_fn(p: Point, q: Other):
            pass

        with pytest.raises(ValueError) as excinfo:
            detect_record_param(test_fn, XY, None)
        assert str(excinfo.value) == (
            "parameters 'p' (Point) and 'q' (Other) are each annotated with a record type "
            "whose fields are the strategy's arguments (x, y), so it is not clear which one "
            "receives the row. Annotate only one of them with a record type, or take the "
            "arguments as parameters."
        )

    def test_three_exact_matches_are_all_named(self):
        def test_fn(p: Point, q: PointNT, d: PartialTD):
            pass

        with pytest.raises(
            ValueError, match=r"^parameters 'p' \(Point\), 'q' \(PointNT\) and 'd' \(PartialTD\) "
        ):
            detect_record_param(test_fn, XY, None)

    def test_dataclass_and_typeddict_with_an_optional_key(self):
        def test_fn(p: Point, d: PointTD):
            pass

        with pytest.raises(ValueError, match="'p' \\(Point\\) and 'd' \\(PointTD\\)"):
            detect_record_param(test_fn, XY, None)

    def test_dataclass_and_pydantic_model_by_field_names(self):
        pydantic = pytest.importorskip("pydantic")

        class Model(pydantic.BaseModel):
            x: int = pydantic.Field(alias="xa")
            y: int

        class Aliased(pydantic.BaseModel):
            a: int = pydantic.Field(alias="x")
            b: int = pydantic.Field(alias="y")

        def ambiguous(p: Point, m: Model):
            pass

        def by_names(p: Point, m: Aliased):
            pass

        with pytest.raises(ValueError, match="'p' \\(Point\\) and 'm' \\(Model\\)"):
            detect_record_param(ambiguous, XY, None)
        assert _record(by_names) == ("p", Point)


class TestNotSupportedYet:
    @pytest.mark.parametrize(
        "cls, kind",
        [(PointNT, "a NamedTuple"), (PointTuple, "a NamedTuple"), (PointTD, "a TypedDict")],
    )
    def test_namedtuple_and_typeddict(self, cls, kind):
        def test_fn(txn):
            pass

        test_fn.__annotations__ = {"txn": cls}
        with pytest.raises(ValueError) as excinfo:
            detect_record_param(test_fn, XY, None)
        assert str(excinfo.value) == (
            f"parameter 'txn' is annotated with {cls.__name__}, {kind}; record mode supports "
            "dataclasses (NamedTuple, TypedDict and pydantic models are not supported yet). "
            "Take the arguments as parameters or use a dataclass."
        )

    def test_pydantic_model(self):
        pydantic = pytest.importorskip("pydantic")

        class BusTxn(pydantic.BaseModel):
            x: int
            y: int

        def test_fn(txn: BusTxn):
            pass

        with pytest.raises(ValueError, match="^parameter 'txn' is annotated with BusTxn, a pyd"):
            detect_record_param(test_fn, XY, None)

    def test_lone_mismatched_namedtuple(self):
        """A NamedTuple whose fields do not match cannot be used either."""

        def test_fn(txn: PointNT):
            pass

        with pytest.raises(ValueError, match="a NamedTuple; record mode supports dataclasses"):
            detect_record_param(test_fn, ["a", "b"], None)

    def test_named_mode_is_left_alone(self):
        """No error when a fixture asks for the arguments: the strategy passes them by name."""

        def test_fn(txn: PointNT, x, y):
            pass

        assert _record(test_fn) is None


class TestDetectDataclassMode:
    """The 3.0 wrapper reads the test's parameters and never raises."""

    def test_one_argument_record(self):
        def test_fn(w: Width):
            pass

        assert detect_dataclass_mode(test_fn, ["width"]) == (True, Width)

    def test_two_matches_are_not_dataclass_mode(self):
        def test_fn(p: Point, q: Other):
            pass

        assert detect_dataclass_mode(test_fn, XY) == (False, None)

    def test_namedtuple_is_not_dataclass_mode(self):
        def test_fn(txn: PointNT):
            pass

        assert detect_dataclass_mode(test_fn, XY) == (False, None)


# ---------------------------------------------------------------------------
# Signature errors in named mode, and their hints
# ---------------------------------------------------------------------------


class TestValidateSignatureFixtures:
    def test_an_argument_a_fixture_asks_for_is_taken(self):
        def test_fn(x, helper):
            pass

        validate_signature(test_fn, XY, "s", fixturenames=["x", "helper", "y"])

    def test_without_fixturenames_only_parameters_count(self):
        def test_fn(x, helper):
            pass

        with pytest.raises(ValueError, match=r"Missing parameters: \['y'\]"):
            validate_signature(test_fn, XY, "s")

    def test_message_lists_what_fixtures_ask_for(self):
        def test_fn(p: Point, helper):
            pass

        with pytest.raises(ValueError) as excinfo:
            validate_signature(test_fn, ["x", "y", "z"], "s", fixturenames=["p", "helper", "y"])
        assert str(excinfo.value) == (
            "Test function signature mismatch for strategy 's'!\n"
            "  Strategy provides: ['x', 'y', 'z']\n"
            "  Test function expects: []\n"
            "  Fixtures of the test ask for: ['y']\n"
            "  Missing parameters: ['x', 'z']\n"
        )

    def test_missing_parameters_keep_the_argument_order(self):
        def test_fn():
            pass

        with pytest.raises(ValueError, match=r"Missing parameters: \['z', 'a', 'm'\]"):
            validate_signature(test_fn, ["z", "a", "m"], "s")


class TestRecordHints:
    def test_fixture_asks_for_an_argument_of_a_record(self):
        def test_fn(p: Point, helper):
            pass

        assert record_hints(test_fn, XY, ["p", "helper", "x"]) == [
            "A fixture of the test asks for 'x', so the strategy passes its arguments by name: "
            "parameter 'p' (Point) receives the row as a record only when no fixture asks for "
            "an argument."
        ]

    def test_several_arguments_are_listed(self):
        def test_fn(p: Point):
            pass

        hints = record_hints(test_fn, XY, ["p", "x", "y"])
        assert hints[0].startswith("A fixture of the test asks for 'x' and 'y', so")

    def test_no_fixture_hint_without_a_matching_record(self):
        def test_fn(p: Point3, helper):
            pass

        assert record_hints(test_fn, XY, ["p", "helper", "x"]) == []

    def test_unresolvable_annotation(self):
        def test_fn(p: "Local"):  # noqa: F821
            pass

        assert record_hints(test_fn, XY, ["p"]) == [
            "Parameter 'p' is annotated with 'Local', which cannot be resolved in the test "
            "module's globals, so it is not a record type: define the class at module level."
        ]

    def test_forward_reference(self):
        """Python 3.14 reads a name it cannot resolve as a ForwardRef."""

        def test_fn(p):
            pass

        test_fn.__annotations__ = {"p": typing.ForwardRef("Later")}
        assert _record(test_fn) is None
        assert record_hints(test_fn, XY, ["p"]) == [
            "Parameter 'p' is annotated with 'Later', which cannot be resolved in the test "
            "module's globals, so it is not a record type: define the class at module level."
        ]

    def test_union_of_a_record_type(self):
        def test_fn(p: "Optional[Point]", q: "int | None"):  # noqa: UP045
            pass

        assert record_hints(test_fn, XY, ["p", "q"]) == [
            "Parameter 'p' is annotated with Optional[Point], a union, which is not a record "
            "type."
        ]

    def test_record_parameter_with_a_default(self):
        def test_fn(p: Point = None, scale: int = 2):
            pass

        assert record_hints(test_fn, XY, []) == [
            "Parameter 'p' has a default, so pytest does not fill it and it does not receive "
            "the row as a record."
        ]

    def test_no_hints(self):
        def test_fn(p: Width, db):
            pass

        assert record_hints(test_fn, XY, ["p", "db"]) == []


# ---------------------------------------------------------------------------
# Through the resolver
# ---------------------------------------------------------------------------


def _xy(nsamples):
    return Parameter(TestArg("x", value=1), TestArg("y", value=2), nsamples=1)


def _build(test_fn, *, fixturenames=None, validate=True):
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: {"vector_mode": "all"}.get(
        opt, default
    )
    return build_parametrization(
        "s",
        _xy,
        test_fn,
        config=config,
        pytest_fixtures=set(),
        validate=validate,
        fixturenames=fixturenames,
    )


class TestResolver:
    @pytest.mark.parametrize("validate", [True, False])
    def test_validate_signature_does_not_change_the_choice(self, validate):
        def test_fn(p: Point, db):
            pass

        argnames, values, _ = _build(test_fn, fixturenames=["p", "db"], validate=validate)
        assert (argnames, values) == ("p", [Point(1, 2)])

    @pytest.mark.parametrize("validate", [True, False])
    def test_a_fixture_that_consumes_the_arguments_gets_them(self, validate):
        def test_fn(point: Point):
            pass

        argnames, values, _ = _build(test_fn, fixturenames=["point", "x", "y"], validate=validate)
        assert (argnames, values) == ("x,y", [(1, 2)])

    def test_errors_name_the_strategy(self):
        def test_fn(p: Point, q: Other):
            pass

        with pytest.raises(ValueError, match="^Strategy 's': parameters 'p' \\(Point\\) and"):
            _build(test_fn)

    def test_signature_error_carries_the_hints(self):
        def test_fn(p: Point, helper):
            pass

        with pytest.raises(ValueError) as excinfo:
            _build(test_fn, fixturenames=["p", "helper", "x"])
        message = str(excinfo.value)
        assert message.startswith("Signature validation failed for strategy 's': ")
        assert message.endswith(
            "  Missing parameters: ['y']\n"
            "  A fixture of the test asks for 'x', so the strategy passes its arguments by "
            "name: parameter 'p' (Point) receives the row as a record only when no fixture "
            "asks for an argument.\n"
        )

    def test_record_mode_asserts_no_argument_is_asked_for(self, monkeypatch):
        """The resolver checks what pytest requires of a parametrized name."""

        def test_fn(p: Point):
            pass

        record = RecordParam("p", Point, "dataclass", frozenset(XY))
        monkeypatch.setattr(resolver_module, "detect_record_param", lambda *a, **k: record)
        with pytest.raises(AssertionError):
            _build(test_fn, fixturenames=["p", "x"])

    def test_ids_list_the_init_fields(self):
        def test_fn(p: Sum):
            pass

        argnames, values, ids = _build(test_fn)
        assert argnames == "p" and ids == ["x=1,y=2"]
        assert not hasattr(values[0], "total")


class TestDataclassIds:
    def test_init_false_fields_are_left_out(self):
        """3.0 read every field, and failed on one that __init__ does not set."""
        assert generate_dataclass_ids([Sum(1, 2)], Sum) == ["x=1,y=2"]

        @dataclass
        class Computed:
            x: int
            total: int = field(init=False, default=0)

        assert generate_dataclass_ids([Computed(1)], Computed) == ["x=1"]

    def test_fields_keep_their_declaration_order(self):
        assert generate_dataclass_ids([Other(y=2, x=1)], Other) == ["y=2,x=1"]
