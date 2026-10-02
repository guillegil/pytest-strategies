"""
End-to-end tests for record mode, run through pytester: one case per layout of
the record-mode rule.

A test receives a strategy's row as one record when neither the test nor any
fixture it uses asks for one of the strategy's argument names, and exactly one
test parameter is annotated with a record type whose fields are exactly those
names. Dataclasses are built; NamedTuple, TypedDict and pydantic models are
recognized and fail with "not supported yet".

Distinct module and strategy names are used per case on purpose (see
test_session_isolation_integration.py for rationale): ``CASE`` in the sources
below is replaced by the case's name.
"""

import textwrap

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
    from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

    @register("CASE_xy")
    def xy():
        # x is never 0 in a random row, so no random row repeats the ID of "zeros"
        return Parameter(
            TestArg("x", rng_type=RNGInteger(1, 9)),
            TestArg("y", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zeros": {"x": 0, "y": 0}},
            nsamples=2,
        )

    @register("CASE_ab")
    def ab():
        return Parameter(
            TestArg("a", rng_type=RNGInteger(0, 9)),
            TestArg("b", rng_type=RNGInteger(0, 9)),
            nsamples=3,
        )

    @register("CASE_width")
    def width():
        return Parameter(TestArg("width", rng_type=RNGInteger(1, 9)), nsamples=2)

    @register("CASE_empty")
    def empty():
        return Parameter(
            TestArg("x", rng_type=Series([], skip_if_empty="no x in this config")),
            TestArg("y", rng_type=RNGInteger(0, 9)),
        )
    """

# The record type most cases use, and the imports they share
HEADER = """
    import dataclasses
    from dataclasses import dataclass, field

    import pytest
    from pytest_strategy import strategy

    @dataclass
    class Point:
        x: int
        y: int
    """

# Each case: the test module (after HEADER, unless it starts with
# "from __future__"), then the outcomes or the error lines expected. "CASE_xy"
# has 3 rows: the directed vector "zeros" (ID x=0,y=0) and 2 random rows.
CASES = {
    "future_annotations": (
        """
        from __future__ import annotations

        from dataclasses import dataclass

        from pytest_strategy import strategy

        @dataclass
        class Point:
            x: int
            y: int

        @strategy("CASE_xy")
        def test_p(p: Point):
            assert isinstance(p, Point)
        """,
        {"passed": 3},
    ),
    "local_class": (
        """
        from __future__ import annotations

        from dataclasses import dataclass

        from pytest_strategy import strategy

        class TestLocal:
            @dataclass
            class Local:
                x: int
                y: int

            @strategy("CASE_xy")
            def test_p(self, p: Local):
                pass
        """,
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy'*",
            "*Missing parameters: [[]'x', 'y'[]]",
            "*Parameter 'p' is annotated with 'Local', which cannot be resolved in the test "
            "module's globals, so it is not a record type: define the class at module level.",
        ],
    ),
    "annotated": (
        """
        from typing import Annotated

        @strategy("CASE_xy")
        def test_p(p: Annotated[Point, "meta"]):
            assert isinstance(p, Point)
        """,
        {"passed": 3},
    ),
    "optional": (
        """
        from typing import Optional

        @strategy("CASE_xy")
        def test_p(p: Optional[Point]):
            pass
        """,
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy'*",
            "*Parameter 'p' is annotated with Optional[[]*Point[]], a union, which is not a "
            "record type.",
        ],
    ),
    "generic": (
        """
        from typing import Generic, TypeVar

        T = TypeVar("T")

        @dataclass
        class Pair(Generic[T]):
            x: T
            y: T

        @strategy("CASE_xy")
        def test_p(p: Pair[int]):
            assert type(p) is Pair
        """,
        {"passed": 3},
    ),
    "kw_only": (
        """
        @dataclass(kw_only=True)
        class KwPoint:
            y: int
            x: int

        @strategy("CASE_xy")
        def test_p(p: KwPoint):
            assert isinstance(p, KwPoint)
        """,
        {"passed": 3},
    ),
    "init_false_without_default": (
        """
        @dataclass
        class Sum:
            x: int
            y: int
            total: int = field(init=False)

        @strategy("CASE_xy")
        def test_p(p: Sum):
            assert not hasattr(p, "total")
        """,
        # The ID lists the init=True fields only: 3.0 crashed reading p.total
        {"passed": 3, "lines": ["*::test_p[[]x=0,y=0[]] PASSED*"]},
    ),
    "extra_defaulted_field": (
        """
        @dataclass
        class Point3:
            x: int
            y: int
            z: int = 0

        @strategy("CASE_xy")
        def test_p(p: Point3):
            pass
        """,
        [
            "*In test_p: Error converting samples to dataclass for strategy 'CASE_xy': "
            "Dataclass fields don't match strategy parameters!",
            "*Extra in dataclass: [[]'z'[]]",
        ],
    ),
    "one_argument": (
        """
        @dataclass
        class Width:
            width: int

        @strategy("CASE_width")
        def test_w(w: Width):
            assert isinstance(w, Width) and 1 <= w.width <= 9
        """,
        {"passed": 2},
    ),
    "self_and_cls": (
        """
        class TestPoints:
            @strategy("CASE_xy")
            def test_self(self, p: Point):
                assert isinstance(p, Point)

            @classmethod
            @strategy("CASE_xy")
            def test_cls(cls, p: Point):
                assert isinstance(p, Point)
        """,
        {"passed": 6},
    ),
    "fixture_next_to_record": (
        """
        @pytest.fixture
        def db():
            return "db"

        @strategy("CASE_xy")
        def test_on(p: Point, db):
            assert isinstance(p, Point) and db == "db"

        @strategy("CASE_xy", validate_signature=False)
        def test_off(db, p: Point):
            assert isinstance(p, Point) and db == "db"
        """,
        {"passed": 6},
    ),
    "fixture_consumes_arguments": (
        """
        RECEIVED = []

        @pytest.fixture
        def point(x, y):
            RECEIVED.append((x, y))
            return Point(x, y)

        @strategy("CASE_xy")
        def test_on(point: Point):
            assert RECEIVED[-1] == (point.x, point.y)

        @strategy("CASE_xy", validate_signature=False)
        def test_off(point: Point):
            assert RECEIVED[-1] == (point.x, point.y)
        """,
        {"passed": 6},
    ),
    "fixture_asks_for_an_argument": (
        """
        @pytest.fixture
        def helper(x):
            return x

        @strategy("CASE_xy")
        def test_p(p: Point, helper):
            pass
        """,
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy'*",
            "*Fixtures of the test ask for: [[]'x'[]]",
            "*Missing parameters: [[]'y'[]]",
            "*A fixture of the test asks for 'x', so the strategy passes its arguments by "
            "name: parameter 'p' (Point) receives the row as a record only when no fixture "
            "asks for an argument.",
        ],
    ),
    "autouse_fixture_asks_for_an_argument": (
        """
        @pytest.fixture(autouse=True)
        def _trace(y):
            pass

        @strategy("CASE_xy")
        def test_p(p: Point):
            pass
        """,
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy'*",
            "*Missing parameters: [[]'x'[]]",
            "*A fixture of the test asks for 'y', so the strategy passes its arguments by "
            "name: parameter 'p' (Point)*",
        ],
    ),
    "fixtures_ask_for_every_argument": (
        """
        @pytest.fixture
        def both(x, y):
            return (x, y)

        @strategy("CASE_xy")
        @pytest.mark.usefixtures("both")
        def test_p(p: Point):
            pass
        """,
        # Named mode, and the fixtures take every argument, but nothing gives p a value
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy': parameter 'p' "
            "(Point) gets no value: it is not one of the strategy's arguments, and no "
            "fixture or parametrization provides it.",
            "*A fixture of the test asks for 'x' and 'y', so the strategy passes its "
            "arguments by name: parameter 'p' (Point) receives the row as a record only "
            "when no fixture asks for an argument.",
        ],
    ),
    "fixtures_ask_for_every_argument_and_p_is_parametrized": (
        """
        @pytest.fixture
        def both(x, y):
            return (x, y)

        @strategy("CASE_xy")
        @pytest.mark.parametrize("p", [Point(1, 2)])
        def test_p(p: Point, both):
            assert p == Point(1, 2) and isinstance(both[0], int)
        """,
        {"passed": 3},
    ),
    "two_exact_dataclasses": (
        """
        @dataclass
        class Other:
            y: int
            x: int

        @strategy("CASE_xy")
        def test_p(p: Point, q: Other):
            pass
        """,
        [
            "*In test_p: Strategy 'CASE_xy': parameters 'p' (Point) and 'q' (Other) are each "
            "annotated with a record type whose fields are the strategy's arguments (x, y), "
            "so it is not clear which one receives the row.*",
        ],
    ),
    "dataclass_and_namedtuple": (
        """
        from typing import NamedTuple

        class PointNT(NamedTuple):
            x: int
            y: int

        @strategy("CASE_xy")
        def test_p(p: Point, q: PointNT):
            pass
        """,
        ["*In test_p: Strategy 'CASE_xy': parameters 'p' (Point) and 'q' (PointNT) are each*"],
    ),
    "typeddict_optional_keys": (
        """
        from typing import NotRequired, TypedDict

        class PointTD(TypedDict):
            x: int
            y: NotRequired[int]

        @strategy("CASE_xy")
        def test_p(p: Point, d: PointTD):
            pass
        """,
        # The optional key y counts, so both parameters match
        ["*In test_p: Strategy 'CASE_xy': parameters 'p' (Point) and 'd' (PointTD) are each*"],
    ),
    "pydantic_field_names": (
        """
        from pydantic import BaseModel, Field

        class PointModel(BaseModel):
            x: int = Field(alias="x_coord")
            y: int

        @strategy("CASE_xy")
        def test_p(p: Point, m: PointModel):
            pass
        """,
        # The field names count, not the aliases, so both parameters match
        ["*In test_p: Strategy 'CASE_xy': parameters 'p' (Point) and 'm' (PointModel) are*"],
    ),
    "pydantic_aliases": (
        """
        from pydantic import BaseModel, Field

        class AliasModel(BaseModel):
            a: int = Field(alias="x")
            b: int = Field(alias="y")

        @pytest.fixture
        def m():
            return AliasModel(x=1, y=2)

        @strategy("CASE_xy")
        def test_p(p: Point, m: AliasModel):
            assert isinstance(p, Point) and (m.a, m.b) == (1, 2)
        """,
        {"passed": 3},
    ),
    "default_record_parameter": (
        """
        @strategy("CASE_xy")
        def test_p(p: Point = None):
            pass
        """,
        [
            "*In test_p: Signature validation failed for strategy 'CASE_xy'*",
            "*Parameter 'p' has a default, so pytest does not fill it and it does not receive "
            "the row as a record.",
        ],
    ),
    "default_next_to_record": (
        """
        @strategy("CASE_xy")
        def test_p(p: Point, scale: int = 2):
            assert isinstance(p, Point) and scale == 2
        """,
        {"passed": 3},
    ),
    "namedtuple_not_supported": (
        """
        from typing import NamedTuple

        class BusTxn(NamedTuple):
            y: int
            x: int

        @strategy("CASE_xy")
        def test_p(txn: BusTxn):
            pass
        """,
        [
            "*In test_p: Strategy 'CASE_xy': parameter 'txn' is annotated with BusTxn, a "
            "NamedTuple; record mode supports dataclasses (NamedTuple, TypedDict and pydantic "
            "models are not supported yet). Take the arguments as parameters or use a "
            "dataclass.",
        ],
    ),
    "namedtuple_with_other_fields": (
        """
        from typing import NamedTuple

        class AB(NamedTuple):
            a: int
            x: int

        @strategy("CASE_xy")
        def test_p(txn: AB):
            pass
        """,
        # The only record parameter: its fields are listed, as for a dataclass
        [
            "*In test_p: Strategy 'CASE_xy': parameter 'txn' is annotated with AB, a "
            "NamedTuple, whose fields do not match the strategy's arguments (x, y): missing "
            "'y'; extra 'a'. Record mode supports dataclasses (NamedTuple, TypedDict and "
            "pydantic models are not supported yet). Take the arguments as parameters or use "
            "a dataclass with those fields.",
        ],
    ),
    "typeddict_not_supported": (
        """
        from typing import TypedDict

        class BusTxn(TypedDict):
            x: int
            y: int

        @strategy("CASE_xy")
        def test_p(txn: BusTxn):
            pass
        """,
        ["*parameter 'txn' is annotated with BusTxn, a TypedDict; record mode supports*"],
    ),
    "pydantic_model_not_supported": (
        """
        from pydantic import BaseModel

        class BusTxn(BaseModel):
            x: int
            y: int

        @strategy("CASE_xy")
        def test_p(txn: BusTxn):
            pass
        """,
        ["*parameter 'txn' is annotated with BusTxn, a pydantic model; record mode supports*"],
    ),
    "pydantic_dataclass": (
        """
        import pydantic

        @pydantic.dataclasses.dataclass
        class Checked:
            x: int
            y: int

        @strategy("CASE_xy")
        def test_p(p: Checked):
            assert isinstance(p, Checked)
        """,
        {"passed": 3},
    ),
    "pydantic_v1_style_class": (
        """
        import warnings

        with warnings.catch_warnings():
            # pydantic.v1 warns on Python 3.14
            warnings.simplefilter("ignore")
            from pydantic import v1

        class V1Point(v1.BaseModel):
            x: int
            y: int

        @pytest.fixture
        def legacy():
            return V1Point(x=1, y=2)

        @strategy("CASE_xy")
        def test_p(p: Point, legacy: V1Point):
            assert isinstance(p, Point) and legacy.x == 1
        """,
        {"passed": 3},
    ),
    "attrs_class": (
        """
        import attrs

        @attrs.define
        class APoint:
            x: int
            y: int

        @pytest.fixture
        def other():
            return APoint(1, 2)

        @strategy("CASE_xy")
        def test_p(p: Point, other: APoint):
            assert isinstance(p, Point) and other.x == 1
        """,
        {"passed": 3},
    ),
    "stacked_strategies": (
        """
        @dataclass
        class AB:
            a: int
            b: int

        @strategy("CASE_xy")
        @strategy("CASE_ab")
        def test_record_and_named(p: Point, a, b):
            assert isinstance(p, Point) and isinstance(a, int) and isinstance(b, int)

        @strategy("CASE_xy")
        @strategy("CASE_ab")
        def test_two_records(p: Point, q: AB):
            assert isinstance(p, Point) and isinstance(q, AB)
        """,
        {"passed": 9 + 9},
    ),
    "skip_if_empty": (
        """
        @strategy("CASE_empty")
        def test_p(p: Point):
            raise AssertionError("must be skipped")
        """,
        {"skipped": 1, "lines": ["SKIPPED [[]1[]] *: no x in this config"]},
    ),
}

# The cases whose test modules import these
NEEDS = {
    "pydantic": [
        "pydantic_field_names",
        "pydantic_aliases",
        "pydantic_model_not_supported",
        "pydantic_dataclass",
        "pydantic_v1_style_class",
    ],
    "attrs": ["attrs_class"],
}


@pytest.mark.parametrize("case", list(CASES))
def test_record_mode_rule(pytester, case):
    for module, cases in NEEDS.items():
        if case in cases:
            pytest.importorskip(module)
    source, expected = CASES[case]
    source = textwrap.dedent(source)
    if not source.lstrip().startswith("from __future__"):
        source = textwrap.dedent(HEADER) + source
    pytester.makepyfile(
        **{
            f"rec_{case}_strategies": textwrap.dedent(STRATEGIES).replace("CASE", case),
            f"test_rec_{case}": source.replace("CASE", case),
        }
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "-v", "-rs")

    if isinstance(expected, dict):
        outcomes = {key: value for key, value in expected.items() if key != "lines"}
        result.assert_outcomes(**outcomes)
        result.stdout.fnmatch_lines(expected.get("lines", []))
    else:
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines([line.replace("CASE", case) for line in expected])
