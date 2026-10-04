"""
Unit tests for the per-test metadata (D17): VectorInfo and its to_dict(), the value
encoding of schema 1 documents (_encode.py), the VectorInfo the resolver builds for
every kind of row, the pytest.param rows that carry it in a ``strategy`` mark, and
the -v summary's row counts by kind.
"""

import dataclasses
import enum
import json
import math
import os
from unittest.mock import DEFAULT, MagicMock

import pytest

import pytest_strategy
from pytest_strategy import (
    VECTOR_KEY,
    VECTORS_KEY,
    Parameter,
    RNGInteger,
    RNGSequence,
    Series,
    TestArg,
    Vector,
    VectorInfo,
)
from pytest_strategy._encode import encode
from pytest_strategy._resolver import (
    _count_rows,
    _origin,
    build_parametrization,
    resolve_and_parametrize,
)
from pytest_strategy._runtime import Resolution, runtime
from pytest_strategy._vector import vector_type
from pytest_strategy.plugin import _summary_lines


class Color(enum.Enum):
    RED = 1
    GREEN = 2


class Level(enum.IntEnum):
    LOW = 1


class Mode(enum.StrEnum):
    FAST = "fast"


class Perm(enum.Flag):
    R = 1
    W = 2


class Outer:
    class Shade(enum.Enum):
        DARK = 1


class Addr(int):
    """An int subclass that is not an Enum."""


class Plain:
    """An object with the default repr, which has its memory address."""


def _info(**fields):
    """Return a VectorInfo of a random row, with ``fields`` replaced."""
    values = {
        "strategy": "burst",
        "origin": "tests/strategies.py:12",
        "kind": "random",
        "name": None,
        "index": 3,
        "enumerated": (),
        "values": vector_type(("addr", "len"))(4096, 17),
        "id": "rand-3",
        "seed": 1,
        "context": None,
        "constraints_off": (),
    }
    values.update(fields)
    return VectorInfo(**values)


def _make_config(*, ids=None, **options):
    """
    Return a mock pytest.Config whose getoption() serves the given CLI options, and
    whose getini() serves ``ids`` as the strategies_ids ini option when it is given.
    """
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    if ids is not None:
        config.getini.side_effect = lambda name: ids if name == "strategies_ids" else DEFAULT
    return config


def _test_fn(*argnames):
    """Return a test function taking ``argnames``."""
    namespace = {}
    exec(f"def test_fn({', '.join(argnames)}):\n    pass", namespace)
    return namespace["test_fn"]


def _build(param, *argnames, name="strat", **options):
    """Resolve a factory returning ``param`` for a test taking ``argnames``."""
    return build_parametrization(
        name,
        lambda: param,
        _test_fn(*(argnames or param.arg_names)),
        config=_make_config(**options),
        pytest_fixtures=set(),
    )


def _infos(param, *argnames, **options):
    """Return the VectorInfo the resolver gives each row of ``param``."""
    return _build(param, *argnames, **options).infos


class TestEncode:
    """The schema 1 value encoding: JSON that keeps each value's type."""

    @pytest.mark.parametrize(
        "value",
        [None, True, False, 0, -7, 2**70, "", "fast", 1.5, -0.0, 1e300],
        ids=repr,
    )
    def test_json_types_are_written_as_json(self, value):
        assert encode(value) is value

    @pytest.mark.parametrize(
        ("value", "text"),
        [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_a_float_that_is_not_finite_is_tagged(self, value, text):
        assert encode(value) == {"$float": text}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Color.RED, {"$enum": "Color", "member": "RED"}),
            (Level.LOW, {"$enum": "Level", "member": "LOW"}),
            (Mode.FAST, {"$enum": "Mode", "member": "FAST"}),
            (Perm.R | Perm.W, {"$enum": "Perm", "member": "R|W"}),
            (Outer.Shade.DARK, {"$enum": "Outer.Shade", "member": "DARK"}),
        ],
        ids=["Enum", "IntEnum", "StrEnum", "Flag", "nested"],
    )
    def test_an_enum_member_is_its_class_and_name(self, value, expected):
        assert encode(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (b"\x00a", {"$repr": "b'\\x00a'", "$type": "bytes"}),
            ((1, "a"), {"$repr": "(1, 'a')", "$type": "tuple"}),
            ([1, 2], {"$repr": "[1, 2]", "$type": "list"}),
            ({"k": 1}, {"$repr": "{'k': 1}", "$type": "dict"}),
            (Addr(4), {"$repr": "4", "$type": "Addr"}),
            # A Flag value that is no member has no name
            (Perm(0), {"$repr": "<Perm: 0>", "$type": "Perm"}),
        ],
        ids=["bytes", "tuple", "list", "dict", "int-subclass", "Flag-without-name"],
    )
    def test_anything_else_is_its_repr_and_type(self, value, expected):
        assert encode(value) == expected

    def test_the_repr_is_stable(self):
        """Set elements are sorted, and a memory address is replaced by the type."""
        assert encode({"b", "c", "a"}) == {"$repr": "{'a', 'b', 'c'}", "$type": "set"}
        assert encode(Plain()) == {"$repr": "Plain", "$type": "Plain"}

    def test_every_encoding_is_strict_json(self):
        values = [None, 1, "a", 1.5, math.nan, -math.inf, Color.RED, b"x", {1}, Plain()]
        text = json.dumps([encode(v) for v in values], allow_nan=False)

        def no_constant(name):
            raise ValueError(name)

        assert len(json.loads(text, parse_constant=no_constant)) == len(values)


class TestVectorInfo:
    def test_exported(self):
        for name in ("VectorInfo", "VECTOR_KEY", "VECTORS_KEY"):
            assert name in pytest_strategy.__all__
        assert isinstance(VECTOR_KEY, pytest.StashKey)
        assert isinstance(VECTORS_KEY, pytest.StashKey)
        assert VECTOR_KEY is not VECTORS_KEY

    def test_frozen(self):
        info = _info()
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.kind = "directed"
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.values = (1, 2)

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            VectorInfo("burst")

    def test_the_streams_version_defaults_to_1(self):
        assert _info().streams == 1

    def test_to_dict(self):
        info = _info(enumerated=("ch",), constraints_off=("aligned",), context="3f2a9c1e")

        assert info.to_dict() == {
            "schema": 1,
            "strategy": "burst",
            "origin": "tests/strategies.py:12",
            "kind": "random",
            "name": None,
            "index": 3,
            "enumerated": ["ch"],
            "values": {"addr": 4096, "len": 17},
            "id": "rand-3",
            "seed": 1,
            "context": "3f2a9c1e",
            "constraints_off": ["aligned"],
            "streams": 1,
        }
        assert list(info.to_dict())[0] == "schema"

    def test_to_dict_encodes_the_values_by_argument_name(self):
        row = vector_type(("f", "c", "b", "t", "o"))(math.nan, Color.GREEN, b"\x01", (1,), Plain())
        info = _info(values=row)

        assert info.to_dict()["values"] == {
            "f": {"$float": "nan"},
            "c": {"$enum": "Color", "member": "GREEN"},
            "b": {"$repr": "b'\\x01'", "$type": "bytes"},
            "t": {"$repr": "(1,)", "$type": "tuple"},
            "o": {"$repr": "Plain", "$type": "Plain"},
        }
        text = json.dumps(info.to_dict(), allow_nan=False)
        assert json.loads(text)["schema"] == 1

    def test_a_skipped_row_has_none_values(self):
        info = _info(kind="skipped", index=None, id="skipped", values=vector_type(("x",))(None))
        assert json.loads(json.dumps(info.to_dict()))["values"] == {"x": None}


class TestThroughTheResolver:
    """build_parametrization gives every row a VectorInfo, in the rows' order."""

    def test_directed_test_and_random_rows(self):
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            TestArg("len", rng_type=RNGInteger(1, 16)),
            directed_vectors={"zeros": (0, 1), "max": {"addr": 63, "len": 16}},
            test_vectors={"mid": (32, 8)},
            nsamples=2,
        )

        parametrization = _build(param)
        infos = parametrization.infos
        assert [(i.kind, i.name, i.index, i.id) for i in infos] == [
            ("directed", "zeros", 0, "directed-zeros"),
            ("directed", "max", 1, "directed-max"),
            ("random", None, 0, "rand-0"),
            ("random", None, 1, "rand-1"),
        ]
        assert [i.values for i in infos] == parametrization.values
        assert all(type(i.values) is param.vector_type for i in infos)
        assert {i.enumerated for i in infos} == {()}

        [test_row] = _infos(param, vector_mode="test")
        assert (test_row.kind, test_row.name, test_row.index, test_row.id) == (
            "test",
            "mid",
            0,
            "test-mid",
        )
        assert test_row.values == (32, 8)

    def test_a_selected_directed_vector_keeps_its_index(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"low": (0,), "high": (9,)},
        )

        [info] = _infos(param, vector_name="high")
        assert (info.kind, info.name, info.index, info.id, info.values) == (
            "directed",
            "high",
            1,
            "directed-high",
            (9,),
        )

    def test_enumerated_arguments_are_in_declaration_order(self):
        """pos is sorted by name; enumerated keeps the order of the arguments."""
        param = Parameter(
            TestArg("zone", rng_type=Series(["a", "b"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            TestArg("ch", rng_type=Series([0, 1])),
            nsamples=4,
        )

        infos = _infos(param)
        assert {i.enumerated for i in infos} == {("zone", "ch")}
        assert [(i.index, i.id) for i in infos] == [
            (0, "zone=a-ch=0-rand-0"),
            (0, "zone=a-ch=1-rand-0"),
            (0, "zone=b-ch=0-rand-0"),
            (0, "zone=b-ch=1-rand-0"),
        ]

    def test_exhaustive_rows(self):
        param = Parameter(
            TestArg("dev", rng_type=RNGSequence(["a", "b", "c"])),
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
        )

        infos = _infos(param, nsamples="auto")
        assert {(i.kind, i.name, i.enumerated) for i in infos} == {
            ("exhaustive", None, ("dev", "ch"))
        }
        # The index is the position in the declaration-order product, whatever the
        # RNGSequence's order
        assert sorted((i.index, i.id) for i in infos) == [
            (0, "dev=a-ch=0"),
            (1, "dev=a-ch=1"),
            (2, "dev=b-ch=0"),
            (3, "dev=b-ch=1"),
            (4, "dev=c-ch=0"),
            (5, "dev=c-ch=1"),
        ]
        for info in infos:
            assert info.id == f"dev={info.values.dev}-ch={info.values.ch}"

    def test_per_sequence_rows(self):
        param = Parameter(
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            per_sequence_samples=True,
            nsamples=2,
        )

        infos = _infos(param)
        assert [(i.kind, i.index, i.enumerated, i.id) for i in infos] == [
            ("random", 0, ("dev",), "dev=a-rand-0"),
            ("random", 1, ("dev",), "dev=a-rand-1"),
            ("random", 0, ("dev",), "dev=b-rand-0"),
            ("random", 1, ("dev",), "dev=b-rand-1"),
        ]

    @pytest.mark.parametrize("record", [False, True], ids=["named", "record"])
    def test_skipped_row(self, record):
        param = Parameter(
            TestArg("ch", rng_type=Series([], skip_if_empty="no channels")),
            TestArg("x", rng_type=RNGInteger(0, 9)),
        )

        @dataclasses.dataclass
        class Row:
            ch: int
            x: int

        def test_fn(row: Row):
            pass

        if record:
            parametrization = build_parametrization(
                "strat",
                lambda: param,
                test_fn,
                config=_make_config(),
                pytest_fixtures=set(),
            )
        else:
            parametrization = _build(param)
        [info] = parametrization.infos
        assert (info.kind, info.name, info.index, info.id, info.enumerated) == (
            "skipped",
            None,
            None,
            "skipped",
            (),
        )
        assert info.values == (None, None) and type(info.values) is param.vector_type

    def test_record_mode_keeps_the_vector(self):
        """The info has the row's Vector, not the record the test receives."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            TestArg("y", rng_type=RNGInteger(0, 9)),
            directed_vectors={"one": (1, 1)},
            nsamples=1,
        )

        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        def test_fn(p: Point):
            pass

        parametrization = build_parametrization(
            "strat", lambda: param, test_fn, config=_make_config(), pytest_fixtures=set()
        )
        assert parametrization.values[0] == Point(1, 1)
        assert [(i.kind, i.id) for i in parametrization.infos] == [
            ("directed", "directed-one"),
            ("random", "rand-0"),
        ]
        assert parametrization.infos[0].values == (1, 1)
        assert all(isinstance(i.values, Vector) for i in parametrization.infos)

    def test_the_id_is_the_final_one(self):
        """In the values format, duplicate IDs are suffixed, and the info has the suffix."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"a": (1,), "b": (1,)},
            nsamples=0,
        )

        parametrization = _build(param, ids="values")
        assert parametrization.ids == ["x=1_0", "x=1_1"]
        assert [i.id for i in parametrization.infos] == ["x=1_0", "x=1_1"]

    def test_strategy_origin_seed_context_and_constraints_off(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"small": lambda v: v.x < 8, "odd": lambda v: v.x % 2},
            nsamples=1,
        )

        def factory():
            return param

        [info] = build_parametrization(
            "burst",
            factory,
            _test_fn("x"),
            config=_make_config(strategy_constraint_off=["odd"]),
            pytest_fixtures=set(),
        ).infos
        assert info.strategy == "burst"
        assert info.origin == _origin(factory, None)
        assert info.origin.endswith(
            f"{os.path.basename(__file__)}:{factory.__code__.co_firstlineno}"
        )
        assert info.seed == runtime.run_seed()
        assert info.context is None
        assert info.constraints_off == ("odd",)
        assert info.streams == 1

    def test_origin_is_relative_to_the_rootdir(self, tmp_path):
        def factory():
            pass

        config = MagicMock()
        config.rootpath = os.path.dirname(os.path.dirname(__file__))
        line = factory.__code__.co_firstlineno
        assert _origin(factory, config) == f"unittests/{os.path.basename(__file__)}:{line}"

        outside = MagicMock()
        outside.rootpath = tmp_path
        assert _origin(factory, outside) == f"{os.path.realpath(__file__)}:{line}"

    def test_no_rows_no_infos(self):
        """A vector filter that names none of the strategy's vectors gives no rows."""
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), directed_vectors={"a": (1,)})

        parametrization = _build(param, vector_name="other")
        assert parametrization.values == [] and parametrization.infos == ()
        assert parametrization.params() == []


class TestParams:
    """Each row reaches pytest as pytest.param(*values, id=..., marks=[..., strategy mark])."""

    @staticmethod
    def _strategy_marks(param_set):
        return [m for m in param_set.marks if m.name == "strategy"]

    def test_rows_carry_their_id_and_info(self):
        xfail = pytest.mark.xfail(strict=True)
        param = Parameter(
            TestArg("a", rng_type=RNGInteger(0, 9)),
            TestArg("b", rng_type=RNGInteger(0, 9)),
            directed_vectors={"marked": pytest.param(1, 2, marks=xfail), "plain": (3, 4)},
            nsamples=1,
        )

        parametrization = _build(param)
        params = parametrization.params()
        assert [p.id for p in params] == ["directed-marked", "directed-plain", "rand-0"]
        assert [p.values for p in params] == [(1, 2), (3, 4), tuple(parametrization.values[2])]
        # The vector's own marks come first, then the row's strategy mark
        assert [m.name for m in params[0].marks] == ["xfail", "strategy"]
        for p, info in zip(params, parametrization.infos, strict=True):
            [mark] = self._strategy_marks(p)
            assert mark.args == (info,) and mark.kwargs == {}

    def test_a_single_argument_is_one_value(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"pair": ((1, 2),), "marked": pytest.param(5, marks=pytest.mark.skip)},
            nsamples=0,
        )

        params = _build(param).params()
        assert [p.values for p in params] == [((1, 2),), (5,)]
        assert [m.name for m in params[1].marks] == ["skip", "strategy"]

    def test_the_skipped_row_keeps_its_skip_mark(self):
        param = Parameter(TestArg("ch", rng_type=Series([], skip_if_empty="no channels")))

        [row] = _build(param).params()
        assert row.id == "skipped" and row.values == (None,)
        assert [m.name for m in row.marks] == ["skip", "strategy"]
        assert row.marks[0].kwargs == {"reason": "no channels"}

    def test_resolve_and_parametrize_applies_them(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)), directed_vectors={"a": (1,)}, nsamples=1
        )

        marked = resolve_and_parametrize(
            "strat",
            _test_fn("x"),
            registry={"strat": lambda: param},
            config=_make_config(),
            pytest_fixtures=set(),
        )
        [mark] = marked.pytestmark
        assert mark.name == "parametrize" and mark.args[0] == "x" and "ids" not in mark.kwargs
        assert [p.id for p in mark.args[1]] == ["directed-a", "rand-0"]
        assert [p.marks[-1].args[0].kind for p in mark.args[1]] == ["directed", "random"]


class TestCountsByKind:
    """The -v summary counts the rows of each kind."""

    def test_count_rows(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"a": (0, 0)},
            test_vectors={"t": (1, 1)},
        )
        cases = [
            ({"n": 3}, (1, 3, 0, 0, 0)),
            ({"n": 0, "exhaustive": True}, (1, 0, 0, 2, 0)),
            ({"n": 3, "mode": "test"}, (0, 0, 1, 0, 0)),
            ({"n": 3, "mode": "random_only"}, (0, 3, 0, 0, 0)),
            ({"n": 3, "filter_by_name": "a"}, (1, 0, 0, 0, 0)),
        ]
        for options, expected in cases:
            resolution = Resolution(strategy="s", where="s.py")
            _count_rows(resolution, param._generate_rows(**options))
            counts = (
                resolution.directed,
                resolution.random,
                resolution.test,
                resolution.exhaustive,
                resolution.skipped,
            )
            assert counts == expected, options

        skipped = Parameter(TestArg("ch", rng_type=Series([], skip_if_empty="none")))
        resolution = Resolution(strategy="s", where="s.py")
        _count_rows(resolution, skipped._generate_rows(3))
        assert (resolution.random, resolution.skipped) == (0, 1)

    def test_summary_lists_directed_and_random_rows_and_the_other_kinds_there_are(self):
        resolutions = [
            Resolution(strategy="a", where="a.py", directed=1, random=4),
            Resolution(strategy="b", where="b.py", directed=2, exhaustive=6),
            Resolution(strategy="b", where="b.py", directed=2, exhaustive=6),
            Resolution(strategy="c", where="c.py", test=3),
            Resolution(strategy="d", where="d.py", skipped=1),
        ]

        assert _summary_lines(resolutions) == [
            "a (a.py): 1 test(s), 1 directed, 4 random rows",
            "b (b.py): 2 test(s), 4 directed, 0 random, 12 exhaustive rows",
            "c (c.py): 1 test(s), 0 directed, 0 random, 3 test rows",
            "d (d.py): 1 test(s), 0 directed, 0 random, 1 skipped rows",
        ]
