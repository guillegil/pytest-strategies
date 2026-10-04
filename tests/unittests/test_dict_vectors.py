"""
Unit tests for the vector normalizer (D13): directed and test vectors given as
dicts, namedtuples, tuples, lists or pytest.param(...) are stored as Vectors in
declaration order, strings, bytes and scalars fail with a hint, record instances
are not supported yet, vector names are non-empty strings, and
``directed_vectors`` and ``test_vectors`` are read-only mappings.
"""

import copy
import dataclasses
import inspect
from collections import namedtuple
from typing import NamedTuple

import pytest

from pytest_strategy import Parameter, RNGInteger, RNGValueError, TestArg, Vector
from pytest_strategy._resolver import build_parametrization

XFAIL = pytest.mark.xfail

KINDS = pytest.mark.parametrize(
    ("kind", "label"), [("directed_vectors", "Directed"), ("test_vectors", "Test")]
)
ADDS = pytest.mark.parametrize(
    ("add", "kind", "label"),
    [
        ("add_directed_vector", "directed_vectors", "Directed"),
        ("add_test_vector", "test_vectors", "Test"),
    ],
)


def _param(*names, **kwargs):
    return Parameter(*(TestArg(n, rng_type=RNGInteger(0, 99)) for n in names), **kwargs)


def _addr_len(**kwargs):
    return _param("addr", "len", **kwargs)


def _test_fn(*argnames):
    """A dummy test function taking ``argnames``."""

    def test_dummy(*args, **kwargs):
        pass

    test_dummy.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in argnames]
    )
    return test_dummy


def _build(param, *argnames):
    def factory():
        return param

    return build_parametrization(
        "dv", factory, _test_fn(*argnames), config=None, pytest_fixtures=set()
    )


class BusTxn(NamedTuple):
    len: int
    addr: int


@dataclasses.dataclass
class Burst:
    addr: int
    len: int


class TestDictVectors:
    @KINDS
    def test_an_order_free_dict_gives_a_vector_in_declaration_order(self, kind, label):
        param = _addr_len(**{kind: {"zeros": {"len": 16, "addr": 0}}})

        vector = getattr(param, kind)["zeros"]

        assert type(vector) is param.vector_type
        assert repr(vector) == "Vector(addr=0, len=16)"
        assert vector == (0, 16)

    @KINDS
    def test_an_unknown_key_fails_with_did_you_mean(self, kind, label):
        with pytest.raises(
            RNGValueError,
            match=(
                f"^{label} vector 'zeros' has unknown argument 'lenght' "
                r"\(did you mean 'len'\?\) and is missing 'len'\. The keys of a dict "
                "vector are the strategy's arguments: addr, len$"
            ),
        ):
            _addr_len(**{kind: {"zeros": {"addr": 0, "lenght": 16}}})

    def test_a_missing_key_fails(self):
        with pytest.raises(RNGValueError, match="^Directed vector 'zeros' is missing 'len'\\. "):
            _addr_len(directed_vectors={"zeros": {"addr": 0}})

    def test_an_extra_key_fails(self):
        with pytest.raises(
            RNGValueError, match="^Directed vector 'zeros' has unknown arguments 'x', 'y'\\. "
        ):
            _addr_len(directed_vectors={"zeros": {"addr": 0, "len": 1, "x": 2, "y": 3}})

    def test_a_key_that_is_not_a_str_fails(self):
        with pytest.raises(
            RNGValueError, match="^Directed vector 'zeros' has the key 0, which is not a str\\. "
        ):
            _addr_len(directed_vectors={"zeros": {0: 0, "len": 1}})

    def test_one_argument_errors_show_how_to_write_a_dict_value(self):
        with pytest.raises(RNGValueError) as excinfo:
            _param("cfg", directed_vectors={"v": {"a": 1}})

        assert str(excinfo.value) == (
            "Directed vector 'v' has unknown argument 'a' and is missing 'cfg'. The keys of "
            "a dict vector are the strategy's arguments: cfg. A dict value for the one "
            "argument is written ({'a': 1},) or {'cfg': {'a': 1}}"
        )

    def test_a_pytest_param_shows_its_own_way_to_write_a_dict_value(self):
        """Inside a pytest.param, ({'a': 1},) is read by position and passes the tuple."""
        with pytest.raises(RNGValueError) as excinfo:
            _param("cfg", test_vectors={"v": pytest.param({"a": 1}, marks=pytest.mark.xfail)})

        assert str(excinfo.value) == (
            "Test vector 'v' has unknown argument 'a' and is missing 'cfg'. The keys of a dict "
            "vector are the strategy's arguments: cfg. In a pytest.param, a dict value for "
            "the one argument is written pytest.param({'cfg': {'a': 1}}, marks=...)"
        )
        written = _param(
            "cfg", test_vectors={"v": pytest.param({"cfg": {"a": 1}}, marks=pytest.mark.xfail)}
        )
        assert written.test_vectors["v"].values.cfg == {"a": 1}

    def test_tuple_and_dict_vectors_mix(self):
        param = _addr_len(
            directed_vectors={"t": (1, 2), "d": {"len": 4, "addr": 3}, "l": [5, 6]},
            test_vectors={"d": {"addr": 7, "len": 8}, "t": (9, 10)},
        )

        assert param.generate_vectors(0, mode="directed_only") == [(1, 2), (3, 4), (5, 6)]
        assert param.generate_vectors(0, mode="test") == [(7, 8), (9, 10)]

    @ADDS
    def test_add_accepts_a_dict(self, add, kind, label):
        param = _addr_len()

        getattr(param, add)("zeros", {"len": 0, "addr": 1})

        assert getattr(param, kind) == {"zeros": (1, 0)}
        assert getattr(param, kind)["zeros"].addr == 1
        with pytest.raises(RNGValueError, match=f"^{label} vector 'bad' is missing 'addr'"):
            getattr(param, add)("bad", {"len": 0})

    def test_a_one_argument_dict_gives_its_value(self):
        param = _param("x", directed_vectors={"five": {"x": 5}})

        assert param.generate_vectors(0, mode="directed_only") == [(5,)]
        assert _build(param, "x").values[0] == 5

    def test_a_dict_reaches_the_test_as_its_values(self):
        """3.0 stored tuple(dict), the keys, and passed those to the test."""
        param = _addr_len(directed_vectors={"d": {"len": 4, "addr": 3}})

        parametrization = _build(param, "addr", "len")

        assert parametrization.values[0] == (3, 4)

    def test_a_dict_value_for_one_argument_is_wrapped(self):
        param = _param("cfg", directed_vectors={"t": ({"a": 1},), "d": {"cfg": {"a": 2}}})

        assert _build(param, "cfg").values[:2] == [{"a": 1}, {"a": 2}]


class TestPositionalVectors:
    @KINDS
    def test_lists_stay_positional(self, kind, label):
        param = _addr_len(**{kind: {"l": [1, 2]}})

        vector = getattr(param, kind)["l"]

        assert isinstance(vector, Vector)
        assert (vector.addr, vector.len) == (1, 2)

    def test_any_iterable_is_positional(self):
        param = _addr_len(directed_vectors={"r": range(2), "g": (v for v in (3, 4))})

        assert param.directed_vectors == {"r": (0, 1), "g": (3, 4)}

    @KINDS
    def test_the_length_is_checked(self, kind, label):
        with pytest.raises(RNGValueError, match=f"^{label} vector 'l' has 3 values, expected 2$"):
            _addr_len(**{kind: {"l": [1, 2, 3]}})


class TestNamedtupleVectors:
    def test_a_namedtuple_is_placed_by_its_field_names(self):
        param = _addr_len(directed_vectors={"txn": BusTxn(len=4, addr=0)})

        vector = param.get_directed_vector("txn")

        assert type(vector) is param.vector_type
        assert repr(vector) == "Vector(addr=0, len=4)"

    def test_a_collections_namedtuple_is_placed_by_name_too(self):
        Point = namedtuple("Point", "len addr")

        param = _addr_len(test_vectors={"p": Point(4, 0)})

        assert param.get_test_vector("p") == (0, 4)

    def test_a_vector_of_the_strategy_is_kept(self):
        param = _addr_len()
        row = param.vector_type(addr=1, len=2)

        param.add_directed_vector("row", row)

        assert param.get_directed_vector("row") is row

    def test_a_vector_of_another_order_is_placed_by_name(self):
        param = _addr_len()

        param.add_directed_vector("row", _param("len", "addr").vector_type(len=2, addr=1))

        assert param.get_directed_vector("row") == (1, 2)

    def test_other_field_names_fail(self):
        Other = namedtuple("Other", "adr len")

        with pytest.raises(RNGValueError) as excinfo:
            _addr_len(directed_vectors={"o": Other(0, 4)})

        assert str(excinfo.value) == (
            "Directed vector 'o' (Other, a namedtuple placed by its field names) has unknown "
            "argument 'adr' (did you mean 'addr'?) and is missing 'addr'. The fields of a "
            "namedtuple vector are the strategy's arguments: addr, len"
        )


class TestRecordInstances:
    @KINDS
    def test_a_dataclass_instance_fails(self, kind, label):
        with pytest.raises(
            RNGValueError,
            match=(
                f"^{label} vector 'b' is an instance of Burst, a dataclass: record instances "
                "as vectors are not supported yet; use a dict, such as "
                r"\{'addr': \.\.\., 'len': \.\.\.\}$"
            ),
        ):
            _addr_len(**{kind: {"b": Burst(addr=0, len=4)}})

    def test_a_pydantic_model_instance_fails(self):
        pydantic = pytest.importorskip("pydantic")

        class Txn(pydantic.BaseModel):
            addr: int
            len: int

        with pytest.raises(RNGValueError, match="not supported yet; use a dict"):
            _addr_len(directed_vectors={"m": Txn(addr=0, len=4)})

    def test_a_pydantic_dataclass_instance_fails(self):
        pydantic = pytest.importorskip("pydantic")

        @pydantic.dataclasses.dataclass
        class Txn:
            addr: int
            len: int

        with pytest.raises(RNGValueError, match="not supported yet; use a dict"):
            _addr_len(directed_vectors={"m": Txn(addr=0, len=4)})

    def test_a_dataclass_class_is_not_a_record_instance(self):
        with pytest.raises(RNGValueError, match=r"\(type\), not a tuple of values"):
            _addr_len(directed_vectors={"b": Burst})


class TestScalarVectors:
    @pytest.mark.parametrize("raw", ["a", b"\x00", 5, None])
    @KINDS
    def test_a_one_argument_strategy_gets_the_hint(self, kind, label, raw):
        with pytest.raises(RNGValueError) as excinfo:
            _param("x", **{kind: {"v": raw}})

        shown = repr(raw)
        assert str(excinfo.value) == (
            f"{label} vector 'v' is {shown} ({type(raw).__name__}), not a tuple of values. "
            f"For a one-argument strategy write ({shown},) or {{'x': {shown}}}"
        )

    def test_the_hint_shows_the_example_of_the_plan(self):
        with pytest.raises(RNGValueError, match=r"write \('a',\) or \{'x': 'a'\}$"):
            _param("x", directed_vectors={"v": "a"})

    def test_bytearray_fails(self):
        with pytest.raises(RNGValueError, match=r"\(bytearray\), not a tuple of values"):
            _param("x", directed_vectors={"v": bytearray(b"ab")})

    def test_a_scalar_in_a_wider_strategy_names_the_arguments(self):
        with pytest.raises(
            RNGValueError,
            match=(
                r"^Directed vector 'v' is 0 \(int\), not a tuple of values\. Give one value per "
                r"argument \(addr, len\), as a tuple or a dict of argument names to values$"
            ),
        ):
            _addr_len(directed_vectors={"v": 0})

    @ADDS
    def test_add_rejects_a_scalar(self, add, kind, label):
        param = _param("x")

        with pytest.raises(RNGValueError, match=f"^{label} vector 'v' is 'abc' \\(str\\)"):
            getattr(param, add)("v", "abc")
        assert getattr(param, kind) == {}


class TestPytestParamVectors:
    @KINDS
    def test_a_dict_in_a_pytest_param_is_a_named_vector(self, kind, label):
        vector = pytest.param({"len": 0, "addr": 1}, marks=XFAIL)

        stored = getattr(_addr_len(**{kind: {"z": vector}}), kind)["z"]

        assert repr(stored.values) == "Vector(addr=1, len=0)"
        assert stored.marks == vector.marks
        assert stored.id is None

    def test_a_dict_in_a_one_argument_pytest_param_is_a_named_vector(self):
        param = _param("cfg", directed_vectors={"c": pytest.param({"cfg": {"a": 1}})})

        assert param.get_directed_vector("c").values == ({"a": 1},)
        assert [row.values for row in _build(param, "cfg").values[:1]] == [({"a": 1},)]

    def test_a_dict_in_a_pytest_param_is_checked(self):
        with pytest.raises(RNGValueError, match="^Test vector 'z' is missing 'len'"):
            _addr_len(test_vectors={"z": pytest.param({"addr": 1}, marks=XFAIL)})

    def test_values_are_positional_otherwise(self):
        param = _addr_len(directed_vectors={"p": pytest.param(1, 2, marks=XFAIL)})

        stored = param.get_vector_by_index(0)

        assert type(stored.values) is param.vector_type
        assert stored == pytest.param(1, 2, marks=XFAIL)

    def test_a_string_value_is_one_value(self):
        param = _param("x", directed_vectors={"s": pytest.param("ab")})

        assert param.get_directed_vector("s").values == ("ab",)

    @pytest.mark.parametrize(
        "value", [(-2,), [-2], namedtuple("N", "n")(n=-2)], ids=["tuple", "list", "namedtuple"]
    )
    def test_a_tuple_list_or_namedtuple_inside_is_one_value(self, value):
        """
        The docs say a pytest.param takes the values spread or one dict: a tuple,
        list or namedtuple inside it is the one argument's value, not the row.
        """
        param = _param("n", directed_vectors={"v": pytest.param(value, marks=XFAIL)})

        stored = param.get_directed_vector("v")

        assert stored.values == (value,)
        assert stored.values.n is value
        assert _param("n", directed_vectors={"v": pytest.param(-2)}).get_directed_vector(
            "v"
        ).values == (-2,)

    def test_a_namedtuple_inside_is_not_placed_by_its_field_names(self):
        assert _addr_len(directed_vectors={"nt": BusTxn(len=8, addr=7)}).get_directed_vector(
            "nt"
        ) == (7, 8)
        with pytest.raises(RNGValueError, match="^Directed vector 'nt' has 1 values, expected 2$"):
            _addr_len(directed_vectors={"nt": pytest.param(BusTxn(len=8, addr=7), marks=XFAIL)})


class TestVectorNames:
    @pytest.mark.parametrize("name", [0, "", None, ("a",)])
    @KINDS
    def test_a_name_must_be_a_non_empty_str(self, kind, label, name):
        with pytest.raises(
            RNGValueError, match=f"^{label} vector names must be non-empty strings, got "
        ):
            _addr_len(**{kind: {name: (0, 0)}})

    @ADDS
    def test_add_checks_the_name(self, add, kind, label):
        param = _addr_len()

        with pytest.raises(RNGValueError, match=f"^{label} vector names must be non-empty"):
            getattr(param, add)(1, (0, 0))
        with pytest.raises(RNGValueError, match=f"^{label} vector names must be non-empty"):
            getattr(param, add)("", (0, 0))

    def test_any_other_str_is_a_name(self):
        param = _addr_len(directed_vectors={"café": (0, 0), "a b": (1, 1), "x=1": (2, 2)})

        assert param.vector_names == ["café", "a b", "x=1"]


class TestReadOnlyMappings:
    @KINDS
    def test_item_assignment_raises(self, kind, label):
        param = _addr_len(**{kind: {"a": (1, 1)}})

        with pytest.raises(TypeError):
            getattr(param, kind)["x"] = (1, 2)
        with pytest.raises(TypeError):
            del getattr(param, kind)["a"]
        assert getattr(param, kind) == {"a": (1, 1)}

    @KINDS
    def test_the_attribute_cannot_be_replaced(self, kind, label):
        param = _addr_len()

        with pytest.raises(AttributeError):
            setattr(param, kind, {"x": (1, 2)})

    def test_mappings_compare_equal_to_dicts(self):
        param = _addr_len(directed_vectors={"a": (1, 1)}, test_vectors={"b": {"addr": 2, "len": 3}})

        assert param.directed_vectors == {"a": (1, 1)}
        assert param.test_vectors == {"b": (2, 3)}
        assert dict(param.directed_vectors) == {"a": (1, 1)}

    def test_add_and_remove_change_the_mapping(self):
        param = _addr_len(directed_vectors={"a": (1, 1)})
        view = param.directed_vectors

        param.add_directed_vector("b", {"addr": 2, "len": 2})
        param.remove_directed_vector("a")

        assert list(param.directed_vectors) == ["b"]
        assert view == {"b": (2, 2)}

    def test_the_callers_dict_is_copied(self):
        given = {"a": (1, 1)}
        param = _addr_len(directed_vectors=given)

        param.add_directed_vector("b", (2, 2))

        assert given == {"a": (1, 1)}

    def test_a_parameter_can_still_be_deep_copied(self):
        param = _addr_len(directed_vectors={"a": {"addr": 1, "len": 1}})

        assert copy.deepcopy(param).directed_vectors == {"a": (1, 1)}


class TestAccessorsReturnVectors:
    def test_every_accessor_returns_the_stored_vector(self):
        param = _addr_len(directed_vectors={"d": (1, 2)}, test_vectors={"t": [3, 4]})

        for vector in (
            param.get_directed_vector("d"),
            param.get_vector_by_name("d"),
            param.get_vector_by_index(0),
            param.get_test_vector("t"),
            *param.generate_vectors(0, mode="directed_only"),
            *param.generate_vectors(0, mode="test"),
            *param.generate_vectors(0, filter_by_name="d"),
        ):
            assert type(vector) is param.vector_type

    def test_to_dict_lists_the_values(self):
        data = _addr_len(directed_vectors={"d": {"len": 2, "addr": 1}}).to_dict()

        # In declaration order, whatever the dict's order
        assert data["directed_vectors"] == [
            {"name": "d", "id": "directed-d", "values": {"addr": 1, "len": 2}}
        ]
        assert list(data["directed_vectors"][0]["values"]) == ["addr", "len"]
