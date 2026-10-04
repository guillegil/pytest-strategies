"""
Unit tests for Vector: the rows of a strategy are tuples whose fields are the
argument names, every generation path builds them by filling the arguments in
declaration order, and argument names must be valid fields.
"""

import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import pytest_strategy
from pytest_strategy import (
    Parameter,
    RNGInteger,
    RNGSequence,
    RNGValueError,
    Series,
    TestArg,
    Vector,
)
from pytest_strategy._vector import vector_type

ADDR_LEN = ("addr", "len")


class TestVectorClass:
    def test_is_a_tuple_and_a_vector(self):
        v = vector_type(ADDR_LEN)(0, 4)

        assert isinstance(v, tuple)
        assert isinstance(v, Vector)
        assert type(v) is not tuple

    def test_compares_and_hashes_like_the_plain_tuple(self):
        v = vector_type(ADDR_LEN)(0, 4)

        assert v == (0, 4)
        assert hash(v) == hash((0, 4))
        assert {(0, 4): "row"}[v] == "row"

    def test_other_names_with_equal_values_are_equal(self):
        assert vector_type(ADDR_LEN)(0, 4) == vector_type(("x", "y"))(0, 4)

    def test_fields_are_the_names_in_order(self):
        v = vector_type(ADDR_LEN)(0, 4)

        assert v.addr == v[0] == 0
        assert v.len == v[1] == 4
        assert type(v)._fields == ADDR_LEN
        addr, length = v
        assert (addr, length, len(v)) == (0, 4, 2)

    def test_repr_names_the_fields(self):
        assert repr(vector_type(ADDR_LEN)(0, 16)) == "Vector(addr=0, len=16)"

    def test_asdict(self):
        assert vector_type(ADDR_LEN)(0, 4)._asdict() == {"addr": 0, "len": 4}

    def test_replace_keeps_the_class(self):
        v = vector_type(ADDR_LEN)(0, 4)

        replaced = v._replace(len=8)

        assert replaced == (0, 8)
        assert type(replaced) is type(v)

    def test_one_class_per_tuple_of_names(self):
        assert vector_type(ADDR_LEN) is vector_type(ADDR_LEN)
        assert vector_type(["addr", "len"]) is vector_type(ADDR_LEN)
        assert vector_type(("len", "addr")) is not vector_type(ADDR_LEN)

    def test_missing_field_lists_the_arguments(self):
        v = vector_type(ADDR_LEN)(0, 4)

        with pytest.raises(AttributeError) as excinfo:
            _ = v.lenght

        assert str(excinfo.value) == "Vector has no argument 'lenght'; its arguments are addr, len"

    def test_prefix_class_lists_only_its_own_fields(self):
        """The class of the arguments declared before another one (for 4.1)."""
        v = vector_type(("addr",))(0)

        assert isinstance(v, Vector)
        assert repr(v) == "Vector(addr=0)"
        with pytest.raises(AttributeError) as excinfo:
            _ = v.len
        assert str(excinfo.value) == "Vector has no argument 'len'; its arguments are addr"

    def test_private_names_are_never_arguments(self):
        v = vector_type(ADDR_LEN)(0, 4)

        # copy and pickle probe such names with getattr(..., None)
        assert getattr(v, "__deepcopy__", None) is None
        with pytest.raises(AttributeError, match="'Vector' object has no attribute '_x'"):
            _ = v._x

    def test_fields_named_count_and_index(self):
        v = vector_type(("count", "index"))(3, 7)

        assert (v.count, v.index) == (3, 7)
        assert tuple.index(v, 7) == 1
        assert tuple.count(v, 3) == 1
        assert len(v) == 2

    def test_pickle_round_trip(self):
        v = vector_type(ADDR_LEN)(0, [4])

        loaded = pickle.loads(pickle.dumps(v))

        assert loaded == v
        assert type(loaded) is type(v)

    def test_pickle_loads_in_a_fresh_process(self):
        """The classes are built at run time, so pickle must not look them up by name."""
        rows = [vector_type(ADDR_LEN)(0, 4), vector_type(("addr",))(1)]
        code = (
            "import pickle, sys\n"
            "rows = pickle.loads(sys.stdin.buffer.read())\n"
            "print([(repr(r), type(r)._fields, r.addr) for r in rows])\n"
        )
        package_root = str(Path(pytest_strategy.__file__).resolve().parents[1])
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [package_root, os.environ.get("PYTHONPATH")])
            ),
        }

        result = subprocess.run(
            [sys.executable, "-c", code],
            input=pickle.dumps(rows),
            capture_output=True,
            check=True,
            env=env,
        )

        assert result.stdout.decode().strip() == (
            "[('Vector(addr=0, len=4)', ('addr', 'len'), 0), ('Vector(addr=1)', ('addr',), 1)]"
        )

    def test_deepcopy(self):
        v = vector_type(ADDR_LEN)(0, [4])

        copied = copy.deepcopy(v)

        assert copied == v
        assert type(copied) is type(v)
        assert copied.len is not v.len
        assert type(copy.copy(v)) is type(v)

    def test_vector_itself_is_not_instantiated(self):
        with pytest.raises(TypeError, match="Parameter.vector_type"):
            Vector((1, 2))

    def test_names_that_are_not_fields_are_rejected(self):
        with pytest.raises(ValueError):
            vector_type(("_x",))


class TestArgumentNames:
    @pytest.mark.parametrize(
        "name, problem",
        [
            ("_x", "starts with '_'"),
            ("a-b", "is not a Python identifier"),
            ("class", "is a Python keyword"),
            ("1st", "is not a Python identifier"),
            ("", "is not a Python identifier"),
        ],
    )
    def test_invalid_name_raises_naming_the_argument(self, name, problem):
        with pytest.raises(RNGValueError) as excinfo:
            Parameter(TestArg("ok", value=1), TestArg(name, value=2))

        assert f"Parameter argument {name!r} {problem}" in str(excinfo.value)

    @pytest.mark.parametrize("name", ["type", "match", "case", "café"])
    def test_soft_keywords_and_non_ascii_names_work(self, name):
        param = Parameter(TestArg(name, rng_type=RNGInteger(0, 9)))

        row = param.generate_vector()

        assert getattr(row, name) == row[0]
        assert type(row)._fields == (name,)


class Recorder:
    """An rng_type that logs each draw and returns its own name."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def generate(self):
        self.log.append(self.name)
        return self.name


class TestGenerationPaths:
    """Every path hands the constraints and returns a Vector over arg_names."""

    @staticmethod
    def _recording():
        seen = []

        def record(v):
            seen.append(v)
            return True

        return seen, record

    def _check(self, param, seen, rows):
        assert seen, "the constraint was not called"
        assert all(type(v)._fields == param.arg_names for v in seen)
        assert rows
        assert all(type(row) is param.vector_type for row in rows)

    def test_vector_type_is_the_class_for_the_argument_names(self):
        param = Parameter(TestArg("addr", value=0), TestArg("len", value=4))

        assert param.vector_type is vector_type(("addr", "len"))
        assert param.vector_type._fields == param.arg_names

    def test_generate_vector(self):
        seen, record = self._recording()
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 9)),
            TestArg("len", rng_type=RNGInteger(0, 9)),
            vector_constraints=[record],
        )

        self._check(param, seen, [param.generate_vector()])

    def test_plain_path(self):
        seen, record = self._recording()
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 9)),
            TestArg("len", rng_type=RNGInteger(0, 9)),
            vector_constraints=[record],
        )

        self._check(param, seen, param.generate_vectors(5, mode="random_only"))

    def test_finite_series_path(self):
        seen, record = self._recording()
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("data", rng_type=RNGInteger(0, 9)),
            vector_constraints=[record],
        )

        rows = param.generate_vectors(4, mode="random_only")

        self._check(param, seen, rows)
        assert [row.ch for row in rows] == [0, 1, 0, 1]

    def test_series_only_path(self):
        seen, record = self._recording()
        param = Parameter(TestArg("ch", rng_type=Series([0, 1])), vector_constraints=[record])

        self._check(param, seen, param.generate_vectors(2, mode="random_only"))

    def test_per_sequence_samples_path(self):
        seen, record = self._recording()
        param = Parameter(
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            TestArg("data", rng_type=RNGInteger(0, 9)),
            vector_constraints=[record],
            per_sequence_samples=True,
        )

        rows = param.generate_vectors(2, mode="random_only")

        self._check(param, seen, rows)
        assert [row.dev for row in rows] == ["a", "a", "b", "b"]

    def test_exhaustive_path(self):
        seen, record = self._recording()
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            TestArg("data", rng_type=RNGInteger(0, 9)),
            vector_constraints=[record],
        )

        rows = param.generate_exhaustive()

        self._check(param, seen, rows)
        assert sorted((row.ch, row.dev) for row in rows) == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]

    def test_rows_are_filled_in_declaration_order(self):
        """The drawn arguments are generated in declaration order around the Series one,
        and the constraint sees the complete row."""
        log = []
        seen, record = self._recording()
        param = Parameter(
            TestArg("a", rng_type=Recorder("a", log)),
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("b", rng_type=Recorder("b", log)),
            vector_constraints=[record],
        )

        rows = param.generate_vectors(2, mode="random_only")

        assert log == ["a", "b", "a", "b"]
        assert rows == [("a", 0, "b"), ("a", 1, "b")]
        assert seen == rows

    def test_rejected_rows_are_redrawn(self):
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 15)),
            TestArg("len", rng_type=RNGInteger(1, 4)),
            vector_constraints=[lambda v: v.addr % 4 == 0],
        )

        rows = param.generate_vectors(20, mode="random_only")

        assert all(row.addr % 4 == 0 for row in rows)
