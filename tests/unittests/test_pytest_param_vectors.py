"""
Unit tests for pytest.param(*values, marks=...) as a directed or test vector.

A ParameterSet is a tuple of (values, marks, id), so the vector is checked by the
number of its values, not by its own length, and it reaches pytest with its marks.
An id= fails: the vector's name is its ID.
"""

import inspect

import pytest

from pytest_strategy import Parameter, RNGInteger, RNGValueError, TestArg
from pytest_strategy._resolver import build_parametrization

XFAIL = pytest.mark.xfail


def _args(*names):
    return [TestArg(name, rng_type=RNGInteger(0, 9)) for name in names]


def _test_fn(*argnames):
    """A dummy test function taking ``argnames``."""

    def test_dummy(*args, **kwargs):
        pass

    test_dummy.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in argnames]
    )
    return test_dummy


class TestLengthIsTheNumberOfValues:
    @pytest.mark.parametrize("kind", ["directed_vectors", "test_vectors"])
    def test_marked_vector_with_matching_values_builds(self, kind):
        vector = pytest.param(1, 2, marks=XFAIL)

        param = Parameter(*_args("a", "b"), **{kind: {"bad": vector}})

        assert getattr(param, kind) == {"bad": vector}

    @pytest.mark.parametrize("kind", ["directed_vectors", "test_vectors"])
    def test_one_argument_strategy_takes_a_one_value_param(self, kind):
        """3.0 compared the ParameterSet's own length (3) and rejected it."""
        param = Parameter(*_args("a"), **{kind: {"five": pytest.param(5, marks=XFAIL)}})

        assert getattr(param, kind)["five"].values == (5,)

    @pytest.mark.parametrize(
        ("kind", "label"), [("directed_vectors", "Directed"), ("test_vectors", "Test")]
    )
    def test_too_few_values_fail_when_the_parameter_is_built(self, kind, label):
        """3.0 accepted pytest.param(1, 2) for three arguments: the ParameterSet has length 3."""
        with pytest.raises(ValueError, match=f"^{label} vector 'short' has 2 values, expected 3$"):
            Parameter(*_args("a", "b", "c"), **{kind: {"short": pytest.param(1, 2)}})

    @pytest.mark.parametrize("kind", ["directed_vectors", "test_vectors"])
    def test_too_many_values_fail(self, kind):
        with pytest.raises(ValueError, match="'long' has 3 values, expected 2"):
            Parameter(*_args("a", "b"), **{kind: {"long": pytest.param(1, 2, 3, marks=XFAIL)}})

    @pytest.mark.parametrize(
        ("add", "kind", "label"),
        [
            ("add_directed_vector", "directed_vectors", "Directed"),
            ("add_test_vector", "test_vectors", "Test"),
        ],
    )
    def test_add_checks_the_values(self, add, kind, label):
        param = Parameter(*_args("a", "b"))
        vector = pytest.param(1, 2, marks=XFAIL)

        getattr(param, add)("bad", vector)

        with pytest.raises(ValueError, match=f"^{label} vector 'long' has 3 values, expected 2$"):
            getattr(param, add)("long", pytest.param(1, 2, 3))
        with pytest.raises(ValueError, match=f"^{label} vector 'short' has 1 values, expected 2$"):
            getattr(param, add)("short", pytest.param(1))
        assert getattr(param, kind) == {"bad": vector}


class TestAnIdFails:
    @pytest.mark.parametrize(
        ("kind", "label", "prefix"),
        [("directed_vectors", "Directed", "directed"), ("test_vectors", "Test", "test")],
    )
    def test_when_the_parameter_is_built(self, kind, label, prefix):
        with pytest.raises(RNGValueError) as excinfo:
            Parameter(*_args("a", "b"), **{kind: {"five": pytest.param(1, 2, id="x")}})

        assert str(excinfo.value) == (
            f"{label} vector 'five' is a pytest.param with id='x', but the vector's name is "
            f"its ID ({prefix}-five). Remove id=, and name the vector after the ID it should "
            "have"
        )

    @pytest.mark.parametrize("add", ["add_directed_vector", "add_test_vector"])
    def test_when_the_vector_is_added(self, add):
        param = Parameter(*_args("a"))

        with pytest.raises(RNGValueError, match="'five' is a pytest.param with id='five'"):
            getattr(param, add)("five", pytest.param(5, id="five", marks=XFAIL))
        assert not param.directed_vectors and not param.test_vectors

    def test_with_a_dict_value(self):
        with pytest.raises(RNGValueError, match="the vector's name is its ID"):
            Parameter(*_args("a"), directed_vectors={"z": pytest.param({"a": 0}, id="zero")})


class TestGeneration:
    def _param(self):
        return Parameter(
            *_args("a", "b"),
            directed_vectors={"bad": pytest.param(1, 2, marks=XFAIL), "ok": (3, 4)},
            test_vectors={"tv": pytest.param(5, 6, marks=XFAIL)},
        )

    def test_vectors_keep_their_marks(self):
        param = self._param()

        assert param.generate_vectors(0, mode="directed_only") == [
            pytest.param(1, 2, marks=XFAIL),
            (3, 4),
        ]
        assert param.generate_vectors(0, mode="test") == [pytest.param(5, 6, marks=XFAIL)]
        assert param.generate_vectors(0, filter_by_name="bad") == [pytest.param(1, 2, marks=XFAIL)]
        assert param.generate_vectors(0, filter_by_index=0) == [pytest.param(1, 2, marks=XFAIL)]

    def test_to_dict_lists_the_values(self):
        data = self._param().to_dict()

        assert data["directed_vectors"] == {"bad": ["1", "2"], "ok": ["3", "4"]}
        assert data["test_vectors"] == {"tv": ["5", "6"]}


class TestParametrization:
    def _build(self, param, *argnames):
        def factory(nsamples):
            return param

        return build_parametrization(
            "pp", factory, _test_fn(*argnames), config=None, pytest_fixtures=set()
        )

    def test_rows_reach_pytest_with_marks_and_their_name_as_id(self):
        param = Parameter(
            *_args("a", "b"),
            directed_vectors={
                "bad": pytest.param(1, 2, marks=XFAIL),
                "named": pytest.param({"b": 4, "a": 3}),
            },
        )

        parametrization = self._build(param, "a", "b")

        assert parametrization.argnames == "a,b"
        assert parametrization.values[:2] == [
            pytest.param(1, 2, marks=XFAIL),
            pytest.param(3, 4),
        ]
        assert parametrization.ids[:2] == ["directed-bad", "directed-named"]

    def test_one_argument_row_is_passed_as_the_param(self):
        param = Parameter(*_args("a"), directed_vectors={"five": pytest.param(5, marks=XFAIL)})

        parametrization = self._build(param, "a")

        assert parametrization.values[0] == pytest.param(5, marks=XFAIL)
