"""
Unit tests for the keyword-only options and the rng_type check of 4.0.

The options after TestArg's rng_type, strategy()'s validate_signature,
export_strategies()'s format, Parameter.generate_vectors()'s options after n and
Parameter.add_constraint()'s name are keyword-only, so an option a 4.x release
adds cannot shift a positional call.
An rng_type that is neither an RNGType nor has a generate() method is a TypeError,
so a later release can give other callables a meaning there without changing what
an existing call does.
"""

import inspect
from typing import Any

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGBoolean,
    RNGInteger,
    RNGType,
    Series,
    Strategy,
    TestArg,
    export_strategies,
    register,
    strategy,
)

POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD = inspect.Parameter.KEYWORD_ONLY

RNG_TYPE_ERROR = r"TestArg 'x' rng_type must be an RNGType or have a generate\(\) method, got "


def _param():
    return Parameter(
        TestArg("x", rng_type=RNGInteger(0, 9)),
        directed_vectors={"zero": (0,)},
    )


class Constant(RNGType[int]):
    """An RNG type of the user's own."""

    def generate(self) -> int:
        return 7

    @property
    def python_type(self) -> type[int]:
        return int


class DuckTyped:
    """Not an RNGType, but it has generate() (and python_type)."""

    python_type = int

    def generate(self):
        return 3


class GenerateOnly:
    """The least an rng_type needs: generate(), without python_type."""

    def generate(self):
        return 4


class TestSignatures:
    @pytest.mark.parametrize(
        "fn, positional, keyword_only",
        [
            (TestArg, ["name", "rng_type"], {"value", "validator", "description"}),
            (strategy, ["name"], {"validate_signature"}),
            (export_strategies, [], {"format"}),
            (
                Parameter.generate_vectors,
                ["self", "n"],
                {"mode", "filter_by_name", "filter_by_index", "constraints_off"},
            ),
            (Parameter.generate_exhaustive, ["self"], {"constraints_off"}),
            (Parameter.add_constraint, ["self", "fn"], {"name"}),
            # Unchanged: 4.1 adds its options after a * as well
            (register, ["name"], set()),
        ],
        ids=[
            "TestArg",
            "strategy",
            "export_strategies",
            "generate_vectors",
            "generate_exhaustive",
            "add_constraint",
            "register",
        ],
    )
    def test_only_the_leading_parameters_are_positional(self, fn, positional, keyword_only):
        params = inspect.signature(fn).parameters.values()

        assert [(p.name, p.kind) for p in params if p.kind is not KEYWORD] == [
            (name, POSITIONAL) for name in positional
        ]
        assert keyword_only <= {p.name for p in params if p.kind is KEYWORD}

    def test_strategy_facade_has_the_same_functions(self):
        assert Strategy.strategy is strategy
        assert Strategy.export_strategies is export_strategies


class TestPositionalCalls:
    @pytest.mark.parametrize("rng_type", [RNGInteger(0, 1), None], ids=["rng_type", "None"])
    def test_testarg_value_by_position(self, rng_type):
        with pytest.raises(
            TypeError, match="takes from 2 to 3 positional arguments but 4 were given"
        ):
            TestArg("x", rng_type, 5)

    def test_testarg_validator_and_description_by_position(self):
        with pytest.raises(
            TypeError, match="takes from 2 to 3 positional arguments but 6 were given"
        ):
            TestArg("x", RNGInteger(0, 1), None, bool, "a description")

    def test_strategy_flag_by_position(self):
        with pytest.raises(TypeError, match=r"strategy\(\) takes 1 positional argument"):
            strategy("s", False)
        with pytest.raises(TypeError, match=r"strategy\(\) takes 1 positional argument"):
            Strategy.strategy("s", False)

    def test_generate_vectors_mode_by_position(self):
        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
            _param().generate_vectors(1, "all")

    def test_generate_vectors_filter_by_position(self):
        with pytest.raises(TypeError, match="takes 2 positional arguments but 4 were given"):
            _param().generate_vectors(1, "all", "zero")

    def test_add_constraint_name_by_position(self):
        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
            _param().add_constraint(lambda v: True, "n")

    def test_export_format_by_position(self):
        with pytest.raises(TypeError, match=r"export_strategies\(\) takes 0 positional"):
            export_strategies("json")
        with pytest.raises(TypeError, match=r"export_strategies\(\) takes 0 positional"):
            Strategy.export_strategies("json")


class TestKeywordCalls:
    def test_testarg_options_by_keyword(self):
        arg = TestArg(
            "x", RNGInteger(0, 9), validator=lambda v: v >= 0, description="a description"
        )
        static = TestArg("y", value=5, description="fixed")

        assert 0 <= arg.generate() <= 9
        assert str(arg) == "x: a description"
        assert static.generate() == 5

    def test_strategy_flag_by_keyword(self):
        @strategy("s", validate_signature=False)
        def takes_x(x):
            pass

        mark = takes_x.pytestmark[0]
        assert mark.name == "strategy"
        assert mark.args == ("s",)
        assert mark.kwargs == {"validate_signature": False}

    def test_generate_vectors_options_by_keyword(self):
        param = _param()

        assert len(param.generate_vectors(3, mode="random_only")) == 3
        assert param.generate_vectors(3, filter_by_name="zero") == [(0,)]
        assert param.generate_vectors(3, filter_by_index=0) == [(0,)]

    def test_add_constraint_name_by_keyword(self):
        param = _param()

        assert param.add_constraint(lambda v: True, name="n") == "n"
        assert list(param.vector_constraints) == ["n"]

    def test_export_format_by_keyword_is_still_checked(self):
        with pytest.raises(ValueError, match="Unsupported format: xml"):
            export_strategies(format="xml")


class TestRngTypeCheck:
    @pytest.mark.parametrize(
        "rng_type",
        [lambda v: 1, len, 5, "abc", [1, 2], object(), int],
        ids=["lambda", "function", "int", "str", "list", "object", "class"],
    )
    def test_without_generate_is_a_type_error(self, rng_type):
        with pytest.raises(TypeError, match=RNG_TYPE_ERROR):
            TestArg("x", rng_type=rng_type)

    @pytest.mark.parametrize(
        "rng_type",
        [RNGBoolean, RNGInteger, Constant, DuckTyped],
        ids=["RNGBoolean", "RNGInteger", "RNGType subclass", "duck-typed"],
    )
    def test_class_instead_of_an_instance_is_a_type_error(self, rng_type):
        """The class has generate(), but unbound: the error says to build an instance."""
        name = rng_type.__name__
        with pytest.raises(TypeError) as exc_info:
            TestArg("x", rng_type=rng_type)

        assert str(exc_info.value) == (
            "TestArg 'x' rng_type must be an RNGType or have a generate() method, got the "
            f"class {name} instead of an instance (did you mean {name}(...)?)"
        )

    def test_lambda_by_position_is_a_type_error(self):
        with pytest.raises(TypeError, match=RNG_TYPE_ERROR + "<function"):
            TestArg("x", lambda v: 1)

    def test_generate_that_is_not_callable_is_a_type_error(self):
        class Holder:
            generate = 5

        with pytest.raises(TypeError, match=RNG_TYPE_ERROR):
            TestArg("x", rng_type=Holder())

    def test_checked_also_next_to_a_value(self):
        """A static value does not make a bad rng_type acceptable."""
        with pytest.raises(TypeError, match=RNG_TYPE_ERROR):
            TestArg("x", rng_type=lambda v: 1, value=5)

    def test_missing_value_and_rng_type_is_still_a_value_error(self):
        with pytest.raises(ValueError, match="TestArg 'x' must have a value or an rng_type"):
            TestArg("x")

    @pytest.mark.parametrize(
        "rng_type, expected",
        [(RNGInteger(4, 4), 4), (Series([6]), 6), (Constant(), 7), (DuckTyped(), 3)],
        ids=["RNGInteger", "Series", "RNGType subclass", "duck-typed"],
    )
    def test_rng_types_and_objects_with_generate_are_accepted(self, rng_type, expected):
        RNG.seed(1)
        param = Parameter(TestArg("x", rng_type=rng_type))

        assert param.generate_vectors(2, mode="random_only") == [(expected,), (expected,)]
        assert param.arg_types == (int,)

    def test_object_with_generate_only_has_type_any(self):
        """generate() is all an rng_type needs: without python_type the type is Any."""
        arg = TestArg("x", rng_type=GenerateOnly())
        param = Parameter(arg)

        assert param.generate_vectors(2, mode="random_only") == [(4,), (4,)]
        assert arg.type is Any
        assert param.arg_types == (Any,)
        assert repr(arg) == "TestArg(name='x', type=Any)"
        assert arg.to_dict()["rng"]["type"] == "GenerateOnly"
