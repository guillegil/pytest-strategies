"""
Unit tests for the APIs that 3.0 deprecated and 4.0 removes.

Tuple factories, RNG.set_max_retries(), configure(), Strategy.set_config() and the
TestArg directed_values=, test_values= and always_include_directed= options are
gone, without stubs and without warnings left behind.
"""

import json
import os

import pytest

import pytest_strategy
from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    Strategy,
    TestArg,
    export_strategies,
    register,
)
from pytest_strategy._resolver import build_parametrization, check_factory_result
from pytest_strategy._runtime import StrategyRuntime

REMOVED_TO_DICT_KEYS = {"has_directed_values", "has_test_values", "always_include_directed"}


def _tuple_factory(nsamples):
    return ("x",), [(1,), (2,)]


def takes_x(x):
    """Stand in for a test that takes the strategy's argument."""


class TestRemovedFunctions:
    def test_rng_has_no_set_max_retries(self):
        assert not hasattr(RNG, "set_max_retries")

    def test_package_has_no_configure(self):
        assert not hasattr(pytest_strategy, "configure")
        with pytest.raises(ImportError, match="configure"):
            from pytest_strategy import configure  # noqa: F401

    def test_strategy_has_no_set_config(self):
        assert not hasattr(Strategy, "set_config")

    def test_runtime_has_no_config_property(self):
        """set_config() was the only writer of runtime.config; each session keeps its own."""
        assert not hasattr(StrategyRuntime, "config")


class TestTestArgOptions:
    @pytest.mark.parametrize(
        "option, value",
        [("directed_values", [1]), ("test_values", [1]), ("always_include_directed", True)],
    )
    def test_removed_option_is_a_type_error(self, option, value):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{option}'"):
            TestArg("x", rng_type=RNGInteger(0, 9), **{option: value})

    def test_value_or_rng_type_is_required(self):
        with pytest.raises(ValueError, match="TestArg 'x' must have a value or an rng_type"):
            TestArg("x")

    @pytest.mark.parametrize("name", ["directed_values", "test_values", "has_directed_values"])
    def test_removed_properties(self, name):
        assert not hasattr(TestArg("x", value=1), name)

    @pytest.mark.parametrize(
        "arg", [TestArg("x", value=4), TestArg("x", rng_type=RNGInteger(0, 9))], ids=repr
    )
    def test_to_dict_has_none_of_the_removed_keys(self, arg):
        assert not REMOVED_TO_DICT_KEYS & set(arg.to_dict())

    def test_generate_samples_draws_n_values(self):
        RNG.seed(7)
        samples = TestArg("x", rng_type=RNGInteger(0, 9)).generate_samples(3)

        RNG.seed(7)
        assert samples == [RNG.integer(0, 9) for _ in range(3)]

    def test_generate_samples_of_a_static_value(self):
        assert TestArg("x", value=4).generate_samples(3) == [4]

    def test_repr_lists_no_directed_values(self):
        assert repr(TestArg("x", rng_type=RNGInteger(0, 9))) == "TestArg(name='x', type=int)"

    def test_parameter_keeps_its_own_options(self):
        """The Parameter options that replace the TestArg ones stay."""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zero": (0,)},
            test_vectors={"nine": (9,)},
            always_include_directed=False,
        )

        assert param.generate_vectors(0, mode="directed_only") == [(0,)]
        assert param.generate_vectors(0, mode="test") == [(9,)]
        assert param.to_dict()["always_include_directed"] is False


class TestFactoryResults:
    """A factory must return a Parameter; a tuple gets the migration message."""

    def test_parameter_is_returned_as_it_is(self):
        param = Parameter(TestArg("x", value=1))

        assert check_factory_result("s", _tuple_factory, param) is param

    def test_tuple_factory_fails_naming_the_factory(self):
        with pytest.raises(ValueError) as excinfo:
            build_parametrization(
                "v4_tuple", _tuple_factory, takes_x, config=None, pytest_fixtures=set()
            )

        message = str(excinfo.value)
        where = f"{os.path.realpath(__file__)}:{_tuple_factory.__code__.co_firstlineno}"
        assert message.startswith(
            "Strategy 'v4_tuple' returned an (argnames, samples) tuple "
            f"(factory: {where}:_tuple_factory). "
            "Returning a tuple was deprecated in 3.0 and is no longer supported in 4.0: "
            "return a Parameter, with one TestArg per argument and fixed rows as "
            "directed_vectors"
        )
        assert "@pytest.mark.parametrize" in message

    def test_none_asks_whether_the_factory_forgot_to_return(self):
        with pytest.raises(ValueError) as excinfo:
            check_factory_result("s", _tuple_factory, None)

        assert str(excinfo.value) == (
            "Strategy 's' must return a Parameter, got NoneType "
            "(did the factory forget to return?)"
        )

    @pytest.mark.parametrize(
        "result, shown",
        [({"x": 1}, "dict"), ((1, 2, 3), "tuple"), ([("x",), [(1,)]], "list"), (5, "int")],
    )
    def test_any_other_result_names_its_type(self, result, shown):
        with pytest.raises(ValueError) as excinfo:
            check_factory_result("s", _tuple_factory, result)

        assert str(excinfo.value) == f"Strategy 's' must return a Parameter, got {shown}"

    def test_export_reports_a_tuple_factory_as_an_error(self):
        register("v4_export_tuple")(_tuple_factory)

        data = json.loads(export_strategies())

        assert list(data["v4_export_tuple"]) == ["error"]
        assert "returned an (argnames, samples) tuple" in data["v4_export_tuple"]["error"]
        assert "legacy_tuple" not in json.dumps(data)
