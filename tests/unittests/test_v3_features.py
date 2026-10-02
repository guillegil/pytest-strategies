"""Unit tests for the 3.0.0 public API and error messages."""

import pytest

import pytest_strategy
from pytest_strategy import (
    RNG,
    Parameter,
    PytestStrategiesWarning,
    RNGInteger,
    RNGValueError,
    Strategy,
    TestArg,
    register,
    strategy,
)
from pytest_strategy._registry import registry


@pytest.fixture
def clean_registry():
    saved = registry.snapshot()
    try:
        yield
    finally:
        registry.restore(saved)


class TestPublicApi:
    def test_plain_functions_are_the_class_aliases(self):
        assert pytest_strategy.strategy is Strategy.strategy is strategy
        assert pytest_strategy.register is Strategy.register is register

    def test_compatibility_module_imports(self):
        from pytest_strategy.strategy import PytestStrategiesWarning as Warning2
        from pytest_strategy.strategy import Strategy as Strategy2

        assert Strategy2 is Strategy
        assert Warning2 is PytestStrategiesWarning

    def test_import_as_gives_the_decorator(self):
        import pytest_strategy.strategy as m

        assert m is strategy

    def test_warning_filter_path_resolves_to_the_class(self):
        module = __import__("pytest_strategy.strategy", fromlist=["PytestStrategiesWarning"])

        assert module.PytestStrategiesWarning is PytestStrategiesWarning

    def test_strategy_only_marks_the_test(self):
        def test_x(x):
            pass

        marked = strategy("anything")(test_x)

        assert marked is test_x
        (mark,) = test_x.pytestmark
        assert mark.name == "strategy"
        assert mark.args == ("anything",)
        assert mark.kwargs == {"validate_signature": True}

    def test_strategy_stores_a_factory_instead_of_calling_it(self):
        def factory():
            raise AssertionError("called")

        def test_x(x):
            pass

        strategy(factory, validate_signature=False)(test_x)

        assert test_x.pytestmark[0].args == (factory,)
        assert test_x.pytestmark[0].kwargs == {"validate_signature": False}

    @pytest.mark.parametrize("bad", [None, 3])
    def test_strategy_rejects_other_references(self, bad):
        with pytest.raises(TypeError, match="strategy name or a factory"):
            strategy(bad)

    def test_register_rejects_a_non_string_name(self):
        with pytest.raises(TypeError, match="register\\(\\) takes a strategy name"):
            register(lambda nsamples: None)


class TestScopedRegistry:
    def test_nearest_directory_wins(self, clean_registry):
        top = compile("def f(nsamples): pass", "/virtual/tests/strategies.py", "exec")
        sub = compile("def f(nsamples): pass", "/virtual/tests/esm/strategies.py", "exec")
        top_ns, sub_ns = {}, {}
        exec(top, top_ns)
        exec(sub, sub_ns)
        registry.add("v3_scoped", top_ns["f"])
        registry.add("v3_scoped", sub_ns["f"])

        def lookup(path):
            import os

            found = registry.nearest("v3_scoped", os.path.normcase(os.path.realpath(path)))
            return found.factory if found else None

        assert lookup("/virtual/tests/esm") is sub_ns["f"]
        assert lookup("/virtual/tests/esm/deeper") is sub_ns["f"]
        assert lookup("/virtual/tests/dma") is top_ns["f"]
        assert lookup("/virtual/other") is None

    def test_registry_view_behaves_like_the_2x_dict(self, clean_registry):
        def factory(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        Strategy._registry["v3_view"] = factory

        assert "v3_view" in Strategy._registry
        assert Strategy._registry["v3_view"] is factory
        assert dict(Strategy._registry)["v3_view"] is factory
        del Strategy._registry["v3_view"]
        assert "v3_view" not in registry


class TestErrorMessages:
    def test_rng_value_error_is_a_value_error(self):
        assert issubclass(RNGValueError, ValueError)

    def test_duplicate_argument_names_are_rejected(self):
        with pytest.raises(RNGValueError, match="two test args named 'x'"):
            Parameter(TestArg("x", value=1), TestArg("x", value=2))

    def test_exhausted_retries_name_each_constraint(self):
        def below_zero(vector):
            return vector[0] < 0

        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda vector: vector[0] >= 5, below_zero],
            max_retries=40,
        )
        RNG.seed(3)

        with pytest.raises(ValueError) as excinfo:
            param.generate_vector()

        message = str(excinfo.value)
        assert "Could not generate valid vector after 40 attempts" in message
        lambda_count = int(message.split("constraint #0 (lambda) rejected ")[1].split(",")[0])
        named_count = int(message.split("below_zero (constraint #1) rejected ")[1].rstrip("."))
        assert lambda_count + named_count == 40
        assert lambda_count > 0 and named_count > 0

    def test_max_exhaustive_must_be_a_positive_int(self):
        with pytest.raises(ValueError, match="max_exhaustive must be None or an int >= 1"):
            Parameter(TestArg("x", value=1), max_exhaustive=0)
