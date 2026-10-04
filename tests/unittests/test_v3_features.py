"""Unit tests for the 3.0.0 public API and error messages."""

import re

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

    def test_a_registration_keeps_the_files_whose_calls_made_it(self, clean_registry):
        import os

        def key(path):
            return os.path.normcase(os.path.realpath(path))

        namespace = {}
        exec(compile("def f(nsamples): pass", "/virtual/tests/strategies.py", "exec"), namespace)
        call = "register('v3_callers')(f)"
        for file in ("/virtual/tests/strategies.py", "/virtual/tests/b/strategies.py", "<string>"):
            exec(compile(call, file, "exec"), {"register": register, "f": namespace["f"]})

        (registration,) = registry.registrations("v3_callers")
        # The same function registered again by another file adds that file; code
        # without a file adds none
        assert registration.callers == {
            key("/virtual/tests/strategies.py"),
            key("/virtual/tests/b/strategies.py"),
        }

    def test_the_registry_view_records_its_caller(self, clean_registry):
        import os

        def factory(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        Strategy._registry["v3_view_caller"] = factory

        (registration,) = registry.registrations("v3_view_caller")
        assert registration.callers == {os.path.normcase(os.path.realpath(__file__))}

    def test_nearest_counts_only_the_registrations_it_accepts(self, clean_registry):
        import os

        top = compile("def f(nsamples): pass", "/virtual/tests/strategies.py", "exec")
        sub = compile("def f(nsamples): pass", "/virtual/tests/esm/strategies.py", "exec")
        top_ns, sub_ns = {}, {}
        exec(top, top_ns)
        exec(sub, sub_ns)
        registry.add("v3_accept", top_ns["f"])
        registry.add("v3_accept", sub_ns["f"])
        directory = os.path.normcase(os.path.realpath("/virtual/tests/esm"))

        found = registry.nearest("v3_accept", directory, lambda r: r.factory is not sub_ns["f"])

        assert found is not None and found.factory is top_ns["f"]
        assert registry.nearest("v3_accept", directory, lambda r: False) is None


class TestErrorMessages:
    def test_rng_value_error_is_a_value_error(self):
        assert issubclass(RNGValueError, ValueError)

    def test_duplicate_argument_names_are_rejected(self):
        with pytest.raises(RNGValueError, match="two test args named 'x'"):
            Parameter(TestArg("x", value=1), TestArg("x", value=2))

    def test_exhausted_retries_name_each_constraint(self):
        """Since 4.0 the counts are by constraint name, with the first row each rejected."""

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
        line = param.vector_constraints["constraint_0"].__code__.co_firstlineno
        parsed = re.fullmatch(
            r"Could not generate a random row after max_retries=40 draws\. "
            r"Rejected by \(first failing constraint per draw\): "
            r"constraint_0=(\d+), below_zero=(\d+)\. "
            rf"First rows rejected: constraint_0 \(lambda at test_v3_features\.py:{line}\): "
            r"Vector\(x=[0-4]\); below_zero: Vector\(x=[5-9]\)\. "
            r"Raise Parameter\(max_retries=\.\.\.\) or relax a constraint\.",
            message,
        )
        assert parsed is not None, message
        lambda_count, named_count = map(int, parsed.groups())
        assert lambda_count + named_count == 40
        assert lambda_count > 0 and named_count > 0

    def test_max_exhaustive_must_be_a_positive_int(self):
        with pytest.raises(ValueError, match="max_exhaustive must be None or an int >= 1"):
            Parameter(TestArg("x", value=1), max_exhaustive=0)
