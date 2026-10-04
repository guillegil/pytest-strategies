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

    def key(self, path):
        """A path as caller_file and _factory_origin give it."""
        import os

        return os.path.normcase(os.path.realpath(path))

    def run_module(self, monkeypatch, name, source, file="/virtual/tests/factories.py"):
        """Run ``source`` as the module ``name`` of the file ``file``."""
        import sys
        import types

        module = types.ModuleType(name)
        module.__file__ = file
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, file, "exec"), vars(module))
        return module

    def test_a_registration_keeps_its_own_files_calls(self, clean_registry):
        from pytest_strategy._registry import _factory_origin

        namespace = {"register": register}
        source = "def f(nsamples): pass\n\nregister('v3_calls')(f)\n"
        exec(compile(source, "/virtual/tests/strategies.py", "exec"), namespace)
        f = namespace["f"]
        call = "register('v3_calls')(f)\nregister('v3_alias')(f)\n"
        for file in ("/virtual/tests/b/strategies.py", "<string>"):
            exec(compile(call, file, "exec"), {"register": register, "f": f})

        # Registered again by another file and by code without a file: only the
        # call of the file that defines it is kept
        (registration,) = registry.registrations("v3_calls")
        assert registration.own_calls == ((_factory_origin(f), f),)
        (alias,) = registry.registrations("v3_alias")
        assert alias.own_calls == ()
        assert registry.own_names(f) == ["v3_calls"]

    def test_the_registry_view_records_its_caller(self, clean_registry):
        namespace = {"view": Strategy._registry}
        source = "def f(nsamples): pass\n\nview['v3_view_caller'] = f\n"
        exec(compile(source, "/virtual/tests/conftest.py", "exec"), namespace)
        # Set by this test function, which no module's import runs
        Strategy._registry["v3_view_late"] = namespace["f"]

        assert registry.own_names(namespace["f"]) == ["v3_view_caller"]

    def test_a_call_in_a_helper_counts_for_the_module_whose_import_ran_it(self, clean_registry):
        helper = {"register": register}
        source = "def f(nsamples): pass\n\ndef register_as(name):\n    return register(name)(f)\n"
        exec(compile(source, "/virtual/tests/common.py", "exec"), helper)
        exec(compile("register_as('v3_helper')", "/virtual/tests/a/test_a.py", "exec"), helper)
        exec(compile("register_as('v3_helper_own')", "/virtual/tests/common.py", "exec"), helper)
        # Called from this test function, not by a module's import: the module that
        # started the process, never the helper's file
        helper["register_as"]("v3_helper_late")

        for name in ("v3_helper", "v3_helper_late"):
            (registration,) = registry.registrations(name)
            assert registration.own_calls == ()
        assert registry.own_names(helper["f"]) == ["v3_helper_own"]

    def test_own_names_are_the_names_its_file_registers_it_under(self, clean_registry, monkeypatch):
        source = (
            "from pytest_strategy import register\n\n"
            "@register('v3_own_z')\n"
            "def burst(nsamples):\n    pass\n\n"
            "register('v3_own_a')(burst)\n"
        )
        module = self.run_module(monkeypatch, "tests.factories", source)
        # Another file registers it under another name, and under one of its names
        alias = "register('v3_own_0')(burst)\nregister('v3_own_z')(burst)\n"
        exec(
            compile(alias, "/virtual/tests/b/strategies.py", "exec"),
            {"register": register, "burst": module.burst},
        )

        assert registry.own_names(module.burst) == ["v3_own_a", "v3_own_z"]
        (registration,) = registry.registrations("v3_own_z")
        assert [registered for _, registered in registration.own_calls] == [module.burst]

    @pytest.mark.parametrize("caller", ["/virtual/tests/other.py", "/virtual/tests/a/conftest.py"])
    def test_another_factory_that_replaces_its_registration_takes_no_name_away(
        self, clean_registry, monkeypatch, caller
    ):
        # Another factory of the same folder registered under the name (a clash),
        # by its own file or by a conftest.py that only some runs load before the
        # session starts, where the clash only warns
        source = "from pytest_strategy import register\n\n@register('v3_taken')\ndef burst(nsamples):\n    pass\n"
        module = self.run_module(monkeypatch, "tests.factories", source)
        other = self.run_module(
            monkeypatch,
            "tests.other",
            "def other(nsamples):\n    pass\n",
            "/virtual/tests/other.py",
        )
        registry.add("v3_taken", other.other, self.key(caller))

        (registration,) = registry.registrations("v3_taken")
        assert registration.factory is other.other
        assert registry.own_names(module.burst) == ["v3_taken"]
        own = caller == "/virtual/tests/other.py"
        assert registry.own_names(other.other) == (["v3_taken"] if own else [])

    def test_a_method_bound_again_to_the_same_object_is_the_same_factory(
        self, clean_registry, monkeypatch
    ):
        source = (
            "from pytest_strategy import register\n\n"
            "class Burst:\n    def make(self, nsamples):\n        pass\n\n"
            "small, big = Burst(), Burst()\n"
            "register('v3_small')(small.make)\n"
            "register('v3_big')(big.make)\n"
        )
        module = self.run_module(monkeypatch, "tests.bursts", source, "/virtual/tests/bursts.py")

        assert module.small.make is not module.small.make
        assert registry.own_names(module.small.make) == ["v3_small"]
        assert registry.own_names(module.big.make) == ["v3_big"]
        # The method of an object that no register() call got: the names of the
        # objects of its origin
        assert registry.own_names(module.Burst().make) == ["v3_big", "v3_small"]

    def test_a_wrapper_counts_as_the_function_it_wraps(self, clean_registry, monkeypatch):
        deco = self.run_module(
            monkeypatch,
            "v3_deco",
            (
                "import functools\n\n"
                "def logged(fn):\n"
                "    @functools.wraps(fn)\n"
                "    def wrapper(*args, **kwargs):\n"
                "        return fn(*args, **kwargs)\n\n"
                "    return wrapper\n"
            ),
            "/virtual/tests/deco.py",
        )
        source = (
            "from pytest_strategy import register\n"
            "from v3_deco import logged\n\n"
            "@register('v3_logged')\n"
            "@logged\n"
            "def burst(nsamples):\n    pass\n\n"
            "def other(nsamples):\n    pass\n\n"
            "wrapped = register('v3_wrapped')(logged(other))\n"
        )
        module = self.run_module(monkeypatch, "tests.factories", source)

        assert registry.own_names(module.burst) == ["v3_logged"]
        assert registry.own_names(module.burst.__wrapped__) == ["v3_logged"]
        assert registry.own_names(module.wrapped) == ["v3_wrapped"]
        assert registry.own_names(module.other) == ["v3_wrapped"]
        # A wrapper made by another file (this one)
        assert registry.own_names(deco.logged(module.other)) == ["v3_wrapped"]

    def test_a_file_run_again_in_another_module_keeps_each_objects_names(
        self, clean_registry, monkeypatch
    ):
        # tests/factories.py imported as tests.factories and as factories: the
        # second module's register() calls replace the first's registrations
        # without a clash, and the calls of both stay
        source = (
            "import functools\n\n"
            "from pytest_strategy import register\n\n"
            "@register('v3_copy')\n"
            "def burst(nsamples):\n    pass\n\n"
            "def make(n):\n    def f(nsamples):\n        pass\n\n    return f\n\n"
            "class Maker:\n    def __call__(self, nsamples):\n        pass\n\n"
            "small = register('v3_copy_small')(make(1))\n"
            "big = register('v3_copy_big')(make(2))\n"
            "half = register('v3_copy_partial')(functools.partial(make, 2))\n"
            "maker = register('v3_copy_object')(Maker())\n"
        )
        first = self.run_module(monkeypatch, "tests.factories", source)
        second = self.run_module(monkeypatch, "factories", source)

        (registration,) = registry.registrations("v3_copy")
        assert registration.factory is second.burst
        for module in (first, second):
            assert registry.own_names(module.burst) == ["v3_copy"]
            # Not "v3_copy_big", the first name of an object of its origin
            assert registry.own_names(module.small) == ["v3_copy_small"]
            assert registry.own_names(module.big) == ["v3_copy_big"]
            assert registry.own_names(module.half) == ["v3_copy_partial"]
            assert registry.own_names(module.maker) == ["v3_copy_object"]
        # Objects that no register() call got: the names of those of their origin
        assert registry.own_names(first.make(3)) == ["v3_copy_big", "v3_copy_small"]
        assert registry.own_names(first.Maker()) == ["v3_copy_object"]

    def test_code_without_a_file_names_nothing(self, clean_registry):
        first, second = {}, {}
        for namespace in (first, second):
            exec(compile("def f(nsamples): pass", "<string>", "exec"), namespace)
        exec(
            compile("register('v3_no_file')(f)", "<string>", "exec"),
            {"register": register, **first},
        )

        assert registry.own_names(first["f"]) == []
        assert registry.own_names(second["f"]) == []


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
