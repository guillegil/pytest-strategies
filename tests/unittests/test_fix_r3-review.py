"""
Unit tests for the fixes from the final review before 2.0.0.

RNG bounds and weights, list vectors, an empty vector-name filter, decorated
factories, test IDs of dataclass and namedtuple values with sets, duplicate
registrations that were not reported, and strategy file discovery (unreadable
directories, conda environments, norecursedirs path patterns, symlinks and the
lowercase ``@strategy.register``).
"""

import functools
import json
import os
import random
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGFloat,
    RNGInteger,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Strategy,
    TestArg,
)
from pytest_strategy._resolver import call_factory
from pytest_strategy._runtime import runtime
from pytest_strategy.plugin import PytestStrategyPlugin
from pytest_strategy.rng import RNGEnum, RNGValueError
from pytest_strategy.strategy import PytestStrategiesWarning

SRC_DIR = Path(__file__).resolve().parents[2] / "src"

STRATEGY_SOURCE = """
from pytest_strategy import Parameter, Strategy, TestArg

@Strategy.register("r3_unit_strat")
def r3_unit_strat(nsamples):
    return Parameter(TestArg("x", value=1), nsamples=1)
"""


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what these tests change globally: the seed, random state and registry."""
    seed = RNG.get_seed()
    state = random.getstate()
    registry = dict(Strategy._registry)
    yield
    RNG._seed = seed
    random.setstate(state)
    Strategy._registry.clear()
    Strategy._registry.update(registry)


def _write(path, source=STRATEGY_SOURCE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


class TestFloatRanges:
    """RNGFloat and RNGWeightedFloat stay inside their bounds."""

    @pytest.mark.parametrize(
        "bounds", [(-sys.float_info.max, sys.float_info.max), (-1e308, 1e308), (-9e307, 9e307)]
    )
    def test_range_wider_than_the_largest_float_draws_finite_values(self, bounds):
        low, high = bounds
        RNG.seed(1)
        values = [RNGFloat(low, high).generate() for _ in range(2000)]
        values += [RNGWeightedFloat({bounds: 1.0}).generate() for _ in range(2000)]

        assert all(low <= v <= high for v in values)

    def test_normal_ranges_draw_what_random_uniform_draws(self):
        """The same seed gives the same values as before the fix."""
        stdlib = random.Random(7)
        expected = [stdlib.uniform(-3.5, 7.25) for _ in range(500)]
        RNG.seed(7)

        assert [RNG.float(-3.5, 7.25) for _ in range(500)] == expected

    @pytest.mark.parametrize(
        "bounds",
        [
            (float("-inf"), 0.0),
            (0.0, float("inf")),
            (float("inf"), float("inf")),
            (float("nan"), 1.0),
        ],
    )
    def test_non_finite_float_bounds_are_rejected(self, bounds):
        with pytest.raises(RNGValueError, match="finite"):
            RNGFloat(*bounds)
        with pytest.raises(RNGValueError, match="finite bounds"):
            RNGWeightedFloat({bounds: 1.0})


class TestWeightedRanges:
    """A weighted range must be a (min, max) tuple with min <= max."""

    def test_reversed_integer_range_is_rejected_at_construction(self):
        """It used to fail only for the seeds that drew that range."""
        with pytest.raises(RNGValueError, match=r"range \(200, 100\) must have min <= max"):
            RNGWeightedInteger({(0, 10): 0.95, (200, 100): 0.05})

    def test_reversed_float_range_is_rejected(self):
        with pytest.raises(RNGValueError, match="min <= max"):
            RNGWeightedFloat({(10.0, 0.0): 1.0})

    @pytest.mark.parametrize("key", [5, (1, 2, 3)])
    def test_key_that_is_not_a_pair_is_rejected(self, key):
        with pytest.raises(RNGValueError, match=r"must be a \(min, max\) tuple"):
            RNGWeightedInteger({key: 1.0})

    def test_equal_bounds_are_allowed(self):
        assert RNGWeightedInteger({(3, 3): 1.0}).generate() == 3


class TestWeightTotals:
    """Weights whose total overflows are rejected when the RNG type is built."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: RNGWeightedInteger({(0, 1): 1e308, (2, 3): 1e308}),
            lambda: RNGWeightedFloat({(0.0, 1.0): 1e308, (2.0, 3.0): 1e308}),
            lambda: RNGWeightedInteger({(0, 1): 10**308, (2, 3): 10**308}),
        ],
    )
    def test_total_that_overflows(self, make):
        with pytest.raises(RNGValueError, match="finite total"):
            make()

    def test_enum_weights_total_that_overflows(self):
        from enum import Enum

        class Color(Enum):
            RED = 1
            BLUE = 2

        with pytest.raises(RNGValueError, match="finite total"):
            RNGEnum(Color, weights={Color.RED: 1e308, Color.BLUE: 1e308})

    def test_int_weight_too_large_for_a_float(self):
        """math.isfinite raised OverflowError instead of RNGValueError."""
        with pytest.raises(RNGValueError, match="must be a finite number"):
            RNGWeightedInteger({(0, 1): 10**400})


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestListVectors:
    """A vector given as a list (e.g. from JSON or YAML) is stored as a tuple."""

    def _single(self, **kwargs):
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 10)), **kwargs)

    def test_directed_list_gives_the_element(self):
        param = self._single(directed_vectors={"five": [5]})

        assert param.generate_vectors(0, mode="directed_only") == [(5,)]

    def test_test_list_gives_the_element(self):
        param = self._single(test_vectors={"six": [6]})

        assert param.generate_vectors(0, mode="test") == [(6,)]

    def test_added_lists_are_stored_as_tuples(self):
        param = self._single()
        param.add_directed_vector("a", [1])
        param.add_test_vector("b", [2])

        assert param.directed_vectors == {"a": (1,)}
        assert param.test_vectors == {"b": (2,)}

    def test_pytest_param_vector_is_kept(self):
        """A tuple is not rebuilt, so a pytest.param keeps its marks and id."""
        vector = pytest.param(1, 2, 3, marks=pytest.mark.xfail, id="pp")
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            TestArg("z", rng_type=RNGInteger(0, 10)),
            directed_vectors={"pp": vector},
        )

        assert param.get_vector_by_name("pp") is vector


class TestEmptyVectorName:
    """Only None means "no filter"; an empty name is a name like any other."""

    def test_empty_name_that_no_vector_has_raises(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            directed_vectors={"zero": (0,), "max": (100,)},
        )

        with pytest.raises(KeyError):
            param.generate_vectors(10, filter_by_name="")

    def test_vector_named_empty_can_be_selected(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 100)),
            directed_vectors={"": (7,), "max": (100,)},
        )

        assert param.generate_vectors(10, filter_by_name="") == [(7,)]


# ---------------------------------------------------------------------------
# Factory calls
# ---------------------------------------------------------------------------


def _logged(fn):
    """A functools.wraps decorator whose wrapper only has *args/**kwargs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class TestDecoratedFactories:
    """The wrapped function's signature decides how a decorated factory is called."""

    def test_cached_zero_argument_factory(self):
        @functools.cache
        def factory():
            return "made"

        assert call_factory("s", factory, 10) == "made"

    def test_wrapped_zero_argument_factory(self):
        @_logged
        def factory():
            return "made"

        assert call_factory("s", factory, 10) == "made"

    def test_wrapped_positional_only_factory(self):
        @_logged
        def factory(n, /):
            return n

        assert call_factory("s", factory, 10) == 10

    def test_wrapped_ctx_only_factory_gets_the_hook_result(self, monkeypatch):
        monkeypatch.setattr(runtime, "strategy_context", lambda: {"channels": [1, 2]})

        @_logged
        def factory(ctx):
            return ctx

        assert call_factory("s", factory, 10) == {"channels": [1, 2]}

    def test_nsamples_is_not_passed_as_a_ctx_that_keeps_its_default(self, monkeypatch):
        monkeypatch.setattr(runtime, "strategy_context", lambda: None)

        @_logged
        def factory(ctx="default"):
            return ctx

        assert call_factory("s", factory, 10) == "default"

    def test_type_error_inside_a_wrapped_factory_calls_it_once(self):
        calls = []

        @_logged
        def factory(nsamples):
            calls.append(nsamples)
            raise TypeError("inside the factory")

        with pytest.raises(ValueError, match="TypeError: inside the factory"):
            call_factory("s", factory, 10)
        assert calls == [10]

    def test_args_only_wrapper_is_called_positionally(self):
        def outer(fn):
            @functools.wraps(fn)
            def wrapper(*args):
                return fn(*args)

            return wrapper

        @outer
        def factory(nsamples):
            return nsamples

        assert call_factory("s", factory, 3) == 3

    def test_mock_patch_still_gets_its_mock(self):
        from unittest import mock

        @mock.patch("os.getcwd", return_value="/patched")
        def factory(nsamples, getcwd=None):
            return nsamples, os.getcwd()

        assert call_factory("s", factory, 4) == (4, "/patched")


# ---------------------------------------------------------------------------
# Test IDs
# ---------------------------------------------------------------------------

IDS_SCRIPT = """
import json, sys
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple
sys.path.insert(0, sys.argv[1])
from pytest_strategy._ids import generate_test_ids

@dataclass(frozen=True)
class Board:
    name: str
    features: frozenset

class Pins(NamedTuple):
    names: frozenset

Legacy = namedtuple("Legacy", "tags")

values = [
    Board("a", frozenset({"uart", "spi", "i2c", "can", "usb"})),
    Pins(frozenset({"tx", "rx", "cts", "rts", "gnd"})),
    Legacy({"alpha", "beta", "gamma", "delta"}),
]
print(json.dumps([generate_test_ids(["v"], [(v,)])[0] for v in values]))
"""


class TestIdsOfValuesHoldingSets:
    """Dataclass and namedtuple values with a set field get the same ID in every process."""

    def _ids(self, hashseed):
        env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
        out = subprocess.run(
            [sys.executable, "-c", IDS_SCRIPT, str(SRC_DIR)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return json.loads(out)

    def test_same_ids_under_different_hash_seeds(self):
        ids = [self._ids(seed) for seed in (1, 2, 3)]

        assert ids[0] == ids[1] == ids[2]
        assert (
            ids[0][0]
            == "v=Board(name='a', features=frozenset({'can', 'i2c', 'spi', 'uart', 'usb'}))"
        )
        assert ids[0][2] == "v=Legacy(tags={'alpha', 'beta', 'delta', 'gamma'})"

    def test_values_without_sets_keep_their_repr(self):
        from dataclasses import dataclass

        from pytest_strategy._ids import _stable_repr

        @dataclass
        class Point:
            x: int
            label: str

        class Custom:
            def __repr__(self):
                return "Custom!"

        assert _stable_repr(Point(1, "a")) == repr(Point(1, "a"))
        assert _stable_repr(Custom()) == "Custom!"


# ---------------------------------------------------------------------------
# Duplicate registrations
# ---------------------------------------------------------------------------


def _exec(source, filename, namespace=None):
    exec(compile(textwrap.dedent(source), filename, "exec"), namespace if namespace else {})


class TestDuplicateRegistrationIsReported:
    """Clashes that looked like a re-run of the same function now warn."""

    def test_two_functions_of_the_same_name_in_one_file(self):
        source = """
            from pytest_strategy import Parameter, Strategy, TestArg

            @Strategy.register("r3_small")
            def factory(nsamples):
                return Parameter(TestArg("x", value=1), nsamples=1)

            @Strategy.register("r3_small")
            def factory(nsamples):
                return Parameter(TestArg("x", value=2), nsamples=1)
            """
        with pytest.warns(PytestStrategiesWarning, match=r"my_strategies\.py:8:factory replaces"):
            _exec(source, "/virtual/r3/my_strategies.py")

    def test_decorated_factories_in_two_files_name_those_files(self):
        source = """
            from pytest_strategy import Parameter, Strategy, TestArg

            @Strategy.register("r3_shared")
            @logged
            def factory(nsamples):
                return Parameter(TestArg("x", value=1), nsamples=1)
            """
        _exec(source, "/virtual/r3/a_strategies.py", {"logged": _logged})
        with pytest.warns(PytestStrategiesWarning) as record:
            _exec(source, "/virtual/r3/b_strategies.py", {"logged": _logged})

        message = str(record[0].message)
        assert "b_strategies.py:4:factory replaces" in message
        assert "a_strategies.py:4:factory" in message

    def test_cached_factories(self):
        @functools.cache
        def first(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        @functools.cache
        def second(nsamples):
            return Parameter(TestArg("x", value=2), nsamples=1)

        Strategy.register("r3_cached")(first)
        with pytest.warns(PytestStrategiesWarning, match="second replaces"):
            Strategy.register("r3_cached")(second)

    def test_partials_of_different_functions(self):
        def low(nsamples, bound):
            return Parameter(TestArg("x", value=bound), nsamples=1)

        def high(nsamples, bound):
            return Parameter(TestArg("x", value=bound), nsamples=1)

        Strategy.register("r3_partial")(functools.partial(low, bound=1))
        with pytest.warns(PytestStrategiesWarning, match="high replaces"):
            Strategy.register("r3_partial")(functools.partial(high, bound=2))

    def test_reexecuted_decorated_and_cached_factories_are_silent(self):
        source = """
            import functools
            from pytest_strategy import Parameter, Strategy, TestArg

            @Strategy.register("r3_again")
            @logged
            def factory(nsamples):
                return Parameter(TestArg("x", value=1), nsamples=1)

            @Strategy.register("r3_again_cached")
            @functools.lru_cache
            def cached(nsamples):
                return Parameter(TestArg("x", value=1), nsamples=1)

            @Strategy.register("r3_again_partial")
            def _partial_target(nsamples, bound=0):
                return Parameter(TestArg("x", value=bound), nsamples=1)

            Strategy.register("r3_again_partial")(functools.partial(_partial_target, bound=1))
            """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec(source, "/virtual/r3/again_strategies.py", {"logged": _logged})
            _exec(source, "/virtual/r3/again_strategies.py", {"logged": _logged})


# ---------------------------------------------------------------------------
# Strategy file discovery
# ---------------------------------------------------------------------------


class TestDiscoverySkips:
    """Discovery skips what pytest's collection skips, and never crashes on a directory."""

    def test_unreadable_directory_does_not_raise(self, tmp_path, monkeypatch):
        """Path.is_file raised PermissionError there before Python 3.14 (INTERNALERROR)."""
        project = tmp_path / "proj"
        strategy_file = _write(project / "tests" / "strategies.py")
        (project / "data" / "pg").mkdir(parents=True)
        real_stat = os.stat
        blocked = str(project / "data" / "pg") + os.sep

        def stat(path, *args, **kwargs):
            if str(path).startswith(blocked):
                raise PermissionError(13, "Permission denied", str(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", stat)

        assert PytestStrategyPlugin()._discover_strategy_files([project]) == [strategy_file]

    def test_conda_environment_is_skipped(self, tmp_path):
        project = tmp_path / "proj"
        strategy_file = _write(project / "strategies.py")
        (project / "env" / "conda-meta").mkdir(parents=True)
        (project / "env" / "conda-meta" / "history").write_text("")
        _write(project / "env" / "lib" / "site-packages" / "pytest_strategy" / "strategy.py")

        assert PytestStrategyPlugin()._discover_strategy_files([project]) == [strategy_file]

    @pytest.mark.parametrize("pattern", ["tests/data", "*/data", "data/*", "tests/data/sample"])
    def test_norecursedirs_path_patterns(self, tmp_path, pattern):
        project = tmp_path / "proj"
        strategy_file = _write(project / "tests" / "a_strategies.py")
        _write(project / "tests" / "data" / "sample" / "strategies.py")

        found = PytestStrategyPlugin()._discover_strategy_files([project], [pattern])

        assert found == [strategy_file]

    def test_name_pattern_does_not_match_a_parent_path(self, tmp_path):
        """A pattern without a separator only matches directory names, as in pytest."""
        project = tmp_path / "proj"
        strategy_file = _write(project / "data_tests" / "strategies.py")

        assert PytestStrategyPlugin()._discover_strategy_files([project], ["data"]) == [
            strategy_file
        ]

    def test_lowercase_register_is_not_imported(self, tmp_path):
        """A functools.singledispatch function named strategy is not a strategy file."""
        pricing = _write(
            tmp_path / "shop" / "pricing_strategy.py",
            "from functools import singledispatch\n\n"
            "@singledispatch\ndef strategy(x):\n    return x\n\n"
            "@strategy.register(int)\ndef _(x):\n    return x\n",
        )
        unimported = []

        found = PytestStrategyPlugin()._discover_strategy_files([tmp_path], (), unimported)

        assert found == []
        assert unimported == [pricing]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
class TestDiscoveryFollowsSymlinks:
    """Symlinked directories are searched, as pytest's collection does."""

    def _link(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:  # e.g. Windows without the privilege
            pytest.skip(f"cannot create a symlink: {e}")

    def test_strategy_file_in_a_symlinked_directory(self, tmp_path):
        shared = _write(tmp_path / "shared" / "shared_strategies.py")
        (tmp_path / "proj" / "tests").mkdir(parents=True)
        self._link(tmp_path / "proj" / "tests" / "shared", tmp_path / "shared")

        found = PytestStrategyPlugin()._discover_strategy_files([tmp_path / "proj" / "tests"])

        assert found == [tmp_path / "proj" / "tests" / "shared" / shared.name]

    def test_link_loop_ends_and_each_file_is_found_once(self, tmp_path):
        project = tmp_path / "proj"
        strategy_file = _write(project / "tests" / "strategies.py")
        self._link(project / "tests" / "loop", project)
        self._link(project / "again", project / "tests")

        found = PytestStrategyPlugin()._discover_strategy_files([project])

        assert [Path(os.path.realpath(p)) for p in found] == [Path(os.path.realpath(strategy_file))]


class TestLoadedModuleReuse:
    """An import of a loaded strategy file gets the module the plugin loaded."""

    def _config(self, rootpath):
        return SimpleNamespace(
            option=SimpleNamespace(verbose=0),
            rootpath=rootpath,
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        )

    def test_import_reuses_the_module(self, tmp_path, monkeypatch):
        source = STRATEGY_SOURCE + "\nclass Mode:\n    pass\n"
        strategy_file = _write(tmp_path / "r3_reuse_strategies.py", source)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "r3_reuse_strategies", raising=False)
        meta_path = list(sys.meta_path)
        runtime.push(None)
        try:
            PytestStrategyPlugin()._load_strategy_files([strategy_file], self._config(tmp_path))
            import r3_reuse_strategies

            loaded = runtime.current.strategy_modules[
                os.path.normcase(os.path.realpath(strategy_file))
            ]
            assert r3_reuse_strategies is loaded
            assert r3_reuse_strategies.__name__.startswith("pytest_strategies_discovered.")
            assert r3_reuse_strategies.__spec__.origin == str(strategy_file)
        finally:
            runtime.pop()
            sys.meta_path[:] = meta_path
            sys.modules.pop("r3_reuse_strategies", None)

    def test_another_file_of_the_same_name_is_imported_as_usual(self, tmp_path, monkeypatch):
        loaded_file = _write(tmp_path / "one" / "r3_same_strategies.py")
        _write(tmp_path / "two" / "r3_same_strategies.py", "VALUE = 'two'\n")
        monkeypatch.syspath_prepend(str(tmp_path / "two"))
        monkeypatch.delitem(sys.modules, "r3_same_strategies", raising=False)
        meta_path = list(sys.meta_path)
        runtime.push(None)
        try:
            PytestStrategyPlugin()._load_strategy_files([loaded_file], self._config(tmp_path))
            import r3_same_strategies

            assert r3_same_strategies.VALUE == "two"
        finally:
            runtime.pop()
            sys.meta_path[:] = meta_path
            sys.modules.pop("r3_same_strategies", None)


class TestAttributedWarningLocation:
    """Re-emitted strategy warnings point at the test module's current path."""

    def test_uses_the_module_file_not_a_stale_co_filename(self):
        from pytest_strategy._resolver import _attributed_warnings

        # A test compiled at its old location (a cached pyc after a checkout moved)
        namespace = {"__file__": "/new/checkout/test_moved.py"}
        exec(
            compile("def test_moved():\n    pass\n", "/old/checkout/test_moved.py", "exec"),
            namespace,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _attributed_warnings("r3_moved", namespace["test_moved"]):
                warnings.warn("combination skipped", PytestStrategiesWarning, stacklevel=1)

        assert [(w.filename, w.lineno) for w in caught] == [("/new/checkout/test_moved.py", 1)]
        assert str(caught[0].message).startswith("Strategy 'r3_moved' (test_moved): ")
