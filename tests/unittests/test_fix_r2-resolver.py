"""
Regression tests for the second round of resolver, introspection, dataclass and ID fixes.

Each test here failed before its fix, except those whose docstring says they guard
behaviour that already worked. Most drive resolve_and_parametrize directly with a
mocked pytest.Config and inspect the pytest.mark.parametrize it applies.
"""

import functools
import inspect
import json
import os
import random
import subprocess
import sys
import textwrap
import warnings
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest

import pytest_strategy
from pytest_strategy import RNG, RNGInteger, Strategy, StrategyOptions, _resolver
from pytest_strategy._dataclass import convert_to_dataclass
from pytest_strategy._factory import FactoryInputs, call_factory
from pytest_strategy._ids import generate_dataclass_ids, generate_test_ids
from pytest_strategy._introspection import detect_dataclass_param
from pytest_strategy._resolver import resolve_and_parametrize
from pytest_strategy.parameters import Parameter
from pytest_strategy.rng import Series
from pytest_strategy.strategy import PytestStrategiesWarning
from pytest_strategy.test_args import TestArg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what these tests change globally: the registry, the RNG seed and random state."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


def _make_config(rootpath=None, **options):
    """Return a mock pytest.Config serving the given CLI options (and rootpath, if given)."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    if rootpath is not None:
        config.rootpath = Path(rootpath)
    return config


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    _fn.__signature__ = inspect.Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


def _parametrize(factory, test_fn, *, validate=True, config=None):
    """Resolve ``factory`` for ``test_fn``; return the mark's (argnames, values, ids)."""
    marked = resolve_and_parametrize(
        "strat",
        test_fn,
        registry={"strat": factory},
        config=config if config is not None else _make_config(),
        pytest_fixtures=set(),
        validate=validate,
    )
    mark = marked.pytestmark[-1]
    return mark.args[0], list(mark.args[1]), mark.kwargs["ids"]


def _resolve(factory, argnames, **options):
    """Resolve ``factory`` for a test taking ``argnames`` under the given CLI options."""
    return _parametrize(
        factory, _make_test_fn(argnames), validate=False, config=_make_config(**options)
    )


def _with_nsamples(param, nsamples):
    """Set ``param.nsamples`` as ``Parameter(..., nsamples=nsamples)`` would.

    It is assigned after construction so that these tests exercise only the
    resolver, whatever Parameter's own validation of the value is.
    """
    param.nsamples = nsamples
    return param


def _series_param():
    """Two Series args (3x2 product) plus one directed vector."""
    return Parameter(
        TestArg("x", rng_type=Series([1, 2, 3])),
        TestArg("y", rng_type=Series(["a", "b"])),
        directed_vectors={"corner": (99, "z")},
    )


def _random_param():
    """A single random (non-sequence) arg plus one directed vector."""
    return Parameter(
        TestArg("code", rng_type=RNGInteger(200, 400)),
        directed_vectors={"corner": (500,)},
    )


PRODUCT = [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")]


# ---------------------------------------------------------------------------
# nsamples="auto" forwarded into Parameter.nsamples
# ---------------------------------------------------------------------------


class TestParameterNsamplesAuto:
    """A Parameter whose own nsamples is "auto" is exhaustive, or falls back to 10."""

    def test_passthrough_with_series_under_auto_is_exhaustive(self):
        """Guards behaviour that already worked in the resolver."""
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_series_param(), nsamples), ["x", "y"], nsamples="auto"
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_auto_default_without_cli_flag_is_exhaustive(self):
        """Guards behaviour that already worked in the resolver."""
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_series_param(), "auto"), ["x", "y"]
        )
        assert samples == [(99, "z")] + PRODUCT

    def test_random_only_passthrough_under_auto_falls_back_to_10(self):
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_random_param(), nsamples), ["code"], nsamples="auto"
        )
        assert samples[0] == 500
        assert len(samples) == 1 + 10
        assert all(200 <= s <= 400 for s in samples[1:])

    def test_random_only_auto_default_without_cli_flag_falls_back_to_10(self):
        _, samples, _ = _resolve(lambda nsamples: _with_nsamples(_random_param(), "auto"), ["code"])
        assert len(samples) == 1 + 10

    def test_int_param_nsamples_still_used_as_fallback(self):
        _, samples, _ = _resolve(
            lambda nsamples: _with_nsamples(_random_param(), 3), ["code"], nsamples="auto"
        )
        assert len(samples) == 1 + 3


# ---------------------------------------------------------------------------
# Decorated factories
# ---------------------------------------------------------------------------


def _call_factory(factory, nsamples=3):
    """Call ``factory`` as the resolver does, with ``nsamples`` as the run's count."""
    inputs = FactoryInputs(
        options=StrategyOptions(strategy="s", nsamples=nsamples),
        rng=RNG.generator(),
        ctx=lambda: None,
    )
    return call_factory("s", factory, inputs)


def _inject_rng(fn):
    """A functools.wraps decorator that injects an extra argument."""

    @functools.wraps(fn)
    def wrapper(nsamples):
        return fn(nsamples, "rng")

    return wrapper


def _adapt_zero_arg(fn):
    """A functools.wraps adapter that lets a zero-argument function take nsamples."""

    @functools.wraps(fn)
    def wrapper(nsamples):
        return fn()

    return wrapper


def _passthrough(fn):
    """A decorator without functools.wraps."""

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _wraps_passthrough(fn):
    """The usual functools.wraps decorator with *args/**kwargs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class TestDecoratedFactories:
    """The signature of the wrapper that is called decides what a decorated factory gets."""

    def test_wraps_decorator_injecting_an_argument(self):
        @_inject_rng
        def make(nsamples, rng):
            return nsamples, rng

        assert _call_factory(make) == (3, "rng")

    def test_mock_patch_decorator(self):
        # mock.patch passes its mocks to the first parameters
        @mock.patch("os.getcwd", return_value="/fake")
        def make(getcwd, nsamples):
            return nsamples, os.getcwd()

        assert _call_factory(make) == (3, "/fake")

    def test_wraps_adapter_around_zero_argument_function(self):
        @_adapt_zero_arg
        def make():
            return "called"

        assert _call_factory(make) == "called"

    def test_var_args_wrapper_without_wraps_around_zero_argument_factory(self):
        calls = []

        @_passthrough
        def make():
            calls.append("made")
            return "made"

        assert _call_factory(make) == "made"
        assert calls == ["made"]

    def test_var_args_wrapper_without_wraps_around_nsamples_factory(self):
        """The wrapper hides the signature, so the factory is called with no arguments."""
        calls = []

        @_passthrough
        def make(nsamples):
            calls.append(nsamples)
            return nsamples

        with pytest.raises(ValueError, match="@functools.wraps") as exc_info:
            _call_factory(make)
        assert "missing 1 required positional argument: 'nsamples'" in str(exc_info.value)
        assert calls == []

    def test_wraps_var_args_wrapper(self):
        @_wraps_passthrough
        def make(nsamples):
            return nsamples

        assert _call_factory(make) == 3

    def test_opaque_wrapper_reports_the_original_error(self):
        @_passthrough
        def make(a, b):
            return a, b

        with pytest.raises(ValueError, match="Error calling strategy factory 's'") as exc_info:
            _call_factory(make)

        assert "missing 2 required positional arguments: 'a' and 'b'" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_opaque_wrapper_non_type_error_is_not_retried(self):
        calls = []

        @_passthrough
        def make():
            calls.append("made")
            raise RuntimeError("boom")

        with pytest.raises(ValueError, match="RuntimeError: boom") as exc_info:
            _call_factory(make)
        assert "functools.wraps" not in str(exc_info.value)
        assert calls == ["made"]

    def test_informative_wrapper_is_still_called_once(self):
        """A wrapper that names nsamples keeps the single-call guarantee."""
        calls = []

        def make(nsamples, rng):
            calls.append(nsamples)
            return None + 1

        with pytest.raises(ValueError, match="unsupported operand"):
            _call_factory(_inject_rng(make))
        assert calls == [3]

    def test_resolver_uses_decorated_factory(self):
        # The value carries the nsamples the factory received, which must be the run's:
        # the row count alone comes from --nsamples, whatever the factory received
        @_inject_rng
        def make(nsamples, rng):
            return Parameter(TestArg("x", value=(nsamples, rng)))

        _, samples, _ = _resolve(make, ["x"], nsamples=2)
        assert samples == [(2, "rng"), (2, "rng")]

    def test_export_strategies_lists_decorated_factories(self):
        @Strategy.register("fix_r2_injected")
        @_inject_rng
        def injected(nsamples, rng):
            return Parameter(TestArg("x", value=rng), nsamples=1)

        @Strategy.register("fix_r2_patched")
        @mock.patch("os.getcwd", return_value="/fake")
        def patched(getcwd, nsamples):
            return Parameter(TestArg("x", value=os.getcwd()), nsamples=1)

        @Strategy.register("fix_r2_adapted")
        @_adapt_zero_arg
        def adapted():
            return Parameter(TestArg("x", value=1), nsamples=1)

        @Strategy.register("fix_r2_opaque")
        @_passthrough
        def opaque():
            return Parameter(TestArg("x", value=1), nsamples=1)

        data = json.loads(Strategy.export_strategies())

        for name in ("fix_r2_injected", "fix_r2_patched", "fix_r2_adapted", "fix_r2_opaque"):
            assert [arg["name"] for arg in data[name]["arguments"]] == ["x"]


# ---------------------------------------------------------------------------
# Duplicate registration: one file reached through different path strings
# ---------------------------------------------------------------------------

STRATEGY_SOURCE = textwrap.dedent("""
    from pytest_strategy import Parameter, Strategy, TestArg

    @Strategy.register("fix_r2_dup")
    def factory(nsamples):
        return Parameter(TestArg("x", value=1), nsamples=1)
    """)


def _exec_strategy(filename, **module_globals):
    """Execute STRATEGY_SOURCE as if loaded from ``filename``."""
    exec(compile(STRATEGY_SOURCE, str(filename), "exec"), module_globals)


class TestDuplicateRegistrationPaths:
    """The same file is recognized whatever path string it was executed under."""

    def test_parent_directory_segments_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec_strategy("/virtual/proj/../shared/strategies.py")
            _exec_strategy("/virtual/shared/strategies.py")

    def test_symlinked_path_is_silent(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        (real / "strategies.py").write_text(STRATEGY_SOURCE)
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not supported here")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec_strategy(real / "strategies.py")
            _exec_strategy(link / "strategies.py")

    def test_relative_and_absolute_paths_are_silent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec_strategy("strategies.py")
            _exec_strategy(tmp_path / "strategies.py")

    def test_module_file_is_preferred_over_a_stale_code_filename(self):
        """A rewritten pyc cached before the checkout moved keeps the old co_filename."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec_strategy("/old/checkout/strategies.py", __file__="/new/checkout/strategies.py")
            _exec_strategy("/new/checkout/strategies.py", __file__="/new/checkout/strategies.py")

    def test_another_file_in_the_same_folder_still_warns(self):
        _exec_strategy("/virtual/users/strategies.py")
        with pytest.warns(PytestStrategiesWarning, match="more_strategies.py"):
            _exec_strategy("/virtual/users/more_strategies.py")

    def test_another_folder_registers_its_own_strategy_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _exec_strategy("/virtual/users/strategies.py")
            _exec_strategy("/virtual/billing/strategies.py")

    def test_warning_as_error_still_registers_the_new_factory(self):
        @Strategy.register("fix_r2_dup_error")
        def first(nsamples):
            return Parameter(TestArg("x", value=1), nsamples=1)

        def second(nsamples):
            return Parameter(TestArg("y", value=2), nsamples=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PytestStrategiesWarning):
                Strategy.register("fix_r2_dup_error")(second)

        assert Strategy._registry["fix_r2_dup_error"] is second


# ---------------------------------------------------------------------------
# Dataclass detection without signature validation
# ---------------------------------------------------------------------------


@dataclass
class Server:
    host: str
    port: int


@dataclass
class StartedServer:
    host: str
    port: int
    started: bool = False


@dataclass
class Point:
    x: int
    y: int


def _one_server(nsamples):
    """A strategy with the single row host="localhost", port=8000."""
    return Parameter(TestArg("host", value="localhost"), TestArg("port", value=8000), nsamples=1)


class TestDataclassTypedFixture:
    """With allow_fixtures=False, a dataclass parameter next to others is a fixture."""

    def test_exact_match_next_to_fixture_is_left_alone(self):
        def test_fn(server: Server, client):
            pass

        assert detect_dataclass_param(test_fn, ["host", "port"], allow_fixtures=False) == (
            False,
            None,
            None,
        )

    def test_field_mismatch_next_to_fixture_is_left_alone(self):
        def test_fn(server: StartedServer, client):
            pass

        assert detect_dataclass_param(test_fn, ["host", "port"], allow_fixtures=False) == (
            False,
            None,
            None,
        )

    def test_default_still_allows_fixtures(self):
        def test_fn(server: Server, client):
            pass

        assert detect_dataclass_param(test_fn, ["host", "port"]) == (True, Server, "server")

    def test_only_parameter_is_still_dataclass_mode(self):
        def test_fn(p: Point):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"], allow_fixtures=False) == (
            True,
            Point,
            "p",
        )

    def test_self_and_builtin_fixtures_do_not_count(self):
        class TestPoints:
            def test_point(self, p: Point, tmp_path):
                pass

        assert detect_dataclass_param(TestPoints.test_point, ["x", "y"], allow_fixtures=False) == (
            True,
            Point,
            "p",
        )

    def test_resolver_parametrizes_the_argnames_without_validation(self):
        def test_server(server: Server, client):
            pass

        argstr, samples, _ = _parametrize(_one_server, test_server, validate=False)
        assert argstr == "host,port"
        assert samples == [("localhost", 8000)]

    def test_resolver_keeps_dataclass_mode_next_to_fixture_with_validation(self):
        def test_server(server: Server, client):
            pass

        argstr, samples, _ = _parametrize(_one_server, test_server)
        assert argstr == "server"
        assert samples == [Server("localhost", 8000)]


# ---------------------------------------------------------------------------
# Test IDs: strings containing " at 0x", and sets
# ---------------------------------------------------------------------------


class Codec:
    """A value whose repr is the default ``<... object at 0x...>``."""

    def encode(self):
        pass


@dataclass
class Fault:
    msg: str
    code: int


class TestIdsOfStringsWithAddresses:
    """Only default object reprs collapse to the type name, never string data."""

    def test_single_string(self):
        assert generate_test_ids(["line"], [("segfault at 0x0",), ("jump at 0x401000",)]) == [
            "line='segfault at 0x0'",
            "line='jump at 0x401000'",
        ]

    def test_string_shaped_like_a_default_repr(self):
        assert generate_test_ids(["s"], [("<x at 0x10>",)]) == ["s='<x at 0x10>'"]

    def test_multi_arg_string(self):
        assert generate_test_ids(["msg", "code"], [("fault at 0x10", 1)]) == [
            "msg='fault at 0x10',code=1"
        ]

    def test_bytes_and_containers_of_strings(self):
        assert generate_test_ids(["b"], [(b"read at 0x10",)]) == ["b=b'read at 0x10'"]
        assert generate_test_ids(["regs"], [(["r0 at 0x0"],)]) == ["regs=['r0 at 0x0']"]

    def test_dataclass_field(self):
        assert generate_dataclass_ids([Fault("fault at 0x10", 1)], Fault) == [
            "msg='fault at 0x10',code=1"
        ]

    def test_default_reprs_still_use_the_type_name(self):
        assert generate_test_ids(["c"], [(Codec(),)]) == ["c=Codec"]
        assert generate_test_ids(["m"], [(Codec().encode,)]) == ["m=method"]
        assert generate_test_ids(["cs"], [([Codec()],)]) == ["cs=list"]


class TestIdsOfSets:
    """Set elements are shown in a deterministic order."""

    def test_set_of_strings_is_sorted(self):
        assert generate_test_ids(["perms"], [({"write", "read", "admin"},)]) == [
            "perms={'admin', 'read', 'write'}"
        ]

    def test_frozenset_of_strings_is_sorted(self):
        assert generate_test_ids(["perms"], [(frozenset({"write", "read"}),)]) == [
            "perms=frozenset({'read', 'write'})"
        ]

    def test_unsortable_elements_are_sorted_by_repr(self):
        assert generate_test_ids(["v"], [({"b", 1, "a"},)]) == ["v={'a', 'b', 1}"]

    def test_nested_sets_are_sorted(self):
        assert generate_test_ids(["v"], [(({"b", "a"}, [frozenset({"d", "c"})]),)]) == [
            "v=({'a', 'b'}, [frozenset({'c', 'd'})])"
        ]

    def test_other_values_keep_their_repr(self):
        values = [set(), frozenset(), {3, 1, 2}, (1,), (), [1, "a"], {"k": 1}, 1.5]
        assert generate_test_ids(["v"], [(v,) for v in values]) == [f"v={v!r}" for v in values]

    def test_ids_do_not_depend_on_the_hash_seed(self):
        code = (
            "from pytest_strategy._ids import generate_dataclass_ids, generate_test_ids\n"
            "rows = [({'read', 'write', 'admin', 'exec'}, frozenset({'x', 'y', 'z'}))]\n"
            "print(generate_test_ids(['perms', 'more'], rows))\n"
            "print(generate_test_ids(['perms'], [({'alpha', 'beta', 'gamma', 'delta'},)]))\n"
        )
        package_root = str(Path(pytest_strategy.__file__).resolve().parents[1])
        python_path = os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONPATH": python_path, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for hash_seed in ("1", "2", "3", "4")
        }
        assert len(outputs) == 1


# ---------------------------------------------------------------------------
# `p: DC = None` (Python 3.10's get_type_hints wrapped it in Optional)
# ---------------------------------------------------------------------------


class TestNoneDefaultDataclassParam:
    """``p: DC = None`` is detected whether or not get_type_hints adds Optional."""

    def test_none_default(self):
        def test_fn(p: Point = None):
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")

    def test_explicit_optional_is_not_dataclass_mode(self):
        """An Optional written by the user is kept."""

        def test_fn(p: Optional[Point] = None):  # noqa: UP045
            pass

        assert detect_dataclass_param(test_fn, ["x", "y"]) == (False, None, None)


# ---------------------------------------------------------------------------
# Dataclasses with a hand-written __init__
# ---------------------------------------------------------------------------


@dataclass
class Rect:
    width: int
    height: int

    def __init__(self, w, h, /):
        self.width = w
        self.height = h


@dataclass
class RenamedRect:
    width: int
    height: int

    def __init__(self, w, h):
        self.width = w
        self.height = h


@dataclass
class WithInitVar:
    a: int
    scale: InitVar[int] = 100
    b: int = 0

    def __post_init__(self, scale):
        self.scaled = self.a * scale


class TestConvertCustomInit:
    """A hand-written __init__ gets the values positionally, in field order."""

    @pytest.mark.parametrize("dc_type", [Rect, RenamedRect])
    def test_custom_init(self, dc_type):
        result = convert_to_dataclass([(1, 2), (3, 4)], ["width", "height"], dc_type)
        assert [(r.width, r.height) for r in result] == [(1, 2), (3, 4)]

    def test_custom_init_with_reordered_argnames(self):
        result = convert_to_dataclass([(2, 1)], ["height", "width"], Rect)
        assert (result[0].width, result[0].height) == (1, 2)

    def test_generated_init_with_init_var_still_uses_keywords(self):
        """Guards behaviour that already worked: positional values would fill the InitVar."""
        result = convert_to_dataclass([(1, 2)], ["a", "b"], WithInitVar)
        assert (result[0].a, result[0].b, result[0].scaled) == (1, 2, 100)

    def test_resolver_builds_custom_init_dataclass(self):
        def test_rect(r: Rect):
            pass

        _, samples, ids = _parametrize(
            lambda nsamples: Parameter(
                TestArg("width", value=1), TestArg("height", value=2), nsamples=1
            ),
            test_rect,
        )
        assert [(r.width, r.height) for r in samples] == [(1, 2)]
        assert ids == ["width=1,height=2"]


# ---------------------------------------------------------------------------
# pytest.param samples in dataclass mode
# ---------------------------------------------------------------------------


class TestDataclassModePytestParam:
    """A pytest.param sample is converted from its values and keeps its marks and id."""

    def test_values_marks_and_id_are_kept(self):
        slow = pytest.mark.slow

        def test_point(p: Point):
            pass

        argstr, samples, ids = _parametrize(
            lambda nsamples: Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                TestArg("y", rng_type=RNGInteger(0, 9)),
                directed_vectors={
                    "plain": (1, 2),
                    "slow": pytest.param(3, 4, marks=slow),
                    "custom": pytest.param(5, 6, id="custom"),
                },
                nsamples=0,
            ),
            test_point,
        )

        assert argstr == "p"
        assert samples[0] == Point(1, 2)
        assert samples[1] == pytest.param(Point(3, 4), marks=slow)
        assert samples[2] == pytest.param(Point(5, 6), id="custom")
        assert ids == ["x=1,y=2", "x=3,y=4", "x=5,y=6"]

    def test_parameter_rows_are_still_converted(self):
        """Guards behaviour that already worked for rows that are not pytest.param."""

        def test_point(p: Point):
            pass

        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            TestArg("y", rng_type=RNGInteger(0, 9)),
            directed_vectors={"origin": (0, 0)},
        )
        _, samples, ids = _parametrize(
            lambda nsamples: param,
            test_point,
            config=_make_config(vector_mode="directed_only"),
        )
        assert samples == [Point(0, 0)]
        assert ids == ["x=0,y=0"]


# ---------------------------------------------------------------------------
# Per-test random stream: keyed by rootdir-relative file path, not __module__
# ---------------------------------------------------------------------------


def _random_values(test_fn, rootpath):
    """Values a 3-sample random strategy gives ``test_fn`` with seed 7."""
    RNG.seed(7)
    _, samples, _ = _parametrize(
        lambda nsamples: Parameter(TestArg("v", rng_type=RNGInteger(0, 10**9)), nsamples=3),
        test_fn,
        validate=False,
        config=_make_config(rootpath=rootpath),
    )
    return samples


def _module_function(module_name, filename):
    """Return ``test_values`` defined in a fresh module named ``module_name`` at ``filename``."""
    namespace = {"__name__": module_name, "__file__": str(filename)}
    exec(compile("def test_values(v):\n    pass\n", str(filename), "exec"), namespace)
    return namespace["test_values"]


class TestStreamKeyIgnoresImportMode:
    """The same test file and qualname give the same values under any module name."""

    def test_location_is_rootdir_relative_posix_path(self, tmp_path):
        test_fn = _module_function("tests.sub.test_b", tmp_path / "tests" / "sub" / "test_b.py")
        config = SimpleNamespace(rootpath=tmp_path)
        assert _resolver._test_location(test_fn, config) == "tests/sub/test_b.py"

    def test_location_falls_back_to_module_without_config(self, tmp_path):
        test_fn = _module_function("tests.sub.test_b", tmp_path / "test_b.py")
        assert _resolver._test_location(test_fn, None) == "tests.sub.test_b"

    def test_location_falls_back_to_module_outside_rootdir(self, tmp_path):
        test_fn = _module_function("test_b", tmp_path / "elsewhere" / "test_b.py")
        config = SimpleNamespace(rootpath=tmp_path / "root")
        assert _resolver._test_location(test_fn, config) == "test_b"

    def test_prepend_and_importlib_module_names_give_the_same_values(self, tmp_path):
        filename = tmp_path / "tests" / "sub" / "test_b.py"
        prepend = _module_function("test_b", filename)
        importlib_mode = _module_function("tests.sub.test_b", filename)

        assert _random_values(prepend, tmp_path) == _random_values(importlib_mode, tmp_path)

    def test_different_files_still_get_different_values(self, tmp_path):
        first = _module_function("test_b", tmp_path / "a" / "test_b.py")
        second = _module_function("test_b", tmp_path / "b" / "test_b.py")

        assert _random_values(first, tmp_path) != _random_values(second, tmp_path)
