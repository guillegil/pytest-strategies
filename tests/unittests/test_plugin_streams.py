"""
Tests for the plugin's random streams other than the rows (streams v1, D5): the
stream a block runs on (``_Stream``), ``RNG.refresh_seed(key)``, a session's save and
restore of the ambient generator, the factory's stream T/"factory" and a strategy
file's stream. Each one reseeds the ambient generator in place from its key and
puts back its state, the installed generator and the seed when it ends.

The fixture, test phase, test module and export streams run inside pytest; their
tests are in tests/integration/test_plugin_streams_integration.py.
"""

import dataclasses
import random
from types import SimpleNamespace

import pytest

from pytest_strategy import RNG, Parameter, RNGInteger, TestArg
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy._streams import StreamKey
from pytest_strategy.plugin import PytestStrategyPlugin
from pytest_strategy.rng import _Stream

KEY = StreamKey.root(7, "body", "tests/test_x.py::test_x", "call")
OTHER = StreamKey.root(7, "fixture", "", "tb", 0)


def first_draws(key, n=3):
    """The first n random() draws of the stream of ``key``."""
    generator = random.Random(key.seed_int())
    return [generator.random() for _ in range(n)]


@dataclasses.dataclass(frozen=True)
class FrozenError(Exception):
    code: int


# ---------------------------------------------------------------------------
# The stream of a block
# ---------------------------------------------------------------------------


class TestTheStream:
    def test_draws_come_from_the_key(self):
        with _Stream(KEY) as rng:
            assert rng is RNG.generator() is RNG._ambient
            drawn = [RNG.generator().random() for _ in range(3)]

        assert drawn == first_draws(KEY)

    def test_the_helpers_draw_from_it(self):
        with _Stream(KEY):
            drawn = RNG.integer(0, 10**9)

        assert drawn == random.Random(KEY.seed_int()).randint(0, 10**9)

    def test_the_ambient_generator_is_left_where_it_was(self):
        RNG.seed(3)
        RNG.generator().random()
        with _Stream(KEY):
            RNG.generator().random()
        after = [RNG.generator().random() for _ in range(2)]

        generator = random.Random(3)
        assert after == [generator.random() for _ in range(3)][1:]

    def test_a_reseed_in_the_block_changes_only_the_rest_of_the_block(self):
        RNG.seed(3)
        with _Stream(KEY):
            RNG.seed(5)
            assert RNG.get_seed() == 5
            assert RNG.generator().random() == random.Random(5).random()

        assert RNG.get_seed() == 3
        assert RNG.generator().random() == random.Random(3).random()

    def test_a_generator_kept_from_before_draws_from_the_stream(self):
        kept = RNG.generator()
        with _Stream(KEY):
            drawn = kept.random()

        assert drawn == first_draws(KEY, 1)[0]

    def test_streams_nest(self):
        with _Stream(KEY):
            outer = [RNG.generator().random()]
            with _Stream(OTHER):
                inner = [RNG.generator().random() for _ in range(2)]
            outer.append(RNG.generator().random())

        assert outer == first_draws(KEY, 2)
        assert inner == first_draws(OTHER, 2)

    def test_the_installed_generator_is_put_back(self):
        """Inside an argument's draw, a stream installs the ambient generator."""
        argument = random.Random(1)
        RNG._generator = argument
        try:
            with _Stream(KEY):
                assert RNG.generator() is RNG._ambient
            assert RNG.generator() is argument
        finally:
            RNG._generator = RNG._ambient

    def test_everything_is_put_back_when_the_block_raises(self):
        RNG.seed(3)
        state = RNG._ambient.getstate()
        with pytest.raises(ZeroDivisionError), _Stream(KEY):
            RNG.seed(5)
            RNG.generator().random()
            raise ZeroDivisionError

        assert (RNG.get_seed(), RNG._ambient.getstate()) == (3, state)
        assert RNG.generator() is RNG._ambient

    def test_a_frozen_exception_goes_through_unchanged(self):
        """contextlib.contextmanager would turn it into a FrozenInstanceError."""
        error = FrozenError(3)
        with pytest.raises(FrozenError) as excinfo, _Stream(KEY):
            raise error

        assert excinfo.value is error


# ---------------------------------------------------------------------------
# RNG.refresh_seed(key)
# ---------------------------------------------------------------------------


class TestRefreshSeed:
    def test_a_key_gets_the_user_stream_of_the_seed(self):
        RNG.seed(42)
        RNG.refresh_seed(key="tests/test_x.py::test_x")

        assert [RNG.generator().random() for _ in range(3)] == first_draws(
            StreamKey.root(42, "user", "tests/test_x.py::test_x")
        )

    def test_the_seed_and_the_key_cannot_run_into_each_other(self):
        """The 3.x key f"{seed}:{key}" gave these two pairs one stream."""
        RNG.seed("a")
        RNG.refresh_seed(key="b:c")
        first = RNG.generator().random()
        RNG.seed("a:b")
        RNG.refresh_seed(key="c")

        assert RNG.generator().random() != first

    @pytest.mark.parametrize("seed", [1.5, b"\x00", "fast"])
    def test_a_seed_that_is_not_an_int(self, seed):
        RNG.seed(seed)
        RNG.refresh_seed(key="k")
        drawn = RNG.generator().random()
        RNG.seed(seed)
        RNG.refresh_seed(key="k")

        assert RNG.generator().random() == drawn

    @pytest.mark.parametrize("key", [1.5, ("a", 1), True])
    def test_a_key_that_is_not_a_str_is_refused(self, key):
        with pytest.raises(TypeError, match="must be an int or a str"):
            RNG.refresh_seed(key=key)

    def test_without_a_key_the_generator_restarts_from_the_seed(self):
        RNG.seed(42)
        RNG.generator().random()
        RNG.refresh_seed()

        assert RNG.generator().random() == random.Random(42).random()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_a_session_puts_back_the_seed_and_the_ambient_generator(self):
        rt = StrategyRuntime()
        RNG.seed(3)
        RNG.generator().random()
        state = RNG._ambient.getstate()

        rt.push()
        RNG.seed(42)
        RNG.refresh_seed(key="inner")
        RNG.generator().random()
        RNG._generator = random.Random(1)  # left installed by the inner session
        rt.pop()

        assert (RNG.get_seed(), RNG._ambient.getstate()) == (3, state)
        assert RNG.generator() is RNG._ambient


# ---------------------------------------------------------------------------
# The factory's stream, T/"factory"
# ---------------------------------------------------------------------------


def _test_fn(f, x):
    pass


class TestFactoryStream:
    @staticmethod
    def build(factory, test_key="tests/test_x.py::test_x", name="fs"):
        return build_parametrization(
            name, factory, _test_fn, config=None, pytest_fixtures=set(), test_key=test_key
        )

    @staticmethod
    def factory(seen):
        def factory(rng):
            seen.append(rng is RNG.generator() is RNG._ambient)
            return Parameter(
                TestArg("f", value=(rng.random(), RNG.integer(0, 10**9))),
                TestArg("x", rng_type=RNGInteger(0, 10**9)),
                nsamples=3,
            )

        return factory

    def test_the_factory_draws_from_its_key(self):
        seen = []
        infos = self.build(self.factory(seen)).infos

        key = StreamKey.root(runtime.run_seed(), "test", "fs", "tests/test_x.py::test_x")
        generator = random.Random(key.child("factory").seed_int())
        assert seen == [True]
        assert {info.values.f for info in infos} == {
            (generator.random(), generator.randint(0, 10**9))
        }

    def test_two_tests_get_different_draws(self):
        one = self.build(self.factory([]), test_key="t.py::test_one").infos
        other = self.build(self.factory([]), test_key="t.py::test_other").infos

        assert one[0].values.f != other[0].values.f

    def test_an_extra_draw_leaves_the_rows(self):
        def drawing(rng):
            rng.random()
            return self.factory([])(rng)

        plain = self.build(self.factory([])).infos
        drawn = self.build(drawing).infos

        assert [info.values.x for info in drawn] == [info.values.x for info in plain]
        assert drawn[0].values.f != plain[0].values.f

    def test_the_seed_and_the_ambient_generator_are_put_back(self):
        def reseeding(rng):
            RNG.seed(5)
            rng.random()
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        RNG.seed(3)
        state = RNG._ambient.getstate()
        self.build(reseeding)

        assert (RNG.get_seed(), RNG._ambient.getstate()) == (3, state)

    def test_a_seed_set_while_the_rows_are_drawn_is_put_back(self):
        class Reseeding(RNGInteger):
            def generate(self):
                RNG.seed(5)
                return super().generate()

        RNG.seed(3)
        self.build(lambda: Parameter(TestArg("x", rng_type=Reseeding(0, 9))))

        assert RNG.get_seed() == 3


# ---------------------------------------------------------------------------
# A strategy file's stream, root(S, "file", path)
# ---------------------------------------------------------------------------


FILE_SOURCE = """
from pytest_strategy import RNG, register

RNG.seed(5)
RESEEDED = RNG.integer(0, 10**9)
SEED = RNG.get_seed()

@register("ps_unit_file")
def make():
    return None
"""

DRAWING_SOURCE = """
from pytest_strategy import RNG, register

DRAW = RNG.integer(0, 10**9)

@register("ps_unit_drawing")
def make():
    return None
"""


@pytest.fixture
def session():
    """A runtime session whose run seed is 99."""
    state = runtime.push()
    state.run_seed = 99
    try:
        yield state
    finally:
        runtime.pop()


def _config(rootpath):
    """A minimal config for _load_strategy_files, without a terminal reporter."""
    return SimpleNamespace(
        option=SimpleNamespace(verbose=0),
        rootpath=rootpath,
        pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
    )


class TestFileStream:
    def test_a_file_draws_from_its_path_under_the_run_seed(self, tmp_path, session):
        path = tmp_path / "pkg" / "strategies.py"
        path.parent.mkdir()
        path.write_text(DRAWING_SOURCE)
        RNG.seed(3)  # The run seed, not the current one, keys the stream

        PytestStrategyPlugin()._load_strategy_files([path], _config(tmp_path))

        module = session.strategy_modules[next(iter(session.strategy_modules))]
        key = StreamKey.root(99, "file", "pkg/strategies.py")
        assert random.Random(key.seed_int()).randint(0, 10**9) == module.DRAW

    def test_a_file_outside_the_rootdir_is_keyed_relative_to_it(self, tmp_path, session):
        path = tmp_path / "shared" / "strategies.py"
        path.parent.mkdir()
        path.write_text(DRAWING_SOURCE)
        (tmp_path / "proj").mkdir()

        PytestStrategyPlugin()._load_strategy_files([path], _config(tmp_path / "proj"))

        module = session.strategy_modules[next(iter(session.strategy_modules))]
        key = StreamKey.root(99, "file", "../shared/strategies.py")
        assert random.Random(key.seed_int()).randint(0, 10**9) == module.DRAW

    def test_a_reseed_in_a_file_stays_in_the_file(self, tmp_path, session):
        path = tmp_path / "strategies.py"
        path.write_text(FILE_SOURCE)
        RNG.seed(3)
        state = RNG._ambient.getstate()

        PytestStrategyPlugin()._load_strategy_files([path], _config(tmp_path))

        module = session.strategy_modules[next(iter(session.strategy_modules))]
        assert (5, random.Random(5).randint(0, 10**9)) == (module.SEED, module.RESEEDED)
        assert (RNG.get_seed(), RNG._ambient.getstate()) == (3, state)
