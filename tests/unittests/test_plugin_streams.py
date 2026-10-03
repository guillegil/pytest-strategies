"""
Tests for the plugin's random streams other than the rows (streams v1, D5): the
stream a block runs on (``_Stream``), also with another thread drawing,
``RNG.refresh_seed(key)``, a session's save and restore of the ambient generator, the
factory's stream T/"factory", a strategy file's stream, and the parts of a fixture's
and an export's keys that say where they are defined. Each one reseeds the
ambient generator in place from its key, when the block first uses it, and puts
back its state, the installed generator and the seed when it ends.

The fixture, test phase, test module and export streams run inside pytest; their
tests are in tests/integration/test_plugin_streams_integration.py.
"""

import _random
import contextlib
import copy
import dataclasses
import functools
import importlib.util
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pytest_strategy import RNG, Parameter, RNGInteger, TestArg
from pytest_strategy._registry import source_part
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy._streams import StreamKey, path_part
from pytest_strategy.plugin import PytestStrategyPlugin, _fixture_base, _fixture_definition
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


def operations(rng):
    """Draw with every kind of method random.Random has, from ``rng``."""
    items = list(range(10))
    rng.shuffle(items)
    return [
        rng.random(),
        rng.randint(0, 10**9),
        rng.getrandbits(70),
        rng.randbytes(5),
        rng.choice("abcdef"),
        rng.choices("abc", weights=[1, 2, 3], k=4),
        rng.sample(range(100), 5),
        items,
        rng.uniform(-1.0, 1.0),
        rng.gauss(0.0, 1.0),
        rng.gauss(0.0, 1.0),
        rng.expovariate(2.0),
    ]


class TestTheStreamIsSeededWhenUsed:
    """
    The ambient generator seeds a stream at its first draw, or when its state is
    read, so a block that draws nothing costs no seeding, saving or restoring of
    the generator's state; the values are those of a stream seeded on entry.
    """

    def test_a_block_that_draws_nothing_leaves_the_state_alone(self):
        calls = []

        def key():
            calls.append(1)
            return KEY

        RNG.seed(3)
        state = _random.Random.getstate(RNG._ambient)
        with _Stream(key):
            # Neither the key nor the generator's state is touched
            assert calls == []
            assert _random.Random.getstate(RNG._ambient) == state

        assert calls == []
        assert RNG.generator().random() == random.Random(3).random()

    def test_the_key_is_built_once_at_the_first_draw(self):
        calls = []

        def key():
            calls.append(1)
            return KEY

        with _Stream(key):
            drawn = [RNG.generator().random() for _ in range(3)]

        assert calls == [1]
        assert drawn == first_draws(KEY)

    def test_every_method_draws_what_a_generator_seeded_on_entry_draws(self):
        with _Stream(KEY) as rng:
            drawn = operations(rng)

        assert drawn == operations(random.Random(KEY.seed_int()))

    def test_a_cached_gauss_value_stays_with_its_stream(self):
        """random.Random.gauss() keeps a second value for its next call."""
        RNG.seed(3)
        outer = [RNG.generator().gauss(0.0, 1.0)]
        with _Stream(KEY):
            inner = [RNG.generator().gauss(0.0, 1.0) for _ in range(3)]
        outer += [RNG.generator().gauss(0.0, 1.0) for _ in range(2)]

        reference = random.Random(3)
        assert outer == [reference.gauss(0.0, 1.0) for _ in range(3)]
        reference = random.Random(KEY.seed_int())
        assert inner == [reference.gauss(0.0, 1.0) for _ in range(3)]

    def test_an_inner_stream_leaves_an_outer_one_that_has_not_drawn(self):
        RNG.seed(3)
        with _Stream(KEY):
            with _Stream(OTHER):
                inner = RNG.generator().random()
            outer = [RNG.generator().random() for _ in range(2)]
        after = RNG.generator().random()

        assert inner == first_draws(OTHER, 1)[0]
        assert outer == first_draws(KEY, 2)
        assert after == random.Random(3).random()

    def test_an_inner_stream_leaves_an_outer_one_that_has_drawn(self):
        with _Stream(KEY):
            outer = [RNG.generator().random()]
            with _Stream(OTHER), _Stream(StreamKey.root(7, "ctx")):
                RNG.generator().random()
            outer.append(RNG.generator().random())

        assert outer == first_draws(KEY, 2)

    def test_reading_the_state_gives_the_streams(self):
        with _Stream(KEY):
            state = RNG.generator().getstate()
            drawn = RNG.generator().random()

        generator = random.Random()
        generator.setstate(state)
        assert generator.random() == drawn == first_draws(KEY, 1)[0]

    @pytest.mark.parametrize("replace", ["seed", "setstate"])
    def test_replacing_the_state_before_a_draw_changes_only_the_block(self, replace):
        RNG.seed(3)
        RNG.generator().random()
        with _Stream(KEY):
            if replace == "seed":
                RNG.generator().seed(5)
            else:
                RNG.generator().setstate(random.Random(5).getstate())
            drawn = RNG.generator().random()
        after = RNG.generator().random()

        assert drawn == random.Random(5).random()
        generator = random.Random(3)
        assert after == [generator.random() for _ in range(2)][1]

    def test_the_position_is_the_pending_key_until_the_stream_is_used(self):
        def key():
            return KEY

        with _Stream(key):
            assert RNG._ambient._position() is key
            with _Stream(OTHER):
                pass
            assert RNG._ambient._position() is key
            RNG.generator().random()
            assert RNG._ambient._position() == RNG._ambient.getstate()

    def test_a_copy_is_a_plain_generator_on_the_same_state(self):
        with _Stream(KEY):
            copied = copy.copy(RNG.generator())
            drawn = RNG.generator().random()

        assert copied.random() == drawn


class TestAnotherThreadDraws:
    """
    A thread that draws from the RNG (a stimulus thread started by a fixture)
    while the main thread enters and ends streams: seeding a pending stream holds
    the generator's lock, as entering and ending one do, so a draw never seeds the
    generator from a stream that has ended, or saves the state on another stream
    (an IndexError outside every stream).
    """

    WHERE = ["outside every stream", "in a stream that has drawn"]

    @contextlib.contextmanager
    def around(self, where):
        """Run the block outside every stream, or in a stream that has drawn."""
        if where == "outside every stream":
            # Without the test's own call stream, as between two tests
            with mock.patch.object(RNG._ambient, "_streams", []):
                yield
        else:
            with _Stream(OTHER):
                RNG.generator().random()
                yield

    @pytest.mark.parametrize("where", WHERE)
    def test_a_stream_ends_while_another_thread_seeds_it(self, where):
        settling = threading.Event()
        ending = threading.Event()
        drawn = []
        errors = []

        def key():
            settling.set()
            ending.wait(5)
            # Time for the stream to end, unless ending it waits for the seeding
            time.sleep(0.02)
            return KEY

        def draw():
            try:
                drawn.append(RNG._ambient.random())
            except Exception as error:
                errors.append(error)

        RNG.seed(3)  # seeds the test's call stream, so that nothing is pending
        before = RNG._ambient.getstate()
        with self.around(where):
            inside = RNG._ambient.getstate()
            stream = _Stream(key)
            stream.__enter__()
            thread = threading.Thread(target=draw)
            thread.start()
            assert settling.wait(5)
            ending.set()
            stream.__exit__(None, None, None)
            thread.join()

            assert errors == []
            # The draw seeded the stream before it ended, which put its state back
            assert drawn == first_draws(KEY, 1)
            assert RNG._ambient.getstate() == inside
            assert RNG._ambient._pending is None
        assert RNG._ambient.getstate() == before

    @pytest.mark.parametrize("where", WHERE)
    def test_streams_begin_and_end_while_another_thread_draws(self, where):
        errors = []
        stop = threading.Event()

        def draw():
            while not stop.is_set():
                try:
                    RNG.integer(0, 255)
                except Exception as error:
                    errors.append(error)
                    return

        RNG.seed(3)
        before = RNG._ambient.getstate()
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with self.around(where):
                streams = list(RNG._ambient._streams)
                thread = threading.Thread(target=draw)
                thread.start()
                try:
                    for i in range(5000):
                        # Keyed lazily from a path, as the module and fixture streams are
                        with _Stream(
                            lambda i=i: StreamKey.root(7, "module", path_part(__file__, None), i)
                        ):
                            pass
                finally:
                    stop.set()
                    thread.join()
                assert (RNG._ambient._pending, RNG._ambient._streams) == (None, streams)
        finally:
            sys.setswitchinterval(interval)

        assert errors == []
        if where != "outside every stream":
            # The thread drew from the outer stream, which put back the state
            assert RNG._ambient.getstate() == before
        with _Stream(KEY):
            assert RNG.generator().random() == first_draws(KEY, 1)[0]


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

    @pytest.mark.parametrize("key", [1.5, ("a", 1), True, Path("tests") / "test_x.py"])
    def test_another_key_names_the_stream_of_its_str(self, key):
        """3.x formatted any key into its string, so any key still works."""
        RNG.seed(42)
        RNG.refresh_seed(key=key)

        assert [RNG.generator().random() for _ in range(3)] == first_draws(
            StreamKey.root(42, "user", str(key))
        )

    def test_an_int_key_and_its_str_are_two_streams(self):
        RNG.seed(42)
        RNG.refresh_seed(key=3)
        drawn = RNG.generator().random()
        RNG.refresh_seed(key="3")

        assert drawn == first_draws(StreamKey.root(42, "user", 3), 1)[0]
        assert RNG.generator().random() != drawn

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


# ---------------------------------------------------------------------------
# Where a fixture is defined, a part of its stream's key
# ---------------------------------------------------------------------------


FIXTURES_SOURCE = """
import functools


def wrap(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def conn():
    pass


@wrap
def wrapped():
    pass


class TestDb:
    def conn(self):
        pass
"""


def load(path, name):
    """Import the file at ``path`` as the module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


class TestFixtureDefinition:
    def test_a_fixture_is_named_by_its_file_and_qualified_name(self, tmp_path):
        path = tmp_path / "tests" / "conftest.py"
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_fixtures")

        assert _fixture_definition(module.conn, tmp_path) == ("tests/conftest.py", "conn")
        assert _fixture_definition(module.TestDb().conn, tmp_path) == (
            "tests/conftest.py",
            "TestDb.conn",
        )
        # A functools.wraps decorator is looked through
        assert _fixture_definition(module.wrapped, tmp_path) == ("tests/conftest.py", "wrapped")

    def test_a_file_outside_the_rootdir_is_relative_to_it(self, tmp_path):
        path = tmp_path / "shared" / "fixtures.py"
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        (tmp_path / "proj").mkdir()
        module = load(path, "ps_unit_shared_fixtures")

        assert _fixture_definition(module.conn, tmp_path / "proj") == (
            "../shared/fixtures.py",
            "conn",
        )

    @pytest.mark.parametrize("folder", ["site-packages", "dist-packages"])
    def test_an_installed_packages_fixture_is_named_by_its_module(self, tmp_path, folder):
        """Its path depends on where the package is installed; its module does not."""
        path = tmp_path / ".venv" / "lib" / "python3.11" / folder / "acme" / "fixtures.py"
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "acme.fixtures")

        assert _fixture_definition(module.conn, tmp_path) == ("acme.fixtures", "conn")
        assert _fixture_definition(module.TestDb().conn, tmp_path) == (
            "acme.fixtures",
            "TestDb.conn",
        )

    def test_a_partial_counts_as_the_function_it_wraps(self, tmp_path):
        path = tmp_path / "conftest.py"
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_partial_fixtures")

        assert _fixture_definition(functools.partial(module.conn), tmp_path) == (
            "conftest.py",
            "conn",
        )

    @pytest.mark.parametrize(
        ("namespace", "where"), [({}, ""), ({"__name__": "made"}, "made")], ids=["bare", "named"]
    )
    def test_code_without_a_file_is_named_by_its_module(
        self, tmp_path, monkeypatch, namespace, where
    ):
        """Its file, "<string>", would resolve against the working directory."""
        exec("def made():\n    pass\n", namespace)
        (tmp_path / "tests").mkdir()

        found = []
        for cwd in (tmp_path, tmp_path / "tests"):
            monkeypatch.chdir(cwd)
            found.append(_fixture_definition(namespace["made"], tmp_path))

        assert found == [(where, "made")] * 2


class TestFixtureBase:
    """Where pytest registered a fixture: a part of its key."""

    @pytest.mark.parametrize(
        ("nodeid", "base"),
        [
            # pytest 9's node for the rootdir, as pytest 8's baseid
            (".", ""),
            ("", ""),
            ("tests/a", "tests/a"),
            ("tests/test_a.py::TestDb", "tests/test_a.py::TestDb"),
        ],
    )
    def test_the_node_it_is_registered_for(self, nodeid, base):
        fixturedef = SimpleNamespace(node=SimpleNamespace(nodeid=nodeid), baseid=nodeid)

        assert _fixture_base(fixturedef) == base

    def test_without_a_node_its_baseid(self):
        """pytest 8 has no FixtureDef.node, and some fixtures have none on pytest 9."""
        assert _fixture_base(SimpleNamespace(baseid="tests/a")) == "tests/a"
        assert _fixture_base(SimpleNamespace(node=None, baseid="tests/a")) == "tests/a"
        assert _fixture_base(SimpleNamespace(node=None, baseid="")) == ""


class TestSourcePart:
    """Where a factory is defined, as the export stream's folder."""

    def test_the_folder_of_its_file(self, tmp_path):
        path = tmp_path / "tests" / "a" / "strategies.py"
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_source_folder")

        assert source_part(module.conn, tmp_path, folder=True) == "tests/a"
        assert source_part(module.conn, tmp_path) == "tests/a/strategies.py"

    @pytest.mark.parametrize("folder", ["site-packages", "dist-packages"])
    def test_an_installed_package_is_named_by_its_module(self, tmp_path, folder):
        """The same package installed in two environments gives the same part."""
        found = set()
        for env in ("env_a", "env_b"):
            path = tmp_path / env / "lib" / "python3" / folder / "acme" / "strategies.py"
            path.parent.mkdir(parents=True)
            path.write_text(FIXTURES_SOURCE)
            module = load(path, "acme.strategies")
            found.add(source_part(module.conn, tmp_path, folder=True))
            found.add(source_part(functools.partial(module.conn), tmp_path, folder=True))
            found.add(source_part(module.TestDb, tmp_path, folder=True))

        assert found == {"acme.strategies"}
