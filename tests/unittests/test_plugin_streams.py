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
import json
import os
import random
import signal
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, export_strategies, register
from pytest_strategy import rng as rng_module
from pytest_strategy._registry import registry, source_part
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._runtime import StrategyRuntime, runtime
from pytest_strategy._streams import StreamKey, path_part
from pytest_strategy.plugin import (
    PytestStrategyPlugin,
    _fixture_base,
    _fixture_definition,
    definition_part,
)
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

    @pytest.mark.parametrize("call", ["draw", "seed"])
    @pytest.mark.parametrize("step", ["_mt_getstate", "_mt_seed"])
    def test_a_draw_while_another_thread_seeds_the_stream_waits_for_it(self, step, call):
        """
        The stream stays pending until it is seeded: the main thread's draw, or its
        RNG.seed(), while a thread seeds the stream waits for the seeding, and
        neither comes from the state the seeding replaces nor is overwritten by it.

        The thread pauses in each step of _settle that runs while the stream is
        pending: where it saves the state of the stream around (_mt_getstate), and
        where it seeds the stream (_mt_seed), the last step before the stream stops
        being pending.
        """
        paused = threading.Event()
        resume = threading.Event()
        original = getattr(rng_module, step)

        def pausing(generator, *args):
            if threading.current_thread() is thread:
                paused.set()
                resume.wait(5)
            return original(generator, *args)

        drawn = []
        thread = threading.Thread(target=lambda: drawn.append(RNG._ambient.random()))
        with _Stream(KEY):
            outer = [RNG.generator().random()]
            with _Stream(OTHER), mock.patch.object(rng_module, step, pausing):
                thread.start()
                assert paused.wait(5)
                # Checked once the thread has finished, so that it is never left paused
                pending = RNG._ambient._pending
                # Let the thread go on once this thread waits for it
                timer = threading.Timer(0.05, resume.set)
                timer.start()
                if call == "seed":
                    RNG.seed(42)
                inner = [RNG.generator().random()]
                thread.join()
                timer.join()
                # The seeding did not replace what this thread drew from, or seeded
                inner.append(RNG.generator().random())
            # The thread never drew from the outer stream
            outer.append(RNG.generator().random())

        assert drawn == first_draws(OTHER, 1)
        if call == "draw":
            assert inner == first_draws(OTHER, 3)[1:]
        else:
            seeded = random.Random(42)
            assert inner == [seeded.random(), seeded.random()]
        assert outer == first_draws(KEY, 2)
        assert pending is OTHER

    @pytest.mark.parametrize(
        "first_draws_before", [False, True], ids=["first_pending", "first_drawn"]
    )
    @pytest.mark.parametrize(
        "second_draws_before", [False, True], ids=["second_pending", "second_drawn"]
    )
    def test_streams_that_end_in_the_order_they_began(
        self, first_draws_before, second_draws_before
    ):
        """
        Streams that two threads enter end in any order: one that ends before a
        stream entered after it hands what it would put back to that stream, which
        puts it back when it ends, and stays in use until then.
        """
        RNG.seed(3)
        RNG.generator().random()
        ambient = RNG._ambient
        before = (ambient.getstate(), ambient._pending, list(ambient._streams))
        first, second = _Stream(KEY), _Stream(OTHER)

        first.__enter__()
        if first_draws_before:
            RNG.generator().random()
        second.__enter__()
        drawn = [RNG.generator().random()] if second_draws_before else []
        first.__exit__(None, None, None)

        assert ambient._streams == [*before[2], second]
        # The second stream is still in use
        drawn.append(RNG.generator().random())
        assert drawn == first_draws(OTHER, len(drawn))
        RNG.seed(7)
        second.__exit__(None, None, None)

        assert (ambient.getstate(), ambient._pending, ambient._streams) == before
        assert (RNG.get_seed(), RNG.generator()) == (3, ambient)
        generator = random.Random(3)
        assert RNG.generator().random() == [generator.random() for _ in range(2)][1]

    def test_export_in_a_thread_ends_its_stream_while_a_test_phase_runs(self, monkeypatch):
        """
        export_strategies() in a thread enters its export stream before a test
        phase's stream begins in the main thread, and ends it first: the phase still
        draws from its stream, and when it ends nothing is left pending.
        """
        monkeypatch.setattr(runtime, "_stack", [])
        runtime.push(SimpleNamespace(getoption=lambda name, default=None: default))
        runtime.current.all_loaded = True
        runtime.current.run_seed = 7
        registry.restore({})
        exporting = threading.Event()
        go = threading.Event()

        @register("ps_unit_threaded_export")
        def factory(rng):
            rng.random()
            exporting.set()
            go.wait(5)
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        ambient = RNG._ambient
        before = (ambient.getstate(), ambient._pending, list(ambient._streams), ambient)
        exported = []
        thread = threading.Thread(target=lambda: exported.append(export_strategies()))
        try:
            thread.start()
            assert exporting.wait(5)
            with _Stream(KEY):
                drawn = [RNG.generator().random()]
                go.set()
                thread.join()
                drawn.append(RNG.generator().random())
            after = (ambient.getstate(), ambient._pending, list(ambient._streams), RNG.generator())
        finally:
            go.set()
            thread.join()
            runtime.pop()

        assert "ps_unit_threaded_export" in json.loads(exported[0])
        assert drawn == first_draws(KEY, 2)
        assert after == before

    def test_rows_drawn_by_two_threads_put_back_the_generator(self):
        """
        Each row installs its arguments' generators as RNG.generator() (a class
        attribute) under the ambient generator's lock: a row that another thread
        draws meanwhile waits for it, so neither takes the other's generator for
        the one to put back.
        """
        first_entered = threading.Event()
        first_go = threading.Event()
        second_go = threading.Event()

        class Waiting(RNGInteger):
            def generate(self):
                if threading.current_thread() is first:
                    first_entered.set()
                    first_go.wait(5)
                elif threading.current_thread() is second:
                    second_go.wait(5)
                return super().generate()

        param = Parameter(TestArg("x", rng_type=Waiting(0, 9)))
        first = threading.Thread(target=lambda: param.generate_vectors(1))
        second = threading.Thread(target=lambda: param.generate_vectors(1))
        try:
            first.start()
            assert first_entered.wait(5)
            second.start()
            # Without the lock, the second row would now hold the first's generator
            # as the one to put back, and end after the first
            first_go.set()
            first.join()
            second_go.set()
            second.join()
        finally:
            first_go.set()
            second_go.set()
            first.join()
            second.join()

        assert RNG.generator() is RNG._ambient

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork() is POSIX only")
    def test_a_child_forked_while_another_thread_seeds_a_stream_can_draw(self):
        """
        A fork waits for the seeding, and the child gets a lock of its own: the
        parent's lock, held by a thread the child does not have, would block the
        child's first draw forever.
        """
        building = threading.Event()
        built = threading.Event()

        def key():
            building.set()
            built.wait(5)
            return KEY

        thread = threading.Thread(target=lambda: RNG._ambient.random())
        with _Stream(key):
            thread.start()
            assert building.wait(5)
            timer = threading.Timer(0.05, built.set)
            timer.start()
            with warnings.catch_warnings():
                # Python 3.12+ warns that the process has other threads
                warnings.simplefilter("ignore", DeprecationWarning)
                pid = os.fork()
            if pid == 0:
                # The child: draws with a time limit, and leaves without pytest
                code = 1
                try:
                    signal.alarm(5)
                    with _Stream(OTHER):
                        code = 0 if RNG._ambient.random() == first_draws(OTHER, 1)[0] else 1
                finally:
                    os._exit(code)
            thread.join()
            timer.join()
        _, status = os.waitpid(pid, 0)

        assert os.waitstatus_to_exitcode(status) == 0


class TestTheSeedingThreadDraws:
    """
    A draw from the thread that seeds a stream, while it does (a signal handler,
    or a finalizer the garbage collector runs), comes from the state in use, and
    leaves the state the stream puts back alone.
    """

    def test_while_the_key_is_built(self):
        reentered = []

        calls = []

        def key():
            calls.append(1)
            if len(calls) == 1:
                reentered.append(RNG.generator().random())
            return OTHER

        with _Stream(KEY):
            outer = [RNG.generator().random()]
            with _Stream(key):
                inner = [RNG.generator().random() for _ in range(2)]
            outer.append(RNG.generator().random())

        assert inner == first_draws(OTHER, 2)
        assert outer == first_draws(KEY, 2)
        # From the stream around, whose state was saved before
        assert reentered == first_draws(KEY, 2)[1:]

    def test_while_the_state_is_saved(self):
        """Before the stream is marked as being seeded: the draw seeds it itself."""
        reentered = []
        getstate = rng_module._mt_getstate
        calls = []

        def drawing_getstate(generator):
            calls.append(generator)
            if len(calls) == 1:
                reentered.append(RNG.generator().random())
            return getstate(generator)

        with _Stream(KEY):
            outer = [RNG.generator().random()]
            with _Stream(OTHER), mock.patch.object(rng_module, "_mt_getstate", drawing_getstate):
                inner = [RNG.generator().random() for _ in range(2)]
            outer.append(RNG.generator().random())

        assert reentered + inner == first_draws(OTHER, 3)
        assert outer == first_draws(KEY, 2)


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

    def test_a_folder_linked_from_outside_is_keyed_as_pytest_spells_it(self, tmp_path, session):
        """
        Collection loads a test folder's strategy files by their real paths; a folder
        linked into the rootdir from a place that does not move with the checkout
        keys them by its link, as pytest spells it, in every checkout.
        """
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "strategies.py").write_text(DRAWING_SOURCE)
        proj = tmp_path / "deep" / "proj"
        proj.mkdir(parents=True)
        try:
            os.symlink(shared, proj / "tests_shared", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")

        PytestStrategyPlugin()._load_strategy_files(
            [shared / "strategies.py"], _config(proj), spelled=proj / "tests_shared"
        )

        module = session.strategy_modules[next(iter(session.strategy_modules))]
        key = StreamKey.root(99, "file", "tests_shared/strategies.py")
        assert random.Random(key.seed_int()).randint(0, 10**9) == module.DRAW

    def test_a_loaded_file_is_recorded_as_imported_by_its_path(self, tmp_path, session):
        """Its factories are then keyed by its path (definition_part), also outside the rootdir."""
        path = tmp_path / "other" / "strategies.py"
        path.parent.mkdir()
        path.write_text(DRAWING_SOURCE)
        (tmp_path / "proj").mkdir()

        PytestStrategyPlugin()._load_strategy_files([path], _config(tmp_path / "proj"))

        assert session.imported_files == {os.path.normcase(os.path.realpath(path))}

    @pytest.mark.parametrize("folder", ["site-packages", "dist-packages"])
    def test_an_installed_packages_file_is_keyed_below_its_folder(self, tmp_path, session, folder):
        path = tmp_path / "venv" / "lib" / folder / "acme" / "strategies.py"
        path.parent.mkdir(parents=True)
        path.write_text(DRAWING_SOURCE)

        PytestStrategyPlugin()._load_strategy_files([path], _config(tmp_path))

        module = session.strategy_modules[next(iter(session.strategy_modules))]
        key = StreamKey.root(99, "file", "acme/strategies.py")
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


def _ini_config(rootpath, **ini):
    """
    A config with the ini values that tell where pytest collects test modules
    (pytest's defaults, with ``ini``'s values instead).
    """
    values = {
        "python_files": ["test_*.py", "*_test.py"],
        "testpaths": [],
        "norecursedirs": ["*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv"],
        "consider_namespace_packages": False,
        **ini,
    }
    return SimpleNamespace(rootpath=rootpath, getini=values.__getitem__)


def load(path, name):
    """
    Import the file at ``path`` as the module ``name``, and leave it out of
    ``sys.modules``, as a module loaded by its path.
    """
    with imported(path, name) as module:
        return module


@contextlib.contextmanager
def imported(path, name):
    """Import the file at ``path`` as the module ``name``, in ``sys.modules`` in the block."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[name]


class TestFixtureDefinition:
    def test_a_fixture_is_named_by_its_file_and_qualified_name(self, tmp_path):
        path = tmp_path / "tests" / "conftest.py"
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_fixtures")

        assert _fixture_definition(module.conn, _config(tmp_path)) == ("tests/conftest.py", "conn")
        assert _fixture_definition(module.TestDb().conn, _config(tmp_path)) == (
            "tests/conftest.py",
            "TestDb.conn",
        )
        # A functools.wraps decorator is looked through
        assert _fixture_definition(module.wrapped, _config(tmp_path)) == (
            "tests/conftest.py",
            "wrapped",
        )

    def test_a_file_outside_the_rootdir_is_relative_to_it(self, tmp_path):
        """A module that sys.modules does not have under its name: loaded by its path."""
        path = tmp_path / "shared" / "fixtures.py"
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        (tmp_path / "proj").mkdir()
        module = load(path, "ps_unit_shared_fixtures")

        assert _fixture_definition(module.conn, _config(tmp_path / "proj")) == (
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

        assert _fixture_definition(module.conn, _config(tmp_path)) == ("acme.fixtures", "conn")
        assert _fixture_definition(module.TestDb().conn, _config(tmp_path)) == (
            "acme.fixtures",
            "TestDb.conn",
        )

    def test_a_package_is_named_by_its_module_wherever_it_is(self, tmp_path):
        """
        Installed, installed in editable mode (its source tree's src/ folder, or
        setuptools' strict editable folder), or in another checkout: one key, so a
        seed from a run of the installed package reruns in a checkout of it.
        """
        found = set()
        for folder in (
            tmp_path / "proj" / ".tox" / "py311" / "lib" / "python3.11" / "site-packages",
            tmp_path / "proj" / "src",
            tmp_path / "proj" / "build" / "__editable__.acme-1.0",
            tmp_path / "other" / "src",
        ):
            path = folder / "acme" / "testing.py"
            path.parent.mkdir(parents=True)
            path.write_text(FIXTURES_SOURCE)
            with imported(path, "acme.testing") as module:
                found.add(_fixture_definition(module.conn, _config(tmp_path / "proj")))
                found.add((source_part(module.conn, tmp_path / "proj", folder=True), "export"))

        assert found == {("acme.testing", "conn"), ("acme.testing", "export")}

    @pytest.mark.parametrize(
        "name", ["conftest.py", "test_db.py", "db_test.py", "strategies.py", "db_strategies.py"]
    )
    def test_a_file_pytest_imports_by_its_path_is_named_by_its_path(self, tmp_path, name):
        """Its module's name depends on --import-mode and the __init__.py files."""
        path = tmp_path / "tests" / "acme" / name
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)

        with imported(path, "acme." + name.removesuffix(".py")) as module:
            assert _fixture_definition(module.conn, _config(tmp_path)) == (
                f"tests/acme/{name}",
                "conn",
            )
            assert source_part(module.conn, tmp_path, folder=True) == "tests/acme"

    def test_the_test_modules_are_those_of_python_files(self, tmp_path):
        config = SimpleNamespace(rootpath=tmp_path, getini={"python_files": ["check_*.py"]}.get)
        found = []
        for name in ("check_db.py", "test_db.py"):
            path = tmp_path / "acme" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(FIXTURES_SOURCE)
            with imported(path, "acme." + name.removesuffix(".py")) as module:
                found.append(_fixture_definition(module.conn, config)[0])

        assert found == ["acme/check_db.py", "acme.test_db"]

    @pytest.mark.parametrize("module", ["tests.helpers", "proj.tests.helpers"])
    def test_a_name_that_begins_above_the_rootdir_is_not_used(self, tmp_path, module):
        """
        A rootdir with an __init__.py (pytest.ini in tests/, or a package checkout):
        a module's name then begins with the name of the rootdir's folder, or of
        one above it, which another checkout may not have.
        """
        path = tmp_path / "proj" / "tests" / "helpers.py"
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)

        with imported(path, module) as loaded:
            assert _fixture_definition(loaded.conn, _config(path.parent))[0] == "helpers.py"
            # From the folder the name is relative to, or a folder below it: the name
            assert _fixture_definition(loaded.conn, _config(path.parent.parent))[0] == (
                "tests.helpers" if module == "tests.helpers" else "tests/helpers.py"
            )

    def test_a_package_next_to_a_rootdir_in_a_subfolder_is_named_by_its_module(self, tmp_path):
        """
        A flat layout run with its rootdir in tests/ (tests/pytest.ini, or
        --rootdir=tests): the names begin next to the rootdir, not with its folder
        or one above it, so the checkout draws as the installed package does.
        """
        found = []
        for folder in (
            tmp_path / "proj",
            tmp_path / "proj" / ".venv" / "lib" / "python3.11" / "site-packages",
        ):
            (folder / "acme").mkdir(parents=True)
            for module in ("acme.testing", "helpers"):
                path = folder.joinpath(*module.split(".")).with_suffix(".py")
                path.write_text(FIXTURES_SOURCE)
                with imported(path, module) as loaded:
                    config = _ini_config(tmp_path / "proj" / "tests")
                    found.append(_fixture_definition(loaded.conn, config)[0])
                    found.append(definition_part(loaded.conn, config, folder=True))

        assert found == ["acme.testing", "acme.testing", "helpers", "helpers"] * 2

    def test_a_name_whose_first_folder_holds_the_rootdir_is_not_used(self, tmp_path):
        """
        Also for a file outside the rootdir: proj.util, which the tests of a
        checkout that is a package import (rootdir proj/tests), has the checkout
        folder's name in its name.
        """
        path = tmp_path / "proj" / "util.py"
        (tmp_path / "proj" / "tests").mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)

        with imported(path, "proj.util") as loaded:
            config = _ini_config(tmp_path / "proj" / "tests")
            assert _fixture_definition(loaded.conn, config)[0] == "../util.py"
            assert definition_part(loaded.conn, config, folder=True) == ".."

    @pytest.mark.parametrize("name", ["test_utils.py", "utils_test.py", "strategies.py"])
    def test_a_package_module_named_like_a_test_module_keeps_its_path_in_a_checkout(
        self, tmp_path, name
    ):
        """
        A file named like a test module or a strategy file inside the rootdir is keyed
        by its path, so src/acme/test_utils.py, which pytest never collects with
        testpaths = tests, is keyed by its path in a checkout and by its module's
        name installed: the two draw different values (the limitation docs/dev.md
        states; renaming the module avoids it).
        """
        module = "acme." + name.removesuffix(".py")
        found = []
        for folder in (
            tmp_path / "proj" / "src",
            tmp_path / "proj" / ".venv" / "lib" / "python3.11" / "site-packages",
        ):
            path = folder / "acme" / name
            path.parent.mkdir(parents=True)
            path.write_text(FIXTURES_SOURCE)
            with imported(path, module) as loaded:
                config = _ini_config(tmp_path / "proj", testpaths=["tests"])
                found.append(_fixture_definition(loaded.conn, config)[0])
                found.append(definition_part(loaded.conn, config, folder=True))

        assert found == [f"src/acme/{name}", "src/acme", module, module]

    @pytest.mark.parametrize("name", ["test_utils.py", "utils_test.py", "strategies.py"])
    def test_a_package_module_named_like_a_test_module_next_to_the_rootdir_has_its_name(
        self, tmp_path, name
    ):
        """
        A flat layout run with its rootdir in tests/: acme/test_utils.py is outside
        the rootdir and the testpaths, and the session did not import it by its path,
        so it is named by its module in the checkout as installed.
        """
        module = "acme." + name.removesuffix(".py")
        found = []
        for folder in (
            tmp_path / "proj",
            tmp_path / "proj" / ".venv" / "lib" / "python3.11" / "site-packages",
        ):
            path = folder / "acme" / name
            path.parent.mkdir(parents=True)
            path.write_text(FIXTURES_SOURCE)
            with imported(path, module) as loaded:
                config = _ini_config(tmp_path / "proj" / "tests", testpaths=["unit"])
                found.append(_fixture_definition(loaded.conn, config)[0])
                found.append(definition_part(loaded.conn, config, folder=True))

        assert found == [module] * 4

    @pytest.mark.parametrize(
        "folder",
        [
            "tests/acme",
            "src/acme",
            # Folders pytest does not enter (norecursedirs, hidden)
            "build/acme",
            ".cache/acme",
        ],
    )
    @pytest.mark.parametrize("name", ["test_utils.py", "strategies.py", "conftest.py"])
    def test_inside_the_rootdir_a_test_module_is_named_by_its_path_in_every_run(
        self, tmp_path, folder, name
    ):
        """
        A conftest.py, a file named like a test module or a strategy file inside the
        rootdir is keyed by its path whatever the testpaths, norecursedirs and the
        folders named on the command line, which differ between a full run and a run
        of one node ID or of a folder outside the testpaths: pytest imports it by its
        path in some of them, under a module name that depends on --import-mode.
        """
        path = tmp_path / "proj" / folder / name
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)

        found = set()
        with imported(path, "acme." + name.removesuffix(".py")) as loaded:
            for testpaths in ([], ["tests"], ["t*s"], ["src"], ["../shared"], ["tests/unit"]):
                config = _ini_config(tmp_path / "proj", testpaths=testpaths)
                found.add(_fixture_definition(loaded.conn, config)[0])
                found.add(definition_part(loaded.conn, config, folder=True))

        assert found == {f"{folder}/{name}", folder}

    @pytest.mark.parametrize(
        ("testpaths", "keyed"),
        [
            (["../shared"], True),
            (["../sh*"], True),
            (["tests", "../shared/acme"], True),
            ([], False),
            (["tests"], False),
            (["../shared/other"], False),
        ],
    )
    @pytest.mark.parametrize("name", ["test_utils.py", "strategies.py", "conftest.py"])
    def test_outside_the_rootdir_a_testpaths_entry_keys_it_by_its_path(
        self, tmp_path, name, testpaths, keyed
    ):
        """
        Outside the rootdir, a file below a testpaths entry (testpaths = ../shared) is
        keyed by its path in every run. One that is not is a module of a library on
        sys.path, named by its module, when the session did not import it by its path.
        """
        path = tmp_path / "shared" / "acme" / name
        path.parent.mkdir(parents=True)
        (tmp_path / "shared" / "other").mkdir()
        (tmp_path / "proj").mkdir()
        path.write_text(FIXTURES_SOURCE)
        module = "acme." + name.removesuffix(".py")

        with imported(path, module) as loaded:
            config = _ini_config(tmp_path / "proj", testpaths=testpaths)
            found = [
                _fixture_definition(loaded.conn, config)[0],
                definition_part(loaded.conn, config, folder=True),
            ]

        assert found == (
            [f"../shared/acme/{name}", "../shared/acme"] if keyed else [module, module]
        )

    @pytest.mark.parametrize("name", ["test_utils.py", "strategies.py", "conftest.py"])
    def test_outside_the_rootdir_a_file_the_session_imported_by_its_path_has_its_path(
        self, tmp_path, name
    ):
        """
        A test module that pytest collected from a folder named on the command line
        outside the rootdir and the testpaths, its folder's conftest.py, or a strategy
        file the plugin loaded there, in a folder that is not a package: the session
        imported it by its path, under a module name that --import-mode derives from
        its path, so its path keys it.
        """
        path = tmp_path / "other" / name
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        root = tmp_path / "proj"
        root.mkdir()
        module = name.removesuffix(".py")

        with imported(path, module) as loaded:
            found = [
                source_part(loaded.conn, root),
                source_part(loaded.conn, root, imported={os.path.normcase(os.path.realpath(path))}),
                source_part(
                    loaded.conn,
                    root,
                    folder=True,
                    imported={os.path.normcase(os.path.realpath(path))},
                ),
            ]

        assert found == [module, f"../other/{name}", "../other"]

    @pytest.mark.parametrize("name", ["test_utils.py", "strategies.py", "conftest.py"])
    @pytest.mark.parametrize(
        ("folders", "module"),
        [
            # acme/ has an __init__.py, proj/ next to it does not
            (["acme"], "acme"),
            (["acme", "acme/sub"], "acme.sub"),
            # The chain ends at the first folder without an __init__.py
            (["acme/sub"], "sub"),
        ],
    )
    def test_a_package_module_the_session_imported_by_its_path_keeps_its_name(
        self, tmp_path, name, folders, module
    ):
        """
        A module of a regular package outside the rootdir and the testpaths that the
        session imported by its path (pytest collected its folder, as pytest .
        ../acme does from a rootdir in tests/): without consider_namespace_packages
        (source_part's default), pytest 8 and 9 import it under its package name,
        the dotted name of the folders with an __init__.py above it, in every import
        mode, the name another module's import gives it, so it keeps that name in
        the runs that collect it and in those that do not.
        """
        for folder in folders:
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
            (tmp_path / folder / "__init__.py").write_text("")
        path = tmp_path / folders[-1] / name
        path.write_text(FIXTURES_SOURCE)
        root = tmp_path / "tests"
        root.mkdir()
        module = f"{module}.{name.removesuffix('.py')}"
        recorded = {os.path.normcase(os.path.realpath(path))}

        with imported(path, module) as loaded:
            found = [
                source_part(loaded.conn, root),
                source_part(loaded.conn, root, imported=recorded),
                source_part(loaded.conn, root, folder=True, imported=recorded),
            ]

        assert found == [module] * 3

    @pytest.mark.parametrize(
        ("held", "keyed"),
        [
            # Under its package name: the name keys it
            ("acme.test_utils", "acme.test_utils"),
            # Only under a name that is not its package name (a namespace package's
            # name above the chain, or a rootless basename): the path keys it
            ("ns.acme.test_utils", "../acme/test_utils.py"),
            ("test_utils", "../acme/test_utils.py"),
        ],
    )
    def test_a_package_module_held_under_another_name_keeps_its_path(self, tmp_path, held, keyed):
        """
        A recorded package module that sys.modules does not hold under its package
        name was imported under a name pytest derived otherwise, so rule 3 keys it
        by its path.
        """
        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "__init__.py").write_text("")
        path = tmp_path / "acme" / "test_utils.py"
        path.write_text(FIXTURES_SOURCE)
        root = tmp_path / "tests"
        root.mkdir()
        recorded = {os.path.normcase(os.path.realpath(path))}

        with imported(path, held) as loaded:
            found = source_part(loaded.conn, root, imported=recorded)

        assert found == keyed

    def test_another_file_under_the_package_name_does_not_count(self, tmp_path):
        """
        The package name must hold that very file: another file under it (a second
        checkout's acme package first on sys.path) leaves rule 3 to key it by its path.
        """
        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "__init__.py").write_text("")
        path = tmp_path / "acme" / "test_utils.py"
        path.write_text(FIXTURES_SOURCE)
        other = tmp_path / "elsewhere" / "acme" / "test_utils.py"
        other.parent.mkdir(parents=True)
        other.write_text(FIXTURES_SOURCE)
        root = tmp_path / "tests"
        root.mkdir()
        recorded = {os.path.normcase(os.path.realpath(path))}

        with imported(path, "test_utils") as loaded, imported(other, "acme.test_utils"):
            found = source_part(loaded.conn, root, imported=recorded)

        assert found == "../acme/test_utils.py"

    @pytest.mark.parametrize(
        ("layout", "held", "keyed"),
        [
            # The package's own __init__.py, named like a test module by python_files
            (("acme/__init__.py",), "acme", "acme"),
            # A folder whose name is not an identifier ends the chain, as in pytest,
            # whose importlib mode then makes a name from the path
            (("my-acme/__init__.py", "my-acme/test_utils.py"), "my-acme.test_utils", None),
            # No __init__.py: a rootless basename
            (("other/test_utils.py",), "test_utils", None),
        ],
    )
    def test_the_package_name_follows_pytests_package_path(self, tmp_path, layout, held, keyed):
        """The chain of folders with an __init__.py, as _pytest.pathlib.resolve_package_path."""
        for file in layout:
            (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file).write_text(FIXTURES_SOURCE)
        path = tmp_path / layout[-1]
        root = tmp_path / "tests"
        root.mkdir()
        recorded = {os.path.normcase(os.path.realpath(path))}

        with imported(path, held) as loaded:
            found = source_part(loaded.conn, root, test_files=["*.py"], imported=recorded)

        assert found == (keyed or path_part(path, root))

    def test_a_package_module_in_a_linked_folder_has_its_spelled_package_name(self, tmp_path):
        """
        The chain is read from the path as it is spelled and from the real path: a
        module whose folder is linked into a package (acme/linked -> ../shared, a
        package too) is held under acme.linked.test_utils or shared.test_utils.
        """
        base = tmp_path / "b"
        (base / "shared").mkdir(parents=True)
        (base / "shared" / "__init__.py").write_text("")
        (base / "shared" / "test_utils.py").write_text(FIXTURES_SOURCE)
        (base / "acme").mkdir()
        (base / "acme" / "__init__.py").write_text("")
        try:
            os.symlink(base / "shared", base / "acme" / "linked", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")
        spelled = base / "acme" / "linked" / "test_utils.py"
        root = tmp_path / "a" / "tests"
        root.mkdir(parents=True)
        recorded = {os.path.normcase(os.path.realpath(spelled))}

        found = []
        for module in ("acme.linked.test_utils", "shared.test_utils", "test_utils"):
            with imported(spelled, module) as loaded:
                found.append(source_part(loaded.conn, root, imported=recorded))

        assert found == [
            "acme.linked.test_utils",
            "shared.test_utils",
            "../../b/shared/test_utils.py",
        ]

    def test_the_files_the_session_imported_count_for_its_own_config_only(self, tmp_path):
        """
        definition_part() reads the files the active session imported by their paths
        for that session's config; another config (or none) sees none of them.
        """
        path = tmp_path / "other" / "test_utils.py"
        path.parent.mkdir()
        path.write_text(FIXTURES_SOURCE)
        config = _ini_config(tmp_path / "proj")
        state = runtime.push(config)
        try:
            state.imported_files.add(os.path.normcase(os.path.realpath(path)))
            with imported(path, "test_utils") as loaded:
                found = [
                    definition_part(loaded.conn, config),
                    definition_part(loaded.conn, _ini_config(tmp_path / "proj")),
                    definition_part(loaded.conn, None),
                ]
        finally:
            runtime.pop()

        assert found == ["../other/test_utils.py", "test_utils", "test_utils"]

    def test_a_folder_linked_into_the_rootdir_is_inside_it_as_spelled(self, tmp_path):
        """
        A file reached through a folder linked into the checkout from a place that
        does not move with it is inside the rootdir as it is spelled, so it is keyed
        by that path; reached by its real path, it is outside.
        """
        shared = tmp_path / "shared"
        shared.mkdir()
        path = shared / "test_utils.py"
        path.write_text(FIXTURES_SOURCE)
        proj = tmp_path / "deep" / "proj"
        proj.mkdir(parents=True)
        try:
            os.symlink(shared, proj / "linked", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")

        found = []
        for spelled in (proj / "linked" / "test_utils.py", path):
            with imported(spelled, "test_utils") as loaded:
                found.append(definition_part(loaded.conn, _ini_config(proj)))

        assert found == ["linked/test_utils.py", "test_utils"]

    def test_a_partial_counts_as_the_function_it_wraps(self, tmp_path):
        path = tmp_path / "conftest.py"
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_partial_fixtures")

        assert _fixture_definition(functools.partial(module.conn), _config(tmp_path)) == (
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
            found.append(_fixture_definition(namespace["made"], _config(tmp_path)))

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


class TestImportedFiles:
    """
    The files a session records as imported by their paths
    (SessionState.imported_files): its test modules (tests/integration), its
    conftest.py files and the strategy files the plugin loaded (TestFileStream).
    """

    def test_a_conftest_registered_under_its_path_is_recorded(self, tmp_path):
        path = tmp_path / "other" / "conftest.py"
        path.parent.mkdir()
        path.write_text("")
        manager = object()
        state = runtime.push(SimpleNamespace(pluginmanager=manager))
        try:
            conftest = load(path, "conftest")
            hook = PytestStrategyPlugin().pytest_plugin_registered
            # A plugin registered by its module's name (-p acme.conftest), another
            # session's plugin manager, and a plugin that is not a module
            hook(plugin=conftest, plugin_name="acme.conftest", manager=manager)
            hook(plugin=conftest, plugin_name=str(path), manager=object())
            hook(plugin=SimpleNamespace(__file__=str(path)), plugin_name=str(path), manager=manager)
            before = set(state.imported_files)
            hook(plugin=conftest, plugin_name=str(path), manager=manager)
        finally:
            runtime.pop()

        assert before == set()
        assert state.imported_files == {os.path.normcase(os.path.realpath(path))}

    def test_only_a_conftest_counts(self, tmp_path):
        """A plugin module named otherwise is imported by its name (pytest_plugins, -p)."""
        path = tmp_path / "helpers.py"
        path.write_text("")
        manager = object()
        state = runtime.push(SimpleNamespace(pluginmanager=manager))
        try:
            PytestStrategyPlugin().pytest_plugin_registered(
                plugin=load(path, "helpers"), plugin_name=str(path), manager=manager
            )
        finally:
            runtime.pop()

        assert state.imported_files == set()


class TestSourcePart:
    """Where a factory is defined, as the export stream's folder."""

    def test_the_folder_of_its_file(self, tmp_path):
        path = tmp_path / "tests" / "a" / "strategies.py"
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURES_SOURCE)
        module = load(path, "ps_unit_source_folder")

        assert source_part(module.conn, tmp_path, folder=True) == "tests/a"
        assert source_part(module.conn, tmp_path) == "tests/a/strategies.py"

    def test_outside_a_session_a_module_is_named_by_its_module(self, tmp_path):
        """
        Without a session there is no rootdir, testpaths or file imported by its
        path: a strategy file imported by its module's name has that name, and one
        that sys.modules does not have under its name its folder's absolute path.
        """
        path = tmp_path / "strategies.py"
        path.write_text(FIXTURES_SOURCE)

        with imported(path, "strategies") as module:
            named = (
                source_part(module.conn, None, folder=True),
                definition_part(module.conn, None),
            )
        loaded = load(path, "ps_unit_outside_a_session")

        assert named == ("strategies", "strategies")
        assert (
            source_part(loaded.conn, None, folder=True)
            == Path(os.path.realpath(tmp_path)).as_posix()
        )

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
