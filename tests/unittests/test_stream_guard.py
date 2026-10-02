"""
Tests for the guard on draws outside the rows' streams (D5).

While the resolver generates a test's rows, each drawn argument draws from a
generator of its own, so the ambient generator (``RNG._ambient``) changes only when
something else draws from it: a constraint that calls ``RNG.*``, or an RNG type that
draws from a generator kept from the factory. The resolver then emits one
``PytestStrategiesWarning`` that names the strategy and the test.

The runs under pytest (the warnings summary, ``filterwarnings = error``) are in
tests/integration/test_stream_guard_integration.py.
"""

import warnings
from unittest.mock import MagicMock

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    PytestStrategiesWarning,
    RNGInteger,
    RNGSequence,
    RNGType,
    Series,
    TestArg,
)
from pytest_strategy._resolver import build_parametrization
from pytest_strategy._streams import StreamKey
from pytest_strategy.rng import _Stream

MESSAGE = "something drew from the plugin's generator while the rows were generated"


class Kept(RNGType[int]):
    """An RNG type that draws from a generator it was given, not from RNG.generator()."""

    def __init__(self, generator):
        self.generator = generator

    def generate(self):
        return self.generator.randint(0, 10**9)

    @property
    def python_type(self):
        return int


class AtDrawTime(RNGType[int]):
    """An RNG type that keeps the contract: it calls RNG.generator() when it draws."""

    def generate(self):
        return RNG.generator().randint(0, 10**9)

    @property
    def python_type(self):
        return int


class Reseeding(RNGInteger):
    """An RNG type that reseeds the generator it draws from: its argument's."""

    def generate(self):
        RNG.seed(5)
        return super().generate()


def _test_fn(a, b):
    pass


class _Bench:
    def test_m(self, a, b):
        pass


def _config(**options):
    """Return a stand-in config whose getoption() serves the given CLI options."""
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    return config


def build(factory, *, test_fn=_test_fn, config=None, name="guard"):
    """
    Resolve ``factory`` for ``test_fn`` and return the parametrization and the
    PytestStrategiesWarnings it emitted.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parametrization = build_parametrization(
            name,
            factory,
            test_fn,
            config=config,
            pytest_fixtures=set(),
            test_key="tests/test_g.py::test_g",
        )
    return parametrization, [w for w in caught if issubclass(w.category, PytestStrategiesWarning)]


def b_arg():
    return TestArg("b", rng_type=RNGInteger(0, 10**9))


def kept_rng(rng):
    """An RNG type that draws from the factory's rng."""
    return Parameter(TestArg("a", rng_type=Kept(rng)), b_arg())


def kept_generator():
    """An RNG type that draws from RNG.generator() as the factory saw it."""
    return Parameter(TestArg("a", rng_type=Kept(RNG.generator())), b_arg())


def constraint_calls_rng():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9)),
        b_arg(),
        vector_constraints={"coin": lambda v: RNG.integer(0, 1) >= 0},
    )


def constraint_draws_from_the_generator():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9)),
        b_arg(),
        vector_constraints={"coin": lambda v: RNG.generator().random() >= 0},
    )


DRAWING = [
    kept_rng,
    kept_generator,
    constraint_calls_rng,
    constraint_draws_from_the_generator,
]


# ---------------------------------------------------------------------------
# What the guard reports
# ---------------------------------------------------------------------------


class TestTheGuardWarns:
    @pytest.mark.parametrize("factory", DRAWING)
    def test_once_for_a_draw_outside_the_streams(self, factory):
        _, caught = build(factory)

        assert len(caught) == 1
        assert str(caught[0].message).startswith(f"Strategy 'guard' (_test_fn): {MESSAGE}")

    def test_it_points_at_the_test(self):
        _, caught = build(kept_rng)

        assert (caught[0].filename, caught[0].lineno) == (
            __file__,
            _test_fn.__code__.co_firstlineno,
        )

    def test_it_names_a_test_method_by_its_class(self):
        _, caught = build(kept_rng, test_fn=_Bench.test_m, name="burst")

        assert len(caught) == 1
        assert str(caught[0].message).startswith(f"Strategy 'burst' (_Bench.test_m): {MESSAGE}")

    def test_it_says_what_to_do(self):
        _, caught = build(constraint_calls_rng)

        message = str(caught[0].message)
        assert "a vector constraint that calls RNG.*" in message
        assert "a generator kept from the factory (rng)" in message
        assert message.endswith(
            "Draw only in an RNG type's generate(), from RNG.generator() or the RNG.* helpers."
        )

    def test_once_whatever_the_number_of_rows(self):
        def both(rng):
            return Parameter(
                TestArg("a", rng_type=Kept(rng)),
                b_arg(),
                vector_constraints={"coin": lambda v: RNG.boolean() or True},
                nsamples=50,
            )

        parametrization, caught = build(both)

        assert len(parametrization.values) == 50
        assert len(caught) == 1

    @pytest.mark.parametrize(
        ("options", "per_sequence"),
        [
            ({}, False),
            ({}, True),
            ({"nsamples": "auto"}, False),
        ],
        ids=["finite", "per_sequence_samples", "auto"],
    )
    def test_in_every_kind_of_random_row(self, options, per_sequence):
        def enumerated(rng):
            return Parameter(
                TestArg("ch", rng_type=Series([0, 1])),
                TestArg("dev", rng_type=RNGSequence(["x", "y"])),
                TestArg("a", rng_type=Kept(rng)),
                nsamples=2,
                per_sequence_samples=per_sequence,
            )

        def test_fn(ch, dev, a):
            pass

        parametrization, caught = build(enumerated, test_fn=test_fn, config=_config(**options))

        assert parametrization.values
        assert len(caught) == 1

    def test_the_rows_are_generated_all_the_same(self):
        # What it warns about: the kept rng continues the ambient generator, so its
        # values repeat from the same state, and move with whatever drew before
        state = RNG._ambient.getstate()
        first, _ = build(kept_rng)
        RNG._ambient.setstate(state)
        again, _ = build(kept_rng)
        RNG._ambient.setstate(state)
        RNG._ambient.random()
        moved, _ = build(kept_rng)

        assert [info.values for info in again.infos] == [info.values for info in first.infos]
        assert [info.values.a for info in moved.infos] != [info.values.a for info in first.infos]
        # The other argument draws from its own streams
        assert [info.values.b for info in moved.infos] == [info.values.b for info in first.infos]

    def test_under_an_error_filter_it_fails_the_strategy(self):
        RNG.seed(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PytestStrategiesWarning)
            with pytest.raises(ValueError) as excinfo:
                build_parametrization(
                    "guard", kept_rng, _test_fn, config=None, pytest_fixtures=set()
                )

        assert str(excinfo.value).startswith(
            f"Error generating samples for strategy 'guard': Strategy 'guard' (_test_fn): "
            f"{MESSAGE}"
        )
        assert isinstance(excinfo.value.__cause__, PytestStrategiesWarning)
        assert RNG.get_seed() == 3


# ---------------------------------------------------------------------------
# What the guard leaves alone
# ---------------------------------------------------------------------------


def plain():
    return Parameter(TestArg("a", rng_type=RNGInteger(0, 10**9)), b_arg())


def at_draw_time():
    return Parameter(TestArg("a", rng_type=AtDrawTime()), b_arg())


def with_a_predicate():
    return Parameter(TestArg("a", rng_type=RNGInteger(0, 10**9, lambda x: x % 2 == 0)), b_arg())


def validator_draws():
    # A validator runs on its argument's stream
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9), validator=lambda x: RNG.boolean() or True),
        b_arg(),
    )


def rng_type_reseeds():
    return Parameter(TestArg("a", rng_type=Reseeding(0, 10**9)), b_arg())


def factory_draws(rng):
    rng.random()
    RNG.integer(0, 9)
    return Parameter(TestArg("a", value=rng.random()), b_arg())


def constraint_without_draws():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9)),
        b_arg(),
        vector_constraints={"ordered": lambda v: v.a <= v.b or True},
    )


def constraint_on_a_stream_of_its_own():
    # As a strategy file imported from inside a constraint: its stream puts the
    # ambient generator back
    def own_stream(v):
        with _Stream(StreamKey.root(1, "file", "strategies.py")):
            RNG.integer(0, 9)
        return True

    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 10**9)), b_arg(), vector_constraints=[own_stream]
    )


class TestNoWarning:
    @pytest.mark.parametrize(
        "factory",
        [
            plain,
            at_draw_time,
            with_a_predicate,
            validator_draws,
            rng_type_reseeds,
            factory_draws,
            constraint_without_draws,
            constraint_on_a_stream_of_its_own,
        ],
    )
    def test_draws_on_the_streams(self, factory):
        parametrization, caught = build(factory)

        assert parametrization.values
        assert caught == []

    @pytest.mark.parametrize("factory", [kept_rng, constraint_calls_rng])
    def test_without_random_rows(self, factory):
        def directed(rng):
            param = factory(rng) if factory is kept_rng else factory()
            param.add_directed_vector("zeros", (0, 0))
            return param

        parametrization, caught = build(directed, config=_config(vector_mode="directed_only"))

        assert parametrization.ids == ["directed-zeros"]
        assert caught == []

    def test_when_the_generation_fails(self):
        def rejecting(rng):
            return Parameter(
                TestArg("a", rng_type=Kept(rng)),
                b_arg(),
                vector_constraints={"never": lambda v: RNG.boolean() and False},
                max_retries=3,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Could not generate random row 0"):
                build_parametrization(
                    "guard", rejecting, _test_fn, config=None, pytest_fixtures=set()
                )

        assert caught == []
