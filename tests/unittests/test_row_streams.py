"""
Tests for row-stable streams (streams v1, D5): every random and exhaustive row draws
each argument from a stream of its own, keyed by the generation call's key T, the
row's identity (pos and j) and the argument's name. A row's values therefore do not
depend on n, on the other rows, on the other arguments, or on a constraint that
accepts it, and one row can be computed on its own.
"""

import enum
import random

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGValueError,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    TestArg,
    Vector,
)
from pytest_strategy._resolver import _fallback_test_key, build_parametrization
from pytest_strategy._runtime import runtime
from pytest_strategy._streams import VERSION, StreamKey
from pytest_strategy._vector import VectorInfo
from pytest_strategy.parameters import _ArgRandom

# The key of a generation call, as the plugin builds T for a strategy and a test
KEY = StreamKey.root(1, "test", "rs", "test_rs.py::test_rs")


@pytest.fixture(autouse=True)
def _seed():
    RNG.seed(1234)


class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def drawn_args():
    """One drawing argument per built-in RNG type, a predicate and a static argument."""
    return [
        TestArg("i", rng_type=RNGInteger(0, 10**6)),
        TestArg("f", rng_type=RNGFloat(0.0, 1.0)),
        TestArg("b", rng_type=RNGBoolean()),
        TestArg("c", rng_type=RNGChoice(list("abcdefgh"))),
        TestArg("e", rng_type=RNGEnum(Color)),
        TestArg("s", rng_type=RNGString(min_length=1, max_length=8)),
        TestArg("wi", rng_type=RNGWeightedInteger({(0, 9): 1.0, (100, 109): 1.0})),
        TestArg("wf", rng_type=RNGWeightedFloat({(0.0, 1.0): 1.0, (10.0, 11.0): 1.0})),
        # An RNGSequence is drawn, not enumerated, in a finite run
        TestArg("q", rng_type=RNGSequence([1, 2, 3, 4, 5])),
        TestArg("p", rng_type=RNGInteger(0, 1000, predicate=lambda x: x % 7 == 0)),
        TestArg("k", value=7),
    ]


def series_param(**kwargs):
    """Two Series arguments around the drawing ones."""
    return Parameter(
        TestArg("ch", rng_type=Series([0, 1, 2])),
        *drawn_args(),
        TestArg("dev", rng_type=Series(["a", "b"])),
        **kwargs,
    )


def sequence_param(**kwargs):
    """A Series and an RNGSequence, enumerated with per_sequence_samples or auto."""
    return Parameter(
        TestArg("ch", rng_type=Series([0, 1])),
        TestArg("dev", rng_type=RNGSequence(["a", "b", "c"])),
        *drawn_args(),
        **kwargs,
    )


def by_identity(rows):
    """Return each random or exhaustive row's values by name, keyed by (pos, j)."""
    return {
        (row.pos, row.j): row.values._asdict()
        for row in rows
        if row.kind in ("random", "exhaustive")
    }


def by_name(rows):
    """Return each row's values by argument name."""
    return [row._asdict() for row in rows]


def column(rows, name):
    """Return one argument's values, row by row."""
    return [getattr(row, name) for row in rows]


def random_rows(param, n=20, **kwargs):
    """Return the n random rows of ``param`` under KEY."""
    return param.generate_vectors(n, mode="random_only", _key=KEY, **kwargs)


# ---------------------------------------------------------------------------
# The keys
# ---------------------------------------------------------------------------


class TestTheKeys:
    """Each drawn argument of a row draws from T/"row"/pos/j/argname (pos flattened)."""

    def test_a_plain_row(self):
        param = Parameter(TestArg("a", rng_type=RNGInteger(0, 10**6)), TestArg("b", value=1))

        rows = random_rows(param, 3)

        for j, row in enumerate(rows):
            seed = KEY.child("row", j, "a").seed_int()
            assert row.a == random.Random(seed).randint(0, 10**6)

    def test_a_row_of_enumerated_values(self):
        param = Parameter(
            TestArg("dev", rng_type=Series(["x", "y"])),
            TestArg("a", rng_type=RNGInteger(0, 10**6)),
            TestArg("ch", rng_type=Series([3, 4])),
        )

        for row in param._generate_rows(8, key=KEY):
            # pos is sorted by name: ch before dev
            assert row.pos == (("ch", f"i:{row.values.ch}"), ("dev", f"s:{row.values.dev}"))
            ch, dev = row.pos
            seed = KEY.child("row", *ch, *dev, row.j, "a").seed_int()
            assert row.values.a == random.Random(seed).randint(0, 10**6)

    def test_the_order_of_an_rng_sequence(self):
        param = Parameter(TestArg("ch", rng_type=RNGSequence(list(range(10)))))

        rows = param._generate_rows(0, exhaustive=True, key=KEY)

        order = random.Random(KEY.child("order", "ch").seed_int()).sample(range(10), 10)
        assert [row.values.ch for row in rows] == order

    def test_each_row_has_values_of_its_own(self):
        plain = random_rows(Parameter(*three_args()), 20)
        assert len({tuple(row) for row in plain}) == 20

        series = Parameter(TestArg("ch", rng_type=Series([0, 0, 1])), *three_args())
        rows = series._generate_rows(6, key=KEY)
        drawn = [row.values[1:] for row in rows]
        assert len(set(drawn)) == 6


# ---------------------------------------------------------------------------
# n and the other rows
# ---------------------------------------------------------------------------


class TestMoreRowsKeepTheFirstOnes:
    def test_plain_rows(self):
        param = Parameter(*drawn_args())

        assert random_rows(param, 50)[:10] == random_rows(param, 10)

    def test_plain_rows_of_direct_calls(self):
        param = Parameter(*drawn_args())

        RNG.seed(5)
        short = param.generate_vectors(10)
        RNG.seed(5)
        long = param.generate_vectors(50)

        assert long[:10] == short

    def test_finite_series_rows(self):
        param = series_param()

        assert param.generate_vectors(50, _key=KEY)[:10] == param.generate_vectors(10, _key=KEY)

    def test_per_sequence_rows_per_combination(self):
        param = sequence_param(per_sequence_samples=True)

        def per_combination(n):
            grouped = {}
            for row in param._generate_rows(n, key=KEY):
                grouped.setdefault(row.pos, []).append(row.values)
            return grouped

        short, long = per_combination(10), per_combination(50)

        # ch x dev x q: per_sequence_samples enumerates every RNGSequence too
        assert len(short) == 2 * 3 * 5
        assert short.keys() == long.keys()
        for pos, rows in short.items():
            assert len(long[pos]) == 50
            assert long[pos][:10] == rows

    def test_the_mode_and_the_directed_vectors_do_not_move_random_rows(self):
        param = Parameter(*drawn_args())
        rows = random_rows(param, 5)
        with_vectors = Parameter(*drawn_args(), directed_vectors={"one": rows[0]})

        for mode in ("all", "mixed"):
            got = with_vectors.generate_vectors(5, mode=mode, _key=KEY)
            assert got[1:] == rows
        assert random_rows(with_vectors, 5) == rows


# ---------------------------------------------------------------------------
# One row on its own
# ---------------------------------------------------------------------------


class TestOneRowOnItsOwn:
    @pytest.mark.parametrize(
        ("make", "n", "exhaustive"),
        [
            (lambda: Parameter(*drawn_args()), 20, False),
            (series_param, 15, False),
            (lambda: sequence_param(per_sequence_samples=True), 3, False),
            (sequence_param, 0, True),
            (
                lambda: Parameter(*drawn_args(), vector_constraints={"even": lambda v: v.i % 2}),
                20,
                False,
            ),
        ],
        ids=["plain", "series", "per_sequence", "exhaustive", "constraint"],
    )
    def test_every_row(self, make, n, exhaustive):
        param = make()
        rows = param._generate_rows(n, exhaustive=exhaustive, mode="random_only", key=KEY)
        assert rows

        for row in rows:
            alone = param._generate_row(KEY, row.pos, row.j)
            assert type(alone) is param.vector_type
            assert alone == row.values

    def test_pos_in_any_order(self):
        param = series_param()
        row = param._generate_rows(10, key=KEY)[7]

        assert param._generate_row(KEY, reversed(row.pos), row.j) == row.values

    def test_constraints_turned_off(self):
        param = Parameter(*drawn_args(), vector_constraints={"even": lambda v: v.i % 2 == 0})
        rows = param._generate_rows(10, constraints_off=["even"], key=KEY)

        assert [param._generate_row(KEY, (), k, constraints_off=["even"]) for k in range(10)] == [
            row.values for row in rows
        ]

    @pytest.mark.parametrize(
        ("pos", "message"),
        [
            ((("nope", "i:0"),), "No argument named 'nope'"),
            ((("i", "i:0"),), "'i' is not a Series or RNGSequence argument"),
            ((("ch", "i:9"),), "'ch' has no value with the token 'i:9'"),
            ((("ch", "i:0"), ("ch", "i:1")), "pos names an argument twice"),
        ],
    )
    def test_a_pos_the_parameter_cannot_have(self, pos, message):
        with pytest.raises(KeyError, match=message):
            series_param()._generate_row(KEY, pos, 0)

    def test_a_row_the_constraints_reject(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": lambda v: False},
            max_retries=3,
        )

        with pytest.raises(ValueError, match=r"random row 4 \(ch=1\) after max_retries=3.*never=3"):
            param._generate_row(KEY, [("ch", "i:1")], 4)

    def test_an_enumerated_row_the_constraints_reject(self):
        """A row with nothing to redraw is rejected for its values."""
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            vector_constraints={"never": lambda v: False},
        )

        with pytest.raises(
            ValueError, match=r"random row 0 \(ch=1\): the constraints reject its values"
        ):
            param._generate_row(KEY, [("ch", "i:1")], 0)


# ---------------------------------------------------------------------------
# The other arguments
# ---------------------------------------------------------------------------


def three_args():
    return [
        TestArg("a", rng_type=RNGInteger(0, 10**6)),
        TestArg("b", rng_type=RNGFloat(0.0, 1.0)),
        TestArg("c", rng_type=RNGChoice(list("abcdefgh"))),
    ]


class TestTheOtherArguments:
    @pytest.mark.parametrize("position", [0, 1, 3], ids=["start", "middle", "end"])
    def test_adding_a_drawn_argument(self, position):
        base = by_name(random_rows(Parameter(*three_args())))
        args = three_args()
        args.insert(position, TestArg("extra", rng_type=RNGInteger(0, 255)))

        added = by_name(random_rows(Parameter(*args)))

        assert [{n: v for n, v in row.items() if n != "extra"} for row in added] == base

    def test_reordering_the_drawn_arguments(self):
        base = by_name(random_rows(Parameter(*three_args())))

        assert by_name(random_rows(Parameter(*reversed(three_args())))) == base

    @pytest.mark.parametrize(
        "rng_type",
        [
            RNGInteger(0, 5),
            RNGFloat(0.0, 1.0, predicate=lambda x: x > 0.5),
            RNGString(length=4),
        ],
        ids=["another_type", "a_predicate", "a_string"],
    )
    def test_changing_one_drawn_arguments_type(self, rng_type):
        base = random_rows(Parameter(*three_args()))
        args = three_args()
        args[1] = TestArg("b", rng_type=rng_type)

        changed = random_rows(Parameter(*args))

        assert column(changed, "a") == column(base, "a")
        assert column(changed, "c") == column(base, "c")
        assert column(changed, "b") != column(base, "b")

    def test_adding_a_drawn_argument_to_series_rows(self):
        series = TestArg("ch", rng_type=Series([0, 1, 2]))
        base = by_identity(Parameter(series, *three_args())._generate_rows(9, key=KEY))
        added = Parameter(series, TestArg("x", rng_type=RNGInteger(0, 9)), *three_args())

        rows = by_identity(added._generate_rows(9, key=KEY))

        assert rows.keys() == base.keys()
        assert [{n: v for n, v in row.items() if n != "x"} for row in rows.values()] == list(
            base.values()
        )

    def test_swapping_two_series_arguments(self):
        ch = TestArg("ch", rng_type=Series([0, 1, 2]))
        dev = TestArg("dev", rng_type=Series(["a", "b"]))
        before = Parameter(ch, dev, *three_args())
        after = Parameter(dev, ch, *three_args())

        # Two whole cycles: the same rows, in another order
        assert by_identity(after._generate_rows(12, key=KEY)) == by_identity(
            before._generate_rows(12, key=KEY)
        )
        # Part of a cycle: the rows both runs have keep their values
        first, second = (by_identity(p._generate_rows(4, key=KEY)) for p in (before, after))
        common = first.keys() & second.keys()
        assert common
        assert all(first[k] == second[k] for k in common)

    def test_an_argument_that_draws_often_leaves_the_others(self):
        """Each argument has its own generator: its draws never move another's."""
        base = random_rows(Parameter(*three_args()))
        greedy = TestArg("extra", rng_type=RNGInteger(0, 9, predicate=lambda x: x >= 5))

        rows = random_rows(Parameter(greedy, *three_args()))

        assert [{n: v for n, v in r._asdict().items() if n != "extra"} for r in rows] == by_name(
            base
        )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_a_constraint_keeps_the_rows_it_accepts_and_redraws_the_others(self):
        param = Parameter(*three_args())
        base = random_rows(param, 30)

        param.add_constraint(lambda v: v.a % 2 == 0, name="even")
        constrained = random_rows(param, 30)

        kept = redrawn = 0
        for before, after in zip(base, constrained, strict=True):
            if before.a % 2 == 0:
                assert after == before
                kept += 1
            else:
                assert after != before
                assert after.a % 2 == 0
                redrawn += 1
        assert kept and redrawn

        # Turned off for a call, or removed: the values without it
        assert random_rows(param, 30, constraints_off=["even"]) == base
        param.remove_constraint("even")
        assert random_rows(param, 30) == base

    def test_on_series_rows(self):
        param = Parameter(TestArg("ch", rng_type=Series([0, 1, 2])), *three_args())
        base = by_identity(param._generate_rows(12, key=KEY))

        param.add_constraint(lambda v: v.a % 3 != 0, name="no_3")
        constrained = by_identity(param._generate_rows(12, key=KEY))

        assert constrained.keys() == base.keys()
        for identity, before in base.items():
            after = constrained[identity]
            assert after == before if before["a"] % 3 != 0 else after["a"] % 3 != 0

    def test_a_constraint_on_exhaustive_rows(self):
        param = sequence_param()
        base = by_identity(param._generate_rows(0, exhaustive=True, key=KEY))

        param.add_constraint(lambda v: v.i % 2 == 0, name="even")
        constrained = by_identity(param._generate_rows(0, exhaustive=True, key=KEY))

        assert constrained.keys() == base.keys()
        for identity, before in base.items():
            after = constrained[identity]
            assert after == before if before["i"] % 2 == 0 else after["i"] % 2 == 0


# ---------------------------------------------------------------------------
# Series values
# ---------------------------------------------------------------------------


def with_series(values):
    return Parameter(TestArg("ch", rng_type=Series(values)), *three_args())


class TestSeriesValues:
    @pytest.mark.parametrize(
        "values", [[1, 2, 3, 4], [1, 9, 2, 3], [0, 1, 2, 3]], ids=["append", "insert", "prepend"]
    )
    def test_adding_a_scalar_value_keeps_the_existing_rows(self, values):
        base = by_identity(with_series([1, 2, 3])._generate_rows(6, key=KEY))

        rows = by_identity(with_series(values)._generate_rows(8, key=KEY))

        assert base.keys() <= rows.keys()
        assert all(rows[identity] == row for identity, row in base.items())

    def test_adding_a_value_keeps_the_exhaustive_rows(self):
        base = by_identity(with_series([1, 2, 3])._generate_rows(0, exhaustive=True, key=KEY))

        rows = by_identity(with_series([1, 2, 3, 4])._generate_rows(0, exhaustive=True, key=KEY))

        assert all(rows[identity] == row for identity, row in base.items())

    @pytest.mark.parametrize("exhaustive", [False, True], ids=["finite", "auto"])
    def test_a_duplicated_value_gives_distinct_rows(self, exhaustive):
        rows = with_series([1, 1, 2])._generate_rows(3, exhaustive=exhaustive, key=KEY)

        assert [row.pos for row in rows] == [
            (("ch", "i:1"),),
            (("ch", "i:1~1"),),
            (("ch", "i:2"),),
        ]
        assert rows[0].values.ch == rows[1].values.ch == 1
        assert rows[0].values[1:] != rows[1].values[1:]


# ---------------------------------------------------------------------------
# --nsamples=auto
# ---------------------------------------------------------------------------


class Reversed(RNGSequence):
    """An RNGSequence walked backwards under --nsamples=auto."""

    def _auto_positions(self):
        return list(reversed(range(len(self.sequence))))


def auto_rows(*args):
    return Parameter(*args)._generate_rows(0, exhaustive=True, key=KEY)


class TestAutoOrder:
    def test_the_order_of_an_rng_sequence_depends_only_on_its_argument(self):
        def ch():
            return TestArg("ch", rng_type=RNGSequence(list(range(8))))

        alone = column([r.values for r in auto_rows(ch(), *three_args())], "ch")
        assert alone != sorted(alone)

        # Other drawn arguments, before and after it
        others = auto_rows(TestArg("y", rng_type=RNGInteger(0, 9)), ch(), *three_args())
        assert column([r.values for r in others], "ch") == alone

        # Another enumerated argument: each of its values walks ch in the same order
        dev = TestArg("dev", rng_type=RNGSequence(["a", "b"]))
        both = [r.values for r in auto_rows(dev, ch(), *three_args())]
        assert column(both[:8], "ch") == alone
        assert column(both[8:], "ch") == alone

    def test_the_values_of_a_combination_do_not_depend_on_the_order(self):
        def args(sequence_type):
            return (TestArg("ch", rng_type=sequence_type(list(range(6)))), *three_args())

        shuffled = auto_rows(*args(RNGSequence))
        declared = auto_rows(*args(Series))
        backwards = auto_rows(*args(Reversed))

        assert [row.values.ch for row in declared] == list(range(6))
        assert [row.values.ch for row in backwards] == list(reversed(range(6)))
        assert by_identity(shuffled) == by_identity(declared) == by_identity(backwards)


# ---------------------------------------------------------------------------
# The generator the code runs with
# ---------------------------------------------------------------------------


class Boom(RNGType):
    def generate(self):
        raise RuntimeError("boom")


class NoOrder(RNGSequence):
    def _auto_positions(self):
        raise RuntimeError("no order")


def _fails_validation(value):
    return False


def _raises(value):
    raise ZeroDivisionError("validator")


class TestTheAmbientGenerator:
    @pytest.mark.parametrize(
        ("make", "error"),
        [
            (
                lambda: Parameter(TestArg("a", rng_type=RNGInteger(0, 9), validator=_raises)),
                ZeroDivisionError,
            ),
            (
                lambda: Parameter(
                    TestArg("a", rng_type=RNGInteger(0, 9), validator=_fails_validation)
                ),
                ValueError,
            ),
            (lambda: Parameter(TestArg("a", rng_type=Boom())), RuntimeError),
            (
                lambda: Parameter(TestArg("a", rng_type=RNGInteger(0, 9, predicate=bool))),
                None,
            ),
            (
                lambda: Parameter(
                    TestArg("a", rng_type=RNGInteger(0, 9)),
                    vector_constraints={"boom": lambda v: 1 / 0},
                ),
                ZeroDivisionError,
            ),
        ],
        ids=[
            "raising_validator",
            "failing_validator",
            "raising_type",
            "fine",
            "raising_constraint",
        ],
    )
    def test_is_restored(self, make, error):
        ambient = RNG.generator()
        param = make()

        if error is None:
            param.generate_vectors(3)
        else:
            with pytest.raises(error):
                param.generate_vectors(3)

        assert RNG.generator() is ambient

    def test_is_restored_when_a_sequence_order_raises(self):
        ambient = RNG.generator()
        param = Parameter(
            TestArg("ch", rng_type=NoOrder([1, 2])), TestArg("a", rng_type=RNGInteger(0, 9))
        )

        with pytest.raises(RuntimeError, match="no order"):
            param._generate_rows(0, exhaustive=True, key=KEY)

        assert RNG.generator() is ambient

    def test_an_argument_draws_from_its_own_generator_and_a_constraint_from_the_ambient_one(
        self,
    ):
        ambient = RNG.generator()
        seen = {"a": set(), "b": set(), "constraint": set()}

        def watch(name):
            def validator(value):
                seen[name].add(id(RNG.generator()))
                return True

            return validator

        def constraint(v):
            seen["constraint"].add(id(RNG.generator()))
            return True

        param = Parameter(
            TestArg("a", rng_type=RNGInteger(0, 9), validator=watch("a")),
            TestArg("b", rng_type=RNGInteger(0, 9), validator=watch("b")),
            vector_constraints=[constraint],
        )
        param.generate_vectors(4, _key=KEY)

        assert seen["constraint"] == {id(ambient)}
        assert len(seen["a"]) == len(seen["b"]) == 1
        assert seen["a"] != seen["b"]
        assert id(ambient) not in seen["a"] | seen["b"]

    def test_a_static_argument_draws_nothing(self):
        def from_row(params):
            return [row.a for row in params.generate_vectors(5, _key=KEY)]

        alone = from_row(Parameter(TestArg("a", rng_type=RNGInteger(0, 10**6))))
        with_static = from_row(
            Parameter(TestArg("s", value=1), TestArg("a", rng_type=RNGInteger(0, 10**6)))
        )

        assert with_static == alone

    def test_reseed_seeds_like_seed(self):
        generator = _ArgRandom(0)
        # gauss() caches a second value, which seeding drops
        generator.gauss(0.0, 1.0)

        generator.reseed(2**127 + 12345)

        assert generator.getstate() == random.Random(2**127 + 12345).getstate()


# ---------------------------------------------------------------------------
# Direct calls
# ---------------------------------------------------------------------------


class TestDirectCalls:
    def test_consecutive_calls_differ_and_reseeding_repeats_them(self):
        param = Parameter(*three_args())

        RNG.seed(42)
        first, second = param.generate_vectors(10), param.generate_vectors(10)
        RNG.seed(42)

        assert first != second
        assert [param.generate_vectors(10), param.generate_vectors(10)] == [first, second]

    def test_generate_vector_is_the_first_random_row(self):
        param = Parameter(*drawn_args())

        assert param.generate_vector(_key=KEY) == random_rows(param, 1)[0]
        RNG.seed(9)
        single = param.generate_vector()
        RNG.seed(9)
        assert single == param.generate_vectors(1)[0]

    def test_generate_vector_draws_a_series_argument(self):
        """
        It draws every argument, so its row is not the first of a call that
        enumerates the Series (whose row 0 is pos ch=0).
        """
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("x", rng_type=RNGInteger(0, 10**9)),
        )

        row = param.generate_vector(_key=KEY)

        streams = KEY.child("row")
        assert row == (
            random.Random(streams.child(0, "ch").seed_int()).choice([0, 1, 2]),
            random.Random(streams.child(0, "x").seed_int()).randint(0, 10**9),
        )
        assert param._generate_rows(1, mode="random_only", key=KEY)[0].pos == (("ch", "i:0"),)

    def test_generate_exhaustive_takes_a_key(self):
        param = sequence_param()
        rows = param._generate_rows(0, exhaustive=True, key=KEY)

        assert param.generate_exhaustive(_key=KEY) == [row.values for row in rows]

    @pytest.mark.parametrize("seed", [1.5, "fast", b"\x00\x01", True])
    def test_a_seed_that_is_not_an_int(self, seed):
        param = Parameter(*three_args())

        RNG.seed(seed)
        rows = param.generate_vectors(3)
        RNG.seed(seed)

        assert param.generate_vectors(3) == rows

    def test_a_call_without_random_rows_draws_nothing(self):
        param = Parameter(*three_args(), directed_vectors={"zeros": (0, 0.0, "a")})
        state = RNG.generator().getstate()

        param.generate_vectors(5, mode="directed_only")
        param.generate_vectors(0, filter_by_name="zeros")
        # n=0 in the modes that draw random rows: the direct key is not drawn
        assert param.generate_vectors(0) == [(0, 0.0, "a")]
        param.generate_vectors(0, mode="mixed")
        param.generate_vectors(0, mode="random_only")
        Parameter(TestArg("ch", rng_type=Series([1, 2])), *three_args()).generate_vectors(0)
        sequence_param(per_sequence_samples=True).generate_vectors(0)

        assert RNG.generator().getstate() == state


# ---------------------------------------------------------------------------
# Through the resolver
# ---------------------------------------------------------------------------


def _test_fn(a, b, c):
    pass


class TestLike:
    def test_method(self):
        pass


class TestThroughTheResolver:
    def build(self, factory, **kwargs):
        return build_parametrization(
            "rs", factory, _test_fn, config=None, pytest_fixtures=set(), **kwargs
        )

    def test_rows_come_from_the_tests_key(self):
        param = Parameter(*three_args())

        infos = self.build(lambda: param, test_key="tests/test_x.py::test_x").infos

        key = StreamKey.root(runtime.run_seed(), "test", "rs", "tests/test_x.py::test_x")
        assert [info.values for info in infos] == [
            row.values for row in param._generate_rows(10, key=key)
        ]

    def test_the_test_key_selects_the_rows(self):
        param = Parameter(*three_args())

        one = self.build(lambda: param, test_key="t.py::TestA::test_m").infos
        other = self.build(lambda: param, test_key="t.py::TestB::test_m").infos

        assert [info.values for info in one] != [info.values for info in other]

    def test_without_a_node_id_the_key_is_the_location_and_qualified_name(self):
        assert _fallback_test_key(_test_fn, None) == f"{__name__}::_test_fn"
        assert (
            _fallback_test_key(TestLike.test_method, None) == f"{__name__}::TestLike::test_method"
        )

        param = Parameter(*three_args())
        infos = self.build(lambda: param).infos
        key = StreamKey.root(runtime.run_seed(), "test", "rs", f"{__name__}::_test_fn")
        assert [info.values for info in infos] == [
            row.values for row in param._generate_rows(10, key=key)
        ]

    def test_factory_draws_do_not_move_the_rows(self):
        param = Parameter(*three_args())

        def drawing():
            RNG.integer(0, 9)
            RNG.generator().random()
            return param

        plain = self.build(lambda: param, test_key="t.py::test").infos
        drawn = self.build(drawing, test_key="t.py::test").infos

        assert [info.values for info in drawn] == [info.values for info in plain]

    def test_the_rows_report_streams_v1(self):
        infos = self.build(lambda: Parameter(*three_args()), test_key="t.py::test").infos

        assert VERSION == 1
        assert {info.streams for info in infos} == {VERSION}
        assert VectorInfo.__dataclass_fields__["streams"].default == VERSION


def test_rows_are_vectors():
    param = series_param()

    assert all(isinstance(v, Vector) for v in param.generate_vectors(4, _key=KEY))


def test_an_rng_value_error_names_the_argument():
    """A predicate that rejects every draw still names its argument (in its stream)."""
    param = Parameter(TestArg("w", rng_type=RNGInteger(0, 9, predicate=lambda x: x > 9)))

    with pytest.raises(RNGValueError, match="Argument 'w' could not draw"):
        param.generate_vectors(1, _key=KEY)
