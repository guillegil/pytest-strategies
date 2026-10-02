"""
Tests for the row model (D2): Parameter._generate_rows() gives every row its kind,
name, index, pos and j in every mode, and _position_keys() gives each position of an
enumerated argument its label and token.
"""

import enum
import numbers
import sys
import warnings
from fractions import Fraction

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    PytestStrategiesWarning,
    RNGInteger,
    RNGSequence,
    RNGValueError,
    SequenceLike,
    Series,
    TestArg,
    Vector,
)
from pytest_strategy._streams import StreamKey
from pytest_strategy.parameters import (
    _auto_order,
    _auto_order_from,
    _describe_row,
    _position_keys,
    _Row,
)


@pytest.fixture(autouse=True)
def _seed():
    RNG.seed(1234)


class Color(enum.IntEnum):
    RED = 1
    BLUE = 2


class Mode(enum.StrEnum):
    FAST = "fast"
    SLOW = "slow"


class Perm(enum.Flag):
    R = 1
    W = 2


class Word:
    """A numbers.Integral that is not an int, as numpy's integers are."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

    def __repr__(self):
        return f"Word({self.value})"


numbers.Integral.register(Word)


class NoIndex:
    """A numbers.Integral whose __index__ raises."""

    def __index__(self):
        raise TypeError("no index")


numbers.Integral.register(NoIndex)


class Text(str):
    """A str subclass that is not an Enum."""


def helper():
    """A function value, labeled by its name."""


def identity(rows):
    """The identity of each row: kind, name, index, pos, j and labels."""
    return [(r.kind, r.name, r.index, r.pos, r.j, r.labels) for r in rows]


def keys(arg, sequence):
    """The (label, token) of each position."""
    return [(key.label, key.token) for key in _position_keys(arg, sequence)]


# ---------------------------------------------------------------------------
# Labels and tokens
# ---------------------------------------------------------------------------


class TestPositionKeys:
    def test_type_order(self):
        sequence = [
            None,
            True,
            Color.RED,
            Mode.FAST,
            42,
            Word(7),
            2.5,
            "fast",
            b"\x00\xff",
            (1, 2),
        ]

        assert keys("x", sequence) == [
            ("x=None", "n"),
            ("x=True", "b:True"),
            ("x=RED", "e:Color.RED"),
            ("x=FAST", "e:Mode.FAST"),
            ("x=42", "i:42"),
            ("x=7", "i:7"),
            ("x=2.5", f"f:{(2.5).hex()}"),
            ("x=fast", "s:fast"),
            ("x8", "y:00ff"),
            ("x9", "#9"),
        ]

    def test_an_int_enum_member_is_a_member_not_an_int(self):
        assert keys("ch", [Color.RED, 1]) == [("ch=RED", "e:Color.RED"), ("ch=1", "i:1")]

    def test_a_str_enum_member_is_a_member_not_a_str(self):
        assert keys("m", [Mode.FAST, "fast"]) == [("m=FAST", "e:Mode.FAST"), ("m=fast", "s:fast")]

    def test_a_bool_is_not_an_int(self):
        assert keys("b", [True, 1, False, 0]) == [
            ("b=True", "b:True"),
            ("b=1", "i:1"),
            ("b=False", "b:False"),
            ("b=0", "i:0"),
        ]

    def test_an_integral_that_is_not_an_int_goes_through_index(self):
        assert keys("w", [Word(-3), Word(2**70)]) == [
            ("w=-3", "i:-3"),
            (f"w={2**70}", f"i:{2**70}"),
        ]

    def test_a_real_that_is_not_a_float_goes_through_float(self):
        assert keys("r", [Fraction(1, 2)]) == [("r=0.5", "f:0x1.0000000000000p-1")]

    def test_a_flag_combination_is_a_member_and_a_flag_value_without_a_name_is_positional(self):
        assert keys("p", [Perm.R | Perm.W, Perm.R, Perm(0)]) == [
            ("p=R|W", "e:Perm.R|W"),
            ("p=R", "e:Perm.R"),
            ("p2", "#2"),
        ]

    def test_an_integral_whose_index_raises_is_keyed_by_position(self):
        assert keys("w", [NoIndex(), 1]) == [("w0", "#0"), ("w=1", "i:1")]

    def test_a_real_float_cannot_hold_is_keyed_by_position(self):
        assert keys("r", [Fraction(10**400), Fraction(1, 2)]) == [
            ("r0", "#0"),
            ("r=0.5", "f:0x1.0000000000000p-1"),
        ]

    def test_an_int_with_more_digits_than_str_allows_is_keyed_by_position(self):
        limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(640)
        try:
            big = 10**700
            assert keys("n", [big, big, 1]) == [("n0", "#0"), ("n1", "#1"), ("n=1", "i:1")]
        finally:
            sys.set_int_max_str_digits(limit)

    def test_a_str_subclass_is_a_str(self):
        assert keys("s", [Text("a"), "a"]) == [("s=a", "s:a"), ("s=a~1", "s:a~1")]

    def test_floats_are_keyed_by_their_hex(self):
        assert keys("f", [0.1, -0.0, 0.0, float("inf"), float("nan")]) == [
            ("f=0.1", "f:0x1.999999999999ap-4"),
            ("f=-0.0", "f:-0x0.0p+0"),
            ("f=0.0", "f:0x0.0p+0"),
            ("f=inf", "f:inf"),
            ("f=nan", "f:nan"),
        ]

    def test_an_int_and_an_equal_float_are_different_values(self):
        assert keys("n", [1, 1.0]) == [("n=1", "i:1"), ("n=1.0", "f:0x1.0000000000000p+0")]

    def test_a_repeated_value_gets_its_count(self):
        assert keys("ch", [1, 1, 2, 1]) == [
            ("ch=1", "i:1"),
            ("ch=1~1", "i:1~1"),
            ("ch=2", "i:2"),
            ("ch=1~2", "i:1~2"),
        ]

    def test_a_repeated_nan_gets_its_count(self):
        """Two NaN objects are the same value by their token, though not equal."""
        assert keys("f", [float("nan"), float("nan")]) == [
            ("f=nan", "f:nan"),
            ("f=nan~1", "f:nan~1"),
        ]

    def test_values_with_one_text_are_labeled_by_position(self):
        assert keys("ch", [1, "1", 2]) == [("ch0", "i:1"), ("ch1", "s:1"), ("ch=2", "i:2")]
        assert keys("v", [None, "None", True, "True"]) == [
            ("v0", "n"),
            ("v1", "s:None"),
            ("v2", "b:True"),
            ("v3", "s:True"),
        ]

    def test_every_position_of_the_values_with_one_text_is_positional(self):
        assert keys("ch", [1, "1", 1]) == [("ch0", "i:1"), ("ch1", "s:1"), ("ch2", "i:1~1")]

    @pytest.mark.parametrize(
        "text",
        ["", "a b", "a\tb", "a=b", "a~b", "a[0]", "x]", "\x00", "x" * 41],
        ids=[
            "empty",
            "space",
            "tab",
            "equals",
            "tilde",
            "brackets",
            "bracket",
            "unprintable",
            "41",
        ],
    )
    def test_a_str_the_id_cannot_show_is_labeled_by_position(self, text):
        assert keys("s", ["ok", text]) == [("s=ok", "s:ok"), ("s1", f"s:{text.replace('~', '~~')}")]

    def test_a_str_the_id_can_show(self):
        assert keys("s", ["x" * 40, "café", "fast-mode", "a.b/c:d"]) == [
            ("s=" + "x" * 40, "s:" + "x" * 40),
            ("s=café", "s:café"),
            ("s=fast-mode", "s:fast-mode"),
            ("s=a.b/c:d", "s:a.b/c:d"),
        ]

    def test_a_tilde_in_a_str_never_reads_as_a_repeat(self):
        tokens = [token for _, token in keys("s", ["a", "a", "a~1", "a~", "a~"])]

        assert tokens == ["s:a", "s:a~1", "s:a~~1", "s:a~~", "s:a~~~1"]
        assert len(set(tokens)) == len(tokens)

    def test_classes_and_functions_show_their_name_and_are_keyed_by_position(self):
        assert keys("fn", [int, len, helper, helper]) == [
            ("fn=int", "#0"),
            ("fn=len", "#1"),
            ("fn=helper", "#2"),
            ("fn=helper~1", "#3"),
        ]

    def test_other_objects_are_keyed_and_labeled_by_position(self):
        marker = object()

        assert keys("o", [(1, 2), [3], marker, bytearray(b"a"), 1j]) == [
            ("o0", "#0"),
            ("o1", "#1"),
            ("o2", "#2"),
            ("o3", "#3"),
            ("o4", "#4"),
        ]

    def test_bytes_have_a_token_but_no_text(self):
        assert keys("b", [b"", b"ab", b"ab"]) == [
            ("b0", "y:"),
            ("b1", "y:6162"),
            ("b2", "y:6162~1"),
        ]

    def test_an_enum_token_uses_the_qualname(self):
        class Local(enum.Enum):
            A = (1, 2)

        assert keys("e", [Local.A]) == [
            ("e=A", "e:TestPositionKeys.test_an_enum_token_uses_the_qualname.<locals>.Local.A")
        ]

    def test_inserting_a_scalar_keeps_the_other_positions_keys(self):
        before = dict(zip([1, 2, "x"], keys("ch", [1, 2, "x"])))
        after = dict(zip([0, 1, 2, "x"], keys("ch", [0, 1, 2, "x"])))

        assert all(after[value] == before[value] for value in before)

    def test_tokens_and_labels_are_unique_in_a_mixed_sequence(self):
        sequence = [1, 1, "1", 1.0, True, None, "None", (1,), (1,), b"1", Color.RED, helper]
        result = keys("v", sequence)

        assert len({label for label, _ in result}) == len(sequence)
        assert len({token for _, token in result}) == len(sequence)


class TestDescribeRow:
    def test_a_row_without_enumerated_arguments(self):
        assert _describe_row("random", 3) == "random row 3"

    def test_a_row_with_enumerated_arguments(self):
        assert _describe_row("exhaustive", 5, ("ch=2", "dev1")) == "exhaustive row 5 (ch=2, dev1)"

    def test_a_row_without_a_number(self):
        assert _describe_row("random", None) == "the row"


# ---------------------------------------------------------------------------
# Rows in every mode
# ---------------------------------------------------------------------------


def vectors_param(**kwargs):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 255)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        directed_vectors={"zeros": (0, 0), "max": {"len": 16, "addr": 255}},
        test_vectors={"smoke": (8, 4)},
        **kwargs,
    )


class TestDirectedTestAndRandomRows:
    def test_all(self):
        rows = vectors_param()._generate_rows(3)

        assert identity(rows) == [
            ("directed", "zeros", 0, (), None, ()),
            ("directed", "max", 1, (), None, ()),
            ("random", None, 0, (), 0, ()),
            ("random", None, 1, (), 1, ()),
            ("random", None, 2, (), 2, ()),
        ]
        assert rows[1].values == (255, 16)
        assert all(type(row.values) is vectors_param().vector_type for row in rows)

    def test_random_only(self):
        assert identity(vectors_param()._generate_rows(2, mode="random_only")) == [
            ("random", None, 0, (), 0, ()),
            ("random", None, 1, (), 1, ()),
        ]

    def test_directed_only(self):
        assert identity(vectors_param()._generate_rows(5, mode="directed_only")) == [
            ("directed", "zeros", 0, (), None, ()),
            ("directed", "max", 1, (), None, ()),
        ]

    @pytest.mark.parametrize("always", [True, False])
    def test_mixed(self, always):
        rows = vectors_param(always_include_directed=always)._generate_rows(1, mode="mixed")

        directed = [("directed", "zeros", 0, (), None, ()), ("directed", "max", 1, (), None, ())]
        assert identity(rows) == (directed if always else []) + [("random", None, 0, (), 0, ())]

    def test_test(self):
        rows = vectors_param()._generate_rows(5, mode="test")

        assert identity(rows) == [("test", "smoke", 0, (), None, ())]
        assert rows[0].values == (8, 4)

    def test_filter_by_name_keeps_the_vectors_index(self):
        assert identity(vectors_param()._generate_rows(5, filter_by_name="max")) == [
            ("directed", "max", 1, (), None, ())
        ]

    def test_filter_by_index(self):
        assert identity(vectors_param()._generate_rows(5, filter_by_index=1)) == [
            ("directed", "max", 1, (), None, ())
        ]

    def test_a_pytest_param_vector_keeps_its_marks(self):
        marked = pytest.param({"addr": 1, "len": 2}, marks=pytest.mark.xfail)
        param = vectors_param()
        param.add_directed_vector("marked", marked)

        row = param._generate_rows(0)[2]

        assert (row.kind, row.name, row.index) == ("directed", "marked", 2)
        assert type(row.values) is param.vector_type
        assert row.values == (1, 2)
        assert row.param is param.directed_vectors["marked"]
        assert row.param.marks == marked.marks
        assert row.sample is row.param

    def test_a_plain_vector_is_its_own_sample(self):
        row = vectors_param()._generate_rows(0)[0]

        assert row.param is None
        assert row.sample is row.values

    def test_generate_vectors_returns_the_rows_samples(self):
        param = vectors_param()
        marked = pytest.param(3, 4, marks=pytest.mark.xfail)
        param.add_test_vector("marked", marked)
        for kwargs in (
            {"mode": "all"},
            {"mode": "random_only"},
            {"mode": "directed_only"},
            {"mode": "mixed"},
            {"mode": "test"},
            {"filter_by_name": "zeros"},
            {"filter_by_index": 1},
        ):
            RNG.seed(7)
            expected = [row.sample for row in param._generate_rows(4, **kwargs)]
            RNG.seed(7)
            assert param.generate_vectors(4, **kwargs) == expected, kwargs

    def test_a_random_rows_index_and_j_do_not_depend_on_the_directed_vectors(self):
        with_directed = vectors_param()._generate_rows(2)
        without = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 255)),
            TestArg("len", rng_type=RNGInteger(1, 16)),
        )._generate_rows(2)

        assert identity(with_directed[2:]) == identity(without)


class TestSkippedRows:
    def skipped_param(self):
        return Parameter(
            TestArg("ch", rng_type=Series([], skip_if_empty="no channels")),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zeros": (0, 0)},
            test_vectors={"smoke": (1, 1)},
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"mode": "random_only"},
            {"mode": "directed_only"},
            {"mode": "test"},
            {"filter_by_name": "zeros"},
            {"filter_by_index": 0},
            {"exhaustive": True},
        ],
        ids=["all", "random_only", "directed_only", "test", "name", "index", "exhaustive"],
    )
    def test_one_skipped_row(self, kwargs):
        param = self.skipped_param()

        rows = param._generate_rows(3, **kwargs)

        assert identity(rows) == [("skipped", None, None, (), None, ())]
        assert type(rows[0].values) is param.vector_type
        assert rows[0].values == (None, None)

    def test_the_generators_return_no_skipped_row(self):
        param = self.skipped_param()

        assert param.generate_vectors(3) == []
        assert param.generate_exhaustive() == []

    def test_a_filter_that_names_no_vector_still_raises(self):
        with pytest.raises(KeyError):
            self.skipped_param()._generate_rows(3, filter_by_name="nope")


class TestFiniteSeriesRows:
    def test_j_is_the_cycle(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
        )

        rows = param._generate_rows(7)

        assert identity(rows) == [
            ("random", None, j, (("ch", f"i:{ch}"),), j, (f"ch={ch}",))
            for j, ch in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
        ]
        assert [row.values.ch for row in rows] == [0, 1, 2, 0, 1, 2, 0]

    def test_a_series_only_strategy_repeats_its_combinations(self):
        rows = Parameter(TestArg("ch", rng_type=Series(["a", "b"])))._generate_rows(5)

        assert [(row.labels, row.j) for row in rows] == [
            (("ch=a",), 0),
            (("ch=b",), 0),
            (("ch=a",), 1),
            (("ch=b",), 1),
            (("ch=a",), 2),
        ]

    def test_a_skipped_combination_leaves_a_gap_in_j(self):
        """The constraints reject the first visit of ch=1: its first row is j=1."""
        rejected = []

        def first_visit_of_one(v):
            if v.ch == 1 and not rejected:
                rejected.append(v)
                return False
            return True

        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[first_visit_of_one],
            max_retries=1,
        )

        with pytest.warns(PytestStrategiesWarning, match=r"Series combination \(ch=1\) skipped"):
            rows = param._generate_rows(3)

        assert [(row.values.ch, row.index, row.j) for row in rows] == [
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
        ]

    def test_pos_is_sorted_by_argument_name_and_labels_follow_declaration_order(self):
        param = Parameter(
            TestArg("zeta", rng_type=Series(["a", "b"])),
            TestArg("mid", rng_type=RNGInteger(0, 9)),
            TestArg("alpha", rng_type=Series([1, 2])),
        )

        rows = param._generate_rows(4)

        assert [(row.pos, row.labels) for row in rows] == [
            ((("alpha", "i:1"), ("zeta", "s:a")), ("zeta=a", "alpha=1")),
            ((("alpha", "i:2"), ("zeta", "s:a")), ("zeta=a", "alpha=2")),
            ((("alpha", "i:1"), ("zeta", "s:b")), ("zeta=b", "alpha=1")),
            ((("alpha", "i:2"), ("zeta", "s:b")), ("zeta=b", "alpha=2")),
        ]
        assert [row.j for row in rows] == [0, 0, 0, 0]

    def test_reordering_the_series_arguments_keeps_each_combinations_pos(self):
        def pos_by_values(*args):
            rows = Parameter(*args)._generate_rows(4)
            return {(row.values.a, row.values.b): row.pos for row in rows}

        forward = pos_by_values(
            TestArg("a", rng_type=Series([1, 2])), TestArg("b", rng_type=Series(["x", "y"]))
        )
        reverse = pos_by_values(
            TestArg("b", rng_type=Series(["x", "y"])), TestArg("a", rng_type=Series([1, 2]))
        )

        assert forward == reverse

    def test_an_rng_sequence_is_drawn_not_enumerated(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("dev", rng_type=RNGSequence(["a", "b", "c"])),
        )

        rows = param._generate_rows(3)

        assert [(row.pos, row.j) for row in rows] == [
            ((("ch", "i:0"),), 0),
            ((("ch", "i:1"),), 0),
            ((("ch", "i:0"),), 1),
        ]

    def test_an_rng_sequence_alone_gives_plain_random_rows(self):
        rows = Parameter(TestArg("dev", rng_type=RNGSequence(["a", "b"])))._generate_rows(3)

        assert identity(rows) == [("random", None, k, (), k, ()) for k in range(3)]

    def test_a_repeated_series_value_gives_two_rows(self):
        rows = Parameter(TestArg("ch", rng_type=Series([1, 1, 2])))._generate_rows(3)

        assert [(row.labels, row.pos) for row in rows] == [
            (("ch=1",), (("ch", "i:1"),)),
            (("ch=1~1",), (("ch", "i:1~1"),)),
            (("ch=2",), (("ch", "i:2"),)),
        ]


class TestPerSequenceRows:
    def test_j_counts_the_rows_of_each_combination(self):
        param = Parameter(
            TestArg("dev", rng_type=Series(["a", "b"])),
            TestArg("ch", rng_type=RNGSequence([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            per_sequence_samples=True,
        )

        rows = param._generate_rows(2)

        assert identity(rows) == [
            (
                "random",
                None,
                j,
                (("ch", f"i:{ch}"), ("dev", f"s:{dev}")),
                j,
                (f"dev={dev}", f"ch={ch}"),
            )
            for dev in "ab"
            for ch in (0, 1)
            for j in (0, 1)
        ]

    def test_a_combination_cut_short(self):
        """The constraints reject the second row of dev=b: it gets j=0 only."""
        seen = []

        def one_row_of_b(v):
            if v.dev == "b":
                seen.append(v)
                return len(seen) == 1
            return True

        param = Parameter(
            TestArg("dev", rng_type=RNGSequence(["a", "b", "c"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[one_row_of_b],
            per_sequence_samples=True,
            max_retries=1,
        )

        with pytest.warns(PytestStrategiesWarning, match=r"\(dev='b'\) produced 1 of 2 rows"):
            rows = param._generate_rows(2)

        assert [(row.labels, row.index, row.j) for row in rows] == [
            (("dev=a",), 0, 0),
            (("dev=a",), 1, 1),
            (("dev=b",), 0, 0),
            (("dev=c",), 0, 0),
            (("dev=c",), 1, 1),
        ]

    def test_without_sequence_arguments_the_rows_are_plain(self):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), per_sequence_samples=True)

        assert identity(param._generate_rows(2)) == [
            ("random", None, k, (), k, ()) for k in range(2)
        ]


class TestExhaustiveRows:
    def auto_param(self, **kwargs):
        return Parameter(
            TestArg("dev", rng_type=Series(["a", "b"])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            TestArg("ch", rng_type=RNGSequence([10, 20, 30])),
            **kwargs,
        )

    def test_the_index_is_the_position_in_the_declaration_order_product(self):
        rows = self.auto_param()._generate_rows(0, exhaustive=True, mode="random_only")

        assert {row.kind for row in rows} == {"exhaustive"}
        assert {row.j for row in rows} == {0}
        for row in rows:
            dev, ch = row.values.dev, row.values.ch
            assert row.index == "ab".index(dev) * 3 + [10, 20, 30].index(ch)
            assert row.labels == (f"dev={dev}", f"ch={ch}")
            assert row.pos == (("ch", f"i:{ch}"), ("dev", f"s:{dev}"))
        assert sorted(row.index for row in rows) == list(range(6))

    def test_rows_follow_the_permutation_and_the_index_does_not(self):
        key = StreamKey.root(1, "test", "s", "t.py::test")
        sequence = RNGSequence([10, 20, 30])
        # The permutation drawn from the argument's order stream
        order = [value for _, value in _auto_order_from(key.child("order", "ch"), sequence)]
        assert order != [10, 20, 30]

        rows = Parameter(TestArg("ch", rng_type=sequence))._generate_rows(
            0, exhaustive=True, key=key
        )

        assert [row.values.ch for row in rows] == order
        assert [row.index for row in rows] == [[10, 20, 30].index(ch) for ch in order]

    def test_directed_vectors_come_first(self):
        param = self.auto_param(directed_vectors={"zeros": ("a", 0, 10)})

        rows = param._generate_rows(0, exhaustive=True)

        assert identity(rows[:1]) == [("directed", "zeros", 0, (), None, ())]
        assert [row.kind for row in rows[1:]] == ["exhaustive"] * 6

    def test_a_combination_left_out_leaves_its_index_unused(self):
        param = self.auto_param(vector_constraints={"not_20": lambda v: v.ch != 20})

        rows = param._generate_rows(0, exhaustive=True)

        assert sorted(row.index for row in rows) == [0, 2, 3, 5]

    def test_generate_exhaustive_returns_the_rows_values(self):
        param = self.auto_param()
        RNG.seed(11)
        expected = [row.values for row in param._generate_rows(0, exhaustive=True)]
        RNG.seed(11)

        assert param.generate_exhaustive() == expected

    def test_without_sequence_arguments_it_raises(self):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

        with pytest.raises(ValueError, match="No sequence arguments found"):
            param._generate_rows(0, exhaustive=True)

    def test_n_is_not_checked(self):
        rows = self.auto_param()._generate_rows(-1, exhaustive=True)

        assert len(rows) == 6


class TestAutoOrder:
    def test_a_series_keeps_its_positions(self):
        assert _auto_order(Series(["a", "b", "a"])) == [(0, "a"), (1, "b"), (2, "a")]

    def test_an_rng_sequence_gives_each_value_its_declared_position(self):
        values = [[1], [2], [3], [4]]
        RNG.seed(5)

        order = _auto_order(RNGSequence(values))

        assert sorted(position for position, _ in order) == [0, 1, 2, 3]
        assert all(values[position] is value for position, value in order)

    def test_a_subclass_that_returns_equal_copies(self):
        class Reversed(SequenceLike):
            def _get_auto_sequence(self):
                return [list(value) for value in reversed(self.sequence)]

        assert _auto_order(Reversed([[1], [2], [3]])) == [(2, [3]), (1, [2]), (0, [1])]
        # Equal copies cannot be told apart: they take their positions in order
        assert _auto_order(Reversed([[1], [2], [1]])) == [(0, [1]), (1, [2]), (2, [1])]

    def test_an_rng_sequence_draws_the_permutation_3_0_drew(self):
        values = ["a", "b", "c", "d", "e"]
        RNG.seed(9)
        drawn = RNG._generator.sample(values, len(values))
        RNG.seed(9)

        assert RNGSequence(values)._get_auto_sequence() == drawn

    def test_a_subclass_that_gives_positions(self):
        class Backwards(SequenceLike):
            def _auto_positions(self):
                return list(reversed(range(len(self.sequence))))

        sequence = Backwards(["a", "b", "a"])

        assert sequence._get_auto_sequence() == ["a", "b", "a"]
        assert _auto_order(sequence) == [(2, "a"), (1, "b"), (0, "a")]

    def test_a_subclass_without_an_order_raises(self):
        with pytest.raises(NotImplementedError):
            SequenceLike([1, 2])._get_auto_sequence()

    def test_a_subclass_that_returns_a_new_value_fails(self):
        class Doubled(SequenceLike):
            def _get_auto_sequence(self):
                return [value * 2 for value in self.sequence]

        with pytest.raises(RNGValueError, match=r"Doubled._get_auto_sequence\(\) returned 4"):
            _auto_order(Doubled([2, 3]))


# ---------------------------------------------------------------------------
# The rows in the messages
# ---------------------------------------------------------------------------


class TestRowsInMessages:
    def notes(self, param, **kwargs):
        with pytest.raises(ZeroDivisionError) as excinfo:
            param._generate_rows(2, **kwargs)
        return excinfo.value.__notes__

    def test_a_per_sequence_row(self):
        param = Parameter(
            TestArg("ch", rng_type=RNGSequence([3, 0])),
            TestArg("x", value=1),
            vector_constraints={"ratio": lambda v: 6 / v.ch},
            per_sequence_samples=True,
        )

        assert self.notes(param) == [
            "Raised by constraint 'ratio' on random row 0 (ch=0), Vector(ch=0, x=1)"
        ]

    def test_an_exhaustive_row(self):
        param = Parameter(
            TestArg("dev", rng_type=Series(["a", "b"])),
            TestArg("ch", rng_type=Series([3, 0])),
            vector_constraints={"ratio": lambda v: 6 / v.ch},
        )

        assert self.notes(param, exhaustive=True) == [
            "Raised by constraint 'ratio' on exhaustive row 1 (dev=a, ch=0), Vector(dev='a', ch=0)"
        ]

    def test_a_series_row_of_a_later_cycle(self):
        calls = []

        def ratio(v):
            calls.append(v)
            return 6 / (len(calls) - 4)

        param = Parameter(
            TestArg("ch", rng_type=Series(["a", "b"])),
            vector_constraints=[ratio],
        )

        with pytest.raises(ZeroDivisionError) as excinfo:
            param._generate_rows(4)

        assert excinfo.value.__notes__ == [
            "Raised by constraint 'ratio' on random row 1 (ch=b), Vector(ch='b')"
        ]


class TestWarningsPointAtTheCaller:
    def test_series_skip(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"no_one": lambda v: v.ch != 1},
            max_retries=1,
        )

        with pytest.warns(PytestStrategiesWarning, match="skipped") as record:
            param.generate_vectors(2)

        assert {w.filename for w in record} == {__file__}

    def test_per_sequence_short(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"no_one": lambda v: v.ch != 1},
            per_sequence_samples=True,
            max_retries=1,
        )

        with pytest.warns(PytestStrategiesWarning, match="produced 0 of 2 rows") as record:
            param.generate_vectors(2)

        assert {w.filename for w in record} == {__file__}


def test_row_values_are_vectors():
    param = Parameter(TestArg("a", rng_type=Series([1])), TestArg("b", rng_type=RNGInteger(0, 9)))

    rows = param._generate_rows(1) + param._generate_rows(0, exhaustive=True)

    assert all(isinstance(row, _Row) and isinstance(row.values, Vector) for row in rows)


def test_no_warning_without_a_skip():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vectors_param()._generate_rows(5)
