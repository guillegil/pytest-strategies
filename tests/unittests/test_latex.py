"""
The LaTeX of the export (_latex.py): an argument's ``latex`` (the set its values
are drawn from) and a constraint's, translated from simple predicates and
constraints only.
"""

import functools
import json
import linecache
import os
import warnings
from enum import Enum, Flag

import pytest

from pytest_strategy import (
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    TestArg,
)
from pytest_strategy._latex import argument_latex, constraint_latex, predicate_latex


class Access(Flag):
    R = 1
    W = 2
    RW = 3


class Status(Enum):
    IDLE = 1
    BUSY = 2
    DONE = 3


MEMBERS = frozenset({Status.DONE, Status.IDLE})
MIXED = {3, "a", 1, 2.5}
LIMIT = 4096
RESERVED = (12, 13)
ODD = frozenset({3, 1})
EVENS = tuple(range(0, 20, 2))
TINY = -1.5e-7
NAN = float("nan")
NANS = (NAN, 1)
FLAGS = (True, 2)


class Spaced:
    def __repr__(self):
        return " a  b "


class Custom(RNGType[int]):
    def generate(self) -> int:
        return 0


def linecache_entry(filename, source):
    """Give linecache the source of code compiled from a string."""
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)


def latex(rng_type=None, name="x", **kwargs):
    return TestArg(name, rng_type=rng_type, **kwargs).to_dict()["latex"]


def even(x):
    """Whether x is even."""
    return x % 2 == 0


def two_statements(x):
    y = x + 1
    return y > 2


def above(low):
    return lambda x: x > low


def helper(x):
    return True


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


class TestArgumentLatex:
    @pytest.mark.parametrize(
        ("rng_type", "expected"),
        [
            (RNGInteger(1, 20), r"x \in [1, 20]"),
            (RNGInteger(-5, 5), r"x \in [-5, 5]"),
            (RNGInteger(), r"x \in [-2^{31}, 2^{31} - 1]"),
            (RNGInteger(0, 2**16), r"x \in [0, 2^{16}]"),
            (RNGInteger(0, 65535), r"x \in [0, 65535]"),
            (RNGFloat(0.0, 1.0), r"x \in [0.0, 1.0] \subset \mathbb{R}"),
            (RNGFloat(0, 1.5e-7), r"x \in [0, 1.5 \times 10^{-7}] \subset \mathbb{R}"),
            (RNGFloat(1e-9, 1e20), r"x \in [10^{-9}, 10^{20}] \subset \mathbb{R}"),
            (RNGWeightedInteger({(0, 9): 3, (100, 199): 1}), r"x \in [0, 9] \cup [100, 199]"),
            (RNGWeightedFloat({(0.0, 1.0): 1}), r"x \in [0.0, 1.0] \subset \mathbb{R}"),
            (RNGWeightedInteger({(0, 9): 0, (100, 199): 1}), r"x \in [100, 199]"),
            (RNGChoice([1, 2, 4, 8]), r"x \in \{1, 2, 4, 8\}"),
            (RNGSequence([1, 2]), r"x \in \{1, 2\}"),
            (
                RNGSequence([0, 3, 3, True, [1], [1], [2]]),
                r"x \in \{0, 3, \text{True}, \text{[1]}, \text{[2]}\}",
            ),
            (RNGSequence([0, 3, 3, 0.0, False]), r"x \in \{0, 3, 0.0, \text{False}\}"),
            (Series(range(8)), r"x \in \{0, 1, 2, 3, 4, 5, 6, 7\}"),
            (Series(range(256)), r"x \in \{0, 1, 2, \dots, 255\}"),
            (Series([], skip_if_empty="no devices"), r"x \in \emptyset"),
            (Series(range(10), predicate=lambda v: v % 2 == 0), r"x \in \{0, 2, 4, 6, 8\}"),
            (RNGEnum(Status), r"x \in \{\text{IDLE}, \text{BUSY}, \text{DONE}\}"),
            (RNGEnum(Access), r"x \in \{\text{R}, \text{W}\}"),
            (RNGBoolean(), r"x \in \{\text{True}, \text{False}\}"),
            (RNGBoolean(0.3), r"x \in \{\text{True}, \text{False}\},\ P(\text{True}) = 0.3"),
            (RNGString(length=4), r"\lvert x \rvert = 4,\ x \in \Sigma^{*}"),
            (RNGString(), r"\lvert x \rvert \in [1, 20],\ x \in \Sigma^{*}"),
        ],
    )
    def test_a_built_in_type_is_the_set_it_draws_from(self, rng_type, expected):
        assert latex(rng_type) == expected

    def test_a_fixed_value(self):
        assert latex(value=5, name="count") == r"\mathit{count} = 5"
        assert latex(value=Status.BUSY) == r"x = \text{BUSY}"
        assert latex(value="fast") == r'x = \text{"fast"}'

    def test_a_name_is_a_variable(self):
        assert latex(RNGInteger(0, 1), name="burst_len") == r"\mathit{burst\_len} \in [0, 1]"
        assert latex(RNGInteger(0, 1), name="n") == r"n \in [0, 1]"

    def test_values_are_text_with_latex_escaped(self):
        choices = RNGChoice(["a_b$", None, 2.5, Status.IDLE, (1, 2), "x\\y{}^~\n"])

        assert latex(choices) == (
            r"x \in \{\text{"
            r'"a\_b\$"}, \text{None}, 2.5, \text{IDLE}, \text{(1, 2)}, '
            r'\text{"x\textbackslash{}\textbackslash{}y\{\}\textasciicircum{}\textasciitilde{}'
            r'\textbackslash{}n"}\}'
        )

    def test_a_string_is_quoted_as_python_escapes_it(self):
        assert latex(value='a"b') == r'x = \text{"a\textbackslash{}"b"}'
        assert latex(value="a\nb") != latex(value="a\\nb")

    def test_text_keeps_its_dashes_quotes_and_spaces(self):
        assert latex(value="--fast") == r'x = \text{"{-}{-}fast"}'
        assert latex(value="it's `q`") == r"""x = \text{"it{'}s {`}q{`}"}"""
        assert latex(value=" a  b ") == r'x = \text{" a ~b "}'
        assert latex(RNGChoice([" ", "a"])) == r'x \in \{\text{" "}, \text{"a"}\}'
        assert latex(value=Spaced()) == r"x = \text{~a ~b~}"
        assert latex(value="  ") == r'x = \text{" ~"}'

    def test_a_long_text_is_cut(self):
        assert latex(value="a" * 100) == r'x = \text{"' + "a" * 40 + r'"}\dots'
        assert latex(value="a" * 40) == r'x = \text{"' + "a" * 40 + r'"}'
        assert latex(value=tuple(range(30))) == (
            r"x = \text{(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11...}"
        )

    def test_a_cut_string_ends_between_characters(self):
        for tail in ("\nb", '"b', "\\"):
            assert latex(value="a" * 39 + tail) == r'x = \text{"' + "a" * 39 + r'"}\dots'
        assert latex(value="a" * 38 + "\\") == (
            r'x = \text{"' + "a" * 38 + r'\textbackslash{}\textbackslash{}"}'
        )

    def test_a_combining_mark_is_written_as_its_escape(self):
        assert latex(value="a_\u0301") == r'x = \text{"a\_\textbackslash{}u0301"}'

    def test_a_cut_set_does_not_repeat_a_value(self):
        assert latex(RNGChoice([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])) == r"x \in \{1, 2, 3, \dots\}"
        assert latex(RNGChoice([1, 2, 3, 4, 5, 6, 7, 8, 9, 1.0])) == (
            r"x \in \{1, 2, 3, \dots, 1.0\}"
        )

    def test_a_name_that_math_mode_cannot_write(self):
        assert latex(RNGInteger(0, 1), name="ℝ") is None
        assert latex(RNGInteger(0, 1), name="α_1") is None
        assert latex(RNGInteger(0, 1), name="e\u0301t") == r"\textit{ét} \in [0, 1]"
        assert latex(RNGInteger(0, 1), name="größe") == r"\textit{größe} \in [0, 1]"
        assert latex(RNGInteger(0, 1), name="a{") is None
        assert latex(RNGInteger(0, 1), name="a b") is None

    def test_an_enum_lists_the_members_it_draws_from(self):
        weighted = RNGEnum(Status, weights={Status.DONE: 1, Status.IDLE: 0, Status.BUSY: 2})
        assert latex(weighted) == r"x \in \{\text{DONE}, \text{BUSY}\}"

    def test_an_enum_predicate_is_a_condition_and_is_not_called(self):
        calls = []

        def record(member):
            calls.append(member)
            return True

        rng_type = RNGEnum(Status, predicate=record)
        calls.clear()

        assert latex(rng_type) == r"x \in \{\text{IDLE}, \text{BUSY}, \text{DONE}\},\ p(x)"
        assert calls == []
        assert latex(RNGEnum(Status, predicate=lambda s: s is not Status.BUSY)) == (
            r"x \in \{\text{IDLE}, \text{BUSY}, \text{DONE}\},\ x \ne \text{BUSY}"
        )

    def test_a_huge_sequence_is_not_walked(self):
        class Counting(list):
            walked = 0

            def __iter__(self):
                for item in super().__iter__():
                    type(self).walked += 1
                    yield item

        series = Series(range(1000))
        series.sequence = Counting(series.sequence)

        assert argument_latex(TestArg("x", rng_type=series)) == r"x \in \{0, 1, 2, \dots, 999\}"
        assert Counting.walked == 9

    def test_a_predicate_is_a_condition_after_the_set(self):
        assert latex(RNGInteger(1, 20, predicate=lambda v: v not in (12, 13)), name="addr") == (
            r"\mathit{addr} \in [1, 20],\ \mathit{addr} \notin \{12, 13\}"
        )
        assert latex(RNGFloat(0, 1, predicate=lambda v: v != 0.5)) == (
            r"x \in [0, 1] \subset \mathbb{R},\ x \ne 0.5"
        )
        assert latex(RNGWeightedInteger({(0, 9): 1}, predicate=even)) == (
            r"x \in [0, 9],\ x \bmod 2 = 0"
        )

    def test_a_predicate_that_is_not_translated_is_p(self):
        assert latex(RNGInteger(1, 20, predicate=helper)) == r"x \in [1, 20],\ p(x)"
        assert latex(RNGInteger(1, 20, predicate=lambda v: helper(v)), name="addr") == (
            r"\mathit{addr} \in [1, 20],\ p(\mathit{addr})"
        )

    def test_a_custom_type_has_none(self):
        class Wider(RNGInteger):
            pass

        assert latex(Custom()) is None
        assert latex(Wider(0, 9)) is None

    def test_the_export_holds_it_as_json(self):
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 4095)),
            TestArg("len", rng_type=RNGInteger(1, 64)),
            vector_constraints={
                "fits": lambda v: v.addr + v.len <= LIMIT,
                "opaque": lambda v: helper(v),
            },
        )

        data = json.loads(json.dumps(param.to_dict(), allow_nan=False))

        assert [arg["latex"] for arg in data["arguments"]] == [
            r"\mathit{addr} \in [0, 4095]",
            r"\mathit{len} \in [1, 64]",
        ]
        assert data["constraints"] == [
            {"name": "fits", "enabled": True, "latex": r"\mathit{addr} + \mathit{len} \le 4096"},
            {"name": "opaque", "enabled": True, "latex": None},
        ]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicateLatex:
    @pytest.mark.parametrize(
        ("predicate", "expected"),
        [
            (lambda x: x % 2 == 0, r"x \bmod 2 = 0"),
            (lambda x: x != 7, r"x \ne 7"),
            (lambda x: 0 < x <= 9, r"0 < x \le 9"),
            (lambda x: x >= -1, r"x \ge -1"),
            (lambda x: x not in (12, 13), r"x \notin \{12, 13\}"),
            (lambda x: x in [1, 2], r"x \in \{1, 2\}"),
            (lambda x: x not in range(12, 14), r"x \notin \{12, 13\}"),
            (lambda x: x in range(5), r"x \in \{0, 1, 2, 3, 4\}"),
            (lambda x: x in range(100), r"x \in \{0, 1, \dots, 99\}"),
            (lambda x: x in range(9, -1, -1), r"x \in \{0, 1, \dots, 9\}"),
            (lambda x: x in EVENS, None),
            (lambda x: x in (1, 2, 3, 4, 5, 6, 7, 8, 9), None),
            (lambda x: x not in RESERVED, r"x \notin \{12, 13\}"),
            (lambda x: x in ODD, r"x \in \{1, 3\}"),
            (lambda x: x < LIMIT, r"x < 4096"),
            (lambda x: x != Status.BUSY, r"x \ne \text{BUSY}"),
            (lambda x: x is not None, r"x \ne \text{None}"),
            (lambda x: x != "a", r'x \ne \text{"a"}'),
            (lambda x: x > 1 and x != 5, r"x > 1 \land x \ne 5"),
            (lambda x: x < 1 or x > 5, r"x < 1 \lor x > 5"),
            (
                lambda x: (x < 1 or x > 5) and x != 9,
                r"\left(x < 1 \lor x > 5\right) \land x \ne 9",
            ),
            (lambda x: not x < 3, r"x \nless 3"),
            (lambda x: not x >= 3, r"x \ngeq 3"),
            (lambda x: not x == 3, r"x \ne 3"),  # noqa: SIM201
            (lambda x: not x in (1, 2), r"x \notin \{1, 2\}"),  # noqa: E713
            (lambda x: x is not Status.BUSY, r"x \ne \text{BUSY}"),
            (lambda x: not (x < 3 or x > 9), r"\lnot \left(x < 3 \lor x > 9\right)"),
            (lambda x: x % 2, r"x \bmod 2 \ne 0"),
            (lambda x: not x % 4, r"x \bmod 4 = 0"),
            (lambda x: x & 1, r"\left(x \mathbin{\&} 1\right) \ne 0"),
            (
                lambda x: (x ^ 1) & 0xFF == 3,
                r"\left(\left(x \oplus 1\right) \mathbin{\&} 255\right) = 3",
            ),
            (lambda x: x & 3 & 5 == 1, r"\left(x \mathbin{\&} 3 \mathbin{\&} 5\right) = 1"),
            (lambda x: x | 1 == 3, None),
            (lambda x: abs(x | 1) > 2, None),
            (lambda x: (x >> 2) & 0xFF == 3, None),
            (lambda x: x << 1 > 4, None),
            (lambda x: x % 3 + 1 == 2, r"\left(x \bmod 3\right) + 1 = 2"),
            (lambda x: x % 2 * 3 == 3, r"\left(x \bmod 2\right) \cdot 3 = 3"),
            (lambda x: x * 2 % 3 == 0, r"x \cdot 2 \bmod 3 = 0"),
            (lambda x: (x + 1) % 3 == 0, r"\left(x + 1\right) \bmod 3 = 0"),
            (lambda x: x % (5 * 2) == 0, r"x \bmod \left(5 \cdot 2\right) = 0"),
            (lambda x: x % 8 % 3 == 0, r"\left(x \bmod 8\right) \bmod 3 = 0"),
            (lambda x: -x % 3 == 1, r"\left(-x\right) \bmod 3 = 1"),
            (lambda x: 1 + x % 4 == 0, r"1 + \left(x \bmod 4\right) = 0"),
            (lambda x: (x / 2) ** 2 > 3, r"\left(\frac{x}{2}\right)^{2} > 3"),
            (lambda x: -(-x) < 1, r"-\left(-x\right) < 1"),  # noqa: B002
            (lambda x: x % LIMIT, r"x \bmod 4096 \ne 0"),
            (lambda x: x % 4**2 == 1, r"x \bmod 4^{2} = 1"),
            (lambda x: x % -3 == 0, r"x \bmod \left(-3\right) = 0"),
            (lambda x: x % TINY == 0, r"x \bmod \left(-1.5 \times 10^{-7}\right) = 0"),
            (lambda x: x - TINY > 0, r"x - \left(-1.5 \times 10^{-7}\right) > 0"),
            (lambda x: 0 < x in (1, 2), r"0 < x \in \{1, 2\}"),
            (lambda x: x in (1, 2) == True, None),  # noqa: E712
            (lambda x: x in (1, 2) < 3, None),
            (lambda x: x == NAN, None),
            (lambda x: x < NAN, None),
            (lambda x: x in NANS, None),
            (lambda x: x == True, None),  # noqa: E712
            (lambda x: x != False, None),  # noqa: E712
            (lambda x: x in (True, 2), None),
            (lambda x: x in FLAGS, None),
            (lambda x: x is True, r"x = \text{True}"),
            (lambda x: x in MEMBERS, r"x \in \{\text{IDLE}, \text{DONE}\}"),
            (lambda x: x in MIXED, r'x \in \{1, 2.5, 3, \text{"a"}\}'),
            (lambda x: x % x, None),
            (lambda x: x & RESERVED, None),
            (lambda x: x ^ 1 > 0, r"\left(x \oplus 1\right) > 0"),
            (lambda x: (x + 1) * 2 > 4, r"\left(x + 1\right) \cdot 2 > 4"),
            (lambda x: x - (x - 1) > 0, r"x - \left(x - 1\right) > 0"),
            (lambda x: x - -1 > 0, r"x - \left(-1\right) > 0"),
            (lambda x: x / 2 > 1, r"\frac{x}{2} > 1"),
            (lambda x: x // 4 == 1, r"\left\lfloor \frac{x}{4} \right\rfloor = 1"),
            (lambda x: x**2 < 2**20, r"x^{2} < 2^{20}"),
            (lambda x: (x + 1) ** 2 > 4, r"\left(x + 1\right)^{2} > 4"),
            (lambda x: -(x**2) < -(x + 1), r"-x^{2} < -\left(x + 1\right)"),
            (lambda x: abs(x - 10) >= 2, r"\lvert x - 10 \rvert \ge 2"),
            (lambda x: len(x) > 3, r"\lvert x \rvert > 3"),
            (
                lambda x: min(x, 3) == max(x, 1, 2),
                r"\min\left(x, 3\right) = \max\left(x, 1, 2\right)",
            ),
            (lambda x: x < 1e-9, r"x < 10^{-9}"),
            (lambda x, n=3: x % n == 0, r"x \bmod 3 = 0"),
            (even, r"x \bmod 2 = 0"),
            (above(-5), r"x > -5"),
        ],
    )
    def test_a_predicate_is_translated_when_it_is_simple(self, predicate, expected):
        assert predicate_latex(predicate, "x") == expected

    def test_the_subject_replaces_the_parameter(self):
        assert predicate_latex(lambda value: value > 1, r"\mathit{addr}") == r"\mathit{addr} > 1"

    def test_two_lambdas_on_one_line_are_told_apart(self):
        first, second = (lambda x: x > 1), (lambda x: x > 2)  # fmt: skip

        assert predicate_latex(first, "x") == "x > 1"
        assert predicate_latex(second, "x") == "x > 2"

    def test_a_lambda_inside_a_lambda(self):
        make = lambda low: lambda x: x > low  # noqa: E731

        assert predicate_latex(make(3), "x") == "x > 3"

    @pytest.mark.parametrize(
        "predicate",
        [
            lambda x: helper(x),
            lambda x: x,
            lambda x: True,
            lambda x: x + 1,
            lambda x: x.real > 1,
            lambda x: x[0] > 1,
            lambda x: x > 1 if x else False,
            lambda x: x in (),
            lambda x: x in range(0, 10, 2),
            lambda x: x in range(0),
            lambda x: x in helper,
            lambda x: x in [helper],
            lambda x: x < helper,
            lambda x: x is helper,
            lambda x: x == unknown_name,  # noqa: F821
            lambda x: any(x > n for n in (1, 2)),
            lambda x: ~x > 0,
            lambda x: (x > 1) == (x < 5),
            lambda x: x @ 2 > 1,
            lambda x: abs(x, 1) > 0,
            lambda x: max(x) > 0,
            lambda x: abs(x=x) > 0,
            lambda x, n: x > n,
            lambda *x: x > 1,
            lambda x, **k: x > 1,
            lambda x, *, n=1: x > n,
            lambda: True,
            two_statements,
            functools.partial(even),
            even.__call__,
            int,
            "not a function",
            None,
        ],
    )
    def test_anything_else_is_not_translated(self, predicate):
        assert predicate_latex(predicate, "x") is None

    def test_a_rebound_builtin_is_not_translated(self):
        namespace = {"abs": lambda x: 0}
        code = "predicate = lambda x: abs(x) > 1\n"
        exec(compile(code, "<rebound>", "exec"), namespace)

        assert predicate_latex(namespace["predicate"], "x") is None

    def test_a_function_without_source_is_not_translated(self):
        namespace = {}
        exec("predicate = lambda x: x > 1\n", namespace)

        assert predicate_latex(namespace["predicate"], "x") is None

    def test_source_edited_since_the_import_is_not_translated(self, tmp_path, monkeypatch):
        module = tmp_path / "edited_predicates.py"
        module.write_text("predicate = lambda x: x > 1\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        import edited_predicates

        assert predicate_latex(edited_predicates.predicate, "x") == "x > 1"

        # linecache sees an edit by the file's size and mtime. Each write has the
        # same size, and on Windows two quick writes can share an mtime, so give
        # each one its own.
        mtime = module.stat().st_mtime

        def edit(text):
            nonlocal mtime
            module.write_text(text, encoding="utf-8")
            mtime += 10
            os.utime(module, (mtime, mtime))

        # The same shape: every position still matches, the code does not
        edit("predicate = lambda x: x < 1\n")

        assert edited_predicates.predicate(2)
        assert predicate_latex(edited_predicates.predicate, "x") is None

        edit("predicate = lambda x: x > 1\n")

        assert predicate_latex(edited_predicates.predicate, "x") == "x > 1"

    def test_code_compiled_under_another_files_name_is_not_translated(self, tmp_path):
        host = tmp_path / "host_predicates.py"
        host.write_text("shown = lambda x: x < 5\n", encoding="utf-8")
        namespace = {}
        exec(compile("hidden = lambda x: x > 9\n", str(host), "exec"), namespace)

        assert predicate_latex(namespace["hidden"], "x") is None

    def test_a_function_that_returns_a_function_is_not_translated(self):
        outer = lambda x: lambda x: x > 3  # noqa: E731

        def same(x):
            def same(x):
                return x > 3

            return same

        assert predicate_latex(outer, "x") is None
        assert predicate_latex(same, "x") is None
        assert predicate_latex(outer(0), "x") == "x > 3"

    def test_defaults_are_bound_as_a_call_binds_them(self):
        predicate = lambda x, n=10: x < n  # noqa: E731
        predicate.__defaults__ = (10, 20, 30)

        assert predicate(25)
        assert predicate_latex(predicate, "x") == "x < 30"

    def test_a_file_is_searched_once_per_function(self, monkeypatch):
        from pytest_strategy import _latex

        predicate = lambda x: x > 41  # noqa: E731
        assert predicate_latex(predicate, "x") == "x > 41"
        searched = []
        monkeypatch.setattr(_latex, "_find", lambda *args: searched.append(args))

        assert predicate_latex(predicate, "x") == "x > 41"
        assert searched == []

    def test_a_decorated_function(self):
        def keep(fn):
            return fn

        @keep
        @keep
        def small(x):
            return x < 3

        assert predicate_latex(small, "x") == "x < 3"

    def test_a_class_private_name_is_not_read_from_the_globals(self):
        namespace = {}
        code = "__limit = 5\n_K__limit = 100\nclass K:\n    p = lambda x: x > __limit\n    q = lambda x: x > 7\n"
        filename = "<private names>"
        exec(compile(code, filename, "exec"), namespace)
        linecache_entry(filename, code)

        assert not namespace["K"].p(50)
        assert predicate_latex(namespace["K"].p, "x") is None
        assert predicate_latex(namespace["K"].q, "x") == "x > 7"

    def test_a_closure_variable_not_yet_bound_is_not_read_from_the_globals(self):
        def make():
            predicate = lambda x: x < LIMIT  # noqa: E731
            before = predicate_latex(predicate, "x")
            LIMIT = 5
            return before, predicate_latex(predicate, "x")

        assert make() == (None, "x < 5")

    def test_the_files_syntax_warnings_are_not_given_again(self, tmp_path, monkeypatch):
        module = tmp_path / "warning_predicates.py"
        module.write_text("pattern = '\\d'\npredicate = lambda x: x > 5\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import warning_predicates

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert predicate_latex(warning_predicates.predicate, "x") == "x > 5"

    def test_a_predicate_in_a_class_body_or_a_staticmethod(self):
        class Rules:
            small = staticmethod(lambda x: x < 3)

            @staticmethod
            def big(x):
                return x > 100

        assert predicate_latex(Rules.small, "x") == "x < 3"
        assert predicate_latex(Rules.big, "x") == "x > 100"
        assert predicate_latex(Rules().big, "x") == "x > 100"

    def test_a_lambda_over_several_lines_after_other_text(self):
        # ünïcödé before a lambda moves its byte columns, not its characters
        pair = ("ünïcödé", lambda x: (
            x > 1
            and x != 5
        ))  # fmt: skip

        assert predicate_latex(pair[1], "x") == r"x > 1 \land x \ne 5"


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraintLatex:
    @pytest.mark.parametrize(
        ("constraint", "expected"),
        [
            (lambda v: v.addr + v.len <= LIMIT, r"\mathit{addr} + \mathit{len} \le 4096"),
            (lambda v: v.addr % 4 == 0, r"\mathit{addr} \bmod 4 = 0"),
            (lambda v: v[0] != v[-1], r"\mathit{addr} \ne \mathit{len}"),
            (lambda v: v[1] > 0, r"\mathit{len} > 0"),
            (lambda row: row.len in (1, 2), r"\mathit{len} \in \{1, 2\}"),
        ],
    )
    def test_a_simple_constraint_is_translated(self, constraint, expected):
        assert constraint_latex(constraint, ("addr", "len")) == expected

    @pytest.mark.parametrize(
        "constraint",
        [
            lambda v: v.nope > 1,
            lambda v: v[2] > 1,
            lambda v: v[-3] > 1,
            lambda v: v["addr"] > 1,
            lambda v: v > 1,
            lambda v: sum(v) > 1,
            lambda v: v.addr.real > 1,
            lambda v: True,
        ],
    )
    def test_anything_else_is_not_translated(self, constraint):
        assert constraint_latex(constraint, ("addr", "len")) is None
