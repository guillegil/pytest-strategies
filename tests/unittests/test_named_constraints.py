"""
Unit tests for named constraints and their diagnostics (4.0).

vector_constraints takes a dict of names to functions or a list, whose names come
from the function names (constraint_<i> for lambdas, partials and callable
objects), and is a read-only mapping changed through add_constraint(),
remove_constraint() and clear_constraints(). When the draws run out, the error
counts the rejections by the name of the first failing constraint and shows the
first row each constraint rejected; the Series and per_sequence_samples warnings
count them too. A constraint that raises is named with the row, and a predicate
that runs out of draws names its argument.
"""

import functools
import re
import warnings

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    PytestStrategiesWarning,
    RNGEnum,
    RNGInteger,
    RNGSequence,
    RNGValueError,
    Series,
    TestArg,
)
from pytest_strategy._resolver import _exhausted_message
from pytest_strategy.parameters import _constraint_failure, _GenerationStats

FIRST_FAILING = "Rejected by (first failing constraint per draw): "


def aligned(v):
    return v.addr % 4 == 0


def no_4k_cross(v):
    return v.addr % 4096 + v.len <= 4096


def never(v):
    return False


def constraint_0(v):
    """A function whose name is the one a lambda in first position would get."""
    return True


class Within:
    """A callable object as a constraint."""

    def __init__(self, high):
        self.high = high

    def __call__(self, v):
        return v.addr <= self.high


class Checks:
    def small(self, v):
        return v.len < 8


def within(high):
    """Make closures that share one function name."""

    def check(v):
        return v.addr <= high

    return check


def _burst(**kwargs):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 8191)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        **kwargs,
    )


def _line(fn):
    """The line the messages give for a lambda or function."""
    return fn.__code__.co_firstlineno


class TestNames:
    def test_a_dict_keeps_its_keys_and_order(self):
        param = _burst(vector_constraints={"no_cross": no_4k_cross, "aligned": aligned})

        assert list(param.vector_constraints) == ["no_cross", "aligned"]
        assert dict(param.vector_constraints) == {"no_cross": no_4k_cross, "aligned": aligned}

    def test_a_list_names_functions_by_their_name(self):
        param = _burst(vector_constraints=[aligned, no_4k_cross])

        assert list(param.vector_constraints) == ["aligned", "no_4k_cross"]
        assert param.vector_constraints["aligned"] is aligned

    def test_a_bound_method_is_named_by_its_name(self):
        param = _burst(vector_constraints=[Checks().small])

        assert list(param.vector_constraints) == ["small"]

    def test_lambdas_partials_and_callable_objects_are_numbered_by_position(self):
        param = _burst(
            vector_constraints=[
                aligned,
                lambda v: v.len > 0,
                functools.partial(lambda v, high: v.addr <= high, high=99),
                Within(100),
            ]
        )

        assert list(param.vector_constraints) == [
            "aligned",
            "constraint_1",
            "constraint_2",
            "constraint_3",
        ]

    def test_a_number_already_taken_is_increased(self):
        # The lambda at position 0 would be constraint_0, which a later function has
        param = _burst(vector_constraints=[lambda v: True, constraint_0])

        assert list(param.vector_constraints) == ["constraint_1", "constraint_0"]

    def test_names_may_contain_dots(self):
        param = _burst(vector_constraints={"model.addr": aligned})

        assert list(param.vector_constraints) == ["model.addr"]

    def test_any_iterable_of_callables_works(self):
        assert list(_burst(vector_constraints=(aligned,)).vector_constraints) == ["aligned"]
        assert list(_burst(vector_constraints=iter([aligned])).vector_constraints) == ["aligned"]
        assert list(_burst(vector_constraints=[]).vector_constraints) == []
        assert list(_burst().vector_constraints) == []

    def test_another_parameters_constraints_keep_their_names(self):
        first = _burst(vector_constraints=[aligned, lambda v: True])

        second = _burst(vector_constraints=first.vector_constraints)

        assert list(second.vector_constraints) == ["aligned", "constraint_1"]

    def test_the_mapping_is_read_only(self):
        param = _burst(vector_constraints=[aligned])

        with pytest.raises(TypeError):
            param.vector_constraints["other"] = never
        with pytest.raises(TypeError):
            del param.vector_constraints["aligned"]
        with pytest.raises(AttributeError):
            param.vector_constraints = {}
        assert list(param.vector_constraints) == ["aligned"]

    def test_the_callers_dict_is_not_changed(self):
        given = {"aligned": aligned}
        param = _burst(vector_constraints=given)

        param.add_constraint(no_4k_cross)

        assert given == {"aligned": aligned}
        assert param.vector_constraints == {"aligned": aligned, "no_4k_cross": no_4k_cross}


class TestConstructionErrors:
    def test_two_functions_with_one_name_fail_naming_both(self):
        low, high = within(10), within(20)

        with pytest.raises(RNGValueError) as excinfo:
            _burst(vector_constraints=[low, high])

        message = str(excinfo.value)
        origin = f"within.<locals>.check at test_named_constraints.py:{_line(low)}"
        assert message.startswith(f"Two constraints are named 'check': {origin} and {origin}.")
        assert "pass a dict, such as vector_constraints={'check_a': ..., 'check_b': ...}" in message

    def test_the_same_function_twice_fails(self):
        with pytest.raises(RNGValueError, match="aligned at .* is given twice"):
            _burst(vector_constraints=[aligned, aligned])

    def test_the_same_partial_twice_fails(self):
        def is_even(v):
            return v.addr % 2 == 0

        check = functools.partial(is_even)
        with pytest.raises(RNGValueError, match="partial of .*is_even.* is given twice"):
            _burst(vector_constraints=[check, check])

    def test_the_same_function_under_two_keys_fails(self):
        with pytest.raises(RNGValueError, match="already the constraint 'a'"):
            _burst(vector_constraints={"a": aligned, "b": aligned})

    @pytest.mark.parametrize("value", [5, None, "aligned"], ids=["int", "None", "str"])
    def test_a_value_that_is_not_callable_fails(self, value):
        with pytest.raises(TypeError, match=r"vector_constraints\[1\] is .*, not a callable"):
            _burst(vector_constraints=[aligned, value])
        with pytest.raises(TypeError, match=r"vector_constraints\['b'\] is .*, not a callable"):
            _burst(vector_constraints={"b": value})

    @pytest.mark.parametrize("name", ["a:b", "", "a b", "a\tb", "a,b", "a=b", 1, None])
    def test_an_invalid_name_fails(self, name):
        with pytest.raises(RNGValueError, match=f"Constraint name {re.escape(repr(name))} is not"):
            _burst(vector_constraints={name: aligned})

    def test_a_single_callable_fails(self):
        with pytest.raises(TypeError, match="not a single callable .*put it in a list"):
            _burst(vector_constraints=aligned)

    def test_a_string_fails(self):
        with pytest.raises(TypeError, match=r"not a str \('aligned'\)"):
            _burst(vector_constraints="aligned")

    def test_a_scalar_fails(self):
        with pytest.raises(
            TypeError, match="dict of names to constraints or a list of them, not int"
        ):
            _burst(vector_constraints=5)


class TestAddRemove:
    def test_add_returns_the_name_it_used(self):
        param = _burst()

        assert param.add_constraint(aligned) == "aligned"
        assert param.add_constraint(lambda v: True) == "constraint_1"
        assert param.add_constraint(Within(5)) == "constraint_2"
        assert param.add_constraint(never, name="rule.never") == "rule.never"
        assert list(param.vector_constraints) == [
            "aligned",
            "constraint_1",
            "constraint_2",
            "rule.never",
        ]

    def test_add_increases_a_taken_number(self):
        param = _burst(vector_constraints=[lambda v: True, lambda v: True])
        param.remove_constraint("constraint_0")

        # Position 1 is taken by constraint_1
        assert param.add_constraint(lambda v: True) == "constraint_2"

    def test_add_takes_the_name_by_keyword_only(self):
        param = _burst()

        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
            param.add_constraint(aligned, "n")

    def test_add_rejects_a_taken_name(self):
        param = _burst(vector_constraints=[aligned])

        with pytest.raises(RNGValueError, match="Two constraints are named 'aligned'"):
            param.add_constraint(no_4k_cross, name="aligned")

    def test_add_rejects_an_invalid_name(self):
        with pytest.raises(RNGValueError, match="Constraint name 'a b' is not valid"):
            _burst().add_constraint(aligned, name="a b")

    def test_add_rejects_a_function_already_there(self):
        param = _burst(vector_constraints=[aligned])

        with pytest.raises(RNGValueError, match="given twice"):
            param.add_constraint(aligned, name="again")

    def test_add_rejects_a_non_callable(self):
        with pytest.raises(TypeError, match="add_constraint\\(\\) takes a callable, got 5"):
            _burst().add_constraint(5)

    def test_added_constraints_are_evaluated_last(self):
        calls = []

        def first(v):
            calls.append("first")
            return True

        def second(v):
            calls.append("second")
            return True

        param = _burst(vector_constraints=[second])
        param.add_constraint(first)
        param.generate_vector()

        assert calls == ["second", "first"]

    def test_remove_an_unknown_name_lists_the_names(self):
        param = _burst(vector_constraints=[aligned, lambda v: True])

        with pytest.raises(
            KeyError, match="No constraint named 'nope'. Constraints: aligned, constraint_1"
        ):
            param.remove_constraint("nope")
        with pytest.raises(KeyError, match="Constraints: none"):
            _burst().remove_constraint("nope")

    def test_remove_and_clear(self):
        param = _burst(vector_constraints={"aligned": aligned, "never": never})

        param.remove_constraint("never")
        assert list(param.vector_constraints) == ["aligned"]
        assert param.generate_vector().addr % 4 == 0

        param.clear_constraints()
        assert len(param.vector_constraints) == 0
        assert param.add_constraint(lambda v: True) == "constraint_0"


class TestEvaluation:
    def test_the_first_falsy_result_rejects_and_stops(self):
        calls = []

        def record(name, result):
            def check(v):
                calls.append(name)
                return result

            return check

        param = Parameter(
            TestArg("x", value=1),
            vector_constraints={"a": record("a", 1), "b": record("b", 0), "c": record("c", 1)},
            max_retries=1,
        )

        with pytest.raises(ValueError, match="b=1"):
            param.generate_vector()
        assert calls == ["a", "b"]

    def test_a_constraint_may_rely_on_the_ones_before_it(self):
        param = Parameter(
            TestArg("len", rng_type=RNGInteger(0, 3)),
            vector_constraints={"nonzero": lambda v: v.len != 0, "ratio": lambda v: 6 / v.len > 2},
        )

        assert all(row.len in (1, 2) for row in param.generate_vectors(30))

    def test_directed_and_test_vectors_are_not_checked(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zero": (0,)},
            test_vectors={"one": (1,)},
            vector_constraints=[never],
        )

        assert param.generate_vectors(0) == [(0,)]
        assert param.generate_vectors(0, mode="test") == [(1,)]


class TestExhaustedRetries:
    def test_counts_and_first_rows_by_name(self):
        param = _burst(
            vector_constraints={"aligned": lambda v: v.addr % 4 == 0, "no_4k_cross": never},
            max_retries=40,
        )
        RNG.seed(3)

        with pytest.raises(ValueError) as excinfo:
            param.generate_vectors(3)

        message = str(excinfo.value)
        assert message.startswith("Could not generate random row 0 after max_retries=40 draws. ")
        counts = re.search(
            re.escape(FIRST_FAILING) + r"aligned=(\d+), no_4k_cross=(\d+)\.", message
        )
        assert counts is not None, message
        a, b = map(int, counts.groups())
        assert a + b == 40 and a > 0 and b > 0
        # aligned rejects the unaligned draws, no_4k_cross the aligned ones
        first = re.search(
            r"First rows rejected: aligned: Vector\(addr=(\d+), len=\d+\); "
            r"no_4k_cross: Vector\(addr=(\d+), len=\d+\)\. ",
            message,
        )
        assert first is not None, message
        assert int(first.group(2)) % 4 == 0
        assert message.endswith("Raise Parameter(max_retries=...) or relax a constraint.")

    def test_names_the_row_that_ran_out(self):
        accepted = []

        def first_three(v):
            accepted.append(v)
            return len(accepted) <= 3

        param = _burst(vector_constraints=[first_three], max_retries=5)

        with pytest.raises(
            ValueError,
            match="^Could not generate random row 3 after max_retries=5 draws. .*first_three=5\\.",
        ):
            param.generate_vectors(10)

    def test_counts_only_the_row_that_ran_out(self):
        draws = []

        def every_other(v):
            draws.append(v)
            return len(draws) % 2 == 0 and len(draws) < 10

        param = _burst(vector_constraints=[every_other], max_retries=7)

        with pytest.raises(ValueError, match=f"{re.escape(FIRST_FAILING)}every_other=7\\."):
            param.generate_vectors(10)

    def test_generate_vector_says_a_random_row(self):
        param = _burst(vector_constraints=[never], max_retries=3)

        with pytest.raises(
            ValueError, match="^Could not generate a random row after max_retries=3 draws. "
        ):
            param.generate_vector()

    def test_unnamed_constraints_show_where_they_come_from(self):
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[lambda v: v.x >= 5, functools.partial(never), Within(-1)],
            max_retries=60,
        )
        RNG.seed(1)

        with pytest.raises(ValueError) as excinfo:
            param.generate_vector()

        line = _line(param.vector_constraints["constraint_0"])
        message = str(excinfo.value)
        assert f"constraint_0 (lambda at test_named_constraints.py:{line}): Vector(x=" in message
        assert "constraint_1 (partial of never at test_named_constraints.py:" in message
        assert "constraint_2" not in message  # never reached: the partial rejects first

    def test_a_callable_object_is_named_by_its_class(self):
        param = Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 9)),
            vector_constraints=[Within(-1)],
            max_retries=2,
        )

        with pytest.raises(ValueError, match=r"constraint_0 \(Within instance\): Vector\(addr="):
            param.generate_vector()

    def test_a_constraint_that_returned_none_gets_a_hint(self):
        def forgot(v):
            v.addr > 4  # noqa: B015

        param = _burst(vector_constraints=[forgot, lambda v: None], max_retries=3)

        with pytest.raises(
            ValueError, match=r"forgot \(returned None, missing return\?\): Vector\("
        ):
            param.generate_vector()

    def test_a_falsy_result_besides_none_gets_no_hint(self):
        results = iter([None, False, None])

        def mixed(v):
            return next(results)

        param = _burst(vector_constraints=[mixed], max_retries=3)

        with pytest.raises(ValueError) as excinfo:
            param.generate_vector()
        assert "missing return" not in str(excinfo.value)
        assert "First rows rejected: mixed: Vector(" in str(excinfo.value)

    def test_example_rows_are_cut(self):
        param = Parameter(
            TestArg("blob", value="x" * 500),
            TestArg("n", rng_type=RNGInteger(0, 9)),
            vector_constraints=[never],
            max_retries=1,
        )

        with pytest.raises(ValueError) as excinfo:
            param.generate_vector()
        example = str(excinfo.value).split("never: ")[1].split(". Raise")[0]
        assert example.startswith("Vector(blob='xxx")
        assert example.endswith("...") and len(example) == 120

    def test_the_series_cycle_error_counts_the_cycle(self):
        param = Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            vector_constraints={"never": never},
        )

        with pytest.raises(ValueError) as excinfo:
            param.generate_vectors(5)

        message = str(excinfo.value)
        assert message.startswith(
            "Could not generate valid vector: none of the 3 Series combinations satisfied "
            "the vector constraints (1 attempt(s) each). "
        )
        assert f"{FIRST_FAILING}never=3. First rows rejected: never: Vector(s=1)." in message
        # Without random arguments, more retries cannot help
        assert message.endswith(" Relax a constraint.")

    def test_the_series_cycle_error_leaves_out_earlier_rows(self):
        draws = []

        def first_rows_only(v):
            # Rejects draws 1 and 3 before the two rows, then everything
            draws.append(v)
            return len(draws) in (2, 4)

        param = Parameter(
            TestArg("s", rng_type=Series([1, 2])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints=[first_rows_only],
            max_retries=4,
        )

        # The two visits of the cycle that failed, not the draws before the rows
        with pytest.raises(ValueError, match=f"{re.escape(FIRST_FAILING)}first_rows_only=8\\. "):
            param.generate_vectors(5)
        assert len(draws) == 12

    def test_the_series_cycle_error_leaves_out_the_visits_before_a_row(self):
        draws = []

        def gate(v):
            # Accepts only the 5th draw: ch=2, after the 4 draws of ch=1 ran out
            draws.append(v)
            return len(draws) == 5

        param = Parameter(
            TestArg("ch", rng_type=Series([1, 2, 3])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"gate": gate},
            max_retries=4,
        )
        stats = _GenerationStats()

        with pytest.raises(ValueError) as excinfo:
            param.generate_vectors(2, _stats=stats)

        # The cycle that failed (ch=3, 1, 2), not the visit of ch=1 before the row
        assert f"{FIRST_FAILING}gate=12. First rows rejected: gate: Vector(ch=3, " in str(
            excinfo.value
        )
        assert len(draws) == 17
        # The -v summary counts every rejected draw of the call
        assert stats.rejected == {"gate": 16}

    def test_a_tie_names_the_constraint_evaluated_first(self):
        draws = []

        def zeta(v):
            # Rejects the odd draws; never rejects the even ones
            draws.append(v)
            return len(draws) % 2 == 0

        param = _burst(vector_constraints={"zeta": zeta, "alpha": never}, max_retries=4)

        with pytest.raises(ValueError) as excinfo:
            param.generate_vectors(1)

        assert f"{FIRST_FAILING}zeta=2, alpha=2. " in str(excinfo.value)
        # The advice turns off the first of the strictest in evaluation order
        assert excinfo.value.strictest == "zeta"
        assert _exhausted_message("s", excinfo.value).endswith(
            "turn one off for this run with --strategy-constraint-off=s:zeta."
        )

    def test_the_per_sequence_error_counts_by_name(self):
        param = Parameter(
            TestArg("ch", rng_type=RNGSequence([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": never},
            per_sequence_samples=True,
            max_retries=3,
        )

        with pytest.raises(ValueError, match=f"{re.escape(FIRST_FAILING)}never=6\\. "):
            param.generate_vectors(2)

    def test_the_exhaustive_error_counts_by_name(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            vector_constraints={"never": never},
        )

        with pytest.raises(ValueError, match=f"{re.escape(FIRST_FAILING)}never=2\\. "):
            param.generate_exhaustive()


class TestWarnings:
    def test_the_series_skip_warning_counts_by_name(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"ch1_needs_big_x": lambda v: v.ch == 0 or v.x > 100},
            max_retries=3,
        )

        with pytest.warns(PytestStrategiesWarning) as caught:
            rows = param.generate_vectors(3)

        assert [row.ch for row in rows] == [0, 0, 0]
        assert len(caught) == 1
        assert str(caught[0].message).startswith(
            "Series combination (ch=1) skipped: the vector constraints rejected "
            f"max_retries=3 draws of the non-Series args. {FIRST_FAILING}ch1_needs_big_x=3. "
            "First rows rejected: ch1_needs_big_x: Vector(ch=1, x="
        )

    def test_the_per_sequence_warning_counts_by_name(self):
        param = Parameter(
            TestArg("ch", rng_type=RNGSequence([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"ch1_needs_big_x": lambda v: v.ch == 0 or v.x > 100},
            per_sequence_samples=True,
            max_retries=3,
        )

        with pytest.warns(PytestStrategiesWarning) as caught:
            param.generate_vectors(2)

        assert len(caught) == 1
        assert str(caught[0].message).startswith(
            "Sequence combination (ch=1) produced 0 of 2 rows: the vector constraints "
            f"rejected max_retries=3 draws of the other args. {FIRST_FAILING}"
            "ch1_needs_big_x=3. First rows rejected: ch1_needs_big_x: Vector(ch=1, x="
        )


class TestRaisingConstraints:
    def test_names_the_constraint_and_the_row(self):
        param = Parameter(
            TestArg("len", value=0),
            vector_constraints={"ratio": lambda v: 10 / v.len > 1},
        )

        with pytest.raises(ZeroDivisionError) as excinfo:
            param.generate_vectors(2)

        # The constraint's own exception, with a note, and its frame in the traceback
        assert str(excinfo.value) == "division by zero"
        assert excinfo.value.__notes__ == [
            "Raised by constraint 'ratio' on random row 0, Vector(len=0)"
        ]
        assert any(entry.name == "<lambda>" for entry in excinfo.traceback)
        # The resolver's collection error
        assert _constraint_failure(excinfo.value).message("--strategy-constraint-off") == (
            "Constraint 'ratio' raised ZeroDivisionError on random row 0, Vector(len=0): "
            "division by zero"
        )

    @pytest.mark.parametrize(
        "generate",
        [
            lambda p: p.generate_vector(),
            lambda p: p.generate_vectors(5, mode="random_only"),
            lambda p: p.generate_exhaustive(),
        ],
        ids=["generate_vector", "generate_vectors", "generate_exhaustive"],
    )
    def test_callers_catch_the_constraints_own_exception(self, generate):
        """3.0 let the constraint's exception through: code that catches it keeps working."""
        param = Parameter(
            TestArg("x", rng_type=RNGSequence([0, 3])),
            vector_constraints={"known": lambda v: {1: True}[v.x]},
        )

        with pytest.raises(KeyError) as excinfo:
            generate(param)

        assert type(excinfo.value) is KeyError
        assert excinfo.value.__notes__[0].startswith("Raised by constraint 'known' on ")

    def test_an_unnamed_constraint_shows_where_it_comes_from(self):
        param = Parameter(TestArg("len", value=0), vector_constraints=[lambda v: v.lenght > 0])

        with pytest.raises(AttributeError) as excinfo:
            param.generate_vector()

        line = _line(param.vector_constraints["constraint_0"])
        origin = f"'constraint_0' (lambda at test_named_constraints.py:{line})"
        assert excinfo.value.__notes__ == [
            f"Raised by constraint {origin} on the row Vector(len=0)"
        ]
        assert _constraint_failure(excinfo.value).message("-") == (
            f"Constraint {origin} raised AttributeError on the row Vector(len=0): Vector has "
            "no argument 'lenght'; its arguments are len"
        )

    def test_a_series_row_is_named_by_its_values(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([3, 0])),
            vector_constraints={"ratio": lambda v: 6 / v.ch},
        )

        with pytest.raises(ZeroDivisionError) as excinfo:
            param.generate_vectors(2)

        assert excinfo.value.__notes__ == ["Raised by constraint 'ratio' on the row Vector(ch=0)"]

    def test_an_exception_raised_again_keeps_one_note(self):
        """A constraint that raises one exception object again: one note, for the last row."""
        error = LookupError("no such mode")

        def lookup(v):
            raise error

        for x in (1, 2):
            param = Parameter(TestArg("x", value=x), vector_constraints=[lookup])
            with pytest.raises(LookupError):
                param.generate_vector()
            assert error.__notes__ == [f"Raised by constraint 'lookup' on the row Vector(x={x})"]

        # A note the caller added in between keeps its place
        error.add_note("seen by the caller")
        with pytest.raises(LookupError):
            Parameter(TestArg("x", value=3), vector_constraints=[lookup]).generate_vectors(1)
        assert error.__notes__ == [
            "Raised by constraint 'lookup' on random row 0, Vector(x=3)",
            "seen by the caller",
        ]

    def test_a_truth_value_that_raises_is_named_too(self):
        class Ambiguous:
            def __bool__(self):
                raise ValueError("ambiguous")

        param = Parameter(TestArg("x", value=1), vector_constraints={"odd": lambda v: Ambiguous()})

        with pytest.raises(ValueError, match="ambiguous") as excinfo:
            param.generate_vectors(1)

        assert excinfo.value.__notes__ == [
            "Raised by constraint 'odd' on random row 0, Vector(x=1)"
        ]


class TestPredicates:
    def test_a_predicate_that_runs_out_names_its_argument(self):
        param = Parameter(TestArg("width", rng_type=RNGInteger(0, 9, predicate=lambda x: x > 100)))

        with pytest.raises(RNGValueError) as excinfo:
            param.generate_vector()

        assert str(excinfo.value) == (
            "Argument 'width' could not draw a value its predicate accepts: No valid value "
            "found after 100 attempts"
        )

    def test_an_enum_predicate_set_later_names_its_argument(self):
        from enum import Enum

        class Color(Enum):
            RED = 1
            BLUE = 2

        rng_type = RNGEnum(Color)
        arg = TestArg("color", rng_type=rng_type)
        rng_type.predicate = lambda c: False

        with pytest.raises(
            RNGValueError, match="^Argument 'color' could not draw .*no member of Color"
        ):
            arg.generate()

    def test_other_errors_pass_unchanged(self):
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9), validator=lambda x: x > 100))

        with pytest.raises(ValueError, match="^Value .* failed validation for argument 'x'$"):
            param.generate_vector()


class TestStats:
    def test_counts_every_rejection_by_name(self):
        draws = []

        def odd_draws(v):
            draws.append(v)
            return len(draws) % 2 == 0

        stats = _GenerationStats()
        param = _burst(vector_constraints={"always": lambda v: True, "odd": odd_draws})

        param.generate_vectors(5, _stats=stats)

        assert stats.rejected == {"odd": 5}
        assert stats.left_out == 0

    def test_counts_the_combinations_auto_leaves_out(self):
        stats = _GenerationStats()
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            vector_constraints={"not_two": lambda v: v.ch != 2},
        )

        assert param.generate_exhaustive(_stats=stats) == [(0,), (1,)]
        assert stats.rejected == {"not_two": 1}
        assert stats.left_out == 1

    def test_a_series_run_counts_the_skipped_visits(self):
        stats = _GenerationStats()
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"ch1_needs_big_x": lambda v: v.ch == 0 or v.x > 100},
            max_retries=3,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PytestStrategiesWarning)
            param.generate_vectors(3, _stats=stats)

        # Rows 0, 1, 2 at ch=0, with ch=1 skipped twice in between
        assert stats.rejected == {"ch1_needs_big_x": 6}
