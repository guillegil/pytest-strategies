"""
Unit tests for Parameter.extend(): a new Parameter built on another, which is
left as it was.
"""

import pytest

from pytest_strategy import (
    Parameter,
    RNGChoice,
    RNGInteger,
    RNGValueError,
    TestArg,
)


def aligned(v):
    return v.addr % 4 == 0


def short(v):
    return v.length <= 256


def base():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 0xFFFF)),
        TestArg("length", rng_type=RNGInteger(1, 256)),
        directed_vectors={
            "zeros": {"addr": 0, "length": 1},
            "known": pytest.param({"addr": 4, "length": 4}, marks=pytest.mark.xfail),
        },
        test_vectors={"max": (0xFFFC, 256)},
        vector_constraints={"aligned": aligned, "short": short},
        nsamples=7,
        max_retries=50,
        per_sequence_samples=True,
        max_exhaustive=500,
        ids="values",
        always_include_directed=False,
    )


class TestBaseUnchanged:
    def test_extend_returns_a_new_parameter_and_leaves_the_base(self):
        b = base()
        new = b.extend(
            TestArg("prio", value=0),
            directed_vectors={"zeros": None},
            vector_constraints={"aligned": None},
            nsamples=1,
        )

        assert new is not b
        assert b.arg_names == ("addr", "length")
        assert list(b.directed_vectors) == ["zeros", "known"]
        assert list(b.vector_constraints) == ["aligned", "short"]
        assert b.nsamples == 7

    def test_no_arguments_copies_everything(self):
        b = base()
        new = b.extend()

        assert new.arg_names == b.arg_names
        assert dict(new.directed_vectors) == dict(b.directed_vectors)
        assert dict(new.test_vectors) == dict(b.test_vectors)
        assert dict(new.vector_constraints) == dict(b.vector_constraints)
        for setting in (
            "nsamples",
            "max_retries",
            "per_sequence_samples",
            "max_exhaustive",
            "ids",
            "always_include_directed",
        ):
            assert getattr(new, setting) == getattr(b, setting), setting


class TestArguments:
    def test_a_known_name_replaces_the_argument_in_place(self):
        narrow = RNGInteger(1, 8)
        new = base().extend(TestArg("addr", rng_type=narrow))

        assert new.arg_names == ("addr", "length")
        assert new.get_arg("addr").rng_type is narrow
        assert new.directed_vectors["zeros"] == (0, 1)

    def test_a_new_name_is_added_last_with_defaults_in_the_kept_vectors(self):
        new = base().extend(TestArg("prio", rng_type=RNGChoice([0, 1, 2])), defaults={"prio": 1})

        assert new.arg_names == ("addr", "length", "prio")
        assert new.directed_vectors["zeros"] == (0, 1, 1)
        assert new.directed_vectors["known"].values == (4, 4, 1)
        assert new.test_vectors["max"] == (0xFFFC, 256, 1)

    def test_a_fixed_value_fills_the_kept_vectors(self):
        new = base().extend(TestArg("mode", value="fast"))

        assert new.directed_vectors["zeros"].mode == "fast"
        assert new.test_vectors["max"].mode == "fast"

    def test_defaults_win_over_the_fixed_value(self):
        new = base().extend(TestArg("mode", value="fast"), defaults={"mode": "slow"})

        assert new.directed_vectors["zeros"].mode == "slow"

    def test_a_pytest_param_vector_keeps_its_marks(self):
        new = base().extend(TestArg("mode", value="fast"))

        known = new.directed_vectors["known"]
        assert [m.name for m in known.marks] == ["xfail"]
        assert known.values == (4, 4, "fast")

    def test_a_kept_vector_without_a_value_for_an_added_argument_fails(self):
        with pytest.raises(
            RNGValueError,
            match=r"directed vector 'zeros' has no value for the added argument 'prio'.*defaults=\{'prio': \.\.\.\}",
        ):
            base().extend(TestArg("prio", rng_type=RNGChoice([0, 1])))

    def test_vectors_given_again_or_removed_need_no_default(self):
        new = base().extend(
            TestArg("prio", rng_type=RNGChoice([0, 1])),
            directed_vectors={"zeros": {"addr": 0, "length": 1, "prio": 1}, "known": None},
            test_vectors={"max": None},
        )

        assert dict(new.directed_vectors) == {"zeros": (0, 1, 1)}
        assert dict(new.test_vectors) == {}

    def test_two_args_with_one_name_fail(self):
        with pytest.raises(RNGValueError, match="two test args named 'p'"):
            base().extend(TestArg("p", value=1), TestArg("p", value=2))

    def test_a_default_for_an_argument_extend_does_not_add_fails(self):
        with pytest.raises(
            RNGValueError, match="defaults name 'addr', which extend.. does not add"
        ):
            base().extend(TestArg("addr", rng_type=RNGInteger(0, 1)), defaults={"addr": 0})

    def test_a_positional_that_is_not_a_test_arg_fails(self):
        with pytest.raises(TypeError, match="TestArg instances"):
            base().extend({"prio": 0})  # type: ignore[arg-type]

    def test_a_bad_new_name_fails_as_in_the_constructor(self):
        with pytest.raises(RNGValueError):
            base().extend(TestArg("_private", value=1))


class TestVectors:
    def test_a_new_vector_is_added_last(self):
        new = base().extend(directed_vectors={"page_end": (4092, 4)})

        assert list(new.directed_vectors) == ["zeros", "known", "page_end"]

    def test_a_known_name_is_replaced_in_place(self):
        new = base().extend(directed_vectors={"zeros": {"length": 2, "addr": 0}})

        assert list(new.directed_vectors) == ["zeros", "known"]
        assert new.directed_vectors["zeros"] == (0, 2)

    def test_removing_a_vector_that_does_not_exist_fails(self):
        with pytest.raises(
            RNGValueError, match="removes the test vector 'nope'.*test vectors: max"
        ):
            base().extend(test_vectors={"nope": None})

    def test_a_bad_new_vector_fails_as_in_the_constructor(self):
        with pytest.raises(RNGValueError, match="did you mean"):
            base().extend(directed_vectors={"bad": {"adr": 0, "length": 1}})


class TestConstraints:
    def test_a_dict_adds_replaces_and_removes_by_name(self):
        def ordered(v):
            return True

        def shorter(v):
            return v.length <= 8

        new = base().extend(
            vector_constraints={"aligned": None, "short": shorter, "ordered": ordered}
        )

        assert dict(new.vector_constraints) == {"short": shorter, "ordered": ordered}

    def test_a_list_adds_named_after_each_function(self):
        def even(v):
            return v.addr % 2 == 0

        new = base().extend(vector_constraints=[even, lambda v: True])

        assert list(new.vector_constraints) == ["aligned", "short", "even", "constraint_3"]

    def test_a_list_with_a_taken_name_fails(self):
        def aligned(v):  # noqa: F811 - the same name as the base's constraint
            return True

        with pytest.raises(RNGValueError, match="Two constraints are named 'aligned'"):
            base().extend(vector_constraints=[aligned])

    def test_removing_a_constraint_that_does_not_exist_fails(self):
        with pytest.raises(RNGValueError, match="removes the constraint 'nope'.*aligned, short"):
            base().extend(vector_constraints={"nope": None})

    def test_a_single_callable_fails(self):
        with pytest.raises(TypeError, match="dict of names to constraints"):
            base().extend(vector_constraints=aligned)  # type: ignore[arg-type]

    def test_an_unnamed_constraint_keeps_its_origin_in_messages(self):
        b = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), vector_constraints=[lambda v: True])
        new = b.extend(TestArg("y", value=1))

        assert new._unnamed_origin("constraint_0") is not None

    def test_the_kept_constraints_reject_rows(self):
        new = base().extend(nsamples=30, per_sequence_samples=False)

        rows = new.generate_vectors(n=30, mode="random_only")
        assert rows and all(row[0] % 4 == 0 for row in rows)


class TestSettings:
    def test_a_given_setting_overrides_the_base(self):
        new = base().extend(
            nsamples=None,
            max_retries=5,
            per_sequence_samples=False,
            max_exhaustive=None,
            ids="names",
            always_include_directed=True,
        )

        assert new.nsamples is None
        assert new.max_retries == 5
        assert new.per_sequence_samples is False
        assert new.max_exhaustive is None
        assert new.ids == "names"
        assert new.always_include_directed is True

    def test_a_bad_setting_fails_as_in_the_constructor(self):
        with pytest.raises(ValueError, match="max_retries"):
            base().extend(max_retries=0)
