"""
Unit tests for _ids module.

Tests generate_test_ids, generate_dataclass_ids and make_unique_ids as pure functions.
"""

from dataclasses import dataclass

from pytest_strategy._ids import generate_dataclass_ids, generate_test_ids, make_unique_ids


@dataclass
class Vec2:
    x: int
    y: int


@dataclass
class RGB:
    r: int
    g: int
    b: int


class TestGenerateTestIds:
    # ---- single-parameter mode ----

    def test_single_param_integer(self):
        ids = generate_test_ids(["n"], [1, 2, 3])
        assert ids == ["n=1", "n=2", "n=3"]

    def test_single_param_string(self):
        ids = generate_test_ids(["s"], ["hello", "world"])
        assert ids == ["s='hello'", "s='world'"]

    def test_single_param_from_tuple(self):
        """Single-element tuples should be unwrapped."""
        ids = generate_test_ids(["n"], [(1,), (2,)])
        assert ids == ["n=1", "n=2"]

    def test_single_param_truncation(self):
        long_val = "x" * 200
        ids = generate_test_ids(["s"], [long_val], max_length=20)
        assert len(ids[0]) <= 20
        assert ids[0].endswith("...")

    # ---- multi-parameter mode ----

    def test_multi_param_basic(self):
        ids = generate_test_ids(["a", "b"], [(1, 2), (3, 4)])
        assert ids[0] == "a=1,b=2"
        assert ids[1] == "a=3,b=4"

    def test_multi_param_string_values(self):
        ids = generate_test_ids(["x", "y"], [("foo", "bar")])
        assert ids[0] == "x='foo',y='bar'"

    def test_multi_param_value_truncation(self):
        """Individual values longer than 20 chars should be truncated."""
        long_val = "z" * 50
        ids = generate_test_ids(["a", "b"], [(long_val, 1)])
        # value repr is >20 chars so should be truncated to 17 + "..."
        assert "..." in ids[0]

    def test_multi_param_id_truncation(self):
        ids = generate_test_ids(["a", "b", "c"], [(1, 2, 3)], max_length=10)
        assert len(ids[0]) <= 10
        assert ids[0].endswith("...")

    def test_empty_samples(self):
        ids = generate_test_ids(["x"], [])
        assert ids == []

    def test_empty_argnames_single(self):
        """Edge case: empty argnames with single tuples."""
        ids = generate_test_ids([], [])
        assert ids == []

    def test_default_max_length_is_80(self):
        """IDs should not exceed 80 chars by default."""
        long_val = "v" * 100
        ids = generate_test_ids(["a", "b"], [(long_val, long_val)])
        assert len(ids[0]) <= 80


class TestGenerateDataclassIds:
    def test_two_field_dataclass(self):
        samples = [Vec2(1, 2), Vec2(3, 4)]
        ids = generate_dataclass_ids(samples, Vec2)
        assert ids[0] == "x=1,y=2"
        assert ids[1] == "x=3,y=4"

    def test_three_field_dataclass(self):
        samples = [RGB(255, 128, 0)]
        ids = generate_dataclass_ids(samples, RGB)
        assert ids[0] == "r=255,g=128,b=0"

    def test_field_value_truncation(self):
        @dataclass
        class Wide:
            label: str

        long_label = "L" * 50
        samples = [Wide(long_label), Wide(long_label)]
        ids = generate_dataclass_ids(samples, Wide)
        for id_ in ids:
            assert "..." in id_

    def test_id_truncation_at_max_length(self):
        @dataclass
        class Triple:
            alpha: str
            beta: str
            gamma: str

        samples = [Triple("aaa", "bbb", "ccc")]
        ids = generate_dataclass_ids(samples, Triple, max_length=10)
        assert len(ids[0]) <= 10
        assert ids[0].endswith("...")

    def test_empty_samples(self):
        ids = generate_dataclass_ids([], Vec2)
        assert ids == []

    def test_string_field_quoted(self):
        @dataclass
        class Named:
            name: str

        # One argname means detect_dataclass_mode returns False, but
        # generate_dataclass_ids is a pure function — we can call it directly.
        samples = [Named("alice")]
        ids = generate_dataclass_ids(samples, Named)
        assert ids[0] == "name='alice'"


class TestMakeUniqueIds:
    """Duplicates are suffixed exactly as pytest suffixes them itself."""

    def test_unique_ids_unchanged(self):
        assert make_unique_ids(["x=1", "x=2"]) == ["x=1", "x=2"]

    def test_empty(self):
        assert make_unique_ids([]) == []

    def test_underscore_before_counter_when_id_ends_in_digit(self):
        assert make_unique_ids(["x=1", "x=1", "x=1"]) == ["x=1_0", "x=1_1", "x=1_2"]

    def test_counter_directly_after_other_ids(self):
        assert make_unique_ids(["s='a'", "s='a'"]) == ["s='a'0", "s='a'1"]

    def test_counter_per_id(self):
        ids = ["a", "b", "a", "c", "b"]
        assert make_unique_ids(ids) == ["a0", "b0", "a1", "c", "b1"]

    def test_suffixed_id_in_use_is_skipped(self):
        assert make_unique_ids(["x=1", "x=1", "x=1_0"]) == ["x=1_1", "x=1_2", "x=1_0"]

    def test_id_whose_duplicates_were_all_suffixed_is_free(self):
        """As in pytest, a suffixed ID is checked against the IDs replaced so far."""
        assert make_unique_ids(["b0", "b0", "b", "b"]) == ["b0_0", "b0_1", "b0", "b1"]

    def test_digit_check_applies_to_escaped_id(self):
        # pytest escapes "é" to "\xe9", which ends in a digit
        cafe = "café"
        assert make_unique_ids([cafe, cafe]) == [cafe + "_0", cafe + "_1"]
        assert make_unique_ids([cafe, cafe], escape=False) == [cafe + "0", cafe + "1"]

    def test_non_string_ids_unchanged(self):
        hidden = object()  # stands in for pytest.HIDDEN_PARAM
        assert make_unique_ids([hidden, "a", "a"]) == [hidden, "a0", "a1"]
