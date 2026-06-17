"""
Unit tests for the Series deterministic sequence type.
"""

import pytest

from pytest_strategy import Parameter, RNGInteger, TestArg
from pytest_strategy.rng import RNG, RNGSequence, RNGValueError


class TestSeriesExportsAndBase:
    """Tests for S1 and S2: exports and base class hierarchy."""

    def test_series_importable_from_package(self):
        import pytest_strategy
        from pytest_strategy import Series  # noqa: F401

        assert "Series" in pytest_strategy.__all__

    def test_rngsequence_in_all(self):
        import pytest_strategy

        assert "RNGSequence" in pytest_strategy.__all__

    def test_sequencelike_importable(self):
        from pytest_strategy import SequenceLike  # noqa: F401

        assert True

    def test_series_is_sequencelike(self):
        from pytest_strategy import SequenceLike, Series

        assert isinstance(Series([1, 2, 3]), SequenceLike)

    def test_rngsequence_is_sequencelike(self):
        from pytest_strategy import SequenceLike
        from pytest_strategy.rng import RNGSequence

        assert isinstance(RNGSequence([1, 2, 3]), SequenceLike)


class TestSeriesUnit:
    """Unit tests for the Series class itself (no Parameter). Covers S2 structural, S5."""

    def test_initialization_stores_sequence(self):
        from pytest_strategy import Series

        s = Series([10, 20, 30])
        assert s.sequence == [10, 20, 30]

    def test_initialization_empty_raises(self):
        from pytest_strategy import Series

        with pytest.raises(RNGValueError, match="Sequence cannot be empty"):
            Series([])

    def test_initialization_with_predicate(self):
        from pytest_strategy import Series

        s = Series([1, 2, 3, 4, 5], predicate=lambda x: x % 2 == 0)
        assert s.sequence == [2, 4]

    def test_get_auto_sequence_returns_ordered(self):
        from pytest_strategy import Series

        s = Series([3, 1, 2])
        assert s._get_auto_sequence() == [3, 1, 2]

    def test_generate_returns_element_in_sequence(self):
        from pytest_strategy import Series

        RNG.seed(42)
        s = Series([10, 20, 30])
        value = s.generate()
        assert value in {10, 20, 30}

    def test_python_type_int(self):
        from pytest_strategy import Series

        assert Series([1, 2, 3]).python_type is int

    def test_python_type_str(self):
        from pytest_strategy import Series

        assert Series(["a", "b"]).python_type is str

    def test_rngsequence_get_auto_sequence_is_permutation(self):
        RNG.seed(42)
        rng_seq = RNGSequence([1, 2, 3, 4, 5])
        result = rng_seq._get_auto_sequence()
        assert sorted(result) == [1, 2, 3, 4, 5]
        assert len(result) == 5


class TestSeriesExhaustive:
    """Tests for generate_exhaustive with Series. Covers S3, S4, S7, S12a, S12b."""

    def test_single_series_auto_ordered(self):
        """S3: single Series auto -> ordered rows."""
        from pytest_strategy import Series

        param = Parameter(TestArg("item", rng_type=Series([1, 2, 3])))
        samples = param.generate_exhaustive()
        assert samples == [(1,), (2,), (3,)]

    def test_two_series_auto_cartesian_ordered(self):
        """S4: two Series auto -> full Cartesian product in order."""
        from pytest_strategy import Series

        param = Parameter(
            TestArg("a", rng_type=Series([1, 2])),
            TestArg("b", rng_type=Series(["x", "y", "z"])),
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 6
        assert samples == [(1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z")]

    def test_mixed_series_rngsequence_rnginteger_auto(self):
        """S7: mixed auto — row count = 2×3 = 6; Series column ordered."""
        from pytest_strategy import Series

        RNG.seed(42)
        param = Parameter(
            TestArg("s", rng_type=Series([1, 2])),
            TestArg("r", rng_type=RNGSequence(["a", "b", "c"])),
            TestArg("i", rng_type=RNGInteger(0, 99)),
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 6
        col0 = [s[0] for s in samples]
        assert col0.count(1) == 3
        assert col0.count(2) == 3
        col1 = [s[1] for s in samples]
        assert set(col1) == {"a", "b", "c"}

    def test_exhaustive_only_rngsequence_succeeds(self):
        """S12a: only RNGSequence -> succeeds, order-insensitive."""
        param = Parameter(TestArg("x", rng_type=RNGSequence([1, 2, 3])))
        samples = param.generate_exhaustive()
        assert len(samples) == 3
        assert {s[0] for s in samples} == {1, 2, 3}

    def test_exhaustive_no_sequencelike_raises(self):
        """S12b: no SequenceLike -> ValueError."""
        param = Parameter(TestArg("x", rng_type=RNGInteger(0, 100)))
        with pytest.raises(ValueError, match="No sequence arguments found"):
            param.generate_exhaustive()


class TestSeriesFiniteMode:
    """Tests for generate_vectors with Series in finite mode. Covers S8, S9, S10, S13."""

    def test_finite_k_gte_len_cycles_in_order(self):
        """S8: K=10, len=5 -> cycles [0,1,2,3,4,0,1,2,3,4]."""
        from pytest_strategy import Series

        RNG.seed(42)
        param = Parameter(
            TestArg("s", rng_type=Series([0, 1, 2, 3, 4])),
            TestArg("i", rng_type=RNGInteger(0, 100)),
        )
        samples = param.generate_vectors(n=10)
        assert len(samples) == 10
        series_col = [s[0] for s in samples]
        assert series_col == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    def test_finite_k_lt_len_first_k_in_order(self):
        """S9: K=10, len=20 -> first 10 in order."""
        from pytest_strategy import Series

        param = Parameter(TestArg("s", rng_type=Series(list(range(20)))))
        samples = param.generate_vectors(n=10)
        assert len(samples) == 10
        assert [s[0] for s in samples] == list(range(10))

    def test_finite_multi_series_k_lt_product_first_k(self):
        """S10: Series([0,1,2]) x Series([10,20,30,40]) = product 12; K=5 -> first 5."""
        from pytest_strategy import Series

        param = Parameter(
            TestArg("a", rng_type=Series([0, 1, 2])),
            TestArg("b", rng_type=Series([10, 20, 30, 40])),
        )
        samples = param.generate_vectors(n=5)
        assert len(samples) == 5
        assert samples == [(0, 10), (0, 20), (0, 30), (0, 40), (1, 10)]

    def test_cycle_unsatisfiable_constraint_raises(self):
        """S13: always-False constraint -> raises, does not hang."""
        from pytest_strategy import Series

        param = Parameter(
            TestArg("s", rng_type=Series([1, 2, 3])),
            vector_constraints=[lambda v: False],
        )
        with pytest.raises(ValueError, match="Could not generate valid vector"):
            param.generate_vectors(n=5)


class TestRNGSequencePermutationSemantics:
    """Tests for RNGSequence permutation semantics. Covers S5, S6, S11."""

    def test_single_rngsequence_auto_is_permutation(self):
        """S5: single RNGSequence auto -> each value exactly once, order NOT asserted."""
        param = Parameter(TestArg("x", rng_type=RNGSequence([1, 2, 3])))
        samples = param.generate_exhaustive()
        assert len(samples) == 3
        assert {s[0] for s in samples} == {1, 2, 3}

    def test_multi_rngsequence_auto_product_of_permutations(self):
        """S6: product of independent permutations -> 6 rows."""
        param = Parameter(
            TestArg("a", rng_type=RNGSequence([1, 2, 3])),
            TestArg("b", rng_type=RNGSequence(["x", "y"])),
        )
        samples = param.generate_exhaustive()
        assert len(samples) == 6
        col0 = [s[0] for s in samples]
        col1 = [s[1] for s in samples]
        assert col0.count(1) == 2
        assert col0.count(2) == 2
        assert col0.count(3) == 2
        assert col1.count("x") == 3
        assert col1.count("y") == 3

    def test_rngsequence_finite_unchanged(self):
        """S11: K random picks, repetition allowed, no ordering."""
        RNG.seed(42)
        param = Parameter(TestArg("x", rng_type=RNGSequence([1, 2, 3, 4, 5])))
        samples = param.generate_vectors(n=10)
        assert len(samples) == 10
        assert all(s[0] in {1, 2, 3, 4, 5} for s in samples)
