"""
End-to-end regression tests for resolver fixes, run in isolated pytester sessions.

Distinct strategy names are used per test on purpose: the strategy registry is
process-global and shared by every in-process pytester run.
"""

pytest_plugins = ["pytester"]


AUTO_MODULE = """
    from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger
    from pytest_strategy.rng import Series

    @Strategy.register("fix_auto_seq")
    def seq(nsamples):
        return Parameter(
            TestArg("x", rng_type=Series([1, 2, 3])),
            TestArg("y", rng_type=Series(["a", "b"])),
            directed_vectors={"corner": (99, "z")},
            test_vectors={"tv": (42, "t")},
        )

    @Strategy.register("fix_auto_noseq")
    def noseq(nsamples):
        return Parameter(
            TestArg("code", rng_type=RNGInteger(200, 400)),
            directed_vectors={"edge": (500,)},
            test_vectors={"ok": (200,)},
            nsamples=4,
        )

    @Strategy.strategy("fix_auto_seq")
    def test_seq(x, y):
        pass

    @Strategy.strategy("fix_auto_noseq")
    def test_noseq(code):
        pass
"""


class TestAutoModeIntegration:
    """--nsamples=auto combined with vector modes and filters."""

    def test_auto_all_mode_includes_directed_vectors(self, pytester):
        pytester.makepyfile(test_fix_auto=AUTO_MODULE)
        result = pytester.runpytest("--nsamples=auto", "-v")
        # seq: corner + 3x2 product; noseq (no sequence args): edge + 4 random
        result.assert_outcomes(passed=7 + 5)
        result.stdout.fnmatch_lines(["*test_seq[[]x=99,y='z'[]] PASSED*"])
        result.stdout.fnmatch_lines(["*test_noseq[[]code=500[]] PASSED*"])

    def test_auto_test_mode_runs_only_test_vectors(self, pytester):
        pytester.makepyfile(test_fix_auto=AUTO_MODULE)
        result = pytester.runpytest("--nsamples=auto", "--vector-mode=test", "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            ["*test_seq[[]x=42,y='t'[]] PASSED*", "*test_noseq[[]code=200[]] PASSED*"]
        )

    def test_auto_directed_only_mode(self, pytester):
        pytester.makepyfile(test_fix_auto=AUTO_MODULE)
        result = pytester.runpytest("--nsamples=auto", "--vector-mode=directed_only", "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            ["*test_seq[[]x=99,y='z'[]] PASSED*", "*test_noseq[[]code=500[]] PASSED*"]
        )

    def test_auto_vector_name_filters_across_strategies(self, pytester):
        pytester.makepyfile(test_fix_auto=AUTO_MODULE)
        result = pytester.runpytest("--nsamples=auto", "--vector-name=corner", "-v")
        # Only fix_auto_seq has "corner"; fix_auto_noseq gets an empty parameter set
        result.assert_outcomes(passed=1, skipped=1)
        result.stdout.fnmatch_lines(["*test_seq[[]x=99,y='z'[]] PASSED*"])


INDEX_MODULE = """
    from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

    @Strategy.register("fix_idx_two")
    def two(nsamples):
        return Parameter(
            TestArg("y", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zero": (0,), "ten": (10,)},
        )

    @Strategy.register("fix_idx_one")
    def one(nsamples):
        return Parameter(TestArg("z", rng_type=RNGInteger(0, 9)), directed_vectors={"zero": (0,)})

    @Strategy.register("fix_idx_none")
    def none(nsamples):
        return Parameter(TestArg("w", rng_type=RNGInteger(0, 9)))

    @Strategy.strategy("fix_idx_two")
    def test_two(y):
        pass

    @Strategy.strategy("fix_idx_one")
    def test_one(z):
        pass

    @Strategy.strategy("fix_idx_none")
    def test_none(w):
        pass

    def test_plain():
        pass
"""


class TestVectorIndexIntegration:
    """--vector-index skips strategies that lack the index, like --vector-name does."""

    def test_index_valid_for_some_strategies(self, pytester):
        pytester.makepyfile(test_fix_idx=INDEX_MODULE)
        result = pytester.runpytest("--vector-index=1", "-v")
        # fix_idx_two has index 1 and test_plain is unaffected; the others are skipped
        result.assert_outcomes(passed=2, skipped=2)
        result.stdout.fnmatch_lines(["*test_two[[]y=10[]] PASSED*"])

    def test_index_zero_with_strategy_without_directed_vectors(self, pytester):
        pytester.makepyfile(test_fix_idx=INDEX_MODULE)
        result = pytester.runpytest("--vector-index=0")
        result.assert_outcomes(passed=3, skipped=1)

    def test_index_out_of_range_everywhere(self, pytester):
        pytester.makepyfile(test_fix_idx=INDEX_MODULE)
        result = pytester.runpytest("--vector-index=5")
        result.assert_outcomes(passed=1, skipped=3)


class TestSingleArgumentIdsIntegration:
    """Tuple-valued single-argument strategies get IDs showing the whole value."""

    def test_tuple_values_get_distinct_ids(self, pytester):
        pytester.makepyfile(test_fix_single_ids="""
            from pytest_strategy import Strategy, Parameter, TestArg
            from pytest_strategy.rng import RNGChoice

            @Strategy.register("fix_ids_point")
            def point(nsamples):
                return Parameter(
                    TestArg("pt", rng_type=RNGChoice([(1, 2)])),
                    directed_vectors={"a": ((1, 2),), "b": ((1, 3),), "c": ((1, 4),)},
                )

            @Strategy.strategy("fix_ids_point")
            def test_point(pt):
                assert len(pt) == 2
            """)
        result = pytester.runpytest("--vector-mode=directed_only", "-v")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(
            [
                "*test_point[[]pt=(1, 2)[]] PASSED*",
                "*test_point[[]pt=(1, 3)[]] PASSED*",
                "*test_point[[]pt=(1, 4)[]] PASSED*",
            ]
        )


class TestLegacyTupleStrategiesIntegration:
    """Legacy (argnames, samples) strategies behave like pytest.mark.parametrize."""

    def test_comma_separated_argnames(self, pytester):
        pytester.makepyfile(test_fix_legacy_csv="""
            from pytest_strategy import Strategy

            @Strategy.register("fix_legacy_csv")
            def csv(nsamples):
                return "x, y", [(1, 2), (3, 4)]

            @Strategy.strategy("fix_legacy_csv")
            def test_csv(x, y):
                assert y == x + 1
            """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*test_csv[[]x=1,y=2[]] PASSED*"])

    def test_generator_samples(self, pytester):
        pytester.makepyfile(test_fix_legacy_gen="""
            from pytest_strategy import Strategy

            @Strategy.register("fix_legacy_gen")
            def gen(nsamples):
                return ("a", "b"), ((i, i + 1) for i in range(3))

            @Strategy.strategy("fix_legacy_gen")
            def test_gen(a, b):
                assert b == a + 1
            """)
        result = pytester.runpytest()
        result.assert_outcomes(passed=3)

    def test_pytest_param_marks_and_ids_are_kept(self, pytester):
        pytester.makepyfile(test_fix_legacy_param="""
            import pytest
            from pytest_strategy import Strategy

            @Strategy.register("fix_legacy_param_one")
            def one(nsamples):
                return ("x",), [
                    pytest.param(1, marks=pytest.mark.xfail(strict=True)),
                    pytest.param(5, id="five"),
                    2,
                ]

            @Strategy.register("fix_legacy_param_two")
            def two(nsamples):
                return ("a", "b"), [pytest.param(1, 2, marks=pytest.mark.xfail(strict=True)), (3, 3)]

            @Strategy.strategy("fix_legacy_param_one")
            def test_one(x):
                assert x in (2, 5)

            @Strategy.strategy("fix_legacy_param_two")
            def test_two(a, b):
                assert a == b
            """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3, xfailed=2)
        result.stdout.fnmatch_lines(
            [
                "*test_one[[]x=1[]] XFAIL*",
                "*test_one[[]five[]] PASSED*",
                "*test_one[[]x=2[]] PASSED*",
                "*test_two[[]a=1,b=2[]] XFAIL*",
                "*test_two[[]a=3,b=3[]] PASSED*",
            ]
        )
