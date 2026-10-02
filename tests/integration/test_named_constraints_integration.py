"""
End-to-end tests for named constraints, run through pytester: when the draws run
out, the collection error counts the rejections by constraint name and ends with
the --strategy-constraint-off item for the strictest constraint; a constraint that
raises is named with the row and the user's frame; a predicate that runs out
names its argument; and -v counts the rejections per strategy.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import re

pytest_plugins = ["pytester"]


def test_exhausted_retries_name_the_strictest_constraint(pytester):
    pytester.makepyfile(
        test_nc_exhausted="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        def never(v):
            return False

        @register("nc_dma_burst")
        def burst():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(0, 63)),
                TestArg("len", rng_type=RNGInteger(1, 16)),
                vector_constraints={
                    "aligned": lambda v: v.addr % 8 == 0,
                    "no_4k_cross": never,
                },
                max_retries=40,
            )

        @strategy("nc_dma_burst")
        def test_burst(addr, len):
            pass
        """,
        test_nc_unnamed="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("nc_unnamed")
        def unnamed():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                vector_constraints=[lambda v: v.x > 9],
                max_retries=5,
            )

        @strategy("nc_unnamed")
        def test_unnamed(x):
            pass
        """,
    )

    result = pytester.runpytest("--rng-seed=1")

    result.assert_outcomes(errors=2)
    output = result.stdout.str()
    burst = re.search(
        r"In test_burst: Error generating samples for strategy 'nc_dma_burst': Could not "
        r"generate random row 0 after max_retries=40 draws\. Rejected by \(first failing "
        r"constraint per draw\): aligned=(\d+), no_4k_cross=(\d+)\. First rows rejected: "
        r"aligned: Vector\(addr=\d+, len=\d+\); no_4k_cross: Vector\(addr=\d+, len=\d+\)\. "
        r"Raise Parameter\(max_retries=\.\.\.\), relax a constraint, or turn one off for this "
        r"run with --strategy-constraint-off=nc_dma_burst:aligned\.\n",
        output,
    )
    assert burst is not None, output
    aligned, no_4k_cross = map(int, burst.groups())
    # aligned rejects 7 draws in 8, so it is the strictest
    assert aligned + no_4k_cross == 40 and aligned > no_4k_cross > 0
    result.stdout.fnmatch_lines(
        [
            "In test_unnamed: Error generating samples for strategy 'nc_unnamed': Could not "
            "generate random row 0 after max_retries=5 draws. Rejected by (first failing "
            "constraint per draw): constraint_0=5. First rows rejected: constraint_0 (lambda "
            "at test_nc_unnamed.py:7): Vector(x=*). Raise Parameter(max_retries=...), relax "
            "a constraint, or turn one off for this run with "
            "--strategy-constraint-off=nc_unnamed:constraint_0."
        ]
    )


def test_series_only_advice_leaves_out_max_retries(pytester):
    pytester.makepyfile(test_nc_series="""
        from pytest_strategy import Parameter, Series, TestArg, register, strategy

        @register("nc_series_never")
        def series():
            return Parameter(
                TestArg("ch", rng_type=Series([0, 1])),
                vector_constraints={"never": lambda v: False},
            )

        @strategy("nc_series_never")
        def test_series(ch):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "*none of the 2 Series combinations satisfied the vector constraints (1 attempt(s) "
            "each). Rejected by (first failing constraint per draw): never=2. First rows "
            "rejected: never: Vector(ch=0). Relax a constraint, or turn one off for this run "
            "with --strategy-constraint-off=nc_series_never:never."
        ]
    )


def test_a_raising_constraint_names_itself_the_row_and_the_users_frame(pytester):
    pytester.makepyfile(test_nc_raising="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        def ratio(v):
            return 64 / v.len > 4

        @register("nc_ratio")
        def factory():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(0, 9)),
                TestArg("len", value=0),
                vector_constraints={"ratio": ratio},
            )

        @strategy("nc_ratio")
        def test_ratio(addr, len):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "In test_ratio: Error generating samples for strategy 'nc_ratio': Constraint "
            "'ratio' raised ZeroDivisionError on random row 0, Vector(addr=*, len=0): "
            "division by zero",
            '*test_nc_raising.py", line 4, in ratio',
            "*return 64 / v.len > 4",
            "ZeroDivisionError: division by zero",
        ]
    )
    # The user's frame only, not the plugin's
    result.stdout.no_fnmatch_line("*parameters.py*")


def test_a_predicate_that_runs_out_names_its_argument(pytester):
    pytester.makepyfile(test_nc_predicate="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("nc_predicate")
        def factory():
            return Parameter(TestArg("width", rng_type=RNGInteger(0, 9, predicate=lambda w: w > 9)))

        @strategy("nc_predicate")
        def test_width(width):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "In test_width: Error generating samples for strategy 'nc_predicate': Argument "
            "'width' could not draw a value its predicate accepts: No valid value found after "
            "100 attempts"
        ]
    )


SUMMARY = """
    from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register, strategy

    @register("nc_sum_aligned")
    def aligned_burst():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            vector_constraints={
                "not_two": lambda v: v.ch != 2,
                "aligned": lambda v: v.addr % 4 == 0,
            },
            max_retries=50,
        )

    @register("nc_sum_plain")
    def plain():
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

    @strategy("nc_sum_aligned")
    def test_aligned(ch, addr):
        assert addr % 4 == 0 and ch != 2

    @strategy("nc_sum_aligned")
    def test_aligned_again(ch, addr):
        pass

    @strategy("nc_sum_plain")
    def test_plain(x):
        pass
"""


def test_verbose_summary_counts_rejections_by_name(pytester):
    pytester.makepyfile(test_nc_summary=SUMMARY)

    # A subprocess keeps this suite's filterwarnings=error out of the inner run: each
    # test skips the ch=2 combination with a warning
    result = pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", "-v", "--rng-seed=2", "--nsamples=4"
    )

    result.assert_outcomes(passed=12, warnings=2)
    # Each test visits ch=0, 1, 2, 0, 1: ch=2 is rejected max_retries=50 times by not_two
    line = re.search(
        r"  nc_sum_aligned \(test_nc_summary\.py\): 2 test\(s\), 0 directed, 8 random rows; "
        r"nsamples=4 from --nsamples; rejected: not_two=100, aligned=(\d+)\n",
        result.stdout.str(),
    )
    assert line is not None, result.stdout.str()
    assert int(line.group(1)) > 0
    result.stdout.fnmatch_lines(
        [
            "  nc_sum_plain (test_nc_summary.py): 1 test(s), 0 directed, 4 random rows; "
            "nsamples=4 from --nsamples"
        ]
    )
    result.stdout.no_fnmatch_line("*nc_sum_plain*rejected*")


def test_verbose_summary_counts_the_combinations_auto_left_out(pytester):
    pytester.makepyfile(test_nc_summary_auto=SUMMARY)

    result = pytester.runpytest("-v", "--nsamples=auto")

    # Two combinations per test, and the plain strategy's 10 rows
    result.assert_outcomes(passed=2 + 2 + 10)
    result.stdout.re_match_lines(
        [
            r"  nc_sum_aligned \(test_nc_summary_auto\.py\): 2 test\(s\), 0 directed, 4 "
            r"random rows; nsamples=auto from --nsamples; rejected: not_two=100, aligned=\d+; "
            r"left out: 2 combinations$"
        ]
    )
