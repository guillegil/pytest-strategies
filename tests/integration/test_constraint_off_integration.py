"""
End-to-end tests for --strategy-constraint-off, run through pytester: items turn
a constraint off in every strategy or in one, directed vectors do not change,
factories see the names in options.constraints_off, an item that matches no
constraint is a usage error in a run of the whole suite and a red line in a
narrowed one, malformed items fail when the command line is parsed, the header
and -v show what is off, and a raising constraint names the constraint turned
off before it.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json
import textwrap

import pytest

pytest_plugins = ["pytester"]


# Two strategies whose constraint named never rejects every draw. A strategy
# that fails stops the collection of its module, so each has a module of its own.
NEVER = """
    from pytest_strategy import Parameter, RNGInteger, TestArg, register

    def make():
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": lambda v: False},
            max_retries=5,
        )

    @register("co_burst")
    def burst():
        return make()

    @register("co_other")
    def other():
        return make()
"""


def _tests(*names):
    """One test module per strategy name: test_<name>.py with one test using it."""
    return {f"test_{name}": f"""
            from pytest_strategy import strategy

            @strategy("{name}")
            def test_{name}(x):
                pass
            """ for name in names}


@pytest.mark.parametrize(
    ("flags", "failing"),
    [
        ([], ["co_burst", "co_other"]),
        (["--strategy-constraint-off=never"], []),
        # Aimed at co_other only: co_burst still fails
        (["--strategy-constraint-off=co_other:never"], ["co_burst"]),
    ],
    ids=["none", "everywhere", "one_strategy"],
)
def test_a_constraint_that_rejects_every_draw_can_be_turned_off(pytester, flags, failing):
    pytester.makepyfile(strategies=NEVER, **_tests("co_burst", "co_other"))

    result = pytester.runpytest("--rng-seed=1", *flags)

    for name in failing:
        result.stdout.fnmatch_lines(
            [f"*Error generating samples for strategy '{name}': Could not generate*never=5*"]
        )
    for name in {"co_burst", "co_other"} - set(failing):
        result.stdout.no_fnmatch_line(f"*strategy '{name}'*")
    if failing:
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.assert_outcomes(errors=len(failing))
    else:
        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=20)


# Two strategies with the same two constraints: a rejects every draw, and so does
# b. Each factory writes the names it sees in options.constraints_off to
# <strategy>.json.
TWO_CONSTRAINTS = """
    import json
    from pathlib import Path

    from pytest_strategy import Parameter, RNGInteger, TestArg, register

    def make(options):
        Path(f"{options.strategy}.json").write_text(json.dumps(sorted(options.constraints_off)))
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"a": lambda v: False, "b": lambda v: v.x > 9},
            max_retries=5,
        )

    @register("co_s")
    def s(options):
        return make(options)

    @register("co_t")
    def t(options):
        return make(options)
"""


@pytest.mark.parametrize(
    ("flags", "failing", "seen"),
    [
        # S:a,b turns a off in S only, and b everywhere
        (
            ["--strategy-constraint-off=co_s:a,b"],
            {"co_t": "a=5"},
            {"co_s": ["a", "b"], "co_t": ["b"]},
        ),
        (
            ["--strategy-constraint-off=co_s:a", "--strategy-constraint-off=b"],
            {"co_t": "a=5"},
            {"co_s": ["a", "b"], "co_t": ["b"]},
        ),
        (["--strategy-constraint-off=a,b"], {}, {"co_s": ["a", "b"], "co_t": ["a", "b"]}),
        (
            ["--strategy-constraint-off=a", "--strategy-constraint-off=b"],
            {},
            {"co_s": ["a", "b"], "co_t": ["a", "b"]},
        ),
        # One of the two alone: the other still rejects every draw in both
        (["--strategy-constraint-off=b"], {"co_s": "a=5", "co_t": "a=5"}, {"co_t": ["b"]}),
        (["--strategy-constraint-off=a"], {"co_s": "b=5", "co_t": "b=5"}, {"co_s": ["a"]}),
    ],
    ids=["aimed_and_bare", "repeated", "comma", "repeated_bare", "b_only", "a_only"],
)
def test_items_are_independent(pytester, flags, failing, seen):
    pytester.makepyfile(strategies=TWO_CONSTRAINTS, **_tests("co_s", "co_t"))

    result = pytester.runpytest("--rng-seed=1", *flags)

    for name, counts in failing.items():
        result.stdout.fnmatch_lines(
            [
                f"*Error generating samples for strategy '{name}': Could not generate random "
                f"row 0 after max_retries=5 draws. Rejected by (first failing constraint per "
                f"draw): {counts}. First rows rejected:*"
            ]
        )
    for name in {"co_s", "co_t"} - set(failing):
        result.stdout.no_fnmatch_line(f"*strategy '{name}'*")
    if failing:
        result.assert_outcomes(errors=len(failing))
    else:
        result.assert_outcomes(passed=20)
    # A factory that declares options sees the bare names and those aimed at it
    for name, names in seen.items():
        assert json.loads((pytester.path / f"{name}.json").read_text()) == names


# Writes the values of every collected row to rows.json
DUMP_ROWS = """
    import json

    def pytest_collection_modifyitems(session, config, items):
        rows = [[item.name, item.callspec.params] for item in items]
        (config.rootpath / "rows.json").write_text(json.dumps(rows))
"""


def test_directed_vectors_are_the_same_with_and_without_the_flag(pytester):
    pytester.makeconftest(DUMP_ROWS)
    pytester.makepyfile(test_co_directed="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("co_directed")
        def directed():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                directed_vectors={"odd": (1,), "even": (2,), "nine": {"x": 9}},
                vector_constraints={"even": lambda v: v.x % 2 == 0},
            )

        @strategy("co_directed")
        def test_directed(x):
            pass
        """)

    def rows(*flags):
        pytester.runpytest("--rng-seed=3", "--collect-only", *flags).assert_outcomes()
        return json.loads((pytester.path / "rows.json").read_text())

    on = rows()
    off = rows("--strategy-constraint-off=even")

    assert [params for _, params in on[:3]] == [{"x": 1}, {"x": 2}, {"x": 9}]
    assert on[:3] == off[:3]
    assert all(params["x"] % 2 == 0 for _, params in on[3:])
    # The flag took effect: with the constraint off, odd values are drawn too
    assert any(params["x"] % 2 for _, params in off[3:])


@pytest.mark.parametrize(
    ("flags", "rows", "summary"),
    [
        ([], 2, r"rejected: even=2; left out: 2 combinations$"),
        (["--strategy-constraint-off=even"], 4, r"off: even$"),
    ],
    ids=["on", "off"],
)
def test_auto_generates_the_combinations_the_constraint_left_out(pytester, flags, rows, summary):
    pytester.makepyfile(test_co_auto="""
        from pytest_strategy import Parameter, Series, TestArg, register, strategy

        @register("co_auto")
        def auto():
            return Parameter(
                TestArg("ch", rng_type=Series([0, 1, 2, 3])),
                vector_constraints={"even": lambda v: v.ch % 2 == 0},
            )

        @strategy("co_auto")
        def test_auto(ch):
            pass
        """)

    result = pytester.runpytest("-v", "--nsamples=auto", *flags)

    result.assert_outcomes(passed=rows)
    result.stdout.re_match_lines(
        [
            rf"  co_auto \(test_co_auto\.py\): 1 test\(s\), 0 directed, 0 random, {rows} "
            rf"exhaustive rows; nsamples=auto from --nsamples; {summary}"
        ]
    )


def test_a_raising_constraint_names_the_one_turned_off_before_it(pytester):
    pytester.makepyfile(test_co_ratio="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        def ratio(v):
            return 64 / v.len > 4

        @register("co_ratio")
        def factory():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(0, 9)),
                TestArg("len", value=0),
                vector_constraints={"nonzero": lambda v: v.len != 0, "ratio": ratio},
            )

        @strategy("co_ratio")
        def test_ratio(addr, len):
            pass
        """)

    result = pytester.runpytest("--strategy-constraint-off=nonzero")

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "In test_ratio: Error generating samples for strategy 'co_ratio': Constraint "
            "'ratio' raised ZeroDivisionError on random row 0, Vector(addr=*, len=0): "
            "division by zero (constraint 'nonzero' before it is turned off by "
            "--strategy-constraint-off)",
            '*test_co_ratio.py", line 4, in ratio',
            "*return 64 / v.len > 4",
            "ZeroDivisionError: division by zero",
        ]
    )
    result.stdout.no_fnmatch_line("*parameters.py*")


# A strategy whose never constraint rejects every draw, its directed vector
# (which constraints never check), and a strategy without constraints
UNMATCHED = {
    "test_co_um_never": """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("co_um_never")
        def never():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                directed_vectors={"zeros": (0,)},
                vector_constraints={"never": lambda v: False},
                max_retries=5,
            )

        @strategy("co_um_never")
        def test_never(x):
            pass
        """,
    "test_co_um_plain": """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("co_um_plain")
        def plain():
            return Parameter(TestArg("y", rng_type=RNGInteger(0, 9)))

        @strategy("co_um_plain")
        def test_plain(y):
            pass
        """,
}

UNMATCHED_ITEM = "--strategy-constraint-off=nevr matched no constraint. Did you mean 'never'?"

# The rest of a module whose strategy has the constraint aligned. The tests put
# lines before it that skip the module, fail its import, or set BROKEN, which
# makes the factory raise.
ALIGNED = textwrap.dedent("""
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    BROKEN = globals().get("BROKEN", False)

    @register("co_um_aligned")
    def aligned():
        if BROKEN:
            raise RuntimeError("the factory broke")
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 9)),
            directed_vectors={"zero": (0,)},
            vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        )

    @strategy("co_um_aligned")
    def test_aligned(addr):
        pass
    """)
UNMATCHED_MESSAGE = (
    f"{UNMATCHED_ITEM} Constraints by strategy: co_um_never: never; co_um_plain: none"
)


class TestUnmatchedItems:
    @pytest.fixture(autouse=True)
    def _project(self, pytester):
        pytester.makepyfile(**UNMATCHED)

    def test_a_whole_suite_run_fails_with_a_usage_error(self, pytester):
        # The directed rows only, so co_um_never collects (its never is not evaluated)
        result = pytester.runpytest("--vector-mode=directed_only", "--strategy-constraint-off=nevr")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines([f"ERROR: {UNMATCHED_MESSAGE}"])

    def test_a_run_with_a_collection_error_prints_a_red_line(self, pytester):
        # never rejects every draw, so co_um_never fails to collect, and a module
        # that fails stops collecting the tests after it: the run counts as narrowed
        result = pytester.runpytest("--strategy-constraint-off=nevr")

        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines([f"pytest-strategies: {UNMATCHED_MESSAGE}"])

    def test_a_name_that_exists_counts_even_where_it_is_not_evaluated(self, pytester):
        result = pytester.runpytest(
            "--vector-mode=directed_only", "--strategy-constraint-off=never"
        )

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=1, skipped=1)

    def test_an_aimed_item_is_matched_in_its_strategy_only(self, pytester):
        result = pytester.runpytest(
            "--vector-mode=directed_only", "--strategy-constraint-off=co_um_plain:never"
        )

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "ERROR: --strategy-constraint-off=co_um_plain:never matched no constraint. "
                "Did you mean 'co_um_never:never'? Constraints by strategy: co_um_never: "
                "never; co_um_plain: none"
            ]
        )

    def test_every_unmatched_item_is_reported(self, pytester):
        result = pytester.runpytest(
            "--vector-mode=directed_only", "--strategy-constraint-off=never,nevr,x:y"
        )

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "ERROR: --strategy-constraint-off=nevr matched no constraint. Did you mean "
                "'never'?",
                "--strategy-constraint-off=x:y matched no constraint. Constraints by "
                "strategy: co_um_never: never; co_um_plain: none",
            ]
        )

    @pytest.mark.parametrize(
        ("narrowing", "resolved"),
        [
            (["test_co_um_never.py"], "co_um_never: never"),
            (["test_co_um_never.py::test_never"], "co_um_never: never"),
            (["--lf"], "co_um_never: never; co_um_plain: none"),
            (["--sw"], "co_um_never: never; co_um_plain: none"),
            (["--ignore=test_co_um_other.py"], "co_um_never: never; co_um_plain: none"),
            (["--ignore-glob=*_other.py"], "co_um_never: never; co_um_plain: none"),
        ],
        ids=["path", "node_id", "lf", "sw", "ignore", "ignore_glob"],
    )
    def test_a_narrowed_run_prints_a_red_line_and_goes_on(self, pytester, narrowing, resolved):
        pytester.makepyfile(test_co_um_other="def test_other():\n    pass\n")
        # The directed rows only, so co_um_never collects (its never is not evaluated)
        args = [*narrowing, "--vector-mode=directed_only"]
        without = pytester.runpytest(*args)

        result = pytester.runpytest(*args, "--strategy-constraint-off=nevr")

        assert without.ret == pytest.ExitCode.OK
        assert result.ret == without.ret
        assert result.parseoutcomes() == without.parseoutcomes()
        # After pytest's collection report, listing the strategies this run resolved
        result.stdout.fnmatch_lines(
            [
                "collected * item*",
                f"pytest-strategies: {UNMATCHED_ITEM} Constraints by strategy: {resolved}",
            ]
        )
        assert "ERROR: --strategy-constraint-off" not in result.stderr.str()
        without.stdout.no_fnmatch_line("*matched no constraint*")

    def test_a_narrowed_run_keeps_a_failing_exit_code(self, pytester):
        # co_um_never fails to collect, with or without the unmatched item
        without = pytester.runpytest("test_co_um_never.py")

        result = pytester.runpytest("test_co_um_never.py", "--strategy-constraint-off=nevr")

        assert without.ret == result.ret == pytest.ExitCode.INTERRUPTED
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [f"pytest-strategies: {UNMATCHED_ITEM} Constraints by strategy: " "co_um_never: never"]
        )

    def test_a_narrowed_run_that_passes_still_passes(self, pytester):
        result = pytester.runpytest("test_co_um_plain.py", "--strategy-constraint-off=never")

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=10)
        # co_um_never was not resolved, so its constraint is unknown in this run
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: --strategy-constraint-off=never matched no constraint. "
                "Constraints by strategy: co_um_plain: none"
            ]
        )

    @pytest.mark.parametrize(
        ("prefix", "ret"),
        [
            ('import pytest\npytest.importorskip("co_um_not_installed")\n', pytest.ExitCode.OK),
            (
                'import pytest\npytest.skip("not here", allow_module_level=True)\n',
                pytest.ExitCode.OK,
            ),
            ("import co_um_not_installed\n", pytest.ExitCode.INTERRUPTED),
            ("BROKEN = True\n", pytest.ExitCode.INTERRUPTED),
        ],
        ids=["importorskip", "module_skip", "import_error", "factory_raises"],
    )
    def test_a_run_that_did_not_resolve_a_module_prints_a_red_line(self, pytester, prefix, ret):
        """A whole-suite run whose module was skipped or failed did not resolve that
        module's strategies, so a correct item cannot be told from a typo."""
        pytester.makepyfile(test_co_um_aligned=prefix + ALIGNED)
        args = ["--vector-mode=directed_only"]
        without = pytester.runpytest(*args)

        result = pytester.runpytest(*args, "--strategy-constraint-off=co_um_aligned:aligned")

        assert without.ret == result.ret == ret
        assert result.parseoutcomes() == without.parseoutcomes()
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: --strategy-constraint-off=co_um_aligned:aligned matched no "
                "constraint. Constraints by strategy: co_um_never: never; co_um_plain: none"
            ]
        )
        assert "ERROR: --strategy-constraint-off" not in result.stderr.str()

    def test_not_checked_under_list_strategies(self, pytester):
        result = pytester.runpytest(
            "--list-strategies", "--vector-mode=directed_only", "--strategy-constraint-off=nevr"
        )

        assert result.ret == pytest.ExitCode.OK
        result.stdout.fnmatch_lines(["*co_um_never", "*co_um_plain"])
        assert "matched no constraint" not in result.stdout.str() + result.stderr.str()

    def test_not_checked_without_a_resolved_parameter_strategy(self, pytester):
        # The strategies are registered in strategies.py, but no collected test uses them
        pytester.makepyfile(
            strategies=UNMATCHED["test_co_um_never"].split("@strategy")[0],
            test_co_um_never="def test_plain():\n    pass\n",
            test_co_um_plain="def test_other():\n    pass\n",
        )

        result = pytester.runpytest("--strategy-constraint-off=nevr")

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=2)
        assert "matched no constraint" not in result.stdout.str() + result.stderr.str()

    def test_under_xdist_a_whole_suite_run_stops_with_the_message(self, pytester):
        pytest.importorskip("xdist")

        result = pytester.runpytest_subprocess(
            "-p",
            "no:cacheprovider",
            "--vector-mode=directed_only",
            "--strategy-constraint-off=nevr",
            "-n",
            "2",
        )

        output = result.stdout.str()
        assert result.ret == pytest.ExitCode.INTERRUPTED
        assert "INTERNALERROR" not in output
        assert UNMATCHED_MESSAGE in " ".join(output.split())
        # The workers ran nothing
        result.assert_outcomes()

    def test_under_xdist_a_run_that_skipped_a_module_runs_its_tests(self, pytester):
        pytest.importorskip("xdist")
        pytester.makepyfile(
            test_co_um_aligned='import pytest\npytest.importorskip("co_um_not_installed")\n'
            + ALIGNED
        )

        result = pytester.runpytest_subprocess(
            "-p",
            "no:cacheprovider",
            "--vector-mode=directed_only",
            "--strategy-constraint-off=aligned",
            "-n",
            "2",
        )

        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        result.assert_outcomes(passed=1, skipped=2)
        result.stdout.fnmatch_lines(
            ["pytest-strategies: --strategy-constraint-off=aligned matched no constraint. *"]
        )

    def test_under_xdist_a_narrowed_run_prints_the_workers_red_line(self, pytester):
        pytest.importorskip("xdist")

        result = pytester.runpytest_subprocess(
            "-p",
            "no:cacheprovider",
            "test_co_um_plain.py",
            "--strategy-constraint-off=never",
            "-n",
            "2",
        )

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=10)
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: --strategy-constraint-off=never matched no constraint. "
                "Constraints by strategy: co_um_plain: none"
            ]
        )


# Two folders under testpaths, with a strategy each; the ini turns off the
# constraint of the strategy in tests/a for every run
FOLDERS = {
    "tests/a/test_co_sd_a": """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("co_sd_a")
        def small():
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                vector_constraints={"small": lambda v: v.x < 5},
            )

        @strategy("co_sd_a")
        def test_a(x):
            pass
        """,
    "tests/b/test_co_sd_b": """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("co_sd_b")
        def plain():
            return Parameter(TestArg("y", rng_type=RNGInteger(0, 9)))

        @strategy("co_sd_b")
        def test_b(y):
            pass
        """,
}


class TestTheItemInAddopts:
    """
    A run without arguments counts as the whole suite only when it starts in the
    rootdir: pytest then collects testpaths, and from a folder below the rootdir
    only that folder. So an item kept in addopts works in both.
    """

    @staticmethod
    def _ini(pytester, item):
        pytester.makeini(f"""
            [pytest]
            testpaths = tests
            addopts = --strategy-constraint-off={item}
            """)

    @pytest.fixture(autouse=True)
    def _project(self, pytester):
        pytester.makepyfile(**FOLDERS)

    def test_a_run_from_the_rootdir_matches_the_item(self, pytester):
        self._ini(pytester, "co_sd_a:small")

        result = pytester.runpytest("-v")

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=20)
        result.stdout.fnmatch_lines(["  co_sd_a (*): *; off: small"])
        assert "matched no constraint" not in result.stdout.str() + result.stderr.str()

    def test_a_typo_stops_a_run_from_the_rootdir(self, pytester):
        self._ini(pytester, "co_sd_a:smal")

        result = pytester.runpytest()

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "ERROR: --strategy-constraint-off=co_sd_a:smal matched no constraint. Did you "
                "mean 'co_sd_a:small'? Constraints by strategy: co_sd_a: small; co_sd_b: none"
            ]
        )

    @pytest.mark.parametrize(
        ("folder", "args"),
        [("tests/b", []), (".", ["tests/b"]), ("tests", ["b"])],
        ids=["from_the_folder", "path_from_the_rootdir", "path_from_another_folder"],
    )
    def test_a_run_of_another_folder_prints_a_red_line(self, pytester, monkeypatch, folder, args):
        self._ini(pytester, "co_sd_a:small")
        monkeypatch.chdir(pytester.path / folder)

        result = pytester.runpytest(*args)

        assert result.ret == pytest.ExitCode.OK
        result.assert_outcomes(passed=10)
        result.stdout.fnmatch_lines(
            [
                "pytest-strategies: --strategy-constraint-off=co_sd_a:small matched no "
                "constraint. Constraints by strategy: co_sd_b: none"
            ]
        )
        assert "ERROR: --strategy-constraint-off" not in result.stderr.str()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (":x", "the item ':x' has no strategy name before ':'"),
        ("x:", "the item 'x:' has no constraint name"),
        ("a b", "'a b' contains whitespace"),
        ("a,,b", "'a,,b' has an empty item"),
    ],
)
def test_malformed_items_are_usage_errors_at_parse_time(pytester, value, message):
    pytester.makepyfile(**UNMATCHED)

    result = pytester.runpytest(f"--strategy-constraint-off={value}")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        [f"*error: argument --strategy-constraint-off: {message}. Expected ITEM*"]
    )
    # Nothing was collected
    assert "collected" not in result.stdout.str()


CACHED = """
    import functools

    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    @register("co_cached")
    @functools.cache
    def cached():
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 9)),
            vector_constraints={"never": lambda v: False, "small": lambda v: v.x < 5},
            max_retries=50,
        )

    @strategy("co_cached")
    def test_cached(x):
        assert x < 5

    def test_the_cached_parameter_keeps_its_constraints():
        # The Parameter the plugin generated the rows from
        assert list(cached().vector_constraints) == ["never", "small"]
"""


class TestReporting:
    def test_the_header_line_appears_only_with_the_option(self, pytester):
        pytester.makepyfile(test_co_cached=CACHED)

        with_option = pytester.runpytest(
            "--strategy-constraint-off=co_cached:never,small",
            "--strategy-constraint-off=co_cached:never",
        )
        without = pytester.runpytest("--vector-mode=test")

        with_option.stdout.fnmatch_lines(
            [
                "pytest-strategies: RNG seed = *",
                "pytest-strategies: constraints off: co_cached:never, small",
            ]
        )
        without.stdout.no_fnmatch_line("*constraints off*")

    def test_verbose_summary_shows_off_and_a_cached_factory_keeps_its_constraints(self, pytester):
        pytester.makepyfile(test_co_cached=CACHED)

        result = pytester.runpytest("-v", "--nsamples=4", "--strategy-constraint-off=never")

        result.assert_outcomes(passed=5)
        result.stdout.re_match_lines(
            [
                r"  co_cached \(test_co_cached\.py\): 1 test\(s\), 0 directed, 4 random rows; "
                r"nsamples=4 from --nsamples; rejected: small=\d+; off: never$"
            ]
        )
