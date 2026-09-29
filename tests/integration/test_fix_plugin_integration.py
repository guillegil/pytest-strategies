"""
End-to-end tests for plugin fixes, run through pytester.

Subprocess runs are used where a test needs a fresh process (its own RNG seed,
its own strategy registry) or has to exit the inner session.
"""

import re

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

@Strategy.register("big_ints")
def big_ints(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)
"""

TESTS = """
from pytest_strategy import Strategy

@Strategy.strategy("big_ints")
def test_first(x):
    assert 0 <= x <= 10**9

@Strategy.strategy("big_ints")
def test_second(x):
    assert 0 <= x <= 10**9
"""


class TestDiscoveryUnderDotDirectory:
    """A project whose path contains a dot directory must still be discovered."""

    def test_project_under_dot_directory_runs(self, pytester, monkeypatch):
        project = pytester.path / ".hidden" / "proj"
        project.mkdir(parents=True)
        (project / "pytest.ini").write_text("[pytest]\n")
        (project / "strategies.py").write_text(STRATEGIES)
        (project / "test_dot.py").write_text(TESTS)
        # A strategy file inside a hidden directory of the project stays ignored.
        (project / ".venvlike").mkdir()
        (project / ".venvlike" / "other_strategies.py").write_text(
            STRATEGIES.replace('"big_ints"', '"hidden_strat"')
        )
        monkeypatch.chdir(project)

        result = pytester.runpytest_subprocess()
        result.assert_outcomes(passed=6)

        listing = pytester.runpytest_subprocess("--list-strategies")
        listing.stdout.fnmatch_lines(["*big_ints*"])
        listing.stdout.no_fnmatch_line("*hidden_strat*")


def _seed_from_header(result):
    """Return the seed printed in the 'pytest-strategies: RNG seed = N' header."""
    match = re.search(r"RNG seed = (\d+)", result.stdout.str())
    assert match, result.stdout.str()
    return int(match.group(1))


def _test_ids(result):
    """Return the parametrized test IDs (``test_x[...]``) found in a run's output."""
    return re.findall(r"test_\w+\[[^\]]*\]", result.stdout.str())


class TestUnseededRunReproducibility:
    """The seed printed by an unseeded run must reproduce it exactly."""

    def test_printed_seed_reproduces_module_level_draws(self, pytester):
        """Randomness drawn before the first factory runs must also be reproduced.

        The strategy file draws an offset at import time, before any factory
        runs. Without a seed the global random state used to come from OS
        entropy, so rerunning with the printed seed gave different vectors.
        """
        pytester.makepyfile(offset_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNG

            OFFSET = RNG.integer(0, 10**9)

            @Strategy.register("offset_strat")
            def offset(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(OFFSET, OFFSET + 10**6)), nsamples=3
                )
            """)
        pytester.makepyfile(test_offset="""
            from pytest_strategy import Strategy

            @Strategy.strategy("offset_strat")
            def test_offset(x):
                pass
            """)

        unseeded = pytester.runpytest_subprocess("--collect-only")
        seed = _seed_from_header(unseeded)
        reproduced = pytester.runpytest_subprocess("--collect-only", f"--rng-seed={seed}")

        unseeded_ids = _test_ids(unseeded)
        assert len(unseeded_ids) == 3
        assert _test_ids(reproduced) == unseeded_ids


class TestXdistSeedSharing:
    """pytest-xdist workers must all use the controller's seed."""

    def test_unseeded_xdist_run_collects_the_same_tests_on_every_worker(self, pytester):
        pytest.importorskip("xdist")
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_xdist=TESTS)

        result = pytester.runpytest_subprocess("-n", "2")

        result.stdout.no_fnmatch_line("*Different tests were collected*")
        result.assert_outcomes(passed=6)

    def test_worker_uses_seed_from_workerinput(self, pytester):
        """Without --rng-seed, a worker takes the seed the controller sent."""
        pytester.makeconftest("""
            import pytest

            @pytest.hookimpl(tryfirst=True)
            def pytest_configure(config):
                # Stand in for an xdist worker (workerinput is set before configure).
                config.workerinput = {"pytest_strategies_seed": 4242}
            """)
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_worker=TESTS)

        worker = pytester.runpytest_subprocess("-p", "no:xdist", "--collect-only")
        seeded = pytester.runpytest_subprocess(
            "-p", "no:xdist", "--collect-only", "--rng-seed=4242"
        )

        assert _seed_from_header(worker) == 4242
        assert _test_ids(worker) == _test_ids(seeded)

    def test_plugin_works_without_xdist(self, pytester):
        """The optional xdist hook must not break a run where xdist is absent."""
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_no_xdist=TESTS)

        result = pytester.runpytest_subprocess("-p", "no:xdist")

        result.assert_outcomes(passed=6)


class TestPerTestRandomStreams:
    """Each (strategy, test) pair draws from its own reproducible stream."""

    TWIN_STRATEGIES = """
        from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

        @Strategy.register("twin_a")
        def twin_a(nsamples):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)

        @Strategy.register("twin_b")
        def twin_b(nsamples):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=3)
        """

    @staticmethod
    def _ids_by_test(result):
        """Map each test name to the list of IDs it was collected with."""
        ids = {}
        for test_id in _test_ids(result):
            ids.setdefault(test_id.split("[")[0], []).append(test_id.split("[", 1)[1])
        return ids

    @staticmethod
    def _test_module(*names):
        """Build a test module whose tests use the strategies in the given order."""
        blocks = [
            f"@Strategy.strategy({strategy!r})\ndef {test}(x):\n    pass\n"
            for test, strategy in names
        ]
        return "from pytest_strategy import Strategy\n\n" + "\n".join(blocks)

    def test_tests_and_strategies_get_different_values(self, pytester):
        pytester.makepyfile(twin_strategies=self.TWIN_STRATEGIES)
        pytester.makepyfile(
            test_twins=self._test_module(
                ("test_one", "twin_a"), ("test_two", "twin_a"), ("test_three", "twin_b")
            )
        )

        ids = self._ids_by_test(pytester.runpytest_subprocess("--collect-only", "--rng-seed=42"))

        # Same strategy, different tests: different values.
        assert ids["test_one"] != ids["test_two"]
        # Identical strategy definitions under different names: different values.
        assert ids["test_one"] != ids["test_three"]

    def test_values_do_not_depend_on_collection_order(self, pytester):
        pytester.makepyfile(twin_strategies=self.TWIN_STRATEGIES)
        pytester.makepyfile(
            test_twins=self._test_module(("test_one", "twin_a"), ("test_two", "twin_b"))
        )
        forward = self._ids_by_test(
            pytester.runpytest_subprocess("--collect-only", "--rng-seed=42")
        )

        pytester.makepyfile(
            test_twins=self._test_module(("test_two", "twin_b"), ("test_one", "twin_a"))
        )
        backward = self._ids_by_test(
            pytester.runpytest_subprocess("--collect-only", "--rng-seed=42")
        )

        assert len(forward["test_one"]) == 3
        assert forward == backward


class TestNestedSessionSeed:
    """An in-process session's --rng-seed must not leak out of it."""

    def test_inner_rng_seed_is_restored_after_the_run(self, pytester):
        from pytest_strategy import RNG

        pytester.makepyfile(nested_strategies=STRATEGIES.replace("big_ints", "nested_ints"))
        pytester.makepyfile(test_nested=TESTS.replace("big_ints", "nested_ints"))
        RNG.seed(7)  # a known seed for the enclosing session

        seeded = pytester.runpytest_inprocess("--rng-seed=42")
        seeded.assert_outcomes(passed=6)
        assert _seed_from_header(seeded) == 42
        assert RNG.get_seed() == 7

        # A later run without --rng-seed uses the enclosing seed, not the leaked 42.
        unseeded = pytester.runpytest_inprocess()
        unseeded.assert_outcomes(passed=6)
        assert _seed_from_header(unseeded) == 7


class TestSkipInStrategyFile:
    """pytest.skip / importorskip in a strategy file must not crash the session."""

    def test_importorskip_in_strategy_test_module_skips_only_that_module(self, pytester):
        """A test module named *_strategies.py is also loaded as a strategy file."""
        pytester.makepyfile(test_optional_strategies="""
            import pytest

            missing = pytest.importorskip("pytest_strategies_missing_module")

            from pytest_strategy import Strategy

            @Strategy.register("optional_strat")
            def optional(nsamples):
                return ("x",), [(1,)]

            def test_optional():
                pass
            """)
        pytester.makepyfile(test_other="def test_other():\n    pass\n")

        result = pytester.runpytest_subprocess()

        result.stdout.no_fnmatch_line("*INTERNALERROR*")
        result.assert_outcomes(passed=1, skipped=1)

    def test_module_level_skip_in_strategies_file_is_reported_in_verbose_mode(self, pytester):
        pytester.makepyfile(strategies="""
            import pytest

            pytest.skip("no gpu", allow_module_level=True)

            from pytest_strategy import Strategy

            @Strategy.register("gpu_strat")
            def gpu(nsamples):
                return ("x",), [(1,)]
            """)
        pytester.makepyfile(test_other="def test_other():\n    pass\n")

        quiet = pytester.runpytest_subprocess()
        verbose = pytester.runpytest_subprocess("-v")

        quiet.assert_outcomes(passed=1)
        quiet.stdout.no_fnmatch_line("*pytest-strategies: Skipped*")
        verbose.assert_outcomes(passed=1)
        verbose.stdout.fnmatch_lines(["pytest-strategies: Skipped *strategies.py: no gpu"])

    def test_module_level_fail_in_strategies_file_does_not_crash(self, pytester):
        pytester.makepyfile(strategies="""
            import pytest

            pytest.fail("broken setup")

            from pytest_strategy import Strategy

            @Strategy.register("failing_strat")
            def failing(nsamples):
                return ("x",), [(1,)]
            """)
        pytester.makepyfile(test_other="def test_other():\n    pass\n")

        result = pytester.runpytest_subprocess()

        result.stdout.no_fnmatch_line("*INTERNALERROR*")
        result.assert_outcomes(passed=1)


class TestLoadErrorsAreVisible:
    """A strategy file that fails to load must be reported without -v."""

    def test_failed_import_is_reported_and_named_in_not_found_error(self, pytester):
        pytester.makepyfile(strategies="""
            from pytest_strategies_missing_helper import LIMIT

            from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

            @Strategy.register("limited")
            def limited(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(0, LIMIT)))
            """)
        pytester.makepyfile(test_limited="""
            from pytest_strategy import Strategy

            @Strategy.strategy("limited")
            def test_limited(x):
                pass
            """)

        result = pytester.runpytest_subprocess()

        error = "ModuleNotFoundError: No module named 'pytest_strategies_missing_helper'"
        result.stdout.fnmatch_lines(
            [
                f"pytest-strategies: Warning - Failed to load *strategies.py: {error}",
                "*ValueError: Strategy 'limited' not found. Available strategies: none",
                "*Strategy files that failed to load:",
                f"*strategies.py: {error}",
            ]
        )
        assert result.ret == pytest.ExitCode.INTERRUPTED

    def test_syntax_error_is_reported_in_quiet_mode(self, pytester):
        pytester.makepyfile(strategies="""
            from pytest_strategy import Strategy

            @Strategy.register("broken")
            def broken(nsamples)
                return ("x",), [(1,)]
            """)
        pytester.makepyfile(test_other="def test_other():\n    pass\n")

        result = pytester.runpytest_subprocess("-q")

        result.stdout.fnmatch_lines(
            ["pytest-strategies: Warning - Failed to load *strategies.py: SyntaxError: *"]
        )
        result.assert_outcomes(passed=1)


class TestNsamplesValidation:
    """Invalid --nsamples values are usage errors, reported before collection."""

    @pytest.mark.parametrize("value", ["abc", "-1", "2.5"])
    def test_invalid_value_is_a_usage_error(self, pytester, value):
        pytester.makepyfile(strategies=STRATEGIES)
        pytester.makepyfile(test_ns=TESTS)

        result = pytester.runpytest_subprocess(f"--nsamples={value}")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [f"*argument --nsamples: expected an integer >= 0 or 'auto', got '{value}'"]
        )

    def test_invalid_value_is_rejected_without_strategy_tests(self, pytester):
        pytester.makepyfile(test_plain="def test_plain():\n    pass\n")

        result = pytester.runpytest_subprocess("--nsamples=abc")

        assert result.ret == pytest.ExitCode.USAGE_ERROR

    def test_auto_is_case_insensitive(self, pytester):
        pytester.makepyfile(series_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, Series

            @Strategy.register("series_strat")
            def series(nsamples):
                return Parameter(TestArg("x", rng_type=Series([1, 2, 3])))
            """)
        pytester.makepyfile(test_series="""
            from pytest_strategy import Strategy

            @Strategy.strategy("series_strat")
            def test_series(x):
                assert x in (1, 2, 3)
            """)

        result = pytester.runpytest_subprocess("--nsamples=AUTO")

        result.assert_outcomes(passed=3)
