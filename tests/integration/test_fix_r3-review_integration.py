"""
End-to-end tests for the fixes from the final review before 2.0.0, run through
pytester in subprocesses.

Strategy files imported by test modules or conftest.py, a duplicate name in
strategy files with the warning turned into an error, an empty --vector-name,
list vectors, and discovery below symlinks and norecursedirs path patterns.
"""

import os
import re

import pytest

pytest_plugins = ["pytester"]

IMPORTED_STRATEGIES = """
from enum import Enum

from pytest_strategy import RNG, Parameter, RNGEnum, RNGInteger, Strategy, TestArg


class Mode(Enum):
    FAST = 1
    SLOW = 2


# Drawn when the file is imported
OFFSET = RNG.integer(0, 10**6)


@Strategy.register("r3_offset")
def r3_offset(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(OFFSET, OFFSET + 9)), nsamples=2)


@Strategy.register("r3_modes")
def r3_modes(nsamples):
    return Parameter(TestArg("mode", rng_type=RNGEnum(Mode)), nsamples=2)
"""


def _node_ids(result, test_name):
    return sorted(
        line.strip()
        for line in result.outlines
        if re.match(rf"\s*tests/test_\w+\.py::{test_name}\[", line)
    )


class TestImportedStrategyFile:
    """A test module or conftest.py that imports a strategy file gets the loaded module."""

    @pytest.fixture
    def project(self, pytester):
        pytester.makeini("[pytest]\ntestpaths = tests\n")
        tests = pytester.mkdir("tests")
        (tests / "my_strategies.py").write_text(IMPORTED_STRATEGIES)
        (tests / "test_a.py").write_text(
            "from my_strategies import OFFSET, Mode\n"
            "from pytest_strategy import Strategy\n\n"
            '@Strategy.strategy("r3_offset")\n'
            "def test_a(x):\n"
            "    assert OFFSET <= x <= OFFSET + 9\n\n"
            '@Strategy.strategy("r3_modes")\n'
            "def test_mode(mode):\n"
            "    assert isinstance(mode, Mode)\n"
            "    assert mode in (Mode.FAST, Mode.SLOW)\n"
        )
        (tests / "test_b.py").write_text(
            "from pytest_strategy import Strategy\n\n"
            '@Strategy.strategy("r3_offset")\n'
            "def test_b(x):\n"
            "    pass\n"
        )
        return pytester

    def test_same_node_ids_for_the_whole_suite_and_one_file(self, project):
        """The import used to run the file again and redraw OFFSET for later tests."""
        args = ("-p", "no:cacheprovider", "--rng-seed=1", "--collect-only", "-q")

        full = project.runpytest_subprocess(*args)
        alone = project.runpytest_subprocess(*args, "tests/test_b.py")
        reversed_order = project.runpytest_subprocess(*args, "tests/test_b.py", "tests/test_a.py")

        assert _node_ids(full, "test_b")
        assert _node_ids(full, "test_b") == _node_ids(alone, "test_b")
        assert _node_ids(full, "test_b") == _node_ids(reversed_order, "test_b")

    def test_importing_test_gets_the_strategy_classes_and_values(self, project):
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "--rng-seed=3")

        result.assert_outcomes(passed=6)

    def test_strategy_file_imported_by_conftest_is_not_loaded_again(self, project):
        """Its registrations used to come from a second copy with its own Mode class."""
        project.makeconftest(
            "import sys\n"
            "sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), 'tests'))\n"
            "from my_strategies import Mode\n\n"
            "import pytest\n\n"
            "@pytest.fixture\n"
            "def fast():\n"
            "    return Mode.FAST\n"
        )
        (project.path / "tests" / "test_c.py").write_text(
            "from my_strategies import Mode\n"
            "from pytest_strategy import Strategy\n\n"
            '@Strategy.strategy("r3_modes")\n'
            "def test_modes(mode, fast):\n"
            "    assert isinstance(mode, Mode)\n"
            "    assert isinstance(fast, Mode)\n"
        )

        result = project.runpytest_subprocess("-p", "no:cacheprovider", "--rng-seed=3", "-v")

        result.assert_outcomes(passed=8)
        result.stdout.no_fnmatch_line("*Failed to load*")

    def test_test_module_that_is_a_strategy_file_is_still_rewritten(self, pytester):
        """The finder comes after pytest's assertion rewriting hook."""
        pytester.makepyfile(test_strategies="""
            from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg

            @Strategy.register("r3_in_test")
            def r3_in_test(nsamples):
                return Parameter(TestArg("x", rng_type=RNGInteger(1, 1)), nsamples=1)

            @Strategy.strategy("r3_in_test")
            def test_rewritten(x):
                assert x + 3 == -1
            """)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*assert (1 + 3) == -1*"])


class TestDuplicateNameAsError:
    """A name registered twice in one folder stops the run with a usage error."""

    @pytest.mark.parametrize(
        "filterwarnings",
        [
            None,
            "error::pytest_strategy.strategy.PytestStrategiesWarning",
            "error::pytest_strategy.PytestStrategiesWarning",
        ],
    )
    def test_usage_error_names_both_files(self, pytester, filterwarnings):
        if filterwarnings is not None:
            pytester.makeini(f"[pytest]\nfilterwarnings =\n    {filterwarnings}\n")
        for name, low in (("a_strategies", 0), ("b_strategies", 200)):
            pytester.makepyfile(**{name: f"""
                from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg

                @Strategy.register("r3_dup")
                def factory_{low}(nsamples):
                    return Parameter(TestArg("x", rng_type=RNGInteger({low}, {low + 100})))
                """})
        pytester.makepyfile(test_dup="""
            from pytest_strategy import Strategy

            @Strategy.strategy("r3_dup")
            def test_x(x):
                pass
            """)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "ERROR: Strategy 'r3_dup' is registered twice in the same folder: "
                "*b_strategies.py:*factory_200 replaces *a_strategies.py:*factory_0"
            ]
        )


class TestVectorOptions:
    """An empty --vector-name and vectors given as lists."""

    @pytest.fixture
    def project(self, pytester):
        pytester.makepyfile(test_vectors="""
            from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg

            @Strategy.register("r3_single")
            def r3_single(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 10)),
                    directed_vectors={"five": [5]},
                    test_vectors={"six": [6]},
                )

            @Strategy.strategy("r3_single")
            def test_single(x):
                assert isinstance(x, int), repr(x)
            """)
        return pytester

    def test_empty_vector_name_is_a_usage_error(self, project):
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "--vector-name=")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*--vector-name= matched no directed vector*"])

    @pytest.mark.parametrize("args", [("--nsamples=0",), ("--vector-mode=test",)])
    def test_list_vector_gives_the_element(self, project, args):
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "-v", *args)

        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_single[[]x=[56][]] PASSED*"])


class TestDiscoveryScope:
    """Discovery enters symlinked directories and honours norecursedirs path patterns."""

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_strategy_file_in_a_symlinked_test_directory(self, pytester):
        shared = pytester.mkdir("shared")
        (shared / "shared_strategies.py").write_text(
            "from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg\n\n"
            '@Strategy.register("r3_shared")\n'
            "def r3_shared(nsamples):\n"
            '    return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)), nsamples=2)\n'
        )
        (shared / "test_shared.py").write_text(
            "from pytest_strategy import Strategy\n\n"
            '@Strategy.strategy("r3_shared")\n'
            "def test_x(x):\n"
            "    pass\n"
        )
        project = pytester.mkdir("proj")
        (project / "tests").mkdir()
        try:
            (project / "tests" / "shared").symlink_to(shared, target_is_directory=True)
        except OSError as e:
            pytest.skip(f"cannot create a symlink: {e}")
        (project / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
        os.chdir(project)

        whole = pytester.runpytest_subprocess("-p", "no:cacheprovider")
        named = pytester.runpytest_subprocess("-p", "no:cacheprovider", "tests/shared")

        whole.assert_outcomes(passed=2)
        named.assert_outcomes(passed=2)

    def test_norecursedirs_path_pattern_skips_test_data(self, pytester):
        pytester.makeini("[pytest]\ntestpaths = tests\nnorecursedirs = tests/data\n")
        tests = pytester.mkdir("tests")
        (tests / "a_strategies.py").write_text(
            "from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg\n\n"
            '@Strategy.register("r3_real")\n'
            "def r3_real(nsamples):\n"
            '    return Parameter(TestArg("x", rng_type=RNGInteger(0, 100)), nsamples=2)\n'
        )
        (tests / "data" / "sample").mkdir(parents=True)
        (tests / "data" / "sample" / "strategies.py").write_text(
            "from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg\n\n"
            '@Strategy.register("r3_real")\n'
            "def fixture_copy(nsamples):\n"
            '    return Parameter(TestArg("x", rng_type=RNGInteger(1000, 2000)))\n'
        )
        (tests / "test_real.py").write_text(
            "from pytest_strategy import Strategy\n\n"
            '@Strategy.strategy("r3_real")\n'
            "def test_x(x):\n"
            "    assert 0 <= x <= 100\n"
        )

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-W", "error")

        result.assert_outcomes(passed=2)
