"""
End-to-end tests for the 3.0.0 changes, run through pytester: plain decorators,
factory references, folder-scoped names, lazy loading, the private RNG, the size
guard and the reporting.

Subprocess runs give each test a fresh process (its own registry and sys.path).
"""

import re

import pytest

pytest_plugins = ["pytester"]


def _node_ids(result):
    """Return the node IDs printed by --collect-only -q."""
    return [line for line in result.outlines if "::" in line]


def _default_strategy(low, marker=None):
    """A strategies file registering "default" with values from low to low + 9."""
    side_effect = f"open({marker!r}, 'w').close()\n" if marker else ""
    return (
        "from pytest_strategy import Parameter, RNGInteger, TestArg, register\n"
        f"{side_effect}\n"
        '@register("default")\n'
        "def default(nsamples):\n"
        f"    return Parameter(TestArg('x', rng_type=RNGInteger({low}, {low + 9})), nsamples=3)\n"
    )


def _range_test(name, low):
    return (
        "from pytest_strategy import strategy\n\n"
        '@strategy("default")\n'
        f"def {name}(x):\n"
        f"    assert {low} <= x <= {low + 9}\n"
    )


class TestPlainDecorators:
    def test_plain_and_class_decorators_give_the_same_ids(self, pytester):
        new_api = (
            "from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy\n\n"
            '@register("v3_same")\n'
            "def same(nsamples):\n"
            "    return Parameter(TestArg('x', rng_type=RNGInteger(0, 10**9)))\n\n"
            '@strategy("v3_same")\n'
            "def test_same(x):\n"
            "    pass\n"
        )
        old_api = (
            new_api.replace(
                "from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy",
                "from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg",
            )
            .replace("@register(", "@Strategy.register(")
            .replace("@strategy(", "@Strategy.strategy(")
        )
        args = ("-p", "no:cacheprovider", "--rng-seed=7", "--collect-only", "-q")

        pytester.makepyfile(test_same=new_api)
        new = _node_ids(pytester.runpytest_subprocess(*args))
        pytester.makepyfile(test_same=old_api)
        old = _node_ids(pytester.runpytest_subprocess(*args))

        assert len(new) == 10
        assert new == old

    @pytest.mark.parametrize(
        "path",
        [
            "pytest_strategy.strategy.PytestStrategiesWarning",
            "pytest_strategy.PytestStrategiesWarning",
        ],
    )
    def test_both_warning_paths_work_in_filterwarnings(self, pytester, path):
        pytester.makeini(f"[pytest]\nfilterwarnings =\n    error::{path}\n")
        pytester.makepyfile(test_warn="""
            import warnings
            from pytest_strategy import PytestStrategiesWarning

            def test_warn():
                warnings.warn("boom", PytestStrategiesWarning)
            """)

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*PytestStrategiesWarning: boom*"])


class TestFactoryReference:
    SOURCE = """
        from dataclasses import dataclass

        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("v3_ref")
        def registered(nsamples):
            return Parameter(
                TestArg("a", rng_type=RNGInteger(0, 10**9)),
                TestArg("b", rng_type=RNGInteger(0, 10**9)),
                nsamples=4,
            )

        def unregistered():
            return Parameter(TestArg("n", rng_type=RNGInteger(0, 9)), nsamples=2)

        @dataclass
        class Pair:
            a: int
            b: int

        @strategy({ref})
        def test_named(a, b):
            pass

        @strategy(registered)
        def test_dataclass(pair: Pair):
            assert isinstance(pair, Pair)

        @strategy(unregistered)
        def test_unregistered(n):
            assert 0 <= n <= 9
        """

    def test_registered_and_unregistered_factories(self, pytester):
        pytester.makepyfile(test_ref=self.SOURCE.format(ref="registered"))

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=10)

    def test_ids_do_not_depend_on_how_the_factory_is_referenced(self, pytester):
        args = ("-p", "no:cacheprovider", "--rng-seed=3", "--collect-only", "-q")
        pytester.makepyfile(test_ref=self.SOURCE.format(ref="registered"))
        by_object = _node_ids(pytester.runpytest_subprocess(*args))
        pytester.makepyfile(test_ref=self.SOURCE.format(ref='"v3_ref"'))
        by_name = _node_ids(pytester.runpytest_subprocess(*args))

        assert [i for i in by_object if "test_named" in i]
        assert by_object == by_name


class TestScopedNames:
    @pytest.fixture
    def project(self, pytester):
        pytester.makeini("[pytest]\ntestpaths = tests\n")
        pytester.makepyfile(
            **{
                "tests/strategies": _default_strategy(1000),
                "tests/esm/strategies": _default_strategy(100),
                "tests/esm/test_esm": _range_test("test_esm", 100),
                "tests/dma/strategies": _default_strategy(200, marker="dma_loaded"),
                "tests/dma/test_dma": _range_test("test_dma", 200),
                "tests/other/test_other": _range_test("test_other", 1000),
                "tests/test_top": _range_test("test_top", 1000),
            }
        )
        return pytester

    def test_nearest_folder_wins(self, project):
        result = project.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=12)

    def test_unrelated_folder_is_not_loaded(self, project):
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "tests/esm")

        result.assert_outcomes(passed=3)
        assert not (project.path / "dma_loaded").exists()

    def test_list_strategies_loads_every_folder(self, project):
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "--list-strategies")

        result.stdout.fnmatch_lines(
            [
                "*✓ default (tests/dma/strategies.py)",
                "*✓ default (tests/esm/strategies.py)",
                "*✓ default (tests/strategies.py)",
            ]
        )
        assert (project.path / "dma_loaded").exists()

    def test_misspelled_name_suggests_the_visible_names(self, project):
        project.makepyfile(
            **{
                "tests/esm/test_typo": (
                    "from pytest_strategy import strategy\n\n"
                    '@strategy("defualt")\n'
                    "def test_typo(x):\n"
                    "    pass\n"
                )
            }
        )

        result = project.runpytest_subprocess("-p", "no:cacheprovider", "tests/esm")

        result.stdout.fnmatch_lines(
            [
                "In test_typo: Strategy 'defualt' not found. Available strategies: ['default']. "
                "Did you mean 'default'?"
            ]
        )

    def test_sibling_folder_is_used_when_it_is_the_only_one(self, pytester):
        pytester.makepyfile(
            **{
                "tests/strategies/payment_strategies": _default_strategy(300),
                "tests/test_pay": _range_test("test_pay", 300),
            }
        )

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=3)

    def test_two_sibling_folders_are_ambiguous(self, pytester):
        pytester.makepyfile(
            **{
                "tests/a/strategies": _default_strategy(0),
                "tests/b/strategies": _default_strategy(10),
                "tests/test_pay": _range_test("test_pay", 0),
            }
        )

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.stdout.fnmatch_lines(
            [
                "In test_pay: Strategy 'default' is not registered in * or a folder above it, "
                "and several other folders register it:",
                "*a?strategies.py:*default",
                "*b?strategies.py:*default",
            ]
        )

    def test_registration_outside_the_rootdir_is_found_from_any_folder(self, pytester):
        pytester.makepyfile(**{"ext/ext_strategies": _default_strategy(500)})
        pytester.makepyfile(
            **{
                "proj/conftest": (
                    "import sys\n"
                    f"sys.path.insert(0, {str(pytester.path / 'ext')!r})\n"
                    "import ext_strategies\n"
                ),
                "proj/a/test_a": _range_test("test_a", 500),
                "proj/b/c/test_c": _range_test("test_c", 500),
            }
        )
        (pytester.path / "proj" / "pytest.ini").write_text("[pytest]\n")

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "proj")

        result.assert_outcomes(passed=6)

    def test_same_name_in_conftest_and_strategies_file_is_a_usage_error(self, pytester):
        pytester.makeconftest(_default_strategy(0).replace("def default", "def from_conftest"))
        pytester.makepyfile(strategies=_default_strategy(0), test_x=_range_test("test_x", 0))

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            ["ERROR: Strategy 'default' is registered twice in the same folder: *conftest.py*"]
        )


class TestLoading:
    @pytest.mark.parametrize("import_mode", ["prepend", "importlib"])
    def test_test_gets_the_module_the_plugin_loaded(self, pytester, import_mode):
        pytester.makepyfile(
            **{
                "tests/__init__": "",
                "tests/esm/__init__": "",
                "tests/esm/helpers": "LOW = 40\n",
                "tests/esm/strategies": (
                    "from pytest_strategy import Parameter, RNGInteger, TestArg, register\n"
                    "from .helpers import LOW\n\n"
                    '@register("default")\n'
                    "def default(nsamples):\n"
                    "    return Parameter(TestArg('x', rng_type=RNGInteger(LOW, LOW + 9)), nsamples=2)\n"
                ),
                "tests/esm/test_esm": (
                    "from pytest_strategy import strategy\n"
                    "from pytest_strategy._registry import registry\n"
                    "from .strategies import default\n\n"
                    "@strategy(default)\n"
                    "def test_esm(x):\n"
                    "    assert registry.registrations('default')[-1].factory is default\n"
                    "    assert 40 <= x <= 49\n"
                ),
            }
        )

        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", f"--import-mode={import_mode}"
        )

        result.assert_outcomes(passed=2)

    def test_test_strategies_module_keeps_assertion_rewriting(self, pytester):
        pytester.makepyfile(test_strategies="""
            from pytest_strategy import Parameter, TestArg, register, strategy

            @register("v3_rewrite")
            def rewrite(nsamples):
                return Parameter(TestArg("x", value=1), nsamples=1)

            @strategy("v3_rewrite")
            def test_rewrite(x):
                assert x + 1 == 3
            """)
        pytester.makepyfile(**{"sub/test_sub": """
            from pytest_strategy import strategy

            @strategy("v3_rewrite")
            def test_sub(x):
                assert x == 1
            """})

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(["*assert (1 + 1) == 3*"])

    def test_import_time_draws_do_not_depend_on_the_order(self, pytester):
        drawn = (
            "from pytest_strategy import RNG, Parameter, RNGInteger, TestArg, register\n"
            "OFFSET = RNG.integer(0, 10**9)\n\n"
            '@register("v3_{name}")\n'
            "def factory(nsamples):\n"
            "    return Parameter(TestArg('x', rng_type=RNGInteger(OFFSET, OFFSET)), nsamples=1)\n"
        )
        using = (
            "from pytest_strategy import strategy\n\n"
            '@strategy("v3_{name}")\n'
            "def test_{name}(x):\n"
            "    pass\n"
        )
        pytester.makepyfile(
            **{f"{name}/strategies": drawn.format(name=name) for name in ("alpha", "beta")},
            **{f"{name}/test_{name}": using.format(name=name) for name in ("alpha", "beta")},
        )
        args = ("-p", "no:cacheprovider", "--rng-seed=11", "--collect-only", "-q")

        full = _node_ids(pytester.runpytest_subprocess(*args))
        beta = _node_ids(pytester.runpytest_subprocess(*args, "beta"))
        reverse = _node_ids(pytester.runpytest_subprocess(*args, "beta", "alpha"))

        assert [i for i in full if "beta" in i] == beta == [i for i in reverse if "beta" in i]


class TestPrivateRandom:
    SOURCE = """
        import random

        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("v3_rand")
        def rand(nsamples):
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 10**9)), nsamples=1)

        @strategy("v3_rand")
        def test_rand(x):
            print("DRAW", random.random())
        """

    def test_plain_random_is_not_seeded(self, pytester):
        pytester.makepyfile(test_rand=self.SOURCE)
        args = ("-p", "no:cacheprovider", "--rng-seed=5", "-s")

        first = pytester.runpytest_subprocess(*args)
        second = pytester.runpytest_subprocess(*args)

        draws = [
            re.search(r"DRAW (\S+)", result.stdout.str()).group(1) for result in (first, second)
        ]
        assert draws[0] != draws[1]
        # The strategy's values are still reproduced by the seed
        ids = [
            _node_ids(pytester.runpytest_subprocess("--rng-seed=5", "--collect-only", "-q"))
            for _ in range(2)
        ]
        assert ids[0] == ids[1]


class TestSizeGuard:
    SOURCE = """
        from pytest_strategy import Parameter, Series, TestArg, register, strategy

        @register("v3_big")
        def big(nsamples):
            return Parameter(
                TestArg("a", rng_type=Series(range(1000))),
                TestArg("b", rng_type=Series(range(1000))),
                TestArg("c", rng_type=Series(range(1000))),
                {limit}
            )

        @strategy("v3_big")
        def test_big(a, b, c):
            pass
        """

    def test_too_many_combinations_fail_before_generating(self, pytester):
        pytester.makepyfile(test_big=self.SOURCE.format(limit=""))

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--nsamples=auto")

        result.stdout.fnmatch_lines(
            [
                "In test_big: Strategy 'v3_big': --nsamples=auto would generate "
                "1,000,000,000 rows (a=1,000 x b=1,000 x c=1,000), more than the limit of "
                "100,000. Raise it with Parameter(max_exhaustive=...) or the "
                "strategies_max_exhaustive ini option, or use fewer values."
            ]
        )
        assert result.ret == pytest.ExitCode.INTERRUPTED

    def test_limit_can_be_raised_per_strategy_and_per_project(self, pytester):
        small = self.SOURCE.replace("range(1000)", "range(10)")
        pytester.makepyfile(test_big=small.format(limit="max_exhaustive=1000,"))
        per_strategy = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", "--nsamples=auto", "--collect-only", "-q"
        )
        pytester.makepyfile(test_big=small.format(limit=""))
        pytester.makeini("[pytest]\nstrategies_max_exhaustive = 999\n")
        too_low = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--nsamples=auto")
        pytester.makeini("[pytest]\nstrategies_max_exhaustive = 1000\n")
        per_project = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", "--nsamples=auto", "--collect-only", "-q"
        )

        assert len(_node_ids(per_strategy)) == 1000
        too_low.stdout.fnmatch_lines(["*more than the limit of 999*"])
        assert len(_node_ids(per_project)) == 1000

    def test_invalid_ini_value_is_a_usage_error(self, pytester):
        pytester.makepyfile(test_big=self.SOURCE.format(limit=""))
        pytester.makeini("[pytest]\nstrategies_max_exhaustive = lots\n")

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--nsamples=auto")

        result.stdout.fnmatch_lines(
            ["*strategies_max_exhaustive must be an integer >= 1, got 'lots'*"]
        )


class TestReporting:
    SOURCE = """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("v3_report")
        def report(nsamples):
            return Parameter(
                TestArg("x", rng_type=RNGInteger(0, 9)),
                directed_vectors={"zero": (0,)},
                nsamples=4,
            )

        @strategy("v3_report")
        def test_report(x):
            assert {condition}
        """

    @pytest.mark.parametrize("quiet", [False, True])
    def test_failed_run_prints_how_to_reproduce(self, pytester, quiet):
        pytester.makepyfile(test_report=self.SOURCE.replace("{condition}", "x < 0"))
        args = ["-p", "no:cacheprovider", "--rng-seed=21"] + (["-q"] if quiet else [])

        result = pytester.runpytest_subprocess(*args)

        result.stdout.fnmatch_lines(["pytest-strategies: reproduce with --rng-seed=21"])

    def test_passing_run_does_not(self, pytester):
        pytester.makepyfile(test_report=self.SOURCE.replace("{condition}", "x >= 0"))

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

        result.assert_outcomes(passed=5)
        result.stdout.no_fnmatch_line("*reproduce with*")

    def test_verbose_summary_counts_rows(self, pytester):
        pytester.makepyfile(test_report=self.SOURCE.replace("{condition}", "x >= 0"))

        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-v")

        result.stdout.fnmatch_lines(
            [
                "*Strategy Summary*",
                "  v3_report (test_report.py): 1 test(s), 1 directed, 4 random rows; "
                "nsamples=4 from Parameter(nsamples=)",
            ]
        )
