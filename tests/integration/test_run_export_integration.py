"""
The run export (--strategy-export, the strategies_export_dir ini option and the
pytest_strategies_export_dir hook) and Parameter(pin_nsamples=True), end to end.
"""

import json

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
    from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

    @register("rw")
    def rw(device="esm", width=8):
        return Parameter(
            TestArg("channel", rng_type=Series([0, 1])),
            TestArg("data", rng_type=RNGInteger(0, 2**width - 1)),
            directed_vectors={"zeros": (0, 0)},
            test_vectors={"max": (1, 2**width - 1)},
        )

    @register("pinned")
    def pinned():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("v", rng_type=RNGInteger(0, 9)),
            nsamples="auto",
            pin_nsamples=True,
        )
    """

TESTS = """
    import pytest
    from pytest_strategy import strategy

    @strategy("rw", device="bpam", width=4)
    def test_rw(channel, data, request):
        assert "zeros" not in request.node.name

    @strategy("pinned")
    def test_pinned(ch, v):
        pass

    @pytest.mark.xfail(reason="known")
    @strategy("pinned")
    def test_xfail(ch, v):
        assert False

    def test_plain():
        pass
    """


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_test(document):
    return {test["nodeid"]: test["strategies"] for test in document["tests"]}


@pytest.fixture
def suite(pytester):
    pytester.makepyfile(rx_strategies=STRATEGIES)
    pytester.makepyfile(test_rx=TESTS)
    return pytester


class TestFiles:
    def test_both_files_are_written_with_definitions_and_rows(self, suite):
        result = suite.runpytest_inprocess("--strategy-export=out", "--nsamples=4", "--rng-seed=1")
        result.assert_outcomes(passed=8, failed=1, xfailed=3)
        result.stdout.fnmatch_lines(
            ["pytest-strategies: wrote strategies.json and strategies.vectors.json to *out"]
        )
        definitions = _load(suite.path / "out" / "strategies.json")
        vectors = _load(suite.path / "out" / "strategies.vectors.json")
        assert definitions["schema"] == vectors["schema"] == 1
        assert definitions["kind"] == "strategy_definitions"
        assert vectors["kind"] == "strategy_vectors"
        assert definitions["seed"] == 1 and definitions["nsamples"] == 4
        # Only the tests with a strategy, in collection order
        assert list(_by_test(definitions)) == [
            "test_rx.py::test_rw",
            "test_rx.py::test_pinned",
            "test_rx.py::test_xfail",
        ]
        (rw,) = _by_test(definitions)["test_rx.py::test_rw"]
        assert rw["name"] == "rw"
        assert rw["origin"]["file"] == "rx_strategies.py"
        assert rw["arguments"] == {"device": "bpam", "width": 4}
        assert rw["nsamples"] == {"value": 4, "source": "--nsamples"}
        assert "rows" not in rw
        parameter = rw["parameter"]
        # The definition as the test resolved it: width=4 gives max 15
        assert parameter["arguments"][1]["rng"] == {
            "type": "RNGInteger",
            "min": 0,
            "max": 15,
            "predicate": False,
        }
        assert parameter["directed_vectors"] == [
            {"name": "zeros", "id": "directed-zeros", "values": {"channel": 0, "data": 0}}
        ]
        assert parameter["test_vectors"] == [
            {"name": "max", "id": "test-max", "values": {"channel": 1, "data": 15}}
        ]
        # The vectors file is the same document with the rows
        (rw_rows,) = _by_test(vectors)["test_rx.py::test_rw"]
        assert {k: v for k, v in rw_rows.items() if k != "rows"} == rw
        rows = rw_rows["rows"]
        assert [row["id"] for row in rows] == [
            "directed-zeros",
            "channel=0-rand-0",
            "channel=1-rand-0",
            "channel=0-rand-1",
            "channel=1-rand-1",
        ]
        assert rows[0]["kind"] == "directed" and rows[0]["name"] == "zeros"
        assert rows[0]["items"] == [
            {"nodeid": "test_rx.py::test_rw[directed-zeros]", "outcome": "failed"}
        ]
        assert {row["kind"] for row in rows[1:]} == {"random"}
        for row in rows[1:]:
            assert 0 <= row["values"]["data"] <= 15
            assert [item["outcome"] for item in row["items"]] == ["passed"]

    def test_pinned_auto_and_xfail_outcomes(self, suite):
        suite.runpytest_inprocess("--strategy-export=out", "--nsamples=4")
        vectors = _by_test(_load(suite.path / "out" / "strategies.vectors.json"))
        (pinned,) = vectors["test_rx.py::test_pinned"]
        assert pinned["nsamples"] == {"value": "auto", "source": "Parameter(nsamples=, pinned)"}
        assert pinned["parameter"]["pin_nsamples"] is True
        assert [row["id"] for row in pinned["rows"]] == ["ch=0", "ch=1", "ch=2"]
        assert {row["kind"] for row in pinned["rows"]} == {"exhaustive"}
        (xfail,) = vectors["test_rx.py::test_xfail"]
        outcomes = [item["outcome"] for row in xfail["rows"] for item in row["items"]]
        assert outcomes == ["xfailed"] * 3

    def test_test_vectors_run_in_test_mode(self, suite):
        suite.runpytest_inprocess("--strategy-export=out", "--vector-mode=test")
        vectors = _by_test(_load(suite.path / "out" / "strategies.vectors.json"))
        (rw,) = vectors["test_rx.py::test_rw"]
        assert [(row["id"], row["kind"], row["values"]) for row in rw["rows"]] == [
            ("test-max", "test", {"channel": 1, "data": 15})
        ]

    def test_only_selected_items_and_no_outcome_without_a_run(self, suite):
        suite.runpytest_inprocess("--strategy-export=out", "--collect-only", "-k", "zeros")
        vectors = _load(suite.path / "out" / "strategies.vectors.json")
        (test,) = vectors["tests"]
        assert test["nodeid"] == "test_rx.py::test_rw"
        (row,) = test["strategies"][0]["rows"]
        assert row["items"] == [{"nodeid": "test_rx.py::test_rw[directed-zeros]", "outcome": None}]

    def test_nothing_is_written_without_the_option(self, suite):
        result = suite.runpytest_inprocess()
        assert "strategies.json" not in result.stdout.str()
        assert not list(suite.path.rglob("strategies.json"))

    def test_an_error_writing_is_reported(self, suite):
        (suite.path / "blocked").write_text("a file, not a folder")
        result = suite.runpytest_inprocess("--strategy-export=blocked/out")
        result.stdout.fnmatch_lines(
            ["pytest-strategies: could not write the strategy export to *blocked*out: *"]
        )


class TestWhere:
    def test_ini_option_is_relative_to_the_rootdir(self, suite):
        suite.makeini("[pytest]\nstrategies_export_dir = reports/strategies\n")
        sub = suite.mkdir("sub")
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(sub)
            suite.runpytest_inprocess(str(suite.path))
        assert not (sub / "reports").exists()
        assert (suite.path / "reports" / "strategies" / "strategies.json").is_file()

    def test_hook_from_conftest_wins_over_the_ini_option(self, suite):
        suite.makeini("[pytest]\nstrategies_export_dir = from_ini\n")
        suite.makeconftest("""
            import pytest

            @pytest.hookimpl(optionalhook=True)
            def pytest_strategies_export_dir(config):
                return "from_hook"
            """)
        suite.runpytest_inprocess()
        assert (suite.path / "from_hook" / "strategies.vectors.json").is_file()
        assert not (suite.path / "from_ini").exists()

    def test_the_option_wins_over_the_hook(self, suite):
        suite.makeconftest("""
            import pytest

            @pytest.hookimpl(optionalhook=True)
            def pytest_strategies_export_dir(config):
                return "from_hook"
            """)
        suite.runpytest_inprocess("--strategy-export=from_option")
        assert (suite.path / "from_option" / "strategies.json").is_file()
        assert not (suite.path / "from_hook").exists()

    def test_conftest_can_set_the_option(self, suite):
        suite.makeconftest("""
            def pytest_configure(config):
                config.option.strategy_export = "from_configure"
            """)
        suite.runpytest_inprocess()
        assert (suite.path / "from_configure" / "strategies.json").is_file()


class TestXdist:
    def test_the_controller_writes_the_workers_rows_with_outcomes(self, suite):
        pytest.importorskip("xdist")
        local = suite.runpytest_inprocess("--strategy-export=local", "--rng-seed=7")
        local.assert_outcomes(passed=14, failed=1, xfailed=3)
        result = suite.runpytest_subprocess(
            "--strategy-export=dist", "--rng-seed=7", "-n", "2", "-p", "no:cacheprovider"
        )
        result.assert_outcomes(passed=14, failed=1, xfailed=3)
        dist = _load(suite.path / "dist" / "strategies.vectors.json")
        same = _load(suite.path / "local" / "strategies.vectors.json")
        assert dist["tests"] == same["tests"]


class TestPinNsamplesValidation:
    def test_needs_a_count(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        with pytest.raises(ValueError, match="pin_nsamples=True needs nsamples"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), pin_nsamples=True)

    def test_must_be_a_bool(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        with pytest.raises(ValueError, match="pin_nsamples must be a bool"):
            Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), nsamples=2, pin_nsamples=1)

    def test_extend_keeps_it(self):
        from pytest_strategy import Parameter, RNGInteger, TestArg

        base = Parameter(TestArg("x", rng_type=RNGInteger(0, 1)), nsamples=2, pin_nsamples=True)
        assert base.extend(nsamples=5).pin_nsamples is True
        assert base.extend(pin_nsamples=False).pin_nsamples is False
        assert base.to_dict()["pin_nsamples"] is True


class TestPinNsamples:
    STRATEGY = """
        from pytest_strategy import Parameter, RNGInteger, TestArg, register

        @register("{name}")
        def factory():
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), {settings})
        """
    TEST = """
        from pytest_strategy import strategy

        @strategy("{name}")
        def test_x(x):
            pass
        """

    def _suite(self, pytester, name, settings):
        pytester.makepyfile(
            **{f"{name}_strategies": self.STRATEGY.format(name=name, settings=settings)}
        )
        pytester.makepyfile(**{f"test_{name}": self.TEST.format(name=name)})

    def test_a_pinned_count_ignores_the_option(self, pytester):
        self._suite(pytester, "pin_a", "nsamples=3, pin_nsamples=True")
        pytester.runpytest_inprocess("--nsamples=50").assert_outcomes(passed=3)
        pytester.runpytest_inprocess("--nsamples=auto").assert_outcomes(passed=3)

    def test_an_unpinned_count_follows_the_option(self, pytester):
        self._suite(pytester, "pin_b", "nsamples=3")
        pytester.runpytest_inprocess("--nsamples=5").assert_outcomes(passed=5)

    def test_pinned_auto_stays_auto_under_an_integer_option(self, pytester):
        pytester.makepyfile(pin_c_strategies="""
            from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

            @register("pin_c")
            def factory():
                return Parameter(
                    TestArg("ch", rng_type=Series([0, 1])),
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    nsamples="auto",
                    pin_nsamples=True,
                )
            """)
        pytester.makepyfile(test_pin_c="""
            from pytest_strategy import strategy

            @strategy("pin_c")
            def test_x(ch, x):
                pass
            """)
        result = pytester.runpytest_inprocess("--nsamples=50", "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*test_x?ch=0? PASSED*", "*test_x?ch=1? PASSED*"])
        result.stdout.fnmatch_lines(["*Parameter(nsamples=, pinned)*"])

    def test_vector_name_still_applies(self, pytester):
        self._suite(
            pytester, "pin_d", 'nsamples=3, pin_nsamples=True, directed_vectors={"z": (0,)}'
        )
        result = pytester.runpytest_inprocess("--vector-name=z")
        result.assert_outcomes(passed=1)


class TestDocumentedExportValues:
    """What the skill's reference says the export holds in its less common cases."""

    def test_pinned_auto_without_sequences_reads_default(self, pytester):
        pytester.makepyfile(doc_a_strategies="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("doc_a")
            def factory():
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 3)), nsamples="auto", pin_nsamples=True
                )

            @register("doc_b")
            def counted():
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 3)), nsamples=3)
            """)
        pytester.makepyfile(test_doc_a="""
            from pytest_strategy import strategy

            @strategy("doc_a")
            def test_pinned(x):
                pass

            @strategy("doc_b")
            def test_counted(x):
                pass
            """)
        result = pytester.runpytest_inprocess("--nsamples=5", "-v", "--strategy-export=out")
        result.assert_outcomes(passed=15)
        result.stdout.fnmatch_lines(
            ["*doc_a (*nsamples=10 from default, no Series/RNGSequence for auto"]
        )
        definitions = _by_test(_load(pytester.path / "out" / "strategies.json"))
        (pinned,) = definitions["test_doc_a.py::test_pinned"]
        assert pinned["nsamples"] == {
            "value": 10,
            "source": "default, no Series/RNGSequence for auto",
        }
        assert pinned["parameter"]["pin_nsamples"] is True

        result = pytester.runpytest_inprocess("--nsamples=auto", "--strategy-export=out")
        result.assert_outcomes(passed=13)
        definitions = _by_test(_load(pytester.path / "out" / "strategies.json"))
        (counted,) = definitions["test_doc_a.py::test_counted"]
        assert counted["nsamples"] == {
            "value": 3,
            "source": "Parameter(nsamples=), no Series/RNGSequence for auto",
        }

    def test_outcomes_of_failed_setup_and_teardown(self, pytester):
        pytester.makepyfile(doc_c_strategies="""
            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register

            @register("doc_c")
            def factory():
                return Parameter(
                    TestArg("y", rng_type=RNGInteger(0, 3)),
                    nsamples=1,
                    directed_vectors={"xf": pytest.param(1, marks=pytest.mark.xfail)},
                )
            """)
        pytester.makepyfile(test_doc_c="""
            import pytest
            from pytest_strategy import strategy

            @pytest.fixture
            def bad_setup():
                raise RuntimeError("setup")

            @pytest.fixture
            def bad_teardown():
                yield
                raise RuntimeError("teardown")

            @strategy("doc_c")
            def test_setup(y, bad_setup):
                pass

            @strategy("doc_c")
            def test_teardown(y, bad_teardown):
                pass

            @strategy("doc_c")
            def test_both(y, bad_teardown):
                assert False

            @pytest.mark.xfail(raises=ValueError)
            @strategy("doc_c")
            def test_raises(y, bad_setup):
                pass
            """)
        pytester.runpytest_inprocess("--strategy-export=out")
        vectors = _by_test(_load(pytester.path / "out" / "strategies.vectors.json"))

        def outcomes(test):
            (used,) = vectors[f"test_doc_c.py::{test}"]
            return {row["id"]: row["items"][0]["outcome"] for row in used["rows"]}

        assert outcomes("test_setup") == {"directed-xf": "xfailed", "rand-0": "error"}
        assert outcomes("test_teardown") == {"directed-xf": "xpassed", "rand-0": "error"}
        assert outcomes("test_both") == {"directed-xf": "xfailed", "rand-0": "failed"}
        assert outcomes("test_raises") == {"directed-xf": "error", "rand-0": "error"}


class TestIniValue:
    def test_a_toml_value_that_is_not_a_string_stops_the_run(self, pytester):
        pytester.makefile(".toml", pytest="[pytest]\nstrategies_export_dir = 1\n")
        result = pytester.runpytest("-p", "no:cacheprovider")
        if result.ret != pytest.ExitCode.USAGE_ERROR:
            pytest.skip("this pytest reads a TOML number as a string option")
        # pytest 9.2 rejects it itself, with its own message
        result.stderr.fnmatch_lines(["ERROR: *strategies_export_dir*"])
