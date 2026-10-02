"""
End-to-end regression tests for the introspection, dataclass and ID fixes.

These tests use pytester to run isolated in-process pytest sessions. Distinct
strategy filenames are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

pytest_plugins = ["pytester"]

POINT_STRATEGY = """
    from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger

    @Strategy.register("{name}")
    def factory(nsamples):
        return Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", rng_type=RNGInteger(0, 10)),
            nsamples=3,
        )
    """


class TestDataclassModeDetection:
    """Dataclass mode must work in classes, with custom fixtures and PEP 563."""

    def test_method_and_custom_fixtures(self, pytester):
        pytester.makepyfile(fi_dc_a_strategies=POINT_STRATEGY.format(name="fi_dc_a"))
        pytester.makeconftest("""
            import pytest

            @pytest.fixture
            def db():
                return "db"
            """)
        pytester.makepyfile(test_fi_dc_a="""
            from dataclasses import dataclass

            from pytest_strategy import Strategy

            @dataclass
            class Point:
                x: int
                y: int

            class TestInClass:
                @Strategy.strategy("fi_dc_a")
                def test_method(self, p: Point):
                    assert isinstance(p, Point)

            @Strategy.strategy("fi_dc_a")
            def test_fixture_after(p: Point, db):
                assert isinstance(p, Point) and db == "db"

            @Strategy.strategy("fi_dc_a")
            def test_fixture_before(db, p: Point):
                assert isinstance(p, Point) and db == "db"
            """)
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=9)

    def test_postponed_annotations(self, pytester):
        pytester.makepyfile(fi_dc_b_strategies=POINT_STRATEGY.format(name="fi_dc_b"))
        pytester.makepyfile(test_fi_dc_b="""
            from __future__ import annotations

            from dataclasses import dataclass

            from pytest_strategy import Strategy

            @dataclass
            class Point:
                x: int
                y: int

            @Strategy.strategy("fi_dc_b")
            def test_future(p: Point, tmp_path):
                assert isinstance(p, Point)
            """)
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=3)


class TestDataclassConversion:
    """Keyword-only and init=False dataclasses must be built end to end."""

    def test_kw_only_and_init_false_dataclasses(self, pytester):
        pytester.makepyfile(fi_dc_c_strategies=POINT_STRATEGY.format(name="fi_dc_c"))
        pytester.makepyfile(test_fi_dc_c="""
            from dataclasses import dataclass, field

            from pytest_strategy import Strategy

            @dataclass(kw_only=True)
            class KwPoint:
                x: int
                y: int

            @dataclass
            class Computed:
                x: int
                y: int
                total: int = field(init=False)

                def __post_init__(self):
                    self.total = self.x + self.y

            @Strategy.strategy("fi_dc_c")
            def test_kw_only(p: KwPoint):
                assert isinstance(p, KwPoint)

            @Strategy.strategy("fi_dc_c")
            def test_init_false(p: Computed):
                assert p.total == p.x + p.y
            """)
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=6)


class TestStableIds:
    """Node IDs must not embed memory addresses, so reruns by node ID work."""

    def test_default_repr_values_give_stable_node_ids(self, pytester):
        pytester.makepyfile(fi_ids_a_strategies="""
            from pytest_strategy import Parameter, RNGChoice, Strategy, TestArg

            class Codec:
                pass

            CODECS = [Codec(), Codec()]

            @Strategy.register("fi_ids_a")
            def factory(nsamples):
                return Parameter(
                    TestArg("codec", rng_type=RNGChoice(CODECS)),
                    directed_vectors={"first": (CODECS[0],), "second": (CODECS[1],)},
                    nsamples=0,
                )
            """)
        pytester.makepyfile(test_fi_ids_a="""
            from pytest_strategy import Strategy

            @Strategy.strategy("fi_ids_a")
            def test_codec(codec):
                pass
            """)
        result = pytester.runpytest_inprocess("--collect-only", "-q")
        node_ids = [line for line in result.outlines if "::" in line]
        assert node_ids == [
            "test_fi_ids_a.py::test_codec[codec=Codec0]",
            "test_fi_ids_a.py::test_codec[codec=Codec1]",
        ]


class TestArgnameShadowingBuiltinFixture:
    """A strategy argname named like a built-in fixture must pass validation."""

    def test_argname_named_cache(self, pytester):
        pytester.makepyfile(fi_sig_a_strategies="""
            from pytest_strategy import Strategy, Parameter, TestArg, RNGBoolean, RNGInteger

            @Strategy.register("fi_sig_a")
            def factory(nsamples):
                return Parameter(
                    TestArg("cache", rng_type=RNGBoolean()),
                    TestArg("size", rng_type=RNGInteger(1, 9)),
                    nsamples=3,
                )
            """)
        pytester.makepyfile(test_fi_sig_a="""
            from pytest_strategy import Strategy

            @Strategy.strategy("fi_sig_a")
            def test_lookup(cache, size):
                assert isinstance(cache, bool)
                assert 1 <= size <= 9
            """)
        result = pytester.runpytest_inprocess()
        result.assert_outcomes(passed=3)
