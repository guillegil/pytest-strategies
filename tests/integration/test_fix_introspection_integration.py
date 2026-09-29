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
