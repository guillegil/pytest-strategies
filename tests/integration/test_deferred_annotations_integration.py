"""A strategy test annotated with a TYPE_CHECKING-only name collects on Python 3.14+."""

import sys

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="annotations are evaluated at def time before 3.14"
)


def test_type_checking_only_annotations(pytester):
    pytester.makepyfile(test_deferred="""
        import dataclasses
        from typing import TYPE_CHECKING

        import pytest
        from pytest_strategy import Parameter, RNGInteger, Strategy, TestArg

        if TYPE_CHECKING:
            from decimal import Decimal

        @Strategy.register("deferred_ab")
        def _factory(nsamples: int) -> Decimal:
            return Parameter(
                TestArg("a", rng_type=RNGInteger(0, 9)),
                TestArg("b", rng_type=RNGInteger(0, 9)),
            )

        @pytest.fixture
        def helper():
            return 1

        @dataclasses.dataclass
        class AB:
            a: int
            b: Decimal

        @Strategy.strategy("deferred_ab")
        def test_named(a, b, helper: Decimal):
            assert helper == 1

        @Strategy.strategy("deferred_ab")
        def test_dataclass(p: AB, helper: Decimal):
            assert isinstance(p, AB)
        """)
    result = pytester.runpytest("--nsamples=3")
    result.assert_outcomes(passed=6)
