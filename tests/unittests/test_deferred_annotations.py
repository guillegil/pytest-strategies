"""
Python 3.14 evaluates annotations lazily (PEP 649/749).

A test or factory may then be annotated with a name that is not defined when
the decorator runs: a ``TYPE_CHECKING``-only import or a class defined later.
Plain pytest accepts such a test; introspection must not evaluate them either.
On older versions the ``def`` itself raises ``NameError``, so these tests only
run on 3.14+.
"""

import sys
from dataclasses import dataclass

import pytest

from pytest_strategy import Parameter, RNGInteger, TestArg
from pytest_strategy._dataclass import convert_to_dataclass
from pytest_strategy._introspection import detect_dataclass_param, validate_signature
from pytest_strategy._resolver import call_factory

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="annotations are evaluated at def time before 3.14"
)


@dataclass
class Point:
    x: int
    y: int


def test_validate_signature_ignores_undefined_fixture_annotation():
    def test_fn(x, y, helper: NotDefinedYet):  # noqa: F821
        pass

    validate_signature(test_fn, ["x", "y"], "s")


def test_dataclass_detected_next_to_undefined_fixture_annotation():
    def test_fn(p: Point, helper: NotDefinedYet):  # noqa: F821
        pass

    assert detect_dataclass_param(test_fn, ["x", "y"]) == (True, Point, "p")


def test_undefined_dataclass_annotation_is_not_dataclass_mode():
    def test_fn(p: LaterDataclass):  # noqa: F821
        pass

    assert detect_dataclass_param(test_fn, ["x", "y"]) == (False, None, None)


def test_factory_with_undefined_return_annotation_is_called():
    def factory(nsamples: int) -> NotDefinedYet:  # noqa: F821
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 1)))

    assert isinstance(call_factory("s", factory, 3), Parameter)


def test_dataclass_with_undefined_field_annotation_is_converted():
    @dataclass
    class Later:
        x: int
        y: NotDefinedYet  # noqa: F821

    assert convert_to_dataclass([(1, 2)], ["x", "y"], Later) == [Later(1, 2)]
