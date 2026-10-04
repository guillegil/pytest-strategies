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

from pytest_strategy import RNG, Parameter, RNGInteger, StrategyOptions, TestArg
from pytest_strategy._factory import FactoryInputs, call_factory
from pytest_strategy._introspection import validate_signature
from pytest_strategy._records import convert_to_dataclass, detect_record_param, record_hints

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="annotations are evaluated at def time before 3.14"
)


@dataclass
class Point:
    x: int
    y: int


def _record(test_fn, argnames):
    """The (parameter, record type) of record mode, or None in named mode."""
    record = detect_record_param(test_fn, argnames, None)
    return None if record is None else (record.name, record.record_type)


def test_validate_signature_ignores_undefined_fixture_annotation():
    def test_fn(x, y, helper: NotDefinedYet):  # noqa: F821
        pass

    validate_signature(test_fn, ["x", "y"], "s")


def test_dataclass_detected_next_to_undefined_fixture_annotation():
    def test_fn(p: Point, helper: NotDefinedYet):  # noqa: F821
        pass

    assert _record(test_fn, ["x", "y"]) == ("p", Point)


def test_undefined_dataclass_annotation_is_not_dataclass_mode():
    def test_fn(p: LaterDataclass):  # noqa: F821
        pass

    assert _record(test_fn, ["x", "y"]) is None


def test_undefined_annotation_is_named_in_the_signature_hint():
    def test_fn(p: LaterDataclass):  # noqa: F821
        pass

    assert record_hints(test_fn, ["x", "y"], None) == [
        "Parameter 'p' is annotated with 'LaterDataclass', which cannot be resolved in the "
        "test module's globals, so it is not a record type: define the class at module level."
    ]


def test_factory_with_undefined_return_annotation_is_called():
    def factory(nsamples: int) -> NotDefinedYet:  # noqa: F821
        return Parameter(TestArg("x", rng_type=RNGInteger(0, 1)))

    inputs = FactoryInputs(
        options=StrategyOptions(strategy="s", nsamples=3), rng=RNG.generator(), ctx=lambda: None
    )
    assert isinstance(call_factory("s", factory, inputs), Parameter)


def test_dataclass_with_undefined_field_annotation_is_converted():
    @dataclass
    class Later:
        x: int
        y: NotDefinedYet  # noqa: F821

    assert convert_to_dataclass([(1, 2)], ["x", "y"], Later) == [Later(1, 2)]
