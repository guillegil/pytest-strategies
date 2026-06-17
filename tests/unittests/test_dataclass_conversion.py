"""
Unit tests for _dataclass module.

Tests convert_to_dataclass as a pure function.
"""

from dataclasses import dataclass

import pytest

from pytest_strategy._dataclass import convert_to_dataclass


@dataclass
class Point:
    x: int
    y: int


@dataclass
class RGB:
    r: int
    g: int
    b: int


@dataclass
class Reversed:
    """Fields in reversed order compared to strategy argnames."""

    b: int
    a: int


class TestConvertToDataclass:
    def test_basic_conversion(self):
        samples = [(1, 2), (3, 4)]
        result = convert_to_dataclass(samples, ["x", "y"], Point)
        assert result[0] == Point(x=1, y=2)
        assert result[1] == Point(x=3, y=4)

    def test_three_field_dataclass(self):
        samples = [(255, 128, 0)]
        result = convert_to_dataclass(samples, ["r", "g", "b"], RGB)
        assert result[0] == RGB(r=255, g=128, b=0)

    def test_field_order_reordering(self):
        """
        When strategy argnames are in a different order than dataclass fields,
        values should be reordered to match field declaration order.
        """
        # Strategy provides (a, b) but dataclass declares (b, a)
        samples = [(10, 20)]  # a=10, b=20 in strategy order
        result = convert_to_dataclass(samples, ["a", "b"], Reversed)
        # Reversed(b=20, a=10)
        assert result[0] == Reversed(b=20, a=10)

    def test_missing_field_raises(self):
        """Strategy provides extra field not in dataclass."""
        with pytest.raises(ValueError, match="don't match"):
            convert_to_dataclass([(1, 2, 3)], ["x", "y", "z"], Point)

    def test_extra_field_raises(self):
        """Dataclass has field not provided by strategy."""
        with pytest.raises(ValueError, match="don't match"):
            convert_to_dataclass([(1,)], ["x"], Point)

    def test_error_message_includes_missing(self):
        with pytest.raises(ValueError, match="Missing in dataclass"):
            convert_to_dataclass([(1, 2, 3)], ["x", "y", "z"], Point)

    def test_error_message_includes_extra(self):
        with pytest.raises(ValueError, match="Extra in dataclass"):
            convert_to_dataclass([(1,)], ["x"], Point)

    def test_empty_samples(self):
        result = convert_to_dataclass([], ["x", "y"], Point)
        assert result == []

    def test_single_sample(self):
        result = convert_to_dataclass([(7, 8)], ["x", "y"], Point)
        assert len(result) == 1
        assert result[0].x == 7
        assert result[0].y == 8

    def test_returns_list(self):
        result = convert_to_dataclass([(1, 2)], ["x", "y"], Point)
        assert isinstance(result, list)
