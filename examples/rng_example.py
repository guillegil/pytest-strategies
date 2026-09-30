"""
The RNG helpers and RNG types on their own, outside a strategy.

Run it as a script to print a few values (``python examples/rng_example.py``), or
with pytest to check them. Values come from the plugin's own generator, so
``RNG.seed(42)`` (or ``--rng-seed=42``) reproduces them without touching the
global ``random`` state.
"""

from enum import Enum

from pytest_strategy import (
    RNG,
    RNGChoice,
    RNGEnum,
    RNGInteger,
    RNGWeightedInteger,
)


class Status(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def test_basic_generators():
    assert 1 <= RNG.integer(1, 100) <= 100
    assert RNG.integer(1, 100, predicate=lambda x: x % 2 == 0) % 2 == 0
    assert 0.0 <= RNG.float(0.0, 10.0) <= 10.0
    assert isinstance(RNG.boolean(), bool)
    assert RNG.choice(["apple", "banana", "cherry"]) in ("apple", "banana", "cherry")
    assert len(RNG.string(length=10)) == 10


def test_weighted_generators():
    assert 0 <= RNG.winteger({(0, 20): 0.8, (21, 100): 0.2}) <= 100
    assert 0.0 <= RNG.wfloat({(0.0, 1.0): 0.7, (1.0, 10.0): 0.3}) <= 10.0


def test_rng_types():
    assert 1 <= RNGInteger(min=1, max=100).generate() <= 100
    assert 0 <= RNGWeightedInteger(ranges={(0, 20): 0.8, (21, 100): 0.2}).generate() <= 100
    assert RNGChoice(choices=["fast", "slow", "medium"]).generate() in ("fast", "slow", "medium")


def test_rng_enum():
    assert isinstance(RNGEnum(Status).generate(), Status)
    weighted = RNGEnum(
        Status, weights={Status.SUCCESS: 0.7, Status.PENDING: 0.2, Status.FAILED: 0.1}
    )
    assert weighted.generate() in (Status.SUCCESS, Status.PENDING, Status.FAILED)
    assert RNGEnum(Status, predicate=lambda s: s != Status.ERROR).generate() != Status.ERROR
    priority = RNGEnum(
        Priority,
        weights={Priority.HIGH: 0.6, Priority.MEDIUM: 0.3, Priority.LOW: 0.1},
        predicate=lambda p: p != Priority.CRITICAL,
    )
    assert priority.generate() != Priority.CRITICAL


def test_same_seed_same_values():
    RNG.seed(42)
    first = [RNG.integer(0, 1000) for _ in range(5)]
    RNG.seed(42)
    assert [RNG.integer(0, 1000) for _ in range(5)] == first


if __name__ == "__main__":
    RNG.seed(42)

    print("=== Basic Generators ===")
    print(f"Integer (1-100): {RNG.integer(1, 100)}")
    print(f"Even integer (1-100): {RNG.integer(1, 100, predicate=lambda x: x % 2 == 0)}")
    print(f"Float (0-10): {RNG.float(0.0, 10.0)}")
    print(f"Boolean: {RNG.boolean()}")
    print(f"Choice: {RNG.choice(['apple', 'banana', 'cherry'])}")
    print(f"String (length 10): {RNG.string(length=10)}")

    print("\n=== Weighted Generators ===")
    print(
        f"Weighted integer (80% 0-20, 20% 21-100): {RNG.winteger({(0, 20): 0.8, (21, 100): 0.2})}"
    )
    print(f"Weighted float (70% 0-1, 30% 1-10): {RNG.wfloat({(0.0, 1.0): 0.7, (1.0, 10.0): 0.3})}")

    print("\n=== RNG Types ===")
    print(f"RNGInteger: {RNGInteger(min=1, max=100).generate()}")
    print(
        f"RNGWeightedInteger: {RNGWeightedInteger(ranges={(0, 20): 0.8, (21, 100): 0.2}).generate()}"
    )
    print(f"RNGChoice: {RNGChoice(choices=['fast', 'slow', 'medium']).generate()}")

    print("\n=== RNGEnum ===")
    print(f"Uniform: {RNGEnum(Status).generate()}")
    print(f"Filtered: {RNGEnum(Status, predicate=lambda s: s != Status.ERROR).generate()}")
