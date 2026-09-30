"""
The decorators keep the decorated function's type.

CI also type-checks this file with ``mypy --strict``: ``assert_type`` fails the
check if a decorator loses the type (a call on ``Callable[..., Any]`` returns
``Any``, not the declared type). At runtime it only runs the code.
"""

from typing import assert_type

from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy


@register("typing_ints")
def typing_ints(nsamples: int) -> Parameter:
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))


def unregistered() -> Parameter:
    return Parameter(TestArg("x", value=1))


def _labelled(x: int, label: str = "a") -> str:
    return f"{label}{x}"


def _same(x: int) -> int:
    return x


def test_register_keeps_the_factory_type() -> None:
    assert isinstance(assert_type(typing_ints(3), Parameter), Parameter)


def test_strategy_keeps_the_test_type() -> None:
    by_name = strategy("typing_ints")(_labelled)
    by_factory = strategy(unregistered, validate_signature=False)(_same)

    assert assert_type(by_name(1, label="b"), str) == "b1"
    assert assert_type(by_factory(2), int) == 2
