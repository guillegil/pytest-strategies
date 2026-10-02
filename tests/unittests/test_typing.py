"""
The decorators keep the decorated function's type, the keyword-only options are
keyword-only for type checkers too, and StrategyOptions is typed as frozen.

CI also type-checks this file with ``mypy --strict``: ``assert_type`` fails the
check if a decorator loses the type (a call on ``Callable[..., Any]`` returns
``Any``, not the declared type), and ``--strict`` includes
``--warn-unused-ignores``, so a ``type: ignore[call-arg]`` on a call mypy accepts
fails it too. At runtime it only runs the code.
"""

import dataclasses
from typing import Literal, assert_type

import pytest

from pytest_strategy import (
    Parameter,
    RNGInteger,
    StrategyOptions,
    TestArg,
    export_strategies,
    register,
    strategy,
)


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


def test_mypy_rejects_positional_options() -> None:
    """Each ignore below is needed: mypy reports the call as an error, as Python does."""
    param = Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))

    with pytest.raises(TypeError):
        TestArg("x", RNGInteger(0, 1), 5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        strategy("typing_ints", False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        param.generate_vectors(1, "all")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        export_strategies("json")  # type: ignore[call-arg]


def test_mypy_accepts_the_keyword_forms() -> None:
    arg = TestArg("x", RNGInteger(0, 9), validator=lambda v: v >= 0, description="d")
    param = Parameter(arg)

    assert len(param.generate_vectors(2, mode="random_only")) == 2


def test_strategy_options_types() -> None:
    """Each ignore below is needed: mypy reports the line as an error, as Python does."""
    options = StrategyOptions(strategy="s", nsamples="auto", mode="test")

    assert_type(options.nsamples, int | Literal["auto"])
    assert_type(options.mode, Literal["all", "random_only", "directed_only", "mixed", "test"])
    assert assert_type(options.filtered, bool) is False
    with pytest.raises(TypeError):
        StrategyOptions("s")  # type: ignore[call-arg]
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.mode = "all"  # type: ignore[misc]
