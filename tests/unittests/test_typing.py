"""
The decorators keep the decorated function's type, the keyword-only options are
keyword-only for type checkers too, StrategyOptions is typed as frozen, a
Vector's fields type-check by name, vectors given by name type-check next to
tuples while the vector mappings are read-only, and constraints type-check by
name too, and can be turned off per generation call.

CI also type-checks this file with ``mypy --strict``: ``assert_type`` fails the
check if a decorator loses the type (a call on ``Callable[..., Any]`` returns
``Any``, not the declared type), and ``--strict`` includes
``--warn-unused-ignores``, so a ``type: ignore[call-arg]`` on a call mypy accepts
fails it too. At runtime it only runs the code.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal, assert_type

import pytest

from pytest_strategy import (
    Parameter,
    RNGInteger,
    Series,
    StrategyOptions,
    TestArg,
    Vector,
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


def aligned(v: Vector) -> bool:
    return bool(v.addr % 4 == 0)


def first_small(v: tuple[int, int]) -> bool:
    return v[0] < 64


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
    with pytest.raises(TypeError):
        param.add_constraint(aligned, "n")  # type: ignore[call-arg]


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


def test_vector_fields_type_check() -> None:
    """A constraint reads the row's fields by name, and the generators return Vectors."""
    param = Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        vector_constraints=[aligned, first_small, lambda v: v.len > 0],
    )
    assert assert_type(param.add_constraint(lambda v: bool(v.len < 64), name="short"), str)

    row = assert_type(param.generate_vector(), Vector)
    assert_type(param.vector_type, type[Vector])
    assert assert_type(row._asdict(), dict[str, Any])["addr"] == row.addr
    assert assert_type(row._replace(len=1), Vector).len == 1
    assert assert_type(param.vector_type(addr=4, len=1), Vector) == (4, 1)
    assert aligned(row) and first_small((row.addr, row.len))


def test_rows_unpack_and_read_fields_as_in_3_0() -> None:
    """
    Every row is typed as a Vector, also where a pytest.param(...) vector could come
    back, so code typed for 3.0's tuple rows unpacks them, and fields type-check.
    """
    param = Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        directed_vectors={"zeros": (0, 1)},
        test_vectors={"max": {"addr": 63, "len": 16}},
    )

    total = 0
    for addr, length in param.generate_vectors(3):
        total += int(addr) + int(length)
    rows = assert_type(param.generate_vectors(3, mode="random_only"), list[Vector])
    addrs: list[int] = [int(v.addr) for v in rows]
    a, b = param.get_directed_vector("zeros")
    c, d = param.get_vector_by_name("zeros")
    e, f = param.get_vector_by_index(0)
    g, h = param.get_test_vector("max")
    first = param.directed_vectors["zeros"]
    assert_type(first, Vector)
    assert (a, b) == (c, d) == (e, f) == tuple(first) == (0, 1) and (g, h) == (63, 16)
    assert total > 0 and len(addrs) == 3 and int(first[0]) + int(first.len) == 1


def test_constraint_lists_typed_for_tuples_still_type_check() -> None:
    """Constraints written for 3.0's tuple rows are accepted, also in a typed list."""
    constraints: list[Callable[[tuple[Any, ...]], bool]] = [lambda v: v[0] >= 0]
    param = Parameter(TestArg("addr", rng_type=RNGInteger(0, 9)), vector_constraints=constraints)

    assert len(param.generate_vectors(2, mode="random_only")) == 2


def test_named_constraints_type_check() -> None:
    """Each ignore below is needed: mypy reports the line as an error, as Python does."""
    constraints: dict[str, Callable[[Vector], bool]] = {"aligned": aligned}
    param = Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        vector_constraints={"aligned": aligned, "short": lambda v: v.len < 16},
    )
    typed = Parameter(TestArg("addr", rng_type=RNGInteger(0, 63)), vector_constraints=constraints)

    names = assert_type(param.vector_constraints, Mapping[str, Callable[[Vector], object]])
    assert list(names) == ["aligned", "short"] and list(typed.vector_constraints) == ["aligned"]
    assert assert_type(param.add_constraint(first_small, name="first"), str) == "first"
    param.remove_constraint("first")
    with pytest.raises(TypeError):
        param.vector_constraints["x"] = aligned  # type: ignore[index]


def test_constraints_off_type_checks() -> None:
    """The generators take the names to turn off as any collection of str, by keyword."""
    param = Parameter(
        TestArg("ch", rng_type=Series([0, 4])),
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        vector_constraints={"aligned": aligned, "never": lambda v: False},
    )
    options = StrategyOptions(strategy="s", constraints_off=frozenset({"never"}))

    rows = param.generate_vectors(2, mode="random_only", constraints_off=options.constraints_off)
    assert len(rows) == 2
    assert len(assert_type(param.generate_exhaustive(constraints_off=["never"]), list[Vector])) == 2


def test_vectors_by_name_type_check() -> None:
    """Each ignore below is needed: mypy reports the line as an error, as Python does."""
    vectors = {"zeros": (0, 0), "max": {"len": 16, "addr": 63}, "list": [1, 2]}
    param = Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        directed_vectors=vectors,
        test_vectors={"marked": pytest.param({"addr": 4, "len": 1}, marks=pytest.mark.xfail)},
    )
    param.add_directed_vector("one", {"addr": 1, "len": 1})
    param.add_test_vector("row", param.vector_type(addr=2, len=2))

    vector = param.get_directed_vector("max")
    assert isinstance(vector, Vector) and vector.len == 16
    assert param.get_test_vector("marked").values == (4, 1)
    with pytest.raises(TypeError):
        param.directed_vectors["x"] = (1, 1)  # type: ignore[index]
    with pytest.raises(AttributeError):
        param.test_vectors = {}  # type: ignore[misc]
