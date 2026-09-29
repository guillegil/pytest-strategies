"""
Dataclass conversion utilities for strategy samples.
"""

from collections.abc import Sequence
from dataclasses import fields


def convert_to_dataclass(
    samples: Sequence[tuple],
    argnames: Sequence[str],
    dataclass_type: type,
) -> list:
    """
    Convert a sequence of tuple samples to dataclass instances.

    The function validates that the dataclass ``__init__`` fields (fields with
    ``init=False`` are skipped) exactly match the strategy *argnames*, then
    builds each instance with keyword arguments, so argname order and
    keyword-only fields do not matter.

    Args:
        samples: Sequence of value tuples from the strategy.
        argnames: Argument names from the strategy (must match dataclass fields).
        dataclass_type: The dataclass type to instantiate.

    Returns:
        List of dataclass instances.

    Raises:
        ValueError: When the dataclass fields do not match *argnames*.
    """
    dc_fields = {f.name for f in fields(dataclass_type) if f.init}
    strategy_fields = set(argnames)

    if dc_fields != strategy_fields:
        missing = strategy_fields - dc_fields
        extra = dc_fields - strategy_fields

        error_msg = "Dataclass fields don't match strategy parameters!\n"
        error_msg += f"  Strategy provides: {list(argnames)}\n"
        error_msg += f"  Dataclass expects: {list(dc_fields)}\n"

        if missing:
            error_msg += f"  Missing in dataclass: {list(missing)}\n"
        if extra:
            error_msg += f"  Extra in dataclass: {list(extra)}\n"

        raise ValueError(error_msg)

    return [dataclass_type(**dict(zip(argnames, sample))) for sample in samples]
