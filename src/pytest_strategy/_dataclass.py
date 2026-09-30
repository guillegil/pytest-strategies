"""
Dataclass conversion utilities for strategy samples.
"""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

from ._introspection import lazy_signature


def convert_to_dataclass(
    samples: Sequence[tuple[Any, ...]],
    argnames: Sequence[str],
    dataclass_type: type,
) -> list[Any]:
    """
    Convert a sequence of tuple samples to dataclass instances.

    The function validates that the dataclass ``__init__`` fields (fields with
    ``init=False`` are skipped) exactly match the strategy *argnames*, then
    builds each instance with keyword arguments, so argname order and
    keyword-only fields do not matter. A hand-written ``__init__`` that does not
    take the field names as keywords (other names, or positional-only) gets the
    values positionally, in field declaration order.

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

    try:
        lazy_signature(dataclass_type).bind(**dict.fromkeys(argnames))
    except TypeError:
        # Hand-written __init__ with other or positional-only parameter names
        order = [f.name for f in fields(dataclass_type) if f.init]
        return [
            dataclass_type(*(values[name] for name in order))
            for values in (dict(zip(argnames, sample)) for sample in samples)
        ]
    except ValueError:
        # No inspectable signature: keep the keyword construction
        pass

    return [dataclass_type(**dict(zip(argnames, sample))) for sample in samples]
