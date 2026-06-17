"""
Test ID generation for parametrized strategies.
"""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any


def generate_test_ids(
    argnames: Sequence[str],
    samples: Sequence[Any],
    max_length: int = 80,
) -> list[str]:
    """
    Generate concise test IDs from argument names and sample values.

    Args:
        argnames: Sequence of argument names.
        samples: Sequence of sample values (tuples or single values).
        max_length: Maximum character length per ID (default 80).

    Returns:
        List of test ID strings, one per sample.
    """
    ids: list[str] = []

    for sample in samples:
        if len(argnames) == 1:
            value = sample if not isinstance(sample, tuple) else sample[0]
            val_str = repr(value)
            if len(val_str) > max_length - len(argnames[0]) - 1:
                val_str = val_str[: max_length - len(argnames[0]) - 4] + "..."
            ids.append(f"{argnames[0]}={val_str}")
        else:
            parts = []
            for arg_name, value in zip(argnames, sample):
                val_str = repr(value)
                if len(val_str) > 20:
                    val_str = val_str[:17] + "..."
                parts.append(f"{arg_name}={val_str}")

            full_id = ",".join(parts)
            if len(full_id) > max_length:
                full_id = full_id[: max_length - 3] + "..."
            ids.append(full_id)

    return ids


def generate_dataclass_ids(
    dataclass_samples: list,
    dc_type: type,
    max_length: int = 80,
) -> list[str]:
    """
    Generate test IDs for dataclass-mode parametrization.

    Args:
        dataclass_samples: List of dataclass instances.
        dc_type: The dataclass type (used to enumerate fields in order).
        max_length: Maximum character length per ID (default 80).

    Returns:
        List of test ID strings, one per dataclass instance.
    """
    ids: list[str] = []
    for dc_instance in dataclass_samples:
        field_strs = []
        for f in fields(dc_type):
            val = getattr(dc_instance, f.name)
            val_str = repr(val)
            if len(val_str) > 20:
                val_str = val_str[:17] + "..."
            field_strs.append(f"{f.name}={val_str}")

        full_id = ",".join(field_strs)
        if len(full_id) > max_length:
            full_id = full_id[: max_length - 3] + "..."
        ids.append(full_id)

    return ids
