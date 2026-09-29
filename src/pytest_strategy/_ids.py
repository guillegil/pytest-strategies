"""
Test ID generation for parametrized strategies.
"""

import re
import reprlib
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

# The memory address in a default repr (``<Foo object at 0x7f...>``, a function, a
# bound method or a container of such objects) is directly followed by ``>``.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")


def _sorted_elements(values: set | frozenset) -> list:
    """Return set elements in a deterministic order: by value when they sort, else by repr."""
    # Sets are only partially ordered (by inclusion), so nested sets sort by repr
    if not any(isinstance(v, (set, frozenset)) for v in values):
        try:
            return sorted(values)
        except TypeError:
            pass
    return sorted(values, key=_stable_repr)


@reprlib.recursive_repr()
def _stable_repr(value: Any) -> str:
    """
    Return ``repr(value)``, with the elements of sets in a deterministic order.

    A set of strings reprs in hash order, which changes with ``PYTHONHASHSEED``
    (so between xdist workers). Sets nested in plain tuples, lists and dicts
    are ordered too; a container that contains itself is shown as ``...``.
    """
    if isinstance(value, (set, frozenset)) and value:
        body = "{" + ", ".join(_stable_repr(v) for v in _sorted_elements(value)) + "}"
        return body if type(value) is set else f"{type(value).__name__}({body})"
    if type(value) is tuple:
        items = [_stable_repr(v) for v in value]
        return f"({items[0]},)" if len(items) == 1 else "(" + ", ".join(items) + ")"
    if type(value) is list:
        return "[" + ", ".join(_stable_repr(v) for v in value) + "]"
    if type(value) is dict:
        pairs = (f"{_stable_repr(k)}: {_stable_repr(v)}" for k, v in value.items())
        return "{" + ", ".join(pairs) + "}"
    return repr(value)


def _value_repr(value: Any) -> str:
    """
    Return a repr of *value* that is stable across runs with the same seed.

    The default object repr (``<Foo object at 0x7f...>``) differs on every run,
    so a value whose repr embeds a memory address is shown by its type name.
    Strings and bytes always keep their repr, even when they contain text such as
    ``"fault at 0x10"``. Set elements are shown in a deterministic order.
    """
    val_str = _stable_repr(value)
    if not isinstance(value, (str, bytes)) and _ADDRESS.search(val_str):
        return type(value).__name__
    return val_str


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
            val_str = _value_repr(value)
            if len(val_str) > max_length - len(argnames[0]) - 1:
                val_str = val_str[: max_length - len(argnames[0]) - 4] + "..."
            ids.append(f"{argnames[0]}={val_str}")
        else:
            parts = []
            for arg_name, value in zip(argnames, sample):
                val_str = _value_repr(value)
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
            val_str = _value_repr(val)
            if len(val_str) > 20:
                val_str = val_str[:17] + "..."
            field_strs.append(f"{f.name}={val_str}")

        full_id = ",".join(field_strs)
        if len(full_id) > max_length:
            full_id = full_id[: max_length - 3] + "..."
        ids.append(full_id)

    return ids
