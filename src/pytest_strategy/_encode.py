"""
The value encoding of the schema 1 documents (``VectorInfo.to_dict()``, and the
strategy export): every value becomes JSON that keeps its type.

- None, bool, int, str and finite floats are written as JSON;
- a float that is not finite is ``{"$float": "nan"}``, ``"inf"`` or ``"-inf"``;
- an Enum member is ``{"$enum": "Color", "member": "RED"}``;
- anything else is ``{"$repr": <stable repr>, "$type": <qualified type name>}``.

Only those exact types are written as JSON: a subclass of int or str that is not
an Enum (an ``Addr(int)``) is a ``$repr``, so a reader can rely on the JSON type
being the value's type. Nothing here is NaN or infinity, so the documents can be
written with ``json.dumps(..., allow_nan=False)``. Schema 1 does not promise that
a ``$repr`` value can be read back.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from ._ids import _value_repr

# The types written as plain JSON (exact types: a subclass is a "$repr")
_PLAIN = (bool, int, str)


def encode(value: Any) -> Any:
    """Return ``value`` in the schema 1 encoding (see the module docstring)."""
    if value is None or type(value) in _PLAIN:
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    # A Flag value that is not a member (no name) is shown by its repr
    if isinstance(value, Enum) and isinstance(value.name, str):
        return {"$enum": type(value).__qualname__, "member": value.name}
    return {"$repr": _value_repr(value), "$type": type(value).__qualname__}
