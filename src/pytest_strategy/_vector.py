"""
The rows of a strategy: tuples whose fields are the strategy's argument names.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Self, cast


class Vector(tuple[Any, ...]):
    """
    One row of a strategy: a tuple whose fields are the strategy's argument names.

    The rows that ``Parameter`` generates, and that its constraints receive, are
    instances of a class from :func:`vector_type`: a namedtuple over the argument
    names that also subclasses ``Vector``. So ``v.addr`` works, and ``v[0]``,
    unpacking and ``len(v)`` keep working. A Vector compares and hashes like the
    plain tuple of its values, whatever its field names, and its repr is
    ``Vector(addr=0, len=16)``.

    Fields named ``count`` or ``index`` shadow the tuple methods of those names;
    ``tuple.index(v, x)`` still reaches them. ``Vector`` itself is not
    instantiated: ``Parameter.vector_type`` is the class of a strategy's rows.
    """

    __slots__ = ()

    _fields: ClassVar[tuple[str, ...]] = ()

    if TYPE_CHECKING:
        # Any argument name type-checks (as Any), so a constraint annotated
        # ``-> bool`` returns ``bool(...)`` of an expression on the fields
        def __getattr__(self, name: str) -> Any: ...

        @classmethod
        def _make(cls, iterable: Iterable[Any]) -> Self: ...

        def _asdict(self) -> dict[str, Any]: ...

        def _replace(self, **values: Any) -> Self: ...

    else:

        def __getattr__(self, name: str) -> NoReturn:
            # Only reached when name is not a field (or another attribute) of the class
            if name.startswith("_"):
                raise AttributeError(f"'Vector' object has no attribute {name!r}")
            fields = ", ".join(type(self)._fields) or "none"
            raise AttributeError(f"Vector has no argument {name!r}; its arguments are {fields}")

    def __new__(cls, *_values: Any, **_fields: Any) -> Self:
        # The classes from vector_type() have their own __new__, the namedtuple's,
        # which takes the values by position or by field name
        raise TypeError(
            "Vector is the base class of the rows of every strategy and is not "
            "instantiated; Parameter.vector_type is the class of a Parameter's rows"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        # The classes are built at run time, so pickle cannot find them by name: a
        # Vector is rebuilt from its field names, also in a fresh process
        return (_rebuild, (type(self)._fields, tuple(self)))


# One class per tuple of argument names, shared by every Parameter with those names
_CLASSES: dict[tuple[str, ...], type[Vector]] = {}


def vector_type(names: Iterable[str]) -> type[Vector]:
    """
    Return the Vector class whose fields are ``names``, in that order.

    The class is built once per tuple of names and then reused, so two Parameters
    with the same argument names share it, and so does any prefix of a strategy's
    names (the arguments declared before a given one).

    Raises:
        ValueError: If a name is not a valid namedtuple field (not an identifier,
            a keyword, starting with ``_``, or repeated). ``Parameter`` checks the
            names first and says which argument is wrong.
    """
    key = tuple(names)
    cls = _CLASSES.get(key)
    if cls is None:
        fields = collections.namedtuple("Vector", key, module=__name__)  # type: ignore[misc]
        built = cast(
            "type[Vector]",
            type(
                "Vector",
                (fields, Vector),
                {"__slots__": (), "__module__": __name__, "__doc__": fields.__doc__},
            ),
        )
        # Another thread may have built the same class meanwhile: keep the first one
        cls = _CLASSES.setdefault(key, built)
    return cls


def _rebuild(names: tuple[str, ...], values: tuple[Any, ...]) -> Vector:
    """Rebuild a pickled Vector (see ``Vector.__reduce__``)."""
    return vector_type(names)._make(values)
