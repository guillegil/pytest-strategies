"""
The rows of a strategy: tuples whose fields are the strategy's argument names, and
the metadata of each row's test (``VectorInfo``, stored under ``VECTOR_KEY``).
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Self, cast

import pytest

from ._encode import encode

# The kinds of rows a strategy generates
RowKind = Literal["directed", "test", "random", "exhaustive", "skipped"]


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


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorInfo:
    """
    The strategy row a test item runs: ``item.stash[VECTOR_KEY]``.

    The plugin fills it for every row of a strategy while the tests are collected,
    before any ``pytest_collection_modifyitems`` hook runs, so hooks, fixtures and
    reports can read it. An item without a strategy, and the item pytest makes for
    an empty parameter set, have none: ``item.stash.get(VECTOR_KEY, None)`` is None.
    A test with several ``@strategy`` decorators has one per strategy, in
    ``item.stash[VECTORS_KEY]``, in the order of the node ID.

    It is frozen and read, not built: later releases may add fields, always
    keyword-only and with defaults.

    Attributes:
        strategy: The strategy's resolved name, as in the ``-v`` summary
        origin: Where the factory is defined, as ``"tests/dma/strategies.py:12"``
            (relative to the rootdir, in posix form), or None when unknown
        kind: "directed", "test", "random", "exhaustive", or "skipped" for the one
            row of a strategy whose skip_if_empty sequence has no values
        name: The directed or test vector's name, else None
        index: The vector's position in ``directed_vectors`` (the number
            ``--vector-index`` takes) or ``test_vectors``; the row's number within
            its combination for a random row; the row's position in the product of
            the enumerated arguments' sequences, in declaration order, for an
            exhaustive row; None for "skipped"
        enumerated: The arguments the row enumerates (Series arguments, and with
            ``--nsamples=auto`` or ``per_sequence_samples=True`` RNGSequence
            arguments), in declaration order: in the names format, the arguments
            whose values are in the ID
        values: The row, by argument name (Nones for "skipped"); a record-mode
            test receives a record built from it
        id: This strategy's part of the test ID, before pytest escapes it
        seed: The run's seed (``--rng-seed``)
        context: The fingerprint of the context the factory received, or None
        constraints_off: The constraints turned off in this strategy, in order
        streams: The version of the random streams the values come from
    """

    strategy: str
    origin: str | None
    kind: RowKind
    name: str | None
    index: int | None
    enumerated: tuple[str, ...]
    values: Vector
    id: str
    seed: int
    context: str | None
    constraints_off: tuple[str, ...]
    streams: int = 1

    def to_dict(self) -> dict[str, Any]:
        """
        Return the row as a JSON-ready dict with ``"schema": 1``, its values by
        argument name in the schema's value encoding (``{"$float": "nan"}``,
        ``{"$enum": "Color", "member": "RED"}``, ``{"$repr": ..., "$type": ...}``).
        """
        return {
            "schema": 1,
            "strategy": self.strategy,
            "origin": self.origin,
            "kind": self.kind,
            "name": self.name,
            "index": self.index,
            "enumerated": list(self.enumerated),
            "values": {
                name: encode(value)
                for name, value in zip(type(self.values)._fields, self.values, strict=True)
            },
            "id": self.id,
            "seed": self.seed,
            "context": self.context,
            "constraints_off": list(self.constraints_off),
            "streams": self.streams,
        }


# The VectorInfo of an item's first strategy in the node ID, and of each of them
VECTOR_KEY = pytest.StashKey[VectorInfo]()
VECTORS_KEY = pytest.StashKey[tuple[VectorInfo, ...]]()
