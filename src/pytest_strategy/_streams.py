"""
Random streams v1 (D5): the keys that seed the plugin's random streams.

A stream key is a path of parts, each an int or a str, under a seed:
``StreamKey.root(seed, "test", strategy, nodeid)`` keys what one strategy draws for
one test, and ``.child(...)`` keys a part of that, such as one argument of one row.
``seed_int()`` hashes the path into a 128-bit int, which seeds a
``random.Random``.

The derivation is streams version 1 (``VERSION``), the version that
``VectorInfo.streams`` reports. A key's int, and so every value drawn from its
stream, must not change within a major release: the goldens in
tests/unittests/test_streams.py pin it.

- Each part is encoded as one type byte, ``i`` for an int or ``s`` for a str, the
  length of its payload in 8 bytes (unsigned, big-endian), and the payload: the
  int in two's complement, big-endian, in ``(n.bit_length() + 8) // 8`` bytes, or
  the str in UTF-8 (a lone surrogate with ``surrogatepass``). An int or str
  subclass is encoded by its int or str value. bools, floats and anything else
  are a TypeError.
- The path is the seed's encoding followed by each part's, so
  ``root(s, a).child(b)`` is ``root(s, a, b)``. The encoding is typed and
  length-prefixed, so ``("a:b", "c")`` and ``("a", "b:c")`` are different paths,
  and so are ``"3"`` and ``3``; the 3.x key ``f"{seed}:{key}"`` gave such pairs
  one stream.
- ``seed_int()`` is ``blake2b(path, digest_size=16, person=b"pst-stream/1")``,
  read as an unsigned big-endian int. BLAKE2b is built into CPython (no OpenSSL,
  so FIPS builds have it), and nothing here uses ``hash()``, so a key's int is the
  same in every process, whatever ``PYTHONHASHSEED`` or the OS.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePath
from typing import Any

# The version of the derivation: the streams version of the generated values
VERSION = 1

# BLAKE2b's personalization string: the same bytes give other digests in other uses
_PERSON = f"pst-stream/{VERSION}".encode("ascii")

# Bytes in the length of a part's payload
_LENGTH_SIZE = 8

# A BLAKE2b hasher with nothing fed yet, copied for each key (cheaper than a new one)
_HASHER = hashlib.blake2b(digest_size=16, person=_PERSON)

# The folders installed packages live in, wherever the environment is
INSTALLED_FOLDERS = frozenset({"site-packages", "dist-packages"})


def _part(value: Any) -> bytes:
    """
    Encode one part of a stream key: its type byte (``i`` or ``s``), the length of
    its payload in 8 bytes, and the payload (see the module docstring).

    Raises:
        TypeError: If the part is not an int or a str. A bool is refused, though it
            is an int: ``True`` would key the stream of ``1``.
    """
    if isinstance(value, str):
        tag = b"s"
        # str.encode() and int.to_bytes() read the value itself, not a subclass's override
        payload = str.encode(value, "utf-8", "surrogatepass")
    elif isinstance(value, int) and not isinstance(value, bool):
        tag = b"i"
        payload = int.to_bytes(value, (int.bit_length(value) + 8) // 8, "big", signed=True)
    else:
        raise TypeError(
            f"A stream key part must be an int or a str, got {value!r} ({type(value).__name__})"
        )
    return tag + len(payload).to_bytes(_LENGTH_SIZE, "big") + payload


def encode(*parts: int | str) -> bytes:
    """
    Return the encoding of ``parts`` in a key's path, for :meth:`StreamKey.child_seed_int`.

    Raises:
        TypeError: If a part is not an int or a str (a bool or a float included)
    """
    return b"".join([_part(p) for p in parts])


def seed_part(seed: object) -> int | str:
    """
    Return a seed as a key's first part: an int or a str as it is, and any other
    seed ``random`` accepts (a bool, a float or bytes, given to ``RNG.seed()``) as
    its repr, so that building a key never fails on the seed.
    """
    if isinstance(seed, str) or (isinstance(seed, int) and not isinstance(seed, bool)):
        return seed
    return repr(seed)


def path_part(path: str | os.PathLike[str], rootpath: str | os.PathLike[str] | None) -> str:
    """
    Return a file or folder as a key's part: its path relative to the rootdir, in
    posix form, so that two checkouts in different folders key it alike:

    - its real path's, when that is in the rootdir (also through a link inside
      the rootdir, or a rootdir reached through a link);
    - otherwise the path's as it is spelled, when that is in the rootdir: a
      folder linked into the checkout from a place that does not move with it
      (pytest's node IDs spell it so too);
    - otherwise its real path's, also outside the rootdir (``../shared/strategies.py``,
      for a folder next to the checkout).

    The absolute real path in posix form only when there is no relative one
    (another drive on Windows, or no rootdir). The file system's spelling is kept,
    so the key is the same on every OS.
    """
    real = os.path.realpath(path)
    if rootpath is not None:
        try:
            relative = os.path.relpath(real, os.path.realpath(rootpath))
        except ValueError:
            # Another drive on Windows
            return Path(real).as_posix()
        if _leaves(relative):
            try:
                spelled = os.path.relpath(os.path.abspath(path), os.path.abspath(rootpath))
            except ValueError:
                spelled = relative
            if not _leaves(spelled):
                relative = spelled
        return Path(relative).as_posix()
    return Path(real).as_posix()


def _leaves(relative: str) -> bool:
    """Whether a relative path leaves the folder it is relative to."""
    return relative == os.pardir or relative.startswith(os.pardir + os.sep)


def installed_part(path: str | os.PathLike[str]) -> str | None:
    """
    Return the part of a file's path below the last site-packages or dist-packages
    folder in it (``acme/tests/test_x.py`` for an installed package's file), in posix
    form, or None when it is in no such folder: an installed package's files are
    keyed alike wherever it is installed.
    """
    parts = PurePath(os.path.abspath(path)).parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in INSTALLED_FOLDERS:
            return PurePath(*parts[index + 1 :]).as_posix()
    return None


def file_part(path: str | os.PathLike[str], rootpath: str | os.PathLike[str] | None) -> str:
    """
    Return a file that pytest or the plugin imports by its path (a test module, a
    strategy file) as a key's part: :func:`installed_part` for an installed
    package's file (tests run with ``--pyargs``), :func:`path_part` otherwise.
    """
    installed = installed_part(path)
    return installed if installed is not None else path_part(path, rootpath)


class StreamKey:
    """
    The key of one random stream: a seed and a path of int and str parts.

    Build one with :meth:`root` and extend it with :meth:`child`; a key is not
    changed once built. Keys are equal when their paths are equal, and
    :meth:`seed_int` gives the int that seeds the stream's ``random.Random``.
    """

    __slots__ = ("_parts", "_path")

    _parts: tuple[Any, ...]
    _path: bytes

    def __init__(self) -> None:
        raise TypeError("StreamKey is built with StreamKey.root(seed, *parts)")

    @classmethod
    def _make(cls, parts: tuple[Any, ...], path: bytes) -> StreamKey:
        key = object.__new__(cls)
        key._parts = parts
        key._path = path
        return key

    @classmethod
    def root(cls, seed: int | str, *parts: int | str) -> StreamKey:
        """
        Return the key of a stream under ``seed``.

        Args:
            seed: The seed the stream derives from: the run's seed, or
                ``RNG.get_seed()`` for the streams of direct calls, through
                :func:`seed_part`
            *parts: The stream's path, each part an int or a str

        Raises:
            TypeError: If the seed or a part is not an int or a str (a bool or a
                float included)
        """
        everything = (seed, *parts)
        return cls._make(everything, b"".join([_part(p) for p in everything]))

    def child(self, *parts: int | str) -> StreamKey:
        """
        Return the key of a stream below this one: ``root(s, a).child(b)`` is
        ``root(s, a, b)``.

        Raises:
            TypeError: If a part is not an int or a str (a bool or a float included)
        """
        return self._make((*self._parts, *parts), self._path + b"".join([_part(p) for p in parts]))

    def seed_int(self) -> int:
        """Return the 128-bit int that seeds this key's stream."""
        hasher = _HASHER.copy()
        hasher.update(self._path)
        return int.from_bytes(hasher.digest(), "big")

    def child_seed_int(self, encoded: bytes) -> int:
        """
        Return ``self.child(*parts).seed_int()``, given ``encode(*parts)``, without
        building that key: the row streams derive one int per row and argument, and
        encode the argument names once.
        """
        hasher = _HASHER.copy()
        hasher.update(self._path)
        hasher.update(encoded)
        return int.from_bytes(hasher.digest(), "big")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamKey):
            return NotImplemented
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    def __repr__(self) -> str:
        return f"StreamKey({', '.join(repr(p) for p in self._parts)})"
