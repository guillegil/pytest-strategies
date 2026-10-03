# rng.py

import _random
import builtins
import math
import os
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from ._streams import StreamKey, seed_part

T = TypeVar("T")
E = TypeVar("E", bound=Enum)

# The Mersenne Twister's own methods, which the ambient generator calls once its
# stream is seeded
_mt_random = _random.Random.random
_mt_getrandbits = _random.Random.getrandbits
_mt_seed = _random.Random.seed
_mt_getstate = _random.Random.getstate
_mt_setstate = _random.Random.setstate


class RNGValueError(ValueError):
    """Exception raised when an invalid value is provided to RNG operations."""


class _NoValidValue(RNGValueError):
    """A predicate rejected every draw. TestArg.generate re-raises it naming the argument."""


class _Ambient(random.Random):
    """
    The plugin's generator, ``RNG._ambient``: a ``random.Random`` that seeds the
    streams it is put on (``_Stream``) only when they are used.

    Entering a stream records the stream's key as pending; the generator seeds
    itself from it at the first draw, or when its state is read, and only then
    saves the state that the stream puts back when it ends. A stream that nothing
    draws from, such as most test phases and fixtures, then costs no seeding,
    saving or restoring of the Mersenne Twister state, about 20 us each.

    ``random.Random`` draws only through ``random()`` and ``getrandbits()``: its
    other methods call these two, and ``_randbelow`` stays the one that uses
    ``getrandbits()``. So the values are those of a generator seeded when the
    stream is entered. ``seed()`` and ``setstate()`` replace the state, so they
    skip the pending seeding, but still save the state the stream puts back.

    Seeding a pending stream, and entering or ending a stream, hold the
    generator's lock: a thread that draws while another enters or ends a stream
    (a stimulus thread started by a fixture) then never seeds the generator from
    a stream that has ended, or saves the state on another stream. A stream
    stays pending until it is seeded, so such a draw waits for the seeding
    instead of drawing from the state the seeding replaces. Its draws still come
    from whichever stream is in use. Streams that two threads enter may end in
    any order (``_Stream.__exit__``).
    """

    # random.Random's cached second value of gauss(), part of its state
    gauss_next: float | None

    def __init__(self, x: Any = None) -> None:
        # The key of the stream entered last, while nothing has seeded it (or a
        # function that returns the key); None when the state in use is the real one
        self._pending: StreamKey | Callable[[], StreamKey] | None = None
        # The streams entered and not yet ended, the innermost last
        self._streams: list[_Stream] = []
        # Held while a stream is seeded, entered or ended (see the class docstring),
        # and while a row is drawn (parameters.py); reentrant, so that a signal
        # handler, or a finalizer the garbage collector runs, that draws while its
        # thread holds it cannot deadlock. A forked child gets a new one.
        self._lock = threading.RLock()
        # True while the holder of the lock seeds a stream (_settle)
        self._settling = False
        super().__init__(x)

    def _settle(self, seed: bool = True) -> None:
        """
        Make the pending stream's state the real one: save the state in use, which
        belongs to the innermost stream seeded around it (or to no stream), for the
        stream to put back when it ends, and seed the generator from the stream's
        key, unless ``seed`` is False because the caller replaces the state.
        """
        with self._lock:
            # Read again under the lock: another thread may have seeded the stream,
            # or ended it, since the caller saw it pending. A draw from this thread
            # while it seeds the stream (a signal handler, or a finalizer) draws from
            # the state in use: seeding the stream there too would save the state of
            # the stream as the one to put back.
            pending = self._pending
            if pending is None or self._settling:
                return
            # Saved first, so that such a draw while the key is built does not move
            # the state the stream puts back
            state = _mt_getstate(self)
            self._settling = True
            try:
                # Such a draw before the flag was set seeded the stream itself
                if self._pending is not pending:
                    return
                seed_int = None
                if seed:
                    seed_int = (pending if isinstance(pending, StreamKey) else pending()).seed_int()
                stream = self._streams[-1]
                # Unless a stream that ended before it handed its own over (_Stream)
                if stream._state is None:
                    stream._state = state
                if seed_int is not None:
                    # As random.Random.seed() seeds with an int; gauss_next was cleared
                    # when the stream was entered
                    _mt_seed(self, seed_int)
                # Last: until then another thread's draw sees the stream pending, and
                # waits for the lock
                self._pending = None
            finally:
                self._settling = False

    def random(self) -> float:
        if self._pending is not None:
            self._settle()
        return _mt_random(self)

    def getrandbits(self, k: int, /) -> int:
        if self._pending is not None:
            self._settle()
        return _mt_getrandbits(self, k)

    def seed(self, a: Any = None, version: int = 2) -> None:
        if self._pending is not None:
            self._settle(seed=False)
        super().seed(a, version)

    def getstate(self) -> tuple[Any, ...]:
        if self._pending is not None:
            self._settle()
        return super().getstate()

    def setstate(self, state: tuple[Any, ...]) -> None:
        if self._pending is not None:
            self._settle(seed=False)
        super().setstate(state)

    def _position(self) -> object:
        """
        Return what the generator would draw from: the pending key while nothing
        has used the stream entered last, its state otherwise. It changes when
        something draws from the generator, seeds it or sets its state, and costs
        nothing while a stream is pending (the guard on the row streams compares it).
        """
        pending = self._pending
        return pending if pending is not None else self.getstate()


class RNG:
    """
    Core RNG singleton managing the seed and the random state.

    Every value the RNG types draw comes from a generator the plugin owns
    (:meth:`generator`), not from the global ``random`` state, so ``--rng-seed``
    reproduces them without seeding or disturbing the ``random`` calls of the
    code under test.
    """

    _seed = time.time_ns()
    # The draws an RNG type's predicate= gets before RNGValueError (fixed)
    _max_retries = 100
    # The plugin's generator. Each of the plugin's random streams (a factory call, a
    # strategy file's import, a test phase; see _Stream) reseeds this object in
    # place, when it is first used, and restores its state afterwards.
    _ambient = _Ambient(_seed)
    # The generator the RNG types and helpers draw from: the ambient generator,
    # except while an argument of a random row is drawn, which has its own
    _generator: random.Random = _ambient

    # ====
    # Seed Management
    # ====

    @staticmethod
    def seed(seed: int | None = None) -> None:
        """Set the seed and restart the generator from it.

        With ``None`` the current seed is kept, and the generator restarts from it.
        """
        if seed is not None:
            RNG._seed = seed
        RNG._generator.seed(RNG._seed)

    @staticmethod
    def get_seed() -> int:
        """Get the current seed value"""
        return RNG._seed

    @staticmethod
    def generator() -> random.Random:
        """
        Return the generator the RNG types draw from.

        A factory that needs other random operations (``shuffle``, ``gauss``) can
        draw from it, and gets values that ``--rng-seed`` reproduces. In a pytest
        run the plugin positions it on a stream of its own for each factory call,
        strategy file, test module, fixture and test phase (streams v1), so what
        one of them draws does not depend on what the others drew. Call it when
        you draw instead of keeping its result: while an argument of a random row
        is drawn it returns that argument's own generator, and a generator kept
        from a factory draws from whatever stream runs when it is used.
        """
        return RNG._generator

    @staticmethod
    def refresh_seed(key: str | int | None = None) -> None:
        """Restart the generator from the current seed.

        Args:
            key: Optional stream name, a str (or an int). With a key, the generator
                is seeded from the stream key ``(seed, "user", key)`` of streams v1,
                so each key gets its own stream that is the same on every run with
                this seed, in every process, and does not depend on the order in
                which keys are used. Any other object (a ``pathlib.Path``, a
                tuple) names the stream of its ``str()``, as 3.x formatted it.
        """
        if key is None:
            RNG._generator.seed(RNG._seed)
        else:
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                key = str(key)
            # Hashed with BLAKE2b (see _streams), so stable across processes, unlike
            # hash(), which is salted per process
            key_int = StreamKey.root(seed_part(RNG._seed), "user", key).seed_int()
            RNG._generator.seed(key_int)

    # ====
    # Internal Helper
    # ====

    @staticmethod
    def _generate_with_constraint(
        generator: Callable[[], T], predicate: Callable[[T], bool] | None = None
    ) -> T:
        """
        Helper to generate values with optional predicate constraint.

        Args:
            generator: Function that generates a random value
            predicate: Optional function to validate the generated value

        Returns:
            Generated value that satisfies the predicate

        Raises:
            RNGValueError: If no valid value found after max_retries attempts
        """
        if predicate is None:
            return generator()

        for _ in range(RNG._max_retries):
            value = generator()
            if predicate(value):
                return value

        raise _NoValidValue(f"No valid value found after {RNG._max_retries} attempts")

    @staticmethod
    def _string_args_error(
        length: int | None, min_length: int, max_length: int, charset: str
    ) -> str | None:
        """
        Describe what is wrong with string generation arguments, if anything.

        min_length and max_length are only checked when length is None, since
        they are ignored otherwise.

        Args:
            length: Fixed length, or None for a random length
            min_length: Minimum length if length is None
            max_length: Maximum length if length is None
            charset: Characters to choose from

        Returns:
            An error message starting with the argument name, or None if the arguments are valid
        """
        if length is not None:
            if length < 0:
                return f"length cannot be negative (got length={length})"
            longest = length
        else:
            if min_length < 0:
                return f"min_length cannot be negative (got min_length={min_length})"
            if min_length > max_length:
                return f"min_length ({min_length}) must be <= max_length ({max_length})"
            longest = max_length

        if longest > 0 and not charset:
            return "charset cannot be empty unless the length is 0"
        return None

    # ====
    # Basic Generators
    # ====

    @staticmethod
    def integer(
        min: int = -(2**31), max: int = 2**31 - 1, predicate: Callable[[int], bool] | None = None
    ) -> int:
        """
        Generate a random integer within the specified range.

        Args:
            min: Minimum value (inclusive)
            max: Maximum value (inclusive)
            predicate: Optional constraint function

        Returns:
            Random integer satisfying constraints

        Example:
            RNG.integer(1, 100)
            RNG.integer(1, 100, predicate=lambda x: x % 2 == 0)  # Even numbers only
        """
        if predicate is None:
            # The helper's draw without its call and closure, and with one lookup on
            # RNG fewer: random rows assign RNG._generator before each argument's
            # draw, and a lookup on a class whose attribute was just assigned costs more
            return RNG._generator.randint(min, max)
        return RNG._generate_with_constraint(lambda: RNG._generator.randint(min, max), predicate)

    @staticmethod
    def float(
        min: float = 0.0, max: float = 1.0, predicate: Callable[[float], bool] | None = None
    ) -> float:
        """
        Generate a random float within the specified range.

        Args:
            min: Minimum value (inclusive)
            max: Maximum value (inclusive)
            predicate: Optional constraint function

        Returns:
            Random float satisfying constraints

        Example:
            RNG.float(0.0, 10.0)
            RNG.float(0.0, 1.0, predicate=lambda x: x > 0.5)
        """
        if predicate is None:
            # The same draw without the helper (see integer())
            return _uniform(min, max)
        return RNG._generate_with_constraint(lambda: _uniform(min, max), predicate)

    @staticmethod
    def boolean(true_probability: builtins.float = 0.5) -> bool:
        """
        Generate a random boolean value.

        Args:
            true_probability: Probability of returning True (0.0 to 1.0)

        Returns:
            Random boolean

        Example:
            RNG.boolean()  # 50/50
            RNG.boolean(0.8)  # 80% True, 20% False
        """
        return RNG._generator.random() < true_probability

    @staticmethod
    def choice(items: list[T]) -> T:
        """
        Choose a random item from a list.

        Args:
            items: List of items to choose from

        Returns:
            Random item from the list

        Raises:
            RNGValueError: If the list is empty

        Example:
            RNG.choice(['a', 'b', 'c'])
        """
        if not items:
            raise RNGValueError("The choices list cannot be empty.")
        return RNG._generator.choice(items)

    @staticmethod
    def string(
        length: int | None = None,
        min_length: int = 1,
        max_length: int = 20,
        charset: str = "abcdefghijklmnopqrstuvwxyz",
    ) -> str:
        """
        Generate a random string.

        Args:
            length: Fixed length (if None, random between min_length and max_length)
            min_length: Minimum length if length is None (default: 1)
            max_length: Maximum length if length is None (default: 20)
            charset: Characters to choose from (default: lowercase letters)

        Returns:
            Random string of specified length

        Raises:
            ValueError: If length or min_length is negative, if min_length > max_length,
                or if charset is empty while the length can be greater than 0

        Example:
            RNG.string(length=10)  # Fixed length of 10
            RNG.string(min_length=5, max_length=15)  # Variable length 5-15
            RNG.string(length=8, charset="0123456789")  # Numeric string
            RNG.string(length=6, charset="ABCDEF0123456789")  # Hex string
        """
        error = RNG._string_args_error(length, min_length, max_length, charset)
        if error:
            raise ValueError(f"String {error}")

        if length is None:
            length = RNG._generator.randint(min_length, max_length)
        return "".join(RNG._generator.choice(charset) for _ in range(length))

    # ====
    # Weighted Generators
    # ====

    @staticmethod
    def winteger(
        ranges: dict[tuple[int, int], builtins.float],
        predicate: Callable[[int], bool] | None = None,
    ) -> int:
        """
        Generate a weighted integer from multiple ranges.

        Args:
            ranges: Dictionary mapping (min, max) tuples to weights
            predicate: Optional constraint function

        Returns:
            Random integer from weighted ranges

        Example:
            RNG.winteger({
                (0, 20): 0.8,      # 80% from 0-20
                (21, 100): 0.2     # 20% from 21-100
            })
        """
        range_list = list(ranges.keys())
        weights = list(ranges.values())

        def generator() -> int:
            # Choose range using random.choices (handles normalization). The range
            # is re-chosen on every predicate retry so that a range with no valid
            # value cannot exhaust all retries while other ranges could succeed.
            min_val, max_val = RNG._generator.choices(range_list, weights=weights, k=1)[0]
            return RNG.integer(min_val, max_val)

        return RNG._generate_with_constraint(generator, predicate)

    @staticmethod
    def wfloat(
        ranges: dict[tuple[builtins.float, builtins.float], builtins.float],
        predicate: Callable[[builtins.float], bool] | None = None,
    ) -> builtins.float:
        """
        Generate a weighted float from multiple ranges.

        Args:
            ranges: Dictionary mapping (min, max) tuples to weights
            predicate: Optional constraint function

        Returns:
            Random float from weighted ranges

        Example:
            RNG.wfloat({
                (0.0, 10.0): 0.8,
                (10.0, 100.0): 0.2
            })
        """
        range_list = list(ranges.keys())
        weights = list(ranges.values())

        def generator() -> builtins.float:
            # Re-choose the range on every predicate retry (see winteger)
            min_val, max_val = RNG._generator.choices(range_list, weights=weights, k=1)[0]
            return RNG.float(min_val, max_val)

        return RNG._generate_with_constraint(generator, predicate)


class _Stream:
    """
    Run a block on one of the plugin's random streams (streams v1, D5): ``with
    _Stream(key) as rng:``.

    In the block the ambient generator (``RNG._ambient``) draws from the stream of
    ``key``, as if reseeded in place from it, and is installed as
    ``RNG._generator``, so ``rng is RNG.generator()`` in the block, and a generator
    kept from it is the one the next stream reseeds. On exit its state, the seed
    (``RNG._seed``) and the generator installed before are put back, also when the
    block raises. Streams nest: an inner stream leaves the outer one where it was.
    So what the block draws, and an ``RNG.seed()`` call in it, change only the rest
    of the block.

    The generator is seeded only when the block first uses it (see ``_Ambient``),
    and ``key`` may be a function that returns the key, called then: a block that
    draws nothing costs neither the seeding nor the key.

    A class rather than a ``contextlib.contextmanager``: that one assigns the
    exception's ``__traceback__`` on the way out, which a frozen dataclass exception
    rejects.
    """

    __slots__ = ("_key", "_saved", "_state")

    def __init__(self, key: StreamKey | Callable[[], StreamKey]) -> None:
        self._key = key

    def __enter__(self) -> random.Random:
        ambient = RNG._ambient
        # Under the generator's lock, so that another thread's draw seeds no stream
        # meanwhile (see _Ambient)
        with ambient._lock:
            self._saved = (ambient, ambient._pending, ambient.gauss_next, RNG._generator, RNG._seed)
            # The state to put back, saved when the stream is seeded (_Ambient._settle)
            self._state: tuple[Any, ...] | None = None
            ambient._streams.append(self)
            ambient._pending = self._key
            # random.Random.seed() clears it; a pending stream must not see the outer one's
            ambient.gauss_next = None
            RNG._generator = ambient
        return ambient

    def __exit__(self, *exc_info: object) -> None:
        ambient, pending, gauss_next, generator, seed = self._saved
        with ambient._lock:
            streams = ambient._streams
            if streams and streams[-1] is self:
                streams.pop()
            elif self in streams:
                # Streams that two threads entered (export_strategies() in a thread)
                # can end in any order. One that ends before a stream entered after it
                # leaves the state in use, which is that stream's, and hands what it
                # would put back to it: the state it saved, when it was seeded, and
                # the pending key, seed and generator in use before it
                later = streams[streams.index(self) + 1]
                later._saved = self._saved
                if self._state is not None:
                    later._state = self._state
                streams.remove(self)
                return
            if self._state is not None:
                _mt_setstate(ambient, self._state)
            ambient._pending = pending
            ambient.gauss_next = gauss_next
            RNG._generator = generator
            RNG._seed = seed


def _before_fork() -> None:
    """
    Before ``os.fork()`` (a multiprocessing pool that forks), wait for any other
    thread to finish seeding, entering or ending a stream or drawing a row, so that
    the child starts from a generator nothing is changing.
    """
    RNG._ambient._lock.acquire()


def _after_fork_in_parent() -> None:
    RNG._ambient._lock.release()


def _after_fork_in_child() -> None:
    """
    Give the ambient generator a new lock in a forked child, as ``logging`` does for
    its own: the one the parent held for the fork would otherwise stay held.
    """
    RNG._ambient._lock = threading.RLock()


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_after_fork_in_child,
    )


# ====
# RNG Type Classes
# ====


def _uniform(a: float, b: float) -> float:
    """
    Return uniform(a, b) from the RNG generator, without overflowing when b - a exceeds the
    largest float.

    The same draw and formula as the standard library, so a seed gives the same
    values. b - a only overflows when a and b have opposite signs, and the second
    form cannot overflow then, and stays within [a, b].
    """
    r = RNG._generator.random()
    x = a + (b - a) * r
    return x if math.isfinite(x) else a * (1.0 - r) + b * r


def _is_finite(value: float) -> bool:
    """math.isfinite, also False for an int too large for a float."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_finite(owner: str, *bounds: float) -> None:
    """Raise RNGValueError if a float bound is inf or nan."""
    for bound in bounds:
        if not _is_finite(bound):
            raise RNGValueError(f"{owner} bounds must be finite, got {bound!r}")


def _check_ranges(owner: str, ranges: Mapping[Any, float], finite: bool = False) -> None:
    """
    Raise RNGValueError unless every key of ranges is a (min, max) pair with min <= max.

    Args:
        owner: Name of the RNG type, used in the error message
        ranges: Mapping of (min, max) tuples to their weights
        finite: Also require finite bounds (float ranges)
    """
    for key in ranges:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise RNGValueError(f"{owner} range {key!r} must be a (min, max) tuple")
        if finite and not all(_is_finite(bound) for bound in key):
            raise RNGValueError(f"{owner} range {key!r} must have finite bounds")
        if not key[0] <= key[1]:  # also rejects NaN, like _check_bounds
            raise RNGValueError(f"{owner} range {key!r} must have min <= max")


def _check_bounds(
    owner: str, min_val: float, max_val: float, min_given: bool, max_given: bool
) -> None:
    """
    Raise RNGValueError if min_val > max_val, naming the bound that fell back to its default.

    Args:
        owner: Name of the RNG type, used in the error message
        min_val: Minimum value after filling defaults
        max_val: Maximum value after filling defaults
        min_given: Whether the caller passed min explicitly
        max_given: Whether the caller passed max explicitly

    Raises:
        RNGValueError: If min_val > max_val
    """
    if min_val <= max_val:
        return

    note = ""
    if not max_given:
        note = f" (max was not given and defaults to {max_val})"
    elif not min_given:
        note = f" (min was not given and defaults to {min_val})"
    raise RNGValueError(f"{owner} min ({min_val}) must be <= max ({max_val}){note}")


def _check_weights(owner: str, weights: Mapping[Any, float]) -> None:
    """
    Raise RNGValueError unless weights can be used for weighted selection.

    Individual zero weights are allowed (they exclude an entry), but random.choices
    silently skews the distribution for negative weights and only rejects an
    all-zero total when a value is generated.

    Args:
        owner: Name of the RNG type, used in the error message
        weights: Mapping of choices to their weights

    Raises:
        RNGValueError: If weights is empty, contains a negative or non-finite
            weight, all weights are zero, or their total is not finite
    """
    if not weights:
        raise RNGValueError(f"{owner} weights cannot be empty")

    for key, weight in weights.items():
        if not _is_finite(weight) or weight < 0:
            raise RNGValueError(
                f"{owner} weight for {key!r} must be a finite number >= 0, got {weight!r}"
            )

    # random.choices rejects a total that overflows, on every draw
    try:
        total = float(sum(weights.values()))
    except OverflowError:
        total = math.inf
    if not math.isfinite(total):
        raise RNGValueError(f"{owner} weights must have a finite total")
    if total <= 0:
        raise RNGValueError(f"{owner} weights cannot all be zero")


class RNGType(Generic[T]):
    """
    Base class for all RNG types.

    A subclass implements ``generate()``, which is called once per draw of an
    argument of a random or exhaustive row, while ``RNG.generator()`` is that
    argument's own stream for the row (streams v1). So that a row's values depend
    only on the seed, the test, the row and the argument:

    - draw inside ``generate()``, from ``RNG.generator()`` or the ``RNG.*``
      helpers called there; never from a generator kept from earlier, such as
      the factory's ``rng``, which draws from whatever stream runs when it is used;
    - return a value that does not depend on state kept between calls. A counter
      that walks a pattern (walking ones, for example) makes row k depend on the
      rows drawn before it.

    The plugin warns (``PytestStrategiesWarning``) when something draws from its
    generator while a test's rows are generated, as a kept ``rng`` does.
    """

    def generate(self) -> T:
        """
        Generate a random value based on this type's configuration.

        Draw from ``RNG.generator()`` or the ``RNG.*`` helpers here, not from a
        generator kept from earlier, and keep no state between calls (see the
        class).
        """
        raise NotImplementedError

    @property
    def python_type(self) -> type[T]:
        """Return the Python type this RNG type generates"""
        raise NotImplementedError


class RNGInteger(RNGType[int]):
    """RNG type for generating integers"""

    def __init__(
        self,
        min: int | None = None,
        max: int | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> None:
        self.min = min if min is not None else -(2**31)
        self.max = max if max is not None else 2**31 - 1
        self.predicate = predicate
        _check_bounds("RNGInteger", self.min, self.max, min is not None, max is not None)

    def generate(self) -> int:
        return RNG.integer(self.min, self.max, self.predicate)

    @property
    def python_type(self) -> type[int]:
        return int


class RNGFloat(RNGType[float]):
    """RNG type for generating floats"""

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        predicate: Callable[[float], bool] | None = None,
    ) -> None:
        self.min = min if min is not None else 0.0
        self.max = max if max is not None else 1.0
        self.predicate = predicate
        _check_finite("RNGFloat", self.min, self.max)
        _check_bounds("RNGFloat", self.min, self.max, min is not None, max is not None)

    def generate(self) -> float:
        return RNG.float(self.min, self.max, self.predicate)

    @property
    def python_type(self) -> type[float]:
        return float


class RNGBoolean(RNGType[bool]):
    """RNG type for generating booleans"""

    def __init__(self, true_probability: float = 0.5) -> None:
        self.true_probability = true_probability

    def generate(self) -> bool:
        return RNG.boolean(self.true_probability)

    @property
    def python_type(self) -> type[bool]:
        return bool


class RNGChoice(RNGType[T]):
    """RNG type for choosing from a list of options"""

    def __init__(self, choices: list[T]) -> None:
        if not choices:
            raise RNGValueError("Choices list cannot be empty")
        self.choices = choices

    def generate(self) -> T:
        return RNG.choice(self.choices)

    @property
    def python_type(self) -> type[T]:
        # object when the list was emptied after construction
        return type(self.choices[0]) if self.choices else cast("type[T]", object)


class RNGEnum(RNGType[E]):
    """
    RNG type for choosing from Python Enum values.

    Supports:
    - Simple random selection from all enum members
    - Weighted probabilities for specific enum values
    - Predicate constraints to filter valid values

    Examples:
        # Simple enum selection
        RNGEnum(MyEnum)

        # Weighted selection (70% SUCCESS, 20% PENDING, 10% FAILED)
        RNGEnum(Status, weights={Status.SUCCESS: 0.7, Status.PENDING: 0.2, Status.FAILED: 0.1})

        # With predicate (only non-error statuses)
        RNGEnum(Status, predicate=lambda s: s != Status.ERROR)

        # Weighted with predicate
        RNGEnum(Priority, weights={Priority.HIGH: 0.6, Priority.MEDIUM: 0.3}, predicate=lambda p: p != Priority.LOW)
    """

    def __init__(
        self,
        enum_class: type[E],
        weights: dict[E, float] | None = None,
        predicate: Callable[[E], bool] | None = None,
    ) -> None:
        """
        Initialize RNGEnum.

        Args:
            enum_class: The Enum class to generate values from
            weights: Optional dictionary mapping enum members to their weights.
                    If provided, only weighted members will be selected.
                    Weights don't need to sum to 1.0 (they'll be normalized).
            predicate: Optional function to filter valid enum values. Each draw picks
                    only among the candidate members it accepts, using the current
                    predicate and weights. It is also checked here against every
                    candidate member, so an unsatisfiable predicate fails at once.

        Raises:
            RNGValueError: If enum_class is not an Enum class, if it has no members to
                draw from and no weights are given, if weights reference non-existent
                members, if weights are empty, negative, non-finite or all zero, or if
                no member (with a positive weight) satisfies the predicate
        """
        # isinstance guard first: issubclass raises TypeError for non-classes (e.g. a member)
        if not (isinstance(enum_class, type) and issubclass(enum_class, Enum)):
            raise RNGValueError(f"{enum_class!r} is not an Enum class")
        # Uniform selection draws from iteration, which for a Flag skips zero-valued and
        # multi-bit members. Weighted selection draws from the weights keys instead.
        if weights is None and len(enum_class) == 0:
            raise RNGValueError(f"{enum_class.__name__} has no members to choose from")

        self.enum_class = enum_class
        self.weights = weights
        self.predicate = predicate

        # Validate weights if provided
        if weights is not None:
            for member in weights:
                if not isinstance(member, enum_class):
                    raise RNGValueError(
                        f"Weight key {member} is not a member of {enum_class.__name__}"
                    )
            _check_weights("RNGEnum", weights)

        # Fail now, not at the first draw, when no member satisfies the predicate
        if predicate:
            self._filter_by_predicate(predicate)

    def _filter_by_predicate(self, predicate: Callable[[E], bool]) -> tuple[list[E], list[float]]:
        """
        Return the candidate members the predicate accepts, with their weights.

        Filtering the finite set of candidates (instead of retrying draws) means a draw
        never fails while a valid member exists. The candidates come from the current
        weights on every call, so reassigning predicate or weights takes effect.

        Args:
            predicate: The predicate to filter the candidate members with

        Raises:
            RNGValueError: If no candidate member with a positive weight is accepted
        """
        # Same candidates generate() draws from: the weighted members, else all members
        candidates = self.weights if self.weights else dict.fromkeys(self.enum_class, 1.0)
        members: list[E] = []
        weights: list[float] = []
        for member, weight in candidates.items():
            if predicate(member):
                members.append(member)
                weights.append(weight)

        if sum(weights) <= 0:
            which = "weighted member with a positive weight" if self.weights else "member"
            raise _NoValidValue(
                f"No valid value found: no {which} of {self.enum_class.__name__} "
                "satisfies the predicate"
            )
        return members, weights

    def generate(self) -> E:
        """
        Generate a random enum value.

        Returns:
            Random enum member satisfying constraints
        """
        if self.weights:
            # Weighted selection
            if self.predicate:
                # With predicate: choose among the weighted members it accepts
                members, weights = self._filter_by_predicate(self.predicate)
                return RNG._generator.choices(members, weights=weights, k=1)[0]

            # Without predicate: direct selection
            members = list(self.weights.keys())
            weights = list(self.weights.values())
            return RNG._generator.choices(members, weights=weights, k=1)[0]
        else:
            # Uniform selection from all members
            if self.predicate:
                # With predicate: choose among the members it accepts
                return RNG._generator.choice(self._filter_by_predicate(self.predicate)[0])

            # Without predicate: direct selection
            return RNG._generator.choice(list(self.enum_class))

    @property
    def python_type(self) -> type[E]:
        """Return the Enum class type"""
        return self.enum_class


class SequenceLike(RNGType[T]):
    """
    Abstract base class for sequence-based RNG types.

    Subclasses differ in the order they give the sequence's positions in exhaustive
    (auto) mode, via ``_auto_positions()``. In finite mode, Parameter cycles
    through Series values in order, and RNGSequence draws random elements via
    ``generate()``. With ``Parameter(per_sequence_samples=True)``, both are walked
    in declaration order with n rows for each value.

    An empty sequence (or one the predicate empties) raises, unless
    ``skip_if_empty`` gives a reason: the tests of a strategy with such an arg are
    then reported as skipped with that reason. Use it when the values come from
    configuration that may legitimately have none.
    """

    def __init__(
        self,
        sequence: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
        *,
        skip_if_empty: str | None = None,
    ) -> None:
        # A config lookup with no entry often yields None rather than an empty list
        if sequence is None:
            raise RNGValueError(f"{type(self).__name__} requires a sequence, got None")
        # Sets iterate in hash order, which for str/bytes changes with PYTHONHASHSEED,
        # so the same --rng-seed would give different values in each process
        if isinstance(sequence, (set, frozenset)):
            raise RNGValueError(
                f"{type(self).__name__} requires an ordered sequence, got a "
                f"{type(sequence).__name__} whose iteration order is not reproducible "
                "across runs; use sorted(...) or a list"
            )
        if predicate is not None and not callable(predicate):
            raise RNGValueError(
                f"{type(self).__name__} predicate must be callable, got {predicate!r}"
                + (" (did you mean skip_if_empty=...?)" if isinstance(predicate, str) else "")
            )
        if skip_if_empty is not None and (
            not isinstance(skip_if_empty, str) or not skip_if_empty.strip()
        ):
            raise RNGValueError(
                f"{type(self).__name__} skip_if_empty must be a non-empty reason string, "
                f"got {skip_if_empty!r}"
            )

        self.sequence = list(sequence)
        self.skip_if_empty = skip_if_empty

        # Apply predicate if provided
        if predicate:
            self.sequence = [x for x in self.sequence if predicate(x)]

        if not self.sequence and skip_if_empty is None:
            raise RNGValueError(
                "Sequence cannot be empty (or all items were filtered by predicate). "
                "Pass skip_if_empty='<reason>' to skip the strategy's tests instead."
            )

    def _get_auto_sequence(self) -> list[T]:
        """Return the ordered list to use for exhaustive (auto) mode: the values in
        the order of ``_auto_positions()``.

        A subclass that overrides this instead of ``_auto_positions()`` gets each
        value's position found in the sequence, so it must return values of its
        sequence, each at most as often as the sequence lists it: any other value
        fails ``--nsamples=auto`` with ``RNGValueError``.
        """
        return [self.sequence[position] for position in self._auto_positions()]

    def _auto_positions(self) -> list[int]:
        """Return the positions of the sequence in exhaustive (auto) mode's order.

        The rows of exhaustive mode are keyed and labeled by these positions, so a
        value listed twice still gives two rows. Subclasses MUST override this
        method (or ``_get_auto_sequence()``).
        """
        raise NotImplementedError

    @property
    def skip_reason(self) -> str | None:
        """The skip_if_empty reason when the sequence is empty, otherwise None."""
        return None if self.sequence else self.skip_if_empty

    def generate(self) -> T:
        """Generate a random value from the sequence (normal / finite mode)."""
        if not self.sequence:
            raise RNGValueError(
                f"{type(self).__name__} has no values to draw from ({self.skip_if_empty})"
            )
        return RNG.choice(self.sequence)

    @property
    def python_type(self) -> type[T]:
        # object for an empty sequence (skip_if_empty), which has no element type
        return type(self.sequence[0]) if self.sequence else cast("type[T]", object)


class RNGSequence(SequenceLike[T]):
    """
    RNG type for sequences of values.

    In normal mode, acts like RNGChoice (picks random values), unless the
    Parameter sets per_sequence_samples=True, which walks the values in
    declaration order with n rows each.
    In exhaustive mode (nsamples="auto"), produces a permutation of the
    sequence (each value exactly once, random order).
    """

    def _auto_positions(self) -> list[int]:
        """Return a random permutation of the sequence's positions for exhaustive mode."""
        # sample() picks positions whatever the population holds, so this draws the
        # permutation sample(self.sequence, ...) drew in 3.0
        size = len(self.sequence)
        return RNG._generator.sample(range(size), size)


class Series(SequenceLike[T]):
    """
    Deterministic ordered sequence type.

    In auto mode, produces values in their original declaration order.
    In finite mode, cycles (K >= len) or truncates to the first K (K < len), or
    gives K rows for each value when the Parameter sets per_sequence_samples=True.
    Multiple Series args produce the full Cartesian product in
    itertools.product order (leftmost arg is the slowest counter).
    """

    def _auto_positions(self) -> list[int]:
        """Return the sequence's positions in their original order for exhaustive mode."""
        return list(range(len(self.sequence)))


class RNGString(RNGType[str]):
    """RNG type for generating strings"""

    def __init__(
        self,
        length: int | None = None,
        min_length: int = 1,
        max_length: int = 20,
        charset: str = "abcdefghijklmnopqrstuvwxyz",
    ) -> None:
        # Fail when the strategy is defined, not only for the seeds that hit the bad case
        error = RNG._string_args_error(length, min_length, max_length, charset)
        if error:
            raise RNGValueError(f"RNGString {error}")

        self.length = length
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset

    def generate(self) -> str:
        return RNG.string(self.length, self.min_length, self.max_length, self.charset)

    @property
    def python_type(self) -> type[str]:
        return str


class RNGWeightedInteger(RNGType[int]):
    """RNG type for generating weighted integers from multiple ranges"""

    def __init__(
        self, ranges: dict[tuple[int, int], float], predicate: Callable[[int], bool] | None = None
    ) -> None:
        _check_weights("RNGWeightedInteger", ranges)
        _check_ranges("RNGWeightedInteger", ranges)
        self.ranges = ranges
        self.predicate = predicate

    def generate(self) -> int:
        return RNG.winteger(self.ranges, self.predicate)

    @property
    def python_type(self) -> type[int]:
        return int


class RNGWeightedFloat(RNGType[float]):
    """RNG type for generating weighted floats from multiple ranges"""

    def __init__(
        self,
        ranges: dict[tuple[float, float], float],
        predicate: Callable[[float], bool] | None = None,
    ) -> None:
        _check_weights("RNGWeightedFloat", ranges)
        _check_ranges("RNGWeightedFloat", ranges, finite=True)
        self.ranges = ranges
        self.predicate = predicate

    def generate(self) -> float:
        return RNG.wfloat(self.ranges, self.predicate)

    @property
    def python_type(self) -> type[float]:
        return float
