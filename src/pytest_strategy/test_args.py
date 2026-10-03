# test_args.py

import builtins
from collections.abc import Callable
from typing import Any

from .rng import RNGType, RNGValueError, _NoValidValue


class TestArg:
    """
    Represents a single test argument with its generation strategy.

    A TestArg can be:
    - Static (fixed value)
    - Random (generated using an RNG type)

    Fixed rows of several arguments are the Parameter's directed_vectors and
    test_vectors.
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    def __init__(
        self,
        name: str,
        rng_type: Any = None,
        *,
        value: Any = None,
        validator: Callable[[Any], bool] | None = None,
        description: str = "",
    ) -> None:
        """
        Initialize a test argument.

        Args:
            name: Argument name (must match test function parameter)
            rng_type: RNG type for random generation (required if value is None): an
                RNGType, or an object with a generate() method
            value: Single static value (for directed tests)
            validator: Optional function to validate generated values
            description: Human-readable description of the argument

        The arguments after rng_type are keyword-only.

        Raises:
            ValueError: If neither value nor rng_type is provided
            TypeError: If rng_type is neither an RNGType nor has a generate() method
                (a bare function or lambda, for example), or is a class rather than
                an instance (RNGBoolean for RNGBoolean())

        Examples:
            # Pure random
            TestArg("count", rng_type=RNGInteger(1, 100))

            # Static value
            TestArg("count", value=0, description="Edge case")
        """
        self._name = name
        self._rng_type = rng_type
        self._description = description
        self._value = value
        self._validator = validator

        # Validation: must have one way to produce values
        if value is None and rng_type is None:
            raise ValueError(f"TestArg '{name}' must have a value or an rng_type")
        # A callable without generate() is rejected rather than called, so that a
        # later release can give such callables a meaning of their own. A class is
        # rejected too: its generate() is unbound, and RNGBoolean for RNGBoolean() is
        # the likely mistake
        if rng_type is not None and (
            isinstance(rng_type, type)
            or not (isinstance(rng_type, RNGType) or callable(getattr(rng_type, "generate", None)))
        ):
            got = repr(rng_type)
            if isinstance(rng_type, type) and callable(getattr(rng_type, "generate", None)):
                got = (
                    f"the class {rng_type.__name__} instead of an instance "
                    f"(did you mean {rng_type.__name__}(...)?)"
                )
            raise TypeError(
                f"TestArg '{name}' rng_type must be an RNGType or have a generate() method, "
                f"got {got}"
            )

    def generate(self) -> Any:
        """
        Generate a single value.

        Returns:
            Generated or static value

        Raises:
            ValueError: If generated value fails validation
            RNGValueError: If the rng_type's predicate rejected every draw; the
                message names this argument
        """
        # If static value, return it
        if self._value is not None:
            return self._validate(self._value)

        # Generate and validate
        try:
            value = self._rng_type.generate()
        except _NoValidValue as e:
            raise RNGValueError(
                f"Argument {self._name!r} could not draw a value its predicate accepts: {e}"
            ) from e
        return self._validate(value)

    def to_dict(self) -> dict[str, Any]:
        """
        Describe the argument as a JSON-ready dict: its ``name``, ``description``,
        ``python_type`` (the qualified name of its values' type, or None when unknown),
        ``validator`` (whether one is set) and ``source``: ``"value"`` with the
        ``value`` in the schema 1 value encoding, or ``"rng"`` with ``rng``, its RNG
        type's ``to_dict()``. A fragment of ``Parameter.to_dict()``, without a schema
        field of its own.
        """
        # Imported here: the export module imports this one
        from ._export import argument_dict

        return argument_dict(self)

    def generate_samples(self, n: int) -> list[Any]:
        """
        Generate n samples.

        Args:
            n: Number of random samples to generate

        Returns:
            List of n generated values, or [value] for a static argument.

        Examples:
            # Random, n=10
            arg = TestArg("x", rng_type=RNGInteger(1, 100))
            samples = arg.generate_samples(10)  # Returns 10 random values

            # Static value
            arg = TestArg("x", value=42)
            samples = arg.generate_samples(10)  # Returns [42]
        """
        # A static value is a single sample
        if self._value is not None:
            return [self._value]

        return [self.generate() for _ in range(n)]

    def _validate(self, value: Any) -> Any:
        """
        Validate a value using the validator function.

        Args:
            value: Value to validate

        Returns:
            The value if validation passes

        Raises:
            ValueError: If validation fails
        """
        if self._validator and not self._validator(value):
            raise ValueError(f"Value {value!r} failed validation for argument '{self._name}'")
        return value

    # ====
    # Properties
    # ====

    @property
    def name(self) -> str:
        """Get the argument name"""
        return self._name

    @property
    def description(self) -> str:
        """Get the argument description"""
        return self._description

    @property
    def type(self) -> builtins.type:
        """
        Get the Python type of this argument.

        Returns:
            Python type (int, float, str, etc.) or Any if unknown, as for an
            rng_type with a generate() method but no python_type
        """
        if self._rng_type:
            python_type: builtins.type = getattr(self._rng_type, "python_type", Any)
            return python_type
        if self._value is not None:
            return type(self._value)
        return Any

    @property
    def is_static(self) -> bool:
        """Check if this argument has a static value"""
        return self._value is not None

    @property
    def rng_type(self) -> Any:
        """Get the RNG type for this argument"""
        return self._rng_type

    # ====
    # String Representation
    # ====

    def __repr__(self) -> str:
        """String representation for debugging"""
        if self._value is not None:
            return f"TestArg(name={self._name!r}, value={self._value!r})"

        parts = [f"name={self._name!r}"]

        if self._rng_type:
            parts.append(f"type={self.type.__name__}")

        return f"TestArg({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string representation"""
        if self._description:
            return f"{self._name}: {self._description}"
        return self._name
