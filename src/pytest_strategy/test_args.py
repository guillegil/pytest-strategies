# test_args.py

import builtins
from collections.abc import Callable
from typing import Any


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
        value: Any = None,
        validator: Callable[[Any], bool] | None = None,
        description: str = "",
    ) -> None:
        """
        Initialize a test argument.

        Args:
            name: Argument name (must match test function parameter)
            rng_type: RNG type for random generation (required if value is None)
            description: Human-readable description of the argument
            value: Single static value (for directed tests)
            validator: Optional function to validate generated values

        Raises:
            ValueError: If neither value nor rng_type is provided

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

    def generate(self) -> Any:
        """
        Generate a single value.

        Returns:
            Generated or static value

        Raises:
            ValueError: If generated value fails validation
        """
        # If static value, return it
        if self._value is not None:
            return self._validate(self._value)

        # Generate and validate
        value = self._rng_type.generate()
        return self._validate(value)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the test argument metadata to a dictionary.
        """
        from enum import Enum

        data = {
            "name": self._name,
            "description": self._description,
            "has_static_value": self._value is not None,
        }

        if self._value is not None:
            data["static_value"] = str(self._value)

        if self._rng_type:
            data["rng_type"] = self._rng_type.__class__.__name__
            # Add RNG specific details if available
            if hasattr(self._rng_type, "__dict__"):
                # Filter out private attributes and callables, except Enum classes (the
                # enum_class of an RNGEnum), which are configuration and exported by name
                rng_details: dict[str, Any] = {}
                for k, v in self._rng_type.__dict__.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, type) and issubclass(v, Enum):
                        rng_details[k] = v.__name__
                    elif not callable(v):
                        rng_details[k] = str(v)
                # A set predicate is a callable and left out above, so say whether one is set
                if "predicate" in self._rng_type.__dict__:
                    rng_details["has_predicate"] = self._rng_type.predicate is not None
                if rng_details:
                    data["rng_details"] = rng_details

        return data

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
            Python type (int, float, str, etc.) or Any if unknown
        """
        if self._rng_type:
            python_type: builtins.type = self._rng_type.python_type
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
