# test_args.py

from typing import Any, Callable


class TestArg:
    """
    Represents a single test argument with its generation strategy.

    A TestArg can be:
    - Static (fixed value)
    - Random (generated using an RNG type)
    - Directed (from a predefined list of values)
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    def __init__(
        self,
        name: str,
        rng_type: Any = None,
        value: Any = None,
        directed_values: list[Any] | None = None,
        test_values: list[Any] | None = None,
        validator: Callable[[Any], bool] | None = None,
        # Control
        always_include_directed: bool = True,
        description: str = "",  # Re-added description as it was removed in the instruction but not explicitly stated
    ):
        """
        Initialize a test argument.

        Args:
            name: Argument name (must match test function parameter)
            rng_type: RNG type for random generation (required if value is None)
            description: Human-readable description of the argument
            value: Single static value (for directed tests)
            directed_values: List of specific values to always test
            test_values: List of specific values for test mode only
            always_include_directed: If True, directed values are always included in samples
            validator: Optional function to validate generated values

        Raises:
            ValueError: If neither value, rng_type, nor directed_values are provided

        Examples:
            # Pure random
            TestArg("count", rng_type=RNGInteger(1, 100))

            # Static value
            TestArg("count", value=0, description="Edge case")

            # Mixed directed + random
            TestArg("count", rng_type=RNGInteger(1, 100), directed_values=[0, 1])
        """
        self._name = name
        self._rng_type = rng_type
        self._description = description
        self._value = value
        self._directed_values = directed_values or []
        self._test_values = test_values or []
        self._always_include_directed = always_include_directed
        self._validator = validator

        # Validation: must have at least one way to produce values
        if value is None and rng_type is None and not directed_values and not test_values:
            raise ValueError(
                f"TestArg '{name}' must have either a value, rng_type, directed_values, or test_values"
            )

    def generate(self) -> Any:
        """
        Generate a single value.

        Returns:
            Generated or static value

        Raises:
            ValueError: If no rng_type is available for generation
            ValueError: If generated value fails validation
        """
        # If static value, return it
        if self._value is not None:
            return self._validate(self._value)

        # If no RNG type, can't generate
        if self._rng_type is None:
            raise ValueError(f"Cannot generate value for '{self._name}' without rng_type")

        # Generate and validate
        value = self._rng_type.generate()
        return self._validate(value)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the test argument metadata to a dictionary.
        """
        data = {
            "name": self._name,
            "description": self._description,
            "has_static_value": self._value is not None,
            "has_directed_values": bool(self._directed_values),
            "has_test_values": bool(self._test_values),
            "always_include_directed": self._always_include_directed,
        }

        if self._value is not None:
            data["static_value"] = str(self._value)

        if self._rng_type:
            data["rng_type"] = self._rng_type.__class__.__name__
            # Add RNG specific details if available
            if hasattr(self._rng_type, "__dict__"):
                # Filter out private attributes and callables
                rng_details = {
                    k: str(v)
                    for k, v in self._rng_type.__dict__.items()
                    if not k.startswith("_") and not callable(v)
                }
                if rng_details:
                    data["rng_details"] = rng_details

        return data

    def generate_samples(self, n: int) -> list[Any]:
        """
        Generate n samples, optionally including directed values.

        Args:
            n: Number of random samples to generate

        Returns:
            List of samples. If always_include_directed is True and directed_values
            exist, the list will contain directed values + n random samples.
            If value is set (static), returns directed values or [value].

        Examples:
            # With directed values and n=10
            arg = TestArg("x", rng_type=RNGInteger(1, 100), directed_values=[0, 1])
            samples = arg.generate_samples(10)  # Returns 12 samples: [0, 1, ...10 random...]

            # Static value
            arg = TestArg("x", value=42)
            samples = arg.generate_samples(10)  # Returns [42]
        """
        samples = []

        # Add directed values if configured
        if self._always_include_directed and self._directed_values:
            samples.extend(self._directed_values)

        # If we have a static value, just return it (with directed values if any)
        if self._value is not None:
            if not samples:  # Only add static value if no directed values
                samples.append(self._value)
            return samples

        # Generate random samples
        if self._rng_type:
            for _ in range(n):
                samples.append(self.generate())

        return samples

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
    def name(self):
        """Get the argument name"""
        return self._name

    @property
    def description(self):
        """Get the argument description"""
        return self._description

    @property
    def type(self):
        """
        Get the Python type of this argument.

        Returns:
            Python type (int, float, str, etc.) or Any if unknown
        """
        if self._rng_type:
            return self._rng_type.python_type
        if self._value is not None:
            return type(self._value)
        if self._directed_values:
            return type(self._directed_values[0])
        return Any

    @property
    def is_static(self):
        """Check if this argument has a static value"""
        return self._value is not None

    @property
    def has_directed_values(self):
        """Check if this argument has directed test values"""
        return bool(self._directed_values)

    @property
    def rng_type(self):
        """Get the RNG type for this argument"""
        return self._rng_type

    @property
    def directed_values(self) -> list[Any]:
        """Get list of directed values."""
        return self._directed_values

    @property
    def test_values(self) -> list[Any]:
        """Get list of test values."""
        return self._test_values

    # ====
    # String Representation
    # ====

    def __repr__(self):
        """String representation for debugging"""
        if self._value is not None:
            return f"TestArg(name={self._name!r}, value={self._value!r})"

        parts = [f"name={self._name!r}"]

        if self._rng_type:
            parts.append(f"type={self.type.__name__}")

        if self._directed_values:
            parts.append(f"directed={len(self._directed_values)}")

        return f"TestArg({', '.join(parts)})"

    def __str__(self):
        """Human-readable string representation"""
        if self._description:
            return f"{self._name}: {self._description}"
        return self._name
