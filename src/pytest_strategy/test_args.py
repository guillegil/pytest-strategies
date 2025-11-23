# test_args.py

from typing import Any, Callable
from .rng import RNGType


class TestArg:
    """
    Define a single test argument for pytest parametrization.

    TestArg supports three modes:
    1. Static value: Fixed value for directed testing
    2. Random generation: Uses RNGType to generate random values
    3. Mixed: Directed values + random generation
    """

    def __init__(
        self,
        name: str,
        rng_type: RNGType | None = None,
        description: str = "",
        # For directed testing
        value: Any = None,
        directed_values: list[Any] | None = None,
        # Control
        always_include_directed: bool = True,
        validator: Callable[[Any], bool] | None = None,
    ):
        """
        Initialize a test argument.

        Args:
            name: Argument name (must match test function parameter)
            rng_type: RNG type for random generation (required if value is None)
            description: Human-readable description of the argument
            value: Single static value (for directed tests)
            directed_values: List of specific values to always test
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
        self._always_include_directed = always_include_directed
        self._validator = validator

        # Validation: must have at least one way to produce values
        if value is None and rng_type is None and not directed_values:
            raise ValueError(
                f"TestArg '{name}' must have either a value, rng_type, or directed_values"
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
            raise ValueError(
                f"Cannot generate value for '{self._name}' without rng_type"
            )

        # Generate and validate
        value = self._rng_type.generate()
        return self._validate(value)

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
            raise ValueError(
                f"Value {value!r} failed validation for argument '{self._name}'"
            )
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
    def directed_values(self):
        """Get the list of directed values"""
        return self._directed_values

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


# ====
# Example Usage
# ====

if __name__ == "__main__":
    from rng import RNGInteger, RNGChoice, RNGWeightedInteger, RNGFloat

    print("=== TestArg Examples ===\n")

    # Example 1: Pure random
    print("1. Pure Random Generation:")
    arg1 = TestArg(
        name="count",
        rng_type=RNGInteger(min=1, max=100),
        description="Random count between 1-100"
    )
    print(f"   {arg1}")
    print(f"   Type: {arg1.type.__name__}")
    print(f"   Sample: {arg1.generate()}")
    print(f"   5 Samples: {arg1.generate_samples(5)}\n")

    # Example 2: Static value (directed test)
    print("2. Static Value (Directed Test):")
    arg2 = TestArg(
        name="count",
        value=0,
        description="Edge case: zero"
    )
    print(f"   {arg2}")
    print(f"   Is static: {arg2.is_static}")
    print(f"   Samples: {arg2.generate_samples(10)}\n")

    # Example 3: Mix of directed + random
    print("3. Mixed (Directed + Random):")
    arg3 = TestArg(
        name="count",
        rng_type=RNGInteger(min=1, max=100),
        directed_values=[0, 1, 99, 100],
        description="Count with edge cases"
    )
    print(f"   {arg3}")
    print(f"   Has directed: {arg3.has_directed_values}")
    print(f"   10 Samples (includes 4 directed): {arg3.generate_samples(10)}\n")

    # Example 4: Weighted with validation
    print("4. Weighted with Validation:")
    arg4 = TestArg(
        name="port",
        rng_type=RNGWeightedInteger(
            ranges={
                (1024, 49151): 0.9,   # User ports (90%)
                (49152, 65535): 0.1   # Dynamic ports (10%)
            }
        ),
        validator=lambda x: 1024 <= x <= 65535,
        description="Network port number"
    )
    print(f"   {arg4}")
    print(f"   5 Samples: {arg4.generate_samples(5)}\n")

    # Example 5: Choice type
    print("5. Choice Type:")
    arg5 = TestArg(
        name="mode",
        rng_type=RNGChoice(choices=["fast", "slow", "medium"]),
        directed_values=["fast"],  # Always test fast mode
        description="Processing mode"
    )
    print(f"   {arg5}")
    print(f"   5 Samples: {arg5.generate_samples(5)}\n")

    # Example 6: Float with directed values
    print("6. Float with Directed Values:")
    arg6 = TestArg(
        name="threshold",
        rng_type=RNGFloat(min=0.0, max=1.0),
        directed_values=[0.0, 0.5, 1.0],
        description="Threshold value"
    )
    print(f"   {arg6}")
    print(f"   7 Samples: {arg6.generate_samples(7)}\n")

    # Example 7: Validation failure
    print("7. Validation Example:")
    arg7 = TestArg(
        name="positive",
        rng_type=RNGInteger(min=-10, max=10),
        validator=lambda x: x > 0,
        description="Must be positive"
    )
    print(f"   {arg7}")
    try:
        # This will retry until it gets a positive number
        print(f"   Valid sample: {arg7.generate()}")
    except ValueError as e:
        print(f"   Error: {e}")