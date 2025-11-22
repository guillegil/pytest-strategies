# parameter.py

from typing import Callable
from test_args import TestArg


class Parameter:
    """
    Container for multiple TestArg instances that define test parameters.
    Handles generation of parameter vectors (tuples of values).

    A parameter vector is a tuple of values, one for each TestArg.
    Directed vectors are named, predefined test cases that can be
    referenced by name or index via CLI arguments.
    """

    def __init__(
        self,
        *test_args: TestArg,
        directed_vectors: dict[str, tuple] | None = None,
        always_include_directed: bool = True,
        vector_constraints: list[Callable[[tuple], bool]] | None = None,
    ):
        """
        Initialize a Parameter container.

        Args:
            *test_args: Variable number of TestArg instances
            directed_vectors: Dictionary mapping vector names to value tuples
            always_include_directed: If True, directed vectors are included in "mixed" mode
            vector_constraints: List of functions that validate entire parameter vectors

        Raises:
            ValueError: If directed vectors don't match the number of test args

        Examples:
            # Simple parameter with 2 args
            param = Parameter(
                TestArg("x", rng_type=RNGInteger(0, 10)),
                TestArg("y", rng_type=RNGInteger(0, 10))
            )

            # With directed vectors
            param = Parameter(
                TestArg("x", rng_type=RNGInteger(0, 10)),
                TestArg("y", rng_type=RNGInteger(0, 10)),
                directed_vectors={
                    "origin": (0, 0),
                    "max": (10, 10),
                }
            )
        """
        self.test_args = list(test_args)
        self.directed_vectors = directed_vectors or {}
        self.always_include_directed = always_include_directed
        self.vector_constraints = vector_constraints or []

        # Validate directed vectors on initialization
        self._validate_directed_vectors()

    def _validate_directed_vectors(self):
        """
        Ensure all directed vectors match the number of test args.

        Raises:
            ValueError: If any directed vector has wrong number of values
        """
        expected_len = len(self.test_args)
        for name, vector in self.directed_vectors.items():
            if len(vector) != expected_len:
                raise ValueError(
                    f"Directed vector '{name}' has {len(vector)} values, "
                    f"expected {expected_len}"
                )

    def _validate_vector(self, vector: tuple) -> bool:
        """
        Validate a vector against all constraints.

        Args:
            vector: Parameter vector to validate

        Returns:
            True if all constraints pass, False otherwise
        """
        for constraint in self.vector_constraints:
            if not constraint(vector):
                return False
        return True

    # ====
    # Vector Management
    # ====

    def add_directed_vector(self, name: str, values: tuple):
        """
        Add a named directed test vector.

        Args:
            name: Unique name for the vector
            values: Tuple of values matching test_args length

        Raises:
            ValueError: If vector length doesn't match test_args

        Example:
            param.add_directed_vector("edge_case", (0, 100, "fast"))
        """
        if len(values) != len(self.test_args):
            raise ValueError(
                f"Vector must have {len(self.test_args)} values, got {len(values)}"
            )
        self.directed_vectors[name] = values

    def remove_directed_vector(self, name: str):
        """
        Remove a directed vector by name.

        Args:
            name: Name of the vector to remove

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self.directed_vectors:
            raise KeyError(f"No directed vector named '{name}'")
        del self.directed_vectors[name]

    def get_directed_vector(self, name: str) -> tuple:
        """
        Get a specific directed vector by name.

        Args:
            name: Name of the vector

        Returns:
            The directed vector tuple

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self.directed_vectors:
            raise KeyError(f"No directed vector named '{name}'")
        return self.directed_vectors[name]

    # ====
    # Sample Generation
    # ====

    def generate_vector(self) -> tuple:
        """
        Generate a single random parameter vector.

        Returns:
            Tuple of generated values, one per TestArg

        Raises:
            ValueError: If generated vector fails constraints

        Example:
            vector = param.generate_vector()  # e.g., (5, 3.14, "fast")
        """
        max_retries = 100

        for _ in range(max_retries):
            vector = tuple(arg.generate() for arg in self.test_args)

            # Check constraints
            if self._validate_vector(vector):
                return vector

        raise ValueError(
            f"Could not generate valid vector after {max_retries} attempts. "
            "Check your constraints."
        )

    def generate_samples(
        self, 
        n: int,
        mode: str = "all",
        filter_by_name: str | None = None,
        filter_by_index: int | None = None,
    ) -> list[tuple]:
        """
        Generate parameter vectors.

        Args:
            n: Number of random samples to generate
            mode: Sampling mode
                - "all": All directed vectors + n random samples (default)
                - "random_only": Only n random samples, no directed
                - "directed_only": Only directed vectors, ignore n
                - "mixed": Directed (if always_include_directed=True) + n random
            filter_by_name: Only return this directed vector (for -vn CLI)
            filter_by_index: Only return directed vector at index (for -vi CLI)

        Returns:
            List of parameter vectors (tuples)

        Examples:
            # All directed + 10 random
            samples = param.generate_samples(10, mode="all")

            # Only random
            samples = param.generate_samples(10, mode="random_only")

            # Only directed
            samples = param.generate_samples(0, mode="directed_only")

            # Get specific vector by name
            samples = param.generate_samples(0, filter_by_name="edge_case")

            # Get specific vector by index
            samples = param.generate_samples(0, filter_by_index=0)
        """
        samples = []

        # Handle CLI filters first (override mode)
        if filter_by_name:
            return [self.get_vector_by_name(filter_by_name)]

        if filter_by_index is not None:
            return [self.get_vector_by_index(filter_by_index)]

        # Validate mode
        valid_modes = ["all", "random_only", "directed_only", "mixed"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        # Mode: directed_only
        if mode == "directed_only":
            return list(self.directed_vectors.values())

        # Mode: all - always include all directed vectors
        if mode == "all":
            samples.extend(self.directed_vectors.values())

        # Mode: mixed - respect always_include_directed flag
        elif mode == "mixed":
            if self.always_include_directed:
                samples.extend(self.directed_vectors.values())

        # Mode: random_only - skip directed vectors entirely
        # (no action needed, samples stays empty)

        # Generate random samples (for all modes except directed_only)
        if mode != "directed_only":
            for _ in range(n):
                samples.append(self.generate_vector())

        return samples

    # ====
    # CLI Support
    # ====

    def get_vector_by_name(self, name: str) -> tuple:
        """
        Get directed vector by name (for -vn CLI argument).

        Args:
            name: Name of the directed vector

        Returns:
            The directed vector tuple

        Raises:
            KeyError: If vector name doesn't exist
        """
        if name not in self.directed_vectors:
            available = ', '.join(self.directed_vectors.keys())
            raise KeyError(
                f"No directed vector named '{name}'. "
                f"Available: {available}"
            )
        return self.directed_vectors[name]

    def get_vector_by_index(self, index: int) -> tuple:
        """
        Get directed vector by index (for -vi CLI argument).

        Args:
            index: Index of the directed vector (0-based)

        Returns:
            The directed vector tuple

        Raises:
            IndexError: If index is out of range
        """
        names = list(self.directed_vectors.keys())
        if index < 0 or index >= len(names):
            raise IndexError(
                f"Vector index {index} out of range. "
                f"Valid range: 0-{len(names)-1}"
            )
        return self.directed_vectors[names[index]]

    def list_vector_names(self) -> list[str]:
        """
        List all directed vector names.

        Returns:
            List of vector names in order
        """
        return list(self.directed_vectors.keys())

    # ====
    # Constraint Management
    # ====

    def add_constraint(self, constraint: Callable[[tuple], bool]):
        """
        Add a constraint that validates entire parameter vectors.

        Args:
            constraint: Function that takes a vector tuple and returns bool

        Example:
            # Ensure first arg < second arg
            param.add_constraint(lambda v: v[0] < v[1])
        """
        self.vector_constraints.append(constraint)

    def clear_constraints(self):
        """Remove all vector constraints."""
        self.vector_constraints = []

    # ====
    # Introspection
    # ====

    @property
    def arg_names(self) -> tuple[str, ...]:
        """Get tuple of argument names."""
        return tuple(arg.name for arg in self.test_args)

    @property
    def arg_types(self) -> tuple[type, ...]:
        """Get tuple of argument types."""
        return tuple(arg.type for arg in self.test_args)

    @property
    def vector_names(self) -> list[str]:
        """Get list of directed vector names."""
        return list(self.directed_vectors.keys())

    @property
    def num_args(self) -> int:
        """Get number of test arguments."""
        return len(self.test_args)

    @property
    def num_directed_vectors(self) -> int:
        """Get number of directed vectors."""
        return len(self.directed_vectors)

    def get_arg(self, name: str) -> TestArg:
        """
        Get TestArg by name.

        Args:
            name: Name of the argument

        Returns:
            The TestArg instance

        Raises:
            KeyError: If argument name doesn't exist
        """
        for arg in self.test_args:
            if arg.name == name:
                return arg
        raise KeyError(f"No argument named '{name}'")

    # ====
    # String Representation
    # ====

    def __repr__(self):
        """String representation for debugging."""
        return (
            f"Parameter(args={self.num_args}, "
            f"directed_vectors={self.num_directed_vectors})"
        )

    def __str__(self):
        """Human-readable string representation."""
        args_str = ", ".join(self.arg_names)
        vectors_str = ", ".join(self.vector_names) if self.vector_names else "none"
        return (
            f"Parameter({args_str})\n"
            f"  Directed vectors: {vectors_str}"
        )


# ====
# Example Usage
# ====

if __name__ == "__main__":
    from rng import RNGInteger, RNGFloat, RNGChoice

    print("=== Parameter Class Examples ===\n")

    # Example 1: Basic parameter with directed vectors
    print("1. Basic Parameter with Directed Vectors:")
    param1 = Parameter(
        TestArg("count", rng_type=RNGInteger(0, 100)),
        TestArg("timeout", rng_type=RNGFloat(0.1, 10.0)),
        TestArg("mode", rng_type=RNGChoice(["fast", "slow"])),
        directed_vectors={
            "edge_zero": (0, 0.1, "fast"),
            "edge_max": (100, 10.0, "slow"),
            "typical": (50, 5.0, "fast"),
        }
    )
    print(f"   {param1}")
    print(f"   Arg names: {param1.arg_names}")
    print(f"   Arg types: {param1.arg_types}\n")

    # Example 2: Generate samples - all mode
    print("2. Generate Samples - 'all' mode (3 directed + 5 random):")
    samples = param1.generate_samples(5, mode="all")
    print(f"   Total samples: {len(samples)}")
    for i, sample in enumerate(samples[:3]):
        print(f"   Sample {i}: {sample}")
    print(f"   ... and {len(samples) - 3} more random samples\n")

    # Example 3: Generate samples - random_only mode
    print("3. Generate Samples - 'random_only' mode:")
    samples = param1.generate_samples(5, mode="random_only")
    print(f"   Total samples: {len(samples)}")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}: {sample}")
    print()

    # Example 4: Generate samples - directed_only mode
    print("4. Generate Samples - 'directed_only' mode:")
    samples = param1.generate_samples(0, mode="directed_only")
    print(f"   Total samples: {len(samples)}")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}: {sample}")
    print()

    # Example 5: CLI filter by name
    print("5. CLI Filter - Get vector by name:")
    sample = param1.generate_samples(0, filter_by_name="edge_zero")
    print(f"   Vector 'edge_zero': {sample[0]}\n")

    # Example 6: CLI filter by index
    print("6. CLI Filter - Get vector by index:")
    sample = param1.generate_samples(0, filter_by_index=1)
    print(f"   Vector at index 1: {sample[0]}\n")

    # Example 7: Add directed vector dynamically
    print("7. Add Directed Vector Dynamically:")
    param1.add_directed_vector("custom", (25, 2.5, "slow"))
    print(f"   Vector names: {param1.vector_names}")
    print(f"   New vector: {param1.get_directed_vector('custom')}\n")

    # Example 8: Parameter with constraints
    print("8. Parameter with Vector Constraints:")
    param2 = Parameter(
        TestArg("min_val", rng_type=RNGInteger(0, 100)),
        TestArg("max_val", rng_type=RNGInteger(0, 100)),
        vector_constraints=[
            lambda v: v[0] < v[1],  # min < max
        ]
    )
    samples = param2.generate_samples(5, mode="random_only")
    print(f"   Samples (min < max constraint):")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}: min={sample[0]}, max={sample[1]}")
    print()

    # Example 9: Add constraint dynamically
    print("9. Add Constraint Dynamically:")
    param3 = Parameter(
        TestArg("x", rng_type=RNGInteger(0, 10)),
        TestArg("y", rng_type=RNGInteger(0, 10)),
    )
    param3.add_constraint(lambda v: v[0] + v[1] <= 10)  # x + y <= 10
    samples = param3.generate_samples(5, mode="random_only")
    print(f"   Samples (x + y <= 10 constraint):")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}: x={sample[0]}, y={sample[1]}, sum={sample[0]+sample[1]}")
    print()

    # Example 10: Mixed mode with always_include_directed
    print("10. Mixed Mode - always_include_directed=True:")
    param4 = Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
        directed_vectors={"zero": (0,), "max": (100,)},
        always_include_directed=True
    )
    samples = param4.generate_samples(3, mode="mixed")
    print(f"   Total samples: {len(samples)} (2 directed + 3 random)")
    print(f"   Samples: {samples}\n")

    print("11. Mixed Mode - always_include_directed=False:")
    param5 = Parameter(
        TestArg("value", rng_type=RNGInteger(0, 100)),
        directed_vectors={"zero": (0,), "max": (100,)},
        always_include_directed=False
    )
    samples = param5.generate_samples(3, mode="mixed")
    print(f"   Total samples: {len(samples)} (only 3 random)")
    print(f"   Samples: {samples}")