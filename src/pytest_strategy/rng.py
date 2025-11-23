# rng.py

from typing import Callable, Type
from enum import Enum
import random
import time


class RNGValueError(Exception):
    """Exception raised when an invalid value is provided to RNG operations."""
    pass


class RNG:
    """Core RNG singleton managing seed and random state"""

    _seed = time.time_ns()
    _max_retries = 100

    # ====
    # Seed Management
    # ====

    @staticmethod
    def seed(seed: int | None = None):
        """Set the random seed and refresh the random state"""
        if seed is not None:
            RNG._seed = seed
            random.seed(RNG._seed)

    @staticmethod
    def get_seed():
        """Get the current seed value"""
        return RNG._seed

    @staticmethod
    def refresh_seed():
        """Refresh the random state with the current seed"""
        random.seed(RNG._seed)

    @staticmethod
    def set_max_retries(retries: int):
        """Set the maximum number of retries for constrained generation"""
        RNG._max_retries = retries

    # ====
    # Internal Helper
    # ====

    @staticmethod
    def _generate_with_constraint(generator: Callable, predicate: Callable | None = None):
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

        raise RNGValueError(
            f"No valid value found after {RNG._max_retries} attempts"
        )

    # ====
    # Basic Generators
    # ====

    @staticmethod
    def integer(min: int = -2**31, max: int = 2**31-1, predicate: Callable | None = None) -> int:
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
        return RNG._generate_with_constraint(
            lambda: random.randint(min, max),
            predicate
        )

    @staticmethod
    def float(min: float = 0.0, max: float = 1.0, predicate: Callable | None = None) -> float:
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
        return RNG._generate_with_constraint(
            lambda: random.uniform(min, max),
            predicate
        )

    @staticmethod
    def boolean(true_probability: float = 0.5) -> bool:
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
        return random.random() < true_probability

    @staticmethod
    def choice(items: list):
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
        return random.choice(items)

    @staticmethod
    def string(
        length: int | None = None,
        min_length: int = 1,
        max_length: int = 20,
        charset: str = "abcdefghijklmnopqrstuvwxyz"
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
            ValueError: If length is negative

        Example:
            RNG.string(length=10)  # Fixed length of 10
            RNG.string(min_length=5, max_length=15)  # Variable length 5-15
            RNG.string(length=8, charset="0123456789")  # Numeric string
            RNG.string(length=6, charset="ABCDEF0123456789")  # Hex string
        """
        if length is not None and length < 0:
            raise ValueError("String length cannot be negative")
        
        if length is None:
            length = random.randint(min_length, max_length)
        return ''.join(random.choice(charset) for _ in range(length))

    # ====
    # Weighted Generators
    # ====

    @staticmethod
    def winteger(
        ranges: dict[tuple[int, int], float],
        predicate: Callable | None = None
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

        # Choose range using random.choices (handles normalization)
        chosen_range = random.choices(range_list, weights=weights, k=1)[0]
        min_val, max_val = chosen_range

        return RNG.integer(min_val, max_val, predicate)

    @staticmethod
    def wfloat(
        ranges: dict[tuple[float, float], float],
        predicate: Callable | None = None
    ) -> float:
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

        chosen_range = random.choices(range_list, weights=weights, k=1)[0]
        min_val, max_val = chosen_range

        return RNG.float(min_val, max_val, predicate)


# ====
# RNG Type Classes
# ====

class RNGType:
    """Base class for all RNG types"""

    def generate(self):
        """Generate a random value based on this type's configuration"""
        raise NotImplementedError

    @property
    def python_type(self):
        """Return the Python type this RNG type generates"""
        raise NotImplementedError


class RNGInteger(RNGType):
    """RNG type for generating integers"""

    def __init__(
        self,
        min: int | None = None,
        max: int | None = None,
        predicate: Callable | None = None
    ):
        self.min = min if min is not None else -2**31
        self.max = max if max is not None else 2**31 - 1
        self.predicate = predicate

    def generate(self):
        return RNG.integer(self.min, self.max, self.predicate)

    @property
    def python_type(self):
        return int


class RNGFloat(RNGType):
    """RNG type for generating floats"""

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        predicate: Callable | None = None
    ):
        self.min = min if min is not None else 0.0
        self.max = max if max is not None else 1.0
        self.predicate = predicate

    def generate(self):
        return RNG.float(self.min, self.max, self.predicate)

    @property
    def python_type(self):
        return float


class RNGBoolean(RNGType):
    """RNG type for generating booleans"""

    def __init__(self, true_probability: float = 0.5):
        self.true_probability = true_probability

    def generate(self):
        return RNG.boolean(self.true_probability)

    @property
    def python_type(self):
        return bool


class RNGChoice(RNGType):
    """RNG type for choosing from a list of options"""

    def __init__(self, choices: list):
        if not choices:
            raise RNGValueError("Choices list cannot be empty")
        self.choices = choices

    def generate(self):
        return RNG.choice(self.choices)

    @property
    def python_type(self):
        return type(self.choices[0]) if self.choices else object


class RNGEnum(RNGType):
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
        enum_class: Type[Enum],
        weights: dict[Enum, float] | None = None,
        predicate: Callable[[Enum], bool] | None = None
    ):
        """
        Initialize RNGEnum.
        
        Args:
            enum_class: The Enum class to generate values from
            weights: Optional dictionary mapping enum members to their weights.
                    If provided, only weighted members will be selected.
                    Weights don't need to sum to 1.0 (they'll be normalized).
            predicate: Optional function to filter valid enum values
            
        Raises:
            RNGValueError: If enum_class is not an Enum, or if weights reference non-existent members
        """
        if not issubclass(enum_class, Enum):
            raise RNGValueError(f"{enum_class} is not an Enum class")
        
        self.enum_class = enum_class
        self.weights = weights
        self.predicate = predicate
        
        # Validate weights if provided
        if weights:
            for member in weights.keys():
                if not isinstance(member, enum_class):
                    raise RNGValueError(
                        f"Weight key {member} is not a member of {enum_class.__name__}"
                    )
    
    def generate(self) -> Enum:
        """
        Generate a random enum value.
        
        Returns:
            Random enum member satisfying constraints
            
        Raises:
            RNGValueError: If no valid value found after max_retries attempts
        """
        if self.weights:
            # Weighted selection
            members = list(self.weights.keys())
            weights = list(self.weights.values())
            
            if self.predicate:
                # With predicate: use retry logic
                def generator():
                    return random.choices(members, weights=weights, k=1)[0]
                return RNG._generate_with_constraint(generator, self.predicate)
            else:
                # Without predicate: direct selection
                return random.choices(members, weights=weights, k=1)[0]
        else:
            # Uniform selection from all members
            members = list(self.enum_class)
            
            if self.predicate:
                # With predicate: use retry logic
                def generator():
                    return random.choice(members)
                return RNG._generate_with_constraint(generator, self.predicate)
            else:
                # Without predicate: direct selection
                return random.choice(members)
    
    @property
    def python_type(self):
        """Return the Enum class type"""
        return self.enum_class


class RNGString(RNGType):
    """RNG type for generating strings"""

    def __init__(
        self,
        length: int | None = None,
        min_length: int = 1,
        max_length: int = 20,
        charset: str = "abcdefghijklmnopqrstuvwxyz"
    ):
        self.length = length
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset

    def generate(self):
        return RNG.string(self.length, self.min_length, self.max_length, self.charset)

    @property
    def python_type(self):
        return str


class RNGWeightedInteger(RNGType):
    """RNG type for generating weighted integers from multiple ranges"""

    def __init__(
        self,
        ranges: dict[tuple[int, int], float],
        predicate: Callable | None = None
    ):
        self.ranges = ranges
        self.predicate = predicate

    def generate(self):
        return RNG.winteger(self.ranges, self.predicate)

    @property
    def python_type(self):
        return int


class RNGWeightedFloat(RNGType):
    """RNG type for generating weighted floats from multiple ranges"""

    def __init__(
        self,
        ranges: dict[tuple[float, float], float],
        predicate: Callable | None = None
    ):
        self.ranges = ranges
        self.predicate = predicate

    def generate(self):
        return RNG.wfloat(self.ranges, self.predicate)

    @property
    def python_type(self):
        return float


# ====
# Example Usage
# ====

if __name__ == "__main__":
    from enum import Enum
    
    # Define example enums
    class Status(Enum):
        PENDING = "pending"
        SUCCESS = "success"
        FAILED = "failed"
        ERROR = "error"
    
    class Priority(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4
    
    # Set seed for reproducibility
    RNG.seed(42)

    print("=== Basic Generators ===")
    print(f"Integer (1-100): {RNG.integer(1, 100)}")
    print(f"Even integer (1-100): {RNG.integer(1, 100, predicate=lambda x: x % 2 == 0)}")
    print(f"Float (0-10): {RNG.float(0.0, 10.0)}")
    print(f"Boolean: {RNG.boolean()}")
    print(f"Choice: {RNG.choice(['apple', 'banana', 'cherry'])}")
    print(f"String (length 10): {RNG.string(length=10)}")

    print("\n=== Weighted Generators ===")
    print(f"Weighted integer (80% 0-20, 20% 21-100): {RNG.winteger({(0, 20): 0.8, (21, 100): 0.2})}")
    print(f"Weighted float (70% 0-1, 30% 1-10): {RNG.wfloat({(0.0, 1.0): 0.7, (1.0, 10.0): 0.3})}")

    print("\n=== RNG Type Classes ===")
    int_type = RNGInteger(min=1, max=100)
    print(f"RNGInteger: {int_type.generate()}, type: {int_type.python_type}")

    weighted_type = RNGWeightedInteger(ranges={(0, 20): 0.8, (21, 100): 0.2})
    print(f"RNGWeightedInteger: {weighted_type.generate()}, type: {weighted_type.python_type}")

    choice_type = RNGChoice(choices=['fast', 'slow', 'medium'])
    print(f"RNGChoice: {choice_type.generate()}, type: {choice_type.python_type}")
    
    print("\n=== RNGEnum Examples ===")
    # Simple enum selection
    enum_type = RNGEnum(Status)
    print(f"RNGEnum (uniform): {enum_type.generate()}")
    
    # Weighted enum selection
    weighted_enum = RNGEnum(Status, weights={
        Status.SUCCESS: 0.7,
        Status.PENDING: 0.2,
        Status.FAILED: 0.1
    })
    print(f"RNGEnum (weighted): {weighted_enum.generate()}")
    
    # Enum with predicate
    filtered_enum = RNGEnum(Status, predicate=lambda s: s != Status.ERROR)
    print(f"RNGEnum (filtered): {filtered_enum.generate()}")
    
    # Weighted with predicate
    priority_enum = RNGEnum(
        Priority,
        weights={Priority.HIGH: 0.6, Priority.MEDIUM: 0.3, Priority.LOW: 0.1},
        predicate=lambda p: p != Priority.CRITICAL
    )
    print(f"RNGEnum (weighted + filtered): {priority_enum.generate()}")
    
    # Generate multiple samples
    print("\nMultiple samples:")
    for i in range(5):
        print(f"  Sample {i+1}: {weighted_enum.generate()}")