
# rcg.py

from typing import Callable
import random

import time


class RNGValueError(Exception):
    """Exception raised when an invalid value is provided to RNG operations."""
    pass    

class RNG:
 
    _seed = time.time_ns()

    # ========================================
    #           Private Methods
    # ========================================

    @staticmethod
    def refresh_seed():
        random.seed(RNG._seed)

    # ========================================
    #           Public Methods
    # ========================================
    @staticmethod
    def seed(seed: int | None = None):
        if seed is not None:
            RNG._seed = seed

    @staticmethod
    def get_seed():
        return RNG._seed

    @staticmethod
    def integers(min: int, max: int, predicate: Callable | None = None, *args) -> int:
        
        predicate = predicate or (lambda x, args: True)

        tries: int = 100

        while True:
            x = random.randint(min, max)
            if predicate(x, *args):
                return x
        
            if tries > 0:
                tries -= 1
            else:
                raise RNGValueError("No valid integer found within the given constraints.")

    @staticmethod
    def numbers(min: int | float, max: int | float, predicate: Callable | None = None, *args):
        
        predicate = predicate or (lambda x, args: True)

        tries: int = 100

        while True:
            x = random.uniform(min, max)
            if predicate(x, *args):
                return x
        
            if tries > 0:
                tries -= 1
            else:
                raise RNGValueError("No valid number found within the given constraints.")
    
    @staticmethod
    def choice(items: list):
        if not items:
            raise RNGValueError("The choices items list cannot be empty.")

        return random.choice(items)