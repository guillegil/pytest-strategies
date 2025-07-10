# rcg.py

import random

from typing import Callable

import pytest

from .rng import RNG
from .strategies import Strategy

from pytest_report import log

def pytest_addoption(parser):
    parser.addoption(
        "--nsamples",
        action="store",
        type=int,
        default=10,
        help="Number of random samples to generate"
    )

    parser.addoption(
        "--rng-seed",
        type=int,
        default=None,
        help="Number of random samples to generate"
    )



@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    rng_seed = config.getoption("--rng-seed", None)

    Strategy.set_config(config)
    RNG.seed(rng_seed)
    

def pytest_report_header(config):
    # Return empty list to suppress header
    return []

def pytest_collection_finish(session):
    # Suppress collection finish output
    pass

def pytest_report_collectionfinish(config, start_path, items):
    # Return empty string to suppress "collected X items"
    return ""