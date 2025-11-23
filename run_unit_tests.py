#!/usr/bin/env python
"""
Simple test runner to validate unit tests without pytest plugin issues.
"""
import sys
sys.path.insert(0, 'src')

# Import test modules
from tests.unittests import test_test_args
from tests.unittests import test_parameters

# Import pytest
import pytest

if __name__ == "__main__":
    # Run tests with minimal plugins
    exit_code = pytest.main([
        'tests/unittests/test_test_args.py',
        'tests/unittests/test_parameters.py',
        '-v',
        '-p', 'no:cacheprovider',
        '--override-ini=addopts=',  # Clear addopts
    ])
    
    sys.exit(exit_code)
