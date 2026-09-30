"""Run the ``pytest-strategies`` command as ``python -m pytest_strategy``."""

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
