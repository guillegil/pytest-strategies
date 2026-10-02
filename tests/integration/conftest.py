"""Shared fixtures for the integration tests."""

import json

import pytest

# A conftest for a pytester project that writes the parameter values of every
# collected row, in collection order, to values.json, or to values-<worker>.json on
# a pytest-xdist worker. Node IDs name the rows, so runs are compared by their values.
DUMP_VALUES = """
import json

def pytest_collection_modifyitems(session, config, items):
    worker = getattr(config, "workerinput", {}).get("workerid")
    rows = [[item.nodeid, repr(item.callspec.params)] for item in items if hasattr(item, "callspec")]
    name = f"values-{worker}.json" if worker else "values.json"
    (config.rootpath / name).write_text(json.dumps(rows))
"""


class ValuesDump:
    """
    Runs pytest on a pytester project whose conftest includes ``conftest`` and reads
    the values each run collected.

    Each run first deletes the files earlier runs wrote and checks its exit code, so
    a run that stops before it collects (a usage error, a crash) fails the test
    instead of leaving an earlier run's values to be read.
    """

    conftest = DUMP_VALUES

    def __init__(self, pytester: pytest.Pytester) -> None:
        self._pytester = pytester

    def run(self, *args: str, ret: int = 0, subprocess: bool = True) -> pytest.RunResult:
        """
        Run pytest with ``args``, in a subprocess or in-process, and return its result
        once its exit code is ``ret``.
        """
        for path in self._pytester.path.glob("values*.json"):
            path.unlink()
        if subprocess:
            result = self._pytester.runpytest_subprocess(*args)
        else:
            result = self._pytester.runpytest(*args)
        assert result.ret == ret, (
            f"pytest {' '.join(args)} exited with {result.ret}, not {ret}:\n"
            f"{result.stdout.str()}\n{result.stderr.str()}"
        )
        return result

    def read(self, name: str = "values.json") -> list[list[str]]:
        """Return the [node ID, values] of each row the last run collected."""
        path = self._pytester.path / name
        assert path.exists(), f"the last run wrote no {name}"
        rows: list[list[str]] = json.loads(path.read_text())
        return rows

    def collect(self, *args: str, subprocess: bool = True) -> list[list[str]]:
        """Run pytest with ``args`` and return the rows it collected (``read()``)."""
        self.run(*args, subprocess=subprocess)
        return self.read()


@pytest.fixture
def values_dump(pytester: pytest.Pytester) -> ValuesDump:
    """Run pytest on the pytester project and read the values of the rows it collected."""
    return ValuesDump(pytester)
