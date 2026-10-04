"""
End-to-end tests for the name of a factory that a test passes by object
(``@strategy(burst)``), which keys the test's random streams.

The name is one the factory is registered under in the test's folder or above,
by a file that every run collecting the test loads: the factory's own file, the
test module, or a strategy file or ``conftest.py`` of those folders. Otherwise it
is the factory's qualified name. So the full run, the printed rerun command,
``--lf`` and a run of the test's folder alone give a row the same values, and a
registration in another folder does not change them.

Each project's conftest.py writes the strategy name and the values of every
collected row to ``rows.json`` in the rootdir. The printed rerun command is run
through the platform's shell (``sh`` on POSIX, ``cmd.exe`` on Windows) from the
rootdir, with this Python's pytest in place of ``pytest``.
"""

import json
import os
import re
import subprocess
import sys
from textwrap import dedent

import pytest

from pytest_strategy._repro import quote

pytest_plugins = ["pytester"]

SEED = 1

CONFTEST = """
import json

def pytest_collection_modifyitems(config, items):
    from pytest_strategy import VECTOR_KEY

    rows = {
        item.nodeid: [item.stash[VECTOR_KEY].strategy, repr(item.callspec.params)]
        for item in items
        if VECTOR_KEY in item.stash
    }
    (config.rootpath / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
"""

IMPORTS = "from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy\n"


def factory(name, arg="x"):
    """The source of a factory function ``name`` with one random argument."""
    return (
        f"def {name}():\n"
        f"    return Parameter(TestArg({arg!r}, rng_type=RNGInteger(0, 10**6)), nsamples=4)\n"
    )


def module(name, ref, arg="x", fails=None):
    """
    The test ``name`` of a test module, which uses ``@strategy(ref)`` and fails on
    the row named ``fails``, showing its value.
    """
    check = f"assert request.node.name != {fails!r}, f'{arg}={{{arg}}}'" if fails else "pass"
    return f"@strategy({ref})\ndef {name}(request, {arg}):\n    {check}\n"


def project(pytester, files):
    """The conftest.py, an ini file that puts the top and tests/ on sys.path, and ``files``."""
    pytester.makeini("[pytest]\npythonpath = . tests\n")
    pytester.makeconftest(CONFTEST)
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text), encoding="utf-8")


def run(pytester, *args, ret=0, cache=False, seed=True):
    """
    Run pytest with ``args``, and the seed unless ``seed`` is false, and return its
    result and the rows it collected, {node ID: [strategy, values]}.
    """
    path = pytester.path / "rows.json"
    path.unlink(missing_ok=True)
    options = (() if cache else ("-p", "no:cacheprovider")) + (
        (f"--rng-seed={SEED}",) if seed else ()
    )
    result = pytester.runpytest_subprocess(*options, *args)
    assert result.ret == ret, result.stdout.str()
    found: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return result, found


def rows(pytester, *args):
    """Run pytest with ``args`` and the seed, and return the rows it collected."""
    return run(pytester, *args)[1]


def names(found, test):
    """The strategy names of the rows of the test function ``test``."""
    return {name for nodeid, (name, _) in found.items() if nodeid.split("::")[1].startswith(test)}


def same_rows(part, full):
    """Whether every row of ``part`` has the strategy name and values it has in ``full``."""
    return bool(part) and all(full[nodeid] == row for nodeid, row in part.items())


def sections(lines):
    """The pytest-strategies sections in the output, each one's lines up to its rerun line."""
    found = []
    current = None
    for line in lines:
        if re.fullmatch(r"=+ short test summary info =+", line):
            break
        if re.fullmatch(r"-+ pytest-strategies -+", line):
            current = []
            found.append(current)
        elif current is not None:
            current.append(line)
            if line.startswith("rerun "):
                current = None
    return found


def errors(lines):
    """The ``E`` lines of the output, up to the short test summary."""
    found = []
    for line in lines:
        if re.fullmatch(r"=+ short test summary info =+", line):
            break
        if line.startswith("E "):
            found.append(line)
    return found


def shell(command, cwd):
    """
    Run a printed rerun command through the platform's shell from ``cwd``, with this
    Python's pytest in place of ``pytest``, and return its exit code and output lines.
    """
    assert command.startswith("pytest ")
    line = f"{quote(sys.executable)} -m pytest -p no:cacheprovider {command[len('pytest ') :]}"
    done = subprocess.run(
        line,
        shell=True,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
    )
    return done.returncode, done.stdout.splitlines()


class TestRegisteredElsewhere:
    """The reviewer's layout: a helper factory that only a sibling folder registers."""

    FILES = {
        "helpers/__init__.py": "",
        "helpers/factories.py": IMPORTS + factory("burst"),
        # A plain call registers the helper as "dma"; the decorated factory makes
        # the plugin import the file
        "tests/a/strategies.py": (
            IMPORTS
            + "from helpers.factories import burst\n\n"
            + 'register("dma")(burst)\n\n'
            + '@register("other")\n'
            + factory("other", "y")
        ),
        "tests/a/test_a.py": IMPORTS + module("test_a", '"dma"'),
        "tests/b/test_b.py": (
            IMPORTS
            + "from helpers.factories import burst\n\n"
            + module("test_b", "burst", fails="test_b[rand-2]")
        ),
    }
    NODEID = "tests/b/test_b.py::test_b[rand-2]"

    def test_the_rerun_command_lf_and_the_folder_give_the_values_of_the_full_run(self, pytester):
        project(pytester, self.FILES)

        result, full = run(pytester, ret=1, cache=True)
        (section,) = sections(result.stdout.lines)
        failure = errors(result.stdout.lines)
        (rerun,) = [line[len("rerun") :].strip() for line in section if line.startswith("rerun ")]
        (pytester.path / "rows.json").unlink()
        code, lines = shell(rerun, pytester.path)
        rerun_rows = json.loads((pytester.path / "rows.json").read_text(encoding="utf-8"))
        lf_result, lf = run(pytester, "--lf", ret=1, cache=True, seed=False)
        _, folder = run(pytester, "tests/b", ret=1)

        # The helper's qualified name: the registration as "dma" is in helpers/,
        # not in the test's folder or above, and only the runs that load
        # tests/a/strategies.py make it
        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_a") == {"dma"}
        assert section[0].startswith("strategy  burst (helpers/factories.py:")
        assert code == 1, "\n".join(lines)
        assert sections(lines) == [section]
        assert errors(lines) == failure != []
        assert list(rerun_rows) == [self.NODEID]
        assert same_rows(rerun_rows, full)
        lf_result.stdout.fnmatch_lines(["pytest-strategies: seed reused from the failed run*"])
        assert errors(lf_result.stdout.lines) == failure
        assert self.NODEID in lf
        assert same_rows(lf, full)
        assert len(folder) == 4
        assert same_rows(folder, full)


class TestNames:
    """Which name a factory passed by object gets, the same in a full run and a part."""

    def test_the_nearest_of_two_names_in_two_ancestor_folders_wins(self, pytester):
        # The factory is in tests/: both names are visible from tests/a and
        # tests/b. tests/b's strategy file registers it again as "z_alias", so
        # tests/b's test gets that name, before "a_base" in alphabetical order;
        # tests/a's chain does not load that file
        project(
            pytester,
            {
                "tests/common_strategies.py": IMPORTS + '@register("a_base")\n' + factory("burst"),
                "tests/b/b_strategies.py": (
                    IMPORTS
                    + "from common_strategies import burst\n\n"
                    + 'register("z_alias")(burst)\n\n'
                    + '@register("b_other")\n'
                    + factory("other", "y")
                ),
                "tests/a/test_a.py": (
                    IMPORTS + "from common_strategies import burst\n\n" + module("test_a", "burst")
                ),
                "tests/b/test_b.py": (
                    IMPORTS + "from common_strategies import burst\n\n" + module("test_b", "burst")
                ),
            },
        )

        full = rows(pytester)
        folder_a = rows(pytester, "tests/a")
        folder_b = rows(pytester, "tests/b")
        alone = rows(pytester, "tests/b/test_b.py::test_b[rand-1]")

        assert names(full, "test_a") == {"a_base"}
        assert names(full, "test_b") == {"z_alias"}
        assert same_rows(folder_a, full)
        assert same_rows(folder_b, full)
        assert same_rows(alone, full)

    def test_a_helper_registered_by_two_ancestor_folders_keeps_its_qualified_name(self, pytester):
        # The skeptic's layout: the registrations are in helpers/, and names_of()
        # followed the import order, "base" in a full run and "b_alias" in tests/b
        def registers(name, other):
            return (
                IMPORTS
                + "from helpers.factories import burst\n\n"
                + f"register({name!r})(burst)\n\n"
                + f"@register({other!r})\n"
                + factory(other, "y")
            )

        project(
            pytester,
            {
                "helpers/__init__.py": "",
                "helpers/factories.py": IMPORTS + factory("burst"),
                "tests/strategies.py": registers("base", "other_root"),
                "tests/b/strategies.py": registers("b_alias", "other_b"),
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": (
                    IMPORTS + "from helpers.factories import burst\n\n" + module("test_b", "burst")
                ),
            },
        )

        full = rows(pytester)
        folder = rows(pytester, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert same_rows(folder, full)

    @pytest.mark.parametrize(
        ("caller", "expected"),
        [
            # Loaded before every run's test_b is collected
            ("tests/conftest.py", "alias"),
            ("tests/b/test_b.py", "alias"),
            # Loaded only by the runs that collect it (test_0.py before test_b.py)
            ("tests/b/test_0.py", "base"),
            ("tests/a/a_strategies.py", "base"),
        ],
    )
    def test_an_alias_names_it_when_every_run_registers_it(self, pytester, caller, expected):
        # "alias" comes before "base" in alphabetical order
        alias = IMPORTS + "from common_strategies import burst\n\n" + 'register("alias")(burst)\n\n'
        files = {
            "tests/common_strategies.py": IMPORTS + '@register("base")\n' + factory("burst"),
            "tests/a/a_strategies.py": IMPORTS + '@register("a_other")\n' + factory("other", "y"),
            "tests/a/test_a.py": IMPORTS + module("test_a", '"a_other"', arg="y"),
            "tests/b/test_0.py": "def test_0():\n    pass\n",
            "tests/b/test_b.py": (
                IMPORTS + "from common_strategies import burst\n\n" + module("test_b", "burst")
            ),
        }
        files[caller] = alias + files.get(caller, "")
        project(pytester, files)

        full = rows(pytester)
        alone = rows(pytester, "tests/b/test_b.py")

        assert names(full, "test_b") == {expected}
        assert same_rows(alone, full)

    def test_a_name_another_factory_has_in_the_test_folder_does_not_name_it(self, pytester):
        # tests/b registers "dma" for another factory: @strategy("dma") there
        # finds that one, so the test in tests/b gets the factory's other name
        project(
            pytester,
            {
                "tests/common_strategies.py": (
                    IMPORTS + '@register("dma")\n@register("zz_burst")\n' + factory("burst")
                ),
                "tests/b/b_strategies.py": IMPORTS + '@register("dma")\n' + factory("other"),
                "tests/test_top.py": (
                    IMPORTS
                    + "from common_strategies import burst\n\n"
                    + module("test_top", "burst")
                ),
                "tests/b/test_b.py": (
                    IMPORTS + "from common_strategies import burst\n\n" + module("test_b", "burst")
                ),
            },
        )

        full = rows(pytester)
        folder = rows(pytester, "tests/b")

        assert names(full, "test_top") == {"dma"}
        assert names(full, "test_b") == {"zz_burst"}
        assert same_rows(folder, full)


class TestSameAsTheName:
    @pytest.mark.parametrize("args", [(), ("tests/b",)], ids=["full", "folder"])
    def test_a_factory_registered_in_the_test_folder_gives_the_values_of_its_name(
        self, pytester, args
    ):
        # Two names in one file: the first in alphabetical order
        files = {
            "tests/b/b_strategies.py": (
                IMPORTS + '@register("zz_burst")\n@register("dma")\n' + factory("burst")
            ),
            "tests/a/test_a.py": "def test_a():\n    pass\n",
        }
        path = pytester.path / "tests" / "b" / "test_b.py"
        by_object = IMPORTS + "from b_strategies import burst\n\n" + module("test_b", "burst")

        project(pytester, {**files, "tests/b/test_b.py": by_object})
        factory_rows = rows(pytester, *args)
        path.write_text(IMPORTS + module("test_b", '"dma"'), encoding="utf-8")
        name_rows = rows(pytester, *args)
        path.write_text(IMPORTS + module("test_b", '"zz_burst"'), encoding="utf-8")
        other_rows = rows(pytester, *args)

        assert names(factory_rows, "test_b") == {"dma"}
        assert factory_rows == name_rows
        assert len(factory_rows) == 4
        assert {nodeid: values for nodeid, (_, values) in other_rows.items()} != {
            nodeid: values for nodeid, (_, values) in factory_rows.items()
        }
