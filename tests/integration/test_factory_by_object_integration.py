"""
End-to-end tests for the name of a factory that a test passes by object
(``@strategy(burst)``), which keys the test's random streams.

The name is one the factory is registered under by a file that every run
collecting the test loads first: the factory's own file, the test module, a
strategy file or ``conftest.py`` of the test's folder or above, or a module that
every run loads as a plugin. It does not depend on where the factory's file is,
in the test's folder, another folder or a package, installed or checked out.
Otherwise it is the factory's qualified name. So the full run, the printed rerun
command, ``--lf`` and a run of the test's folder alone give a row the same
values, and a registration in another folder, another test module or a plugin
that only some runs load does not change them, nor does a copy of the factory
made by importing its file under another module name.

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


def check_reruns(pytester, nodeid, folder, *args):
    """
    Run the project with ``args``, its row ``nodeid`` failing, then the row's
    printed rerun command, ``--lf`` and a run of ``folder``, and check that each
    gives the row the values and the failure of the full run. Return the full
    run's rows, the row's section and the rows of the run of ``folder``.
    """
    result, full = run(pytester, *args, ret=1, cache=True)
    (section,) = sections(result.stdout.lines)
    failure = errors(result.stdout.lines)
    (rerun,) = [line[len("rerun") :].strip() for line in section if line.startswith("rerun ")]
    (pytester.path / "rows.json").unlink()
    code, lines = shell(rerun, pytester.path)
    rerun_rows = json.loads((pytester.path / "rows.json").read_text(encoding="utf-8"))
    lf_result, lf = run(pytester, "--lf", ret=1, cache=True, seed=False)
    _, part = run(pytester, folder, ret=1)

    assert code == 1, "\n".join(lines)
    assert sections(lines) == [section]
    assert errors(lines) == failure != []
    assert list(rerun_rows) == [nodeid]
    assert same_rows(rerun_rows, full)
    lf_result.stdout.fnmatch_lines(["pytest-strategies: seed reused from the failed run*"])
    assert errors(lf_result.stdout.lines) == failure
    assert nodeid in lf
    assert same_rows(lf, full)
    assert nodeid in part
    assert same_rows(part, full)
    return full, section, part


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

        full, section, folder = check_reruns(pytester, self.NODEID, "tests/b")

        # The helper's qualified name: only the runs that load
        # tests/a/strategies.py register it as "dma"
        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_a") == {"dma"}
        assert section[0].startswith("strategy  burst (helpers/factories.py:")
        assert len(folder) == 4


class TestRegisteredInSomeRuns:
    """Registrations that only the runs importing another test module make."""

    @pytest.mark.parametrize(
        ("files", "nodeid", "folder", "name"),
        [
            pytest.param(
                {
                    "tests/strategies.py": IMPORTS + '@register("burst")\n' + factory("f_burst"),
                    # A local override of "burst" in a test module of the test's
                    # folder, collected before test_x.py: @strategy("burst") in
                    # test_x.py finds it only in the runs that collect it
                    "tests/b/test_a_other.py": (
                        IMPORTS
                        + '@register("burst")\n'
                        + factory("local_burst", "y")
                        + module("test_other", '"burst"', arg="y")
                    ),
                    "tests/b/test_x.py": (
                        IMPORTS
                        + "from strategies import f_burst\n\n"
                        + module("test_x", "f_burst", fails="test_x[rand-1]")
                    ),
                },
                "tests/b/test_x.py::test_x[rand-1]",
                "tests/b/test_x.py",
                "burst",
                id="shadowed_by_a_test_module",
            ),
            pytest.param(
                {
                    "tests/common.py": (
                        IMPORTS
                        + '@register("zbase")\n'
                        + factory("f_burst")
                        + "\n\ndef register_as(name):\n    return register(name)(f_burst)\n"
                    ),
                    # The helper in common.py registers the factory as "alpha"
                    # (before "zbase" in alphabetical order) when test_a.py is
                    # imported: the call counts for test_a.py, not for common.py
                    "tests/a/test_a.py": (
                        "from common import register_as\n\n"
                        + 'register_as("alpha")\n\n\n'
                        + "def test_a():\n    pass\n"
                    ),
                    "tests/b/test_b.py": (
                        IMPORTS
                        + "from common import f_burst\n\n"
                        + module("test_b", "f_burst", fails="test_b[rand-1]")
                    ),
                },
                "tests/b/test_b.py::test_b[rand-1]",
                "tests/b",
                "zbase",
                id="registered_by_a_helper",
            ),
        ],
    )
    def test_they_do_not_change_its_name(self, pytester, files, nodeid, folder, name):
        project(pytester, files)

        full, section, _ = check_reruns(pytester, nodeid, folder)

        test = nodeid.split("::")[1].split("[")[0]
        assert names(full, test) == {name}
        assert section[0].startswith(f"strategy  {name} (tests/")


class TestAliasedBySiblingFolder:
    """A helper factory on the test's path that only a sibling folder registers."""

    @pytest.mark.parametrize(
        ("caller", "ini", "files"),
        [
            # Loaded with tests/a's tests: only the runs that collect them load it
            ("tests/a/strategies.py", "", {}),
            # pytest loads it for tests/a's tests, not for tests/b's
            ("tests/a/conftest.py", "", {}),
            # An initial conftest of a run without arguments, loaded before the
            # collection, but not by a run of tests/b or of a node ID
            ("tests/a/conftest.py", "testpaths = tests/a tests/b\n", {}),
            # A plugin module that such a conftest's pytest_plugins loads
            (
                "tests/plugin_mod.py",
                "testpaths = tests/a tests/b\n",
                {"tests/a/conftest.py": 'pytest_plugins = ["plugin_mod"]\n'},
            ),
        ],
        ids=["strategy_file", "conftest", "conftest_in_testpaths", "plugin_of_that_conftest"],
    )
    def test_it_keeps_its_qualified_name(self, pytester, caller, ini, files):
        # tests/common.py is in tests/, above tests/b, but registers nothing: the
        # alias as "dma" is made by a file that only the runs collecting tests/a
        # load
        nodeid = "tests/b/test_b.py::test_b[rand-1]"
        project(
            pytester,
            {
                "tests/common.py": IMPORTS + factory("burst"),
                caller: (
                    IMPORTS
                    + "from common import burst\n\n"
                    + 'register("dma")(burst)\n\n'
                    + '@register("a_other")\n'
                    + factory("a_other", "y")
                ),
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma"'),
                "tests/b/test_b.py": (
                    IMPORTS
                    + "from common import burst\n\n"
                    + module("test_b", "burst", fails="test_b[rand-1]")
                ),
                **files,
            },
        )
        pytester.makeini(f"[pytest]\npythonpath = . tests\n{ini}")

        full, section, _ = check_reruns(pytester, nodeid, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_a") == {"dma"}
        assert section[0].startswith("strategy  burst (tests/common.py:")


class TestImportedUnderTwoNames:
    """A factory's file that test modules import under two module names."""

    def test_the_copy_that_replaced_its_registration_still_names_it(self, pytester):
        # With tests/ and the top on sys.path, tests/factories.py is the module
        # tests.factories for test_a.py and test_d.py, and factories for
        # test_c.py. The second import runs the file again: its register() call
        # replaces the registration of the copy test_d.py holds, in the runs that
        # collect test_c.py only
        nodeid = "tests/b/test_d.py::test_d[rand-1]"
        project(
            pytester,
            {
                "tests/factories.py": IMPORTS + '@register("dma")\n' + factory("burst"),
                "tests/a/test_a.py": (
                    "from tests.factories import burst\n\n\ndef test_a():\n    pass\n"
                ),
                "tests/a/test_c.py": "from factories import burst\n\n\ndef test_c():\n    pass\n",
                "tests/b/test_d.py": (
                    IMPORTS
                    + "from tests.factories import burst\n\n"
                    + module("test_d", "burst", fails="test_d[rand-1]")
                ),
            },
        )

        full, section, _ = check_reruns(pytester, nodeid, "tests/b")

        assert names(full, "test_d") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/factories.py:")

    def test_a_copy_made_after_an_alias_keeps_the_alias(self, pytester):
        # tests/b/b_strategies.py registers burst again as "a_base", which its
        # own file registers, and as "z_alias": both calls are in tests/b, so
        # "a_base" comes first. test_c.py, collected after the strategy file,
        # imports tests/factories.py as factories: the copy's register() call
        # replaces "a_base" in the runs that collect test_c.py, and still counts
        # tests/b/b_strategies.py's call
        project(
            pytester,
            {
                "tests/__init__.py": "",
                "tests/factories.py": IMPORTS + '@register("a_base")\n' + factory("burst"),
                "tests/b/b_strategies.py": (
                    IMPORTS
                    + "from tests.factories import burst\n\n"
                    + 'register("a_base")(burst)\n'
                    + 'register("z_alias")(burst)\n\n'
                    + '@register("b_other")\n'
                    + factory("other", "y")
                ),
                "tests/b/test_c.py": "from factories import burst\n\n\ndef test_c():\n    pass\n",
                "tests/b/test_d.py": (
                    IMPORTS + "from tests.factories import burst\n\n" + module("test_d", "burst")
                ),
            },
        )

        full = rows(pytester)
        alone = rows(pytester, "tests/b/test_d.py")

        assert names(full, "test_d") == {"a_base"}
        assert same_rows(alone, full)


class TestInstalled:
    """
    A factory of a package, installed or checked out, or of a plugin, as
    @strategy("dma_burst") finds it.
    """

    def package(self, site, caller):
        """The package ps_kit in ``site``, whose ``caller`` registers burst as "dma_burst"."""
        registers = 'register("dma_burst")(burst)\n'
        files = {
            "__init__.py": "",
            "factories.py": IMPORTS + factory("burst"),
            "plugin.py": IMPORTS + "from ps_kit.factories import burst\n\n",
        }
        if caller == "factories.py":
            files["factories.py"] += "\n" + registers
        else:
            files[caller] += registers
        for name, text in files.items():
            path = site / "ps_kit" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def rows(self, pytester, root, *args):
        """Run pytest in ``root`` and return the rows it collected."""
        (root / "rows.json").unlink(missing_ok=True)
        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", f"--rng-seed={SEED}", *args
        )
        assert result.ret == 0, result.stdout.str()
        found: dict[str, list[str]] = json.loads((root / "rows.json").read_text(encoding="utf-8"))
        return found

    @pytest.mark.parametrize("where", ["outside", "venv", "src"])
    @pytest.mark.parametrize("caller", ["factories.py", "plugin.py"])
    def test_it_gets_the_values_of_its_name(self, pytester, monkeypatch, where, caller):
        # The project is in proj/: lib/ is outside its rootdir, and a virtualenv's
        # site-packages and a src/ layout's folder are inside it. -p loads the
        # plugin module in every run.
        root = pytester.path / "proj"
        site = {
            "outside": pytester.path / "lib",
            "venv": root / ".venv" / "lib" / "site-packages",
            "src": root / "src",
        }[where]
        self.package(site, caller)
        (root / "tests").mkdir(parents=True)
        (root / "pytest.ini").write_text(
            f"[pytest]\naddopts = -p ps_kit.plugin\npythonpath = {site.as_posix()}\n",
            encoding="utf-8",
        )
        (root / "conftest.py").write_text(dedent(CONFTEST), encoding="utf-8")
        path = root / "tests" / "test_x.py"
        monkeypatch.chdir(root)

        path.write_text(
            IMPORTS + "from ps_kit.factories import burst\n\n" + module("test_x", "burst"),
            encoding="utf-8",
        )
        factory_rows = self.rows(pytester, root)
        path.write_text(IMPORTS + module("test_x", '"dma_burst"'), encoding="utf-8")
        name_rows = self.rows(pytester, root)

        assert names(factory_rows, "test_x") == {"dma_burst"}
        assert factory_rows == name_rows
        assert len(factory_rows) == 4

    @pytest.mark.parametrize("caller", ["factories.py", "plugin.py"])
    def test_checked_out_and_installed_it_gets_the_same_values(self, pytester, caller):
        # One project with the package in src/ (a checkout, or an editable
        # install) and a copy in a virtualenv's site-packages (pip install .): a
        # CI job that installs the package and a developer's checkout give the
        # test the same name and values
        src = pytester.path / "src"
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(src, caller)
        self.package(site, caller)
        project(
            pytester,
            {
                "tests/test_x.py": (
                    IMPORTS + "from ps_kit.factories import burst\n\n" + module("test_x", "burst")
                ),
            },
        )
        found = []
        for folder in (src, site):
            pytester.makeini(
                f"[pytest]\naddopts = -p ps_kit.plugin\npythonpath = . tests {folder.as_posix()}\n"
            )
            found.append(rows(pytester))

        assert names(found[0], "test_x") == {"dma_burst"}
        assert found[0] == found[1]
        assert len(found[0]) == 4

    def test_a_name_two_factories_have_elsewhere_does_not_name_it(self, pytester):
        # Every run loads ps_kit's plugin, which registers burst as "dma_burst",
        # then another package's plugin, which registers its own factory under
        # that name: @strategy("dma_burst") finds two, so burst keeps its
        # qualified name
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        other = site / "other_kit"
        other.mkdir()
        (other / "__init__.py").write_text("", encoding="utf-8")
        (other / "plugin.py").write_text(
            IMPORTS + '@register("dma_burst")\n' + factory("other"), encoding="utf-8"
        )
        project(
            pytester,
            {
                "tests/b/test_b.py": (
                    IMPORTS + "from ps_kit.factories import burst\n\n" + module("test_b", "burst")
                ),
            },
        )
        pytester.makeini(
            "[pytest]\naddopts = -p ps_kit.plugin -p other_kit.plugin\n"
            f"pythonpath = . tests {site.as_posix()}\n"
        )

        full = rows(pytester)
        folder = rows(pytester, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert same_rows(folder, full)

    def test_a_plugin_that_a_sibling_folders_conftest_loads_does_not_name_it(self, pytester):
        # tests/a/conftest.py is an initial conftest of a run without arguments
        # (testpaths), whose pytest_plugins loads the plugin that registers burst
        # as "dma_burst". A run of tests/b, or of a node ID there, does not load it
        nodeid = "tests/b/test_b.py::test_b[rand-1]"
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        project(
            pytester,
            {
                "tests/a/conftest.py": 'pytest_plugins = ["ps_kit.plugin"]\n',
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma_burst"'),
                "tests/b/test_b.py": (
                    IMPORTS
                    + "from ps_kit.factories import burst\n\n"
                    + module("test_b", "burst", fails="test_b[rand-1]")
                ),
            },
        )
        pytester.makeini(
            f"[pytest]\npythonpath = . tests {site.as_posix()}\ntestpaths = tests/a tests/b\n"
        )

        full, section, _ = check_reruns(pytester, nodeid, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_a") == {"dma_burst"}
        assert section[0].startswith("strategy  burst (")

    @pytest.mark.parametrize("args", [(), ("-p", "ps_kit")], ids=["autoloaded", "given"])
    def test_a_plugin_of_an_entry_point_names_it(self, pytester, args):
        # ps_kit.plugin is the pytest11 entry point "ps_kit" of an installed
        # distribution: pytest loads it in every run, so giving it with -p on the
        # command line, which the rerun command does not carry, changes nothing
        nodeid = "tests/b/test_b.py::test_b[rand-1]"
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        info = site / "ps_kit-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: ps-kit\nVersion: 1.0\n", encoding="utf-8"
        )
        (info / "entry_points.txt").write_text(
            "[pytest11]\nps_kit = ps_kit.plugin\n", encoding="utf-8"
        )
        project(
            pytester,
            {
                "tests/b/test_b.py": (
                    IMPORTS
                    + "from ps_kit.factories import burst\n\n"
                    + module("test_b", "burst", fails="test_b[rand-1]")
                ),
            },
        )
        pytester.makeini(f"[pytest]\npythonpath = . tests {site.as_posix()}\n")

        full, section, _ = check_reruns(pytester, nodeid, "tests/b", *args)

        assert names(full, "test_b") == {"dma_burst"}
        assert section[0].startswith("strategy  dma_burst (")

    def test_a_name_another_factory_has_in_the_test_folder_does_not_name_it(self, pytester):
        # tests/b registers "dma_burst" for another factory: @strategy("dma_burst")
        # there finds that one, so the package's factory keeps its qualified name
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "factories.py")
        project(
            pytester,
            {
                "tests/b/b_strategies.py": IMPORTS + '@register("dma_burst")\n' + factory("local"),
                "tests/b/test_b.py": (
                    IMPORTS
                    + "from ps_kit.factories import burst\n\n"
                    + module("test_b", "burst")
                    + "\n\n"
                    + module("test_local", '"dma_burst"')
                ),
            },
        )
        pytester.makeini(f"[pytest]\npythonpath = . tests {site.as_posix()}\n")

        full = rows(pytester)
        folder = rows(pytester, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_local") == {"dma_burst"}
        assert same_rows(folder, full)

    def test_a_plugin_that_a_test_module_loads_does_not_name_it(self, pytester):
        # pytest loads a test module's pytest_plugins when it collects the module:
        # only the runs that collect test_a.py register burst as "dma_burst"
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        project(
            pytester,
            {
                "tests/test_a.py": 'pytest_plugins = ["ps_kit.plugin"]\n\n\ndef test_a():\n    pass\n',
                "tests/test_b.py": (
                    IMPORTS + "from ps_kit.factories import burst\n\n" + module("test_b", "burst")
                ),
            },
        )
        pytester.makeini(f"[pytest]\npythonpath = . tests {site.as_posix()}\n")

        full = rows(pytester)
        alone = rows(pytester, "tests/test_b.py")

        assert names(full, "test_b") == {"burst"}
        assert same_rows(alone, full)


class TestPlugins:
    """Registrations made by a plugin module or a hook, inside the rootdir."""

    def project(self, pytester, folder="tests", files=None, conftest=""):
        """
        The factory burst in ``folder``/common.py, registered as "dma" by
        tests/plugin_mod.py, a test in tests/b that passes it and fails on rand-1,
        ``files``, and ``conftest`` added to the root conftest.py.
        """
        package = "helpers." if folder == "helpers" else ""
        project(
            pytester,
            {
                f"{folder}/common.py": IMPORTS + factory("burst"),
                "tests/plugin_mod.py": (
                    IMPORTS + f"from {package}common import burst\n\n" + 'register("dma")(burst)\n'
                ),
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": (
                    IMPORTS
                    + f"from {package}common import burst\n\n"
                    + module("test_b", "burst", fails="test_b[rand-1]")
                ),
                **({"helpers/__init__.py": ""} if package else {}),
                **(files or {}),
            },
        )
        if conftest:
            pytester.makeconftest(CONFTEST + conftest)

    CHAIN = {"tests/plugin_chain.py": 'pytest_plugins = ["plugin_mod"]\n'}

    @pytest.mark.parametrize(
        ("folder", "ini", "files", "conftest", "addopts", "args"),
        [
            # On the test's path, and in helpers/, which is not
            pytest.param("tests", "addopts = -p plugin_mod\n", {}, "", "", (), id="addopts"),
            pytest.param(
                "helpers", "addopts = -p plugin_mod\n", {}, "", "", (), id="addopts_off_path"
            ),
            # pytest_plugins of a conftest.py in the test's folder, an initial one
            # in every run that collects the test
            pytest.param(
                "tests",
                "testpaths = tests/a tests/b\n",
                {"tests/b/conftest.py": 'pytest_plugins = ["plugin_mod"]\n'},
                "",
                "",
                (),
                id="conftest_on_path",
            ),
            # The root conftest's plugin loads it with its own pytest_plugins
            pytest.param(
                "tests", "", CHAIN, 'pytest_plugins = ["plugin_chain"]\n', "", (), id="chain"
            ),
            # Given on the command line too: addopts or PYTEST_ADDOPTS loads it in
            # every run, or a plugin that addopts loads
            pytest.param(
                "tests", "addopts = -pplugin_mod\n", {}, "", "", ("-p", "plugin_mod"), id="given"
            ),
            pytest.param(
                "tests", "", {}, "", "-p plugin_mod", ("-p", "plugin_mod"), id="given_env"
            ),
            pytest.param(
                "tests",
                "addopts = -p plugin_chain\n",
                CHAIN,
                "",
                "",
                ("-p", "plugin_mod"),
                id="given_chain",
            ),
        ],
    )
    def test_a_plugin_that_every_run_loads_names_it(
        self, pytester, monkeypatch, folder, ini, files, conftest, addopts, args
    ):
        nodeid = "tests/b/test_b.py::test_b[rand-1]"
        self.project(pytester, folder, files, conftest)
        pytester.makeini(f"[pytest]\npythonpath = . tests\n{ini}")
        if addopts:
            # Also for the rerun command, run through the shell
            monkeypatch.setenv("PYTEST_ADDOPTS", addopts)

        full, section, _ = check_reruns(pytester, nodeid, "tests/b", *args)

        assert names(full, "test_b") == {"dma"}
        assert section[0].startswith(f"strategy  dma ({folder}/common.py:")

    def test_a_plugin_given_on_the_command_line_only_does_not_name_it(self, pytester):
        # The printed rerun command does not carry -p: the run that gives it gets
        # the name of the runs that do not, also for the module that the given
        # plugin's pytest_plugins loads
        nodeid = "tests/b/test_b.py::test_b[rand-1]"
        self.project(pytester, files=self.CHAIN)

        full, section, _ = check_reruns(pytester, nodeid, "tests/b", "-p", "plugin_chain")

        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith("strategy  burst (tests/common.py:")

    def test_a_registration_in_a_hook_does_not_name_it(self, pytester):
        # The root conftest's pytest_configure registers burst as "dma": the call
        # is made by no module's import, so it counts for none of the test's files
        hook = (
            "\n\ndef pytest_configure(config):\n"
            "    from common import burst\n"
            "    from pytest_strategy import register\n\n"
            '    register("dma")(burst)\n'
        )
        self.project(
            pytester,
            files={"tests/b/test_name.py": IMPORTS + module("test_name", '"dma"')},
            conftest=hook,
        )
        (pytester.path / "tests" / "plugin_mod.py").unlink()

        full = run(pytester, ret=1)[1]
        folder = run(pytester, "tests/b", ret=1)[1]

        assert names(full, "test_b") == {"burst"}
        assert names(full, "test_name") == {"dma"}
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

    def test_a_helper_registered_by_two_ancestor_folders_gets_the_nearest_name(self, pytester):
        # The skeptic's layout: names_of() followed the import order, "base" in a
        # full run and "b_alias" in tests/b. Both strategy files are loaded before
        # tests/b's tests in every run: tests/b's is the deeper one
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
                "tests/a/test_a.py": (
                    IMPORTS + "from helpers.factories import burst\n\n" + module("test_a", "burst")
                ),
                "tests/b/test_b.py": (
                    IMPORTS + "from helpers.factories import burst\n\n" + module("test_b", "burst")
                ),
            },
        )

        full = rows(pytester)
        folder_a = rows(pytester, "tests/a")
        folder_b = rows(pytester, "tests/b")

        assert names(full, "test_a") == {"base"}
        assert names(full, "test_b") == {"b_alias"}
        assert same_rows(folder_a, full)
        assert same_rows(folder_b, full)

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

    def test_a_factory_imported_from_another_folders_strategy_file_gets_its_name(self, pytester):
        # Every run that collects test_d.py imports c_strategies.py, which
        # registers the factory as "c_burst" (the run that collects tests/c
        # loads it first), and the test module registers it as "d_alias", in
        # the test's folder: the deeper call wins. In tests/e the name is
        # "c_burst", the only registration elsewhere, which @strategy("c_burst")
        # there finds too
        project(
            pytester,
            {
                "tests/c/c_strategies.py": IMPORTS + '@register("c_burst")\n' + factory("cburst"),
                "tests/c/test_c.py": IMPORTS + module("test_c", '"c_burst"'),
                "tests/d/test_d.py": (
                    IMPORTS
                    + "from c_strategies import cburst\n\n"
                    + 'register("d_alias")(cburst)\n\n'
                    + module("test_d", "cburst")
                ),
                "tests/e/test_e.py": (
                    IMPORTS + "from c_strategies import cburst\n\n" + module("test_e", "cburst")
                ),
            },
        )
        pytester.makeini("[pytest]\npythonpath = . tests tests/c\n")

        full = rows(pytester)
        folder_d = rows(pytester, "tests/d")
        folder_e = rows(pytester, "tests/e")

        assert names(full, "test_d") == {"d_alias"}
        assert names(full, "test_e") == {"c_burst"}
        assert same_rows(folder_d, full)
        assert same_rows(folder_e, full)

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
