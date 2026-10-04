"""
End-to-end tests for the name of a factory that a test passes by object
(``@strategy(burst)``), which keys the test's random streams.

The name is one that the factory's own file, the module that defines it,
registers it under, the first in alphabetical order, or else the factory's
qualified name. The test holds the factory, so that module has run, and with it
its module-level ``register()`` calls, in every run that collects the test. A
registration made by any other file (a strategy file or ``conftest.py`` of any
folder, a test module, a module that calls a helper of the factory's file, a
plugin, a hook) does not name it, and neither does another factory's
registration of the same name nearer to the test. So the full run, the printed
rerun command, ``--lf`` and a run of the test's folder give a row the same name
and values, which each test checks for its layout with ``check_reruns``.

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

# The test of most layouts: tests/b/test_b.py fails on this row
NODEID = "tests/b/test_b.py::test_b[rand-1]"


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


def alias(source, *names, decorated=None):
    """
    A file that imports burst from the module ``source`` and registers it under
    ``names``, and with ``decorated``, a factory of that name registered with a
    decorator, which makes the plugin import a strategy file.
    """
    text = IMPORTS + f"from {source} import burst\n\n"
    text += "".join(f"register({name!r})(burst)\n" for name in names)
    if decorated:
        text += f"\n\n@register({decorated!r})\n" + factory(decorated, "y")
    return text


def module_b(source, prefix=None):
    """tests/b/test_b.py: it passes burst from the module ``source`` and fails on rand-1."""
    head = prefix if prefix is not None else IMPORTS + f"from {source} import burst\n\n"
    return head + "\n\n" + module("test_b", "burst", fails="test_b[rand-1]")


def write(root, files):
    """Write ``files`` (path relative to ``root``: text) into ``root``."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text), encoding="utf-8")


def project(pytester, files, ini="", pythonpath=". tests", root=None):
    """
    Write the conftest.py, a pytest.ini that puts ``pythonpath`` on sys.path and
    has the lines ``ini``, and ``files``, in ``root`` (the pytester folder by
    default).
    """
    ini_text = f"[pytest]\npythonpath = {pythonpath}\n{ini}"
    write(root or pytester.path, {"pytest.ini": ini_text, "conftest.py": CONFTEST, **files})


def run(pytester, *args, ret=0, cache=False, seed=True, root=None):
    """
    Run pytest with ``args``, and the seed unless ``seed`` is false, in the
    current folder, whose rootdir is ``root`` (the pytester folder by default),
    and return its result and the rows it collected, {node ID: [strategy, values]}.
    """
    path = (root or pytester.path) / "rows.json"
    path.unlink(missing_ok=True)
    options = (() if cache else ("-p", "no:cacheprovider")) + (
        (f"--rng-seed={SEED}",) if seed else ()
    )
    result = pytester.runpytest_subprocess(*options, *args)
    assert result.ret == ret, result.stdout.str()
    found: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return result, found


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


def check_reruns(pytester, nodeid, folder, *args, root=None):
    """
    Run the project with ``args``, its row ``nodeid`` failing, then the row's
    printed rerun command, ``--lf`` and a run of ``folder``, and check that each
    gives the row the strategy name, the values and the failure of the full run.
    Return the full run's rows and the row's section.
    """
    root = root or pytester.path
    result, full = run(pytester, *args, ret=1, cache=True, root=root)
    (section,) = sections(result.stdout.lines)
    failure = errors(result.stdout.lines)
    (rerun,) = [line[len("rerun") :].strip() for line in section if line.startswith("rerun ")]
    (root / "rows.json").unlink()
    code, lines = shell(rerun, root)
    rerun_rows = json.loads((root / "rows.json").read_text(encoding="utf-8"))
    lf_result, lf = run(pytester, "--lf", ret=1, cache=True, seed=False, root=root)
    _, part = run(pytester, folder, ret=1, root=root)

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
    return full, section


def values(found):
    """The values of each row, without the strategy names."""
    return {nodeid: row_values for nodeid, (_, row_values) in found.items()}


# tests/bursts.py: two objects of a factory class, registered as "small" and "big"
BURSTS = (
    IMPORTS
    + "\n\nclass Burst:\n"
    + "    def __init__(self, hi):\n"
    + "        self.hi = hi\n\n"
    + "    def __call__(self):\n"
    + "        return Parameter(TestArg('x', rng_type=RNGInteger(0, self.hi)), nsamples=4)\n\n\n"
    + 'small = register("small")(Burst(10**6))\n'
    + 'big = register("big")(Burst(10**6))\n'
)

# The start of a test module that passes small as burst (test_b) and an object of
# Burst that no file registers (test_c)
BURSTS_USER = (
    "from bursts import Burst\n"
    + "from bursts import small as burst\n\n\n"
    + module("test_c", "Burst(10**6)")
)

# tests/common.py: the factory burst, which its file does not register
COMMON = IMPORTS + factory("burst")

# tests/common_strategies.py: the factory burst, which its file registers as "base"
BASE = IMPORTS + '@register("base")\n' + factory("burst")


class TestOwnFile:
    """A name that the factory's own file registers it under names it."""

    def test_a_name_its_strategy_file_registers_names_it_in_another_folder(self, pytester):
        # The final review's layout: tests/a/strategies.py registers burst as "dma"
        # with a decorator, and a test in tests/b imports and passes it. It gets the
        # values of @strategy("dma"), which finds that registration from tests/b too
        path = pytester.path / "tests" / "b" / "test_b.py"
        project(
            pytester,
            {
                "tests/a/strategies.py": IMPORTS + '@register("dma")\n' + factory("burst"),
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma"'),
                "tests/b/test_b.py": module_b("strategies"),
            },
            pythonpath=". tests tests/a",
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")
        path.write_text(IMPORTS + module("test_b", '"dma"', fails="test_b[rand-1]"), "utf-8")
        by_name = run(pytester, ret=1)[1]

        assert names(full, "test_b") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/a/strategies.py:")
        assert by_name == full

    def test_of_two_names_it_gets_the_first_in_alphabetical_order(self, pytester):
        # Registered as "zz_burst" first, then as "dma", by the strategy file of
        # the test's folder: the test gets "dma", and the values of @strategy("dma")
        path = pytester.path / "tests" / "b" / "test_b.py"
        project(
            pytester,
            {
                "tests/b/b_strategies.py": (
                    IMPORTS
                    + '@register("zz_burst")\n'
                    + factory("burst")
                    + '\n\nregister("dma")(burst)\n'
                ),
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b("b_strategies"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")
        path.write_text(IMPORTS + module("test_b", '"dma"', fails="test_b[rand-1]"), "utf-8")
        by_name = run(pytester, ret=1)[1]
        path.write_text(IMPORTS + module("test_b", '"zz_burst"', fails="test_b[rand-1]"), "utf-8")
        other = run(pytester, ret=1)[1]

        assert names(full, "test_b") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/b/b_strategies.py:")
        assert by_name == full
        assert values(other) != values(full)

    def test_a_call_in_a_helper_counts_for_the_module_whose_import_made_it(self, pytester):
        # tests/common.py registers burst through its helper register_as() when it
        # is imported: its own name "zbase". test_a.py calls the helper too, which
        # registers "alpha" (first in alphabetical order): that call is test_a.py's
        project(
            pytester,
            {
                "tests/common.py": (
                    IMPORTS
                    + factory("burst")
                    + "\n\ndef register_as(name):\n    return register(name)(burst)\n\n\n"
                    + 'register_as("zbase")\n'
                ),
                "tests/a/test_a.py": (
                    "from common import register_as\n\n"
                    + 'register_as("alpha")\n\n\n'
                    + "def test_a():\n    pass\n"
                ),
                "tests/b/test_b.py": module_b("common"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"zbase"}
        assert section[0].startswith("strategy  zbase (tests/common.py:")

    def test_objects_of_a_factory_class_get_their_own_names(self, pytester):
        # Two objects of one class, registered by their file as "small" and "big":
        # test_b's object gets its own name, though "big" comes first in
        # alphabetical order. An object that the file did not register gets the
        # first name its file registers an object of the class under
        project(
            pytester,
            {
                "tests/bursts.py": BURSTS,
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b("bursts", prefix=IMPORTS + BURSTS_USER),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"small"}
        assert names(full, "test_c") == {"big"}
        assert section[0].startswith("strategy  small (tests/bursts.py")

    @pytest.mark.parametrize(
        ("where", "name"),
        [
            # Made and registered by the file that defines ranged
            ("tests/ranges.py", "dma"),
            # Made and registered by the strategy file of the test's folder: the
            # partial counts as ranged, whose file does not register it
            ("tests/b/b_strategies.py", "ranged"),
        ],
        ids=["own_file", "strategy_file"],
    )
    def test_a_partial_counts_as_the_function_it_wraps(self, pytester, where, name):
        ranged = (
            "def ranged(hi):\n"
            "    return Parameter(TestArg('x', rng_type=RNGInteger(0, hi)), nsamples=4)\n"
        )
        partial = 'burst = register("dma")(functools.partial(ranged, hi=10**6))\n'
        files = {"tests/ranges.py": "import functools\n\n" + IMPORTS + "\n\n" + ranged}
        if where == "tests/ranges.py":
            files[where] += "\n\n" + partial
        else:
            files[where] = (
                "import functools\n\n"
                + IMPORTS
                + "from ranges import ranged\n\n"
                + partial
                + '\n\n@register("b_other")\n'
                + factory("other", "y")
            )
        source = "ranges" if where == "tests/ranges.py" else "b_strategies"
        # test_c passes a partial of its own, which no file registers
        user = (
            "import functools\n\n"
            + IMPORTS
            + f"from {source} import burst\nfrom ranges import ranged\n\n\n"
            + module("test_c", "functools.partial(ranged, hi=10**6)")
        )
        project(
            pytester,
            {
                **files,
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b(source, prefix=user),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == names(full, "test_c") == {name}
        assert section[0].startswith(f"strategy  {name} (tests/ranges.py:")


class TestOtherFiles:
    """A registration that another file makes does not name the factory."""

    @pytest.mark.parametrize(
        ("files", "source", "ini", "name"),
        [
            # The final review's layout: a helper outside tests/ that only a sibling
            # folder's strategy file registers, loaded with that folder's tests
            pytest.param(
                {
                    "helpers/__init__.py": "",
                    "helpers/factories.py": COMMON,
                    "tests/a/strategies.py": alias("helpers.factories", "dma", decorated="a_other"),
                },
                "helpers.factories",
                "",
                "burst",
                id="helper_aliased_by_a_sibling_strategy_file",
            ),
            # A helper above the test's folder, aliased by a sibling folder's files
            pytest.param(
                {
                    "tests/common.py": COMMON,
                    "tests/a/strategies.py": alias("common", "dma", decorated="a_other"),
                },
                "common",
                "",
                "burst",
                id="sibling_strategy_file",
            ),
            pytest.param(
                {"tests/common.py": COMMON, "tests/a/conftest.py": alias("common", "dma")},
                "common",
                "",
                "burst",
                id="sibling_conftest",
            ),
            # An initial conftest of a run without arguments, not of a run of tests/b
            pytest.param(
                {"tests/common.py": COMMON, "tests/a/conftest.py": alias("common", "dma")},
                "common",
                "testpaths = tests/a tests/b\n",
                "burst",
                id="sibling_conftest_in_testpaths",
            ),
            # A plugin module that such a conftest's pytest_plugins loads
            pytest.param(
                {
                    "tests/common.py": COMMON,
                    "tests/plugin_mod.py": alias("common", "dma"),
                    "tests/a/conftest.py": 'pytest_plugins = ["plugin_mod"]\n',
                },
                "common",
                "testpaths = tests/a tests/b\n",
                "burst",
                id="plugin_of_a_sibling_conftest",
            ),
            # Two strategy files on the test's path, each with its own alias
            pytest.param(
                {
                    "helpers/__init__.py": "",
                    "helpers/factories.py": COMMON,
                    "tests/strategies.py": alias(
                        "helpers.factories", "base", decorated="other_root"
                    ),
                    "tests/b/strategies.py": alias(
                        "helpers.factories", "b_alias", decorated="other_b"
                    ),
                },
                "helpers.factories",
                "",
                "burst",
                id="two_ancestor_strategy_files",
            ),
            # The factory's own file registers it as "base"; each alias "alias"
            # comes before it in alphabetical order, and every run that collects the
            # test makes some of them
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/conftest.py": alias("common_strategies", "alias"),
                },
                "common_strategies",
                "",
                "base",
                id="root_conftest",
            ),
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/b/conftest.py": alias("common_strategies", "alias"),
                },
                "common_strategies",
                "",
                "base",
                id="conftest_of_the_test_folder",
            ),
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/b/b_strategies.py": alias(
                        "common_strategies", "alias", decorated="b_other"
                    ),
                },
                "common_strategies",
                "",
                "base",
                id="strategy_file_of_the_test_folder",
            ),
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/a/a_strategies.py": alias(
                        "common_strategies", "alias", decorated="a_other"
                    ),
                },
                "common_strategies",
                "",
                "base",
                id="sibling_strategy_file_of_a_registered_factory",
            ),
            # The same name again, which replaces the registration of the factory's
            # own file in its folder, in the runs that collect tests/a
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/a/a_strategies.py": alias(
                        "common_strategies", "base", "alias", decorated="a_other"
                    ),
                },
                "common_strategies",
                "",
                "base",
                id="sibling_strategy_file_with_its_name",
            ),
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/b/test_0.py": (
                        alias("common_strategies", "alias") + "\n\ndef test_0():\n    pass\n"
                    ),
                },
                "common_strategies",
                "",
                "base",
                id="another_test_module",
            ),
            pytest.param(
                {
                    "tests/common_strategies.py": BASE,
                    "tests/b/test_b.py": alias("common_strategies", "alias"),
                },
                "common_strategies",
                "",
                "base",
                id="the_test_module",
            ),
        ],
    )
    def test_an_alias_does_not_name_it(self, pytester, files, source, ini, name):
        files = dict(files)
        prefix = files.pop("tests/b/test_b.py", None)
        project(
            pytester,
            {
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                **files,
                "tests/b/test_b.py": module_b(source, prefix),
            },
            ini=ini,
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {name}
        assert section[0].startswith(f"strategy  {name} (")

    def test_a_name_that_only_an_alias_gives_is_still_found_by_name(self, pytester):
        # @strategy("dma") in tests/a finds the alias that tests/a/strategies.py
        # makes; the test in tests/b that passes the factory keeps its qualified name
        project(
            pytester,
            {
                "tests/common.py": COMMON,
                "tests/a/strategies.py": alias("common", "dma", decorated="a_other"),
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma"'),
                "tests/b/test_b.py": module_b("common"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_a") == {"dma"}
        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith("strategy  burst (tests/common.py:")


class TestOtherFactories:
    """Another factory registered under the name nearer to the test does not matter."""

    def test_a_local_strategy_in_another_test_module(self, pytester):
        # tests/b/test_a_other.py registers "burst" for its own factory, and its
        # test uses it by name; f_burst's file registers it as "burst" too
        nodeid = "tests/b/test_x.py::test_x[rand-1]"
        project(
            pytester,
            {
                "tests/strategies.py": IMPORTS + '@register("burst")\n' + factory("f_burst"),
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
        )

        full, section = check_reruns(pytester, nodeid, "tests/b/test_x.py")

        assert names(full, "test_x") == {"burst"}
        assert section[0].startswith("strategy  burst (tests/strategies.py:")

    def test_a_strategy_file_of_the_test_folder(self, pytester):
        # tests/b registers "dma" for another factory: @strategy("dma") there finds
        # that one, and a test there that passes burst still gets burst's "dma"
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
                "tests/b/test_b.py": module_b("common_strategies"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_top") == names(full, "test_b") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/common_strategies.py:")


class TestImportedUnderTwoNames:
    """A factory's file that test modules import under two module names."""

    def test_the_copy_that_replaced_its_registration_still_names_it(self, pytester):
        # With tests/ and the top on sys.path, tests/factories.py is the module
        # tests.factories for test_a.py and test_b.py, and factories for test_c.py.
        # The second import runs the file again: its register() call replaces the
        # registration of the copy test_b.py holds, in the runs that collect
        # test_c.py only
        project(
            pytester,
            {
                "tests/factories.py": IMPORTS + '@register("dma")\n' + factory("burst"),
                "tests/a/test_a.py": (
                    "from tests.factories import burst\n\n\ndef test_a():\n    pass\n"
                ),
                "tests/a/test_c.py": "from factories import burst\n\n\ndef test_c():\n    pass\n",
                "tests/b/test_b.py": module_b("tests.factories"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/factories.py:")

    def test_a_copy_made_after_an_alias_keeps_its_names(self, pytester):
        # tests/b/b_strategies.py registers burst again as "a_base", which its own
        # file registers, and as "0_alias". test_c.py, collected after the strategy
        # file, imports tests/factories.py as factories: the copy's register() call
        # replaces "a_base" in the runs that collect test_c.py
        project(
            pytester,
            {
                "tests/__init__.py": "",
                "tests/factories.py": IMPORTS + '@register("a_base")\n' + factory("burst"),
                "tests/b/b_strategies.py": (
                    alias("tests.factories", "a_base", "0_alias", decorated="b_other")
                ),
                "tests/b/test_c.py": "from factories import burst\n\n\ndef test_c():\n    pass\n",
                "tests/b/test_d.py": module_b("tests.factories").replace("test_b", "test_d"),
            },
        )
        nodeid = "tests/b/test_d.py::test_d[rand-1]"

        full, section = check_reruns(pytester, nodeid, "tests/b/test_d.py")

        assert names(full, "test_d") == {"a_base"}
        assert section[0].startswith("strategy  a_base (tests/factories.py:")

    def test_objects_of_a_factory_class_keep_their_own_names(self, pytester):
        # tests/bursts.py registers two objects of its class Burst, as "small" and
        # "big". test_b.py holds tests.bursts' small, then imports the file as
        # bursts too: in a run of tests/b that import runs the file again, and its
        # copies replace both registrations, while in the full run test_a.py has
        # imported bursts before. small is named "small" either way, not "big",
        # the first name its file gives an object of the class
        prefix = IMPORTS + "from tests.bursts import small as burst\n\nimport bursts  # noqa\n"
        project(
            pytester,
            {
                "tests/bursts.py": BURSTS,
                "tests/a/test_a.py": "import bursts  # noqa\n\n\ndef test_a():\n    pass\n",
                "tests/b/test_b.py": module_b("tests.bursts", prefix=prefix),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"small"}
        assert section[0].startswith("strategy  small (tests/bursts.py")


class TestPackages:
    """
    A factory of a package, installed or checked out, inside the rootdir or
    outside it, as its own file or a plugin module registers it.
    """

    def package(self, site, caller, name="dma_burst"):
        """
        The package ps_kit in ``site``: factories.py defines burst, and ``caller``
        (factories.py or plugin.py) registers it as ``name``.
        """
        registers = f"register({name!r})(burst)\n"
        files = {
            "__init__.py": "",
            "factories.py": IMPORTS + factory("burst"),
            "plugin.py": IMPORTS + "from ps_kit.factories import burst\n\n",
        }
        files[caller] += "\n" + registers
        write(site / "ps_kit", files)

    @pytest.mark.parametrize("where", ["outside", "venv", "src"])
    @pytest.mark.parametrize(
        ("caller", "name"), [("factories.py", "dma_burst"), ("plugin.py", "burst")]
    )
    def test_its_file_names_it_wherever_it_is(self, pytester, monkeypatch, where, caller, name):
        # The project is in proj/: lib/ is outside its rootdir, and a virtualenv's
        # site-packages and a src/ layout's folder are inside it. -p loads the
        # plugin module in every run. The test gets the values of
        # @strategy("dma_burst") only when the package's own file registers burst
        root = pytester.path / "proj"
        site = {
            "outside": pytester.path / "lib",
            "venv": root / ".venv" / "lib" / "site-packages",
            "src": root / "src",
        }[where]
        self.package(site, caller)
        project(
            pytester,
            {
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b("ps_kit.factories"),
            },
            ini="addopts = -p ps_kit.plugin\n",
            pythonpath=f". tests {site.as_posix()}",
            root=root,
        )
        monkeypatch.chdir(root)

        full, section = check_reruns(pytester, NODEID, "tests/b", root=root)
        (root / "tests" / "b" / "test_b.py").write_text(
            IMPORTS + module("test_b", '"dma_burst"', fails="test_b[rand-1]"), "utf-8"
        )
        by_name = run(pytester, ret=1, root=root)[1]

        assert names(full, "test_b") == {name}
        assert section[0].startswith(f"strategy  {name} (")
        assert (values(by_name) == values(full)) == (name == "dma_burst")

    @pytest.mark.parametrize(
        ("caller", "name"), [("factories.py", "dma_burst"), ("plugin.py", "burst")]
    )
    def test_checked_out_and_installed_it_gets_the_same_values(self, pytester, caller, name):
        # One project with the package in src/ (a checkout, or an editable install)
        # and a copy in a virtualenv's site-packages (pip install .): a CI job that
        # installs the package and a developer's checkout give the test the same
        # name and values
        src = pytester.path / "src"
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(src, caller)
        self.package(site, caller)
        found = []
        for folder in (src, site):
            project(
                pytester,
                {"tests/b/test_b.py": module_b("ps_kit.factories")},
                ini="addopts = -p ps_kit.plugin\n",
                pythonpath=f". tests {folder.as_posix()}",
            )
            found.append(run(pytester, ret=1)[1])

        assert names(found[0], "test_b") == {name}
        assert found[0] == found[1]

    @pytest.mark.parametrize("args", [(), ("-p", "ps_kit")], ids=["autoloaded", "given"])
    def test_a_plugin_of_an_entry_point_does_not_name_it(self, pytester, args):
        # ps_kit.plugin is the pytest11 entry point "ps_kit" of an installed
        # distribution, which pytest loads in every run, and registers burst
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        info = site / "ps_kit-1.0.dist-info"
        write(
            info,
            {
                "METADATA": "Metadata-Version: 2.1\nName: ps-kit\nVersion: 1.0\n",
                "entry_points.txt": "[pytest11]\nps_kit = ps_kit.plugin\n",
            },
        )
        project(
            pytester,
            {
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma_burst"'),
                "tests/b/test_b.py": module_b("ps_kit.factories"),
            },
            pythonpath=f". tests {site.as_posix()}",
        )

        full, section = check_reruns(pytester, NODEID, "tests/b", *args)

        assert names(full, "test_a") == {"dma_burst"}
        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith("strategy  burst (")

    @pytest.mark.parametrize(
        ("files", "ini"),
        [
            # tests/a/conftest.py, an initial conftest of a run without arguments
            # (testpaths), loads the plugin; a run of tests/b does not
            (
                {"tests/a/conftest.py": 'pytest_plugins = ["ps_kit.plugin"]\n'},
                "testpaths = tests/a tests/b\n",
            ),
            # pytest loads a test module's pytest_plugins when it collects it
            (
                {
                    "tests/a/test_0.py": (
                        'pytest_plugins = ["ps_kit.plugin"]\n\n\ndef test_0():\n    pass\n'
                    )
                },
                "",
            ),
        ],
        ids=["sibling_conftest", "test_module"],
    )
    def test_a_plugin_that_pytest_plugins_loads_does_not_name_it(self, pytester, files, ini):
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "plugin.py")
        project(
            pytester,
            {**files, "tests/b/test_b.py": module_b("ps_kit.factories")},
            ini=ini,
            pythonpath=f". tests {site.as_posix()}",
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith("strategy  burst (")

    @pytest.mark.parametrize(
        "files",
        [
            # tests/b registers "dma_burst" for another factory, which
            # @strategy("dma_burst") there finds
            {"tests/b/b_strategies.py": IMPORTS + '@register("dma_burst")\n' + factory("local")},
            # Another package's plugin, which every run loads, registers its own
            # factory under the name
            {
                "other_kit/__init__.py": "",
                "other_kit/plugin.py": IMPORTS + '@register("dma_burst")\n' + factory("other"),
            },
        ],
        ids=["test_folder", "another_plugin"],
    )
    def test_another_factory_of_the_same_name_does_not_change_it(self, pytester, files):
        site = pytester.path / ".venv" / "lib" / "site-packages"
        self.package(site, "factories.py")
        project(
            pytester,
            {
                **files,
                "tests/b/test_b.py": module_b("ps_kit.factories"),
            },
            ini="addopts = -p other_kit.plugin\n" if "other_kit/plugin.py" in files else "",
            pythonpath=f". tests {site.as_posix()}",
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_b") == {"dma_burst"}
        assert section[0].startswith("strategy  dma_burst (")


class TestPlugins:
    """Registrations made by a plugin module or a hook, inside the rootdir."""

    # tests/plugin_mod.py registers the burst of ``folder``/common.py as "dma"
    CHAIN = {"tests/plugin_chain.py": 'pytest_plugins = ["plugin_mod"]\n'}

    @pytest.mark.parametrize(
        ("folder", "ini", "files", "conftest", "addopts", "args"),
        [
            # -p in addopts, for a factory on the test's path and in helpers/
            pytest.param("tests", "addopts = -p plugin_mod\n", {}, "", "", (), id="addopts"),
            pytest.param(
                "helpers", "addopts = -p plugin_mod\n", {}, "", "", (), id="addopts_off_path"
            ),
            # pytest_plugins of a conftest.py in the test's folder
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
            # PYTEST_ADDOPTS, and -p on the command line, which the rerun command
            # does not carry
            pytest.param("tests", "", {}, "", "-p plugin_mod", (), id="pytest_addopts"),
            pytest.param("tests", "", {}, "", "", ("-p", "plugin_mod"), id="command_line"),
            pytest.param("tests", "", CHAIN, "", "", ("-p", "plugin_chain"), id="command_chain"),
        ],
    )
    def test_a_plugin_that_registers_another_files_factory_does_not_name_it(
        self, pytester, monkeypatch, folder, ini, files, conftest, addopts, args
    ):
        package = "helpers." if folder == "helpers" else ""
        project(
            pytester,
            {
                f"{folder}/common.py": COMMON,
                "tests/plugin_mod.py": alias(f"{package}common", "dma"),
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b(f"{package}common"),
                **({"helpers/__init__.py": ""} if package else {}),
                **files,
            },
            ini=ini,
        )
        if conftest:
            write(pytester.path, {"conftest.py": CONFTEST + conftest})
        if addopts:
            # Also for the rerun command, run through the shell
            monkeypatch.setenv("PYTEST_ADDOPTS", addopts)

        full, section = check_reruns(pytester, NODEID, "tests/b", *args)

        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith(f"strategy  burst ({folder}/common.py:")

    @pytest.mark.parametrize(
        ("ini", "args"),
        [("addopts = -p plugin_mod\n", ()), ("", ("-p", "plugin_mod"))],
        ids=["addopts", "command_line"],
    )
    def test_a_plugin_module_names_its_own_factory(self, pytester, ini, args):
        # The test imports the factory from the plugin module: every run that
        # collects it has run the module, whether -p loads it or not
        project(
            pytester,
            {
                "tests/plugin_mod.py": IMPORTS + '@register("dma")\n' + factory("burst"),
                "tests/a/test_a.py": "def test_a():\n    pass\n",
                "tests/b/test_b.py": module_b("plugin_mod"),
            },
            ini=ini,
        )

        full, section = check_reruns(pytester, NODEID, "tests/b", *args)

        assert names(full, "test_b") == {"dma"}
        assert section[0].startswith("strategy  dma (tests/plugin_mod.py:")

    def test_a_registration_in_a_hook_does_not_name_it(self, pytester):
        # The root conftest's pytest_configure registers burst as "dma": no
        # module's import makes the call, @strategy("dma") finds it
        hook = (
            "\n\ndef pytest_configure(config):\n"
            "    from common import burst\n"
            "    from pytest_strategy import register\n\n"
            '    register("dma")(burst)\n'
        )
        project(
            pytester,
            {
                "conftest.py": CONFTEST + hook,
                "tests/common.py": COMMON,
                "tests/a/test_a.py": IMPORTS + module("test_a", '"dma"'),
                "tests/b/test_b.py": module_b("common"),
            },
        )

        full, section = check_reruns(pytester, NODEID, "tests/b")

        assert names(full, "test_a") == {"dma"}
        assert names(full, "test_b") == {"burst"}
        assert section[0].startswith("strategy  burst (tests/common.py:")
