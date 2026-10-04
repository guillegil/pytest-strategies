"""
End-to-end tests for what a failed strategy row reports (D18): the
``pytest-strategies`` section under its traceback, and the list of failed rows
after the line that says how to reproduce the run.

The printed rerun commands are run through the platform's shell (``sh`` on POSIX,
``cmd.exe`` on Windows) from the folder the run started in, with this Python's
pytest in place of ``pytest``, so every CI cell checks that the command's quoting
works on its OS and that it runs exactly the failed row, which fails the same way:
with the same section, values included. Each project's conftest.py writes the node
ID of every test it sets up to ``ran.txt`` in the rootdir.
"""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent

import pytest

from pytest_strategy._repro import quote

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _strategy_details(pytester, monkeypatch):
    """These runs check the per-row section, which --strategy-details turns on
    (also for the rerun commands run through the shell, which inherit it). After
    pytester, which clears PYTEST_ADDOPTS."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "--strategy-details")


SEED = 21

CONFTEST = """
def pytest_strategies_context(config):
    return {"top": 4095}

def pytest_runtest_setup(item):
    with open(item.config.rootpath / "ran.txt", "a", encoding="utf-8") as ran:
        ran.write(item.nodeid + "\\n")
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, RNGSequence, Series, TestArg, register

@register("burst")
def burst(ctx):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, ctx["top"])),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={"zeros": {"addr": 0, "len": 1}, "it's a&b (x)": {"addr": 4, "len": 2}},
        test_vectors={"max": {"addr": ctx["top"], "len": 64}},
        vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        nsamples=3,
    )

@register("chan")
def chan():
    return Parameter(
        TestArg("ch", rng_type=Series([0, 1, 2])),
        TestArg("dev", rng_type=RNGSequence(["a", "b"])),
        TestArg("x", rng_type=RNGInteger(0, 99)),
        nsamples=2,
    )
"""


def module(condition, strategy="burst", args="addr, len", name="test_write"):
    """A test module whose test fails when ``condition`` holds, showing its values."""
    return f"""
from pytest_strategy import strategy

@strategy({strategy!r})
def {name}(request, {args}):
    assert not ({condition}), f"{{request.node.name}}: {{{args}}}"
"""


def write(root, files):
    """Write ``files`` (relative path -> text) below ``root``."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text), encoding="utf-8")


def project(root, test, folder="."):
    """The conftest.py, strategies.py and test_dma.py (``test``) in ``folder``."""
    base = Path(folder)
    write(
        root,
        {
            str(base / "conftest.py"): CONFTEST,
            str(base / "strategies.py"): STRATEGIES,
            str(base / "test_dma.py"): test,
        },
    )


def run(pytester, *args):
    """Run pytest in a subprocess with the seed, from the current folder."""
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", f"--rng-seed={SEED}", *args)


def sections(lines):
    """
    The pytest-strategies sections in the output: each one's lines, up to its rerun
    line and note. The short test summary is left out: with CI set, pytest does not
    cut its lines, and under xdist an error without a crash line (a missing fixture)
    repeats its whole report there.
    """
    found = []
    current = None
    for line in lines:
        if re.fullmatch(r"=+ short test summary info =+", line):
            break
        if re.fullmatch(r"-+ pytest-strategies -+", line):
            current = []
            found.append(current)
        elif current is not None:
            if (
                current
                and current[-1].startswith(("rerun ", "note "))
                and not line.startswith("note ")
            ):
                current = None
            else:
                current.append(line)
    return found


def errors(lines, name):
    """The ``E`` lines of the failure of the test named ``name``."""
    start = next(
        i for i, line in enumerate(lines) if re.fullmatch(rf"_+ {re.escape(name)} _+", line)
    )
    found = []
    for line in lines[start + 1 :]:
        if re.fullmatch(r"_+ .+ _+|-+ pytest-strategies -+", line):
            break
        if line.startswith("E "):
            found.append(line)
    return found


def error_section(path, name):
    """The section lines in the failure or error text of the test case ``name`` of a JUnit XML report."""
    (case,) = [c for c in ET.parse(path).getroot().iter("testcase") if c.get("name") == name]
    (outcome,) = [child for child in case if child.tag in ("failure", "error")]
    return sections((outcome.text or "").splitlines())


def rerun_command(section):
    """The command on a section's rerun line."""
    (line,) = [line for line in section if line.startswith("rerun ")]
    return line[len("rerun") :].strip()


def failed_rows(lines):
    """The lines that list the failed rows, without their indent."""
    start = lines.index("pytest-strategies: failed rows:")
    rows = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        rows.append(line[2:])
    return rows


def shell(command, cwd):
    """
    Run a printed rerun command through the platform's shell from ``cwd``, with this
    Python's pytest in place of ``pytest``, and return its output lines.
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


def assert_reruns(command, cwd, rootdir, nodeid, section, output=None, name=None):
    """
    Run ``command`` from ``cwd`` and check that it runs only ``nodeid`` (rootdir-
    relative), which fails with the same section, and the same error lines when
    ``output`` (the first run's lines) is given: those of the failure headed
    ``name``, by default the test's name.
    """
    ran = rootdir / "ran.txt"
    ran.unlink(missing_ok=True)

    code, lines = shell(command, cwd)

    assert code == 1, "\n".join(lines)
    assert ran.read_text(encoding="utf-8").splitlines() == [nodeid], "\n".join(lines)
    assert sections(lines) == [section], "\n".join(lines)
    if output is not None:
        name = name or nodeid.rpartition("::")[2]
        assert errors(lines, name) == errors(output, name) != []


class TestSection:
    @pytest.mark.parametrize("quiet", [[], ["-q"]], ids=["default", "q"])
    def test_a_failed_row_shows_what_it_is(self, pytester, quiet):
        pytester.makeini("[pytest]\n")
        project(pytester.path, module('"directed-zeros" in request.node.name'))

        result = run(pytester, *quiet)

        result.assert_outcomes(failed=1, passed=4)
        (section,) = sections(result.stdout.lines)
        assert section[:6] == [
            "strategy  burst (strategies.py:4)",
            "vector    directed-zeros (directed vector 'zeros', #0)",
            "values    addr=0",
            "          len=1",
            f"seed      {SEED}",
            section[5],
        ]
        assert re.fullmatch(r"context   [0-9a-f]{8}", section[5])
        assert section[6:] == [
            f"rerun     pytest {quote('test_dma.py::test_write[directed-zeros]')} --rng-seed={SEED}"
        ]

    def test_no_context_line_when_the_factory_did_not_receive_one(self, pytester):
        project(
            pytester.path,
            module("ch == 1 and dev == 'b'", strategy="chan", args="ch, dev, x", name="test_chan"),
        )

        result = run(pytester, "--nsamples=auto")

        result.assert_outcomes(failed=1, passed=5)
        (section,) = sections(result.stdout.lines)
        assert section[1] == "vector    ch=1-dev=b (exhaustive row 3)"
        assert not any(line.startswith("context") for line in section)

    def test_the_section_follows_the_traceback(self, pytester):
        project(pytester.path, module('"rand-1" in request.node.name'))

        result = run(pytester, "--show-capture=no")

        lines = result.stdout.lines
        start = lines.index(next(line for line in lines if "_ test_write[rand-1] _" in line))
        error = next(i for i, line in enumerate(lines) if line.startswith("E ") and i > start)
        header = next(i for i, line in enumerate(lines) if "- pytest-strategies -" in line)
        assert start < error < header

    def test_none_for_a_test_without_a_strategy_or_a_passing_run(self, pytester):
        project(pytester.path, module("False"))
        write(pytester.path, {"test_plain.py": "def test_plain():\n    assert False\n"})

        failing = run(pytester)
        pytester.path.joinpath("test_plain.py").unlink()
        passing = run(pytester)

        failing.assert_outcomes(failed=1, passed=5)
        assert sections(failing.stdout.lines) == []
        failing.stdout.no_fnmatch_line("*failed rows*")
        passing.assert_outcomes(passed=5)
        passing.stdout.no_fnmatch_line("*pytest-strategies -*")
        passing.stdout.no_fnmatch_line("*failed rows*")

    def test_stacked_strategies_get_a_block_each_and_one_rerun_line(self, pytester):
        project(
            pytester.path,
            """
from pytest_strategy import strategy

@strategy("burst")
@strategy("chan")
def test_both(addr, len, ch, dev, x):
    assert not (ch == 1 and addr == 0), f"{ch=} {dev=} {x=}"
""",
        )

        result = run(pytester)

        (section,) = sections(result.stdout.lines)
        assert [line for line in section if line.startswith("strategy")] == [
            "strategy  chan (strategies.py:15)",
            "strategy  burst (strategies.py:4)",
        ]
        assert sum(line.startswith("rerun") for line in section) == 1
        command = rerun_command(section)
        assert "test_both[ch=1-rand-0-directed-zeros]" in command
        nodeid = command.split()[1].strip("'\"")
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    def test_a_long_value_is_cut_below_vv(self, pytester):
        write(
            pytester.path,
            {
                "test_long.py": """
from pytest_strategy import Parameter, TestArg, strategy

def blob():
    return Parameter(TestArg("blob", value="x" * 5000))

@strategy(blob)
def test_blob(blob):
    assert False
""",
            },
        )

        cut = run(pytester)
        full = run(pytester, "-vv")

        assert "values    blob='" + "x" * 3999 + "... (5002 characters; -vv shows all)" in (
            cut.stdout.lines
        )
        assert "values    blob='" + "x" * 5000 + "'" in full.stdout.lines

    @pytest.mark.parametrize("family", ["xunit2", "xunit1"])
    def test_a_value_whose_repr_raises_is_shown_by_its_type(self, pytester, family):
        write(
            pytester.path,
            {
                "test_reg.py": """
from pytest_strategy import Parameter, RNGChoice, TestArg, strategy

class Reg:
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        raise RuntimeError("no repr while the device is off")

def regs():
    return Parameter(TestArg("reg", rng_type=RNGChoice([Reg(1), Reg(2)])), nsamples=2)

@strategy(regs)
def test_reg(reg):
    assert reg.n > 5
""",
            },
        )

        result = run(pytester, "--junitxml=out.xml", "-o", f"junit_family={family}")

        # The failures are reported, and the session goes on
        result.assert_outcomes(failed=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert [section[2] for section in sections(result.stdout.lines)] == [
            "values    reg=<Reg: repr() raised RuntimeError>"
        ] * 2
        assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()


class TestRerun:
    """The printed command runs exactly the failed row, which fails the same way."""

    @pytest.mark.parametrize(
        "condition, args, nodeid, options",
        [
            (
                '"rand-12" in request.node.name',
                ["--nsamples=13"],
                "test_dma.py::test_write[rand-12]",
                "--nsamples=13",
            ),
            (
                '"test-max" in request.node.name',
                ["--vector-mode=test"],
                "test_dma.py::test_write[test-max]",
                "--vector-mode=test",
            ),
            (
                "addr % 4 != 0",
                ["--strategy-constraint-off=aligned", "--nsamples=4"],
                None,
                "--nsamples=4 --strategy-constraint-off=burst:aligned",
            ),
            (
                '"it\'s" in request.node.name',
                [],
                "test_dma.py::test_write[directed-it's a&b (x)]",
                "",
            ),
        ],
        ids=["random_row", "test_vector", "constraint_off", "special_characters"],
    )
    def test_a_row_of_burst(self, pytester, condition, args, nodeid, options):
        pytester.makeini("[pytest]\n")
        project(pytester.path, module(condition))

        result = run(pytester, *args)

        assert result.ret == 1
        section = sections(result.stdout.lines)[0]
        command = rerun_command(section)
        if nodeid is None:
            # The first failed row, from the short test summary
            nodeid = next(
                line.split()[1]
                for line in result.stdout.lines
                if line.startswith("FAILED test_dma.py::")
            )
        expected = f"pytest {quote(nodeid)} --rng-seed={SEED}" + (f" {options}" if options else "")
        assert command == expected
        assert failed_rows(result.stdout.lines)[0].startswith(f"{command}  # burst ")
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    @pytest.mark.parametrize(
        "name, nodeid",
        [
            ("café ü", r"test_dma.py::test_name[directed-caf\xe9 \xfc]"),
            ('q"uote\\', r'test_dma.py::test_name[directed-q"uote\\]'),
        ],
        ids=["non_ascii", "quote_and_backslash"],
    )
    def test_a_name_that_pytest_escapes(self, pytester, monkeypatch, name, nodeid):
        # pytest escapes non-ASCII characters and backslashes in node IDs, so the
        # command holds backslashes, and a double quote that Windows escapes
        monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
        project(
            pytester.path,
            f"""
from pytest_strategy import Parameter, RNGInteger, TestArg, strategy

def named():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(2, 9)),
        directed_vectors={{{name!r}: {{"a": 1}}}},
        nsamples=1,
    )

@strategy(named)
def test_name(request, a):
    assert a != 1, request.node.name
""",
        )

        result = run(pytester)

        result.assert_outcomes(failed=1, passed=1)
        (section,) = sections(result.stdout.lines)
        assert section[1] == f"vector    directed-{name} (directed vector {name!r}, #0)"
        command = rerun_command(section)
        assert command == f"pytest {quote(nodeid)} --rng-seed={SEED}"
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    @pytest.mark.parametrize(
        "option, row",
        [
            ("--vector-mode=test", "test-max"),
            ("--vector-name=zeros", "directed-zeros"),
            ("--vector-index=0", "directed-zeros"),
        ],
        ids=["vector_mode", "vector_name", "vector_index"],
    )
    def test_a_project_whose_empty_parameter_sets_fail_at_collect(self, pytester, option, row):
        # test_chan's strategy has no such vector: its empty parameter set needs the
        # run's -o, the last one, which the rerun of the node ID needs too, as it
        # collects the whole module
        pytester.makeini("[pytest]\nempty_parameter_set_mark = fail_at_collect\n")
        chan = '\n@strategy("chan")\ndef test_chan(ch, dev, x):\n    pass\n'
        project(pytester.path, module("True") + chan)
        mark = ["-o", "empty_parameter_set_mark=xfail", "-o", "empty_parameter_set_mark=skip"]

        refused = run(pytester, option)
        result = run(pytester, option, *mark)

        assert refused.ret == pytest.ExitCode.INTERRUPTED
        result.assert_outcomes(failed=1, skipped=1)
        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        nodeid = f"test_dma.py::test_write[{row}]"
        assert command == " ".join(
            ["pytest", quote(nodeid), f"--rng-seed={SEED}", quote(option), "-o"]
            + [quote("empty_parameter_set_mark=skip")]
        )
        assert failed_rows(result.stdout.lines) == [f"{command}  # burst {row.replace('-', ' ')}"]
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    def test_an_exhaustive_row(self, pytester):
        project(
            pytester.path,
            module("ch == 1 and dev == 'b'", strategy="chan", args="ch, dev, x", name="test_chan"),
        )

        result = run(pytester, "--nsamples=auto")

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        nodeid = "test_dma.py::test_chan[ch=1-dev=b]"
        assert command == f"pytest {quote(nodeid)} --rng-seed={SEED} --nsamples=auto"
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    def test_a_run_with_an_ini_file_from_c(self, pytester):
        # -c makes the ini file's folder the rootdir, so the tests below it keep their
        # node IDs; another pytest.ini at the top would give the rerun another rootdir
        pytester.makeini("[pytest]\n")
        write(pytester.path, {"ci/pytest.ini": "[pytest]\nstrategies_ids = names\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="ci")

        result = run(pytester, "-c", str(Path("ci", "pytest.ini")))

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        nodeid = str(Path("ci", "test_dma.py")) + "::test_write[rand-1]"
        assert command == (
            f"pytest {quote(nodeid)} --rng-seed={SEED} -c {quote(str(Path('ci', 'pytest.ini')))}"
        )
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "test_dma.py::test_write[rand-1]",
            section,
            result.stdout.lines,
        )

    def test_a_run_with_an_ini_file_from_c_and_a_rootdir(self, pytester):
        write(pytester.path, {"ci/pytest.ini": "[pytest]\nstrategies_ids = names\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="tests")

        result = run(pytester, "-c", "ci/pytest.ini", "--rootdir=.")

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        nodeid = "tests/test_dma.py::test_write[rand-1]"
        assert command == (f"pytest {quote(nodeid)} --rng-seed={SEED} -c ci/pytest.ini --rootdir=.")
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)

    def test_a_test_outside_the_rootdir_is_selected_with_k(self, pytester):
        # pytest names a test outside the rootdir by the path the run started from
        # (here the folder it runs in), so the command starts from the same path and
        # selects the row with -k
        write(pytester.path, {"ci/pytest.ini": "[pytest]\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="tests")

        result = run(pytester, "-c", "ci/pytest.ini")

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        assert command == (
            f"pytest . --rng-seed={SEED} -c ci/pytest.ini -k {quote('test_write[rand-1]')}"
        )
        assert not section[-1].startswith("note")
        assert failed_rows(result.stdout.lines) == [f"{command}  # burst random 1"]
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "tests/test_dma.py::test_write[rand-1]",
            section,
            result.stdout.lines,
        )

    def test_outside_the_rootdir_the_run_s_own_paths(self, pytester):
        write(pytester.path, {"ci/pytest.ini": "[pytest]\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="tests")
        nodeid = str(Path("tests", "test_dma.py")) + "::test_write"

        result = run(pytester, "-c", "ci/pytest.ini", nodeid)

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        assert command == (
            f"pytest {quote(nodeid)} --rng-seed={SEED} -c ci/pytest.ini"
            f" -k {quote('test_write[rand-1]')}"
        )
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "::test_write[rand-1]",
            section,
            result.stdout.lines,
        )

    def test_outside_the_rootdir_k_names_the_module_when_needed(self, pytester):
        # test_other.py has a test of the same name, which -k deselects: the rerun
        # command, which runs without that -k, must not select it either
        write(pytester.path, {"ci/pytest.ini": "[pytest]\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="tests")
        write(pytester.path, {"tests/test_other.py": module('"rand-1" in request.node.name')})

        result = run(pytester, "-c", "ci/pytest.ini", "-k", "test_dma")

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        expression = "test_dma.py and test_write[rand-1]"
        assert command == f"pytest . --rng-seed={SEED} -c ci/pytest.ini -k {quote(expression)}"
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "tests/test_dma.py::test_write[rand-1]",
            section,
            result.stdout.lines,
        )

    def test_outside_the_rootdir_k_names_the_class_when_needed(self, pytester):
        # Two classes of the module have a test of the same name
        write(pytester.path, {"ci/pytest.ini": "[pytest]\n"})
        test = """
from pytest_strategy import strategy

class TestA:
    @strategy("burst")
    def test_write(self, request, addr, len):
        assert "rand-1" not in request.node.name, (addr, len)

class TestB:
    @strategy("burst")
    def test_write(self, addr, len):
        pass
"""
        project(pytester.path, test, folder="tests")

        result = run(pytester, "-c", "ci/pytest.ini")

        result.assert_outcomes(failed=1, passed=9)
        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        expression = "test_dma.py and TestA and test_write[rand-1]"
        assert command == f"pytest . --rng-seed={SEED} -c ci/pytest.ini -k {quote(expression)}"
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "tests/test_dma.py::TestA::test_write[rand-1]",
            section,
            result.stdout.lines,
            name="TestA.test_write[rand-1]",
        )

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--ignore", str(Path("tests", "b"))), ("--ignore-glob", str(Path("*", "b", "*")))],
        ids=["ignore", "ignore_glob"],
    )
    def test_outside_the_rootdir_the_run_s_ignored_files_stay_out(self, pytester, option, value):
        # tests/b/test_other.py has a test of the same name, which the run never
        # collected: the rerun command, which starts from the same path, leaves it out too
        write(pytester.path, {"ci/pytest.ini": "[pytest]\n"})
        project(pytester.path, module('"rand-1" in request.node.name'), folder="tests")
        write(pytester.path, {"tests/b/test_other.py": module('"rand-1" in request.node.name')})

        result = run(pytester, "-c", "ci/pytest.ini", f"{option}={value}")

        result.assert_outcomes(failed=1, passed=4)
        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        assert command == (
            f"pytest . {option} {quote(value)} --rng-seed={SEED} -c ci/pytest.ini"
            f" -k {quote('test_write[rand-1]')}"
        )
        assert_reruns(
            command,
            pytester.path,
            pytester.path / "ci",
            "tests/test_dma.py::test_write[rand-1]",
            section,
            result.stdout.lines,
        )

    @pytest.mark.parametrize(
        ("files", "failing"),
        [
            # Two modules of the same name: no -k expression tells their rows apart
            (
                {"tests/b/test_dma.py": module('"rand-1" in request.node.name')},
                "test_write[rand-1]",
            ),
            # A name -k cannot hold
            ({}, "test_chan[ch=1-dev=b]"),
        ],
        ids=["same_names", "not_a_k_name"],
    )
    def test_outside_the_rootdir_without_a_k_expression_a_note(self, pytester, files, failing):
        # importlib lets two test modules have one name
        ini = "[pytest]\naddopts = --import-mode=importlib\n"
        write(pytester.path, {"ci/pytest.ini": ini, **files})
        project(
            pytester.path,
            module('"rand-1" in request.node.name')
            + module("ch == 1 and dev == 'b'", "chan", "ch, dev, x", "test_chan"),
            folder="tests",
        )

        result = run(pytester, "-c", "ci/pytest.ini", "--nsamples=auto")

        nodeid = str(Path("tests", "test_dma.py")) + f"::{failing}"
        (section,) = [
            s
            for s in sections(result.stdout.lines)
            if rerun_command(s).startswith(f"pytest {quote(nodeid)} ")
        ]
        assert rerun_command(section) == (
            f"pytest {quote(nodeid)} --rng-seed={SEED} --nsamples=auto -c ci/pytest.ini"
        )
        assert section[-1].startswith("note      the test is outside the rootdir (ci), so pytest")
        assert f"{rerun_command(section)}  # " in "\n".join(failed_rows(result.stdout.lines))
        assert "(outside the rootdir)" in "\n".join(failed_rows(result.stdout.lines))

    def test_only_the_constraints_of_the_failed_row_s_strategy(self, pytester):
        write(
            pytester.path,
            {
                "pytest.ini": "[pytest]\n",
                "conftest.py": CONFTEST,
                "strategies.py": """
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("strat_a")
def strat_a():
    return Parameter(
        TestArg("a", rng_type=RNGInteger(0, 9)),
        vector_constraints={"a_only": lambda v: v.a < 5},
        nsamples=2,
    )

@register("strat_b")
def strat_b():
    return Parameter(
        TestArg("b", rng_type=RNGInteger(0, 9)),
        vector_constraints={"b_only": lambda v: v.b < 5},
        nsamples=2,
    )
""",
                "test_mod_a.py": module(
                    '"rand-0" in request.node.name', strategy="strat_a", args="a", name="test_a"
                ),
                "test_mod_b.py": module("False", strategy="strat_b", args="b", name="test_b"),
            },
        )

        result = run(pytester, "--strategy-constraint-off=a_only,b_only")

        result.assert_outcomes(failed=1, passed=3)
        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        nodeid = "test_mod_a.py::test_a[rand-0]"
        assert command == (
            f"pytest {quote(nodeid)} --rng-seed={SEED} --strategy-constraint-off=strat_a:a_only"
        )
        code, lines = shell(command, pytester.path)
        assert code == 1, "\n".join(lines)
        assert not any("matched no constraint" in line for line in lines)
        assert_reruns(command, pytester.path, pytester.path, nodeid, section, result.stdout.lines)
        # The run's own option would name a constraint that module A does not have
        bare = command.replace("strat_a:a_only", "a_only,b_only")
        code, lines = shell(bare, pytester.path)
        assert "--strategy-constraint-off=b_only matched no constraint." in "\n".join(lines)

    def test_from_a_subfolder_the_commands_are_relative_to_it(self, pytester, monkeypatch):
        pytester.makeini("[pytest]\n")
        project(pytester.path, module('"rand-2" in request.node.name'), folder="sub")
        monkeypatch.chdir(pytester.path / "sub")

        result = run(pytester)

        (section,) = sections(result.stdout.lines)
        command = rerun_command(section)
        assert command == f"pytest {quote('test_dma.py::test_write[rand-2]')} --rng-seed={SEED}"
        assert failed_rows(result.stdout.lines) == [f"{command}  # burst random 2"]
        assert_reruns(
            command,
            pytester.path / "sub",
            pytester.path,
            "sub/test_dma.py::test_write[rand-2]",
            section,
            result.stdout.lines,
        )

    @pytest.mark.parametrize("xdist", [False, True], ids=["one_process", "xdist"])
    def test_a_terminal_that_cannot_encode_a_value_or_a_name(self, pytester, xdist):
        # A Windows CI log is written in cp1252, which has no Greek letters, arrows or
        # CJK characters. pytest writes a text that holds one escaped as a whole: the
        # section would print as one line, and the backslashes of the escapes pytest
        # puts in the node ID would be doubled, so the command would run no test
        if xdist:
            pytest.importorskip("xdist")
        write(
            pytester.path,
            {"conftest.py": CONFTEST, "strategies.py": PHASE, "test_dma.py": PHASE_TEST},
        )
        nodeid = r"test_dma.py::test_phase[directed-\u03b8\u2192max]"
        command = f"pytest {quote(nodeid)} --rng-seed={SEED}"
        section = [
            "strategy  \u7b56\u7565 (strategies.py:4)",
            "vector    directed-\u03b8\u2192max (directed vector '\u03b8\u2192max', #0)",
            "values    r='1k\u03a9'",
            "          deg=360",
            f"seed      {SEED}",
            f"rerun     {command}",
        ]

        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--rng-seed={SEED}"]
            + ["--junitxml=report.xml"]
            + (["-n", "2"] if xdist else []),
            cwd=pytester.path,
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
            timeout=300,
        )

        lines = done.stdout.decode("cp1252").splitlines()
        assert done.returncode == 1, "\n".join(lines)
        assert sections(lines) == [
            [
                r"strategy  \u7b56\u7565 (strategies.py:4)",
                r"vector    directed-\u03b8\u2192max (directed vector '\u03b8\u2192max', #0)",
                r"values    r='1k\u03a9'",
                "          deg=360",
                f"seed      {SEED}",
                f"rerun     {command}",
            ]
        ], "\n".join(lines)
        assert failed_rows(lines) == [command + r"  # \u7b56\u7565 directed \u03b8\u2192max"]
        # The JUnit XML report, written in UTF-8, keeps the characters: the plugin
        # escapes the report after the other plugins read it
        junit = pytester.path / "report.xml"
        assert error_section(junit, nodeid.partition("::")[2]) == [section]
        # Run as printed, on a terminal that writes every character
        assert_reruns(command, pytester.path, pytester.path, nodeid, section)


# A strategy whose name, directed vector name and value hold characters cp1252 has
# not, and its test. The files hold escapes, so they are ASCII.
PHASE = r"""
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register

@register("\u7b56\u7565")
def phase():
    return Parameter(
        TestArg("r", rng_type=Series(["1k\u03a9"])),
        TestArg("deg", rng_type=RNGInteger(0, 99)),
        directed_vectors={"\u03b8\u2192max": {"r": "1k\u03a9", "deg": 360}},
        nsamples=1,
    )
"""

PHASE_TEST = r"""
from pytest_strategy import strategy

@strategy("\u7b56\u7565")
def test_phase(r, deg):
    assert deg < 100
"""


class TestFailedRows:
    def test_the_commands_follow_the_reproduce_line(self, pytester):
        project(pytester.path, module('"rand" in request.node.name'))

        result = run(pytester)

        lines = result.stdout.lines
        start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith(f"pytest-strategies: reproduce with --rng-seed={SEED} (context ")
        )
        assert lines[start + 1] == "pytest-strategies: failed rows:"
        commands = [rerun_command(section) for section in sections(lines)]
        assert lines[start + 2 : start + 5] == [
            f"  {command}  # burst random {i}" for i, command in enumerate(commands)
        ]
        assert not lines[start + 5].startswith("  ")

    @pytest.mark.parametrize(
        "args, shown, more",
        [([], 10, "  ... and 5 more"), (["-q"], 10, "  ... and 5 more"), (["-v"], 15, None)],
        ids=["default", "q", "v"],
    )
    def test_at_most_ten_below_v(self, pytester, args, shown, more):
        project(pytester.path, module("True"))

        result = run(pytester, "--nsamples=13", *args)

        result.assert_outcomes(failed=15)
        rows = failed_rows(result.stdout.lines)
        assert len(rows) == shown + (more is not None)
        assert rows[0].endswith("  # burst directed zeros")
        if more is not None:
            assert rows[-1] == more[2:]

    def test_qq_prints_only_the_reproduce_line(self, pytester):
        project(pytester.path, module("True"))

        result = run(pytester, "-qq")

        assert len(sections(result.stdout.lines)) == 5
        summary = [line for line in result.stdout.lines if line.startswith("pytest-strategies:")]
        assert re.fullmatch(
            rf"pytest-strategies: reproduce with --rng-seed={SEED} \(context [0-9a-f]{{8}}\)",
            summary[-1],
        )
        assert not any(line.startswith("  pytest ") for line in result.stdout.lines)
        result.stdout.no_fnmatch_line("*failed rows*")

    def test_a_setup_error_gives_a_section_and_a_row(self, pytester):
        project(
            pytester.path,
            """
import pytest
from pytest_strategy import strategy

@pytest.fixture
def device(request):
    if "rand-1" in request.node.name:
        raise RuntimeError("no device")

@strategy("burst")
def test_write(device, addr, len):
    pass
""",
        )

        result = run(pytester)

        result.assert_outcomes(errors=1, passed=4)
        (section,) = sections(result.stdout.lines)
        assert section[1] == "vector    rand-1 (random row 1)"
        assert failed_rows(result.stdout.lines) == [f"{rerun_command(section)}  # burst random 1"]

    @pytest.mark.parametrize("xdist", [False, True], ids=["one_process", "xdist"])
    def test_a_missing_fixture_gives_a_section_and_a_row(self, pytester, xdist):
        # pytest reports a missing fixture without a traceback, in a report that
        # takes no sections: it is wrapped in one that does
        project(
            pytester.path,
            """
from pytest_strategy import strategy

@strategy("burst")
def test_write(not_a_fixture, addr, len):
    pass
""",
        )
        args = ["-n", "2"] if xdist else []
        if xdist:
            pytest.importorskip("xdist")
        junit = pytester.path / "report.xml"

        result = run(pytester, "-k", "rand-1", f"--junitxml={junit}", *args)

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*fixture 'not_a_fixture' not found*"])
        (section,) = sections(result.stdout.lines)
        assert section[1] == "vector    rand-1 (random row 1)"
        assert rerun_command(section) == (
            f"pytest {quote('test_dma.py::test_write[rand-1]')} --rng-seed={SEED}"
        )
        assert failed_rows(result.stdout.lines) == [f"{rerun_command(section)}  # burst random 1"]
        assert error_section(junit, "test_write[rand-1]") == [section]

    def test_a_teardown_error_gives_neither(self, pytester):
        project(
            pytester.path,
            """
import pytest
from pytest_strategy import strategy

@pytest.fixture
def device(request):
    yield
    if "rand-1" in request.node.name:
        raise RuntimeError("stuck")

@strategy("burst")
def test_write(device, addr, len):
    pass
""",
        )

        result = run(pytester)

        result.assert_outcomes(errors=1, passed=5)
        assert sections(result.stdout.lines) == []
        result.stdout.fnmatch_lines([f"pytest-strategies: reproduce with --rng-seed={SEED}*"])
        result.stdout.no_fnmatch_line("*failed rows*")

    def test_a_strict_xpass_gives_a_row_without_a_section(self, pytester):
        project(
            pytester.path,
            """
import pytest
from pytest_strategy import strategy

@pytest.mark.xfail(strict=True, reason="not fixed yet")
@strategy("burst")
def test_write(request, addr, len):
    assert "rand-1" in request.node.name
""",
        )

        result = run(pytester, "-rfx")

        result.assert_outcomes(failed=1, xfailed=4)
        result.stdout.fnmatch_lines(["*XPASS(strict)*not fixed yet*"])
        assert sections(result.stdout.lines) == []
        assert failed_rows(result.stdout.lines) == [
            f"pytest {quote('test_dma.py::test_write[rand-1]')} --rng-seed={SEED}  # burst random 1"
        ]

    def test_a_failure_a_later_wrapper_sets_gets_a_section_and_a_row(self, pytester):
        # sub/conftest.py is loaded while the tests are collected, after the plugin
        # registered: its makereport wrapper runs inside the plugin's (tryfirst)
        project(pytester.path, module("False"), folder="sub")
        (pytester.path / "sub" / "conftest.py").write_text(
            CONFTEST + """
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if call.when == "call" and item.name == "test_write[rand-1]":
        try:
            raise AssertionError("flagged by the bus checker")
        except AssertionError:
            excinfo = pytest.ExceptionInfo.from_current()
        report.outcome = "failed"
        report.longrepr = item.repr_failure(excinfo)
    return report
""",
            encoding="utf-8",
        )

        result = run(pytester)

        result.assert_outcomes(failed=1, passed=4)
        (section,) = sections(result.stdout.lines)
        # The run starts in the rootdir, so the command holds pytest's own node ID
        nodeid = "sub/test_dma.py::test_write[rand-1]"
        assert rerun_command(section) == f"pytest {quote(nodeid)} --rng-seed={SEED}"
        assert failed_rows(result.stdout.lines) == [
            f"pytest {quote(nodeid)} --rng-seed={SEED}  # burst random 1"
        ]

    def test_the_guard_s_message_and_the_row_share_one_section(self, pytester):
        # test_b asks for strategies_ctx in another context than test_root, which
        # requested it first: the guard's message goes into the row's section
        write(
            pytester.path,
            {
                "conftest.py": 'def pytest_strategies_context(config):\n    return {"name": "root"}\n',
                "test_0root.py": "def test_root(strategies_ctx):\n    pass\n",
                "tests/b/conftest.py": (
                    'def pytest_strategies_context(config):\n    return {"name": "B"}\n'
                ),
                "tests/b/test_b.py": """
from pytest_strategy import Parameter, RNGInteger, TestArg, strategy

def burst(ctx):
    return Parameter(TestArg("addr", rng_type=RNGInteger(0, 9)), nsamples=1)

@strategy(burst)
def test_b(request, addr):
    assert request.getfixturevalue("strategies_ctx") == {"name": "B"}
""",
            },
        )
        junit = pytester.path / "report.xml"

        result = run(pytester, f"--junitxml={junit}")

        result.assert_outcomes(failed=1, passed=1)
        (section,) = sections(result.stdout.lines)
        nodeid = "tests/b/test_b.py::test_b[rand-0]"
        assert section[:4] == [
            "strategies_ctx is a session fixture, but the tests that use it have different "
            f"contexts (conftest.py: test_0root.py::test_root; tests/b/conftest.py: {nodeid}). "
            "In a folder with its own pytest_strategies_context, use "
            "pytest_strategy.get_context(request.config, __file__) in that folder's conftest.py "
            "fixtures.",
            "",
            "strategy  burst (tests/b/test_b.py:4)",
            "vector    rand-0 (random row 0)",
        ]
        assert re.fullmatch(r"values    addr=\d", section[4])
        assert section[5] == f"seed      {SEED}"
        assert re.fullmatch(r"context   [0-9a-f]{8}", section[6])
        assert section[7:] == [f"rerun     pytest {quote(nodeid)} --rng-seed={SEED}"]
        assert failed_rows(result.stdout.lines) == [f"{rerun_command(section)}  # burst random 0"]
        assert error_section(junit, "test_b[rand-0]") == [section]

    def test_the_report_carries_the_row_as_strings(self, pytester):
        project(pytester.path, module('"rand-12" in request.node.name'))

        reprec = pytester.inline_run(
            "-p",
            "no:cacheprovider",
            f"--rng-seed={SEED}",
            "--nsamples=13",
            "-o",
            "strategies_ids=names",
        )

        reports = reprec.getreports("pytest_runtest_logreport")
        (report,) = [report for report in reports if hasattr(report, "pytest_strategies")]
        assert (report.nodeid, report.when) == ("test_dma.py::test_write[rand-12]", "call")
        assert report.pytest_strategies == {
            "command": f"pytest {quote(report.nodeid)} --rng-seed={SEED} --nsamples=13"
            " -o strategies_ids=names",
            "row": "burst random 12",
            "seed": str(SEED),
            "options": '["--nsamples=13", "-o", "strategies_ids=names"]',
        }
        # pytest-xdist sends the report to the controller in this form
        assert report._to_json()["pytest_strategies"] == report.pytest_strategies

    def test_under_xdist_the_controller_lists_the_workers_rows(self, pytester):
        pytest.importorskip("xdist")
        project(pytester.path, module('"rand" in request.node.name'))

        result = run(pytester, "-n", "2")

        result.assert_outcomes(failed=3, passed=2)
        commands = sorted(rerun_command(section) for section in sections(result.stdout.lines))
        rows = failed_rows(result.stdout.lines)
        assert sorted(row.partition("  # ")[0] for row in rows) == commands
        assert sorted(row.partition("  # ")[2] for row in rows) == [
            f"burst random {i}" for i in range(3)
        ]
        for command in commands:
            nodeid = "test_dma.py::" + command.split("::", 1)[1].split()[0].strip("'\"")
            section = next(s for s in sections(result.stdout.lines) if rerun_command(s) == command)
            assert_reruns(command, pytester.path, pytester.path, nodeid, section)
