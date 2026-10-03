"""
End-to-end tests for what a ``--junitxml`` report holds for failed strategy rows
(D18): the ``pytest-strategies`` section in each failed row's failure text, the
seed and the failed rows' rerun commands as properties of the test suite, and,
with the ``xunit1`` and ``legacy`` families, what each failed row is as
properties of its test case, whose ``command`` runs from the rootdir.

pytest's default family, ``xunit2``, has no properties per test case in its schema,
and junitxml writes a test's ``user_properties`` whatever the family, so there the
plugin adds none. The runs set ``filterwarnings = error``, so a ``PytestWarning``
(``record_property`` warns under ``xunit2``) would fail them.
"""

import json
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

SEED = 21

INI = "[pytest]\nfilterwarnings =\n    error\n"

CONFTEST = """
def pytest_strategies_context(config):
    return {"top": 4095}

def pytest_runtest_setup(item):
    with open(item.config.rootpath / "ran.txt", "a", encoding="utf-8") as ran:
        ran.write(item.nodeid + "\\n")
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, TestArg, register

@register("burst")
def burst(ctx):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, ctx["top"])),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={"zeros": {"addr": 0, "len": 1}},
        vector_constraints={"aligned": lambda v: v.addr % 4 == 0},
        nsamples=3,
    )

@register("mode")
def mode():
    return Parameter(
        TestArg("mode", rng_type=RNGInteger(0, 1)),
        directed_vectors={"fast": ("fast",), "slow": ("slow",)},
        nsamples=0,
    )
"""

# A call failure, a setup error, a failure of stacked strategies and a failure of a
# test without a strategy
TESTS = """
import pytest
from pytest_strategy import strategy

@strategy("burst")
def test_write(request, addr, len):
    assert "rand-1" not in request.node.name, (addr, len)

@pytest.fixture
def device(request):
    if "directed-zeros" in request.node.name:
        raise RuntimeError("no device")

@strategy("burst")
def test_setup(device, addr, len):
    pass

@strategy("burst")
@strategy("mode")
def test_both(addr, len, mode):
    assert not (mode == "slow" and addr == 0)

def test_plain():
    assert False
"""

# The failed rows of TESTS, in the order they fail
FAILED = [
    "test_jx_rows.py::test_write[rand-1]",
    "test_jx_rows.py::test_setup[directed-zeros]",
    "test_jx_rows.py::test_both[directed-slow-directed-zeros]",
]


def write(root, files):
    """Write ``files`` (relative path -> text) below ``root``."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text), encoding="utf-8")


def project(root, folder=".", tests=TESTS):
    """The pytest.ini in ``root``, and conftest.py, strategies.py and test_jx_rows.py in ``folder``."""
    base = Path(folder)
    write(
        root,
        {
            "pytest.ini": INI,
            str(base / "conftest.py"): CONFTEST,
            str(base / "strategies.py"): STRATEGIES,
            str(base / "test_jx_rows.py"): tests,
        },
    )


def run(pytester, *args):
    """Run pytest in a subprocess with the seed and a report.xml, from the current folder."""
    return pytester.runpytest_subprocess(
        "-p",
        "no:cacheprovider",
        f"--rng-seed={SEED}",
        f"--junitxml={pytester.path}/report.xml",
        *args,
    )


def report(pytester):
    """The test suite element of report.xml."""
    (suite,) = ET.parse(pytester.path / "report.xml").getroot().iter("testsuite")
    return suite


def properties(element):
    """The (name, value) properties of a test suite or test case element, in order."""
    found = element.find("properties")
    return [] if found is None else [(p.get("name"), p.get("value")) for p in found]


def cases(suite, name):
    """The test case elements of the test named ``name``."""
    return [case for case in suite.iter("testcase") if case.get("name") == name]


def failure_text(case):
    """The text of a test case's failure or error."""
    (outcome,) = [child for child in case if child.tag in ("failure", "error")]
    return outcome.text or ""


def sections(lines):
    """The pytest-strategies sections in the output: each one's lines, up to its rerun line."""
    found = []
    current = None
    for line in lines:
        if re.fullmatch(r"-+ pytest-strategies -+", line):
            current = []
            found.append(current)
        elif current is not None:
            current.append(line)
            if line.startswith("rerun "):
                current = None
    return found


def section_in(text):
    """The lines after the pytest-strategies rule in a failure text, or None without one."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(r"-+ pytest-strategies -+", line)]
    return lines[starts[0] + 1 :] if starts else None


def failed_rows(lines):
    """The commands of the list of failed rows."""
    start = lines.index("pytest-strategies: failed rows:")
    commands = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        commands.append(line[2:].partition("  # ")[0])
    return commands


def shell(command, cwd):
    """
    Run a rerun command through the platform's shell from ``cwd``, with this
    Python's pytest in place of ``pytest``, and return its exit code and lines.
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


def assert_reruns(command, cwd, rootdir, nodeid):
    """Run ``command`` from ``cwd`` and check that it runs only ``nodeid``, which fails."""
    ran = rootdir / "ran.txt"
    ran.unlink(missing_ok=True)

    code, lines = shell(command, cwd)

    assert code == 1, "\n".join(lines)
    assert ran.read_text(encoding="utf-8").splitlines() == [nodeid], "\n".join(lines)


class TestFamilies:
    @pytest.mark.parametrize("family", [None, "xunit2", "xunit1", "legacy"])
    def test_the_suite_properties_and_the_failure_texts(self, pytester, family):
        project(pytester.path)
        args = [] if family is None else ["-o", f"junit_family={family}"]

        result = run(pytester, *args)

        result.assert_outcomes(failed=3, errors=1, passed=13)
        assert "warnings" not in result.parseoutcomes()
        suite = report(pytester)
        commands = failed_rows(result.stdout.lines)
        assert [command.split()[1].strip("'\"") for command in commands] == FAILED
        assert properties(suite) == [("pytest_strategies.seed", str(SEED))] + [
            (f"pytest_strategies.failed.{i}", command) for i, command in enumerate(commands)
        ]
        shown = sections(result.stdout.lines)
        assert len(shown) == len(FAILED)
        for nodeid in FAILED:
            (section,) = [
                s for s in shown if s[-1].startswith(f"rerun     pytest {quote(nodeid)} ")
            ]
            (case,) = cases(suite, nodeid.partition("::")[2])
            assert section_in(failure_text(case)) == section
        (plain,) = cases(suite, "test_plain")
        assert section_in(failure_text(plain)) is None

    @pytest.mark.parametrize("family", [None, "xunit2"])
    def test_xunit2_test_cases_have_no_properties(self, pytester, family):
        project(pytester.path)
        args = [] if family is None else ["-o", f"junit_family={family}"]

        run(pytester, *args)

        testcases = list(report(pytester).iter("testcase"))
        assert len(testcases) == 17
        assert all(case.find("properties") is None for case in testcases)

    @pytest.mark.parametrize("family", ["xunit1", "legacy"])
    def test_xunit1_and_legacy_failed_rows_have_properties(self, pytester, family):
        project(pytester.path)

        result = run(pytester, "-o", f"junit_family={family}")

        suite = report(pytester)
        (write_case,) = cases(suite, "test_write[rand-1]")
        found = properties(write_case)
        context = dict(found)["pytest_strategies.context"]
        assert re.fullmatch(r"[0-9a-f]{8}", context)
        (section,) = [s for s in sections(result.stdout.lines) if "rand-1" in s[1]]
        values = [line.split()[-1] for line in section if line.startswith(("values", "   "))]
        command = f"pytest {quote(FAILED[0])} --rng-seed={SEED}"
        assert found == [
            ("pytest_strategies.strategy", "burst"),
            ("pytest_strategies.kind", "random"),
            ("pytest_strategies.index", "1"),
            ("pytest_strategies.id", "rand-1"),
            ("pytest_strategies.value.addr", values[0].partition("=")[2]),
            ("pytest_strategies.value.len", values[1].partition("=")[2]),
            ("pytest_strategies.seed", str(SEED)),
            ("pytest_strategies.context", context),
            ("pytest_strategies.command", command),
        ]
        (setup_case,) = cases(suite, "test_setup[directed-zeros]")
        assert dict(properties(setup_case))["pytest_strategies.name"] == "zeros"
        (both_case,) = cases(suite, "test_both[directed-slow-directed-zeros]")
        both = dict(properties(both_case))
        assert (both["pytest_strategies.0.strategy"], both["pytest_strategies.0.name"]) == (
            "mode",
            "slow",
        )
        assert (both["pytest_strategies.1.strategy"], both["pytest_strategies.1.name"]) == (
            "burst",
            "zeros",
        )
        with_properties = [case.get("name") for case in suite.iter("testcase") if properties(case)]
        assert with_properties == [
            "test_write[rand-1]",
            "test_setup[directed-zeros]",
            "test_both[directed-slow-directed-zeros]",
        ]

    def test_the_properties_are_strings_and_only_with_xunit1(self, pytester):
        project(pytester.path)
        common = ["-p", "no:cacheprovider", f"--rng-seed={SEED}"]
        xml = f"--junitxml={pytester.path / 'report.xml'}"

        runs = {
            "none": pytester.inline_run(*common),
            "xunit2": pytester.inline_run(*common, xml),
            "xunit1": pytester.inline_run(*common, xml, "-o", "junit_family=xunit1"),
        }

        found = {}
        for name, reprec in runs.items():
            reports = reprec.getreports("pytest_runtest_logreport")
            found[name] = {(r.nodeid, r.when): r.user_properties for r in reports}
        assert not any(any(props.values()) for name, props in found.items() if name != "xunit1")
        teardown = found["xunit1"][(FAILED[0], "teardown")]
        assert teardown and all(
            isinstance(name, str) and isinstance(value, str) for name, value in teardown
        )
        # The failing report too, which junitxml reads when the teardown fails as well
        assert found["xunit1"][(FAILED[0], "call")] == teardown
        assert found["xunit1"][(FAILED[1], "setup")] == found["xunit1"][(FAILED[1], "teardown")]
        assert found["xunit1"][(FAILED[0], "setup")] == []


class TestCommands:
    def test_the_test_case_command_runs_from_the_rootdir(self, pytester, monkeypatch):
        project(pytester.path, folder="sub")
        monkeypatch.chdir(pytester.path / "sub")

        result = run(pytester, "-o", "junit_family=xunit1", "--nsamples=2")

        suite = report(pytester)
        (case,) = cases(suite, "test_write[rand-1]")
        command = dict(properties(case))["pytest_strategies.command"]
        nodeid = "sub/test_jx_rows.py::test_write[rand-1]"
        assert command == f"pytest {quote(nodeid)} --rng-seed={SEED} --nsamples=2"
        # The suite property is the command of the list of failed rows, run from sub
        failed = dict(properties(suite))["pytest_strategies.failed.0"]
        assert failed == failed_rows(result.stdout.lines)[0]
        assert failed.split()[1].strip("'\"") == "test_jx_rows.py::test_write[rand-1]"
        assert_reruns(command, pytester.path, pytester.path, nodeid)
        assert_reruns(failed, pytester.path / "sub", pytester.path, nodeid)

    def test_with_an_ini_file_from_c_the_command_names_it_from_the_rootdir(self, pytester):
        # -c makes the ini file's folder the rootdir
        project(pytester.path, folder="ci")
        write(pytester.path, {"ci/pytest.ini": INI})

        run(pytester, "-c", str(Path("ci", "pytest.ini")), "-o", "junit_family=xunit1")

        suite = report(pytester)
        (case,) = cases(suite, "test_write[rand-1]")
        command = dict(properties(case))["pytest_strategies.command"]
        assert command == f"pytest {quote(FAILED[0])} --rng-seed={SEED} -c pytest.ini"
        failed = dict(properties(suite))["pytest_strategies.failed.0"]
        assert failed == " ".join(
            [
                "pytest",
                quote(str(Path("ci", "test_jx_rows.py")) + "::test_write[rand-1]"),
                f"--rng-seed={SEED}",
                "-c",
                quote(str(Path("ci", "pytest.ini"))),
            ]
        )
        assert_reruns(command, pytester.path / "ci", pytester.path / "ci", FAILED[0])
        assert_reruns(failed, pytester.path, pytester.path / "ci", FAILED[0])

    def test_a_test_outside_the_rootdir_is_selected_with_k_from_the_rootdir(self, pytester):
        # -c ci/pytest.ini makes ci the rootdir, and the tests in tests/ are outside it
        project(pytester.path, folder="tests")
        write(pytester.path, {"ci/pytest.ini": INI})

        result = run(pytester, "-c", "ci/pytest.ini", "-o", "junit_family=xunit1")

        suite = report(pytester)
        (case,) = cases(suite, "test_write[rand-1]")
        command = dict(properties(case))["pytest_strategies.command"]
        expression = quote("test_write[rand-1]")
        assert command == f"pytest .. --rng-seed={SEED} -c pytest.ini -k {expression}"
        failed = dict(properties(suite))["pytest_strategies.failed.0"]
        assert failed == f"pytest . --rng-seed={SEED} -c ci/pytest.ini -k {expression}"
        assert failed == failed_rows(result.stdout.lines)[0]
        nodeid = "tests/test_jx_rows.py::test_write[rand-1]"
        assert_reruns(command, pytester.path / "ci", pytester.path / "ci", nodeid)
        assert_reruns(failed, pytester.path, pytester.path / "ci", nodeid)


class TestOutcomes:
    def test_a_failure_and_a_teardown_error_both_carry_the_properties(self, pytester):
        project(
            pytester.path,
            tests="""
import pytest
from pytest_strategy import strategy

@pytest.fixture
def device(request):
    yield
    if "rand-1" in request.node.name:
        raise RuntimeError("stuck")

@strategy("burst")
def test_write(request, device, addr, len):
    assert "rand-1" not in request.node.name
""",
        )

        result = run(pytester, "-o", "junit_family=xunit1")

        result.assert_outcomes(failed=1, errors=1, passed=3)
        # junitxml writes the failure and the teardown error as two test cases
        failure, error = cases(report(pytester), "test_write[rand-1]")
        assert failure.find("failure") is not None and error.find("error") is not None
        assert properties(failure) == properties(error) != []
        assert dict(properties(failure))["pytest_strategies.id"] == "rand-1"

    def test_a_failed_test_run_again_gets_its_properties_once(self, pytester):
        # As pytest-rerunfailures does: the protocol runs again, with the item's
        # user_properties of the first run, and the last run is reported
        project(pytester.path)
        rerun = """
import json

import pytest
from _pytest.runner import runtestprotocol

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    reports = runtestprotocol(item, nextitem=nextitem, log=False)
    if any(report.failed for report in reports):
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
    for report in reports:
        item.ihook.pytest_runtest_logreport(report=report)
        if report.failed:
            with open(item.config.rootpath / "failed.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(report.user_properties) + "\\n")
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True
"""
        write(pytester.path, {"conftest.py": CONFTEST + rerun})

        result = run(pytester, "-o", "junit_family=xunit1")

        result.assert_outcomes(failed=3, errors=1, passed=13)
        # Each test ran twice
        ran = (pytester.path / "ran.txt").read_text(encoding="utf-8").splitlines()
        assert ran.count(FAILED[0]) == 2
        suite = report(pytester)
        written = []
        for nodeid in FAILED:
            (case,) = cases(suite, nodeid.partition("::")[2])
            found = properties(case)
            assert found == list(dict.fromkeys(found)) != [], nodeid
            written.append(found)
        # The 9 of test_write[rand-1], as after one run
        assert len(written[0]) == 9
        # And in the failed reports, which junitxml reads when the teardown fails too
        # (test_plain's has none)
        lines = (pytester.path / "failed.jsonl").read_text(encoding="utf-8").splitlines()
        reported = [[tuple(prop) for prop in json.loads(line)] for line in lines]
        assert sorted(reported, key=len) == [[], *written]

    def test_a_strict_xpass_has_properties_but_no_section(self, pytester):
        project(
            pytester.path,
            tests="""
import pytest
from pytest_strategy import strategy

@pytest.mark.xfail(strict=True, reason="not fixed yet")
@strategy("burst")
def test_write(request, addr, len):
    assert "rand-1" in request.node.name
""",
        )

        run(pytester, "-o", "junit_family=xunit1")

        suite = report(pytester)
        (case,) = cases(suite, "test_write[rand-1]")
        assert section_in(failure_text(case)) is None
        assert "XPASS(strict)" in case.find("failure").get("message")
        assert dict(properties(case))["pytest_strategies.id"] == "rand-1"
        assert dict(properties(suite))["pytest_strategies.failed.0"] == (
            f"pytest {quote(FAILED[0])} --rng-seed={SEED}"
        )

    def test_a_passing_run_has_only_the_seed(self, pytester):
        project(pytester.path, tests="def test_ok():\n    pass\n")

        result = run(pytester, "-o", "junit_family=xunit1")

        result.assert_outcomes(passed=1)
        assert properties(report(pytester)) == [("pytest_strategies.seed", str(SEED))]
        assert properties(next(report(pytester).iter("testcase"))) == []

    def test_a_run_without_the_junitxml_plugin(self, pytester):
        project(pytester.path)

        result = pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", "-p", "no:junitxml", f"--rng-seed={SEED}"
        )

        result.assert_outcomes(failed=3, errors=1, passed=13)
        assert len(sections(result.stdout.lines)) == 3


class TestXdist:
    @pytest.mark.parametrize("family", ["xunit2", "xunit1"])
    def test_the_controller_writes_the_workers_rows(self, pytester, family):
        pytest.importorskip("xdist")
        project(pytester.path)

        result = run(pytester, "-n", "2", "-o", f"junit_family={family}")

        result.assert_outcomes(failed=3, errors=1, passed=13)
        commands = failed_rows(result.stdout.lines)
        assert sorted(command.split()[1].strip("'\"") for command in commands) == sorted(FAILED)
        suite = report(pytester)
        assert properties(suite) == [("pytest_strategies.seed", str(SEED))] + [
            (f"pytest_strategies.failed.{i}", command) for i, command in enumerate(commands)
        ]
        for nodeid in FAILED:
            (case,) = cases(suite, nodeid.partition("::")[2])
            assert section_in(failure_text(case))[-1] == (
                f"rerun     pytest {quote(nodeid)} --rng-seed={SEED}"
            )
            found = properties(case)
            if family == "xunit2":
                assert found == []
            else:
                commands = {value for key, value in found if key.endswith(".command")}
                assert commands == {f"pytest {quote(nodeid)} --rng-seed={SEED}"}
                assert {value for key, value in found if key.endswith(".seed")} == {str(SEED)}
