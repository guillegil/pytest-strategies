"""
End-to-end checks that documented examples and CLI options actually work.

The package docstring, the runnable snippets of README.md / docs/dev.md and the
scripts in examples/ are copied into a pytester sandbox and run the way a user
would run them. These documents went stale before (they called the renamed
``Parameter.generate_samples`` and a ``--seed`` flag that never existed), and
nothing ran examples/, so a flaky example went unnoticed.
"""

import argparse
import importlib.util
import random
import re
import shlex
import textwrap
from pathlib import Path

import pytest

import pytest_strategy
from pytest_strategy import RNG, Strategy, _cli
from pytest_strategy._registry import registry

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((REPO_ROOT / "examples").glob("*.py"))
OPTION_RE = re.compile(r"(?<![\w-])--[a-z][\w-]*")
DEFINED_OPTION_RE = re.compile(r"addoption\(\s*[\"'](--[a-z][\w-]*)")
REGISTER_RE = re.compile(r"@(?:Strategy\.)?register\(")
STRATEGY_RE = re.compile(r"@(?:Strategy\.)?strategy\(")


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Undo what the in-process runs change globally: the registry and the RNG seed."""
    registry = dict(Strategy._registry)
    seed = RNG.get_seed()
    state = random.getstate()
    yield
    Strategy._registry.clear()
    Strategy._registry.update(registry)
    RNG.seed(seed)
    random.setstate(state)


def _cli_actions(parser: argparse.ArgumentParser) -> list:
    """The actions of ``parser`` and of its subcommands, recursively."""
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                actions.extend(_cli_actions(subparser))
    return actions


def _package_docstring_example() -> str:
    """The "Example Usage" section of ``help(pytest_strategy)``."""
    doc = pytest_strategy.__doc__ or ""
    return textwrap.dedent(doc.split("Example Usage:", 1)[1].split("Dataclass Support:", 1)[0])


def _package_docstring_cli_lines() -> list:
    """The pytest command lines listed under "CLI Options" in the package docstring."""
    section = (pytest_strategy.__doc__ or "").split("CLI Options:", 1)[1]
    commands = (line.split("#", 1)[0].strip() for line in section.splitlines())
    return [shlex.split(command)[1:] for command in commands if command.startswith("pytest ")]


def _runnable_markdown_blocks() -> list:
    """Python blocks of README.md and docs/dev.md that both register and use a strategy."""
    params = []
    for doc in ("README.md", "docs/dev.md"):
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        for match in re.finditer(r"```python\n(.*?)```", text, re.S):
            source = match.group(1)
            if REGISTER_RE.search(source) and STRATEGY_RE.search(source):
                line = text[: match.start()].count("\n") + 2
                params.append(pytest.param(source, id=f"{doc}:{line}"))
    return params


class TestPackageDocstring:
    """The example in ``help(pytest_strategy)`` must run as documented."""

    @pytest.mark.parametrize("args", _package_docstring_cli_lines(), ids=" ".join)
    def test_example_runs_with_documented_cli_line(self, pytester, args):
        pytester.makepyfile(test_doc_example=_package_docstring_example())
        result = pytester.runpytest(*args)
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        assert result.parseoutcomes().get("passed", 0) > 0

    def test_example_honours_vector_mode(self, pytester):
        """The example's directed vectors are the rows --vector-mode=directed_only keeps."""
        pytester.makepyfile(test_doc_example=_package_docstring_example())
        pytester.runpytest("--vector-mode", "directed_only").assert_outcomes(passed=2)


class TestMarkdownExamples:
    """Self-contained strategy snippets in the Markdown docs must run."""

    @pytest.mark.parametrize("source", _runnable_markdown_blocks())
    def test_snippet_runs(self, pytester, source):
        pytester.makepyfile(test_snippet=source)
        result = pytester.runpytest("--rng-seed", "1")
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()


class TestDocumentedCliOptions:
    def test_every_documented_option_exists(self, pytester):
        """Every --option mentioned in the docs and examples must be accepted by pytest."""
        sources = [
            (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs" / "dev.md").read_text(encoding="utf-8"),
            pytest_strategy.__doc__ or "",
            *(path.read_text(encoding="utf-8") for path in EXAMPLES),
        ]
        documented = {option for text in sources for option in OPTION_RE.findall(text)}
        # Options an example conftest.py adds itself (parser.addoption("--x"))
        defined = {option for text in sources for option in DEFINED_OPTION_RE.findall(text)}

        known = set(OPTION_RE.findall(pytester.runpytest("--help").stdout.str()))
        # The options of the pytest-strategies command (skill install)
        parser = _cli.build_parser()
        known.update(
            option
            for action in _cli_actions(parser)
            for option in action.option_strings
            if option.startswith("--")
        )
        # Options of the other tools the contributing guide runs (black, mypy)
        known.update({"--check", "--strict"})
        assert documented - known - defined == set()


def _load_example_factories(path: Path, monkeypatch) -> tuple:
    """
    Import an example file and return (module, registry of its strategy factories).

    The example's ``@strategy`` tests are only marked: only the factories are
    needed, and they are registered in an empty registry so they neither clash
    with nor leak into the session's strategies.
    """
    saved = registry.snapshot()
    registry.clear()
    try:
        spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, dict(Strategy._registry)
    finally:
        registry.restore(saved)


class TestExamples:
    """Every script in examples/ must pass."""

    @pytest.mark.parametrize("seed", [2, 4, 5])
    @pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
    def test_example_passes(self, pytester, example, seed):
        """Run each example file as a test module with a few fixed seeds."""
        pytester.makepyfile(**{f"test_{example.stem}": example.read_text(encoding="utf-8")})
        result = pytester.runpytest(f"--rng-seed={seed}")
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()

    def test_role_strategy_never_draws_a_guest_with_a_write_method(self, monkeypatch):
        """enum_example.py's role_based_strategy drew a GUEST with POST/PUT (which
        test_role_based_access forbids) for about a third of the seeds.

        Which seeds fail depends on the random stream key, which includes the test
        file's path, so this checks the strategy itself over many seeds instead of
        running the example file with a few.
        """
        module, factories = _load_example_factories(
            REPO_ROOT / "examples" / "enum_example.py", monkeypatch
        )
        param = factories["role_based_strategy"](nsamples=10)

        bad = []
        for seed in range(100):
            RNG.seed(seed)
            bad.extend(
                (seed, role, method)
                for role, method in param.generate_vectors(20)
                if role is module.UserRole.GUEST and method is not module.RequestMethod.GET
            )
        assert bad == []

    @staticmethod
    def _run_as_ci(pytester, *args):
        """
        Run test_values_example.py as CI does, from the rootdir with the example's
        folder as testpaths and no path on the command line.
        """
        example = REPO_ROOT / "examples" / "test_values_example.py"
        pytester.mkdir("examples")
        (pytester.path / "examples" / example.name).write_text(
            example.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return pytester.runpytest(
            "-o", "testpaths=examples", "-o", "python_files=test_values_example.py", *args
        )

    @pytest.mark.parametrize("seed", [2, 4, 5])
    def test_values_example_passes_with_its_constraint_turned_off(self, pytester, seed):
        """CI runs test_values_example.py with --strategy-constraint-off=date_range_test:ordered."""
        result = self._run_as_ci(
            pytester,
            f"--rng-seed={seed}",
            "--strategy-constraint-off=date_range_test:ordered",
            "-v",
        )

        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        result.stdout.fnmatch_lines(["  date_range_test (*): *; off: ordered"])

    def test_ci_run_of_the_example_counts_as_the_whole_suite(self, pytester):
        """The run collects testpaths, so a misspelled name fails CI instead of
        printing a red line."""
        result = self._run_as_ci(pytester, "--strategy-constraint-off=date_range_test:orderd")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "ERROR: --strategy-constraint-off=date_range_test:orderd matched no "
                "constraint. Did you mean 'date_range_test:ordered'? *"
            ]
        )

    def test_values_example_rejects_the_ranges_its_constraint_keeps_out(self, monkeypatch):
        """With 'ordered' off, date_range_test draws ranges that end before they
        start, and test_date_ranges expects days_in_range to reject them."""
        module, factories = _load_example_factories(
            REPO_ROOT / "examples" / "test_values_example.py", monkeypatch
        )
        param = factories["date_range_test"](nsamples=10)
        assert list(param.vector_constraints) == ["ordered"]

        RNG.seed(1)
        assert all(row.start_day <= row.end_day for row in param.generate_vectors(50))
        rows = param.generate_vectors(50, constraints_off=("ordered",))
        reversed_rows = [row for row in rows if row.start_day > row.end_day]
        assert reversed_rows
        for row in reversed_rows:
            module.test_date_ranges(*row)
