"""
Unit tests keeping the documentation, examples and packaging metadata honest.

Each test pins a defect that shipped: a README snippet that did not compile, a
missing PEP 561 marker, trove classifiers and formatter targets that disagreed
with ``requires-python``, a CI workflow that never ran on the main line branch,
and example docstrings describing behaviour the library does not have.

The APIs that 4.0 removed may appear in the README, docs/dev.md and the skill only in
their upgrade sections (D21).

Example files are parsed with ``ast`` rather than imported: importing one runs
its ``@Strategy.strategy`` decorators and registers its strategies globally.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _python_blocks(relative_path: str):
    """Yield (line, source) for every ```python block of a Markdown file."""
    text = _read(relative_path)
    for match in re.finditer(r"```python\n(.*?)```", text, re.S):
        yield text[: match.start()].count("\n") + 2, match.group(1)


@pytest.fixture(scope="module")
def pyproject():
    """Parsed pyproject.toml."""
    return tomllib.loads(_read("pyproject.toml"))


# ---------------------------------------------------------------------------
# Markdown documentation
# ---------------------------------------------------------------------------


class TestMarkdownDocs:
    """Snippets and tables in README.md and docs/dev.md."""

    @pytest.mark.parametrize("doc", ["README.md", "docs/dev.md"])
    def test_python_blocks_compile(self, doc):
        """Every python block must be valid Python.

        The README "Cross-Argument Constraints" block had a stray ``]`` / ``)``
        after the closing parenthesis and raised SyntaxError when copied.
        """
        for line, source in _python_blocks(doc):
            try:
                compile(source, f"{doc}:{line}", "exec")
            except SyntaxError as e:
                pytest.fail(f"{doc}:{line}: python block does not compile: {e}")

    def test_vector_index_is_documented_as_a_directed_vector_index(self):
        """--vector-index only indexes directed vectors, not samples in general."""
        row = next(
            line
            for line in _read("README.md").splitlines()
            if line.startswith("| `--vector-index`")
        )
        assert "directed vector" in row


# The APIs that 4.0 removed (D14), as they are written in prose and in code
REMOVED_APIS = {
    "tuple factories": r"\(argnames, samples\)|\b[Tt]uple factor(?:y|ies)\b|\blegacy_tuple\b",
    "RNG.set_max_retries()": r"\bset_max_retries\b",
    "configure()": (
        r"(?<![\w.])configure\(|\bpytest_strategy\.configure\b"
        r"|\bimport\b[^\n]*(?<![\w.])configure\b"
    ),
    "Strategy.set_config()": r"\bset_config\b|\bruntime\.config\b",
    "TestArg directed_values": r"(?<!\w)(?:has_)?directed_values\b",
    "TestArg test_values": r"(?<![\w/-])(?:has_)?test_values\b(?![-.])",
    "TestArg always_include_directed": (
        r"TestArg\((?:[^()]|\([^()]*\))*?\balways_include_directed\b"
    ),
}

# The documents D21 keeps free of removed APIs outside their upgrade sections
SKILL_DIR = "src/pytest_strategy/skill/pytest-strategies"
DOCS_WITHOUT_REMOVED_APIS = [
    "README.md",
    "docs/dev.md",
    f"{SKILL_DIR}/SKILL.md",
    f"{SKILL_DIR}/references/api.md",
]

UPGRADE_HEADING = re.compile(r"(?i)upgrad|migrat|removed in|deprecat")


def _upgrade_lines(text: str) -> list[bool]:
    """
    For each line of a Markdown text, whether it is in an upgrade section.

    An upgrade section starts at a heading that names upgrading, migrating, removals
    or deprecations, and ends at the next heading of the same or a higher level.
    Lines inside code fences are not headings (``# conftest.py`` is a comment).
    """
    flags = []
    in_fence = False
    level = None
    for line in text.split("\n"):
        if re.match(r"\s*(```|~~~)", line):
            in_fence = not in_fence
        elif not in_fence and (heading := re.match(r"(#{1,6})\s+(.*)", line)):
            depth = len(heading.group(1))
            if level is not None and depth <= level:
                level = None
            if level is None and UPGRADE_HEADING.search(heading.group(2)):
                level = depth
        flags.append(level is not None)
    return flags


def _removed_api_mentions(text: str) -> list[str]:
    """Return ``line: API`` for each removed API mentioned outside an upgrade section."""
    in_upgrade = _upgrade_lines(text)
    found = []
    for api, pattern in REMOVED_APIS.items():
        for match in re.finditer(pattern, text, re.S):
            line = text.count("\n", 0, match.start())
            if not in_upgrade[line]:
                found.append(f"{line + 1}: {api}")
    return sorted(found, key=lambda entry: int(entry.split(":")[0]))


class TestRemovedApisInDocs:
    """
    The documentation mentions the APIs that 4.0 removed only where it says how to
    upgrade (D21), so a user who copies an example never gets a removed API.
    """

    @pytest.mark.parametrize("doc", DOCS_WITHOUT_REMOVED_APIS)
    def test_removed_apis_appear_only_in_upgrade_sections(self, doc):
        found = _removed_api_mentions(_read(doc))
        assert not found, (
            f"{doc} mentions removed APIs outside an upgrade section "
            f"(under a heading that names upgrading, migrating, removals or deprecations):\n"
            + "\n".join(f"  {doc}:{entry}" for entry in found)
        )

    def test_the_readme_has_an_upgrade_section(self):
        """The README's removed APIs live in "Upgrading to 4.0", so the check is not vacuous."""
        text = _read("README.md")
        in_upgrade = _upgrade_lines(text)
        lines = text.split("\n")
        mentioned = [
            api
            for api, pattern in REMOVED_APIS.items()
            if any(
                in_upgrade[text.count("\n", 0, m.start())] for m in re.finditer(pattern, text, re.S)
            )
        ]
        assert any(re.match(r"#+ .*Upgrading to 4\.0", line) for line in lines)
        assert mentioned == list(REMOVED_APIS)

    @pytest.mark.parametrize(
        "snippet",
        [
            "A factory returns an `(argnames, samples)` tuple.",
            "Tuple factories are fine.",
            "Call `RNG.set_max_retries(5)` first.",
            "from pytest_strategy import configure, register",
            "configure()",
            "pytest_strategy.configure(verbose=True)",
            "Strategy.set_config(config)",
            "TestArg('x', rng_type=RNGInteger(0, 9), directed_values=[1])",
            "arg.has_directed_values",
            "TestArg('x', value=1, test_values=[2])",
            "`has_test_values` is False",
            "TestArg(\n    'x',\n    rng_type=RNGInteger(0, 9),\n    always_include_directed=False,\n)",
        ],
    )
    def test_a_removed_api_outside_an_upgrade_section_is_found(self, snippet):
        text = f"# Guide\n\n## Usage\n\n{snippet}\n\n## Upgrading from 3.x\n\n{snippet}\n"
        found = _removed_api_mentions(text)
        assert found and all(entry.startswith("5: ") for entry in found)

    @pytest.mark.parametrize(
        "snippet",
        [
            "def pytest_configure(config):\n    config.addinivalue_line('markers', 'x')",
            "pytest examples/test_values_example.py --vector-mode=test",
            "Parameter(TestArg('x', value=1), always_include_directed=False)",
            "Parameter(max_retries=50, directed_vectors={'zero': (0,)})",
            "test_vectors={'max': (9,)} and directed values",
            "A tuple of values; `request.config.getoption('--rng-seed')`",
        ],
    )
    def test_current_apis_are_not_mistaken_for_removed_ones(self, snippet):
        assert _removed_api_mentions(f"## Usage\n\n{snippet}\n") == []

    def test_an_upgrade_section_ends_at_the_next_heading_of_its_level(self):
        text = "\n".join(
            [
                "## Upgrading to 4.0",  # 1
                "`RNG.set_max_retries()` is gone.",  # 2
                "### Details",  # 3
                "`Strategy.set_config()` is gone.",  # 4
                "```python",  # 5
                "# conftest.py: a comment, not a heading",  # 6
                "```",  # 7
                "`configure()` is gone.",  # 8
                "## Options",  # 9
                "`RNG.set_max_retries()` again.",  # 10
            ]
        )
        assert _removed_api_mentions(text) == ["10: RNG.set_max_retries()"]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


class TestExampleDocstrings:
    """Docstrings in examples/ must match what the library does."""

    def test_test_values_example_does_not_promise_test_vectors_outside_test_mode(self):
        """Test vectors only run with --vector-mode=test, never in the default "all" mode."""
        tree = ast.parse(_read("examples/test_values_example.py"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            doc = " ".join((ast.get_docstring(node) or "").split())
            for sentence in re.split(r"\s(?=With\s)", doc):
                if "default" not in sentence and "--vector-mode=all" not in sentence:
                    continue
                claim = sentence.replace("test vectors are not included", "")
                assert not re.search(r"\btest\b", claim), f"{node.name}: {sentence!r}"

    def test_sequence_example_does_not_call_rngsequence_deterministic(self):
        """Since 1.1.0a2 RNGSequence yields a random permutation under --nsamples=auto."""
        tree = ast.parse(_read("examples/sequence_example.py"))
        assert "deterministic" not in (ast.get_docstring(tree) or "").lower()


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


class TestPackagingMetadata:
    """pyproject.toml must describe the package that is actually shipped."""

    def test_py_typed_marker_exists(self):
        """The "Typing :: Typed" classifier and package-data promise a PEP 561 marker."""
        assert (REPO_ROOT / "src" / "pytest_strategy" / "py.typed").is_file()

    def test_py_typed_marker_is_declared_as_package_data(self, pyproject):
        assert "py.typed" in pyproject["tool"]["setuptools"]["package-data"]["pytest_strategy"]

    def test_python_classifiers_match_ci_matrix_and_requires_python(self, pyproject):
        """Advertise exactly the versions CI tests, none below requires-python."""
        project = pyproject["project"]
        classified = sorted(
            (c.rsplit(" :: ", 1)[1] for c in project["classifiers"] if re.search(r":: 3\.\d+$", c)),
            key=lambda v: int(v.split(".")[1]),
        )
        matrix = re.search(r"python-version: \[([^\]]*)\]", _read(".github/workflows/tests.yml"))
        assert matrix is not None
        tested = [v.strip().strip("\"'") for v in matrix.group(1).split(",")]

        assert classified == tested
        minimum = re.fullmatch(r">=3\.(\d+)", project["requires-python"])
        assert minimum is not None
        assert int(classified[0].split(".")[1]) == int(minimum.group(1))

    def test_development_status_matches_prerelease_version(self, pyproject):
        """An alpha/beta version must not be classified as Production/Stable."""
        project = pyproject["project"]
        status = [c for c in project["classifiers"] if c.startswith("Development Status")]
        pre = re.fullmatch(r"[\d.]+(?:(a|b|rc)\d+)?", project["version"])
        assert pre is not None
        if pre.group(1) == "a":
            assert status == ["Development Status :: 3 - Alpha"]
        elif pre.group(1):
            assert "Development Status :: 5 - Production/Stable" not in status

    def test_black_targets_match_requires_python(self, pyproject):
        """black must not target Python versions the package does not support."""
        minors = [int(t[len("py3") :]) for t in pyproject["tool"]["black"]["target-version"]]
        minimum = re.fullmatch(r">=3\.(\d+)", pyproject["project"]["requires-python"])
        assert minimum is not None
        assert min(minors) == int(minimum.group(1))

    def test_package_version_matches_pyproject(self, pyproject):
        """1.0.0 shipped with ``__version__ = "0.1.0"``."""
        tree = ast.parse(_read("src/pytest_strategy/__init__.py"))
        versions = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "__version__" for t in node.targets)
        ]
        assert versions == [pyproject["project"]["version"]]

    def test_final_version_has_a_changelog_section(self, pyproject):
        version = pyproject["project"]["version"]
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            assert f"\n## [{version}] - " in _read("CHANGELOG.md")

    def test_changelog_agrees_with_the_development_status(self, pyproject):
        """The 2.0.0 notes said "development status is Alpha" for a Production/Stable package."""
        project = pyproject["project"]
        (status,) = [c for c in project["classifiers"] if c.startswith("Development Status")]
        section = _read("CHANGELOG.md").split(f"\n## [{project['version']}]", 1)[-1]
        section = section.split("\n## [", 1)[0]
        for claimed in re.findall(r"development status is ([\w/-]+)", section, re.I):
            assert (
                claimed.lower() in status.lower()
            ), f"CHANGELOG says {claimed!r}, pyproject {status!r}"

    def test_license_is_the_same_everywhere(self, pyproject):
        """LICENSE said GPL-3.0, the metadata Apache-2.0 and the README MIT."""
        project = pyproject["project"]
        assert project["license"] == "MIT"
        assert project["license-files"] == ["LICENSE"]
        # setuptools rejects a license classifier next to an SPDX license expression
        assert not [c for c in project["classifiers"] if c.startswith("License ::")]
        assert _read("LICENSE").startswith("MIT License\n")
        assert "MIT License. See [LICENSE](LICENSE)" in _read("README.md")
        assert "MIT License. See [LICENSE](../LICENSE)" in _read("docs/dev.md")


# ---------------------------------------------------------------------------
# Continuous integration
# ---------------------------------------------------------------------------


class TestContinuousIntegration:
    def test_workflow_runs_on_main_line_branch(self):
        """All development happens on ``migration``; CI must run for it."""
        workflow = _read(".github/workflows/tests.yml")
        for event in ("push", "pull_request"):
            match = re.search(rf"^  {event}:\n    branches: \[([^\]]*)\]", workflow, re.M)
            assert match is not None, f"no branch filter found for {event}"
            branches = [b.strip() for b in match.group(1).split(",")]
            assert "migration" in branches, f"{event} does not run on migration: {branches}"
