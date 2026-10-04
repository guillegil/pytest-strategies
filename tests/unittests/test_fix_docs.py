"""
Unit tests keeping the documentation, examples and packaging metadata honest.

Each test pins a defect that shipped: a README snippet that did not compile, a
missing PEP 561 marker, trove classifiers and formatter targets that disagreed
with ``requires-python``, a CI workflow that never ran on the main line branch,
and example docstrings describing behaviour the library does not have.

The APIs that 4.0 removed may appear in the README, docs/dev.md and the skill only in
their upgrade sections (D21). The CHANGELOG section of each major release from 4.0.0 on
starts with "### Migrating from", which names them, and the release notes that
release.yml cuts from the CHANGELOG include it.

Example files are parsed with ``ast`` rather than imported: importing one runs
its ``@Strategy.strategy`` decorators and registers its strategies globally.
"""

import ast
import os
import re
import subprocess
import sys
import textwrap
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

    def test_readme_installs_the_final_version(self, pyproject):
        """The README's install command pins the release tag, a fourth copy of the version."""
        version = pyproject["project"]["version"]
        pins = re.findall(r"pytest-strategies\.git@v([^\s\"'`]+)", _read("README.md"))
        assert pins, "README.md has no install command pinned to a release tag"
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            assert pins == [version] * len(pins)

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
# CHANGELOG
# ---------------------------------------------------------------------------

# The APIs each major release removed, which its "### Migrating from" subsection names
REMOVED_IN_MAJOR = {4: REMOVED_APIS}

# 3.0.0 and 2.0.0 have "Breaking changes and how to upgrade from ..." sections instead
FIRST_MAJOR_WITH_MIGRATION = 4

RELEASE_SECTION = re.compile(
    r"^## \[(\d+)\.(\d+)\.(\d+)\] - [^\n]*\n(.*?)(?=^## \[|^\[[^\]\n]+\]: |\Z)", re.M | re.S
)


def _major_sections(changelog: str) -> dict[str, str]:
    """The body of each ``## [X.0.0] - <date>`` section from 4.0.0 on, by version."""
    return {
        f"{major}.0.0": match.group(4)
        for match in RELEASE_SECTION.finditer(changelog)
        if (major := int(match.group(1))) >= FIRST_MAJOR_WITH_MIGRATION
        and match.group(2, 3) == ("0", "0")
    }


def _migration_subsection(section: str) -> str | None:
    """The text of a section's ``### Migrating from ...`` subsection, heading included."""
    match = re.search(r"^### Migrating from [^\n]*\n.*?(?=^### |\Z)", section, re.M | re.S)
    return match.group(0) if match else None


def _migration_problems(changelog: str) -> list[str]:
    """
    What the major sections from 4.0.0 on lack: a first subsection ``### Migrating from``
    that names each API the release removed, as ``REMOVED_IN_MAJOR`` lists them.
    """
    problems = []
    for version, section in _major_sections(changelog).items():
        migration = _migration_subsection(section)
        if migration is None:
            problems.append(f"{version}: no '### Migrating from' subsection")
            continue
        first = re.search(r"^### [^\n]*", section, re.M)
        if first is not None and not migration.startswith(first.group(0) + "\n"):
            problems.append(
                f"{version}: '### Migrating from' is not the first subsection "
                f"('{first.group(0)}' comes before it)"
            )
        removed = REMOVED_IN_MAJOR.get(int(version.split(".")[0]))
        if removed is None:
            problems.append(
                f"{version}: list the APIs {version} removed in REMOVED_IN_MAJOR "
                f"(tests/unittests/test_fix_docs.py), as patterns"
            )
            continue
        for api, pattern in removed.items():
            if not re.search(pattern, migration, re.S):
                problems.append(f"{version}: '### Migrating from' does not name {api}")
    return problems


def _release_notes_script() -> str:
    """The Python script of release.yml's "Release notes from the CHANGELOG" step."""
    workflow = _read(".github/workflows/release.yml")
    step = re.search(
        r"^( *)- name: Release notes from the CHANGELOG\n((?:\1  (?!run:)[^\n]*\n)*)"
        r"\1  run: \|\n((?:\1    [^\n]*\n|\n)+)",
        workflow,
        re.M,
    )
    assert step is not None, "release.yml has no 'Release notes from the CHANGELOG' run step"
    assert re.search(r"^ *shell: python$", step.group(2), re.M), step.group(2)
    return textwrap.dedent(step.group(3))


def _release_notes(changelog: str, version: str, tmp_path: Path) -> str:
    """The notes release.yml publishes for ``version``, from its own script."""
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", _release_notes_script()],
        cwd=tmp_path,
        env={**os.environ, "VERSION": version},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return (tmp_path / "notes.md").read_text(encoding="utf-8")


def _changelog(*sections: str) -> str:
    """A CHANGELOG made of ``sections``, with the 4.0.0 links at the end."""
    return "\n".join(
        [
            "# Changelog",
            "",
            "## [Unreleased]",
            "",
            *sections,
            "[Unreleased]: https://example.com/compare/v4.0.0...HEAD",
            "[4.0.0]: https://example.com/compare/v3.0.0...v4.0.0",
            "",
        ]
    )


# A 4.0.0 migration subsection naming every API that 4.0 removed
MIGRATING_FROM_3 = "\n".join(
    [
        "### Migrating from 3.x",
        "",
        "1. **Tuple factories are removed.** Return a `Parameter` instead of an"
        " `(argnames, samples)` tuple.",
        "2. **The other APIs that 3.0 deprecated are removed:**",
        "    - `TestArg(directed_values=..., test_values=..., always_include_directed=...)`:"
        " use `directed_vectors`;",
        "    - `RNG.set_max_retries()`: use `Parameter(max_retries=...)`;",
        "    - `configure()`: delete the call;",
        "    - `Strategy.set_config()`: delete the call.",
        "",
    ]
)


class TestChangelogMigration:
    """
    The section of each major release from 4.0.0 on starts with "### Migrating from",
    which names every API the release removed (D21), and the GitHub Release notes,
    which release.yml cuts from the CHANGELOG, include it.
    """

    def test_major_sections_have_a_migration_subsection_naming_the_removed_apis(self):
        problems = _migration_problems(_read("CHANGELOG.md"))
        assert not problems, "CHANGELOG.md:\n" + "\n".join(f"  {p}" for p in problems)

    def test_the_4_0_0_section_is_checked(self):
        """The check is not vacuous: 4.0.0 is a major section from 4.0.0 on."""
        sections = _major_sections(_read("CHANGELOG.md"))
        assert "4.0.0" in sections
        assert _migration_subsection(sections["4.0.0"]).startswith("### Migrating from 3.x\n")

    def test_release_notes_contain_the_migration_subsection(self, tmp_path):
        """release.yml's notes regex keeps the ### subsection in the 4.0.0 section."""
        changelog = _read("CHANGELOG.md")
        sections = _major_sections(changelog)
        assert sections
        for version, section in sections.items():
            notes = _release_notes(changelog, version, tmp_path)
            assert notes == section.strip() + "\n"
            assert _migration_subsection(section).strip() in notes
            assert not re.search(r"^## \[|^\[[^\]\n]+\]: ", notes, re.M)
        assert _migration_subsection(_release_notes(changelog, "4.0.0", tmp_path))

    def test_a_section_without_the_subsection_fails(self):
        changelog = _changelog(
            "## [4.0.0] - 2026-10-03",
            "",
            "### Breaking changes and how to upgrade from 3.0.0",
            "",
            MIGRATING_FROM_3.split("\n", 1)[1],
        )
        assert _migration_problems(changelog) == ["4.0.0: no '### Migrating from' subsection"]

    def test_a_removed_api_named_only_outside_the_subsection_fails(self):
        migration = MIGRATING_FROM_3.replace("    - `configure()`: delete the call;\n", "")
        changelog = _changelog(
            "## [4.0.0] - 2026-10-03",
            "",
            migration,
            "### Removed",
            "- `configure()`, which did nothing.",
            "",
        )
        assert _migration_problems(changelog) == [
            "4.0.0: '### Migrating from' does not name configure()"
        ]

    def test_the_subsection_must_come_first(self):
        changelog = _changelog(
            "## [4.0.0] - 2026-10-03",
            "",
            "### Added",
            "- `Vector`.",
            "",
            MIGRATING_FROM_3,
        )
        assert _migration_problems(changelog) == [
            "4.0.0: '### Migrating from' is not the first subsection ('### Added' comes before it)"
        ]

    def test_a_major_without_its_removed_apis_fails(self):
        changelog = _changelog(
            "## [5.0.0] - 2027-01-01",
            "",
            "### Migrating from 4.x",
            "",
            "1. Nothing to do.",
            "",
            "## [4.0.0] - 2026-10-03",
            "",
            MIGRATING_FROM_3,
        )
        assert _migration_problems(changelog) == [
            "5.0.0: list the APIs 5.0.0 removed in REMOVED_IN_MAJOR "
            "(tests/unittests/test_fix_docs.py), as patterns"
        ]

    def test_earlier_majors_minors_and_patches_are_not_checked(self):
        changelog = _changelog(
            "## [4.1.0] - 2026-11-01",
            "",
            "### Added",
            "- Dependent arguments.",
            "",
            "## [4.0.1] - 2026-10-10",
            "",
            "### Fixed",
            "- A typo.",
            "",
            "## [4.0.0] - 2026-10-03",
            "",
            MIGRATING_FROM_3,
            "## [3.0.0] - 2026-09-30",
            "",
            "### Breaking changes and how to upgrade from 2.0.0",
            "",
            "## [2.0.0] - 2026-09-30",
            "",
            "### Changed",
            "",
        )
        assert list(_major_sections(changelog)) == ["4.0.0"]
        assert _migration_problems(changelog) == []

    def test_the_release_notes_script_is_the_workflow_step(self, tmp_path):
        """The test runs release.yml's own step, which stops at the next version or link."""
        script = _release_notes_script()
        assert script.startswith("import os, pathlib, re, sys\n")
        assert 'pathlib.Path("notes.md").write_text(' in script
        changelog = _changelog("## [4.0.0] - 2026-10-03", "", "Summary.", "", MIGRATING_FROM_3)
        notes = _release_notes(changelog, "4.0.0", tmp_path)
        assert notes == "Summary.\n\n" + MIGRATING_FROM_3.strip() + "\n"


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
