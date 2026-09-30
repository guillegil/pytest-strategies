"""
Unit tests for the agent skill and ``pytest-strategies skill install``.

Every test runs in a temporary project folder with a fake home (``HOME``,
``USERPROFILE`` and ``Path.home()``), so nothing is written to the real home or
to the repository. Paths are built with pathlib and no symlinks are used, so the
tests run on Windows too.
"""

import ast
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from types import SimpleNamespace

import pytest

import pytest_strategy
from pytest_strategy import _cli

SKILL = "pytest-strategies"

# The library version the skill documents.
# TODO(3.0.0 version bump): compare against pytest_strategy.__version__ instead of
# this constant once __version__ is "3.0.0", and delete SKILL_VERSION.
SKILL_VERSION = "3.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "src" / "pytest_strategy"

# Each target flag (None: no flag) and the targets it installs, in install order
TARGET_FLAGS: dict[str | None, list[str]] = {
    "--claude": ["claude"],
    "--agents": ["agents"],
    "--generic": ["agents"],
    "--all": ["claude", "agents"],
    None: ["claude", "agents"],
}

INSTALLED_PREFIX = f"Installed the {SKILL} skill in "


def _read_tree(node: Traversable, prefix: str = "") -> dict[str, bytes]:
    """Return {posix relative path: bytes} for every file below a resource folder."""
    files: dict[str, bytes] = {}
    for child in node.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir():
            files.update(_read_tree(child, f"{prefix}{child.name}/"))
        else:
            files[f"{prefix}{child.name}"] = child.read_bytes()
    return files


def _installed(folder: Path) -> dict[str, bytes]:
    """Return {posix relative path: bytes} for every file below an installed folder."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def _frontmatter(text: str) -> dict[str, str]:
    """
    Parse the YAML frontmatter of a SKILL.md made of one-line ``key: value`` pairs.

    The skill keeps its frontmatter to plain one-line scalars, so this needs no YAML
    library; anything else (a nested key, a multi-line value) fails the test.
    """
    lines = text.splitlines()
    assert lines[0] == "---", "SKILL.md must start with a '---' frontmatter line"
    end = lines.index("---", 1)
    data: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        assert sep and key and not key[0].isspace(), f"not a 'key: value' line: {line!r}"
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        data[key] = value
    return data


@pytest.fixture(scope="module")
def packaged() -> dict[str, bytes]:
    """The skill files as the installed package ships them."""
    return _read_tree(resources.files("pytest_strategy") / "skill" / SKILL)


@pytest.fixture(scope="module")
def skill_md(packaged) -> str:
    return packaged["SKILL.md"].decode("utf-8")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A temporary project (the current directory) and a fake, empty home folder."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.chdir(project)
    return SimpleNamespace(home=home, project=project, tmp=tmp_path)


def _dest(dirs, target: str, use_global: bool) -> Path:
    """The expected install folder of a target (without CLAUDE_CONFIG_DIR)."""
    base = dirs.home if use_global else dirs.project
    return base / f".{target}" / "skills" / SKILL


def _reported(out: str) -> list[str]:
    """The paths from the 'Installed ...' lines of the command's output."""
    lines = out.splitlines()
    assert all(line.startswith(INSTALLED_PREFIX) for line in lines), out
    return [line[len(INSTALLED_PREFIX) :] for line in lines]


# ---------------------------------------------------------------------------
# The packaged skill
# ---------------------------------------------------------------------------


class TestPackagedSkill:
    def test_skill_files_are_packaged(self, packaged):
        assert {"SKILL.md", "references/api.md"} <= set(packaged)

    def test_frontmatter_name_and_description(self, skill_md):
        meta = _frontmatter(skill_md)
        assert meta["name"] == SKILL
        # Agent Skills: lowercase letters, digits and hyphens, at most 64 characters
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
        assert len(meta["name"]) <= 64
        description = meta["description"]
        assert description.strip()
        assert len(description) < 1024
        assert "<" not in description and ">" not in description
        allowed = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
        assert set(meta) <= allowed

    def test_frontmatter_is_valid_yaml(self, skill_md):
        """Cross-check the simple parser with PyYAML when it is available."""
        yaml = pytest.importorskip("yaml")
        raw = skill_md.split("\n---", 1)[0].removeprefix("---\n")
        assert yaml.safe_load(raw) == _frontmatter(skill_md)

    def test_skill_documents_the_release_version(self, skill_md):
        # TODO(3.0.0 version bump): use pytest_strategy.__version__ here
        assert f"Documents pytest-strategies {SKILL_VERSION}." in skill_md

    def test_referenced_files_exist(self, packaged, skill_md):
        referenced = set(re.findall(r"references/[\w.-]+\.md", skill_md))
        assert referenced, "SKILL.md should point to its reference files"
        assert referenced <= set(packaged)

    @pytest.mark.parametrize("name", ["SKILL.md", "references/api.md"])
    def test_python_blocks_parse(self, name, packaged):
        """Code an agent copies from the skill must at least be valid Python."""
        text = packaged[name].decode("utf-8").replace("\r\n", "\n")
        blocks = re.findall(r"^( *)```python\n(.*?)^\1```", text, re.S | re.M)
        assert blocks
        for _indent, code in blocks:
            ast.parse(textwrap.dedent(code))

    def test_skill_body_is_concise(self, skill_md):
        """Agent Skills guidance: SKILL.md under 500 lines, details in references/."""
        assert len(skill_md.splitlines()) < 500


@pytest.fixture(scope="module")
def pyproject():
    """Parsed pyproject.toml."""
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestPackagingMetadata:
    def test_console_script(self, pyproject):
        assert pyproject["project"]["scripts"]["pytest-strategies"] == "pytest_strategy._cli:main"

    def test_package_data_covers_every_skill_file(self, pyproject):
        patterns = pyproject["tool"]["setuptools"]["package-data"]["pytest_strategy"]
        matched = {path for pattern in patterns for path in PACKAGE_DIR.glob(pattern)}
        skill_files = {
            path
            for path in (PACKAGE_DIR / "skill").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        assert skill_files
        assert skill_files <= matched, sorted(str(p) for p in skill_files - matched)


# ---------------------------------------------------------------------------
# skill install
# ---------------------------------------------------------------------------


class TestSkillInstall:
    @pytest.mark.parametrize("use_global", [False, True], ids=["project", "global"])
    @pytest.mark.parametrize("flag", list(TARGET_FLAGS), ids=lambda flag: flag or "no-flag")
    def test_flag_combinations(self, flag, use_global, dirs, packaged, capsys):
        argv = ["skill", "install"]
        argv += [flag] if flag else []
        argv += ["--global"] if use_global else []

        assert _cli.main(argv) == 0

        wanted = [_dest(dirs, target, use_global) for target in TARGET_FLAGS[flag]]
        for folder in wanted:
            assert _installed(folder) == packaged
        for target in {"claude", "agents"} - set(TARGET_FLAGS[flag]):
            assert not _dest(dirs, target, use_global).parent.parent.exists()
        # Nothing is written to the other location (home for a project install)
        untouched = dirs.project if use_global else dirs.home
        assert list(untouched.iterdir()) == []

        reported = _reported(capsys.readouterr().out)
        assert len(reported) == len(wanted)
        for path, folder in zip(reported, wanted):
            assert Path(path).samefile(folder)

    def test_global_claude_honors_claude_config_dir(self, dirs, packaged, monkeypatch):
        config_dir = dirs.tmp / "claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        assert _cli.main(["skill", "install", "--global"]) == 0

        assert _installed(config_dir / "skills" / SKILL) == packaged
        assert not (dirs.home / ".claude").exists()
        # Other agents still use the home folder
        assert _installed(_dest(dirs, "agents", use_global=True)) == packaged

    def test_empty_claude_config_dir_falls_back_to_home(self, dirs, packaged, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")

        assert _cli.main(["skill", "install", "--claude", "--global"]) == 0

        assert _installed(_dest(dirs, "claude", use_global=True)) == packaged

    def test_project_install_ignores_claude_config_dir(self, dirs, packaged, monkeypatch):
        config_dir = dirs.tmp / "claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        assert _cli.main(["skill", "install", "--claude"]) == 0

        assert _installed(_dest(dirs, "claude", use_global=False)) == packaged
        assert not config_dir.exists()

    @pytest.mark.parametrize("use_global", [False, True], ids=["project", "global"])
    def test_reinstall_replaces_old_copy_and_keeps_siblings(
        self, use_global, dirs, packaged, capsys
    ):
        dest = _dest(dirs, "claude", use_global)
        (dest / "references").mkdir(parents=True)
        (dest / "SKILL.md").write_text("an older version", encoding="utf-8")
        (dest / "references" / "stale.md").write_text("gone after reinstall", encoding="utf-8")
        (dest / "obsolete").mkdir()
        sibling = dest.parent / "other-skill"
        sibling.mkdir()
        (sibling / "SKILL.md").write_text("keep me", encoding="utf-8")

        argv = ["skill", "install", "--claude"] + (["--global"] if use_global else [])
        assert _cli.main(argv) == 0
        assert _installed(dest) == packaged
        assert not (dest / "obsolete").exists()
        assert _installed(sibling) == {"SKILL.md": b"keep me"}

        # Installing again gives the same result
        assert _cli.main(argv) == 0
        assert _installed(dest) == packaged
        assert _installed(sibling) == {"SKILL.md": b"keep me"}
        assert sorted(p.name for p in dest.parent.iterdir()) == ["other-skill", SKILL]
        capsys.readouterr()

    def test_reinstall_replaces_a_file_in_the_way(self, dirs, packaged):
        dest = _dest(dirs, "agents", use_global=False)
        dest.parent.mkdir(parents=True)
        dest.write_text("not a folder", encoding="utf-8")

        assert _cli.main(["skill", "install", "--agents"]) == 0

        assert _installed(dest) == packaged

    @pytest.mark.parametrize(
        "flags",
        [
            ["--claude", "--agents"],
            ["--claude", "--generic"],
            ["--all", "--claude"],
            ["--agents", "--all"],
            ["--generic", "--all", "--global"],
        ],
        ids=lambda flags: " ".join(flags),
    )
    def test_conflicting_target_flags_are_a_usage_error(self, flags, dirs, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _cli.main(["skill", "install", *flags])

        assert exc_info.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err
        assert list(dirs.project.iterdir()) == []
        assert list(dirs.home.iterdir()) == []

    @pytest.mark.parametrize(
        "argv",
        [[], ["skill"], ["skill", "uninstall"], ["skill", "install", "--codex"], ["other"]],
        ids=["no-command", "no-skill-command", "unknown-skill-command", "unknown-flag", "other"],
    )
    def test_other_usage_errors_exit_2(self, argv, dirs, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _cli.main(argv)

        assert exc_info.value.code == 2
        assert capsys.readouterr().err
        assert list(dirs.project.iterdir()) == []

    def test_write_error_exits_1_with_a_message(self, dirs, capsys):
        # A file where the .claude folder should be makes creating the folder fail
        (dirs.project / ".claude").write_text("", encoding="utf-8")

        assert _cli.main(["skill", "install", "--claude"]) == 1

        err = capsys.readouterr().err
        assert "could not install the skill in" in err
        assert str(Path.cwd() / ".claude" / "skills" / SKILL) in err

    def test_missing_skill_files_exit_1(self, dirs, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr(_cli, "skill_source", lambda: tmp_path / "nowhere")

        assert _cli.main(["skill", "install"]) == 1

        assert "skill files are missing" in capsys.readouterr().err
        assert list(dirs.project.iterdir()) == []

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _cli.main(["--version"])

        assert exc_info.value.code == 0
        assert capsys.readouterr().out.strip() == f"pytest-strategies {pytest_strategy.__version__}"


def test_python_dash_m_installs_the_skill(tmp_path, packaged):
    """``python -m pytest_strategy skill install --claude`` in a real subprocess."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    env = dict(os.environ)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    # Import the same pytest_strategy as this test process, wherever it is installed
    package_parent = str(Path(pytest_strategy.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (package_parent, env.get("PYTHONPATH")) if path
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest_strategy", "skill", "install", "--claude"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    dest = project / ".claude" / "skills" / SKILL
    assert _installed(dest) == packaged
    reported = _reported(result.stdout)
    assert len(reported) == 1
    assert Path(reported[0]).samefile(dest)
    assert list(home.iterdir()) == []
    assert not (project / ".agents").exists()
