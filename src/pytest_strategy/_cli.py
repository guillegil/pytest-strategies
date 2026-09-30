"""
The ``pytest-strategies`` command (also ``python -m pytest_strategy``).

``pytest-strategies skill install`` copies the agent skill shipped in this package
(``skill/pytest-strategies/``) into the folders coding agents read skills from:
``.claude/skills/`` for Claude and ``.agents/skills/`` for Codex and other agents,
in the current directory or, with ``--global``, in the user's home folders.
"""

import argparse
import os
import shutil
import sys
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from . import __version__

PROG = "pytest-strategies"
SKILL_NAME = "pytest-strategies"

# Install targets in the order they are installed and reported
CLAUDE = "claude"
AGENTS = "agents"
ALL = "all"
TARGETS: dict[str, tuple[str, ...]] = {CLAUDE: (CLAUDE,), AGENTS: (AGENTS,), ALL: (CLAUDE, AGENTS)}

# The folder of each target, under the current directory or the home folder
_FOLDERS = {CLAUDE: ".claude", AGENTS: ".agents"}

# Names that never belong in an installed skill (left by tools in a source checkout)
_IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store"})


def skill_source() -> Traversable:
    """Return the skill folder shipped in the package, read through importlib.resources."""
    return resources.files("pytest_strategy") / "skill" / SKILL_NAME


def skills_dir(target: str, *, use_global: bool) -> Path:
    """
    Return the folder that holds the skills of an install target.

    Project installs go under the current directory. Global installs use
    ``$CLAUDE_CONFIG_DIR/skills`` (when the variable is set and not empty) or
    ``~/.claude/skills`` for Claude, and ``~/.agents/skills`` for other agents.
    """
    if not use_global:
        return Path.cwd() / _FOLDERS[target] / "skills"
    if target == CLAUDE:
        config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
        if config_dir.strip():
            return Path(config_dir).expanduser() / "skills"
    return Path.home() / _FOLDERS[target] / "skills"


def _remove(path: Path) -> None:
    """Remove a file, a symlink (not its target) or a folder tree, if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_tree(source: Traversable, dest: Path) -> None:
    """Copy a resource folder into dest, file by file (never as symlinks)."""
    dest.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir(), key=lambda entry: entry.name):
        if child.name in _IGNORED_NAMES:
            continue
        if child.is_dir():
            _copy_tree(child, dest / child.name)
        else:
            (dest / child.name).write_bytes(child.read_bytes())


def install_skill(skills_root: Path) -> Path:
    """
    Install the skill as ``skills_root / "pytest-strategies"`` and return that path.

    An existing ``pytest-strategies`` folder is removed first, so the installed skill
    always matches this version of the library. Other skills are left alone.
    """
    dest = skills_root / SKILL_NAME
    _remove(dest)
    _copy_tree(skill_source(), dest)
    return dest


def _skill_install(args: argparse.Namespace) -> int:
    """Handle ``pytest-strategies skill install``."""
    source = skill_source()
    if not source.is_dir() or not (source / "SKILL.md").is_file():
        print(
            f"{PROG}: error: the skill files are missing from this installation of "
            f"pytest-strategies ({source}); reinstall the package",
            file=sys.stderr,
        )
        return 1

    for target in TARGETS[args.target]:
        root = skills_dir(target, use_global=args.use_global)
        try:
            dest = install_skill(root)
        except OSError as error:
            print(
                f"{PROG}: error: could not install the skill in {root / SKILL_NAME}: {error}",
                file=sys.stderr,
            )
            return 1
        print(f"Installed the {SKILL_NAME} skill in {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``pytest-strategies`` command."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Tools for the pytest-strategies pytest plugin.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    skill = commands.add_parser(
        "skill",
        help="manage the agent skill for coding agents",
        description="Manage the pytest-strategies agent skill for coding agents.",
    )
    skill_commands = skill.add_subparsers(dest="skill_command", metavar="COMMAND", required=True)

    install = skill_commands.add_parser(
        "install",
        help="copy the skill into the agents' skills folders",
        description=(
            "Copy the pytest-strategies skill into .claude/skills/ (Claude) and "
            ".agents/skills/ (Codex and other agents). A previous copy of the "
            "pytest-strategies skill is replaced; other skills are left alone."
        ),
    )
    target = install.add_mutually_exclusive_group()
    target.add_argument(
        "--claude",
        dest="target",
        action="store_const",
        const=CLAUDE,
        help="install for Claude, in .claude/skills/",
    )
    target.add_argument(
        "--agents",
        "--generic",
        dest="target",
        action="store_const",
        const=AGENTS,
        help="install for Codex and other agents, in .agents/skills/",
    )
    target.add_argument(
        "--all",
        dest="target",
        action="store_const",
        const=ALL,
        help="install for both (the default)",
    )
    install.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help=(
            "install in the home folders instead of the current directory: "
            "$CLAUDE_CONFIG_DIR/skills (or ~/.claude/skills) and ~/.agents/skills"
        ),
    )
    install.set_defaults(target=ALL, handler=_skill_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run the ``pytest-strategies`` command and return its exit code.

    Returns 0 on success and 1 when files cannot be written. Usage errors exit with
    code 2 through argparse (``SystemExit``).
    """
    args = build_parser().parse_args(argv)
    code: int = args.handler(args)
    return code
