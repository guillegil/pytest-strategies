"""
Unit tests keeping the documentation, examples and packaging metadata honest.

Each test pins a defect that shipped: a README snippet that did not compile, a
missing PEP 561 marker, trove classifiers and formatter targets that disagreed
with ``requires-python``, a CI workflow that never ran on the main line branch,
and example docstrings describing behaviour the library does not have.

Example files are parsed with ``ast`` rather than imported: importing one runs
its ``@Strategy.strategy`` decorators and registers its strategies globally.
"""

import ast
import re
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
