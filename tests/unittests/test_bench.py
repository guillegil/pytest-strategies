"""
Tests for the benchmark script, benchmarks/bench.py.

CI runs the script as an informational step that never fails the job, so these
tests keep it working: its cases generate rows on this version, its rejection
sweep rejects the share its labels say, it prints and writes every measurement,
and its memory measurement collects a project in a fresh interpreter and reads the
peak RSS in KiB on every Unix system. The timings themselves are not checked here:
they depend on the machine.
"""

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pytest_strategy
from pytest_strategy import RNG

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks" / "bench.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def bench():
    """The script, imported as a module (its main() runs only as a script)."""
    spec = importlib.util.spec_from_file_location("bench", BENCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCases:
    def test_each_case_generates_rows_of_the_arguments_its_label_says(self, bench):
        assert len(bench.CASES) == 3
        for label, n, make in bench.CASES:
            param = make()
            width = len(param.arg_names)
            assert label.startswith(f"{n:,} rows x {width} arguments"), label
            RNG.seed(1)
            rows = param.generate_vectors(20, mode="random_only")
            assert len(rows) == 20
            assert {len(row) for row in rows} == {width}

    def test_the_sweep_rejects_the_share_its_labels_say(self, bench):
        # Draws per accepted row: 1 at 0% rejection, about 2 at 50%, about 10 at 90%
        expected = {
            "~0% rejected": (1.0, 1.0),
            "~50% rejected": (1.8, 2.2),
            "~90% rejected": (9.0, 11.0),
        }
        assert [label for label, _ in bench.SWEEP] == list(expected)
        for label, make in bench.SWEEP:
            low, high = expected[label]
            assert low <= bench.draws_per_row(make(), 2_000) <= high, label

    def test_case_3_is_the_sweeps_50_percent(self, bench):
        """Case 3 and the sweep's middle rate time the same constraints."""
        case_3 = bench.CASES[2][2]()
        sweep_50 = bench.SWEEP[1][1]()
        assert list(case_3.vector_constraints) == list(sweep_50.vector_constraints)
        assert "~50% rejected" in bench.CASES[2][0]


class TestMain:
    def test_it_prints_and_writes_every_measurement(self, bench, monkeypatch, tmp_path, capsys):
        # Few rows, one run each, and a stand-in for the collection in a new interpreter
        monkeypatch.setattr(bench, "CASES", [(label, 50, make) for label, _, make in bench.CASES])
        monkeypatch.setattr(bench, "time_sweep", functools.partial(bench.time_sweep, n=100))
        collected = []

        def collect(n):
            collected.append(n)
            return {"rows": n, "seconds": 2.5 if n > 1 else 0.1, "peak_rss_mib": 30.0 + n / 1000}

        monkeypatch.setattr(bench, "collect", collect)
        out = tmp_path / "bench.json"
        bench.main(["--sweep", "--memory", "--repeat", "1", "--min-time", "0", "--json", str(out)])

        printed = capsys.readouterr().out
        lines = printed.splitlines()
        assert lines[0].startswith(f"pytest-strategies {pytest_strategy.__version__}, Python ")
        for label, _, _ in bench.CASES:
            assert re.search(rf"^  {re.escape(label)} +[\d.]+ s .* us/argument/row", printed, re.M)
        assert "Rejection sweep (per accepted row):" in lines
        for label, _ in bench.SWEEP:
            assert f"5,000 rows x 2 arguments, {label}" in printed
        assert "  100,000 rows    2.50 s  peak RSS 130.0 MiB, 100.0 MiB above one row" in lines
        # Three runs of each size, the smallest kept
        assert collected == [1, 1, 1, 100_000, 100_000, 100_000]

        results = json.loads(out.read_text(encoding="utf-8"))
        assert set(results) == {"header", "cases", "sweep", "memory"}
        assert results["header"] == lines[0]
        assert [case["rows"] for case in results["cases"]] == [50, 50, 50]
        assert [case["arguments"] for case in results["cases"]] == [5, 4, 2]
        assert "draws_per_row" in results["cases"][2]
        assert [entry["draws_per_row"] for entry in results["sweep"]][0] == 1.0
        assert results["memory"][1]["rss_above_one_row_mib"] == 100.0

    def test_without_options_it_times_only_the_cases(self, bench, monkeypatch, capsys):
        monkeypatch.setattr(bench, "CASES", [(label, 20, make) for label, _, make in bench.CASES])

        def no_call(*args, **kwargs):
            raise AssertionError("not asked for")

        monkeypatch.setattr(bench, "time_sweep", no_call)
        monkeypatch.setattr(bench, "measure_memory", no_call)
        bench.main(["--repeat", "1", "--min-time", "0"])

        assert len(capsys.readouterr().out.splitlines()) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="peak RSS is read on Linux and other Unix")
def test_collect_times_a_project_in_a_new_interpreter(bench):
    result = bench.collect(20)

    assert result["rows"] == 20
    assert result["seconds"] > 0
    assert result["peak_rss_mib"] > 0


@pytest.mark.skipif(sys.platform == "win32", reason="peak RSS is read on Linux and other Unix")
@pytest.mark.parametrize(("platform", "kib"), [("darwin", 2048), ("freebsd14", 2097152)])
def test_without_proc_the_peak_comes_from_ru_maxrss(bench, monkeypatch, platform, kib):
    """ru_maxrss is in bytes on macOS and in KiB elsewhere; the script reports KiB."""
    import resource

    namespace = {}
    exec(bench._MEASURE.split("\nstart = ")[0], namespace)

    def no_proc(*args, **kwargs):
        raise OSError("no /proc")

    namespace["open"] = no_proc
    namespace["sys"] = SimpleNamespace(platform=platform)
    usage = SimpleNamespace(ru_maxrss=2 * 1024 * 1024)
    monkeypatch.setattr(resource, "getrusage", lambda who: usage)

    assert namespace["peak_kib"]() == kib


class TestContinuousIntegration:
    def test_the_benchmark_step_never_fails_the_job(self):
        steps = re.split(r"\n      - ", WORKFLOW.read_text(encoding="utf-8"))
        (step,) = [step for step in steps if "benchmarks/bench.py" in step]

        assert "continue-on-error: true" in step
        assert "run: python benchmarks/bench.py --sweep --memory" in step

    def test_lint_checks_the_script(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        assert "ruff check src/ tests/ benchmarks/" in workflow
        assert "black --check src/ tests/ benchmarks/" in workflow
