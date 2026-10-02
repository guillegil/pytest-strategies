"""
Benchmark of row generation.

    python benchmarks/bench.py [--repeat 5] [--min-time 1.0] [--sweep] [--memory] [--json PATH]

Times ``Parameter.generate_vectors(n, mode="random_only")`` after ``RNG.seed(1)``,
the best of at least --repeat runs and --min-time seconds, for three cases:

  1. 10,000 rows of 5 arguments (RNGInteger(0, 1000) each);
  2. 100,000 rows of 4 arguments (RNGInteger, RNGFloat, RNGChoice, RNGBoolean);
  3. 5,000 rows of 2 arguments with 2 constraints at about 50% rejection.

--sweep adds case 3's arguments at about 0%, 50% and 90% rejection, per accepted
row: since streams v1 a rejected row redraws from where its arguments' streams
stopped, so the cost the streams add per accepted row should not grow with the
rejection rate. --memory adds the collection of 100,000 exhaustive rows (pytest
--collect-only --nsamples=auto on a generated project, in a fresh interpreter):
wall time and peak RSS (VmHWM on Linux, ru_maxrss on other Unix systems), the
smallest of three runs, next to a one-row project's.

pytest_strategy must be importable by the interpreter running this script: the
version under test. To compare two versions, run the script with each on the same
machine, one after the other, for example with PYTHONPATH=<other checkout>/src for
the other one; timings from different machines do not compare. CI runs it with
--sweep --memory as an informational step, which never fails the job. The API used
works unchanged on 3.0 and 4.0 (keyword-only options, named constraint functions
that index the row), so 3.0 can be the other version.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest_strategy
from pytest_strategy import (
    RNG,
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGFloat,
    RNGInteger,
    TestArg,
)

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def five_integers() -> Parameter:
    return Parameter(*[TestArg(f"a{i}", rng_type=RNGInteger(0, 1000)) for i in range(5)])


def four_types() -> Parameter:
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 2**32 - 1)),
        TestArg("ratio", rng_type=RNGFloat(0.0, 1.0)),
        TestArg("mode", rng_type=RNGChoice(["rd", "wr", "rmw", "nop"])),
        TestArg("burst", rng_type=RNGBoolean()),
    )


# Case 3: addr % 4 != 0 accepts 3/4 and size % 3 != 0 accepts 2/3 (size in 1..96),
# so about half the draws are rejected.
def unaligned(v: Sequence[int]) -> bool:
    return v[0] % 4 != 0


def size_not_x3(v: Sequence[int]) -> bool:
    return v[1] % 3 != 0


# The sweep's other rates: both constraints accept everything (0%), or addr % 10 == 0
# accepts 1/10 (90%).
def accept_addr(v: Sequence[int]) -> bool:
    return v[0] >= 0


def accept_size(v: Sequence[int]) -> bool:
    return v[1] >= 0


def addr_x10(v: Sequence[int]) -> bool:
    return v[0] % 10 == 0


# A constraint: a function of the row (a tuple in 3.0, a Vector since 4.0)
Constraint = Callable[[Any], object]


def two_constrained(constraints: list[Constraint], max_retries: int = 100) -> Parameter:
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 0xFFFF)),
        TestArg("size", rng_type=RNGInteger(1, 96)),
        vector_constraints=constraints,
        max_retries=max_retries,
    )


CASES: list[tuple[str, int, Callable[[], Parameter]]] = [
    ("10,000 rows x 5 arguments", 10_000, five_integers),
    ("100,000 rows x 4 arguments", 100_000, four_types),
    (
        "5,000 rows x 2 arguments, 2 constraints (~50% rejected)",
        5_000,
        lambda: two_constrained([unaligned, size_not_x3]),
    ),
]

SWEEP: list[tuple[str, Callable[[], Parameter]]] = [
    ("~0% rejected", lambda: two_constrained([accept_addr, accept_size])),
    ("~50% rejected", lambda: two_constrained([unaligned, size_not_x3])),
    # More retries: at 90% rejection, 100 draws fail for about 1 row in 40,000
    ("~90% rejected", lambda: two_constrained([addr_x10, accept_size], max_retries=1000)),
]

# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def best_of(param: Parameter, n: int, repeat: int, min_time: float) -> float:
    """Best wall time of generating n random rows, in seconds, over at least repeat runs
    and min_time seconds (short cases run more often, which steadies their best time)."""
    times: list[float] = []
    while len(times) < repeat or sum(times) < min_time:
        gc.collect()
        RNG.seed(1)
        start = time.perf_counter()
        rows = param.generate_vectors(n, mode="random_only")
        times.append(time.perf_counter() - start)
        assert len(rows) == n, (len(rows), n)
    return min(times)


def constraint_functions(param: Parameter) -> list[Constraint]:
    """The constraint functions, from a 3.0 list or a 4.0 mapping of names to functions."""
    constraints = param.vector_constraints
    if isinstance(constraints, list):
        return constraints
    return [constraints[name] for name in constraints]


def draws_per_row(param: Parameter, n: int) -> float:
    """Average draws per accepted row of a case-3 Parameter: its first constraint sees every draw."""
    first, *others = constraint_functions(param)
    calls = 0

    def counting(v: Any) -> object:
        nonlocal calls
        calls += 1
        return first(v)

    RNG.seed(1)
    two_constrained([counting, *others], param.max_retries).generate_vectors(n, mode="random_only")
    return calls / n


def time_cases(repeat: int, min_time: float) -> list[dict[str, Any]]:
    results = []
    for label, n, make in CASES:
        param = make()
        seconds = best_of(param, n, repeat, min_time)
        nargs = len(param.test_args)
        entry = {
            "case": label,
            "rows": n,
            "arguments": nargs,
            "seconds": round(seconds, 4),
            "us_per_row": round(seconds / n * 1e6, 2),
            "us_per_argument_row": round(seconds / n / nargs * 1e6, 3),
        }
        if param.vector_constraints:
            entry["draws_per_row"] = round(draws_per_row(param, n), 3)
        results.append(entry)
    return results


def time_sweep(repeat: int, min_time: float, n: int = 5_000) -> list[dict[str, Any]]:
    results = []
    for label, make in SWEEP:
        param = make()
        seconds = best_of(param, n, repeat, min_time)
        results.append(
            {
                "case": f"5,000 rows x 2 arguments, {label}",
                "rows": n,
                "draws_per_row": round(draws_per_row(param, n), 3),
                "seconds": round(seconds, 4),
                "us_per_accepted_row": round(seconds / n * 1e6, 2),
            }
        )
    return results


_PROJECT = """
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, strategy


def rows():
    return Parameter(
        TestArg("ch", rng_type=Series(range({n}))),
        TestArg("data", rng_type=RNGInteger(0, 255)),
    )


@strategy(rows)
def test_rows(ch, data):
    pass
"""

# Peak RSS of the collecting process: VmHWM belongs to the new process image, while
# ru_maxrss also counts the parent's memory when the child was forked from it
_MEASURE = """
import sys, time
import pytest


def peak_kib():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


start = time.perf_counter()
code = pytest.main(sys.argv[1:])
seconds = time.perf_counter() - start
print("BENCH", int(code), seconds, peak_kib())
"""


def collect(n: int) -> dict[str, Any]:
    """Collect one test with n exhaustive rows in a fresh interpreter: time and peak RSS."""
    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        project = Path(tmp)
        (project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (project / "test_rows.py").write_text(_PROJECT.format(n=n), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", _MEASURE, "-p", "no:cacheprovider", "--collect-only", "-q"]
            + ["--nsamples=auto", "--rng-seed=1", "-p", "no:xdist", "test_rows.py"],
            cwd=project,
            capture_output=True,
            text=True,
        )
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("BENCH ")]
    if result.returncode != 0 or not line:
        raise RuntimeError(f"collection of {n} rows failed:\n{result.stdout}{result.stderr}")
    code, seconds, maxrss_kib = line[-1].split()[1:]
    if int(code) != 0:
        raise RuntimeError(f"collection of {n} rows exited with {code}:\n{result.stdout}")
    return {"rows": n, "seconds": round(float(seconds), 2), "peak_rss_mib": int(maxrss_kib) / 1024}


def measure_memory(repeat: int = 3) -> list[dict[str, Any]]:
    """Collect 1 and 100,000 exhaustive rows, keeping the smallest time and peak RSS of each."""
    results = []
    for n in (1, 100_000):
        runs = [collect(n) for _ in range(repeat)]
        results.append(
            {
                "rows": n,
                "seconds": min(r["seconds"] for r in runs),
                "peak_rss_mib": round(min(r["peak_rss_mib"] for r in runs), 1),
            }
        )
    results[1]["rss_above_one_row_mib"] = round(
        results[1]["peak_rss_mib"] - results[0]["peak_rss_mib"], 1
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=5, help="least runs per case (best is kept)")
    parser.add_argument("--min-time", type=float, default=1.0, help="least seconds per case")
    parser.add_argument("--sweep", action="store_true", help="add the rejection-rate sweep")
    parser.add_argument("--memory", action="store_true", help="add 100,000 exhaustive rows")
    parser.add_argument("--json", type=Path, help="also write the results to this file")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    header = (
        f"pytest-strategies {pytest_strategy.__version__}, Python {platform.python_version()} "
        f"({platform.python_implementation()}), {platform.system()} {platform.machine()}, "
        f"best of {args.repeat}+ runs and {args.min_time:g}+ s per case"
    )
    print(header)
    results: dict[str, Any] = {"header": header, "cases": time_cases(args.repeat, args.min_time)}
    for entry in results["cases"]:
        extra = f", {entry['draws_per_row']} draws per row" if "draws_per_row" in entry else ""
        print(
            f"  {entry['case']:58s} {entry['seconds']:8.3f} s  {entry['us_per_row']:7.2f} us/row  "
            f"{entry['us_per_argument_row']:6.3f} us/argument/row{extra}"
        )
    if args.sweep:
        results["sweep"] = time_sweep(args.repeat, args.min_time)
        print("Rejection sweep (per accepted row):")
        for entry in results["sweep"]:
            print(
                f"  {entry['case']:58s} {entry['seconds']:8.3f} s  "
                f"{entry['us_per_accepted_row']:7.2f} us/row  {entry['draws_per_row']} draws per row"
            )
    if args.memory:
        results["memory"] = measure_memory()
        print("Collection with --nsamples=auto (Series ch x drawn data; smallest of 3 runs):")
        for entry in results["memory"]:
            extra = (
                f", {entry['rss_above_one_row_mib']} MiB above one row"
                if "rss_above_one_row_mib" in entry
                else ""
            )
            print(
                f"  {entry['rows']:>7,} rows  {entry['seconds']:6.2f} s  "
                f"peak RSS {entry['peak_rss_mib']} MiB{extra}"
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
