"""
End-to-end tests for row-stable streams (streams v1, D5), run through pytester: a
row's node ID names the same values whatever --nsamples, a node ID run alone gets
the values of the full run, and turning a constraint off gives the values the
strategy has without it. Runs are compared by their values (DUMP_VALUES), row by
row through the node IDs, which do not depend on the seed.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import ast

import pytest

pytest_plugins = ["pytester"]


STRATEGIES = """
    import os

    from pytest_strategy import (
        Parameter, RNGChoice, RNGFloat, RNGInteger, RNGSequence, Series, TestArg, register,
    )

    @register("rs_plain")
    def plain():
        # RS_CONSTRAINT adds a constraint that rejects about three draws in four
        constraints = {"aligned": lambda v: v.addr % 4 == 0} if os.environ.get("RS_CONSTRAINT") else None
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 2**32 - 1)),
            TestArg("ratio", rng_type=RNGFloat(0.0, 1.0)),
            TestArg("op", rng_type=RNGChoice(["rd", "wr", "rmw"])),
            vector_constraints=constraints,
        )

    @register("rs_series")
    def series():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("data", rng_type=RNGInteger(0, 255)),
        )

    @register("rs_persample")
    def persample():
        return Parameter(
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            TestArg("ch", rng_type=Series([0, 1])),
            TestArg("data", rng_type=RNGInteger(0, 255)),
            per_sequence_samples=True,
        )

    @register("rs_auto")
    def auto():
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("dev", rng_type=RNGSequence(["a", "b", "c"])),
            TestArg("data", rng_type=RNGInteger(0, 255)),
        )
"""

TESTS = """
    from pytest_strategy import strategy

    @strategy("rs_plain")
    def test_plain(addr, ratio, op):
        pass

    @strategy("rs_series")
    def test_series(ch, data):
        pass

    @strategy("rs_persample")
    def test_persample(dev, ch, data):
        pass

    @strategy("rs_auto")
    def test_auto(ch, dev, data):
        pass
"""


@pytest.fixture
def project(pytester, values_dump):
    pytester.makepyfile(strategies=STRATEGIES, test_rs=TESTS)
    pytester.makeconftest(values_dump.conftest)
    return values_dump


def by_node(rows):
    """Return the collected rows' values by node ID."""
    return dict(rows)


def test_more_rows_keep_the_first_ones(project):
    ten = by_node(
        project.collect("--collect-only", "--rng-seed=7", "--nsamples=10", subprocess=False)
    )
    fifty = by_node(
        project.collect("--collect-only", "--rng-seed=7", "--nsamples=50", subprocess=False)
    )

    counts = {
        test: sum(nodeid.startswith(f"test_rs.py::{test}[") for nodeid in ten)
        for test in ("test_plain", "test_series", "test_persample", "test_auto")
    }
    # per_sequence_samples gives 10 rows to each of its 4 combinations
    assert counts == {"test_plain": 10, "test_series": 10, "test_persample": 40, "test_auto": 10}
    assert len(fifty) == 5 * len(ten)
    for nodeid, values in ten.items():
        assert fifty[nodeid] == values, nodeid


@pytest.mark.parametrize(
    ("nodeid", "options"),
    [
        ("test_rs.py::test_plain[rand-3]", []),
        ("test_rs.py::test_series[ch=1-rand-2]", []),
        ("test_rs.py::test_persample[dev=b-ch=0-rand-7]", []),
        ("test_rs.py::test_auto[ch=0-rand-3]", []),
        ("test_rs.py::test_auto[ch=2-dev=b]", ["--nsamples=auto"]),
    ],
    ids=["plain", "series", "per_sequence", "drawn_sequence", "auto"],
)
def test_a_node_id_run_alone_gets_the_values_of_the_full_run(project, nodeid, options):
    full = by_node(project.collect("--collect-only", "--rng-seed=11", *options, subprocess=False))

    alone = project.collect("--collect-only", "--rng-seed=11", *options, nodeid, subprocess=False)

    assert alone == [[nodeid, full[nodeid]]]


def test_a_constraint_turned_off_gives_the_values_without_it(project, monkeypatch):
    def plain_rows(*options):
        rows = project.collect("--collect-only", "--rng-seed=3", "test_rs.py::test_plain", *options)
        return {nodeid: ast.literal_eval(values) for nodeid, values in rows}

    base = plain_rows()
    monkeypatch.setenv("RS_CONSTRAINT", "1")
    constrained = plain_rows()
    turned_off = plain_rows("--strategy-constraint-off=aligned")

    assert turned_off == base
    assert constrained.keys() == base.keys()
    kept = 0
    for nodeid, before in base.items():
        after = constrained[nodeid]
        assert after["addr"] % 4 == 0
        if before["addr"] % 4 == 0:
            assert after == before
            kept += 1
        else:
            assert after != before
    assert 0 < kept < len(base)


def test_inherited_methods_get_rows_of_their_own(pytester, values_dump):
    pytester.makepyfile(
        strategies=STRATEGIES,
        test_inherit="""
            from pytest_strategy import strategy

            class Base:
                @strategy("rs_series")
                def test_inherited(self, ch, data):
                    pass

            class TestA(Base):
                pass

            class TestB(Base):
                pass
            """,
    )
    pytester.makeconftest(values_dump.conftest)

    rows = values_dump.collect("--collect-only", "--rng-seed=5", subprocess=False)

    values = {nodeid.split("::")[1]: [] for nodeid, _ in rows}
    for nodeid, params in rows:
        values[nodeid.split("::")[1]].append(params)
    assert set(values) == {"TestA", "TestB"}
    assert values["TestA"] != values["TestB"]
    assert values_dump.collect("--collect-only", "--rng-seed=5", subprocess=False) == rows
