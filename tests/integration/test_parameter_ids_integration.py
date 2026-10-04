"""
End-to-end tests for ``Parameter(ids=...)`` (D3), run through pytester: a callable
names the rows from their VectorInfo, its duplicates are suffixed (also under
strict_parametrization_ids) and the stashed VectorInfo has the final ID; a result
that is not a str or None, or an exception, fails collection; "names" and "values"
override the strategies_ids ini option; the skipped row keeps its ID; and the
values do not depend on ids=.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import json
import random

import pytest

from pytest_strategy import RNG, Strategy
from pytest_strategy._ids import make_unique_ids

pytest_plugins = ["pytester"]


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


# Writes [node ID, VectorInfo.id, kind, values] for each strategy row, in order
CONFTEST = """
import json

from pytest_strategy import VECTOR_KEY


def pytest_collection_modifyitems(config, items):
    rows = []
    for item in items:
        info = item.stash.get(VECTOR_KEY, None)
        if info is not None:
            rows.append([item.nodeid, info.id, info.kind, list(info.values)])
    (config.rootpath / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
"""

STRATEGIES = """
import os

import pytest
from pytest_strategy import Parameter, RNGInteger, Series, TestArg, register


def by_parity(info):
    # A random row is named by the parity of its address; the others keep their IDs
    if info.kind == "random":
        return "even" if info.values.addr % 2 == 0 else "odd"
    return None


IDS = {"none": None, "names": "names", "values": "values", "parity": by_parity}


@register("pi_burst")
def burst():
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 63)),
        TestArg("len", rng_type=RNGInteger(1, 16)),
        directed_vectors={
            "zeros": (0, 1),
            "marked": pytest.param(1, 2, marks=pytest.mark.xfail(strict=True)),
        },
        nsamples=6,
        ids=IDS[os.environ.get("PI_IDS", "parity")],
    )


@register("pi_same")
def same():
    return Parameter(
        TestArg("ch", rng_type=Series([0, 1])),
        TestArg("x", rng_type=RNGInteger(0, 9)),
        nsamples=4,
        ids=lambda info: "same",
    )


@register("pi_names")
def names():
    return Parameter(TestArg("x", value=1), directed_vectors={"one": (1,)}, nsamples=0, ids="names")


@register("pi_values")
def values():
    return Parameter(TestArg("x", value=1), directed_vectors={"one": (1,)}, nsamples=0, ids="values")


@register("pi_ini")
def ini():
    return Parameter(TestArg("x", value=1), directed_vectors={"one": (1,)}, nsamples=0)


def never(info):
    raise AssertionError("ids= was called for the skipped row")


@register("pi_empty")
def empty():
    return Parameter(
        TestArg("ch", rng_type=Series([], skip_if_empty="no channels")),
        TestArg("x", rng_type=RNGInteger(0, 9)),
        ids=never,
    )
"""

TESTS = """
from pytest_strategy import strategy


@strategy("pi_burst")
def test_burst(addr, len):
    assert (addr, len) != (1, 2)


@strategy("pi_same")
def test_same(ch, x):
    pass


@strategy("pi_names")
def test_names(x):
    pass


@strategy("pi_values")
def test_values(x):
    pass


@strategy("pi_ini")
def test_ini(x):
    pass


@strategy("pi_empty")
def test_empty(ch, x):
    pass
"""


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(pi_strategies=STRATEGIES, test_pi=TESTS)
    return pytester


def _rows(pytester, test):
    """Return the [row ID, VectorInfo.id, kind, values] of one test's rows, in order."""
    rows = json.loads((pytester.path / "rows.json").read_text(encoding="utf-8"))
    prefix = f"test_pi.py::{test}["
    return [
        [nodeid[len(prefix) : -1], *rest] for nodeid, *rest in rows if nodeid.startswith(prefix)
    ]


def _collect(pytester, *args):
    result = pytester.runpytest("-p", "no:cacheprovider", "--collect-only", "-q", *args)
    assert result.ret == 0, result.stdout.str()
    return result


def test_the_callable_names_the_rows(project):
    _collect(project, "--rng-seed=1")

    rows = _rows(project, "test_burst")
    parity = ["even" if values[0] % 2 == 0 else "odd" for _, _, _, values in rows[2:]]
    assert [row[0] for row in rows] == [
        "directed-zeros",
        "directed-marked",
        *make_unique_ids(parity),
    ]
    assert [row[2] for row in rows] == ["directed"] * 2 + ["random"] * 6
    # The stashed VectorInfo has the final ID, after the suffixes
    assert all(row_id == info_id for row_id, info_id, _, _ in rows)


def test_rows_keep_their_marks(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "test_pi.py::test_burst")

    result.assert_outcomes(passed=7, xfailed=1)


def test_duplicates_are_suffixed(project):
    _collect(project)

    rows = _rows(project, "test_same")
    assert [row[0] for row in rows] == ["same0", "same1", "same2", "same3"]
    assert [row[1] for row in rows] == ["same0", "same1", "same2", "same3"]


def test_duplicates_pass_under_strict_parametrization_ids(project, pytestconfig):
    try:
        pytestconfig.getini("strict_parametrization_ids")
    except ValueError:
        pytest.skip("this pytest has no strict_parametrization_ids option")

    result = project.runpytest(
        "-p", "no:cacheprovider", "-o", "strict_parametrization_ids=true", "-k", "same"
    )

    result.assert_outcomes(passed=4)


@pytest.mark.parametrize(
    ("ini", "expected"),
    [
        ((), ["directed-one", "x=1", "directed-one"]),
        (("-o", "strategies_ids=values"), ["directed-one", "x=1", "x=1"]),
    ],
    ids=["names", "values"],
)
def test_names_and_values_override_the_ini_option(project, ini, expected):
    """test_names has ids="names", test_values ids="values", and test_ini no ids=."""
    _collect(project, *ini)

    ids = [_rows(project, test)[0][0] for test in ("test_names", "test_values", "test_ini")]
    assert ids == expected


def test_the_callable_is_not_called_for_the_skipped_row(project):
    result = project.runpytest("-p", "no:cacheprovider", "-rs", "test_pi.py::test_empty")

    result.assert_outcomes(skipped=1)
    assert _rows(project, "test_empty") == [["skipped", "skipped", "skipped", [None, None]]]


def test_values_do_not_depend_on_ids(project, monkeypatch):
    runs = {}
    for ids in ("none", "names", "values", "parity"):
        monkeypatch.setenv("PI_IDS", ids)
        _collect(project, "--rng-seed=7", "test_pi.py::test_burst")
        runs[ids] = _rows(project, "test_burst")

    assert len({json.dumps([row[3] for row in rows]) for rows in runs.values()}) == 1
    assert runs["none"] == runs["names"]
    assert [row[0] for row in runs["values"]][:2] == ["addr=0,len=1", "addr=1,len=2"]
    assert [row[0] for row in runs["parity"]][2] in ("even0", "odd0")


class TestTheCallableFails:
    def test_when_it_returns_something_else(self, pytester):
        pytester.makepyfile(test_pi_bad="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @register("pi_bad")
            def bad():
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    nsamples=5,
                    ids=lambda info: 42 if info.index == 3 else None,
                )

            @strategy("pi_bad")
            def test_bad(x):
                pass
            """)

        result = pytester.runpytest("-p", "no:cacheprovider", "--collect-only", "-q")

        assert result.ret == 2
        result.stdout.fnmatch_lines(
            [
                "*In test_bad: Strategy 'pi_bad': ids= returned 42 for row rand-3; return a str or None"
            ]
        )

    def test_when_it_raises(self, pytester):
        pytester.makepyfile(test_pi_boom="""
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            def boom(info):
                return 1 / 0

            @register("pi_boom")
            def boom_factory():
                return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), nsamples=1, ids=boom)

            @strategy("pi_boom")
            def test_boom(x):
                pass
            """)

        result = pytester.runpytest("-p", "no:cacheprovider", "--collect-only", "-q")

        assert result.ret == 2
        result.stdout.fnmatch_lines(
            [
                "In test_boom: Strategy 'pi_boom': ids= raised ZeroDivisionError for row "
                "rand-0: division by zero",
                '*test_pi_boom.py", line 4, in boom',
                "*return 1 / 0",
                "ZeroDivisionError: division by zero",
            ]
        )
        # Only the callable's frames, not the plugin's
        assert "_resolver.py" not in result.stdout.str()
