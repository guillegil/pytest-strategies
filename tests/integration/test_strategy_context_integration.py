"""
End-to-end tests for the pytest_strategies_context hook through real pytest sessions.

The project mirrors a testbench whose configuration file is named on the command
line: the strategy draws channels from the Esm peripherals the file lists.
"""

import json
import random

import pytest

from pytest_strategy import RNG, Strategy

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


CONFTEST = """
import json
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--tb-config")

def pytest_strategies_context(config):
    with open(Path(config.rootpath, "hook_calls.txt"), "a") as calls:
        calls.write("called\\n")
    return json.loads(Path(config.getoption("--tb-config")).read_text())
"""

STRATEGIES = """
from pytest_strategy import Parameter, RNGInteger, RNGSequence, Strategy, TestArg

@Strategy.register("esm_rw")
def esm_rw(nsamples, ctx):
    channels = [p["channel"] for p in ctx["peripherals"] if p["type"] == "Esm"]
    return Parameter(
        TestArg("channel", rng_type=RNGSequence(channels, skip_if_empty="no Esm peripheral")),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        per_sequence_samples=True,
    )

@Strategy.register("plain")
def plain(nsamples):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)))
"""

TEST_RW = """
from pytest_strategy import Strategy

@Strategy.strategy("esm_rw")
def test_write_read(channel, wdata):
    assert channel in (3, 5)
    assert 0 <= wdata <= 255
"""

TEST_PLAIN = """
from pytest_strategy import Strategy

@Strategy.strategy("plain")
def test_plain(x):
    assert 0 <= x <= 9
"""

TWO_ESM = {
    "peripherals": [
        {"type": "Esm", "channel": 3},
        {"type": "Uart", "channel": 4},
        {"type": "Esm", "channel": 5},
    ]
}
NO_ESM = {"peripherals": [{"type": "Uart", "channel": 4}]}


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        esm_strategies=STRATEGIES,
        test_rw=TEST_RW,
        test_rw_again=TEST_RW,
        test_plain=TEST_PLAIN,
    )
    pytester.path.joinpath("two_esm.json").write_text(json.dumps(TWO_ESM))
    pytester.path.joinpath("no_esm.json").write_text(json.dumps(NO_ESM))
    return pytester


def _hook_calls(pytester):
    path = pytester.path / "hook_calls.txt"
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_factory_builds_its_vectors_from_the_config(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "--tb-config=two_esm.json")

    # 10 writes per Esm channel in each of the two modules, plus 10 plain rows
    result.assert_outcomes(passed=50)
    assert _hook_calls(project) == 1


def test_config_without_the_peripheral_skips_with_the_reason(project):
    result = project.runpytest(
        "-p", "no:cacheprovider", "--rng-seed=1", "--tb-config=no_esm.json", "-rs"
    )

    result.assert_outcomes(passed=10, skipped=2)
    result.stdout.fnmatch_lines(["*no Esm peripheral*"])


def test_hook_error_fails_collection_of_modules_that_need_ctx(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1")

    # --tb-config missing: Path(None) raises TypeError inside the hook. Both modules
    # that use the strategy report it; collection errors stop the run.
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(
        [
            "*Strategy factory 'esm_rw' has a 'ctx' parameter, but the "
            "pytest_strategies_context hook raised TypeError*"
        ]
    )
    assert _hook_calls(project) == 1


def test_strategies_that_do_not_need_ctx_never_call_the_hook(project):
    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "test_plain.py")

    result.assert_outcomes(passed=10)
    assert _hook_calls(project) == 0


def test_without_an_implementation_ctx_is_none(pytester):
    pytester.makepyfile(
        default_strategies="""
from pytest_strategy import Parameter, RNGSequence, Strategy, TestArg

@Strategy.register("default_channels")
def default_channels(nsamples, ctx=None):
    channels = [1, 2] if ctx is None else ctx["channels"]
    return Parameter(TestArg("channel", rng_type=RNGSequence(channels)))
""",
        test_default="""
from pytest_strategy import Strategy

@Strategy.strategy("default_channels")
def test_channel(channel):
    assert channel in (1, 2)
""",
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "--nsamples=4")

    result.assert_outcomes(passed=4)


def test_export_strategies_passes_ctx(project):
    project.makepyfile(test_export="""
import json

from pytest_strategy import Strategy

def test_export():
    exported = json.loads(Strategy.export_strategies())["strategies"]
    [esm_rw] = [entry for entry in exported if entry["name"] == "esm_rw"]
    assert "parameter" in esm_rw, esm_rw
""")

    result = project.runpytest(
        "-p", "no:cacheprovider", "--tb-config=two_esm.json", "test_export.py"
    )

    result.assert_outcomes(passed=1)


def test_every_xdist_worker_builds_the_same_tests(project, values_dump):
    pytest.importorskip("xdist")
    project.makeconftest(CONFTEST + values_dump.conftest)

    result = values_dump.run("-n", "2", "--tb-config=two_esm.json")

    result.assert_outcomes(passed=50)
    # Each worker collects, so each calls the hook once; the controller does not
    assert _hook_calls(project) == 2
    # The node IDs do not show every value: the workers drew the same ones
    gw0 = values_dump.read("values-gw0.json")
    assert len(gw0) == 50
    assert gw0 == values_dump.read("values-gw1.json")


def test_hook_can_skip_the_modules_that_need_ctx(project):
    project.makeconftest("""
import pytest

def pytest_strategies_context(config):
    pytest.skip("no testbench configured", allow_module_level=True)
""")

    result = project.runpytest("-p", "no:cacheprovider", "--rng-seed=1", "-rs")

    result.assert_outcomes(passed=10, skipped=2)
    result.stdout.fnmatch_lines(["*no testbench configured*"])


TEST_MIXED = """
from pytest_strategy import Strategy

@Strategy.strategy("esm_rw")
def test_write_read(channel, wdata):
    pass

@Strategy.strategy("plain")
def test_plain_here(x):
    pass

def test_without_strategy():
    pass
"""


@pytest.mark.parametrize(
    "hook, outcomes",
    [
        ("    raise RuntimeError('no bench')\n", {"passed": 10, "errors": 1}),
        (
            "    import pytest\n    pytest.skip('no bench', allow_module_level=True)\n",
            {"passed": 10, "skipped": 1},
        ),
    ],
    ids=["raise", "skip"],
)
def test_a_hook_error_or_skip_takes_the_whole_module(pytester, hook, outcomes):
    """
    The module's tests that do not use ctx do not run either: the error or the skip
    is the module's, as the docs say.
    """
    pytester.makeconftest(f"def pytest_strategies_context(config):\n{hook}")
    pytester.makepyfile(esm_strategies=STRATEGIES, test_mixed=TEST_MIXED, test_plain=TEST_PLAIN)

    args = ("-p", "no:cacheprovider", "--rng-seed=1", "--continue-on-collection-errors")
    result = pytester.runpytest(*args, "-v")

    result.assert_outcomes(**outcomes)
    result.stdout.no_fnmatch_line("*test_mixed.py::*")
    assert "test_plain.py::test_plain[rand-0] PASSED" in result.stdout.str()


TEST_CLASS = """
from pytest_strategy import Strategy

class TestBench:
    @Strategy.strategy("esm_rw")
    def test_write_read(self, channel, wdata):
        pass

    def test_in_the_class(self):
        pass

@Strategy.strategy("plain")
def test_plain_here(x):
    pass

def test_without_strategy():
    pass
"""


@pytest.mark.parametrize(
    "hook, outcomes",
    [
        ("    raise RuntimeError('no bench')\n", {"passed": 21, "errors": 1}),
        (
            "    import pytest\n    pytest.skip('no bench', allow_module_level=True)\n",
            {"passed": 21, "skipped": 1},
        ),
    ],
    ids=["raise", "skip"],
)
def test_in_a_test_class_a_hook_error_or_skip_takes_the_class(pytester, hook, outcomes):
    """
    pytest generates a method's tests when it collects the class: the error or the
    skip is the class's, and the module's other tests run, as the docs say.
    """
    pytester.makeconftest(f"def pytest_strategies_context(config):\n{hook}")
    pytester.makepyfile(esm_strategies=STRATEGIES, test_mixed=TEST_CLASS, test_plain=TEST_PLAIN)

    args = ("-p", "no:cacheprovider", "--rng-seed=1", "--continue-on-collection-errors")
    result = pytester.runpytest(*args, "-v")

    result.assert_outcomes(**outcomes)
    result.stdout.no_fnmatch_line("*test_mixed.py::TestBench::*")
    output = result.stdout.str()
    assert "test_mixed.py::test_plain_here[rand-0] PASSED" in output
    assert "test_mixed.py::test_without_strategy PASSED" in output


def test_random_draws_in_the_hook_do_not_change_the_vectors(project, values_dump):
    project.makeconftest("""
from pytest_strategy import RNG

def pytest_strategies_context(config):
    channels = [3, 5]
    RNG.generator().shuffle(channels)
    return {"peripherals": [{"type": "Esm", "channel": c} for c in channels]}
""" + values_dump.conftest)

    def collected(*paths):
        args = ("-p", "no:cacheprovider", "--rng-seed=42", "--collect-only", "-q", *paths)
        rows = values_dump.collect(*args, subprocess=False)
        return [row for row in rows if "test_rw_again.py::" in row[0]]

    # The first test that needs ctx triggers the hook: test_rw.py in the full run,
    # test_rw_again.py when it runs alone
    full = collected()
    assert len(full) == 20
    assert full == collected("test_rw_again.py")


def test_optional_hook_conftest_works_without_the_plugin(pytester):
    pytester.makeconftest("""
import pytest

@pytest.hookimpl(optionalhook=True)
def pytest_strategies_context(config):
    return {"channels": [3, 5]}
""")
    pytester.makepyfile(test_plain="def test_plain():\n    pass\n")

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:pytest_strategy")

    result.assert_outcomes(passed=1)
