"""
The LaTeX of the export (_latex.py) in a session: strategies in a conftest.py and
in a test module, both of which pytest compiles with its assertion rewriting,
and the files of ``--strategy-export``.
"""

import json

pytest_plugins = ["pytester"]

CONFTEST = """
    from pytest_strategy import Parameter, RNGInteger, TestArg, register

    LIMIT = 4096

    @register("burst")
    def burst():
        assert LIMIT
        return Parameter(
            TestArg("addr", rng_type=RNGInteger(0, 4095, predicate=lambda x: x % 4 == 0)),
            TestArg("len", rng_type=RNGInteger(1, 64)),
            vector_constraints={"fits": lambda v: v.addr + v.len <= LIMIT},
        )
"""

TESTS = """
    from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

    def odd(n):
        return n % 2 == 1

    @register("local")
    def local():
        assert odd(1)
        return Parameter(
            TestArg("n", rng_type=RNGInteger(1, 20, predicate=lambda n: n not in (12, 13))),
            TestArg("m", rng_type=RNGInteger(1, 9, predicate=odd)),
            vector_constraints=[lambda v: v.n > v.m, lambda v: odd(v.n + v.m) or True],
        )

    @strategy("burst")
    def test_burst(addr, len):
        assert addr % 4 == 0

    @strategy("local")
    def test_local(n, m):
        assert n > m
"""


def test_the_run_export_has_the_latex_of_each_strategy(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_latex_run=TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--strategy-export=out")

    assert result.ret == 0, result.stdout.str()
    for file in ("strategies.json", "strategies.vectors.json"):
        document = json.loads((pytester.path / "out" / file).read_text(encoding="utf-8"))
        shown = {}
        for test in document["tests"]:
            for used in test["strategies"]:
                parameter = used["parameter"]
                shown[used["name"]] = (
                    [argument["latex"] for argument in parameter["arguments"]],
                    [constraint["latex"] for constraint in parameter["constraints"]],
                )
        assert shown == {
            "burst": (
                [
                    r"\mathit{addr} \in [0, 4095],\ \mathit{addr} \bmod 4 = 0",
                    r"\mathit{len} \in [1, 64]",
                ],
                [r"\mathit{addr} + \mathit{len} \le 4096"],
            ),
            "local": (
                [r"n \in [1, 20],\ n \notin \{12, 13\}", r"m \in [1, 9],\ m \bmod 2 = 1"],
                ["n > m", None],
            ),
        }
