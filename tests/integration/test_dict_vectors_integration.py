"""
End-to-end tests for directed and test vectors given by name, run through
pytester: a dict vector reaches the test as its values (3.0 passed its keys), in
named and dataclass mode, with or without pytest.param marks, a namedtuple is
placed by its field names, and a vector that is not one value per argument fails
collection with a hint.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

pytest_plugins = ["pytester"]


def test_dict_vectors_reach_the_test_as_their_values(pytester):
    pytester.makepyfile(test_dv_values="""
        from dataclasses import dataclass
        from typing import NamedTuple

        import pytest
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        class BusTxn(NamedTuple):
            len: int
            addr: int

        SEEN = []

        @register("dv_values_two")
        def two():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(100, 200)),
                TestArg("len", rng_type=RNGInteger(100, 200)),
                directed_vectors={
                    "named": {"len": 16, "addr": 0},
                    "tuple": (4, 8),
                    "txn": BusTxn(len=2, addr=1),
                    "marked": pytest.param({"len": 0, "addr": 0}, marks=pytest.mark.xfail),
                },
                test_vectors={"tv": {"len": 32, "addr": 64}},
                nsamples=0,
            )

        @register("dv_values_one")
        def one():
            return Parameter(
                TestArg("cfg", value={"default": True}),
                directed_vectors={
                    "five": {"cfg": 5},
                    "dict": pytest.param({"cfg": {"a": 1}}),
                    "wrapped": ({"b": 2},),
                },
                nsamples=0,
            )

        @dataclass
        class Burst:
            addr: int
            len: int

        @strategy("dv_values_two")
        def test_two(addr, len):
            SEEN.append(("two", addr, len))
            assert len != 0

        @strategy("dv_values_two")
        def test_record(burst: Burst):
            SEEN.append(("record", burst.addr, burst.len))
            assert burst.len != 0

        @strategy("dv_values_one")
        def test_one(cfg):
            SEEN.append(("one", cfg))

        def test_seen():
            assert sorted(SEEN, key=repr) == sorted(
                [
                    ("two", 0, 16), ("two", 4, 8), ("two", 1, 2), ("two", 0, 0),
                    ("record", 0, 16), ("record", 4, 8), ("record", 1, 2), ("record", 0, 0),
                    ("one", 5), ("one", {"a": 1}), ("one", {"b": 2}),
                ],
                key=repr,
            )
        """)

    result = pytester.runpytest()

    # The xfailed rows are the marked {"len": 0} vector, in named and dataclass mode
    result.assert_outcomes(passed=3 + 3 + 3 + 1, xfailed=2)


def test_test_vectors_by_name_run_in_test_mode(pytester):
    pytester.makepyfile(test_dv_test_mode="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("dv_test_mode")
        def factory():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(100, 200)),
                TestArg("len", rng_type=RNGInteger(100, 200)),
                test_vectors={"tv": {"len": 32, "addr": 64}},
            )

        @strategy("dv_test_mode")
        def test_two(addr, len):
            assert (addr, len) == (64, 32)
        """)

    result = pytester.runpytest("--vector-mode=test")

    result.assert_outcomes(passed=1)


def test_a_misspelled_key_fails_collection_with_did_you_mean(pytester):
    pytester.makepyfile(test_dv_typo="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("dv_typo")
        def factory():
            return Parameter(
                TestArg("addr", rng_type=RNGInteger(0, 9)),
                TestArg("len", rng_type=RNGInteger(0, 9)),
                directed_vectors={"zeros": {"addr": 0, "lenght": 0}},
            )

        @strategy("dv_typo")
        def test_two(addr, len):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "*In test_two: *RNGValueError: Directed vector 'zeros' has unknown argument "
            "'lenght' (did you mean 'len'?) and is missing 'len'*"
        ]
    )


def test_a_string_vector_fails_collection_with_the_hint(pytester):
    """3.0 split "a" into ("a",), so a one-character string happened to work."""
    pytester.makepyfile(test_dv_str="""
        from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

        @register("dv_str")
        def factory():
            return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), directed_vectors={"v": "a"})

        @strategy("dv_str")
        def test_x(x):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*In test_x: *RNGValueError: Directed vector 'v' is 'a' (str), not a tuple of values.*"]
    )
    result.stdout.fnmatch_lines(["*For a one-argument strategy write ('a',) or {'x': 'a'}*"])
