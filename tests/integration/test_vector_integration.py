"""
End-to-end tests for Vector rows, run through pytester: constraints read the
row's fields by name in every generation path, the tests still receive the
arguments (or a dataclass record), and an argument name that cannot be a field
fails collection.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

import pytest

pytest_plugins = ["pytester"]

STRATEGIES = """
    from dataclasses import dataclass

    from pytest_strategy import (
        Parameter, RNGInteger, RNGSequence, Series, TestArg, Vector, register, strategy,
    )

    SEEN = []

    def aligned(v):
        # Every row the constraint sees is a Vector over the argument names
        assert isinstance(v, Vector) and type(v)._fields == ("ch", "addr", "len"), v
        SEEN.append(v)
        return v.addr % 4 == 0 and v.len > v.ch

    @register("vec_{case}_plain")
    def plain(nsamples):
        return Parameter(
            TestArg("ch", value=0),
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            TestArg("len", rng_type=RNGInteger(1, 8)),
            vector_constraints=[aligned],
            directed_vectors={{"zeros": (0, 0, 1)}},
        )

    @register("vec_{case}_series")
    def series(nsamples):
        return Parameter(
            TestArg("ch", rng_type=Series([0, 1, 2])),
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            TestArg("len", rng_type=RNGInteger(1, 8)),
            vector_constraints=[aligned],
        )

    @register("vec_{case}_per_seq")
    def per_seq(nsamples):
        return Parameter(
            TestArg("ch", rng_type=RNGSequence([0, 1])),
            TestArg("addr", rng_type=RNGInteger(0, 63)),
            TestArg("len", rng_type=RNGInteger(1, 8)),
            vector_constraints=[aligned],
            per_sequence_samples=True,
        )

    @dataclass
    class Burst:
        ch: int
        addr: int
        len: int

    @strategy("vec_{case}_plain")
    def test_plain(ch, addr, len):
        assert addr % 4 == 0 and len > ch

    @strategy("vec_{case}_series")
    def test_series(ch, addr, len):
        assert addr % 4 == 0 and len > ch

    @strategy("vec_{case}_per_seq")
    def test_per_seq(ch, addr, len):
        assert addr % 4 == 0 and len > ch

    @strategy("vec_{case}_series")
    def test_record(burst: Burst):
        assert isinstance(burst, Burst)
        assert burst.addr % 4 == 0 and burst.len > burst.ch

    def test_constraint_saw_rows():
        assert SEEN
"""


@pytest.mark.parametrize(
    "case, options, passed",
    [
        # 4 plain (1 directed + 3), 3 Series, 3 per combination x 2, 3 records, 1
        ("finite", ["--nsamples=3"], 4 + 3 + 6 + 3 + 1),
        # plain falls back to 10 rows; Series and RNGSequence enumerate once each
        ("auto", ["--nsamples=auto"], 11 + 3 + 2 + 3 + 1),
    ],
)
def test_constraints_read_fields_in_every_path(pytester, case, options, passed):
    pytester.makepyfile(**{f"test_vec_{case}": STRATEGIES.format(case=case)})

    result = pytester.runpytest(*options)

    result.assert_outcomes(passed=passed)


@pytest.mark.parametrize("name", ["a-b", "_hidden", "class"])
def test_invalid_argument_name_fails_collection(pytester, name):
    pytester.makepyfile(test_vec_names=f"""
        from pytest_strategy import Parameter, RNGInteger, TestArg, strategy

        def bad(nsamples):
            return Parameter(TestArg({name!r}, rng_type=RNGInteger(0, 9)))

        @strategy(bad, validate_signature=False)
        def test_bad(**kwargs):
            pass
        """)

    result = pytester.runpytest()

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([f"*Parameter argument {name!r}*"])
