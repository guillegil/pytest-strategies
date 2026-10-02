"""
End-to-end tests for pytest.param(*values, marks=...) as a directed or test vector,
run through pytester: the row keeps its marks and its name is its ID, and a vector
whose values do not match the arguments, or that has an id=, fails collection.

Distinct module and strategy names are used per run on purpose (see
test_session_isolation_integration.py for rationale).
"""

pytest_plugins = ["pytester"]


def _two_arg_module(name, vectors, test_body="assert (a, b) != (1, 2)"):
    """A test module registering ``name`` with two arguments and the given vectors."""
    return (
        "import pytest\n"
        "from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy\n\n"
        f"@register({name!r})\n"
        "def factory(nsamples):\n"
        "    return Parameter(\n"
        "        TestArg('a', rng_type=RNGInteger(3, 9)),\n"
        "        TestArg('b', rng_type=RNGInteger(3, 9)),\n"
        f"        {vectors},\n"
        "        nsamples=2,\n"
        "    )\n\n"
        f"@strategy({name!r})\n"
        "def test_ab(a, b):\n"
        f"    {test_body}\n"
    )


class TestMarkedVectors:
    def test_xfail_directed_vector_is_reported_xfailed(self, pytester):
        pytester.makepyfile(
            test_pp_directed=_two_arg_module(
                "pp_directed",
                "directed_vectors={'bad': pytest.param(1, 2, marks=pytest.mark.xfail)}",
            )
        )

        result = pytester.runpytest_inprocess("-rx")

        result.assert_outcomes(passed=2, xfailed=1)
        result.stdout.fnmatch_lines(["XFAIL test_pp_directed.py::test_ab[[]*"])

    def test_xfail_test_vector_is_reported_xfailed(self, pytester):
        pytester.makepyfile(
            test_pp_test=_two_arg_module(
                "pp_test",
                "test_vectors={'bad': pytest.param(1, 2, marks=pytest.mark.xfail), 'ok': (3, 4)}",
            )
        )

        result = pytester.runpytest_inprocess("--vector-mode=test")

        result.assert_outcomes(passed=1, xfailed=1)

    def test_vector_name_keeps_the_marks(self, pytester):
        pytester.makepyfile(
            test_pp_filter=_two_arg_module(
                "pp_filter",
                "directed_vectors={'ok': (3, 4), "
                "'bad': pytest.param(1, 2, marks=pytest.mark.xfail)}",
            )
        )

        result = pytester.runpytest_inprocess("--vector-name=bad")

        result.assert_outcomes(xfailed=1)

    def test_the_vectors_name_is_the_test_id(self, pytester):
        pytester.makepyfile(
            test_pp_name=_two_arg_module(
                "pp_name",
                "directed_vectors={'named': pytest.param(1, 2, marks=pytest.mark.slow)}",
                "pass",
            )
        )

        result = pytester.runpytest_inprocess("--collect-only", "-q", "--nsamples=0")

        assert [line for line in result.outlines if "::" in line] == [
            "test_pp_name.py::test_ab[directed-named]"
        ]

    def test_an_explicit_id_fails_collection(self, pytester):
        """3.0 used the id as the test ID; in 4.0 the vector's name is its ID."""
        pytester.makepyfile(
            test_pp_id=_two_arg_module(
                "pp_id", "directed_vectors={'named': pytest.param(1, 2, id='x')}", "pass"
            )
        )

        result = pytester.runpytest_inprocess("--collect-only", "-q", "--nsamples=0")

        assert result.ret == 2
        result.stdout.fnmatch_lines(
            [
                "*In test_ab: *Directed vector 'named' is a pytest.param with id='x', but the "
                "vector's name is its ID (directed-named). Remove id=, and name the vector "
                "after the ID it should have*"
            ]
        )

    def test_one_argument_strategy(self, pytester):
        pytester.makepyfile(test_pp_single="""
            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @register("pp_single")
            def factory(nsamples):
                return Parameter(
                    TestArg("x", rng_type=RNGInteger(0, 9)),
                    directed_vectors={
                        "big": pytest.param(99, marks=pytest.mark.xfail(strict=True)),
                        "skipped": pytest.param(5, marks=pytest.mark.skip(reason="not now")),
                    },
                    nsamples=1,
                )

            @strategy("pp_single")
            def test_x(x):
                assert x < 10
            """)

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(passed=1, xfailed=1, skipped=1)

    def test_dataclass_mode(self, pytester):
        pytester.makepyfile(test_pp_record="""
            from dataclasses import dataclass

            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @dataclass
            class AB:
                a: int
                b: int

            @register("pp_record")
            def factory(nsamples):
                return Parameter(
                    TestArg("a", rng_type=RNGInteger(3, 9)),
                    TestArg("b", rng_type=RNGInteger(3, 9)),
                    directed_vectors={"bad": pytest.param(1, 2, marks=pytest.mark.xfail)},
                    nsamples=2,
                )

            @strategy("pp_record")
            def test_ab(ab: AB):
                assert isinstance(ab, AB)
                assert (ab.a, ab.b) != (1, 2)
            """)

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(passed=2, xfailed=1)


class TestWrongNumberOfValues:
    def test_too_few_values_fail_collection(self, pytester):
        pytester.makepyfile(test_pp_short="""
            import pytest
            from pytest_strategy import Parameter, RNGInteger, TestArg, register, strategy

            @register("pp_short")
            def factory(nsamples):
                return Parameter(
                    TestArg("a", rng_type=RNGInteger(0, 9)),
                    TestArg("b", rng_type=RNGInteger(0, 9)),
                    TestArg("c", rng_type=RNGInteger(0, 9)),
                    directed_vectors={"short": pytest.param(1, 2)},
                )

            @strategy("pp_short")
            def test_abc(a, b, c):
                pass
            """)

        result = pytester.runpytest_inprocess()

        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            ["*In test_abc: *ValueError: Directed vector 'short' has 2 values, expected 3*"]
        )
