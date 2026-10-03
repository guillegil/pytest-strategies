"""
Unit tests for the names format of test IDs (D3): names_id() for every kind of row,
the IDs the resolver gives each kind, with the labels of enumerated values, and the
strategies_ids ini option.
"""

import enum
import inspect
from unittest.mock import DEFAULT, MagicMock

import pytest

from pytest_strategy import Parameter, RNGInteger, RNGSequence, Series, TestArg
from pytest_strategy._ids import ID_FORMATS, names_id
from pytest_strategy._resolver import build_parametrization, check_ids_format, ids_format


class Color(enum.Enum):
    RED = 1
    GREEN = 2


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Perm(enum.Flag):
    R = 1
    W = 2


def _double(x):
    return 2 * x


def _make_config(*, ids=None, **options):
    """
    Return a mock pytest.Config whose getoption() serves the given CLI options, and
    whose getini() serves ``ids`` as the strategies_ids ini option when it is given.
    """
    values = {"nsamples": None, "vector_mode": "all", "vector_name": None, "vector_index": None}
    values.update(options)
    config = MagicMock()
    config.getoption.side_effect = lambda opt, default=None: values.get(opt, default)
    if ids is not None:
        config.getini.side_effect = lambda name: ids if name == "strategies_ids" else DEFAULT
    return config


def _make_test_fn(argnames):
    """Return a dummy test function with the given argument names as its signature."""

    def _fn(*args, **kwargs):
        pass

    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in argnames]
    _fn.__signature__ = inspect.Signature(params)
    _fn.__name__ = "test_dummy"
    return _fn


def _ids(param, **options):
    """Return the test IDs the resolver gives the rows of ``param`` under the options."""
    return build_parametrization(
        "strat",
        lambda: param,
        _make_test_fn(param.arg_names),
        config=_make_config(**options),
        pytest_fixtures=set(),
        validate=False,
    ).ids


def _burst(**kwargs):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 4095)),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={"zeros": (0, 1), "max": {"len": 64, "addr": 4095}},
        test_vectors={"max": (4095, 64), "mid": (2048, 32)},
        **kwargs,
    )


def _esm(**kwargs):
    return Parameter(
        TestArg("ch", rng_type=Series([0, 2])),
        TestArg("dev", rng_type=Series(["a", "b"])),
        TestArg("wdata", rng_type=RNGInteger(0, 255)),
        **kwargs,
    )


def _labels(sequence, **options):
    """Return the IDs of one Series argument ``v``, enumerated under --nsamples=auto."""
    param = Parameter(TestArg("v", rng_type=Series(sequence)))
    return _ids(param, nsamples="auto", **options)


class TestNamesId:
    """names_id() builds the ID of each kind of row (D3's table)."""

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (("directed", "zeros", None, ()), "directed-zeros"),
            (("test", "max", None, ()), "test-max"),
            (("random", None, 3, ()), "rand-3"),
            (("random", None, 1, ("ch=2",)), "ch=2-rand-1"),
            (("random", None, 0, ("ch=0", "dev=b")), "ch=0-dev=b-rand-0"),
            (("exhaustive", None, 0, ("ch=2",)), "ch=2"),
            (("exhaustive", None, 0, ("ch=0", "dev=b")), "ch=0-dev=b"),
            (("skipped", None, None, ()), "skipped"),
        ],
    )
    def test_every_kind(self, row, expected):
        assert names_id(*row) == expected


class TestIdsOfEveryKind:
    """The resolver names every row by its kind, whatever its values."""

    def test_directed_and_random_rows(self):
        assert _ids(_burst(nsamples=3)) == [
            "directed-zeros",
            "directed-max",
            "rand-0",
            "rand-1",
            "rand-2",
        ]

    def test_test_vectors(self):
        assert _ids(_burst(), vector_mode="test") == ["test-max", "test-mid"]

    def test_random_only(self):
        assert _ids(_burst(nsamples=2), vector_mode="random_only") == ["rand-0", "rand-1"]

    @pytest.mark.parametrize(("option", "value"), [("vector_name", "max"), ("vector_index", 1)])
    def test_a_selected_directed_vector_keeps_its_id(self, option, value):
        assert _ids(_burst(nsamples=3), **{option: value}) == ["directed-max"]

    def test_a_series_value_is_in_its_random_rows_ids(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 2])),
            TestArg("wdata", rng_type=RNGInteger(0, 255)),
            nsamples=4,
        )

        assert _ids(param) == ["ch=0-rand-0", "ch=2-rand-0", "ch=0-rand-1", "ch=2-rand-1"]

    def test_two_series_arguments(self):
        assert _ids(_esm(nsamples=5)) == [
            "ch=0-dev=a-rand-0",
            "ch=0-dev=b-rand-0",
            "ch=2-dev=a-rand-0",
            "ch=2-dev=b-rand-0",
            "ch=0-dev=a-rand-1",
        ]

    def test_a_series_only_strategy_repeats_its_values_as_new_rows(self):
        param = Parameter(TestArg("ch", rng_type=Series([0, 1])), nsamples=3)

        assert _ids(param) == ["ch=0-rand-0", "ch=1-rand-0", "ch=0-rand-1"]

    def test_exhaustive_rows_have_no_kind(self):
        assert _ids(_esm(directed_vectors={"edge": (2, "b", 255)}), nsamples="auto") == [
            "directed-edge",
            "ch=0-dev=a",
            "ch=0-dev=b",
            "ch=2-dev=a",
            "ch=2-dev=b",
        ]

    def test_a_drawn_sequence_is_not_in_the_id(self):
        param = Parameter(TestArg("device", rng_type=RNGSequence(["devA", "devB"])), nsamples=2)

        assert _ids(param) == ["rand-0", "rand-1"]

    def test_an_enumerated_sequence_is(self):
        param = Parameter(TestArg("device", rng_type=RNGSequence(["devA", "devB"])), nsamples=2)

        assert sorted(_ids(param, nsamples="auto")) == ["device=devA", "device=devB"]

    def test_per_sequence_samples(self):
        param = Parameter(
            TestArg("device", rng_type=RNGSequence(["devA", "devB"])),
            TestArg("width", rng_type=RNGInteger(8, 32)),
            per_sequence_samples=True,
            nsamples=2,
        )

        assert _ids(param) == [
            "device=devA-rand-0",
            "device=devA-rand-1",
            "device=devB-rand-0",
            "device=devB-rand-1",
        ]

    def test_combinations_a_constraint_rejects_have_no_rows(self):
        param = Parameter(
            TestArg("lo", rng_type=Series([1, 2])),
            TestArg("hi", rng_type=Series([1, 2])),
            vector_constraints={"ordered": lambda v: v.lo < v.hi},
            nsamples=2,
        )

        assert _ids(param) == ["lo=1-hi=2-rand-0", "lo=1-hi=2-rand-1"]

    def test_skipped(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([], skip_if_empty="no channel")),
            TestArg("x", rng_type=RNGInteger(0, 9)),
        )

        assert _ids(param) == ["skipped"]

    def test_the_ids_do_not_depend_on_the_values(self):
        """The same rows in the values format show their values."""
        assert _ids(_burst(nsamples=0)) == ["directed-zeros", "directed-max"]
        assert _ids(_burst(nsamples=0), ids="values") == ["addr=0,len=1", "addr=4095,len=64"]


class TestLabels:
    """The label each type of value gets in an enumerated argument (D3)."""

    @pytest.mark.parametrize(
        ("sequence", "expected"),
        [
            pytest.param([0, -3, 1000], ["v=0", "v=-3", "v=1000"], id="ints"),
            pytest.param([1.5, -0.0, float("inf")], ["v=1.5", "v=-0.0", "v=inf"], id="floats"),
            pytest.param([True, False, None], ["v=True", "v=False", "v=None"], id="bools-None"),
            pytest.param(["fast", "a.b/c", "é"], ["v=fast", "v=a.b/c", "v=é"], id="strings"),
            pytest.param(["x" * 40], ["v=" + "x" * 40], id="40-characters"),
            pytest.param([Color.RED, Color.GREEN], ["v=RED", "v=GREEN"], id="Enum"),
            pytest.param([Level.LOW, 2], ["v=LOW", "v=2"], id="IntEnum"),
            pytest.param([Perm.R | Perm.W, Perm.R, Perm(0)], ["v=R|W", "v=R", "v2"], id="Flag"),
            pytest.param([Color, _double, len], ["v=Color", "v=_double", "v=len"], id="__name__"),
            pytest.param([1, 1, 2, 1], ["v=1", "v=1~1", "v=2", "v=1~2"], id="repeats"),
            pytest.param([1, "1", 2], ["v0", "v1", "v=2"], id="equal-text"),
            pytest.param([True, "True"], ["v0", "v1"], id="equal-text-bool"),
            pytest.param([object(), object()], ["v0", "v1"], id="objects"),
            pytest.param([(1, 2), (3, 4), 5], ["v0", "v1", "v=5"], id="tuples"),
            pytest.param([b"ab", "ab"], ["v0", "v=ab"], id="bytes"),
            pytest.param(["x" * 41, "ok"], ["v0", "v=ok"], id="over-40-characters"),
        ],
    )
    def test_label(self, sequence, expected):
        assert _labels(sequence) == expected

    @pytest.mark.parametrize(
        "text", ["a b", "a\tb", "a\nb", "a=b", "a~1", "a[0]", "b]", "", "\x00"]
    )
    def test_a_string_the_id_cannot_show_is_labeled_by_position(self, text):
        assert _labels(["ok", text]) == ["v=ok", "v1"]

    def test_a_random_row_starts_with_the_labels(self):
        param = Parameter(TestArg("v", rng_type=Series([1, 1, "x y"])), nsamples=3)

        assert _ids(param) == ["v=1-rand-0", "v=1~1-rand-0", "v2-rand-0"]


class TestCollidingPair:
    """D3's pair of rows that would share an ID if a label could contain "="."""

    def _param(self, **kwargs):
        return Parameter(
            TestArg("a", rng_type=Series(["x-b=y", "x"])),
            TestArg("b", rng_type=Series(["z", "y-b=z"])),
            **kwargs,
        )

    def test_exhaustive(self):
        ids = _ids(self._param(), nsamples="auto")

        assert ids == ["a0-b=z", "a0-b1", "a=x-b=z", "a=x-b1"]
        assert len(set(ids)) == 4

    def test_random(self):
        ids = _ids(self._param(nsamples=4))

        assert ids == ["a0-b=z-rand-0", "a0-b1-rand-0", "a=x-b=z-rand-0", "a=x-b1-rand-0"]


class TestIdsFormat:
    """The strategies_ids ini option: names (the default) or values."""

    def test_names_is_the_default(self):
        assert ID_FORMATS[0] == "names"
        assert ids_format(None) == "names"

    @pytest.mark.parametrize(
        ("raw", "expected"), [("names", "names"), ("values", "values"), (" values ", "values")]
    )
    def test_the_ini_option(self, raw, expected):
        assert ids_format(_make_config(ids=raw)) == expected

    def test_a_config_without_the_option(self):
        """A stand-in config (export_strategies() outside a session) gives names."""
        config = MagicMock()
        config.getini.side_effect = ValueError("unknown name 'strategies_ids'")

        assert ids_format(config) == "names"
        assert ids_format(MagicMock()) == "names"

    @pytest.mark.parametrize("raw", ["names", "values", " names"])
    def test_check_accepts_the_formats(self, raw):
        check_ids_format(_make_config(ids=raw))

    @pytest.mark.parametrize("raw", ["foo", "Names", "", "names,values"])
    def test_check_names_the_option_and_its_values(self, raw):
        with pytest.raises(pytest.UsageError) as excinfo:
            check_ids_format(_make_config(ids=raw))

        assert str(excinfo.value) == f"strategies_ids must be 'names' or 'values', got {raw!r}"

    def test_check_a_value_pytest_cannot_read_as_a_string(self):
        """pytest 9 raises TypeError for a TOML value that is not a string."""
        error = TypeError(
            "pytest.toml: config option 'strategies_ids' expects a string, got int: 1"
        )
        config = MagicMock()
        config.getini.side_effect = error

        with pytest.raises(pytest.UsageError) as excinfo:
            check_ids_format(config)

        assert str(excinfo.value) == f"strategies_ids must be 'names' or 'values'; {error}"
        assert excinfo.value.__suppress_context__
