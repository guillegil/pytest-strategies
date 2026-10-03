"""
Unit tests for ``Parameter(ids=...)`` (D3): the option and its check, the formats
that override the strategies_ids ini option, and the callable, which receives each
row's VectorInfo with the row's ID in the effective format and returns the ID to
use, or None to keep it, before duplicates are suffixed.
"""

import dataclasses
import inspect
from unittest.mock import DEFAULT, MagicMock

import pytest

from pytest_strategy import (
    RNG,
    Parameter,
    RNGInteger,
    RNGSequence,
    Series,
    TestArg,
    VectorInfo,
)
from pytest_strategy._resolver import build_parametrization


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


def _named_test(addr, len):
    pass


@dataclasses.dataclass
class Burst:
    addr: int
    len: int


def _record_test(b: Burst):
    pass


def _build(param, test_fn=_named_test, *, name="strat", **options):
    """Resolve a factory returning ``param`` for ``test_fn``."""
    return build_parametrization(
        name, lambda: param, test_fn, config=_make_config(**options), pytest_fixtures=set()
    )


def _burst(**kwargs):
    kwargs.setdefault("nsamples", 3)
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 4095)),
        TestArg("len", rng_type=RNGInteger(1, 64)),
        directed_vectors={"zeros": (0, 1), "max": {"len": 64, "addr": 4095}},
        test_vectors={"mid": (2048, 32)},
        **kwargs,
    )


class _Recorder:
    """An ids= callable that records the VectorInfo of each call and returns ``result``."""

    def __init__(self, result=None):
        self.infos = []
        self._result = result

    def __call__(self, info):
        self.infos.append(info)
        return self._result(info) if callable(self._result) else self._result


class TestTheOption:
    def test_the_default_is_none(self):
        assert _burst().ids is None

    @pytest.mark.parametrize("ids", ["names", "values", str, _Recorder()])
    def test_a_format_or_a_callable_is_kept(self, ids):
        assert _burst(ids=ids).ids is ids

    @pytest.mark.parametrize("ids", ["foo", "", "Names", " values", 42, True, ["names"]], ids=repr)
    def test_anything_else_fails(self, ids):
        expected = f'ids must be None, "names", "values" or a callable, got {ids!r}'
        with pytest.raises(ValueError) as excinfo:
            _burst(ids=ids)

        assert str(excinfo.value) == expected

    def test_it_is_keyword_only(self):
        option = inspect.signature(Parameter).parameters["ids"]

        assert option.kind is inspect.Parameter.KEYWORD_ONLY and option.default is None

    def test_a_value_assigned_later_is_checked_when_the_strategy_is_resolved(self):
        param = _burst()
        param.ids = "Names"

        with pytest.raises(ValueError) as excinfo:
            _build(param)

        assert str(excinfo.value) == (
            'Strategy \'strat\': ids must be None, "names", "values" or a callable, ' "got 'Names'"
        )

    @pytest.mark.parametrize("ids", ["names", "values", _Recorder("x")])
    def test_to_dict_is_unaffected(self, ids):
        data = _burst(ids=ids).to_dict()

        assert data == _burst().to_dict()
        assert "ids" not in data


class TestFormatsOverrideTheIniOption:
    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    def test_values_under_the_names_default(self, test_fn):
        under_ini = _build(_burst(), test_fn, ids="values").ids

        assert _build(_burst(ids="values"), test_fn).ids == under_ini
        assert under_ini[:2] == ["addr=0,len=1", "addr=4095,len=64"]
        assert all(row_id.startswith("addr=") for row_id in under_ini[2:])

    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    def test_names_under_the_values_option(self, test_fn):
        assert _build(_burst(ids="names"), test_fn, ids="values").ids == [
            "directed-zeros",
            "directed-max",
            "rand-0",
            "rand-1",
            "rand-2",
        ]

    @pytest.mark.parametrize("ini", ["names", "values"])
    def test_none_follows_the_ini_option(self, ini):
        expected = _build(_burst(ids=ini)).ids

        assert _build(_burst(ids=None), ids=ini).ids == expected

    def test_the_infos_carry_the_ids(self):
        parametrization = _build(_burst(ids="values"))

        assert [info.id for info in parametrization.infos] == parametrization.ids


class TestTheCallable:
    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    @pytest.mark.parametrize("ini", ["names", "values"])
    def test_it_receives_each_rows_vector_info_with_the_default_id(self, test_fn, ini):
        default = _build(_burst(), test_fn, ids=ini)
        RNG.seed(1234)
        recorder = _Recorder()

        parametrization = _build(_burst(ids=recorder), test_fn, ids=ini)

        assert all(type(info) is VectorInfo for info in recorder.infos)
        assert [info.id for info in recorder.infos] == default.ids
        assert tuple(recorder.infos) == default.infos == parametrization.infos

    def test_it_is_called_once_per_row_of_every_kind(self):
        recorder = _Recorder()

        _build(_burst(ids=recorder))
        _build(_burst(ids=recorder), vector_mode="test")

        assert [(info.kind, info.name, info.index) for info in recorder.infos] == [
            ("directed", "zeros", 0),
            ("directed", "max", 1),
            ("random", None, 0),
            ("random", None, 1),
            ("random", None, 2),
            ("test", "mid", 0),
        ]

    def test_the_enumerated_rows(self):
        param = Parameter(
            TestArg("ch", rng_type=Series([0, 2])),
            TestArg("dev", rng_type=RNGSequence(["a", "b"])),
            nsamples=2,
            ids=lambda info: "-".join(f"{n}{getattr(info.values, n)}" for n in info.enumerated),
        )

        assert _build(param, lambda ch, dev: None).ids == ["ch0", "ch2"]
        # The order of the RNGSequence's values under auto depends on the seed
        assert sorted(_build(param, lambda ch, dev: None, nsamples="auto").ids) == [
            "ch0-deva",
            "ch0-devb",
            "ch2-deva",
            "ch2-devb",
        ]

    def test_none_keeps_the_id(self):
        assert _build(_burst(ids=_Recorder(None))).ids == [
            "directed-zeros",
            "directed-max",
            "rand-0",
            "rand-1",
            "rand-2",
        ]

    def test_a_str_replaces_the_id(self):
        def ids(info):
            if info.kind == "random":
                return f"addr{info.values.addr}"
            return None if info.name == "zeros" else "largest"

        parametrization = _build(_burst(ids=ids))

        addrs = [f"addr{info.values.addr}" for info in parametrization.infos[2:]]
        assert parametrization.ids == ["directed-zeros", "largest", *addrs]
        assert [info.id for info in parametrization.infos] == parametrization.ids

    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    def test_duplicates_are_suffixed_and_the_infos_have_the_final_ids(self, test_fn):
        parametrization = _build(_burst(ids=_Recorder("row")), test_fn)

        assert parametrization.ids == ["row0", "row1", "row2", "row3", "row4"]
        assert [info.id for info in parametrization.infos] == parametrization.ids
        assert [param.id for param in parametrization.params()] == parametrization.ids

    def test_an_id_ending_in_a_digit_gets_an_underscore(self):
        parametrization = _build(_burst(ids=_Recorder("row1")))

        assert parametrization.ids == ["row1_0", "row1_1", "row1_2", "row1_3", "row1_4"]

    def test_an_id_that_another_row_has(self):
        ids = _build(_burst(ids=lambda info: "directed-zeros" if info.index == 0 else None)).ids

        assert ids == ["directed-zeros0", "directed-max", "directed-zeros1", "rand-1", "rand-2"]

    def test_it_sees_the_id_before_the_suffix(self):
        """A repeated row's ID in the values format is suffixed after the callable."""
        recorder = _Recorder()
        param = Parameter(TestArg("x", rng_type=RNGSequence([1])), nsamples=3, ids=recorder)

        parametrization = _build(param, lambda x: None, ids="values")

        assert [info.id for info in recorder.infos] == ["x=1", "x=1", "x=1"]
        assert parametrization.ids == ["x=1_0", "x=1_1", "x=1_2"]
        assert [info.id for info in parametrization.infos] == parametrization.ids
        assert [dataclasses.replace(info, id="x=1") for info in parametrization.infos] == (
            recorder.infos
        )

    def test_under_a_vector_filter(self):
        recorder = _Recorder("only")

        parametrization = _build(_burst(ids=recorder), vector_name="max")

        assert [(info.kind, info.name, info.id) for info in recorder.infos] == [
            ("directed", "max", "directed-max")
        ]
        assert parametrization.ids == ["only"]

    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    def test_it_is_not_called_for_the_skipped_row(self, test_fn):
        def ids(info):
            raise AssertionError("called")

        param = Parameter(
            TestArg("addr", rng_type=Series([], skip_if_empty="no addresses")),
            TestArg("len", rng_type=RNGInteger(1, 64)),
            ids=ids,
        )

        parametrization = _build(param, test_fn)

        assert parametrization.ids == ["skipped"]
        assert [info.id for info in parametrization.infos] == ["skipped"]


class TestTheCallableFails:
    @pytest.mark.parametrize(
        ("result", "shown"),
        [(42, "42"), (b"x", "b'x'"), (1.5, "1.5"), (list(range(100)), "[0, 1, 2, 3, 4, 5, ...]")],
        ids=["int", "bytes", "float", "list"],
    )
    def test_when_it_returns_something_else(self, result, shown):
        param = _burst(ids=lambda info: result if info.index == 2 else None, nsamples=4)

        with pytest.raises(ValueError) as excinfo:
            _build(param, vector_mode="random_only")

        assert str(excinfo.value) == (
            f"Strategy 'strat': ids= returned {shown} for row rand-2; return a str or None"
        )
        assert excinfo.value.__cause__ is None

    def test_when_it_returns_an_empty_str(self):
        with pytest.raises(ValueError) as excinfo:
            _build(_burst(ids=lambda info: ""))

        assert str(excinfo.value) == (
            "Strategy 'strat': ids= returned '' for row directed-zeros; "
            "return a non-empty str or None"
        )

    def test_the_row_is_named_by_its_default_id(self):
        with pytest.raises(ValueError, match=r"returned 42 for row addr=0,len=1; return a str"):
            _build(_burst(ids=lambda info: 42), ids="values")

    def test_when_it_raises(self):
        def ids(info):
            return 1 / 0

        with pytest.raises(ValueError) as excinfo:
            _build(_burst(ids=ids))

        assert str(excinfo.value) == (
            "Strategy 'strat': ids= raised ZeroDivisionError for row directed-zeros: "
            "division by zero"
        )
        assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


class TestValuesDoNotChange:
    @pytest.mark.parametrize("test_fn", [_named_test, _record_test], ids=["named", "record"])
    @pytest.mark.parametrize("ini", ["names", "values"])
    def test_with_and_without_ids(self, test_fn, ini):
        rows = []
        for ids in [None, "names", "values", _Recorder("custom")]:
            RNG.seed(99)
            parametrization = _build(_burst(ids=ids, nsamples=5), test_fn, ids=ini)
            rows.append(
                (parametrization.values, [tuple(info.values) for info in parametrization.infos])
            )

        assert all(row == rows[0] for row in rows)
