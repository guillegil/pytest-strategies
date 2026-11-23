"""
Parameter(description=..., note=...): free text about a strategy, never used to
generate rows, always in Parameter.to_dict() ("" when not given).
"""

import json

import pytest

from pytest_strategy import Parameter, RNGInteger, TestArg
from pytest_strategy._streams import StreamKey

KEY = StreamKey.root(1, "test", "described", "test_d.py::test_d")


def _param(**kwargs):
    return Parameter(TestArg("x", rng_type=RNGInteger(0, 9)), **kwargs)


class TestDescriptionAndNote:
    def test_empty_when_not_given(self):
        param = _param()
        assert (param.description, param.note) == ("", "")
        data = param.to_dict()
        assert (data["description"], data["note"]) == ("", "")

    def test_kept_and_exported(self):
        param = _param(description="Bursts across the window", note="See ticket 42")
        assert param.description == "Bursts across the window"
        assert param.note == "See ticket 42"
        data = json.loads(json.dumps(param.to_dict(), allow_nan=False))
        assert data["description"] == "Bursts across the window"
        assert data["note"] == "See ticket 42"

    @pytest.mark.parametrize("given", [None, "", "   \n"])
    def test_none_or_blank_is_empty(self, given):
        param = _param(description=given, note=given)
        assert (param.description, param.note) == ("", "")

    def test_text_is_stripped(self):
        param = _param(description="  Line one\nline two\n", note="\tremark ")
        assert param.description == "Line one\nline two"
        assert param.note == "remark"

    @pytest.mark.parametrize("field", ["description", "note"])
    @pytest.mark.parametrize("value", [1, b"bytes", ["text"]])
    def test_must_be_text(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be a str or None"):
            _param(**{field: value})

    def test_do_not_change_the_rows(self):
        plain = _param().generate_vectors(5, _key=KEY)
        described = _param(description="d", note="n").generate_vectors(5, _key=KEY)
        assert plain == described

    def test_extend_keeps_replaces_and_clears(self):
        base = _param(description="Base", note="Base note")
        kept = base.extend(nsamples=3)
        assert (kept.description, kept.note) == ("Base", "Base note")
        changed = base.extend(description="Narrow", note="Other")
        assert (changed.description, changed.note) == ("Narrow", "Other")
        cleared = base.extend(description=None, note="")
        assert (cleared.description, cleared.note) == ("", "")
        assert (base.description, base.note) == ("Base", "Base note")
