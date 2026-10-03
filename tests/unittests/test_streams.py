"""
Tests for the random stream keys, streams v1 (D5): the typed, length-prefixed
encoding of a key's parts, the goldens that pin the derivation and the tokens that
key enumerated positions, and the same ints and draws in every process.

If a golden here fails, the values of random rows change for every seed: that is a
new streams version, which only a major release can bring.
"""

import enum
import json
import numbers
import os
import random
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import pytest_strategy
from pytest_strategy._streams import VERSION, StreamKey, _part, encode, path_part, seed_part
from pytest_strategy.parameters import _position_keys


def length(n):
    """The 8-byte length that precedes a part's payload."""
    return n.to_bytes(8, "big")


class Lane(enum.IntEnum):
    A = 0
    B = 1


class Speed(enum.StrEnum):
    FAST = "fast"
    SLOW = "slow"


class Word:
    """A numbers.Integral that is not an int, as numpy's integers are."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


numbers.Integral.register(Word)


class Sneaky(str):
    """A str subclass whose methods lie: the key reads the str itself."""

    def encode(self, *args, **kwargs):
        return b"other"

    def __str__(self):
        return "other"


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


class TestPart:
    @pytest.mark.parametrize(
        ("value", "payload"),
        [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x00\x80"),
            (-1, b"\xff"),
            (-128, b"\xff\x80"),
            (2**64, b"\x01" + bytes(8)),
        ],
    )
    def test_an_int_is_its_type_byte_length_and_twos_complement(self, value, payload):
        assert _part(value) == b"i" + length(len(payload)) + payload

    @pytest.mark.parametrize(
        ("value", "payload"),
        [
            ("", b""),
            ("a:b", b"a:b"),
            ("é", b"\xc3\xa9"),
            # A lone surrogate, as a path decoded with surrogateescape can hold
            ("\udc80", b"\xed\xb2\x80"),
        ],
    )
    def test_a_str_is_its_type_byte_length_and_utf8(self, value, payload):
        assert _part(value) == b"s" + length(len(payload)) + payload

    def test_a_subclass_is_encoded_by_its_value(self):
        assert _part(Lane.B) == _part(1)
        assert _part(Speed.FAST) == _part("fast")
        assert _part(Sneaky("abc")) == _part("abc")

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(True, id="True"),
            pytest.param(False, id="False"),
            pytest.param(1.0, id="float"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(None, id="None"),
            pytest.param(b"a", id="bytes"),
            pytest.param(bytearray(b"a"), id="bytearray"),
            pytest.param((1,), id="tuple"),
            pytest.param([1], id="list"),
            pytest.param(Decimal(1), id="Decimal"),
            pytest.param(Fraction(1, 2), id="Fraction"),
            pytest.param(1j, id="complex"),
            pytest.param(Word(1), id="Integral"),
            pytest.param(object(), id="object"),
        ],
    )
    def test_anything_but_an_int_or_a_str_is_refused(self, value):
        with pytest.raises(TypeError, match=r"A stream key part must be an int or a str, got "):
            _part(value)

    def test_the_message_names_the_value_and_its_type(self):
        with pytest.raises(TypeError) as excinfo:
            _part(True)

        assert str(excinfo.value) == "A stream key part must be an int or a str, got True (bool)"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: StreamKey.root(True),
            lambda: StreamKey.root(1.5, "a"),
            lambda: StreamKey.root(1, "a", 2.0),
            lambda: StreamKey.root(1).child(None),
            lambda: StreamKey.root(1).child("a", False),
        ],
        ids=["bool seed", "float seed", "float part", "None child", "bool child"],
    )
    def test_keys_refuse_the_same_values(self, build):
        with pytest.raises(TypeError, match="must be an int or a str"):
            build()


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def differ(first, second):
    """Assert that two keys are different keys with different ints."""
    assert first != second
    assert first.seed_int() != second.seed_int()


class TestKeys:
    def test_parts_are_delimited(self):
        """The 3.x key f"{seed}:{key}" gave ("a:b", "c") and ("a", "b:c") one stream."""
        differ(StreamKey.root(1, "a:b", "c"), StreamKey.root(1, "a", "b:c"))
        differ(StreamKey.root(1, "ab"), StreamKey.root(1, "a", "b"))
        differ(StreamKey.root(1, ""), StreamKey.root(1))
        differ(StreamKey.root(1, "a", ""), StreamKey.root(1, "a"))
        differ(StreamKey.root(1, 1, 2), StreamKey.root(1, 258))

    def test_a_str_and_an_int_are_different_parts(self):
        differ(StreamKey.root(1, "3"), StreamKey.root(1, 3))
        differ(StreamKey.root(1, "a", "3"), StreamKey.root(1, "a", 3))
        differ(StreamKey.root(0, ""), StreamKey.root(0, 0))

    def test_ints_of_one_byte_pattern_are_different_parts(self):
        differ(StreamKey.root(1, 255), StreamKey.root(1, -1))
        differ(StreamKey.root(1, 128), StreamKey.root(1, -128))

    def test_the_seed_is_part_of_the_key(self):
        differ(StreamKey.root(1, "a"), StreamKey.root(2, "a"))
        differ(StreamKey.root(-1, "a"), StreamKey.root(1, "a"))

    def test_a_child_extends_the_path(self):
        key = StreamKey.root(7, "test", "burst", "t.py::test_w")

        assert key.child("row", 3) == StreamKey.root(7, "test", "burst", "t.py::test_w", "row", 3)
        assert key.child("row").child(3) == key.child("row", 3)
        assert key.child() == key
        assert key.child("row", 3).seed_int() == key.child("row").child(3).seed_int()
        differ(key.child("row", 3), key.child("row", 4))
        differ(key.child("row"), key)

    def test_a_child_leaves_its_parent_unchanged(self):
        key = StreamKey.root(7, "test")
        before = key.seed_int()

        key.child("factory")

        assert key == StreamKey.root(7, "test")
        assert key.seed_int() == before

    def test_keys_compare_and_hash_by_path(self):
        first = StreamKey.root(1, "a", 2)
        second = StreamKey.root(1, "a", 2)

        assert first == second
        assert hash(first) == hash(second)
        assert {first: "x"}[second] == "x"
        assert StreamKey.root(1, Lane.B) == StreamKey.root(1, 1)
        assert first != (1, "a", 2)

    def test_seed_int_is_a_128_bit_int(self):
        value = StreamKey.root(1, "a").seed_int()

        assert type(value) is int
        assert 0 <= value < 2**128
        assert StreamKey.root(1, "a").seed_int() == value

    def test_a_key_is_built_with_root(self):
        with pytest.raises(TypeError, match=r"StreamKey\.root\(seed, \*parts\)"):
            StreamKey()

    def test_repr_shows_the_seed_and_the_parts(self):
        key = StreamKey.root(1, "test", "burst").child("row", 3)

        assert repr(key) == "StreamKey(1, 'test', 'burst', 'row', 3)"

    def test_child_seed_int_is_the_childs_seed_int(self):
        key = StreamKey.root(7, "test", "burst", "t.py::test_w").child("row")

        assert key.child_seed_int(encode("ch", "i:2", 3, "addr")) == (
            key.child("ch", "i:2", 3, "addr").seed_int()
        )
        assert key.child_seed_int(b"") == key.seed_int()
        assert encode("a", 1) == _part("a") + _part(1)
        assert encode() == b""

    def test_encode_refuses_what_a_key_refuses(self):
        with pytest.raises(TypeError, match="must be an int or a str"):
            encode("a", 1.5)


class TestSeedPart:
    @pytest.mark.parametrize("seed", [0, -5, 2**100, "fast", Lane.B])
    def test_an_int_or_a_str_is_kept(self, seed):
        assert seed_part(seed) is seed

    @pytest.mark.parametrize(
        ("seed", "part"), [(True, "True"), (1.5, "1.5"), (b"\x00", "b'\\x00'")]
    )
    def test_any_other_seed_is_its_repr(self, seed, part):
        assert seed_part(seed) == part
        StreamKey.root(seed_part(seed), "direct")


class TestPathPart:
    def test_a_file_in_the_rootdir_is_relative_in_posix_form(self, tmp_path):
        path = tmp_path / "tests" / "a" / "strategies.py"

        assert path_part(path, tmp_path) == "tests/a/strategies.py"
        assert path_part(tmp_path, tmp_path) == "."

    def test_a_file_outside_the_rootdir_is_relative_too(self, tmp_path):
        """Two checkouts of proj/ and shared/ in different folders key it alike."""
        for base in (tmp_path / "one", tmp_path / "deep" / "two"):
            path = base / "proj" / ".." / "shared" / "strategies.py"
            assert path_part(path, base / "proj") == "../shared/strategies.py"

    def test_links_are_resolved(self, tmp_path):
        (tmp_path / "real").mkdir()
        try:
            os.symlink(tmp_path / "real", tmp_path / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")

        assert path_part(tmp_path / "link" / "s.py", tmp_path) == "real/s.py"

    def test_a_rootdir_reached_through_a_link_is_resolved(self, tmp_path):
        """A checkout reached through a linked folder keys its files as any other."""
        (tmp_path / "real").mkdir()
        try:
            os.symlink(tmp_path / "real", tmp_path / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")

        for base in ("link", "real"):
            path = tmp_path / base / "tests" / "s.py"
            assert path_part(path, tmp_path / "link") == "tests/s.py"

    def test_without_a_rootdir_the_real_path(self, tmp_path):
        path = tmp_path / "s.py"

        assert path_part(path, None) == Path(os.path.realpath(path)).as_posix()

    def test_a_path_on_another_drive_is_its_real_path(self, tmp_path, monkeypatch):
        """On Windows, os.path.relpath() raises ValueError for a path on another drive."""
        path = tmp_path / "s.py"

        def relpath(path, start=None):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")

        with monkeypatch.context() as patch:
            patch.setattr(os.path, "relpath", relpath)
            part = path_part(path, tmp_path)

        assert part == Path(os.path.realpath(path)).as_posix()


# ---------------------------------------------------------------------------
# Goldens: streams v1
# ---------------------------------------------------------------------------

# Three fixed keys: their ints and the first random() of the random.Random each
# seeds. The stdlib promises that random() gives the same sequence for an int
# seed on every Python version, so these hold on every CI cell.
GOLDEN_KEYS = [
    pytest.param(
        (0,),
        250266628006471901534958674779253611222,
        0.14440742731805578,
        id="seed-only",
    ),
    pytest.param(
        (1, "test", "dma_burst", "tests/test_dma.py::TestDma::test_write"),
        265666453351985275240043220467828261702,
        0.11572503753963836,
        id="test",
    ),
    pytest.param(
        (-(2**70), "file", "../shared/stratégies.py", "", 0, -1, 128, -128, 2**64, "\udc80"),
        140385725292838019574657793313592786298,
        0.017182662990541164,
        id="mixed-parts",
    ),
]


class TestGoldens:
    def test_the_version_is_1(self):
        assert VERSION == 1

    @pytest.mark.parametrize(("parts", "seed_int", "first"), GOLDEN_KEYS)
    def test_seed_int(self, parts, seed_int, first):
        assert StreamKey.root(*parts).seed_int() == seed_int
        assert random.Random(seed_int).random() == first

    @pytest.mark.parametrize(("parts", "seed_int", "first"), GOLDEN_KEYS)
    def test_a_child_gives_the_same_int(self, parts, seed_int, first):
        key = StreamKey.root(parts[0])
        for part in parts[1:]:
            key = key.child(part)

        assert key.seed_int() == seed_int

    def test_the_encoding_of_a_key(self):
        """The bytes the derivation hashes: the seed 1, then the part "a"."""
        assert StreamKey.root(1, "a")._path.hex() == (
            "69" "0000000000000001" "01" "73" "0000000000000001" "61"
        )


# ---------------------------------------------------------------------------
# Goldens: the tokens that key enumerated positions (D2)
# ---------------------------------------------------------------------------

# A row's streams are keyed by the tokens of its enumerated values, so these are
# part of streams v1: an IntEnum or StrEnum member is a member (by its qualname
# and name), not an int or a str; a bool is not an int; another numbers.Integral
# is the int it stands for.
GOLDEN_TOKENS = [
    pytest.param(
        [Lane.A, Lane.B, Lane.A, 0, 1],
        ["e:Lane.A", "e:Lane.B", "e:Lane.A~1", "i:0", "i:1"],
        id="int-enum",
    ),
    pytest.param(
        [Speed.FAST, Speed.SLOW, "fast", Speed.FAST],
        ["e:Speed.FAST", "e:Speed.SLOW", "s:fast", "e:Speed.FAST~1"],
        id="str-enum",
    ),
    pytest.param(
        [True, False, True, 1, 0],
        ["b:True", "b:False", "b:True~1", "i:1", "i:0"],
        id="bool",
    ),
    pytest.param(
        [Word(7), Word(-3), 7, Word(2**70)],
        ["i:7", "i:-3", "i:7~1", "i:1180591620717411303424"],
        id="integral",
    ),
]


class TestTokenGoldens:
    @pytest.mark.parametrize(("sequence", "tokens"), GOLDEN_TOKENS)
    def test_tokens(self, sequence, tokens):
        assert [key.token for key in _position_keys("ch", sequence)] == tokens


# ---------------------------------------------------------------------------
# The same ints and draws in every process
# ---------------------------------------------------------------------------

# collect() returns the ints of a few keys, the first draws of the generators they
# seed, and hash() of a str, which shows that PYTHONHASHSEED took effect. As a
# script it prints them as JSON.
SCRIPT = """
import json
import random

from pytest_strategy._streams import StreamKey


def collect():
    keys = [
        StreamKey.root(0),
        StreamKey.root(1, "test", "dma_burst", "tests/test_dma.py::TestDma::test_write"),
        StreamKey.root(1, "test", "burst", "t.py::test_w").child("row", "ch", "s:fast", 3, "addr"),
        StreamKey.root(-5, "file", "../shared/stratégies.py"),
        StreamKey.root(2**40, "fixture", "", "tb", 0),
    ]
    out = []
    for key in keys:
        rng = random.Random(key.seed_int())
        draws = [rng.random(), rng.randint(0, 10**9), rng.choice("abcdefgh"), rng.getrandbits(64)]
        out.append([key.seed_int(), draws])
    return {"keys": out, "hash": hash("pst-stream")}


if __name__ == "__main__":
    print(json.dumps(collect()))
"""


def run_script(hashseed):
    """Run SCRIPT in a new interpreter with PYTHONHASHSEED set, and read its output."""
    source = str(Path(pytest_strategy.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestProcesses:
    def test_every_hash_seed_gives_the_same_ints_and_draws(self):
        runs = {hashseed: run_script(hashseed) for hashseed in ("0", "1", "random")}
        namespace = {"__name__": "streams_script"}
        exec(SCRIPT, namespace)
        here = json.loads(json.dumps(namespace["collect"]()))

        # PYTHONHASHSEED changed hash() in the processes, and nothing else
        assert runs["0"]["hash"] != runs["1"]["hash"]
        for run in runs.values():
            assert run["keys"] == here["keys"]
