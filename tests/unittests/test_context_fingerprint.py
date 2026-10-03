"""
Unit tests for the fingerprint of a testbench context (``_fingerprint.fingerprint``):
the same contents give the same 8 hex characters in every process, checkout and
import mode, volatile parts (memory addresses, hash order, excluded pydantic fields,
secrets) are left out, and nothing makes it raise.
"""

import argparse
import dataclasses
import datetime
import decimal
import enum
import json
import math
import os
import subprocess
import sys
import uuid
from collections import OrderedDict, deque, namedtuple
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from pytest_strategy import _fingerprint
from pytest_strategy._fingerprint import UNAVAILABLE, fingerprint


def fp(value, rootpath=None):
    """The fingerprint's hex alone."""
    return fingerprint(value, rootpath)[0]


class Color(enum.Enum):
    RED = 1
    GREEN = 2


class Level(enum.IntEnum):
    ONE = 1


class Perm(enum.Flag):
    R = 1
    W = 2


class Plain:
    """A class that keeps the default repr."""

    def __init__(self, value):
        self.value = value


class Other:
    """Another class that keeps the default repr."""


class Shown:
    """A class with a repr of its own, which shows its value."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Shown({self.value!r})"


@dataclasses.dataclass
class Bench:
    name: str
    lanes: frozenset[str]


# Prints the fingerprints of the four contexts the acceptance names and of other
# objects whose repr shows a set in hash order (or a mock's address), with the
# types reported partial, one JSON list, and how the set of strings iterates in
# this process (which PYTHONHASHSEED changes)
SCRIPT = """
import argparse, collections, dataclasses, datetime, enum, json, types
from unittest.mock import MagicMock
from pytest_strategy._fingerprint import fingerprint

NAMES = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}

class Color(enum.Enum):
    RED = 1

@dataclasses.dataclass
class Bench:
    lanes: frozenset

class Shown:
    \"\"\"A class whose own repr shows a set of strings in hash order\"\"\"
    def __init__(self, lanes):
        self.lanes = lanes
    def __repr__(self):
        return f"Shown({self.lanes!r})"

values = [
    Bench(frozenset(NAMES)),
    {Color.RED: datetime.date(2026, 10, 3), "names": NAMES},
    float("nan"),
    types.SimpleNamespace(lanes=NAMES, width=8),
    argparse.Namespace(lanes=NAMES, width=16),
    {"dut": MagicMock(name="dut"), "width": 8},
    Shown(frozenset(NAMES)),
    collections.deque([NAMES]),
]
if EXTRA == "pydantic":
    import pydantic

    class Config(pydantic.BaseModel):
        names: set[str]
        nested: dict[str, frozenset[str]]

    values.append(Config(names=NAMES, nested={"lanes": frozenset(NAMES)}))
if EXTRA == "attrs":
    import attrs

    @attrs.frozen
    class Frozen:
        lanes: frozenset

    @attrs.define(slots=False)
    class Defined:
        lanes: set
        nested: dict

    values += [Frozen(frozenset(NAMES)), Defined(NAMES, {"lanes": Frozen(frozenset(NAMES))})]
found = [fingerprint(v) for v in values]
print(json.dumps({
    "fingerprints": [f[0] for f in found],
    "partial": [f[1] for f in found],
    "order": list(NAMES),
}))
"""


def _run(extra, hash_seed):
    """
    Run SCRIPT in a new interpreter under PYTHONHASHSEED ``hash_seed``, with the
    objects of the library ``extra`` too.
    """
    output = subprocess.run(
        [sys.executable, "-c", f"EXTRA = {extra!r}\n{SCRIPT}"],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


class TestAcrossProcesses:
    @pytest.mark.parametrize("extra", [None, "pydantic", "attrs"])
    def test_one_hex_under_every_hash_seed(self, extra):
        if extra is not None:
            pytest.importorskip(extra)
        runs = [_run(extra, seed) for seed in ("1", "2", "3")]

        # The set really iterates in other orders, so its order is left out
        assert len({tuple(run["order"]) for run in runs}) > 1
        # A mock's repr holds its address, which differs in every process too
        assert len({tuple(run["fingerprints"]) for run in runs}) == 1
        fingerprints = runs[0]["fingerprints"]
        assert all(len(f) == 8 and int(f, 16) >= 0 for f in fingerprints)
        assert len(set(fingerprints)) == len(fingerprints)
        # The objects whose own repr shows the set count by their type alone
        partial = runs[0]["partial"]
        assert partial[6:8] == [["Shown"], ["deque"]]
        assert not any(partial[:6] + partial[8:])


class TestPaths:
    def test_a_path_inside_two_rootdirs_gives_one_hex(self, tmp_path):
        one, two = tmp_path / "ci" / "proj", tmp_path / "home" / "checkout"

        assert fp(one / "tb" / "bench.yaml", one) == fp(two / "tb" / "bench.yaml", two)
        assert fp({"config": one / "tb.yaml"}, one) == fp({"config": two / "tb.yaml"}, two)

    def test_inside_the_rootdir_it_is_the_relative_path(self, tmp_path):
        assert fp(tmp_path / "tb" / "bench.yaml", tmp_path) == fp(Path("tb/bench.yaml"))
        assert fp(tmp_path, tmp_path) == fp(Path("."))

    def test_a_path_outside_the_rootdir_stays_absolute(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        assert fp(tmp_path / "shared.yaml", one) != fp(tmp_path / "other.yaml", one)
        assert fp(tmp_path / "shared.yaml", one) == fp(tmp_path / "shared.yaml", two)

    def test_a_rootdir_reached_through_a_link(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("cannot create a symbolic link here")

        assert fp(real / "tb.yaml", link) == fp(Path("tb.yaml"))
        assert fp(link / "tb.yaml", real) == fp(Path("tb.yaml"))

    def test_pure_paths_in_posix_form(self):
        assert fp(PureWindowsPath("tb\\bench.yaml")) == fp(PurePosixPath("tb/bench.yaml"))

    def test_a_path_is_not_its_string(self):
        assert fp(Path("tb.yaml")) != fp("tb.yaml")


class TestPydantic:
    @pytest.fixture
    def model(self):
        pydantic = pytest.importorskip("pydantic")

        class Testbench(pydantic.BaseModel):
            name: str
            pid: int = pydantic.Field(default=0, exclude=True)
            password: pydantic.SecretStr = pydantic.SecretStr("")

        return Testbench

    def test_an_excluded_field_does_not_change_it(self, model):
        assert fp(model(name="tb", pid=1)) == fp(model(name="tb", pid=2))

    def test_a_secret_does_not_change_it(self, model):
        assert fp(model(name="tb", password="one")) == fp(model(name="tb", password="two"))

    def test_the_other_fields_do(self, model):
        assert fp(model(name="tb")) != fp(model(name="other"))

    def test_a_model_is_not_its_dump(self, model):
        assert fp(model(name="tb")) != fp({"name": "tb", "password": "**********"})


class TestPartial:
    def test_a_default_repr_object_is_reported_partial(self):
        digest, partial = fingerprint({"bench": Plain(1)})

        assert partial == ("Plain",)
        # Its type alone: two instances, at other addresses and with other state
        assert digest == fp({"bench": Plain(2)})

    def test_each_type_once_sorted(self):
        assert fingerprint([Plain(1), Other(), {"x": Plain(2)}])[1] == ("Other", "Plain")

    def test_a_type_is_named_by_its_qualified_name(self):
        class Local:
            pass

        name = "TestPartial.test_a_type_is_named_by_its_qualified_name.<locals>.Local"
        assert fingerprint(Local())[1] == (name,)

    def test_an_object_with_its_own_repr_is_not(self):
        class Shown:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"Shown({self.value})"

        assert fingerprint(Shown(1))[1] == ()
        assert fp(Shown(1)) != fp(Shown(2))

    def test_an_address_inside_a_custom_repr_is_removed(self):
        class Wrapper:
            def __init__(self):
                self.inner = Plain(0)
                self.callback = lambda: None

            def __repr__(self):
                return f"Wrapper({self.inner!r}, {self.callback!r})"

        first, second = Wrapper(), Wrapper()
        assert " at 0x" in repr(first)
        assert fingerprint(first) == fingerprint(second)
        assert fingerprint(first)[1] == ()

    def test_a_hex_value_in_a_repr_is_kept(self):
        class Register:
            def __init__(self, address):
                self.address = address

            def __repr__(self):
                return f"Register(0x{self.address:x})"

        assert fp(Register(0x1000)) != fp(Register(0x2000))

    def test_a_mock_s_address_is_removed(self):
        first, second = MagicMock(name="dut"), MagicMock(name="dut")

        assert "id='" in repr(first)
        assert fingerprint({"dut": first}) == fingerprint({"dut": second})
        assert fingerprint(first)[1] == ()
        # Its name and spec are in its repr
        assert fp(first) != fp(MagicMock(name="other"))
        assert fp(Mock(spec=Plain)) != fp(Mock(spec=Other))
        # Nested in a repr of its own
        assert fp(Shown(first)) == fp(Shown(second))

    def test_an_id_that_does_not_end_a_repr_is_kept(self):
        class Device:
            def __init__(self, ident):
                self.ident = ident

            def __repr__(self):
                return f"<Device id='{self.ident}' bus=0>"

        assert fp(Device(1)) != fp(Device(2))

    @pytest.mark.parametrize(
        "lanes",
        [
            frozenset({"alpha", "beta"}),
            [{"alpha", "beta"}],
            {"lanes": frozenset({b"a", b"b"})},
            Shown({Color.RED, Color.GREEN}),
            deque([{Plain(1), Plain(2)}]),
        ],
        ids=["strings", "in_a_list", "bytes_in_a_dict", "enums_in_an_object", "objects_in_a_deque"],
    )
    def test_a_repr_that_may_show_a_set_in_hash_order_counts_by_its_type(self, lanes):
        digest, partial = fingerprint(Shown(lanes))

        assert partial == ("Shown",)
        # Its type alone: another such set gives the same fingerprint
        assert digest == fp(Shown({"other", "set"}))

    @pytest.mark.parametrize(
        "lanes",
        [frozenset({1, 2, 3}), {(1, 2.5), (None, True)}, frozenset({"one"}), set()],
        ids=["numbers", "tuples_of_numbers", "one_element", "empty"],
    )
    def test_a_set_that_iterates_alike_in_every_process_keeps_the_repr(self, lanes):
        digest, partial = fingerprint(Shown(lanes))

        assert partial == ()
        assert digest != fp(Shown({"other", "set"}))

    def test_slots_are_looked_into_without_getattr(self):
        class Slotted:
            __slots__ = ("lanes", "unset")

            def __init__(self, lanes):
                self.lanes = lanes

            def __getattr__(self, name):
                raise AssertionError(f"looked up {name}")

            def __repr__(self):
                return f"Slotted({self.lanes!r})"

        assert fingerprint(Slotted({"alpha", "beta"}))[1] == (Slotted.__qualname__,)
        assert fingerprint(Slotted({1, 2}))[1] == ()

    def test_the_look_into_the_attributes_is_bounded(self, monkeypatch):
        monkeypatch.setattr(_fingerprint, "_ATTRIBUTE_LIMIT", 3)
        # The set is beyond the objects looked at
        deep = Shown([0, 1, 2, [frozenset({"alpha", "beta"})]])

        assert fingerprint(deep)[1] == ()
        monkeypatch.setattr(_fingerprint, "_ATTRIBUTE_LIMIT", 10_000)
        assert fingerprint(deep)[1] == ("Shown",)


class TestNeverRaises:
    def test_a_list_that_contains_itself(self):
        cycle = [1]
        cycle.append(cycle)

        digest, partial = fingerprint(cycle)

        assert digest != UNAVAILABLE and partial == ()
        other = [1]
        other.append(other)
        assert fp(other) == digest

    def test_a_mapping_inside_a_dataclass_inside_itself(self):
        @dataclasses.dataclass
        class Node:
            links: dict

        node = Node({})
        node.links["self"] = node

        assert fp(node) != UNAVAILABLE

    def test_an_object_seen_twice_is_not_a_cycle(self):
        shared = [1, 2]

        assert fp([shared, shared]) == fp([[1, 2], [1, 2]])

    def test_a_recursion_error_gives_unavailable(self):
        class Endless:
            def __repr__(self):
                return repr(self)

        assert fingerprint({"bench": Endless()}) == (UNAVAILABLE, ())

    def test_nesting_deeper_than_the_recursion_limit_gives_unavailable(self):
        deep: list = []
        for _ in range(sys.getrecursionlimit() + 10):
            deep = [deep]

        assert fingerprint(deep) == (UNAVAILABLE, ())

    def test_a_repr_that_raises_gives_unavailable(self):
        class Broken:
            def __repr__(self):
                raise ValueError("no repr")

        assert fingerprint([Broken()]) == (UNAVAILABLE, ())

    def test_a_model_dump_that_raises_gives_unavailable(self):
        class Model:
            model_fields: dict = {}

            def model_dump(self, mode):
                raise RuntimeError("no dump")

        assert fingerprint(Model()) == (UNAVAILABLE, ())

    def test_a_dataclass_field_never_set(self):
        @dataclasses.dataclass
        class Late:
            name: str
            handle: int = dataclasses.field(init=False)

        assert fp(Late("tb")) != fp(Late("other"))


class TestEncoding:
    def test_scalars_of_other_types_differ(self):
        values = [None, True, 1, "1", 1.0, b"1", Level.ONE, [1], {1}, {1: 1}]

        assert len({fp(v) for v in values}) == len(values)

    def test_floats_by_their_repr(self):
        assert fp(float("nan")) == fp(math.nan)
        assert len({fp(float("inf")), fp(float("-inf")), fp(0.0), fp(-0.0)}) == 4

    def test_lists_and_tuples_are_arrays(self):
        assert fp([1, 2]) == fp((1, 2))
        assert fp([1, 2]) != fp([2, 1])

    def test_sets_are_sorted(self):
        assert fp({3, 1, 2}) == fp({1, 2, 3}) == fp(frozenset({2, 3, 1}))
        assert fp({"b", "a"}) == fp({"a", "b"})
        assert fp({(1, "a"), (0, "b")}) == fp({(0, "b"), (1, "a")})

    def test_mappings_are_pairs_in_their_order(self):
        assert fp({"a": 1, "b": 2}) == fp(OrderedDict([("a", 1), ("b", 2)]))
        assert fp({"a": 1, "b": 2}) != fp({"b": 2, "a": 1})
        assert fp({"a": 1}) != fp([["a", 1]])

    def test_enum_members_by_type_and_name(self):
        assert fp(Color.RED) != fp(Color.GREEN)
        assert fp(Level.ONE) != fp(1)
        assert fp(Perm.R | Perm.W) != fp(Perm(0))

    def test_tagged_strings(self):
        values = [
            datetime.datetime(2026, 10, 3, 12, 0),
            datetime.date(2026, 10, 3),
            datetime.time(12, 0),
            datetime.timedelta(days=1),
            decimal.Decimal("1.10"),
            uuid.UUID(int=1),
            complex(1, 2),
            b"\x00\xff",
        ]

        fingerprints = [fp(v) for v in values]
        assert UNAVAILABLE not in fingerprints
        assert len(set(fingerprints)) == len(values)
        assert fp(datetime.date(2026, 10, 3)) != fp("2026-10-03")
        assert fp(decimal.Decimal("1.10")) != fp(decimal.Decimal("1.1"))

    def test_records_by_field(self):
        Point = namedtuple("Point", "x y")

        assert fp(Bench("tb", frozenset({"a"}))) != fp(Bench("tb", frozenset({"b"})))
        assert fp(Point(1, 2)) != fp((1, 2))
        assert fp(Point(1, 2)) != fp(Point(2, 1))

    @pytest.mark.parametrize("slots", [True, False], ids=["slots", "dict"])
    def test_attrs_classes_by_field(self, slots):
        attrs = pytest.importorskip("attrs")

        @attrs.define(slots=slots)
        class Lanes:
            names: frozenset[str]
            width: int = 8
            later: int = attrs.field(init=False)

        one = Lanes(frozenset({"alpha", "beta"}))

        # By field, so the set is sorted, and not by its repr
        assert fingerprint(one) == (fp(Lanes(frozenset({"beta", "alpha"}))), ())
        assert fp(one) != fp(Lanes(frozenset({"alpha"})))
        assert fp(one) != fp(Lanes(frozenset({"alpha", "beta"}), width=16))
        assert fp(one) != fp(Bench("Lanes", frozenset({"alpha", "beta"})))
        # A field never set
        assert fp(one) != UNAVAILABLE
        one.later = 1
        assert fp(one) != fp(Lanes(frozenset({"alpha", "beta"})))

    @pytest.mark.parametrize("kind", [SimpleNamespace, argparse.Namespace])
    def test_namespaces_by_attribute(self, kind):
        one = kind(lanes={"alpha", "beta"}, width=8)

        assert fingerprint(one) == (fp(kind(lanes={"beta", "alpha"}, width=8)), ())
        assert fp(one) != fp(kind(lanes={"alpha"}, width=8))
        assert fp(one) != fp({"lanes": {"alpha", "beta"}, "width": 8})

    def test_type_names_are_qualified_names_not_module_names(self):
        def make(module):
            return dataclasses.make_dataclass("Bench", ["name"], namespace={"__module__": module})

        one, two = make("tests.tb.conftest"), make("conftest")

        assert fp(one("tb")) == fp(two("tb"))
        assert fp(one) == fp(two)
        assert fp(type("Plain", (), {"__module__": "a.b"})()) == fp(Plain(0))

    def test_subclasses_of_int_and_str_by_their_values(self):
        class Addr(int):
            def __int__(self):
                return 0

        class Name(str):
            def __str__(self):
                return "other"

        assert fp(Addr(4096)) == fp(4096)
        assert fp(Name("tb")) == fp("tb")

    def test_the_hex_is_8_characters(self):
        digest, partial = fingerprint({"channels": [3, 5]})

        assert len(digest) == 8 and int(digest, 16) >= 0
        assert partial == ()

    def test_none_has_a_fingerprint_too(self):
        # The store gives None no fingerprint; the function itself encodes it
        assert fp(None) != UNAVAILABLE
