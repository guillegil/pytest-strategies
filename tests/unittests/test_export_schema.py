"""
The schema 1 export (D19): RNGType.to_dict(), TestArg.to_dict(), Parameter.to_dict()
and the document export_strategies() returns, outside a pytest session.

tests/integration/test_export_schema_integration.py checks a whole document against
a golden file in a session, with contexts and folders.
"""

import json
import math
from enum import Enum, Flag

import pytest

import pytest_strategy
from pytest_strategy import (
    RNG,
    Parameter,
    RNGBoolean,
    RNGChoice,
    RNGEnum,
    RNGFloat,
    RNGInteger,
    RNGSequence,
    RNGString,
    RNGType,
    RNGWeightedFloat,
    RNGWeightedInteger,
    Series,
    TestArg,
    export_strategies,
    register,
)
from pytest_strategy._export import document, origin, parameter_dict
from pytest_strategy._factory import FactoryError
from pytest_strategy._registry import registry
from pytest_strategy._runtime import runtime


class Color(Enum):
    RED = 1
    GREEN = 2


class Perm(Flag):
    R = 4
    W = 2


def _no_constant(name):
    raise ValueError(f"{name} in the export")


def loads(text):
    """Read JSON strictly: NaN and infinity are errors."""
    return json.loads(text, parse_constant=_no_constant)


# ---------------------------------------------------------------------------
# RNGType.to_dict()
# ---------------------------------------------------------------------------


class Walk(RNGType[int]):
    """A custom type with public, private and container attributes."""

    def __init__(self):
        self.width = 8
        self.tags = {"b", "a"}
        self.color = Color.RED
        self._position = 0

    def generate(self):
        return 1


class Slotted(RNGType[int]):
    """A custom type whose attributes are slots, one of them never set."""

    __slots__ = ("depth", "unset", "_hidden")

    def __init__(self):
        self.depth = 3
        self._hidden = 1

    def generate(self):
        return self.depth


class EvenInteger(RNGInteger):
    """A subclass of a built-in type is a custom type."""


class Named(RNGType[int]):
    """A custom type that writes its own fields."""

    def generate(self):
        return 0

    def to_dict(self):
        return {"type": "Named", "law": "zero"}


class GenerateOnly:
    """Not an RNGType: an object with a generate() method."""

    def __init__(self):
        self.step = 2

    def generate(self):
        return self.step

    def to_dict(self):
        raise AssertionError("only an RNGType's to_dict() is called")


class Unwritable(RNGType[int]):
    """A custom type whose to_dict() returns what JSON cannot hold."""

    def generate(self):
        return 0

    def to_dict(self):
        return {"type": "Unwritable", "lanes": {1, 2}}


class TestRngTypeToDict:
    @pytest.mark.parametrize(
        ("rng_type", "expected"),
        [
            (RNGInteger(0, 9), {"type": "RNGInteger", "min": 0, "max": 9, "predicate": False}),
            (
                RNGInteger(0, 9, predicate=lambda v: v > 1),
                {"type": "RNGInteger", "min": 0, "max": 9, "predicate": True},
            ),
            (
                RNGFloat(-0.5, 1.5),
                {"type": "RNGFloat", "min": -0.5, "max": 1.5, "predicate": False},
            ),
            (RNGBoolean(0.25), {"type": "RNGBoolean", "true_probability": 0.25}),
            (
                RNGChoice(["rd", b"\x00", (1, 2)]),
                {
                    "type": "RNGChoice",
                    "choices": [
                        "rd",
                        {"$repr": "b'\\x00'", "$type": "bytes"},
                        {"$repr": "(1, 2)", "$type": "tuple"},
                    ],
                },
            ),
            (
                RNGEnum(Color),
                {
                    "type": "RNGEnum",
                    "enum": "Color",
                    "members": [
                        {"$enum": "Color", "member": "RED"},
                        {"$enum": "Color", "member": "GREEN"},
                    ],
                    "weights": None,
                    "predicate": False,
                },
            ),
            (
                RNGEnum(Color, weights={Color.GREEN: 2}, predicate=bool),
                {
                    "type": "RNGEnum",
                    "enum": "Color",
                    "members": [
                        {"$enum": "Color", "member": "RED"},
                        {"$enum": "Color", "member": "GREEN"},
                    ],
                    "weights": [{"member": {"$enum": "Color", "member": "GREEN"}, "weight": 2}],
                    "predicate": True,
                },
            ),
            (
                RNGSequence([3, 1], predicate=lambda v: v > 1),
                {"type": "RNGSequence", "sequence": [3], "skip_if_empty": None},
            ),
            (
                Series([], skip_if_empty="no lanes"),
                {"type": "Series", "sequence": [], "skip_if_empty": "no lanes"},
            ),
            (
                RNGString(length=4, charset="01"),
                {
                    "type": "RNGString",
                    "length": 4,
                    "min_length": 1,
                    "max_length": 20,
                    "charset": "01",
                },
            ),
            (
                RNGWeightedInteger({(0, 9): 3, (10, 19): 1}, predicate=bool),
                {
                    "type": "RNGWeightedInteger",
                    "ranges": [
                        {"min": 0, "max": 9, "weight": 3},
                        {"min": 10, "max": 19, "weight": 1},
                    ],
                    "predicate": True,
                },
            ),
            (
                RNGWeightedFloat({(0.0, 1.0): 0.5}),
                {
                    "type": "RNGWeightedFloat",
                    "ranges": [{"min": 0.0, "max": 1.0, "weight": 0.5}],
                    "predicate": False,
                },
            ),
        ],
        ids=lambda value: type(value).__name__ if isinstance(value, RNGType) else "",
    )
    def test_a_built_in_type_has_typed_fields(self, rng_type, expected):
        assert rng_type.to_dict() == expected

    def test_every_built_in_type_has_typed_fields(self):
        built_in = {
            obj
            for name in pytest_strategy.__all__
            if isinstance(obj := getattr(pytest_strategy, name), type)
            and issubclass(obj, RNGType)
            and obj.__module__ == "pytest_strategy.rng"
            and obj.__init__ is not RNGType.__init__
        }

        from pytest_strategy._export import _TYPED

        assert built_in - {pytest_strategy.SequenceLike} == set(_TYPED)

    def test_a_custom_type_has_its_public_attributes(self):
        assert Walk().to_dict() == {
            "type": "Walk",
            "attributes": {
                "width": 8,
                # Sorted, whatever the hash seed
                "tags": {"$repr": "{'a', 'b'}", "$type": "set"},
                "color": {"$enum": "Color", "member": "RED"},
            },
        }

    def test_slots_that_are_set_count(self):
        assert Slotted().to_dict() == {"type": "Slotted", "attributes": {"depth": 3}}

    def test_a_subclass_of_a_built_in_type_is_a_custom_type(self):
        assert EvenInteger(0, 8).to_dict() == {
            "type": "EvenInteger",
            "attributes": {"min": 0, "max": 8, "predicate": None},
        }

    def test_the_result_is_json(self):
        text = json.dumps(Walk().to_dict(), allow_nan=False)

        assert loads(text)["type"] == "Walk"


# ---------------------------------------------------------------------------
# TestArg.to_dict()
# ---------------------------------------------------------------------------


class TestTestArgToDict:
    def test_a_value_argument(self):
        arg = TestArg("mode", value="fast", description="the mode")

        assert arg.to_dict() == {
            "name": "mode",
            "description": "the mode",
            "python_type": "str",
            "validator": False,
            "source": "value",
            "value": "fast",
        }

    def test_a_value_wins_over_an_rng_type(self):
        data = TestArg("x", RNGInteger(0, 9), value=math.nan).to_dict()

        assert (data["source"], data["value"]) == ("value", {"$float": "nan"})
        assert "rng" not in data

    def test_an_rng_argument(self):
        arg = TestArg("addr", rng_type=RNGInteger(0, 9), validator=lambda v: True)

        assert arg.to_dict() == {
            "name": "addr",
            "description": "",
            "python_type": "int",
            "validator": True,
            "source": "rng",
            "rng": {"type": "RNGInteger", "min": 0, "max": 9, "predicate": False},
        }

    def test_an_rng_type_writes_its_own_fields(self):
        assert TestArg("n", rng_type=Named()).to_dict()["rng"] == {"type": "Named", "law": "zero"}

    def test_an_object_with_generate_is_described_by_its_attributes(self):
        data = TestArg("n", rng_type=GenerateOnly()).to_dict()

        assert data["rng"] == {"type": "GenerateOnly", "attributes": {"step": 2}}
        # It has no python_type
        assert data["python_type"] is None

    def test_an_rng_type_without_python_type_has_none(self):
        # RNGType.python_type raises NotImplementedError
        assert TestArg("w", rng_type=Walk()).to_dict()["python_type"] is None

    @pytest.mark.parametrize(
        ("arg", "python_type"),
        [
            (TestArg("c", rng_type=RNGEnum(Color)), "Color"),
            (TestArg("p", rng_type=RNGEnum(Perm)), "Perm"),
            (TestArg("s", rng_type=Series([1.5])), "float"),
            (TestArg("v", value=(1, 2)), "tuple"),
        ],
        ids=["enum", "flag", "series", "value"],
    )
    def test_python_type_is_the_qualified_name(self, arg, python_type):
        assert arg.to_dict()["python_type"] == python_type


# ---------------------------------------------------------------------------
# Parameter.to_dict()
# ---------------------------------------------------------------------------


def _bus(**kwargs):
    return Parameter(
        TestArg("addr", rng_type=RNGInteger(0, 255)),
        TestArg("op", rng_type=RNGEnum(Color)),
        **kwargs,
    )


class TestParameterToDict:
    def test_schema_1_with_every_field_in_order(self):
        data = _bus(nsamples=4).to_dict()

        assert list(data) == [
            "schema",
            "arguments",
            "directed_vectors",
            "test_vectors",
            "constraints",
            "always_include_directed",
            "max_retries",
            "nsamples",
            "per_sequence_samples",
            "max_exhaustive",
            "skip_reason",
        ]
        assert data["schema"] == 1
        assert [arg["name"] for arg in data["arguments"]] == ["addr", "op"]
        assert (data["nsamples"], data["max_retries"], data["max_exhaustive"]) == (4, 100, None)
        # A JSON bool, whatever value the Parameter was given
        assert _bus(always_include_directed=0).to_dict()["always_include_directed"] is False

    def test_vectors_are_lists_with_their_names_ids_and_values_by_name(self):
        param = _bus(
            directed_vectors={
                "zeros": {"op": Color.RED, "addr": 0},
                "marked": pytest.param(1, Color.GREEN, marks=pytest.mark.skip),
            },
            test_vectors={"inf": (math.inf, Color.GREEN)},
        )

        data = param.to_dict()

        assert data["directed_vectors"] == [
            {
                "name": "zeros",
                "id": "directed-zeros",
                "values": {"addr": 0, "op": {"$enum": "Color", "member": "RED"}},
            },
            {
                "name": "marked",
                "id": "directed-marked",
                "values": {"addr": 1, "op": {"$enum": "Color", "member": "GREEN"}},
            },
        ]
        assert data["test_vectors"] == [
            {
                "name": "inf",
                "id": "test-inf",
                "values": {"addr": {"$float": "inf"}, "op": {"$enum": "Color", "member": "GREEN"}},
            }
        ]
        # Written without NaN or infinity
        assert loads(json.dumps(data, allow_nan=False))["test_vectors"][0]["name"] == "inf"

    def test_the_id_is_the_names_format_one_whatever_ids_is(self):
        data = _bus(directed_vectors={"zeros": (0, Color.RED)}, ids="values").to_dict()

        assert data["directed_vectors"][0]["id"] == "directed-zeros"

    def test_constraints_are_named_and_enabled(self):
        param = _bus(vector_constraints={"low": lambda v: v.addr < 9, "red": lambda v: True})

        assert param.to_dict()["constraints"] == [
            {"name": "low", "enabled": True},
            {"name": "red", "enabled": True},
        ]
        assert parameter_dict(param, {"red", "other"})["constraints"] == [
            {"name": "low", "enabled": True},
            {"name": "red", "enabled": False},
        ]


# ---------------------------------------------------------------------------
# The document, outside a session
# ---------------------------------------------------------------------------


@pytest.fixture
def own_registry(monkeypatch):
    """No session, and only the test's own registrations."""
    monkeypatch.setattr(runtime, "_stack", [])
    saved = registry.snapshot()
    registry.clear()
    yield
    registry.restore(saved)


def _defined_in(path, source, name):
    """Return the function ``name`` that ``source`` defines, as if in the file ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    namespace = {"__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[name]


FACTORY = """
from pytest_strategy import Parameter, RNGInteger, TestArg

def bus():
    return Parameter(TestArg("addr", rng_type=RNGInteger(0, 9)))
"""


class TestTheDocument:
    def test_the_envelope(self, own_registry):
        RNG.seed(42)

        data = loads(export_strategies())

        assert {key: data[key] for key in ("schema", "kind", "seed", "nsamples")} == {
            "schema": 1,
            "kind": "strategies",
            "seed": 42,
            "nsamples": 10,
        }
        assert data["generator"] == {
            "name": "pytest-strategies",
            "version": pytest_strategy.__version__,
        }
        assert data["strategies"] == []

    def test_every_registration_sorted_by_name_and_folder(self, own_registry, tmp_path):
        for folder in ("b", "a"):
            register("doc_bus")(_defined_in(tmp_path / folder / "s.py", FACTORY, "bus"))
        register("doc_alpha")(_defined_in(tmp_path / "c" / "s.py", FACTORY, "bus"))

        entries = document()["strategies"]

        # Outside a session, folders are absolute, in posix form
        assert [(entry["name"], entry["origin"]) for entry in entries] == [
            (
                "doc_alpha",
                {
                    "folder": (tmp_path / "c").resolve().as_posix(),
                    "file": (tmp_path / "c" / "s.py").resolve().as_posix(),
                    "qualname": "bus",
                    "line": 4,
                },
            ),
            *[
                (
                    "doc_bus",
                    {
                        "folder": (tmp_path / folder).resolve().as_posix(),
                        "file": (tmp_path / folder / "s.py").resolve().as_posix(),
                        "qualname": "bus",
                        "line": 4,
                    },
                )
                for folder in ("a", "b")
            ],
        ]
        assert all(list(entry) == ["name", "origin", "context", "parameter"] for entry in entries)
        assert all(entry["context"] is None for entry in entries)

    def test_a_factory_that_raises_is_an_error_with_its_type_and_message(self, own_registry):
        @register("doc_broken")
        def broken():
            raise KeyError("lane")

        [entry] = document()["strategies"]

        assert list(entry) == ["name", "origin", "context", "error"]
        assert entry["error"] == {"type": "KeyError", "message": "'lane'"}

    def test_the_plugins_note_is_kept(self, own_registry):
        def opaque(*args):
            raise TypeError("needs a lane")

        register("doc_opaque")(opaque)

        [entry] = document()["strategies"]

        assert entry["error"]["type"] == "TypeError"
        assert entry["error"]["message"] == "needs a lane"
        assert entry["error"]["note"].startswith("The plugin called it with no arguments")

    def test_the_plugins_own_errors(self, own_registry):
        @register("doc_none")
        def returns_none():
            return None

        @register("doc_rejected")
        def rejected(n):
            raise AssertionError("not called")

        entries = {entry["name"]: entry for entry in document()["strategies"]}

        assert entries["doc_none"]["error"] == {
            "type": "ValueError",
            "message": "Strategy 'doc_none' must return a Parameter, got NoneType "
            "(did the factory forget to return?)",
        }
        assert entries["doc_rejected"]["error"]["type"] == "ValueError"
        assert "has a parameter 'n'" in entries["doc_rejected"]["error"]["message"]

    def test_an_entry_json_cannot_hold_is_an_error(self, own_registry):
        @register("doc_unwritable")
        def unwritable():
            return Parameter(TestArg("n", rng_type=Unwritable()))

        @register("doc_fine")
        def fine():
            return Parameter(TestArg("n", value=1))

        entries = {entry["name"]: entry for entry in loads(export_strategies())["strategies"]}

        assert entries["doc_unwritable"]["error"] == {
            "type": "TypeError",
            "message": "Object of type set is not JSON serializable",
        }
        assert "parameter" in entries["doc_fine"]

    def test_a_factory_without_a_file(self, own_registry):
        namespace = {"Parameter": Parameter, "TestArg": TestArg}
        exec("def made():\n    return Parameter(TestArg('x', value=1))\n", namespace)
        register("doc_exec")(namespace["made"])

        [entry] = document()["strategies"]

        assert entry["origin"] == {"folder": None, "file": None, "qualname": "made", "line": 1}
        assert entry["parameter"]["arguments"][0]["value"] == 1

    def test_origin_inside_the_rootdir_is_relative(self, tmp_path):
        factory = _defined_in(tmp_path / "tests" / "dma" / "strategies.py", FACTORY, "bus")

        assert origin(factory, tmp_path) == {
            "folder": "tests/dma",
            "file": "tests/dma/strategies.py",
            "qualname": "bus",
            "line": 4,
        }
        assert origin(factory, tmp_path / "tests" / "dma")["folder"] == "."


class TestFactoryError:
    def test_the_message_names_the_strategy_and_the_error(self, own_registry):
        from pytest_strategy._factory import FactoryInputs, call_factory
        from pytest_strategy._options import StrategyOptions

        error = RuntimeError("boom")

        def broken():
            raise error

        inputs = FactoryInputs(options=StrategyOptions(strategy="s"), rng=RNG.generator(), ctx=dict)
        with pytest.raises(FactoryError) as excinfo:
            call_factory("s", broken, inputs)

        assert (
            str(excinfo.value)
            == "Error calling strategy factory 's' (nsamples=10): RuntimeError: boom"
        )
        assert isinstance(excinfo.value, ValueError)
        assert (excinfo.value.error, excinfo.value.note) == (error, None)
        assert excinfo.value.__cause__ is error
