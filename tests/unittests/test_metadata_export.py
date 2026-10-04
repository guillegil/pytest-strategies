import json
from enum import Enum

from pytest_strategy import Parameter, RNGEnum, RNGInteger, Strategy, TestArg


class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class TestMetadataExport:
    def test_test_arg_to_dict(self):
        """Test TestArg.to_dict() serialization"""
        # Static arg
        arg1 = TestArg("static", value=42, description="A static value")
        data1 = arg1.to_dict()
        assert data1["name"] == "static"
        assert data1["description"] == "A static value"
        # The value keeps its type (3.0 wrote has_static_value and str(value))
        assert data1["source"] == "value"
        assert data1["value"] == 42

        # RNG arg
        arg2 = TestArg("rng", rng_type=RNGInteger(0, 10))
        data2 = arg2.to_dict()
        assert data2["name"] == "rng"
        assert data2["source"] == "rng"
        assert data2["rng"]["type"] == "RNGInteger"
        assert data2["rng"]["min"] == 0
        assert data2["rng"]["max"] == 10

    def test_parameter_to_dict(self):
        """Test Parameter.to_dict() serialization"""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", value=5),
            directed_vectors={"edge": (0, 5)},
        )

        data = param.to_dict()
        assert data["schema"] == 1
        assert len(data["arguments"]) == 2
        assert data["arguments"][0]["name"] == "x"
        assert data["arguments"][1]["name"] == "y"
        assert data["directed_vectors"] == [
            {"name": "edge", "id": "directed-edge", "values": {"x": 0, "y": 5}}
        ]

    def test_strategy_export(self):
        """Test Strategy.export_strategies()"""

        # Register a test strategy
        @Strategy.register("export_test_strategy")
        def strategy_factory(nsamples):
            return Parameter(
                TestArg("status", rng_type=RNGEnum(Status)),
                TestArg("count", rng_type=RNGInteger(1, 100)),
            )

        # Export
        json_str = Strategy.export_strategies()
        data = json.loads(json_str)

        assert (data["schema"], data["kind"]) == (1, "strategies")
        [entry] = [e for e in data["strategies"] if e["name"] == "export_test_strategy"]
        strategy_data = entry["parameter"]
        assert len(strategy_data["arguments"]) == 2
        assert strategy_data["arguments"][0]["rng"]["type"] == "RNGEnum"
        assert strategy_data["arguments"][0]["python_type"] == "Status"
