import pytest
import json
from enum import Enum
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNGEnum

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
        assert data1["has_static_value"] is True
        assert data1["static_value"] == "42"
        
        # RNG arg
        arg2 = TestArg("rng", rng_type=RNGInteger(0, 10))
        data2 = arg2.to_dict()
        assert data2["name"] == "rng"
        assert data2["rng_type"] == "RNGInteger"
        assert "rng_details" in data2
        assert data2["rng_details"]["min"] == "0"
        assert data2["rng_details"]["max"] == "10"

    def test_parameter_to_dict(self):
        """Test Parameter.to_dict() serialization"""
        param = Parameter(
            TestArg("x", rng_type=RNGInteger(0, 10)),
            TestArg("y", value=5),
            directed_vectors={"edge": (0, 5)}
        )
        
        data = param.to_dict()
        assert len(data["arguments"]) == 2
        assert data["arguments"][0]["name"] == "x"
        assert data["arguments"][1]["name"] == "y"
        assert "edge" in data["directed_vectors"]
        assert data["directed_vectors"]["edge"] == ["0", "5"]

    def test_strategy_export(self):
        """Test Strategy.export_strategies()"""
        # Register a test strategy
        @Strategy.register("export_test_strategy")
        def strategy_factory(n):
            return Parameter(
                TestArg("status", rng_type=RNGEnum(Status)),
                TestArg("count", rng_type=RNGInteger(1, 100))
            )
            
        # Export
        json_str = Strategy.export_strategies()
        data = json.loads(json_str)
        
        assert "export_test_strategy" in data
        strategy_data = data["export_test_strategy"]
        assert len(strategy_data["arguments"]) == 2
        assert strategy_data["arguments"][0]["rng_type"] == "RNGEnum"
