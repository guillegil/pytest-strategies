"""
Tests demonstrating that strategies work alongside regular pytest fixtures.

This file shows that:
1. Strategy parameters can be combined with pytest fixtures
2. Fixtures are properly excluded from signature validation
3. Both features work together seamlessly
"""

import pytest
from typing import Dict, List
from pytest_strategy import Strategy, Parameter, TestArg, RNGInteger, RNGFloat, RNGChoice


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_config() -> Dict[str, int]:
    """Fixture providing configuration."""
    return {
        "max_retries": 3,
        "timeout": 30,
        "batch_size": 100
    }


@pytest.fixture
def api_client() -> str:
    """Fixture providing an API client (simulated)."""
    return "MockAPIClient"


@pytest.fixture
def database_connection() -> str:
    """Fixture providing a database connection (simulated)."""
    return "MockDatabaseConnection"


@pytest.fixture
def temp_data() -> List[int]:
    """Fixture providing temporary test data."""
    return [1, 2, 3, 4, 5]


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

@Strategy.register("fixture_compatible_strategy")
def create_fixture_compatible_strategy(nsamples: int) -> Parameter:
    """Strategy that works with fixtures."""
    return Parameter(
        TestArg("user_id", rng_type=RNGInteger(1, 1000)),
        TestArg("operation", rng_type=RNGChoice(["read", "write", "delete"])),
        directed_vectors={
            "admin": (1, "read"),
            "user": (500, "write"),
        }
    )


@Strategy.register("api_test_strategy")
def create_api_test_strategy(nsamples: int) -> Parameter:
    """Strategy for API testing with fixtures."""
    return Parameter(
        TestArg("endpoint", rng_type=RNGChoice(["/users", "/posts", "/comments"])),
        TestArg("status_code", rng_type=RNGInteger(200, 500)),
        directed_vectors={
            "success": ("/users", 200),
            "not_found": ("/missing", 404),
            "error": ("/error", 500),
        }
    )


# ============================================================================
# TESTS WITH FIXTURES AND STRATEGIES
# ============================================================================

@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_single_fixture(user_id: int, operation: str, sample_config: Dict[str, int]):
    """Test that strategy parameters work with a single fixture."""
    # Strategy parameters
    assert isinstance(user_id, int)
    assert 1 <= user_id <= 1000
    assert operation in ["read", "write", "delete"]
    
    # Fixture parameter
    assert isinstance(sample_config, dict)
    assert "max_retries" in sample_config
    assert sample_config["max_retries"] == 3


@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_multiple_fixtures(
    user_id: int,
    operation: str,
    sample_config: Dict[str, int],
    api_client: str,
    database_connection: str
):
    """Test that strategy parameters work with multiple fixtures."""
    # Strategy parameters
    assert isinstance(user_id, int)
    assert operation in ["read", "write", "delete"]
    
    # Fixture parameters
    assert sample_config["timeout"] == 30
    assert api_client == "MockAPIClient"
    assert database_connection == "MockDatabaseConnection"


@Strategy.strategy("api_test_strategy")
def test_api_with_client_fixture(endpoint: str, status_code: int, api_client: str):
    """Test API endpoints with client fixture."""
    # Strategy parameters
    assert endpoint in ["/users", "/posts", "/comments", "/missing", "/error"]
    assert 200 <= status_code <= 500
    
    # Fixture parameter
    assert api_client == "MockAPIClient"
    
    # Simulate API call logic
    if status_code == 200:
        assert endpoint in ["/users", "/posts", "/comments"]


# ============================================================================
# TESTS WITH FIXTURE SETUP/TEARDOWN
# ============================================================================

@pytest.fixture
def resource_with_cleanup():
    """Fixture with setup and teardown."""
    # Setup
    resource = {"initialized": True, "data": []}
    
    yield resource
    
    # Teardown
    resource["initialized"] = False
    resource["data"].clear()


@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_cleanup_fixture(
    user_id: int,
    operation: str,
    resource_with_cleanup: Dict
):
    """Test that cleanup fixtures work correctly with strategies."""
    assert resource_with_cleanup["initialized"] is True
    
    # Use strategy parameters
    resource_with_cleanup["data"].append({
        "user_id": user_id,
        "operation": operation
    })
    
    assert len(resource_with_cleanup["data"]) == 1


# ============================================================================
# TESTS WITH PARAMETRIZED FIXTURES
# ============================================================================

@pytest.fixture(params=["sqlite", "postgres", "mysql"])
def database_type(request):
    """Parametrized fixture for database types."""
    return request.param


@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_parametrized_fixture(
    user_id: int,
    operation: str,
    database_type: str
):
    """Test that parametrized fixtures work with strategies."""
    # Strategy parameters
    assert isinstance(user_id, int)
    assert operation in ["read", "write", "delete"]
    
    # Parametrized fixture
    assert database_type in ["sqlite", "postgres", "mysql"]


# ============================================================================
# TESTS WITH REQUEST FIXTURE
# ============================================================================

@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_request_fixture(
    user_id: int,
    operation: str,
    request: pytest.FixtureRequest
):
    """Test that special pytest fixtures like 'request' work."""
    # Strategy parameters
    assert isinstance(user_id, int)
    
    # Request fixture
    assert hasattr(request, 'node')
    assert hasattr(request, 'config')


# ============================================================================
# TESTS WITH TMP_PATH FIXTURE
# ============================================================================

@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_tmp_path(
    user_id: int,
    operation: str,
    tmp_path: pytest.TempPathFactory
):
    """Test that tmp_path fixture works with strategies."""
    # Strategy parameters
    assert isinstance(user_id, int)
    
    # tmp_path fixture
    test_file = tmp_path / f"user_{user_id}.txt"
    test_file.write_text(f"Operation: {operation}")
    
    assert test_file.exists()
    assert operation in test_file.read_text()


# ============================================================================
# TESTS WITH MONKEYPATCH FIXTURE
# ============================================================================

@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_monkeypatch(
    user_id: int,
    operation: str,
    monkeypatch: pytest.MonkeyPatch
):
    """Test that monkeypatch fixture works with strategies."""
    # Use monkeypatch
    monkeypatch.setenv("TEST_USER_ID", str(user_id))
    monkeypatch.setenv("TEST_OPERATION", operation)
    
    import os
    assert os.getenv("TEST_USER_ID") == str(user_id)
    assert os.getenv("TEST_OPERATION") == operation


# ============================================================================
# TESTS WITH CAPSYS FIXTURE
# ============================================================================

@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_capsys(
    user_id: int,
    operation: str,
    capsys: pytest.CaptureFixture
):
    """Test that capsys fixture works with strategies."""
    # Print something
    print(f"User {user_id} performing {operation}")
    
    # Capture output
    captured = capsys.readouterr()
    assert str(user_id) in captured.out
    assert operation in captured.out


# ============================================================================
# COMPLEX INTEGRATION TEST
# ============================================================================

@Strategy.strategy("api_test_strategy")
def test_complex_integration(
    endpoint: str,
    status_code: int,
    api_client: str,
    database_connection: str,
    sample_config: Dict[str, int],
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch
):
    """Complex test combining strategy with multiple fixtures."""
    # Strategy parameters
    assert endpoint.startswith("/")
    assert isinstance(status_code, int)
    
    # Multiple fixtures
    assert api_client == "MockAPIClient"
    assert database_connection == "MockDatabaseConnection"
    assert sample_config["batch_size"] == 100
    
    # Use tmp_path
    log_file = tmp_path / "api_log.txt"
    log_file.write_text(f"{endpoint}: {status_code}")
    
    # Use monkeypatch
    monkeypatch.setenv("API_ENDPOINT", endpoint)
    
    # Verify everything works together
    assert log_file.exists()
    import os
    assert os.getenv("API_ENDPOINT") == endpoint
