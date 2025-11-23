"""
Example demonstrating RNGEnum usage with pytest-strategies.

This example shows how to use RNGEnum for testing with Python Enum types,
including weighted probabilities and predicate filtering.
"""

from enum import Enum
from pytest_strategy import Strategy, Parameter, TestArg, RNGEnum, RNGInteger


# Define example Enums
class HttpStatus(Enum):
    """HTTP status codes"""
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_ERROR = 500


class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class UserRole(Enum):
    """User roles"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


# ============================================================================
# STRATEGY 1: Simple Enum Strategy
# ============================================================================

@Strategy.register("http_status_strategy")
def create_http_status_strategy(nsamples: int) -> Parameter:
    """Strategy with simple enum selection"""
    return Parameter(
        TestArg("status_code", rng_type=RNGEnum(HttpStatus)),
        TestArg("method", rng_type=RNGEnum(RequestMethod)),
        directed_vectors={
            "success": (HttpStatus.OK, RequestMethod.GET),
            "not_found": (HttpStatus.NOT_FOUND, RequestMethod.GET),
            "server_error": (HttpStatus.INTERNAL_ERROR, RequestMethod.POST),
        }
    )


@Strategy.strategy("http_status_strategy")
def test_http_status(status_code: HttpStatus, method: RequestMethod):
    """Test HTTP status codes with methods"""
    assert isinstance(status_code, HttpStatus)
    assert isinstance(method, RequestMethod)
    assert 200 <= status_code.value <= 599


# ============================================================================
# STRATEGY 2: Weighted Enum Strategy
# ============================================================================

@Strategy.register("weighted_status_strategy")
def create_weighted_status_strategy(nsamples: int) -> Parameter:
    """Strategy with weighted enum selection (mostly success cases)"""
    return Parameter(
        TestArg("status_code", rng_type=RNGEnum(
            HttpStatus,
            weights={
                HttpStatus.OK: 0.7,           # 70% success
                HttpStatus.CREATED: 0.1,      # 10% created
                HttpStatus.BAD_REQUEST: 0.1,  # 10% client error
                HttpStatus.NOT_FOUND: 0.05,   # 5% not found
                HttpStatus.INTERNAL_ERROR: 0.05  # 5% server error
            }
        )),
        TestArg("user_id", rng_type=RNGInteger(1, 1000)),
        directed_vectors={
            "happy_path": (HttpStatus.OK, 1),
            "error_case": (HttpStatus.INTERNAL_ERROR, 999),
        }
    )


@Strategy.strategy("weighted_status_strategy")
def test_weighted_status(status_code: HttpStatus, user_id: int):
    """Test with weighted status codes (mostly successful requests)"""
    assert isinstance(status_code, HttpStatus)
    assert 1 <= user_id <= 1000
    
    # Most tests should be successful
    if status_code in {HttpStatus.OK, HttpStatus.CREATED}:
        assert status_code.value < 300


# ============================================================================
# STRATEGY 3: Enum with Predicate
# ============================================================================

@Strategy.register("filtered_status_strategy")
def create_filtered_status_strategy(nsamples: int) -> Parameter:
    """Strategy with predicate filtering (only success codes)"""
    return Parameter(
        TestArg("status_code", rng_type=RNGEnum(
            HttpStatus,
            predicate=lambda s: s.value < 400  # Only 2xx and 3xx
        )),
        TestArg("method", rng_type=RNGEnum(
            RequestMethod,
            predicate=lambda m: m != RequestMethod.DELETE  # No DELETE
        )),
        directed_vectors={
            "get_success": (HttpStatus.OK, RequestMethod.GET),
            "post_created": (HttpStatus.CREATED, RequestMethod.POST),
        }
    )


@Strategy.strategy("filtered_status_strategy")
def test_filtered_status(status_code: HttpStatus, method: RequestMethod):
    """Test with filtered status codes (only success)"""
    # Should only get success codes
    assert status_code.value < 400
    # Should never get DELETE
    assert method != RequestMethod.DELETE


# ============================================================================
# STRATEGY 4: Weighted with Predicate
# ============================================================================

@Strategy.register("role_based_strategy")
def create_role_based_strategy(nsamples: int) -> Parameter:
    """Strategy with weighted roles and filtered methods"""
    return Parameter(
        TestArg("role", rng_type=RNGEnum(
            UserRole,
            weights={
                UserRole.USER: 0.6,      # 60% regular users
                UserRole.ADMIN: 0.3,     # 30% admins
                UserRole.GUEST: 0.1,     # 10% guests
            },
            predicate=lambda r: r != UserRole.SUPERADMIN  # No superadmin in tests
        )),
        TestArg("method", rng_type=RNGEnum(
            RequestMethod,
            weights={
                RequestMethod.GET: 0.5,
                RequestMethod.POST: 0.3,
                RequestMethod.PUT: 0.2,
            }
        )),
        directed_vectors={
            "guest_read": (UserRole.GUEST, RequestMethod.GET),
            "user_write": (UserRole.USER, RequestMethod.POST),
            "admin_update": (UserRole.ADMIN, RequestMethod.PUT),
        }
    )


@Strategy.strategy("role_based_strategy")
def test_role_based_access(role: UserRole, method: RequestMethod):
    """Test role-based access control"""
    assert isinstance(role, UserRole)
    assert isinstance(method, RequestMethod)
    
    # Superadmin should never appear (filtered by predicate)
    assert role != UserRole.SUPERADMIN
    
    # Guests can only read
    if role == UserRole.GUEST:
        assert method == RequestMethod.GET
    
    # Admins can do anything
    if role == UserRole.ADMIN:
        assert method in {RequestMethod.GET, RequestMethod.POST, RequestMethod.PUT}


# ============================================================================
# STRATEGY 5: Complex Multi-Enum Strategy
# ============================================================================

@Strategy.register("api_test_strategy")
def create_api_test_strategy(nsamples: int) -> Parameter:
    """Complex strategy with multiple enums and constraints"""
    return Parameter(
        TestArg("method", rng_type=RNGEnum(RequestMethod)),
        TestArg("status_code", rng_type=RNGEnum(HttpStatus)),
        TestArg("role", rng_type=RNGEnum(UserRole)),
        TestArg("retry_count", rng_type=RNGInteger(0, 5)),
        vector_constraints=[
            # GET requests should mostly succeed
            lambda v: v[0] != RequestMethod.GET or v[1].value < 500,
            # Guests can't get server errors
            lambda v: v[2] != UserRole.GUEST or v[1].value != 500,
        ],
        directed_vectors={
            "guest_get_ok": (RequestMethod.GET, HttpStatus.OK, UserRole.GUEST, 0),
            "user_post_created": (RequestMethod.POST, HttpStatus.CREATED, UserRole.USER, 0),
            "admin_delete_ok": (RequestMethod.DELETE, HttpStatus.OK, UserRole.ADMIN, 0),
            "retry_scenario": (RequestMethod.POST, HttpStatus.INTERNAL_ERROR, UserRole.USER, 3),
        }
    )


@Strategy.strategy("api_test_strategy")
def test_api_endpoint(
    method: RequestMethod,
    status_code: HttpStatus,
    role: UserRole,
    retry_count: int
):
    """Test API endpoint with various combinations"""
    assert isinstance(method, RequestMethod)
    assert isinstance(status_code, HttpStatus)
    assert isinstance(role, UserRole)
    assert 0 <= retry_count <= 5
    
    # Verify constraints
    if method == RequestMethod.GET:
        assert status_code.value < 500
    
    if role == UserRole.GUEST:
        assert status_code.value != 500


# ============================================================================
# CLI Usage Examples
# ============================================================================

"""
Run these examples with different CLI options:

# Run with default settings (10 samples)
pytest examples/enum_example.py -v

# Generate more samples
pytest examples/enum_example.py --nsamples=50 -v

# Run only directed vectors
pytest examples/enum_example.py --vector-mode=directed_only -v

# Run specific directed vector
pytest examples/enum_example.py --vector-name=success -v

# Run with specific seed for reproducibility
pytest examples/enum_example.py --rng-seed=42 -v

# Combine options
pytest examples/enum_example.py --nsamples=30 --vector-mode=mixed -v
"""
