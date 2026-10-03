"""
Unit tests for ToolSchema, ToolParameter, and ToolRegistry in src/llm/tool_schema.py.
"""

from src.llm.tool_schema import ToolParameter, ToolSchema
from src.tools.schemas import (
    create_exchange_rate_schema,
    create_flight_search_schema,
    get_default_tool_registry,
)


def test_tool_parameter_schema_generation():
    """Verify conversion of ToolParameter to JSON Schema property."""
    param = ToolParameter(
        name="currency",
        param_type="string",
        description="3-letter code",
        required=True,
        enum=["USD", "EUR", "INR"],
    )
    js = param.to_json_schema()
    assert js["type"] == "string"
    assert js["description"] == "3-letter code"
    assert js["enum"] == ["USD", "EUR", "INR"]


def test_tool_schema_to_ollama_format():
    """Verify full ToolSchema conversion to Ollama/OpenAI tool format."""
    schema = create_exchange_rate_schema()
    ollama_dict = schema.to_ollama_tool()

    assert ollama_dict["type"] == "function"
    fn = ollama_dict["function"]
    assert fn["name"] == "get_exchange_rate"
    assert "parameters" in fn
    assert fn["parameters"]["type"] == "object"
    assert "from_currency" in fn["parameters"]["properties"]
    assert "to_currency" in fn["parameters"]["properties"]
    assert set(fn["parameters"]["required"]) == {"from_currency", "to_currency"}


def test_validate_arguments_valid():
    """Valid arguments must pass validation without error."""
    schema = create_flight_search_schema()
    valid_args = {
        "origin": "JFK",
        "destination": "LHR",
        "departure_date": "2026-12-01",
        "max_price": 450.50,
    }
    is_valid, error, validated = schema.validate_arguments(valid_args)

    assert is_valid is True
    assert error is None
    assert validated["origin"] == "JFK"
    assert validated["max_price"] == 450.50


def test_validate_arguments_missing_required():
    """Missing a required parameter must fail validation."""
    schema = create_flight_search_schema()
    missing_dest = {
        "origin": "JFK",
        "departure_date": "2026-12-01",
    }
    is_valid, error, _ = schema.validate_arguments(missing_dest)

    assert is_valid is False
    assert "destination" in str(error)


def test_validate_arguments_invalid_type_string_instead_of_number():
    """Passing invalid data types must fail validation."""
    schema = create_flight_search_schema()
    invalid_price = {
        "origin": "JFK",
        "destination": "LHR",
        "departure_date": "2026-12-01",
        "max_price": "four hundred",  # String instead of float/int
    }
    is_valid, error, _ = schema.validate_arguments(invalid_price)

    assert is_valid is False
    assert "max_price" in str(error)
    assert "expected type 'number'" in str(error)


def test_validate_arguments_boolean_not_int():
    """Booleans should not pass for integers."""
    schema = ToolSchema(name="sample", description="sample")
    schema.add_parameter(
        ToolParameter(name="count", param_type="integer", required=True)
    )

    is_valid, error, _ = schema.validate_arguments({"count": True})
    assert is_valid is False
    assert "count" in str(error)


def test_tool_registry_lookup_and_unknown_tool():
    """Registry must validate known tools and reject unknown tools."""
    registry = get_default_tool_registry()

    assert registry.get("get_exchange_rate") is not None
    assert registry.get("search_flights") is not None
    assert registry.get("track_order") is not None
    assert registry.get("non_existent_tool") is None

    # Test unknown tool validation
    is_valid, error, _ = registry.validate_proposal(
        "delete_database", {"confirm": True}
    )
    assert is_valid is False
    assert "Unknown tool: 'delete_database'" in str(error)


def test_tool_registry_valid_proposal():
    """Registry correctly validates a proposal for a registered tool."""
    registry = get_default_tool_registry()

    is_valid, error, validated = registry.validate_proposal(
        "get_exchange_rate",
        {"from_currency": "USD", "to_currency": "INR"},
    )
    assert is_valid is True
    assert error is None
    assert validated["from_currency"] == "USD"
    assert validated["to_currency"] == "INR"
