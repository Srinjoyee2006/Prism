"""
Unit tests for ToolProposalParser in src/llm/parser.py.
"""

from src.llm.parser import ToolProposalParser
from src.tools.schemas import get_default_tool_registry


def test_parse_native_tool_call_valid():
    """Verify parsing a valid tool call from Ollama native tool_calls structure."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_12345",
                    "function": {
                        "name": "get_exchange_rate",
                        "arguments": {
                            "from_currency": "USD",
                            "to_currency": "INR",
                        },
                    },
                }
            ],
        },
        "done": True,
    }

    _text, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "get_exchange_rate"
    assert proposal.call_id == "call_12345"
    assert proposal.arguments == {"from_currency": "USD", "to_currency": "INR"}
    assert proposal.is_valid is True
    assert proposal.validation_error is None


def test_parse_native_tool_call_with_json_string_arguments():
    """Ollama/OpenAI sometimes delivers arguments as serialized JSON string."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_str_args",
                    "function": {
                        "name": "track_order",
                        "arguments": '{"order_id": "ORD-98765"}',
                    },
                }
            ],
        }
    }

    _, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "track_order"
    assert proposal.arguments["order_id"] == "ORD-98765"
    assert proposal.is_valid is True


def test_parse_native_tool_call_malformed_json_arguments():
    """Malformed JSON string in arguments must produce an invalid ToolProposal."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_malformed",
                    "function": {
                        "name": "track_order",
                        "arguments": '{"order_id": "broken_json...',
                    },
                }
            ],
        }
    }

    _, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.is_valid is False
    assert "Malformed JSON arguments" in str(proposal.validation_error)


def test_parse_native_tool_call_invalid_schema():
    """A tool call with missing required arguments fails schema validation."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_missing_arg",
                    "function": {
                        "name": "get_exchange_rate",
                        "arguments": {"from_currency": "USD"},  # Missing to_currency
                    },
                }
            ],
        }
    }

    _, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.is_valid is False
    assert "Missing required parameter" in str(proposal.validation_error)


def test_parse_conversational_response_no_tools():
    """Normal conversational responses return text with zero tool proposals."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I assist you with your travel today?",
        }
    }

    text, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 0
    assert text == "Hello! How can I assist you with your travel today?"


def test_parse_embedded_tool_call_in_markdown_json():
    """Fallback: Model outputs tool call inside markdown ```json block."""
    registry = get_default_tool_registry()

    content = (
        "Let me look up that flight for you:\n"
        "```json\n"
        '{\n  "name": "search_flights",\n  "arguments": {\n    "origin": "SFO",\n    "destination": "JFK",\n    "departure_date": "2026-11-20"\n  }\n}\n'
        "```\n"
    )

    response_data = {"message": {"role": "assistant", "content": content}}

    text, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "search_flights"
    assert proposal.arguments["origin"] == "SFO"
    assert proposal.is_valid is True
    # The JSON block should be stripped from clean text
    assert "search_flights" not in text
    assert "Let me look up that flight for you:" in text


def test_parse_embedded_tool_call_in_tool_tags():
    """Fallback: Model outputs tool call inside <tool_call> tags."""
    registry = get_default_tool_registry()

    content = (
        "Checking now: "
        '<tool_call>{"name": "track_order", "arguments": {"order_id": "ORD-111"}}</tool_call>'
    )
    response_data = {"message": {"role": "assistant", "content": content}}

    text, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    assert proposals[0].tool_name == "track_order"
    assert proposals[0].arguments["order_id"] == "ORD-111"
    assert "<tool_call>" not in text
    assert "Checking now:" in text


def test_parse_clean_text_echo_with_native_tool_calls():
    """When native tool_calls are present, redundant JSON in content is cleaned."""
    registry = get_default_tool_registry()

    response_data = {
        "message": {
            "role": "assistant",
            "content": (
                "I will check the rate for you.\n"
                '{"name": "get_exchange_rate", "arguments": {"from_currency": "USD", "to_currency": "INR"}}\n'
            ),
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_exchange_rate",
                        "arguments": {"from_currency": "USD", "to_currency": "INR"},
                    },
                }
            ],
        }
    }

    text, proposals = ToolProposalParser.parse_from_ollama_response(
        response_data, registry=registry
    )

    assert len(proposals) == 1
    assert text == "I will check the rate for you."
