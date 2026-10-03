"""
Unit tests for OllamaClient in src/llm/ollama_client.py.

Uses httpx.MockTransport to test client behaviors deterministically
without requiring a live running Ollama instance.
"""

import json

import httpx
import pytest

from src.llm.base import LLMAPIError, LLMConnectionError, LLMTimeoutError
from src.llm.models import ChatMessage, MessageRole
from src.llm.ollama_client import OllamaClient
from src.tools.schemas import get_default_tool_registry


@pytest.mark.asyncio
async def test_ollama_client_generates_valid_tool_proposal():
    """Verify OllamaClient parses mock tool response into validated ToolProposal."""
    registry = get_default_tool_registry()

    mock_ollama_payload = {
        "model": "qwen2.5:1.5b",
        "message": {
            "role": "assistant",
            "content": "Checking the exchange rate.",
            "tool_calls": [
                {
                    "id": "call_abc123",
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
        "done_reason": "stop",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        req_json = json.loads(request.content)
        assert req_json["model"] == "qwen2.5:1.5b"
        assert len(req_json["tools"]) == 3  # From default registry
        return httpx.Response(200, json=mock_ollama_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(
            base_url="http://mock-ollama:11434",
            registry=registry,
            http_client=http_client,
        )

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What is the exchange rate from USD to INR?",
            )
        ]
        response = await client.generate(messages)

        assert response.has_tool_proposals is True
        assert len(response.tool_proposals) == 1
        proposal = response.tool_proposals[0]

        # Crucial: Proposal exists, arguments are validated, NO execution took place
        assert proposal.tool_name == "get_exchange_rate"
        assert proposal.arguments == {"from_currency": "USD", "to_currency": "INR"}
        assert proposal.is_valid is True
        assert proposal.validation_error is None


@pytest.mark.asyncio
async def test_ollama_client_conversational_response():
    """Verify OllamaClient returns pure text without tool proposals for regular chat."""
    registry = get_default_tool_registry()

    mock_ollama_payload = {
        "model": "qwen2.5:1.5b",
        "message": {
            "role": "assistant",
            "content": "Hello! I am your voice assistant. How can I help?",
        },
        "done": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_ollama_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(registry=registry, http_client=http_client)

        messages = [ChatMessage(role=MessageRole.USER, content="Hello!")]
        response = await client.generate(messages)

        assert response.has_tool_proposals is False
        assert len(response.tool_proposals) == 0
        assert response.text == "Hello! I am your voice assistant. How can I help?"


@pytest.mark.asyncio
async def test_ollama_client_connection_error():
    """When Ollama is unreachable, LLMConnectionError must be raised."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to establish connection")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(
            base_url="http://unreachable-host:11434",
            http_client=http_client,
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            await client.generate([ChatMessage(role=MessageRole.USER, content="Hi")])

        assert "unavailable" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ollama_client_timeout_error():
    """When Ollama request times out, LLMTimeoutError must be raised."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(timeout=1.0, http_client=http_client)

        with pytest.raises(LLMTimeoutError):
            await client.generate([ChatMessage(role=MessageRole.USER, content="Hi")])


@pytest.mark.asyncio
async def test_ollama_client_api_error_500():
    """When Ollama returns HTTP 500, LLMAPIError must be raised."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Model Crash")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(http_client=http_client)

        with pytest.raises(LLMAPIError) as exc_info:
            await client.generate([ChatMessage(role=MessageRole.USER, content="Hi")])

        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_ollama_client_is_available():
    """Verify is_available endpoint check."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.35.0"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(http_client=http_client)
        assert await client.is_available() is True
