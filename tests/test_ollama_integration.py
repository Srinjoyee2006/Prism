"""
Integration tests against the live local Ollama server.

These tests are marked with `@pytest.mark.integration` and will dynamically skip
if the local Ollama daemon is unreachable or if model 'qwen2.5:1.5b' is not present.
"""

import httpx
import pytest

from src.llm.models import ChatMessage, MessageRole
from src.llm.ollama_client import OllamaClient
from src.tools.schemas import get_default_tool_registry


async def _check_ollama_ready() -> bool:
    """Check if local Ollama daemon is running on port 11434 with qwen2.5:1.5b."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return any("qwen2.5:1.5b" in m for m in models)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_tool_proposal_get_exchange_rate():
    """Live Test: Verify qwen2.5:1.5b proposes get_exchange_rate without tool execution."""
    if not await _check_ollama_ready():
        pytest.skip("Local Ollama daemon or model 'qwen2.5:1.5b' is not available.")

    registry = get_default_tool_registry()

    async with OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:1.5b",
        registry=registry,
        timeout=30.0,
    ) as client:
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What is the exchange rate from USD to INR?",
            )
        ]

        response = await client.generate(messages)

        # 1. Assert proposal was generated
        assert response.has_tool_proposals is True, f"Expected tool proposal, got: {response.text}"
        proposal = response.tool_proposals[0]

        # 2. Assert tool identity and arguments
        assert proposal.tool_name == "get_exchange_rate"
        assert proposal.arguments.get("from_currency") == "USD"
        assert proposal.arguments.get("to_currency") == "INR"

        # 3. Assert proposal passed schema validation
        assert proposal.is_valid is True
        assert proposal.validation_error is None

        # 4. Crucial: The tool was NOT executed; only a ToolProposal was created.
        assert isinstance(proposal.arguments, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_conversational_response():
    """Live Test: Verify qwen2.5:1.5b returns plain conversational text when no tools needed."""
    if not await _check_ollama_ready():
        pytest.skip("Local Ollama daemon or model 'qwen2.5:1.5b' is not available.")

    registry = get_default_tool_registry()

    async with OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:1.5b",
        registry=registry,
        timeout=30.0,
    ) as client:
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Say hello in French and nothing else.",
            )
        ]

        response = await client.generate(messages)

        assert response.has_tool_proposals is False
        assert len(response.text.strip()) > 0
        assert "bonjour" in response.text.lower() or "hello" in response.text.lower()
