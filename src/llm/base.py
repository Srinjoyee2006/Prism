"""
Abstract base protocols and exceptions for LLM providers.

Ensures that the voice agent architecture remains completely decoupled
from any specific inference backend or vendor implementation.
"""

from typing import Protocol

from src.llm.models import ChatMessage, LLMResponse
from src.llm.tool_schema import ToolSchema


class LLMError(Exception):
    """Base exception for all LLM errors."""


class LLMConnectionError(LLMError):
    """Raised when the LLM service cannot be reached (e.g. Ollama daemon down)."""


class LLMTimeoutError(LLMError):
    """Raised when an inference call times out."""


class LLMAPIError(LLMError):
    """Raised when the LLM service returns an HTTP error or malformed payload."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"LLM API Error ({status_code}): {message}")
        self.status_code = status_code
        self.message = message


class LLMClient(Protocol):
    """Protocol defining the standard interface for any language model client."""

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Execute a completion/chat request against the model.

        The model may return text and/or a list of proposed tool calls.
        Crucial Rule: The model never executes the tools.

        Args:
            messages: Conversation history leading up to the prompt.
            tools: Optional tool schemas available for the model to propose.
            temperature: Sampling temperature.
            timeout: Maximum seconds to wait for response.

        Returns:
            An LLMResponse containing conversational text and validated ToolProposal instances.
        """
        ...

    async def is_available(self) -> bool:
        """Check if the backend model service is reachable and responsive."""
        ...
