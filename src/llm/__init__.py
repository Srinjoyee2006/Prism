"""
LLM Client and Tool Proposal subsystem.

Provides decoupled LLM interfaces, local Ollama integration,
and tool proposal generation with strict schema validation.
"""

from src.llm.base import (
    LLMAPIError,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
)
from src.llm.models import (
    ChatMessage,
    LLMResponse,
    MessageRole,
    ToolProposal,
)
from src.llm.ollama_client import OllamaClient
from src.llm.parser import ToolProposalParser
from src.llm.tool_schema import (
    ToolParameter,
    ToolRegistry,
    ToolSchema,
)

__all__ = [
    "ChatMessage",
    "LLMAPIError",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMResponse",
    "LLMTimeoutError",
    "MessageRole",
    "OllamaClient",
    "ToolParameter",
    "ToolProposal",
    "ToolProposalParser",
    "ToolRegistry",
    "ToolSchema",
]
