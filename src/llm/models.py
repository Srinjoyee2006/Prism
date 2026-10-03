"""
Data models for LLM requests, responses, and tool proposals.

Defines the core data contracts between the agent planner and LLM providers.
Crucial Rule: The LLM only produces ToolProposal instances; it never executes tools.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Roles within an LLM conversational context."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message in an LLM conversation turn."""

    role: MessageRole
    content: str
    name: str | None = None


@dataclass
class ToolProposal:
    """A tool call proposed by the LLM, subject to verification before execution.

    The existence of this proposal does NOT imply tool execution.
    The proposal carries validation metadata indicating whether its arguments
    conformed to the tool's registered schema.
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    raw_arguments: str | None = None
    is_valid: bool = True
    validation_error: str | None = None
    proposed_at: float = field(default_factory=time.time)

    def mark_invalid(self, error: str) -> None:
        """Mark this proposal as invalid with an error message."""
        self.is_valid = False
        self.validation_error = error


@dataclass
class LLMResponse:
    """Structured response from an LLM invocation."""

    text: str = ""
    tool_proposals: list[ToolProposal] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: dict[str, Any] | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_tool_proposals(self) -> bool:
        """Return True if the model proposed one or more tool calls."""
        return len(self.tool_proposals) > 0

    @property
    def valid_tool_proposals(self) -> list[ToolProposal]:
        """Return only the tool proposals that passed schema validation."""
        return [p for p in self.tool_proposals if p.is_valid]

    @property
    def invalid_tool_proposals(self) -> list[ToolProposal]:
        """Return tool proposals that failed schema validation."""
        return [p for p in self.tool_proposals if not p.is_valid]
