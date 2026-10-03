"""
Agent Orchestrator subsystem:
Coordinates Audio I/O, VAD, ASR, LLM, and the Tool Commit Gate.
"""

from typing import Protocol

from src.agent.orchestrator import VoiceAgentOrchestrator, register_default_mock_tools
from src.core.state import AgentState, SessionState


class VoiceAgent(Protocol):
    """Protocol for the top-level voice agent loop."""

    state: SessionState

    async def start(self) -> None:
        """Start the full-duplex agent loop."""
        ...

    async def stop(self) -> None:
        """Gracefully terminate the agent loop."""
        ...


__all__ = [
    "AgentState",
    "SessionState",
    "VoiceAgent",
    "VoiceAgentOrchestrator",
    "register_default_mock_tools",
]
