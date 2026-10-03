"""
Data models for the Tool Commit Gate subsystem.

Defines the proposal lifecycle enum and the StagedProposal record that
tracks a ToolProposal through PROPOSED → WAITING → COMMITTED / DROPPED / SUPERSEDED.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProposalStatus(str, Enum):
    """Lifecycle states of a staged tool proposal inside the ToolController."""

    PROPOSED = "proposed"        # Received from LLM; not yet in quiet window
    WAITING = "waiting"          # Quiet window is counting down
    COMMITTED = "committed"      # Quiet window elapsed; execution authorised
    EXECUTING = "executing"      # Dispatched to the tool handler
    SUCCEEDED = "succeeded"      # Tool returned successfully
    FAILED = "failed"            # Tool returned an exception
    DROPPED = "dropped"          # Interrupted before commit
    SUPERSEDED = "superseded"    # Replaced by a newer proposal for the same tool


@dataclass
class StagedProposal:
    """A ToolProposal held inside the ToolController pending commit or rollback.

    ``normalised_key`` is a stable, order-independent string built from
    (tool_name, sorted arguments) and is used for idempotency / supersession
    checks without touching the original ToolProposal.
    """

    # Core identity
    proposal_id: str = field(default_factory=lambda: f"sp_{uuid.uuid4().hex[:10]}")
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""                       # Original ToolProposal.call_id

    # Lifecycle
    status: ProposalStatus = ProposalStatus.PROPOSED
    normalised_key: str = ""                # Built on creation; used for dedup

    # Timing
    proposed_at: float = field(default_factory=time.time)
    committed_at: float | None = None
    completed_at: float | None = None

    # Results
    result: Any = None
    error: str | None = None

    def mark_waiting(self) -> None:
        """Move to WAITING status (quiet window started)."""
        self.status = ProposalStatus.WAITING

    def mark_committed(self) -> None:
        """Move to COMMITTED; record commit timestamp."""
        self.status = ProposalStatus.COMMITTED
        self.committed_at = time.time()

    def mark_executing(self) -> None:
        """Move to EXECUTING; tool handler has been called."""
        self.status = ProposalStatus.EXECUTING

    def mark_succeeded(self, result: Any) -> None:
        """Record a successful tool result."""
        self.status = ProposalStatus.SUCCEEDED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        """Record a tool execution error."""
        self.status = ProposalStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def mark_dropped(self) -> None:
        """Mark as dropped due to interruption before commit."""
        self.status = ProposalStatus.DROPPED
        self.completed_at = time.time()

    def mark_superseded(self) -> None:
        """Mark as superseded by a more-recent proposal for the same tool."""
        self.status = ProposalStatus.SUPERSEDED
        self.completed_at = time.time()

    @property
    def is_terminal(self) -> bool:
        """Return True once the proposal has reached a final state."""
        return self.status in {
            ProposalStatus.SUCCEEDED,
            ProposalStatus.FAILED,
            ProposalStatus.DROPPED,
            ProposalStatus.SUPERSEDED,
        }

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since this proposal was first staged."""
        return time.time() - self.proposed_at


def build_normalised_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a stable, order-independent deduplication key.

    Key format:  ``<tool_name>|arg1=val1|arg2=val2|…``  (sorted by arg name).

    Args:
        tool_name: The name of the tool being proposed.
        arguments: The validated argument mapping.

    Returns:
        A deterministic string suitable for use as a dict key or set member.
    """
    parts = [f"{k}={v!r}" for k, v in sorted(arguments.items())]
    return f"{tool_name}|{'|'.join(parts)}"
