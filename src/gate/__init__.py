"""
Tool Commit Gate subsystem public API.

Exports the primary classes and enums needed by the rest of the project.
"""

from src.gate.commit_gate import CommitGate, GateDecision, GateOutcome
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus, StagedProposal, build_normalised_key

__all__ = [
    "CommitGate",
    "GateDecision",
    "GateOutcome",
    "ProposalStatus",
    "StagedProposal",
    "ToolController",
    "build_normalised_key",
]
