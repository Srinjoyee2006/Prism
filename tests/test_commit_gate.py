"""
Stage 3 — ToolController and CommitGate test suite.

Tests labelled A–L per project specification:

A  CommitGate commits after full quiet window
B  CommitGate drops when interrupted before window
C  CommitGate handles task cancellation gracefully
D  ProposalStatus lifecycle (PROPOSED → WAITING → COMMITTED → SUCCEEDED)
E  Proposal drops on interrupt (PROPOSED → WAITING → DROPPED)
F  Supersession: new proposal for same tool cancels waiting predecessor
G  Idempotency: identical in-flight proposal is returned, not duplicated
H  No-handler: FAILED with informative message
I  Blocking handler runs via asyncio.to_thread without blocking event loop
J  Interrupt clears after clear_interrupt(); next proposal commits normally
K  Concurrent proposals for different tools both commit independently
L  drain() waits for all in-flight tasks
"""

import asyncio
import time

import pytest

from src.gate.commit_gate import CommitGate, GateOutcome
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus, StagedProposal, build_normalised_key
from src.llm.models import ToolProposal
from src.tools.mock_tools import (
    blocking_track_order,
    mock_failing_tool,
    mock_get_exchange_rate,
    mock_set_temperature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_proposal(
    tool_name: str = "set_temperature",
    arguments: dict | None = None,
    call_id: str | None = None,
) -> ToolProposal:
    """Build a minimal ToolProposal for testing."""
    if arguments is None:
        arguments = {"zone": "living_room", "temperature_c": 22}
    return ToolProposal(
        tool_name=tool_name,
        arguments=arguments,
        call_id=call_id or f"call_test_{tool_name}",
    )


def make_controller(quiet_window: float = 0.05) -> ToolController:
    """Build a ToolController with a very short quiet window for fast tests."""
    return ToolController(quiet_window=quiet_window)


# ---------------------------------------------------------------------------
# Unit tests — build_normalised_key (no async)
# ---------------------------------------------------------------------------


class TestNormalisedKey:
    def test_same_args_same_key(self):
        k1 = build_normalised_key("foo", {"a": 1, "b": 2})
        k2 = build_normalised_key("foo", {"b": 2, "a": 1})
        assert k1 == k2

    def test_different_tool_different_key(self):
        k1 = build_normalised_key("foo", {"a": 1})
        k2 = build_normalised_key("bar", {"a": 1})
        assert k1 != k2

    def test_different_args_different_key(self):
        k1 = build_normalised_key("foo", {"temp": 22})
        k2 = build_normalised_key("foo", {"temp": 24})
        assert k1 != k2

    def test_empty_args(self):
        k = build_normalised_key("ping", {})
        assert "ping" in k

    def test_key_contains_tool_name(self):
        k = build_normalised_key("search_flights", {"origin": "JFK"})
        assert k.startswith("search_flights|")


# ---------------------------------------------------------------------------
# Unit tests — StagedProposal transitions (no async)
# ---------------------------------------------------------------------------


class TestStagedProposalTransitions:
    def test_initial_status_is_proposed(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        assert sp.status == ProposalStatus.PROPOSED
        assert not sp.is_terminal

    def test_waiting(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_waiting()
        assert sp.status == ProposalStatus.WAITING

    def test_committed_records_timestamp(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_committed()
        assert sp.status == ProposalStatus.COMMITTED
        assert sp.committed_at is not None

    def test_succeeded_is_terminal(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_succeeded("result_value")
        assert sp.status == ProposalStatus.SUCCEEDED
        assert sp.is_terminal
        assert sp.result == "result_value"

    def test_failed_is_terminal(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_failed("boom")
        assert sp.status == ProposalStatus.FAILED
        assert sp.is_terminal
        assert sp.error == "boom"

    def test_dropped_is_terminal(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_dropped()
        assert sp.status == ProposalStatus.DROPPED
        assert sp.is_terminal

    def test_superseded_is_terminal(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        sp.mark_superseded()
        assert sp.status == ProposalStatus.SUPERSEDED
        assert sp.is_terminal

    def test_elapsed_seconds_positive(self):
        sp = StagedProposal(tool_name="foo", arguments={})
        time.sleep(0.01)
        assert sp.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# A — CommitGate commits after full quiet window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A_commit_gate_commits_after_window():
    gate = CommitGate(quiet_window=0.05)
    event = asyncio.Event()  # never set → no interruption

    decision = await gate.wait_for_commit(event)

    assert decision.outcome == GateOutcome.COMMITTED
    assert decision.waited_seconds >= 0.04  # at least close to the window


# ---------------------------------------------------------------------------
# B — CommitGate drops when interrupted before window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_B_commit_gate_interrupted_before_window():
    gate = CommitGate(quiet_window=5.0)  # long window
    event = asyncio.Event()

    async def _set_soon():
        await asyncio.sleep(0.02)
        event.set()

    asyncio.create_task(_set_soon())
    decision = await gate.wait_for_commit(event)

    assert decision.outcome == GateOutcome.INTERRUPTED
    assert decision.waited_seconds < 5.0


# ---------------------------------------------------------------------------
# C — CommitGate handles task cancellation gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_C_commit_gate_cancelled():
    gate = CommitGate(quiet_window=5.0)
    event = asyncio.Event()

    async def _run():
        return await gate.wait_for_commit(event)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.02)
    task.cancel()
    decision = await task

    assert decision.outcome == GateOutcome.CANCELLED


# ---------------------------------------------------------------------------
# D — Full happy path: PROPOSED → WAITING → COMMITTED → SUCCEEDED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_D_happy_path_succeeds():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("set_temperature", mock_set_temperature)

    proposal = make_proposal()
    staged = await ctrl.submit_proposal(proposal)

    assert staged.status in {ProposalStatus.PROPOSED, ProposalStatus.WAITING}

    result = await ctrl.wait_for_proposal(staged.proposal_id, timeout=2.0)

    assert result is not None
    assert result.status == ProposalStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["status"] == "ok"


# ---------------------------------------------------------------------------
# E — Proposal drops when interrupt fires during wait window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_E_proposal_drops_on_interrupt():
    ctrl = make_controller(quiet_window=5.0)  # very long window
    ctrl.register_handler("set_temperature", mock_set_temperature)

    proposal = make_proposal()
    staged = await ctrl.submit_proposal(proposal)

    # Wait until gate is actively waiting
    await asyncio.sleep(0.02)
    ctrl.notify_user_speech_started()

    result = await ctrl.wait_for_proposal(staged.proposal_id, timeout=2.0)

    assert result is not None
    assert result.status == ProposalStatus.DROPPED


# ---------------------------------------------------------------------------
# F — Supersession: new proposal replaces WAITING predecessor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_F_supersession_replaces_waiting_proposal():
    ctrl = make_controller(quiet_window=5.0)
    ctrl.register_handler("set_temperature", mock_set_temperature)

    proposal_old = make_proposal(arguments={"zone": "bedroom", "temperature_c": 20})
    staged_old = await ctrl.submit_proposal(proposal_old)

    # Give time for old proposal to enter WAITING
    await asyncio.sleep(0.03)
    assert staged_old.status == ProposalStatus.WAITING

    # Submit a new proposal for the same tool with different args
    proposal_new = make_proposal(arguments={"zone": "bedroom", "temperature_c": 24})
    staged_new = await ctrl.submit_proposal(proposal_new)

    assert staged_old.status == ProposalStatus.SUPERSEDED
    assert staged_old.proposal_id != staged_new.proposal_id

    # Clean up: interrupt so the new proposal doesn't wait 5 s
    ctrl.notify_user_speech_started()
    await ctrl.drain(timeout=2.0)


# ---------------------------------------------------------------------------
# G — Idempotency: identical in-flight proposal returns same object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_G_idempotency_returns_same_proposal():
    ctrl = make_controller(quiet_window=5.0)
    ctrl.register_handler("set_temperature", mock_set_temperature)

    args = {"zone": "kitchen", "temperature_c": 21}
    p1 = make_proposal(arguments=args)
    p2 = make_proposal(arguments=args)  # same arguments, different call_id

    staged_1 = await ctrl.submit_proposal(p1)
    staged_2 = await ctrl.submit_proposal(p2)

    assert staged_1.proposal_id == staged_2.proposal_id  # same object returned

    ctrl.notify_user_speech_started()
    await ctrl.drain(timeout=2.0)


# ---------------------------------------------------------------------------
# H — No handler registered → proposal fails with informative error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_H_no_handler_marks_failed():
    ctrl = make_controller(quiet_window=0.05)
    # Deliberately do NOT register any handler

    proposal = make_proposal(tool_name="unregistered_tool")
    staged = await ctrl.submit_proposal(proposal)
    result = await ctrl.wait_for_proposal(staged.proposal_id, timeout=2.0)

    assert result is not None
    assert result.status == ProposalStatus.FAILED
    assert "No handler" in (result.error or "")


# ---------------------------------------------------------------------------
# I — Blocking handler wrapped in asyncio.to_thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_I_blocking_handler_runs_in_thread():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_blocking_handler("track_order", blocking_track_order)

    proposal = make_proposal(
        tool_name="track_order",
        arguments={"order_id": "ORD-99999"},
    )
    staged = await ctrl.submit_proposal(proposal)
    result = await ctrl.wait_for_proposal(staged.proposal_id, timeout=3.0)

    assert result is not None
    assert result.status == ProposalStatus.SUCCEEDED
    assert result.result["shipment_status"] == "in_transit"


# ---------------------------------------------------------------------------
# J — Interrupt then clear_interrupt(); next proposal commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_J_clear_interrupt_allows_next_proposal():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("set_temperature", mock_set_temperature)

    # Trigger interrupt, drop first proposal
    first = await ctrl.submit_proposal(make_proposal())
    ctrl.notify_user_speech_started()
    r1 = await ctrl.wait_for_proposal(first.proposal_id, timeout=2.0)
    assert r1 is not None
    assert r1.status == ProposalStatus.DROPPED

    # Clear interrupt and submit a new proposal — it should commit
    ctrl.clear_interrupt()

    second = await ctrl.submit_proposal(make_proposal(arguments={"zone": "office", "temperature_c": 23}))
    r2 = await ctrl.wait_for_proposal(second.proposal_id, timeout=2.0)

    assert r2 is not None
    assert r2.status == ProposalStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# K — Concurrent proposals for different tools both commit independently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_K_concurrent_different_tools_both_succeed():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("set_temperature", mock_set_temperature)
    ctrl.register_handler("get_exchange_rate", mock_get_exchange_rate)

    p_temp = make_proposal("set_temperature", {"zone": "hall", "temperature_c": 19})
    p_fx = make_proposal("get_exchange_rate", {"from_currency": "USD", "to_currency": "INR"})

    s_temp, s_fx = await asyncio.gather(
        ctrl.submit_proposal(p_temp),
        ctrl.submit_proposal(p_fx),
    )

    r_temp, r_fx = await asyncio.gather(
        ctrl.wait_for_proposal(s_temp.proposal_id, timeout=3.0),
        ctrl.wait_for_proposal(s_fx.proposal_id, timeout=3.0),
    )

    assert r_temp is not None and r_temp.status == ProposalStatus.SUCCEEDED
    assert r_fx is not None and r_fx.status == ProposalStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# L — drain() waits for all in-flight tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_L_drain_waits_for_all_tasks():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("set_temperature", mock_set_temperature)
    ctrl.register_handler("get_exchange_rate", mock_get_exchange_rate)

    proposals = [
        make_proposal("set_temperature", {"zone": f"room_{i}", "temperature_c": 20 + i})
        for i in range(3)
    ]
    staged_list = [await ctrl.submit_proposal(p) for p in proposals]

    await ctrl.drain(timeout=5.0)

    for s in staged_list:
        assert s.is_terminal, f"Expected terminal, got {s.status}"


# ---------------------------------------------------------------------------
# Extra: tool failure is recorded, not propagated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_tool_records_error():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("bad_tool", mock_failing_tool)

    staged = await ctrl.submit_proposal(make_proposal(tool_name="bad_tool"))
    result = await ctrl.wait_for_proposal(staged.proposal_id, timeout=2.0)

    assert result is not None
    assert result.status == ProposalStatus.FAILED
    assert "RuntimeError" in (result.error or "")


# ---------------------------------------------------------------------------
# Extra: pending_count() decrements to 0 after all proposals finish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_count_reaches_zero():
    ctrl = make_controller(quiet_window=0.05)
    ctrl.register_handler("set_temperature", mock_set_temperature)

    staged = await ctrl.submit_proposal(make_proposal())
    await ctrl.wait_for_proposal(staged.proposal_id, timeout=2.0)

    assert ctrl.pending_count() == 0


# ---------------------------------------------------------------------------
# Extra: CommitGate rejects non-positive quiet_window at construction time
# ---------------------------------------------------------------------------


def test_commit_gate_invalid_window_raises():
    with pytest.raises(ValueError, match="quiet_window"):
        CommitGate(quiet_window=0.0)

    with pytest.raises(ValueError, match="quiet_window"):
        CommitGate(quiet_window=-1.0)
