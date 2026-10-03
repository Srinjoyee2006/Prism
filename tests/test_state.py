"""
Unit tests for agent state and session manager in src/core/state.py.
"""

import asyncio

import pytest

from src.core.state import (
    AgentState,
    InvalidStateTransitionError,
    SessionState,
    TurnSpeaker,
)


@pytest.mark.asyncio
async def test_session_state_initialization():
    """Verify clean initial state of SessionState."""
    session = SessionState()
    assert session.current_state == AgentState.IDLE
    assert not session.is_active
    assert not session.is_speaking
    assert not session.is_interrupted
    assert session.active_turn is None
    assert len(session.history) == 0


@pytest.mark.asyncio
async def test_valid_state_transitions():
    """Verify normal progression: IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE."""
    session = SessionState()

    await session.transition_to(AgentState.LISTENING)
    assert session.current_state == AgentState.LISTENING
    assert session.is_active

    await session.transition_to(AgentState.THINKING)
    assert session.current_state == AgentState.THINKING

    await session.transition_to(AgentState.SPEAKING)
    assert session.current_state == AgentState.SPEAKING
    assert session.is_speaking

    await session.transition_to(AgentState.IDLE)
    assert session.current_state == AgentState.IDLE


@pytest.mark.asyncio
async def test_invalid_state_transition_raises():
    """Illegal state jumps must raise InvalidStateTransitionError."""
    session = SessionState()
    # IDLE cannot jump directly to SPEAKING
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        await session.transition_to(AgentState.SPEAKING)

    assert exc_info.value.from_state == AgentState.IDLE
    assert exc_info.value.to_state == AgentState.SPEAKING
    # State should remain unchanged
    assert session.current_state == AgentState.IDLE


@pytest.mark.asyncio
async def test_forced_state_transition():
    """Force=True should bypass transition validation (e.g. emergency reset)."""
    session = SessionState()
    await session.transition_to(AgentState.SPEAKING, force=True)
    assert session.current_state == AgentState.SPEAKING


@pytest.mark.asyncio
async def test_barge_in_and_interruption_synchronization():
    """Transitioning to INTERRUPTED must flag barge-in and mark active turn."""
    session = SessionState()
    await session.transition_to(AgentState.LISTENING)
    turn = await session.start_turn(TurnSpeaker.AGENT)

    await session.transition_to(AgentState.INTERRUPTED)
    assert session.current_state == AgentState.INTERRUPTED
    assert session.is_interrupted
    assert turn.was_interrupted is True

    # Transitioning back to LISTENING clears barge-in flag
    await session.transition_to(AgentState.LISTENING)
    assert not session.is_interrupted


@pytest.mark.asyncio
async def test_wait_for_state_success():
    """wait_for_state resolves when target state is entered."""
    session = SessionState()

    async def _transition_delayed():
        await asyncio.sleep(0.05)
        await session.transition_to(AgentState.LISTENING)

    asyncio.create_task(_transition_delayed())
    reached = await session.wait_for_state(AgentState.LISTENING, timeout=1.0)
    assert reached is True


@pytest.mark.asyncio
async def test_wait_for_state_timeout():
    """wait_for_state returns False when timeout expires before transition."""
    session = SessionState()
    reached = await session.wait_for_state(AgentState.SPEAKING, timeout=0.05)
    assert reached is False


@pytest.mark.asyncio
async def test_turn_management():
    """Verify turn recording, completion, and history tracking."""
    session = SessionState()

    turn = await session.start_turn(TurnSpeaker.USER)
    assert session.active_turn is turn
    assert turn.speaker == TurnSpeaker.USER
    assert turn.end_timestamp is None

    completed = await session.complete_turn(transcript="hello world", was_interrupted=False)
    assert completed is turn
    assert completed.transcript == "hello world"
    assert completed.end_timestamp is not None
    assert session.active_turn is None
    assert len(session.history) == 1
    assert session.history[0].turn_id == turn.turn_id
