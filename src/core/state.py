"""
Agent and Session state management for asynchronous voice pipelines.

Provides thread-safe and cancellation-safe state machine transitions,
conversation turn tracking, and interruption state flags using asyncio primitives.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentState(str, Enum):
    """Lifecycle states of the voice agent."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class TurnSpeaker(str, Enum):
    """Participant initiating a conversation turn."""

    USER = "user"
    AGENT = "agent"


# Map of permissible state transitions to ensure predictable workflow
VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.LISTENING, AgentState.THINKING},
    AgentState.LISTENING: {AgentState.THINKING, AgentState.IDLE, AgentState.INTERRUPTED},
    AgentState.THINKING: {AgentState.SPEAKING, AgentState.INTERRUPTED, AgentState.IDLE},
    AgentState.SPEAKING: {AgentState.LISTENING, AgentState.INTERRUPTED, AgentState.IDLE},
    AgentState.INTERRUPTED: {AgentState.LISTENING, AgentState.IDLE},
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, from_state: AgentState, to_state: AgentState, reason: str = ""):
        message = (
            f"Invalid state transition from '{from_state.value}' to '{to_state.value}'. "
            f"Reason: {reason or 'Not permitted by transition graph.'}"
        )
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason


@dataclass
class ConversationTurn:
    """Represents a single conversational turn in a session."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    speaker: TurnSpeaker = TurnSpeaker.USER
    start_timestamp: float = field(default_factory=time.time)
    end_timestamp: float | None = None
    transcript: str = ""
    was_interrupted: bool = False
    metadata: dict = field(default_factory=dict)

    def close(self, was_interrupted: bool = False) -> None:
        """Mark the turn as finalized."""
        self.end_timestamp = time.time()
        self.was_interrupted = was_interrupted


class SessionState:
    """Async thread-safe session manager coordinating agent state and turn tracking.

    Guarantees atomic transitions and clean event triggers across concurrent tasks.
    """

    def __init__(self, session_id: str | None = None):
        self.session_id: str = session_id or uuid.uuid4().hex
        self.created_at_utc: str = datetime.now(timezone.utc).isoformat()

        # State machine components
        self._state: AgentState = AgentState.IDLE
        self._lock: asyncio.Lock = asyncio.Lock()
        self._state_changed: asyncio.Event = asyncio.Event()
        self._barge_in_event: asyncio.Event = asyncio.Event()

        # Turn history and tracking
        self._history: list[ConversationTurn] = []
        self._active_turn: ConversationTurn | None = None

    @property
    def current_state(self) -> AgentState:
        """Get the current state of the agent."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Return True if agent is actively listening, thinking, or speaking."""
        return self._state in {
            AgentState.LISTENING,
            AgentState.THINKING,
            AgentState.SPEAKING,
        }

    @property
    def is_speaking(self) -> bool:
        """Return True if the agent is actively outputting audio."""
        return self._state == AgentState.SPEAKING

    @property
    def is_interrupted(self) -> bool:
        """Return True if barge-in was signaled and not yet cleared."""
        return self._barge_in_event.is_set()

    @property
    def active_turn(self) -> ConversationTurn | None:
        """Get the currently active conversation turn, if any."""
        return self._active_turn

    @property
    def history(self) -> list[ConversationTurn]:
        """Get a copy of the completed turns history."""
        return list(self._history)

    async def transition_to(
        self, new_state: AgentState, reason: str = "", force: bool = False
    ) -> AgentState:
        """Atomically transition to a new state with validation and event signaling.

        Args:
            new_state: Target AgentState.
            reason: Optional justification for auditing.
            force: If True, bypass transition graph validation (e.g. emergency shutdown).

        Returns:
            The newly assigned AgentState.

        Raises:
            InvalidStateTransitionError: If the transition is illegal and force=False.
        """
        async with self._lock:
            if not force and new_state not in VALID_TRANSITIONS.get(self._state, set()):
                raise InvalidStateTransitionError(self._state, new_state, reason)

            self._state = new_state

            # Handle barge-in flag synchronization
            if new_state == AgentState.INTERRUPTED:
                self._barge_in_event.set()
                if self._active_turn:
                    self._active_turn.was_interrupted = True
            elif new_state in {AgentState.LISTENING, AgentState.IDLE}:
                self._barge_in_event.clear()

            # Signal listeners waiting for state updates
            self._state_changed.set()
            self._state_changed.clear()

            return self._state

    async def wait_for_state(
        self, target_state: AgentState, timeout: float | None = None
    ) -> bool:
        """Wait asynchronously until the agent reaches a target state.

        Args:
            target_state: The state to await.
            timeout: Maximum seconds to wait (None for infinite).

        Returns:
            True when reached, False if timeout expired.
        """
        if self._state == target_state:
            return True

        start_time = time.time()
        while True:
            if self._state == target_state:
                return True
            elapsed = time.time() - start_time
            remaining = timeout - elapsed if timeout is not None else None
            if remaining is not None and remaining <= 0:
                return False

            try:
                # Wait for the next state transition event
                await asyncio.wait_for(self._state_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

    async def start_turn(self, speaker: TurnSpeaker) -> ConversationTurn:
        """Start a new conversation turn, closing any lingering active turn."""
        async with self._lock:
            if self._active_turn is not None:
                self._active_turn.close()
                self._history.append(self._active_turn)

            new_turn = ConversationTurn(speaker=speaker)
            self._active_turn = new_turn
            return new_turn

    async def complete_turn(
        self, transcript: str = "", was_interrupted: bool = False
    ) -> ConversationTurn | None:
        """Finalize the active conversation turn and record it in session history."""
        async with self._lock:
            if self._active_turn is None:
                return None

            turn = self._active_turn
            turn.transcript = transcript
            turn.close(was_interrupted=was_interrupted)
            self._history.append(turn)
            self._active_turn = None
            return turn

    def trigger_barge_in(self) -> None:
        """Synchronously trigger the barge-in event flag for immediate reaction."""
        self._barge_in_event.set()

    def clear_barge_in(self) -> None:
        """Reset the barge-in event flag."""
        self._barge_in_event.clear()

    async def wait_for_barge_in(self, timeout: float | None = None) -> bool:
        """Asynchronously wait until a barge-in event is signaled."""
        try:
            if timeout is not None:
                await asyncio.wait_for(self._barge_in_event.wait(), timeout=timeout)
            else:
                await self._barge_in_event.wait()
            return True
        except asyncio.TimeoutError:
            return False
