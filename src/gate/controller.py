"""
ToolController: central authority for authorising tool execution.

Architecture position::

    LLM → ToolProposal → ToolController → CommitGate → MockTool / RealTool

Rules
-----
* The LLM MUST NEVER directly call a tool.
* The ToolController is the **only** component that may execute tools.
* Every ToolProposal must pass through the CommitGate.
* Blocking tool implementations run in a thread pool via ``asyncio.to_thread``.

Concurrency model
-----------------
* One ``asyncio.Task`` per active StagedProposal (see ``_gate_tasks``).
* A shared ``asyncio.Lock`` serialises proposal registration and supersession.
* An ``asyncio.Event`` (``_interrupt_event``) is set by ``notify_user_speech_started``
  and broadcast to *all* in-flight gate tasks.

Proposal supersession
---------------------
If a new proposal for tool X arrives while tool X is still WAITING, the
older proposal is marked SUPERSEDED and its gate task is cancelled.  The
new proposal then starts its own gate window.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from src.gate.commit_gate import CommitGate, GateOutcome
from src.gate.models import ProposalStatus, StagedProposal, build_normalised_key
from src.llm.models import ToolProposal

logger = logging.getLogger(__name__)

# Type alias: a tool handler is either a regular async callable or one that
# wraps a blocking implementation (the controller always invokes it as async).
ToolHandler = Callable[..., Awaitable[Any]]


class ToolController:
    """Receives ToolProposal objects, manages the commit gate, and executes tools.

    Args:
        quiet_window: Seconds of silence required before committing a proposal
                      (forwarded to ``CommitGate``).

    Usage::

        controller = ToolController(quiet_window=0.9)
        controller.register_handler("set_temperature", handle_temperature)

        # When LLM produces a proposal:
        staged = await controller.submit_proposal(proposal)

        # When VAD detects user speech:
        controller.notify_user_speech_started()
    """

    def __init__(self, quiet_window: float = 0.9) -> None:
        self._quiet_window = quiet_window
        self._gate = CommitGate(quiet_window=quiet_window)

        # proposal_id → StagedProposal
        self._proposals: dict[str, StagedProposal] = {}

        # normalised_key → proposal_id  (for supersession / idempotency)
        self._key_index: dict[str, str] = {}

        # proposal_id → asyncio.Task  (gate wait + execution)
        self._gate_tasks: dict[str, asyncio.Task[None]] = {}

        # tool_name → async handler
        self._handlers: dict[str, ToolHandler] = {}

        # Serialises state mutations
        self._lock = asyncio.Lock()

        # Set when user speech starts; broadcast to all gate waiters
        self._interrupt_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, tool_name: str, handler: ToolHandler) -> None:
        """Register an async handler for a named tool.

        Blocking handlers should be wrapped with ``asyncio.to_thread`` before
        passing here, **or** the ToolController will do it automatically if
        the handler is marked as blocking via ``register_blocking_handler``.

        Args:
            tool_name: Must match the ``tool_name`` field of ToolProposal.
            handler: Async callable ``(arguments: dict) -> Any``.
        """
        self._handlers[tool_name] = handler
        logger.debug("ToolController: registered handler for '%s'", tool_name)

    def register_blocking_handler(
        self, tool_name: str, blocking_fn: Callable[..., Any]
    ) -> None:
        """Register a *synchronous* (blocking) tool implementation.

        The controller wraps it in ``asyncio.to_thread`` so the event loop
        is never blocked.

        Args:
            tool_name: Must match the ``tool_name`` field of ToolProposal.
            blocking_fn: Synchronous callable ``(arguments: dict) -> Any``.
        """

        async def _wrapped(arguments: dict[str, Any]) -> Any:
            return await asyncio.to_thread(blocking_fn, arguments)

        self._handlers[tool_name] = _wrapped
        logger.debug(
            "ToolController: registered blocking handler for '%s' (wrapped in to_thread)",
            tool_name,
        )

    # ------------------------------------------------------------------
    # Interrupt notification
    # ------------------------------------------------------------------

    def notify_user_speech_started(self) -> None:
        """Signal that user speech has been detected.

        Sets the shared interrupt event, which is immediately observed by
        all currently-waiting CommitGate instances.  The event is **not**
        automatically cleared — callers must call ``clear_interrupt()`` once
        processing has been reset (e.g. after transition back to LISTENING).
        """
        logger.info("ToolController: interrupt signaled — dropping uncommitted proposals")
        self._interrupt_event.set()

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag to allow future proposals to commit.

        Call this after the barge-in has been handled and the pipeline is
        ready for a new turn (e.g. after SessionState transitions to LISTENING).
        """
        self._interrupt_event.clear()
        logger.debug("ToolController: interrupt flag cleared")

    @property
    def is_interrupted(self) -> bool:
        """Return True if the interrupt event is currently set."""
        return self._interrupt_event.is_set()

    # ------------------------------------------------------------------
    # Proposal submission
    # ------------------------------------------------------------------

    async def submit_proposal(self, proposal: ToolProposal) -> StagedProposal:
        """Accept a ToolProposal from the LLM layer and begin gate processing.

        This method:
        1. Builds the normalised deduplication key.
        2. Checks for an identical in-flight proposal (idempotency).
        3. Supersedes any WAITING proposal for the same tool.
        4. Creates a new StagedProposal and schedules its gate task.

        Args:
            proposal: A ``ToolProposal`` produced by the LLM parser.

        Returns:
            The newly created (or existing identical) ``StagedProposal``.
        """
        norm_key = build_normalised_key(proposal.tool_name, proposal.arguments)

        async with self._lock:
            # --- Idempotency: exact same proposal already in-flight ---
            existing_id = self._key_index.get(norm_key)
            if existing_id:
                existing = self._proposals.get(existing_id)
                if existing and not existing.is_terminal:
                    logger.info(
                        "ToolController: duplicate proposal '%s' ignored (in-flight: %s)",
                        norm_key,
                        existing.proposal_id,
                    )
                    return existing

            # --- Supersession: same tool, different arguments ---
            # Find any WAITING proposal for this tool and cancel it.
            for pid, sp in list(self._proposals.items()):
                if (
                    sp.tool_name == proposal.tool_name
                    and sp.status == ProposalStatus.WAITING
                ):
                    logger.info(
                        "ToolController: superseding proposal %s with new proposal for '%s'",
                        pid,
                        proposal.tool_name,
                    )
                    sp.mark_superseded()
                    # Remove old key from index
                    self._key_index.pop(sp.normalised_key, None)
                    # Cancel the gate task
                    task = self._gate_tasks.pop(pid, None)
                    if task and not task.done():
                        task.cancel()

            # --- Create and register the new StagedProposal ---
            staged = StagedProposal(
                tool_name=proposal.tool_name,
                arguments=dict(proposal.arguments),
                call_id=proposal.call_id,
                normalised_key=norm_key,
            )
            self._proposals[staged.proposal_id] = staged
            self._key_index[norm_key] = staged.proposal_id

            # Schedule the gate-wait → execute pipeline as a Task
            task = asyncio.create_task(
                self._run_proposal(staged),
                name=f"gate-{staged.proposal_id}",
            )
            self._gate_tasks[staged.proposal_id] = task

        logger.info(
            "ToolController: staged proposal %s for tool '%s'",
            staged.proposal_id,
            staged.tool_name,
        )
        return staged

    # ------------------------------------------------------------------
    # Internal gate + execution pipeline
    # ------------------------------------------------------------------

    async def _run_proposal(self, staged: StagedProposal) -> None:
        """Gate-wait → commit → execute pipeline for one StagedProposal.

        This runs as an independent asyncio.Task.  It is cancellation-safe:
        if the Task is cancelled (supersession or teardown) the StagedProposal
        transitions to DROPPED unless it was already SUPERSEDED.
        """
        staged.mark_waiting()

        try:
            decision = await self._gate.wait_for_commit(self._interrupt_event)

            if decision.outcome == GateOutcome.COMMITTED:
                await self._execute_proposal(staged)

            elif decision.outcome == GateOutcome.INTERRUPTED:
                async with self._lock:
                    if staged.status == ProposalStatus.WAITING:
                        staged.mark_dropped()
                logger.info(
                    "ToolController: proposal %s DROPPED (interrupted after %.3fs)",
                    staged.proposal_id,
                    decision.waited_seconds,
                )

            else:  # CANCELLED (task cancellation propagated from wait)
                async with self._lock:
                    if staged.status == ProposalStatus.WAITING:
                        staged.mark_dropped()

        except asyncio.CancelledError:
            async with self._lock:
                if staged.status not in {
                    ProposalStatus.SUPERSEDED,
                    ProposalStatus.DROPPED,
                }:
                    staged.mark_dropped()
            logger.debug(
                "ToolController: gate task for %s cancelled", staged.proposal_id
            )
            # Do NOT re-raise; task cancellation is expected behaviour here.

        finally:
            # Clean up tracking dictionaries
            async with self._lock:
                self._gate_tasks.pop(staged.proposal_id, None)
                self._key_index.pop(staged.normalised_key, None)

    async def _execute_proposal(self, staged: StagedProposal) -> None:
        """Commit and run the tool handler for an approved StagedProposal."""
        async with self._lock:
            staged.mark_committed()

        handler = self._handlers.get(staged.tool_name)
        if handler is None:
            error_msg = f"No handler registered for tool '{staged.tool_name}'"
            logger.error("ToolController: %s", error_msg)
            async with self._lock:
                staged.mark_failed(error_msg)
            return

        async with self._lock:
            staged.mark_executing()

        logger.info(
            "ToolController: executing tool '%s' (proposal %s)",
            staged.tool_name,
            staged.proposal_id,
        )

        try:
            result = await handler(staged.arguments)
            async with self._lock:
                staged.mark_succeeded(result)
            logger.info(
                "ToolController: tool '%s' succeeded -> %r",
                staged.tool_name,
                result,
            )

        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            async with self._lock:
                staged.mark_failed(error_msg)
            logger.error(
                "ToolController: tool '%s' raised %s",
                staged.tool_name,
                error_msg,
            )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_proposal(self, proposal_id: str) -> StagedProposal | None:
        """Retrieve a StagedProposal by its ID (for testing / observation)."""
        return self._proposals.get(proposal_id)

    def all_proposals(self) -> list[StagedProposal]:
        """Return a snapshot of all tracked proposals."""
        return list(self._proposals.values())

    def pending_count(self) -> int:
        """Return the number of proposals not yet in a terminal state."""
        return sum(1 for p in self._proposals.values() if not p.is_terminal)

    async def wait_for_proposal(
        self, proposal_id: str, timeout: float = 5.0
    ) -> StagedProposal | None:
        """Poll until a proposal reaches a terminal state or timeout expires.

        Useful in tests and integration scenarios.

        Args:
            proposal_id: The ``StagedProposal.proposal_id`` to observe.
            timeout: Maximum seconds to wait.

        Returns:
            The terminal ``StagedProposal``, or ``None`` if not found or timed out.
        """
        staged = self._proposals.get(proposal_id)
        if staged is None:
            return None

        deadline = asyncio.get_event_loop().time() + timeout
        while not staged.is_terminal:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return staged          # Return current state (not terminal) on timeout
            await asyncio.sleep(min(0.02, remaining))

        return staged

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for all active gate tasks to finish (used in teardown / tests).

        Args:
            timeout: Maximum seconds to wait before returning.
        """
        tasks = list(self._gate_tasks.values())
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=timeout)
