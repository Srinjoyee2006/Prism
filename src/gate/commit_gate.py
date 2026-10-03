"""
CommitGate: quiet-window timer and interruption barrier.

Responsibilities
----------------
* Start an ``asyncio.sleep`` quiet window (default 0.9 s).
* Cancel the window immediately when the user barges in.
* Return a ``GateDecision`` to the caller describing the outcome.

The CommitGate is intentionally *stateless* between calls: every
``wait_for_commit`` invocation is independent so the ToolController can
manage one gate per StagedProposal.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GateOutcome(str, Enum):
    """Result returned by the commit gate after waiting."""

    COMMITTED = "committed"     # Quiet window elapsed; tool may execute
    INTERRUPTED = "interrupted" # Barge-in arrived before window elapsed
    CANCELLED = "cancelled"     # Caller cancelled the awaiting task


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Immutable result of one gate evaluation."""

    outcome: GateOutcome
    waited_seconds: float       # Actual wall-clock wait before decision


class CommitGate:
    """Async quiet-window gate that can be interrupted by user speech.

    Args:
        quiet_window: Seconds of user silence required before committing
                      (default 0.9 s, matching Full-Duplex-Bench v3 spec).

    Usage::

        gate = CommitGate(quiet_window=0.9)
        decision = await gate.wait_for_commit(interruption_event)
        if decision.outcome == GateOutcome.COMMITTED:
            # safe to execute
        else:
            # drop proposal
    """

    def __init__(self, quiet_window: float = 0.9) -> None:
        if quiet_window <= 0:
            raise ValueError(f"quiet_window must be positive, got {quiet_window!r}")
        self._quiet_window = quiet_window

    @property
    def quiet_window(self) -> float:
        """The configured quiet window in seconds."""
        return self._quiet_window

    async def wait_for_commit(
        self,
        interruption_event: asyncio.Event,
    ) -> GateDecision:
        """Wait for the quiet window or until an interruption fires.

        The method races two coroutines:
        1. ``asyncio.sleep(quiet_window)`` — success path.
        2. ``interruption_event.wait()`` — barge-in path.

        Whichever finishes first wins.  If the caller's *own* Task is
        cancelled (e.g. session teardown), the gate returns CANCELLED.

        Args:
            interruption_event: An ``asyncio.Event`` set by the VAD/barge-in
                                 layer when the user starts speaking.

        Returns:
            A ``GateDecision`` describing the outcome and elapsed time.
        """
        import time
        start = time.monotonic()

        try:
            _, pending = await asyncio.wait(
                [
                    asyncio.ensure_future(asyncio.sleep(self._quiet_window)),
                    asyncio.ensure_future(interruption_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the losing coroutine to avoid background leaks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # expected — we cancelled it
                except Exception as exc:  # pragma: no cover  # noqa: BLE001
                    logger.warning("CommitGate: error cleaning up pending task: %s", exc)

            elapsed = time.monotonic() - start

            # Determine which finished first
            if interruption_event.is_set():
                logger.debug(
                    "CommitGate: INTERRUPTED after %.3fs (window=%.3fs)",
                    elapsed,
                    self._quiet_window,
                )
                return GateDecision(outcome=GateOutcome.INTERRUPTED, waited_seconds=elapsed)

            logger.debug(
                "CommitGate: COMMITTED after %.3fs (window=%.3fs)",
                elapsed,
                self._quiet_window,
            )
            return GateDecision(outcome=GateOutcome.COMMITTED, waited_seconds=elapsed)

        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            logger.debug("CommitGate: CANCELLED after %.3fs", elapsed)
            return GateDecision(outcome=GateOutcome.CANCELLED, waited_seconds=elapsed)
