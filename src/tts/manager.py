"""
TTS and Audio Playback Coordinator / Manager.

Coordinates speech synthesis streaming, non-blocking playback queuing,
and immediate barge-in interruption (cancel TTS -> flush queue -> stop audio -> return to LISTENING).
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from src.core.async_utils import AsyncEventBus
from src.core.events import (
    InterruptionEvent,
    TTSChunkEvent,
    TTSPlaybackCompletedEvent,
    TTSPlaybackStartedEvent,
)
from src.core.state import AgentState, SessionState, TurnSpeaker
from src.tts.base import AudioChunk, TTSProvider
from src.tts.player import AudioPlayer

logger = logging.getLogger(__name__)


async def _invoke_callback(cb: Callable[..., Any] | None, *args: Any) -> None:
    """Safely invoke a callback whether it is synchronous or an async coroutine."""
    if cb is None:
        return
    result = cb(*args)
    if asyncio.iscoroutine(result):
        await result


class TTSPlaybackManager:
    """Manages the lifecycle of speech generation, queueing, and interruptible playback.

    Implements full-duplex conversational barge-in semantics:
    User speech onset during active playback immediately cancels synthesis,
    flushes pending audio buffers, halts hardware playback, and resets state.
    """

    def __init__(
        self,
        tts: TTSProvider,
        player: AudioPlayer,
        session_state: SessionState | None = None,
        event_bus: AsyncEventBus | None = None,
        on_chunk: Callable[[AudioChunk], Any] | None = None,
        on_playback_started: Callable[[], Any] | None = None,
        on_playback_completed: Callable[[], Any] | None = None,
        on_interrupted: Callable[[], Any] | None = None,
    ) -> None:
        self.tts = tts
        self.player = player
        self.session_state = session_state
        self.event_bus = event_bus

        # Lifecycle callbacks (accepts both sync and async callables)
        self.on_chunk = on_chunk
        self.on_playback_started = on_playback_started
        self.on_playback_completed = on_playback_completed
        self.on_interrupted = on_interrupted

        self._gen_task: asyncio.Task | None = None
        self._is_synthesizing: bool = False
        self._interrupted: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_speaking(self) -> bool:
        """True if speech is actively synthesizing or playing."""
        return (
            self._is_synthesizing
            or self.player.is_playing
            or (self._gen_task is not None and not self._gen_task.done())
        )

    async def _synthesis_stream_worker(self, text: str, utterance_id: str) -> None:
        """Internal worker consuming chunks from TTS and pushing to player queue."""
        async for chunk in self.tts.synthesize_stream(text):
            if self._interrupted:
                logger.debug("TTSPlaybackManager: Synthesis loop detected interruption flag")
                break

            # Enqueue chunk for asynchronous playback
            self.player.play(chunk)

            if self.event_bus:
                await self.event_bus.publish(
                    TTSChunkEvent(
                        audio_bytes=chunk.pcm_data,
                        text_segment=chunk.text_segment or text,
                        is_last=chunk.is_terminal,
                    )
                )

            await _invoke_callback(self.on_chunk, chunk)

    async def speak(self, text: str) -> bool:
        """Synthesize text and stream audio chunks to the player.

        Blocks until all chunks finish playing, or returns early if interrupted.

        Args:
            text: Text to synthesize and speak.

        Returns:
            True if utterance completed normally, False if interrupted or cancelled.
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        async with self._lock:
            self._interrupted = False
            self._is_synthesizing = True
            utterance_id = uuid.uuid4().hex[:8]

            # Transition session state to SPEAKING via valid state graph (IDLE/LISTENING -> THINKING -> SPEAKING)
            if self.session_state is not None:
                await self.session_state.start_turn(speaker=TurnSpeaker.AGENT)
                current = self.session_state.current_state
                if current in {AgentState.IDLE, AgentState.LISTENING}:
                    await self.session_state.transition_to(
                        AgentState.THINKING,
                        reason="Agent preparing speech response",
                    )
                if self.session_state.current_state == AgentState.THINKING:
                    await self.session_state.transition_to(
                        AgentState.SPEAKING,
                        reason="Agent began speech playback",
                    )

            if self.event_bus:
                await self.event_bus.publish(
                    TTSPlaybackStartedEvent(utterance_id=utterance_id)
                )

            await _invoke_callback(self.on_playback_started)

            # Spawn synthesis streaming into an internal task
            self._gen_task = asyncio.create_task(
                self._synthesis_stream_worker(clean_text, utterance_id),
                name=f"TTSGenWorker-{utterance_id}",
            )

            try:
                # 1. Await synthesis stream completion
                await self._gen_task
            except asyncio.CancelledError:
                logger.info("TTSPlaybackManager: Synthesis task cancelled for utterance '%s'", utterance_id)
                self._interrupted = True
            finally:
                self._is_synthesizing = False
                self._gen_task = None

            # 2. Wait for audio player to finish playing queued chunks (if not interrupted)
            if not self._interrupted:
                await self.player.wait_until_done()

            # 3. Finalize utterance if not interrupted
            if not self._interrupted:
                if self.session_state is not None and self.session_state.current_state == AgentState.SPEAKING:
                    await self.session_state.transition_to(
                        AgentState.LISTENING,
                        reason="Speech playback completed normally",
                    )

                if self.event_bus:
                    await self.event_bus.publish(
                        TTSPlaybackCompletedEvent(utterance_id=utterance_id)
                    )

                await _invoke_callback(self.on_playback_completed)

                return True
            else:
                return False

    async def handle_user_speech_started(self) -> bool:
        """Handle user speech onset during agent speech (Barge-in / Interruption).

        Execution Sequence:
        1. Cancel active TTS generation
        2. Flush playback queue
        3. Stop player playback
        4. Return session state to LISTENING

        Returns:
            True if an active speech playback was interrupted, False if agent was not speaking.
        """
        if not self.is_speaking and not self._is_synthesizing:
            return False

        logger.info("TTSPlaybackManager: Interruption detected -> cancelling TTS and flushing playback")
        self._interrupted = True

        # 1. Cancel active TTS generation
        self.tts.cancel()
        if self._gen_task is not None and not self._gen_task.done():
            self._gen_task.cancel()

        # 2. Flush playback queue
        self.player.flush()

        # 3. Stop playback
        self.player.cancel()

        # 4. Return to LISTENING state
        if self.session_state is not None:
            current = self.session_state.current_state
            if current in {AgentState.SPEAKING, AgentState.THINKING}:
                await self.session_state.transition_to(
                    AgentState.LISTENING,
                    reason="User speech interrupted agent playback (barge-in)",
                )

        if self.event_bus:
            await self.event_bus.publish(
                InterruptionEvent(reason="user_speech_barge_in")
            )

        await _invoke_callback(self.on_interrupted)

        return True

    def cancel(self) -> None:
        """Synchronously request immediate cancellation of TTS and playback."""
        self._interrupted = True
        self.tts.cancel()
        if self._gen_task is not None and not self._gen_task.done():
            self._gen_task.cancel()
        self.player.flush()
        self.player.cancel()
