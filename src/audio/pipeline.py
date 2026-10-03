"""
Asynchronous Voice Input Pipeline.

Coordinates streaming audio flow:
AudioSource -> VAD -> SpeechSegmenter -> STT -> Events
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.asr.base import SpeechToText, STTError, Transcript
from src.audio.frame import AudioFrame
from src.audio.source import AudioSource
from src.core.async_utils import AsyncEventBus
from src.core.events import (
    AudioInputEvent,
    SpeechEndedEvent,
    SpeechSegmentAvailableEvent,
    SpeechStartedEvent,
    STTFailureEvent,
    TranscriptFinalEvent,
)
from src.vad.base import VoiceActivityDetector
from src.vad.segmenter import SpeechSegment, SpeechSegmenter

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Asynchronous pipeline connecting AudioSource, VAD, Segmenter, and STT."""

    def __init__(
        self,
        source: AudioSource,
        vad: VoiceActivityDetector,
        segmenter: SpeechSegmenter,
        stt: SpeechToText,
        event_bus: AsyncEventBus | None = None,
        on_transcript: Callable[[TranscriptFinalEvent], Awaitable[None]] | None = None,
        on_speech_start: Callable[[], Awaitable[None]] | None = None,
        on_speech_end: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.source = source
        self.vad = vad
        self.segmenter = segmenter
        self.stt = stt
        self.event_bus = event_bus

        self.on_transcript = on_transcript
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Return True if the pipeline background processing loop is active."""
        return self._running

    async def process_frame(self, frame: AudioFrame) -> Transcript | None:
        """Process a single audio frame through the VAD -> Segmenter -> STT chain.

        Returns:
            Transcript if this frame triggered segment completion and successful transcription,
            otherwise None.
        """
        # Publish raw audio input event
        if self.event_bus:
            await self.event_bus.publish(
                AudioInputEvent(
                    pcm_data=frame.pcm_data,
                    sample_rate=frame.sample_rate,
                    channels=frame.channels,
                    sample_width=frame.sample_width,
                )
            )

        # 1. Run VAD
        decision = self.vad.process_frame(frame)

        # 2. Feed to SpeechSegmenter
        segment = self.segmenter.process(frame, decision)

        # Handle speech onset transition
        if self.segmenter.just_started_speech:
            logger.debug("AudioPipeline: Speech start detected")
            if self.event_bus:
                await self.event_bus.publish(
                    SpeechStartedEvent(
                        confidence=decision.confidence,
                        frame_index=frame.frame_index,
                    )
                )
            if self.on_speech_start:
                await self.on_speech_start()

        # Handle speech offset transition
        if self.segmenter.just_ended_speech:
            logger.debug("AudioPipeline: Speech end detected")
            if self.event_bus:
                await self.event_bus.publish(
                    SpeechEndedEvent(
                        duration_ms=segment.duration_ms if segment else 0.0,
                        total_samples=segment.total_samples if segment else 0,
                    )
                )
            if self.on_speech_end:
                await self.on_speech_end()

        # 3. If a segment was finalized, transcribe it
        if segment is not None:
            return await self._transcribe_segment(segment)

        return None

    async def _transcribe_segment(self, segment: SpeechSegment) -> Transcript | None:
        """Execute STT on a finalized speech segment and emit events."""
        logger.debug(
            "AudioPipeline: Transcribing segment %s (duration=%.1fms, samples=%d)",
            segment.segment_id,
            segment.duration_ms,
            segment.total_samples,
        )

        if self.event_bus:
            await self.event_bus.publish(
                SpeechSegmentAvailableEvent(
                    segment_id=segment.segment_id,
                    duration_ms=segment.duration_ms,
                    sample_rate=segment.sample_rate,
                    total_samples=segment.total_samples,
                    pcm_bytes_length=len(segment.pcm_data),
                )
            )

        try:
            transcript = await self.stt.transcribe(segment)

            event = TranscriptFinalEvent(
                text=transcript.text,
                confidence=transcript.confidence,
                duration_ms=segment.duration_ms,
                segment_id=segment.segment_id,
            )

            if self.event_bus:
                await self.event_bus.publish(event)

            if self.on_transcript:
                await self.on_transcript(event)

            return transcript

        except STTError as exc:
            logger.error("AudioPipeline: STT error for segment %s: %s", segment.segment_id, exc)
            if self.event_bus:
                await self.event_bus.publish(
                    STTFailureEvent(
                        segment_id=segment.segment_id,
                        error_message=str(exc),
                        recoverable=True,
                    )
                )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioPipeline: Unexpected error during STT: %s", exc)
            if self.event_bus:
                await self.event_bus.publish(
                    STTFailureEvent(
                        segment_id=segment.segment_id,
                        error_message=f"Unexpected error: {exc}",
                        recoverable=False,
                    )
                )
            return None

    async def run(self) -> None:
        """Continuous pipeline loop reading from AudioSource until EOF or stopped."""
        self._running = True
        try:
            await self.source.open()
            while self._running:
                frame = await self.source.read_frame()
                if frame is None:
                    # Source closed or reached EOF
                    break
                await self.process_frame(frame)

            # Flush any trailing segment at end of stream
            final_segment = self.segmenter.flush()
            if final_segment is not None:
                await self._transcribe_segment(final_segment)

        except asyncio.CancelledError:
            logger.debug("AudioPipeline run loop cancelled")
        finally:
            self._running = False
            await self.source.close()

    def start(self) -> asyncio.Task[None]:
        """Start the pipeline as an asynchronous background task."""
        if self._task is not None and not self._task.done():
            return self._task

        self._running = True
        self._task = asyncio.create_task(self.run(), name="audio-pipeline")
        return self._task

    async def stop(self) -> None:
        """Stop the pipeline task and release the audio source."""
        self._running = False
        if self._task is not None:
            if not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        await self.source.close()

    def reset(self) -> None:
        """Reset internal VAD and segmenter states."""
        self.vad.reset()
        self.segmenter.reset()
