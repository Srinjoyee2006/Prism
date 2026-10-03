"""
Deterministic Fake/Mock Text-to-Speech (TTS) implementation for unit and integration testing.

Allows testing of full-duplex conversational streaming, chunk queuing, and cancellation
semantics without requiring audio hardware, GPU, or Kokoro model dependencies.
"""

import asyncio
import logging
import math
import struct
from collections.abc import AsyncIterator

from src.tts.base import AudioChunk, TTSProvider

logger = logging.getLogger(__name__)


def _generate_synthetic_pcm(
    frequency_hz: float = 440.0,
    duration_seconds: float = 0.05,
    sample_rate: int = 24000,
    amplitude: float = 0.2,
) -> bytes:
    """Generate linear 16-bit mono PCM sine wave bytes."""
    total_samples = int(sample_rate * duration_seconds)
    samples: list[int] = []
    max_val = 32767.0 * amplitude
    for i in range(total_samples):
        val = int(max_val * math.sin(2.0 * math.pi * frequency_hz * (i / sample_rate)))
        samples.append(max(min(val, 32767), -32768))
    return struct.pack(f"<{len(samples)}h", *samples)


class FakeTTS(TTSProvider):
    """Deterministic TTS engine producing synthetic PCM audio chunks for tests.

    Tracks synthesized text history and supports controlled latency and mid-stream cancellation.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_count: int = 4,
        chunk_duration_seconds: float = 0.05,
        delay_per_chunk: float = 0.01,
        available: bool = True,
        generate_pcm: bool = True,
    ) -> None:
        """Initialize FakeTTS.

        Args:
            sample_rate: Native sample rate in Hz.
            chunk_count: Number of chunks yielded per synthesis request.
            chunk_duration_seconds: Audio duration represented by each chunk.
            delay_per_chunk: Simulated generation latency (sleep) before each chunk.
            available: Value returned by is_available().
            generate_pcm: If True, generates valid sine wave PCM; if False, generates null bytes.
        """
        self._sample_rate = sample_rate
        self.chunk_count = chunk_count
        self.chunk_duration_seconds = chunk_duration_seconds
        self.delay_per_chunk = delay_per_chunk
        self._available = available
        self.generate_pcm = generate_pcm

        self.synthesized_texts: list[str] = []
        self.cancelled_count: int = 0
        self._cancel_flag: bool = False
        self._active_sessions: set[asyncio.Task] = set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        """Dynamically toggle availability status for failure testing."""
        self._available = available

    def cancel(self) -> None:
        """Signal cancellation to all active synthetic generations."""
        self._cancel_flag = True
        self.cancelled_count += 1
        for task in list(self._active_sessions):
            if not task.done():
                task.cancel()

    def reset_cancellation(self) -> None:
        """Clear cancellation flag for subsequent tests."""
        self._cancel_flag = False

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[AudioChunk]:
        """Yield deterministic AudioChunk segments with simulated latency.

        Args:
            text: Text utterance being synthesized.

        Yields:
            AudioChunk instances.

        Raises:
            asyncio.CancelledError: If generation is cancelled mid-stream.
        """
        self.synthesized_texts.append(text)
        self._cancel_flag = False

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_sessions.add(current_task)

        try:
            for i in range(self.chunk_count):
                if self._cancel_flag:
                    logger.debug("FakeTTS: Cancel flag detected before chunk %d", i)
                    raise asyncio.CancelledError()

                if self.delay_per_chunk > 0:
                    await asyncio.sleep(self.delay_per_chunk)

                if self._cancel_flag:
                    logger.debug("FakeTTS: Cancel flag detected after sleep on chunk %d", i)
                    raise asyncio.CancelledError()

                is_terminal = (i == self.chunk_count - 1)
                if self.generate_pcm:
                    freq = 440.0 + (i * 50.0)
                    pcm_data = _generate_synthetic_pcm(
                        frequency_hz=freq,
                        duration_seconds=self.chunk_duration_seconds,
                        sample_rate=self._sample_rate,
                    )
                else:
                    byte_len = int(self._sample_rate * self.chunk_duration_seconds * 2)
                    pcm_data = b"\x00" * byte_len

                chunk = AudioChunk(
                    pcm_data=pcm_data,
                    sample_rate=self._sample_rate,
                    channels=1,
                    sample_width=2,
                    text_segment=f"{text} [chunk {i+1}/{self.chunk_count}]",
                    is_terminal=is_terminal,
                )
                yield chunk

        except asyncio.CancelledError:
            self.cancelled_count += 1
            logger.debug("FakeTTS: Generation task cancelled for text '%s'", text)
            raise
        finally:
            if current_task is not None:
                self._active_sessions.discard(current_task)
