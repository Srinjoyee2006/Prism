"""
Base abstractions, data models, and exceptions for Text-to-Speech (TTS) subsystem.

Provides provider-independent interfaces supporting streaming chunk generation,
low-latency asynchronous execution, and cancellation safety.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class TTSError(Exception):
    """Base exception for all TTS subsystem errors."""


class TTSConnectionError(TTSError):
    """Raised when the TTS backend engine or server cannot be reached."""


class TTSCancelledError(TTSError):
    """Raised when speech synthesis generation is cancelled mid-stream."""


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A discrete synthesized PCM audio chunk.

    Attributes:
        pcm_data: Raw byte representation of linear PCM samples.
        sample_rate: Audio sampling frequency in Hz (typically 24000 for Kokoro).
        channels: Channel count (1 for mono, 2 for stereo).
        sample_width: Byte depth per sample (2 for 16-bit PCM).
        text_segment: Corresponding text substring synthesized for this chunk.
        is_terminal: True if this is the final chunk of the utterance.
    """

    pcm_data: bytes = field(default=b"", repr=False)
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2
    text_segment: str = ""
    is_terminal: bool = False

    @property
    def sample_count(self) -> int:
        """Total number of audio samples in this chunk."""
        bytes_per_sample = self.channels * self.sample_width
        return len(self.pcm_data) // bytes_per_sample if bytes_per_sample > 0 else 0

    @property
    def duration_seconds(self) -> float:
        """Calculated temporal duration of this chunk in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.sample_count / self.sample_rate

    @property
    def duration_ms(self) -> float:
        """Temporal duration in milliseconds."""
        return self.duration_seconds * 1000.0


class TTSProvider(ABC):
    """Abstract base class defining the provider-independent TTS interface."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native sampling rate of audio produced by this provider."""
        ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[AudioChunk]:
        """Asynchronously synthesize text into a stream of linear PCM audio chunks.

        Must yield AudioChunk instances as soon as audio packets are synthesized.
        Must respond immediately to asyncio task cancellation.

        Args:
            text: Text utterance to speak.

        Yields:
            AudioChunk instances representing progressive segments of speech.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the underlying TTS engine or service is reachable and ready."""
        ...

    def cancel(self) -> None:
        """Signal cancellation to any currently active background synthesis task.

        Providers should override this if they manage asynchronous streaming sessions
        or persistent connections that must be aborted immediately.
        """
