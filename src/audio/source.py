"""
Abstract base class and protocol for audio sources.

Allows interchangeable audio inputs:
- Real microphone streams (sounddevice)
- Prerecorded WAV files (wave)
- Synthetic / test audio generators
"""

from abc import ABC, abstractmethod
from types import TracebackType

from typing_extensions import Self

from src.audio.frame import AudioFrame


class AudioSource(ABC):
    """Abstract interface for audio capture sources."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        chunk_size_samples: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.chunk_size_samples = chunk_size_samples
        self._is_open: bool = False

    @property
    def is_open(self) -> bool:
        """Return True if the audio source is active and capturing."""
        return self._is_open

    @property
    def bytes_per_frame(self) -> int:
        """Total byte size of one audio frame."""
        return self.chunk_size_samples * self.channels * self.sample_width

    @abstractmethod
    async def open(self) -> None:
        """Initialize and start the audio input source."""

    @abstractmethod
    async def close(self) -> None:
        """Stop and release all audio input resources."""

    @abstractmethod
    async def read_frame(self) -> AudioFrame | None:
        """Asynchronously read the next audio frame.

        Returns:
            AudioFrame if audio data is available.
            None if stream reached End-Of-File (EOF) or was closed.
        """

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
