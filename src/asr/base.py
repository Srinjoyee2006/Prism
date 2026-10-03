from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.vad.segmenter import SpeechSegment


class STTError(Exception):
    """Base exception for speech-to-text inference and provider errors."""


class STTModelLoadError(STTError):
    """Raised when an STT model fails to load into memory."""


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """Word-level timing and confidence metadata."""

    word: str
    start_time: float
    end_time: float
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class Transcript:
    """The structured result of an ASR transcription operation."""

    text: str
    language: str = "en"
    confidence: float = 1.0
    duration_ms: float = 0.0
    words: tuple[TranscriptWord, ...] = field(default_factory=tuple)
    raw_response: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        """Return True if no text was recognized or text is purely whitespace."""
        return len(self.text.strip()) == 0


class SpeechToText(ABC):
    """Abstract interface for local speech-to-text inference engines."""

    @abstractmethod
    async def transcribe(self, segment: SpeechSegment) -> Transcript:
        """Asynchronously transcribe a speech segment to text.

        Must NOT block the calling asyncio event loop.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying model is loaded and ready for inference."""
