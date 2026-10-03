from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.audio.frame import AudioFrame


class VADState(str, Enum):
    """Current conversational speech state determined by the VAD."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True, slots=True)
class VADDecision:
    """Decision output produced by a VAD evaluation on an AudioFrame."""

    is_speech: bool
    confidence: float
    frame_index: int = 0
    timestamp: float = 0.0


class VoiceActivityDetector(ABC):
    """Abstract protocol for voice activity detectors."""

    @abstractmethod
    def process_frame(self, frame: AudioFrame) -> VADDecision:
        """Evaluate an audio frame and return a speech activity decision.

        Args:
            frame: 16-bit PCM AudioFrame (typically 16kHz mono).

        Returns:
            VADDecision indicating if speech was present and confidence score.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recurrent states, buffers, and session history."""
