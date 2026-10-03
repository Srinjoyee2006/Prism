"""
Deterministic mock and energy-based VAD implementations for offline testing.

Allows precise, reproducible unit tests without neural networks or hardware dependencies.
"""

from collections.abc import Sequence

from src.audio.frame import AudioFrame
from src.vad.base import VADDecision, VoiceActivityDetector


class FakeVAD(VoiceActivityDetector):
    """Deterministic VAD for unit testing.

    Can be configured in several modes:
    1. Scripted sequence: returns predetermined boolean decisions.
    2. Energy threshold: computes frame RMS and compares to threshold.
    3. Fixed response: always returns speech or always returns silence.
    """

    def __init__(
        self,
        scripted_decisions: Sequence[bool] | None = None,
        energy_threshold: float | None = None,
        always_speech: bool = False,
        default_confidence: float = 0.95,
    ) -> None:
        self.scripted_decisions: list[bool] = (
            list(scripted_decisions) if scripted_decisions is not None else []
        )
        self.energy_threshold: float | None = energy_threshold
        self.always_speech: bool = always_speech
        self.default_confidence: float = default_confidence

        self._cursor: int = 0
        self._processed_frames: int = 0
        self.reset_count: int = 0

    def process_frame(self, frame: AudioFrame) -> VADDecision:
        """Process frame and return scripted or energy-based decision."""
        self._processed_frames += 1

        if self.scripted_decisions:
            if self._cursor < len(self.scripted_decisions):
                is_speech = self.scripted_decisions[self._cursor]
                self._cursor += 1
            else:
                is_speech = False  # Default to silence once script exhausted
            conf = self.default_confidence if is_speech else (1.0 - self.default_confidence)

        elif self.energy_threshold is not None:
            rms = frame.calculate_rms()
            is_speech = rms >= self.energy_threshold
            conf = min(1.0, rms * 10.0) if is_speech else 0.1

        else:
            is_speech = self.always_speech
            conf = self.default_confidence if is_speech else 0.0

        return VADDecision(
            is_speech=is_speech,
            confidence=conf,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
        )

    def reset(self) -> None:
        """Reset internal cursor and state."""
        self._cursor = 0
        self.reset_count += 1
