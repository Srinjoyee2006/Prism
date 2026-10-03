from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.asr.base import SpeechToText, STTError, Transcript

if TYPE_CHECKING:
    from src.vad.segmenter import SpeechSegment


class FakeSTT(SpeechToText):
    """Deterministic STT engine for unit and pipeline tests."""

    def __init__(
        self,
        canned_responses: Sequence[str] | None = None,
        default_response: str = "Hello, world!",
        delay_seconds: float = 0.0,
        should_fail: bool = False,
        error_message: str = "Simulated ASR failure",
    ) -> None:
        self.canned_responses: list[str] = (
            list(canned_responses) if canned_responses is not None else []
        )
        self.default_response = default_response
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail
        self.error_message = error_message

        self._cursor: int = 0
        self.transcribe_count: int = 0
        self.last_segment: SpeechSegment | None = None

    def is_available(self) -> bool:
        """Always available in mock mode."""
        return True

    async def transcribe(self, segment: SpeechSegment) -> Transcript:
        """Return next canned response or default text."""
        self.transcribe_count += 1
        self.last_segment = segment

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_fail:
            raise STTError(self.error_message)

        if len(segment.pcm_data) == 0:
            return Transcript(text="", duration_ms=0.0)

        if self.canned_responses and self._cursor < len(self.canned_responses):
            text = self.canned_responses[self._cursor]
            self._cursor += 1
        else:
            text = self.default_response

        return Transcript(
            text=text,
            language="en",
            confidence=0.98,
            duration_ms=segment.duration_ms,
        )
