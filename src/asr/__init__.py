"""
Automated Speech Recognition (ASR) / Speech-to-Text package.
"""

from src.asr.base import (
    SpeechToText,
    STTError,
    STTModelLoadError,
    Transcript,
    TranscriptWord,
)
from src.asr.faster_whisper_asr import FasterWhisperSTT
from src.asr.mock_asr import FakeSTT

__all__ = [
    "FakeSTT",
    "FasterWhisperSTT",
    "STTError",
    "STTModelLoadError",
    "SpeechToText",
    "Transcript",
    "TranscriptWord",
]
