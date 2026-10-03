"""
Voice Activity Detection (VAD) and speech segmentation package.
"""

from src.vad.base import VADDecision, VADState, VoiceActivityDetector
from src.vad.mock_vad import FakeVAD
from src.vad.segmenter import SpeechSegment, SpeechSegmenter
from src.vad.silero_vad import SileroVAD

__all__ = [
    "FakeVAD",
    "SileroVAD",
    "SpeechSegment",
    "SpeechSegmenter",
    "VADDecision",
    "VADState",
    "VoiceActivityDetector",
]
