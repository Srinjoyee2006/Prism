"""
Text-to-Speech (TTS) and interruptible audio playback subsystem.

Provides provider-independent streaming speech synthesis (with Kokoro and FakeTTS),
asynchronous playback queuing, and low-latency cancellation on user barge-in.
"""

from src.tts.base import (
    AudioChunk,
    TTSCancelledError,
    TTSConnectionError,
    TTSError,
    TTSProvider,
)
from src.tts.kokoro import KokoroTTS
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS
from src.tts.player import (
    AudioPlayer,
    FakeAudioPlayer,
    SoundDeviceAudioPlayer,
)

__all__ = [
    "AudioChunk",
    "AudioPlayer",
    "FakeAudioPlayer",
    "FakeTTS",
    "KokoroTTS",
    "SoundDeviceAudioPlayer",
    "TTSCancelledError",
    "TTSConnectionError",
    "TTSError",
    "TTSPlaybackManager",
    "TTSProvider",
]
