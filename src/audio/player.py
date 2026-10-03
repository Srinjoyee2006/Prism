"""
Audio player re-exports for audio subsystem backward compatibility.
"""

from src.tts.player import (
    AudioPlayer,
    FakeAudioPlayer,
    SoundDeviceAudioPlayer,
)

__all__ = [
    "AudioPlayer",
    "FakeAudioPlayer",
    "SoundDeviceAudioPlayer",
]
