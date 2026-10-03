"""
Audio frame representations and PCM validation utilities.

Provides the fundamental data structures for raw audio streams:
16000 Hz, mono (1 channel), 16-bit signed PCM (int16).
"""

import math
import struct
import time
from dataclasses import dataclass, field

import numpy as np


class AudioFormatError(ValueError):
    """Raised when audio PCM bytes violate byte alignment or format invariants."""


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """An immutable chunk of raw PCM audio data captured from an audio stream.

    Defaults match standard voice processing pipelines:
    16kHz, mono, 16-bit linear PCM.
    """

    pcm_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM = 2 bytes per sample
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0

    def __post_init__(self) -> None:
        """Validate byte alignment and invariants."""
        if not isinstance(self.pcm_data, (bytes, bytearray)):
            raise AudioFormatError(
                f"pcm_data must be bytes or bytearray, got {type(self.pcm_data).__name__}"
            )
        bytes_per_sample = self.channels * self.sample_width
        if bytes_per_sample <= 0:
            raise AudioFormatError(f"Invalid bytes_per_sample: {bytes_per_sample}")
        if len(self.pcm_data) % bytes_per_sample != 0:
            raise AudioFormatError(
                f"pcm_data length ({len(self.pcm_data)} bytes) is not aligned to "
                f"frame size ({bytes_per_sample} bytes = {self.channels}ch * {self.sample_width}b)"
            )
        if self.sample_rate <= 0:
            raise AudioFormatError(f"Sample rate must be positive, got {self.sample_rate}")

    @property
    def num_samples(self) -> int:
        """Number of discrete audio samples per channel in this frame."""
        return len(self.pcm_data) // (self.channels * self.sample_width)

    @property
    def duration_ms(self) -> float:
        """Duration of this frame in milliseconds."""
        if self.sample_rate == 0:
            return 0.0
        return (self.num_samples / self.sample_rate) * 1000.0

    @property
    def duration_seconds(self) -> float:
        """Duration of this frame in seconds."""
        if self.sample_rate == 0:
            return 0.0
        return self.num_samples / self.sample_rate

    def to_numpy(self, dtype: str = "float32") -> np.ndarray:
        """Convert raw int16 PCM bytes to a normalized float32 or int16 NumPy array.

        Args:
            dtype: Target data type, typically 'float32' in range [-1.0, 1.0]
                   for VAD/Whisper models, or 'int16' for raw integer representation.

        Returns:
            1D NumPy array for mono audio, or 2D array for multi-channel audio.
        """
        if len(self.pcm_data) == 0:
            return np.zeros(0, dtype=dtype)

        # 16-bit PCM is signed little-endian
        raw_int16 = np.frombuffer(self.pcm_data, dtype=np.int16)

        if self.channels > 1:
            raw_int16 = raw_int16.reshape(-1, self.channels)

        if dtype == "float32":
            # Normalize int16 range [-32768, 32767] to [-1.0, 1.0]
            return raw_int16.astype(np.float32) / 32768.0

        return raw_int16.astype(dtype)

    def calculate_rms(self) -> float:
        """Calculate Root Mean Square (RMS) amplitude normalized to [0.0, 1.0]."""
        if len(self.pcm_data) == 0:
            return 0.0

        count = self.num_samples * self.channels
        # Unpack as signed 16-bit integers
        format_str = f"<{count}h"
        samples = struct.unpack(format_str, self.pcm_data)

        sum_squares = sum(s * s for s in samples)
        mean_square = sum_squares / count
        rms = math.sqrt(mean_square)

        # Normalize relative to max int16 value (32767)
        return min(1.0, rms / 32767.0)

    def is_silent(self, threshold_rms: float = 0.005) -> bool:
        """Return True if frame RMS amplitude is below the silence threshold."""
        return self.calculate_rms() < threshold_rms

    @classmethod
    def create_silence(
        cls,
        num_samples: int = 512,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        frame_index: int = 0,
    ) -> "AudioFrame":
        """Factory method to generate a frame of pure digital silence."""
        pcm_data = b"\x00" * (num_samples * channels * sample_width)
        return cls(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frame_index=frame_index,
        )
