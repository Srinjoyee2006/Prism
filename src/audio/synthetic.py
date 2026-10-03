"""
Synthetic and test audio source generators for offline deterministic testing.

Enables testing the entire voice pipeline without microphones, WAV files,
or live audio hardware.
"""

import asyncio
import math
import struct
import time
from collections.abc import Sequence

from src.audio.frame import AudioFrame
from src.audio.source import AudioSource


class SyntheticAudioSource(AudioSource):
    """Audio source providing programmatically generated PCM audio frames.

    Can emit:
    1. Predefined sequence of AudioFrame objects
    2. Continuous silence
    3. Sine wave audio tone bursts (simulating speech activity)
    """

    def __init__(
        self,
        frames: Sequence[AudioFrame] | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        chunk_size_samples: int = 512,
        simulate_realtime: bool = False,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            chunk_size_samples=chunk_size_samples,
        )
        self._frames: list[AudioFrame] = list(frames) if frames else []
        self._cursor: int = 0
        self._simulate_realtime: bool = simulate_realtime

    def add_frame(self, frame: AudioFrame) -> None:
        """Enqueue an additional frame to the synthetic source."""
        self._frames.append(frame)

    def add_silence(self, duration_ms: float) -> None:
        """Append silence frames for the specified duration."""
        samples_needed = int((duration_ms / 1000.0) * self.sample_rate)
        while samples_needed > 0:
            count = min(samples_needed, self.chunk_size_samples)
            frame = AudioFrame.create_silence(
                num_samples=count,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
                frame_index=len(self._frames),
            )
            self._frames.append(frame)
            samples_needed -= count

    def add_sine_tone(
        self,
        duration_ms: float,
        frequency_hz: float = 440.0,
        amplitude: float = 0.5,
    ) -> None:
        """Append sine wave audio tone frames (simulates audio/speech energy)."""
        samples_needed = int((duration_ms / 1000.0) * self.sample_rate)
        max_int16 = 32767.0
        current_sample_idx = 0

        while samples_needed > 0:
            count = min(samples_needed, self.chunk_size_samples)
            pcm_bytes = bytearray()
            for _ in range(count):
                t = current_sample_idx / self.sample_rate
                val = math.sin(2.0 * math.pi * frequency_hz * t) * amplitude * max_int16
                clamped = max(-32768, min(32767, int(val)))
                for _ in range(self.channels):
                    pcm_bytes.extend(struct.pack("<h", clamped))
                current_sample_idx += 1

            frame = AudioFrame(
                pcm_data=bytes(pcm_bytes),
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
                timestamp=time.time(),
                frame_index=len(self._frames),
            )
            self._frames.append(frame)
            samples_needed -= count

    async def open(self) -> None:
        """Start synthetic source."""
        self._is_open = True
        self._cursor = 0

    async def close(self) -> None:
        """Close synthetic source."""
        self._is_open = False

    async def read_frame(self) -> AudioFrame | None:
        """Read next synthetic frame."""
        if not self._is_open or self._cursor >= len(self._frames):
            return None

        frame = self._frames[self._cursor]
        self._cursor += 1

        if self._simulate_realtime:
            await asyncio.sleep(frame.duration_seconds)

        return frame
