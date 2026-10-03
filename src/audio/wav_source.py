"""
WAV file audio source for reading prerecorded audio datasets.

Reads standard 16-bit PCM WAV files, validates audio format invariants,
and streams discrete AudioFrame instances asynchronously.
"""

import asyncio
import time
import wave
from pathlib import Path

from src.audio.frame import AudioFormatError, AudioFrame
from src.audio.source import AudioSource


class WavFileAudioSource(AudioSource):
    """Audio source reading from a local .wav file."""

    def __init__(
        self,
        file_path: str | Path,
        chunk_size_samples: int = 512,
        simulate_realtime: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self._wav_file: wave.Wave_read | None = None
        self._simulate_realtime = simulate_realtime
        self._frame_index: int = 0

        # We will populate sample_rate, channels, and sample_width when opened
        super().__init__(chunk_size_samples=chunk_size_samples)

    async def open(self) -> None:
        """Open the WAV file and validate audio header format."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"WAV file not found: {self.file_path}")

        try:
            self._wav_file = wave.open(str(self.file_path), "rb")  # noqa: SIM115
        except Exception as exc:
            raise AudioFormatError(f"Failed to open WAV file: {exc}") from exc

        self.channels = self._wav_file.getnchannels()
        self.sample_width = self._wav_file.getsampwidth()
        self.sample_rate = self._wav_file.getframerate()
        self._frame_index = 0

        # Verify 16-bit signed PCM
        if self.sample_width != 2:
            self._wav_file.close()
            raise AudioFormatError(
                f"Unsupported sample width: {self.sample_width} bytes ({self.sample_width * 8}-bit). "
                "Only 16-bit PCM (2 bytes) is supported."
            )

        self._is_open = True

    async def close(self) -> None:
        """Close the underlying WAV file descriptor."""
        self._is_open = False
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

    async def read_frame(self) -> AudioFrame | None:
        """Read the next chunk from the WAV file."""
        if not self._is_open or self._wav_file is None:
            return None

        # wave.readframes expects number of multi-channel audio frames (samples)
        raw_bytes = self._wav_file.readframes(self.chunk_size_samples)
        if len(raw_bytes) == 0:
            return None  # End of file reached

        frame = AudioFrame(
            pcm_data=raw_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            timestamp=time.time(),
            frame_index=self._frame_index,
        )
        self._frame_index += 1

        if self._simulate_realtime:
            await asyncio.sleep(frame.duration_seconds)

        return frame
