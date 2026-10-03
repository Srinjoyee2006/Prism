"""
Microphone audio source adapter using sounddevice.

Captures live microphone audio frames on Windows/macOS/Linux via portaudio/sounddevice
and delivers them asynchronously without blocking the asyncio event loop.
"""

import asyncio
import logging
import time
from typing import Any

from src.audio.frame import AudioFrame
from src.audio.source import AudioSource

logger = logging.getLogger(__name__)


class AudioDeviceError(RuntimeError):
    """Raised when audio hardware / microphone initialization fails."""


class MicrophoneAudioSource(AudioSource):
    """Real microphone audio source wrapping sounddevice.RawInputStream."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        chunk_size_samples: int = 512,
        device_index: int | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            chunk_size_samples=chunk_size_samples,
        )
        self.device_index = device_index
        self._stream: Any | None = None
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_index: int = 0

    @classmethod
    def is_available(cls) -> bool:
        """Check if sounddevice and at least one audio input device are available."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            # Look for at least one input channel
            return any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception:  # noqa: BLE001
            return False

    async def open(self) -> None:
        """Open the sounddevice input stream and start capture."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                "sounddevice is not installed. Please install it to use live microphone input."
            ) from exc

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._frame_index = 0

        dtype = "int16" if self.sample_width == 2 else "float32"

        def _audio_callback(
            indata: bytes,
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            if status:
                logger.warning("Microphone overflow/status: %s", status)
            if self._loop and self._loop.is_running():
                # indata is a buffer/bytes
                self._loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size_samples,
                device=self.device_index,
                channels=self.channels,
                dtype=dtype,
                callback=_audio_callback,
            )
            self._stream.start()
            self._is_open = True
            logger.info(
                "MicrophoneAudioSource opened (rate=%d, channels=%d, blocksize=%d)",
                self.sample_rate,
                self.channels,
                self.chunk_size_samples,
            )
        except Exception as exc:
            raise AudioDeviceError(f"Failed to open microphone stream: {exc}") from exc

    async def close(self) -> None:
        """Stop and close the microphone input stream."""
        self._is_open = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing microphone stream: %s", exc)
            finally:
                self._stream = None

        # Signal any waiter on the queue
        await self._queue.put(None)

    async def read_frame(self) -> AudioFrame | None:
        """Read the next audio frame from the microphone buffer queue."""
        if not self._is_open and self._queue.empty():
            return None

        pcm_bytes = await self._queue.get()
        if pcm_bytes is None:
            return None

        frame = AudioFrame(
            pcm_data=pcm_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            timestamp=time.time(),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame
