"""
Asynchronous, interruptible audio playback subsystem.

Provides playback queue management and non-blocking speaker output
with instantaneous cancellation on user speech onset.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.core.async_utils import InterruptibleQueue
from src.tts.base import AudioChunk

logger = logging.getLogger(__name__)


class AudioPlayer(ABC):
    """Abstract interface for asynchronous, interruptible audio players."""

    @abstractmethod
    def play(self, audio_chunk: AudioChunk | bytes) -> None:
        """Enqueue an audio chunk into the playback queue.

        Must never block the asyncio event loop.
        """
        ...

    @abstractmethod
    def flush(self) -> list[AudioChunk]:
        """Atomically discard all pending unplayed chunks from the queue.

        Returns:
            List of discarded AudioChunk instances.
        """
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Immediately stop currently playing audio and discard all queued chunks."""
        ...

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True if the player is actively outputting audio or has chunks queued."""
        ...

    @abstractmethod
    async def wait_until_done(self, timeout: float | None = None) -> bool:
        """Wait until all queued audio chunks have finished playing.

        Returns:
            True if all chunks played, False if timed out or cancelled.
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the background playback consumer worker."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop and tear down the background playback worker."""
        ...


class SoundDeviceAudioPlayer(AudioPlayer):
    """Hardware audio player utilizing sounddevice and PortAudio for low-latency playback.

    Consumes PCM chunks from an internal InterruptibleQueue asynchronously,
    writing to the output stream via thread execution to avoid blocking the event loop.
    Immediate cancellation is achieved via stream.abort().
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        self._queue: InterruptibleQueue[AudioChunk] = InterruptibleQueue()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._stream: Any = None
        self._is_active_chunk_playing = False
        self._cancel_flag = False
        self._done_event = asyncio.Event()
        self._done_event.set()
        import threading
        self._write_lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        """True if audio is actively playing or queued to be played."""
        return not self._cancel_flag and (
            self._is_active_chunk_playing or not self._queue.empty()
        )

    def _get_stream(self) -> Any:
        """Initialize or return sounddevice.RawOutputStream."""
        with self._write_lock:
            if self._stream is None:
                import sounddevice as sd

                self._stream = sd.RawOutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self.device,
                )
                self._stream.start()
            elif not self._stream.active:
                self._stream.start()
            return self._stream

    async def start(self) -> None:
        """Start background worker task to consume playback queue."""
        if self._running:
            return
        self._running = True
        self._cancel_flag = False
        self._worker_task = asyncio.create_task(
            self._playback_worker(), name="SoundDevicePlaybackWorker"
        )
        logger.info("SoundDeviceAudioPlayer started.")

    def _close_stream(self) -> None:
        """Safely stop and close the stream under lock."""
        with self._write_lock:
            if self._stream is not None:
                try:
                    if self._stream.active:
                        self._stream.stop()
                    self._stream.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Error closing audio stream: %s", exc)
                self._stream = None

    async def stop(self) -> None:
        """Cancel worker task and release audio stream."""
        self._running = False
        self.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self._close_stream)
        logger.info("SoundDeviceAudioPlayer stopped.")

    def play(self, audio_chunk: AudioChunk | bytes) -> None:
        """Enqueue an audio chunk for asynchronous playback."""
        if isinstance(audio_chunk, bytes):
            chunk = AudioChunk(
                pcm_data=audio_chunk,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
        else:
            chunk = audio_chunk

        self._cancel_flag = False
        self._done_event.clear()
        self._queue.put_nowait(chunk)

    def flush(self) -> list[AudioChunk]:
        """Atomically discard all pending chunks in the queue."""
        flushed = self._queue.flush()
        if flushed:
            logger.debug("Flushed %d unplayed audio chunks from playback queue", len(flushed))
        return flushed

    def cancel(self) -> None:
        """Instantly abort hardware stream and discard queued chunks."""
        self._cancel_flag = True
        self.flush()
        self._is_active_chunk_playing = False

        with self._write_lock:
            if self._stream is not None:
                try:
                    # stream.abort() instantaneously halts output buffer playback
                    if self._stream.active:
                        self._stream.abort()
                        self._stream.start()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("SoundDeviceAudioPlayer: Stream abort exception: %s", exc)

        self._done_event.set()
        logger.info("SoundDeviceAudioPlayer: Playback cancelled and stream aborted.")

    async def wait_until_done(self, timeout: float | None = None) -> bool:
        """Wait until all chunks are finished playing."""
        if not self.is_playing:
            return True
        try:
            if timeout is not None:
                await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            else:
                await self._done_event.wait()
            return True
        except asyncio.TimeoutError:
            return False

    def _write_pcm(self, pcm_data: bytes) -> None:
        """Thread worker executing stream.write under lock."""
        try:
            with self._write_lock:
                if self._stream is not None and self._stream.active and not self._cancel_flag:
                    self._stream.write(pcm_data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SoundDevice stream write aborted or error: %s", exc)

    async def _playback_worker(self) -> None:
        """Worker loop reading chunks from queue and writing to sounddevice."""
        while self._running:
            try:
                chunk = await self._queue.get_cancellable()
            except asyncio.CancelledError:
                break

            if self._cancel_flag:
                self._queue.task_done()
                if self._queue.empty():
                    self._done_event.set()
                continue

            if not chunk.pcm_data:
                self._queue.task_done()
                if self._queue.empty():
                    self._done_event.set()
                continue

            self._is_active_chunk_playing = True
            try:
                # Run synchronous blocking stream.write in thread pool safely under lock
                await asyncio.to_thread(self._write_pcm, chunk.pcm_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error writing audio chunk to sound device: %s", exc)
            finally:
                self._is_active_chunk_playing = False
                self._queue.task_done()
                if self._queue.empty() and not self._is_active_chunk_playing:
                    self._done_event.set()


class FakeAudioPlayer(AudioPlayer):
    """Deterministic mock audio player for testing without audio hardware or sounddevice.

    Accurately simulates asynchronous playback queues, consumption latency,
    and immediate cancellation semantics.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_delay_seconds: float = 0.0,
        auto_start: bool = True,
    ) -> None:
        """Initialize FakeAudioPlayer.

        Args:
            sample_rate: Audio sampling frequency in Hz.
            chunk_delay_seconds: Simulated delay per chunk to test async concurrency / interruption.
            auto_start: Automatically start playback worker upon instantiation.
        """
        self.sample_rate = sample_rate
        self.chunk_delay_seconds = chunk_delay_seconds

        self._queue: InterruptibleQueue[AudioChunk] = InterruptibleQueue()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._is_active_chunk_playing = False
        self._cancel_flag = False
        self._done_event = asyncio.Event()
        self._done_event.set()

        # Telemetry for unit testing assertions
        self.played_chunks: list[AudioChunk] = []
        self.flushed_chunks: list[AudioChunk] = []
        self.flush_count: int = 0
        self.cancel_count: int = 0
        self._current_chunk_sleep_task: asyncio.Task | None = None

        if auto_start:
            self._start_worker_sync()

    def _start_worker_sync(self) -> None:
        """Start worker if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._running = True
                self._worker_task = loop.create_task(
                    self._playback_worker(), name="FakeAudioPlayerWorker"
                )
        except RuntimeError:
            pass

    async def start(self) -> None:
        """Explicitly start background playback worker."""
        if self._running:
            return
        self._running = True
        self._cancel_flag = False
        self._worker_task = asyncio.create_task(
            self._playback_worker(), name="FakeAudioPlayerWorker"
        )

    async def stop(self) -> None:
        """Stop worker and cancel any active tasks."""
        self._running = False
        self.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    @property
    def is_playing(self) -> bool:
        """True if audio is actively playing or queued to be played."""
        return not self._cancel_flag and (
            self._is_active_chunk_playing or not self._queue.empty()
        )

    @property
    def played_bytes(self) -> bytes:
        """Concatenated byte payload of all played chunks."""
        return b"".join(chunk.pcm_data for chunk in self.played_chunks)

    def play(self, audio_chunk: AudioChunk | bytes) -> None:
        """Enqueue an audio chunk for asynchronous playback."""
        if not self._running:
            self._start_worker_sync()

        if isinstance(audio_chunk, bytes):
            chunk = AudioChunk(
                pcm_data=audio_chunk,
                sample_rate=self.sample_rate,
            )
        else:
            chunk = audio_chunk

        self._cancel_flag = False
        self._done_event.clear()
        self._queue.put_nowait(chunk)

    def flush(self) -> list[AudioChunk]:
        """Atomically discard all pending unplayed chunks from the queue."""
        flushed = self._queue.flush()
        self.flush_count += 1
        self.flushed_chunks.extend(flushed)
        return flushed

    def cancel(self) -> None:
        """Immediately abort active playback and discard queued chunks."""
        self._cancel_flag = True
        self.cancel_count += 1
        self.flush()
        self._is_active_chunk_playing = False

        if self._current_chunk_sleep_task and not self._current_chunk_sleep_task.done():
            self._current_chunk_sleep_task.cancel()

        self._done_event.set()

    async def wait_until_done(self, timeout: float | None = None) -> bool:
        """Wait until all queued audio chunks have finished playing."""
        if not self.is_playing:
            return True
        try:
            if timeout is not None:
                await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            else:
                await self._done_event.wait()
            return True
        except asyncio.TimeoutError:
            return False

    async def _playback_worker(self) -> None:
        """Consume chunks from queue and simulate playback."""
        while self._running:
            try:
                chunk = await self._queue.get_cancellable()
            except asyncio.CancelledError:
                break

            if self._cancel_flag:
                self._queue.task_done()
                if self._queue.empty():
                    self._done_event.set()
                continue

            if not chunk.pcm_data:
                self._queue.task_done()
                if self._queue.empty():
                    self._done_event.set()
                continue

            self._is_active_chunk_playing = True
            try:
                if self.chunk_delay_seconds > 0:
                    self._current_chunk_sleep_task = asyncio.create_task(
                        asyncio.sleep(self.chunk_delay_seconds)
                    )
                    await self._current_chunk_sleep_task

                if not self._cancel_flag:
                    self.played_chunks.append(chunk)
            except asyncio.CancelledError:
                logger.debug("FakeAudioPlayer: Active chunk playback cancelled during sleep")
            finally:
                self._current_chunk_sleep_task = None
                self._is_active_chunk_playing = False
                self._queue.task_done()
                if self._queue.empty() and not self._is_active_chunk_playing:
                    self._done_event.set()
