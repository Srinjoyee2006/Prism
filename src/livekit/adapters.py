"""
LiveKit Transport Adapters for Prism (Stage 7).

Provides bi-directional audio adapters between LiveKit WebRTC transport
and the Prism Voice Agent pipeline:
1. LiveKitAudioSource: Adapts LiveKit remote participant audio tracks into 16kHz
   16-bit mono AudioFrame streams for the existing Silero VAD + faster-whisper pipeline.
2. LiveKitAudioPlayer: Adapts synthesized TTS chunks into LiveKit AudioSource frames,
   providing instantaneous barge-in cancellation by flushing queues and clearing
   LiveKit's internal playout buffers.

Architectural Rule:
The adapter layer bridges WebRTC I/O without modifying core Stage 1-6 logic.
All tool proposals, commit gates, VAD, ASR, and TTS remain independent of LiveKit.
"""

import asyncio
import logging
from typing import Any

from src.audio.frame import AudioFrame
from src.audio.source import AudioSource
from src.core.async_utils import InterruptibleQueue
from src.tts.base import AudioChunk
from src.tts.player import AudioPlayer

logger = logging.getLogger(__name__)


class LiveKitAudioSource(AudioSource):
    """AudioSource implementation adapting LiveKit audio tracks to Prism.

    Receives incoming WebRTC audio from an rtc.Track or rtc.AudioStream,
    buffers the 16kHz 16-bit mono PCM bytes, and yields 512-sample (32ms)
    AudioFrame instances expected by Silero VAD and AudioPipeline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        chunk_size_samples: int = 512,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            chunk_size_samples=chunk_size_samples,
        )
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._buffer = bytearray()
        self._frame_count = 0
        self._stream_task: asyncio.Task[None] | None = None
        self._current_track: Any = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Initialize the audio source."""
        self._is_open = True
        self._buffer.clear()
        self._frame_count = 0
        logger.info("LiveKitAudioSource opened (expecting %dHz %dch).", self.sample_rate, self.channels)

    async def close(self) -> None:
        """Stop and release all streaming resources."""
        self._is_open = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        self._current_track = None
        # Put None into queue to unblock any waiting consumer
        self._queue.put_nowait(None)
        logger.info("LiveKitAudioSource closed.")

    def attach_track(self, track: Any) -> asyncio.Task[None]:
        """Attach to a remote participant's LiveKit AudioTrack and start consuming frames.

        Args:
            track: An rtc.Track or rtc.RemoteAudioTrack instance.

        Returns:
            The background task consuming from the track's AudioStream.
        """
        self._current_track = track
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()

        track_id = getattr(track, "sid", str(id(track)))
        self._stream_task = asyncio.create_task(
            self._stream_worker(track),
            name=f"LiveKitStreamWorker-{track_id}",
        )
        return self._stream_task

    async def _stream_worker(self, track: Any) -> None:
        """Background worker consuming frames from an rtc.AudioStream."""
        try:
            from livekit import rtc

            # If track already is an AudioStream or iterable of frames, or create AudioStream from track
            if isinstance(track, rtc.AudioStream) or hasattr(track, "__aiter__"):
                audio_stream = track
            else:
                audio_stream = rtc.AudioStream(
                    track,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels,
                )

            async for event in audio_stream:
                if not self._is_open:
                    break
                frame = getattr(event, "frame", event)
                self.push_audio_frame(frame)

        except asyncio.CancelledError:
            logger.debug("LiveKitAudioSource stream worker cancelled.")
        except Exception as exc:  # noqa: BLE001
            logger.error("LiveKitAudioSource stream worker encountered error: %s", exc)
        finally:
            if hasattr(audio_stream, "aclose"):
                try:
                    await audio_stream.aclose()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LiveKitAudioSource: Error closing stream: %s", exc)

    def push_audio_frame(self, frame: Any) -> None:
        """Push an rtc.AudioFrame or frame object into the internal buffer."""
        if hasattr(frame, "data"):
            pcm_bytes = bytes(frame.data)
        elif isinstance(frame, (bytes, bytearray)):
            pcm_bytes = bytes(frame)
        else:
            return
        self.push_frame(pcm_bytes)

    def push_frame(self, pcm_data: bytes) -> None:
        """Push raw PCM bytes into the internal chunking buffer.

        Slices incoming bytes into exact chunk_size_samples AudioFrame objects
        ready for VAD processing.
        """
        if not self._is_open:
            return

        self._buffer.extend(pcm_data)
        bytes_per_chunk = self.chunk_size_samples * self.channels * self.sample_width

        while len(self._buffer) >= bytes_per_chunk:
            chunk = bytes(self._buffer[:bytes_per_chunk])
            del self._buffer[:bytes_per_chunk]

            audio_frame = AudioFrame(
                pcm_data=chunk,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
                frame_index=self._frame_count,
            )
            self._frame_count += 1
            self._queue.put_nowait(audio_frame)

    async def read_frame(self) -> AudioFrame | None:
        """Read the next audio frame from the queue.

        Returns:
            AudioFrame if available, None if closed/EOF.
        """
        if not self._is_open and self._queue.empty():
            return None

        try:
            frame = await self._queue.get()
            return frame
        except asyncio.CancelledError:
            return None


class LiveKitAudioPlayer(AudioPlayer):
    """AudioPlayer implementation streaming synthesized TTS chunks to a LiveKit AudioSource.

    Plays audio asynchronously into a WebRTC track. Provides instantaneous
    barge-in cancellation by flushing pending chunks and clearing LiveKit's internal buffer.
    """

    def __init__(
        self,
        rtc_source: Any | None = None,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

        # Use passed rtc_source or initialize a default rtc.AudioSource
        if rtc_source is not None:
            self._rtc_source = rtc_source
        else:
            from livekit import rtc

            self._rtc_source = rtc.AudioSource(sample_rate=sample_rate, num_channels=channels)

        self._queue: InterruptibleQueue[AudioChunk] = InterruptibleQueue()
        self._running = False
        self._cancel_flag = False
        self._is_active_chunk_playing = False
        self._done_event = asyncio.Event()
        self._done_event.set()
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def rtc_source(self) -> Any:
        """Return the underlying rtc.AudioSource."""
        return self._rtc_source

    @property
    def queued_duration(self) -> float:
        """Return the duration of audio queued inside the LiveKit AudioSource."""
        return getattr(self._rtc_source, "queued_duration", 0.0)

    @property
    def is_playing(self) -> bool:
        """True if the player is actively outputting audio or has chunks queued."""
        if self._cancel_flag:
            return False
        return (
            self._is_active_chunk_playing
            or not self._queue.empty()
            or self.queued_duration > 0.001
        )

    def play(self, audio_chunk: AudioChunk | bytes) -> None:
        """Enqueue an audio chunk into the playback queue.

        Must never block the asyncio event loop.
        """
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
        """Atomically discard all pending unplayed chunks from the queue."""
        flushed = self._queue.flush()
        if hasattr(self._rtc_source, "clear_queue"):
            try:
                self._rtc_source.clear_queue()
            except Exception as exc:  # noqa: BLE001
                logger.debug("LiveKitAudioPlayer: Error clearing rtc_source queue: %s", exc)

        if flushed:
            logger.debug("Flushed %d unplayed chunks from LiveKit playback queue", len(flushed))
        return flushed

    def cancel(self) -> None:
        """Immediately stop currently playing audio and discard all queued chunks."""
        self._cancel_flag = True
        self.flush()
        self._is_active_chunk_playing = False
        if hasattr(self._rtc_source, "clear_queue"):
            try:
                self._rtc_source.clear_queue()
            except Exception as exc:  # noqa: BLE001
                logger.debug("LiveKitAudioPlayer: Error clearing rtc_source queue on cancel: %s", exc)
        self._done_event.set()
        logger.info("LiveKitAudioPlayer: Playback cancelled and LiveKit buffer cleared.")

    async def wait_until_done(self, timeout: float | None = None) -> bool:
        """Wait until all queued audio chunks have finished playing.

        Returns:
            True if all chunks played, False if timed out or cancelled.
        """
        if not self.is_playing:
            return True

        try:
            if timeout is not None:
                await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            else:
                await self._done_event.wait()

            # Also wait for playout on rtc_source if available
            if hasattr(self._rtc_source, "wait_for_playout") and not self._cancel_flag:
                res = self._rtc_source.wait_for_playout()
                if asyncio.iscoroutine(res):
                    if timeout is not None:
                        await asyncio.wait_for(res, timeout=timeout)
                    else:
                        await res
            return True
        except asyncio.TimeoutError:
            return False

    async def start(self) -> None:
        """Start the background playback consumer worker."""
        if self._running:
            return
        self._running = True
        self._cancel_flag = False
        self._worker_task = asyncio.create_task(
            self._playback_worker(),
            name="LiveKitAudioPlaybackWorker",
        )
        logger.info("LiveKitAudioPlayer started.")

    async def stop(self) -> None:
        """Stop and tear down the background playback worker."""
        self._running = False
        self.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        if hasattr(self._rtc_source, "aclose"):
            try:
                await self._rtc_source.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("LiveKitAudioPlayer: Error closing rtc_source: %s", exc)

        logger.info("LiveKitAudioPlayer stopped.")

    async def _playback_worker(self) -> None:
        """Worker loop consuming chunks from queue and pushing to LiveKit AudioSource."""
        from livekit import rtc

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

            try:
                self._is_active_chunk_playing = True
                bytes_per_sample = chunk.channels * chunk.sample_width
                num_samples = len(chunk.pcm_data) // (bytes_per_sample if bytes_per_sample > 0 else 2)

                frame = rtc.AudioFrame(
                    data=chunk.pcm_data,
                    sample_rate=chunk.sample_rate,
                    num_channels=chunk.channels,
                    samples_per_channel=num_samples,
                )

                if hasattr(self._rtc_source, "capture_frame"):
                    res = self._rtc_source.capture_frame(frame)
                    if asyncio.iscoroutine(res):
                        await res

            except Exception as exc:  # noqa: BLE001
                logger.debug("LiveKitAudioPlayer: Exception during frame capture: %s", exc)
            finally:
                self._is_active_chunk_playing = False
                self._queue.task_done()
                if self._queue.empty():
                    self._done_event.set()
