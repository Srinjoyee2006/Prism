from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.asr.base import (
    SpeechToText,
    STTError,
    STTModelLoadError,
    Transcript,
    TranscriptWord,
)

if TYPE_CHECKING:
    from src.vad.segmenter import SpeechSegment

logger = logging.getLogger(__name__)


class FasterWhisperSTT(SpeechToText):
    """Local Speech-to-Text provider backed by faster-whisper."""

    def __init__(
        self,
        model_size_or_path: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        download_root: str | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.download_root = download_root

        self._model: Any | None = None
        self._lock = asyncio.Lock()

        if not lazy_load:
            self._load_model_sync()

    def _load_model_sync(self) -> None:
        """Synchronously load the WhisperModel into memory."""
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_size_or_path,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                model_size_or_path=self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
            logger.info("faster-whisper model '%s' successfully loaded.", self.model_size_or_path)
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            raise STTModelLoadError(
                f"Failed to load faster-whisper model '{self.model_size_or_path}': {exc}"
            ) from exc

    def is_available(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._model is not None

    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded asynchronously without blocking the loop during loading."""
        if self._model is not None:
            return

        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model_sync)

    def _sync_transcribe(self, segment: SpeechSegment) -> Transcript:
        """Synchronous transcription execution running inside asyncio.to_thread."""
        if self._model is None:
            raise STTError("WhisperModel is not loaded")

        if len(segment.pcm_data) == 0:
            return Transcript(text="", duration_ms=0.0)

        # Convert int16 PCM to float32 normalized NumPy array
        audio_float32 = segment.to_numpy(dtype="float32")
        if len(audio_float32) == 0:
            return Transcript(text="", duration_ms=0.0)

        try:
            segments, info = self._model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=False,  # VAD already handled by segmenter
            )

            # Segments is a generator; consume it in the worker thread
            recognized_texts: list[str] = []
            words_list: list[TranscriptWord] = []

            for seg in segments:
                recognized_texts.append(seg.text.strip())
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words_list.append(
                            TranscriptWord(
                                word=w.word,
                                start_time=w.start,
                                end_time=w.end,
                                probability=getattr(w, "probability", 1.0),
                            )
                        )

            full_text = " ".join(recognized_texts).strip()

            return Transcript(
                text=full_text,
                language=info.language if hasattr(info, "language") else self.language,
                confidence=getattr(info, "language_probability", 1.0),
                duration_ms=segment.duration_ms,
                words=tuple(words_list),
            )
        except Exception as exc:
            logger.error("Whisper transcription error: %s", exc)
            raise STTError(f"Inference error during transcription: {exc}") from exc

    async def transcribe(self, segment: SpeechSegment) -> Transcript:
        """Transcribe speech segment without blocking the asyncio event loop."""
        if len(segment.pcm_data) == 0:
            return Transcript(text="", duration_ms=0.0)

        await self._ensure_model_loaded()

        # Delegate CPU-heavy inference to thread pool
        return await asyncio.to_thread(self._sync_transcribe, segment)
