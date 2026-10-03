"""
Speech segmentation state machine.

Converts a continuous stream of AudioFrames and VADDecisions into discrete
SpeechSegment objects ready for Speech-to-Text (STT) inference.
"""

import collections
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.audio.frame import AudioFrame
from src.vad.base import VADDecision, VADState


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """An isolated segment of human speech audio packaged for STT transcription."""

    segment_id: str = field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:10]}")
    pcm_data: bytes = field(default=b"", repr=False)
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    duration_ms: float = 0.0
    total_samples: int = 0

    def to_numpy(self, dtype: str = "float32") -> np.ndarray:
        """Convert accumulated PCM data to a normalized NumPy array."""
        if len(self.pcm_data) == 0:
            return np.zeros(0, dtype=dtype)

        raw_int16 = np.frombuffer(self.pcm_data, dtype=np.int16)
        if self.channels > 1:
            raw_int16 = raw_int16.reshape(-1, self.channels)

        if dtype == "float32":
            return raw_int16.astype(np.float32) / 32768.0

        return raw_int16.astype(dtype)


class SpeechSegmenter:
    """State machine converting streaming AudioFrames and VAD decisions into SpeechSegments.

    Features:
    - Pre-speech padding ring-buffer to prevent clipping initial consonants.
    - Configurable minimum speech duration to discard spurious noise spikes.
    - Configurable minimum trailing silence duration to mark utterance completion.
    - Clean reset and flush capabilities.
    """

    def __init__(
        self,
        min_speech_duration_ms: float = 100.0,
        min_silence_duration_ms: float = 400.0,
        speech_pad_ms: float = 30.0,
        max_speech_duration_ms: float = 30000.0,  # 30s max before forcing segment
    ) -> None:
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_speech_duration_ms = max_speech_duration_ms

        self._state: VADState = VADState.SILENCE
        self._current_segment_frames: list[AudioFrame] = []
        self._silence_frames: list[AudioFrame] = []
        self._start_timestamp: float = 0.0

        # Circular buffer for pre-speech audio frames
        # Approximate frame duration: ~32ms; 30ms pad ~ 1-2 frames
        self._pre_buffer_max_frames: int = max(2, int(speech_pad_ms / 30.0) + 1)
        self._pre_speech_buffer: collections.deque[AudioFrame] = collections.deque(
            maxlen=self._pre_buffer_max_frames
        )

        # State transition flags
        self._just_started_speech: bool = False
        self._just_ended_speech: bool = False

    @property
    def state(self) -> VADState:
        """Current state (VADState.SILENCE or VADState.SPEECH)."""
        return self._state

    @property
    def is_in_speech(self) -> bool:
        """Return True if currently accumulating speech."""
        return self._state == VADState.SPEECH

    @property
    def just_started_speech(self) -> bool:
        """Return True if speech started on the most recent frame."""
        return self._just_started_speech

    @property
    def just_ended_speech(self) -> bool:
        """Return True if speech ended on the most recent frame."""
        return self._just_ended_speech

    def process(self, frame: AudioFrame, vad_decision: VADDecision) -> SpeechSegment | None:
        """Process an audio frame alongside its VAD decision.

        Returns:
            SpeechSegment if an utterance was completed, otherwise None.
        """
        self._just_started_speech = False
        self._just_ended_speech = False

        if self._state == VADState.SILENCE:
            if vad_decision.is_speech:
                # Transition: SILENCE -> SPEECH
                self._state = VADState.SPEECH
                self._just_started_speech = True
                self._start_timestamp = (
                    self._pre_speech_buffer[0].timestamp
                    if self._pre_speech_buffer
                    else frame.timestamp
                )

                # Prepend buffered pre-speech frames to avoid clipped onset
                self._current_segment_frames = list(self._pre_speech_buffer)
                self._current_segment_frames.append(frame)
                self._silence_frames.clear()
            else:
                # Still in silence; keep pre-speech ring buffer fresh
                self._pre_speech_buffer.append(frame)
            return None

        # self._state == VADState.SPEECH
        if vad_decision.is_speech:
            # Continuing speech: absorb any pending silence frames as mid-speech pauses
            if self._silence_frames:
                self._current_segment_frames.extend(self._silence_frames)
                self._silence_frames.clear()
            self._current_segment_frames.append(frame)

            # Check if exceeded maximum speech duration limit
            total_duration_ms = sum(f.duration_ms for f in self._current_segment_frames)
            if total_duration_ms >= self.max_speech_duration_ms:
                return self._finalize_segment(forced=True)

            return None

        # Speech detected silence frame
        self._silence_frames.append(frame)
        silence_duration_ms = sum(f.duration_ms for f in self._silence_frames)

        if silence_duration_ms >= self.min_silence_duration_ms:
            # Trailing silence threshold met: finalize the speech segment
            self._just_ended_speech = True
            return self._finalize_segment(forced=False)

        return None

    def _finalize_segment(self, forced: bool = False) -> SpeechSegment | None:
        """Finalize and validate the speech segment."""
        # Include silence frames up to speech_pad_ms to avoid clipping word endings
        if not forced and self._silence_frames:
            pad_samples_needed = int(
                (self.speech_pad_ms / 1000.0) * self._current_segment_frames[0].sample_rate
            )
            samples_added = 0
            for sf in self._silence_frames:
                if samples_added < pad_samples_needed:
                    self._current_segment_frames.append(sf)
                    samples_added += sf.num_samples

        total_duration_ms = sum(f.duration_ms for f in self._current_segment_frames)
        all_frames = list(self._current_segment_frames)

        # Reset internal buffers
        self._state = VADState.SILENCE
        self._current_segment_frames.clear()
        self._silence_frames.clear()
        self._pre_speech_buffer.clear()

        # Reject empty or segments shorter than min_speech_duration_ms
        if not all_frames or total_duration_ms < self.min_speech_duration_ms:
            return None

        combined_pcm = b"".join(f.pcm_data for f in all_frames)
        if len(combined_pcm) == 0:
            return None

        sample_rate = all_frames[0].sample_rate
        channels = all_frames[0].channels
        sample_width = all_frames[0].sample_width
        total_samples = len(combined_pcm) // (channels * sample_width)
        end_timestamp = all_frames[-1].timestamp + all_frames[-1].duration_seconds

        return SpeechSegment(
            pcm_data=combined_pcm,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            start_timestamp=self._start_timestamp,
            end_timestamp=end_timestamp,
            duration_ms=total_duration_ms,
            total_samples=total_samples,
        )

    def flush(self) -> SpeechSegment | None:
        """Force completion of any in-progress speech segment."""
        if self._state == VADState.SPEECH and self._current_segment_frames:
            return self._finalize_segment(forced=True)
        self.reset()
        return None

    def reset(self) -> None:
        """Reset the segmenter to idle silence state, clearing all audio buffers."""
        self._state = VADState.SILENCE
        self._current_segment_frames.clear()
        self._silence_frames.clear()
        self._pre_speech_buffer.clear()
        self._just_started_speech = False
        self._just_ended_speech = False
        self._start_timestamp = 0.0
