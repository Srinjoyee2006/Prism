"""
Event definitions for full-duplex conversational voice agent.

Provides shared dataclasses representing signals across the pipeline:
VAD -> STT -> LLM -> ToolController -> Tools -> TTS.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _generate_event_id() -> str:
    """Generate a unique event identifier."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True)
class BaseEvent:
    """Base class for all system events across the pipeline."""

    event_id: str = field(default_factory=_generate_event_id)
    timestamp: float = field(default_factory=time.time)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# --- Audio Subsystem Events ---


@dataclass(frozen=True, slots=True)
class AudioInputEvent(BaseEvent):
    """Raw microphone audio frame captured from the input stream."""

    pcm_data: bytes = field(default=b"", repr=False)
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM = 2 bytes


@dataclass(frozen=True, slots=True)
class AudioOutputEvent(BaseEvent):
    """Audio frame queued for speaker playback."""

    pcm_data: bytes = field(default=b"", repr=False)
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    is_terminal: bool = False  # True when last audio chunk of current utterance


# --- VAD (Voice Activity Detection) Events ---


@dataclass(frozen=True, slots=True)
class SpeechStartedEvent(BaseEvent):
    """Emitted when Voice Activity Detection detects the onset of user speech."""

    confidence: float = 1.0
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class SpeechEndedEvent(BaseEvent):
    """Emitted when Voice Activity Detection detects user speech cessation/silence."""

    duration_ms: float = 0.0
    total_samples: int = 0


@dataclass(frozen=True, slots=True)
class BargeInDetectedEvent(BaseEvent):
    """Emitted when user speaks while agent is active (thinking or speaking).

    Signals an immediate cancellation request across all active output pipelines.
    """

    latency_ms: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class SpeechSegmentAvailableEvent(BaseEvent):
    """Emitted when the speech segmenter finalizes a discrete speech segment ready for STT."""

    segment_id: str = ""
    duration_ms: float = 0.0
    sample_rate: int = 16000
    total_samples: int = 0
    pcm_bytes_length: int = 0


# --- ASR (Speech-to-Text) Events ---


@dataclass(frozen=True, slots=True)
class TranscriptDeltaEvent(BaseEvent):
    """Partial / unfinalized speech transcription from streaming ASR."""

    text: str = ""
    is_final: bool = False
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class TranscriptFinalEvent(BaseEvent):
    """Final, stabilized speech transcription for a completed user utterance."""

    text: str = ""
    confidence: float = 1.0
    duration_ms: float = 0.0
    segment_id: str = ""


@dataclass(frozen=True, slots=True)
class STTFailureEvent(BaseEvent):
    """Emitted when speech-to-text inference encounters an error."""

    segment_id: str = ""
    error_message: str = ""
    recoverable: bool = True


# --- LLM Subsystem Events ---


@dataclass(frozen=True, slots=True)
class LLMTokenEvent(BaseEvent):
    """Incremental text token streamed from the LLM."""

    token: str = ""
    index: int = 0


@dataclass(frozen=True, slots=True)
class LLMCompletionEvent(BaseEvent):
    """Full completed text response from the LLM."""

    full_text: str = ""
    total_tokens: int = 0
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class ToolCallPlannedEvent(BaseEvent):
    """Emitted when LLM decides to execute an external tool/function."""

    call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


# --- Tool Execution Events ---


@dataclass(frozen=True, slots=True)
class ToolExecutionResultEvent(BaseEvent):
    """Emitted upon completion of a tool execution."""

    call_id: str = ""
    tool_name: str = ""
    success: bool = True
    result: Any = None
    error_message: str | None = None


# --- TTS Subsystem Events ---


@dataclass(frozen=True, slots=True)
class TTSChunkEvent(BaseEvent):
    """Synthesized audio packet produced by the TTS engine."""

    audio_bytes: bytes = field(default=b"", repr=False)
    text_segment: str = ""
    is_last: bool = False


@dataclass(frozen=True, slots=True)
class TTSPlaybackStartedEvent(BaseEvent):
    """Emitted when the speaker starts playing an utterance."""

    utterance_id: str = ""


@dataclass(frozen=True, slots=True)
class TTSPlaybackCompletedEvent(BaseEvent):
    """Emitted when playback finishes without interruption."""

    utterance_id: str = ""
    duration_seconds: float = 0.0


# --- Interruption & Control Events ---


@dataclass(frozen=True, slots=True)
class InterruptionEvent(BaseEvent):
    """Broadcast to immediately halt playback and flush in-flight generation."""

    reason: str = "user_barge_in"
    target_tasks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SessionControlEvent(BaseEvent):
    """Session lifecycle command (e.g. pause, resume, terminate)."""

    action: str = "terminate"  # 'pause', 'resume', 'terminate', 'reset'
    details: dict[str, Any] = field(default_factory=dict)
