"""
Configuration models and loaders for the voice agent pipeline.

Provides structured, typed configurations for audio streaming, VAD, ASR,
and agent parameters with sensible zero-cost defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    """Audio capture and playback format configuration."""

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM = 2 bytes per sample
    chunk_size_samples: int = 512  # 32ms window at 16kHz
    format: str = "int16"

    @property
    def frame_duration_ms(self) -> float:
        """Calculate duration of one audio frame/chunk in milliseconds."""
        return (self.chunk_size_samples / self.sample_rate) * 1000.0

    @property
    def bytes_per_sample(self) -> int:
        """Bytes consumed per sample (channels * sample_width)."""
        return self.channels * self.sample_width

    @property
    def chunk_size_bytes(self) -> int:
        """Expected byte length of a single raw audio frame."""
        return self.chunk_size_samples * self.bytes_per_sample


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    engine: str = "silero"  # 'silero', 'fake', or 'energy'
    threshold: float = 0.5  # Probability threshold for speech
    neg_threshold: float = 0.35  # Threshold to drop back to silence
    min_speech_duration_ms: int = 100  # Avoid noise spikes (<100ms)
    min_silence_duration_ms: int = 400  # Pause duration to consider speech segment ended
    speech_pad_ms: int = 30  # Padding buffer around detected speech


@dataclass
class ASRConfig:
    """Speech-to-Text inference configuration."""

    engine: str = "faster_whisper"
    model_size: str = "tiny.en"  # Lightweight CPU-friendly default
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    vad_filter: bool = False  # Pipeline performs VAD upstream


@dataclass
class AgentAppConfig:
    """Root configuration aggregating all subsystem configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentAppConfig":
        """Instantiate AgentAppConfig from a raw dictionary."""
        audio_data = data.get("audio", {})
        vad_data = data.get("vad", {})
        asr_data = data.get("asr", {})

        return cls(
            audio=AudioConfig(
                sample_rate=audio_data.get("sample_rate", 16000),
                channels=audio_data.get("channels", 1),
                sample_width=audio_data.get("sample_width", 2),
                chunk_size_samples=audio_data.get("chunk_size_samples", 512),
                format=audio_data.get("format", "int16"),
            ),
            vad=VADConfig(
                engine=vad_data.get("engine", "silero"),
                threshold=float(vad_data.get("threshold", 0.5)),
                neg_threshold=float(vad_data.get("neg_threshold", 0.35)),
                min_speech_duration_ms=int(vad_data.get("min_speech_duration_ms", 100)),
                min_silence_duration_ms=int(vad_data.get("min_silence_duration_ms", 400)),
                speech_pad_ms=int(vad_data.get("speech_pad_ms", 30)),
            ),
            asr=ASRConfig(
                engine=asr_data.get("engine", "faster_whisper"),
                model_size=asr_data.get("model_size", "tiny.en"),
                device=asr_data.get("device", "cpu"),
                compute_type=asr_data.get("compute_type", "int8"),
                language=asr_data.get("language", "en"),
                beam_size=int(asr_data.get("beam_size", 1)),
                vad_filter=bool(asr_data.get("vad_filter", False)),
            ),
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "AgentAppConfig":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)
