"""
End-to-end and component tests for Audio Sources and the AudioPipeline.

Tests requirements:
13. complete pipeline using fake VAD + fake STT
14. cancellation behavior
15. reset behavior
- SyntheticAudioSource & WavFileAudioSource validation
- EventBus integration
"""

import asyncio
import tempfile
import wave
from pathlib import Path

import pytest

from src.asr.mock_asr import FakeSTT
from src.audio.frame import AudioFormatError, AudioFrame
from src.audio.microphone import MicrophoneAudioSource
from src.audio.pipeline import AudioPipeline
from src.audio.synthetic import SyntheticAudioSource
from src.audio.wav_source import WavFileAudioSource
from src.core.async_utils import AsyncEventBus
from src.core.events import (
    SpeechEndedEvent,
    SpeechSegmentAvailableEvent,
    SpeechStartedEvent,
    STTFailureEvent,
    TranscriptFinalEvent,
)
from src.vad.mock_vad import FakeVAD
from src.vad.segmenter import SpeechSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_wav_file(
    file_path: Path,
    num_samples: int = 16000,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Create a temporary valid PCM WAV file for testing."""
    import struct

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        # Write sine tone or dummy samples
        raw = struct.pack(f"<{num_samples * channels}h", *([1000] * (num_samples * channels)))
        wf.writeframes(raw)


# ---------------------------------------------------------------------------
# Audio Source Tests
# ---------------------------------------------------------------------------


class TestAudioSources:
    @pytest.mark.asyncio
    async def test_synthetic_audio_source_silence_and_tone(self):
        source = SyntheticAudioSource(sample_rate=16000, chunk_size_samples=512)
        source.add_silence(duration_ms=64.0)  # 2 frames
        source.add_sine_tone(duration_ms=32.0, frequency_hz=440.0)  # 1 frame

        async with source:
            f1 = await source.read_frame()
            assert f1 is not None
            assert f1.num_samples == 512
            assert f1.is_silent()

            f2 = await source.read_frame()
            assert f2 is not None

            f3 = await source.read_frame()
            assert f3 is not None
            assert not f3.is_silent()

            eof = await source.read_frame()
            assert eof is None

    @pytest.mark.asyncio
    async def test_wav_file_audio_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test.wav"
            make_test_wav_file(wav_path, num_samples=1024, sample_rate=16000)

            source = WavFileAudioSource(wav_path, chunk_size_samples=512)
            async with source:
                assert source.is_open
                assert source.sample_rate == 16000
                assert source.channels == 1

                f1 = await source.read_frame()
                assert f1 is not None
                assert f1.num_samples == 512

                f2 = await source.read_frame()
                assert f2 is not None

                eof = await source.read_frame()
                assert eof is None

    @pytest.mark.asyncio
    async def test_wav_file_audio_source_invalid_sample_width(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test_8bit.wav"
            # Create 8-bit WAV (sample width = 1)
            make_test_wav_file(wav_path, sample_width=1)

            source = WavFileAudioSource(wav_path)
            with pytest.raises(AudioFormatError, match="Unsupported sample width"):
                await source.open()

    def test_microphone_source_availability(self):
        # Checking availability must return bool and not raise exceptions
        avail = MicrophoneAudioSource.is_available()
        assert isinstance(avail, bool)


# ---------------------------------------------------------------------------
# 13. Complete Pipeline Using Fake VAD + Fake STT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_pipeline_flow():
    """Verify that an audio stream flows through VAD, Segmenter, and STT to emit transcripts."""
    event_bus = AsyncEventBus()
    received_transcripts: list[TranscriptFinalEvent] = []
    received_events: list[str] = []

    async def _on_transcript(event: TranscriptFinalEvent):
        received_transcripts.append(event)
        received_events.append("transcript")

    async def _on_speech_start(event: SpeechStartedEvent):
        received_events.append("speech_start")

    async def _on_speech_end(event: SpeechEndedEvent):
        received_events.append("speech_end")

    async def _on_segment(event: SpeechSegmentAvailableEvent):
        received_events.append("segment")

    event_bus.subscribe(TranscriptFinalEvent, _on_transcript)
    event_bus.subscribe(SpeechStartedEvent, _on_speech_start)
    event_bus.subscribe(SpeechEndedEvent, _on_speech_end)
    event_bus.subscribe(SpeechSegmentAvailableEvent, _on_segment)

    # 1. Source: 2 frames silence, 6 frames speech, 6 frames silence (each frame = 32ms)
    source = SyntheticAudioSource(chunk_size_samples=512)
    source.add_silence(duration_ms=64.0)  # 2 frames
    source.add_sine_tone(duration_ms=192.0, frequency_hz=300.0)  # 6 frames speech (~192ms)
    source.add_silence(duration_ms=192.0)  # 6 frames trailing silence (trigger end)

    # 2. VAD: 2 False, 6 True, 6 False
    decisions = [False, False] + [True] * 6 + [False] * 6
    vad = FakeVAD(scripted_decisions=decisions)

    # 3. Segmenter: min_speech=100ms, min_silence=120ms (~4 frames)
    segmenter = SpeechSegmenter(
        min_speech_duration_ms=100.0,
        min_silence_duration_ms=120.0,
        speech_pad_ms=0.0,
    )

    # 4. STT: Fake with canned transcription
    stt = FakeSTT(canned_responses=["Open the garage door"])

    pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
        event_bus=event_bus,
    )

    # Run pipeline to completion
    await pipeline.run()

    # Allow event bus async subscriber queues to process
    await asyncio.sleep(0.05)

    assert len(received_transcripts) == 1
    assert received_transcripts[0].text == "Open the garage door"
    assert "speech_start" in received_events
    assert "speech_end" in received_events
    assert "segment" in received_events
    assert "transcript" in received_events


# ---------------------------------------------------------------------------
# 14. Pipeline Cancellation Behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_cancellation():
    """Verify that a running pipeline can be cleanly cancelled without hanging."""
    source = SyntheticAudioSource()
    # Add a large number of frames
    source.add_silence(duration_ms=10000.0)

    vad = FakeVAD(always_speech=False)
    segmenter = SpeechSegmenter()
    stt = FakeSTT()

    pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
    )

    task = pipeline.start()
    assert pipeline.is_running

    # Let it run briefly
    await asyncio.sleep(0.03)

    # Stop pipeline
    await pipeline.stop()
    assert not pipeline.is_running
    assert task.done()


# ---------------------------------------------------------------------------
# 15. Pipeline Reset Behavior
# ---------------------------------------------------------------------------


def test_pipeline_reset():
    source = SyntheticAudioSource()
    vad = FakeVAD(scripted_decisions=[True, True])
    segmenter = SpeechSegmenter()
    stt = FakeSTT()

    pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
    )

    frame = AudioFrame.create_silence(num_samples=512)
    vad.process_frame(frame)
    segmenter.process(frame, vad.process_frame(frame))

    assert vad.reset_count == 0
    pipeline.reset()
    assert vad.reset_count == 1
    assert not segmenter.is_in_speech


# ---------------------------------------------------------------------------
# STT Failure Event Handling in Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_stt_failure_emits_event():
    event_bus = AsyncEventBus()
    failure_events: list[STTFailureEvent] = []

    async def _on_failure(event: STTFailureEvent):
        failure_events.append(event)

    event_bus.subscribe(STTFailureEvent, _on_failure)

    source = SyntheticAudioSource()
    source.add_sine_tone(duration_ms=150.0)
    source.add_silence(duration_ms=150.0)

    vad = FakeVAD(scripted_decisions=[True] * 5 + [False] * 5)
    segmenter = SpeechSegmenter(min_speech_duration_ms=80.0, min_silence_duration_ms=100.0)
    stt = FakeSTT(should_fail=True, error_message="CTranslate2 segmentation fault")

    pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
        event_bus=event_bus,
    )

    await pipeline.run()
    await asyncio.sleep(0.05)

    assert len(failure_events) == 1
    assert "CTranslate2 segmentation fault" in failure_events[0].error_message
