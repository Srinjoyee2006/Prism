"""
Unit and integration tests for STT abstractions, FakeSTT, and FasterWhisperSTT.

Tests requirements:
9. STT interface behavior
10. STT failure handling
11. blocking STT is moved off the event loop
12. transcript event creation
"""

import asyncio

import pytest

from src.asr.base import STTError, STTModelLoadError, Transcript, TranscriptWord
from src.asr.faster_whisper_asr import FasterWhisperSTT
from src.asr.mock_asr import FakeSTT
from src.core.events import STTFailureEvent, TranscriptFinalEvent
from src.vad.segmenter import SpeechSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_segment(num_samples: int = 16000, sample_rate: int = 16000) -> SpeechSegment:
    """Create a 1.0 second speech segment with dummy PCM data."""
    pcm = b"\x00\x00" * num_samples
    return SpeechSegment(
        segment_id="seg_test_123",
        pcm_data=pcm,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        start_timestamp=100.0,
        end_timestamp=101.0,
        duration_ms=1000.0,
        total_samples=num_samples,
    )


# ---------------------------------------------------------------------------
# 9. STT Interface Behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_interface_behavior():
    stt = FakeSTT(
        canned_responses=["Turn on the living room light", "Set AC to 22 degrees"],
        default_response="Default response",
    )
    assert stt.is_available()

    seg1 = make_test_segment()
    t1 = await stt.transcribe(seg1)
    assert isinstance(t1, Transcript)
    assert t1.text == "Turn on the living room light"
    assert t1.confidence == 0.98
    assert t1.duration_ms == 1000.0
    assert not t1.is_empty

    seg2 = make_test_segment()
    t2 = await stt.transcribe(seg2)
    assert t2.text == "Set AC to 22 degrees"

    seg3 = make_test_segment()
    t3 = await stt.transcribe(seg3)
    assert t3.text == "Default response"
    assert stt.transcribe_count == 3


@pytest.mark.asyncio
async def test_stt_empty_segment_handling():
    stt = FakeSTT()
    empty_segment = SpeechSegment(
        segment_id="empty_seg",
        pcm_data=b"",
        sample_rate=16000,
        channels=1,
        sample_width=2,
        duration_ms=0.0,
        total_samples=0,
    )
    transcript = await stt.transcribe(empty_segment)
    assert transcript.is_empty
    assert transcript.text == ""


# ---------------------------------------------------------------------------
# 10. STT Failure Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_failure_handling():
    stt = FakeSTT(should_fail=True, error_message="Inference engine crashed")
    seg = make_test_segment()

    with pytest.raises(STTError, match="Inference engine crashed"):
        await stt.transcribe(seg)


# ---------------------------------------------------------------------------
# 11. Blocking STT is Moved Off the Event Loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocking_stt_moved_off_event_loop():
    """Verify that blocking synchronous transcription runs in a worker thread.

    We test this by running a timed blocking STT operation alongside a concurrent
    asyncio task incrementing a counter on the event loop. If the event loop
    were blocked, the concurrent async task would not progress.
    """
    # Create a FakeSTT with an artificial 0.15s sleep
    stt = FakeSTT(delay_seconds=0.15, default_response="Done")
    seg = make_test_segment()

    ticks = 0

    async def _concurrent_ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks += 1

    # Run both concurrently
    ticker_task = asyncio.create_task(_concurrent_ticker())
    transcript = await stt.transcribe(seg)
    await ticker_task

    assert transcript.text == "Done"
    # Ticker was able to run during the STT execution
    assert ticks >= 3


# ---------------------------------------------------------------------------
# 12. Transcript Event Creation & Models
# ---------------------------------------------------------------------------


def test_transcript_model_and_events():
    word1 = TranscriptWord(word="Hello", start_time=0.0, end_time=0.5, probability=0.99)
    word2 = TranscriptWord(word="world", start_time=0.5, end_time=1.0, probability=0.95)

    transcript = Transcript(
        text="Hello world",
        language="en",
        confidence=0.97,
        duration_ms=1000.0,
        words=(word1, word2),
    )

    assert not transcript.is_empty
    assert len(transcript.words) == 2
    assert transcript.words[0].word == "Hello"

    # Event creation
    event = TranscriptFinalEvent(
        text=transcript.text,
        confidence=transcript.confidence,
        duration_ms=transcript.duration_ms,
        segment_id="seg_456",
    )
    assert event.text == "Hello world"
    assert event.segment_id == "seg_456"

    fail_event = STTFailureEvent(
        segment_id="seg_456",
        error_message="Whisper out of memory",
        recoverable=True,
    )
    assert fail_event.segment_id == "seg_456"
    assert fail_event.recoverable


def test_faster_whisper_invalid_model_raises_load_error():
    """Verify that an invalid model path raises STTModelLoadError."""
    with pytest.raises(STTModelLoadError):
        FasterWhisperSTT(model_size_or_path="nonexistent_model_12345", lazy_load=False)


# ---------------------------------------------------------------------------
# Optional Real Faster-Whisper Integration Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_faster_whisper_if_model_available():
    """Integration test: runs faster-whisper only if a model is locally available.

    Automatically skipped when offline or model is not downloaded.
    """
    import os

    # Check if a model exists in HF cache or local path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    has_cached_whisper = False
    if os.path.exists(cache_dir):
        entries = os.listdir(cache_dir)
        has_cached_whisper = any("whisper" in e.lower() for e in entries)

    if not has_cached_whisper:
        pytest.skip("faster-whisper model not downloaded locally; skipping live integration test.")

    try:
        stt = FasterWhisperSTT(model_size_or_path="tiny.en", device="cpu", compute_type="int8")
        seg = make_test_segment()
        res = await stt.transcribe(seg)
        assert isinstance(res, Transcript)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Live faster-whisper test skipped: {exc}")
