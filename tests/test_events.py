"""
Unit tests for event definitions in src/core/events.py.
"""

from dataclasses import FrozenInstanceError

import pytest

from src.core.events import (
    AudioInputEvent,
    AudioOutputEvent,
    BargeInDetectedEvent,
    BaseEvent,
    InterruptionEvent,
    LLMCompletionEvent,
    LLMTokenEvent,
    SpeechEndedEvent,
    SpeechStartedEvent,
    ToolCallPlannedEvent,
    ToolExecutionResultEvent,
    TranscriptDeltaEvent,
    TranscriptFinalEvent,
    TTSChunkEvent,
)


def test_base_event_defaults():
    """Verify that BaseEvent generates unique IDs and timestamps."""
    event1 = BaseEvent()
    event2 = BaseEvent()

    assert event1.event_id != event2.event_id
    assert len(event1.event_id) == 12
    assert event1.timestamp > 0
    assert "T" in event1.created_at_utc


def test_event_immutability():
    """Events must be immutable dataclasses to prevent side-effects in pipeline."""
    event = TranscriptDeltaEvent(text="hello", is_final=False)
    with pytest.raises(FrozenInstanceError):
        event.text = "world"  # type: ignore


def test_audio_events():
    """Verify audio event payloads and properties."""
    pcm = b"\x00\x00" * 512
    in_event = AudioInputEvent(pcm_data=pcm, sample_rate=16000, channels=1)
    assert in_event.sample_rate == 16000
    assert in_event.channels == 1
    assert in_event.pcm_data == pcm

    out_event = AudioOutputEvent(pcm_data=pcm, is_terminal=True)
    assert out_event.is_terminal is True


def test_vad_and_barge_in_events():
    """Verify speech detection and barge-in event modeling."""
    speech_start = SpeechStartedEvent(confidence=0.95, frame_index=42)
    assert speech_start.confidence == 0.95
    assert speech_start.frame_index == 42

    speech_end = SpeechEndedEvent(duration_ms=1200.0)
    assert speech_end.duration_ms == 1200.0

    barge_in = BargeInDetectedEvent(latency_ms=45.0, confidence=0.98)
    assert barge_in.latency_ms == 45.0
    assert barge_in.confidence == 0.98


def test_transcript_events():
    """Verify partial and finalized speech transcripts."""
    delta = TranscriptDeltaEvent(text="book a", is_final=False, confidence=0.8)
    assert delta.text == "book a"
    assert not delta.is_final

    final = TranscriptFinalEvent(text="book a table", confidence=0.96, duration_ms=850.0)
    assert final.text == "book a table"
    assert final.confidence == 0.96


def test_llm_and_tool_events():
    """Verify LLM streaming tokens and structured tool events."""
    token = LLMTokenEvent(token="Yes", index=0)
    assert token.token == "Yes"

    completion = LLMCompletionEvent(full_text="Yes, reservation confirmed.", total_tokens=5)
    assert completion.total_tokens == 5

    tool_call = ToolCallPlannedEvent(
        call_id="call_123",
        tool_name="reserve_slot",
        arguments={"party_size": 2, "time": "19:00"},
    )
    assert tool_call.tool_name == "reserve_slot"
    assert tool_call.arguments["party_size"] == 2

    tool_res = ToolExecutionResultEvent(
        call_id="call_123",
        tool_name="reserve_slot",
        success=True,
        result={"booking_id": "BK-999"},
    )
    assert tool_res.success is True
    assert tool_res.result["booking_id"] == "BK-999"


def test_tts_and_interruption_events():
    """Verify TTS and interruption events."""
    tts_chunk = TTSChunkEvent(audio_bytes=b"\x01\x02", text_segment="Hello", is_last=False)
    assert tts_chunk.text_segment == "Hello"
    assert not tts_chunk.is_last

    interrupt = InterruptionEvent(reason="user_barge_in", target_tasks=("tts_playback",))
    assert interrupt.reason == "user_barge_in"
    assert "tts_playback" in interrupt.target_tasks
