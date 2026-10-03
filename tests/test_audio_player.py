"""
Unit and integration tests for interruptible audio playback (Stage 6).

Verifies requirements:
C. Audio chunks enter playback queue.
D. Playback consumes chunks asynchronously.
E. User speech interrupts playback.
F. Queue is flushed after interruption.
H. No audio continues playing after cancellation.
I. Existing Stage 1–5 tests continue passing.
"""

import asyncio

import pytest

from src.core.async_utils import AsyncEventBus
from src.core.events import TTSChunkEvent, TTSPlaybackCompletedEvent
from src.core.state import AgentState, SessionState
from src.tts.base import AudioChunk
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS
from src.tts.player import FakeAudioPlayer


@pytest.mark.asyncio
async def test_requirement_c_audio_chunks_enter_playback_queue():
    """Requirement C: Audio chunks enter playback queue without blocking."""
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.05, auto_start=False)

    chunk1 = AudioChunk(pcm_data=b"\x01\x00" * 100, sample_rate=24000)
    chunk2 = AudioChunk(pcm_data=b"\x02\x00" * 100, sample_rate=24000)

    player.play(chunk1)
    player.play(chunk2)

    # Queue should contain both chunks before worker consumes them
    assert player.is_playing is True
    assert player._queue.qsize() == 2

    # Flush them and verify queue had them
    flushed = player.flush()
    assert len(flushed) == 2
    assert flushed[0] == chunk1
    assert flushed[1] == chunk2


@pytest.mark.asyncio
async def test_requirement_d_playback_consumes_chunks_asynchronously():
    """Requirement D: Playback consumes chunks asynchronously from queue."""
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.01, auto_start=True)

    chunks = [
        AudioChunk(pcm_data=b"\x10\x00" * 100, sample_rate=24000),
        AudioChunk(pcm_data=b"\x20\x00" * 100, sample_rate=24000),
        AudioChunk(pcm_data=b"\x30\x00" * 100, sample_rate=24000),
    ]

    for chunk in chunks:
        player.play(chunk)

    done = await player.wait_until_done(timeout=1.0)
    assert done is True
    assert len(player.played_chunks) == 3
    assert player.is_playing is False
    assert len(player.played_bytes) == 600

    await player.stop()


@pytest.mark.asyncio
async def test_requirement_e_user_speech_interrupts_playback():
    """Requirement E: User speech onset interrupts active playback immediately."""
    tts = FakeTTS(chunk_count=6, chunk_duration_seconds=0.05, delay_per_chunk=0.02)
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.03, auto_start=True)
    state = SessionState()

    manager = TTSPlaybackManager(
        tts=tts,
        player=player,
        session_state=state,
    )

    speak_task = asyncio.create_task(manager.speak("This sentence will be interrupted by user speech."))

    # Wait until speech starts and agent enters SPEAKING state
    for _ in range(20):
        if state.current_state == AgentState.SPEAKING:
            break
        await asyncio.sleep(0.01)

    assert state.current_state == AgentState.SPEAKING
    assert manager.is_speaking is True

    # User starts speaking -> trigger interruption
    interrupted = await manager.handle_user_speech_started()
    assert interrupted is True

    # Wait for speak task to complete (should return False)
    result = await speak_task
    assert result is False

    # State must transition back to LISTENING
    assert state.current_state == AgentState.LISTENING
    assert player.is_playing is False

    await player.stop()


@pytest.mark.asyncio
async def test_requirement_f_queue_is_flushed_after_interruption():
    """Requirement F: Playback queue is flushed after interruption and pending audio discarded."""
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.05, auto_start=True)

    # Queue up 5 audio chunks
    for i in range(5):
        player.play(AudioChunk(pcm_data=b"\x00" * 200, sample_rate=24000, text_segment=f"chunk_{i}"))

    assert player.is_playing is True

    # Allow worker to start first chunk
    await asyncio.sleep(0.01)

    # Interruption occurs
    flushed = player.flush()
    player.cancel()

    # Flushed chunks must have been purged from the pending queue
    assert player.flush_count >= 1
    assert len(flushed) > 0 or len(player.flushed_chunks) > 0
    assert player.is_playing is False
    assert player._queue.empty() is True

    await player.stop()


@pytest.mark.asyncio
async def test_requirement_h_no_audio_continues_playing_after_cancellation():
    """Requirement H: No audio continues playing after cancellation."""
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.05, auto_start=True)

    for _ in range(4):
        player.play(AudioChunk(pcm_data=b"\x55" * 400, sample_rate=24000))

    await asyncio.sleep(0.01)
    assert player.is_playing is True

    # Cancel everything
    player.cancel()

    # Verify instantaneous stop
    assert player.is_playing is False
    chunks_before = len(player.played_chunks)

    # Wait to ensure background worker does not continue processing any further chunks
    await asyncio.sleep(0.1)
    assert len(player.played_chunks) == chunks_before
    assert player.is_playing is False

    await player.stop()


@pytest.mark.asyncio
async def test_tts_playback_manager_full_utterance_workflow():
    """Verify normal uninterrupted utterance completes and triggers events."""
    tts = FakeTTS(chunk_count=3, delay_per_chunk=0.005)
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.005, auto_start=True)
    state = SessionState()
    bus = AsyncEventBus()

    received_events = []
    bus.subscribe(TTSChunkEvent, lambda ev: received_events.append(ev))
    bus.subscribe(TTSPlaybackCompletedEvent, lambda ev: received_events.append(ev))

    started_called = False
    completed_called = False

    def on_started():
        nonlocal started_called
        started_called = True

    def on_completed():
        nonlocal completed_called
        completed_called = True

    manager = TTSPlaybackManager(
        tts=tts,
        player=player,
        session_state=state,
        event_bus=bus,
        on_playback_started=on_started,
        on_playback_completed=on_completed,
    )

    success = await manager.speak("Uninterrupted sentence")
    assert success is True
    assert started_called is True
    assert completed_called is True
    assert state.current_state == AgentState.LISTENING

    chunk_events = [e for e in received_events if isinstance(e, TTSChunkEvent)]
    assert len(chunk_events) == 3

    completed_events = [e for e in received_events if isinstance(e, TTSPlaybackCompletedEvent)]
    assert len(completed_events) == 1

    await player.stop()


@pytest.mark.asyncio
async def test_tool_controller_and_tts_interruption_remain_separate():
    """Requirement 7: ToolController and TTS interruption operate on separate domains."""
    from src.gate.controller import ToolController
    from src.gate.models import ProposalStatus
    from src.llm.models import ToolProposal

    # 1. Setup ToolController with pending proposal
    controller = ToolController(quiet_window=0.5)
    controller.register_handler("test_tool", lambda args: "executed")

    proposal = ToolProposal(tool_name="test_tool", arguments={"param": 1})
    staged = await controller.submit_proposal(proposal)
    assert staged.status in {ProposalStatus.WAITING, ProposalStatus.PROPOSED}

    # 2. Setup TTS and Audio Player
    tts = FakeTTS(chunk_count=6, delay_per_chunk=0.05)
    player = FakeAudioPlayer(sample_rate=24000, chunk_delay_seconds=0.05, auto_start=True)
    state = SessionState()
    manager = TTSPlaybackManager(tts=tts, player=player, session_state=state)

    speak_task = asyncio.create_task(manager.speak("Agent is speaking while tool is staged."))
    await asyncio.sleep(0.02)

    # 3. Simulate user speech onset
    # ToolController handles pending proposal:
    controller.notify_user_speech_started()
    # TTS handles active audio playback:
    interrupted = await manager.handle_user_speech_started()

    await asyncio.sleep(0.02)

    assert interrupted is True
    assert staged.status in {ProposalStatus.DROPPED, ProposalStatus.SUPERSEDED}
    assert player.is_playing is False
    assert state.current_state == AgentState.LISTENING

    await speak_task
    await controller.drain(timeout=1.0)
    await player.stop()
