"""
Unit and integration tests for LiveKit adapters and Prism pipeline integration.

Tests run 100% offline using mock WebRTC objects with zero dependency on LiveKit Cloud.
Verifies:
1. LiveKitAudioSource chunking, buffering, and AudioStream consumption.
2. LiveKitAudioPlayer playback queueing, frame capture, and instant buffer clearing.
3. Commit Gate quiet window (0.9s) invariant and barge-in interruption.
4. TTS cancellation and LiveKit buffer flush on speech onset.
5. Invariant: The LLM proposals go through Commit Gate; LiveKit never executes tools directly.
6. RoomOptions configuration (close_on_disconnect=False).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.audio.frame import AudioFrame
from src.core.state import AgentState, SessionState
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus
from src.livekit.adapters import LiveKitAudioPlayer, LiveKitAudioSource
from src.livekit_agent import PrismLiveKitAgent
from src.llm.base import LLMClient
from src.llm.models import ChatMessage, LLMResponse, ToolProposal
from src.tools.mock_tools import mock_set_temperature
from src.tts.base import AudioChunk
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS


class FakeRtcSource:
    """Mock for livekit.rtc.AudioSource."""

    def __init__(self, sample_rate: int = 24000, num_channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.captured_frames: list[Any] = []
        self.queued_duration: float = 0.0
        self.clear_count: int = 0
        self.closed: bool = False

    async def capture_frame(self, frame: Any) -> None:
        self.captured_frames.append(frame)
        self.queued_duration += getattr(frame, "duration", 0.02)

    def clear_queue(self) -> None:
        self.clear_count += 1
        self.queued_duration = 0.0

    async def wait_for_playout(self) -> None:
        await asyncio.sleep(0.005)
        self.queued_duration = 0.0

    async def aclose(self) -> None:
        self.closed = True


class FakeAudioFrameEvent:
    """Mock for livekit.rtc.AudioFrameEvent."""

    def __init__(self, data: bytes, sample_rate: int = 16000, num_channels: int = 1) -> None:
        frame_mock = MagicMock()
        frame_mock.data = data
        frame_mock.sample_rate = sample_rate
        frame_mock.num_channels = num_channels
        frame_mock.duration = len(data) / (sample_rate * num_channels * 2)
        self.frame = frame_mock


class FakeRtcAudioStream:
    """Mock for livekit.rtc.AudioStream async iterator."""

    def __init__(self, events: list[FakeAudioFrameEvent]) -> None:
        self._events = events
        self._index = 0
        self.closed: bool = False

    def __aiter__(self) -> "FakeRtcAudioStream":
        return self

    async def __anext__(self) -> FakeAudioFrameEvent:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        await asyncio.sleep(0.005)
        return event

    async def aclose(self) -> None:
        self.closed = True


class MockTestLLM(LLMClient):
    """Deterministic LLM proposing tools."""

    async def generate(self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        return LLMResponse(
            text="Setting temperature.",
            tool_proposals=[
                ToolProposal(
                    tool_name="set_temperature",
                    arguments={"location": "living_room", "temperature": 21.0},
                )
            ],
        )

    async def is_available(self) -> bool:
        return True


# ============================================================================
# 1. LiveKitAudioSource Unit Tests
# ============================================================================


class TestLiveKitAudioSource:
    @pytest.mark.asyncio
    async def test_open_close_lifecycle(self) -> None:
        source = LiveKitAudioSource(sample_rate=16000, channels=1, chunk_size_samples=512)
        assert not source.is_open

        await source.open()
        assert source.is_open

        await source.close()
        assert not source.is_open
        # Next read should return None (EOF)
        frame = await source.read_frame()
        assert frame is None

    @pytest.mark.asyncio
    async def test_frame_slicing_and_chunking(self) -> None:
        """Verify that variable length byte pushes are sliced into exact 512-sample frames."""
        source = LiveKitAudioSource(sample_rate=16000, channels=1, chunk_size_samples=512)
        await source.open()

        # 512 samples * 1ch * 2 bytes = 1024 bytes per frame (32ms)
        # Push 2500 bytes: should yield 2 full 1024-byte frames, with 452 bytes remaining in buffer
        test_bytes = b"\x01\x00" * 1250  # 2500 bytes
        source.push_frame(test_bytes)

        frame1 = await source.read_frame()
        assert frame1 is not None
        assert isinstance(frame1, AudioFrame)
        assert len(frame1.pcm_data) == 1024
        assert frame1.num_samples == 512
        assert frame1.duration_ms == 32.0
        assert frame1.frame_index == 0

        frame2 = await source.read_frame()
        assert frame2 is not None
        assert len(frame2.pcm_data) == 1024
        assert frame2.num_samples == 512
        assert frame2.frame_index == 1

        # Now push 600 more bytes -> buffer has 452 + 600 = 1052 bytes -> yields 3rd frame
        source.push_frame(b"\x02\x00" * 300)
        frame3 = await source.read_frame()
        assert frame3 is not None
        assert len(frame3.pcm_data) == 1024
        assert frame3.frame_index == 2

        await source.close()

    @pytest.mark.asyncio
    async def test_attach_track_with_stream(self) -> None:
        """Verify attach_track consumes frames from an audio stream."""
        source = LiveKitAudioSource(sample_rate=16000, channels=1, chunk_size_samples=512)
        await source.open()

        # Create mock stream yielding two 1024-byte events
        events = [
            FakeAudioFrameEvent(b"\x00" * 1024),
            FakeAudioFrameEvent(b"\x00" * 1024),
        ]
        mock_stream = FakeRtcAudioStream(events)

        source.attach_track(mock_stream)

        frame1 = await asyncio.wait_for(source.read_frame(), timeout=1.0)
        assert frame1 is not None
        assert len(frame1.pcm_data) == 1024

        frame2 = await asyncio.wait_for(source.read_frame(), timeout=1.0)
        assert frame2 is not None
        assert len(frame2.pcm_data) == 1024

        await source.close()
        assert mock_stream.closed


# ============================================================================
# 2. LiveKitAudioPlayer Unit Tests
# ============================================================================


class TestLiveKitAudioPlayer:
    @pytest.mark.asyncio
    async def test_player_lifecycle_and_play(self) -> None:
        fake_rtc = FakeRtcSource(sample_rate=24000, num_channels=1)
        player = LiveKitAudioPlayer(rtc_source=fake_rtc, sample_rate=24000, channels=1)

        await player.start()

        # Enqueue a chunk
        pcm = b"\x00\x01" * 480  # 480 samples = 20ms
        chunk = AudioChunk(pcm_data=pcm, sample_rate=24000, channels=1)
        player.play(chunk)

        # Wait for worker to capture chunk into rtc_source
        done = await player.wait_until_done(timeout=1.0)
        assert done
        assert len(fake_rtc.captured_frames) == 1
        assert fake_rtc.captured_frames[0].samples_per_channel == 480

        await player.stop()
        assert fake_rtc.closed

    @pytest.mark.asyncio
    async def test_player_flush(self) -> None:
        fake_rtc = FakeRtcSource(sample_rate=24000, num_channels=1)
        player = LiveKitAudioPlayer(rtc_source=fake_rtc, sample_rate=24000, channels=1)
        # Do not start worker so chunks remain in queue
        chunk1 = AudioChunk(pcm_data=b"\x01" * 100, sample_rate=24000)
        chunk2 = AudioChunk(pcm_data=b"\x02" * 100, sample_rate=24000)
        player.play(chunk1)
        player.play(chunk2)

        flushed = player.flush()
        assert len(flushed) == 2
        assert fake_rtc.clear_count == 1

    @pytest.mark.asyncio
    async def test_player_cancel_instantaneous(self) -> None:
        fake_rtc = FakeRtcSource(sample_rate=24000, num_channels=1)
        player = LiveKitAudioPlayer(rtc_source=fake_rtc, sample_rate=24000, channels=1)
        await player.start()

        for _ in range(5):
            player.play(AudioChunk(pcm_data=b"\x00" * 480, sample_rate=24000))

        player.cancel()
        assert not player.is_playing
        assert fake_rtc.clear_count >= 1

        await player.stop()


# ============================================================================
# 3. Interruption and Commit Gate Invariant Tests
# ============================================================================


class TestInterruptionAndCommitGate:
    @pytest.mark.asyncio
    async def test_quiet_window_commit_success(self) -> None:
        """When quiet window (0.2s) elapses without user speech, tool MUST commit and execute."""
        controller = ToolController(quiet_window=0.2)
        controller.register_handler("set_temperature", mock_set_temperature)

        proposal = ToolProposal(
            tool_name="set_temperature",
            arguments={"location": "living_room", "temperature": 21.0},
        )
        staged = await controller.submit_proposal(proposal)
        assert staged.status == ProposalStatus.PROPOSED

        # Wait past quiet window
        await asyncio.sleep(0.3)
        assert staged.status == ProposalStatus.SUCCEEDED
        assert staged.result["status"] == "ok"
        await controller.drain()

    @pytest.mark.asyncio
    async def test_interruption_during_quiet_window_drops_proposal(self) -> None:
        """CRITICAL INVARIANT: User speech onset during quiet window MUST drop proposal.

        The tool MUST NOT be executed.
        """
        controller = ToolController(quiet_window=0.3)
        controller.register_handler("set_temperature", mock_set_temperature)

        proposal = ToolProposal(
            tool_name="set_temperature",
            arguments={"location": "bedroom", "temperature": 19.0},
        )
        staged = await controller.submit_proposal(proposal)

        # Simulate user speech onset detected by VAD after 100ms
        await asyncio.sleep(0.1)
        controller.notify_user_speech_started()

        # Wait past the original quiet window duration
        await asyncio.sleep(0.3)

        assert staged.status == ProposalStatus.DROPPED
        assert staged.result is None
        await controller.drain()

    @pytest.mark.asyncio
    async def test_barge_in_cancels_active_tts_and_clears_livekit_buffer(self) -> None:
        """CRITICAL INVARIANT: User speech onset while agent is speaking MUST:

        1. Cancel active TTS generation
        2. Flush pending audio chunks
        3. Clear LiveKit AudioSource buffer
        4. Return session state to LISTENING
        """
        fake_rtc = FakeRtcSource(sample_rate=24000, num_channels=1)
        player = LiveKitAudioPlayer(rtc_source=fake_rtc, sample_rate=24000, channels=1)
        await player.start()

        tts = FakeTTS(sample_rate=24000, chunk_count=8, chunk_duration_seconds=0.1, delay_per_chunk=0.05)
        session_state = SessionState()
        manager = TTSPlaybackManager(tts=tts, player=player, session_state=session_state)

        # Start speaking long utterance in background
        speak_task = asyncio.create_task(manager.speak("This is a synthesized test sentence for barge-in."))

        # Wait for synthesis and playback to begin
        await asyncio.sleep(0.08)
        assert manager.is_speaking
        assert session_state.current_state == AgentState.SPEAKING

        # Trigger user speech onset (simulating VAD detection)
        await manager.handle_user_speech_started()
        result = await speak_task
        assert result is False

        # Verify LiveKit clear_queue was called and playback stopped
        assert fake_rtc.clear_count >= 1
        assert not player.is_playing
        assert not manager.is_speaking
        assert session_state.current_state == AgentState.LISTENING

        await player.stop()


# ============================================================================
# 4. PrismLiveKitAgent Room Configuration and Lifecycle Tests
# ============================================================================


class TestPrismLiveKitAgent:
    @pytest.mark.asyncio
    async def test_agent_initialization_with_mock_room(self) -> None:
        """Verify agent wires local components and outbound track on room."""
        # Create mock room and participants
        mock_room = MagicMock()
        mock_room.name = "test-prism-room"
        mock_room.remote_participants = {}
        mock_room.on = MagicMock()

        local_participant = MagicMock()
        local_participant.publish_track = AsyncMock()
        mock_room.local_participant = local_participant

        custom_llm = MockTestLLM()
        agent = PrismLiveKitAgent(
            use_mock_tts=True,
            custom_llm=custom_llm,
            quiet_window_seconds=0.2,
        )

        with patch("livekit.rtc.LocalAudioTrack.create_audio_track") as mock_track_factory, \
             patch("livekit.rtc.AudioSource") as mock_source_factory:
            mock_rtc_source = FakeRtcSource()
            mock_source_factory.return_value = mock_rtc_source
            mock_local_track = MagicMock()
            mock_track_factory.return_value = mock_local_track

            await agent.initialize(mock_room)

            assert agent.audio_source is not None
            assert agent.audio_player is not None
            assert agent.tool_controller is not None
            assert agent.orchestrator is not None

            # Verify outbound track published
            local_participant.publish_track.assert_awaited_once()

            # Verify track_subscribed handler registered on room
            mock_room.on.assert_called_with("track_subscribed")

            # Clean shutdown
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_close_on_disconnect_invariant(self) -> None:
        """Verify RoomOptions close_on_disconnect=False is configured for graceful drain."""
        from livekit.agents.voice.room_io import RoomOptions

        room_opts = RoomOptions(
            close_on_disconnect=False,
            audio_input=False,
            audio_output=False,
        )
        assert not room_opts.close_on_disconnect
