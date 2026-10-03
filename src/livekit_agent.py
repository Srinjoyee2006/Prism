"""
LiveKit Agent Entrypoint for Prism (Stage 7).

Integrates the full-duplex conversational pipeline with LiveKit WebRTC:
LiveKit room
    ↓
LiveKitAudioSource (inbound participant audio)
    ↓
AudioPipeline (Silero VAD + SpeechSegmenter + faster-whisper STT)
    ↓
VoiceAgentOrchestrator
    ↓
OllamaClient (qwen2.5:1.5b) -> ToolProposal
    ↓
ToolController / Commit Gate (0.9s quiet window)
    ↓
Deterministic Mock Tools
    ↓
KokoroTTS / TTSPlaybackManager
    ↓
LiveKitAudioPlayer (outbound agent audio)
    ↓
LiveKit room

CRITICAL ARCHITECTURAL INVARIANTS:
1. The LLM only PROPOSES tools; it NEVER directly executes tools.
2. LiveKit NEVER executes tools directly.
3. Every proposed tool call passes through:
   ToolProposal -> ToolController -> 0.9s quiet window -> commit/cancel/supersede -> execution.
4. Interruption Handling:
   A. User begins speaking while tool proposal is waiting:
      -> cancel/supersede proposal, DO NOT execute.
   B. User begins speaking while agent is speaking:
      -> cancel TTS generation, flush audio queue, clear LiveKit audio source buffer,
         stop playback, return to LISTENING.
5. Graceful disconnect:
   close_on_disconnect=False in RoomOptions preserves the session so the Commit Gate
   and pending tool executions can finish cleanly upon participant disconnect.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.voice.room_io import RoomOptions

from livekit import rtc
from src.agent.orchestrator import (
    DEFAULT_SYSTEM_PROMPT,
    VoiceAgentOrchestrator,
    register_default_mock_tools,
)
from src.asr.faster_whisper_asr import FasterWhisperSTT
from src.audio.pipeline import AudioPipeline
from src.core.events import TranscriptFinalEvent
from src.core.state import SessionState
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus, StagedProposal
from src.livekit.adapters import LiveKitAudioPlayer, LiveKitAudioSource
from src.llm.base import LLMClient
from src.llm.models import LLMResponse, ToolProposal
from src.llm.ollama_client import OllamaClient
from src.tts.kokoro import KokoroTTS
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS
from src.vad.segmenter import SpeechSegmenter
from src.vad.silero_vad import SileroVAD

# Load environment configuration
load_dotenv()

logger = logging.getLogger("livekit_agent")


class PrismLiveKitAgent:
    """Encapsulates the complete Prism Voice Agent connected to a LiveKit Room."""

    def __init__(
        self,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        whisper_model: str = "tiny.en",
        kokoro_base_url: str | None = None,
        kokoro_voice: str | None = None,
        quiet_window_seconds: float = 0.9,
        use_mock_tts: bool = False,
        use_mock_llm: bool = False,
        custom_llm: LLMClient | None = None,
    ) -> None:
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.whisper_model = whisper_model or os.getenv("WHISPER_MODEL", "tiny.en")
        self.kokoro_base_url = kokoro_base_url or os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
        self.kokoro_voice = kokoro_voice or os.getenv("KOKORO_VOICE", "af_heart")
        self.quiet_window_seconds = quiet_window_seconds
        self.use_mock_tts = use_mock_tts
        self.use_mock_llm = use_mock_llm
        self.custom_llm = custom_llm

        # Pipeline components
        self.session_state = SessionState()
        self.audio_source: LiveKitAudioSource | None = None
        self.audio_player: LiveKitAudioPlayer | None = None
        self.tts_manager: TTSPlaybackManager | None = None
        self.tool_controller: ToolController | None = None
        self.orchestrator: VoiceAgentOrchestrator | None = None
        self.agent_session: AgentSession | None = None
        self.local_track: rtc.LocalAudioTrack | None = None

        self._shutdown_event = asyncio.Event()

    async def initialize(self, room: rtc.Room) -> None:
        """Initialize and wire all local Prism components with the LiveKit room."""
        logger.info("Initializing Prism LiveKit Agent for room '%s'...", room.name)

        # 1. Outbound Audio Track and LiveKitAudioPlayer
        rtc_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        self.local_track = rtc.LocalAudioTrack.create_audio_track("prism-agent-voice", rtc_source)
        self.audio_player = LiveKitAudioPlayer(
            rtc_source=rtc_source,
            sample_rate=24000,
            channels=1,
        )
        await self.audio_player.start()

        # Publish the outbound track
        await room.local_participant.publish_track(
            self.local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("Published outbound audio track 'prism-agent-voice' to room.")

        # 2. Local TTS Engine & TTSPlaybackManager
        if self.use_mock_tts:
            logger.info("Using FakeTTS (mock mode enabled).")
            tts = FakeTTS(sample_rate=24000, chunk_count=6, chunk_duration_seconds=0.25)
        else:
            kokoro = KokoroTTS(
                base_url=self.kokoro_base_url,
                voice=self.kokoro_voice,
                sample_rate=24000,
            )
            if await kokoro.is_available():
                logger.info("Connected to local Kokoro-FastAPI TTS at %s", self.kokoro_base_url)
                tts = kokoro
            else:
                logger.warning(
                    "Kokoro TTS not reachable at %s. Falling back to FakeTTS for local development.",
                    self.kokoro_base_url,
                )
                tts = FakeTTS(sample_rate=24000, chunk_count=6, chunk_duration_seconds=0.25)

        self.tts_manager = TTSPlaybackManager(
            tts=tts,
            player=self.audio_player,
            session_state=self.session_state,
            on_playback_started=lambda: logger.info("Agent began speaking to LiveKit room."),
            on_playback_completed=lambda: logger.info("Agent finished speaking."),
            on_interrupted=lambda: logger.info(">>> Agent speech INTERRUPTED by user barge-in! <<<"),
        )

        # 3. ToolController and Commit Gate
        self.tool_controller = ToolController(
            quiet_window=self.quiet_window_seconds,
        )
        register_default_mock_tools(self.tool_controller)
        logger.info("ToolController initialized with %.2fs quiet window.", self.quiet_window_seconds)

        # 4. LLM Planner
        if self.custom_llm:
            llm_client = self.custom_llm
        else:
            llm_client = OllamaClient(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
            )
            if not await llm_client.is_available():
                logger.warning(
                    "Ollama model '%s' not reachable at %s. Ensure 'ollama serve' is running.",
                    self.ollama_model,
                    self.ollama_base_url,
                )

        # 5. Inbound Audio Pipeline (LiveKitAudioSource + Silero VAD + faster-whisper)
        self.audio_source = LiveKitAudioSource(
            sample_rate=16000,
            channels=1,
            chunk_size_samples=512,
        )
        await self.audio_source.open()

        vad = SileroVAD(threshold=0.5)
        segmenter = SpeechSegmenter(
            min_speech_duration_ms=100.0,
            min_silence_duration_ms=400.0,
            speech_pad_ms=30.0,
        )
        stt = FasterWhisperSTT(model_size_or_path=self.whisper_model, device="cpu", compute_type="int8")

        audio_pipeline = AudioPipeline(
            source=self.audio_source,
            vad=vad,
            segmenter=segmenter,
            stt=stt,
        )

        # 6. Monitor and Speak Callbacks for Orchestrator
        async def on_tool_proposed(proposal: ToolProposal, staged: StagedProposal) -> None:
            logger.info("Proposal '%s' staged (quiet window active). Monitoring...", staged.tool_name)
            asyncio.create_task(self._monitor_staged_proposal(staged))

        async def on_llm_response(response: LLMResponse) -> None:
            # If model produced conversational response without proposing tools, speak it
            if response.text and not response.has_tool_proposals and self.tts_manager:
                logger.info("LLM conversational reply: '%s'", response.text)
                await self.tts_manager.speak(response.text)

        async def on_transcript(event: TranscriptFinalEvent) -> None:
            logger.info("Transcribed user utterance: '%s'", event.text)

        # 7. VoiceAgentOrchestrator
        self.orchestrator = VoiceAgentOrchestrator(
            llm_client=llm_client,
            tool_controller=self.tool_controller,
            audio_pipeline=audio_pipeline,
            session_state=self.session_state,
            tts_manager=self.tts_manager,
            on_tool_proposed=on_tool_proposed,
            on_llm_response=on_llm_response,
            on_transcript=on_transcript,
        )

        # 8. Subscribe to LiveKit Participant Audio Tracks
        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and self.audio_source:
                logger.info(
                    "Subscribed to participant audio track: identity=%s, sid=%s",
                    participant.identity,
                    track.sid,
                )
                self.audio_source.attach_track(track)

        # Attach to any participants already present in the room
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info("Attaching to pre-existing participant track: %s", pub.track.sid)
                    self.audio_source.attach_track(pub.track)

        logger.info("Prism LiveKit Agent initialized successfully.")

    async def _monitor_staged_proposal(self, staged: StagedProposal) -> None:
        """Monitor proposal through the Commit Gate and speak result upon commit."""
        while not staged.is_terminal:
            await asyncio.sleep(0.05)

        if staged.status == ProposalStatus.SUCCEEDED and self.tts_manager:
            result_str = str(staged.result)
            spoken_feedback = f"Done. {staged.tool_name} returned: {result_str}"
            logger.info("Proposal SUCCEEDED -> Speaking to user: '%s'", spoken_feedback)
            await self.tts_manager.speak(spoken_feedback)
        elif staged.status == ProposalStatus.DROPPED:
            logger.info("Proposal '%s' was DROPPED due to user speech interruption.", staged.tool_name)
        elif staged.status == ProposalStatus.SUPERSEDED:
            logger.info("Proposal '%s' was SUPERSEDED by newer proposal.", staged.tool_name)
        elif staged.status == ProposalStatus.FAILED and self.tts_manager:
            error_feedback = f"Sorry, executing {staged.tool_name} failed."
            await self.tts_manager.speak(error_feedback)

    async def run(self, ctx: JobContext) -> None:
        """Main execution lifecycle within a LiveKit JobContext."""
        room = ctx.room

        # Initialize local components
        await self.initialize(room)

        # Start LiveKit AgentSession with RoomOptions
        # CRITICAL: close_on_disconnect=False prevents abrupt session destruction
        # when a participant leaves, allowing pending Commit Gate executions to finish.
        room_options = RoomOptions(
            close_on_disconnect=False,
            audio_input=False,
            audio_output=False,
        )
        self.agent_session = AgentSession()
        agent = Agent(instructions=DEFAULT_SYSTEM_PROMPT)

        session_task = asyncio.create_task(
            self.agent_session.start(agent=agent, room=room, room_options=room_options),
            name="LiveKitAgentSessionHost",
        )

        # Start the Prism orchestrator
        if self.orchestrator:
            await self.orchestrator.start()

        logger.info("Prism LiveKit Agent is RUNNING in room '%s'.", room.name)

        # Register shutdown handler
        def on_shutdown() -> None:
            logger.info("LiveKit JobContext requested shutdown.")
            self._shutdown_event.set()

        ctx.add_shutdown_callback(on_shutdown)

        try:
            # Wait for shutdown event
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("Agent run task cancelled.")
        finally:
            await self.shutdown()
            if not session_task.done():
                session_task.cancel()
                try:
                    await session_task
                except asyncio.CancelledError:
                    pass

    async def shutdown(self) -> None:
        """Gracefully drain the Commit Gate and shut down all pipeline components."""
        logger.info("Initiating graceful shutdown of Prism LiveKit Agent...")

        # 1. Drain pending proposals in ToolController (allows quiet window and execution to complete)
        if self.tool_controller:
            logger.info("Draining ToolController Commit Gate...")
            await self.tool_controller.drain(timeout=2.0)

        # 2. Stop Orchestrator
        if self.orchestrator:
            await self.orchestrator.stop()

        # 3. Stop Audio Player
        if self.audio_player:
            await self.audio_player.stop()

        # 4. Close Audio Source
        if self.audio_source:
            await self.audio_source.close()

        # 5. Close AgentSession
        if self.agent_session:
            await self.agent_session.aclose()

        logger.info("Prism LiveKit Agent shutdown complete.")


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit Agents Worker entrypoint."""
    logger.info("Connecting to LiveKit room: %s", ctx.room.name)
    await ctx.connect()
    logger.info("Connected to room: %s. Launching Prism Agent...", ctx.room.name)

    agent = PrismLiveKitAgent()
    await agent.run(ctx)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
