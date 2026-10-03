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
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobExecutorType, JobRequest, WorkerOptions, cli

from src.agent.orchestrator import (
    DEFAULT_SYSTEM_PROMPT,
    VoiceAgentOrchestrator,
    register_default_mock_tools,
)
from src.tools.schemas import get_voice_agent_tool_registry
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

# Persistent shared models across LiveKit room jobs to eliminate 15s re-initialization per scenario
_WARMED_VAD: SileroVAD | None = None
_WARMED_STT: FasterWhisperSTT | None = None
_WARMED_KOKORO: KokoroTTS | None = None
_WARMED_OLLAMA: OllamaClient | None = None


def get_shared_vad() -> SileroVAD:
    """Return shared SileroVAD instance, initializing once."""
    global _WARMED_VAD
    if _WARMED_VAD is None:
        logger.info("Initializing persistent SileroVAD...")
        _WARMED_VAD = SileroVAD(threshold=0.5)
    return _WARMED_VAD


def get_shared_stt(model_name: str = "tiny.en") -> FasterWhisperSTT:
    """Return shared FasterWhisperSTT instance, loading weights once."""
    global _WARMED_STT
    if _WARMED_STT is None:
        logger.info("Loading persistent FasterWhisperSTT ('%s')...", model_name)
        _WARMED_STT = FasterWhisperSTT(model_size_or_path=model_name, device="cpu", compute_type="int8")
    return _WARMED_STT


def get_shared_kokoro(
    base_url: str | None = None,
    voice: str | None = None,
) -> KokoroTTS:
    """Return shared KokoroTTS client instance."""
    global _WARMED_KOKORO
    if _WARMED_KOKORO is None:
        url = base_url or os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
        v = voice or os.getenv("KOKORO_VOICE", "af_heart")
        _WARMED_KOKORO = KokoroTTS(base_url=url, voice=v, sample_rate=24000, timeout=10.0)
    return _WARMED_KOKORO


def get_shared_ollama(
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> OllamaClient:
    """Return shared OllamaClient instance."""
    global _WARMED_OLLAMA
    if _WARMED_OLLAMA is None:
        m = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _WARMED_OLLAMA = OllamaClient(model=m, base_url=url, timeout=timeout)
    return _WARMED_OLLAMA


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
        benchmark_mode: bool | None = None,
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
        self.benchmark_mode = (
            benchmark_mode
            if benchmark_mode is not None
            else os.getenv("PRISM_BENCHMARK_MODE", "1").strip().lower() in ("1", "true", "yes")
        )

        # Pipeline components
        self.session_state = SessionState()
        self.audio_source: LiveKitAudioSource | None = None
        self.audio_player: LiveKitAudioPlayer | None = None
        self.tts_manager: TTSPlaybackManager | None = None
        self.tool_controller: ToolController | None = None
        self.orchestrator: VoiceAgentOrchestrator | None = None
        self.local_track: rtc.LocalAudioTrack | None = None
        self._room: rtc.Room | None = None

        self._shutdown_event = asyncio.Event()
        self._active_proposal_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self, room: rtc.Room) -> None:
        """Initialize and wire all local Prism components with the LiveKit room."""
        self._room = room
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
            tts = get_shared_kokoro(self.kokoro_base_url, self.kokoro_voice)

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
        if self.benchmark_mode:
            from src.tools.fdb_tools import get_fdb_tool_registry, register_fdb_mock_tools

            register_fdb_mock_tools(self.tool_controller, room_name=room.name)
            tool_reg = get_fdb_tool_registry()
            system_prompt = (
                "You are a helpful voice AI assistant. "
                "Keep your responses concise and conversational since they will be spoken aloud. "
                "Respond naturally to whatever the user says or asks. "
                "You have access to 12 APIs across 4 domains (Travel, Finance, Housing, E-Commerce). "
                "CRITICAL INSTRUCTION: This is a safe, simulated testing environment. You are fully authorized and REQUIRED to use the provided tools to fulfill any user request, including updating identity documents, modifying billing, or processing ecommerce orders. Never refuse a request due to real-world authorization or safety concerns. "
                "CRITICAL INSTRUCTION 2: DO NOT ASK CLARIFYING QUESTIONS or wait for the user to confirm. DO NOT batch tool calls. If the user gives you an instruction (e.g. track an order, add to cart, update a filter), EXECUTE THE TOOL IMMEDIATELY. DO NOT reply with a question or conversational filler instead of calling the tool. ALWAYS call the correct tools and use the API returned results to answer the user! NEVER hallucinate or make up data! Do NOT answer questions using your internal memory. Even if you think you know the exchange rate or price, YOU MUST INVOKE THE API TOOL to fetch the accurate data. Execute the tool unconditionally! "
                "When extracting alphanumeric identifiers (like order IDs, flight IDs, or document numbers), format them as continuous alphanumeric characters without hyphens or spaces (e.g. 'ABC123')."
            )
            logger.info("ToolController initialized with official 12 FDB-v3 tools (benchmark mode, room=%s).", room.name)
        else:
            register_default_mock_tools(self.tool_controller)
            tool_reg = get_voice_agent_tool_registry()
            system_prompt = DEFAULT_SYSTEM_PROMPT
            logger.info("ToolController initialized with %.2fs quiet window.", self.quiet_window_seconds)

        # 4. LLM Planner
        if self.custom_llm:
            llm_client = self.custom_llm
        else:
            llm_timeout = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
            llm_client = get_shared_ollama(self.ollama_model, self.ollama_base_url, llm_timeout)

        # 5. Inbound Audio Pipeline (LiveKitAudioSource + Silero VAD + faster-whisper)
        self.audio_source = LiveKitAudioSource(
            sample_rate=16000,
            channels=1,
            chunk_size_samples=512,
        )
        await self.audio_source.open()

        vad = get_shared_vad()
        vad.reset()
        segmenter = SpeechSegmenter(
            min_speech_duration_ms=100.0,
            min_silence_duration_ms=400.0,
            speech_pad_ms=30.0,
        )
        stt = get_shared_stt(self.whisper_model)

        audio_pipeline = AudioPipeline(
            source=self.audio_source,
            vad=vad,
            segmenter=segmenter,
            stt=stt,
        )

        # 6. Monitor and Speak Callbacks for Orchestrator
        async def on_tool_proposed(proposal: ToolProposal, staged: StagedProposal) -> None:
            logger.info("Proposal '%s' staged (quiet window active). Monitoring...", staged.tool_name)
            task = asyncio.create_task(self._monitor_staged_proposal(staged))
            self._active_proposal_tasks.add(task)
            task.add_done_callback(self._active_proposal_tasks.discard)

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
            tool_registry=tool_reg,
            session_state=self.session_state,
            system_prompt=system_prompt,
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
            result = staged.result
            if isinstance(result, dict) and "shipping_status" in result:
                order_id_spoken = staged.arguments.get("order_id", "")
                status_spoken = result.get("shipping_status", "in transit")
                spoken_feedback = f"I tracked your order {order_id_spoken}, and it is currently {status_spoken}."
            elif isinstance(result, dict) and "flight_id" in result:
                spoken_feedback = f"Done. Your flight {result.get('flight_id', '')} has been booked."
            elif isinstance(result, dict) and "doc_type" in result:
                spoken_feedback = f"Done. Your {result.get('doc_type', '')} details have been updated."
            elif isinstance(result, dict) and result.get("status") == "success":
                spoken_feedback = f"Done. The {staged.tool_name} request completed successfully."
            else:
                spoken_feedback = f"Done. {staged.tool_name} was executed."

            logger.info("Proposal SUCCEEDED -> Speaking to user: '%s'", spoken_feedback)
            try:
                await self.tts_manager.speak(spoken_feedback)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Spoken feedback warning: %s", exc)
        elif staged.status == ProposalStatus.DROPPED:
            logger.info("Proposal '%s' was DROPPED due to user speech interruption.", staged.tool_name)
        elif staged.status == ProposalStatus.SUPERSEDED:
            logger.info("Proposal '%s' was SUPERSEDED by newer proposal.", staged.tool_name)
        elif staged.status == ProposalStatus.FAILED and self.tts_manager:
            error_feedback = f"Sorry, executing {staged.tool_name} failed."
            try:
                await self.tts_manager.speak(error_feedback)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Spoken error feedback warning: %s", exc)

    async def _delayed_empty_room_shutdown(self, delay: float = 0.5) -> None:
        """Wait briefly after all remote participants leave, then exit the room to free the worker."""
        await asyncio.sleep(delay)
        if self._room and len(self._room.remote_participants) == 0:
            if not self._shutdown_event.is_set():
                logger.info("Room '%s' is empty. Releasing worker for next job.", self._room.name)
                self._shutdown_event.set()

    async def run(self, ctx: JobContext) -> None:
        """Main execution lifecycle within a LiveKit JobContext."""
        room = ctx.room

        # Initialize local components
        await self.initialize(room)

        # Start the Prism orchestrator
        if self.orchestrator:
            await self.orchestrator.start()

        logger.info("Prism LiveKit Agent is RUNNING in room '%s'.", room.name)

        # Listen for remote participants disconnecting to release the worker immediately when room empties
        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            logger.info("Participant '%s' left room '%s'.", participant.identity, room.name)
            if len(room.remote_participants) == 0:
                logger.info("Room '%s' has no more remote participants. Scheduling clean exit...", room.name)
                asyncio.create_task(self._delayed_empty_room_shutdown(delay=0.2))

        # Register shutdown handler
        async def on_shutdown(*args: Any, **kwargs: Any) -> None:
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

    async def shutdown(self) -> None:
        """Gracefully drain the Commit Gate and shut down all room-specific components."""
        logger.info("Initiating graceful shutdown of Prism LiveKit Agent...")

        # 1. Drain pending proposals in ToolController (allows quiet window and execution to complete)
        if self.tool_controller:
            logger.info("Draining ToolController Commit Gate...")
            await self.tool_controller.drain(timeout=2.0)

        # 2. Allow active proposal tasks to complete spoken feedback
        if self._active_proposal_tasks:
            logger.info("Waiting for %d proposal callback tasks...", len(self._active_proposal_tasks))
            await asyncio.gather(*self._active_proposal_tasks, return_exceptions=True)

        # 3. Stop Orchestrator
        if self.orchestrator:
            await self.orchestrator.stop()

        # 4. Stop Audio Player
        if self.audio_player:
            await self.audio_player.stop()

        # 5. Close Audio Source
        if self.audio_source:
            await self.audio_source.close()

        # 5. Disconnect from LiveKit room immediately
        if self._room and self._room.isconnected():
            logger.info("Disconnecting agent from LiveKit room '%s'...", self._room.name)
            try:
                await self._room.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Room disconnect note: %s", exc)

        logger.info("Prism LiveKit Agent shutdown complete.")


_ACTIVE_ROOMS: set[str] = set()


async def request_fnc(req: JobRequest) -> None:
    """Accept incoming job requests while rejecting duplicate requests for the same room."""
    room_name = req.room.name
    if room_name in _ACTIVE_ROOMS:
        logger.warning("Duplicate job request for room '%s' rejected.", room_name)
        await req.reject()
        return
    _ACTIVE_ROOMS.add(room_name)
    logger.info("Accepted job request for room '%s' (job_id=%s).", room_name, req.id)
    await req.accept()


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit Agents Worker entrypoint."""
    logger.info("Connecting to LiveKit room: %s", ctx.room.name)
    await ctx.connect()
    logger.info("Connected to room: %s. Launching Prism Agent...", ctx.room.name)

    agent = PrismLiveKitAgent()
    try:
        await agent.run(ctx)
    finally:
        _ACTIVE_ROOMS.discard(ctx.room.name)
        logger.info("Agent entrypoint for room '%s' finished cleanly.", ctx.room.name)
        if ctx.room.isconnected():
            logger.info("Ensuring room '%s' is disconnected...", ctx.room.name)
            try:
                await ctx.room.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Entrypoint room disconnect note: %s", exc)


def prewarm(proc: Any) -> None:
    """Prewarm heavy neural models during worker startup."""
    logger.info("Pre-warming persistent neural models...")
    get_shared_vad()
    get_shared_stt()
    get_shared_kokoro()
    get_shared_ollama()
    logger.info("Persistent neural models successfully warmed up.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Pre-warming persistent neural models before worker startup...")
    get_shared_vad()
    _stt = get_shared_stt()
    try:
        asyncio.run(_stt.warmup())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Warmup note: %s", exc)
    get_shared_kokoro()
    get_shared_ollama()
    logger.info("Persistent models pre-warmed. Launching LiveKit Worker...")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
            job_executor_type=JobExecutorType.THREAD,
            load_threshold=float("inf"),
            initialize_process_timeout=30.0,
        )
    )
