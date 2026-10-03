"""
Voice Agent Orchestrator (Stage 5).

Coordinates the end-to-end full-duplex conversational flow:
AudioSource -> VAD -> SpeechSegmenter -> STT -> Transcript -> LLM (Ollama)
            -> ToolProposal -> ToolController / Commit Gate -> Mock Tool Execution.

Crucial Architectural Invariants:
1. The LLM only proposes tools (ToolProposal); it NEVER directly executes tools.
2. The ToolController is the sole component authorised to commit and execute tools.
3. User speech detected during the quiet window immediately triggers an interrupt,
   dropping or superseding uncommitted proposals.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from src.audio.pipeline import AudioPipeline
from src.core.async_utils import AsyncEventBus
from src.core.events import (
    SpeechEndedEvent,
    SpeechStartedEvent,
    TranscriptFinalEvent,
)
from src.core.state import AgentState, SessionState, TurnSpeaker
from src.gate.controller import ToolController
from src.gate.models import StagedProposal
from src.llm.base import LLMClient
from src.llm.models import ChatMessage, LLMResponse, MessageRole, ToolProposal
from src.llm.tool_schema import ToolRegistry
from src.tools.mock_tools import (
    blocking_track_order,
    mock_get_exchange_rate,
    mock_search_flights,
    mock_set_temperature,
)
from src.tools.schemas import get_voice_agent_tool_registry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, responsive smart voice assistant. "
    "When the user requests an action covered by your available tools (such as setting temperature, "
    "searching flights, checking exchange rates, or tracking orders), propose the appropriate tool call. "
    "Be direct and concise."
)


def register_default_mock_tools(controller: ToolController) -> None:
    """Register all standard mock tools on a ToolController instance."""
    controller.register_handler("set_temperature", mock_set_temperature)
    controller.register_handler("search_flights", mock_search_flights)
    controller.register_handler("get_exchange_rate", mock_get_exchange_rate)
    controller.register_blocking_handler("track_order", blocking_track_order)


class VoiceAgentOrchestrator:
    """Central full-duplex orchestrator connecting Audio, VAD, ASR, LLM, and Commit Gate."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_controller: ToolController,
        audio_pipeline: AudioPipeline | None = None,
        tool_registry: ToolRegistry | None = None,
        session_state: SessionState | None = None,
        event_bus: AsyncEventBus | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_transcript: Callable[[TranscriptFinalEvent], Awaitable[None]] | None = None,
        on_llm_response: Callable[[LLMResponse], Awaitable[None]] | None = None,
        on_tool_proposed: (
            Callable[[ToolProposal, StagedProposal], Awaitable[None]] | None
        ) = None,
        on_speech_start: Callable[[], Awaitable[None]] | None = None,
        on_speech_end: Callable[[], Awaitable[None]] | None = None,
        tts_manager: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_controller = tool_controller
        self.audio_pipeline = audio_pipeline
        self.tool_registry = tool_registry or get_voice_agent_tool_registry()
        self.session_state = session_state or SessionState()
        self.event_bus = event_bus
        self.system_prompt = system_prompt
        self.tts_manager = tts_manager

        # Observability callbacks
        self.on_transcript = on_transcript
        self.on_llm_response = on_llm_response
        self.on_tool_proposed = on_tool_proposed
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._running: bool = False
        self._turn_lock = asyncio.Lock()

        # Wire pipeline callbacks if pipeline provided
        if self.audio_pipeline is not None:
            self._wire_pipeline(self.audio_pipeline)

    @property
    def state(self) -> SessionState:
        """Access the agent's current session state."""
        return self.session_state

    @property
    def is_running(self) -> bool:
        """Return True if the orchestrator is actively running."""
        return self._running

    def _wire_pipeline(self, pipeline: AudioPipeline) -> None:
        """Attach orchestrator event handlers to the audio pipeline."""
        pipeline.on_speech_start = self._handle_speech_start
        pipeline.on_speech_end = self._handle_speech_end
        pipeline.on_transcript = self._handle_transcript

    async def _handle_speech_start(self) -> None:
        """Triggered immediately when VAD detects the onset of user speech.

        1. ToolController drops/supersedes any staged proposal in the quiet window.
        2. TTSPlaybackManager cancels active synthesis, flushes playback queue, and stops speaker.
        """
        logger.info("Orchestrator: User speech onset detected -> notifying ToolController and TTSManager")
        self.tool_controller.notify_user_speech_started()

        if self.tts_manager is not None:
            await self.tts_manager.handle_user_speech_started()

        if self.session_state.current_state == AgentState.THINKING:
            await self.session_state.transition_to(
                AgentState.INTERRUPTED,
                reason="User speech interrupted thinking",
            )

        if self.event_bus:
            await self.event_bus.publish(SpeechStartedEvent())

        if self.on_speech_start:
            await self.on_speech_start()

    async def _handle_speech_end(self) -> None:
        """Triggered when VAD detects user speech cessation/silence."""
        logger.info("Orchestrator: User speech ended")
        if self.session_state.current_state == AgentState.INTERRUPTED:
            await self.session_state.transition_to(
                AgentState.LISTENING,
                reason="User speech ended after interruption",
            )

        if self.event_bus:
            await self.event_bus.publish(SpeechEndedEvent())

        if self.on_speech_end:
            await self.on_speech_end()

    async def _handle_transcript(self, event: TranscriptFinalEvent) -> None:
        """Triggered when streaming ASR produces a final transcript."""
        if self.on_transcript:
            await self.on_transcript(event)

        await self.process_transcript(event.text)

    async def process_transcript(self, text: str) -> list[StagedProposal]:
        """Core turn processing: Transcript -> LLM -> ToolProposal -> ToolController.

        Can be called directly for offline deterministic tests or via streaming audio.

        Args:
            text: The user's finalized speech transcript.

        Returns:
            A list of StagedProposal objects submitted to the ToolController.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        async with self._turn_lock:
            # Clear previous interruption flag so the new proposal can commit
            self.tool_controller.clear_interrupt()

            # Transition state machine
            await self.session_state.start_turn(speaker=TurnSpeaker.USER)
            current = self.session_state.current_state
            if current in {AgentState.IDLE, AgentState.LISTENING, AgentState.INTERRUPTED}:
                await self.session_state.transition_to(
                    AgentState.THINKING,
                    reason="Processing user transcript with LLM",
                )

            # Build conversational prompt
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(role=MessageRole.USER, content=clean_text),
            ]

            logger.info("Orchestrator: Querying LLM for transcript: '%s'", clean_text)

            # 1. Query LLM (LLM ONLY PROPOSES, NEVER EXECUTES)
            tools_list = self.tool_registry.list_schemas()
            llm_response = await self.llm_client.generate(
                messages=messages,
                tools=tools_list,
            )

            if self.on_llm_response:
                await self.on_llm_response(llm_response)

            # 2. Stage valid tool proposals through the ToolController Commit Gate
            staged_proposals: list[StagedProposal] = []
            for proposal in llm_response.valid_tool_proposals:
                logger.info(
                    "Orchestrator: Submitting proposal '%s' (%s) to ToolController",
                    proposal.tool_name,
                    proposal.arguments,
                )
                staged = await self.tool_controller.submit_proposal(proposal)
                staged_proposals.append(staged)

                if self.on_tool_proposed:
                    await self.on_tool_proposed(proposal, staged)

            # Return session to listening/idle state
            if self.session_state.current_state == AgentState.THINKING:
                await self.session_state.transition_to(
                    AgentState.IDLE,
                    reason="LLM turn processing completed",
                )

            return staged_proposals

    async def start(self) -> None:
        """Start the orchestrator and the underlying audio pipeline."""
        self._running = True
        if self.session_state.current_state == AgentState.IDLE:
            await self.session_state.transition_to(
                AgentState.LISTENING,
                reason="Orchestrator started",
            )

        if self.audio_pipeline is not None:
            self.audio_pipeline.start()

        logger.info("VoiceAgentOrchestrator started in LISTENING state.")

    async def stop(self) -> None:
        """Stop the orchestrator, cancel active tasks, and drain the controller."""
        self._running = False
        if self.tts_manager is not None:
            self.tts_manager.cancel()

        if self.audio_pipeline is not None:
            await self.audio_pipeline.stop()

        await self.tool_controller.drain(timeout=2.0)

        if self.session_state.current_state != AgentState.IDLE:
            await self.session_state.transition_to(
                AgentState.IDLE,
                reason="Orchestrator stopped",
                force=True,
            )

        logger.info("VoiceAgentOrchestrator stopped.")
