"""
Automated test suite for the Stage 5 Voice Agent Orchestrator.

Tests requirements:
A. Normal flow: transcript -> LLM proposal -> commit gate -> tool execution
B. User correction: proposal waits -> speech-start interrupt -> proposal superseded/dropped -> new proposal executes
C. Duplicate proposal: identical tool + normalized args -> only one execution (idempotency)
D. No direct execution: verify LLM layer only produces ToolProposal data objects and never executes tools
E. End-to-end integration: Synthetic audio -> VAD -> Segmenter -> STT -> LLM -> Controller -> Mock tool
"""

import asyncio

import pytest

from src.agent.orchestrator import VoiceAgentOrchestrator, register_default_mock_tools
from src.asr.mock_asr import FakeSTT
from src.audio.pipeline import AudioPipeline
from src.audio.synthetic import SyntheticAudioSource
from src.core.state import AgentState, SessionState
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus, StagedProposal
from src.llm.base import LLMClient
from src.llm.models import ChatMessage, LLMResponse, ToolProposal
from src.llm.tool_schema import ToolRegistry, ToolSchema
from src.tools.mock_tools import mock_set_temperature
from src.tools.schemas import create_temperature_schema

# ---------------------------------------------------------------------------
# Test Doubles / Fakes
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Deterministic LLM client for testing without Ollama or network."""

    def __init__(self, canned_proposals: list[list[ToolProposal]] | None = None) -> None:
        self.canned_proposals = canned_proposals or []
        self._call_count = 0
        self.recorded_messages: list[list[ChatMessage]] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> LLMResponse:
        self.recorded_messages.append(messages)
        if self._call_count < len(self.canned_proposals):
            proposals = self.canned_proposals[self._call_count]
        else:
            proposals = []
        self._call_count += 1

        return LLMResponse(
            text="I will execute that request for you.",
            tool_proposals=proposals,
            finish_reason="tool_calls" if proposals else "stop",
        )

    async def is_available(self) -> bool:
        return True


def create_test_orchestrator(
    llm_client: LLMClient,
    quiet_window: float = 0.05,
) -> tuple[VoiceAgentOrchestrator, ToolController]:
    """Helper to assemble a test orchestrator with standard mock tools."""
    controller = ToolController(quiet_window=quiet_window)
    register_default_mock_tools(controller)

    registry = ToolRegistry()
    registry.register(create_temperature_schema())

    orchestrator = VoiceAgentOrchestrator(
        llm_client=llm_client,
        tool_controller=controller,
        tool_registry=registry,
        session_state=SessionState(),
    )
    return orchestrator, controller


# ---------------------------------------------------------------------------
# Test A: Normal Flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_A_normal_flow_transcript_to_tool_execution():
    """Requirement A: Transcript -> LLM proposal -> commit gate -> tool execution."""
    proposal = ToolProposal(
        tool_name="set_temperature",
        arguments={"zone": "living_room", "temperature_c": 21.0},
    )
    fake_llm = FakeLLMClient(canned_proposals=[[proposal]])
    orchestrator, controller = create_test_orchestrator(fake_llm, quiet_window=0.05)

    # 1. Process transcript
    staged_list = await orchestrator.process_transcript("Set the living room temperature to 21")
    assert len(staged_list) == 1
    staged = staged_list[0]

    # Verify initial staged state
    assert staged.tool_name == "set_temperature"
    assert staged.arguments == {"zone": "living_room", "temperature_c": 21.0}
    assert staged.status in {ProposalStatus.PROPOSED, ProposalStatus.WAITING}

    # 2. Await commit gate execution (0.05s window)
    result = await controller.wait_for_proposal(staged.proposal_id, timeout=2.0)
    assert result is not None
    assert result.status == ProposalStatus.SUCCEEDED
    assert result.result["status"] == "ok"
    assert result.result["zone"] == "living_room"
    assert result.result["temperature_c"] == 21.0
    assert result.committed_at is not None
    assert result.completed_at is not None


# ---------------------------------------------------------------------------
# Test B: User Correction / Speech Interruption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_B_user_correction_interrupts_first_proposal():
    """Requirement B: First proposal waiting -> user speech detected -> first dropped -> second succeeds."""
    p1 = ToolProposal(tool_name="set_temperature", arguments={"zone": "living_room", "temperature_c": 20.0})
    p2 = ToolProposal(tool_name="set_temperature", arguments={"zone": "living_room", "temperature_c": 24.0})

    fake_llm = FakeLLMClient(canned_proposals=[[p1], [p2]])
    # Use a longer quiet window so we can interrupt during it
    orchestrator, controller = create_test_orchestrator(fake_llm, quiet_window=0.5)

    # 1. User says: "Set temperature to 20"
    staged_list1 = await orchestrator.process_transcript("Set temperature to 20")
    staged1 = staged_list1[0]
    await asyncio.sleep(0.05)
    assert staged1.status == ProposalStatus.WAITING

    # 2. User interrupts by speaking: "Wait, make it 24 instead"
    await orchestrator._handle_speech_start()

    # Proposal 1 should be dropped immediately before committing
    r1 = await controller.wait_for_proposal(staged1.proposal_id, timeout=2.0)
    assert r1 is not None
    assert r1.status == ProposalStatus.DROPPED
    assert r1.result is None  # Tool was NEVER executed!

    # 3. User finishes speaking correction
    await orchestrator._handle_speech_end()

    # 4. Correction transcript is processed
    staged_list2 = await orchestrator.process_transcript("Wait, make it 24 instead")
    staged2 = staged_list2[0]

    r2 = await controller.wait_for_proposal(staged2.proposal_id, timeout=2.0)
    assert r2 is not None
    assert r2.status == ProposalStatus.SUCCEEDED
    assert r2.result["temperature_c"] == 24.0


# ---------------------------------------------------------------------------
# Test B2: Direct Supersession (Same tool proposed during quiet window)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_B2_supersession_replaces_waiting_proposal_for_same_tool():
    """If a new proposal for the same tool arrives while waiting, the predecessor is superseded."""
    p1 = ToolProposal(tool_name="set_temperature", arguments={"zone": "bedroom", "temperature_c": 19.0})
    p2 = ToolProposal(tool_name="set_temperature", arguments={"zone": "bedroom", "temperature_c": 23.0})

    fake_llm = FakeLLMClient(canned_proposals=[[p1], [p2]])
    orchestrator, controller = create_test_orchestrator(fake_llm, quiet_window=0.5)

    staged_list1 = await orchestrator.process_transcript("Bedroom 19")
    staged1 = staged_list1[0]
    await asyncio.sleep(0.04)
    assert staged1.status == ProposalStatus.WAITING

    # Second proposal submitted directly
    staged_list2 = await orchestrator.process_transcript("Bedroom 23")
    staged2 = staged_list2[0]

    # Predecessor superseded
    assert staged1.status == ProposalStatus.SUPERSEDED
    assert staged1.result is None

    # New proposal commits and succeeds
    r2 = await controller.wait_for_proposal(staged2.proposal_id, timeout=2.0)
    assert r2 is not None
    assert r2.status == ProposalStatus.SUCCEEDED
    assert r2.result["temperature_c"] == 23.0


# ---------------------------------------------------------------------------
# Test C: Duplicate Proposal (Idempotency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_C_duplicate_proposal_idempotency():
    """Requirement C: Identical tool + normalized args returns the same in-flight proposal."""
    p1 = ToolProposal(tool_name="set_temperature", arguments={"zone": "office", "temperature_c": 22.0})
    p2 = ToolProposal(tool_name="set_temperature", arguments={"zone": "office", "temperature_c": 22.0})

    fake_llm = FakeLLMClient(canned_proposals=[[p1], [p2]])
    orchestrator, controller = create_test_orchestrator(fake_llm, quiet_window=0.5)

    s1 = (await orchestrator.process_transcript("Set office to 22"))[0]
    s2 = (await orchestrator.process_transcript("Set office to 22"))[0]

    # Exact same proposal object returned (in-flight deduplication)
    assert s1.proposal_id == s2.proposal_id

    # Wait for completion
    res = await controller.wait_for_proposal(s1.proposal_id, timeout=2.0)
    assert res is not None
    assert res.status == ProposalStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Test D: No Direct Execution by LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_D_no_direct_execution_by_llm():
    """Requirement D: LLM layer only produces ToolProposal and cannot execute tools itself."""
    proposal = ToolProposal(tool_name="set_temperature", arguments={"zone": "kitchen", "temperature_c": 25.0})
    fake_llm = FakeLLMClient(canned_proposals=[[proposal]])

    # Generate from LLM directly
    messages = [ChatMessage(role="user", content="Set kitchen to 25")]
    response = await fake_llm.generate(messages)

    # 1. LLM only returned a response with a proposal
    assert len(response.tool_proposals) == 1
    p = response.tool_proposals[0]
    assert isinstance(p, ToolProposal)

    # 2. Confirm the proposal itself is purely a descriptor
    assert not hasattr(p, "execute")
    assert not hasattr(p, "run")
    assert not hasattr(fake_llm, "execute")
    assert not hasattr(fake_llm, "handlers")

    # 3. Execution can ONLY happen when submitted to ToolController
    controller = ToolController(quiet_window=0.05)
    controller.register_handler("set_temperature", mock_set_temperature)

    staged = await controller.submit_proposal(p)
    assert staged.status in {ProposalStatus.PROPOSED, ProposalStatus.WAITING}

    result = await controller.wait_for_proposal(staged.proposal_id, timeout=2.0)
    assert result is not None
    assert result.status == ProposalStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Test E: End-to-End Audio Pipeline + Orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_E_end_to_end_audio_pipeline_to_orchestrator():
    """Requirement E: Synthetic audio -> VAD -> Segmenter -> STT -> Orchestrator -> Tool Execution."""
    from src.vad.mock_vad import FakeVAD
    from src.vad.segmenter import SpeechSegmenter

    # 1. Setup synthetic audio: 2 frames silence, 6 frames speech, 6 frames silence (32ms frames)
    source = SyntheticAudioSource(chunk_size_samples=512)
    source.add_silence(duration_ms=64.0)
    source.add_sine_tone(duration_ms=192.0, frequency_hz=300.0)
    source.add_silence(duration_ms=192.0)

    # 2. VAD & Segmenter
    vad = FakeVAD(scripted_decisions=[False, False] + [True] * 6 + [False] * 6)
    segmenter = SpeechSegmenter(min_speech_duration_ms=100.0, min_silence_duration_ms=120.0)

    # 3. STT produces transcript
    stt = FakeSTT(canned_responses=["Set temperature to 22 in living room"])

    # 4. AudioPipeline
    pipeline = AudioPipeline(source=source, vad=vad, segmenter=segmenter, stt=stt)

    # 5. LLM proposes tool call
    proposal = ToolProposal(tool_name="set_temperature", arguments={"zone": "living_room", "temperature_c": 22.0})
    fake_llm = FakeLLMClient(canned_proposals=[[proposal]])

    controller = ToolController(quiet_window=0.05)
    register_default_mock_tools(controller)

    executed_proposals: list[ToolProposal] = []

    async def _on_tool(p: ToolProposal, s: StagedProposal):
        executed_proposals.append(p)

    orchestrator = VoiceAgentOrchestrator(
        llm_client=fake_llm,
        tool_controller=controller,
        audio_pipeline=pipeline,
        on_tool_proposed=_on_tool,
    )

    # Run the pipeline
    await orchestrator.start()
    # Wait for audio to process
    await asyncio.sleep(0.3)
    await orchestrator.stop()

    assert len(executed_proposals) == 1
    assert executed_proposals[0].tool_name == "set_temperature"

    # Confirm execution succeeded in controller
    all_p = controller.all_proposals()
    assert len(all_p) >= 1
    assert any(p.status == ProposalStatus.SUCCEEDED for p in all_p)


# ---------------------------------------------------------------------------
# State Transitions & Observability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_state_transitions():
    """Verify session state transitions from LISTENING -> THINKING -> IDLE."""
    fake_llm = FakeLLMClient(canned_proposals=[[]])
    orchestrator, _ = create_test_orchestrator(fake_llm)

    assert orchestrator.state.current_state == AgentState.IDLE

    await orchestrator.start()
    assert orchestrator.state.current_state == AgentState.LISTENING

    await orchestrator.process_transcript("What is the weather?")
    # Returns to IDLE or LISTENING upon completion
    assert orchestrator.state.current_state in {AgentState.IDLE, AgentState.LISTENING}

    await orchestrator.stop()
    assert orchestrator.state.current_state == AgentState.IDLE


@pytest.mark.asyncio
async def test_empty_transcript_ignored():
    """Empty or whitespace transcripts must not trigger LLM calls."""
    fake_llm = FakeLLMClient()
    orchestrator, _ = create_test_orchestrator(fake_llm)

    proposals = await orchestrator.process_transcript("   ")
    assert proposals == []
    assert fake_llm._call_count == 0
