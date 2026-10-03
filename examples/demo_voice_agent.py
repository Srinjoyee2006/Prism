"""
Interactive Local Full-Duplex Voice Agent Demo (Stage 5).

Connects:
Microphone -> Silero VAD -> Speech Segmenter -> faster-whisper -> Transcript
           -> Ollama (Qwen 2.5 1.5B) -> ToolProposal -> ToolController / Commit Gate
           -> Mock Tool Execution.

Usage:
    python examples/demo_voice_agent.py

Press Ctrl+C to stop.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent.orchestrator import VoiceAgentOrchestrator, register_default_mock_tools
from src.asr.faster_whisper_asr import FasterWhisperSTT
from src.audio.microphone import MicrophoneAudioSource
from src.audio.pipeline import AudioPipeline
from src.core.events import TranscriptFinalEvent
from src.gate.controller import ToolController
from src.gate.models import ProposalStatus, StagedProposal
from src.llm.models import ToolProposal
from src.llm.ollama_client import OllamaClient
from src.vad.segmenter import SpeechSegmenter
from src.vad.silero_vad import SileroVAD

logging.basicConfig(level=logging.WARNING)


async def monitor_proposal_lifecycle(controller: ToolController, staged: StagedProposal) -> None:
    """Asynchronously monitor and log state changes for a staged proposal."""
    last_status = staged.status
    print(f"  [Commit Gate] Proposal '{staged.tool_name}' staged: status = {last_status.value.upper()} (0.9s quiet window active)")

    while not staged.is_terminal:
        await asyncio.sleep(0.05)
        if staged.status != last_status:
            last_status = staged.status
            print(f"  [Commit Gate] Proposal '{staged.tool_name}' transitioned to: {last_status.value.upper()}")

    # Terminal state reached
    if staged.status == ProposalStatus.SUCCEEDED:
        print(f"  [Execution] SUCCESS: {staged.tool_name} returned -> {staged.result}\n")
    elif staged.status == ProposalStatus.DROPPED:
        print(f"  [Commit Gate] DROPPED: {staged.tool_name} was cancelled by user speech interruption before commit.\n")
    elif staged.status == ProposalStatus.SUPERSEDED:
        print(f"  [Commit Gate] SUPERSEDED: {staged.tool_name} was replaced by a more recent proposal.\n")
    elif staged.status == ProposalStatus.FAILED:
        print(f"  [Execution] FAILED: {staged.tool_name} raised error: {staged.error}\n")


async def main() -> None:
    if not MicrophoneAudioSource.is_available():
        print("ERROR: No microphone input device found. Please connect a microphone.", flush=True)
        sys.exit(1)

    ollama_client = OllamaClient(model="qwen2.5:1.5b")
    if not await ollama_client.is_available():
        print("WARNING: Ollama daemon does not appear to be running or qwen2.5:1.5b is not loaded.", flush=True)
        print("Make sure 'ollama serve' is running and 'ollama pull qwen2.5:1.5b' has been executed.", flush=True)
        print("Continuing with initialization...\n", flush=True)

    print("=" * 70, flush=True)
    print("Prism Stage 5: Full-Duplex Voice Agent + Commit Gate Demo", flush=True)
    print("=" * 70, flush=True)
    print("Pipeline Components:")
    print("  1. Audio Input:  Microphone (16kHz Mono 16-bit PCM)")
    print("  2. VAD:          Local Silero VAD (ONNX Runtime, 32ms frames)")
    print("  3. Segmentation: SpeechSegmenter (100ms min speech, 400ms pause)")
    print("  4. ASR:          faster-whisper (tiny.en INT8 CPU)")
    print("  5. LLM Planner:  Ollama (qwen2.5:1.5b)")
    print("  6. Gate:         ToolController (0.9s quiet window, barge-in drop)")
    print("  7. Tools:        Deterministic Mock Tools (temperature, flights, FX, order)")
    print("=" * 70, flush=True)
    print("Try saying:")
    print("  - 'Set the living room temperature to 21 degrees'")
    print("  - 'What is the exchange rate from USD to INR?'")
    print("  - 'Track order ORD-12345'")
    print("  - Or speak to interrupt while a proposal is waiting in the quiet window!")
    print("\nListening... (Press Ctrl+C to exit)\n", flush=True)

    # 1. Audio Source & VAD Pipeline
    source = MicrophoneAudioSource(sample_rate=16000, chunk_size_samples=512)
    vad = SileroVAD(threshold=0.5)
    segmenter = SpeechSegmenter(
        min_speech_duration_ms=100.0,
        min_silence_duration_ms=400.0,
        speech_pad_ms=30.0,
    )
    stt = FasterWhisperSTT(model_size_or_path="tiny.en", device="cpu", compute_type="int8")
    audio_pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
    )

    # 2. Tool Controller with standard mock tools
    controller = ToolController(quiet_window=0.9)
    register_default_mock_tools(controller)

    # 3. Observability Callbacks
    async def on_speech_start() -> None:
        print("\n>>> [User Speaking...]", end="", flush=True)

    async def on_speech_end() -> None:
        print(" [Speech Ended. Transcribing...]", flush=True)

    async def on_transcript(event: TranscriptFinalEvent) -> None:
        print(f"\n[Transcript] \"{event.text}\" (confidence={event.confidence:.2f})", flush=True)

    async def on_tool_proposed(proposal: ToolProposal, staged: StagedProposal) -> None:
        print(f"  [LLM Tool Proposal] Tool: '{proposal.tool_name}' | Arguments: {proposal.arguments}", flush=True)
        # Spawn monitoring task to report lifecycle updates (WAITING -> COMMITTED -> SUCCEEDED/DROPPED)
        asyncio.create_task(monitor_proposal_lifecycle(controller, staged))

    # 4. Orchestrator
    orchestrator = VoiceAgentOrchestrator(
        llm_client=ollama_client,
        tool_controller=controller,
        audio_pipeline=audio_pipeline,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        on_transcript=on_transcript,
        on_tool_proposed=on_tool_proposed,
    )

    # 5. Run until Ctrl+C
    await orchestrator.start()
    try:
        # Keep main coroutine alive while pipeline processes frames
        while orchestrator.is_running:
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down voice agent...", flush=True)
    finally:
        await orchestrator.stop()
        print("Voice agent cleanly stopped.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
