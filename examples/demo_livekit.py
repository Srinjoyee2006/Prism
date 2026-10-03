"""
LiveKit Full-Duplex Voice Agent Development Demo (Stage 7).

Demonstrates the Prism voice agent connected to a LiveKit WebRTC room.

Architecture:
    LiveKit Room (Participant)
        ↓ WebRTC Audio
    LiveKitAudioSource
        ↓ 16kHz 16-bit Mono Frames
    AudioPipeline (Silero VAD + SpeechSegmenter + faster-whisper)
        ↓ Transcript
    VoiceAgentOrchestrator
        ↓
    Ollama LLM (qwen2.5:1.5b) -> ToolProposal
        ↓
    ToolController / Commit Gate (0.9s quiet window)
        ↓
    Deterministic Mock Tools
        ↓ Tool Execution Result
    Kokoro-FastAPI TTS / FakeTTS
        ↓ 24kHz AudioChunks
    LiveKitAudioPlayer -> LiveKit LocalAudioTrack
        ↓ WebRTC Audio
    LiveKit Room (Speaker)

Usage:
    # 1. Run full LiveKit worker with local components:
    python examples/demo_livekit.py dev

    # 2. Check environment and component connectivity:
    python examples/demo_livekit.py --check-env

    # 3. Run simulated offline WebRTC pipeline test (no LiveKit server needed):
    python examples/demo_livekit.py --offline-test
"""

import argparse
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
from livekit.agents import WorkerOptions, cli

from src.gate.controller import ToolController
from src.gate.models import ProposalStatus
from src.livekit.adapters import LiveKitAudioPlayer, LiveKitAudioSource
from src.livekit_agent import entrypoint
from src.llm.base import LLMClient
from src.llm.models import ChatMessage, LLMResponse, ToolProposal
from src.llm.ollama_client import OllamaClient
from src.tools.mock_tools import mock_set_temperature
from src.tts.kokoro import KokoroTTS
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_livekit")


async def check_environment() -> None:
    """Validate connectivity of all local and LiveKit dependencies."""
    print("=" * 70)
    print("  PRISM STAGE 7: LIVEKIT & LOCAL ENVIRONMENT PRE-FLIGHT CHECK")
    print("=" * 70)

    # 1. LiveKit Credentials
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    print("\n[1/4] LiveKit Server Configuration:")
    print(f"  LIVEKIT_URL:        {url or '(not set)'}")
    print(f"  LIVEKIT_API_KEY:    {'***' + api_key[-4:] if api_key and len(api_key) > 4 else (api_key or '(not set)')}")
    print(f"  LIVEKIT_API_SECRET: {'***' if api_secret else '(not set)'}")

    if not url or not api_key or not api_secret:
        print("\n  [INFO] LiveKit credentials not fully configured in .env.")
        print("  To run a local development LiveKit server using Docker:")
        print("    docker run --rm -it -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\")
        print("      livekit/livekit-server --dev")
        print("  Then configure in your .env:")
        print("    LIVEKIT_URL=ws://127.0.0.1:7880")
        print("    LIVEKIT_API_KEY=devkey")
        print("    LIVEKIT_API_SECRET=secret")
    else:
        print("  [OK] LiveKit configuration found.")

    # 2. Ollama Local LLM
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    print(f"\n[2/4] Ollama LLM ({ollama_model} @ {ollama_url}):")
    ollama = OllamaClient(base_url=ollama_url, model=ollama_model)
    if await ollama.is_available():
        print(f"  [OK] Ollama is active and model '{ollama_model}' is ready.")
    else:
        print(f"  [WARNING] Ollama is not responding at {ollama_url}.")
        print("  Run 'ollama serve' and 'ollama pull qwen2.5:1.5b'.")

    # 3. Kokoro-FastAPI TTS
    kokoro_url = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
    kokoro_voice = os.getenv("KOKORO_VOICE", "af_heart")
    print(f"\n[3/4] Kokoro-FastAPI TTS ({kokoro_voice} @ {kokoro_url}):")
    kokoro = KokoroTTS(base_url=kokoro_url, voice=kokoro_voice)
    if await kokoro.is_available():
        print(f"  [OK] Kokoro-FastAPI is active and responding at {kokoro_url}.")
    else:
        print(f"  [WARNING] Kokoro-FastAPI is not responding at {kokoro_url}.")
        print("  Run Docker container:")
        print("    docker run -d -p 8880:8880 --name kokoro-fastapi ghcr.io/remsky/kokoro-fastapi-cpu:latest")

    # 4. Audio Adapters
    print("\n[4/4] LiveKit Audio Adapters:")
    print("  LiveKitAudioSource: Ready (16kHz 16-bit mono -> Silero VAD).")
    print("  LiveKitAudioPlayer: Ready (24kHz 16-bit mono -> WebRTC playout).")
    print("=" * 70 + "\n")


class MockRtcSource:
    """Mock for livekit.rtc.AudioSource to simulate LiveKit WebRTC playout."""

    def __init__(self, sample_rate: int = 24000, num_channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.captured_frames: list[Any] = []
        self.queued_duration: float = 0.0
        self.clear_count: int = 0

    async def capture_frame(self, frame: Any) -> None:
        self.captured_frames.append(frame)
        self.queued_duration += getattr(frame, "duration", 0.02)

    def clear_queue(self) -> None:
        self.clear_count += 1
        self.queued_duration = 0.0

    async def wait_for_playout(self) -> None:
        await asyncio.sleep(0.01)
        self.queued_duration = 0.0

    async def aclose(self) -> None:
        pass


class OfflineFakeLLM(LLMClient):
    """Deterministic LLM for offline demonstration."""

    async def generate(self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        user_msg = messages[-1].content.lower()
        if "temperature" in user_msg or "degrees" in user_msg:
            return LLMResponse(
                text="Setting living room temperature.",
                tool_proposals=[
                    ToolProposal(
                        tool_name="set_temperature",
                        arguments={"location": "living room", "temperature": 21.0},
                    )
                ],
            )
        return LLMResponse(text="I am your offline Prism assistant.")

    async def is_available(self) -> bool:
        return True


async def run_offline_simulation() -> None:
    """Run an end-to-end offline simulation of LiveKit adapters and the Commit Gate."""
    print("=" * 70)
    print("  PRISM STAGE 7: OFFLINE SIMULATION OF LIVEKIT ADAPTERS & COMMIT GATE")
    print("=" * 70)

    # 1. Initialize Adapters
    mock_rtc = MockRtcSource(sample_rate=24000, num_channels=1)
    player = LiveKitAudioPlayer(rtc_source=mock_rtc, sample_rate=24000, channels=1)
    await player.start()

    source = LiveKitAudioSource(sample_rate=16000, channels=1, chunk_size_samples=512)
    await source.open()

    controller = ToolController(quiet_window=0.5)
    controller.register_handler("set_temperature", mock_set_temperature)

    tts = FakeTTS(sample_rate=24000, chunk_count=6, chunk_duration_seconds=0.2, delay_per_chunk=0.08)
    tts_manager = TTSPlaybackManager(tts=tts, player=player)

    # --- Scenario 1: Clean Commit Gate Execution ---
    print("\n[Scenario 1] Proposal commits cleanly through Commit Gate (no interruption):")
    proposal = ToolProposal(
        tool_name="set_temperature",
        arguments={"location": "bedroom", "temperature": 22.0},
    )
    print(f"  Proposing tool: {proposal.tool_name} with args {proposal.arguments}")
    staged = await controller.submit_proposal(proposal)
    print(f"  Staged: status = {staged.status.value.upper()} (quiet window 0.5s active)...")

    await asyncio.sleep(0.6)  # Wait past 0.5s quiet window
    print(f"  After quiet window: status = {staged.status.value.upper()}")
    print(f"  Mock Tool Result: {staged.result}")
    assert staged.status == ProposalStatus.SUCCEEDED

    # Synthesize speech into LiveKitAudioPlayer
    await tts_manager.speak(f"Temperature updated: {staged.result.get('message', staged.result)}")
    await player.wait_until_done(timeout=1.0)
    print(f"  LiveKitAudioPlayer captured {len(mock_rtc.captured_frames)} audio frames into WebRTC track.")

    # --- Scenario 2: Barge-in Interruption during Quiet Window ---
    print("\n[Scenario 2] User interrupts while proposal is waiting in quiet window:")
    mock_rtc.captured_frames.clear()
    proposal2 = ToolProposal(
        tool_name="set_temperature",
        arguments={"location": "kitchen", "temperature": 18.0},
    )
    staged2 = await controller.submit_proposal(proposal2)
    print(f"  Staged: status = {staged2.status.value.upper()} (quiet window active)...")

    # Simulate user speech onset via VAD event after 200ms
    await asyncio.sleep(0.2)
    print("  [VAD Event] User started speaking! Triggering notify_user_speech_started()...")
    controller.notify_user_speech_started()

    await asyncio.sleep(0.4)
    print(f"  Result: status = {staged2.status.value.upper()}")
    assert staged2.status == ProposalStatus.DROPPED
    print("  [INVARIANT VERIFIED] Proposal was DROPPED! Mock tool was NEVER executed.")

    # --- Scenario 3: Barge-in Interruption during Agent Speech ---
    print("\n[Scenario 3] User interrupts while agent is playing speech to LiveKit:")
    speak_task = asyncio.create_task(tts_manager.speak("This is a long synthesized response to be interrupted."))
    await asyncio.sleep(0.1)

    print("  Agent is speaking to LiveKit WebRTC track...")
    print(f"  Player is_playing: {player.is_playing}")
    print("  [VAD Event] User speaks mid-utterance! Triggering handle_user_speech_started()...")
    await tts_manager.handle_user_speech_started()

    print(f"  LiveKit buffer clear_queue was called {mock_rtc.clear_count} time(s).")
    print(f"  Player is_playing after cancel: {player.is_playing}")
    assert not player.is_playing
    print("  [INVARIANT VERIFIED] Outbound speech cancelled, queue flushed, buffer cleared immediately.")

    await speak_task
    await player.stop()
    await source.close()
    await controller.drain()

    print("\n" + "=" * 70)
    print("  ALL OFFLINE LIVEKIT ADAPTER & COMMIT GATE SCENARIOS PASSED!")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prism LiveKit Agent Development Demo")
    parser.add_argument("--check-env", action="store_true", help="Check local environment and LiveKit configuration")
    parser.add_argument("--offline-test", action="store_true", help="Run simulated offline pipeline test without LiveKit server")

    # Check for CLI commands or subcommands
    args, unknown = parser.parse_known_args()

    if args.check_env:
        asyncio.run(check_environment())
        return

    if args.offline_test:
        asyncio.run(run_offline_simulation())
        return

    # Default: Run LiveKit Agent CLI
    print("Starting Prism LiveKit Agent Worker...")
    sys.argv = [sys.argv[0]] + unknown
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
