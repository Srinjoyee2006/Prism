"""
Local TTS and Interruptible Audio Playback Demo.

Demonstrates local Kokoro speech synthesis and asynchronous,
interruptible audio playback through default Windows audio output.

Usage:
    # Run with local Kokoro-FastAPI:
    python examples/demo_tts.py

    # Run in mock/test mode without needing Kokoro:
    python examples/demo_tts.py --mock

    # Run with a custom sentence:
    python examples/demo_tts.py --text "Prism is an interruptible conversational AI agent."

    # Demonstrate barge-in cancellation:
    python examples/demo_tts.py --mock --test-interrupt
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.state import SessionState
from src.tts.base import TTSConnectionError, TTSError
from src.tts.kokoro import KokoroTTS
from src.tts.manager import TTSPlaybackManager
from src.tts.mock import FakeTTS
from src.tts.player import SoundDeviceAudioPlayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_tts")


async def run_demo(
    text: str,
    use_mock: bool = False,
    voice: str = "af_heart",
    base_url: str = "http://localhost:8880/v1",
    test_interrupt: bool = False,
) -> None:
    """Run local TTS synthesis and audio playback."""
    print("=" * 65)
    print("  PRISM LOCAL TTS & INTERRUPTIBLE PLAYBACK DEMO")
    print("=" * 65)

    # 1. Initialize TTS Engine
    if use_mock:
        print("[Engine] Using FakeTTS (Deterministic synthetic audio for testing)")
        tts = FakeTTS(sample_rate=24000, chunk_count=6, chunk_duration_seconds=0.3, delay_per_chunk=0.08)
    else:
        print(f"[Engine] Using KokoroTTS (Connecting to {base_url}, voice: {voice})")
        tts = KokoroTTS(base_url=base_url, voice=voice, sample_rate=24000)

        # Check availability
        is_ready = await tts.is_available()
        if not is_ready:
            print("\n[WARNING] Kokoro-FastAPI was not detected at", base_url)
            print("To start local Kokoro-FastAPI, run:")
            print("    docker run -d -p 8880:8880 --name kokoro-fastapi ghcr.io/remsky/kokoro-fastapi-cpu:latest")
            print("\nFalling back to FakeTTS so you can verify audio playback architecture...")
            tts = FakeTTS(sample_rate=24000, chunk_count=8, chunk_duration_seconds=0.25, delay_per_chunk=0.05)

    # 2. Initialize Audio Player
    print("[Player] Initializing SoundDeviceAudioPlayer (24kHz 16-bit mono)...")
    player = SoundDeviceAudioPlayer(sample_rate=24000, channels=1)
    await player.start()

    session_state = SessionState()
    manager = TTSPlaybackManager(
        tts=tts,
        player=player,
        session_state=session_state,
        on_playback_started=lambda: logger.info("Playback started! Audio flowing to speaker."),
        on_playback_completed=lambda: logger.info("Playback completed successfully."),
        on_interrupted=lambda: logger.info(">>> Playback was INTERRUPTED! Queue flushed, audio halted. <<<"),
    )

    print(f"\n[Synthesizing]: \"{text}\"")

    try:
        if test_interrupt:
            # Demonstrate interruption: spawn speak and trigger user speech onset after 400ms
            speak_task = asyncio.create_task(manager.speak(text))

            async def simulate_user_barge_in() -> None:
                await asyncio.sleep(0.4)
                print("\n[EVENT] >>> User speech detected! Simulating barge-in interruption... <<<")
                await manager.handle_user_speech_started()

            interrupt_task = asyncio.create_task(simulate_user_barge_in())
            await asyncio.gather(speak_task, interrupt_task, return_exceptions=True)
            print(f"[Result] Interruption demo finished. Agent state: {session_state.current_state.value}")
        else:
            success = await manager.speak(text)
            if success:
                print("\n[Result] Synthesis and playback completed successfully.")
            else:
                print("\n[Result] Playback was cancelled or ended early.")

    except (TTSConnectionError, TTSError) as err:
        print(f"\n[TTS Error]: {err}")
    except asyncio.CancelledError:
        print("\n[Demo cancelled by user]")
    finally:
        print("\n[Teardown] Stopping audio player...")
        await player.stop()
        if hasattr(tts, "close"):
            await tts.close()
        print("[Teardown] Done.")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prism Local TTS and Playback Demo")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is Kokoro Text-to-Speech running entirely locally on Windows.",
        help="Text sentence to synthesize and speak",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use FakeTTS instead of Kokoro",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="af_heart",
        help="Kokoro voice name (e.g. af_heart, af_bella, am_adam)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8880/v1",
        help="Base URL for Kokoro-FastAPI server",
    )
    parser.add_argument(
        "--test-interrupt",
        action="store_true",
        help="Simulate a barge-in interruption during playback",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_demo(
                text=args.text,
                use_mock=args.mock,
                voice=args.voice,
                base_url=args.base_url,
                test_interrupt=args.test_interrupt,
            )
        )
    except KeyboardInterrupt:
        print("\nExited cleanly via Ctrl+C.")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"\nDemo exited with {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
