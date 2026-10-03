"""
Interactive Local Microphone Transcription Demo (Stage 4).

Usage:
    python examples/demo_mic_transcription.py

Listens to the default microphone, performs local ONNX VAD, segments speech,
and transcribes speech utterances using local faster-whisper without paid APIs.
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

from src.asr.faster_whisper_asr import FasterWhisperSTT
from src.audio.microphone import MicrophoneAudioSource
from src.audio.pipeline import AudioPipeline
from src.core.events import TranscriptFinalEvent
from src.vad.segmenter import SpeechSegmenter
from src.vad.silero_vad import SileroVAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo")


async def main() -> None:
    if not MicrophoneAudioSource.is_available():
        print("ERROR: No compatible microphone input device found.")
        sys.exit(1)

    print("=" * 60, flush=True)
    print("Prism Stage 4: Live Local Microphone Transcription Demo", flush=True)
    print("=" * 60, flush=True)
    print("- VAD: Local Silero VAD (ONNX Runtime)", flush=True)
    print("- ASR: faster-whisper (tiny.en quantized int8 on CPU)", flush=True)
    print("- Audio: 16kHz Mono 16-bit PCM (sounddevice)", flush=True)
    print("Speak into your microphone. Press Ctrl+C to exit.\n", flush=True)

    source = MicrophoneAudioSource(sample_rate=16000, chunk_size_samples=512)
    vad = SileroVAD(threshold=0.5)
    segmenter = SpeechSegmenter(
        min_speech_duration_ms=100.0,
        min_silence_duration_ms=400.0,
        speech_pad_ms=30.0,
    )
    stt = FasterWhisperSTT(model_size_or_path="tiny.en", device="cpu", compute_type="int8")

    async def on_speech_start() -> None:
        print("\n>>> [Speech Detected...]", end="", flush=True)

    async def on_speech_end() -> None:
        print(" [Speech Ended. Transcribing...]", flush=True)

    async def on_transcript(event: TranscriptFinalEvent) -> None:
        print(f"TRANSCRIPT: \"{event.text}\" (confidence={event.confidence:.2f}, duration={event.duration_ms:.0f}ms)\n")

    pipeline = AudioPipeline(
        source=source,
        vad=vad,
        segmenter=segmenter,
        stt=stt,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        on_transcript=on_transcript,
    )

    try:
        await pipeline.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping microphone capture...")
    finally:
        await pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
