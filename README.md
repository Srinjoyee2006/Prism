# Prism: Interruptible Voice Agent with Tool Commit Gate

> Built for Full-Duplex-Bench v3 benchmarking. 100% Zero-Cost, Local-First Architecture.

---

## Overview

Prism is a low-latency, full-duplex conversational voice agent equipped with:
1. **Low-Latency Barge-in / Interruption Handling**: Instantly stops agent playback when user speech is detected via low-overhead VAD.
2. **Tool Commit Gate**: Prevents premature or hallucinated tool execution during tentative or interrupted speech turns by staging actions until user stability is confirmed.
3. **Local Zero-Cost Stack**: Powered by Silero VAD (ONNX), faster-whisper (CTranslate2 int8), Ollama (Qwen 2.5), and edge-tts / pyttsx3.

---

## Project Structure

```
Prism/
├── configs/
│   ├── agent_config.yaml          # Agent latency thresholds, VAD & model configs
│   └── tools_config.yaml          # Tool definitions & commit gate policies
├── docs/
│   ├── ARCHITECTURE.md            # System architecture & full-duplex event loops
│   └── SETUP_GUIDE.md             # Prerequisites installation guide (Ollama, FFmpeg)
├── src/
│   ├── audio/                     # Audio capture and non-blocking playback streams
│   ├── vad/                       # Low-latency Voice Activity Detection (Silero)
│   ├── asr/                       # Streaming speech recognition (faster-whisper)
│   ├── llm/                       # Ollama async client & function calling
│   ├── tts/                       # Streaming TTS synthesis with interruption support
│   ├── gate/                      # Tool Commit Gate state machine (Staged -> Commit)
│   ├── tools/                     # Mock & local tools registry
│   └── agent/                     # Top-level orchestrator & state machine
├── tests/
│   ├── test_vad.py                # VAD unit tests
│   ├── test_commit_gate.py        # Tool commit gate logic tests
│   └── test_audio_pipeline.py     # End-to-end loopback audio tests
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Production dependencies
└── requirements-dev.txt           # Test & development dependencies
```

---

## Setup & Verification

See [SETUP_GUIDE.md](file:///d:/Prism/docs/SETUP_GUIDE.md) for step-by-step instructions.

To run the automated test suite:
```powershell
pytest -q
```

---

## Interactive Demos

### 1. Stage 4: Local Microphone Transcription Demo
Transcribes live microphone speech using local Silero VAD and faster-whisper on CPU:
```powershell
python examples/demo_mic_transcription.py
```

### 2. Stage 5: Full-Duplex Voice Agent with Tool Commit Gate Demo
Complete end-to-end flow: Microphone -> Silero VAD -> faster-whisper -> Ollama Qwen 2.5 -> ToolProposal -> ToolController / Commit Gate (0.9s quiet window) -> Mock Tool Execution:
```powershell
# Ensure Ollama daemon is running with Qwen 2.5 1.5B
ollama run qwen2.5:1.5b

# Run the voice agent
python examples/demo_voice_agent.py
```

### 3. Stage 6: Local TTS & Interruptible Audio Playback Demo
Synthesizes speech using local Kokoro TTS and plays audio through the default Windows speaker output with immediate barge-in interruption support:
```powershell
# Run with local Kokoro-FastAPI:
python examples/demo_tts.py

# Run in mock mode (no Kokoro service or hardware required):
python examples/demo_tts.py --mock

# Demonstrate barge-in cancellation (interrupted mid-playback):
python examples/demo_tts.py --mock --test-interrupt

# Custom sentence or voice:
python examples/demo_tts.py --text "Prism is an interruptible conversational AI agent." --voice af_heart
```

---

## Local Kokoro TTS Setup

Prism utilizes **Kokoro** as its local, zero-cost neural text-to-speech engine. The recommended deployment uses the OpenAI-compatible Kokoro-FastAPI server:

```powershell
docker run -d -p 8880:8880 --name kokoro-fastapi ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

Once running, Kokoro serves streaming 24000Hz 16-bit PCM at `http://localhost:8880/v1`.

### Troubleshooting Kokoro & Audio Playback
- **Connection Error / Server not running**: If Kokoro-FastAPI is not running, `examples/demo_tts.py` will warn and automatically fall back to `FakeTTS` so you can verify the audio playback and barge-in architecture without interruptions.
- **Port Conflicts**: If port 8880 is in use, pass `--base-url http://localhost:<port>/v1` to `demo_tts.py` or set the `KOKORO_BASE_URL` environment variable.
- **No Sound / Driver Issues**: `SoundDeviceAudioPlayer` uses the default Windows output device. You can verify available devices via `python -c "import sounddevice as sd; print(sd.query_devices())"`.

---

## Technical Documentation
- [System Architecture](file:///d:/Prism/docs/ARCHITECTURE.md)
- [Local Voice Input Pipeline (Stage 4)](file:///d:/Prism/docs/VOICE_PIPELINE.md)
- [Voice Agent & Tool Commit Gate Pipeline (Stage 5)](file:///d:/Prism/docs/VOICE_AGENT_PIPELINE.md)
- [Local TTS & Interruptible Playback Pipeline (Stage 6)](file:///d:/Prism/docs/TTS_PIPELINE.md)
- [Setup & Environment Guide](file:///d:/Prism/docs/SETUP_GUIDE.md)
