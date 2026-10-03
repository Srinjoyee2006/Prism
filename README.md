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
