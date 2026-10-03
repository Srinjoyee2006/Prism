# Full-Duplex Voice Agent Pipeline with Tool Commit Gate (Stage 5)

This document details the end-to-end integration of the voice input pipeline (Stage 4), local LLM planning (Stage 2), and the Tool Commit Gate (Stage 3) into an interruptible voice agent architecture.

---

## 1. End-to-End Architecture Diagram

```
+-------------------------------------------------------------------------------------------------+
|                                         AUDIO SUBSYSTEM                                         |
|                                                                                                 |
|   [MicrophoneAudioSource]                                                                       |
|   (sounddevice / WASAPI 16kHz Mono 16-bit PCM)                                                  |
+-----------------------------------------------+-------------------------------------------------+
                                                |
                                                | 32ms AudioFrame (512 samples)
                                                v
+-------------------------------------------------------------------------------------------------+
|                                 VAD & SPEECH SEGMENTATION                                       |
|                                                                                                 |
|   [SileroVAD (ONNX)] ────────► [SpeechSegmenter]                                                |
|   - 32ms evaluation            - 30ms pre-speech padding ring-buffer                            |
|   - stateful RNN hidden states - 100ms noise spike gate                                         |
|   - speech onset / offset       - 400ms trailing silence gate                                   |
+-----------------------------------------------+-------------------------------------------------+
                                                |
                                                | SpeechSegment (raw PCM bytes + duration)
                                                v
+-------------------------------------------------------------------------------------------------+
|                                SPEECH-TO-TEXT (ASR INFERENCE)                                   |
|                                                                                                 |
|   [FasterWhisperSTT (asyncio.to_thread)]                                                        |
|   - CTranslate2 INT8 quantized on CPU                                                           |
|   - Non-blocking thread-pool execution                                                          |
|   - Generates TranscriptFinalEvent                                                              |
+-----------------------------------------------+-------------------------------------------------+
                                                |
                                                | Transcript: "Set temperature to 21 degrees"
                                                v
+-------------------------------------------------------------------------------------------------+
|                                LOCAL LLM PLANNER (THE BRAIN)                                    |
|                                                                                                 |
|   [OllamaClient (qwen2.5:1.5b)]                                                                 |
|   - Zero-cost, 100% local CPU inference                                                         |
|   - Native function / tool calling format                                                       |
|   - Emits validated ToolProposal instances                                                      |
|                                                                                                 |
|   CRUCIAL RULE: The LLM NEVER directly executes any tool. It only produces proposals.           |
+-----------------------------------------------+-------------------------------------------------+
                                                |
                                                | ToolProposal (name="set_temperature", args={...})
                                                v
+-------------------------------------------------------------------------------------------------+
|                                   TOOL COMMIT GATE                                              |
|                                                                                                 |
|   [ToolController] ──────────► [CommitGate]                                                     |
|   - Normalised argument key    - 0.9s configurable quiet window countdown                       |
|   - In-flight deduplication    - Races timer against user barge-in event                        |
|   - Proposal supersession      - DROPS proposal if user begins speaking                         |
+-----------------------------------------------+-------------------------------------------------+
                                                | (Quiet window elapses without barge-in)
                                                v
+-------------------------------------------------------------------------------------------------+
|                                   MOCK TOOL EXECUTION                                           |
|                                                                                                 |
|   [Deterministic Handlers]                                                                      |
|   - Async handlers: mock_set_temperature, mock_search_flights, mock_get_exchange_rate           |
|   - Blocking handlers: blocking_track_order (wrapped in asyncio.to_thread)                       |
|   - Emits structured results: {'status': 'ok', 'temperature_c': 21.0, ...}                      |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Principle: Why `LLM Proposal != Tool Execution`

In traditional conversational architectures and naive agent loops, tool calls produced by an LLM are executed immediately. In a **full-duplex conversational voice agent**, direct execution is a critical design failure for three fundamental reasons:

### 1. Natural Human Speech is Non-Atomic and Self-Correcting
Human conversation is filled with false starts, mid-sentence hesitations, and real-time corrections:
> *"Computer, set the thermostat in the living room to 20... actually no, wait, make it 24 degrees."*

If the system executed tools the moment the first utterance or token completed, the thermostat would adjust to $20^\circ\text{C}$ before the user could finish speaking. Side effects (e.g., locking doors, executing payments, booking reservations, or firing physical actuators) would occur prematurely.

### 2. Side Effects Cannot Simply Be "Un-Said"
Unlike text chat where an errant message is easily edited, real-world tools have external side effects:
- Updating databases
- Calling third-party APIs
- Adjusting physical hardware
- Triggering notifications

Once a side effect is committed, reversing it requires compensating transactions or manual recovery. Decoupling the proposal from execution ensures that actions are only authorized when the user has genuinely finished speaking.

### 3. The Tool Commit Gate as the Interruption Barrier
The `ToolController` enforces that every tool proposal must pass through a **Commit Gate** with a configurable **quiet window** (default $0.9\text{ seconds}$, per the Full-Duplex-Bench v3 specification).

- When a proposal is generated, it enters the `WAITING` state.
- If the user starts speaking again during this $0.9\text{s}$ interval (detected instantaneously by Silero VAD), an interrupt signal is broadcast.
- The `CommitGate` cancels the pending execution task and marks the proposal as `DROPPED` or `SUPERSEDED`.
- The physical tool handler is **never invoked**.

---

## 3. Data Flow and Component Responsibilities

1. **Audio Ingestion**:
   - `MicrophoneAudioSource` streams 16kHz mono 16-bit PCM frames (512 samples / 32ms) from PortAudio/sounddevice into an async queue.
2. **VAD & Segmentation**:
   - `SileroVAD` evaluates each 32ms frame locally via ONNX Runtime, preserving recurrent hidden states ($h, c$).
   - `SpeechSegmenter` uses a 30ms pre-speech ring buffer to prevent consonant clipping and a 400ms silence threshold to delineate completed utterances.
3. **Speech-to-Text**:
   - `FasterWhisperSTT` transcribes the segment using INT8 quantized Whisper on CPU inside `asyncio.to_thread()`, keeping the asyncio event loop responsive.
4. **LLM Planning**:
   - `VoiceAgentOrchestrator` formats the transcript into a conversational message and queries local Ollama (`qwen2.5:1.5b`) with registered tool schemas.
   - The LLM produces one or more `ToolProposal` objects.
5. **Gate Authorization**:
   - `ToolController` receives the `ToolProposal`, builds a canonical normalized key for idempotency, cancels any existing waiting proposal for the same tool, and starts the 0.9s quiet window.
6. **Execution**:
   - Once the quiet window completes with zero user interruptions, the controller authorizes execution and invokes the tool handler (using `asyncio.to_thread` for blocking functions).

---

## 4. Running the Interactive Demo

To test the complete end-to-end voice pipeline:

```powershell
# 1. Activate virtual environment
cd D:\Prism
.\.venv\Scripts\Activate.ps1

# 2. Ensure Ollama is running locally
ollama run qwen2.5:1.5b

# 3. Launch the full-duplex voice agent demo
python examples/demo_voice_agent.py
```

Try speaking:
- *"Set the living room temperature to 21 degrees"* -> watch the proposal enter `WAITING`, transition to `COMMITTED`, and then `SUCCEEDED`.
- *"Set the bedroom temperature to 18 degrees... wait, make it 23"* -> observe the first proposal get `DROPPED` by barge-in and the second proposal execute.
