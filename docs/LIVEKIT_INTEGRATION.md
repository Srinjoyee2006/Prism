# Stage 7: LiveKit WebRTC Transport Integration

This document outlines the architecture, integration patterns, audio lifecycle, and verification workflows for connecting the Prism Full-Duplex Voice Agent to LiveKit WebRTC transport.

---

## 1. LiveKit's Role in Prism

In the Prism architecture, **LiveKit acts purely as an edge transport layer** providing real-time WebRTC audio I/O between remote participants (web browsers, mobile apps, telephony SIP trunks) and the agent pipeline.

### Architectural Boundary
* **No Core Logic in LiveKit**: LiveKit does not manage conversational state machines, intent parsing, or tool gating.
* **No Direct Tool Execution**: LiveKit **NEVER** executes tools directly.
* **Preservation of Stages 1–6**: All Stage 1–6 components (`ToolController`, `CommitGate`, `VoiceAgentOrchestrator`, `SileroVAD`, `SpeechSegmenter`, `faster-whisper`, `OllamaClient`, `KokoroTTS`, `TTSPlaybackManager`) operate unmodified behind thin adapter interfaces.

---

## 2. End-to-End Audio & Data Flow

```
                      +-----------------------------------+
                      |       LiveKit WebRTC Room         |
                      +-----------------------------------+
                             |                     ^
             Remote Audio    |                     | LocalAudioTrack
             (WebRTC Opus)   v                     | (24kHz 16-bit PCM)
                      +-------------------+  +-------------------+
                      | LiveKitAudioSource|  | LiveKitAudioPlayer|
                      +-------------------+  +-------------------+
                             |                     ^
              16kHz PCM      |                     | 24kHz AudioChunk
              AudioFrames    v                     | (non-blocking)
                      +-------------------+  +-------------------+
                      |   AudioPipeline   |  | TTSPlaybackManager|
                      | (Silero VAD +     |  +-------------------+
                      |  Segmenter + STT) |        ^
                      +-------------------+        | Synthesized
                             |                     | Audio Chunks
              Transcript     |                     |
              Final Event    v               +-------------------+
                      +-------------------+  |  Kokoro-FastAPI   |
                      | VoiceAgent-       |  |  (Local CPU TTS)  |
                      | Orchestrator      |  +-------------------+
                      +-------------------+        ^
                             |                     | Response Text
              User Message   |                     |
                             v                     |
                      +-------------------+        |
                      |  Ollama (Local)   |        |
                      |  (qwen2.5:1.5b)   |        |
                      +-------------------+        |
                             |                     |
              ToolProposal   | (Proposals ONLY,    |
                             |  NO direct exec)    |
                             v                     |
                      +-------------------+        |
                      |  ToolController / |--------+ (On tool success)
                      |    Commit Gate    |
                      +-------------------+
                             |
              0.9s Quiet     | (No user speech interruption)
              Window Cleared v
                      +-------------------+
                      | Deterministic     |
                      | Mock Tools        |
                      +-------------------+
```

---

## 3. Provider / Transport Adapters

Prism defines two decoupled transport adapters in `src/livekit/adapters.py`:

### `LiveKitAudioSource` (Inbound Audio)
* Inherits from `src.audio.source.AudioSource`.
* Subscribes to remote participant audio tracks via `rtc.AudioStream(track, sample_rate=16000, num_channels=1)`.
* Buffers variable-length WebRTC incoming audio frames and slices them into exact 512-sample (32ms at 16kHz) `AudioFrame` instances.
* Emits frames into `AudioPipeline` for real-time Silero VAD evaluation and speech segmentation.
* Allows offline testing via `push_frame(pcm_data)` without requiring any network connections or LiveKit servers.

### `LiveKitAudioPlayer` (Outbound Audio)
* Inherits from `src.tts.player.AudioPlayer`.
* Holds a reference to a LiveKit `rtc.AudioSource(sample_rate=24000, num_channels=1)` published to the room as an `rtc.LocalAudioTrack`.
* Receives `AudioChunk` objects from `TTSPlaybackManager` and asynchronously pushes `rtc.AudioFrame` objects to `rtc.AudioSource.capture_frame()`.
* **Instantaneous Barge-In Cancellation**: On `cancel()` or `flush()`, it clears the internal playback queue and immediately calls `rtc_source.clear_queue()`, terminating in-flight WebRTC playout to participant speakers.

---

## 4. The Commit Gate & Critical Invariants

### Invariant 1: LLM Tool Proposal != Tool Execution
The local LLM (`qwen2.5:1.5b`) is **strictly a planner**. It can only output `ToolProposal` objects containing tool names and arguments:
```
LLM Output -> ToolProposal -> ToolController -> Commit Gate -> Execution
```
Under no circumstances may LiveKit or the LLM trigger tool actions directly.

### Invariant 2: 0.9-Second Quiet Window
Every proposed tool call must survive a **0.9-second quiet window** in the `ToolController`:
* If the user remains silent for 0.9s, the proposal is committed and executed by the registered handler.
* If user speech onset is detected by Silero VAD before 0.9s elapses:
  * `ToolController.notify_user_speech_started()` is invoked immediately.
  * The staged proposal is transitioned to `DROPPED` or `SUPERSEDED`.
  * The tool is **NEVER** executed.

### Invariant 3: Mid-Utterance TTS Barge-In
If the agent is speaking to the room and the user begins speaking:
1. Silero VAD detects user speech onset.
2. `AudioPipeline` invokes `_handle_speech_start()`.
3. `TTSPlaybackManager.handle_user_speech_started()`:
   * Cancels active Kokoro synthesis tasks.
   * Flushes pending audio chunks from `LiveKitAudioPlayer`.
   * Calls `clear_queue()` on LiveKit's `rtc.AudioSource`.
   * Halts output audio playout to participants.
   * Transitions agent session state back to `LISTENING`.

---

## 5. Graceful Disconnect (`close_on_disconnect=False`)

LiveKit's default `RoomIO` behavior terminates the agent worker session immediately when the participant disconnects.

In benchmark evaluations (e.g., FDB-v3), participants may hang up or disconnect right as their command ends. If the agent session is aborted immediately upon disconnect, in-flight tool proposals in the 0.9s quiet window and asynchronous execution logging would be abruptly killed.

Prism configures `RoomOptions` with:
```python
room_options = RoomOptions(
    close_on_disconnect=False,
    audio_input=False,
    audio_output=False,
)
```
This guarantees that when a participant disconnects, Prism has sufficient time to execute:
```python
await self.tool_controller.drain(timeout=2.0)
```
allowing pending quiet windows, execution handlers, and telemetry logging to finalize cleanly before shutdown.

---

## 6. Local & Free Component Stack

Prism uses 100% free and local components:
* **Transport**: LiveKit WebRTC (runs locally via official Docker image or LiveKit Cloud free tier).
* **VAD**: Local Silero VAD ONNX Runtime (CPU inference, 32ms frames).
* **STT**: `faster-whisper` (`tiny.en` INT8 on CPU).
* **LLM**: Local Ollama (`qwen2.5:1.5b` running on CPU/GPU at `http://localhost:11434`).
* **TTS**: Local Kokoro-FastAPI (`af_heart` neural voice running via Docker CPU container at `http://localhost:8880/v1`).
* **Tools**: Deterministic mock tools (temperature control, flight search, currency exchange, order tracking).

No paid APIs, OpenAI keys, or proprietary cloud subscriptions are required.

---

## 7. Setup & Execution Guide

### Prerequisites
1. **Docker Desktop** (running).
2. **Local Kokoro TTS Container**:
   ```powershell
   docker run -d -p 8880:8880 --name kokoro-fastapi ghcr.io/remsky/kokoro-fastapi-cpu:latest
   ```
3. **Local Ollama**:
   ```powershell
   ollama serve
   ollama pull qwen2.5:1.5b
   ```

### Running a Local LiveKit Server
To test WebRTC connections locally without LiveKit Cloud, run the official LiveKit development server in Docker:
```powershell
docker run --rm -it -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```
This starts a local WebRTC server with default development credentials:
* **URL**: `ws://127.0.0.1:7880`
* **API Key**: `devkey`
* **API Secret**: `secret`

Add these to your local `.env`:
```ini
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

---

## 8. Development & Demo Commands

### Environment Pre-flight Check
Validates connectivity to Ollama, Kokoro, and LiveKit configuration:
```powershell
python examples/demo_livekit.py --check-env
```

### Offline Simulation (No LiveKit Server Needed)
Runs an automated, end-to-end offline simulation demonstrating:
* Scenario 1: Clean Commit Gate execution past the quiet window with audio playout to WebRTC track.
* Scenario 2: Barge-in interruption during the quiet window dropping the uncommitted proposal.
* Scenario 3: Mid-utterance barge-in cancelling active TTS and flushing the LiveKit audio buffer.
```powershell
python examples/demo_livekit.py --offline-test
```

### Run LiveKit Agent Worker (Development Mode)
Connects the agent to the LiveKit server to handle incoming rooms:
```powershell
python src/livekit_agent.py dev
# OR
python examples/demo_livekit.py dev
```

---

## 9. Testing & Verification

### Offline Test Suite
The unit and integration test suite (`tests/test_livekit_adapters.py`) uses mocked WebRTC objects (`FakeRtcSource`, `FakeRtcAudioStream`) and runs **100% offline**:
```powershell
# Run LiveKit adapter tests
python -m pytest tests/test_livekit_adapters.py -v

# Run entire Prism test suite (147+ tests)
python -m pytest -q
```

### Which Parts Require LiveKit Credentials vs Run Offline?
| Component / Workflow | Requires LiveKit Server/Credentials? | Can Be Tested Offline? |
| :--- | :--- | :--- |
| `LiveKitAudioSource` unit tests | **No** (Uses synthetic PCM frames / streams) | **Yes** |
| `LiveKitAudioPlayer` unit tests | **No** (Uses `FakeRtcSource`) | **Yes** |
| Commit Gate Quiet Window & Barge-in tests | **No** (Simulated VAD speech onset) | **Yes** |
| Full offline simulation (`demo_livekit.py --offline-test`) | **No** | **Yes** |
| Live WebRTC connection with browser/participant | **Yes** (Local Docker LiveKit or LiveKit Cloud) | **No** |
