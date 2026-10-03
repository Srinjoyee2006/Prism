# Text-to-Speech (TTS) & Interruptible Playback Subsystem

Stage 6 of the Prism project introduces zero-cost, fully local Text-to-Speech (TTS) synthesis and low-latency, interruptible audio playback.

---

## 1. Architectural Overview

The audio output pipeline consists of two decoupled primary stages:
1. **Streaming Speech Synthesis**: Converts text into progressive linear 16-bit PCM audio chunks asynchronously.
2. **Interruptible Playback Queue**: Asynchronously buffers and outputs audio to the speaker while providing instantaneous cancellation on user speech onset (barge-in).

```
                                  NORMAL PLAYBACK PIPELINE
+-------------+      +--------------+      +--------------+      +------------------+      +-----------+
| Agent Text  | ---> | Kokoro / TTS | ---> | Audio Chunks | ---> |  Playback Queue  | ---> |  Speaker  |
| (Transcript)|      | (Synthesizer)|      | (24kHz PCM)  |      | InterruptibleQue |      | (Hardware)|
+-------------+      +--------------+      +--------------+      +------------------+      +-----------+

                               BARGE-IN / INTERRUPTION PIPELINE
                    +--------------------+
                    | SpeechStartedEvent | (User speech detected by VAD)
                    +--------------------+
                               |
                               v
                     [ 1. Cancel TTS ] (Abort HTTP stream / worker task)
                               |
                               v
                    [ 2. Flush Queue ] (Atomically discard pending chunks)
                               |
                               v
                    [ 3. Stop Playback ] (Hardware stream.abort() via PortAudio)
                               |
                               v
                   [ 4. Return to LISTENING ] (Reset session state machine)
```

---

## 2. Core Components

### 2.1 Provider-Independent TTS (`src/tts/base.py`)
- **`AudioChunk`**: Immutable dataclass containing `pcm_data: bytes`, `sample_rate: int = 24000`, `channels: int = 1`, `sample_width: int = 2` (16-bit PCM), and `is_terminal: bool`. Calculates duration in milliseconds and total sample count.
- **`TTSProvider(ABC)`**: Abstract base class requiring:
  - `synthesize_stream(text: str) -> AsyncIterator[AudioChunk]`: Asynchronous chunk generator.
  - `is_available() -> bool`: Service readiness probe.
  - `cancel() -> None`: Immediate cancellation signal.

### 2.2 Local Kokoro TTS (`src/tts/kokoro.py`)
- Interfaces with the OpenAI-compatible **Kokoro-FastAPI** service (`POST /v1/audio/speech`).
- Streams 24000Hz 16-bit mono linear PCM audio chunks with sub-second time-to-first-chunk (TTFC).
- Strictly local and free; requires no cloud keys or network access outside localhost.
- Automatically handles stream teardown and client session lifecycle.

### 2.3 Interruptible Audio Player (`src/tts/player.py` / `src/audio/player.py`)
- **`AudioPlayer(ABC)`**: Defines `play(chunk)`, `flush()`, `cancel()`, `is_playing`, and `wait_until_done()`.
- **`SoundDeviceAudioPlayer`**:
  - Uses `sounddevice.RawOutputStream` configured for 24kHz 16-bit PCM.
  - Backed by an `InterruptibleQueue[AudioChunk]`.
  - Background worker writes audio via `asyncio.to_thread` with thread-safe locks, ensuring the main asyncio event loop is never blocked.
  - On `cancel()`, executes PortAudio `stream.abort()` for instantaneous hardware buffer cutoff.
- **`FakeAudioPlayer`**:
  - Deterministic test double that simulates chunk queuing, playback latency, and cancellation without requiring audio hardware.

### 2.4 Orchestration & Barge-In Coordinator (`src/tts/manager.py`)
- Coordinates `TTSProvider` streaming and `AudioPlayer` consumption.
- On `SpeechStartedEvent`, executes the strict cancellation sequence:
  1. `tts.cancel()` and cancels background synthesis task.
  2. `player.flush()` purging pending queue buffers.
  3. `player.cancel()` terminating speaker output.
  4. Transitions `SessionState` back to `LISTENING`.

---

## 3. Separation of Interruption Domains (Crucial Invariant)

Prism enforces a strict architectural boundary between two distinct forms of interruption:

| Domain | Component | Trigger | Action |
| :--- | :--- | :--- | :--- |
| **Tool Execution** | `ToolController` / `CommitGate` | User speaks during 0.9s quiet window | Pending `ToolProposal` is dropped or superseded before execution. |
| **Speech Playback** | `TTSPlaybackManager` / `AudioPlayer` | User speaks while agent is speaking | Active TTS generation is aborted, queue flushed, audio stopped, state resets to `LISTENING`. |

These two subsystems communicate through discrete events (`SpeechStartedEvent`), ensuring tool safety and conversational latency remain completely decoupled.

---

## 4. Test Doubles & Deterministic Verification

All automated tests run 100% offline without requiring Kokoro or audio hardware:
- **`FakeTTS`**: Generates synthetic PCM sine waves or silence across configurable chunk counts and simulated delays; records synthesis history and responds to cancellation.
- **`FakeAudioPlayer`**: Tracks played chunks, flushed chunks, and playback state.

---

## 5. Local Setup and Execution

### Starting Kokoro-FastAPI (Local Docker or Pip)
```bash
# Run CPU version via Docker:
docker run -d -p 8880:8880 --name kokoro-fastapi ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

### Running the Local TTS Demo
```bash
# Run with Kokoro-FastAPI:
python examples/demo_tts.py

# Run in mock test mode (no Kokoro required):
python examples/demo_tts.py --mock

# Demonstrate user speech barge-in cancellation:
python examples/demo_tts.py --mock --test-interrupt

# Custom voice or text:
python examples/demo_tts.py --text "Prism is an interruptible conversational voice agent." --voice af_bella
```
