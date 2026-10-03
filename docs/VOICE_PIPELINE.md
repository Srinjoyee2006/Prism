# Local Voice Input Pipeline (Stage 4)

This document details the architecture, signal processing concepts, data structures, and execution flow of the local audio input pipeline in **Prism**.

---

## 1. Pipeline Architecture Diagram

```
+-----------------------------------------------------------------------------------------+
|                                    AUDIO INGESTION                                      |
|                                                                                         |
|   [MicrophoneAudioSource]        [WavFileAudioSource]         [SyntheticAudioSource]   |
|   (sounddevice / WASAPI)         (16-bit PCM WAV)             (deterministic tests)    |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              | AudioFrame (16kHz, mono, 16-bit PCM, 32ms)
                                              v
+-----------------------------------------------------------------------------------------+
|                           VOICE ACTIVITY DETECTION (VAD)                                |
|                                                                                         |
|       [SileroVAD (ONNX Runtime)]                 [FakeVAD (Scripted/Energy)]           |
|       - bundled silero_vad_v6.onnx               - reproducible offline tests           |
|       - 512 samples + 64 context window                                                 |
|       - stateful recurrent hidden states (h, c)                                         |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              | (AudioFrame, VADDecision)
                                              v
+-----------------------------------------------------------------------------------------+
|                              SPEECH SEGMENTATION                                        |
|                                                                                         |
|   [SpeechSegmenter State Machine]                                                       |
|   - Pre-speech padding ring-buffer (30ms pad preserves initial consonants)              |
|   - Minimum speech duration gate (100ms threshold filters noise spikes)                 |
|   - Trailing silence threshold (400ms pause finalizes utterance)                        |
|   - Rejects empty / sub-minimum segments                                                |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              | SpeechSegment (raw PCM bytes + timing)
                                              v
+-----------------------------------------------------------------------------------------+
|                         SPEECH-TO-TEXT (ASR INFERENCE)                                  |
|                                                                                         |
|   [FasterWhisperSTT]                            [FakeSTT (Mocks / Tests)]               |
|   - CTranslate2 INT8 quantized on CPU           - canned responses                      |
|   - tiny.en / small.en models                   - zero latency in CI                    |
|   - asyncio.to_thread() offloads CPU work       - failure injection                     |
+---------------------------------------------+-------------------------------------------+
                                              |
                                              | Transcript (text, confidence, words)
                                              v
+-----------------------------------------------------------------------------------------+
|                                   CORE EVENT BUS                                        |
|                                                                                         |
|   SpeechStartedEvent -> SpeechEndedEvent -> SpeechSegmentAvailableEvent ->             |
|   TranscriptFinalEvent / STTFailureEvent                                                |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Concepts

### A. PCM Audio
**Pulse Code Modulation (PCM)** is the uncompressed, raw digital representation of an analog audio signal.
- **Sample Rate**: $16,000 \text{ Hz}$ ($16 \text{ kHz}$), meaning 16,000 discrete amplitude measurements per second. $16 \text{ kHz}$ captures frequencies up to $8 \text{ kHz}$ (Nyquist limit), which covers the full intelligible range of human speech.
- **Channels**: $1$ (mono). Directional / spatial audio is unnecessary for speech recognition and doubles CPU processing overhead.
- **Sample Width**: $2 \text{ bytes}$ ($16\text{-bit}$ signed integers, little-endian, range $-32,768$ to $+32,767$).

### B. Audio Frames
Audio arrives as a continuous temporal stream. To process it in real time, the stream is sliced into uniform chunks called **frames**:
- **Frame Size**: $512 \text{ samples}$ per frame.
- **Duration**:
  $$\text{Duration} = \frac{512 \text{ samples}}{16,000 \text{ samples/sec}} = 0.032 \text{ s} = 32 \text{ ms}$$
- **Byte Size**:
  $$\text{Bytes} = 512 \text{ samples} \times 1 \text{ channel} \times 2 \text{ bytes/sample} = 1,024 \text{ bytes}$$

The `AudioFrame` dataclass encapsulates this raw byte buffer, calculates Root-Mean-Square (RMS) energy, verifies byte alignment invariants, and converts PCM bytes to normalized `float32` arrays in range $[-1.0, 1.0]$.

### C. Voice Activity Detection (VAD)
VAD continuously classifies each $32\text{ms}$ audio frame as **speech** or **silence**:
- **Zero Cost & Local**: Uses the bundled **Silero VAD v6** model running locally on CPU via **ONNX Runtime** (zero downloads, zero external API keys).
- **Stateful Recurrent Hidden States**: Silero uses an RNN. Instead of treating each $32\text{ms}$ slice in isolation, `SileroVAD` preserves the hidden state matrices $h$ and $c$ (shape `[1, 1, 128]`) and a $64\text{-sample}$ context buffer between consecutive frames.
- **Hysteresis Thresholding**: A high threshold ($0.50$) enters speech mode, while a lower threshold ($0.35$) returns to silence, avoiding rapid jitter on borderline frames.

### D. Speech Segmentation
VAD alone only yields per-frame boolean decisions. The **`SpeechSegmenter`** converts those raw decisions into coherent, bounded utterances (`SpeechSegment`):
1. **Pre-speech Padding**: A circular buffer stores the most recent $30\text{ms}$ of audio while in silence. When speech is detected, these frames are prepended to ensure leading consonants (e.g., /p/, /t/, /s/) are not clipped.
2. **Noise Filtering**: If speech ceases before `min_speech_duration_ms` ($100\text{ms}$), the audio is classified as a noise artifact (e.g., keyboard tap or cough) and discarded.
3. **Turn Finalization**: When trailing silence persists for `min_silence_duration_ms` ($400\text{ms}$), the segmenter seals the accumulated frames, computes total duration, and yields a complete `SpeechSegment`.

### E. Speech-to-Text (STT) and faster-whisper
- **Local Neural Inference**: Uses `faster-whisper`, a reimplementation of OpenAI's Whisper model powered by CTranslate2.
- **Quantization**: Runs `int8` quantization on CPU, delivering up to 4x faster execution and 75% memory reduction compared to raw PyTorch Whisper.
- **Default Model**: `tiny.en` (~75MB) or `small.en`, tuned for real-time conversational latency on consumer laptop CPUs.

---

## 3. Why Blocking STT Must NOT Run on the Event Loop

In Python's `asyncio`, all asynchronous tasks, timers, and I/O callbacks run on a **single OS thread**.

If a CPU-intensive neural network inference function (like Whisper transcription taking $200\text{ms} - 800\text{ms}$) runs directly inside an `async def` function:
1. The **entire event loop freezes**.
2. No new microphone frames can be read from the audio hardware queue.
3. Audio hardware buffer **overflows** and sound is dropped.
4. User barge-in interruptions cannot be detected.
5. Outgoing audio playback stutters.

### The Solution: `asyncio.to_thread`
In `FasterWhisperSTT`, blocking inference is offloaded to Python's underlying `concurrent.futures.ThreadPoolExecutor`:
```python
async def transcribe(self, segment: SpeechSegment) -> Transcript:
    # Non-blocking invocation: thread pool executes CTranslate2 C++ loops
    # while the main asyncio loop continues processing audio frames & VAD
    return await asyncio.to_thread(self._sync_transcribe, segment)
```

---

## 4. End-to-End Data Flow

```
1. Microphone / Audio Source
      │  (yields 512-sample PCM frames every 32ms)
      ▼
2. AudioFrame (1024 bytes int16)
      │
      ├──────────────────────────────► [Publishes AudioInputEvent]
      ▼
3. Voice Activity Detector (Silero VAD / FakeVAD)
      │  (evaluates probabilities with RNN context)
      ▼
4. VADDecision (is_speech=True/False, confidence)
      │
      ▼
5. SpeechSegmenter State Machine
      │
      ├── On onset transition  ────────► [Publishes SpeechStartedEvent]
      ├── Accumulates speech frames
      └── On offset transition ────────► [Publishes SpeechEndedEvent]
            │
            ▼ (valid segment >= 100ms)
6. SpeechSegment (concatenated PCM bytes, duration, sample rate)
      │
      ├──────────────────────────────► [Publishes SpeechSegmentAvailableEvent]
      ▼
7. FasterWhisperSTT (asyncio.to_thread worker)
      │  (CTranslate2 INT8 inference on CPU)
      ▼
8. Transcript
      │
      ├── On Success ────────────────► [Publishes TranscriptFinalEvent]
      └── On Error   ────────────────► [Publishes STTFailureEvent]
```
