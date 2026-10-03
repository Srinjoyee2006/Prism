# Full-Duplex-Bench v3 (FDB-v3) Integration Guide

This document describes the integration between the **Prism Voice Agent** and the official **Full-Duplex-Bench v3 (FDB-v3)** multi-step tool calling benchmark.

---

## 1. Overview & Provider Identity

- **Benchmark Name**: Full-Duplex-Bench v3 (FDB-v3)
- **Target Provider**: `prism_local`
- **Dataset Structure**: 100 spoken examples across 4 domains (Travel & Identity, Finance & Billing, Housing & Location, E-Commerce Support) recorded from 12 speakers with natural conversational disfluencies.
- **Goal**: Allow the official FDB-v3 evaluation pipeline to stream realistic spoken queries into a local LiveKit room, drive the full Prism conversational voice agent, record agent audio and tool telemetry, and evaluate task completion.

---

## 2. Architecture & Data Flow

Prism maintains its core architectural invariants during benchmark execution. The LLM never calls tools directly, and the 0.9s Commit Gate is strictly enforced:

```
[Official input.wav (48 kHz)]
            │
            ▼
[Headless LiveKit WebRTC Client (benchmark/prism_adapter.py)]
            │
            ▼
[Local LiveKit Server (ws://127.0.0.1:7880)]
            │
            ▼
[Inbound Participant Track]
            │
            ▼
[Silero VAD + SpeechSegmenter (src/vad)]
            │
            ▼
[Local faster-whisper STT (src/asr)]
            │
            ▼
[Local Ollama LLM: qwen2.5:1.5b (src/llm)]
            │ (Proposes tools only)
            ▼
[ToolProposal (src/llm/models.py)]
            │
            ▼
[ToolController & 0.9s Commit Gate (src/gate)]
            │
            ├── (If user interrupts during 0.9s quiet window -> DROPPED / SUPERSEDED)
            │
            └── (After 0.9s quiet window without interruption -> COMMITTED)
                        │
                        ▼
            [FDB Mock Tool Execution (src/tools/fdb_tools.py)]
                        │
                        ├── Logs Telemetry to agent_tool_calls.log
                        │
                        ▼
            [Local Kokoro TTS (src/tts/kokoro.py)]
                        │
                        ▼
            [LiveKitAudioPlayer -> Outbound LiveKit Track]
                        │
                        ▼
            [Benchmark Receiver records output_prism_local.wav]
```

---

## 3. Strict Zero-Cost & Local Constraints

To ensure 100% free and local operation:
- **No Paid APIs**: No OpenAI, GPT-4o judge, Anthropic, Gemini, or cloud services.
- **Local LiveKit Server**: Running locally via Docker on `ws://127.0.0.1:7880`.
- **Local Neural Models**:
  - Silero VAD (ONNX, local CPU)
  - faster-whisper (`tiny.en`, local CPU)
  - Ollama (`qwen2.5:1.5b`, local GPU/CPU)
  - Kokoro TTS (Docker CPU FastAPI endpoint at `http://localhost:8880/v1`)
- **No-LLM Rule-Based Evaluation**: Both `evaluate_pass_rate.py` and `evaluate_tool_calls.py` are executed without `--use-llm`, performing deterministic exact matching on tool selection and arguments.

---

## 4. The 12 Official FDB-v3 Domain Tools

Prism registers the official 12 FDB-v3 tool schemas and deterministic mock handlers under `src/tools/fdb_tools.py`:

| Domain | Tool Name | Parameters |
|---|---|---|
| Travel & Identity | `search_flights` | `destination` (str), `date` (str) |
| Travel & Identity | `book_flight` | `passenger_name` (str), `flight_id` (str, optional) |
| Travel & Identity | `update_identity_doc` | `doc_type` (str), `doc_number` (str) |
| Finance & Billing | `get_card_benefits` | `card_type` (str) |
| Finance & Billing | `get_exchange_rate` | `amount` (float), `from_currency` (str), `to_currency` (str) |
| Finance & Billing | `modify_autopay` | `bill_type` (str), `source_account` (str) |
| Housing & Location | `search_apartments` | `city` (str), `bedrooms` (int), `max_price` (float) |
| Housing & Location | `calculate_commute` | `origin_address` (str), `destination_address` (str), `mode` (str, optional) |
| Housing & Location | `update_search_filter` | `filter_name` (str), `value` (str) |
| E-Commerce | `track_order` | `order_id` (str) |
| E-Commerce | `search_products` | `query` (str), `max_price` (float, optional) |
| E-Commerce | `add_to_cart` | `product_id` (str), `quantity` (int, optional) |

---

## 5. Discovered Incompatibilities & Resolutions

During integration of the official FDB-v3 repository on Windows, the following incompatibilities were resolved:

1. **Telemetry File Path on Windows**:
   - Official FDB-v3 hardcodes `Path("/tmp/agent_tool_calls.log")`. On Windows, this resolves to `D:\tmp\agent_tool_calls.log` or fails if `D:\tmp` does not exist.
   - *Resolution*: `FDBTelemetryCollector` automatically creates parent directories and writes to both `Path("/tmp/agent_tool_calls.log")` and `benchmark/agent_tool_calls.log`, while `prism_adapter.py` checks both locations.
2. **ASR Model Dependency**:
   - Official FDB-v3 specifies `nvidia/parakeet-tdt-0.6b-v2` via `nemo_toolkit[asr]`. NeMo is heavy, platform-sensitive, and not installed in standard Windows environments.
   - *Resolution*: Implemented `run_local_asr()` using `faster-whisper`, generating word-level timestamp chunks in the exact format required by FDB-v3 evaluation scripts without external dependencies.
3. **Spoken Identifier Disfluencies**:
   - Spoken strings like "The order ID is A-B-C-1-2-3" produce hyphenated tokens in Whisper. The ground truth in `metadata.json` expects `"ABC123"`.
   - *Resolution*: System prompt instructs continuous alphanumeric representation for identifiers, and exact-match normalization strips spacing and punctuation.

---

## 6. Commands

### Step 1: Ensure Local Services Are Running
```bash
# 1. LiveKit server (Docker)
docker run --rm -it -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev

# 2. Kokoro TTS (Docker)
docker run --rm -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

# 3. Ollama server
ollama serve
```

### Step 2: Start Prism Agent Worker in Benchmark Mode
Set `PRISM_BENCHMARK_MODE=1` to register all 12 FDB tools:
```powershell
$env:PRISM_BENCHMARK_MODE="1"
python -m src.livekit_agent dev
```

### Step 3: Run the 1-Example Smoke Test
```powershell
python benchmark/run_prism_benchmark.py --smoke-test --evaluate
```

### Step 4: Run a Specific Scenario
```powershell
python benchmark/run_prism_benchmark.py --example ecommerce_01 --evaluate
```

### Step 5: Run the Full 100-Example Benchmark (Future Batch Run)
*(Note: Do not run before smoke test verification)*
```powershell
python benchmark/run_prism_benchmark.py --all --evaluate
```

---

## 7. Output Artifacts

For each evaluated example, the following files are produced:
- `output_prism_local.wav`: The agent's spoken WebRTC audio response recorded from the room.
- `result_prism_local.json`: Complete scenario execution record containing:
  - `input_transcript` and `input_asr_chunks` (word timestamps)
  - `user_speech_end_rel` (timestamp when user finished speaking)
  - `audio_agent_speech_start` and `perceived_total_latency`
  - `actual_tool_calls` (tool name, arguments, `timestamp_start`, `timestamp_end` relative to stream start)
- `prism_local_pass_rate_report.json`: Binary pass/fail metric summary generated by `evaluate_pass_rate.py`.
- `prism_local_evaluation_report.json`: Multi-metric evaluation report generated by `evaluate_tool_calls.py`.
