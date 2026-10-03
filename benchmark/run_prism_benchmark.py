"""
Unified Prism Benchmark Runner for Full-Duplex-Bench v3 (FDB-v3).

Executes benchmark scenarios through local LiveKit rooms against the Prism agent,
measures audio latencies, extracts tool-call telemetry, saves result_prism_local.json,
and evaluates task completion and tool accuracy using official FDB-v3 evaluation logic
without paid APIs or cloud dependencies.

Usage:
    # 1. Single example smoke test:
    python benchmark/run_prism_benchmark.py --smoke-test

    # 2. Run specific example:
    python benchmark/run_prism_benchmark.py --example ecommerce_01

    # 3. Evaluate results with official non-LLM judge:
    python benchmark/run_prism_benchmark.py --smoke-test --evaluate

    # 4. Full 100-example batch run:
    python benchmark/run_prism_benchmark.py --all --evaluate
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from benchmark.prism_adapter import (
    extract_tool_telemetry,
    measure_audio_latency,
    run_livekit_stream,
    run_local_asr,
)

logger = logging.getLogger("run_prism_benchmark")

# Directory configurations
DATA_DIR = PROJECT_ROOT / "benchmark" / "data" / "fdb_v3_data_released"
SCENARIOS_JSON = PROJECT_ROOT / "benchmark" / "official_v3" / "benchmark_data_v2.json"
OFFICIAL_EVAL_DIR = PROJECT_ROOT / "benchmark" / "official_v3"

_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def load_scenarios() -> dict[str, Any]:
    """Load scenario metadata from benchmark_data_v2.json."""
    if not SCENARIOS_JSON.exists():
        return {}
    with open(SCENARIOS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            data = data["scenarios"]
        return {item["id"]: item for item in data}


def discover_inputs(root_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover all input.wav folders in the released flat layout."""
    inputs = []
    if not root_dir.exists():
        return inputs

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        m = _FOLDER_RE.match(folder.name)
        if not m:
            continue
        example_id = m.group(1)
        speaker_id = m.group(2)
        input_path = folder / "input.wav"
        if input_path.exists():
            inputs.append((speaker_id, example_id, input_path))
    return inputs


def process_example(
    speaker_id: str,
    example_id: str,
    input_path: Path,
    provider: str,
    scenarios: dict[str, Any],
    force: bool = False,
) -> dict[str, Any] | None:
    """Process a single benchmark example end-to-end."""
    example_dir = input_path.parent
    output_path = example_dir / f"output_{provider}.wav"
    result_path = example_dir / f"result_{provider}.json"

    if result_path.exists() and not force:
        logger.info("Example '%s' already evaluated. Skipping (use --force to re-run).", example_id)
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Resolve scenario metadata
    item = scenarios.get(example_id)
    if item is None:
        meta_path = example_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                item = json.load(f)
        else:
            item = {"title": example_id, "domain": "unknown", "expected_tool_calls": []}

    category = item.get("domain", item.get("category", "unknown"))
    title = item.get("title", example_id)

    print(f"\n{'='*70}")
    print(f"▶ Processing Scenario: [{example_id}] {title}")
    print(f"  Domain: {category}")
    print(f"  Input:  {input_path}")
    print(f"{'='*70}")

    room_name = f"fdb-{example_id}-{uuid.uuid4().hex[:6]}"
    inference_start = time.time()

    # Step 1: Pre-compute ASR on input audio to detect user speech boundary
    print("  🗣️  Running local ASR on input...")
    input_asr = run_local_asr(input_path)
    input_chunks = input_asr.get("chunks", [])

    user_speech_end_rel = 0.0
    if input_chunks:
        # Find end of user turn (split if gap > 2s, else end of speech)
        for i in range(len(input_chunks) - 1):
            curr_end = input_chunks[i]["timestamp"][1]
            next_start = input_chunks[i + 1]["timestamp"][0]
            if next_start - curr_end > 2.0:
                user_speech_end_rel = curr_end
                break
        else:
            user_speech_end_rel = input_chunks[-1]["timestamp"][1]

    print(f"  📝 User Speech End: {user_speech_end_rel:.2f}s | Input Text: \"{input_asr.get('text', '')[:70]}...\"")

    # Step 2: Stream input.wav into LiveKit room and capture output.wav
    print(f"  🎙️  Streaming input into room '{room_name}'...")
    try:
        _, stream_start_time = asyncio.run(
            run_livekit_stream(
                input_wav=input_path,
                output_wav=output_path,
                room_name=room_name,
                user_speech_end=user_speech_end_rel,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ Streaming inference error: {exc}")
        return {
            "example_id": example_id,
            "status": "inference_error",
            "error": str(exc),
        }

    inference_duration = time.time() - inference_start
    print(f"  ✅ LiveKit audio exchange completed in {inference_duration:.2f}s")

    # Step 3: Measure Audio Latency
    latency_info = measure_audio_latency(input_path, output_path)

    # Step 4: ASR on Agent Output Audio
    print("  🤖 Running local ASR on agent response...")
    agent_asr = run_local_asr(output_path)
    agent_chunks = agent_asr.get("chunks", [])
    agent_text = agent_asr.get("text", "")
    print(f"  🔊 Agent Spoke: \"{agent_text[:80]}\"")

    audio_agent_speech_start = agent_chunks[0]["timestamp"][0] if agent_chunks else 0.0
    perceived_total_latency = (
        round(audio_agent_speech_start - user_speech_end_rel, 3) if agent_chunks and user_speech_end_rel else 0.0
    )

    # Step 5: Extract Actual Tool Telemetry
    actual_tool_calls = extract_tool_telemetry(room_name, stream_start_time)
    print(f"  ⚙️  Actual Committed Tool Calls: {len(actual_tool_calls)}")
    for call in actual_tool_calls:
        print(f"     -> {call.get('function')} (args: {call.get('args')}) at {call.get('timestamp_start')}s")

    # Step 6: Construct Final Result Object
    result = {
        "pid": speaker_id,
        "example_id": example_id,
        "category": category,
        "title": title,
        "provider": provider,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inference_time_s": round(inference_duration, 2),
        "room_name": room_name,
        "stream_start_time": stream_start_time,
        "input_transcript": input_asr.get("text", ""),
        "input_asr_chunks": input_chunks,
        "user_speech_end_rel": round(user_speech_end_rel, 3),
        "latency": latency_info,
        "transcript": agent_text,
        "asr_chunks": agent_chunks,
        "audio_agent_speech_start": round(audio_agent_speech_start, 3),
        "perceived_total_latency": perceived_total_latency,
        "actual_tool_calls": actual_tool_calls,
        "status": "completed",
    }

    # Step 7: Save result JSON in example directory
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved Benchmark Result: {result_path}")
    return result


def run_evaluation(provider: str, root_dir: Path) -> None:
    """Run official evaluation scripts (evaluate_pass_rate.py & evaluate_tool_calls.py) strictly without LLM."""
    print(f"\n{'='*70}")
    print("📊 RUNNING OFFICIAL FDB-v3 EVALUATION (Rule-based, No Paid APIs)")
    print(f"{'='*70}")

    pass_rate_script = OFFICIAL_EVAL_DIR / "evaluate_pass_rate.py"
    tool_calls_script = OFFICIAL_EVAL_DIR / "evaluate_tool_calls.py"
    pass_report_path = PROJECT_ROOT / f"{provider}_pass_rate_report.json"
    tool_report_path = PROJECT_ROOT / f"{provider}_evaluation_report.json"

    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # 1. evaluate_pass_rate.py
    if pass_rate_script.exists():
        print("▶ Running evaluate_pass_rate.py...")
        cmd = [
            sys.executable,
            str(pass_rate_script),
            "--benchmark",
            str(SCENARIOS_JSON),
            "--results-dir",
            str(root_dir),
            "--provider",
            provider,
            "--output",
            str(pass_report_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=sub_env, check=False)
        print(res.stdout)
        if res.stderr:
            print(f"  Notice: {res.stderr[:200]}")

    # 2. evaluate_tool_calls.py
    if tool_calls_script.exists():
        print("▶ Running evaluate_tool_calls.py...")
        cmd = [
            sys.executable,
            str(tool_calls_script),
            "--benchmark",
            str(SCENARIOS_JSON),
            "--results-dir",
            str(root_dir),
            "--provider",
            provider,
            "--output",
            str(tool_report_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=sub_env, check=False)
        print(res.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prism Full-Duplex-Bench v3 Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run exactly one official benchmark example")
    parser.add_argument("--example", type=str, default=None, help="Process a specific example ID (e.g. ecommerce_01)")
    parser.add_argument("--all", action="store_true", help="Process all available examples in the dataset")
    parser.add_argument("--provider", type=str, default="prism_local", help="Provider identifier (default: prism_local)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Root directory containing released examples")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--evaluate", action="store_true", help="Run official pass-rate and tool-call evaluations")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    scenarios = load_scenarios()
    inputs = discover_inputs(data_dir)

    if not inputs:
        print(f"❌ No input.wav files found in {data_dir}. Run benchmark/download_fdb_data.py first.")
        sys.exit(1)

    print("🚀 FDB-v3 Prism Runner Started")
    print(f"   Provider: {args.provider}")
    print(f"   Total Discovered Examples: {len(inputs)}")

    # Filter target inputs
    if args.smoke_test:
        target_inputs = [inputs[0]]
        print(f"   Mode: Smoke Test (single example: {target_inputs[0][1]})")
    elif args.example:
        target_inputs = [(s, e, p) for s, e, p in inputs if e == args.example]
        if not target_inputs:
            print(f"❌ Example '{args.example}' not found in dataset.")
            sys.exit(1)
        print(f"   Mode: Single Example ({args.example})")
    elif args.all:
        target_inputs = inputs
        print(f"   Mode: Full Benchmark Run ({len(target_inputs)} examples)")
    else:
        print("Please specify --smoke-test, --example <id>, or --all. (Defaulting to --smoke-test)")
        target_inputs = [inputs[0]]

    success_count = 0
    for speaker_id, example_id, input_path in target_inputs:
        result = process_example(
            speaker_id=speaker_id,
            example_id=example_id,
            input_path=input_path,
            provider=args.provider,
            scenarios=scenarios,
            force=args.force,
        )
        if result and result.get("status") == "completed":
            success_count += 1

    print(f"\n🏁 Finished processing {len(target_inputs)} examples ({success_count} succeeded).")

    if args.evaluate or args.smoke_test:
        run_evaluation(provider=args.provider, root_dir=data_dir)


if __name__ == "__main__":
    main()
