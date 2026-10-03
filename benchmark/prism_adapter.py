"""
Prism Benchmark Adapter for Full-Duplex-Bench v3 (FDB-v3).

Provides headless WebRTC streaming into local LiveKit rooms, zero-cost local
faster-whisper ASR with word timestamps, audio latency measurement, and
telemetry extraction conforming to the official FDB-v3 specification.
"""

import asyncio
import json
import logging
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from faster_whisper import WhisperModel
from livekit import api, rtc

logger = logging.getLogger("prism_adapter")

SAMPLE_WIDTH = 2
PUBLISH_SAMPLE_RATE = 48000
CAPTURE_SAMPLE_RATE = 24000
CHUNK_DURATION_MS = 20

# Search locations for telemetry log
DEFAULT_TELEMETRY_PATHS = [
    Path("/tmp/agent_tool_calls.log"),
    Path(__file__).resolve().parent / "agent_tool_calls.log",
]


def read_wav_pcm16(path: str | Path, target_rate: int = 48000) -> tuple[bytes, int, int]:
    """Read an audio file and return (pcm_bytes, sample_rate, channels) at target_rate mono PCM-16."""
    path_str = str(path)
    try:
        with wave.open(path_str, "rb") as wf:
            rate = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if ch == 1 and sw == 2 and rate == target_rate:
                return wf.readframes(wf.getnframes()), rate, 1
    except wave.Error:
        pass

    logger.info("Converting '%s' to %d Hz mono 16-bit PCM via ffmpeg...", path_str, target_rate)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            path_str,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_rate),
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout, target_rate, 1


def write_wav(path: str | Path, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> None:
    """Write raw PCM-16 bytes to a standard WAV file."""
    path_str = str(path)
    with wave.open(path_str, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


async def run_livekit_stream(
    input_wav: str | Path,
    output_wav: str | Path,
    room_name: str,
    url: str = "ws://127.0.0.1:7880",
    api_key: str = "devkey",
    api_secret: str = "secret",
    user_speech_end: float | None = None,
) -> tuple[str, float]:
    """Stream input_wav into a LiveKit room as a participant and capture the agent's output.

    Returns (room_name, stream_start_time_unix).
    """
    input_path = Path(input_wav)
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcm_data, src_rate, src_channels = read_wav_pcm16(input_path, PUBLISH_SAMPLE_RATE)
    total_samples = len(pcm_data) // SAMPLE_WIDTH
    duration_sec = total_samples / src_rate

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("fdb-benchmark-user")
        .with_name("FDB Benchmark User")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    target_samples = int(duration_sec * CAPTURE_SAMPLE_RATE)
    output_buf = np.zeros(target_samples, dtype=np.int16)
    write_pos = 0

    recording_started = asyncio.Event()
    recording_stop = asyncio.Event()

    room = rtc.Room()
    receiving_task: asyncio.Task | None = None
    agent_spoke = False
    last_agent_audio_time: float = 0.0

    stream_start_time: float = 0.0

    speech_frame_count = 0

    async def _receive_agent_audio(track: rtc.Track) -> None:
        nonlocal write_pos, agent_spoke, last_agent_audio_time, speech_frame_count
        stream = rtc.AudioStream(track, sample_rate=CAPTURE_SAMPLE_RATE, num_channels=1)
        await recording_started.wait()

        thresh_sec = (user_speech_end or 0.0) + 3.0

        try:
            while not recording_stop.is_set():
                try:
                    frame_event = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                    frame = frame_event.frame
                    samples = np.frombuffer(bytes(frame.data), dtype=np.int16)
                    n = len(samples)
                    cur_elapsed = (time.time() - stream_start_time) if stream_start_time > 0 else 0.0

                    if cur_elapsed >= thresh_sec and n > 0 and np.max(np.abs(samples)) > 2000:
                        speech_frame_count += 1
                        if speech_frame_count >= 6:
                            agent_spoke = True
                            last_agent_audio_time = time.time()
                    elif agent_spoke and n > 0 and np.max(np.abs(samples)) > 1200:
                        last_agent_audio_time = time.time()

                    remaining = target_samples - write_pos
                    if remaining <= 0:
                        break
                    to_write = min(n, remaining)
                    output_buf[write_pos : write_pos + to_write] = samples[:to_write]
                    write_pos += to_write
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent audio receive note: %s", exc)
        finally:
            await stream.aclose()

    agent_ready = asyncio.Event()

    def _check_existing_tracks() -> None:
        nonlocal receiving_task
        for p in room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    if pub.track and receiving_task is None:
                        receiving_task = asyncio.create_task(_receive_agent_audio(pub.track))
                    agent_ready.set()
                    return

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            nonlocal receiving_task
            if receiving_task is None:
                receiving_task = asyncio.create_task(_receive_agent_audio(track))
            agent_ready.set()

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            agent_ready.set()

    logger.info("Connecting to room '%s' at %s...", room_name, url)
    await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=True))

    # Check if agent was already connected and published its track
    _check_existing_tracks()

    # Publish audio track from WAV
    source = rtc.AudioSource(src_rate, src_channels)
    local_track = rtc.LocalAudioTrack.create_audio_track("wav-input", source)
    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(local_track, pub_opts)

    # Wait for the agent to connect and publish its audio track so it doesn't miss speech
    if not agent_ready.is_set():
        print(f"  [WAIT] Waiting for agent track in room '{room_name}'...")
        try:
            await asyncio.wait_for(agent_ready.wait(), timeout=10.0)
            print("  [AGENT] Agent connected and ready in room. Beginning audio stream.")
        except asyncio.TimeoutError:
            print("  [WARN] Agent track not detected within 10s; beginning audio stream anyway.")
    else:
        print(f"  [AGENT] Agent already ready in room '{room_name}'. Beginning audio stream.")

    await asyncio.sleep(0.2)

    samples_per_chunk = src_rate * CHUNK_DURATION_MS // 1000
    chunk_bytes = samples_per_chunk * src_channels * SAMPLE_WIDTH

    stream_start_time = time.time()
    recording_started.set()

    offset = 0
    silence_chunk = b"\x00" * chunk_bytes
    # If user speech end is known, require at least user_speech_end + 4.5s before early completion
    min_stream_sec = (user_speech_end + 4.5) if user_speech_end is not None else duration_sec
    # Ceiling to prevent lingering if agent produces no response
    max_stream_sec = min(duration_sec, (user_speech_end + 15.0) if user_speech_end is not None else duration_sec)

    while True:
        elapsed = time.time() - stream_start_time
        if elapsed >= duration_sec:
            break

        # Check for early completion:
        # All human speech has been streamed, agent has responded and finished speaking (>2.0s post-response silence)
        if elapsed >= min_stream_sec and agent_spoke and last_agent_audio_time > 0:
            if time.time() - last_agent_audio_time > 2.0:
                logger.info("Agent response complete (spoken & 2.0s silence). Ending exchange at %.2fs.", elapsed)
                break

        if elapsed >= max_stream_sec:
            logger.info("Reached maximum exchange ceiling (%.2fs). Ending exchange.", max_stream_sec)
            break

        if offset < len(pcm_data):
            end = min(offset + chunk_bytes, len(pcm_data))
            chunk = pcm_data[offset:end]
            if len(chunk) < chunk_bytes:
                chunk += b"\x00" * (chunk_bytes - len(chunk))
            offset = end
        else:
            chunk = silence_chunk

        num_samples = len(chunk) // (SAMPLE_WIDTH * src_channels)
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=src_rate,
            num_channels=src_channels,
            samples_per_channel=num_samples,
        )
        await source.capture_frame(frame)
        await asyncio.sleep(CHUNK_DURATION_MS / 1000)

    recording_stop.set()

    if receiving_task and not receiving_task.done():
        await asyncio.sleep(0.1)
        if not receiving_task.done():
            receiving_task.cancel()
            try:
                await receiving_task
            except asyncio.CancelledError:
                pass

    if write_pos > 0:
        out_bytes = output_buf[:write_pos].tobytes()
        captured_duration = write_pos / CAPTURE_SAMPLE_RATE
    else:
        out_bytes = output_buf.tobytes()
        captured_duration = duration_sec

    write_wav(output_path, out_bytes, CAPTURE_SAMPLE_RATE, 1)
    logger.info("Saved agent output audio: %s (%.2fs)", output_path, captured_duration)

    await room.disconnect()
    async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
        try:
            await lk_api.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Room cleanup note: %s", exc)

    return room_name, stream_start_time


# ---------------------------------------------------------------------------
# Zero-Cost Local ASR (faster-whisper)
# ---------------------------------------------------------------------------

_ASR_MODEL: WhisperModel | None = None


def get_asr_model(model_size: str = "tiny.en") -> WhisperModel:
    global _ASR_MODEL
    if _ASR_MODEL is None:
        logger.info("Loading local faster-whisper model '%s' for benchmark evaluation...", model_size)
        _ASR_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _ASR_MODEL


def run_local_asr(audio_path: str | Path, model: WhisperModel | None = None) -> dict[str, Any]:
    """Run local zero-cost ASR on an audio file and return word-level chunks.

    Matches the format expected by FDB-v3 official evaluation:
    {"text": str, "chunks": [{"text": str, "timestamp": [start, end]}]}
    """
    model = model or get_asr_model()
    try:
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        chunks: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for seg in segments:
            text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    chunks.append({
                        "text": w.word.strip(),
                        "timestamp": [round(w.start, 2), round(w.end, 2)],
                    })

        full_text = " ".join(text_parts).strip()
        return {"text": full_text, "chunks": chunks}
    except Exception as exc:  # noqa: BLE001
        logger.error("Local ASR error on %s: %s", audio_path, exc)
        return {"text": "", "chunks": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Audio Latency Measurement (RMS / dBFS via NumPy, zero external deps)
# ---------------------------------------------------------------------------


def measure_audio_latency(
    input_path: str | Path,
    output_path: str | Path,
    silence_threshold_db: float = -40.0,
) -> dict[str, float | str]:
    """Measure response latency from end of input audio to first non-silence in output audio."""
    try:
        with wave.open(str(input_path), "rb") as win:
            in_rate = win.getframerate()
            in_frames = win.getnframes()
            input_duration_s = in_frames / in_rate

        with wave.open(str(output_path), "rb") as wout:
            out_rate = wout.getframerate()
            out_frames = wout.getnframes()
            output_duration_s = out_frames / out_rate
            raw_output = wout.readframes(out_frames)

        samples = np.frombuffer(raw_output, dtype=np.int16)
        chunk_samples = int(out_rate * (50 / 1000.0))  # 50ms chunk
        first_speech_ms: float | None = None

        for idx in range(0, len(samples), chunk_samples):
            chunk = samples[idx : idx + chunk_samples]
            if len(chunk) == 0:
                continue
            rms = math.sqrt(float(np.mean(chunk.astype(np.float64) ** 2)))
            dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else -100.0
            if dbfs > silence_threshold_db:
                first_speech_ms = (idx / out_rate) * 1000.0
                break

        first_speech_s = (first_speech_ms / 1000.0) if first_speech_ms is not None else output_duration_s

        return {
            "input_duration_s": round(input_duration_s, 3),
            "output_duration_s": round(output_duration_s, 3),
            "first_speech_s": round(first_speech_s, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Telemetry Extraction
# ---------------------------------------------------------------------------


def extract_tool_telemetry(
    room_name: str,
    stream_start_time: float,
    log_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Extract tool calls for room_name and compute stream-relative timestamps."""
    paths = log_paths or DEFAULT_TELEMETRY_PATHS
    actual_tool_calls: list[dict[str, Any]] = []
    seen_lines: set[str] = set()
    seen_calls: set[tuple[str, float]] = set()

    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line in seen_lines:
                        continue
                    seen_lines.add(line)
                    try:
                        t_data = json.loads(line)
                        if t_data.get("room") == room_name:
                            call_data = t_data.get("call", {})
                            fn_name = str(call_data.get("function", ""))
                            t_start_raw = float(call_data.get("timestamp_start", 0.0))
                            call_key = (fn_name, round(t_start_raw, 4))
                            if call_key in seen_calls:
                                continue
                            seen_calls.add(call_key)

                            rel_call = dict(call_data)
                            if stream_start_time:
                                if "timestamp_start" in rel_call:
                                    rel_call["timestamp_start"] = round(rel_call["timestamp_start"] - stream_start_time, 2)
                                if "timestamp_end" in rel_call:
                                    rel_call["timestamp_end"] = round(rel_call["timestamp_end"] - stream_start_time, 2)
                                if "timestamp" in rel_call:
                                    rel_call["timestamp"] = round(rel_call["timestamp"] - stream_start_time, 2)
                            actual_tool_calls.append(rel_call)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error reading telemetry file %s: %s", path, exc)

    return actual_tool_calls
