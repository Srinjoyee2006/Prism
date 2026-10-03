"""
Unit tests for Text-to-Speech (TTS) subsystem (Stage 6).

Verifies requirements:
A. Text is passed to TTS.
B. TTS produces multiple audio chunks.
G. TTS generation is cancelled after interruption.
Plus Kokoro client request format, availability, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.tts.base import (
    AudioChunk,
    TTSConnectionError,
)
from src.tts.kokoro import KokoroTTS
from src.tts.mock import FakeTTS


@pytest.mark.asyncio
async def test_requirement_a_text_is_passed_to_tts():
    """Requirement A: Text is passed to TTS engine and recorded."""
    tts = FakeTTS(chunk_count=2, delay_per_chunk=0.001)
    text = "Hello Prism voice agent"

    chunks = []
    async for chunk in tts.synthesize_stream(text):
        chunks.append(chunk)

    assert len(tts.synthesized_texts) == 1
    assert tts.synthesized_texts[0] == text
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_requirement_b_tts_produces_multiple_audio_chunks():
    """Requirement B: TTS produces multiple audio chunks with proper PCM metadata."""
    chunk_count = 5
    sample_rate = 24000
    tts = FakeTTS(
        sample_rate=sample_rate,
        chunk_count=chunk_count,
        chunk_duration_seconds=0.02,
        delay_per_chunk=0.001,
        generate_pcm=True,
    )

    chunks: list[AudioChunk] = []
    async for chunk in tts.synthesize_stream("Testing multi-chunk synthesis"):
        chunks.append(chunk)

    assert len(chunks) == chunk_count
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, AudioChunk)
        assert len(chunk.pcm_data) > 0
        assert chunk.sample_rate == sample_rate
        assert chunk.channels == 1
        assert chunk.sample_width == 2
        assert chunk.sample_count > 0
        assert chunk.duration_seconds > 0.0
        if i == chunk_count - 1:
            assert chunk.is_terminal is True
        else:
            assert chunk.is_terminal is False


@pytest.mark.asyncio
async def test_requirement_g_tts_generation_is_cancelled_after_interruption():
    """Requirement G: TTS generation is cleanly cancelled mid-stream upon interruption."""
    tts = FakeTTS(
        chunk_count=10,
        delay_per_chunk=0.05,  # 50ms per chunk gives ample time to cancel
    )

    received_chunks: list[AudioChunk] = []

    async def consume_stream():
        async for chunk in tts.synthesize_stream("Long sentence to be interrupted"):
            received_chunks.append(chunk)

    task = asyncio.create_task(consume_stream())

    # Wait for at least 1 chunk to be yielded
    await asyncio.sleep(0.06)

    # Interrupt / cancel generation
    tts.cancel()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Synthesis must have halted early
    assert len(received_chunks) < 10
    assert tts.cancelled_count >= 1


def test_audio_chunk_duration_and_samples():
    """Verify AudioChunk duration and sample math."""
    # 24000 samples at 16-bit mono = 48000 bytes = 1.0 second
    data = b"\x00" * 48000
    chunk = AudioChunk(pcm_data=data, sample_rate=24000, channels=1, sample_width=2)
    assert chunk.sample_count == 24000
    assert pytest.approx(chunk.duration_seconds, 0.001) == 1.0
    assert pytest.approx(chunk.duration_ms, 0.1) == 1000.0


@pytest.mark.asyncio
async def test_kokoro_tts_availability_success():
    """Verify KokoroTTS.is_available() returns True when server responds 200."""
    tts = KokoroTTS(base_url="http://localhost:8880/v1")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(status_code=200)
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False

    tts._client = mock_client
    is_ready = await tts.is_available()
    assert is_ready is True
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_kokoro_tts_availability_failure():
    """Verify KokoroTTS.is_available() returns False when connection fails."""
    tts = KokoroTTS(base_url="http://localhost:8880/v1")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_client.is_closed = False

    tts._client = mock_client
    is_ready = await tts.is_available()
    assert is_ready is False


@pytest.mark.asyncio
async def test_kokoro_tts_synthesize_stream_pcm_flow():
    """Verify KokoroTTS constructs correct request and streams PCM chunks."""
    tts = KokoroTTS(
        base_url="http://localhost:8880/v1",
        voice="af_heart",
        model="kokoro",
        response_format="pcm",
        sample_rate=24000,
    )

    # 4800 bytes = 2400 samples = 0.1s at 24kHz
    mock_pcm = b"\x01\x00" * 2400

    async def mock_aiter_bytes(chunk_size=4096):
        yield mock_pcm[:2400]
        yield mock_pcm[2400:]

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_bytes = mock_aiter_bytes

    class MockStreamContext:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockStreamContext())
    mock_client.is_closed = False
    tts._client = mock_client

    chunks: list[AudioChunk] = []
    async for chunk in tts.synthesize_stream("Hello from Kokoro"):
        chunks.append(chunk)

    mock_client.stream.assert_called_once_with(
        "POST",
        "http://localhost:8880/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello from Kokoro",
            "voice": "af_heart",
            "response_format": "pcm",
        },
        headers={"Content-Type": "application/json"},
    )

    # Last chunk is terminal signal
    assert len(chunks) == 3
    assert chunks[0].pcm_data == mock_pcm[:2400]
    assert chunks[1].pcm_data == mock_pcm[2400:]
    assert chunks[2].is_terminal is True


@pytest.mark.asyncio
async def test_kokoro_tts_connection_error_raises_friendly_exception():
    """Verify TTSConnectionError provides clear actionable troubleshooting."""
    tts = KokoroTTS(base_url="http://localhost:8880/v1")

    class FailingStreamContext:
        async def __aenter__(self):
            raise httpx.ConnectError("Failed to connect")

        async def __aexit__(self, *args):
            pass

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=FailingStreamContext())
    mock_client.is_closed = False
    tts._client = mock_client

    with pytest.raises(TTSConnectionError) as exc_info:
        async for _ in tts.synthesize_stream("Will fail"):
            pass

    assert "Cannot connect to Kokoro-FastAPI" in str(exc_info.value)
