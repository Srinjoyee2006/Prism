"""
Local Kokoro Text-to-Speech (TTS) implementation using the OpenAI-compatible Kokoro-FastAPI engine.

Provides zero-cost, local neural TTS with low latency streaming,
conforming to the provider-independent TTSProvider abstraction.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator

import httpx

from src.tts.base import (
    AudioChunk,
    TTSConnectionError,
    TTSError,
    TTSProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")


class KokoroTTS(TTSProvider):
    """Local Kokoro TTS client interfacing with Kokoro-FastAPI or any OpenAI-compatible TTS server.

    Attributes:
        base_url: Base endpoint URL (default: http://localhost:8880/v1).
        voice: Kokoro voice identifier (e.g. 'af_heart', 'af_bella', 'am_adam').
        model: Model identifier passed to speech API (default: 'kokoro').
        response_format: Streaming audio format, preferably 'pcm' (24kHz 16-bit linear PCM).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_KOKORO_BASE_URL,
        voice: str = DEFAULT_VOICE,
        model: str = "kokoro",
        response_format: str = "pcm",
        sample_rate: int = 24000,
        chunk_size_bytes: int = 4096,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self._sample_rate = sample_rate
        self.chunk_size_bytes = chunk_size_bytes
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

        self._active_stream: httpx.Response | None = None
        self._active_task: asyncio.Task | None = None
        self._cancel_flag = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize or return an async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout, verify=False)
            self._owns_client = True
        return self._client

    async def is_available(self) -> bool:
        """Check if local Kokoro-FastAPI service is accessible."""
        client = self._get_client()
        try:
            # Check models list or base health endpoint
            resp = await client.get(f"{self.base_url}/models", timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            pass

        # Fallback check on root or /health
        root_url = self.base_url.replace("/v1", "")
        try:
            resp = await client.get(f"{root_url}/health", timeout=2.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            return False

    def cancel(self) -> None:
        """Immediately cancel any in-flight synthesis stream."""
        self._cancel_flag = True
        if self._active_stream is not None:
            try:
                # Close the streaming response transport to terminate generation immediately
                asyncio.create_task(self._active_stream.aclose())
            except Exception as exc:  # noqa: BLE001
                logger.debug("KokoroTTS: Exception while closing active stream: %s", exc)

        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[AudioChunk]:
        """Stream linear PCM audio chunks from Kokoro-FastAPI.

        Args:
            text: Text to synthesize.

        Yields:
            AudioChunk instances containing raw 24kHz 16-bit PCM bytes.

        Raises:
            TTSConnectionError: If the server cannot be contacted.
            TTSError: If synthesis fails.
            asyncio.CancelledError: If cancelled mid-stream.
        """
        clean_text = text.strip()
        if not clean_text:
            return

        client = self._get_client()
        url = f"{self.base_url}/audio/speech"
        payload = {
            "model": self.model,
            "input": clean_text,
            "voice": self.voice,
            "response_format": self.response_format,
        }

        self._cancel_flag = False
        self._active_task = asyncio.current_task()

        try:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                self._active_stream = response

                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_msg = error_bytes.decode(errors="replace")
                    raise TTSError(
                        f"Kokoro-FastAPI speech synthesis failed [{response.status_code}]: {error_msg}"
                    )

                is_first_chunk = True

                async for raw_bytes in response.aiter_bytes(chunk_size=self.chunk_size_bytes):
                    if self._cancel_flag:
                        logger.debug("KokoroTTS: Synthesis stream cancelled by flag")
                        raise asyncio.CancelledError()

                    chunk_data = raw_bytes
                    # If response is WAV format, strip the 44-byte standard RIFF header on first chunk
                    if is_first_chunk:
                        is_first_chunk = False
                        if chunk_data.startswith(b"RIFF") and len(chunk_data) >= 44:
                            chunk_data = chunk_data[44:]

                    if not chunk_data:
                        continue

                    # Ensure chunk length is even for 16-bit PCM samples
                    if len(chunk_data) % 2 != 0:
                        chunk_data = chunk_data[:-1]

                    if not chunk_data:
                        continue

                    yield AudioChunk(
                        pcm_data=chunk_data,
                        sample_rate=self._sample_rate,
                        channels=1,
                        sample_width=2,
                        text_segment=clean_text,
                        is_terminal=False,
                    )

                # Yield terminal zero-length signal chunk
                yield AudioChunk(
                    pcm_data=b"",
                    sample_rate=self._sample_rate,
                    channels=1,
                    sample_width=2,
                    text_segment=clean_text,
                    is_terminal=True,
                )

        except httpx.ConnectError as err:
            raise TTSConnectionError(
                f"Cannot connect to Kokoro-FastAPI at {self.base_url}. "
                "Ensure local service is running (e.g. via docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest)."
            ) from err
        except asyncio.CancelledError:
            logger.info("KokoroTTS: Speech synthesis cancelled for '%s'", clean_text[:30])
            raise
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Unexpected error during Kokoro TTS synthesis: {exc}") from exc
        finally:
            self._active_stream = None
            self._active_task = None

    async def close(self) -> None:
        """Close underlying HTTP client session."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
