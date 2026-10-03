"""
Asynchronous Ollama client implementation.

Communicates with local Ollama daemon over HTTP, formatting tool calling schemas
and returning structured LLMResponse and ToolProposal objects without executing tools.
"""

import logging
from types import TracebackType
from typing import Any, Self

import httpx

from src.llm.base import (
    LLMAPIError,
    LLMClient,
    LLMConnectionError,
    LLMTimeoutError,
)
from src.llm.models import ChatMessage, LLMResponse
from src.llm.parser import ToolProposalParser
from src.llm.tool_schema import ToolRegistry, ToolSchema

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Local LLM client using Ollama's REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: float = 30.0,
        registry: ToolRegistry | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: float = timeout
        self.registry: ToolRegistry | None = registry
        self._external_client: bool = http_client is not None
        self._http_client: httpx.AsyncClient = http_client or httpx.AsyncClient(
            timeout=timeout
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client if managed internally."""
        if not self._external_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def is_available(self) -> bool:
        """Check if the local Ollama daemon is reachable and responding."""
        try:
            resp = await self._http_client.get(
                f"{self.base_url}/api/version", timeout=3.0
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Execute a conversational completion with optional tool schemas.

        Args:
            messages: List of chat messages in conversational sequence.
            tools: Optional list of tool schemas for proposing calls.
            temperature: Sampling temperature (0.0 to 1.0).
            timeout: Specific timeout for this call.

        Returns:
            An LLMResponse containing conversational text and validated ToolProposal objects.

        Raises:
            LLMConnectionError: If Ollama daemon is not running.
            LLMTimeoutError: If the model generation times out.
            LLMAPIError: If Ollama returns an HTTP error status code.
        """
        request_timeout = timeout or self.timeout

        # 1. Format messages for Ollama API
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        # 2. Attach tools if provided or if registry is available
        target_tools = tools
        if target_tools is None and self.registry is not None:
            target_tools = self.registry.list_schemas()

        if target_tools:
            payload["tools"] = [t.to_ollama_tool() for t in target_tools]

        # 3. Dispatch HTTP request to Ollama
        endpoint = f"{self.base_url}/api/chat"
        logger.debug(
            "Dispatching request to Ollama at %s with %d messages, %d tools",
            endpoint,
            len(messages),
            len(target_tools or []),
        )

        try:
            response = await self._http_client.post(
                endpoint,
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as exc:
            logger.error("Failed to connect to Ollama at %s: %s", self.base_url, exc)
            raise LLMConnectionError(
                f"Ollama service unavailable at {self.base_url}. Is Ollama running?"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out after %s seconds", request_timeout)
            raise LLMTimeoutError(
                f"Ollama request timed out after {request_timeout}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama API error: status %d: %s", exc.response.status_code, exc)
            raise LLMAPIError(
                status_code=exc.response.status_code,
                message=exc.response.text,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during Ollama call: %s", exc)
            raise LLMAPIError(status_code=500, message=str(exc)) from exc

        # 4. Parse text and tool proposals
        active_registry = self.registry
        if active_registry is None and target_tools:
            # Build an ad-hoc registry from the supplied tools for argument validation
            active_registry = ToolRegistry()
            for t in target_tools:
                active_registry.register(t)

        clean_text, proposals = ToolProposalParser.parse_from_ollama_response(
            data, registry=active_registry
        )

        return LLMResponse(
            text=clean_text,
            tool_proposals=proposals,
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
        )
