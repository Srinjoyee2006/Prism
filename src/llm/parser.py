"""
Parser for model-generated tool proposals and conversational text.

Extracts structured tool calls from both native Ollama/OpenAI tool_calls responses
and fallback text-embedded tool call representations, validating arguments against
the tool schema registry.
"""

import json
import logging
import re
from typing import Any

from src.llm.models import ToolProposal
from src.llm.tool_schema import ToolRegistry

logger = logging.getLogger(__name__)

# Patterns for models that emit tool calls in content text
_JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)
_TOOL_TAG_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_BARE_TOOL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"([a-zA-Z0-9_]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


class ToolProposalParser:
    """Extracts, parses, and validates tool proposals from raw LLM output."""

    @classmethod
    def parse_from_ollama_response(
        cls,
        response_data: dict[str, Any],
        registry: ToolRegistry | None = None,
    ) -> tuple[str, list[ToolProposal]]:
        """Parse text and tool proposals from an Ollama /api/chat response payload.

        Args:
            response_data: JSON dictionary returned by Ollama's chat endpoint.
            registry: Optional ToolRegistry to validate proposals against.

        Returns:
            Tuple of (conversational_text, list_of_tool_proposals).
        """
        message = response_data.get("message", {})
        raw_content = message.get("content", "").strip()
        raw_tool_calls = message.get("tool_calls", [])

        proposals: list[ToolProposal] = []

        # 1. First Priority: Parse native tool_calls structure
        if raw_tool_calls and isinstance(raw_tool_calls, list):
            for tc in raw_tool_calls:
                proposal = cls._parse_native_tool_call(tc, registry)
                if proposal:
                    proposals.append(proposal)

            # If native tool calls exist, strip any redundant echo from conversational text
            clean_text = cls._clean_text_echo(raw_content)
            return clean_text, proposals

        # 2. Second Priority: Fallback to text-embedded tool calls
        embedded_text, embedded_proposals = cls._parse_embedded_tool_calls(
            raw_content, registry
        )
        if embedded_proposals:
            return embedded_text, embedded_proposals

        # 3. Pure conversational response
        return raw_content, []

    @classmethod
    def _parse_native_tool_call(
        cls,
        tool_call_data: dict[str, Any],
        registry: ToolRegistry | None = None,
    ) -> ToolProposal | None:
        """Parse a single tool call from Ollama's native tool_calls array."""
        call_id = tool_call_data.get("id") or ""
        fn_data = tool_call_data.get("function", {})
        tool_name = fn_data.get("name", "").strip()

        if not tool_name:
            return None

        raw_args = fn_data.get("arguments", {})

        # Parse string arguments if model emitted raw JSON string
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                proposal = ToolProposal(
                    tool_name=tool_name,
                    arguments={},
                    raw_arguments=raw_args,
                    call_id=call_id or f"call_{tool_name}",
                    is_valid=False,
                    validation_error=f"Malformed JSON arguments: {exc}",
                )
                return proposal
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}

        # Validate against registry if supplied
        if registry is not None:
            is_valid, error, validated_args = registry.validate_proposal(
                tool_name, parsed_args
            )
            return ToolProposal(
                tool_name=tool_name,
                arguments=validated_args if is_valid else parsed_args,
                raw_arguments=json.dumps(parsed_args),
                call_id=call_id or f"call_{tool_name}",
                is_valid=is_valid,
                validation_error=error,
            )

        return ToolProposal(
            tool_name=tool_name,
            arguments=parsed_args,
            raw_arguments=json.dumps(parsed_args),
            call_id=call_id or f"call_{tool_name}",
            is_valid=True,
        )

    @classmethod
    def _parse_embedded_tool_calls(
        cls,
        content: str,
        registry: ToolRegistry | None = None,
    ) -> tuple[str, list[ToolProposal]]:
        """Fallback parser: inspects response text for markdown or tag-wrapped tool calls."""
        proposals: list[ToolProposal] = []
        cleaned_text = content

        # Check for <tool_call> tags
        for match in _TOOL_TAG_PATTERN.finditer(content):
            snippet = match.group(1)
            proposal = cls._parse_json_snippet(snippet, registry)
            if proposal:
                proposals.append(proposal)
                cleaned_text = cleaned_text.replace(match.group(0), "")

        if proposals:
            return cleaned_text.strip(), proposals

        # Check for ```json blocks containing tool schemas
        for match in _JSON_CODE_BLOCK_PATTERN.finditer(content):
            snippet = match.group(1)
            proposal = cls._parse_json_snippet(snippet, registry)
            if proposal:
                proposals.append(proposal)
                cleaned_text = cleaned_text.replace(match.group(0), "")

        if proposals:
            return cleaned_text.strip(), proposals

        # Check for bare {"name": ..., "arguments": ...} in text
        for match in _BARE_TOOL_PATTERN.finditer(content):
            name = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args)
                is_valid = True
                error = None
                if registry:
                    is_valid, error, args = registry.validate_proposal(name, args)
                proposals.append(
                    ToolProposal(
                        tool_name=name,
                        arguments=args,
                        raw_arguments=raw_args,
                        is_valid=is_valid,
                        validation_error=error,
                    )
                )
                cleaned_text = cleaned_text.replace(match.group(0), "")
            except json.JSONDecodeError:
                proposals.append(
                    ToolProposal(
                        tool_name=name,
                        arguments={},
                        raw_arguments=raw_args,
                        is_valid=False,
                        validation_error="Malformed JSON arguments in bare tool call",
                    )
                )

        return cleaned_text.strip(), proposals

    @classmethod
    def _parse_json_snippet(
        cls, snippet: str, registry: ToolRegistry | None = None
    ) -> ToolProposal | None:
        """Helper to parse a single JSON tool call snippet."""
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError as exc:
            return ToolProposal(
                tool_name="unknown",
                arguments={},
                raw_arguments=snippet,
                is_valid=False,
                validation_error=f"Malformed JSON snippet: {exc}",
            )

        if not isinstance(data, dict):
            return None

        tool_name = data.get("name") or data.get("tool_name")
        if not tool_name or not isinstance(tool_name, str):
            return None

        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            return ToolProposal(
                tool_name=tool_name,
                arguments={},
                raw_arguments=str(arguments),
                is_valid=False,
                validation_error="Tool arguments must be a JSON object",
            )

        if registry:
            is_valid, error, val_args = registry.validate_proposal(tool_name, arguments)
            return ToolProposal(
                tool_name=tool_name,
                arguments=val_args if is_valid else arguments,
                raw_arguments=snippet,
                is_valid=is_valid,
                validation_error=error,
            )

        return ToolProposal(
            tool_name=tool_name,
            arguments=arguments,
            raw_arguments=snippet,
            is_valid=True,
        )

    @staticmethod
    def _clean_text_echo(content: str) -> str:
        """Strip raw JSON tool call echos from conversational text."""
        # Remove embedded tool call json blocks if model also output conversational framing
        cleaned = _JSON_CODE_BLOCK_PATTERN.sub("", content)
        cleaned = _TOOL_TAG_PATTERN.sub("", cleaned)
        cleaned = _BARE_TOOL_PATTERN.sub("", cleaned)
        return cleaned.strip()
