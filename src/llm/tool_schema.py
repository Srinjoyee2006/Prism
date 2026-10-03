"""
Tool schema definitions and argument validation system.

Defines schemas conforming to standard JSON-schema / OpenAI / Ollama tool-calling
formats and performs deterministic type and requirement validation on model proposals.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Definition of a single parameter in a tool's schema."""

    name: str
    param_type: str  # 'string', 'integer', 'number', 'boolean', 'array', 'object'
    description: str = ""
    required: bool = True
    enum: list[Any] | None = None
    default: Any | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameter to JSON schema property representation."""
        schema: dict[str, Any] = {
            "type": self.param_type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolSchema:
    """Full schema definition for a tool/function available to the LLM."""

    name: str
    description: str
    parameters: dict[str, ToolParameter] = field(default_factory=dict)

    def add_parameter(self, param: ToolParameter) -> "ToolSchema":
        """Add a parameter to this schema (builder pattern)."""
        self.parameters[param.name] = param
        return self

    @property
    def required_parameters(self) -> list[str]:
        """Return list of parameter names that are marked required."""
        return [name for name, p in self.parameters.items() if p.required]

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to the standard tool definition format accepted by Ollama / OpenAI."""
        properties = {
            name: param.to_json_schema() for name, param in self.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_parameters,
                },
            },
        }

    def validate_arguments(
        self, arguments: dict[str, Any]
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate provided arguments against this tool's schema.

        Returns:
            Tuple of (is_valid, error_message, validated_arguments).
        """
        if not isinstance(arguments, dict):
            return False, f"Arguments must be a dictionary, got {type(arguments).__name__}", {}

        # 1. Check for missing required parameters
        for req in self.required_parameters:
            if req not in arguments:
                return False, f"Missing required parameter: '{req}'", arguments

        validated: dict[str, Any] = {}

        # 2. Validate types and enums for present parameters
        for key, value in arguments.items():
            param = self.parameters.get(key)
            if param is None:
                # Allow unspecified extra parameters, but preserve them
                validated[key] = value
                continue

            # Check None values for required fields
            if value is None:
                if param.required:
                    return False, f"Required parameter '{key}' cannot be null", arguments
                validated[key] = None
                continue

            # Type checking
            is_valid_type, type_error = self._check_type(param.param_type, value)
            if not is_valid_type:
                return (
                    False,
                    f"Parameter '{key}' expected type '{param.param_type}', got {type(value).__name__} ({type_error})",
                    arguments,
                )

            # Enum checking
            if param.enum and value not in param.enum:
                return (
                    False,
                    f"Parameter '{key}' value {value!r} is not in allowed choices: {param.enum}",
                    arguments,
                )

            validated[key] = value

        return True, None, validated

    @staticmethod
    def _check_type(expected_type: str, value: Any) -> tuple[bool, str]:
        """Verify that a value matches the declared JSON Schema type."""
        # Note: In Python, bool is a subclass of int, so we must explicitly check bool first
        if expected_type == "boolean":
            return isinstance(value, bool), ""
        if expected_type == "string":
            return isinstance(value, str), ""
        if expected_type == "integer":
            if isinstance(value, bool):
                return False, "boolean is not an integer"
            return isinstance(value, int), ""
        if expected_type == "number":
            if isinstance(value, bool):
                return False, "boolean is not a number"
            return isinstance(value, (int, float)), ""
        if expected_type == "array":
            return isinstance(value, list), ""
        if expected_type == "object":
            return isinstance(value, dict), ""
        return True, ""


class ToolRegistry:
    """Registry maintaining available tool schemas and providing validation services."""

    def __init__(self):
        self._schemas: dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema) -> None:
        """Register a tool schema."""
        self._schemas[schema.name] = schema

    def get(self, name: str) -> ToolSchema | None:
        """Retrieve a tool schema by name."""
        return self._schemas.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        """Return all registered schemas."""
        return list(self._schemas.values())

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        """Export all registered schemas in Ollama tool specification format."""
        return [schema.to_ollama_tool() for schema in self._schemas.values()]

    def validate_proposal(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate a tool proposal against the registry.

        Returns:
            Tuple of (is_valid, error_message, validated_arguments).
        """
        schema = self.get(tool_name)
        if schema is None:
            return (
                False,
                f"Unknown tool: '{tool_name}'. Available tools: {list(self._schemas.keys())}",
                arguments,
            )

        return schema.validate_arguments(arguments)
