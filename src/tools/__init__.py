"""
Local Tool Registry:
Mock and utility functions used for testing tool execution policies and commit gates.
"""

from collections.abc import Callable
from typing import Any

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}
