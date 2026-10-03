"""
Text-to-Speech (TTS) subsystem:
Synthesizes speech audio streams with low latency and immediate cancellation support.
"""

from typing import Protocol


class TTSEngine(Protocol):
    """Protocol for streaming text-to-speech synthesis with interruption support."""
