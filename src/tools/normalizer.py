"""
Generic argument normalizer for voice agents and benchmark evaluation.

Ensures spoken identifiers (like order IDs, flight IDs, document numbers)
with whitespace, commas, hyphens, or spoken number words are canonicalized
into deterministic alphanumeric strings, while preserving formatting,
punctuation, and types for non-identifier fields (e.g., addresses, queries).
"""

import re
from typing import Any

# Spoken number words mapping
WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_NUM_WORD_RE = re.compile(
    r"\b(" + "|".join(WORD_TO_DIGIT.keys()) + r")\b",
    flags=re.IGNORECASE,
)

# Known identifier parameter names across FDB-v3 and voice tool domains
IDENTIFIER_PARAMS = {
    "order_id",
    "flight_id",
    "doc_number",
    "product_id",
    "tracking_id",
    "tracking_number",
    "booking_ref",
}

CURRENCY_PARAMS = {
    "from_currency",
    "to_currency",
}


def normalize_identifier(val: str) -> str:
    """Normalize a spoken or punctuated identifier into a canonical alphanumeric string.

    Examples:
        'ABC123' -> 'ABC123'
        'A-B-C-1-2-3' -> 'ABC123'
        'A B C 1 2 3' -> 'ABC123'
        'A, B, C, 1, 2, 3' -> 'ABC123'
        'A, B, C, one, two, three' -> 'ABC123'
        'bob12' -> 'BOB12'
        'fl-123' -> 'FL123'
    """
    if not isinstance(val, str):
        return val

    # 1. Substitute spoken number words if present (e.g. 'one' -> '1')
    def _sub_word(match: re.Match[str]) -> str:
        word = match.group(1).lower()
        return WORD_TO_DIGIT.get(word, match.group(1))

    replaced = _NUM_WORD_RE.sub(_sub_word, val)

    # 2. Strip all punctuation, whitespace, commas, hyphens
    cleaned = re.sub(r"[^A-Za-z0-9]", "", replaced)

    # 3. Return uppercase canonical identifier if characters remain
    return cleaned.upper() if cleaned else val.strip().upper()


def normalize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Deterministically normalize tool arguments.

    Preserves semantic content and punctuation for non-identifier fields
    (e.g., addresses, conversational queries, filter values).
    Canonicalizes identifier-like fields (order_id, flight_id, doc_number, product_id)
    and currency codes.
    """
    normalized: dict[str, Any] = {}
    for key, val in arguments.items():
        k_lower = key.lower()
        if (
            k_lower in IDENTIFIER_PARAMS
            or k_lower.endswith("_id")
            or k_lower.endswith("_number")
        ) and isinstance(val, str):
            normalized[key] = normalize_identifier(val)
        elif k_lower in CURRENCY_PARAMS and isinstance(val, str):
            normalized[key] = val.strip().upper()
        else:
            normalized[key] = val
    return normalized
