"""
Unit tests for deterministic argument normalization.
"""

from src.tools.normalizer import normalize_identifier, normalize_tool_arguments


def test_normalize_identifier_variations():
    assert normalize_identifier("ABC123") == "ABC123"
    assert normalize_identifier("A-B-C-1-2-3") == "ABC123"
    assert normalize_identifier("A B C 1 2 3") == "ABC123"
    assert normalize_identifier("A, B, C, 1, 2, 3") == "ABC123"
    assert normalize_identifier("A, B, C, one, two, three") == "ABC123"
    assert normalize_identifier("abc-123") == "ABC123"
    assert normalize_identifier("fl-123") == "FL123"
    assert normalize_identifier("FL 456") == "FL456"
    assert normalize_identifier("987654321") == "987654321"


def test_normalize_tool_arguments_order_id():
    args = {"order_id": "A, B, C, 1, 2, 3"}
    res = normalize_tool_arguments("track_order", args)
    assert res == {"order_id": "ABC123"}


def test_normalize_tool_arguments_flight_and_passenger():
    args = {"passenger_name": "Alice Smith", "flight_id": "fl-101"}
    res = normalize_tool_arguments("book_flight", args)
    assert res == {"passenger_name": "Alice Smith", "flight_id": "FL101"}


def test_normalize_tool_arguments_preserves_non_identifiers():
    # Commas and hyphens in queries, addresses, or text must be preserved
    args = {
        "query": "wireless headphones, noise-cancelling",
        "max_price": 99.99,
    }
    res = normalize_tool_arguments("search_products", args)
    assert res["query"] == "wireless headphones, noise-cancelling"
    assert res["max_price"] == 99.99


def test_normalize_tool_arguments_currency():
    args = {"from_currency": "usd", "to_currency": "eur", "amount": 100}
    res = normalize_tool_arguments("get_exchange_rate", args)
    assert res == {"from_currency": "USD", "to_currency": "EUR", "amount": 100}


def test_normalize_tool_arguments_doc_number():
    args = {"doc_type": "passport", "doc_number": "u-123-456-78"}
    res = normalize_tool_arguments("update_identity_doc", args)
    assert res == {"doc_type": "passport", "doc_number": "U12345678"}
