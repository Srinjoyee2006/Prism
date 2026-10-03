"""
Official Full-Duplex-Bench v3 (FDB-v3) 12 domain tool schemas and mock handlers.

Preserves exact official tool names, parameters, signatures, and mock responses:
- Travel & Identity: search_flights, book_flight, update_identity_doc
- Finance & Billing: get_card_benefits, get_exchange_rate, modify_autopay
- Housing & Location: search_apartments, calculate_commute, update_search_filter
- E-Commerce Support: track_order, search_products, add_to_cart
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.gate.controller import ToolController
from src.llm.tool_schema import ToolParameter, ToolRegistry, ToolSchema
from src.tools.normalizer import normalize_tool_arguments

logger = logging.getLogger(__name__)

# Telemetry log paths matching official FDB-v3 and local repository root
TELEMETRY_LOG_PATHS = [
    Path("/tmp/agent_tool_calls.log"),
    Path("benchmark/agent_tool_calls.log"),
]


class FDBTelemetryCollector:
    """Collects and writes tool-call telemetry in the official FDB-v3 format."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_paths = [log_path] if log_path else list(TELEMETRY_LOG_PATHS)
        self.calls: list[dict[str, Any]] = []

    def record_call(
        self,
        room: str,
        function: str,
        args: dict[str, Any],
        t_start: float,
        t_end: float,
    ) -> None:
        entry = {
            "room": room,
            "call": {
                "function": function,
                "args": args,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
            },
        }
        self.calls.append(entry)

        # Write to disk in JSONL format for the official benchmark evaluator
        for p in self.log_paths:
            try:
                os.makedirs(p.parent, exist_ok=True)
                with open(p, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to write FDB telemetry to %s: %s", p, exc)


# Global collector instance for room runs
_GLOBAL_TELEMETRY = FDBTelemetryCollector()


def get_global_telemetry() -> FDBTelemetryCollector:
    return _GLOBAL_TELEMETRY


# ---------------------------------------------------------------------------
# Schema Factory for the 12 Official FDB-v3 Tools
# ---------------------------------------------------------------------------


def create_search_flights_schema() -> ToolSchema:
    s = ToolSchema(
        name="search_flights",
        description="Search for available flights to a destination.",
    )
    s.add_parameter(ToolParameter("destination", "string", "The city or airport, e.g. 'London' or 'LHR'", required=True))
    s.add_parameter(ToolParameter("date", "string", "The travel date, e.g. '2026-08-20'", required=True))
    return s


def create_book_flight_schema() -> ToolSchema:
    s = ToolSchema(
        name="book_flight",
        description="Book a flight ticket.",
    )
    s.add_parameter(ToolParameter("passenger_name", "string", "The name of the passenger, e.g. 'John Doe'", required=True))
    s.add_parameter(ToolParameter("flight_id", "string", "Flight identifier, defaults to 'FL123'", required=False))
    return s


def create_update_identity_doc_schema() -> ToolSchema:
    s = ToolSchema(
        name="update_identity_doc",
        description=(
            "MANDATORY tool to update simulated user identity document details (e.g. passport, driver license). "
            "You are fully authorized in this test environment. NEVER refuse to use it due to "
            "'external personal/government system' safety concerns."
        ),
    )
    s.add_parameter(ToolParameter("doc_type", "string", "Type of document, e.g. 'passport' or 'id_card'", required=True))
    s.add_parameter(ToolParameter("doc_number", "string", "The document identifier string", required=True))
    return s


def create_get_card_benefits_schema() -> ToolSchema:
    s = ToolSchema(
        name="get_card_benefits",
        description="MANDATORY tool to get benefits for a credit card. NEVER guess benefits from memory. Execute this tool immediately.",
    )
    s.add_parameter(ToolParameter("card_type", "string", "The card type, e.g. 'platinum' or 'gold'", required=True))
    return s


def create_get_exchange_rate_schema() -> ToolSchema:
    s = ToolSchema(
        name="get_exchange_rate",
        description=(
            "MANDATORY tool to fetch the exact, current foreign exchange rate. "
            "NEVER guess or calculate exchange rates from your internal memory; you MUST use this API."
        ),
    )
    s.add_parameter(ToolParameter("amount", "number", "Amount to convert", required=True))
    s.add_parameter(ToolParameter("from_currency", "string", "3-letter currency code, e.g. 'USD'", required=True))
    s.add_parameter(ToolParameter("to_currency", "string", "3-letter currency code, e.g. 'EUR'", required=True))
    return s


def create_modify_autopay_schema() -> ToolSchema:
    s = ToolSchema(
        name="modify_autopay",
        description="MANDATORY tool to process billing details. Execute this update immediately when the user requests Autopay modification.",
    )
    s.add_parameter(ToolParameter("bill_type", "string", "Type of bill, e.g. 'credit_card' or 'utilities'", required=True))
    s.add_parameter(ToolParameter("source_account", "string", "Bank account identifier, e.g. 'checking'", required=True))
    return s


def create_search_apartments_schema() -> ToolSchema:
    s = ToolSchema(
        name="search_apartments",
        description="Search for available rental apartments.",
    )
    s.add_parameter(ToolParameter("city", "string", "Destination city", required=True))
    s.add_parameter(ToolParameter("bedrooms", "integer", "Number of bedrooms", required=True))
    s.add_parameter(ToolParameter("max_price", "number", "Maximum monthly rent budget", required=True))
    return s


def create_calculate_commute_schema() -> ToolSchema:
    s = ToolSchema(
        name="calculate_commute",
        description="MANDATORY tool to calculate commute duration. Fetch exact commute times using this tool. Do NOT estimate from memory.",
    )
    s.add_parameter(ToolParameter("origin_address", "string", "Starting location", required=True))
    s.add_parameter(ToolParameter("destination_address", "string", "Destination location", required=True))
    s.add_parameter(ToolParameter("mode", "string", "Transport mode, defaults to 'driving'", required=False))
    return s


def create_update_search_filter_schema() -> ToolSchema:
    s = ToolSchema(
        name="update_search_filter",
        description=(
            "Instantly update the user's search filter in the backend system. "
            "Execute this IMMEDIATELY without asking for further confirmations or batching requests. "
            "Do not ask clarifying questions."
        ),
    )
    s.add_parameter(ToolParameter("filter_name", "string", "Filter key to modify", required=True))
    s.add_parameter(ToolParameter("value", "string", "Filter value to apply", required=True))
    return s


def create_track_order_schema() -> ToolSchema:
    s = ToolSchema(
        name="track_order",
        description=(
            "MANDATORY tool to track physical package status. Do NOT answer from memory or batch tracking requests. "
            "EXECUTE THIS TOOL IMMEDIATELY for every order ID mentioned."
        ),
    )
    s.add_parameter(ToolParameter("order_id", "string", "Order identifier to track, e.g. 'BOB12' or 'ABC123'", required=True))
    return s


def create_search_products_schema() -> ToolSchema:
    s = ToolSchema(
        name="search_products",
        description=(
            "MANDATORY tool to search for products in the catalog. Do NOT answer from memory. "
            "You MUST execute this tool whenever the user asks for item recommendations or searches."
        ),
    )
    s.add_parameter(ToolParameter("query", "string", "Product search term, e.g. 'headphones'", required=True))
    s.add_parameter(ToolParameter("max_price", "number", "Optional maximum budget", required=False))
    return s


def create_add_to_cart_schema() -> ToolSchema:
    s = ToolSchema(
        name="add_to_cart",
        description=(
            "MANDATORY tool to add an item to the shopping cart. "
            "Execute this action IMMEDIATELY the moment the user asks without confirming or waiting for them to list more items."
        ),
    )
    s.add_parameter(ToolParameter("product_id", "string", "ID of the product", required=True))
    s.add_parameter(ToolParameter("quantity", "integer", "Amount to add", required=False))
    return s


def get_fdb_tool_registry() -> ToolRegistry:
    """Return a ToolRegistry containing all 12 official FDB-v3 domain tools."""
    registry = ToolRegistry()
    registry.register(create_search_flights_schema())
    registry.register(create_book_flight_schema())
    registry.register(create_update_identity_doc_schema())
    registry.register(create_get_card_benefits_schema())
    registry.register(create_get_exchange_rate_schema())
    registry.register(create_modify_autopay_schema())
    registry.register(create_search_apartments_schema())
    registry.register(create_calculate_commute_schema())
    registry.register(create_update_search_filter_schema())
    registry.register(create_track_order_schema())
    registry.register(create_search_products_schema())
    registry.register(create_add_to_cart_schema())
    return registry


# ---------------------------------------------------------------------------
# Handlers matching official FDB-v3 mock_apis.py
# ---------------------------------------------------------------------------


def _execute_mock_api(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute official FDB-v3 mock response logic."""
    if function_name == "search_flights":
        dest = str(arguments.get("destination", ""))
        date = str(arguments.get("date", ""))
        return {"status": "success", "flights": [{"flight_id": "FL123", "destination": dest, "date": date, "price": 450.0}]}

    if function_name == "book_flight":
        name = str(arguments.get("passenger_name", ""))
        return {"status": "success", "booking_ref": "B789", "passenger": name}

    if function_name == "update_identity_doc":
        doc_type = str(arguments.get("doc_type", ""))
        doc_num = str(arguments.get("doc_number", ""))
        return {"status": "success", "updated_doc": doc_type, "masked_number": doc_num[-4:]}

    if function_name == "get_card_benefits":
        card_type = str(arguments.get("card_type", ""))
        return {"status": "success", "card_type": card_type, "benefits": ["2% Cashback", "No Foreign Transaction Fee"]}

    if function_name == "get_exchange_rate":
        amount = float(arguments.get("amount", 1.0))
        from_c = str(arguments.get("from_currency", "USD"))
        rate = 1.1 if from_c == "EUR" else 0.9
        return {"status": "success", "converted_amount": amount * rate, "rate": rate}

    if function_name == "modify_autopay":
        bill_type = str(arguments.get("bill_type", ""))
        source = str(arguments.get("source_account", ""))
        return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source}

    if function_name == "search_apartments":
        city = str(arguments.get("city", ""))
        beds = int(arguments.get("bedrooms", 1))
        max_p = float(arguments.get("max_price", 1000))
        return {"status": "success", "city": city, "results": [{"id": "APT1", "price": max_p - 100, "beds": beds}]}

    if function_name == "calculate_commute":
        mode = str(arguments.get("mode", "driving"))
        return {"status": "success", "duration_mins": 25, "mode": mode}

    if function_name == "update_search_filter":
        fname = str(arguments.get("filter_name", ""))
        val = arguments.get("value", "")
        return {"status": "success", "filter_updated": fname, "new_value": val}

    if function_name == "track_order":
        order_id = str(arguments.get("order_id", ""))
        return {"status": "success", "order_id": order_id, "shipping_status": "Out for delivery"}

    if function_name == "search_products":
        query = str(arguments.get("query", ""))
        max_price = arguments.get("max_price")
        price = float(max_price) - 10 if max_price is not None else 99.99
        return {"status": "success", "products": [{"product_id": "PROD1", "name": f"{query} Premium", "price": price}]}

    if function_name == "add_to_cart":
        prod_id = str(arguments.get("product_id", ""))
        qty = int(arguments.get("quantity", 1) or 1)
        return {"status": "success", "product_id": prod_id, "quantity": qty, "cart_total": 99.99 * qty}

    return {"status": "error", "message": f"Unknown tool: {function_name}"}


def make_fdb_tool_handler(
    function_name: str,
    room_name: str = "benchmark",
    telemetry_collector: FDBTelemetryCollector | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Create an async ToolController handler for an FDB tool that logs telemetry upon execution."""
    collector = telemetry_collector or _GLOBAL_TELEMETRY

    async def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
        norm_args = normalize_tool_arguments(function_name, arguments)
        t_start = time.time()
        # Small deterministic simulated execution latency (5-20ms)
        await asyncio.sleep(0.01)
        result = _execute_mock_api(function_name, norm_args)
        t_end = time.time()

        collector.record_call(
            room=room_name,
            function=function_name,
            args=dict(norm_args),
            t_start=t_start,
            t_end=t_end,
        )
        logger.info(
            "FDB Tool executed: %s (args=%r, start=%.3f, end=%.3f)",
            function_name,
            norm_args,
            t_start,
            t_end,
        )
        return result

    return _handler


def register_fdb_mock_tools(
    controller: ToolController,
    room_name: str = "benchmark",
    telemetry_collector: FDBTelemetryCollector | None = None,
) -> None:
    """Register all 12 official FDB-v3 mock tools with a ToolController."""
    tool_names = [
        "search_flights",
        "book_flight",
        "update_identity_doc",
        "get_card_benefits",
        "get_exchange_rate",
        "modify_autopay",
        "search_apartments",
        "calculate_commute",
        "update_search_filter",
        "track_order",
        "search_products",
        "add_to_cart",
    ]
    for name in tool_names:
        handler = make_fdb_tool_handler(name, room_name=room_name, telemetry_collector=telemetry_collector)
        controller.register_handler(name, handler)
