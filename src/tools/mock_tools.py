"""
Mock tool implementations for Stage 3 testing.

Every function here simulates realistic latency using asyncio.sleep (async
mock tools) or time.sleep wrapped in asyncio.to_thread (blocking mock tools).
None of these functions make network calls or modify real system state.
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async mock tools
# ---------------------------------------------------------------------------


async def mock_set_temperature(arguments: dict[str, Any]) -> dict[str, Any]:
    """Simulate setting a smart thermostat temperature.

    Args:
        arguments: Expected keys: ``zone`` (str), ``temperature_c`` (number).

    Returns:
        A dict confirming the new setpoint.
    """
    zone = arguments.get("zone", "living_room")
    temperature_c = arguments.get("temperature_c", 22)
    await asyncio.sleep(0.05)  # simulate device round-trip
    result = {
        "status": "ok",
        "zone": zone,
        "temperature_c": temperature_c,
        "message": f"Temperature in '{zone}' set to {temperature_c}°C",
    }
    logger.debug("mock_set_temperature → %r", result)
    return result


async def mock_search_flights(arguments: dict[str, Any]) -> dict[str, Any]:
    """Simulate a flight search API call.

    Args:
        arguments: Expected keys: ``origin``, ``destination``, ``departure_date``.

    Returns:
        A dict containing a list of mock flight offers.
    """
    origin = arguments.get("origin", "JFK")
    destination = arguments.get("destination", "LHR")
    date = arguments.get("departure_date", "2025-01-01")
    await asyncio.sleep(0.08)
    result = {
        "status": "ok",
        "origin": origin,
        "destination": destination,
        "departure_date": date,
        "flights": [
            {"flight_id": "AA100", "price_usd": 450, "duration_h": 7.5},
            {"flight_id": "BA202", "price_usd": 510, "duration_h": 8.1},
        ],
    }
    logger.debug("mock_search_flights → %d results", len(result["flights"]))
    return result


async def mock_get_exchange_rate(arguments: dict[str, Any]) -> dict[str, Any]:
    """Simulate a currency exchange rate lookup.

    Args:
        arguments: Expected keys: ``from_currency``, ``to_currency``.

    Returns:
        A dict with a mock exchange rate.
    """
    from_c = arguments.get("from_currency", "USD")
    to_c = arguments.get("to_currency", "INR")
    await asyncio.sleep(0.03)
    # Deterministic fake rates for testing
    rates = {("USD", "INR"): 83.5, ("EUR", "USD"): 1.08, ("GBP", "EUR"): 1.17}
    rate = rates.get((from_c, to_c), 1.0)
    result = {"status": "ok", "from": from_c, "to": to_c, "rate": rate}
    logger.debug("mock_get_exchange_rate → %r", result)
    return result


async def mock_failing_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Simulate a tool that always raises an exception.

    Used to verify ToolController error handling.
    """
    await asyncio.sleep(0.01)
    raise RuntimeError("Simulated tool failure")


# ---------------------------------------------------------------------------
# Blocking (synchronous) mock tools
# ---------------------------------------------------------------------------


def blocking_track_order(arguments: dict[str, Any]) -> dict[str, Any]:
    """Simulate a synchronous order-tracking call (e.g. legacy SDK).

    This function intentionally uses ``time.sleep`` to represent a blocking
    operation.  Register it via ``ToolController.register_blocking_handler``
    so it runs safely inside ``asyncio.to_thread``.

    Args:
        arguments: Expected key: ``order_id`` (str).

    Returns:
        A dict with mock shipment status.
    """
    order_id = arguments.get("order_id", "ORD-00000")
    time.sleep(0.05)  # blocking sleep — must not run on event loop thread
    result = {
        "status": "ok",
        "order_id": order_id,
        "shipment_status": "in_transit",
        "estimated_delivery": "2025-03-15",
    }
    logger.debug("blocking_track_order → %r", result)
    return result
