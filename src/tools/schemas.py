"""
Representative domain tool schemas for testing and verification.

Defines standard tool schemas (search_flights, get_exchange_rate, track_order)
without benchmark hardcoding.
"""

from src.llm.tool_schema import ToolParameter, ToolRegistry, ToolSchema


def create_flight_search_schema() -> ToolSchema:
    """Schema for searching available commercial flights."""
    schema = ToolSchema(
        name="search_flights",
        description="Search for available flights between two airports for a given date.",
    )
    schema.add_parameter(
        ToolParameter(
            name="origin",
            param_type="string",
            description="3-letter IATA departure airport code (e.g. JFK, SFO).",
            required=True,
        )
    )
    schema.add_parameter(
        ToolParameter(
            name="destination",
            param_type="string",
            description="3-letter IATA arrival airport code (e.g. LHR, DEL).",
            required=True,
        )
    )
    schema.add_parameter(
        ToolParameter(
            name="departure_date",
            param_type="string",
            description="Date of departure in YYYY-MM-DD format.",
            required=True,
        )
    )
    schema.add_parameter(
        ToolParameter(
            name="max_price",
            param_type="number",
            description="Maximum ticket price in USD.",
            required=False,
        )
    )
    return schema


def create_exchange_rate_schema() -> ToolSchema:
    """Schema for retrieving foreign currency exchange rates."""
    schema = ToolSchema(
        name="get_exchange_rate",
        description="Get the current foreign exchange rate between two currency codes.",
    )
    schema.add_parameter(
        ToolParameter(
            name="from_currency",
            param_type="string",
            description="The 3-letter currency code to convert from (e.g. USD, EUR).",
            required=True,
        )
    )
    schema.add_parameter(
        ToolParameter(
            name="to_currency",
            param_type="string",
            description="The 3-letter currency code to convert to (e.g. INR, GBP).",
            required=True,
        )
    )
    return schema


def create_order_tracking_schema() -> ToolSchema:
    """Schema for tracking shipping status of an order."""
    schema = ToolSchema(
        name="track_order",
        description="Check current shipment status and delivery estimate for an order.",
    )
    schema.add_parameter(
        ToolParameter(
            name="order_id",
            param_type="string",
            description="Alphanumeric customer order identifier (e.g. ORD-12345).",
            required=True,
        )
    )
    return schema


def create_temperature_schema() -> ToolSchema:
    """Schema for adjusting the smart thermostat temperature."""
    schema = ToolSchema(
        name="set_temperature",
        description="Set the target thermostat temperature in degrees Celsius for a specific room or zone.",
    )
    schema.add_parameter(
        ToolParameter(
            name="zone",
            param_type="string",
            description="The room or zone name (e.g. living_room, bedroom, office).",
            required=True,
        )
    )
    schema.add_parameter(
        ToolParameter(
            name="temperature_c",
            param_type="number",
            description="Target temperature in Celsius (e.g. 21, 22.5).",
            required=True,
        )
    )
    return schema


def get_default_tool_registry() -> ToolRegistry:
    """Instantiate and populate a ToolRegistry with standard domain tool schemas."""
    registry = ToolRegistry()
    registry.register(create_flight_search_schema())
    registry.register(create_exchange_rate_schema())
    registry.register(create_order_tracking_schema())
    return registry


def get_voice_agent_tool_registry() -> ToolRegistry:
    """Instantiate ToolRegistry with full agent domain tools including temperature control."""
    registry = get_default_tool_registry()
    registry.register(create_temperature_schema())
    return registry
