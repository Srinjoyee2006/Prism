"""
Unit tests for Full-Duplex-Bench v3 (FDB-v3) integration and benchmark adapter.

Verifies:
1. All 12 official tools schemas and parameters
2. Mock API execution and return values
3. ToolProposal -> ToolController -> 0.9s Commit Gate invariant
4. Telemetry logging format matching official evaluator expectations
5. Interruption during quiet window drops proposal and prevents tool execution
"""

import asyncio
import json
from pathlib import Path

import pytest

from src.gate.controller import ToolController
from src.gate.models import ProposalStatus
from src.llm.models import ToolProposal
from src.tools.fdb_tools import (
    FDBTelemetryCollector,
    get_fdb_tool_registry,
    register_fdb_mock_tools,
)


def test_fdb_12_tools_registered() -> None:
    """Verify all 12 official FDB-v3 domain tools are present in registry."""
    registry = get_fdb_tool_registry()
    schemas = registry.list_schemas()
    assert len(schemas) == 12

    names = {s.name for s in schemas}
    expected_12 = {
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
    }
    assert names == expected_12


def test_fdb_tool_schemas_parameters() -> None:
    """Verify parameters for representative FDB tools."""
    registry = get_fdb_tool_registry()

    # track_order
    track_schema = registry.get("track_order")
    assert track_schema is not None
    assert "order_id" in track_schema.parameters
    assert track_schema.parameters["order_id"].required is True

    # search_flights
    flight_schema = registry.get("search_flights")
    assert flight_schema is not None
    assert "destination" in flight_schema.parameters
    assert "date" in flight_schema.parameters
    assert flight_schema.parameters["destination"].required is True

    # add_to_cart
    cart_schema = registry.get("add_to_cart")
    assert cart_schema is not None
    assert "product_id" in cart_schema.parameters
    assert cart_schema.parameters["product_id"].required is True


@pytest.mark.asyncio
async def test_fdb_mock_tools_through_controller(tmp_path: Path) -> None:
    """Verify that FDB tools strictly follow: ToolProposal -> ToolController -> CommitGate -> Execution."""
    telemetry_file = tmp_path / "test_telemetry.log"
    collector = FDBTelemetryCollector(log_path=telemetry_file)

    controller = ToolController(quiet_window=0.05)  # Fast quiet window for unit test
    register_fdb_mock_tools(controller, room_name="test-room-1", telemetry_collector=collector)

    proposal = ToolProposal(
        tool_name="track_order",
        arguments={"order_id": "ABC123"},
        call_id="call-001",
    )

    staged = await controller.submit_proposal(proposal)
    assert staged.status in (ProposalStatus.PROPOSED, ProposalStatus.WAITING)

    # Wait for quiet window to commit
    completed = await controller.wait_for_proposal(staged.proposal_id, timeout=1.0)
    assert completed is not None
    assert completed.status == ProposalStatus.SUCCEEDED
    assert completed.result == {"status": "success", "order_id": "ABC123", "shipping_status": "Out for delivery"}

    # Verify telemetry was recorded
    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["room"] == "test-room-1"
    assert call["call"]["function"] == "track_order"
    assert call["call"]["args"] == {"order_id": "ABC123"}
    assert call["call"]["timestamp_end"] >= call["call"]["timestamp_start"]

    # Verify file was written
    assert telemetry_file.exists()
    line = json.loads(telemetry_file.read_text(encoding="utf-8").strip())
    assert line["room"] == "test-room-1"
    assert line["call"]["function"] == "track_order"


@pytest.mark.asyncio
async def test_fdb_interruption_drops_proposal(tmp_path: Path) -> None:
    """Verify that user speech interruption during the quiet window drops the proposal and prevents execution."""
    telemetry_file = tmp_path / "test_telemetry_interrupt.log"
    collector = FDBTelemetryCollector(log_path=telemetry_file)

    controller = ToolController(quiet_window=0.5)
    register_fdb_mock_tools(controller, room_name="test-room-interrupt", telemetry_collector=collector)

    proposal = ToolProposal(
        tool_name="book_flight",
        arguments={"passenger_name": "Alice"},
        call_id="call-002",
    )

    staged = await controller.submit_proposal(proposal)
    assert staged.status in (ProposalStatus.PROPOSED, ProposalStatus.WAITING)

    # Simulate user barge-in after 50ms (before 500ms quiet window closes)
    await asyncio.sleep(0.05)
    controller.notify_user_speech_started()

    completed = await controller.wait_for_proposal(staged.proposal_id, timeout=1.0)
    assert completed is not None
    assert completed.status == ProposalStatus.DROPPED
    assert completed.result is None

    # Telemetry should NOT record uncommitted/dropped proposals
    assert len(collector.calls) == 0


def test_measure_audio_latency(tmp_path: Path) -> None:
    """Verify audio latency calculation with synthetic silent/speech WAV."""
    from benchmark.prism_adapter import measure_audio_latency, write_wav

    # 1 second of silence + 1 second of tone
    sample_rate = 24000
    silence = b"\x00" * (sample_rate * 2)  # 1 sec silence
    import numpy as np

    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()

    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    write_wav(in_wav, silence, sample_rate=sample_rate)
    write_wav(out_wav, silence + tone, sample_rate=sample_rate)

    metrics = measure_audio_latency(in_wav, out_wav)
    assert "first_speech_s" in metrics
    # first speech starts after the 1 second silence
    assert 0.9 <= metrics["first_speech_s"] <= 1.1
