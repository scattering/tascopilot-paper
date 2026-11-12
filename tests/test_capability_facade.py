"""Tests for the ontology-routed capability facade (A/B alternative to the flat MCP surface)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tas_copilot_core.capability_facade import (
    FORBIDDEN_CAPABILITY_NAMES,
    capability_tools,
    dispatch_capability,
    load_routing,
    resolve,
)
from tas_copilot_core.workflows.copilot_router import COPILOT_INTENT_EXAMPLES


def test_routing_loads_and_covers_every_intent() -> None:
    routing = load_routing()

    routed_intents = {route["intent"] for spec in routing.values() for route in spec["routes"]}
    all_intents = {str(item["intent"]) for item in COPILOT_INTENT_EXAMPLES}
    assert routed_intents == all_intents, f"unrouted intents: {sorted(all_intents - routed_intents)}"
    # Ontology expansion happened: routes that name concepts carry their keywords.
    plan_routes = {route["intent"]: route for route in routing["plan_measurement"]["routes"]}
    assert any("spin wave" in term.lower() or "spinwave" in term.lower() for term in plan_routes["dispersion_scan_plan"]["terms"])


def test_capability_names_cannot_express_execution() -> None:
    routing = load_routing()

    assert set(routing) == {"plan_measurement", "interpret", "model", "plan", "verify"}
    assert not (set(routing) & FORBIDDEN_CAPABILITY_NAMES)
    # A config edit introducing an execution-shaped capability fails closed.
    bad = Path(__file__).parent / "fixtures"  # reuse tmp-free path check via inline yaml
    import yaml

    text = yaml.safe_dump({"capabilities": {"measure": {"description": "x", "routes": [{"intent": "copilot_status", "phrases": ["status"]}]}}})
    with pytest.raises(ValueError, match="Forbidden capability name"):
        load_routing_from_text(text)


def load_routing_from_text(text: str):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        path = handle.name
    return load_routing(path)


def test_unknown_intent_and_unknown_ontology_concept_fail_closed(tmp_path: Path) -> None:
    import yaml

    bad_intent = tmp_path / "bad_intent.yaml"
    bad_intent.write_text(
        yaml.safe_dump({"capabilities": {"verify": {"description": "x", "routes": [{"intent": "fire_the_laser", "phrases": ["x"]}]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown intent"):
        load_routing(bad_intent)

    bad_concept = tmp_path / "bad_concept.yaml"
    bad_concept.write_text(
        yaml.safe_dump(
            {"capabilities": {"verify": {"description": "x", "routes": [{"intent": "copilot_status", "ontology_concepts": ["not_a_concept"]}]}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown ontology concept"):
        load_routing(bad_concept)


def test_resolve_matches_ontology_terms_and_fails_closed() -> None:
    routing = load_routing()

    hit = resolve("plan_measurement", "extend the spin wave dispersion coverage", routing)
    assert hit["status"] == "ok"
    assert hit["intent"] == "dispersion_scan_plan"

    spurion = resolve("verify", "could this narrow feature be a spurion?", routing)
    assert spurion["status"] == "ok"
    assert spurion["intent"] == "spurion_diagnostic_survey"

    miss = resolve("plan_measurement", "quantum chromodynamics", routing)
    assert miss["status"] == "error"
    assert miss["available_concepts"]

    empty = resolve("interpret", "", routing)
    assert empty["status"] == "missing_input"

    unknown = resolve("teleport", "anything", routing)
    assert unknown["status"] == "error"
    assert "plan_measurement" in unknown["known_capabilities"]


def test_dispatch_capability_routes_through_hardened_gateway(tmp_path: Path) -> None:
    response = dispatch_capability("plan", {"concept": "what can you do - status"}, tmp_path / "runs")

    assert response["status"] == "ok"
    assert response["intent"] == "copilot_status"
    assert response["capability_routing"]["capability"] == "plan"
    assert response["capability_routing"]["resolved_intent"] == "copilot_status"
    assert response["safety_note"].startswith("Read-only")
    # The hardened seam applies: the identical capability call dedupes.
    repeat = dispatch_capability("plan", {"concept": "what can you do - status"}, tmp_path / "runs")
    assert repeat["status"] == "duplicate_call"


def test_capability_tools_expose_concept_plus_flat_fields() -> None:
    tools = capability_tools()

    by_name = {tool["name"]: tool for tool in tools}
    assert set(by_name) == {"plan_measurement", "interpret", "model", "plan", "verify"}
    interpret = by_name["interpret"]["inputSchema"]
    assert interpret["required"] == ["concept"]
    assert "data_root" in interpret["properties"]
    assert interpret["additionalProperties"] is False
    # Safety note rides on every capability description.
    assert all("Read-only" in tool["description"] for tool in tools)


def test_create_server_capability_surface(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from mcp.types import ListToolsRequest

    from tas_copilot_core.mcp_gateway import create_server

    def _names(server) -> set[str]:
        handler = server.request_handlers[ListToolsRequest]
        result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
        return {tool.name for tool in result.root.tools}

    caps = _names(create_server(tmp_path / "runs", surface="capabilities"))
    assert {"plan_measurement", "interpret", "model", "plan", "verify", "read_report", "evidence_graph_search"} <= caps
    assert "experiment_dashboard" not in caps

    both = _names(create_server(tmp_path / "runs2", surface="both"))
    assert {"plan_measurement", "experiment_dashboard"} <= both

    with pytest.raises(ValueError, match="Unknown surface"):
        create_server(tmp_path / "runs3", surface="wide_open")
