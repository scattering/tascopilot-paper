import json
from pathlib import Path

import pytest

from tas_copilot_contracts import CopilotAskRequest

from tas_copilot_core.mcp_gateway import (
    _CALL_GUARD,
    EVIDENCE_GRAPH_SEARCH_TOOL,
    HAND_WRITTEN_TOOLS,
    READ_REPORT_TOOL,
    REQUIRED_INPUT_FIELDS,
    dispatch,
    dispatch_with_timeout,
    intent_tools,
)
from tas_copilot_core.workflows.copilot_router import COPILOT_INTENT_EXAMPLES, _route, copilot_ask

FIXTURE = Path(__file__).parent / "fixtures" / "proposal_retrospective" / "ansto_dto_thin_film"


@pytest.fixture(autouse=True)
def _fresh_call_guard():
    """Each test starts with clean dedupe/rate-limit state."""
    _CALL_GUARD.reset()
    yield
    _CALL_GUARD.reset()


def test_every_intent_example_generates_a_tool() -> None:
    tools = intent_tools()
    tool_names = {tool["name"] for tool in tools}
    intents = {example["intent"] for example in COPILOT_INTENT_EXAMPLES}
    assert intents <= tool_names
    assert READ_REPORT_TOOL in tool_names
    assert EVIDENCE_GRAPH_SEARCH_TOOL in tool_names
    # The surface is generated: no tool exists without an intent example,
    # except the two hand-written read-only lookups.
    assert tool_names == intents | set(HAND_WRITTEN_TOOLS)


def test_tool_schemas_are_flat_and_closed() -> None:
    for tool in intent_tools():
        schema = tool["inputSchema"]
        assert schema["additionalProperties"] is False
        # Requirements live in descriptions; the router enforces them and
        # returns structured missing_inputs (MCP SDK schema validation would
        # otherwise short-circuit the slot-filling flow). The hand-written
        # lookups do not slot-fill, so they may declare protocol-level
        # requirements.
        if tool["name"] not in HAND_WRITTEN_TOOLS:
            assert "required" not in schema
        for name, prop in schema["properties"].items():
            assert prop["type"] in {"string", "integer", "number", "boolean"}, f"{tool['name']}.{name} is not flat"
            assert prop.get("description"), f"{tool['name']}.{name} lacks a description"
        assert tool["description"]
        assert "planning evidence" in tool["description"]


def test_descriptions_come_from_intent_examples() -> None:
    by_name = {tool["name"]: tool for tool in intent_tools()}
    for example in COPILOT_INTENT_EXAMPLES:
        tool = by_name[example["intent"]]
        assert str(example["situation"]) in tool["description"]
        assert str(example["example_question"]) in tool["description"]
        if example.get("required_inputs"):
            assert "Requires" in tool["description"] or "at least one of" in tool["description"]


def test_required_input_labels_all_have_field_mappings() -> None:
    labels = {label for example in COPILOT_INTENT_EXAMPLES for label in example.get("required_inputs") or []}
    unmapped = labels - set(REQUIRED_INPUT_FIELDS)
    assert not unmapped, f"router required-input labels without gateway field mappings: {unmapped}"


def test_probe_invariant_every_example_question_routes_to_a_known_resolution() -> None:
    # Forced intents are resolved by routing the intent's canonical example
    # question; each must land on its own intent or a context-needed fallback.
    for example in COPILOT_INTENT_EXAMPLES:
        decision = _route(CopilotAskRequest(question=str(example["example_question"]), output_dir="/tmp/probe"))
        assert decision.intent in {example["intent"], "needs_more_context", "paper_or_figure_context_needed"}, (
            example["intent"],
            decision.intent,
        )


def test_forced_intent_overrides_keyword_routing(tmp_path: Path) -> None:
    result = copilot_ask({"question": "tell me about zzz", "intent": "experiment_review", "output_dir": str(tmp_path / "r")})
    assert result["route"]["intent"] == "experiment_review"
    assert result["status"] == "missing_input"
    assert result["missing_inputs"] == ["data_root"]
    assert "supplied explicitly" in result["route"]["rationale"][0]


def test_forced_intent_never_bypasses_live_control_block(tmp_path: Path) -> None:
    result = copilot_ask(
        {"question": "move motor a3 to 45 and start scan now", "intent": "experiment_review", "output_dir": str(tmp_path / "r")}
    )
    assert result["route"]["intent"] == "blocked_live_control"
    assert result["status"] == "blocked"


def test_unknown_forced_intent_fails_closed(tmp_path: Path) -> None:
    result = copilot_ask({"question": "x", "intent": "not_a_real_intent", "output_dir": str(tmp_path / "r")})
    assert result["route"]["intent"] == "needs_more_context"
    assert "Unknown explicit intent" in result["route"]["rationale"][0]


def test_dispatch_status_intent(tmp_path: Path) -> None:
    result = dispatch("copilot_status", {}, tmp_path / "runs")
    assert result["status"] == "ok"
    assert result["intent"] == "copilot_status"
    assert result["answer"]
    assert result["answer_envelope"]
    assert result["safety_note"].startswith("Read-only")
    audit = (tmp_path / "data" / "audit" / "mcp_gateway.jsonl").read_text(encoding="utf-8")
    assert json.loads(audit.splitlines()[0])["tool"] == "copilot_status"


def test_dispatch_missing_input_drives_slot_filling(tmp_path: Path) -> None:
    result = dispatch("spurion_diagnostic_survey", {}, tmp_path / "runs")
    assert result["status"] == "missing_input"
    assert result["missing_inputs"] == ["data_root"]


def test_dispatch_runs_scan_plan_suggestions_from_fixture(tmp_path: Path) -> None:
    result = dispatch("scan_plan_suggestions", {"fixture": str(FIXTURE)}, tmp_path / "runs")
    assert result["status"] in {"ok", "partial"}
    assert result["intent"] == "scan_plan_suggestions"
    assert result["routed_workflows"]
    # Artifacts are referenced by path and readable through read_report.
    artifacts = result.get("artifacts") or {}
    markdown_path = artifacts.get("markdown") or artifacts.get("md")
    if markdown_path:
        report = dispatch(READ_REPORT_TOOL, {"path": markdown_path}, tmp_path / "runs")
        assert report["status"] == "ok"
        assert report["content"]


def test_dispatch_rejects_unknown_tool_and_arguments(tmp_path: Path) -> None:
    unknown_tool = dispatch("not_a_tool", {}, tmp_path / "runs")
    assert unknown_tool["status"] == "error"
    assert "copilot_status" in unknown_tool["known_tools"]

    unknown_argument = dispatch("copilot_status", {"data_rooot": "/x"}, tmp_path / "runs")
    assert unknown_argument["status"] == "error"
    assert "data_rooot" in unknown_argument["error"]


def test_read_report_is_confined_to_runs_root(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    inside = runs / "mcp_gateway" / "x" / "report.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("# hello", encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")

    assert dispatch(READ_REPORT_TOOL, {"path": str(inside)}, runs)["status"] == "ok"
    refused = dispatch(READ_REPORT_TOOL, {"path": str(outside)}, runs)
    assert refused["status"] == "error"
    assert "readable" in refused["error"]
    refused_type = dispatch(READ_REPORT_TOOL, {"path": str(inside.with_suffix(".py"))}, runs)
    assert refused_type["status"] == "error"


def test_create_server_registers_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from tas_copilot_core.mcp_gateway import create_server

    server = create_server(tmp_path / "runs")
    assert server.name == "tascopilot"

    from mcp.types import ListToolsRequest

    handler = server.request_handlers[ListToolsRequest]
    result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
    names = {tool.name for tool in result.root.tools}
    assert "copilot_status" in names
    assert READ_REPORT_TOOL in names
    assert EVIDENCE_GRAPH_SEARCH_TOOL in names
    assert len(names) == len(COPILOT_INTENT_EXAMPLES) + len(HAND_WRITTEN_TOOLS)


def _graph_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "source_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "text": "Ordered moment of 3.4 muB in the ab plane from powder neutron diffraction.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (catalog / "evidence_graph_nodes.jsonl").write_text(
        "\n".join(
            json.dumps(node)
            for node in [
                {"node_id": "Paper:paper-1", "node_type": "Paper", "label": "CaMn2Sb2 powder study", "provenance": []},
                {"node_id": "SourceChunk:chunk-1", "node_type": "SourceChunk", "label": "ordered moment chunk", "source_object_id": "chunk-1", "provenance": []},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (catalog / "evidence_graph_edges.jsonl").write_text(
        "\n".join(
            json.dumps(edge)
            for edge in [
                {
                    "edge_id": "has_chunk-1",
                    "source_node_id": "Paper:paper-1",
                    "target_node_id": "SourceChunk:chunk-1",
                    "edge_type": "HAS_CHUNK",
                    "confidence": 0.95,
                    "status": "accepted",
                    "evidence": ["ordered moment"],
                    "provenance": [],
                },
                {
                    "edge_id": "consistent_with-1",
                    "source_node_id": "Paper:paper-1",
                    "target_node_id": "SourceChunk:chunk-1",
                    "edge_type": "CONSISTENT_WITH",
                    "confidence": 0.8,
                    "status": "proposed",
                    "evidence": ["ordered moment agreement"],
                    "provenance": [],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return catalog


def test_evidence_graph_search_serves_typed_paths(tmp_path: Path) -> None:
    catalog = _graph_catalog(tmp_path)

    result = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"corpus_catalog": str(catalog), "query": "ordered moment"}, tmp_path / "runs")

    assert result["status"] == "ok"
    assert result["result_count"] >= 2
    first = result["results"][0]
    assert first["path"] and first["edge_type"] and first["status"]
    assert any("untrusted" in note for note in result["notes"])
    # Review status is visible so clients can distinguish knowledge from candidates.
    statuses = {item["status"] for item in result["results"]}
    assert "accepted" in statuses and "proposed" in statuses


def test_evidence_graph_search_min_status_filters_to_reviewed(tmp_path: Path) -> None:
    catalog = _graph_catalog(tmp_path)

    result = dispatch(
        EVIDENCE_GRAPH_SEARCH_TOOL,
        {"corpus_catalog": str(catalog), "query": "ordered moment", "min_status": "accepted"},
        tmp_path / "runs",
    )

    assert result["status"] == "ok"
    assert result["results"]
    assert all(item["status"] == "accepted" for item in result["results"])


def test_evidence_graph_search_fails_closed(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    missing_catalog = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"query": "moment"}, runs)
    assert missing_catalog["status"] == "missing_input"
    no_graph = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"corpus_catalog": str(tmp_path), "query": "moment"}, runs)
    assert no_graph["status"] == "error"
    catalog = _graph_catalog(tmp_path)
    bad_status = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"corpus_catalog": str(catalog), "query": "moment", "min_status": "gospel"}, runs)
    assert bad_status["status"] == "error"
    unknown_arg = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"corpus_catalog": str(catalog), "query": "moment", "sql": "drop"}, runs)
    assert unknown_arg["status"] == "error"


def test_dispatch_dedupes_identical_workflow_calls(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = dispatch("copilot_status", {}, runs)
    assert first["status"] == "ok"

    repeat = dispatch("copilot_status", {}, runs)
    assert repeat["status"] == "duplicate_call"
    assert repeat["prior_status"] == "ok"
    assert "mcp_gateway/copilot_status" in repeat["prior_output_dir"]

    # The canonical example question and an explicit identical question are the same call.
    example_question = next(item["example_question"] for item in COPILOT_INTENT_EXAMPLES if item["intent"] == "copilot_status")
    explicit = dispatch("copilot_status", {"question": example_question}, runs)
    assert explicit["status"] == "duplicate_call"

    # Changed arguments are a new call, not a duplicate.
    changed = dispatch("copilot_status", {"question": "Status again, but summarize readiness."}, runs)
    assert changed["status"] == "ok"


def test_dispatch_rate_limits_workflow_bursts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAS_COPILOT_MCP_MAX_CALLS_PER_MINUTE", "2")
    runs = tmp_path / "runs"
    assert dispatch("copilot_status", {"question": "status one"}, runs)["status"] == "ok"
    assert dispatch("copilot_status", {"question": "status two"}, runs)["status"] == "ok"

    burst = dispatch("copilot_status", {"question": "status three"}, runs)
    assert burst["status"] == "rate_limited"
    assert burst["retry_after_seconds"] > 0
    # Reads stay available while workflows are throttled: read_report answers
    # its own not-found error, not rate_limited.
    report = dispatch(READ_REPORT_TOOL, {"path": str(runs / "nope.md")}, runs)
    assert report["status"] == "error"
    assert "does not exist" in report["error"]


def test_dispatch_wraps_workflow_exceptions_in_error_envelopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tas_copilot_core.mcp_gateway as gateway

    def _boom(request):  # noqa: ANN001
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(gateway, "copilot_ask", _boom)
    result = dispatch("copilot_status", {}, tmp_path / "runs")
    assert result["status"] == "error"
    assert "workflow exploded" in result["error"]
    assert "do not retry the identical call" in result["hint"].lower()
    audit = (tmp_path / "data" / "audit" / "mcp_gateway.jsonl").read_text(encoding="utf-8")
    assert '"status": "error"' in audit


def test_dispatch_confines_path_arguments_when_allowlist_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("TAS_COPILOT_MCP_ALLOWED_PATH_ROOTS", str(allowed))
    runs = tmp_path / "runs"

    refused = dispatch("spurion_diagnostic_survey", {"data_root": str(outside)}, runs)
    assert refused["status"] == "error"
    assert "outside the allowed path roots" in refused["error"]
    assert refused["allowed_path_roots"] == [str(allowed)]

    graph_refused = dispatch(EVIDENCE_GRAPH_SEARCH_TOOL, {"corpus_catalog": str(outside), "query": "moment"}, runs)
    assert graph_refused["status"] == "error"
    assert "outside the allowed path roots" in graph_refused["error"]

    inside = dispatch("spurion_diagnostic_survey", {"data_root": str(allowed / "expt")}, runs)
    assert inside["status"] != "error" or "outside the allowed path roots" not in str(inside.get("error"))

    # Unset allowlist means no confinement (local single-user default).
    monkeypatch.delenv("TAS_COPILOT_MCP_ALLOWED_PATH_ROOTS")
    _CALL_GUARD.reset()
    unconfined = dispatch("spurion_diagnostic_survey", {"data_root": str(outside)}, runs)
    assert "outside the allowed path roots" not in str(unconfined.get("error"))


def test_dispatch_with_timeout_returns_typed_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import asyncio
    import time as _time

    import tas_copilot_core.mcp_gateway as gateway

    def _slow(name, arguments, runs_root):  # noqa: ANN001
        _time.sleep(2.0)
        return {"status": "ok"}

    monkeypatch.setattr(gateway, "dispatch", _slow)
    result = asyncio.run(gateway.dispatch_with_timeout("copilot_status", {}, tmp_path / "runs", timeout_seconds=0.2))
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert READ_REPORT_TOOL in result["error"]

    fast = asyncio.run(dispatch_with_timeout("copilot_status", {}, tmp_path / "runs", timeout_seconds=30.0))
    assert fast["status"] == "ok"
