from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from tas_copilot_contracts import (
    Availability, PermissionLevel, ReductionCorrectionRef, ReductionDAGEdge,
    ReductionDAGNode, ReductionDAGTemplate, ToolResult, ToolResultStatus,
)
from tas_copilot_core.reduction_dag import execute_reduction_dag, replay_reduction_dag_run
from tas_copilot_core.reduction_pipeline import scan_reduction_template
from tas_copilot_core.workflows.reduce_scan_set import reduce_scan_set


def _arguments():
    return {
        "steps": [
            {"tool": "add_scans", "arguments": {"scans": [{"scan_id": "a"}, {"scan_id": "b"}]}},
            {"tool": "normalize_to_monitor", "arguments": {"target_monitor": 100.0}},
            {"tool": "subtract_background", "arguments": {"background": {"scan_id": "bg", "marker": 0}}},
        ],
        "input_datafile_ids": ["a", "b", "bg"], "in_memory": True,
    }


def test_numerical_dag_preserves_inputs_tool_results_and_audits(dag_runner):
    arguments = _arguments()
    original = deepcopy(arguments)
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert report["status"] == "success"
    assert arguments == original
    nodes = report["reduction_dag"]["node_results"]
    assert all(node["tool_result"]["audit_id"] for node in nodes)
    assert all(node["tool_call"]["provenance_context"] for node in nodes)
    assert dag_runner.calls[2].arguments["signal"] == nodes[1]["output"]["scan"]
    assert dag_runner.calls[2].arguments["background"] == {"scan_id": "bg", "marker": 0}
    assert nodes[1]["output"]["scan"]["scan_id"].startswith("reduction:")


def test_shared_background_is_bound_only_to_subtraction(tmp_path, dag_runner):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    arguments["background"] = arguments["steps"][2]["arguments"].pop("background")
    reduce_scan_set(arguments, runner=dag_runner)
    assert "background" not in dag_runner.calls[0].arguments
    assert dag_runner.calls[2].arguments["background"] == arguments["background"]
    arguments["background"]["marker"] = 42
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert [node["cache_hit"] for node in report["reduction_dag"]["node_results"]] == [True, True, False]


def test_cache_reuses_numerical_outputs_and_invalidates_only_descendants(tmp_path, dag_runner):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    first = reduce_scan_set(arguments, runner=dag_runner)
    second = reduce_scan_set(arguments, runner=dag_runner)
    assert len(dag_runner.calls) == 3
    assert second["output"] == first["output"]
    assert all(node["cache_hit"] and node["cache_audit_id"] for node in second["reduction_dag"]["node_results"])
    arguments["steps"][1]["arguments"]["target_monitor"] = 200.0
    third = reduce_scan_set(arguments, runner=dag_runner)
    assert len(dag_runner.calls) == 5
    assert [node["cache_hit"] for node in third["reduction_dag"]["node_results"]] == [True, False, False]
    arguments["steps"][2]["arguments"]["background"]["marker"] = 9
    fourth = reduce_scan_set(arguments, runner=dag_runner)
    assert len(dag_runner.calls) == 6
    assert [node["cache_hit"] for node in fourth["reduction_dag"]["node_results"]] == [True, True, False]


def test_cache_invalidates_when_tool_version_changes(tmp_path, dag_runner):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    reduce_scan_set(arguments, runner=dag_runner)
    dag_runner.registry.get("tas.corrections", "normalize_to_monitor").spec.version = "fixture-2"
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert [node["cache_hit"] for node in report["reduction_dag"]["node_results"]] == [True, False, False]


@pytest.mark.parametrize("restriction", ["disabled", "unavailable", "live_control", "approval", "unsafe_arguments"])
def test_cache_cannot_bypass_current_permissions_or_availability(tmp_path, dag_runner, restriction):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    reduce_scan_set(arguments, runner=dag_runner)
    spec = dag_runner.registry.get("tas.file_arithmetic", "add_scans").spec
    if restriction == "disabled":
        spec.enabled_by_default = False
    elif restriction == "unavailable":
        spec.availability = Availability.unavailable
    elif restriction == "live_control":
        spec.enabled_by_default = False
        spec.permissions_required = [PermissionLevel.live_control]
    elif restriction == "approval":
        spec.permissions_required = [PermissionLevel.requires_human_approval]
    else:
        arguments["steps"][0]["arguments"]["move_motor"] = "A3"
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert report["status"] == ("unavailable" if restriction == "unavailable" else "blocked")
    assert len(dag_runner.calls) == 3
    assert report["reduction_dag"]["node_results"][1]["status"] == "blocked"


def test_cache_hit_still_calls_current_policy(tmp_path, dag_runner):
    from tas_copilot_core.registry.permissions import PermissionPolicy

    class DenyPolicy(PermissionPolicy):
        def check(self, spec, call):
            return False, "policy changed"

    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    reduce_scan_set(arguments, runner=dag_runner)
    dag_runner.policy = DenyPolicy()
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert report["status"] == "blocked"
    assert len(dag_runner.calls) == 3


@pytest.mark.parametrize("status", ["error", "missing_input", "validation_failed", "unavailable", "blocked"])
def test_failed_node_blocks_descendants_and_is_not_cached(tmp_path, dag_runner, status):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    registered = dag_runner.registry.get("tas.file_arithmetic", "add_scans")
    original = registered.handler
    registered.handler = lambda call: ToolResult(
        status=ToolResultStatus(status), tool_namespace=call.tool_namespace, tool_name=call.tool_name,
    )
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert report["status"] == status
    assert not dag_runner.calls
    assert [node["status"] for node in report["reduction_dag"]["node_results"]][1:] == ["blocked", "blocked"]
    registered.handler = original
    assert reduce_scan_set(arguments, runner=dag_runner)["status"] == "success"
    assert len(dag_runner.calls) == 3


def test_unregistered_tool_returns_unavailable(dag_runner):
    template, inputs = scan_reduction_template(_arguments())
    template.nodes[0].tool.tool_name = "not_installed"
    result = execute_reduction_dag(template, inputs=inputs, runner=dag_runner)
    assert result.status == "unavailable"
    assert result.node_results[0].tool_result.audit_id
    assert not dag_runner.calls


def test_missing_upstream_terminal_is_not_replaced_with_raw_input(dag_runner):
    registered = dag_runner.registry.get("tas.file_arithmetic", "add_scans")
    registered.handler = lambda call: ToolResult(
        status="success", output={"summary": "no numerical scan"},
        tool_namespace=call.tool_namespace, tool_name=call.tool_name,
    )
    report = reduce_scan_set(_arguments(), runner=dag_runner)
    assert report["status"] == "missing_input"
    assert not dag_runner.calls


def test_corrupt_cache_is_recomputed(tmp_path, dag_runner):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    first = reduce_scan_set(arguments, runner=dag_runner)
    key = first["reduction_dag"]["node_results"][-1]["cache_key"]
    (tmp_path / "cache" / f"{key}.json").write_text('{"result": "damaged"}')
    second = reduce_scan_set(arguments, runner=dag_runner)
    assert second["status"] == "success"
    assert len(dag_runner.calls) == 4


def test_file_content_changes_invalidate_translation_and_descendants(tmp_path, dag_runner):
    source = tmp_path / "source.bt7"
    source.write_text("first")
    arguments = {
        "steps": [{"tool": "translate_ice_file", "arguments": {"path": str(source)}}, "normalize_to_monitor"],
        "input_files": [str(source)], "in_memory": True, "cache_dir": str(tmp_path / "cache"),
    }
    reduce_scan_set(arguments, runner=dag_runner)
    reduce_scan_set(arguments, runner=dag_runner)
    assert len(dag_runner.calls) == 2
    source.write_text("other")
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert len(dag_runner.calls) == 4
    assert not any(node["cache_hit"] for node in report["reduction_dag"]["node_results"])
    file_provenance = [value for value in report["reduction_dag"]["node_results"][0]["provenance"] if value["source_type"] == "data_file"]
    assert file_provenance[0]["source_id"] == str(source)
    assert file_provenance[0]["metadata"]["sha256"]


def test_replay_uses_materialized_cache_and_preserves_target(tmp_path, dag_runner):
    template, inputs = scan_reduction_template(_arguments())
    prior = execute_reduction_dag(
        template, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache", target_node_id=template.nodes[1].node_id,
    )
    replay = replay_reduction_dag_run(
        prior.model_dump(mode="json"), template=template, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache",
    )
    assert len(dag_runner.calls) == 2
    assert replay["cache_hit_node_ids"] == prior.ordered_node_ids
    assert replay["changed_count"] == 0


def test_reviewed_correction_parameters_drive_tool_and_cache(tmp_path, dag_runner):
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache")}
    arguments["steps"][1]["correction_id"] = "monitor"
    arguments["corrections"] = [ReductionCorrectionRef(
        correction_id="monitor", correction_type="monitor_normalization", status="accepted",
        parameters={"target_monitor": 900.0}, version="1",
    ).model_dump(mode="json")]
    reduce_scan_set(arguments, runner=dag_runner)
    assert dag_runner.calls[1].arguments["target_monitor"] == 900.0
    arguments["corrections"][0]["parameters"]["target_monitor"] = 800.0
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert dag_runner.calls[3].arguments["target_monitor"] == 800.0
    assert [node["cache_hit"] for node in report["reduction_dag"]["node_results"]] == [True, False, False]
    arguments["corrections"][0]["status"] = "proposed"
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert report["status"] == "missing_input"
    assert len(dag_runner.calls) == 5


def test_raw_source_cannot_be_output(tmp_path, dag_runner):
    source = tmp_path / "raw.bt7"
    source.write_text("original")
    report = reduce_scan_set({**_arguments(), "input_files": [str(source)], "output_path": str(source)}, runner=dag_runner)
    assert report["validation_failures"][0]["code"] == "raw_input_overwrite"
    assert source.read_text() == "original"
    assert not dag_runner.calls


def test_cached_final_output_can_be_materialized_again(tmp_path, dag_runner):
    output = tmp_path / "reduced.json"
    arguments = {**_arguments(), "cache_dir": str(tmp_path / "cache"), "output_path": str(output)}
    first = reduce_scan_set(arguments, runner=dag_runner)
    output.unlink()
    second = reduce_scan_set(arguments, runner=dag_runner)
    assert json.loads(output.read_text())["output"] == first["output"] == second["output"]
    assert len(dag_runner.calls) == 3


def test_graph_validation_rejects_duplicates_dangling_edges_and_cycles():
    node = ReductionDAGNode(node_id="a", kind="input", operation="input.scan_series")
    with pytest.raises(ValidationError, match="unique"):
        ReductionDAGTemplate(template_id="bad", label="bad", measurement_family="test", nodes=[node, node])
    for target, message in [("missing", "unknown node"), ("a", "cycle")]:
        with pytest.raises(ValidationError, match=message):
            ReductionDAGTemplate(
                template_id="bad", label="bad", measurement_family="test", nodes=[node],
                edges=[ReductionDAGEdge(edge_id="e", source_node_id="a", target_node_id=target)],
            )


def test_forked_graph_reuses_unaffected_branch(tmp_path, dag_runner):
    template, inputs = scan_reduction_template(_arguments())
    root = template.nodes[0]
    left = template.nodes[1]
    right = left.model_copy(deep=True, update={"node_id": "right"})
    join = template.nodes[2].model_copy(deep=True)
    join.tool.input_bindings = {}
    join.input_ids = []
    graph = ReductionDAGTemplate(
        template_id="fork", label="fork", measurement_family="test",
        nodes=[root, left, right, join], default_target_node_id=join.node_id,
        edges=[
            ReductionDAGEdge(edge_id="left", source_node_id=root.node_id, target_node_id=left.node_id, source_terminal_id="scan", target_input_name="scan"),
            ReductionDAGEdge(edge_id="right", source_node_id=root.node_id, target_node_id=right.node_id, source_terminal_id="scan", target_input_name="scan"),
            ReductionDAGEdge(edge_id="signal", source_node_id=left.node_id, target_node_id=join.node_id, source_terminal_id="scan", target_input_name="signal"),
            ReductionDAGEdge(edge_id="bg", source_node_id=right.node_id, target_node_id=join.node_id, source_terminal_id="scan", target_input_name="background"),
        ],
    )
    execute_reduction_dag(graph, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache")
    right.tool.arguments["target_monitor"] = 999.0
    result = execute_reduction_dag(graph, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache")
    assert [node.cache_hit for node in result.node_results] == [True, True, False, False]
    assert len(dag_runner.calls) == 6


def test_noncacheable_node_and_changed_output_invalidate_descendants(tmp_path, dag_runner):
    template, inputs = scan_reduction_template(_arguments())
    template.nodes[1].cacheable = False
    execute_reduction_dag(template, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache")
    result = execute_reduction_dag(template, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache")
    assert [node.cache_hit for node in result.node_results] == [True, False, False]
    assert len(dag_runner.calls) == 5


@pytest.mark.parametrize("output_argument", ["output_path", "output_dir"])
def test_file_writing_tool_call_is_never_cached(tmp_path, dag_runner, output_argument):
    template, inputs = scan_reduction_template(_arguments())
    template.nodes[0].tool.arguments[output_argument] = str(tmp_path / "tool-output.json")
    for _ in range(2):
        result = execute_reduction_dag(template, inputs=inputs, runner=dag_runner, cache_dir=tmp_path / "cache")
        assert result.node_results[0].cache_hit is False
    assert len(dag_runner.calls) == 6


def test_unreadable_source_is_not_cached_even_if_tool_returns_success(tmp_path, dag_runner):
    arguments = {
        "steps": [{"tool": "translate_ice_file", "arguments": {"path": str(tmp_path / "missing.bt7")}}],
        "input_datafile_ids": ["missing"], "in_memory": True, "cache_dir": str(tmp_path / "cache"),
    }
    for _ in range(2):
        report = reduce_scan_set(arguments, runner=dag_runner)
        assert not report["reduction_dag"]["node_results"][0]["cache_hit"]
    assert len(dag_runner.calls) == 2


def test_mutating_handler_cannot_change_raw_inputs_or_recorded_call(dag_runner):
    arguments = _arguments()
    original = deepcopy(arguments)
    registered = dag_runner.registry.get("tas.file_arithmetic", "add_scans")
    handler = registered.handler

    def mutating_handler(call):
        call.arguments["scans"][0]["scan_id"] = "mutated"
        return handler(call)

    registered.handler = mutating_handler
    report = reduce_scan_set(arguments, runner=dag_runner)
    assert arguments == original
    assert report["reduction_dag"]["node_results"][0]["tool_call"]["arguments"]["scans"][0]["scan_id"] == "a"


def test_numerical_replay_round_trips_saved_artifact(tmp_path, dag_runner, monkeypatch):
    from tas_copilot_core.workflows.experiment_dashboard import dag_run_replay
    from tas_copilot_core.workflows import replay_reduction

    output = tmp_path / "reduced.json"
    cache = tmp_path / "cache"
    original = reduce_scan_set({**_arguments(), "output_path": str(output), "cache_dir": str(cache)}, runner=dag_runner)
    monkeypatch.setattr(replay_reduction, "default_runner", lambda: dag_runner)
    report = dag_run_replay({"dag_run_json": str(output), "cache_dir": str(cache), "output_dir": str(tmp_path / "replay")})
    assert report["status"] == "success"
    assert len(report["cache_replay"]["cache_hit_node_ids"]) == 3
    assert len(dag_runner.calls) == 3
    assert report["reduction_dag"]["node_results"][-1]["output"] == original["output"]
    assert (tmp_path / "replay" / "dag_run_replay.json").exists()
