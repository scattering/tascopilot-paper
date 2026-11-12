from pathlib import Path

from plugin_paths import plugin_path
from copy import deepcopy

import pytest

from tas_copilot_contracts import PermissionLevel, SideEffectLevel, ToolCall, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.registry import ToolRegistry, ToolRunner
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.workflows.reduce_scan_set import reduce_scan_set


TASREDUCTION_PATH = (plugin_path("tasreduction"))


def _payload(scan_id, counts, monitor=None):
    return {
        "scan_id": scan_id,
        "axis": [0.0, 1.0],
        "counts": counts,
        "errors": [1.0, 1.0],
        "monitor": monitor or [1.0, 1.0],
        "metadata": {"ScanVariable": "a3"},
    }


@pytest.mark.skipif(not TASREDUCTION_PATH.exists(), reason="local tasreduction checkout is not available")
def test_local_tasreduction_plugin_registers_and_runs_through_registry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TASREDUCTION_PATH))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tasreduction.copilot_plugin")
    assert any(plugin.__name__ == "tasreduction.copilot_plugin" for plugin in discover_entrypoint_plugins())
    from tasreduction import copilot_plugin

    registry = ToolRegistry()
    specs = copilot_plugin.get_tool_specs()
    write_capable = {
        "add_scans",
        "subtract_background",
        "normalize_to_monitor",
        "apply_detailed_balance",
        "translate_ice_file",
    }
    for spec in specs:
        if spec.name in write_capable:
            assert PermissionLevel.writes_local_files in spec.permissions_required
            assert spec.side_effect_level == SideEffectLevel.local_files
        registry.register(spec, copilot_plugin.run_tool)

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.file_arithmetic",
            tool_name="add_scans",
            arguments={
                "scans": [
                    _payload("scan-a", [100.0, 120.0]),
                    _payload("scan-b", [200.0, 220.0], [2.0, 2.0]),
                ],
                "input_datafile_ids": ["df-a", "df-b"],
                "in_memory": True,
            },
            caller_workflow="test",
        )
    )

    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert result.output["lineage"]["input_ids"] == ["scan-a", "scan-b"]
    assert "Combined 2 scans" in result.output["scan"]["treatments"][-1]


@pytest.mark.skipif(not TASREDUCTION_PATH.exists(), reason="local tasreduction checkout is not available")
def test_numerical_dag_matches_direct_plugin_chain_and_replays(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TASREDUCTION_PATH))
    from tasreduction import copilot_plugin

    calls = []

    def handler(call):
        calls.append(call)
        return copilot_plugin.run_tool(call)

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, handler)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    scans = [_payload("a", [100, 200], [10, 20]), _payload("b", [40, 80], [5, 10])]
    background = _payload("bg", [4, 8], [1, 1])
    for scan in [*scans, background]:
        scan["metadata"] = {"ScanVariable": "E", "E": [1.0, 2.0], "Temp": [100.0, 100.0]}
    arguments = {
        "steps": [
            {"tool": "add_scans", "arguments": {"scans": scans}},
            {"tool": "normalize_to_monitor", "arguments": {"target_monitor": 100.0}},
            {"tool": "subtract_background", "arguments": {"background": background}},
            {"tool": "apply_detailed_balance"},
        ],
        "input_datafile_ids": ["a", "b", "bg"],
        "in_memory": True, "cache_dir": str(tmp_path / "cache"),
    }
    original = deepcopy(arguments)
    dag = reduce_scan_set(arguments, runner=runner)
    assert dag["status"] == "success", dag
    assert len(calls) == 4
    direct_scan = None
    for step in arguments["steps"]:
        name = step["tool"]
        namespace = "tas.file_arithmetic" if name in {"add_scans", "subtract_background"} else "tas.corrections"
        parameters = deepcopy(step.get("arguments", {}))
        if direct_scan is not None:
            parameters["signal" if name == "subtract_background" else "scan"] = direct_scan
        result = copilot_plugin.run_tool(ToolCall(tool_namespace=namespace, tool_name=name, arguments=parameters))
        assert result.status == ToolResultStatus.success, result
        direct_scan = result.output["scan"]
    for field in ("axis", "counts", "errors", "monitor"):
        assert dag["output"]["scan"][field] == pytest.approx(direct_scan[field], rel=1e-13)
    assert dag["output"]["scan"]["treatments"] == direct_scan["treatments"]
    assert arguments == original
    replay = reduce_scan_set(arguments, runner=runner)
    assert len(calls) == 4
    assert replay["output"] == dag["output"]
    assert all(node["cache_hit"] for node in replay["reduction_dag"]["node_results"])
    arguments["steps"][2]["arguments"]["background"]["counts"][0] = 6
    changed = reduce_scan_set(arguments, runner=runner)
    assert len(calls) == 6
    assert changed["output"]["scan"]["counts"] != dag["output"]["scan"]["counts"]
    assert [node["cache_hit"] for node in changed["reduction_dag"]["node_results"]] == [True, True, False, False]
