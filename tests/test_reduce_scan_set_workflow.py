from tas_copilot_contracts import ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.workflows.reduce_scan_set import reduce_scan_set


def test_reduce_scan_set_requires_lineage_and_output(tmp_path):
    result = reduce_scan_set({"steps": [{"tool": "add_scans"}]}, audit_logger=JsonlAuditLogger(tmp_path / "audit"))

    assert result["status"] == ToolResultStatus.missing_input.value
    codes = {item["code"] for item in result["validation_failures"]}
    assert {"missing_input_lineage", "missing_output_artifact"} <= codes
    assert result["workflow_audit_id"]


def test_reduce_scan_set_routes_supported_steps_with_lineage(tmp_path, dag_runner):
    runner = dag_runner
    result = reduce_scan_set(
        {
            "steps": [
                {"tool": "add_scans", "arguments": {"scans": [{"scan_id": "a"}, {"scan_id": "b"}]}},
                {"tool": "normalize_to_monitor", "arguments": {"target_monitor": 1.0}},
                {"tool": "apply_detailed_balance", "arguments": {"temperature": {"value": 10, "unit": "K"}}},
            ],
            "input_datafile_ids": ["df-a", "df-b"],
            "output_path": str(tmp_path / "reduced.json"),
        },
        runner=runner,
    )

    assert result["status"] == ToolResultStatus.success.value
    assert [call.tool_name for call in runner.calls] == ["add_scans", "normalize_to_monitor", "apply_detailed_balance"]
    assert all(call.arguments["input_datafile_ids"] == ["df-a", "df-b"] for call in runner.calls)
    assert all(call.arguments["in_memory"] is True for call in runner.calls)
    assert all("output_path" not in call.arguments for call in runner.calls)
    assert runner.calls[1].arguments["scan"]["marker"] == 1
    assert runner.calls[2].arguments["scan"]["marker"] == 2
    assert (tmp_path / "reduced.json").exists()
    assert result["reduction_dag"]["status"] == "success"
    assert result["lineage"]["input_datafile_ids"] == ["df-a", "df-b"]


def test_reduce_scan_set_reports_unsupported_step(tmp_path):
    result = reduce_scan_set(
        {"steps": [{"tool": "unknown"}], "input_files": ["a.bt7"], "output_path": "out.json"},
        audit_logger=JsonlAuditLogger(tmp_path / "audit"),
    )

    assert result["status"] == ToolResultStatus.missing_input.value
    assert result["validation_failures"][0]["code"] == "unsupported_step"
