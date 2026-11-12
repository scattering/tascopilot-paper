from __future__ import annotations

from tas_copilot_contracts import ProvenanceRecord, ToolCall, ToolError, ToolResult, ToolResultStatus

from .common import default_runner


def composite_status(results: list[ToolResult]) -> ToolResultStatus:
    statuses = [result.status for result in results]
    if statuses and all(status == ToolResultStatus.success for status in statuses):
        return ToolResultStatus.success
    if statuses and all(status == ToolResultStatus.unavailable for status in statuses):
        return ToolResultStatus.unavailable
    if any(status == ToolResultStatus.blocked for status in statuses):
        return ToolResultStatus.blocked
    if any(status == ToolResultStatus.success for status in statuses):
        return ToolResultStatus.success
    if any(status == ToolResultStatus.validation_failed for status in statuses):
        return ToolResultStatus.validation_failed
    return statuses[0] if statuses else ToolResultStatus.missing_input


def read_bt7_data(arguments: dict, runner=None) -> ToolResult:
    path = arguments.get("path") or arguments.get("file_path")
    if not path:
        return _missing("read_bt7_data", "tas.data_reader", "path is required")
    runner = runner or default_runner()
    calls = [
        ("format", ToolCall(tool_namespace="tas.data_reader", tool_name="classify_tas_format", arguments={"path": path}, caller_workflow="read-bt7-data")),
        ("metadata", ToolCall(tool_namespace="tas.data_reader", tool_name="read_tas_metadata", arguments={"path": path}, caller_workflow="read-bt7-data")),
        ("bt7_file", ToolCall(tool_namespace="tas.data_reader", tool_name="read_bt7_file", arguments={"path": path}, caller_workflow="read-bt7-data")),
    ]
    results = {name: runner.run(call) for name, call in calls}
    return _composite_result("read_bt7_data", "tas.data_reader", "Read BT-7 data compatibility workflow completed.", results)


def combine_files(arguments: dict, runner=None) -> ToolResult:
    operation = arguments.get("operation")
    if operation not in {"add_scans", "subtract_background"}:
        return _missing("combine_files", "tas.file_arithmetic", "operation must be 'add_scans' or 'subtract_background'")
    runner = runner or default_runner()
    tool_name = operation
    result = runner.run(ToolCall(tool_namespace="tas.file_arithmetic", tool_name=tool_name, arguments=arguments, caller_workflow="combine-files"))
    return _composite_result("combine_files", "tas.file_arithmetic", f"combine_files dispatched to {tool_name}.", {tool_name: result})


def apply_corrections(arguments: dict, runner=None) -> ToolResult:
    steps = arguments.get("steps") or []
    if not steps:
        return _missing("apply_corrections", "tas.corrections", "steps is required; supported tools are normalize_to_monitor and apply_detailed_balance")
    supported = {"normalize_to_monitor", "apply_detailed_balance"}
    for step in steps:
        name = step.get("tool") if isinstance(step, dict) else str(step)
        if name not in supported:
            return ToolResult(
                status=ToolResultStatus.validation_failed,
                errors=[ToolError(code="unsupported_correction_step", message=f"Unsupported correction step {name}")],
                tool_name="apply_corrections",
                tool_namespace="tas.corrections",
            )
    from .reduce_scan_set import reduce_scan_set

    report = reduce_scan_set({**arguments, "in_memory": True}, runner=runner)
    results = {}
    for key, value in report["results"].items():
        tool_result = ToolResult.model_validate(value)
        name = tool_result.tool_name
        results[name if name not in results else key] = tool_result
    result = _composite_result("apply_corrections", "tas.corrections", "Correction DAG completed.", results)
    result.status = ToolResultStatus(report["status"])
    result.output.update({
        "reduction_dag": report.get("reduction_dag"),
        "replay": report.get("replay"),
        "validation_failures": report["validation_failures"],
        **report.get("output", {}),
    })
    return result


def _composite_result(tool_name: str, namespace: str, summary: str, results: dict[str, ToolResult]) -> ToolResult:
    status = composite_status(list(results.values()))
    return ToolResult(
        status=status,
        output={
            "summary": summary,
            "results": {name: result.model_dump(mode="json") for name, result in results.items()},
            "tool_audit_ids": {name: result.audit_id for name, result in results.items()},
        },
        warnings=["Composite workflow only; scientific operations remain owned by external tool plugins."],
        provenance=[ProvenanceRecord(source_type="workflow", source_id=f"{namespace}.{tool_name}", method="composite_registry_calls")],
        tool_name=tool_name,
        tool_namespace=namespace,
    )


def _missing(tool_name: str, namespace: str, message: str) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.missing_input,
        errors=[ToolError(code="missing_input", message=message)],
        provenance=[ProvenanceRecord(source_type="workflow", source_id=f"{namespace}.{tool_name}", method="input_validation")],
        tool_name=tool_name,
        tool_namespace=namespace,
    )
