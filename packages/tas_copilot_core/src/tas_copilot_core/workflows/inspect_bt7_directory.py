from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import AuditRecord, ToolCall, ToolResult

from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_runner import ToolRunner, stable_hash

from .common import default_runner


def inspect_bt7_directory(arguments: dict, runner: ToolRunner | None = None, audit_logger: JsonlAuditLogger | None = None) -> dict:
    """Inspect a TAS/BT-7 directory by composing registry tools only."""

    path = arguments.get("path") or arguments.get("root")
    if not path:
        return {
            "workflow_name": "inspect-bt7-directory",
            "status": "missing_input",
            "errors": [{"code": "missing_path", "message": "path is required"}],
        }

    path = str(Path(path).expanduser())
    file_paths = _file_paths(arguments)
    runner = runner or default_runner()
    audit_logger = audit_logger or JsonlAuditLogger(CoreConfig().audit_dir)
    max_files = arguments.get("max_files")
    max_metadata_files = int(arguments.get("max_metadata_files", 25))
    results: dict[str, Any] = {}
    warnings: list[str] = []

    scan_result = _run_tool(
        runner,
        "tas.data_reader",
        "scan_tas_directory",
        {
            "root": path,
            **({"paths": file_paths} if file_paths else {}),
            **({"max_files": max_files} if max_files is not None else {}),
            **({"path_only": bool(arguments["path_only"])} if "path_only" in arguments else {}),
        },
    )
    results["scan_tas_directory"] = _dump_result(scan_result)
    discovered_files = _paths_from_scan_result(scan_result)

    classifications = []
    metadata = []
    for file_path in discovered_files[:max_metadata_files]:
        classifications.append(
            _dump_result(
                _run_tool(
                    runner,
                    "tas.data_reader",
                    "classify_tas_format",
                    {"path": file_path},
                )
            )
        )
        metadata.append(
            _dump_result(
                _run_tool(
                    runner,
                    "tas.data_reader",
                    "read_tas_metadata",
                    {"path": file_path, **({"path_only": bool(arguments["path_only"])} if "path_only" in arguments else {})},
                )
            )
        )
    if len(discovered_files) > max_metadata_files:
        warnings.append(f"metadata/classification limited to first {max_metadata_files} discovered files")
    results["classify_tas_format"] = classifications
    results["read_tas_metadata"] = metadata

    group_args = {
        "root_dirs": [path],
        "recursive": arguments.get("recursive", True),
    }
    for key in ["file_extensions", "q_tolerance", "e_tolerance", "temp_tolerance", "group_by_experiment", "separate_alignment"]:
        if key in arguments:
            group_args[key] = arguments[key]
    group_result = _run_tool(runner, "tas.scan_grouping", "group_bt7_scans", group_args)
    results["group_bt7_scans"] = _dump_result(group_result)

    status = _workflow_status(
        [scan_result.status.value, group_result.status.value] + [item.get("status", "error") for item in classifications + metadata]
    )
    output = {
        "workflow_name": "inspect-bt7-directory",
        "status": status,
        "path": path,
        "discovered_file_count": len(discovered_files),
        "discovered_files": discovered_files,
        "format_counts": (scan_result.output or {}).get("format_counts") if scan_result.output else None,
        "instrument_counts": (scan_result.output or {}).get("instrument_counts") if scan_result.output else None,
        "group_count": (group_result.output or {}).get("group_count") if group_result.output else None,
        "warnings": warnings,
        "tool_results": results,
        "audit_ids": _audit_ids(results),
        "provenance_ids": _provenance_ids(results),
    }
    audit = AuditRecord(
        event_type="workflow",
        status=status,
        input_hash=stable_hash(arguments),
        output_hash=stable_hash(output),
        details={
            "workflow_name": "inspect-bt7-directory",
            "tools_selected": [
                "tas.data_reader.scan_tas_directory",
                "tas.data_reader.classify_tas_format",
                "tas.data_reader.read_tas_metadata",
                "tas.scan_grouping.group_bt7_scans",
            ],
        },
    )
    audit_logger.write(audit)
    output["workflow_audit_id"] = audit.audit_id
    return output


def _run_tool(runner: ToolRunner, namespace: str, name: str, arguments: dict) -> ToolResult:
    return runner.run(ToolCall(tool_namespace=namespace, tool_name=name, arguments=arguments, caller_workflow="inspect-bt7-directory"))


def _file_paths(arguments: dict) -> list[str]:
    paths = arguments.get("file_paths") or arguments.get("files") or arguments.get("paths") or []
    return [str(Path(path).expanduser()) for path in paths]


def _dump_result(result: ToolResult) -> dict:
    return result.model_dump(mode="json")


def _paths_from_scan_result(result: ToolResult) -> list[str]:
    if result.status.value != "success" or not result.output:
        return []
    paths = []
    for record in result.output.get("records", []):
        if record.get("path"):
            paths.append(str(record["path"]))
    return paths


def _workflow_status(statuses: list[str]) -> str:
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if statuses and all(status == "unavailable" for status in statuses):
        return "unavailable"
    if any(status == "success" for status in statuses):
        return "partial"
    if any(status == "missing_input" for status in statuses):
        return "missing_input"
    if any(status == "blocked" for status in statuses):
        return "blocked"
    return statuses[0] if statuses else "unavailable"


def _audit_ids(results: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for payload in _flatten_results(results):
        audit_id = payload.get("audit_id")
        if audit_id:
            ids.append(audit_id)
    return ids


def _provenance_ids(results: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for payload in _flatten_results(results):
        for item in payload.get("provenance", []):
            source_id = item.get("source_id")
            if source_id:
                ids.append(source_id)
    return ids


def _flatten_results(value: Any) -> list[dict]:
    if isinstance(value, dict) and "status" in value and "tool_name" in value:
        return [value]
    if isinstance(value, dict):
        flattened: list[dict] = []
        for item in value.values():
            flattened.extend(_flatten_results(item))
        return flattened
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_results(item))
        return flattened
    return []
