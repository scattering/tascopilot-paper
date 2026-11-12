from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, ToolCall, ToolError, ToolResult, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_runner import stable_hash

from .common import default_runner


WORKFLOW_NAME = "check-geometry"


def check_geometry(arguments: dict, runner=None, audit_logger: JsonlAuditLogger | None = None) -> dict:
    audit_logger = audit_logger or getattr(runner, "audit_logger", None) or JsonlAuditLogger()
    arguments = _expand_scan_geometry(arguments)
    mode = _geometry_mode(arguments)
    missing_fields = _missing_geometry_fields(arguments)
    if missing_fields:
        report = {
            "workflow_name": WORKFLOW_NAME,
            "geometry_mode": mode,
            "status": ToolResultStatus.missing_input.value,
            "missing_fields": missing_fields,
            "results": {},
            "validation_failures": arguments.get("_validation_failures", []),
            "provenance": [ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="input_validation").model_dump(mode="json")],
        }
        report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
        return report

    runner = runner or default_runner()
    results = _run_geometry_tools(runner, arguments, mode)
    statuses = [result.status.value for result in results.values()]
    report = {
        "workflow_name": WORKFLOW_NAME,
        "geometry_mode": mode,
        "status": _composite_status(statuses),
        "missing_fields": [],
        "validation_failures": [],
        "results": {name: result.model_dump(mode="json") for name, result in results.items()},
        "tool_audit_ids": {name: result.audit_id for name, result in results.items()},
        "provenance": [ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="registry_tool_calls").model_dump(mode="json")],
    }
    report["workflow_audit_id"] = _write_workflow_audit(getattr(runner, "audit_logger", audit_logger), report, arguments)
    return report


def _run_tool(runner, namespace: str, name: str, arguments: dict) -> ToolResult:
    return runner.run(ToolCall(tool_namespace=namespace, tool_name=name, arguments=arguments, caller_workflow=WORKFLOW_NAME))


def _run_geometry_tools(runner, arguments: dict, mode: str) -> dict[str, ToolResult]:
    if mode == "compatible_ub":
        return {
            "device_angles": _unsupported_compatible_ub_result("tas.device_angles", "calculate_angles", arguments),
            "resolution": _unsupported_compatible_ub_result("tas.resolution", "calculate_resolution", arguments),
            "spurions": _unsupported_compatible_ub_result("tas.spurion", "check_spurions", arguments),
        }
    results = {
        "device_angles": _run_tool(runner, "tas.device_angles", "calculate_angles", arguments),
        "resolution": _run_tool(runner, "tas.resolution", "calculate_resolution", arguments),
        "spurions": _run_tool(runner, "tas.spurion", "check_spurions", arguments),
    }
    if mode == "eulerian_ub":
        results = {
            "ub_matrix": _run_tool(runner, "tas.ubmatrix", "calculate_ub", {"lattice": arguments["lattice"], "ub_id": arguments.get("ub_id")}),
            **results,
        }
    return results


def _unsupported_compatible_ub_result(namespace: str, name: str, arguments: dict) -> ToolResult:
    convention = arguments.get("ub_convention") or "unknown"
    return ToolResult(
        status=ToolResultStatus.unavailable,
        tool_namespace=namespace,
        tool_name=name,
        output={
            "geometry_mode": "compatible_ub",
            "ub_convention": convention,
            "adapter_required": True,
        },
        errors=[
            ToolError(
                code="unsupported_geometry_convention",
                message=(
                    "compatible_ub preserves an external UB convention, but no reviewed adapter is registered "
                    "to convert it into lattice-plane geometry for this tool"
                ),
                details={"ub_convention": str(convention)},
            )
        ],
        warnings=["No lattice-plane or Eulerian-cradle geometry was guessed from the compatible UB matrix."],
        provenance=[
            ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="unsupported_compatible_ub_guard")
        ],
    )


def _selected_tool_names(mode: str) -> list[str]:
    if mode == "compatible_ub":
        return []
    names = [
        "tas.device_angles.calculate_angles",
        "tas.resolution.calculate_resolution",
        "tas.spurion.check_spurions",
    ]
    if mode == "eulerian_ub":
        return ["tas.ubmatrix.calculate_ub", *names]
    return names


def _expand_scan_geometry(arguments: dict) -> dict:
    payload = dict(arguments)
    scan_id = payload.get("scan_id")
    if not scan_id:
        return payload

    db_path = Path(payload.get("db_path") or CoreConfig().data_catalog_db_path)
    if not db_path.exists():
        payload.setdefault("_validation_failures", []).append(
            {
                "code": "catalog_missing",
                "message": f"Scan catalog does not exist: {db_path}",
                "db_path": str(db_path),
            }
        )
        return payload

    scan = SQLiteDataCatalog(db_path).get_scan(str(scan_id))
    if scan is None:
        payload.setdefault("_validation_failures", []).append(
            {
                "code": "scan_not_found",
                "message": f"Scan id not found in catalog: {scan_id}",
                "scan_id": str(scan_id),
                "db_path": str(db_path),
            }
        )
        return payload

    scan_payload = scan.model_dump(mode="json")
    extracted = {
        "q_point": scan_payload.get("q_center"),
        "energy_transfer": scan_payload.get("energy_transfer"),
        "instrument_config": scan_payload.get("instrument_config"),
        "lattice": _lattice_from_scan_metadata(scan_payload.get("metadata") or {}),
        "scattering_plane": _scattering_plane_from_scan_metadata(scan_payload.get("metadata") or {}),
        "ub_matrix": _ub_from_scan_metadata(scan_payload.get("metadata") or {}),
        "ub_convention": (scan_payload.get("metadata") or {}).get("ub_convention"),
    }
    for key, value in extracted.items():
        if value is not None and key not in payload:
            payload[key] = value
    payload["source_scan"] = {
        "scan_id": scan.scan_id,
        "experiment_id": scan.experiment_id,
        "db_path": str(db_path),
    }
    payload.setdefault("provenance_context", []).append(
        ProvenanceRecord(source_type="scan", source_id=scan.scan_id, method="sqlite_data_catalog").model_dump(mode="json")
    )
    return payload


def _lattice_from_scan_metadata(metadata: dict) -> dict | None:
    lattice = metadata.get("lattice")
    if isinstance(lattice, dict):
        return lattice
    geometry = metadata.get("geometry")
    if isinstance(geometry, dict) and isinstance(geometry.get("lattice"), dict):
        return geometry["lattice"]
    return None


def _scattering_plane_from_scan_metadata(metadata: dict) -> dict | None:
    plane = metadata.get("scattering_plane") or metadata.get("orientation_reflections")
    return plane if isinstance(plane, dict) else None


def _ub_from_scan_metadata(metadata: dict):
    return metadata.get("ub_matrix") or metadata.get("ub")


def _missing_geometry_fields(arguments: dict) -> list[str]:
    mode = _geometry_mode(arguments)
    required_paths = [
        ("q_point", "h"),
        ("q_point", "k"),
        ("q_point", "l"),
        ("energy_transfer",),
        ("instrument_config",),
    ]
    if mode in {"lattice", "eulerian_ub"}:
        required_paths.extend(
            [
                ("lattice", "a"),
                ("lattice", "b"),
                ("lattice", "c"),
                ("lattice", "alpha"),
                ("lattice", "beta"),
                ("lattice", "gamma"),
            ]
        )
    if mode == "lattice" and not _has_scattering_plane(arguments):
        required_paths.append(("scattering_plane",))
    if mode == "eulerian_ub":
        required_paths.extend([("lattice", "orient1"), ("lattice", "orient2")])
    if mode == "compatible_ub":
        required_paths.append(("ub_matrix",))
        if not arguments.get("ub_convention") or arguments.get("ub_convention") in {"unknown", "eulerian_cradle"}:
            required_paths.append(("ub_convention",))
    missing = [".".join(path) for path in required_paths if _lookup(arguments, path) is None]
    if arguments.get("scan_id") and arguments.get("_validation_failures"):
        missing.extend(f"scan_id.{item['code']}" for item in arguments["_validation_failures"])
    return missing


def _geometry_mode(arguments: dict) -> str:
    requested = str(arguments.get("geometry_mode") or arguments.get("mode") or "auto").replace("-", "_").lower()
    if requested in {"eulerian", "eulerian_ub"}:
        return "eulerian_ub"
    if requested in {"ub", "compatible_ub", "ub_geometry"}:
        return "compatible_ub"
    if requested in {"lattice", "lattice_angle", "auto"}:
        if requested == "auto" and arguments.get("ub_matrix") is not None and arguments.get("ub_convention") not in {None, "unknown", "eulerian_cradle"} and not _has_scattering_plane(arguments):
            return "compatible_ub"
        return "lattice"
    return requested


def _has_scattering_plane(arguments: dict) -> bool:
    lattice = arguments.get("lattice")
    if isinstance(lattice, dict) and lattice.get("orient1") is not None and lattice.get("orient2") is not None:
        return True
    plane = arguments.get("scattering_plane") or arguments.get("orientation_reflections")
    return bool(plane)


def _lookup(arguments: dict, path: tuple[str, ...]):
    value = arguments
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _composite_status(statuses: list[str]) -> str:
    if all(status == ToolResultStatus.success.value for status in statuses):
        return ToolResultStatus.success.value
    if all(status == ToolResultStatus.unavailable.value for status in statuses):
        return ToolResultStatus.unavailable.value
    if any(status == ToolResultStatus.blocked.value for status in statuses):
        return ToolResultStatus.blocked.value
    if any(status in {ToolResultStatus.success.value, ToolResultStatus.unavailable.value} for status in statuses):
        return "partial"
    return ToolResultStatus.error.value


def _write_workflow_audit(audit_logger, report: dict, arguments: dict) -> str | None:
    if audit_logger is None:
        return None
    audit = AuditRecord(
        event_type="workflow",
        status=report["status"],
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": arguments}),
        output_hash=stable_hash(report),
        details={
            "workflow_name": WORKFLOW_NAME,
            "input_arguments": arguments,
            "tools_selected": _selected_tool_names(report.get("geometry_mode", "lattice")),
            "tool_audit_ids": report.get("tool_audit_ids", {}),
            "missing_fields": report.get("missing_fields", []),
            "validation_failures": report.get("validation_failures", []),
            "provenance": report.get("provenance", []),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id
