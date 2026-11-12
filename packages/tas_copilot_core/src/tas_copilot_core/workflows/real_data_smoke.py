from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.completeness import eligible_scans_report, metadata_completeness_report
from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.registry.tool_inventory import build_tool_inventory
from tas_copilot_core.registry.tool_runner import stable_hash

from .catalog_tas_directory import catalog_tas_directory
from .check_geometry import check_geometry
from .inspect_bt7_directory import inspect_bt7_directory
from .retrieve_similar_scans import retrieve_similar_scans


WORKFLOW_NAME = "real-data-smoke"


def real_data_smoke(arguments: dict, audit_logger: JsonlAuditLogger | None = None) -> dict[str, Any]:
    path = arguments.get("path")
    catalog_path = arguments.get("catalog") or arguments.get("db_path")
    output_dir = arguments.get("output_dir")
    if not path or not catalog_path or not output_dir:
        return {
            "workflow_name": WORKFLOW_NAME,
            "status": ToolResultStatus.missing_input.value,
            "errors": [{"code": "missing_required_argument", "message": "path, catalog, and output_dir are required"}],
        }

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    audit_logger = audit_logger or JsonlAuditLogger(output_path / "audit")

    inventory = build_tool_inventory().as_dict()
    inspect_args = {
        "path": path,
        **{key: arguments[key] for key in ["max_files", "max_metadata_files", "file_paths", "files", "paths", "path_only"] if key in arguments},
    }
    inspect_result = inspect_bt7_directory(inspect_args, audit_logger=audit_logger)

    catalog_args = {
        "path": path,
        "db_path": str(catalog_path),
        **{key: arguments[key] for key in ["max_files", "max_records", "skip_bt7_enrichment", "file_paths", "files", "paths", "path_only"] if key in arguments},
    }
    catalog_result = catalog_tas_directory(catalog_args)
    quality = data_catalog_quality_report(catalog_path)
    completeness = metadata_completeness_report(catalog_path)
    eligible = eligible_scans_report(catalog_path, purpose="geometry")
    eligible_compatible_ub = eligible_scans_report(catalog_path, purpose="compatible_ub")
    eligible_eulerian_ub = eligible_scans_report(catalog_path, purpose="eulerian_ub")

    similar = _similarity_probe(catalog_path, completeness)
    geometry = _geometry_probe(catalog_path, eligible, eligible_compatible_ub, eligible_eulerian_ub, completeness, audit_logger)

    status = _status([inspect_result.get("status"), catalog_result.get("status"), geometry.get("status")])
    report = {
        "workflow_name": WORKFLOW_NAME,
        "status": status,
        "input_path": str(Path(path).expanduser()),
        "catalog_path": str(Path(catalog_path).expanduser()),
        "output_dir": str(output_path),
        "tools_discovered": inventory["summary"],
        "tools_used": [
            "tas.data_reader.scan_tas_directory",
            "tas.data_reader.classify_tas_format",
            "tas.data_reader.read_tas_metadata",
            "tas.scan_grouping.group_bt7_scans",
            "tas.ubmatrix.calculate_ub",
            "tas.device_angles.calculate_angles",
            "tas.resolution.calculate_resolution",
            "tas.spurion.check_spurions",
        ],
        "unavailable_tools": _tool_names(inventory, "unavailable"),
        "disabled_tools": _tool_names(inventory, "disabled"),
        "inspect": inspect_result,
        "catalog": catalog_result,
        "catalog_quality": quality,
        "metadata_completeness": completeness,
        "eligible_scans": eligible,
        "eligible_compatible_ub_scans": eligible_compatible_ub,
        "eligible_eulerian_ub_scans": eligible_eulerian_ub,
        "similar_scan_retrieval": similar,
        "geometry_check": geometry,
        "audit_log_dir": str(output_path / "audit"),
        "warnings": _warnings(inspect_result, catalog_result, quality, completeness, eligible, geometry),
        "next_recommended_action": _next_action(eligible, geometry),
        "provenance": [ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="vertical_slice_orchestration").model_dump(mode="json")],
    }
    report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
    report["artifacts"] = write_report_bundle(output_path, report, markdown_renderer=_markdown_report)
    return report


def _similarity_probe(catalog_path: str | Path, completeness: dict[str, Any]) -> dict[str, Any]:
    scans = completeness.get("scans", [])
    if len(scans) < 2:
        return {
            "status": ToolResultStatus.missing_input.value,
            "reason": "similar-scan retrieval requires at least two cataloged scans",
            "inspected_scan_count": len(scans),
        }
    reference = sorted(scans, key=lambda item: item["completeness_score"], reverse=True)[0]["scan_id"]
    return {
        "status": "ok",
        "reference_scan_id": reference,
        "results": retrieve_similar_scans(reference, limit=5, catalog=str(catalog_path)),
    }


def _geometry_probe(
    catalog_path: str | Path,
    eligible_lattice: dict[str, Any],
    eligible_compatible_ub: dict[str, Any],
    eligible_eulerian_ub: dict[str, Any],
    completeness: dict[str, Any],
    audit_logger,
) -> dict[str, Any]:
    mode, scan_id = _select_geometry_candidate(eligible_lattice, eligible_compatible_ub, eligible_eulerian_ub)
    if not scan_id:
        missing_counts: dict[str, int] = {}
        for item in completeness.get("scans", []):
            for field in item.get("geometry_missing_fields", []):
                missing_counts[field] = missing_counts.get(field, 0) + 1
            for field in item.get("compatible_ub_missing_fields", []):
                missing_counts[f"compatible_ub.{field}"] = missing_counts.get(f"compatible_ub.{field}", 0) + 1
        alignment_rejections = sum(1 for item in completeness.get("scans", []) if item.get("scan_role") == "alignment")
        payload = {
            "status": ToolResultStatus.missing_input.value,
            "reason": "no eligible scan found; no Q/E/config/lattice values were guessed",
            "inspected_scan_count": len(completeness.get("scans", [])),
            "missing_field_counts": missing_counts,
            "available_metadata_fields": _available_metadata_fields(completeness),
            "suggested_next_metadata_inputs": ["q_point", "energy_transfer", "instrument_config", "lattice_parameters", "scattering_plane_or_ub"],
        }
        if alignment_rejections:
            payload["alignment_scan_count"] = alignment_rejections
            payload["reason"] = "no eligible geometry scan found; alignment scans may lack scattering-plane or compatible UB metadata"
            payload["suggested_next_metadata_inputs"].append("select_non_alignment_scan_if_available")
        return payload
    return check_geometry({"scan_id": scan_id, "db_path": str(catalog_path), "geometry_mode": mode}, audit_logger=audit_logger)


def _select_geometry_candidate(
    eligible_lattice: dict[str, Any],
    eligible_compatible_ub: dict[str, Any],
    eligible_eulerian_ub: dict[str, Any],
) -> tuple[str, str | None]:
    if eligible_lattice.get("best_candidate"):
        return "lattice", eligible_lattice["best_candidate"]
    if eligible_compatible_ub.get("best_candidate"):
        return "compatible_ub", eligible_compatible_ub["best_candidate"]
    if eligible_eulerian_ub.get("best_candidate"):
        return "eulerian_ub", eligible_eulerian_ub["best_candidate"]
    return "lattice", None


def _available_metadata_fields(completeness: dict[str, Any]) -> list[str]:
    fields: set[str] = set()
    for scan in completeness.get("scans", []):
        fields.update(scan.get("available_metadata_fields", []))
    return sorted(fields)


def _tool_names(inventory: dict[str, Any], status: str) -> list[str]:
    return [
        f"{tool['namespace']}.{tool['name']}"
        for tool in inventory.get("tools", [])
        if tool.get("effective") and tool.get("status") == status
    ]


def _status(statuses: list[str | None]) -> str:
    if any(status == ToolResultStatus.error.value for status in statuses):
        return ToolResultStatus.error.value
    if any(status == ToolResultStatus.success.value for status in statuses):
        return "partial" if any(status == ToolResultStatus.missing_input.value for status in statuses) else ToolResultStatus.success.value
    if any(status == "ok" for status in statuses):
        return "partial"
    return ToolResultStatus.missing_input.value


def _warnings(*sections: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for section in sections:
        for warning in section.get("warnings", []) or []:
            warnings.append(str(warning))
        if section.get("status") == ToolResultStatus.missing_input.value and section.get("reason"):
            warnings.append(str(section["reason"]))
    return warnings


def _next_action(eligible: dict[str, Any], geometry: dict[str, Any]) -> str:
    if eligible.get("best_candidate") and geometry.get("status") == ToolResultStatus.success.value:
        return "Review geometry outputs and compare against known instrument metadata before expanding the scan set."
    if not eligible.get("best_candidate"):
        return "Enrich reader metadata with q_point, energy_transfer, instrument_config, and lattice_parameters, then rerun real-data-smoke."
    return "Install or configure geometry plugins, then rerun check-geometry for the selected scan."


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TAS Real-Data Smoke Report",
        "",
        f"- Status: {report['status']}",
        f"- Input path: `{report['input_path']}`",
        f"- Catalog: `{report['catalog_path']}`",
        f"- Output directory: `{report['output_dir']}`",
        f"- Audit log directory: `{report['audit_log_dir']}`",
        f"- Workflow audit ID: `{report.get('workflow_audit_id')}`",
        "",
        "## Catalog",
        "",
        f"- Imported: `{report['catalog'].get('imported', {})}`",
        f"- Quality: `{report['catalog_quality'].get('status')}`",
        f"- Completeness summary: `{report['metadata_completeness'].get('summary', {})}`",
        "",
        "## Geometry",
        "",
        f"- Eligible scan: `{report['eligible_scans'].get('best_candidate')}`",
        f"- Geometry status: `{report['geometry_check'].get('status')}`",
        f"- Geometry reason: `{report['geometry_check'].get('reason', '')}`",
        "",
        "## Next Action",
        "",
        report["next_recommended_action"],
    ]
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _write_workflow_audit(audit_logger, report: dict[str, Any], arguments: dict) -> str | None:
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
            "tools_selected": report["tools_used"],
            "output_artifacts": [str(Path(report["output_dir"]) / "report.json"), str(Path(report["output_dir"]) / "report.md")],
            "provenance": report.get("provenance", []),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id
