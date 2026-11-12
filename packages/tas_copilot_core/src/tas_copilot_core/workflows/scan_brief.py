from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import AuditRecord, ProvenanceRecord
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.completeness import scan_completeness
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_runner import stable_hash
from tas_copilot_core.reporting import write_report_bundle

from .check_geometry import check_geometry
from .retrieve_similar_scans import retrieve_similar_scans


WORKFLOW_NAME = "scan-brief"


def scan_brief(arguments: dict[str, Any], audit_logger: JsonlAuditLogger | None = None) -> dict[str, Any]:
    audit_logger = audit_logger or JsonlAuditLogger()
    scan_id = str(arguments.get("scan_id") or "")
    db_path = Path(arguments.get("catalog") or arguments.get("db_path") or CoreConfig().data_catalog_db_path)
    limit = int(arguments.get("limit", 5))
    if not scan_id:
        return _missing("missing_scan_id", "scan_id is required", db_path, arguments, audit_logger)
    if not db_path.exists():
        return _missing("missing_database", f"catalog does not exist: {db_path}", db_path, arguments, audit_logger)

    catalog = SQLiteDataCatalog(db_path)
    scan = catalog.get_scan(scan_id)
    if scan is None:
        return _missing("scan_not_found", f"scan id not found in catalog: {scan_id}", db_path, arguments, audit_logger)

    datafiles = catalog.datafiles_for_scan(scan_id)
    completeness = scan_completeness(scan)
    similar = retrieve_similar_scans(scan_id, limit=limit, catalog=db_path)
    geometry = check_geometry({"scan_id": scan_id, "db_path": str(db_path), "geometry_mode": arguments.get("geometry_mode", "auto")}, audit_logger=audit_logger)

    report = {
        "workflow_name": WORKFLOW_NAME,
        "status": _status(completeness, geometry),
        "catalog_path": str(db_path),
        "scan": scan.model_dump(mode="json"),
        "data_files": [datafile.model_dump(mode="json") for datafile in datafiles],
        "metadata_completeness": completeness,
        "similar_scans": similar,
        "geometry_check": geometry,
        "safe_next_actions": _safe_next_actions(completeness, geometry, datafiles),
        "provenance": [
            ProvenanceRecord(source_type="scan", source_id=scan.scan_id, method="sqlite_data_catalog").model_dump(mode="json"),
            ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="catalog_brief_composition").model_dump(mode="json"),
        ],
    }
    report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
    if arguments.get("output_dir"):
        report["artifacts"] = write_report_bundle(
            arguments["output_dir"],
            report,
            markdown_renderer=_markdown_report,
            json_filename=f"{_safe_filename(scan_id)}_scan_brief.json",
            markdown_filename=f"{_safe_filename(scan_id)}_scan_brief.md",
        )
    return report


def _missing(code: str, message: str, db_path: Path, arguments: dict[str, Any], audit_logger: JsonlAuditLogger) -> dict[str, Any]:
    report = {
        "workflow_name": WORKFLOW_NAME,
        "status": "missing_input",
        "catalog_path": str(db_path),
        "errors": [{"code": code, "message": message}],
        "provenance": [ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="input_validation").model_dump(mode="json")],
    }
    report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
    return report


def _status(completeness: dict[str, Any], geometry: dict[str, Any]) -> str:
    if geometry.get("status") == "success":
        return "ok"
    if geometry.get("status") in {"missing_input", "unavailable", "partial"}:
        return "partial"
    if completeness.get("path_only"):
        return "partial"
    return str(geometry.get("status") or "ok")


def _safe_next_actions(completeness: dict[str, Any], geometry: dict[str, Any], datafiles: list[Any]) -> list[str]:
    actions: list[str] = []
    if completeness.get("path_only"):
        actions.append("Hydrate/read the source file contents before using this scan for geometry or resolution decisions.")
    if not datafiles:
        actions.append("Link source data files to this scan before reduction or provenance-sensitive reporting.")
    if geometry.get("status") == "missing_input":
        missing = ", ".join(geometry.get("missing_fields", [])) or "explicit geometry fields"
        actions.append(f"Add {missing} before running geometry, resolution, or spurion checks.")
    if not completeness.get("eligible_for_resolution_check"):
        actions.append("Add explicit Q/E/instrument configuration metadata before resolution or spurion checks.")
    if not actions:
        actions.append("Review the geometry outputs and similar scans before selecting any follow-up analysis workflow.")
    return actions


def _markdown_report(report: dict[str, Any]) -> str:
    scan = report["scan"]
    completeness = report["metadata_completeness"]
    lines = [
        "# TAS Scan Brief",
        "",
        f"- Status: `{report['status']}`",
        f"- Scan: `{scan['scan_id']}`",
        f"- Experiment: `{scan.get('experiment_id')}`",
        f"- Instrument: `{scan.get('instrument')}`",
        f"- Scan type: `{scan.get('scan_type')}`",
        f"- Catalog: `{report['catalog_path']}`",
        f"- Completeness score: `{completeness.get('completeness_score')}`",
        f"- Geometry status: `{report['geometry_check'].get('status')}`",
        "",
        "## Present Metadata",
        "",
    ]
    lines.extend(f"- {field}" for field in completeness.get("present_fields", [])) if completeness.get("present_fields") else lines.append("- None")
    lines.extend(["", "## Missing Metadata", ""])
    lines.extend(f"- {field}" for field in completeness.get("missing_fields", [])) if completeness.get("missing_fields") else lines.append("- None")
    lines.extend(["", "## Data Files", ""])
    lines.extend(f"- `{item['datafile_id']}`: `{item['path']}`" for item in report["data_files"]) if report["data_files"] else lines.append("- None")
    lines.extend(["", "## Similar Scans", ""])
    lines.extend(_similar_lines(report.get("similar_scans", [])))
    lines.extend(["", "## Safe Next Actions", ""])
    lines.extend(f"- {action}" for action in report["safe_next_actions"])
    return "\n".join(lines) + "\n"


def _similar_lines(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return ["- None"]
    return [
        f"- `{item.get('scan_id')}` score={round(float(item.get('similarity_score', 0.0)), 3)} matched={','.join(item.get('matched_dimensions', [])) or 'none'}"
        for item in results[:10]
    ]


def _write_workflow_audit(audit_logger: JsonlAuditLogger | None, report: dict[str, Any], arguments: dict[str, Any]) -> str | None:
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
            "scan_id": (report.get("scan") or {}).get("scan_id"),
            "catalog_path": report.get("catalog_path"),
            "geometry_status": (report.get("geometry_check") or {}).get("status"),
            "provenance": report.get("provenance", []),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "scan"
