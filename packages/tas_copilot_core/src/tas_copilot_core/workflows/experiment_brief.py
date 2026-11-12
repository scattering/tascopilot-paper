from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, Scan
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.completeness import scan_completeness
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_runner import stable_hash
from tas_copilot_core.reporting import write_report_bundle

from .retrieve_similar_scans import retrieve_similar_scans


WORKFLOW_NAME = "experiment-brief"


def experiment_brief(arguments: dict[str, Any], audit_logger: JsonlAuditLogger | None = None) -> dict[str, Any]:
    audit_logger = audit_logger or JsonlAuditLogger()
    experiment_id = str(arguments.get("experiment_id") or "")
    db_path = Path(arguments.get("catalog") or arguments.get("db_path") or CoreConfig().data_catalog_db_path)
    limit = int(arguments.get("limit", 10))
    if not experiment_id:
        return _missing("missing_experiment_id", "experiment_id is required", db_path, arguments, audit_logger)
    if not db_path.exists():
        return _missing("missing_database", f"catalog does not exist: {db_path}", db_path, arguments, audit_logger)

    catalog = SQLiteDataCatalog(db_path)
    experiment = next((item for item in catalog.list_experiments() if item.experiment_id == experiment_id), None)
    if experiment is None:
        return _missing("experiment_not_found", f"experiment id not found in catalog: {experiment_id}", db_path, arguments, audit_logger)

    scans = [scan for scan in catalog.list_scans() if scan.experiment_id == experiment_id]
    datafiles = [datafile for datafile in catalog.list_datafiles() if datafile.experiment_id == experiment_id]
    links = [link for link in catalog.list_paper_experiment_links() if link.experiment_id == experiment_id]
    completeness = [scan_completeness(scan) for scan in scans]
    best_scan = _best_scan(scans, completeness)
    similar = retrieve_similar_scans(best_scan.scan_id, limit=limit, catalog=db_path) if best_scan else []

    report = {
        "workflow_name": WORKFLOW_NAME,
        "status": "ok" if scans or datafiles else "partial",
        "catalog_path": str(db_path),
        "experiment": experiment.model_dump(mode="json"),
        "scan_count": len(scans),
        "data_file_count": len(datafiles),
        "paper_mapping_count": len(links),
        "scans": [_scan_summary(scan, item) for scan, item in zip(scans, completeness)],
        "data_files": [datafile.model_dump(mode="json") for datafile in datafiles[:limit]],
        "paper_experiment_links": [link.model_dump(mode="json") for link in links[:limit]],
        "metadata_completeness_summary": _completeness_summary(completeness),
        "best_scan_id": best_scan.scan_id if best_scan else None,
        "similar_scans_for_best_scan": similar,
        "safe_next_actions": _safe_next_actions(scans, datafiles, completeness, links),
        "provenance": [
            ProvenanceRecord(source_type="experiment", source_id=experiment.experiment_id, method="sqlite_data_catalog").model_dump(mode="json"),
            ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="catalog_brief_composition").model_dump(mode="json"),
        ],
    }
    report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
    if arguments.get("output_dir"):
        report["artifacts"] = write_report_bundle(
            arguments["output_dir"],
            report,
            markdown_renderer=_markdown_report,
            json_filename=f"{_safe_filename(experiment_id)}_experiment_brief.json",
            markdown_filename=f"{_safe_filename(experiment_id)}_experiment_brief.md",
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


def _best_scan(scans: list[Scan], completeness: list[dict[str, Any]]) -> Scan | None:
    if not scans:
        return None
    scores = {item["scan_id"]: item["completeness_score"] for item in completeness}
    return sorted(scans, key=lambda scan: scores.get(scan.scan_id, 0.0), reverse=True)[0]


def _scan_summary(scan: Scan, completeness: dict[str, Any]) -> dict[str, Any]:
    return {
        "scan_id": scan.scan_id,
        "instrument": scan.instrument,
        "scan_type": scan.scan_type,
        "physics_target": scan.physics_target,
        "completeness_score": completeness["completeness_score"],
        "eligible_for_geometry_check": completeness["eligible_for_geometry_check"],
        "eligible_for_resolution_check": completeness["eligible_for_resolution_check"],
        "missing_fields": completeness["missing_fields"],
        "reason": completeness["reason"],
    }


def _completeness_summary(completeness: list[dict[str, Any]]) -> dict[str, Any]:
    if not completeness:
        return {"scan_count": 0, "average_completeness_score": 0.0}
    return {
        "scan_count": len(completeness),
        "eligible_for_geometry_check": sum(1 for item in completeness if item["eligible_for_geometry_check"]),
        "eligible_for_resolution_check": sum(1 for item in completeness if item["eligible_for_resolution_check"]),
        "path_only_scans": sum(1 for item in completeness if item["path_only"]),
        "average_completeness_score": round(sum(item["completeness_score"] for item in completeness) / len(completeness), 3),
    }


def _safe_next_actions(scans: list[Scan], datafiles: list[Any], completeness: list[dict[str, Any]], links: list[Any]) -> list[str]:
    actions: list[str] = []
    if not scans:
        actions.append("Import or link scans for this experiment before running scan-level analysis workflows.")
    if not datafiles:
        actions.append("Link source data files to the experiment for provenance-sensitive reporting.")
    if completeness and not any(item["eligible_for_geometry_check"] for item in completeness):
        actions.append("Add explicit Q/E/config/lattice/scattering-plane or compatible UB metadata before geometry checks.")
    if completeness and any(item["path_only"] for item in completeness):
        actions.append("Hydrate/read path-only files before relying on header-derived metadata.")
    if not links:
        actions.append("Run or import paper-to-experiment mapping candidates if publication linkage is needed.")
    if not actions:
        actions.append("Use the best scan brief to inspect geometry readiness, similar scans, and reduction candidates.")
    return actions


def _markdown_report(report: dict[str, Any]) -> str:
    experiment = report["experiment"]
    lines = [
        "# TAS Experiment Brief",
        "",
        f"- Status: `{report['status']}`",
        f"- Experiment: `{experiment['experiment_id']}`",
        f"- Title: `{experiment.get('title')}`",
        f"- Instrument: `{experiment.get('instrument')}`",
        f"- Catalog: `{report['catalog_path']}`",
        f"- Scans: `{report['scan_count']}`",
        f"- Data files: `{report['data_file_count']}`",
        f"- Paper mappings: `{report['paper_mapping_count']}`",
        f"- Best scan: `{report.get('best_scan_id')}`",
        f"- Completeness summary: `{report['metadata_completeness_summary']}`",
        "",
        "## Scans",
        "",
    ]
    lines.extend(
        f"- `{item['scan_id']}` score={item['completeness_score']} type={item.get('scan_type') or ''} eligible_geometry={item['eligible_for_geometry_check']}"
        for item in report["scans"][:20]
    ) if report["scans"] else lines.append("- None")
    lines.extend(["", "## Paper Links", ""])
    lines.extend(
        f"- `{item['paper_id']}` confidence={item['confidence']} status={item['status']}" for item in report["paper_experiment_links"]
    ) if report["paper_experiment_links"] else lines.append("- None")
    lines.extend(["", "## Similar Scans For Best Scan", ""])
    lines.extend(_similar_lines(report.get("similar_scans_for_best_scan", [])))
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
            "experiment_id": (report.get("experiment") or {}).get("experiment_id"),
            "catalog_path": report.get("catalog_path"),
            "scan_count": report.get("scan_count", 0),
            "provenance": report.get("provenance", []),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "experiment"
