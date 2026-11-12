from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import DataFile, Experiment, ProvenanceRecord, Scan
from tas_copilot_core.catalogs.completeness import metadata_completeness_report
from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_ontology
from tas_copilot_core.ontology.linker import match_concepts
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.retrieval.keyword import tokenize


WORKFLOW_NAME = "catalog-report"


def catalog_report(arguments: dict[str, Any]) -> dict[str, Any]:
    db_path = Path(arguments.get("catalog") or arguments.get("db_path") or CoreConfig().data_catalog_db_path)
    query = str(arguments.get("query") or "")
    limit = int(arguments.get("limit", 10))
    if not db_path.exists():
        return {
            "workflow_name": WORKFLOW_NAME,
            "status": "missing_input",
            "errors": [{"code": "missing_database", "message": f"catalog does not exist: {db_path}"}],
        }

    catalog = SQLiteDataCatalog(db_path)
    concepts = load_ontology(CoreConfig().ontology_path)
    query_concepts = match_concepts(query, concepts) if query else []
    query_terms = set(tokenize(query))

    experiment_results = _ranked_records("experiment", catalog.list_experiments(), query_terms, query_concepts, limit)
    scan_results = _ranked_records("scan", catalog.list_scans(), query_terms, query_concepts, limit)
    datafile_results = _ranked_records("data_file", catalog.list_datafiles(), query_terms, query_concepts, limit)

    quality = data_catalog_quality_report(db_path)
    completeness = metadata_completeness_report(db_path)
    report = {
        "workflow_name": WORKFLOW_NAME,
        "status": "ok",
        "query": query,
        "catalog_path": str(db_path),
        "matched_concepts": [_concept_summary(concept) for concept in query_concepts],
        "catalog_summary": catalog.summary(),
        "quality_status": quality["status"],
        "quality_warnings": quality.get("warnings", []),
        "completeness_summary": completeness.get("summary", {}),
        "results": {
            "experiments": experiment_results,
            "scans": scan_results,
            "data_files": datafile_results,
        },
        "provenance": [
            ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="sqlite_catalog_keyword_ontology_report").model_dump(mode="json")
        ],
    }
    report["safe_next_actions"] = _safe_next_actions(report)
    output_dir = arguments.get("output_dir")
    if output_dir:
        report["artifacts"] = write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_markdown_report,
            json_filename="catalog_report.json",
            markdown_filename="catalog_report.md",
        )
    return report


def _ranked_records(record_type: str, records: list[Any], query_terms: set[str], query_concepts: list[Any], limit: int) -> list[dict[str, Any]]:
    ranked = [_record_result(record_type, record, query_terms, query_concepts) for record in records]
    ranked = [item for item in ranked if item["score"] > 0 or not query_terms]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def _record_result(record_type: str, record: Experiment | Scan | DataFile, query_terms: set[str], query_concepts: list[Any]) -> dict[str, Any]:
    text = _record_text(record)
    text_terms = set(tokenize(text))
    matched_terms = sorted(query_terms & text_terms)
    record_concepts = match_concepts(text, query_concepts)
    matched_concepts = sorted({concept.id for concept in record_concepts})
    score = (len(matched_terms) / max(len(query_terms), 1) if query_terms else 0.1) + 0.2 * len(matched_concepts)
    return {
        "source_type": record_type,
        "source_id": _record_id(record),
        "title": _record_title(record),
        "score": round(score, 3),
        "matched_terms": matched_terms,
        "matched_concepts": matched_concepts,
        "summary": _record_summary(record),
        "provenance": [
            ProvenanceRecord(source_type=record_type, source_id=_record_id(record), method="catalog_report").model_dump(mode="json")
        ],
    }


def _record_text(record: Experiment | Scan | DataFile) -> str:
    payload = record.model_dump(mode="json")
    return " ".join(str(value) for value in payload.values())


def _record_id(record: Experiment | Scan | DataFile) -> str:
    if isinstance(record, Experiment):
        return record.experiment_id
    if isinstance(record, Scan):
        return record.scan_id
    return record.datafile_id


def _record_title(record: Experiment | Scan | DataFile) -> str | None:
    if isinstance(record, Experiment):
        return record.title
    if isinstance(record, Scan):
        return record.scan_type or record.scan_id
    return Path(record.path).name


def _record_summary(record: Experiment | Scan | DataFile) -> dict[str, Any]:
    if isinstance(record, Experiment):
        return {
            "instrument": record.instrument,
            "facility": record.metadata.get("facility"),
            "native_experiment_id": record.metadata.get("native_experiment_id"),
        }
    if isinstance(record, Scan):
        return {
            "experiment_id": record.experiment_id,
            "instrument": record.instrument,
            "scan_type": record.scan_type,
            "q_center": record.q_center.model_dump(mode="json") if record.q_center else None,
            "metadata_mode": (record.metadata.get("raw_metadata") or {}).get("metadata_mode"),
        }
    return {
        "path": record.path,
        "format": record.format,
        "instrument": record.metadata.get("instrument"),
        "native_scan_id": record.metadata.get("native_scan_id"),
        "metadata_mode": (record.metadata.get("raw_metadata") or {}).get("metadata_mode"),
    }


def _concept_summary(concept: Any) -> dict[str, str]:
    return {"id": concept.id, "label": concept.label, "family": concept.family}


def _safe_next_actions(report: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if report["quality_warnings"]:
        actions.append("Review catalog quality warnings before using this catalog for scientific decisions.")
    if report["completeness_summary"].get("eligible_for_geometry_check", 0) == 0:
        actions.append("Add explicit Q/E/config/lattice/scattering-plane or compatible UB metadata before running geometry checks.")
    if report["completeness_summary"].get("eligible_for_resolution_check", 0) == 0:
        actions.append("Add Q/E/instrument configuration metadata before running resolution or spurion checks.")
    if not any(report["results"].values()):
        actions.append("Broaden the query or import more catalog records.")
    if not actions:
        actions.append("Inspect top-ranked scans and data files, then run registry workflows such as check-geometry only on records with complete metadata.")
    return actions


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TAS Catalog Report",
        "",
        f"- Status: {report['status']}",
        f"- Query: `{report['query']}`",
        f"- Catalog: `{report['catalog_path']}`",
        f"- Quality: `{report['quality_status']}`",
        f"- Catalog summary: `{report['catalog_summary']}`",
        f"- Completeness summary: `{report['completeness_summary']}`",
        "",
        "## Matched Concepts",
        "",
    ]
    if report["matched_concepts"]:
        lines.extend(f"- `{item['id']}`: {item['label']} ({item['family']})" for item in report["matched_concepts"])
    else:
        lines.append("- None")
    lines.extend(["", "## Top Scans", ""])
    lines.extend(_result_lines(report["results"].get("scans", [])))
    lines.extend(["", "## Top Data Files", ""])
    lines.extend(_result_lines(report["results"].get("data_files", [])))
    lines.extend(["", "## Top Experiments", ""])
    lines.extend(_result_lines(report["results"].get("experiments", [])))
    lines.extend(["", "## Quality Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["quality_warnings"]) if report["quality_warnings"] else lines.append("- None")
    lines.extend(["", "## Safe Next Actions", ""])
    lines.extend(f"- {action}" for action in report["safe_next_actions"])
    return "\n".join(lines) + "\n"


def _result_lines(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return ["- None"]
    return [
        f"- `{item['source_id']}` score={item['score']} title={item.get('title') or ''} concepts={','.join(item.get('matched_concepts', [])) or 'none'}"
        for item in results[:10]
    ]
