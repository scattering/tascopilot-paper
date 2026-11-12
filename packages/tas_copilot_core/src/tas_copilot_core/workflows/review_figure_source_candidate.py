from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tas_copilot_contracts import AuditRecord, DataFile, ProvenanceRecord
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.psd_calibration import accepted_psd_calibrations, match_psd_calibration
from tas_copilot_core.registry.tool_runner import stable_hash
from tas_copilot_core.source_triage import check_psd_channel_artifact, compare_psd_detector_cuts, hydrate_psd_records, plot_psd_detector_cuts, q_temperature_reconstruction_blocked
from tas_copilot_core.workflows.read_bt7_file import read_bt7_file
from tas_copilot_core.workflows.psd_mask_sensitivity import psd_mask_sensitivity


WORKFLOW_NAME = "review-figure-source-candidate"


def review_figure_source_candidate(arguments: dict[str, Any], reader: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or CoreConfig().root / "data" / "processed" / "figure_source_review").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.get("figure_report") or arguments.get("report")
    catalog_path = arguments.get("catalog")
    if not report_path or not catalog_path:
        return _missing(output_dir, ["figure_report", "catalog"], "A figure-source report and TAS data catalog are required.")
    report_file = Path(report_path).expanduser()
    if not report_file.exists():
        return _missing(output_dir, ["figure_report"], f"Figure-source report does not exist: {report_file}")
    catalog_file = Path(catalog_path).expanduser()
    if not catalog_file.exists():
        return _missing(output_dir, ["catalog"], f"TAS data catalog does not exist: {catalog_file}")

    report = json.loads(report_file.read_text(encoding="utf-8"))
    candidate = _select_candidate(report, arguments.get("candidate_id"))
    if candidate is None:
        return _missing(output_dir, ["candidate"], "No candidate was available in the figure-source report.")

    catalog = SQLiteDataCatalog(catalog_file)
    records = _candidate_records(catalog, candidate)
    if not records:
        return _missing(output_dir, ["candidate_records"], f"No catalog records matched candidate {candidate.get('candidate_id')}.")
    hydration_results: list[dict[str, Any]] = []
    if arguments.get("hydrate", False):
        records, hydration_results = hydrate_psd_records(records, reader or read_bt7_file)

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    detector_cut_results = []
    for record in records:
        record_id = str(record.get("datafile_id") or record.get("scan_id") or record.get("path"))
        result = plot_psd_detector_cuts(
            records,
            diagnostics_dir / _safe(record_id),
            scan_id=record_id,
            temperature_min=_optional_float(arguments.get("temperature_min")),
            temperature_max=_optional_float(arguments.get("temperature_max")),
        )
        detector_cut_results.append({"source_id": record_id, "result": result})

    successful_cut_records = [item["source_id"] for item in detector_cut_results if item["result"].get("status") == "success"]
    comparison = None
    if len(successful_cut_records) >= 2:
        comparison = compare_psd_detector_cuts(
            records,
            diagnostics_dir / "compare_first_two",
            left_scan_id=successful_cut_records[0],
            right_scan_id=successful_cut_records[1],
            temperature_min=_optional_float(arguments.get("temperature_min")),
            temperature_max=_optional_float(arguments.get("temperature_max")),
            temperature_tolerance=float(arguments.get("temperature_tolerance", 2.0)),
        )
    channels = [float(item) for item in arguments.get("channels", []) or []]
    channel_artifact = None
    if channels and successful_cut_records:
        channel_artifact = check_psd_channel_artifact(
            records,
            diagnostics_dir / "channel_artifact",
            scan_id=successful_cut_records[0],
            channels_of_interest=channels,
            temperature_min=_optional_float(arguments.get("temperature_min")),
            temperature_max=_optional_float(arguments.get("temperature_max")),
        )

    q_gate = q_temperature_reconstruction_blocked(str(candidate.get("candidate_id")), arguments.get("calibration_registry")).model_dump(mode="json")
    accepted_calibrations = _accepted_calibrations(arguments.get("calibration_registry"))
    calibration_matches = _calibration_matches(
        records,
        arguments.get("psd_calibration_dir"),
        path_only=bool(arguments.get("psd_calibration_path_only", False)),
    )
    mask_sensitivity = _mask_sensitivity(
        records,
        calibration_matches,
        arguments,
        output_dir / "psd_mask_sensitivity",
    )
    status = "success" if successful_cut_records else "missing_input"
    payload: dict[str, Any] = {
        "workflow_name": WORKFLOW_NAME,
        "status": status,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_type": candidate.get("candidate_type"),
        "candidate_score": candidate.get("total_score"),
        "candidate_scan_ids": candidate.get("scan_ids", []),
        "candidate_datafile_ids": candidate.get("datafile_ids", []),
        "records_selected": len(records),
        "selected_record_ids": [str(record.get("datafile_id") or record.get("scan_id") or record.get("path")) for record in records],
        "hydration_results": hydration_results,
        "detector_cut_results": detector_cut_results,
        "comparison_result": comparison,
        "channel_artifact_result": channel_artifact,
        "mask_sensitivity_result": mask_sensitivity,
        "q_temperature_gate": q_gate,
        "accepted_psd_calibrations": accepted_calibrations,
        "psd_calibration_matches": calibration_matches,
        "safe_conclusion": _safe_conclusion(status, candidate, successful_cut_records, q_gate, mask_sensitivity),
        "next_checks": _next_checks(successful_cut_records, q_gate),
        "provenance": [
            ProvenanceRecord(source_type="figure_source_report", source_id=str(report_file), method="candidate_review").model_dump(mode="json"),
            ProvenanceRecord(source_type="data_catalog", source_id=str(catalog_file), method="candidate_review").model_dump(mode="json"),
        ],
    }
    payload["audit_ids"] = _write_audit(output_dir, payload, arguments)
    artifacts = _write_review(output_dir, payload)
    payload["artifacts"] = artifacts
    (output_dir / "figure_source_candidate_review.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _calibration_matches(records: list[dict[str, Any]], calibration_dir: str | None, *, path_only: bool) -> list[dict[str, Any]]:
    if not calibration_dir:
        return []
    results: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("datafile_id") or record.get("scan_id") or record.get("path"))
        path = record.get("path")
        if not path:
            results.append(
                {
                    "source_id": source_id,
                    "status": "missing_input",
                    "missing_fields": ["path"],
                    "warnings": ["No local path is available for PSD calibration matching."],
                }
            )
            continue
        try:
            match = match_psd_calibration(calibration_dir, path, path_only=path_only)
        except OSError as exc:
            match = {
                "status": "error",
                "errors": [{"code": exc.__class__.__name__, "message": str(exc)}],
                "safe_conclusion": "PSD calibration matching failed while reading local files.",
            }
        results.append(
            {
                "source_id": source_id,
                "path": str(path),
                "status": match.get("status"),
                "path_only": path_only,
                "best_match": match.get("best_match"),
                "top_matches": (match.get("matches") or [])[:5],
                "missing_fields": match.get("missing_fields", []),
                "safe_conclusion": match.get("safe_conclusion"),
                "matching_policy": match.get("matching_policy", {}),
            }
        )
    return results


def _mask_sensitivity(
    records: list[dict[str, Any]],
    calibration_matches: list[dict[str, Any]],
    arguments: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    calibration_dir = arguments.get("psd_calibration_dir")
    annotated_q = arguments.get("annotated_q")
    if not calibration_dir or annotated_q is None:
        return None
    data_files = [str(record["path"]) for record in records if record.get("path")]
    if not data_files:
        return {
            "workflow_name": "psd-mask-sensitivity",
            "status": "missing_input",
            "missing_fields": ["data_files"],
            "warnings": ["No local data-file paths were available for PSD mask sensitivity."],
        }
    calibration_files = list(arguments.get("psd_calibration_files") or arguments.get("calibration_files") or [])
    if not calibration_files:
        calibration_files = [
            str(match["best_match"]["filename"])
            for match in calibration_matches
            if isinstance(match.get("best_match"), dict) and match["best_match"].get("filename")
        ]
    path_only = bool(arguments.get("psd_calibration_path_only", False))
    if path_only and not calibration_files:
        return {
            "workflow_name": "psd-mask-sensitivity",
            "status": "missing_input",
            "missing_fields": ["calibration_files"],
            "warnings": ["Mask sensitivity needs hydrated calibration rows or explicit calibration files; path-only calibration discovery is not enough."],
        }
    return psd_mask_sensitivity(
        {
            "data_files": data_files,
            "psd_calibration_dir": calibration_dir,
            "calibration_files": sorted(set(calibration_files)),
            "annotated_q": annotated_q,
            "output_dir": str(output_dir),
            "path_only": path_only,
        }
    )


def _accepted_calibrations(registry: str | None) -> list[dict[str, Any]]:
    return [
        {
            "calibration_id": record.calibration_id,
            "instrument": record.instrument,
            "detector_mode": record.detector_mode,
            "date_range": record.date_range,
            "status": record.status,
            "reviewed_by": record.reviewed_by,
            "reviewed_at": record.reviewed_at,
            "channel_mapping": record.channel_mapping,
            "detector_geometry": record.detector_geometry,
            "notes": record.notes,
        }
        for record in accepted_psd_calibrations(registry, instrument="BT-7")
    ]


def _select_candidate(report: dict[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
    candidates = list(report.get("candidates") or [])
    candidates.extend(report.get("candidate_links") or [])
    candidates.extend(report.get("panel_source_candidates") or [])
    state = report.get("source_triage_state") or {}
    candidates.extend(state.get("candidates") or [])
    candidates.extend(_group_candidates(state.get("candidate_groups") or []))
    if not candidates:
        return None
    if candidate_id:
        return next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
    return candidates[0]


def _group_candidates(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in groups:
        candidates.append(
            {
                "candidate_id": group.get("group_id"),
                "candidate_type": "scan_group",
                "scan_ids": group.get("member_scan_ids") or [],
                "datafile_ids": group.get("member_datafile_ids") or [],
                "total_score": group.get("grouping_confidence"),
                "source_family": group.get("group_type"),
            }
        )
    return candidates


def _candidate_records(catalog: SQLiteDataCatalog, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    datafiles_by_id = {item.datafile_id: item for item in catalog.list_datafiles()}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    if candidate.get("candidate_type") == "datafile" and candidate.get("candidate_id"):
        datafile = datafiles_by_id.get(str(candidate["candidate_id"]))
        if datafile:
            records.append(_record(catalog, datafile))
            seen.add(datafile.datafile_id)
    for datafile_id in candidate.get("datafile_ids", []) or []:
        datafile = datafiles_by_id.get(datafile_id)
        if datafile and datafile.datafile_id not in seen:
            records.append(_record(catalog, datafile))
            seen.add(datafile.datafile_id)
    for scan_id in candidate.get("scan_ids", []) or []:
        for datafile in catalog.datafiles_for_scan(scan_id):
            if datafile.datafile_id not in seen:
                records.append(_record(catalog, datafile))
                seen.add(datafile.datafile_id)
    return records


def _record(catalog: SQLiteDataCatalog, datafile: DataFile) -> dict[str, Any]:
    scan = catalog.get_scan(datafile.scan_id) if datafile.scan_id else None
    payload = datafile.model_dump(mode="json")
    if scan:
        payload["scan"] = scan.model_dump(mode="json")
        payload["scan_id"] = scan.scan_id
        payload["metadata"] = {**(payload.get("metadata") or {}), **scan.metadata}
    psd_map = (payload.get("metadata") or {}).get("psd_map")
    if isinstance(psd_map, dict):
        payload.setdefault("psd_channel_axis", psd_map.get("psd_channel_axis") or psd_map.get("q_axis"))
        payload.setdefault("temperature_axis", psd_map.get("temperature_axis"))
        payload.setdefault("intensity", psd_map.get("intensity"))
    return payload


def _missing(output_dir: Path, missing_fields: list[str], warning: str) -> dict[str, Any]:
    payload = {
        "workflow_name": WORKFLOW_NAME,
        "status": "missing_input",
        "missing_fields": missing_fields,
        "warnings": [warning],
        "safe_conclusion": "Candidate review could not run because required inputs are missing.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_source_candidate_review.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "figure_source_candidate_review.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _write_review(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    json_path = output_dir / "figure_source_candidate_review.json"
    md_path = output_dir / "figure_source_candidate_review.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "diagnostics_dir": str(output_dir / "diagnostics")}


def _write_audit(output_dir: Path, payload: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    logger = JsonlAuditLogger(output_dir / "audit")
    audit = AuditRecord(
        event_type="workflow",
        status=payload["status"],
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": arguments}),
        output_hash=stable_hash(payload),
        details={
            "workflow_name": WORKFLOW_NAME,
            "candidate_id": payload.get("candidate_id"),
            "records_selected": payload.get("records_selected"),
            "detector_cut_statuses": [item.get("result", {}).get("status") for item in payload.get("detector_cut_results", [])],
            "mask_sensitivity_status": (payload.get("mask_sensitivity_result") or {}).get("status"),
            "accepted_psd_calibration_ids": [item.get("calibration_id") for item in payload.get("accepted_psd_calibrations", [])],
        },
    )
    logger.write(audit)
    return [audit.audit_id]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Figure Source Candidate Review",
        "",
        f"Status: `{payload.get('status')}`",
        f"Candidate: `{payload.get('candidate_id')}`",
        f"Candidate type: `{payload.get('candidate_type')}`",
        f"Selected records: `{payload.get('records_selected', 0)}`",
        "",
        "## Detector-Channel Diagnostics",
    ]
    for item in payload.get("detector_cut_results", []):
        result = item.get("result", {})
        lines.append(f"- `{item.get('source_id')}`: `{result.get('status')}`")
        if result.get("missing_fields"):
            lines.append(f"  - missing: `{', '.join(result['missing_fields'])}`")
    comparison = payload.get("comparison_result")
    lines.extend(["", "## Warm/Cool Or First-Pair Comparison", f"- Status: `{comparison.get('status') if comparison else 'not_run'}`"])
    artifact = payload.get("channel_artifact_result")
    lines.extend(["", "## Channel Artifact Diagnostic", f"- Status: `{artifact.get('status') if artifact else 'not_run'}`"])
    mask_sensitivity = payload.get("mask_sensitivity_result")
    lines.extend(["", "## PSD Mask Sensitivity"])
    if mask_sensitivity:
        assessment = mask_sensitivity.get("artifact_assessment") or {}
        lines.append(f"- Status: `{mask_sensitivity.get('status')}`")
        lines.append(f"- Conclusion: `{assessment.get('conclusion')}`")
        lines.append(f"- Heuristic probability: `{assessment.get('artifact_probability')}`")
        if mask_sensitivity.get("safe_conclusion"):
            lines.append(f"- {mask_sensitivity['safe_conclusion']}")
        markdown = (mask_sensitivity.get("artifacts") or {}).get("markdown")
        if markdown:
            lines.append(f"- Detail report: `{markdown}`")
        if mask_sensitivity.get("missing_fields"):
            lines.append(f"- Missing: `{', '.join(mask_sensitivity['missing_fields'])}`")
    else:
        lines.append("- Not run.")
    lines.extend(["", "## Accepted PSD Calibration Policy"])
    accepted = payload.get("accepted_psd_calibrations") or []
    if accepted:
        for item in accepted:
            mapping = item.get("channel_mapping") or {}
            geometry = item.get("detector_geometry") or {}
            mask = (mapping.get("mask_policy") or {}).get("masked_one_based_channels")
            lines.append(f"- `{item.get('calibration_id')}` status `{item.get('status')}` reviewed `{item.get('reviewed_at')}`")
            lines.append(f"  - source: `{mapping.get('source_filename')}` AnaTrans `{geometry.get('analyzer_detector_translation')}`")
            lines.append(f"  - mask policy one-based channels: `{mask}`")
        lines.append("- Accepted calibration records are distinct from raw DAT candidate matching; Q-space reconstruction still requires the reviewed conversion adapter gate below.")
    else:
        lines.append("- None registered for this review.")
    lines.extend(["", "## PSD Calibration Candidates"])
    calibration_matches = payload.get("psd_calibration_matches") or []
    if calibration_matches:
        for item in calibration_matches:
            best = item.get("best_match") or {}
            lines.append(f"- `{item.get('source_id')}`: `{item.get('status')}`")
            if best:
                lines.append(
                    f"  - best: `{best.get('filename')}` score `{best.get('score')}` AnaTrans `{best.get('analyzer_detector_translation')}`"
                )
            if item.get("missing_fields"):
                lines.append(f"  - missing: `{', '.join(item['missing_fields'])}`")
            if item.get("safe_conclusion"):
                lines.append(f"  - {item['safe_conclusion']}")
    else:
        lines.append("- Not run.")
    gate = payload.get("q_temperature_gate", {})
    lines.extend(["", "## Q-Space Gate", f"- Status: `{gate.get('status')}`", f"- Missing: `{', '.join(gate.get('missing_fields', []))}`"])
    for warning in gate.get("warnings", []):
        lines.append(f"- {warning}")
    lines.extend(["", "## Safe Conclusion", "", payload.get("safe_conclusion", "")])
    lines.extend(["", "## Next Checks"])
    lines.extend(f"- {item}" for item in payload.get("next_checks", []))
    return "\n".join(lines) + "\n"


def _safe_conclusion(
    status: str,
    candidate: dict[str, Any],
    successful_cut_records: list[str],
    q_gate: dict[str, Any],
    mask_sensitivity: dict[str, Any] | None,
) -> str:
    if status != "success":
        return "No detector-channel review could be completed. Hydrate/read the candidate files and rerun before judging feature reality."
    conclusion = (
        f"Detector-channel diagnostics were produced for {len(successful_cut_records)} record(s) from candidate {candidate.get('candidate_id')}. "
        "These diagnostics can identify raw detector-channel behavior, but they do not prove a Q-space feature or artifact. "
        f"Q-space reconstruction remains {q_gate.get('status')} until {', '.join(q_gate.get('missing_fields', []))} is resolved."
    )
    assessment = (mask_sensitivity or {}).get("artifact_assessment") or {}
    if assessment.get("conclusion") == "probable_artifact":
        conclusion += (
            " PSD mask sensitivity assigns a probability-level artifact assessment for the annotated-Q feature, "
            f"with heuristic probability {assessment.get('artifact_probability')}; this is evidence, not certainty."
        )
    return conclusion


def _next_checks(successful_cut_records: list[str], q_gate: dict[str, Any]) -> list[str]:
    checks = ["Review raw detector-channel cuts before interpolation, smoothing, or background subtraction."]
    if len(successful_cut_records) >= 2:
        checks.append("Compare warm/cool or repeated scans for consistency of the detector-channel feature.")
    else:
        checks.append("Hydrate more repeated/warm/cool records before making a repeatability assessment.")
    missing = set(q_gate.get("missing_fields") or [])
    if "reviewed_q_conversion_adapter" in missing and "reviewed_psd_channel_to_a4_calibration" not in missing:
        checks.append("Complete reviewed conversion-adapter and lineage checks before any Q-space claim.")
    elif q_gate.get("missing_fields"):
        checks.append("Complete PSD calibration review and conversion-adapter review before any Q-space claim.")
    checks.append("Do not claim feature reality or artifact status without raw-cut, repeatability, background, and calibration evidence.")
    return checks


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "record"
