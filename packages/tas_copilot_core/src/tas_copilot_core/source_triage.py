from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tas_copilot_contracts import (
    CandidateEvidence,
    CandidateMismatch,
    CandidateSourceData,
    FigureSourceQuery,
    HypothesisState,
    MaterialIdentityRecord,
    PSDMapReconstructionResult,
    PSDSourceGroup,
    ProvenanceRecord,
    RejectedCandidate,
    ScanFamilyClassification,
    SourceTriageState,
)

from tas_copilot_core.material_identity_registry import material_identity_record_via_registry
from tas_copilot_core.psd_calibration import accepted_psd_calibrations
from tas_copilot_core.reporting import write_report_bundle


PSD_CALIBRATION_NAME_PATTERNS = ("psdcalib", "psd_calib", "psd-calib", "psd_a3", "focusana_psd", "flatana_psd")
PSD_CALIBRATION_HEADER_TERMS = ("#AnalyzerPSDGroup", "#DetectorEfficiencies", "#AnalyzerDetectorMode")


def classify_source_family(record: dict[str, Any]) -> ScanFamilyClassification:
    source_id = str(record.get("scan_id") or record.get("datafile_id") or record.get("path") or record.get("file") or "unknown")
    metadata = _metadata(record)
    text = _blob(record).lower()
    detector_mode = str(metadata.get("analyzer_detector_mode") or metadata.get("detector_mode") or record.get("detector_mode") or "")
    scan_type = str(metadata.get("scan_type") or metadata.get("ScanType") or record.get("scan_type") or "").upper()
    energy = _energy(record)
    has_psd = "psd" in detector_mode.lower() or "psdc" in text or "psd" in text
    has_temp_axis = bool(_temperature_values(record))
    has_psd_channels = bool(_psd_channels(record)) or "psdc00" in text
    evidence: list[CandidateEvidence] = []
    missing: list[str] = []
    mismatches: list[CandidateMismatch] = []

    if has_psd:
        evidence.append(CandidateEvidence(evidence_type="detector_mode", summary=f"PSD evidence from detector mode/text: {detector_mode}", score=1.0))
    if scan_type:
        evidence.append(CandidateEvidence(evidence_type="scan_type", summary=f"Scan type {scan_type}", score=0.5))
    if energy is not None:
        evidence.append(CandidateEvidence(evidence_type="energy_transfer", summary=f"Energy transfer {energy:g} meV", score=1.0 if abs(energy) < 1e-6 else 0.2))

    if has_psd and scan_type == "ENV" and has_temp_axis and has_psd_channels:
        family = "single_file_psd_temperature_map"
        confidence = 0.9
    elif has_psd and scan_type == "ANGLE":
        family = "multi_file_psd_angle_temperature_series"
        confidence = 0.75 if has_temp_axis else 0.65
    elif not has_psd and energy is not None and abs(float(energy)) > 1e-6:
        family = "single_detector_inelastic_line_scan"
        confidence = 0.85
    elif not has_psd and energy is not None and abs(float(energy)) <= 1e-6:
        family = "single_detector_elastic_line_scan"
        confidence = 0.8
    elif scan_type == "ANGLE":
        family = "angle_scan_temperature_series"
        confidence = 0.45
    elif scan_type == "VECTOR":
        family = "vector_scan_series"
        confidence = 0.45
    else:
        family = "unknown_or_sparse"
        confidence = 0.0
        missing.extend(["detector_mode", "scan_type"])

    return ScanFamilyClassification(
        source_id=source_id,
        source_family=family,
        confidence=confidence,
        evidence=evidence,
        missing_fields=sorted(set(missing)),
        mismatches=mismatches,
        provenance=[ProvenanceRecord(source_type="catalog_record", source_id=source_id, method="deterministic_source_family_classifier")],
    )


def group_psd_sources(records: list[dict[str, Any]], family: str = "lu_femn_o3", runner=None) -> list[PSDSourceGroup]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        classification = classify_source_family(record)
        if not classification.source_family.startswith(("single_file_psd", "multi_file_psd")):
            continue
        material = material_identity_record_via_registry(_filename(record), family=family, source_field="filename", runner=runner, caller_workflow="psd-source-grouping")
        x_mn = material.x_variables.get("x_Mn")
        material_key = f"xmn-{x_mn:g}" if x_mn is not None else f"ambiguous-{_family_stem(_filename(record))}"
        if classification.source_family == "single_file_psd_temperature_map":
            key = f"{classification.source_family}:{_source_id(record)}"
        else:
            key = f"{classification.source_family}:{material_key}:{_experiment(record)}:{_family_stem(_filename(record))}"
        buckets.setdefault(key, []).append(record)

    groups: list[PSDSourceGroup] = []
    for index, (key, members) in enumerate(sorted(buckets.items()), start=1):
        classifications = [classify_source_family(member) for member in members]
        material_records = [
            material_identity_record_via_registry(_filename(member), family=family, source_field="filename", runner=runner, caller_workflow="psd-source-grouping")
            for member in members
        ]
        temperatures = sorted({value for member in members for value in _temperature_values(member)})
        detector_modes = sorted({_metadata(member).get("analyzer_detector_mode") or member.get("detector_mode") for member in members if _metadata(member).get("analyzer_detector_mode") or member.get("detector_mode")})
        group_type = classifications[0].source_family
        groups.append(
            PSDSourceGroup(
                group_id=f"psd-group-{index}",
                group_type=group_type,
                member_scan_ids=[str(member.get("scan_id")) for member in members if member.get("scan_id")],
                member_datafile_ids=[str(member.get("datafile_id") or _source_id(member)) for member in members],
                instrument=_metadata(members[0]).get("instrument") or members[0].get("instrument"),
                material_identity_summary={
                    "records": [item.model_dump(mode="json") for item in material_records],
                    "resolved_x_mn": sorted({item.x_variables.get("x_Mn") for item in material_records if item.x_variables.get("x_Mn") is not None}),
                    "ambiguous_count": sum(1 for item in material_records if item.ambiguity_reason),
                },
                detector_mode_summary={"modes": detector_modes},
                scan_type_summary={"families": sorted({item.source_family for item in classifications})},
                energy_transfer_summary=_energy_summary(members),
                temperature_values=temperatures,
                temperature_range={"min": min(temperatures), "max": max(temperatures)} if temperatures else {},
                q_or_angle_coverage=_angle_summary(members),
                psd_channel_count=max((_psd_channel_count(member) for member in members), default=None),
                fixed_angle_or_scanned_angle="fixed_or_scanned_angle_from_metadata",
                missing_metadata=_group_missing_metadata(members),
                grouping_confidence=round(sum(item.confidence for item in classifications) / max(len(classifications), 1), 4),
                evidence=[CandidateEvidence(evidence_type="grouping", summary=f"Grouped {len(members)} records by {key}", score=1.0)],
                provenance=[ProvenanceRecord(source_type="source_records", source_id=key, method="deterministic_psd_grouping")],
            )
        )
    return groups


def reconstruct_channel_temperature(group: PSDSourceGroup, records: list[dict[str, Any]], output_dir: str | Path) -> PSDMapReconstructionResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    members = [record for record in records if str(record.get("datafile_id") or _source_id(record)) in set(group.member_datafile_ids)]
    if group.group_type != "single_file_psd_temperature_map":
        return PSDMapReconstructionResult(
            status="unsupported",
            candidate_id=group.group_id,
            reconstruction_level="channel_temperature",
            missing_fields=["single_file_psd_temperature_map"],
            warnings=["This minimal reconstruction slice supports single-file PSD temperature maps only."],
        )
    if not members:
        return PSDMapReconstructionResult(status="missing_input", candidate_id=group.group_id, missing_fields=["member_records"])
    record = members[0]
    intensity = _intensity(record)
    temperatures = _temperature_values(record)
    channels = _psd_channels(record) or list(range(len(intensity[0]) if intensity else 0))
    missing = []
    if not intensity:
        missing.append("psd_intensity")
    if not temperatures:
        missing.append("temperature_axis")
    if not channels:
        missing.append("psd_channel_axis")
    if missing:
        return PSDMapReconstructionResult(status="missing_input", candidate_id=group.group_id, missing_fields=missing)
    json_path = output / f"{group.group_id}_channel_temperature.json"
    csv_path = output / f"{group.group_id}_channel_temperature.csv"
    payload = {"group_id": group.group_id, "channels": channels, "temperatures": temperatures, "intensity": intensity}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_K", *[f"channel_{channel}" for channel in channels]])
        for temp, row in zip(temperatures, intensity):
            writer.writerow([temp, *row])
    return PSDMapReconstructionResult(
        status="success",
        candidate_id=group.group_id,
        reconstruction_level="channel_temperature",
        input_files=[_filename(record)],
        detector_axis_source="PSD channel axis",
        temperature_source="metadata.temperature_axis",
        intensity_source="PSD channel intensity",
        detector_axis_range={"min": min(channels), "max": max(channels)},
        temperature_range={"min": min(temperatures), "max": max(temperatures)},
        shape=[len(temperatures), len(channels)],
        output_artifact_paths=[str(json_path), str(csv_path)],
        warnings=["Q x temperature reconstruction is not attempted without reviewed PSD channel-to-A4/sample two-theta calibration."],
        provenance=[ProvenanceRecord(source_type="psd_group", source_id=group.group_id, method="channel_temperature_reconstruction")],
    )


def q_temperature_reconstruction_blocked(group_id: str, calibration_registry: str | Path | None = None) -> PSDMapReconstructionResult:
    accepted = accepted_psd_calibrations(calibration_registry, instrument="BT-7")
    if accepted:
        return PSDMapReconstructionResult(
            status="unsupported",
            candidate_id=group_id,
            reconstruction_level="q_temperature",
            q_axis_source=None,
            missing_fields=["reviewed_q_conversion_adapter"],
            warnings=[
                "Accepted PSD calibration registry entries exist, but Q x temperature conversion is still unsupported until a reviewed adapter applies detector mode, date range, PSD channel-to-A4/sample two-theta calibration, sign convention, elastic Q formula, and lineage checks.",
                f"Accepted calibration ids: {', '.join(record.calibration_id for record in accepted)}",
            ],
        )
    return PSDMapReconstructionResult(
        status="unsupported",
        candidate_id=group_id,
        reconstruction_level="q_temperature",
        q_axis_source=None,
            missing_fields=["reviewed_psd_channel_to_a4_calibration"],
        warnings=["Q x temperature reconstruction is blocked until a reviewed PSD channel-to-A4/sample two-theta calibration is available. For elastic diffraction, Q from A4/sample two-theta is then formulaic."],
    )


def discover_psd_calibration_candidates(data_root: str | Path, output_dir: str | Path | None = None, max_files: int = 100000) -> dict[str, Any]:
    root = Path(data_root).expanduser()
    if not root.exists():
        return {
            "workflow_name": "discover-psd-calibrations",
            "status": "missing_input",
            "missing_fields": ["data_root"],
            "data_root": str(root),
            "warnings": ["The requested data root does not exist."],
        }
    candidates: list[dict[str, Any]] = []
    inspected = 0
    considered = 0
    for path in sorted(root.rglob("*")):
        if max_files > 0 and inspected >= max_files:
            break
        if not path.is_file():
            continue
        inspected += 1
        if path.suffix.lower() not in {"", ".bt7", ".txt", ".dat", ".seq"}:
            continue
        if not _psd_calibration_name_hint(path.name):
            continue
        considered += 1
        candidate = _psd_calibration_candidate(path, root)
        if candidate["score"] > 0:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-float(item["score"]), item["relative_path"]))
    payload = {
        "workflow_name": "discover-psd-calibrations",
        "status": "success",
        "data_root": str(root),
        "files_inspected": inspected,
        "files_considered": considered,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "safe_conclusion": (
            "Candidate PSD calibration files were discovered, but none are accepted calibration records. "
            "Q x temperature reconstruction remains blocked until a reviewed calibration is registered."
        ),
        "warnings": [
            "This is discovery only; it does not validate channel efficiencies or channel-to-angle mapping.",
            "Filename prefiltering is used for speed; calibration files with no PSD/calibration/focus hint in the name may require manual review.",
            "Do not use discovered files to unlock Q conversion without expert review and provenance.",
        ],
        "provenance": [ProvenanceRecord(source_type="filesystem", source_id=str(root), method="local_psd_calibration_discovery").model_dump(mode="json")],
    }
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "psd_calibration_candidates.json"
        md_path = output / "psd_calibration_candidates.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(_psd_calibration_markdown(payload), encoding="utf-8")
        payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _psd_calibration_candidate(path: Path, root: Path) -> dict[str, Any]:
    text = _read_text_prefix(path)
    lower_name = path.name.lower()
    lower_text = text.lower()
    evidence: list[str] = []
    score = 0.0
    if any(pattern in lower_name for pattern in PSD_CALIBRATION_NAME_PATTERNS):
        score += 0.35
        evidence.append("filename suggests PSD calibration or PSD angle/focus scan")
    if "psd" in lower_name and ("cal" in lower_name or "a3" in lower_name or "focus" in lower_name or "flat" in lower_name):
        score += 0.2
        evidence.append("filename combines PSD with calibration/angle/focus terms")
    if "#analyzerpsdgroup" in lower_text:
        score += 0.15
        evidence.append("header defines AnalyzerPSDGroup")
    if "psdc00" in lower_text and "psdc47" in lower_text:
        score += 0.1
        evidence.append("header or columns include PSDC00..PSDC47")
    if "#detectorefficiencies" in lower_text and "psdc" in lower_text:
        score += 0.1
        evidence.append("header includes PSD detector-efficiency fields")
    if "psddiff" in lower_text or "psdflat" in lower_text:
        score += 0.1
        evidence.append("header names PSDDiff or PSDFlat detector mode")
    headers = _header_summary(text)
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "filename": path.name,
        "score": round(min(score, 1.0), 4),
        "status": "candidate_unreviewed",
        "q_temperature_unlock": False,
        "evidence": evidence,
        "header_summary": headers,
        "sha256_prefix": _sha256_prefix(path),
        "warnings": ["Candidate is not a reviewed calibration record; Q conversion remains blocked."],
        "provenance": [ProvenanceRecord(source_type="file", source_id=str(path), method="psd_calibration_candidate_discovery").model_dump(mode="json")],
    }


def _psd_calibration_name_hint(name: str) -> bool:
    lower = name.lower()
    if any(pattern in lower for pattern in PSD_CALIBRATION_NAME_PATTERNS):
        return True
    return "psd" in lower and any(term in lower for term in ("cal", "a3", "focus", "flat", "diff", "mode"))


def _read_text_prefix(path: Path, max_bytes: int = 32768) -> str:
    try:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    except OSError:
        return ""


def _header_summary(text: str) -> dict[str, Any]:
    keys = {
        "filename": "#Filename",
        "date": "#Date",
        "instrument": "#InstrName",
        "experiment_id": "#ExptID",
        "scan_type": "#ScanType",
        "scan": "#Scan",
        "scan_description": "#ScanDescr",
        "analyzer_detector_mode": "#AnalyzerDetectorMode",
    }
    summary: dict[str, Any] = {}
    for line in text.splitlines():
        for key, prefix in keys.items():
            if line.startswith(prefix) and key not in summary:
                summary[key] = line[len(prefix) :].strip()
    summary["has_detector_efficiencies"] = "#DetectorEfficiencies" in text
    summary["has_analyzer_psd_group"] = "#AnalyzerPSDGroup" in text
    summary["has_psd_columns"] = "PSDC00" in text and "PSDC47" in text
    return summary


def _sha256_prefix(path: Path, max_bytes: int = 1024 * 1024) -> str | None:
    try:
        digest = hashlib.sha256()
        remaining = max_bytes
        with path.open("rb") as handle:
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _psd_calibration_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Calibration Candidate Discovery",
        "",
        f"Status: `{payload['status']}`",
        f"Data root: `{payload['data_root']}`",
        f"Files inspected: {payload['files_inspected']}",
        f"Files considered after PSD/calibration filename prefilter: {payload.get('files_considered', 0)}",
        f"Candidate count: {payload['candidate_count']}",
        "",
        payload["safe_conclusion"],
        "",
        "## Top Candidates",
        "",
    ]
    candidates = payload.get("candidates", [])
    if candidates:
        lines.extend(["| Rank | Score | File | Mode | Scan type | Evidence |", "| ---: | ---: | --- | --- | --- | --- |"])
        for index, candidate in enumerate(candidates[:25], start=1):
            header = candidate.get("header_summary", {})
            lines.append(
                f"| {index} | {candidate['score']} | `{candidate['relative_path']}` | `{header.get('analyzer_detector_mode', '')}` | `{header.get('scan_type', '')}` | {', '.join(candidate.get('evidence', []))} |"
            )
    else:
        lines.append("- No PSD calibration candidates found.")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines) + "\n"


def plot_psd_detector_cuts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    scan_id: str | None = None,
    group_id: str | None = None,
    temperature_min: float | None = None,
    temperature_max: float | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_detector_cut_records(records, scan_id=scan_id, group_id=group_id)
    if not selected:
        return {
            "workflow_name": "plot-psd-detector-cuts",
            "status": "missing_input",
            "missing_fields": ["scan_id_or_group_id"],
            "warnings": ["No catalog record matched the requested scan/group."],
        }

    rows: list[dict[str, Any]] = []
    missing: set[str] = set()
    provenance = []
    for record in selected:
        channels = _psd_channels(record)
        temperatures = _temperature_values(record)
        intensity = _intensity(record)
        if not channels:
            missing.add("psd_channel_axis")
        if not temperatures:
            missing.add("temperature_axis")
        if not intensity:
            missing.add("psd_intensity")
        for index, temp in enumerate(temperatures):
            if temperature_min is not None and temp < temperature_min:
                continue
            if temperature_max is not None and temp > temperature_max:
                continue
            if index >= len(intensity):
                missing.add("psd_intensity_row_for_temperature")
                continue
            rows.append({"source_id": _source_id(record), "temperature_K": temp, "channels": channels, "intensity": intensity[index]})
        provenance.append(ProvenanceRecord(source_type="catalog_record", source_id=_source_id(record), method="psd_detector_cut_diagnostic").model_dump(mode="json"))

    if missing or not rows:
        if not rows:
            missing.add("temperature_window_with_data")
        return {
            "workflow_name": "plot-psd-detector-cuts",
            "status": "missing_input",
            "missing_fields": sorted(missing),
            "selected_records": [_source_id(record) for record in selected],
            "warnings": ["Detector-cut diagnostics require PSD channel axes, intensity rows, and temperatures in the requested window."],
            "provenance": provenance,
        }

    csv_path = output / "psd_detector_cuts.csv"
    json_path = output / "psd_detector_cuts.json"
    md_path = output / "psd_detector_cuts.md"
    png_path = output / "psd_detector_cuts.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        max_channels = max(len(row["channels"]) for row in rows)
        writer.writerow(["source_id", "temperature_K", *[f"channel_{idx}" for idx in range(max_channels)]])
        for row in rows:
            writer.writerow([row["source_id"], row["temperature_K"], *row["intensity"]])
    payload = {
        "workflow_name": "plot-psd-detector-cuts",
        "status": "success",
        "scan_id": scan_id,
        "group_id": group_id,
        "temperature_window": {"min": temperature_min, "max": temperature_max},
        "cut_count": len(rows),
        "channel_count": max(len(row["channels"]) for row in rows),
        "selected_records": [_source_id(record) for record in selected],
        "artifacts": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "warnings": ["Detector-channel cuts are not Q cuts; Q conversion requires reviewed PSD channel-to-A4/sample two-theta calibration."],
        "provenance": provenance,
    }
    plot_warning = _write_detector_cut_plot(rows, png_path)
    if plot_warning:
        payload["warnings"].append(plot_warning)
    else:
        payload["artifacts"]["png"] = str(png_path)
    json_path.write_text(json.dumps({**payload, "cuts": rows}, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_detector_cuts_markdown(payload), encoding="utf-8")
    return payload


def compare_psd_detector_cuts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    left_scan_id: str,
    right_scan_id: str,
    temperature_min: float | None = None,
    temperature_max: float | None = None,
    temperature_tolerance: float = 2.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    left_records = select_detector_cut_records(records, scan_id=left_scan_id, group_id=None)
    right_records = select_detector_cut_records(records, scan_id=right_scan_id, group_id=None)
    if not left_records or not right_records:
        return {
            "workflow_name": "compare-psd-detector-cuts",
            "status": "missing_input",
            "missing_fields": [field for field, found in [("left_scan_id", left_records), ("right_scan_id", right_records)] if not found],
            "warnings": ["Both left and right PSD records are required."],
        }
    left = left_records[0]
    right = right_records[0]
    left_rows, left_missing = _detector_cut_rows(left, temperature_min, temperature_max)
    right_rows, right_missing = _detector_cut_rows(right, temperature_min, temperature_max)
    missing = sorted(set(left_missing + right_missing))
    if missing or not left_rows or not right_rows:
        if not left_rows:
            missing.append("left_temperature_window_with_data")
        if not right_rows:
            missing.append("right_temperature_window_with_data")
        return {
            "workflow_name": "compare-psd-detector-cuts",
            "status": "missing_input",
            "missing_fields": sorted(set(missing)),
            "selected_records": [_source_id(left), _source_id(right)],
            "warnings": ["Warm/cool comparison requires PSD channels, intensity rows, and overlapping temperatures."],
        }

    pairs = _match_temperature_rows(left_rows, right_rows, temperature_tolerance)
    if not pairs:
        return {
            "workflow_name": "compare-psd-detector-cuts",
            "status": "missing_input",
            "missing_fields": ["overlapping_temperatures"],
            "selected_records": [_source_id(left), _source_id(right)],
            "warnings": [f"No temperature pairs matched within {temperature_tolerance:g} K."],
        }

    comparisons = []
    for left_row, right_row in pairs:
        channels = left_row["channels"]
        if channels != right_row["channels"]:
            channels = list(range(min(len(left_row["intensity"]), len(right_row["intensity"]))))
        left_vals = left_row["intensity"][: len(channels)]
        right_vals = right_row["intensity"][: len(channels)]
        diff = [left_vals[idx] - right_vals[idx] for idx in range(len(channels))]
        comparisons.append(
            {
                "left_temperature_K": left_row["temperature_K"],
                "right_temperature_K": right_row["temperature_K"],
                "temperature_delta_K": left_row["temperature_K"] - right_row["temperature_K"],
                "left_peak_channel": _peak_channel(channels, left_vals),
                "right_peak_channel": _peak_channel(channels, right_vals),
                "left_total": sum(left_vals),
                "right_total": sum(right_vals),
                "difference_l1": sum(abs(value) for value in diff),
                "channels": channels,
                "left_intensity": left_vals,
                "right_intensity": right_vals,
                "difference": diff,
            }
        )

    csv_path = output / "psd_detector_cut_comparison.csv"
    json_path = output / "psd_detector_cut_comparison.json"
    md_path = output / "psd_detector_cut_comparison.md"
    png_path = output / "psd_detector_cut_comparison.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_temperature_K", "right_temperature_K", "channel", "left_intensity", "right_intensity", "difference"])
        for item in comparisons:
            for channel, left_value, right_value, diff_value in zip(item["channels"], item["left_intensity"], item["right_intensity"], item["difference"]):
                writer.writerow([item["left_temperature_K"], item["right_temperature_K"], channel, left_value, right_value, diff_value])
    payload = {
        "workflow_name": "compare-psd-detector-cuts",
        "status": "success",
        "left_scan_id": left_scan_id,
        "right_scan_id": right_scan_id,
        "selected_records": [_source_id(left), _source_id(right)],
        "temperature_window": {"min": temperature_min, "max": temperature_max},
        "temperature_tolerance": temperature_tolerance,
        "matched_temperature_count": len(comparisons),
        "channel_count": len(comparisons[0]["channels"]),
        "comparison_summary": {
            "mean_difference_l1": sum(item["difference_l1"] for item in comparisons) / len(comparisons),
            "left_peak_channels": sorted({item["left_peak_channel"] for item in comparisons}),
            "right_peak_channels": sorted({item["right_peak_channel"] for item in comparisons}),
        },
        "artifacts": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "warnings": ["Comparison remains in detector-channel space; Q conversion requires reviewed PSD channel-to-A4/sample two-theta calibration."],
        "provenance": [
            ProvenanceRecord(source_type="catalog_record", source_id=_source_id(left), method="psd_detector_cut_comparison").model_dump(mode="json"),
            ProvenanceRecord(source_type="catalog_record", source_id=_source_id(right), method="psd_detector_cut_comparison").model_dump(mode="json"),
        ],
    }
    plot_warning = _write_detector_comparison_plot(comparisons, png_path)
    if plot_warning:
        payload["warnings"].append(plot_warning)
    else:
        payload["artifacts"]["png"] = str(png_path)
    json_path.write_text(json.dumps({**payload, "comparisons": comparisons}, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_detector_comparison_markdown(payload), encoding="utf-8")
    return payload


def check_psd_channel_artifact(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    scan_id: str,
    channels_of_interest: list[int | float],
    temperature_min: float | None = None,
    temperature_max: float | None = None,
    neighbor_width: int = 2,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_detector_cut_records(records, scan_id=scan_id, group_id=None)
    if not selected:
        return {
            "workflow_name": "check-psd-channel-artifact",
            "status": "missing_input",
            "missing_fields": ["scan_id"],
            "warnings": ["A PSD record is required for channel-artifact diagnostics."],
        }
    rows, missing = _detector_cut_rows(selected[0], temperature_min, temperature_max)
    if missing or not rows:
        if not rows:
            missing.append("temperature_window_with_data")
        return {
            "workflow_name": "check-psd-channel-artifact",
            "status": "missing_input",
            "missing_fields": sorted(set(missing)),
            "selected_records": [_source_id(selected[0])],
            "warnings": ["Channel-artifact diagnostics require PSD channels, intensity rows, and temperatures."],
        }
    if not channels_of_interest:
        channels_of_interest = sorted({_peak_channel(row["channels"], row["intensity"]) for row in rows if _peak_channel(row["channels"], row["intensity"]) is not None})
    channel_set = [float(item) for item in channels_of_interest]
    observations = []
    peak_channels = []
    for row in rows:
        peak = _peak_channel(row["channels"], row["intensity"])
        peak_channels.append(peak)
        row_observation = {"temperature_K": row["temperature_K"], "peak_channel": peak, "channels": []}
        for channel in channel_set:
            idx = _nearest_channel_index(row["channels"], channel)
            value = row["intensity"][idx]
            neighbor_values = [
                row["intensity"][candidate]
                for candidate in range(max(0, idx - neighbor_width), min(len(row["intensity"]), idx + neighbor_width + 1))
                if candidate != idx
            ]
            neighbor_mean = sum(neighbor_values) / len(neighbor_values) if neighbor_values else None
            row_observation["channels"].append(
                {
                    "channel": row["channels"][idx],
                    "requested_channel": channel,
                    "intensity": value,
                    "neighbor_mean": neighbor_mean,
                    "local_contrast": value - neighbor_mean if neighbor_mean is not None else None,
                    "is_peak_channel": row["channels"][idx] == peak,
                    "rank": _intensity_rank(row["intensity"], idx),
                }
            )
        observations.append(row_observation)
    summary_by_channel = []
    for channel in channel_set:
        channel_obs = [entry for row in observations for entry in row["channels"] if entry["requested_channel"] == channel]
        contrasts = [entry["local_contrast"] for entry in channel_obs if entry["local_contrast"] is not None]
        summary_by_channel.append(
            {
                "channel": channel,
                "peak_fraction": sum(1 for entry in channel_obs if entry["is_peak_channel"]) / len(channel_obs),
                "top3_fraction": sum(1 for entry in channel_obs if entry["rank"] <= 3) / len(channel_obs),
                "mean_local_contrast": sum(contrasts) / len(contrasts) if contrasts else None,
            }
        )
    stable_peak_fraction = max((peak_channels.count(channel) / len(peak_channels) for channel in set(peak_channels)), default=0.0)
    conclusions = _artifact_observations(summary_by_channel, stable_peak_fraction)
    csv_path = output / "psd_channel_artifact_check.csv"
    json_path = output / "psd_channel_artifact_check.json"
    md_path = output / "psd_channel_artifact_check.md"
    png_path = output / "psd_channel_artifact_check.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_K", "peak_channel", "channel", "intensity", "neighbor_mean", "local_contrast", "is_peak_channel", "rank"])
        for row in observations:
            for item in row["channels"]:
                writer.writerow([row["temperature_K"], row["peak_channel"], item["channel"], item["intensity"], item["neighbor_mean"], item["local_contrast"], item["is_peak_channel"], item["rank"]])
    payload = {
        "workflow_name": "check-psd-channel-artifact",
        "status": "success",
        "scan_id": scan_id,
        "selected_records": [_source_id(selected[0])],
        "temperature_window": {"min": temperature_min, "max": temperature_max},
        "channels_of_interest": channel_set,
        "neighbor_width": neighbor_width,
        "observation_count": len(observations),
        "summary_by_channel": summary_by_channel,
        "peak_channel_summary": {"unique_peak_channels": sorted(set(peak_channels)), "stable_peak_fraction": stable_peak_fraction},
        "diagnostic_observations": conclusions,
        "safe_conclusion": "This diagnostic reports detector-channel behavior only. It does not by itself prove a feature is real or an artifact.",
        "artifacts": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "warnings": ["Detector-channel artifact checks require human review and do not replace Q-space reconstruction or background checks."],
        "provenance": [ProvenanceRecord(source_type="catalog_record", source_id=_source_id(selected[0]), method="psd_channel_artifact_check").model_dump(mode="json")],
    }
    plot_warning = _write_channel_artifact_plot(observations, png_path)
    if plot_warning:
        payload["warnings"].append(plot_warning)
    else:
        payload["artifacts"]["png"] = str(png_path)
    json_path.write_text(json.dumps({**payload, "observations": observations}, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_channel_artifact_markdown(payload), encoding="utf-8")
    return payload


def hydrate_psd_records(records: list[dict[str, Any]], reader) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hydrated: list[dict[str, Any]] = []
    hydration_results: list[dict[str, Any]] = []
    for record in records:
        if _intensity(record) and _temperature_values(record):
            hydrated.append(record)
            continue
        path = record.get("path")
        if not path:
            hydrated.append(record)
            hydration_results.append({"source_id": _source_id(record), "status": "missing_input", "missing_fields": ["path"]})
            continue
        result = reader({"path": path})
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        output = getattr(result, "output", None) or {}
        hydration_results.append(
            {
                "source_id": _source_id(record),
                "path": path,
                "status": status_value,
                "audit_id": getattr(result, "audit_id", None),
                "warnings": [getattr(warning, "message", str(warning)) for warning in getattr(result, "warnings", [])],
                "errors": [getattr(error, "message", str(error)) for error in getattr(result, "errors", [])],
            }
        )
        if status_value != "success":
            hydrated.append(record)
            continue
        merged = dict(record)
        psd_map = output.get("psd_map") if isinstance(output, dict) else None
        if isinstance(psd_map, dict):
            merged["psd_map"] = psd_map
            merged.setdefault("psd_channel_axis", psd_map.get("psd_channel_axis"))
            merged.setdefault("temperature_axis", psd_map.get("temperature_axis"))
            merged.setdefault("intensity", psd_map.get("intensity"))
        metadata = output.get("metadata") if isinstance(output, dict) else None
        if isinstance(metadata, dict):
            merged["metadata"] = {**(merged.get("metadata") or {}), **metadata}
        hydrated.append(merged)
    return hydrated, hydration_results


def triage_records(query: FigureSourceQuery, records: list[dict[str, Any]], output_dir: str | Path | None = None, runner=None) -> dict[str, Any]:
    if runner is None:
        from tas_copilot_core.workflows.common import default_runner

        runner = default_runner()
    groups = group_psd_sources(records, runner=runner)
    rejected: list[RejectedCandidate] = []
    candidates: list[CandidateSourceData] = []
    target_x = (query.target_composition or {}).get("x_mn")
    sample = material_identity_record_via_registry(
        query.user_sample_formula,
        source_field="query.user_sample_formula",
        runner=runner,
        caller_workflow="figure-source-triage",
    )
    active = [
        HypothesisState(hypothesis_id=f"target-panel-{query.target_panel}", label=f"Panel {query.target_panel} x_Mn={target_x}"),
    ]
    for panel in query.alternate_panels:
        active.append(HypothesisState(hypothesis_id=f"alternate-panel-{panel.panel_id}", label=f"Panel {panel.panel_id} x_Mn={panel.composition.get('x_mn')}"))
    for record in records:
        classification = classify_source_family(record)
        material = material_identity_record_via_registry(
            _filename(record),
            source_field="filename",
            runner=runner,
            caller_workflow="figure-source-triage",
        )
        evidence = list(classification.evidence)
        mismatches: list[CandidateMismatch] = []
        score = classification.confidence
        if query.measurement_mode.lower() == "psd" and "psd" in classification.source_family:
            score += 0.1
        if "inelastic" in classification.source_family and _requires_elastic(query):
            rejected.append(
                RejectedCandidate(
                    candidate_id=_source_id(record),
                    rejection_reason="Rejected for elastic source query because record is a nonzero-energy line scan.",
                    rejected_by="validator",
                    evidence=evidence,
                )
            )
            continue
        if material.ambiguity_reason:
            mismatches.append(CandidateMismatch(mismatch_type="material_identity", summary=material.ambiguity_reason, severity="warning"))
        elif target_x is not None and material.x_variables.get("x_Mn") is not None and abs(float(target_x) - float(material.x_variables["x_Mn"])) > 0.025:
            mismatches.append(CandidateMismatch(mismatch_type="panel_composition", summary=f"Candidate x_Mn={material.x_variables['x_Mn']:g} does not match target panel x_Mn={float(target_x):g}."))
        candidates.append(
            CandidateSourceData(
                candidate_id=_source_id(record),
                candidate_type="datafile",
                interpretation=f"source_family:{classification.source_family}",
                datafile_ids=[_source_id(record)],
                total_score=round(min(score, 1.0), 4),
                confidence_label="high" if score >= 0.75 else "medium" if score >= 0.5 else "low",
                evidence=evidence,
                mismatches=mismatches,
                source_family=classification.source_family,
                material_identity=material,
                provenance=[ProvenanceRecord(source_type="source_record", source_id=_source_id(record), method="source_triage")],
            )
        )
    state = SourceTriageState(
        query=query,
        active_hypotheses=active,
        accepted_constraints={"measurement_kind": "elastic" if _requires_elastic(query) else "unknown", "measurement_mode": query.measurement_mode},
        candidate_groups=groups,
        candidates=sorted(candidates, key=lambda item: item.total_score, reverse=True),
        rejected_candidates=rejected,
        next_checks=[
            "Review material identity for ambiguous LuFeMnO3/LuMnFeO3 filenames against sample logs or paper mappings.",
            "Run channel x temperature reconstruction for single-file PSD temperature maps.",
            "Do not run Q x temperature reconstruction until PSD calibration is reviewed.",
        ],
        provenance=[ProvenanceRecord(source_type="workflow", source_id="figure-source-triage", method="deterministic_source_triage")],
    )
    payload = {
        "workflow_name": "figure-source-triage",
        "status": "success",
        "panel_sample_warnings": _panel_sample_warnings(query, sample),
        "source_triage_state": state.model_dump(mode="json"),
        "triage_summary": _triage_summary(state),
        "safe_conclusion": _safe_conclusion(state),
    }
    if output_dir:
        payload["artifacts"] = _write_triage_reports(payload, output_dir)
    return payload


def _write_triage_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = payload["source_triage_state"]
    (output / "source_triage_state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    (output / "rejected_candidates.json").write_text(json.dumps(state["rejected_candidates"], indent=2, sort_keys=True), encoding="utf-8")
    review_template = _write_figure_source_review_template(output, payload)
    artifacts = write_report_bundle(output, payload, markdown_renderer=_triage_markdown, json_filename="figure_source_report.json", markdown_filename="figure_source_report.md")
    artifacts["figure_source_review_template_jsonl"] = str(review_template)
    return artifacts


def _write_figure_source_review_template(output: Path, payload: dict[str, Any]) -> Path:
    state = payload["source_triage_state"]
    query = state.get("query") or {}
    path = output / "figure_source_review_template.jsonl"
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(state.get("candidates") or [], start=1):
        rows.append(
            {
                "paper_id": query.get("paper_id"),
                "figure_id": query.get("figure_id"),
                "panel_id": query.get("target_panel") or query.get("panel_id") or "unknown-panel",
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": candidate.get("candidate_type"),
                "candidate_rank": index,
                "score": candidate.get("total_score"),
                "confidence_label": candidate.get("confidence_label"),
                "candidate_status": "candidate",
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence", "ambiguous", "not_source", "duplicate"],
                "evidence_channels_relied_on": ["evidence_channel_panel_label", "evidence_channel_material_identity", "evidence_channel_scan_family"],
                "source_workflow": "figure-source-triage",
                "review_notes": "",
                "human_review_required": True,
                "provenance": "figure-source-triage review template",
            }
        )
    for index, candidate in enumerate(state.get("rejected_candidates") or [], start=1):
        rows.append(
            {
                "paper_id": query.get("paper_id"),
                "figure_id": query.get("figure_id"),
                "panel_id": query.get("target_panel") or query.get("panel_id") or "unknown-panel",
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": "rejected_candidate",
                "candidate_rank": len(rows) + index,
                "score": 0.0,
                "confidence_label": "rejected_by_workflow",
                "candidate_status": "rejected_by_workflow",
                "rejected_reason": candidate.get("rejection_reason"),
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence", "ambiguous", "not_source", "duplicate"],
                "evidence_channels_relied_on": ["evidence_channel_scan_family"],
                "source_workflow": "figure-source-triage",
                "review_notes": "",
                "human_review_required": True,
                "provenance": "figure-source-triage review template",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _triage_markdown(payload: dict[str, Any]) -> str:
    state = payload["source_triage_state"]
    lines = ["# Figure Source Triage", "", "## Warnings"]
    lines.extend(f"- {warning}" for warning in payload.get("panel_sample_warnings", []) or ["None"])
    summary = payload.get("triage_summary", {})
    lines.extend(["", "## Triage Summary"])
    lines.append(f"- Recommended source branch: {summary.get('recommended_source_branch', 'none')}")
    lines.append(f"- Calibration blocker: {summary.get('calibration_blocker', 'not evaluated')}")
    lines.append(f"- Rejected incompatible candidates: {summary.get('rejected_candidate_count', 0)}")
    if summary.get("top_group_next_checks"):
        lines.append("- Top-group next checks:")
        lines.extend(f"  - {item}" for item in summary["top_group_next_checks"])
    lines.extend(["", "## Candidate Groups"])
    for group in summary.get("ranked_groups", state.get("candidate_groups", [])):
        lines.append(
            f"- `{group['group_id']}`: {group['group_type']} members={group.get('member_count', len(group.get('member_datafile_ids', [])))} "
            f"confidence={group.get('grouping_confidence')} priority={group.get('review_priority', 'unknown')} "
            f"temp={group.get('temperature_range', {})} angle={group.get('q_or_angle_coverage', {})} material={group.get('material_status', 'unknown')}"
        )
    lines.extend(["", "## Rejected Candidates"])
    for rejected in state.get("rejected_candidates", []):
        lines.append(f"- `{rejected['candidate_id']}`: {rejected['rejection_reason']}")
    lines.extend(["", "## Safe Conclusion", "", payload.get("safe_conclusion", "")])
    return "\n".join(lines)


def _panel_sample_warnings(query: FigureSourceQuery, sample: MaterialIdentityRecord) -> list[str]:
    warnings: list[str] = []
    sample_x = sample.x_variables.get("x_Mn")
    target_x = (query.target_composition or {}).get("x_mn")
    if sample_x is not None and target_x is not None and abs(float(sample_x) - float(target_x)) > 0.025:
        warnings.append(
            f"Panel/sample mismatch: user sample {query.user_sample_formula} has x_Mn={float(sample_x):g}, while panel {query.target_panel} has x_Mn={float(target_x):g}."
        )
    return warnings


def _safe_conclusion(state: SourceTriageState) -> str:
    rejected = len(state.rejected_candidates)
    groups = len(state.candidate_groups)
    return (
        f"Found {groups} PSD candidate group(s) and rejected {rejected} incompatible candidate(s). "
        "Candidate groups are not definitive panel-source claims until reconstruction, calibration, and artifact diagnostics are reviewed."
    )


def _triage_summary(state: SourceTriageState) -> dict[str, Any]:
    ranked_groups = sorted((_group_summary(group) for group in state.candidate_groups), key=lambda item: (-item["review_priority_score"], item["group_id"]))
    top = ranked_groups[0] if ranked_groups else None
    return {
        "recommended_source_branch": _recommended_branch(top),
        "ranked_groups": ranked_groups,
        "rejected_candidate_count": len(state.rejected_candidates),
        "rejected_candidate_reasons": sorted({item.rejection_reason for item in state.rejected_candidates}),
        "calibration_blocker": "Q-space interpretation of PSD maps requires accepted PSD calibration and a reviewed PSD channel-to-A4/sample two-theta adapter; Q from A4 is formulaic for elastic diffraction.",
        "top_group_next_checks": _top_group_next_checks(top),
    }


def _group_summary(group: PSDSourceGroup) -> dict[str, Any]:
    material = group.material_identity_summary or {}
    resolved = material.get("resolved_x_mn") or []
    ambiguous = int(material.get("ambiguous_count") or 0)
    priority = 0.0
    if group.group_type == "single_file_psd_temperature_map":
        priority += 0.45
    elif group.group_type == "multi_file_psd_angle_temperature_series":
        priority += 0.3
    if group.temperature_values:
        priority += 0.2
    if group.psd_channel_count:
        priority += 0.15
    if group.q_or_angle_coverage:
        priority += 0.1
    if ambiguous:
        priority -= 0.1
    missing = set(group.missing_metadata)
    if "temperature" in missing or "psd_channel_axis" in missing:
        priority -= 0.2
    material_status = "ambiguous" if ambiguous else f"x_Mn={','.join(str(item) for item in resolved)}" if resolved else "unknown"
    return {
        "group_id": group.group_id,
        "group_type": group.group_type,
        "member_count": len(group.member_datafile_ids),
        "member_datafile_ids": group.member_datafile_ids,
        "member_scan_ids": group.member_scan_ids,
        "grouping_confidence": group.grouping_confidence,
        "review_priority_score": round(max(0.0, min(priority, 1.0)), 4),
        "review_priority": "high" if priority >= 0.65 else "medium" if priority >= 0.35 else "low",
        "material_status": material_status,
        "temperature_range": group.temperature_range,
        "temperature_values": group.temperature_values,
        "q_or_angle_coverage": group.q_or_angle_coverage,
        "psd_channel_count": group.psd_channel_count,
        "missing_metadata": group.missing_metadata,
    }


def _recommended_branch(group: dict[str, Any] | None) -> str:
    if not group:
        return "No PSD source branch found."
    if group["group_type"] == "single_file_psd_temperature_map":
        return f"{group['group_id']} is the first branch to inspect because it is a single-file PSD temperature map."
    if group["group_type"] == "multi_file_psd_angle_temperature_series":
        return f"{group['group_id']} is the first branch to inspect because it is a multi-file PSD angle/temperature series."
    return f"{group['group_id']} is the highest-priority available branch."


def _top_group_next_checks(group: dict[str, Any] | None) -> list[str]:
    if not group:
        return ["Load or catalog source records before triage."]
    checks = [
        "Inspect raw detector-channel cuts before any interpolation or smoothing.",
        "Keep Q-space claims blocked until PSD calibration review is accepted.",
    ]
    if group.get("material_status") == "ambiguous":
        checks.append("Resolve ambiguous LuFeMnO3/LuMnFeO3 composition using sample logs, paper mapping, or human review.")
    if "temperature" in group.get("missing_metadata", []):
        checks.append("Recover temperature metadata before treating this as a temperature-map source.")
    if "angle_axis" in group.get("missing_metadata", []):
        checks.append("Recover detector-angle/A4 metadata before detector-angle reconstruction.")
    return checks


def _requires_elastic(query: FigureSourceQuery) -> bool:
    return "elastic" in " ".join([query.measurement_mode, query.notes or ""]).lower() or query.target_composition.get("measurement_kind") == "elastic"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _source_id(record: dict[str, Any]) -> str:
    return str(record.get("datafile_id") or record.get("scan_id") or _filename(record))


def _filename(record: dict[str, Any]) -> str:
    return str(record.get("filename") or record.get("file") or Path(str(record.get("path") or "")).name or _source_id(record))


def _experiment(record: dict[str, Any]) -> str:
    return str(record.get("experiment_id") or _metadata(record).get("native_experiment_id") or "unknown")


def _family_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\d+$", "", stem)
    stem = re.sub(r"(?i)(warm|cool)(oc)?$", "", stem)
    return stem.strip("_-") or "unknown"


def _blob(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _energy(record: dict[str, Any]) -> float | None:
    for key in ["energy_transfer_meV", "energy", "E"]:
        if record.get(key) is not None:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                pass
    metadata = _metadata(record)
    for key in ["energy_transfer_meV", "energy", "E"]:
        if metadata.get(key) is not None:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                pass
    return None


def _temperature_values(record: dict[str, Any]) -> list[float]:
    values = record.get("temperature_axis") or record.get("temperature_values")
    if values:
        return [float(item) for item in values]
    metadata = _metadata(record)
    for key in ["temperature_axis", "temperature_values"]:
        if metadata.get(key):
            return [float(item) for item in metadata[key]]
    if record.get("temperature_K") is not None:
        return [float(record["temperature_K"])]
    if metadata.get("temperature_K") is not None:
        return [float(metadata["temperature_K"])]
    temp_range = record.get("temperature_range") or metadata.get("temperature_range")
    if isinstance(temp_range, dict) and temp_range.get("min") is not None and temp_range.get("max") is not None:
        return [float(temp_range["min"]), float(temp_range["max"])]
    return []


def _psd_channels(record: dict[str, Any]) -> list[float]:
    channels = record.get("psd_channel_axis") or _metadata(record).get("psd_channel_axis")
    if channels:
        return [float(item) for item in channels]
    intensity = _intensity(record)
    return list(range(len(intensity[0]))) if intensity else []


def _intensity(record: dict[str, Any]) -> list[list[float]]:
    intensity = record.get("intensity") or _metadata(record).get("intensity")
    if not intensity:
        psd_map = record.get("psd_map") or _metadata(record).get("psd_map")
        if isinstance(psd_map, dict):
            intensity = psd_map.get("intensity")
    return [[float(value) for value in row] for row in intensity] if intensity else []


def _psd_channel_count(record: dict[str, Any]) -> int:
    return len(_psd_channels(record))


def _energy_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_energy(member) for member in members if _energy(member) is not None]
    return {"values": values, "min": min(values), "max": max(values)} if values else {}


def _angle_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for member in members:
        for source in [member, _metadata(member)]:
            if source.get("a4") is not None:
                values.append(float(source["a4"]))
            two_theta = source.get("two_theta_axis")
            if isinstance(two_theta, dict) and two_theta.get("value") is not None:
                values.append(float(two_theta["value"]))
    return {"axis": "A4/two_theta", "min": min(values), "max": max(values)} if values else {}


def _group_missing_metadata(members: list[dict[str, Any]]) -> list[str]:
    missing = set()
    if not any(_temperature_values(member) for member in members):
        missing.add("temperature")
    if not any(_psd_channels(member) for member in members):
        missing.add("psd_channel_axis")
    if not any(_angle_summary([member]) for member in members):
        missing.add("angle_axis")
    return sorted(missing)


def select_detector_cut_records(records: list[dict[str, Any]], scan_id: str | None, group_id: str | None) -> list[dict[str, Any]]:
    if group_id:
        group = next((item for item in group_psd_sources(records) if item.group_id == group_id), None)
        if group is None:
            return []
        ids = set(group.member_datafile_ids) | set(group.member_scan_ids)
        return [record for record in records if _source_id(record) in ids or str(record.get("scan_id")) in ids]
    if scan_id:
        return [record for record in records if _source_id(record) == scan_id or str(record.get("scan_id")) == scan_id]
    return []


def _detector_cut_rows(record: dict[str, Any], temperature_min: float | None, temperature_max: float | None) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    channels = _psd_channels(record)
    temperatures = _temperature_values(record)
    intensity = _intensity(record)
    if not channels:
        missing.append("psd_channel_axis")
    if not temperatures:
        missing.append("temperature_axis")
    if not intensity:
        missing.append("psd_intensity")
    if missing:
        return [], missing
    for index, temp in enumerate(temperatures):
        if temperature_min is not None and temp < temperature_min:
            continue
        if temperature_max is not None and temp > temperature_max:
            continue
        if index >= len(intensity):
            missing.append("psd_intensity_row_for_temperature")
            continue
        rows.append({"source_id": _source_id(record), "temperature_K": temp, "channels": channels, "intensity": intensity[index]})
    return rows, missing


def _match_temperature_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], tolerance: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    used: set[int] = set()
    for left in left_rows:
        candidates = [
            (idx, abs(left["temperature_K"] - right["temperature_K"]), right)
            for idx, right in enumerate(right_rows)
            if idx not in used and abs(left["temperature_K"] - right["temperature_K"]) <= tolerance
        ]
        if candidates:
            idx, _, right = min(candidates, key=lambda item: item[1])
            used.add(idx)
            pairs.append((left, right))
    return pairs


def _peak_channel(channels: list[float], values: list[float]) -> float | None:
    if not channels or not values:
        return None
    idx = max(range(min(len(channels), len(values))), key=lambda item: values[item])
    return channels[idx]


def _nearest_channel_index(channels: list[float], requested: float) -> int:
    return min(range(len(channels)), key=lambda idx: abs(channels[idx] - requested))


def _intensity_rank(values: list[float], idx: int) -> int:
    target = values[idx]
    return 1 + sum(1 for value in values if value > target)


def _artifact_observations(summary_by_channel: list[dict[str, Any]], stable_peak_fraction: float) -> list[str]:
    observations = []
    if stable_peak_fraction > 0.75:
        observations.append("Peak detector channel is highly stable across the selected temperature range; check whether this follows a fixed detector/channel feature.")
    elif stable_peak_fraction < 0.35:
        observations.append("Peak detector channel moves across the selected temperature range; a fixed channel-boundary artifact is less strongly indicated by this diagnostic alone.")
    for item in summary_by_channel:
        if item["peak_fraction"] > 0.5:
            observations.append(f"Requested channel {item['channel']:g} is the peak channel in more than half of selected cuts.")
        elif item["top3_fraction"] > 0.75:
            observations.append(f"Requested channel {item['channel']:g} remains among the top three detector channels in most selected cuts.")
        if item["mean_local_contrast"] is not None and item["mean_local_contrast"] < 0:
            observations.append(f"Requested channel {item['channel']:g} is weaker than neighboring channels on average.")
    if not observations:
        observations.append("No single detector-channel artifact signature dominates this limited diagnostic.")
    return observations


def _write_detector_cut_plot(rows: list[dict[str, Any]], png_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting install
        return f"matplotlib plot was not written: {exc}"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        ax.plot(row["channels"], row["intensity"], marker="o", label=f"{row['temperature_K']:g} K")
    ax.set_xlabel("PSD detector channel")
    ax.set_ylabel("Intensity")
    ax.set_title("PSD detector-channel cuts")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return None


def _write_channel_artifact_plot(observations: list[dict[str, Any]], png_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting install
        return f"matplotlib plot was not written: {exc}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    temperatures = [row["temperature_K"] for row in observations]
    peak_channels = [row["peak_channel"] for row in observations]
    axes[0].plot(temperatures, peak_channels, marker="o")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Peak PSD channel")
    axes[0].set_title("Peak channel trajectory")
    for channel_entry in observations[0]["channels"]:
        channel = channel_entry["channel"]
        contrasts = [next(item["local_contrast"] for item in row["channels"] if item["channel"] == channel) for row in observations]
        axes[1].plot(temperatures, contrasts, marker="o", label=f"ch {channel:g}")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Local contrast")
    axes[1].set_title("Requested-channel contrast")
    axes[1].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return None


def _write_detector_comparison_plot(comparisons: list[dict[str, Any]], png_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting install
        return f"matplotlib plot was not written: {exc}"
    first = comparisons[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(first["channels"], first["left_intensity"], marker="o", label=f"left {first['left_temperature_K']:g} K")
    axes[0].plot(first["channels"], first["right_intensity"], marker="s", label=f"right {first['right_temperature_K']:g} K")
    axes[0].set_xlabel("PSD detector channel")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Matched detector cut")
    axes[0].legend(loc="best", fontsize="small")
    mean_diff = [sum(item["difference"][idx] for item in comparisons) / len(comparisons) for idx in range(len(first["channels"]))]
    axes[1].plot(first["channels"], mean_diff, marker="o")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("PSD detector channel")
    axes[1].set_ylabel("Mean left-right intensity")
    axes[1].set_title("Mean difference")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return None


def _detector_cuts_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Detector-Channel Cuts",
        "",
        f"Status: `{payload['status']}`",
        f"Selected records: {', '.join(payload.get('selected_records', []))}",
        f"Temperature window: {payload.get('temperature_window')}",
        f"Cut count: {payload.get('cut_count')}",
        "",
        "These are detector-channel cuts, not Q cuts. Q conversion remains blocked until reviewed PSD calibration exists.",
        "",
        "## Artifacts",
    ]
    for key, path in payload.get("artifacts", {}).items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines)


def _channel_artifact_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Channel-Artifact Diagnostic",
        "",
        f"Status: `{payload['status']}`",
        f"Scan: `{payload['scan_id']}`",
        f"Temperature window: {payload.get('temperature_window')}",
        f"Channels of interest: {payload.get('channels_of_interest')}",
        "",
        "## Observations",
    ]
    lines.extend(f"- {item}" for item in payload.get("diagnostic_observations", []))
    lines.extend(["", "## Summary By Channel"])
    for item in payload.get("summary_by_channel", []):
        lines.append(f"- channel {item['channel']:g}: peak_fraction={item['peak_fraction']:.3f}, top3_fraction={item['top3_fraction']:.3f}, mean_local_contrast={item['mean_local_contrast']}")
    lines.extend(["", "## Safe Conclusion", "", payload.get("safe_conclusion", "")])
    lines.extend(["", "## Artifacts"])
    for key, path in payload.get("artifacts", {}).items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines)


def _detector_comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Detector-Channel Cut Comparison",
        "",
        f"Status: `{payload['status']}`",
        f"Left: `{payload['left_scan_id']}`",
        f"Right: `{payload['right_scan_id']}`",
        f"Matched temperatures: {payload.get('matched_temperature_count')}",
        f"Summary: `{payload.get('comparison_summary')}`",
        "",
        "This comparison is detector-channel space only. It does not establish feature reality or Q-space location by itself.",
        "",
        "## Artifacts",
    ]
    for key, path in payload.get("artifacts", {}).items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines)
