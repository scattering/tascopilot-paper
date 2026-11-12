from __future__ import annotations

import csv
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from tas_copilot_contracts import PSDCalibrationRecord, ProvenanceRecord


ALLOWED_STATUSES = {"candidate_unreviewed", "proposed", "reviewed", "accepted", "rejected", "deprecated", "unavailable"}
MEV_ANGSTROM_SQUARED = 81.804209


def default_psd_calibration_registry_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "psd_calibration" / "registry.yaml"


def load_psd_calibration_registry(path: str | Path | None = None) -> list[PSDCalibrationRecord]:
    registry_path = Path(path).expanduser() if path else default_psd_calibration_registry_path()
    if not registry_path.exists():
        return []
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return [PSDCalibrationRecord(**item) for item in payload.get("psd_calibrations", [])]


def accepted_psd_calibrations(path: str | Path | None = None, instrument: str | None = None, detector_mode: str | None = None) -> list[PSDCalibrationRecord]:
    records = load_psd_calibration_registry(path)
    accepted = [record for record in records if record.status == "accepted"]
    if instrument:
        accepted = [record for record in accepted if record.instrument.lower() == instrument.lower()]
    if detector_mode:
        accepted = [record for record in accepted if (record.detector_mode or "").lower() == detector_mode.lower()]
    return accepted


def validate_psd_calibration_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path).expanduser() if path else default_psd_calibration_registry_path()
    records = load_psd_calibration_registry(registry_path)
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    for record in records:
        if record.calibration_id in ids:
            errors.append(f"duplicate calibration_id: {record.calibration_id}")
        ids.add(record.calibration_id)
        if record.status not in ALLOWED_STATUSES:
            errors.append(f"{record.calibration_id}: invalid status {record.status!r}")
        if record.status == "accepted":
            if not record.reviewed_by or not record.reviewed_at:
                errors.append(f"{record.calibration_id}: accepted calibration requires reviewed_by and reviewed_at")
            if not record.channel_mapping:
                errors.append(f"{record.calibration_id}: accepted calibration requires channel_mapping")
            if not record.detector_geometry:
                errors.append(f"{record.calibration_id}: accepted calibration requires detector_geometry")
            if not record.validation_evidence:
                errors.append(f"{record.calibration_id}: accepted calibration requires validation_evidence")
        elif record.channel_mapping or record.detector_geometry:
            warnings.append(f"{record.calibration_id}: non-accepted calibration has mapping/geometry metadata that must not unlock Q conversion")
    return {
        "status": "error" if errors else "ok",
        "registry_path": str(registry_path),
        "record_count": len(records),
        "accepted_count": sum(1 for record in records if record.status == "accepted"),
        "status_counts": _status_counts(records),
        "errors": errors,
        "warnings": warnings,
    }


def psd_calibration_report(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path).expanduser() if path else default_psd_calibration_registry_path()
    records = load_psd_calibration_registry(registry_path)
    validation = validate_psd_calibration_registry(registry_path)
    return {
        "status": validation["status"],
        "registry_path": str(registry_path),
        "record_count": len(records),
        "accepted_count": validation["accepted_count"],
        "status_counts": validation["status_counts"],
        "records": [record.model_dump(mode="json") for record in records],
        "q_temperature_unlock_available": validation["accepted_count"] > 0 and validation["status"] == "ok",
        "safe_conclusion": (
            "No accepted PSD calibration is registered; Q x temperature reconstruction remains blocked."
            if validation["accepted_count"] == 0
            else "Accepted PSD calibration records exist. Conversion code still must verify detector mode, date range, geometry, and lineage before use."
        ),
        "errors": validation["errors"],
        "warnings": validation["warnings"],
    }


def discover_psd_dat_calibrations(calibration_dir: str | Path, *, path_only: bool = False) -> dict[str, Any]:
    root = Path(calibration_dir).expanduser()
    if not root.exists():
        return {"status": "missing_input", "missing_fields": ["calibration_dir"], "calibration_dir": str(root), "records": []}
    records = [_parse_psd_dat(path, read_contents=not path_only) for path in sorted(root.glob("**/*.dat"))]
    return {
        "status": "ok",
        "calibration_dir": str(root),
        "record_count": len(records),
        "records": records,
        "safe_conclusion": "These DAT files are calibration candidates only. They do not unlock Q conversion until date, detector mode, analyzer-detector distance, and review status are accepted.",
    }


def match_psd_calibration(calibration_dir: str | Path, data_file: str | Path, *, path_only: bool = False) -> dict[str, Any]:
    source = _parse_bt7_source(data_file)
    discovered = discover_psd_dat_calibrations(calibration_dir, path_only=path_only)
    if discovered["status"] != "ok":
        return discovered
    matches = [_score_calibration(record, source) for record in discovered["records"]]
    matches.sort(key=lambda item: (-item["score"], item["calibration_file"]))
    missing = []
    if source.get("analyzer_detector_translation") is None:
        missing.append("analyzer_detector_translation")
    if source.get("detector_mode") is None:
        missing.append("detector_mode")
    payload = {
        "status": "ok" if matches else "missing_input",
        "source_file": source,
        "calibration_dir": str(Path(calibration_dir).expanduser()),
        "matches": matches,
        "best_match": matches[0] if matches else None,
        "missing_fields": missing,
        "safe_conclusion": _match_safe_conclusion(matches, missing),
        "matching_policy": {
            "ice_control_software_note": "Older ICE-era BT-7 files do not store AnaTrans/analyzer-to-detector translation distance in the scan file.",
            "calibration_lag_policy": "Exact calibration dates up to one week after the measurement are treated as plausible; filename month/year evidence is lower confidence.",
            "q_formula_policy": "For elastic diffraction, Q is deterministic from A4/sample two-theta and wavelength: Q = 4*pi*sin(A4/2)/lambda. PSD review is needed to map each detector channel to A4/sample two-theta.",
            "path_only": path_only,
        },
        "provenance": [
            ProvenanceRecord(source_type="bt7_file", source_id=str(Path(data_file).expanduser()), method="psd_calibration_match").model_dump(mode="json"),
            ProvenanceRecord(source_type="calibration_dir", source_id=str(Path(calibration_dir).expanduser()), method="psd_calibration_match").model_dump(mode="json"),
        ],
    }
    return payload


def compare_psd_calibration_matches(
    calibration_dir: str | Path,
    data_file: str | Path,
    *,
    top_k: int = 5,
    output_dir: str | Path | None = None,
    path_only: bool = False,
) -> dict[str, Any]:
    """Hydrate and compare candidate PSD DAT files without accepting a calibration."""
    match = match_psd_calibration(calibration_dir, data_file, path_only=path_only)
    if match.get("status") != "ok":
        return match
    source = match["source_file"]
    comparisons = [_calibration_axis_comparison(item, source) for item in match["matches"][:top_k]]
    payload = {
        "status": "ok",
        "source_file": source,
        "calibration_dir": match["calibration_dir"],
        "top_k": top_k,
        "comparisons": comparisons,
        "best_match": comparisons[0] if comparisons else None,
        "matching_policy": match["matching_policy"],
        "safe_conclusion": (
            "Calibration DAT files were parsed and compared, but every result remains candidate_unreviewed. "
            "Do not enable Q x temperature reconstruction until an instrument scientist accepts the PSD channel-to-A4/sample two-theta mapping, "
            "AnaTrans/distance choice, sign convention, and lineage."
        ),
        "provenance": match["provenance"],
    }
    if output_dir is not None:
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "psd_calibration_comparison.json"
        md_path = output / "psd_calibration_comparison.md"
        csv_path = output / "psd_calibration_comparison.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(_comparison_markdown(payload), encoding="utf-8")
        _write_comparison_csv(comparisons, csv_path)
        payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path)}
    return payload


def wavelength_from_energy_mev(energy_mev: float) -> float:
    if energy_mev <= 0:
        raise ValueError("energy_mev must be positive")
    return math.sqrt(MEV_ANGSTROM_SQUARED / energy_mev)


def elastic_q_from_two_theta(two_theta_degrees: float, *, energy_mev: float | None = None, wavelength_angstrom: float | None = None) -> float:
    if wavelength_angstrom is None:
        if energy_mev is None:
            raise ValueError("energy_mev or wavelength_angstrom is required")
        wavelength_angstrom = wavelength_from_energy_mev(energy_mev)
    if wavelength_angstrom <= 0:
        raise ValueError("wavelength_angstrom must be positive")
    theta_radians = math.radians(two_theta_degrees / 2.0)
    return 4.0 * math.pi * math.sin(theta_radians) / wavelength_angstrom


def psd_channel_two_theta(a4_degrees: float, relative_a4_degrees: float, *, sign: int = 1) -> float:
    if sign not in {-1, 1}:
        raise ValueError("sign must be +1 or -1")
    return a4_degrees + sign * relative_a4_degrees


def psd_channel_q_axis(
    a4_degrees: float,
    relative_a4_degrees: list[float],
    *,
    energy_mev: float | None = None,
    wavelength_angstrom: float | None = None,
    sign: int = 1,
) -> dict[str, Any]:
    two_theta = [psd_channel_two_theta(a4_degrees, offset, sign=sign) for offset in relative_a4_degrees]
    q_axis = [elastic_q_from_two_theta(value, energy_mev=energy_mev, wavelength_angstrom=wavelength_angstrom) for value in two_theta]
    return {
        "status": "computed_from_candidate_channel_to_a4_mapping",
        "two_theta_axis": two_theta,
        "q_axis": q_axis,
        "q_axis_source": "Q = 4*pi*sin(two_theta/2)/lambda",
        "warnings": [
            "For elastic diffraction, Q is computed directly from A4/sample two-theta and wavelength.",
            "Do not treat this as an accepted Q axis unless the PSD channel-to-A4 mapping, mask, sign convention, and calibration record are reviewed and accepted.",
        ],
    }


def reviewed_psd_channel_to_a4_mapping(
    *,
    source_a4_degrees: float,
    detector_mode: str | None = None,
    source_date: str | None = None,
    registry_path: str | Path | None = None,
    calibration_dirs: list[str | Path] | None = None,
    energy_mev: float | None = None,
) -> dict[str, Any]:
    """Return a reviewed channel-to-A4 axis when a scoped accepted record applies.

    This is deliberately a narrow gate around accepted registry records. It does
    not discover or promote candidate calibrations, and it does not make a
    publication-like Q map. If the accepted record cannot be matched to source
    metadata and a local DAT file, the result fails closed.
    """

    accepted = accepted_psd_calibrations(registry_path, instrument="BT-7")
    matches = [
        record
        for record in accepted
        if _record_matches_source_metadata(record, detector_mode=detector_mode, source_date=source_date)
    ]
    if not matches:
        return {
            "status": "missing_input",
            "missing_fields": ["accepted_psd_calibration_for_source"],
            "safe_use": False,
            "warnings": [
                "No accepted PSD calibration registry record matched the source detector mode/date.",
                "Candidate DAT files must not unlock channel-to-A4 conversion until reviewed and accepted.",
            ],
        }
    record = matches[0]
    calibration_file = _find_registered_calibration_file(record, calibration_dirs or [])
    if calibration_file is None:
        return {
            "status": "missing_input",
            "calibration_id": record.calibration_id,
            "missing_fields": ["registered_psd_dat_file"],
            "safe_use": False,
            "warnings": [
                "An accepted PSD calibration registry record matched, but the DAT row file was not found in the supplied calibration directories.",
                "Channel-to-A4 conversion remains unavailable until the registered DAT file is available.",
            ],
        }
    parsed = _parse_psd_dat(calibration_file, read_contents=True)
    rows = parsed.get("channels") or []
    expected_rows = record.channel_mapping.get("row_count")
    warnings: list[str] = []
    if expected_rows and len(rows) != int(expected_rows):
        return {
            "status": "error",
            "calibration_id": record.calibration_id,
            "calibration_file": str(calibration_file),
            "safe_use": False,
            "missing_fields": ["expected_psd_dat_row_count"],
            "warnings": [f"Registered row_count is {expected_rows}, but parsed DAT rows are {len(rows)}."],
        }
    sign = 1 if str(record.channel_mapping.get("sign_convention") or "plus_relative_a4").lower().startswith("plus") else -1
    relative_a4 = [float(row["relative_a4"]) for row in rows]
    two_theta = [psd_channel_two_theta(float(source_a4_degrees), value, sign=sign) for value in relative_a4]
    q_payload = None
    if energy_mev is not None:
        q_payload = psd_channel_q_axis(float(source_a4_degrees), relative_a4, energy_mev=float(energy_mev), sign=sign)
        q_preview = _formula_preview(q_payload, [])
        q_preview["q_axis"] = q_payload["q_axis"]
    else:
        q_preview = {"status": "not_requested"}
    warnings.extend(
        [
            "This reviewed mapping is scoped to the accepted registry record and matching source metadata.",
            "For elastic diffraction, Q is formulaic after channel-to-A4 mapping; publication-like maps still require documented binning, smoothing, normalization, mask, and background provenance.",
        ]
    )
    return {
        "status": "success",
        "safe_use": True,
        "calibration_id": record.calibration_id,
        "calibration_file": str(calibration_file),
        "reviewed_by": record.reviewed_by,
        "reviewed_at": record.reviewed_at,
        "source_a4_degrees": float(source_a4_degrees),
        "detector_mode": detector_mode,
        "source_date": source_date,
        "channel_mapping_policy": record.channel_mapping.get("channel_to_a4_policy"),
        "sign_convention": record.channel_mapping.get("sign_convention"),
        "relative_a4_degrees": relative_a4,
        "two_theta_axis_degrees": two_theta,
        "efficiency": [float(row["efficiency"]) for row in rows],
        "masked_one_based_channels": (record.channel_mapping.get("mask_policy") or {}).get("masked_one_based_channels") or parsed.get("masked_channels") or [],
        "q_preview": q_preview,
        "warnings": warnings,
        "provenance": [item.model_dump(mode="json") for item in record.provenance],
    }


def _record_matches_source_metadata(record: PSDCalibrationRecord, *, detector_mode: str | None, source_date: str | None) -> bool:
    if detector_mode and record.detector_mode and record.detector_mode.lower() not in detector_mode.lower():
        return False
    if source_date and record.date_range:
        parsed = _parse_date_only(source_date)
        start = _parse_date_only(record.date_range.get("start"))
        end = _parse_date_only(record.date_range.get("end"))
        if parsed and start and parsed < start:
            return False
        if parsed and end and parsed > end:
            return False
    return True


def _parse_date_only(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _find_registered_calibration_file(record: PSDCalibrationRecord, calibration_dirs: list[str | Path]) -> Path | None:
    filename = str(record.channel_mapping.get("source_filename") or "").strip()
    if not filename:
        return None
    search_dirs = [Path(item).expanduser() for item in calibration_dirs if item]
    seen: set[Path] = set()
    for root in search_dirs:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / filename
        if candidate.exists():
            return candidate
        if root.exists():
            matches = sorted(root.glob(f"**/{filename}"))
            if matches:
                return matches[0]
    return None


def _calibration_axis_comparison(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("channels") or []
    first_a4 = source.get("A4")
    ef = source.get("Ef")
    q_preview = None
    missing_formula_fields = []
    if first_a4 is None:
        missing_formula_fields.append("source_A4")
    if ef is None:
        missing_formula_fields.append("source_Ef")
    if rows and not missing_formula_fields:
        relative = [float(row["relative_a4"]) for row in rows]
        q_preview = psd_channel_q_axis(float(first_a4), relative, energy_mev=float(ef))
    return {
        "filename": record.get("filename"),
        "calibration_file": record.get("calibration_file"),
        "score": record.get("score"),
        "review_status": record.get("review_status", record.get("status")),
        "safe_use": False,
        "analyzer_detector_translation": record.get("analyzer_detector_translation"),
        "source_reference": record.get("source_reference"),
        "created_date": record.get("created_date"),
        "metadata_mode": record.get("metadata_mode"),
        "row_count": record.get("row_count", 0),
        "relative_a4_range": record.get("relative_a4_range", {}),
        "efficiency_range": record.get("efficiency_range", {}),
        "relative_a4_samples": _row_samples(rows, "relative_a4"),
        "efficiency_samples": _row_samples(rows, "efficiency"),
        "formula_preview": _formula_preview(q_preview, missing_formula_fields),
        "evidence": record.get("evidence", []),
        "missing_fields": sorted(set(record.get("missing_fields", []) + missing_formula_fields)),
        "warnings": [
            "This is a candidate comparison only; it does not accept or apply a PSD calibration.",
            "ICE-era BT-7 source files may not expose AnaTrans, so distance selection remains a human-reviewed decision.",
            "Q values shown here, if present, are computed from source A4 after applying an unaccepted candidate PSD channel-to-A4 mapping.",
        ],
    }


def _formula_preview(q_preview: dict[str, Any] | None, missing_fields: list[str]) -> dict[str, Any]:
    if q_preview is None:
        return {"status": "missing_input", "missing_fields": missing_fields}
    two_theta = q_preview["two_theta_axis"]
    q_axis = q_preview["q_axis"]
    return {
        "status": "computed_from_candidate_mapping_not_accepted",
        "two_theta_range": {"min": min(two_theta), "max": max(two_theta)},
        "q_range": {"min": min(q_axis), "max": max(q_axis)},
        "two_theta_samples": _value_samples(two_theta),
        "q_samples": _value_samples(q_axis),
        "warnings": q_preview["warnings"],
    }


def _row_samples(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if key in row]
    return _value_samples(values)


def _value_samples(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    middle = len(values) // 2
    return {"first": values[0], "middle": values[middle], "last": values[-1]}


def _comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Calibration Candidate Comparison",
        "",
        f"Source file: `{payload['source_file'].get('path')}`",
        f"Calibration directory: `{payload['calibration_dir']}`",
        "",
        "## Safe Conclusion",
        "",
        payload["safe_conclusion"],
        "",
        "## Candidates",
        "",
        "| Candidate | Score | AnaTrans | Rows | Relative A4 range | Efficiency range | Formula status |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in payload.get("comparisons", []):
        rel = item.get("relative_a4_range") or {}
        eff = item.get("efficiency_range") or {}
        formula = item.get("formula_preview") or {}
        lines.append(
            f"| `{item.get('filename')}` | {item.get('score')} | {item.get('analyzer_detector_translation')} | {item.get('row_count')} | "
            f"{rel.get('min')} to {rel.get('max')} | {eff.get('min')} to {eff.get('max')} | `{formula.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The DAT rows provide detector efficiency and relative A4 offsets for PSD channels.",
            "- The current comparison does not choose between AnaTrans values.",
            "- For elastic diffraction, Q is computed directly from A4/sample two-theta; the PSD channel-to-A4 mapping must be reviewed first.",
        ]
    )
    return "\n".join(lines)


def _write_comparison_csv(comparisons: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "score",
                "analyzer_detector_translation",
                "row_count",
                "relative_a4_min",
                "relative_a4_max",
                "efficiency_min",
                "efficiency_max",
                "formula_status",
                "safe_use",
            ]
        )
        for item in comparisons:
            rel = item.get("relative_a4_range") or {}
            eff = item.get("efficiency_range") or {}
            formula = item.get("formula_preview") or {}
            writer.writerow(
                [
                    item.get("filename"),
                    item.get("score"),
                    item.get("analyzer_detector_translation"),
                    item.get("row_count"),
                    rel.get("min"),
                    rel.get("max"),
                    eff.get("min"),
                    eff.get("max"),
                    formula.get("status"),
                    item.get("safe_use"),
                ]
            )


def _status_counts(records: list[PSDCalibrationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return dict(sorted(counts.items()))


def _parse_psd_dat(path: Path, *, read_contents: bool = True) -> dict[str, Any]:
    text = ""
    if read_contents:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    lines = text.splitlines()
    source = None
    created = None
    masked_channels: list[int] = []
    rows = []
    for line in lines:
        if line.startswith("#BT7 PSD calibration based on"):
            source = line.split("based on", 1)[1].strip()
        elif line.lower().startswith("#channels to be masked:"):
            masked_channels = _parse_masked_channels(line.split(":", 1)[1])
        elif line.startswith("#Created on:"):
            created = _parse_created(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append({"efficiency": float(parts[0]), "relative_a4": float(parts[1]), "ef": float(parts[2])})
                except ValueError:
                    pass
    anatrans = _anatrans(path.name)
    filename_date = _filename_year_month(path.name)
    return {
        "calibration_file": str(path),
        "filename": path.name,
        "source_reference": source,
        "created_at": created.isoformat() if created else None,
        "created_date": created.date().isoformat() if created else None,
        "filename_year_month": filename_date,
        "analyzer_detector_translation": anatrans,
        "masked_channels": masked_channels,
        "masked_channel_count": len(masked_channels),
        "detector_mode_hint": "PSDDiff" if "diff" in path.name.lower() or "lufemno" in path.name.lower() else None,
        "material_hint": _material_hint(path.name),
        "row_count": len(rows),
        "channels": [{"channel": index, **row} for index, row in enumerate(rows)],
        "relative_a4_range": {"min": min(row["relative_a4"] for row in rows), "max": max(row["relative_a4"] for row in rows)} if rows else {},
        "efficiency_range": {"min": min(row["efficiency"] for row in rows), "max": max(row["efficiency"] for row in rows)} if rows else {},
        "status": "candidate_unreviewed",
        "metadata_mode": "hydrated" if text else "path_only",
        "q_temperature_unlock": False,
    }


def _parse_masked_channels(value: str) -> list[int]:
    channels: set[int] = set()
    for token in re.split(r"[,;\s]+", value.strip()):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            try:
                start = int(left)
                stop = int(right)
            except ValueError:
                continue
            channels.update(range(min(start, stop), max(start, stop) + 1))
        else:
            try:
                channels.add(int(token))
            except ValueError:
                continue
    return sorted(channel for channel in channels if channel > 0)


def _parse_bt7_source(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).expanduser()
    text = source_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header: dict[str, Any] = {"path": str(source_path), "filename": source_path.name}
    columns: list[str] = []
    data_rows: list[list[str]] = []
    fixed_fields: list[str] = []
    for line in lines:
        if line.startswith("#Filename"):
            header["native_filename"] = line.removeprefix("#Filename").strip()
        elif line.startswith("#Date"):
            raw = line.removeprefix("#Date").strip()
            header["date_raw"] = raw
            header["date"] = raw[:10]
        elif line.startswith("#AnalyzerDetectorMode"):
            header["detector_mode"] = line.removeprefix("#AnalyzerDetectorMode").strip()
        elif line.startswith("#ScanDescr"):
            header["scan_description"] = line.removeprefix("#ScanDescr").strip()
        elif line.startswith("#Fixed"):
            fixed_fields = line.split()[1:]
            header["fixed_fields"] = fixed_fields
        elif line.startswith("#Columns"):
            columns = line.split()[1:]
        elif columns and line and not line.startswith("#"):
            data_rows.append(line.split())
    header["columns"] = columns
    header["row_count"] = len(data_rows)
    psd_columns = [column for column in columns if column.startswith("PSDC")]
    if psd_columns:
        header["psd_channel_count"] = len(psd_columns)
        header["intensity_source"] = "PSDC columns"
    if data_rows:
        first_data = data_rows[0]
        values = {name: first_data[index] for index, name in enumerate(columns) if index < len(first_data)}
        for key in ["A4", "Ef", "E", "Temp", "PSDet", "AnalyzerRotation"]:
            if key in values:
                header[key] = _float_or_raw(values[key])
    if data_rows and "A4" in columns:
        header["two_theta_axis_source"] = "A4"
        header["two_theta_summary"] = _numeric_column_summary(data_rows, columns.index("A4"))
    header["q_axis"] = None
    header["q_axis_source"] = None
    if data_rows and "Temp" in columns:
        header["temperature_axis_source"] = "Temp"
        header["temperature_summary"] = _numeric_column_summary(data_rows, columns.index("Temp"))
    header["analyzer_detector_translation"] = None
    header["analyzer_detector_translation_source"] = "not_exposed_in_bt7_reader_or_scan_columns"
    header["fixed_analyzer_detector_angle_present"] = "ANADET" in fixed_fields
    header["reader_check"] = (
        "The registered read_bt7_file adapter exposes PSDC channel intensities and A4/two-theta metadata, "
        "intentionally leaves q_axis unset, and does not currently expose the AnaTrans analyzer-to-detector translation distance."
    )
    return header


def _score_calibration(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    source_name = str(source.get("native_filename") or source.get("filename") or "")
    if source_name and source_name in str(record.get("source_reference") or ""):
        score += 0.45
        evidence.append("calibration source reference names this BT-7 file")
    day_delta = _day_delta(record.get("created_date"), source.get("date"))
    if day_delta is not None:
        if day_delta == 0:
            score += 0.25
            evidence.append("calibration creation date matches data date")
        elif day_delta <= 7:
            score += 0.2
            evidence.append(f"calibration creation date is within {day_delta} day(s) of data")
        elif day_delta <= 45:
            score += 0.1
            evidence.append(f"calibration creation date is within {day_delta} day(s) of data")
    signed_day_delta = _signed_day_delta(record.get("created_date"), source.get("date"))
    if signed_day_delta is not None and 0 <= signed_day_delta <= 7:
        score += 0.08
        evidence.append(f"calibration was created {signed_day_delta} day(s) after the measurement")
    month_delta = _month_delta(record.get("filename_year_month"), source.get("date"))
    if month_delta == 0:
        score += 0.16
        evidence.append("calibration filename month/year matches measurement month")
    elif month_delta == 1:
        score += 0.12
        evidence.append("calibration filename month is one month after measurement")
    elif month_delta == -1:
        score += 0.06
        evidence.append("calibration filename month is one month before measurement")
    source_mode = str(source.get("detector_mode") or "")
    if record.get("detector_mode_hint") and record["detector_mode_hint"].lower() in source_mode.lower():
        score += 0.1
        evidence.append("detector mode hint matches source detector mode")
    source_trans = source.get("analyzer_detector_translation")
    if source_trans is None:
        evidence.append("source file does not expose AnaTrans/analyzer-to-detector translation distance; human review required")
    elif source_trans == record.get("analyzer_detector_translation"):
        score += 0.2
        evidence.append("analyzer-detector translation matches")
    material_score = _material_score(record.get("material_hint"), source_name)
    if material_score:
        score += material_score
        evidence.append("calibration filename material hint overlaps source filename")
    prior_score = _historical_anatrans_prior_score(record.get("analyzer_detector_translation"), source.get("date"))
    if prior_score:
        score += prior_score
        evidence.append("historical ICE-era prior favors AnaTrans200 as the common analyzer-to-detector distance")
    return {
        **record,
        "score": round(min(score, 1.0), 4),
        "evidence": evidence,
        "missing_fields": ["analyzer_detector_translation"] if source_trans is None else [],
        "review_status": "candidate_unreviewed",
        "safe_use": False,
    }


def _match_safe_conclusion(matches: list[dict[str, Any]], missing: list[str]) -> str:
    if not matches:
        return "No PSD calibration candidate matched the requested source file."
    best = matches[0]
    text = f"Best candidate is {Path(best['calibration_file']).name} with score {best['score']}. "
    if missing:
        return text + "Do not use it for Q conversion yet because AnaTrans/analyzer-to-detector translation distance is missing from the BT-7 source record and must be confirmed."
    return text + "Candidate still requires instrument-scientist review before any Q conversion is enabled."


def _parse_created(value: str) -> datetime | None:
    for fmt in ["%a, %d %b %Y %H:%M:%S", "%d %b %Y %H:%M:%S"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def _anatrans(name: str) -> float | None:
    match = re.search(r"AnaTrans(?P<value>\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
    return float(match.group("value")) if match else None


def _filename_year_month(name: str) -> str | None:
    month_map = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "sep": "09",
        "sept": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    match = re.search(r"(?P<month>jan|feb|march|mar|apr|may|june|jun|july|jul|aug|sept|sep|oct|nov|dec)[a-z]*[_-]?(?P<year>20\d{2}|19\d{2})", name, re.IGNORECASE)
    if not match:
        return None
    month = month_map[match.group("month").lower()]
    return f"{match.group('year')}-{month}"


def _material_hint(name: str) -> str | None:
    stem = Path(name).stem.lower()
    stem = re.sub(r"^(psd|psdeff|psdfit|psdcali|psdcalibration)[_-]?", "", stem)
    stem = re.sub(r"ana(trans)?\d+.*$", "", stem)
    stem = re.sub(r"(jan|feb|march|mar|apr|may|june|jun|july|jul|aug|sept|sep|oct|nov|dec)[a-z]*20\d{2}", "", stem)
    tokens = [token for token in re.split(r"[_\W]+", stem) if token and token not in {"singlecrystal", "test", "powder", "xtal", "coll40"}]
    return " ".join(tokens) or None


def _material_score(material_hint: Any, source_name: str) -> float:
    if not material_hint:
        return 0.0
    hint_tokens = set(re.findall(r"[a-z]+\d*", str(material_hint).lower()))
    source_tokens = set(re.findall(r"[a-z]+\d*", source_name.lower()))
    if not hint_tokens or not source_tokens:
        return 0.0
    overlap = hint_tokens & source_tokens
    if not overlap and not _same_lu_fe_mn_o_family(str(material_hint), source_name):
        return 0.0
    return 0.08 if len(overlap) == 1 else 0.12


def _same_lu_fe_mn_o_family(left: str, right: str) -> bool:
    left_flat = re.sub(r"[^a-z0-9]", "", left.lower())
    right_flat = re.sub(r"[^a-z0-9]", "", right.lower())
    required = ("lu", "fe", "mn", "o")
    return all(token in left_flat for token in required) and all(token in right_flat for token in required)


def _historical_anatrans_prior_score(anatrans: Any, source_date: str | None) -> float:
    """Weak prior: AnaTrans200 was historically common in ICE-era BT-7 PSD data."""
    if anatrans is None or not source_date:
        return 0.0
    try:
        year = date.fromisoformat(source_date).year
        distance = float(anatrans)
    except (TypeError, ValueError):
        return 0.0
    if 2012 <= year <= 2016 and distance == 200.0:
        return 0.08
    return 0.0


def _day_delta(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        return abs((date.fromisoformat(left) - date.fromisoformat(right)).days)
    except ValueError:
        return None


def _signed_day_delta(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        return (date.fromisoformat(left) - date.fromisoformat(right)).days
    except ValueError:
        return None


def _month_delta(left_year_month: str | None, right_date: str | None) -> int | None:
    if not left_year_month or not right_date:
        return None
    try:
        left_year, left_month = [int(part) for part in left_year_month.split("-", 1)]
        right = date.fromisoformat(right_date)
    except ValueError:
        return None
    return (left_year - right.year) * 12 + (left_month - right.month)


def _float_or_raw(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _numeric_column_summary(rows: list[list[str]], index: int) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        if index >= len(row):
            continue
        try:
            values.append(float(row[index]))
        except ValueError:
            continue
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "constant": max(values) == min(values),
        "first": values[0],
        "last": values[-1],
    }
