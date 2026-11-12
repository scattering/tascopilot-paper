from __future__ import annotations

from pathlib import Path
from typing import Any

from tas_copilot_contracts import Scan

from .sqlite_data_catalog import SQLiteDataCatalog


COMPLETENESS_FIELDS = [
    "instrument",
    "scan_type",
    "q_point",
    "q_path",
    "energy_transfer",
    "fixed_ei",
    "fixed_ef",
    "monochromator",
    "analyzer",
    "filter",
    "collimation",
    "ub_matrix",
    "lattice_parameters",
    "sample_environment",
    "temperature",
    "field",
    "motor_positions",
    "monitor_counts",
    "count_time",
]


def metadata_completeness_report(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {
            "status": "missing_input",
            "db_path": str(path),
            "errors": [{"code": "missing_database", "message": f"database does not exist: {path}"}],
            "scans": [],
            "summary": {"scan_count": 0},
        }
    catalog = SQLiteDataCatalog(path)
    scan_reports = [scan_completeness(scan) for scan in catalog.list_scans()]
    return {
        "status": "ok",
        "db_path": str(path),
        "summary": {
            "scan_count": len(scan_reports),
            "eligible_for_geometry_check": sum(1 for item in scan_reports if item["eligible_for_geometry_check"]),
            "eligible_for_lattice_angle_check": sum(1 for item in scan_reports if item["eligible_for_lattice_angle_check"]),
            "eligible_for_compatible_ub_check": sum(1 for item in scan_reports if item["eligible_for_compatible_ub_check"]),
            "eligible_for_eulerian_ub_check": sum(1 for item in scan_reports if item["eligible_for_eulerian_ub_check"]),
            "eligible_for_resolution_check": sum(1 for item in scan_reports if item["eligible_for_resolution_check"]),
            "eligible_for_spurion_check": sum(1 for item in scan_reports if item["eligible_for_spurion_check"]),
            "average_completeness_score": _average([item["completeness_score"] for item in scan_reports]),
        },
        "scans": scan_reports,
    }


def eligible_scans_report(db_path: str | Path, purpose: str = "geometry", limit: int | None = None) -> dict[str, Any]:
    completeness = metadata_completeness_report(db_path)
    if completeness["status"] != "ok":
        return {**completeness, "purpose": purpose, "candidate_scan_ids": [], "best_candidate": None, "rejected": []}
    eligibility_key = _eligibility_key(purpose)
    candidates = [item for item in completeness["scans"] if item.get(eligibility_key)]
    rejected = [
        {
            "scan_id": item["scan_id"],
            "missing_fields": item["missing_fields"],
            "reason": item["reason"] or f"not eligible for {purpose}",
        }
        for item in completeness["scans"]
        if not item.get(eligibility_key)
    ]
    candidates = sorted(candidates, key=lambda item: item["completeness_score"], reverse=True)
    if limit is not None:
        candidates = candidates[: int(limit)]
    return {
        "status": "ok",
        "db_path": completeness["db_path"],
        "purpose": purpose,
        "candidate_scan_ids": [item["scan_id"] for item in candidates],
        "best_candidate": candidates[0]["scan_id"] if candidates else None,
        "candidates": candidates,
        "rejected": rejected,
        "inspected_scan_count": len(completeness["scans"]),
    }


def scan_completeness(scan: Scan) -> dict[str, Any]:
    presence = _field_presence(scan)
    present = [field for field, value in presence.items() if value]
    missing = [field for field, value in presence.items() if not value]
    resolution_missing = [field for field in ["q_point", "energy_transfer", "instrument_config"] if not _resolution_field_present(scan, field)]
    lattice_angle_missing = resolution_missing + ([] if _lattice_present(scan) else ["lattice_parameters"])
    if _lattice_present(scan) and not _scattering_plane_present(scan):
        lattice_angle_missing.append("scattering_plane")
    compatible_ub_missing = resolution_missing + ([] if _compatible_ub_present(scan) else ["compatible_ub_matrix"])
    eulerian_ub_missing = resolution_missing + ([] if _eulerian_orientation_present(scan) else ["eulerian_orientation_vectors"])
    geometry_missing = lattice_angle_missing
    score = len(present) / len(COMPLETENESS_FIELDS) if COMPLETENESS_FIELDS else 0.0
    return {
        "scan_id": scan.scan_id,
        "experiment_id": scan.experiment_id,
        "instrument": scan.instrument,
        "scan_type": scan.scan_type,
        "completeness_score": round(score, 3),
        "present_fields": present,
        "missing_fields": missing,
        "eligible_for_geometry_check": not lattice_angle_missing,
        "eligible_for_lattice_angle_check": not lattice_angle_missing,
        "eligible_for_compatible_ub_check": not compatible_ub_missing,
        "eligible_for_eulerian_ub_check": not eulerian_ub_missing,
        "eligible_for_resolution_check": not resolution_missing,
        "eligible_for_spurion_check": not resolution_missing,
        "reason": _reason(scan, geometry_missing),
        "geometry_missing_fields": geometry_missing,
        "lattice_angle_missing_fields": lattice_angle_missing,
        "compatible_ub_missing_fields": compatible_ub_missing,
        "eulerian_ub_missing_fields": eulerian_ub_missing,
        "resolution_missing_fields": resolution_missing,
        "available_metadata_fields": sorted(scan.metadata.keys()),
        "scan_role": _scan_role(scan),
        "metadata_mode": _metadata_mode(scan),
        "path_only": _metadata_mode(scan) == "path_only",
        "ub_convention": _ub_convention(scan),
    }


def _field_presence(scan: Scan) -> dict[str, bool]:
    config = scan.instrument_config
    metadata = scan.metadata or {}
    return {
        "instrument": bool(scan.instrument and scan.instrument.lower() != "unknown"),
        "scan_type": bool(scan.scan_type),
        "q_point": scan.q_center is not None,
        "q_path": bool(scan.q_path and scan.q_path.points),
        "energy_transfer": scan.energy_transfer is not None,
        "fixed_ei": bool(config and config.fixed_energy and config.fixed_energy_mode == "fixed_incident_energy"),
        "fixed_ef": bool(config and config.fixed_energy and config.fixed_energy_mode == "fixed_final_energy"),
        "monochromator": bool(config and config.monochromator),
        "analyzer": bool(config and config.analyzer),
        "filter": bool(config and config.filters),
        "collimation": bool(config and config.collimation),
        "ub_matrix": bool(metadata.get("ub_matrix") or metadata.get("ub")),
        "lattice_parameters": _lattice_present(scan),
        "sample_environment": bool((config and config.sample_environment) or metadata.get("sample_environment")),
        "temperature": bool(scan.temperature_range or metadata.get("temperature") is not None),
        "field": bool(scan.magnetic_field_range or metadata.get("magnetic_field") is not None),
        "motor_positions": bool(scan.motor_trajectory and scan.motor_trajectory.positions),
        "monitor_counts": bool(metadata.get("monitor_counts") is not None or metadata.get("monitor") is not None),
        "count_time": bool(metadata.get("count_time") is not None or metadata.get("time") is not None),
    }


def _resolution_field_present(scan: Scan, field: str) -> bool:
    if field == "q_point":
        return scan.q_center is not None
    if field == "energy_transfer":
        return scan.energy_transfer is not None
    if field == "instrument_config":
        return scan.instrument_config is not None
    return False


def _lattice_present(scan: Scan) -> bool:
    metadata = scan.metadata or {}
    lattice = metadata.get("lattice")
    if not isinstance(lattice, dict):
        geometry = metadata.get("geometry")
        lattice = geometry.get("lattice") if isinstance(geometry, dict) else None
    if not isinstance(lattice, dict):
        return False
    return all(lattice.get(key) is not None for key in ["a", "b", "c", "alpha", "beta", "gamma"])


def _orientation_present(scan: Scan) -> bool:
    lattice = _lattice_payload(scan)
    if not isinstance(lattice, dict):
        return False
    if lattice.get("orient1") is not None and lattice.get("orient2") is not None:
        return True
    metadata = scan.metadata or {}
    return bool(
        metadata.get("scattering_plane")
        or metadata.get("orientation_reflections")
        or metadata.get("ub_matrix")
        or metadata.get("ub")
    )


def _lattice_payload(scan: Scan) -> dict | None:
    metadata = scan.metadata or {}
    lattice = metadata.get("lattice")
    if isinstance(lattice, dict):
        return lattice
    geometry = metadata.get("geometry")
    if isinstance(geometry, dict) and isinstance(geometry.get("lattice"), dict):
        return geometry["lattice"]
    return None


def _eligibility_key(purpose: str) -> str:
    normalized = purpose.replace("-", "_").lower()
    if normalized in {"lattice", "lattice_angle", "lattice_angle_check", "geometry"}:
        return "eligible_for_lattice_angle_check"
    if normalized in {"ub", "compatible_ub", "ub_geometry", "compatible_ub_check"}:
        return "eligible_for_compatible_ub_check"
    if normalized in {"eulerian", "eulerian_ub", "eulerian_ub_check"}:
        return "eligible_for_eulerian_ub_check"
    if normalized in {"resolution", "resolution_check"}:
        return "eligible_for_resolution_check"
    if normalized in {"spurion", "spurion_check"}:
        return "eligible_for_spurion_check"
    return "eligible_for_geometry_check"


def _average(values: list[float]) -> float:
    return 0.0 if not values else round(sum(values) / len(values), 3)


def _scan_role(scan: Scan) -> str | None:
    metadata = scan.metadata or {}
    raw_metadata = metadata.get("raw_metadata")
    if isinstance(raw_metadata, dict) and raw_metadata.get("scan_role"):
        return str(raw_metadata["scan_role"])
    return metadata.get("scan_role") if isinstance(metadata.get("scan_role"), str) else None


def _metadata_mode(scan: Scan) -> str:
    metadata = scan.metadata or {}
    raw_metadata = metadata.get("raw_metadata")
    if isinstance(raw_metadata, dict) and raw_metadata.get("metadata_mode"):
        return str(raw_metadata["metadata_mode"])
    return "read_prefix"


def _reason(scan: Scan, geometry_missing: list[str]) -> str | None:
    if not geometry_missing:
        return None
    if _metadata_mode(scan) == "path_only":
        return "path-only metadata; file content was not read"
    role = _scan_role(scan)
    if role == "alignment" and geometry_missing == ["scattering_plane"]:
        return "alignment scan lacks scattering-plane metadata"
    if "scattering_plane" in geometry_missing:
        return "missing scattering-plane metadata"
    return "missing " + ", ".join(geometry_missing)


def _scattering_plane_present(scan: Scan) -> bool:
    metadata = scan.metadata or {}
    lattice = _lattice_payload(scan)
    return bool(
        (isinstance(lattice, dict) and lattice.get("orient1") is not None and lattice.get("orient2") is not None)
        or metadata.get("scattering_plane")
        or metadata.get("orientation_reflections")
    )


def _compatible_ub_present(scan: Scan) -> bool:
    metadata = scan.metadata or {}
    return bool((metadata.get("ub_matrix") or metadata.get("ub")) and _ub_convention(scan) not in {None, "unknown", "eulerian_cradle"})


def _eulerian_orientation_present(scan: Scan) -> bool:
    lattice = _lattice_payload(scan)
    metadata = scan.metadata or {}
    return bool(
        (isinstance(lattice, dict) and lattice.get("orient1") is not None and lattice.get("orient2") is not None)
        or _ub_convention(scan) == "eulerian_cradle"
    )


def _ub_convention(scan: Scan) -> str | None:
    metadata = scan.metadata or {}
    value = metadata.get("ub_convention")
    return str(value) if value else None
