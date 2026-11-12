from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def data_catalog_quality_report(db_path: str | Path) -> dict[str, Any]:
    """Return read-only quality diagnostics for a TAS data-file SQLite catalog."""

    path = Path(db_path)
    if not path.exists():
        return {
            "status": "missing_input",
            "db_path": str(path),
            "errors": [{"code": "missing_database", "message": f"database does not exist: {path}"}],
        }
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        summary = _summary(connection)
        diagnostics = {
            "file_coverage": _file_coverage(connection),
            "null_native_ids": _null_native_ids(connection),
            "duplicate_ids": _duplicate_ids(connection),
            "orphan_links": _orphan_links(connection),
            "missing_provenance": _missing_provenance(connection),
            "metadata_modes": _metadata_modes(connection),
            "reader_strategy": _reader_strategy(connection),
            "scientific_metadata_coverage": _scientific_metadata_coverage(connection),
            "geometry_readiness": _geometry_readiness(connection),
            "quality_categories": {},
            "bt7_enrichment": _bt7_enrichment(connection),
            "reader_metadata_enrichment": _reader_metadata_enrichment(connection),
            "per_instrument": _per_instrument(connection),
        }
        diagnostics["quality_categories"] = _quality_categories(diagnostics)
    finally:
        connection.close()
    warnings = _warnings(diagnostics)
    return {
        "status": "warning" if warnings else "ok",
        "db_path": str(path),
        "summary": summary,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def markdown_quality_report(report: dict[str, Any]) -> str:
    lines = [
        "# TAS Data Catalog Quality Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Catalog: `{report.get('db_path')}`",
        f"- Summary: `{report.get('summary', {})}`",
        f"- File coverage: `{report.get('diagnostics', {}).get('file_coverage', {})}`",
        f"- Metadata modes: `{report.get('diagnostics', {}).get('metadata_modes', {})}`",
        f"- Geometry readiness: `{report.get('diagnostics', {}).get('geometry_readiness', {})}`",
        f"- Quality categories: `{report.get('diagnostics', {}).get('quality_categories', {})}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    return "\n".join(lines) + "\n"


def _summary(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ["experiments", "scans", "data_files", "derived_data_files", "paper_experiment_links"]
    }


def _file_coverage(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT checksum, payload_json FROM data_files").fetchall()
    coverage = {
        "total_data_files": len(rows),
        "path_only_files": 0,
        "hydrated_files": 0,
        "partial_files": 0,
        "files_with_stat_metadata": 0,
        "files_with_checksums": 0,
        "files_with_readable_metadata": 0,
        "files_with_reader_warnings": 0,
        "files_with_reader_errors": 0,
    }
    for row in rows:
        payload = json.loads(row["payload_json"])
        metadata = payload.get("metadata", {})
        raw = metadata.get("raw_metadata") or {}
        mode = _metadata_mode(metadata)
        if mode == "path_only":
            coverage["path_only_files"] += 1
        elif mode == "partial":
            coverage["partial_files"] += 1
        else:
            coverage["hydrated_files"] += 1
        if row["checksum"]:
            coverage["files_with_checksums"] += 1
        if any(raw.get(key) is not None for key in ["stat", "file_size", "size", "mtime", "stat_status"]):
            coverage["files_with_stat_metadata"] += 1
        if mode not in {"path_only", "unknown"} and bool(metadata):
            coverage["files_with_readable_metadata"] += 1
        if raw.get("warnings"):
            coverage["files_with_reader_warnings"] += 1
        if raw.get("errors"):
            coverage["files_with_reader_errors"] += 1
    return coverage


def _null_native_ids(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "data_files_native_experiment_id": _count(connection, "SELECT COUNT(*) FROM data_files WHERE native_experiment_id IS NULL"),
        "data_files_native_scan_id": _count(connection, "SELECT COUNT(*) FROM data_files WHERE native_scan_id IS NULL"),
        "scans_native_scan_id": _count(connection, "SELECT COUNT(*) FROM scans WHERE native_scan_id IS NULL"),
        "experiments_native_experiment_id": _count(connection, "SELECT COUNT(*) FROM experiments WHERE native_experiment_id IS NULL"),
    }


def _duplicate_ids(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "experiment_id": _duplicate_count(connection, "experiments", "experiment_id"),
        "scan_id": _duplicate_count(connection, "scans", "scan_id"),
        "datafile_id": _duplicate_count(connection, "data_files", "datafile_id"),
        "data_file_path": _duplicate_count(connection, "data_files", "path"),
        "native_scan_key": _count(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT facility || ':' || instrument || ':' || native_experiment_id || ':' || native_scan_id AS key_value,
                       COUNT(*) AS count
                FROM data_files
                WHERE facility IS NOT NULL
                  AND instrument IS NOT NULL
                  AND native_experiment_id IS NOT NULL
                  AND native_scan_id IS NOT NULL
                GROUP BY facility, instrument, native_experiment_id, native_scan_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
    }


def _orphan_links(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "scans_without_experiment": _count(
            connection,
            """
            SELECT COUNT(*) FROM scans s
            LEFT JOIN experiments e ON e.experiment_id = s.experiment_id
            WHERE s.experiment_id IS NOT NULL AND e.experiment_id IS NULL
            """,
        ),
        "data_files_without_scan": _count(
            connection,
            """
            SELECT COUNT(*) FROM data_files d
            LEFT JOIN scans s ON s.scan_id = d.scan_id
            WHERE d.scan_id IS NOT NULL AND s.scan_id IS NULL
            """,
        ),
        "data_files_without_experiment": _count(
            connection,
            """
            SELECT COUNT(*) FROM data_files d
            LEFT JOIN experiments e ON e.experiment_id = d.experiment_id
            WHERE d.experiment_id IS NOT NULL AND e.experiment_id IS NULL
            """,
        ),
    }


def _missing_provenance(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: _count(connection, f"SELECT COUNT(*) FROM {table} WHERE payload_json NOT LIKE '%\"provenance\":%'")
        for table in ["experiments", "scans", "data_files", "derived_data_files", "paper_experiment_links"]
    }


def _bt7_enrichment(connection: sqlite3.Connection) -> dict[str, int | float]:
    total = _count(connection, "SELECT COUNT(*) FROM scans WHERE instrument = 'BT-7'")
    with_enrichment = 0
    with_scan_type = 0
    with_q_center = 0
    for row in connection.execute("SELECT payload_json FROM scans WHERE instrument = 'BT-7'").fetchall():
        payload = json.loads(row["payload_json"])
        enrichment = payload.get("metadata", {}).get("bt7_enrichment") or {}
        if enrichment:
            with_enrichment += 1
        if payload.get("scan_type"):
            with_scan_type += 1
        if payload.get("q_center"):
            with_q_center += 1
    return {
        "bt7_scan_count": total,
        "with_bt7_enrichment": with_enrichment,
        "with_scan_type": with_scan_type,
        "with_q_center": with_q_center,
        "enrichment_fraction": 0.0 if total == 0 else with_enrichment / total,
    }


def _reader_metadata_enrichment(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute("SELECT facility, instrument, payload_json FROM data_files").fetchall()
    totals: dict[str, dict[str, Any]] = {}
    fields = ["sample_name", "temperature", "q_hint", "energy_hint", "scan_variable"]
    for row in rows:
        key = f"{row['facility'] or 'unknown'} {row['instrument'] or 'unknown'}"
        item = totals.setdefault(
            key,
            {
                "facility": row["facility"] or "unknown",
                "instrument": row["instrument"] or "unknown",
                "data_files": 0,
                "with_sample_name": 0,
                "with_temperature": 0,
                "with_q_hint": 0,
                "with_energy_hint": 0,
                "with_scan_variable": 0,
                "reader_strategies": {},
                "metadata_modes": {},
                "path_only_data_files": 0,
            },
        )
        item["data_files"] += 1
        metadata = json.loads(row["payload_json"]).get("metadata", {})
        for field in fields:
            if metadata.get(field) not in (None, {}, []):
                item[f"with_{field}"] += 1
        strategy = (metadata.get("raw_metadata") or {}).get("reader_strategy")
        if strategy:
            item["reader_strategies"][strategy] = item["reader_strategies"].get(strategy, 0) + 1
        mode = _metadata_mode(metadata)
        item["metadata_modes"][mode] = item["metadata_modes"].get(mode, 0) + 1
        if mode == "path_only":
            item["path_only_data_files"] += 1
    return {"per_instrument": list(totals.values())}


def _metadata_modes(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in connection.execute("SELECT payload_json FROM data_files").fetchall():
        metadata = json.loads(row["payload_json"]).get("metadata", {})
        mode = _metadata_mode(metadata)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _reader_strategy(connection: sqlite3.Connection) -> dict[str, Any]:
    strategies: dict[str, int] = {}
    fallbacks: dict[str, int] = {}
    source_adapters: dict[str, int] = {}
    source_readers: dict[str, int] = {}
    for row in connection.execute("SELECT source_adapter, payload_json FROM data_files").fetchall():
        payload = json.loads(row["payload_json"])
        metadata = payload.get("metadata", {})
        raw = metadata.get("raw_metadata") or {}
        _bump(str(raw.get("reader_strategy") or "unknown"), strategies)
        fallback = raw.get("reader_fallback") or raw.get("fallback_behavior")
        if fallback:
            _bump(str(fallback), fallbacks)
        _bump(str(metadata.get("source_adapter") or row["source_adapter"] or "unknown"), source_adapters)
        _bump(str(metadata.get("source_reader") or "unknown"), source_readers)
    return {
        "reader_strategies": strategies,
        "fallback_behaviors": fallbacks,
        "source_adapters": source_adapters,
        "source_readers": source_readers,
    }


def _scientific_metadata_coverage(connection: sqlite3.Connection) -> dict[str, int]:
    fields = {
        "instrument": 0,
        "scan_type": 0,
        "q_point": 0,
        "q_path": 0,
        "energy_transfer": 0,
        "fixed_ei": 0,
        "fixed_ef": 0,
        "monochromator": 0,
        "analyzer": 0,
        "filter": 0,
        "collimation": 0,
        "ub_matrix": 0,
        "ub_convention": 0,
        "geometry_mode": 0,
        "lattice_parameters": 0,
        "scattering_plane": 0,
        "sample_environment": 0,
        "temperature": 0,
        "field": 0,
        "monitor_counts": 0,
        "count_time": 0,
        "motor_positions": 0,
        "motor_trajectory": 0,
    }
    for row in connection.execute("SELECT payload_json FROM scans").fetchall():
        scan = json.loads(row["payload_json"])
        metadata = scan.get("metadata") or {}
        config = scan.get("instrument_config") or {}
        lattice = _lattice(metadata)
        if scan.get("instrument") and str(scan.get("instrument")).lower() != "unknown":
            fields["instrument"] += 1
        if scan.get("scan_type"):
            fields["scan_type"] += 1
        if scan.get("q_center"):
            fields["q_point"] += 1
        if scan.get("q_path"):
            fields["q_path"] += 1
        if scan.get("energy_transfer"):
            fields["energy_transfer"] += 1
        if config.get("fixed_energy") and config.get("fixed_energy_mode") == "fixed_incident_energy":
            fields["fixed_ei"] += 1
        if config.get("fixed_energy") and config.get("fixed_energy_mode") == "fixed_final_energy":
            fields["fixed_ef"] += 1
        for field, key in [("monochromator", "monochromator"), ("analyzer", "analyzer"), ("collimation", "collimation")]:
            if config.get(key):
                fields[field] += 1
        if config.get("filters"):
            fields["filter"] += 1
        if metadata.get("ub_matrix") or metadata.get("ub"):
            fields["ub_matrix"] += 1
        if metadata.get("ub_convention"):
            fields["ub_convention"] += 1
        if _geometry_mode(scan) != "unknown":
            fields["geometry_mode"] += 1
        if _lattice_ready(lattice, require_orientation=False):
            fields["lattice_parameters"] += 1
        if _scattering_plane_present(metadata, lattice):
            fields["scattering_plane"] += 1
        if config.get("sample_environment") or metadata.get("sample_environment"):
            fields["sample_environment"] += 1
        if scan.get("temperature_range") or metadata.get("temperature") is not None:
            fields["temperature"] += 1
        if scan.get("magnetic_field_range") or metadata.get("magnetic_field") is not None:
            fields["field"] += 1
        if metadata.get("monitor_counts") is not None or metadata.get("monitor") is not None:
            fields["monitor_counts"] += 1
        if metadata.get("count_time") is not None or metadata.get("time") is not None:
            fields["count_time"] += 1
        motor = scan.get("motor_trajectory") or {}
        if motor.get("positions"):
            fields["motor_positions"] += 1
            fields["motor_trajectory"] += 1
    fields["total_scans"] = _count(connection, "SELECT COUNT(*) FROM scans")
    return fields


def _geometry_readiness(connection: sqlite3.Connection) -> dict[str, Any]:
    readiness: dict[str, Any] = {
        "geometry_mode_counts": {},
        "lattice_ready_scans": 0,
        "compatible_ub_scans": 0,
        "eulerian_ub_scans": 0,
        "geometry_eligible_scans": 0,
        "resolution_eligible_scans": 0,
        "spurion_eligible_scans": 0,
        "rejected_missing_orientation": 0,
        "rejected_unsupported_ub_convention": 0,
    }
    for row in connection.execute("SELECT payload_json FROM scans").fetchall():
        scan = json.loads(row["payload_json"])
        metadata = scan.get("metadata") or {}
        lattice = _lattice(metadata)
        mode = _geometry_mode(scan)
        _bump(mode, readiness["geometry_mode_counts"])
        resolution_ready = bool(scan.get("q_center") and scan.get("energy_transfer") and scan.get("instrument_config"))
        lattice_ready = bool(resolution_ready and _lattice_ready(lattice, require_orientation=True))
        compatible_ub = bool(resolution_ready and (metadata.get("ub_matrix") or metadata.get("ub")) and _ub_convention(metadata) not in {None, "unknown", "eulerian_cradle"})
        eulerian_ub = bool(resolution_ready and _ub_convention(metadata) == "eulerian_cradle")
        if lattice_ready:
            readiness["lattice_ready_scans"] += 1
            readiness["geometry_eligible_scans"] += 1
        if resolution_ready:
            readiness["resolution_eligible_scans"] += 1
            readiness["spurion_eligible_scans"] += 1
        if compatible_ub:
            readiness["compatible_ub_scans"] += 1
            readiness["rejected_unsupported_ub_convention"] += 1
        if eulerian_ub:
            readiness["eulerian_ub_scans"] += 1
        if resolution_ready and _lattice_ready(lattice, require_orientation=False) and not _scattering_plane_present(metadata, lattice):
            readiness["rejected_missing_orientation"] += 1
    return readiness


def _quality_categories(diagnostics: dict[str, Any]) -> dict[str, bool]:
    file_coverage = diagnostics["file_coverage"]
    geometry = diagnostics["geometry_readiness"]
    scientific = diagnostics["scientific_metadata_coverage"]
    total_files = file_coverage["total_data_files"]
    total_scans = scientific["total_scans"]
    return {
        "path_only_catalog": file_coverage["path_only_files"] > 0,
        "hydrated_catalog": total_files > 0 and file_coverage["hydrated_files"] == total_files,
        "metadata_sparse": total_scans > 0 and geometry["resolution_eligible_scans"] == 0,
        "geometry_ready": geometry["geometry_eligible_scans"] > 0,
        "ub_preserved_but_unsupported": geometry["compatible_ub_scans"] > 0,
        "scientifically_incomplete": total_scans > 0 and geometry["geometry_eligible_scans"] < total_scans,
        "report_ready": total_files > 0 and (geometry["geometry_eligible_scans"] > 0 or file_coverage["path_only_files"] == 0),
    }


def _per_instrument(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT COALESCE(facility, 'unknown') AS facility,
               COALESCE(instrument, 'unknown') AS instrument,
               COUNT(*) AS data_files,
               COUNT(DISTINCT experiment_id) AS experiments,
               COUNT(DISTINCT scan_id) AS scans,
               SUM(CASE WHEN native_experiment_id IS NULL THEN 1 ELSE 0 END) AS null_native_experiment_ids,
               SUM(CASE WHEN native_scan_id IS NULL THEN 1 ELSE 0 END) AS null_native_scan_ids
        FROM data_files
        GROUP BY COALESCE(facility, 'unknown'), COALESCE(instrument, 'unknown')
        ORDER BY facility, instrument
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _warnings(diagnostics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    file_coverage = diagnostics.get("file_coverage", {})
    if file_coverage.get("files_with_reader_errors"):
        warnings.append(f"reader errors in data files: {file_coverage['files_with_reader_errors']}")
    if file_coverage.get("files_with_reader_warnings"):
        warnings.append(f"reader warnings in data files: {file_coverage['files_with_reader_warnings']}")
    for key, value in diagnostics["duplicate_ids"].items():
        if value:
            warnings.append(f"duplicate {key}: {value}")
    for key, value in diagnostics["orphan_links"].items():
        if value:
            warnings.append(f"orphan {key}: {value}")
    for key, value in diagnostics["missing_provenance"].items():
        if value:
            warnings.append(f"missing provenance in {key}: {value}")
    for item in diagnostics["per_instrument"]:
        if item["null_native_experiment_ids"]:
            warnings.append(f"{item['facility']} {item['instrument']} null native experiment ids: {item['null_native_experiment_ids']}")
        if item["null_native_scan_ids"]:
            warnings.append(f"{item['facility']} {item['instrument']} null native scan ids: {item['null_native_scan_ids']}")
    path_only_count = diagnostics.get("metadata_modes", {}).get("path_only", 0)
    if path_only_count:
        warnings.append(f"path-only metadata records: {path_only_count}")
    geometry = diagnostics.get("geometry_readiness", {})
    if geometry.get("rejected_missing_orientation"):
        warnings.append(f"scans rejected due to missing orientation/scattering plane: {geometry['rejected_missing_orientation']}")
    if geometry.get("rejected_unsupported_ub_convention"):
        warnings.append(f"compatible UB scans require reviewed convention adapter: {geometry['rejected_unsupported_ub_convention']}")
    bt7 = diagnostics["bt7_enrichment"]
    if bt7["bt7_scan_count"] and bt7["with_bt7_enrichment"] < bt7["bt7_scan_count"]:
        warnings.append(f"BT-7 scans missing enrichment: {bt7['bt7_scan_count'] - bt7['with_bt7_enrichment']}")
    return warnings


def _duplicate_count(connection: sqlite3.Connection, table: str, expression: str) -> int:
    row = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT {expression} AS key_value, COUNT(*) AS count
            FROM {table}
            WHERE {expression} IS NOT NULL
            GROUP BY {expression}
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    return int(row[0])


def _count(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _metadata_mode(metadata: dict[str, Any]) -> str:
    raw = metadata.get("raw_metadata") or {}
    return str(raw.get("metadata_mode") or metadata.get("metadata_mode") or "read_prefix")


def _lattice(metadata: dict[str, Any]) -> dict[str, Any] | None:
    lattice = metadata.get("lattice")
    if isinstance(lattice, dict):
        return lattice
    geometry = metadata.get("geometry")
    if isinstance(geometry, dict) and isinstance(geometry.get("lattice"), dict):
        return geometry["lattice"]
    return None


def _lattice_ready(lattice: dict[str, Any] | None, require_orientation: bool) -> bool:
    if not isinstance(lattice, dict):
        return False
    if not all(lattice.get(key) is not None for key in ["a", "b", "c", "alpha", "beta", "gamma"]):
        return False
    if require_orientation and not (lattice.get("orient1") is not None and lattice.get("orient2") is not None):
        return False
    return True


def _scattering_plane_present(metadata: dict[str, Any], lattice: dict[str, Any] | None) -> bool:
    return bool(
        (isinstance(lattice, dict) and lattice.get("orient1") is not None and lattice.get("orient2") is not None)
        or metadata.get("scattering_plane")
        or metadata.get("orientation_reflections")
    )


def _ub_convention(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("ub_convention")
    return str(value) if value else None


def _geometry_mode(scan: dict[str, Any]) -> str:
    metadata = scan.get("metadata") or {}
    convention = _ub_convention(metadata)
    if (metadata.get("ub_matrix") or metadata.get("ub")) and convention not in {None, "unknown", "eulerian_cradle"}:
        return "compatible_ub"
    if convention == "eulerian_cradle":
        return "eulerian_ub"
    if _lattice_ready(_lattice(metadata), require_orientation=True):
        return "lattice"
    return "unknown"


def _bump(key: str, counts: dict[str, int]) -> None:
    counts[key] = counts.get(key, 0) + 1
