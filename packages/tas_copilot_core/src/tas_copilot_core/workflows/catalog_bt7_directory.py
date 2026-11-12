from __future__ import annotations

import hashlib
import time
from pathlib import Path

from tas_copilot_contracts import DataFile, Experiment, ProvenanceRecord, QPoint, Scan

from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig

from .group_bt7_scans import group_bt7_scans


def catalog_bt7_directory(arguments: dict) -> dict:
    """Catalog a BT-7 data directory into the data-file SQLite catalog."""

    started = time.perf_counter()
    db_path = arguments.get("db_path") or str(CoreConfig().data_catalog_db_path)
    result = group_bt7_scans(arguments)
    if result.status.value != "success":
        return {"status": result.status.value, "tool_result": result.model_dump(mode="json"), "db_path": db_path}

    output = result.output or {}
    records = output.get("records", [])
    source_record_count = len(records)
    max_records = arguments.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]
    experiments = _experiments_from_records(records)
    scans = _scans_from_records(records)
    datafiles = _datafiles_from_records(records)

    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments(experiments)
    catalog.upsert_scans(scans)
    catalog.upsert_datafiles(datafiles)
    elapsed = time.perf_counter() - started
    return {
        "status": "success",
        "db_path": db_path,
        "imported": {
            "experiments": len(experiments),
            "scans": len(scans),
            "data_files": len(datafiles),
        },
        "source_record_count": source_record_count,
        "cataloged_record_count": len(records),
        "source_group_count": output.get("group_count", 0),
        "catalog_summary": catalog.summary(),
        "catalog_metrics": catalog.metrics(),
        "elapsed_seconds": elapsed,
        "records_per_second": 0.0 if elapsed == 0 else len(records) / elapsed,
    }


def _experiments_from_records(records: list[dict]) -> list[Experiment]:
    by_id: dict[str, Experiment] = {}
    for record in records:
        experiment_id = _experiment_id(record)
        if experiment_id not in by_id:
            by_id[experiment_id] = Experiment(
                experiment_id=experiment_id,
                instrument=record.get("instrument") or "BT-7",
                title=record.get("expt_name") or None,
                metadata={"bt7grouping_expt_id": record.get("expt_id"), "source": "bt7grouping"},
                provenance=[_record_provenance(record, "bt7grouping.catalog_experiment")],
            )
    return list(by_id.values())


def _scans_from_records(records: list[dict]) -> list[Scan]:
    scans = []
    for record in records:
        q_center = None
        if any(record.get(axis) is not None for axis in ["h", "k", "l"]):
            q_center = QPoint(h=float(record.get("h") or 0.0), k=float(record.get("k") or 0.0), l=float(record.get("l") or 0.0))
        scans.append(
            Scan(
                scan_id=_scan_id(record),
                experiment_id=_experiment_id(record),
                instrument=record.get("instrument") or "BT-7",
                scan_type=record.get("scan_type") or None,
                q_center=q_center,
                metadata={
                    "filename": record.get("filename"),
                    "path": record.get("path"),
                    "scan_variable": record.get("scan_variable"),
                    "scan_description": record.get("scan_description"),
                    "format_type": record.get("format_type"),
                    "fixed_e": record.get("fixed_e"),
                    "detector_mode": record.get("detector_mode"),
                    "timestamp": record.get("timestamp"),
                    "datetime": record.get("datetime"),
                    "temperature": record.get("temp"),
                    "temperature_setpoint": record.get("temp_setpoint"),
                },
                provenance=[_record_provenance(record, "bt7grouping.catalog_scan")],
            )
        )
    return scans


def _datafiles_from_records(records: list[dict]) -> list[DataFile]:
    return [
        DataFile(
            datafile_id=_datafile_id(record.get("path") or ""),
            path=record.get("path") or "",
            scan_id=_scan_id(record),
            experiment_id=_experiment_id(record),
            format=record.get("format_type") or None,
            metadata={"filename": record.get("filename"), "source": "bt7grouping"},
            provenance=[_record_provenance(record, "bt7grouping.catalog_datafile")],
        )
        for record in records
    ]


def _experiment_id(record: dict) -> str:
    if record.get("expt_id"):
        return f"bt7-experiment-{record['expt_id']}"
    parent = Path(record.get("path") or "").parent.name or "unknown"
    return f"bt7-directory-{_slug(parent)}"


def _scan_id(record: dict) -> str:
    path = record.get("path") or record.get("filename") or "unknown"
    return f"bt7-scan-{_hash(path)[:16]}"


def _datafile_id(path: str) -> str:
    return f"bt7-datafile-{_hash(path)[:16]}"


def _record_provenance(record: dict, method: str) -> ProvenanceRecord:
    return ProvenanceRecord(source_type="data_file", source_id=record.get("path"), method=method)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"
