from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

from tas_copilot_contracts import DataFile, EnergyTransfer, Experiment, InstrumentConfig, ProvenanceRecord, QPoint, Scan, UnitValue

from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig

from .group_bt7_scans import group_bt7_scans
from .scan_tas_directory import scan_tas_directory


def catalog_tas_directory(arguments: dict) -> dict:
    """Catalog a TAS data directory into the data-file SQLite catalog."""

    started = time.perf_counter()
    file_paths = _file_paths(arguments)
    path = arguments.get("path") or arguments.get("root") or _common_root(file_paths)
    if not path:
        return {"status": "missing_input", "errors": [{"code": "missing_path", "message": "path or file_paths is required"}]}
    path = str(Path(path).expanduser())
    db_path = arguments.get("db_path") or str(CoreConfig().data_catalog_db_path)

    scan_args = {"root": path}
    if file_paths:
        scan_args["paths"] = file_paths
    if "path_only" in arguments:
        scan_args["path_only"] = bool(arguments["path_only"])
    for key in ["max_files"]:
        if key in arguments:
            scan_args[key] = arguments[key]
    scan_result = scan_tas_directory(scan_args)
    if scan_result.status.value != "success":
        return {"status": scan_result.status.value, "tool_result": scan_result.model_dump(mode="json"), "db_path": db_path}

    records = list((scan_result.output or {}).get("records", []))
    source_record_count = len(records)
    max_records = arguments.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]

    bt7_records_by_path = _bt7_enrichment(path, records, arguments)
    experiments = _experiments_from_records(records, bt7_records_by_path)
    scans = _scans_from_records(records, bt7_records_by_path)
    datafiles = _datafiles_from_records(records, bt7_records_by_path)

    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments(experiments)
    catalog.upsert_scans(scans)
    catalog.upsert_datafiles(datafiles)

    elapsed = time.perf_counter() - started
    return {
        "status": "success",
        "db_path": db_path,
        "path": path,
        "imported": {
            "experiments": len(experiments),
            "scans": len(scans),
            "data_files": len(datafiles),
        },
        "source_record_count": source_record_count,
        "cataloged_record_count": len(records),
        "format_counts": (scan_result.output or {}).get("format_counts", {}),
        "instrument_counts": (scan_result.output or {}).get("instrument_counts", {}),
        "bt7_enrichment": {
            "attempted": _has_instrument(records, "BT-7") and not arguments.get("skip_bt7_enrichment", False),
            "record_count": len(bt7_records_by_path),
        },
        "scan_tool_audit_id": scan_result.audit_id,
        "catalog_summary": catalog.summary(),
        "catalog_metrics": catalog.metrics(),
        "elapsed_seconds": elapsed,
        "records_per_second": 0.0 if elapsed == 0 else len(records) / elapsed,
    }


def _bt7_enrichment(path: str, records: list[dict], arguments: dict) -> dict[str, dict]:
    if arguments.get("skip_bt7_enrichment", False) or not _has_instrument(records, "BT-7"):
        return {}
    group_args = {
        "root_dirs": [path],
        "recursive": arguments.get("recursive", True),
    }
    for key in ["file_extensions", "q_tolerance", "e_tolerance", "temp_tolerance", "group_by_experiment", "separate_alignment"]:
        if key in arguments:
            group_args[key] = arguments[key]
    result = group_bt7_scans(group_args)
    if result.status.value != "success" or not result.output:
        return {}
    return {str(record.get("path")): record for record in result.output.get("records", []) if record.get("path")}


def _file_paths(arguments: dict) -> list[str]:
    paths = arguments.get("file_paths") or arguments.get("files") or arguments.get("paths") or []
    return [str(Path(path).expanduser()) for path in paths]


def _common_root(paths: list[str]) -> str | None:
    if not paths:
        return None
    try:
        if len(paths) == 1:
            return str(Path(paths[0]).parent)
        return os.path.commonpath([str(Path(path).parent) for path in paths])
    except Exception:
        return str(Path(paths[0]).parent)


def _experiments_from_records(records: list[dict], bt7_records_by_path: dict[str, dict] | None = None) -> list[Experiment]:
    by_id: dict[str, Experiment] = {}
    for record in records:
        enrichment = (bt7_records_by_path or {}).get(str(record.get("path") or ""), {})
        experiment_id = _experiment_id(record, enrichment)
        if experiment_id not in by_id:
            metadata = _common_metadata(record, enrichment)
            by_id[experiment_id] = Experiment(
                experiment_id=experiment_id,
                instrument=record.get("instrument") or "unknown",
                title=enrichment.get("expt_name") or _experiment_title(record, enrichment),
                metadata=metadata,
                provenance=[_record_provenance(record, "tas_data_readers.catalog_experiment")],
            )
    return list(by_id.values())


def _scans_from_records(records: list[dict], bt7_records_by_path: dict[str, dict]) -> list[Scan]:
    scans = []
    for record in records:
        path = str(record.get("path") or "")
        enrichment = bt7_records_by_path.get(path, {})
        header_metadata = _bt7_header_metadata(record)
        q_center = None
        if enrichment and any(enrichment.get(axis) is not None for axis in ["h", "k", "l"]):
            q_center = QPoint(
                h=float(enrichment.get("h") or 0.0),
                k=float(enrichment.get("k") or 0.0),
                l=float(enrichment.get("l") or 0.0),
            )
        elif record.get("q_hint"):
            q_hint = record.get("q_hint") or {}
            if any(q_hint.get(axis) is not None for axis in ["h", "k", "l"]):
                q_center = QPoint(
                    h=float(q_hint.get("h") or 0.0),
                    k=float(q_hint.get("k") or 0.0),
                    l=float(q_hint.get("l") or 0.0),
                )
        raw_metadata = record.get("raw_metadata") or {}
        metadata = {
            **_common_metadata(record, enrichment),
            **_reader_metadata(record),
            **header_metadata,
            "path": path,
            "filename": Path(path).name,
            "bt7_enrichment": _bt7_metadata(enrichment),
        }
        lattice = _lattice_from_record(record)
        if lattice:
            metadata["lattice"] = lattice
        raw_metadata = record.get("raw_metadata") or {}
        if isinstance(raw_metadata.get("scattering_plane"), dict):
            metadata["scattering_plane"] = raw_metadata["scattering_plane"]
            if isinstance(metadata.get("lattice"), dict):
                _attach_plane_to_lattice(metadata["lattice"], raw_metadata["scattering_plane"])
        if isinstance(raw_metadata.get("ub_matrix"), list):
            metadata["ub_matrix"] = raw_metadata["ub_matrix"]
            metadata["ub_convention"] = raw_metadata.get("ub_convention") or "unknown"
        if raw_metadata.get("monitor_counts") is not None:
            metadata["monitor_counts"] = raw_metadata["monitor_counts"]
        if raw_metadata.get("count_time") is not None:
            metadata["count_time"] = raw_metadata["count_time"]
        scans.append(
            Scan(
                scan_id=_scan_id(record, enrichment),
                experiment_id=_experiment_id(record, enrichment),
                instrument=record.get("instrument") or "unknown",
                scan_type=(enrichment.get("scan_type") if enrichment else None) or header_metadata.get("scan_type"),
                q_center=q_center,
                energy_transfer=_energy_transfer_from_record(record),
                instrument_config=_instrument_config_from_record(record),
                temperature_range=_temperature_range_from_metadata(header_metadata),
                metadata=metadata,
                provenance=[_record_provenance(record, "tas_data_readers.catalog_scan")],
            )
        )
    return scans


def _datafiles_from_records(records: list[dict], bt7_records_by_path: dict[str, dict] | None = None) -> list[DataFile]:
    datafiles = []
    for record in records:
        path = str(record.get("path") or "")
        enrichment = (bt7_records_by_path or {}).get(path, {})
        datafiles.append(
            DataFile(
                datafile_id=_datafile_id(path),
                path=path,
                checksum=None if _path_only_record(record) else _file_checksum(path),
                scan_id=_scan_id(record, enrichment),
                experiment_id=_experiment_id(record, enrichment),
                format=record.get("format_family") or None,
                metadata={
                    **_common_metadata(record, enrichment),
                    **_reader_metadata(record),
                    **_bt7_header_metadata(record),
                    "filename": Path(path).name,
                    "confidence": record.get("confidence"),
                    "source_reader": record.get("source_reader"),
                },
                provenance=[_record_provenance(record, "tas_data_readers.catalog_datafile")],
            )
        )
    return datafiles


def _common_metadata(record: dict, bt7_enrichment: dict | None = None) -> dict:
    bt7_enrichment = bt7_enrichment or {}
    return {
        "facility": record.get("facility"),
        "instrument": record.get("instrument"),
        "native_format": record.get("native_format"),
        "native_scan_id": _native_scan_id(record, bt7_enrichment),
        "native_experiment_id": _native_experiment_id(record, bt7_enrichment),
        "source_adapter": record.get("source_adapter"),
        "format_family": record.get("format_family"),
    }


def _reader_metadata(record: dict) -> dict:
    return {
        key: record[key]
        for key in [
            "scan_variable",
            "sample_name",
            "temperature",
            "magnetic_field",
            "q_hint",
            "energy_hint",
            "raw_metadata",
        ]
        if record.get(key) not in (None, {}, [])
    }


def _energy_transfer_from_record(record: dict) -> EnergyTransfer | None:
    hint = record.get("energy_hint")
    if isinstance(hint, dict) and hint.get("value") is not None:
        return EnergyTransfer(value=UnitValue(value=float(hint["value"]), unit=str(hint.get("unit") or "meV")))
    header = _bt7_header_metadata(record)
    transfer = header.get("energy_transfer")
    if isinstance(transfer, dict) and transfer.get("value") is not None:
        return EnergyTransfer(value=UnitValue(value=float(transfer["value"]), unit=str(transfer.get("unit") or "meV")))
    filename_match = re.search(r"(?i)(?P<energy>\d+(?:\.\d+)?)\s*mev", Path(str(record.get("path") or "")).name)
    if filename_match:
        return EnergyTransfer(value=UnitValue(value=float(filename_match.group("energy")), unit="meV"))
    return None


def _bt7_header_metadata(record: dict) -> dict:
    raw_metadata = record.get("raw_metadata") or {}
    header_fields = raw_metadata.get("header_fields")
    if not isinstance(header_fields, dict):
        return {}
    scan_description = _first_header_value(header_fields, "scandescr")
    detector_efficiencies = _first_header_value(header_fields, "detectorefficiencies")
    parsed = _parse_scan_description(scan_description)
    metadata: dict = {}
    if scan_description:
        metadata["scan_description"] = scan_description
    if parsed.get("JType"):
        metadata["scan_type"] = str(parsed["JType"]).upper()
    if parsed.get("Npts") is not None:
        try:
            metadata["npts"] = int(float(str(parsed["Npts"])))
        except ValueError:
            pass
    temperature_range = _parse_range_values(str(parsed.get("Range.Temp") or ""))
    if temperature_range:
        metadata["temperature_range"] = {"min": min(temperature_range), "max": max(temperature_range), "unit": "K", "source": "ScanDescr Range=Temp"}
    energy_range = _parse_range_values(str(parsed.get("Range.E") or ""))
    if energy_range:
        metadata["energy_range"] = {"min": min(energy_range), "max": max(energy_range), "unit": "meV", "source": "ScanDescr Range=E"}
        if abs(energy_range[0] - energy_range[-1]) < 1e-9:
            metadata["energy_transfer"] = {"value": energy_range[0], "unit": "meV", "source": "ScanDescr Range=E"}
    q_range = str(parsed.get("Range.Q") or "")
    if q_range:
        metadata["q_path_hint"] = {"raw_range": q_range, "source": "ScanDescr Range=Q"}
    psd_channels = _psd_channels_from_header(detector_efficiencies or "")
    if psd_channels:
        metadata["psd_channel_axis"] = psd_channels
        metadata["psd_channel_count"] = len(psd_channels)
        metadata.setdefault("detector_mode", "PSD")
        metadata.setdefault("analyzer_detector_mode", "PSD")
    a4 = _a4_from_filename(str(raw_metadata.get("filename") or record.get("path") or ""))
    if a4 is not None:
        metadata["a4"] = a4
        metadata["two_theta_axis"] = {"value": a4, "unit": "degree", "source": "filename"}
    return metadata


def _first_header_value(header_fields: dict, prefix: str) -> str | None:
    prefix = prefix.lower()
    for key, value in header_fields.items():
        if str(key).lower().startswith(prefix):
            return str(value)
    return None


def _parse_scan_description(description: str | None) -> dict[str, str]:
    if not description:
        return {}
    parsed: dict[str, str] = {}
    current_range: str | None = None
    for token in str(description).split(":"):
        if token.startswith("Range="):
            body = token[len("Range=") :]
            if "=" in body:
                axis, value = body.split("=", 1)
                current_range = f"Range.{axis}"
                parsed[current_range] = value
            else:
                current_range = None
        elif current_range and "=" not in token:
            parsed[current_range] = f"{parsed[current_range]}:{token}"
        elif "=" in token:
            key, value = token.split("=", 1)
            parsed[key] = value
            current_range = None
        else:
            current_range = None
    return parsed


def _parse_range_values(value: str) -> list[float]:
    if not value:
        return []
    numbers = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if len(numbers) < 2:
        return []
    return [float(numbers[0]), float(numbers[1])]


def _psd_channels_from_header(value: str) -> list[int]:
    channels = sorted({int(match) for match in re.findall(r"\bPSDC(\d{2})\s*=", value)})
    return channels


def _a4_from_filename(value: str) -> float | None:
    match = re.search(r"(?i)(?:^|[_-])(?P<whole>\d+)pt(?P<fraction>\d+)(?:[_-]|$)", Path(value).name)
    if not match:
        return None
    return float(f"{match.group('whole')}.{match.group('fraction')}")


def _temperature_range_from_metadata(metadata: dict) -> list[UnitValue]:
    temp_range = metadata.get("temperature_range")
    if not isinstance(temp_range, dict) or temp_range.get("min") is None or temp_range.get("max") is None:
        return []
    return [UnitValue(value=float(temp_range["min"]), unit="K"), UnitValue(value=float(temp_range["max"]), unit="K")]


def _instrument_config_from_record(record: dict) -> InstrumentConfig | None:
    raw_metadata = record.get("raw_metadata") or {}
    hint = raw_metadata.get("instrument_config_hint")
    if not isinstance(hint, dict) or not hint:
        return None
    fixed_energy = hint.get("fixed_energy")
    return InstrumentConfig(
        instrument=record.get("instrument") or "unknown",
        fixed_energy_mode=hint.get("fixed_energy_mode"),
        fixed_energy=UnitValue.model_validate(fixed_energy) if isinstance(fixed_energy, dict) else None,
        monochromator=hint.get("monochromator"),
        analyzer=hint.get("analyzer"),
        filters=list(hint.get("filters") or []),
        collimation=hint.get("collimation"),
    )


def _lattice_from_record(record: dict) -> dict | None:
    raw_metadata = record.get("raw_metadata") or {}
    lattice = raw_metadata.get("lattice")
    if isinstance(lattice, dict):
        return lattice
    legacy = raw_metadata.get("legacy_metadata")
    if isinstance(legacy, dict) and isinstance(legacy.get("lattice"), dict):
        return legacy["lattice"]
    return None


def _attach_plane_to_lattice(lattice: dict, scattering_plane: dict) -> None:
    if lattice.get("orient1") is None and scattering_plane.get("orient1") is not None:
        lattice["orient1"] = scattering_plane["orient1"]
    if lattice.get("orient2") is None and scattering_plane.get("orient2") is not None:
        lattice["orient2"] = scattering_plane["orient2"]


def _bt7_metadata(record: dict) -> dict:
    if not record:
        return {}
    return {
        "scan_type": record.get("scan_type"),
        "scan_variable": record.get("scan_variable"),
        "scan_description": record.get("scan_description"),
        "fixed_e": record.get("fixed_e"),
        "detector_mode": record.get("detector_mode"),
        "temperature": record.get("temp"),
        "temperature_setpoint": record.get("temp_setpoint"),
        "expt_id": record.get("expt_id"),
        "expt_name": record.get("expt_name"),
    }


def _experiment_id(record: dict, bt7_enrichment: dict | None = None) -> str:
    native = _native_experiment_id(record, bt7_enrichment or {})
    instrument = _slug(record.get("instrument") or "unknown")
    facility = _slug(record.get("facility") or "unknown")
    if native:
        return f"tas-experiment-{facility}-{instrument}-{_slug(str(native))}"
    parent = Path(record.get("path") or "").parent.name or "unknown"
    return f"tas-directory-{facility}-{instrument}-{_hash(parent)[:12]}"


def _scan_id(record: dict, bt7_enrichment: dict | None = None) -> str:
    native = _native_scan_id(record, bt7_enrichment or {})
    instrument = _slug(record.get("instrument") or "unknown")
    experiment = _slug(str(_native_experiment_id(record, bt7_enrichment or {}) or "unknown"))
    if native:
        return f"tas-scan-{instrument}-{experiment}-{_slug(str(native))}"
    return f"tas-scan-{_hash(record.get('path') or '')[:16]}"


def _datafile_id(path: str) -> str:
    return f"tas-datafile-{_hash(path)[:16]}"


def _file_checksum(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_only_record(record: dict) -> bool:
    raw_metadata = record.get("raw_metadata")
    return isinstance(raw_metadata, dict) and raw_metadata.get("metadata_mode") == "path_only"


def _experiment_title(record: dict, bt7_enrichment: dict | None = None) -> str:
    facility = record.get("facility") or "unknown facility"
    instrument = record.get("instrument") or "unknown instrument"
    native = _native_experiment_id(record, bt7_enrichment or {})
    if native:
        return f"{facility} {instrument} experiment {native}"
    return f"{facility} {instrument} directory import"


def _native_experiment_id(record: dict, bt7_enrichment: dict) -> str | None:
    return record.get("native_experiment_id") or bt7_enrichment.get("expt_id")


def _native_scan_id(record: dict, bt7_enrichment: dict) -> str | None:
    native = record.get("native_scan_id")
    if native:
        return native
    if record.get("instrument") != "BT-7":
        return None
    path = str(record.get("path") or "")
    match = re.search(r"(\d+)(?:\.bt7)?$", Path(path).name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _record_provenance(record: dict, method: str) -> ProvenanceRecord:
    return ProvenanceRecord(source_type="data_file", source_id=record.get("path"), method=method)


def _has_instrument(records: list[dict], instrument: str) -> bool:
    return any(record.get("instrument") == instrument for record in records)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"
