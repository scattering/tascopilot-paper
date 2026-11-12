from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import FigureAxisConstraint, FigurePanelConstraint, FigureSourceQuery
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.source_triage import triage_records


def figure_source_triage(arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or CoreConfig().root / "data" / "processed" / "figure_source_triage").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    query = _load_query(arguments)
    records = _load_records(arguments)
    if not records:
        payload = {
            "workflow_name": "figure-source-triage",
            "status": "missing_input",
            "errors": [{"code": "missing_catalog_records", "message": "No catalog records were available for source triage."}],
            "safe_conclusion": "No candidate source data can be ranked until a catalog or fixture records are available.",
        }
        (output_dir / "figure_source_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    return triage_records(query, records, output_dir=output_dir)


def _load_query(arguments: dict[str, Any]) -> FigureSourceQuery:
    raw: dict[str, Any] = {}
    if arguments.get("query_file"):
        with Path(arguments["query_file"]).expanduser().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    raw.update(arguments.get("query", {}))
    raw.setdefault("query_id", "figure-source-triage")
    raw.setdefault("target_panel", raw.get("panel", "unknown"))
    raw.setdefault("material_family", raw.get("material_family", "LuFe_{1-x}Mn_xO3"))
    raw.setdefault("user_sample_formula", raw.get("sample", ""))
    raw.setdefault("instrument", raw.get("instrument", "BT-7"))
    raw.setdefault("measurement_mode", raw.get("expected_detector_mode", "PSD"))
    raw.setdefault("target_composition", {})
    if raw.get("target_panel_x_mn") is not None:
        raw["target_composition"]["x_mn"] = raw.pop("target_panel_x_mn")
    raw.setdefault("alternate_panels", [])
    if raw.get("alternate_panel") and raw.get("alternate_panel_x_mn") is not None:
        raw["alternate_panels"].append(
            FigurePanelConstraint(panel_id=str(raw.pop("alternate_panel")), composition={"x_mn": raw.pop("alternate_panel_x_mn")}).model_dump(mode="json")
        )
    for item in raw.pop("other_panels", []) or []:
        raw["alternate_panels"].append(FigurePanelConstraint(panel_id=str(item["panel"]), composition={"x_mn": item.get("x_mn")}).model_dump(mode="json"))
    if isinstance(raw.get("q_range"), dict):
        raw["q_range"] = FigureAxisConstraint(axis_name="Q", min_value=raw["q_range"].get("min"), max_value=raw["q_range"].get("max"), unit="angstrom^-1").model_dump(mode="json")
    if isinstance(raw.get("temperature_range"), dict):
        raw["temperature_range"] = FigureAxisConstraint(axis_name="temperature", min_value=raw["temperature_range"].get("min"), max_value=raw["temperature_range"].get("max"), unit="K").model_dump(mode="json")
    if raw.get("expected_source_families") or raw.get("excluded_source_families") or raw.get("measurement_kind"):
        raw["notes"] = "\n".join(
            [
                str(raw.get("notes") or ""),
                f"measurement_kind={raw.get('measurement_kind')}",
                f"expected_source_families={raw.get('expected_source_families')}",
                f"excluded_source_families={raw.get('excluded_source_families')}",
            ]
        )
    raw["dates"] = [str(item) for item in raw.get("dates", [])]
    allowed = set(FigureSourceQuery.model_fields)
    return FigureSourceQuery.model_validate({key: value for key, value in raw.items() if key in allowed})


def _load_records(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    if arguments.get("records"):
        return list(arguments["records"])
    catalog = arguments.get("catalog")
    if not catalog:
        return []
    path = Path(catalog).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload.get("records") or payload.get("datafiles") or []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("records") if isinstance(payload, dict) else payload
    db = SQLiteDataCatalog(path)
    scans = {scan.scan_id: scan for scan in db.list_scans()}
    records = []
    for datafile in db.list_datafiles():
        payload = datafile.model_dump(mode="json")
        if datafile.scan_id and datafile.scan_id in scans:
            payload["scan"] = scans[datafile.scan_id].model_dump(mode="json")
            payload["metadata"] = {**payload.get("metadata", {}), **scans[datafile.scan_id].metadata}
        records.append(payload)
    return records
