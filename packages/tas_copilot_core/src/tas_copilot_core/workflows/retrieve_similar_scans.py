from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import Scan

from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.retrieval.similarity import compare_scans


def retrieve_similar_scans(scan_id: str, limit: int = 5, catalog: str | Path | None = None) -> list[dict]:
    scans = _load_scans(catalog)
    reference = next((scan for scan in scans if scan.scan_id == scan_id), None)
    if reference is None:
        return [
            {
                "status": "missing_input",
                "scan_id": scan_id,
                "missing_dimensions": ["scan_id"],
                "matched_dimensions": [],
                "mismatched_dimensions": [],
                "similarity_score": 0.0,
                "provenance": [],
            }
        ]
    scored = []
    for scan in scans:
        if scan.scan_id == scan_id:
            continue
        comparison = compare_scans(reference, scan)
        comparison["scan_id"] = scan.scan_id
        comparison["status"] = "success"
        scored.append(comparison)
    return sorted(scored, key=lambda item: item["similarity_score"], reverse=True)[:limit]


def _load_scans(catalog: str | Path | None) -> list[Scan]:
    if catalog:
        path = Path(catalog)
        if path.exists():
            return SQLiteDataCatalog(path).list_scans()
        return []
    config = CoreConfig()
    return JsonlCatalog(Path(config.catalog_dir) / "scans.jsonl", Scan).load()
