from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report
from tas_copilot_core.config import CoreConfig

from .catalog_tas_directory import catalog_tas_directory


def catalog_tas_batch(manifest_path: str | Path) -> dict[str, Any]:
    """Run catalog-tas-directory plus quality diagnostics for a YAML matrix."""

    manifest = _load_manifest(manifest_path)
    defaults = manifest.get("defaults", {})
    entries = manifest.get("directories", [])
    if not entries:
        return {
            "status": "missing_input",
            "manifest_path": str(manifest_path),
            "errors": [{"code": "missing_directories", "message": "manifest must contain at least one directory entry"}],
        }

    results = []
    for index, entry in enumerate(entries):
        label = entry.get("label") or f"entry-{index + 1}"
        db_path = entry.get("db_path") or _default_db_path(label)
        args = {
            **defaults,
            **{key: value for key, value in entry.items() if key not in {"label", "description"}},
            "db_path": str(db_path),
        }
        catalog_result = catalog_tas_directory(args)
        quality = data_catalog_quality_report(db_path) if catalog_result.get("status") == "success" else None
        results.append(
            {
                "label": label,
                "description": entry.get("description"),
                "path": entry.get("path") or entry.get("root"),
                "db_path": str(db_path),
                "catalog_status": catalog_result.get("status"),
                "quality_status": quality.get("status") if quality else "not_run",
                "catalog": catalog_result,
                "quality": quality,
            }
        )

    status = _batch_status(results)
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "result_count": len(results),
        "ok_count": sum(1 for item in results if item["catalog_status"] == "success" and item["quality_status"] == "ok"),
        "warning_count": sum(1 for item in results if item["quality_status"] == "warning"),
        "failure_count": sum(1 for item in results if item["catalog_status"] != "success"),
        "results": results,
    }


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _default_db_path(label: str) -> Path:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in label).strip("-") or "tas"
    return CoreConfig().catalog_dir / f"{slug}_tas_catalog.sqlite"


def _batch_status(results: list[dict[str, Any]]) -> str:
    if any(item["catalog_status"] != "success" for item in results):
        return "error"
    if any(item["quality_status"] == "warning" for item in results):
        return "warning"
    return "ok"
