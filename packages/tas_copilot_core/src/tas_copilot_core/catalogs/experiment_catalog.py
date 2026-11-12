from pathlib import Path

from tas_copilot_contracts import Experiment, Scan

from .jsonl_catalog import JsonlCatalog


def experiment_catalog(path: str | Path) -> JsonlCatalog[Experiment]:
    return JsonlCatalog(path, Experiment)


def scan_catalog(path: str | Path) -> JsonlCatalog[Scan]:
    return JsonlCatalog(path, Scan)

