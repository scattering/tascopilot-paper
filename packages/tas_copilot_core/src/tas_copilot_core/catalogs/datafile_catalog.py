from pathlib import Path

from tas_copilot_contracts import DataFile

from .jsonl_catalog import JsonlCatalog


def datafile_catalog(path: str | Path) -> JsonlCatalog[DataFile]:
    return JsonlCatalog(path, DataFile)

