from pathlib import Path

from tas_copilot_contracts import PaperExperimentLink

from .jsonl_catalog import JsonlCatalog


def mapping_catalog(path: str | Path) -> JsonlCatalog[PaperExperimentLink]:
    return JsonlCatalog(path, PaperExperimentLink)

