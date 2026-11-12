from pathlib import Path

from tas_copilot_contracts import Paper, SourceChunk, TextbookSection

from .jsonl_catalog import JsonlCatalog


def paper_catalog(path: str | Path) -> JsonlCatalog[Paper]:
    return JsonlCatalog(path, Paper)


def chunk_catalog(path: str | Path) -> JsonlCatalog[SourceChunk]:
    return JsonlCatalog(path, SourceChunk)


def textbook_catalog(path: str | Path) -> JsonlCatalog[TextbookSection]:
    return JsonlCatalog(path, TextbookSection)

