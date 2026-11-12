from __future__ import annotations

from tas_copilot_contracts import RetrievalResult

from .common import default_retriever


def find_papers(query: str, limit: int = 10) -> list[RetrievalResult]:
    return [result for result in default_retriever().search(query, limit=limit) if result.source_type in {"paper", "textbook"}]

