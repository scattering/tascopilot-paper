from __future__ import annotations

from tas_copilot_contracts import RetrievalResult


def citation_label(result: RetrievalResult) -> str:
    title = result.title or result.source_id
    return f"{title} ({result.source_type}:{result.source_id})"

