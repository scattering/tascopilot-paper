from __future__ import annotations

from pathlib import Path

import yaml
from tas_copilot_contracts import TASConcept


def load_ontology(path: str | Path) -> list[TASConcept]:
    """Load a concept file. A top-level ``includes:`` list names further concept
    files (relative to the including file) whose concepts are appended, so a
    generated family (for example the MagMO-derived concepts) can live beside
    the hand-curated seed without being merged into it. Duplicate ids across
    files are an error."""
    return _load(Path(path), seen=set())


def _load(path: Path, *, seen: set[Path]) -> list[TASConcept]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"ontology include cycle at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if isinstance(raw, list):
        raw = {"concepts": raw}
    concepts = [TASConcept.model_validate(item) for item in raw.get("concepts") or []]
    for include in raw.get("includes") or []:
        concepts.extend(_load(path.parent / str(include), seen=seen))
    ids = [c.id for c in concepts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate ontology concept ids across included files: {duplicates}")
    return concepts
