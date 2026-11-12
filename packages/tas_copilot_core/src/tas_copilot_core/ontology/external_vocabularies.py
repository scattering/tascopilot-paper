from __future__ import annotations

from pathlib import Path

import yaml
from tas_copilot_contracts import ExternalVocabulary

from tas_copilot_core.config import CoreConfig


def load_external_vocabularies(path: str | Path | None = None) -> list[ExternalVocabulary]:
    source = Path(path) if path else CoreConfig().external_vocabularies_path
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    items = raw.get("external_vocabularies", raw if isinstance(raw, list) else [])
    return [ExternalVocabulary.model_validate(item) for item in items]


def vocabulary_by_id(path: str | Path | None = None) -> dict[str, ExternalVocabulary]:
    return {vocabulary.id: vocabulary for vocabulary in load_external_vocabularies(path)}
