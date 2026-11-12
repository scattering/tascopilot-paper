from __future__ import annotations

from pathlib import Path

import yaml
from tas_copilot_contracts import ConceptMappingSet, ExternalMapping, TASConcept

from tas_copilot_core.config import CoreConfig


def normalize_concept_id(concept_id: str) -> str:
    return concept_id.removeprefix("tas:")


def load_external_mappings(path: str | Path | None = None) -> list[ConceptMappingSet]:
    source = Path(path) if path else CoreConfig().external_mappings_path
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    items = raw.get("concept_mappings", raw if isinstance(raw, list) else [])
    return [ConceptMappingSet.model_validate(item) for item in items]


def mappings_by_concept(mapping_sets: list[ConceptMappingSet]) -> dict[str, ConceptMappingSet]:
    return {normalize_concept_id(item.local_concept_id): item for item in mapping_sets}


def mappings_for_concept(concept_id: str, mapping_sets: list[ConceptMappingSet] | None = None) -> ConceptMappingSet | None:
    mapping_sets = mapping_sets if mapping_sets is not None else load_external_mappings()
    return mappings_by_concept(mapping_sets).get(normalize_concept_id(concept_id))


def attach_external_mappings(concepts: list[TASConcept], mapping_sets: list[ConceptMappingSet]) -> list[TASConcept]:
    mapping_index = mappings_by_concept(mapping_sets)
    aligned = []
    for concept in concepts:
        mapping_set = mapping_index.get(normalize_concept_id(concept.id))
        if mapping_set is None:
            aligned.append(concept)
            continue
        aligned.append(
            concept.model_copy(
                update={
                    "external_mappings": list(mapping_set.mappings),
                    "xrefs": mapping_set.xrefs,
                    "retrieval_policy": mapping_set.retrieval_policy,
                }
            )
        )
    return aligned


def mapping_counts_by_vocabulary(mapping_sets: list[ConceptMappingSet]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mapping in _iter_mappings(mapping_sets):
        counts[mapping.vocabulary_id] = counts.get(mapping.vocabulary_id, 0) + 1
    return dict(sorted(counts.items()))


def mapping_counts_by_type(mapping_sets: list[ConceptMappingSet]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mapping in _iter_mappings(mapping_sets):
        key = str(mapping.mapping_type)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def mapping_counts_by_status(mapping_sets: list[ConceptMappingSet]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mapping in _iter_mappings(mapping_sets):
        key = str(mapping.mapping_status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _iter_mappings(mapping_sets: list[ConceptMappingSet]) -> list[ExternalMapping]:
    return [mapping for mapping_set in mapping_sets for mapping in mapping_set.mappings]
