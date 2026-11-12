from __future__ import annotations

from tas_copilot_contracts import TASConcept

from .linker import match_concepts


def supernode_diagnostics(texts: list[str], concepts: list[TASConcept], max_fraction: float = 0.6) -> list[dict]:
    warnings = []
    total = max(len(texts), 1)
    for concept in concepts:
        count = sum(1 for text in texts if concept in match_concepts(text, [concept]))
        fraction = count / total
        if fraction > max_fraction or concept.avoid_if_matches_too_broadly:
            warnings.append({"concept_id": concept.id, "match_count": count, "fraction": fraction})
    return warnings


def external_alignment_diagnostics(concepts: list[TASConcept]) -> list[dict]:
    warnings = []
    for concept in concepts:
        broad_mappings = [mapping for mapping in concept.external_mappings if mapping.mapping_type == "broadMatch"]
        if concept.avoid_if_matches_too_broadly and broad_mappings:
            warnings.append(
                {
                    "concept_id": concept.id,
                    "code": "broad_local_concept_with_broad_external_mapping",
                    "message": "Use this concept as metadata/alignment, not as a primary retrieval node.",
                }
            )
        if concept.external_mappings and len(broad_mappings) == len(concept.external_mappings):
            warnings.append(
                {
                    "concept_id": concept.id,
                    "code": "only_broad_external_mappings",
                    "message": "Concept has only broad external mappings and should be reviewed before retrieval expansion.",
                }
            )
        if concept.retrieval_policy in {"schema_alignment_only", "validation_only", "provenance_alignment_only"}:
            warnings.append(
                {
                    "concept_id": concept.id,
                    "code": "not_primary_retrieval_node",
                    "message": f"Concept retrieval policy is {concept.retrieval_policy}.",
                }
            )
    return warnings
