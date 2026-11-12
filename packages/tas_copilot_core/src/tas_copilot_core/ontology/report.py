from __future__ import annotations

from tas_copilot_contracts import ConceptMappingSet, ExternalVocabulary, TASConcept

from .mappings import mapping_counts_by_status, mapping_counts_by_type, mapping_counts_by_vocabulary, mappings_by_concept, normalize_concept_id
from .validators import validate_ontology_alignment


def ontology_alignment_report(
    vocabularies: list[ExternalVocabulary],
    concepts: list[TASConcept],
    mapping_sets: list[ConceptMappingSet],
) -> dict:
    mapping_index = mappings_by_concept(mapping_sets)
    concept_ids = {normalize_concept_id(item.id) for item in concepts}
    mapped_concept_ids = set(mapping_index)
    validation = validate_ontology_alignment(vocabularies, concepts, mapping_sets)
    only_broad = [
        concept_id
        for concept_id, mapping_set in mapping_index.items()
        if mapping_set.mappings and all(mapping.mapping_type == "broadMatch" for mapping in mapping_set.mappings)
    ]
    no_mappings = sorted(concept_ids - mapped_concept_ids)
    metadata_only = sorted(
        normalize_concept_id(concept.id)
        for concept in concepts
        if concept.avoid_if_matches_too_broadly or concept.retrieval_policy in {"metadata_filter_only", "schema_alignment_only", "validation_only"}
    )
    return {
        "local_concept_count": len(concepts),
        "mapped_concept_count": len(mapped_concept_ids & concept_ids),
        "mapping_set_count": len(mapping_sets),
        "mapping_count": sum(len(item.mappings) for item in mapping_sets),
        "counts_by_vocabulary": mapping_counts_by_vocabulary(mapping_sets),
        "counts_by_mapping_type": mapping_counts_by_type(mapping_sets),
        "counts_by_status": mapping_counts_by_status(mapping_sets),
        "concepts_with_no_mappings": no_mappings,
        "concepts_with_only_broad_mappings": sorted(only_broad),
        "metadata_or_schema_only_concepts": metadata_only,
        "warnings": validation.warnings,
        "errors": validation.errors,
        "status": validation.status,
    }


def format_alignment_report(report: dict) -> str:
    lines = [
        "TAS ontology external alignment report",
        "",
        f"Status: {report['status']}",
        f"Local concepts: {report['local_concept_count']}",
        f"Mapped concepts: {report['mapped_concept_count']}",
        f"Mapping records: {report['mapping_count']}",
        "",
        "Mappings by vocabulary:",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["counts_by_vocabulary"].items())
    lines.append("")
    lines.append("Mappings by type:")
    lines.extend(f"- {key}: {value}" for key, value in report["counts_by_mapping_type"].items())
    lines.append("")
    lines.append("Mappings by status:")
    lines.extend(f"- {key}: {value}" for key, value in report["counts_by_status"].items())
    if report["concepts_with_only_broad_mappings"]:
        lines.append("")
        lines.append("Concepts with only broad mappings:")
        lines.extend(f"- {item}" for item in report["concepts_with_only_broad_mappings"])
    if report["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {item['code']}: {item['message']}" for item in report["warnings"])
    if report["errors"]:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {item['code']}: {item['message']}" for item in report["errors"])
    return "\n".join(lines)


def export_alignment_json(vocabularies: list[ExternalVocabulary], concepts: list[TASConcept], mapping_sets: list[ConceptMappingSet]) -> dict:
    mapping_index = mappings_by_concept(mapping_sets)
    exported_concepts = []
    for concept in concepts:
        concept_id = normalize_concept_id(concept.id)
        item = concept.model_dump(mode="json")
        if concept_id in mapping_index:
            item["external_alignment"] = mapping_index[concept_id].model_dump(mode="json")
        exported_concepts.append(item)
    return {
        "external_vocabularies": [item.model_dump(mode="json") for item in vocabularies],
        "concepts": exported_concepts,
        "alignment_report": ontology_alignment_report(vocabularies, concepts, mapping_sets),
    }
