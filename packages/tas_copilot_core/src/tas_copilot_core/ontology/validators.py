from __future__ import annotations

from dataclasses import dataclass, field

from tas_copilot_contracts import ConceptMappingSet, ExternalVocabulary, TASConcept

from .mappings import normalize_concept_id


BROAD_EXTERNAL_LABELS = {
    "neutron scattering",
    "inelastic neutron spectroscopy",
    "neutron diffraction",
    "triple-axis spectroscopy",
    "dataset",
    "entity",
    "activity",
    "procedure",
}

SCHEMA_MAPPING_TYPES = {"sourceSchema"}
QUDT_MAPPING_TYPES = {"quantityKind", "unitKind", "unit"}
PROV_MAPPING_TYPES = {"provenanceRelation"}


@dataclass
class OntologyValidationReport:
    status: str = "ok"
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def add_error(self, code: str, message: str, **details) -> None:
        self.status = "error"
        self.errors.append({"code": code, "message": message, **details})

    def add_warning(self, code: str, message: str, **details) -> None:
        if self.status == "ok":
            self.status = "warning"
        self.warnings.append({"code": code, "message": message, **details})

    def as_dict(self) -> dict:
        return {"status": self.status, "error_count": len(self.errors), "warning_count": len(self.warnings), "errors": self.errors, "warnings": self.warnings}


def validate_ontology_alignment(
    vocabularies: list[ExternalVocabulary],
    concepts: list[TASConcept],
    mapping_sets: list[ConceptMappingSet],
) -> OntologyValidationReport:
    report = OntologyValidationReport()
    vocabulary_ids = {item.id for item in vocabularies}
    concept_ids = [normalize_concept_id(item.id) for item in concepts]
    concept_id_set = set(concept_ids)
    if len(concept_id_set) != len(concept_ids):
        report.add_error("duplicate_local_concept", "Local TAS ontology contains duplicate concept ids.")

    seen_mapping_concepts: set[str] = set()
    for mapping_set in mapping_sets:
        concept_id = normalize_concept_id(mapping_set.local_concept_id)
        if concept_id in seen_mapping_concepts:
            report.add_error("duplicate_mapping_set", "External mappings define the same local concept more than once.", local_concept_id=concept_id)
        seen_mapping_concepts.add(concept_id)
        if concept_id not in concept_id_set:
            report.add_error("unknown_local_concept", "External mapping references a local concept that is not in tas_bt7_seed.yaml.", local_concept_id=concept_id)

        concept = next((item for item in concepts if normalize_concept_id(item.id) == concept_id), None)
        if concept and concept.avoid_if_matches_too_broadly and mapping_set.retrieval_policy == "low_weight_synonym_expansion":
            report.add_warning("broad_concept_retrieval_expansion", "Broad local concept should not be configured for synonym expansion.", local_concept_id=concept_id)

        if mapping_set.retrieval_policy == "low_weight_synonym_expansion" and _only_broad_mappings(mapping_set):
            report.add_warning("only_broad_mappings_expand", "Concept has only broad external mappings but asks for synonym expansion.", local_concept_id=concept_id)

        for mapping in mapping_set.mappings:
            if mapping.vocabulary_id not in vocabulary_ids:
                report.add_error("unknown_vocabulary_id", "Mapping references an unknown external vocabulary id.", local_concept_id=concept_id, vocabulary_id=mapping.vocabulary_id)
            if mapping.mapping_type == "exactMatch" and not (mapping.external_iri or mapping.external_id):
                report.add_error("exact_match_missing_identifier", "exactMatch mappings require external_iri or external_id.", local_concept_id=concept_id, vocabulary_id=mapping.vocabulary_id)
            if not (mapping.external_iri or mapping.external_path or mapping.external_id):
                if mapping.mapping_status != "proposed" or not mapping.notes:
                    report.add_error("mapping_missing_reference", "Mapping needs external_iri, external_path, or external_id unless it is proposed with notes.", local_concept_id=concept_id, vocabulary_id=mapping.vocabulary_id)
            if mapping.mapping_type in SCHEMA_MAPPING_TYPES and mapping.vocabulary_id == "nexus" and not mapping.external_path:
                report.add_error("nexus_schema_mapping_missing_path", "NXtas sourceSchema mappings should use paths/references, not imported schema nodes.", local_concept_id=concept_id)
            if mapping.vocabulary_id == "qudt" and mapping.mapping_type not in QUDT_MAPPING_TYPES:
                report.add_warning("qudt_non_validation_mapping", "QUDT should normally be used for quantity kinds and units, not retrieval expansion.", local_concept_id=concept_id, mapping_type=mapping.mapping_type)
            if mapping.vocabulary_id == "prov_o" and mapping.mapping_type not in PROV_MAPPING_TYPES:
                report.add_warning("prov_non_provenance_mapping", "PROV-O should normally be used for provenance and lineage semantics.", local_concept_id=concept_id, mapping_type=mapping.mapping_type)
            if mapping.mapping_type == "broadMatch" and mapping.retrieval_policy == "low_weight_synonym_expansion":
                report.add_warning("broad_mapping_low_weight_only", "Broad mapping may be used only as low-weight metadata/synonym expansion.", local_concept_id=concept_id, vocabulary_id=mapping.vocabulary_id)
            if (mapping.external_label or "").lower() in BROAD_EXTERNAL_LABELS and mapping.mapping_type in {"exactMatch", "closeMatch"}:
                report.add_warning("suspicious_broad_external_class", "Mapping points to a broad external class; review before using for retrieval.", local_concept_id=concept_id, external_label=mapping.external_label)

    return report


def _only_broad_mappings(mapping_set: ConceptMappingSet) -> bool:
    return bool(mapping_set.mappings) and all(mapping.mapping_type == "broadMatch" for mapping in mapping_set.mappings)
