import pytest

from tas_copilot_contracts import ConceptMappingSet, ExternalMapping
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import (
    attach_external_mappings,
    external_alignment_diagnostics,
    load_external_mappings,
    load_external_vocabularies,
    load_ontology,
    ontology_alignment_report,
    validate_ontology_alignment,
)


def _fixtures():
    config = CoreConfig()
    return (
        load_external_vocabularies(config.external_vocabularies_path),
        load_ontology(config.ontology_path),
        load_external_mappings(config.external_mappings_path),
    )


def test_local_ontology_and_mapping_seed_load_with_xrefs():
    vocabularies, concepts, mapping_sets = _fixtures()
    report = validate_ontology_alignment(vocabularies, concepts, mapping_sets)
    aligned = attach_external_mappings(concepts, mapping_sets)
    constant_q = next(item for item in aligned if item.id == "constant_q_scan")

    assert report.status in {"ok", "warning"}
    assert constant_q.external_mappings
    assert constant_q.xrefs.exactMatch == []


def test_invalid_vocabulary_id_fails_validation():
    vocabularies, concepts, _ = _fixtures()
    mapping_sets = [
        ConceptMappingSet(
            local_concept_id="constant_q_scan",
            mappings=[ExternalMapping(vocabulary_id="missing_vocab", mapping_type="relatedMatch", external_id="x")],
        )
    ]

    report = validate_ontology_alignment(vocabularies, concepts, mapping_sets)

    assert report.status == "error"
    assert report.errors[0]["code"] == "unknown_vocabulary_id"


def test_invalid_mapping_type_and_status_fail_contract_validation():
    with pytest.raises(ValueError):
        ExternalMapping(vocabulary_id="qudt", mapping_type="badType", external_id="Energy")

    with pytest.raises(ValueError):
        ExternalMapping(vocabulary_id="qudt", mapping_type="quantityKind", external_id="Energy", mapping_status="badStatus")


def test_proposed_mappings_without_exact_iris_are_allowed_only_with_notes():
    vocabularies, concepts, _ = _fixtures()
    ok = [
        ConceptMappingSet(
            local_concept_id="constant_q_scan",
            mappings=[
                ExternalMapping(
                    vocabulary_id="chameo",
                    mapping_type="relatedMatch",
                    mapping_status="proposed",
                    notes="Needs expert review; no exact class selected.",
                )
            ],
        )
    ]
    bad = [
        ConceptMappingSet(
            local_concept_id="constant_q_scan",
            mappings=[ExternalMapping(vocabulary_id="chameo", mapping_type="relatedMatch", mapping_status="reviewed")],
        )
    ]

    assert validate_ontology_alignment(vocabularies, concepts, ok).status in {"ok", "warning"}
    assert validate_ontology_alignment(vocabularies, concepts, bad).status == "error"


def test_exact_match_requires_external_iri_or_id():
    vocabularies, concepts, _ = _fixtures()
    mapping_sets = [
        ConceptMappingSet(
            local_concept_id="constant_q_scan",
            mappings=[ExternalMapping(vocabulary_id="panet", mapping_type="exactMatch", external_label="something", notes="not enough")],
        )
    ]

    report = validate_ontology_alignment(vocabularies, concepts, mapping_sets)

    assert report.status == "error"
    assert any(error["code"] == "exact_match_missing_identifier" for error in report.errors)


def test_nexus_qudt_and_prov_mapping_policies_are_accepted():
    vocabularies, concepts, mapping_sets = _fixtures()
    report = validate_ontology_alignment(vocabularies, concepts, mapping_sets)
    mappings = [mapping for mapping_set in mapping_sets for mapping in mapping_set.mappings]

    assert report.status in {"ok", "warning"}
    assert any(mapping.vocabulary_id == "nexus" and mapping.mapping_type == "sourceSchema" and mapping.external_path for mapping in mappings)
    assert any(mapping.vocabulary_id == "qudt" and mapping.mapping_type == "quantityKind" for mapping in mappings)
    assert any(mapping.vocabulary_id == "prov_o" and mapping.mapping_type == "provenanceRelation" for mapping in mappings)


def test_broad_mappings_and_supernode_alignment_diagnostics_report_warnings():
    _, concepts, mapping_sets = _fixtures()
    aligned = attach_external_mappings(concepts, mapping_sets)
    diagnostics = external_alignment_diagnostics(aligned)
    report = ontology_alignment_report(*_fixtures())

    assert any(item["code"] == "only_broad_external_mappings" for item in diagnostics)
    assert "inelastic_neutron_spectroscopy" in report["concepts_with_only_broad_mappings"]
