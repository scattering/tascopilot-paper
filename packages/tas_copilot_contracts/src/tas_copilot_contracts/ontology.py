from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .base import ContractModel, JsonDict


MappingType = Literal[
    "exactMatch",
    "closeMatch",
    "broadMatch",
    "narrowMatch",
    "relatedMatch",
    "sourceSchema",
    "unitKind",
    "unit",
    "quantityKind",
    "provenanceRelation",
    "catalogTerm",
    "observationTerm",
    "characterizationTerm",
    "materialStructureTerm",
]

MappingStatus = Literal["proposed", "reviewed", "accepted", "rejected", "deprecated"]
RetrievalPolicy = Literal[
    "no_expansion",
    "metadata_filter_only",
    "low_weight_synonym_expansion",
    "schema_alignment_only",
    "provenance_alignment_only",
    "validation_only",
    "expert_review_required",
]


class ExternalMapping(ContractModel):
    vocabulary_id: str
    mapping_type: MappingType
    external_id: str | None = None
    external_label: str | None = None
    external_iri: str | None = None
    external_path: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mapping_status: MappingStatus = "proposed"
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None
    source: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    retrieval_policy: RetrievalPolicy = "no_expansion"


class ConceptXrefs(ContractModel):
    exactMatch: list[str] = Field(default_factory=list)
    closeMatch: list[str] = Field(default_factory=list)
    broadMatch: list[str] = Field(default_factory=list)
    narrowMatch: list[str] = Field(default_factory=list)
    relatedMatch: list[str] = Field(default_factory=list)
    source_schema: list[str] = Field(default_factory=list)
    nexus_paths: list[str] = Field(default_factory=list)
    qudt_quantity_kinds: list[str] = Field(default_factory=list)
    qudt_units: list[str] = Field(default_factory=list)
    prov_o_relations: list[str] = Field(default_factory=list)
    cif_terms: list[str] = Field(default_factory=list)
    panet_terms: list[str] = Field(default_factory=list)
    sosa_terms: list[str] = Field(default_factory=list)
    chameo_terms: list[str] = Field(default_factory=list)
    dcat_terms: list[str] = Field(default_factory=list)
    dcterms: list[str] = Field(default_factory=list)


class ExternalVocabulary(ContractModel):
    id: str
    label: str
    description: str
    role: str
    status: str = "reference"
    canonical_url: str
    documentation_url: str | None = None
    namespace_prefix: str
    namespace_iri: str
    local_usage: list[str] = Field(default_factory=list)
    use_for: list[str] = Field(default_factory=list)
    do_not_use_for: list[str] = Field(default_factory=list)
    retrieval_policy: str = "alignment_only"
    import_policy: str = "do_not_bulk_import"
    license_notes: str | None = None
    last_reviewed: str | None = None
    examples: list[str] = Field(default_factory=list)


class ConceptMappingSet(ContractModel):
    local_concept_id: str
    retrieval_policy: RetrievalPolicy = "no_expansion"
    mappings: list[ExternalMapping] = Field(default_factory=list)
    xrefs: ConceptXrefs = Field(default_factory=ConceptXrefs)
    notes: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class TASConcept(ContractModel):
    id: str
    label: str
    family: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    expected_modalities: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    avoid_if_matches_too_broadly: bool = False
    retrieval_policy: RetrievalPolicy = "no_expansion"
    external_mappings: list[ExternalMapping] = Field(default_factory=list)
    xrefs: ConceptXrefs = Field(default_factory=ConceptXrefs)
