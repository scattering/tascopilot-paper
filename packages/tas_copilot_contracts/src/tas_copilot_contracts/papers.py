from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord


class Paper(ContractModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    instrument: str | None = None
    abstract: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Corpus(ContractModel):
    corpus_id: str
    description: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Document(ContractModel):
    document_id: str
    document_type: str
    title: str | None = None
    paper_id: str | None = None
    local_path: str | None = None
    source_hash: str | None = None
    copyright_status: str | None = None
    ingest_status: str = "proposed"
    extraction_method: str | None = None
    parser_version: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PublicationDataLagPrior(ContractModel):
    prior_id: str
    paper_id: str
    publication_year: int | None = None
    typical_lag_years: float = 3.0
    broad_lag_min_years: int = 1
    broad_lag_max_years: int = 6
    prior_center_year: int | None = None
    plausible_start_year: int | None = None
    plausible_end_year: int | None = None
    ncnr_shutdown_cap_date: str | None = None
    ncnr_cap_applied: bool = False
    confidence: float = 0.35
    evidence_role: str = "prior_not_evidence"
    override_evidence_types: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class DocumentExtractionInput(ContractModel):
    path: str
    backend: str = "pdftotext_baseline"
    max_pages: int = 8
    max_chars: int = 20000
    output_dir: str | None = None
    write_artifacts: bool = False
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class DocumentExtractionResult(ContractModel):
    status: str
    path: str
    backend: str
    extraction_method: str | None = None
    parser_version: str | None = None
    source_hash: str | None = None
    text_char_count: int = 0
    page_count_requested: int | None = None
    chunk_candidates: list[JsonDict] = Field(default_factory=list)
    quality_metrics: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class DocumentExtractorBackendReport(ContractModel):
    backend: str
    status: str
    dependency_status: str = "unknown"
    extraction_quality: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ReviewArticle(Paper):
    review_scope: str | None = None


class TextbookSection(ContractModel):
    section_id: str
    title: str
    book_title: str | None = None
    section_path: str | None = None
    text: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SourceChunk(ContractModel):
    chunk_id: str
    source_type: str
    source_id: str
    text: str
    document_id: str | None = None
    paper_id: str | None = None
    chunk_type: str = "unknown"
    title: str | None = None
    span: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    subsection: str | None = None
    figure_id: str | None = None
    panel_id: str | None = None
    table_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    source_hash: str | None = None
    chunk_hash: str | None = None
    extraction_method: str | None = None
    parser_version: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PaperFactEvidence(ContractModel):
    chunk_id: str
    # The exact source text when the quote could be located (see quote_match);
    # otherwise the model's own transcription.
    quote: str | None = None
    # The model's transcription, kept only when it differs from the source text.
    quote_as_extracted: str | None = None
    # Character offsets of the quote in the cited chunk's text, when located.
    char_start: int | None = None
    char_end: int | None = None
    # verbatim | snapped | split_fragment | unverified
    quote_match: str | None = None
    page: int | None = None
    section: str | None = None
    support_status: str = "proposed"
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PaperFact(ContractModel):
    fact_id: str
    paper_id: str
    document_id: str | None = None
    fact_type: str
    value: JsonDict = Field(default_factory=dict)
    answer_status: str = "not_found"
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[PaperFactEvidence] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    extractor: str = "bounded_paper_fact_extractor"
    model: str | None = None
    prompt_version: str = "paper_fact_extraction_v0"
    human_review_required: bool = True
    warnings: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Table(ContractModel):
    table_id: str
    document_id: str | None = None
    paper_id: str | None = None
    title: str | None = None
    caption: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CorpusIngestRun(ContractModel):
    run_id: str
    corpus_id: str
    manifest_path: str
    status: str
    document_count: int = 0
    chunk_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CorpusRetrievalRun(ContractModel):
    run_id: str
    corpus_id: str | None = None
    query: str
    status: str
    result_count: int = 0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class EvidenceGraphNode(ContractModel):
    node_id: str
    node_type: str
    label: str | None = None
    source_object_id: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class EvidenceGraphEdge(ContractModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    confidence: float = 0.0
    status: str = "proposed"
    evidence: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    created_by: str = "workflow"
    created_at: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    rejection_reason: str | None = None


class ExperimentPlanningRisk(ContractModel):
    risk_id: str
    category: str
    summary: str
    predicted_from_proposal: bool = False
    promptable_from_proposal: bool = False
    actual_observed: bool = False
    planning_classification: str | None = None
    retrospective_classification: str = "not_observed"
    probability_score: float = 0.0
    proposal_evidence: list[str] = Field(default_factory=list)
    actual_evidence: list[str] = Field(default_factory=list)
    missing_prebeam_inputs: list[str] = Field(default_factory=list)
    recommended_prebeam_checks: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ProposalEvidenceRecord(ContractModel):
    record_id: str
    evidence_role: str
    evidence_type: str
    text: str
    source_path: str | None = None
    source_hash: str | None = None
    extraction_method: str | None = None
    extraction_status: str = "extracted"
    confidence: float | None = Field(default=None, ge=0, le=1)
    page_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    # Compatibility fields for current JSONL proposal/actual evidence workflows.
    chunk_id: str | None = None
    chunk_type: str | None = None
    evidence_id: str | None = None
    source_type: str | None = None


class ProposalEvidenceDiagnostic(ContractModel):
    diagnostic_id: str
    evidence_role: str
    source_path: str
    status: str
    extraction_method: str | None = None
    text_char_count: int = 0
    source_hash: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ProposalExperimentRetrospective(ContractModel):
    retrospective_id: str
    proposal_id: str
    experiment_id: str | None = None
    status: str
    risk_predictions: list[ExperimentPlanningRisk] = Field(default_factory=list)
    predicted_observed_count: int = 0
    promptable_observed_count: int = 0
    missed_observed_count: int = 0
    safe_conclusion: str
    limitations: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ExperimentScanSuggestion(ContractModel):
    suggestion_id: str
    priority: int = 0
    phase: str
    scan_type: str
    target: str
    scan_direction: str | None = None
    suggested_range: JsonDict = Field(default_factory=dict)
    fixed_coordinates: JsonDict = Field(default_factory=dict)
    step: float | None = None
    count_time: str | None = None
    temperature_targets: list[str] = Field(default_factory=list)
    purpose: str
    probability_score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    required_before_run: list[str] = Field(default_factory=list)
    artifact_checks: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ExperimentScanPlan(ContractModel):
    plan_id: str
    proposal_id: str
    status: str
    suggestions: list[ExperimentScanSuggestion] = Field(default_factory=list)
    missing_input_questions: list[str] = Field(default_factory=list)
    dashboard_recommendations: list[str] = Field(default_factory=list)
    not_approved_operations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    safe_conclusion: str
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ProposalPlanningBrief(ContractModel):
    brief_id: str
    proposal_id: str
    status: str
    risk_predictions: list[ExperimentPlanningRisk] = Field(default_factory=list)
    missing_input_questions: list[str] = Field(default_factory=list)
    required_prebeam_checks: list[str] = Field(default_factory=list)
    safe_conclusion: str
    not_approved_operations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
