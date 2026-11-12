from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord


class WorkflowResult(ContractModel):
    workflow_name: str
    status: str
    output: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CopilotWorkflowCall(ContractModel):
    workflow_name: str
    status: str
    output_dir: str | None = None
    summary: str | None = None
    artifacts: JsonDict = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CopilotRouteDecision(ContractModel):
    intent: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    runnable: bool = False
    safety_classification: str = "read_only_planning"
    blocked_reasons: list[str] = Field(default_factory=list)


class CopilotAskRequest(ContractModel):
    question: str
    output_dir: str

    # Optional explicit intent (e.g. from the MCP gateway). When set, the
    # router skips keyword matching for intent selection but still applies
    # the live-control block and the same input-validation/runnable logic.
    intent: str | None = None

    # Existing corpus / experiment inputs.
    corpus_catalog: str | None = None
    data_catalog: str | None = None
    data_root: str | None = None
    precedent_index: str | None = None
    paper_id: str | None = None
    figure_id: str | None = None
    proposal_evidence_dir: str | None = None
    actual_evidence_dir: str | None = None
    fixture: str | None = None
    experiment_config: str | None = None

    # New local document chunking / corpus QA inputs.
    # source_dir is used to create chunks from a directory of local files.
    source_dir: str | None = None

    # chunk_catalog is an optional alias for a chunks JSONL file.
    # If not supplied, corpus_catalog can still point directly to chunks.jsonl.
    chunk_catalog: str | None = None

    # Chunking controls.
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Existing run controls.
    max_files: int = 1000
    post_year: int = 2011
    hours: float | None = None
    since: str | None = None
    top_k: int = 10
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CopilotAskResponse(ContractModel):
    workflow_name: str = "copilot-ask"
    status: str
    question: str
    route: CopilotRouteDecision
    answer: str
    answer_envelope: JsonDict = Field(default_factory=dict)
    workflow_backbone_status: JsonDict = Field(default_factory=dict)
    routed_workflows: list[CopilotWorkflowCall] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    safe_next_actions: list[str] = Field(default_factory=list)
    not_approved_operations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    artifacts: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class StructureDatabaseCandidate(ContractModel):
    candidate_id: str
    source_database: str
    source_record_id: str
    formula: str
    normalized_formula: str | None = None
    composition: JsonDict = Field(default_factory=dict)
    chemical_name: str | None = None
    space_group: str | None = None
    space_group_number: int | None = None
    lattice: JsonDict = Field(default_factory=dict)
    year: int | None = None
    r_value: float | None = None
    score: float = Field(default=0.0, ge=0.0)
    score_components: JsonDict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class AlignmentPeakCandidate(ContractModel):
    peak_id: str
    h: int
    k: int
    l: int
    d_spacing: float
    q_inv_angstrom: float
    two_theta_deg: float
    scattering_plane_status: str
    out_of_plane_angle_deg: float | None = None
    relative_structure_factor: float | None = None
    structure_factor_status: str = "not_calculated"
    background_warnings: list[str] = Field(default_factory=list)
    multiple_scattering_warning: str | None = None
    score: float = Field(default=0.0, ge=0.0)
    score_components: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class AlignmentPeakSuggestionReport(ContractModel):
    workflow_name: str = "suggest-alignment-peaks"
    status: str
    formula: str
    scattering_plane: str | None = None
    energy_mev: float | None = None
    selected_structure: StructureDatabaseCandidate | None = None
    candidate_structures: list[StructureDatabaseCandidate] = Field(default_factory=list)
    alignment_peaks: list[AlignmentPeakCandidate] = Field(default_factory=list)
    background_materials: list[JsonDict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    safe_next_actions: list[str] = Field(default_factory=list)
    artifacts: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanelConstraint(ContractModel):
    panel_id: str
    label: str | None = None
    composition: JsonDict = Field(default_factory=dict)
    notes: str | None = None


class FigureImageConstraint(ContractModel):
    image_path: str | None = None
    manually_supplied_context: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigureAxisConstraint(ContractModel):
    axis_name: str
    min_value: float | None = None
    max_value: float | None = None
    unit: str | None = None


class FigureFeatureConstraint(ContractModel):
    feature_type: str
    value: float | None = None
    unit: str | None = None
    tolerance: float | None = None
    notes: str | None = None


class ScientificIntentConstraint(ContractModel):
    measurement_kind: str = "unknown"
    required_energy_transfer: float | None = None
    allowed_energy_transfer_range: JsonDict = Field(default_factory=dict)
    expected_detector_mode: str | None = None
    expected_scan_families: list[str] = Field(default_factory=list)
    excluded_scan_families: list[str] = Field(default_factory=list)
    expected_axes: list[str] = Field(default_factory=list)
    instrument: str | None = None
    dates: list[str] = Field(default_factory=list)
    date_range: JsonDict = Field(default_factory=dict)
    sample_type: str | None = None
    material_family: str | None = None
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanelComposition(ContractModel):
    panel_id: str
    variable_name: str | None = None
    value: float | None = None
    formula: str | None = None
    confidence: float = 0.0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanelLabel(ContractModel):
    panel_id: str
    label_text: str
    parsed_fields: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigureAxis(ContractModel):
    axis_name: str
    label: str | None = None
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigureAnnotation(ContractModel):
    annotation_id: str
    annotation_type: str
    label: str | None = None
    axis_name: str | None = None
    value: float | None = None
    unit: str | None = None
    notes: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigureCaption(ContractModel):
    caption_id: str
    figure_id: str
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PaperVersion(ContractModel):
    paper_id: str
    version_id: str
    citation: str | None = None
    doi: str | None = None
    source_uri: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Figure(ContractModel):
    figure_id: str
    paper_id: str
    label: str | None = None
    caption: FigureCaption | None = None
    panels: list["FigurePanel"] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanel(ContractModel):
    panel_id: str
    figure_id: str | None = None
    paper_id: str | None = None
    label: str | None = None
    panel_index: int | None = None
    panel_label: FigurePanelLabel | None = None
    composition: FigurePanelComposition | None = None
    material_identity: JsonDict = Field(default_factory=dict)
    x_variables: JsonDict = Field(default_factory=dict)
    measurement_kind: str | None = None
    detector_mode: str | None = None
    axes: list[FigureAxisConstraint | FigureAxis] = Field(default_factory=list)
    annotations: list[FigureAnnotation] = Field(default_factory=list)
    features: list[FigureFeatureConstraint] = Field(default_factory=list)
    caption_span: JsonDict = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_support_records: list[SourceSupportRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigureSourceQuery(ContractModel):
    query_id: str
    paper_id: str | None = None
    figure_id: str | None = None
    target_panel: str
    target_panel_label: str | None = None
    alternate_panels: list[FigurePanelConstraint] = Field(default_factory=list)
    material_family: str
    user_sample_formula: str
    target_composition: JsonDict = Field(default_factory=dict)
    alternate_compositions: list[JsonDict] = Field(default_factory=list)
    instrument: str
    measurement_mode: str
    measurement_kind: str | None = None
    required_energy_transfer: float | None = None
    allowed_energy_transfer_range: JsonDict = Field(default_factory=dict)
    expected_source_families: list[str] = Field(default_factory=list)
    excluded_source_families: list[str] = Field(default_factory=list)
    sample_type: str | None = None
    dates: list[str] = Field(default_factory=list)
    experimenters: list[str] = Field(default_factory=list)
    q_range: FigureAxisConstraint | None = None
    temperature_range: FigureAxisConstraint | None = None
    annotated_q_values: list[float] = Field(default_factory=list)
    expected_features: list[FigureFeatureConstraint] = Field(default_factory=list)
    figure_image_path: str | None = None
    attempt_reconstruction: bool = False
    max_reconstruction_candidates: int = 5
    notes: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ScientificSourceQuery(ContractModel):
    query_id: str
    query_type: str = "figure_source"
    figure_query: FigureSourceQuery | None = None
    intent: ScientificIntentConstraint = Field(default_factory=ScientificIntentConstraint)
    panels: list[FigurePanel] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CandidateEvidence(ContractModel):
    evidence_type: str
    summary: str
    score: float = 0.0
    source_id: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CandidateMismatch(ContractModel):
    mismatch_type: str
    summary: str
    severity: str = "warning"


class MaterialMention(ContractModel):
    raw_text: str
    source_object_type: str | None = None
    source_object_id: str | None = None
    source_field: str | None = None
    confidence: float = 0.0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CompositionParseResult(ContractModel):
    raw_string: str
    normalized_formula: str | None = None
    reduced_formula: str | None = None
    anonymous_formula: str | None = None
    element_amounts: JsonDict = Field(default_factory=dict)
    fractional_composition: JsonDict = Field(default_factory=dict)
    variable_composition: JsonDict = Field(default_factory=dict)
    x_variables: JsonDict = Field(default_factory=dict)
    parser_backend: str = "unavailable"
    parser_version: str | None = None
    interpretation_method: str | None = None
    confidence: float = 0.0
    ambiguity_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class MaterialIdentityEvidence(ContractModel):
    source: str = "unknown"
    extracted_text: str
    normalized_result: CompositionParseResult | None = None
    confidence: float = 0.0
    parser_backend: str | None = None
    parser_version: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class MaterialIdentityRecord(ContractModel):
    material_id: str
    raw_string: str
    source_object_type: str | None = None
    source_object_id: str | None = None
    source_field: str | None = None
    normalized_formula: str | None = None
    reduced_formula: str | None = None
    anonymous_formula: str | None = None
    element_amounts: JsonDict = Field(default_factory=dict)
    fractional_composition: JsonDict = Field(default_factory=dict)
    variable_composition: JsonDict = Field(default_factory=dict)
    x_variables: JsonDict = Field(default_factory=dict)
    parser_backend: str = "unavailable"
    parser_version: str | None = None
    interpretation_method: str | None = None
    confidence: float = 0.0
    ambiguity_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    evidence: list[MaterialIdentityEvidence] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FilenameCompositionEvidence(ContractModel):
    filename: str
    family_id: str | None = None
    parse_result: CompositionParseResult
    matched_pattern: str | None = None
    confidence: float = 0.0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PanelCompositionHypothesis(ContractModel):
    hypothesis_id: str
    panel_id: str | None = None
    material_identity: MaterialIdentityRecord | None = None
    variable_name: str | None = None
    variable_value: float | None = None
    label: str | None = None
    status: str = "active"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    mismatches: list[CandidateMismatch] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ScanFamilyClassification(ContractModel):
    source_id: str
    source_family: str
    confidence: float = 0.0
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    mismatches: list[CandidateMismatch] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class CandidateScanGroup(ContractModel):
    group_id: str
    member_scan_ids: list[str] = Field(default_factory=list)
    member_datafile_ids: list[str] = Field(default_factory=list)
    temperature_values: list[float] = Field(default_factory=list)
    q_coverage: JsonDict = Field(default_factory=dict)
    configuration_summary: JsonDict = Field(default_factory=dict)
    missing_metadata: list[str] = Field(default_factory=list)
    grouping_confidence: float = 0.0
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDTemperatureSeriesCandidate(CandidateScanGroup):
    psd_mode_evidence: list[str] = Field(default_factory=list)
    temperature_series_evidence: list[str] = Field(default_factory=list)


class PSDSourceGroup(ContractModel):
    group_id: str
    group_type: str
    member_scan_ids: list[str] = Field(default_factory=list)
    member_datafile_ids: list[str] = Field(default_factory=list)
    instrument: str | None = None
    material_identity_summary: JsonDict = Field(default_factory=dict)
    detector_mode_summary: JsonDict = Field(default_factory=dict)
    scan_type_summary: JsonDict = Field(default_factory=dict)
    energy_transfer_summary: JsonDict = Field(default_factory=dict)
    temperature_values: list[float] = Field(default_factory=list)
    temperature_range: JsonDict = Field(default_factory=dict)
    q_or_angle_coverage: JsonDict = Field(default_factory=dict)
    psd_channel_count: int | None = None
    fixed_angle_or_scanned_angle: str | None = None
    configuration_summary: JsonDict = Field(default_factory=dict)
    missing_metadata: list[str] = Field(default_factory=list)
    grouping_confidence: float = 0.0
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDTemperatureMap(ContractModel):
    map_id: str
    source_group_id: str | None = None
    channel_axis: list[float] = Field(default_factory=list)
    temperature_axis: list[float] = Field(default_factory=list)
    intensity_shape: list[int] = Field(default_factory=list)
    intensity_source: str | None = None
    axis_source: str = "channel_temperature"
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDMapReconstructionInput(ContractModel):
    candidate_id: str
    scan_ids: list[str] = Field(default_factory=list)
    datafile_ids: list[str] = Field(default_factory=list)
    output_dir: str
    reconstruction_level: str = "channel_temperature"
    parameters: JsonDict = Field(default_factory=dict)


class PSDMapReconstructionResult(ContractModel):
    status: str
    candidate_id: str
    reconstruction_level: str = "channel_temperature"
    input_files: list[str] = Field(default_factory=list)
    input_scans: list[str] = Field(default_factory=list)
    q_axis_source: str | None = None
    detector_axis_source: str | None = None
    temperature_source: str | None = None
    intensity_source: str | None = None
    normalization_applied: str | None = None
    corrections_applied: list[str] = Field(default_factory=list)
    smoothing_applied: str | None = None
    binning_applied: str | None = None
    q_range: JsonDict = Field(default_factory=dict)
    detector_axis_range: JsonDict = Field(default_factory=dict)
    temperature_range: JsonDict = Field(default_factory=dict)
    shape: list[int] = Field(default_factory=list)
    output_artifact_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    figure_comparison: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    audit_ids: list[str] = Field(default_factory=list)


class ReconstructionArtifact(ContractModel):
    artifact_id: str
    artifact_type: str
    path: str
    reconstruction_level: str
    input_source_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ArtifactDiagnostic(ContractModel):
    diagnostic_id: str
    diagnostic_type: str
    status: str
    observations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ReconstructionArtifact] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FeatureRealityAssessment(ContractModel):
    assessment_id: str
    source_data_identification: str = "candidate"
    reconstruction_status: str = "not_attempted"
    raw_data_evidence: list[CandidateEvidence] = Field(default_factory=list)
    processing_sensitivity: list[ArtifactDiagnostic] = Field(default_factory=list)
    detector_artifact_checks: list[ArtifactDiagnostic] = Field(default_factory=list)
    repeat_consistency: list[ArtifactDiagnostic] = Field(default_factory=list)
    background_sideband_evidence: list[CandidateEvidence] = Field(default_factory=list)
    conclusion: str = "insufficient_evidence"
    confidence: float = 0.0
    required_follow_up: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PaperFigureLink(ContractModel):
    paper_id: str
    figure_id: str
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanelSourceCandidate(ContractModel):
    panel_id: str
    candidate_id: str
    candidate_type: str = "file"
    score: float = 0.0
    confidence_label: str | None = None
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    mismatches: list[CandidateMismatch] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    rejected_reason: str | None = None
    status: str = "proposed"
    reconstruction_required: bool = True
    human_review_required: bool = True
    audit_ids: list[str] = Field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class FigurePanelSourceLink(ContractModel):
    figure_id: str
    panel_id: str
    candidate_source_id: str | None = None
    source_object_id: str | None = None
    source_object_type: str | None = None
    status: str = "candidate"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    reconstruction_required: bool = True
    human_review_required: bool = True
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PublishedArtifact(ContractModel):
    artifact_id: str
    artifact_type: str
    paper_id: str | None = None
    figure_id: str | None = None
    panel_id: str | None = None
    path_or_uri: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SourceSupportRecord(ContractModel):
    support_id: str
    claim_id: str
    source_object_id: str
    source_object_type: str
    status: str = "proposed"
    evidence_summary: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDCalibrationRecord(ContractModel):
    calibration_id: str
    instrument: str
    detector_mode: str | None = None
    date_range: JsonDict = Field(default_factory=dict)
    channel_mapping: JsonDict = Field(default_factory=dict)
    detector_geometry: JsonDict = Field(default_factory=dict)
    validation_evidence: list[CandidateEvidence] = Field(default_factory=list)
    status: str = "unavailable"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDChannelCalibrationRow(ContractModel):
    """One PSD channel of a FindPeak-derived calibration (channel index is zero based:
    ICE ``PSDC00`` / NICE ``PSD01`` is channel 0)."""

    channel: int
    efficiency: float
    relative_a4: float
    center_a4: float | None = None
    dcenter: float | None = None
    sigma: float | None = None
    area: float | None = None
    masked: bool = False


class PSDChannelCalibration(ContractModel):
    """Per-channel efficiency and A4 offset derived from one FindPeak A4 scan.

    ``relative_a4`` follows the historical BT-7 ``PSD_*.dat`` convention: the A4
    at which the channel saw the peak minus the scan's nominal A4, so the
    channel's scattering angle is ``A4 - scale * relative_a4`` once a scan-based
    check has confirmed the sign.  ``status`` starts as ``proposed``; the
    record never unlocks Q conversion by itself.
    """

    calibration_id: str
    instrument: str = "BT-7"
    source_file: str
    source_path: str | None = None
    source_date: str | None = None
    source_format: str | None = None
    method: str
    nominal_a4: float | None = None
    reference_channels: list[int] = Field(default_factory=list)
    channel_to_two_theta_policy: str = "two_theta = A4 - scale * relative_a4[channel]"
    analyzer_detector_translation: float | None = None
    channel_pitch_degrees: float | None = None
    channels: list[PSDChannelCalibrationRow] = Field(default_factory=list)
    status: str = "proposed"
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PSDDiffractionPattern(ContractModel):
    """A PSD scan binned into sample two-theta: intensity per bin with counting
    uncertainties, the bin totals behind them, and the uncalibrated channel sum
    against A4 for comparison.  Two-theta only; no Q axis."""

    source_file: str
    source_path: str | None = None
    calibration_id: str
    calibration_status: str = "proposed"
    sign_convention: str
    channel_to_two_theta_policy: str
    scale: float = 1.0
    bin_width_degrees: float
    monitor_reference: float
    two_theta: list[float] = Field(default_factory=list)
    intensity: list[float] = Field(default_factory=list)
    error: list[float] = Field(default_factory=list)
    counts: list[float] = Field(default_factory=list)
    exposure: list[float] = Field(default_factory=list)
    contributions: list[int] = Field(default_factory=list)
    a4: list[float] = Field(default_factory=list)
    a4_sum_intensity: list[float] = Field(default_factory=list)
    a4_sum_error: list[float] = Field(default_factory=list)
    masked_channels: list[int] = Field(default_factory=list)
    temperature_mean: float | None = None
    points: int = 0
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class InstrumentCalibrationRecord(ContractModel):
    """General instrument-calibration registry record.

    ``kind`` selects a kind-specific ``payload`` schema (validated by the
    loader in tas_copilot_core.instrument_calibrations). Designed so new
    kinds such as ``he3_cell_polarization`` or ``psd_channel_to_a4`` can be
    added without changing this envelope.
    """

    id: str
    kind: str
    instrument: str
    scope: JsonDict = Field(default_factory=dict)
    payload: JsonDict = Field(default_factory=dict)
    status: str = "candidate"
    evidence: list[str] = Field(default_factory=list)
    reviewed_by: str | None = None
    notes: list[str] = Field(default_factory=list)


class FigureSourceCandidate(ContractModel):
    candidate_id: str
    candidate_type: str
    interpretation: str
    scan_ids: list[str] = Field(default_factory=list)
    datafile_ids: list[str] = Field(default_factory=list)
    experiment_id: str | None = None
    total_score: float = 0.0
    score_breakdown: JsonDict = Field(default_factory=dict)
    confidence_label: str = "insufficient"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    mismatches: list[CandidateMismatch] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    scan_group: CandidateScanGroup | None = None
    reconstruction: PSDMapReconstructionResult | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    audit_ids: list[str] = Field(default_factory=list)


class CandidateSourceData(FigureSourceCandidate):
    source_family: str | None = None
    material_identity: MaterialIdentityRecord | None = None


class RejectedCandidate(ContractModel):
    candidate_id: str
    rejection_reason: str
    rejected_by: str = "workflow"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class HumanTriageCorrection(ContractModel):
    correction_id: str
    correction_type: str
    summary: str
    corrected_by: str = "human"
    affected_candidate_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class HypothesisState(ContractModel):
    hypothesis_id: str
    label: str
    status: str = "active"
    material_hypothesis: PanelCompositionHypothesis | None = None
    accepted_constraints: JsonDict = Field(default_factory=dict)
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    mismatches: list[CandidateMismatch] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SourceTriageState(ContractModel):
    query: ScientificSourceQuery | FigureSourceQuery
    active_hypotheses: list[HypothesisState] = Field(default_factory=list)
    accepted_constraints: JsonDict = Field(default_factory=dict)
    candidate_groups: list[PSDSourceGroup] = Field(default_factory=list)
    candidates: list[CandidateSourceData] = Field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)
    human_corrections: list[HumanTriageCorrection] = Field(default_factory=list)
    generated_artifacts: list[ReconstructionArtifact] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    audit_ids: list[str] = Field(default_factory=list)


class FigureSourceReport(ContractModel):
    workflow_name: str = "find-figure-source-data"
    status: str
    query: FigureSourceQuery
    panel_sample_warnings: list[str] = Field(default_factory=list)
    interpretations: list[JsonDict] = Field(default_factory=list)
    candidates: list[FigureSourceCandidate] = Field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)
    safe_conclusion: str
    next_actions: list[str] = Field(default_factory=list)
    artifacts: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    audit_id: str | None = None


class DashboardOverlayLayer(ContractModel):
    overlay_id: str
    overlay_type: str
    label: str
    material: str | None = None
    enabled_by_default: bool = False
    parameters: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class DashboardScanRecord(ContractModel):
    scan_id: str
    path: str
    instrument: str | None = None
    experiment_id: str | None = None
    title: str | None = None
    scan_type: str | None = None
    scan_variable: str | None = None
    date: str | None = None
    measurement_kind: str = "unknown"
    detector_mode: str | None = None
    row_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    q_center: JsonDict = Field(default_factory=dict)
    energy_range: JsonDict = Field(default_factory=dict)
    temperature_range: JsonDict = Field(default_factory=dict)
    intensity_summary: JsonDict = Field(default_factory=dict)
    narrow_feature_flags: list[JsonDict] = Field(default_factory=list)
    plot_data: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ExperimentDashboardSpec(ContractModel):
    dashboard_id: str
    title: str
    data_root: str | None = None
    update_policy: JsonDict = Field(default_factory=dict)
    scan_records: list[DashboardScanRecord] = Field(default_factory=list)
    overlay_layers: list[DashboardOverlayLayer] = Field(default_factory=list)
    artifact_checks: list[str] = Field(default_factory=list)
    generated_artifacts: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SpurionLikelihoodCandidate(ContractModel):
    candidate_id: str
    scan_id: str
    path: str
    candidate_type: str
    likelihood_score: float = Field(default=0.0, ge=0, le=1)
    likelihood_label: str = "insufficient_evidence"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    counter_evidence: list[CandidateEvidence] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    label_informed_prior: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
