from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .base import ContractModel, JsonDict, utc_now
from .provenance import ProvenanceRecord


ReviewVerdict = Literal["positive", "negative", "inconclusive", "superseded", "unreviewed"]
ClaimState = Literal["proposed", "candidate", "accepted", "rejected", "stale", "superseded"]
RecommendationType = Literal["filter_out", "downweight", "calibrate_prior", "escalate", "no_action"]
UncertaintyClass = Literal[
    "missing_input",
    "probability_style_triage_prior",
    "private_precedent_prior",
    "planning_recommendation_requires_review",
    "source_grounded_candidate_evidence",
    "router_self_report",
    "unknown",
]


class ReviewLabel(ContractModel):
    """Human review label for calibration and replay-aware ranking updates.

    The label is intentionally workflow-neutral. Type-specific status values
    are preserved, while `verdict` and `claim_state` provide normalized axes
    that can be queried across order-parameter, spurion, paper-link, and
    figure-source workflows.
    """

    label_id: str = Field(default_factory=lambda: f"review-label-{uuid4()}")
    object_type: str
    object_id: str
    object_secondary_id: str | None = None

    source_workflow: str
    source_run_id: str | None = None
    source_artifact_path: str | None = None
    source_artifact_hash: str | None = None

    verdict: ReviewVerdict = "unreviewed"
    type_specific_status: str | None = None
    type_specific_allowed: list[str] = Field(default_factory=list)

    claim_state: ClaimState = "proposed"
    suggested_status: str = "proposed"

    evidence_channels_relied_on: list[str] = Field(default_factory=list)
    counter_evidence_observed: list[str] = Field(default_factory=list)
    recommendation: RecommendationType = "no_action"
    uncertainty_class: UncertaintyClass = "unknown"
    false_positive_reason: str | None = None
    review_notes: str = ""

    workflow_version: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    calibration_state_hash: str | None = None
    correction_status_snapshot: JsonDict = Field(default_factory=dict)
    dag_run_id: str | None = None
    cache_key: str | None = None
    cache_scope: str | None = None

    reviewer_id: str | None = None
    reviewer_role: str | None = None
    reviewed_at: datetime | None = None

    label_schema_version: str = "review_label_v0"
    legacy_provenance: str | list[ProvenanceRecord] | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _type_specific_status_must_be_allowed(self) -> "ReviewLabel":
        if self.type_specific_status and self.type_specific_allowed:
            if self.type_specific_status not in self.type_specific_allowed:
                raise ValueError("type_specific_status must be present in type_specific_allowed")
        return self


class ReviewLabelBatch(ContractModel):
    batch_id: str = Field(default_factory=lambda: f"review-label-batch-{uuid4()}")
    created_at: datetime = Field(default_factory=utc_now)
    source_template_paths: list[str] = Field(default_factory=list)
    labels: list[ReviewLabel] = Field(default_factory=list)
    decision_ontology_version: str = "decision_state_v0"
    schema_version: str = "review_label_v0"
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
