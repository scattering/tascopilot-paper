from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord

RepresentationModality = Literal[
    "text",
    "scan_geometry",
    "spectrum",
    "motor_trajectory",
    "instrument_config",
    "resolution_summary",
    "correction_result",
    "spinwave_model_output",
    "tasai_plan",
    "andie_result",
    "spurion_check",
    "paper_experiment_mapping_signal",
]


class EmbeddingRecord(ContractModel):
    embedding_id: str
    source_object_type: str
    source_object_id: str
    modality: str
    encoder_name: str
    encoder_version: str
    vector_store_ref: str
    vector_dimension: int | None = Field(default=None, gt=0)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class RepresentationRecord(ContractModel):
    representation_id: str
    source_object_type: str
    source_object_id: str
    modality: RepresentationModality
    payload_ref: str | None = None
    payload: JsonDict = Field(default_factory=dict)
    embedding_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

