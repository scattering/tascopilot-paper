from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord


class Proposal(ContractModel):
    proposal_id: str
    title: str | None = None
    pi: str | None = None
    cycle: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Sample(ContractModel):
    sample_id: str
    name: str
    composition: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class Experiment(ContractModel):
    experiment_id: str
    instrument: str = "BT-7"
    proposal_id: str | None = None
    title: str | None = None
    sample_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    concepts: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

