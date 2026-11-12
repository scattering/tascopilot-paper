from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord


class RetrievalResult(ContractModel):
    source_type: str
    source_id: str
    title: str | None = None
    score: float = 0
    matched_terms: list[str] = Field(default_factory=list)
    matched_concepts: list[str] = Field(default_factory=list)
    snippet: str | None = None
    structured_summary: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    retrieval_method: str
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Claim(ContractModel):
    claim_id: str
    text: str
    claim_type: str
    support: list[ProvenanceRecord] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)

