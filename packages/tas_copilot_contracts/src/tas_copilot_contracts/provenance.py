from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from .base import ContractModel, JsonDict, utc_now


class ProvenanceRecord(ContractModel):
    provenance_id: str = Field(default_factory=lambda: f"prov-{uuid4()}")
    source_type: str = Field(default="unknown")
    source_id: str | None = None
    source_uri: str | None = None
    source_span: str | None = None
    generated_by: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    method: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: JsonDict = Field(default_factory=dict)

