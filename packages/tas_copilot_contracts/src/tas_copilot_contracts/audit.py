from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from .base import ContractModel, JsonDict, utc_now


class AuditRecord(ContractModel):
    audit_id: str = Field(default_factory=lambda: f"audit-{uuid4()}")
    event_type: str
    actor: str | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    status: str
    input_hash: str | None = None
    output_hash: str | None = None
    tool_namespace: str | None = None
    tool_name: str | None = None
    details: JsonDict = Field(default_factory=dict)

