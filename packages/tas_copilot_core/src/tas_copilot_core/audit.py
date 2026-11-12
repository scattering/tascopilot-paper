from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import AuditRecord, ProvenanceRecord

from .config import CoreConfig


class JsonlAuditLogger:
    def __init__(self, audit_dir: str | Path | None = None) -> None:
        self.audit_dir = Path(audit_dir) if audit_dir else CoreConfig().audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.audit_dir / "audit.jsonl"

    def write(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")


def provenance_for(method: str, source_type: str = "system", source_id: str | None = None) -> ProvenanceRecord:
    return ProvenanceRecord(source_type=source_type, source_id=source_id, method=method)

