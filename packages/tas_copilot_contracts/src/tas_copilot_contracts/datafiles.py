from __future__ import annotations

from pydantic import Field, model_validator

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord


class DataFile(ContractModel):
    datafile_id: str
    path: str
    scan_id: str | None = None
    experiment_id: str | None = None
    format: str | None = None
    checksum: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class TASDataFileMetadata(ContractModel):
    path: str
    format_family: str
    facility: str | None = None
    instrument: str | None = None
    native_format: str | None = None
    native_experiment_id: str | None = None
    native_scan_id: str | None = None
    source_adapter: str
    source_reader: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    scan_variable: str | None = None
    sample_name: str | None = None
    temperature: float | None = None
    magnetic_field: float | None = None
    q_hint: JsonDict = Field(default_factory=dict)
    energy_hint: JsonDict = Field(default_factory=dict)
    raw_metadata: JsonDict = Field(default_factory=dict)


class DerivedDataFile(DataFile):
    input_datafile_ids: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None
    parameters: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def lineage_required(self) -> "DerivedDataFile":
        if not self.input_datafile_ids:
            raise ValueError("derived data files must record input_datafile_ids")
        if not self.tool_call_id:
            raise ValueError("derived data files must record tool_call_id")
        return self
