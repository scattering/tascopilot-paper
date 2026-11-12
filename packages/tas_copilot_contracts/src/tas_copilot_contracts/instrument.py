from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord
from .units import UnitValue


class QPoint(ContractModel):
    h: float
    k: float
    l: float
    frame: str = "rlu"


class QPath(ContractModel):
    points: list[QPoint] = Field(default_factory=list)


class EnergyTransfer(ContractModel):
    value: UnitValue


class UBMatrix(ContractModel):
    ub_id: str
    matrix: list[list[float]] = Field(min_length=3, max_length=3)
    lattice_parameters: dict[str, UnitValue] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class MotorPosition(ContractModel):
    name: str
    position: UnitValue


class MotorTrajectory(ContractModel):
    trajectory_id: str
    positions: list[MotorPosition] = Field(default_factory=list)
    summary: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class InstrumentConfig(ContractModel):
    instrument: str = "BT-7"
    fixed_energy_mode: str | None = None
    fixed_energy: UnitValue | None = None
    monochromator: str | None = None
    analyzer: str | None = None
    filters: list[str] = Field(default_factory=list)
    collimation: str | None = None
    focusing: list[str] = Field(default_factory=list)
    sample_environment: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class InstrumentGraphAlias(ContractModel):
    alias: str
    source_namespace: str = "unknown"
    facility: str | None = None
    instrument: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    notes: str | None = None


class InstrumentDeviceGraphNode(ContractModel):
    node_id: str
    label: str | None = None
    node_kind: str = "unknown"
    canonical_role: str | None = None
    native_name: str | None = None
    native_namespace: str | None = None
    facility: str | None = None
    instrument: str | None = None
    device_id: str | None = None
    data_type: str | None = None
    unit: str | None = None
    storage_mode: str | None = None
    has_control_input: bool | None = None
    aliases: list[InstrumentGraphAlias] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class InstrumentDeviceGraphEdge(ContractModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: str = "proposed"
    evidence: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class InstrumentDeviceGraph(ContractModel):
    graph_id: str
    facility: str | None = None
    instrument: str | None = None
    source_namespace: str = "unknown"
    source_ref: str | None = None
    nodes: list[InstrumentDeviceGraphNode] = Field(default_factory=list)
    edges: list[InstrumentDeviceGraphEdge] = Field(default_factory=list)
    canonical_roles: list[str] = Field(default_factory=list)
    alias_policy: str = "preserve_native_names_align_to_canonical_roles"
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ResolutionInput(ContractModel):
    q_point: QPoint
    energy_transfer: EnergyTransfer
    instrument_config: InstrumentConfig


class ResolutionResult(ContractModel):
    summary: str
    resolution_matrix: list[list[float]] | None = None
    fwhm: dict[str, UnitValue] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class DeviceAngleInput(ContractModel):
    q_point: QPoint
    energy_transfer: EnergyTransfer
    instrument_config: InstrumentConfig
    ub_matrix: UBMatrix | None = None


class DeviceAngleResult(ContractModel):
    angles: list[MotorPosition] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
