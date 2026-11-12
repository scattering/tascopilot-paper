from __future__ import annotations

from pydantic import Field

from .base import ContractModel, JsonDict
from .instrument import EnergyTransfer, InstrumentConfig, MotorTrajectory, QPoint, QPath
from .provenance import ProvenanceRecord
from .units import UnitValue


class Scan(ContractModel):
    scan_id: str
    experiment_id: str | None = None
    instrument: str = "BT-7"
    scan_type: str | None = None
    q_center: QPoint | None = None
    q_path: QPath | None = None
    energy_transfer: EnergyTransfer | None = None
    energy_range: list[UnitValue] = Field(default_factory=list)
    instrument_config: InstrumentConfig | None = None
    temperature_range: list[UnitValue] = Field(default_factory=list)
    magnetic_field_range: list[UnitValue] = Field(default_factory=list)
    motor_trajectory: MotorTrajectory | None = None
    physics_target: str | None = None
    concepts: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ScanSet(ContractModel):
    scanset_id: str
    scan_ids: list[str] = Field(default_factory=list)
    description: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

