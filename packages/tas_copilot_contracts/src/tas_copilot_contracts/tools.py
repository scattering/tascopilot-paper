from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from uuid import uuid4

from pydantic import Field, model_validator

from .base import ContractModel, JsonDict, utc_now
from .provenance import ProvenanceRecord


class PermissionLevel(StrEnum):
    read_only = "read_only"
    writes_local_files = "writes_local_files"
    simulation_only = "simulation_only"
    planning_only = "planning_only"
    requires_human_approval = "requires_human_approval"
    live_control = "live_control"
    disabled = "disabled"


class Availability(StrEnum):
    available = "available"
    unavailable = "unavailable"
    optional = "optional"
    disabled = "disabled"


class SideEffectLevel(StrEnum):
    none = "none"
    local_files = "local_files"
    simulation = "simulation"
    planning = "planning"
    live_control = "live_control"


class ImplementationType(StrEnum):
    python_entrypoint = "python_entrypoint"
    cli = "cli"
    service = "service"
    idl = "idl"
    vm = "vm"
    manifest_only = "manifest_only"
    stub = "stub"


class ToolRuntimeMode(StrEnum):
    native_python = "native_python"
    conda_env = "conda_env"
    container_once = "container_once"
    container_service = "container_service"
    remote_hpc_job = "remote_hpc_job"
    unavailable = "unavailable"


class ToolRuntimeProtocol(StrEnum):
    stdin_stdout_json = "stdin_stdout_json"
    json_files = "json_files"
    service_json = "service_json"


class ToolResultStatus(StrEnum):
    success = "success"
    error = "error"
    unavailable = "unavailable"
    blocked = "blocked"
    validation_failed = "validation_failed"
    missing_input = "missing_input"


class ToolRuntimeSpec(ContractModel):
    runtime_id: str
    mode: ToolRuntimeMode
    label: str | None = None
    command: list[str] = Field(default_factory=list)
    conda_env_name: str | None = None
    conda_env_path: str | None = None
    python_executable: str | None = None
    container_image: str | None = None
    container_digest: str | None = None
    apptainer_image: str | None = None
    service_url: str | None = None
    hpc_profile: str | None = None
    input_protocol: ToolRuntimeProtocol = ToolRuntimeProtocol.stdin_stdout_json
    output_protocol: ToolRuntimeProtocol = ToolRuntimeProtocol.stdin_stdout_json
    timeout_seconds: int = Field(default=120, ge=1)
    working_directory: str | None = None
    environment: JsonDict = Field(default_factory=dict)
    read_only: bool = True
    network_access: bool = False
    mounts: list[JsonDict] = Field(default_factory=list)
    version_command: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def runtime_requires_explicit_boundary(self) -> "ToolRuntimeSpec":
        if self.mode == ToolRuntimeMode.unavailable:
            return self
        if self.mode in {ToolRuntimeMode.native_python, ToolRuntimeMode.conda_env} and not self.command:
            raise ValueError("native_python and conda_env runtime specs require an explicit command argv")
        if self.mode == ToolRuntimeMode.container_once and not (self.container_image or self.apptainer_image):
            raise ValueError("container_once runtime specs require container_image or apptainer_image")
        if self.mode == ToolRuntimeMode.container_service and not self.service_url:
            raise ValueError("container_service runtime specs require service_url")
        return self


class ToolRuntimeInvocation(ContractModel):
    runtime_id: str
    payload: JsonDict = Field(default_factory=dict)
    invocation_id: str = Field(default_factory=lambda: f"runtime-{uuid4()}")
    output_dir: str | None = None
    extra_arguments: list[str] = Field(default_factory=list)
    environment: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ToolError(ContractModel):
    code: str
    message: str
    details: JsonDict = Field(default_factory=dict)


class ToolSpec(ContractModel):
    name: str
    namespace: str
    version: str
    description: str
    input_schema: JsonDict
    output_schema: JsonDict
    side_effect_level: SideEffectLevel
    availability: Availability
    implementation_type: ImplementationType
    required_dependencies: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    provenance_policy: str
    permissions_required: list[PermissionLevel]
    enabled_by_default: bool = True
    owner_repo: str | None = None
    documentation_url: str | None = None
    contact: str | None = None
    maintainer: str | None = None

    @model_validator(mode="after")
    def live_control_cannot_default_enable(self) -> "ToolSpec":
        if PermissionLevel.live_control in self.permissions_required and self.enabled_by_default:
            raise ValueError("live_control tools must not be enabled by default")
        return self


class ToolCall(ContractModel):
    tool_name: str
    tool_namespace: str
    arguments: JsonDict = Field(default_factory=dict)
    caller_workflow: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    request_id: str = Field(default_factory=lambda: f"req-{uuid4()}")
    dry_run: bool = True
    user_context: JsonDict = Field(default_factory=dict)
    session_context: JsonDict = Field(default_factory=dict)
    approval_token: str | None = None
    provenance_context: list[ProvenanceRecord] = Field(default_factory=list)


class ToolResult(ContractModel):
    status: ToolResultStatus
    output: JsonDict | None = None
    errors: list[ToolError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    audit_id: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    tool_name: str
    tool_namespace: str
    tool_version: str | None = None
    git_sha: str | None = None
    runtime_environment: JsonDict = Field(default_factory=dict)
    derived_artifacts: list[JsonDict] = Field(default_factory=list)


class CorrectionInput(ContractModel):
    datafile_ids: list[str]
    correction_type: str
    parameters: JsonDict = Field(default_factory=dict)


class CorrectionResult(ContractModel):
    corrected_datafile_ids: list[str] = Field(default_factory=list)
    summary: str
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SpinWaveSiteSpec(ContractModel):
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    spin: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0], min_length=3, max_length=3)
    name: str = "X"


class SpinWaveExchangeSpec(ContractModel):
    type: Literal["heisenberg"] = "heisenberg"
    cell_offset: list[int] = Field(default_factory=lambda: [1, 0, 0], min_length=3, max_length=3)
    j: float = -1.0


class SpinWaveModelSpec(ContractModel):
    kind: str = "generic"
    material: str | None = None
    unit_cell: list[float] | None = Field(default=None, min_length=3, max_length=3)
    site: SpinWaveSiteSpec | None = None
    exchange: SpinWaveExchangeSpec | None = None
    model_parameters: JsonDict = Field(default_factory=dict)
    constraints: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ferromagnetic_chain_requires_typed_fields(self) -> "SpinWaveModelSpec":
        if self.kind == "ferromagnetic_chain" and not all([self.unit_cell, self.site, self.exchange]):
            raise ValueError("ferromagnetic_chain requires unit_cell, site, and exchange")
        return self


class SpinWaveCalculationInput(ContractModel):
    model: SpinWaveModelSpec
    q_path: list[list[float]] = Field(default_factory=lambda: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], min_length=2)
    n_points_per_segment: int = Field(default=5, ge=2)
    use_rust: bool = False

    @model_validator(mode="after")
    def q_points_are_three_vectors(self) -> "SpinWaveCalculationInput":
        for point in self.q_path:
            if len(point) != 3:
                raise ValueError("q_path entries must be 3-vectors")
        return self


class SpinWaveCalculationResult(ContractModel):
    summary: str
    simulation_only: bool = True
    model_kind: str | None = None
    q_point_count: int | None = None
    energy_shape: list[int] = Field(default_factory=list)
    intensity_shape: list[int] = Field(default_factory=list)
    energy_preview: list[Any] = Field(default_factory=list)
    intensity_preview: list[Any] = Field(default_factory=list)
    spectra_refs: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class TasaiPlan(ContractModel):
    plan_id: str
    objective: str
    suggested_scans: list[JsonDict] = Field(default_factory=list)
    safety_note: str = "Planning/simulation only; not approved instrument operation."
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class AdaptiveCandidatePoint(ContractModel):
    point_id: str
    nodes: JsonDict = Field(default_factory=dict)
    count_time: float | None = Field(default=None, gt=0)
    metadata: JsonDict = Field(default_factory=dict)


class AdaptiveObservation(ContractModel):
    step: int = Field(ge=0)
    point_id: str
    nodes: JsonDict = Field(default_factory=dict)
    count_time: float | None = Field(default=None, gt=0)
    expected_counts: float | None = Field(default=None, ge=0)
    observed_counts: float | None = Field(default=None, ge=0)
    uncertainty: float | None = Field(default=None, ge=0)
    policy_score: float | None = None
    policy_reason: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class AdaptivePolicyRequest(ContractModel):
    step: int = Field(ge=0)
    candidates: list[AdaptiveCandidatePoint] = Field(default_factory=list)
    observations: list[AdaptiveObservation] = Field(default_factory=list)
    model_spec: JsonDict = Field(default_factory=dict)
    software_limits: JsonDict = Field(default_factory=dict)
    policy_context: JsonDict = Field(default_factory=dict)


class AdaptivePolicyProposal(ContractModel):
    point_id: str
    nodes: JsonDict = Field(default_factory=dict)
    score: float
    reason: str
    count_time: float | None = Field(default=None, gt=0)
    metadata: JsonDict = Field(default_factory=dict)


class AdaptiveCountRequest(ContractModel):
    point: AdaptiveCandidatePoint
    model_spec: JsonDict = Field(default_factory=dict)
    count_time: float = Field(default=1.0, gt=0)
    count_context: JsonDict = Field(default_factory=dict)


class AdaptiveCountResult(ContractModel):
    expected_counts: float = Field(ge=0)
    expected_rate_hz: float | None = Field(default=None, ge=0)
    uncertainty: float | None = Field(default=None, ge=0)
    metadata: JsonDict = Field(default_factory=dict)


class AndieInput(ContractModel):
    powder_scan_set_id: str
    parameters: JsonDict = Field(default_factory=dict)


class AndieResult(ContractModel):
    summary: str
    output_refs: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class SpurionCheckInput(ContractModel):
    q_e_config: JsonDict


class SpurionCheckResult(ContractModel):
    summary: str
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class PaperExperimentLink(ContractModel):
    paper_id: str
    experiment_id: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    mapping_method: str
    experiment_year: int | None = None
    experiment_date: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    notes: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    status: str = Field(pattern="^(candidate|reviewed|rejected|accepted)$")
