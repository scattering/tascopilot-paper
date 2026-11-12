from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel, JsonDict
from .provenance import ProvenanceRecord
from .tools import ToolCall, ToolResult


ReductionNodeKind = Literal["input", "correction", "operation", "diagnostic", "artifact"]
ReductionCorrectionStatus = Literal["missing", "placeholder", "proposed", "reviewed", "accepted", "stale", "rejected"]
ReductionArtifactRole = Literal["raw_input", "correction", "intermediate", "derived_preview", "derived_result", "diagnostic"]
ReductionRunStatus = Literal["success", "missing_input", "unavailable", "unsupported", "error", "blocked", "validation_failed"]


class ReductionInputRef(ContractModel):
    input_id: str
    source_object_type: str
    source_object_id: str
    role: ReductionArtifactRole = "raw_input"
    immutable: bool = True
    content_hash: str | None = None
    path: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ReductionCorrectionRef(ContractModel):
    correction_id: str
    correction_type: str
    status: ReductionCorrectionStatus = "missing"
    applies_to: list[str] = Field(default_factory=list)
    version: str | None = None
    made_at: str | None = None
    valid_time_range: JsonDict = Field(default_factory=dict)
    stale_ok_for_preview: bool = False
    parameters: JsonDict = Field(default_factory=dict)
    source_object_id: str | None = None
    source_object_type: str | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def accepted_corrections_need_source(self) -> "ReductionCorrectionRef":
        if self.status == "accepted" and not (self.source_object_id or self.parameters):
            raise ValueError("accepted corrections require source_object_id or parameters")
        return self


class ReductionDAGTerminal(ContractModel):
    terminal_id: str
    data_type: str
    role: ReductionArtifactRole = "intermediate"
    metadata: JsonDict = Field(default_factory=dict)


class ReductionToolBinding(ContractModel):
    tool_namespace: str
    tool_name: str
    arguments: JsonDict = Field(default_factory=dict)
    input_bindings: dict[str, str] = Field(default_factory=dict)
    correction_id: str | None = None


class ReductionDAGNode(ContractModel):
    node_id: str
    kind: ReductionNodeKind
    operation: str
    label: str | None = None
    version: str = "0.1"
    input_ids: list[str] = Field(default_factory=list)
    correction_ids: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    parameters: JsonDict = Field(default_factory=dict)
    outputs: list[ReductionDAGTerminal] = Field(default_factory=list)
    cacheable: bool = True
    tool: ReductionToolBinding | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ReductionDAGEdge(ContractModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    source_terminal_id: str | None = None
    target_input_name: str | None = None
    role: str = "data_dependency"
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ReductionDAGTemplate(ContractModel):
    template_id: str
    label: str
    measurement_family: str
    nodes: list[ReductionDAGNode]
    edges: list[ReductionDAGEdge] = Field(default_factory=list)
    default_target_node_id: str | None = None
    raw_input_policy: str = "raw inputs are immutable; corrections and reductions are separate DAG inputs/artifacts"
    cache_policy: str = "cache keys include template, node, inputs, corrections, parameters, upstream fingerprints, and software versions"
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "ReductionDAGTemplate":
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("reduction node IDs must be unique")
        if len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("reduction edge IDs must be unique")
        if self.default_target_node_id and self.default_target_node_id not in nodes:
            raise ValueError("unknown reduction target node")
        incoming = {node_id: set() for node_id in nodes}
        bound_arguments: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
                raise ValueError("reduction edge references an unknown node")
            if edge.source_terminal_id and edge.source_terminal_id not in {
                terminal.terminal_id for terminal in nodes[edge.source_node_id].outputs
            }:
                raise ValueError("reduction edge references an unknown output terminal")
            if edge.target_input_name:
                binding = (edge.target_node_id, edge.target_input_name)
                if binding in bound_arguments:
                    raise ValueError("multiple edges bind the same input argument")
                bound_arguments.add(binding)
            incoming[edge.target_node_id].add(edge.source_node_id)
        while incoming:
            ready = {node_id for node_id, parents in incoming.items() if not parents}
            if not ready:
                raise ValueError("reduction DAG contains a cycle")
            incoming = {node_id: parents - ready for node_id, parents in incoming.items() if node_id not in ready}
        return self


class ReductionNodeResult(ContractModel):
    node_id: str
    status: ReductionRunStatus
    cache_key: str
    output: JsonDict = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    cache_hit: bool = False
    cache_audit_id: str | None = None


class ReductionDAGRunResult(ContractModel):
    run_id: str
    template_id: str
    status: ReductionRunStatus
    target_node_id: str | None = None
    ordered_node_ids: list[str] = Field(default_factory=list)
    node_results: list[ReductionNodeResult] = Field(default_factory=list)
    cache_keys: JsonDict = Field(default_factory=dict)
    raw_inputs: list[ReductionInputRef] = Field(default_factory=list)
    corrections: list[ReductionCorrectionRef] = Field(default_factory=list)
    derived_artifacts: list[JsonDict] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
