from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import (
    InstrumentDeviceGraph,
    InstrumentDeviceGraphEdge,
    InstrumentDeviceGraphNode,
    InstrumentGraphAlias,
    ProvenanceRecord,
)

from .config import CoreConfig
from .reporting import write_report_bundle


def default_alias_config_path() -> Path:
    return CoreConfig().root / "configs" / "instrument_graph" / "canonical_aliases.yaml"


def instrument_graph_from_nice_report(arguments: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(arguments["nice_report"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    alias_config = Path(arguments.get("alias_config") or default_alias_config_path()).expanduser()
    graph = build_instrument_graph_from_nice_report(report_path, alias_config=alias_config)
    payload = {
        "workflow_name": "instrument-device-graph-from-nice-report",
        "status": "success",
        "source_report": str(report_path),
        "alias_config": str(alias_config),
        "summary": {
            "graph_id": graph.graph_id,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "canonical_roles": graph.canonical_roles,
            "warning_count": len(graph.warnings),
        },
        "graph": graph.model_dump(mode="json"),
        "warnings": graph.warnings,
    }
    payload["artifacts"] = write_report_bundle(
        output_dir,
        payload,
        markdown_renderer=_instrument_graph_markdown,
        json_filename="instrument_device_graph_report.json",
        markdown_filename="instrument_device_graph_report.md",
    )
    graph_path = output_dir / "instrument_device_graph.json"
    graph_path.write_text(json.dumps(graph.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    payload["artifacts"]["graph_json"] = str(graph_path)
    return payload


def build_instrument_graph_from_nice_report(
    report_path: str | Path,
    *,
    alias_config: str | Path | None = None,
) -> InstrumentDeviceGraph:
    report_path = Path(report_path).expanduser()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    dag = payload.get("instrument_dag") or {}
    raw_nodes = dag.get("all_nodes") or []
    config_path = Path(alias_config).expanduser() if alias_config else default_alias_config_path()
    alias_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    roles = alias_payload.get("roles") or {}
    alias_index = _build_alias_index(roles)
    provenance = [
        ProvenanceRecord(
            source_type="nice_replay_report",
            source_id=str(report_path),
            source_uri=str(report_path),
            generated_by="instrument-device-graph-from-nice-report",
            method="canonical_alias_alignment_preserve_native_names",
            confidence=0.8,
            metadata={"alias_config": str(config_path)},
        )
    ]

    nodes: dict[str, InstrumentDeviceGraphNode] = {}
    edges: dict[str, InstrumentDeviceGraphEdge] = {}
    warnings: list[str] = []

    def add_edge(source: str, target: str, edge_type: str, confidence: float, evidence: list[str], status: str = "proposed") -> None:
        edge_id = f"edge-{_short_hash(source + target + edge_type)}"
        edges.setdefault(
            edge_id,
            InstrumentDeviceGraphEdge(
                edge_id=edge_id,
                source_node_id=source,
                target_node_id=target,
                edge_type=edge_type,
                confidence=confidence,
                status=status,
                evidence=evidence,
                provenance=provenance,
            ),
        )

    graph_node_id = "instrument-graph:nice_bt7"
    nodes[graph_node_id] = InstrumentDeviceGraphNode(
        node_id=graph_node_id,
        label="BT-7 NICE instrument graph",
        node_kind="instrument_graph",
        native_namespace="nice_bt7",
        facility="NCNR",
        instrument="BT-7",
        metadata={"source_report": str(report_path), "source_node_count": len(raw_nodes)},
        provenance=provenance,
    )

    for role, spec in roles.items():
        role_node_id = f"canonical-role:{role}"
        alias_records = [
            InstrumentGraphAlias(**alias)
            for alias in spec.get("aliases", [])
        ]
        nodes[role_node_id] = InstrumentDeviceGraphNode(
            node_id=role_node_id,
            label=spec.get("label") or role,
            node_kind=spec.get("node_kind") or "canonical_role",
            canonical_role=role,
            native_namespace="canonical_tas_role",
            aliases=alias_records,
            metadata={"alias_count": len(alias_records)},
            provenance=provenance,
        )

    matched_roles: set[str] = set()
    for raw in raw_nodes:
        native_name = str(raw.get("id") or raw.get("nodeID") or "")
        if not native_name:
            continue
        match = _match_alias(native_name, alias_index)
        role = match["role"] if match else None
        confidence = float(match["confidence"]) if match else 0.0
        node_id = f"native:nice_bt7:{native_name}"
        native_aliases = [
            InstrumentGraphAlias(
                alias=native_name,
                source_namespace="nice_bt7",
                facility="NCNR",
                instrument="BT-7",
                confidence=1.0,
                notes="Native NICE node id from getAllNodes().",
            )
        ]
        nodes[node_id] = InstrumentDeviceGraphNode(
            node_id=node_id,
            label=raw.get("description") or native_name,
            node_kind=_node_kind_for_native(raw, role, roles),
            canonical_role=role,
            native_name=native_name,
            native_namespace="nice_bt7",
            facility="NCNR",
            instrument="BT-7",
            device_id=raw.get("deviceID"),
            data_type=raw.get("dataType"),
            unit=raw.get("userUnit") or raw.get("internalUnit"),
            storage_mode=raw.get("storageMode"),
            has_control_input=raw.get("hasControlInput"),
            aliases=native_aliases,
            metadata={k: v for k, v in raw.items() if k not in {"repr"}},
            provenance=provenance,
        )
        add_edge(graph_node_id, node_id, "HAS_DEVICE_NODE", 0.95, [f"NICE node {native_name} was present in getAllNodes()."], status="observed")
        if raw.get("hasControlInput") is True:
            add_edge(graph_node_id, node_id, "HAS_CONTROL_INPUT", 0.9, [f"{native_name} reports hasControlInput=true."], status="observed")
        elif raw.get("hasControlInput") is False:
            add_edge(graph_node_id, node_id, "HAS_READBACK", 0.8, [f"{native_name} reports hasControlInput=false."], status="observed")
        if role:
            matched_roles.add(role)
            role_node_id = f"canonical-role:{role}"
            add_edge(node_id, role_node_id, "ALIGNED_TO_CANONICAL_ROLE", confidence, [f"{native_name} matched canonical role {role} via alias table."], status="observed")
            add_edge(role_node_id, node_id, "HAS_NATIVE_ALIAS", confidence, [f"{role} has native NICE alias {native_name}."], status="observed")

    _add_dependency_hints(nodes, edges, add_edge)

    unmatched_important = [
        role for role in roles if role not in matched_roles and _role_has_facility_alias(roles[role], "NCNR", "BT-7")
    ]
    if unmatched_important:
        warnings.append(
            "Some BT-7/NICE canonical roles were not observed in this report: "
            + ", ".join(sorted(unmatched_important))
        )
    warnings.append(
        "Native names are preserved. Canonical roles are alignment labels, not proof that all facilities use the same field names."
    )
    warnings.append(
        "Dependency edges are initial hints from observed nodes and known TAS semantics; future work should validate them by observing move effects and NICE internals."
    )

    return InstrumentDeviceGraph(
        graph_id=f"instrument-graph-{_short_hash(str(report_path) + str(len(raw_nodes)))}",
        facility="NCNR",
        instrument="BT-7",
        source_namespace="nice_bt7",
        source_ref=str(report_path),
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        canonical_roles=sorted(matched_roles),
        warnings=warnings,
        provenance=provenance,
    )


def _build_alias_index(roles: dict[str, Any]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for role, spec in roles.items():
        for alias in spec.get("aliases", []):
            index.append({"role": role, **alias})
    return index


def _match_alias(native_name: str, alias_index: list[dict[str, Any]]) -> dict[str, Any] | None:
    exact = [item for item in alias_index if item.get("alias") == native_name]
    if exact:
        return max(exact, key=lambda item: float(item.get("confidence", 0.0)))
    for item in alias_index:
        alias = str(item.get("alias", ""))
        if alias.endswith("*") and native_name.startswith(alias[:-1]):
            return item
    return None


def _node_kind_for_native(raw: dict[str, Any], role: str | None, roles: dict[str, Any]) -> str:
    if role and role in roles:
        return roles[role].get("node_kind") or "native_node"
    native_name = str(raw.get("id") or "").lower()
    if "counts" in native_name or "counter" in native_name or "monitor" in native_name:
        return "detector_readback"
    if "softposition" in native_name or "hardposition" in native_name:
        return "motor"
    if raw.get("hasControlInput") is True:
        return "control_node"
    return "readback_node"


def _add_dependency_hints(nodes: dict[str, InstrumentDeviceGraphNode], edges: dict[str, InstrumentDeviceGraphEdge], add_edge) -> None:
    existing = set(nodes)

    def role(role_id: str) -> str:
        return f"canonical-role:{role_id}"

    hints = [
        ("q_h", "sample_two_theta", "DEPENDS_ON", 0.45, "H-to-motor dependency depends on UB/sample state and powder mode."),
        ("q_k", "sample_two_theta", "DEPENDS_ON", 0.45, "K-to-motor dependency depends on UB/sample state and powder mode."),
        ("q_l", "sample_two_theta", "DEPENDS_ON", 0.45, "L-to-motor dependency depends on UB/sample state and powder mode."),
        ("q_norm", "sample_two_theta", "MOVES_MOTOR", 0.65, "Q magnitude maps to sample two-theta for elastic diffraction once wavelength is known."),
        ("powder_mode", "q_h", "CONSTRAINS", 0.7, "NICE Q.powderMode says moving H,K,L only moves A4 when true."),
        ("powder_mode", "q_k", "CONSTRAINS", 0.7, "NICE Q.powderMode says moving H,K,L only moves A4 when true."),
        ("powder_mode", "q_l", "CONSTRAINS", 0.7, "NICE Q.powderMode says moving H,K,L only moves A4 when true."),
        ("energy_transfer", "incident_energy", "DEPENDS_ON", 0.7, "NICE et.deltaE changes Ei or Ef depending on fixedEnergyMode."),
        ("energy_transfer", "final_energy", "DEPENDS_ON", 0.7, "NICE et.deltaE changes Ei or Ef depending on fixedEnergyMode."),
        ("fixed_energy_mode", "energy_transfer", "CONSTRAINS", 0.8, "NICE et.fixedEnergyMode controls which energy remains fixed."),
        ("detector_counts", "roi_readback", "READS_DETECTOR", 0.65, "ROI readback is derived from the active detector counts."),
        ("monitor_readback", "roi_readback", "NORMALIZES", 0.5, "Monitor is available as a normalization readback, but use is workflow-dependent."),
    ]
    for source_role, target_role, edge_type, confidence, evidence in hints:
        if role(source_role) in existing and role(target_role) in existing:
            add_edge(role(source_role), role(target_role), edge_type, confidence, [evidence], status="proposed")


def _role_has_facility_alias(spec: dict[str, Any], facility: str, instrument: str) -> bool:
    for alias in spec.get("aliases", []):
        if alias.get("facility") == facility and alias.get("instrument") == instrument:
            return True
    return False


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _instrument_graph_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    graph = report.get("graph", {})
    lines = [
        "# Instrument Device Graph Report",
        "",
        f"Status: {report.get('status')}",
        f"Source report: `{report.get('source_report')}`",
        f"Alias config: `{report.get('alias_config')}`",
        "",
        "## Summary",
        "",
        f"- Graph ID: `{summary.get('graph_id')}`",
        f"- Nodes: {summary.get('node_count')}",
        f"- Edges: {summary.get('edge_count')}",
        f"- Canonical roles observed: {', '.join(summary.get('canonical_roles') or []) or 'none'}",
        "",
        "## Naming Policy",
        "",
        "Native names are preserved. Canonical TAS roles are alignment labels used for routing, validation, and graph queries; they are not a claim that NICE, ICE, ANSTO, ORNL, SICS, and NeXus use the same names.",
        "",
        "## Warnings",
        "",
        *[f"- {item}" for item in report.get("warnings", [])],
        "",
        "## Edge Types",
        "",
    ]
    edge_counts: dict[str, int] = {}
    for edge in graph.get("edges", []):
        edge_counts[edge.get("edge_type", "unknown")] = edge_counts.get(edge.get("edge_type", "unknown"), 0) + 1
    for edge_type, count in sorted(edge_counts.items()):
        lines.append(f"- `{edge_type}`: {count}")
    lines.extend(["", "## Next Checks", "", "- Validate dependency hints by moving coordinate nodes in NICE and observing which motor nodes change.", "- Add facility-specific alias packs as ANSTO/HFIR/NeXus examples become available.", "- Keep aliases versioned and provenance-linked instead of renaming raw file fields."])
    return "\n".join(lines) + "\n"

