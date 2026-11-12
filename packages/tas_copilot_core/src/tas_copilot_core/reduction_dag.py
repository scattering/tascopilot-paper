from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tas_copilot_contracts import (
    ProvenanceRecord,
    ReductionCorrectionRef,
    ReductionDAGEdge,
    ReductionDAGNode,
    ReductionDAGRunResult,
    ReductionDAGTemplate,
    ReductionDAGTerminal,
    ReductionInputRef,
    ReductionNodeResult,
)
from tas_copilot_core.reduction_cache import ReductionCache
from tas_copilot_core.reduction_tools import (
    audit_cache_hit, bound_tool_call, cacheable_tool_call, tool_fingerprint, tool_node_result,
)
from tas_copilot_core.registry import ToolRunner
from tas_copilot_core.registry.tool_runner import stable_hash


NodeHandler = Callable[[ReductionDAGNode, dict[str, Any]], dict[str, Any]]


def order_parameter_temperature_series_template() -> ReductionDAGTemplate:
    """Template for the common elastic peak-vs-temperature reduction family."""
    return ReductionDAGTemplate(
        template_id="tas.order_parameter_temperature_series.v0",
        label="Order parameter / temperature-series preview",
        measurement_family="elastic_temperature_series",
        default_target_node_id="assemble_order_parameter_preview",
        nodes=[
            ReductionDAGNode(
                node_id="raw_scan_series",
                kind="input",
                operation="input.scan_series",
                label="Raw scan series",
                input_ids=["scan_series"],
                required_fields=["scan_ids", "temperature_values"],
                outputs=[ReductionDAGTerminal(terminal_id="scan_series", data_type="scan_series", role="raw_input")],
            ),
            ReductionDAGNode(
                node_id="monitor_normalization",
                kind="correction",
                operation="correction.monitor_normalization",
                label="Monitor normalization correction",
                correction_ids=["monitor_normalization"],
                parameters={"optional_for_preview": True},
                outputs=[ReductionDAGTerminal(terminal_id="normalization", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="background_model",
                kind="correction",
                operation="correction.background_model",
                label="Background model or sideband choice",
                correction_ids=["background_model"],
                parameters={"optional_for_preview": True},
                outputs=[ReductionDAGTerminal(terminal_id="background", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="assemble_order_parameter_preview",
                kind="operation",
                operation="preview.order_parameter_temperature_series",
                label="Assemble order-parameter preview",
                required_fields=["scan_ids", "temperature_values"],
                outputs=[ReductionDAGTerminal(terminal_id="preview", data_type="temperature_series_preview", role="derived_preview")],
            ),
        ],
        edges=[
            ReductionDAGEdge(edge_id="e_raw_to_preview", source_node_id="raw_scan_series", target_node_id="assemble_order_parameter_preview"),
            ReductionDAGEdge(edge_id="e_norm_to_preview", source_node_id="monitor_normalization", target_node_id="assemble_order_parameter_preview", role="correction_dependency"),
            ReductionDAGEdge(edge_id="e_bg_to_preview", source_node_id="background_model", target_node_id="assemble_order_parameter_preview", role="correction_dependency"),
        ],
        warnings=[
            "This template creates a preview/lineage plan, not an accepted order-parameter fit.",
            "Peak model, background choice, and transition/splitting logic must remain explicit DAG branches.",
        ],
        provenance=[ProvenanceRecord(source_type="template", source_id="order_parameter_temperature_series", method="redactus_inspired_tas_native_template")],
    )


def order_parameter_fit_branch_template(*, model_branch: str = "single_peak_power_law", bic_tag: str = "not_computed") -> ReductionDAGTemplate:
    """Order-parameter template with a lightweight fit/model branch node.

    This keeps the first DAG proof of concept scoped to lineage and label
    re-evaluation. The node records the selected model branch and fit evidence;
    it does not replace the existing fit implementation.
    """

    template = order_parameter_temperature_series_template()
    nodes = list(template.nodes)
    edges = list(template.edges)
    nodes.append(
        ReductionDAGNode(
            node_id="order_parameter_model_branch",
            kind="operation",
            operation="model_branch.order_parameter_fit",
            label="Order-parameter model branch",
            parameters={"model_branch": model_branch, "bic_tag": bic_tag},
            outputs=[ReductionDAGTerminal(terminal_id="model_branch", data_type="order_parameter_model_branch", role="derived_result")],
        )
    )
    edges.append(
        ReductionDAGEdge(
            edge_id="e_preview_to_model_branch",
            source_node_id="assemble_order_parameter_preview",
            target_node_id="order_parameter_model_branch",
            role="model_branch_dependency",
        )
    )
    return ReductionDAGTemplate(
        template_id="tas.order_parameter_temperature_series.fit_branch.v0",
        label="Order parameter / temperature-series fit branch",
        measurement_family=template.measurement_family,
        default_target_node_id="order_parameter_model_branch",
        nodes=nodes,
        edges=edges,
        warnings=[
            *template.warnings,
            "The model-branch node records fit lineage and model-selection tags; it does not accept a fit without review.",
        ],
        provenance=[ProvenanceRecord(source_type="template", source_id="order_parameter_fit_branch", method="redactus_inspired_tas_native_template")],
    )


def psd_channel_temperature_preview_template() -> ReductionDAGTemplate:
    """Template for safe PSD channel x temperature previews before reviewed Q calibration."""
    return ReductionDAGTemplate(
        template_id="tas.psd_channel_temperature_preview.v0",
        label="PSD channel x temperature preview",
        measurement_family="elastic_psd_temperature_map",
        default_target_node_id="assemble_psd_channel_temperature_preview",
        nodes=[
            ReductionDAGNode(
                node_id="raw_psd_scan",
                kind="input",
                operation="input.psd_temperature_scan",
                label="Raw PSD scan or scan group",
                input_ids=["psd_scan"],
                required_fields=["temperature_values", "psd_counts"],
                outputs=[ReductionDAGTerminal(terminal_id="psd_scan", data_type="psd_scan", role="raw_input")],
            ),
            ReductionDAGNode(
                node_id="psd_efficiency_correction",
                kind="correction",
                operation="correction.psd_efficiency",
                label="PSD efficiency correction",
                correction_ids=["psd_efficiency"],
                parameters={"optional_for_preview": True},
                outputs=[ReductionDAGTerminal(terminal_id="efficiency", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="psd_channel_to_a4_mapping",
                kind="correction",
                operation="correction.psd_channel_to_a4",
                label="PSD channel-to-A4 mapping",
                correction_ids=["psd_channel_to_a4"],
                parameters={"optional_for_channel_temperature": True, "required_for_q_temperature": True},
                outputs=[ReductionDAGTerminal(terminal_id="channel_to_a4", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="assemble_psd_channel_temperature_preview",
                kind="operation",
                operation="preview.psd_channel_temperature",
                label="Assemble detector-channel x temperature preview",
                required_fields=["temperature_values", "psd_counts"],
                parameters={"reconstruction_level": "channel_temperature"},
                outputs=[ReductionDAGTerminal(terminal_id="preview", data_type="psd_channel_temperature_preview", role="derived_preview")],
            ),
        ],
        edges=[
            ReductionDAGEdge(edge_id="e_raw_psd_to_preview", source_node_id="raw_psd_scan", target_node_id="assemble_psd_channel_temperature_preview"),
            ReductionDAGEdge(edge_id="e_eff_to_preview", source_node_id="psd_efficiency_correction", target_node_id="assemble_psd_channel_temperature_preview", role="correction_dependency"),
            ReductionDAGEdge(edge_id="e_mapping_to_preview", source_node_id="psd_channel_to_a4_mapping", target_node_id="assemble_psd_channel_temperature_preview", role="correction_dependency"),
        ],
        warnings=[
            "Channel x temperature preview does not claim Q x temperature reconstruction.",
            "Q conversion remains blocked until reviewed PSD channel-to-A4 calibration is accepted.",
        ],
        provenance=[ProvenanceRecord(source_type="template", source_id="psd_channel_temperature_preview", method="redactus_inspired_tas_native_template")],
    )


def constant_q_energy_scan_preview_template() -> ReductionDAGTemplate:
    """Template for safe inelastic constant-Q energy-scan previews."""
    return ReductionDAGTemplate(
        template_id="tas.constant_q_energy_scan_preview.v0",
        label="Constant-Q energy-scan preview",
        measurement_family="inelastic_constant_q_energy_scan",
        default_target_node_id="assemble_constant_q_energy_preview",
        nodes=[
            ReductionDAGNode(
                node_id="raw_energy_scan_series",
                kind="input",
                operation="input.energy_scan_series",
                label="Raw constant-Q energy scan series",
                input_ids=["energy_scan_series"],
                required_fields=["scan_ids", "q_values", "energy_values"],
                outputs=[ReductionDAGTerminal(terminal_id="energy_scan_series", data_type="scan_series", role="raw_input")],
            ),
            ReductionDAGNode(
                node_id="monitor_normalization",
                kind="correction",
                operation="correction.monitor_normalization",
                label="Monitor normalization correction",
                correction_ids=["monitor_normalization"],
                parameters={"optional_for_preview": True},
                outputs=[ReductionDAGTerminal(terminal_id="normalization", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="background_model",
                kind="correction",
                operation="correction.background_model",
                label="Background model or sideband choice",
                correction_ids=["background_model"],
                parameters={"optional_for_preview": True},
                outputs=[ReductionDAGTerminal(terminal_id="background", data_type="correction", role="correction")],
            ),
            ReductionDAGNode(
                node_id="assemble_constant_q_energy_preview",
                kind="operation",
                operation="preview.constant_q_energy_scan",
                label="Assemble constant-Q energy-scan preview",
                required_fields=["scan_ids", "q_values", "energy_values"],
                outputs=[ReductionDAGTerminal(terminal_id="preview", data_type="constant_q_energy_preview", role="derived_preview")],
            ),
        ],
        edges=[
            ReductionDAGEdge(edge_id="e_raw_energy_to_preview", source_node_id="raw_energy_scan_series", target_node_id="assemble_constant_q_energy_preview"),
            ReductionDAGEdge(edge_id="e_norm_energy_to_preview", source_node_id="monitor_normalization", target_node_id="assemble_constant_q_energy_preview", role="correction_dependency"),
            ReductionDAGEdge(edge_id="e_bg_energy_to_preview", source_node_id="background_model", target_node_id="assemble_constant_q_energy_preview", role="correction_dependency"),
        ],
        warnings=[
            "This template creates a preview/lineage plan, not a spin-wave fit or accepted dispersion.",
            "Background, monitor normalization, collimations, filters, and analyzer state must remain explicit DAG inputs.",
        ],
        provenance=[ProvenanceRecord(source_type="template", source_id="constant_q_energy_scan_preview", method="redactus_inspired_tas_native_template")],
    )


def reduction_cache_key(
    template: ReductionDAGTemplate,
    node: ReductionDAGNode,
    *,
    inputs: dict[str, Any],
    corrections: dict[str, ReductionCorrectionRef],
    upstream_cache_keys: dict[str, str] | None = None,
    software_versions: dict[str, str] | None = None,
    execution_fingerprint: dict[str, Any] | None = None,
    upstream_output_hashes: dict[str, str] | None = None,
) -> str:
    correction_ids = set(node.correction_ids)
    if node.tool and node.tool.correction_id:
        correction_ids.add(node.tool.correction_id)
    return stable_hash(
        {
            "executor_version": "1",
            "template_id": template.template_id,
            "node_id": node.node_id,
            "operation": node.operation,
            "node_version": node.version,
            "parameters": node.parameters,
            "tool": node.tool.model_dump(mode="json") if node.tool else None,
            "bindings": [edge.model_dump(mode="json") for edge in template.edges if edge.target_node_id == node.node_id],
            "input_refs": _selected_inputs(node, inputs),
            "corrections": {key: corrections[key].model_dump(mode="json") for key in sorted(correction_ids) if key in corrections},
            "upstream_cache_keys": upstream_cache_keys or {},
            "upstream_output_hashes": upstream_output_hashes or {},
            "software_versions": software_versions or {},
            "execution_fingerprint": execution_fingerprint or {},
        }
    )


def execute_reduction_dag(
    template: ReductionDAGTemplate,
    *,
    inputs: dict[str, Any],
    corrections: list[ReductionCorrectionRef] | None = None,
    target_node_id: str | None = None,
    handlers: dict[str, NodeHandler] | None = None,
    software_versions: dict[str, str] | None = None,
    runner: ToolRunner | None = None,
    cache_dir: str | Path | None = None,
) -> ReductionDAGRunResult:
    target = target_node_id or template.default_target_node_id
    correction_map = {correction.correction_id: correction for correction in corrections or []}
    handler_map = dict(_BUILTIN_HANDLERS)
    handler_map.update(handlers or {})
    ordered = _topological_order(template, target)
    node_by_id = {node.node_id: node for node in template.nodes}
    results: list[ReductionNodeResult] = []
    result_by_node: dict[str, ReductionNodeResult] = {}
    cache_keys: dict[str, str] = {}
    run_missing: list[str] = []
    run_warnings = list(template.warnings)
    cache = ReductionCache(cache_dir) if cache_dir is not None else None

    for node_id in ordered:
        node = node_by_id[node_id]
        upstream = _upstream_cache_keys(template, node_id, cache_keys)
        context = {
            "inputs": deepcopy(_selected_inputs(node, inputs) if node.tool else inputs),
            "corrections": deepcopy(correction_map),
            "upstream_results": {key: deepcopy(result_by_node[key].output) for key in _upstream_node_ids(template, node_id) if key in result_by_node},
            "template": template,
            "software_versions": software_versions or {},
        }
        missing = _missing_for_node(node, inputs, correction_map)
        failed_parents = [key for key in _upstream_node_ids(template, node_id) if result_by_node[key].status != "success"]
        call = None
        binding_error = None
        if node.tool and not missing and not failed_parents:
            try:
                call = bound_tool_call(node, context)
            except (KeyError, ValueError) as exc:
                binding_error = str(exc)
        fingerprint = tool_fingerprint(call, runner) if call else None
        if call is not None:
            call.provenance_context.extend(
                ProvenanceRecord(
                    source_type="data_file", source_id=path, method="input_content_fingerprint",
                    metadata={"sha256": digest},
                )
                for path, digest in fingerprint["source_files"].items()
            )
        cache_key = reduction_cache_key(
            template, node, inputs=inputs, corrections=correction_map,
            upstream_cache_keys=upstream, software_versions=software_versions,
            execution_fingerprint=fingerprint,
            upstream_output_hashes={key: stable_hash(value) for key, value in context["upstream_results"].items()},
        )
        cache_keys[node_id] = cache_key
        if failed_parents:
            result = ReductionNodeResult(
                node_id=node_id, status="blocked", cache_key=cache_key,
                warnings=[f"Upstream nodes did not succeed: {', '.join(failed_parents)}"],
                provenance=[ProvenanceRecord(source_type="reduction_node", source_id=node_id, method="upstream_failure")],
            )
        elif missing and not _node_allows_missing_corrections(node, missing):
            result = ReductionNodeResult(
                node_id=node_id,
                status="missing_input",
                cache_key=cache_key,
                missing_inputs=missing,
                warnings=["Node was not executed because required inputs/corrections are missing."],
                provenance=[ProvenanceRecord(source_type="reduction_node", source_id=node_id, method="validate_required_inputs")],
            )
        elif binding_error:
            result = ReductionNodeResult(
                node_id=node_id, status="missing_input", cache_key=cache_key,
                missing_inputs=[binding_error], warnings=["Numerical input binding failed; tool was not called."],
            )
        elif node.tool:
            if runner is None:
                result = ReductionNodeResult(
                    node_id=node_id, status="unavailable", cache_key=cache_key,
                    warnings=["Numerical DAG nodes require a ToolRunner."],
                )
            else:
                assert call is not None
                failure = runner.preflight(call)
                if failure is not None:
                    result = tool_node_result(node, cache_key, call, failure)
                else:
                    sources_hashed = all(value is not None for value in fingerprint["source_files"].values())
                    can_cache = cache is not None and node.cacheable and sources_hashed and cacheable_tool_call(call, runner)
                    result = cache.load(cache_key, node_id) if can_cache else None
                    if result is not None:
                        audit_cache_hit(result, call, runner)
                    else:
                        result = tool_node_result(node, cache_key, deepcopy(call), runner.run(call))
                        if can_cache and not result.tool_result.derived_artifacts:
                            cache.save(result)
        elif node.operation not in handler_map:
            result = ReductionNodeResult(
                node_id=node_id,
                status="unavailable",
                cache_key=cache_key,
                missing_inputs=[f"handler:{node.operation}"],
                warnings=["No handler is registered for this reduction operation."],
                provenance=[ProvenanceRecord(source_type="reduction_node", source_id=node_id, method="missing_handler")],
            )
        else:
            try:
                output = handler_map[node.operation](node, context)
                status = output.pop("status", "success")
                result = ReductionNodeResult(
                    node_id=node_id,
                    status=status,
                    cache_key=cache_key,
                    output=output,
                    missing_inputs=output.get("missing_inputs", []),
                    warnings=output.get("warnings", []),
                    provenance=[ProvenanceRecord(source_type="reduction_node", source_id=node_id, method=node.operation)],
                )
            except Exception as exc:
                result = ReductionNodeResult(
                    node_id=node_id,
                    status="error",
                    cache_key=cache_key,
                    missing_inputs=[],
                    warnings=[f"{exc.__class__.__name__}: {exc}"],
                    provenance=[ProvenanceRecord(source_type="reduction_node", source_id=node_id, method="handler_exception")],
                )
        results.append(result)
        result_by_node[node_id] = result
        run_missing.extend(result.missing_inputs)
        run_warnings.extend(result.warnings)

    status = _run_status(results)
    raw_inputs = [_input_ref(key, value) for key, value in sorted(inputs.items())]
    return ReductionDAGRunResult(
        run_id=f"reduction-run-{uuid4()}",
        template_id=template.template_id,
        status=status,
        target_node_id=target,
        ordered_node_ids=ordered,
        node_results=results,
        cache_keys=cache_keys,
        raw_inputs=raw_inputs,
        corrections=list(correction_map.values()),
        derived_artifacts=_derived_artifacts(results),
        missing_inputs=sorted(set(run_missing)),
        warnings=sorted(set(run_warnings)),
        provenance=[ProvenanceRecord(source_type="reduction_template", source_id=template.template_id, method="execute_reduction_dag")],
    )


def replay_reduction_dag_run(
    prior_run: dict[str, Any],
    *,
    template: ReductionDAGTemplate,
    inputs: dict[str, Any],
    corrections: list[ReductionCorrectionRef] | None = None,
    handlers: dict[str, NodeHandler] | None = None,
    software_versions: dict[str, str] | None = None,
    runner: ToolRunner | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-execute a DAG and report which node cache keys changed.

    Numerical nodes reuse successful materialized results when cache_dir is
    supplied. Changed dependencies execute again through the tool registry.
    """

    new_result = execute_reduction_dag(
        template,
        inputs=inputs,
        corrections=corrections,
        handlers=handlers,
        software_versions=software_versions,
        runner=runner,
        cache_dir=cache_dir,
        target_node_id=prior_run.get("target_node_id"),
    )
    old_keys = dict(prior_run.get("cache_keys") or {})
    new_keys = dict(new_result.cache_keys)
    changed = [node_id for node_id in new_result.ordered_node_ids if old_keys.get(node_id) != new_keys.get(node_id)]
    reusable = [node_id for node_id in new_result.ordered_node_ids if old_keys.get(node_id) == new_keys.get(node_id)]
    return {
        "status": new_result.status,
        "prior_run_id": prior_run.get("run_id"),
        "new_run_id": new_result.run_id,
        "changed_node_ids": changed,
        "reusable_node_ids": reusable,
        "changed_count": len(changed),
        "reusable_count": len(reusable),
        "cache_hit_node_ids": [result.node_id for result in new_result.node_results if result.cache_hit],
        "prior_cache_keys": old_keys,
        "new_cache_keys": new_keys,
        "new_result": new_result.model_dump(mode="json"),
    }


def _selected_inputs(node: ReductionDAGNode, inputs: dict[str, Any]) -> dict[str, Any]:
    if node.tool:
        selected = set(node.input_ids) | set(node.tool.input_bindings.values())
        return {key: inputs.get(key) for key in sorted(selected)}
    if node.input_ids:
        return {key: inputs.get(key) for key in node.input_ids}
    return {key: inputs.get(key) for key in sorted(inputs)}


def _topological_order(template: ReductionDAGTemplate, target_node_id: str | None = None) -> list[str]:
    nodes = {node.node_id for node in template.nodes}
    if target_node_id and target_node_id not in nodes:
        raise ValueError(f"unknown target node: {target_node_id}")
    needed = _upstream_closure(template, target_node_id) if target_node_id else nodes
    incoming_count: dict[str, int] = {node_id: 0 for node_id in needed}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in template.edges:
        if edge.source_node_id in needed and edge.target_node_id in needed:
            outgoing[edge.source_node_id].append(edge.target_node_id)
            incoming_count[edge.target_node_id] += 1
    queue = deque([node.node_id for node in template.nodes if node.node_id in needed and incoming_count[node.node_id] == 0])
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for child in outgoing[node_id]:
            incoming_count[child] -= 1
            if incoming_count[child] == 0:
                queue.append(child)
    if len(ordered) != len(needed):
        raise ValueError("reduction DAG contains a cycle or dangling edge")
    return ordered


def _upstream_closure(template: ReductionDAGTemplate, target_node_id: str | None) -> set[str]:
    if target_node_id is None:
        return {node.node_id for node in template.nodes}
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in template.edges:
        parents[edge.target_node_id].append(edge.source_node_id)
    needed = {target_node_id}
    stack = [target_node_id]
    while stack:
        current = stack.pop()
        for parent in parents[current]:
            if parent not in needed:
                needed.add(parent)
                stack.append(parent)
    return needed


def _upstream_node_ids(template: ReductionDAGTemplate, node_id: str) -> list[str]:
    return [edge.source_node_id for edge in template.edges if edge.target_node_id == node_id]


def _upstream_cache_keys(template: ReductionDAGTemplate, node_id: str, cache_keys: dict[str, str]) -> dict[str, str]:
    return {upstream: cache_keys[upstream] for upstream in _upstream_node_ids(template, node_id) if upstream in cache_keys}


def _missing_for_node(node: ReductionDAGNode, inputs: dict[str, Any], corrections: dict[str, ReductionCorrectionRef]) -> list[str]:
    missing: list[str] = []
    input_ids = set(node.input_ids) | (set(node.tool.input_bindings.values()) if node.tool else set())
    for input_id in sorted(input_ids):
        if input_id not in inputs:
            missing.append(f"input:{input_id}")
            continue
        value = inputs[input_id]
        if isinstance(value, dict):
            for field in node.required_fields:
                if field not in value or value[field] in (None, [], {}):
                    missing.append(f"input:{input_id}.{field}")
    correction_ids = set(node.correction_ids)
    if node.tool and node.tool.correction_id:
        correction_ids.add(node.tool.correction_id)
    for correction_id in sorted(correction_ids):
        correction = corrections.get(correction_id)
        if correction is None or correction.status in {"missing", "rejected"}:
            missing.append(f"correction:{correction_id}")
        elif node.tool and correction.status not in {"reviewed", "accepted"}:
            missing.append(f"reviewed_correction:{correction_id}")
    return missing


def _node_allows_missing_corrections(node: ReductionDAGNode, missing: list[str]) -> bool:
    if node.tool:
        return False
    if not missing:
        return True
    if not all(item.startswith("correction:") for item in missing):
        return False
    return bool(node.parameters.get("optional_for_preview") or node.parameters.get("optional_for_channel_temperature"))


def _input_ref(input_id: str, value: Any) -> ReductionInputRef:
    if isinstance(value, ReductionInputRef):
        return value
    if isinstance(value, dict):
        return ReductionInputRef(
            input_id=input_id,
            source_object_type=str(value.get("source_object_type", "fixture_input")),
            source_object_id=str(value.get("source_object_id", input_id)),
            path=value.get("path"),
            content_hash=value.get("content_hash") or stable_hash(value),
            metadata={key: value for key, value in value.items() if key not in {"path", "content_hash"}},
            provenance=[ProvenanceRecord(source_type="reduction_input", source_id=input_id, method="input_ref")],
        )
    return ReductionInputRef(
        input_id=input_id,
        source_object_type="literal",
        source_object_id=input_id,
        content_hash=stable_hash(value),
        metadata={"value_type": type(value).__name__},
        provenance=[ProvenanceRecord(source_type="reduction_input", source_id=input_id, method="input_ref")],
    )


def _run_status(results: list[ReductionNodeResult]) -> str:
    statuses = {result.status for result in results}
    if "error" in statuses:
        return "error"
    if "missing_input" in statuses:
        return "missing_input"
    if "unsupported" in statuses:
        return "unsupported"
    if "unavailable" in statuses:
        return "unavailable"
    if "validation_failed" in statuses:
        return "validation_failed"
    if "blocked" in statuses:
        return "blocked"
    return "success"


def _derived_artifacts(results: list[ReductionNodeResult]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for result in results:
        if result.tool_result:
            artifacts.extend(result.tool_result.derived_artifacts)
        artifact = result.output.get("derived_artifact")
        if artifact:
            artifacts.append(artifact)
    return artifacts


def _input_passthrough(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    payload = {input_id: context["inputs"][input_id] for input_id in node.input_ids if input_id in context["inputs"]}
    return {"status": "success", "payload": payload}


def _correction_record(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    corrections = context["corrections"]
    attached = {correction_id: corrections[correction_id].model_dump(mode="json") for correction_id in node.correction_ids if correction_id in corrections}
    missing = [correction_id for correction_id in node.correction_ids if correction_id not in corrections]
    warnings = []
    if missing and _node_allows_missing_corrections(node, [f"correction:{item}" for item in missing]):
        warnings.append("Optional correction is missing; preview remains uncorrected or partially corrected.")
    return {"status": "success", "corrections": attached, "missing_optional_corrections": missing, "warnings": warnings}


def _order_parameter_preview(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    scan_series = context["inputs"].get("scan_series", {})
    scan_ids = list(scan_series.get("scan_ids", []))
    temperatures = list(scan_series.get("temperature_values", []))
    return {
        "status": "success",
        "scan_count": len(scan_ids),
        "temperature_point_count": len(temperatures),
        "temperature_range": [min(temperatures), max(temperatures)] if temperatures else [],
        "preview_type": "order_parameter_temperature_series",
        "derived_artifact": {
            "artifact_id": f"artifact-{stable_hash({'node': node.node_id, 'scan_ids': scan_ids})[:12]}",
            "artifact_type": "derived_preview",
            "lineage": {"scan_ids": scan_ids, "corrections_are_separate_inputs": True},
        },
        "warnings": ["No peak model, background model, or transition model has been accepted by this preview node."],
    }


def _order_parameter_model_branch(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    upstream = context.get("upstream_results") or {}
    preview = upstream.get("assemble_order_parameter_preview") or {}
    parameters = dict(node.parameters or {})
    fit_evidence = dict((context.get("inputs") or {}).get("fit_evidence") or {})
    model_branch = str(parameters.get("model_branch") or fit_evidence.get("model_branch") or "unknown")
    bic_tag = str(parameters.get("bic_tag") or fit_evidence.get("bic_tag") or "not_computed")
    warnings = [
        "Model branch is lineage for review/replay; it is not an accepted order-parameter result.",
        "BIC tags are only meaningful when alternate branches were evaluated with comparable inputs.",
    ]
    return {
        "status": "success",
        "model_branch": model_branch,
        "bic_tag": bic_tag,
        "fit_evidence": fit_evidence,
        "upstream_preview": {
            "preview_type": preview.get("preview_type"),
            "scan_count": preview.get("scan_count"),
            "temperature_range": preview.get("temperature_range"),
        },
        "derived_artifact": {
            "artifact_id": f"artifact-{stable_hash({'node': node.node_id, 'model_branch': model_branch, 'bic_tag': bic_tag, 'fit': fit_evidence})[:12]}",
            "artifact_type": "model_branch",
            "lineage": {"model_branch": model_branch, "bic_tag": bic_tag, "upstream_preview_node": "assemble_order_parameter_preview"},
        },
        "warnings": warnings,
    }


def _psd_channel_temperature_preview(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    psd_scan = context["inputs"].get("psd_scan", {})
    temperatures = list(psd_scan.get("temperature_values", []))
    counts = psd_scan.get("psd_counts", [])
    if not isinstance(counts, list) or not counts:
        return {"status": "missing_input", "missing_inputs": ["input:psd_scan.psd_counts"]}
    channel_count = len(counts[0]) if isinstance(counts[0], list) else 0
    row_lengths = sorted({len(row) for row in counts if isinstance(row, list)})
    warnings = ["This preview is channel x temperature only; it is not a Q x temperature map."]
    if len(row_lengths) != 1:
        warnings.append("PSD rows have inconsistent channel counts.")
    correction_status = {key: value.status for key, value in context["corrections"].items() if key.startswith("psd_")}
    return {
        "status": "success",
        "preview_type": "psd_channel_temperature",
        "temperature_point_count": len(temperatures),
        "temperature_range": [min(temperatures), max(temperatures)] if temperatures else [],
        "psd_channel_count": channel_count,
        "shape": [len(counts), channel_count],
        "correction_status": correction_status,
        "q_temperature_status": "blocked_without_reviewed_channel_to_a4_calibration",
        "derived_artifact": {
            "artifact_id": f"artifact-{stable_hash({'node': node.node_id, 'source': psd_scan.get('source_object_id')})[:12]}",
            "artifact_type": "derived_preview",
            "lineage": {"source_object_id": psd_scan.get("source_object_id"), "corrections_are_separate_inputs": True},
        },
        "warnings": warnings,
    }


def _constant_q_energy_preview(node: ReductionDAGNode, context: dict[str, Any]) -> dict[str, Any]:
    series = context["inputs"].get("energy_scan_series", {})
    scan_ids = list(series.get("scan_ids", []))
    q_values = list(series.get("q_values", []))
    energy_values = list(series.get("energy_values", []))
    warnings = [
        "This preview groups constant-Q energy-scan intent only; it does not fit a dispersion.",
        "Background, filter/analyzer state, and collimations are not accepted by this preview node.",
    ]
    return {
        "status": "success",
        "scan_count": len(scan_ids),
        "q_point_count": len(q_values),
        "energy_point_count": len(energy_values),
        "energy_range": [min(energy_values), max(energy_values)] if energy_values else [],
        "preview_type": "constant_q_energy_scan",
        "fit_status": "blocked_until_background_resolution_and_model_are_reviewed",
        "derived_artifact": {
            "artifact_id": f"artifact-{stable_hash({'node': node.node_id, 'scan_ids': scan_ids, 'energy_values': energy_values})[:12]}",
            "artifact_type": "derived_preview",
            "lineage": {"scan_ids": scan_ids, "corrections_are_separate_inputs": True},
        },
        "warnings": warnings,
    }


_BUILTIN_HANDLERS: dict[str, NodeHandler] = {
    "input.scan_series": _input_passthrough,
    "input.psd_temperature_scan": _input_passthrough,
    "input.energy_scan_series": _input_passthrough,
    "correction.monitor_normalization": _correction_record,
    "correction.background_model": _correction_record,
    "correction.psd_efficiency": _correction_record,
    "correction.psd_channel_to_a4": _correction_record,
    "preview.order_parameter_temperature_series": _order_parameter_preview,
    "model_branch.order_parameter_fit": _order_parameter_model_branch,
    "preview.psd_channel_temperature": _psd_channel_temperature_preview,
    "preview.constant_q_energy_scan": _constant_q_energy_preview,
}
