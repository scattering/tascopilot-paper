"""Registry boundary for numerical reduction DAG nodes; no scientific kernels."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

from tas_copilot_contracts import (
    AuditRecord, ProvenanceRecord, ReductionDAGNode, ReductionDAGTemplate,
    ReductionNodeResult, SideEffectLevel, ToolCall, ToolResult,
)
from tas_copilot_core.registry import ToolRunner


def bound_tool_call(node: ReductionDAGNode, context: dict[str, Any]) -> ToolCall:
    binding = node.tool
    assert binding is not None
    arguments = deepcopy(binding.arguments)
    if binding.correction_id:
        correction = context["corrections"].get(binding.correction_id)
        if correction is None or correction.status not in {"reviewed", "accepted"}:
            raise ValueError(f"reviewed correction required: {binding.correction_id}")
        arguments.update(deepcopy(correction.parameters))
    for argument, input_id in binding.input_bindings.items():
        if input_id not in context["inputs"]:
            raise ValueError(f"missing input: {input_id}")
        arguments[argument] = deepcopy(context["inputs"][input_id])
    template: ReductionDAGTemplate = context["template"]
    for edge in template.edges:
        if edge.target_node_id != node.node_id or not edge.target_input_name:
            continue
        output = context["upstream_results"][edge.source_node_id]
        if edge.source_terminal_id:
            if edge.source_terminal_id not in output:
                raise ValueError(f"missing upstream output: {edge.source_node_id}.{edge.source_terminal_id}")
            output = output[edge.source_terminal_id]
        arguments[edge.target_input_name] = deepcopy(output)
    return ToolCall(
        tool_namespace=binding.tool_namespace,
        tool_name=binding.tool_name,
        arguments=arguments,
        caller_workflow="reduction-dag",
        provenance_context=[ProvenanceRecord(
            source_type="reduction_node", source_id=node.node_id, method=template.template_id,
        )],
    )


def source_file_fingerprints(arguments: dict) -> dict[str, str | None]:
    """Hash actual file contents, not just paths or modification times."""
    fingerprints: dict[str, str | None] = {}
    for key in ("path", "file_path", "input_files"):
        value = arguments.get(key)
        paths = value if isinstance(value, list) else [value] if value else []
        for name in paths:
            path = Path(name).expanduser()
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                fingerprints[str(path)] = digest.hexdigest()
            except OSError:
                fingerprints[str(path)] = None
    return fingerprints


def tool_fingerprint(call: ToolCall, runner: ToolRunner | None) -> dict:
    try:
        spec = runner.registry.get(call.tool_namespace, call.tool_name).spec if runner else None
    except KeyError:
        spec = None
    return {
        "arguments": call.arguments,
        "source_files": source_file_fingerprints(call.arguments),
        "tool_spec": spec.model_dump(mode="json") if spec else None,
    }


def cacheable_tool_call(call: ToolCall, runner: ToolRunner) -> bool:
    spec = runner.registry.get(call.tool_namespace, call.tool_name).spec
    if any(call.arguments.get(key) for key in ("output_path", "derived_output_path", "output_dir", "output_files")):
        return False
    return spec.side_effect_level == SideEffectLevel.none or (
        spec.side_effect_level == SideEffectLevel.local_files and call.arguments.get("in_memory") is True
    )


def tool_node_result(node: ReductionDAGNode, key: str, call: ToolCall, result: ToolResult) -> ReductionNodeResult:
    output = deepcopy(result.output or {})
    # The plugin's ToolResult is kept intact; descendants also need a stable ID
    # for the derived scan because some plugins return only numerical arrays.
    if result.status == "success" and isinstance(output.get("scan"), dict):
        output["scan"].setdefault("scan_id", f"reduction:{key}")
    return ReductionNodeResult(
        node_id=node.node_id, status=result.status.value, cache_key=key, output=output,
        warnings=[*result.warnings, *(f"{error.code}: {error.message}" for error in result.errors)],
        missing_inputs=[error.message for error in result.errors] if result.status == "missing_input" else [],
        provenance=[*call.provenance_context, *result.provenance],
        tool_call=call, tool_result=result,
    )


def audit_cache_hit(result: ReductionNodeResult, call: ToolCall, runner: ToolRunner) -> None:
    audit = AuditRecord(
        event_type="reduction_cache_hit", status="success", request_id=call.request_id,
        tool_namespace=call.tool_namespace, tool_name=call.tool_name,
        input_hash=result.cache_key,
        output_hash=result.tool_result.output_hash if result.tool_result else None,
        details={"node_id": result.node_id, "source_tool_audit_id": result.tool_result.audit_id if result.tool_result else None},
    )
    runner.audit_logger.write(audit)
    result.cache_hit = True
    result.cache_audit_id = audit.audit_id
