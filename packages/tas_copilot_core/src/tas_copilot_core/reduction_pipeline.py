"""Build executable scan-reduction graphs from the existing workflow steps."""

from __future__ import annotations

from copy import deepcopy

from tas_copilot_contracts import (
    ReductionDAGEdge, ReductionDAGNode, ReductionDAGTemplate,
    ReductionDAGTerminal, ReductionToolBinding,
)


SUPPORTED_STEPS = {
    "add_scans": ("tas.file_arithmetic", "add_scans"),
    "subtract_background": ("tas.file_arithmetic", "subtract_background"),
    "normalize_to_monitor": ("tas.corrections", "normalize_to_monitor"),
    "apply_detailed_balance": ("tas.corrections", "apply_detailed_balance"),
    "translate_ice_file": ("tas.reduction", "translate_ice_file"),
}
_CHAIN_INPUT = {
    "subtract_background": "signal",
    "normalize_to_monitor": "scan",
    "apply_detailed_balance": "scan",
}
_RAW_ARGUMENTS = {"scan", "scans", "signal", "background", "path", "file_path"}
_WORKFLOW_ARGUMENTS = {
    "steps", "cache_dir", "software_versions", "corrections", "output_path",
    "derived_output_path", "input_files", "lineage", "source_datafile_ids",
}


def scan_reduction_template(arguments: dict) -> tuple[ReductionDAGTemplate, dict]:
    """Bind each returned scan to the following correction's numerical input."""
    nodes = []
    edges = []
    inputs = {}
    lineage_ids = arguments.get("input_datafile_ids") or arguments.get("input_files") or arguments.get("source_datafile_ids") or []
    previous = None
    for index, step in enumerate(arguments["steps"]):
        name = step["tool"] if isinstance(step, dict) else str(step)
        namespace, tool_name = SUPPORTED_STEPS[name]
        node_id = f"step_{index}_{name}"
        step_args = {key: deepcopy(value) for key, value in arguments.items() if key not in _WORKFLOW_ARGUMENTS}
        for key in _RAW_ARGUMENTS:
            if key == "background":
                if name != "subtract_background":
                    step_args.pop(key, None)
            elif index:
                step_args.pop(key, None)
        if isinstance(step, dict):
            step_args.update(deepcopy(step.get("arguments", {})))
        # Numerical nodes are in-memory. The workflow materializes only its
        # final derived artifact, even when intermediate results are cached.
        step_args.pop("output_path", None)
        step_args.pop("derived_output_path", None)
        step_args["in_memory"] = True
        ids = list(dict.fromkeys([*lineage_ids, *(step_args.get("input_datafile_ids") or [])]))
        step_args["input_datafile_ids"] = ids
        step_args["lineage"] = {"input_datafile_ids": ids}
        chain_input = _CHAIN_INPUT.get(name) if previous else None
        if chain_input:
            if isinstance(step, dict) and chain_input in step.get("arguments", {}):
                raise ValueError(f"{node_id}: {chain_input} comes from the preceding step; use a separate graph for independent scans")
            step_args.pop(chain_input, None)
        bindings = {}
        for key in sorted(_RAW_ARGUMENTS & step_args.keys()):
            input_id = f"{node_id}.{key}"
            inputs[input_id] = step_args.pop(key)
            bindings[key] = input_id
        correction_id = step.get("correction_id") if isinstance(step, dict) else None
        nodes.append(ReductionDAGNode(
            node_id=node_id, kind="operation", operation="registry.tool", version="1",
            input_ids=list(bindings.values()), correction_ids=[correction_id] if correction_id else [],
            tool=ReductionToolBinding(
                tool_namespace=namespace, tool_name=tool_name, arguments=step_args,
                input_bindings=bindings, correction_id=correction_id,
            ),
            outputs=[ReductionDAGTerminal(terminal_id="scan", data_type="scan", role="derived_result")],
        ))
        if previous:
            edges.append(ReductionDAGEdge(
                edge_id=f"{previous}_to_{node_id}", source_node_id=previous, target_node_id=node_id,
                source_terminal_id="scan" if chain_input else None, target_input_name=chain_input,
            ))
        previous = node_id
    return ReductionDAGTemplate(
        template_id="tas.scan_reduction.v1", label="Registry-backed scan reduction",
        measurement_family="scan_reduction", nodes=nodes, edges=edges,
        default_target_node_id=previous,
    ), inputs
