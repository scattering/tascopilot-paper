from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, ReductionCorrectionRef, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.reduction_cache import write_json_atomic
from tas_copilot_core.reduction_dag import execute_reduction_dag
from tas_copilot_core.reduction_pipeline import SUPPORTED_STEPS, scan_reduction_template
from tas_copilot_core.registry.tool_runner import stable_hash

from .common import default_runner


WORKFLOW_NAME = "reduce-scan-set"


def reduce_scan_set(arguments: dict, runner=None, audit_logger: JsonlAuditLogger | None = None) -> dict:
    audit_logger = audit_logger or getattr(runner, "audit_logger", None) or JsonlAuditLogger()
    validation_failures = _validate(arguments)
    if not validation_failures:
        try:
            template, inputs = scan_reduction_template(arguments)
            corrections = [ReductionCorrectionRef.model_validate(value) for value in arguments.get("corrections", [])]
        except (ValueError, TypeError, ValidationError) as exc:
            validation_failures.append({"code": "invalid_reduction_graph", "message": str(exc)})
    if validation_failures:
        report = _report(
            status=ToolResultStatus.missing_input.value,
            arguments=arguments,
            results={},
            validation_failures=validation_failures,
        )
        report["workflow_audit_id"] = _write_workflow_audit(audit_logger, report, arguments)
        return report

    runner = runner or default_runner()
    lineage = _lineage(arguments)
    versions = {"reduction_executor": "1", **arguments.get("software_versions", {})}
    dag = execute_reduction_dag(
        template, inputs=inputs, corrections=corrections, runner=runner,
        cache_dir=arguments.get("cache_dir"), software_versions=versions,
    )
    results = {result.node_id: result.tool_result for result in dag.node_results if result.tool_result is not None}
    report = _report(
        status=dag.status,
        arguments=arguments,
        results=results,
        validation_failures=[],
        lineage=lineage,
    )
    report["reduction_dag"] = dag.model_dump(mode="json")
    report["replay"] = {
        "template": template.model_dump(mode="json"), "inputs": inputs,
        "corrections": [correction.model_dump(mode="json") for correction in corrections],
        "software_versions": versions,
    }
    if dag.status == "success":
        report["output"] = dag.node_results[-1].output
        if lineage["output_path"]:
            artifact_path = Path(lineage["output_path"]).expanduser()
            write_json_atomic(artifact_path, {
                "output": report["output"], "lineage": lineage,
                "reduction_dag": report["reduction_dag"], "replay": report["replay"],
            })
            report["artifact"] = str(artifact_path)
    report["workflow_audit_id"] = _write_workflow_audit(getattr(runner, "audit_logger", audit_logger), report, arguments)
    return report


def _validate(arguments: dict) -> list[dict]:
    failures = []
    steps = arguments.get("steps")
    if not isinstance(steps, list) or not steps:
        failures.append({"code": "missing_steps", "message": "steps is required"})
        return failures
    for index, step in enumerate(steps):
        step_name = step.get("tool") if isinstance(step, dict) else str(step)
        if step_name not in SUPPORTED_STEPS:
            failures.append({"code": "unsupported_step", "message": f"Unsupported reduction step: {step_name}", "step_index": index})
    if not (arguments.get("input_datafile_ids") or arguments.get("input_files") or arguments.get("source_datafile_ids")):
        failures.append({"code": "missing_input_lineage", "message": "input_datafile_ids, input_files, or source_datafile_ids is required"})
    if not (arguments.get("output_path") or arguments.get("derived_output_path") or arguments.get("derived_artifact_id") or arguments.get("in_memory") is True):
        failures.append({"code": "missing_output_artifact", "message": "output_path, derived_output_path, or derived_artifact_id is required"})
    output = arguments.get("output_path") or arguments.get("derived_output_path")
    if output:
        sources = list(arguments.get("input_files") or [])
        for item in [arguments, *(step.get("arguments", {}) for step in steps if isinstance(step, dict))]:
            sources.extend(item[key] for key in ("path", "file_path") if item.get(key))
        if Path(output).expanduser().resolve() in {Path(source).expanduser().resolve() for source in sources}:
            failures.append({"code": "raw_input_overwrite", "message": "Derived output must not overwrite a source file."})
    return failures


def _lineage(arguments: dict) -> dict:
    input_ids = arguments.get("input_datafile_ids") or arguments.get("input_files") or arguments.get("source_datafile_ids") or []
    output_path = arguments.get("output_path") or arguments.get("derived_output_path")
    derived_artifact_id = arguments.get("derived_artifact_id") or output_path
    return {
        "input_datafile_ids": list(input_ids),
        "output_path": output_path,
        "derived_artifact_id": derived_artifact_id,
        "lineage": {"input_datafile_ids": list(input_ids), "derived_artifact_id": derived_artifact_id},
    }


def _report(status: str, arguments: dict, results: dict, validation_failures: list[dict], lineage: dict | None = None) -> dict:
    return {
        "workflow_name": WORKFLOW_NAME,
        "status": status,
        "results": {name: result.model_dump(mode="json") for name, result in results.items()},
        "tool_audit_ids": {name: result.audit_id for name, result in results.items()},
        "validation_failures": validation_failures,
        "lineage": lineage or {},
        "warnings": ["Scientific reduction internals remain owned by external plugins."],
        "provenance": [ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="registry_tool_calls").model_dump(mode="json")],
    }


def _write_workflow_audit(audit_logger, report: dict, arguments: dict) -> str | None:
    if audit_logger is None:
        return None
    audit = AuditRecord(
        event_type="workflow",
        status=report["status"],
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": arguments}),
        output_hash=stable_hash(report),
        details={
            "workflow_name": WORKFLOW_NAME,
            "input_arguments": arguments,
            "tools_selected": [f"{namespace}.{name}" for namespace, name in SUPPORTED_STEPS.values()],
            "tool_audit_ids": report.get("tool_audit_ids", {}),
            "validation_failures": report.get("validation_failures", []),
            "lineage": report.get("lineage", {}),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id
