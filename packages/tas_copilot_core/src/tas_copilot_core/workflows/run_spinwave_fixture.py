from __future__ import annotations

from pydantic import ValidationError

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, SpinWaveCalculationInput, ToolCall, ToolError, ToolResult, ToolResultStatus

from .common import default_runner


WORKFLOW_NAME = "run-spinwave-fixture"


def run_spinwave_fixture(arguments: dict, runner=None) -> ToolResult:
    try:
        request = _request_from_arguments(arguments)
    except ValidationError as exc:
        return ToolResult(
            status=ToolResultStatus.validation_failed,
            errors=[
                ToolError(
                    code="invalid_spinwave_input",
                    message="arguments must match SpinWaveCalculationInput",
                    details={"validation_errors": exc.errors()},
                )
            ],
            provenance=[ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="input_validation")],
            tool_name="run_spinwave",
            tool_namespace="tas.pyspinw",
        )

    runner = runner or default_runner()
    call = ToolCall(
        tool_namespace="tas.pyspinw",
        tool_name="run_spinwave",
        arguments={
            "model": request.model.model_dump(mode="json", exclude_none=True),
            "calculation": {
                "q_path": request.q_path,
                "n_points_per_segment": request.n_points_per_segment,
                "use_rust": request.use_rust,
            },
        },
        caller_workflow=WORKFLOW_NAME,
    )
    result = runner.run(call)
    result.provenance.append(ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="registry_tool_call"))
    workflow_audit_id = _write_workflow_audit(runner, call, result)
    result.runtime_environment["workflow_audit_id"] = workflow_audit_id
    return result


def _request_from_arguments(arguments: dict) -> SpinWaveCalculationInput:
    calculation = arguments.get("calculation")
    if calculation is None:
        calculation = {
            key: arguments[key]
            for key in ["q_path", "n_points_per_segment", "use_rust"]
            if key in arguments
        }
    return SpinWaveCalculationInput(model=arguments.get("model"), **calculation)


def _write_workflow_audit(runner, call: ToolCall, result: ToolResult) -> str | None:
    audit_logger = getattr(runner, "audit_logger", None)
    if audit_logger is None:
        return None
    audit = AuditRecord(
        event_type="workflow",
        request_id=call.request_id,
        status=result.status.value,
        input_hash=result.input_hash,
        output_hash=result.output_hash,
        tool_namespace=call.tool_namespace,
        tool_name=call.tool_name,
        details={
            "workflow_name": WORKFLOW_NAME,
            "tool_audit_id": result.audit_id,
            "simulation_only": True,
        },
    )
    audit_logger.write(audit)
    return audit.audit_id
