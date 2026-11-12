from __future__ import annotations

from typing import Any

from tas_copilot_contracts import MaterialIdentityRecord, ProvenanceRecord, ToolCall, ToolResultStatus


def material_identity_record_via_registry(
    text: str,
    *,
    family: str = "lu_femn_o3",
    source_field: str = "text",
    runner: Any | None = None,
    caller_workflow: str = "material-identity",
) -> MaterialIdentityRecord:
    """Resolve material identity through the registry-facing tool boundary.

    Workflows should use this adapter when material identity is a tool
    invocation. The parser implementation remains in material_identity.py, but
    this keeps orchestration paths auditable and fail-closed if the tool is not
    registered.
    """

    raw = str(text or "").strip()
    if not raw:
        return _unavailable_record(raw, source_field, "No material text was supplied.")
    try:
        runner = runner or _default_runner()
        result = runner.run(
            ToolCall(
                tool_namespace="tas.material_identity",
                tool_name="parse_filename_composition",
                arguments={"text": raw, "family": family, "source_field": source_field},
                caller_workflow=caller_workflow,
            )
        )
    except Exception as exc:
        return _unavailable_record(raw, source_field, f"Material identity registry call failed closed: {type(exc).__name__}: {exc}")

    if result.status != ToolResultStatus.success or not isinstance(result.output, dict):
        reason = "; ".join(error.message for error in result.errors) if getattr(result, "errors", None) else f"Tool returned {result.status.value}."
        return _unavailable_record(raw, source_field, reason)
    payload = result.output.get("material_identity")
    if not isinstance(payload, dict):
        return _unavailable_record(raw, source_field, "Material identity tool did not return a material_identity record.")
    record = MaterialIdentityRecord.model_validate(payload)
    if result.audit_id:
        record.provenance.append(
            ProvenanceRecord(
                source_type="tool_result",
                source_id=result.audit_id,
                method="registry_material_identity_tool",
                confidence=record.confidence,
            )
        )
    return record


def _default_runner():
    from tas_copilot_core.workflows.common import default_runner

    return default_runner()


def _unavailable_record(raw: str, source_field: str, warning: str) -> MaterialIdentityRecord:
    return MaterialIdentityRecord(
        material_id="material-unavailable",
        raw_string=raw,
        source_field=source_field,
        parser_backend="unavailable",
        interpretation_method="registry_tool_unavailable",
        confidence=0.0,
        warnings=[warning],
        provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="registry_material_identity_unavailable")],
    )
