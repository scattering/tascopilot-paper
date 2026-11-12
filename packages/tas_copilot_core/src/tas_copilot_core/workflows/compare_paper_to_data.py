from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, ToolCall, ToolError, ToolResult, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger

from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.registry.tool_runner import stable_hash

from .common import default_runner


WORKFLOW_NAME = "compare-paper-to-data"


def compare_paper_to_data(
    paper_id: str,
    arguments: dict | None = None,
    runner=None,
    audit_logger: JsonlAuditLogger | None = None,
) -> ToolResult:
    audit_logger = audit_logger or getattr(runner, "audit_logger", None) or JsonlAuditLogger()
    payload = dict(arguments or {})
    paper = dict(payload.get("paper") or {})
    paper.setdefault("paper_id", paper_id)
    payload["paper"] = paper

    validation_failures = []
    if "experiments" not in payload and payload.get("db_path"):
        db_path = Path(payload["db_path"])
        if db_path.exists():
            catalog = SQLiteDataCatalog(db_path)
            payload["experiments"] = catalog.mapping_candidate_experiments(limit=payload.get("candidate_limit"))
        else:
            validation_failures.append(
                {
                    "code": "catalog_missing",
                    "message": f"Experiment catalog does not exist: {db_path}",
                    "db_path": str(db_path),
                }
            )
    if "experiments" not in payload:
        validation_failures.append(
            {
                "code": "missing_experiment_candidates",
                "message": "Provide experiments directly or pass db_path for the TAS data-file catalog.",
            }
        )
    if validation_failures:
        result = ToolResult(
            status=ToolResultStatus.missing_input,
            output={"paper_id": paper_id, "validation_failures": validation_failures},
            errors=[
                ToolError(
                    code=item["code"],
                    message=item["message"],
                    details={key: value for key, value in item.items() if key not in {"code", "message"}},
                )
                for item in validation_failures
            ],
            warnings=["No paper-to-experiment mapping was attempted because required catalog inputs are missing."],
            provenance=[ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="input_validation")],
            tool_name="map_paper_to_experiments",
            tool_namespace="tas.paper_mapping",
            input_hash=stable_hash({"paper_id": paper_id, "arguments": payload}),
        )
        result.output_hash = stable_hash(result.output)
        _attach_workflow_audit(result, payload, audit_logger)
        return result

    call = ToolCall(
        tool_namespace="tas.paper_mapping",
        tool_name="map_paper_to_experiments",
        arguments=payload,
        caller_workflow=WORKFLOW_NAME,
    )
    runner = runner or default_runner()
    result = runner.run(call)
    result.provenance.append(ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="registry_tool_call"))
    _attach_workflow_audit(result, payload, getattr(runner, "audit_logger", audit_logger))
    return result


def _attach_workflow_audit(result: ToolResult, payload: dict, audit_logger) -> None:
    if audit_logger is None:
        return
    audit = AuditRecord(
        event_type="workflow",
        status=result.status.value,
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": payload}),
        output_hash=stable_hash(result.model_dump(mode="json")),
        details={
            "workflow_name": WORKFLOW_NAME,
            "input_arguments": payload,
            "tools_selected": ["tas.paper_mapping.map_paper_to_experiments"],
            "tool_audit_ids": {"paper_mapping": result.audit_id} if result.audit_id else {},
            "validation_failures": result.output.get("validation_failures", []) if result.output else [],
            "provenance": [item.model_dump(mode="json") for item in result.provenance],
        },
    )
    audit_logger.write(audit)
    result.runtime_environment["workflow_audit_id"] = audit.audit_id
