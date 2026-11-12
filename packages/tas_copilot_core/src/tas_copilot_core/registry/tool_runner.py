from __future__ import annotations

import hashlib
import json
import platform

from tas_copilot_contracts import (
    AuditRecord,
    Availability,
    ProvenanceRecord,
    ToolCall,
    ToolError,
    ToolResult,
    ToolResultStatus,
)

from .permissions import PermissionPolicy
from .tool_registry import ToolRegistry
from tas_copilot_core.validation.safety_gates import validate_no_live_control_arguments


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolRunner:
    def __init__(self, registry: ToolRegistry, audit_logger, policy: PermissionPolicy | None = None) -> None:
        self.registry = registry
        self.audit_logger = audit_logger
        self.policy = policy or PermissionPolicy()

    def preflight(self, call: ToolCall) -> ToolResult | None:
        """Check current permissions and availability, including before cache reuse."""
        input_hash = stable_hash(call.model_dump(mode="json"))
        try:
            registered = self.registry.get(call.tool_namespace, call.tool_name)
        except KeyError:
            return self._finalize(call, ToolResult(
                status=ToolResultStatus.unavailable,
                errors=[ToolError(code="tool_unregistered", message="tool is not registered")],
                tool_namespace=call.tool_namespace,
                tool_name=call.tool_name,
                input_hash=input_hash,
            ))
        spec = registered.spec
        allowed, reason = self.policy.check(spec, call)
        safety_reason = validate_no_live_control_arguments(call.arguments)
        if safety_reason:
            allowed, reason = False, safety_reason
        if not allowed:
            return self._finalize(
                call,
                ToolResult(
                    status=ToolResultStatus.blocked,
                    output=None,
                    errors=[ToolError(code="permission_blocked", message=reason or "blocked")],
                    tool_name=spec.name,
                    tool_namespace=spec.namespace,
                    tool_version=spec.version,
                    input_hash=input_hash,
                    provenance=[ProvenanceRecord(source_type="tool", source_id=f"{spec.namespace}.{spec.name}", method="permission_policy")],
                ),
            )
        if spec.availability != Availability.available or registered.handler is None:
            return self._finalize(
                call,
                ToolResult(
                    status=ToolResultStatus.unavailable,
                    output=None,
                    errors=[ToolError(code="tool_unavailable", message="registered adapter is unavailable")],
                    tool_name=spec.name,
                    tool_namespace=spec.namespace,
                    tool_version=spec.version,
                    input_hash=input_hash,
                    runtime_environment={"python": platform.python_version(), "platform": platform.platform()},
                    provenance=[ProvenanceRecord(source_type="tool", source_id=f"{spec.namespace}.{spec.name}", method="unavailable_stub")],
                ),
            )
        return None

    def run(self, call: ToolCall) -> ToolResult:
        failure = self.preflight(call)
        if failure is not None:
            return failure
        registered = self.registry.get(call.tool_namespace, call.tool_name)
        spec = registered.spec
        input_hash = stable_hash(call.model_dump(mode="json"))
        try:
            result = registered.handler(call)
        except Exception as exc:
            result = ToolResult(
                status=ToolResultStatus.error,
                output=None,
                errors=[ToolError(code=exc.__class__.__name__, message=str(exc))],
                tool_name=spec.name,
                tool_namespace=spec.namespace,
                tool_version=spec.version,
                input_hash=input_hash,
            )
        result.input_hash = result.input_hash or input_hash
        result.runtime_environment = result.runtime_environment or {"python": platform.python_version(), "platform": platform.platform()}
        return self._finalize(call, result)

    def _finalize(self, call: ToolCall, result: ToolResult) -> ToolResult:
        result.output_hash = result.output_hash or stable_hash(result.output)
        audit = AuditRecord(
            event_type="tool_call",
            actor=call.user_context.get("user"),
            request_id=call.request_id,
            status=result.status.value,
            input_hash=result.input_hash,
            output_hash=result.output_hash,
            tool_namespace=call.tool_namespace,
            tool_name=call.tool_name,
            details={"caller_workflow": call.caller_workflow, "dry_run": call.dry_run},
        )
        self.audit_logger.write(audit)
        result.audit_id = audit.audit_id
        return result
