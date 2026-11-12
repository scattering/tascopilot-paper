from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def apply_detailed_balance(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.corrections",
        tool_name="apply_detailed_balance",
        arguments=arguments,
        caller_workflow="apply-detailed-balance",
    )
    return default_runner().run(call)
