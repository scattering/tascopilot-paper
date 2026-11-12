from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def calculate_ub(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.ubmatrix",
        tool_name="calculate_ub",
        arguments=arguments,
        caller_workflow="calculate-ub",
    )
    return default_runner().run(call)
