from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def normalize_to_monitor(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.corrections",
        tool_name="normalize_to_monitor",
        arguments=arguments,
        caller_workflow="normalize-to-monitor",
    )
    return default_runner().run(call)
