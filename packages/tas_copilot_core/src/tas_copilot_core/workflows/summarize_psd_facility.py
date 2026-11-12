from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def summarize_psd_facility(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.psd_planner",
        tool_name="summarize_facility",
        arguments=arguments,
        caller_workflow="summarize-psd-facility",
    )
    return default_runner().run(call)
