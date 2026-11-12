from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def add_scans(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.file_arithmetic",
        tool_name="add_scans",
        arguments=arguments,
        caller_workflow="add-scans",
    )
    return default_runner().run(call)
