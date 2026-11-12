from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def scan_tas_directory(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.data_reader",
        tool_name="scan_tas_directory",
        arguments=arguments,
        caller_workflow="scan-tas-directory",
    )
    return default_runner().run(call)
