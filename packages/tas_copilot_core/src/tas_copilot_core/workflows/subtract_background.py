from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def subtract_background(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.file_arithmetic",
        tool_name="subtract_background",
        arguments=arguments,
        caller_workflow="subtract-background",
    )
    return default_runner().run(call)
