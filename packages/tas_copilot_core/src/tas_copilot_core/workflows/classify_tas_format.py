from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def classify_tas_format(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.data_reader",
        tool_name="classify_tas_format",
        arguments=arguments,
        caller_workflow="classify-tas-format",
    )
    return default_runner().run(call)
