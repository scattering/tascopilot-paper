from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def read_tas_metadata(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.data_reader",
        tool_name="read_tas_metadata",
        arguments=arguments,
        caller_workflow="read-tas-metadata",
    )
    return default_runner().run(call)
