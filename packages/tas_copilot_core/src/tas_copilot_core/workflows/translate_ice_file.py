from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def translate_ice_file(arguments: dict) -> ToolResult:
    call = ToolCall(
        tool_namespace="tas.reduction",
        tool_name="translate_ice_file",
        arguments=arguments,
        caller_workflow="translate-ice-file",
    )
    return default_runner().run(call)
