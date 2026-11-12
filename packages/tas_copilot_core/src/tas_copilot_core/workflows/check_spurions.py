from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def check_spurions(arguments: dict) -> ToolResult:
    call = ToolCall(tool_namespace="tas.spurion", tool_name="check_spurions", arguments=arguments, caller_workflow="check-spurions")
    return default_runner().run(call)
