from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def calculate_angles(arguments: dict) -> ToolResult:
    call = ToolCall(tool_namespace="tas.device_angles", tool_name="calculate_angles", arguments=arguments, caller_workflow="calculate-angles")
    return default_runner().run(call)

