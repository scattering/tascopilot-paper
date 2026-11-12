from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def group_bt7_scans(arguments: dict) -> ToolResult:
    call = ToolCall(tool_namespace="tas.scan_grouping", tool_name="group_bt7_scans", arguments=arguments, caller_workflow="group-bt7-scans")
    return default_runner().run(call)

