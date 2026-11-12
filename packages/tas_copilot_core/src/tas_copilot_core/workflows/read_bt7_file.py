from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def read_bt7_file(arguments: dict) -> ToolResult:
    call = ToolCall(tool_namespace="tas.data_reader", tool_name="read_bt7_file", arguments=arguments, caller_workflow="read-bt7-file")
    return default_runner().run(call)

