from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def powder_order_parameter(arguments: dict) -> ToolResult:
    call = ToolCall(tool_namespace="tas.andie", tool_name="powder_order_parameter", arguments=arguments, caller_workflow="powder-order-parameter")
    return default_runner().run(call)
