from __future__ import annotations

from tas_copilot_contracts import ToolCall, ToolResult

from .common import default_runner


def plan_spinwave(arguments: dict) -> ToolResult:
    safe_args = dict(arguments)
    safe_args["planning_only"] = True
    call = ToolCall(tool_namespace="tas.tasai", tool_name="plan_spinwave", arguments=safe_args, caller_workflow="plan-spinwave")
    return default_runner().run(call)

