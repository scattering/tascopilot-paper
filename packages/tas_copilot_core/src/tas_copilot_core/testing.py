from __future__ import annotations

from typing import Any

from tas_copilot_contracts import ToolCall, ToolResult, ToolSpec


def assert_valid_tool_spec(spec: Any) -> ToolSpec:
    return ToolSpec.model_validate(spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec)


def assert_tool_result_contract(result: Any) -> ToolResult:
    return ToolResult.model_validate(result.model_dump(mode="json") if hasattr(result, "model_dump") else result)


def assert_valid_plugin(plugin: Any) -> list[ToolSpec]:
    assert hasattr(plugin, "get_tool_specs"), "plugin must expose get_tool_specs()"
    assert hasattr(plugin, "run_tool"), "plugin must expose run_tool(call)"
    specs = [assert_valid_tool_spec(spec) for spec in plugin.get_tool_specs()]
    assert specs, "plugin must expose at least one ToolSpec"
    for spec in specs:
        result = plugin.run_tool(ToolCall(tool_namespace=spec.namespace, tool_name="__contract_probe__", arguments={}, caller_workflow="plugin_contract_test"))
        assert_tool_result_contract(result)
    return specs
