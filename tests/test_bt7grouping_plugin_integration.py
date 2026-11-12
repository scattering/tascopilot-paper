import sys
from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolCall, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.registry import ToolRegistry, ToolRunner
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins


BT7GROUPING_PATH = (plugin_path("bt7grouping"))


@pytest.mark.skipif(not BT7GROUPING_PATH.exists(), reason="local bt7grouping checkout is not available")
def test_local_bt7grouping_plugin_registers_and_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BT7GROUPING_PATH))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "bt7grouping.copilot_plugin")
    assert any(plugin.__name__ == "bt7grouping.copilot_plugin" for plugin in discover_entrypoint_plugins())
    from bt7grouping import copilot_plugin

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)

    data_file = tmp_path / "legacy.bt7"
    data_file.write_text("'legacy.bt7' '01-Jan-2020 12:00' 'I'\n", encoding="latin-1")

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.data_reader",
            tool_name="read_bt7_file",
            arguments={"path": str(data_file)},
            caller_workflow="test",
        )
    )

    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert result.output["metadata"]["format_type"] == "legacy"
