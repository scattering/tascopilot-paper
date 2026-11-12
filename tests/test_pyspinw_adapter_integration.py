from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolCall, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.registry.tool_inventory import build_tool_inventory
from tas_copilot_core.registry.tool_registry import ToolRegistry
from tas_copilot_core.registry.tool_runner import ToolRunner


PYSPINW_ADAPTER_PATH = (plugin_path("tas-pyspinw-adapter"))


@pytest.mark.skipif(not PYSPINW_ADAPTER_PATH.exists(), reason="local tas-pyspinw-adapter checkout is not available")
def test_local_pyspinw_adapter_registers_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PYSPINW_ADAPTER_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_pyspinw_adapter.copilot_plugin")

    plugins = discover_entrypoint_plugins()
    plugin = next(item for item in plugins if item.__name__ == "tas_pyspinw_adapter.copilot_plugin")
    specs = plugin.get_tool_specs()
    spec = specs[0]
    assert spec.namespace == "tas.pyspinw"
    assert spec.name == "run_spinwave"
    assert [permission.value for permission in spec.permissions_required] == ["simulation_only"]

    registry = ToolRegistry()
    registry.register(spec, plugin.run_tool)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path))

    missing = runner.run(ToolCall(tool_namespace="tas.pyspinw", tool_name="run_spinwave", arguments={}))
    assert missing.status == ToolResultStatus.missing_input
    assert missing.audit_id

    def import_missing(name):
        if name in {"pyspinw", "spinw"}:
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(plugin, "import_module", import_missing)

    unavailable = runner.run(ToolCall(tool_namespace="tas.pyspinw", tool_name="run_spinwave", arguments={"model": {"material": "fixture"}}))
    assert unavailable.status == ToolResultStatus.unavailable
    assert unavailable.errors[0].code == "pyspinw_unavailable"
    assert unavailable.runtime_environment["pyspinw_importable"] is False


@pytest.mark.skipif(not PYSPINW_ADAPTER_PATH.exists(), reason="local tas-pyspinw-adapter checkout is not available")
def test_pyspinw_adapter_shadows_manifest_stub(monkeypatch):
    monkeypatch.syspath_prepend(str(PYSPINW_ADAPTER_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_pyspinw_adapter.copilot_plugin")

    inventory = build_tool_inventory(CoreConfig())
    data = inventory.as_dict()

    tool = next(item for item in data["tools"] if item["namespace"] == "tas.pyspinw" and item["name"] == "run_spinwave" and item["effective"])
    assert tool["status"] == "available"
    assert tool["source"] == "plugin:tas_pyspinw_adapter.copilot_plugin"
    assert tool["permissions_required"] == ["simulation_only"]
    assert any(
        item["namespace"] == "tas.pyspinw" and item["name"] == "run_spinwave"
        for item in data["shadowed_manifest_stubs"]
    )
