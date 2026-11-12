from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolCall, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.registry import ToolRegistry, ToolRunner
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins


SPURION_PATH = (plugin_path("spurion"))


def _arguments():
    return {
        "q_point": {"h": 1.0, "k": 0.0, "l": 0.0},
        "energy_transfer": {"value": {"value": 0.0, "unit": "meV"}},
        "instrument_config": {
            "fixed_energy_mode": "fixed_final_energy",
            "fixed_energy": {"value": 14.7, "unit": "meV"},
        },
        "lattice": {
            "a": 6.283185307179586,
            "b": 6.283185307179586,
            "c": 6.283185307179586,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "orient1": [[1, 0, 0]],
            "orient2": [[0, 1, 0]],
        },
        "q_tolerance": 1e-8,
        "hkl_max": 1,
        "harmonics": 1,
        "include_powder_rings": False,
        "include_incoherent": False,
    }


@pytest.mark.skipif(not SPURION_PATH.exists(), reason="local spurion checkout is not available")
def test_local_spurion_plugin_registers_and_runs_through_registry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SPURION_PATH))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_spurion.copilot_plugin")
    assert any(plugin.__name__ == "tas_spurion.copilot_plugin" for plugin in discover_entrypoint_plugins())
    from tas_spurion import copilot_plugin

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)

    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    result = runner.run(
        ToolCall(
            tool_namespace="tas.spurion",
            tool_name="check_spurions",
            arguments=_arguments(),
            caller_workflow="test",
        )
    )

    assert result.status == ToolResultStatus.success
    assert result.audit_id
    assert result.output["spurion_check"]["summary"].startswith("1 candidate")
    assert result.output["details"]["candidates"][0]["kind"] == "reciprocal_lattice_primary"
