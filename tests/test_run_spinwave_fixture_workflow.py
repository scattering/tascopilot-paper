from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry import ToolRegistry, ToolRunner, load_tool_manifest
from tas_copilot_core.workflows.run_spinwave_fixture import run_spinwave_fixture


PYSPINW_ADAPTER_PATH = (plugin_path("tas-pyspinw-adapter"))


def _ferromagnetic_chain_arguments() -> dict:
    return {
        "model": {
            "kind": "ferromagnetic_chain",
            "unit_cell": [1, 1, 1],
            "site": {"position": [0, 0, 0], "spin": [0, 0, 1], "name": "X"},
            "exchange": {"type": "heisenberg", "cell_offset": [1, 0, 0], "j": -1},
        },
        "calculation": {
            "q_path": [[0, 0, 0], [1, 0, 0]],
            "n_points_per_segment": 5,
            "use_rust": False,
        },
    }


def test_run_spinwave_fixture_returns_unavailable_with_manifest_stub(tmp_path):
    spec = next(
        item
        for item in load_tool_manifest(CoreConfig().tools_manifest_path)
        if item.namespace == "tas.pyspinw" and item.name == "run_spinwave"
    )
    registry = ToolRegistry()
    registry.register(spec)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    result = run_spinwave_fixture(_ferromagnetic_chain_arguments(), runner=runner)

    assert result.status == ToolResultStatus.unavailable
    assert result.errors[0].code == "tool_unavailable"
    assert result.audit_id
    assert result.runtime_environment["workflow_audit_id"]


def test_run_spinwave_fixture_rejects_invalid_q_path(tmp_path):
    result = run_spinwave_fixture(
        {
            "model": {
                "kind": "ferromagnetic_chain",
                "unit_cell": [1, 1, 1],
                "site": {"position": [0, 0, 0], "spin": [0, 0, 1], "name": "X"},
                "exchange": {"type": "heisenberg", "cell_offset": [1, 0, 0], "j": -1},
            },
            "calculation": {"q_path": [[0, 0, 0], [1, 0]], "n_points_per_segment": 5},
        }
    )

    assert result.status == ToolResultStatus.validation_failed
    assert result.errors[0].code == "invalid_spinwave_input"


@pytest.mark.skipif(not PYSPINW_ADAPTER_PATH.exists(), reason="local tas-pyspinw-adapter checkout is not available")
def test_run_spinwave_fixture_calls_installed_adapter(monkeypatch, tmp_path):
    pytest.importorskip("pyspinw", reason="upstream pyspinw is not installed")
    monkeypatch.syspath_prepend(str(PYSPINW_ADAPTER_PATH / "src"))
    from tas_pyspinw_adapter import copilot_plugin

    registry = ToolRegistry()
    for spec in copilot_plugin.get_tool_specs():
        registry.register(spec, copilot_plugin.run_tool)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    result = run_spinwave_fixture(_ferromagnetic_chain_arguments(), runner=runner)

    assert result.status == ToolResultStatus.success
    assert result.output["summary"] == "pySpinW ferromagnetic_chain calculation completed."
    assert result.output["simulation_only"] is True
    assert result.output["model_kind"] == "ferromagnetic_chain"
    assert result.output["q_point_count"] == 5
    assert result.runtime_environment["workflow_audit_id"]
