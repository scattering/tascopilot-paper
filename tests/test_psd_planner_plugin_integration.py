from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolResultStatus
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.workflows.summarize_psd_facility import summarize_psd_facility


PSD_PLANNER_PATH = (plugin_path("psdplanner"))


@pytest.mark.skipif(not PSD_PLANNER_PATH.exists(), reason="local tas-psd-planner checkout is not available")
def test_local_psd_planner_plugin_registers_and_summarizes_facility(monkeypatch):
    monkeypatch.syspath_prepend(str(PSD_PLANNER_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_psd_planner.copilot_plugin")
    assert any(plugin.__name__ == "tas_psd_planner.copilot_plugin" for plugin in discover_entrypoint_plugins())

    result = summarize_psd_facility(
        {
            "facility_text": '"Facility": "NCNR", "Instrument": "BT7", "Type": "Thermal", "Monochromator": {"M1": {"Name": "PG(002)"}}',
            "source_uri": "fixture",
        }
    )

    assert result.status == ToolResultStatus.success
    assert result.output["facility_summary"]["instrument"] == "BT7"
    assert "Planning/simulation only" in result.output["safety_note"]
