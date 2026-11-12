from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolResultStatus
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.workflows.classify_tas_format import classify_tas_format
from tas_copilot_core.workflows.read_tas_metadata import read_tas_metadata
from tas_copilot_core.workflows.scan_tas_directory import scan_tas_directory


TAS_DATA_READERS_PATH = (plugin_path("tas-data-readers"))


@pytest.mark.skipif(not TAS_DATA_READERS_PATH.exists(), reason="local tas-data-readers checkout is not available")
def test_local_tas_data_readers_plugin_registers_and_runs(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TAS_DATA_READERS_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "tas_data_readers.copilot_plugin")
    assert any(plugin.__name__ == "tas_data_readers.copilot_plugin" for plugin in discover_entrypoint_plugins())

    classify = classify_tas_format({"path": "TRIAX_exp0153_scan0001.dat"})
    assert classify.status == ToolResultStatus.success
    assert classify.output["facility"] == "MURR"

    metadata = read_tas_metadata({"path": "HB1A_exp1083_scan0001.dat"})
    assert metadata.status == ToolResultStatus.success
    assert metadata.output["native_experiment_id"] == "1083"

    (tmp_path / "TPN0115523.nx.hdf").write_text("fixture", encoding="utf-8")
    (tmp_path / "fmx01008.bt4.html").write_text("fixture", encoding="utf-8")
    scan = scan_tas_directory({"root": str(tmp_path)})
    assert scan.status == ToolResultStatus.success
    assert scan.output["format_counts"]["ansto_taipan_nexus_hdf5"] == 1
    assert scan.output["format_counts"]["ncnr_bt4_legacy"] == 1
