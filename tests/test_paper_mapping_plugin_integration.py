from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import ToolResultStatus
from tas_copilot_contracts import DataFile, Experiment, Scan
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.workflows.compare_paper_to_data import compare_paper_to_data


PAPER_MAPPING_PATH = (plugin_path("bt7-paper-mapping"))


def test_compare_workflow_returns_missing_input_without_experiment_candidates(tmp_path):
    result = compare_paper_to_data(
        "paper-missing",
        {"paper": {"paper_id": "paper-missing", "title": "Unmapped paper"}},
        audit_logger=JsonlAuditLogger(tmp_path / "audit"),
    )

    assert result.status == ToolResultStatus.missing_input
    assert result.errors[0].code == "missing_experiment_candidates"
    assert result.runtime_environment["workflow_audit_id"]


@pytest.mark.skipif(not PAPER_MAPPING_PATH.exists(), reason="local bt7-paper-mapping checkout is not available")
def test_local_paper_mapping_plugin_registers_and_compare_workflow_runs(monkeypatch):
    monkeypatch.syspath_prepend(str(PAPER_MAPPING_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "bt7_paper_mapping.copilot_plugin")
    assert any(plugin.__name__ == "bt7_paper_mapping.copilot_plugin" for plugin in discover_entrypoint_plugins())

    result = compare_paper_to_data(
        "paper-ho2ti2o7",
        {
            "paper": {
                "paper_id": "paper-ho2ti2o7",
                "title": "BT-7 measurements of Ho2Ti2O7",
                "year": 2020,
                "text": "NCNR BT-7 proposal 26983 measured Ho2Ti2O7 in September 2020.",
            },
            "experiments": [
                {
                    "experiment_id": "exp-26983",
                    "proposal_id": "26983",
                    "sample": "Ho2Ti2O7",
                    "start_date": "2020-09-01",
                }
            ],
        },
    )

    assert result.status == ToolResultStatus.success
    assert result.output["link_count"] == 1
    assert result.output["links"][0]["experiment_id"] == "exp-26983"


@pytest.mark.skipif(not PAPER_MAPPING_PATH.exists(), reason="local bt7-paper-mapping checkout is not available")
def test_compare_workflow_loads_candidate_experiments_from_sqlite_catalog(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PAPER_MAPPING_PATH / "src"))
    monkeypatch.setenv("TAS_COPILOT_PLUGIN_MODULES", "bt7_paper_mapping.copilot_plugin")
    db_path = tmp_path / "bt7.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments(
        [
            Experiment(
                experiment_id="exp-26983",
                proposal_id="26983",
                title="Ho2Ti2O7 BT-7 experiment",
                start_date="2020-09-01",
            )
        ]
    )
    catalog.upsert_scans([Scan(scan_id="scan-1", experiment_id="exp-26983", scan_type="constant-Q scan")])
    catalog.upsert_datafiles([DataFile(datafile_id="df-1", path="/box/bt7/scan001.bt7", experiment_id="exp-26983", scan_id="scan-1")])

    result = compare_paper_to_data(
        "paper-ho2ti2o7",
        {
            "db_path": str(db_path),
            "paper": {
                "paper_id": "paper-ho2ti2o7",
                "title": "BT-7 measurements of Ho2Ti2O7",
                "year": 2020,
                "text": "NCNR BT-7 proposal 26983 measured Ho2Ti2O7 in September 2020.",
            },
        },
    )

    assert result.status == ToolResultStatus.success
    assert result.output["link_count"] == 1
    assert result.output["links"][0]["experiment_id"] == "exp-26983"
