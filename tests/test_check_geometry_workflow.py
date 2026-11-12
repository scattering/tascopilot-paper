from pathlib import Path

from plugin_paths import plugin_path

import pytest

from tas_copilot_contracts import Experiment, InstrumentConfig, QPoint, Scan, ToolResultStatus, UnitValue, EnergyTransfer
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.completeness import eligible_scans_report, metadata_completeness_report
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry import ToolRegistry, ToolRunner, load_tool_manifest
from tas_copilot_core.workflows.check_geometry import check_geometry


RESCALCULATOR_PATH = (plugin_path("rescalculator"))
SPURION_PATH = (plugin_path("spurion"))


def _geometry_arguments() -> dict:
    return {
        "q_point": {"h": 1.0, "k": 0.0, "l": 0.0},
        "energy_transfer": {"value": {"value": 0.0, "unit": "meV"}},
        "instrument_config": {
            "fixed_energy_mode": "fixed_final_energy",
            "fixed_energy": {"value": 14.7, "unit": "meV"},
            "monochromator": "PG(002)",
            "analyzer": "PG(002)",
            "collimation": "40-40-40-80",
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
        "backend": "numpy",
        "q_tolerance": 1e-8,
        "hkl_max": 1,
        "harmonics": 1,
        "include_powder_rings": False,
        "include_incoherent": False,
    }


def test_check_geometry_reports_missing_fields(tmp_path):
    result = check_geometry({"q_point": {"h": 1}}, audit_logger=JsonlAuditLogger(tmp_path / "audit"))

    assert result["status"] == ToolResultStatus.missing_input.value
    assert "q_point.k" in result["missing_fields"]
    assert "energy_transfer" in result["missing_fields"]
    assert result["workflow_audit_id"]


def test_check_geometry_returns_unavailable_with_manifest_stubs(tmp_path):
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(CoreConfig().tools_manifest_path))
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    result = check_geometry(_geometry_arguments(), runner=runner)

    assert result["status"] == ToolResultStatus.unavailable.value
    assert result["geometry_mode"] == "lattice"
    assert set(result["results"]) == {"device_angles", "resolution", "spurions"}
    assert all(item["status"] == ToolResultStatus.unavailable.value for item in result["results"].values())
    assert all(result["tool_audit_ids"].values())
    assert result["workflow_audit_id"]


def test_check_geometry_eulerian_mode_routes_to_ub_tool(tmp_path):
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(CoreConfig().tools_manifest_path))
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    arguments = {**_geometry_arguments(), "geometry_mode": "eulerian_ub"}

    result = check_geometry(arguments, runner=runner)

    assert result["status"] == ToolResultStatus.unavailable.value
    assert result["geometry_mode"] == "eulerian_ub"
    assert set(result["results"]) == {"ub_matrix", "device_angles", "resolution", "spurions"}


def test_check_geometry_compatible_ub_mode_does_not_route_to_eulerian_ub_tool(tmp_path):
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(CoreConfig().tools_manifest_path))
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))
    arguments = {
        **_geometry_arguments(),
        "geometry_mode": "compatible_ub",
        "ub_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "ub_convention": "hfir_murr_instrument_control",
    }

    result = check_geometry(arguments, runner=runner)

    assert result["status"] == ToolResultStatus.unavailable.value
    assert result["geometry_mode"] == "compatible_ub"
    assert set(result["results"]) == {"device_angles", "resolution", "spurions"}
    assert all(item["errors"][0]["code"] == "unsupported_geometry_convention" for item in result["results"].values())
    assert not any(result["tool_audit_ids"].values())


def test_check_geometry_resolves_scan_id_from_sqlite_catalog(tmp_path):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    catalog.upsert_scans(
        [
            Scan(
                scan_id="scan-1",
                experiment_id="exp-1",
                q_center=QPoint(h=1, k=0, l=0),
                energy_transfer=EnergyTransfer(value=UnitValue(value=0, unit="meV")),
                instrument_config=InstrumentConfig(
                    fixed_energy_mode="fixed_final_energy",
                    fixed_energy=UnitValue(value=14.7, unit="meV"),
                    monochromator="PG(002)",
                    analyzer="PG(002)",
                    collimation="40-40-40-80",
                ),
                metadata={
                    "lattice": {
                        "a": 6.283185307179586,
                        "b": 6.283185307179586,
                        "c": 6.283185307179586,
                        "alpha": 90,
                        "beta": 90,
                        "gamma": 90,
                        "orient1": [[1, 0, 0]],
                        "orient2": [[0, 1, 0]],
                    }
                },
            )
        ]
    )
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(CoreConfig().tools_manifest_path))
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    result = check_geometry({"scan_id": "scan-1", "db_path": str(db_path)}, runner=runner)

    assert result["missing_fields"] == []
    assert result["status"] == ToolResultStatus.unavailable.value
    assert result["workflow_audit_id"]
    assert result["results"]["resolution"]["status"] == ToolResultStatus.unavailable.value


def test_check_geometry_scan_id_reports_catalog_misses(tmp_path):
    result = check_geometry({"scan_id": "missing", "db_path": str(tmp_path / "missing.sqlite")}, audit_logger=JsonlAuditLogger(tmp_path / "audit"))

    assert result["status"] == ToolResultStatus.missing_input.value
    assert "scan_id.catalog_missing" in result["missing_fields"]
    assert result["validation_failures"][0]["code"] == "catalog_missing"


def test_compatible_ub_scan_is_preserved_but_fails_closed_from_catalog(tmp_path):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-murr", instrument="TRIAX")])
    catalog.upsert_scans(
        [
            Scan(
                scan_id="scan-compatible-ub",
                experiment_id="exp-murr",
                instrument="TRIAX",
                q_center=QPoint(h=1, k=0, l=0),
                energy_transfer=EnergyTransfer(value=UnitValue(value=0, unit="meV")),
                instrument_config=InstrumentConfig(
                    fixed_energy_mode="fixed_final_energy",
                    fixed_energy=UnitValue(value=14.7, unit="meV"),
                    monochromator="PG(002)",
                    analyzer="PG(002)",
                ),
                metadata={
                    "ub_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "ub_convention": "hfir_murr_instrument_control",
                },
            )
        ]
    )
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(CoreConfig().tools_manifest_path))
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    completeness = metadata_completeness_report(db_path)
    eligible = eligible_scans_report(db_path, purpose="geometry")
    result = check_geometry({"scan_id": "scan-compatible-ub", "db_path": str(db_path), "geometry_mode": "compatible_ub"}, runner=runner)

    scan_report = completeness["scans"][0]
    assert scan_report["ub_convention"] == "hfir_murr_instrument_control"
    assert scan_report["eligible_for_compatible_ub_check"] is True
    assert scan_report["eligible_for_geometry_check"] is False
    assert eligible["best_candidate"] is None
    assert eligible["rejected"][0]["reason"] == "missing lattice_parameters"
    assert result["status"] == ToolResultStatus.unavailable.value
    assert all(item["errors"][0]["code"] == "unsupported_geometry_convention" for item in result["results"].values())
    assert not any(result["tool_audit_ids"].values())


@pytest.mark.skipif(
    not (RESCALCULATOR_PATH.exists() and SPURION_PATH.exists()),
    reason="local rescalculator and spurion checkouts are not available",
)
def test_check_geometry_routes_to_installed_plugins(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(RESCALCULATOR_PATH))
    monkeypatch.syspath_prepend(str(SPURION_PATH))
    import copilot_plugin as rescalculator_plugin
    from tas_spurion import copilot_plugin as spurion_plugin

    registry = ToolRegistry()
    for spec in rescalculator_plugin.get_tool_specs():
        registry.register(spec, rescalculator_plugin.run_tool)
    for spec in spurion_plugin.get_tool_specs():
        registry.register(spec, spurion_plugin.run_tool)
    runner = ToolRunner(registry, JsonlAuditLogger(tmp_path / "audit"))

    result = check_geometry(_geometry_arguments(), runner=runner)

    if result["status"] == "partial" and any(
        item["status"] == ToolResultStatus.unavailable.value
        and any(error["code"] == "optional_dependency_missing" for error in item.get("errors", []))
        for item in result["results"].values()
    ):
        pytest.skip("local geometry plugins are present but optional rescalculator dependencies are not importable")
    assert result["status"] == ToolResultStatus.success.value
    assert "ub_matrix" not in result["results"]
    assert result["results"]["device_angles"]["status"] == ToolResultStatus.success.value
    assert result["results"]["resolution"]["status"] == ToolResultStatus.success.value
    assert result["results"]["spurions"]["status"] == ToolResultStatus.success.value
    assert result["results"]["resolution"]["output"]["backend_used"] == "numpy"
    assert result["results"]["device_angles"]["output"]["device_angles"]["angles"]
    assert result["results"]["spurions"]["output"]["spurion_check"]["summary"]
