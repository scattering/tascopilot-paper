import importlib
from pathlib import Path

from tas_copilot_contracts import EnergyTransfer, Experiment, InstrumentConfig, QPoint, Scan, ToolResult, ToolResultStatus, UnitValue
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.completeness import eligible_scans_report, metadata_completeness_report
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.retrieve_similar_scans import retrieve_similar_scans


def _complete_scan(scan_id="scan-complete"):
    return Scan(
        scan_id=scan_id,
        experiment_id="exp-1",
        instrument="BT-7",
        scan_type="constant-Q scan",
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
                "a": 6.28,
                "b": 6.28,
                "c": 6.28,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "orient1": [[1, 0, 0]],
                "orient2": [[0, 1, 0]],
            },
            "temperature": 4.0,
            "monitor_counts": 1000,
            "count_time": 60,
        },
    )


def test_metadata_completeness_and_eligible_scan_selection(tmp_path):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    catalog.upsert_scans([_complete_scan(), Scan(scan_id="scan-sparse", experiment_id="exp-1", instrument="HB-1A")])

    completeness = metadata_completeness_report(db_path)
    eligible = eligible_scans_report(db_path, purpose="geometry")

    assert completeness["summary"]["scan_count"] == 2
    assert completeness["summary"]["eligible_for_geometry_check"] == 1
    assert eligible["best_candidate"] == "scan-complete"
    assert eligible["rejected"][0]["scan_id"] == "scan-sparse"


def test_metadata_completeness_explains_alignment_scan_geometry_gap(tmp_path):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    scan = _complete_scan("fp001")
    scan.metadata.pop("lattice")
    scan.metadata["lattice"] = {"a": 6.28, "b": 6.28, "c": 6.28, "alpha": 90, "beta": 90, "gamma": 90}
    scan.metadata["raw_metadata"] = {"scan_role": "alignment"}
    catalog.upsert_scans([scan])

    report = metadata_completeness_report(db_path)
    item = report["scans"][0]

    assert item["scan_role"] == "alignment"
    assert item["metadata_mode"] == "read_prefix"
    assert item["path_only"] is False
    assert item["geometry_missing_fields"] == ["scattering_plane"]
    assert item["reason"] == "alignment scan lacks scattering-plane metadata"


def test_retrieve_similar_scans_uses_sqlite_catalog_and_sparse_metadata(tmp_path):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    catalog.upsert_scans([_complete_scan("scan-a"), _complete_scan("scan-b"), Scan(scan_id="scan-c", experiment_id="exp-1", instrument="TAIPAN")])

    results = retrieve_similar_scans("scan-a", limit=2, catalog=db_path)

    assert results[0]["scan_id"] == "scan-b"
    assert results[0]["similarity_score"] > results[1]["similarity_score"]
    assert "instrument" in results[0]["matched_dimensions"]
    assert results[1]["missing_dimensions"]


def test_catalog_completeness_cli_json(tmp_path, capsys):
    db_path = tmp_path / "tas.sqlite"
    catalog = SQLiteDataCatalog(db_path)
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    catalog.upsert_scans([_complete_scan()])

    assert main(["catalog", "completeness", "--catalog", str(db_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"eligible_for_geometry_check": 1' in output


def test_real_data_smoke_writes_reports_and_handles_missing_geometry(tmp_path, monkeypatch):
    smoke_module = importlib.import_module("tas_copilot_core.workflows.real_data_smoke")
    db_path = tmp_path / "smoke.sqlite"
    output_dir = tmp_path / "reports"

    def fake_inspect(arguments, audit_logger=None):
        assert arguments.get("file_paths") == ["/synthetic/scan001.dat"]
        assert arguments.get("path_only") is True
        return {
            "workflow_name": "inspect-bt7-directory",
            "status": "success",
            "discovered_files": ["/synthetic/scan001.dat"],
            "format_counts": {"synthetic": 1},
            "metadata": [],
            "group_count": 0,
            "workflow_audit_id": "inspect-audit",
        }

    def fake_catalog(arguments):
        assert arguments.get("file_paths") == ["/synthetic/scan001.dat"]
        assert arguments.get("path_only") is True
        catalog = SQLiteDataCatalog(arguments["db_path"])
        catalog.upsert_experiments([Experiment(experiment_id="exp-1", instrument="HB-1A")])
        catalog.upsert_scans([Scan(scan_id="scan-sparse", experiment_id="exp-1", instrument="HB-1A", metadata={"raw_metadata": {"scan_role": "alignment"}})])
        return {
            "status": "success",
            "db_path": arguments["db_path"],
            "imported": {"experiments": 1, "scans": 1, "data_files": 0},
            "catalog_summary": catalog.summary(),
        }

    monkeypatch.setattr(smoke_module, "inspect_bt7_directory", fake_inspect)
    monkeypatch.setattr(smoke_module, "catalog_tas_directory", fake_catalog)

    result = smoke_module.real_data_smoke(
        {
            "path": str(tmp_path),
            "file_paths": ["/synthetic/scan001.dat"],
            "path_only": True,
            "catalog": str(db_path),
            "output_dir": str(output_dir),
        },
        audit_logger=JsonlAuditLogger(tmp_path / "audit"),
    )

    assert result["status"] == "partial"
    assert result["geometry_check"]["status"] == "missing_input"
    assert result["geometry_check"]["reason"].startswith("no eligible geometry scan")
    assert result["geometry_check"]["alignment_scan_count"] == 1
    assert result["artifacts"] == {"json": str(output_dir / "report.json"), "markdown": str(output_dir / "report.md")}
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()


def test_catalog_tas_directory_records_file_hash(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "scan001.dat"
    data_file.write_text("# synthetic\n", encoding="utf-8")

    def fake_scan(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "TEST",
                        "instrument": "TAS",
                        "format_family": "synthetic",
                        "native_format": "synthetic",
                        "native_scan_id": "1",
                        "native_experiment_id": "exp",
                        "source_adapter": "fixture",
                    }
                ],
                "format_counts": {"synthetic": 1},
                "instrument_counts": {"TAS": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    result = catalog_module.catalog_tas_directory({"path": str(tmp_path), "db_path": str(db_path)})
    datafiles = SQLiteDataCatalog(db_path).list_datafiles()

    assert result["status"] == "success"
    assert datafiles[0].checksum
    assert len(datafiles[0].checksum) == 64


def test_catalog_tas_directory_normalizes_bt7_header_hints(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "LuMnFeO3_29pt5_TdepWarm102246.bt7"
    data_file.write_text("# synthetic BT-7 header\n", encoding="utf-8")

    def fake_scan(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "NCNR",
                        "instrument": "BT-7",
                        "format_family": "ncnr_bt7_ice_nice",
                        "native_format": "BT-7 ICE/NICE",
                        "native_scan_id": "102246",
                        "source_adapter": "fixture",
                        "raw_metadata": {
                            "filename": data_file.name,
                            "header_fields": {
                                "detectorefficiencies_monitor": "PSDC00=1.0 PSDC01=1.0 PSDC47=1.0",
                                "scandescr_scan_subid": "18445:JType=ENV:Npts=81:Filename=LuMnFeO3_29pt5_TdepWarm:Range=Temp=2.5 202.5 s:Title=TscanInc",
                            },
                        },
                    }
                ],
                "format_counts": {"ncnr_bt7_ice_nice": 1},
                "instrument_counts": {"BT-7": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    result = catalog_module.catalog_tas_directory({"path": str(tmp_path), "db_path": str(db_path), "skip_bt7_enrichment": True})
    catalog = SQLiteDataCatalog(db_path)
    scan = catalog.list_scans()[0]
    datafile = catalog.list_datafiles()[0]

    assert result["status"] == "success"
    assert scan.scan_type == "ENV"
    assert [item.value for item in scan.temperature_range] == [2.5, 202.5]
    assert scan.metadata["temperature_range"] == {"min": 2.5, "max": 202.5, "unit": "K", "source": "ScanDescr Range=Temp"}
    assert scan.metadata["psd_channel_axis"] == [0, 1, 47]
    assert scan.metadata["psd_channel_count"] == 3
    assert scan.metadata["detector_mode"] == "PSD"
    assert scan.metadata["a4"] == 29.5
    assert datafile.metadata["scan_type"] == "ENV"
    assert datafile.metadata["psd_channel_axis"] == [0, 1, 47]


def test_catalog_tas_directory_normalizes_bt7_energy_range(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "LuFeMnO3_Tdep_3meV102289.bt7"
    data_file.write_text("# synthetic BT-7 header\n", encoding="utf-8")

    def fake_scan(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "NCNR",
                        "instrument": "BT-7",
                        "format_family": "ncnr_bt7_ice_nice",
                        "native_format": "BT-7 ICE/NICE",
                        "native_scan_id": "102289",
                        "source_adapter": "fixture",
                        "raw_metadata": {
                            "filename": data_file.name,
                            "header_fields": {
                                "scandescr_scan_subid": "18445:JType=VECTOR:FixedE=14.7:Npts=37:Range=Q=0.8~0.0~0.0 2.6~0.0~0.0 s:Range=E=3.0 3.0 s:Title=E3T",
                            },
                        },
                    }
                ],
                "format_counts": {"ncnr_bt7_ice_nice": 1},
                "instrument_counts": {"BT-7": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    catalog_module.catalog_tas_directory({"path": str(tmp_path), "db_path": str(db_path), "skip_bt7_enrichment": True})
    scan = SQLiteDataCatalog(db_path).list_scans()[0]

    assert scan.scan_type == "VECTOR"
    assert scan.energy_transfer is not None
    assert scan.energy_transfer.value.value == 3.0
    assert scan.metadata["energy_range"] == {"min": 3.0, "max": 3.0, "unit": "meV", "source": "ScanDescr Range=E"}
    assert scan.metadata["q_path_hint"]["source"] == "ScanDescr Range=Q"


def test_catalog_tas_directory_accepts_explicit_file_paths_without_path(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "HB1A_exp1083_scan0059.dat"
    data_file.write_text("# h k l en\n1 0 0 0\n", encoding="utf-8")

    def fake_scan(arguments):
        assert arguments["root"] == str(tmp_path)
        assert arguments["paths"] == [str(data_file)]
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "HFIR",
                        "instrument": "HB-1A",
                        "format_family": "hfir_hb1a_dat",
                        "native_format": "HB-1A dat",
                        "native_scan_id": "0059",
                        "native_experiment_id": "1083",
                        "source_adapter": "fixture",
                    }
                ],
                "format_counts": {"hfir_hb1a_dat": 1},
                "instrument_counts": {"HB-1A": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    result = catalog_module.catalog_tas_directory({"file_paths": [str(data_file)], "db_path": str(db_path)})

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path)
    assert SQLiteDataCatalog(db_path).summary()["data_files"] == 1


def test_catalog_tas_directory_path_only_records_skip_file_hash(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "HB1A_exp1083_scan0059.dat"
    data_file.write_text("# synthetic\n", encoding="utf-8")

    def fake_scan(arguments):
        assert arguments["path_only"] is True
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "HFIR",
                        "instrument": "HB-1A",
                        "format_family": "hfir_hb1a_dat",
                        "native_format": "HB-1A dat",
                        "native_scan_id": "0059",
                        "native_experiment_id": "1083",
                        "source_adapter": "fixture",
                        "raw_metadata": {"metadata_mode": "path_only"},
                    }
                ],
                "format_counts": {"hfir_hb1a_dat": 1},
                "instrument_counts": {"HB-1A": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    catalog_module.catalog_tas_directory({"file_paths": [str(data_file)], "path_only": True, "db_path": str(db_path)})

    catalog = SQLiteDataCatalog(db_path)
    assert catalog.list_datafiles()[0].checksum is None
    completeness = metadata_completeness_report(db_path)
    assert completeness["scans"][0]["path_only"] is True
    assert completeness["scans"][0]["reason"] == "path-only metadata; file content was not read"


def test_catalog_tas_directory_maps_reader_geometry_hints(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")
    data_file = tmp_path / "TAIPAN_exp1337_scan115387.dat"
    data_file.write_text("# synthetic\n", encoding="utf-8")

    def fake_scan(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            output={
                "records": [
                    {
                        "path": str(data_file),
                        "facility": "ANSTO",
                        "instrument": "TAIPAN",
                        "format_family": "ansto_taipan_dat",
                        "native_format": "TAIPAN dat",
                        "native_scan_id": "115387",
                        "native_experiment_id": "1337",
                        "source_adapter": "fixture",
                        "q_hint": {"h": 1.0, "k": 0.0, "l": 0.0},
                        "energy_hint": {"value": 0.5, "unit": "meV", "source": "fixture"},
                        "raw_metadata": {
                            "instrument_config_hint": {
                                "fixed_energy_mode": "fixed_final_energy",
                                "fixed_energy": {"value": 14.7, "unit": "meV"},
                                "monochromator": "PG(002)",
                                "analyzer": "PG(002)",
                                "filters": ["PG"],
                                "collimation": "open-open-open-open",
                            },
                            "lattice": {"a": 10.1, "b": 10.1, "c": 10.1, "alpha": 90, "beta": 90, "gamma": 90},
                            "monitor_counts": 10000,
                            "count_time": 60,
                        },
                    }
                ],
                "format_counts": {"ansto_taipan_dat": 1},
                "instrument_counts": {"TAIPAN": 1},
            },
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan)
    db_path = tmp_path / "tas.sqlite"
    result = catalog_module.catalog_tas_directory({"path": str(tmp_path), "db_path": str(db_path)})
    scan = SQLiteDataCatalog(db_path).list_scans()[0]
    completeness = metadata_completeness_report(db_path)

    assert result["status"] == "success"
    assert scan.q_center.h == 1.0
    assert scan.energy_transfer.value.value == 0.5
    assert scan.instrument_config.fixed_energy.value == 14.7
    assert scan.metadata["lattice"]["a"] == 10.1
    assert scan.metadata["monitor_counts"] == 10000
    assert completeness["summary"]["eligible_for_resolution_check"] == 1


def test_no_plugin_smoke_cli_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tas_copilot_core.registry.plugin_discovery.discover_entrypoint_plugins", lambda: [])
    assert main(["tools", "doctor"]) == 0
    assert main(["workflow", "check-geometry", "--scan-id", "missing", "--db", str(tmp_path / "missing.sqlite")]) == 0
    output = capsys.readouterr().out
    assert "TASCopilot tool doctor" in output
    assert "missing_input" in output
