import sqlite3
import importlib

from tas_copilot_contracts import (
    DataFile,
    DerivedDataFile,
    EnergyTransfer,
    Experiment,
    InstrumentConfig,
    PaperExperimentLink,
    QPoint,
    Scan,
    UnitValue,
)
from tas_copilot_core.catalogs.import_data_catalog import import_jsonl_catalog_dir
from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.cli import main
from tas_copilot_core.config import CoreConfig
from tas_copilot_contracts import ToolResult, ToolResultStatus


def test_sqlite_data_catalog_round_trip_and_lineage(tmp_path):
    db_path = tmp_path / "bt7.sqlite"
    catalog = SQLiteDataCatalog(db_path)

    experiment = Experiment(experiment_id="exp-1", title="Data experiment")
    scan = Scan(scan_id="scan-1", experiment_id="exp-1", q_center=QPoint(h=1, k=0, l=0))
    datafile = DataFile(
        datafile_id="df-1",
        path="data/raw/scan.bt7",
        scan_id="scan-1",
        experiment_id="exp-1",
        metadata={
            "facility": "NCNR",
            "instrument": "BT-7",
            "native_format": "BT-7 ICE/NICE",
            "native_scan_id": "scan001",
            "native_experiment_id": "123",
            "source_adapter": "fixture",
        },
    )
    derived = DerivedDataFile(
        datafile_id="df-derived-1",
        path="data/processed/scan.dat",
        scan_id="scan-1",
        experiment_id="exp-1",
        input_datafile_ids=["df-1"],
        tool_call_id="tool-call-1",
        parameters={"operation": "normalize"},
    )
    link = PaperExperimentLink(
        paper_id="paper-external-id",
        experiment_id="exp-1",
        confidence=0.75,
        evidence=["shared participant"],
        mapping_method="fixture",
        status="candidate",
    )

    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    catalog.upsert_derived_datafiles([derived])
    catalog.upsert_paper_experiment_links([link])

    assert catalog.list_experiments()[0].experiment_id == "exp-1"
    assert catalog.list_scans()[0].scan_id == "scan-1"
    assert catalog.datafiles_for_scan("scan-1")[0].datafile_id == "df-1"
    assert catalog.datafiles_for_scan("scan-1")[0].metadata["facility"] == "NCNR"
    assert catalog.list_paper_experiment_links()[0].paper_id == "paper-external-id"
    candidates = catalog.mapping_candidate_experiments()
    assert candidates[0]["experiment_id"] == "exp-1"
    assert candidates[0]["metadata"]["datafile_count"] == 1
    metrics = catalog.metrics()
    assert metrics["data_file_count"] == 1
    assert metrics["facility_count"] == 1
    assert metrics["instrument_count"] == 1
    assert metrics["db_size_bytes"] > 0

    lineage = catalog.lineage_for_derived_file("df-derived-1")
    assert lineage["found"] is True
    assert lineage["tool_call_id"] == "tool-call-1"
    assert lineage["input_datafile_ids"] == ["df-1"]


def test_data_catalog_quality_report_clean_catalog(tmp_path):
    catalog = SQLiteDataCatalog(tmp_path / "clean.sqlite")
    experiment = Experiment(experiment_id="tas-experiment-ncnr-bt-7-25860", title="fixture", metadata={"facility": "NCNR", "native_experiment_id": "25860", "source_adapter": "fixture"})
    scan = Scan(
        scan_id="tas-scan-bt-7-25860-2527",
        experiment_id=experiment.experiment_id,
        instrument="BT-7",
        scan_type="VECTOR",
        metadata={
            "facility": "NCNR",
            "native_scan_id": "2527",
            "source_adapter": "fixture",
            "bt7_enrichment": {"expt_id": "25860", "scan_variable": "L"},
        },
    )
    datafile = DataFile(
        datafile_id="tas-datafile-1",
        path="/box/bt7/L_scan_2527.bt7",
        scan_id=scan.scan_id,
        experiment_id=experiment.experiment_id,
        metadata={
            "facility": "NCNR",
            "instrument": "BT-7",
            "native_format": "BT-7 ICE/NICE",
            "native_scan_id": "2527",
            "native_experiment_id": "25860",
            "source_adapter": "fixture",
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])

    report = data_catalog_quality_report(catalog.path)

    assert report["status"] == "ok"
    assert report["diagnostics"]["duplicate_ids"]["scan_id"] == 0
    assert report["diagnostics"]["metadata_modes"] == {"read_prefix": 1}
    assert report["diagnostics"]["bt7_enrichment"]["with_bt7_enrichment"] == 1
    assert report["diagnostics"]["reader_metadata_enrichment"]["per_instrument"][0]["data_files"] == 1
    assert report["warnings"] == []


def test_data_catalog_quality_report_warns_on_gaps(tmp_path):
    catalog = SQLiteDataCatalog(tmp_path / "gaps.sqlite")
    experiment = Experiment(experiment_id="exp-gap", metadata={"facility": "NCNR"})
    scan = Scan(scan_id="scan-gap", experiment_id="exp-gap", instrument="BT-7", metadata={"facility": "NCNR"})
    datafile = DataFile(
        datafile_id="df-gap",
        path="/box/bt7/no_native.bt7",
        scan_id="scan-gap",
        experiment_id="exp-gap",
        metadata={"facility": "NCNR", "instrument": "BT-7"},
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])

    report = data_catalog_quality_report(catalog.path)

    assert report["status"] == "warning"
    assert report["diagnostics"]["null_native_ids"]["data_files_native_scan_id"] == 1
    assert report["diagnostics"]["duplicate_ids"]["native_scan_key"] == 0
    assert report["diagnostics"]["bt7_enrichment"]["with_bt7_enrichment"] == 0
    assert report["diagnostics"]["reader_metadata_enrichment"]["per_instrument"][0]["with_sample_name"] == 0
    assert any("null native scan ids" in warning for warning in report["warnings"])
    assert any("BT-7 scans missing enrichment" in warning for warning in report["warnings"])


def test_data_catalog_quality_report_surfaces_path_only_records(tmp_path):
    catalog = SQLiteDataCatalog(tmp_path / "path-only.sqlite")
    experiment = Experiment(experiment_id="exp-hfir", instrument="HB-1A", metadata={"facility": "HFIR", "native_experiment_id": "1083"})
    scan = Scan(
        scan_id="scan-hfir-0059",
        experiment_id="exp-hfir",
        instrument="HB-1A",
        metadata={"facility": "HFIR", "native_scan_id": "0059", "raw_metadata": {"metadata_mode": "path_only"}},
    )
    datafile = DataFile(
        datafile_id="df-hfir-0059",
        path="/box/HB1A_exp1083_scan0059.dat",
        scan_id="scan-hfir-0059",
        experiment_id="exp-hfir",
        metadata={
            "facility": "HFIR",
            "instrument": "HB-1A",
            "native_format": "HB-1A dat",
            "native_scan_id": "0059",
            "native_experiment_id": "1083",
            "source_adapter": "fixture",
            "raw_metadata": {"metadata_mode": "path_only"},
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])

    report = data_catalog_quality_report(catalog.path)
    reader_item = report["diagnostics"]["reader_metadata_enrichment"]["per_instrument"][0]

    assert report["status"] == "warning"
    assert report["diagnostics"]["metadata_modes"] == {"path_only": 1}
    assert reader_item["metadata_modes"] == {"path_only": 1}
    assert reader_item["path_only_data_files"] == 1
    assert any("path-only metadata records: 1" == warning for warning in report["warnings"])


def test_data_catalog_quality_report_surfaces_geometry_and_reader_diagnostics(tmp_path):
    catalog = SQLiteDataCatalog(tmp_path / "compatible-ub.sqlite")
    experiment = Experiment(experiment_id="exp-murr", instrument="TRIAX", metadata={"facility": "MURR", "native_experiment_id": "153"})
    scan = Scan(
        scan_id="scan-murr-0023",
        experiment_id="exp-murr",
        instrument="TRIAX",
        scan_type="constant-Q scan",
        q_center=QPoint(h=1, k=0, l=0),
        energy_transfer=EnergyTransfer(value=UnitValue(value=0, unit="meV")),
        instrument_config=InstrumentConfig(
            fixed_energy_mode="fixed_final_energy",
            fixed_energy=UnitValue(value=14.7, unit="meV"),
            monochromator="PG(002)",
            analyzer="PG(002)",
            filters=["PG"],
            collimation="open-open-open-open",
        ),
        metadata={
            "facility": "MURR",
            "native_scan_id": "0023",
            "ub_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ub_convention": "hfir_murr_instrument_control",
            "monitor_counts": 1000,
            "count_time": 30,
        },
    )
    datafile = DataFile(
        datafile_id="df-murr-0023",
        path="/box/TRIAX_exp0153_scan0023.dat",
        checksum="a" * 64,
        scan_id=scan.scan_id,
        experiment_id=experiment.experiment_id,
        metadata={
            "facility": "MURR",
            "instrument": "TRIAX",
            "native_format": "TRIAX dat",
            "native_scan_id": "0023",
            "native_experiment_id": "153",
            "source_adapter": "fixture",
            "source_reader": "murr_reader.py",
            "raw_metadata": {
                "metadata_mode": "hydrated",
                "reader_strategy": "fixture_hydrated_reader",
                "stat_status": "available",
                "warnings": ["fixture warning"],
            },
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])

    report = data_catalog_quality_report(catalog.path)

    assert report["status"] == "warning"
    assert report["diagnostics"]["file_coverage"]["files_with_checksums"] == 1
    assert report["diagnostics"]["file_coverage"]["files_with_reader_warnings"] == 1
    assert report["diagnostics"]["metadata_modes"] == {"hydrated": 1}
    assert report["diagnostics"]["reader_strategy"]["reader_strategies"] == {"fixture_hydrated_reader": 1}
    assert report["diagnostics"]["scientific_metadata_coverage"]["ub_convention"] == 1
    assert report["diagnostics"]["scientific_metadata_coverage"]["monitor_counts"] == 1
    assert report["diagnostics"]["scientific_metadata_coverage"]["count_time"] == 1
    assert report["diagnostics"]["geometry_readiness"]["compatible_ub_scans"] == 1
    assert report["diagnostics"]["geometry_readiness"]["rejected_unsupported_ub_convention"] == 1
    assert report["diagnostics"]["quality_categories"]["ub_preserved_but_unsupported"] is True
    assert any("compatible UB scans require reviewed convention adapter" in warning for warning in report["warnings"])


def test_catalog_quality_cli_writes_report_bundle(tmp_path, capsys):
    catalog = SQLiteDataCatalog(tmp_path / "quality.sqlite")
    catalog.upsert_experiments([Experiment(experiment_id="exp-1")])
    output_dir = tmp_path / "quality-report"

    assert main(["catalog", "quality", "--catalog", str(catalog.path), "--output-dir", str(output_dir)]) == 0
    output = capsys.readouterr().out

    assert '"workflow_name"' not in output
    assert (output_dir / "catalog_quality.json").exists()
    assert (output_dir / "catalog_quality.md").exists()
    assert "TAS Data Catalog Quality Report" in (output_dir / "catalog_quality.md").read_text(encoding="utf-8")


def test_sqlite_data_catalog_has_no_paper_corpus_table(tmp_path):
    catalog = SQLiteDataCatalog(tmp_path / "bt7.sqlite")
    with sqlite3.connect(catalog.path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert "papers" not in tables
    assert "source_chunks" not in tables
    assert "paper_experiment_links" in tables


def test_import_jsonl_catalog_dir(tmp_path):
    summary = import_jsonl_catalog_dir(CoreConfig().catalog_dir, tmp_path / "imported.sqlite")
    assert summary["experiments"] >= 5
    assert summary["scans"] >= 6
    assert summary["data_files"] >= 5
    assert summary["paper_experiment_links"] >= 2


def test_catalog_bt7_directory_from_grouping_result(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_bt7_directory")
    def fake_group_bt7_scans(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            output={
                "record_count": 1,
                "group_count": 1,
                "records": [
                    {
                        "path": "/box/bt7/scan001.bt7",
                        "filename": "scan001.bt7",
                        "timestamp": 1.0,
                        "datetime": None,
                        "scan_type": "Q-space",
                        "scan_variable": "L",
                        "scan_description": "scan L",
                        "expt_id": "123",
                        "expt_name": "fixture",
                        "h": 1.0,
                        "k": 0.0,
                        "l": 0.5,
                        "e": 0.0,
                        "temp": 10.0,
                        "temp_setpoint": 10.0,
                        "fixed_e": "Ef 14.7",
                        "detector_mode": "PSD",
                        "format_type": "legacy",
                    }
                ],
            },
            tool_name="group_bt7_scans",
            tool_namespace="tas.scan_grouping",
            tool_version="0.1.0",
        )

    monkeypatch.setattr(catalog_module, "group_bt7_scans", fake_group_bt7_scans)
    result = catalog_module.catalog_bt7_directory({"root_dirs": ["/box/bt7"], "db_path": str(tmp_path / "catalog.sqlite")})

    assert result["status"] == "success"
    assert result["imported"]["experiments"] == 1
    assert result["imported"]["scans"] == 1
    assert result["imported"]["data_files"] == 1
    assert result["cataloged_record_count"] == 1
    assert result["records_per_second"] >= 0
    assert result["catalog_metrics"]["data_file_count"] == 1


def test_catalog_tas_directory_from_reader_metadata(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")

    def fake_scan_tas_directory(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            output={
                "record_count": 3,
                "records": [
                    {
                        "path": "/box/hfir/HB1A_exp1083_scan0059.dat",
                        "format_family": "hfir_hb1a_dat",
                        "facility": "HFIR",
                        "instrument": "HB-1A",
                        "native_format": "HB-1A dat",
                        "native_experiment_id": "1083",
                        "native_scan_id": "0059",
                        "source_adapter": "tas-data-readers.format_detection",
                        "source_reader": "read_hfir.py",
                        "confidence": 0.95,
                    },
                    {
                        "path": "/box/murr/TRIAX_exp0153_scan0023.dat",
                        "format_family": "murr_triax_dat",
                        "facility": "MURR",
                        "instrument": "TRIAX",
                        "native_format": "TRIAX dat",
                        "native_experiment_id": "0153",
                        "native_scan_id": "0023",
                        "source_adapter": "tas-data-readers.format_detection",
                        "source_reader": "murr_reader.py",
                        "confidence": 0.95,
                    },
                    {
                        "path": "/box/bt4/CaB10003.bt4",
                        "format_family": "ncnr_bt4_legacy",
                        "facility": "NCNR",
                        "instrument": "BT-4",
                        "native_format": "BT-4 legacy",
                        "native_experiment_id": None,
                        "native_scan_id": None,
                        "source_adapter": "tas-data-readers.format_detection",
                        "source_reader": "readncnr.py",
                        "confidence": 0.95,
                        "scan_variable": "T",
                        "sample_name": "CaB10H10/0.5396g/fltplt/dried200C/4K/2020Cu",
                        "temperature": 4.2205,
                        "q_hint": {"h": 0.0, "k": 0.0, "l": 0.0},
                        "energy_hint": {"value": 50.16, "unit": "meV", "source": "bt4_first_data_row"},
                        "raw_metadata": {"reader_strategy": "ncnr_legacy_metadata_port"},
                    },
                ],
                "format_counts": {"hfir_hb1a_dat": 1, "murr_triax_dat": 1, "ncnr_bt4_legacy": 1},
                "instrument_counts": {"HB-1A": 1, "TRIAX": 1, "BT-4": 1},
            },
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            tool_version="0.1.0",
            audit_id="audit-scan",
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan_tas_directory)
    result = catalog_module.catalog_tas_directory({"path": "/box", "db_path": str(tmp_path / "tas.sqlite")})

    assert result["status"] == "success"
    assert result["imported"]["experiments"] == 3
    assert result["imported"]["scans"] == 3
    assert result["imported"]["data_files"] == 3
    assert result["format_counts"]["hfir_hb1a_dat"] == 1
    assert result["bt7_enrichment"]["attempted"] is False

    catalog = SQLiteDataCatalog(tmp_path / "tas.sqlite")
    datafiles = catalog.list_datafiles()
    assert {item.metadata["instrument"] for item in datafiles} == {"HB-1A", "TRIAX", "BT-4"}
    bt4 = next(item for item in datafiles if item.metadata["instrument"] == "BT-4")
    assert bt4.metadata["sample_name"] == "CaB10H10/0.5396g/fltplt/dried200C/4K/2020Cu"
    assert bt4.metadata["temperature"] == 4.2205
    assert bt4.metadata["raw_metadata"]["reader_strategy"] == "ncnr_legacy_metadata_port"
    quality = data_catalog_quality_report(catalog.path)
    reader_items = quality["diagnostics"]["reader_metadata_enrichment"]["per_instrument"]
    bt4_quality = next(item for item in reader_items if item["instrument"] == "BT-4")
    assert bt4_quality["with_sample_name"] == 1
    assert bt4_quality["with_temperature"] == 1
    assert bt4_quality["with_q_hint"] == 1
    assert bt4_quality["reader_strategies"] == {"ncnr_legacy_metadata_port": 1}
    assert catalog.metrics()["data_file_count"] == 3


def test_catalog_tas_directory_uses_bt7_enrichment(tmp_path, monkeypatch):
    catalog_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_directory")

    def fake_scan_tas_directory(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            output={
                "record_count": 1,
                "records": [
                    {
                        "path": "/box/bt7/L_scan_2527.bt7",
                        "format_family": "ncnr_bt7_ice_nice",
                        "facility": "NCNR",
                        "instrument": "BT-7",
                        "native_format": "BT-7 ICE/NICE",
                        "native_experiment_id": None,
                        "native_scan_id": None,
                        "source_adapter": "tas-data-readers.format_detection",
                    }
                ],
                "format_counts": {"ncnr_bt7_ice_nice": 1},
                "instrument_counts": {"BT-7": 1},
            },
            tool_name="scan_tas_directory",
            tool_namespace="tas.data_reader",
            tool_version="0.1.0",
        )

    def fake_group_bt7_scans(arguments):
        return ToolResult(
            status=ToolResultStatus.success,
            output={
                "record_count": 1,
                "group_count": 1,
                "records": [
                    {
                        "path": "/box/bt7/L_scan_2527.bt7",
                        "scan_type": "Q-space",
                        "scan_variable": "L",
                        "scan_description": "scan L",
                        "h": 1.0,
                        "k": 0.0,
                        "l": 0.5,
                        "fixed_e": "Ef 14.7",
                        "detector_mode": "PSD",
                        "temp": 100.0,
                        "temp_setpoint": 100.0,
                        "expt_id": "25860",
                        "expt_name": "fixture",
                    }
                ],
            },
            tool_name="group_bt7_scans",
            tool_namespace="tas.scan_grouping",
            tool_version="0.1.0",
        )

    monkeypatch.setattr(catalog_module, "scan_tas_directory", fake_scan_tas_directory)
    monkeypatch.setattr(catalog_module, "group_bt7_scans", fake_group_bt7_scans)
    result = catalog_module.catalog_tas_directory({"path": "/box/bt7", "db_path": str(tmp_path / "tas.sqlite")})

    assert result["status"] == "success"
    assert result["bt7_enrichment"]["attempted"] is True
    assert result["bt7_enrichment"]["record_count"] == 1
    catalog = SQLiteDataCatalog(tmp_path / "tas.sqlite")
    scan = catalog.list_scans()[0]
    datafile = catalog.list_datafiles()[0]
    experiment = catalog.list_experiments()[0]
    assert scan.experiment_id == "tas-experiment-ncnr-bt-7-25860"
    assert scan.scan_id == "tas-scan-bt-7-25860-2527"
    assert scan.scan_type == "Q-space"
    assert scan.q_center.h == 1.0
    assert experiment.title == "fixture"
    assert datafile.metadata["native_experiment_id"] == "25860"
    assert datafile.metadata["native_scan_id"] == "2527"
    assert scan.metadata["bt7_enrichment"]["scan_variable"] == "L"
