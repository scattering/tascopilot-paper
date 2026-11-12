from tas_copilot_contracts import DataFile, EnergyTransfer, Experiment, InstrumentConfig, QPoint, Scan, UnitValue
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.catalog_report import catalog_report


def _catalog(path):
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(
        experiment_id="exp-geometry",
        instrument="BT-7",
        title="BT-7 spin-wave constant-Q geometry fixture",
        metadata={"facility": "NCNR", "native_experiment_id": "synthetic"},
    )
    scan = Scan(
        scan_id="scan-geometry-001",
        experiment_id="exp-geometry",
        instrument="BT-7",
        scan_type="constant-Q scan",
        physics_target="spin-wave dispersion",
        q_center=QPoint(h=1.0, k=0.0, l=0.0),
        energy_transfer=EnergyTransfer(value=UnitValue(value=0.0, unit="meV")),
        instrument_config=InstrumentConfig(
            fixed_energy_mode="fixed_final_energy",
            fixed_energy=UnitValue(value=14.7, unit="meV"),
            monochromator="PG(002)",
            analyzer="PG(002)",
            collimation="40-40-40-80",
        ),
        metadata={
            "facility": "NCNR",
            "native_scan_id": "synthetic-001",
            "bt7_enrichment": {"expt_id": "synthetic", "scan_variable": "E"},
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
        },
    )
    datafile = DataFile(
        datafile_id="df-geometry-001",
        path="/synthetic/geometry_scan.bt7",
        scan_id=scan.scan_id,
        experiment_id=experiment.experiment_id,
        format="ncnr_bt7_ice_nice",
        metadata={
            "facility": "NCNR",
            "instrument": "BT-7",
            "native_format": "BT-7 ICE/NICE",
            "native_scan_id": "synthetic-001",
            "native_experiment_id": "synthetic",
            "source_adapter": "fixture",
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    return catalog


def test_catalog_report_retrieves_catalog_records_and_ontology_concepts(tmp_path):
    db_path = tmp_path / "catalog.sqlite"
    _catalog(db_path)

    report = catalog_report({"catalog": str(db_path), "query": "BT-7 constant-Q spin-wave fixed final energy", "limit": 5})

    assert report["status"] == "ok"
    assert report["catalog_summary"]["scans"] == 1
    assert report["quality_status"] == "ok"
    assert "constant_q_scan" in {item["id"] for item in report["matched_concepts"]}
    assert report["results"]["scans"][0]["source_id"] == "scan-geometry-001"
    assert "constant_q_scan" in report["results"]["scans"][0]["matched_concepts"]
    assert report["results"]["data_files"][0]["summary"]["native_scan_id"] == "synthetic-001"


def test_catalog_report_cli_json(tmp_path, capsys):
    db_path = tmp_path / "catalog.sqlite"
    _catalog(db_path)

    assert main(["workflow", "catalog-report", "--catalog", str(db_path), "--query", "constant-Q spin-wave"]) == 0
    output = capsys.readouterr().out

    assert '"workflow_name": "catalog-report"' in output
    assert '"source_id": "scan-geometry-001"' in output


def test_catalog_report_writes_json_and_markdown_artifacts(tmp_path):
    db_path = tmp_path / "catalog.sqlite"
    output_dir = tmp_path / "report"
    _catalog(db_path)

    report = catalog_report(
        {
            "catalog": str(db_path),
            "query": "BT-7 constant-Q spin-wave fixed final energy",
            "limit": 5,
            "output_dir": str(output_dir),
        }
    )

    json_path = output_dir / "catalog_report.json"
    md_path = output_dir / "catalog_report.md"
    assert report["artifacts"] == {"json": str(json_path), "markdown": str(md_path)}
    assert json_path.exists()
    assert md_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "# TAS Catalog Report" in markdown
    assert "`constant_q_scan`" in markdown
    assert "`scan-geometry-001`" in markdown
    assert "Safe Next Actions" in markdown
    assert "Inspect top-ranked scans" in markdown


def test_catalog_report_cli_writes_artifacts(tmp_path, capsys):
    db_path = tmp_path / "catalog.sqlite"
    output_dir = tmp_path / "report"
    _catalog(db_path)

    assert (
        main(
            [
                "workflow",
                "catalog-report",
                "--catalog",
                str(db_path),
                "--query",
                "constant-Q spin-wave",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (output_dir / "catalog_report.json").exists()
    assert (output_dir / "catalog_report.md").exists()
