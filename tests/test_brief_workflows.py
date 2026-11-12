import json

from tas_copilot_contracts import (
    DataFile,
    EnergyTransfer,
    Experiment,
    InstrumentConfig,
    PaperExperimentLink,
    QPoint,
    Scan,
    UnitValue,
)
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.experiment_brief import experiment_brief
from tas_copilot_core.workflows.scan_brief import scan_brief


def _scan(scan_id: str, experiment_id: str = "exp-brief", *, sparse: bool = False) -> Scan:
    if sparse:
        return Scan(scan_id=scan_id, experiment_id=experiment_id, instrument="HB-1A", metadata={"raw_metadata": {"metadata_mode": "path_only"}})
    return Scan(
        scan_id=scan_id,
        experiment_id=experiment_id,
        instrument="BT-7",
        scan_type="constant-Q scan",
        physics_target="spin-wave dispersion",
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


def _catalog(path):
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(
        experiment_id="exp-brief",
        instrument="BT-7",
        title="Brief fixture experiment",
        metadata={"facility": "NCNR", "native_experiment_id": "fixture"},
    )
    scans = [_scan("scan-brief-001"), _scan("scan-brief-002"), _scan("scan-sparse", sparse=True)]
    datafiles = [
        DataFile(
            datafile_id="df-brief-001",
            path="/synthetic/scan001.bt7",
            scan_id="scan-brief-001",
            experiment_id="exp-brief",
            format="ncnr_bt7_ice_nice",
            metadata={"facility": "NCNR", "instrument": "BT-7", "native_scan_id": "001"},
        )
    ]
    links = [
        PaperExperimentLink(
            paper_id="paper-fixture",
            experiment_id="exp-brief",
            confidence=0.8,
            evidence=["fixture match"],
            mapping_method="fixture",
            status="candidate",
        )
    ]
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans(scans)
    catalog.upsert_datafiles(datafiles)
    catalog.upsert_paper_experiment_links(links)
    return catalog


def test_scan_brief_composes_catalog_completeness_similarity_and_geometry(tmp_path):
    db_path = tmp_path / "catalog.sqlite"
    output_dir = tmp_path / "briefs"
    _catalog(db_path)

    report = scan_brief({"catalog": str(db_path), "scan_id": "scan-brief-001", "output_dir": str(output_dir)})

    assert report["workflow_name"] == "scan-brief"
    assert report["scan"]["scan_id"] == "scan-brief-001"
    assert report["data_files"][0]["datafile_id"] == "df-brief-001"
    assert report["metadata_completeness"]["eligible_for_resolution_check"] is True
    assert report["similar_scans"][0]["scan_id"] == "scan-brief-002"
    assert report["geometry_check"]["workflow_name"] == "check-geometry"
    assert report["artifacts"]["json"] == str(output_dir / "scan-brief-001_scan_brief.json")
    assert (output_dir / "scan-brief-001_scan_brief.md").exists()


def test_scan_brief_cli_and_missing_scan(tmp_path, capsys):
    db_path = tmp_path / "catalog.sqlite"
    _catalog(db_path)

    assert main(["workflow", "scan-brief", "--catalog", str(db_path), "--scan-id", "missing"]) == 0
    output = capsys.readouterr().out
    assert '"status": "missing_input"' in output
    assert '"scan_not_found"' in output


def test_experiment_brief_summarizes_scans_datafiles_and_mappings(tmp_path):
    db_path = tmp_path / "catalog.sqlite"
    output_dir = tmp_path / "briefs"
    _catalog(db_path)

    report = experiment_brief({"catalog": str(db_path), "experiment_id": "exp-brief", "output_dir": str(output_dir)})

    assert report["workflow_name"] == "experiment-brief"
    assert report["experiment"]["experiment_id"] == "exp-brief"
    assert report["scan_count"] == 3
    assert report["data_file_count"] == 1
    assert report["paper_mapping_count"] == 1
    assert report["best_scan_id"] in {"scan-brief-001", "scan-brief-002"}
    assert report["metadata_completeness_summary"]["path_only_scans"] == 1
    assert report["paper_experiment_links"][0]["paper_id"] == "paper-fixture"
    assert report["artifacts"]["markdown"] == str(output_dir / "exp-brief_experiment_brief.md")
    assert json.loads((output_dir / "exp-brief_experiment_brief.json").read_text(encoding="utf-8"))["workflow_name"] == "experiment-brief"


def test_experiment_brief_cli(tmp_path, capsys):
    db_path = tmp_path / "catalog.sqlite"
    _catalog(db_path)

    assert main(["workflow", "experiment-brief", "--catalog", str(db_path), "--experiment-id", "exp-brief"]) == 0
    output = capsys.readouterr().out
    assert '"workflow_name": "experiment-brief"' in output
    assert '"paper_mapping_count": 1' in output
