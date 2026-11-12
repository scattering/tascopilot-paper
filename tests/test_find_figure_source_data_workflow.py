from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_contracts import DataFile, Experiment, Scan
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.find_figure_source_data import (
    build_interpretations,
    composition_aliases,
    find_figure_source_data,
    load_figure_source_query,
    panel_sample_warnings,
)


ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "examples" / "figure_queries" / "lu_femn_o3_bt7_psd_2013_panel_k.yaml"


def test_query_file_parses_and_detects_panel_sample_mismatch() -> None:
    query = load_figure_source_query({"query_file": str(QUERY_FILE)})

    assert query.dates == ["2013-05-17", "2013-06-01"]
    warnings = panel_sample_warnings(query)
    assert warnings
    assert "panel (j)" in warnings[0]
    assert "panel (k)" in warnings[0]


def test_cli_dotted_dates_and_alias_generation() -> None:
    query = load_figure_source_query(
        {
            "instrument": "BT-7",
            "mode": "PSD",
            "sample_type": "powder",
            "material_family": "LuFe_{1-x}Mn_xO3",
            "sample": "LuFe0.6Mn0.4O3",
            "target_panel": "k",
            "target_panel_xmn": 0.5,
            "alternate_panel": "j",
            "alternate_panel_xmn": 0.4,
            "date": ["5.172013", "6.1.2013"],
            "q_min": 1.10,
            "q_max": 1.45,
            "temperature_min": 0,
            "temperature_max": 200,
        }
    )

    assert query.dates == ["2013-05-17", "2013-06-01"]
    aliases = composition_aliases(query.material_family, 0.4, query.user_sample_formula)
    assert "LuFe0.6Mn0.4O3" in aliases
    assert "x_Mn = 0.4" in aliases
    assert "xMn=0.4" in aliases
    assert any(item["panel_id"] == "j" and item["x_mn"] == 0.4 for item in build_interpretations(query))


def test_candidate_scoring_grouping_and_synthetic_reconstruction(tmp_path: Path) -> None:
    catalog = _synthetic_catalog(tmp_path / "figure.sqlite", hydrated=True, with_psd_map=True)
    output_dir = tmp_path / "report"

    report = find_figure_source_data(
        {
            "query_file": str(QUERY_FILE),
            "catalog": str(catalog.path),
            "output_dir": str(output_dir),
            "attempt_reconstruction": True,
            "top_k": 10,
        }
    )

    assert report["status"] == "success"
    assert report["panel_sample_warnings"]
    assert report["decision_summary"]["top_candidate_id"]
    assert "calibration" in report["decision_summary"]["calibration_blocker"].lower()
    assert any("target_panel_k" in item["interpretation"] for item in report["candidates"])
    assert any("user_sample_panel_j" in item["interpretation"] for item in report["candidates"])
    top = report["candidates"][0]
    assert top["candidate_type"] == "scan_group"
    assert top["total_score"] >= 0.7
    assert top["reconstruction"]["status"] == "success"
    assert "annotated_q_1.24" in top["reconstruction"]["figure_comparison"]["matched_features"]
    assert (output_dir / "figure_source_report.md").exists()
    assert (output_dir / "figure_source_report.json").exists()
    assert (output_dir / "candidate_table.csv").exists()
    assert (output_dir / "figure_source_review_template.jsonl").exists()
    assert report["artifacts"]["figure_source_review_template_jsonl"] == str(output_dir / "figure_source_review_template.jsonl")
    review_rows = [
        json.loads(line)
        for line in (output_dir / "figure_source_review_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["candidate_id"] == top["candidate_id"] and row["source_workflow"] == "find-figure-source-data" for row in review_rows)
    assert "requires hydrated reconstruction" not in report["safe_conclusion"].lower()


def test_path_only_reconstruction_fails_closed(tmp_path: Path) -> None:
    catalog = _synthetic_catalog(tmp_path / "path-only.sqlite", hydrated=False, with_psd_map=False)
    report = find_figure_source_data(
        {
            "query_file": str(QUERY_FILE),
            "catalog": str(catalog.path),
            "output_dir": str(tmp_path / "path-only-report"),
            "attempt_reconstruction": True,
        }
    )

    assert report["status"] == "success"
    reconstruction = report["candidates"][0]["reconstruction"]
    assert reconstruction["status"] == "missing_input"
    assert "hydrated_data" in reconstruction["missing_fields"]
    assert "guess" not in reconstruction["warnings"][0].lower()


def test_elastic_query_rejects_nonzero_energy_false_positive(tmp_path: Path) -> None:
    catalog = _synthetic_catalog(tmp_path / "inelastic.sqlite", hydrated=True, with_psd_map=True, include_inelastic=True)
    output_dir = tmp_path / "inelastic-report"

    report = find_figure_source_data(
        {
            "query_file": str(QUERY_FILE),
            "catalog": str(catalog.path),
            "output_dir": str(output_dir),
            "top_k": 10,
        }
    )

    rejected = report["rejected_candidates"]
    assert rejected
    assert any("nonzero energy transfer" in item["rejection_reason"] for item in rejected)
    assert all("scan-lu-femn-3mev" not in candidate["scan_ids"] for candidate in report["candidates"])
    assert (output_dir / "rejected_candidates.json").exists()
    review_rows = [
        json.loads(line)
        for line in (output_dir / "figure_source_review_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row["candidate_status"] == "rejected_by_workflow"
        and "nonzero energy transfer" in row["rejected_reason"]
        for row in review_rows
    )
    assert "Rejected Candidates" in (output_dir / "figure_source_report.md").read_text(encoding="utf-8")


def test_missing_catalog_returns_report_and_next_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing"
    report = find_figure_source_data({"query_file": str(QUERY_FILE), "catalog": str(tmp_path / "missing.sqlite"), "output_dir": str(output_dir)})

    assert report["status"] == "missing_input"
    assert (output_dir / "figure_source_report.md").exists()
    assert "data root" in " ".join(report["next_actions"]).lower() or "--data-root" in " ".join(report["next_actions"])


def test_cli_find_figure_source_data_writes_report(tmp_path: Path, capsys) -> None:
    catalog = _synthetic_catalog(tmp_path / "cli.sqlite", hydrated=True, with_psd_map=True)
    output_dir = tmp_path / "cli-report"

    assert (
        main(
            [
                "workflow",
                "find-figure-source-data",
                "--query-file",
                str(QUERY_FILE),
                "--catalog",
                str(catalog.path),
                "--output-dir",
                str(output_dir),
                "--attempt-reconstruction",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"workflow_name": "find-figure-source-data"' in captured.out
    assert (output_dir / "figure_source_report.md").exists()


def _synthetic_catalog(path: Path, hydrated: bool, with_psd_map: bool, include_inelastic: bool = False) -> SQLiteDataCatalog:
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(
        experiment_id="exp-lu-femn-2013",
        instrument="BT-7",
        title="Disseler Ratcliff Lynn Cheong LuFe_{1-x}Mn_xO3 PSD powder map",
        start_date="2013-05-17",
        end_date="2013-06-01",
        metadata={"experimenters": ["Disseler", "Ratcliff", "Lynn", "Cheong"], "sample_type": "powder"},
    )
    psd_map = {
        "q_axis": [1.10, 1.20, 1.24, 1.33, 1.45],
        "temperature_axis": [0, 50, 100, 150, 200],
        "intensity": [
            [0.1, 0.6, 0.9, 0.2, 0.1],
            [0.1, 0.7, 1.0, 0.3, 0.1],
            [0.1, 0.5, 0.8, 0.2, 0.1],
            [0.1, 0.3, 0.6, 0.2, 0.1],
            [0.1, 0.2, 0.4, 0.2, 0.1],
        ],
    }
    common_metadata = {
        "date": "2013-05-17",
        "measurement_mode": "PSD",
        "detector": "position sensitive detector",
        "sample_type": "powder",
        "sample_formula": "LuFe0.6Mn0.4O3",
        "x_Mn": 0.4,
        "q_range": {"min": 1.10, "max": 1.45},
        "temperature_range": {"min": 0, "max": 200},
        "temperature_values": [0, 50, 100, 150, 200],
        "experimenters": ["Disseler", "Ratcliff", "Lynn", "Cheong"],
    }
    if with_psd_map:
        common_metadata["psd_map"] = psd_map
    scans = [
        Scan(
            scan_id="scan-lu-femn-psd-001",
            experiment_id=experiment.experiment_id,
            instrument="BT-7",
            scan_type="PSD temperature scan",
            metadata=common_metadata,
        ),
        Scan(
            scan_id="scan-lu-femn-psd-002",
            experiment_id=experiment.experiment_id,
            instrument="BT-7",
            scan_type="PSD temperature scan",
            metadata={**common_metadata, "date": "2013-06-01"},
        ),
    ]
    if include_inelastic:
        scans.append(
            Scan(
                scan_id="scan-lu-femn-3mev",
                experiment_id=experiment.experiment_id,
                instrument="BT-7",
                scan_type="single detector inelastic line scan",
                metadata={
                    **common_metadata,
                    "measurement_mode": "single detector",
                    "detector": "single detector",
                    "sample_formula": "LuFe0.6Mn0.4O3",
                    "energy_transfer": {"value": 3.0, "unit": "meV"},
                    "energy_range": {"min": 3.0, "max": 3.0, "unit": "meV"},
                    "filename": "LuFeMnO3_Tdep_3meV102300.bt7",
                },
            )
        )
    mode = "hydrated" if hydrated else "path_only"
    status = "hydrated" if hydrated else "path_only"
    datafiles = [
        DataFile(
            datafile_id=f"df-{scan.scan_id}",
            path=f"/synthetic/{scan.scan_id}.bt7",
            scan_id=scan.scan_id,
            experiment_id=experiment.experiment_id,
            format="BT7",
            metadata={"metadata_mode": mode, "hydration_status": status, "instrument": "BT-7", "native_format": "BT7"},
        )
        for scan in scans
    ]
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans(scans)
    catalog.upsert_datafiles(datafiles)
    return catalog
