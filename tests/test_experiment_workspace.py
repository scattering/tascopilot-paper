from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.cli import main
from tas_copilot_core.workflows.experiment_workspace import _write_psd_channel_temperature_plot, experiment_workspace


def test_experiment_workspace_composes_dashboard_planning_and_recipes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "LSMO_220_Q0p15_001.bt7", e_start=2.0, e_stop=15.0, title="2m20_Q0p15")
    _write_bt7_scan(data_root / "LSMO_220_Q0p15_background_002.bt7", e_start=2.0, e_stop=15.0, title="background 2m20_Q0p15")
    _write_psd_env_scan(data_root / "LuMnFeO3_29pt5_TdepWarm102246.bt7")
    index = _write_precedent_index(tmp_path)
    calibration_dir, calibration_registry = _write_psd_calibration_fixture(tmp_path)

    result = experiment_workspace(
        {
            "material": "La0.7Sr0.3MnO3",
            "goal": "measure spin waves and extract stiffness",
            "data_root": str(data_root),
            "precedent_index": str(index),
            "psd_calibration_dirs": [str(calibration_dir)],
            "psd_calibration_registry": str(calibration_registry),
            "background_branch": "constant_offset",
            "output_dir": str(tmp_path / "workspace"),
            "max_files": 10,
        }
    )

    assert result["status"] == "ok"
    assert result["material_identity"]["parser_backend"] in {"pymatgen", "ase", "tas_simple_decimal_fallback"}
    assert result["workspace_sections"]["dashboard"]["status"] == "ok"
    assert result["workspace_sections"]["planning_brief"]["status"] == "ok"
    assert result["workspace_sections"]["scan_recipes"]
    assert result["workspace_sections"]["scan_recipes"][0]["range_axes"] == ["E", "Q"]
    assert result["workspace_summary"]["recipe_count"] >= 1
    assert result["workspace_sections"]["review_queue"]
    assert result["workspace_sections"]["top_precedents"]
    assert result["workspace_sections"]["artifact_index"]
    assert result["workspace_sections"]["corpus_status"]["status"] == "not_supplied"
    assert result["workspace_sections"]["reduction_previews"]
    assert any(item["template_id"] == "tas.constant_q_energy_scan_preview.v0" for item in result["workspace_sections"]["reduction_previews"])
    assert result["workspace_sections"]["reduction_runs"]
    assert any(item["template_id"] == "tas.constant_q_energy_scan_preview.v0" and item["status"] == "success" for item in result["workspace_sections"]["reduction_runs"])
    assert result["workspace_sections"]["user_facing_recommendations"]["status"] == "ok"
    assert result["workspace_sections"]["user_facing_recommendations"]["privacy_boundary"]
    artifacts = result["workspace_sections"]["reduction_artifacts"]
    assert any(item["artifact_type"] == "constant_q_energy_preview" and item["status"] == "success" for item in artifacts)
    constant_q_artifact = next(item for item in artifacts if item["artifact_type"] == "constant_q_energy_preview")
    assert Path(constant_q_artifact["reader_manifest"]).exists()
    constant_q_manifest = json.loads(Path(constant_q_artifact["reader_manifest"]).read_text(encoding="utf-8"))
    assert constant_q_manifest["source_policy"] == "full_reader_output_preferred_dashboard_summary_fallback"
    assert constant_q_manifest["reader_source_count"] >= 1
    constant_q_background = json.loads(Path(constant_q_artifact["background_branch_manifest"]).read_text(encoding="utf-8"))
    assert constant_q_background["selected_background_branch"]["branch_type"] == "constant_offset"
    assert constant_q_background["selected_background_branch"]["status"] == "manual_selected_preview"
    assert constant_q_background["selected_background_branch_execution"]["status"] == "success"
    assert Path(constant_q_background["selected_background_branch_execution"]["path"]).exists()
    paired = next(branch for branch in constant_q_background["branches"] if branch["branch_type"] == "paired_background_scan")
    assert paired["review_status"] == "proposed_review_required"
    assert paired["candidate_matches"]
    assert paired["candidate_matches"][0]["status"] == "proposed_review_required"
    constant_q_csv = Path(constant_q_artifact["path"]).read_text(encoding="utf-8")
    assert "monitor_normalized_intensity" in constant_q_csv
    assert "source_object_type" in constant_q_csv
    constant_q_background_csv = Path(constant_q_artifact["background_branch_execution"]["path"]).read_text(encoding="utf-8")
    assert "background_subtracted_intensity" in constant_q_background_csv
    assert "constant_offset" in constant_q_background_csv
    assert any(item["artifact_type"] == "temperature_series_preview" and item["status"] == "success" for item in artifacts)
    temperature_artifact = next(item for item in artifacts if item["artifact_type"] == "temperature_series_preview")
    assert Path(temperature_artifact["reader_manifest"]).exists()
    temperature_manifest = json.loads(Path(temperature_artifact["reader_manifest"]).read_text(encoding="utf-8"))
    assert temperature_manifest["source_policy"] == "full_reader_output_preferred_dashboard_summary_fallback"
    temperature_background = json.loads(Path(temperature_artifact["background_branch_manifest"]).read_text(encoding="utf-8"))
    assert temperature_background["selected_background_branch"]["branch_type"] == "constant_offset"
    assert temperature_background["selected_background_branch_execution"]["status"] == "success"
    assert Path(temperature_background["selected_background_branch_execution"]["path"]).exists()
    assert any(item["artifact_type"] == "psd_channel_temperature_preview" and item["status"] == "success" for item in artifacts)
    psd_artifact = next(item for item in artifacts if item["artifact_type"] == "psd_channel_temperature_preview")
    assert psd_artifact["source_object_type"] == "reader_output_scan_file"
    assert psd_artifact["psd_calibration"]["status"] == "success"
    assert psd_artifact["q_temperature_status"] == "formula_preview_available_from_reviewed_channel_to_a4"
    assert psd_artifact["plot_y_axis"] == "temperature_K_sorted_ascending"
    assert psd_artifact["temperature_sort_applied"] is True
    assert Path(psd_artifact["reader_manifest"]).exists()
    manifest = json.loads(Path(psd_artifact["reader_manifest"]).read_text(encoding="utf-8"))
    assert manifest["source_object_type"] == "reader_output_scan_file"
    assert manifest["psd_channel_count"] == 4
    assert manifest["calibration_status"] == "success"
    readiness = result["workspace_sections"]["reduction_readiness"]["psd_channel_temperature"]
    assert readiness["status"] == "success"
    assert readiness["calibration_status"] == "success"
    assert readiness["reader_manifest"] == psd_artifact["reader_manifest"]
    assert result["workspace_sections"]["reduction_readiness"]["constant_q_energy"]["reader_manifest"] == constant_q_artifact["reader_manifest"]
    assert result["workspace_sections"]["reduction_readiness"]["constant_q_energy"]["selected_background_branch"]["status"] == "manual_selected_preview"
    assert result["workspace_sections"]["reduction_readiness"]["constant_q_energy"]["background_branch_execution"]["status"] == "success"
    assert result["workspace_sections"]["reduction_readiness"]["order_parameter_temperature"]["reader_manifest"] == temperature_artifact["reader_manifest"]
    assert result["workspace_sections"]["reduction_readiness"]["order_parameter_temperature"]["background_branch_manifest"] == temperature_artifact["background_branch_manifest"]
    assert result["workspace_sections"]["reduction_readiness"]["order_parameter_temperature"]["background_branch_execution"]["status"] == "success"
    csv_text = Path(psd_artifact["path"]).read_text(encoding="utf-8")
    assert "two_theta_deg" in csv_text
    assert "efficiency_monitor_normalized_counts" in csv_text
    assert result["workspace_sections"]["alignment_status"]["status"] == "not_supplied"
    assert result["workspace_sections"]["scan_recipes"][0]["recipe_rank_score"] >= result["workspace_sections"]["scan_recipes"][-1]["recipe_rank_score"]
    replay_plan = result["workspace_sections"]["nice_replay_training_plan"]
    assert replay_plan["workflow_boundary"] == "simulation_only_training_harness"
    assert replay_plan["recipe_count"] >= 1
    assert Path(replay_plan["artifact_path"]).exists()
    assert replay_plan["recipes"][0]["translation_status"] == "node_hints_only_review_required"
    assert Path(result["artifacts"]["html"]).exists()
    assert Path(result["artifacts"]["json"]).exists()
    html = Path(result["artifacts"]["html"]).read_text(encoding="utf-8")
    assert "Experiment Workspace" in html
    assert "Dashboard Entry Point" in html
    assert "Review Queue" in html
    assert "Top Precedent Patterns" in html
    assert "recipeSearch" in html
    assert "Corpus Evidence" in html
    assert "Reduction DAG Preview" in html
    assert "Preview Execution" in html
    assert "Materialized Preview Artifacts" in html
    assert "User-Facing Recommendation Summary" in html
    assert "Reduction Readiness" in html
    assert "Reader manifest" in html
    assert "Background branches" in html
    assert "Selected background preview" in html
    assert "Alignment and Structure Factors" in html
    assert "Rank Rationale" in html
    assert "Header-Derived Scan Recipes" in html
    assert "Private precedent pattern" in html
    assert "NICE Replay Training Harness" in html


def test_experiment_workspace_cli(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "LSMO_220_Q0p15_001.bt7", e_start=2.0, e_stop=15.0, title="2m20_Q0p15")

    code = main(
        [
            "workflow",
            "experiment-workspace",
            "--material",
            "La0.7Sr0.3MnO3",
            "--goal",
            "measure spin waves",
            "--data-root",
            str(data_root),
            "--precedent-index",
            str(_write_precedent_index(tmp_path)),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads((tmp_path / "out" / "experiment_workspace.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "experiment-workspace"
    assert payload["artifacts"]["html"].endswith("experiment_workspace.html")
    assert (tmp_path / "out" / "experiment_workspace.html").exists()


def test_psd_channel_temperature_plot_uses_science_temperature_order(tmp_path: Path) -> None:
    result = _write_psd_channel_temperature_plot(
        tmp_path / "psd.png",
        {
            "source_object_id": "decreasing-temperature-scan",
            "temperature_values": [100.0, 50.0, 10.0],
            "psd_counts": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        },
    )

    assert result["plot_status"] == "success"
    assert result["plot_y_axis"] == "temperature_K_sorted_ascending"
    assert result["temperature_sort_applied"] is True
    assert result["plot_temperature_range"] == [10.0, 100.0]
    assert (tmp_path / "psd.png").exists()


def _write_precedent_index(tmp_path: Path) -> Path:
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "experiment_id": "exp-lsmo",
                "title": "Summer school LSMO spin waves",
                "datafile_count": 4,
                "scan_count": 4,
                "path_clues": {"top_filename_prefixes": [{"value": "LSMO_220_Q0p15"}]},
                "hydrated_clue_summary": {
                    "top_values": {
                        "fixed_e": [{"value": "Ef 14.7", "count": 4}],
                        "temp_mean": [{"value": "300.0", "count": 2}, {"value": "345.0", "count": 2}],
                        "scan_variable": [{"value": "E", "count": 4}],
                        "scan_description": [
                            {
                                "value": "Scan:SubID=18546:JType=VECTOR:Fixed=1:FixedE=14.7:Npts=27:Counts=145000.0:"
                                "DetectorType=Detector:CountType=Monitor:Filename=LSMO_220_Q0p15_:"
                                "Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0 s:Range=E=2.0 15.0 s:Title=2m20_Q0p15"
                            }
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def _write_psd_calibration_fixture(tmp_path: Path) -> tuple[Path, Path]:
    calibration_dir = tmp_path / "PSD"
    calibration_dir.mkdir()
    (calibration_dir / "accepted_workspace.dat").write_text(
        "\n".join(
            [
                "#Created on: Thu, 01 Jan 2015 00:00:00",
                "#Efficiency  A4  Ef",
                "1.0 -0.3 0",
                "1.0 -0.1 0",
                "0.5 0.1 0",
                "1.5 0.3 0",
            ]
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "psd_registry.yaml"
    registry.write_text(
        """
psd_calibrations:
  - calibration_id: workspace-accepted
    instrument: BT-7
    detector_mode: PSDDiff
    date_range:
      start: "2015-01-01"
      end: "2015-01-02"
    status: accepted
    reviewed_by: fixture-reviewer
    reviewed_at: "2026-05-24"
    channel_mapping:
      type: bt7_psd_dat_relative_a4
      source_filename: accepted_workspace.dat
      row_count: 4
      sign_convention: plus_relative_a4
      channel_to_a4_policy: "two_theta = A4 + relative_a4[channel]"
      mask_policy:
        masked_one_based_channels: [3]
    detector_geometry:
      analyzer_detector_translation: 200
    validation_evidence:
      - evidence_type: fixture
        summary: accepted fixture calibration
        score: 1.0
""",
        encoding="utf-8",
    )
    return calibration_dir, registry


def _write_bt7_scan(path: Path, *, e_start: float, e_stop: float, title: str) -> None:
    rows = []
    for index, energy in enumerate([e_start, (e_start + e_stop) / 2.0, e_stop]):
        rows.append(f"1.85 {energy:.3f} 14.7 14.7 100000 {100 + index * 10} 300")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2015-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      18546",
                "#ExptName    Summer school LSMO",
                "#ExptDetails H02 high temp displex for x=0.3 LSMO",
                "#ScanType   VECTOR",
                "#Scan        4 E",
                f"#ScanDescr   Scan:SubID=18546:JType=VECTOR:Fixed=1:FixedE=14.7:Npts=3:Counts=145000.0:DetectorType=Detector:CountType=Monitor:Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0 s:Range=E={e_start} {e_stop} s:Title={title}",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                "#Columns QX E Ef Ei Monitor Detector Temp",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_psd_env_scan(path: Path) -> None:
    rows = [
        "0 0 0 0 29.5 100 2.5 1000 300 10 11 12 13",
        "0 0 0 0 29.5 100 5.0 1000 330 20 22 24 26",
        "0 0 0 0 29.5 100 7.5 1000 360 30 33 36 39",
    ]
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2015-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      18546",
                "#ExptName    Synthetic PSD fixture",
                "#ScanType   ENV",
                "#Scan        7 Temp",
                "#ScanDescr   Scan:SubID=18546:JType=ENV:Npts=3:Counts=1000.0:DetectorType=Detector:CountType=Monitor:Range=Temp=2.5 7.5 s:Title=TscanInc",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#Columns QX QY QZ E A4 Time Temp Monitor Detector PSDC00 PSDC01 PSDC02 PSDC03",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
