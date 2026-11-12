from __future__ import annotations

import json
import py_compile
from pathlib import Path

from tas_copilot_contracts import ReviewLabel, ToolResult, ToolResultStatus
from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.experiment_dashboard import build_experiment_dashboard, dag_run_replay, draft_overnight_report, experiment_review, fit_order_parameter, inelastic_series_report, order_parameter_triage, overnight_scan_summary, survey_inelastic_spurion_evidence
from tas_copilot_core.workflows.trajectory_clustering import cluster_order_parameter_trajectories, order_parameter_trajectory_report, plot_order_parameter_candidate_scans


def test_experiment_dashboard_builds_static_and_dash_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, title="E6 scan", detector=[100, 101, 102, 500, 103, 102, 100])
    _write_bt7_scan(data_root / "elastic_002.bt7", e_value=0.0, title="Bragg scan", detector=[100, 130, 180, 220, 190, 140, 100])

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "dashboard")})

    assert result["status"] == "ok"
    assert result["scan_count"] == 2
    assert result["inelastic_scan_count"] == 1
    assert Path(result["generated_artifacts"]["dashboard_html"]).exists()
    assert Path(result["generated_artifacts"]["dash_app"]).exists()
    assert Path(result["generated_artifacts"]["scan_table_csv"]).exists()
    assert Path(result["generated_artifacts"]["dashboard_manifest"]).exists()
    assert Path(result["generated_artifacts"]["overlay_layers_csv"]).exists()
    assert Path(result["generated_artifacts"]["diagnostic_recommendations_csv"]).exists()
    dashboard_html = Path(result["generated_artifacts"]["dashboard_html"]).read_text(encoding="utf-8")
    assert "scan-search" in dashboard_html
    assert "kind-filter" in dashboard_html
    assert "plot-scan" in dashboard_html
    assert "drawScan" in dashboard_html
    assert "Reload local JSON" in dashboard_html
    assert "Copy scan link" in dashboard_html
    assert "auto-refresh-status" in dashboard_html
    assert "selected-scan-detail" in dashboard_html
    assert "data-search" in dashboard_html
    assert "data-scan-id" in dashboard_html
    assert "Experiment Overlay Hooks" in dashboard_html
    assert result["spurion_likelihood_candidates"]
    assert result["spurion_likelihood_candidates"][0]["candidate_type"] == "narrow_inelastic_feature"
    assert result["dashboard_scope"]["scope"] == "single_experiment_or_data_root"
    assert result["spurion_ranking_diagnostics"]["probability_status"] == "heuristic_not_calibrated"


def test_experiment_dashboard_max_files_zero_means_no_explicit_cap(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "scan_001.bt7", e_value=0.0, title="scan 1", detector=[100, 120, 100])
    _write_bt7_scan(data_root / "scan_002.bt7", e_value=0.0, title="scan 2", detector=[100, 140, 100])

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "dashboard"), "max_files": 0})

    assert result["status"] == "ok"
    assert result["scan_count"] == 2


def test_survey_inelastic_spurion_evidence_reports_candidates(tmp_path: Path) -> None:
    data_root = tmp_path / "2015" / "12345" / "data"
    data_root.mkdir(parents=True)
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=9.0, title="E9 scan with cryostat", detector=[100, 101, 102, 700, 103, 101, 100], details="CCR PG filter")

    result = survey_inelastic_spurion_evidence({"data_root": str(tmp_path), "output_dir": str(tmp_path / "survey"), "post_year": 2010})

    assert result["status"] == "ok"
    assert result["inelastic_scan_count"] == 1
    assert result["candidate_count"] >= 1
    candidate = result["candidates"][0]
    assert candidate["likelihood_score"] > 0
    assert candidate["recommended_checks"]
    assert result["ranking_diagnostics"]["probability_status"] == "heuristic_not_calibrated"


def test_experiment_dashboard_writes_overlay_pack_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "scan.bt7", e_value=3.0, title="E3", detector=[100, 100, 400, 99, 98])

    result = build_experiment_dashboard(
        {
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "dashboard"),
            "overlay_packs": ["bt7_common"],
            "materials_in_beam": {"mount": ["Al", "Cu"]},
        }
    )

    assert result["overlay_summary"]["overlay_count"] >= 6
    manifest = json.loads(Path(result["generated_artifacts"]["dashboard_manifest"]).read_text(encoding="utf-8"))
    assert manifest["scope"]["overlay_policy"].startswith("experiment_specific")
    overlay_csv = Path(result["generated_artifacts"]["overlay_layers_csv"]).read_text(encoding="utf-8")
    assert "Aluminum in beam" in overlay_csv


def test_experiment_dashboard_cli_writes_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "scan.bt7", e_value=3.0, title="E3", detector=[100, 100, 400, 99, 98])
    output_dir = tmp_path / "out"

    code = main(["workflow", "experiment-dashboard", "--data-root", str(data_root), "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "experiment_dashboard_report.json").exists()
    payload = json.loads((output_dir / "dashboard_data.json").read_text(encoding="utf-8"))
    assert payload["scan_count"] == 1
    assert (output_dir / "refresh_dashboard.sh").exists()
    py_compile.compile(str(output_dir / "app.py"), doraise=True)


def test_experiment_review_composes_dashboard_triage_and_gallery(tmp_path: Path) -> None:
    data_root = tmp_path / "review_data"
    data_root.mkdir()
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=temp < 90, amplitude=280 - temp)
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, title="E6 scan CCR", detector=[100, 101, 102, 700, 102, 101, 100], details="CCR PG")

    result = experiment_review({"data_root": str(data_root), "output_dir": str(tmp_path / "review"), "max_files": 0, "top_k": 3, "allow_rebin": True})

    assert result["status"] == "ok"
    assert result["workflow_results"]["dashboard"]["status"] == "ok"
    assert result["workflow_results"]["overnight_summary"]["status"] == "ok"
    assert result["workflow_results"]["order_parameter_triage"]["status"] == "ok"
    assert result["workflow_results"]["order_parameter_followup"]["status"] == "ok"
    assert result["ranked_next_steps"]
    assert result["calibration_diagnostics"]["human_review_required"] is True
    assert any(step["action_type"] == "review_order_parameter_gallery" for step in result["ranked_next_steps"])
    assert "limitations" in result["fit_handoff"]
    assert Path(result["generated_artifacts"]["order_parameter_gallery"]).exists()
    assert (tmp_path / "review" / "experiment_review.md").exists()


def test_experiment_review_cli_writes_report(tmp_path: Path) -> None:
    data_root = tmp_path / "review_data"
    data_root.mkdir()
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=False, amplitude=280 - temp)
    output_dir = tmp_path / "review"

    code = main(["workflow", "experiment-review", "--data-root", str(data_root), "--output-dir", str(output_dir), "--max-files", "0", "--top-k", "2", "--allow-rebin"])

    assert code == 0
    payload = json.loads((output_dir / "experiment_review.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "experiment-review"
    assert payload["probability_model"]["status"] == "heuristic_priors"
    assert payload["probability_model"]["calibration_status"] == "not_statistically_calibrated"
    assert payload["calibration_diagnostics"]["human_review_required"] is True
    assert Path(payload["generated_artifacts"]["review_outcome_template_jsonl"]).exists()
    assert Path(payload["generated_artifacts"]["review_outcomes_sqlite"]).exists()
    store_summary = ReviewOutcomeSQLiteStore(payload["generated_artifacts"]["review_outcomes_sqlite"]).summary()
    assert store_summary["label_count"] >= 1
    assert store_summary["by_source_workflow"]["order-parameter-triage"] >= 1
    first_label = json.loads(Path(payload["generated_artifacts"]["review_outcome_template_jsonl"]).read_text(encoding="utf-8").splitlines()[0])
    assert first_label["object_type"] == "order_parameter_candidate"
    assert first_label["dag_run_id"]
    assert first_label["cache_key"]
    assert first_label["correction_status_snapshot"]
    assert first_label["workflow_version"]
    assert first_label["tool_versions"]
    assert "reviewed_at" in first_label
    assert payload["calibration_diagnostics"]["review_label_store_diagnostics"]["order_parameter"]["status"] == "not_provided"
    assert payload["ranked_next_steps"]


def test_experiment_review_can_skip_spurion_branch(tmp_path: Path) -> None:
    data_root = tmp_path / "review_data"
    data_root.mkdir()
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=False, amplitude=280 - temp)
    output_dir = tmp_path / "review"

    code = main(["workflow", "experiment-review", "--data-root", str(data_root), "--output-dir", str(output_dir), "--max-files", "0", "--top-k", "2", "--allow-rebin", "--skip-spurion"])

    assert code == 0
    payload = json.loads((output_dir / "experiment_review.json").read_text(encoding="utf-8"))
    assert payload["workflow_results"]["spurion_survey"]["status"] == "skipped"
    assert payload["spurion_candidate_count"] == 0
    assert not (output_dir / "spurion_survey").exists()


def test_bt7_corpus_candidate_pass_runs_alignment_spurion_and_order_parameter(tmp_path: Path) -> None:
    data_root = tmp_path / "bt7"
    data_root.mkdir()
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, title="E6 cryostat", detector=[100, 101, 750, 101, 100])
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=False, amplitude=280 - temp)

    output_dir = tmp_path / "corpus_pass"
    code = main(
        [
            "workflow",
            "bt7-corpus-candidate-pass",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--max-files",
            "0",
            "--post-year",
            "0",
            "--no-logs",
        ]
    )

    assert code == 0
    payload = json.loads((output_dir / "bt7_corpus_candidate_pass.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "bt7-corpus-candidate-pass"
    assert payload["summary"]["completed_root_count"] == 1
    assert payload["summary"]["spurion_candidate_count"] >= 1
    assert payload["summary"]["order_parameter_candidate_count"] >= 1
    assert Path(payload["per_root"][0]["artifacts"]["alignment_events_jsonl"]).exists()


def test_overnight_scan_summary_classifies_scan_families(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "energy_scan.bt7")
    _write_temperature_scan(data_root / "order_parameter.bt7")
    _write_bt7_scan(data_root / "elastic_q.bt7", e_value=0.0, title="elastic q scan", detector=[100, 130, 180, 220, 190, 140, 100])
    precedent_index = _write_scan_corpus_precedent_index(tmp_path)

    result = overnight_scan_summary({"data_root": str(data_root), "output_dir": str(tmp_path / "summary"), "precedent_index": str(precedent_index)})

    assert result["status"] == "ok"
    families = {item["family"] for item in result["series"]}
    assert "constant_q_inelastic_energy_series" in families
    assert "temperature_dependence_or_order_parameter" in families
    assert result["suggested_reports"][0]["probability"] > 0
    assert any(item["report_type"] == "inelastic_dispersion_or_lineshape_report" for item in result["suggested_reports"])
    assert result["suggested_reports"][0]["probability_diagnostics"]["total_scan_count"] == 3
    assert result["scan_corpus_context"]["status"] == "used"
    assert "constant_q_inelastic_energy_series" in result["scan_corpus_context"]["matched_families"]
    inelastic_series = next(item for item in result["series"] if item["family"] == "constant_q_inelastic_energy_series")
    assert inelastic_series["scan_corpus_precedent"]["status"] == "matched"
    assert inelastic_series["scan_corpus_precedent"]["support_label"] in {"medium", "high"}
    assert any((item.get("scan_corpus_adjustment") or {}).get("status") == "applied" for item in result["suggested_reports"])
    assert result["next_actions"]
    assert any(item["action_type"] == "inelastic_cut_overlay" for item in result["next_actions"])
    assert any(item["action_type"] == "compare_against_scan_corpus_patterns" for item in result["next_actions"])
    ranked_actions = [item for item in result["next_actions"] if item.get("action_type") != "human_review"]
    assert all("probability_of_value" in item for item in ranked_actions)
    assert all("probability_basis" in item for item in ranked_actions)
    assert Path(result["generated_artifacts"]["series_table_csv"]).exists()
    assert Path(result["generated_artifacts"]["report_suggestions_csv"]).exists()
    assert Path(result["generated_artifacts"]["scan_family_examples_csv"]).exists()
    assert Path(result["generated_artifacts"]["next_actions_csv"]).exists()
    assert Path(result["generated_artifacts"]["scan_corpus_context_csv"]).exists()
    assert (tmp_path / "summary" / "overnight_scan_summary.json").exists()
    assert (tmp_path / "summary" / "overnight_scan_summary.md").exists()
    markdown = (tmp_path / "summary" / "overnight_scan_summary.md").read_text(encoding="utf-8")
    assert "Probability basis" in markdown
    assert "Basis:" in markdown
    assert "Next Actions" in markdown
    assert "Scan Corpus Context" in markdown


def test_overnight_scan_summary_cli_writes_report(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "energy_scan.bt7")
    output_dir = tmp_path / "out"

    code = main(["workflow", "overnight-scan-summary", "--data-root", str(data_root), "--output-dir", str(output_dir)])

    assert code == 0
    payload = json.loads((output_dir / "overnight_scan_summary.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "overnight-scan-summary"
    assert payload["series"]
    assert (output_dir / "series_table.csv").exists()


def test_draft_overnight_report_from_summary_json(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "energy_scan.bt7")
    summary = overnight_scan_summary({"data_root": str(data_root), "output_dir": str(tmp_path / "summary")})
    output_dir = tmp_path / "draft"

    result = draft_overnight_report({"summary_json": summary["artifacts"]["json"], "output_dir": str(output_dir)})

    assert result["status"] == "ok"
    assert result["draft_type"] == "inelastic_dispersion_or_lineshape_report"
    assert result["sections"]
    assert result["followup_commands"]
    assert (output_dir / "morning_report_draft.json").exists()
    assert (output_dir / "morning_report_draft.md").exists()
    assert (output_dir / "morning_report_sections.csv").exists()
    assert "not a completed reduction" in (output_dir / "morning_report_draft.md").read_text(encoding="utf-8")


def test_draft_overnight_report_cli_builds_summary_when_needed(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_temperature_scan(data_root / "order_parameter.bt7")
    output_dir = tmp_path / "draft"

    code = main(["workflow", "draft-overnight-report", "--data-root", str(data_root), "--output-dir", str(output_dir)])

    assert code == 0
    payload = json.loads((output_dir / "morning_report_draft.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "draft-overnight-report"
    assert payload["source_summary"]["scan_count"] == 1
    assert (output_dir / "overnight_scan_summary" / "overnight_scan_summary.json").exists()


def test_inelastic_series_report_writes_overlay_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "energy_scan_a.bt7")
    _write_hkl_energy_scan(data_root / "energy_scan_b.bt7")
    _write_temperature_scan(data_root / "order_parameter.bt7")

    result = inelastic_series_report({"data_root": str(data_root), "output_dir": str(tmp_path / "inelastic")})

    assert result["status"] == "ok"
    assert result["selected_series"]["family"] == "constant_q_inelastic_energy_series"
    assert result["scan_count"] == 2
    assert result["tentative_peak_or_narrow_feature_count"] >= 1
    artifacts = result["generated_artifacts"]
    assert Path(artifacts["scan_table_csv"]).exists()
    assert Path(artifacts["overlay_points_csv"]).exists()
    assert Path(artifacts["tentative_narrow_features_csv"]).exists()
    assert artifacts["overlay_plot_status"] in {"success", "unavailable"}
    assert (tmp_path / "inelastic" / "inelastic_series_report.json").exists()
    assert (tmp_path / "inelastic" / "inelastic_series_report.md").exists()


def test_inelastic_series_report_cli_writes_report(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "energy_scan.bt7")
    output_dir = tmp_path / "inelastic"

    code = main(["workflow", "inelastic-series-report", "--data-root", str(data_root), "--output-dir", str(output_dir)])

    assert code == 0
    payload = json.loads((output_dir / "inelastic_series_report.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "inelastic-series-report"
    assert payload["selected_series"]["family"] == "constant_q_inelastic_energy_series"


def test_order_parameter_triage_finds_splitting_candidate(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 40, 80, 120, 160, 200], start=1):
        _write_order_parameter_angle_scan(data_root / f"BaSr36Ni2As2_004_PSD8260{index:02d}.bt7", temp=temp, splitting=temp < 110)

    result = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "op_out"), "min_scans": 4})

    assert result["status"] == "ok"
    assert result["candidate_count"] >= 1
    top = result["candidates"][0]
    assert top["classification"] in {"structural_splitting_candidate", "mixed_or_nuisance_peak_order_parameter_candidate"}
    assert top["scan_count"] == 6
    assert top["interpretation_readiness"]["safe_claim"] == "candidate_only"
    assert top["model_selection_plan"]
    assert top["reduction_dag"]["dag_run_id"]
    assert top["reduction_dag"]["cache_key"]
    assert top["reduction_dag"]["correction_status_snapshot"]["background_model"]["status"] == "placeholder"
    assert result["reduction_dag"]["status"] == "executed"
    assert Path(result["generated_artifacts"]["dag_runs_json"]).exists()
    assert Path(result["generated_artifacts"]["candidates_csv"]).exists()
    assert Path(result["generated_artifacts"]["per_scan_peaks_csv"]).exists()
    assert result["fit_backend_status"]["bumps"] in {"available", "unavailable"}
    assert (tmp_path / "op_out" / "order_parameter_triage.json").exists()
    assert "Bumps" in (tmp_path / "op_out" / "order_parameter_triage.md").read_text(encoding="utf-8")


def test_order_parameter_triage_cli_writes_report(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 80, 140, 200], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=False, amplitude=300 - temp)

    code = main(["workflow", "order-parameter-triage", "--data-root", str(data_root), "--output-dir", str(tmp_path / "op_out"), "--min-scans", "4"])

    assert code == 0
    payload = json.loads((tmp_path / "op_out" / "order_parameter_triage.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "order-parameter-triage"
    assert payload["candidate_count"] >= 1


def test_order_parameter_triage_handles_missing_peak_center_span(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        _write_peakless_angle_scan(data_root / f"MonotonicOP{index:03d}.bt7", temp=temp, amplitude=50 + temp)

    result = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "op_out"), "min_scans": 4})

    assert result["status"] == "ok"
    top = result["candidates"][0]
    assert top["center_span"] is None
    assert top["classification"] == "mixed_or_nuisance_peak_order_parameter_candidate"
    assert any("peak-center evidence is missing" in reason for reason in top["score_reasons"])


def test_trajectory_cluster_order_parameters_groups_shape_priors(tmp_path: Path) -> None:
    table = tmp_path / "trajectories.csv"
    table.write_text(
        "\n".join(
            [
                "trajectory_id,temperature,intensity",
                "op_a,10,100",
                "op_a,50,80",
                "op_a,100,20",
                "op_b,10,90",
                "op_b,50,70",
                "op_b,100,10",
                "flat,10,5",
                "flat,50,5",
                "flat,100,5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = cluster_order_parameter_trajectories({"input": str(table), "output_dir": str(tmp_path / "clusters")})

    assert result["status"] == "ok"
    labels = {label for cluster in result["clusters"] for label in cluster["shape_labels"]}
    assert "order_parameter_like_on_cooling" in labels
    assert "nearly_flat_or_background_like" in labels
    assert Path(result["generated_artifacts"]["normalized_trajectories_csv"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_trajectory_cluster_order_parameters_reads_triage_json(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 80, 140, 200], start=1):
        _write_order_parameter_angle_scan(data_root / f"OPscan{index:03d}.bt7", temp=temp, splitting=False, amplitude=300 - temp)
    triage = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "triage"), "min_scans": 4})

    result = cluster_order_parameter_trajectories({"triage_json": triage["artifacts"]["json"], "output_dir": str(tmp_path / "clusters"), "observable": "integrated_signal"})

    assert result["status"] in {"ok", "missing_input"}
    assert result["input_type"] == "order_parameter_triage_json"
    assert result["observable"] == "integrated_signal"
    assert result["trajectory_count"] >= 1
    assert Path(result["artifacts"]["json"]).exists()


def test_order_parameter_trajectory_report_joins_triage_and_clusters(tmp_path: Path) -> None:
    triage = {
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "classification": "structural_splitting_candidate",
                "probability": 0.8,
                "scan_count": 4,
                "temperature_range": {"min": 10, "max": 100},
                "area_ratio": 2.0,
                "center_span": 0.05,
                "per_scan_peak_summary": [{"path": "/tmp/a.bt7"}, {"path": "/tmp/b.bt7"}],
            }
        ]
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    normalized = trajectory_dir / "normalized_trajectories.csv"
    normalized.write_text(
        "\n".join(
            [
                "trajectory_id,parameter,normalized_value,shape_label,shape_confidence",
                "candidate-a,10,1.0,order_parameter_like_on_cooling,0.9",
                "candidate-a,100,0.0,order_parameter_like_on_cooling,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory_report = {
        "clusters": [
            {
                "cluster_id": "order_parameter_like_on_cooling",
                "trajectory_ids": ["candidate-a"],
                "shape_labels": ["order_parameter_like_on_cooling"],
                "mean_shape_confidence": 0.9,
            }
        ],
        "generated_artifacts": {"normalized_trajectories_csv": str(normalized)},
    }
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory_report), encoding="utf-8")

    result = order_parameter_trajectory_report({"triage_json": str(triage_path), "trajectory_json": str(trajectory_path), "output_dir": str(tmp_path / "report")})

    assert result["status"] == "ok"
    assert result["top_candidates"][0]["candidate_id"] == "candidate-a"
    assert result["top_candidates"][0]["trajectory_shape_label"] == "order_parameter_like_on_cooling"
    assert result["top_candidates"][0]["review_tier"] == "fit_ready_split_peak_review"
    assert "split-peak" in result["top_candidates"][0]["recommended_next_action"]
    assert Path(result["generated_artifacts"]["top_candidates_csv"]).exists()


def test_order_parameter_trajectory_report_keeps_temperature_spread_paths(tmp_path: Path) -> None:
    per_scan = [{"path": f"/tmp/scan_{index}.bt7", "temperature_mean": temp} for index, temp in enumerate([5, 10, 20, 40, 80, 160, 300])]
    triage = {
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "classification": "single_peak_intensity_order_parameter_candidate",
                "probability": 0.8,
                "scan_count": len(per_scan),
                "temperature_range": {"min": 5, "max": 300},
                "per_scan_peak_summary": per_scan,
            }
        ]
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps({"clusters": []}), encoding="utf-8")

    result = order_parameter_trajectory_report({"triage_json": str(triage_path), "trajectory_json": str(trajectory_path), "output_dir": str(tmp_path / "report"), "max_paths_per_candidate": 3})

    paths = result["top_candidates"][0]["representative_paths"]
    assert paths == ["/tmp/scan_0.bt7", "/tmp/scan_3.bt7", "/tmp/scan_6.bt7"]


def test_plot_order_parameter_candidate_scans_writes_review_plot(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    paths = []
    for index, temp in enumerate([20, 60, 100, 140], start=1):
        path = data_root / f"OPscan{index:03d}.bt7"
        _write_order_parameter_angle_scan(path, temp=temp, splitting=False, amplitude=300 - temp)
        paths.append(str(path))
    report = {
        "top_candidates": [
            {
                "candidate_id": "candidate-a",
                "classification": "single_peak_intensity_order_parameter_candidate",
                "trajectory_shape_label": "order_parameter_like_on_cooling",
                "combined_priority": 0.8,
                "recommended_next_action": "Plot raw scans.",
                "representative_paths": paths,
            }
        ]
    }
    report_path = tmp_path / "combined.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = plot_order_parameter_candidate_scans({"report_json": str(report_path), "output_dir": str(tmp_path / "plots"), "top_k": 1})

    assert result["status"] == "ok"
    assert result["plot_count"] == 1
    assert Path(result["plots"][0]["plot_png"]).exists()
    assert result["plots"][0]["mesh_status"] == "success"
    assert Path(result["plots"][0]["mesh_png"]).exists()
    assert result["plots"][0]["review_tier"] == "fit_ready_intensity_review"
    assert Path(result["generated_artifacts"]["plot_index_csv"]).exists()
    assert Path(result["generated_artifacts"]["gallery_html"]).exists()
    plotted = (tmp_path / "plots" / "candidate_a_plotted_scans.csv").read_text(encoding="utf-8")
    assert "20" in plotted
    assert "140" in plotted


def test_plot_order_parameter_candidate_scans_rebins_only_when_explicit(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    paths = []
    for index, (temp, points) in enumerate([(20, 41), (80, 51), (140, 61)], start=1):
        path = data_root / f"OPscan{index:03d}.bt7"
        _write_variable_point_order_parameter_angle_scan(path, temp=temp, points=points)
        paths.append(str(path))
    report = {
        "top_candidates": [
            {
                "candidate_id": "candidate-a",
                "classification": "single_peak_intensity_order_parameter_candidate",
                "trajectory_shape_label": "order_parameter_like_on_cooling",
                "combined_priority": 0.8,
                "review_tier": "fit_ready_intensity_review",
                "recommended_next_action": "Plot raw scans.",
                "representative_paths": paths,
            }
        ]
    }
    report_path = tmp_path / "combined.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    blocked = plot_order_parameter_candidate_scans({"report_json": str(report_path), "output_dir": str(tmp_path / "blocked"), "top_k": 1})
    allowed = plot_order_parameter_candidate_scans({"report_json": str(report_path), "output_dir": str(tmp_path / "allowed"), "top_k": 1, "allow_rebin": True, "mesh_grid_points": 25})

    assert blocked["plots"][0]["mesh_status"] == "missing_input"
    assert "allow-rebin" in blocked["plots"][0]["mesh_warning"]
    assert allowed["plots"][0]["mesh_status"] == "success_rebinned"
    assert allowed["plots"][0]["mesh_rebin_applied"] is True
    assert Path(allowed["plots"][0]["mesh_png"]).exists()


def test_plot_order_parameter_candidate_cli_accepts_gallery_flags(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    paths = []
    for index, (temp, points) in enumerate([(20, 41), (80, 51), (140, 61), (180, 71)], start=1):
        path = data_root / f"OPscan{index:03d}.bt7"
        _write_variable_point_order_parameter_angle_scan(path, temp=temp, points=points)
        paths.append(str(path))
    report = {
        "top_candidates": [
            {
                "candidate_id": "candidate-a",
                "classification": "single_peak_intensity_order_parameter_candidate",
                "trajectory_shape_label": "order_parameter_like_on_cooling",
                "combined_priority": 0.8,
                "recommended_next_action": "Plot raw scans.",
                "representative_paths": paths,
            }
        ]
    }
    report_path = tmp_path / "combined.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "plots"

    code = main(
        [
            "workflow",
            "plot-order-parameter-candidates",
            "--report-json",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--experiment-id",
            "fixture_experiment",
            "--allow-rebin",
            "--mesh-grid-points",
            "25",
        ]
    )

    assert code == 0
    payload = json.loads((output_dir / "order_parameter_candidate_plots.json").read_text(encoding="utf-8"))
    assert payload["allow_rebin"] is True
    assert payload["plots"][0]["mesh_status"] == "success_rebinned"
    assert payload["plots"][0]["review_tier"] == "fit_ready_intensity_review"
    markdown = (output_dir / "order_parameter_candidate_plots.md").read_text(encoding="utf-8")
    assert "review tier" in markdown
    assert "success_rebinned" in markdown
    assert (output_dir / "order_parameter_candidate_gallery.html").exists()


def test_fit_order_parameter_uses_triage_observations(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 40, 80, 120, 160, 200], start=1):
        _write_order_parameter_angle_scan(data_root / f"BaSr36Ni2As2_004_PSD8260{index:02d}.bt7", temp=temp, splitting=temp < 110)
    triage = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "triage"), "min_scans": 4})

    result = fit_order_parameter(
        {
            "triage_json": triage["artifacts"]["json"],
            "output_dir": str(tmp_path / "fit"),
            "fit_method": "lm",
            "parallel": 1,
            "fit_steps": 20,
        }
    )

    if result["status"] == "unavailable":
        assert "bumps" in result["missing_fields"]
    else:
        assert result["status"] == "ok"
        assert result["fit_backend"] == "bumps"
        assert result["parallel"] == 1
        assert result["fit_acceptance"]["status"] == "candidate_fit_review_required"
        assert result["fit_acceptance"]["accepted"] is False
        assert result["model_comparison_status"] == "not_performed_single_model_fit"
        assert result["background_model"] == "fitted_constant"
        assert result["candidate_summary"]["candidate_id"] == result["candidate_id"]
        assert result["model_selection_plan"]
        assert "Tc" in result["parameters"]
        assert result["fit_quality"]["human_review_required"] is True
        assert result["fit_quality"]["quality_flags"]
        assert result["reduction_dag"]["dag_run_id"]
        assert result["reduction_dag"]["model_branch"] in {"single_peak_power_law", "split_peak_power_law"}
        assert result["reduction_dag"]["bic_tag"]
        assert Path(result["generated_artifacts"]["dag_model_branch_json"]).exists()
        assert Path(result["generated_artifacts"]["fit_curve_csv"]).exists()
        markdown = Path(result["artifacts"]["markdown"]).read_text(encoding="utf-8")
        assert "Acceptance Gate" in markdown
        assert "candidate_fit_review_required" in markdown


def test_dag_run_replay_reexecutes_changed_order_parameter_nodes(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 40, 80, 120, 160, 200], start=1):
        _write_order_parameter_angle_scan(data_root / f"BaSr36Ni2As2_004_PSD8260{index:02d}.bt7", temp=temp, splitting=temp < 110)
    triage = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "triage"), "min_scans": 4})
    dag_json = triage["generated_artifacts"]["dag_runs_json"]
    dag_run_id = triage["candidates"][0]["reduction_dag"]["dag_run_id"]

    result = dag_run_replay(
        {
            "dag_run_json": dag_json,
            "dag_run_id": dag_run_id,
            "output_dir": str(tmp_path / "replay"),
            "correction_overrides": {"background_model": {"version": "reviewed_background_v1", "status": "proposed"}},
        }
    )

    assert result["status"] == "success"
    assert result["cache_replay"]["prior_run_id"] == dag_run_id
    assert "background_model" in result["cache_replay"]["changed_node_ids"]
    assert Path(result["artifacts"]["json"]).exists()

    cli_out = tmp_path / "replay_cli"
    code = main(
        [
            "workflow",
            "dag-run-replay",
            "--dag-run-json",
            dag_json,
            "--dag-run-id",
            dag_run_id,
            "--output-dir",
            str(cli_out),
            "--json",
            '{"correction_overrides":{"background_model":{"version":"reviewed_background_v2","status":"proposed"}}}',
        ]
    )
    assert code == 0
    payload = json.loads((cli_out / "dag_run_replay.json").read_text(encoding="utf-8"))
    assert "background_model" in payload["cache_replay"]["changed_node_ids"]


def test_fit_order_parameter_panweb_backend_rejects_invalid_path(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 40, 80, 120], start=1):
        _write_order_parameter_angle_scan(data_root / f"BaSr36Ni2As2_004_PSD8260{index:02d}.bt7", temp=temp, splitting=temp < 110)
    triage = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "triage"), "min_scans": 4})

    result = fit_order_parameter(
        {
            "triage_json": triage["artifacts"]["json"],
            "output_dir": str(tmp_path / "fit"),
            "fit_backend": "panweb_global",
            "panweb_path": str(tmp_path / "missing-panweb"),
        }
    )

    assert result["status"] == "missing_input"
    assert "panweb_path" in result["missing_fields"]


def test_fit_order_parameter_panweb_model_comparison_ranks_grid(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 40, 80, 120, 160], start=1):
        _write_order_parameter_angle_scan(data_root / f"BaSr36Ni2As2_004_PSD8260{index:02d}.bt7", temp=temp, splitting=temp < 110)
    triage = order_parameter_triage({"data_root": str(data_root), "output_dir": str(tmp_path / "triage"), "min_scans": 4})

    # Fake pan-web checkout: fit quality is deterministic per (model, op_function)
    # so the BIC ranking is predictable; magnetic:tanh is the planted winner.
    panweb = tmp_path / "panweb" / "src"
    panweb.mkdir(parents=True)
    (panweb / "auto_peaks.py").write_text(
        '''
CHI2 = {
    ("split", "power_law"): 500.0,
    ("split", "brillouin"): 400.0,
    ("split", "tanh"): 300.0,
    ("magnetic", "power_law"): 200.0,
    ("magnetic", "brillouin"): 150.0,
    ("magnetic", "tanh"): 100.0,
}


def fit_global_order_parameter(scan_data, initial_Tc=110.0, initial_beta=0.5,
                               use_Tc_smearing=False, model_type="split",
                               inverse=False, op_function="power_law", tanh_width=5.0):
    if op_function == "brillouin":
        raise ValueError("brillouin unavailable in fake pan-web")
    chi2_per_scan = CHI2[(model_type, op_function)] / len(scan_data)
    global_params = {
        "Tc": {"value": 110.0}, "beta": {"value": 0.35}, "Tc_width": {"value": 0.0},
        "delta_0": {"value": 0.8}, "x0": {"value": 25.0}, "sigma": {"value": 0.2},
        "background": {"value": 1.0}, "slope": {"value": 0.0}, "I_total": {"value": 100.0},
    }
    results = []
    for x, y, errors, T in sorted(scan_data, key=lambda item: item[3]):
        results.append({
            "temperature": {"value": T}, "order_parameter": {"value": 0.5},
            "splitting_delta": {"value": 0.4}, "splitting_center": {"value": 25.0},
            "amp_center": {"value": 50.0}, "amp_side": {"value": 25.0},
            "chi_squared": {"value": chi2_per_scan},
        })
    return global_params, results
''',
        encoding="utf-8",
    )

    result = fit_order_parameter(
        {
            "triage_json": triage["artifacts"]["json"],
            "output_dir": str(tmp_path / "fit"),
            "fit_backend": "panweb_global",
            "panweb_path": str(tmp_path / "panweb"),
            "compare_models": True,
        }
    )

    assert result["status"] == "ok"
    assert result["model_comparison_status"] == "external_panweb_global_grid"
    comparison = result["model_comparison"]
    assert len(comparison) == 6
    by_key = {(entry["model_type"], entry["op_function"]): entry for entry in comparison}
    # brillouin variants fail in the fake backend and must be reported, not hidden
    assert by_key[("split", "brillouin")]["status"] == "error"
    assert by_key[("magnetic", "brillouin")]["status"] == "error"
    winner = by_key[("magnetic", "tanh")]
    assert winner["delta_bic"] == 0.0
    assert winner["delta_aic"] == 0.0
    assert winner["aic"] == winner["chi_squared_total"] + 2 * winner["n_parameters"]
    assert result["selected_model"]["model_type"] == "magnetic"
    assert result["selected_model"]["op_function"] == "tanh"
    assert result["selected_model"]["selection_basis"] == "lowest_bic"
    # selection is evidence, not acceptance: the review gate must still apply
    acceptance = result["fit_acceptance"]
    assert acceptance["accepted"] is False
    assert Path(result["generated_artifacts"]["model_comparison_csv"]).exists()
    markdown = Path(result["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "Model Comparison (pan-web grid)" in markdown
    assert "magnetic | tanh" in markdown

    # subset grid via CLI-style string
    subset = fit_order_parameter(
        {
            "triage_json": triage["artifacts"]["json"],
            "output_dir": str(tmp_path / "fit_subset"),
            "fit_backend": "panweb_global",
            "panweb_path": str(tmp_path / "panweb"),
            "compare_models": True,
            "model_grid": "split:power_law,split:tanh",
        }
    )
    assert [(entry["model_type"], entry["op_function"]) for entry in subset["model_comparison"]] == [("split", "power_law"), ("split", "tanh")]
    assert subset["selected_model"]["op_function"] == "tanh"


def test_experiment_dashboard_diversifies_spurion_candidates_by_experiment(tmp_path: Path) -> None:
    first = tmp_path / "2015" / "11111" / "data"
    second = tmp_path / "2016" / "22222" / "data"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for index in range(3):
        _write_bt7_scan(first / f"a_{index}.bt7", e_value=6.0, title="E6 cryostat", detector=[100, 101, 800, 101, 100], details="CCR PG")
    _write_bt7_scan(second / "b_0.bt7", e_value=6.0, title="E6 cryostat", detector=[100, 101, 750, 101, 100], details="CCR PG")

    result = survey_inelastic_spurion_evidence(
        {
            "data_root": str(tmp_path),
            "output_dir": str(tmp_path / "survey"),
            "post_year": 2010,
            "spurion_top_k": 2,
            "ranking_mode": "diversified",
        }
    )

    paths = [candidate["path"] for candidate in result["candidates"]]
    assert any("/11111/" in path for path in paths)
    assert any("/22222/" in path for path in paths)


def test_experiment_dashboard_loads_overlay_config(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "scan.bt7", e_value=3.0, title="E3", detector=[100, 100, 400, 99, 98])
    overlay_config = tmp_path / "overlays.yaml"
    overlay_config.write_text(
        "\n".join(
            [
                "overlays:",
                "  - overlay_id: substrate_si_111",
                "    overlay_type: material_peak_reference",
                "    label: Si substrate 111",
                "    material: Si",
                "    parameters:",
                "      hkl: [1, 1, 1]",
            ]
        ),
        encoding="utf-8",
    )

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "out"), "overlay_config": str(overlay_config)})

    assert result["overlay_layers"][0]["overlay_id"] == "substrate_si_111"
    assert result["overlay_layers"][0]["parameters"]["hkl"] == [1, 1, 1]


def test_experiment_dashboard_runs_spurion_check_when_explicit_inputs_exist(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "scan.bt7")

    class FakeRunner:
        def run(self, call):
            assert call.tool_namespace == "tas.spurion"
            assert call.tool_name == "check_spurions"
            assert call.arguments["q_point"] == {"h": 1.0, "k": 0.0, "l": 0.0}
            return ToolResult(
                status=ToolResultStatus.success,
                tool_name=call.tool_name,
                tool_namespace=call.tool_namespace,
                output={"details": {"candidates": [{"kind": "reciprocal_lattice_harmonic"}]}},
                audit_id="audit-spurion",
            )

    monkeypatch.setattr("tas_copilot_core.workflows.experiment_dashboard.default_runner", lambda: FakeRunner())
    result = build_experiment_dashboard(
        {
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
            "spurion_check": {
                "enabled": True,
                "lattice": {"a": 6.28, "b": 6.28, "c": 6.28, "alpha": 90, "beta": 90, "gamma": 90},
                "instrument_config": {"fixed_energy": 14.7, "fixed_energy_mode": "fixed_final_energy"},
            },
        }
    )

    evidence_types = {item["evidence_type"] for item in result["spurion_likelihood_candidates"][0]["evidence"]}
    assert "spurion2_followup" in evidence_types
    assert result["spurion_likelihood_candidates"][0]["likelihood_score"] > 0.62


def test_experiment_dashboard_applies_ready_label_calibration_to_spurion_scores(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7_scan(data_root / "scan.bt7", e_value=6.0, title="E6 cryostat", detector=[100, 101, 750, 101, 100])
    store_path = tmp_path / "review_outcomes.sqlite"
    labels = []
    for index in range(20):
        labels.append(
            ReviewLabel(
                object_type="spurion_candidate",
                object_id=f"positive-{index}",
                source_workflow="survey-inelastic-spurion-evidence",
                verdict="positive",
                claim_state="accepted",
                evidence_channels_relied_on=["evidence_channel_scan_family", "evidence_channel_feature_width"],
            )
        )
    for index in range(10):
        labels.append(
            ReviewLabel(
                object_type="spurion_candidate",
                object_id=f"negative-{index}",
                source_workflow="survey-inelastic-spurion-evidence",
                verdict="negative",
                claim_state="rejected",
                evidence_channels_relied_on=["evidence_channel_scan_family", "evidence_channel_feature_width"],
            )
        )
    ReviewOutcomeSQLiteStore(store_path).upsert_labels(labels)

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "out"), "review_label_store": str(store_path)})

    candidate = result["spurion_likelihood_candidates"][0]
    assert candidate["label_informed_prior"]["applied"] is True
    assert candidate["likelihood_score"] > candidate["label_informed_prior"]["base_score"]
    assert result["spurion_ranking_diagnostics"]["probability_status"] == "weakly_calibrated_from_labels"


def test_experiment_dashboard_uses_filter_and_velocity_selector_state_as_probability_evidence(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for index, y in enumerate([100, 101, 900, 100, 101]):
        rows.append(f"{index:.3f} 6.0 14.7 14.7 100000 {y} IN ON")
    (data_root / "filter_velocity.bt7").write_text(
        "\n".join(
            [
                "#Filename    filter_velocity",
                "#Date        2018-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ExptDetails CCR",
                "#FixedE      Ef 14.7",
                "#ScanType   VECTOR",
                "#Scan        1 QX",
                "#ScanDescr   Scan:SubID=12345:JType=VECTOR:Range=E=6 6 s:Title=E scan",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#Columns QX E Ef Ei Monitor Detector FilTran Velsel",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "out")})

    record = result["scan_records"][0]
    assert record["metadata"]["filter_state"]["state"] == "IN"
    assert record["metadata"]["filter_state"]["material"] == "PG"
    assert record["metadata"]["velocity_selector_state"]["state"] == "in"
    evidence = {item["evidence_type"]: item for item in result["spurion_likelihood_candidates"][0]["evidence"]}
    assert "filter_state" in evidence
    assert "lower but not zero" in evidence["filter_state"]["summary"]
    assert "velocity_selector_state" in evidence


def test_experiment_config_propagates_overlays_and_resolution_prior(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_hkl_energy_scan(data_root / "scan.bt7")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "instrument_config:",
                "  fixed_energy: 14.7",
                "  fixed_energy_mode: fixed_final_energy",
                "resolution_proxy:",
                "  expected_peak_width_points: 6",
                "overlay_packs:",
                "  - bt7_common",
                "materials_in_beam:",
                "  mount:",
                "    - Cu",
                "lattice:",
                "  a: 6.28",
                "  b: 6.28",
                "  c: 6.28",
                "  alpha: 90",
                "  beta: 90",
                "  gamma: 90",
            ]
        ),
        encoding="utf-8",
    )

    result = build_experiment_dashboard({"data_root": str(data_root), "output_dir": str(tmp_path / "out"), "experiment_config": str(config)})

    assert result["experiment_config"]["has_lattice"] is True
    assert result["experiment_config"]["spurion_check_enabled"] is True
    assert any(layer["material"] == "Cu" for layer in result["overlay_layers"])
    evidence_types = {item["evidence_type"] for item in result["spurion_likelihood_candidates"][0]["evidence"]}
    assert "resolution_width_proxy" in evidence_types
    assert result["recommended_diagnostic_scans"]
    diagnostic = result["recommended_diagnostic_scans"][0]
    assert diagnostic["suggested_scan"]["planning_only"] is True
    assert diagnostic["suggested_scan"]["planned_points"] > 0
    assert diagnostic["suggested_scan"]["estimated_duration_minutes"] > 0
    assert diagnostic["spurion2_eligibility"]["status"] in {"eligible", "missing_input"}
    assert diagnostic["missing_input_prompts"]


def test_repeatability_context_updates_related_scan_family(tmp_path: Path) -> None:
    family = tmp_path / "2018" / "33333" / "data"
    family.mkdir(parents=True)
    _write_bt7_scan(family / "scan_family_001.bt7", e_value=6.0, title="E6", detector=[100, 101, 800, 101, 100], details="CCR PG")
    _write_bt7_scan(family / "scan_family_002.bt7", e_value=6.0, title="E6", detector=[100, 101, 780, 101, 100], details="CCR PG")

    result = survey_inelastic_spurion_evidence({"data_root": str(tmp_path), "output_dir": str(tmp_path / "survey"), "post_year": 2010, "spurion_top_k": 4})

    assert any(
        evidence["evidence_type"] == "scan_family_repeatability"
        for candidate in result["candidates"]
        for evidence in candidate["evidence"]
    )


def _write_bt7_scan(path: Path, *, e_value: float, title: str, detector: list[int], details: str = "BT7 PG") -> None:
    rows = []
    for index, y in enumerate(detector):
        rows.append(f"{index:.3f} {e_value:.3f} 14.7 14.7 100000 {y} {index}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2015-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ExptDetails " + details,
                "#ScanType   VECTOR",
                "#Scan        1 QX",
                f"#ScanDescr   Scan:SubID=12345:JType=VECTOR:Range=E={e_value} {e_value} s:Title={title}",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#Columns QX E Ef Ei Monitor Detector Temp",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_hkl_energy_scan(path: Path) -> None:
    rows = []
    for index, energy in enumerate([-1.0, 0.0, 1.0, 2.0, 3.0]):
        detector = 700 if index == 2 else 100
        rows.append(f"1.0 0.0 0.0 {energy:.3f} 14.7 14.7 100000 {detector} {index}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2015-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ExptDetails CCR PG",
                "#FixedE      Ef 14.7",
                "#ScanType   VECTOR",
                "#Scan        1 E",
                "#ScanDescr   Scan:SubID=12345:JType=VECTOR:Range=E=-1 3 s:Title=E scan",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#Columns H K L E Ef Ei Monitor Detector Temp",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_temperature_scan(path: Path) -> None:
    rows = []
    for index, temp in enumerate([10, 20, 30, 40, 50]):
        detector = 100 + index * 25
        rows.append(f"{temp:.3f} 0.0 14.7 14.7 100000 {detector}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2015-01-01 02:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ExptDetails CCR PG",
                "#FixedE      Ef 14.7",
                "#ScanType   ENV",
                "#Scan        1 Temp",
                "#ScanDescr   Scan:SubID=12345:JType=ENV:Range=Temp=10 50 s:Title=Temp scan",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#Columns Temp E Ef Ei Monitor Detector",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scan_corpus_precedent_index(tmp_path: Path) -> Path:
    path = tmp_path / "scan_corpus_index.jsonl"
    records = [
        {
            "experiment_id": "private-inelastic-pattern",
            "title": "private inelastic pattern",
            "scan_count": 12,
            "datafile_count": 12,
            "hydrated_clue_summary": {
                "top_values": {
                    "scan_variable": [{"value": "E", "count": 12}],
                    "fixed_e": [{"value": "Ef 14.7", "count": 12}],
                    "scan_description": [{"value": "Scan:SubID=1:JType=VECTOR:Range=E=2.0 15.0 s:Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0 s:Title=E scan"}],
                },
                "reader_subgroups": [{"detector_modes": ["SingDetFlat"]}],
            },
        },
        {
            "experiment_id": "private-temperature-pattern",
            "title": "private temperature pattern",
            "scan_count": 8,
            "datafile_count": 8,
            "hydrated_clue_summary": {
                "top_values": {
                    "scan_variable": [{"value": "Temp", "count": 8}],
                    "temp_mean": [{"value": "20", "count": 2}, {"value": "50", "count": 2}],
                    "scan_description": [{"value": "Scan:SubID=2:JType=ENV:Range=Temp=10 80 s:Title=Tdep"}],
                }
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _write_order_parameter_angle_scan(path: Path, *, temp: float, splitting: bool, amplitude: float = 300.0) -> None:
    rows = []
    for index in range(51):
        a3 = 112.0 + index * 0.1
        centers = [114.5]
        amps = [amplitude]
        if splitting:
            centers = [114.15, 114.85]
            amps = [amplitude * 0.75, amplitude * 0.75]
        y = 20.0
        for center, amp in zip(centers, amps):
            y += amp * __import__("math").exp(-0.5 * ((a3 - center) / 0.18) ** 2)
        rows.append(f"0.0 0.0 0.0 0.0 49.75 {temp:.3f} 100000 {y:.5f} {a3:.5f}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2016-08-01 07:02:27 EDT",
                "#InstrName   BT7",
                "#ExptID      22322",
                "#ScanType   ANGLE",
                "#Scan        12 A3",
                "#FixedE",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#ScanDescr   Scan:SubID=22322:JType=ANGLE:Npts=51:Counts=300.0:Range=A3=112.0 117.0 s:Range=A4=49.75 49.75 s:Title=BP004",
                "#Columns QX QY QZ E A4 Temp Monitor Detector A3",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_variable_point_order_parameter_angle_scan(path: Path, *, temp: float, points: int, amplitude: float = 300.0) -> None:
    rows = []
    for index in range(points):
        frac = index / float(points - 1)
        a3 = 112.0 + frac * 5.0
        y = 20.0 + amplitude * __import__("math").exp(-0.5 * ((a3 - 114.5) / 0.18) ** 2)
        rows.append(f"0.0 0.0 0.0 0.0 49.75 {temp:.3f} 100000 {y:.5f} {a3:.5f}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2016-08-01 07:02:27 EDT",
                "#InstrName   BT7",
                "#ExptID      22322",
                "#ScanType   ANGLE",
                "#Scan        12 A3",
                "#FixedE",
                "#AnalyzerDetectorMode 4 PSDDiff",
                f"#ScanDescr   Scan:SubID=22322:JType=ANGLE:Npts={points}:Counts=300.0:Range=A3=112.0 117.0 s:Range=A4=49.75 49.75 s:Title=BP004",
                "#Columns QX QY QZ E A4 Temp Monitor Detector A3",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_peakless_angle_scan(path: Path, *, temp: float, amplitude: float) -> None:
    rows = []
    for index in range(51):
        a3 = 112.0 + index * 0.1
        y = 20.0 + amplitude * (index / 50.0)
        rows.append(f"0.0 0.0 0.0 0.0 49.75 {temp:.3f} 100000 {y:.5f} {a3:.5f}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2016-08-01 07:02:27 EDT",
                "#InstrName   BT7",
                "#ExptID      22322",
                "#ScanType   ANGLE",
                "#Scan        12 A3",
                "#FixedE",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#ScanDescr   Scan:SubID=22322:JType=ANGLE:Npts=51:Counts=300.0:Range=A3=112.0 117.0 s:Range=A4=49.75 49.75 s:Title=PeaklessOP",
                "#Columns QX QY QZ E A4 Temp Monitor Detector A3",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_alignment_fit_overlays_maps_event_stems(tmp_path):
    import json as _json

    from tas_copilot_core.workflows.experiment_dashboard import _load_alignment_fit_overlays

    state = {
        "peak_fit_diagnostics": [
            {"event_id": "scan:fpx840986.bt7", "status": "success", "axis": "A3", "model_family": "panweb_bumps_bic_gaussian", "fits": [{"center": 31.0, "amplitude": 1300.0, "sigma_preview": 0.24}]},
            {"event_id": "scan:fpx840988.bt7", "status": "ramp_saturation_candidate", "axis": "SmplWdth"},
        ]
    }
    path = tmp_path / "alignment_state_machine.json"
    path.write_text(_json.dumps(state), encoding="utf-8")

    overlays = _load_alignment_fit_overlays(str(path))

    assert set(overlays) == {"fpx840986"}  # success-only
    assert overlays["fpx840986"]["fits"][0]["center"] == 31.0
    assert overlays["fpx840986"]["human_review_required"] is True
    assert _load_alignment_fit_overlays(None) == {}


def test_experiment_day_index_orders_days_newest_first(tmp_path):
    from tas_copilot_core.workflows.experiment_dashboard import experiment_day_index

    for date, align_rows in (("2018-05-23", "| 24858-alignment_or_rocking_scan-a3 | alignment_or_rocking_scan | 6 | 120 | high | x |"), ("2018-07-07", "| 24858-alignment_tilt_scan-smplltilt | alignment_tilt_scan | 2 | 30 | high | x |")):
        day = tmp_path / date
        (day / "report").mkdir(parents=True)
        (day / "dashboard").mkdir()
        (day / "dashboard/dashboard.html").write_text("<html></html>", encoding="utf-8")
        (day / "report/morning_report_draft.md").write_text(
            f"# Morning Report Draft\n- Parsed scans: 25\n- Draft type: `alignment_progress_report`\n{align_rows}\n",
            encoding="utf-8",
        )

    report = experiment_day_index({"reports_root": str(tmp_path), "output_dir": str(tmp_path / "index_out")})

    assert report["status"] == "success"
    assert [day["date"] for day in report["days"]] == ["2018-07-07", "2018-05-23"]
    assert report["days"][1]["parsed_scans"] == 25
    assert (tmp_path / "index.html").exists()
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert html.index("2018-07-07") < html.index("2018-05-23")


def test_write_report_bundle_optionally_renders_html(tmp_path):
    from tas_copilot_core.reporting import write_report_bundle

    def renderer(report):
        return "# Morning Report Draft\n\n- Parsed scans: 25\n\n| a | b |\n| --- | --- |\n| 1 | `code` |\n"

    artifacts = write_report_bundle(
        tmp_path, {"status": "ok"}, markdown_renderer=renderer,
        json_filename="r.json", markdown_filename="r.md", html_filename="r.html",
    )

    assert set(artifacts) == {"json", "markdown", "html"}
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<title>Morning Report Draft</title>" in html
    assert "<h1>Morning Report Draft</h1>" in html
    assert "<li>Parsed scans: 25</li>" in html
    assert "<table>" in html and "<td><code>code</code></td>" in html
    assert "Read-only decision support" not in html  # no generic banner: warnings are for specific findings

    no_html = write_report_bundle(tmp_path / "b", {"status": "ok"}, markdown_renderer=renderer, json_filename="r.json", markdown_filename="r.md")
    assert "html" not in no_html


def test_measurement_kind_treats_zero_energy_transfer_as_elastic(tmp_path):
    from pathlib import Path as _Path

    from tas_copilot_core.workflows.experiment_dashboard import _measurement_kind

    findpeak_header = {"ScanDescr": "Scan:Type=6:Fixed=0:FixedE=1:Range=A3=31.0 0.1:Npts=21:Title=FindPeak"}
    zero_e_rows = [{"E": 0.0002, "A3": 30.9}, {"E": -0.0001, "A3": 31.1}]
    assert _measurement_kind(_Path("fpx840986.bt7"), findpeak_header, zero_e_rows) == "elastic_or_diffraction"

    # No E column at all: the FindPeak vocabulary still classifies it.
    assert _measurement_kind(_Path("fpx840986.bt7"), findpeak_header, [{"A3": 31.0}]) == "elastic_or_diffraction"

    # Genuinely inelastic data stays inelastic.
    assert _measurement_kind(_Path("SMS_HH_Q002_e10.bt7"), {}, [{"E": 10.0}]) == "inelastic"

    # No evidence at all remains unknown.
    assert _measurement_kind(_Path("mystery.bt7"), {"ScanDescr": "Scan:Title=Stuff"}, [{"A3": 1.0}]) == "unknown"


def test_measurement_kind_treats_zero_energy_transfer_as_elastic():
    from pathlib import Path as _Path

    from tas_copilot_core.workflows.experiment_dashboard import _measurement_kind

    findpeak_header = {"ScanDescr": "Scan:Type=6:FixedE=1:Range=A3=31.0 0.1:Npts=21:Title=FindPeak"}
    zero_e_rows = [{"E": 0.0002, "A3": 30.9}, {"E": -0.0001, "A3": 31.1}]
    assert _measurement_kind(_Path("fpx840986.bt7"), findpeak_header, zero_e_rows) == "elastic_or_diffraction"
    # No E column: the FindPeak vocabulary still classifies it.
    assert _measurement_kind(_Path("fpx840986.bt7"), findpeak_header, [{"A3": 31.0}]) == "elastic_or_diffraction"
    # Genuinely inelastic stays inelastic; no evidence stays unknown.
    assert _measurement_kind(_Path("SMS_HH_Q002_e10.bt7"), {}, [{"E": 10.0}]) == "inelastic"
    assert _measurement_kind(_Path("mystery.bt7"), {"ScanDescr": "Scan:Title=Stuff"}, [{"A3": 1.0}]) == "unknown"


def test_overnight_summary_emits_family_plots_and_morning_report_embeds_them(tmp_path):
    import json as _json

    from tas_copilot_core.workflows.experiment_dashboard import draft_overnight_report, overnight_scan_summary

    data_root = tmp_path / "data"
    data_root.mkdir()
    header = (
        "#ICE           0.4.0\n#ICERepositoryInfo ice/server\n#Filename      fpx1\n"
        "#Date          2018-07-07 14:12:31 EDT\n#InstrName     BT7\n#ExptID        24858\n"
        "#ScanDescr     Scan:Type=6:FixedE=1:Range=A3=21.6 0.05:Npts=5:Title=FindPeak\n"
        "#Reference     6 Time\n#Signal        8 Detector\n"
        "#Columns       QX QY QZ E A3 Time Monitor Detector\n"
    )
    rows = "\n".join(f"1.0 1.0 0.0 0.0 {21.5 + 0.05 * i} 1.0 1000 {100 + 400 * (i == 2)}" for i in range(5))
    (data_root / "fpx841836.bt7").write_text(header + rows + "\n", encoding="utf-8")

    summary = overnight_scan_summary({"data_root": str(data_root), "output_dir": str(tmp_path / "summary")})

    plotted = [s for s in summary["series"] if s.get("plot_png")]
    assert plotted, "expected at least one series plot"
    assert (tmp_path / "summary" / plotted[0]["plot_png"]).exists()

    report = draft_overnight_report({"summary_json": summary["artifacts"]["json"], "output_dir": str(tmp_path / "report")})

    assert report["family_plots"], "morning report should reference family plots"
    relative = report["family_plots"][0]["plot"]
    assert ((tmp_path / "report") / relative).resolve().exists()
    md = (tmp_path / "report/morning_report_draft.md").read_text(encoding="utf-8")
    assert "## Scan Family Plots" in md and "![" in md
    html = (tmp_path / "report/morning_report_draft.html").read_text(encoding="utf-8")
    assert "<img src=" in html and "figcaption" in html


def test_experiment_story_report_sections_and_dispersion_pooling(tmp_path):
    from tas_copilot_core.workflows.experiment_story import experiment_story_report

    data_root = tmp_path / "data"
    data_root.mkdir()
    header = (
        "#ICE           0.4.0\n#Filename      escan\n#Date          2018-07-09 10:00:00 EDT\n"
        "#InstrName     BT7\n#ExptID        24858\n"
        "#ScanDescr     Scan:Type=1:FixedE=1:Range=E=5.0 1.0:Npts=6:Title=Escan\n"
        "#Reference     6 Time\n#Signal        9 Detector\n"
        "#Columns       QX QY QZ H K L E Time Monitor Detector\n"
    )
    for q, name in ((0.1, "e1"), (0.2, "e2"), (0.3, "e3")):
        rows = "\n".join(
            f"{q} {q} 2.0 {q} {q} 2.0 {2.0 + i} 1.0 10000 {50 + 10 * i}" for i in range(10)
        )
        (data_root / f"SMS_{name}_84100{int(q*10)}.bt7").write_text(header + rows + "\n", encoding="utf-8")

    report = experiment_story_report({"data_roots": [str(data_root)], "output_dir": str(tmp_path / "story")})

    assert report["status"] == "ok"
    dispersion = report["sections"]["dispersion"]
    assert dispersion["point_count"] == 30
    assert dispersion["panels"], "expected an E-vs-Q panel from three Q points"
    assert (tmp_path / "story" / dispersion["panels"][0]["png"]).exists()
    html = (tmp_path / "story/experiment_story_report.html").read_text(encoding="utf-8")
    assert "<img src=" in html

    # Section limiting works and missing data stays honest.
    limited = experiment_story_report({"data_roots": [str(data_root)], "sections": ["order_parameters"], "output_dir": str(tmp_path / "op")})
    assert set(limited["sections"]) == {"order_parameters"}
    assert limited["sections"]["order_parameters"]["status"] == "no_temperature_series"


def test_story_report_derives_alignment_state_when_not_supplied(tmp_path):
    from tas_copilot_core.workflows.experiment_story import experiment_story_report

    data_root = tmp_path / "data"
    data_root.mkdir()
    header = (
        "#ICE           0.4.0\n#Filename      fpx\n#Date          2018-07-09 10:00:00 EDT\n"
        "#InstrName     BT7\n#ExptID        24858\n"
        "#ScanDescr     Scan:Type=6:FixedE=1:Range=A3=31.0 0.1:Npts=5:Title=FindPeak\n"
        "#Reference     6 Time\n#Signal        9 Detector\n"
        "#Columns       QX QY QZ H K L A3 Time Monitor Detector\n"
    )
    rows = "\n".join(f"1.0 1.0 0.0 1.0 1.0 0.0 {30.8 + 0.1 * i} 1.0 1000 {100 + 400 * (i == 2)}" for i in range(5))
    (data_root / "fpx841836.bt7").write_text(header + rows + "\n", encoding="utf-8")
    (data_root / "2018_07_09.console.log").write_text(
        "Wed Jul  9 10:00:00 2018 sample setlattice 4.58947 4.58947 7.7189 90 90 120\n",
        encoding="utf-8",
    )

    report = experiment_story_report({"data_roots": [str(data_root)], "sections": ["alignment"], "output_dir": str(tmp_path / "story")})

    alignment = report["sections"]["alignment"]
    # The chain (alignment-corpus-index -> alignment-state-machine) derived
    # itself from the data root; HKL assignments still need a planning
    # report, and the section says so instead of presenting "unassigned"
    # as a finding.
    assert alignment["assignments_source"] == "derived_without_planning_report"
    assert alignment["scans_by_reflection"] == {"unassigned": 1}
    assert any("alignment peak planning report" in note for note in alignment["notes"])
    # Software-change notes derive from the console log without any extra input.
    assert any("setlattice" in change for change in alignment["software_changes"])
    assert (tmp_path / "story" / "derived_alignment_state" / "session_00" / "state" / "alignment_state_machine.json").exists()
    md = (tmp_path / "story" / "experiment_story_report.md").read_text(encoding="utf-8")
    assert "Assignments source: derived_without_planning_report" in md


def test_story_report_supplied_state_json_still_wins(tmp_path):
    import json as _json

    from tas_copilot_core.workflows.experiment_story import experiment_story_report

    data_root = tmp_path / "data"
    data_root.mkdir()
    header = (
        "#ICE           0.4.0\n#Filename      fpx\n#Date          2018-07-09 10:00:00 EDT\n"
        "#InstrName     BT7\n#ExptID        24858\n"
        "#ScanDescr     Scan:Type=6:FixedE=1:Range=A3=31.0 0.1:Npts=5:Title=FindPeak\n"
        "#Reference     6 Time\n#Signal        9 Detector\n"
        "#Columns       QX QY QZ H K L A3 Time Monitor Detector\n"
    )
    rows = "\n".join(f"1.0 1.0 0.0 1.0 1.0 0.0 {30.8 + 0.1 * i} 1.0 1000 {100 + 400 * (i == 2)}" for i in range(5))
    (data_root / "fpx841836.bt7").write_text(header + rows + "\n", encoding="utf-8")
    state_json = tmp_path / "alignment_state_machine.json"
    state_json.write_text(
        _json.dumps(
            {
                "a4_prediction_checks": [
                    {"event_id": f"scan:{data_root / 'fpx841836.bt7'}", "best_candidate_hkl": "1 1 0", "discrimination_status": "resolved_unique"}
                ]
            }
        ),
        encoding="utf-8",
    )

    report = experiment_story_report(
        {"data_roots": [str(data_root)], "sections": ["alignment"], "alignment_state_json": [str(state_json)], "output_dir": str(tmp_path / "story")}
    )

    alignment = report["sections"]["alignment"]
    assert alignment["assignments_source"] == "supplied"
    assert alignment["scans_by_reflection"] == {"(1 1 0)": 1}
    # No derivation ran when the state JSON was supplied.
    assert not (tmp_path / "story" / "derived_alignment_state").exists()


def test_copilot_router_routes_semantic_plot_questions(tmp_path):
    from tas_copilot_core.workflows.copilot_router import copilot_ask

    for question, intent in (
        ("Plot the dispersion measured so far.", "dispersion_map_report"),
        ("Plot the alignment scans so far.", "alignment_scans_plot"),
        ("Plot the order parameter measured so far.", "order_parameter_plot"),
        ("Where does the experiment stand? Group the scans by scientific object.", "experiment_story_report"),
    ):
        result = copilot_ask({"question": question, "output_dir": str(tmp_path / intent)})
        assert result["route"]["intent"] == intent, (question, result["route"]["intent"])
        assert result["status"] == "missing_input"
        assert result["missing_inputs"] == ["data_root"]


def test_html_renderer_allows_brackets_in_image_alt(tmp_path):
    from tas_copilot_core.reporting import render_markdown_report_html

    html = render_markdown_report_html("# T\n\n![E vs HH at L=2 - 2018-05 [1 1 0 0 0 1] Ef 13.7](plots/x.png)\n")
    assert '<img src="plots/x.png"' in html


def test_dispersion_scan_plan_routes_and_slot_fills(tmp_path):
    from tas_copilot_core.workflows.copilot_router import copilot_ask

    result = copilot_ask({"question": "Plan the scans to map out the dispersion.", "output_dir": str(tmp_path / "r")})
    assert result["route"]["intent"] == "dispersion_scan_plan"
    assert result["missing_inputs"] == ["data_root"]


def test_dispersion_scan_plan_proposes_uncovered_rows(tmp_path):
    from tas_copilot_core.workflows.scan_planning import dispersion_scan_plan

    data_root = tmp_path / "data"
    data_root.mkdir()
    header = (
        "#ICE           0.4.0\n#Filename      q1\n#Date          2018-07-09 10:00:00 EDT\n"
        "#InstrName     BT7\n#ExptID        24858\n"
        "#Lattice       4.58 4.58 7.73 90 90 120\n#Orient        1 1 0 0 0 1\n"
        "#FixedE        Ef 14.7\n"
        "#ScanDescr     Scan:Type=1:FixedE=1:Range=QX=0.0 0.05:Npts=9:Title=qscan\n"
        "#Reference     6 Time\n#Signal        9 Detector\n"
        "#Columns       QX QY QZ H K L E Time Monitor Detector\n"
    )
    # One covered E row (4 meV) spanning HH -0.2..0.2 at L=2.
    rows = "\n".join(f"{q} {q} 2.0 {q} {q} 2.0 4.0 1.0 10000 {100 + i}" for i, q in enumerate([-0.2, -0.1, 0.0, 0.1, 0.2]))
    (data_root / "SMS_q4_841001.bt7").write_text(header + rows + "\n", encoding="utf-8")

    report = dispersion_scan_plan({"data_roots": [str(data_root)], "output_dir": str(tmp_path / "plan"), "e_min_mev": 2.0, "e_max_mev": 8.0})

    assert report["status"] == "ok"
    energies = {scan["energy_mev"] for scan in report["proposed_scans"]}
    assert 2.0 in energies and 6.0 in energies and 8.0 in energies
    assert all(scan["step_rlu"] > 0 for scan in report["proposed_scans"])
    # Precedent-mined alignment heuristics ride along.
    assert "A3" in report["alignment_scan_heuristics"]
    assert report["alignment_scan_heuristics"]["A3"]["step_deg"] == 0.2
    assert (tmp_path / "plan/dispersion_scan_plan.csv").exists()


def test_parse_scan_file_reads_nice_0_2_header(tmp_path: Path) -> None:
    """BT-7 files written by NICE 0.2 (2020+) carry '# key value' headers and a
    '## pt A3(deg) ...' column line instead of the ICE '#Date' / '#Columns' pair."""
    from tas_copilot_core.workflows.experiment_dashboard import _parse_scan_file, _record_datetime

    path = tmp_path / "Ima2_super_mesh518.dat.bt7"
    path.write_text(
        "# NICE 0.2 : 9 header lines\n"
        "# numPoints 6\n"
        "# date 2026-09-04T05:14:26.637-04:00 \n"
        "# filename Ima2_super_mesh518.dat.bt7 \n"
        "# experiment.instrument BT7 \n"
        "# experiment.proposalId 16 \n"
        "# experiment.title Search for Ferrielectricity \n"
        "# primaryDetector singleDetector \n"
        "# trajectory.config {   \"init\": [ [ \"hierarchy\", \"'loop1: A3= i:-16 e:131 s:0.1'\" ] ] }\n"
        "# trajectory.controls [A3] \n"
        "# trajectory.name Ima2_super_mesh \n"
        "## pt    A3(deg)    A4(deg)    monitor   Detector    time(s)    temp(K) temp.set1(K)    et(meV)\n"
        "    1    -16.000     26.400       4800         21      4.075      2.790        1.000       0e-3\n"
        "    2    -15.900     26.400       4800         40      4.080      2.780        1.000       0e-3\n"
        "    3    -15.800     26.400       4800         19      4.070      2.760        1.000       0e-3\n"
        "    4    -15.700     26.400       4800         22      4.070      2.770        1.000       0e-3\n"
        "    5    -15.600     26.400       4800         18      4.070      2.780        1.000       0e-3\n"
        "    6    -15.500     26.400       4800         20      4.070      2.775        1.000       0e-3\n",
        encoding="utf-8",
    )

    record = _parse_scan_file(path, max_plot_points=50)

    assert record.scan_id == "Ima2_super_mesh518.dat.bt7"
    assert _record_datetime(record) is not None and _record_datetime(record).isoformat() == "2026-09-04T05:14:26"
    assert record.column_names == ["Pt.", "A3", "A4", "Monitor", "Detector", "Time", "Temp", "TemperatureSetpoint", "E"]
    assert record.scan_variable == "A3"
    assert record.row_count == 6
    assert record.instrument == "BT7"
    assert record.experiment_id == "16"
    assert record.title == "Ima2_super_mesh"
    assert record.detector_mode == "singleDetector"
    assert record.measurement_kind == "elastic_or_diffraction"
    assert record.temperature_range["min"] == 2.76 and record.temperature_range["max"] == 2.79
    assert record.intensity_summary["status"] == "ok"
    assert record.metadata["scan_description"] == "Title=Ima2_super_mesh:Range=A3=loop1: A3= i:-16 e:131 s:0.1"


def _nice_file(tmp_path, name, scanned, columns, rows, extra_header=""):
    path = tmp_path / name
    body = (
        "# NICE 0.2 : 7 header lines\n"
        f"# numPoints {len(rows)}\n"
        "# date 2026-10-25T11:08:40.406-04:00 \n"
        f"# filename {name} \n"
        "# experiment.instrument BT7 \n"
        "# experiment.proposalId 30210 \n"
        "# primaryDetector singleDetector \n"
        f"# trajectory.controls [{scanned}] \n"
        f"# trajectory.scanned [{scanned}] \n"
        f"# trajectory.name {name.split('_')[0]}_ \n"
        + extra_header
        + "## pt " + " ".join(columns) + "\n"
        + "".join(" ".join(str(v) for v in row) + "\n" for row in rows)
    )
    path.write_text(body, encoding="utf-8")
    return path


def test_nice_device_names_classify_tilt_slit_cooling_and_jittered_q_scans(tmp_path):
    """BT-7 under NICE 0.2 (experiment 30210, Oct 2026) names devices
    sampleLTilt, postSampleSlitWidth and trajectoryData._points, and logs a K
    scan with H jittering by 0.001 rlu. All of these were 'unclassified' on the
    live summary page."""
    from tas_copilot_core.workflows.experiment_dashboard import _overnight_family, _parse_scan_file

    tilt = _nice_file(tmp_path, "_sampleLTilt_992.dat.bt7", "sampleLTilt",
                      ["sampleLTilt(deg)", "monitor", "Detector", "time(s)", "et(meV)"],
                      [[i, -4 + i, 743, 850 + 40 * i, 1.0, "0e-3"] for i in range(9)])
    slit = _nice_file(tmp_path, "fp_backSlitWidth_1005.dat.bt7", "postSampleSlitWidth",
                      ["postSampleSlitWidth(mm)", "monitor", "Detector", "time(s)"],
                      [[i, 10 + 5 * i, 700, 600 + 30 * i, 1.0] for i in range(7)])
    cooling = _nice_file(tmp_path, "Cooling_1p5_0_0_1011.dat.bt7", "trajectoryData._points",
                         ["trajectoryData._points", "monitor", "Detector", "time(s)", "temp(K)", "et(meV)"],
                         [[i, i, 4000, 7 + i, 5.5, 89.6 - 0.6 * i, "0e-3"] for i in range(1, 60)])
    kscan = _nice_file(tmp_path, "Kscan_1p5K0_1083.dat.bt7", "Q.H, Q.K, Q.L, et",
                       ["Q.H(rlu)", "Q.K(rlu)", "Q.L(rlu)", "et(meV)", "ei(meV)", "ef(meV)", "monitor", "Detector", "time(s)", "temp(K)"],
                       [[i, 1.499 if i % 2 else 1.500, round(-0.04 + 0.002 * i, 3), "0e-3", "0e-3", 14.7, 14.7, 1000, 20 + i, 1.0, 27.0] for i in range(41)])

    records = {p.name: _parse_scan_file(p, max_plot_points=40) for p in (tilt, slit, cooling, kscan)}
    families = {name: _overnight_family(rec) for name, rec in records.items()}
    assert families["_sampleLTilt_992.dat.bt7"] == "alignment_tilt_scan"
    assert families["fp_backSlitWidth_1005.dat.bt7"] == "alignment_slit_or_translation_scan"
    assert families["Cooling_1p5_0_0_1011.dat.bt7"] == "temperature_dependence_or_order_parameter"
    assert families["Kscan_1p5K0_1083.dat.bt7"] == "elastic_q_scan_or_reciprocal_space_map"
    assert records["Kscan_1p5K0_1083.dat.bt7"].scan_variable == "K"  # not the jittering H
