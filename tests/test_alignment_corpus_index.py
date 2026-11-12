import json
from pathlib import Path

from tas_copilot_core.cli import main
import pytest

from tas_copilot_core.catalogs import ReviewOutcomeSQLiteStore
from tas_copilot_core.workflows.alignment_corpus_index import _classify_scan, alignment_corpus_index
from tas_copilot_core.workflows.alignment_review_dashboard import build_alignment_review_dashboard
from tas_copilot_core.workflows.alignment_state import _a4_window_preview, alignment_adaptive_nice_loop, alignment_fine_replay_plan, alignment_planning_brief, alignment_state_epochs, alignment_state_machine


def _write_bt7(path: Path, *, scan_descr: str, scan_type: str = "Find Peak Scan") -> None:
    path.write_text(
        "\n".join(
            [
                "#ICE           0.4.0",
                f"#Filename    {path.stem}",
                "#Date        2019-11-06 16:14:59 EDT",
                "#ExptID      26202",
                "#ExptName    Alignment fixture",
                f"#ScanType   {scan_type}",
                "#Scan        12 A3",
                "#Lattice     3.9 3.9 3.9 90 90 90",
                "#Orient      1 0 0 0 1 0",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                f"#ScanDescr   {scan_descr}",
                "#Columns A3 A4 Detector Monitor Time",
                "0 0 10 100 1",
            ]
        ),
        encoding="utf-8",
    )


def test_alignment_corpus_index_classifies_scans_and_logs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "fpx0001.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=SmplUTilt=0 0.5:Npts=17:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )
    _write_bt7(
        data_root / "fpx0002.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=BkSltHght=1.0 5.0:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )
    (data_root / "2019_11_06.console.log").write_text(
        "\n".join(
            [
                "06-Nov 16:14:59> findpeak SmplUTilt 8.0 0.5 Time 1.0 Detector -func Gauss",
                "06-Nov 16:17:09> move A3 20.138 A4 40.276",
                "06-Nov 16:40:38> findpeak A4 4.0 0.2 Time 1.0 Detector -bragg A3 -func Gauss",
                "06-Nov 16:43:22> Device [A3] will move with [A4]",
                "06-Nov 17:01:57> sample setlattice -a 3.9 -b 3.9 -c 3.9 -alpha 90 -beta 90 -gamma 90",
            ]
        ),
        encoding="utf-8",
    )

    result = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "out")})

    assert result["status"] == "success"
    assert result["summary"]["alignment_scan_events"] == 2
    assert result["summary"]["alignment_log_events"] == 5
    assert result["summary"]["alignment_stages"]["second_reflection_search_or_plane_flattening"] == 1
    assert result["summary"]["alignment_stages"]["fine_alignment"] == 1
    assert result["alignment_process_model"]["representation"] == "temporal_evidence_graph_plus_alignment_state_machine"
    events = (tmp_path / "out" / "alignment_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 7
    assert any(json.loads(line)["event_type"] == "manual_search_move" for line in events)


def test_alignment_corpus_index_cli(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "rocking.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A3=34.395 0.2:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )

    assert main(["workflow", "alignment-corpus-index", "--data-root", str(data_root), "--output-dir", str(tmp_path / "out"), "--no-logs"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow_name"] == "alignment-corpus-index"
    assert payload["summary"]["alignment_scan_events"] == 1
    assert payload["summary"]["alignment_log_events"] == 0


def test_alignment_corpus_index_zero_max_files_means_no_cap(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for index in range(6):
        _write_bt7(
            data_root / f"fpx{index:04d}.bt7",
            scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A3=34.395 0.2:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
        )

    result = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "out"), "max_files": 0, "include_logs": False})

    assert result["summary"]["bt7_files_seen"] == 6
    assert result["summary"]["alignment_scan_events"] == 6


def test_alignment_state_epochs_emit_review_template(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "fpx_a3_a4.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=39.8 0.2:Range=A3=19.9 0.1:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )
    _write_bt7(
        data_root / "fpx_tilt.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=SmplLTilt=-1 1:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )
    (data_root / "2019_11_06.console.log").write_text(
        "\n".join(
            [
                "06-Nov 16:14:59> findpeak A4 4.0 0.2 Time 1.0 Detector -bragg A3 -func Gauss",
                "06-Nov 16:30:00> move A3 20.138 A4 40.276",
                "06-Nov 17:01:57> sample setlattice -a 3.9 -b 3.9 -c 3.9 -alpha 90 -beta 90 -gamma 90",
                "06-Nov 17:10:00> findpeak SmplLTilt 2.0 0.2 Time 1.0 Detector -func Gauss",
            ]
        ),
        encoding="utf-8",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index")})

    result = alignment_state_epochs({"events_jsonl": index["artifacts"]["events_jsonl"], "output_dir": str(tmp_path / "epochs")})

    assert result["status"] == "success"
    assert result["summary"]["epoch_count"] >= 2
    assert result["summary"]["review_template_rows"] >= 2
    rows = [(json.loads(line)) for line in (tmp_path / "epochs" / "alignment_review_template.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["object_type"] for row in rows} >= {"alignment_state_transition", "alignment_peak_candidate"}
    assert all(row["verdict"] == "unreviewed" for row in rows)


def test_alignment_review_dashboard_builds_html_and_seed_label_store(tmp_path: Path) -> None:
    template = tmp_path / "alignment_review_template.jsonl"
    rows = [
        {
            "object_type": "alignment_state_transition",
            "object_id": "alignment-epoch-001",
            "source_workflow": "alignment-state-epochs",
            "verdict": "unreviewed",
            "claim_state": "candidate",
            "type_specific_allowed": ["accepted_continuous_epoch", "rejected_epoch_break", "needs_more_context"],
            "evidence_channels_relied_on": ["alignment_event_timeline", "scan_recipe"],
            "metadata": {
                "dominant_stage": "fine_alignment",
                "confidence": 0.74,
                "devices": {"A3": 3, "A4": 2},
                "break_reasons": [],
                "event_ref": "fpx0001.bt7",
            },
            "provenance": [],
        },
        {
            "object_type": "alignment_peak_candidate",
            "object_id": "scan:fpx0001.bt7:peak",
            "source_workflow": "alignment-state-epochs",
            "verdict": "unreviewed",
            "claim_state": "candidate",
            "type_specific_allowed": ["accepted_peak_assignment", "rejected_powder_or_background", "rejected_lambda_half_alias", "needs_more_context"],
            "evidence_channels_relied_on": ["peak_fit", "a4_prediction", "filter_state"],
            "metadata": {
                "alignment_stage": "first_or_second_reflection_localization",
                "confidence": 0.61,
                "scanned_devices": ["A4"],
                "filename": "fpx0001.bt7",
            },
            "provenance": [],
        },
        {
            "object_type": "alignment_plane_candidate",
            "object_id": "alignment-epoch-002:plane",
            "source_workflow": "alignment-state-epochs",
            "verdict": "unreviewed",
            "claim_state": "candidate",
            "type_specific_allowed": ["accepted_plane_hypothesis", "rejected_plane_hypothesis", "needs_third_reflection", "needs_more_context"],
            "evidence_channels_relied_on": ["tilt_scan", "second_reflection"],
            "metadata": {
                "dominant_stage": "second_reflection_search_or_plane_flattening",
                "confidence": 0.52,
                "devices": {"SmplLTilt": 1},
            },
            "provenance": [],
        },
    ]
    template.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    result = build_alignment_review_dashboard({"review_template": str(template), "output_dir": str(tmp_path / "dashboard"), "max_dashboard_rows": 2})

    assert result["status"] == "ok"
    assert result["review_row_count"] == 3
    assert result["dashboard_row_count"] == 2
    assert Path(result["artifacts"]["html"]).exists()
    assert Path(result["artifacts"]["review_rows_csv"]).exists()
    html = Path(result["artifacts"]["html"]).read_text(encoding="utf-8")
    assert "BT-7 Alignment Review Dashboard" in html
    assert "Export reviewed JSONL" in html
    assert "Sample removal/remounting" in html
    assert "Dashboard rows: <code>2</code> of <code>3</code>" in html
    labels = ReviewOutcomeSQLiteStore(tmp_path / "dashboard" / "review_outcomes.sqlite").list_labels()
    assert {label.object_type for label in labels} == {"alignment_state_transition", "alignment_peak_candidate", "alignment_plane_candidate"}
    peak = [label for label in labels if label.object_type == "alignment_peak_candidate"][0]
    assert "rejected_lambda_half_alias" in peak.type_specific_allowed

    uncapped = build_alignment_review_dashboard({"review_template": str(template), "output_dir": str(tmp_path / "dashboard_uncapped"), "max_dashboard_rows": 0})
    assert uncapped["dashboard_row_count"] == 3


def test_alignment_review_dashboard_cli_roundtrip(tmp_path: Path, capsys) -> None:
    template = tmp_path / "alignment_review_template.jsonl"
    template.write_text(
        json.dumps(
            {
                "object_type": "alignment_peak_candidate",
                "object_id": "scan:fpx001.bt7:peak",
                "source_workflow": "alignment-state-epochs",
                "verdict": "unreviewed",
                "claim_state": "candidate",
                "metadata": {"dominant_stage": "first_or_second_reflection_localization"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["workflow", "alignment-review-dashboard", "--review-template", str(template), "--output-dir", str(tmp_path / "dashboard")]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["workflow_name"] == "alignment-review-dashboard"
    assert payload["status"] == "ok"
    assert payload["review_row_count"] == 1
    assert (tmp_path / "dashboard" / "alignment_review_dashboard.html").exists()


def test_alignment_planning_brief_and_replay_plan(tmp_path: Path) -> None:
    events_path = tmp_path / "alignment_events.jsonl"
    events = [
        {
            "event_id": "scan:first",
            "event_type": "planned_scan",
            "source_type": "bt7_file",
            "source_path": "first.bt7",
            "filename": "first.bt7",
            "alignment_stage": "first_or_second_reflection_localization",
            "alignment_intent": "theta_two_theta_or_bragg_coupled_scan",
            "automation_level": "supervised_autonomy_candidate",
            "evidence_strength": "strong_scan_data",
            "state_epoch_effect": "may_refine_peak_center",
            "scanned_devices": ["A3", "A4"],
            "warnings": [],
        },
        {
            "event_id": "scan:slit",
            "event_type": "planned_scan",
            "source_type": "bt7_file",
            "source_path": "slit.bt7",
            "filename": "slit.bt7",
            "alignment_stage": "fine_alignment",
            "alignment_intent": "slit_translation_optimization",
            "automation_level": "supervised_autonomy_candidate",
            "evidence_strength": "strong_scan_data",
            "state_epoch_effect": "refines_sample_centering_or_beam_acceptance",
            "scanned_devices": ["BkSltHght"],
            "warnings": [],
        },
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    peak_report = tmp_path / "alignment_peak_report.json"
    peak_report.write_text(json.dumps({"formula": "La0.7Sr0.3MnO3", "scattering_plane": "[1,0,0];[0,0,1]", "alignment_peaks": [{"hkl": [0, 0, 1]}]}), encoding="utf-8")
    epochs = alignment_state_epochs({"events_jsonl": str(events_path), "output_dir": str(tmp_path / "epochs")})

    brief = alignment_planning_brief(
        {
            "events_jsonl": str(events_path),
            "state_json": epochs["artifacts"]["json"],
            "alignment_peak_report": str(peak_report),
            "output_dir": str(tmp_path / "brief"),
        }
    )
    replay = alignment_fine_replay_plan({"state_json": epochs["artifacts"]["json"], "brief_json": brief["artifacts"]["json"], "output_dir": str(tmp_path / "replay")})

    assert brief["planning_evidence"]["status"] == "loaded"
    assert brief["observational_evidence"]["empirical_state"] == "fine_alignment_candidate"
    assert any(action["action"] == "prepare_simulator_only_fine_alignment_replay" for action in brief["recommended_next_actions"])
    assert replay["simulation_only"] is True
    assert replay["replay_plan"]["mode"] == "simulation_only"
    assert replay["replay_plan"]["requires_reviewed_node_aliases"] is True


def test_alignment_planning_brief_skips_accessibility_without_ef(tmp_path: Path) -> None:
    peak_report = tmp_path / "alignment_peak_report.json"
    peak_report.write_text(
        json.dumps({"formula": "SrTiO3", "alignment_peaks": [{"peak_id": "hkl:1,0,0", "q_inv_angstrom": 1.6}]}),
        encoding="utf-8",
    )

    brief = alignment_planning_brief({"alignment_peak_report": str(peak_report), "output_dir": str(tmp_path / "brief")})

    assert brief["kinematic_accessibility"]["status"] == "skipped_missing_ef_mev"
    assert "ef_mev" in brief["kinematic_accessibility"]["note"]


def test_alignment_planning_brief_annotates_peak_reachability(tmp_path: Path) -> None:
    pytest.importorskip("tas_psd_planner")
    peak_report = tmp_path / "alignment_peak_report.json"
    peak_report.write_text(
        json.dumps(
            {
                "formula": "SrTiO3",
                "scattering_plane": "[1,0,0];[0,1,0]",
                "alignment_peaks": [
                    {"peak_id": "hkl:1,0,0", "q_inv_angstrom": 1.609},
                    {"peak_id": "hkl:4,4,0", "q_inv_angstrom": 9.1},
                ],
            }
        ),
        encoding="utf-8",
    )

    brief = alignment_planning_brief(
        {
            "alignment_peak_report": str(peak_report),
            "ef_mev": 14.7,
            "output_dir": str(tmp_path / "brief"),
        }
    )

    accessibility = brief["kinematic_accessibility"]
    assert accessibility["status"] == "annotated"
    assert accessibility["annotated_peak_count"] == 2
    assert accessibility["reachable_peak_count"] == 1
    assert accessibility["parity_status"] == "ported_awaiting_matlab_fixture_check"
    candidates = brief["planning_evidence"]["top_candidates"]
    low_q, high_q = candidates
    assert low_q["kinematic_accessibility"]["reachable_at_ef_a4_limits"] is True
    assert high_q["kinematic_accessibility"]["reachable_at_ef_a4_limits"] is False
    markdown = Path(brief["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "Kinematic Accessibility" in markdown
    assert "reachable=`True`" in markdown


def test_alignment_state_cli_roundtrip(tmp_path: Path, capsys) -> None:
    events_path = tmp_path / "alignment_events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event_id": "scan:rocking",
                "event_type": "planned_scan",
                "source_type": "bt7_file",
                "source_path": "rocking.bt7",
                "filename": "rocking.bt7",
                "alignment_stage": "first_or_second_reflection_localization",
                "alignment_intent": "rocking_scan",
                "automation_level": "supervised_autonomy_candidate",
                "evidence_strength": "strong_scan_data",
                "state_epoch_effect": "may_refine_peak_center",
                "scanned_devices": ["A3"],
                "warnings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["workflow", "alignment-state-epochs", "--events-jsonl", str(events_path), "--output-dir", str(tmp_path / "epochs")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow_name"] == "alignment-state-epochs"
    assert payload["summary"]["epoch_count"] == 1


def test_committed_alignment_fixture_vertical_slice(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "alignment_corpus" / "session_2019_style"
    index = alignment_corpus_index(
        {
            "data_root": str(fixture),
            "alignment_peak_report": str(fixture / "alignment_peak_report.json"),
            "output_dir": str(tmp_path / "index"),
        }
    )
    epochs = alignment_state_epochs({"events_jsonl": index["artifacts"]["events_jsonl"], "output_dir": str(tmp_path / "epochs")})
    brief = alignment_planning_brief(
        {
            "events_jsonl": index["artifacts"]["events_jsonl"],
            "state_json": epochs["artifacts"]["json"],
            "alignment_peak_report": str(fixture / "alignment_peak_report.json"),
            "output_dir": str(tmp_path / "brief"),
        }
    )

    assert index["summary"]["alignment_scan_events"] == 3
    assert index["structure_factor_planning_link"]["status"] == "loaded"
    assert epochs["summary"]["epoch_count"] >= 2
    assert epochs["review_template"]["object_types"] == ["alignment_state_transition", "alignment_peak_candidate", "alignment_plane_candidate"]
    assert brief["planning_evidence"]["formula"] == "La0.7Sr0.3MnO3"
    assert brief["observational_evidence"]["event_count"] == index["summary"]["alignment_event_count"]


@pytest.mark.parametrize(
    ("scan_descr", "scan", "scan_type", "filename", "expected_stage", "expected_intent"),
    [
        ("Scan:Type=6:Range=A3=34.395 0.2:Npts=21:Title=FindPeak", "12 A3", "Find Peak Scan", "fpx0001.bt7", "first_or_second_reflection_localization", "rocking_scan"),
        ("Scan:Type=6:Range=A4=45.0 0.25:Npts=31:Title=FindPeak", "13 A4", "Find Peak Scan", "fpx0002.bt7", "gross_reflection_search_or_two_theta_localization", "sample_two_theta_scan"),
        ("Scan:Type=6:Range=A4=45.0 0.25:Range=A3=22.5 0.125:Npts=31:Title=FindPeak", "14 A4 A3", "Find Peak Scan", "fpx0003.bt7", "first_or_second_reflection_localization", "theta_two_theta_or_bragg_coupled_scan"),
        ("Scan:Type=6:Range=SmplLTilt=-1 1:Npts=21:Title=FindPeak", "15 SmplLTilt", "Find Peak Scan", "fpx0004.bt7", "second_reflection_search_or_plane_flattening", "tilt_axis_search_or_refinement"),
        ("Scan:Type=6:Range=SmplUTilt=-0.5 0.5:Npts=17:Title=FindPeak", "16 SmplUTilt", "Find Peak Scan", "fpx0005.bt7", "second_reflection_search_or_plane_flattening", "tilt_axis_search_or_refinement"),
        ("Scan:Type=6:Range=BkSltHght=0 50:Npts=26:Title=Slit", "17 BkSltHght", "Alignment Scan", "slit0001.bt7", "fine_alignment", "slit_translation_optimization"),
        ("Scan:Type=6:Range=BkSltWdth=0 50:Npts=26:Title=Slit", "18 BkSltWdth", "Alignment Scan", "slit0002.bt7", "fine_alignment", "slit_translation_optimization"),
        ("Scan:Type=6:Range=SmplHght=-5 5:Npts=21:Title=Center", "19 SmplHght", "Alignment Scan", "center0001.bt7", "fine_alignment", "slit_translation_optimization"),
        ("Scan:Type=6:Range=FrontSlitWidth=0 50:Npts=26:Title=FrontSlit", "20 FrontSlitWidth", "Alignment Scan", "slit0003.bt7", "fine_alignment", "slit_translation_optimization"),
        ("Scan:Type=6:Range=QX=0 1:Npts=21:Title=FindPeak", "21 QX", "Find Peak Scan", "fpx_unknown.bt7", "unknown_alignment_action", "alignment_related_scan"),
    ],
)
def test_alignment_scan_descr_classification_regressions(scan_descr: str, scan: str, scan_type: str, filename: str, expected_stage: str, expected_intent: str) -> None:
    result = _classify_scan(scan_descr, scan, scan_type, filename)

    assert result["alignment_stage"] == expected_stage
    assert result["alignment_intent"] == expected_intent


def test_alignment_state_machine_flags_observed_prediction_contradiction(tmp_path: Path) -> None:
    events_path = tmp_path / "alignment_events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event_id": "scan:fit",
                "event_type": "planned_scan",
                "source_type": "bt7_file",
                "source_path": "fit.bt7",
                "filename": "fit.bt7",
                "fit_hkl": [1, 1, 1],
                "alignment_stage": "first_or_second_reflection_localization",
                "alignment_intent": "rocking_scan",
                "automation_level": "supervised_autonomy_candidate",
                "evidence_strength": "strong_scan_data",
                "state_epoch_effect": "may_refine_peak_center",
                "scanned_devices": ["A3"],
                "warnings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    peak_report = tmp_path / "alignment_peak_report.json"
    peak_report.write_text(json.dumps({"alignment_peaks": [{"hkl": [0, 0, 1]}]}), encoding="utf-8")

    result = alignment_state_machine({"events_jsonl": str(events_path), "alignment_peak_report": str(peak_report), "output_dir": str(tmp_path / "machine")})

    assert result["status"] == "success"
    assert result["state_estimates"][0]["state"] == "reflection_localization"
    assert result["contradiction_checks"][0]["status"] == "contradiction_detected"


def test_alignment_state_machine_peak_fit_and_a4_prediction_checks(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "a4_scan.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=40.0 0.5:Npts=5:DetectorType=Detector:Filename=fpx:Title=FindPeak",
    )
    (data_root / "a4_scan.bt7").write_text(
        "\n".join(
            [
                "#ICE           0.4.0",
                "#Filename    a4_scan",
                "#Date        2019-11-06 16:14:59 EDT",
                "#ExptID      alignment_fixture",
                "#ExptName    Sanitized alignment fixture",
                "#ScanType   Find Peak Scan",
                "#Scan        12 A4",
                "#Lattice     3.9 3.9 3.9 90 90 90",
                "#Orient      1 0 0 0 1 0",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                "#ScanDescr   Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=40.0 0.5:Npts=5:DetectorType=Detector:Filename=fpx:Title=FindPeak",
                "#Columns A4 Detector Monitor Time",
                "39.0 10 100 1",
                "39.5 25 100 1",
                "40.0 100 100 1",
                "40.5 25 100 1",
                "41.0 10 100 1",
            ]
        ),
        encoding="utf-8",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})
    peak_report = tmp_path / "alignment_peak_report.json"
    peak_report.write_text(json.dumps({"alignment_peaks": [{"hkl": [0, 0, 1], "two_theta_deg": 40.0}]}), encoding="utf-8")

    result = alignment_state_machine(
        {
            "events_jsonl": index["artifacts"]["events_jsonl"],
            "alignment_peak_report": str(peak_report),
            "output_dir": str(tmp_path / "machine"),
        }
    )

    assert result["peak_fit_diagnostics"][0]["status"] == "success"
    assert result["peak_fit_diagnostics"][0]["fits"][0]["center"] == 40.0
    assert result["peak_fit_diagnostics"][0]["model_selection"]["model_family"] == "internal_bic_gaussian_preview"
    assert result["peak_fit_diagnostics"][0]["heuristic_assessment"]["status"] == "single_peak_candidate"
    assert result["a4_prediction_checks"][0]["status"] == "a4_prediction_in_scan_window"
    assert result["lambda_half_and_background_scoring_inputs"]["needed_for_lambda_half_scoring"]


def test_alignment_state_machine_too_many_peaks_uses_configurable_heuristic(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    path = data_root / "multi_peak.bt7"
    path.write_text(
        "\n".join(
            [
                "#ICE           0.4.0",
                "#Filename    multi_peak",
                "#Date        2019-11-06 16:14:59 EDT",
                "#ExptID      alignment_fixture",
                "#ExptName    Sanitized alignment fixture",
                "#ScanType   Find Peak Scan",
                "#Scan        12 A3",
                "#Lattice     3.9 3.9 3.9 90 90 90",
                "#Orient      1 0 0 0 1 0",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                "#ScanDescr   Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A3=0 1:Npts=9:DetectorType=Detector:Filename=fpx:Title=FindPeak",
                "#Columns A3 Detector Monitor Time",
                "0 10 100 1",
                "1 80 100 1",
                "2 10 100 1",
                "3 75 100 1",
                "4 10 100 1",
                "5 70 100 1",
                "6 10 100 1",
                "7 65 100 1",
                "8 10 100 1",
            ]
        ),
        encoding="utf-8",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine({"events_jsonl": index["artifacts"]["events_jsonl"], "max_alignment_peaks": 3, "output_dir": str(tmp_path / "machine")})

    assert result["peak_fit_diagnostics"][0]["peak_count"] > 3
    assert result["peak_fit_diagnostics"][0]["heuristic_assessment"]["status"] == "too_many_peaks_bad_crystal_or_background_heuristic"


def test_alignment_state_machine_slit_scan_uses_ramp_saturation_profile(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    path = data_root / "slit.bt7"
    path.write_text(
        "\n".join(
            [
                "#ICE           0.4.0",
                "#Filename    slit",
                "#Date        2019-11-06 16:14:59 EDT",
                "#ExptID      alignment_fixture",
                "#ExptName    Sanitized alignment fixture",
                "#ScanType   Alignment Scan",
                "#Scan        12 BkSltHght",
                "#Lattice     3.9 3.9 3.9 90 90 90",
                "#Orient      1 0 0 0 1 0",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                "#ScanDescr   Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=BkSltHght=0 5:Npts=6:DetectorType=Detector:Filename=slit:Title=Slit",
                "#Columns BkSltHght Detector Monitor Time",
                "0 0 100 1",
                "1 0 100 1",
                "2 5 100 1",
                "3 20 100 1",
                "4 25 100 1",
                "5 26 100 1",
            ]
        ),
        encoding="utf-8",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine({"events_jsonl": index["artifacts"]["events_jsonl"], "output_dir": str(tmp_path / "machine")})

    diagnostic = result["peak_fit_diagnostics"][0]
    assert diagnostic["model_family"] in {"piecewise_linear_then_saturation", "tanh_s_curve"}
    assert diagnostic["profile"]["status"] == "late_uptick_possible_decentered_sample"
    assert diagnostic["profile"]["late_uptick"] is True
    assert diagnostic["profile"]["model_selection"]["selected_model_family"] in {"piecewise_linear_then_saturation", "tanh_s_curve"}
    assert diagnostic["profile"]["onset_offset_proxy_axis_units"] == 2.0
    assert diagnostic["profile"]["translation_recommendation"]["status"] == "review_required_possible_decentered_sample"
    assert diagnostic["profile"]["translation_recommendation"]["calibration_status"] == "heuristic_pending_labeled_real_sessions"


def test_alignment_state_machine_scores_lambda_half_and_powder_background(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    path = data_root / "a4_background.bt7"
    path.write_text(
        "\n".join(
            [
                "#ICE           0.4.0",
                "#Filename    a4_background",
                "#Date        2019-11-06 16:14:59 EDT",
                "#ExptID      alignment_fixture",
                "#ExptName    Sanitized alignment fixture",
                "#ScanType   Find Peak Scan",
                "#Scan        12 A4",
                "#Lattice     3.9 3.9 3.9 90 90 90",
                "#Orient      1 0 0 0 1 0",
                "#AnalyzerDetectorMode 2 SingDetFlat",
                "#ScanDescr   Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=39.0 0.5:Npts=5:DetectorType=Detector:Filename=fpx:Title=FindPeak",
                "#Columns A4 Detector Monitor Time FilTran Velsel",
                "38.0 10 100 1 OUT IN",
                "38.5 25 100 1 OUT IN",
                "39.0 100 100 1 OUT IN",
                "39.5 25 100 1 OUT IN",
                "40.0 10 100 1 OUT IN",
            ]
        ),
        encoding="utf-8",
    )
    peak_table = tmp_path / "background_peaks.json"
    peak_table.write_text(json.dumps({"peaks": [{"material": "Al", "hkl": [1, 1, 1], "two_theta_deg": 39.0, "relative_intensity": 100}]}), encoding="utf-8")
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine(
        {
            "events_jsonl": index["artifacts"]["events_jsonl"],
            "background_material": ["Al"],
            "background_peak_table": str(peak_table),
            "output_dir": str(tmp_path / "machine"),
        }
    )

    scores = result["lambda_half_and_background_scores"]
    assert scores["status"] == "scored"
    assert scores["filter_state_evidence"] == ["OUT"]
    assert scores["velocity_selector_evidence"] == ["IN"]
    assert scores["powder_background_scores"][0]["status"] == "candidate_powder_or_background_overlap"
    assert scores["powder_background_scores"][0]["matches"][0]["material"] == "Al"


def test_alignment_adaptive_nice_loop_prepares_dry_run_rehearsal(tmp_path: Path) -> None:
    state_json = tmp_path / "state.json"
    state_json.write_text(
        json.dumps(
            {
                "state_epochs": [
                    {
                        "epoch_id": "alignment-epoch-001",
                        "current_state_estimate": {"state": "fine_alignment_or_sample_centering"},
                    }
                ],
                "state_estimates": [{"epoch_id": "alignment-epoch-001", "state": "fine_alignment_or_sample_centering", "confidence": 0.7}],
            }
        ),
        encoding="utf-8",
    )

    result = alignment_adaptive_nice_loop(
        {
            "state_json": str(state_json),
            "output_dir": str(tmp_path / "adaptive"),
            "max_iterations": 2,
            "execute_dry_run": True,
        }
    )

    assert result["status"] == "success"
    assert result["simulation_only"] is True
    assert len(result["iterations"]) == 2
    assert result["nice_replay_executions"]
    assert result["nice_replay_executions"][0]["status"] == "success"


def test_alignment_adaptive_nice_loop_can_execute_mock_simulator_path(tmp_path: Path) -> None:
    state_json = tmp_path / "state.json"
    state_json.write_text(
        json.dumps(
            {
                "state_epochs": [{"epoch_id": "alignment-epoch-001", "current_state_estimate": {"state": "fine_alignment_or_sample_centering"}}],
                "state_estimates": [{"epoch_id": "alignment-epoch-001", "state": "fine_alignment_or_sample_centering", "confidence": 0.7}],
            }
        ),
        encoding="utf-8",
    )

    result = alignment_adaptive_nice_loop(
        {
            "state_json": str(state_json),
            "output_dir": str(tmp_path / "adaptive_mock"),
            "executor": "mock",
            "allow_simulator_execution": True,
            "max_iterations": 1,
        }
    )

    assert result["status"] == "success"
    assert result["planning_only"] is False
    assert result["executor"] == "mock"
    assert result["nice_replay_executions"][0]["execution"]["status"] == "mock_executed"


def test_alignment_fine_replay_plan_writes_nice_compatible_plan(tmp_path: Path) -> None:
    state_json = tmp_path / "state.json"
    state_json.write_text(
        json.dumps(
            {
                "state_epochs": [
                    {
                        "epoch_id": "alignment-epoch-001",
                        "current_state_estimate": {"state": "fine_alignment_or_sample_centering"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = alignment_fine_replay_plan({"state_json": str(state_json), "output_dir": str(tmp_path / "replay")})

    assert result["nice_replay_training_plan"]["workflow_boundary"] == "simulation_only_training_harness"
    assert result["nice_replay_training_plan"]["recipes"][0]["translation_status"] == "reviewed_simulator_aliases"
    assert (tmp_path / "replay" / "alignment_nice_replay_training_plan.json").exists()


def test_a4_window_preview_treats_findpeak_range_value_as_center() -> None:
    # Ground truth: bt7/201804/24858/data/fpx840982.bt7 records this exact ScanDescr
    # and its A4 column sweeps 33.5374 -> 37.5374, i.e. the Range value is the CENTER.
    event = {
        "event_id": "scan:fpx840982.bt7",
        "scan_descr": "Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=35.5374 0.2:Npts=21:DetectorType=Detector:Filename=fpx:Range=A3=109.948 0.1:Title=FindPeak",
        "ranges": [{"device": "A4", "raw": "35.5374 0.2"}, {"device": "A3", "raw": "109.948 0.1"}],
    }

    window = _a4_window_preview(event)

    assert window["a4_center_deg"] == pytest.approx(35.5374)
    assert window["a4_min_deg"] == pytest.approx(33.5374)
    assert window["a4_max_deg"] == pytest.approx(37.5374)
    assert window["npts"] == 21


def test_a4_window_preview_start_stop_form_is_not_recentred() -> None:
    # ICE also emits explicit start/stop ranges with a trailing "s" flag
    # (e.g. Aluminum_backgroundRT840405.bt7 sweeps A4 exactly 38 -> 48).
    event = {
        "event_id": "scan:background",
        "scan_descr": "Scan:SubID=23020:JType=ANGLE:Npts=41:Counts=20.0:DetectorType=Detector:CountType=Time:Range=A4=38.0 48.0 s:Title=Backgound",
        "ranges": [{"device": "A4", "raw": "38.0 48.0 s"}],
    }

    window = _a4_window_preview(event)

    assert window["a4_min_deg"] == pytest.approx(38.0)
    assert window["a4_max_deg"] == pytest.approx(48.0)
    assert window["a4_center_deg"] == pytest.approx(43.0)


def test_alignment_state_machine_lambda_half_center_uses_findpeak_center(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "fpx840982_style.bt7",
        scan_descr="Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=35.5374 0.2:Npts=21:DetectorType=Detector:Filename=fpx:Range=A3=109.948 0.1:Title=FindPeak",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine({"events_jsonl": index["artifacts"]["events_jsonl"], "output_dir": str(tmp_path / "machine")})

    scores = result["lambda_half_and_background_scores"]
    assert scores["status"] == "scored"
    centers = [row["center_deg"] for row in scores["lambda_half_scores"]]
    assert 35.5374 in centers
    # The old start-based interpretation produced 37.5374; it must not reappear.
    assert 37.5374 not in centers


def test_read_bt7_numeric_table_keeps_rows_with_non_numeric_cells(tmp_path: Path) -> None:
    # Real BT-7 rows mix numeric cells with strings (Filtran=OUT, HKL text);
    # the parser must keep the numeric cells instead of dropping the row.
    from tas_copilot_core.workflows.alignment_state import _read_bt7_numeric_table

    path = tmp_path / "mixed.bt7"
    path.write_text(
        "#Columns A3 Detector Filtran HKL\n"
        "30.0 100 OUT 1,1,0\n"
        "30.1 200 OUT 1,1,0\n"
        "30.2 150 OUT 1,1,0\n",
        encoding="utf-8",
    )

    rows = _read_bt7_numeric_table(path)

    assert len(rows) == 3
    assert rows[0]["A3"] == pytest.approx(30.0)
    assert rows[1]["Detector"] == pytest.approx(200.0)
    assert "Filtran" not in rows[0]


def _write_nice_scan(path: Path, device: str, points: list[tuple[float, float]]) -> Path:
    """A minimal NICE-format scan file: '## pt <name> ...' and no '#Columns' line."""
    lines = [
        "# NICE 0.2 : 3 header lines",
        "# numPoints %d" % len(points),
        "# trajectory.scanned [%s]" % device,
        "## pt %s(mm) Detector Monitor" % device,
    ]
    lines += ["%d %.3f %.0f 150000" % (i + 1, x, y) for i, (x, y) in enumerate(points)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# A pre-sample slit width scan: dark until the aperture opens past the beam
# edge, then a ramp, then flat. Counts are the real ones from fp_preSampleWidth_750.
_SLIT_RAMP = [(1.0, 2974.0), (5.0, 20000.0), (10.0, 45000.0), (15.0, 65000.0),
              (19.0, 74084.0), (25.0, 74090.0), (35.0, 74100.0)]


def test_read_bt7_numeric_table_reads_nice_column_header(tmp_path: Path) -> None:
    # NICE writes '## pt <name> ...' and no '#Columns' line anywhere in the file.
    # Recognising only the ICE form returned zero rows for every modern scan,
    # and the caller dropped those with no diagnostic.
    from tas_copilot_core.workflows.alignment_state import _read_bt7_numeric_table

    path = _write_nice_scan(tmp_path / "fp_preSampleWidth_750.dat.bt7",
                            "preSampleSlitWidth", _SLIT_RAMP)

    rows = _read_bt7_numeric_table(path)

    assert len(rows) == len(_SLIT_RAMP)
    assert rows[0]["preSampleSlitWidth(mm)"] == pytest.approx(1.0)
    assert rows[-1]["Detector"] == pytest.approx(74100.0)


def test_read_bt7_numeric_table_fallback_reads_nice_without_the_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bt7grouping reader is the preferred path but is an optional plugin.
    # With it unavailable core must still read a NICE file rather than degrade
    # to the silent zero-row behaviour it had before.
    import sys

    from tas_copilot_core.workflows.alignment_state import _read_bt7_numeric_table

    monkeypatch.setitem(sys.modules, "bt7grouping", None)
    monkeypatch.setitem(sys.modules, "bt7grouping.reader", None)

    path = _write_nice_scan(tmp_path / "scan.dat.bt7", "preSampleSlitWidth", _SLIT_RAMP)
    rows = _read_bt7_numeric_table(path)

    assert len(rows) == len(_SLIT_RAMP)
    assert rows[0]["preSampleSlitWidth(mm)"] == pytest.approx(1.0)


def test_fit_axis_matches_device_name_against_unit_suffixed_column(tmp_path: Path) -> None:
    # trajectory.scanned reports 'preSampleSlitWidth'; the data column is
    # 'preSampleSlitWidth(mm)'. Comparing the two verbatim matches nothing.
    from tas_copilot_core.workflows.alignment_state import (
        _fit_axis_for_event,
        _read_bt7_numeric_table,
    )

    path = _write_nice_scan(tmp_path / "scan.dat.bt7", "preSampleSlitWidth", _SLIT_RAMP)
    table = _read_bt7_numeric_table(path)

    axis = _fit_axis_for_event({"scanned_devices": ["preSampleSlitWidth"]}, table)

    assert axis == "preSampleSlitWidth(mm)"


def test_slit_scan_is_not_fitted_as_a_gaussian_peak(tmp_path: Path) -> None:
    # The dangerous failure mode: a slit scan whose axis name is missing from
    # _SLIT_TRANSLATION_AXES goes down the Bragg-peak branch, which reports a
    # confident centre fitted to a monotonic saturation curve.
    from tas_copilot_core.workflows.alignment_state import _peak_fit_diagnostics

    path = _write_nice_scan(tmp_path / "fp_preSampleWidth_750.dat.bt7",
                            "preSampleSlitWidth", _SLIT_RAMP)
    event = {
        "event_type": "planned_scan",
        "event_id": "slit",
        "source_path": str(path),
        "scanned_devices": ["preSampleSlitWidth"],
    }

    diagnostics = _peak_fit_diagnostics([event], max_alignment_peaks=3, arguments={})

    assert len(diagnostics) == 1
    entry = diagnostics[0]
    assert entry["axis"] == "preSampleSlitWidth(mm)"
    assert entry["model_family"] == "piecewise_linear_then_saturation"
    # The Gaussian branch is the one that reports peaks; this must not be it.
    assert "peak_count" not in entry
    assert entry["profile"]["status"] == "ramp_saturation_candidate"


def test_slit_opening_recommendation_picks_the_knee_not_the_landmarks(tmp_path: Path) -> None:
    # Counts from fp_preSampleHeight_749, where the slit was in fact left at
    # 40.003 mm. The two numbers the profile already reported are both wrong to
    # use as a setting: the 90%-of-max knee at 28 mm throws away 9% of the beam,
    # and the plateau centre only lands near 40 because the scan stopped at 50.
    from tas_copilot_core.workflows.alignment_state import _slit_opening_recommendation

    x = [1.998, 3.998, 5.998, 7.999, 9.999, 11.999, 13.999, 16.000, 18.000,
         20.000, 22.000, 24.001, 26.001, 28.001, 30.001, 32.002, 34.002,
         36.002, 38.002, 40.003, 42.003, 44.003, 45.997, 47.997, 49.997]
    y = [6139, 14979, 22898, 29304, 34885, 40326, 45376, 49238, 53770,
         56786, 60480, 63272, 66230, 68428, 70172, 71748, 72594,
         73220, 73430, 74258, 74232, 74562, 73910, 74280, 74731]

    rec = _slit_opening_recommendation(x, y)

    assert rec["status"] == "recommended"
    assert rec["recommended_axis_value"] == pytest.approx(40.003)
    assert rec["fraction_at_recommendation"] > 0.99
    assert rec["tolerance_basis"] == "target_fraction"


def test_slit_opening_recommendation_survives_a_noisy_low_count_plateau() -> None:
    # A plateau of a few hundred counts scatters by ~4%, far more than the 1%
    # target. Judged against the target alone nothing holds the level and the
    # scan looks as if it never saturated, so the tolerance has to fall back to
    # counting statistics rather than refuse.
    import random

    from tas_copilot_core.workflows.alignment_state import _slit_opening_recommendation

    rng = random.Random(20260912)
    x = [2.0 * i for i in range(1, 31)]
    y = [
        rng.gauss(550.0 * min(1.0, value / 28.0), 23.0)
        for value in x
    ]

    rec = _slit_opening_recommendation(x, y)

    assert rec["status"] in {"recommended", "recommended_within_one_step"}
    assert rec["tolerance_basis"] == "counting_statistics"
    assert rec["recommended_axis_value"] == pytest.approx(28.0, abs=6.0)


def test_slit_opening_recommendation_flags_a_scan_that_never_saturates() -> None:
    # Still climbing at the widest point measured. Nothing in the data
    # distinguishes "the curve flattened" from "the scan ran out", so the value
    # falls at the edge of the range and must carry the flag rather than be
    # offered as a setting.
    from tas_copilot_core.workflows.alignment_state import _slit_opening_recommendation

    x = [2.0 * i for i in range(1, 12)]
    y = [1000.0 * i for i in range(1, 12)]

    rec = _slit_opening_recommendation(x, y)

    assert rec["status"] == "scan_range_may_be_short"
    assert rec["points_beyond_recommendation"] < 2
    assert "widen the scan" in rec["reason"]


def test_opening_recommendation_is_not_offered_for_a_sample_translation(tmp_path: Path) -> None:
    # A translation scan has the same ramp/saturation shape, but the sample is
    # moving through the beam rather than an aperture opening: the number worth
    # having is a centre, and an "opening" would be meaningless.
    from tas_copilot_core.workflows.alignment_state import (
        _intensity_column,
        _read_bt7_numeric_table,
        _slit_profile_preview,
    )

    path = _write_nice_scan(tmp_path / "fp_sampleUTrans_900.dat.bt7",
                            "sampleUTrans", _SLIT_RAMP)
    table = _read_bt7_numeric_table(path)
    profile = _slit_profile_preview(table, axis="sampleUTrans(mm)",
                                    intensity_key=_intensity_column(table))

    assert profile["opening_recommendation"]["status"] == "not_an_aperture_axis"


def test_unreadable_scan_file_is_reported_not_skipped(tmp_path: Path) -> None:
    # An aborted scan has a header and no rows. It must leave a diagnostic:
    # the silent skip is how an entire file-format generation went unnoticed.
    from tas_copilot_core.workflows.alignment_state import _peak_fit_diagnostics

    path = _write_nice_scan(tmp_path / "aborted.dat.bt7", "preSampleSlitWidth", [])
    event = {"event_type": "planned_scan", "event_id": "aborted", "source_path": str(path)}

    diagnostics = _peak_fit_diagnostics([event], max_alignment_peaks=3, arguments={})

    assert [entry["status"] for entry in diagnostics] == ["unreadable_file"]


def test_a4_prediction_checks_match_beyond_top_five_candidates(tmp_path: Path) -> None:
    # Which reflection a scan sat on is a recall question: weak workhorse
    # peaks ranked below the top five (e.g. (002)) must stay matchable.
    from tas_copilot_core.workflows.alignment_state import _a4_prediction_checks

    peaks = [
        {"h": 0, "k": 0, "l": 4, "two_theta_deg": 75.23, "score": 0.9},
        {"h": 1, "k": 1, "l": 4, "two_theta_deg": 106.0, "score": 0.8},
        {"h": 0, "k": 0, "l": 1, "two_theta_deg": 17.55, "score": 0.7},
        {"h": 0, "k": 0, "l": 3, "two_theta_deg": 54.49, "score": 0.6},
        {"h": 1, "k": 1, "l": 3, "two_theta_deg": 87.11, "score": 0.5},
        {"h": 0, "k": 0, "l": 2, "two_theta_deg": 35.54, "score": 0.4},  # rank 6
    ]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")
    events = [
        {
            "event_id": "scan:fpx840982.bt7",
            "scan_descr": "Scan:Type=6:Range=A4=35.5374 0.2:Npts=21:Title=FindPeak",
            "ranges": [{"device": "A4", "raw": "35.5374 0.2"}],
        }
    ]

    checks = _a4_prediction_checks(events=events, planning_report_path=str(report))

    assert checks[0]["status"] == "a4_prediction_in_scan_window"
    matched = {tuple(c.get("hkl", "").split(",")) if isinstance(c.get("hkl"), str) else tuple(c.get("hkl") or []) for c in checks[0]["matched_predictions"]}
    assert any("2" in str(m) for m in matched)


def test_a4_prediction_d_spacing_discrimination_resolves_close_reflections(tmp_path: Path) -> None:
    # (004) and (112) share a 1-degree A4 window at 14.7 meV; the fitted
    # center plus incident energy must separate them by d-spacing.
    from tas_copilot_core.workflows.alignment_state import _a4_prediction_checks

    peaks = [
        {"h": 0, "k": 0, "l": 4, "two_theta_deg": 75.23, "d_spacing": 1.9325},
        {"h": 1, "k": 1, "l": 2, "two_theta_deg": 73.55, "d_spacing": 1.9705},
    ]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")
    events = [
        {
            "event_id": "scan:fpx840998.bt7",
            "incident_energy_mev": 14.7,
            "scan_descr": "Scan:Type=6:Range=A4=74.9933 0.2:Npts=21:Title=FindPeak",
            "ranges": [{"device": "A4", "raw": "74.9933 0.2"}],
        }
    ]
    diagnostics = [{"event_id": "scan:fpx840998.bt7", "axis": "A4", "status": "success", "fits": [{"center": 74.99}]}]

    checks = _a4_prediction_checks(events=events, planning_report_path=str(report), peak_diagnostics=diagnostics)

    check = checks[0]
    assert check["discrimination_status"] == "resolved_by_d_spacing"
    assert check["best_candidate_hkl"] == "0,0,4"
    assert check["runner_up_hkl"] == "1,1,2"
    assert check["observed_center_source"] == "peak_fit_center"

    # Without incident energy the check stays honest instead of guessing.
    no_energy = [dict(events[0])]
    no_energy[0].pop("incident_energy_mev")
    checks2 = _a4_prediction_checks(events=no_energy, planning_report_path=str(report), peak_diagnostics=diagnostics)
    assert checks2[0]["discrimination_status"] == "no_incident_energy_metadata"


def test_contradiction_checks_use_d_resolved_assignments(tmp_path: Path) -> None:
    from tas_copilot_core.workflows.alignment_state import _contradiction_checks

    peaks = [
        {"h": 0, "k": 0, "l": 4, "two_theta_deg": 75.23, "d_spacing": 1.9325},
        {"h": 0, "k": 0, "l": 2, "two_theta_deg": 35.54, "d_spacing": 3.865},
    ]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")

    # Resolved-only event: becomes a reviewable candidate assignment.
    a4_checks = [
        {"event_id": "scan:a.bt7", "discrimination_status": "resolved_by_d_spacing", "best_candidate_hkl": "0,0,4", "best_candidate_d_mismatch_fraction": 0.003}
    ]
    checks = _contradiction_checks(events=[{"event_id": "scan:a.bt7"}], planning_report_path=str(report), a4_checks=a4_checks)
    assert checks[0]["status"] == "candidate_assignment_from_a4_and_d_spacing"
    assert checks[0]["d_spacing_assigned_hkl"] == "0,0,4"

    # Declared HKL disagreeing with the d-resolved assignment is a contradiction.
    events = [{"event_id": "scan:a.bt7", "metadata": {"HKL": "0 0 2"}}]
    checks = _contradiction_checks(events=events, planning_report_path=str(report), a4_checks=a4_checks)
    assert checks[0]["status"] == "contradiction_detected"
    assert checks[0]["human_review_required"] is True


def test_epoch_dominant_stage_prefers_specific_alignment_stages() -> None:
    from tas_copilot_core.workflows.alignment_state import _epoch_record

    events = [{"event_id": f"log:{i}", "alignment_stage": "recorded_or_queued_scan", "event_type": "recorded_scan"} for i in range(10)]
    events.append({"event_id": "scan:x", "alignment_stage": "first_or_second_reflection_localization", "event_type": "planned_scan", "scanned_devices": ["A3"]})

    epoch = _epoch_record(1, events, break_reasons=[])

    # Ten bookkeeping events must not outvote one informative stage.
    assert epoch["dominant_stage"] == "first_or_second_reflection_localization"


def test_planning_structure_consistency_flags_wrong_polymorph() -> None:
    from tas_copilot_core.workflows.alignment_state import _planning_structure_consistency

    events = [{"source_type": "bt7_file", "metadata": {"Lattice": "5.99022 5.99022 17.0073 90 90 120"}}]
    # beta-RuCl3-like cell selected by formula-only lookup: c is wildly off.
    planning = {"selected_structure_lattice": {"a": 6.02, "b": 6.02, "c": 5.65, "alpha": 90, "beta": 90, "gamma": 120}}
    result = _planning_structure_consistency(events=events, planning=planning)
    assert result["status"] == "lattice_mismatch"
    assert result["max_length_mismatch_fraction"] > 0.5

    planning_ok = {"selected_structure_lattice": {"a": 5.96, "b": 5.96, "c": 17.2, "alpha": 90, "beta": 90, "gamma": 120}}
    assert _planning_structure_consistency(events=events, planning=planning_ok)["status"] == "lattice_consistent"
    assert _planning_structure_consistency(events=[], planning=planning_ok)["status"] == "insufficient_metadata"


def test_contradiction_checks_classify_non_integral_hkl_as_offset(tmp_path: Path) -> None:
    from tas_copilot_core.workflows.alignment_state import _contradiction_checks

    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": [{"h": 0, "k": 0, "l": 6, "two_theta_deg": 48.6, "d_spacing": 2.867}]}), encoding="utf-8")
    events = [{"event_id": "log:x:1", "metadata": {"HKL": "0 0 6.1"}}]

    checks = _contradiction_checks(events=events, planning_report_path=str(report), a4_checks=[])

    assert checks[0]["status"] == "intentional_offset_scan_candidate"
    assert checks[0]["human_review_required"] is False


def test_build_epochs_breaks_on_long_time_gaps() -> None:
    from tas_copilot_core.workflows.alignment_state import _build_epochs

    events = [
        {"event_id": "scan:a", "date": "2013-08-10 12:00:00 EDT", "alignment_stage": "fine_alignment", "event_type": "planned_scan"},
        {"event_id": "scan:b", "date": "2013-08-10 14:00:00 EDT", "alignment_stage": "fine_alignment", "event_type": "planned_scan"},
        # Two months later: a probable remount in a log-less archive.
        {"event_id": "scan:c", "date": "2013-10-02 09:00:00 EDT", "alignment_stage": "fine_alignment", "event_type": "planned_scan"},
    ]

    epochs = _build_epochs(events)

    assert len(epochs) == 2
    assert "long_time_gap_possible_remount" in epochs[1]["break_reasons"]


def test_state_machine_warnings_powder_and_mismatch() -> None:
    from tas_copilot_core.workflows.alignment_state import _state_machine_warnings

    events = [{"source_type": "bt7_file", "metadata": {"ExptName": "Yb2Pt2Pb  Magnetic Diffraction Powder"}}]
    warnings = _state_machine_warnings(events=events, structure_consistency={"status": "lattice_mismatch", "max_length_mismatch_fraction": 0.66, "max_angle_mismatch_deg": 0.0})
    assert any("powder" in w.lower() for w in warnings)
    assert any("polymorph" in w or "lattice" in w for w in warnings)


def test_a3_relation_discrimination_resolves_ambiguous_window() -> None:
    # (004) vs (112) ambiguous by d-spacing alone; the A3 relation against a
    # same-day (110) anchor separates them. Angle math goes through the
    # tas.ubmatrix registry tool (icp-lattice-calculator); skip if absent.
    import pytest as _pytest

    from tas_copilot_core.workflows.alignment_state import _a3_relation_discrimination, _reciprocal_angle_via_registry

    lattice = {"a": 4.58947, "b": 4.58947, "c": 7.7189, "alpha": 90, "beta": 90, "gamma": 120}
    if _reciprocal_angle_via_registry(lattice, [1, 1, 0], [0, 0, 4]) is None:
        _pytest.skip("tas.ubmatrix.calculate_reciprocal_angle unavailable")

    lattice_raw = "4.58947 4.58947 7.7189 90 90 120"
    events = [
        {"event_id": "scan:anchor", "date": "2018-05-23 13:48:00 EDT", "ranges": [{"device": "A3", "raw": "31.0 0.1"}], "metadata": {"Lattice": lattice_raw}},
        {"event_id": "scan:amb", "date": "2018-05-23 13:38:00 EDT", "ranges": [{"device": "A3", "raw": "127.5 0.1"}], "metadata": {"Lattice": lattice_raw}},
    ]
    checks = [
        {"event_id": "scan:anchor", "discrimination_status": "resolved_by_d_spacing", "best_candidate_hkl": "1,1,0", "observed_center_deg": 62.0},
        {
            "event_id": "scan:amb",
            "discrimination_status": "ambiguous_d_spacing",
            "observed_center_deg": 74.99,
            "matched_predictions": [
                {"hkl": "0,0,4", "d_mismatch_fraction": 0.004},
                {"hkl": "1,1,2", "d_mismatch_fraction": 0.005},
            ],
        },
    ]

    _a3_relation_discrimination(checks, events)

    amb = checks[1]
    # (004) is 90 deg from (110) in azimuth; observed dA3 - dtheta = 90.0.
    # (112) sits ~57 deg from (110), inconsistent with the observed A3s.
    assert amb["a3_relation_status"] == "resolved"
    assert amb["discrimination_status"] == "resolved_by_a3_relation"
    assert amb["best_candidate_hkl"] == "0,0,4"


def test_wide_survey_sweeps_are_not_peak_assignments(tmp_path: Path) -> None:
    from tas_copilot_core.workflows.alignment_state import _a4_prediction_checks

    peaks = [{"h": 0, "k": 0, "l": 2, "two_theta_deg": 35.54, "d_spacing": 3.865}]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")
    events = [
        {
            "event_id": "scan:survey",
            "incident_energy_mev": 14.7,
            "scan_descr": "Scan:JType=ANGLE:Npts=131:Range=A4=25.0 90.0 s:Title=survey",
            "ranges": [{"device": "A4", "raw": "25.0 90.0 s"}],
        }
    ]

    checks = _a4_prediction_checks(events=events, planning_report_path=str(report), peak_diagnostics=[])

    assert checks[0]["discrimination_status"] == "survey_sweep_window_not_a_peak_assignment"


def test_static_a4_assigns_non_a4_scans(tmp_path: Path) -> None:
    # Tilt/slit/A3-rocking scans do not scan A4; the constant A4 column
    # defines the reflection they sit on.
    from tas_copilot_core.workflows.alignment_state import _a4_prediction_checks

    peaks = [{"h": 1, "k": 1, "l": 0, "two_theta_deg": 62.0, "d_spacing": 2.2947}]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")
    events = [
        {
            "event_id": "scan:tilt.bt7",
            "incident_energy_mev": 14.7,
            "scan_descr": "Scan:Type=6:Range=SmplLTilt=0 0.5:Npts=11:Title=FindPeak",
            "ranges": [{"device": "SmplLTilt", "raw": "0 0.5"}],
            "a4_static_deg": 62.0038,
        }
    ]

    checks = _a4_prediction_checks(events=events, planning_report_path=str(report), peak_diagnostics=[])

    assert checks and checks[0]["observed_center_source"] == "static_a4_column"
    assert checks[0]["best_candidate_hkl"] == "1,1,0"


def test_alignment_corpus_index_extracts_static_a4(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_bt7(
        data_root / "fpx_tilt.bt7",
        scan_descr="Scan:Type=6:FixedE=1:Range=SmplLTilt=0 0.5:Npts=3:Title=FindPeak",
    )
    # append A4/A3 columns via a custom file: write a minimal file directly
    (data_root / "tilt2.bt7").write_text(
        "#ICE           0.4.0\n#Filename      tilt2\n#Date          2018-05-23 12:41:00 EDT\n"
        "#InstrName     BT7\n#ExptID        24858\n"
        "#ScanDescr     Scan:Type=6:FixedE=1:Range=SmplLTilt=0 0.5:Npts=3:Title=FindPeak\n"
        "#Columns       SmplLTilt A3 A4 Time Monitor Detector\n"
        "-0.5 31.0 62.0038 1.0 1000 10\n0.0 31.0 62.0038 1.0 1000 40\n0.5 31.0 62.0038 1.0 1000 12\n",
        encoding="utf-8",
    )

    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    events = [json.loads(line) for line in Path(index["artifacts"]["events_jsonl"]).read_text(encoding="utf-8").splitlines()]
    tilt = next(e for e in events if "tilt2" in str(e.get("filename")))
    assert tilt["a4_static_deg"] == pytest.approx(62.0038)
    assert tilt["a3_static_deg"] == pytest.approx(31.0)


def test_candidate_matching_uses_event_energy(tmp_path: Path) -> None:
    # Planning report generated at 14.7 meV, but the scan ran at 28 meV:
    # (002) then sits at A4=25.55, not 35.54. Matching must recompute the
    # candidate position at the event's own energy.
    from tas_copilot_core.workflows.alignment_state import _a4_prediction_checks

    peaks = [{"h": 0, "k": 0, "l": 2, "two_theta_deg": 35.54, "d_spacing": 3.865}]
    report = tmp_path / "alignment_peak_report.json"
    report.write_text(json.dumps({"alignment_peaks": peaks}), encoding="utf-8")
    events = [
        {
            "event_id": "scan:a3rock28mev.bt7",
            "incident_energy_mev": 28.0,
            "scan_descr": "Scan:Type=6:Range=A3=102.8 0.1:Npts=11:Title=FindPeak",
            "ranges": [{"device": "A3", "raw": "102.8 0.1"}],
            "a4_static_deg": 25.55,
        }
    ]

    checks = _a4_prediction_checks(events=events, planning_report_path=str(report), peak_diagnostics=[])

    assert checks and checks[0]["best_candidate_hkl"] == "0,0,2"
    assert checks[0]["matched_predictions"][0]["expected_two_theta_deg"] == pytest.approx(25.55, abs=0.2)
