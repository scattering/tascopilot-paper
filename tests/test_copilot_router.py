from __future__ import annotations

import json
from pathlib import Path

import yaml

from tas_copilot_contracts import CopilotAskRequest, CopilotAskResponse
from tas_copilot_core.cli import main
from tas_copilot_core.workflows.copilot_router import copilot_ask, run_copilot_route_quality_eval


PROPOSAL_FIXTURE = "tests/fixtures/proposal_retrospective/ansto_dto_thin_film"


def test_copilot_router_overnight_next_actions_eval_loads() -> None:
    suite = yaml.safe_load(Path("eval/copilot_router_overnight_next_actions.yaml").read_text(encoding="utf-8"))

    assert suite["suite_id"] == "copilot_router_overnight_next_actions"
    task = suite["tasks"][0]
    assert task["expected_intent"] == "overnight_scan_summary"
    assert "probability_of_value" in task["expected_next_action_fields"]
    assert "no_live_control" in task["safety_expectations"]


def test_copilot_router_route_quality_eval_runs(tmp_path: Path) -> None:
    result = run_copilot_route_quality_eval(
        {
            "suite": "eval/copilot_router_route_quality.yaml",
            "output_dir": str(tmp_path / "route_quality"),
        }
    )

    assert result["status"] == "ok"
    assert result["suite_id"] == "copilot_router_route_quality"
    assert result["pass_count"] == result["task_count"]
    assert result["metrics"]["route_accuracy"] == 1.0
    assert result["metrics"]["envelope_completeness_rate"] == 1.0
    assert (tmp_path / "route_quality" / "copilot_route_quality_eval.json").exists()
    assert (tmp_path / "route_quality" / "copilot_route_quality_eval.md").exists()


def test_copilot_router_reports_missing_input_for_spurion_question(tmp_path: Path) -> None:
    result = copilot_ask({"question": "Could this narrow feature be a spurion?", "output_dir": str(tmp_path / "out")})

    assert result["status"] == "missing_input"
    assert result["route"]["intent"] == "spurion_diagnostic_survey"
    assert result["answer_envelope"]["uncertainty_class"] == "missing_input"
    assert result["answer_envelope"]["route_reasoning_snippet"]
    assert "data_root" in result["missing_inputs"]
    assert any("--data-root [DATA_ROOT]" in item for item in result["safe_next_actions"])
    assert result["routed_workflows"] == []
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_copilot_router_reports_workflow_backbone_status(tmp_path: Path) -> None:
    result = copilot_ask({"question": "So high level status?", "output_dir": str(tmp_path / "out")})

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "copilot_status"
    assert result["answer_envelope"]["grounding_status"] == "grounded_in_router_capability_inventory"
    assert result["routed_workflows"] == []
    assert result["workflow_backbone_status"]["workflow_backbone_assessment"] == "usable_for_bounded_copilot_interface"
    assert "live instrument control" in result["workflow_backbone_status"]["not_ready_for"]
    assert "NICE replay adapter loop validation in simulation-only mode" in result["workflow_backbone_status"]["required_before_broader_agent"]
    markdown = (tmp_path / "out" / "copilot_response.md").read_text(encoding="utf-8")
    assert "Workflow Backbone Status" in markdown


def test_copilot_router_starts_experiment_review_with_missing_input(tmp_path: Path) -> None:
    result = copilot_ask({"question": "Please do a full experiment review.", "output_dir": str(tmp_path / "out")})

    assert result["status"] == "missing_input"
    assert result["route"]["intent"] == "experiment_review"
    assert "data_root" in result["missing_inputs"]
    assert result["routed_workflows"] == []


def test_copilot_router_runs_spurion_survey_when_data_root_supplied(tmp_path: Path) -> None:
    data_root = tmp_path / "2018" / "12345" / "data"
    data_root.mkdir(parents=True)
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, detector=[100, 101, 650, 102, 100], details="CCR PG filter")

    result = copilot_ask(
        {
            "question": "Check this experiment for spurions and suspicious narrow features.",
            "data_root": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "max_files": 20,
            "top_k": 5,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "spurion_diagnostic_survey"
    calls = result["routed_workflows"]
    assert calls[0]["workflow_name"] == "survey-inelastic-spurion-evidence"
    assert calls[0]["status"] == "ok"
    assert "not a definitive artifact call" in result["answer"]
    assert result["not_approved_operations"]


def test_copilot_router_routes_scan_planning_to_proposal_workflow(tmp_path: Path) -> None:
    result = copilot_ask(
        {
            "question": "What scans should I try first for alignment and the 003 peak?",
            "fixture": PROPOSAL_FIXTURE,
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "scan_plan_suggestions"
    assert result["routed_workflows"][0]["workflow_name"] == "suggest-experiment-scans"
    assert "not an approved run list" in result["answer"]
    assert any("human instrument-scientist review" in item for item in result["not_approved_operations"])


def test_copilot_router_routes_alignment_evidence_index(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "alignment_corpus" / "session_2019_style"

    result = copilot_ask(
        {
            "question": "Find alignment evidence in this experiment and summarize alignment logs.",
            "data_root": str(fixture),
            "output_dir": str(tmp_path / "out"),
            "max_files": 50,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "alignment_evidence_index"
    assert result["routed_workflows"][0]["workflow_name"] == "alignment-corpus-index"
    assert "observational evidence" in result["answer"]


def test_copilot_router_uses_lsm_spinwave_precedent_index(tmp_path: Path) -> None:
    precedent_index = tmp_path / "precedents.jsonl"
    _write_lsm_precedent_index(precedent_index)

    result = copilot_ask(
        {
            "question": "I want to measure spinwaves in La.7Sr.3MnO3. What are good collimations and temperatures?",
            "precedent_index": str(precedent_index),
            "output_dir": str(tmp_path / "out"),
            "top_k": 3,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "precedent_experiment_planning"
    assert result["routed_workflows"][0]["workflow_name"] == "precedent-experiment-planning"
    assert "internal BT-7 precedent patterns" in result["answer"]
    workflow_report = json.loads((tmp_path / "out" / "spinwave_measurement_planning" / "report.json").read_text(encoding="utf-8"))
    assert workflow_report["candidates"][0]["candidate_id"] == "lsmo-precedent"
    assert workflow_report["candidates"][0]["why_ranked"]
    assert workflow_report["recommendations"]["top_precedents"][0]["candidate_id"] == "lsmo-precedent"
    assert workflow_report["recommendations"]["temperature_plan"]
    assert workflow_report["recommendations"]["collimation_plan"]
    assert workflow_report["evidence_channels"]["scan_precedents"]["status"] == "used"
    assert workflow_report["privacy_boundary"]["default_user_facing_policy"]
    assert workflow_report["evidence_channels"]["text_corpus"]["status"] == "not_supplied"
    weak = next(candidate for candidate in workflow_report["candidates"] if candidate["candidate_id"] == "weak-unknown-lsmo")
    assert weak["score"] < workflow_report["candidates"][0]["score"]
    markdown = (tmp_path / "out" / "spinwave_measurement_planning" / "report.md").read_text(encoding="utf-8")
    assert "Pattern prior" in markdown
    assert "lsmo-precedent" not in markdown
    assert "Paramagnetic scattering in LSMO" not in markdown


def test_copilot_router_recommends_dispersion_stiffness_scans_and_compares_precedents(tmp_path: Path) -> None:
    precedent_index = tmp_path / "precedents.jsonl"
    _write_lsm_precedent_index(precedent_index)

    result = copilot_ask(
        {
            "question": "For La.7Sr.3MnO3, recommend some scans to map the dispersion if we want to extract the stiffness.",
            "precedent_index": str(precedent_index),
            "output_dir": str(tmp_path / "out"),
            "top_k": 3,
        }
    )

    assert result["status"] == "ok"
    workflow_report = json.loads((tmp_path / "out" / "spinwave_measurement_planning" / "report.json").read_text(encoding="utf-8"))
    scan_plan = workflow_report["recommendations"]["dispersion_stiffness_scan_plan"]
    assert any("constant-Q energy scans" in item["recommendation"] for item in scan_plan)
    assert workflow_report["actual_measurement_comparison"]["status"] == "ok"
    assert workflow_report["actual_measurement_comparison"]["measured_patterns"]
    table = workflow_report["actual_measurement_comparison"]["recommended_vs_measured"]
    assert any(row["recommended_step"] == "small-q constant-Q energy scans" and row["measured_precedent_status"] == "observed" for row in table)
    assert any(example.get("energy_range") for row in table for example in row["measured_examples"])


def test_copilot_router_reports_missing_lsm_precedent_index(tmp_path: Path) -> None:
    result = copilot_ask(
        {
            "question": "I want to measure spinwaves in La.7Sr.3MnO3. What collimations should I use?",
            "precedent_index": str(tmp_path / "missing.jsonl"),
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "missing_input"
    assert result["route"]["intent"] == "precedent_experiment_planning"
    assert "precedent_index" in result["missing_inputs"][0]


def test_copilot_router_routes_paper_experiment_linkage(monkeypatch, tmp_path: Path) -> None:
    def fake_link(arguments: dict) -> dict:
        assert arguments["corpus_catalog"] == "corpus.sqlite"
        assert arguments["data_catalog"] == "data.sqlite"
        assert arguments["paper_id"] == "paper-1"
        return {
            "workflow_name": "link-paper-to-experiments",
            "status": "ok",
            "candidate_links": [{"experiment_id": "exp-1", "probability_score": 0.6}],
            "artifacts": {"json": str(tmp_path / "links.json"), "markdown": str(tmp_path / "links.md")},
        }

    monkeypatch.setattr("tas_copilot_core.workflows.copilot_router.link_paper_to_experiments", fake_link)
    result = copilot_ask(
        {
            "question": "Find candidate experiment links for this paper without making source-data claims.",
            "corpus_catalog": "corpus.sqlite",
            "data_catalog": "data.sqlite",
            "paper_id": "paper-1",
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "paper_experiment_linkage"
    assert result["routed_workflows"][0]["workflow_name"] == "link-paper-to-experiments"
    assert "not accepted source-data claims" in result["answer"]


def test_copilot_router_paper_experiment_linkage_requires_data_catalog(tmp_path: Path) -> None:
    result = copilot_ask(
        {
            "question": "Find candidate experiment links for this paper.",
            "corpus_catalog": "corpus.sqlite",
            "paper_id": "paper-1",
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "missing_input"
    assert result["route"]["intent"] == "paper_experiment_linkage"
    assert "data_catalog" in result["missing_inputs"]


def test_copilot_router_routes_overnight_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, detector=[100, 101, 650, 102, 100])

    result = copilot_ask(
        {
            "question": "Can you make an automatic summary of the scans performed overnight?",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "overnight_scan_summary"
    assert result["routed_workflows"][0]["workflow_name"] == "overnight-scan-summary"
    assert "morning scan-family summary" in result["answer"]


def test_copilot_router_passes_overnight_time_filter(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_bt7_scan(data_root / "old_scan.bt7", e_value=6.0, detector=[100, 101, 650, 102, 100], date="2018-01-01 00:00:00 EDT")
    _write_bt7_scan(data_root / "new_scan.bt7", e_value=6.0, detector=[100, 101, 650, 102, 100], date="2018-01-02 00:00:00 EDT")

    result = copilot_ask(
        {
            "question": "Can you make an automatic summary of the scans performed overnight?",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
            "since": "2018-01-01T12:00:00",
            "post_year": 0,
        }
    )

    assert result["status"] == "ok"
    workflow_report = json.loads((tmp_path / "out" / "overnight_scan_summary" / "overnight_scan_summary.json").read_text(encoding="utf-8"))
    assert workflow_report["time_filter"]["mode"] == "since"
    assert workflow_report["time_filter"]["kept_scan_count"] == 1
    assert workflow_report["time_filter"]["dropped_scan_count"] == 1


def test_copilot_router_routes_overnight_report_draft(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_bt7_scan(data_root / "inelastic_001.bt7", e_value=6.0, detector=[100, 101, 650, 102, 100])

    result = copilot_ask(
        {
            "question": "Please draft the morning report for scans performed overnight.",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "overnight_scan_summary"
    assert result["routed_workflows"][0]["workflow_name"] == "draft-overnight-report"
    assert "morning report draft" in result["answer"]


def test_copilot_router_ranks_next_diagnostics_after_overnight_data(tmp_path: Path) -> None:
    data_root = tmp_path / "overnight"
    data_root.mkdir()
    _write_energy_scan(data_root / "energy_scan_a.bt7")
    _write_energy_scan(data_root / "energy_scan_b.bt7")

    result = copilot_ask(
        {
            "question": "What should I do next after the overnight data?",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
            "post_year": 0,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "overnight_scan_summary"
    assert "ranked the next diagnostic actions" in result["answer"]
    workflow_report = json.loads((tmp_path / "out" / "overnight_scan_summary" / "overnight_scan_summary.json").read_text(encoding="utf-8"))
    diagnostic_actions = [item for item in workflow_report["next_actions"] if item["action_type"] != "human_review"]
    assert diagnostic_actions
    assert all(0.0 <= item["probability_of_value"] <= 1.0 for item in diagnostic_actions)
    assert any(item["action_type"] == "inelastic_cut_overlay" for item in diagnostic_actions)
    assert all("probability_basis" in item for item in diagnostic_actions)
    markdown = (tmp_path / "out" / "copilot_response.md").read_text(encoding="utf-8")
    assert "Safety Boundary" in markdown


def test_copilot_router_routes_order_parameter_triage(tmp_path: Path) -> None:
    data_root = tmp_path / "op"
    data_root.mkdir()
    for index, temp in enumerate([20, 80, 140, 200], start=1):
        _write_temperature_scan(data_root / f"order_parameter_{index}.bt7", temp=temp)

    result = copilot_ask(
        {
            "question": "Can you check for an order parameter or structural transition in these temperature-dependent peaks?",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
            "max_files": 20,
            "post_year": 0,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "order_parameter_triage"
    assert result["routed_workflows"][0]["workflow_name"] == "order-parameter-triage"
    assert "not a final fit" in result["answer"]


def test_copilot_router_routes_inelastic_series_report(tmp_path: Path) -> None:
    data_root = tmp_path / "inelastic"
    data_root.mkdir()
    _write_energy_scan(data_root / "energy_scan_a.bt7")
    _write_energy_scan(data_root / "energy_scan_b.bt7")

    result = copilot_ask(
        {
            "question": "Please make an inelastic report for these energy scans.",
            "data_root": str(data_root),
            "output_dir": str(tmp_path / "out"),
            "max_files": 20,
            "post_year": 0,
        }
    )

    assert result["status"] == "ok"
    assert result["route"]["intent"] == "inelastic_series_report"
    assert result["routed_workflows"][0]["workflow_name"] == "inelastic-series-report"
    assert "not a spin-wave fit" in result["answer"]


def test_copilot_router_blocks_live_control_requests(tmp_path: Path) -> None:
    result = copilot_ask({"question": "Start scan now and move motor A3.", "output_dir": str(tmp_path / "out")})

    assert result["status"] == "blocked"
    assert result["route"]["intent"] == "blocked_live_control"
    assert result["routed_workflows"] == []
    assert "cannot execute" in result["answer"].lower()


def test_copilot_ask_cli_writes_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    code = main(
        [
            "workflow",
            "copilot-ask",
            "--question",
            "What scans should I try for alignment?",
            "--fixture",
            PROPOSAL_FIXTURE,
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    payload = json.loads((output_dir / "copilot_response.json").read_text(encoding="utf-8"))
    assert payload["route"]["intent"] == "scan_plan_suggestions"
    assert (output_dir / "copilot_response.md").exists()


def test_copilot_ask_cli_lists_intents_without_output_dir(capsys) -> None:
    assert main(["workflow", "copilot-ask", "--list-intents"]) == 0
    output = capsys.readouterr().out
    assert "Bounded TASCopilot situations" in output
    assert "experiment_review" in output
    assert "paper_brief" in output
    assert "figure_brief" in output
    assert "corpus_search" in output
    assert "Could this narrow feature be a spurion?" in output


def test_copilot_contracts_round_trip() -> None:
    request = CopilotAskRequest(question="Build a dashboard", output_dir="local_runs/copilot")
    response = CopilotAskResponse(
        status="missing_input",
        question=request.question,
        route={"intent": "experiment_dashboard", "required_inputs": ["data_root"]},
        answer="Need data_root.",
    )

    assert response.route.intent == "experiment_dashboard"
    assert response.status == "missing_input"


def _write_bt7_scan(path: Path, *, e_value: float, detector: list[int], details: str = "BT7 PG", date: str = "2018-01-01 00:00:00 EDT") -> None:
    rows = []
    for index, y in enumerate(detector):
        rows.append(f"{index:.3f} {e_value:.3f} 14.7 14.7 100000 {y} {index}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        " + date,
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ExptDetails " + details,
                "#ScanType   VECTOR",
                "#Scan        1 QX",
                f"#ScanDescr   Scan:SubID=12345:JType=VECTOR:Range=E={e_value} {e_value} s:Title=E scan",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#Columns QX E Ef Ei Monitor Detector Temp",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_lsm_precedent_index(path: Path) -> None:
    record = {
        "experiment_id": "lsmo-precedent",
        "title": "Paramagnetic scattering in LSMO",
        "start_date": "2012-08-10",
        "end_date": "2012-08-30",
        "datafile_count": 12,
        "scan_count": 12,
        "path_clues": {"top_filename_prefixes": [{"value": "lsmo", "count": 12}]},
        "hydrated_clue_summary": {
            "top_values": {
                "fixed_e": [{"value": "Ef 14.7", "count": 12}],
                "lattice": [{"value": "3.8755 3.8755 3.8755 90 90 90", "count": 12}],
                "scan_variable": [{"value": "E", "count": 8}, {"value": "Qy", "count": 4}],
                "temp_mean": [{"value": "100", "count": 3}, {"value": "300", "count": 3}, {"value": "345", "count": 3}],
                "scan_description": [
                    {
                        "value": "Scan:SubID=17691:JType=VECTOR:Fixed=1:FixedE=14.7:Range=Q=1.4~0~0 1.4~0.3~0:Range=E=3.0 3.0:Comment=open-50 radial PG filter:Title=Transverse",
                        "count": 12,
                    },
                    {
                        "value": "Scan:SubID=17691:JType=VECTOR:Fixed=1:FixedE=14.7:Filename=LSMO_Q0p15:Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0:Range=E=2.0 17.0:Title=LSMO_Q0p15",
                        "count": 4,
                    }
                ],
            },
            "reader_subgroups": [{"detector_modes": [{"value": "3 SingDetHFoc", "count": 12}]}],
        },
    }
    false_positive = {
        "experiment_id": "layered-manganite",
        "title": "Spinwave excitation in LaSrMnO4",
        "datafile_count": 20,
        "scan_count": 20,
        "path_clues": {"top_filename_prefixes": [{"value": "lsmno", "count": 20}]},
        "hydrated_clue_summary": {"top_values": {"scan_description": [{"value": "Range=E=5 5", "count": 20}]}},
    }
    weak_unknown = {
        "experiment_id": "weak-unknown-lsmo",
        "title": "unknown experiment / lsmo / single_detector_inelastic_line_scan / unknown / 2005-10",
        "datafile_count": 30,
        "scan_count": 30,
        "path_clues": {"top_filename_prefixes": [{"value": "lsmo", "count": 30}]},
        "hydrated_clue_summary": {
            "top_values": {
                "fixed_e": [{"value": "Ef 14.7", "count": 30}],
                "lattice": [{"value": "3.8755 3.8755 3.8755 90 90 90", "count": 30}],
                "scan_variable": [{"value": "E", "count": 30}],
                "temp_mean": [{"value": "100", "count": 10}, {"value": "300", "count": 10}],
                "scan_description": [{"value": "Scan:JType=VECTOR:FixedE=14.7:Range=E=3.0 3.0", "count": 30}],
            }
        },
    }
    path.write_text(json.dumps(record) + "\n" + json.dumps(false_positive) + "\n" + json.dumps(weak_unknown) + "\n", encoding="utf-8")


def _write_temperature_scan(path: Path, *, temp: float) -> None:
    rows = []
    for index in range(51):
        a3 = 112.0 + index * 0.1
        center = 114.5
        detector = 25.0 + (320.0 - temp) * __import__("math").exp(-0.5 * ((a3 - center) / 0.18) ** 2)
        rows.append(f"0.0 0.0 0.0 0.0 49.75 {temp:.3f} 100000 {detector} {a3:.5f}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2018-01-01 00:00:00 EDT",
                "#InstrName   BT7",
                "#ExptID      12345",
                "#ScanType   ANGLE",
                "#Scan        1 A3",
                "#FixedE",
                "#AnalyzerDetectorMode 0 SingleDet",
                "#ScanDescr   Scan:SubID=12345:JType=ANGLE:Npts=51:Range=A3=112.0 117.0 s:Title=temperature dependent peak",
                "#Columns QX QY QZ E A4 Temp Monitor Detector A3",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_energy_scan(path: Path) -> None:
    rows = []
    for index, energy in enumerate([-1.0, 0.0, 1.0, 2.0, 3.0]):
        detector = 700 if index == 2 else 100
        rows.append(f"1.0 0.0 0.0 {energy:.3f} 14.7 14.7 100000 {detector} {index}")
    path.write_text(
        "\n".join(
            [
                "#Filename    " + path.stem,
                "#Date        2018-01-01 00:00:00 EDT",
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
