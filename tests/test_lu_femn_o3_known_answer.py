from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_contracts import Figure, FigurePanel, FigurePanelSourceCandidate, FigurePanelSourceLink, PSDCalibrationRecord, SourceSupportRecord
from tas_copilot_core.corpus.fixtures import compile_figure_query, corpus_fixture_report, load_corpus_fixtures, validate_source_support
from tas_copilot_core.workflows.lu_femn_o3_regression import link_figure_panel_to_source, run_lu_femn_o3_regression, source_triage_doctor


def test_source_triage_doctor_reports_capabilities() -> None:
    result = source_triage_doctor({})
    assert result["status"] == "ok"
    assert result["checks"]["q_temperature_requires_reviewed_calibration"]
    assert result["checks"]["live_control_disabled_by_policy"]


def test_paper_figure_panel_contracts_validate() -> None:
    figure = Figure(figure_id="fig", paper_id="paper")
    panel = FigurePanel(panel_id="i", figure_id="fig", measurement_kind="elastic", detector_mode="PSD")
    candidate = FigurePanelSourceCandidate(panel_id="i", candidate_id="candidate", candidate_type="scan_group")
    link = FigurePanelSourceLink(figure_id="fig", panel_id="i", source_object_id="candidate", source_object_type="scan_group")
    support = SourceSupportRecord(support_id="support", claim_id="claim", source_object_id="chunk", source_object_type="source_chunk")
    calibration = PSDCalibrationRecord(calibration_id="cal", instrument="BT-7", status="proposed")
    assert figure.figure_id == "fig"
    assert panel.measurement_kind == "elastic"
    assert candidate.status == "proposed"
    assert link.human_review_required
    assert support.status == "proposed"
    assert calibration.status == "proposed"


def test_corpus_fixture_loads_paper_figure_panel_records(tmp_path: Path) -> None:
    result = load_corpus_fixtures("tests/fixtures/lu_femn_o3_paper_figure", tmp_path)
    assert result["status"] == "ok"
    report = corpus_fixture_report(tmp_path)
    assert report["papers"] == 1
    assert report["figures"] == 1
    assert report["figure_panels"] == 3
    assert report["source_support_records"] == 1
    assert "separate" in report["separation_policy"]


def test_compile_figure_query_preserves_panel_constraints_and_mismatch(tmp_path: Path) -> None:
    corpus_catalog = tmp_path / "corpus"
    load_corpus_fixtures("tests/fixtures/lu_femn_o3_paper_figure", corpus_catalog)

    result = compile_figure_query(
        corpus_catalog,
        paper_id="paper-lu-femn-o3-synthetic",
        figure_id="fig-lu-femn-o3-1",
        panel_id="k",
        user_sample="LuFe0.6Mn0.4O3",
        output=tmp_path / "panel_k_query.json",
    )

    assert result["status"] == "ok"
    query = result["query"]
    assert query["target_panel"] == "k"
    assert query["target_composition"]["x_mn"] == 0.5
    assert query["measurement_kind"] == "elastic"
    assert query["required_energy_transfer"] == 0
    assert query["measurement_mode"] == "PSD"
    assert query["q_range"]["min_value"] == 1.1
    assert query["temperature_range"]["max_value"] == 200
    assert query["annotated_q_values"] == [1.24]
    assert result["warnings"]
    assert "panel j" in result["warnings"][0]
    assert (tmp_path / "panel_k_query.json").exists()


def test_source_support_validator_keeps_candidate_links_non_definitive(tmp_path: Path) -> None:
    corpus_catalog = tmp_path / "corpus"
    load_corpus_fixtures("tests/fixtures/lu_femn_o3_paper_figure", corpus_catalog)

    result = validate_source_support(corpus_catalog)

    assert result["status"] == "ok"
    claims = {item["claim"] for item in result["supported_claims"]}
    assert "panel_i_x_mn" in claims
    assert "panel_j_x_mn" in claims
    assert "panel_k_x_mn" in claims
    assert "user_sample_lufe06mn04_x_mn" in claims
    assert "candidate_link_remains_non_definitive" in claims
    assert not result["unsupported_claims"]


def test_lu_femn_o3_known_answer_regression(tmp_path: Path) -> None:
    result = run_lu_femn_o3_regression({"suite": "eval/figure_source_lu_femn_o3_known_answer.yaml", "output_dir": str(tmp_path)})
    assert result["status"] == "pass"
    check_names = {item["name"] for item in result["checks"]}
    assert "sample_mismatches_panel_k" in check_names
    assert "inelastic_false_positive_rejected" in check_names
    assert "q_temperature_blocked" in check_names
    assert (tmp_path / "lu_femn_o3_regression_report.md").exists()
    assert (tmp_path / "expected_vs_actual.json").exists()


def test_link_figure_panel_to_source_writes_proposed_links(tmp_path: Path) -> None:
    corpus_catalog = tmp_path / "corpus"
    load_corpus_fixtures("tests/fixtures/lu_femn_o3_paper_figure", corpus_catalog)
    result = link_figure_panel_to_source(
        {
            "corpus_catalog": str(corpus_catalog),
            "data_catalog": "eval/figure_source_lu_femn_o3_triage.yaml",
            "paper_id": "paper-lu-femn-o3-synthetic",
            "figure_id": "fig-lu-femn-o3-1",
            "panel_id": "i",
            "output_dir": str(tmp_path / "linkage"),
        }
    )
    assert result["status"] == "ok"
    assert result["candidate_links"]
    assert result["compiled_query"]["target_composition"]["x_mn"] == 0.25
    assert result["source_support_summary"]["status"] == "ok"
    assert all(item["status"] == "proposed" for item in result["candidate_links"])
    assert json.loads((tmp_path / "linkage" / "figure_panel_source_candidates.json").read_text(encoding="utf-8"))
    assert json.loads((tmp_path / "linkage" / "compiled_panel_query.json").read_text(encoding="utf-8"))
