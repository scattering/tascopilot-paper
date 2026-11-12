from __future__ import annotations

import json
from pathlib import Path

import yaml

from tas_copilot_contracts import DataFile, Experiment, FigureSourceQuery, Scan
from tas_copilot_core.composite_tools import run_tool as run_composite_tool
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.material_identity import compare_material_identity, material_identity_backend_status, material_identity_record, parse_formula, parse_material_text
from tas_copilot_core.source_triage import (
    check_psd_channel_artifact,
    classify_source_family,
    compare_psd_detector_cuts,
    discover_psd_calibration_candidates,
    group_psd_sources,
    hydrate_psd_records,
    plot_psd_detector_cuts,
    q_temperature_reconstruction_blocked,
    reconstruct_channel_temperature,
    select_detector_cut_records,
    triage_records,
)
from tas_copilot_core.workflows.figure_source_triage import figure_source_triage
from tas_copilot_core.workflows.psd_mask_sensitivity import psd_mask_sensitivity
from tas_copilot_core.workflows.review_figure_source_candidate import review_figure_source_candidate


def fixture_records() -> list[dict]:
    with Path("eval/figure_source_lu_femn_o3_triage.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["records"]


def test_lu_femn_o3_filename_composition_interpreter() -> None:
    assert parse_material_text("LuFe75Mn25O3_Correlations102248.bt7").x_variables["x_Mn"] == 0.25
    assert parse_material_text("LuFe60Mn40O3_Correlations102400.bt7").x_variables["x_Mn"] == 0.4
    assert parse_material_text("LuFe50Mn50O3_Correlations102500.bt7").x_variables["x_Mn"] == 0.5
    assert parse_material_text("LuFe0.6Mn0.4O3").x_variables["x_Mn"] == 0.4


def test_ambiguous_lu_femn_shorthands_are_not_guessed() -> None:
    for text in ["LuFeMnO3_Tdep_3meV102311.bt7", "LuMnFeO3_29pt5_TdepWarm102246.bt7"]:
        result = parse_material_text(text)
        assert result.x_variables == {}
        assert result.ambiguity_reason
        assert result.confidence < 0.5


def test_material_identity_compare() -> None:
    result = compare_material_identity("LuFe0.6Mn0.4O3", "LuFe60Mn40O3")
    assert result["status"] == "match"
    ambiguous = compare_material_identity("LuFeMnO3", "LuFe60Mn40O3")
    assert ambiguous["status"] == "ambiguous"


def test_standard_formula_parser_prefers_optional_backends_and_preserves_composition() -> None:
    srtio3 = material_identity_record("SrTiO3", source_field="user_formula")
    assert srtio3.parser_backend in {"pymatgen", "ase", "tas_simple_decimal_fallback"}
    assert srtio3.element_amounts == {"Sr": 1.0, "Ti": 1.0, "O": 3.0}
    assert srtio3.confidence > 0.5

    lsmo = parse_formula("La0.7Sr0.3MnO3")
    assert lsmo.parser_backend in {"pymatgen", "tas_simple_decimal_fallback"}
    assert lsmo.element_amounts["La"] == 0.7
    assert lsmo.element_amounts["Sr"] == 0.3
    assert lsmo.element_amounts["Mn"] == 1.0
    assert lsmo.element_amounts["O"] == 3.0


def test_generic_material_identity_compare_uses_composition_when_no_x_variable() -> None:
    result = compare_material_identity("SrTiO3", "TiO3Sr")
    assert result["status"] == "match"
    mismatch = compare_material_identity("SrTiO3", "Al2O3")
    assert mismatch["status"] == "mismatch"


def test_material_identity_backend_status_reports_optional_parsers() -> None:
    status = material_identity_backend_status()
    assert status["preferred_backend"] == "pymatgen"
    assert status["fallback_backend"] == "ase"
    assert status["filename_interpreter"] == "available"


def test_psd_source_family_classifier() -> None:
    records = {record["datafile_id"]: record for record in fixture_records()}
    assert classify_source_family(records["synthetic-lufe75mn25-angle-001"]).source_family == "multi_file_psd_angle_temperature_series"
    assert classify_source_family(records["synthetic-lumnfeo3-tdep-warm"]).source_family == "single_file_psd_temperature_map"
    assert classify_source_family(records["synthetic-lufemno3-3mev-line"]).source_family == "single_detector_inelastic_line_scan"


def test_psd_grouping_and_reconstruction(tmp_path: Path) -> None:
    records = fixture_records()
    groups = group_psd_sources(records)
    group_types = {group.group_type for group in groups}
    assert "multi_file_psd_angle_temperature_series" in group_types
    assert "single_file_psd_temperature_map" in group_types
    single = next(group for group in groups if group.group_type == "single_file_psd_temperature_map")
    result = reconstruct_channel_temperature(single, records, tmp_path)
    assert result.status == "success"
    assert result.reconstruction_level == "channel_temperature"
    assert result.shape == [3, 4]
    assert len(result.output_artifact_paths) == 2
    blocked = q_temperature_reconstruction_blocked(single.group_id)
    assert blocked.status == "unsupported"
    assert "reviewed_q_conversion_adapter" in blocked.missing_fields


def test_psd_calibration_discovery_is_candidate_only(tmp_path: Path) -> None:
    data_root = tmp_path / "bt7"
    data_root.mkdir()
    calibration = data_root / "PSDcalib61938.bt7"
    calibration.write_text(
        "\n".join(
            [
                "#Filename    PSDcalib61938",
                "#Date        2008-10-07 13:35:54 EDT",
                "#InstrName   BT7",
                "#ExptID      13451",
                "#ScanType   MOTOR",
                "#DetectorEfficiencies Monitor=1.0 PSDC00=1.0 PSDC47=1.0",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#AnalyzerPSDGroup PSDC00 PSDC01 PSDC47",
                "#ScanDescr   Scan:SubID=13451:JType=MOTOR:Range=A4=79.5 79.5 s:Title=van",
                "#Columns A4 PSDC00 PSDC47",
            ]
        ),
        encoding="utf-8",
    )
    (data_root / "ordinary.bt7").write_text("#Filename ordinary\n#ScanType VECTOR\n", encoding="utf-8")

    result = discover_psd_calibration_candidates(data_root, output_dir=tmp_path / "report")

    assert result["status"] == "success"
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["filename"] == "PSDcalib61938.bt7"
    assert result["candidates"][0]["q_temperature_unlock"] is False
    assert result["candidates"][0]["status"] == "candidate_unreviewed"
    assert "reviewed calibration" in result["safe_conclusion"]
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_psd_detector_cut_diagnostic(tmp_path: Path) -> None:
    records = fixture_records()
    result = plot_psd_detector_cuts(
        records,
        tmp_path,
        scan_id="synthetic-lumnfeo3-tdep-warm",
        temperature_min=100,
        temperature_max=150,
    )
    assert result["status"] == "success"
    assert result["cut_count"] == 3
    assert Path(result["artifacts"]["csv"]).exists()
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()
    assert "reviewed PSD channel-to-A4/sample two-theta calibration" in " ".join(result["warnings"])


def test_psd_detector_cut_comparison(tmp_path: Path) -> None:
    result = compare_psd_detector_cuts(
        fixture_records(),
        tmp_path,
        left_scan_id="synthetic-lumnfeo3-tdep-warm",
        right_scan_id="synthetic-lumnfeo3-tdep-cool",
        temperature_min=100,
        temperature_max=150,
        temperature_tolerance=2.0,
    )
    assert result["status"] == "success"
    assert result["matched_temperature_count"] == 2
    assert result["channel_count"] == 4
    assert Path(result["artifacts"]["csv"]).exists()
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()
    assert "reviewed PSD channel-to-A4/sample two-theta calibration" in " ".join(result["warnings"])


def test_psd_channel_artifact_diagnostic(tmp_path: Path) -> None:
    result = check_psd_channel_artifact(
        fixture_records(),
        tmp_path,
        scan_id="synthetic-lumnfeo3-tdep-warm",
        channels_of_interest=[2],
        temperature_min=100,
        temperature_max=150,
    )
    assert result["status"] == "success"
    assert result["summary_by_channel"][0]["channel"] == 2
    assert result["summary_by_channel"][0]["peak_fraction"] > 0
    assert result["diagnostic_observations"]
    assert Path(result["artifacts"]["csv"]).exists()
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_psd_detector_cut_hydration_uses_reader() -> None:
    calls = []

    class ReaderResult:
        status = "success"
        audit_id = "audit-reader"
        warnings = []
        errors = []
        output = {
            "psd_map": {
                "psd_channel_axis": [0, 1],
                "temperature_axis": [100, 125],
                "intensity": [[1, 2], [3, 4]],
            },
            "metadata": {"scan_type": "ENV"},
        }

    def reader(arguments: dict):
        calls.append(arguments)
        return ReaderResult()

    records, hydration = hydrate_psd_records([{"datafile_id": "needs-hydration", "path": "/tmp/fake.bt7"}], reader)
    assert calls == [{"path": "/tmp/fake.bt7"}]
    assert hydration[0]["status"] == "success"
    assert records[0]["temperature_axis"] == [100, 125]
    assert records[0]["intensity"] == [[1, 2], [3, 4]]


def test_detector_cut_selection_limits_hydration_scope() -> None:
    selected = select_detector_cut_records(fixture_records(), scan_id="synthetic-lumnfeo3-tdep-warm", group_id=None)
    assert [record["datafile_id"] for record in selected] == ["synthetic-lumnfeo3-tdep-warm"]


def test_figure_source_triage_workflow(tmp_path: Path) -> None:
    result = figure_source_triage(
        {
            "catalog": "eval/figure_source_lu_femn_o3_triage.yaml",
            "query_file": "examples/figure_queries/lu_femn_o3_elastic_psd_triage.yaml",
            "output_dir": str(tmp_path),
        }
    )
    assert result["status"] == "success"
    assert result["panel_sample_warnings"]
    state = result["source_triage_state"]
    assert state["candidate_groups"]
    summary = result["triage_summary"]
    assert summary["ranked_groups"]
    assert "PSD" in summary["recommended_source_branch"]
    assert "calibration" in summary["calibration_blocker"].lower()
    rejected_ids = {item["candidate_id"] for item in state["rejected_candidates"]}
    assert "synthetic-lufemno3-3mev-line" in rejected_ids
    assert (tmp_path / "figure_source_report.md").exists()
    assert (tmp_path / "source_triage_state.json").exists()
    assert (tmp_path / "figure_source_review_template.jsonl").exists()
    assert result["artifacts"]["figure_source_review_template_jsonl"] == str(tmp_path / "figure_source_review_template.jsonl")
    review_rows = [
        json.loads(line)
        for line in (tmp_path / "figure_source_review_template.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["candidate_id"] == "synthetic-lufemno3-3mev-line" and row["candidate_status"] == "rejected_by_workflow" for row in review_rows)
    persisted = json.loads((tmp_path / "source_triage_state.json").read_text(encoding="utf-8"))
    assert persisted["active_hypotheses"]


def test_source_triage_uses_material_identity_registry_tool() -> None:
    calls = []

    class RecordingRunner:
        def run(self, call):
            calls.append((call.tool_namespace, call.tool_name, call.arguments))
            return run_composite_tool(call)

    query = FigureSourceQuery(
        query_id="registry-material-test",
        target_panel="j",
        material_family="LuFe_{1-x}Mn_xO3",
        user_sample_formula="LuFe0.6Mn0.4O3",
        target_composition={"x_mn": 0.4},
        instrument="BT-7",
        measurement_mode="PSD",
        measurement_kind="elastic",
        required_energy_transfer=0.0,
    )
    result = triage_records(query, fixture_records()[:3], runner=RecordingRunner())

    assert result["status"] == "success"
    material_calls = [call for call in calls if call[0] == "tas.material_identity"]
    assert material_calls
    assert {call[1] for call in material_calls} == {"parse_filename_composition"}
    assert any(call[2]["source_field"] == "query.user_sample_formula" for call in material_calls)


def test_review_figure_source_candidate_runs_detector_channel_diagnostics(tmp_path: Path) -> None:
    catalog = SQLiteDataCatalog(tmp_path / "review.sqlite")
    experiment = Experiment(experiment_id="exp-review", instrument="BT-7")
    scan = Scan(
        scan_id="scan-review",
        experiment_id="exp-review",
        instrument="BT-7",
        scan_type="ENV",
        metadata={
            "analyzer_detector_mode": "4 PSDDiff",
            "scan_type": "ENV",
            "psd_map": {
                "psd_channel_axis": [0, 1, 2, 3],
                "temperature_axis": [100, 125, 150],
                "intensity": [[1, 2, 4, 2], [1, 3, 5, 2], [1, 2, 4, 3]],
            },
        },
    )
    raw_file = tmp_path / "LuMnFeO3_29pt5_TdepWarm102246.bt7"
    raw_file.write_text(
        "\n".join(
            [
                "#Filename    LuMnFeO3_29pt5_TdepWarm102246",
                "#Date        2013-06-01 12:52:46 EDT",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#Fixed       A3 A5 A6 ANACOLL ANADET AnalyzerRotation Ef PSDet",
                "#ScanDescr   Scan:SubID=18445:JType=ENV:Npts=3:Range=Temp=100 150 s",
                "#Columns A4 Temp PSDC00 PSDC01 PSDC02 PSDC03 PSDet AnalyzerRotation Ef",
                "29.5 100 1 2 4 2 0 90.0 13.7",
                "29.5 125 1 3 5 2 0 90.0 13.7",
                "29.5 150 1 2 4 3 0 90.0 13.7",
            ]
        ),
        encoding="utf-8",
    )
    datafile = DataFile(
        datafile_id="df-review",
        path=str(raw_file),
        scan_id="scan-review",
        experiment_id="exp-review",
        format="BT7",
        metadata={"metadata_mode": "hydrated", "hydration_status": "hydrated"},
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    report_path = tmp_path / "figure_source_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source_triage_state": {
                    "candidate_groups": [
                        {
                            "group_id": "psd-group-review",
                            "group_type": "single_file_psd_temperature_map",
                            "member_scan_ids": ["scan-review"],
                            "member_datafile_ids": ["df-review"],
                            "grouping_confidence": 0.9,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    calibration_dir = tmp_path / "psdcali"
    calibration_dir.mkdir()
    (calibration_dir / "PSD_June2013_LuFeMnO_AnaTrans50.dat").write_text("", encoding="utf-8")
    (calibration_dir / "PSD_May2013_LuFeMnO_AnaTrans200.dat").write_text("", encoding="utf-8")

    result = review_figure_source_candidate(
        {
            "catalog": str(catalog.path),
            "figure_report": str(report_path),
            "candidate_id": "psd-group-review",
            "temperature_min": 100,
            "temperature_max": 150,
            "channels": [2],
            "psd_calibration_dir": str(calibration_dir),
            "psd_calibration_path_only": True,
            "output_dir": str(tmp_path / "review"),
        }
    )

    assert result["status"] == "success"
    assert result["candidate_id"] == "psd-group-review"
    assert result["detector_cut_results"][0]["result"]["status"] == "success"
    assert result["channel_artifact_result"]["status"] == "success"
    assert result["q_temperature_gate"]["status"] == "unsupported"
    assert result["psd_calibration_matches"][0]["best_match"]["filename"] == "PSD_June2013_LuFeMnO_AnaTrans50.dat"
    assert result["psd_calibration_matches"][0]["best_match"]["safe_use"] is False
    assert "analyzer_detector_translation" in result["psd_calibration_matches"][0]["missing_fields"]
    assert result["audit_ids"]
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()
    markdown = Path(result["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "PSD channel-to-A4/sample two-theta" in markdown
    assert (tmp_path / "review" / "audit" / "audit.jsonl").exists()


def test_psd_mask_sensitivity_reports_probability_not_certainty(tmp_path: Path) -> None:
    data_file = tmp_path / "synthetic_psd.bt7"
    data_file.write_text(
        "\n".join(
            [
                "#Filename    synthetic_psd",
                "#Date        2013-06-01 12:00:00 EDT",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#ScanDescr   Scan:SubID=1:JType=ENV:Npts=3:Range=Temp=50 100 s",
                "#Columns A4 Temp Ef PSDC00 PSDC01 PSDC02 PSDC03",
                "29.5 50 13.7 100 100 100 100",
                "29.5 75 13.7 100 100 100 100",
                "29.5 100 13.7 100 100 100 100",
            ]
        ),
        encoding="utf-8",
    )
    calibration_dir = tmp_path / "psd"
    calibration_dir.mkdir()
    (calibration_dir / "PSD_June2013_Test_AnaTrans200.dat").write_text(
        "\n".join(
            [
                "#BT7 PSD calibration based on synthetic_psd.bt7",
                "#Created on: Mon, 03 Jun 2013 17:44:00",
                "#Channels to be masked: 4",
                "#Efficiency  A4  Ef",
                "1.0 -2.0 0",
                "1.0 -1.8 0",
                "1.0 -1.7 0",
                "0.1 -1.53471 0",
            ]
        ),
        encoding="utf-8",
    )

    result = psd_mask_sensitivity(
        {
            "data_files": [str(data_file)],
            "psd_calibration_dir": str(calibration_dir),
            "calibration_files": ["PSD_June2013_Test_AnaTrans200.dat"],
            "annotated_q": 1.2426,
            "output_dir": str(tmp_path / "mask"),
        }
    )

    assert result["status"] == "success"
    assert result["artifact_assessment"]["conclusion"] == "probable_artifact"
    assert 0.0 < result["artifact_assessment"]["artifact_probability"] < 1.0
    assert "not certainty" in result["safe_conclusion"].lower()
    declared = next(item for item in result["summaries"] if item["mask_label"] == "declared_calibration_mask")
    unmasked = next(item for item in result["summaries"] if item["mask_label"] == "none")
    assert unmasked["q_stripe_metric"] > 3.0
    assert declared["q_stripe_metric"] < 1.2
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_review_figure_source_candidate_includes_mask_sensitivity_probability(tmp_path: Path) -> None:
    catalog = SQLiteDataCatalog(tmp_path / "review_mask.sqlite")
    experiment = Experiment(experiment_id="exp-review-mask", instrument="BT-7")
    scan = Scan(
        scan_id="scan-review-mask",
        experiment_id="exp-review-mask",
        instrument="BT-7",
        scan_type="ENV",
        metadata={
            "analyzer_detector_mode": "4 PSDDiff",
            "scan_type": "ENV",
            "psd_map": {
                "psd_channel_axis": [0, 1, 2, 3],
                "temperature_axis": [50, 75, 100],
                "intensity": [[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]],
            },
        },
    )
    raw_file = tmp_path / "LuMnFeO3_29pt5_TdepWarm102246.bt7"
    raw_file.write_text(
        "\n".join(
            [
                "#Filename    LuMnFeO3_29pt5_TdepWarm102246",
                "#Date        2013-06-01 12:52:46 EDT",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#ScanDescr   Scan:SubID=18445:JType=ENV:Npts=3:Range=Temp=50 100 s",
                "#Columns A4 Temp Ef PSDC00 PSDC01 PSDC02 PSDC03",
                "29.5 50 13.7 100 100 100 100",
                "29.5 75 13.7 100 100 100 100",
                "29.5 100 13.7 100 100 100 100",
            ]
        ),
        encoding="utf-8",
    )
    datafile = DataFile(
        datafile_id="df-review-mask",
        path=str(raw_file),
        scan_id="scan-review-mask",
        experiment_id="exp-review-mask",
        format="BT7",
        metadata={"metadata_mode": "hydrated", "hydration_status": "hydrated"},
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    report_path = tmp_path / "figure_source_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source_triage_state": {
                    "candidate_groups": [
                        {
                            "group_id": "psd-group-review-mask",
                            "group_type": "single_file_psd_temperature_map",
                            "member_scan_ids": ["scan-review-mask"],
                            "member_datafile_ids": ["df-review-mask"],
                            "grouping_confidence": 0.9,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    calibration_dir = tmp_path / "psdcali"
    calibration_dir.mkdir()
    (calibration_dir / "PSD_June2013_Test_AnaTrans200.dat").write_text(
        "\n".join(
            [
                "#BT7 PSD calibration based on synthetic_psd.bt7",
                "#Created on: Mon, 03 Jun 2013 17:44:00",
                "#Channels to be masked: 4",
                "#Efficiency  A4  Ef",
                "1.0 -2.0 0",
                "1.0 -1.8 0",
                "1.0 -1.7 0",
                "0.1 -1.53471 0",
            ]
        ),
        encoding="utf-8",
    )

    result = review_figure_source_candidate(
        {
            "catalog": str(catalog.path),
            "figure_report": str(report_path),
            "candidate_id": "psd-group-review-mask",
            "temperature_min": 50,
            "temperature_max": 100,
            "psd_calibration_dir": str(calibration_dir),
            "psd_calibration_files": ["PSD_June2013_Test_AnaTrans200.dat"],
            "annotated_q": 1.2426,
            "output_dir": str(tmp_path / "review_mask"),
        }
    )

    mask = result["mask_sensitivity_result"]
    assert result["accepted_psd_calibrations"][0]["calibration_id"] == "bt7-psd-201305-lufemno-anatrans200-reviewed"
    assert mask["status"] == "success"
    assert mask["artifact_assessment"]["conclusion"] == "probable_artifact"
    assert mask["artifact_assessment"]["artifact_probability"] >= 0.75
    assert "evidence, not certainty" in result["safe_conclusion"]
    markdown = Path(result["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "PSD Mask Sensitivity" in markdown
    assert "Accepted PSD Calibration Policy" in markdown
    assert "probable_artifact" in markdown
