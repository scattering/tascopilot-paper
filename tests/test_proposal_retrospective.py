from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from tas_copilot_core.cli import main
import tas_copilot_core.workflows.proposal_retrospective as proposal_module
from tas_copilot_core.workflows.proposal_retrospective import ingest_proposal_evidence, proposal_experiment_retrospective, proposal_planning_brief, suggest_experiment_scans


FIXTURE = "tests/fixtures/proposal_retrospective/ansto_dto_thin_film"


def test_ansto_proposal_planning_brief_flags_prebeam_risks_and_questions(tmp_path: Path) -> None:
    result = proposal_planning_brief({"fixture": FIXTURE, "output_dir": str(tmp_path / "planning")})

    assert result["status"] == "ok"
    by_id = {risk["risk_id"]: risk for risk in result["risk_predictions"]}
    assert by_id["si_substrate_background"]["planning_classification"] == "explicit_warning"
    assert by_id["lambda_half_higher_order"]["planning_classification"] == "explicit_warning"
    assert by_id["temperature_sidecar_metadata"]["planning_classification"] == "explicit_warning"
    assert by_id["film_substrate_alignment"]["planning_classification"] == "explicit_warning"
    assert by_id["weak_signal_background_resolution"]["planning_classification"] == "explicit_warning"
    assert by_id["cu_mount_powder_rings"]["planning_classification"] == "missing_input_prompt"
    assert by_id["cu_mount_powder_rings"]["predicted_from_proposal"] is False
    assert by_id["cu_mount_powder_rings"]["promptable_from_proposal"] is True
    assert any("mount material" in item for item in result["missing_input_questions"])
    assert any("temperature" in item.lower() for item in result["missing_input_questions"])
    assert any("lambda/2" in item.lower() for item in result["required_prebeam_checks"])
    assert result["not_approved_operations"]
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_proposal_planning_brief_cli_writes_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "planning_cli"
    code = main(["workflow", "proposal-planning-brief", "--fixture", FIXTURE, "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "proposal_planning_brief.json").exists()
    assert (output_dir / "proposal_planning_brief.md").exists()


def test_ansto_proposal_retrospective_classifies_predictable_and_promptable_risks(tmp_path: Path) -> None:
    result = proposal_experiment_retrospective({"fixture": FIXTURE, "output_dir": str(tmp_path / "retro")})

    assert result["status"] == "ok"
    by_id = {risk["risk_id"]: risk for risk in result["risk_predictions"]}
    assert by_id["si_substrate_background"]["retrospective_classification"] == "predicted_and_observed"
    assert by_id["lambda_half_higher_order"]["retrospective_classification"] == "predicted_and_observed"
    assert by_id["temperature_sidecar_metadata"]["retrospective_classification"] == "predicted_and_observed"
    assert by_id["film_substrate_alignment"]["retrospective_classification"] == "predicted_and_observed"
    assert by_id["weak_signal_background_resolution"]["retrospective_classification"] == "predicted_and_observed"
    assert by_id["cu_mount_powder_rings"]["retrospective_classification"] == "not_in_proposal_but_promptable"
    assert by_id["cu_mount_powder_rings"]["predicted_from_proposal"] is False
    assert by_id["cu_mount_powder_rings"]["promptable_from_proposal"] is True
    assert "mount material" in by_id["cu_mount_powder_rings"]["missing_prebeam_inputs"]
    assert result["missed_observed_count"] == 0
    assert "probabilities" not in result["safe_conclusion"].lower()
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_proposal_retrospective_cli_writes_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    code = main(["workflow", "proposal-retrospective", "--fixture", FIXTURE, "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "proposal_experiment_retrospective.json").exists()
    assert (output_dir / "proposal_experiment_retrospective.md").exists()


def test_suggest_experiment_scans_fixture_recommends_staged_alignment_and_science_sequence(tmp_path: Path) -> None:
    result = suggest_experiment_scans({"fixture": FIXTURE, "output_dir": str(tmp_path / "scan_plan")})

    assert result["status"] == "ok"
    by_id = {item["suggestion_id"]: item for item in result["suggestions"]}
    assert "substrate_si111_rocking" in by_id
    assert "film_dy222_search" in by_id
    assert "science_003_coarse_longitudinal" in by_id
    assert "science_003_transverse" in by_id
    assert "temperature_series_minimal" in by_id
    assert "mesh_after_1d_checks" in by_id
    assert by_id["science_003_coarse_longitudinal"]["scan_direction"] == "longitudinal"
    assert by_id["science_003_coarse_longitudinal"]["suggested_range"]["L_min"] == 2.5
    assert by_id["science_003_coarse_longitudinal"]["suggested_range"]["L_max"] == 3.5
    assert by_id["science_003_coarse_longitudinal"]["step"] == 0.05
    assert by_id["science_003_transverse"]["scan_direction"] == "transverse"
    assert "science_003_coarse_longitudinal" in by_id["science_003_transverse"]["required_before_run"]
    assert any("1.2 K" in item for item in by_id["temperature_series_minimal"]["temperature_targets"])
    assert result["not_approved_operations"]
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_suggest_experiment_scans_reports_dashboard_generalization_gap(tmp_path: Path) -> None:
    result = suggest_experiment_scans({"fixture": FIXTURE, "output_dir": str(tmp_path / "scan_plan")})

    recommendations = "\n".join(result["dashboard_recommendations"]).lower()
    assert "no generalized experiment dashboard implementation" in recommendations
    assert "plotly" in recommendations
    assert "matplotlib" in recommendations


def test_suggest_experiment_scans_cli_writes_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "scan_plan_cli"
    code = main(["workflow", "suggest-experiment-scans", "--fixture", FIXTURE, "--output-dir", str(output_dir)])

    assert code == 0
    assert (output_dir / "experiment_scan_suggestions.json").exists()
    assert (output_dir / "experiment_scan_suggestions.md").exists()


def test_proposal_retrospective_missing_fixture_fails_closed(tmp_path: Path) -> None:
    result = proposal_experiment_retrospective({"fixture": str(tmp_path / "missing"), "output_dir": str(tmp_path / "retro")})

    assert result["status"] == "missing_input"
    assert "fixture_or_evidence_dir" in result["missing_fields"]


def test_local_private_evidence_ingest_feeds_planning_and_retrospective(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.md"
    actual = tmp_path / "actual_notes.md"
    proposal.write_text(
        "\n".join(
            [
                "Synthetic local proposal evidence: thin film on Si (111) substrate.",
                "The experiment uses reflection geometry, dilution refrigerator base temperature near 60 mK, graphite filters, and fixed final energy.",
                "Weak features motivate low background and long counting.",
            ]
        ),
        encoding="utf-8",
    )
    actual.write_text(
        "\n".join(
            [
                "Synthetic local actual evidence: notes mention Si 111 alignment, lambda/2 checks, and temperature was not stored.",
                "A Cu plate and sample plate created Cu reflection concerns.",
                "The run needed working frame, tilt, and background contamination checks.",
            ]
        ),
        encoding="utf-8",
    )

    ingest = ingest_proposal_evidence(
        {
            "proposal_documents": [str(proposal)],
            "actual_documents": [str(actual)],
            "output_dir": str(tmp_path / "ingest"),
        }
    )

    evidence_dir = Path(ingest["evidence_dir"])
    assert ingest["status"] == "ok"
    assert (evidence_dir / "proposal_chunks.jsonl").exists()
    assert (evidence_dir / "actual_evidence.jsonl").exists()
    assert "source_hash" in (evidence_dir / "proposal_chunks.jsonl").read_text(encoding="utf-8")
    assert "proposal_document_text" in (evidence_dir / "proposal_chunks.jsonl").read_text(encoding="utf-8")

    planning = proposal_planning_brief({"evidence_dir": str(evidence_dir), "output_dir": str(tmp_path / "planning")})
    by_id = {risk["risk_id"]: risk for risk in planning["risk_predictions"]}
    assert planning["status"] == "ok"
    assert by_id["si_substrate_background"]["planning_classification"] == "explicit_warning"
    assert by_id["cu_mount_powder_rings"]["planning_classification"] == "missing_input_prompt"

    retrospective = proposal_experiment_retrospective({"evidence_dir": str(evidence_dir), "output_dir": str(tmp_path / "retro")})
    by_id = {risk["risk_id"]: risk for risk in retrospective["risk_predictions"]}
    assert retrospective["status"] == "ok"
    assert by_id["cu_mount_powder_rings"]["retrospective_classification"] == "not_in_proposal_but_promptable"
    assert retrospective["missed_observed_count"] == 0


def test_proposal_evidence_ingest_classifies_codex_ocr_and_analysis_scripts(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.txt"
    codex_ocr = tmp_path / "codex_multimodal_evidence_summary.md"
    script = tmp_path / "diffraction_plot.py"
    log = tmp_path / "LogFile.txt"
    proposal.write_text("Synthetic proposal: thin film reflection geometry dilution 60 mK.", encoding="utf-8")
    codex_ocr.write_text(
        "\n".join(
            [
                "# Codex Multimodal Evidence Summary",
                "Local OCR notes mention Si 111 alignment, lambda/2 checks, Cu reflection overlays, and temperature was not stored.",
            ]
        ),
        encoding="utf-8",
    )
    script.write_text("# Synthetic analysis script evidence. Compute Cu reflection overlays; do not execute during ingest.\n", encoding="utf-8")
    log.write_text("Synthetic instrument log: motor error stopped scan; Si reflections recorded.\n", encoding="utf-8")

    result = ingest_proposal_evidence(
        {
            "proposal_documents": [str(proposal)],
            "actual_documents": [str(codex_ocr), str(script), str(log)],
            "output_dir": str(tmp_path / "ingest"),
        }
    )

    evidence_dir = Path(result["evidence_dir"])
    actual_records = [line for line in (evidence_dir / "actual_evidence.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert result["status"] == "ok"
    assert len(actual_records) == 3
    actual_text = "\n".join(actual_records)
    assert "codex_multimodal_ocr_summary" in actual_text
    assert "analysis_script" in actual_text
    assert "instrument_log" in actual_text
    assert "adapter_note" in actual_text
    assert all(diagnostic["diagnostic_id"] for diagnostic in result["document_diagnostics"])
    assert all(diagnostic["evidence_role"] in {"proposal", "actual"} for diagnostic in result["document_diagnostics"])


def test_proposal_evidence_ingest_cli_and_evidence_dir_cli(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.txt"
    actual = tmp_path / "actual.txt"
    proposal.write_text("Synthetic proposal: Si substrate thin film reflection geometry dilution 60 mK graphite filters.", encoding="utf-8")
    actual.write_text("Synthetic actual: Si reflections lambda/2 Cu plate temperature was not stored.", encoding="utf-8")
    ingest_dir = tmp_path / "cli_ingest"

    code = main(
        [
            "workflow",
            "proposal-evidence-ingest",
            "--proposal-document",
            str(proposal),
            "--actual-document",
            str(actual),
            "--output-dir",
            str(ingest_dir),
        ]
    )
    assert code == 0
    evidence_dir = ingest_dir / "evidence"
    assert (ingest_dir / "proposal_evidence_ingest.json").exists()
    assert (evidence_dir / "proposal_chunks.jsonl").exists()

    planning_dir = tmp_path / "cli_planning"
    code = main(["workflow", "proposal-planning-brief", "--evidence-dir", str(evidence_dir), "--output-dir", str(planning_dir)])
    assert code == 0
    assert (planning_dir / "proposal_planning_brief.json").exists()

    retro_dir = tmp_path / "cli_retro"
    code = main(["workflow", "proposal-retrospective", "--evidence-dir", str(evidence_dir), "--output-dir", str(retro_dir)])
    assert code == 0
    assert (retro_dir / "proposal_experiment_retrospective.json").exists()


def test_proposal_evidence_ingest_missing_documents_fails_closed(tmp_path: Path) -> None:
    result = ingest_proposal_evidence(
        {
            "proposal_documents": [str(tmp_path / "missing_proposal.docx")],
            "actual_documents": [],
            "output_dir": str(tmp_path / "ingest"),
        }
    )

    assert result["status"] == "missing_input"
    assert result["proposal_chunk_count"] == 0
    assert any("missing_proposal.docx" in warning for warning in result["warnings"])


def test_proposal_evidence_ingest_reports_zero_text_pdf_diagnostics(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "image_only_notes.pdf"
    pdf.write_bytes(b"%PDF synthetic placeholder")

    monkeypatch.setattr(proposal_module.shutil, "which", lambda name: "/usr/bin/pdftotext" if name == "pdftotext" else None)

    def fake_run(*args, **kwargs):
        return CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(proposal_module.subprocess, "run", fake_run)
    result = ingest_proposal_evidence(
        {
            "proposal_documents": [],
            "actual_documents": [str(pdf)],
            "output_dir": str(tmp_path / "ingest"),
        }
    )

    assert result["status"] == "missing_input"
    assert result["actual_evidence_count"] == 0
    diagnostic = result["document_diagnostics"][0]
    assert diagnostic["status"] == "no_text"
    assert diagnostic["extraction_method"] == "pdftotext_first_pages"
    assert any("image-only or scanned PDF" in warning for warning in diagnostic["warnings"])
