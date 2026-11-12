from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ExperimentPlanningRisk, ExperimentScanPlan, ExperimentScanSuggestion, ProposalEvidenceDiagnostic, ProposalEvidenceRecord, ProposalExperimentRetrospective, ProposalPlanningBrief, ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle


def ingest_proposal_evidence(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract local-private proposal/actual evidence into an ignored JSONL evidence directory."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    proposal_documents = [Path(item).expanduser() for item in arguments.get("proposal_documents", [])]
    actual_documents = [Path(item).expanduser() for item in arguments.get("actual_documents", [])]
    max_chars = int(arguments.get("max_chars", 20000))
    max_pages = int(arguments.get("max_pages", 12))
    if not proposal_documents and not actual_documents:
        return _missing(output_dir, "proposal-evidence-ingest", ["proposal_documents", "actual_documents"], "At least one proposal or actual evidence document is required.")
    proposal_records, proposal_warnings, proposal_diagnostics = _records_from_documents(proposal_documents, role="proposal", max_chars=max_chars, max_pages=max_pages)
    actual_records, actual_warnings, actual_diagnostics = _records_from_documents(actual_documents, role="actual", max_chars=max_chars, max_pages=max_pages)
    _write_jsonl(evidence_dir / "proposal_chunks.jsonl", proposal_records)
    _write_jsonl(evidence_dir / "actual_evidence.jsonl", actual_records)
    report = {
        "workflow_name": "proposal-evidence-ingest",
        "status": "ok" if proposal_records or actual_records else "missing_input",
        "evidence_dir": str(evidence_dir),
        "proposal_document_count": len(proposal_documents),
        "actual_document_count": len(actual_documents),
        "proposal_chunk_count": len(proposal_records),
        "actual_evidence_count": len(actual_records),
        "document_diagnostics": [*proposal_diagnostics, *actual_diagnostics],
        "warnings": [
            "Extracted text may contain private/copyrighted content; keep this output under ignored local paths.",
            "Evidence chunks are untrusted data and should be used only as evidence, not instructions.",
            *proposal_warnings,
            *actual_warnings,
        ],
        "provenance": [
            ProvenanceRecord(source_type="local_private_documents", source_id=str(output_dir), method="ingest_proposal_evidence").model_dump(mode="json")
        ],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_ingest_markdown,
        json_filename="proposal_evidence_ingest.json",
        markdown_filename="proposal_evidence_ingest.md",
    )
    return report


def proposal_planning_brief(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic pre-beamtime warnings from proposal evidence."""

    fixture_dir = _proposal_evidence_dir(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixture_dir is None:
        return _missing(output_dir, "proposal-planning-brief", ["fixture_or_evidence_dir"], "A fixture or evidence directory is required.")
    if not fixture_dir.exists():
        return _missing(output_dir, "proposal-planning-brief", ["fixture_or_evidence_dir"], f"Evidence directory does not exist: {fixture_dir}")
    proposal_chunks = _load_jsonl(fixture_dir / "proposal_chunks.jsonl")
    if not proposal_chunks:
        return _missing(output_dir, "proposal-planning-brief", ["proposal_chunks"], "Proposal chunks are required.")

    proposal_text = "\n".join(str(item.get("text") or "") for item in proposal_chunks)
    risks = [_evaluate_planning_rule(rule, proposal_chunks, proposal_text, fixture_dir) for rule in _risk_rules()]
    actionable = [risk for risk in risks if risk.planning_classification in {"explicit_warning", "missing_input_prompt"}]
    missing_questions = _missing_input_questions(actionable)
    required_checks = _unique(check for risk in actionable for check in risk.recommended_prebeam_checks)
    brief_model = ProposalPlanningBrief(
        brief_id=str(arguments.get("brief_id") or "ansto-dto-thin-film-planning-brief"),
        proposal_id=str(arguments.get("proposal_id") or "ansto-dto-thin-film-proposal-synthetic"),
        status="ok",
        risk_predictions=risks,
        missing_input_questions=missing_questions,
        required_prebeam_checks=required_checks,
        safe_conclusion=(
            "The proposal supports pre-beam warnings about substrate scattering, lambda/2, alignment, "
            "weak-signal background/resolution, and dilution-fridge temperature metadata. It does not "
            "explicitly state mount materials, so the co-pilot should ask for mount/cold-finger and "
            "adhesive materials before making mount-contamination predictions."
        ),
        not_approved_operations=[
            "No live instrument control is authorized.",
            "No measurement is approved by this planning brief.",
            "No sample-environment safety decision is delegated to the co-pilot.",
        ],
        limitations=[
            "Fixture text is synthetic/paraphrased and does not commit private proposal text.",
            "Risk probabilities are planning priors for prioritizing checks, not certainty claims.",
            "Substrate, mount, and lambda/2 checks require facility geometry and material information before quantitative overlays.",
        ],
        provenance=[ProvenanceRecord(source_type="proposal_evidence_directory", source_id=str(fixture_dir), method="proposal_planning_brief")],
    )
    report = brief_model.model_dump(mode="json")
    report["workflow_name"] = "proposal-planning-brief"
    report["proposal_chunk_count"] = len(proposal_chunks)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_planning_markdown,
        json_filename="proposal_planning_brief.json",
        markdown_filename="proposal_planning_brief.md",
    )
    return report


def proposal_experiment_retrospective(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compare proposal-derived planning risks against retrospective experiment evidence."""

    fixture_dir = _proposal_evidence_dir(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixture_dir is None:
        return _missing(output_dir, "proposal-experiment-retrospective", ["fixture_or_evidence_dir"], "A fixture or proposal evidence directory is required.")
    actual_dir = Path(arguments.get("actual_evidence_dir") or fixture_dir).expanduser()
    if not fixture_dir.exists():
        return _missing(output_dir, "proposal-experiment-retrospective", ["fixture_or_evidence_dir"], f"Evidence directory does not exist: {fixture_dir}")
    if not actual_dir.exists():
        return _missing(output_dir, "proposal-experiment-retrospective", ["actual_evidence_dir"], f"Actual evidence directory does not exist: {actual_dir}")
    proposal_chunks = _load_jsonl(fixture_dir / "proposal_chunks.jsonl")
    actual_evidence = _load_jsonl(actual_dir / "actual_evidence.jsonl")
    if not proposal_chunks or not actual_evidence:
        missing = []
        if not proposal_chunks:
            missing.append("proposal_chunks")
        if not actual_evidence:
            missing.append("actual_evidence")
        return _missing(output_dir, "proposal-experiment-retrospective", missing, "Proposal chunks and actual evidence are required.")

    proposal_text = "\n".join(str(item.get("text") or "") for item in proposal_chunks)
    actual_text = "\n".join(str(item.get("text") or "") for item in actual_evidence)
    risks = [_evaluate_rule(rule, proposal_chunks, proposal_text, actual_evidence, actual_text, fixture_dir) for rule in _risk_rules()]
    predicted_observed = sum(1 for risk in risks if risk.retrospective_classification == "predicted_and_observed")
    promptable_observed = sum(1 for risk in risks if risk.retrospective_classification == "not_in_proposal_but_promptable")
    missed_observed = sum(1 for risk in risks if risk.retrospective_classification == "missed_observed_issue")
    report_model = ProposalExperimentRetrospective(
        retrospective_id=str(arguments.get("retrospective_id") or "ansto-dto-thin-film-retrospective"),
        proposal_id=str(arguments.get("proposal_id") or "ansto-dto-thin-film-proposal-synthetic"),
        experiment_id=arguments.get("experiment_id") or "ansto-exp1337-synthetic",
        status="ok",
        risk_predictions=risks,
        predicted_observed_count=predicted_observed,
        promptable_observed_count=promptable_observed,
        missed_observed_count=missed_observed,
        safe_conclusion=(
            "The proposal contained enough information to warn about substrate, lambda/2, "
            "alignment, weak-signal background, and dilution-temperature metadata risks. "
            "Cu mount scattering was not stated in the proposal, but the co-pilot should have "
            "asked for mount/cold-finger materials and then computed Cu overlays once known."
        ),
        limitations=[
            "Fixture text is synthetic/paraphrased and does not commit private proposal, notes, or raw data.",
            "This is deterministic risk triage, not a physics simulation or approved experiment plan.",
            "Risk probabilities are planning priors, not certainty claims.",
            "Exact Cu contamination requires actual mount geometry and beam footprint, not just material name.",
        ],
        next_actions=[
            "Add a proposal-planning brief that asks for missing mount/sample-environment materials before beamtime.",
            "Connect substrate, mount, filter, and lambda/2 risk rules to facility-specific instrument configuration records.",
            "Add optional local-private ingestion of proposal/notes paths into ignored reports for human review.",
            "Use retrospective scores as regression tests before building a bounded experiment-planning router.",
        ],
        provenance=[ProvenanceRecord(source_type="proposal_evidence_directory", source_id=str(fixture_dir), method="proposal_experiment_retrospective")],
    )
    report = report_model.model_dump(mode="json")
    report["workflow_name"] = "proposal-experiment-retrospective"
    report["proposal_chunk_count"] = len(proposal_chunks)
    report["actual_evidence_count"] = len(actual_evidence)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_retrospective_markdown,
        json_filename="proposal_experiment_retrospective.json",
        markdown_filename="proposal_experiment_retrospective.md",
    )
    return report


def suggest_experiment_scans(arguments: dict[str, Any]) -> dict[str, Any]:
    """Suggest a staged, non-executable scan plan from proposal evidence."""

    fixture_dir = _proposal_evidence_dir(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixture_dir is None:
        return _missing(output_dir, "suggest-experiment-scans", ["fixture_or_evidence_dir"], "A fixture or evidence directory is required.")
    if not fixture_dir.exists():
        return _missing(output_dir, "suggest-experiment-scans", ["fixture_or_evidence_dir"], f"Evidence directory does not exist: {fixture_dir}")
    actual_dir = Path(arguments.get("actual_evidence_dir") or fixture_dir).expanduser()
    proposal_chunks = _load_jsonl(fixture_dir / "proposal_chunks.jsonl")
    actual_evidence = _load_jsonl(actual_dir / "actual_evidence.jsonl")
    if not proposal_chunks:
        return _missing(output_dir, "suggest-experiment-scans", ["proposal_chunks"], "Proposal chunks are required.")

    proposal_text = "\n".join(str(item.get("text") or "") for item in proposal_chunks)
    actual_text = "\n".join(str(item.get("text") or "") for item in actual_evidence)
    suggestion_specs = _scan_suggestion_specs()
    suggestions = [
        _build_scan_suggestion(spec, proposal_chunks, proposal_text, actual_evidence, actual_text, fixture_dir)
        for spec in suggestion_specs
    ]
    plan = ExperimentScanPlan(
        plan_id=str(arguments.get("plan_id") or "ansto-dto-thin-film-scan-suggestions"),
        proposal_id=str(arguments.get("proposal_id") or "ansto-dto-thin-film-proposal-synthetic"),
        status="ok",
        suggestions=suggestions,
        missing_input_questions=[
            "Confirm substrate orientation, film orientation matrix, and whether the displayed reciprocal coordinates are substrate or film coordinates.",
            "Confirm mount, cold-finger, adhesive, and heat-link materials before finalizing background and lambda/2 overlays.",
            "Confirm filter stack, fixed final energy, analyzer mode, collimation, and temperature logging sidecar plan.",
            "Confirm which peaks are required for science versus alignment so long maps are not started before 1D checks pass.",
        ],
        dashboard_recommendations=[
            "No generalized experiment dashboard implementation was found in the tascopilot core repo; ANSTO/ORNL dashboards should be treated as ad hoc local analyses until lifted into a typed dashboard-spec workflow.",
            "A reusable dashboard should consume scan-plan, scan-catalog, overlay, temperature-sidecar, and provenance JSON rather than hard-coding one experiment directory.",
            "The first generalized dashboard surface should show scan table, 1D cuts, mesh view, substrate/Cu/lambda/2 overlays, temperature metadata, and source/provenance panels.",
            "For web output, Plotly is the lightest reasonable interactive plotting target; matplotlib should remain the static/report baseline.",
        ],
        not_approved_operations=[
            "These are planning suggestions, not approved instrument operations.",
            "No live instrument control is authorized.",
            "A human instrument scientist must review ranges, count times, temperature waits, and safety constraints before execution.",
        ],
        limitations=[
            "Suggested ranges are probability-style planning defaults from proposal and retrospective evidence, not optimized instrument commands.",
            "The plan does not compute TAS angles or collision/safety constraints.",
            "Substrate, Cu, and lambda/2 overlays require reviewed material, wavelength, and geometry inputs.",
            "Retrospective evidence can validate the suggestion pattern, but should not leak private local text into committed fixtures.",
        ],
        safe_conclusion=(
            "The co-pilot can propose a staged scan strategy for this thin-film experiment: first setup and metadata checks, "
            "then substrate and film alignment scans, then coarse/refined one-dimensional 003 scans, then temperature repeats "
            "and meshes. The output should be treated as decision support with probabilities and prerequisites, not as an "
            "autonomous run list."
        ),
        provenance=[ProvenanceRecord(source_type="proposal_evidence_directory", source_id=str(fixture_dir), method="suggest_experiment_scans")],
    )
    report = plan.model_dump(mode="json")
    report["workflow_name"] = "suggest-experiment-scans"
    report["proposal_chunk_count"] = len(proposal_chunks)
    report["actual_evidence_count"] = len(actual_evidence)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_scan_suggestions_markdown,
        json_filename="experiment_scan_suggestions.json",
        markdown_filename="experiment_scan_suggestions.md",
    )
    return report


def _evaluate_rule(
    rule: dict[str, Any],
    proposal_chunks: list[dict[str, Any]],
    proposal_text: str,
    actual_evidence: list[dict[str, Any]],
    actual_text: str,
    fixture_dir: Path,
) -> ExperimentPlanningRisk:
    predicted_terms = _matched_terms(proposal_text, rule.get("proposal_terms", []))
    prompt_terms = _matched_terms(proposal_text, rule.get("prompt_terms", []))
    actual_terms = _matched_terms(actual_text, rule.get("actual_terms", []))
    predicted = bool(predicted_terms)
    promptable = bool(prompt_terms)
    actual = bool(actual_terms)
    if actual and predicted:
        classification = "predicted_and_observed"
        probability = 0.8
    elif actual and promptable:
        classification = "not_in_proposal_but_promptable"
        probability = 0.55
    elif actual:
        classification = "missed_observed_issue"
        probability = 0.2
    elif predicted:
        classification = "predicted_not_observed_in_fixture"
        probability = 0.65
    else:
        classification = "not_observed"
        probability = 0.1
    probability += min(0.12, 0.02 * (len(predicted_terms) + len(prompt_terms) + len(actual_terms)))
    return ExperimentPlanningRisk(
        risk_id=rule["risk_id"],
        category=rule["category"],
        summary=rule["summary"],
        predicted_from_proposal=predicted,
        promptable_from_proposal=promptable,
        actual_observed=actual,
        retrospective_classification=classification,
        probability_score=round(min(probability, 0.95), 3),
        proposal_evidence=_evidence_hits(proposal_chunks, [*rule.get("proposal_terms", []), *rule.get("prompt_terms", [])]),
        actual_evidence=_evidence_hits(actual_evidence, rule.get("actual_terms", [])),
        missing_prebeam_inputs=rule.get("missing_prebeam_inputs", []),
        recommended_prebeam_checks=rule.get("recommended_prebeam_checks", []),
        provenance=[ProvenanceRecord(source_type="proposal_evidence_directory", source_id=str(fixture_dir), method="proposal_risk_rule", metadata={"matched_proposal_terms": predicted_terms, "matched_prompt_terms": prompt_terms, "matched_actual_terms": actual_terms})],
    )


def _proposal_evidence_dir(arguments: dict[str, Any]) -> Path | None:
    value = arguments.get("evidence_dir") or arguments.get("proposal_evidence_dir") or arguments.get("fixture")
    return Path(value).expanduser() if value else None


def _evaluate_planning_rule(
    rule: dict[str, Any],
    proposal_chunks: list[dict[str, Any]],
    proposal_text: str,
    fixture_dir: Path,
) -> ExperimentPlanningRisk:
    predicted_terms = _matched_terms(proposal_text, rule.get("proposal_terms", []))
    prompt_terms = _matched_terms(proposal_text, rule.get("prompt_terms", []))
    predicted = bool(predicted_terms)
    promptable = bool(prompt_terms)
    if predicted:
        planning = "explicit_warning"
        probability = 0.72
    elif promptable:
        planning = "missing_input_prompt"
        probability = 0.5
    else:
        planning = "not_triggered"
        probability = 0.1
    probability += min(0.16, 0.03 * (len(predicted_terms) + len(prompt_terms)))
    return ExperimentPlanningRisk(
        risk_id=rule["risk_id"],
        category=rule["category"],
        summary=rule["summary"],
        predicted_from_proposal=predicted,
        promptable_from_proposal=promptable,
        planning_classification=planning,
        retrospective_classification="not_evaluated_prebeam",
        probability_score=round(min(probability, 0.92), 3),
        proposal_evidence=_evidence_hits(proposal_chunks, [*rule.get("proposal_terms", []), *rule.get("prompt_terms", [])]),
        missing_prebeam_inputs=rule.get("missing_prebeam_inputs", []),
        recommended_prebeam_checks=rule.get("recommended_prebeam_checks", []),
        provenance=[ProvenanceRecord(source_type="proposal_evidence_directory", source_id=str(fixture_dir), method="proposal_planning_rule", metadata={"matched_proposal_terms": predicted_terms, "matched_prompt_terms": prompt_terms})],
    )


def _risk_rules() -> list[dict[str, Any]]:
    return [
        {
            "risk_id": "si_substrate_background",
            "category": "substrate",
            "summary": "Si substrate scattering/extinction assumptions can dominate or distort weak thin-film maps.",
            "proposal_terms": ["si substrate", "substrate", "extinction", "weak", "diffuse scattering"],
            "actual_terms": ["si substrate", "si reflections", "si 111", "tilt", "substrate frame"],
            "recommended_prebeam_checks": ["Overlay allowed and lambda/2 Si reflections for each planned map.", "Measure substrate/background checks where possible."],
        },
        {
            "risk_id": "lambda_half_higher_order",
            "category": "higher_order",
            "summary": "Higher-order contamination must be checked rather than assumed removed by filters.",
            "proposal_terms": ["higher-order", "graphite filters", "filters", "fixed final energy"],
            "actual_terms": ["lambda/2", "higher-order", "pg filter", "sapphire filter"],
            "recommended_prebeam_checks": ["Verify filter configuration and include lambda/2 overlays for substrate, film, and mount materials."],
        },
        {
            "risk_id": "cu_mount_powder_rings",
            "category": "sample_environment",
            "summary": "Cu mount or cold-finger material can introduce powder rings near weak film features.",
            "proposal_terms": ["cu mount", "copper", "cu plate"],
            "prompt_terms": ["dilution", "reflection geometry", "thin film", "60 mK"],
            "actual_terms": ["cu plate", "cu mount", "cu reflection", "cu powder", "sample plate"],
            "missing_prebeam_inputs": ["mount material", "cold-finger material", "beam footprint on mount", "sample adhesive/material stack"],
            "recommended_prebeam_checks": ["Require a sample-environment material inventory.", "Compute Cu/Al/cryostat lambda and lambda/2 positions before beamtime."],
        },
        {
            "risk_id": "temperature_sidecar_metadata",
            "category": "metadata",
            "summary": "Dilution-fridge temperature values may be sidecar metadata and must be synchronized with scan files.",
            "proposal_terms": ["60 mK", "dilution", "base temperature", "400 mK"],
            "actual_terms": ["temperature was not stored", "sharefile", "temperature set points", "mixing chamber"],
            "recommended_prebeam_checks": ["Confirm which temperature sensors are written into NeXus/data files.", "Create a sidecar-log synchronization plan."],
        },
        {
            "risk_id": "film_substrate_alignment",
            "category": "alignment",
            "summary": "Thin-film/substrate orientation, miscut, and coordinate-frame conventions need explicit alignment checks.",
            "proposal_terms": ["si (111)", "dy2ti2o7 (222)", "hhl", "film", "relaxed"],
            "actual_terms": ["si 111 alignment", "dy2ti2o7 222", "working frame", "tilt", "backlash"],
            "recommended_prebeam_checks": ["Precompute substrate-to-film coordinate conversions.", "Plan Si and film alignment scans plus domain/miscut checks."],
        },
        {
            "risk_id": "weak_signal_background_resolution",
            "category": "background_resolution",
            "summary": "Weak thin-film elastic features require background, resolution, and normalization checks.",
            "proposal_terms": ["weak features", "low background", "thin film", "absorption", "reflection geometry", "6 hours"],
            "actual_terms": ["main beam falloff", "background contamination", "high-flux", "resolution", "voigt", "weak signal"],
            "recommended_prebeam_checks": ["Plan empty/mount/background scans.", "Estimate resolution before interpreting peak widths.", "Track normalization and subtraction provenance."],
        },
    ]


def _scan_suggestion_specs() -> list[dict[str, Any]]:
    return [
        {
            "suggestion_id": "setup_metadata_inventory",
            "priority": 1,
            "phase": "setup",
            "scan_type": "prebeam_checklist",
            "target": "instrument/sample-environment metadata",
            "purpose": "Avoid starting alignment before filter, mount, temperature, and coordinate-frame metadata are explicit.",
            "probability_score": 0.92,
            "terms": ["fixed final energy", "filters", "dilution", "reflection geometry", "temperature"],
            "actual_terms": ["temperature was not stored", "cu plate", "cu reflection", "LogFile", "History"],
            "required_before_run": ["filter stack", "fixed final energy", "mount/cold-finger/adhesive materials", "temperature sidecar logging", "substrate and film coordinate frames"],
            "artifact_checks": ["prepare Si, Cu, film, and lambda/2 overlay inputs before long scans"],
        },
        {
            "suggestion_id": "substrate_si111_rocking",
            "priority": 2,
            "phase": "substrate_alignment",
            "scan_type": "rocking_or_theta_scan",
            "target": "Si(111)",
            "scan_direction": "rocking",
            "purpose": "Center the substrate normal and establish whether substrate extinction/background can dominate weak film signal.",
            "probability_score": 0.86,
            "terms": ["si (111)", "substrate", "reflection geometry"],
            "actual_terms": ["si 111", "si reflections", "si 111 alignment"],
            "suggested_range": {"axis": "rocking_angle", "center": "predicted Si(111)", "half_width_deg": 1.0},
            "step": 0.05,
            "count_time": "short alignment counts",
            "artifact_checks": ["check lambda and lambda/2 Si reflection positions"],
        },
        {
            "suggestion_id": "substrate_offaxis_orientation",
            "priority": 3,
            "phase": "substrate_alignment",
            "scan_type": "off_axis_alignment_scan",
            "target": "Si(112) or Si(113)",
            "scan_direction": "reciprocal_space_alignment",
            "purpose": "Determine azimuth, miscut, and substrate-to-film coordinate conversion before film maps.",
            "probability_score": 0.78,
            "terms": ["hhl", "si (111)", "substrate"],
            "actual_terms": ["si 11-2", "si113", "substrate frame", "tilt", "working frame"],
            "required_before_run": ["candidate substrate off-axis peak list", "substrate-to-film transform convention"],
        },
        {
            "suggestion_id": "film_dy222_search",
            "priority": 4,
            "phase": "film_alignment",
            "scan_type": "rocking_and_longitudinal_q",
            "target": "Dy2Ti2O7(222)",
            "scan_direction": "longitudinal_then_rocking",
            "purpose": "Find the film reference peak and validate film orientation before targeting weak forbidden or diffuse features.",
            "probability_score": 0.84,
            "terms": ["dy2ti2o7 (222)", "film", "hhl"],
            "actual_terms": ["dy2ti2o7 222", "dy222", "115533"],
            "suggested_range": {"axis": "L_or_2theta", "center": "predicted Dy2Ti2O7(222)", "half_width_rlu": 0.15},
            "step": 0.01,
            "count_time": "alignment to moderate counts",
        },
        {
            "suggestion_id": "film_secondary_peak_check",
            "priority": 5,
            "phase": "film_alignment",
            "scan_type": "secondary_peak_validation",
            "target": "Dy2Ti2O7(224), 004, or 440",
            "scan_direction": "longitudinal_or_transverse",
            "purpose": "Check lattice/orientation consistency and reduce the chance of interpreting substrate-frame features as film scattering.",
            "probability_score": 0.72,
            "terms": ["hhl", "film", "relaxed"],
            "actual_terms": ["dy224", "440", "004"],
        },
        {
            "suggestion_id": "science_003_coarse_longitudinal",
            "priority": 6,
            "phase": "science_scouting",
            "scan_type": "longitudinal_q_scan",
            "target": "Dy2Ti2O7 003 region",
            "scan_direction": "longitudinal",
            "purpose": "Before a 2D map, test whether the 003-region feature is present and where the local maximum lies.",
            "probability_score": 0.88,
            "terms": ["(003)", "003", "weak features", "low background"],
            "actual_terms": ["003", "115532", "115546"],
            "suggested_range": {"H": 0, "K": 0, "L_min": 2.5, "L_max": 3.5},
            "fixed_coordinates": {"H": 0, "K": 0},
            "step": 0.05,
            "count_time": "short scouting counts; increase only after feature/local background is visible",
            "artifact_checks": ["overlay Si, Cu, and lambda/2 positions before interpreting a peak"],
        },
        {
            "suggestion_id": "science_003_refined_longitudinal",
            "priority": 7,
            "phase": "science_refinement",
            "scan_type": "longitudinal_q_scan",
            "target": "observed 003-region maximum",
            "scan_direction": "longitudinal",
            "purpose": "Refine peak center/width after the coarse scan, with background sidebands recorded in the same coordinate frame.",
            "probability_score": 0.8,
            "terms": ["(003)", "003", "6 hours", "weak"],
            "actual_terms": ["003", "longitudinal", "voigt"],
            "suggested_range": {"axis": "L", "center": "coarse_scan_maximum", "half_width_rlu": 0.2},
            "fixed_coordinates": {"H": 0, "K": 0},
            "step": 0.02,
            "count_time": "moderate counts after coarse feature check",
            "required_before_run": ["science_003_coarse_longitudinal"],
            "artifact_checks": ["fit raw points before smoothing or meshing"],
        },
        {
            "suggestion_id": "science_003_transverse",
            "priority": 8,
            "phase": "science_refinement",
            "scan_type": "transverse_q_scan",
            "target": "003-region transverse width/background",
            "scan_direction": "transverse",
            "purpose": "Separate film peak shape from background streaks and substrate/mount contamination by scanning transverse to the 003 direction.",
            "probability_score": 0.78,
            "terms": ["(003)", "weak features", "hhl"],
            "actual_terms": ["transverse", "115532", "background contamination"],
            "suggested_range": {"axis": "H=K", "min": -0.3, "max": 0.3, "fixed_L": "coarse_scan_maximum_near_3"},
            "step": 0.025,
            "count_time": "short to moderate counts",
            "required_before_run": ["science_003_coarse_longitudinal"],
        },
        {
            "suggestion_id": "control_002_222_repeats",
            "priority": 9,
            "phase": "controls",
            "scan_type": "reference_peak_repeats",
            "target": "002 and 222 reference regions",
            "scan_direction": "longitudinal_or_rocking",
            "purpose": "Track alignment, normalization, and temperature dependence against reference peaks during the same setup.",
            "probability_score": 0.76,
            "terms": ["(002)", "dy2ti2o7 (222)", "temperature"],
            "actual_terms": ["002", "222", "115553", "115554", "115555"],
            "temperature_targets": ["base 45-60 mK", "400-500 mK", "800 mK", "1.2 K"],
        },
        {
            "suggestion_id": "temperature_series_minimal",
            "priority": 10,
            "phase": "temperature_dependence",
            "scan_type": "temperature_series",
            "target": "003-region feature plus reference/background checks",
            "purpose": "Measure a small temperature ladder before spending beamtime on long maps.",
            "probability_score": 0.82,
            "terms": ["60 mK", "400 mK", "1.2 K", "base temperature"],
            "actual_terms": ["45", "600mk", "800mk", "temperature set points", "mixing chamber"],
            "temperature_targets": ["base 45-60 mK", "400-500 mK", "800 mK", "1.2 K", "high-temperature reference if time allows"],
            "required_before_run": ["temperature sidecar logging"],
            "artifact_checks": ["repeat at least one reference/background scan after temperature changes"],
        },
        {
            "suggestion_id": "mesh_after_1d_checks",
            "priority": 11,
            "phase": "mapping",
            "scan_type": "2d_mesh",
            "target": "003-region HHL map",
            "scan_direction": "H=K versus L",
            "purpose": "Run a 2D map only after 1D scans show a feature and overlays do not make a background explanation more likely.",
            "probability_score": 0.7,
            "terms": ["maps", "six-hour maps", "hhl", "003"],
            "actual_terms": ["mesh", "115536", "115537", "dashboard"],
            "suggested_range": {"H=K_min": -0.5, "H=K_max": 0.5, "L_min": 2.5, "L_max": 3.5},
            "step": 0.05,
            "count_time": "long counts only after 1D and overlay checks pass",
            "required_before_run": ["science_003_coarse_longitudinal", "science_003_transverse", "control_002_222_repeats"],
            "artifact_checks": ["compare raw points, binned map, smoothed map, and sidebands separately"],
        },
    ]


def _build_scan_suggestion(
    spec: dict[str, Any],
    proposal_chunks: list[dict[str, Any]],
    proposal_text: str,
    actual_evidence: list[dict[str, Any]],
    actual_text: str,
    fixture_dir: Path,
) -> ExperimentScanSuggestion:
    proposal_terms = _matched_terms(proposal_text, spec.get("terms", []))
    actual_terms = _matched_terms(actual_text, spec.get("actual_terms", []))
    probability = float(spec.get("probability_score", 0.5)) + min(0.08, 0.01 * (len(proposal_terms) + len(actual_terms)))
    evidence = [
        *_evidence_hits(proposal_chunks, spec.get("terms", [])),
        *_evidence_hits(actual_evidence, spec.get("actual_terms", [])),
    ]
    return ExperimentScanSuggestion(
        suggestion_id=spec["suggestion_id"],
        priority=int(spec["priority"]),
        phase=spec["phase"],
        scan_type=spec["scan_type"],
        target=spec["target"],
        scan_direction=spec.get("scan_direction"),
        suggested_range=spec.get("suggested_range", {}),
        fixed_coordinates=spec.get("fixed_coordinates", {}),
        step=spec.get("step"),
        count_time=spec.get("count_time"),
        temperature_targets=spec.get("temperature_targets", []),
        purpose=spec["purpose"],
        probability_score=round(min(probability, 0.95), 3),
        evidence=evidence,
        required_before_run=spec.get("required_before_run", []),
        artifact_checks=spec.get("artifact_checks", []),
        safety_notes=[
            "Human review required before instrument execution.",
            "This suggestion is not an approved operation.",
        ],
        provenance=[
            ProvenanceRecord(
                source_type="proposal_evidence_directory",
                source_id=str(fixture_dir),
                method="deterministic_scan_suggestion_rule",
                metadata={"matched_proposal_terms": proposal_terms, "matched_actual_terms": actual_terms},
            )
        ],
    )


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    text_lower = text.lower()
    return [term for term in terms if term.lower() in text_lower]


def _evidence_hits(records: list[dict[str, Any]], terms: list[str]) -> list[str]:
    hits = []
    for record in records:
        text = str(record.get("text") or "")
        matched = _matched_terms(text, terms)
        if matched:
            hits.append(f"{record.get('evidence_id') or record.get('chunk_id')}: {', '.join(matched)}")
    return hits


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _records_from_documents(paths: list[Path], *, role: str, max_chars: int, max_pages: int) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        source_hash = _file_sha256(path) if path.exists() else None
        diagnostic_id = f"{role}-diagnostic-{index}-{_safe_stem(path)}"
        if not path.exists():
            message = f"{role} document missing: {path}"
            warnings.append(message)
            diagnostics.append(
                _diagnostic_record(
                    diagnostic_id=diagnostic_id,
                    role=role,
                    path=path,
                    status="missing_input",
                    source_hash=source_hash,
                    warnings=[message],
                )
            )
            continue
        text, method, extract_warnings = _extract_private_text(path, max_chars=max_chars, max_pages=max_pages)
        warnings.extend(extract_warnings)
        if not text.strip():
            message = f"{role} document produced no text: {path}"
            warnings.append(message)
            diagnostics.append(
                _diagnostic_record(
                    diagnostic_id=diagnostic_id,
                    role=role,
                    path=path,
                    status="no_text",
                    source_hash=source_hash,
                    extraction_method=method,
                    warnings=[*extract_warnings, message],
                )
            )
            continue
        record_id = f"{role}-{index}-{_safe_stem(path)}"
        evidence_type, confidence, metadata = _classify_local_evidence(path, role=role, text=text, extraction_method=method)
        provenance = [
            ProvenanceRecord(
                source_type=evidence_type,
                source_id=str(path),
                method=method,
                confidence=confidence,
                metadata={"source_hash": source_hash, **metadata},
            ).model_dump(mode="json")
        ]
        diagnostics.append(
            _diagnostic_record(
                diagnostic_id=diagnostic_id,
                role=role,
                path=path,
                status="extracted",
                source_hash=source_hash,
                extraction_method=method,
                text_char_count=len(text),
                confidence=confidence,
                warnings=extract_warnings,
                metadata=metadata,
            )
        )
        record = ProposalEvidenceRecord(
            record_id=record_id,
            evidence_role=role,
            evidence_type=evidence_type,
            text=text,
            source_path=str(path),
            source_hash=source_hash,
            extraction_method=method,
            extraction_status="extracted",
            confidence=confidence,
            warnings=extract_warnings,
            metadata=metadata,
            provenance=provenance,
        )
        if role == "proposal":
            record.chunk_id = record_id
            record.chunk_type = evidence_type
            record.source_type = evidence_type
        else:
            record.evidence_id = record_id
        records.append(record.model_dump(mode="json"))
    return records, warnings, diagnostics


def _diagnostic_record(
    *,
    diagnostic_id: str,
    role: str,
    path: Path,
    status: str,
    source_hash: str | None = None,
    extraction_method: str | None = None,
    text_char_count: int = 0,
    confidence: float | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = ProposalEvidenceDiagnostic(
        diagnostic_id=diagnostic_id,
        evidence_role=role,
        source_path=str(path),
        status=status,
        extraction_method=extraction_method,
        text_char_count=text_char_count,
        source_hash=source_hash,
        confidence=confidence,
        warnings=warnings or [],
        metadata=metadata or {},
        provenance=[
            ProvenanceRecord(
                source_type="local_private_evidence_diagnostic",
                source_id=str(path),
                method=extraction_method,
                confidence=confidence,
            )
        ],
    ).model_dump(mode="json")
    # Backward-compatible aliases for existing reports and tests.
    record["role"] = record["evidence_role"]
    record["path"] = record["source_path"]
    return record


def _classify_local_evidence(path: Path, *, role: str, text: str, extraction_method: str) -> tuple[str, float, dict[str, Any]]:
    """Classify local-private evidence without treating its contents as instructions."""

    suffix = path.suffix.lower()
    name = path.name.lower()
    text_lower = text.lower()
    metadata: dict[str, Any] = {"role": role, "suffix": suffix}
    if "codex_multimodal" in name or "codex multimodal evidence summary" in text_lower:
        return "codex_multimodal_ocr_summary", 0.65, {**metadata, "adapter_note": "model-read OCR triage evidence requiring human review"}
    if suffix == ".py":
        return "analysis_script", 0.7, {**metadata, "adapter_note": "local analysis code evidence; content is not executed by ingest"}
    if name in {"logfile.txt", "history.txt"} or "log" in name:
        return "instrument_log", 0.72, metadata
    if suffix == ".docx" and role == "proposal":
        return "proposal_document_text", 0.82, metadata
    if suffix == ".docx":
        return "actual_notes_document_text", 0.82, metadata
    if suffix == ".pdf":
        return "pdf_extracted_text", 0.5, {**metadata, "adapter_note": "baseline PDF text extraction, layout and handwriting may be missing"}
    if extraction_method == "safe_text_file_first_chars" and role == "proposal":
        return "proposal_document_text", 0.75, metadata
    if extraction_method == "safe_text_file_first_chars":
        return f"{role}_text_evidence", 0.75, metadata
    return f"{role}_local_private_evidence", 0.6, metadata


def _extract_private_text(path: Path, *, max_chars: int, max_pages: int) -> tuple[str, str, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".py", ".csv", ".json", ".yaml", ".yml", ".html"}:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars], "safe_text_file_first_chars", []
    if suffix == ".docx":
        textutil = shutil.which("textutil")
        if textutil:
            try:
                completed = subprocess.run([textutil, "-convert", "txt", "-stdout", str(path)], check=False, capture_output=True, text=True, timeout=120)
            except Exception as exc:  # pragma: no cover - subprocess behavior varies
                return "", "textutil_docx", [f"textutil failed for {path}: {exc}"]
            if completed.returncode == 0:
                return completed.stdout[:max_chars], "textutil_docx", []
            return "", "textutil_docx", [f"textutil returned {completed.returncode} for {path}: {completed.stderr[:300]}"]
        return "", "textutil_docx_unavailable", ["textutil is not available for DOCX extraction."]
    if suffix == ".pdf":
        pdftotext = shutil.which("pdftotext")
        if not pdftotext:
            return "", "pdftotext_unavailable", ["pdftotext is not available for PDF extraction."]
        try:
            completed = subprocess.run([pdftotext, "-enc", "UTF-8", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"], check=False, capture_output=True, text=True, timeout=120)
        except Exception as exc:  # pragma: no cover - subprocess behavior varies
            return "", "pdftotext_first_pages", [f"pdftotext failed for {path}: {exc}"]
        if completed.returncode == 0:
            warnings = ["pdftotext is a baseline extractor; scanned PDFs or figure captions may be incomplete."]
            if not completed.stdout.strip():
                warnings.append("pdftotext returned no text; this is likely an image-only or scanned PDF and requires OCR-capable extraction before evidence use.")
            return completed.stdout[:max_chars], "pdftotext_first_pages", warnings
        return "", "pdftotext_first_pages", [f"pdftotext returned {completed.returncode} for {path}: {completed.stderr[:300]}"]
    return "", "unsupported_extension", [f"Unsupported evidence document extension: {path.suffix}"]


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in path.stem.lower()).strip("-")[:80] or "document"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing(output_dir: Path, workflow_name: str, fields: list[str], message: str) -> dict[str, Any]:
    report = {
        "workflow_name": workflow_name,
        "status": "missing_input",
        "missing_fields": fields,
        "warnings": [message],
    }
    titles = {
        "proposal-planning-brief": "Proposal Planning Brief",
        "proposal-experiment-retrospective": "Proposal Experiment Retrospective",
        "proposal-evidence-ingest": "Proposal Evidence Ingest",
        "suggest-experiment-scans": "Experiment Scan Suggestions",
    }
    json_names = {
        "proposal-planning-brief": "proposal_planning_brief.json",
        "proposal-experiment-retrospective": "proposal_experiment_retrospective.json",
        "proposal-evidence-ingest": "proposal_evidence_ingest.json",
        "suggest-experiment-scans": "experiment_scan_suggestions.json",
    }
    markdown_names = {
        "proposal-planning-brief": "proposal_planning_brief.md",
        "proposal-experiment-retrospective": "proposal_experiment_retrospective.md",
        "proposal-evidence-ingest": "proposal_evidence_ingest.md",
        "suggest-experiment-scans": "experiment_scan_suggestions.md",
    }
    title = titles.get(workflow_name, workflow_name)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=lambda item: f"# {title}\n\nStatus: missing_input\n\n" + "\n".join(f"- {warning}" for warning in item.get("warnings", [])) + "\n",
        json_filename=json_names.get(workflow_name, "proposal_workflow.json"),
        markdown_filename=markdown_names.get(workflow_name, "proposal_workflow.md"),
    )
    return report


def _ingest_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Proposal Evidence Ingest",
        "",
        f"Status: {report.get('status')}",
        f"Evidence directory: {report.get('evidence_dir')}",
        "",
        "## Counts",
        "",
        f"- Proposal documents: {report.get('proposal_document_count', 0)}",
        f"- Actual documents: {report.get('actual_document_count', 0)}",
        f"- Proposal chunks: {report.get('proposal_chunk_count', 0)}",
        f"- Actual evidence records: {report.get('actual_evidence_count', 0)}",
        "",
        "## Document Diagnostics",
        "",
        "| Role | Status | Method | Characters | Path |",
        "|---|---:|---|---:|---|",
    ]
    for diagnostic in report.get("document_diagnostics", []):
        lines.append(
            "| {role} | {status} | {method} | {chars} | {path} |".format(
                role=diagnostic.get("role"),
                status=diagnostic.get("status"),
                method=diagnostic.get("extraction_method") or "none",
                chars=diagnostic.get("text_char_count", 0),
                path=diagnostic.get("path"),
            )
        )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    return "\n".join(lines) + "\n"


def _scan_suggestions_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Scan Suggestions",
        "",
        f"Status: {report.get('status')}",
        f"Proposal: {report.get('proposal_id')}",
        "",
        "These suggestions are planning evidence, not approved instrument operations.",
        "",
        "## Suggested Sequence",
        "",
        "| Priority | Suggestion | Phase | Type | Target | Direction | Probability |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for item in sorted(report.get("suggestions", []), key=lambda entry: entry.get("priority", 0)):
        lines.append(
            "| {priority} | {sid} | {phase} | {scan_type} | {target} | {direction} | {probability:.3f} |".format(
                priority=item.get("priority"),
                sid=item.get("suggestion_id"),
                phase=item.get("phase"),
                scan_type=item.get("scan_type"),
                target=item.get("target"),
                direction=item.get("scan_direction") or "n/a",
                probability=float(item.get("probability_score", 0)),
            )
        )
    lines.extend(["", "## Details", ""])
    for item in sorted(report.get("suggestions", []), key=lambda entry: entry.get("priority", 0)):
        lines.extend(
            [
                f"### {item.get('priority')}. {item.get('suggestion_id')}",
                "",
                f"- Target: {item.get('target')}",
                f"- Purpose: {item.get('purpose')}",
                f"- Suggested range: {json.dumps(item.get('suggested_range', {}), sort_keys=True)}",
                f"- Fixed coordinates: {json.dumps(item.get('fixed_coordinates', {}), sort_keys=True)}",
                f"- Step: {item.get('step') if item.get('step') is not None else 'not specified'}",
                f"- Count time: {item.get('count_time') or 'human review required'}",
                f"- Temperature targets: {', '.join(item.get('temperature_targets', [])) or 'not specified'}",
                f"- Required before run: {', '.join(item.get('required_before_run', [])) or 'none listed'}",
                f"- Artifact checks: {', '.join(item.get('artifact_checks', [])) or 'none listed'}",
                f"- Evidence: {'; '.join(item.get('evidence', [])) or 'none'}",
                "",
            ]
        )
    lines.extend(["## Dashboard Generalization", ""])
    lines.extend(f"- {item}" for item in report.get("dashboard_recommendations", []))
    lines.extend(["", "## Missing Inputs", ""])
    lines.extend(f"- {item}" for item in report.get("missing_input_questions", []))
    lines.extend(["", "## Safe Conclusion", "", str(report.get("safe_conclusion", "")), "", "## Not Approved Operations", ""])
    lines.extend(f"- {item}" for item in report.get("not_approved_operations", []))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def _missing_input_questions(risks: list[ExperimentPlanningRisk]) -> list[str]:
    questions = []
    for risk in risks:
        for item in risk.missing_prebeam_inputs:
            questions.append(f"{risk.risk_id}: provide {item}.")
    questions.extend(
        [
            "Confirm which filters, analyzers, and sample-environment components will be in the beam for production scans.",
            "Confirm how temperature, field, and sample-environment logs will be synchronized with scan files.",
        ]
    )
    return _unique(questions)


def _unique(items) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _planning_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Proposal Planning Brief",
        "",
        f"Status: {report.get('status')}",
        f"Proposal: {report.get('proposal_id')}",
        "",
        "## Risk Priorities",
        "",
        "| Risk | Planning classification | Probability | Evidence | Checks |",
        "|---|---:|---:|---|---|",
    ]
    for risk in report.get("risk_predictions", []):
        if risk.get("planning_classification") == "not_triggered":
            continue
        lines.append(
            "| {risk} | {classification} | {probability:.3f} | {evidence} | {checks} |".format(
                risk=risk.get("risk_id"),
                classification=risk.get("planning_classification"),
                probability=float(risk.get("probability_score", 0)),
                evidence="<br>".join(risk.get("proposal_evidence", [])) or "none",
                checks="<br>".join(risk.get("recommended_prebeam_checks", [])) or "none",
            )
        )
    lines.extend(["", "## Missing Inputs", ""])
    lines.extend(f"- {item}" for item in report.get("missing_input_questions", []))
    lines.extend(["", "## Required Prebeam Checks", ""])
    lines.extend(f"- {item}" for item in report.get("required_prebeam_checks", []))
    lines.extend(["", "## Safe Conclusion", "", str(report.get("safe_conclusion", "")), "", "## Not Approved Operations", ""])
    lines.extend(f"- {item}" for item in report.get("not_approved_operations", []))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def _retrospective_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Proposal Experiment Retrospective",
        "",
        f"Status: {report.get('status')}",
        f"Proposal: {report.get('proposal_id')}",
        f"Experiment: {report.get('experiment_id')}",
        "",
        "## Summary",
        "",
        f"- Predicted and observed: {report.get('predicted_observed_count', 0)}",
        f"- Not in proposal but promptable: {report.get('promptable_observed_count', 0)}",
        f"- Missed observed issues: {report.get('missed_observed_count', 0)}",
        "",
        "## Risk Checks",
        "",
        "| Risk | Classification | Probability | Proposal evidence | Actual evidence |",
        "|---|---:|---:|---|---|",
    ]
    for risk in report.get("risk_predictions", []):
        lines.append(
            "| {risk} | {classification} | {probability:.3f} | {proposal} | {actual} |".format(
                risk=risk.get("risk_id"),
                classification=risk.get("retrospective_classification"),
                probability=float(risk.get("probability_score", 0)),
                proposal="<br>".join(risk.get("proposal_evidence", [])) or "none",
                actual="<br>".join(risk.get("actual_evidence", [])) or "none",
            )
        )
    lines.extend(["", "## Safe Conclusion", "", str(report.get("safe_conclusion", "")), "", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report.get("next_actions", []))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"
