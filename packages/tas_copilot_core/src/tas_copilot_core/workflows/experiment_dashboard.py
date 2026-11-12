from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from tas_copilot_contracts import (
    CandidateEvidence,
    DashboardOverlayLayer,
    DashboardScanRecord,
    ExperimentDashboardSpec,
    ProvenanceRecord,
    ReductionCorrectionRef,
    ReviewLabel,
    ReviewLabelBatch,
    SpurionLikelihoodCandidate,
    ToolCall,
)
from tas_copilot_core.counting import counting_measurement
from tas_copilot_core.catalogs import ReviewOutcomeSQLiteStore
from tas_copilot_core.instrument_calibrations import VELOCITY_SELECTOR_KIND, find_instrument_calibration
from tas_copilot_core.instrument_state import harmonic_suppression_state
from tas_copilot_core.review_labels import label_calibrated_score, label_informed_priors, review_label_calibration_diagnostics
from tas_copilot_core.workflows.common import default_runner
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.reduction_dag import (
    execute_reduction_dag,
    order_parameter_fit_branch_template,
    order_parameter_temperature_series_template,
    replay_reduction_dag_run,
)


SCAN_PATTERNS = ("*.bt7", "*.dat", "*.ng5", "*.bt4", "*.bt9")
INTENSITY_COLUMNS = ("Detector", "SingleDet", "PSDet", "DiffDet", "Detector1")
SCAN_AXIS_COLUMNS = ("E", "QX", "QY", "QZ", "H", "K", "L", "A4", "Temp", "TemperatureSetpoint", "Time")
REVIEW_COLUMNS = frozenset({
    "H", "K", "L", "QX", "QY", "QZ", "E", "Ef", "Ei",
    "A1", "A2", "A3", "A4", "A5", "A6",
    "SmplUTilt", "SmplLTilt", "SmplHght", "SmplElev",
    "Detector", "SingleDet", "PSDet", "DiffDet", "Monitor",
    "Temp", "TemperatureSetpoint", "Time",
})
MATERIAL_TERMS = {
    "Al": ("al", "aluminum", "aluminium"),
    "Cu": ("cu", "copper"),
    "sample_environment": ("cryostat", "ccr", "dilution", "fridge", "can", "mount", "cold finger"),
    "PG": ("pg", "graphite", "pyrolytic graphite"),
    "velocity_selector": ("velocity selector", "velsel", "selector"),
}
OVERLAY_PACKS = {
    "bt7_common": [
        {"overlay_type": "material_reference", "label": "Aluminum in beam", "material": "Al", "role": "mount_or_sample_environment"},
        {"overlay_type": "material_reference", "label": "Copper in beam", "material": "Cu", "role": "mount_or_dilution_refrigerator"},
        {"overlay_type": "material_reference", "label": "Silicon substrate", "material": "Si", "role": "substrate"},
        {"overlay_type": "material_reference", "label": "Pyrolytic graphite", "material": "PG", "role": "monochromator_analyzer_or_filter"},
    ],
    "thin_film_common": [
        {"overlay_type": "material_reference", "label": "Substrate peak family", "material": "substrate", "role": "substrate"},
        {"overlay_type": "material_reference", "label": "Film peak family", "material": "sample", "role": "sample"},
        {"overlay_type": "material_reference", "label": "Cryostat or fridge background", "material": "cryostat", "role": "sample_environment"},
    ],
}
DEFAULT_PRECEDENT_INDEXES = (
    Path("local_runs/bt7_hydrated_index/bt7_hydrated_index_full_source_units.jsonl"),
    Path("local_runs/bt7_hydrated_index/bt7_hydrated_index_full.jsonl"),
)


def _with_experiment_config(arguments: dict[str, Any]) -> dict[str, Any]:
    merged = dict(arguments)
    config_path = arguments.get("experiment_config")
    config: dict[str, Any] = {}
    if config_path:
        config = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8")) or {}
        merged["experiment_config_path"] = str(config_path)
        merged["experiment_config_data"] = config

    for key in ("overlay_packs", "materials_in_beam"):
        if key not in merged and key in config:
            merged[key] = config[key]
    if "overlays" in config:
        merged.setdefault("config_overlays", config["overlays"])
    if "resolution_proxy" in config:
        proxy = dict(config["resolution_proxy"] or {})
        proxy.update(dict(merged.get("resolution_proxy") or {}))
        merged["resolution_proxy"] = proxy

    instrument_config = dict(config.get("instrument_config") or {})
    instrument_config.update(dict(merged.get("instrument_config") or {}))
    lattice = merged.get("lattice") or config.get("lattice")
    if instrument_config:
        merged["instrument_config"] = instrument_config
    if lattice:
        merged["lattice"] = lattice

    spurion_check = dict(config.get("spurion_check") or {})
    spurion_check.update(dict(merged.get("spurion_check") or {}))
    if lattice and "lattice" not in spurion_check:
        spurion_check["lattice"] = lattice
    if instrument_config and "instrument_config" not in spurion_check:
        spurion_check["instrument_config"] = instrument_config
    if lattice and instrument_config and "enabled" not in spurion_check:
        spurion_check["enabled"] = True
    if spurion_check:
        merged["spurion_check"] = spurion_check
    return merged


def build_experiment_dashboard(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a reusable local experiment dashboard package from scan files."""

    arguments = _with_experiment_config(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments["data_root"]).expanduser() if arguments.get("data_root") else None
    if data_root is None:
        return _missing(output_dir, "experiment-dashboard", ["data_root"], "A data root is required.")
    if not data_root.exists():
        return _missing(output_dir, "experiment-dashboard", ["data_root"], f"Data root does not exist: {data_root}")

    max_files = int(arguments.get("max_files", 500))
    post_year = int(arguments.get("post_year", 0) or 0)
    patterns = tuple(arguments.get("patterns") or SCAN_PATTERNS)
    paths = _find_scan_paths(data_root, patterns=patterns, max_files=max_files, post_year=post_year)
    records = [_parse_scan_file(path, max_plot_points=int(arguments.get("max_plot_points", 600))) for path in paths]
    alignment_fits = _load_alignment_fit_overlays(arguments.get("alignment_state_json"))
    if alignment_fits:
        for record in records:
            fit = alignment_fits.get(Path(str(record.path)).stem) if record.path else None
            if fit:
                record.metadata["alignment_fit"] = fit
    spurion_candidates = _rank_spurion_likelihood(
        records,
        top_k=int(arguments.get("spurion_top_k", 50)),
        ranking_mode=str(arguments.get("ranking_mode") or "diversified"),
        spurion_check=arguments.get("spurion_check") or {},
        resolution_proxy=arguments.get("resolution_proxy") or {},
        review_label_store=arguments.get("review_label_store") or arguments.get("review_outcomes"),
    )
    diagnostic_scans = _recommended_diagnostic_scans(spurion_candidates, records, arguments)
    overlays = _overlay_layers(arguments)

    spec = ExperimentDashboardSpec(
        dashboard_id=str(arguments.get("dashboard_id") or f"dashboard-{_safe_id(data_root.name)}"),
        title=str(arguments.get("title") or f"Experiment Dashboard: {data_root.name}"),
        data_root=str(data_root),
        update_policy={
            "mode": "manual_refresh_or_periodic_json_reload",
            "static_dashboard": "dashboard.html",
            "dash_app": "app.py",
            "note": "Dash app reloads dashboard_data.json periodically; rerun this workflow or enable the app-side rescan hook to refresh source data.",
        },
        scan_records=records,
        overlay_layers=overlays,
        artifact_checks=[
            "Flag unusually narrow features relative to neighboring peaks in the same scan.",
            "Treat material overlays as experiment-specific layers that require known materials in beam.",
            "Call tas.spurion.check_spurions only when explicit Q/E, lattice, and instrument configuration are available.",
            "Use scan-family repeatability and width-vs-resolution proxies as probability evidence, not conclusions.",
            "Do not claim spurion/artifact status without follow-up scans, background checks, and human review.",
        ],
        warnings=[
            "Dashboard scan parsing is read-only and local.",
            "Generated app.py requires optional dash; dashboard.html works without dash.",
            "Spurion likelihood scores are triage priors, not definitive artifact claims.",
        ],
        provenance=[ProvenanceRecord(source_type="data_root", source_id=str(data_root), method="build_experiment_dashboard")],
    )
    report = spec.model_dump(mode="json")
    report["workflow_name"] = "experiment-dashboard"
    report["status"] = "ok"
    report["scan_count"] = len(records)
    report["inelastic_scan_count"] = sum(1 for item in records if item.measurement_kind == "inelastic")
    report["spurion_likelihood_candidates"] = [item.model_dump(mode="json") for item in spurion_candidates]
    report["recommended_diagnostic_scans"] = diagnostic_scans
    report["experiment_config"] = _reportable_experiment_config(arguments)
    report["dashboard_scope"] = {
        "scope": "single_experiment_or_data_root",
        "generalization_status": "adapter_and_config_driven",
        "overlay_policy": "experiment_specific_overlays_only; no global material overlays are assumed",
        "refresh_policy": "rerun workflow or generated refresh_dashboard.sh; dashboard artifacts reload local JSON only",
    }
    report["overlay_summary"] = _overlay_summary(overlays)
    report["spurion_ranking_diagnostics"] = _spurion_ranking_diagnostics(
        spurion_candidates,
        review_label_store=arguments.get("review_label_store") or arguments.get("review_outcomes"),
    )
    report["summary"] = _dashboard_summary(records, spurion_candidates)
    report["refresh_command"] = _refresh_command(arguments)
    artifact_paths = _write_dashboard_artifacts(output_dir, report)
    report["generated_artifacts"] = artifact_paths
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_dashboard_markdown,
        json_filename="experiment_dashboard_report.json",
        markdown_filename="experiment_dashboard_report.md",
    )
    return report


def survey_inelastic_spurion_evidence(arguments: dict[str, Any]) -> dict[str, Any]:
    """Survey post-2010 inelastic scan files for spurion-like evidence."""

    payload = dict(arguments)
    payload["title"] = payload.get("title") or "Inelastic Spurion Evidence Survey"
    payload["post_year"] = payload.get("post_year", 2010)
    payload["max_files"] = payload.get("max_files", 1000)
    dashboard = build_experiment_dashboard(payload)
    if dashboard.get("status") != "ok":
        return dashboard
    candidates = [
        item for item in dashboard.get("spurion_likelihood_candidates", [])
        if item.get("candidate_type") in {"narrow_inelastic_feature", "inelastic_environment_risk"}
    ]
    report = {
        "workflow_name": "survey-inelastic-spurion-evidence",
        "status": "ok",
        "data_root": dashboard.get("data_root"),
        "scan_count": dashboard.get("scan_count"),
        "inelastic_scan_count": dashboard.get("inelastic_scan_count"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "generated_artifacts": dashboard.get("generated_artifacts", {}),
        "recommended_diagnostic_scans": dashboard.get("recommended_diagnostic_scans", []),
        "ranking_diagnostics": dashboard.get("spurion_ranking_diagnostics", {}),
        "safe_conclusion": (
            "The survey identifies inelastic scans with narrow, high-prominence features and sample-environment/filter clues. "
            "These are spurion triage candidates requiring human review, not confirmed spurions."
        ),
        "limitations": [
            "No full TAS resolution calculation is performed; narrowness is judged relative to neighboring scan features.",
            "Manual collimation and PG/velocity-selector state may be missing or stale in scan titles.",
            "High-energy neutron population is treated as a prior penalty because the source spectrum peaks around thermal energies.",
            "Spurion2 follow-up is attempted only when explicit lattice/orientation/configuration and candidate Q/E are available.",
            "Resolution-width proxies use supplied manual collimations when available; 40/40 is only a labeled ballpark prior.",
        ],
        "provenance": dashboard.get("provenance", []),
    }
    report["artifacts"] = write_report_bundle(
        Path(arguments["output_dir"]).expanduser(),
        report,
        markdown_renderer=_spurion_survey_markdown,
        json_filename="inelastic_spurion_survey.json",
        markdown_filename="inelastic_spurion_survey.md",
    )
    return report


def experiment_review(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one experiment-scoped review bundle across dashboard, triage, and next checks.

    This is the high-level experimenter-facing entry point. It composes existing
    read-only workflows and preserves each sub-report rather than collapsing the
    evidence into a single authoritative answer.
    """

    arguments = _with_experiment_config(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments["data_root"]).expanduser() if arguments.get("data_root") else None
    if data_root is None:
        return _missing(output_dir, "experiment-review", ["data_root"], "A data root is required.")
    if not data_root.exists():
        return _missing(output_dir, "experiment-review", ["data_root"], f"Data root does not exist: {data_root}")

    max_files = int(arguments.get("max_files", 1000))
    post_year = int(arguments.get("post_year", 0) or 0)
    top_k = int(arguments.get("top_k", 8))
    shared = {
        "data_root": str(data_root),
        "max_files": max_files,
        "post_year": post_year,
        "patterns": arguments.get("patterns"),
        "experiment_config": arguments.get("experiment_config"),
        "precedent_index": arguments.get("precedent_index"),
        "review_label_store": arguments.get("review_label_store") or arguments.get("review_outcomes"),
        "spurion_top_k": int(arguments.get("spurion_top_k", top_k)),
    }

    dashboard = build_experiment_dashboard({**shared, "output_dir": str(output_dir / "dashboard"), "title": arguments.get("title") or f"Experiment Review: {data_root.name}"})
    overnight = overnight_scan_summary({**shared, "output_dir": str(output_dir / "overnight_summary"), "hours": arguments.get("hours"), "since": arguments.get("since")})
    if arguments.get("skip_spurion"):
        spurions = {
            "workflow_name": "survey-inelastic-spurion-evidence",
            "status": "skipped",
            "candidate_count": 0,
            "candidates": [],
            "summary": "Spurion triage was skipped by request.",
            "warnings": ["No spurion survey was run for this experiment review."],
            "artifacts": {},
            "generated_artifacts": {},
        }
    else:
        spurions = survey_inelastic_spurion_evidence({**shared, "output_dir": str(output_dir / "spurion_survey"), "post_year": max(post_year, 2010), "ranking_mode": arguments.get("ranking_mode", "diversified")})
    order_triage = order_parameter_triage({**shared, "output_dir": str(output_dir / "order_parameter_triage"), "min_scans": int(arguments.get("min_scans", 4)), "top_k": top_k})
    order_bundle = _experiment_review_order_parameter_bundle(order_triage, output_dir=output_dir, arguments=arguments)
    corpus_context = _experiment_review_corpus_context(arguments, output_dir=output_dir)
    review_outcome_template = _write_review_outcome_template(output_dir, dashboard, spurions, order_triage, order_bundle)

    workflow_results = {
        "dashboard": _workflow_result_summary(dashboard),
        "overnight_summary": _workflow_result_summary(overnight),
        "spurion_survey": _workflow_result_summary(spurions),
        "order_parameter_triage": _workflow_result_summary(order_triage),
        "order_parameter_followup": _workflow_result_summary(order_bundle),
        "corpus_context": _workflow_result_summary(corpus_context),
    }
    report = {
        "workflow_name": "experiment-review",
        "status": "ok" if any(item.get("status") == "ok" for item in workflow_results.values()) else "missing_input",
        "data_root": str(data_root),
        "workflow_results": workflow_results,
        "scan_count": dashboard.get("scan_count"),
        "series_count": overnight.get("series_count"),
        "order_parameter_candidate_count": order_triage.get("candidate_count"),
        "spurion_candidate_count": spurions.get("candidate_count"),
        "probability_model": _experiment_review_probability_model(),
        "calibration_diagnostics": _experiment_review_calibration_diagnostics(dashboard, overnight, spurions, order_triage, order_bundle, corpus_context, review_label_store=arguments.get("review_label_store") or arguments.get("review_outcomes")),
        "label_informed_priors": _experiment_review_label_informed_priors(arguments.get("review_label_store") or arguments.get("review_outcomes")),
        "ranked_next_steps": _experiment_review_next_steps(dashboard, overnight, spurions, order_triage, order_bundle, corpus_context),
        "fit_handoff": order_bundle.get("fit_handoff", {}),
        "generated_artifacts": {
            "dashboard_html": (dashboard.get("generated_artifacts") or {}).get("dashboard_html"),
            "overnight_markdown": (overnight.get("artifacts") or {}).get("markdown"),
            "spurion_markdown": (spurions.get("artifacts") or {}).get("markdown"),
            "order_parameter_markdown": (order_triage.get("artifacts") or {}).get("markdown"),
            "order_parameter_gallery": ((order_bundle.get("plots") or {}).get("generated_artifacts") or {}).get("gallery_html"),
            "corpus_context_markdown": (corpus_context.get("artifacts") or {}).get("markdown"),
            "review_outcome_template_jsonl": review_outcome_template,
            "review_outcomes_sqlite": str(output_dir / "review_outcomes.sqlite"),
        },
        "safe_conclusion": (
            "Generated a per-experiment review bundle with dashboard, scan-family summary, spurion triage, "
            "order-parameter candidates, and next-check recommendations. All rankings are probability-style "
            "decision support and require human review before operational or scientific claims."
        ),
        "warnings": [
            "This workflow is experiment-scoped; it does not scan or summarize every experiment in the catalog.",
            "Fit handoff commands are suggested next analyses, not automatically approved fitting conclusions.",
            "Spurion and order-parameter probabilities are calibrated triage priors, not certainties.",
            "Corpus context is optional and remains source-support evidence, not instructions.",
        ],
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=str(data_root), method="experiment_review").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_experiment_review_markdown, json_filename="experiment_review.json", markdown_filename="experiment_review.md")
    return report


def overnight_scan_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Summarize recent scan families into an instrument-scientist morning report."""

    arguments = _with_experiment_config(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments["data_root"]).expanduser() if arguments.get("data_root") else None
    if data_root is None:
        return _missing(output_dir, "overnight-scan-summary", ["data_root"], "A data root is required.")
    if not data_root.exists():
        return _missing(output_dir, "overnight-scan-summary", ["data_root"], f"Data root does not exist: {data_root}")

    max_files = int(arguments.get("max_files", 1000))
    post_year = int(arguments.get("post_year", 0) or 0)
    patterns = tuple(arguments.get("patterns") or SCAN_PATTERNS)
    paths = _find_scan_paths(data_root, patterns=patterns, max_files=max_files, post_year=post_year)
    records = [_parse_scan_file(path, max_plot_points=int(arguments.get("max_plot_points", 300))) for path in paths]
    records, time_filter = _filter_records_for_summary(records, arguments)
    series = _overnight_series(records)
    alignment_context = _load_alignment_plot_context(arguments.get("alignment_state_json"))
    series_plots = _write_overnight_series_plots(records, output_dir, alignment_context=alignment_context)
    for item in series:
        entries = series_plots.get(str(item.get("series_id")))
        if entries:
            item["plots"] = entries
            item["plot_png"] = entries[0]["png"]
            if item.get("family") in _ALIGNMENT_PLOT_FAMILIES and alignment_context.get("software_changes"):
                item["software_change_notes"] = list(alignment_context["software_changes"])[:12]
    scan_corpus_context = _overnight_scan_corpus_context(series, arguments)
    series = _attach_scan_corpus_context_to_series(series, scan_corpus_context)
    suggested_reports = _calibrate_overnight_report_recommendations(_overnight_report_recommendations(series, records), scan_corpus_context)
    next_actions = _overnight_next_actions(series, suggested_reports, scan_corpus_context)
    discovery = {"data_root": str(data_root), "patterns": list(patterns), "file_count": len(paths)}

    report = {
        "workflow_name": "overnight-scan-summary",
        "discovery": discovery,
        "status": "ok",
        "data_root": str(data_root),
        "scan_count": len(records),
        "time_filter": time_filter,
        "series_count": len(series),
        "series": series,
        "scan_corpus_context": scan_corpus_context,
        "suggested_reports": suggested_reports,
        "next_actions": next_actions,
        "privacy_boundary": {
            "user_facing_policy": "Summaries may name current local scans, but should not expose unrelated precedent experiments or private notes.",
            "generated_artifact_policy": "Keep real experiment summaries in ignored local output directories unless explicitly sanitized.",
        },
        "safe_conclusion": _overnight_safe_conclusion(series, suggested_reports, discovery),
        "warnings": (
            []
            if paths
            else [
                f"No files matched {', '.join(patterns)} under {data_root} (searched recursively). "
                "Empty result is a discovery failure, not an empty night."
            ]
        ),
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=str(data_root), method="overnight_scan_summary").model_dump(mode="json")],
    }
    report["generated_artifacts"] = _write_overnight_artifacts(output_dir, report)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_overnight_markdown,
        json_filename="overnight_scan_summary.json",
        markdown_filename="overnight_scan_summary.md",
        html_filename="overnight_scan_summary.html",
    )
    return report


def draft_overnight_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn an overnight scan-family summary into a local morning report draft."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_or_build_overnight_summary(arguments, output_dir)
    if summary.get("status") != "ok":
        report = {
            "workflow_name": "draft-overnight-report",
            "status": summary.get("status", "missing_input"),
            "summary_status": summary.get("status"),
            "missing_fields": summary.get("missing_fields", []),
            "warnings": summary.get("warnings", []),
            "safe_conclusion": "No morning report draft was produced because the overnight summary could not be built.",
            "provenance": [
                ProvenanceRecord(source_type="overnight_summary", source_id=str(arguments.get("summary_json") or arguments.get("data_root") or "missing"), method="draft_overnight_report").model_dump(mode="json")
            ],
        }
        report["artifacts"] = write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_draft_overnight_markdown,
            json_filename="morning_report_draft.json",
            markdown_filename="morning_report_draft.md",
            html_filename="morning_report_draft.html",
        )
        return report

    top_report = (summary.get("suggested_reports") or [{}])[0]
    sections = _draft_morning_sections(summary)
    followup_commands = _draft_followup_commands(summary, output_dir)
    summary_dir = Path(str(arguments["summary_json"])).expanduser().parent if arguments.get("summary_json") else output_dir / "overnight_scan_summary"
    family_plots = []
    for item in summary.get("series") or []:
        entries = item.get("plots") or ([{"label": item.get("series_id"), "png": item["plot_png"]}] if item.get("plot_png") else [])
        for entry in entries:
            relative = os.path.relpath((summary_dir / str(entry["png"])).resolve(), output_dir)
            family_plots.append(
                {
                    "series_id": item.get("series_id"),
                    "family": item.get("family"),
                    "scan_count": item.get("scan_count"),
                    "label": entry.get("label"),
                    "plot": relative.replace(os.sep, "/"),
                    "software_change_notes": item.get("software_change_notes") if entry is entries[-1] else None,
                }
            )
    report = {
        "workflow_name": "draft-overnight-report",
        "status": "ok",
        "family_plots": family_plots,
        "source_summary": {
            "data_root": summary.get("data_root"),
            "scan_count": summary.get("scan_count"),
            "series_count": summary.get("series_count"),
            "summary_artifacts": summary.get("artifacts", {}),
        },
        "draft_type": top_report.get("report_type") or "generic_overnight_scan_log",
        "draft_probability": top_report.get("probability"),
        "probability_diagnostics": top_report.get("probability_diagnostics", {}),
        "sections": sections,
        "followup_commands": followup_commands,
        "next_actions": summary.get("next_actions", []),
        "privacy_boundary": {
            "user_facing_policy": "The draft may discuss current experiment scan families, but should not expose unrelated precedent experiment identities.",
            "artifact_policy": "Keep real morning drafts in ignored local output directories unless explicitly sanitized.",
        },
        "safe_conclusion": (
            f"A `{top_report.get('report_type') or 'generic_overnight_scan_log'}` draft was prepared from "
            f"{summary.get('scan_count', 0)} parsed scan(s)."
        ),
        "warnings": [
            "This draft is generated from parsed headers and simple scan-family heuristics.",
            "Peak positions, fitted widths, backgrounds, and physical conclusions require separate reduction and human review.",
            "Suggested follow-up commands are local report/diagnostic commands, not approved instrument operations.",
        ],
        "provenance": [
            ProvenanceRecord(
                source_type="overnight_scan_summary",
                source_id=str(summary.get("data_root") or arguments.get("summary_json") or "summary"),
                method="draft_overnight_report",
                confidence=float(top_report.get("probability") or 0.0),
            ).model_dump(mode="json")
        ],
    }
    report["generated_artifacts"] = _write_morning_draft_artifacts(output_dir, report)
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_draft_overnight_markdown,
        json_filename="morning_report_draft.json",
        markdown_filename="morning_report_draft.md",
        html_filename="morning_report_draft.html",
    )
    return report


def inelastic_series_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Draft a local report for one inelastic scan-family series."""

    arguments = _with_experiment_config(arguments)
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments["data_root"]).expanduser() if arguments.get("data_root") else None
    if data_root is None:
        return _missing(output_dir, "inelastic-series-report", ["data_root"], "A data root is required to plot scan points.")
    if not data_root.exists():
        return _missing(output_dir, "inelastic-series-report", ["data_root"], f"Data root does not exist: {data_root}")

    records, time_filter = _parsed_summary_records(arguments, data_root, max_plot_points=int(arguments.get("max_plot_points", 2000)))
    groups = _overnight_record_groups(records)
    series_summaries = [_series_summary(key, items) for key, items in groups.items()]
    series_summaries.sort(key=lambda item: (item["scan_count"], item["point_count"], item["family"]), reverse=True)
    selected_summary = _select_inelastic_series(series_summaries, arguments.get("series_id"))
    if selected_summary is None:
        report = {
            "workflow_name": "inelastic-series-report",
            "status": "missing_input",
            "data_root": str(data_root),
            "missing_fields": ["inelastic_series"],
            "series_count": len(series_summaries),
            "available_series": series_summaries,
            "safe_conclusion": "No inelastic scan-family series was found for the requested files.",
            "warnings": ["Check data_root, file patterns, time filters, and whether the scans vary E or Q at nonzero energy transfer."],
            "provenance": [ProvenanceRecord(source_type="data_root", source_id=str(data_root), method="inelastic_series_report").model_dump(mode="json")],
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_inelastic_series_markdown, json_filename="inelastic_series_report.json", markdown_filename="inelastic_series_report.md")
        return report

    selected_records = groups[_series_key_from_summary(selected_summary)]
    scan_table, point_rows, peak_rows = _inelastic_report_tables(selected_records)
    generated = _write_inelastic_report_artifacts(output_dir, scan_table, point_rows, peak_rows, selected_records)
    report = {
        "workflow_name": "inelastic-series-report",
        "status": "ok",
        "data_root": str(data_root),
        "time_filter": time_filter,
        "selected_series": selected_summary,
        "scan_count": len(selected_records),
        "point_count": sum(record.row_count for record in selected_records),
        "scan_table": scan_table,
        "tentative_peak_or_narrow_feature_count": len(peak_rows),
        "tentative_peak_or_narrow_features": peak_rows,
        "generated_artifacts": generated,
        "safe_conclusion": (
            f"Prepared an inelastic series report for `{selected_summary.get('series_id')}` with {len(selected_records)} scan(s). "
            "This is a raw-overlay and metadata report; it does not fit a dispersion, subtract background, or confirm peak assignments."
        ),
        "warnings": [
            "Overlay curves use parser-normalized detector/monitor values where monitor is available.",
            "Tentative peak rows are narrow-feature flags from a robust local diagnostic, not fitted excitations.",
            "Background, resolution, collimations, spurion checks, and repeat scans must be reviewed before physical interpretation.",
        ],
        "provenance": [
            ProvenanceRecord(
                source_type="scan_family",
                source_id=str(selected_summary.get("series_id")),
                method="inelastic_series_report",
                confidence=0.65 if len(selected_records) >= 3 else 0.45,
            ).model_dump(mode="json")
        ],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_inelastic_series_markdown, json_filename="inelastic_series_report.json", markdown_filename="inelastic_series_report.md")
    return report


def order_parameter_triage(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search hydrated scans for temperature-dependent peak/order-parameter candidates."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(arguments["data_root"]).expanduser() if arguments.get("data_root") else None
    if data_root is None:
        return _missing(output_dir, "order-parameter-triage", ["data_root"], "A data root is required.")
    if not data_root.exists():
        return _missing(output_dir, "order-parameter-triage", ["data_root"], f"Data root does not exist: {data_root}")

    records, time_filter = _parsed_summary_records(arguments, data_root, max_plot_points=int(arguments.get("max_plot_points", 2000)))
    synthetic = _aggregate_single_point_records(records)
    all_records = records + synthetic
    eligible = _order_parameter_eligible_records(all_records)
    regular = [r for r in eligible if not (r.metadata or {}).get("cross_file_series")]
    cross_file = [r for r in eligible if (r.metadata or {}).get("cross_file_series")]
    groups = _order_parameter_groups(regular, min_scans=int(arguments.get("min_scans", 4)))
    candidates = [_order_parameter_candidate_summary(group_id, items) for group_id, items in groups.items()]
    for cf in cross_file:
        candidates.append(_cross_file_order_parameter_candidate(cf))
    candidates = [item for item in candidates if item["score"] > 0]
    label_priors = label_informed_priors(
        arguments.get("review_label_store") or arguments.get("review_outcomes"),
        object_type="order_parameter_candidate",
        source_workflow="order-parameter-triage",
    )
    _apply_order_parameter_label_calibration(candidates, label_priors)
    candidates.sort(key=lambda item: (item["score"], item["scan_count"], item["temperature_range"].get("width", 0)), reverse=True)
    top_k = int(arguments.get("top_k", 20))
    candidates = candidates[:top_k]
    dag_runs = _attach_order_parameter_dag_runs(candidates, output_dir=output_dir, workflow_version="order_parameter_triage_v1")
    per_scan_rows = [row for candidate in candidates for row in candidate["per_scan_peak_summary"]]
    generated = _write_order_parameter_artifacts(output_dir, candidates, per_scan_rows)
    if dag_runs:
        dag_manifest = output_dir / "order_parameter_dag_runs.json"
        dag_manifest.write_text(json.dumps({"workflow_name": "order-parameter-triage", "dag_runs": dag_runs}, indent=2, sort_keys=True), encoding="utf-8")
        generated["dag_runs_json"] = str(dag_manifest)
    review_label_diagnostics = review_label_calibration_diagnostics(
        arguments.get("review_label_store") or arguments.get("review_outcomes"),
        object_type="order_parameter_candidate",
        source_workflow="order-parameter-triage",
    )
    report = {
        "workflow_name": "order-parameter-triage",
        "status": "ok" if candidates else "missing_input",
        "data_root": str(data_root),
        "time_filter": time_filter,
        "eligible_scan_count": len(records),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "reduction_dag": {
            "status": "executed" if dag_runs else "not_executed_no_candidates",
            "template_id": "tas.order_parameter_temperature_series.v0",
            "dag_run_count": len(dag_runs),
            "manifest": generated.get("dag_runs_json"),
        },
        "review_label_diagnostics": review_label_diagnostics,
        "label_informed_priors": label_priors,
        "label_calibration_summary": _label_calibration_summary(candidates, score_key="score"),
        "generated_artifacts": generated,
        "fit_backend_status": {
            "bumps": _module_status("bumps"),
            "lmfit": _module_status("lmfit"),
            "scipy": _module_status("scipy"),
            "note": "This workflow currently performs triage/model-classification only. Bumps DREAM plus LM refinement is planned behind a reviewed fitting boundary.",
        },
        "safe_conclusion": (
            f"Found {len(candidates)} candidate temperature-dependent peak family/families. "
            "These are candidates for order-parameter or structural-splitting analysis, not final fits."
            if candidates
            else "No order-parameter candidate families were found with the requested filters."
        ),
        "warnings": [
            "Peak/splitting labels are heuristic model-selection priors from raw scan profiles.",
            "A structural splitting candidate should be fit with constrained peak models before interpreting a transition.",
            "A single-peak order-parameter candidate still needs background, nuisance-peak, and repeatability checks.",
            "Do not run spin-wave fits from this workflow.",
        ],
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=str(data_root), method="order_parameter_triage").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_order_parameter_markdown, json_filename="order_parameter_triage.json", markdown_filename="order_parameter_triage.md")
    return report


def fit_order_parameter(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fit a triaged order-parameter candidate with a small Bumps model."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    triage_json = Path(arguments["triage_json"]).expanduser() if arguments.get("triage_json") else None
    if triage_json is None:
        return _missing(output_dir, "fit-order-parameter", ["triage_json"], "A triage JSON report is required.")
    if not triage_json.exists():
        return _missing(output_dir, "fit-order-parameter", ["triage_json"], f"Triage report does not exist: {triage_json}")

    triage = json.loads(triage_json.read_text(encoding="utf-8"))
    candidate = _select_order_parameter_candidate(triage, arguments.get("candidate_id"))
    if not candidate:
        return _missing(output_dir, "fit-order-parameter", ["candidate_id"], "No matching candidate was found in the triage report.")

    backend = str(arguments.get("fit_backend") or arguments.get("backend") or "core_preview")
    review_gate = _order_parameter_fit_review_gate(candidate, arguments, backend=backend)
    if backend == "panweb_global":
        return _fit_order_parameter_with_panweb(candidate, arguments, triage_json=triage_json, output_dir=output_dir, review_gate=review_gate)

    try:
        import numpy as np
        from bumps.fitters import fit as bumps_fit
        from bumps.names import Curve, FitProblem, Parameter
    except Exception as exc:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "unavailable",
            "missing_fields": ["bumps"],
            "error": f"Bumps is unavailable: {type(exc).__name__}: {exc}",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    observable = str(arguments.get("observable") or "auto")
    observations = _order_parameter_observations(candidate, observable=observable)
    if len(observations) < 4:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "missing_input",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["at_least_four_temperature_observations"],
            "observations": observations,
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    temperatures = np.array([row["temperature"] for row in observations], dtype=float)
    values = np.array([row["value"] for row in observations], dtype=float)
    uncertainties = np.array([max(row["uncertainty"], 1e-6) for row in observations], dtype=float)
    chosen_observable = observations[0]["observable"]
    normal_transition = bool(arguments.get("normal_transition", True))
    power = 1.0 if chosen_observable == "splitting" else 2.0
    model_func = _bumps_order_parameter_model_normal if normal_transition else _bumps_order_parameter_model_inverse

    y_min = float(values.min())
    y_max = float(values.max())
    y_span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.05)
    t_min = float(temperatures.min())
    t_max = float(temperatures.max())
    t_span = max(t_max - t_min, 1.0)
    initial_tc = float(arguments.get("initial_tc") or (t_min + 0.7 * t_span))
    tc = Parameter(initial_tc, name="Tc").range(max(0.1, t_min - 0.5 * t_span), t_max + 0.5 * t_span)
    beta = Parameter(float(arguments.get("initial_beta") or 0.5), name="beta").range(0.05, 1.5)
    amplitude = Parameter(max(y_span, 1e-6), name="amplitude").range(0.0, max(y_span * 4.0, y_max * 4.0, 1.0))
    background = Parameter(max(0.0, y_min), name="background").range(min(0.0, y_min - y_span), y_max + y_span)

    curve = Curve(model_func, temperatures, values, uncertainties, background=background, amplitude=amplitude, Tc=tc, beta=beta, power=power)
    problem = FitProblem(curve, name=f"order_parameter_{candidate.get('candidate_id')}")
    method = str(arguments.get("fit_method") or "dream")
    parallel = int(arguments.get("parallel", 1))
    fit_options = {
        "steps": int(arguments.get("fit_steps", 800)),
        "burn": int(arguments.get("fit_burn", 200)),
        "verbose": bool(arguments.get("verbose", False)),
        "parallel": parallel,
    }
    import warnings as warnings_module

    with warnings_module.catch_warnings(record=True) as captured_warnings:
        warnings_module.simplefilter("always")
        result = bumps_fit(problem, method=method, **fit_options)
    parameter_values = {label: float(value) for label, value in zip(problem.labels(), problem.getp(), strict=False)}
    fit_rows = _order_parameter_fit_rows(temperatures, values, uncertainties, parameter_values, power=power, normal_transition=normal_transition)
    chisq = float(problem.chisq())
    dof = int(getattr(result, "dof", max(0, len(observations) - len(parameter_values))))
    fit_quality = _order_parameter_fit_quality(
        fit_rows,
        chisq=chisq,
        dof=dof,
        fit_success=bool(getattr(result, "success", False)),
        fit_message=str(getattr(result, "message", "")),
        captured_warnings=[str(item.message) for item in captured_warnings],
    )
    fit_artifacts = _write_order_parameter_fit_artifacts(output_dir, observations, fit_rows)
    dag_branch = _order_parameter_fit_dag_branch(candidate, triage, fit_quality, parameter_values, arguments, output_dir=output_dir)
    if dag_branch.get("artifact"):
        fit_artifacts["dag_model_branch_json"] = dag_branch["artifact"]
    report = {
        "workflow_name": "fit-order-parameter",
        "status": "ok",
        "triage_json": str(triage_json),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_classification": candidate.get("classification"),
        "candidate_summary": _order_parameter_fit_candidate_summary(candidate),
        "observable": chosen_observable,
        "transition_direction": "low_temperature_ordered" if normal_transition else "high_temperature_ordered",
        "model": "background + amplitude * m(T)^power",
        "model_source": "tas_copilot_core_preview",
        "model_selection_plan": candidate.get("model_selection_plan") or [],
        "model_comparison_status": "not_performed_single_model_fit",
        "reduction_dag": dag_branch,
        "background_model": str(arguments.get("background_model") or "fitted_constant"),
        "fit_acceptance": review_gate,
        "order_parameter_function": "m(T)=(1-T/Tc)^beta below Tc" if normal_transition else "m(T)=(T/Tc-1)^beta above Tc",
        "power": power,
        "fit_backend": "bumps",
        "fit_method": method,
        "parallel": parallel,
        "fit_options": fit_options,
        "fit_success": bool(getattr(result, "success", False)),
        "fit_message": str(getattr(result, "message", "")),
        "chisq": chisq,
        "dof": dof,
        "fit_quality": fit_quality,
        "parameters": parameter_values,
        "observations": observations,
        "fit_rows": fit_rows,
        "generated_artifacts": fit_artifacts,
        "warnings": [
            "This is a first-pass global order-parameter fit to triage-derived observables, not a final publication model.",
            "For splitting cases, prefer the pan-web global splitting backend when pan-web is available.",
            "Peak-shape model selection and background checks must be reviewed before interpreting Tc or beta.",
            "For structural splitting, this fit uses extracted peak separation only when at least two peaks were detected; single-peak scans are treated as zero splitting.",
            "Fit status remains candidate_fit_review_required unless a human review record accepts the model, background, and residuals.",
        ],
        "provenance": [ProvenanceRecord(source_type="triage_report", source_id=str(triage_json), method="fit_order_parameter").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
    return report


def dag_run_replay(arguments: dict[str, Any]) -> dict[str, Any]:
    """Replay an order-parameter DAG run manifest and report cache-key changes."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(arguments["dag_run_json"]).expanduser() if arguments.get("dag_run_json") else None
    if manifest_path is None:
        return _missing(output_dir, "dag-run-replay", ["dag_run_json"], "A DAG run manifest JSON is required.")
    if not manifest_path.exists():
        return _missing(output_dir, "dag-run-replay", ["dag_run_json"], f"DAG run manifest does not exist: {manifest_path}")
    requested_run_id = arguments.get("dag_run_id")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "reduction_dag" in manifest and "replay" in manifest:
        from .replay_reduction import replay_numerical_reduction

        if requested_run_id and manifest["reduction_dag"].get("run_id") != requested_run_id:
            return _missing(output_dir, "dag-run-replay", ["dag_run_id"], f"No DAG run matched {requested_run_id!r}.")
        return replay_numerical_reduction(manifest, arguments)
    entries = list(manifest.get("dag_runs") or [])
    entry = next((item for item in entries if not requested_run_id or item.get("dag_run_id") == requested_run_id or (item.get("result") or {}).get("run_id") == requested_run_id), None)
    if not entry:
        return _missing(output_dir, "dag-run-replay", ["dag_run_id"], f"No DAG run matched {requested_run_id!r}.")

    correction_overrides = arguments.get("correction_overrides") or {}
    corrections = _order_parameter_corrections(correction_overrides=correction_overrides)
    template = order_parameter_temperature_series_template()
    software_versions = _order_parameter_tool_versions(workflow_version=str(arguments.get("workflow_version") or entry.get("workflow_version") or "order_parameter_triage_v1"))
    replay = replay_reduction_dag_run(
        dict(entry.get("result") or {}),
        template=template,
        inputs=dict(entry.get("inputs") or {}),
        corrections=corrections,
        software_versions=software_versions,
    )
    report = {
        "workflow_name": "dag-run-replay",
        "status": replay["status"],
        "dag_run_id": requested_run_id or entry.get("dag_run_id"),
        "candidate_id": entry.get("candidate_id"),
        "manifest": str(manifest_path),
        "template_id": template.template_id,
        "cache_replay": replay,
        "correction_status_snapshot": _correction_status_snapshot(corrections),
        "tool_versions": software_versions,
        "safe_conclusion": (
            f"Replayed DAG run and found {replay['changed_count']} changed node cache key(s)."
            if replay["changed_count"]
            else "Replayed DAG run; all node cache keys were reusable."
        ),
        "provenance": [ProvenanceRecord(source_type="reduction_dag_manifest", source_id=str(manifest_path), method="dag_run_replay").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_dag_run_replay_markdown, json_filename="dag_run_replay.json", markdown_filename="dag_run_replay.md")
    return report


def _attach_order_parameter_dag_runs(candidates: list[dict[str, Any]], *, output_dir: Path, workflow_version: str) -> list[dict[str, Any]]:
    template = order_parameter_temperature_series_template()
    corrections = _order_parameter_corrections()
    software_versions = _order_parameter_tool_versions(workflow_version=workflow_version)
    dag_runs: list[dict[str, Any]] = []
    for candidate in candidates:
        inputs = {"scan_series": _order_parameter_scan_series_input(candidate)}
        result = execute_reduction_dag(template, inputs=inputs, corrections=corrections, software_versions=software_versions)
        target_cache_key = result.cache_keys.get(result.target_node_id or "", "")
        snapshot = _correction_status_snapshot(corrections)
        dag_summary = {
            "dag_run_id": result.run_id,
            "template_id": result.template_id,
            "status": result.status,
            "target_node_id": result.target_node_id,
            "cache_key": target_cache_key,
            "cache_keys": result.cache_keys,
            "correction_status_snapshot": snapshot,
            "workflow_version": workflow_version,
            "tool_versions": software_versions,
            "manifest_role": "order_parameter_label_replay_input",
        }
        candidate["reduction_dag"] = dag_summary
        dag_runs.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "dag_run_id": result.run_id,
                "workflow_version": workflow_version,
                "inputs": inputs,
                "corrections": [correction.model_dump(mode="json") for correction in corrections],
                "correction_status_snapshot": snapshot,
                "tool_versions": software_versions,
                "result": result.model_dump(mode="json"),
                "review_label_use": "Labels should store dag_run_id/cache_key so they can be re-evaluated when corrections or software change.",
            }
        )
    return dag_runs


def _order_parameter_scan_series_input(candidate: dict[str, Any]) -> dict[str, Any]:
    per_scan = list(candidate.get("per_scan_peak_summary") or [])
    scan_ids = [str(row.get("scan_id") or row.get("path") or f"scan-{index}") for index, row in enumerate(per_scan)]
    temperatures = [float(row["temperature_mean"]) for row in per_scan if row.get("temperature_mean") is not None]
    return {
        "source_object_type": "order_parameter_candidate",
        "source_object_id": candidate.get("candidate_id"),
        "scan_ids": scan_ids,
        "temperature_values": temperatures,
        "scan_paths": [row.get("path") for row in per_scan if row.get("path")],
        "observable_summary": {
            "classification": candidate.get("classification"),
            "area_ratio": candidate.get("area_ratio"),
            "center_span": candidate.get("center_span"),
            "scan_count": candidate.get("scan_count"),
        },
    }


def _order_parameter_corrections(*, correction_overrides: dict[str, Any] | None = None) -> list[ReductionCorrectionRef]:
    overrides = correction_overrides or {}
    monitor = dict(overrides.get("monitor_normalization") or {})
    background = dict(overrides.get("background_model") or {})
    return [
        ReductionCorrectionRef(
            correction_id="monitor_normalization",
            correction_type="monitor_normalization",
            status=str(monitor.get("status") or "placeholder"),
            version=str(monitor.get("version") or "monitor_normalization_preview_v0"),
            stale_ok_for_preview=True,
            parameters={
                "method": monitor.get("method") or "use_monitor_column_when_available",
                "review_status": monitor.get("review_status") or "not_human_reviewed",
            },
            provenance=[ProvenanceRecord(source_type="workflow_default", source_id="order_parameter_triage", method="monitor_normalization_correction_ref")],
        ),
        ReductionCorrectionRef(
            correction_id="background_model",
            correction_type="background_model",
            status=str(background.get("status") or "placeholder"),
            version=str(background.get("version") or "background_model_preview_v0"),
            stale_ok_for_preview=True,
            parameters={
                "method": background.get("method") or "no_subtraction_preview",
                "review_status": background.get("review_status") or "not_human_reviewed",
            },
            provenance=[ProvenanceRecord(source_type="workflow_default", source_id="order_parameter_triage", method="background_model_correction_ref")],
        ),
    ]


def _correction_status_snapshot(corrections: list[ReductionCorrectionRef]) -> dict[str, Any]:
    return {
        correction.correction_id: {
            "correction_type": correction.correction_type,
            "status": correction.status,
            "version": correction.version,
            "stale_ok_for_preview": correction.stale_ok_for_preview,
            "parameters": correction.parameters,
        }
        for correction in corrections
    }


def _order_parameter_tool_versions(*, workflow_version: str) -> dict[str, str]:
    return {
        "workflow_version": workflow_version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "tas_copilot_core": "local_worktree",
        "bumps": _module_status("bumps"),
        "scipy": _module_status("scipy"),
        "lmfit": _module_status("lmfit"),
    }


def _order_parameter_fit_dag_branch(
    candidate: dict[str, Any],
    triage: dict[str, Any],
    fit_quality: dict[str, Any],
    parameter_values: dict[str, float],
    arguments: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    model_branch = "split_peak_power_law" if candidate.get("classification") == "structural_splitting_candidate" else "single_peak_power_law"
    bic_tag = _order_parameter_bic_tag(fit_quality)
    template = order_parameter_fit_branch_template(model_branch=model_branch, bic_tag=bic_tag)
    corrections = _order_parameter_corrections(
        correction_overrides={
            "background_model": {
                "status": "proposed",
                "version": str(arguments.get("background_model") or "fitted_constant"),
                "method": str(arguments.get("background_model") or "fitted_constant"),
            }
        }
    )
    inputs = {
        "scan_series": _order_parameter_scan_series_input(candidate),
        "fit_evidence": {
            "model_branch": model_branch,
            "bic_tag": bic_tag,
            "model_comparison_status": "single_model_fit_no_alternate_branch_comparison",
            "candidate_id": candidate.get("candidate_id"),
            "fit_quality": fit_quality,
            "parameters": parameter_values,
            "source_triage_dag_run_id": ((candidate.get("reduction_dag") or {}).get("dag_run_id")),
            "source_triage_json": (triage.get("artifacts") or {}).get("json"),
        },
    }
    software_versions = _order_parameter_tool_versions(workflow_version="fit_order_parameter_v1")
    result = execute_reduction_dag(template, inputs=inputs, corrections=corrections, software_versions=software_versions)
    artifact = output_dir / "order_parameter_fit_dag_branch.json"
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "dag_run_id": result.run_id,
        "template_id": result.template_id,
        "model_branch": model_branch,
        "bic_tag": bic_tag,
        "inputs": inputs,
        "corrections": [correction.model_dump(mode="json") for correction in corrections],
        "correction_status_snapshot": _correction_status_snapshot(corrections),
        "tool_versions": software_versions,
        "result": result.model_dump(mode="json"),
    }
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": result.status,
        "dag_run_id": result.run_id,
        "template_id": result.template_id,
        "target_node_id": result.target_node_id,
        "cache_key": result.cache_keys.get(result.target_node_id or "", ""),
        "cache_keys": result.cache_keys,
        "model_branch": model_branch,
        "bic_tag": bic_tag,
        "correction_status_snapshot": _correction_status_snapshot(corrections),
        "tool_versions": software_versions,
        "artifact": str(artifact),
    }


def _order_parameter_bic_tag(fit_quality: dict[str, Any]) -> str:
    if fit_quality.get("reduced_chisq") is None:
        return "bic_not_computed"
    if fit_quality.get("relative_rms_residual") is not None and float(fit_quality["relative_rms_residual"]) < 0.2:
        return "single_branch_fit_residuals_low_bic_comparison_missing"
    return "single_branch_fit_residuals_need_alternate_model_comparison"


def _order_parameter_fit_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "classification": candidate.get("classification"),
        "probability": candidate.get("probability"),
        "scan_count": candidate.get("scan_count"),
        "temperature_range": candidate.get("temperature_range"),
        "interpretation_readiness": candidate.get("interpretation_readiness"),
        "recommended_next_checks": candidate.get("recommended_next_checks") or [],
    }


def _order_parameter_fit_review_gate(candidate: dict[str, Any], arguments: dict[str, Any], *, backend: str) -> dict[str, Any]:
    review_status = str(arguments.get("review_status") or "unreviewed")
    reviewed_by = str(arguments.get("reviewed_by") or "").strip() or None
    accept_fit = bool(arguments.get("accept_fit", False))
    model_plan = candidate.get("model_selection_plan") or []
    required_models = [item.get("model") for item in model_plan if item.get("required")]
    missing_review = [
        "raw_scan_overlay_review",
        "background_choice_review",
        "residual_review",
        "alternate_model_comparison",
    ]
    if required_models:
        missing_review.append("required_model_plan_review")
    accepted = accept_fit and review_status == "human_reviewed" and bool(reviewed_by)
    status = "accepted_human_reviewed_fit" if accepted else "candidate_fit_review_required"
    return {
        "status": status,
        "review_status": review_status,
        "reviewed_by": reviewed_by,
        "accepted": accepted,
        "backend": backend,
        "background_model": str(arguments.get("background_model") or ("panweb_profile_background" if backend == "panweb_global" else "fitted_constant")),
        "required_models_from_triage": required_models,
        "missing_review_items": [] if accepted else missing_review,
        "policy": "Fit parameters are candidate evidence only until a human review accepts the model, background choice, residuals, and required model comparisons.",
    }


def _fit_order_parameter_with_panweb(candidate: dict[str, Any], arguments: dict[str, Any], *, triage_json: Path, output_dir: Path, review_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    panweb_root = _resolve_panweb_root(arguments.get("panweb_path"))
    if panweb_root is None:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "missing_input",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["panweb_path"],
            "error": "The pan-web backend requires --panweb-path or PANWEB_PATH.",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report
    try:
        import importlib.util
        import numpy as np
    except Exception as exc:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "unavailable",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["numpy"],
            "error": f"Cannot load pan-web backend dependencies: {type(exc).__name__}: {exc}",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    auto_peaks_path = panweb_root / "src" / "auto_peaks.py"
    if not auto_peaks_path.exists() and panweb_root.name == "src":
        auto_peaks_path = panweb_root / "auto_peaks.py"
    if not auto_peaks_path.exists():
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "missing_input",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["panweb_auto_peaks"],
            "error": f"Could not find pan-web auto_peaks.py under {panweb_root}",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    try:
        spec = importlib.util.spec_from_file_location("tas_copilot_external_panweb_auto_peaks", auto_peaks_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build import spec for {auto_peaks_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "unavailable",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["panweb_auto_peaks"],
            "error": f"Could not import pan-web auto_peaks.py: {type(exc).__name__}: {exc}",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    scan_data = []
    scan_refs = []
    for row in candidate.get("per_scan_peak_summary", []):
        path = Path(str(row.get("path") or "")).expanduser()
        temp = row.get("temperature_mean")
        if not path.exists() or temp is None:
            continue
        record = _parse_scan_file(path, max_plot_points=int(arguments.get("max_plot_points", 10000)))
        plot = record.plot_data or {}
        x_values = np.array(plot.get("x") or [], dtype=float)
        y_values = np.array(plot.get("y") or [], dtype=float)
        n = min(len(x_values), len(y_values))
        if n < 5:
            continue
        x_values = x_values[:n]
        y_values = y_values[:n]
        errors = counting_measurement(y_values).errors
        scan_data.append((x_values, y_values, errors, float(temp)))
        scan_refs.append({"scan_id": row.get("scan_id"), "path": str(path), "temperature": float(temp)})

    if len(scan_data) < 3:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "missing_input",
            "candidate_id": candidate.get("candidate_id"),
            "missing_fields": ["at_least_three_hydrated_scans"],
            "scan_count": len(scan_data),
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    if arguments.get("compare_models"):
        return _compare_order_parameter_models_with_panweb(
            module,
            candidate,
            arguments,
            triage_json=triage_json,
            output_dir=output_dir,
            scan_data=scan_data,
            scan_refs=scan_refs,
            auto_peaks_path=auto_peaks_path,
            review_gate=review_gate,
        )

    try:
        global_params, scan_results = module.fit_global_order_parameter(
            scan_data,
            initial_Tc=float(arguments.get("initial_tc") or 110.0),
            initial_beta=float(arguments.get("initial_beta") or 0.5),
            use_Tc_smearing=bool(arguments.get("use_tc_smearing", False)),
            model_type=str(arguments.get("model_type") or "split"),
            inverse=not bool(arguments.get("normal_transition", True)),
            op_function=str(arguments.get("op_function") or "power_law"),
            tanh_width=float(arguments.get("tanh_width") or 5.0),
        )
    except Exception as exc:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "error",
            "candidate_id": candidate.get("candidate_id"),
            "error": f"Pan-web global order-parameter fit failed: {type(exc).__name__}: {exc}",
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    parameter_values = {key: value.get("value") if isinstance(value, dict) else value for key, value in global_params.items()}
    fit_rows = []
    for ref, result in zip(sorted(scan_refs, key=lambda item: item["temperature"]), scan_results, strict=False):
        fit_rows.append(
            {
                "scan_id": ref.get("scan_id"),
                "path": ref.get("path"),
                "temperature": _nested_value(result, "temperature"),
                "order_parameter": _nested_value(result, "order_parameter"),
                "splitting_delta": _nested_value(result, "splitting_delta"),
                "splitting_center": _nested_value(result, "splitting_center"),
                "amp_center": _nested_value(result, "amp_center"),
                "amp_side": _nested_value(result, "amp_side"),
                "chi_squared": _nested_value(result, "chi_squared"),
            }
        )
    fit_artifacts = _write_panweb_order_parameter_artifacts(output_dir, parameter_values, fit_rows)
    report = {
        "workflow_name": "fit-order-parameter",
        "status": "ok",
        "triage_json": str(triage_json),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_classification": candidate.get("classification"),
        "candidate_summary": _order_parameter_fit_candidate_summary(candidate),
        "fit_backend": "panweb_global",
        "model_source": str(auto_peaks_path),
        "model_type": str(arguments.get("model_type") or "split"),
        "op_function": str(arguments.get("op_function") or "power_law"),
        "model_selection_plan": candidate.get("model_selection_plan") or [],
        "model_comparison_status": "external_panweb_global_model_only",
        "background_model": str(arguments.get("background_model") or "panweb_profile_background"),
        "fit_acceptance": review_gate or _order_parameter_fit_review_gate(candidate, arguments, backend="panweb_global"),
        "parallel": int(arguments.get("parallel", 1)),
        "parallel_note": "Pan-web global order-parameter fit uses scipy L-BFGS-B; Bumps parallel applies to pan-web per-scan DREAM splitting fits, not this global optimizer.",
        "parameters": parameter_values,
        "scan_count": len(scan_data),
        "fit_rows": fit_rows,
        "generated_artifacts": fit_artifacts,
        "warnings": [
            "This uses the external pan-web global splitting/order-parameter fit and does not copy that scientific implementation into tascopilot.",
            "Review profile fits, background, and model choice before interpreting Tc or beta.",
            "Fit status remains candidate_fit_review_required unless a human review record accepts the model, background, and residuals.",
        ],
        "provenance": [ProvenanceRecord(source_type="triage_report", source_id=str(triage_json), method="fit_order_parameter_panweb").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
    return report


# The full pan-web global-fit grid: peak-shape transition models x order-
# parameter functions, as documented in pan-web fit_global_order_parameter.
PANWEB_MODEL_GRID: list[tuple[str, str]] = [
    ("split", "power_law"),
    ("split", "brillouin"),
    ("split", "tanh"),
    ("magnetic", "power_law"),
    ("magnetic", "brillouin"),
    ("magnetic", "tanh"),
]


def _parse_panweb_model_grid(raw: Any) -> list[tuple[str, str]]:
    if not raw:
        return list(PANWEB_MODEL_GRID)
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    grid: list[tuple[str, str]] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        model_type, _, op_function = text.partition(":")
        grid.append((model_type.strip(), (op_function or "power_law").strip()))
    return grid or list(PANWEB_MODEL_GRID)


def _compare_order_parameter_models_with_panweb(
    module: Any,
    candidate: dict[str, Any],
    arguments: dict[str, Any],
    *,
    triage_json: Path,
    output_dir: Path,
    scan_data: list[tuple],
    scan_refs: list[dict[str, Any]],
    auto_peaks_path: Path,
    review_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fit the pan-web model grid and rank variants by information criteria.

    Per-scan chi_squared from pan-web is the total weighted chi-square, so with
    known Poisson errors AIC = chi2_total + 2k and BIC = chi2_total + k*ln(N)
    up to a model-independent constant. The ranking is model-selection
    evidence, not fit acceptance: the review gate still applies to whichever
    model a human accepts.
    """
    import math

    grid = _parse_panweb_model_grid(arguments.get("model_grid"))
    n_points = sum(len(x) for x, _, _, _ in scan_data)
    variants: list[dict[str, Any]] = []
    fits: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for model_type, op_function in grid:
        entry: dict[str, Any] = {"model_type": model_type, "op_function": op_function}
        try:
            global_params, scan_results = module.fit_global_order_parameter(
                scan_data,
                initial_Tc=float(arguments.get("initial_tc") or 110.0),
                initial_beta=float(arguments.get("initial_beta") or 0.5),
                use_Tc_smearing=bool(arguments.get("use_tc_smearing", False)),
                model_type=model_type,
                inverse=not bool(arguments.get("normal_transition", True)),
                op_function=op_function,
                tanh_width=float(arguments.get("tanh_width") or 5.0),
            )
        except Exception as exc:
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            variants.append(entry)
            continue
        parameter_values = {key: value.get("value") if isinstance(value, dict) else value for key, value in global_params.items()}
        chi2_total = 0.0
        for result in scan_results:
            chi2_total += float(_nested_value(result, "chi_squared") or 0.0)
        k = len(parameter_values)
        entry.update(
            {
                "status": "ok",
                "chi_squared_total": chi2_total,
                "n_points": n_points,
                "n_parameters": k,
                "reduced_chi_squared": chi2_total / max(1, n_points - k),
                "aic": chi2_total + 2 * k,
                "bic": chi2_total + k * math.log(max(2, n_points)),
                "Tc": parameter_values.get("Tc"),
                "beta": parameter_values.get("beta"),
            }
        )
        fits[(model_type, op_function)] = (parameter_values, scan_results)
        variants.append(entry)

    fitted = [entry for entry in variants if entry.get("status") == "ok"]
    if not fitted:
        report = {
            "workflow_name": "fit-order-parameter",
            "status": "error",
            "candidate_id": candidate.get("candidate_id"),
            "error": "All pan-web model-grid fits failed.",
            "model_comparison": variants,
            "artifacts": {},
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
        return report

    best_bic = min(entry["bic"] for entry in fitted)
    best_aic = min(entry["aic"] for entry in fitted)
    for entry in fitted:
        entry["delta_bic"] = entry["bic"] - best_bic
        entry["delta_aic"] = entry["aic"] - best_aic
    fitted.sort(key=lambda entry: entry["bic"])
    selected = fitted[0]
    selected_key = (selected["model_type"], selected["op_function"])
    selected_params, selected_results = fits[selected_key]
    fit_rows = []
    for ref, result in zip(sorted(scan_refs, key=lambda item: item["temperature"]), selected_results, strict=False):
        fit_rows.append(
            {
                "scan_id": ref.get("scan_id"),
                "path": ref.get("path"),
                "temperature": _nested_value(result, "temperature"),
                "order_parameter": _nested_value(result, "order_parameter"),
                "splitting_delta": _nested_value(result, "splitting_delta"),
                "splitting_center": _nested_value(result, "splitting_center"),
                "amp_center": _nested_value(result, "amp_center"),
                "amp_side": _nested_value(result, "amp_side"),
                "chi_squared": _nested_value(result, "chi_squared"),
            }
        )
    fit_artifacts = _write_panweb_order_parameter_artifacts(output_dir, selected_params, fit_rows)
    comparison_csv = output_dir / "panweb_model_comparison.csv"
    _write_csv(
        comparison_csv,
        variants,
        ["model_type", "op_function", "status", "chi_squared_total", "reduced_chi_squared", "n_parameters", "aic", "bic", "delta_aic", "delta_bic", "Tc", "beta", "error"],
    )
    fit_artifacts["model_comparison_csv"] = str(comparison_csv)
    report = {
        "workflow_name": "fit-order-parameter",
        "status": "ok",
        "triage_json": str(triage_json),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_classification": candidate.get("classification"),
        "candidate_summary": _order_parameter_fit_candidate_summary(candidate),
        "fit_backend": "panweb_global",
        "model_source": str(auto_peaks_path),
        "model_comparison_status": "external_panweb_global_grid",
        "model_comparison": variants,
        "model_comparison_metric_note": (
            "AIC = chi2_total + 2k and BIC = chi2_total + k*ln(N) assume known Poisson "
            "count errors; deltas are relative to the best variant and are model-selection "
            "evidence, not acceptance."
        ),
        "selected_model": {
            "model_type": selected["model_type"],
            "op_function": selected["op_function"],
            "selection_basis": "lowest_bic",
            "delta_bic_runner_up": (fitted[1]["bic"] - selected["bic"]) if len(fitted) > 1 else None,
        },
        "model_type": selected["model_type"],
        "op_function": selected["op_function"],
        "background_model": str(arguments.get("background_model") or "panweb_profile_background"),
        "fit_acceptance": review_gate or _order_parameter_fit_review_gate(candidate, arguments, backend="panweb_global"),
        "parameters": selected_params,
        "scan_count": len(scan_data),
        "fit_rows": fit_rows,
        "generated_artifacts": fit_artifacts,
        "warnings": [
            "Model comparison uses the external pan-web global fit grid; the scientific implementation stays in pan-web.",
            "The lowest-BIC variant is reported as selection evidence; review profile fits, background, and residuals before accepting any model.",
            "Fit status remains candidate_fit_review_required unless a human review record accepts the model, background, and residuals.",
        ],
        "provenance": [ProvenanceRecord(source_type="triage_report", source_id=str(triage_json), method="fit_order_parameter_panweb_model_comparison").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_fit_order_parameter_markdown, json_filename="order_parameter_fit.json", markdown_filename="order_parameter_fit.md")
    return report


def _resolve_panweb_root(explicit_path: Any) -> Path | None:
    if explicit_path:
        candidate = Path(str(explicit_path)).expanduser()
        return candidate if (candidate / "src" / "auto_peaks.py").exists() or (candidate.name == "src" and (candidate / "auto_peaks.py").exists()) else None
    env_path = os.environ.get("PANWEB_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        return candidate if (candidate / "src" / "auto_peaks.py").exists() or (candidate.name == "src" and (candidate / "auto_peaks.py").exists()) else None
    return None


def _nested_value(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def _write_panweb_order_parameter_artifacts(output_dir: Path, parameters: dict[str, Any], fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    params_csv = output_dir / "panweb_order_parameter_parameters.csv"
    _write_csv(params_csv, [{"parameter": key, "value": value} for key, value in parameters.items()], ["parameter", "value"])
    fit_csv = output_dir / "panweb_order_parameter_fit_rows.csv"
    _write_csv(fit_csv, fit_rows, ["scan_id", "temperature", "order_parameter", "splitting_delta", "splitting_center", "amp_center", "amp_side", "chi_squared", "path"])
    return {"parameters_csv": str(params_csv), "fit_rows_csv": str(fit_csv)}


def _bumps_order_parameter_model_normal(temperature: Any, background: float, amplitude: float, Tc: float, beta: float, power: float) -> Any:
    import numpy as np

    t = np.asarray(temperature, dtype=float)
    reduced = np.maximum(0.0, 1.0 - t / max(float(Tc), 1e-12))
    return background + amplitude * np.power(reduced, beta * power)


def _bumps_order_parameter_model_inverse(temperature: Any, background: float, amplitude: float, Tc: float, beta: float, power: float) -> Any:
    import numpy as np

    t = np.asarray(temperature, dtype=float)
    reduced = np.maximum(0.0, t / max(float(Tc), 1e-12) - 1.0)
    return background + amplitude * np.power(reduced, beta * power)


def _select_order_parameter_candidate(triage: dict[str, Any], candidate_id: str | None) -> dict[str, Any] | None:
    candidates = triage.get("candidates") or []
    if candidate_id:
        return next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
    return candidates[0] if candidates else None


def _order_parameter_observations(candidate: dict[str, Any], *, observable: str) -> list[dict[str, Any]]:
    rows = [row for row in candidate.get("per_scan_peak_summary", []) if row.get("temperature_mean") is not None]
    if observable == "auto":
        observable = "splitting" if candidate.get("classification") == "structural_splitting_candidate" else "integrated_intensity"
    observations = []
    for row in rows:
        temp = float(row["temperature_mean"])
        if observable == "splitting":
            peaks = row.get("peak_candidates") or []
            if len(peaks) >= 2:
                xs = sorted(float(peak["x"]) for peak in peaks[:2] if peak.get("x") is not None)
                if len(xs) < 2:
                    continue
                value = 0.5 * abs(xs[-1] - xs[0])
                width = median([float(peak.get("width") or 0.0) for peak in peaks[:2]]) if peaks else 0.0
                uncertainty = max(width * 0.05, value * 0.05, 1e-3)
            elif row.get("peak_count") == 1:
                value = 0.0
                uncertainty = max(float(row.get("dominant_peak_width") or 0.0) * 0.05, 1e-3)
            else:
                continue
        else:
            value = float(row.get("integrated_signal") or 0.0)
            uncertainty = max(abs(value) ** 0.5, abs(value) * 0.05, 1e-3)
        observations.append({"scan_id": row.get("scan_id"), "path": row.get("path"), "temperature": temp, "value": value, "uncertainty": uncertainty, "observable": observable})
    observations.sort(key=lambda item: item["temperature"])
    return observations


def _order_parameter_fit_rows(temperatures: Any, values: Any, uncertainties: Any, parameters: dict[str, float], *, power: float, normal_transition: bool) -> list[dict[str, Any]]:
    rows = []
    for temp, value, uncertainty in zip(temperatures, values, uncertainties, strict=False):
        tc = parameters.get("Tc", 1.0)
        beta = parameters.get("beta", 0.5)
        if normal_transition:
            reduced = max(0.0, 1.0 - float(temp) / max(tc, 1e-12))
        else:
            reduced = max(0.0, float(temp) / max(tc, 1e-12) - 1.0)
        m = reduced**beta
        fit_value = parameters.get("background", 0.0) + parameters.get("amplitude", 0.0) * (m**power)
        rows.append({"temperature": float(temp), "observed": float(value), "uncertainty": float(uncertainty), "fit": float(fit_value), "order_parameter": float(m), "residual": float(value - fit_value)})
    return rows


def _write_order_parameter_fit_artifacts(output_dir: Path, observations: list[dict[str, Any]], fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations_csv = output_dir / "order_parameter_fit_observations.csv"
    _write_csv(observations_csv, observations, ["scan_id", "temperature", "value", "uncertainty", "observable", "path"])
    fit_csv = output_dir / "order_parameter_fit_curve.csv"
    _write_csv(fit_csv, fit_rows, ["temperature", "observed", "uncertainty", "fit", "order_parameter", "residual"])
    plot_result = _write_order_parameter_fit_plot(output_dir / "order_parameter_fit.png", fit_rows)
    return {"observations_csv": str(observations_csv), "fit_curve_csv": str(fit_csv), **plot_result}


def _write_order_parameter_fit_plot(path: Path, fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"fit_plot_status": "unavailable", "fit_plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    if not fit_rows:
        return {"fit_plot_status": "missing_input"}
    rows = sorted(fit_rows, key=lambda item: item["temperature"])
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    temps = [row["temperature"] for row in rows]
    obs = [row["observed"] for row in rows]
    err = [row["uncertainty"] for row in rows]
    fit = [row["fit"] for row in rows]
    axes[0].errorbar(temps, obs, yerr=err, fmt="o", label="observed")
    axes[0].plot(temps, fit, "-", label="Bumps fit")
    axes[0].set_ylabel("observable")
    axes[0].legend()
    axes[1].plot(temps, [row["order_parameter"] for row in rows], "o-")
    axes[1].set_xlabel("temperature (K)")
    axes[1].set_ylabel("m(T)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"fit_plot_status": "success", "fit_plot_png": str(path)}


def _order_parameter_fit_quality(fit_rows: list[dict[str, Any]], *, chisq: float, dof: int, fit_success: bool, fit_message: str, captured_warnings: list[str]) -> dict[str, Any]:
    residuals = [float(row.get("residual") or 0.0) for row in fit_rows]
    observed = [float(row.get("observed") or 0.0) for row in fit_rows]
    reduced_chisq = chisq / dof if dof > 0 else None
    rms_residual = math.sqrt(sum(value * value for value in residuals) / len(residuals)) if residuals else None
    span = (max(observed) - min(observed)) if observed else 0.0
    relative_rms = (rms_residual / span) if rms_residual is not None and span > 1e-12 else None
    flags = []
    if captured_warnings:
        flags.append("optimizer_warning")
    if not fit_success:
        flags.append("fit_success_false")
    if reduced_chisq is not None and reduced_chisq > 5:
        flags.append("large_reduced_chisq")
    if relative_rms is not None and relative_rms > 0.35:
        flags.append("large_relative_residual")
    if len(fit_rows) < 8:
        flags.append("sparse_temperature_observations")
    if not flags:
        flags.append("basic_fit_quality_checks_passed")
    return {
        "status": "review_required",
        "fit_success": fit_success,
        "fit_message": fit_message,
        "captured_warnings": captured_warnings,
        "chisq": chisq,
        "dof": dof,
        "reduced_chisq": reduced_chisq,
        "rms_residual": rms_residual,
        "relative_rms_residual": relative_rms,
        "quality_flags": flags,
        "human_review_required": True,
        "limitations": [
            "This is not model selection; compare residuals, BIC/AIC, backgrounds, and alternate peak models before interpreting parameters.",
            "Optimizer warnings are preserved even when the backend reports fit_success true.",
        ],
    }


def _fit_order_parameter_markdown(report: dict[str, Any]) -> str:
    lines = ["# Order-Parameter Fit", "", f"Status: {report.get('status')}", ""]
    if report.get("status") != "ok":
        if report.get("error"):
            lines.append(str(report["error"]))
        return "\n".join(lines)
    lines.extend(
        [
            f"Candidate: `{report.get('candidate_id')}`",
            f"Observable: `{report.get('observable')}`",
            f"Backend: `{report.get('fit_backend')}` method `{report.get('fit_method')}` parallel `{report.get('parallel')}`",
            f"Background model: `{report.get('background_model')}`",
            f"Fit success: `{report.get('fit_success')}`",
            f"Chi-square: `{report.get('chisq')}`",
            "",
            "## Acceptance Gate",
            "",
        ]
    )
    acceptance = report.get("fit_acceptance") or {}
    if acceptance:
        lines.append(f"- Status: `{acceptance.get('status')}`")
        lines.append(f"- Review status: `{acceptance.get('review_status')}`")
        lines.append(f"- Accepted: `{acceptance.get('accepted')}`")
        if acceptance.get("missing_review_items"):
            lines.append(f"- Missing review items: {', '.join(acceptance.get('missing_review_items') or [])}")
        lines.append(f"- Policy: {acceptance.get('policy')}")
        lines.append("")
    if report.get("model_comparison"):
        lines.extend(["## Model Comparison (pan-web grid)", ""])
        selected = report.get("selected_model") or {}
        if selected:
            lines.append(f"Selected by `{selected.get('selection_basis')}`: `{selected.get('model_type')}:{selected.get('op_function')}` (runner-up delta BIC: `{selected.get('delta_bic_runner_up')}`)")
            lines.append("")
        lines.append("| model | op_function | chi2_total | red. chi2 | k | AIC | dAIC | BIC | dBIC | Tc | beta |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in report.get("model_comparison") or []:
            if entry.get("status") != "ok":
                lines.append(f"| {entry.get('model_type')} | {entry.get('op_function')} | fit error | | | | | | | | |")
                continue
            lines.append(
                "| {model_type} | {op_function} | {chi2:.2f} | {red:.3f} | {k} | {aic:.2f} | {daic:.2f} | {bic:.2f} | {dbic:.2f} | {tc} | {beta} |".format(
                    model_type=entry.get("model_type"),
                    op_function=entry.get("op_function"),
                    chi2=entry.get("chi_squared_total", 0.0),
                    red=entry.get("reduced_chi_squared", 0.0),
                    k=entry.get("n_parameters"),
                    aic=entry.get("aic", 0.0),
                    daic=entry.get("delta_aic", 0.0),
                    bic=entry.get("bic", 0.0),
                    dbic=entry.get("delta_bic", 0.0),
                    tc=entry.get("Tc"),
                    beta=entry.get("beta"),
                )
            )
        if report.get("model_comparison_metric_note"):
            lines.extend(["", str(report.get("model_comparison_metric_note"))])
        lines.append("")
    if report.get("model_selection_plan"):
        lines.extend(["## Model Selection Plan", ""])
        for step in report.get("model_selection_plan") or []:
            lines.append(f"- `{step.get('model')}` required={step.get('required')}: {step.get('purpose')}")
        lines.append("")
    lines.extend(["## Fit Quality", ""])
    quality = report.get("fit_quality") or {}
    if quality:
        lines.append(f"- Status: `{quality.get('status')}`")
        lines.append(f"- Reduced chi-square: `{quality.get('reduced_chisq')}`")
        lines.append(f"- RMS residual: `{quality.get('rms_residual')}`")
        lines.append(f"- Relative RMS residual: `{quality.get('relative_rms_residual')}`")
        lines.append(f"- Quality flags: {', '.join(quality.get('quality_flags') or [])}")
        for warning in quality.get("captured_warnings") or []:
            lines.append(f"- Optimizer warning: {warning}")
        lines.append("")
    lines.extend(["## Parameters", ""])
    for key, value in (report.get("parameters") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report.get("warnings", []))
    if report.get("generated_artifacts"):
        lines.extend(["", "## Artifacts", ""])
        for key, value in report["generated_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def _dag_run_replay_markdown(report: dict[str, Any]) -> str:
    replay = report.get("cache_replay") or {}
    lines = [
        "# DAG Run Replay",
        "",
        f"Status: `{report.get('status')}`",
        f"Candidate: `{report.get('candidate_id')}`",
        f"Prior DAG run: `{replay.get('prior_run_id')}`",
        f"New DAG run: `{replay.get('new_run_id')}`",
        "",
        "## Cache-Key Comparison",
        "",
        f"- Changed nodes: {', '.join(replay.get('changed_node_ids') or []) or 'none'}",
        f"- Reusable nodes: {', '.join(replay.get('reusable_node_ids') or []) or 'none'}",
        "",
        "## Safe Conclusion",
        "",
        report.get("safe_conclusion") or "",
        "",
        "## Correction Snapshot",
        "",
    ]
    for key, value in (report.get("correction_status_snapshot") or {}).items():
        lines.append(f"- `{key}`: status `{value.get('status')}`, version `{value.get('version')}`")
    return "\n".join(lines) + "\n"


def _aggregate_single_point_records(records: list[DashboardScanRecord]) -> list[DashboardScanRecord]:
    single_point = [r for r in records if r.row_count == 1 and r.temperature_range and r.temperature_range.get("mean") is not None]
    if len(single_point) < 4:
        return []
    groups: dict[str, list[DashboardScanRecord]] = {}
    for r in single_point:
        q = r.q_center or {}
        q_key = f"{round(float(q.get('H') or q.get('QX') or 0), 1) + 0.0},{round(float(q.get('K') or q.get('QY') or 0), 1) + 0.0},{round(float(q.get('L') or q.get('QZ') or 0), 1) + 0.0}"
        family = _filename_family(r.path)
        key = f"{r.experiment_id or 'unknown'}|{family}|{q_key}"
        groups.setdefault(key, []).append(r)
    synthetic: list[DashboardScanRecord] = []
    for key, items in groups.items():
        if len(items) < 4:
            continue
        items.sort(key=lambda r: float(r.temperature_range.get("mean", 0)))
        temps = [float(r.temperature_range["mean"]) for r in items]
        intensities = []
        for r in items:
            y_vals = (r.plot_data or {}).get("y") or []
            intensities.append(float(y_vals[0]) if y_vals else 0.0)
        experiment_id, family, q_key = key.split("|", 2)
        scan_id = f"{family}_cross_file_series"
        synthetic.append(DashboardScanRecord(
            scan_id=scan_id,
            path=items[0].path,
            instrument=items[0].instrument,
            experiment_id=items[0].experiment_id,
            title=f"{family} cross-file temperature series",
            scan_type="CROSS_FILE_SERIES",
            scan_variable="Temp",
            date=items[0].date,
            measurement_kind="elastic_or_diffraction",
            detector_mode=items[0].detector_mode,
            row_count=len(items),
            column_names=["Temp", "Detector"],
            q_center=items[0].q_center or {},
            energy_range={},
            temperature_range={"min": min(temps), "max": max(temps), "mean": sum(temps) / len(temps), "width": max(temps) - min(temps)},
            intensity_summary={"status": "ok", "point_count": len(items)},
            narrow_feature_flags=[],
            plot_data={"x": temps, "y": intensities, "x_column": "Temp", "y_column": "Detector/Monitor", "all_columns_data": {"Temp": temps, "Detector": intensities}},
            metadata={"cross_file_series": True, "source_scan_ids": [r.scan_id for r in items], "source_file_count": len(items)},
        ))
    return synthetic


def _order_parameter_eligible_records(records: list[DashboardScanRecord]) -> list[DashboardScanRecord]:
    eligible = []
    for record in records:
        is_cross_file = (record.metadata or {}).get("cross_file_series", False)
        variable = (record.scan_variable or "").upper()
        detector_mode = str(record.detector_mode or "").lower()
        if record.measurement_kind == "inelastic":
            continue
        if not is_cross_file and variable not in {"A3", "A4", "QX", "QY", "QZ", "H", "K", "L"}:
            continue
        if record.row_count < 8 and not is_cross_file:
            continue
        if is_cross_file and record.row_count < 4:
            continue
        if not record.temperature_range:
            continue
        if not (record.plot_data or {}).get("x"):
            continue
        if "psd" in detector_mode or record.detector_mode or record.intensity_summary.get("status") == "ok":
            eligible.append(record)
    return eligible


def _order_parameter_groups(records: list[DashboardScanRecord], *, min_scans: int) -> dict[str, list[DashboardScanRecord]]:
    groups: dict[str, list[DashboardScanRecord]] = {}
    for record in records:
        key = "|".join(
            [
                str(record.experiment_id or "unknown"),
                _filename_family(record.path),
                _safe_id(record.title or "untitled"),
                str(record.scan_variable or "unknown"),
                _scan_axis_range_key(record),
                _fixed_angle_key(record),
                str(record.detector_mode or "unknown"),
            ]
        )
        groups.setdefault(key, []).append(record)
    return {key: sorted(items, key=lambda item: float((item.temperature_range or {}).get("mean") or 0.0)) for key, items in groups.items() if len(items) >= min_scans}


def _filename_family(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"\d+$", "", stem)
    stem = re.sub(r"[_-]+$", "", stem)
    return stem or Path(path).stem


def _fixed_angle_key(record: DashboardScanRecord) -> str:
    raw = record.metadata.get("raw_header") or {}
    descr = str(raw.get("ScanDescr") or record.metadata.get("scan_description") or "")
    for axis in ("A4", "A3"):
        match = re.search(rf"Range={axis}=([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", descr)
        if match and abs(float(match.group(1)) - float(match.group(2))) < 1e-6:
            return f"{axis}={float(match.group(1)):.3f}"
    return "fixed_angle_unknown"


def _scan_axis_range_key(record: DashboardScanRecord) -> str:
    plot = record.plot_data or {}
    xs = [float(value) for value in (plot.get("x") or []) if isinstance(value, (int, float))]
    if not xs:
        return "xrange_unknown"
    return f"x={min(xs):.3f}:{max(xs):.3f}"


def _order_parameter_candidate_summary(group_id: str, records: list[DashboardScanRecord]) -> dict[str, Any]:
    per_scan = [_scan_peak_summary(record) for record in records]
    temps = [row["temperature_mean"] for row in per_scan if row.get("temperature_mean") is not None]
    areas = [row["integrated_signal"] for row in per_scan if row.get("integrated_signal") is not None]
    peak_counts = [row["peak_count"] for row in per_scan]
    low = per_scan[: max(1, len(per_scan) // 3)]
    high = per_scan[-max(1, len(per_scan) // 3) :]
    low_peak_count = sum(row["peak_count"] for row in low) / len(low)
    high_peak_count = sum(row["peak_count"] for row in high) / len(high)
    center_span = _center_span(per_scan)
    area_ratio = (max(areas) / max(min(value for value in areas if value > 0), 1e-12)) if any(value > 0 for value in areas) else 1.0
    temp_width = max(temps) - min(temps) if temps else 0.0

    classification = "ambiguous_temperature_dependent_peak_family"
    score = 0.2
    reasons = []
    if temp_width >= 10 and len(records) >= 4:
        score += 0.2
        reasons.append("temperature series coverage")
    if low_peak_count >= 1.7 and high_peak_count <= 1.3:
        classification = "structural_splitting_candidate"
        score += 0.35
        reasons.append("more peaks at low temperature than high temperature")
    elif area_ratio >= 1.5 and center_span is not None and center_span <= 0.12:
        classification = "single_peak_intensity_order_parameter_candidate"
        score += 0.3
        reasons.append("integrated signal changes with relatively stable peak center")
    elif area_ratio >= 1.5:
        classification = "mixed_or_nuisance_peak_order_parameter_candidate"
        score += 0.2
        if center_span is None:
            reasons.append("intensity changes but peak-center evidence is missing")
        else:
            reasons.append("intensity changes but peak center/extra peaks need review")
    if max(peak_counts or [0]) >= 2:
        score += 0.08
        reasons.append("multi-peak scans present")
    if not reasons:
        score = 0.0
        reasons.append("insufficient temperature-dependent peak evidence")

    experiment_id, filename_family, title_key, scan_variable, axis_range, fixed_angle, detector_mode = group_id.split("|", 6)
    candidate = {
        "candidate_id": f"{experiment_id}-{_safe_id(filename_family)}-{_safe_id(scan_variable)}",
        "experiment_id": None if experiment_id == "unknown" else experiment_id,
        "filename_family": filename_family,
        "title_key": title_key,
        "scan_variable": scan_variable,
        "axis_range": axis_range,
        "fixed_angle": fixed_angle,
        "detector_mode": detector_mode,
        "classification": classification,
        "probability": round(min(0.95, score), 3),
        "score": round(min(0.95, score), 3),
        "score_reasons": reasons,
        "scan_count": len(records),
        "temperature_range": {"min": min(temps) if temps else None, "max": max(temps) if temps else None, "width": temp_width},
        "area_ratio": round(area_ratio, 3),
        "center_span": round(center_span, 5) if center_span is not None else None,
        "low_temperature_mean_peak_count": round(low_peak_count, 3),
        "high_temperature_mean_peak_count": round(high_peak_count, 3),
        "per_scan_peak_summary": per_scan,
        "next_checks": _order_parameter_next_checks(classification),
        "provenance": [ProvenanceRecord(source_type="scan_family", source_id=group_id, method="order_parameter_candidate_summary").model_dump(mode="json")],
    }
    candidate["interpretation_readiness"] = _order_parameter_interpretation_readiness(candidate)
    candidate["model_selection_plan"] = _order_parameter_model_selection_plan(candidate)
    return candidate


def _cross_file_order_parameter_candidate(record: DashboardScanRecord) -> dict[str, Any]:
    plot = record.plot_data or {}
    temps = [float(v) for v in (plot.get("x") or []) if v is not None]
    intensities = [float(v) for v in (plot.get("y") or []) if v is not None]
    n = min(len(temps), len(intensities))
    temps, intensities = temps[:n], intensities[:n]
    if not temps:
        return {"candidate_id": record.scan_id, "classification": "cross_file_no_data", "score": 0, "scan_count": 0}
    area_ratio = (max(intensities) / max(min(v for v in intensities if v > 0), 1e-12)) if any(v > 0 for v in intensities) else 1.0
    temp_width = max(temps) - min(temps)
    score = 0.2
    reasons = []
    if temp_width >= 10 and n >= 4:
        score += 0.2
        reasons.append("temperature series coverage")
    if area_ratio >= 1.5:
        score += 0.3
        reasons.append("intensity changes significantly with temperature")
    if area_ratio >= 5.0:
        score += 0.15
        reasons.append("large intensity ratio suggests magnetic order parameter")
    if not reasons:
        reasons.append("cross-file temperature series with insufficient variation")
    per_scan = [{"scan_id": record.scan_id, "temperature_mean": t, "integrated_signal": y, "peak_count": 1, "dominant_peak_x": None, "dominant_peak_height": y, "dominant_peak_width": None, "x_min": None, "x_max": None} for t, y in zip(temps, intensities)]
    meta = record.metadata or {}
    return {
        "candidate_id": f"{record.experiment_id or 'unknown'}-{_filename_family(record.path)}-cross-file",
        "experiment_id": record.experiment_id,
        "filename_family": _filename_family(record.path),
        "title_key": record.title or "",
        "scan_variable": "Temp",
        "axis_range": f"T={min(temps):.1f}:{max(temps):.1f}",
        "fixed_angle": f"Q=[{record.q_center.get('H', record.q_center.get('QX', '?'))},{record.q_center.get('K', record.q_center.get('QY', '?'))},{record.q_center.get('L', record.q_center.get('QZ', '?'))}]",
        "detector_mode": record.detector_mode or "",
        "classification": "cross_file_intensity_order_parameter_candidate",
        "probability": round(min(0.95, score), 3),
        "score": round(min(0.95, score), 3),
        "score_reasons": reasons,
        "scan_count": n,
        "temperature_range": {"min": min(temps), "max": max(temps), "width": temp_width},
        "area_ratio": round(area_ratio, 3),
        "center_span": None,
        "low_temperature_mean_peak_count": 1.0,
        "high_temperature_mean_peak_count": 1.0,
        "per_scan_peak_summary": per_scan,
        "next_checks": ["Verify the cross-file intensity vs temperature curve is a real order parameter.", "Check for background contributions and normalization issues.", "Fit with a power-law model if transition temperature is visible."],
        "cross_file_series": True,
        "source_file_count": meta.get("source_file_count", n),
    }


def _apply_order_parameter_label_calibration(candidates: list[dict[str, Any]], label_priors: dict[str, Any]) -> None:
    for candidate in candidates:
        channels = _order_parameter_candidate_evidence_channels(candidate)
        prior = label_calibrated_score(float(candidate.get("score") or 0.0), evidence_channels=channels, priors=label_priors)
        candidate["v0_score"] = candidate.get("score")
        candidate["label_informed_prior"] = prior
        if prior["applied"]:
            candidate["score"] = round(float(prior["applied_score"]), 3)
            candidate["probability"] = candidate["score"]
            candidate.setdefault("score_reasons", []).append("reviewed label priors adjusted candidate score")
            candidate["calibration_status"] = "weakly_calibrated_from_labels"
        else:
            candidate["calibration_status"] = "heuristic_or_label_preview"


def _order_parameter_candidate_evidence_channels(candidate: dict[str, Any]) -> list[str]:
    channels = ["evidence_channel_scan_family", "evidence_channel_repeatability", "evidence_channel_reconstruction"]
    classification = str(candidate.get("classification") or "")
    if "splitting" in classification:
        channels.append("evidence_channel_model_selection")
    if "single_peak" in classification:
        channels.append("evidence_channel_temperature_dependence")
    return channels


def _scan_peak_summary(record: DashboardScanRecord) -> dict[str, Any]:
    plot = record.plot_data or {}
    xs = [float(value) for value in (plot.get("x") or []) if isinstance(value, (int, float))]
    ys = [float(value) for value in (plot.get("y") or []) if isinstance(value, (int, float))]
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    temp = (record.temperature_range or {}).get("mean")
    peaks = _simple_peak_candidates(xs, ys)
    background = _percentile(ys, 20) if ys else 0.0
    integrated = sum(max(0.0, y - background) for y in ys)
    dominant = peaks[0] if peaks else {}
    return {
        "candidate_id": None,
        "scan_id": record.scan_id,
        "path": record.path,
        "temperature_mean": temp,
        "peak_count": len(peaks),
        "peak_candidates": peaks,
        "dominant_peak_x": dominant.get("x"),
        "dominant_peak_height": dominant.get("height"),
        "dominant_peak_width": dominant.get("width"),
        "integrated_signal": integrated,
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
    }


def _simple_peak_candidates(xs: list[float], ys: list[float]) -> list[dict[str, Any]]:
    if len(xs) < 5 or len(ys) < 5:
        return []
    background = _percentile(ys, 20)
    ymax = max(ys)
    threshold = background + 0.18 * max(1e-12, ymax - background)
    candidates = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] > threshold:
            half = background + 0.5 * (ys[i] - background)
            left = i
            while left > 0 and ys[left] > half:
                left -= 1
            right = i
            while right < len(ys) - 1 and ys[right] > half:
                right += 1
            candidates.append({"x": xs[i], "height": ys[i], "width": abs(xs[right] - xs[left]) if right > left else 0.0})
    candidates.sort(key=lambda item: item["height"], reverse=True)
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(abs(candidate["x"] - kept["x"]) > max(candidate["width"], kept["width"], 1e-6) * 0.4 for kept in filtered):
            filtered.append(candidate)
    return filtered[:5]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _center_span(per_scan: list[dict[str, Any]]) -> float | None:
    centers = [float(row["dominant_peak_x"]) for row in per_scan if row.get("dominant_peak_x") is not None]
    if not centers:
        return None
    return max(centers) - min(centers)


def _order_parameter_next_checks(classification: str) -> list[str]:
    checks = [
        "Plot every scan in the candidate family and inspect raw counts before fitting.",
        "Check background and temperature-independent nuisance peaks.",
        "Keep the result probabilistic until a reviewed fit model is run.",
    ]
    if classification == "structural_splitting_candidate":
        checks.extend(["Compare one-, two-, and constrained three-peak models with BIC/AIC.", "Use Bumps DREAM initialization followed by LM refinement for the splitting model."])
    elif classification == "single_peak_intensity_order_parameter_candidate":
        checks.extend(["Fit peak area/intensity versus temperature with background checks.", "Verify peak center and width are stable enough for a single-peak order parameter model."])
    else:
        checks.append("Ask whether this is expected to be structural splitting, magnetic intensity growth, or a mixed case before fitting.")
    return checks


def _order_parameter_interpretation_readiness(candidate: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    cautions: list[str] = []
    if int(candidate.get("scan_count") or 0) < 6:
        missing.append("denser_temperature_series")
    if not candidate.get("temperature_range") or float((candidate.get("temperature_range") or {}).get("width") or 0.0) < 20:
        missing.append("wider_temperature_range")
    if candidate.get("center_span") is None:
        missing.append("peak_center_tracking")
    if candidate.get("classification") == "mixed_or_nuisance_peak_order_parameter_candidate":
        cautions.append("temperature-dependent signal may include nuisance peaks or background")
    if candidate.get("classification") == "structural_splitting_candidate":
        missing.append("one_two_split_peak_model_comparison")
    else:
        missing.append("single_peak_background_model_comparison")
    status = "fit_ready_with_review" if not missing else "needs_review_inputs"
    return {
        "status": status,
        "missing_inputs": missing,
        "cautions": cautions,
        "human_review_required": True,
        "safe_claim": "candidate_only",
    }


def _order_parameter_model_selection_plan(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    classification = candidate.get("classification")
    common = [
        {
            "model": "constant_background_only",
            "purpose": "Reject cases where apparent temperature dependence is just background or scale drift.",
            "required": True,
        },
        {
            "model": "single_peak_with_background_by_temperature",
            "purpose": "Test whether one peak with varying area/width explains the family.",
            "required": True,
        },
    ]
    if classification == "structural_splitting_candidate":
        common.extend(
            [
                {
                    "model": "two_peak_or_split_peak_global_model",
                    "purpose": "Test whether splitting below/above a transition is statistically preferred.",
                    "required": True,
                },
                {
                    "model": "panweb_global_splitting_backend",
                    "purpose": "Use the pan-web BIC/global-fit path when available for this splitting class.",
                    "required": False,
                },
            ]
        )
    else:
        common.append(
            {
                "model": "order_parameter_power_law_on_integrated_intensity",
                "purpose": "Fit intensity-like observable only after peak/background stability is reviewed.",
                "required": False,
            }
        )
    common.append(
        {
            "model": "reviewed_human_decision",
            "purpose": "Promote candidate to physical interpretation only after residuals, BIC/AIC, and provenance are reviewed.",
            "required": True,
        }
    )
    return common


def _module_status(module_name: str) -> str:
    try:
        __import__(module_name)
    except Exception:
        return "unavailable"
    return "available"


def _write_order_parameter_artifacts(output_dir: Path, candidates: list[dict[str, Any]], per_scan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates_csv = output_dir / "order_parameter_candidates.csv"
    _write_csv(
        candidates_csv,
        candidates,
        ["candidate_id", "experiment_id", "filename_family", "classification", "probability", "scan_count", "area_ratio", "center_span", "low_temperature_mean_peak_count", "high_temperature_mean_peak_count", "interpretation_readiness"],
    )
    scan_csv = output_dir / "order_parameter_per_scan_peaks.csv"
    _write_csv(scan_csv, per_scan_rows, ["scan_id", "path", "temperature_mean", "peak_count", "dominant_peak_x", "dominant_peak_height", "dominant_peak_width", "integrated_signal", "x_min", "x_max"])
    plot_result = _write_order_parameter_plot(output_dir / "order_parameter_candidate_summary.png", candidates[0] if candidates else None)
    return {"candidates_csv": str(candidates_csv), "per_scan_peaks_csv": str(scan_csv), **plot_result}


def _write_order_parameter_plot(path: Path, candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {"summary_plot_status": "not_generated"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"summary_plot_status": "unavailable", "summary_plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    rows = [row for row in candidate.get("per_scan_peak_summary", []) if row.get("temperature_mean") is not None]
    if not rows:
        return {"summary_plot_status": "missing_input", "summary_plot_warning": "No temperature values available for top candidate."}
    rows.sort(key=lambda item: item["temperature_mean"])
    temps = [row["temperature_mean"] for row in rows]
    areas = [row["integrated_signal"] for row in rows]
    peak_counts = [row["peak_count"] for row in rows]
    centers = [row["dominant_peak_x"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    axes[0].plot(temps, areas, "o-")
    axes[0].set_ylabel("integrated signal")
    axes[1].plot(temps, peak_counts, "o-")
    axes[1].set_ylabel("peak count")
    axes[2].plot(temps, centers, "o-")
    axes[2].set_ylabel("dominant center")
    axes[2].set_xlabel("temperature (K)")
    fig.suptitle(candidate.get("candidate_id", "order parameter candidate"))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"summary_plot_status": "success", "summary_plot_png": str(path)}


def _order_parameter_markdown(report: dict[str, Any]) -> str:
    lines = ["# Order-Parameter Triage", "", f"Status: {report.get('status')}", "", "## Safe Conclusion", str(report.get("safe_conclusion") or ""), ""]
    if report.get("fit_backend_status"):
        lines.extend(["## Fit Backend Status", ""])
        for key, value in report["fit_backend_status"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## Candidate Summary", ""])
    candidates = report.get("candidates") or []
    if candidates:
        lines.append("| candidate | classification | probability | readiness | scans | T range | area ratio |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in candidates[:20]:
            tr = item.get("temperature_range") or {}
            readiness = (item.get("interpretation_readiness") or {}).get("status")
            lines.append(f"| {item.get('candidate_id')} | {item.get('classification')} | {item.get('probability')} | {readiness} | {item.get('scan_count')} | {tr.get('min')} - {tr.get('max')} | {item.get('area_ratio')} |")
    else:
        lines.append("No candidate families found.")
    lines.append("")
    if candidates:
        top = candidates[0]
        lines.extend(["## Top Candidate Checks", ""])
        lines.extend(f"- {item}" for item in top.get("next_checks", []))
        if top.get("interpretation_readiness"):
            readiness = top["interpretation_readiness"]
            lines.extend(["", "## Interpretation Readiness", ""])
            lines.append(f"- Status: `{readiness.get('status')}`")
            lines.append(f"- Missing inputs: `{readiness.get('missing_inputs')}`")
            lines.append(f"- Safe claim: `{readiness.get('safe_claim')}`")
        if top.get("model_selection_plan"):
            lines.extend(["", "## Model Selection Plan", ""])
            for step in top["model_selection_plan"]:
                lines.append(f"- `{step.get('model')}` required={step.get('required')}: {step.get('purpose')}")
        lines.append("")
    if report.get("generated_artifacts"):
        lines.extend(["## Artifacts", ""])
        for key, value in report["generated_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _parsed_summary_records(arguments: dict[str, Any], data_root: Path, *, max_plot_points: int) -> tuple[list[DashboardScanRecord], dict[str, Any]]:
    max_files = int(arguments.get("max_files", 1000))
    post_year = int(arguments.get("post_year", 0) or 0)
    patterns = tuple(arguments.get("patterns") or SCAN_PATTERNS)
    paths = _find_scan_paths(data_root, patterns=patterns, max_files=max_files, post_year=post_year)
    records = [_parse_scan_file(path, max_plot_points=max_plot_points) for path in paths]
    return _filter_records_for_summary(records, arguments)


def _overnight_record_groups(records: list[DashboardScanRecord]) -> dict[str, list[DashboardScanRecord]]:
    groups: dict[str, list[DashboardScanRecord]] = {}
    for record in records:
        key = "|".join([str(record.experiment_id or "unknown"), _overnight_family(record), str(record.scan_variable or "unknown")])
        groups.setdefault(key, []).append(record)
    return groups


def _select_inelastic_series(series: list[dict[str, Any]], requested_series_id: Any) -> dict[str, Any] | None:
    if requested_series_id:
        return next((item for item in series if item.get("series_id") == requested_series_id), None)
    inelastic_families = {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"}
    return next((item for item in series if item.get("family") in inelastic_families), None)


def _series_key_from_summary(summary: dict[str, Any]) -> str:
    experiment_id = summary.get("experiment_id") or "unknown"
    variable = (summary.get("scan_variables") or ["unknown"])[0]
    return "|".join([str(experiment_id), str(summary.get("family") or "unknown"), str(variable)])


def _inelastic_report_tables(records: list[DashboardScanRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scan_table: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for record in records:
        q_label = _q_label(record)
        temp = (record.temperature_range or {}).get("mean")
        scan_table.append(
            {
                "scan_id": record.scan_id,
                "title": record.title,
                "path": record.path,
                "scan_variable": record.scan_variable,
                "row_count": record.row_count,
                "q_label": q_label,
                "temperature_mean": temp,
                "energy_min": (record.energy_range or {}).get("min"),
                "energy_max": (record.energy_range or {}).get("max"),
                "narrow_feature_count": len(record.narrow_feature_flags),
            }
        )
        plot = record.plot_data or {}
        xs = plot.get("x") or []
        ys = plot.get("y") or []
        for index, (x, y) in enumerate(zip(xs, ys)):
            point_rows.append(
                {
                    "scan_id": record.scan_id,
                    "path": record.path,
                    "point_index": index,
                    "x": x,
                    "y": y,
                    "x_column": plot.get("x_column"),
                    "y_column": plot.get("y_column"),
                    "q_label": q_label,
                    "temperature_mean": temp,
                }
            )
        for flag in record.narrow_feature_flags:
            peak_rows.append(
                {
                    "scan_id": record.scan_id,
                    "path": record.path,
                    "x": flag.get("x"),
                    "point_index": flag.get("point_index"),
                    "width_points": flag.get("width_points"),
                    "prominence_sigma": flag.get("prominence_sigma"),
                    "note": flag.get("note"),
                }
            )
    return scan_table, point_rows, peak_rows


def _q_label(record: DashboardScanRecord) -> str:
    if all(axis in record.q_center for axis in ("H", "K", "L")):
        return f"({record.q_center['H']:.3g},{record.q_center['K']:.3g},{record.q_center['L']:.3g})"
    if record.q_center:
        return ", ".join(f"{key}={value:.3g}" for key, value in sorted(record.q_center.items()))
    return "unknown_Q"


def _write_inelastic_report_artifacts(output_dir: Path, scan_table: list[dict[str, Any]], point_rows: list[dict[str, Any]], peak_rows: list[dict[str, Any]], records: list[DashboardScanRecord]) -> dict[str, Any]:
    scan_csv = output_dir / "inelastic_scan_table.csv"
    _write_csv(scan_csv, scan_table, ["scan_id", "title", "path", "scan_variable", "row_count", "q_label", "temperature_mean", "energy_min", "energy_max", "narrow_feature_count"])
    points_csv = output_dir / "inelastic_overlay_points.csv"
    _write_csv(points_csv, point_rows, ["scan_id", "path", "point_index", "x", "y", "x_column", "y_column", "q_label", "temperature_mean"])
    peaks_csv = output_dir / "tentative_narrow_features.csv"
    _write_csv(peaks_csv, peak_rows, ["scan_id", "path", "x", "point_index", "width_points", "prominence_sigma", "note"])
    plot_result = _write_inelastic_overlay_plot(output_dir / "inelastic_overlay.png", records)
    generated: dict[str, Any] = {
        "scan_table_csv": str(scan_csv),
        "overlay_points_csv": str(points_csv),
        "tentative_narrow_features_csv": str(peaks_csv),
    }
    generated.update(plot_result)
    return generated


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_inelastic_overlay_plot(path: Path, records: list[DashboardScanRecord]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"overlay_plot_status": "unavailable", "overlay_plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plotted = 0
    for record in records[:40]:
        plot = record.plot_data or {}
        xs = plot.get("x") or []
        ys = plot.get("y") or []
        if not xs or not ys:
            continue
        label = f"{record.scan_id} {_q_label(record)}"
        ax.plot(xs, ys, lw=1.1, alpha=0.75, label=label[:80])
        plotted += 1
    ax.set_title("Inelastic scan-family overlay")
    ax.set_xlabel(str(((records[0].plot_data or {}).get("x_column") if records else None) or "scan axis"))
    ax.set_ylabel(str(((records[0].plot_data or {}).get("y_column") if records else None) or "normalized intensity"))
    if plotted <= 12:
        ax.legend(fontsize="x-small")
    else:
        ax.text(0.01, 0.99, f"{plotted} scans plotted; legend suppressed", transform=ax.transAxes, va="top", ha="left", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"overlay_plot_status": "success", "overlay_plot_png": str(path), "overlay_plot_scan_count": plotted}


def _inelastic_series_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Inelastic Series Report",
        "",
        f"Status: {report.get('status')}",
        "",
        "## Safe Conclusion",
        str(report.get("safe_conclusion") or ""),
        "",
    ]
    selected = report.get("selected_series") or {}
    if selected:
        lines.extend(
            [
                "## Selected Series",
                f"- Series: `{selected.get('series_id')}`",
                f"- Family: `{selected.get('family')}`",
                f"- Scans: {selected.get('scan_count')}",
                f"- Points: {selected.get('point_count')}",
                f"- Energy range: `{selected.get('energy_range')}`",
                f"- Temperature range: `{selected.get('temperature_range')}`",
                "",
            ]
        )
    if report.get("generated_artifacts"):
        lines.extend(["## Artifacts", ""])
        for key, value in report["generated_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    features = report.get("tentative_peak_or_narrow_features") or []
    lines.extend(["## Tentative Narrow Features", ""])
    if features:
        lines.append("| scan_id | x | width_points | prominence_sigma |")
        lines.append("| --- | --- | --- | --- |")
        for item in features[:20]:
            lines.append(f"| {item.get('scan_id')} | {item.get('x')} | {item.get('width_points')} | {item.get('prominence_sigma')} |")
    else:
        lines.append("No narrow-feature flags were produced by the local diagnostic.")
    lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    lines.extend(["## Review Required", "Use this as a raw-overlay report. Fit peaks, check backgrounds/repeats, and review spurion evidence before interpreting a dispersion.", ""])
    return "\n".join(lines)


def _load_or_build_overnight_summary(arguments: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary_json = arguments.get("summary_json")
    if summary_json:
        path = Path(str(summary_json)).expanduser()
        if not path.exists():
            return {
                "status": "missing_input",
                "missing_fields": ["summary_json"],
                "warnings": [f"Summary JSON does not exist: {path}"],
            }
        return json.loads(path.read_text(encoding="utf-8"))
    payload = dict(arguments)
    payload["output_dir"] = str(output_dir / "overnight_scan_summary")
    return overnight_scan_summary(payload)


def _draft_morning_sections(summary: dict[str, Any]) -> list[dict[str, Any]]:
    series = list(summary.get("series") or [])
    suggested_reports = list(summary.get("suggested_reports") or [])
    top_report = suggested_reports[0] if suggested_reports else {}
    sections: list[dict[str, Any]] = [
        {
            "section_id": "executive_summary",
            "title": "Executive Summary",
            "status": "draft",
            "bullets": [
                f"Parsed {summary.get('scan_count', 0)} scan(s) into {summary.get('series_count', 0)} scan-family group(s).",
                f"Most likely report type: {top_report.get('report_type', 'generic_overnight_scan_log')} with probability {top_report.get('probability', 'n/a')}.",
            ],
        },
        {
            "section_id": "scan_family_overview",
            "title": "Scan Family Overview",
            "status": "draft",
            "table": [
                {
                    "series_id": item.get("series_id"),
                    "family": item.get("family"),
                    "scan_count": item.get("scan_count"),
                    "point_count": item.get("point_count"),
                    "confidence": item.get("confidence"),
                    "probable_report_type": item.get("probable_report_type"),
                    "energy_range": item.get("energy_range"),
                    "temperature_range": item.get("temperature_range"),
                }
                for item in series[:20]
            ],
        },
        {
            "section_id": "recommended_report",
            "title": "Recommended Report",
            "status": "draft",
            "bullets": _recommended_report_bullets(top_report),
        },
        {
            "section_id": "metadata_gaps",
            "title": "Metadata Gaps",
            "status": "draft",
            "bullets": _draft_metadata_gap_bullets(summary),
        },
        {
            "section_id": "safe_next_actions",
            "title": "Next Actions",
            "status": "planning_only",
            "table": summary.get("next_actions", []),
        },
    ]
    return [section for section in sections if section.get("bullets") or section.get("table")]


def _recommended_report_bullets(top_report: dict[str, Any]) -> list[str]:
    if not top_report:
        return ["No specific report type was identified; start with a generic scan log and metadata audit."]
    bullets = [
        str(top_report.get("why") or "Report type selected from scan-family evidence."),
        "Include: " + ", ".join(map(str, top_report.get("include_sections") or [])),
    ]
    diagnostics = top_report.get("probability_diagnostics") or {}
    if diagnostics:
        bullets.append(
            "Probability basis: "
            f"{diagnostics.get('matching_scan_count', 0)}/{diagnostics.get('total_scan_count', 0)} scans match "
            f"{', '.join(diagnostics.get('matching_families') or [])}."
        )
    return bullets


def _draft_metadata_gap_bullets(summary: dict[str, Any]) -> list[str]:
    bullets = []
    families = {item.get("family") for item in summary.get("series") or []}
    if "psd_temperature_map_or_series" in families or "psd_scan" in families:
        bullets.append("PSD-derived Q maps require reviewed channel-to-A4/two-theta calibration before Q-space claims.")
    if {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"} & families:
        bullets.append("For inelastic families, review resolution/collimation/filter state and narrow-feature spurion diagnostics.")
    if "temperature_dependence_or_order_parameter" in families:
        bullets.append("For temperature series, split warming/cooling branches and check thermometer stability before fitting transitions.")
    return bullets


def _draft_followup_commands(summary: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    data_root = summary.get("data_root")
    if not data_root:
        return []
    base = [
        {
            "command_type": "dashboard",
            "description": "Open a scan browser for the same files.",
            "command": f"python -m tas_copilot_core.cli workflow experiment-dashboard --data-root {shlex.quote(str(data_root))} --output-dir {shlex.quote(str(output_dir / 'dashboard'))}",
        },
        {
            "command_type": "spurion_survey",
            "description": "Rank suspicious narrow inelastic features, if inelastic scan families are present.",
            "command": f"python -m tas_copilot_core.cli workflow survey-inelastic-spurion-evidence --data-root {shlex.quote(str(data_root))} --output-dir {shlex.quote(str(output_dir / 'spurion_survey'))}",
        },
    ]
    return base


def _write_morning_draft_artifacts(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    section_csv = output_dir / "morning_report_sections.csv"
    with section_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section_id", "title", "status", "item_count"])
        writer.writeheader()
        for section in report.get("sections", []):
            count = len(section.get("bullets") or section.get("table") or [])
            writer.writerow({"section_id": section.get("section_id"), "title": section.get("title"), "status": section.get("status"), "item_count": count})
    command_csv = output_dir / "followup_commands.csv"
    with command_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["command_type", "description", "command"])
        writer.writeheader()
        for item in report.get("followup_commands", []):
            writer.writerow(item)
    return {"sections_csv": str(section_csv), "followup_commands_csv": str(command_csv)}


def _draft_overnight_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Morning Report Draft",
        "",
        f"Status: {report.get('status')}",
        "",
        "## Summary",
        str(report.get("safe_conclusion") or ""),
        "",
    ]
    if report.get("family_plots"):
        lines.append("## Scan Family Plots")
        for item in report["family_plots"]:
            lines.append(f"### {item.get('family')}: {item.get('label')} ({item.get('scan_count')} scans in family)")
            lines.append(f"![{item.get('label')}]({item.get('plot')})")
            for note in item.get("software_change_notes") or []:
                lines.append(f"- Software change (console log): `{note}`")
        lines.append("")
    source = report.get("source_summary") or {}
    if source:
        lines.extend(
            [
                "## Source Summary",
                f"- Data root: `{source.get('data_root')}`",
                f"- Parsed scans: {source.get('scan_count')}",
                f"- Scan-family groups: {source.get('series_count')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Draft Probability",
            f"- Draft type: `{report.get('draft_type')}`",
            f"- Probability: {report.get('draft_probability')}",
        ]
    )
    diagnostics = report.get("probability_diagnostics") or {}
    if diagnostics:
        lines.append(f"- Basis: {diagnostics.get('matching_scan_count', 0)}/{diagnostics.get('total_scan_count', 0)} matching scans")
    lines.append("")
    for section in report.get("sections", []):
        lines.extend([f"## {section.get('title')}", ""])
        if section.get("bullets"):
            lines.extend(f"- {item}" for item in section["bullets"])
            lines.append("")
        if section.get("table"):
            rows = section["table"]
            if rows and isinstance(rows[0], dict):
                keys = [key for key in rows[0].keys() if key in {"series_id", "family", "scan_count", "point_count", "confidence", "probable_report_type", "action_type", "priority", "summary"}]
                lines.append("| " + " | ".join(keys) + " |")
                lines.append("| " + " | ".join("---" for _ in keys) + " |")
                for row in rows[:20]:
                    lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
                lines.append("")
    if report.get("followup_commands"):
        lines.extend(["## Follow-Up Commands", ""])
        for item in report["followup_commands"]:
            lines.extend([f"### {item.get('command_type')}", str(item.get("description") or ""), "", "```bash", str(item.get("command") or ""), "```", ""])
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    lines.extend(["## Human Review", "This draft is evidence for a morning update. It is not a completed reduction, source-data claim, or approved run plan.", ""])
    return "\n".join(lines)


def _find_scan_paths(data_root: Path, *, patterns: tuple[str, ...], max_files: int, post_year: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(data_root.rglob(pattern))
    filtered = []
    has_limit = max_files > 0
    for path in sorted(set(paths)):
        if post_year and (_path_year(path) or 0) < post_year:
            continue
        if _is_hdf5_file(path):
            # NICE writes a NeXus twin (*.nxs.bt7) beside every columns file;
            # it is HDF5, not text, and would otherwise be parsed as an
            # undated scan with no points.
            continue
        filtered.append(path)
        if has_limit and len(filtered) >= max_files:
            break
    return filtered


def _is_hdf5_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _filter_records_for_summary(records: list[DashboardScanRecord], arguments: dict[str, Any]) -> tuple[list[DashboardScanRecord], dict[str, Any]]:
    since_raw = arguments.get("since")
    hours_raw = arguments.get("hours")
    cutoff: datetime | None = None
    mode = "all_parsed_scans"
    if since_raw:
        cutoff = _parse_summary_datetime(str(since_raw))
        mode = "since"
    elif hours_raw:
        cutoff = datetime.now() - timedelta(hours=float(hours_raw))
        mode = "last_hours"
    if cutoff is None:
        return records, {"mode": mode, "cutoff": None, "kept_scan_count": len(records), "dropped_scan_count": 0}
    kept = []
    dropped = 0
    for record in records:
        stamp = _record_datetime(record)
        if stamp is None:
            kept.append(record)
            continue
        if stamp >= cutoff:
            kept.append(record)
        else:
            dropped += 1
    return kept, {"mode": mode, "cutoff": cutoff.isoformat(timespec="seconds"), "kept_scan_count": len(kept), "dropped_scan_count": dropped, "undated_scans_kept": sum(1 for record in kept if _record_datetime(record) is None)}


_ALIGNMENT_PLOT_FAMILIES = {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"}


def _load_alignment_plot_context(alignment_state_json: Any) -> dict[str, Any]:
    """Reflection assignments and software-angle-change notes for report plots.

    Assignments come from the state machine's A4/d-spacing checks; software
    changes (setlattice/setOrient/softposition) come from the console-log
    events the state report points to."""
    if not alignment_state_json:
        return {}
    paths = alignment_state_json if isinstance(alignment_state_json, (list, tuple)) else [alignment_state_json]
    assignments: dict[str, dict[str, Any]] = {}
    software_changes: list[str] = []
    for item in paths:
        context = _load_single_alignment_state(Path(str(item)).expanduser())
        assignments.update(context.get("assignments") or {})
        software_changes.extend(context.get("software_changes") or [])
    return {"assignments": assignments, "software_changes": software_changes}


def _load_single_alignment_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    assignments: dict[str, dict[str, Any]] = {}
    for check in state.get("a4_prediction_checks") or []:
        hkl = check.get("best_candidate_hkl")
        if not hkl:
            continue
        stem = Path(str(check.get("event_id") or "").split(":", 1)[-1]).stem
        resolved = str(check.get("discrimination_status") or "").startswith("resolved")
        assignments[stem] = {"hkl": str(hkl), "resolved": resolved}
    software_changes: list[str] = []
    events_path = Path(str(state.get("events_jsonl") or ""))
    if state.get("events_jsonl") and events_path.is_file():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except Exception:
                continue
            if str(event.get("event_type")) == "software_state_update" and event.get("raw_line"):
                software_changes.append(str(event["raw_line"]).strip())
    return {"assignments": assignments, "software_changes": software_changes}


def _reflection_label(record: DashboardScanRecord, assignments: dict[str, dict[str, Any]]) -> str:
    stem = Path(str(record.path)).stem if record.path else str(record.scan_id)
    assignment = assignments.get(stem)
    if not assignment:
        return "unassigned"
    hkl = f"({assignment['hkl'].replace(',', ' ')})"
    return hkl if assignment.get("resolved") else f"{hkl}?"


def _write_overnight_series_plots(
    records: list[DashboardScanRecord],
    output_dir: Path,
    *,
    alignment_context: dict[str, Any] | None = None,
    max_series: int = 16,
    max_scans_per_series: int = 15,
) -> dict[str, list[dict[str, str]]]:
    """Report plots per scan-family series, keyed by series_id.

    Non-alignment families get one overlay. Alignment families get both
    views: per-reflection summary overlays (grouped by the state machine's
    assignments) and a chronological grid of individual scans so each one can
    be judged on its own. Paths are relative to ``output_dir``; fails soft
    without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    assignments = (alignment_context or {}).get("assignments") or {}
    groups: dict[str, list[DashboardScanRecord]] = {}
    for record in records:
        key = "|".join([str(record.experiment_id or "unknown"), _overnight_family(record), str(record.scan_variable or "unknown")])
        groups.setdefault(key, []).append(record)
    ordered = sorted(groups.items(), key=lambda item: (len(item[1]), sum(r.row_count for r in item[1])), reverse=True)
    result: dict[str, list[dict[str, str]]] = {}

    def overlay(plottable: list[DashboardScanRecord], filename: str, title: str) -> str | None:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plotted = 0
        for record in plottable[:max_scans_per_series]:
            plot = record.plot_data or {}
            ax.plot(plot.get("x") or [], plot.get("y") or [], lw=1.1, alpha=0.8, marker=".", ms=3, label=str(record.scan_id)[:60])
            plotted += 1
        if not plotted:
            plt.close(fig)
            return None
        ax.set_title(title)
        ax.set_xlabel(str((plottable[0].plot_data or {}).get("x_column") or "scan axis"))
        ax.set_ylabel(str((plottable[0].plot_data or {}).get("y_column") or "normalized intensity"))
        if plotted <= 12:
            ax.legend(fontsize="x-small")
        fig.savefig(plots_dir / filename, dpi=140)
        plt.close(fig)
        return f"plots/{filename}"

    def centered_overlay(plottable: list[DashboardScanRecord], filename: str, title: str, axis: str) -> str | None:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plotted = 0
        for record in plottable[:max_scans_per_series]:
            plot = record.plot_data or {}
            xs = plot.get("x") or []
            ys = plot.get("y") or []
            if not xs or not ys:
                continue
            center = xs[max(range(len(ys)), key=lambda index: ys[index])]
            ax.plot([x - center for x in xs], ys, lw=1.1, alpha=0.8, marker=".", ms=3, label=f"{record.scan_id} @{center:.2f}"[:60])
            plotted += 1
        if not plotted:
            plt.close(fig)
            return None
        ax.set_title(title)
        ax.set_xlabel(f"{axis} - peak center (deg)")
        ax.set_ylabel(str((plottable[0].plot_data or {}).get("y_column") or "normalized intensity"))
        if plotted <= 12:
            ax.legend(fontsize="x-small")
        fig.savefig(plots_dir / filename, dpi=140)
        plt.close(fig)
        return f"plots/{filename}"

    def chronological_grid(plottable: list[DashboardScanRecord], filename: str, title: str) -> str | None:
        items = sorted(plottable, key=lambda record: str(record.date or ""))[: max_scans_per_series * 2]
        if not items:
            return None
        cols = 3
        rows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows), constrained_layout=True, squeeze=False)
        for index, record in enumerate(items):
            ax = axes[index // cols][index % cols]
            plot = record.plot_data or {}
            ax.plot(plot.get("x") or [], plot.get("y") or [], lw=1.0, marker=".", ms=3)
            time_part = str(record.date or "")[11:16]
            ax.set_title(f"{record.scan_id} {time_part} {_reflection_label(record, assignments)}"[:60], fontsize=8)
            ax.tick_params(labelsize=7)
        for index in range(len(items), rows * cols):
            axes[index // cols][index % cols].axis("off")
        fig.suptitle(title, fontsize=10)
        fig.savefig(plots_dir / filename, dpi=140)
        plt.close(fig)
        return f"plots/{filename}"

    for key, items in ordered[:max_series]:
        experiment_id, family, scan_variable = key.split("|", 2)
        series_id = f"{experiment_id}-{family}-{_safe_id(scan_variable)}"
        plottable = [record for record in items if (record.plot_data or {}).get("x")]
        if not plottable:
            continue
        entries: list[dict[str, str]] = []
        if family in _ALIGNMENT_PLOT_FAMILIES and assignments:
            # Group by reflection AND scan axis, and overlay in relative
            # coordinates: the same reflection sits at different absolute
            # angles at different incident energies (e.g. (004) at A4=75.2
            # deg for 14.7 meV vs 52.5 deg at 28 meV), and A3 zeros shift
            # across remounts. Centered overlays compare widths and shapes,
            # which is what a summary is for.
            by_group: dict[tuple[str, str], list[DashboardScanRecord]] = {}
            for record in plottable:
                axis = str(record.scan_variable or "scan axis")
                by_group.setdefault((_reflection_label(record, assignments), axis), []).append(record)
            for (label, axis), members in sorted(by_group.items()):
                png = centered_overlay(members, f"{_safe_id(series_id)}_{_safe_id(label)}_{_safe_id(axis)}.png", f"{label} {axis} scans, centered - {len(members)} scan(s)", axis)
                if png:
                    entries.append({"label": f"{label} {axis}", "png": png, "view": "reflection_summary"})
            png = chronological_grid(plottable, f"{_safe_id(series_id)}_chronological.png", f"{family} ({scan_variable}) in time order")
            if png:
                entries.append({"label": "chronological", "png": png, "view": "chronological"})
        else:
            png = overlay(plottable, f"{_safe_id(series_id)}.png", f"{family} ({scan_variable}) - {len(items)} scan(s)")
            if png:
                entries.append({"label": f"{family} ({scan_variable})", "png": png, "view": "overlay"})
        if entries:
            result[series_id] = entries
    return result


def _overnight_series(records: list[DashboardScanRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[DashboardScanRecord]] = {}
    for record in records:
        key = "|".join([str(record.experiment_id or "unknown"), _overnight_family(record), str(record.scan_variable or "unknown")])
        groups.setdefault(key, []).append(record)
    series = [_series_summary(key, items) for key, items in groups.items()]
    series.sort(key=lambda item: (item["scan_count"], item["point_count"], item["family"]), reverse=True)
    return series


def _series_summary(key: str, records: list[DashboardScanRecord]) -> dict[str, Any]:
    experiment_id, family, scan_variable = key.split("|", 2)
    kinds = sorted({record.measurement_kind for record in records if record.measurement_kind})
    variables = sorted({record.scan_variable for record in records if record.scan_variable})
    detector_modes = sorted({record.detector_mode for record in records if record.detector_mode})
    energy_ranges = [record.energy_range for record in records if record.energy_range]
    temp_ranges = [record.temperature_range for record in records if record.temperature_range]
    q_centers = [record.q_center for record in records if record.q_center]
    examples = [
        {
            "scan_id": record.scan_id,
            "title": record.title,
            "path": record.path,
            "scan_variable": record.scan_variable,
            "measurement_kind": record.measurement_kind,
            "row_count": record.row_count,
        }
        for record in records[:5]
    ]
    return {
        "series_id": f"{experiment_id}-{family}-{_safe_id(scan_variable)}",
        "experiment_id": None if experiment_id == "unknown" else experiment_id,
        "family": family,
        "scan_count": len(records),
        "point_count": sum(record.row_count for record in records),
        "measurement_kinds": kinds,
        "scan_variables": variables,
        "detector_modes": detector_modes,
        "energy_range": _merge_ranges(energy_ranges),
        "temperature_range": _merge_ranges(temp_ranges),
        "q_center_coverage": _q_coverage(q_centers),
        "examples": examples,
        "probable_report_type": _probable_report_type(family, records),
        "confidence": _series_confidence(family, records),
        "provenance": [ProvenanceRecord(source_type="scan_family", source_id=key, method="overnight_series_summary").model_dump(mode="json")],
    }


def _looks_like_reciprocal_space_map(record: DashboardScanRecord, variable: str) -> bool:
    """A scan called a mesh, or an A3 sweep far wider than any rocking curve
    (a superlattice search over a large sample rotation), is a reciprocal-space
    map even though its scan variable is a sample angle."""
    name = " ".join([Path(record.path).name, str(record.title or ""), str(record.metadata.get("scan_description") or "")]).lower()
    if "mesh" in name:
        return True
    xs = [value for value in (record.plot_data or {}).get("x", []) if isinstance(value, (int, float))]
    return variable == "A3" and len(xs) > 1 and (max(xs) - min(xs)) > 30.0


# NICE 0.2 device names -> the ICE-era vocabulary the family rules use.
_NICE_VARIABLE_ALIASES = {
    "SAMPLEUTILT": "SMPLUTILT", "SAMPLELTILT": "SMPLLTILT",
    "SAMPLEUTRANSLATION": "SMPLUTRN", "SAMPLELTRANSLATION": "SMPLLTRN", "SAMPLEELEVATION": "SMPLELEV",
    "PRESAMPLESLITWIDTH": "SMPLWDTH", "PRESAMPLESLITHEIGHT": "SMPLHGHT",
    "POSTSAMPLESLITWIDTH": "BKSLTWDTH", "POSTSAMPLESLITHEIGHT": "BKSLTHGHT",
    "BACKSLITWIDTH": "BKSLTWDTH", "BACKSLITHEIGHT": "BKSLTHGHT",
    "Q.H": "H", "Q.K": "K", "Q.L": "L", "ET": "E", "ET.DELTAE": "E", "TEMPERATURE": "TEMP", "TEMP.SET1": "TEMPERATURESETPOINT",
}
# Scans indexed by point or time rather than by a device: a count at fixed
# position while the temperature drifts is an order-parameter measurement.
_POINT_INDEX_VARIABLES = {"TRAJECTORYDATA._POINTS", "PT.", "TIME", "POINTS"}


def _canonical_scan_variable(name: str | None) -> str:
    variable = (name or "").strip().upper()
    if variable.endswith(".SOFTPOSITION"):
        variable = variable[: -len(".SOFTPOSITION")]
    return _NICE_VARIABLE_ALIASES.get(variable, variable)


def _overnight_family(record: DashboardScanRecord) -> str:
    variable = _canonical_scan_variable(record.scan_variable)
    title = " ".join([str(record.title or ""), str(record.metadata.get("scan_description") or "")]).lower()
    detector_mode = str(record.detector_mode or "").lower()
    if "psd" in detector_mode or variable == "PSDET":
        if _range_width(record.temperature_range) > 1:
            return "psd_temperature_map_or_series"
        return "psd_scan"
    if record.measurement_kind == "inelastic":
        if variable == "E" or _range_width(record.energy_range) > 0.5:
            return "constant_q_inelastic_energy_series"
        if variable in {"QX", "QY", "QZ", "H", "K", "L"} or any(term in title for term in ("transverse", "longitudinal", "qscan")):
            return "fixed_energy_inelastic_q_series"
        return "inelastic_measurement_series"
    if variable in {"TEMP", "TEMPERATURESETPOINT"} or "temp" in title:
        return "temperature_dependence_or_order_parameter"
    if variable in {"QX", "QY", "QZ", "H", "K", "L"}:
        return "elastic_q_scan_or_reciprocal_space_map"
    if variable in {"A3", "A4", "A2"} or any(term in title for term in ("rock", "bragg", "theta")):
        if _looks_like_reciprocal_space_map(record, variable):
            return "elastic_q_scan_or_reciprocal_space_map"
        return "alignment_or_rocking_scan"
    # Fine-alignment devices, same vocabulary as the alignment corpus index:
    # tilts flatten the scattering plane; slit/translation scans center the
    # sample in the beam.
    if variable in {"SMPLLTILT", "SMPLUTILT"}:
        return "alignment_tilt_scan"
    if variable in {"SMPLWDTH", "SMPLHGHT", "SMPLELEV", "SMPLLTRN", "SMPLUTRN", "BKSLTWDTH", "BKSLTHGHT"}:
        return "alignment_slit_or_translation_scan"
    if variable in _POINT_INDEX_VARIABLES and _range_width(record.temperature_range) > 1:
        return "temperature_dependence_or_order_parameter"
    return "unclassified_scan_family"


def _overnight_report_recommendations(series: list[dict[str, Any]], records: list[DashboardScanRecord]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    families = {item["family"] for item in series}
    if {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"} & families:
        families_used = {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"}
        recommendations.append(
            {
                "report_type": "inelastic_dispersion_or_lineshape_report",
                "probability": _report_probability(series, families_used),
                "why": "Overnight files include nonzero-energy scans or inelastic scan titles/ranges.",
                "probability_diagnostics": _report_probability_diagnostics(series, families_used),
                "include_sections": ["scan table", "E/Q cut overlays", "temperature/configuration summary", "spurion/background diagnostics", "candidate peak positions"],
            }
        )
    if "temperature_dependence_or_order_parameter" in families:
        families_used = {"temperature_dependence_or_order_parameter"}
        recommendations.append(
            {
                "report_type": "order_parameter_or_temperature_series_report",
                "probability": _report_probability(series, families_used),
                "why": "One or more scan families vary temperature or are titled as temperature scans.",
                "probability_diagnostics": _report_probability_diagnostics(series, families_used),
                "include_sections": ["temperature trajectory", "intensity vs temperature", "warming/cooling split", "transition-region scan list"],
            }
        )
    if "elastic_q_scan_or_reciprocal_space_map" in families or "psd_temperature_map_or_series" in families:
        families_used = {"elastic_q_scan_or_reciprocal_space_map", "psd_temperature_map_or_series"}
        recommendations.append(
            {
                "report_type": "reciprocal_space_or_detector_map_report",
                "probability": _report_probability(series, families_used),
                "why": "The parsed families include Q-varying scans or PSD detector/temperature series.",
                "probability_diagnostics": _report_probability_diagnostics(series, families_used),
                "include_sections": ["Q/angle coverage", "detector mode summary", "map-readiness check", "missing calibration warnings"],
            }
        )
    if families & {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"}:
        families_used = families & {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"}
        recommendations.append(
            {
                "report_type": "alignment_progress_report",
                "probability": _report_probability(series, families_used),
                "why": "The scan families include Bragg/rocking/angle alignment scans.",
                "probability_diagnostics": _report_probability_diagnostics(series, families_used),
                "include_sections": ["rocking curve table", "peak centers", "width trends", "instrument/sample changes"],
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "report_type": "generic_overnight_scan_log",
                "probability": 0.4 if records else 0.0,
                "why": "Parsed scans were sparse or did not form a recognizable family.",
                "include_sections": ["scan table", "metadata gaps", "next human classification step"],
            }
        )
    recommendations.sort(key=lambda item: item["probability"], reverse=True)
    return recommendations


def _overnight_next_actions(series: list[dict[str, Any]], suggested_reports: list[dict[str, Any]], scan_corpus_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not series:
        return [
            {
                "action_type": "missing_input_check",
                "priority": "high",
                "probability_of_value": 0.95,
                "probability_basis": "No parsed scan families; checking inputs is the highest-value next step.",
                "summary": "No scan families were parsed; verify data_root, file patterns, and whether scans are still being written.",
                "provenance": [ProvenanceRecord(source_type="overnight_summary", source_id="no_series", method="overnight_next_actions").model_dump(mode="json")],
            }
        ]
    top_report = suggested_reports[0]["report_type"] if suggested_reports else "generic_scan_log"
    actions.append(
        {
            "action_type": "draft_report",
            "priority": "high",
            "probability_of_value": round(float(suggested_reports[0].get("probability") or 0.5), 3) if suggested_reports else 0.5,
            "probability_basis": "Uses the highest-probability report type inferred from scan-family counts.",
            "summary": f"Draft a `{top_report}` from the grouped scan families before sending a user update.",
            "provenance": [ProvenanceRecord(source_type="overnight_report_suggestion", source_id=top_report, method="overnight_next_actions").model_dump(mode="json")],
        }
    )
    if scan_corpus_context and scan_corpus_context.get("status") == "used":
        actions.append(
            {
                "action_type": "compare_against_scan_corpus_patterns",
                "priority": "medium",
                "probability_of_value": round(min(0.9, 0.48 + 0.08 * len(scan_corpus_context.get("matched_families") or [])), 3),
                "probability_basis": "Uses private/local BT-7 scan-corpus family counts as comparative precedent without exposing prior experiment identities.",
                "summary": "Compare overnight scan-family balance against historical BT-7 patterns to spot missing follow-up scans or likely report type.",
                "provenance": [ProvenanceRecord(source_type="scan_corpus_context", source_id=str(scan_corpus_context.get("index_path")), method="overnight_next_actions").model_dump(mode="json")],
            }
        )
    for item in series[:6]:
        family = item.get("family")
        probability = _next_action_probability(item)
        basis = _next_action_probability_basis(item)
        if family == "constant_q_inelastic_energy_series":
            actions.append(
                {
                    "action_type": "inelastic_cut_overlay",
                    "priority": "high",
                    "series_id": item.get("series_id"),
                    "probability_of_value": probability,
                    "probability_basis": basis,
                    "summary": "Overlay E scans by Q point and temperature; extract tentative peak positions only after background and repeatability checks.",
                }
            )
        elif family == "fixed_energy_inelastic_q_series":
            actions.append(
                {
                    "action_type": "q_cut_overlay",
                    "priority": "medium",
                    "series_id": item.get("series_id"),
                    "probability_of_value": probability,
                    "probability_basis": basis,
                    "summary": "Overlay fixed-energy Q scans and check whether peak positions agree with constant-Q E scans.",
                }
            )
        elif family == "temperature_dependence_or_order_parameter":
            actions.append(
                {
                    "action_type": "temperature_series_plot",
                    "priority": "high",
                    "series_id": item.get("series_id"),
                    "probability_of_value": probability,
                    "probability_basis": basis,
                    "summary": "Plot intensity versus temperature, split warming and cooling branches if metadata supports it, and flag missing background scans.",
                }
            )
        elif family in {"elastic_q_scan_or_reciprocal_space_map", "psd_temperature_map_or_series"}:
            actions.append(
                {
                    "action_type": "map_readiness_check",
                    "priority": "medium",
                    "series_id": item.get("series_id"),
                    "probability_of_value": probability,
                    "probability_basis": basis,
                    "summary": "Check Q/angle coverage, detector mode, calibration requirements, and whether a map can be reconstructed without guessing.",
                }
            )
        elif family in {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"}:
            actions.append(
                {
                    "action_type": "alignment_progress_check",
                    "priority": "medium",
                    "series_id": item.get("series_id"),
                    "probability_of_value": probability,
                    "probability_basis": basis,
                    "summary": "Summarize peak centers, widths, and instrument/sample moves to decide if alignment is stable.",
                }
            )
    return actions


def _next_action_probability(series_item: dict[str, Any]) -> float:
    confidence = {"high": 0.14, "medium": 0.08, "low": 0.02}.get(str(series_item.get("confidence") or "").lower(), 0.0)
    scan_count = int(series_item.get("scan_count") or 0)
    point_count = int(series_item.get("point_count") or 0)
    probability = 0.42 + confidence + 0.2 * min(1.0, scan_count / 5.0) + 0.1 * min(1.0, point_count / 50.0)
    if series_item.get("family") in {"constant_q_inelastic_energy_series", "temperature_dependence_or_order_parameter"}:
        probability += 0.08
    return round(min(0.95, max(0.05, probability)), 3)


def _next_action_probability_basis(series_item: dict[str, Any]) -> str:
    base = (
        f"Based on family `{series_item.get('family')}`, confidence `{series_item.get('confidence')}`, "
        f"{series_item.get('scan_count', 0)} scan(s), and {series_item.get('point_count', 0)} parsed point(s)."
    )
    corpus = series_item.get("scan_corpus_precedent") or {}
    if corpus.get("status") == "matched":
        base += f" Historical scan-corpus support is `{corpus.get('support_label')}` from {corpus.get('precedent_family_count')} comparable private/local pattern(s)."
    return base


def _overnight_scan_corpus_context(series: list[dict[str, Any]], arguments: dict[str, Any]) -> dict[str, Any]:
    index_path = _resolve_precedent_index(arguments.get("precedent_index"))
    if index_path is None:
        return {
            "status": "not_supplied",
            "missing_fields": ["precedent_index"],
            "policy": "Supply --precedent-index or build the hydrated BT-7 scan-corpus index to calibrate overnight summaries against actual experiments.",
        }
    records = _load_precedent_records(index_path, max_records=int(arguments.get("precedent_max_records") or 5000))
    if not records:
        return {
            "status": "missing_input",
            "index_path": str(index_path),
            "missing_fields": ["precedent_records"],
            "policy": "No precedent records were readable from the supplied scan-corpus index.",
        }
    family_counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        families = _precedent_families_from_record(record)
        weight = int(record.get("scan_count") or record.get("datafile_count") or 1)
        for family in families:
            family_counts[family] = family_counts.get(family, 0) + max(1, min(weight, 50))
            examples.setdefault(family, [])
            if len(examples[family]) < 5:
                examples[family].append(_sanitized_precedent_example(record, family))
    current_families = {item.get("family") for item in series}
    matched = sorted(family for family in current_families if family in family_counts)
    return {
        "status": "used",
        "index_path": str(index_path),
        "record_count": len(records),
        "matched_families": matched,
        "family_counts": dict(sorted(family_counts.items(), key=lambda item: item[1], reverse=True)),
        "sanitized_examples": {family: examples.get(family, []) for family in matched},
        "user_facing_policy": "Historical BT-7 scan-corpus patterns are used as private precedent statistics; ordinary user-facing reports should not expose prior experiment IDs, raw paths, proposal IDs, or user identities.",
        "limitations": [
            "Precedent counts are heuristic and depend on hydrated metadata coverage.",
            "Similar scan family does not prove the same sample, phase, background, or scientific goal.",
            "Use this channel to calibrate priorities, not to make final interpretation claims.",
        ],
        "provenance": [ProvenanceRecord(source_type="precedent_index", source_id=str(index_path), method="overnight_scan_corpus_context").model_dump(mode="json")],
    }


def _attach_scan_corpus_context_to_series(series: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    counts = context.get("family_counts") or {}
    total = sum(int(value) for value in counts.values()) or 0
    updated = []
    for item in series:
        clone = dict(item)
        family = clone.get("family")
        count = int(counts.get(family, 0) or 0)
        support = "high" if count >= 25 else "medium" if count >= 5 else "low" if count > 0 else "none"
        clone["scan_corpus_precedent"] = {
            "status": "matched" if count else context.get("status", "not_supplied"),
            "support_label": support,
            "precedent_family_count": count,
            "precedent_fraction": round(count / total, 4) if total else 0.0,
            "user_facing_note": "Comparable historical scan-family support; private source experiment identities are withheld.",
        }
        updated.append(clone)
    return updated


def _calibrate_overnight_report_recommendations(recommendations: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    if context.get("status") != "used":
        return recommendations
    family_counts = context.get("family_counts") or {}
    report_to_families = {
        "inelastic_dispersion_or_lineshape_report": {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"},
        "order_parameter_or_temperature_series_report": {"temperature_dependence_or_order_parameter"},
        "reciprocal_space_or_detector_map_report": {"elastic_q_scan_or_reciprocal_space_map", "psd_temperature_map_or_series"},
        "alignment_progress_report": {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"},
    }
    total = sum(int(value) for value in family_counts.values()) or 0
    calibrated = []
    for item in recommendations:
        clone = dict(item)
        families = report_to_families.get(str(clone.get("report_type")), set())
        support = sum(int(family_counts.get(family, 0) or 0) for family in families)
        if total and support:
            adjustment = min(0.08, 0.02 + 0.12 * support / total)
            clone["probability"] = round(min(0.97, float(clone.get("probability") or 0.0) + adjustment), 3)
            clone["scan_corpus_adjustment"] = {
                "status": "applied",
                "support_count": support,
                "total_precedent_family_count": total,
                "adjustment": round(adjustment, 3),
                "policy": "Private/local scan-corpus support nudges report priority but does not decide interpretation.",
            }
        else:
            clone["scan_corpus_adjustment"] = {"status": "no_matching_precedent_family"}
        calibrated.append(clone)
    calibrated.sort(key=lambda item: item["probability"], reverse=True)
    return calibrated


def _resolve_precedent_index(value: Any) -> Path | None:
    if value:
        path = Path(str(value)).expanduser()
        return path if path.exists() else None
    for path in DEFAULT_PRECEDENT_INDEXES:
        if path.exists():
            return path
    return None


def _load_precedent_records(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= max_records:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _precedent_families_from_record(record: dict[str, Any]) -> set[str]:
    hydrated = record.get("hydrated_clue_summary") or {}
    top = hydrated.get("top_values") or {}
    subgroups = hydrated.get("reader_subgroups") or []
    text_parts = [str(record.get("title") or ""), json.dumps(record.get("path_clues") or {}, default=str)]
    for key in ("scan_description", "scan_variable", "detector_mode", "fixed_e", "temp_mean", "temp_start"):
        text_parts.append(json.dumps(top.get(key) or [], default=str))
    for subgroup in subgroups:
        text_parts.append(json.dumps(subgroup, default=str))
    text = " ".join(text_parts).lower()
    families: set[str] = set()
    if any(term in text for term in ("range=e=", "jtype=vector", '"value": "e"', "nonzero_energy_scan", "inelastic", "mev")):
        families.add("constant_q_inelastic_energy_series")
    if any(term in text for term in ("range=q=", "transverse", "longitudinal", "qscan", "qx", "qy", "qz")):
        families.add("fixed_energy_inelastic_q_series" if "mev" in text or "range=e=" in text else "elastic_q_scan_or_reciprocal_space_map")
    if any(term in text for term in ("temp", "temperature", "tdep", "jtype=env")):
        families.add("temperature_dependence_or_order_parameter")
    if any(term in text for term in ("psd", "psddiff", "psdflat", "psdc")):
        families.add("psd_temperature_map_or_series")
    if any(term in text for term in ("range=a3", "range=a4", "rock", "bragg", "alignment", "theta")):
        families.add("alignment_or_rocking_scan")
    return families or {"unclassified_scan_family"}


def _sanitized_precedent_example(record: dict[str, Any], family: str) -> dict[str, Any]:
    hydrated = record.get("hydrated_clue_summary") or {}
    top = hydrated.get("top_values") or {}
    return {
        "pattern_id": f"scan_corpus_pattern_{stable_text_id(str(record.get('experiment_id') or record.get('title') or family))}",
        "family": family,
        "scan_count": record.get("scan_count"),
        "datafile_count": record.get("datafile_count"),
        "scan_variables": list(top.get("scan_variable") or [])[:3],
        "fixed_energy_values": list(top.get("fixed_e") or [])[:3],
        "temperature_values": (list(top.get("temp_mean") or []) + list(top.get("temp_start") or []))[:3],
        "privacy_note": "Sanitized private/local scan-corpus pattern; source identity and paths withheld.",
    }


def stable_text_id(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _overnight_safe_conclusion(
    series: list[dict[str, Any]],
    suggested_reports: list[dict[str, Any]],
    discovery: dict[str, Any] | None = None,
) -> str:
    if not series:
        # Naming a time interval when none was applied sends people hunting for a
        # date problem when the real cause is discovery: a wrong path, unreadable
        # files, or a filename that does not match the patterns. BT-7 archive
        # directories are beamtime cycles, not months (201804 spans
        # 2018-04-26..2018-06-14), so a "wrong month" guess is a common dead end.
        info = discovery or {}
        if info.get("file_count") == 0:
            return (
                f"No scan files matched. Searched {info.get('data_root')} recursively for "
                f"{', '.join(info.get('patterns') or [])} and found 0 files. Check the path, that the files "
                "are readable by this process, and that extensions match (matching is case-sensitive). "
                "Note BT-7 directories are named by beamtime cycle, not month."
            )
        if info.get("file_count"):
            return (
                f"{info['file_count']} file(s) matched under {info.get('data_root')} but none parsed into scan "
                "families; the files may be truncated, a different format, or outside any requested interval."
            )
        return "No scan families were parsed for the requested interval."
    top = suggested_reports[0] if suggested_reports else {}
    return (
        f"The overnight files form {len(series)} scan-family group(s). "
        f"The strongest report suggestion is `{top.get('report_type')}` with probability {top.get('probability')}; "
        "this is a triage prior, not a completed reduction."
    )


def _probable_report_type(family: str, records: list[DashboardScanRecord]) -> str:
    if family in {"constant_q_inelastic_energy_series", "fixed_energy_inelastic_q_series", "inelastic_measurement_series"}:
        return "inelastic_dispersion_or_lineshape_report"
    if family == "temperature_dependence_or_order_parameter":
        return "order_parameter_or_temperature_series_report"
    if family in {"elastic_q_scan_or_reciprocal_space_map", "psd_temperature_map_or_series", "psd_scan"}:
        return "reciprocal_space_or_detector_map_report"
    if family in {"alignment_or_rocking_scan", "alignment_tilt_scan", "alignment_slit_or_translation_scan"}:
        return "alignment_progress_report"
    return "generic_scan_log"


def _series_confidence(family: str, records: list[DashboardScanRecord]) -> str:
    score = 0.25
    if family != "unclassified_scan_family":
        score += 0.25
    if len(records) >= 3:
        score += 0.2
    if any(record.row_count >= 5 for record in records):
        score += 0.1
    if any(record.energy_range or record.temperature_range or record.q_center for record in records):
        score += 0.1
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"


def _report_probability(series: list[dict[str, Any]], families: set[str]) -> float:
    if not series:
        return 0.0
    matching_scans = sum(item["scan_count"] for item in series if item["family"] in families)
    total_scans = sum(item["scan_count"] for item in series) or 1
    matching_series = sum(1 for item in series if item["family"] in families)
    probability = 0.35 + 0.45 * (matching_scans / total_scans) + 0.1 * min(1.0, matching_series / 3.0)
    return round(min(0.95, probability), 3)


def _report_probability_diagnostics(series: list[dict[str, Any]], families: set[str]) -> dict[str, Any]:
    total_scans = sum(item["scan_count"] for item in series) or 0
    matching_scans = sum(item["scan_count"] for item in series if item["family"] in families)
    matching_series = [item["series_id"] for item in series if item["family"] in families]
    return {
        "matching_families": sorted(families),
        "matching_scan_count": matching_scans,
        "total_scan_count": total_scans,
        "matching_scan_fraction": round(matching_scans / total_scans, 3) if total_scans else 0.0,
        "matching_series_ids": matching_series,
        "formula": "0.35 base + 0.45 * matching_scan_fraction + 0.1 * capped_matching_series_fraction",
    }


def _merge_ranges(ranges: list[dict[str, Any]]) -> dict[str, float]:
    mins = [float(item["min"]) for item in ranges if item.get("min") is not None]
    maxes = [float(item["max"]) for item in ranges if item.get("max") is not None]
    means = [float(item["mean"]) for item in ranges if item.get("mean") is not None]
    if not mins or not maxes:
        return {}
    merged = {"min": min(mins), "max": max(maxes)}
    if means:
        merged["mean"] = sum(means) / len(means)
    return merged


def _q_coverage(q_centers: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    coverage: dict[str, dict[str, float]] = {}
    for axis in ("QX", "QY", "QZ", "H", "K", "L"):
        values = [float(item[axis]) for item in q_centers if item.get(axis) is not None]
        if values:
            coverage[axis] = {"min": min(values), "max": max(values), "distinct_estimate": len({round(value, 4) for value in values})}
    return coverage


def _range_width(value: dict[str, Any]) -> float:
    if not value or value.get("min") is None or value.get("max") is None:
        return 0.0
    return abs(float(value["max"]) - float(value["min"]))


def _record_datetime(record: DashboardScanRecord) -> datetime | None:
    raw = record.date or (record.metadata.get("raw_header") or {}).get("Date")
    return _parse_summary_datetime(str(raw)) if raw else None


def _parse_summary_datetime(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"\s+[A-Z]{2,4}$", "", text)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_scan_file(path: Path, *, max_plot_points: int) -> DashboardScanRecord:
    header: dict[str, str] = {}
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("##"):
                # NICE 0.2 (BT-7 since ~2020) puts the column names on a
                # "## pt A3(deg) A4(deg) ..." line; strip the unit suffixes
                # and map the handful of renamed columns onto the ICE names
                # the rest of this module reads (Temp, Monitor, Time, E).
                columns = [_normalize_nice_column(name) for name in stripped[2:].split()]
                continue
            if stripped.startswith("#"):
                key, value = _parse_header_line(stripped)
                if key:
                    header[key] = value
                if key == "Columns":
                    columns = value.split()
                continue
            if columns:
                row = _parse_data_row(stripped, columns)
                if row:
                    rows.append(row)

    header = _finalize_nice_header(header)
    scan_id = header.get("Filename") or path.stem
    scan_variable = _scan_variable(header, columns, rows)
    x_col = _best_x_column(scan_variable, columns, rows)
    y_col = _first_existing(INTENSITY_COLUMNS, columns)
    q_center = {key: _mean_numeric(rows, key) for key in ("QX", "QY", "QZ", "H", "K", "L") if key in columns}
    energy_range = _range_numeric(rows, "E") if "E" in columns else {}
    temp_range = _range_numeric(rows, "Temp") if "Temp" in columns else _range_numeric(rows, "TemperatureSetpoint") if "TemperatureSetpoint" in columns else {}
    measurement_kind = _measurement_kind(path, header, rows)
    intensity_summary, flags, plot_data = _intensity_diagnostics(rows, x_col=x_col, y_col=y_col, max_plot_points=max_plot_points)
    metadata = {
        "raw_header": _small_header(header),
        "fixed_energy": header.get("FixedE"),
        "scan_description": header.get("ScanDescr"),
        "expt_details": header.get("ExptDetails"),
        "filter_clues": _material_clues(header, rows),
        "filter_state": _filter_state_summary(header, rows),
        "velocity_selector_state": _velocity_selector_state(header, rows),
        "year_hint": _path_year(path),
        "collimation_hint": _collimation_hint(header, rows),
        **_typed_harmonic_state_metadata(header, rows),
    }
    return DashboardScanRecord(
        scan_id=str(scan_id),
        path=str(path),
        instrument=header.get("InstrName"),
        experiment_id=header.get("ExptID"),
        title=_scan_title(header),
        scan_type=header.get("ScanType"),
        scan_variable=scan_variable,
        date=header.get("Date"),
        measurement_kind=measurement_kind,
        detector_mode=header.get("AnalyzerDetectorMode"),
        row_count=len(rows),
        column_names=columns,
        q_center={k: v for k, v in q_center.items() if v is not None},
        energy_range=energy_range,
        temperature_range=temp_range,
        intensity_summary=intensity_summary,
        narrow_feature_flags=flags,
        plot_data=plot_data,
        metadata=metadata,
        provenance=[ProvenanceRecord(source_type="scan_file", source_id=str(path), method="dashboard_local_parser")],
    )


NICE_HEADER_ALIASES = {
    "date": "Date",
    "filename": "Filename",
    "experiment.instrument": "InstrName",
    "experiment.proposalId": "ExptID",
    "experiment.title": "ExptName",
    "experiment.participants": "ExptParticipants",
    "primaryDetector": "AnalyzerDetectorMode",
    "sample": "Sample",
}
NICE_COLUMN_ALIASES = {"pt": "Pt.", "temp": "Temp", "temp.set1": "TemperatureSetpoint", "monitor": "Monitor", "time": "Time", "et": "E", "ei": "Ei", "ef": "Ef", "Q.H": "H", "Q.K": "K", "Q.L": "L"}
_NICE_UNIT_SUFFIX = re.compile(r"\([^()]*\)$")


def _normalize_nice_column(name: str) -> str:
    base = _NICE_UNIT_SUFFIX.sub("", name.strip())
    return NICE_COLUMN_ALIASES.get(base, base)


def _finalize_nice_header(header: dict[str, str]) -> dict[str, str]:
    """Map a NICE 0.2 header ("# date ...", "# trajectory.controls [A3]") onto the
    legacy ICE keys (Date, Filename, Scan, ScanDescr, ...) this module reads.
    Legacy ICE headers pass through untouched."""
    if "Date" in header or "date" not in header:
        return header
    for source, target in NICE_HEADER_ALIASES.items():
        if source in header and target not in header:
            header[target] = header[source]
    controls = [name.strip() for name in header.get("trajectory.controls", "").strip("[]").split(",") if name.strip()]
    if controls and "Scan" not in header:
        header["Scan"] = f"1 {controls[0]}"
    name = header.get("trajectory.name") or header.get("trajectoryData.trajName") or ""
    hierarchy = re.search(r"\"hierarchy\",\s*\"'([^']*)'\"", header.get("trajectory.config", ""))
    if "ScanDescr" not in header:
        parts = [f"Title={name}"] if name else []
        if controls and hierarchy:
            parts.append(f"Range={controls[0]}={hierarchy.group(1)}")
        if parts:
            header["ScanDescr"] = ":".join(parts)
    if name and "Comment" not in header:
        header["Comment"] = name
    return header


def _parse_header_line(line: str) -> tuple[str, str]:
    body = line[1:].strip()
    if not body:
        return "", ""
    parts = body.split(None, 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def _parse_data_row(line: str, columns: list[str]) -> dict[str, Any] | None:
    values = line.split()
    if len(values) < min(4, len(columns)):
        return None
    row: dict[str, Any] = {}
    for column, value in zip(columns, values):
        row[column] = _number_or_text(value)
    return row


def _number_or_text(value: str) -> float | str | None:
    if value.upper() in {"N/A", "NA", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _scan_variable(header: dict[str, str], columns: list[str], rows: list[dict[str, Any]]) -> str | None:
    scan = header.get("Scan") or ""
    parts = scan.split()
    if len(parts) >= 2 and parts[-1] in columns:
        return parts[-1]
    descr = header.get("ScanDescr") or ""
    match = re.search(r"Range=([A-Za-z0-9_]+)=", descr)
    if match and match.group(1) in columns:
        return match.group(1)
    return _best_x_column(None, columns, rows)


# Smallest span that counts as "this axis was scanned". NICE logs the
# positions it reached, so a K scan carries H values that jitter by 0.001 rlu;
# without a tolerance the first axis column with two distinct values wins.
_SCAN_AXIS_MIN_SPAN = {"E": 0.05, "QX": 0.01, "QY": 0.01, "QZ": 0.01, "H": 0.01, "K": 0.01, "L": 0.01, "A4": 0.05, "Temp": 0.5, "TemperatureSetpoint": 0.5, "Time": 1.0}


def _best_x_column(scan_variable: str | None, columns: list[str], rows: list[dict[str, Any]]) -> str | None:
    if scan_variable and scan_variable in columns:
        return scan_variable
    for column in SCAN_AXIS_COLUMNS:
        if column not in columns:
            continue
        values = [_numeric(row.get(column)) for row in rows]
        values = [v for v in values if v is not None]
        if len(set(values)) > 1 and (max(values) - min(values)) >= _SCAN_AXIS_MIN_SPAN.get(column, 0.0):
            return column
    return columns[0] if columns else None


def _first_existing(candidates: tuple[str, ...], columns: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _measurement_kind(path: Path, header: dict[str, str], rows: list[dict[str, Any]]) -> str:
    text = " ".join([path.name, header.get("ScanDescr", ""), header.get("Comment", ""), header.get("ExptDetails", "")]).lower()
    e_range = _range_numeric(rows, "E")
    if e_range and max(abs(float(e_range.get("min", 0))), abs(float(e_range.get("max", 0)))) > 0.15:
        return "inelastic"
    if re.search(r"\b[1-9]\d*(?:\.\d+)?\s*mev\b", text):
        return "inelastic"
    if any(term in text for term in ("elastic", "bragg", "diffraction")):
        return "elastic_or_diffraction"
    # An energy-transfer column that stays at zero is positive evidence of an
    # elastic scan, not absence of evidence: FindPeak/rocking alignment scans
    # land here (their headers never say "elastic" or "bragg").
    if e_range:
        return "elastic_or_diffraction"
    if any(term in text for term in ("findpeak", "find peak", "rocking")):
        return "elastic_or_diffraction"
    return "unknown"


def _sampled_columns(valid_rows: list[dict[str, Any]], stride: int) -> dict[str, list]:
    """Reviewable columns, decimated by `stride`, keyed by column name."""
    sampled: dict[str, list] = {}
    for col in (valid_rows[0].keys() if valid_rows else []):
        if col not in REVIEW_COLUMNS:
            continue
        vals = [_numeric(row.get(col)) for row in valid_rows]
        if any(v is not None for v in vals):
            sampled[col] = vals[::stride]
    return sampled


def _intensity_diagnostics(rows: list[dict[str, Any]], *, x_col: str | None, y_col: str | None, max_plot_points: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not rows or not y_col:
        return {"status": "missing_input", "missing": "intensity column"}, [], {}
    xs: list[float] = []
    ys: list[float] = []
    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        y = _numeric(row.get(y_col))
        if y is None:
            continue
        monitor = _numeric(row.get("Monitor"))
        if monitor and monitor > 0:
            y = y / monitor
        x = _numeric(row.get(x_col)) if x_col else None
        xs.append(float(x if x is not None else index))
        ys.append(float(y))
        valid_rows.append(row)
    if len(ys) < 5:
        # Peak finding needs >= 5 points, but the columns themselves are still
        # real data. One-point-per-file adaptive runs (tasAI on BT7) consist
        # entirely of such scans, and the pooling reports read the columns from
        # all_columns_data -- dropping them here made those runs invisible.
        return (
            {"status": "sparse", "point_count": len(ys)},
            [],
            {"x": xs, "y": ys, "x_column": x_col, "y_column": y_col, "all_columns_data": _sampled_columns(valid_rows, 1)},
        )
    baseline = median(ys)
    absdev = [abs(value - baseline) for value in ys]
    mad = median(absdev) or max(1e-12, abs(baseline) * 0.01)
    flags = []
    for i in range(1, len(ys) - 1):
        if ys[i] <= ys[i - 1] or ys[i] <= ys[i + 1]:
            continue
        prominence_sigma = (ys[i] - baseline) / mad
        if prominence_sigma < 6:
            continue
        half = baseline + (ys[i] - baseline) / 2
        left = i
        while left > 0 and ys[left - 1] > half:
            left -= 1
        right = i
        while right < len(ys) - 1 and ys[right + 1] > half:
            right += 1
        width_points = right - left + 1
        if width_points <= 3:
            flags.append(
                {
                    "flag_type": "unusually_narrow_peak",
                    "x": xs[i],
                    "point_index": i,
                    "width_points": width_points,
                    "prominence_sigma": round(prominence_sigma, 3),
                    "note": "Feature is only a few points wide relative to this scan; treat as possible spurion/artifact until checked.",
                }
            )
    stride = max(1, math.ceil(len(xs) / max_plot_points))
    all_columns_data = _sampled_columns(valid_rows, stride)
    plot_data = {"x": xs[::stride], "y": ys[::stride], "x_column": x_col, "y_column": y_col, "sample_stride": stride, "all_columns_data": all_columns_data}
    return {
        "status": "ok",
        "point_count": len(ys),
        "min": min(ys),
        "max": max(ys),
        "median": baseline,
        "mad": mad,
        "narrow_feature_count": len(flags),
    }, flags, plot_data


def _rank_spurion_likelihood(
    records: list[DashboardScanRecord],
    *,
    top_k: int,
    ranking_mode: str = "diversified",
    spurion_check: dict[str, Any] | None = None,
    resolution_proxy: dict[str, Any] | None = None,
    review_label_store: str | Path | None = None,
) -> list[SpurionLikelihoodCandidate]:
    candidates: list[SpurionLikelihoodCandidate] = []
    label_priors = label_informed_priors(
        review_label_store,
        object_type="spurion_candidate",
        source_workflow="survey-inelastic-spurion-evidence",
    )
    try:
        selector_calibration = find_instrument_calibration(kind=VELOCITY_SELECTOR_KIND)
    except Exception:
        selector_calibration = None
    for record in records:
        clues = record.metadata.get("filter_clues", {})
        high_energy_penalty = _high_energy_penalty(record.energy_range)
        for flag_index, flag in enumerate(record.narrow_feature_flags):
            score = 0.35
            evidence = [
                CandidateEvidence(evidence_type="narrow_feature", summary=f"{flag['flag_type']} at {flag.get('x')} with width {flag.get('width_points')} point(s).", score=0.35, source_id=str(flag.get("x")))
            ]
            if record.measurement_kind == "inelastic":
                score += 0.2
                evidence.append(CandidateEvidence(evidence_type="inelastic_scan", summary="Scan has nonzero energy transfer or inelastic title/header clues.", score=0.2))
            if clues.get("sample_environment"):
                score += 0.15
                evidence.append(CandidateEvidence(evidence_type="sample_environment", summary="Header/title includes sample-environment clues such as cryostat, dilution, can, or mount.", score=0.15))
            filter_state = record.metadata.get("filter_state") or {}
            velocity_selector = record.metadata.get("velocity_selector_state") or {}
            if str(filter_state.get("state") or "").upper() == "IN":
                score -= 0.08
                evidence.append(
                    CandidateEvidence(
                        evidence_type="filter_state",
                        summary="In-pile PG filter appears IN from scan columns; lambda/2 probability is lower but not zero for very strong scatterers or unknown post-sample filtering.",
                        score=-0.08,
                    )
                )
            elif str(filter_state.get("state") or "").upper() == "OUT":
                score += 0.04
                evidence.append(CandidateEvidence(evidence_type="filter_state", summary="In-pile PG filter appears OUT from scan columns; higher-order/lambda/2 probability is increased.", score=0.04))
            elif clues.get("PG"):
                score -= 0.03
                evidence.append(CandidateEvidence(evidence_type="filter_state", summary="PG/filter text clue present, but in/out state is not reviewed; this is weak counter-evidence only.", score=-0.03))
            if velocity_selector.get("state") in {"in", "on", "present"} or clues.get("velocity_selector"):
                score -= 0.1
                evidence.append(
                    CandidateEvidence(
                        evidence_type="velocity_selector_state",
                        summary="Velocity-selector evidence lowers lambda/2 probability, but does not make it impossible without reviewed configuration and material geometry.",
                        score=-0.1,
                    )
                )
            typed_filter = record.metadata.get("harmonic_filter_translation")
            typed_selector_speed = record.metadata.get("velocity_selector_speed")
            if typed_filter not in {None, "", "unknown"} or typed_selector_speed is not None:
                harmonic_state = harmonic_suppression_state(
                    typed_filter,
                    typed_selector_speed,
                    record.metadata.get("incident_energy_mev"),
                    selector_calibration,
                )
                evidence.append(
                    CandidateEvidence(
                        evidence_type="harmonic_suppression_state",
                        summary=(
                            f"Typed harmonic-suppression verdict: {harmonic_state['verdict']} "
                            f"(filter_state={harmonic_state['filter_state']}, calibration_status={harmonic_state['calibration_status']}). "
                            + " ".join(harmonic_state.get("rationale") or [])
                        ).strip(),
                        score=0.0,
                    )
                )
            if high_energy_penalty:
                score -= high_energy_penalty
                evidence.append(CandidateEvidence(evidence_type="high_energy_population_penalty", summary="Very high implied neutron energies are down-weighted because the thermal source spectrum is weak there.", score=-high_energy_penalty))
            width_context = _resolution_width_context(record, flag, resolution_proxy or {})
            if width_context["score"]:
                score += float(width_context["score"])
            evidence.append(CandidateEvidence(evidence_type="resolution_width_proxy", summary=width_context["summary"], score=float(width_context["score"])))
            spurion_followup = _maybe_run_spurion_check(record, flag, spurion_check or {})
            if spurion_followup["status"] == "success":
                count = int(spurion_followup.get("candidate_count") or 0)
                delta = min(0.18, 0.04 * count)
                score += delta
                evidence.append(CandidateEvidence(evidence_type="spurion2_followup", summary=f"tas.spurion.check_spurions returned {count} candidate(s).", score=delta, source_id=spurion_followup.get("audit_id")))
            elif spurion_followup["status"] in {"missing_input", "unavailable", "error"}:
                evidence.append(CandidateEvidence(evidence_type="spurion2_followup_not_run", summary=spurion_followup["summary"], score=0.0))
            score = max(0.0, min(1.0, score))
            evidence_channels = _spurion_candidate_evidence_channels(evidence)
            label_prior = label_calibrated_score(score, evidence_channels=evidence_channels, priors=label_priors)
            if label_prior["applied"]:
                score = float(label_prior["applied_score"])
                evidence.append(
                    CandidateEvidence(
                        evidence_type="label_informed_prior",
                        summary="Reviewed label priors adjusted this spurion triage score through " + ", ".join(label_prior["applied_channels"]) + ".",
                        score=round(float(label_prior["applied_score"]) - float(label_prior["base_score"]), 3),
                    )
                )
            label = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low" if score > 0 else "insufficient_evidence"
            missing_fields = _spurion_missing_fields(record)
            missing_fields.extend(spurion_followup.get("missing_fields", []))
            candidates.append(
                SpurionLikelihoodCandidate(
                    candidate_id=f"{record.scan_id}-narrow-{flag_index + 1}",
                    scan_id=record.scan_id,
                    path=record.path,
                    candidate_type="narrow_inelastic_feature" if record.measurement_kind == "inelastic" else "narrow_feature",
                    likelihood_score=round(score, 3),
                    likelihood_label=label,
                    evidence=evidence,
                    missing_fields=sorted(set(missing_fields)),
                    recommended_checks=[
                        "Compare feature width with nearby real peaks in the same scan family.",
                        "Check repeated scans and adjacent Q/E points for persistence.",
                        "Run a short diagnostic scan through the feature and nearby sidebands before interpreting it physically.",
                        "Run tas.spurion.check_spurions with explicit lattice/orientation and instrument configuration if available.",
                        "Check empty-can/sample-environment/background scans and filter/analyzer state.",
                    ],
                    label_informed_prior=label_prior,
                    provenance=[ProvenanceRecord(source_type="dashboard_scan_record", source_id=record.scan_id, method="rank_spurion_likelihood")],
                )
            )
    _apply_repeatability_context(candidates)
    return _select_ranked_candidates(candidates, top_k=top_k, ranking_mode=ranking_mode)


def _spurion_candidate_evidence_channels(evidence: list[CandidateEvidence]) -> list[str]:
    channels = {"evidence_channel_scan_family"}
    evidence_types = {item.evidence_type for item in evidence}
    if "filter_state" in evidence_types or "velocity_selector_state" in evidence_types:
        channels.add("evidence_channel_filter_state")
    if "resolution_width_proxy" in evidence_types:
        channels.add("evidence_channel_collimation")
    if "spurion2_followup" in evidence_types:
        channels.add("evidence_channel_spurion_calculator")
    if "narrow_feature" in evidence_types:
        channels.add("evidence_channel_feature_width")
    return sorted(channels)


def _label_calibration_summary(candidates: list[Any], *, score_key: str) -> dict[str, Any]:
    applied = 0
    preview = 0
    channels: dict[str, int] = {}
    for candidate in candidates:
        if isinstance(candidate, SpurionLikelihoodCandidate):
            prior = candidate.label_informed_prior or {}
        else:
            prior = candidate.get("label_informed_prior") or {}
        if not prior:
            continue
        if prior.get("applied"):
            applied += 1
        else:
            preview += 1
        for channel in prior.get("applied_channels") or []:
            channels[channel] = channels.get(channel, 0) + 1
    return {
        "status": "weakly_calibrated_from_labels" if applied else "label_preview_or_no_matching_labels",
        "score_field": score_key,
        "candidate_count": len(candidates),
        "applied_candidate_count": applied,
        "preview_candidate_count": preview,
        "applied_channels": dict(sorted(channels.items())),
        "policy": "Label priors adjust candidate scores only for evidence channels with at least the behavior-change minimum labels per direction.",
    }


def _resolution_width_context(record: DashboardScanRecord, flag: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    width = float(flag.get("width_points") or 0)
    expected = _expected_peak_width_points(record, config)
    source = expected.get("source")
    expected_width = float(expected.get("width_points") or 0)
    if expected_width <= 0:
        return {"score": 0.0, "summary": "No resolution-width proxy available; manual collimations or reviewed resolution inputs are missing."}
    if width <= max(1.0, expected_width * 0.55):
        score = 0.08 if source == "supplied_collimation_proxy" else 0.03
        return {
            "score": score,
            "summary": (
                f"Feature width {width:g} point(s) is narrow relative to {source} expected width "
                f"~{expected_width:g} point(s). This is a probability prior, not a resolution calculation."
            ),
        }
    return {
        "score": -0.02 if source == "supplied_collimation_proxy" else 0.0,
        "summary": f"Feature width {width:g} point(s) is not unusually narrow against {source} expected width ~{expected_width:g} point(s).",
    }


def _expected_peak_width_points(record: DashboardScanRecord, config: dict[str, Any]) -> dict[str, Any]:
    if "expected_peak_width_points" in config:
        return {"width_points": float(config["expected_peak_width_points"]), "source": "supplied_expected_peak_width"}
    collimation = config.get("collimation") or record.metadata.get("collimation_hint") or {}
    values = [float(value) for value in collimation.values() if isinstance(value, (int, float))]
    if values:
        mean_coll = sum(values) / len(values)
        return {"width_points": max(2.5, min(8.0, mean_coll / 12.0)), "source": "supplied_collimation_proxy"}
    if config.get("use_heuristic_40_40_prior", True):
        return {"width_points": 3.5, "source": "heuristic_40_40_prior"}
    return {"width_points": 0.0, "source": "missing"}


def _apply_repeatability_context(candidates: list[SpurionLikelihoodCandidate]) -> None:
    by_family: dict[str, list[SpurionLikelihoodCandidate]] = {}
    for candidate in candidates:
        by_family.setdefault(_scan_family_key(candidate), []).append(candidate)
    for family_candidates in by_family.values():
        for candidate in family_candidates:
            x = _candidate_feature_x(candidate)
            repeat_count = 0
            if x is not None:
                for other in family_candidates:
                    if other is candidate:
                        continue
                    other_x = _candidate_feature_x(other)
                    if other_x is not None and abs(other_x - x) <= max(0.15, abs(x) * 0.03):
                        repeat_count += 1
            if repeat_count:
                candidate.evidence.append(
                    CandidateEvidence(
                        evidence_type="scan_family_repeatability",
                        summary=f"Similar narrow feature appears in {repeat_count} related scan(s) from the same scan family.",
                        score=0.08,
                    )
                )
                candidate.likelihood_score = min(1.0, round(candidate.likelihood_score + 0.08, 3))
            else:
                candidate.counter_evidence.append(
                    CandidateEvidence(
                        evidence_type="scan_family_repeatability",
                        summary="No matching narrow feature found in this parsed scan-family sample; repeatability remains unconfirmed.",
                        score=-0.02,
                    )
                )
            candidate.likelihood_label = "high" if candidate.likelihood_score >= 0.7 else "medium" if candidate.likelihood_score >= 0.45 else "low" if candidate.likelihood_score > 0 else "insufficient_evidence"


def _candidate_feature_x(candidate: SpurionLikelihoodCandidate) -> float | None:
    for evidence in candidate.evidence:
        if evidence.evidence_type != "narrow_feature":
            continue
        if evidence.source_id is not None:
            try:
                return float(evidence.source_id)
            except (TypeError, ValueError):
                pass
        match = re.search(r" at ([-+]?\d+(?:\.\d+)?) ", evidence.summary)
        if match:
            return float(match.group(1))
    return None


def _scan_family_key(candidate: SpurionLikelihoodCandidate) -> str:
    stem = Path(candidate.path).stem
    normalized = re.sub(r"\d+", "#", stem)
    normalized = re.sub(r"#+$", "#", normalized)
    return f"{_candidate_bucket(candidate)}::{normalized[:24]}"


def _maybe_run_spurion_check(record: DashboardScanRecord, flag: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("enabled"):
        return {"status": "not_requested", "summary": "Spurion2 follow-up not requested.", "missing_fields": []}
    arguments, missing = _spurion_check_arguments(record, flag, config)
    if missing:
        return {
            "status": "missing_input",
            "summary": "tas.spurion.check_spurions not run; missing explicit " + ", ".join(missing) + ".",
            "missing_fields": missing,
        }
    try:
        result = default_runner().run(
            ToolCall(
                tool_namespace="tas.spurion",
                tool_name="check_spurions",
                arguments=arguments,
                caller_workflow="experiment-dashboard",
            )
        )
    except Exception as exc:
        return {"status": "error", "summary": f"tas.spurion.check_spurions failed before returning a tool result: {exc}", "missing_fields": []}
    if str(result.status) != "success" and getattr(result.status, "value", None) != "success":
        message = "; ".join(error.message for error in result.errors) or "tool returned non-success status"
        return {"status": str(getattr(result.status, "value", result.status)), "summary": f"tas.spurion.check_spurions did not succeed: {message}", "missing_fields": []}
    details = ((result.output or {}).get("details") or {})
    candidates = details.get("candidates") or []
    return {
        "status": "success",
        "summary": f"tas.spurion.check_spurions returned {len(candidates)} candidate(s).",
        "candidate_count": len(candidates),
        "audit_id": result.audit_id,
        "missing_fields": [],
    }


def _spurion_check_arguments(record: DashboardScanRecord, flag: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    lattice = config.get("lattice")
    if not lattice:
        missing.append("lattice")
    instrument_config = dict(config.get("instrument_config") or {})
    fixed_energy = instrument_config.get("fixed_energy") or _fixed_energy_value(record.metadata.get("fixed_energy"))
    if fixed_energy is not None:
        instrument_config["fixed_energy"] = fixed_energy
    if "fixed_energy_mode" not in instrument_config:
        instrument_config["fixed_energy_mode"] = config.get("fixed_energy_mode") or "fixed_final_energy"
    if fixed_energy is None and "fixed_energy" not in instrument_config:
        missing.append("fixed_energy")

    q_point = _hkl_q_point(record, config)
    if not q_point:
        missing.append("q_point_hkl")
    energy = _candidate_energy(record, flag, config)
    if energy is None:
        missing.append("energy_transfer")
    arguments = {
        "q_point": q_point or {},
        "energy_transfer": {"value": energy} if energy is not None else {},
        "instrument_config": instrument_config,
        "lattice": lattice or {},
        "q_tolerance": config.get("q_tolerance", 0.08),
        "energy_tolerance": config.get("energy_tolerance", 0.5),
        "hkl_max": config.get("hkl_max", 4),
        "harmonics": config.get("harmonics", 4),
        "include_powder_rings": config.get("include_powder_rings", True),
        "include_incoherent": config.get("include_incoherent", True),
    }
    return arguments, missing


def _hkl_q_point(record: DashboardScanRecord, config: dict[str, Any]) -> dict[str, float] | None:
    explicit = config.get("q_point")
    if explicit:
        return {"h": float(explicit["h"]), "k": float(explicit["k"]), "l": float(explicit["l"])}
    if all(axis in record.q_center for axis in ("H", "K", "L")):
        return {"h": float(record.q_center["H"]), "k": float(record.q_center["K"]), "l": float(record.q_center["L"])}
    return None


def _candidate_energy(record: DashboardScanRecord, flag: dict[str, Any], config: dict[str, Any]) -> float | None:
    explicit = config.get("energy_transfer")
    if explicit is not None:
        value = explicit.get("value") if isinstance(explicit, dict) else explicit
        return float(value)
    if (record.plot_data or {}).get("x_column") == "E" and flag.get("x") is not None:
        return float(flag["x"])
    if record.energy_range and record.energy_range.get("min") is not None and record.energy_range.get("max") is not None:
        return (float(record.energy_range["min"]) + float(record.energy_range["max"])) / 2.0
    return None


def _fixed_energy_value(raw: Any) -> float | None:
    if raw is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw))
    return float(match.group(0)) if match else None


def _select_ranked_candidates(candidates: list[SpurionLikelihoodCandidate], *, top_k: int, ranking_mode: str) -> list[SpurionLikelihoodCandidate]:
    ordered = sorted(candidates, key=lambda item: item.likelihood_score, reverse=True)
    if ranking_mode != "diversified":
        return ordered[:top_k]
    by_year: dict[str, list[SpurionLikelihoodCandidate]] = {}
    for candidate in ordered:
        by_year.setdefault(_candidate_year(candidate), []).append(candidate)
    year_order = sorted(by_year, key=lambda key: by_year[key][0].likelihood_score, reverse=True)
    per_year = {year: _diversify_by_bucket(items, top_k=len(items)) for year, items in by_year.items()}
    selected: list[SpurionLikelihoodCandidate] = []
    while len(selected) < top_k and year_order:
        next_years: list[str] = []
        for year in year_order:
            if per_year[year] and len(selected) < top_k:
                selected.append(per_year[year].pop(0))
            if per_year[year]:
                next_years.append(year)
        year_order = next_years
    return selected


def _diversify_by_bucket(candidates: list[SpurionLikelihoodCandidate], *, top_k: int) -> list[SpurionLikelihoodCandidate]:
    ordered = sorted(candidates, key=lambda item: item.likelihood_score, reverse=True)
    buckets: dict[str, list[SpurionLikelihoodCandidate]] = {}
    for candidate in ordered:
        buckets.setdefault(_candidate_bucket(candidate), []).append(candidate)
    bucket_order = sorted(buckets, key=lambda key: buckets[key][0].likelihood_score, reverse=True)
    selected: list[SpurionLikelihoodCandidate] = []
    while len(selected) < top_k and bucket_order:
        next_order: list[str] = []
        for key in bucket_order:
            if buckets[key] and len(selected) < top_k:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_order.append(key)
        bucket_order = next_order
    return selected


def _candidate_year(candidate: SpurionLikelihoodCandidate) -> str:
    for part in Path(candidate.path).parts:
        if re.fullmatch(r"20\d{4}", part):
            return part[:4]
    return "unknown"


def _candidate_bucket(candidate: SpurionLikelihoodCandidate) -> str:
    path = Path(candidate.path)
    parts = path.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"20\d{4}", part) and index + 1 < len(parts):
            return f"{part}/{parts[index + 1]}"
    return str(path.parent)


def _recommended_diagnostic_scans(candidates: list[SpurionLikelihoodCandidate], records: list[DashboardScanRecord], arguments: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_scan = {record.scan_id: record for record in records}
    recommendations: list[dict[str, Any]] = []
    for candidate in candidates[:20]:
        record = records_by_scan.get(candidate.scan_id)
        x_value = _candidate_feature_x(candidate)
        x_label = f"{x_value:g}" if x_value is not None else "the flagged position"
        base_probability = min(0.95, max(0.05, candidate.likelihood_score))
        recommendations.append(
            _diagnostic_recommendation(
                candidate,
                record,
                "repeat_feature_scan",
                base_probability,
                f"Repeat a short scan through the suspicious feature near {x_label} on axis {((record.plot_data or {}).get('x_column') if record else 'unknown')}.",
                "Checks whether the narrow feature persists in raw data with the same configuration.",
                arguments,
            )
        )
        recommendations.append(
            _diagnostic_recommendation(
                candidate,
                record,
                "sideband_background_scan",
                min(0.9, base_probability * 0.9),
                "Measure adjacent Q/E sidebands with the same counting plan.",
                "Separates a localized excitation/peak from flat or sample-environment background.",
                arguments,
            )
        )
        if _has_evidence(candidate, "sample_environment"):
            recommendations.append(
                _diagnostic_recommendation(
                    candidate,
                    record,
                    "sample_environment_or_empty_can_check",
                    min(0.9, base_probability * 0.85),
                    "Measure empty-can, mount, substrate, or sample-environment reference if experimentally practical.",
                    "Tests whether the feature follows known material or sample-environment scattering.",
                    arguments,
                )
            )
        if _has_missing(candidate, "confirmed PG/filter/analyzer state"):
            recommendations.append(
                _diagnostic_recommendation(
                    candidate,
                    record,
                    "filter_state_check",
                    min(0.75, base_probability * 0.7),
                    "Record and, if safe, compare filter/analyzer/velocity-selector state for the candidate scan family.",
                    "Higher-order and filter-dependent paths cannot be ranked confidently when the state is unlogged.",
                    arguments,
                )
            )
    return recommendations[:80]


def _diagnostic_recommendation(candidate: SpurionLikelihoodCandidate, record: DashboardScanRecord | None, check_type: str, probability: float, action: str, purpose: str, arguments: dict[str, Any]) -> dict[str, Any]:
    scan_plan = _diagnostic_scan_plan(candidate, record, check_type, arguments)
    return {
        "recommendation_id": f"{candidate.candidate_id}-{check_type}",
        "candidate_id": candidate.candidate_id,
        "scan_id": candidate.scan_id,
        "scan_path": candidate.path,
        "check_type": check_type,
        "probability_of_diagnostic_value": round(probability, 3),
        "action": action,
        "purpose": purpose,
        "not_approved_operation": True,
        "suggested_axis": (record.plot_data or {}).get("x_column") if record else None,
        "feature_position": _candidate_feature_x(candidate),
        "suggested_scan": scan_plan,
        "spurion2_eligibility": _spurion2_eligibility(candidate, record, arguments),
        "missing_input_prompts": _missing_input_prompts(candidate, arguments),
        "provenance": [ProvenanceRecord(source_type="spurion_likelihood_candidate", source_id=candidate.candidate_id, method="recommended_diagnostic_scans").model_dump(mode="json")],
    }


def _diagnostic_scan_plan(candidate: SpurionLikelihoodCandidate, record: DashboardScanRecord | None, check_type: str, arguments: dict[str, Any]) -> dict[str, Any]:
    axis = (record.plot_data or {}).get("x_column") if record else None
    center = _candidate_feature_x(candidate)
    step = _axis_step(record)
    span = _diagnostic_span(axis, step, check_type)
    count_time = float((arguments.get("diagnostic_scan_defaults") or {}).get("count_time_seconds", 60.0))
    if check_type == "sideband_background_scan":
        ranges = _sideband_ranges(center, span, step)
    elif check_type in {"sample_environment_or_empty_can_check", "filter_state_check"}:
        ranges = [{"center": center, "min": center - span if center is not None else None, "max": center + span if center is not None else None}]
    else:
        ranges = [{"center": center, "min": center - span if center is not None else None, "max": center + span if center is not None else None}]
    planned_points = sum(_point_count(item.get("min"), item.get("max"), step) for item in ranges)
    return {
        "scan_family": "diagnostic_repeatability",
        "scan_type": _scan_type_for_axis(axis),
        "axis": axis,
        "center": center,
        "ranges": ranges,
        "step": step,
        "planned_points": planned_points,
        "count_time_seconds": count_time,
        "estimated_duration_minutes": round(planned_points * count_time / 60.0, 2),
        "preserve_configuration": True,
        "planning_only": True,
        "notes": [
            "Use as an instrument-scientist planning suggestion only.",
            "Preserve analyzer/filter/collimation/sample-environment state unless the diagnostic explicitly compares it.",
            "Record manual collimations, filter state, and sample-environment materials in the scan log.",
        ],
    }


def _axis_step(record: DashboardScanRecord | None) -> float:
    xs = ((record.plot_data or {}).get("x") if record else None) or []
    numeric = [float(value) for value in xs if isinstance(value, (int, float))]
    diffs = sorted(abs(b - a) for a, b in zip(numeric, numeric[1:]) if abs(b - a) > 1e-12)
    if diffs:
        return round(median(diffs), 5)
    axis = (record.plot_data or {}).get("x_column") if record else None
    return 0.1 if axis == "E" else 0.01 if axis in {"QX", "QY", "QZ", "H", "K", "L"} else 1.0


def _diagnostic_span(axis: str | None, step: float, check_type: str) -> float:
    multiplier = 5.0 if check_type == "repeat_feature_scan" else 8.0
    minimum = 0.5 if axis == "E" else 0.05 if axis in {"QX", "QY", "QZ", "H", "K", "L"} else 3.0
    return max(minimum, step * multiplier)


def _sideband_ranges(center: float | None, span: float, step: float) -> list[dict[str, float | None]]:
    if center is None:
        return [{"center": None, "min": None, "max": None}, {"center": None, "min": None, "max": None}]
    gap = max(step * 2.0, span * 0.5)
    return [
        {"center": center - span - gap, "min": center - 2 * span - gap, "max": center - gap},
        {"center": center + span + gap, "min": center + gap, "max": center + 2 * span + gap},
    ]


def _point_count(start: float | None, stop: float | None, step: float) -> int:
    if start is None or stop is None or step <= 0:
        return 0
    return int(abs(stop - start) / step) + 1


def _scan_type_for_axis(axis: str | None) -> str:
    if axis == "E":
        return "energy_scan"
    if axis in {"QX", "QY", "QZ", "H", "K", "L"}:
        return "q_scan"
    if axis in {"A3", "A4", "TwoTheta"}:
        return "angle_scan"
    return "repeat_scan"


def _spurion2_eligibility(candidate: SpurionLikelihoodCandidate, record: DashboardScanRecord | None, arguments: dict[str, Any]) -> dict[str, Any]:
    spurion_check = arguments.get("spurion_check") or {}
    if not spurion_check.get("enabled"):
        return {"status": "not_requested", "missing_fields": ["spurion_check.enabled"], "can_run": False}
    missing = []
    if not (spurion_check.get("lattice") or arguments.get("lattice")):
        missing.append("lattice")
    if not (spurion_check.get("instrument_config") or arguments.get("instrument_config")):
        missing.append("instrument_config")
    if record and not (all(axis in record.q_center for axis in ("H", "K", "L")) or spurion_check.get("q_point")):
        missing.append("q_point_hkl")
    if record and _candidate_energy(record, {"x": _candidate_feature_x(candidate)}, spurion_check) is None:
        missing.append("energy_transfer")
    return {"status": "eligible" if not missing else "missing_input", "missing_fields": missing, "can_run": not missing}


def _missing_input_prompts(candidate: SpurionLikelihoodCandidate, arguments: dict[str, Any]) -> list[str]:
    prompts = []
    for field in candidate.missing_fields:
        if "collimation" in field:
            prompts.append("What manual pre/post-sample collimations were used for this scan family? If unknown, should the dashboard keep the weak 40/40 prior?")
        elif "PG" in field or "filter" in field:
            prompts.append("Was PG, a filter, or a velocity selector in the beam before/after the sample for this scan family?")
        elif "lattice" in field:
            prompts.append("Provide reviewed lattice parameters and scattering-plane orientation before running Spurion2 follow-up.")
        elif "q_point" in field:
            prompts.append("Provide or verify H/K/L at the candidate feature; do not infer it from a title alone.")
        elif "energy_transfer" in field:
            prompts.append("Provide or verify the energy transfer at the candidate feature.")
    if not (arguments.get("materials_in_beam") or (arguments.get("experiment_config_data") or {}).get("materials_in_beam")):
        prompts.append("Which substrate, mount, cryostat, filters, and other materials were plausibly in the beam?")
    return sorted(set(prompts))


def _has_evidence(candidate: SpurionLikelihoodCandidate, evidence_type: str) -> bool:
    return any(evidence.evidence_type == evidence_type for evidence in candidate.evidence)


def _has_missing(candidate: SpurionLikelihoodCandidate, text: str) -> bool:
    return any(text in item for item in candidate.missing_fields)


def _spurion_missing_fields(record: DashboardScanRecord) -> list[str]:
    missing = []
    if not record.metadata.get("collimation_hint"):
        missing.append("manual collimations; use 40-40 as only a ballpark prior if unknown")
    if not record.detector_mode:
        missing.append("analyzer/detector mode")
    if (record.metadata.get("filter_state") or {}).get("status") == "missing":
        missing.append("confirmed PG/filter/analyzer state")
    if (record.metadata.get("velocity_selector_state") or {}).get("status") == "missing":
        missing.append("velocity selector state if relevant for this BT-7 era")
    return missing


def _high_energy_penalty(energy_range: dict[str, Any]) -> float:
    if not energy_range:
        return 0.0
    max_energy = max(abs(float(energy_range.get("min", 0))), abs(float(energy_range.get("max", 0))))
    if max_energy > 80:
        return 0.18
    if max_energy > 50:
        return 0.1
    return 0.0


def _overlay_layers(arguments: dict[str, Any]) -> list[DashboardOverlayLayer]:
    layers: list[DashboardOverlayLayer] = []
    for pack_name in arguments.get("overlay_packs") or []:
        for item in OVERLAY_PACKS.get(str(pack_name), []):
            layers.append(
                DashboardOverlayLayer(
                    overlay_id=f"pack-{_safe_id(str(pack_name))}-{_safe_id(str(item.get('label')))}",
                    overlay_type=str(item.get("overlay_type") or "material_reference"),
                    label=str(item.get("label")),
                    material=item.get("material"),
                    enabled_by_default=False,
                    parameters={"pack": pack_name, "role": item.get("role"), "status": "reference_hook_requires_geometry_review"},
                    provenance=[ProvenanceRecord(source_type="overlay_pack", source_id=str(pack_name), method="load_dashboard_overlay_pack")],
                )
            )
    config_path = arguments.get("overlay_config")
    if config_path:
        payload = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8")) or {}
        for item in payload.get("overlays", []):
            layers.append(
                DashboardOverlayLayer(
                    overlay_id=str(item.get("overlay_id") or f"overlay-{_safe_id(item.get('label') or item.get('material') or 'custom')}"),
                    overlay_type=str(item.get("overlay_type") or "custom_reference"),
                    label=str(item.get("label") or item.get("material") or "custom overlay"),
                    material=item.get("material"),
                    enabled_by_default=bool(item.get("enabled_by_default", False)),
                    parameters=dict(item.get("parameters") or {}),
                    provenance=[ProvenanceRecord(source_type="overlay_config", source_id=str(config_path), method="load_dashboard_overlay")],
                )
            )
    for item in arguments.get("config_overlays") or []:
        layers.append(
            DashboardOverlayLayer(
                overlay_id=str(item.get("overlay_id") or f"overlay-{_safe_id(item.get('label') or item.get('material') or 'custom')}"),
                overlay_type=str(item.get("overlay_type") or "custom_reference"),
                label=str(item.get("label") or item.get("material") or "custom overlay"),
                material=item.get("material"),
                enabled_by_default=bool(item.get("enabled_by_default", False)),
                parameters=dict(item.get("parameters") or {}),
                provenance=[ProvenanceRecord(source_type="experiment_config", source_id=str(arguments.get("experiment_config_path") or "inline"), method="load_dashboard_overlay")],
            )
        )
    for role, materials in (arguments.get("materials_in_beam") or {}).items():
        for material in (materials if isinstance(materials, list) else [materials]):
            layers.append(
                DashboardOverlayLayer(
                    overlay_id=f"material-{_safe_id(str(role))}-{_safe_id(str(material))}",
                    overlay_type="material_reference",
                    label=f"{material} ({role})",
                    material=str(material),
                    enabled_by_default=False,
                    parameters={"role": role, "status": "material_declared_in_experiment_config"},
                    provenance=[ProvenanceRecord(source_type="experiment_config", source_id=str(arguments.get("experiment_config_path") or "inline"), method="materials_in_beam_overlay")],
                )
            )
    materials = arguments.get("overlay_materials") or []
    layers.extend(
        DashboardOverlayLayer(
            overlay_id=f"overlay-{_safe_id(material)}",
            overlay_type="material_reference",
            label=f"{material} reference overlay",
            material=material,
            enabled_by_default=False,
            parameters={"status": "experiment_specific_hook", "note": "Overlay positions require experiment geometry and reviewed material inputs."},
            provenance=[ProvenanceRecord(source_type="workflow_default", source_id="experiment-dashboard", method="default_overlay_layer")],
        )
        for material in materials
    )
    return layers


def _write_dashboard_artifacts(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    data_path = output_dir / "dashboard_data.json"
    table_path = output_dir / "scan_table.csv"
    html_path = output_dir / "dashboard.html"
    app_path = output_dir / "app.py"
    refresh_path = output_dir / "refresh_dashboard.sh"
    overlay_path = output_dir / "overlay_layers.csv"
    diagnostics_path = output_dir / "diagnostic_recommendations.csv"
    manifest_path = output_dir / "dashboard_manifest.json"
    artifact_paths = {
        "dashboard_data": str(data_path),
        "scan_table_csv": str(table_path),
        "overlay_layers_csv": str(overlay_path),
        "diagnostic_recommendations_csv": str(diagnostics_path),
        "dashboard_manifest": str(manifest_path),
        "dashboard_html": str(html_path),
        "dash_app": str(app_path),
        "refresh_script": str(refresh_path),
    }
    report["generated_artifacts"] = artifact_paths
    data_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_scan_table(table_path, report.get("scan_records", []))
    _write_overlay_table(overlay_path, report.get("overlay_layers", []))
    _write_dashboard_diagnostic_table(diagnostics_path, report.get("recommended_diagnostic_scans", []))
    manifest_path.write_text(json.dumps(_dashboard_manifest(report), indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_static_dashboard_html(report), encoding="utf-8")
    app_path.write_text(_dash_app_template(), encoding="utf-8")
    refresh_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + report.get("refresh_command", "") + "\n", encoding="utf-8")
    refresh_path.chmod(0o755)
    return artifact_paths


def _write_overnight_artifacts(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    series_path = output_dir / "series_table.csv"
    suggestions_path = output_dir / "report_suggestions.csv"
    examples_path = output_dir / "scan_family_examples.csv"
    actions_path = output_dir / "next_actions.csv"
    corpus_path = output_dir / "scan_corpus_context.csv"
    _write_series_table(series_path, report.get("series", []))
    _write_report_suggestions_table(suggestions_path, report.get("suggested_reports", []))
    _write_series_examples_table(examples_path, report.get("series", []))
    _write_next_actions_table(actions_path, report.get("next_actions", []))
    _write_scan_corpus_context_table(corpus_path, report.get("scan_corpus_context") or {})
    return {
        "series_table_csv": str(series_path),
        "report_suggestions_csv": str(suggestions_path),
        "scan_family_examples_csv": str(examples_path),
        "next_actions_csv": str(actions_path),
        "scan_corpus_context_csv": str(corpus_path),
    }


def _write_series_table(path: Path, series: list[dict[str, Any]]) -> None:
    fields = [
        "series_id",
        "experiment_id",
        "family",
        "scan_count",
        "point_count",
        "confidence",
        "probable_report_type",
        "measurement_kinds",
        "scan_variables",
        "detector_modes",
        "energy_min",
        "energy_max",
        "temperature_min",
        "temperature_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in series:
            energy = item.get("energy_range") or {}
            temp = item.get("temperature_range") or {}
            writer.writerow(
                {
                    "series_id": item.get("series_id"),
                    "experiment_id": item.get("experiment_id"),
                    "family": item.get("family"),
                    "scan_count": item.get("scan_count"),
                    "point_count": item.get("point_count"),
                    "confidence": item.get("confidence"),
                    "probable_report_type": item.get("probable_report_type"),
                    "measurement_kinds": ";".join(item.get("measurement_kinds") or []),
                    "scan_variables": ";".join(item.get("scan_variables") or []),
                    "detector_modes": ";".join(item.get("detector_modes") or []),
                    "energy_min": energy.get("min"),
                    "energy_max": energy.get("max"),
                    "temperature_min": temp.get("min"),
                    "temperature_max": temp.get("max"),
                }
            )


def _write_scan_corpus_context_table(path: Path, context: dict[str, Any]) -> None:
    fields = ["family", "precedent_family_count", "matched_current_family", "example_count", "status", "index_path"]
    counts = context.get("family_counts") or {}
    matched = set(context.get("matched_families") or [])
    examples = context.get("sanitized_examples") or {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if not counts:
            writer.writerow({"status": context.get("status"), "index_path": context.get("index_path")})
            return
        for family, count in counts.items():
            writer.writerow(
                {
                    "family": family,
                    "precedent_family_count": count,
                    "matched_current_family": family in matched,
                    "example_count": len(examples.get(family) or []),
                    "status": context.get("status"),
                    "index_path": context.get("index_path"),
                }
            )


def _write_report_suggestions_table(path: Path, suggestions: list[dict[str, Any]]) -> None:
    fields = ["report_type", "probability", "why", "include_sections", "matching_scan_count", "total_scan_count", "scan_corpus_support_count", "scan_corpus_adjustment"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in suggestions:
            diagnostics = item.get("probability_diagnostics") or {}
            writer.writerow(
                {
                    "report_type": item.get("report_type"),
                    "probability": item.get("probability"),
                    "why": item.get("why"),
                    "include_sections": ";".join(item.get("include_sections") or []),
                    "matching_scan_count": diagnostics.get("matching_scan_count"),
                    "total_scan_count": diagnostics.get("total_scan_count"),
                    "scan_corpus_support_count": (item.get("scan_corpus_adjustment") or {}).get("support_count"),
                    "scan_corpus_adjustment": (item.get("scan_corpus_adjustment") or {}).get("adjustment"),
                }
            )


def _write_series_examples_table(path: Path, series: list[dict[str, Any]]) -> None:
    fields = ["series_id", "family", "scan_id", "title", "path", "scan_variable", "measurement_kind", "row_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in series:
            for example in item.get("examples") or []:
                writer.writerow(
                    {
                        "series_id": item.get("series_id"),
                        "family": item.get("family"),
                        "scan_id": example.get("scan_id"),
                        "title": example.get("title"),
                        "path": example.get("path"),
                        "scan_variable": example.get("scan_variable"),
                        "measurement_kind": example.get("measurement_kind"),
                        "row_count": example.get("row_count"),
                    }
                )


def _write_next_actions_table(path: Path, actions: list[dict[str, Any]]) -> None:
    fields = ["action_type", "priority", "probability_of_value", "series_id", "summary", "probability_basis"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in actions:
            writer.writerow({field: item.get(field) for field in fields})


def _refresh_command(arguments: dict[str, Any]) -> str:
    payload_keys = ("spurion_check", "resolution_proxy", "overlay_packs", "materials_in_beam", "instrument_config", "lattice")
    payload = {key: arguments[key] for key in payload_keys if key in arguments}
    cmd = [
        sys.executable,
        "-m",
        "tas_copilot_core.cli",
        "workflow",
        "experiment-dashboard",
        "--data-root",
        str(arguments.get("data_root")),
        "--output-dir",
        str(arguments.get("output_dir")),
        "--max-files",
        str(arguments.get("max_files", 500)),
        "--post-year",
        str(arguments.get("post_year", 0) or 0),
        "--spurion-top-k",
        str(arguments.get("spurion_top_k", 50)),
        "--ranking-mode",
        str(arguments.get("ranking_mode") or "diversified"),
    ]
    if arguments.get("experiment_config_path"):
        cmd.extend(["--experiment-config", str(arguments["experiment_config_path"])])
    if arguments.get("overlay_config"):
        cmd.extend(["--overlay-config", str(arguments["overlay_config"])])
    for pattern in arguments.get("patterns") or []:
        cmd.extend(["--pattern", str(pattern)])
    for material in arguments.get("overlay_materials") or []:
        cmd.extend(["--overlay-material", str(material)])
    if payload:
        cmd.extend(["--json", json.dumps(payload, sort_keys=True)])
    return " ".join(shlex.quote(part) for part in cmd)


def _reportable_experiment_config(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_config_path": arguments.get("experiment_config_path"),
        "overlay_packs": arguments.get("overlay_packs") or [],
        "materials_in_beam": arguments.get("materials_in_beam") or {},
        "has_lattice": bool(arguments.get("lattice") or (arguments.get("spurion_check") or {}).get("lattice")),
        "has_instrument_config": bool(arguments.get("instrument_config") or (arguments.get("spurion_check") or {}).get("instrument_config")),
        "spurion_check_enabled": bool((arguments.get("spurion_check") or {}).get("enabled")),
        "resolution_proxy": arguments.get("resolution_proxy") or {},
    }


def _write_scan_table(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["scan_id", "path", "instrument", "experiment_id", "date", "title", "scan_type", "scan_variable", "measurement_kind", "detector_mode", "row_count", "narrow_feature_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record, field) for field in fields})


def _write_overlay_table(path: Path, overlays: list[dict[str, Any]]) -> None:
    fields = ["overlay_id", "overlay_type", "label", "material", "enabled_by_default", "role", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in overlays:
            params = item.get("parameters") or {}
            writer.writerow(
                {
                    "overlay_id": item.get("overlay_id"),
                    "overlay_type": item.get("overlay_type"),
                    "label": item.get("label"),
                    "material": item.get("material"),
                    "enabled_by_default": item.get("enabled_by_default"),
                    "role": params.get("role"),
                    "status": params.get("status"),
                }
            )


def _write_dashboard_diagnostic_table(path: Path, recommendations: list[dict[str, Any]]) -> None:
    fields = ["recommendation_id", "candidate_id", "check_type", "probability_of_diagnostic_value", "axis", "ranges", "planned_points", "action"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in recommendations:
            plan = item.get("suggested_scan") or {}
            writer.writerow(
                {
                    "recommendation_id": item.get("recommendation_id"),
                    "candidate_id": item.get("candidate_id"),
                    "check_type": item.get("check_type"),
                    "probability_of_diagnostic_value": item.get("probability_of_diagnostic_value"),
                    "axis": plan.get("axis"),
                    "ranges": json.dumps(plan.get("ranges") or []),
                    "planned_points": plan.get("planned_points"),
                    "action": item.get("action"),
                }
            )


def _dashboard_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dashboard_id": report.get("dashboard_id"),
        "title": report.get("title"),
        "data_root": report.get("data_root"),
        "scope": report.get("dashboard_scope"),
        "scan_count": report.get("scan_count"),
        "overlay_count": len(report.get("overlay_layers") or []),
        "diagnostic_recommendation_count": len(report.get("recommended_diagnostic_scans") or []),
        "refresh_command": report.get("refresh_command"),
        "linked_plot_policy": "Rows, candidate rows, and scan dropdown selections target the same active scan plot. Static HTML can auto-refresh by reloading the local artifact; Dash app reloads dashboard_data.json on an interval.",
        "review_policy": "Dashboard artifacts are local derived review aids, not accepted reductions or approved scan operations.",
        "generated_artifacts": report.get("generated_artifacts"),
        "safety_boundary": "read_only_local_artifact; no live control; recommendations require human review",
        "provenance": report.get("provenance", []),
    }


def experiment_day_index(arguments: dict[str, Any]) -> dict[str, Any]:
    """Index per-day report bundles for a whole experiment, newest day first.

    Expects a reports root containing YYYY-MM-DD directories, each with the
    day's summary/, report/, and dashboard/ artifacts (as produced by
    overnight-scan-summary, draft-overnight-report, and experiment-dashboard).
    """
    from tas_copilot_core.reporting import write_report_bundle

    output_dir = Path(str(arguments.get("output_dir") or arguments.get("reports_root") or "")).expanduser()
    reports_root = Path(str(arguments.get("reports_root") or "")).expanduser()
    if not reports_root.exists():
        report = {"workflow_name": "experiment-day-index", "status": "missing_input", "missing_inputs": ["reports_root"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_day_index, json_filename="experiment_day_index.json", markdown_filename="experiment_day_index.md")
        return report
    day_dirs = sorted((p for p in reports_root.iterdir() if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name)), key=lambda p: p.name, reverse=True)
    days = []
    for day_dir in day_dirs:
        entry: dict[str, Any] = {"date": day_dir.name}
        md_path = day_dir / "report/morning_report_draft.md"
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8", errors="replace")
            parsed = re.search(r"Parsed scans: (\d+)", text)
            draft = re.search(r"Draft type: `([^`]+)`", text)
            entry["parsed_scans"] = int(parsed.group(1)) if parsed else None
            entry["draft_type"] = draft.group(1) if draft else None
            entry["alignment_scan_rows"] = sum(int(n) for n in re.findall(r"alignment_\w+ \| (\d+) \|", text))
        for key, candidates in (
            ("dashboard", ("dashboard/dashboard.html",)),
            ("morning_report", ("report/morning_report_draft.html", "report/morning_report_draft.md")),
            ("summary", ("summary/overnight_scan_summary.html", "summary/overnight_scan_summary.md")),
        ):
            for rel in candidates:
                if (day_dir / rel).exists():
                    entry[key] = str(Path(day_dir.name) / rel)
                    break
        days.append(entry)
    story_html = reports_root / "story/experiment_story_report.html"
    title = str(arguments.get("title") or f"Day-by-day reports: {reports_root.name}")
    rows = "\n".join(
        "<tr>"
        + f"<td><b>{_html(day['date'])}</b></td>"
        + f"<td>{day.get('parsed_scans', '')}</td>"
        + f"<td>{day.get('alignment_scan_rows', '')}</td>"
        + f"<td>{_html(day.get('draft_type') or '')}</td>"
        + "<td>"
        + " | ".join(f"<a href=\"{_html(day[key])}\">{label}</a>" for key, label in (("dashboard", "dashboard + plots"), ("morning_report", "morning report"), ("summary", "overnight summary")) if day.get(key))
        + "</td></tr>"
        for day in days
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{_html(title)}</title>
<style>html{{color-scheme:light;background:#ffffff;}}body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2937;background:#ffffff;}}table{{border-collapse:collapse;width:100%;}}td,th{{border-bottom:1px solid #ddd;padding:0.5rem;text-align:left;}}th{{background:#f4f4f5;}}.muted{{color:#b45309;}}</style></head>
<body><h1>{_html(title)}</h1>
<p class="muted">Newest day first.{(' <a href="story/experiment_story_report.html"><b>Experiment so far (semantic report)</b></a>' if story_html.exists() else '')}</p>
<table><thead><tr><th>Day</th><th>Scans</th><th>Alignment scans</th><th>Morning draft type</th><th>Artifacts</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
    (reports_root / "index.html").write_text(html, encoding="utf-8")
    report = {
        "workflow_name": "experiment-day-index",
        "status": "success",
        "reports_root": str(reports_root),
        "day_count": len(days),
        "order": "reverse_chronological",
        "days": days,
        "index_html": str(reports_root / "index.html"),
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_day_index, json_filename="experiment_day_index.json", markdown_filename="experiment_day_index.md")
    return report


def _markdown_day_index(report: dict[str, Any]) -> str:
    lines = [f"# Experiment Day Index", "", f"Status: `{report.get('status')}` | days: {report.get('day_count', 0)} (newest first)", ""]
    for day in report.get("days") or []:
        lines.append(f"- **{day.get('date')}**: scans={day.get('parsed_scans')} alignment={day.get('alignment_scan_rows')} draft={day.get('draft_type')}")
    if report.get("index_html"):
        lines += ["", f"Browsable index: `{report['index_html']}`"]
    return "\n".join(lines) + "\n"


def _load_alignment_fit_overlays(state_json_path: Any) -> dict[str, dict[str, Any]]:
    """Map scan-file stems to fitted peak parameters from an
    alignment-state-machine report, for overlaying on dashboard plots."""
    if not state_json_path:
        return {}
    path = Path(str(state_json_path)).expanduser()
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    overlays: dict[str, dict[str, Any]] = {}
    for diagnostic in state.get("peak_fit_diagnostics") or []:
        event_id = str(diagnostic.get("event_id") or "")
        stem = Path(event_id.split(":", 1)[-1]).stem if event_id else ""
        if not stem or diagnostic.get("status") != "success":
            continue
        overlays[stem] = {
            "status": diagnostic.get("status"),
            "model_family": diagnostic.get("model_family"),
            "axis": diagnostic.get("axis"),
            "fits": diagnostic.get("fits") or [],
            "heuristic_assessment": diagnostic.get("heuristic_assessment"),
            "human_review_required": True,
        }
    return overlays


def _render_static_dashboard_html(report: dict[str, Any]) -> str:
    records = report.get("scan_records", [])
    rows = "\n".join(
        "<tr data-scan-id=\""
        + _html(record.get("scan_id"))
        + "\" data-kind=\""
        + _html(record.get("measurement_kind"))
        + "\" data-search=\""
        + _html(" ".join(str(record.get(field) or "") for field in ["scan_id", "experiment_id", "date", "title", "measurement_kind", "scan_variable", "path"]))
        + "\">"
        + "".join(f"<td>{_html(_csv_value(record, field))}</td>" for field in ["scan_id", "experiment_id", "date", "title", "measurement_kind", "scan_variable", "row_count", "narrow_feature_count"])
        + "</tr>"
        for record in records[:500]
    )
    candidates = report.get("spurion_likelihood_candidates", [])
    cand_rows = "\n".join(
        f"<tr data-scan-id=\"{_html(item.get('scan_id'))}\"><td>{_html(item.get('scan_id'))}</td><td>{_html(item.get('candidate_type'))}</td><td>{item.get('likelihood_score')}</td><td>{_html(item.get('likelihood_label'))}</td><td>{_html(item.get('path'))}</td></tr>"
        for item in candidates[:100]
    )
    overlay_rows = "\n".join(
        f"<tr><td>{_html(item.get('label'))}</td><td>{_html(item.get('material'))}</td><td>{_html(item.get('overlay_type'))}</td><td>{_html((item.get('parameters') or {}).get('status'))}</td></tr>"
        for item in (report.get("overlay_layers") or [])[:100]
    )
    plot_records = [
        {
            "scan_id": record.get("scan_id"),
            "title": record.get("title") or record.get("scan_id"),
            "kind": record.get("measurement_kind"),
            "axis": record.get("scan_variable"),
            "experiment_id": record.get("experiment_id"),
            "date": record.get("date"),
            "path": record.get("path"),
            "row_count": record.get("row_count"),
            "plot_data": record.get("plot_data") or {},
            "narrow_feature_flags": record.get("narrow_feature_flags") or [],
            "alignment_fit": (record.get("metadata") or {}).get("alignment_fit"),
        }
        for record in records[:300]
        if (record.get("plot_data") or {}).get("x")
    ]
    plot_options = "\n".join(f"<option value=\"{_html(item.get('scan_id'))}\">{_html(item.get('scan_id'))}</option>" for item in plot_records)
    plot_script = f"""
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const plotRecords = {json.dumps(plot_records)};
function drawScan(scanId) {{
  const record = plotRecords.find(item => item.scan_id === scanId) || plotRecords[0];
  if (!record) return;
  const plot = record.plot_data || {{}};
  const trace = {{x: plot.x || [], y: plot.y || [], mode: 'lines+markers', type: 'scatter', name: record.scan_id}};
  const shapes = (record.narrow_feature_flags || []).filter(flag => flag.x !== undefined && flag.x !== null).map(flag => ({{
    type: 'line', xref: 'x', yref: 'paper', x0: flag.x, x1: flag.x, y0: 0, y1: 1,
    line: {{color: '#b91c1c', width: 1, dash: 'dash'}}
  }}));
  const traces = [trace];
  const fit = record.alignment_fit;
  if (fit && Array.isArray(fit.fits) && fit.fits.length && (plot.x || []).length > 1) {{
    const xs = plot.x, ys = plot.y || [0];
    const xmin = Math.min(...xs), xmax = Math.max(...xs), base = Math.min(...ys);
    for (const p of fit.fits) {{
      if (p.center !== undefined && p.center !== null) shapes.push({{
        type: 'line', xref: 'x', yref: 'paper', x0: p.center, x1: p.center, y0: 0, y1: 1,
        line: {{color: '#16a34a', width: 1}}
      }});
    }}
    if (fit.fits.some(p => p.sigma_preview)) {{
      const fx = [], fm = [];
      for (let i = 0; i < 200; i++) {{
        const x = xmin + (xmax - xmin) * i / 199;
        let m = 0;
        for (const p of fit.fits) if (p.sigma_preview) m += p.amplitude * Math.exp(-0.5 * Math.pow((x - p.center) / p.sigma_preview, 2));
        fx.push(x); fm.push(m);
      }}
      // Fit amplitudes are count-rate units; the plot may be
      // monitor-normalized. Positions/widths carry the alignment content, so
      // scale the model curve to the plotted range and say so in the legend.
      const mmax = Math.max(...fm);
      const scale = mmax > 0 ? (Math.max(...ys) - base) / mmax : 1;
      traces.push({{x: fx, y: fm.map(m => base + m * scale), mode: 'lines', type: 'scatter', name: 'fit (scaled): ' + (fit.model_family || 'gaussian'), line: {{color: '#16a34a', width: 2, dash: 'dot'}}}});
    }}
  }}
  Plotly.newPlot('scan-plot', traces, {{
    title: record.title || record.scan_id,
    xaxis: {{title: plot.x_column || 'scan axis'}},
    yaxis: {{title: plot.y_column || 'intensity'}},
    shapes: shapes,
    margin: {{l: 60, r: 20, t: 50, b: 50}}
  }});
  const select = document.getElementById('plot-scan');
  if (select) select.value = record.scan_id;
  document.querySelectorAll('tr[data-scan-id]').forEach(row => row.classList.toggle('selected', row.getAttribute('data-scan-id') === record.scan_id));
  const detail = document.getElementById('selected-scan-detail');
  if (detail) {{
    detail.innerHTML = `<b>${{record.scan_id || ''}}</b><br>${{record.title || ''}}<br><span class="muted">kind=${{record.kind || ''}} axis=${{record.axis || ''}} rows=${{record.row_count || ''}}</span><br><code>${{record.path || ''}}</code>`;
  }}
  window.location.hash = record.scan_id ? `scan=${{encodeURIComponent(record.scan_id)}}` : '';
}}
function copySelectedScanLink() {{
  const scanId = document.getElementById('plot-scan')?.value || '';
  const url = `${{window.location.href.split('#')[0]}}#scan=${{encodeURIComponent(scanId)}}`;
  navigator.clipboard?.writeText(url);
}}
let refreshTimer = null;
function configureAutoRefresh() {{
  const input = document.getElementById('auto-refresh-seconds');
  const status = document.getElementById('auto-refresh-status');
  if (!input || !status) return;
  const seconds = Number(input.value || localStorage.getItem('tas-dashboard-refresh-seconds') || 0);
  input.value = seconds || '';
  if (refreshTimer) clearInterval(refreshTimer);
  if (seconds > 0) {{
    localStorage.setItem('tas-dashboard-refresh-seconds', String(seconds));
    let remaining = seconds;
    status.textContent = `auto-refresh in ${{remaining}} s`;
    refreshTimer = setInterval(() => {{
      remaining -= 1;
      status.textContent = `auto-refresh in ${{remaining}} s`;
      if (remaining <= 0) window.location.reload();
    }}, 1000);
  }} else {{
    localStorage.removeItem('tas-dashboard-refresh-seconds');
    status.textContent = 'auto-refresh off';
  }}
}}
document.addEventListener('DOMContentLoaded', () => {{
  const hashScan = decodeURIComponent((window.location.hash || '').replace(/^#scan=/, ''));
  drawScan(hashScan || (plotRecords[0] || {{}}).scan_id);
  document.getElementById('plot-scan')?.addEventListener('change', event => drawScan(event.target.value));
  document.getElementById('copy-scan-link')?.addEventListener('click', copySelectedScanLink);
  document.getElementById('auto-refresh-seconds')?.addEventListener('change', configureAutoRefresh);
  configureAutoRefresh();
  document.querySelectorAll('tr[data-scan-id]').forEach(row => {{
    row.addEventListener('click', () => drawScan(row.getAttribute('data-scan-id')));
  }});
  document.getElementById('reload-dashboard')?.addEventListener('click', () => window.location.reload());
}});
</script>"""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{_html(report.get('title') or 'Experiment Dashboard')}</title>
<style>
html {{ color-scheme: light; background: #ffffff; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #ffffff; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f0f4f8; position: sticky; top: 0; }}
tbody tr {{ cursor: pointer; }}
tbody tr:hover {{ background: #f8fafc; }}
tbody tr.selected {{ outline: 2px solid #2563eb; outline-offset: -2px; background: #eff6ff; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.metric {{ border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; background: #f8fafc; }}
.warn {{ color: #8a4b00; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; align-items: end; }}
.controls label {{ display: grid; gap: 4px; font-size: 13px; color: #52616f; }}
.controls input, .controls select {{ padding: 6px 8px; border: 1px solid #bcccdc; border-radius: 4px; }}
.muted {{ color: #52616f; font-size: 12px; }}
.detail {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 10px; background: #f8fafc; margin: 8px 0 16px; }}
</style>
</head>
<body>
<h1>{_html(report.get('title') or 'Experiment Dashboard')}</h1>
<div class="summary">
<div class="metric"><b>Scans</b><br>{report.get('scan_count', 0)}</div>
<div class="metric"><b>Inelastic</b><br>{report.get('inelastic_scan_count', 0)}</div>
<div class="metric"><b>Spurion triage candidates</b><br>{len(candidates)}</div>
<div class="metric"><b>Data root</b><br>{_html(report.get('data_root'))}</div>
</div>
<div class="controls">
  <label>Linked plot scan <select id="plot-scan">{plot_options}</select></label>
  <button id="reload-dashboard" type="button">Reload local JSON</button>
  <button id="copy-scan-link" type="button">Copy scan link</button>
  <label>Auto-refresh seconds <input id="auto-refresh-seconds" type="number" min="0" step="1" placeholder="off"></label>
  <span id="auto-refresh-status" class="muted">auto-refresh off</span>
</div>
<div id="selected-scan-detail" class="detail">Select a scan.</div>
<div id="scan-plot" style="height:420px;"></div>
{plot_script}
<h2>Spurion Triage Candidates</h2>
<table><thead><tr><th>Scan</th><th>Type</th><th>Score</th><th>Label</th><th>Path</th></tr></thead><tbody>{cand_rows}</tbody></table>
<h2>Experiment Overlay Hooks</h2>
<p>Overlay rows are experiment-specific references. They are not applied as global assumptions.</p>
<table><thead><tr><th>Label</th><th>Material</th><th>Type</th><th>Status</th></tr></thead><tbody>{overlay_rows}</tbody></table>
<h2>Scan Table</h2>
<div class="controls">
  <label>Text filter <input id="scan-search" type="search" placeholder="scan, title, path, expt"></label>
  <label>Kind <select id="kind-filter"><option value="">all</option><option value="elastic">elastic</option><option value="inelastic">inelastic</option><option value="unknown">unknown</option></select></label>
  <span id="visible-count"></span>
</div>
<table id="scan-table"><thead><tr><th>Scan</th><th>Expt</th><th>Date</th><th>Title</th><th>Kind</th><th>Axis</th><th>Rows</th><th>Narrow</th></tr></thead><tbody>{rows}</tbody></table>
<script>
const search = document.getElementById('scan-search');
const kind = document.getElementById('kind-filter');
const visibleCount = document.getElementById('visible-count');
function applyFilters() {{
  const q = (search.value || '').toLowerCase();
  const k = kind.value;
  let shown = 0;
  document.querySelectorAll('#scan-table tbody tr').forEach(row => {{
    const text = ((row.dataset.search || '') + ' ' + row.innerText).toLowerCase();
    const rowKind = row.dataset.kind || '';
    const ok = (!q || text.includes(q)) && (!k || rowKind === k);
    row.style.display = ok ? '' : 'none';
    if (ok) shown += 1;
  }});
  visibleCount.textContent = `${{shown}} visible`;
}}
search.addEventListener('input', applyFilters);
kind.addEventListener('change', applyFilters);
applyFilters();
</script>
</body>
</html>
"""


def _dash_app_template() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import Input, Output, dcc, dash_table, html
import plotly.graph_objects as go

DATA = Path(__file__).with_name("dashboard_data.json")


def load_data():
    return json.loads(DATA.read_text(encoding="utf-8"))


def scan_table(records, kinds=None, search=""):
    kinds = set(kinds or [])
    search = (search or "").lower()
    rows = []
    for record in records:
        if kinds and record.get("measurement_kind") not in kinds:
            continue
        haystack = " ".join(str(record.get(key, "")) for key in ("scan_id", "experiment_id", "title", "path")).lower()
        if search and search not in haystack:
            continue
        rows.append(
            {
                "scan_id": record.get("scan_id"),
                "experiment_id": record.get("experiment_id"),
                "date": record.get("date"),
                "title": record.get("title"),
                "kind": record.get("measurement_kind"),
                "axis": record.get("scan_variable"),
                "rows": record.get("row_count"),
                "narrow": len(record.get("narrow_feature_flags", [])),
            }
        )
    return rows


def candidate_table(candidates):
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "scan_id": item.get("scan_id"),
            "score": item.get("likelihood_score"),
            "label": item.get("likelihood_label"),
            "type": item.get("candidate_type"),
            "path": item.get("path"),
        }
        for item in candidates
    ]


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Interval(id="refresh", interval=15000, n_intervals=0),
        html.H1(id="title"),
        html.Div(id="summary"),
        html.Div(
            [
                html.Label("Measurement kind"),
                dcc.Dropdown(id="kind-filter", multi=True),
                html.Label("Scan text filter"),
                dcc.Input(id="text-filter", type="text", debounce=True, style={"width": "100%"}),
                html.Label("Scan"),
                dcc.Dropdown(id="scan", clearable=False),
                html.Label("Spurion candidate"),
                dcc.Dropdown(id="candidate", clearable=True),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px"},
        ),
        dcc.Graph(id="plot"),
        html.H2("Spurion triage candidates"),
        dash_table.DataTable(id="candidate-table", page_size=12, filter_action="native", sort_action="native", row_selectable="single", style_table={"overflowX": "auto"}),
        html.H2("Recommended diagnostic scans"),
        dash_table.DataTable(id="diagnostic-table", page_size=12, filter_action="native", sort_action="native", style_table={"overflowX": "auto"}),
        html.H2("Scan table"),
        dash_table.DataTable(id="table", page_size=20, filter_action="native", sort_action="native", style_table={"overflowX": "auto"}),
    ],
    style={"fontFamily": "system-ui", "margin": "24px"},
)


@app.callback(
    Output("title", "children"),
    Output("summary", "children"),
    Output("kind-filter", "options"),
    Output("scan", "options"),
    Output("scan", "value"),
    Output("candidate", "options"),
    Output("candidate-table", "data"),
    Output("diagnostic-table", "data"),
    Input("refresh", "n_intervals"),
)
def refresh(_):
    data = load_data()
    records = data.get("scan_records", [])
    candidates = data.get("spurion_likelihood_candidates", [])
    kinds = sorted({record.get("measurement_kind") for record in records if record.get("measurement_kind")})
    kind_options = [{"label": kind, "value": kind} for kind in kinds]
    options = [{"label": record.get("scan_id"), "value": record.get("scan_id")} for record in records if (record.get("plot_data") or {}).get("x")]
    candidate_options = [{"label": item.get("candidate_id"), "value": item.get("candidate_id")} for item in candidates]
    summary = f"Scans: {data.get('scan_count', 0)} | Inelastic: {data.get('inelastic_scan_count', 0)} | Spurion candidates: {len(data.get('spurion_likelihood_candidates', []))}"
    return data.get("title", "Experiment Dashboard"), summary, kind_options, options, options[0]["value"] if options else None, candidate_options, candidate_table(candidates), data.get("recommended_diagnostic_scans", [])


@app.callback(Output("table", "data"), Input("kind-filter", "value"), Input("text-filter", "value"), Input("refresh", "n_intervals"))
def filter_table(kinds, search, _):
    data = load_data()
    return scan_table(data.get("scan_records", []), kinds=kinds, search=search)


@app.callback(Output("scan", "value", allow_duplicate=True), Input("candidate", "value"), prevent_initial_call=True)
def scan_from_candidate(candidate_id):
    if not candidate_id:
        return dash.no_update
    data = load_data()
    candidate = next((item for item in data.get("spurion_likelihood_candidates", []) if item.get("candidate_id") == candidate_id), None)
    return candidate.get("scan_id") if candidate else dash.no_update


@app.callback(Output("plot", "figure"), Input("scan", "value"), Input("refresh", "n_intervals"))
def plot_scan(scan_id, _):
    data = load_data()
    record = next((item for item in data.get("scan_records", []) if item.get("scan_id") == scan_id), None)
    fig = go.Figure()
    if not record:
        return fig
    plot = record.get("plot_data") or {}
    fig.add_trace(go.Scatter(x=plot.get("x", []), y=plot.get("y", []), mode="lines+markers", name=record.get("scan_id")))
    for flag in record.get("narrow_feature_flags", []):
        if flag.get("x") is not None:
            fig.add_vline(x=flag.get("x"), line_dash="dash", line_color="#b91c1c", annotation_text="narrow")
    fig.update_layout(title=record.get("title") or record.get("scan_id"), xaxis_title=plot.get("x_column"), yaxis_title=plot.get("y_column"))
    return fig


if __name__ == "__main__":
    app.run(debug=True)
'''


def _dashboard_summary(records: list[DashboardScanRecord], candidates: list[SpurionLikelihoodCandidate]) -> dict[str, Any]:
    return {
        "scan_count": len(records),
        "inelastic_scan_count": sum(1 for item in records if item.measurement_kind == "inelastic"),
        "scans_with_narrow_features": sum(1 for item in records if item.narrow_feature_flags),
        "top_spurion_candidates": [item.candidate_id for item in candidates[:10]],
    }


def _overlay_summary(overlays: list[DashboardOverlayLayer]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for overlay in overlays:
        by_type[overlay.overlay_type] = by_type.get(overlay.overlay_type, 0) + 1
        status = str((overlay.parameters or {}).get("status") or "unspecified")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "overlay_count": len(overlays),
        "by_type": by_type,
        "by_status": by_status,
        "policy": "Overlays are local experiment-specific hooks and require reviewed material/geometry inputs before scientific interpretation.",
    }


def _spurion_ranking_diagnostics(candidates: list[SpurionLikelihoodCandidate], review_label_store: str | Path | None = None) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    counter_evidence_counts: dict[str, int] = {}
    for candidate in candidates:
        label_counts[candidate.likelihood_label] = label_counts.get(candidate.likelihood_label, 0) + 1
        for item in candidate.evidence:
            evidence_counts[item.evidence_type] = evidence_counts.get(item.evidence_type, 0) + 1
        for item in candidate.counter_evidence:
            counter_evidence_counts[item.evidence_type] = counter_evidence_counts.get(item.evidence_type, 0) + 1
        for field in candidate.missing_fields:
            missing_counts[field] = missing_counts.get(field, 0) + 1
    review_label_diagnostics = review_label_calibration_diagnostics(
        review_label_store,
        object_type="spurion_candidate",
        source_workflow="survey-inelastic-spurion-evidence",
    )
    label_summary = _label_calibration_summary(candidates, score_key="likelihood_score")
    if review_label_diagnostics.get("status") in {"not_provided", "missing_input"}:
        probability_status = "heuristic_not_calibrated"
    elif label_summary["applied_candidate_count"]:
        probability_status = "weakly_calibrated_from_labels"
    else:
        probability_status = "heuristic_or_label_preview"
    return {
        "candidate_count": len(candidates),
        "label_counts": label_counts,
        "evidence_counts": evidence_counts,
        "counter_evidence_counts": counter_evidence_counts,
        "missing_field_counts": missing_counts,
        "probability_status": probability_status,
        "review_label_diagnostics": review_label_diagnostics,
        "label_calibration_summary": label_summary,
        "calibration_need": "Requires reviewed true-positive/false-positive labels before likelihood scores can be treated as calibrated probabilities.",
    }


def _dashboard_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Dashboard",
        "",
        f"Status: {report.get('status')}",
        f"Data root: `{report.get('data_root')}`",
        f"Scans: {report.get('scan_count', 0)}",
        f"Inelastic scans: {report.get('inelastic_scan_count', 0)}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in report.get("generated_artifacts", {}).items():
        lines.append(f"- {key}: `{value}`")
    if report.get("dashboard_scope"):
        lines.extend(["", "## Dashboard Scope", ""])
        for key, value in report["dashboard_scope"].items():
            lines.append(f"- `{key}`: {value}")
    if report.get("overlay_summary"):
        lines.extend(["", "## Overlay Summary", ""])
        lines.append(f"- Overlay count: {report['overlay_summary'].get('overlay_count')}")
        lines.append(f"- Policy: {report['overlay_summary'].get('policy')}")
    if report.get("spurion_ranking_diagnostics"):
        diag = report["spurion_ranking_diagnostics"]
        lines.extend(["", "## Spurion Ranking Diagnostics", ""])
        lines.append(f"- Probability status: `{diag.get('probability_status')}`")
        lines.append(f"- Label counts: `{diag.get('label_counts')}`")
        lines.append(f"- Calibration need: {diag.get('calibration_need')}")
    lines.extend(["", "## Top Spurion Triage Candidates", ""])
    for item in report.get("spurion_likelihood_candidates", [])[:20]:
        lines.append(f"- `{item.get('candidate_id')}` score={item.get('likelihood_score')} label={item.get('likelihood_label')} path=`{item.get('path')}`")
    lines.extend(["", "## Diagnostic Scan Suggestions", ""])
    for item in report.get("recommended_diagnostic_scans", [])[:20]:
        plan = item.get("suggested_scan", {})
        lines.append(
            f"- `{item.get('recommendation_id')}` p={item.get('probability_of_diagnostic_value')} "
            f"axis={plan.get('axis')} points={plan.get('planned_points')} est_min={plan.get('estimated_duration_minutes')}: {item.get('action')}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("warnings", []))
    return "\n".join(lines) + "\n"


def _spurion_survey_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Inelastic Spurion Evidence Survey",
        "",
        f"Status: {report.get('status')}",
        f"Data root: `{report.get('data_root')}`",
        f"Scans: {report.get('scan_count', 0)}",
        f"Inelastic scans: {report.get('inelastic_scan_count', 0)}",
        f"Candidates: {report.get('candidate_count', 0)}",
        "",
        "## Candidates",
        "",
    ]
    for item in report.get("candidates", [])[:50]:
        evidence = "; ".join(ev.get("summary", "") for ev in item.get("evidence", []))
        lines.append(f"- `{item.get('candidate_id')}` score={item.get('likelihood_score')} label={item.get('likelihood_label')}: {evidence}")
    if report.get("ranking_diagnostics"):
        diag = report["ranking_diagnostics"]
        lines.extend(["", "## Ranking Diagnostics", ""])
        lines.append(f"- Probability status: `{diag.get('probability_status')}`")
        lines.append(f"- Label counts: `{diag.get('label_counts')}`")
        lines.append(f"- Missing-field counts: `{diag.get('missing_field_counts')}`")
        lines.append(f"- Calibration need: {diag.get('calibration_need')}")
    lines.extend(["", "## Recommended Diagnostic Scans", ""])
    for item in report.get("recommended_diagnostic_scans", [])[:30]:
        plan = item.get("suggested_scan", {})
        lines.append(
            f"- `{item.get('recommendation_id')}` p={item.get('probability_of_diagnostic_value')} "
            f"axis={plan.get('axis')} range={plan.get('ranges')} step={plan.get('step')} "
            f"points={plan.get('planned_points')} est_min={plan.get('estimated_duration_minutes')}"
        )
    lines.extend(["", "## Safe Conclusion", "", report.get("safe_conclusion", ""), "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def _overnight_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Overnight Scan Summary",
        "",
        f"Status: {report.get('status')}",
        f"Data root: `{report.get('data_root')}`",
        f"Scans summarized: {report.get('scan_count', 0)}",
        f"Series groups: {report.get('series_count', 0)}",
        "",
        "## Summary",
        report.get("safe_conclusion") or "",
        "",
        "## Scan Family Plots",
    ]
    plotted_any = False
    for item in report.get("series", []):
        entries = item.get("plots") or ([{"label": item.get("series_id"), "png": item["plot_png"]}] if item.get("plot_png") else [])
        if not entries:
            continue
        lines.append(f"### {item.get('series_id')}")
        for entry in entries:
            lines.append(f"![{entry.get('label')}]({entry.get('png')})")
        for note in item.get("software_change_notes") or []:
            lines.append(f"- Software change (console log): `{note}`")
        plotted_any = True
    if not plotted_any:
        lines.append("(no plottable scan data or matplotlib unavailable)")
    lines.extend([
        "",
        "## Artifacts",
    ])
    for key, value in report.get("generated_artifacts", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Suggested Report Types",
    ])
    for item in report.get("suggested_reports", []):
        lines.append(f"- `{item.get('report_type')}` p={item.get('probability')}: {item.get('why')}")
        diagnostics = item.get("probability_diagnostics") or {}
        if diagnostics:
            lines.append(
                f"  Probability basis: {diagnostics.get('matching_scan_count', 0)}/"
                f"{diagnostics.get('total_scan_count', 0)} scans in matching families."
            )
        sections = ", ".join(item.get("include_sections", []))
        if sections:
            lines.append(f"  Include: {sections}")
        adjustment = item.get("scan_corpus_adjustment") or {}
        if adjustment.get("status") == "applied":
            lines.append(
                f"  Scan-corpus adjustment: +{adjustment.get('adjustment')} from "
                f"{adjustment.get('support_count')} comparable private/local scan-family precedent count(s)."
            )
    corpus = report.get("scan_corpus_context") or {}
    lines.extend(["", "## Scan Corpus Context"])
    lines.append(f"- Status: `{corpus.get('status')}`")
    if corpus.get("index_path"):
        lines.append(f"- Index: `{corpus.get('index_path')}`")
    if corpus.get("record_count") is not None:
        lines.append(f"- Records inspected: `{corpus.get('record_count')}`")
    if corpus.get("matched_families"):
        lines.append(f"- Matched families: {', '.join(corpus.get('matched_families') or [])}")
    if corpus.get("user_facing_policy"):
        lines.append(f"- Policy: {corpus.get('user_facing_policy')}")
    lines.extend(["", "## Next Actions"])
    for item in report.get("next_actions", []):
        prefix = f"[{item.get('priority')}] " if item.get("priority") else ""
        probability = item.get("probability_of_value")
        suffix = f" p={probability}" if probability is not None else ""
        lines.append(f"- {prefix}{item.get('summary')}{suffix}")
        if item.get("probability_basis"):
            lines.append(f"  Basis: {item.get('probability_basis')}")
    lines.extend(["", "## Scan Families"])
    for item in report.get("series", [])[:30]:
        energy = item.get("energy_range") or {}
        temp = item.get("temperature_range") or {}
        details = []
        if energy:
            details.append(f"E={energy.get('min')}-{energy.get('max')} meV")
        if temp:
            details.append(f"T={temp.get('min')}-{temp.get('max')} K")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(
            f"- `{item.get('family')}` scans={item.get('scan_count')} points={item.get('point_count')} "
            f"confidence={item.get('confidence')} report={item.get('probable_report_type')}{suffix}"
        )
    lines.extend(["", "## Time Filter"])
    filt = report.get("time_filter") or {}
    lines.append(f"- mode: {filt.get('mode')}")
    lines.append(f"- cutoff: {filt.get('cutoff')}")
    lines.append(f"- undated scans kept: {filt.get('undated_scans_kept', 0)}")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {item}" for item in report.get("warnings", []))
    return "\n".join(lines) + "\n"


def _missing(output_dir: Path, workflow_name: str, fields: list[str], message: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"workflow_name": workflow_name, "status": "missing_input", "missing_fields": fields, "warnings": [message]}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=lambda item: f"# {workflow_name}\n\nStatus: missing_input\n\n- {message}\n")
    return report


def _numeric(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _mean_numeric(rows: list[dict[str, Any]], column: str) -> float | None:
    values = [_numeric(row.get(column)) for row in rows]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _range_numeric(rows: list[dict[str, Any]], column: str) -> dict[str, float]:
    values = [_numeric(row.get(column)) for row in rows]
    values = [float(value) for value in values if value is not None]
    if not values:
        return {}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _small_header(header: dict[str, str]) -> dict[str, str]:
    keys = ["Filename", "Date", "InstrName", "ExptID", "ExptName", "ExptParticipants", "ExptDetails", "ScanType", "Scan", "ScanDescr", "AnalyzerDetectorMode", "FixedE", "Lattice", "Orient"]
    return {key: header[key] for key in keys if key in header}


def _scan_title(header: dict[str, str]) -> str | None:
    descr = header.get("ScanDescr") or ""
    match = re.search(r"Title=([^:]+)", descr)
    return match.group(1).strip() if match else header.get("Comment") or header.get("Filename")


def _material_clues(header: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, bool]:
    text = " ".join(str(value) for value in header.values()).lower()
    clues = {material: any(term in text for term in terms) for material, terms in MATERIAL_TERMS.items()}
    filter_state = _filter_state_summary(header, rows)
    if str(filter_state.get("state") or "").upper() == "IN":
        clues["PG"] = True
        clues["filter_in"] = True
    elif str(filter_state.get("state") or "").upper() == "OUT":
        clues["filter_out"] = True
    velocity_selector = _velocity_selector_state(header, rows)
    if velocity_selector.get("state") in {"in", "on", "present"}:
        clues["velocity_selector"] = True
    if any("Velsel" in row for row in rows[:3]):
        clues["velocity_selector"] = True
    return clues


def _constantish_column(rows: list[dict[str, Any]], names: tuple[str, ...], *, sample_limit: int = 50) -> dict[str, Any]:
    for name in names:
        values = [row.get(name) for row in rows[:sample_limit] if name in row and row.get(name) not in {None, ""}]
        if not values:
            continue
        normalized = [str(value).strip() for value in values]
        counts = {value: normalized.count(value) for value in sorted(set(normalized))}
        dominant = max(counts, key=counts.get)
        return {
            "column": name,
            "value": dominant,
            "constant_within_sample": len(counts) == 1,
            "sampled_point_count": len(normalized),
            "distinct_values": counts,
        }
    return {}


def _typed_harmonic_state_metadata(header: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Typed harmonic filter/velocity-selector fields for scoring layers.

    BT-7 column-name knowledge stays here in the reading layer; scoring code
    consumes only harmonic_filter_translation (IN|OUT|unknown),
    velocity_selector_speed (float|None), and incident_energy_mev (float|None).
    """

    filter_summary = _filter_state_summary(header, rows)
    state = str(filter_summary.get("state") or "").strip().upper()
    velocity_summary = _velocity_selector_state(header, rows)
    speed = None
    raw_speed = velocity_summary.get("raw_value")
    if raw_speed is not None:
        try:
            speed = float(str(raw_speed))
        except ValueError:
            speed = None
    ei = _mean_numeric(rows, "Ei") if any("Ei" in row for row in rows[:3]) else None
    return {
        "harmonic_filter_translation": state if state in {"IN", "OUT"} else "unknown",
        "velocity_selector_speed": speed,
        "incident_energy_mev": ei,
    }


def _filter_state_summary(header: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    column = _constantish_column(rows, ("FilTran", "Filtran", "filterTrans", "FilterTrans", "filterControl", "Filter", "PGFilter"))
    if column:
        state = str(column["value"]).strip().upper()
        material = "PG" if state in {"IN", "OUT"} else None
        return {
            "status": "observed_in_scan_columns",
            "state": state,
            "material": material,
            "source": column,
            "interpretation": "BT-7 FilTran is treated as in-pile PG-filter state when present; it shifts lambda/2 probability but is not conclusive by itself.",
        }
    text = " ".join(str(value) for value in header.values()).lower()
    if "pg filter" in text or "graphite filter" in text:
        return {
            "status": "text_hint_only",
            "state": "UNKNOWN",
            "material": "PG",
            "source": "header_text",
            "interpretation": "Header text mentions a PG/graphite filter but does not provide a reviewed in/out state.",
        }
    return {
        "status": "missing",
        "state": "UNKNOWN",
        "material": None,
        "source": "not_observed",
        "interpretation": "Filter state is missing; historic files may not record manual post-sample filters.",
    }


def _velocity_selector_state(header: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    column = _constantish_column(
        rows,
        (
            "Velsel",
            "VelSel",
            "VelocitySelector",
            "velocitySelector",
            "velocitySelectorTrans",
            "velocitySelectorTransMotor",
            "VSel",
        ),
    )
    if column:
        raw = str(column["value"]).strip()
        lowered = raw.lower()
        state = "in" if lowered in {"in", "on", "1", "true", "yes"} else "out" if lowered in {"out", "off", "0", "false", "no"} else "present"
        return {
            "status": "observed_in_scan_columns",
            "state": state,
            "raw_value": raw,
            "source": column,
            "interpretation": "Velocity selector evidence lowers lambda/2 probability when it is in/on, but does not replace material/filter geometry review.",
        }
    text = " ".join(str(value) for value in header.values()).lower()
    if "velocity selector" in text or "velsel" in text:
        return {
            "status": "text_hint_only",
            "state": "present",
            "raw_value": None,
            "source": "header_text",
            "interpretation": "Velocity selector is mentioned in text but no reviewed in/out state was found.",
        }
    return {
        "status": "missing",
        "state": "unknown",
        "raw_value": None,
        "source": "not_observed",
        "interpretation": "Velocity selector state is missing or not relevant for this file.",
    }


def _collimation_hint(header: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in ("PreMonoColl", "PostMonoColl", "PreAnaColl", "PostAnaColl"):
        if rows:
            value = _numeric(rows[0].get(key))
            if value is not None:
                values[key] = value
    if not values and "collimation" in " ".join(header.values()).lower():
        values["text_hint"] = True
    return values


def _experiment_review_order_parameter_bundle(order_triage: dict[str, Any], *, output_dir: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    if order_triage.get("status") != "ok":
        return {
            "workflow_name": "experiment-review-order-parameter-followup",
            "status": "missing_input",
            "summary": "No order-parameter candidates were available for trajectory clustering or gallery generation.",
            "missing_fields": ["order_parameter_candidates"],
        }
    triage_json = (order_triage.get("artifacts") or {}).get("json") or str(output_dir / "order_parameter_triage" / "order_parameter_triage.json")
    try:
        from tas_copilot_core.workflows.trajectory_clustering import cluster_order_parameter_trajectories, order_parameter_trajectory_report, plot_order_parameter_candidate_scans

        trajectory = cluster_order_parameter_trajectories(
            {
                "triage_json": triage_json,
                "output_dir": str(output_dir / "trajectory_clusters"),
                "observable": str(arguments.get("trajectory_observable") or "integrated_signal"),
                "backend": str(arguments.get("trajectory_backend") or "shape_rules"),
            }
        )
        trajectory_json = (trajectory.get("artifacts") or {}).get("json") or str(output_dir / "trajectory_clusters" / "trajectory_clustering_report.json")
        combined = order_parameter_trajectory_report(
            {
                "triage_json": triage_json,
                "trajectory_json": trajectory_json,
                "output_dir": str(output_dir / "order_parameter_trajectory_report"),
                "top_k": int(arguments.get("top_k", 8)),
            }
        )
        combined_json = (combined.get("artifacts") or {}).get("json") or str(output_dir / "order_parameter_trajectory_report" / "order_parameter_trajectory_report.json")
        plots = plot_order_parameter_candidate_scans(
            {
                "report_json": combined_json,
                "output_dir": str(output_dir / "order_parameter_candidate_gallery"),
                "top_k": int(arguments.get("plot_top_k", min(int(arguments.get("top_k", 8)), 5))),
                "max_scans_per_candidate": int(arguments.get("max_scans_per_candidate", 12)),
                "experiment_id": str(arguments.get("experiment_id") or Path(str(order_triage.get("data_root") or "experiment")).name),
                "allow_rebin": bool(arguments.get("allow_rebin")),
                "mesh_grid_points": int(arguments.get("mesh_grid_points", 120)),
            }
        )
        fit_handoff = _order_parameter_fit_handoff(combined, triage_json=triage_json, output_dir=output_dir)
        return {
            "workflow_name": "experiment-review-order-parameter-followup",
            "status": "ok",
            "trajectory": trajectory,
            "combined_report": combined,
            "plots": plots,
            "fit_handoff": fit_handoff,
            "summary": f"Prepared trajectory labels, candidate gallery, and {len(fit_handoff.get('commands', []))} fit handoff command(s).",
        }
    except Exception as exc:  # pragma: no cover - defensive fail-closed composition.
        return {
            "workflow_name": "experiment-review-order-parameter-followup",
            "status": "error",
            "summary": f"{type(exc).__name__}: {exc}",
            "warnings": ["Order-parameter follow-up failed closed; inspect triage report and inputs before retrying."],
        }


def _order_parameter_fit_handoff(combined: dict[str, Any], *, triage_json: str, output_dir: Path) -> dict[str, Any]:
    commands = []
    for item in (combined.get("top_candidates") or [])[:5]:
        tier = str(item.get("review_tier") or "")
        if not tier.startswith("fit_ready"):
            continue
        candidate_id = str(item.get("candidate_id"))
        backend = "panweb_global" if "split" in tier else "core_preview"
        commands.append(
            {
                "candidate_id": candidate_id,
                "review_tier": tier,
                "probability": item.get("combined_priority"),
                "status": "proposed_fit_review",
                "command": (
                    "python -m tas_copilot_core.cli workflow fit-order-parameter "
                    f"--triage-json {shlex.quote(triage_json)} "
                    f"--candidate-id {shlex.quote(candidate_id)} "
                    f"--output-dir {shlex.quote(str(output_dir / 'order_parameter_fits' / _safe_id(candidate_id)))} "
                    f"--fit-backend {backend}"
                ),
                "reason": "Candidate has high order-parameter trajectory priority and enough scans for controlled fit review.",
                "human_review_required": True,
            }
        )
    return {
        "status": "ok" if commands else "missing_input",
        "commands": commands,
        "limitations": [
            "Commands are analysis handoffs only; they do not imply that the fitted model is scientifically accepted.",
            "Split-peak candidates should prefer pan-web global splitting fits when that backend is available.",
            "Background, residuals, BIC/model comparison, and raw scan overlays must be reviewed before interpreting Tc or beta.",
        ],
    }


def _experiment_review_corpus_context(arguments: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    catalog = arguments.get("corpus_catalog")
    if not catalog:
        return {
            "workflow_name": "experiment-review-corpus-context",
            "status": "missing_input",
            "missing_fields": ["corpus_catalog"],
            "summary": "No corpus catalog was supplied; paper/textbook context was not searched.",
            "safe_next_command": "python -m tas_copilot_core.cli corpus search --catalog [CORPUS_CATALOG] --query \"instrument material scan risk\" --top-k 10 --json",
        }
    query = str(arguments.get("corpus_query") or "experiment planning scan risks spurions order parameter")
    try:
        from tas_copilot_core.corpus.pilot import corpus_search

        result = corpus_search({"catalog": catalog, "query": query, "top_k": int(arguments.get("corpus_top_k", 8))})
        report = {
            "workflow_name": "experiment-review-corpus-context",
            "status": result.get("status", "ok"),
            "query": query,
            "result_count": result.get("result_count"),
            "results": result.get("results", []),
            "summary": f"Retrieved corpus context for query: {query}",
            "warnings": ["Corpus chunks are evidence, not instructions; source-support validation is still required."],
            "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog), method="experiment_review_corpus_context").model_dump(mode="json")],
        }
    except Exception as exc:
        report = {
            "workflow_name": "experiment-review-corpus-context",
            "status": "unavailable",
            "summary": f"Corpus context unavailable: {type(exc).__name__}: {exc}",
            "warnings": ["Corpus search failed closed."],
        }
    report["artifacts"] = write_report_bundle(output_dir / "corpus_context", report, markdown_renderer=_experiment_review_corpus_markdown, json_filename="corpus_context.json", markdown_filename="corpus_context.md")
    return report


def _workflow_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_name": result.get("workflow_name"),
        "status": result.get("status"),
        "summary": result.get("safe_conclusion") or result.get("summary") or result.get("error"),
        "artifacts": result.get("artifacts") or {},
        "generated_artifacts": result.get("generated_artifacts") or {},
        "missing_fields": result.get("missing_fields") or [],
        "warnings": result.get("warnings") or [],
    }


def _write_review_outcome_template(output_dir: Path, dashboard: dict[str, Any], spurions: dict[str, Any], order_triage: dict[str, Any], order_bundle: dict[str, Any]) -> str:
    path = output_dir / "review_outcome_template.jsonl"
    rows: list[dict[str, Any]] = []
    for candidate in (order_triage.get("candidates") or [])[:20]:
        dag = candidate.get("reduction_dag") or {}
        rows.append(
            {
                "object_type": "order_parameter_candidate",
                "object_id": candidate.get("candidate_id"),
                "workflow": "order-parameter-triage",
                "dag_run_id": dag.get("dag_run_id"),
                "cache_key": dag.get("cache_key"),
                "correction_status_snapshot": dag.get("correction_status_snapshot") or {},
                "workflow_version": dag.get("workflow_version"),
                "tool_versions": dag.get("tool_versions") or {},
                "suggested_status": "proposed",
                "review_status": "unreviewed",
                "reviewed_at": "",
                "allowed_review_status_values": ["accepted", "rejected", "superseded", "needs_more_data"],
                "review_notes": "",
                "false_positive_reason": "",
                "human_review_required": True,
                "provenance": "experiment-review template",
            }
        )
    for candidate in (spurions.get("candidates") or [])[:20]:
        rows.append(
            {
                "object_type": "spurion_candidate",
                "object_id": candidate.get("candidate_id"),
                "workflow": "survey-inelastic-spurion-evidence",
                "suggested_status": "proposed",
                "review_status": "unreviewed",
                "allowed_review_status_values": ["likely_real_feature", "likely_spurion", "ambiguous", "needs_more_data"],
                "review_notes": "",
                "false_positive_reason": "",
                "human_review_required": True,
                "provenance": "experiment-review template",
            }
        )
    for command in ((order_bundle.get("fit_handoff") or {}).get("commands") or [])[:10]:
        rows.append(
            {
                "object_type": "fit_handoff",
                "object_id": command.get("candidate_id"),
                "workflow": "fit-order-parameter",
                "suggested_status": "proposed",
                "review_status": "unreviewed",
                "allowed_review_status_values": ["fit_accepted_for_review", "fit_rejected", "model_mismatch", "did_not_converge", "needs_alternate_model"],
                "review_notes": "",
                "false_positive_reason": "",
                "human_review_required": True,
                "provenance": "experiment-review template",
            }
        )
    if not rows:
        rows.append(
            {
                "object_type": "experiment_review",
                "object_id": dashboard.get("dashboard_id"),
                "workflow": "experiment-review",
                "suggested_status": "no_candidates",
                "review_status": "unreviewed",
                "allowed_review_status_values": ["no_action", "needs_more_data", "parser_issue"],
                "review_notes": "",
                "false_positive_reason": "",
                "human_review_required": True,
                "provenance": "experiment-review template",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_experiment_review_label_store(output_dir, path, rows)
    return str(path)


def _write_experiment_review_label_store(output_dir: Path, template_path: Path, rows: list[dict[str, Any]]) -> str:
    store_path = output_dir / "review_outcomes.sqlite"
    source_hash = _file_sha256(template_path)
    labels = [_experiment_review_row_to_label(row, template_path=template_path, source_hash=source_hash) for row in rows]
    batch = ReviewLabelBatch(
        batch_id=f"review-label-batch-{_stable_hash(str(template_path))}",
        source_template_paths=[str(template_path)],
        labels=labels,
        provenance=[ProvenanceRecord(source_type="review_template", source_id=str(template_path), method="experiment_review_dual_write")],
    )
    ReviewOutcomeSQLiteStore(store_path).upsert_batch(batch)
    return str(store_path)


def _experiment_review_row_to_label(row: dict[str, Any], *, template_path: Path, source_hash: str) -> ReviewLabel:
    object_type = str(row.get("object_type") or "experiment_review")
    status = str(row.get("review_status") or "unreviewed")
    return ReviewLabel(
        label_id=f"review-label-{_stable_hash('|'.join([str(template_path), str(row.get('object_type')), str(row.get('object_id')), str(row.get('workflow'))]))}",
        object_type=object_type,
        object_id=str(row.get("object_id") or "unknown"),
        source_workflow=str(row.get("workflow") or "experiment-review"),
        source_artifact_path=str(template_path),
        source_artifact_hash=source_hash,
        verdict=_experiment_review_status_to_verdict(object_type, status),
        type_specific_status=None if status == "unreviewed" else status,
        type_specific_allowed=[str(item) for item in row.get("allowed_review_status_values") or []],
        claim_state=_review_status_to_claim_state(object_type, status),
        suggested_status=str(row.get("suggested_status") or "proposed"),
        # Reviewer-selected channels are stronger evidence than object-type
        # defaults; defaults only seed blank dashboard review rows.
        evidence_channels_relied_on=[
            str(item)
            for item in (
                row.get("evidence_channels_relied_on")
                or row.get("evidence_channels")
                or _experiment_review_evidence_channels(row)
            )
        ],
        counter_evidence_observed=[],
        recommendation="no_action",
        uncertainty_class="probability_style_triage_prior",
        false_positive_reason=row.get("false_positive_reason") or None,
        review_notes=str(row.get("review_notes") or ""),
        workflow_version=row.get("workflow_version"),
        tool_versions={str(key): str(value) for key, value in (row.get("tool_versions") or {}).items()},
        correction_status_snapshot=row.get("correction_status_snapshot") or {},
        dag_run_id=row.get("dag_run_id"),
        cache_key=row.get("cache_key"),
        cache_scope="full_dag" if row.get("cache_key") else None,
        legacy_provenance=row.get("provenance"),
        provenance=[ProvenanceRecord(source_type="review_template", source_id=str(template_path), method="experiment_review_row_to_label")],
    )


def _experiment_review_status_to_verdict(object_type: str, status: str) -> str:
    if object_type == "spurion_candidate":
        if status == "likely_spurion":
            return "positive"
        if status == "likely_real_feature":
            return "negative"
        if status in {"ambiguous", "needs_more_data"}:
            return "inconclusive"
        return "unreviewed"
    if status in {"accepted", "fit_accepted_for_review"}:
        return "positive"
    if status in {"rejected", "fit_rejected", "model_mismatch", "did_not_converge"}:
        return "negative"
    if status in {"superseded"}:
        return "superseded"
    if status in {"needs_more_data", "ambiguous", "needs_alternate_model", "parser_issue", "no_action"}:
        return "inconclusive"
    return "unreviewed"


def _review_status_to_claim_state(object_type: str, status: str) -> str:
    if object_type == "spurion_candidate":
        if status == "likely_spurion":
            return "accepted"
        if status == "likely_real_feature":
            return "rejected"
        return "proposed"
    if status in {"accepted", "fit_accepted_for_review"}:
        return "accepted"
    if status in {"rejected", "fit_rejected", "model_mismatch", "did_not_converge"}:
        return "rejected"
    if status == "superseded":
        return "superseded"
    return "proposed"


def _experiment_review_evidence_channels(row: dict[str, Any]) -> list[str]:
    """Default evidence channels for unedited review rows.

    Reviewer-provided ``evidence_channels_relied_on`` values are preserved by
    ``_experiment_review_row_to_label`` and should not be overwritten here.
    """
    object_type = str(row.get("object_type") or "")
    if object_type == "order_parameter_candidate":
        return ["evidence_channel_scan_family", "evidence_channel_repeatability", "evidence_channel_reconstruction"]
    if object_type == "spurion_candidate":
        return ["evidence_channel_scan_family", "evidence_channel_filter_state", "evidence_channel_collimation"]
    if object_type == "fit_handoff":
        return ["evidence_channel_reconstruction"]
    return []


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _experiment_review_probability_model() -> dict[str, Any]:
    return {
        "model_id": "experiment_review_probability_priors_v0",
        "status": "heuristic_priors",
        "calibration_status": "not_statistically_calibrated",
        "interpretation": "Scores are triage probabilities for where human review is likely valuable, not probabilities that a scientific claim is true.",
        "confidence_bands": {
            "high": "usually worth immediate human review, not a true-positive guarantee",
            "medium": "worth reviewing after stronger candidates or when scientifically relevant",
            "low": "retain for provenance, but do not lead with it",
            "insufficient_evidence": "missing metadata or sparse scan evidence dominates",
        },
        "channels": [
            "scan-family classification",
            "trajectory shape",
            "temperature coverage",
            "raw profile narrowness",
            "scan-family repeatability",
            "sample-environment/filter clues",
            "optional corpus/source-support context",
        ],
        "not_used_as_certainty": [
            "No automatic source-data acceptance.",
            "No automatic real/artifact conclusion.",
            "No autonomous instrument operation.",
        ],
        "calibration_plan": [
            "Track accepted, rejected, and superseded candidates from human review.",
            "Record whether follow-up fits converged and whether residuals/BIC support the proposed model.",
            "Record false positives from substrate, mount, spurion, and sparse-metadata cases.",
            "Use known-answer experiment reviews to tune channel weights before presenting probabilities as calibrated.",
        ],
    }


def _experiment_review_calibration_diagnostics(
    dashboard: dict[str, Any],
    overnight: dict[str, Any],
    spurions: dict[str, Any],
    order_triage: dict[str, Any],
    order_bundle: dict[str, Any],
    corpus_context: dict[str, Any],
    *,
    review_label_store: str | Path | None = None,
) -> dict[str, Any]:
    candidate_count = int(order_triage.get("candidate_count") or 0)
    spurion_count = int(spurions.get("candidate_count") or 0)
    scan_count = int(dashboard.get("scan_count") or 0)
    fit_commands = (order_bundle.get("fit_handoff") or {}).get("commands") or []
    quality_flags = []
    if scan_count == 0:
        quality_flags.append("no_scan_records")
    if candidate_count and not fit_commands:
        quality_flags.append("order_parameter_candidates_not_fit_ready")
    if corpus_context.get("status") not in {"ok", "success"}:
        quality_flags.append("no_corpus_context")
    if spurion_count:
        quality_flags.append("spurion_candidates_require_followup")
    if not quality_flags:
        quality_flags.append("basic_review_evidence_present")
    return {
        "status": "heuristic_review_diagnostics",
        "scan_count": scan_count,
        "series_count": overnight.get("series_count"),
        "order_parameter_candidate_count": candidate_count,
        "fit_handoff_count": len(fit_commands),
        "spurion_candidate_count": spurion_count,
        "corpus_context_status": corpus_context.get("status"),
        "quality_flags": quality_flags,
        "human_review_required": True,
        "review_label_store_diagnostics": {
            "order_parameter": review_label_calibration_diagnostics(
                review_label_store,
                object_type="order_parameter_candidate",
                source_workflow="order-parameter-triage",
            ),
            "spurion": review_label_calibration_diagnostics(
                review_label_store,
                object_type="spurion_candidate",
                source_workflow="survey-inelastic-spurion-evidence",
            ),
        },
        "limitations": [
            "These diagnostics describe evidence completeness, not scientific truth.",
            "A workflow status of ok can coexist with non-converged fits, sparse metadata, or missing corpus context.",
            "The next calibration step is to persist human review outcomes and fit-quality outcomes as labeled cases.",
        ],
    }


def _experiment_review_label_informed_priors(review_label_store: str | Path | None) -> dict[str, Any]:
    return {
        "status": "rendered_without_behavior_change",
        "behavior_change": "none",
        "order_parameter": label_informed_priors(
            review_label_store,
            object_type="order_parameter_candidate",
            source_workflow="order-parameter-triage",
        ),
        "spurion": label_informed_priors(
            review_label_store,
            object_type="spurion_candidate",
            source_workflow="survey-inelastic-spurion-evidence",
        ),
        "fit_handoff": label_informed_priors(
            review_label_store,
            object_type="fit_handoff",
            source_workflow="fit-order-parameter",
        ),
        "interpretation": (
            "These are per-evidence-channel human-review priors from the unified label store. "
            "They are shown beside v0 priors and do not alter ranking behavior yet."
        ),
    }


def _experiment_review_next_steps(dashboard: dict[str, Any], overnight: dict[str, Any], spurions: dict[str, Any], order_triage: dict[str, Any], order_bundle: dict[str, Any], corpus_context: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    gallery = (((order_bundle.get("plots") or {}).get("generated_artifacts") or {}).get("gallery_html"))
    if gallery:
        steps.append({"priority": 0.92, "action_type": "review_order_parameter_gallery", "action": "Open the candidate gallery and inspect raw overlays and temperature meshes.", "artifact": gallery, "provenance": "order_parameter_followup"})
    fit_commands = (order_bundle.get("fit_handoff") or {}).get("commands") or []
    if fit_commands:
        steps.append({"priority": 0.86, "action_type": "run_controlled_order_parameter_fit", "action": "Run the top proposed fit command after raw profile review.", "command": fit_commands[0]["command"], "provenance": "order_parameter_fit_handoff"})
    diagnostic = dashboard.get("recommended_diagnostic_scans") or spurions.get("recommended_diagnostic_scans") or []
    if diagnostic:
        steps.append({"priority": 0.78, "action_type": "diagnostic_spurion_check", "action": "Use the ranked diagnostic scan suggestion to test narrow-feature/spurion hypotheses.", "details": diagnostic[0], "provenance": "dashboard_spurion_triage"})
    next_actions = overnight.get("next_actions") or []
    if next_actions:
        steps.append({"priority": 0.72, "action_type": "overnight_followup", "action": "Review overnight scan-family next actions before user-facing interpretation.", "details": next_actions[0], "provenance": "overnight_scan_summary"})
    if corpus_context.get("status") in {"ok", "success"}:
        steps.append({"priority": 0.58, "action_type": "review_corpus_context", "action": "Review retrieved corpus chunks as supporting evidence, not instructions.", "artifact": (corpus_context.get("artifacts") or {}).get("markdown"), "provenance": "corpus_context"})
    else:
        steps.append({"priority": 0.35, "action_type": "optional_corpus_context", "action": "Run corpus search with a pilot/private corpus catalog if literature or textbook grounding is needed.", "command": corpus_context.get("safe_next_command"), "provenance": "corpus_context_missing"})
    return sorted(steps, key=lambda item: item["priority"], reverse=True)


def _experiment_review_markdown(report: dict[str, Any]) -> str:
    lines = ["# Experiment Review", "", f"Status: {report.get('status')}", "", "## Safe Conclusion", str(report.get("safe_conclusion") or ""), ""]
    lines.extend(["## Workflow Status", "", "| workflow | status | summary |", "| --- | --- | --- |"])
    for name, item in (report.get("workflow_results") or {}).items():
        lines.append(f"| {name} | {item.get('status')} | {str(item.get('summary') or '').replace('|', '/')} |")
    lines.append("")
    lines.extend(["## Ranked Next Steps", "", "| priority | action | provenance |", "| --- | --- | --- |"])
    for step in report.get("ranked_next_steps") or []:
        lines.append(f"| {step.get('priority')} | {step.get('action')} | {step.get('provenance')} |")
    lines.append("")
    diagnostics = report.get("calibration_diagnostics") or {}
    if diagnostics:
        lines.extend(["## Calibration Diagnostics", ""])
        lines.append(f"- Status: `{diagnostics.get('status')}`")
        lines.append(f"- Quality flags: {', '.join(diagnostics.get('quality_flags') or [])}")
        lines.append("")
    priors = report.get("label_informed_priors") or {}
    if priors:
        lines.extend(["## Label-Informed Priors", ""])
        lines.append(f"- Behavior change: `{priors.get('behavior_change')}`")
        for key in ("order_parameter", "spurion", "fit_handoff"):
            item = priors.get(key) or {}
            lines.append(f"- `{key}`: {item.get('status')} reviewed={item.get('reviewed_label_count', 0)}")
        lines.append("")
    fit_commands = (report.get("fit_handoff") or {}).get("commands") or []
    if fit_commands:
        lines.extend(["## Fit Handoff Commands", ""])
        for item in fit_commands:
            lines.append(f"- `{item.get('candidate_id')}` ({item.get('review_tier')}, probability {item.get('probability')}):")
            lines.append("")
            lines.append("```bash")
            lines.append(str(item.get("command")))
            lines.append("```")
            lines.append("")
    lines.extend(["## Key Artifacts", ""])
    for key, value in (report.get("generated_artifacts") or {}).items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _experiment_review_corpus_markdown(report: dict[str, Any]) -> str:
    lines = ["# Experiment Review Corpus Context", "", f"Status: {report.get('status')}", "", str(report.get("summary") or ""), ""]
    for item in (report.get("results") or [])[:10]:
        lines.append(f"- `{item.get('source_id') or item.get('chunk_id')}` score={item.get('score')}: {item.get('snippet') or str(item.get('text') or '')[:160]}")
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report["warnings"])
    return "\n".join(lines)


def _path_year(path: Path) -> int | None:
    for part in path.parts:
        if re.fullmatch(r"20\d{4}", part):
            return int(part[:4])
        if re.fullmatch(r"20\d{2}", part):
            return int(part)
    return None


def _csv_value(record: dict[str, Any], field: str) -> Any:
    if field == "narrow_feature_count":
        return len(record.get("narrow_feature_flags", []))
    return record.get(field)


def _html(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower() or "dashboard"
