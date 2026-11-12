from __future__ import annotations

import html
import json
import csv
import math
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.material_identity_registry import material_identity_record_via_registry
from tas_copilot_core.reduction_dag import (
    constant_q_energy_scan_preview_template,
    execute_reduction_dag,
    order_parameter_temperature_series_template,
    psd_channel_temperature_preview_template,
)
from tas_copilot_core.psd_calibration import reviewed_psd_channel_to_a4_mapping
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.experiment_dashboard import build_experiment_dashboard
from tas_copilot_core.workflows.experiment_planning_brief import experiment_planning_brief


def experiment_workspace(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a local, read-only experimenter workspace around one data root.

    This is an orchestration layer over existing dashboard and planning workflows.
    It does not control instruments and does not make source-data claims.
    """

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    material = str(arguments.get("material") or arguments.get("formula") or "").strip()
    if not material:
        report = _missing(output_dir, ["material"], "A material or formula is required.")
        return report

    data_root = _resolve_path(arguments.get("data_root"))
    material_identity = material_identity_record_via_registry(
        material,
        family=str(arguments.get("family") or "lu_femn_o3"),
        source_field="workspace.material",
        caller_workflow="experiment-workspace",
    )

    dashboard = _run_dashboard(arguments, data_root, output_dir / "dashboard")
    planning = _run_planning(arguments, material, output_dir / "planning")
    recipes = _rank_scan_recipes(_collect_scan_recipes(planning), str(arguments.get("goal") or ""), dashboard)
    nice_replay_training_plan = _collect_nice_replay_training_plan(recipes, planning, dashboard, arguments, output_dir)
    recommendations = _collect_recommendations(planning, dashboard, recipes)
    review_queue = _collect_review_queue(material_identity.model_dump(mode="json"), planning, dashboard, recipes)
    artifact_index = _collect_artifact_index(dashboard, planning)
    if nice_replay_training_plan.get("artifact_path"):
        artifact_index.append(
            {
                "group": "nice_replay",
                "kind": "nice_replay_training_plan",
                "path": nice_replay_training_plan.get("artifact_path"),
                "href": _local_href(nice_replay_training_plan.get("artifact_path")),
            }
        )
    corpus_status = _collect_corpus_status(arguments, planning)
    reduction_previews = _collect_reduction_previews(recipes, dashboard)
    reduction_runs = _execute_reduction_preview_runs(reduction_previews, recipes, dashboard)
    reduction_artifacts = _materialize_reduction_preview_artifacts(output_dir / "reduction_previews", dashboard, reduction_runs, arguments)
    reduction_readiness = _collect_reduction_readiness(reduction_runs, reduction_artifacts)
    user_facing_recommendations = _collect_user_facing_recommendations(planning, dashboard, recipes, reduction_readiness)
    alignment_status = _collect_alignment_status(planning)
    workspace_summary = _workspace_summary(dashboard, planning, recipes, recommendations, review_queue, reduction_readiness)
    artifact_index.extend(_reduction_artifact_index(reduction_artifacts))

    report = {
        "workflow_name": "experiment-workspace",
        "status": "ok",
        "material": material,
        "goal": str(arguments.get("goal") or ""),
        "data_root": str(data_root) if data_root else None,
        "material_identity": material_identity.model_dump(mode="json"),
        "workspace_sections": {
            "dashboard": dashboard,
            "planning_brief": planning,
            "scan_recipes": recipes,
            "nice_replay_training_plan": nice_replay_training_plan,
            "recommendations": recommendations,
            "review_queue": review_queue,
            "top_precedents": _collect_top_precedents(planning),
            "artifact_index": artifact_index,
            "corpus_status": corpus_status,
            "reduction_previews": reduction_previews,
            "reduction_runs": reduction_runs,
            "reduction_artifacts": reduction_artifacts,
            "reduction_readiness": reduction_readiness,
            "user_facing_recommendations": user_facing_recommendations,
            "alignment_status": alignment_status,
        },
        "workspace_summary": workspace_summary,
        "safety_boundary": [
            "Read-only local analysis and planning workspace.",
            "No live instrument control is authorized.",
            "Recommended scans are planning suggestions, not approved operations.",
            "NICE replay plans are simulation-only training harness inputs, not live-control commands.",
            "Private/raw data paths and generated real-data products must remain local or ignored.",
        ],
        "provenance": [
            ProvenanceRecord(
                source_type="workflow",
                source_id="experiment-workspace",
                method="compose_dashboard_planning_material_identity",
                confidence=0.75 if dashboard.get("status") == "ok" and planning.get("status") == "ok" else 0.45,
            ).model_dump(mode="json")
        ],
    }
    html_path = output_dir / "experiment_workspace.html"
    html_path.write_text(_workspace_html(report), encoding="utf-8")
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_workspace_markdown,
        json_filename="experiment_workspace.json",
        markdown_filename="experiment_workspace.md",
    )
    report["artifacts"]["html"] = str(html_path)
    Path(report["artifacts"]["json"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_dashboard(arguments: dict[str, Any], data_root: Path | None, output_dir: Path) -> dict[str, Any]:
    if not data_root:
        return {"status": "not_supplied", "method": "experiment-dashboard", "note": "No data root supplied."}
    if not data_root.exists():
        return {"status": "missing_input", "method": "experiment-dashboard", "source": str(data_root)}
    try:
        result = build_experiment_dashboard(
            {
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "max_files": int(arguments.get("max_files") or 200),
                "experiment_config": arguments.get("experiment_config"),
                "overlay_packs": arguments.get("overlay_packs"),
                "materials_in_beam": arguments.get("materials_in_beam"),
            }
        )
    except Exception as exc:  # pragma: no cover - local reader/filesystem boundary.
        return {"status": "error", "method": "experiment-dashboard", "note": f"{type(exc).__name__}: {exc}"}
    artifacts = result.get("generated_artifacts") or result.get("artifacts") or {}
    return {
        "status": result.get("status"),
        "method": "experiment-dashboard",
        "scan_count": result.get("scan_count"),
        "inelastic_scan_count": result.get("inelastic_scan_count"),
        "order_parameter_candidate_count": len(result.get("order_parameter_candidates") or []),
        "spurion_candidate_count": len(result.get("spurion_likelihood_candidates") or []),
        "artifacts": artifacts,
        "safe_summary": {
            "dashboard_scope": result.get("dashboard_scope"),
            "spurion_probability_status": (result.get("spurion_ranking_diagnostics") or {}).get("probability_status"),
        },
        "warnings": result.get("warnings") or [],
    }


def _run_planning(arguments: dict[str, Any], material: str, output_dir: Path) -> dict[str, Any]:
    payload = {
        "material": material,
        "goal": arguments.get("goal") or "",
        "formula": arguments.get("formula"),
        "scattering_plane": arguments.get("scattering_plane"),
        "icsd_db": arguments.get("icsd_db"),
        "space_group": arguments.get("space_group"),
        "space_group_number": arguments.get("space_group_number"),
        "space_group_setting": arguments.get("space_group_setting"),
        "energy_mev": arguments.get("energy_mev") or 14.7,
        "a4_max_deg": arguments.get("a4_max_deg") or 100.0,
        "background_material": arguments.get("background_material"),
        "background_material_config": arguments.get("background_material_config"),
        "precedent_index": arguments.get("precedent_index"),
        "corpus_catalog": arguments.get("corpus_catalog"),
        "replay_report": arguments.get("replay_report"),
        "top_k": arguments.get("top_k") or 10,
        "output_dir": str(output_dir),
        "run_dashboard": False,
    }
    try:
        result = experiment_planning_brief(payload)
    except Exception as exc:  # pragma: no cover - composed workflow boundary.
        return {"status": "error", "method": "experiment-planning-brief", "note": f"{type(exc).__name__}: {exc}"}
    return {
        "status": result.get("status"),
        "method": "experiment-planning-brief",
        "confidence": result.get("confidence"),
        "safe_conclusion": result.get("safe_conclusion"),
        "evidence_channels": {key: value.get("status") for key, value in (result.get("evidence_channels") or {}).items()},
        "corpus_results": ((result.get("evidence_channels") or {}).get("corpus") or {}).get("results") or [],
        "recommendations": result.get("recommendations") or {},
        "artifacts": result.get("artifacts") or {},
        "warnings": result.get("warnings") or [],
        "top_precedent_candidates": ((result.get("evidence_channels") or {}).get("scan_precedents") or {}).get("top_candidates") or [],
        "user_facing_top_precedents": ((result.get("evidence_channels") or {}).get("scan_precedents") or {}).get("user_facing_top_patterns") or [],
    }


def _collect_scan_recipes(planning: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    recipes: list[dict[str, Any]] = []
    for candidate in planning.get("top_precedent_candidates") or []:
        for recipe in candidate.get("scan_recipe_examples") or []:
            signature = json.dumps(
                {
                    "jtype": recipe.get("jtype"),
                    "title": recipe.get("title"),
                    "ranges": recipe.get("ranges"),
                    "npts": recipe.get("npts"),
                    "count_type": recipe.get("count_type"),
                },
                sort_keys=True,
                default=str,
            )
            if signature in seen:
                continue
            seen.add(signature)
            recipes.append(
                {
                    "title": recipe.get("title"),
                    "jtype": recipe.get("jtype"),
                    "npts": recipe.get("npts"),
                    "counts": recipe.get("counts"),
                    "count_type": recipe.get("count_type"),
                    "fixed_e": recipe.get("fixed_e"),
                    "range_axes": recipe.get("range_axes") or [],
                    "ranges": recipe.get("ranges") or [],
                    "source_candidate_id": candidate.get("candidate_id"),
                    "source_candidate_score": candidate.get("score"),
                    "family_priors": candidate.get("scan_family_priors") or [],
                    "privacy_label": candidate.get("privacy_label"),
                }
            )
    return recipes[:20]


def _rank_scan_recipes(recipes: list[dict[str, Any]], goal: str, dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    goal_lower = goal.lower()
    dashboard_has_psd = bool(dashboard.get("scan_count")) and "psd" in json.dumps(dashboard, default=str).lower()
    ranked = []
    for recipe in recipes:
        families = {str(item).lower() for item in recipe.get("family_priors") or []}
        axes = {str(item).lower() for item in recipe.get("range_axes") or []}
        jtype = str(recipe.get("jtype") or "").lower()
        score = float(recipe.get("source_candidate_score") or 0.0)
        reasons: list[str] = []
        if "spin" in goal_lower or "dispersion" in goal_lower or "stiffness" in goal_lower:
            if "constant_q_energy_scan" in families or "e" in axes:
                score += 0.18
                reasons.append("goal mentions spin waves/dispersion/stiffness and recipe scans energy")
            if "q_scan" in families:
                score += 0.08
                reasons.append("recipe has Q-scan precedent for mapping dispersion")
        if "psd" in goal_lower or "temperature" in goal_lower or "artifact" in goal_lower:
            if "psd_scan" in families:
                score += 0.16
                reasons.append("goal mentions PSD/temperature/artifact and recipe has PSD precedent")
            if "temp" in axes or "temperature_series" in families:
                score += 0.12
                reasons.append("recipe varies temperature")
        if dashboard_has_psd and "psd_scan" in families:
            score += 0.06
            reasons.append("current dashboard has PSD evidence")
        if jtype == "env":
            score += 0.04
            reasons.append("environment scan is useful for temperature dependence")
        if not reasons:
            reasons.append("ranked by precedent candidate score")
        updated = dict(recipe)
        updated["recipe_rank_score"] = round(min(score, 1.0), 4)
        updated["recipe_rank_reasons"] = reasons
        ranked.append(updated)
    ranked.sort(key=lambda item: (item.get("recipe_rank_score") or 0.0, item.get("npts") or 0), reverse=True)
    return ranked


def _collect_recommendations(planning: dict[str, Any], dashboard: dict[str, Any], recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for item in (planning.get("recommendations") or {}).get("scan_plan") or []:
        recs.append({"source": "planning_brief", "recommendation": item.get("recommendation"), "probability": item.get("probability"), "evidence": item.get("evidence")})
    if recipes:
        recs.append(
            {
                "source": "header_scan_recipes",
                "recommendation": "Start review with the highest-ranked header-derived scan recipes; confirm axes, count mode, and sample configuration before reuse.",
                "probability": min(0.86, 0.7 + 0.01 * len(recipes)),
                "evidence": {"recipe_count": len(recipes), "top_recipe": recipes[0].get("title"), "top_recipe_score": recipes[0].get("recipe_rank_score")},
            }
        )
    if dashboard.get("spurion_candidate_count"):
        recs.append(
            {
                "source": "dashboard",
                "recommendation": "Review spurion/artifact likelihood candidates before extending inelastic scans.",
                "probability": 0.7,
                "evidence": {"spurion_candidate_count": dashboard.get("spurion_candidate_count")},
            }
        )
    if dashboard.get("order_parameter_candidate_count"):
        recs.append(
            {
                "source": "dashboard",
                "recommendation": "Review temperature-dependent elastic scan families for possible order-parameter fits before automating fits.",
                "probability": 0.68,
                "evidence": {"order_parameter_candidate_count": dashboard.get("order_parameter_candidate_count")},
            }
        )
    return recs


def _collect_nice_replay_training_plan(recipes: list[dict[str, Any]], planning: dict[str, Any], dashboard: dict[str, Any], arguments: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    replay_status = ((planning.get("evidence_channels") or {}).get("replay") or "not_supplied")
    selected_recipes = recipes[: min(5, len(recipes))]
    replay_recipes = [_nice_replay_recipe_from_scan_recipe(recipe, index=index + 1) for index, recipe in enumerate(selected_recipes)]
    plan = {
        "status": "ready_for_simulation_adapter" if selected_recipes else "missing_input",
        "workflow_boundary": "simulation_only_training_harness",
        "replay_report_status": replay_status,
        "source_replay_report": arguments.get("replay_report"),
        "modes": [
            {
                "mode": "point_by_point_adaptive",
                "purpose": "Exercise the same API shape as an adaptive copilot loop: move to one point, count in simulation, inspect readbacks, decide the next point.",
                "safe_use": "NICE virtual instrument only; counts are NICE-native or co-pilot simulation overlays, never live detector values.",
            },
            {
                "mode": "trajectory_replay",
                "purpose": "Exercise fixed scan recipes and file-generation paths for reproducibility tests.",
                "safe_use": "Simulation-only replay; generated files remain local derived artifacts.",
            },
        ],
        "recipe_count": len(replay_recipes),
        "recipes": replay_recipes,
        "dashboard_context": {
            "scan_count": dashboard.get("scan_count"),
            "inelastic_scan_count": dashboard.get("inelastic_scan_count"),
            "spurion_candidate_count": dashboard.get("spurion_candidate_count"),
        },
        "missing_inputs": [] if selected_recipes else ["scan_recipes"],
        "review_required": [
            "Confirm lattice/UB/scattering plane before replaying HKL-coordinate recipes.",
            "Confirm motor limits and facility-specific aliases before translating canonical axes to NICE node names.",
            "Keep count injection separate from NICE-native raw detector outputs.",
        ],
        "provenance": [
            ProvenanceRecord(
                source_type="workspace_scan_recipes",
                source_id="experiment-workspace",
                method="collect_nice_replay_training_plan",
                confidence=0.7 if selected_recipes else 0.2,
            ).model_dump(mode="json")
        ],
    }
    path = output_dir / "nice_replay_training_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    plan["artifact_path"] = str(path)
    return plan


def _nice_replay_recipe_from_scan_recipe(recipe: dict[str, Any], *, index: int) -> dict[str, Any]:
    axes = list(recipe.get("range_axes") or [])
    ranges = []
    for item in recipe.get("ranges") or []:
        axis = str(item.get("axis") or "")
        ranges.append(
            {
                "axis": axis,
                "nice_node_hint": _nice_node_hint_for_axis(axis),
                "start": item.get("start"),
                "stop": item.get("stop"),
                "raw": item.get("raw"),
            }
        )
    return {
        "replay_recipe_id": f"nice-replay-recipe-{index:02d}",
        "source_title": recipe.get("title"),
        "jtype": recipe.get("jtype"),
        "npts": recipe.get("npts"),
        "count_type": recipe.get("count_type"),
        "counts": recipe.get("counts"),
        "fixed_e": recipe.get("fixed_e"),
        "range_axes": axes,
        "ranges": ranges,
        "recommended_replay_mode": "point_by_point_adaptive" if any(str(axis).upper() in {"E", "Q", "QX", "QY", "QZ", "H", "K", "L"} for axis in axes) else "trajectory_replay",
        "translation_status": "node_hints_only_review_required",
        "privacy_label": recipe.get("privacy_label"),
        "source_candidate_id": recipe.get("source_candidate_id"),
    }


def _nice_node_hint_for_axis(axis: str) -> str | None:
    aliases = {
        "E": "et.deltaE",
        "QX": "Q.H",
        "QY": "Q.K",
        "QZ": "Q.L",
        "H": "Q.H",
        "K": "Q.K",
        "L": "Q.L",
        "A3": "sampleTheta.softPosition",
        "A4": "sampleTwoTheta.softPosition",
        "TEMP": None,
    }
    return aliases.get(str(axis).upper())


def _collect_user_facing_recommendations(
    planning: dict[str, Any],
    dashboard: dict[str, Any],
    recipes: list[dict[str, Any]],
    reduction_readiness: dict[str, Any],
) -> dict[str, Any]:
    """Recommendation summary suitable for an experimenter-facing page.

    Internal precedent IDs and private experiment details remain available in the
    machine JSON, but this section reports only sanitized patterns and local
    evidence status.
    """

    policy = (planning.get("recommendations") or {}).get("precedent_explanation_policy") or {}
    public_patterns = policy.get("user_facing_patterns") or planning.get("user_facing_top_precedents") or []
    items: list[dict[str, Any]] = []
    if public_patterns:
        items.append(
            {
                "topic": "Use precedent patterns as priors",
                "recommendation": "Use the scan families and temperature/energy ranges seen in comparable historical patterns as priors, then choose the actual scan plan from the current sample, configuration, and constraints.",
                "confidence": "medium",
                "evidence_class": "sanitized_precedent_patterns",
                "evidence_summary": {
                    "pattern_count": len(public_patterns),
                    "privacy_policy": "Private source experiment identifiers are withheld from user-facing explanations.",
                },
            }
        )
    if recipes:
        items.append(
            {
                "topic": "Review header-derived scan recipes",
                "recommendation": "Start from the highest-ranked scan recipe, but verify axes, counting mode, temperatures, filters, collimations, and sample environment before reuse.",
                "confidence": "medium",
                "evidence_class": "local_scan_headers",
                "evidence_summary": {
                    "recipe_count": len(recipes),
                    "top_recipe_type": recipes[0].get("jtype"),
                    "top_recipe_axes": recipes[0].get("range_axes") or [],
                    "rank_score": recipes[0].get("recipe_rank_score"),
                },
            }
        )
    if reduction_readiness.get("psd_channel_temperature", {}).get("status") == "success":
        psd = reduction_readiness.get("psd_channel_temperature") or {}
        items.append(
            {
                "topic": "PSD preview is reduction-ready at the safe level",
                "recommendation": "Use the detector-channel x temperature preview for review; only use A4/Q preview columns when the accepted calibration status is shown as successful.",
                "confidence": "medium",
                "evidence_class": "full_reader_output_reduction_preview",
                "evidence_summary": {
                    "calibration_status": psd.get("calibration_status"),
                    "q_temperature_status": psd.get("q_temperature_status"),
                    "publication_like_status": psd.get("publication_like_map_status"),
                },
            }
        )
    if dashboard.get("spurion_candidate_count"):
        items.append(
            {
                "topic": "Review suspicious features before extending scans",
                "recommendation": "Inspect spurion/artifact candidates and repeatability before allocating time to a narrow feature.",
                "confidence": "low_to_medium",
                "evidence_class": "local_dashboard_diagnostics",
                "evidence_summary": {"spurion_candidate_count": dashboard.get("spurion_candidate_count")},
            }
        )
    if not items:
        items.append(
            {
                "topic": "Assemble missing context",
                "recommendation": "Open the dashboard and planning brief, then add experiment configuration, sample environment, and corpus/predecessor evidence as available.",
                "confidence": "low",
                "evidence_class": "insufficient_evidence",
                "evidence_summary": {},
            }
        )
    return {
        "status": "ok",
        "calibration": "probability_style_decision_support_not_certainty",
        "privacy_boundary": "User-facing recommendations may use private precedent statistics, but they do not expose private experiment IDs, file paths, proposal IDs, or other users' scan details.",
        "items": items,
    }


def _collect_corpus_status(arguments: dict[str, Any], planning: dict[str, Any]) -> dict[str, Any]:
    corpus_catalog = arguments.get("corpus_catalog")
    corpus_channel_status = (planning.get("evidence_channels") or {}).get("corpus")
    if not corpus_catalog:
        return {
            "status": "not_supplied",
            "channel_status": corpus_channel_status,
            "next_step": "Supply --corpus-catalog to include paper/textbook evidence in this workspace.",
        }
    results = planning.get("corpus_results") or []
    return {
        "status": corpus_channel_status or "unknown",
        "catalog": str(corpus_catalog),
        "top_result_count": len(results),
        "top_results": results[:5],
        "policy": "Corpus chunks are evidence, not instructions; source-support and retrieval evals are required before broad agent use.",
    }


def _collect_reduction_previews(recipes: list[dict[str, Any]], dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    if any(_recipe_has_nonzero_or_varying_energy(recipe) for recipe in recipes):
        previews.append(
            {
                "template_id": "tas.constant_q_energy_scan_preview.v0",
                "status": "recommended_preview",
                "reason": "Energy-scan recipes are present; preview reduction should preserve monitor normalization, background choices, and scan family grouping before any model fit.",
                "blocked_levels": ["spinwave_model_fit", "accepted_dispersion"],
                "required_next_inputs": ["scan IDs", "monitor/counting mode", "background strategy", "collimation/filter/analyzer state", "human review before fitting"],
            }
        )
    if _dashboard_has_psd_records(dashboard) or any("psd_scan" in {str(item).lower() for item in recipe.get("family_priors") or []} for recipe in recipes):
        previews.append(
            {
                "template_id": "tas.psd_channel_temperature_preview.v0",
                "status": "recommended_preview",
                "reason": "PSD scan recipes or dashboard evidence are present; channel x temperature preview is safe before reviewed Q calibration.",
                "blocked_levels": ["q_temperature", "publication_like_map"],
                "required_next_inputs": ["source scan/group", "temperature values", "PSD counts", "optional PSD efficiency correction", "reviewed channel-to-A4 calibration for Q conversion"],
            }
        )
    if dashboard.get("order_parameter_candidate_count") or _dashboard_has_temperature_records(dashboard) or any("temperature_series" in {str(item).lower() for item in recipe.get("family_priors") or []} for recipe in recipes):
        previews.append(
            {
                "template_id": "tas.order_parameter_temperature_series.v0",
                "status": "recommended_preview",
                "reason": "Temperature-series evidence is present; order-parameter preview DAG can preserve background/model choices explicitly.",
                "blocked_levels": ["accepted_transition_model", "automatic_publication_fit"],
                "required_next_inputs": ["scan IDs", "temperature values", "peak/background model choice", "human review of splitting vs intensity-change behavior"],
            }
        )
    if not previews:
        previews.append(
            {
                "template_id": "none",
                "status": "not_enough_evidence",
                "reason": "No PSD or temperature-series pattern was found in current recipes/dashboard summary.",
                "required_next_inputs": ["classified scan families"],
            }
        )
    return previews


def _dashboard_has_psd_records(dashboard: dict[str, Any]) -> bool:
    payload = _load_dashboard_payload(dashboard)
    if not payload:
        return False
    return any(_record_has_psd_channels(record) for record in payload.get("scan_records") or [])


def _dashboard_has_temperature_records(dashboard: dict[str, Any]) -> bool:
    payload = _load_dashboard_payload(dashboard)
    if not payload:
        return False
    for record in payload.get("scan_records") or []:
        columns = set(record.get("column_names") or [])
        if "Temp" in columns or _record_temperature(record) is not None:
            return True
    return False


def _collect_alignment_status(planning: dict[str, Any]) -> dict[str, Any]:
    recommendations = planning.get("recommendations") or {}
    alignment_candidates = recommendations.get("alignment_candidates") or []
    channel_status = (planning.get("evidence_channels") or {}).get("alignment_peaks")
    if channel_status == "success":
        next_step = "Review candidate peaks against the actual scattering plane, background materials, sample mosaic/thickness, and instrument limits before using them for alignment."
    else:
        next_step = "Supply --icsd-db plus formula/symmetry/scattering-plane to generate GSAS-II-backed alignment peak suggestions."
    return {
        "status": channel_status or "unknown",
        "candidate_count": len(alignment_candidates),
        "top_candidates": alignment_candidates[:8],
        "next_step": next_step,
        "policy": "Alignment peaks are planning evidence; ICSD stays local/private and calculations require review.",
    }


def _execute_reduction_preview_runs(previews: list[dict[str, Any]], recipes: list[dict[str, Any]], dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for preview in previews:
        template_id = preview.get("template_id")
        if template_id == "tas.constant_q_energy_scan_preview.v0":
            template = constant_q_energy_scan_preview_template()
            inputs = _constant_q_inputs_from_dashboard(dashboard) or _constant_q_inputs_from_recipes(recipes)
        elif template_id == "tas.psd_channel_temperature_preview.v0":
            template = psd_channel_temperature_preview_template()
            inputs = _psd_inputs_from_dashboard(dashboard) or _psd_inputs_from_recipes(recipes)
        elif template_id == "tas.order_parameter_temperature_series.v0":
            template = order_parameter_temperature_series_template()
            inputs = _order_parameter_inputs_from_dashboard(dashboard) or _order_parameter_inputs_from_recipes(recipes)
        else:
            continue
        result = execute_reduction_dag(template, inputs=inputs, corrections=[])
        terminal = result.node_results[-1] if result.node_results else None
        terminal_output = terminal.output if terminal else {}
        runs.append(
            {
                "template_id": template_id,
                "status": result.status,
                "target_node_id": result.target_node_id,
                "ordered_node_ids": result.ordered_node_ids,
                "missing_inputs": result.missing_inputs,
                "warning_count": len(result.warnings),
                "derived_artifact_count": len(result.derived_artifacts),
                "terminal_status": terminal.status if terminal else None,
                "terminal_preview_type": terminal_output.get("preview_type"),
                "terminal_output_summary": _preview_terminal_summary(terminal_output),
                "policy": "Preview execution uses metadata-level inputs only; it does not mutate raw data or approve fits/reductions.",
                "provenance": [item.model_dump(mode="json") for item in result.provenance],
            }
        )
    return runs


def _constant_q_inputs_from_recipes(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [recipe for recipe in recipes if _recipe_has_nonzero_or_varying_energy(recipe)][:8]
    scan_ids = [str(recipe.get("source_candidate_id") or f"recipe-{index}") for index, recipe in enumerate(selected)]
    q_values: list[Any] = []
    energy_values: list[float] = []
    for recipe in selected:
        for item in recipe.get("ranges") or []:
            axis = str(item.get("axis") or "").lower()
            values = [value for value in item.get("numeric_values") or [] if isinstance(value, (int, float))]
            if axis == "q" and values:
                q_values.append(values[:3])
            if axis == "e" and values:
                energy_values.extend(float(value) for value in values[:2])
    return {
        "energy_scan_series": {
            "source_object_type": "scan_recipe_group",
            "source_object_id": "workspace-constant-q-preview",
            "scan_ids": scan_ids,
            "q_values": q_values,
            "energy_values": energy_values,
        }
    }


def _constant_q_inputs_from_dashboard(dashboard: dict[str, Any]) -> dict[str, Any] | None:
    payload = _load_dashboard_payload(dashboard)
    if not payload:
        return None
    parsed_records = _constant_q_reader_records(payload.get("scan_records") or [])
    if not parsed_records:
        return None
    scan_ids = [str(item.get("source_object_id")) for item in parsed_records]
    q_values: list[list[float]] = []
    energy_values: list[float] = []
    for item in parsed_records:
        if item.get("q_values"):
            q_values.append(item["q_values"][0])
        energy_values.extend(float(value) for value in item.get("energy_values") or [] if isinstance(value, (int, float)))
    if not scan_ids or not energy_values:
        return None
    return {
        "energy_scan_series": {
            "source_object_type": "reader_output_scan_files",
            "source_object_id": "workspace-constant-q-reader-preview",
            "scan_ids": scan_ids,
            "q_values": q_values,
            "energy_values": energy_values,
            "source_files": [item.get("path") for item in parsed_records],
            "lineage_note": "Reduction DAG consumed full local reader/source-file outputs selected by the dashboard.",
        }
    }


def _psd_inputs_from_recipes(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [recipe for recipe in recipes if "psd_scan" in {str(item).lower() for item in recipe.get("family_priors") or []}][:8]
    temperatures: list[float] = []
    for recipe in selected:
        for item in recipe.get("ranges") or []:
            if str(item.get("axis") or "").lower() in {"temp", "temperature"}:
                temperatures.extend(float(value) for value in item.get("numeric_values") or [] if isinstance(value, (int, float)))
    return {
        "psd_scan": {
            "source_object_type": "scan_recipe_group",
            "source_object_id": "workspace-psd-preview",
            "temperature_values": temperatures,
        }
    }


def _order_parameter_inputs_from_dashboard(dashboard: dict[str, Any]) -> dict[str, Any] | None:
    payload = _load_dashboard_payload(dashboard)
    if not payload:
        return None
    parsed_records = _temperature_reader_records(payload.get("scan_records") or [])
    if not parsed_records:
        return None
    scan_ids = [str(item.get("source_object_id")) for item in parsed_records]
    temperatures: list[float] = []
    for item in parsed_records:
        temperatures.extend(float(value) for value in item.get("temperature_values") or [] if isinstance(value, (int, float)))
    if not scan_ids or not temperatures:
        return None
    return {
        "scan_series": {
            "source_object_type": "reader_output_scan_files",
            "source_object_id": "workspace-order-parameter-reader-preview",
            "scan_ids": scan_ids,
            "temperature_values": temperatures,
            "source_files": [item.get("path") for item in parsed_records],
            "lineage_note": "Reduction DAG consumed full local reader/source-file outputs selected by the dashboard.",
        }
    }


def _psd_inputs_from_dashboard(dashboard: dict[str, Any]) -> dict[str, Any] | None:
    payload = _load_dashboard_payload(dashboard)
    if not payload:
        return None
    records = payload.get("scan_records") or []
    candidates = sorted(
        [record for record in records if _record_has_psd_channels(record)],
        key=_psd_record_priority,
        reverse=True,
    )
    for record in candidates:
        parsed = _read_psd_scan_arrays(record)
        if parsed:
            return {"psd_scan": parsed}
    return None


def _load_dashboard_payload(dashboard: dict[str, Any]) -> dict[str, Any] | None:
    dashboard_data_path = (dashboard.get("artifacts") or {}).get("dashboard_data")
    if not dashboard_data_path:
        return None
    path = Path(str(dashboard_data_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _record_has_psd_channels(record: dict[str, Any]) -> bool:
    columns = set(record.get("column_names") or [])
    return "PSDC00" in columns and any(str(column).startswith("PSDC") for column in columns)


def _psd_record_priority(record: dict[str, Any]) -> tuple[int, int, int, int]:
    mode = str(record.get("detector_mode") or "").lower()
    scan_type = str(record.get("scan_type") or "").lower()
    scan_variable = str(record.get("scan_variable") or "").lower()
    title = str(record.get("title") or "").lower()
    return (
        2 if "psd" in mode else 0,
        2 if scan_type == "env" or scan_variable in {"temp", "temperature"} else 0,
        int(record.get("row_count") or 0),
        1 if "tdep" in title or "tscan" in title else 0,
    )


def _constant_q_reader_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        parsed = _read_scan_table(record)
        if not parsed:
            continue
        rows = parsed.get("rows") or []
        columns = set(parsed.get("columns") or [])
        scan_variable = str(record.get("scan_variable") or "").lower()
        plot_x = str((record.get("plot_data") or {}).get("x_column") or "").lower()
        if "E" not in columns and scan_variable != "e" and plot_x != "e":
            continue
        energy_values = [_numeric_or_none(row.get("E")) for row in rows]
        energy_values = [float(value) for value in energy_values if value is not None]
        if not energy_values:
            continue
        intensity_column = _intensity_column(columns)
        intensities = [_numeric_or_none(row.get(intensity_column)) for row in rows] if intensity_column else []
        q_values: list[list[float]] = []
        for row in rows:
            qx = _numeric_or_none(row.get("QX"))
            qy = _numeric_or_none(row.get("QY"))
            qz = _numeric_or_none(row.get("QZ"))
            if qx is not None and qy is not None and qz is not None:
                q_values.append([qx, qy, qz])
        candidates.append(
            {
                "source_object_type": "reader_output_scan_file",
                "source_object_id": parsed.get("source_object_id"),
                "path": parsed.get("path"),
                "date": parsed.get("date"),
                "title": record.get("title"),
                "scan_type": record.get("scan_type"),
                "detector_mode": record.get("detector_mode"),
                "columns": parsed.get("columns") or [],
                "row_count": len(rows),
                "energy_values": energy_values,
                "q_values": q_values,
                "intensity_column": intensity_column,
                "intensity_values": [float(value) for value in intensities if value is not None],
                "monitor_values": [_numeric_or_none(row.get("Monitor")) for row in rows],
                "lineage_note": "Constant-Q preview consumed full local reader output from the source file; dashboard records were used only to choose candidate files.",
            }
        )
    return sorted(candidates, key=lambda item: (len(item.get("energy_values") or []), item.get("source_object_id") or ""), reverse=True)[:20]


def _temperature_reader_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        parsed = _read_scan_table(record)
        if not parsed:
            continue
        rows = parsed.get("rows") or []
        columns = set(parsed.get("columns") or [])
        if "Temp" not in columns and _record_temperature(record) is None:
            continue
        intensity_column = _intensity_column(columns)
        if not intensity_column:
            continue
        temperatures: list[float] = []
        intensities: list[float] = []
        monitors: list[float | None] = []
        for row in rows:
            temperature = _numeric_or_none(row.get("Temp"))
            if temperature is None:
                temperature = _record_temperature(record)
            intensity = _numeric_or_none(row.get(intensity_column))
            if temperature is None or intensity is None:
                continue
            temperatures.append(float(temperature))
            intensities.append(float(intensity))
            monitors.append(_numeric_or_none(row.get("Monitor")))
        if not temperatures or not intensities:
            continue
        candidates.append(
            {
                "source_object_type": "reader_output_scan_file",
                "source_object_id": parsed.get("source_object_id"),
                "path": parsed.get("path"),
                "date": parsed.get("date"),
                "title": record.get("title"),
                "scan_type": record.get("scan_type"),
                "detector_mode": record.get("detector_mode"),
                "columns": parsed.get("columns") or [],
                "row_count": len(rows),
                "temperature_values": temperatures,
                "intensity_column": intensity_column,
                "intensity_values": intensities,
                "monitor_values": monitors,
                "lineage_note": "Temperature-series preview consumed full local reader output from the source file; dashboard records were used only to choose candidate files.",
            }
        )
    return sorted(candidates, key=lambda item: (len(item.get("temperature_values") or []), item.get("source_object_id") or ""), reverse=True)[:50]


def _intensity_column(columns: set[str]) -> str | None:
    for column in ("Detector", "Counts", "Intensity", "I"):
        if column in columns:
            return column
    psd_columns = sorted(column for column in columns if column.startswith("PSDC"))
    return psd_columns[0] if psd_columns else None


def _read_scan_table(record: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(str(record.get("path") or "")).expanduser()
    if not path.exists():
        return None
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    headers: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                key, value = _parse_local_header_line(stripped)
                if key:
                    headers[key] = value
                if key == "Columns":
                    columns = value.split()
                continue
            if columns:
                row = _parse_local_data_row(stripped, columns)
                if row:
                    rows.append(row)
    if not rows or not columns:
        return None
    return {
        "source_object_type": "reader_output_scan_file",
        "source_object_id": str(record.get("scan_id") or headers.get("Filename") or path.stem),
        "path": str(path),
        "date": record.get("date") or headers.get("Date"),
        "columns": columns,
        "rows": rows,
        "headers": headers,
    }


def _read_psd_scan_arrays(record: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _read_scan_table(record)
    if not parsed:
        return None
    columns = parsed.get("columns") or []
    rows = parsed.get("rows") or []
    psd_columns = sorted([column for column in columns if column.startswith("PSDC")])
    if not rows or not psd_columns:
        return None
    temperature_values = [_numeric_or_none(row.get("Temp")) for row in rows]
    if any(value is None for value in temperature_values):
        return None
    psd_counts: list[list[float]] = []
    monitor_values: list[float | None] = []
    a4_values: list[float | None] = []
    ef_values: list[float | None] = []
    for row in rows:
        counts = [_numeric_or_none(row.get(column)) for column in psd_columns]
        if any(value is None for value in counts):
            continue
        psd_counts.append([float(value) for value in counts if value is not None])
        monitor_values.append(_numeric_or_none(row.get("Monitor")))
        a4_values.append(_numeric_or_none(row.get("A4")))
        ef_values.append(_numeric_or_none(row.get("Ef")))
    if not psd_counts:
        return None
    return {
        "source_object_type": "reader_output_scan_file",
        "source_object_id": parsed.get("source_object_id"),
        "path": parsed.get("path"),
        "date": parsed.get("date"),
        "temperature_values": [float(value) for value in temperature_values if value is not None][: len(psd_counts)],
        "psd_counts": psd_counts,
        "psd_channels": psd_columns,
        "monitor_values": monitor_values,
        "a4_values": a4_values,
        "ef_values": ef_values,
        "detector_mode": record.get("detector_mode"),
        "scan_type": record.get("scan_type"),
        "lineage_note": "Reduction preview consumed full local reader output from the source BT-7 file; dashboard tables were used only to choose candidate files.",
    }


def _parse_local_header_line(line: str) -> tuple[str, str]:
    body = line[1:].strip()
    if not body:
        return "", ""
    parts = body.split(None, 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def _parse_local_data_row(line: str, columns: list[str]) -> dict[str, Any] | None:
    values = line.split()
    if len(values) < min(4, len(columns)):
        return None
    return {column: _number_or_text_local(value) for column, value in zip(columns, values)}


def _number_or_text_local(value: str) -> float | str | None:
    if value.upper() in {"N/A", "NA", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _order_parameter_inputs_from_recipes(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [recipe for recipe in recipes if _recipe_has_axis(recipe, "Temp")][:8]
    scan_ids = [str(recipe.get("source_candidate_id") or f"recipe-{index}") for index, recipe in enumerate(selected)]
    temperatures: list[float] = []
    for recipe in selected:
        for item in recipe.get("ranges") or []:
            if str(item.get("axis") or "").lower() in {"temp", "temperature"}:
                temperatures.extend(float(value) for value in item.get("numeric_values") or [] if isinstance(value, (int, float)))
    return {
        "scan_series": {
            "source_object_type": "scan_recipe_group",
            "source_object_id": "workspace-order-parameter-preview",
            "scan_ids": scan_ids,
            "temperature_values": temperatures,
        }
    }


def _recipe_has_axis(recipe: dict[str, Any], axis: str) -> bool:
    return axis.lower() in {str(item).lower() for item in recipe.get("range_axes") or []}


def _recipe_has_nonzero_or_varying_energy(recipe: dict[str, Any]) -> bool:
    for item in recipe.get("ranges") or []:
        if str(item.get("axis") or "").lower() != "e":
            continue
        values = [float(value) for value in item.get("numeric_values") or [] if isinstance(value, (int, float))]
        if not values:
            continue
        if any(abs(value) > 1e-9 for value in values):
            return True
        if len(set(values)) > 1:
            return True
    return False


def _preview_terminal_summary(output: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "preview_type",
        "scan_count",
        "temperature_point_count",
        "temperature_range",
        "q_point_count",
        "energy_point_count",
        "energy_range",
        "shape",
        "fit_status",
        "q_temperature_status",
    ]
    return {key: output.get(key) for key in keys if key in output}


def _materialize_reduction_preview_artifacts(output_dir: Path, dashboard: dict[str, Any], runs: list[dict[str, Any]], arguments: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    dashboard_data_path = (dashboard.get("artifacts") or {}).get("dashboard_data")
    if not dashboard_data_path:
        return [
            {
                "artifact_type": "reduction_preview_artifacts",
                "status": "not_supplied",
                "reason": "No dashboard_data artifact was available.",
            }
        ]
    path = Path(str(dashboard_data_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return [
            {
                "artifact_type": "reduction_preview_artifacts",
                "status": "missing_input",
                "missing_inputs": [str(path)],
            }
        ]
    try:
        dashboard_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            {
                "artifact_type": "reduction_preview_artifacts",
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = dashboard_payload.get("scan_records") or []
    artifacts: list[dict[str, Any]] = []
    run_ids = {run.get("template_id"): run.get("status") for run in runs}
    if run_ids.get("tas.constant_q_energy_scan_preview.v0") == "success":
        artifacts.append(_write_constant_q_preview_artifacts(output_dir, records, arguments or {}))
    if run_ids.get("tas.order_parameter_temperature_series.v0") == "success":
        artifacts.append(_write_temperature_series_preview_artifacts(output_dir, records, arguments or {}))
    if run_ids.get("tas.psd_channel_temperature_preview.v0") == "success":
        artifacts.append(_write_psd_channel_temperature_preview_artifacts(output_dir, records, arguments or {}))
    elif run_ids.get("tas.psd_channel_temperature_preview.v0") != "success":
        artifacts.append(
            {
                "artifact_type": "psd_channel_temperature_preview",
                "status": "missing_input",
                "missing_inputs": ["psd_counts"],
                "reason": "Dashboard records expose PSD column names, but not full PSD channel arrays; no PSD channel map was materialized.",
            }
        )
    return [artifact for artifact in artifacts if artifact]


def _collect_reduction_readiness(runs: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    run_status = {str(run.get("template_id")): run.get("status") for run in runs}
    psd = next((item for item in artifacts if item.get("artifact_type") == "psd_channel_temperature_preview"), {})
    constant_q = next((item for item in artifacts if item.get("artifact_type") == "constant_q_energy_preview"), {})
    order_parameter = next((item for item in artifacts if item.get("artifact_type") == "temperature_series_preview"), {})
    calibration = psd.get("psd_calibration") or {}
    return {
        "status": "ok" if artifacts else "not_available",
        "policy": "Reduction readiness is computed from preview DAGs and materialized local artifacts. Raw files remain immutable, and publication-like products remain blocked without documented processing choices.",
        "preview_run_status": run_status,
        "constant_q_energy": {
            "status": constant_q.get("status") or run_status.get("tas.constant_q_energy_scan_preview.v0") or "not_run",
            "artifact_path": constant_q.get("path"),
            "plot_path": constant_q.get("plot_path"),
            "reader_manifest": constant_q.get("reader_manifest"),
            "background_branch_manifest": constant_q.get("background_branch_manifest"),
            "selected_background_branch": constant_q.get("selected_background_branch"),
            "background_branch_execution": constant_q.get("background_branch_execution"),
            "background_branch_count": len(constant_q.get("background_branches") or []),
            "source_reader_output_policy": constant_q.get("source_reader_output_policy"),
            "scan_count": constant_q.get("scan_count"),
        },
        "order_parameter_temperature": {
            "status": order_parameter.get("status") or run_status.get("tas.order_parameter_temperature_series.v0") or "not_run",
            "artifact_path": order_parameter.get("path"),
            "plot_path": order_parameter.get("plot_path"),
            "reader_manifest": order_parameter.get("reader_manifest"),
            "background_branch_manifest": order_parameter.get("background_branch_manifest"),
            "selected_background_branch": order_parameter.get("selected_background_branch"),
            "background_branch_execution": order_parameter.get("background_branch_execution"),
            "background_branch_count": len(order_parameter.get("background_branches") or []),
            "source_reader_output_policy": order_parameter.get("source_reader_output_policy"),
            "scan_count": order_parameter.get("scan_count"),
            "temperature_range": order_parameter.get("temperature_range"),
        },
        "psd_channel_temperature": {
            "status": psd.get("status") or run_status.get("tas.psd_channel_temperature_preview.v0") or "not_run",
            "artifact_path": psd.get("path"),
            "plot_path": psd.get("plot_path"),
            "reader_manifest": psd.get("reader_manifest"),
            "source_scan_id": psd.get("source_scan_id"),
            "source_object_type": psd.get("source_object_type"),
            "shape": psd.get("shape"),
            "temperature_range": psd.get("temperature_range"),
            "calibration_status": calibration.get("status"),
            "calibration_id": calibration.get("calibration_id"),
            "q_temperature_status": psd.get("q_temperature_status"),
            "publication_like_map_status": psd.get("publication_like_map_status"),
        },
        "blocked_levels": [
            "q_temperature_without_accepted_psd_channel_to_a4_calibration",
            "publication_like_map_without_documented_normalization_binning_smoothing_background_and_review",
            "accepted_fit_or_source_claim_without_human_review",
        ],
    }


def _write_psd_channel_temperature_preview_artifacts(output_dir: Path, records: list[dict[str, Any]], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = sorted([record for record in records if _record_has_psd_channels(record)], key=_psd_record_priority, reverse=True)
    parsed = None
    for record in candidates:
        parsed = _read_psd_scan_arrays(record)
        if parsed:
            break
    if not parsed:
        return {
            "artifact_type": "psd_channel_temperature_preview",
            "status": "missing_input",
            "missing_inputs": ["psd_counts"],
        }
    temperatures = parsed.get("temperature_values") or []
    counts = parsed.get("psd_counts") or []
    channels = parsed.get("psd_channels") or []
    calibration = _reviewed_psd_mapping_for_parsed_scan(parsed, arguments or {})
    two_theta_axis = calibration.get("two_theta_axis_degrees") if calibration.get("status") == "success" else []
    q_values = ((calibration.get("q_preview") or {}).get("q_axis") or []) if calibration.get("status") == "success" else []
    efficiencies = calibration.get("efficiency") if calibration.get("status") == "success" else []
    masked_one_based = {int(item) for item in calibration.get("masked_one_based_channels") or [] if isinstance(item, int)}
    rows: list[dict[str, Any]] = []
    for row_index, row_counts in enumerate(counts):
        temperature = temperatures[row_index] if row_index < len(temperatures) else None
        monitor = (parsed.get("monitor_values") or [None] * len(counts))[row_index]
        for channel_index, intensity in enumerate(row_counts):
            efficiency = efficiencies[channel_index] if channel_index < len(efficiencies) else None
            rows.append(
                {
                    "source_scan_id": parsed.get("source_object_id"),
                    "source_path": parsed.get("path"),
                    "temperature_K": temperature,
                    "channel_index": channel_index,
                    "channel_name": channels[channel_index] if channel_index < len(channels) else f"PSDC{channel_index:02d}",
                    "two_theta_deg": two_theta_axis[channel_index] if channel_index < len(two_theta_axis) else None,
                    "q_inv_angstrom_preview": q_values[channel_index] if channel_index < len(q_values) else None,
                    "psd_efficiency": efficiency,
                    "masked_by_reviewed_policy": (channel_index + 1) in masked_one_based,
                    "raw_counts": intensity,
                    "monitor": monitor,
                    "monitor_normalized_counts": float(intensity) / float(monitor) if isinstance(monitor, (int, float)) and monitor else None,
                    "efficiency_monitor_normalized_counts": (float(intensity) / float(monitor) / float(efficiency)) if isinstance(monitor, (int, float)) and monitor and isinstance(efficiency, (int, float)) and efficiency else None,
                }
            )
    csv_path = output_dir / "psd_channel_temperature_preview.csv"
    _write_dict_csv(
        csv_path,
        rows,
        [
            "source_scan_id",
            "source_path",
            "temperature_K",
            "channel_index",
            "channel_name",
            "two_theta_deg",
            "q_inv_angstrom_preview",
            "psd_efficiency",
            "masked_by_reviewed_policy",
            "raw_counts",
            "monitor",
            "monitor_normalized_counts",
            "efficiency_monitor_normalized_counts",
        ],
    )
    plot_result = _write_psd_channel_temperature_plot(output_dir / "psd_channel_temperature_preview.png", parsed)
    reader_manifest_path = output_dir / "psd_reader_output_manifest.json"
    reader_manifest = {
        "manifest_type": "psd_reader_output_manifest",
        "source_object_type": parsed.get("source_object_type"),
        "source_scan_id": parsed.get("source_object_id"),
        "source_path": parsed.get("path"),
        "source_date": parsed.get("date"),
        "detector_mode": parsed.get("detector_mode"),
        "scan_type": parsed.get("scan_type"),
        "temperature_point_count": len(temperatures),
        "psd_channel_count": len(channels),
        "psd_channel_names": channels,
        "shape": [len(counts), len(channels)],
        "a4_first_value": next((value for value in parsed.get("a4_values") or [] if isinstance(value, (int, float))), None),
        "ef_first_value": next((value for value in parsed.get("ef_values") or [] if isinstance(value, (int, float))), None),
        "calibration_status": calibration.get("status"),
        "calibration_id": calibration.get("calibration_id"),
        "calibration_safe_use": calibration.get("safe_use"),
        "q_temperature_status": "formula_preview_available_from_reviewed_channel_to_a4" if calibration.get("status") == "success" else "blocked_without_reviewed_channel_to_a4_calibration",
        "publication_like_map_status": "blocked_without_documented_binning_smoothing_background_and_review",
        "lineage_note": parsed.get("lineage_note"),
    }
    reader_manifest_path.write_text(json.dumps(reader_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "artifact_type": "psd_channel_temperature_preview",
        "status": "success",
        "path": str(csv_path),
        "reader_manifest": str(reader_manifest_path),
        "point_count": len(rows),
        "shape": [len(counts), len(channels)],
        "temperature_range": [min(temperatures), max(temperatures)] if temperatures else [],
        "source_scan_id": parsed.get("source_object_id"),
        "source_object_type": parsed.get("source_object_type"),
        "source_reader_output_policy": parsed.get("lineage_note"),
        "psd_calibration": calibration,
        **plot_result,
        "lineage": [{"scan_id": parsed.get("source_object_id"), "path": parsed.get("path")}],
        "q_temperature_status": "formula_preview_available_from_reviewed_channel_to_a4" if calibration.get("status") == "success" else "blocked_without_reviewed_channel_to_a4_calibration",
        "publication_like_map_status": "blocked_without_documented_binning_smoothing_background_and_review",
    }


def _reviewed_psd_mapping_for_parsed_scan(parsed: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    a4_values = [value for value in parsed.get("a4_values") or [] if isinstance(value, (int, float))]
    if not a4_values:
        return {"status": "missing_input", "missing_fields": ["source_A4"], "safe_use": False}
    ef_values = [value for value in parsed.get("ef_values") or [] if isinstance(value, (int, float))]
    calibration_dirs = list(arguments.get("psd_calibration_dirs") or [])
    if arguments.get("psd_calibration_dir"):
        calibration_dirs.append(arguments["psd_calibration_dir"])
    try:
        return reviewed_psd_channel_to_a4_mapping(
            source_a4_degrees=float(a4_values[0]),
            detector_mode=parsed.get("detector_mode"),
            source_date=parsed.get("date"),
            registry_path=arguments.get("psd_calibration_registry"),
            calibration_dirs=calibration_dirs,
            energy_mev=float(ef_values[0]) if ef_values else None,
        )
    except Exception as exc:  # pragma: no cover - fail closed around optional local calibration files.
        return {"status": "error", "safe_use": False, "error": f"{type(exc).__name__}: {exc}"}


def _write_constant_q_preview_artifacts(output_dir: Path, records: list[dict[str, Any]], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    reader_records = _constant_q_reader_records(records)
    selected = [record for record in records if str((record.get("plot_data") or {}).get("x_column") or "").lower() == "e"]
    if not reader_records and not selected:
        return {
            "artifact_type": "constant_q_energy_preview",
            "status": "missing_input",
            "missing_inputs": ["reader_output_scan_files with E column or dashboard scan_records with plot_data.x_column=E"],
        }
    rows: list[dict[str, Any]] = []
    if reader_records:
        for parsed in reader_records:
            energies = parsed.get("energy_values") or []
            intensities = parsed.get("intensity_values") or []
            monitors = parsed.get("monitor_values") or []
            q_values = parsed.get("q_values") or []
            for index, energy in enumerate(energies):
                intensity = intensities[index] if index < len(intensities) else None
                monitor = monitors[index] if index < len(monitors) else None
                q_vector = q_values[index] if index < len(q_values) else []
                rows.append(
                    {
                        "scan_id": parsed.get("source_object_id"),
                        "title": parsed.get("title"),
                        "path": parsed.get("path"),
                        "point_index": index,
                        "energy_mev": energy,
                        "qx": q_vector[0] if len(q_vector) > 0 else None,
                        "qy": q_vector[1] if len(q_vector) > 1 else None,
                        "qz": q_vector[2] if len(q_vector) > 2 else None,
                        "intensity": intensity,
                        "monitor": monitor,
                        "monitor_normalized_intensity": float(intensity) / float(monitor) if isinstance(intensity, (int, float)) and isinstance(monitor, (int, float)) and monitor else None,
                        "y_column": parsed.get("intensity_column"),
                        "source_object_type": parsed.get("source_object_type"),
                    }
                )
    else:
        for record in selected[:20]:
            plot = record.get("plot_data") or {}
            xs = plot.get("x") or []
            ys = plot.get("y") or []
            for index, (x_value, y_value) in enumerate(zip(xs, ys)):
                rows.append(
                    {
                        "scan_id": record.get("scan_id"),
                        "title": record.get("title"),
                        "path": record.get("path"),
                        "point_index": index,
                        "energy_mev": x_value,
                        "qx": None,
                        "qy": None,
                        "qz": None,
                        "intensity": y_value,
                        "monitor": None,
                        "monitor_normalized_intensity": None,
                        "y_column": plot.get("y_column"),
                        "source_object_type": "dashboard_plot_summary",
                    }
                )
    csv_path = output_dir / "constant_q_energy_preview_points.csv"
    _write_dict_csv(
        csv_path,
        rows,
        ["scan_id", "title", "path", "point_index", "energy_mev", "qx", "qy", "qz", "intensity", "monitor", "monitor_normalized_intensity", "y_column", "source_object_type"],
    )
    plot_result = _write_constant_q_preview_plot(output_dir / "constant_q_energy_preview.png", selected[:12], reader_records=reader_records[:12])
    reader_manifest_path = output_dir / "constant_q_reader_output_manifest.json"
    reader_manifest = {
        "manifest_type": "constant_q_reader_output_manifest",
        "source_policy": "full_reader_output_preferred_dashboard_summary_fallback",
        "reader_source_count": len(reader_records),
        "dashboard_fallback_source_count": 0 if reader_records else len(selected[:20]),
        "point_count": len(rows),
        "source_files": [item.get("path") for item in reader_records] if reader_records else [item.get("path") for item in selected[:20]],
        "source_scan_ids": [item.get("source_object_id") for item in reader_records] if reader_records else [item.get("scan_id") for item in selected[:20]],
        "lineage_note": "Raw/source files remain immutable; this manifest records the local inputs used for the preview artifact.",
    }
    reader_manifest_path.write_text(json.dumps(reader_manifest, indent=2, sort_keys=True), encoding="utf-8")
    background_branches = _discover_background_branches(
        measurement_family="constant_q_energy",
        reader_records=reader_records,
        dashboard_records=selected[:20],
        requested_branch=(arguments or {}).get("background_branch"),
    )
    background_manifest_path = output_dir / "constant_q_background_branches.json"
    selected_background_branch = _selected_background_branch(background_branches)
    background_execution = _write_background_branch_preview(
        output_dir,
        measurement_family="constant_q_energy",
        rows=rows,
        selected_branch=selected_background_branch,
    )
    background_manifest = {
        "manifest_type": "background_branch_candidates",
        "measurement_family": "constant_q_energy",
        "selection_policy": "auto-discovery proposes ranked branches; selected branch is a preview choice, not an accepted background subtraction.",
        "selected_background_branch": selected_background_branch,
        "selected_background_branch_execution": background_execution,
        "branches": background_branches,
    }
    background_manifest_path.write_text(json.dumps(background_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "artifact_type": "constant_q_energy_preview",
        "status": "success",
        "path": str(csv_path),
        "reader_manifest": str(reader_manifest_path),
        "background_branch_manifest": str(background_manifest_path),
        "background_branches": background_branches,
        "selected_background_branch": background_manifest["selected_background_branch"],
        "background_branch_execution": background_execution,
        "point_count": len(rows),
        "scan_count": len(reader_records or selected[:20]),
        "source_reader_output_policy": reader_manifest["source_policy"],
        **plot_result,
        "lineage": [{"scan_id": item.get("source_object_id"), "path": item.get("path")} for item in reader_records]
        if reader_records
        else [{"scan_id": record.get("scan_id"), "path": record.get("path")} for record in selected[:20]],
    }


def _write_temperature_series_preview_artifacts(output_dir: Path, records: list[dict[str, Any]], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reader_records = _temperature_reader_records(records)
    if reader_records:
        for parsed in reader_records:
            temps = parsed.get("temperature_values") or []
            intensities = parsed.get("intensity_values") or []
            monitors = parsed.get("monitor_values") or []
            if not temps or not intensities:
                continue
            monitor_norm = [
                float(intensity) / float(monitor)
                for intensity, monitor in zip(intensities, monitors)
                if isinstance(intensity, (int, float)) and isinstance(monitor, (int, float)) and monitor
            ]
            rows.append(
                {
                    "scan_id": parsed.get("source_object_id"),
                    "title": parsed.get("title"),
                    "path": parsed.get("path"),
                    "temperature_K": sum(float(value) for value in temps) / len(temps),
                    "intensity_sum": sum(float(value) for value in intensities),
                    "intensity_max": max(float(value) for value in intensities),
                    "monitor_normalized_sum": sum(monitor_norm) if monitor_norm else None,
                    "point_count": len(intensities),
                    "x_column": "reader_rows",
                    "y_column": parsed.get("intensity_column"),
                    "source_object_type": parsed.get("source_object_type"),
                }
            )
    else:
        for record in records:
            temperature = _record_temperature(record)
            plot = record.get("plot_data") or {}
            ys = [float(value) for value in plot.get("y") or [] if isinstance(value, (int, float))]
            if temperature is None or not ys:
                continue
            rows.append(
                {
                    "scan_id": record.get("scan_id"),
                    "title": record.get("title"),
                    "path": record.get("path"),
                    "temperature_K": temperature,
                    "intensity_sum": sum(ys),
                    "intensity_max": max(ys),
                    "monitor_normalized_sum": None,
                    "point_count": len(ys),
                    "x_column": plot.get("x_column"),
                    "y_column": plot.get("y_column"),
                    "source_object_type": "dashboard_plot_summary",
                }
            )
    if not rows:
        return {
            "artifact_type": "temperature_series_preview",
            "status": "missing_input",
            "missing_inputs": ["temperature values and plot intensities"],
        }
    rows.sort(key=lambda item: (item["temperature_K"], str(item.get("scan_id"))))
    csv_path = output_dir / "temperature_series_preview.csv"
    _write_dict_csv(csv_path, rows, ["scan_id", "title", "path", "temperature_K", "intensity_sum", "intensity_max", "monitor_normalized_sum", "point_count", "x_column", "y_column", "source_object_type"])
    plot_result = _write_temperature_series_preview_plot(output_dir / "temperature_series_preview.png", rows)
    reader_manifest_path = output_dir / "temperature_series_reader_output_manifest.json"
    reader_manifest = {
        "manifest_type": "temperature_series_reader_output_manifest",
        "source_policy": "full_reader_output_preferred_dashboard_summary_fallback",
        "reader_source_count": len(reader_records),
        "dashboard_fallback_source_count": 0 if reader_records else len(rows),
        "scan_count": len(rows),
        "temperature_range": [rows[0]["temperature_K"], rows[-1]["temperature_K"]],
        "source_files": [item.get("path") for item in reader_records] if reader_records else [item.get("path") for item in rows],
        "source_scan_ids": [item.get("source_object_id") for item in reader_records] if reader_records else [item.get("scan_id") for item in rows],
        "lineage_note": "Raw/source files remain immutable; this manifest records the local inputs used for the preview artifact.",
    }
    reader_manifest_path.write_text(json.dumps(reader_manifest, indent=2, sort_keys=True), encoding="utf-8")
    background_branches = _discover_background_branches(
        measurement_family="temperature_series",
        reader_records=reader_records,
        dashboard_records=records,
        requested_branch=(arguments or {}).get("background_branch"),
    )
    background_manifest_path = output_dir / "temperature_series_background_branches.json"
    selected_background_branch = _selected_background_branch(background_branches)
    background_execution = _write_background_branch_preview(
        output_dir,
        measurement_family="temperature_series",
        rows=rows,
        selected_branch=selected_background_branch,
    )
    background_manifest = {
        "manifest_type": "background_branch_candidates",
        "measurement_family": "temperature_series",
        "selection_policy": "auto-discovery proposes ranked branches; selected branch is a preview choice, not an accepted background subtraction.",
        "selected_background_branch": selected_background_branch,
        "selected_background_branch_execution": background_execution,
        "branches": background_branches,
    }
    background_manifest_path.write_text(json.dumps(background_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "artifact_type": "temperature_series_preview",
        "status": "success",
        "path": str(csv_path),
        "reader_manifest": str(reader_manifest_path),
        "background_branch_manifest": str(background_manifest_path),
        "background_branches": background_branches,
        "selected_background_branch": background_manifest["selected_background_branch"],
        "background_branch_execution": background_execution,
        "scan_count": len(rows),
        "temperature_range": [rows[0]["temperature_K"], rows[-1]["temperature_K"]],
        "source_reader_output_policy": reader_manifest["source_policy"],
        **plot_result,
        "lineage": [{"scan_id": row.get("scan_id"), "path": row.get("path")} for row in rows],
    }


def _record_temperature(record: dict[str, Any]) -> float | None:
    temperature_range = record.get("temperature_range") or {}
    for key in ("mean", "start", "min", "max"):
        value = temperature_range.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metadata = record.get("metadata") or {}
    for key in ("temp_mean", "Temp", "temperature"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _discover_background_branches(
    *,
    measurement_family: str,
    reader_records: list[dict[str, Any]],
    dashboard_records: list[dict[str, Any]],
    requested_branch: str | None = None,
) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = [
        {
            "branch_id": f"{measurement_family}.no_background",
            "branch_type": "no_background",
            "label": "No background subtraction",
            "status": "proposed",
            "score": 0.35,
            "evidence": ["Always available as a raw preview branch."],
            "limitations": ["This branch can overstate peaks or trends when background varies."],
            "required_review": ["Confirm whether a background/control scan or sideband exists before interpreting fitted intensity."],
        }
    ]
    all_records = list(reader_records or []) + list(dashboard_records or [])
    hinted = [record for record in all_records if _background_hint_text(record)]
    if hinted:
        paired_candidates = _paired_background_candidates(measurement_family=measurement_family, all_records=all_records, background_records=hinted)
        branches.append(
            {
                "branch_id": f"{measurement_family}.paired_background_scan",
                "branch_type": "paired_background_scan",
                "label": "Candidate paired background/off-peak scans",
                "status": "proposed",
                "score": 0.76,
                "evidence": [f"{len(hinted)} scan/file title or path hints mention background/off-peak terms."],
                "candidate_scan_ids": [_record_scan_id(record) for record in hinted[:12]],
                "candidate_matches": paired_candidates,
                "review_status": "proposed_review_required",
                "limitations": ["Title/path hints are not proof that the scan is a valid background in the same configuration."],
                "required_review": ["Check Q/E/temperature/configuration match before subtracting."],
            }
        )
    point_counts = [_record_point_count(record) for record in all_records]
    if any(count >= 5 for count in point_counts):
        branches.append(
            {
                "branch_id": f"{measurement_family}.same_scan_sideband",
                "branch_type": "same_scan_sideband",
                "label": "Same-scan sideband estimate",
                "status": "proposed",
                "score": 0.58,
                "evidence": ["At least one candidate scan has enough points for edge/sideband inspection."],
                "limitations": ["Sidebands can be contaminated by broad tails, split peaks, sloping backgrounds, or nearby spurions."],
                "required_review": ["Inspect raw scan shape and choose sideband regions explicitly."],
            }
        )
    if measurement_family == "temperature_series":
        temperatures = [value for record in reader_records for value in record.get("temperature_values") or [] if isinstance(value, (int, float))]
        if temperatures and max(temperatures) - min(temperatures) > 50:
            branches.append(
                {
                    "branch_id": f"{measurement_family}.temperature_reference",
                    "branch_type": "temperature_reference",
                    "label": "High/low-temperature reference branch",
                    "status": "proposed",
                    "score": 0.52,
                    "evidence": [f"Temperature span is {min(temperatures):.3g} K to {max(temperatures):.3g} K."],
                    "limitations": ["A reference temperature is only a background if the signal is known to vanish or become featureless there."],
                    "required_review": ["Confirm phase behavior before treating a temperature point as background."],
                }
            )
    if all_records:
        branches.append(
            {
                "branch_id": f"{measurement_family}.constant_offset",
                "branch_type": "constant_offset",
                "label": "Constant offset branch",
                "status": "proposed",
                "score": 0.28,
                "evidence": ["Intensity rows exist, so a constant baseline can be previewed if a user selects it."],
                "limitations": ["Constant offsets are rarely a defensible final background model without residual checks."],
                "required_review": ["Use only as a quick visualization branch unless validated by sidebands or controls."],
            }
        )
    if requested_branch:
        requested = str(requested_branch).strip()
        matched = False
        for branch in branches:
            if requested in {branch.get("branch_id"), branch.get("branch_type")}:
                branch["status"] = "manual_selected_preview"
                branch["selection_note"] = "Selected by --background-branch or JSON payload; still not an accepted reduction choice."
                matched = True
            elif branch.get("status") != "manual_selected_preview":
                branch["status"] = "proposed"
        if not matched:
            branches.append(
                {
                    "branch_id": f"{measurement_family}.manual_user_selected",
                    "branch_type": requested,
                    "label": f"Manual background branch: {requested}",
                    "status": "manual_selected_preview",
                    "score": 0.5,
                    "evidence": ["User requested this branch name; no deterministic candidate matched it."],
                    "limitations": ["The branch is recorded for provenance but may not be executable until inputs are supplied."],
                    "required_review": ["Supply scan IDs/regions/correction files needed by this manual branch."],
                }
            )
    else:
        selected = max(branches, key=lambda item: float(item.get("score") or 0.0))
        selected["status"] = "auto_selected_preview"
        selected["selection_note"] = "Highest-scoring discovered branch for preview only; not an accepted background choice."
    return sorted(branches, key=lambda item: (item.get("status") in {"auto_selected_preview", "manual_selected_preview"}, float(item.get("score") or 0.0)), reverse=True)


def _selected_background_branch(branches: list[dict[str, Any]]) -> dict[str, Any]:
    for branch in branches:
        if branch.get("status") in {"auto_selected_preview", "manual_selected_preview"}:
            return {
                "branch_id": branch.get("branch_id"),
                "branch_type": branch.get("branch_type"),
                "status": branch.get("status"),
                "score": branch.get("score"),
                "selection_note": branch.get("selection_note"),
            }
    return {}


def _write_background_branch_preview(output_dir: Path, *, measurement_family: str, rows: list[dict[str, Any]], selected_branch: dict[str, Any]) -> dict[str, Any]:
    branch_type = str(selected_branch.get("branch_type") or "").strip()
    if not branch_type:
        return {"status": "missing_input", "missing_inputs": ["selected_background_branch"]}
    if not rows:
        return {"status": "missing_input", "missing_inputs": ["preview_rows"]}
    if branch_type not in {"no_background", "constant_offset", "same_scan_sideband", "temperature_reference"}:
        return {
            "status": "unsupported",
            "branch_type": branch_type,
            "reason": "This branch is recorded for provenance but does not have an executable preview implementation yet, or requires reviewed external inputs.",
            "review_gate": "paired background and manual background branches require reviewed source/target matching before subtraction.",
        }
    if branch_type == "same_scan_sideband" and measurement_family != "constant_q_energy":
        return {
            "status": "unsupported",
            "branch_type": branch_type,
            "reason": "Same-scan sideband execution currently requires per-point scan rows, not one integrated row per temperature.",
        }
    if branch_type == "temperature_reference" and measurement_family != "temperature_series":
        return {
            "status": "unsupported",
            "branch_type": branch_type,
            "reason": "Temperature-reference execution only applies to temperature-series previews.",
        }

    execution_rows = []
    if measurement_family == "constant_q_energy":
        execution_rows = _constant_q_background_execution_rows(rows, branch_type)
        fieldnames = [
            "scan_id",
            "title",
            "path",
            "point_index",
            "energy_mev",
            "qx",
            "qy",
            "qz",
            "intensity",
            "monitor",
            "monitor_normalized_intensity",
            "background_estimate_intensity",
            "background_subtracted_intensity",
            "background_estimate_monitor_normalized",
            "background_subtracted_monitor_normalized",
            "background_branch_type",
            "background_execution_note",
            "y_column",
            "source_object_type",
        ]
    elif measurement_family == "temperature_series":
        execution_rows = _temperature_background_execution_rows(rows, branch_type)
        fieldnames = [
            "scan_id",
            "title",
            "path",
            "temperature_K",
            "intensity_sum",
            "intensity_max",
            "monitor_normalized_sum",
            "background_estimate_intensity",
            "background_subtracted_intensity",
            "background_estimate_monitor_normalized",
            "background_subtracted_monitor_normalized",
            "background_branch_type",
            "background_execution_note",
            "point_count",
            "x_column",
            "y_column",
            "source_object_type",
        ]
    else:
        return {"status": "unsupported", "branch_type": branch_type, "reason": f"No executable preview for measurement family {measurement_family}."}

    csv_path = output_dir / f"{measurement_family}_selected_background_preview.csv"
    _write_dict_csv(csv_path, execution_rows, fieldnames)
    return {
        "status": "success",
        "branch_type": branch_type,
        "path": str(csv_path),
        "row_count": len(execution_rows),
        "policy": "Executable preview branch only. This is not an accepted subtraction or final reduction without review.",
        "lineage_note": "Raw files remain immutable; background estimates are local derived columns.",
    }


def _constant_q_background_execution_rows(rows: list[dict[str, Any]], branch_type: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("scan_id") or "unknown"), []).append(row)
    execution_rows: list[dict[str, Any]] = []
    for scan_rows in grouped.values():
        ordered = sorted(scan_rows, key=lambda item: (_numeric_or_none(item.get("energy_mev")) is None, _numeric_or_none(item.get("energy_mev")) or 0.0, item.get("point_index") or 0))
        intensity_background = _background_estimate_for_values([row.get("intensity") for row in ordered], branch_type)
        monitor_background = _background_estimate_for_values([row.get("monitor_normalized_intensity") for row in ordered], branch_type)
        note = _background_execution_note(branch_type)
        for row in ordered:
            execution = dict(row)
            execution.update(
                {
                    "background_estimate_intensity": intensity_background,
                    "background_subtracted_intensity": _subtract_numeric(row.get("intensity"), intensity_background),
                    "background_estimate_monitor_normalized": monitor_background,
                    "background_subtracted_monitor_normalized": _subtract_numeric(row.get("monitor_normalized_intensity"), monitor_background),
                    "background_branch_type": branch_type,
                    "background_execution_note": note,
                }
            )
            execution_rows.append(execution)
    return execution_rows


def _temperature_background_execution_rows(rows: list[dict[str, Any]], branch_type: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: (_numeric_or_none(item.get("temperature_K")) is None, _numeric_or_none(item.get("temperature_K")) or 0.0, str(item.get("scan_id") or "")))
    if branch_type == "temperature_reference":
        reference = ordered[-1] if ordered else {}
        intensity_background = _numeric_or_none(reference.get("intensity_sum"))
        monitor_background = _numeric_or_none(reference.get("monitor_normalized_sum"))
        note = "Background estimate is the highest-temperature preview point; valid only if reviewed as a true reference state."
    else:
        intensity_background = _background_estimate_for_values([row.get("intensity_sum") for row in ordered], branch_type)
        monitor_background = _background_estimate_for_values([row.get("monitor_normalized_sum") for row in ordered], branch_type)
        note = _background_execution_note(branch_type)
    execution_rows: list[dict[str, Any]] = []
    for row in ordered:
        execution = dict(row)
        execution.update(
            {
                "background_estimate_intensity": intensity_background,
                "background_subtracted_intensity": _subtract_numeric(row.get("intensity_sum"), intensity_background),
                "background_estimate_monitor_normalized": monitor_background,
                "background_subtracted_monitor_normalized": _subtract_numeric(row.get("monitor_normalized_sum"), monitor_background),
                "background_branch_type": branch_type,
                "background_execution_note": note,
            }
        )
        execution_rows.append(execution)
    return execution_rows


def _background_estimate_for_values(values: list[Any], branch_type: str) -> float | None:
    numeric = [_numeric_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    if branch_type == "no_background":
        return 0.0
    if branch_type == "constant_offset":
        return min(numeric)
    if branch_type == "same_scan_sideband":
        sideband_count = max(1, int(round(len(numeric) * 0.2)))
        sideband_values = numeric[:sideband_count] + numeric[-sideband_count:]
        return sum(sideband_values) / len(sideband_values)
    return None


def _background_execution_note(branch_type: str) -> str:
    if branch_type == "no_background":
        return "Raw preview branch; no background estimate was subtracted."
    if branch_type == "constant_offset":
        return "Preview subtracts the minimum available value as a constant offset per scan/series."
    if branch_type == "same_scan_sideband":
        return "Preview subtracts the mean of edge points as a same-scan sideband estimate."
    return "Preview branch execution."


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _subtract_numeric(value: Any, background: float | None) -> float | None:
    numeric = _numeric_or_none(value)
    if numeric is None or background is None:
        return None
    return numeric - background


def _background_hint_text(record: dict[str, Any]) -> bool:
    text = " ".join(str(record.get(key) or "") for key in ("title", "path", "source_object_id", "scan_id")).lower()
    return any(term in text for term in ("background", "bkg", "offpeak", "off-peak", "off_peak", "empty", "mount"))


def _paired_background_candidates(*, measurement_family: str, all_records: list[dict[str, Any]], background_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signal_records = [record for record in all_records if record not in background_records]
    candidates: list[dict[str, Any]] = []
    for background in background_records[:20]:
        scored = sorted(
            (_paired_background_score(measurement_family, signal, background) for signal in signal_records[:80]),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        for match in [item for item in scored if float(item.get("score") or 0.0) > 0.25][:3]:
            candidates.append(match)
    return sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:20]


def _paired_background_score(measurement_family: str, signal: dict[str, Any], background: dict[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    mismatches: list[str] = []
    score = 0.2
    signal_variable = _record_scan_variable(signal)
    background_variable = _record_scan_variable(background)
    if signal_variable and background_variable and signal_variable.lower() == background_variable.lower():
        score += 0.18
        evidence.append(f"scan variable matches: {signal_variable}")
    elif signal_variable or background_variable:
        mismatches.append(f"scan variable differs: {signal_variable or 'missing'} vs {background_variable or 'missing'}")

    signal_mode = str(signal.get("detector_mode") or "").strip()
    background_mode = str(background.get("detector_mode") or "").strip()
    if signal_mode and background_mode and signal_mode == background_mode:
        score += 0.12
        evidence.append(f"detector mode matches: {signal_mode}")
    elif signal_mode or background_mode:
        mismatches.append(f"detector mode differs: {signal_mode or 'missing'} vs {background_mode or 'missing'}")

    signal_q = _record_q_center(signal)
    background_q = _record_q_center(background)
    if signal_q and background_q:
        q_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(signal_q, background_q, strict=False)))
        if q_distance <= 0.05:
            score += 0.2
            evidence.append(f"Q centers are close: delta={q_distance:.4g}")
        elif q_distance <= 0.2:
            score += 0.08
            evidence.append(f"Q centers are moderately close: delta={q_distance:.4g}")
        else:
            mismatches.append(f"Q centers differ: delta={q_distance:.4g}")

    if measurement_family == "constant_q_energy":
        signal_energy = _numeric_range(signal.get("energy_values") or [])
        background_energy = _numeric_range(background.get("energy_values") or [])
        if signal_energy and background_energy:
            overlap = _range_overlap_fraction(signal_energy, background_energy)
            if overlap > 0.8:
                score += 0.2
                evidence.append(f"energy coverage overlaps strongly: {overlap:.2f}")
            elif overlap > 0.25:
                score += 0.08
                evidence.append(f"energy coverage partially overlaps: {overlap:.2f}")
            else:
                mismatches.append("energy coverage has little overlap")
    elif measurement_family == "temperature_series":
        signal_temp = _record_temperature(signal)
        background_temp = _record_temperature(background)
        if signal_temp is not None and background_temp is not None:
            delta_t = abs(signal_temp - background_temp)
            if delta_t <= 5:
                score += 0.16
                evidence.append(f"temperature is close: delta={delta_t:.3g} K")
            elif delta_t <= 25:
                score += 0.06
                evidence.append(f"temperature is moderately close: delta={delta_t:.3g} K")
            else:
                mismatches.append(f"temperature differs: delta={delta_t:.3g} K")

    if _record_point_count(signal) and _record_point_count(background):
        ratio = min(_record_point_count(signal), _record_point_count(background)) / max(_record_point_count(signal), _record_point_count(background))
        if ratio > 0.7:
            score += 0.08
            evidence.append(f"point counts are similar: ratio={ratio:.2f}")
        elif ratio < 0.25:
            mismatches.append(f"point counts differ strongly: ratio={ratio:.2f}")

    return {
        "match_id": f"paired-bg-{_safe_id(str(_record_scan_id(signal) or 'signal'))}-{_safe_id(str(_record_scan_id(background) or 'background'))}",
        "target_scan_id": _record_scan_id(signal),
        "background_scan_id": _record_scan_id(background),
        "target_path": signal.get("path"),
        "background_path": background.get("path"),
        "score": round(min(score, 0.95), 3),
        "status": "proposed_review_required",
        "evidence": evidence or ["background filename/title hint exists"],
        "mismatches": mismatches,
        "required_review": ["Confirm matching Q/E/temperature/configuration and counting conditions before subtraction."],
    }


def _record_scan_variable(record: dict[str, Any]) -> str | None:
    value = record.get("scan_variable")
    if value:
        return str(value)
    plot = record.get("plot_data") or {}
    value = plot.get("x_column")
    return str(value) if value else None


def _record_q_center(record: dict[str, Any]) -> tuple[float, float, float] | None:
    q_values = record.get("q_values") or []
    if q_values and isinstance(q_values[0], (list, tuple)) and len(q_values[0]) >= 3:
        numeric = [_numeric_or_none(item) for item in q_values[0][:3]]
        if all(value is not None for value in numeric):
            return (float(numeric[0]), float(numeric[1]), float(numeric[2]))
    q_center = record.get("q_center") or record.get("q_center_coverage") or {}
    if isinstance(q_center, dict):
        values = []
        for key in ("H", "K", "L", "QX", "QY", "QZ", "qx", "qy", "qz"):
            value = q_center.get(key)
            if isinstance(value, dict):
                value = value.get("mean") or value.get("min") or value.get("max")
            number = _numeric_or_none(value)
            if number is not None:
                values.append(number)
            if len(values) == 3:
                return (values[0], values[1], values[2])
    return None


def _numeric_range(values: list[Any]) -> tuple[float, float] | None:
    numeric = [_numeric_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return (min(numeric), max(numeric))


def _range_overlap_fraction(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_min, left_max = left
    right_min, right_max = right
    width = max(left_max - left_min, right_max - right_min, 1e-12)
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    return overlap / width


def _safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:80] or "unknown"


def _record_scan_id(record: dict[str, Any]) -> str | None:
    value = record.get("source_object_id") or record.get("scan_id")
    return str(value) if value is not None else None


def _record_point_count(record: dict[str, Any]) -> int:
    if isinstance(record.get("row_count"), int):
        return int(record["row_count"])
    plot = record.get("plot_data") or {}
    if isinstance(plot.get("x"), list):
        return len(plot["x"])
    values = record.get("energy_values") or record.get("temperature_values") or record.get("intensity_values") or []
    return len(values) if isinstance(values, list) else 0


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_constant_q_preview_plot(path: Path, records: list[dict[str, Any]], reader_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting install.
        return {"plot_status": "unavailable", "plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plotted = 0
    if reader_records:
        for record in reader_records:
            xs = record.get("energy_values") or []
            ys = record.get("intensity_values") or []
            if not xs or not ys:
                continue
            ax.plot(xs[: len(ys)], ys, marker="o", linewidth=1.1, markersize=2.5, alpha=0.75, label=str(record.get("source_object_id")))
            plotted += 1
    else:
        for record in records:
            plot = record.get("plot_data") or {}
            xs = plot.get("x") or []
            ys = plot.get("y") or []
            if not xs or not ys:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.1, markersize=2.5, alpha=0.75, label=str(record.get("scan_id")))
            plotted += 1
    ax.set_xlabel("Energy transfer or E axis")
    ax.set_ylabel("Intensity")
    ax.set_title("Constant-Q Energy Preview")
    if plotted <= 10:
        ax.legend(fontsize="x-small")
    else:
        ax.text(0.01, 0.99, f"{plotted} scans plotted; legend suppressed", transform=ax.transAxes, va="top", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"plot_status": "success", "plot_path": str(path), "plot_scan_count": plotted}


def _write_temperature_series_preview_plot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting install.
        return {"plot_status": "unavailable", "plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    temps = [row["temperature_K"] for row in rows]
    sums = [row["intensity_sum"] for row in rows]
    ax.plot(temps, sums, "o", alpha=0.75)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Integrated preview intensity")
    ax.set_title("Temperature-Series Preview")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"plot_status": "success", "plot_path": str(path)}


def _write_psd_channel_temperature_plot(path: Path, parsed: dict[str, Any]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting install.
        return {"plot_status": "unavailable", "plot_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    counts = parsed.get("psd_counts") or []
    temperatures = parsed.get("temperature_values") or []
    if not counts or not temperatures:
        return {"plot_status": "missing_input", "plot_warning": "PSD counts or temperatures are missing."}
    paired_rows = sorted(
        [(float(temperature), row) for temperature, row in zip(temperatures, counts) if isinstance(temperature, (int, float))],
        key=lambda item: item[0],
    )
    if not paired_rows:
        return {"plot_status": "missing_input", "plot_warning": "PSD temperature values are not numeric."}
    sorted_temperatures = [item[0] for item in paired_rows]
    sorted_counts = [item[1] for item in paired_rows]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = ax.imshow(
        sorted_counts,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[-0.5, len(sorted_counts[0]) - 0.5, min(sorted_temperatures), max(sorted_temperatures)],
    )
    ax.set_xlabel("PSD channel")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"PSD Channel x Temperature Preview: {parsed.get('source_object_id')}")
    if len(sorted_temperatures) >= 2:
        ticks = [min(sorted_temperatures), sorted_temperatures[len(sorted_temperatures) // 2], max(sorted_temperatures)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{temperature:.1f} K" for temperature in ticks])
    fig.colorbar(image, ax=ax, label="Raw counts")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "plot_status": "success",
        "plot_path": str(path),
        "plot_y_axis": "temperature_K_sorted_ascending",
        "plot_temperature_range": [min(sorted_temperatures), max(sorted_temperatures)],
        "temperature_sort_applied": True,
    }


def _reduction_artifact_index(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for item in artifacts:
        for kind in ("path", "plot_path", "reader_manifest", "background_branch_manifest"):
            if item.get(kind):
                index.append({"group": "reduction_preview", "kind": f"{item.get('artifact_type')}.{kind}", "path": item.get(kind), "href": _local_href(item.get(kind))})
    return index


def _collect_review_queue(material: dict[str, Any], planning: dict[str, Any], dashboard: dict[str, Any], recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    if material.get("ambiguity_reason"):
        queue.append(
            {
                "priority": "high",
                "topic": "Resolve material ambiguity",
                "action": material.get("ambiguity_reason"),
                "source": "material_identity",
            }
        )
    missing_inputs = (planning.get("recommendations") or {}).get("missing_inputs_to_request") or []
    for item in missing_inputs:
        queue.append({"priority": "high", "topic": "Missing input", "action": item, "source": "planning_brief"})
    if dashboard.get("spurion_candidate_count"):
        queue.append(
            {
                "priority": "medium",
                "topic": "Spurion/artifact triage",
                "action": "Review suspicious scans and suggested diagnostic scans before extending inelastic measurement families.",
                "source": "experiment_dashboard",
            }
        )
    if dashboard.get("order_parameter_candidate_count"):
        queue.append(
            {
                "priority": "medium",
                "topic": "Order-parameter candidates",
                "action": "Review temperature-dependent elastic scan families before fitting or interpreting transition behavior.",
                "source": "experiment_dashboard",
            }
        )
    if recipes:
        queue.append(
            {
                "priority": "medium",
                "topic": "Scan recipes",
                "action": f"Review the top-ranked header-derived scan recipes first; current top recipe is {recipes[0].get('title') or 'untitled'}. Confirm axes, count mode, and range before reuse.",
                "source": "header_scan_recipes",
            }
        )
    if not queue:
        queue.append(
            {
                "priority": "normal",
                "topic": "Review assembled workspace",
                "action": "Open the dashboard and planning brief, then add missing experiment-specific configuration if needed.",
                "source": "experiment_workspace",
            }
        )
    return queue


def _collect_top_precedents(planning: dict[str, Any]) -> list[dict[str, Any]]:
    public = planning.get("user_facing_top_precedents") or []
    if public:
        return [
            {
                "candidate_id": item.get("pattern_id"),
                "title": "Private precedent pattern",
                "score": item.get("score"),
                "confidence_label": item.get("confidence_label"),
                "scan_count": item.get("scan_count"),
                "datafile_count": item.get("datafile_count"),
                "start_date": None,
                "scan_family_priors": item.get("scan_family_priors") or [],
                "fixed_energy_values": item.get("fixed_energy_values") or [],
                "temperature_values": item.get("temperature_values") or [],
                "privacy_label": item.get("privacy_note"),
            }
            for item in public[:12]
        ]
    precedents = []
    for candidate in (planning.get("top_precedent_candidates") or [])[:12]:
        precedents.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "title": candidate.get("title"),
                "score": candidate.get("score"),
                "confidence_label": candidate.get("confidence_label"),
                "scan_count": candidate.get("scan_count"),
                "datafile_count": candidate.get("datafile_count"),
                "start_date": candidate.get("start_date"),
                "scan_family_priors": candidate.get("scan_family_priors") or [],
                "fixed_energy_values": candidate.get("fixed_energy_values") or [],
                "temperature_values": candidate.get("temperature_values") or [],
                "privacy_label": candidate.get("privacy_label"),
            }
        )
    return precedents


def _collect_artifact_index(dashboard: dict[str, Any], planning: dict[str, Any]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for group, artifacts in [("dashboard", dashboard.get("artifacts") or {}), ("planning", planning.get("artifacts") or {})]:
        for kind, path in artifacts.items():
            index.append({"group": group, "kind": kind, "path": str(path), "href": _local_href(path)})
    return index


def _workspace_summary(
    dashboard: dict[str, Any],
    planning: dict[str, Any],
    recipes: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    review_queue: list[dict[str, Any]],
    reduction_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence = planning.get("confidence") or {}
    psd_readiness = (reduction_readiness or {}).get("psd_channel_temperature") or {}
    return {
        "dashboard_status": dashboard.get("status"),
        "planning_status": planning.get("status"),
        "scan_count": dashboard.get("scan_count"),
        "inelastic_scan_count": dashboard.get("inelastic_scan_count"),
        "recipe_count": len(recipes),
        "recommendation_count": len(recommendations),
        "review_item_count": len(review_queue),
        "psd_reduction_status": psd_readiness.get("status"),
        "psd_calibration_status": psd_readiness.get("calibration_status"),
        "psd_q_temperature_status": psd_readiness.get("q_temperature_status"),
        "overall_probability_style_confidence": confidence.get("overall_probability_style_confidence"),
        "evidence_channels": planning.get("evidence_channels") or {},
    }


def _workspace_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Workspace",
        "",
        f"Status: `{report.get('status')}`",
        f"Material: `{report.get('material')}`",
        f"Goal: {report.get('goal') or 'not supplied'}",
        "",
        "## Material Identity",
    ]
    material = report.get("material_identity") or {}
    lines.extend(
        [
            f"- Backend: `{material.get('parser_backend')}`",
            f"- Normalized formula: `{material.get('normalized_formula')}`",
            f"- Confidence: `{material.get('confidence')}`",
        ]
    )
    if material.get("ambiguity_reason"):
        lines.append(f"- Ambiguity: {material.get('ambiguity_reason')}")
    sections = report.get("workspace_sections") or {}
    summary = report.get("workspace_summary") or {}
    dashboard = sections.get("dashboard") or {}
    planning = sections.get("planning_brief") or {}
    lines.extend(
        [
            "",
            "## Summary",
            f"- Dashboard status: `{summary.get('dashboard_status')}`",
            f"- Planning status: `{summary.get('planning_status')}`",
            f"- Scans: `{summary.get('scan_count')}`",
            f"- Header-derived recipes: `{summary.get('recipe_count')}`",
            f"- Review items: `{summary.get('review_item_count')}`",
            f"- Probability-style confidence: `{summary.get('overall_probability_style_confidence')}`",
            f"- PSD reduction status: `{summary.get('psd_reduction_status')}`",
            f"- PSD calibration status: `{summary.get('psd_calibration_status')}`",
            "",
            "## Main Entry Points",
        ]
    )
    for item in sections.get("artifact_index") or []:
        if item.get("kind") in {"dashboard_html", "html", "report_html"} or str(item.get("path", "")).endswith(".html"):
            lines.append(f"- {item.get('group')}.{item.get('kind')}: `{item.get('path')}`")
    lines.extend(
        [
            "",
            "## Review Queue",
        ]
    )
    for item in sections.get("review_queue") or []:
        lines.append(f"- `{item.get('priority')}` {item.get('topic')}: {item.get('action')}")
    lines.extend(["", "## User-Facing Recommendation Summary"])
    user_recs = sections.get("user_facing_recommendations") or {}
    lines.append(f"- Policy: {user_recs.get('privacy_boundary') or 'not available'}")
    for item in user_recs.get("items") or []:
        lines.append(f"- `{item.get('confidence')}` {item.get('topic')}: {item.get('recommendation')}")
    lines.extend(
        [
            "",
            "## Dashboard",
            f"- Status: `{dashboard.get('status')}`",
            f"- Scan count: `{dashboard.get('scan_count')}`",
            f"- Spurion/artifact candidates: `{dashboard.get('spurion_candidate_count')}`",
            "",
            "## Planning Brief",
            f"- Status: `{planning.get('status')}`",
            f"- Safe conclusion: {planning.get('safe_conclusion') or 'not available'}",
            "",
            "## Header-Derived Scan Recipes",
        ]
    )
    for recipe in sections.get("scan_recipes") or []:
        lines.append(
            f"- `{recipe.get('title')}` `{recipe.get('jtype')}` score=`{recipe.get('recipe_rank_score')}` "
            f"axes={recipe.get('range_axes')} npts={recipe.get('npts')} count_type={recipe.get('count_type')}"
        )
    if not sections.get("scan_recipes"):
        lines.append("- No header-derived scan recipes were available.")
    replay_plan = sections.get("nice_replay_training_plan") or {}
    lines.extend(["", "## NICE Replay Training Plan"])
    lines.append(f"- Status: `{replay_plan.get('status')}`")
    lines.append(f"- Boundary: `{replay_plan.get('workflow_boundary')}`")
    lines.append(f"- Recipe count: `{replay_plan.get('recipe_count')}`")
    if replay_plan.get("artifact_path"):
        lines.append(f"- Artifact: `{replay_plan.get('artifact_path')}`")
    lines.extend(["", "## Corpus Status"])
    corpus = sections.get("corpus_status") or {}
    lines.append(f"- Status: `{corpus.get('status')}`")
    lines.append(f"- Top result count: `{corpus.get('top_result_count')}`")
    if corpus.get("next_step"):
        lines.append(f"- Next step: {corpus.get('next_step')}")
    lines.extend(["", "## Reduction DAG Previews"])
    for item in sections.get("reduction_previews") or []:
        lines.append(f"- `{item.get('template_id')}` `{item.get('status')}`: {item.get('reason')}")
    for item in sections.get("reduction_runs") or []:
        lines.append(f"- run `{item.get('template_id')}` status=`{item.get('status')}` missing={item.get('missing_inputs')}")
    for item in sections.get("reduction_artifacts") or []:
        lines.append(f"- artifact `{item.get('artifact_type')}` status=`{item.get('status')}` path=`{item.get('path')}`")
    readiness = sections.get("reduction_readiness") or {}
    psd = readiness.get("psd_channel_temperature") or {}
    lines.extend(
        [
            "",
            "## Reduction Readiness",
            f"- PSD channel-temperature status: `{psd.get('status')}`",
            f"- PSD calibration status: `{psd.get('calibration_status')}`",
            f"- Q/T status: `{psd.get('q_temperature_status')}`",
            f"- Publication-like map status: `{psd.get('publication_like_map_status')}`",
        ]
    )
    if psd.get("reader_manifest"):
        lines.append(f"- Reader-output manifest: `{psd.get('reader_manifest')}`")
    lines.extend(["", "## Alignment / Structure Factors"])
    alignment = sections.get("alignment_status") or {}
    lines.append(f"- Status: `{alignment.get('status')}`")
    lines.append(f"- Candidate count: `{alignment.get('candidate_count')}`")
    if alignment.get("next_step"):
        lines.append(f"- Next step: {alignment.get('next_step')}")
    lines.extend(["", "## Top Precedent Patterns"])
    for item in sections.get("top_precedents") or []:
        lines.append(f"- `{item.get('candidate_id')}` score=`{item.get('score')}` families={item.get('scan_family_priors')} title={item.get('title')}")
    lines.extend(["", "## Recommended Next Actions"])
    for item in sections.get("recommendations") or []:
        lines.append(f"- {item.get('recommendation')} (probability-style score: `{item.get('probability')}`)")
    lines.extend(["", "## Safety Boundary"])
    for item in report.get("safety_boundary") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Artifacts"])
    for label, artifacts in [("dashboard", dashboard.get("artifacts") or {}), ("planning", planning.get("artifacts") or {})]:
        for key, path in artifacts.items():
            lines.append(f"- {label}.{key}: `{path}`")
    return "\n".join(lines) + "\n"


def _workspace_html(report: dict[str, Any]) -> str:
    sections = report.get("workspace_sections") or {}
    summary = report.get("workspace_summary") or {}
    material = report.get("material_identity") or {}
    dashboard = sections.get("dashboard") or {}
    planning = sections.get("planning_brief") or {}
    recipes = sections.get("scan_recipes") or []
    recs = sections.get("recommendations") or []
    user_recs = sections.get("user_facing_recommendations") or {}
    reduction_readiness = sections.get("reduction_readiness") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Workspace</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18222e; background: #f7f9fb; }}
header {{ padding: 24px 32px; background: #ffffff; border-bottom: 1px solid #d9e1ea; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
nav a {{ text-decoration: none; border: 1px solid #c7d3df; border-radius: 6px; padding: 6px 9px; background: #f8fbff; }}
section {{ background: #ffffff; border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
h1, h2 {{ margin-top: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
.metric {{ border: 1px solid #e1e7ef; border-radius: 8px; padding: 12px; background: #fbfcfe; }}
.metric .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 14px; }}
.controls input, .controls select {{ padding: 8px; border: 1px solid #c7d3df; border-radius: 6px; min-width: 220px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e1e7ef; vertical-align: top; }}
code {{ background: #eef3f8; padding: 1px 4px; border-radius: 4px; }}
a {{ color: #205493; }}
.warn {{ color: #7a4c00; }}
.pill {{ display: inline-block; border: 1px solid #d3dde8; border-radius: 999px; padding: 2px 8px; margin: 1px; background: #f8fbff; }}
.priority-high {{ border-left: 5px solid #b03a2e; }}
.priority-medium {{ border-left: 5px solid #c48100; }}
.priority-normal {{ border-left: 5px solid #4c6f91; }}
.muted {{ color: #5f6f7f; }}
</style>
</head>
<body>
<header>
<h1>Experiment Workspace</h1>
<p>{_esc(report.get("material"))} | {_esc(report.get("goal") or "No goal supplied")}</p>
<nav>
<a href="#summary">Summary</a>
<a href="#dashboard-entry">Dashboard</a>
<a href="#review">Review Queue</a>
<a href="#user-recommendations">Recommendations</a>
<a href="#artifacts">Artifacts</a>
<a href="#recipes">Scan Recipes</a>
<a href="#nice-replay">NICE Replay</a>
<a href="#corpus">Corpus</a>
<a href="#reduction">Reduction DAG</a>
<a href="#reduction-readiness">Readiness</a>
<a href="#alignment">Alignment</a>
<a href="#precedents">Precedents</a>
<a href="#actions">Internal Actions</a>
<a href="#safety">Safety</a>
</nav>
</header>
<main>
<section id="summary">
<h2>Summary</h2>
<div class="grid">
<div class="metric"><strong>Dashboard</strong><div class="value">{_esc(summary.get("dashboard_status"))}</div><span class="muted">{_esc(summary.get("scan_count"))} scans</span></div>
<div class="metric"><strong>Planning</strong><div class="value">{_esc(summary.get("planning_status"))}</div><span class="muted">confidence {_esc(summary.get("overall_probability_style_confidence"))}</span></div>
<div class="metric"><strong>Recipes</strong><div class="value">{_esc(summary.get("recipe_count"))}</div><span class="muted">deduplicated from headers</span></div>
<div class="metric"><strong>Review Items</strong><div class="value">{_esc(summary.get("review_item_count"))}</div><span class="muted">human review queue</span></div>
<div class="metric"><strong>PSD Preview</strong><div class="value">{_esc(summary.get("psd_reduction_status"))}</div><span class="muted">calibration {_esc(summary.get("psd_calibration_status"))}</span></div>
</div>
</section>
<section id="dashboard-entry">
<h2>Dashboard Entry Point</h2>
{_dashboard_entry_html(dashboard)}
</section>
<section id="material">
<h2>Material Identity</h2>
<div class="grid">
<div class="metric"><strong>Backend</strong><br><code>{_esc(material.get("parser_backend"))}</code></div>
<div class="metric"><strong>Normalized Formula</strong><br><code>{_esc(material.get("normalized_formula"))}</code></div>
<div class="metric"><strong>Confidence</strong><br><code>{_esc(material.get("confidence"))}</code></div>
</div>
{f'<p class="warn">{_esc(material.get("ambiguity_reason"))}</p>' if material.get("ambiguity_reason") else ''}
</section>
<section id="review">
<h2>Review Queue</h2>
{_review_queue_html(sections.get("review_queue") or [])}
</section>
<section id="user-recommendations">
<h2>User-Facing Recommendation Summary</h2>
{_user_recommendation_html(user_recs)}
</section>
<section id="channels">
<h2>Workspace Status</h2>
<div class="grid">
<div class="metric"><strong>Dashboard</strong><br>{_esc(dashboard.get("status"))}<br>scans: <code>{_esc(dashboard.get("scan_count"))}</code></div>
<div class="metric"><strong>Planning Brief</strong><br>{_esc(planning.get("status"))}<br>confidence: <code>{_esc((planning.get("confidence") or {}).get("overall_probability_style_confidence"))}</code></div>
<div class="metric"><strong>Recipes</strong><br><code>{len(recipes)}</code> deduplicated header-derived patterns</div>
</div>
<h3>Evidence Channels</h3>
{_evidence_channel_html(summary.get("evidence_channels") or {})}
</section>
<section id="artifacts">
<h2>Key Artifacts</h2>
{_artifact_table(sections.get("artifact_index") or [])}
</section>
<section id="recipes">
<h2>Header-Derived Scan Recipes</h2>
<div class="controls">
<input id="recipeSearch" type="search" placeholder="Filter recipes by title, axis, type, family">
<select id="recipeFamily"><option value="">All recipe families</option>{_recipe_family_options(recipes)}</select>
</div>
{_recipe_table(recipes)}
</section>
<section id="nice-replay">
<h2>NICE Replay Training Harness</h2>
{_nice_replay_training_plan_html(sections.get("nice_replay_training_plan") or {})}
</section>
<section id="corpus">
<h2>Corpus Evidence</h2>
{_corpus_status_html(sections.get("corpus_status") or {})}
</section>
<section id="reduction">
<h2>Reduction DAG Preview</h2>
{_reduction_preview_html(sections.get("reduction_previews") or [])}
{_reduction_run_html(sections.get("reduction_runs") or [])}
{_reduction_artifact_html(sections.get("reduction_artifacts") or [])}
</section>
<section id="reduction-readiness">
<h2>Reduction Readiness</h2>
{_reduction_readiness_html(reduction_readiness)}
</section>
<section id="alignment">
<h2>Alignment and Structure Factors</h2>
{_alignment_status_html(sections.get("alignment_status") or {})}
</section>
<section id="precedents">
<h2>Top Precedent Patterns</h2>
{_precedent_table(sections.get("top_precedents") or [])}
</section>
<section id="actions">
<h2>Recommended Next Actions</h2>
{_recommendation_table(recs)}
</section>
<section id="safety">
<h2>Safety Boundary</h2>
<ul>{''.join(f'<li>{_esc(item)}</li>' for item in report.get("safety_boundary") or [])}</ul>
</section>
</main>
<script>
const search = document.getElementById('recipeSearch');
const family = document.getElementById('recipeFamily');
function filterRecipes() {{
  const text = (search?.value || '').toLowerCase();
  const fam = (family?.value || '').toLowerCase();
  document.querySelectorAll('tr[data-recipe-row]').forEach(row => {{
    const haystack = row.textContent.toLowerCase();
    const families = (row.getAttribute('data-families') || '').toLowerCase();
    row.style.display = haystack.includes(text) && (!fam || families.includes(fam)) ? '' : 'none';
  }});
}}
search?.addEventListener('input', filterRecipes);
family?.addEventListener('change', filterRecipes);
</script>
</body>
</html>
"""


def _dashboard_entry_html(dashboard: dict[str, Any]) -> str:
    artifacts = dashboard.get("artifacts") or {}
    html_path = (
        artifacts.get("dashboard_html")
        or artifacts.get("html")
        or artifacts.get("report_html")
        or artifacts.get("dashboard")
    )
    refresh_script = artifacts.get("refresh_script")
    dash_app = artifacts.get("dash_app")
    if not html_path:
        return (
            f"<p>Dashboard status: <code>{_esc(dashboard.get('status'))}</code>.</p>"
            "<p class=\"muted\">No dashboard HTML artifact was available for inline preview.</p>"
        )
    href = _local_href(html_path)
    extras = []
    if refresh_script:
        extras.append(f'<a href="{_esc(_local_href(refresh_script))}">Refresh script</a>')
    if dash_app:
        extras.append(f'<a href="{_esc(_local_href(dash_app))}">Auto-refresh Dash app</a>')
    extra_html = f'<p class="muted">{" | ".join(extras)}. The Dash app watches local JSON and refreshes linked plots when rerun.</p>' if extras else ""
    return (
        f'<p><a href="{_esc(href)}">Open the full experiment dashboard</a>. '
        "This linked dashboard remains the richer plot/table review surface for the experiment.</p>"
        f"{extra_html}"
        f'<iframe title="Experiment dashboard preview" src="{_esc(href)}" '
        'style="width:100%; min-height:520px; border:1px solid #d9e1ea; border-radius:8px; background:#fff;"></iframe>'
    )


def _review_queue_html(queue: list[dict[str, Any]]) -> str:
    rows = []
    for item in queue:
        priority = str(item.get("priority") or "normal")
        rows.append(
            f'<div class="metric priority-{_esc(priority)}">'
            f"<strong>{_esc(priority.upper())}: {_esc(item.get('topic'))}</strong>"
            f"<p>{_esc(item.get('action'))}</p>"
            f'<span class="muted">{_esc(item.get("source"))}</span>'
            "</div>"
        )
    return '<div class="grid">' + "".join(rows) + "</div>"


def _user_recommendation_html(summary: dict[str, Any]) -> str:
    if not summary:
        return "<p>No user-facing recommendation summary was available.</p>"
    rows = []
    for item in summary.get("items") or []:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('topic'))}</td>"
            f"<td>{_esc(item.get('recommendation'))}</td>"
            f"<td>{_esc(item.get('confidence'))}</td>"
            f"<td>{_esc(item.get('evidence_class'))}</td>"
            f"<td><code>{_esc(json.dumps(item.get('evidence_summary') or {}, sort_keys=True, default=str))}</code></td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No recommendation items were generated.</td></tr>')
    return (
        f'<p class="muted">{_esc(summary.get("privacy_boundary") or "")}</p>'
        f'<p class="muted">Calibration: {_esc(summary.get("calibration") or "probability-style decision support")}</p>'
        "<table><thead><tr><th>Topic</th><th>Recommendation</th><th>Confidence</th><th>Evidence Class</th><th>Summary</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _evidence_channel_html(channels: dict[str, Any]) -> str:
    if not channels:
        return "<p>No evidence-channel status was available.</p>"
    return "<p>" + " ".join(f'<span class="pill">{_esc(key)}: {_esc(value)}</span>' for key, value in channels.items()) + "</p>"


def _artifact_table(artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        return "<p>No linked artifacts were available.</p>"
    rows = []
    for item in artifacts:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('group'))}</td>"
            f"<td>{_esc(item.get('kind'))}</td>"
            f'<td><a href="{_esc(item.get("href"))}">{_esc(Path(str(item.get("path"))).name)}</a></td>'
            "</tr>"
        )
    return "<table><thead><tr><th>Group</th><th>Kind</th><th>File</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _recipe_table(recipes: list[dict[str, Any]]) -> str:
    if not recipes:
        return "<p>No header-derived scan recipes were available.</p>"
    rows = []
    for recipe in recipes:
        ranges = "; ".join(_format_range(item) for item in recipe.get("ranges") or [])
        families = recipe.get("family_priors") or []
        score = recipe.get("recipe_rank_score")
        reasons = "; ".join(str(item) for item in recipe.get("recipe_rank_reasons") or [])
        rows.append(
            f'<tr data-recipe-row data-families="{_esc(" ".join(str(item) for item in families))}">'
            f"<td>{_esc(recipe.get('title'))}</td>"
            f"<td>{_esc(score)}</td>"
            f"<td>{_esc(reasons)}</td>"
            f"<td>{_esc(recipe.get('jtype'))}</td>"
            f"<td>{_esc(recipe.get('npts'))}</td>"
            f"<td>{_esc(recipe.get('counts'))}</td>"
            f"<td>{_esc(recipe.get('count_type'))}</td>"
            f"<td>{_esc(ranges)}</td>"
            f"<td>{_pill_list(families)}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Title</th><th>Rank</th><th>Rank Rationale</th><th>Type</th><th>Npts</th><th>Counts</th><th>Count Type</th><th>Ranges</th><th>Family Priors</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _nice_replay_training_plan_html(plan: dict[str, Any]) -> str:
    if not plan:
        return "<p>No NICE replay training plan was available.</p>"
    rows = []
    for recipe in plan.get("recipes") or []:
        rows.append(
            "<tr>"
            f"<td><code>{_esc(recipe.get('replay_recipe_id'))}</code><br>{_esc(recipe.get('source_title'))}</td>"
            f"<td>{_esc(recipe.get('recommended_replay_mode'))}</td>"
            f"<td>{_esc(recipe.get('jtype'))}</td>"
            f"<td>{_esc(recipe.get('range_axes'))}</td>"
            f"<td>{_esc(recipe.get('translation_status'))}</td>"
            "</tr>"
        )
    modes = "".join(f"<li><code>{_esc(item.get('mode'))}</code>: {_esc(item.get('purpose'))}</li>" for item in plan.get("modes") or [])
    artifact = f'<p><a href="{_esc(_local_href(plan.get("artifact_path")))}">Replay training plan JSON</a></p>' if plan.get("artifact_path") else ""
    return (
        f'<p class="muted">Status: <code>{_esc(plan.get("status"))}</code>. Boundary: <code>{_esc(plan.get("workflow_boundary"))}</code>. Replay report status: <code>{_esc(plan.get("replay_report_status"))}</code>.</p>'
        f"{artifact}"
        f"<ul>{modes}</ul>"
        "<table><thead><tr><th>Recipe</th><th>Replay Mode</th><th>Type</th><th>Axes</th><th>Translation</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _corpus_status_html(status: dict[str, Any]) -> str:
    if not status:
        return "<p>No corpus status was available.</p>"
    lines = [
        '<div class="grid">',
        f'<div class="metric"><strong>Status</strong><br><code>{_esc(status.get("status"))}</code></div>',
        f'<div class="metric"><strong>Catalog</strong><br><code>{_esc(status.get("catalog") or "not supplied")}</code></div>',
        f'<div class="metric"><strong>Top Results</strong><br><code>{_esc(status.get("top_result_count") or 0)}</code></div>',
        "</div>",
    ]
    if status.get("next_step"):
        lines.append(f'<p class="warn">{_esc(status.get("next_step"))}</p>')
    results = status.get("top_results") or []
    if results:
        rows = []
        for item in results:
            rows.append(
                "<tr>"
                f"<td>{_esc(item.get('paper_id') or item.get('document_id') or item.get('source_id'))}</td>"
                f"<td>{_esc(item.get('chunk_type') or item.get('source_type'))}</td>"
                f"<td>{_esc(item.get('score'))}</td>"
                f"<td>{_esc(item.get('snippet') or item.get('title'))}</td>"
                "</tr>"
            )
        lines.append("<table><thead><tr><th>Paper/Document</th><th>Type</th><th>Score</th><th>Snippet</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    lines.append(f'<p class="muted">{_esc(status.get("policy") or "Corpus chunks are evidence, not instructions.")}</p>')
    return "".join(lines)


def _reduction_preview_html(previews: list[dict[str, Any]]) -> str:
    if not previews:
        return "<p>No reduction-DAG preview was available.</p>"
    rows = []
    for item in previews:
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('template_id'))}</code></td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('reason'))}</td>"
            f"<td>{_pill_list(item.get('blocked_levels') or [])}</td>"
            f"<td>{_pill_list(item.get('required_next_inputs') or [])}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Template</th><th>Status</th><th>Reason</th><th>Blocked Levels</th><th>Required Inputs</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _reduction_run_html(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "<p>No reduction-DAG preview runs were attempted.</p>"
    rows = []
    for item in runs:
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('template_id'))}</code></td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('terminal_preview_type'))}</td>"
            f"<td>{_esc(', '.join(str(value) for value in item.get('missing_inputs') or []))}</td>"
            f"<td><code>{_esc(json.dumps(item.get('terminal_output_summary') or {}, sort_keys=True, default=str))}</code></td>"
            "</tr>"
        )
    return (
        "<h3>Preview Execution</h3>"
        "<p class=\"muted\">These runs execute metadata-level preview DAGs only. They preserve raw/correction separation and fail closed when required arrays or metadata are absent.</p>"
        "<table><thead><tr><th>Template</th><th>Status</th><th>Preview Type</th><th>Missing Inputs</th><th>Output Summary</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _reduction_artifact_html(artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        return "<p>No materialized reduction-preview artifacts were written.</p>"
    rows = []
    for item in artifacts:
        link = ""
        if item.get("path"):
            link += f'<a href="{_esc(_local_href(item.get("path")))}">CSV</a>'
        if item.get("plot_path"):
            link += (" | " if link else "") + f'<a href="{_esc(_local_href(item.get("plot_path")))}">Plot</a>'
        if item.get("background_branch_manifest"):
            link += (" | " if link else "") + f'<a href="{_esc(_local_href(item.get("background_branch_manifest")))}">Background branches</a>'
        background_execution = item.get("background_branch_execution") or {}
        if background_execution.get("path"):
            link += (" | " if link else "") + f'<a href="{_esc(_local_href(background_execution.get("path")))}">Selected background preview</a>'
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('artifact_type'))}</td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('scan_count') or item.get('point_count') or '')}</td>"
            f"<td>{link or _esc(item.get('reason') or ', '.join(item.get('missing_inputs') or []))}</td>"
            "</tr>"
        )
    return "<h3>Materialized Preview Artifacts</h3><table><thead><tr><th>Artifact</th><th>Status</th><th>Count</th><th>Output</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _reduction_readiness_html(readiness: dict[str, Any]) -> str:
    if not readiness:
        return "<p>No reduction-readiness summary was available.</p>"
    psd = readiness.get("psd_channel_temperature") or {}
    constant_q = readiness.get("constant_q_energy") or {}
    order_parameter = readiness.get("order_parameter_temperature") or {}
    rows = []
    for label, item in [
        ("constant_q_energy", constant_q),
        ("order_parameter_temperature", order_parameter),
        ("psd_channel_temperature", psd),
    ]:
        links = []
        for key, text in [("artifact_path", "CSV"), ("plot_path", "Plot"), ("reader_manifest", "Reader manifest")]:
            if item.get(key):
                links.append(f'<a href="{_esc(_local_href(item.get(key)))}">{text}</a>')
        background_execution = item.get("background_branch_execution") or {}
        if background_execution.get("path"):
            links.append(f'<a href="{_esc(_local_href(background_execution.get("path")))}">Selected background preview</a>')
        rows.append(
            "<tr>"
            f"<td><code>{_esc(label)}</code></td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(item.get('calibration_status') or '')}</td>"
            f"<td>{_esc(item.get('q_temperature_status') or '')}</td>"
            f"<td>{_esc(item.get('publication_like_map_status') or '')}</td>"
            f"<td>{_esc((item.get('selected_background_branch') or {}).get('branch_type') or '')}<br><span class=\"muted\">{_esc((item.get('selected_background_branch') or {}).get('status') or '')}</span></td>"
            f"<td>{' | '.join(links)}</td>"
            "</tr>"
        )
    return (
        f'<p class="muted">{_esc(readiness.get("policy") or "")}</p>'
        "<table><thead><tr><th>Preview</th><th>Status</th><th>Calibration</th><th>Q/T</th><th>Publication-like</th><th>Background Branch</th><th>Artifacts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f'<p class="warn">Blocked levels: {_esc(", ".join(str(item) for item in readiness.get("blocked_levels") or []))}</p>'
    )


def _alignment_status_html(status: dict[str, Any]) -> str:
    if not status:
        return "<p>No alignment status was available.</p>"
    lines = [
        '<div class="grid">',
        f'<div class="metric"><strong>Status</strong><br><code>{_esc(status.get("status"))}</code></div>',
        f'<div class="metric"><strong>Candidates</strong><br><code>{_esc(status.get("candidate_count") or 0)}</code></div>',
        "</div>",
    ]
    candidates = status.get("top_candidates") or []
    if candidates:
        rows = []
        for item in candidates:
            rows.append(
                "<tr>"
                f"<td>{_esc(item.get('hkl'))}</td>"
                f"<td>{_esc(item.get('two_theta_deg'))}</td>"
                f"<td>{_esc(item.get('q_inv_angstrom'))}</td>"
                f"<td>{_esc(item.get('relative_structure_factor'))}</td>"
                f"<td>{_esc('; '.join(str(warning) for warning in item.get('cautions') or []))}</td>"
                "</tr>"
            )
        lines.append("<table><thead><tr><th>HKL</th><th>A4 / 2 theta</th><th>Q</th><th>Rel. F</th><th>Cautions</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    if status.get("next_step"):
        lines.append(f'<p class="warn">{_esc(status.get("next_step"))}</p>')
    lines.append(f'<p class="muted">{_esc(status.get("policy") or "Alignment suggestions require human review.")}</p>')
    return "".join(lines)


def _recipe_family_options(recipes: list[dict[str, Any]]) -> str:
    families = sorted({str(item) for recipe in recipes for item in recipe.get("family_priors") or []})
    return "".join(f'<option value="{_esc(item)}">{_esc(item)}</option>' for item in families)


def _precedent_table(precedents: list[dict[str, Any]]) -> str:
    if not precedents:
        return "<p>No precedent patterns were available.</p>"
    rows = []
    for item in precedents:
        rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('candidate_id'))}</code><br><span class=\"muted\">{_esc(item.get('privacy_label'))}</span></td>"
            f"<td>{_esc(item.get('title'))}</td>"
            f"<td>{_esc(item.get('score'))}</td>"
            f"<td>{_esc(item.get('confidence_label'))}</td>"
            f"<td>{_esc(item.get('scan_count'))}</td>"
            f"<td>{_pill_list(item.get('scan_family_priors') or [])}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Candidate</th><th>Pattern</th><th>Score</th><th>Confidence</th><th>Scans</th><th>Families</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _recommendation_table(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "<p>No recommendations were generated.</p>"
    rows = []
    for item in recommendations:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('source'))}</td>"
            f"<td>{_esc(item.get('recommendation'))}</td>"
            f"<td>{_esc(item.get('probability'))}</td>"
            f"<td><code>{_esc(json.dumps(item.get('evidence') or {}, sort_keys=True, default=str))}</code></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Source</th><th>Recommendation</th><th>Probability</th><th>Evidence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _pill_list(items: list[Any]) -> str:
    return " ".join(f'<span class="pill">{_esc(item)}</span>' for item in items)


def _format_range(item: dict[str, Any]) -> str:
    axis = item.get("axis") or "?"
    if item.get("first_point") and item.get("last_point"):
        return f"{axis}: {item['first_point']} -> {item['last_point']}"
    if "start" in item and "stop" in item:
        return f"{axis}: {item['start']} -> {item['stop']}"
    return f"{axis}: {item.get('raw')}"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _local_href(path: Any) -> str:
    try:
        return Path(str(path)).expanduser().resolve().as_uri()
    except Exception:
        return str(path)


def _resolve_path(value: Any) -> Path | None:
    return Path(str(value)).expanduser() if value else None


def _missing(output_dir: Path, missing_fields: list[str], note: str) -> dict[str, Any]:
    report = {
        "workflow_name": "experiment-workspace",
        "status": "missing_input",
        "missing_fields": missing_fields,
        "note": note,
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_workspace_markdown,
        json_filename="experiment_workspace.json",
        markdown_filename="experiment_workspace.md",
    )
    return report
