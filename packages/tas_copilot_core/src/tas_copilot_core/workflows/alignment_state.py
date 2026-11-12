from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tas_copilot_core.counting import counting_measurement
from tas_copilot_core.instrument_calibrations import VELOCITY_SELECTOR_KIND, default_instrument_calibrations_registry_path, find_instrument_calibration
from tas_copilot_core.instrument_state import harmonic_suppression_state, nominal_incident_wavelength_from_ei_mev
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.alignment_axis_policy import apply_axis_policy, frozen_reason, load_axis_policy, policy_summary


_EPOCH_BREAK_EFFECTS = {"starts_or_changes_epoch"}
_EPOCH_BREAK_EVENT_TYPES = {"software_state_update", "possible_state_epoch_break"}
_FINE_STAGE_NAMES = {"fine_alignment", "slit_translation_optimization"}

# Bookkeeping stages that say a scan happened but not what alignment work it
# did; they outnumber the informative stages in real console logs and must not
# win the dominant-stage vote when anything specific is present.
_GENERIC_STAGE_NAMES = {"recorded_or_queued_scan", "recorded_scan_output", "unknown_alignment_action", "unknown"}

# Axes whose scans are ramp/saturation or centering profiles rather than Bragg
# peaks. Both naming generations are listed: ICE files use the abbreviations,
# NICE files spell the device out. Missing the NICE names is not a silent
# skip -- it routes a slit scan to the Gaussian branch, which returns a
# confident "peak centre" fitted to a monotonic saturation curve.
_SLIT_TRANSLATION_AXES = {
    # ICE-era
    "BkSltHght",
    "BkSltWdth",
    "FrontSlitWidth",
    "FrontSlitHeight",
    "BackSlitWidth",
    "BackSlitHeight",
    "SmplHght",
    "SmplWdth",
    "SmplElev",
    "SmplLTrn",
    "SmplUTrn",
    # NICE-era. preSampleHeight/preSampleWidth and preSampleSlitHeight/
    # preSampleSlitWidth are aliases for the same node on BT7; scan files are
    # named for the short form, data columns use the long one.
    "preSampleHeight",
    "preSampleWidth",
    "preSampleSlitHeight",
    "preSampleSlitWidth",
    "postSampleSlitHeight",
    "postSampleSlitWidth",
    "preMonoSlitHeight",
    "preMonoSlitWidth",
    "sampleElev",
    "sampleLTrans",
    "sampleUTrans",
    "sampleTableTrans",
}

# The subset of the above that are apertures rather than sample translations.
# Only an aperture has an "opening" to recommend: for a translation the same
# ramp/saturation shape means the sample is moving through the beam, and the
# number you want out of it is a centre, not an edge.
_SLIT_APERTURE_AXES = {
    name
    for name in _SLIT_TRANSLATION_AXES
    if "Slt" in name or "Slit" in name or name in {"preSampleHeight", "preSampleWidth"}
}


def alignment_state_epochs(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_state_epochs")).expanduser()
    events_path = _required_path(arguments.get("events_jsonl"), workflow_name="alignment-state-epochs")
    if isinstance(events_path, dict):
        events_path["artifacts"] = write_report_bundle(output_dir, events_path, markdown_renderer=_markdown_state_epochs, json_filename="alignment_state_epochs.json", markdown_filename="alignment_state_epochs.md")
        return events_path

    events = _read_jsonl(events_path)
    epochs = _build_epochs(events)
    state_estimates = _state_estimates(epochs)
    contradiction_checks = _contradiction_checks(events=events, planning_report_path=arguments.get("alignment_peak_report"))
    review_rows = _alignment_review_rows(epochs=epochs, events=events, source_path=events_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_template_path = output_dir / "alignment_review_template.jsonl"
    _write_jsonl(review_template_path, review_rows)
    report = {
        "workflow_name": "alignment-state-epochs",
        "status": "success",
        "events_jsonl": str(events_path),
        "summary": {
            "event_count": len(events),
            "epoch_count": len(epochs),
            "state_epoch_breaks": sum(1 for epoch in epochs if epoch.get("break_reasons")),
            "review_template_rows": len(review_rows),
            "dominant_stages": dict(Counter(str(epoch.get("dominant_stage")) for epoch in epochs).most_common()),
        },
        "state_epochs": epochs,
        "current_state_estimates": state_estimates,
        "contradiction_checks": contradiction_checks,
        "review_template": {
            "path": str(review_template_path),
            "object_types": ["alignment_state_transition", "alignment_peak_candidate", "alignment_plane_candidate"],
            "status": "unreviewed_template_written",
        },
        "interpretation": "State epochs are hypotheses about continuity of sample-instrument orientation; they are not accepted alignment states until reviewed.",
        "warnings": [
            "Sample removal/re-mounting can silently invalidate an epoch without appearing in logs.",
            "Manual rate-meter searches are weak evidence unless followed by recorded scans or fits.",
            "A3 soft-position resets can change the alignment frame; A4 hardware changes should be treated as instrument-state warnings.",
        ],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_state_epochs, json_filename="alignment_state_epochs.json", markdown_filename="alignment_state_epochs.md")
    report["artifacts"]["review_template_jsonl"] = str(review_template_path)
    return report


def alignment_state_machine(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_state_machine")).expanduser()
    events_path = _required_path(arguments.get("events_jsonl"), workflow_name="alignment-state-machine")
    if isinstance(events_path, dict):
        events_path["artifacts"] = write_report_bundle(output_dir, events_path, markdown_renderer=_markdown_state_machine, json_filename="alignment_state_machine.json", markdown_filename="alignment_state_machine.md")
        return events_path
    events = _read_jsonl(events_path)
    epochs = _build_epochs(events)
    state_estimates = _state_estimates(epochs)
    max_alignment_peaks = int(arguments.get("max_alignment_peaks") or 3)
    peak_diagnostics = _peak_fit_diagnostics(events, max_alignment_peaks=max_alignment_peaks, arguments=arguments)
    a4_checks = _a4_prediction_checks(events=events, planning_report_path=arguments.get("alignment_peak_report"), peak_diagnostics=peak_diagnostics)
    contradiction_checks = _contradiction_checks(events=events, planning_report_path=arguments.get("alignment_peak_report"), a4_checks=a4_checks)
    structure_consistency = _planning_structure_consistency(events=events, planning=_planning_report_summary(arguments.get("alignment_peak_report")))
    if structure_consistency.get("status") == "lattice_mismatch":
        # Predicted positions belong to a different cell; matches and
        # planning-based contradictions cannot be trusted for this report.
        for check in a4_checks:
            check["structure_mismatch_caveat"] = True
        for check in contradiction_checks:
            check["structure_mismatch_caveat"] = True
            if check.get("status") == "contradiction_detected":
                check["status"] = "unverifiable_planning_structure_mismatch"
                check["human_review_required"] = True
    background_inputs = _lambda_background_input_status(arguments)
    selector_calibration = _velocity_selector_calibration(arguments)
    lambda_background_scores = _lambda_background_probability_scores(
        events=events,
        peak_diagnostics=peak_diagnostics,
        planning_report_path=arguments.get("alignment_peak_report"),
        arguments=arguments,
        selector_calibration=selector_calibration,
    )
    axis_policy = load_axis_policy(arguments.get("axis_policy"))
    policy_actions: list[dict[str, Any]] = []
    if axis_policy.get("frozen"):
        for estimate in state_estimates:
            estimate["checks_blocked_by_axis_policy"] = _checks_blocked_by_policy(estimate.get("next_expected_checks") or [], axis_policy)
        policy_actions.append(_respect_axis_policy_action(axis_policy))
    report = {
        "workflow_name": "alignment-state-machine",
        "axis_policy": policy_summary(axis_policy),
        "recommended_next_actions": policy_actions,
        "status": "success",
        "events_jsonl": str(events_path),
        "summary": {
            "event_count": len(events),
            "epoch_count": len(epochs),
            "state_estimate_count": len(state_estimates),
            "contradiction_check_count": len(contradiction_checks),
            "flagged_contradictions": sum(1 for item in contradiction_checks if item.get("status") == "contradiction_detected"),
            "peak_fit_diagnostic_count": len(peak_diagnostics),
            "a4_prediction_check_count": len(a4_checks),
            "max_alignment_peaks": max_alignment_peaks,
            "peak_fit_backend": str(arguments.get("peak_fit_backend") or "internal_preview"),
        },
        "state_epochs": epochs,
        "state_estimates": state_estimates,
        "peak_fit_diagnostics": peak_diagnostics,
        "planning_structure_consistency": structure_consistency,
        "contradiction_checks": contradiction_checks,
        "a4_prediction_checks": a4_checks,
        "lambda_half_and_background_scoring_inputs": background_inputs,
        "lambda_half_and_background_scores": lambda_background_scores,
        "warnings": _state_machine_warnings(events=events, structure_consistency=structure_consistency),
        "interpretation": "The state machine estimates where alignment appears to be in the process. Estimates remain reviewable hypotheses, not accepted instrument state.",
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_state_machine, json_filename="alignment_state_machine.json", markdown_filename="alignment_state_machine.md")
    return report


def alignment_adaptive_nice_loop(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prepare a simulation-only adaptive fine-alignment loop.

    The loop follows the same safety line as the other NICE surfaces: it can rehearse
    point-by-point alignment logic against the simulator and preserve human
    injection points, but it does not approve or execute live instrument moves.
    """

    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_adaptive_nice_loop")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    execute_dry_run = bool(arguments.get("execute_dry_run") or False)
    executor = str(arguments.get("executor") or "dry_run")
    if executor != "dry_run" and not bool(arguments.get("allow_simulator_execution") or False):
        report = {
            "workflow_name": "alignment-adaptive-nice-loop",
            "status": "blocked",
            "simulation_only": True,
            "blocked_reason": "Only dry_run execution is enabled by default. Pass allow_simulator_execution for explicit local simulator tests.",
            "requested_executor": executor,
            "safety_boundary": "No live-control path is exposed by this workflow.",
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_adaptive_loop, json_filename="alignment_adaptive_nice_loop.json", markdown_filename="alignment_adaptive_nice_loop.md")
        return report

    state_path = arguments.get("state_json")
    events_path = arguments.get("events_jsonl")
    state_payload = _read_json(Path(str(state_path)).expanduser()) if state_path else {}
    events = _read_jsonl(Path(str(events_path)).expanduser()) if events_path else []
    if not state_payload and events:
        epochs = _build_epochs(events)
        state_payload = {"state_epochs": epochs, "state_estimates": _state_estimates(epochs)}
    max_iterations = int(arguments.get("max_iterations") or 3)
    replay_report = alignment_fine_replay_plan(
        {
            "state_json": state_path,
            "rocking_motor": arguments.get("rocking_motor") or "A3",
            "count_time_seconds": arguments.get("count_time_seconds") or 1.0,
            "axis_policy": arguments.get("axis_policy"),
            "output_dir": str(output_dir / "fine_replay_plan"),
        }
    )
    nice_plan_path = (replay_report.get("artifacts") or {}).get("nice_replay_training_plan_json")
    iterations = _adaptive_alignment_iterations(state_payload=state_payload, replay_report=replay_report, max_iterations=max_iterations)
    replay_executions: list[dict[str, Any]] = []
    nice_session: dict[str, Any] | None = None
    nice_session_stop: dict[str, Any] | None = None
    execute_replay = (execute_dry_run or (executor != "dry_run" and bool(arguments.get("allow_simulator_execution") or False))) and bool(nice_plan_path)
    if execute_replay and nice_plan_path:
        from tas_copilot_core.workflows.nice_bt7_replay import nice_bt7_replay, run_nice_container_session_replay, start_nice_container_session, stop_nice_container_session

        try:
            if executor == "nice_container_session":
                nice_session = start_nice_container_session(
                    session_root=output_dir,
                    container_image=str(arguments.get("container_image") or "nice-bt7-replay-smoke:latest"),
                    nice_checkout=str(arguments.get("nice_checkout") or ""),
                    docker_executable=str(arguments.get("docker_executable") or "docker"),
                    timeout_seconds=int(arguments.get("timeout_seconds") or 300),
                )
            for recipe_id in ("alignment_rocking_preview", "alignment_theta_two_theta_check"):
                replay_payload = {
                    "replay_plan": nice_plan_path,
                    "output_dir": str(output_dir / f"nice_{recipe_id}"),
                    "recipe_id": recipe_id,
                    "executor": "dry_run" if executor == "nice_container_session" else executor,
                    "mode": "point_by_point_adaptive",
                    "max_points": arguments.get("max_points") or 25,
                    "preset_time": arguments.get("count_time_seconds") or 1.0,
                    "host": arguments.get("host") or "127.0.0.1",
                    "port": arguments.get("port") or 1232,
                    "client_name": arguments.get("client_name") or "tas-copilot-alignment-adaptive",
                    "container_image": arguments.get("container_image") or "nice-bt7-replay-smoke:latest",
                    "nice_checkout": arguments.get("nice_checkout"),
                    "docker_executable": arguments.get("docker_executable") or "docker",
                    "timeout_seconds": arguments.get("timeout_seconds"),
                    "allow_unreviewed_node_hints": arguments.get("allow_unreviewed_node_hints") or False,
                }
                nice_result = nice_bt7_replay(replay_payload)
                if nice_session is not None:
                    if nice_session.get("status") == "nice_container_session_ready" and nice_result.get("status") == "success":
                        execution = run_nice_container_session_replay(
                            session=nice_session,
                            points_path=Path((nice_result.get("artifacts") or {})["points"]),
                            script_path=Path((nice_result.get("artifacts") or {})["script"]),
                            trace_path=Path(str((nice_result.get("artifacts") or {}).get("trace") or (Path(str(replay_payload["output_dir"])) / "nice_replay_trace.jsonl"))),
                            file_prefix=recipe_id,
                            count_seconds=float(arguments.get("count_time_seconds") or 1.0),
                            timeout_seconds=int(arguments.get("timeout_seconds") or 300),
                            execution=dict(nice_result.get("execution") or {}),
                        )
                        nice_result["execution"] = execution
                        nice_result.setdefault("artifacts", {})["trace"] = execution.get("trace_path")
                        nice_result["status"] = "success" if execution.get("status") == "nice_container_session_executed" else "error"
                        json_artifact = (nice_result.get("artifacts") or {}).get("json")
                        if json_artifact:
                            Path(json_artifact).write_text(json.dumps(nice_result, indent=2, sort_keys=True), encoding="utf-8")
                    elif nice_session.get("status") != "nice_container_session_ready":
                        nice_result["status"] = "error"
                        nice_result["execution"] = {**dict(nice_result.get("execution") or {}), **nice_session}
                replay_executions.append(nice_result)
                if executor == "nice_container_session" and nice_result.get("status") != "success":
                    break
        finally:
            if nice_session is not None:
                nice_session_stop = stop_nice_container_session(nice_session)
    report = {
        "workflow_name": "alignment-adaptive-nice-loop",
        "status": "success",
        "simulation_only": True,
        "planning_only": not execute_replay,
        "executor": executor if execute_replay else "plan_only",
        "state_json": str(Path(str(state_path)).expanduser()) if state_path else None,
        "events_jsonl": str(Path(str(events_path)).expanduser()) if events_path else None,
        "max_iterations": max_iterations,
        "adaptive_loop": {
            "loop_type": "fine_alignment_rehearsal",
            "control_pattern": "point_by_point_with_human_injection_points",
            "human_injection_points": ["after_rocking_preview", "after_theta_two_theta_check", "after_slit_review", "before_any_translation"],
            "stop_conditions": ["peak_not_found", "multimodal_peak_requires_review", "slit_offset_requires_reviewed_translation_axis", "human_stop"],
            "state_awareness": "Uses alignment-state epochs when supplied; otherwise builds provisional epochs from events.",
        },
        "iterations": iterations,
        "axis_policy": replay_report.get("axis_policy"),
        "steps_skipped_by_axis_policy": replay_report.get("steps_skipped_by_axis_policy") or [],
        "fine_replay_plan_report": replay_report,
        "nice_session": nice_session,
        "nice_session_stop": nice_session_stop,
        "nice_replay_executions": replay_executions,
        "safety_boundary": "Simulation/training harness only. Live instrument operation remains outside this workflow.",
        "known_limits": [
            "Counts are not supplied by a full material model in this workflow.",
            "Slit translation remains review-gated until motor direction and geometry are calibrated for the setup.",
            "Human suggestions can be injected between loop stages, not hidden inside the state estimate.",
        ],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_adaptive_loop, json_filename="alignment_adaptive_nice_loop.json", markdown_filename="alignment_adaptive_nice_loop.md")
    return report


def alignment_planning_brief(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_planning_brief")).expanduser()
    events_path = arguments.get("events_jsonl")
    state_path = arguments.get("state_json")
    planning_report_path = arguments.get("alignment_peak_report")
    events = _read_jsonl(Path(str(events_path)).expanduser()) if events_path else []
    state_payload = _read_json(Path(str(state_path)).expanduser()) if state_path else {}
    epochs = list(state_payload.get("state_epochs") or [])
    planning_link = _planning_report_summary(planning_report_path)
    accessibility_annotation = _annotate_peak_accessibility(planning_link, arguments)
    observed = _observed_alignment_summary(events=events, epochs=epochs)
    recommendations = _alignment_recommendations(observed=observed, planning_link=planning_link)
    axis_policy = load_axis_policy(arguments.get("axis_policy"))
    if axis_policy.get("frozen"):
        for estimate in list(state_payload.get("state_estimates") or []):
            estimate["checks_blocked_by_axis_policy"] = _checks_blocked_by_policy(estimate.get("next_expected_checks") or [], axis_policy)
        recommendations.append(_respect_axis_policy_action(axis_policy))
    report = {
        "workflow_name": "alignment-planning-brief",
        "status": "success" if events or epochs or planning_report_path else "missing_input",
        "inputs": {
            "events_jsonl": str(Path(str(events_path)).expanduser()) if events_path else None,
            "state_json": str(Path(str(state_path)).expanduser()) if state_path else None,
            "alignment_peak_report": str(Path(str(planning_report_path)).expanduser()) if planning_report_path else None,
        },
        "planning_evidence": planning_link,
        "kinematic_accessibility": accessibility_annotation,
        "observational_evidence": observed,
        "recommended_next_actions": recommendations,
        "axis_policy": policy_summary(axis_policy),
        "autonomy_boundary": {
            "gross_alignment": "suggest_only_or_human_guided",
            "fine_alignment": "simulator_or_supervised_autonomy_candidate_after_review",
            "live_control": "blocked_in_tascopilot_mvp",
            "human_suggestions": "dynamic suggestions can be injected into simulator/adaptive runs, but are review evidence rather than hidden state.",
        },
        "uncertainty": [
            "The system may not observe physical sample removal or manual rate-meter decisions.",
            "If planning peaks and observed scans disagree, preserve both hypotheses and ask for review rather than forcing a match.",
        ],
    }
    if report["status"] == "missing_input":
        report["missing_inputs"] = ["events_jsonl or state_json or alignment_peak_report"]
        report["try"] = "python -m tas_copilot_core.cli workflow alignment-planning-brief --events-jsonl local_runs/alignment_corpus_index/alignment_events.jsonl --state-json local_runs/alignment_state_epochs/alignment_state_epochs.json --output-dir local_runs/alignment_planning_brief"
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_planning_brief, json_filename="alignment_planning_brief.json", markdown_filename="alignment_planning_brief.md")
    return report


def alignment_fine_replay_plan(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_fine_replay_plan")).expanduser()
    state_path = arguments.get("state_json")
    brief_path = arguments.get("brief_json")
    state_payload = _read_json(Path(str(state_path)).expanduser()) if state_path else {}
    brief_payload = _read_json(Path(str(brief_path)).expanduser()) if brief_path else {}
    epochs = list(state_payload.get("state_epochs") or [])
    latest_epoch = epochs[-1] if epochs else {}
    plan = _fine_alignment_plan(latest_epoch=latest_epoch, brief=brief_payload, arguments=arguments)
    axis_policy = load_axis_policy(arguments.get("axis_policy"))
    kept, skipped = apply_axis_policy(list(plan.get("steps") or []), axis_policy)
    plan["steps"] = kept
    plan["axis_policy"] = policy_summary(axis_policy)
    plan["steps_skipped_by_axis_policy"] = skipped
    nice_plan = _nice_replay_plan_from_alignment_plan(plan, arguments=arguments)
    report = {
        "workflow_name": "alignment-fine-replay-plan",
        "status": "success",
        "simulation_only": True,
        "planning_only": True,
        "state_json": str(Path(str(state_path)).expanduser()) if state_path else None,
        "brief_json": str(Path(str(brief_path)).expanduser()) if brief_path else None,
        "replay_plan": plan,
        "nice_replay_training_plan": nice_plan,
        "axis_policy": policy_summary(axis_policy),
        "steps_skipped_by_axis_policy": skipped,
        "safety_boundary": "This artifact can be consumed by the NICE simulator/replay adapter. It is not an approved live-instrument operation.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "alignment_fine_replay_plan.json"
    replay_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    nice_replay_path = output_dir / "alignment_nice_replay_training_plan.json"
    nice_replay_path.write_text(json.dumps(nice_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_replay_plan, json_filename="alignment_fine_replay_report.json", markdown_filename="alignment_fine_replay_report.md")
    report["artifacts"]["replay_plan_json"] = str(replay_path)
    report["artifacts"]["nice_replay_training_plan_json"] = str(nice_replay_path)
    return report


def _required_path(value: Any, *, workflow_name: str) -> Path | dict[str, Any]:
    if not value:
        return {
            "workflow_name": workflow_name,
            "status": "missing_input",
            "missing_inputs": ["events_jsonl"],
            "try": "python -m tas_copilot_core.cli workflow alignment-state-epochs --events-jsonl local_runs/alignment_corpus_index/alignment_events.jsonl --output-dir local_runs/alignment_state_epochs",
        }
    path = Path(str(value)).expanduser()
    if not path.exists():
        return {
            "workflow_name": workflow_name,
            "status": "missing_input",
            "missing_inputs": ["events_jsonl"],
            "message": f"Alignment events JSONL does not exist: {path}",
        }
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_epochs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epochs: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    break_reasons: list[str] = []
    previous_date = None
    for event in events:
        event_date = _parse_event_date(event)
        # Log-less archives (pre-console-log years) have no explicit epoch
        # boundaries; a long dead time between recorded scans is the best
        # available remount/reconfiguration signal.
        gap_break = bool(previous_date and event_date and (event_date - previous_date).days >= 7)
        is_break = _is_epoch_break(event)
        if current and (is_break or gap_break):
            epochs.append(_epoch_record(len(epochs) + 1, current, break_reasons=break_reasons))
            current = []
            break_reasons = (_break_reasons(event) if is_break else []) + (["long_time_gap_possible_remount"] if gap_break else [])
        elif is_break:
            break_reasons = _break_reasons(event)
        current.append(event)
        if event_date:
            previous_date = event_date
    if current:
        epochs.append(_epoch_record(len(epochs) + 1, current, break_reasons=break_reasons))
    return epochs


def _parse_event_date(event: dict[str, Any]):
    raw = str(event.get("date") or "")[:19]
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _is_epoch_break(event: dict[str, Any]) -> bool:
    return str(event.get("event_type")) in _EPOCH_BREAK_EVENT_TYPES or str(event.get("state_epoch_effect")) in _EPOCH_BREAK_EFFECTS


def _break_reasons(event: dict[str, Any]) -> list[str]:
    reasons = []
    event_type = str(event.get("event_type") or "")
    if event_type == "software_state_update":
        reasons.append("software_lattice_or_orientation_update")
    if event_type == "possible_state_epoch_break":
        reasons.append("abort_or_stop_near_alignment_sequence")
    effect = str(event.get("state_epoch_effect") or "")
    if effect in _EPOCH_BREAK_EFFECTS and not reasons:
        reasons.append(effect)
    return reasons or ["state_epoch_boundary_event"]


def _epoch_record(index: int, events: list[dict[str, Any]], *, break_reasons: list[str]) -> dict[str, Any]:
    stages = Counter(str(event.get("alignment_stage") or "unknown") for event in events)
    intents = Counter(str(event.get("alignment_intent") or "unknown") for event in events)
    event_types = Counter(str(event.get("event_type") or "unknown") for event in events)
    devices = Counter(device for event in events for device in event.get("scanned_devices", []))
    specific_stages = Counter({stage: count for stage, count in stages.items() if stage not in _GENERIC_STAGE_NAMES})
    dominant_stage = (specific_stages or stages).most_common(1)[0][0] if stages else "unknown"
    confidence = _epoch_confidence(events, dominant_stage=dominant_stage, break_reasons=break_reasons)
    current_state = _state_from_epoch(dominant_stage=dominant_stage, events=events, confidence=confidence, break_reasons=break_reasons)
    return {
        "epoch_id": f"alignment-epoch-{index:03d}",
        "epoch_index": index,
        "start_event_id": events[0].get("event_id"),
        "end_event_id": events[-1].get("event_id"),
        "event_count": len(events),
        "event_types": dict(event_types.most_common()),
        "dominant_stage": dominant_stage,
        "dominant_intent": intents.most_common(1)[0][0] if intents else "unknown",
        "devices": dict(devices.most_common()),
        "break_reasons": break_reasons,
        "confidence": confidence,
        "current_state_estimate": current_state,
        "validity_status": "candidate_continuous_state" if confidence >= 0.55 else "needs_human_review",
        "automation_level": _epoch_automation_level(events, dominant_stage=dominant_stage),
        "representative_events": [_event_ref(event) for event in events[:5]],
        "warnings": _epoch_warnings(events, break_reasons=break_reasons),
    }


def _state_estimates(epochs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "epoch_id": epoch.get("epoch_id"),
            "epoch_index": epoch.get("epoch_index"),
            "state": (epoch.get("current_state_estimate") or {}).get("state"),
            "confidence": (epoch.get("current_state_estimate") or {}).get("confidence"),
            "evidence": (epoch.get("current_state_estimate") or {}).get("evidence", []),
            "next_expected_checks": (epoch.get("current_state_estimate") or {}).get("next_expected_checks", []),
        }
        for epoch in epochs
    ]


def _state_from_epoch(*, dominant_stage: str, events: list[dict[str, Any]], confidence: float, break_reasons: list[str]) -> dict[str, Any]:
    stages = {str(event.get("alignment_stage") or "") for event in events}
    devices = {device for event in events for device in event.get("scanned_devices", [])}
    event_types = {str(event.get("event_type") or "") for event in events}
    evidence = sorted(stages | event_types)
    if break_reasons and "software_state_update" in event_types:
        state = "alignment_frame_redefined_or_reset_suspected"
        next_checks = ["verify first reflection after software state update", "preserve previous and new orientation hypotheses"]
    elif stages & _FINE_STAGE_NAMES or devices & {"BkSltHght", "BkSltWdth", "SmplHght", "SmplWdth", "SmplElev"}:
        state = "fine_alignment_or_sample_centering"
        next_checks = ["inspect slit ramp/saturation behavior", "confirm peak center after slit or translation changes"]
    elif "second_reflection_search_or_plane_flattening" in stages or devices & {"SmplLTilt", "SmplUTilt"}:
        state = "second_reflection_or_plane_flattening"
        next_checks = ["compare tilt-scan peak to planned second reflection", "check third reflection if available"]
    elif "first_or_second_reflection_localization" in stages or devices & {"A3", "A4"}:
        state = "reflection_localization"
        next_checks = ["fit rocking/theta-two-theta center", "check lambda/2 and powder/background alternatives"]
    elif events:
        state = "alignment_activity_observed"
        next_checks = ["connect log events to scan files", "ask for missing manual intervention notes"]
    else:
        state = "no_alignment_evidence"
        next_checks = ["run alignment-corpus-index on local scan/log directory"]
    return {
        "state": state,
        "confidence": confidence,
        "evidence": evidence[:12],
        "next_expected_checks": next_checks,
        "human_review_required": True,
    }


def _epoch_confidence(events: list[dict[str, Any]], *, dominant_stage: str, break_reasons: list[str]) -> float:
    score = 0.35
    if any(event.get("source_type") == "bt7_file" for event in events):
        score += 0.25
    if any(str(event.get("event_type")) == "file_creation" for event in events):
        score += 0.1
    if any(str(event.get("event_type")) == "fit_result_hint" for event in events):
        score += 0.1
    if dominant_stage in _FINE_STAGE_NAMES:
        score += 0.1
    if any(str(event.get("event_type")) == "manual_search_move" for event in events):
        score -= 0.1
    if break_reasons:
        score -= 0.05
    return round(max(0.05, min(0.95, score)), 3)


def _epoch_automation_level(events: list[dict[str, Any]], *, dominant_stage: str) -> str:
    levels = {str(event.get("automation_level") or "") for event in events}
    if dominant_stage in _FINE_STAGE_NAMES:
        return "simulator_or_supervised_autonomy_candidate"
    if "supervised_autonomy_candidate" in levels:
        return "supervised_autonomy_candidate_after_review"
    return "suggest_only"


def _epoch_warnings(events: list[dict[str, Any]], *, break_reasons: list[str]) -> list[str]:
    warnings = [str(warning) for event in events for warning in (event.get("warnings") or [])]
    if any(str(event.get("event_type")) == "manual_search_move" for event in events):
        warnings.append("manual_search_move_present_intent_is_weakly_inferred")
    if break_reasons:
        warnings.append("epoch_started_after_state_boundary_review_continuity")
    return sorted(set(warnings))


def _event_ref(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "source_path": event.get("source_path"),
        "line_number": event.get("line_number"),
        "filename": event.get("filename"),
        "alignment_stage": event.get("alignment_stage"),
        "alignment_intent": event.get("alignment_intent"),
    }


def _alignment_review_rows(*, epochs: list[dict[str, Any]], events: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    source_hash = _path_hash(source_path)
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        rows.append(
            {
                "object_type": "alignment_state_transition",
                "object_id": str(epoch["epoch_id"]),
                "source_workflow": "alignment-state-epochs",
                "source_artifact_path": str(source_path),
                "source_artifact_hash": source_hash,
                "verdict": "unreviewed",
                "type_specific_status": None,
                "type_specific_allowed": ["accepted_continuous_epoch", "rejected_epoch_break", "needs_more_context"],
                "claim_state": "candidate",
                "suggested_status": str(epoch.get("validity_status") or "candidate_continuous_state"),
                "evidence_channels_relied_on": ["evidence_channel_state_epoch", "evidence_channel_console_log", "evidence_channel_scan_recipe"],
                "recommendation": "no_action",
                "uncertainty_class": "probability_style_triage_prior",
                "review_notes": "",
                "metadata": {"dominant_stage": epoch.get("dominant_stage"), "confidence": epoch.get("confidence"), "break_reasons": epoch.get("break_reasons")},
            }
        )
        state = (epoch.get("current_state_estimate") or {}).get("state")
        devices = set((epoch.get("devices") or {}).keys())
        if state in {"second_reflection_or_plane_flattening", "fine_alignment_or_sample_centering"} or devices & {"SmplLTilt", "SmplUTilt"}:
            rows.append(
                {
                    "object_type": "alignment_plane_candidate",
                    "object_id": f"{epoch['epoch_id']}:plane",
                    "source_workflow": "alignment-state-epochs",
                    "source_artifact_path": str(source_path),
                    "source_artifact_hash": source_hash,
                    "verdict": "unreviewed",
                    "type_specific_status": None,
                    "type_specific_allowed": ["accepted_plane_hypothesis", "rejected_plane_hypothesis", "needs_third_reflection", "needs_more_context"],
                    "claim_state": "candidate",
                    "suggested_status": "candidate_plane_hypothesis",
                    "evidence_channels_relied_on": ["evidence_channel_state_epoch", "evidence_channel_structure_factor", "evidence_channel_scan_recipe", "evidence_channel_console_log"],
                    "recommendation": "no_action",
                    "uncertainty_class": "probability_style_triage_prior",
                    "review_notes": "",
                    "metadata": {"state": state, "confidence": epoch.get("confidence"), "devices": epoch.get("devices")},
                }
            )
    for event in events:
        if str(event.get("event_type")) == "planned_scan" and str(event.get("alignment_stage")) in {"first_or_second_reflection_localization", "second_reflection_search_or_plane_flattening", "gross_reflection_search_or_two_theta_localization"}:
            rows.append(
                {
                    "object_type": "alignment_peak_candidate",
                    "object_id": str(event.get("event_id")),
                    "source_workflow": "alignment-state-epochs",
                    "source_artifact_path": str(source_path),
                    "source_artifact_hash": source_hash,
                    "verdict": "unreviewed",
                    "type_specific_status": None,
                    "type_specific_allowed": ["accepted_peak_assignment", "rejected_powder_or_background", "rejected_lambda_half_alias", "needs_more_context"],
                    "claim_state": "candidate",
                    "suggested_status": "candidate_peak_assignment",
                    "evidence_channels_relied_on": ["evidence_channel_scan_recipe", "evidence_channel_structure_factor", "evidence_channel_console_log"],
                    "recommendation": "no_action",
                    "uncertainty_class": "probability_style_triage_prior",
                    "review_notes": "",
                    "metadata": {"filename": event.get("filename"), "scanned_devices": event.get("scanned_devices"), "alignment_stage": event.get("alignment_stage")},
                }
            )
    return rows


def _contradiction_checks(*, events: list[dict[str, Any]], planning_report_path: Any, a4_checks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    planning = _planning_report_summary(planning_report_path)
    if planning.get("status") != "loaded":
        return [
            {
                "check_id": "alignment_peak_prediction_contradiction",
                "status": "insufficient_metadata",
                "reason": "No alignment peak planning report was supplied.",
            }
        ]
    predicted = {
        _normalize_hkl(candidate.get("hkl") or [candidate.get("h"), candidate.get("k"), candidate.get("l")])
        for candidate in planning.get("all_candidates") or planning.get("top_candidates") or []
    }
    predicted.discard(None)
    # Two observed-evidence channels: HKL declared in scan metadata/logs, and
    # candidate assignments resolved from the A4 window plus fitted d-spacing.
    declared: dict[str, str] = {}
    for event in events:
        hkl = _observed_hkl(event)
        if hkl:
            declared[str(event.get("event_id"))] = hkl
    resolved: dict[str, dict[str, Any]] = {}
    for check in a4_checks or []:
        if check.get("discrimination_status") in {"resolved_by_d_spacing", "resolved_by_a3_relation"} and check.get("best_candidate_hkl"):
            resolved[str(check.get("event_id"))] = check
    if not predicted or not (declared or resolved):
        return [
            {
                "check_id": "alignment_peak_prediction_contradiction",
                "status": "insufficient_metadata",
                "reason": "Planning peaks or observed HKL evidence (declared metadata or d-spacing-resolved assignments) are missing; no contradiction can be claimed.",
                "predicted_hkl_count": len(predicted),
                "observed_hkl_count": len(declared),
                "d_spacing_resolved_count": len(resolved),
            }
        ]
    checks = []
    for event_id in sorted(set(declared) | set(resolved)):
        declared_hkl = declared.get(event_id)
        assignment = resolved.get(event_id)
        assigned_hkl = assignment.get("best_candidate_hkl") if assignment else None
        entry: dict[str, Any] = {
            "check_id": f"alignment_peak_prediction_contradiction:{event_id}",
            "event_id": event_id,
            "observed_hkl": declared_hkl,
            "d_spacing_assigned_hkl": assigned_hkl,
            "predicted_hkls": sorted(predicted),
        }
        if declared_hkl and not _hkl_is_integral(declared_hkl):
            # e.g. "move hkl 0 0 6.1": a deliberate off-peak (background or
            # spurion check) position, not a reflection claim.
            entry["status"] = "intentional_offset_scan_candidate"
            entry["reason"] = "Declared HKL is non-integral; treated as a deliberate off-peak position rather than a reflection assignment."
            entry["human_review_required"] = False
        elif declared_hkl and assigned_hkl:
            consistent = declared_hkl == assigned_hkl
            entry["status"] = "consistent_declared_matches_d_resolved" if consistent else "contradiction_detected"
            entry["reason"] = (
                "Scan-declared HKL agrees with the d-spacing-resolved assignment."
                if consistent
                else "Scan-declared HKL disagrees with the reflection implied by the fitted center and incident energy."
            )
            entry["human_review_required"] = not consistent
        elif declared_hkl:
            consistent = declared_hkl in predicted
            entry["status"] = "consistent_with_planning" if consistent else "contradiction_detected"
            entry["human_review_required"] = not consistent
        else:
            entry["status"] = "candidate_assignment_from_a4_and_d_spacing"
            entry["reason"] = "No declared HKL; the assignment comes from the A4 window and fitted d-spacing and remains a reviewable candidate."
            entry["d_mismatch_fraction"] = assignment.get("best_candidate_d_mismatch_fraction") if assignment else None
            entry["human_review_required"] = True
        checks.append(entry)
    return checks


def _state_machine_warnings(*, events: list[dict[str, Any]], structure_consistency: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if structure_consistency.get("status") == "lattice_mismatch":
        warnings.append(
            "Planning structure lattice disagrees with the scan-header lattice "
            f"(max length mismatch {structure_consistency.get('max_length_mismatch_fraction')}, "
            f"max angle mismatch {structure_consistency.get('max_angle_mismatch_deg')} deg): likely wrong polymorph/setting; "
            "rerun suggest-alignment-peaks with --lattice from the headers."
        )
    if any(
        "powder" in str(((event.get("metadata") or {}).get("ExptName") or "")).lower()
        for event in events
        if event.get("source_type") == "bt7_file"
    ):
        warnings.append(
            "Experiment name suggests a powder measurement; single-crystal rocking/Bragg alignment semantics "
            "(mosaic peaks, reflection assignment) may not apply to these scans."
        )
    return warnings


def _planning_structure_consistency(*, events: list[dict[str, Any]], planning: dict[str, Any]) -> dict[str, Any]:
    """Compare the planning structure's cell with the cell in the scan headers.

    Formula-only structure lookup can select the wrong polymorph or setting
    (e.g. beta-RuCl3 instead of the honeycomb alpha phase); when the cells
    disagree, predicted peak positions do not correspond to the experimenters'
    lattice and window/contradiction checks must carry a caveat.
    """
    selected = planning.get("selected_structure_lattice")
    header_lattice = None
    for event in events:
        raw = ((event.get("metadata") or {}).get("Lattice")) if event.get("source_type") == "bt7_file" else None
        parts = re.findall(r"-?\d+(?:\.\d+)?", str(raw or ""))
        if len(parts) == 6:
            header_lattice = [float(p) for p in parts]
            break
    if not selected or not header_lattice:
        return {"status": "insufficient_metadata", "selected_structure_lattice": selected, "header_lattice": header_lattice}
    try:
        cand = [float(selected[key]) for key in ("a", "b", "c", "alpha", "beta", "gamma")]
    except (KeyError, TypeError, ValueError):
        return {"status": "insufficient_metadata", "selected_structure_lattice": selected, "header_lattice": header_lattice}
    max_length_fraction = max(abs(h - c) / h for h, c in zip(header_lattice[:3], cand[:3]) if h > 0)
    max_angle_deg = max(abs(h - c) for h, c in zip(header_lattice[3:], cand[3:]))
    mismatched = max_length_fraction > 0.03 or max_angle_deg > 2.0
    return {
        "status": "lattice_mismatch" if mismatched else "lattice_consistent",
        "selected_structure_lattice": selected,
        "header_lattice": header_lattice,
        "max_length_mismatch_fraction": round(max_length_fraction, 4),
        "max_angle_mismatch_deg": round(max_angle_deg, 3),
        "consequence": (
            "Planning peak positions are for a different cell (wrong polymorph or setting); "
            "A4 window matches and planning-based contradictions are unreliable for this report."
            if mismatched
            else None
        ),
    }


def _a4_prediction_checks(*, events: list[dict[str, Any]], planning_report_path: Any, peak_diagnostics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    planning = _planning_report_summary(planning_report_path)
    if planning.get("status") != "loaded":
        return [{"check_id": "alignment_a4_prediction", "status": "insufficient_metadata", "reason": "No alignment peak planning report was supplied."}]
    candidates = []
    for candidate in planning.get("all_candidates") or planning.get("top_candidates") or []:
        two_theta = candidate.get("two_theta_deg") or candidate.get("a4_deg") or candidate.get("A4")
        if two_theta is not None:
            candidates.append(
                {
                    "hkl": _normalize_hkl(candidate.get("hkl") or [candidate.get("h"), candidate.get("k"), candidate.get("l")]),
                    "two_theta_deg": float(two_theta),
                    "d_spacing": float(candidate["d_spacing"]) if candidate.get("d_spacing") is not None else None,
                }
            )
    fitted_a4_centers: dict[str, float] = {}
    for diagnostic in peak_diagnostics or []:
        fits = diagnostic.get("fits") or []
        if diagnostic.get("axis") == "A4" and diagnostic.get("status") == "success" and fits and fits[0].get("center") is not None:
            fitted_a4_centers[str(diagnostic.get("event_id"))] = float(fits[0]["center"])
    incident_energy_by_event = {str(event.get("event_id")): event.get("incident_energy_mev") for event in events}
    windows = []
    for event in events:
        window = _a4_window_preview(event)
        if not window and event.get("a4_static_deg") is not None:
            # Tilt/slit/A3-rocking scans do not scan A4; the reflection they
            # sit on is defined by the constant A4 column value.
            static = float(event["a4_static_deg"])
            window = {
                "event_id": event.get("event_id"),
                "source_path": event.get("source_path"),
                "a4_min_deg": round(static - 0.5, 4),
                "a4_max_deg": round(static + 0.5, 4),
                "a4_center_deg": round(static, 4),
                "static_a4": True,
            }
        if window:
            windows.append(window)
    if not candidates or not windows:
        return [
            {
                "check_id": "alignment_a4_prediction",
                "status": "insufficient_metadata",
                "reason": "Planning report lacks two_theta_deg/A4 or observed scans lack A4 ranges.",
                "candidate_count": len(candidates),
                "observed_a4_window_count": len(windows),
            }
        ]
    checks = []
    tolerance = 1.0
    residual_tolerance = 0.05
    for window in windows:
        # Candidate positions are recomputed at the event's own incident
        # energy when available: the planning report is generated at one
        # energy, but sessions switch energies and the same reflection then
        # sits at a different A4.
        event_ei = incident_energy_by_event.get(str(window["event_id"]))
        matches = []
        for candidate in candidates:
            expected = _candidate_two_theta_at_energy(candidate, event_ei)
            if expected is None:
                continue
            if window["a4_min_deg"] - tolerance <= expected <= window["a4_max_deg"] + tolerance:
                entry = dict(candidate)
                entry["expected_two_theta_deg"] = round(expected, 4)
                matches.append(entry)
        discrimination = _discriminate_matches_by_d_spacing(
            window=window,
            matches=matches,
            fitted_center_deg=fitted_a4_centers.get(str(window["event_id"])),
            incident_energy_mev=incident_energy_by_event.get(str(window["event_id"])),
            lattice=planning.get("selected_structure_lattice"),
        )
        checks.append(
            {
                "check_id": f"alignment_a4_prediction:{window['event_id']}",
                "event_id": window["event_id"],
                "a4_window_preview_deg": [window["a4_min_deg"], window["a4_max_deg"]],
                "matched_predictions": matches,
                "status": "a4_prediction_in_scan_window" if matches else "no_predicted_peak_in_observed_a4_window",
                "tolerance_deg": tolerance,
                **discrimination,
                "uncertainty": "ICE Range parsing is treated as an A4 preview window; review fit center and actual scan recipe before claiming a reflection assignment.",
            }
        )
    _a3_relation_discrimination(checks, events)
    checks.append(_a4_residual_summary(checks, residual_tolerance_deg=residual_tolerance))
    return checks


def _a4_residual_summary(checks: list[dict[str, Any]], *, residual_tolerance_deg: float) -> dict[str, Any]:
    """Compare fitted peak positions against their predicted A4 across the whole stream.

    Only fitted centres are used. A window centre is just where the scan was aimed, and a static A4
    column is where the motor was driven from the UB already in use, so neither is the crystal
    answering back; mixing them in would let the instrument agree with itself.

    A deviation shared by every reflection, with the same sign and little spread, is a cell or zero
    error rather than a misassignment -- the peaks are all where they should be relative to each
    other and collectively displaced. One reflection off on its own is the misassignment signature.
    """
    residuals = []
    for check in checks:
        expected = check.get("best_candidate_expected_two_theta_deg")
        observed = check.get("observed_center_deg")
        if expected is None or observed is None:
            continue
        residual = float(observed) - float(expected)
        check["a4_residual_deg"] = round(residual, 4)
        check["a4_residual_status"] = (
            "within_tolerance" if abs(residual) <= residual_tolerance_deg else "exceeds_tolerance"
        )
        check["a4_residual_tolerance_deg"] = residual_tolerance_deg
        if check.get("observed_center_source") == "peak_fit_center":
            residuals.append({"event_id": check.get("event_id"), "hkl": check.get("best_candidate_hkl"),
                              "residual_deg": round(residual, 4), "observed_deg": round(float(observed), 4),
                              "expected_deg": round(float(expected), 4)})

    summary = {
        "check_id": "alignment_a4_residual_summary",
        "residual_tolerance_deg": residual_tolerance_deg,
        "fitted_residual_count": len(residuals),
        "residuals": residuals,
        "human_review_required": True,
        "uncertainty": "Residuals are against the planning report's cell, which is itself an "
                       "assumption; a shared offset indicts that cell or an axis zero, not the data.",
    }
    if not residuals:
        summary["status"] = "no_fitted_centers_to_compare"
        return summary

    values = [item["residual_deg"] for item in residuals]
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    summary["mean_residual_deg"] = round(mean, 4)
    summary["residual_spread_deg"] = round(spread, 4)

    if abs(mean) <= residual_tolerance_deg and spread <= 2.0 * residual_tolerance_deg:
        summary["status"] = "predictions_match_observations"
        return summary

    same_sign = all(value > 0 for value in values) or all(value < 0 for value in values)
    consistent = len(values) >= 2 and spread <= 2.0 * residual_tolerance_deg and same_sign
    if consistent and abs(mean) > residual_tolerance_deg:
        summary["status"] = "systematic_a4_offset"
        # lambda = 2 d sin(theta) at fixed lambda gives d(a)/a = -d(theta)/tan(theta), so a shared
        # A4 offset converts straight into the fractional cell error that would produce it.
        corrections = []
        for item in residuals:
            theta = math.radians(item["observed_deg"] / 2.0)
            if abs(math.tan(theta)) > 1e-9:
                corrections.append(-math.radians(item["residual_deg"] / 2.0) / math.tan(theta))
        if corrections:
            fraction = sum(corrections) / len(corrections)
            summary["implied_lattice_correction_fraction"] = round(fraction, 6)
            summary["interpretation"] = (
                "Every fitted peak sits off its prediction by about the same amount and in the same "
                "direction, so the reflections are in the right places relative to one another and "
                "the whole set is displaced. That is the planning cell or an axis zero being wrong, "
                f"not a misassignment: scaling the cell by {1.0 + fraction:.5f} would absorb it."
            )
    else:
        summary["status"] = "inconsistent_a4_residuals"
        summary["interpretation"] = (
            "Residuals disagree with each other, so no single cell or zero error explains them. "
            "Check whether one scan latched onto a different reflection than the one assigned."
        )
    return summary


def _candidate_two_theta_at_energy(candidate: dict[str, Any], ei_mev: Any) -> float | None:
    """A4 of a planning candidate at a specific incident energy.

    Falls back to the report's two_theta_deg (its generation energy) when
    the event energy or the candidate d-spacing is unavailable."""
    d_spacing = candidate.get("d_spacing")
    try:
        ei = float(ei_mev)
    except (TypeError, ValueError):
        ei = None
    if d_spacing and ei and ei > 0:
        argument = nominal_incident_wavelength_from_ei_mev(ei) / (2.0 * float(d_spacing))
        if argument <= 1.0:
            return math.degrees(2.0 * math.asin(argument))
        return None
    value = candidate.get("two_theta_deg")
    return float(value) if value is not None else None


def _lattice_metric_tensor(lattice: dict[str, Any]) -> list[list[float]]:
    a, b, c = (float(lattice[key]) for key in ("a", "b", "c"))
    alpha, beta, gamma = (math.radians(float(lattice[key])) for key in ("alpha", "beta", "gamma"))
    return [
        [a * a, a * b * math.cos(gamma), a * c * math.cos(beta)],
        [a * b * math.cos(gamma), b * b, b * c * math.cos(alpha)],
        [a * c * math.cos(beta), b * c * math.cos(alpha), c * c],
    ]


def _lattice_holohedry_ops(lattice: dict[str, Any], *, rtol: float = 1e-6) -> list[tuple[tuple[int, ...], ...]]:
    """Signed permutation matrices that leave the lattice metric unchanged.

    d-spacing is a function of the metric tensor alone, so this is exactly the group that decides
    which reflections must be degenerate in d. A cubic cell returns the 48 operations of m-3m, a
    tetragonal one the 16 of 4/mmm, orthorhombic 8, triclinic just the 2 of -1. That is the crystal
    system dependence: in cubic there is one lattice parameter, so nothing measured at a4 can
    separate (100) from (010), while in an orthorhombic cell with a != b the same two reflections
    sit at different a4 and the separation is a real measurement.

    This is the symmetry of the *lattice*, not of the structure. Two metrically equivalent
    reflections can have different structure factors in a lower-symmetry structure, but that
    changes intensity and never d, so it does not affect the degeneracy detected here.

    Restricting to signed permutations of the axes is exact for the cubic, tetragonal,
    orthorhombic, monoclinic and triclinic metrics. It does not generate the 3-fold of a
    rhombohedral or hexagonal cell, so degeneracies peculiar to those (e.g. (100) and (-110) in
    hexagonal) are missed and simply fall through to the existing ambiguous result -- under-calling
    a degeneracy leaves a reviewable ambiguity, which is the safe direction to be wrong in.
    """
    try:
        metric = _lattice_metric_tensor(lattice)
    except (KeyError, TypeError, ValueError):
        return []
    scale = max(abs(metric[i][j]) for i in range(3) for j in range(3)) or 1.0
    ops = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            op = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for column, row in enumerate(permutation):
                op[row][column] = signs[column]
            product = [[sum(metric[i][k] * op[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            transformed = [[sum(op[k][i] * product[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            if all(abs(transformed[i][j] - metric[i][j]) <= rtol * scale for i in range(3) for j in range(3)):
                ops.append(tuple(tuple(row) for row in op))
    return ops


def _hkl_tuple(label: Any) -> tuple[float, float, float] | None:
    try:
        parts = [float(part) for part in str(label).split(",")]
    except (TypeError, ValueError):
        return None
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def _symmetry_equivalent_hkl(hkl_a: Any, hkl_b: Any, ops: list[tuple[tuple[int, ...], ...]]) -> bool:
    """True when some metric-preserving operation maps one reflection onto the other."""
    first, second = _hkl_tuple(hkl_a), _hkl_tuple(hkl_b)
    if first is None or second is None:
        return False
    for op in ops:
        mapped = tuple(sum(op[row][column] * first[column] for column in range(3)) for row in range(3))
        if all(abs(mapped[i] - second[i]) <= 1e-6 for i in range(3)):
            return True
    return False


def _conventional_representative(hkl_labels: list[str]) -> str:
    """Pick one member of a symmetry-equivalent set to serve as the convention.

    Which member is chosen carries no physical content -- that is the whole point of the set being
    degenerate -- so the only requirement is that the choice be deterministic, so two runs over the
    same data label things the same way. Preferring positive indices in h, k, l order reproduces
    what a person would call (100).
    """
    def sort_key(label: str) -> Any:
        hkl = _hkl_tuple(label) or (0.0, 0.0, 0.0)
        return ([0 if value >= 0 else 1 for value in hkl], [-abs(value) for value in hkl])

    return sorted(hkl_labels, key=sort_key)[0]


def _discriminate_matches_by_d_spacing(
    *,
    window: dict[str, Any],
    matches: list[dict[str, Any]],
    fitted_center_deg: float | None,
    incident_energy_mev: Any,
    lattice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank window-matched planning candidates by fitted d-spacing.

    A 1-degree A4 window can contain several reflections (e.g. (004) vs
    (112)); the fitted scan center plus the incident energy give an observed
    d-spacing that usually separates them. The result stays a candidate
    assignment requiring review, never an accepted reflection.
    """
    if not matches:
        return {"discrimination_status": "no_window_matches"}
    span = abs(float(window.get("a4_max_deg", 0.0)) - float(window.get("a4_min_deg", 0.0)))
    if span > 10.0:
        # Wide start/stop sweeps (e.g. "Range=A4=25.0 90.0 s" pressure-cell
        # surveys) cover many reflections by design; assigning "the"
        # reflection to such a window is meaningless.
        return {"discrimination_status": "survey_sweep_window_not_a_peak_assignment", "window_span_deg": round(span, 2)}
    center_deg = fitted_center_deg if fitted_center_deg is not None else window.get("a4_center_deg")
    if fitted_center_deg is not None:
        center_source = "peak_fit_center"
    elif window.get("static_a4"):
        center_source = "static_a4_column"
    else:
        center_source = "scan_descr_window_center"
    try:
        ei = float(incident_energy_mev)
    except (TypeError, ValueError):
        return {"discrimination_status": "no_incident_energy_metadata", "observed_center_deg": center_deg, "observed_center_source": center_source}
    if center_deg is None or ei <= 0:
        return {"discrimination_status": "no_observed_center", "observed_center_source": center_source}
    wavelength = nominal_incident_wavelength_from_ei_mev(ei)
    observed_d = wavelength / (2.0 * math.sin(math.radians(float(center_deg)) / 2.0))
    scored = []
    for candidate in matches:
        if candidate.get("d_spacing"):
            mismatch = abs(observed_d - float(candidate["d_spacing"])) / float(candidate["d_spacing"])
            candidate["d_mismatch_fraction"] = round(mismatch, 6)
            scored.append(candidate)
    if not scored:
        return {
            "discrimination_status": "candidates_lack_d_spacing",
            "observed_center_deg": center_deg,
            "observed_center_source": center_source,
            "observed_d_spacing": round(observed_d, 6),
        }
    scored.sort(key=lambda item: item["d_mismatch_fraction"])
    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    resolved = best["d_mismatch_fraction"] < 0.005 and (runner_up is None or runner_up["d_mismatch_fraction"] > 2.0 * max(best["d_mismatch_fraction"], 1e-6))

    degenerate_labels: list[str] = []
    blocking_label = None
    if not resolved and best["d_mismatch_fraction"] < 0.005 and runner_up is not None and lattice:
        ops = _lattice_holohedry_ops(lattice)
        if ops:
            # Only the candidates close enough in d to have blocked resolution matter; a distant
            # candidate that happens to be symmetry-equivalent says nothing about this tie.
            contenders = [item for item in scored
                          if item["d_mismatch_fraction"] <= 2.0 * max(best["d_mismatch_fraction"], 1e-6)]
            equivalent = [item for item in contenders
                          if _symmetry_equivalent_hkl(best.get("hkl"), item.get("hkl"), ops)]
            blocking = [item for item in contenders if item not in equivalent]
            if len(equivalent) > 1 and not blocking:
                degenerate_labels = [str(item.get("hkl")) for item in equivalent]
            elif blocking:
                blocking_label = str(blocking[0].get("hkl"))

    if degenerate_labels:
        status = "degenerate_by_lattice_symmetry"
        chosen = _conventional_representative(degenerate_labels)
    else:
        status = "resolved_by_d_spacing" if resolved else "ambiguous_d_spacing"
        chosen = best.get("hkl")

    result = {
        "discrimination_status": status,
        "observed_center_deg": center_deg,
        "observed_center_source": center_source,
        "nominal_incident_wavelength": round(wavelength, 6),
        "observed_d_spacing": round(observed_d, 6),
        "best_candidate_hkl": chosen,
        "best_candidate_d_mismatch_fraction": best["d_mismatch_fraction"],
        "best_candidate_expected_two_theta_deg": best.get("expected_two_theta_deg"),
        "runner_up_hkl": runner_up.get("hkl") if runner_up else None,
        "runner_up_d_mismatch_fraction": runner_up["d_mismatch_fraction"] if runner_up else None,
        "human_review_required": True,
    }
    if degenerate_labels:
        result["symmetry_equivalent_hkls"] = degenerate_labels
        result["assignment_is_conventional"] = True
        result["assignment_note"] = (
            "These reflections are degenerate in d under the symmetry of the supplied cell, so no "
            "a4 measurement can separate them and none is needed: which one is called "
            f"{chosen} is a choice of axes, not a result. The choice is made here so the rest of "
            "the stream can be assigned relative to it, and that relative assignment -- the a3 "
            "separation between reflections -- is the part carrying physical content."
        )
    elif blocking_label is not None:
        result["ambiguity_note"] = (
            f"Tie with {blocking_label}, which is not symmetry-equivalent to {best.get('hkl')} under "
            "the supplied cell. This is a real ambiguity and more measurement can settle it."
        )
    return result


def _observed_axis_center(event: dict[str, Any], device: str) -> float | None:
    """Scan center for any ranged device, using the verified ICE conventions
    (two-value form = center + step; trailing 's' = explicit start/stop)."""
    for item in event.get("ranges") or []:
        if str(item.get("device")).lower() != device.lower():
            continue
        raw = str(item.get("raw") or "")
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
        if not values:
            continue
        if len(values) >= 2 and re.search(r"\bs\s*$", raw.strip(), re.IGNORECASE):
            return 0.5 * (values[0] + values[1])
        return values[0]
    return None


def _reciprocal_angle_via_registry(lattice: dict[str, Any], v1: list[float], v2: list[float]) -> float | None:
    """Angle between two reciprocal-lattice vectors through the registry
    (tas.ubmatrix.calculate_reciprocal_angle, icp-lattice-calculator metric).

    Crystallographic angle math stays behind the plugin boundary; returns
    None (fail-soft) when the plugin is unavailable.
    """
    try:
        from tas_copilot_contracts import ToolCall
        from tas_copilot_core.workflows.common import default_runner

        result = default_runner().run(
            ToolCall(
                tool_namespace="tas.ubmatrix",
                tool_name="calculate_reciprocal_angle",
                arguments={"lattice": lattice, "v1": v1, "v2": v2},
                caller_workflow="alignment-state-machine-a3-relation",
            )
        )
        if result.status.value == "success":
            return float(result.output["angle_deg"])
    except Exception:
        return None
    return None


def _a3_relation_discrimination(checks: list[dict[str, Any]], events: list[dict[str, Any]], *, tolerance_deg: float = 2.0) -> None:
    """Second discriminator for d-spacing-ambiguous windows.

    For two reflections measured in the same alignment frame,
    A3_i - A3_j = (theta_i - theta_j) + (phi_i - phi_j), where |phi_i - phi_j|
    is the angle between the reciprocal-lattice vectors. Anchored on the
    same-day d-spacing-resolved assignments, each ambiguous candidate is
    tested against that relation. Results stay reviewable candidates.
    """
    info: dict[str, dict[str, Any]] = {}
    for event in events:
        a3_center = _observed_axis_center(event, "A3")
        if a3_center is None and event.get("a3_static_deg") is not None:
            a3_center = float(event["a3_static_deg"])
        info[str(event.get("event_id"))] = {
            "a3": a3_center,
            "day": str(event.get("date") or "")[:10],
            "lattice_raw": ((event.get("metadata") or {}).get("Lattice")),
        }
    anchors = []
    for check in checks:
        if check.get("discrimination_status") not in {"resolved_by_d_spacing", "degenerate_by_lattice_symmetry"}:
            continue
        meta = info.get(str(check.get("event_id")), {})
        if meta.get("a3") is None or check.get("observed_center_deg") is None:
            continue
        anchors.append(
            {
                "event_id": check.get("event_id"),
                "hkl": [float(part) for part in str(check["best_candidate_hkl"]).split(",")],
                "hkl_label": check["best_candidate_hkl"],
                "a3": meta["a3"],
                "theta": float(check["observed_center_deg"]) / 2.0,
                "day": meta.get("day"),
                "conventional": check.get("discrimination_status") == "degenerate_by_lattice_symmetry",
            }
        )
    if not anchors:
        for check in checks:
            if check.get("discrimination_status") == "ambiguous_d_spacing":
                check["a3_relation_status"] = "no_anchor_with_observed_a3"
        return
    for check in checks:
        if check.get("discrimination_status") != "degenerate_by_lattice_symmetry":
            continue
        # A degenerate check is not waiting on an anchor; it supplies one. Record which anchors it
        # is consistent with so a wrong *relative* assignment still shows up.
        check["a3_relation_status"] = "anchored_by_convention"
        check["anchor_convention"] = (
            "Labelling fixed by choice of axes, not by measurement; any member of the degenerate "
            "set is an equally correct name for this peak. Everything else in the stream is then "
            "assigned relative to it."
        )
    for check in checks:
        if check.get("discrimination_status") != "ambiguous_d_spacing":
            continue
        meta = info.get(str(check.get("event_id")), {})
        a3 = meta.get("a3")
        center = check.get("observed_center_deg")
        parts = re.findall(r"-?\d+(?:\.\d+)?", str(meta.get("lattice_raw") or ""))
        if a3 is None or center is None or len(parts) != 6:
            check["a3_relation_status"] = "missing_a3_or_lattice_metadata"
            continue
        lattice = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), (float(p) for p in parts)))
        day_anchors = [anchor for anchor in anchors if anchor["day"] == meta.get("day") and anchor["event_id"] != check.get("event_id")]
        if not day_anchors:
            check["a3_relation_status"] = "no_same_day_anchor"
            continue
        theta = float(center) / 2.0
        consistent = []
        tool_missing = False
        for candidate in check.get("matched_predictions") or []:
            hkl_label = candidate.get("hkl")
            if not hkl_label:
                continue
            hkl = [float(part) for part in str(hkl_label).split(",")]
            verdicts = []
            for anchor in day_anchors:
                predicted_azimuth = _reciprocal_angle_via_registry(lattice, hkl, anchor["hkl"])
                if predicted_azimuth is None:
                    tool_missing = True
                    continue
                observed_azimuth = abs((a3 - anchor["a3"]) - (theta - anchor["theta"]))
                verdicts.append(
                    {
                        "anchor_hkl": anchor["hkl_label"],
                        "predicted_azimuth_deg": round(predicted_azimuth, 3),
                        "observed_azimuth_deg": round(observed_azimuth, 3),
                        "mismatch_deg": round(abs(observed_azimuth - predicted_azimuth), 3),
                    }
                )
            candidate["a3_relation_checks"] = verdicts
            if verdicts and all(v["mismatch_deg"] <= tolerance_deg for v in verdicts):
                consistent.append(candidate)
        if tool_missing and not consistent:
            check["a3_relation_status"] = "angle_tool_unavailable"
            continue
        if len(consistent) == 1:
            winner = consistent[0]
            check["discrimination_status"] = "resolved_by_a3_relation"
            check["best_candidate_hkl"] = winner.get("hkl")
            check["best_candidate_d_mismatch_fraction"] = winner.get("d_mismatch_fraction")
            check["a3_relation_status"] = "resolved"
            check["human_review_required"] = True
        elif consistent:
            check["a3_relation_status"] = "still_ambiguous_multiple_a3_consistent"
        else:
            check["a3_relation_status"] = "no_candidate_matches_a3_relation"


def _a4_window_preview(event: dict[str, Any]) -> dict[str, Any] | None:
    for item in event.get("ranges") or []:
        if str(item.get("device")).lower() != "a4":
            continue
        raw = str(item.get("raw") or "")
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
        if not values:
            start_value, stop_value = item.get("start"), item.get("stop")
            if not (isinstance(start_value, (int, float)) and isinstance(stop_value, (int, float))):
                continue
            start, stop = float(start_value), float(stop_value)
            step_value = item.get("step")
            step = abs(float(step_value)) if isinstance(step_value, (int, float)) and float(step_value) else 0.0
            return {
                "event_id": event.get("event_id"),
                "source_path": event.get("source_path"),
                "a4_min_deg": round(min(start, stop), 4),
                "a4_max_deg": round(max(start, stop), 4),
                "a4_center_deg": round(0.5 * (start + stop), 4),
                "range_raw": None,
                "npts": (int(round(abs(stop - start) / step)) + 1) if step else 1,
                "range_source": "structured_start_stop",
            }
        scan_descr = str(event.get("scan_descr") or "")
        match = re.search(r"Npts=(\d+)", scan_descr, re.IGNORECASE)
        npts = int(match.group(1)) if match else 1
        if len(values) >= 2 and re.search(r"\bs\s*$", raw.strip(), re.IGNORECASE):
            # Explicit start/stop form, e.g. "Range=A4=38.0 48.0 s".
            start, stop = values[0], values[1]
            center = 0.5 * (start + stop)
        else:
            # ICE center-step form, e.g. "Range=A4=35.5374 0.2" with Npts=21:
            # the first value is the scan CENTER (FindPeak drives center +/- half span),
            # verified against recorded BT-7 sweeps (fpx840982: 33.5374 -> 37.5374).
            center = values[0]
            step = values[1] if len(values) > 1 else 0.0
            half_span = 0.5 * abs(step) * max(npts - 1, 0)
            start, stop = center - half_span, center + half_span
        return {
            "event_id": event.get("event_id"),
            "source_path": event.get("source_path"),
            "a4_min_deg": round(min(start, stop), 4),
            "a4_max_deg": round(max(start, stop), 4),
            "a4_center_deg": round(center, 4),
            "range_raw": item.get("raw"),
            "npts": npts,
        }
    return None


def _diagnostic_from_event_fit(event: dict[str, Any], *, max_alignment_peaks: int) -> dict[str, Any] | None:
    """Peak diagnostic built from a fit the event already carries.

    Used when the scan file is not readable on this machine, which is the normal case for live
    findPeak events: they name a file prefix and the data lives on the instrument and the
    archive. This is deliberately not a re-fit -- no model selection, no multi-peak search -- so
    it is labeled as supplied and still requires human review.
    """
    fit = event.get("fit") or {}
    parameters = fit.get("fitParameters") or {}
    center = parameters.get("center", parameters.get("Center"))
    devices = event.get("scanned_devices") or []
    if center is None or not devices:
        return None
    entry: dict[str, Any] = {"center": float(center)}
    for source_key, target_key in (("FWHM", "fwhm"), ("centroid", "centroid"), ("moment_fwhm", "moment_fwhm"),
                                   ("reducedChiSquared", "reduced_chi_squared"), ("max_counts", "max_counts")):
        if fit.get(source_key) is not None:
            entry[target_key] = float(fit[source_key])
    for key in ("height", "background"):
        if parameters.get(key) is not None:
            entry[key] = float(parameters[key])
    point_count = fit.get("n_points")
    return {
        "event_id": event.get("event_id"),
        "source_path": str(event.get("source_path") or ""),
        "axis": str(devices[0]),
        "point_count": int(point_count) if point_count is not None else None,
        "status": "success",
        "model_family": "instrument_findpeak_fit",
        "fit_source": "supplied_by_event",
        "peak_count": 1,
        "fits": [entry],
        "heuristic_assessment": _peak_count_assessment(1, max_alignment_peaks=max_alignment_peaks),
        "reason": "Scan file not readable here; the instrument's own findPeak fit is used. Not an "
                  "independent re-fit: no model selection and no multi-peak search were performed, so a "
                  "second peak inside the same window would not be seen.",
        "human_review_required": True,
    }


def _peak_fit_diagnostics(events: list[dict[str, Any]], *, max_alignment_peaks: int, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = []
    backend = str(arguments.get("peak_fit_backend") or "internal_preview")
    for event in events:
        if event.get("event_type") != "planned_scan":
            continue
        path_value = event.get("source_path")
        path = Path(str(path_value)).expanduser() if path_value else None
        if path is None or not path.exists():
            supplied = _diagnostic_from_event_fit(event, max_alignment_peaks=max_alignment_peaks)
            if supplied:
                diagnostics.append(supplied)
            continue
        table = _read_bt7_numeric_table(path)
        if not table:
            # Loudly, not silently: an unreadable file used to be skipped with
            # no trace, which is how a whole format generation went unnoticed.
            diagnostics.append({"event_id": event.get("event_id"), "source_path": str(path), "status": "unreadable_file", "reason": "No numeric rows could be read from the scan file."})
            continue
        axis = _fit_axis_for_event(event, table)
        intensity_key = _intensity_column(table)
        if not axis or not intensity_key:
            diagnostics.append({"event_id": event.get("event_id"), "status": "missing_input", "reason": "Could not identify scan axis or intensity column."})
            continue
        if _axis_device_name(axis) in _SLIT_TRANSLATION_AXES:
            profile = _slit_profile_preview(table, axis=axis, intensity_key=intensity_key)
            diagnostics.append(
                {
                    "event_id": event.get("event_id"),
                    "source_path": str(path),
                    "axis": axis,
                    "intensity_column": intensity_key,
                    "status": profile["status"],
                    "model_family": profile.get("selected_model_family") or "slit_profile_model_selection_preview",
                    "profile": profile,
                    "reason": "Fine-alignment slit/translation scans are analyzed as ramp/saturation or centering profiles, not Gaussian Bragg peaks.",
                    "future_upgrade": "reviewed panweb/bumps slit profile backend and paired before-after translation checks",
                    "human_review_required": True,
                }
            )
            continue
        x = [float(row[axis]) for row in table if axis in row and intensity_key in row]
        y = [float(row[intensity_key]) for row in table if axis in row and intensity_key in row]
        if len(x) < 3:
            diagnostics.append({"event_id": event.get("event_id"), "status": "insufficient_points", "point_count": len(x)})
            continue
        fits, model_selection = _peak_model_selection(
            x,
            y,
            max_alignment_peaks=max_alignment_peaks,
            backend=backend,
            panweb_path=arguments.get("panweb_path"),
            panweb_use_dream=bool(arguments.get("panweb_use_dream") or False),
            panweb_parallel=int(arguments.get("panweb_parallel") or 1),
        )
        peak_count = len(fits)
        diagnostics.append(
            {
                "event_id": event.get("event_id"),
                "source_path": str(path),
                "axis": axis,
                "intensity_column": intensity_key,
                "point_count": len(x),
                "status": "success",
                "model_family": model_selection["model_family"],
                "model_selection": model_selection,
                "future_upgrade": "voigt_and_multi_peak_bic_model_selection",
                "peak_count": peak_count,
                "fits": fits,
                "heuristic_assessment": _peak_count_assessment(peak_count, max_alignment_peaks=max_alignment_peaks),
                "human_review_required": True,
            }
        )
    return diagnostics


def _read_bt7_numeric_table(path: Path) -> list[dict[str, float]]:
    """Numeric rows from a BT-7 scan file, one dict per point.

    Delegates to the bt7grouping reader, which knows the legacy, ICE and NICE
    header generations. The local parser below is the fallback for when that
    plugin is not installed; on its own it recognises only the ICE ``#Columns``
    header, so on a NICE file (``## pt <name> ...``) it returns nothing.
    """
    rows = _read_bt7_numeric_table_via_plugin(path)
    if rows is not None:
        return rows
    return _read_bt7_numeric_table_fallback(path)


def _read_bt7_numeric_table_via_plugin(path: Path) -> list[dict[str, float]] | None:
    """None if the plugin is unavailable or cannot make sense of the file."""
    try:
        from bt7grouping.reader import read_bt7_file
    except ImportError:
        return None
    try:
        data = read_bt7_file(str(path))
    except Exception:  # noqa: BLE001 -- fall back rather than fail the report
        return None
    if not getattr(data, "data", None):
        return None
    count = min(len(values) for values in data.data.values())
    rows: list[dict[str, float]] = []
    for index in range(count):
        row: dict[str, float] = {}
        for name, values in data.data.items():
            try:
                row[name] = float(values[index])
            except (TypeError, ValueError):
                continue
        if row:
            rows.append(row)
    return rows or None


def _read_bt7_numeric_table_fallback(path: Path) -> list[dict[str, float]]:
    columns: list[str] = []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#Columns"):
                columns = stripped[len("#Columns") :].strip().split()
                continue
            # NICE writes the column line as '## pt <name> ...' with no
            # '#Columns' anywhere in the file. Recognising only the ICE form
            # meant every NICE scan parsed to zero rows.
            if stripped.startswith("##"):
                columns = stripped[2:].split()
                continue
            if stripped.startswith("#"):
                continue
            if not columns:
                continue
            parts = stripped.split()
            if len(parts) < len(columns):
                continue
            # Real BT-7 rows mix numeric and non-numeric cells (Filtran=OUT,
            # HKL strings, timestamps); keep the numeric cells per column
            # instead of dropping the whole row on the first ValueError.
            row: dict[str, float] = {}
            for index in range(len(columns)):
                try:
                    row[columns[index]] = float(parts[index])
                except ValueError:
                    continue
            if row:
                rows.append(row)
    return rows


def _axis_device_name(column: str) -> str:
    """Column name without its unit suffix: 'preSampleSlitHeight(mm)' -> the device.

    NICE writes the unit into the column header; the device names that arrive
    in ``scanned_devices`` and in _SLIT_TRANSLATION_AXES do not carry it, so
    every comparison between the two has to go through here.
    """
    return column.split("(", 1)[0].strip()


# Tried in order when the event does not say what was scanned. Both naming
# generations, ICE first only because those files are the older ones.
_PREFERRED_FIT_AXES = (
    "A3", "A4",
    "SmplLTilt", "SmplUTilt", "BkSltHght", "BkSltWdth",
    "SmplHght", "SmplWdth", "SmplElev", "SmplLTrn", "SmplUTrn",
    "a3", "a4",
    "sampleLTilt", "sampleUTilt",
    "preSampleSlitHeight", "preSampleSlitWidth",
    "postSampleSlitHeight", "postSampleSlitWidth",
    "preMonoSlitHeight", "preMonoSlitWidth",
    "sampleElev", "sampleLTrans", "sampleUTrans", "sampleTableTrans",
)


def _fit_axis_for_event(event: dict[str, Any], table: list[dict[str, float]]) -> str | None:
    """The column to fit against, returned as the literal key used by the table."""
    columns = list(table[0].keys()) if table else []
    # Map device name -> column key, so a caller asking for 'preSampleSlitHeight'
    # finds the 'preSampleSlitHeight(mm)' column that actually holds the data.
    by_device: dict[str, str] = {}
    for column in columns:
        by_device.setdefault(_axis_device_name(column), column)

    for device in event.get("scanned_devices") or []:
        name = _axis_device_name(str(device))
        if name in by_device:
            return by_device[name]
    for preferred in _PREFERRED_FIT_AXES:
        if preferred in by_device:
            return by_device[preferred]
    return None


def _intensity_column(table: list[dict[str, float]]) -> str | None:
    columns = set(table[0].keys()) if table else set()
    for name in ("Detector", "detector", "Counts", "Intensity", "I"):
        if name in columns:
            return name
    return None


def _gaussian_peak_preview(x: list[float], y: list[float], *, max_alignment_peaks: int) -> list[dict[str, Any]]:
    if not x or not y:
        return []
    baseline = min(y)
    amplitude_scale = max(y) - baseline
    if amplitude_scale <= 0:
        return []
    threshold = baseline + max(amplitude_scale * 0.2, 1e-9)
    maxima = []
    for index in range(len(y)):
        left = y[index - 1] if index else float("-inf")
        right = y[index + 1] if index < len(y) - 1 else float("-inf")
        if y[index] >= left and y[index] >= right and y[index] >= threshold:
            maxima.append(index)
    maxima = sorted(maxima, key=lambda idx: y[idx], reverse=True)
    selected = sorted(maxima[: max(max_alignment_peaks + 2, max_alignment_peaks)])
    fits = []
    used: set[int] = set()
    for idx in selected:
        if idx in used:
            continue
        cluster = _peak_cluster(y, idx, threshold=baseline + amplitude_scale * 0.1)
        used.update(cluster)
        weights = [max(y[item] - baseline, 0.0) for item in cluster]
        total_weight = sum(weights)
        if total_weight <= 0:
            center = x[idx]
            sigma = None
        else:
            center = sum(x[item] * weights[pos] for pos, item in enumerate(cluster)) / total_weight
            variance = sum(((x[item] - center) ** 2) * weights[pos] for pos, item in enumerate(cluster)) / total_weight
            sigma = variance**0.5
        fits.append(
            {
                "center": round(center, 6),
                "amplitude": round(y[idx] - baseline, 6),
                "sigma_preview": round(sigma, 6) if sigma is not None else None,
                "fwhm_preview": round(2.35482 * sigma, 6) if sigma is not None else None,
                "method": "gaussian_moment_preview",
                "peak_index": idx,
            }
        )
    return sorted(fits, key=lambda item: item["amplitude"], reverse=True)


def _peak_model_selection(
    x: list[float],
    y: list[float],
    *,
    max_alignment_peaks: int,
    backend: str,
    panweb_path: Any,
    panweb_use_dream: bool,
    panweb_parallel: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if backend in {"panweb", "panweb_if_available"}:
        panweb = _panweb_auto_fit(x, y, max_alignment_peaks=max_alignment_peaks, panweb_path=panweb_path, use_dream=panweb_use_dream, parallel=panweb_parallel)
        if panweb.get("status") == "success":
            return panweb["fits"], panweb["model_selection"]
        if backend == "panweb":
            return [], panweb["model_selection"]
    fits = _gaussian_peak_preview(x, y, max_alignment_peaks=max_alignment_peaks)
    return fits, _preview_bic_model_selection(x, y, fits=fits, max_alignment_peaks=max_alignment_peaks)


def _panweb_auto_fit(
    x: list[float],
    y: list[float],
    *,
    max_alignment_peaks: int,
    panweb_path: Any,
    use_dream: bool,
    parallel: int,
) -> dict[str, Any]:
    inserted = False
    path_value = panweb_path or os.environ.get("PANWEB_PATH")
    src = Path(str(path_value)).expanduser() / "src" if path_value else None
    if src is not None and src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        inserted = True
    try:
        import numpy as np  # type: ignore
        from auto_peaks import auto_find_and_fit  # type: ignore

        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        errors = counting_measurement(y_arr).errors
        result = auto_find_and_fit(
            x_arr,
            y_arr,
            errors=errors,
            max_peaks=max_alignment_peaks,
            use_dream=use_dream,
            parallel=parallel,
            mcmc_burn=50,
            mcmc_steps=150,
        )
        params = result.parameters or {}
        fits = []
        for idx in range(int(result.n_peaks or 0)):
            center = (params.get(f"p{idx}_center") or {}).get("value")
            sigma = (params.get(f"p{idx}_sigma") or {}).get("value")
            amp = (params.get(f"p{idx}_amplitude") or {}).get("value")
            if center is None or sigma is None or amp is None:
                continue
            fits.append(
                {
                    "center": round(float(center), 6),
                    "amplitude": round(float(amp), 6),
                    "sigma_preview": round(float(sigma), 6),
                    "fwhm_preview": round(float(sigma) * 2.35482, 6),
                    "method": "panweb_auto_find_and_fit",
                    "peak_index": idx,
                }
            )
        return {
            "status": "success" if result.success else "fit_failed",
            "fits": sorted(fits, key=lambda item: item["amplitude"], reverse=True),
            "model_selection": {
                "model_family": "panweb_bumps_bic_gaussian",
                "backend": "panweb.auto_peaks.auto_find_and_fit",
                "status": "success" if result.success else "fit_failed",
                "n_peaks": int(result.n_peaks or 0),
                "significant_peaks": int(result.significant_peaks or 0),
                "bic": float(result.bic) if result.bic is not None else None,
                "aic": float(result.aic) if result.aic is not None else None,
                "all_bic_values": [(int(n), float(v)) for n, v in (result.all_bic_values or [])],
                "message": str(result.message),
                "use_dream": use_dream,
                "parallel": parallel,
            },
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "fits": [],
            "model_selection": {
                "model_family": "panweb_bumps_bic_gaussian",
                "backend": "panweb.auto_peaks.auto_find_and_fit",
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "fallback": "internal_bic_preview",
            },
        }
    finally:
        if inserted:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass


def _preview_bic_model_selection(x: list[float], y: list[float], *, fits: list[dict[str, Any]], max_alignment_peaks: int) -> dict[str, Any]:
    # This is a light panweb-style preview: candidate peak count is selected by
    # BIC over increasingly many detected Gaussian moment peaks. It is not a
    # substitute for the panweb bumps/DREAM/LM backend.
    n = len(y)
    baseline = min(y) if y else 0.0
    bic_values = []
    best_count = 0
    best_bic = float("inf")
    for count in range(1, min(max_alignment_peaks + 1, len(fits)) + 1):
        model = [baseline for _ in y]
        for fit in fits[:count]:
            center = float(fit["center"])
            sigma = float(fit.get("sigma_preview") or _default_sigma(x))
            amp = float(fit.get("amplitude") or 0.0)
            sigma = max(sigma, _default_sigma(x))
            for index, xpos in enumerate(x):
                model[index] += amp * math.exp(-0.5 * ((xpos - center) / sigma) ** 2)
        rss = sum((obs - pred) ** 2 for obs, pred in zip(y, model))
        k = 1 + 3 * count
        bic = n * math.log(max(rss / max(n, 1), 1e-12)) + k * math.log(max(n, 2))
        bic_values.append((count, round(bic, 6)))
        if bic < best_bic:
            best_bic = bic
            best_count = count
    return {
        "model_family": "internal_bic_gaussian_preview",
        "backend": "internal_moment_gaussian_bic_preview",
        "status": "success" if bic_values else "no_peak_candidates",
        "n_peaks": best_count,
        "significant_peaks": min(best_count, len(fits)),
        "bic": round(best_bic, 6) if bic_values else None,
        "all_bic_values": bic_values,
        "warning": "Preview only; use panweb/bumps backend for production model selection.",
    }


def _default_sigma(x: list[float]) -> float:
    if len(x) < 2:
        return 1.0
    span = max(x) - min(x)
    return max(span / max(len(x), 1), 1e-9)


def _peak_cluster(y: list[float], idx: int, *, threshold: float) -> list[int]:
    start = idx
    while start > 0 and y[start - 1] >= threshold:
        start -= 1
    stop = idx
    while stop < len(y) - 1 and y[stop + 1] >= threshold:
        stop += 1
    return list(range(start, stop + 1))


def _peak_count_assessment(peak_count: int, *, max_alignment_peaks: int) -> dict[str, Any]:
    if peak_count == 0:
        return {"status": "peak_not_found", "recommendation": "repeat_or_expand_search"}
    if peak_count == 1:
        return {"status": "single_peak_candidate", "recommendation": "fit_center_then_check_against_planning_peak"}
    if peak_count <= max_alignment_peaks:
        return {"status": "multiple_peak_candidate", "recommendation": "review mosaic/twin/background alternatives before choosing alignment peak"}
    return {
        "status": "too_many_peaks_bad_crystal_or_background_heuristic",
        "recommendation": "do_not_auto_align; require human review and consider sample quality/background/lambda-half checks",
        "max_alignment_peaks": max_alignment_peaks,
    }


def _slit_profile_preview(table: list[dict[str, float]], *, axis: str, intensity_key: str) -> dict[str, Any]:
    points = sorted((float(row[axis]), float(row[intensity_key])) for row in table if axis in row and intensity_key in row)
    if len(points) < 3:
        return {"status": "insufficient_points", "point_count": len(points)}
    x = [item[0] for item in points]
    y = [item[1] for item in points]
    ymax = max(y)
    ymin = min(y)
    if ymax <= ymin:
        return {"status": "flat_or_no_signal", "point_count": len(points)}
    ten_percent = ymin + 0.1 * (ymax - ymin)
    ninety_percent = ymin + 0.9 * (ymax - ymin)
    first_signal = next((xpos for xpos, intensity in points if intensity >= ten_percent), None)
    saturation_start = next((xpos for xpos, intensity in points if intensity >= ninety_percent), None)
    late_uptick = first_signal is not None and first_signal > (min(x) + 0.15 * (max(x) - min(x)))
    plateau = [xpos for xpos, intensity in points if intensity >= ninety_percent]
    center_estimate = (min(plateau) + max(plateau)) / 2.0 if plateau else saturation_start
    model_candidates = _slit_model_candidates(x, y)
    selected_model = min(model_candidates, key=lambda item: item["bic"]) if model_candidates else {}
    onset_offset = (first_signal - min(x)) if first_signal is not None else None
    translation_recommendation = _slit_translation_recommendation(
        axis=axis,
        x=x,
        first_signal=first_signal,
        saturation_start=saturation_start,
        late_uptick=late_uptick,
        selected_model=selected_model,
    )
    opening_recommendation = (
        _slit_opening_recommendation(x, y)
        if _axis_device_name(axis) in _SLIT_APERTURE_AXES
        else {"status": "not_an_aperture_axis", "axis": axis}
    )
    return {
        "status": "late_uptick_possible_decentered_sample" if late_uptick else "ramp_saturation_candidate",
        "point_count": len(points),
        "first_signal_axis_value": round(first_signal, 6) if first_signal is not None else None,
        "saturation_start_axis_value": round(saturation_start, 6) if saturation_start is not None else None,
        "plateau_center_estimate": round(center_estimate, 6) if center_estimate is not None else None,
        "opening_recommendation": opening_recommendation,
        "onset_offset_proxy_axis_units": round(onset_offset, 6) if onset_offset is not None else None,
        "offset_interpretation": "Onset delay is a centering proxy only; direction and magnitude require paired slit/translation scans and reviewed motor geometry.",
        "translation_recommendation": translation_recommendation,
        "late_uptick": late_uptick,
        "selected_model_family": selected_model.get("model_family"),
        "model_selection": {
            "status": "success" if model_candidates else "unavailable",
            "selected_model_family": selected_model.get("model_family"),
            "candidates": model_candidates,
            "policy": "Compare piecewise ramp/saturation and smooth S-curve/tanh previews; do not use this as an autonomous translation command.",
        },
        "interpretation": "Late first signal in a slit scan is weak evidence that the sample may not be centered in the beam; review geometry and translation availability before acting.",
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _moving_median3(values: list[float]) -> list[float]:
    """Three-point moving median; the two endpoints pass through unchanged.

    Counting statistics leave a few hundred counts of scatter on the plateau of
    a slit scan. That is enough for one high point to set the reference level
    and drag the recommended opening a step or two wider than it needs to be,
    so the reference and the threshold test are both taken on smoothed counts.
    """
    if len(values) < 3:
        return list(values)
    smoothed = list(values)
    for index in range(1, len(values) - 1):
        smoothed[index] = sorted(values[index - 1 : index + 2])[1]
    return smoothed


def _slit_opening_recommendation(
    x: list[float],
    y: list[float],
    *,
    target_fraction: float = 0.99,
    n_sigma: float = 1.0,
) -> dict[str, Any]:
    """Narrowest opening that still passes ``target_fraction`` of the plateau flux.

    This is the number a slit scan is taken to get, and neither of the two the
    profile already reports will serve. The 90%-of-max saturation knee is a
    ramp landmark, not a setting -- using it costs most of a tenth of the beam.
    The plateau centre is a function of where the scan happened to stop: widen
    the scan range and it moves, though the beam has not.

    Narrowest, not widest, because an upstream slit opened past the beam only
    admits background. The recommendation is therefore the first opening from
    which the count rate no longer improves, within counting error.
    """
    if len(x) < 5:
        return {"status": "insufficient_points", "point_count": len(x)}
    smoothed = _moving_median3(y)

    # Median of the widest quarter of the scan, not the largest point seen. The
    # maximum is an extreme-value statistic: it grows with the number of plateau
    # points, so on a long scan it settles on an upward fluctuation and nothing
    # afterwards can match it.
    tail_start = len(smoothed) - max(3, len(smoothed) // 4)
    reference = _median(smoothed[tail_start:])
    if reference <= min(smoothed):
        return {"status": "flat_or_no_signal", "point_count": len(x)}

    # A 1% target is meaningless on a plateau of a few hundred counts, where one
    # sigma is already 4%: every point wanders below it and the scan looks as if
    # it never saturated. Detector values are raw counts, so widen the tolerance
    # to the counting error whenever that is the larger of the two.
    counting_tolerance = n_sigma * math.sqrt(max(reference, 1.0))
    target_tolerance = (1.0 - target_fraction) * reference
    tolerance = max(target_tolerance, counting_tolerance)
    tolerance_basis = "counting_statistics" if counting_tolerance > target_tolerance else "target_fraction"
    threshold = reference - tolerance

    # Two conditions, because either alone is wrong. The point itself must be at
    # plateau, or the recommendation is a slit that is still losing flux. And
    # everything wider must average to plateau too, which is what keeps a lucky
    # high point partway up the ramp from being chosen -- while, unlike
    # requiring every single wider point to clear the bar, still tolerating the
    # occasional low one that a noisy plateau is guaranteed to contain.
    chosen = None
    for index in range(len(x)):
        if smoothed[index] >= threshold and _median(smoothed[index:]) >= threshold:
            chosen = index
            break
    if chosen is None:
        return {
            "status": "no_sustained_plateau",
            "point_count": len(x),
            "plateau_reference_intensity": round(reference, 3),
            "effective_tolerance_counts": round(tolerance, 3),
            "tolerance_basis": tolerance_basis,
            "reason": "No opening in the scanned range reaches the plateau level and holds it out to the widest point.",
        }

    # Two points past the recommendation is the minimum that distinguishes "the
    # curve flattened here" from "the scan stopped here".
    points_beyond = len(x) - 1 - chosen
    margin = smoothed[chosen] - threshold
    # Inside one sigma of the threshold the next narrower step is statistically
    # the same answer, and saying so is more useful than a false precision.
    ambiguous = margin < counting_tolerance and chosen > 0
    if points_beyond < 2:
        # Nothing here distinguishes a curve that flattened from one that was
        # still climbing when the scan ran out, so this is a flag, not a value
        # to dial in.
        status = "scan_range_may_be_short"
    elif ambiguous:
        status = "recommended_within_one_step"
    else:
        status = "recommended"
    narrower = chosen - 1
    return {
        "status": status,
        "recommended_axis_value": round(x[chosen], 6),
        "target_fraction": target_fraction,
        "plateau_reference_intensity": round(reference, 3),
        "effective_tolerance_counts": round(tolerance, 3),
        "tolerance_basis": tolerance_basis,
        "fraction_at_recommendation": round(smoothed[chosen] / reference, 6),
        # How close the call was: this many counts of margin over the threshold.
        # Small margin means the next narrower point is a defensible choice too.
        "margin_counts": round(margin, 3),
        "next_narrower_axis_value": round(x[narrower], 6) if narrower >= 0 else None,
        "next_narrower_fraction": round(smoothed[narrower] / reference, 6) if narrower >= 0 else None,
        "points_beyond_recommendation": points_beyond,
        "reason": (
            "The level was only reached at the edge of the scanned range; widen the scan before trusting this value."
            if points_beyond < 2
            else "The next narrower step is within counting error of this one; either is defensible."
            if ambiguous
            else "The next narrower step loses more than the tolerance allows."
        ),
        "smoothing": "three_point_moving_median",
        "basis": "Narrowest opening whose smoothed count rate reaches the plateau reference within tolerance, and from which the rest of the scan holds that level.",
        "human_review_required": True,
    }


def _slit_translation_recommendation(
    *,
    axis: str,
    x: list[float],
    first_signal: float | None,
    saturation_start: float | None,
    late_uptick: bool,
    selected_model: dict[str, Any],
) -> dict[str, Any]:
    span = max(x) - min(x) if x else 0.0
    if span <= 0 or first_signal is None:
        return {
            "status": "insufficient_evidence",
            "confidence_label": "low",
            "recommended_action": "repeat slit scan or provide a paired translation scan",
            "calibration_status": "missing_scan_span_or_signal_onset",
        }
    offset = first_signal - min(x)
    offset_fraction = offset / span if span > 0 else 0.0
    saturation_fraction = ((saturation_start - min(x)) / span) if saturation_start is not None and span > 0 else None
    model_onset = selected_model.get("onset_axis_value") if selected_model.get("model_family") == "piecewise_linear_then_saturation" else None
    model_agrees = model_onset is not None and abs(float(model_onset) - first_signal) <= max(0.05 * span, 1e-9)
    if late_uptick and offset_fraction >= 0.25:
        status = "review_required_possible_decentered_sample"
        action = "run paired translation scan on the relevant sample translation axis before commanding an offset"
        confidence = "medium" if model_agrees else "low"
    elif late_uptick:
        status = "weak_possible_decentering"
        action = "review scan shape and repeat with finer points before deriving a translation"
        confidence = "low"
    else:
        status = "likely_centered_or_opening_ok"
        action = "keep as evidence; no translation recommendation from this slit scan alone"
        confidence = "low_to_medium" if saturation_fraction is not None else "low"
    return {
        "status": status,
        "confidence_label": confidence,
        "axis": axis,
        "onset_offset_axis_units": round(offset, 6),
        "offset_fraction_of_scan": round(offset_fraction, 6),
        "saturation_fraction_of_scan": round(saturation_fraction, 6) if saturation_fraction is not None else None,
        "recommended_action": action,
        "calibration_status": "heuristic_pending_labeled_real_sessions",
        "missing_for_calibrated_translation": [
            "reviewed mapping from slit axis to sample translation axis",
            "translation motor direction for the mounted sample",
            "paired before/after slit scans or translation scans",
            "sample-environment constraint on whether translation is allowed",
        ],
        "human_review_required": True,
    }


def _slit_model_candidates(x: list[float], y: list[float]) -> list[dict[str, Any]]:
    if len(x) < 3 or max(y) <= min(y):
        return []
    return [
        _piecewise_slit_model(x, y),
        _smooth_slit_model(x, y),
    ]


def _piecewise_slit_model(x: list[float], y: list[float]) -> dict[str, Any]:
    ymin = min(y)
    ymax = max(y)
    best: dict[str, Any] | None = None
    n = len(x)
    for left_index in range(0, max(1, n - 2)):
        for right_index in range(left_index + 1, n):
            left = x[left_index]
            right = x[right_index]
            if right <= left:
                continue
            fitted = []
            for xpos in x:
                if xpos <= left:
                    fitted.append(ymin)
                elif xpos >= right:
                    fitted.append(ymax)
                else:
                    fitted.append(ymin + (ymax - ymin) * (xpos - left) / (right - left))
            rss = _rss(y, fitted)
            bic = _bic(rss, n=n, k=4)
            candidate = {
                "model_family": "piecewise_linear_then_saturation",
                "bic": round(bic, 6),
                "rss": round(rss, 6),
                "onset_axis_value": round(left, 6),
                "saturation_axis_value": round(right, 6),
                "parameters": {"baseline": round(ymin, 6), "plateau": round(ymax, 6)},
            }
            if best is None or candidate["bic"] < best["bic"]:
                best = candidate
    return best or {"model_family": "piecewise_linear_then_saturation", "bic": float("inf"), "rss": float("inf"), "status": "unavailable"}


def _smooth_slit_model(x: list[float], y: list[float]) -> dict[str, Any]:
    ymin = min(y)
    ymax = max(y)
    span = max(x) - min(x)
    amp = ymax - ymin
    if span <= 0 or amp <= 0:
        return {"model_family": "tanh_s_curve", "bic": float("inf"), "rss": float("inf"), "status": "unavailable"}
    mid_level = ymin + 0.5 * amp
    mid_index = min(range(len(x)), key=lambda index: abs(y[index] - mid_level))
    x_mid = x[mid_index]
    ten = ymin + 0.1 * amp
    ninety = ymin + 0.9 * amp
    x10 = next((xpos for xpos, intensity in zip(x, y) if intensity >= ten), min(x))
    x90 = next((xpos for xpos, intensity in zip(x, y) if intensity >= ninety), max(x))
    width = max(abs(x90 - x10), span / max(len(x), 1), 1e-9)
    scale = max(width / 2.944, 1e-9)  # tanh moves from ~0.1 to ~0.9 over 2.944 scale units.
    fitted = [ymin + 0.5 * amp * (1.0 + math.tanh((xpos - x_mid) / scale)) for xpos in x]
    rss = _rss(y, fitted)
    return {
        "model_family": "tanh_s_curve",
        "bic": round(_bic(rss, n=len(x), k=4), 6),
        "rss": round(rss, 6),
        "midpoint_axis_value": round(x_mid, 6),
        "scale_axis_units": round(scale, 6),
        "parameters": {"baseline": round(ymin, 6), "plateau": round(ymax, 6)},
    }


def _rss(observed: list[float], fitted: list[float]) -> float:
    return sum((obs - pred) ** 2 for obs, pred in zip(observed, fitted))


def _bic(rss: float, *, n: int, k: int) -> float:
    return n * math.log(max(rss / max(n, 1), 1e-12)) + k * math.log(max(n, 2))


def _lambda_background_input_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "input_contract_documented",
        "needed_for_lambda_half_scoring": [
            "in-pile PG/filter state from BT-7 headers or logs",
            "velocity-selector state from later BT-7 headers/logs when available",
            "manual post-sample filter note if present",
            "incident/final energy and analyzer state",
            "candidate alignment peak A4/two-theta or fitted center",
            "sample lattice/scattering plane from later accepted scans",
        ],
        "needed_for_powder_background_scoring": [
            "background material pack, usually including Al and optionally Cu/Si/PG/sample-environment materials",
            "powder d-spacings or structure-factor peak table for each material",
            "observed A4/two-theta windows or fitted centers",
            "sample material lattice and intended HKL hypotheses",
        ],
        "current_inputs": {
            "background_material_config": arguments.get("background_material_config"),
            "background_materials": arguments.get("background_material") or [],
            "filter_state_source": "typed_harmonic_filter_translation_from_scan_columns_if_present_else_missing",
            "velocity_selector_source": "typed_velocity_selector_speed_from_scan_columns_if_present_else_missing",
            "instrument_calibrations_registry": str(arguments.get("instrument_calibrations") or default_instrument_calibrations_registry_path()),
            "post_sample_filter_source": "manual_note_required",
        },
        "policy": "Score lambda/2 and powder/background as probability-style warnings. PG filters and velocity selectors lower lambda/2 probability but do not rule it out, especially for very strong scatterers or unknown post-sample filter state.",
    }


def _lambda_background_probability_scores(
    *,
    events: list[dict[str, Any]],
    peak_diagnostics: list[dict[str, Any]],
    planning_report_path: Any,
    arguments: dict[str, Any],
    selector_calibration: Any = None,
) -> dict[str, Any]:
    observed = _observed_peak_centers(events=events, peak_diagnostics=peak_diagnostics)
    peak_table = _read_background_peak_table(arguments.get("background_peak_table"))
    materials = list(arguments.get("background_material") or [])
    if not materials and arguments.get("background_material_config"):
        materials = _materials_from_config(arguments.get("background_material_config"))
    if not materials:
        materials = ["Al"]
    filter_states = sorted(set(_event_metadata_values(events, keys=("filter_state", "FilTran", "filtran")) + _center_metadata_values(observed, keys=("filter_state", "FilTran", "filtran"))))
    velocity_states = sorted(set(_event_metadata_values(events, keys=("velocity_selector_state", "VelSel", "velsel", "velocity_selector")) + _center_metadata_values(observed, keys=("velocity_selector_state", "VelSel", "velsel", "velocity_selector"))))
    typed_filter_states = sorted(
        {str(item.get("harmonic_filter_translation")).upper() for item in list(events) + observed if item.get("harmonic_filter_translation") not in {None, "", "unknown"}}
    )
    typed_velocity_speeds = sorted(
        {_format_selector_speed(item.get("velocity_selector_speed")) for item in list(events) + observed if item.get("velocity_selector_speed") is not None}
    )
    filter_states = sorted(set(filter_states) | set(typed_filter_states))
    velocity_states = sorted(set(velocity_states) | {f"speed={speed}" for speed in typed_velocity_speeds})
    lambda_scores = [
        _lambda_half_score(
            center=item,
            filter_states=filter_states,
            velocity_states=velocity_states,
            planning_report_path=planning_report_path,
            selector_calibration=selector_calibration,
        )
        for item in observed
    ]
    powder_scores = [
        _powder_background_score(center=item, peak_table=peak_table, materials=materials, tolerance_deg=float(arguments.get("background_tolerance_deg") or 0.6))
        for item in observed
    ]
    return {
        "status": "scored" if observed else "insufficient_observed_peak_centers",
        "observed_center_count": len(observed),
        "lambda_half_scores": lambda_scores,
        "powder_background_scores": powder_scores,
        "background_peak_table_status": "loaded" if peak_table else "not_supplied",
        "materials_considered": materials,
        "filter_state_evidence": filter_states,
        "velocity_selector_evidence": velocity_states,
        "velocity_selector_calibration": _selector_calibration_summary(selector_calibration),
        "policy": "Scores are probability-style triage evidence. Filters and velocity selectors reduce lambda/2 likelihood but do not make it impossible.",
    }


def _velocity_selector_calibration(arguments: dict[str, Any]) -> Any:
    path_value = arguments.get("instrument_calibrations")
    path = Path(str(path_value)).expanduser() if path_value else default_instrument_calibrations_registry_path()
    if not path.exists():
        return None
    return find_instrument_calibration(kind=VELOCITY_SELECTOR_KIND, path=path)


def _selector_calibration_summary(selector_calibration: Any) -> dict[str, Any]:
    if selector_calibration is None:
        return {"status": "not_available", "kind": VELOCITY_SELECTOR_KIND}
    return {
        "status": selector_calibration.status,
        "id": selector_calibration.id,
        "kind": selector_calibration.kind,
        "instrument": selector_calibration.instrument,
        "payload": dict(selector_calibration.payload or {}),
    }


def _format_selector_speed(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _observed_peak_centers(*, events: list[dict[str, Any]], peak_diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    by_event = {str(event.get("event_id")): event for event in events}
    for diagnostic in peak_diagnostics:
        if diagnostic.get("status") != "success":
            continue
        axis = str(diagnostic.get("axis") or "")
        if axis not in {"A4", "A3"}:
            continue
        fits = list(diagnostic.get("fits") or [])
        if not fits:
            continue
        center = fits[0].get("center")
        if center is None:
            continue
        event = by_event.get(str(diagnostic.get("event_id")), {})
        file_metadata = _bt7_alignment_metadata(Path(str(diagnostic.get("source_path")))) if diagnostic.get("source_path") else {}
        centers.append(
            {
                "event_id": diagnostic.get("event_id"),
                "source_path": diagnostic.get("source_path"),
                "axis": axis,
                "center_deg": round(float(center), 6),
                "metadata": {**file_metadata, **(event.get("metadata") or {})},
                **_typed_instrument_state(event, file_metadata),
            }
        )
    for event in events:
        window = _a4_window_preview(event)
        if window:
            file_metadata = _bt7_alignment_metadata(Path(str(event.get("source_path")))) if event.get("source_path") else {}
            center = float(window["a4_center_deg"]) if window.get("a4_center_deg") is not None else 0.5 * (float(window["a4_min_deg"]) + float(window["a4_max_deg"]))
            centers.append(
                {
                    "event_id": event.get("event_id"),
                    "source_path": event.get("source_path"),
                    "axis": "A4",
                    "center_deg": round(center, 6),
                    "metadata": {**file_metadata, **(event.get("metadata") or {})},
                    "source": "scan_window_preview",
                    **_typed_instrument_state(event, file_metadata),
                }
            )
    dedup: dict[tuple[str, str, float], dict[str, Any]] = {}
    for item in centers:
        dedup[(str(item.get("event_id")), str(item.get("axis")), round(float(item.get("center_deg") or 0.0), 3))] = item
    return list(dedup.values())


def _typed_instrument_state(event: dict[str, Any], file_metadata: dict[str, Any]) -> dict[str, Any]:
    """Typed harmonic-filter/velocity-selector state for a scored center.

    Prefers the typed event fields written by the alignment corpus index
    (harmonic_filter_translation, velocity_selector_speed, incident_energy_mev)
    and falls back to BT-7 file metadata columns when older events lack them.
    """

    filter_value = event.get("harmonic_filter_translation")
    if filter_value in {None, "", "unknown"}:
        filter_value = file_metadata.get("filter_state")
    normalized_filter = str(filter_value).strip().upper() if filter_value not in {None, ""} else ""
    speed = event.get("velocity_selector_speed")
    if speed is None:
        speed = _float_or_none(file_metadata.get("velocity_selector_state"))
    ei = event.get("incident_energy_mev")
    if ei is None:
        ei = _float_or_none(file_metadata.get("incident_energy_mev"))
    return {
        "harmonic_filter_translation": normalized_filter if normalized_filter in {"IN", "OUT"} else "unknown",
        "velocity_selector_speed": _float_or_none(speed),
        "incident_energy_mev": _float_or_none(ei),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_HARMONIC_VERDICT_PRIORS = {
    "suppressed_by_filter": 0.05,
    "selector_tuned_suppressed": 0.05,
    "selector_tuned_passband_width_unknown": 0.10,
    "unsuppressed_candidate": 0.40,
}


def _lambda_half_score(
    *,
    center: dict[str, Any],
    filter_states: list[str],
    velocity_states: list[str],
    planning_report_path: Any,
    selector_calibration: Any = None,
) -> dict[str, Any]:
    harmonic_state = None
    typed_filter = center.get("harmonic_filter_translation")
    typed_speed = center.get("velocity_selector_speed")
    if typed_filter not in {None, "", "unknown"} or typed_speed is not None:
        harmonic_state = harmonic_suppression_state(typed_filter, typed_speed, center.get("incident_energy_mev"), selector_calibration)
    verdict = str(harmonic_state.get("verdict")) if harmonic_state else "unknown"
    if harmonic_state and verdict in _HARMONIC_VERDICT_PRIORS:
        score = _HARMONIC_VERDICT_PRIORS[verdict]
        evidence = [f"harmonic_suppression_state_verdict_{verdict}"]
        if verdict == "selector_tuned_passband_width_unknown":
            evidence.append(
                "candidate_observation_selector_passband_center_matches_nominal_incident_wavelength_within_2_percent"
            )
            evidence.append(
                "candidate_language_only_passband_width_or_calibration_review_missing_so_harmonic_order_2_suppression_is_not_claimed"
            )
        evidence.extend(str(item) for item in harmonic_state.get("rationale") or [])
    else:
        score = 0.28
        evidence = ["base_lambda_half_prior_for_alignment_peak_review"]
        normalized_filters = {item.upper() for item in filter_states}
        normalized_velocity = {item.upper() for item in velocity_states}
        if "OUT" in normalized_filters:
            score += 0.12
            evidence.append("in_pile_filter_out_increases_lambda_half_prior")
        if "IN" in normalized_filters:
            score -= 0.12
            evidence.append("in_pile_filter_in_lowers_lambda_half_prior_but_does_not_rule_out")
        if normalized_velocity & {"IN", "ON", "TRUE", "YES"}:
            score -= 0.18
            evidence.append("velocity_selector_present_lowers_higher_order_prior")
    if planning_report_path:
        evidence.append("structure_factor_planning_report_available_for_future_harmonic_pair_checks")
    probability = round(max(0.02, min(0.8, score)), 3)
    return {
        "event_id": center.get("event_id"),
        "axis": center.get("axis"),
        "center_deg": center.get("center_deg"),
        "probability_prior": probability,
        "confidence_label": "low" if probability < 0.2 else "medium" if probability >= 0.4 else "low_to_medium",
        "status": "probability_scaffold",
        "harmonic_suppression": harmonic_state,
        "calibration_status": (harmonic_state or {}).get("calibration_status"),
        "evidence": evidence,
        "missing_for_stronger_score": [
            "manual post-sample filter state",
            "reviewed intended HKL and harmonic alternatives",
            "material structure factors for candidate lambda and lambda/2 reflections",
        ],
    }


def _powder_background_score(*, center: dict[str, Any], peak_table: list[dict[str, Any]], materials: list[str], tolerance_deg: float) -> dict[str, Any]:
    matches = []
    center_deg = float(center.get("center_deg") or 0.0)
    allowed_materials = {str(material).lower() for material in materials}
    for row in peak_table:
        material = str(row.get("material") or "")
        if allowed_materials and material.lower() not in allowed_materials:
            continue
        two_theta = row.get("two_theta_deg") or row.get("a4_deg")
        if two_theta is None:
            continue
        delta = abs(center_deg - float(two_theta))
        if delta <= tolerance_deg:
            matches.append(
                {
                    "material": material,
                    "hkl": row.get("hkl"),
                    "two_theta_deg": round(float(two_theta), 6),
                    "delta_deg": round(delta, 6),
                    "relative_intensity": row.get("relative_intensity"),
                }
            )
    if matches:
        best_delta = min(float(match["delta_deg"]) for match in matches)
        probability = round(max(0.15, min(0.85, 0.65 - 0.3 * best_delta / max(tolerance_deg, 1e-9))), 3)
        status = "candidate_powder_or_background_overlap"
    elif peak_table:
        probability = 0.08
        status = "no_background_peak_overlap_in_supplied_table"
    else:
        probability = 0.12 if any(material.lower() == "al" for material in materials) else 0.08
        status = "background_material_prior_without_peak_table"
    return {
        "event_id": center.get("event_id"),
        "axis": center.get("axis"),
        "center_deg": center.get("center_deg"),
        "probability_prior": probability,
        "status": status,
        "matches": matches,
        "materials_considered": materials,
        "tolerance_deg": tolerance_deg,
        "human_review_required": True,
        "missing_for_stronger_score": [] if peak_table else ["background_peak_table with material, hkl, two_theta_deg, and relative_intensity"],
    }


def _read_background_peak_table(path_value: Any) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return []
    try:
        if path.suffix.lower() in {".json", ".jsn"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("peaks") if isinstance(payload, dict) else payload
            return list(rows or [])
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def _materials_from_config(path_value: Any) -> list[str]:
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    materials = payload.get("materials") if isinstance(payload, dict) else []
    return [str(item.get("material") if isinstance(item, dict) else item) for item in materials]


def _event_metadata_values(events: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for event in events:
        metadata = event.get("metadata") or {}
        for key in keys:
            value = metadata.get(key) or event.get(key)
            if value not in {None, ""}:
                values.append(str(value))
    return sorted(set(values))


def _center_metadata_values(centers: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for center in centers:
        metadata = center.get("metadata") or {}
        for key in keys:
            value = metadata.get(key)
            if value not in {None, ""}:
                values.append(str(value))
    return sorted(set(values))


def _bt7_alignment_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    columns: list[str] = []
    values: dict[str, set[str]] = {"FilTran": set(), "Filtran": set(), "Velsel": set(), "VelSel": set(), "VelocitySelector": set(), "Ei": set()}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("#Columns"):
                    columns = stripped[len("#Columns") :].strip().split()
                    continue
                if stripped.startswith("#") or not stripped or not columns:
                    continue
                parts = stripped.split()
                for key in list(values):
                    if key in columns:
                        index = columns.index(key)
                        if index < len(parts):
                            values[key].add(parts[index])
    except OSError:
        return {}
    metadata: dict[str, Any] = {}
    for key in ("FilTran", "Filtran"):
        if len(values[key]) == 1:
            metadata["filter_state"] = next(iter(values[key]))
            break
    for key in ("Velsel", "VelSel", "VelocitySelector"):
        if len(values[key]) == 1:
            metadata["velocity_selector_state"] = next(iter(values[key]))
            break
    if len(values["Ei"]) == 1:
        metadata["incident_energy_mev"] = next(iter(values["Ei"]))
    return metadata


def _hkl_is_integral(hkl: str, tolerance: float = 0.01) -> bool:
    try:
        return all(abs(float(part) - round(float(part))) <= tolerance for part in hkl.split(","))
    except ValueError:
        return False


def _observed_hkl(event: dict[str, Any]) -> str | None:
    for value in (event.get("fit_hkl"), (event.get("metadata") or {}).get("fit_hkl"), (event.get("metadata") or {}).get("HKL")):
        normalized = _normalize_hkl(value)
        if normalized:
            return normalized
    text = " ".join(str(item or "") for item in [event.get("raw_line"), event.get("scan_descr"), event.get("scan")])
    match = re.search(r"HKL\s*=?\s*\[?(-?\d+(?:\.\d+)?)\s*,?\s+(-?\d+(?:\.\d+)?)\s*,?\s+(-?\d+(?:\.\d+)?)\]?", text, re.IGNORECASE)
    if match:
        return _normalize_hkl([match.group(1), match.group(2), match.group(3)])
    return None


def _normalize_hkl(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = re.findall(r"-?\d+(?:\.\d+)?", value)
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        return None
    if len(parts) < 3:
        return None
    normalized = []
    for part in parts[:3]:
        number = float(part)
        normalized.append(str(int(number)) if number.is_integer() else f"{number:g}")
    return ",".join(normalized)


def _planning_report_summary(path_value: Any) -> dict[str, Any]:
    summary = {
        "status": "not_supplied",
        "role": "GSAS-II/ICSD-backed structure-factor suggestions choose likely alignment peaks and likely background conflicts before empirical scans are reviewed.",
    }
    if not path_value:
        return summary
    path = Path(str(path_value)).expanduser()
    summary["path"] = str(path)
    if not path.exists():
        summary["status"] = "missing"
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["status"] = "unreadable"
        summary["error"] = str(exc)
        return summary
    peaks = payload.get("alignment_peaks") or payload.get("candidate_peaks") or []
    summary.update(
        {
            "status": "loaded",
            "formula": payload.get("formula"),
            "scattering_plane": payload.get("scattering_plane"),
            "candidate_peak_count": len(peaks),
            "top_candidates": peaks[:5],
            # Full list for observed-window matching: which reflection a scan
            # sat on is a recall question, not a ranking question, so weak
            # workhorse peaks like (002) must stay matchable.
            "all_candidates": peaks,
            "selected_structure_lattice": (payload.get("selected_structure") or {}).get("lattice"),
        }
    )
    return summary


# Default spectrometer limit tables for kinematic reachability annotation.
# Values are the BT-7 A4Limits table from the Leland ExPlanner facility file
# "NCNR - BT7.json" (February 2012): Ei row in the PG(002) basis (meV), max-A4
# row in degrees. They are labeled defaults, not guesses, and are overridable
# through workflow arguments.
_EXPLANNER_BT7_EI_LIMITS_PG002_MEV = [4.9, 83.2]
_EXPLANNER_BT7_A4_MAX_LIMITS_DEG = [120.0, 120.0]
_EXPLANNER_BT7_LIMITS_SOURCE = "Leland ExPlanner 'NCNR - BT7.json' A4Limits table (February 2012)"
_DEFAULT_A4_MIN_DEG = 1.0


def _candidate_peak_q(peak: dict[str, Any]) -> float | None:
    q_value = peak.get("q_inv_angstrom")
    if q_value is not None:
        return float(q_value)
    d_spacing = peak.get("d_spacing")
    if d_spacing:
        return 2.0 * math.pi / float(d_spacing)
    return None


def _annotate_peak_accessibility(planning_link: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Annotate candidate alignment peaks with elastic kinematic reachability.

    Calls `tas.psd_planner.check_accessibility` (the ExPlanner kinematics port)
    through the registry. Ef must be supplied explicitly; instrument limit
    tables default to the labeled ExPlanner BT-7 values and can be overridden.
    The annotation fails soft: a missing plugin or tool error becomes a status
    note, never a workflow failure.
    """
    if planning_link.get("status") != "loaded":
        return None
    ef_mev = arguments.get("ef_mev")
    if ef_mev is None:
        return {
            "status": "skipped_missing_ef_mev",
            "note": "Supply ef_mev (e.g. 14.7) to annotate candidate peaks with kinematic reachability at your Ef/A4 limits.",
        }
    candidates = list(planning_link.get("top_candidates") or [])
    labeled = [(peak, q) for peak in candidates if (q := _candidate_peak_q(peak)) is not None]
    if not labeled:
        return {
            "status": "skipped_no_candidate_q",
            "note": "Candidate peaks carry neither q_inv_angstrom nor d_spacing, so |Q| reachability cannot be evaluated.",
        }
    from tas_copilot_contracts import ToolCall
    from tas_copilot_contracts.tools import ToolResultStatus

    from .common import default_runner

    tool_arguments = {
        "ef_mev": float(ef_mev),
        "mono_reflection": str(arguments.get("mono_reflection") or "PG(002)"),
        "ana_reflection": str(arguments.get("ana_reflection") or "PG(002)"),
        "ei_limits_pg002_mev": [float(v) for v in (arguments.get("ei_limits_pg002_mev") or _EXPLANNER_BT7_EI_LIMITS_PG002_MEV)],
        "a4_max_limits_deg": [float(v) for v in (arguments.get("a4_max_limits_deg") or _EXPLANNER_BT7_A4_MAX_LIMITS_DEG)],
        "a4_min_deg": float(arguments.get("a4_min_deg") or _DEFAULT_A4_MIN_DEG),
        "q_moduli_inv_ang": [q for _, q in labeled],
        "energy_transfers_mev": [0.0],
    }
    call = ToolCall(
        tool_namespace="tas.psd_planner",
        tool_name="check_accessibility",
        arguments=tool_arguments,
        caller_workflow="alignment-planning-brief",
    )
    result = default_runner().run(call)
    if result.status != ToolResultStatus.success:
        return {
            "status": f"tool_{result.status.value}",
            "note": "tas.psd_planner.check_accessibility did not run; install the tas-psd-planner plugin to enable reachability annotation.",
        }
    accessibility = (result.output or {}).get("accessibility") or {}
    points = accessibility.get("points") or []
    reachable_count = 0
    for (peak, q_value), point in zip(labeled, points):
        elastic_check = (point.get("energy_checks") or [{}])[0]
        reachable = elastic_check.get("accessible")
        if reachable:
            reachable_count += 1
        peak["kinematic_accessibility"] = {
            "reachable_at_ef_a4_limits": reachable,
            "q_inv_ang": q_value,
            "elastic_q_window_inv_ang": elastic_check.get("q_window_inv_ang"),
        }
    return {
        "status": "annotated",
        "ef_mev": float(ef_mev),
        "mono_reflection": tool_arguments["mono_reflection"],
        "ana_reflection": tool_arguments["ana_reflection"],
        "a4_min_deg": tool_arguments["a4_min_deg"],
        "limits_source": _EXPLANNER_BT7_LIMITS_SOURCE if not arguments.get("ei_limits_pg002_mev") else "user_supplied",
        "annotated_peak_count": len(labeled),
        "reachable_peak_count": reachable_count,
        "energy_transfer_window_mev": accessibility.get("energy_transfer_window_mev"),
        "parity_status": accessibility.get("parity_status"),
        "note": "Planning estimate from the ExPlanner kinematics port; not a reviewed instrument-limit claim.",
    }


def _observed_alignment_summary(*, events: list[dict[str, Any]], epochs: list[dict[str, Any]]) -> dict[str, Any]:
    event_types = Counter(str(event.get("event_type") or "unknown") for event in events)
    stages = Counter(str(event.get("alignment_stage") or "unknown") for event in events)
    devices = Counter(device for event in events for device in event.get("scanned_devices", []))
    latest_epoch = epochs[-1] if epochs else None
    return {
        "event_count": len(events),
        "epoch_count": len(epochs),
        "event_types": dict(event_types.most_common()),
        "alignment_stages": dict(stages.most_common()),
        "devices": dict(devices.most_common(20)),
        "latest_epoch": latest_epoch,
        "empirical_state": _empirical_state(events=events, latest_epoch=latest_epoch),
    }


def _empirical_state(*, events: list[dict[str, Any]], latest_epoch: dict[str, Any] | None) -> str:
    if latest_epoch and latest_epoch.get("dominant_stage") in _FINE_STAGE_NAMES:
        return "fine_alignment_candidate"
    stages = {str(event.get("alignment_stage") or "") for event in events}
    if stages & _FINE_STAGE_NAMES:
        return "fine_alignment_candidate"
    if "second_reflection_search_or_plane_flattening" in stages:
        return "gross_plane_establishment_in_progress"
    if "first_or_second_reflection_localization" in stages:
        return "reflection_localization_in_progress"
    if events:
        return "alignment_activity_observed"
    return "no_observational_alignment_evidence"


_CHECK_AXIS_WORDS = {"slit": ["pre_sample_slit_width", "pre_sample_slit_height", "post_sample_slit_width", "post_sample_slit_height"], "translation": ["lower_translation", "upper_translation", "sample_elevation"], "tilt": ["lower_tilt", "upper_tilt"]}


def _respect_axis_policy_action(policy: dict[str, Any]) -> dict[str, str]:
    return {"action": "respect_axis_policy", "reason": "These axes are frozen for this run and must not be scanned or optimized: " + "; ".join(f"{axis} ({reason})" for axis, reason in policy["frozen"].items())}


def _checks_blocked_by_policy(checks: list[str], policy: dict[str, Any]) -> list[dict[str, str]]:
    """Next-expected checks that name a frozen axis family, with the policy reason."""
    blocked = []
    for check in checks:
        for word, axes in _CHECK_AXIS_WORDS.items():
            if word in str(check).lower():
                frozen = [(axis, frozen_reason(policy, axis)) for axis in axes if frozen_reason(policy, axis)]
                if frozen:
                    blocked.append({"check": check, "frozen_axis": frozen[0][0], "reason": frozen[0][1]})
                    break
    return blocked


def _alignment_recommendations(*, observed: dict[str, Any], planning_link: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    empirical_state = str(observed.get("empirical_state"))
    if planning_link.get("status") != "loaded":
        actions.append(
            {
                "action": "run_structure_factor_alignment_peak_suggestions",
                "reason": "Planning peaks provide the expected first/second/third reflection hypotheses before interpreting observed scans.",
                "command": "python -m tas_copilot_core.cli workflow suggest-alignment-peaks --formula [FORMULA] --icsd-db [ICSD.sqlite] --scattering-plane '[HKL1];[HKL2]' --output-dir local_runs/alignment_peaks",
            }
        )
    if empirical_state == "fine_alignment_candidate":
        actions.append({"action": "prepare_simulator_only_fine_alignment_replay", "reason": "Observed evidence suggests gross alignment exists; fine rocking/slit/translation logic can be rehearsed in NICE simulation.", "command": "python -m tas_copilot_core.cli workflow alignment-fine-replay-plan --state-json local_runs/alignment_state_epochs/alignment_state_epochs.json --output-dir local_runs/alignment_fine_replay_plan"})
    else:
        actions.append({"action": "preserve_gross_alignment_hypotheses", "reason": "Gross alignment remains suggestion-heavy until at least two reflection assignments and state continuity are reviewed."})
    actions.append({"action": "review_alignment_state_template", "reason": "Human labels are needed to calibrate which observed scan/log patterns really indicate accepted alignment states."})
    return actions


def _fine_alignment_plan(*, latest_epoch: dict[str, Any], brief: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    count_time = float(arguments.get("count_time_seconds") or 1.0)
    rocking_motor = str(arguments.get("rocking_motor") or "A3")
    plan_id = f"alignment-fine-replay-{_short_hash(json.dumps({'epoch': latest_epoch.get('epoch_id'), 'motor': rocking_motor}, sort_keys=True))}"
    return {
        "plan_id": plan_id,
        "executor": "nice_bt7_replay_or_tasai_adaptive_nice_sim",
        "mode": "simulation_only",
        "requires_reviewed_node_aliases": True,
        "source_epoch_id": latest_epoch.get("epoch_id"),
        "observed_state_hint": (brief.get("observational_evidence") or {}).get("empirical_state"),
        "steps": [
            {"step_id": "rocking_preview", "operation": "scan_motor", "motor": rocking_motor, "relative_range_deg": [-0.4, 0.4], "points": 17, "count_time_seconds": count_time, "analysis": "fit_peak_center_width"},
            {"step_id": "theta_two_theta_check", "operation": "coupled_scan", "motors": ["A3", "A4"], "relative_range_deg": [-0.25, 0.25], "points": 15, "count_time_seconds": count_time, "analysis": "check_center_consistency"},
            {"step_id": "vertical_slit_preview", "operation": "scan_slit_pair", "slits": ["FrontSlitHeight", "BackSlitHeight"], "range_mm": [0.0, 50.0], "points": 26, "count_time_seconds": count_time, "analysis": "ramp_then_saturation_or_late_uptick"},
            {"step_id": "horizontal_slit_preview", "operation": "scan_slit_pair", "slits": ["FrontSlitWidth", "BackSlitWidth"], "range_mm": [0.0, 50.0], "points": 26, "count_time_seconds": count_time, "analysis": "width_and_centering_proxy"},
        ],
        "stop_conditions": ["peak_not_found", "multimodal_peak_requires_review", "unexpected_state_epoch_boundary", "node_alias_unreviewed"],
        "human_injection_points": ["after_rocking_preview", "after_theta_two_theta_check", "after_slit_previews"],
        "safe_conclusion": "This plan tests the API and analysis loop in simulation only; it is not a live-control approval.",
    }


def _nice_replay_plan_from_alignment_plan(plan: dict[str, Any], *, arguments: dict[str, Any]) -> dict[str, Any]:
    count_time = float(arguments.get("count_time_seconds") or 1.0)
    return {
        "workflow_name": "alignment-fine-replay-to-nice",
        "workflow_boundary": "simulation_only_training_harness",
        "source_alignment_plan_id": plan.get("plan_id"),
        "translation_status": "reviewed_simulator_aliases",
        "notes": [
            "Only A3/A4 simulator aliases are emitted as executable replay recipes in this scaffold.",
            "Slit and translation steps require reviewed NICE node aliases before execution.",
            "Counts can be supplied by a replay/model backend; live detector values are not set on a real instrument.",
        ],
        "recipes": [
            {
                "replay_recipe_id": "alignment_rocking_preview",
                "translation_status": "reviewed_simulator_aliases",
                "counts": count_time,
                "npts": 17,
                "ranges": [
                    {
                        "axis": "A3",
                        "nice_node_hint": "sampleTheta.softPosition",
                        "start": -0.4,
                        "stop": 0.4,
                        "relative": True,
                    }
                ],
            },
            {
                "replay_recipe_id": "alignment_theta_two_theta_check",
                "translation_status": "reviewed_simulator_aliases",
                "counts": count_time,
                "npts": 15,
                "ranges": [
                    {
                        "axis": "A3",
                        "nice_node_hint": "sampleTheta.softPosition",
                        "start": -0.25,
                        "stop": 0.25,
                        "relative": True,
                    },
                    {
                        "axis": "A4",
                        "nice_node_hint": "sampleTwoTheta.softPosition",
                        "start": -0.5,
                        "stop": 0.5,
                        "relative": True,
                    },
                ],
            },
        ],
        "untranslated_steps": [
            step
            for step in plan.get("steps") or []
            if step.get("operation") in {"scan_slit_pair"}
        ],
        "human_injection_points": plan.get("human_injection_points") or [],
        "safety_boundary": "simulation_only; not approved live control",
    }


def _adaptive_alignment_iterations(*, state_payload: dict[str, Any], replay_report: dict[str, Any], max_iterations: int) -> list[dict[str, Any]]:
    estimates = list(state_payload.get("state_estimates") or [])
    latest_state = estimates[-1] if estimates else {}
    replay_plan = replay_report.get("replay_plan") or {}
    base_steps = list(replay_plan.get("steps") or [])
    iterations = []
    for index in range(max(1, max_iterations)):
        stage_steps = []
        for step in base_steps:
            operation = str(step.get("operation") or "")
            if operation == "scan_motor":
                decision_rule = "fit one-to-three Gaussian/preview peaks; continue only if one dominant peak or reviewed choice exists"
                next_state = "rocking_center_review"
            elif operation == "coupled_scan":
                decision_rule = "compare theta-two-theta center with rocking center; preserve contradiction if centers disagree"
                next_state = "center_consistency_review"
            elif operation == "scan_slit_pair":
                decision_rule = "fit ramp/saturation or S-curve; request human review before translating sample"
                next_state = "slit_or_translation_review"
            else:
                decision_rule = "preserve evidence and request review"
                next_state = "review_required"
            stage_steps.append(
                {
                    "step_id": step.get("step_id"),
                    "operation": operation,
                    "decision_rule": decision_rule,
                    "next_state_if_successful": next_state,
                    "human_injection_allowed": True,
                    "execution_status": "planned_for_simulator_rehearsal",
                }
            )
        iterations.append(
            {
                "iteration_index": index + 1,
                "starting_state_estimate": latest_state.get("state") or "unknown_alignment_state",
                "steps": stage_steps,
                "stop_if": ["human_stop", "model_selects_multiple_peaks_without_review", "translation_requires_uncalibrated_direction", "state_epoch_break_detected"],
                "expected_output": "updated_alignment_state_estimate_and_reviewable_evidence",
            }
        )
    return iterations


def _path_hash(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _markdown_state_epochs(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Alignment State Epochs",
        "",
        f"Status: `{report.get('status')}`",
        "",
        f"- Events: `{summary.get('event_count', 0)}`",
        f"- Epochs: `{summary.get('epoch_count', 0)}`",
        f"- Review rows: `{summary.get('review_template_rows', 0)}`",
        "",
        "## Epochs",
        "",
    ]
    for epoch in report.get("state_epochs") or []:
        lines.append(f"- `{epoch.get('epoch_id')}`: {epoch.get('dominant_stage')} ({epoch.get('event_count')} events, confidence `{epoch.get('confidence')}`)")
    lines.extend(["", "## Review", "", f"Template: `{(report.get('review_template') or {}).get('path')}`", "", "## Boundary", "", report.get("interpretation", "")])
    return "\n".join(lines)


def _markdown_planning_brief(report: dict[str, Any]) -> str:
    observed = report.get("observational_evidence") or {}
    planning = report.get("planning_evidence") or {}
    lines = [
        "# Alignment Planning Brief",
        "",
        f"Status: `{report.get('status')}`",
        "",
        f"- Planning evidence: `{planning.get('status')}`",
        f"- Observed empirical state: `{observed.get('empirical_state')}`",
        f"- Observed events: `{observed.get('event_count', 0)}`",
        "",
        "## Recommended Next Actions",
        "",
    ]
    for action in report.get("recommended_next_actions") or []:
        lines.append(f"- `{action.get('action')}`: {action.get('reason')}")
    accessibility = report.get("kinematic_accessibility")
    if accessibility:
        lines.extend(["", "## Kinematic Accessibility", "", f"Status: `{accessibility.get('status')}`"])
        if accessibility.get("status") == "annotated":
            lines.append(
                f"- Reachable at Ef={accessibility.get('ef_mev')} meV / A4 limits: "
                f"`{accessibility.get('reachable_peak_count')}/{accessibility.get('annotated_peak_count')}` candidate peaks"
            )
            lines.append(f"- Limits source: `{accessibility.get('limits_source')}`")
            for peak in (planning.get("top_candidates") or []):
                annotation = peak.get("kinematic_accessibility")
                if annotation:
                    lines.append(
                        f"- `{peak.get('peak_id')}`: |Q|={annotation.get('q_inv_ang'):.3f} 1/Ang, "
                        f"reachable=`{annotation.get('reachable_at_ef_a4_limits')}`"
                    )
            lines.append(f"- Note: {accessibility.get('note')}")
        elif accessibility.get("note"):
            lines.append(f"- {accessibility.get('note')}")
    lines.extend(["", "## Autonomy Boundary", "", f"- Live control: `{(report.get('autonomy_boundary') or {}).get('live_control')}`"])
    return "\n".join(lines)


def _markdown_replay_plan(report: dict[str, Any]) -> str:
    plan = report.get("replay_plan") or {}
    lines = [
        "# Alignment Fine Replay Plan",
        "",
        f"Status: `{report.get('status')}`",
        f"Simulation only: `{report.get('simulation_only')}`",
        "",
        f"Plan ID: `{plan.get('plan_id')}`",
        "",
        "## Steps",
        "",
    ]
    for step in plan.get("steps") or []:
        lines.append(f"- `{step.get('step_id')}`: {step.get('operation')}")
    lines.extend(["", "## Safety Boundary", "", str(report.get("safety_boundary") or "")])
    return "\n".join(lines)


def _markdown_adaptive_loop(report: dict[str, Any]) -> str:
    lines = [
        "# Alignment Adaptive NICE Loop",
        "",
        f"Status: `{report.get('status')}`",
        f"Simulation only: `{report.get('simulation_only')}`",
        f"Executor: `{report.get('executor')}`",
        "",
        "## Loop",
        "",
    ]
    loop = report.get("adaptive_loop") or {}
    lines.append(f"- Pattern: `{loop.get('control_pattern')}`")
    lines.append(f"- Human injection points: `{', '.join(loop.get('human_injection_points') or [])}`")
    lines.extend(["", "## Iterations", ""])
    for iteration in report.get("iterations") or []:
        lines.append(f"- Iteration `{iteration.get('iteration_index')}` from `{iteration.get('starting_state_estimate')}`")
    lines.extend(["", "## Boundary", "", str(report.get("safety_boundary") or "")])
    return "\n".join(lines)


def _markdown_state_machine(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Alignment State Machine",
        "",
        f"Status: `{report.get('status')}`",
        "",
        f"- Events: `{summary.get('event_count', 0)}`",
        f"- Epochs: `{summary.get('epoch_count', 0)}`",
        f"- Flagged contradictions: `{summary.get('flagged_contradictions', 0)}`",
        "",
        "## State Estimates",
        "",
    ]
    for estimate in report.get("state_estimates") or []:
        lines.append(f"- `{estimate.get('epoch_id')}`: `{estimate.get('state')}` confidence `{estimate.get('confidence')}`")
    lines.extend(["", "## Contradiction Checks", ""])
    for check in report.get("contradiction_checks") or []:
        lines.append(f"- `{check.get('check_id')}`: `{check.get('status')}`")
    scores = report.get("lambda_half_and_background_scores") or {}
    lines.extend(["", "## Lambda/2 And Background Scores", ""])
    lines.append(f"- Status: `{scores.get('status')}`")
    lines.append(f"- Observed centers: `{scores.get('observed_center_count', 0)}`")
    lines.append(f"- Background peak table: `{scores.get('background_peak_table_status')}`")
    lines.extend(["", "## Interpretation", "", str(report.get("interpretation") or "")])
    return "\n".join(lines)
