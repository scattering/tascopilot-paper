from __future__ import annotations

import json
import math
import hashlib
import getpass
import re
import shlex
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable

import yaml

from tas_copilot_contracts import (
    AdaptiveCandidatePoint,
    AdaptiveCountRequest,
    AdaptiveCountResult,
    AdaptiveObservation,
    AdaptivePolicyProposal,
    AdaptivePolicyRequest,
    ProvenanceRecord,
)
from tas_copilot_core.counting import counting_measurement
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.nice_bt7_replay import (
    nice_bt7_replay,
    run_nice_container_session_replay,
    start_nice_container_session,
    stop_nice_container_session,
)


DEFAULT_SOFTWARE_LIMITS = {
    "Q.H": {"min": -10.0, "max": 10.0},
    "Q.K": {"min": -10.0, "max": 10.0},
    "Q.L": {"min": -10.0, "max": 10.0},
    "et.deltaE": {"min": -80.0, "max": 80.0},
    "sampleTheta.softPosition": {"min": -180.0, "max": 180.0},
    "sampleTwoTheta.softPosition": {"min": -5.0, "max": 140.0},
}

TASAI_ANALYTIC_MODELS = {
    "ferromagnetic_magnon",
    "antiferromagnetic_2d",
    "phonon_dispersion",
    "double_peak",
}

_HPC_POLICY_BACKENDS = {"tacc_tasai", "tasai_tacc", "tasai_bumps_mcmc", "tacc_andie", "andie_tacc", "andie_bumps_mcmc"}
_HPC_COUNT_BACKENDS = {"tacc_spinw", "spinw_tacc", "pyspinw_tacc"}
_DEFAULT_ANDIE_CONTAINER_IMAGE = "tascopilot-andie:latest"
_DEFAULT_ANDIE_CONTAINER_WORKSPACE = "local_runs/external/andie-container"
_DEFAULT_ANDIE_SOURCE_ROOT = "local_runs/external/ANDiE-v1_0"
_DEFAULT_HPC_CONTAINER_RUNTIME = "apptainer"
_DEFAULT_REMOTE_ROOT = "${WORK:-$HOME}/tascopilot"
_DEFAULT_ANDIE_REMOTE_IMAGE = f"{_DEFAULT_REMOTE_ROOT}/images/tascopilot-andie_latest.sif"
_DEFAULT_ANDIE_REMOTE_WORKSPACE = f"{_DEFAULT_REMOTE_ROOT}/andie-container"
_DEFAULT_ANDIE_REMOTE_SOURCE_ROOT = f"{_DEFAULT_REMOTE_ROOT}/ANDiE-v1_0"


def tasai_adaptive_nice_sim(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a simulation-only adaptive tasAI/NICE/SpinW loop.

    This is the first integration slice: a tasAI-style policy proposes points,
    the NICE replay adapter executes simulator moves/counts, and a SpinW-style
    model supplies count overlays for the adaptive update. The default backends
    are deterministic fixtures so the workflow remains testable without a
    running NICE server or external tasAI/pySpinW adapter.
    """

    arguments = _apply_runtime_profile(dict(arguments))
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_backend = str(arguments.get("policy_backend") or "fixture_tasai")
    count_backend = str(arguments.get("count_backend") or "fixture_spinw")
    nice_executor = str(arguments.get("nice_executor") or "mock")
    max_steps = int(arguments.get("max_steps") or 5)
    count_time = float(arguments.get("count_time") or 1.0)
    suggestion_poll_seconds = float(arguments.get("suggestion_poll_seconds") or 0.0)
    step_delay_seconds = float(arguments.get("step_delay_seconds") or 0.0)
    hpc_action = str(arguments.get("hpc_action") or "prepare")
    host = str(arguments.get("host") or "127.0.0.1")
    port = int(arguments.get("port") or 1232)
    client_name = str(arguments.get("client_name") or "tas-copilot-tasai-adaptive")
    allow_unreviewed_node_hints = bool(arguments.get("allow_unreviewed_node_hints") or False)
    allow_live_host = bool(arguments.get("allow_live_host") or False)
    python_executable = arguments.get("python_executable")
    prepare_hpc_handoff = bool(arguments.get("prepare_hpc_handoff") or False)

    hpc_requested = policy_backend in _HPC_POLICY_BACKENDS or count_backend in _HPC_COUNT_BACKENDS
    if hpc_requested and prepare_hpc_handoff:
        return _prepare_hpc_handoff_report(output_dir, arguments, policy_backend=policy_backend, count_backend=count_backend, hpc_action=hpc_action)
    if policy_backend in _HPC_POLICY_BACKENDS:
        return _unavailable(output_dir, f"Policy backend {policy_backend!r} requires a TACC/HPC adapter. Rerun with --prepare-hpc-handoff to write a reviewed job payload, or use fixture_tasai/external_tasai_adapter locally.")
    if policy_backend not in {"fixture_tasai", "external_tasai_adapter", "external_andie_adapter", "andie_container"}:
        return _unavailable(output_dir, f"Policy backend {policy_backend!r} is not installed. Use fixture_tasai, external_tasai_adapter, external_andie_adapter, andie_container, or prepare a TACC handoff.")
    if count_backend in _HPC_COUNT_BACKENDS:
        return _unavailable(output_dir, f"Count backend {count_backend!r} requires a TACC/HPC adapter. Rerun with --prepare-hpc-handoff to write a reviewed job payload, or use fixture_spinw/tasai_analytic locally.")
    if count_backend not in {"fixture_spinw", "tasai_analytic", "external_pyspinw_adapter"}:
        return _unavailable(output_dir, f"Count backend {count_backend!r} is not installed. Use fixture_spinw, tasai_analytic, external_pyspinw_adapter, or register a pySpinW/SpinW adapter.")

    candidates = _load_candidate_points(arguments)
    if not candidates:
        return _missing(output_dir, ["candidate_points"], "Adaptive simulation requires candidate points or a generated fixture grid.")
    model_spec = _load_model_spec(arguments)
    human_suggestions_path = arguments.get("human_suggestions")
    suggestions = _load_human_suggestions(human_suggestions_path)
    software_limits, limits_source = _load_software_limits(arguments)
    count_model, count_model_error = _prepare_count_model(arguments, model_spec, count_backend)
    if count_model_error:
        return _unavailable(output_dir, count_model_error)
    policy_model, policy_model_error = _prepare_policy_model(arguments, policy_backend)
    if policy_model_error:
        return _unavailable(output_dir, policy_model_error)

    trace_path = output_dir / "adaptive_trace.jsonl"
    overlay_path = output_dir / "simulation_count_overlay.jsonl"
    state_path = output_dir / "tasai_policy_state.json"
    review_template_path = output_dir / "adaptive_decision_review_template.jsonl"
    nice_trace_index: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stopped_reason = None
    nice_session: dict[str, Any] | None = None
    nice_session_stop: dict[str, Any] | None = None

    if nice_executor == "nice_container_session":
        nice_session = start_nice_container_session(
            session_root=output_dir,
            container_image=str(arguments.get("container_image") or "nice-bt7-replay-smoke:latest"),
            nice_checkout=str(arguments.get("nice_checkout") or ""),
            docker_executable=str(arguments.get("docker_executable") or "docker"),
            timeout_seconds=int(arguments.get("timeout_seconds") or 300),
        )
        if nice_session.get("status") != "nice_container_session_ready":
            nice_session_stop = stop_nice_container_session(nice_session)
            return _unavailable(
                output_dir,
                "NICE container session could not start: "
                f"{nice_session.get('stderr_tail') or nice_session.get('timeout_message') or nice_session.get('status')}; "
                f"stop_status={nice_session_stop.get('status')}",
            )

    point_session = None
    if nice_executor == "nice_remote_session":
        from .nice_point_session import NicePointSession, NiceSessionError

        # One measurement for the whole run => one data file. Control nodes must
        # be declared up front, so take the union over the candidate set rather
        # than the first proposal.
        control_nodes = sorted({node for item in candidates for node in (item.get("nodes") or {})})
        try:
            point_session = NicePointSession(
                server_script=_point_server_script(),
                spool_dir=output_dir / "nice_session",
                host=host,
                port=port,
                client_name=client_name,
                file_prefix=_safe_prefix(f"tasai_{output_dir.name}"),
                control_nodes=control_nodes,
                python_executable=python_executable,
                title=f"tas-copilot adaptive ({policy_backend})",
                allow_live_host=allow_live_host,
                timeout_seconds=float(arguments.get("timeout_seconds") or 600),
            )
            point_session.start()
        except NiceSessionError as exc:
            return _unavailable(output_dir, f"NICE point session could not start: {exc}")

    try:
        with trace_path.open("w", encoding="utf-8") as trace_handle, overlay_path.open("w", encoding="utf-8") as overlay_handle:
            for step in range(max_steps):
                if human_suggestions_path:
                    suggestions = _load_human_suggestions(human_suggestions_path)
                if suggestion_poll_seconds > 0:
                    time.sleep(suggestion_poll_seconds)
                    if human_suggestions_path:
                        suggestions = _load_human_suggestions(human_suggestions_path)
                action = suggestions.get(step)
                if action and action.get("action") == "stop":
                    stopped_reason = f"human_stop_at_step_{step}"
                    break
                local_count_time = float(action.get("count_time", count_time)) if action else count_time
                try:
                    proposal = _choose_next_point(
                        candidates,
                        observations,
                        model_spec,
                        software_limits,
                        count_backend,
                        count_model,
                        policy_backend,
                        policy_model,
                        step=step,
                        action=action,
                    )
                except Exception as exc:
                    stopped_reason = f"policy_backend_failed:{type(exc).__name__}"
                    rejected.append(
                        {
                            "point_id": f"policy-failure-step-{step:03d}",
                            "rejected_reason": stopped_reason,
                            "policy_reason": str(exc),
                            "policy_backend": policy_backend,
                        }
                    )
                    break
                if proposal is None:
                    stopped_reason = "no_candidate_within_software_limits"
                    break
                if proposal.get("rejected_reason"):
                    rejected.append(proposal)
                    stopped_reason = proposal["rejected_reason"]
                    break
                if not (action and "count_time" in action) and proposal.get("suggested_count_time") is not None:
                    local_count_time = float(proposal["suggested_count_time"])

                step_dir = output_dir / f"nice_step_{step:03d}"
                replay_plan = _write_single_point_replay_plan(step_dir, proposal, local_count_time)
                replay_payload = {
                    "replay_plan": str(replay_plan),
                    "output_dir": str(step_dir),
                    "executor": nice_executor if nice_executor != "nice_container_session" else "dry_run",
                    "mode": "point_by_point_adaptive",
                    "host": host,
                    "port": port,
                    "client_name": client_name,
                    "container_image": arguments.get("container_image"),
                    "nice_checkout": arguments.get("nice_checkout"),
                    "docker_executable": arguments.get("docker_executable"),
                    "timeout_seconds": arguments.get("timeout_seconds"),
                    "file_prefix": f"tasai_adaptive_step_{step:03d}",
                    "preset_time": local_count_time,
                    "max_points": 1,
                    "allow_unreviewed_node_hints": allow_unreviewed_node_hints,
                    "allow_live_host": allow_live_host,
                    "python_executable": python_executable,
                }
                if point_session is not None:
                    # The replay plan above is still written: it records what was
                    # requested at this step, even though the point goes to the
                    # already-open measurement instead of a new subprocess.
                    nice_result = _session_step(point_session, step, proposal, local_count_time, step_dir)
                else:
                    nice_result = nice_bt7_replay(replay_payload)
                if nice_session is not None and nice_result.get("status") == "success":
                    execution = run_nice_container_session_replay(
                        session=nice_session,
                        points_path=Path((nice_result.get("artifacts") or {})["points"]),
                        script_path=Path((nice_result.get("artifacts") or {})["script"]),
                        trace_path=step_dir / "nice_replay_trace.jsonl",
                        file_prefix=f"tasai_adaptive_step_{step:03d}",
                        count_seconds=local_count_time,
                        timeout_seconds=int(arguments.get("timeout_seconds") or 300),
                        execution=dict(nice_result.get("execution") or {}),
                    )
                    nice_result["execution"] = execution
                    step_trace_path = step_dir / "nice_replay_trace.jsonl"
                    nice_result.setdefault("artifacts", {})["trace"] = str(step_trace_path) if step_trace_path.exists() else None
                    nice_result["status"] = "success" if execution.get("status") == "nice_container_session_executed" else "error"
                    json_artifact = (nice_result.get("artifacts") or {}).get("json")
                    if json_artifact:
                        Path(json_artifact).write_text(json.dumps(nice_result, indent=2, sort_keys=True), encoding="utf-8")
                nice_trace_index.append(
                    {
                        "step": step,
                        "status": nice_result.get("status"),
                        "execution_status": (nice_result.get("execution") or {}).get("status"),
                        "report": (nice_result.get("artifacts") or {}).get("json"),
                        "trace": (nice_result.get("artifacts") or {}).get("trace"),
                    }
                )
                if nice_result.get("status") != "success":
                    stopped_reason = f"nice_executor_{nice_result.get('status')}"
                    break

                overlay = _count_overlay(proposal, model_spec, count_time=local_count_time, step=step, count_backend=count_backend, count_model=count_model)
                measured = _counts_from_nice_trace((nice_result.get("execution") or {}).get("trace_path") or (nice_result.get("artifacts") or {}).get("trace"))
                observation = {
                    "step": step,
                    "point_id": proposal["point_id"],
                    "nodes": proposal["nodes"],
                    "policy_score": proposal["policy_score"],
                    "policy_reason": proposal["policy_reason"],
                    "policy_metadata": proposal.get("policy_metadata") or {},
                    "count_time": local_count_time,
                    "nice_execution_status": (nice_result.get("execution") or {}).get("status"),
                    "simulation_count_overlay": overlay,
                }
                if measured is not None and measured.get("error"):
                    # The instrument counted but we could not read the result. The
                    # policy is flying blind from here, so put it in the trace
                    # rather than letting the step look like a missing observation.
                    observation["count_error"] = measured["error"]
                elif measured is not None:
                    # a real (or mock) count came back from NICE: that is the observation
                    observation["observed_counts"] = float(measured["counts"])
                    observation["uncertainty"] = float(counting_measurement([float(measured["counts"])]).errors[0])
                    observation["count_source"] = measured["source"]
                    if measured.get("live_time") is not None:
                        observation["count_time_actual"] = float(measured["live_time"])
                    for extra in ("segments", "monitor", "roi_mismatch",
                                  "achieved_nodes", "achieved_angles"):
                        if measured.get(extra) is not None:
                            observation[extra] = measured[extra]
                    position_error = _position_error(proposal["nodes"], measured.get("achieved_nodes"))
                    if position_error:
                        # The count is real but it is not a count OF the point
                        # the policy chose. Feeding it to the surrogate as if it
                        # were would teach the model the wrong thing.
                        observation["position_error"] = position_error
                elif nice_executor in {"dry_run"}:
                    # no instrument in the loop: fall back to the simulated expectation so
                    # policy adapters still receive an observation stream
                    observation["observed_counts"] = float(overlay.get("expected_counts") or 0.0)
                    observation["uncertainty"] = float(counting_measurement([float(overlay.get("expected_counts") or 0.0)]).errors[0])
                    observation["count_source"] = "simulation_overlay"
                observations.append(observation)
                trace_handle.write(json.dumps(observation, sort_keys=True) + "\n")
                trace_handle.flush()
                overlay_handle.write(json.dumps(overlay, sort_keys=True) + "\n")
                overlay_handle.flush()
                if step_delay_seconds > 0:
                    time.sleep(step_delay_seconds)
    finally:
        if nice_session is not None:
            nice_session_stop = stop_nice_container_session(nice_session)
        if point_session is not None:
            # Closes the measurement, which is what finalises the data file.
            point_session.stop()

    state = {
        "policy_backend": policy_backend,
        "count_backend": count_backend,
        "nice_executor": nice_executor,
        "observed_point_ids": [item["point_id"] for item in observations],
        "observation_count": len(observations),
        "candidate_count": len(candidates),
        "rejected_candidates": rejected,
        "stopped_reason": stopped_reason,
        "software_limits": software_limits,
        "limits_source": limits_source,
        "human_suggestion_count": len(suggestions),
        "human_suggestions_mode": "dynamic_jsonl_poll" if human_suggestions_path else "not_supplied",
        "suggestion_poll_seconds": suggestion_poll_seconds,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    review_count = _write_adaptive_decision_review_template(
        review_template_path,
        observations,
        rejected,
        workflow_context={
            "policy_backend": policy_backend,
            "count_backend": count_backend,
            "nice_executor": nice_executor,
            "software_limits_source": limits_source,
            "state_path": str(state_path),
        },
    )
    report = {
        "workflow_name": "tasai-adaptive-nice-sim",
        "status": "success" if observations else "missing_input",
        "safety_boundary": "simulation_only",
        "policy_backend": policy_backend,
        "count_backend": count_backend,
        "nice_executor": nice_executor,
        "steps_completed": len(observations),
        "stopped_reason": stopped_reason,
        "software_limits": {"source": limits_source, "limits": software_limits},
        "human_intervention_model": "operator may suggest points, adjust count time, or stop; human approval is not required in simulation",
        "human_suggestions": {
            "mode": "dynamic_jsonl_poll" if human_suggestions_path else "not_supplied",
            "path": str(human_suggestions_path) if human_suggestions_path else None,
            "poll_seconds": suggestion_poll_seconds,
        },
        "runtime_policy": {
            "local_backends": ["fixture_tasai", "fixture_spinw", "tasai_analytic"],
            "external_adapter_backends": ["external_tasai_adapter", "external_andie_adapter", "andie_container"],
            "external_count_backends": ["external_pyspinw_adapter"],
            "hpc_backends_expected": ["tasAI MCMC/bumps", "ANDiE model discrimination/MCMC", "SpinW/pySpinW"],
            "recommended_hpc": "TACC",
            "status": "remote_hpc_job adapters not registered in this workflow yet",
        },
        "nice_session": {
            "mode": "persistent_container" if nice_session is not None else "per_step_or_mock",
            "status": nice_session.get("status") if nice_session else None,
            "container_name": nice_session.get("container_name") if nice_session else None,
            "stop_status": nice_session_stop,
            "note": "nice_container_session starts one simulator container and reuses it for all adaptive steps" if nice_session else None,
        },
        "hardware_safety_note": "Hardware interlocks and limit switches are outside this workflow. This workflow only respects software operating limits and simulator state.",
        "observations": observations,
        "nice_trace_index": nice_trace_index,
        "artifacts": {
            "adaptive_trace": str(trace_path),
            "simulation_count_overlay": str(overlay_path),
            "tasai_policy_state": str(state_path),
            "adaptive_decision_review_template_jsonl": str(review_template_path),
        },
        "review_artifacts": {
            "adaptive_decision_review_template_jsonl": str(review_template_path),
            "row_count": review_count,
        },
        "provenance": [
            ProvenanceRecord(
                source_type="workflow",
                source_id="tasai-adaptive-nice-sim",
                method="adaptive_policy_count_overlay_nice_replay_adapter",
                confidence=0.7,
            ).model_dump(mode="json")
        ],
        "limitations": [
            "fixture_tasai is a deterministic policy surrogate until the external tasAI adapter is registered",
            "external_tasai_adapter loads tasAI-owned policy code through a JSON-compatible propose_next_point boundary",
            "external_andie_adapter uses the same boundary for separate script/module ANDiE adapter code",
            "andie_container runs an external ANDiE container workspace through JSON input/output and keeps notebook references outside tascopilot",
            "fixture_spinw is a deterministic count overlay until pySpinW/SpinW is connected",
            "external_pyspinw_adapter uses a JSON-compatible simulate_counts boundary owned by the pySpinW adapter repo and should use TACC/container runtimes for heavy pySpinW use",
            "tasai_analytic imports lightweight analytic examples from an external tasAI checkout and is a simulation count source, not a live detector backend",
            "NICE-native detector readbacks and SpinW/model counts are kept separate",
            "no live-control path is exposed by this workflow",
        ],
    }
    report["artifacts"].update(
        write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_markdown,
            json_filename="tasai_adaptive_nice_sim_report.json",
            markdown_filename="tasai_adaptive_nice_sim_report.md",
        )
    )
    Path(report["artifacts"]["json"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_candidate_points(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    raw = arguments.get("candidate_points")
    if not raw and arguments.get("candidate_points_file"):
        raw = json.loads(Path(arguments["candidate_points_file"]).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, str) and raw.strip():
        raw = json.loads(raw)
    if not raw:
        raw = _fixture_candidate_grid(arguments)
    points = []
    for index, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        nodes = item.get("nodes") or _nodes_from_qe(item)
        if not nodes:
            continue
        preserved = {
            str(key): value
            for key, value in item.items()
            if key not in {"point_id", "nodes", "h", "k", "l", "e", "energy_transfer"}
        }
        preserved.update({"point_id": str(item.get("point_id") or f"candidate-{index:03d}"), "nodes": {str(k): float(v) for k, v in nodes.items()}})
        points.append(preserved)
    return points


def _apply_runtime_profile(arguments: dict[str, Any]) -> dict[str, Any]:
    profile_path = arguments.get("runtime_profile")
    if not profile_path:
        return arguments
    path = Path(str(profile_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"runtime_profile not found: {path}")
    profile = _render_runtime_profile_templates(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    remote_root = str(profile.get("remote_root") or "").rstrip("/")
    andie = profile.get("andie") or {}
    defaults = {
        "hpc_system": profile.get("profile_id") or profile.get("site"),
        "hpc_container_runtime": profile.get("container_runtime"),
        "hpc_account": profile.get("default_account"),
        "hpc_partition": profile.get("default_partition"),
        "hpc_login": profile.get("login_host"),
        "controlpath": profile.get("controlpath"),
        "andie_remote_image": andie.get("image"),
        "andie_remote_workspace": andie.get("workspace"),
        "andie_remote_source_root": andie.get("source_root"),
    }
    if remote_root:
        output_name = _safe_remote_name(Path(str(arguments.get("output_dir") or "tasai_adaptive_nice")).name)
        defaults["hpc_work_dir"] = f"{remote_root}/andie_runtime/runs/{output_name}"
        defaults["hpc_remote_handoff_dir"] = defaults["hpc_work_dir"]
    for key, value in defaults.items():
        if value not in {None, ""} and not arguments.get(key):
            arguments[key] = value
    arguments["runtime_profile_loaded"] = {
        "path": str(path),
        "profile_id": profile.get("profile_id"),
        "architecture": profile.get("architecture"),
        "remote_root": remote_root or None,
        "warnings": ["example_profile_in_use"] if path.name.endswith(".example.yaml") else [],
    }
    return arguments


def _render_runtime_profile_templates(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("{user}", getpass.getuser())
    if isinstance(value, list):
        return [_render_runtime_profile_templates(item) for item in value]
    if isinstance(value, dict):
        return {key: _render_runtime_profile_templates(item) for key, item in value.items()}
    return value


def _safe_remote_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:80] or "tasai_adaptive_nice"


def _fixture_candidate_grid(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    center_h = float(arguments.get("center_h") or 0.0)
    e_value = float(arguments.get("energy_transfer") or 0.0)
    offsets = [-0.2, -0.1, 0.0, 0.1, 0.2]
    return [
        {"point_id": f"fixture-qh-{index}", "nodes": {"Q.H": center_h + offset, "Q.K": 0.0, "Q.L": 0.0, "et.deltaE": e_value}}
        for index, offset in enumerate(offsets)
    ]


def _prepare_hpc_handoff_report(output_dir: Path, arguments: dict[str, Any], *, policy_backend: str, count_backend: str, hpc_action: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidate_points(arguments)
    model_spec = _load_model_spec(arguments)
    software_limits, limits_source = _load_software_limits(arguments)
    candidates, rejected_candidates = _filter_candidates_by_software_limits(candidates, software_limits)
    hpc_dir = output_dir / "tacc_handoff"
    hpc_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    hpc_system = str(arguments.get("hpc_system") or "vista")
    hpc_work_dir = str(arguments.get("hpc_work_dir") or f"{_DEFAULT_REMOTE_ROOT}/tasai_adaptive_nice")
    controlpath = str(arguments.get("controlpath") or "~/.ssh/cm-%r@%h:%p")
    remote_runtime = _hpc_remote_runtime(arguments, policy_backend=policy_backend, count_backend=count_backend)
    payload = {
        "workflow_name": "tasai-adaptive-nice-sim",
        "handoff_kind": "tacc_remote_hpc_job",
        "created_at": created_at,
        "policy_backend": policy_backend,
        "count_backend": count_backend,
        "hpc_system": hpc_system,
        "hpc_action": hpc_action,
        "hpc_work_dir": hpc_work_dir,
        "controlpath": controlpath,
        "hpc_account": arguments.get("hpc_account"),
        "hpc_partition": arguments.get("hpc_partition"),
        "candidate_points": candidates,
        "rejected_candidate_points": rejected_candidates,
        "model_spec": model_spec,
        "software_limits": {"source": limits_source, "limits": software_limits},
        "runtime_profile": arguments.get("runtime_profile_loaded"),
        "human_suggestions": arguments.get("human_suggestions"),
        "expected_remote_components": _hpc_components(policy_backend, count_backend),
        "remote_runtime": remote_runtime,
        "safety_boundary": "simulation_only_remote_planning",
        "notes": [
            "This handoff does not submit a job or contact TACC.",
            "Use SSH ControlPath if submitting manually; stop if MFA is prompted.",
            "Remote adapters must read this JSON and write JSON outputs; do not execute notebooks as workflow code.",
            "NICE simulator execution remains separate from TACC tasAI/pySpinW compute unless explicitly bridged.",
        ],
    }
    payload_path = hpc_dir / "tasai_adaptive_hpc_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    script_paths = _write_hpc_handoff_scripts(hpc_dir, payload)
    readme_path = hpc_dir / "README.md"
    readme_path.write_text(_hpc_handoff_readme(payload), encoding="utf-8")
    action_result = _hpc_action_result(hpc_dir, payload, arguments, hpc_action=hpc_action)
    report = {
        "workflow_name": "tasai-adaptive-nice-sim",
        "status": action_result["status"],
        "safety_boundary": "simulation_only_remote_planning",
        "policy_backend": policy_backend,
        "count_backend": count_backend,
        "steps_completed": 0,
        "hpc_handoff": {
            "system": hpc_system,
            "work_dir": hpc_work_dir,
            "controlpath": controlpath,
            "payload": str(payload_path),
            "readme": str(readme_path),
            "run_script": str(script_paths["run_script"]),
            "slurm_script": str(script_paths["slurm_script"]),
            "submission_status": action_result.get("submission_status", "not_submitted"),
            "requested_action": hpc_action,
            "action_status": action_result["action_status"],
            "action_result": action_result,
        },
        "runtime_policy": {
            "recommended_hpc": "TACC",
            "remote_components": _hpc_components(policy_backend, count_backend),
            "status": "handoff_payload_written; remote scheduler submission is explicit and not automatic",
            "runtime_profile": arguments.get("runtime_profile_loaded"),
            "candidate_filtering": {
                "software_limits_source": limits_source,
                "accepted_count": len(candidates),
                "rejected_count": len(rejected_candidates),
            },
        },
        "provenance": [
            ProvenanceRecord(
                source_type="workflow",
                source_id="tasai-adaptive-nice-sim",
                method="tacc_hpc_handoff_payload",
                confidence=0.65,
            ).model_dump(mode="json")
        ],
        "limitations": _hpc_limitations(hpc_action),
        "artifacts": {
            "hpc_payload": str(payload_path),
            "hpc_readme": str(readme_path),
            "hpc_run_script": str(script_paths["run_script"]),
            "hpc_slurm_script": str(script_paths["slurm_script"]),
        },
    }
    if action_result.get("fetched_output_dir"):
        report["artifacts"]["hpc_fetched_output_dir"] = str(action_result["fetched_output_dir"])
    report["artifacts"].update(
        write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_markdown,
            json_filename="tasai_adaptive_nice_sim_report.json",
            markdown_filename="tasai_adaptive_nice_sim_report.md",
        )
    )
    Path(report["artifacts"]["json"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _hpc_remote_runtime(arguments: dict[str, Any], *, policy_backend: str, count_backend: str) -> dict[str, Any]:
    runtime = {
        "container_runtime": str(arguments.get("hpc_container_runtime") or _DEFAULT_HPC_CONTAINER_RUNTIME),
        "submit_mode": "manual_review_required",
        "submit_poll_fetch": "fail_closed_in_local_runner",
    }
    if policy_backend in {"tacc_andie", "andie_tacc", "andie_bumps_mcmc"}:
        runtime["andie"] = {
            "local_workspace_reference": str(arguments.get("andie_container_workspace") or _DEFAULT_ANDIE_CONTAINER_WORKSPACE),
            "local_source_root_reference": str(arguments.get("andie_source_root") or _DEFAULT_ANDIE_SOURCE_ROOT),
            "local_container_image_reference": str(arguments.get("andie_container_image") or _DEFAULT_ANDIE_CONTAINER_IMAGE),
            "remote_workspace": str(arguments.get("andie_remote_workspace") or _DEFAULT_ANDIE_REMOTE_WORKSPACE),
            "remote_source_root": str(arguments.get("andie_remote_source_root") or _DEFAULT_ANDIE_REMOTE_SOURCE_ROOT),
            "remote_container_image": str(arguments.get("andie_remote_image") or _DEFAULT_ANDIE_REMOTE_IMAGE),
            "adapter_entrypoint": "python -m andie_tascopilot_runtime.adapter",
            "policy_module": "/workspace/src/andie_tascopilot_runtime/policy.py",
            "policy_input_contract": "AdaptivePolicyRequest JSON",
            "policy_output_contract": "AdaptivePolicyProposal-compatible JSON",
            "production_status": "container_boundary_ready; production ANDiE inference must be script/module code inside the external runtime; notebooks are reference only",
        }
    if count_backend in _HPC_COUNT_BACKENDS:
        runtime["count_model"] = {
            "backend": count_backend,
            "production_status": "remote count-model hook declared; pySpinW execution remains external to tascopilot",
        }
    return runtime


def _filter_candidates_by_software_limits(
    candidates: list[dict[str, Any]],
    software_limits: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in candidates:
        violations = []
        for node, value in (item.get("nodes") or {}).items():
            limit = software_limits.get(node)
            if limit and not (limit["min"] <= float(value) <= limit["max"]):
                violations.append({"node": node, "value": float(value), "min": limit["min"], "max": limit["max"]})
        if violations:
            rejected_item = dict(item)
            rejected_item["rejected_reason"] = "software_limit_exceeded"
            rejected_item["limit_violations"] = violations
            rejected.append(rejected_item)
        else:
            accepted.append(item)
    return accepted, rejected


def _write_hpc_handoff_scripts(hpc_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    run_script = hpc_dir / "run_andie_policy_job.sh"
    slurm_script = hpc_dir / "submit_andie_policy.slurm"
    run_script.write_text(_hpc_run_script(payload), encoding="utf-8")
    slurm_script.write_text(_hpc_slurm_script(payload), encoding="utf-8")
    run_script.chmod(0o755)
    slurm_script.chmod(0o755)
    return {"run_script": run_script, "slurm_script": slurm_script}


def _hpc_action_result(hpc_dir: Path, payload: dict[str, Any], arguments: dict[str, Any], *, hpc_action: str) -> dict[str, Any]:
    if hpc_action == "prepare":
        return {
            "status": "hpc_handoff_ready",
            "action_status": "payload_written",
            "submission_status": "not_submitted",
        }
    hpc_login = str(arguments.get("hpc_login") or "").strip()
    remote_dir = str(arguments.get("hpc_remote_handoff_dir") or "").strip()
    remote_output_dir = str(arguments.get("hpc_remote_output_dir") or "").strip()
    if not hpc_login:
        return _hpc_action_unavailable(hpc_action, "missing_hpc_login")
    if not remote_dir:
        return _hpc_action_unavailable(hpc_action, "missing_hpc_remote_handoff_dir")
    if "$" in remote_dir:
        return _hpc_action_unavailable(hpc_action, "hpc_remote_handoff_dir_must_be_expanded_absolute_path")

    ssh_executable = str(arguments.get("hpc_ssh_executable") or "ssh")
    scp_executable = str(arguments.get("hpc_scp_executable") or "scp")
    controlpath = str(payload.get("controlpath") or "~/.ssh/cm-%r@%h:%p")

    if hpc_action == "submit":
        mkdir = _run_hpc_command(
            _ssh_argv(ssh_executable, hpc_login, controlpath)
            + [f"mkdir -p {shlex.quote(remote_dir)}"],
            action="mkdir",
        )
        if mkdir["status"] != "success":
            return _hpc_action_failure(hpc_action, mkdir)
        copy = _run_hpc_command(
            _scp_argv(scp_executable, hpc_login, controlpath, hpc_dir, remote_dir),
            action="copy_handoff",
        )
        if copy["status"] != "success":
            return _hpc_action_failure(hpc_action, copy)
        submit_command = str(arguments.get("hpc_submit_command") or "sbatch submit_andie_policy.slurm")
        submit = _run_hpc_command(
            _ssh_argv(ssh_executable, hpc_login, controlpath)
            + [f"cd {shlex.quote(remote_dir)} && {submit_command}"],
            action="submit",
        )
        if submit["status"] != "success":
            return _hpc_action_failure(hpc_action, submit)
        job_id = _parse_slurm_job_id(submit.get("stdout", ""))
        return {
            "status": "hpc_submit_recorded",
            "action_status": "submitted" if job_id else "submitted_unparsed_job_id",
            "submission_status": "submitted",
            "remote_dir": remote_dir,
            "job_id": job_id,
            "commands": [mkdir, copy, submit],
            "note": "Submission used BatchMode SSH and will fail rather than prompt for MFA.",
        }

    if hpc_action == "poll":
        job_id = str(arguments.get("hpc_job_id") or "").strip()
        if not job_id:
            return _hpc_action_unavailable(hpc_action, "missing_hpc_job_id")
        poll_command = str(arguments.get("hpc_poll_command") or f"squeue -j {shlex.quote(job_id)} -h -o '%i %T %M %R' || sacct -j {shlex.quote(job_id)} --format=JobID,State,Elapsed -n")
        poll = _run_hpc_command(_ssh_argv(ssh_executable, hpc_login, controlpath) + [poll_command], action="poll")
        if poll["status"] != "success":
            return _hpc_action_failure(hpc_action, poll)
        return {
            "status": "hpc_poll_recorded",
            "action_status": "poll_completed",
            "submission_status": "unknown_remote_scheduler_state",
            "remote_dir": remote_dir,
            "job_id": job_id,
            "scheduler_output": poll.get("stdout", ""),
            "commands": [poll],
        }

    if hpc_action == "fetch":
        fetched_dir = hpc_dir.parent / "fetched_remote_output"
        fetched_dir.mkdir(parents=True, exist_ok=True)
        source_dir = remote_output_dir or f"{remote_dir}/remote_job_output"
        fetch = _run_hpc_command(
            _scp_fetch_argv(scp_executable, hpc_login, controlpath, source_dir, fetched_dir),
            action="fetch",
        )
        if fetch["status"] != "success":
            return _hpc_action_failure(hpc_action, fetch)
        proposal = _load_optional_json(fetched_dir / "policy_proposal.json")
        count_result = _load_optional_json(fetched_dir / "count_result.json")
        return {
            "status": "hpc_fetch_recorded",
            "action_status": "fetch_completed",
            "submission_status": "fetched",
            "remote_dir": remote_dir,
            "remote_output_dir": source_dir,
            "fetched_output_dir": str(fetched_dir),
            "policy_proposal": proposal,
            "count_result": count_result,
            "commands": [fetch],
        }

    return _hpc_action_unavailable(hpc_action, f"unsupported_hpc_action:{hpc_action}")


def _hpc_action_unavailable(action: str, reason: str) -> dict[str, Any]:
    return {
        "status": "hpc_action_unavailable",
        "action_status": "not_executed_by_local_runner",
        "submission_status": "not_submitted",
        "reason": reason,
        "note": "TACC submit/poll/fetch requires explicit SSH target fields and uses BatchMode to avoid MFA prompts.",
    }


def _hpc_action_failure(action: str, command_result: dict[str, Any]) -> dict[str, Any]:
    status = "hpc_action_mfa_or_auth_blocked" if command_result.get("blocked_reason") else "hpc_action_failed"
    return {
        "status": status,
        "action_status": f"{action}_failed",
        "submission_status": "not_submitted",
        "failed_command": command_result,
    }


def _hpc_limitations(hpc_action: str) -> list[str]:
    remote_action = (
        "No remote job was submitted, polled, or fetched by this local workflow."
        if hpc_action == "prepare"
        else f"Remote hpc-action={hpc_action!r} was attempted through BatchMode SSH; inspect hpc_handoff.action_result for scheduler state and outputs."
    )
    return [
        remote_action,
        "No pySpinW or tasAI MCMC/bumps count-model calculation ran unless the remote job set PYSPINW_COUNT_COMMAND.",
        "The handoff is a typed payload for a remote adapter, not a live instrument action.",
        "The generated TACC scripts are editable deployment scaffolds; review paths, modules, account, queue, and walltime before manual submission.",
    ]


def _ssh_argv(ssh_executable: str, hpc_login: str, controlpath: str) -> list[str]:
    return [
        ssh_executable,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"ControlPath={controlpath}",
        hpc_login,
    ]


def _scp_argv(scp_executable: str, hpc_login: str, controlpath: str, hpc_dir: Path, remote_dir: str) -> list[str]:
    return [
        scp_executable,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"ControlPath={controlpath}",
        "-r",
        f"{hpc_dir}/.",
        f"{hpc_login}:{remote_dir}/",
    ]


def _scp_fetch_argv(scp_executable: str, hpc_login: str, controlpath: str, remote_output_dir: str, fetched_dir: Path) -> list[str]:
    return [
        scp_executable,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"ControlPath={controlpath}",
        "-r",
        f"{hpc_login}:{remote_output_dir}",
        str(fetched_dir),
    ]


def _run_hpc_command(argv: list[str], *, action: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError as exc:
        return {"status": "failed", "action": action, "argv": _redact_argv(argv), "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"status": "failed", "action": action, "argv": _redact_argv(argv), "error": "timeout", "stdout": exc.stdout or "", "stderr": exc.stderr or ""}
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result = {
        "status": "success" if completed.returncode == 0 else "failed",
        "action": action,
        "argv": _redact_argv(argv),
        "returncode": completed.returncode,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }
    blocked = _mfa_or_interactive_auth_block(stdout + "\n" + stderr)
    if blocked:
        result["blocked_reason"] = blocked
    return result


def _mfa_or_interactive_auth_block(text: str) -> str | None:
    lowered = text.lower()
    if "verification code" in lowered or "multi-factor" in lowered or "mfa" in lowered:
        return "mfa_prompt_detected"
    if "password" in lowered or "passcode" in lowered or "permission denied" in lowered:
        return "interactive_auth_or_permission_denied"
    return None


def _parse_slurm_job_id(stdout: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if match:
        return match.group(1)
    stripped = stdout.strip()
    return stripped if stripped.isdigit() else None


def _redact_argv(argv: list[str]) -> list[str]:
    return [str(item) for item in argv]


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        matches = sorted(path.parent.rglob(path.name)) if path.parent.exists() else []
        if matches:
            path = matches[0]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _hpc_components(policy_backend: str, count_backend: str) -> list[str]:
    components = []
    if policy_backend in {"tacc_andie", "andie_tacc", "andie_bumps_mcmc"}:
        components.append("ANDiE model-discrimination/MCMC policy")
    elif policy_backend in _HPC_POLICY_BACKENDS:
        components.append("tasAI policy inference/MCMC/bumps")
    if count_backend in _HPC_COUNT_BACKENDS:
        components.append("SpinW/pySpinW count model")
    return components


def _hpc_handoff_readme(payload: dict[str, Any]) -> str:
    components = ", ".join(payload["expected_remote_components"]) or "none"
    remote_runtime = payload.get("remote_runtime") or {}
    andie = remote_runtime.get("andie") or {}
    return "\n".join(
        [
            "# tasAI/NICE TACC Handoff",
            "",
            "This directory contains a typed JSON payload and reviewed script templates for remote tasAI/ANDiE/pySpinW compute.",
            "The local workflow does not submit a job and it does not contact a live instrument.",
            "",
            f"- System: `{payload['hpc_system']}`",
            f"- Requested action: `{payload['hpc_action']}`",
            f"- Work directory: `{payload['hpc_work_dir']}`",
            f"- SSH ControlPath hint: `{payload['controlpath']}`",
            f"- Remote components: `{components}`",
            f"- Container runtime: `{remote_runtime.get('container_runtime', 'unspecified')}`",
            f"- Remote ANDiE image: `{andie.get('remote_container_image', 'not requested')}`",
            f"- Remote ANDiE workspace: `{andie.get('remote_workspace', 'not requested')}`",
            "",
            "Generated files:",
            "",
            "- `tasai_adaptive_hpc_payload.json`: canonical handoff payload.",
            "- `run_andie_policy_job.sh`: reviewed remote policy execution scaffold.",
            "- `submit_andie_policy.slurm`: editable SLURM wrapper for manual submission.",
            "",
            "Review account, queue, walltime, image paths, and module names before manual submission.",
            "Use $WORK for shared data/runtime staging. Use $SCRATCH only for temporary job products.",
            "Stop if SSH prompts for MFA. NICE simulator execution and live-instrument control remain separate.",
        ]
    )


def _hpc_run_script(payload: dict[str, Any]) -> str:
    remote_runtime = payload.get("remote_runtime") or {}
    andie = remote_runtime.get("andie") or {}
    container_runtime = str(remote_runtime.get("container_runtime") or _DEFAULT_HPC_CONTAINER_RUNTIME)
    image = str(andie.get("remote_container_image") or _DEFAULT_ANDIE_REMOTE_IMAGE)
    workspace = str(andie.get("remote_workspace") or _DEFAULT_ANDIE_REMOTE_WORKSPACE)
    source_root = str(andie.get("remote_source_root") or _DEFAULT_ANDIE_REMOTE_SOURCE_ROOT)
    return f"""#!/usr/bin/env bash
set -euo pipefail

PAYLOAD="${{1:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/tasai_adaptive_hpc_payload.json}}"
JOB_DIR="${{2:-$(cd "$(dirname "$PAYLOAD")" && pwd)/remote_job_output}}"
mkdir -p "$JOB_DIR"
JOB_DIR="$(cd "$JOB_DIR" && pwd)"

CONTAINER_RUNTIME="${{CONTAINER_RUNTIME:-{container_runtime}}}"
ANDIE_IMAGE="${{ANDIE_IMAGE:-{image}}}"
ANDIE_WORKSPACE="${{ANDIE_WORKSPACE:-{workspace}}}"
ANDIE_SOURCE_ROOT="${{ANDIE_SOURCE_ROOT:-{source_root}}}"
POLICY_REQUEST="$JOB_DIR/policy_request.json"
POLICY_PROPOSAL="$JOB_DIR/policy_proposal.json"
COUNT_REQUEST="$JOB_DIR/count_request.json"
COUNT_RESULT="$JOB_DIR/count_result.json"
JOB_SUMMARY="$JOB_DIR/andie_policy_job_summary.json"

python - "$PAYLOAD" "$POLICY_REQUEST" <<'PY'
import json
import sys
from pathlib import Path

payload_path = Path(sys.argv[1])
request_path = Path(sys.argv[2])
payload = json.loads(payload_path.read_text(encoding="utf-8"))
candidates = []
for item in payload.get("candidate_points", []):
    if not isinstance(item, dict):
        continue
    metadata = {{str(k): v for k, v in item.items() if k not in {{"point_id", "nodes", "count_time", "metadata"}}}}
    metadata.update(item.get("metadata") or {{}})
    candidates.append({{
        "point_id": str(item.get("point_id", f"candidate-{{len(candidates):03d}}")),
        "nodes": item.get("nodes") or {{}},
        "count_time": item.get("count_time"),
        "metadata": metadata,
    }})
request = {{
    "step": 0,
    "candidates": candidates,
    "observations": [],
    "model_spec": payload.get("model_spec") or {{}},
    "software_limits": (payload.get("software_limits") or {{}}).get("limits") or {{}},
    "policy_context": {{
        "policy_backend": payload.get("policy_backend"),
        "count_backend": payload.get("count_backend"),
        "handoff_payload": str(payload_path),
        "remote_runtime": payload.get("remote_runtime") or {{}},
        "required_output": {{
            "point_id": "existing candidate id or new id",
            "nodes": {{"Q.H": "float", "Q.K": "float", "Q.L": "float", "et.deltaE": "float"}},
            "score": "float",
            "reason": "short provenance-friendly reason",
            "count_time": "optional float seconds",
        }},
    }},
}}
request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
PY

case "$CONTAINER_RUNTIME" in
  apptainer|singularity)
    "$CONTAINER_RUNTIME" exec \\
      --bind "$ANDIE_WORKSPACE:/workspace:ro" \\
      --bind "$ANDIE_SOURCE_ROOT:/ANDiE-v1_0:ro" \\
      --bind "$JOB_DIR:/io" \\
      "$ANDIE_IMAGE" \\
      python -m andie_tascopilot_runtime.adapter \\
        --input /io/policy_request.json \\
        --output /io/policy_proposal.json \\
        --andie-root /ANDiE-v1_0
    ;;
  docker)
    docker run --rm \\
      -v "$ANDIE_WORKSPACE:/workspace:ro" \\
      -v "$ANDIE_SOURCE_ROOT:/ANDiE-v1_0:ro" \\
      -v "$JOB_DIR:/io" \\
      "$ANDIE_IMAGE" \\
        --input /io/policy_request.json \\
        --output /io/policy_proposal.json \\
        --andie-root /ANDiE-v1_0
    ;;
  *)
    echo "Unsupported CONTAINER_RUNTIME=$CONTAINER_RUNTIME" >&2
    exit 2
    ;;
esac

python - "$PAYLOAD" "$POLICY_REQUEST" "$POLICY_PROPOSAL" "$JOB_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

payload_path, request_path, proposal_path, summary_path = [Path(arg) for arg in sys.argv[1:]]
summary = {{
    "status": "completed" if proposal_path.exists() else "missing_policy_proposal",
    "handoff_payload": str(payload_path),
    "policy_request": str(request_path),
    "policy_proposal": str(proposal_path),
    "safety_boundary": "simulation_only_remote_planning",
    "notes": [
        "This job evaluates a policy proposal only.",
        "It does not contact NICE, write instrument files, or control hardware.",
    ],
}}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
if not proposal_path.exists():
    raise SystemExit(3)
PY

echo "ANDiE policy proposal: $POLICY_PROPOSAL"

if [ -n "${{PYSPINW_COUNT_COMMAND:-}}" ]; then
  python - "$PAYLOAD" "$POLICY_PROPOSAL" "$COUNT_REQUEST" <<'PY'
import json
import sys
from pathlib import Path

payload_path, proposal_path, request_path = [Path(arg) for arg in sys.argv[1:]]
payload = json.loads(payload_path.read_text(encoding="utf-8"))
proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
request = {{
    "point_id": proposal.get("point_id"),
    "nodes": proposal.get("nodes") or {{}},
    "count_time": proposal.get("count_time"),
    "model_spec": payload.get("model_spec") or {{}},
    "count_context": {{
        "count_backend": payload.get("count_backend"),
        "handoff_payload": str(payload_path),
        "policy_proposal": str(proposal_path),
        "remote_runtime": payload.get("remote_runtime") or {{}},
    }},
}}
request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
PY
  $PYSPINW_COUNT_COMMAND "$COUNT_REQUEST" "$COUNT_RESULT"
  echo "Count result: $COUNT_RESULT"
else
  echo "PYSPINW_COUNT_COMMAND not set; skipping remote count-model execution."
fi

echo "Job summary: $JOB_SUMMARY"
"""


def _hpc_slurm_script(payload: dict[str, Any]) -> str:
    system = str(payload.get("hpc_system") or "vista")
    work_dir = str(payload.get("hpc_work_dir") or f"{_DEFAULT_REMOTE_ROOT}/tasai_adaptive_nice")
    account = str(payload.get("hpc_account") or "").strip()
    partition = str(payload.get("hpc_partition") or _default_hpc_partition(system)).strip()
    account_line = f"#SBATCH -A {account}\n" if account else ""
    partition_line = f"#SBATCH -p {partition}\n" if partition else ""
    return f"""#!/usr/bin/env bash
#SBATCH -J tasai-andie-policy
#SBATCH -o tasai-andie-policy.%j.out
#SBATCH -e tasai-andie-policy.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
{partition_line}{account_line}
set -eo pipefail

echo "Review this SLURM file before submission. Target system: {system}"
set +u
module load tacc-apptainer || module load apptainer || true
set -u

SCRIPT_DIR="${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)}}"
PAYLOAD="$SCRIPT_DIR/tasai_adaptive_hpc_payload.json"
JOB_DIR="{work_dir}/remote_job_output/${{SLURM_JOB_ID:-manual}}"
mkdir -p "$JOB_DIR"

"$SCRIPT_DIR/run_andie_policy_job.sh" "$PAYLOAD" "$JOB_DIR"
"""


def _default_hpc_partition(system: str) -> str:
    lowered = system.lower()
    if "stampede" in lowered:
        return "skx-dev"
    if "vista" in lowered:
        return "gh-dev"
    return ""


def _safe_prefix(value: str) -> str:
    """NICE file prefixes travel through shell and filesystem; keep them plain."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in value)
    return cleaned.strip("_") or "tasai_adaptive"


def _point_server_script() -> Path:
    """scripts/nice_point_server.py, whether or not cwd is the repo root."""
    local = Path("scripts/nice_point_server.py")
    if local.exists():
        return local.resolve()
    return Path(__file__).resolve().parents[5] / "scripts" / "nice_point_server.py"


def _session_step(
    session: Any,
    step: int,
    proposal: dict[str, Any],
    count_time: float,
    step_dir: Path,
) -> dict[str, Any]:
    """Run one point on the open measurement, shaped like a nice_bt7_replay result.

    The row is also written to the step's own trace file so the observation path
    (_counts_from_nice_trace) is identical for both executors.
    """
    from .nice_point_session import NiceSessionError

    step_dir.mkdir(parents=True, exist_ok=True)
    step_trace = step_dir / "nice_replay_trace.jsonl"
    try:
        row = session.execute(step, proposal["nodes"], count_time)
    except NiceSessionError as exc:
        return {"status": "error", "execution": {"status": "nice_remote_session_failed", "error": str(exc)}, "artifacts": {}}

    step_trace.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    failed = row.get("status") == "nice_error"
    return {
        "status": "error" if failed else "success",
        "execution": {
            "status": "nice_remote_session_failed" if failed else "nice_remote_session_executed",
            "trace_path": str(step_trace),
            "error": row.get("error"),
            "entry": row.get("entry"),
        },
        "artifacts": {"trace": str(step_trace)},
    }


def _position_error(
    requested: dict[str, Any],
    achieved: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Nodes that did not arrive, as {node: {requested, achieved, delta}}.

    Tolerance is deliberately loose relative to the ~1e-4 agreement a healthy
    BT7 move actually shows, so this flags a motor that stopped short, not
    ordinary encoder resolution.
    """
    if not achieved:
        return None
    bad: dict[str, Any] = {}
    for node, want in (requested or {}).items():
        got = achieved.get(node)
        try:
            want_f, got_f = float(want), float(got)
        except (TypeError, ValueError):
            # Includes the "<unreadable: ...>" sentinel the replay script
            # writes; not knowing where we are is itself worth reporting.
            bad[node] = {"requested": want, "achieved": got}
            continue
        tol = 2e-3 + 1e-3 * abs(want_f)
        if abs(got_f - want_f) > tol:
            bad[node] = {"requested": want_f, "achieved": got_f,
                         "delta": got_f - want_f, "tolerance": tol}
    return bad or None


def _counts_from_nice_trace(trace_path: str | None) -> dict[str, Any] | None:
    """Last point of a NICE replay trace: the live script writes `counts` (from
    `<countAgainstDetector>.counts`) and `counter_liveTime`; the mock executor
    writes `counter_liveROI`."""
    if not trace_path:
        return None
    path = Path(str(trace_path))
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except ValueError:
                continue
    if not last:
        return None
    for key, source in (("counts", "nice_detector_counts"), ("counter_liveROI", "nice_live_roi")):
        raw = last.get(key)
        if raw is None:
            continue
        # A detector node returns one value per segment (BT7 singleDetector is
        # [SD01, SD02, SD03]) and the scalar the instrument itself reports as
        # "SD Total" is their sum. float() on that list raises TypeError, and
        # swallowing it drops every live observation, which leaves the policy
        # with an empty history and pinned in its first phase for the whole
        # run -- with no error anywhere. Report it instead.
        try:
            if isinstance(raw, (list, tuple)):
                segments = [float(v) for v in raw]
                total = float(sum(segments))
            else:
                segments, total = None, float(raw)
        except (TypeError, ValueError) as exc:
            return {"error": f"{key}={raw!r} is not a count ({type(exc).__name__}: {exc})"}
        measured: dict[str, Any] = {
            "counts": total,
            "live_time": last.get("counter_liveTime"),
            "source": source,
        }
        for field in ("achieved_nodes", "achieved_angles"):
            if last.get(field):
                measured[field] = last[field]
        if segments is not None:
            measured["segments"] = segments
        if last.get("counter_liveMonitor") is not None:
            measured["monitor"] = last["counter_liveMonitor"]
        roi = last.get("counter_liveROI")
        if key == "counts" and roi is not None:
            # Disagreement means the detector grouping is not what we think it
            # is, which would bias every rate in the run with nothing else to
            # catch it.
            try:
                if abs(float(roi) - total) > 0.5:
                    measured["roi_mismatch"] = {"counter_liveROI": float(roi), "segment_sum": total}
            except (TypeError, ValueError):
                pass
        return measured
    return None


def _nodes_from_qe(item: dict[str, Any]) -> dict[str, float]:
    nodes = {}
    aliases = {"h": "Q.H", "k": "Q.K", "l": "Q.L", "e": "et.deltaE", "energy_transfer": "et.deltaE"}
    for key, node in aliases.items():
        if key in item:
            nodes[node] = float(item[key])
    return nodes


def _load_model_spec(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("model_spec")
    if not raw and arguments.get("model_spec_file"):
        raw = json.loads(Path(arguments["model_spec_file"]).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {
        "model": "fixture_spinwave_gaussian",
        "background": 20.0,
        "peaks": [{"center": {"Q.H": 0.0, "Q.K": 0.0, "Q.L": 0.0, "et.deltaE": 0.0}, "amplitude": 1000.0, "width": 0.12}],
    }


def _prepare_count_model(arguments: dict[str, Any], model_spec: dict[str, Any], count_backend: str) -> tuple[dict[str, Any], str | None]:
    if count_backend == "fixture_spinw":
        return {"backend": "fixture_spinw"}, None
    if count_backend == "external_pyspinw_adapter":
        adapter_path = arguments.get("pyspinw_count_adapter") or arguments.get("count_adapter")
        if not adapter_path:
            return {}, "external_pyspinw_adapter requires pyspinw_count_adapter pointing to importable count code."
        resolved = Path(str(adapter_path)).expanduser().resolve()
        if not resolved.exists():
            return {}, f"external_pyspinw_adapter path does not exist: {resolved}"
        try:
            function = _import_adapter_function(resolved, "simulate_counts")
        except Exception as exc:  # pragma: no cover - exact import exceptions vary by external adapter
            return {}, f"external_pyspinw_adapter could not import simulate_counts from {resolved}: {exc}"
        return {"backend": "external_pyspinw_adapter", "adapter_path": str(resolved), "function": function}, None
    if count_backend != "tasai_analytic":
        return {}, f"Count backend {count_backend!r} is unsupported."
    tasai_path = arguments.get("tasai_path") or model_spec.get("tasai_path")
    if not tasai_path:
        return {}, "tasai_analytic count backend requires tasai_path pointing to an external tasAI checkout."
    resolved = Path(str(tasai_path)).expanduser().resolve()
    if not resolved.exists():
        return {}, f"tasai_analytic tasai_path does not exist: {resolved}"
    analytic_model = str(arguments.get("analytic_model") or model_spec.get("analytic_model") or "ferromagnetic_magnon")
    if analytic_model not in TASAI_ANALYTIC_MODELS:
        return {}, f"tasai_analytic model {analytic_model!r} is not allowed. Allowed models: {', '.join(sorted(TASAI_ANALYTIC_MODELS))}"
    try:
        function = _import_tasai_analytic_function(resolved, analytic_model)
    except Exception as exc:  # pragma: no cover - exact import exceptions vary by external checkout
        return {}, f"tasai_analytic could not import tasai.instrument.simulator.{analytic_model}: {exc}"
    parameters = arguments.get("analytic_parameters") or model_spec.get("analytic_parameters") or {}
    if isinstance(parameters, str) and parameters.strip():
        parameters = json.loads(parameters)
    return {
        "backend": "tasai_analytic",
        "tasai_path": str(resolved),
        "source_module": str(resolved / "tasai" / "instrument" / "simulator.py"),
        "analytic_model": analytic_model,
        "function": function,
        "parameters": dict(parameters or {}),
        "background": float(arguments.get("background") or model_spec.get("background", 20.0)),
        "count_scale": float(arguments.get("count_scale") or model_spec.get("count_scale", 1000.0)),
    }, None


def _prepare_policy_model(arguments: dict[str, Any], policy_backend: str) -> tuple[dict[str, Any], str | None]:
    if policy_backend == "fixture_tasai":
        return {"backend": "fixture_tasai"}, None
    if policy_backend == "andie_container":
        workspace = Path(str(arguments.get("andie_container_workspace") or "local_runs/external/andie-container")).expanduser().resolve()
        source_root = Path(str(arguments.get("andie_source_root") or "local_runs/external/ANDiE-v1_0")).expanduser().resolve()
        adapter_script = workspace / "src" / "andie_tascopilot_runtime" / "adapter.py"
        if not adapter_script.exists():
            legacy_adapter_script = workspace / "src" / "andie_container_adapter.py"
            if legacy_adapter_script.exists():
                adapter_script = legacy_adapter_script
        if not adapter_script.exists():
            return {}, f"andie_container workspace is missing adapter module: {adapter_script}"
        if not source_root.exists():
            return {}, f"andie_container source root does not exist: {source_root}"
        source_scripts = list(source_root.rglob("*.py"))
        reference_notebooks = list(source_root.glob("*.ipynb"))
        if not source_scripts:
            return {}, f"andie_container source root has no ANDiE script/module files: {source_root}. Notebooks are reference material only."
        return {
            "backend": "andie_container",
            "workspace": str(workspace),
            "source_root": str(source_root),
            "container_image": str(arguments.get("andie_container_image") or "tascopilot-andie:latest"),
            "docker_executable": str(arguments.get("docker_executable") or "docker"),
            "output_dir": str(Path(arguments["output_dir"]).expanduser()),
            "adapter_script": str(adapter_script),
            "adapter_path": str(adapter_script),
            "source_script_count": len(source_scripts),
            "reference_notebook_count": len(reference_notebooks),
        }, None
    if policy_backend not in {"external_tasai_adapter", "external_andie_adapter"}:
        return {}, f"Policy backend {policy_backend!r} is unsupported."
    adapter_path = (
        arguments.get("andie_policy_adapter")
        if policy_backend == "external_andie_adapter"
        else arguments.get("tasai_policy_adapter")
    )
    adapter_path = adapter_path or arguments.get("policy_adapter") or arguments.get("tasai_adapter")
    if not adapter_path:
        expected = "andie_policy_adapter" if policy_backend == "external_andie_adapter" else "tasai_policy_adapter"
        return {}, f"{policy_backend} requires {expected} pointing to importable external policy adapter code."
    resolved = Path(str(adapter_path)).expanduser().resolve()
    if not resolved.exists():
        return {}, f"{policy_backend} path does not exist: {resolved}"
    try:
        function = _import_adapter_function(resolved, "propose_next_point")
    except Exception as exc:  # pragma: no cover - exact import exceptions vary by external adapter
        return {}, f"{policy_backend} could not import propose_next_point from {resolved}: {exc}"
    return {"backend": policy_backend, "adapter_path": str(resolved), "function": function}, None


def _import_adapter_function(adapter_path: Path, function_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name = f"_tas_copilot_external_adapter_{abs(hash(str(adapter_path)))}"
    spec = importlib_util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {adapter_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{function_name} is not callable")
    return function


def _import_tasai_analytic_function(tasai_path: Path, analytic_model: str) -> Callable[..., float]:
    simulator_path = tasai_path / "tasai" / "instrument" / "simulator.py"
    if not simulator_path.exists():
        raise FileNotFoundError(simulator_path)
    module_name = "tasai.instrument.simulator"
    spec = importlib_util.spec_from_file_location(module_name, simulator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {simulator_path}")
    module = importlib_util.module_from_spec(spec)
    old_path = list(sys.path)
    old_modules = {name: sys.modules.get(name) for name in ["tasai", "tasai.instrument", "tasai.instrument.base", "tasai.instrument.simulator"]}
    tasai_pkg = types.ModuleType("tasai")
    tasai_pkg.__path__ = [str(tasai_path / "tasai")]  # type: ignore[attr-defined]
    instrument_pkg = types.ModuleType("tasai.instrument")
    instrument_pkg.__path__ = [str(tasai_path / "tasai" / "instrument")]  # type: ignore[attr-defined]
    try:
        sys.path.insert(0, str(tasai_path))
        sys.modules["tasai"] = tasai_pkg
        sys.modules["tasai.instrument"] = instrument_pkg
        sys.modules["tasai.instrument.simulator"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    function = getattr(module, analytic_model)
    if not callable(function):
        raise TypeError(f"{analytic_model} is not callable")
    return function


def _load_human_suggestions(path: Any) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    suggestions = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        suggestions[int(row.get("step", 0))] = row
    return suggestions


def _load_software_limits(arguments: dict[str, Any]) -> tuple[dict[str, dict[str, float]], str]:
    raw = arguments.get("software_limits")
    source = "fixture_default_limits"
    if not raw and arguments.get("software_limits_file"):
        raw = json.loads(Path(arguments["software_limits_file"]).expanduser().read_text(encoding="utf-8"))
        source = str(arguments["software_limits_file"])
    elif isinstance(raw, str) and raw.strip():
        raw = json.loads(raw)
        source = "argument"
    limits = raw or DEFAULT_SOFTWARE_LIMITS
    return {str(key): {"min": float(value["min"]), "max": float(value["max"])} for key, value in limits.items()}, source


def _choose_next_point(
    candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    model_spec: dict[str, Any],
    software_limits: dict[str, dict[str, float]],
    count_backend: str,
    count_model: dict[str, Any],
    policy_backend: str,
    policy_model: dict[str, Any],
    *,
    step: int,
    action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    observed = {item["point_id"] for item in observations}
    if action and action.get("action") == "suggest_point":
        point = {"point_id": str(action.get("point_id") or f"human-suggested-{step:03d}"), "nodes": {str(k): float(v) for k, v in (action.get("nodes") or _nodes_from_qe(action)).items()}}
        return _score_or_reject(point, model_spec, software_limits, count_backend, count_model, "human_suggested_point")
    remaining = [item for item in candidates if item["point_id"] not in observed]
    if policy_backend in {"external_tasai_adapter", "external_andie_adapter", "andie_container"}:
        proposal = _external_tasai_policy_proposal(
            remaining,
            observations,
            model_spec,
            software_limits,
            count_backend,
            count_model,
            policy_model,
            step=step,
        )
        if proposal is not None:
            return proposal
    scored = [_score_or_reject(item, model_spec, software_limits, count_backend, count_model, "fixture_tasai_expected_information_gain") for item in remaining]
    allowed = [item for item in scored if not item.get("rejected_reason")]
    if not allowed:
        return None
    return sorted(allowed, key=lambda item: item["policy_score"], reverse=True)[0]


def _score_or_reject(
    point: dict[str, Any],
    model_spec: dict[str, Any],
    software_limits: dict[str, dict[str, float]],
    count_backend: str,
    count_model: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    point = dict(point)
    for node, value in (point.get("nodes") or {}).items():
        limit = software_limits.get(node)
        if limit and not (limit["min"] <= float(value) <= limit["max"]):
            point["rejected_reason"] = f"software_limit_exceeded:{node}"
            point["policy_score"] = 0.0
            point["policy_reason"] = "rejected by software operating limits"
            return point
    overlay = _count_overlay(point, model_spec, count_time=1.0, step=-1, count_backend=count_backend, count_model=count_model)
    point["policy_score"] = float(overlay["expected_counts"])
    point["policy_reason"] = reason
    return point


def _external_tasai_policy_proposal(
    remaining: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    model_spec: dict[str, Any],
    software_limits: dict[str, dict[str, float]],
    count_backend: str,
    count_model: dict[str, Any],
    policy_model: dict[str, Any],
    *,
    step: int,
) -> dict[str, Any] | None:
    if not remaining:
        return None
    request = AdaptivePolicyRequest(
        step=step,
        candidates=[_candidate_contract(item) for item in remaining],
        observations=[_observation_contract(item) for item in observations],
        model_spec=model_spec,
        software_limits=software_limits,
        policy_context={
            "policy_backend": policy_model["backend"],
            "adapter_path": policy_model.get("adapter_path"),
            "required_output": {
                "point_id": "existing candidate id or new id",
                "nodes": {"Q.H": "float", "Q.K": "float", "Q.L": "float", "et.deltaE": "float"},
                "score": "float",
                "reason": "short provenance-friendly reason",
                "count_time": "optional float seconds",
            },
        },
    )
    payload = request.model_dump(mode="json")
    if policy_model["backend"] == "andie_container":
        raw = _run_andie_container_policy(payload, policy_model, step=step)
    else:
        raw = policy_model["function"](payload)
    if not isinstance(raw, dict):
        raise TypeError("external_tasai_adapter propose_next_point must return a dict")
    if raw.get("point_id") and not raw.get("nodes"):
        match = next((item for item in remaining if item["point_id"] == str(raw["point_id"])), None)
        if match:
            raw = {**match, **raw}
    nodes = raw.get("nodes") or _nodes_from_qe(raw)
    if not nodes:
        raise ValueError("external_tasai_adapter response must include nodes or h/k/l/e fields")
    typed = AdaptivePolicyProposal(
        point_id=str(raw.get("point_id") or f"external-tasai-{step:03d}"),
        nodes={str(k): float(v) for k, v in nodes.items()},
        score=float(raw.get("score", 0.0)),
        reason=str(raw.get("reason") or f"external_policy_adapter:{policy_model.get('adapter_path', policy_model['backend'])}"),
        count_time=float(raw["count_time"]) if raw.get("count_time") is not None else None,
        metadata=dict(raw.get("metadata") or {}),
    )
    proposal = {
        "point_id": typed.point_id,
        "nodes": typed.nodes,
    }
    scored = _score_or_reject(
        proposal,
        model_spec,
        software_limits,
        count_backend,
        count_model,
        typed.reason,
    )
    if not scored.get("rejected_reason"):
        scored["policy_score"] = float(typed.score)
    if typed.count_time is not None:
        scored["suggested_count_time"] = float(typed.count_time)
    scored["policy_metadata"] = typed.metadata
    scored["policy_adapter"] = policy_model.get("adapter_path")
    scored["policy_backend"] = policy_model["backend"]
    return scored


def _run_andie_container_policy(payload: dict[str, Any], policy_model: dict[str, Any], *, step: int) -> dict[str, Any]:
    output_dir = Path(str(policy_model["output_dir"])).expanduser()
    io_dir = output_dir / "andie_container_policy" / f"step_{step:03d}"
    io_dir.mkdir(parents=True, exist_ok=True)
    input_path = io_dir / "policy_request.json"
    output_path = io_dir / "policy_proposal.json"
    input_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    argv = [
        str(policy_model["docker_executable"]),
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{policy_model['workspace']}:/workspace:ro",
        "-v",
        f"{policy_model['source_root']}:/ANDiE-v1_0:ro",
        "-v",
        f"{io_dir.resolve()}:/io",
        str(policy_model["container_image"]),
        "--input",
        "/io/policy_request.json",
        "--output",
        "/io/policy_proposal.json",
        "--andie-root",
        "/ANDiE-v1_0",
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"andie_container runtime not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"andie_container policy step timed out: {(exc.stderr or '')[-1000:]}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "andie_container policy step failed: "
            f"returncode={completed.returncode}; stdout={completed.stdout[-1000:]}; stderr={completed.stderr[-2000:]}"
        )
    if not output_path.exists():
        raise RuntimeError(f"andie_container did not write proposal: {output_path}")
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw.setdefault("metadata", {})
        raw["metadata"].setdefault("adapter_path", str(policy_model["adapter_script"]))
        raw["metadata"].setdefault("container_image", str(policy_model["container_image"]))
        raw["metadata"].setdefault("policy_request", str(input_path))
        raw["metadata"].setdefault("policy_proposal", str(output_path))
    return raw


def _candidate_contract(item: dict[str, Any]) -> AdaptiveCandidatePoint:
    metadata = dict(item.get("metadata") or {})
    for key, value in item.items():
        if key not in {"point_id", "nodes", "count_time", "metadata"}:
            metadata.setdefault(str(key), value)
    return AdaptiveCandidatePoint(
        point_id=str(item["point_id"]),
        nodes={str(key): float(value) for key, value in (item.get("nodes") or {}).items()},
        count_time=float(item["count_time"]) if item.get("count_time") is not None else None,
        metadata=metadata,
    )


def _observation_contract(item: dict[str, Any]) -> AdaptiveObservation:
    overlay = item.get("simulation_count_overlay") or {}
    metadata = {
        "nice_execution_status": item.get("nice_execution_status"),
        "simulation_count_overlay": overlay,
        "policy_metadata": item.get("policy_metadata") or {},
    }
    return AdaptiveObservation(
        step=int(item["step"]),
        point_id=str(item["point_id"]),
        nodes={str(key): float(value) for key, value in (item.get("nodes") or {}).items()},
        count_time=float(item["count_time"]) if item.get("count_time") is not None else None,
        expected_counts=float(overlay["expected_counts"]) if overlay.get("expected_counts") is not None else None,
        observed_counts=float(item["observed_counts"]) if item.get("observed_counts") is not None else None,
        uncertainty=float(item["uncertainty"]) if item.get("uncertainty") is not None else (float(overlay["uncertainty"]) if overlay.get("uncertainty") is not None else None),
        policy_score=float(item["policy_score"]) if item.get("policy_score") is not None else None,
        policy_reason=str(item["policy_reason"]) if item.get("policy_reason") is not None else None,
        metadata=metadata,
    )


def _count_overlay(
    point: dict[str, Any],
    model_spec: dict[str, Any],
    *,
    count_time: float,
    step: int,
    count_backend: str,
    count_model: dict[str, Any],
) -> dict[str, Any]:
    if count_backend == "fixture_spinw":
        return _fixture_spinw_count_overlay(point, model_spec, count_time=count_time, step=step)
    if count_backend == "tasai_analytic":
        return _tasai_analytic_count_overlay(point, count_model, count_time=count_time, step=step)
    if count_backend == "external_pyspinw_adapter":
        return _external_pyspinw_count_overlay(point, model_spec, count_model, count_time=count_time, step=step)
    raise ValueError(f"Unsupported count backend {count_backend!r}")


def _fixture_spinw_count_overlay(point: dict[str, Any], model_spec: dict[str, Any], *, count_time: float, step: int) -> dict[str, Any]:
    nodes = point.get("nodes") or {}
    background = float(model_spec.get("background", 20.0))
    expected_rate = background
    for peak in model_spec.get("peaks") or []:
        center = peak.get("center") or {}
        width = float(peak.get("width") or 0.1)
        amplitude = float(peak.get("amplitude") or 100.0)
        dist2 = 0.0
        for node, center_value in center.items():
            if node in nodes:
                dist2 += ((float(nodes[node]) - float(center_value)) / width) ** 2
        expected_rate += amplitude * math.exp(-0.5 * dist2)
    return {
        "step": step,
        "point_id": point["point_id"],
        "backend": "fixture_spinw",
        "expected_rate_hz": expected_rate,
        "count_time": count_time,
        "expected_counts": expected_rate * count_time,
        "nodes": nodes,
    }


def _tasai_analytic_count_overlay(point: dict[str, Any], count_model: dict[str, Any], *, count_time: float, step: int) -> dict[str, Any]:
    nodes = point.get("nodes") or {}
    function = count_model["function"]
    h = float(nodes.get("Q.H", 0.0))
    k = float(nodes.get("Q.K", 0.0))
    l = float(nodes.get("Q.L", 0.0))
    energy = float(nodes.get("et.deltaE", 0.0))
    sqw = float(function(h, k, l, energy, **(count_model.get("parameters") or {})))
    background = float(count_model.get("background", 20.0))
    count_scale = float(count_model.get("count_scale", 1000.0))
    expected_rate = background + count_scale * max(0.0, sqw)
    return {
        "step": step,
        "point_id": point["point_id"],
        "backend": "tasai_analytic",
        "source_module": count_model["source_module"],
        "analytic_model": count_model["analytic_model"],
        "tasai_path": count_model["tasai_path"],
        "analytic_parameters": count_model.get("parameters") or {},
        "sqw": sqw,
        "expected_rate_hz": expected_rate,
        "count_time": count_time,
        "expected_counts": expected_rate * count_time,
        "nodes": nodes,
    }


def _external_pyspinw_count_overlay(
    point: dict[str, Any],
    model_spec: dict[str, Any],
    count_model: dict[str, Any],
    *,
    count_time: float,
    step: int,
) -> dict[str, Any]:
    request = AdaptiveCountRequest(
        point=_candidate_contract(point),
        model_spec=model_spec,
        count_time=count_time,
        count_context={
            "adapter_path": count_model["adapter_path"],
            "required_output": {
                "expected_counts": "float",
                "expected_rate_hz": "optional float",
                "metadata": "optional object",
            },
        },
    )
    payload = request.model_dump(mode="json")
    raw = count_model["function"](payload)
    if not isinstance(raw, dict):
        raise TypeError("external_pyspinw_adapter simulate_counts must return a dict")
    typed = AdaptiveCountResult.model_validate(raw)
    expected_counts = float(typed.expected_counts)
    expected_rate = float(typed.expected_rate_hz if typed.expected_rate_hz is not None else expected_counts / max(count_time, 1e-12))
    return {
        "step": step,
        "point_id": point["point_id"],
        "backend": "external_pyspinw_adapter",
        "adapter_path": count_model["adapter_path"],
        "expected_rate_hz": expected_rate,
        "count_time": count_time,
        "expected_counts": expected_counts,
        "nodes": point.get("nodes") or {},
        "metadata": typed.metadata,
    }


def _write_single_point_replay_plan(output_dir: Path, proposal: dict[str, Any], count_time: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "ready_for_simulation_adapter",
        "workflow_boundary": "simulation_only_training_harness",
        "recipes": [
            {
                "replay_recipe_id": f"tasai-{proposal['point_id']}",
                "source_title": "tasAI adaptive point",
                "translation_status": "reviewed_simulator_aliases",
                "recommended_replay_mode": "point_by_point_adaptive",
                "npts": 1,
                "counts": count_time,
                "points": [{"point_index": 0, "nodes": proposal["nodes"]}],
            }
        ],
    }
    path = output_dir / "nice_replay_training_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_adaptive_decision_review_template(
    path: Path,
    observations: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    workflow_context: dict[str, Any],
) -> int:
    rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for item in observations:
        base = {
            "object_type": "adaptive_measurement_decision",
            "object_id": f"{workflow_context['policy_backend']}:{item['point_id']}:step-{item['step']}",
            "type_specific_status": "proposed_measurement",
            "candidate_status": "executed_in_simulation",
            "verdict": "",
            "claim_state": "proposed",
            "reviewed_by": "",
            "reviewed_at": "",
            "created_at": created_at,
            "point_id": item["point_id"],
            "step": item["step"],
            "nodes": item.get("nodes") or {},
            "policy_backend": workflow_context["policy_backend"],
            "count_backend": workflow_context["count_backend"],
            "nice_executor": workflow_context["nice_executor"],
            "policy_score": item.get("policy_score"),
            "policy_reason": item.get("policy_reason"),
            "count_time": item.get("count_time"),
            "evidence_channels_relied_on": ["policy_score", "software_limits", "simulation_count_overlay"],
            "correction_status_snapshot": {
                "software_limits_source": workflow_context["software_limits_source"],
                "state_path": workflow_context["state_path"],
            },
            "review_question": "Was this adaptive point/count-time suggestion useful and scientifically appropriate for the simulated task?",
            "allowed_type_specific_statuses": ["useful_suggestion", "not_useful", "unsafe_or_invalid", "duplicate", "needs_more_context"],
        }
        base["label_id"] = _stable_review_id(base)
        rows.append(base)
    for item in rejected:
        base = {
            "object_type": "adaptive_measurement_decision",
            "object_id": f"{workflow_context['policy_backend']}:{item.get('point_id', 'unknown')}:rejected",
            "type_specific_status": "rejected_by_workflow",
            "candidate_status": "rejected_by_workflow",
            "verdict": "",
            "claim_state": "rejected",
            "reviewed_by": "",
            "reviewed_at": "",
            "created_at": created_at,
            "point_id": item.get("point_id"),
            "nodes": item.get("nodes") or {},
            "policy_backend": workflow_context["policy_backend"],
            "count_backend": workflow_context["count_backend"],
            "nice_executor": workflow_context["nice_executor"],
            "rejected_reason": item.get("rejected_reason"),
            "evidence_channels_relied_on": ["software_limits"],
            "correction_status_snapshot": {
                "software_limits_source": workflow_context["software_limits_source"],
                "state_path": workflow_context["state_path"],
            },
            "review_question": "Was this adaptive rejection appropriate?",
            "allowed_type_specific_statuses": ["correct_rejection", "incorrect_rejection", "needs_more_context"],
        }
        base["label_id"] = _stable_review_id(base)
        rows.append(base)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _stable_review_id(row: dict[str, Any]) -> str:
    key = {
        "object_type": row.get("object_type"),
        "object_id": row.get("object_id"),
        "candidate_status": row.get("candidate_status"),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"adaptive-review-{digest}"


def _missing(output_dir: Path, fields: list[str], reason: str) -> dict[str, Any]:
    report = {"workflow_name": "tasai-adaptive-nice-sim", "status": "missing_input", "missing_fields": fields, "reason": reason}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="tasai_adaptive_nice_sim_report.json", markdown_filename="tasai_adaptive_nice_sim_report.md")
    return report


def _unavailable(output_dir: Path, reason: str) -> dict[str, Any]:
    report = {"workflow_name": "tasai-adaptive-nice-sim", "status": "unavailable", "reason": reason}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="tasai_adaptive_nice_sim_report.json", markdown_filename="tasai_adaptive_nice_sim_report.md")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# tasAI Adaptive NICE Simulation",
        "",
        f"Status: `{report.get('status')}`",
        f"Boundary: `{report.get('safety_boundary', 'simulation_only')}`",
    ]
    if report.get("reason"):
        lines.append(f"Reason: {report['reason']}")
    lines.extend(
        [
            "",
            "## Backends",
            f"- Policy: `{report.get('policy_backend')}`",
            f"- Count model: `{report.get('count_backend')}`",
            f"- NICE executor: `{report.get('nice_executor')}`",
            f"- Steps completed: `{report.get('steps_completed', 0)}`",
        ]
    )
    if report.get("hardware_safety_note"):
        lines.extend(["", "## Safety Boundary", report["hardware_safety_note"]])
    if report.get("observations"):
        lines.extend(["", "## Observations"])
        for item in report["observations"]:
            overlay = item.get("simulation_count_overlay") or {}
            lines.append(f"- step {item['step']}: `{item['point_id']}` expected_counts={overlay.get('expected_counts'):.3g}")
    if report.get("limitations"):
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in report["limitations"])
    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.extend(["", "## Artifacts"])
        for key, value in artifacts.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)
