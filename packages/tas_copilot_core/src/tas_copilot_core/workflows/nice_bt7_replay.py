from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle


_LOCALHOSTS = {"127.0.0.1", "localhost", "::1"}
_SUCCESS_EXECUTION_STATUSES = {
    "dry_run_script_generated",
    "mock_executed",
    "nice_remote_executed",
    "nice_container_executed",
    "nice_container_session_executed",
}
_REVIEWED_TRANSLATION_STATUSES = {"reviewed", "reviewed_simulator_aliases", "reviewed_for_simulator"}


def nice_bt7_replay(arguments: dict[str, Any]) -> dict[str, Any]:
    """Prepare or execute a NICE BT-7 simulator replay.

    This workflow is simulation-only. It consumes the `nice_replay_training_plan`
    artifact produced by `experiment-workspace` and emits a Hoogerheide-style
    remote-task script. The default executor is `dry_run`; real NICE execution
    requires an explicit `nice_remote` executor and localhost target. For
    isolated dependencies, `nice_container` starts the NICE server and Python
    client inside a local container.
    """

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_plan_path = arguments.get("replay_plan")
    plan_path = Path(str(raw_plan_path)).expanduser() if raw_plan_path else None
    executor = str(arguments.get("executor") or "dry_run")
    mode = str(arguments.get("mode") or "point_by_point_adaptive")
    host = str(arguments.get("host") or "127.0.0.1")
    port = int(arguments.get("port") or 1232)
    client_name = str(arguments.get("client_name") or "tas-copilot-nice-replay")
    max_points = int(arguments.get("max_points") or 25)
    allow_unreviewed_node_hints = bool(arguments.get("allow_unreviewed_node_hints") or False)
    preset_time = arguments.get("preset_time")
    python_executable = str(arguments.get("python_executable") or sys.executable)
    container_image = str(arguments.get("container_image") or "nice-bt7-replay-smoke:latest")
    nice_checkout = str(arguments.get("nice_checkout") or arguments.get("nice_root") or "")
    docker_executable = str(arguments.get("docker_executable") or "docker")
    timeout_default = 300 if executor == "nice_container" else 120
    timeout_seconds = int(arguments.get("timeout_seconds") or timeout_default)

    missing = []
    if not plan_path:
        missing.append("replay_plan")
    if missing:
        return _missing_report(output_dir, missing, "Replay plan and output directory are required.")
    if not plan_path.exists():
        return _missing_report(output_dir, ["replay_plan"], f"Replay plan was not found: {plan_path}")
    if executor not in {"dry_run", "mock", "nice_remote", "nice_container", "nice_container_session"}:
        return _validation_report(output_dir, f"Unsupported NICE replay executor: {executor}")
    if mode not in {"point_by_point_adaptive", "trajectory_replay"}:
        return _validation_report(output_dir, f"Unsupported replay mode: {mode}")
    if host not in _LOCALHOSTS and not bool(arguments.get("allow_live_host")):
        return _blocked_report(output_dir, f"NICE execution against a non-local host {host!r} needs --allow-live-host (live instruments are in scope since 2026-09-09; hardware limits stay with NICE).")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _validation_report(output_dir, f"Could not parse replay plan JSON: {type(exc).__name__}: {exc}")

    if plan.get("workflow_boundary") != "simulation_only_training_harness":
        return _blocked_report(output_dir, "Replay plan is not marked simulation_only_training_harness.")
    recipe = _select_recipe(plan, arguments.get("recipe_id"))
    if recipe is None:
        return _missing_report(output_dir, ["recipe_id"], "No replay recipe matched the request.")

    points = _points_from_recipe(recipe, max_points=max_points)
    if not points:
        return _missing_report(output_dir, ["recipe.points", "recipe.ranges"], "Selected replay recipe did not contain executable points or ranges.")
    translation_status = str(recipe.get("translation_status") or "")
    review_blocked = translation_status not in _REVIEWED_TRANSLATION_STATUSES and not allow_unreviewed_node_hints
    if executor in {"nice_remote", "nice_container", "nice_container_session"} and review_blocked:
        return _blocked_report(
            output_dir,
            "NICE execution requires reviewed simulator aliases or --allow-unreviewed-node-hints for an explicitly local simulation smoke.",
            extra={"translation_status": translation_status, "recipe_id": recipe.get("replay_recipe_id")},
        )

    count_seconds = float(preset_time if preset_time is not None else (recipe.get("counts") or 1.0))
    file_prefix = _safe_file_prefix(str(arguments.get("file_prefix") or recipe.get("replay_recipe_id") or "tas_copilot_nice_replay"))
    points_path = output_dir / "nice_replay_points.json"
    points_path.write_text(json.dumps(points, indent=2, sort_keys=True), encoding="utf-8")
    script_path = output_dir / "nice_hoogerheide_replay.py"
    script_path.write_text(_hoogerheide_replay_script(), encoding="utf-8")
    trace_path = output_dir / "nice_replay_trace.jsonl"

    execution = {
        "executor": executor,
        "status": "dry_run_script_generated",
        "host": host,
        "port": port,
        "client_name": client_name,
        "mode": mode,
        "point_count": len(points),
        "count_seconds": count_seconds,
        "file_prefix": file_prefix,
        "translation_status": translation_status,
        "review_blocked": review_blocked,
        "equivalent_command": (
            f"{python_executable} {script_path} --host {host} --port {port} "
            f"--client-name {client_name} --points {points_path} --trace {trace_path} "
            f"--file-prefix {file_prefix} --preset-time {count_seconds}"
        ),
    }
    if executor == "mock":
        execution = _run_mock(points, trace_path, execution)
    elif executor == "nice_remote":
        execution = _run_nice_remote(
            python_executable=python_executable,
            script_path=script_path,
            host=host,
            port=port,
            client_name=client_name,
            points_path=points_path,
            trace_path=trace_path,
            file_prefix=file_prefix,
            count_seconds=count_seconds,
            timeout_seconds=timeout_seconds,
            execution=execution,
        )
    elif executor == "nice_container":
        execution = _run_nice_container(
            output_dir=output_dir,
            container_image=container_image,
            nice_checkout=nice_checkout,
            docker_executable=docker_executable,
            points_path=points_path,
            script_path=script_path,
            trace_path=trace_path,
            file_prefix=file_prefix,
            count_seconds=count_seconds,
            timeout_seconds=timeout_seconds,
            execution=execution,
        )
    elif executor == "nice_container_session":
        session = start_nice_container_session(
            session_root=output_dir,
            container_image=container_image,
            nice_checkout=nice_checkout,
            docker_executable=docker_executable,
            timeout_seconds=timeout_seconds,
        )
        try:
            if session["status"] == "nice_container_session_ready":
                execution = run_nice_container_session_replay(
                    session=session,
                    points_path=points_path,
                    script_path=script_path,
                    trace_path=trace_path,
                    file_prefix=file_prefix,
                    count_seconds=count_seconds,
                    timeout_seconds=timeout_seconds,
                    execution=execution,
                )
            else:
                execution = dict(execution)
                execution.update(session)
                execution["status"] = session["status"]
        finally:
            stop_nice_container_session(session)

    report = {
        "workflow_name": "nice-bt7-replay",
        "status": "success" if execution["status"] in _SUCCESS_EXECUTION_STATUSES else "error",
        "safety_boundary": "simulation_only",
        "plan_path": str(plan_path),
        "selected_recipe": recipe,
        "points": points,
        "execution": execution,
        "missing_inputs": [],
        "warnings": _warnings(executor, review_blocked, translation_status),
        "provenance": [
            ProvenanceRecord(
                source_type="nice_replay_training_plan",
                source_id=str(plan_path),
                method="hoogerheide_remote_task_adapter",
                confidence=0.75 if executor not in {"nice_remote", "nice_container"} else 0.65,
            ).model_dump(mode="json")
        ],
        "artifacts": {
            "points": str(points_path),
            "script": str(script_path),
            "trace": str(trace_path) if trace_path.exists() else None,
        },
    }
    report["artifacts"].update(
        write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_nice_replay_markdown,
            json_filename="nice_bt7_replay_report.json",
            markdown_filename="nice_bt7_replay_report.md",
        )
    )
    Path(report["artifacts"]["json"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _select_recipe(plan: dict[str, Any], recipe_id: Any) -> dict[str, Any] | None:
    recipes = list(plan.get("recipes") or [])
    if recipe_id:
        for recipe in recipes:
            if str(recipe.get("replay_recipe_id")) == str(recipe_id):
                return dict(recipe)
        return None
    return dict(recipes[0]) if recipes else None


def _points_from_recipe(recipe: dict[str, Any], *, max_points: int) -> list[dict[str, Any]]:
    explicit_points = recipe.get("points")
    if isinstance(explicit_points, list) and explicit_points:
        return [_normalize_point(point, index=i) for i, point in enumerate(explicit_points[:max_points]) if isinstance(point, dict)]
    ranges = [item for item in (recipe.get("ranges") or []) if isinstance(item, dict)]
    ranges = [item for item in ranges if item.get("nice_node_hint") and _is_number(item.get("start")) and _is_number(item.get("stop"))]
    if not ranges:
        return []
    npts = int(recipe.get("npts") or max_points or 1)
    npts = max(1, min(npts, max_points))
    points = []
    for index in range(npts):
        fraction = 0.0 if npts == 1 else index / (npts - 1)
        nodes = {}
        axes = {}
        for item in ranges:
            start = float(item["start"])
            stop = float(item["stop"])
            value = start + (stop - start) * fraction
            nodes[str(item["nice_node_hint"])] = value
            axes[str(item.get("axis") or item["nice_node_hint"])] = value
        points.append({"point_index": index, "nodes": nodes, "source_axes": axes})
    return points


def _normalize_point(point: dict[str, Any], *, index: int) -> dict[str, Any]:
    nodes = point.get("nodes") or point.get("movements") or {}
    normalized_nodes = {str(key): float(value) for key, value in nodes.items() if _is_number(value)}
    return {"point_index": int(point.get("point_index", index)), "nodes": normalized_nodes, "source_axes": dict(point.get("source_axes") or {})}


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _safe_file_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:80] or "tas_copilot_nice_replay"


def _warnings(executor: str, review_blocked: bool, translation_status: str) -> list[str]:
    warnings = [
        "This workflow is simulation-only and is not an approved live-control path.",
        "Detector counts from mock or model overlays are derived simulation artifacts, not raw detector readbacks.",
    ]
    if executor == "dry_run":
        warnings.append("Dry run generated a Hoogerheide-style script but did not contact NICE.")
    if executor == "nice_container_session":
        warnings.append("Container session execution is simulation-only; it reuses one local NICE simulator container for multiple replays.")
    if review_blocked:
        warnings.append(f"Recipe aliases are not reviewed for simulator execution: {translation_status or 'missing'}")
    return warnings


def _run_mock(points: list[dict[str, Any]], trace_path: Path, execution: dict[str, Any]) -> dict[str, Any]:
    with trace_path.open("w", encoding="utf-8") as handle:
        for point in points:
            value_sum = sum(float(value) for value in (point.get("nodes") or {}).values())
            row = {
                "point_index": point.get("point_index"),
                "nodes": point.get("nodes"),
                "status": "mock_count",
                "counter_liveROI": int(round(1000 + 10 * value_sum)),
                "counter_liveTime": execution["count_seconds"],
                "timestamp": time.time(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    execution = dict(execution)
    execution["status"] = "mock_executed"
    execution["trace_path"] = str(trace_path)
    return execution


def _run_nice_remote(
    *,
    python_executable: str,
    script_path: Path,
    host: str,
    port: int,
    client_name: str,
    points_path: Path,
    trace_path: Path,
    file_prefix: str,
    count_seconds: float,
    timeout_seconds: int,
    execution: dict[str, Any],
) -> dict[str, Any]:
    argv = [
        python_executable,
        str(script_path),
        "--host",
        host,
        "--port",
        str(port),
        "--client-name",
        client_name,
        "--points",
        str(points_path),
        "--trace",
        str(trace_path),
        "--file-prefix",
        file_prefix,
        "--preset-time",
        str(count_seconds),
    ]
    execution = dict(execution)
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        execution["returncode"] = None
        execution["stdout_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        execution["stderr_tail"] = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        execution["trace_path"] = str(trace_path) if trace_path.exists() else None
        execution["status"] = "nice_remote_timeout"
        execution["timeout_seconds"] = timeout_seconds
        execution["timeout_message"] = f"NICE remote replay exceeded {timeout_seconds}s"
        return execution
    execution["returncode"] = completed.returncode
    execution["stdout_tail"] = completed.stdout[-4000:]
    execution["stderr_tail"] = completed.stderr[-4000:]
    execution["trace_path"] = str(trace_path) if trace_path.exists() else None
    execution["status"] = "nice_remote_executed" if completed.returncode == 0 else "nice_remote_failed"
    return execution


def _run_nice_container(
    *,
    output_dir: Path,
    container_image: str,
    nice_checkout: str,
    docker_executable: str,
    points_path: Path,
    script_path: Path,
    trace_path: Path,
    file_prefix: str,
    count_seconds: float,
    timeout_seconds: int,
    execution: dict[str, Any],
) -> dict[str, Any]:
    execution = dict(execution)
    nice_root = _resolve_nice_checkout(nice_checkout)
    if nice_root is None:
        execution["status"] = "nice_container_unavailable"
        execution["returncode"] = None
        execution["stderr_tail"] = "NICE checkout not found. Provide --nice-checkout or set NICE_CHECKOUT."
        execution["container_image"] = container_image
        return execution
    if not (nice_root / "gradlew").exists():
        execution["status"] = "nice_container_unavailable"
        execution["returncode"] = None
        execution["stderr_tail"] = f"NICE checkout does not contain gradlew: {nice_root}"
        execution["container_image"] = container_image
        execution["nice_checkout"] = str(nice_root)
        return execution

    run_script = output_dir / "nice_container_replay.sh"
    run_script.write_text(_nice_container_replay_script(), encoding="utf-8")
    argv = [
        docker_executable,
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{nice_root}:/workspace",
        "-v",
        f"{output_dir.resolve()}:/replay_out",
        "-e",
        f"NICE_REPLAY_FILE_PREFIX={file_prefix}",
        "-e",
        f"NICE_REPLAY_PRESET_TIME={count_seconds}",
        "-w",
        "/workspace",
        container_image,
        "bash",
        "/replay_out/nice_container_replay.sh",
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        execution["status"] = "nice_container_unavailable"
        execution["returncode"] = None
        execution["stdout_tail"] = ""
        execution["stderr_tail"] = f"container runtime not found: {exc}"
        execution["container_image"] = container_image
        execution["nice_checkout"] = str(nice_root)
        return execution
    except subprocess.TimeoutExpired as exc:
        execution["status"] = "nice_container_timeout"
        execution["returncode"] = None
        execution["stdout_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        execution["stderr_tail"] = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        execution["trace_path"] = str(trace_path) if trace_path.exists() else None
        execution["timeout_seconds"] = timeout_seconds
        execution["timeout_message"] = f"NICE container replay exceeded {timeout_seconds}s"
        execution["container_image"] = container_image
        execution["nice_checkout"] = str(nice_root)
        execution["run_script"] = str(run_script)
        return execution

    execution["returncode"] = completed.returncode
    execution["stdout_tail"] = completed.stdout[-4000:]
    execution["stderr_tail"] = completed.stderr[-4000:]
    execution["trace_path"] = str(trace_path) if trace_path.exists() else None
    execution["status"] = "nice_container_executed" if completed.returncode == 0 else "nice_container_failed"
    execution["container_image"] = container_image
    execution["nice_checkout"] = str(nice_root)
    execution["run_script"] = str(run_script)
    execution["server_log"] = str(output_dir / "nice_container_server.log")
    execution["writer_log"] = str(output_dir / "nice_container_writer.log")
    execution["created_data_files"] = _created_container_data_files(output_dir)
    return execution


def start_nice_container_session(
    *,
    session_root: Path,
    container_image: str,
    nice_checkout: str,
    docker_executable: str,
    timeout_seconds: int,
    container_name: str | None = None,
) -> dict[str, Any]:
    """Start a long-lived NICE simulator container for repeated replay calls."""

    session_root.mkdir(parents=True, exist_ok=True)
    nice_root = _resolve_nice_checkout(nice_checkout)
    base = {
        "executor": "nice_container_session",
        "container_image": container_image,
        "session_root": str(session_root),
    }
    if nice_root is None:
        return {
            **base,
            "status": "nice_container_session_unavailable",
            "returncode": None,
            "stderr_tail": "NICE checkout not found. Provide --nice-checkout or set NICE_CHECKOUT.",
        }
    if not (nice_root / "gradlew").exists():
        return {
            **base,
            "status": "nice_container_session_unavailable",
            "returncode": None,
            "stderr_tail": f"NICE checkout does not contain gradlew: {nice_root}",
            "nice_checkout": str(nice_root),
        }

    name = container_name or _session_container_name(session_root)
    start_script = session_root / "nice_container_session_start.sh"
    start_script.write_text(_nice_container_session_start_script(), encoding="utf-8")
    ready_path = session_root / "nice_container_session_ready.json"
    if ready_path.exists():
        ready_path.unlink()
    argv = [
        docker_executable,
        "run",
        "-d",
        "--rm",
        "--platform",
        "linux/amd64",
        "--name",
        name,
        "-v",
        f"{nice_root}:/workspace",
        "-v",
        f"{session_root.resolve()}:/session_out",
        "-w",
        "/workspace",
        container_image,
        "bash",
        "/session_out/nice_container_session_start.sh",
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        return {
            **base,
            "status": "nice_container_session_unavailable",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": f"container runtime not found: {exc}",
            "nice_checkout": str(nice_root),
            "container_name": name,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "nice_container_session_timeout",
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout_seconds": timeout_seconds,
            "timeout_message": f"NICE container session start exceeded {timeout_seconds}s",
            "nice_checkout": str(nice_root),
            "container_name": name,
        }
    if completed.returncode != 0:
        return {
            **base,
            "status": "nice_container_session_failed",
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "nice_checkout": str(nice_root),
            "container_name": name,
        }

    wait = _wait_for_session_ready(
        ready_path=ready_path,
        docker_executable=docker_executable,
        container_name=name,
        timeout_seconds=timeout_seconds,
    )
    return {
        **base,
        **wait,
        "docker_executable": docker_executable,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "nice_checkout": str(nice_root),
        "container_name": name,
        "container_id": completed.stdout.strip(),
        "start_script": str(start_script),
        "server_log": str(session_root / "nice_container_session_server.log"),
        "writer_log": str(session_root / "nice_container_session_writer.log"),
    }


def run_nice_container_session_replay(
    *,
    session: dict[str, Any],
    points_path: Path,
    script_path: Path,
    trace_path: Path,
    file_prefix: str,
    count_seconds: float,
    timeout_seconds: int,
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Run one replay script inside an already-started NICE simulator container."""

    execution = dict(execution)
    session_root = Path(str(session["session_root"])).expanduser().resolve()
    step_dir = points_path.parent.resolve()
    try:
        rel_dir = step_dir.relative_to(session_root)
    except ValueError:
        execution["status"] = "nice_container_session_failed"
        execution["returncode"] = None
        execution["stderr_tail"] = f"Replay step directory {step_dir} is not under session root {session_root}"
        execution["container_name"] = session.get("container_name")
        return execution

    exec_script = step_dir / "nice_container_session_exec.sh"
    exec_script.write_text(_nice_container_session_exec_script(), encoding="utf-8")
    container_step_dir = f"/session_out/{rel_dir.as_posix()}"
    argv = [
        str(session.get("docker_executable") or "docker"),
        "exec",
        "-e",
        f"NICE_SESSION_STEP_DIR={container_step_dir}",
        "-e",
        f"NICE_REPLAY_FILE_PREFIX={file_prefix}",
        "-e",
        f"NICE_REPLAY_PRESET_TIME={count_seconds}",
        str(session["container_name"]),
        "bash",
        f"{container_step_dir}/nice_container_session_exec.sh",
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        execution["status"] = "nice_container_session_unavailable"
        execution["returncode"] = None
        execution["stdout_tail"] = ""
        execution["stderr_tail"] = f"container runtime not found: {exc}"
        execution["container_name"] = session.get("container_name")
        return execution
    except subprocess.TimeoutExpired as exc:
        execution["status"] = "nice_container_session_timeout"
        execution["returncode"] = None
        execution["stdout_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        execution["stderr_tail"] = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        execution["trace_path"] = str(trace_path) if trace_path.exists() else None
        execution["timeout_seconds"] = timeout_seconds
        execution["timeout_message"] = f"NICE container session replay exceeded {timeout_seconds}s"
        execution["container_name"] = session.get("container_name")
        return execution

    execution["returncode"] = completed.returncode
    execution["stdout_tail"] = completed.stdout[-4000:]
    execution["stderr_tail"] = completed.stderr[-4000:]
    execution["trace_path"] = str(trace_path) if trace_path.exists() else None
    execution["status"] = "nice_container_session_executed" if completed.returncode == 0 else "nice_container_session_failed"
    execution["container_image"] = session.get("container_image")
    execution["container_name"] = session.get("container_name")
    execution["nice_checkout"] = session.get("nice_checkout")
    execution["run_script"] = str(exec_script)
    execution["server_log"] = session.get("server_log")
    execution["writer_log"] = session.get("writer_log")
    execution["created_data_files"] = _created_container_data_files(step_dir)
    return execution


def stop_nice_container_session(session: dict[str, Any]) -> dict[str, Any]:
    container_name = session.get("container_name")
    docker_executable = str(session.get("docker_executable") or "docker")
    if not container_name:
        return {"status": "not_started"}
    try:
        completed = subprocess.run([docker_executable, "stop", str(container_name)], capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError as exc:
        return {"status": "container_runtime_unavailable", "stderr_tail": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    return {
        "status": "stopped" if completed.returncode == 0 else "stop_failed",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _wait_for_session_ready(*, ready_path: Path, docker_executable: str, container_name: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if ready_path.exists():
            try:
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
            except Exception:
                ready = {}
            return {"status": "nice_container_session_ready", "ready": ready}
        ps = subprocess.run(
            [docker_executable, "ps", "-q", "--filter", f"name=^{container_name}$"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if ps.returncode != 0 or not ps.stdout.strip():
            logs = subprocess.run([docker_executable, "logs", container_name], capture_output=True, text=True, timeout=10, check=False)
            return {
                "status": "nice_container_session_failed",
                "stdout_tail": logs.stdout[-4000:],
                "stderr_tail": logs.stderr[-4000:],
            }
        time.sleep(1)
    return {
        "status": "nice_container_session_timeout",
        "timeout_seconds": timeout_seconds,
        "timeout_message": f"Timed out waiting for NICE container session readiness after {timeout_seconds}s",
    }


def _session_container_name(session_root: Path) -> str:
    import hashlib

    digest = hashlib.sha256(str(session_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"tascopilot-nice-{digest}"


def _resolve_nice_checkout(raw: str) -> Path | None:
    if raw:
        candidate = Path(raw).expanduser()
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        return resolved if resolved.exists() else None
    candidates = []
    env_value = os.environ.get("NICE_CHECKOUT") or os.environ.get("NICE_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return resolved
    return None


def _created_container_data_files(output_dir: Path) -> list[str]:
    manifest = output_dir / "nice_container_created_files.json"
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(item) for item in data.get("created_files", [])]


def _nice_container_replay_script() -> str:
    return textwrap.dedent(
        r'''
        #!/usr/bin/env bash
        set -euo pipefail

        RUN_DIR="/replay_out"
        SERVER_LOG="${RUN_DIR}/nice_container_server.log"
        WRITER_LOG="${RUN_DIR}/nice_container_writer.log"
        CREATED_FILES="${RUN_DIR}/nice_container_created_files.json"
        WRITER_PID=""

        rm -f "${SERVER_LOG}" "${WRITER_LOG}" "${CREATED_FILES}"

        xvfb-run -a ./gradlew --no-daemon -Dorg.gradle.vfs.watch=false runServerFromGradle --args="--resetdata bt7" --stacktrace >"${SERVER_LOG}" 2>&1 &
        SERVER_PID=$!

        cleanup() {
          if [[ -n "${WRITER_PID}" ]] && kill -0 "${WRITER_PID}" >/dev/null 2>&1; then
            kill "${WRITER_PID}" >/dev/null 2>&1 || true
            wait "${WRITER_PID}" >/dev/null 2>&1 || true
          fi
          if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            kill "${SERVER_PID}" >/dev/null 2>&1 || true
            wait "${SERVER_PID}" >/dev/null 2>&1 || true
          fi
        }
        trap cleanup EXIT

        for _ in $(seq 1 360); do
          if grep -q "Server started successfully" "${SERVER_LOG}"; then
            break
          fi
          if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "NICE server exited before startup; see ${SERVER_LOG}" >&2
            exit 10
          fi
          sleep 1
        done

        if ! grep -q "Server started successfully" "${SERVER_LOG}"; then
          echo "Timed out waiting for NICE server; see ${SERVER_LOG}" >&2
          exit 11
        fi

        DATA_DIR="/workspace/server_data/experiments/nonims0/data"
        mkdir -p "${DATA_DIR}"
        BEFORE_FILE="${RUN_DIR}/nice_container_files_before.txt"
        AFTER_FILE="${RUN_DIR}/nice_container_files_after.txt"
        find "${DATA_DIR}" -maxdepth 1 -type f | sort >"${BEFORE_FILE}"

        (
          cd "${DATA_DIR}"
          PYTHONPATH=/workspace/python /workspace/python/nicewriter columns >"${WRITER_LOG}" 2>&1
        ) &
        WRITER_PID=$!

        for _ in $(seq 1 30); do
          if grep -q "NICE writer is ready" "${WRITER_LOG}"; then
            break
          fi
          if ! kill -0 "${WRITER_PID}" >/dev/null 2>&1; then
            echo "NICE writer exited before ready; see ${WRITER_LOG}" >&2
            exit 12
          fi
          sleep 1
        done

        if ! grep -q "NICE writer is ready" "${WRITER_LOG}"; then
          echo "Timed out waiting for NICE writer; see ${WRITER_LOG}" >&2
          exit 13
        fi

        PYTHONPATH=/workspace/python python3 /replay_out/nice_hoogerheide_replay.py \
          --host localhost \
          --port 1232 \
          --client-name tas-copilot-nice-container \
          --points /replay_out/nice_replay_points.json \
          --trace /replay_out/nice_replay_trace.jsonl \
          --file-prefix "${NICE_REPLAY_FILE_PREFIX:-tas_copilot_nice_container}" \
          --preset-time "${NICE_REPLAY_PRESET_TIME:-0.1}"

        find "${DATA_DIR}" -maxdepth 1 -type f | sort >"${AFTER_FILE}"
        python3 - <<'PY'
        import json
        from pathlib import Path
        run_dir = Path("/replay_out")
        before = set((run_dir / "nice_container_files_before.txt").read_text().splitlines())
        after = set((run_dir / "nice_container_files_after.txt").read_text().splitlines())
        created = sorted(after - before)
        (run_dir / "nice_container_created_files.json").write_text(json.dumps({"created_files": created}, indent=2), encoding="utf-8")
        PY
        '''
    ).lstrip()


def _nice_container_session_start_script() -> str:
    return textwrap.dedent(
        r'''
        #!/usr/bin/env bash
        set -euo pipefail

        RUN_DIR="/session_out"
        SERVER_LOG="${RUN_DIR}/nice_container_session_server.log"
        WRITER_LOG="${RUN_DIR}/nice_container_session_writer.log"
        READY_FILE="${RUN_DIR}/nice_container_session_ready.json"
        WRITER_PID=""

        rm -f "${SERVER_LOG}" "${WRITER_LOG}" "${READY_FILE}"

        xvfb-run -a ./gradlew --no-daemon -Dorg.gradle.vfs.watch=false runServerFromGradle --args="--resetdata bt7" --stacktrace >"${SERVER_LOG}" 2>&1 &
        SERVER_PID=$!

        cleanup() {
          if [[ -n "${WRITER_PID}" ]] && kill -0 "${WRITER_PID}" >/dev/null 2>&1; then
            kill "${WRITER_PID}" >/dev/null 2>&1 || true
            wait "${WRITER_PID}" >/dev/null 2>&1 || true
          fi
          if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            kill "${SERVER_PID}" >/dev/null 2>&1 || true
            wait "${SERVER_PID}" >/dev/null 2>&1 || true
          fi
        }
        trap cleanup EXIT TERM INT

        for _ in $(seq 1 360); do
          if grep -q "Server started successfully" "${SERVER_LOG}"; then
            break
          fi
          if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "NICE server exited before startup; see ${SERVER_LOG}" >&2
            exit 10
          fi
          sleep 1
        done

        if ! grep -q "Server started successfully" "${SERVER_LOG}"; then
          echo "Timed out waiting for NICE server; see ${SERVER_LOG}" >&2
          exit 11
        fi

        DATA_DIR="/workspace/server_data/experiments/nonims0/data"
        mkdir -p "${DATA_DIR}"
        (
          cd "${DATA_DIR}"
          PYTHONPATH=/workspace/python /workspace/python/nicewriter columns >"${WRITER_LOG}" 2>&1
        ) &
        WRITER_PID=$!

        for _ in $(seq 1 30); do
          if grep -q "NICE writer is ready" "${WRITER_LOG}"; then
            break
          fi
          if ! kill -0 "${WRITER_PID}" >/dev/null 2>&1; then
            echo "NICE writer exited before ready; see ${WRITER_LOG}" >&2
            exit 12
          fi
          sleep 1
        done

        if ! grep -q "NICE writer is ready" "${WRITER_LOG}"; then
          echo "Timed out waiting for NICE writer; see ${WRITER_LOG}" >&2
          exit 13
        fi

        python3 - <<'PY'
        import json
        import time
        from pathlib import Path
        Path("/session_out/nice_container_session_ready.json").write_text(
            json.dumps({"status": "ready", "created_at": time.time()}, indent=2),
            encoding="utf-8",
        )
        PY

        tail -f /dev/null &
        wait $!
        '''
    ).lstrip()


def _nice_container_session_exec_script() -> str:
    return textwrap.dedent(
        r'''
        #!/usr/bin/env bash
        set -euo pipefail

        STEP_DIR="${NICE_SESSION_STEP_DIR:?NICE_SESSION_STEP_DIR required}"
        DATA_DIR="/workspace/server_data/experiments/nonims0/data"
        BEFORE_FILE="${STEP_DIR}/nice_container_files_before.txt"
        AFTER_FILE="${STEP_DIR}/nice_container_files_after.txt"
        CREATED_FILES="${STEP_DIR}/nice_container_created_files.json"

        mkdir -p "${DATA_DIR}"
        find "${DATA_DIR}" -maxdepth 1 -type f | sort >"${BEFORE_FILE}"

        PYTHONPATH=/workspace/python python3 "${STEP_DIR}/nice_hoogerheide_replay.py" \
          --host localhost \
          --port 1232 \
          --client-name tas-copilot-nice-container-session \
          --points "${STEP_DIR}/nice_replay_points.json" \
          --trace "${STEP_DIR}/nice_replay_trace.jsonl" \
          --file-prefix "${NICE_REPLAY_FILE_PREFIX:-tas_copilot_nice_session}" \
          --preset-time "${NICE_REPLAY_PRESET_TIME:-0.1}"

        find "${DATA_DIR}" -maxdepth 1 -type f | sort >"${AFTER_FILE}"
        python3 - <<'PY'
        import json
        import os
        from pathlib import Path
        step_dir = Path(os.environ["NICE_SESSION_STEP_DIR"])
        before = set((step_dir / "nice_container_files_before.txt").read_text().splitlines())
        after = set((step_dir / "nice_container_files_after.txt").read_text().splitlines())
        created = sorted(after - before)
        (step_dir / "nice_container_created_files.json").write_text(json.dumps({"created_files": created}, indent=2), encoding="utf-8")
        PY
        '''
    ).lstrip()


def _hoogerheide_replay_script() -> str:
    return textwrap.dedent(
        r'''
        #!/usr/bin/env python
        """Hoogerheide-style NICE BT-7 replay task generated by tascopilot.

        Simulation-only. This script expects to run in an environment where
        `nice.remote` is importable and a local NICE BT-7 simulator is running.
        """

        from __future__ import annotations

        import argparse
        import json
        import time

        from nice.remote import Task, connect


        class TasCopilotReplayTask(Task):
            def __init__(self, points, trace, file_prefix, preset_time):
                super().__init__()
                self.points = points
                self.trace = trace
                self.file_prefix = file_prefix
                self.preset_time = preset_time

            def run(self):
                control_nodes = sorted({node for point in self.points for node in point.get("nodes", {})})
                self.api.measurement_start(control_nodes=control_nodes, title="tas-copilot NICE BT-7 replay")
                try:
                    with open(self.trace, "w", encoding="utf-8") as handle:
                        for point in self.points:
                            nodes = point.get("nodes", {})
                            self.api.move(nodes)
                            # api.move blocks (wait=True), but "finished" is not
                            # the same as "arrived": a motor that reaches a soft
                            # limit stops there silently. Read the requested
                            # nodes back, plus the raw angles, so a short move is
                            # visible in the trace instead of showing up as an
                            # unexplained intensity change. Guarded: a readback
                            # problem must not abort a live count.
                            achieved = {}
                            angles = {}
                            for _node in nodes:
                                try:
                                    achieved[_node] = self.api.read(_node)
                                except Exception as _exc:  # noqa: BLE001
                                    achieved[_node] = f"<unreadable: {type(_exc).__name__}>"
                            for _motor in ("a1", "a2", "a3", "a4", "a5", "a6"):
                                try:
                                    angles[_motor] = self.api.read(_motor)
                                except Exception:  # noqa: BLE001
                                    pass
                            active = self.api.measurement_count(
                                filePrefix=self.file_prefix,
                                entry=f"point_{point.get('point_index', 0)}",
                                presetTime=self.preset_time,
                                wait=False,
                            )
                            self.api.wait(active)
                            detector_name = self.api.read("counter.countAgainstDetector")
                            counts = self.api.read(f"{detector_name}.counts")
                            live_time = self.api.read("counter.liveTime")
                            # `counts` is one entry per detector segment (BT7
                            # singleDetector is [SD01, SD02, SD03]). liveROI is the
                            # instrument's own total over those segments, kept as an
                            # independent check on the sum; liveMonitor is what a TAS
                            # measurement is normally normalised to.
                            live_roi = self.api.read("counter.liveROI")
                            live_monitor = self.api.read("counter.liveMonitor")
                            row = {
                                "point_index": point.get("point_index"),
                                "nodes": nodes,
                                "status": "nice_count",
                                "detector_name": detector_name,
                                "counts": counts,
                                "counter_liveTime": live_time,
                                "counter_liveROI": live_roi,
                                "counter_liveMonitor": live_monitor,
                                "achieved_nodes": achieved,
                                "achieved_angles": angles,
                                "timestamp": time.time(),
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                finally:
                    self.api.measurement_end()


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, default=1232)
            parser.add_argument("--client-name", default="tas-copilot-nice-replay")
            parser.add_argument("--points", required=True)
            parser.add_argument("--trace", required=True)
            parser.add_argument("--file-prefix", required=True)
            parser.add_argument("--preset-time", type=float, default=1.0)
            args = parser.parse_args()

            points = json.loads(open(args.points, "r", encoding="utf-8").read())
            task = TasCopilotReplayTask(points, args.trace, args.file_prefix, args.preset_time)
            api = connect(args.host, args.client_name, port=args.port)
            try:
                api.run_task(task, wait=True)
            finally:
                if hasattr(api, "end_serve"):
                    api.end_serve()
                if hasattr(api, "disconnect"):
                    api.disconnect()


        if __name__ == "__main__":
            main()
        '''
    ).lstrip()


def _missing_report(output_dir: Path, fields: list[str], reason: str) -> dict[str, Any]:
    report = {"workflow_name": "nice-bt7-replay", "status": "missing_input", "missing_fields": fields, "reason": reason}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_nice_replay_markdown, json_filename="nice_bt7_replay_report.json", markdown_filename="nice_bt7_replay_report.md")
    return report


def _validation_report(output_dir: Path, reason: str) -> dict[str, Any]:
    report = {"workflow_name": "nice-bt7-replay", "status": "validation_failed", "reason": reason}
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_nice_replay_markdown, json_filename="nice_bt7_replay_report.json", markdown_filename="nice_bt7_replay_report.md")
    return report


def _blocked_report(output_dir: Path, reason: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {"workflow_name": "nice-bt7-replay", "status": "blocked", "reason": reason, "safety_boundary": "simulation_only"}
    if extra:
        report.update(extra)
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_nice_replay_markdown, json_filename="nice_bt7_replay_report.json", markdown_filename="nice_bt7_replay_report.md")
    return report


def _nice_replay_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NICE BT-7 Replay",
        "",
        f"Status: `{report.get('status')}`",
        f"Boundary: `{report.get('safety_boundary', 'simulation_only')}`",
    ]
    if report.get("reason"):
        lines.append(f"Reason: {report['reason']}")
    execution = report.get("execution") or {}
    if execution:
        lines.extend(
            [
                "",
                "## Execution",
                f"- Executor: `{execution.get('executor')}`",
                f"- Execution status: `{execution.get('status')}`",
                f"- Host: `{execution.get('host')}`",
                f"- Point count: `{execution.get('point_count')}`",
                f"- Equivalent command: `{execution.get('equivalent_command')}`",
            ]
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    artifacts = report.get("artifacts") or {}
    if artifacts:
        lines.extend(["", "## Artifacts"])
        for key, value in artifacts.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)
