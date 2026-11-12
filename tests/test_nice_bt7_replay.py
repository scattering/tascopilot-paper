import json
import getpass
import importlib
import subprocess
import sys
from pathlib import Path

from tas_copilot_core.cli import main
from tas_copilot_core.workflows.nice_bt7_replay import nice_bt7_replay
from tas_copilot_core.workflows.tasai_adaptive_nice_sim import tasai_adaptive_nice_sim


def _write_plan(tmp_path: Path, *, translation_status: str = "reviewed_simulator_aliases") -> Path:
    plan = {
        "status": "ready_for_simulation_adapter",
        "workflow_boundary": "simulation_only_training_harness",
        "recipes": [
            {
                "replay_recipe_id": "nice-replay-recipe-01",
                "source_title": "fixture A3 A4 replay",
                "translation_status": translation_status,
                "recommended_replay_mode": "point_by_point_adaptive",
                "npts": 3,
                "counts": 0.1,
                "ranges": [
                    {"axis": "A3", "nice_node_hint": "sampleTheta.softPosition", "start": 10.0, "stop": 12.0},
                    {"axis": "A4", "nice_node_hint": "sampleTwoTheta.softPosition", "start": 20.0, "stop": 24.0},
                ],
            }
        ],
    }
    path = tmp_path / "nice_replay_training_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_nice_bt7_replay_dry_run_generates_hoogerheide_script(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    result = nice_bt7_replay({"replay_plan": str(plan), "output_dir": str(tmp_path / "out")})

    assert result["status"] == "success"
    assert result["execution"]["status"] == "dry_run_script_generated"
    assert result["execution"]["point_count"] == 3
    points = json.loads(Path(result["artifacts"]["points"]).read_text(encoding="utf-8"))
    assert points[0]["nodes"] == {"sampleTheta.softPosition": 10.0, "sampleTwoTheta.softPosition": 20.0}
    script = Path(result["artifacts"]["script"]).read_text(encoding="utf-8")
    assert "class TasCopilotReplayTask(Task)" in script
    assert "api.run_task(task, wait=True)" in script
    assert "measurement_count" in script


def test_nice_bt7_replay_blocks_nonlocal_host(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    result = nice_bt7_replay({"replay_plan": str(plan), "output_dir": str(tmp_path / "out"), "host": "bt7.example.gov"})

    assert result["status"] == "blocked"
    assert "--allow-live-host" in result["reason"]


def test_nice_bt7_replay_allows_live_host_with_flag(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    result = nice_bt7_replay({"replay_plan": str(plan), "output_dir": str(tmp_path / "out"), "host": "bt7.example.gov", "allow_live_host": True})

    # the host gate is passed; the dry-run executor then proceeds normally
    assert result["status"] != "blocked" or "--allow-live-host" not in result.get("reason", "")


def test_nice_bt7_replay_requires_reviewed_aliases_for_remote_execution(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, translation_status="node_hints_only_review_required")
    result = nice_bt7_replay({"replay_plan": str(plan), "output_dir": str(tmp_path / "out"), "executor": "nice_remote"})

    assert result["status"] == "blocked"
    assert result["translation_status"] == "node_hints_only_review_required"


def test_nice_bt7_replay_remote_timeout_returns_typed_error(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)

    def run_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial stdout", stderr="partial stderr")

    module = importlib.import_module("tas_copilot_core.workflows.nice_bt7_replay")
    monkeypatch.setattr(module.subprocess, "run", run_timeout)
    result = nice_bt7_replay(
        {
            "replay_plan": str(plan),
            "output_dir": str(tmp_path / "out"),
            "executor": "nice_remote",
            "timeout_seconds": 1,
        }
    )

    assert result["status"] == "error"
    assert result["execution"]["status"] == "nice_remote_timeout"
    assert result["execution"]["timeout_seconds"] == 1
    assert "exceeded 1s" in result["execution"]["timeout_message"]


def test_nice_bt7_replay_mock_executor_writes_trace(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    result = nice_bt7_replay({"replay_plan": str(plan), "output_dir": str(tmp_path / "out"), "executor": "mock"})

    assert result["status"] == "success"
    assert result["execution"]["status"] == "mock_executed"
    trace = Path(result["artifacts"]["trace"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(trace) == 3
    assert json.loads(trace[0])["status"] == "mock_count"


def test_nice_bt7_replay_container_executor_builds_isolated_runtime_call(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)
    nice_checkout = tmp_path / "NICE-master"
    nice_checkout.mkdir()
    (nice_checkout / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
    calls = []

    def run_success(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="container ok", stderr="")

    monkeypatch.setattr(importlib.import_module("tas_copilot_core.workflows.nice_bt7_replay").subprocess, "run", run_success)
    result = nice_bt7_replay(
        {
            "replay_plan": str(plan),
            "output_dir": str(tmp_path / "out"),
            "executor": "nice_container",
            "nice_checkout": str(nice_checkout),
            "container_image": "nice-test:latest",
        }
    )

    assert result["status"] == "success"
    assert result["execution"]["status"] == "nice_container_executed"
    assert result["execution"]["container_image"] == "nice-test:latest"
    assert calls[0][:4] == ["docker", "run", "--rm", "--platform"]
    assert any(str(nice_checkout) in item and ":/workspace" in item for item in calls[0])
    assert "nice_container_replay.sh" in result["execution"]["run_script"]


def test_nice_bt7_replay_container_executor_fails_closed_without_checkout(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    result = nice_bt7_replay(
        {
            "replay_plan": str(plan),
            "output_dir": str(tmp_path / "out"),
            "executor": "nice_container",
            "nice_checkout": str(tmp_path / "missing-nice"),
        }
    )

    assert result["status"] == "error"
    assert result["execution"]["status"] == "nice_container_unavailable"
    assert "NICE checkout" in result["execution"]["stderr_tail"]


def test_nice_bt7_replay_container_session_executor_reuses_service_boundary(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)
    nice_checkout = tmp_path / "NICE-master"
    nice_checkout.mkdir()
    (nice_checkout / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
    module = importlib.import_module("tas_copilot_core.workflows.nice_bt7_replay")
    calls = []

    def run_success(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(argv, 0, stdout="container-id\n", stderr="")
        if argv[:2] == ["docker", "exec"]:
            step_dir = next(item.split("=", 1)[1] for item in argv if str(item).startswith("NICE_SESSION_STEP_DIR="))
            host_step = Path(str(step_dir).replace("/session_out", str(tmp_path / "out")))
            (host_step / "nice_replay_trace.jsonl").write_text('{"status": "nice_count"}\n', encoding="utf-8")
            (host_step / "nice_container_created_files.json").write_text(json.dumps({"created_files": ["/workspace/server_data/experiments/nonims0/data/session.bt7"]}), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="exec ok", stderr="")
        if argv[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(argv, 0, stdout="stopped\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run_success)
    monkeypatch.setattr(module, "_wait_for_session_ready", lambda **kwargs: {"status": "nice_container_session_ready", "ready": {"status": "ready"}})

    result = nice_bt7_replay(
        {
            "replay_plan": str(plan),
            "output_dir": str(tmp_path / "out"),
            "executor": "nice_container_session",
            "nice_checkout": str(nice_checkout),
            "container_image": "nice-test:latest",
        }
    )

    assert result["status"] == "success"
    assert result["execution"]["status"] == "nice_container_session_executed"
    assert any(call[:2] == ["docker", "run"] and "-d" in call for call in calls)
    assert any(call[:2] == ["docker", "exec"] for call in calls)
    assert any(call[:2] == ["docker", "stop"] for call in calls)


def test_nice_bt7_replay_cli(tmp_path: Path, capsys) -> None:
    plan = _write_plan(tmp_path)
    assert main(["workflow", "nice-bt7-replay", "--replay-plan", str(plan), "--output-dir", str(tmp_path / "cli"), "--executor", "mock"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["workflow_name"] == "nice-bt7-replay"
    assert result["execution"]["status"] == "mock_executed"


def test_tasai_adaptive_nice_sim_runs_fixture_loop(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    candidates = [
        {"point_id": "left", "h": -0.2, "k": 0.0, "l": 0.0, "e": 0.0},
        {"point_id": "center", "h": 0.0, "k": 0.0, "l": 0.0, "e": 0.0},
        {"point_id": "right", "h": 0.2, "k": 0.0, "l": 0.0, "e": 0.0},
    ]
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(candidates), encoding="utf-8")

    result = tasai_adaptive_nice_sim({"candidate_points_file": str(path), "output_dir": str(tmp_path / "adaptive"), "max_steps": 2})

    assert result["status"] == "success"
    assert result["steps_completed"] == 2
    assert result["observations"][0]["point_id"] == "center"
    assert result["runtime_policy"]["recommended_hpc"] == "TACC"
    assert Path(result["artifacts"]["adaptive_trace"]).exists()
    assert Path(result["artifacts"]["simulation_count_overlay"]).exists()
    review_rows = [
        json.loads(line)
        for line in Path(result["artifacts"]["adaptive_decision_review_template_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert review_rows[0]["object_type"] == "adaptive_measurement_decision"
    assert review_rows[0]["candidate_status"] == "executed_in_simulation"


def test_tasai_adaptive_nice_sim_polls_human_stop(tmp_path: Path) -> None:
    suggestions = tmp_path / "suggestions.jsonl"
    suggestions.write_text(json.dumps({"step": 0, "action": "stop"}) + "\n", encoding="utf-8")

    result = tasai_adaptive_nice_sim({"human_suggestions": str(suggestions), "output_dir": str(tmp_path / "adaptive"), "max_steps": 3})

    assert result["steps_completed"] == 0
    assert result["stopped_reason"] == "human_stop_at_step_0"
    assert result["human_suggestions"]["mode"] == "dynamic_jsonl_poll"


def test_tasai_adaptive_nice_sim_hpc_backends_fail_closed(tmp_path: Path) -> None:
    result = tasai_adaptive_nice_sim({"policy_backend": "tasai_bumps_mcmc", "output_dir": str(tmp_path / "adaptive")})

    assert result["status"] == "unavailable"
    assert "TACC/HPC adapter" in result["reason"]


def test_tasai_adaptive_nice_sim_uses_external_analytic_tasai_model(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    checkout = tmp_path / "fake_tasai"
    simulator_dir = checkout / "tasai" / "instrument"
    simulator_dir.mkdir(parents=True)
    (checkout / "tasai" / "__init__.py").write_text("", encoding="utf-8")
    (simulator_dir / "__init__.py").write_text("", encoding="utf-8")
    (simulator_dir / "simulator.py").write_text(
        "def ferromagnetic_magnon(h, k, l, E, J=1.0):\n"
        "    return J * (1.0 + h + E)\n",
        encoding="utf-8",
    )
    for name in ["tasai.instrument.simulator", "tasai.instrument", "tasai"]:
        sys.modules.pop(name, None)

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "count_backend": "tasai_analytic",
            "tasai_path": str(checkout),
            "analytic_model": "ferromagnetic_magnon",
            "analytic_parameters": {"J": 2.0},
            "max_steps": 1,
        }
    )

    overlay = result["observations"][0]["simulation_count_overlay"]
    assert result["status"] == "success"
    assert overlay["backend"] == "tasai_analytic"
    assert overlay["source_module"].endswith("tasai/instrument/simulator.py")
    assert overlay["analytic_parameters"] == {"J": 2.0}
    assert overlay["expected_counts"] > 20.0
    for name in ["tasai.instrument.simulator", "tasai.instrument", "tasai"]:
        sys.modules.pop(name, None)


def test_tasai_adaptive_nice_sim_requires_tasai_path_for_analytic_backend(tmp_path: Path) -> None:
    result = tasai_adaptive_nice_sim({"output_dir": str(tmp_path / "adaptive"), "count_backend": "tasai_analytic"})

    assert result["status"] == "unavailable"
    assert "tasai_path" in result["reason"]


def test_tasai_adaptive_nice_sim_uses_external_andie_policy_adapter(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    adapter = tmp_path / "andie_adapter.py"
    adapter.write_text(
        "def propose_next_point(payload):\n"
        "    return {'point_id': payload['candidates'][-1]['point_id'], 'score': 123.0, 'reason': 'andie_model_discrimination', 'count_time': 7.5}\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {"point_id": "low", "h": -0.1, "k": 0, "l": 0, "e": 0},
                {"point_id": "high", "h": 0.1, "k": 0, "l": 0, "e": 0},
            ]
        ),
        encoding="utf-8",
    )

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "candidate_points_file": str(candidates),
            "policy_backend": "external_andie_adapter",
            "andie_policy_adapter": str(adapter),
            "max_steps": 1,
        }
    )

    assert result["status"] == "success"
    assert result["observations"][0]["point_id"] == "high"
    assert result["observations"][0]["policy_score"] == 123.0
    assert result["observations"][0]["policy_reason"] == "andie_model_discrimination"
    assert result["observations"][0]["count_time"] == 7.5


def test_tasai_adaptive_nice_sim_uses_checked_in_andie_example_adapter(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    adapter = Path("examples/policy_adapters/andie_model_discrimination_adapter.py").resolve()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {"point_id": "agreement", "h": -0.1, "k": 0, "l": 0, "e": 0, "model_predictions": {"ising": 10.0, "weiss": 10.5}},
                {"point_id": "disagreement", "h": 0.1, "k": 0, "l": 0, "e": 0, "model_predictions": {"ising": 2.0, "weiss": 30.0}},
            ]
        ),
        encoding="utf-8",
    )

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "candidate_points_file": str(candidates),
            "policy_backend": "external_andie_adapter",
            "andie_policy_adapter": str(adapter),
            "model_spec": {"model_weights": {"ising": 0.5, "weiss": 0.5}, "default_count_time": 12.0},
            "max_steps": 1,
        }
    )

    assert result["status"] == "success"
    assert result["observations"][0]["point_id"] == "disagreement"
    assert result["observations"][0]["policy_reason"] == "andie_model_discrimination_weighted_variance"
    assert result["observations"][0]["count_time"] == 12.0


def test_tasai_adaptive_nice_sim_uses_andie_container_policy_backend(tmp_path: Path, monkeypatch) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    workspace = tmp_path / "andie-container"
    source_root = tmp_path / "ANDiE-v1_0"
    (workspace / "src" / "andie_tascopilot_runtime").mkdir(parents=True)
    source_root.mkdir()
    (workspace / "src" / "andie_tascopilot_runtime" / "adapter.py").write_text("# adapter\n", encoding="utf-8")
    (workspace / "src" / "andie_tascopilot_runtime" / "policy.py").write_text("# policy\n", encoding="utf-8")
    (source_root / "andie_policy.py").write_text("def propose():\n    return None\n", encoding="utf-8")
    (source_root / "ANDiE_FeTe_Run-07242021.ipynb").write_text("{}", encoding="utf-8")
    calls = []
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        io_arg = next(item for item in argv if str(item).endswith(":/io"))
        host_io = Path(str(io_arg).split(":", 1)[0])
        (host_io / "policy_proposal.json").write_text(
            json.dumps(
                {
                    "point_id": "disagreement",
                    "nodes": {"Q.H": 0.1, "Q.K": 0.0, "Q.L": 0.0, "et.deltaE": 0.0},
                    "score": 196.0,
                    "reason": "andie_container_model_discrimination_weighted_variance",
                    "count_time": 0.1,
                    "metadata": {"container_smoke": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "andie_container",
            "andie_container_workspace": str(workspace),
            "andie_source_root": str(source_root),
            "andie_container_image": "andie-test:latest",
            "max_steps": 1,
            "candidate_points": [
                {"point_id": "agreement", "h": -0.1, "k": 0, "l": 0, "e": 0, "model_predictions": {"ising": 10.0, "weiss": 10.5}, "count_time": 0.1},
                {"point_id": "disagreement", "h": 0.1, "k": 0, "l": 0, "e": 0, "model_predictions": {"ising": 2.0, "weiss": 30.0}, "count_time": 0.1},
            ],
        }
    )

    assert result["status"] == "success"
    assert result["observations"][0]["point_id"] == "disagreement"
    assert result["observations"][0]["policy_reason"] == "andie_container_model_discrimination_weighted_variance"
    assert result["observations"][0]["policy_metadata"]["container_image"] == "andie-test:latest"
    assert calls[0][:4] == ["docker", "run", "--rm", "--platform"]
    assert any(str(workspace) in item and ":/workspace:ro" in item for item in calls[0])
    assert any(str(source_root) in item and ":/ANDiE-v1_0:ro" in item for item in calls[0])


def test_tasai_adaptive_nice_sim_uses_checked_in_tasai_policy_adapter(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    adapter = Path("examples/policy_adapters/tasai_uncertainty_adapter.py").resolve()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {"point_id": "slow", "h": -0.1, "k": 0, "l": 0, "e": 0, "posterior_variance": 10.0, "estimated_move_time": 1000.0},
                {"point_id": "informative", "h": 0.1, "k": 0, "l": 0, "e": 0, "posterior_variance": 8.0, "estimated_move_time": 1.0, "count_time": 9.0},
            ]
        ),
        encoding="utf-8",
    )

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "candidate_points_file": str(candidates),
            "policy_backend": "external_tasai_adapter",
            "tasai_policy_adapter": str(adapter),
            "model_spec": {"eta": 0.7, "default_count_time": 12.0},
            "max_steps": 1,
        }
    )

    assert result["status"] == "success"
    assert result["observations"][0]["point_id"] == "informative"
    assert result["observations"][0]["policy_reason"] == "tasai_uncertainty_information_rate"
    assert result["observations"][0]["count_time"] == 9.0


def test_tasai_adaptive_nice_sim_reuses_persistent_container_session(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")
    session_starts = []
    session_runs = []
    session_stops = []

    def fake_start(**kwargs):
        session_starts.append(kwargs)
        return {
            "status": "nice_container_session_ready",
            "session_root": str(kwargs["session_root"]),
            "container_name": "nice-session-test",
            "docker_executable": "docker",
            "container_image": kwargs["container_image"],
        }

    def fake_run(**kwargs):
        session_runs.append(kwargs)
        execution = dict(kwargs["execution"])
        execution["status"] = "nice_container_session_executed"
        execution["trace_path"] = str(kwargs["trace_path"])
        kwargs["trace_path"].write_text('{"status": "nice_count"}\n', encoding="utf-8")
        return execution

    def fake_stop(session):
        session_stops.append(session)
        return {"status": "stopped"}

    monkeypatch.setattr(module, "start_nice_container_session", fake_start)
    monkeypatch.setattr(module, "run_nice_container_session_replay", fake_run)
    monkeypatch.setattr(module, "stop_nice_container_session", fake_stop)

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "nice_executor": "nice_container_session",
            "nice_checkout": str(tmp_path / "NICE-master"),
            "max_steps": 2,
        }
    )

    assert result["status"] == "success"
    assert result["steps_completed"] == 2
    assert len(session_starts) == 1
    assert len(session_runs) == 2
    assert len(session_stops) == 1
    assert result["nice_session"]["mode"] == "persistent_container"
    assert result["nice_trace_index"][0]["execution_status"] == "nice_container_session_executed"
    assert result["nice_trace_index"][0]["trace"].endswith("nice_step_000/nice_replay_trace.jsonl")


def test_tasai_adaptive_nice_sim_stops_persistent_container_after_mid_loop_failure(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")
    session_runs = []
    session_stops = []

    def fake_start(**kwargs):
        return {
            "status": "nice_container_session_ready",
            "session_root": str(kwargs["session_root"]),
            "container_name": "nice-session-chaos",
            "docker_executable": "docker",
            "container_image": kwargs["container_image"],
        }

    def fake_run(**kwargs):
        session_runs.append(kwargs)
        execution = dict(kwargs["execution"])
        if len(session_runs) == 1:
            execution["status"] = "nice_container_session_executed"
            execution["trace_path"] = str(kwargs["trace_path"])
            kwargs["trace_path"].write_text('{"status": "nice_count"}\n', encoding="utf-8")
            return execution
        execution["status"] = "nice_container_session_failed"
        execution["stderr_tail"] = "simulated container death"
        return execution

    def fake_stop(session):
        session_stops.append(session)
        return {"status": "stopped_after_failure"}

    monkeypatch.setattr(module, "start_nice_container_session", fake_start)
    monkeypatch.setattr(module, "run_nice_container_session_replay", fake_run)
    monkeypatch.setattr(module, "stop_nice_container_session", fake_stop)

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "nice_executor": "nice_container_session",
            "nice_checkout": str(tmp_path / "NICE-master"),
            "max_steps": 3,
        }
    )

    assert result["status"] == "success"
    assert result["steps_completed"] == 1
    assert result["stopped_reason"] == "nice_executor_error"
    assert len(session_runs) == 2
    assert len(session_stops) == 1
    assert result["nice_session"]["stop_status"]["status"] == "stopped_after_failure"
    assert result["nice_trace_index"][1]["execution_status"] == "nice_container_session_failed"


def test_tasai_adaptive_nice_sim_uses_external_pyspinw_count_adapter(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    adapter = Path("examples/count_adapters/pyspinw_count_adapter_example.py").resolve()

    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "count_backend": "external_pyspinw_adapter",
            "pyspinw_count_adapter": str(adapter),
            "max_steps": 1,
            "count_time": 2.0,
        }
    )

    overlay = result["observations"][0]["simulation_count_overlay"]
    assert result["status"] == "success"
    assert overlay["backend"] == "external_pyspinw_adapter"
    assert overlay["metadata"]["adapter_kind"] == "protocol_example"
    assert overlay["expected_counts"] > 0


def test_tasai_adaptive_nice_sim_writes_hpc_handoff_for_andie_and_pyspinw(tmp_path: Path) -> None:
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "andie_tacc",
            "count_backend": "pyspinw_tacc",
            "prepare_hpc_handoff": True,
            "hpc_system": "vista",
            "hpc_work_dir": "/scratch/example/tasai",
        }
    )

    payload = json.loads(Path(result["artifacts"]["hpc_payload"]).read_text(encoding="utf-8"))
    run_script = Path(result["artifacts"]["hpc_run_script"])
    slurm_script = Path(result["artifacts"]["hpc_slurm_script"])
    assert result["status"] == "hpc_handoff_ready"
    assert payload["policy_backend"] == "andie_tacc"
    assert payload["count_backend"] == "pyspinw_tacc"
    assert payload["remote_runtime"]["container_runtime"] == "apptainer"
    assert payload["remote_runtime"]["andie"]["remote_container_image"].endswith("tascopilot-andie_latest.sif")
    assert "ANDiE model-discrimination/MCMC policy" in payload["expected_remote_components"]
    assert "SpinW/pySpinW count model" in payload["expected_remote_components"]
    assert payload["notes"][0] == "This handoff does not submit a job or contact TACC."
    assert run_script.exists()
    assert slurm_script.exists()
    assert "python -m andie_tascopilot_runtime.adapter" in run_script.read_text(encoding="utf-8")
    assert "\nsbatch " not in slurm_script.read_text(encoding="utf-8").lower()


def test_tasai_adaptive_nice_sim_runtime_profile_fills_hpc_fields(tmp_path: Path) -> None:
    profile = tmp_path / "runtime_profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "profile_id: stampede3",
                "site: TACC",
                "scheduler: slurm",
                "login_host: stampede3",
                "controlpath: /tmp/stampede3-{user}.sock",
                "remote_root: /work/project/user/tascopilot",
                "transient_root: /scratch/user/tascopilot",
                "container_runtime: apptainer",
                "architecture: x86_64",
                "default_account: TEST-ACCOUNT",
                "default_partition: skx-dev",
                "andie:",
                "  image: /work/project/user/tascopilot/andie_runtime/images/tascopilot-andie_latest.sif",
                "  workspace: /work/project/user/tascopilot/andie_runtime/andie-container",
                "  source_root: /work/project/user/tascopilot/andie_runtime/ANDiE-v1_0",
            ]
        ),
        encoding="utf-8",
    )
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive_profile"),
            "runtime_profile": str(profile),
            "policy_backend": "andie_tacc",
            "prepare_hpc_handoff": True,
        }
    )
    payload = json.loads(Path(result["artifacts"]["hpc_payload"]).read_text(encoding="utf-8"))
    assert payload["hpc_system"] == "stampede3"
    assert payload["hpc_account"] == "TEST-ACCOUNT"
    assert payload["hpc_partition"] == "skx-dev"
    assert payload["controlpath"] == f"/tmp/stampede3-{getpass.getuser()}.sock"
    assert payload["remote_runtime"]["andie"]["remote_container_image"].endswith("tascopilot-andie_latest.sif")
    assert "adaptive_profile" in payload["hpc_work_dir"]


def test_tasai_adaptive_nice_sim_hpc_handoff_filters_limit_violations(tmp_path: Path) -> None:
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive_limits"),
            "policy_backend": "andie_tacc",
            "prepare_hpc_handoff": True,
            "candidate_points": [
                {"point_id": "allowed", "nodes": {"sampleTwoTheta.softPosition": 50.0}},
                {"point_id": "blocked", "nodes": {"sampleTwoTheta.softPosition": 5.0}},
            ],
            "software_limits": {"sampleTwoTheta.softPosition": {"min": 25.0, "max": 98.0}},
        }
    )
    payload = json.loads(Path(result["artifacts"]["hpc_payload"]).read_text(encoding="utf-8"))
    assert [item["point_id"] for item in payload["candidate_points"]] == ["allowed"]
    assert payload["rejected_candidate_points"][0]["point_id"] == "blocked"
    assert payload["rejected_candidate_points"][0]["limit_violations"][0]["node"] == "sampleTwoTheta.softPosition"


def test_tasai_adaptive_nice_sim_hpc_submit_scaffold_fails_closed(tmp_path: Path) -> None:
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "tasai_bumps_mcmc",
            "prepare_hpc_handoff": True,
            "hpc_action": "submit",
        }
    )

    assert result["status"] == "hpc_action_unavailable"
    assert result["hpc_handoff"]["action_status"] == "not_executed_by_local_runner"
    payload = json.loads(Path(result["artifacts"]["hpc_payload"]).read_text(encoding="utf-8"))
    assert payload["hpc_action"] == "submit"
    assert Path(result["artifacts"]["hpc_run_script"]).exists()


def test_tasai_adaptive_nice_sim_hpc_submit_uses_batchmode_controlpath(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        stdout = "Submitted batch job 12345\n" if argv[-1].startswith("cd ") else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "andie_tacc",
            "prepare_hpc_handoff": True,
            "hpc_action": "submit",
            "hpc_login": "user@stampede3.tacc.utexas.edu",
            "hpc_remote_handoff_dir": "/scratch/user/tascopilot/handoff",
            "hpc_account": "EXAMPLE-ALLOCATION",
            "hpc_partition": "skx-dev",
            "controlpath": "~/.ssh/cm-%r@%h:%p",
        }
    )

    assert result["status"] == "hpc_submit_recorded"
    assert result["hpc_handoff"]["action_result"]["job_id"] == "12345"
    flattened = [item for call in calls for item in call]
    assert "BatchMode=yes" in flattened
    assert "ControlPath=~/.ssh/cm-%r@%h:%p" in flattened
    slurm_script = Path(result["artifacts"]["hpc_slurm_script"]).read_text(encoding="utf-8")
    assert "#SBATCH -A EXAMPLE-ALLOCATION" in slurm_script
    assert "#SBATCH -p skx-dev" in slurm_script


def test_tasai_adaptive_nice_sim_hpc_fetch_loads_nested_policy_proposal(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")

    def fake_run(argv, **kwargs):
        destination = Path(argv[-1])
        remote_leaf = destination / "3155016"
        remote_leaf.mkdir(parents=True)
        (remote_leaf / "policy_proposal.json").write_text(
            json.dumps({"point_id": "fixture-qh-0", "nodes": {"Q.H": -0.2}, "score": 0.0}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "andie_tacc",
            "prepare_hpc_handoff": True,
            "hpc_action": "fetch",
            "hpc_login": "user@stampede3.tacc.utexas.edu",
            "hpc_remote_handoff_dir": "/work/project/tascopilot/handoff",
            "hpc_remote_output_dir": "/work/project/tascopilot/handoff/remote_job_output/3155016",
            "controlpath": "~/.ssh/cm-%r@%h:%p",
        }
    )

    action = result["hpc_handoff"]["action_result"]
    assert result["status"] == "hpc_fetch_recorded"
    assert action["policy_proposal"]["point_id"] == "fixture-qh-0"
    assert "Remote hpc-action='fetch' was attempted" in result["limitations"][0]


def test_tasai_adaptive_nice_sim_hpc_submit_blocks_mfa_prompt(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("tas_copilot_core.workflows.tasai_adaptive_nice_sim")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr="Verification code:")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = tasai_adaptive_nice_sim(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "policy_backend": "andie_tacc",
            "prepare_hpc_handoff": True,
            "hpc_action": "submit",
            "hpc_login": "user@stampede3.tacc.utexas.edu",
            "hpc_remote_handoff_dir": "/scratch/user/tascopilot/handoff",
            "hpc_account": "EXAMPLE-ALLOCATION",
        }
    )

    assert result["status"] == "hpc_action_mfa_or_auth_blocked"
    assert result["hpc_handoff"]["action_result"]["failed_command"]["blocked_reason"] == "mfa_prompt_detected"


def test_tasai_adaptive_nice_sim_cli(tmp_path: Path, capsys) -> None:
    import pytest
    pytest.importorskip("tasreduction.uncertainty", reason="counting uncertainties require tasreduction")
    assert main(["workflow", "tasai-adaptive-nice-sim", "--output-dir", str(tmp_path / "cli"), "--max-steps", "1"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["workflow_name"] == "tasai-adaptive-nice-sim"
    assert result["steps_completed"] == 1
