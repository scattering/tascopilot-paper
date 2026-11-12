"""Offline integration through the real NICE session/server and native files.

Only nice.remote and process transport are replaced. No test opens a network
connection, submits a job, or uses an instrument. Native profile ingestion and
the full-scan owner/executor remain the production implementations.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import types

import pytest

from tas_copilot_core.workflows.nice_point_session import NicePointSession, NiceSessionError

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts/nice_point_server.py"
pytestmark = pytest.mark.skipif(
    not SERVER_PATH.exists(),
    reason="optional NICE server is not distributed in the read-only paper release",
)


def load_mocked_server(monkeypatch, api):
    if not SERVER_PATH.exists():
        pytest.skip("optional NICE server is not distributed in the read-only paper release")
    nice = types.ModuleType("nice")
    nice.__path__ = []
    remote = types.ModuleType("nice.remote")

    class Task:
        def __init__(self, name):
            self.name = name

    def connect(host, client_name, *, port):
        assert host in {"127.0.0.1", "localhost"}, "tests must never select an instrument host"
        api.connections.append((host, client_name, port))
        return api

    remote.Task, remote.connect = Task, connect
    monkeypatch.setitem(sys.modules, "nice", nice)
    monkeypatch.setitem(sys.modules, "nice.remote", remote)
    spec = importlib.util.spec_from_file_location("andie_mocked_nice_point_server", SERVER_PATH)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


class FakeAPI:
    """Only the existing NICE API surface; retains each commanded trajectory."""
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.connections = []
        self.trajectories = []
        self.moves = []
        self.reads = []
        self.tasks = []
        self.nodes = {
            "Q.H": 0.0, "Q.K": 1.0, "Q.L": 0.0, "et.deltaE": 0.0,
            "a1": 20.0, "a2": 40.0, "a3": 80.0, "a4": 45.0, "a5": 20.0, "a6": 40.0,
            "counter.countAgainstDetector": "singleDetector",
            "singleDetector.counts": [101, 203], "counter.liveTime": 30.25,
            "counter.liveROI": 99999, "counter.liveMonitor": 4321,
        }
        self.closed = False
        self.after_trajectory = None

    def run_task(self, task, wait):
        assert wait is True
        task.api = self
        self.tasks.append(task)
        task.run()

    def read(self, node):
        self.reads.append(node)
        return self.nodes[node]

    def move(self, nodes):
        self.moves.append(dict(nodes))
        self.nodes.update(nodes)

    def run_trajectory_contents(self, trajectory, wait):
        assert wait is True
        self.trajectories.append(trajectory)
        for node, value in trajectory.get("init", []):
            self.nodes[node] = ast.literal_eval(value) if isinstance(value, str) else value
        for node, definition in trajectory["loops"][0]["vary"]:
            self.nodes[node] = definition["list"]["value"][-1]
        if self.after_trajectory:
            self.after_trajectory(trajectory)

    def end_serve(self):
        self.closed = True

    def disconnect(self):
        self.closed = True


def install_inprocess_server(monkeypatch, server, api):
    """Popen-shaped transport executes the real server main in a local thread."""
    processes = []

    class LocalProcess:
        def __init__(self, argv, **kwargs):
            self.argv = list(argv)
            self.returncode = None
            self.failure = None
            self.spool = Path(argv[argv.index("--spool") + 1])
            self.argv_for_server = argv[1:]
            processes.append(self)

            def run():
                try:
                    # Tests run one server at a time; this is the real CLI path.
                    monkeypatch.setattr(sys, "argv", self.argv_for_server)
                    self.returncode = server.main()
                except BaseException as error:
                    self.failure = error
                    self.returncode = 1

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise subprocess.TimeoutExpired(self.argv, timeout)
            return self.returncode

        def terminate(self):
            (self.spool / "STOP").write_text("")

        def kill(self):
            self.terminate()

    monkeypatch.setattr("tas_copilot_core.workflows.nice_point_session.subprocess.Popen", LocalProcess)
    return processes


def point_session(spool):
    return NicePointSession(server_script=SERVER_PATH, spool_dir=spool,
                            host="127.0.0.1", port=1232, client_name="offline-test",
                            file_prefix="point_", control_nodes=["Q.K", "et.deltaE"],
                            python_executable=sys.executable, timeout_seconds=2,
                            startup_timeout=2, poll_seconds=0.001)


def test_existing_point_task_keeps_one_point_native_trajectory_and_actual_counters(monkeypatch, tmp_path):
    api = FakeAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    task = server.AdaptivePointServer(str(tmp_path), "point_", 0.001, 1, 3.9005, 0, 0.02, "MONITOR", True)
    task.api = api
    row = task._measure({"step": 7, "nodes": {"Q.H": 0.0, "Q.K": 1.0, "Q.L": 0.0, "et.deltaE": 0.0}, "preset_time": 4800})
    trajectory = api.trajectories[0]
    assert trajectory["trajName"] == trajectory["filePrefix"] == repr(row["file_prefix"])
    assert trajectory["init"][:2] == [["counter.countAgainst", "'MONITOR'"], ["counter.monitorPreset", 4800]]
    assert ["et.deltaE", 0.0] in trajectory["init"]
    assert all(len(value["list"]["value"]) == 1 for _, value in trajectory["loops"][0]["vary"])
    assert row["counter_liveMonitor"] == 4321
    assert row["counter_liveTime"] == 30.25
    assert row["counts"] == [101, 203]
    assert row["counter_liveROI"] == 99999
    assert not api.connections  # Direct Task regression does not connect.


def test_existing_point_session_uses_actual_server_task_transport(monkeypatch, tmp_path):
    api = FakeAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    # Keep legacy kinematics out of this transport regression only.
    monkeypatch.setattr(server.AdaptivePointServer, "_dryrun", lambda *args: ({}, None))
    processes = install_inprocess_server(monkeypatch, server, api)
    session = point_session(tmp_path / "spool")
    try:
        session.start()
        result = session.execute(0, {"Q.H": 0.0, "Q.K": 1.0, "Q.L": 0.0, "et.deltaE": 0.0}, 0.1)
        assert result["status"] == "nice_count"
        assert len(api.tasks) == len(api.connections) == len(api.trajectories) == 1
    finally:
        session.stop(grace=2)
    assert session.point_manifest.exists()
    assert processes[0].failure is None
    assert api.closed


def native_request(action_id="T105"):
    from tas_copilot_core.workflows.andie_scan_execution import FullScanRequest, ScanSpec
    points001 = [{"A4": 21.0 + 0.2 * i, "A3": (21.0 + 0.2 * i) / 2} for i in range(31)]
    points101 = [{"A4": 43.0 + 0.2 * i, "A3": 81.0 + (43.0 + 0.2 * i - 46.0) / 2} for i in range(31)]
    return FullScanRequest(
        run_id="offline-native", action_id=action_id, temperature_k=105,
        selected_temperature_index=1,
        sample_state={"name": "SyntheticSample", "state_id": 2},
        lattice={"a": 4.08, "c": 5.67},
        frames={"film": {"a3_zero_deg": 0.0}},
        settling={"temperature_tolerance_k": 0.1, "hold_seconds": 0},
        instrument_config={
            "nice_node_map": {"A3": "sampleTheta.softPosition", "A4": "sampleTwoTheta.softPosition"},
            "nice_init": {"et.deltaE": 0.0}, "nice_y_axis": "singleDetector.liveROI",
            "nice_state_checks": [
                {"node": "sampleState.A", "request_path": ["lattice", "a"], "tolerance": 1e-6},
                {"node": "sampleState.C", "request_path": ["lattice", "c"], "tolerance": 1e-6},
                {"node": "sampleState.a3Zero", "request_path": ["frames", "film", "a3_zero_deg"], "tolerance": 1e-6}],
            "nice_temperature": {"setpoint_node": "temp.set1", "readback_node": "temp",
                                 "poll_seconds": 0.001, "timeout_seconds": 2},
            "nice_data": {"detector": "singleDetector", "detector_column": "Detector",
                          "detector_segments": ["SD1", "SD2", "SD3"], "monitor_column": "monitor",
                          "time_column": "time", "temperature_column": "temp", "timestamp_column": "timeStamp",
                          "point_column": "pt", "axis_columns": {"A3": "A3", "A4": "A4"},
                          "axis_tolerances": {"A3": 0.01, "A4": 0.01}, "sample_state_column": "sampleStateID"},
        },
        scans=[ScanSpec(scan_id="s001", reflection="001", role="inference", axis="A4", axis_unit="degree",
                        frame_id="film", points=points001, component_roles={"001": "inference"},
                        count_mode="monitor", preset=39048),
               ScanSpec(scan_id="s101", reflection="101", role="inference", axis="A4", axis_unit="degree",
                        frame_id="film", points=points101,
                        component_roles={"lower": "nuisance", "101_central": "inference", "upper": "nuisance"},
                        count_mode="monitor", preset=39048)])


class NativeFileAPI(FakeAPI):
    def __init__(self, data_root):
        super().__init__(data_root)
        self.nodes.update({"sampleStateID": 2, "sampleState.A": 4.08, "sampleState.C": 5.67,
                           "sampleState.a3Zero": 0.0, "counter.primaryDetector": "singleDetector",
                           "temp.set1": 105.0, "temp": 105.0})
        self.native_files = []
        self.file_fault = None
        self.forced_readbacks = {}
        self.dryruns = []
        self.dryrun_point_count = None

    def read(self, node):
        if node in self.forced_readbacks:
            self.reads.append(node)
            return self.forced_readbacks[node]
        return super().read(node)

    def dryrun_trajectory_contents(self, trajectory):
        self.dryruns.append(trajectory)
        count = len(trajectory["loops"][0]["vary"][0][1]["list"]["value"])
        if self.dryrun_point_count is not None:
            count = self.dryrun_point_count
        return types.SimpleNamespace(dryrunErrors={}, points=[{"pt": i + 1} for i in range(count)])

    def run_trajectory_contents(self, trajectory, wait):
        callback = self.after_trajectory
        self.after_trajectory = None
        super().run_trajectory_contents(trajectory, wait)
        self.after_trajectory = callback
        self.write_native_file(trajectory)
        if callback:
            callback(trajectory)

    def write_native_file(self, trajectory):
        import time
        timestamp_base = time.time() * 1000
        prefix = ast.literal_eval(trajectory["filePrefix"])
        axes = {node: item["list"]["value"] for node, item in trajectory["loops"][0]["vary"]}
        a3, a4 = axes["sampleTheta.softPosition"], axes["sampleTwoTheta.softPosition"]
        expected_count = len(a3)
        row_count = expected_count - 1 if self.file_fault == "partial" else expected_count
        columns = ["pt", "A3(deg)", "A4(deg)", "monitor", "Detector", "time(s)", "temp(K)",
                   "SD1", "SD2", "SD3", "sampleStateID", "timeStamp",
                   "sampleState.A(Ang)", "sampleState.C(Ang)", "sampleState.a3Zero(deg)", "sample.description"]
        headers = ["# date 2026-10-31T01:00:00-04:00", "# primaryDetector singleDetector",
                   f"# trajectory.name {ast.literal_eval(trajectory['trajName'])}",
                   f"# trajectoryData.filePrefix {prefix}", f"# trajectory.length {expected_count}",
                   f"# scanLength {expected_count}", f"# numPoints {row_count}",
                   "# trajectory.controls [A4, A3]"]
        if self.file_fault == "wrong_prefix":
            headers[3] = "# trajectoryData.filePrefix unrelated_old_scan"
        if self.file_fault == "missing_monitor":
            columns.remove("monitor")
        values = []
        for i in range(row_count):
            row = {"pt": i + 1, "A3(deg)": a3[i] + 0.001, "A4(deg)": a4[i] - 0.001,
                   "monitor": 39000 + 10 * i, "Detector": 63 + 3 * i, "time(s)": 30.1 + 0.01 * i,
                   "temp(K)": 105 + 0.0005 * i, "SD1": 11 + i, "SD2": 21 + i, "SD3": 31 + i,
                   "sampleStateID": 1 if self.file_fault == "wrong_state" else 2,
                   "timeStamp": timestamp_base + 0.01 * i,
                   "sampleState.A(Ang)": 4.08, "sampleState.C(Ang)": 5.67,
                   "sampleState.a3Zero(deg)": 0.0, "sample.description": "SyntheticSample"}
            if self.file_fault == "bad_detector_sum":
                row["Detector"] += 100
            if self.file_fault == "duplicate_point" and i == 2:
                row["pt"] = 2
            values.append(" ".join(str(row[column]) for column in columns))
        path = self.data_root / f"{prefix}{1000 + len(self.native_files)}.dat.bt7"
        path.write_text("\n".join([*headers, "## " + " ".join(columns), *values]) + "\n")
        self.native_files.append(path)
        return path


def bridge_session(spool, *, enabled=True, owner_id="offline-owner"):
    return NicePointSession(server_script=SERVER_PATH, spool_dir=spool,
                            host="127.0.0.1", port=1232, client_name="offline-full-scan",
                            file_prefix="andie_", control_nodes=[], python_executable=sys.executable,
                            timeout_seconds=2, startup_timeout=2, poll_seconds=0.001,
                            mode="full_scan_bridge", enabled=enabled, owner_id=owner_id)


def test_bridge_disabled_by_default_cannot_spawn_or_connect(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("tas_copilot_core.workflows.nice_point_session.subprocess.Popen",
                        lambda *args, **kwargs: calls.append(args))
    session = bridge_session(tmp_path / "spool", enabled=False)
    with pytest.raises(NiceSessionError, match="explicit enablement"):
        session.start()
    assert calls == []
    assert not session.spool.exists()


def test_new_bridge_namespace_does_not_reuse_old_ready_or_requests(tmp_path):
    first = bridge_session(tmp_path / "spool")
    first.spool.mkdir(parents=True)
    (first.spool / "READY").write_text(json.dumps({"session_id": first.session_id, "owner_id": first.owner_id}))
    (first.spool / "command_old.json").write_text("{}")
    second = bridge_session(tmp_path / "spool")
    assert second.session_id != first.session_id
    assert second.spool != first.spool
    assert not (second.spool / "READY").exists()
    assert not list(second.spool.glob("command_*.json"))


def test_bridge_rejects_stale_response_and_changed_command_identity(tmp_path):
    import hashlib
    session = bridge_session(tmp_path / "spool")
    session.spool.mkdir(parents=True)
    session._proc = types.SimpleNamespace(poll=lambda: None)
    name = hashlib.sha256(b"known-command").hexdigest()
    response = session.spool / f"command_response_{name}.json"
    response.write_text(json.dumps({"status": "complete", "session_id": "old-session", "output": {"ready": True}}))
    with pytest.raises(NiceSessionError, match="stale or mismatched"):
        session.command("configure", {"sample_state_id": 2}, command_id="known-command", timeout_seconds=0.1)
    original = (session.spool / f"command_{name}.json").read_bytes()
    with pytest.raises(NiceSessionError, match="different request"):
        session.command("configure", {"sample_state_id": 1}, command_id="known-command", timeout_seconds=0.1)
    assert (session.spool / f"command_{name}.json").read_bytes() == original
    assert not (session.spool / "STOP").exists()


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 0, -1])
def test_invalid_command_timeout_never_publishes_request(tmp_path, timeout):
    session = bridge_session(tmp_path / "spool")
    session.spool.mkdir(parents=True)
    # A dead process keeps a regression from hanging even for an infinite timeout.
    session._proc = types.SimpleNamespace(poll=lambda: 1)
    with pytest.raises(NiceSessionError, match="timeout"):
        session.command("configure", {}, command_id="invalid-timeout", timeout_seconds=timeout)
    assert not list(session.spool.glob("command_*.json"))


def state_payload():
    return {"state_id": 2, "checks": [
        {"node": "sampleState.A", "expected": 4.08, "tolerance": 1e-6},
        {"node": "sampleState.C", "expected": 5.67, "tolerance": 1e-6},
        {"node": "sampleState.a3Zero", "expected": 0.0, "tolerance": 1e-6}],
        "detector": "singleDetector"}


def test_bridge_rpc_reuses_exact_completion_without_reissuing_state_move(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    processes = install_inprocess_server(monkeypatch, server, api)
    session = bridge_session(tmp_path / "spool")
    try:
        session.start()
        output = session.command("configure", {"state": state_payload()}, command_id="configure-once", timeout_seconds=2)
        repeated = session.command("configure", {"state": state_payload()}, command_id="configure-once", timeout_seconds=2)
        assert output == repeated
        assert api.moves == [{"sampleStateID": 2}]
        assert output["readbacks"]["sampleStateID"] == 2
        assert output["readbacks"]["counter.primaryDetector"] == "singleDetector"
        assert api.tasks[0].mode == "full_scan_bridge"
        assert len(api.connections) == 1
    finally:
        session.stop(grace=2)
    assert processes[0].failure is None


def test_bridge_start_twice_refuses_second_process(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    processes = install_inprocess_server(monkeypatch, server, api)
    session = bridge_session(tmp_path / "spool")
    try:
        session.start()
        with pytest.raises(NiceSessionError):
            session.start()
        assert len(processes) == len(api.connections) == 1
    finally:
        session.stop(grace=2)
    assert all(process.failure is None for process in processes)


def test_copied_rpc_envelope_under_another_filename_never_dispatches_again(monkeypatch, tmp_path):
    import hashlib
    import time
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    session = bridge_session(tmp_path / "spool")
    try:
        session.start()
        session.command("configure", {"state": state_payload()}, command_id="original", timeout_seconds=2)
        original_id = hashlib.sha256(b"original").hexdigest()
        copied_id = "0" * 64
        (session.spool / f"command_{copied_id}.json").write_bytes(
            (session.spool / f"command_{original_id}.json").read_bytes())
        response_path = session.spool / f"command_response_{copied_id}.json"
        deadline = time.monotonic() + 2
        while not response_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert response_path.exists(), "server must explicitly reject the copied envelope"
        response = json.loads(response_path.read_text())
        assert response["status"] != "complete"
        assert api.moves == [{"sampleStateID": 2}]
        assert not (session.spool / f"command_started_{copied_id}.json").exists()
    finally:
        session.stop(grace=2)


def full_adapter(session, api, action_root, *, enabled=True):
    from tas_copilot_core.workflows.andie_nice_session import NiceFullScanSession
    return NiceFullScanSession(session=session, data_root=api.data_root, action_root=action_root,
                               owner_id="offline-owner", enabled=enabled,
                               command_timeout_seconds=2, file_timeout_seconds=0.1, file_poll_seconds=0.001)


def test_full_adapter_runs_complete_coupled_scans_and_ingests_actual_native_rows(monkeypatch, tmp_path):
    import hashlib
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    processes = install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    move = api.move
    preparing_records = []
    def require_durable_preparation(nodes):
        if "sampleStateID" in nodes:
            paths = list(adapter.spool.glob("response_*.json"))
            assert len(paths) == 1
            response = json.loads(paths[0].read_text())
            assert response["result"]["status"] == "running"
            assert response["result"]["metadata"]["stage"] == "preparing"
            preparing_records.append(response)
        move(nodes)
    monkeypatch.setattr(api, "move", require_durable_preparation)
    try:
        adapter.start()
        result = adapter.execute(req)
        assert result.status == "complete", result.reason
        assert len(preparing_records) == 1
        assert len(result.scans) == len(api.trajectories) == 2
        assert len(api.connections) == len(api.tasks) == 1
        assert api.moves[0] == {"sampleStateID": 2}
        assert api.moves[1] == {"temp.set1": 105.0}
        for specification, actual, trajectory in zip(req.scans, result.scans, api.trajectories):
            assert len(actual.rows) == 31
            assert trajectory["init"][0] == ["sampleStateID", 2]
            assert trajectory["filePrefix"] == trajectory["trajName"]
            assert ["counter.countAgainst", "'MONITOR'"] in trajectory["init"]
            assert ["counter.monitorPreset", 39048.0] in trajectory["init"]
            axes = dict(trajectory["loops"][0]["vary"])
            assert axes["sampleTheta.softPosition"]["list"]["value"] == [p["A3"] for p in specification.points]
            assert axes["sampleTwoTheta.softPosition"]["list"]["value"] == [p["A4"] for p in specification.points]
            for index, row in enumerate(actual.rows):
                assert row.point_index == index
                assert row.requested == specification.points[index]
                assert row.achieved["A3"] == pytest.approx(row.requested["A3"] + 0.001)
                assert row.achieved["A4"] == pytest.approx(row.requested["A4"] - 0.001)
                assert row.counts == 63 + 3 * index
                assert row.detector_segments == [11 + index, 21 + index, 31 + index]
                assert row.monitor == 39000 + 10 * index
                assert row.live_time == pytest.approx(30.1 + 0.01 * index)
                assert row.metadata["sampleStateID"] == 2
            artifact = actual.artifacts[0]
            assert artifact.role == "raw"
            assert Path(artifact.path).suffixes[-2:] == [".dat", ".bt7"]
            assert artifact.sha256 == hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest()
        assert "counter.primaryDetector" in api.reads
        assert "counter.countAgainstDetector" in api.reads
        assert adapter.execute(req) == result  # Durable duplicate action, no reacquisition.
        assert len(api.trajectories) == 2
    finally:
        adapter.stop()
    assert processes[0].failure is None


@pytest.mark.parametrize("node,value", [("sampleStateID", 1), ("sampleState.C", 9.0),
                                         ("sampleState.a3Zero", 0.5), ("counter.primaryDetector", "wrongDetector")])
def test_actual_state_frame_or_detector_mismatch_refuses_before_trajectory(monkeypatch, tmp_path, node, value):
    api = NativeFileAPI(tmp_path / "data")
    api.forced_readbacks[node] = value
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    try:
        adapter.start()
        req = native_request()
        result = adapter.execute(req)
        assert result.status == "unknown", result.reason
        assert not api.trajectories
        assert adapter.execute(req).status == "unknown"
        assert api.moves == [{"sampleStateID": 2}]
    finally:
        adapter.stop()


@pytest.mark.parametrize("fault", ["missing_monitor", "bad_detector_sum", "wrong_state", "wrong_prefix", "duplicate_point"])
def test_corrupt_native_profile_never_completes_or_falls_back_to_live_counters(monkeypatch, tmp_path, fault):
    api = NativeFileAPI(tmp_path / "data")
    api.file_fault = fault
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    try:
        adapter.start()
        result = adapter.execute(req)
        assert result.status == "unknown", result.reason
        assert len(api.trajectories) == 1
        assert adapter.execute(req).status == "unknown"
        assert len(api.trajectories) == 1
        assert api.native_files[0].exists()
    finally:
        adapter.stop()


def test_partial_native_profile_retains_rows_and_stops_remaining_scans(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    api.file_fault = "partial"
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    try:
        adapter.start()
        result = adapter.execute(native_request())
        assert result.status == "partial", result.reason
        assert len(result.scans) == 1 and len(result.scans[0].rows) == 30
        assert result.scans[0].status == "partial"
        assert len(api.trajectories) == 1
    finally:
        adapter.stop()


def test_ambiguous_restart_never_replays_native_trajectory(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    def lose_response(_):
        raise RuntimeError("mock connection lost after native file was written")
    api.after_trajectory = lose_response
    first = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    try:
        first.start()
        result = first.execute(req)
        assert result.status == "unknown"
        assert len(api.trajectories) == 1
    finally:
        first.stop()
    api.after_trajectory = None
    second = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    try:
        second.start()
        assert second.execute(req).status == "unknown"
        assert len(api.trajectories) == 1
        assert second.session.session_id != first.session.session_id
        assert not list(second.session.spool.glob("command_*.json"))
    finally:
        second.stop()


def test_cancel_between_scans_preserves_complete_native_scan_without_global_abort(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    cancellations = []
    def cancel_after_first(_):
        cancellations.append(adapter.request_cancel(req))
        assert not (adapter.session.spool / "STOP").exists()
    api.after_trajectory = cancel_after_first
    try:
        adapter.start()
        result = adapter.execute(req)
        assert cancellations[0].status == "cancel_unknown"
        assert result.status == "cancelled", result.reason
        assert len(api.trajectories) == 1
        assert len(result.scans) == 1 and len(result.scans[0].rows) == 31
        assert result.scans[0].status == "complete"
        assert not (adapter.session.spool / "STOP").exists()
        capabilities = json.loads((adapter.spool / "capabilities.json").read_text())
        assert capabilities["inflight_abort"] is False
        assert capabilities["cancel_mode"] == "between_scans"
    finally:
        adapter.stop()


@pytest.mark.parametrize("point_count", [0, 30])
def test_empty_or_partial_native_dryrun_cannot_start_counting(monkeypatch, tmp_path, point_count):
    api = NativeFileAPI(tmp_path / "data")
    api.dryrun_point_count = point_count
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    try:
        adapter.start()
        result = adapter.execute(native_request())
        assert result.status in {"refused", "unknown"}, result.reason
        assert not api.trajectories
        assert api.dryruns
    finally:
        adapter.stop()


def test_later_invalid_file_cannot_publish_an_earlier_partial_snapshot(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    api.file_fault = "partial"
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    real_read_file = adapter._read_file
    partial_snapshots = []

    def mutate_after_real_partial_read(path, *args):
        result = real_read_file(path, *args)
        if not partial_snapshots:
            assert result.status == "partial"
            partial_snapshots.append(result)
            # Simulate the native writer changing a file after a valid partial
            # read. All parsing above remains the actual production reader.
            path.write_text(path.read_text().replace(" monitor ", " missing_monitor "))
        return result

    monkeypatch.setattr(adapter, "_read_file", mutate_after_real_partial_read)
    try:
        adapter.start()
        result = adapter.execute(native_request())
        assert partial_snapshots and len(partial_snapshots[0].rows) == 30
        assert result.status == "unknown", "a stale partial snapshot must not outlive an invalid changed file"
        assert len(api.trajectories) == 1
    finally:
        adapter.stop()


def test_missing_numeric_frame_check_refuses_locally_before_any_state_move(monkeypatch, tmp_path):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    req.frames["film"]["a4_zero_deg"] = 0.0  # Deliberately lacks an actual readback binding.
    try:
        adapter.start()
        result = adapter.execute(req)
        assert result.status == "refused", result.reason
        assert not api.moves and not api.dryruns and not api.trajectories
        assert not list(adapter.session.spool.glob("command_*.json"))
    finally:
        adapter.stop()


def test_adapter_default_disable_blocks_action_without_spawning_owner(monkeypatch, tmp_path):
    from tas_copilot_core.workflows.andie_nice_session import NiceFullScanSession
    api = NativeFileAPI(tmp_path / "data")
    spawned = []
    monkeypatch.setattr("tas_copilot_core.workflows.nice_point_session.subprocess.Popen",
                        lambda *args, **kwargs: spawned.append(args))
    adapter = NiceFullScanSession(session=bridge_session(tmp_path / "spool"),
                                  data_root=api.data_root, action_root=tmp_path / "actions",
                                  owner_id="offline-owner")
    assert adapter.submit(native_request()).status == "blocked"
    with pytest.raises(NiceSessionError, match="disabled"):
        adapter.start()
    assert not spawned
    assert not api.connections
    assert not (adapter.spool / "capabilities.json").exists()


@pytest.mark.parametrize("fault", ["missing_monitor_mapping", "missing_axis_mapping"])
def test_bad_file_configuration_refuses_before_any_native_command(monkeypatch, tmp_path, fault):
    api = NativeFileAPI(tmp_path / "data")
    server = load_mocked_server(monkeypatch, api)
    install_inprocess_server(monkeypatch, server, api)
    adapter = full_adapter(bridge_session(tmp_path / "spool"), api, tmp_path / "actions")
    req = native_request()
    if fault == "missing_monitor_mapping":
        del req.instrument_config["nice_data"]["monitor_column"]
    else:
        del req.instrument_config["nice_data"]["axis_columns"]["A3"]
    try:
        adapter.start()
        result = adapter.execute(req)
        assert result.status == "refused", result.reason
        assert not api.moves and not api.dryruns and not api.trajectories
        assert not list(adapter.session.spool.glob("command_*.json"))
    finally:
        adapter.stop()
