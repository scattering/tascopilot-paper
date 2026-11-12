"""Concrete full-scan binding to the existing two-interpreter NICE session.

NICE/Ice remains in scripts/nice_point_server.py. This driver owns contracts,
existing BT7 readers, complete-file checks and reduction-ready raw rows. Merely
constructing this class never starts a process or contacts an instrument.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .andie_scan_execution import (
    FullScanRequest, FullScanResult, NiceScanOwner, NiceSpoolBackend, RawArtifact,
    ScanExecutor, ScanResult, ScanRow, ScanSpec, native_scan_trajectory, request_hash,
)
from .nice_point_session import NicePointSession, NiceSessionError


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _path_value(request: FullScanRequest, path: list[str]) -> Any:
    value: Any = request.model_dump(mode="json")
    for component in path:
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


def _numeric_paths(value: Any, path: tuple[str, ...]) -> set[tuple[str, ...]]:
    if isinstance(value, dict):
        return set().union(*(_numeric_paths(child, (*path, str(key))) for key, child in value.items()))
    if isinstance(value, list):
        return set().union(*(_numeric_paths(child, (*path, str(index))) for index, child in enumerate(value)))
    return {path} if isinstance(value, (int, float)) and not isinstance(value, bool) else set()


def _prefix(trajectory: dict[str, Any]) -> str:
    value = ast.literal_eval(trajectory["filePrefix"])
    if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("unsafe native file prefix")
    return value


def _stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _column(row: dict[str, float], name: str) -> float:
    matches = [value for key, value in row.items() if key.split("(", 1)[0].casefold() == name.split("(", 1)[0].casefold()]
    if len(matches) != 1 or not math.isfinite(float(matches[0])):
        raise ValueError(f"missing, ambiguous or nonfinite raw column {name!r}")
    return float(matches[0])


def _unquote(value: Any) -> str:
    return str(value).strip().strip("\"'")


class NiceFullScanSession:
    """Use the actual NicePointSession/AdaptivePointServer Task for full scans.

    enabled and session.enabled must both be explicit. A non-local host also
    requires the existing allow_live_host opt-in. start() is the only operation
    that launches the owner; submit/process_one remain distinct, auditable steps.
    """
    def __init__(self, *, session: NicePointSession, data_root: str | Path,
                 action_root: str | Path, owner_id: str, enabled: bool = False,
                 command_timeout_seconds: float = 3600,
                 file_timeout_seconds: float = 30, file_poll_seconds: float = .2):
        if session.mode != "full_scan_bridge" or session.owner_id != owner_id:
            raise ValueError("full-scan session mode and owner identity must match")
        self.session, self.owner_id, self.enabled = session, owner_id, enabled
        self.data_root = Path(data_root)
        self.command_timeout = _positive(command_timeout_seconds, "command timeout")
        self.file_timeout = _positive(file_timeout_seconds, "file timeout")
        self.file_poll = _positive(file_poll_seconds, "file poll interval")
        self.spool = session.base_spool / "full_scans"
        self.executor = ScanExecutor(action_root, NiceSpoolBackend(self.spool, enabled=enabled))
        self.owner = NiceScanOwner(self.spool, owner_id=owner_id, enabled=enabled,
            execute_trajectory=self._execute_trajectory, read_profile=self.read_profile,
            prepare_action=self._prepare_action, validate_trajectory=self._validate_trajectory)
        self.started = False
        self._request: FullScanRequest | None = None
        self._configured_hash: str | None = None
        self._execution: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        if not self.enabled or not self.session.enabled:
            raise NiceSessionError("full-scan execution is disabled; explicit owner/session enablement is required")
        if not self.data_root.is_dir():
            raise ValueError("configured native data_root must already exist")
        self.session.start()
        self.owner.advertise()
        self.started = True

    def submit(self, request: FullScanRequest) -> FullScanResult:
        if self.enabled and not self.started:
            raise NiceSessionError("start the explicitly enabled bridge before submitting")
        return self.executor.submit(request)

    def process_one(self) -> FullScanResult | None:
        if not self.started:
            raise NiceSessionError("full-scan bridge has not been started")
        return self.owner.process_one()

    def reconcile(self, request: FullScanRequest) -> FullScanResult:
        return self.executor.reconcile(request)

    def execute(self, request: FullScanRequest) -> FullScanResult:
        result = self.submit(request)
        if result.status == "awaiting_owner":
            self.process_one()
            return self.reconcile(request)
        return result

    def request_cancel(self, request: FullScanRequest) -> FullScanResult:
        return self.executor.request_cancel(request)

    def stop(self) -> None:
        self.session.stop()
        self.started = False

    def _state(self, request: FullScanRequest) -> dict[str, Any]:
        state_id = request.sample_state.get("state_id")
        if type(state_id) is not int or state_id < 0:
            raise ValueError("sample_state.state_id must be explicit")
        checks = []
        covered = set()
        for item in request.instrument_config.get("nice_state_checks", []):
            path = item["request_path"]
            if not isinstance(path, list) or not path or path[0] not in {"lattice", "frames"}:
                raise ValueError("state checks must reference requested lattice/frame fields")
            expected = _path_value(request, path)
            tolerance = float(item.get("tolerance", 0))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("state check tolerance must be finite and nonnegative")
            if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected):
                raise ValueError("state check expectation must be a finite numeric identity field")
            checks.append({"node": item["node"], "expected": expected, "tolerance": tolerance,
                           "data_column": item.get("data_column", item["node"])})
            covered.add(tuple(str(part) for part in path))
        required = _numeric_paths(request.lattice, ("lattice",))
        if not required:
            raise ValueError("numeric lattice identity fields are required")
        for frame_id in {scan.frame_id for scan in request.scans}:
            frame_paths = _numeric_paths(request.frames[frame_id], ("frames", frame_id))
            if not frame_paths:
                raise ValueError("each scan frame needs explicit numeric identity fields")
            required.update(frame_paths)
        if required - covered:
            raise ValueError(f"missing explicit lattice/frame readback checks: {sorted(required - covered)}")
        data = request.instrument_config["nice_data"]
        if not data.get("detector") or not data.get("detector_segments"):
            raise ValueError("native detector identity and segment columns are required")
        return {"state_id": state_id, "checks": checks, "detector": data["detector"]}

    def _rpc(self, request: FullScanRequest, operation: str, payload: dict[str, Any], suffix="",
             timeout: float | None = None) -> dict[str, Any]:
        return self.session.command(operation, payload,
            command_id=f"{request_hash(request)}:{operation}:{suffix}",
            timeout_seconds=self.command_timeout if timeout is None else timeout)

    def _configure(self, request: FullScanRequest) -> dict[str, Any]:
        self._request = request
        digest = request_hash(request)
        if self._configured_hash != digest:
            result = self._rpc(request, "configure", {"state": self._state(request)})
            self._configured_hash = digest
            self._configured = result
        return self._configured

    def _validate_trajectory(self, request: FullScanRequest, scan: ScanSpec, trajectory: dict[str, Any]) -> bool:
        # NiceScanOwner invokes this before publishing its durable preparing
        # marker. This callback MUST remain local and free of instrument calls.
        self._state(request)
        if trajectory.get("init", [[]])[0] != ["sampleStateID", request.sample_state["state_id"]]:
            raise ValueError("sampleStateID must be first in the native trajectory")
        temperature = request.instrument_config["nice_temperature"]
        _positive(temperature["timeout_seconds"], "temperature command timeout")
        if not temperature.get("setpoint_node") or not temperature.get("readback_node"):
            raise ValueError("explicit temperature setpoint and sample sensor nodes are required")
        data = request.instrument_config["nice_data"]
        for key in ("detector", "detector_column", "monitor_column", "time_column", "temperature_column",
                    "timestamp_column", "point_column", "sample_state_column"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"explicit native data field {key} is required")
        segments = data["detector_segments"]
        if not isinstance(segments, list) or not segments or any(not isinstance(name, str) or not name for name in segments):
            raise ValueError("explicit native detector segment column names are required")
        if len(set(segments)) != len(segments):
            raise ValueError("native detector segment columns must be unique")
        for axis in scan.points[0]:
            if not isinstance(data["axis_columns"].get(axis), str) or not data["axis_columns"][axis]:
                raise ValueError(f"explicit native axis column is required for {axis}")
            tolerance = float(data["axis_tolerances"][axis])
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("native axis tolerances must be finite and nonnegative")
        return True

    def _prepare_action(self, request: FullScanRequest) -> dict[str, Any]:
        # Owner has already persisted stage=preparing before the first state move.
        self._configure(request)
        for scan in request.scans:
            trajectory = native_scan_trajectory(request, scan)
            check = self._rpc(request, "dryrun", {"state": self._state(request),
                "trajectory": trajectory, "expected_points": len(scan.points)}, scan.scan_id)
            if check.get("allowed") is not True:
                return {"ready": False, "temperature_k": request.temperature_k,
                        "reason": "native schema/point-count preflight refused", "preflight": check}
        temperature = request.instrument_config["nice_temperature"]
        timeout = _positive(temperature["timeout_seconds"], "thermal timeout")
        payload = {"state": self._state(request), "temperature_k": request.temperature_k,
                   "setpoint_node": temperature["setpoint_node"], "readback_node": temperature["readback_node"],
                   "controller_settle": dict(temperature.get("controller_settle") or {})}
        # NICE does the settling; this is only how long to wait for its reply
        measured = self._rpc(request, "settle_temperature", payload, timeout=timeout)
        return {**measured, "sample_state": request.sample_state, "lattice": request.lattice,
                "frames": request.frames, "verified_state_readbacks": measured["readbacks"],
                "bridge_session_id": self.session.session_id}

    def _execute_trajectory(self, trajectory: dict[str, Any], *, wait: bool = True) -> dict[str, Any]:
        if not wait or self._request is None:
            raise ValueError("a configured request and synchronous native execution are required")
        request = self._request
        prefix = _prefix(trajectory)
        if request.instrument_config.get("nice_acquisition") == "measurement":
            return self._measure(request, trajectory, prefix)
        baseline = {str(path.resolve()): _stat(path) for path in self.data_root.glob(f"{prefix}*.dat.bt7")}
        result = self._rpc(request, "trajectory", {"state": self._state(request), "trajectory": trajectory}, prefix)
        self._execution[prefix] = {**result, "baseline": baseline, "trajectory": trajectory,
                                   "bridge_session_id": self.session.session_id}
        return result

    def _measure(self, request: FullScanRequest, trajectory: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Live points through NICE's scripted measurement (no file lag).

        The grid, init and preset come from the same compiled trajectory, so
        the two acquisition paths measure identical points.
        """
        config = request.instrument_config
        node_map = config["nice_node_map"]
        axis_of = {node: axis for axis, node in node_map.items()}
        init = {node: value for node, value in trajectory["init"]}
        count_mode = "monitor" if init.get("counter.countAgainst") == repr("MONITOR") else "time"
        preset = init["counter.monitorPreset"] if count_mode == "monitor" else init["counter.timePreset"]
        vary = trajectory["loops"][0]["vary"]
        columns = {axis_of[node]: item["list"]["value"] for node, item in vary}
        points = [dict(zip(columns, values)) for values in zip(*columns.values())]
        extra_init = [[node, value] for node, value in trajectory["init"]
                      if node not in {"sampleStateID", "counter.countAgainst",
                                      "counter.monitorPreset", "counter.timePreset"}]
        payload = {"state": self._state(request), "prefix": prefix, "init": extra_init,
                   "axes": {axis: node_map[axis] for axis in columns},
                   "x_axis": axis_of[trajectory["xAxis"]], "y_axis": trajectory["yAxis"],
                   "points": points, "count_mode": count_mode, "preset": preset,
                   "reads": dict(config["nice_live"])}
        result = self._rpc(request, "measure_scan", payload, prefix)
        # Keep exactly what NICE reported for each point as the raw artifact
        # (NICE also writes its own data file; this is the copy we used).
        record = self.spool / "live" / f"{prefix}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"prefix": prefix, "payload": payload, "result": result},
                          sort_keys=True, allow_nan=False)
        record.write_text(text)
        self._execution[prefix] = {**result, "trajectory": trajectory, "mode": "measurement",
                                   "record": str(record.resolve()),
                                   "record_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                   "bridge_session_id": self.session.session_id}
        return result

    def _measured_profile(self, request: FullScanRequest, scan: ScanSpec, execution: dict[str, Any]) -> ScanResult:
        state = self._state(request)
        rows = [ScanRow(point_index=row["point_index"], requested=row["requested"],
                        achieved=row["achieved"], temperature_k=row["temperature_k"],
                        counts=row["counts"], monitor=row["monitor"], live_time=row["live_time"],
                        detector_segments=row["segments"],
                        timestamp=datetime.fromtimestamp(row["timestamp"], timezone.utc).isoformat(),
                        metadata={"sampleStateID": state["state_id"], "detector": execution["detector"],
                                  "source": "nice measurement_count"})
                for row in execution["rows"]]
        expected = len(scan.points)
        return ScanResult(scan_id=scan.scan_id, status="complete" if len(rows) == expected else "partial",
                          rows=rows,
                          artifacts=[RawArtifact(path=execution["record"], sha256=execution["record_sha256"], role="raw")],
                          metadata={"native_prefix": _prefix(execution["trajectory"]),
                                               "acquisition": "nice measurement_start/count/end",
                                               "requested_points": expected,
                                               "bridge_execution": {k: v for k, v in execution.items() if k != "rows"}})

    def _read_file(self, path: Path, request: FullScanRequest, scan: ScanSpec,
                   trajectory: dict[str, Any], execution: dict[str, Any]) -> ScanResult:
        # Existing readers: retain modern mixed string/numeric rows. Do not use
        # the older installed bt7grouping path which discards such entire rows.
        from .live_experiment import read_nice_scan
        from .alignment_state import _read_bt7_numeric_table_fallback

        before = _stat(path)
        digest_before = hashlib.sha256(path.read_bytes()).hexdigest()
        record = read_nice_scan(path)
        table = _read_bt7_numeric_table_fallback(path)
        if record is None or not table or record["npts"] != len(table):
            raise ValueError("native readers disagree or scan has no complete rows")
        header = record["header"]
        prefix = _prefix(trajectory)
        if _unquote(header.get("trajectoryData.filePrefix", "")) != prefix:
            raise ValueError("native file prefix does not match the immutable requested trajectory")
        if header.get("trajectory.name") and _unquote(header["trajectory.name"]) != ast.literal_eval(trajectory["trajName"]):
            raise ValueError("native trajectory name differs from requested trajectory")
        expected = len(scan.points)
        planned = [int(float(header[key])) for key in ("trajectory.length", "trajectory.scanLength") if key in header]
        if not planned or any(n != expected for n in planned) or int(float(header.get("numPoints", -1))) != len(table):
            raise ValueError("native planned/written point metadata is inconsistent")
        if len(table) > expected:
            raise ValueError("native file contains more points than requested")
        data = request.instrument_config["nice_data"]
        if _unquote(header.get("primaryDetector", "")) != data["detector"] or execution["detector"] != data["detector"]:
            raise ValueError("native detector identity does not match request and live readback")
        if header.get("trajectoryData.yAxis") and _unquote(header["trajectoryData.yAxis"]) != request.instrument_config["nice_y_axis"]:
            raise ValueError("native detector yAxis differs from explicit configuration")
        state = self._state(request)
        rows = []
        timestamps = []
        for index, raw in enumerate(table):
            if _column(raw, data["point_column"]) != index + 1:
                raise ValueError("native point numbers must be contiguous and one-based")
            if _column(raw, data["sample_state_column"]) != state["state_id"]:
                raise ValueError("native row was recorded under the wrong sampleStateID")
            for check in state["checks"]:
                if not isinstance(check["expected"], (int, float)):
                    raise ValueError("native numeric state checks require numeric requested values")
                actual = _column(raw, check["data_column"])
                if abs(actual - check["expected"]) > check["tolerance"]:
                    raise ValueError(f"native lattice/frame readback differs at {check['data_column']}")
            achieved = {}
            for axis, desired in scan.points[index].items():
                actual = _column(raw, data["axis_columns"][axis])
                tolerance = float(data["axis_tolerances"][axis])
                if not math.isfinite(tolerance) or tolerance < 0 or abs(actual - desired) > tolerance:
                    raise ValueError(f"native achieved coordinate differs from request at {axis}")
                achieved[axis] = actual
            detector = _column(raw, data["detector_column"])
            segments = [_column(raw, name) for name in data["detector_segments"]]
            if abs(detector - sum(segments)) > .5:
                raise ValueError("native detector total disagrees with its explicitly named segments")
            temperature = _column(raw, data["temperature_column"])   # recorded by NICE; data, not a gate
            timestamp_ms = _column(raw, data["timestamp_column"])
            timestamps.append(timestamp_ms)
            if timestamp_ms / 1000 < execution["started_at"] - 5 or timestamp_ms / 1000 > execution["completed_at"] + 5:
                raise ValueError("native timestamps are outside the matched trajectory execution")
            rows.append(ScanRow(point_index=index, requested=scan.points[index], achieved=achieved,
                temperature_k=temperature, counts=detector, detector_segments=segments,
                monitor=_column(raw, data["monitor_column"]), live_time=_column(raw, data["time_column"]),
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(),
                metadata={"native_point": index + 1, "sampleStateID": state["state_id"],
                          "detector": data["detector"], "source_path": str(path)}))
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError("native timestamps are duplicated or nonmonotonic")
        digest_after = hashlib.sha256(path.read_bytes()).hexdigest()
        if _stat(path) != before or digest_before != digest_after:
            raise ValueError("native file changed while it was being read")
        return ScanResult(scan_id=scan.scan_id, status="complete" if len(rows) == expected else "partial",
            rows=rows, artifacts=[RawArtifact(path=str(path.resolve()), sha256=digest_after, role="raw")],
            metadata={"native_prefix": prefix, "native_header": header,
                      "bridge_execution": execution, "requested_points": expected,
                      "readers": ["live_experiment.read_nice_scan", "alignment_state._read_bt7_numeric_table_fallback"]})

    def read_profile(self, request: FullScanRequest, scan: ScanSpec, trajectory: dict[str, Any]) -> ScanResult:
        prefix = _prefix(trajectory)
        execution = self._execution[prefix]
        if execution.get("mode") == "measurement":
            return self._measured_profile(request, scan, execution)
        deadline = time.monotonic() + self.file_timeout
        previous = None
        partial = None
        last_error = None
        while time.monotonic() < deadline:
            files = [path for path in self.data_root.glob(f"{prefix}*.dat.bt7")
                     if str(path.resolve()) not in execution["baseline"]]
            if len(files) > 1:
                raise ValueError("ambiguous native files for one unique trajectory prefix")
            if len(files) == 1:
                path = files[0]
                stat = _stat(path)
                if stat[3] / 1e9 < execution["started_at"] - 5:
                    raise ValueError("native candidate predates this trajectory execution")
                if previous != stat:
                    partial = None
                if previous == stat:
                    try:
                        result = self._read_file(path, request, scan, trajectory, execution)
                        if result.status == "complete":
                            return result
                        partial = result
                    except (ValueError, KeyError) as exc:
                        partial = None
                        last_error = exc
                previous = stat
            else:
                previous = None
                partial = None
            time.sleep(self.file_poll)
        if partial is not None:
            # Revalidate the final file, not a previously valid shorter snapshot.
            path = Path(partial.artifacts[0].path)
            if _stat(path) == previous:
                return self._read_file(path, request, scan, trajectory, execution)
        raise ValueError(f"no uniquely matched complete native profile: {last_error or 'writer timeout'}")
