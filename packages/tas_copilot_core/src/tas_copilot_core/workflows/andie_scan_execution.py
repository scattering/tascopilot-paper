"""Full-profile execution boundary for an order-parameter experiment.

This module owns transport and validation, never fitting or instrument control.
The NICE backend only exchanges versioned JSON with a separately operated owner.
It cannot launch NICE, connect to a host, or execute the legacy point server.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator
from tas_copilot_contracts.base import ContractModel


PROTOCOL = "tascopilot.full-scan.v1"
Role = Literal["inference", "diagnostic"]
ComponentRole = Literal["inference", "nuisance", "diagnostic"]
Status = Literal["planned", "running", "complete", "partial", "refused", "failed",
                 "blocked", "unsupported", "awaiting_owner", "cancel_unknown",
                 "cancelled", "unknown"]
TERMINAL = {"complete", "partial", "refused", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or value in {".", ".."}:
        raise ValueError("IDs must be safe nonempty path components")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("numeric values must be finite")
    return value


class ScanSpec(ContractModel):
    scan_id: str
    reflection: str
    role: Role
    axis: str
    axis_unit: str
    frame_id: str
    points: list[dict[str, float]] = Field(min_length=1)
    component_roles: dict[str, ComponentRole] = Field(min_length=1)
    count_mode: Literal["time", "monitor"] = "time"
    preset: float = Field(gt=0, allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _scan_id = field_validator("scan_id")(_identifier)

    @model_validator(mode="after")
    def valid_profile(self) -> "ScanSpec":
        if not all((self.reflection, self.axis, self.axis_unit, self.frame_id)):
            raise ValueError("reflection, axis, unit and frame are required")
        for point in self.points:
            if self.axis not in point or not point:
                raise ValueError("every grid row must contain the scan axis")
            for value in point.values():
                _finite(value)
        if self.role == "diagnostic" and "inference" in self.component_roles.values():
            raise ValueError("diagnostic scans cannot contain inference components")
        if self.role == "inference" and "inference" not in self.component_roles.values():
            raise ValueError("inference scans require an identified inference component")
        return self


class FullScanRequest(ContractModel):
    protocol: Literal["tascopilot.full-scan.v1"] = PROTOCOL
    run_id: str
    action_id: str
    temperature_k: float = Field(gt=0, allow_inf_nan=False)
    selected_temperature_index: int = Field(ge=1)
    sample_state: dict[str, Any] = Field(min_length=1)
    lattice: dict[str, Any] = Field(min_length=1)
    frames: dict[str, Any] = Field(min_length=1)
    instrument_config: dict[str, Any] = Field(min_length=1)
    settling: dict[str, Any] = Field(default_factory=dict)
    scans: list[ScanSpec] = Field(min_length=1)
    provenance: list[dict[str, Any]] = Field(default_factory=list)

    _ids = field_validator("run_id", "action_id")(_identifier)

    @field_validator("settling", mode="before")
    @classmethod
    def canonical_settling(cls, value):
        result = dict(value or {})
        if "tolerance_k" in result:
            old = result.pop("tolerance_k")
            if "temperature_tolerance_k" in result and result["temperature_tolerance_k"] != old:
                raise ValueError("conflicting temperature tolerance aliases")
            result["temperature_tolerance_k"] = old
        for name in ("temperature_tolerance_k", "hold_seconds"):
            if name in result and (_finite(float(result[name])) < 0):
                raise ValueError(f"{name} cannot be negative")
        return result

    @model_validator(mode="after")
    def unique_scans(self) -> "FullScanRequest":
        ids = [scan.scan_id for scan in self.scans]
        if len(ids) != len(set(ids)):
            raise ValueError("scan IDs must be unique within the action")
        if any(scan.frame_id not in self.frames for scan in self.scans):
            raise ValueError("every scan must reference an explicit frame")
        return self


class ScanRow(ContractModel):
    point_index: int = Field(ge=0)
    requested: dict[str, float] = Field(min_length=1)
    achieved: dict[str, float] = Field(min_length=1)
    temperature_k: float = Field(gt=0, allow_inf_nan=False)
    counts: float = Field(ge=0, allow_inf_nan=False)
    monitor: float = Field(gt=0, allow_inf_nan=False)
    live_time: float = Field(gt=0, allow_inf_nan=False)
    timestamp: str
    status: Literal["counted", "invalid"] = "counted"
    detector_segments: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def finite_readbacks(self) -> "ScanRow":
        for value in [*self.requested.values(), *self.achieved.values(), *self.detector_segments]:
            _finite(value)
        if any(value < 0 for value in self.detector_segments):
            raise ValueError("detector segments cannot be negative")
        if not self.timestamp:
            raise ValueError("timestamp is required")
        return self


class RawArtifact(ContractModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["raw", "simulation", "replay"]


class ScanResult(ContractModel):
    scan_id: str
    status: Literal["complete", "partial", "refused", "failed", "cancelled"]
    rows: list[ScanRow] = Field(default_factory=list)
    artifacts: list[RawArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FullScanResult(ContractModel):
    protocol: Literal["tascopilot.full-scan.v1"] = PROTOCOL
    run_id: str
    action_id: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str
    backend: str
    status: Status
    scans: list[ScanResult] = Field(default_factory=list)
    reason: str | None = None
    updated_at: str = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


def request_hash(request: FullScanRequest) -> str:
    data = json.dumps(request.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def validate_result(request: FullScanRequest, result: FullScanResult, attempt_id: str) -> None:
    """Reconcile identity and all profile rows; complete never means last-row-only."""
    if (result.run_id, result.action_id, result.request_hash, result.attempt_id) != (
            request.run_id, request.action_id, request_hash(request), attempt_id):
        raise ValueError("response identity or request hash does not match")
    specs = {scan.scan_id: scan for scan in request.scans}
    seen: set[str] = set()
    for scan in result.scans:
        if scan.scan_id not in specs or scan.scan_id in seen:
            raise ValueError("unknown or duplicate scan in response")
        seen.add(scan.scan_id)
        spec = specs[scan.scan_id]
        indices: set[int] = set()
        for row in scan.rows:
            if row.point_index in indices or row.point_index >= len(spec.points):
                raise ValueError("duplicate or out-of-range point index")
            indices.add(row.point_index)
            if row.requested != spec.points[row.point_index]:
                raise ValueError("response requested grid differs from immutable request")
            if not set(row.requested).issubset(row.achieved):
                raise ValueError("missing achieved coordinate readbacks")
        if scan.status == "complete" and (indices != set(range(len(spec.points))) or
                                            any(row.status != "counted" for row in scan.rows)):
            raise ValueError("complete scan must contain every valid grid row")
    if result.status == "complete" and (seen != set(specs) or
                                          any(scan.status != "complete" for scan in result.scans)):
        raise ValueError("complete action requires every requested scan to be complete")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _result(request: FullScanRequest, attempt_id: str, backend: str,
            status: Status, **kwargs: Any) -> FullScanResult:
    return FullScanResult(run_id=request.run_id, action_id=request.action_id,
                          request_hash=request_hash(request), attempt_id=attempt_id,
                          backend=backend, status=status, **kwargs)


class ScanBackend(Protocol):
    name: str

    def submit(self, request: FullScanRequest, directory: Path, attempt_id: str) -> FullScanResult: ...
    def reconcile(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult: ...
    def cancel(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult: ...


class ScanExecutor:
    """Durable single-action coordinator; ambiguous actions are never auto-retried."""
    def __init__(self, root: str | Path, backend: ScanBackend):
        self.root, self.backend = Path(root), backend

    @contextmanager
    def _action(self, request: FullScanRequest):
        directory = self.root / request.run_id / request.action_id
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saved = directory / "request.json"
                if saved.exists():
                    existing = FullScanRequest.model_validate_json(saved.read_text())
                    if request_hash(existing) != request_hash(request):
                        raise ValueError("action ID already belongs to a different request")
                else:
                    _atomic_json(saved, request.model_dump(mode="json"))
                yield directory
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self, directory: Path, request: FullScanRequest) -> FullScanResult | None:
        path = directory / "result.json"
        if not path.exists():
            return None
        result = FullScanResult.model_validate_json(path.read_text())
        validate_result(request, result, result.attempt_id)
        if result.backend != self.backend.name:
            raise ValueError("action was registered with a different backend")
        return result

    def _save(self, directory: Path, request: FullScanRequest, result: FullScanResult,
              attempt_id: str) -> FullScanResult:
        validate_result(request, result, attempt_id)
        if result.backend != self.backend.name:
            raise ValueError("backend identity differs from registered backend")
        _atomic_json(directory / "result.json", result.model_dump(mode="json"))
        return result

    def submit(self, request: FullScanRequest) -> FullScanResult:
        request = FullScanRequest.model_validate(request.model_dump(mode="json"))
        with self._action(request) as directory:
            previous = self._read(directory, request)
            if previous is not None:
                # Never run the action again, including after a crash/timeout.
                return previous
            attempt = uuid4().hex
            running = _result(request, attempt, self.backend.name, "running")
            self._save(directory, request, running, attempt)
            try:
                result = self.backend.submit(request, directory, attempt)
                return self._save(directory, request, result, attempt)
            except Exception as exc:
                return self._save(directory, request, _result(
                    request, attempt, self.backend.name, "unknown",
                    reason=f"submission outcome requires reconciliation: {type(exc).__name__}: {exc}"), attempt)

    def reconcile(self, request: FullScanRequest) -> FullScanResult:
        with self._action(request) as directory:
            previous = self._read(directory, request)
            if previous is None:
                raise ValueError("action has not been submitted")
            if previous.status in TERMINAL:
                return previous
            try:
                result = self.backend.reconcile(request, directory, previous)
                return self._save(directory, request, result, previous.attempt_id)
            except Exception as exc:
                return self._save(directory, request, previous.model_copy(update={
                    "status": "unknown", "updated_at": _now(),
                    "reason": f"response rejected: {type(exc).__name__}: {exc}"}), previous.attempt_id)

    def request_cancel(self, request: FullScanRequest) -> FullScanResult:
        with self._action(request) as directory:
            previous = self._read(directory, request)
            if previous is None:
                raise ValueError("action has not been submitted")
            if previous.status in TERMINAL:
                return previous
            # Persist intent before transport; a crash must not forget cancellation.
            previous = previous.model_copy(update={"status": "cancel_unknown", "updated_at": _now(),
                "metadata": {**previous.metadata, "cancel_requested": True},
                "reason": "cancellation requested; owner confirmation pending"})
            self._save(directory, request, previous, previous.attempt_id)
            try:
                result = self.backend.cancel(request, directory, previous)
            except Exception as exc:
                result = previous.model_copy(update={"reason": f"cancellation unconfirmed: {exc}"})
            return self._save(directory, request, result, previous.attempt_id)


ProfileGenerator = Callable[[FullScanRequest, ScanSpec, int, dict[str, float]], dict[str, Any]]


class SimulatorBackend:
    """Injected simulator returns detector rows, never fitted/oracle peak areas."""
    name = "simulator"

    def __init__(self, profile_generator: ProfileGenerator):
        self.profile_generator = profile_generator

    def submit(self, request: FullScanRequest, directory: Path, attempt_id: str) -> FullScanResult:
        scans = []
        for spec in request.scans:
            rows = []
            for index, nodes in enumerate(spec.points):
                raw = self.profile_generator(request, spec, index, dict(nodes))
                rows.append(ScanRow.model_validate({
                    "point_index": index, "requested": dict(nodes), "achieved": dict(nodes),
                    "temperature_k": request.temperature_k, "timestamp": _now(), **raw}))
            path = directory / f"raw_{spec.scan_id}.json"
            _atomic_json(path, {"protocol": PROTOCOL, "request_hash": request_hash(request),
                                "scan_id": spec.scan_id, "source": "simulation",
                                "rows": [row.model_dump(mode="json") for row in rows]})
            artifact = RawArtifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   role="simulation")
            scans.append(ScanResult(scan_id=spec.scan_id, status="complete", rows=rows, artifacts=[artifact]))
        return _result(request, attempt_id, self.name, "complete", scans=scans,
                       metadata={"source_kind": "simulation", "area_fitting": "external"})

    def reconcile(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult:
        return previous.model_copy(update={"status": "unknown", "updated_at": _now(),
            "reason": "interrupted simulator action; inspect raw artifacts before an explicit new action"})

    def cancel(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult:
        return previous.model_copy(update={"status": "cancel_unknown", "updated_at": _now(),
                                           "reason": "no active operation handle; cancellation is unconfirmed"})


class ReplayBackend(SimulatorBackend):
    """Replay archived point rows; no fitted areas and no synthetic substitutions."""
    name = "replay"

    def __init__(self, scans: list[ScanResult], provenance: dict[str, Any] | None = None):
        self.scans = [ScanResult.model_validate(scan.model_dump(mode="json")) for scan in scans]
        self.provenance = dict(provenance or {})

    def submit(self, request: FullScanRequest, directory: Path, attempt_id: str) -> FullScanResult:
        status: Status = "complete" if self.scans and all(s.status == "complete" for s in self.scans) else "partial"
        if self.scans and all(s.status == "refused" for s in self.scans):
            status = "refused"
        return _result(request, attempt_id, self.name, status, scans=self.scans,
                       metadata={"source_kind": "replay", "provenance": self.provenance})


class NiceSpoolBackend:
    """Passive full-scan transport. No connections, processes, hardware or STOP files.

    An independently authorized owner must advertise this protocol in
    capabilities.json and consume its native full_scan requests. Responses need
    an increasing integer response_seq and the matching owner/attempt/hash.
    The legacy one-point server does neither, and is explicitly unsupported.
    """
    name = "nice_spool"

    def __init__(self, spool_dir: str | Path, *, enabled: bool = False):
        self.spool = Path(spool_dir)
        self.enabled = enabled

    def _capabilities(self) -> dict[str, Any] | None:
        path = self.spool / "capabilities.json"
        if not path.exists():
            return None
        caps = json.loads(path.read_text())
        if (caps.get("protocol") != PROTOCOL or caps.get("full_scan") is not True or
                not isinstance(caps.get("owner_id"), str) or not caps["owner_id"]):
            return None
        return caps

    def submit(self, request: FullScanRequest, directory: Path, attempt_id: str) -> FullScanResult:
        if not self.enabled:
            return _result(request, attempt_id, self.name, "blocked", reason="NICE spool transport is disabled")
        caps = self._capabilities()
        if caps is None:
            return _result(request, attempt_id, self.name, "unsupported",
                           reason="no matching full-scan owner; legacy point server cannot execute this request")
        context = {"owner_id": caps["owner_id"], "stem": f"{request_hash(request)[:24]}_{attempt_id}",
                   "cancel_supported": caps.get("cancel") is True}
        _atomic_json(directory / "transport.json", context)
        envelope = {"protocol": PROTOCOL, "kind": "full_scan", "owner_id": caps["owner_id"],
                    "attempt_id": attempt_id, "request_hash": request_hash(request),
                    "request": request.model_dump(mode="json")}
        _atomic_json(self.spool / f"request_{context['stem']}.json", envelope)
        return _result(request, attempt_id, self.name, "awaiting_owner",
                       reason="full scan request written; owner execution is not yet confirmed")

    def reconcile(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult:
        context_path = directory / "transport.json"
        if not context_path.exists():
            # These states prove that nothing was dispatched. Explicit reconcile
            # may proceed after caller opt-in or owner capability installation.
            if previous.status in {"blocked", "unsupported"} and self.enabled:
                return self.submit(request, directory, previous.attempt_id)
            return previous
        context = json.loads(context_path.read_text())
        response = self.spool / f"response_{context['stem']}.json"
        if not response.exists():
            return previous
        envelope = json.loads(response.read_text())
        if (envelope.get("protocol") != PROTOCOL or envelope.get("owner_id") != context["owner_id"] or
                envelope.get("attempt_id") != previous.attempt_id or
                envelope.get("request_hash") != request_hash(request)):
            raise ValueError("stale or mismatched owner response envelope")
        sequence = envelope.get("response_seq")
        if type(sequence) is not int or sequence < 1:
            raise ValueError("owner response_seq must be an integer >= 1")
        fingerprint = hashlib.sha256(json.dumps(envelope, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        last_sequence = previous.metadata.get("owner_response_seq", 0)
        if sequence < last_sequence:
            return previous
        if sequence == last_sequence:
            if fingerprint != previous.metadata.get("owner_response_hash"):
                raise ValueError("owner reused response_seq with changed contents")
            return previous
        result = FullScanResult.model_validate(envelope["result"])
        validate_result(request, result, previous.attempt_id)
        # Raw observations are append-only, including when reporting cancellation.
        rows = {(scan.scan_id, row.point_index): row.model_dump(mode="json")
                for scan in result.scans for row in scan.rows}
        for scan in previous.scans:
            for row in scan.rows:
                if rows.get((scan.scan_id, row.point_index)) != row.model_dump(mode="json"):
                    raise ValueError("owner response discarded or changed previously observed raw rows")
        metadata = {**result.metadata, "owner_response_seq": sequence,
                    "owner_response_hash": fingerprint}
        if previous.metadata.get("cancel_requested"):
            metadata["cancel_requested"] = True
            if result.status not in TERMINAL:
                result = result.model_copy(update={"status": "cancel_unknown",
                    "reason": "cancellation still awaits matching owner confirmation"})
        return result.model_copy(update={"metadata": metadata})

    def cancel(self, request: FullScanRequest, directory: Path, previous: FullScanResult) -> FullScanResult:
        context_path = directory / "transport.json"
        if self.enabled and context_path.exists():
            context = json.loads(context_path.read_text())
            if context.get("cancel_supported"):
                _atomic_json(self.spool / f"cancel_{context['stem']}.json", {
                    "protocol": PROTOCOL, "kind": "cancel", "owner_id": context["owner_id"],
                    "attempt_id": previous.attempt_id, "request_hash": request_hash(request),
                    "run_id": request.run_id, "action_id": request.action_id})
        return previous.model_copy(update={"status": "cancel_unknown", "updated_at": _now(),
            "reason": "cancellation requires matching owner confirmation; STOP/drain is not cancellation"})


def native_scan_trajectory(request: FullScanRequest, scan: ScanSpec) -> dict[str, Any]:
    """Compile an explicit full grid using the existing NICE list/vary syntax.

    Native node aliases and initialization are required configuration, never
    inferred from a cubic lattice or UI labels. Temperature/state preparation
    and trajectory preflight belong to the injected authorized owner callbacks.
    """
    config = request.instrument_config
    node_map = config.get("nice_node_map")
    initialization = config.get("nice_init")
    y_axis = config.get("nice_y_axis")
    if not isinstance(node_map, dict) or not isinstance(initialization, dict) or not y_axis:
        raise ValueError("NICE requires explicit nice_node_map, nice_init and nice_y_axis")
    axes = list(scan.points[0])
    if any(set(point) != set(axes) for point in scan.points):
        raise ValueError("native trajectory grid must specify the same axes in every row")
    names = [node_map.get(axis) for axis in axes] + list(initialization) + [y_axis]
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name) for name in names):
        raise ValueError("native node mappings must contain explicit safe node names")
    if len({node_map[axis] for axis in axes}) != len(axes):
        raise ValueError("different axes cannot map to the same native node")
    preset_node = "counter.timePreset" if scan.count_mode == "time" else "counter.monitorPreset"
    if scan.count_mode == "monitor" and not scan.preset.is_integer():
        raise ValueError("monitor preset must be integral")
    state_id = request.sample_state.get("state_id")
    if type(state_id) is not int or state_id < 0:
        raise ValueError("an explicit integer sample_state.state_id is required")
    if "sampleStateID" in initialization and float(initialization["sampleStateID"]) != state_id:
        raise ValueError("nice_init sampleStateID contradicts the requested sample state")
    init = [["sampleStateID", state_id]]
    init += [[node, _finite(float(value))] for node, value in initialization.items() if node != "sampleStateID"]
    init += [["counter.countAgainst", repr(scan.count_mode.upper())], [preset_node, scan.preset]]
    # Readable names (<sample>_<reflection>_tt_T<temp>K) so a series sorts and
    # reads without a lookup table. The short hash keeps each request/scan
    # prefix unique, which the native-file matcher relies on.
    sample = re.sub(r"[^A-Za-z0-9]", "", str(request.sample_state.get("name") or "sample")) or "sample"
    reflection = re.sub(r"[^A-Za-z0-9]", "", scan.reflection) or "scan"
    temp_tag = f"{request.temperature_k:g}".replace(".", "p").replace("-", "m")
    digest = hashlib.sha256((request_hash(request) + scan.scan_id).encode()).hexdigest()[:8]
    prefix = f"{sample}_{reflection}_tt_T{temp_tag}K_{digest}"
    return {"description": f"{scan.reflection} full profile at {request.temperature_k:g} K",
            "trajName": repr(prefix), "filePrefix": repr(prefix),
            "xAxis": node_map[scan.axis], "yAxis": y_axis, "init": init,
            "loops": [{"vary": [[node_map[axis], {"list": {
                "value": [point[axis] for point in scan.points], "cyclic": False}}] for axis in axes]}]}


class NiceScanOwner:
    """Gated owner-side protocol dispatcher with injected NICE integration.

    execute_trajectory must be a callable bound to an existing authorized NICE
    Task context, with the verified run_trajectory_contents(traj, wait=True)
    signature. This helper never imports, connects to, or starts NICE.
    read_profile must return the complete raw rows and source-file hashes.
    Cancellation is cooperative BETWEEN scans; no hardware abort API is guessed.
    A process restart with ambiguous work requires external reconciliation.
    """
    def __init__(self, spool_dir: str | Path, *, owner_id: str,
                 execute_trajectory: Callable[..., Any],
                 read_profile: Callable[[FullScanRequest, ScanSpec, dict[str, Any]], ScanResult],
                 prepare_action: Callable[[FullScanRequest], dict[str, Any]],
                 validate_trajectory: Callable[[FullScanRequest, ScanSpec, dict[str, Any]], bool],
                 enabled: bool = False):
        self.spool = Path(spool_dir)
        self.owner_id = _identifier(owner_id)
        self.execute_trajectory = execute_trajectory
        self.read_profile = read_profile
        self.prepare_action = prepare_action
        self.validate_trajectory = validate_trajectory
        self.enabled = enabled

    def advertise(self) -> bool:
        if not self.enabled:
            return False
        _atomic_json(self.spool / "capabilities.json", {
            "protocol": PROTOCOL, "owner_id": self.owner_id, "full_scan": True,
            "cancel": True, "cancel_mode": "between_scans", "inflight_abort": False})
        return True

    def _cancelled(self, stem: str, request: FullScanRequest, attempt: str) -> bool:
        path = self.spool / f"cancel_{stem}.json"
        if not path.exists():
            return False
        payload = json.loads(path.read_text())
        return (payload.get("protocol"), payload.get("owner_id"), payload.get("attempt_id"),
                payload.get("request_hash"), payload.get("run_id"), payload.get("action_id")) == (
                    PROTOCOL, self.owner_id, attempt, request_hash(request), request.run_id, request.action_id)

    def process_one(self) -> FullScanResult | None:
        """Synchronously dispatch one request; no hidden service/event loop."""
        if not self.enabled:
            return None
        for path in sorted(self.spool.glob("request_*.json")):
            envelope = json.loads(path.read_text())
            if envelope.get("protocol") != PROTOCOL or envelope.get("owner_id") != self.owner_id:
                continue
            if envelope.get("kind") != "full_scan":
                continue
            request = FullScanRequest.model_validate(envelope["request"])
            attempt = _identifier(envelope["attempt_id"])
            if envelope.get("request_hash") != request_hash(request):
                raise ValueError("owner rejected an inconsistent request hash")
            stem = path.stem.removeprefix("request_")
            with (self.spool / f"owner_{stem}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    response_path = self.spool / f"response_{stem}.json"
                    previous_envelope = json.loads(response_path.read_text()) if response_path.exists() else None
                    previous = FullScanResult.model_validate(previous_envelope["result"]) if previous_envelope else None
                    if previous:
                        validate_result(request, previous, attempt)
                        if previous.status in TERMINAL:
                            continue
                    sequence = previous_envelope["response_seq"] if previous_envelope else 0

                    preparation = {}
                    def publish(status, scans, reason=None, **metadata):
                        nonlocal sequence
                        sequence += 1
                        result = _result(request, attempt, "nice_spool", status, scans=list(scans),
                            reason=reason, metadata={"owner_id": self.owner_id, "source_kind": "nice",
                                                     "cancel_mode": "between_scans",
                                                     "preparation": preparation, **metadata})
                        validate_result(request, result, attempt)
                        _atomic_json(response_path, {"protocol": PROTOCOL, "owner_id": self.owner_id,
                            "attempt_id": attempt, "request_hash": request_hash(request),
                            "response_seq": sequence, "result": result.model_dump(mode="json")})
                        return result

                    if previous:
                        return publish("unknown", previous.scans,
                            "existing nonterminal owner action; reconcile instrument/files before any retry")
                    scans: list[ScanResult] = []
                    if self._cancelled(stem, request, attempt):
                        return publish("cancelled", scans, "cancelled before dispatch; no trajectory started")
                    try:
                        trajectories = [native_scan_trajectory(request, spec) for spec in request.scans]
                        if not all(self.validate_trajectory(request, spec, trajectory) is True
                                   for spec, trajectory in zip(request.scans, trajectories)):
                            return publish("refused", scans, "owner trajectory preflight refused the action")
                    except Exception as exc:
                        return publish("refused", scans, f"native trajectory validation failed: {exc}")
                    # Persist before any preparation callback can touch instrument state.
                    publish("running", scans, stage="preparing")
                    try:
                        prepared = self.prepare_action(request)
                        preparation = dict(prepared)
                        if prepared.get("ready") is not True:
                            return publish("refused", scans, "temperature/state preparation not confirmed")
                        for field in ("sample_state", "lattice", "frames"):
                            if prepared.get(field) != getattr(request, field):
                                return publish("refused", scans, f"prepared {field} differs from request")
                        for spec, trajectory in zip(request.scans, trajectories):
                            if self._cancelled(stem, request, attempt):
                                return publish("cancelled", scans, "cancelled between complete scans; no in-flight abort")
                            publish("running", scans, stage="trajectory", inflight_scan_id=spec.scan_id)
                            self.execute_trajectory(trajectory, wait=True)
                            scan = ScanResult.model_validate(self.read_profile(request, spec, trajectory))
                            if scan.scan_id != spec.scan_id or not scan.artifacts:
                                raise ValueError("raw reader must return matching scan ID and hashed source artifacts")
                            validate_result(request, _result(request, attempt, "nice_spool", "running",
                                                             scans=[*scans, scan]), attempt)
                            scans.append(scan)
                            if scan.status != "complete":
                                return publish("partial", scans, "profile was incomplete; remaining scans not dispatched")
                            publish("running", scans, stage="between_scans")
                        return publish("complete", scans)
                    except Exception as exc:
                        return publish("unknown", scans, f"owner execution outcome requires reconciliation: {exc}")
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        return None
