from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tas_copilot_core.workflows.andie_scan_execution import (
    PROTOCOL, FullScanRequest, FullScanResult, NiceSpoolBackend, RawArtifact,
    ReplayBackend, ScanExecutor, ScanResult, ScanRow, ScanSpec, SimulatorBackend,
    NiceScanOwner, native_scan_trajectory, request_hash,
)


def request() -> FullScanRequest:
    return FullScanRequest(
        run_id="run1", action_id="T105", temperature_k=105,
        selected_temperature_index=5,
        sample_state={"name": "SyntheticSample", "state_id": 2},
        lattice={"a": 4.08, "c": 5.70, "angles_deg": [90, 90, 120], "unit": "angstrom"},
        frames={"film": {"coordinate": "motor_readback", "units": "degree"}},
        instrument_config={"Ei_meV": 14.7, "Ef_meV": 14.7, "filters": ["PG", "PG", "PG"]},
        settling={"tolerance_k": 0.1, "hold_seconds": 180},
        scans=[ScanSpec(
            scan_id="s001", reflection="001", role="inference", axis="A4",
            axis_unit="degree", frame_id="film", preset=30,
            points=[{"A4": x, "A3": x / 2} for x in [23.0, 23.5, 24.0]],
            component_roles={"001": "inference"}),
            ScanSpec(scan_id="s101", reflection="101", role="inference", axis="A4",
                     axis_unit="degree", frame_id="film", preset=30,
                     points=[{"A4": x, "A3": 81.0 + (x - 46) / 2} for x in [43.5, 46.0, 49.0]],
                     component_roles={"lower": "diagnostic", "central": "inference", "upper": "diagnostic"}),
            ScanSpec(scan_id="s100", reflection="100", role="diagnostic", axis="A4",
                     axis_unit="degree", frame_id="film", preset=30,
                     points=[{"A4": 38.0, "A3": 109.0}],
                     component_roles={"100": "diagnostic"})])


def generator(req, scan, index, nodes):
    # Deterministic synthetic raw multiplet, not a peak-area oracle.
    import math
    counts = 20 + sum(100 * math.exp(-0.5 * ((nodes[scan.axis] - c) / .3) ** 2)
                      for c in ([23.5] if scan.reflection == "001" else [43.5, 46, 49]))
    return {"counts": round(counts), "monitor": 39048, "live_time": scan.preset}


def prepare_owner(spool: Path, **updates):
    spool.mkdir(parents=True, exist_ok=True)
    caps = {"protocol": PROTOCOL, "owner_id": "owner-1", "full_scan": True, "cancel": True, **updates}
    (spool / "capabilities.json").write_text(json.dumps(caps))


def owner_response(spool: Path, current: FullScanResult, *, status="complete", scans=None,
                   response_seq=1, **changes):
    outgoing = next(spool.glob("request_*.json"))
    envelope = json.loads(outgoing.read_text())
    req = FullScanRequest.model_validate(envelope["request"])
    if scans is None:
        scans = [ScanResult(scan_id=s.scan_id, status="complete", rows=[ScanRow(
            point_index=i, requested=p, achieved=p, temperature_k=req.temperature_k,
            counts=100, monitor=39048, live_time=30, timestamp="2026-10-31T00:00:00Z")
            for i, p in enumerate(s.points)]) for s in req.scans]
    result = current.model_copy(update={"status": status, "scans": scans})
    response = {"protocol": PROTOCOL, "owner_id": "owner-1", "attempt_id": current.attempt_id,
                "response_seq": response_seq,
                "request_hash": current.request_hash, "result": result.model_dump(mode="json"), **changes}
    response_path = spool / outgoing.name.replace("request_", "response_", 1)
    response_path.write_text(json.dumps(response))
    return response_path


def test_simulator_returns_all_raw_rows_and_preserves_multiplet_request(tmp_path):
    req = request()
    executor = ScanExecutor(tmp_path / "actions", SimulatorBackend(generator))
    result = executor.submit(req)
    assert result.status == "complete"
    assert [len(s.rows) for s in result.scans] == [3, 3, 1]
    assert [r.point_index for r in result.scans[0].rows] == [0, 1, 2]
    assert result.scans[0].rows[1].counts > result.scans[0].rows[0].counts
    for scan in result.scans:
        artifact = scan.artifacts[0]
        assert hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest() == artifact.sha256
        assert artifact.role == "simulation"
        assert "area" not in json.loads(Path(artifact.path).read_text())
    saved = FullScanRequest.model_validate_json((tmp_path / "actions/run1/T105/request.json").read_text())
    assert saved == req
    assert saved.scans[1].component_roles["upper"] == "diagnostic"


def test_submit_and_restart_do_not_repeat_acquisition(tmp_path):
    calls = []
    def counted(*args):
        calls.append(args[2])
        return generator(*args)
    req = request()
    first = ScanExecutor(tmp_path, SimulatorBackend(counted)).submit(req)
    second = ScanExecutor(tmp_path, SimulatorBackend(counted)).submit(req)
    assert second == first
    assert len(calls) == 7
    changed = req.model_copy(deep=True)
    changed.scans[0].points[0]["A4"] += .1
    with pytest.raises(ValueError, match="different request"):
        ScanExecutor(tmp_path, SimulatorBackend(counted)).submit(changed)


def test_interrupted_simulator_requires_reconciliation_not_retry(tmp_path):
    calls = []
    def broken(*args):
        calls.append(1)
        raise RuntimeError("process lost")
    executor = ScanExecutor(tmp_path, SimulatorBackend(broken))
    first = executor.submit(request())
    assert first.status == "unknown"
    assert executor.submit(request()) == first
    assert len(calls) == 1
    assert executor.reconcile(request()).status == "unknown"
    assert executor.request_cancel(request()).status == "cancel_unknown"


def test_replay_profiles_without_refit_or_normalization(tmp_path):
    original = ScanExecutor(tmp_path / "simulation", SimulatorBackend(generator)).submit(request())
    replay = ScanExecutor(tmp_path / "replay", ReplayBackend(original.scans, {"snapshot": "fixture"})).submit(request())
    assert replay.status == "complete"
    assert replay.scans == original.scans
    assert replay.backend == "replay"
    assert replay.metadata["provenance"] == {"snapshot": "fixture"}


def test_refused_replay_is_not_success(tmp_path):
    scans = [ScanResult(scan_id=s.scan_id, status="refused") for s in request().scans]
    result = ScanExecutor(tmp_path, ReplayBackend(scans)).submit(request())
    assert result.status == "refused"


def test_nice_disabled_by_default_creates_no_spool_request(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    result = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool)).submit(request())
    assert result.status == "blocked"
    assert not list(spool.glob("request_*"))


@pytest.mark.parametrize("caps", [None, {"protocol": "legacy-point-v0"}, {"full_scan": False}])
def test_missing_or_legacy_owner_is_explicitly_unsupported(tmp_path, caps):
    spool = tmp_path / "spool"
    if caps is not None:
        prepare_owner(spool, **caps)
    result = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True)).submit(request())
    assert result.status == "unsupported"
    assert not list(spool.glob("request_*"))


def test_full_scan_spool_roundtrip_and_restart(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    assert pending.status == "awaiting_owner"
    outgoing = json.loads(next(spool.glob("request_*.json")).read_text())
    assert outgoing["kind"] == "full_scan"
    assert outgoing["request"]["scans"][1]["points"] == request().scans[1].points
    assert executor.reconcile(request()).status == "awaiting_owner"
    owner_response(spool, pending)
    restarted = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    completed = restarted.reconcile(request())
    assert completed.status == "complete"
    assert completed.attempt_id == pending.attempt_id
    assert restarted.submit(request()) == completed
    assert len(list(spool.glob("request_*"))) == 1


@pytest.mark.parametrize("change", [{"attempt_id": "stale"}, {"owner_id": "wrong-owner"},
                                    {"request_hash": "0" * 64}, {"protocol": "legacy"}])
def test_stale_or_mismatched_response_never_completes(tmp_path, change):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    owner_response(spool, pending, **change)
    rejected = executor.reconcile(request())
    assert rejected.status == "unknown"
    assert "mismatched" in rejected.reason
    assert executor.submit(request()).status == "unknown"
    assert len(list(spool.glob("request_*"))) == 1


@pytest.mark.parametrize("fault", ["last-row-only", "duplicate-index", "wrong-grid", "missing-scan", "invalid-row"])
def test_claimed_completion_reconciles_every_scan_and_row(tmp_path, fault):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    path = owner_response(spool, pending)
    response = json.loads(path.read_text())
    scans = response["result"]["scans"]
    if fault == "last-row-only":
        scans[0]["rows"] = scans[0]["rows"][-1:]
    elif fault == "duplicate-index":
        scans[0]["rows"][1]["point_index"] = 0
    elif fault == "wrong-grid":
        scans[0]["rows"][0]["requested"]["A4"] += .1
    elif fault == "missing-scan":
        scans.pop()
    else:
        scans[0]["rows"][0]["status"] = "invalid"
    path.write_text(json.dumps(response))
    assert executor.reconcile(request()).status == "unknown"


def test_partial_and_refused_results_are_explicit(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    owner_response(spool, pending, status="refused", scans=[])
    assert executor.reconcile(request()).status == "refused"


@pytest.mark.parametrize("supported", [True, False])
def test_cancel_is_unknown_until_owner_confirms_and_never_uses_stop(tmp_path, supported):
    spool = tmp_path / "spool"
    prepare_owner(spool, cancel=supported)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    assert executor.request_cancel(request()).status == "cancel_unknown"
    assert bool(list(spool.glob("cancel_*"))) == supported
    assert not (spool / "STOP").exists()
    assert executor.reconcile(request()).status == "cancel_unknown"
    owner_response(spool, pending, status="cancelled", scans=[])
    assert executor.reconcile(request()).status == "cancelled"


def test_backend_identity_cannot_change_on_restart(tmp_path):
    req = request()
    result = ScanExecutor(tmp_path, SimulatorBackend(generator)).submit(req)
    with pytest.raises(ValueError, match="different backend"):
        ScanExecutor(tmp_path, ReplayBackend(result.scans)).submit(req)


def test_known_undispatched_request_can_reconcile_after_enablement(tmp_path):
    spool = tmp_path / "spool"
    req = request()
    blocked = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool)).submit(req)
    assert blocked.status == "blocked"
    enabled = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    assert enabled.reconcile(req).status == "unsupported"
    prepare_owner(spool)
    pending = enabled.reconcile(req)
    assert pending.status == "awaiting_owner"
    assert pending.attempt_id == blocked.attempt_id
    assert len(list(spool.glob("request_*"))) == 1


def test_maximal_valid_ids_have_bounded_spool_names(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    req = request().model_copy(update={"run_id": "r" * 128, "action_id": "a" * 128})
    result = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True)).submit(req)
    assert result.status == "awaiting_owner"
    assert len(next(spool.glob("request_*" )).name) < 200


def test_old_running_response_cannot_undo_cancellation(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    owner_response(spool, pending, status="running", scans=[])
    assert executor.reconcile(request()).status == "running"
    assert executor.request_cancel(request()).status == "cancel_unknown"
    assert executor.reconcile(request()).status == "cancel_unknown"
    owner_response(spool, pending, status="running", scans=[], response_seq=2)
    assert executor.reconcile(request()).status == "cancel_unknown"
    owner_response(spool, pending, status="cancelled", scans=[], response_seq=3)
    assert executor.reconcile(request()).status == "cancelled"


def test_old_response_cannot_erase_progress_and_new_rows_are_append_only(tmp_path):
    spool = tmp_path / "spool"
    prepare_owner(spool)
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    pending = executor.submit(request())
    path = owner_response(spool, pending, status="running", response_seq=2)
    progress = executor.reconcile(request())
    assert len(progress.scans) == 3
    owner_response(spool, pending, status="running", scans=[], response_seq=1)
    assert executor.reconcile(request()) == progress
    owner_response(spool, pending, status="running", scans=[], response_seq=3)
    rejected = executor.reconcile(request())
    assert rejected.status == "unknown"
    assert rejected.scans == progress.scans
    assert "raw rows" in rejected.reason


def test_partial_profile_keeps_all_available_rows(tmp_path):
    original = ScanExecutor(tmp_path / "sim", SimulatorBackend(generator)).submit(request())
    scan = original.scans[0].model_copy(update={"status": "partial", "rows": original.scans[0].rows[:2]})
    partial = ScanExecutor(tmp_path / "replay", ReplayBackend([scan])).submit(request())
    assert partial.status == "partial"
    assert len(partial.scans[0].rows) == 2


def test_contract_rejects_missing_frames_unsafe_ids_and_nan():
    payload = request().model_dump(mode="json")
    for key, value in [("run_id", "../elsewhere"), ("frames", {}), ("temperature_k", float("nan"))]:
        modified = {**payload, key: value}
        with pytest.raises(ValueError):
            FullScanRequest.model_validate(modified)
    payload["scans"][2]["component_roles"] = {"100": "inference"}
    with pytest.raises(ValueError, match="diagnostic scans"):
        FullScanRequest.model_validate(payload)


def native_request():
    req = request()
    req.instrument_config.update({"nice_node_map": {"A3": "sampleTheta.softPosition", "A4": "sampleTwoTheta.softPosition"},
                                  "nice_init": {"et.deltaE": 0.0}, "nice_y_axis": "singleDetector.liveROI"})
    return req


def prepared(req):
    return {"ready": True, "temperature_k": req.temperature_k,
            "sample_state": req.sample_state, "lattice": req.lattice, "frames": req.frames}


def raw_reader(root):
    def read(req, spec, trajectory):
        rows = [ScanRow(point_index=i, requested=point, achieved=point,
                        temperature_k=req.temperature_k, counts=120, monitor=39048,
                        live_time=30, timestamp="2026-10-31T00:00:00Z")
                for i, point in enumerate(spec.points)]
        path = root / f"{spec.scan_id}.json"
        path.write_text(json.dumps([row.model_dump(mode="json") for row in rows]))
        artifact = RawArtifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), role="raw")
        return ScanResult(scan_id=spec.scan_id, status="complete", rows=rows, artifacts=[artifact])
    return read


def test_owner_disabled_never_advertises_or_calls_callbacks(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled owner called a control callback")
    owner = NiceScanOwner(tmp_path, owner_id="owner", execute_trajectory=forbidden,
                         read_profile=forbidden, prepare_action=forbidden, validate_trajectory=forbidden)
    assert owner.advertise() is False
    assert owner.process_one() is None
    assert not (tmp_path / "capabilities.json").exists()


def test_native_trajectory_contains_every_coupled_point_and_explicit_nodes():
    req = native_request()
    req.scans[1].component_roles["lower"] = "nuisance"
    trajectory = native_scan_trajectory(req, req.scans[1])
    axes = dict(trajectory["loops"][0]["vary"])
    assert axes["sampleTwoTheta.softPosition"]["list"]["value"] == [43.5, 46, 49]
    assert axes["sampleTheta.softPosition"]["list"]["value"] == [79.75, 81, 82.5]
    assert trajectory["init"][-1] == ["counter.timePreset", 30]
    assert req.settling["temperature_tolerance_k"] == .1
    with pytest.raises(ValueError, match="explicit"):
        native_scan_trajectory(request(), request().scans[0])


def test_owner_dispatches_full_native_profiles_then_deduplicates(tmp_path):
    calls = []
    spool = tmp_path / "spool"
    owner = NiceScanOwner(spool, owner_id="owner-1", enabled=True,
                         execute_trajectory=lambda trajectory, wait: calls.append((trajectory, wait)),
                         read_profile=raw_reader(tmp_path), prepare_action=prepared,
                         validate_trajectory=lambda *args: True)
    assert owner.advertise()
    req = native_request()
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    executor.submit(req)
    assert owner.process_one().status == "complete"
    result = executor.reconcile(req)
    assert result.status == "complete"
    assert len(calls) == 3 and all(wait for _, wait in calls)
    assert all(scan.artifacts for scan in result.scans)
    assert owner.process_one() is None
    assert len(calls) == 3


def test_owner_preflight_refusal_does_not_prepare_or_execute(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("refused scan invoked a control callback")
    spool = tmp_path / "spool"
    owner = NiceScanOwner(spool, owner_id="owner-1", enabled=True,
                         execute_trajectory=forbidden, read_profile=forbidden,
                         prepare_action=forbidden, validate_trajectory=lambda *args: False)
    owner.advertise()
    req = native_request()
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    executor.submit(req)
    assert owner.process_one().status == "refused"
    assert executor.reconcile(req).status == "refused"


def test_owner_records_prepared_temperature_without_judging_it(tmp_path):
    calls = []
    spool = tmp_path / "spool"
    owner = NiceScanOwner(spool, owner_id="owner-1", enabled=True,
                         execute_trajectory=lambda *a, **kw: calls.append(1), read_profile=raw_reader(tmp_path),
                         prepare_action=lambda req: {**prepared(req), "temperature_k": 200},
                         validate_trajectory=lambda *args: True)
    owner.advertise()
    req = native_request()
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    executor.submit(req)
    # NICE settled and reported 200 K; that is data, not a reason to refuse
    assert owner.process_one().status != "refused"
    assert calls


def test_owner_cancellation_between_scans_retains_completed_profile(tmp_path):
    spool = tmp_path / "spool"
    req = native_request()
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    calls = []
    def execute(trajectory, wait):
        calls.append(trajectory)
        assert executor.request_cancel(req).status == "cancel_unknown"
    owner = NiceScanOwner(spool, owner_id="owner-1", enabled=True, execute_trajectory=execute,
                         read_profile=raw_reader(tmp_path), prepare_action=prepared,
                         validate_trajectory=lambda *args: True)
    owner.advertise()
    executor.submit(req)
    cancelled = owner.process_one()
    assert cancelled.status == "cancelled"
    assert len(calls) == 1
    assert len(cancelled.scans[0].rows) == 3
    assert executor.reconcile(req).status == "cancelled"
    assert not (spool / "STOP").exists()


def test_owner_ambiguous_restart_never_reexecutes_native_trajectory(tmp_path):
    spool = tmp_path / "spool"
    calls = []
    def broken(trajectory, wait):
        calls.append(1)
        raise RuntimeError("connection lost after dispatch")
    owner = NiceScanOwner(spool, owner_id="owner-1", enabled=True, execute_trajectory=broken,
                         read_profile=raw_reader(tmp_path), prepare_action=prepared,
                         validate_trajectory=lambda *args: True)
    owner.advertise()
    req = native_request()
    executor = ScanExecutor(tmp_path / "actions", NiceSpoolBackend(spool, enabled=True))
    executor.submit(req)
    assert owner.process_one().status == "unknown"
    assert owner.process_one().status == "unknown"
    assert calls == [1]
    assert executor.reconcile(req).status == "unknown"
