import pytest

from tas_copilot_core.workflows.andie_temperature_series import (
    PairedAreaObservation, PairedScanCost, TemperatureSeries,
)


def observation(action="t1", temperature=100.0, **extra):
    return {"action_id": action, "temperature_k": temperature,
            "areas": [27.0, 16.0], "covariance": [[4.0, 0.3], [0.3, 2.0]],
            "scan_ids": [action + "-001", action + "-101"], "coordinate": "A4",
            "normalization": "counts/4800monitor*degree", "source_hashes": ["a", "b"], **extra}


def test_paired_covariance_and_component_identity():
    PairedAreaObservation.model_validate(observation())
    for invalid in ({"covariance": [[1, 2], [2, 1]]},
                    {"covariance": [[1, .2], [.3, 1]]},
                    {"component_ids": ["001", "101_upper"]}, {"areas": [float("nan"), 1]}):
        with pytest.raises(ValueError):
            PairedAreaObservation.model_validate(observation(**invalid))


def test_cadence_restart_idempotence_and_diagnostic_invariance(tmp_path):
    path = tmp_path / "state.json"
    series = TemperatureSeries(path, "test")
    for i in range(1, 6):
        action = f"t{i}"
        series.select(action, 100.0 + i, f"scan-{i}")
        series.select(action, 100.0 + i, f"scan-{i}")
        series.accept_pair(observation(action, 100.0 + i))
        series.accept_pair(observation(action, 100.0 + i))
    assert series.state["revision"] == 5
    assert series.diagnostics_due() == ["t5"]
    series = TemperatureSeries(path, "test")
    before = series.science_hash()
    with pytest.raises(RuntimeError, match="diagnostics"):
        series.select("t6", 106, "scan-6")
    series.accept_diagnostics("t5", {"100": {"area": 1e9}, "102": {"area": -1e9}})
    assert series.science_hash() == before
    assert series.diagnostics_due() == []
    series.select("t6", 106, "scan-6")
    assert len(series.state["actions"]) == 6
    with pytest.raises(ValueError, match="different inputs"):
        series.select("t6", 107, "scan-6")


def test_only_complete_pair_and_compatible_units_can_advance(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    series.select("t1", 100, "request")
    with pytest.raises(RuntimeError, match="unfinished"):
        series.select("t2", 120, "request2")
    with pytest.raises(ValueError):
        series.accept_pair(observation(scan_ids=["one"]))
    series.accept_pair(observation())
    series.select("t2", 120, "request2")
    with pytest.raises(ValueError, match="coordinate"):
        series.accept_pair(observation("t2", 120, coordinate="A3"))


def test_actual_temperature_is_retained_with_explicit_acceptance_tolerance(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    series.select("t1", 100, "request", temperature_tolerance_k=.1)
    series.accept_pair(observation(temperature=99.98))
    assert series.science_payload()["observations"][0]["temperature_k"] == 99.98
    assert series.science_payload()["tc_bounds"] == [100, 200]


def test_stale_inference_is_rejected(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    old = series.science_hash()
    series.select("t1", 100, "request")
    series.accept_pair(observation())
    with pytest.raises(ValueError, match="stale"):
        series.accept_inference("job", old, {"posterior": "ref"})
    series.register_inference("job", "hash")
    series.accept_inference("job", series.science_hash(), {"posterior": "ref", "input_hash": "hash"})


def test_old_inference_configuration_cannot_overwrite_latest(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    science_hash = series.science_hash()
    series.register_inference("old-prior", "old")
    series.register_inference("new-prior", "new")
    series.accept_inference("new-prior", science_hash, {"input_hash": "new"})
    with pytest.raises(ValueError, match="inactive"):
        series.accept_inference("old-prior", science_hash, {"input_hash": "old"})
    assert series.state["inference"]["request_id"] == "new-prior"


def test_recorded_temperatures_are_kept_and_unique_scans_are_checked(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    series.select("t1", 150, "request")
    # off the setpoint by a kelvin: kept as recorded, never refused
    series.accept_pair(observation(temperature=150, component_temperatures_k=[149, 151]))
    series.select("t2", 155, "request2")
    with pytest.raises(ValueError, match="source scan"):
        series.accept_pair(observation("t2", 155, scan_ids=["t1-001", "t1-101"]))


def test_diagnostics_reconcile_asynchronous_completion(tmp_path):
    from types import SimpleNamespace
    from tas_copilot_core.workflows.andie_temperature_series import TemperatureSeriesCoordinator
    series = TemperatureSeries(tmp_path / "s.json", "test")
    for i in range(1, 6):
        series.select(f"t{i}", 100+i, f"request{i}")
        series.accept_pair(observation(f"t{i}", 100+i))
    specs = [SimpleNamespace(scan_id=f"d{r}", reflection=r, role="diagnostic") for r in ("100", "102")]
    request = SimpleNamespace(run_id="test", action_id="diag5", temperature_k=105,
                              selected_temperature_index=5, scans=specs)
    class Executor:
        calls = 0
        def submit(self, request):
            return SimpleNamespace(status="awaiting_owner")
        def reconcile(self, request):
            self.calls += 1
            return SimpleNamespace(status="complete", scans=[
                SimpleNamespace(scan_id=f"d{r}", model_dump=lambda mode, r=r: {"reflection": r}) for r in ("100", "102")])
    executor = Executor()
    before = series.science_hash()
    TemperatureSeriesCoordinator(series, executor, None).acquire_diagnostics("t5", request)
    assert executor.calls == 1 and series.diagnostics_due() == []
    assert series.science_hash() == before


def test_planning_uses_registered_posterior_and_validates_returned_action(tmp_path):
    from tas_copilot_core.workflows.andie_temperature_series import TemperatureSeriesCoordinator
    series = TemperatureSeries(tmp_path / "s.json", "test")
    series.select("t1", 100, "request")
    series.accept_pair(observation())
    series.register_inference("job", "hash")
    series.accept_inference("job", series.science_hash(), {"input_hash": "hash", "parameter_names": ["Tc"]})
    coordinator = TemperatureSeriesCoordinator(series, None, None)
    arguments = dict(posterior_samples=[[150]], parameter_names=["Tc"],
                     candidates=[{"point_id": "t2", "temperature_k": 120}], current_temperature_k=100,
                     posterior_quality={"accepted": True, "evidence": "mock"})
    def policy(payload):
        assert "diagnostics" not in payload
        return {"status": "planned", "science_hash": payload["science_hash"], "point_id": "t2", "temperature_k": 120}
    assert coordinator.plan_next(policy=policy, **arguments)["point_id"] == "t2"
    with pytest.raises(ValueError, match="unregistered"):
        coordinator.plan_next(policy=lambda p: {**policy(p), "temperature_k": 121}, **arguments)
    series.register_inference("new-job", "new-hash")
    with pytest.raises(ValueError, match="latest inference"):
        coordinator.plan_next(policy=policy, **arguments)


def test_recorded_quality_flags_prevent_science_admission(tmp_path):
    series = TemperatureSeries(tmp_path / "s.json", "test")
    series.select("t1", 100, "request")
    with pytest.raises(ValueError, match="quality flags"):
        series.accept_pair(observation(quality_flags=["thermal_settling", "structured_residuals"]))
    assert series.science_payload()["observations"] == []
    assert series.state["revision"] == 0


def test_cost_terms_are_disjoint_and_finite():
    cost = PairedScanCost(scan_001_seconds=930, scan_101_seconds=930,
                          within_scan_motion_seconds=434, reflection_switch_seconds=10,
                          temperature_ramp_seconds=120, stabilization_seconds=180)
    assert cost.measurement_seconds == 2294
    assert cost.transition_seconds == 310
    assert cost.total_seconds == 2604
    with pytest.raises(ValueError):
        PairedScanCost(scan_001_seconds=float("inf"), scan_101_seconds=10)
