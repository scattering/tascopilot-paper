"""Offline integration: detector rows -> full multiplet fits -> science ledger."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from tas_copilot_core.workflows.andie_scan_execution import (
    FullScanRequest, ScanSpec, ScanExecutor, SimulatorBackend,
)
from tas_copilot_core.workflows.andie_temperature_series import (
    TemperatureSeries, TemperatureSeriesCoordinator,
)


def component(label, center, area):
    return {"label": label, "center": center, "center_bounds": [center-.3, center+.3],
            "fwhm": .7, "fwhm_bounds": [.3, 1.2], "area": area}


def scan_request():
    specs = []
    for reflection, limits, components in [
        ("001", [22, 27.5], [component("001", 23.9, 25), component("001_neighbor", 26.2, 80)]),
        ("101", [42.5, 50.2], [component("101_lower", 43.75, 100),
          component("101_central", 46.27, 16), component("101_upper", 49.0, 120)])]:
        label = "001" if reflection == "001" else "101_central"
        specs.append(ScanSpec(scan_id=reflection, reflection=reflection, role="inference",
            axis="A4", axis_unit="degree", frame_id="film", preset=30,
            points=[{"A4": float(x), "A3": float(x/2+90)} for x in np.linspace(*limits, 101)],
            component_roles={c["label"]: "inference" if c["label"] == label else "nuisance" for c in components},
            metadata={"components": components}))
    return FullScanRequest(run_id="synthetic", action_id="T100", temperature_k=100,
        selected_temperature_index=1, sample_state={"id": "synthetic-film"},
        lattice={"a": 4.113, "c": 5.598, "gamma": 120}, frames={"film": {"plane": "H0L"}},
        instrument_config={"source": "synthetic", "Ei": 14.7},
        settling={"temperature_tolerance_k": .1}, scans=specs)


def test_full_multiplet_pipeline_retains_only_relevant_areas(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    analysis = pytest.importorskip("andie_scan_analysis", reason="optional ANDiE analysis runtime")
    science = pytest.importorskip("andie_science", reason="optional ANDiE analysis runtime")
    request = scan_request()

    def detector(request, spec, index, nodes):
        components = spec.metadata["components"]
        counts = science.n_gaussian([nodes["A4"]], [c["center"] for c in components],
                    [c["fwhm"] for c in components], [c["area"] for c in components],
                    background=50, slope=.1, reference_x=46 if spec.reflection == "101" else 25)[0]
        return {"counts": float(counts*10), "monitor": 48000., "live_time": 30.}

    executor = ScanExecutor(tmp_path / "scans", SimulatorBackend(detector))
    ledger = TemperatureSeries(tmp_path / "series.json", "synthetic")
    fit_calls = []

    def fit_pair(request, result):
        fit_calls.append(1)
        return analysis.fit_science_pair(request, result, monitor_reference=4800)

    pipeline = TemperatureSeriesCoordinator(ledger, executor, fit_pair)
    result = pipeline.acquire_pair(request)
    assert result.status == "complete"
    observation = ledger.science_payload()["observations"][0]
    assert observation["areas"] == pytest.approx([25, 16], rel=1e-5)
    assert observation["component_ids"] == ["001", "101_central"]
    profiles = observation["fit_provenance"]["full_profile_fits"]
    assert len(profiles[0]["components"]) == 2
    assert len(profiles[1]["components"]) == 3
    assert np.asarray(profiles[1]["covariance"]).shape == (11, 11)
    assert np.linalg.eigvalsh(observation["covariance"]).min() > 0
    pipeline.acquire_pair(request)
    assert len(fit_calls) == 1
    assert ledger.state["revision"] == 1
    job = pipeline.inference_request(parameter_names=science.PAIR_PARAMETER_NAMES,
        model={"id": science.MODEL_ID}, prior={}, sampler={}, seed=42)
    assert job.payload["observations"][0]["areas"] == observation["areas"]
    assert job.payload["science_hash"] == ledger.science_hash()
    assert job.payload["tc_bounds"] == [100, 200]


def test_no_scoring_of_extra_inference_component(tmp_path):
    request = scan_request()
    request.scans[1].component_roles["101_upper"] = "inference"
    pipeline = TemperatureSeriesCoordinator(TemperatureSeries(tmp_path / "s.json", "synthetic"),
                                           None, None)
    with pytest.raises(ValueError, match="only intended"):
        pipeline.acquire_pair(request)


def test_diagnostic_multiplets_have_no_science_component(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    analysis = pytest.importorskip("andie_scan_analysis", reason="optional ANDiE analysis runtime")
    science = pytest.importorskip("andie_science", reason="optional ANDiE analysis runtime")
    request = scan_request()
    diagnostic_specs = []
    for spec, reflection in zip(request.scans, ("100", "102")):
        values = spec.model_dump(mode="json")
        values.update(role="diagnostic", reflection=reflection)
        for i, component_spec in enumerate(values["metadata"]["components"]):
            component_spec["label"] = f"{reflection}_peak_{i}"
        values["component_roles"] = {c["label"]: "diagnostic" for c in values["metadata"]["components"]}
        diagnostic_specs.append(ScanSpec.model_validate(values))
    request.scans = diagnostic_specs
    def generator(request, spec, index, nodes):
        components = spec.metadata["components"]
        y = science.n_gaussian([nodes["A4"]], [c["center"] for c in components],
            [c["fwhm"] for c in components], [c["area"] for c in components], background=50)[0]
        return {"counts": float(y*10), "monitor": 48000., "live_time": 30.}
    result = ScanExecutor(tmp_path / "scans", SimulatorBackend(generator)).submit(request)
    fits = analysis.fit_diagnostic_profiles(request, result, monitor_reference=4800)
    assert set(fits) == {"100", "102"}
    assert all(fit["science"] is None for fit in fits.values())
    assert len(fits["100"]["components"]) == 2 and len(fits["102"]["components"]) == 3
