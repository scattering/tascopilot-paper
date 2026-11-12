"""Reviewed component displays use synthetic fits, never private experiment data."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tas_copilot_core.workflows import live_experiment as live
from tas_copilot_core.workflows.component_series_view import (
    component_series_figures,
    component_series_section,
    validate_component_series,
)


def synthetic_components():
    """Six correlated series: 64 preferred points plus three model alternatives."""
    lengths = [12, 12, 10, 10, 10, 10]
    series = [{"id": f"s{i}", "label": f"Component {i + 1}", "role": "inference" if i < 2 else "diagnostic"}
              for i in range(6)]
    points = []
    for i, count in enumerate(lengths):
        reference = f"s{i}-p0"
        for j in range(count):
            scan = 100 + 20 * i + j
            area = 100 + 10 * i - 4 * j
            # This weak feature must remain in the reviewed comparison.
            if i == 5 and j == count - 1:
                area = .125
            points.append({"id": f"s{i}-p{j}", "series_id": f"s{i}", "scan": scan,
                "raw_file": f"synthetic_{scan}.dat.bt7", "raw_sha256": hashlib.sha256(str(scan).encode()).hexdigest(),
                "temperature_k": 100. + 10 * j, "temperature_min_k": 99.9 + 10 * j,
                "temperature_max_k": 100.1 + 10 * j, "area": area,
                "reference_id": reference, "fit_choice": "preferred"})
    n = len(points)
    # A shared nuisance term produces real off-diagonal area covariance.
    covariance = np.diag(np.linspace(4., 9., n)) + np.full((n, n), .25)
    order = [point["id"] for point in points]
    index = {identity: i for i, identity in enumerate(order)}
    jacobian = np.zeros((n, n))
    for i, point in enumerate(points):
        r = index[point["reference_id"]]
        reference = points[r]["area"]
        point["ratio"] = point["area"] / reference
        jacobian[i, i] += 1 / reference
        jacobian[i, r] -= point["area"] / reference**2
        point["area_stderr"] = float(np.sqrt(covariance[i, i]))
    ratio_covariance = jacobian @ covariance @ jacobian.T
    for i, point in enumerate(points):
        point["ratio_stderr"] = float(np.sqrt(max(0., ratio_covariance[i, i])))
    alternatives = []
    for i in (3, 19, 40):
        point = deepcopy(points[i])
        point.update(id=point["id"] + "-alternative", fit_choice="alternative", note="Synthetic alternative background model")
        point["area"] *= .92
        point["ratio"] *= .92
        point["area_stderr"] *= 1.2
        point["ratio_stderr"] *= 1.2
        alternatives.append(point)
    return {"schema_version": 1, "experiment_id": "synthetic-16",
        "snapshot_utc": "2026-10-30T20:00:00+00:00", "title": "Reviewed synthetic components",
        "area_unit": "counts / 4800 monitor × degree", "normalization_description": "Each series divided by its first measured area; reference uncertainty retained.",
        "series": series, "points": points + alternatives,
        "covariance": {"point_order": order, "area": covariance.tolist(), "ratio": ratio_covariance.tolist()},
        "coverage": {"latest_scan": 209, "latest_complete_scan": 209},
        "excluded_scans": [{"scan": 99, "raw_file": "synthetic_99.dat.bt7", "reason": "Incomplete profile"}],
        "caveats": ["Synthetic model comparison; no transition fit or phase assignment."],
        "provenance": [{"kind": "synthetic", "fit_source": "deterministic test fixture", "sha256": "1" * 64}]}


def validated(payload=None, *, records=None, experiment_id="synthetic-16"):
    warnings = []
    result = validate_component_series(payload or synthetic_components(), experiment_id, records or [], warnings)
    return result, warnings


def test_full_review_preserves_all_components_weak_points_and_alternatives():
    payload = synthetic_components()
    original = deepcopy(payload)
    result, warnings = validated(payload)
    assert result is not None, warnings
    assert payload == original  # Validation must not rewrite reviewed evidence.
    assert len(result["series"]) == 6
    assert len(result["points"]) == 67
    assert sum(point["fit_choice"] == "preferred" for point in result["points"]) == 64
    assert sum(point["fit_choice"] == "alternative" for point in result["points"]) == 3
    weak = next(point for point in result["points"] if point["id"] == "s5-p9")
    assert weak["area"] < weak["area_stderr"]
    np.testing.assert_array_equal(result["covariance"]["area"], payload["covariance"]["area"])
    np.testing.assert_array_equal(result["covariance"]["ratio"], payload["covariance"]["ratio"])
    assert result["provenance"] == payload["provenance"]


@pytest.mark.parametrize("fault", ["wrong_experiment", "duplicate_id", "unknown_series", "missing_covariance_point", "asymmetric_covariance", "negative_variance", "missing_provenance", "missing_hash", "nonfinite_area", "bad_temperature_range", "missing_reference", "reference_from_other_series", "wrong_ratio", "wrong_reference_ratio"])
def test_invalid_review_is_rejected_without_silent_partial_display(fault):
    payload = synthetic_components()
    if fault == "wrong_experiment":
        payload["experiment_id"] = "different-experiment"
    elif fault == "duplicate_id":
        payload["points"][1]["id"] = payload["points"][0]["id"]
    elif fault == "unknown_series":
        payload["points"][0]["series_id"] = "absent"
    elif fault == "missing_covariance_point":
        payload["covariance"]["point_order"].pop()
    elif fault == "asymmetric_covariance":
        payload["covariance"]["area"][0][1] += 1
    elif fault == "negative_variance":
        payload["covariance"]["area"][0][0] = -1
    elif fault == "missing_provenance":
        payload["provenance"] = []
    elif fault == "missing_hash":
        del payload["points"][0]["raw_sha256"]
    elif fault == "nonfinite_area":
        payload["points"][0]["area"] = float("nan")
    elif fault == "bad_temperature_range":
        payload["points"][0]["temperature_max_k"] = 90
    elif fault == "missing_reference":
        payload["points"][1]["reference_id"] = "absent"
    elif fault == "reference_from_other_series":
        payload["points"][1]["reference_id"] = "s1-p0"
    elif fault == "wrong_ratio":
        payload["points"][1]["ratio"] += .2
    elif fault == "wrong_reference_ratio":
        payload["points"][0]["ratio"] = .9
    result, warnings = validated(payload)
    assert result is None
    assert warnings


def test_shared_reference_uncertainty_is_retained_and_self_ratio_is_exact():
    result, warnings = validated()
    assert result is not None, warnings
    points = {point["id"]: point for point in result["points"]}
    order = result["covariance"]["point_order"]
    area = np.asarray(result["covariance"]["area"])
    ratio = np.asarray(result["covariance"]["ratio"])
    i, j, r = map(order.index, ["s0-p1", "s0-p2", "s0-p0"])
    ai, aj, ar = [points[name]["area"] for name in ("s0-p1", "s0-p2", "s0-p0")]
    expected = area[i, j] / ar**2 - aj * area[i, r] / ar**3 - ai * area[r, j] / ar**3 + ai * aj * area[r, r] / ar**4
    assert ratio[i, j] == pytest.approx(expected)
    assert ratio[i, j] > 0
    assert points["s0-p0"]["ratio"] == 1.
    assert points["s0-p0"]["ratio_stderr"] == pytest.approx(0., abs=1e-14)
    assert np.allclose(ratio[r], 0., atol=1e-14)


def test_new_raw_beyond_snapshot_warns_without_extending_reviewed_series():
    payload = synthetic_components()
    records = [{"number": 210, "name": "new_210.dat.bt7", "start": datetime(2026, 10, 31, tzinfo=timezone.utc)}]
    result, warnings = validated(payload, records=records)
    assert result is not None
    assert len(result["points"]) == 67
    assert any("210" in warning or "new" in warning.lower() or "snapshot" in warning.lower() for warning in warnings)


def test_alternatives_stay_out_of_preferred_covariance():
    payload = synthetic_components()
    payload["covariance"]["point_order"][-1] = payload["points"][-1]["id"]
    result, warnings = validated(payload)
    assert result is None and warnings


def test_render_uses_two_shared_axes_for_all_six_series_and_three_alternatives(monkeypatch):
    import matplotlib.axes
    original = matplotlib.axes.Axes.errorbar
    calls = []

    def record(self, x, y, *args, **kwargs):
        calls.append((self, list(np.atleast_1d(x)), list(np.atleast_1d(y)), kwargs))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", record)
    result, warnings = validated()
    assert result is not None, warnings
    figures = component_series_figures(result)
    assert len(figures) == 2
    assert all(image.startswith("data:image/png;base64,") for image, caption in figures)
    axes = {call[0] for call in calls}
    assert len(axes) == 2
    for axis in axes:
        own = [call for call in calls if call[0] is axis]
        assert sum(len(call[1]) for call in own) == 67
        assert len({call[3].get("label") for call in own if call[3].get("label") and "alternative" not in call[3]["label"].lower()}) >= 6
    assert all("Tc" not in caption for image, caption in figures)


def test_section_escapes_notes_and_reports_asof_provenance_and_exclusions():
    payload = synthetic_components()
    payload["title"] = "<script>not executable</script>"
    payload["caveats"].append("<img src=x onerror=alert(1)>")
    warnings = []
    html, status = component_series_section(payload, "synthetic-16", [], warnings)
    assert "<script>not executable</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "\\u003cimg src=x onerror=alert(1)>" in html
    assert "2026-10-30" in html and "Incomplete profile" in html
    assert "synthetic_99" in html
    assert status


def test_external_annotation_payload_is_hash_verified_and_experiment_bound(tmp_path):
    payload = synthetic_components()
    raw = json.dumps(payload).encode()
    (tmp_path / "components.json").write_bytes(raw)
    path = tmp_path / "bt7.yaml"
    path.write_text("experiment_id: synthetic-16\ncomponent_series:\n  path: components.json\n  sha256: " + hashlib.sha256(raw).hexdigest() + "\n")
    warnings = []
    annotations = live.load_annotations(path, "synthetic-16", warnings)
    assert annotations["component_series"]["points"] == payload["points"]
    assert warnings == []
    (tmp_path / "components.json").write_text(json.dumps({**payload, "title": "changed"}))
    warnings = []
    bad = live.load_annotations(path, "synthetic-16", warnings)
    assert warnings
    assert not isinstance(bad.get("component_series"), dict) or "points" not in bad["component_series"]
    assert live.load_annotations(path, "another-experiment", []) == {}


@pytest.mark.parametrize("declared", ["A4", "sampleTwoTheta.softPosition"])
def test_declared_native_xaxis_precedes_coupled_control_order(tmp_path, declared):
    path = tmp_path / "coupled900.dat.bt7"
    path.write_text("\n".join([
        "# date 2026-10-30T18:00:00+00:00", "# trajectory.controls [A3, A4]",
        "# trajectory.name generic_scan", f"# trajectoryData.xAxis {declared}",
        "## pt A3(deg) A4(deg) monitor Detector time(s) temp(K)",
        *[f"{i+1} {80 + .1*i} {44 + .2*i} 4800 {20+i} 2 100" for i in range(5)],
    ]) + "\n")
    record = live.read_nice_scan(path)
    assert record["variable"] == "A4"
    np.testing.assert_allclose(live.scan_axis_values(record), [44, 44.2, 44.4, 44.6, 44.8])


def test_unannotated_experiment_retains_automatic_order_parameter_view(monkeypatch):
    synthetic = {"label": "Automatic peak", "variable": "A3", "other": "A4", "center": 1., "other_val": 20.}
    monkeypatch.setattr(live, "detect_meshes", lambda records: [])
    monkeypatch.setattr(live, "detect_order_parameter_series", lambda records, exclude: [synthetic])
    monkeypatch.setattr(live, "_order_parameter_figures", lambda series, fit: ([('data:image/png;base64,AA==', 'automatic figure')], {
        "n": 4, "T_min": 10., "T_max": 100., "area_first": 20., "area_last": 10.,
        "center_drift": 0., "fwhm_range": [1., 1.], "max_sd": .01, "max_offset": .02}))
    monkeypatch.setattr(live, "_activity_figure", lambda *args: "data:image/png;base64,AA==")
    monkeypatch.setattr(live, "_gallery_figure", lambda *args: "data:image/png;base64,AA==")
    monkeypatch.setattr(live, "_scan_browser_html", lambda *args, **kwargs: "")
    page, status = live.build_page(instrument_label="BT7", records=[], hk={}, meta={},
        discovered={"experiment_id": "unannotated"}, events=[], queue={}, annotations={},
        data_as_of="2026-10-30T20:00:00Z", warnings=[])
    assert "temperature series detected automatically" in page
    assert "Automatic peak" in page and "automatic figure" in page
    assert status["order_parameter_series"] == 1


def test_changed_preferred_raw_file_invalidates_reviewed_values(tmp_path):
    payload = synthetic_components()
    path = tmp_path / payload["points"][0]["raw_file"]
    path.write_text("changed after review")
    record = {"number": 100, "name": path.name, "path": str(path), "npts": 31,
              "start": datetime(2026, 10, 30, 18, tzinfo=timezone.utc)}
    result, warnings = validated(payload, records=[record])
    assert result is None
    assert any("hash" in warning.lower() or "changed" in warning.lower() for warning in warnings)


def test_changed_excluded_partial_warns_without_promoting_it_to_reviewed_data(tmp_path):
    payload = synthetic_components()
    excluded = payload["excluded_scans"][0]
    excluded["raw_sha256"] = hashlib.sha256(b"old partial").hexdigest()
    path = tmp_path / excluded["raw_file"]
    path.write_text("new completed data")
    record = {"number": excluded["scan"], "name": path.name, "path": str(path), "npts": 31,
              "start": datetime(2026, 10, 31, tzinfo=timezone.utc)}
    result, warnings = validated(payload, records=[record])
    assert result is not None
    assert len(result["points"]) == 67
    assert all(point["scan"] != excluded["scan"] for point in result["points"])
    assert any("partial" in warning.lower() or "changed" in warning.lower() or "snapshot" in warning.lower() for warning in warnings)


def test_cycle_binding_rejects_reused_experiment_identifier():
    payload = synthetic_components()
    payload["cycle"] = "synthetic-cycle-a"
    warnings = []
    assert validate_component_series(payload, "synthetic-16", [], warnings, cycle="synthetic-cycle-b") is None
    assert warnings


@pytest.mark.parametrize("matrix_name", ["area", "ratio"])
def test_reported_standard_errors_must_match_covariance(matrix_name):
    payload = synthetic_components()
    key = "area_stderr" if matrix_name == "area" else "ratio_stderr"
    payload["points"][2][key] *= 1.5
    result, warnings = validated(payload)
    assert result is None and warnings


def test_ratio_covariance_cannot_drop_shared_reference_uncertainty():
    payload = synthetic_components()
    matrix = np.asarray(payload["covariance"]["ratio"])
    # Diagonal-only uncertainties look plausible but erase the shared denominator.
    payload["covariance"]["ratio"] = np.diag(np.diag(matrix)).tolist()
    result, warnings = validated(payload)
    assert result is None and warnings


def test_one_component_cannot_mix_normalization_references():
    payload = synthetic_components()
    alternative = payload["points"][-3]
    alternative["reference_id"] = "s0-p1"
    reference = next(point for point in payload["points"] if point["id"] == "s0-p1")
    alternative["ratio"] = alternative["area"] / reference["area"]
    result, warnings = validated(payload)
    assert result is None and warnings


def test_same_component_and_raw_scan_cannot_be_preferred_twice():
    payload = synthetic_components()
    first, duplicate = payload["points"][:2]
    for key in ("scan", "raw_file", "raw_sha256"):
        duplicate[key] = first[key]
    result, warnings = validated(payload)
    assert result is None and warnings


def test_alternative_requires_a_preferred_fit_of_the_same_component_and_scan():
    payload = synthetic_components()
    alternative = payload["points"][-1]
    alternative.update(scan=300, raw_file="synthetic_300.dat.bt7", raw_sha256="2" * 64)
    payload["coverage"]["latest_scan"] = 300
    result, warnings = validated(payload)
    assert result is None and warnings


def test_preferred_scan_cannot_also_be_declared_excluded():
    payload = synthetic_components()
    point = payload["points"][0]
    payload["excluded_scans"].append({"scan": point["scan"], "raw_file": point["raw_file"],
        "raw_sha256": point["raw_sha256"], "reason": "Incomplete scan"})
    result, warnings = validated(payload)
    assert result is None and warnings


def test_preferred_fit_rejects_matching_but_incomplete_raw_record(tmp_path):
    payload = synthetic_components()
    point = payload["points"][0]
    path = tmp_path / point["raw_file"]
    path.write_text("100")  # Exactly matches this point's synthetic source hash.
    record = {"number": point["scan"], "name": path.name, "path": str(path), "npts": 10,
              "header": {"numPoints": "31", "trajectory.length": "31"},
              "start": datetime(2026, 10, 30, 18, tzinfo=timezone.utc)}
    result, warnings = validated(payload, records=[record])
    assert result is None
    assert any("incomplete" in warning.lower() or "partial" in warning.lower() or "point" in warning.lower() for warning in warnings)


def test_invalid_configured_components_do_not_fall_back_to_automatic_single_peak_fits(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configured reviewed data must not silently become automatic fits")
    monkeypatch.setattr(live, "detect_meshes", lambda records: [])
    monkeypatch.setattr(live, "detect_order_parameter_series", forbidden)
    monkeypatch.setattr(live, "_activity_figure", lambda *args: "data:image/png;base64,AA==")
    monkeypatch.setattr(live, "_gallery_figure", lambda *args: "data:image/png;base64,AA==")
    monkeypatch.setattr(live, "_scan_browser_html", lambda *args, **kwargs: "")
    page, status = live.build_page(instrument_label="BT7", records=[], hk={}, meta={},
        discovered={"experiment_id": "other-experiment"}, events=[], queue={},
        annotations={"component_series": synthetic_components()}, data_as_of="2026-10-30T20:00:00Z", warnings=[])
    assert status["component_series"]["valid"] is False
    assert "not substituted" in page
    assert "temperature series detected automatically" not in page
    assert not any("configured reviewed data must not" in warning for warning in status["warnings"])


def test_automatic_gaussian_area_error_keeps_amplitude_width_covariance(monkeypatch):
    from scipy import optimize
    parameters = np.array([20., 0., .5, 3.])
    covariance = np.diag([4., .01, .01, 1.])
    covariance[0, 2] = covariance[2, 0] = -.15
    monkeypatch.setattr(optimize, "curve_fit", lambda *args, **kwargs: (parameters, covariance))
    x = np.linspace(-2., 2., 21)
    result = live.fit_peak(x, live.gaussian(x, *parameters), np.ones_like(x))
    assert result is not None
    assert result["darea"] == pytest.approx(np.sqrt(4 * np.pi))
    assert result["darea"] < np.sqrt(10 * np.pi)  # Diagonal-only propagation was too large.


def test_annotation_sidecar_cannot_escape_its_config_directory(tmp_path):
    root = tmp_path / "annotations"
    root.mkdir()
    outside = tmp_path / "outside.json"
    raw = json.dumps(synthetic_components()).encode()
    outside.write_bytes(raw)
    config = root / "bt7.yaml"
    config.write_text("experiment_id: synthetic-16\ncomponent_series:\n  path: ../outside.json\n  sha256: " + hashlib.sha256(raw).hexdigest() + "\n")
    warnings = []
    annotations = live.load_annotations(config, "synthetic-16", warnings)
    assert "load_error" in annotations["component_series"]
    assert any("directory" in warning for warning in warnings)
