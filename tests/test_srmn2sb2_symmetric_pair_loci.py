from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="Optional symmetric-locus fitting tests require SciPy")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_srmn2sb2_symmetric_pair_loci.py"
SPEC = importlib.util.spec_from_file_location("srmn2sb2_pair_loci", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reader_retains_same_energy_repeat_scan_files(tmp_path):
    path = tmp_path / "repeat_scans.csv"
    fields = [
        "filename", "scan_variable", "H", "K", "L", "E", "Detector",
        "Monitor", "Temp", "path", "fixed_e", "orient", "lattice",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for filename in ("repeat_a", "repeat_b"):
            writer.writerow({
                "filename": filename, "scan_variable": "QY", "H": 0.9,
                "K": 0.9, "L": 2.0, "E": 10.0, "Detector": 100,
                "Monitor": 1000, "Temp": 5, "path": f"/data/{filename}.bt7",
                "fixed_e": "Ef 14.7", "orient": "1 1 0 0 0 1",
                "lattice": "4.59 4.59 7.72 90 90 120",
            })
    groups, _ = MODULE.read_scan_groups(path)
    assert list(groups) == ["repeat_a", "repeat_b"]
    assert groups["repeat_a"][0]["path"] != groups["repeat_b"][0]["path"]


def _synthetic_profile(peaks: list[tuple[float, float, float]], seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.30, 0.30, 61)
    sigma = np.full_like(x, 0.015)
    y = 0.12 + 0.02 * (x - x.min()) / (x.max() - x.min())
    for height, center, width in peaks:
        y = y + height * np.exp(-0.5 * ((x - center) / width) ** 2)
    y = y + rng.normal(0.0, sigma)
    return x, y, sigma


def test_background_only_is_a_real_candidate():
    x, y, sigma = _synthetic_profile([])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    selected = min(
        (variant for variant in variants.values() if variant.get("status") == "ok"),
        key=lambda variant: variant["bic"],
    )
    assert selected["model"] == MODULE.MODEL_BACKGROUND
    assert variants[MODULE.MODEL_BACKGROUND]["n_parameters"] == 2


def test_one_peak_is_not_forced_into_a_pair():
    x, y, sigma = _synthetic_profile([(0.65, 0.065, 0.030)])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    selected = min(
        (variant for variant in variants.values() if variant.get("status") == "ok"),
        key=lambda variant: variant["bic"],
    )
    assert selected["model"] == MODULE.MODEL_ONE
    assert abs(float(selected["parameters"][3]) - 0.065) < 0.01


def test_pair_centers_and_covariance_are_recovered():
    x, y, sigma = _synthetic_profile([
        (0.55, -0.105, 0.028),
        (0.75, 0.125, 0.028),
    ])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    selected = min(
        (variant for variant in variants.values() if variant.get("status") == "ok"),
        key=lambda variant: variant["bic"],
    )
    assert selected["model"] == MODULE.MODEL_PAIR
    pair = MODULE.summarize_pair_fit(selected, 0.0)
    assert pair is not None
    assert abs(pair["left_center_rlu"] + 0.105) < 0.01
    assert abs(pair["right_center_rlu"] - 0.125) < 0.01
    assert np.asarray(pair["center_covariance_rlu2"]).shape == (2, 2)
    assert np.all(
        np.diag(pair["center_covariance_rlu2"])
        >= np.diag(pair["center_covariance_counting_rlu2"])
    )
    assert pair["covariance_scale_applied"] >= 1.0
    assert pair["half_separation_sigma_rlu"] > 0


def test_incomplete_resolution_scenario_does_not_censor_pair():
    x, y, sigma = _synthetic_profile([
        (0.55, -0.030, 0.020),
        (0.55, 0.030, 0.020),
    ])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    pair = MODULE.summarize_pair_fit(variants[MODULE.MODEL_PAIR], 0.0)
    assert pair is not None
    resolution = {
        "status": "ok",
        "coordinate_fwhm_marginal_median_rlu": 0.08,
        "assumptions_review_status": "incomplete_configuration_sensitivity_scenario",
    }
    profile = {"step": 0.01}
    flags, include, censored, _ = MODULE.build_quality(
        variants, MODULE.MODEL_PAIR, pair, resolution, profile
    )
    assert "resolution_classification_indeterminate_incomplete_configuration" in flags
    assert censored is False
    assert include is True


def test_incomplete_resolution_scenario_does_not_exclude_offcenter_pair():
    x, y, sigma = _synthetic_profile([
        (0.55, -0.030, 0.020),
        (0.55, 0.150, 0.020),
    ])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    pair = MODULE.summarize_pair_fit(variants[MODULE.MODEL_PAIR], 0.0)
    assert pair is not None
    pair["projected_coordinate_resolution_fwhm_rlu"] = 0.08
    resolution = {
        "status": "ok",
        "coordinate_fwhm_marginal_median_rlu": 0.08,
        "assumptions_review_status": "incomplete_configuration_sensitivity_scenario",
    }
    flags, include, _, _ = MODULE.build_quality(
        variants, MODULE.MODEL_PAIR, pair, resolution, {"step": 0.01}
    )
    assert "resolution_classification_indeterminate_incomplete_configuration" in flags
    assert include is True


def test_quality_excludes_bad_profile_even_with_resolved_pair():
    x, y, sigma = _synthetic_profile([
        (0.55, -0.105, 0.028),
        (0.75, 0.125, 0.028),
    ])
    variants = MODULE.fit_profile_models(x, y, sigma, 0.0)
    pair = MODULE.summarize_pair_fit(variants[MODULE.MODEL_PAIR], 0.0)
    assert pair is not None
    variants[MODULE.MODEL_PAIR]["reduced_chi2"] = 3.01
    pair["projected_coordinate_resolution_fwhm_rlu"] = 0.08
    resolution = {
        "status": "ok",
        "coordinate_fwhm_marginal_median_rlu": 0.08,
        "assumptions_review_status": "incomplete_configuration_sensitivity_scenario",
    }
    flags, include, _, _ = MODULE.build_quality(
        variants, MODULE.MODEL_PAIR, pair, resolution, {"step": 0.01}
    )
    assert "profile_reduced_chi2_gt_3" in flags
    assert include is False


def test_finite_nonconverged_optimizer_result_is_rejected(monkeypatch):
    class Result:
        x = np.asarray([0.1, 0.1])
        fun = np.asarray([0.0, 0.0, 0.0])
        success = False

    monkeypatch.setattr(MODULE, "least_squares", lambda *args, **kwargs: Result())
    result = MODULE._fit_from_starts(
        MODULE.MODEL_BACKGROUND,
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.1, 0.1, 0.1]),
        np.asarray([0.01, 0.01, 0.01]),
        [np.asarray([0.1, 0.1])],
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
    )
    assert result["status"] == "no_successful_fit"
    assert result["starts_finite_nonconverged"] == 1


def test_parse_scan_configuration_preserves_conflicting_header_evidence(tmp_path):
    path = tmp_path / "scan.bt7"
    path.write_text(
        "#FixedE Ef 28\n"
        "#ExptComment Ef 14.7  25-25-80-120\n"
        "#Columns PreMonoColl PostMonoColl PreAnaColl PostAnaColl\n"
        "25MIN N/A N/A N/A\n",
        encoding="utf-8",
    )
    config = MODULE.parse_scan_configuration([{"path": str(path), "fixed_e": "Ef 28"}])
    assert config["fixed_ef_mev"] == 28.0
    assert config["horizontal_collimation_arcmin"] == (25.0, 25.0, 80.0, 120.0)
    assert config["horizontal_collimation_source"] == "raw_header_ExptComment"
    assert "expt_comment_fixed_energy_conflicts_with_FixedE" in config["quality_flags"]
    assert config["resolution_classification_authority"] is False


def test_filename_temperature_conflict_is_explicit():
    conflict = MODULE.filename_temperature_diagnostic("L_Q002_120K_e10", 4.2)
    consistent = MODULE.filename_temperature_diagnostic("HH2_4K_e4", 4.3)
    absent = MODULE.filename_temperature_diagnostic("SMS_Q002_e4", 4.3)
    assert conflict["conflict"] is True
    assert conflict["filename_temperature_label_k"] == 120.0
    assert consistent["conflict"] is False
    assert absent["filename_temperature_label_k"] is None
