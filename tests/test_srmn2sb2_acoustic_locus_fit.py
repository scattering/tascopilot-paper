from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="Optional position-only fitting tests require SciPy")


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_srmn2sb2_acoustic_loci.py"
SPEC = importlib.util.spec_from_file_location("fit_srmn2sb2_acoustic_loci", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit_module
SPEC.loader.exec_module(fit_module)


def _record_pair(
    *,
    pair_id: str,
    axis: str,
    energy_mev: float,
    left: float,
    right: float,
    center: tuple[float, float, float],
    path_start: tuple[float, float, float],
    path_end: tuple[float, float, float],
) -> list[dict]:
    sigma_q = 0.0015
    covariance = [
        [sigma_q**2, 0.15 * sigma_q**2],
        [0.15 * sigma_q**2, sigma_q**2],
    ]
    coordinate_index = 0 if axis == "H" else 2
    records = []
    for side, coordinate in (("left", left), ("right", right)):
        start = np.asarray(path_start, dtype=float)
        end = np.asarray(path_end, dtype=float)
        fraction = (coordinate - start[coordinate_index]) / (
            end[coordinate_index] - start[coordinate_index]
        )
        hkl = start + fraction * (end - start)
        records.append(
            {
                "point_id": f"{pair_id}-{side}",
                "pair_id": pair_id,
                "scan_id": f"scan-{pair_id}",
                "source_path": f"/synthetic/{pair_id}.bt7",
                "axis": axis,
                "side": side,
                "energy_mev": energy_mev,
                "energy_sigma_mev": 0.02,
                "h": float(hkl[0]),
                "k": float(hkl[1]),
                "l": float(hkl[2]),
                "coordinate_rlu": coordinate,
                "coordinate_offset_rlu": coordinate - center[coordinate_index],
                "coordinate_sigma_rlu": sigma_q,
                "zone_center_hkl": list(center),
                "scan_path_hkl_start": list(path_start),
                "scan_path_hkl_end": list(path_end),
                "resolved": True,
                "censored": False,
                "branch_assignment": "unassigned_symmetric_pair",
                "include": True,
                "quality_flags": [],
                "pair_coordinate_covariance_rlu2": covariance,
            }
        )
    return records


def _synthetic_records(parameters: np.ndarray, *, full: bool = True) -> list[dict]:
    backend = fit_module.McNallyAcousticBackend()
    records: list[dict] = []
    h_offsets = (0.05, 0.12, 0.22, 1.0 / 3.0) if full else (0.12, 1.0 / 3.0)
    for index, offset in enumerate(h_offsets):
        energy = float(backend.energy(np.array([[offset, offset, 2.0]]), parameters)[0])
        records.extend(
            _record_pair(
                pair_id=f"H{index}",
                axis="H",
                energy_mev=energy,
                left=-offset,
                right=offset,
                center=(0.0, 0.0, 2.0),
                path_start=(-0.4, -0.4, 2.0),
                path_end=(0.4, 0.4, 2.0),
            )
        )
    l_offsets = (0.08, 0.20, 0.36, 0.50) if full else (0.20, 0.50)
    for index, offset in enumerate(l_offsets):
        energy = float(backend.energy(np.array([[0.0, 0.0, 2.0 + offset]]), parameters)[0])
        records.extend(
            _record_pair(
                pair_id=f"L{index}",
                axis="L",
                energy_mev=energy,
                left=2.0 - offset,
                right=2.0 + offset,
                center=(0.0, 0.0, 2.0),
                path_start=(0.0, 0.0, 1.5),
                path_end=(0.0, 0.0, 2.5),
            )
        )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_mcnally_acoustic_known_answers() -> None:
    backend = fit_module.McNallyAcousticBackend()
    parameters = np.array([7.9, 0.165, 0.51])
    hkl = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.1, 0.1, 2.0],
            [1.0 / 3.0, 1.0 / 3.0, 2.0],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, 3.0],
        ]
    )
    energy = backend.energy(hkl, parameters)
    assert energy == pytest.approx(
        [0.0, 8.591598152045668, 26.997, 24.086842881540115, 0.0],
        abs=1e-10,
    )


def test_mcnally_acoustic_has_exact_j1_jc_label_swap_symmetry() -> None:
    backend = fit_module.McNallyAcousticBackend()
    parameters = np.array([4.93552, 0.0828042, 0.914322])
    partner = backend.exact_exchange_swap(parameters)
    rng = np.random.default_rng(41)
    hkl = rng.uniform(-2.0, 3.0, size=(200, 3))
    assert backend.energy(hkl, parameters) == pytest.approx(
        backend.energy(hkl, partner), rel=1e-12, abs=1e-11
    )
    assert backend.identifiable_exchange_invariants(parameters) == pytest.approx(
        backend.identifiable_exchange_invariants(partner), rel=1e-12, abs=1e-12
    )
    assert backend.exact_exchange_swap(partner) == pytest.approx(parameters)


def test_locus_loader_rejects_preassigned_branch(tmp_path: Path) -> None:
    records = _synthetic_records(np.array([7.9, 0.165, 0.51]))
    records[0]["branch_assignment"] = "optical"
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(fit_module.LocusSchemaError, match="unassigned symmetric pairs"):
        fit_module.load_locus_pairs(path)


def test_locus_loader_retains_indeterminate_resolution_but_excludes_false(
    tmp_path: Path,
) -> None:
    records = _synthetic_records(np.array([7.9, 0.165, 0.51]), full=False)
    for record in records:
        record["resolved"] = None
        record["resolution_classification"] = "indeterminate_missing_metadata"
    # Explicitly unresolved pairs remain excluded even if producer include was
    # accidentally left true.
    for record in records[:2]:
        record["resolved"] = False
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, records)
    loaded = fit_module.load_locus_pairs(path)
    assert len(loaded.pairs) == 3
    assert {
        pair.resolution_classification for pair in loaded.pairs
    } == {"indeterminate_missing_metadata"}


def test_resolution_scenario_filter_is_explicit_sensitivity_mask(tmp_path: Path) -> None:
    records = _synthetic_records(np.array([7.9, 0.165, 0.51]), full=False)
    for index, record in enumerate(records):
        record["resolved"] = None
        record["resolution_classification"] = "indeterminate_incomplete_configuration"
        record["resolution_scenario_summaries"] = {
            "legacy_assumption": {
                "status": "ok",
                "resolved_by_scenario": index >= 2,
                "energy_resolution_sigma_mev": 0.123,
            }
        }
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, records)
    default = fit_module.load_locus_pairs(path)
    sensitivity = fit_module.load_locus_pairs(
        path, resolution_scenario_filter="legacy_assumption"
    )
    assert len(default.pairs) == 4
    assert len(sensitivity.pairs) == 3
    assert sensitivity.exclusion_counts["not_resolved_in_requested_scenario"] == 2
    assert all(pair.energy_sigma_mev == pytest.approx(0.123) for pair in sensitivity.pairs)
    assert all(
        pair.energy_sigma_source == "resolution_scenario:legacy_assumption"
        for pair in sensitivity.pairs
    )


def test_common_center_offset_mask_is_explicit_pair_sensitivity(tmp_path: Path) -> None:
    records = _synthetic_records(np.array([7.9, 0.165, 0.51]), full=False)
    # Shift one complete pair together, preserving its separation but creating
    # a focusing-style common-center offset.
    for record in records[:2]:
        record["coordinate_rlu"] += 0.06
        record["coordinate_offset_rlu"] += 0.06
        record["h"] += 0.06
        record["k"] += 0.06
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, records)
    default = fit_module.load_locus_pairs(path)
    sensitivity = fit_module.load_locus_pairs(
        path, max_common_center_offset_rlu=0.05
    )
    assert len(default.pairs) == 4
    assert len(sensitivity.pairs) == 3
    assert sensitivity.exclusion_counts[
        "common_center_offset_sensitivity_mask"
    ] == 2
    assert sensitivity.total_rows == sensitivity.excluded_rows + 2 * len(
        sensitivity.pairs
    )


def test_nominal_position_objective_accepts_null_energy_sigma_and_hard_rejects_no_root(
    tmp_path: Path,
) -> None:
    truth = np.array([7.9, 0.165, 0.51])
    records = _synthetic_records(truth, full=False)
    for record in records:
        record["energy_sigma_mev"] = None
        record["resolved"] = None
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, records)
    loaded = fit_module.load_locus_pairs(path)
    assert all(pair.energy_sigma_mev is None for pair in loaded.pairs)

    objective = fit_module.PositionOnlyObjective(loaded.pairs, grid_size=129)
    assert objective.root_feasible(truth)
    assert np.all(np.isfinite(objective.residuals(truth)))
    no_root = np.array([1.0, 0.10, 0.10])
    assert not objective.root_feasible(no_root)
    assert np.all(
        objective.residuals(no_root) == fit_module.HARD_INFEASIBLE_RESIDUAL
    )
    with pytest.raises(fit_module.LocusSchemaError, match="scenario-specific"):
        fit_module.PositionOnlyObjective(
            loaded.pairs, grid_size=65, profile_scan_energy=True
        )


def test_position_only_fit_recovers_synthetic_exchange_and_reports_rank(
    tmp_path: Path,
) -> None:
    truth = np.array([7.9, 0.165, 0.51])
    path = tmp_path / "locus_points.jsonl"
    _write_jsonl(path, _synthetic_records(truth, full=False))
    loaded = fit_module.load_locus_pairs(path)
    assert len(loaded.pairs) == 4

    report = fit_module.run_fit(
        loaded,
        start=[8.5, 0.14, 0.7],
        lower=[4.0, 0.08, 0.1],
        upper=[11.0, 0.20, 1.5],
        multistart=1,
        max_nfev=250,
        profile_points=0,
        profile_max_nfev=40,
        grid_size=129,
        min_band_top_fraction=0.95,
    )
    fitted = np.array([report["fit"]["parameters"][name] for name in fit_module.PARAMETER_NAMES])
    # The coordinate-root objective is intentionally nonlinear at the two band
    # extrema; recovery within one percent is a sufficient end-to-end gate.
    assert fitted == pytest.approx(truth, rel=1e-2, abs=4e-3)
    assert report["model"]["intensities_used"] is False
    assert report["model"]["optical_assignment_used"] is False
    assert "mcnally_camn2sb2_published_sj" in report["reference_evaluations"]
    assert "acoustic_hypothesis_adequate" in report["fit"]
    assert "pairs_without_two_exact_roots" in report["fit"]
    assert report["fit"]["optimizer_success_only"] is True
    assert report["exact_nonidentifiability"][
        "swap_partner_objective_sum_squares"
    ] == pytest.approx(
        report["fit"]["objective_sum_squares"], rel=1e-8, abs=1e-8
    )
    assert report["fit"]["all_included_pairs_have_two_exact_roots"] is True
    assert report["fit"]["objective_contains_nonstatistical_feasibility_penalty"] is False
    assert report["fit"]["objective_sum_squares"] == pytest.approx(
        report["fit"]["statistical_position_chi2"], abs=1e-10
    )
    assert report["identifiability"]["numerical_rank"] == 3
    singular_values = report["identifiability"]["dimensionless_singular_values"]
    assert len(singular_values) == 3
    assert all(value > 0.0 for value in singular_values)
    assert report["profile_likelihood"] == {}
    assert report["coverage"]["H"]["observed_max_to_model_band_top_fraction"] >= 0.95
    assert report["coverage"]["L"]["observed_max_to_model_band_top_fraction"] >= 0.95


def test_multistart_rejects_unstable_and_nonconverged_solutions(monkeypatch) -> None:
    class Objective:
        @staticmethod
        def stability_residual(parameters) -> float:
            return 1.0 if float(parameters[0]) < 0.0 else 0.0

    seen_starts: list[float] = []

    def fake_fit_once(objective, start, lower, upper, *, max_nfev):
        del objective, lower, upper, max_nfev
        seen_starts.append(float(start[0]))
        if len(seen_starts) == 1:
            return SimpleNamespace(
                success=False,
                fun=np.array([0.0]),
                x=np.asarray(start),
                message="iteration limit",
            )
        return SimpleNamespace(
            success=True,
            fun=np.array([2.0]),
            x=np.asarray(start),
            message="converged",
        )

    monkeypatch.setattr(fit_module, "_fit_once", fake_fit_once)
    result = fit_module.fit_multistart(
        Objective(),
        start=[-1.0, 0.1, 0.1],  # rejected before optimization as unstable
        lower=[-2.0, 0.0, 0.0],
        upper=[2.0, 1.0, 1.0],
        multistart=2,
        seed=7,
        max_nfev=3,
    )
    assert result.success is True
    assert all(start >= 0.0 for start in seen_starts)
    assert len(seen_starts) == 2


def test_profile_keeps_signed_delta_when_secondary_start_finds_lower_basin(
    monkeypatch,
) -> None:
    class Objective:
        @staticmethod
        def residuals(parameters) -> np.ndarray:
            # Main point [1,1,1] has chi2=3. A secondary profile start returns
            # free parameters zero, which has a lower chi2 and must be visible.
            return np.asarray(parameters, dtype=float)

        @staticmethod
        def stability_residual(parameters) -> float:
            del parameters
            return 0.0

    def fake_minimize(function, start, **kwargs):
        del function, kwargs
        x = np.zeros_like(np.asarray(start, dtype=float))
        return SimpleNamespace(
            success=True,
            fun=0.0,
            x=x,
            message="converged",
        )

    monkeypatch.setattr(fit_module, "minimize", fake_minimize)
    profiles = fit_module.profile_likelihoods(
        Objective(),
        best=[1.0, 1.0, 1.0],
        lower=[0.0, 0.0, 0.0],
        upper=[2.0, 2.0, 2.0],
        points=3,
        multistart=2,
        max_nfev=2,
    )
    assert any(
        row["signed_delta_objective"] is not None
        and row["signed_delta_objective"] < 0.0
        for rows in profiles.values()
        for row in rows
    )


def test_cli_smoke_writes_report_without_stale_arguments(tmp_path: Path) -> None:
    truth = np.array([7.9, 0.165, 0.51])
    path = tmp_path / "locus_points.jsonl"
    output = tmp_path / "fit"
    _write_jsonl(path, _synthetic_records(truth, full=False))
    status = fit_module.main(
        [
            "--input",
            str(path),
            "--output-dir",
            str(output),
            "--start",
            "7.9",
            "0.165",
            "0.51",
            "--lower",
            "4.0",
            "0.08",
            "0.1",
            "--upper",
            "11.0",
            "0.20",
            "1.5",
            "--multistart",
            "1",
            "--profile-points",
            "0",
            "--max-nfev",
            "300",
            "--root-grid-size",
            "65",
            "--min-band-top-fraction",
            "0.95",
        ]
    )
    assert status == 0
    report = json.loads((output / "fit_report.json").read_text())
    assert report["fit"]["all_included_pairs_have_two_exact_roots"] is True
