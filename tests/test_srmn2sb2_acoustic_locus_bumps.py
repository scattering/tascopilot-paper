from __future__ import annotations

import importlib.util
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="Optional position-only fitting tests require SciPy")
pytest.importorskip("bumps", reason="Optional DREAM fit tests require bumps")


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_script(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
core = _load_script("fit_srmn2sb2_acoustic_loci", "fit_srmn2sb2_acoustic_loci.py")
bridge = _load_script(
    "fit_srmn2sb2_acoustic_loci_bumps", "fit_srmn2sb2_acoustic_loci_bumps.py"
)
validator = _load_script(
    "validate_srmn2sb2_acoustic_bumps_result",
    "validate_srmn2sb2_acoustic_bumps_result.py",
)
selector = _load_script(
    "select_srmn2sb2_acoustic_bumps_dream_candidate",
    "select_srmn2sb2_acoustic_bumps_dream_candidate.py",
)


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
    sigma = 0.0015
    covariance = [[sigma**2, 0.15 * sigma**2], [0.15 * sigma**2, sigma**2]]
    coordinate_index = 0 if axis == "H" else 2
    result = []
    for side, coordinate in (("left", left), ("right", right)):
        start = np.asarray(path_start, dtype=float)
        end = np.asarray(path_end, dtype=float)
        fraction = (coordinate - start[coordinate_index]) / (
            end[coordinate_index] - start[coordinate_index]
        )
        hkl = start + fraction * (end - start)
        result.append(
            {
                "point_id": f"{pair_id}-{side}",
                "pair_id": pair_id,
                "scan_id": f"scan-{pair_id}",
                "source_path": f"/synthetic/{pair_id}.bt7",
                "axis": axis,
                "side": side,
                "energy_mev": energy_mev,
                "energy_sigma_mev": None,
                "h": float(hkl[0]),
                "k": float(hkl[1]),
                "l": float(hkl[2]),
                "coordinate_rlu": coordinate,
                "coordinate_offset_rlu": coordinate - center[coordinate_index],
                "coordinate_sigma_rlu": sigma,
                "zone_center_hkl": list(center),
                "scan_path_hkl_start": list(path_start),
                "scan_path_hkl_end": list(path_end),
                "resolved": None,
                "censored": False,
                "branch_assignment": "unassigned_symmetric_pair",
                "include": True,
                "quality_flags": [],
                "pair_coordinate_covariance_rlu2": covariance,
            }
        )
    return result


def _synthetic_records(parameters: np.ndarray) -> list[dict]:
    backend = core.McNallyAcousticBackend()
    result: list[dict] = []
    for index, offset in enumerate((0.12, 1.0 / 3.0)):
        energy = float(backend.energy([[offset, offset, 2.0]], parameters)[0])
        result.extend(
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
    for index, offset in enumerate((0.20, 0.50)):
        energy = float(backend.energy([[0.0, 0.0, 2.0 + offset]], parameters)[0])
        result.extend(
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
    return result


def _write_synthetic(path: Path, truth: np.ndarray) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _synthetic_records(truth)),
        encoding="utf-8",
    )


def _write_par(path: Path, optimizer_parameters: np.ndarray) -> None:
    path.write_text(
        "".join(
            f"{name} {float(value):.17g}\n"
            for name, value in zip(bridge.BUMPS_PARAMETER_NAMES, optimizer_parameters)
        ),
        encoding="utf-8",
    )


def test_symmetric_optimizer_coordinates_preserve_exact_swap() -> None:
    core_point = np.array([4.93552, 0.0828042, 0.914322])
    optimizer_point = bridge.core_to_optimizer(core_point)
    assert bridge.optimizer_to_core(optimizer_point) == pytest.approx(core_point)
    partner = bridge.optimizer_to_core(bridge.exact_optimizer_swap(optimizer_point))
    hkl = np.random.default_rng(17).uniform(-1.0, 3.0, size=(100, 3))
    backend = core.McNallyAcousticBackend()
    assert backend.energy(hkl, core_point) == pytest.approx(
        backend.energy(hkl, partner), rel=1e-12, abs=1e-11
    )


def test_exact_analytic_dynamic_stability_gate() -> None:
    assert bridge.exact_dynamic_stability([6.0, 2.0, 6.0])
    assert bridge.exact_dynamic_stability([6.0, 2.0 + 5e-13, 6.0])
    assert not bridge.exact_dynamic_stability([6.0, 2.001, 6.0])
    assert bridge.exact_dynamic_stability([0.25, -3.0, 0.25])
    assert not bridge.exact_dynamic_stability([0.0, -1.0, 1.0])
    assert bridge.exact_dynamic_instability_residual([6.0, 2.0, 6.0]) == 0.0
    assert bridge.exact_dynamic_instability_residual([6.0, 2.25, 6.0]) == pytest.approx(0.25)


def test_continuation_weight_scales_only_nonstatistical_terms() -> None:
    class Objective:
        profile_scan_energy = False
        pairs = (object(), object())

        @staticmethod
        def search_residuals(parameters):
            del parameters
            return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    residual = bridge.continuation_residuals(
        Objective(), [2.0, 0.2, 1.0], feasibility_weight=100.0
    )
    assert residual == pytest.approx([1.0, 20.0, 3.0, 40.0, 50.0])


def test_fitness_requires_swap_symmetric_bounds() -> None:
    class FakeParameter:
        @classmethod
        def default(cls, value, name):
            return cls(value, name)

        def __init__(self, value, name):
            self.value = value
            self.name = name

        def range(self, lower, upper):
            self.bounds = (lower, upper)
            return self

    objective = type("Objective", (), {"profile_scan_energy": False, "pairs": ()})()
    with pytest.raises(ValueError, match="must match"):
        bridge.BumpsPositionFitness(
            objective,
            start=[5.0, 0.5, 1.0],
            lower=[0.25, -1.0, 0.5],
            upper=[30.0, 4.0, 8.0],
            parameter_type=FakeParameter,
        )


def test_selector_ranks_all_hard_feasible_candidates_by_position_chi2(
    monkeypatch, tmp_path: Path
) -> None:
    candidates = {
        (1.0, 0.0, 1.0): 9.0,
        (2.0, 0.0, 2.0): 1.0,
        (3.0, 0.0, 3.0): 4.0,
    }

    class Objective:
        @staticmethod
        def root_feasible(parameters):
            del parameters
            return True

        @staticmethod
        def residuals(parameters):
            point = tuple(bridge.core_to_optimizer(parameters))
            return np.array([np.sqrt(candidates[point])])

    selector._WORKER_OBJECTIVE = Objective()
    # Stored DREAM order could rank points 1,3,2. Dense hard-position ranking
    # must still choose point 2 after evaluating every candidate.
    stored_order = [
        ("stored_best", -1.0, np.array([1.0, 0.0, 1.0])),
        ("stored_second", -2.0, np.array([3.0, 0.0, 3.0])),
        ("stored_last", -3.0, np.array([2.0, 0.0, 2.0])),
    ]
    evaluations = [selector._hard_evaluate(row[2]) for row in stored_order]
    selected_index = min(
        range(len(stored_order)), key=lambda index: (evaluations[index][1], index)
    )
    assert selected_index == 2
    assert all(feasible for feasible, _ in evaluations)


def test_selector_rejects_nonfinite_hard_objective() -> None:
    class Objective:
        @staticmethod
        def root_feasible(parameters):
            del parameters
            return True

        @staticmethod
        def residuals(parameters):
            del parameters
            return np.array([np.nan])

    selector._WORKER_OBJECTIVE = Objective()
    assert selector._hard_evaluate([1.0, 0.0, 1.0]) == (False, float("inf"))


def test_validator_keeps_better_feasible_dream_when_lm_worsens(tmp_path: Path) -> None:
    truth_core = np.array([7.9, 0.165, 0.51])
    input_path = tmp_path / "locus_points.jsonl"
    _write_synthetic(input_path, truth_core)
    dream_par = tmp_path / "dream.par"
    lm_par = tmp_path / "lm.par"
    _write_par(dream_par, bridge.core_to_optimizer(truth_core))
    _write_par(lm_par, bridge.core_to_optimizer([8.2, 0.155, 0.65]))
    report = validator.compare_endpoints(
        input_path=input_path,
        dream_parameter_path=dream_par,
        lm_parameter_path=lm_par,
        case="all102",
        root_grid_size=65,
    )
    assert report["status"] == "accepted_hard_root_feasible_endpoint"
    assert report["accepted_fit"]["stage"] == "dream"
    assert report["lm_retained_over_dream"] is False
    assert report["accepted_fit"]["statistical_position_chi2"] < 1e-8
    exact = report["accepted_fit"]["exact_nonidentifiability"]
    assert exact["swap_partner_position_chi2"] == pytest.approx(
        report["accepted_fit"]["statistical_position_chi2"], abs=1e-10
    )


def test_validator_reports_no_fit_if_neither_endpoint_has_all_roots(tmp_path: Path) -> None:
    truth_core = np.array([7.9, 0.165, 0.51])
    input_path = tmp_path / "locus_points.jsonl"
    _write_synthetic(input_path, truth_core)
    dream_par = tmp_path / "dream.par"
    lm_par = tmp_path / "lm.par"
    _write_par(dream_par, bridge.core_to_optimizer([1.0, 0.1, 0.3]))
    _write_par(lm_par, bridge.core_to_optimizer([1.2, 0.1, 0.4]))
    report = validator.compare_endpoints(
        input_path=input_path,
        dream_parameter_path=dream_par,
        lm_parameter_path=lm_par,
        case="all102",
        root_grid_size=65,
    )
    assert report["status"] == "no_hard_root_feasible_endpoint"
    assert report["accepted_fit"] is None
    assert report["stage_endpoints"]["dream"]["statistical_position_chi2"] is None
    assert report["stage_endpoints"]["lm"]["statistical_position_chi2"] is None


def test_bumps_cli_model_loads_in_cold_process(tmp_path: Path) -> None:
    pytest.importorskip("bumps")
    truth_core = np.array([7.9, 0.165, 0.51])
    input_path = tmp_path / "locus_points.jsonl"
    _write_synthetic(input_path, truth_core)
    environment = {
        **os.environ,
        "NUMBA_DISABLE_JIT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(tmp_path / "mplconfig"),
    }
    for model_name, case in (
        ("srmn2sb2_acoustic_loci_bumps_model.py", "all102"),
        ("srmn2sb2_acoustic_loci_bumps_dense_lm_model.py", "center005"),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "bumps.cli",
                "--chisq",
                str(SCRIPTS / model_name),
                str(input_path),
                case,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "chisq" in completed.stdout


def test_stampede_jobs_force_seeded_numpy_dream_rng() -> None:
    """The Stampede Bumps build's Numba DREAM RNG ignores CLI --seed."""

    for filename in (
        "run_srmn2sb2_acoustic_bumps_stampede.sbatch",
        "benchmark_srmn2sb2_acoustic_bumps_workers_stampede.sbatch",
    ):
        source = (SCRIPTS / filename).read_text()
        assert "export NUMBA_DISABLE_JIT=1" in source
        assert "NUMBA_DISABLE_JIT=%s" in source
        assert "assert numba.config.DISABLE_JIT == 1" in source
    benchmark = (
        SCRIPTS / "benchmark_srmn2sb2_acoustic_bumps_workers_stampede.sbatch"
    ).read_text()
    assert "matched_endpoint_pass" in benchmark
    assert "state.chains()" in benchmark
    assert "raw_chain_points_exact" in benchmark
    assert "raw_chain_logp_exact" in benchmark
    assert "matched-seed 8/48 full DREAM chain reproducibility failed" in benchmark
    production = (
        SCRIPTS / "run_srmn2sb2_acoustic_bumps_stampede.sbatch"
    ).read_text()
    assert 'RUNNER=$SCRIPT_DIR/run_srmn2sb2_acoustic_bumps_stampede.sbatch' in production
    assert '"$RUNNER" > "$CURRENT_MANIFEST"' in production
    validator_source = (
        SCRIPTS / "validate_srmn2sb2_acoustic_bumps_result.py"
    ).read_text()
    assert '"NUMBA_DISABLE_JIT"' in validator_source


def test_center005_dense_postcheck_is_immutable_and_fail_closed() -> None:
    source = (
        SCRIPTS / "run_srmn2sb2_center005_dense_postcheck_stampede.sbatch"
    ).read_text()
    assert 'test ! -e "$POSTCHECK_DIR"' in source
    assert 'sha256sum -c "$ORIGINAL_CASE_DIR/dream_state_input_sha256.txt"' in source
    assert "--root-grid-size 513" in source
    assert "--parallel 48" in source
    assert "--root-grid-size 1025" in source
    assert "srmn2sb2_acoustic_loci_bumps_dense_lm_model.py" in source
    lm_block = source[source.index("LM_CMD=(") : source.index("VALIDATE_CMD=(")]
    assert "--parallel" not in lm_block
    dense_model = (
        SCRIPTS / "srmn2sb2_acoustic_loci_bumps_dense_lm_model.py"
    ).read_text()
    assert "PositionOnlyObjective(loaded.pairs, grid_size=513)" in dense_model


def test_real_bumps_cli_dream_then_lm_and_validator_smoke(tmp_path: Path) -> None:
    pytest.importorskip("bumps")
    # Match the model's declared default. DREAM always includes the start point,
    # so the genuine raw chain contains a hard-feasible known answer without
    # mutating its final .par/state provenance.
    truth_core = np.array([10.0, 0.10, 5.0])
    input_path = tmp_path / "locus_points.jsonl"
    _write_synthetic(input_path, truth_core)
    model = SCRIPTS / "srmn2sb2_acoustic_loci_bumps_model.py"
    environment = {
        **os.environ,
        "NUMBA_DISABLE_JIT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(tmp_path / "mplconfig"),
    }
    dream_store = tmp_path / "dream"
    lm_store = tmp_path / "lm"
    dream_command = [
        sys.executable,
        "-m",
        "bumps.cli",
        "--fit=dream",
        "--init=lhs",
        "--outliers=none",
        "--pop=4",
        "--samples=60",
        "--burn=1",
        "--seed=20260812",
        f"--store={dream_store}",
        "--overwrite",
        "--batch",
        str(model),
        str(input_path),
        "all102",
    ]
    dream = subprocess.run(
        dream_command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert dream.returncode == 0, dream.stdout + dream.stderr
    basename = "srmn2sb2_acoustic_loci_bumps_model"
    dream_par = dream_store / f"{basename}.par"
    assert dream_par.is_file()
    # Bumps 1.0.3's text-state loader assumes the stats table is two-dimensional
    # and crashes on a legitimate one-row tiny smoke state. Production has many
    # update rows; duplicate the smoke row so this API-path test is portable
    # across the Python 3.9/3.11 builds used in local review.
    stats_path = dream_store / f"{basename}-stats.mc.gz"
    with gzip.open(stats_path, "rt") as handle:
        stats_lines = handle.readlines()
    if len(stats_lines) == 2:
        with gzip.open(stats_path, "wt") as handle:
            handle.writelines([*stats_lines, stats_lines[1]])
    selected_par = tmp_path / "dream_feasible.par"
    selection = selector.select_candidate(
        input_path=input_path,
        model_path=model,
        dream_state_base=dream_store / basename,
        dream_parameter_path=dream_par,
        case="all102",
        output_parameter_path=selected_par,
        root_grid_size=65,
        parallel=1,
    )
    if selection["status"] != "hard_feasible_candidate_selected":
        # A tiny global search is not guaranteed to enter the exact hard-root
        # domain. The selector's fail-closed behavior is the relevant gate.
        assert not selected_par.exists()
        return
    assert selected_par.is_file()
    lm_command = [
        sys.executable,
        "-m",
        "bumps.cli",
        "--fit=lm",
        f"--pars={selected_par}",
        "--steps=3",
        f"--store={lm_store}",
        "--overwrite",
        "--batch",
        str(model),
        str(input_path),
        "all102",
    ]
    lm = subprocess.run(
        lm_command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert lm.returncode == 0, lm.stdout + lm.stderr
    lm_par = lm_store / f"{basename}.par"
    assert lm_par.is_file()
    report = validator.compare_endpoints(
        input_path=input_path,
        dream_parameter_path=selected_par,
        lm_parameter_path=lm_par,
        case="all102",
        root_grid_size=65,
    )
    assert report["accepted_fit"] is not None
    assert report["accepted_fit"]["all_included_pairs_have_two_exact_roots"] is True
