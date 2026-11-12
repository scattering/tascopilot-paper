from __future__ import annotations

import json
import os
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

for dependency in ("scipy", "pandas", "lattice_calculator", "rescalculator"):
    pytest.importorskip(dependency, reason="Resolution-fit integration requires external scientific dependencies")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import srmn2sb2_official_python_intensity_core as core

RUN_DIR = ROOT / "local_runs" / "srmn2sb2_pyspinw_fit_20260622"
DATA = RUN_DIR / "dispersion_points.csv"
BACKGROUNDS = RUN_DIR / "scan_peak_fits.json"


@pytest.fixture(scope="module")
def prepared():
    if not DATA.is_file() or not BACKGROUNDS.is_file():
        pytest.skip("Full refit requires separately supplied source analysis inputs")
    # Eight samples keep the fixture fast; the production/preflight default is 512.
    return core.prepare_data(DATA, BACKGROUNDS, 8, resolution_policy="legacy")


def test_selected_pilot_shape_and_resolution_provenance(prepared) -> None:
    assert len(prepared.groups) == 20
    assert prepared.npoints == 799
    assert len(prepared.all_sample_hkl) == 799 * 8
    assert prepared.resolution_policy == "legacy"
    assert tuple(group.filename for group in prepared.groups) == core.SELECTED_SCAN_FILENAMES
    assert all(
        group.metadata["horizontal_collimation_source"]
        == "legacy_local_analysis_assumption"
        for group in prepared.groups
    )
    assert all(group.metadata["resolution_configuration_complete"] is False for group in prepared.groups)
    assert all("sigma_floored_point_count" in group.metadata for group in prepared.groups)
    assert all(len(group.scan_coordinate) == len(group.y) for group in prepared.groups)
    assert all(len(group.energy_transfer_mev) == len(group.y) for group in prepared.groups)
    assert all(group.scan_coordinate_label == "H = K (r.l.u.)" for group in prepared.groups[:11])
    assert all(group.scan_coordinate_label == "L (r.l.u.)" for group in prepared.groups[11:])


def test_only_positive_nonzero_modes_can_contribute() -> None:
    energy = np.asarray([[-12.0, -8.0e-6, 8.0e-6, 2.0e-3, 8.0]])
    intensity = np.asarray([[1.0e100, 2.0e100, 3.0e100, 4.0, 5.0]])
    filtered_energy, filtered_intensity = core.physical_positive_modes(energy, intensity)
    np.testing.assert_array_equal(filtered_energy[0, :3], 0.0)
    np.testing.assert_array_equal(filtered_intensity[0, :3], 0.0)
    np.testing.assert_allclose(filtered_energy[0, 3:], [2.0e-3, 8.0])
    np.testing.assert_allclose(filtered_intensity[0, 3:], [4.0, 5.0])


def test_background_input_rejects_duplicates_and_nonfinite(tmp_path: Path) -> None:
    rows = json.loads(BACKGROUNDS.read_text())
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(rows + [rows[0]]))
    with pytest.raises(ValueError, match="duplicate fitted background"):
        core._load_backgrounds(duplicate)

    rows[0]["best"]["background"] = float("nan")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        core._load_backgrounds(nonfinite)


def test_scale_policies_are_ordinary_20_or_5_keys(prepared) -> None:
    per_scan = core.OfficialPythonIntensityObjective(prepared, "per_scan", "single_y")
    shared = core.OfficialPythonIntensityObjective(prepared, "shared_zone_config", "single_y")
    assert len(per_scan.scale_keys) == 20
    assert len(shared.scale_keys) == 5
    assert set(shared.scale_keys) == set(core.START_ZONE_SCALE)
    assert set(per_scan.scale_keys) == set(core.START_SCAN_SCALE)


def test_seed_values_are_physical_orientations() -> None:
    assert core.SEED_PARAMETERS["all102"] == pytest.approx(
        (10.1330288294171, 0.1519244980477527, 0.449970843036889)
    )
    assert core.SEED_PARAMETERS["center005"] == pytest.approx(
        (6.9876109744411945, 0.1299802470982578, 0.7371469051365529)
    )


def test_python_sperp_is_one_explicit_nonrust_array_call() -> None:
    calls = []

    class FakeHamiltonian:
        def _energies_and_intensities(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["use_rust"] is False
            assert kwargs["use_rotating"] is False
            assert kwargs["intensity_unit"] == "spin"
            assert kwargs["components"] == "Sperp"
            count = len(kwargs["q_vectors"])
            return np.tile([-1.0, 1.0], (count, 1)), np.tile([0.2, 0.8], (count, 1))

    q = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    energy, intensity = core.python_sperp(FakeHamiltonian(), q)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["q_vectors"], q)
    assert energy.shape == intensity.shape == (3, 2)


def test_outer_thread1_only_replaces_pyspinw_executor_selector(monkeypatch) -> None:
    def original_selector():
        return object()

    original_selector.__name__ = "get_Executor"
    original_selector.__module__ = "pyspinw.windows_parallelisation"
    spinwave_module = SimpleNamespace(get_Executor=original_selector)
    monkeypatch.setattr(core, "_PINNED_OFFICIAL_EXECUTOR_SELECTOR", original_selector)
    monkeypatch.setattr(core, "_PATCHED_SPINWAVE_MODULE", None)
    monkeypatch.setattr(
        core.importlib,
        "import_module",
        lambda name: spinwave_module
        if name == "pyspinw.calculations.spinwave"
        else pytest.fail(f"unexpected import: {name}"),
    )
    monkeypatch.setattr(
        core,
        "_validate_pinned_python_spinwave",
        lambda module: {"guard": module is spinwave_module},
    )

    class FakeHamiltonian:
        def _energies_and_intensities(self, **kwargs):
            constructor = spinwave_module.get_Executor()
            executor = constructor()
            try:
                assert isinstance(executor, ThreadPoolExecutor)
                assert executor._max_workers == 1
            finally:
                executor.shutdown()
            count = len(kwargs["q_vectors"])
            return np.ones((count, 2)), np.ones((count, 2))

    q = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    core.python_sperp(FakeHamiltonian(), q, scheduling_policy="outer_thread1")
    assert spinwave_module.get_Executor is core._get_thread1_executor
    outer_provenance = core.python_scheduling_provenance("outer_thread1")
    assert outer_provenance["executor"] == "ThreadPoolExecutor"
    assert outer_provenance["executor_max_workers"] == 1

    class OfficialHamiltonian:
        def _energies_and_intensities(self, **kwargs):
            assert spinwave_module.get_Executor is original_selector
            count = len(kwargs["q_vectors"])
            return np.ones((count, 2)), np.ones((count, 2))

    core.python_sperp(
        OfficialHamiltonian(), q, scheduling_policy="official_processpool"
    )
    assert spinwave_module.get_Executor is original_selector
    assert core._PATCHED_SPINWAVE_MODULE is None
    official_provenance = core.python_scheduling_provenance(
        "official_processpool"
    )
    assert official_provenance["executor"] == "ProcessPoolExecutor"
    # The scheduling shim must not alter the module shared with Bumps.
    assert core.multiprocessing.cpu_count() >= 1


def test_official_policy_does_not_import_or_patch_scheduler(monkeypatch) -> None:
    monkeypatch.setattr(
        core.importlib,
        "import_module",
        lambda name: pytest.fail(f"default policy imported scheduler: {name}"),
    )

    class FakeHamiltonian:
        def _energies_and_intensities(self, **kwargs):
            count = len(kwargs["q_vectors"])
            return np.ones((count, 2)), np.ones((count, 2))

    core.python_sperp(
        FakeHamiltonian(),
        np.asarray([[0.1, 0.2, 0.3]]),
        scheduling_policy="official_processpool",
    )


def _guard_fixture(monkeypatch, source: str):
    def calculation(
        rotations, magnitudes, q_vectors, couplings, positions,
        rlu_to_cart=None, field=None, rotating_frame=None, save_sab=False,
        save_wavefunctions=False,
    ):
        del (
            rotations, magnitudes, q_vectors, couplings, positions,
            rlu_to_cart, field, rotating_frame, save_sab, save_wavefunctions,
        )

    def official_selector():
        return None

    official_selector.__name__ = "get_Executor"
    official_selector.__module__ = "pyspinw.windows_parallelisation"
    module = SimpleNamespace(
        spinwave_calculation=calculation,
        get_Executor=official_selector,
    )
    monkeypatch.setattr(core, "_PINNED_OFFICIAL_EXECUTOR_SELECTOR", None)
    monkeypatch.setattr(core.inspect, "getsource", lambda function: source)
    monkeypatch.setattr(
        core,
        "PINNED_PYSPINW_SPINWAVE_SOURCE_SHA256",
        core.hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    return module


def test_pinned_scheduler_guard_accepts_only_reviewed_signature_and_markers(monkeypatch) -> None:
    source = "\n".join(core.PINNED_PYSPINW_SPINWAVE_SOURCE_MARKERS)
    module = _guard_fixture(monkeypatch, source)
    metadata = core._validate_pinned_python_spinwave(module)
    assert metadata["spinwave_signature_parameters"] == list(
        core.PINNED_PYSPINW_SPINWAVE_PARAMETERS
    )
    assert metadata["spinwave_source_sha256"] == core.hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    missing_source = "\n".join(core.PINNED_PYSPINW_SPINWAVE_SOURCE_MARKERS[:-1])
    missing_module = _guard_fixture(monkeypatch, missing_source)
    with pytest.raises(RuntimeError, match="scheduler markers changed"):
        core._validate_pinned_python_spinwave(missing_module)


def test_python_backend_exceptions_propagate() -> None:
    class BrokenHamiltonian:
        def _energies_and_intensities(self, **kwargs):
            raise RuntimeError("backend failure sentinel")

    with pytest.raises(RuntimeError, match="backend failure sentinel"):
        core.python_sperp(BrokenHamiltonian(), np.asarray([[0.1, 0.2, 0.3]]))


def test_resolution_policy_fails_closed_without_raw_header(tmp_path: Path) -> None:
    import pandas as pd

    rows = pd.DataFrame({"path": [str(tmp_path / "absent.bt7")]})
    assert core.horizontal_collimation(rows, "legacy") == (
        core.LEGACY_HCOL_ARCMIN,
        "legacy_local_analysis_assumption",
    )
    with pytest.raises(FileNotFoundError, match="header_horizontal"):
        core.horizontal_collimation(rows, "header_horizontal")
    with pytest.raises(ValueError, match="resolution policy"):
        core.horizontal_collimation(rows, "unknown")


def test_synthetic_objective_is_finite_and_batches_all_q_once(monkeypatch, prepared) -> None:
    objective = core.OfficialPythonIntensityObjective(prepared, "shared_zone_config", "single_y")
    calls = []

    class DummyModel:
        pass

    monkeypatch.setattr(core, "build_official_srmn2sb2_model", lambda *args: DummyModel())

    def fake_sperp(model, q, scheduling_policy):
        assert isinstance(model, DummyModel)
        assert scheduling_policy == "official_processpool"
        calls.append(np.asarray(q))
        # Symmetric positive/negative modes.  Smooth constant intensity is enough
        # to exercise QMC split, f2, energy kernel, scales and fixed backgrounds.
        energy = np.column_stack([-np.full(len(q), 8.0), np.full(len(q), 8.0)])
        intensity = np.column_stack([np.full(len(q), 0.1), np.full(len(q), 1.0)])
        return energy, intensity

    monkeypatch.setattr(core, "python_sperp", fake_sperp)
    scales = {key: objective.scale_start(key) for key in objective.scale_keys}
    predictions = objective.predict_groups(7.9, 0.165, 0.51, 0.08853, scales)
    assert len(predictions) == len(prepared.groups) == 20
    expected = np.concatenate([
        (result.prediction - group.y) / group.sigma
        for group, result in zip(prepared.groups, predictions)
    ])
    calls.clear()
    residual = objective.residuals(7.9, 0.165, 0.51, 0.08853, scales)
    assert len(calls) == 1
    assert len(calls[0]) == len(prepared.all_sample_hkl)
    assert residual.shape == (799,)
    assert np.all(np.isfinite(residual))
    np.testing.assert_array_equal(residual, expected)


def _load_bumps_model(
    seed: str,
    normalization: str,
    fit_mode: str = "full",
    scheduling: str | None = None,
    scheduling_environment: str | None = None,
):
    model = SCRIPT_DIR / "srmn2sb2_official_python_intensity_bumps_model.py"
    prior_argv = sys.argv
    prior_environment = os.environ.get(core.PYTHON_SCHEDULING_ENVIRONMENT)
    try:
        if scheduling_environment is None:
            os.environ.pop(core.PYTHON_SCHEDULING_ENVIRONMENT, None)
        else:
            os.environ[core.PYTHON_SCHEDULING_ENVIRONMENT] = scheduling_environment
        sys.argv = [
            str(model), str(DATA), str(BACKGROUNDS), seed, normalization,
            "single_y", "legacy", "8", fit_mode,
        ]
        if scheduling is not None:
            sys.argv.append(scheduling)
        namespace = runpy.run_path(str(model))
    finally:
        sys.argv = prior_argv
        if prior_environment is None:
            os.environ.pop(core.PYTHON_SCHEDULING_ENVIRONMENT, None)
        else:
            os.environ[core.PYTHON_SCHEDULING_ENVIRONMENT] = prior_environment
    return namespace["problem"]


@pytest.mark.parametrize("seed", ["all102", "center005"])
@pytest.mark.parametrize("normalization,scale_count", [("per_scan", 20), ("shared_zone_config", 5)])
def test_thin_model_uses_complete_stock_parameter_starts(seed, normalization, scale_count) -> None:
    problem = _load_bumps_model(seed, normalization)
    labels = list(problem.labels())
    values = dict(zip(labels, problem.getp()))
    bounds = np.asarray(problem.bounds())
    sj1, ratio, sjc = core.SEED_PARAMETERS[seed]
    assert values["SJ1_meV"] == pytest.approx(sj1)
    assert values["J2_over_J1"] == pytest.approx(ratio)
    assert values["SJc_meV"] == pytest.approx(sjc)
    assert values["D_easy_plane_meV"] == pytest.approx(0.08853)
    assert len(labels) == 4 + scale_count
    assert bounds.shape == (2, len(labels))
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[0] < bounds[1])
    assert np.all((problem.getp() >= bounds[0]) & (problem.getp() <= bounds[1]))


@pytest.mark.parametrize("normalization,scale_count", [("per_scan", 20), ("shared_zone_config", 5)])
def test_scale_only_mode_fixes_only_physics_parameters(normalization, scale_count) -> None:
    problem = _load_bumps_model("all102", normalization, "scale_only")
    labels = list(problem.labels())
    assert len(labels) == scale_count
    assert all(label.startswith("scale_") for label in labels)
    assert np.all(np.isfinite(problem.getp()))
    assert np.asarray(problem.bounds()).shape == (2, scale_count)

    fitness = next(problem.models)
    physics = (fitness.sj1, fitness.j2_ratio, fitness.sjc, fitness.d_easy)
    assert all(parameter.fixed for parameter in physics)
    assert all(not parameter.fixed for parameter in fitness.scale_parameters.values())
    assert [float(parameter.value) for parameter in physics] == pytest.approx(
        [*core.SEED_PARAMETERS["all102"], 0.08853]
    )


def test_scheduling_policy_changes_no_bumps_parameters() -> None:
    official = _load_bumps_model(
        "all102", "shared_zone_config", scheduling="official_processpool"
    )
    outer = _load_bumps_model(
        "all102", "shared_zone_config", scheduling="outer_thread1"
    )
    assert list(outer.labels()) == list(official.labels())
    np.testing.assert_array_equal(outer.getp(), official.getp())
    np.testing.assert_array_equal(outer.bounds(), official.bounds())
    outer_fitness = next(outer.models)
    assert outer_fitness.objective.python_scheduling_policy == "outer_thread1"
    assert outer_fitness.objective.python_scheduling_source == "positional"


def test_scheduling_environment_is_explicit_and_conflicts_fail_closed() -> None:
    environment_model = _load_bumps_model(
        "all102",
        "shared_zone_config",
        scheduling_environment="outer_thread1",
    )
    objective = next(environment_model.models).objective
    assert objective.python_scheduling_policy == "outer_thread1"
    assert objective.python_scheduling_source == (
        f"environment:{core.PYTHON_SCHEDULING_ENVIRONMENT}"
    )

    with pytest.raises(SystemExit, match="conflicting Python scheduling policies"):
        _load_bumps_model(
            "all102",
            "shared_zone_config",
            scheduling="official_processpool",
            scheduling_environment="outer_thread1",
        )


def test_direct_scale_only_par_is_not_a_complete_eps_handoff(tmp_path: Path) -> None:
    from bumps.cli import load_best

    scale_only = _load_bumps_model("all102", "shared_zone_config", "scale_only")
    par_path = tmp_path / "scale-only.par"
    par_path.write_text(
        "".join(
            f"{label} {value:.15g}\n"
            for label, value in zip(scale_only.labels(), scale_only.getp())
        ),
        encoding="utf-8",
    )

    full = _load_bumps_model("all102", "shared_zone_config", "full")
    load_best(full, par_path)
    assert list(full.labels())[:4] == [
        "SJ1_meV", "J2_over_J1", "SJc_meV", "D_easy_plane_meV"
    ]
    np.testing.assert_array_equal(full.undefined[:4], True)
    np.testing.assert_array_equal(full.undefined[4:], False)


def test_thin_model_source_has_no_custom_bumps_layer() -> None:
    source = (SCRIPT_DIR / "srmn2sb2_official_python_intensity_bumps_model.py").read_text()
    assert "Parameter.default" in source
    assert ".range(" in source
    for forbidden in ("initpop", "profile_", "Constraint", "soft_limit", "penalty_residual"):
        assert forbidden not in source
    assert "bounds.limits" not in source


def test_bumps_model_plot_shows_all_scans_from_exact_prediction_path(
    monkeypatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    problem = _load_bumps_model("center005", "per_scan")
    fitness = next(problem.models)
    calls = []

    def fake_predict(sj1, ratio, sjc, d_easy, scales):
        calls.append((sj1, ratio, sjc, d_easy, dict(scales)))
        return tuple(
            core.ScanPrediction(
                filename=group.filename,
                signal=np.full(len(group.y), 0.25),
                prediction=group.y + 0.5 * group.sigma,
                fixed_background=group.background,
            )
            for group in fitness.objective.prepared.groups
        )

    monkeypatch.setattr(fitness.objective, "predict_groups", fake_predict)
    figure = plt.figure()
    try:
        fitness.plot()
        assert len(calls) == 1
        assert len(figure.axes) == 20
        first_group = fitness.objective.prepared.groups[0]
        first_axis = figure.axes[0]
        assert first_group.filename in first_axis.get_title()
        assert "E=2 meV" in first_axis.get_title()
        assert first_axis.get_xlabel() == first_group.scan_coordinate_label
        assert [text.get_text() for text in first_axis.get_legend().get_texts()] == [
            "data", "model", "fixed background"
        ]
        background_line = next(
            line for line in first_axis.lines
            if line.get_label() == "fixed background"
        )
        np.testing.assert_array_equal(
            background_line.get_ydata(),
            [first_group.background, first_group.background],
        )

        # FitProblem.plot invokes fitness.plot twice; the second render must not
        # repeat the expensive official-Python/QMC objective at unchanged p.
        fitness.plot()
        assert len(calls) == 1
    finally:
        plt.close(figure)
