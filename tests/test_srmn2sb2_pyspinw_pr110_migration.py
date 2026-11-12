from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_srmn2sb2_pyspinw_pr110_migration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_srmn2sb2_pyspinw_pr110_migration", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


class _FakeModel:
    def __init__(self, rust_mode: str):
        self.rust_mode = rust_mode
        self.energy = np.array(
            [[-4.0, -1.0, 1.0, 4.0], [-5.0, -2.0, 2.0, 5.0]],
            dtype=float,
        )
        self.intensity = np.array(
            [[0.10, 0.25, 0.80, 1.30], [0.05, 0.40, 0.95, 1.70]],
            dtype=float,
        )

    def energies_and_intensities(
        self,
        q_vectors: np.ndarray,
        *,
        use_rust: bool,
        use_rotating: bool,
        intensity_unit: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert np.array_equal(q_vectors, validation.PAIRING_Q_RLU)
        assert use_rotating is False
        assert intensity_unit == "spin"
        if not use_rust:
            return self.energy.copy(), self.intensity.copy()
        if self.rust_mode == "reversed_intensity":
            return self.energy.copy(), self.intensity[:, ::-1].copy()
        if self.rust_mode == "paired_permutation":
            permutation = np.array([2, 0, 3, 1])
            return (
                self.energy[:, permutation].copy(),
                self.intensity[:, permutation].copy(),
            )
        raise AssertionError(self.rust_mode)


def test_canonicalization_applies_energy_order_to_intensity() -> None:
    energy, intensity = validation._canonicalize_paired_modes(
        np.array([[2.0, -2.0, 1.0, -1.0]]),
        np.array([[40.0, 10.0, 30.0, 20.0]]),
        backend="fixture",
    )
    np.testing.assert_array_equal(energy, [[-2.0, -1.0, 1.0, 2.0]])
    np.testing.assert_array_equal(intensity, [[10.0, 20.0, 30.0, 40.0]])


def test_paired_regression_rejects_same_intensity_multiset_on_wrong_modes() -> None:
    result = validation._validate_paired_rust_python_modes(
        _FakeModel("reversed_intensity")
    )
    assert result["passed"] is False
    assert result["rust_energy_only_valid_for_fixture"] is True
    assert result["rust_sab_sperp_valid_for_fixture"] is False
    assert result["full_mode_axis_reversal_signature_detected"] is True
    assert result["paired_sperp_max_abs"] > 0.0
    assert result["full_mode_axis_reversal_signature_max_abs"] == 0.0
    assert "use_rust=False" in result["blocker"]
    assert "post-hoc" in result["blocker"]


def test_paired_regression_accepts_a_shared_backend_mode_permutation() -> None:
    result = validation._validate_paired_rust_python_modes(
        _FakeModel("paired_permutation")
    )
    assert result["passed"] is True
    assert result["paired_sperp_max_abs"] == 0.0
    assert result["blocker"] is None

