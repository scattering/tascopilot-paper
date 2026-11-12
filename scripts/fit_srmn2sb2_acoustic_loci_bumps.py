#!/usr/bin/env python3
"""Bumps fitness bridge for the SrMn2Sb2 position-only acoustic objective.

This importable module is deliberately not a fit driver.  The reviewed
Stampede workflow runs Bumps DREAM and Bumps LM as separate CLI processes so
DREAM state is checkpointable and the two endpoints can be audited
independently.  ``srmn2sb2_acoustic_loci_bumps_model.py`` is the tiny Bumps
entrypoint; ``validate_srmn2sb2_acoustic_bumps_result.py`` applies the hard
two-root gate and chooses the better feasible endpoint.

Bumps searches in symmetric coordinates ``(SJ_a, SJ2, SJ_b)``.  They map to
the legacy reporting coordinates as ``(SJ1, J2/J1, SJc) =
(SJ_a, SJ2/SJ_a, SJ_b)``.  Exchanging ``SJ_a`` and ``SJ_b`` at fixed ``SJ2``
is an exact acoustic-energy symmetry, so the default bounds are identical for
the two opposite-sublattice exchanges.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from fit_srmn2sb2_acoustic_loci import SEARCH_NO_ROOT_SCALE_MEV


BUMPS_PARAMETER_NAMES = ("SJ_a_meV", "SJ2_meV", "SJ_b_meV")
CORE_PARAMETER_NAMES = ("sj1_mev", "j2_over_j1", "sjc_mev")
DEFAULT_BUMPS_START = np.array([10.0, 1.0, 5.0], dtype=float)
DEFAULT_BUMPS_LOWER = np.array([0.25, -3.0, 0.25], dtype=float)
DEFAULT_BUMPS_UPPER = np.array([30.0, 18.0, 30.0], dtype=float)
DEFAULT_FEASIBILITY_WEIGHT = 1.0e6


def optimizer_to_core(parameters: Sequence[float]) -> np.ndarray:
    """Convert ``(SJ_a,SJ2,SJ_b)`` to ``(SJ1,J2/J1,SJc)``."""

    point = np.asarray(parameters, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[0] <= 0.0:
        return np.full(3, np.nan, dtype=float)
    return np.array([point[0], point[1] / point[0], point[2]], dtype=float)


def core_to_optimizer(parameters: Sequence[float]) -> np.ndarray:
    """Convert ``(SJ1,J2/J1,SJc)`` to ``(SJ_a,SJ2,SJ_b)``."""

    point = np.asarray(parameters, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("core parameters must be three finite values")
    return np.array([point[0], point[0] * point[1], point[2]], dtype=float)


def exact_optimizer_swap(parameters: Sequence[float]) -> np.ndarray:
    """Return the exact acoustic label-swap partner at fixed ``SJ2``."""

    point = np.asarray(parameters, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("optimizer parameters must be three finite values")
    return point[[2, 1, 0]]


def exact_dynamic_stability(parameters: Sequence[float], *, tolerance_mev: float = 1e-12) -> bool:
    """Exact stability gate for the symmetric acoustic Hamiltonian.

    With ``a=SJ_a>0``, ``c=SJ_b>0`` and ``j=SJ2``, the acoustic radicand is
    nonnegative throughout reciprocal space exactly when negative ``j`` is
    unrestricted or nonnegative ``j <= (a+c)/6``.  This analytic gate is used
    in addition to the core diagnostic reciprocal-space grid.
    """

    point = np.asarray(parameters, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False
    a, j, c = point
    if a <= 0.0 or c <= 0.0:
        return False
    return bool(j < 0.0 or j <= (a + c) / 6.0 + tolerance_mev)


def exact_dynamic_instability_residual(parameters: Sequence[float]) -> float:
    """Continuous numerical excess above the exact antiferromagnetic boundary."""

    point = np.asarray(parameters, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return 1.0e6
    a, j, c = point
    if a <= 0.0 or c <= 0.0:
        return 1.0e6
    return max(0.0, j - (a + c) / 6.0)


def continuation_residuals(
    objective: Any,
    optimizer_parameters: Sequence[float],
    *,
    feasibility_weight: float = DEFAULT_FEASIBILITY_WEIGHT,
) -> np.ndarray:
    """Return the fixed-length nonstatistical residual used for global search.

    The core vector contains a coordinate continuation residual and one shared
    missing-root energy mismatch per scan pair, followed by one stability
    residual.  Only the missing-root and stability components are amplified.
    They are zero at every endpoint eligible for scientific reporting.
    """

    if not math.isfinite(feasibility_weight) or feasibility_weight <= 0.0:
        raise ValueError("feasibility_weight must be finite and positive")
    residual = np.asarray(
        objective.search_residuals(optimizer_to_core(optimizer_parameters)),
        dtype=float,
    ).copy()
    stride = 3 if objective.profile_scan_energy else 2
    expected = stride * len(objective.pairs) + 1
    if residual.shape != (expected,):
        raise RuntimeError(
            f"core search residual shape changed: expected {(expected,)}, got {residual.shape}"
        )
    scale = math.sqrt(feasibility_weight)
    residual[stride - 1 : stride * len(objective.pairs) : stride] *= scale
    residual[-1] = max(
        residual[-1], exact_dynamic_instability_residual(optimizer_parameters)
    ) * scale
    return residual


class BumpsPositionFitness:
    """Minimal Bumps protocol around the audited coordinate objective."""

    name = "SrMn2Sb2 position-only acoustic loci search"

    def __init__(
        self,
        objective: Any,
        *,
        start: Sequence[float] = DEFAULT_BUMPS_START,
        lower: Sequence[float] = DEFAULT_BUMPS_LOWER,
        upper: Sequence[float] = DEFAULT_BUMPS_UPPER,
        feasibility_weight: float = DEFAULT_FEASIBILITY_WEIGHT,
        parameter_type: Any | None = None,
    ):
        if parameter_type is None:
            try:
                from bumps.parameter import Parameter as parameter_type
            except Exception as exc:  # pragma: no cover - external runtime only
                raise RuntimeError(
                    "Bumps is required; use the reviewed Stampede tasai environment"
                ) from exc
        start_array = np.asarray(start, dtype=float)
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        if any(array.shape != (3,) for array in (start_array, lower_array, upper_array)):
            raise ValueError("start/lower/upper must each contain three values")
        if not all(np.all(np.isfinite(array)) for array in (start_array, lower_array, upper_array)):
            raise ValueError("start/lower/upper must be finite")
        if np.any(lower_array >= upper_array):
            raise ValueError("each lower bound must be smaller than its upper bound")
        if np.any(start_array < lower_array) or np.any(start_array > upper_array):
            raise ValueError("start must lie inside the supplied bounds")
        if lower_array[0] <= 0.0 or lower_array[2] <= 0.0:
            raise ValueError("SJ_a and SJ_b lower bounds must be strictly positive")
        if lower_array[0] != lower_array[2] or upper_array[0] != upper_array[2]:
            raise ValueError(
                "SJ_a and SJ_b bounds must match so the exact label swap remains in-domain"
            )
        if not math.isfinite(feasibility_weight) or feasibility_weight <= 0.0:
            raise ValueError("feasibility_weight must be finite and positive")
        self.objective = objective
        self.feasibility_weight = float(feasibility_weight)
        self.sj_a = parameter_type.default(
            float(start_array[0]), name=BUMPS_PARAMETER_NAMES[0]
        ).range(float(lower_array[0]), float(upper_array[0]))
        self.sj2 = parameter_type.default(
            float(start_array[1]), name=BUMPS_PARAMETER_NAMES[1]
        ).range(float(lower_array[1]), float(upper_array[1]))
        self.sj_b = parameter_type.default(
            float(start_array[2]), name=BUMPS_PARAMETER_NAMES[2]
        ).range(float(lower_array[2]), float(upper_array[2]))

    def parameters(self) -> list[Any]:
        return [self.sj_a, self.sj2, self.sj_b]

    def current_point(self) -> np.ndarray:
        return np.array(
            [self.sj_a.value, self.sj2.value, self.sj_b.value], dtype=float
        )

    def residuals(self) -> np.ndarray:
        return continuation_residuals(
            self.objective,
            self.current_point(),
            feasibility_weight=self.feasibility_weight,
        )

    def nllf(self) -> float:
        residual = self.residuals()
        return 0.5 * float(np.dot(residual, residual))

    def numpoints(self) -> int:
        stride = 3 if self.objective.profile_scan_energy else 2
        return stride * len(self.objective.pairs) + 1

    def update(self) -> None:
        pass

    def plot(self, view: Any | None = None) -> None:
        del view


def search_objective_provenance(feasibility_weight: float) -> dict[str, Any]:
    return {
        "statistical_likelihood": False,
        "dream_draws_are_posterior_samples": False,
        "feasibility_weight": float(feasibility_weight),
        "missing_root_scale_mev": float(SEARCH_NO_ROOT_SCALE_MEV),
        "description": (
            "coordinate continuation plus one shared missing-root term per scan "
            "and one stability term; used only for optimizer navigation"
        ),
    }
