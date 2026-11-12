#!/usr/bin/env python3
"""Thin stock-Bumps model for the official-Python SrMn2Sb2 EPS pilot.

Usage (arguments follow the Bumps model filename)::

    ... srmn2sb2_official_python_intensity_bumps_model.py \
        DATA_CSV BACKGROUNDS_JSON SEED NORMALIZATION \
        [DOMAIN] [RESOLUTION] [QMC] [FIT_MODE] [PYTHON_SCHEDULING]

``SEED`` is ``all102`` or ``center005``.  ``NORMALIZATION`` is ``per_scan``
or ``shared_zone_config``.  Every variable is an ordinary Bumps ``Parameter``
with a stock ``.range``.  There is no custom initializer, transform, tie,
profiled nuisance parameter, or constraint layer; CLI ``--init=eps`` acts on
these complete current values directly.  ``FIT_MODE`` is ``full`` (default)
or ``scale_only``.  Scale-only mode uses the same ordinary Parameters but
marks the four physics Parameters fixed before constructing ``FitProblem``;
it is intended only for a stock-LM normalization prefit.

``PYTHON_SCHEDULING`` is ``official_processpool`` (default, appropriate for
serial LM) or the explicitly opted-in ``outer_thread1`` mode used with stock
Bumps outer DREAM multiprocessing.  It changes no Bumps object or parameter.
The same opt-in may be supplied with ``SRMN_PYSPINW_SCHEDULING``; conflicting
positional/environment values fail closed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from bumps.fitproblem import FitProblem
from bumps.parameter import Parameter

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from srmn2sb2_official_python_intensity_core import (
    DEFAULT_PYTHON_SCHEDULING_POLICY,
    OfficialPythonIntensityObjective,
    PYTHON_SCHEDULING_ENVIRONMENT,
    SEED_PARAMETERS,
    prepare_data,
    validate_python_scheduling_policy,
)


if not 5 <= len(sys.argv) <= 10:
    raise SystemExit(
        "usage: srmn2sb2_official_python_intensity_bumps_model.py "
        "DATA_CSV BACKGROUNDS_JSON {all102,center005} "
        "{per_scan,shared_zone_config} [single_y|equal3] "
        "[legacy|header_horizontal] [QMC_POWER_OF_TWO] [full|scale_only] "
        "[official_processpool|outer_thread1]"
    )

INPUT_PATH = Path(sys.argv[1]).resolve()
BACKGROUNDS_PATH = Path(sys.argv[2]).resolve()
SEED = sys.argv[3]
NORMALIZATION_POLICY = sys.argv[4]
DOMAIN_POLICY = sys.argv[5] if len(sys.argv) >= 6 else "single_y"
RESOLUTION_POLICY = sys.argv[6] if len(sys.argv) >= 7 else "legacy"
QMC_SAMPLE_COUNT = int(sys.argv[7]) if len(sys.argv) >= 8 else 512
FIT_MODE = sys.argv[8] if len(sys.argv) >= 9 else "full"
POSITIONAL_PYTHON_SCHEDULING = sys.argv[9] if len(sys.argv) >= 10 else None
ENVIRONMENT_PYTHON_SCHEDULING = os.environ.get(PYTHON_SCHEDULING_ENVIRONMENT)

if (
    POSITIONAL_PYTHON_SCHEDULING is not None
    and ENVIRONMENT_PYTHON_SCHEDULING is not None
    and POSITIONAL_PYTHON_SCHEDULING != ENVIRONMENT_PYTHON_SCHEDULING
):
    raise SystemExit(
        f"conflicting Python scheduling policies: positional="
        f"{POSITIONAL_PYTHON_SCHEDULING!r}, "
        f"{PYTHON_SCHEDULING_ENVIRONMENT}={ENVIRONMENT_PYTHON_SCHEDULING!r}"
    )
if POSITIONAL_PYTHON_SCHEDULING is not None:
    PYTHON_SCHEDULING_POLICY = POSITIONAL_PYTHON_SCHEDULING
    PYTHON_SCHEDULING_SOURCE = (
        "matching_positional_and_environment"
        if ENVIRONMENT_PYTHON_SCHEDULING is not None
        else "positional"
    )
elif ENVIRONMENT_PYTHON_SCHEDULING is not None:
    PYTHON_SCHEDULING_POLICY = ENVIRONMENT_PYTHON_SCHEDULING
    PYTHON_SCHEDULING_SOURCE = f"environment:{PYTHON_SCHEDULING_ENVIRONMENT}"
else:
    PYTHON_SCHEDULING_POLICY = DEFAULT_PYTHON_SCHEDULING_POLICY
    PYTHON_SCHEDULING_SOURCE = "default"

if SEED not in SEED_PARAMETERS:
    raise SystemExit(f"unsupported seed {SEED!r}; use all102 or center005")
if FIT_MODE not in {"full", "scale_only"}:
    raise SystemExit(f"unsupported fit mode {FIT_MODE!r}; use full or scale_only")
try:
    validate_python_scheduling_policy(PYTHON_SCHEDULING_POLICY)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc


class OfficialPythonIntensityFitness:
    """Minimal residual protocol containing only ordinary Bumps Parameters."""

    name = "SrMn2Sb2 official-Python resolution-convolved intensity EPS pilot"

    def __init__(self, objective: OfficialPythonIntensityObjective):
        self.objective = objective
        self._plot_cache_key: tuple[float, ...] | None = None
        self._plot_cache: tuple[Any, ...] | None = None
        sj1, ratio, sjc = SEED_PARAMETERS[SEED]
        self.sj1 = Parameter.default(sj1, name="SJ1_meV").range(0.1, 20.0)
        self.j2_ratio = Parameter.default(ratio, name="J2_over_J1").range(-0.05, 0.6)
        self.sjc = Parameter.default(sjc, name="SJc_meV").range(-1.0, 20.0)
        self.d_easy = Parameter.default(0.08853, name="D_easy_plane_meV").range(0.0, 1.0)
        if FIT_MODE == "scale_only":
            self.sj1.fixed = True
            self.j2_ratio.fixed = True
            self.sjc.fixed = True
            self.d_easy.fixed = True
        self.scale_parameters = {
            key: Parameter.default(objective.scale_start(key), name=f"scale_{key}").range(
                0.0, 40.0*objective.scale_start(key)
            )
            for key in objective.scale_keys
        }

    def parameters(self) -> list[Any]:
        return [self.sj1, self.j2_ratio, self.sjc, self.d_easy] + [
            self.scale_parameters[key] for key in self.objective.scale_keys
        ]

    def residuals(self) -> np.ndarray:
        return self.objective.residuals(
            float(self.sj1.value), float(self.j2_ratio.value), float(self.sjc.value),
            float(self.d_easy.value),
            {key: float(value.value) for key, value in self.scale_parameters.items()},
        )

    def nllf(self) -> float:
        residual = self.residuals()
        return 0.5*float(np.dot(residual, residual))

    def numpoints(self) -> int:
        return self.objective.prepared.npoints

    def update(self) -> None:
        pass

    def plot(self, view: Any | None = None) -> None:
        """Render the fitted data and exact objective prediction for all scans."""
        import matplotlib.pyplot as plt

        del view  # The raw profile likelihood is displayed on its linear scale.
        scale_values = {
            key: float(value.value) for key, value in self.scale_parameters.items()
        }
        physics = (
            float(self.sj1.value), float(self.j2_ratio.value),
            float(self.sjc.value), float(self.d_easy.value),
        )
        cache_key = physics + tuple(scale_values[key] for key in self.objective.scale_keys)
        if cache_key != self._plot_cache_key:
            self._plot_cache = self.objective.predict_groups(*physics, scale_values)
            self._plot_cache_key = cache_key
        if self._plot_cache is None:  # pragma: no cover - guarded by the branch above
            raise RuntimeError("plot prediction cache was not populated")

        figure = plt.gcf()
        figure.clear()
        figure.set_size_inches(17.0, 18.0, forward=True)
        axes = figure.subplots(5, 4, squeeze=False)
        axes_flat = axes.ravel()
        for index, (axis, group, result) in enumerate(zip(
            axes_flat, self.objective.prepared.groups, self._plot_cache
        )):
            order = np.argsort(group.scan_coordinate, kind="stable")
            data_artist = axis.errorbar(
                group.scan_coordinate, group.y, yerr=group.sigma,
                fmt="o", color="black", ecolor="0.55", markersize=2.8,
                elinewidth=0.7, capsize=0, label="data",
            )
            model_line, = axis.plot(
                group.scan_coordinate[order], result.prediction[order],
                color="tab:red", linewidth=1.5, label="model",
            )
            background_line = axis.axhline(
                result.fixed_background, color="tab:blue", linestyle="--",
                linewidth=1.0, label="fixed background",
            )
            energy = float(np.median(group.energy_transfer_mev))
            ef = float(group.metadata["fixed_ef_mev"])
            key = (
                group.filename
                if self.objective.normalization_policy == "per_scan"
                else group.zone_key
            )
            axis.set_title(
                f"{group.filename}\n{group.zone}, E={energy:g} meV, "
                f"Ef={ef:g} meV, scale={scale_values[key]:.4g}",
                fontsize=7.5,
            )
            axis.set_xlabel(group.scan_coordinate_label, fontsize=8)
            if index % 4 == 0:
                axis.set_ylabel("corrected counts / monitor", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.15, linewidth=0.5)
            if index == 0:
                axis.legend(
                    [data_artist, model_line, background_line],
                    ["data", "model", "fixed background"],
                    loc="best", fontsize=7, frameon=False,
                )
        for axis in axes_flat[len(self.objective.prepared.groups):]:
            axis.set_visible(False)
        figure.subplots_adjust(
            left=0.06, right=0.99, bottom=0.05, top=0.94,
            hspace=0.62, wspace=0.30,
        )

    def save(self, basename: str) -> list[str]:
        output = Path(f"{basename}-provenance.json")
        payload = self.objective.provenance()
        payload.update({
            "seed": SEED,
            "seed_parameters": {
                "SJ1_meV": SEED_PARAMETERS[SEED][0],
                "J2_over_J1": SEED_PARAMETERS[SEED][1],
                "SJc_meV": SEED_PARAMETERS[SEED][2],
                "D_easy_plane_meV": 0.08853,
            },
            "parameter_values": {parameter.name: float(parameter.value) for parameter in self.parameters()},
            "parameter_ranges": {
                parameter.name: [float(parameter.bounds[0]), float(parameter.bounds[1])]
                for parameter in self.parameters()
            },
            "objective_nllf": self.nllf(),
            "fit_mode": FIT_MODE,
            "physics_parameters_fixed": FIT_MODE == "scale_only",
            "initialization_contract": (
                "stock Bumps --init=eps around complete Parameter current values"
                if FIT_MODE == "full"
                else "stock Bumps LM with ordinary fixed physics Parameters and ordinary free scales"
            ),
            "scale_only_par_handoff_warning": (
                "Bumps .par output contains free scales only; do not pass it directly to full "
                "DREAM because missing physics labels become full-range LHS under --init=eps"
                if FIT_MODE == "scale_only"
                else None
            ),
            "background_policy": "fixed independent peak-fit flat backgrounds",
            "custom_bumps_layer": False,
        })
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return [str(output)]


prepared = prepare_data(
    INPUT_PATH, BACKGROUNDS_PATH, QMC_SAMPLE_COUNT, resolution_policy=RESOLUTION_POLICY
)
objective = OfficialPythonIntensityObjective(
    prepared,
    normalization_policy=NORMALIZATION_POLICY,
    domain_policy=DOMAIN_POLICY,
    python_scheduling_policy=PYTHON_SCHEDULING_POLICY,
    python_scheduling_source=PYTHON_SCHEDULING_SOURCE,
)
fitness = OfficialPythonIntensityFitness(objective)
problem = FitProblem(
    fitness,
    name=(
        f"srmn2sb2_official_python_{SEED}_{NORMALIZATION_POLICY}_"
        f"{DOMAIN_POLICY}_{RESOLUTION_POLICY}_{FIT_MODE}_{PYTHON_SCHEDULING_POLICY}"
    ),
)
