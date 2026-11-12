#!/usr/bin/env python3
"""Extract SrMn2Sb2 constant-energy symmetric peak-pair loci.

This is a position-only, model-independent first stage for a McNally-style
dispersion analysis.  It deliberately does not import a spin-wave model and it
does not assign a fitted pair to an acoustic or optical branch.  Every eligible
low-temperature constant-energy HH or L scan is retained, including repeat
scans.  Three profile hypotheses are compared for every scan:

* a non-negative linear background;
* that background plus one Gaussian;
* that background plus a left/right Gaussian pair with independent heights and
  a shared width.

The pair centers are fitted independently on opposite sides of the nominal
integer zone center.  Their mean and half-separation are derived quantities, so
the common resolution-focusing shift is retained rather than silently forced to
zero.  The complete center covariance is emitted for downstream coordinate-root
fits.  Cooper-Nathans resolution diagnostics preserve scan-specific horizontal
collimations when they are explicitly present in raw headers and retain the
earlier local-analysis settings as a named sensitivity scenario.  Missing
vertical/mosaic configuration makes the resolution classification indeterminate,
so it is never used as a hard inclusion or censoring decision.

Real-data outputs are written only below ``local_runs/`` (ignored by git).
The script itself contains no private data and is safe to version.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "local_runs" / "srmn2sb2_pyspinw_fit_20260622"
DEFAULT_INPUT = DEFAULT_RUN / "dispersion_points.csv"
DEFAULT_OUTPUT = DEFAULT_RUN / "symmetric_pair_loci_all_scans"

MODEL_BACKGROUND = "background_only"
MODEL_ONE = "one_gaussian"
MODEL_PAIR = "symmetric_pair"
BRANCH_ASSIGNMENT = "unassigned_symmetric_pair"
FWHM_FACTOR = 2.3548200450309493

# Resolution scenarios are diagnostics.  ``comment`` is parsed from a scan's
# ExptComment when present; ``legacy`` preserves the earlier local-analysis
# assumption so its selection effect remains reproducible rather than hidden.
LEGACY_HCOL_ARCMIN = (25.0, 40.0, 40.0, 240.0)
DEFAULT_VCOL_ARCMIN = (120.0, 120.0, 120.0, 120.0)
TEMPERATURE_LABEL_TOLERANCE_K = 10.0


@dataclass(frozen=True)
class ScanGeometry:
    axis: str
    coordinate_definition: str
    coordinate: np.ndarray
    zone_center_coordinate: float
    zone_center_hkl: tuple[float, float, float]
    fixed_hkl: tuple[float | None, float | None, float | None]
    path_start_hkl: tuple[float, float, float]
    path_end_hkl: tuple[float, float, float]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{kind}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]}"


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(array)) if len(array) else float("nan")


def _parse_triplets(text: str, fallback: Sequence[float]) -> np.ndarray:
    try:
        values = [float(token) for token in str(text).replace(",", " ").split()]
    except ValueError:
        values = []
    if len(values) < len(fallback):
        values = list(fallback)
    return np.asarray(values, dtype=float)


def read_bt7_header(path: Path) -> dict[str, Any]:
    """Read inert BT-7 header/configuration evidence without executing content."""
    directives: dict[str, str] = {}
    columns: list[str] = []
    first_values: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#Columns"):
                columns = line[len("#Columns"):].split()
                break
            match = re.match(r"#(\S+)\s*(.*)", line.rstrip("\r\n"))
            if match:
                directives[match.group(1)] = match.group(2).strip()
        if columns:
            for line in stream:
                if line.strip() and not line.startswith("#"):
                    first_values = line.split()
                    break
    first_row = dict(zip(columns, first_values))
    return {
        "source_path": str(path),
        "source_sha256": _sha256_file(path),
        "directives": directives,
        "first_row": first_row,
    }


def parse_scan_configuration(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Parse per-scan fixed energy and horizontal-collimation evidence."""
    path = Path(str(rows[0].get("path", "")))
    if not path.exists():
        return {
            "status": "source_file_unavailable",
            "source_path": str(path),
            "fixed_ef_mev": fixed_ef_mev(rows),
            "horizontal_collimation_arcmin": None,
            "horizontal_collimation_source": None,
            "resolution_classification_authority": False,
            "quality_flags": ["raw_scan_header_unavailable", "resolution_configuration_incomplete"],
        }
    header = read_bt7_header(path)
    directives = header["directives"]
    first_row = header["first_row"]
    flags: list[str] = []
    fixed_text = str(directives.get("FixedE", rows[0].get("fixed_e", "")))
    fixed_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", fixed_text)
    fixed_ef = float(fixed_numbers[-1]) if fixed_numbers else fixed_ef_mev(rows)
    comment = str(directives.get("ExptComment", ""))
    collimation_match = re.search(
        r"(?<!\d)(\d{1,3})\s*[-/]\s*(\d{1,3})\s*[-/]\s*(\d{1,3})\s*[-/]\s*(\d{1,3})(?!\d)",
        comment,
    )
    horizontal_collimation = (
        tuple(float(value) for value in collimation_match.groups())
        if collimation_match else None
    )
    collimation_source = "raw_header_ExptComment" if horizontal_collimation else None
    comment_ef_match = re.search(r"\bEf\s*=?\s*([-+]?\d+(?:\.\d+)?)", comment, re.IGNORECASE)
    comment_ef = float(comment_ef_match.group(1)) if comment_ef_match else None
    if comment_ef is not None and not math.isclose(comment_ef, fixed_ef, rel_tol=0.0, abs_tol=0.05):
        flags.append("expt_comment_fixed_energy_conflicts_with_FixedE")
    device_states = {
        name: first_row.get(name)
        for name in ("PreMonoColl", "PostMonoColl", "PreAnaColl", "PostAnaColl")
    }
    if horizontal_collimation is not None:
        premono = str(device_states.get("PreMonoColl", "")).upper()
        if premono.endswith("MIN"):
            device_premono = _finite_float(premono[:-3])
            if math.isfinite(device_premono) and not math.isclose(
                device_premono, horizontal_collimation[0], rel_tol=0.0, abs_tol=0.5
            ):
                flags.append("collimation_comment_conflicts_with_device_state")
    else:
        flags.append("horizontal_collimation_unknown")
    # Vertical collimation, mono/analyzer mosaic and sample mosaic remain absent
    # even when the four horizontal values are present in ExptComment.
    flags.append("resolution_configuration_incomplete")
    return {
        "status": "partial_explicit" if horizontal_collimation is not None else "incomplete",
        "source_path": str(path),
        "source_sha256": header["source_sha256"],
        "fixed_e_directive": fixed_text,
        "fixed_ef_mev": fixed_ef,
        "expt_comment": comment,
        "expt_comment_fixed_ef_mev": comment_ef,
        "horizontal_collimation_arcmin": horizontal_collimation,
        "horizontal_collimation_source": collimation_source,
        "collimator_device_states_first_point": device_states,
        "vertical_collimation_arcmin": None,
        "mono_analyzer_mosaic_arcmin": None,
        "sample_mosaic_arcmin": None,
        "resolution_classification_authority": False,
        "quality_flags": flags,
    }


def filename_temperature_diagnostic(scan_id: str, measured_temperature_k: float) -> dict[str, Any]:
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*K(?:_|\b)", scan_id, re.IGNORECASE)
    label = float(matches[0]) if matches else None
    conflict = bool(
        label is not None
        and math.isfinite(measured_temperature_k)
        and abs(label - measured_temperature_k) > TEMPERATURE_LABEL_TOLERANCE_K
    )
    return {
        "filename_temperature_label_k": label,
        "measured_temperature_median_k": measured_temperature_k,
        "absolute_difference_k": abs(label - measured_temperature_k) if label is not None else None,
        "tolerance_k": TEMPERATURE_LABEL_TOLERANCE_K,
        "conflict": conflict,
    }


def read_scan_groups(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        required = {
            "filename", "scan_variable", "H", "K", "L", "E", "Detector",
            "Monitor", "Temp", "path", "fixed_e", "orient", "lattice",
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            row["_source_csv_row"] = str(row_number)
            groups[str(row["filename"])].append(row)
    return dict(groups), fieldnames


def classify_scan(
    scan_id: str,
    rows: Sequence[dict[str, str]],
    *,
    temperature_max_k: float,
    energy_min_mev: float,
    min_points: int,
    min_points_per_side: int,
) -> tuple[dict[str, Any], ScanGeometry | None]:
    reasons: list[str] = []
    scan_variable = str(rows[0].get("scan_variable", "")).strip().lower()
    h = np.asarray([_finite_float(row.get("H")) for row in rows])
    k = np.asarray([_finite_float(row.get("K")) for row in rows])
    l = np.asarray([_finite_float(row.get("L")) for row in rows])
    e = np.asarray([_finite_float(row.get("E")) for row in rows])
    temp = np.asarray([_finite_float(row.get("Temp")) for row in rows])
    monitor = np.asarray([_finite_float(row.get("Monitor")) for row in rows])
    detector = np.asarray([_finite_float(row.get("Detector")) for row in rows])

    finite = np.isfinite(h + k + l + e + temp + monitor + detector)
    positive_monitor = monitor > 0
    valid = finite & positive_monitor
    if int(np.sum(valid)) < min_points:
        reasons.append("too_few_finite_positive_monitor_points")
    if _median(temp[valid]) >= temperature_max_k:
        reasons.append("temperature_not_below_limit")
    energy = _median(e[valid])
    if not math.isfinite(energy) or energy < energy_min_mev:
        reasons.append("energy_below_limit_or_missing")
    if np.any(valid) and float(np.ptp(e[valid])) > 0.10:
        reasons.append("not_constant_energy")

    geometry: ScanGeometry | None = None
    if scan_variable == "qy" and np.any(valid):
        coordinate = 0.5 * (h + k)
        center = float(round(_median(coordinate[valid])))
        l_center = float(round(_median(l[valid])))
        if _median(np.abs(h[valid] - k[valid])) >= 0.05:
            reasons.append("not_h_equals_k_path")
        if float(np.ptp(l[valid])) >= 0.12 or abs(_median(l[valid]) - l_center) >= 0.15:
            reasons.append("l_not_fixed_near_integer_zone")
        axis = "H"
        coordinate_definition = "absolute H=(H+K)/2 on (H,H,L0)"
        zone_center_hkl = (center, center, l_center)
        fixed_hkl = (None, None, _median(l[valid]))
        imin = int(np.nanargmin(np.where(valid, coordinate, np.inf)))
        imax = int(np.nanargmax(np.where(valid, coordinate, -np.inf)))
        path_start = (float(h[imin]), float(k[imin]), float(l[imin]))
        path_end = (float(h[imax]), float(k[imax]), float(l[imax]))
    elif scan_variable == "qz" and np.any(valid):
        coordinate = l.copy()
        center = float(round(_median(coordinate[valid])))
        h_center = float(round(_median(h[valid])))
        k_center = float(round(_median(k[valid])))
        if float(np.ptp(h[valid])) >= 0.12 or abs(_median(h[valid]) - h_center) >= 0.15:
            reasons.append("h_not_fixed_near_integer_zone")
        if float(np.ptp(k[valid])) >= 0.12 or abs(_median(k[valid]) - k_center) >= 0.15:
            reasons.append("k_not_fixed_near_integer_zone")
        axis = "L"
        coordinate_definition = "absolute L on (H0,K0,L)"
        zone_center_hkl = (h_center, k_center, center)
        fixed_hkl = (_median(h[valid]), _median(k[valid]), None)
        imin = int(np.nanargmin(np.where(valid, coordinate, np.inf)))
        imax = int(np.nanargmax(np.where(valid, coordinate, -np.inf)))
        path_start = (float(h[imin]), float(k[imin]), float(l[imin]))
        path_end = (float(h[imax]), float(k[imax]), float(l[imax]))
    else:
        coordinate = np.full(len(rows), np.nan)
        center = float("nan")
        reasons.append("not_supported_constant_energy_hh_or_l_scan")

    if scan_variable in {"qy", "qz"} and np.any(valid):
        n_left = int(np.sum(valid & (coordinate < center - 1.0e-7)))
        n_right = int(np.sum(valid & (coordinate > center + 1.0e-7)))
        if n_left < min_points_per_side or n_right < min_points_per_side:
            reasons.append("scan_does_not_cover_both_sides_of_zone_center")
        if not reasons:
            geometry = ScanGeometry(
                axis=axis,
                coordinate_definition=coordinate_definition,
                coordinate=coordinate[valid],
                zone_center_coordinate=center,
                zone_center_hkl=zone_center_hkl,
                fixed_hkl=fixed_hkl,
                path_start_hkl=path_start,
                path_end_hkl=path_end,
            )

    audit = {
        "scan_id": scan_id,
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "scan_variable": scan_variable,
        "n_rows": len(rows),
        "n_valid_rows": int(np.sum(valid)),
        "energy_mev": energy,
        "temperature_median_k": _median(temp[valid]),
        "source_path": str(rows[0].get("path", "")),
        "source_csv_row_start": int(rows[0]["_source_csv_row"]),
        "source_csv_row_end": int(rows[-1]["_source_csv_row"]),
    }
    return audit, geometry


def prepare_profile(
    rows: Sequence[dict[str, str]], geometry: ScanGeometry
) -> dict[str, Any]:
    h = np.asarray([_finite_float(row.get("H")) for row in rows])
    k = np.asarray([_finite_float(row.get("K")) for row in rows])
    l = np.asarray([_finite_float(row.get("L")) for row in rows])
    e = np.asarray([_finite_float(row.get("E")) for row in rows])
    detector = np.asarray([_finite_float(row.get("Detector")) for row in rows])
    monitor = np.asarray([_finite_float(row.get("Monitor")) for row in rows])
    temp = np.asarray([_finite_float(row.get("Temp")) for row in rows])
    coordinate = 0.5 * (h + k) if geometry.axis == "H" else l
    good = np.isfinite(coordinate + e + detector + monitor + temp) & (monitor > 0)
    order = np.argsort(coordinate[good])
    coordinate = coordinate[good][order]
    h, k, l, e = h[good][order], k[good][order], l[good][order], e[good][order]
    detector, monitor, temp = detector[good][order], monitor[good][order], temp[good][order]
    y = detector / monitor
    sigma = np.sqrt(np.maximum(detector, 1.0)) / monitor
    finite_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    sigma_floor = float(np.median(finite_sigma) * 0.25) if len(finite_sigma) else 1.0e-8
    sigma = np.maximum(sigma, sigma_floor)
    steps = np.diff(np.unique(coordinate))
    positive_steps = steps[np.isfinite(steps) & (steps > 1.0e-9)]
    step = float(np.median(positive_steps)) if len(positive_steps) else float("nan")
    return {
        "x": coordinate,
        "h": h,
        "k": k,
        "l": l,
        "e": e,
        "detector": detector,
        "monitor": monitor,
        "temperature": temp,
        "y": y,
        "sigma": sigma,
        "sigma_floor": sigma_floor,
        "step": step,
        "normalization": "Detector/Monitor; constant-E multiplicative corrections cancel from peak centers",
        "uncertainty": "sqrt(max(Detector,1))/Monitor with a 0.25*median positive-sigma floor",
    }


def _linear_background(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(x) == 1 or x[-1] == x[0]:
        return np.full_like(x, params[0], dtype=float)
    fraction = (x - x[0]) / (x[-1] - x[0])
    return params[0] * (1.0 - fraction) + params[1] * fraction


def profile_model(model: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = _linear_background(params, x)
    if model == MODEL_BACKGROUND:
        return out
    if model == MODEL_ONE:
        height, center, width = params[2:5]
        return out + height * np.exp(-0.5 * ((x - center) / width) ** 2)
    if model == MODEL_PAIR:
        left_height, left_center, right_height, right_center, width = params[2:7]
        return (
            out
            + left_height * np.exp(-0.5 * ((x - left_center) / width) ** 2)
            + right_height * np.exp(-0.5 * ((x - right_center) / width) ** 2)
        )
    raise ValueError(f"unknown model {model!r}")


def _candidate_peak_indices(x: np.ndarray, residual: np.ndarray, mask: np.ndarray, limit: int = 4) -> list[int]:
    candidates: list[int] = []
    for index in range(1, len(x) - 1):
        if mask[index] and residual[index] >= residual[index - 1] and residual[index] >= residual[index + 1]:
            candidates.append(index)
    candidates.extend(int(index) for index in np.argsort(residual)[::-1] if mask[index])
    unique: list[int] = []
    for index in candidates:
        if index not in unique:
            unique.append(index)
        if len(unique) >= limit:
            break
    return unique


def _covariance_from_solution(solution: Any, n_parameters: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    jacobian = np.asarray(solution.jac, dtype=float)
    information = jacobian.T @ jacobian
    rank = int(np.linalg.matrix_rank(information))
    condition = float(np.linalg.cond(information))
    covariance = np.linalg.pinv(information, rcond=1.0e-12)
    dof = max(1, len(solution.fun) - n_parameters)
    scaled = covariance * (float(np.sum(solution.fun**2)) / dof)
    return covariance, scaled, rank, condition


def _fit_from_starts(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    starts: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    best: Any = None
    attempted = 0
    successful = 0
    finite_nonconverged = 0
    for start in starts:
        attempted += 1
        clipped = np.maximum(start, lower + 1.0e-12)
        clipped = np.minimum(clipped, np.where(np.isfinite(upper), upper - 1.0e-12, clipped))
        try:
            solution = least_squares(
                lambda parameters: (profile_model(model, parameters, x) - y) / sigma,
                clipped,
                bounds=(lower, upper),
                x_scale="jac",
                max_nfev=2000,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(solution.x)) or not np.all(np.isfinite(solution.fun)):
            continue
        if not bool(solution.success):
            finite_nonconverged += 1
            continue
        successful += 1
        chi2 = float(np.sum(solution.fun**2))
        if best is None or chi2 < float(np.sum(best.fun**2)):
            best = solution
    if best is None:
        return {
            "model": model,
            "status": "no_successful_fit",
            "starts_attempted": attempted,
            "starts_finite_nonconverged": finite_nonconverged,
        }

    n_parameters = len(best.x)
    covariance, scaled_covariance, rank, condition = _covariance_from_solution(best, n_parameters)
    chi2 = float(np.sum(best.fun**2))
    n = len(x)
    dof = max(1, n - n_parameters)
    distance_to_bounds = np.minimum(
        np.abs(best.x - lower),
        np.where(np.isfinite(upper), np.abs(upper - best.x), np.inf),
    )
    scale = np.maximum(np.abs(best.x), 1.0)
    at_bound = distance_to_bounds <= 1.0e-6 * scale
    return {
        "model": model,
        "status": "ok",
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "function_evaluations": int(best.nfev),
        "optimality": float(best.optimality),
        "starts_attempted": attempted,
        "starts_successful": successful,
        "starts_finite_nonconverged": finite_nonconverged,
        "n_points": n,
        "n_parameters": n_parameters,
        "degrees_of_freedom": dof,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "aic": chi2 + 2.0 * n_parameters,
        "bic": chi2 + n_parameters * math.log(max(2, n)),
        "parameters": best.x,
        "parameter_covariance": covariance,
        "parameter_covariance_rescaled": scaled_covariance,
        "jacobian_rank": rank,
        "information_condition_number": condition,
        "parameters_at_bound": at_bound,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }


def fit_profile_models(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    zone_center: float,
) -> dict[str, dict[str, Any]]:
    """Fit background, one-peak and symmetric-pair alternatives deterministically."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    good = np.isfinite(x + y + sigma) & (sigma > 0)
    x, y, sigma = x[good], y[good], sigma[good]
    order = np.argsort(x)
    x, y, sigma = x[order], y[order], sigma[order]
    if len(x) < 8 or x[-1] <= x[0]:
        raise ValueError("profile needs at least eight finite points spanning a non-zero coordinate range")
    span = float(x[-1] - x[0])
    steps = np.diff(np.unique(x))
    steps = steps[steps > 1.0e-9]
    step = float(np.median(steps)) if len(steps) else span / max(1, len(x) - 1)
    minimum_width = max(0.35 * step, 1.0e-4)
    maximum_width = max(1.01 * minimum_width, 0.40 * span)
    y_scale = max(float(np.max(y)), float(np.median(sigma)), 1.0e-12)
    background = max(float(np.percentile(y, 20.0)), 0.0)
    background_starts = [
        np.asarray([
            max(float(np.percentile(y[: max(2, len(y) // 5)], q)), 0.0),
            max(float(np.percentile(y[-max(2, len(y) // 5):], q)), 0.0),
        ])
        for q in (10.0, 30.0, 50.0)
    ]
    background_upper = max(5.0 * y_scale, background + 10.0 * float(np.max(sigma)))
    variants: dict[str, dict[str, Any]] = {}
    variants[MODEL_BACKGROUND] = _fit_from_starts(
        MODEL_BACKGROUND,
        x,
        y,
        sigma,
        background_starts,
        np.asarray([0.0, 0.0]),
        np.asarray([background_upper, background_upper]),
    )

    residual = y - background
    all_mask = np.ones(len(x), dtype=bool)
    peak_indices = _candidate_peak_indices(x, residual, all_mask, limit=4)
    width_starts = sorted({
        minimum_width * 1.2,
        min(maximum_width * 0.95, step),
        min(maximum_width * 0.95, 2.0 * step),
        min(maximum_width * 0.95, 0.10 * span),
    })
    one_starts: list[np.ndarray] = []
    for index in peak_indices:
        for width in width_starts:
            one_starts.append(np.asarray([
                background, background, max(float(residual[index]), float(np.median(sigma))),
                float(x[index]), width,
            ]))
    variants[MODEL_ONE] = _fit_from_starts(
        MODEL_ONE,
        x,
        y,
        sigma,
        one_starts,
        np.asarray([0.0, 0.0, 0.0, float(x[0]), minimum_width]),
        np.asarray([background_upper, background_upper, np.inf, float(x[-1]), maximum_width]),
    )

    epsilon = max(0.05 * step, 1.0e-7)
    left_mask = x < zone_center - epsilon
    right_mask = x > zone_center + epsilon
    if int(np.sum(left_mask)) >= 2 and int(np.sum(right_mask)) >= 2:
        left_indices = _candidate_peak_indices(x, residual, left_mask, limit=3)
        right_indices = _candidate_peak_indices(x, residual, right_mask, limit=3)
        pair_starts: list[np.ndarray] = []
        for left_index in left_indices:
            for right_index in right_indices:
                for width in width_starts:
                    pair_starts.append(np.asarray([
                        background,
                        background,
                        max(float(residual[left_index]), float(np.median(sigma))),
                        float(x[left_index]),
                        max(float(residual[right_index]), float(np.median(sigma))),
                        float(x[right_index]),
                        width,
                    ]))
        left_max = float(np.max(x[left_mask]))
        right_min = float(np.min(x[right_mask]))
        variants[MODEL_PAIR] = _fit_from_starts(
            MODEL_PAIR,
            x,
            y,
            sigma,
            pair_starts,
            np.asarray([0.0, 0.0, 0.0, float(x[0]), 0.0, right_min, minimum_width]),
            np.asarray([background_upper, background_upper, np.inf, left_max, np.inf, float(x[-1]), maximum_width]),
        )
    else:
        variants[MODEL_PAIR] = {"model": MODEL_PAIR, "status": "insufficient_points_on_both_sides"}
    return variants


def summarize_pair_fit(pair_fit: dict[str, Any], zone_center: float) -> dict[str, Any] | None:
    if pair_fit.get("status") != "ok":
        return None
    parameters = np.asarray(pair_fit["parameters"], dtype=float)
    covariance = np.asarray(pair_fit["parameter_covariance"], dtype=float)
    scaled_covariance = np.asarray(pair_fit["parameter_covariance_rescaled"], dtype=float)
    left_center = float(parameters[3])
    right_center = float(parameters[5])
    center_indices = np.asarray([3, 5])
    center_covariance = covariance[np.ix_(center_indices, center_indices)]
    center_covariance_scaled = scaled_covariance[np.ix_(center_indices, center_indices)]
    covariance_scale = max(1.0, float(pair_fit["reduced_chi2"]))
    center_covariance_conservative = center_covariance * covariance_scale
    transform = np.asarray([[0.5, 0.5], [-0.5, 0.5]])
    mean_sep_covariance = transform @ center_covariance_conservative @ transform.T
    mean_sep_covariance_counting = transform @ center_covariance @ transform.T
    mean_sep_covariance_scaled = transform @ center_covariance_scaled @ transform.T
    return {
        "left_center_rlu": left_center,
        "right_center_rlu": right_center,
        "left_center_sigma_rlu": math.sqrt(max(float(center_covariance_conservative[0, 0]), 0.0)),
        "right_center_sigma_rlu": math.sqrt(max(float(center_covariance_conservative[1, 1]), 0.0)),
        "left_center_sigma_counting_rlu": math.sqrt(max(float(center_covariance[0, 0]), 0.0)),
        "right_center_sigma_counting_rlu": math.sqrt(max(float(center_covariance[1, 1]), 0.0)),
        "left_center_sigma_rescaled_rlu": math.sqrt(max(float(center_covariance_scaled[0, 0]), 0.0)),
        "right_center_sigma_rescaled_rlu": math.sqrt(max(float(center_covariance_scaled[1, 1]), 0.0)),
        "center_covariance_rlu2": center_covariance_conservative,
        "center_covariance_counting_rlu2": center_covariance,
        "center_covariance_rescaled_rlu2": center_covariance_scaled,
        "covariance_scale_applied": covariance_scale,
        "covariance_primary_definition": "counting covariance multiplied by max(1, reduced_chi2)",
        "common_center_rlu": 0.5 * (left_center + right_center),
        "common_center_offset_rlu": 0.5 * (left_center + right_center) - zone_center,
        "common_center_sigma_rlu": math.sqrt(max(float(mean_sep_covariance[0, 0]), 0.0)),
        "half_separation_rlu": 0.5 * (right_center - left_center),
        "full_separation_rlu": right_center - left_center,
        "half_separation_sigma_rlu": math.sqrt(max(float(mean_sep_covariance[1, 1]), 0.0)),
        "mean_half_separation_covariance": mean_sep_covariance,
        "mean_half_separation_covariance_counting": mean_sep_covariance_counting,
        "mean_half_separation_covariance_rescaled": mean_sep_covariance_scaled,
        "shared_sigma_rlu": float(parameters[6]),
        "shared_fwhm_rlu": FWHM_FACTOR * float(parameters[6]),
        "left_height": float(parameters[2]),
        "right_height": float(parameters[4]),
    }


def fixed_ef_mev(rows: Sequence[dict[str, str]]) -> float:
    text = str(rows[0].get("fixed_e", ""))
    numbers = [float(token) for token in text.replace("Ef", " ").replace("=", " ").split() if token.replace(".", "", 1).isdigit()]
    if numbers:
        return numbers[0]
    return 14.7 if "14" in text else 28.0


def _resolution_experiment(ef: float, horizontal_collimation: Sequence[float]) -> dict[str, Any]:
    return {
        "efixed": float(ef),
        "infin": -1,
        "dir1": 1,
        "dir2": 1,
        "hcol": np.asarray(horizontal_collimation, dtype=float),
        "vcol": np.asarray(DEFAULT_VCOL_ARCMIN, dtype=float),
        "mono": {"tau": "pg(002)", "mosaic": 30.0},
        "ana": {"tau": "pg(002)", "mosaic": 30.0},
        "sample": {"mosaic": 30.0, "vmosaic": 30.0},
        "arms": [200.0, 200.0, 150.0, 150.0, 100.0],
        "horifoc": -1,
        "moncor": 1,
        "method": 0,
    }


def resolution_diagnostics(
    rows: Sequence[dict[str, str]],
    geometry: ScanGeometry,
    profile: dict[str, Any],
    pair: dict[str, Any] | None = None,
    *,
    horizontal_collimation: Sequence[float] = LEGACY_HCOL_ARCMIN,
    scenario_name: str = "legacy_assumption",
    configuration_source: str = "legacy_local_analysis_assumption",
) -> dict[str, Any]:
    """Return projected resolution at the fitted loci and magnetic zone center."""
    try:
        # The installed rescalculator optionally probes a torch backend which can
        # emit binary-compatibility text even when its NumPy backend is available.
        # Capture that dependency chatter, but never suppress an actual exception.
        dependency_stderr = io.StringIO()
        with contextlib.redirect_stderr(dependency_stderr):
            from lattice_calculator import Lattice  # type: ignore
            from rescalculator import TASResolution  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent path
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    try:
        lattice_values = _parse_triplets(rows[0].get("lattice", ""), (4.58947, 4.58947, 7.7189, 90, 90, 120))
        if len(lattice_values) < 6:
            raise ValueError("lattice metadata needs six values")
        orient_values = _parse_triplets(rows[0].get("orient", ""), (1, 1, 0, 0, 0, 1))
        if len(orient_values) < 6:
            raise ValueError("orientation metadata needs six values")
        evaluation_coordinates = [geometry.zone_center_coordinate]
        if pair is not None:
            evaluation_coordinates.extend([
                float(pair["left_center_rlu"]),
                float(pair["right_center_rlu"]),
            ])
        evaluation_coordinates = sorted(set(evaluation_coordinates))
        evaluation_hkl = [
            (
                (coordinate, coordinate, float(geometry.fixed_hkl[2]))
                if geometry.axis == "H"
                else (float(geometry.fixed_hkl[0]), float(geometry.fixed_hkl[1]), coordinate)
            )
            for coordinate in evaluation_coordinates
        ]
        n = len(evaluation_coordinates)
        lattice = Lattice(
            a=np.full(n, lattice_values[0]), b=np.full(n, lattice_values[1]), c=np.full(n, lattice_values[2]),
            alpha=np.full(n, lattice_values[3]), beta=np.full(n, lattice_values[4]), gamma=np.full(n, lattice_values[5]),
            orient1=np.asarray([orient_values[:3]], dtype=float),
            orient2=np.asarray([orient_values[3:6]], dtype=float),
        )
        calculator = TASResolution(lattice, backend="numpy")
        ef = fixed_ef_mev(rows)
        experiments = [_resolution_experiment(ef, horizontal_collimation) for _ in range(n)]
        _, precision = calculator.ResMatS(
            np.asarray([point[0] for point in evaluation_hkl], dtype=float),
            np.asarray([point[1] for point in evaluation_hkl], dtype=float),
            np.asarray([point[2] for point in evaluation_hkl], dtype=float),
            np.full(n, _median(profile["e"]), dtype=float),
            experiments,
        )
        precision = np.asarray(precision, dtype=float)
        xvec = np.asarray(lattice.x, dtype=float)
        yvec = np.asarray(lattice.y, dtype=float)
        zvec = np.asarray(lattice.z, dtype=float)
        point_diagnostics: list[dict[str, float]] = []
        repair_count = 0
        for index in range(n):
            matrix = 0.5 * (precision[:, :, index] + precision[:, :, index].T)
            try:
                covariance = np.linalg.inv(matrix)
                np.linalg.cholesky(0.5 * (covariance + covariance.T))
            except np.linalg.LinAlgError:
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                eigenvalues = np.clip(eigenvalues, 1.0e-12, None)
                covariance = np.linalg.inv((eigenvectors * eigenvalues) @ eigenvectors.T)
                repair_count += 1
            q_indices = [0, 1, 3]
            sigma_qq = covariance[np.ix_(q_indices, q_indices)]
            sigma_qe = covariance[q_indices, 2]
            sigma_ee = float(covariance[2, 2])
            sigma_qq_given_e = sigma_qq - np.outer(sigma_qe, sigma_qe) / max(sigma_ee, 1.0e-15)
            if geometry.axis == "H":
                vector = 0.5 * np.asarray([
                    xvec[0, index] + xvec[1, index],
                    yvec[0, index] + yvec[1, index],
                    zvec[0, index] + zvec[1, index],
                ])
            else:
                vector = np.asarray([xvec[2, index], yvec[2, index], zvec[2, index]])
            coordinate_var = float(vector @ sigma_qq @ vector)
            coordinate_var_given_e = float(vector @ sigma_qq_given_e @ vector)
            sigma_eq = covariance[2, q_indices]
            energy_var_given_q = sigma_ee - float(sigma_eq @ np.linalg.solve(sigma_qq, sigma_eq.T))
            point_diagnostics.append({
                "coordinate_rlu": float(evaluation_coordinates[index]),
                "coordinate_sigma_marginal_rlu": math.sqrt(max(coordinate_var, 0.0)),
                "coordinate_sigma_conditional_on_energy_rlu": math.sqrt(max(coordinate_var_given_e, 0.0)),
                "energy_sigma_marginal_mev": math.sqrt(max(sigma_ee, 0.0)),
                "energy_sigma_conditional_on_q_mev": math.sqrt(max(energy_var_given_q, 0.0)),
            })
        coordinate_sigmas = np.asarray([point["coordinate_sigma_marginal_rlu"] for point in point_diagnostics])
        conditional_coordinate_sigmas = np.asarray([
            point["coordinate_sigma_conditional_on_energy_rlu"] for point in point_diagnostics
        ])
        energy_sigmas = np.asarray([point["energy_sigma_marginal_mev"] for point in point_diagnostics])
        conditional_energy_sigmas = np.asarray([
            point["energy_sigma_conditional_on_q_mev"] for point in point_diagnostics
        ])
        return {
            "status": "ok",
            "dependency_stderr_suppressed": bool(dependency_stderr.getvalue()),
            "method": "Cooper-Nathans precision matrix inverted to a 4D Gaussian covariance",
            "classification_width": "full pair separation / marginal projected coordinate FWHM",
            "evaluation_coordinates": "fitted left/right loci and magnetic zone center",
            "fixed_ef_mev": ef,
            "scenario_name": scenario_name,
            "configuration_source": configuration_source,
            "assumptions_review_status": "incomplete_configuration_sensitivity_scenario",
            "horizontal_collimation_arcmin": list(horizontal_collimation),
            "vertical_collimation_arcmin": list(DEFAULT_VCOL_ARCMIN),
            "mono_analyzer": "PG(002), 30 arcmin mosaic",
            "sample_mosaic_arcmin": 30.0,
            "covariance_repairs": repair_count,
            "coordinate_fwhm_marginal_median_rlu": FWHM_FACTOR * float(np.median(coordinate_sigmas)),
            "coordinate_fwhm_conditional_on_energy_median_rlu": FWHM_FACTOR * float(np.median(conditional_coordinate_sigmas)),
            "energy_sigma_marginal_median_mev": float(np.median(energy_sigmas)),
            "energy_sigma_conditional_on_q_median_mev": float(np.median(conditional_energy_sigmas)),
            "energy_fwhm_marginal_median_mev": FWHM_FACTOR * float(np.median(energy_sigmas)),
            "points": point_diagnostics,
        }
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic, never substitute
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def nearest_resolution_point(resolution: dict[str, Any], coordinate: float) -> dict[str, Any] | None:
    points = resolution.get("points", []) if resolution.get("status") == "ok" else []
    if not points:
        return None
    return min(points, key=lambda point: abs(float(point["coordinate_rlu"]) - coordinate))


def summarize_resolution_scenario(
    resolution: dict[str, Any], pair: dict[str, Any] | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scenario_name": resolution.get("scenario_name"),
        "status": resolution.get("status"),
        "horizontal_collimation_arcmin": resolution.get("horizontal_collimation_arcmin"),
        "configuration_source": resolution.get("configuration_source"),
        "full_separation_over_projected_q_fwhm": None,
        "resolved_by_scenario": None,
        "common_center_offset_over_projected_q_fwhm": None,
    }
    if resolution.get("status") != "ok" or pair is None:
        return summary
    local_points = [
        nearest_resolution_point(resolution, float(pair[key]))
        for key in ("left_center_rlu", "right_center_rlu")
    ]
    local_fwhm = [
        FWHM_FACTOR * float(point["coordinate_sigma_marginal_rlu"])
        for point in local_points if point is not None
    ]
    local_energy_sigma = [
        float(point["energy_sigma_marginal_mev"])
        for point in local_points if point is not None
    ]
    projected_fwhm = max(local_fwhm) if local_fwhm else None
    separation = float(pair["full_separation_rlu"])
    summary.update({
        "projected_coordinate_resolution_fwhm_rlu": projected_fwhm,
        "energy_resolution_sigma_mev": max(local_energy_sigma) if local_energy_sigma else None,
        "full_separation_over_projected_q_fwhm": (
            separation / projected_fwhm if projected_fwhm not in (None, 0.0) else None
        ),
        "resolved_by_scenario": (
            separation >= projected_fwhm if projected_fwhm not in (None, 0.0) else None
        ),
        "common_center_offset_over_projected_q_fwhm": (
            abs(float(pair["common_center_offset_rlu"])) / projected_fwhm
            if projected_fwhm not in (None, 0.0) else None
        ),
    })
    return summary


def _hkl_for_coordinate(geometry: ScanGeometry, coordinate: float) -> tuple[float, float, float]:
    if geometry.axis == "H":
        return (coordinate, coordinate, float(geometry.fixed_hkl[2]))
    return (float(geometry.fixed_hkl[0]), float(geometry.fixed_hkl[1]), coordinate)


def build_quality(
    variants: dict[str, dict[str, Any]],
    selected_model: str,
    pair: dict[str, Any] | None,
    resolution: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[list[str], bool, bool, float | None]:
    flags: list[str] = []
    if pair is None:
        return ["pair_fit_unavailable"], False, True, None
    pair_fit = variants[MODEL_PAIR]
    if not bool(pair_fit.get("optimizer_success", False)):
        flags.append("pair_optimizer_not_converged")
    competitor_bics = [
        float(variant["bic"])
        for name, variant in variants.items()
        if name != MODEL_PAIR and variant.get("status") == "ok"
    ]
    delta_bic = min(competitor_bics) - float(pair_fit["bic"]) if competitor_bics else None
    if selected_model != MODEL_PAIR:
        flags.append("pair_not_selected_by_bic")
    if delta_bic is None or delta_bic < 2.0:
        flags.append("weak_pair_preference_delta_bic_lt_2")
    if int(pair_fit["jacobian_rank"]) < int(pair_fit["n_parameters"]):
        flags.append("fit_covariance_rank_deficient")
    at_bound = np.asarray(pair_fit["parameters_at_bound"], dtype=bool)
    if bool(at_bound[3]) or bool(at_bound[5]):
        flags.append("fitted_center_at_bound")
    step = float(profile["step"])
    if math.isfinite(step) and float(pair["shared_fwhm_rlu"]) < step:
        flags.append("pair_fwhm_below_scan_step")
    separation = 2.0 * float(pair["half_separation_rlu"])
    largest_center_sigma = max(float(pair["left_center_sigma_rlu"]), float(pair["right_center_sigma_rlu"]))
    if not math.isfinite(largest_center_sigma) or largest_center_sigma > 0.5 * separation:
        flags.append("center_uncertainty_large_relative_to_pair_separation")
    if float(pair_fit["reduced_chi2"]) > 3.0:
        flags.append("profile_reduced_chi2_gt_3")

    if resolution.get("status") != "ok":
        flags.append("resolution_sensitivity_unavailable")
    else:
        if resolution.get("assumptions_review_status") != "reviewed":
            flags.append("resolution_classification_indeterminate_incomplete_configuration")
    exclusion_flags = {
        "pair_not_selected_by_bic",
        "weak_pair_preference_delta_bic_lt_2",
        "fit_covariance_rank_deficient",
        "fitted_center_at_bound",
        "pair_fwhm_below_scan_step",
        "center_uncertainty_large_relative_to_pair_separation",
        "profile_reduced_chi2_gt_3",
        "pair_optimizer_not_converged",
    }
    include = not any(flag in exclusion_flags for flag in flags)
    # Censoring is only authoritative with a complete/reviewed configuration.
    # No scan in the present corpus meets that bar, so scenario outcomes are
    # recorded separately and never silently turned into include/exclude truth.
    return flags, include, False, delta_bic


def fit_scan(
    scan_id: str,
    rows: Sequence[dict[str, str]],
    geometry: ScanGeometry,
    *,
    source_csv: Path,
    source_csv_sha256: str,
    resolution_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = prepare_profile(rows, geometry)
    variants = fit_profile_models(
        profile["x"], profile["y"], profile["sigma"], geometry.zone_center_coordinate
    )
    successful = [variant for variant in variants.values() if variant.get("status") == "ok"]
    selected = min(successful, key=lambda variant: float(variant["bic"])) if successful else None
    selected_model = str(selected["model"]) if selected else "none"
    pair = summarize_pair_fit(variants[MODEL_PAIR], geometry.zone_center_coordinate)
    scan_configuration = parse_scan_configuration(rows)
    temperature_diagnostic = filename_temperature_diagnostic(
        scan_id, float(np.median(profile["temperature"]))
    )
    if resolution_mode == "off":
        resolution_scenarios = {
            "legacy_assumption": {"status": "unavailable", "error": "disabled_by_cli"}
        }
    else:
        resolution_scenarios: dict[str, dict[str, Any]] = {}
        explicit_hcol = scan_configuration.get("horizontal_collimation_arcmin")
        if explicit_hcol is not None:
            resolution_scenarios["header_horizontal_collimation"] = resolution_diagnostics(
                rows, geometry, profile, pair,
                horizontal_collimation=explicit_hcol,
                scenario_name="header_horizontal_collimation",
                configuration_source=str(scan_configuration.get("horizontal_collimation_source")),
            )
        resolution_scenarios["legacy_assumption"] = resolution_diagnostics(
            rows, geometry, profile, pair,
            horizontal_collimation=LEGACY_HCOL_ARCMIN,
            scenario_name="legacy_assumption",
            configuration_source="legacy_local_analysis_assumption",
        )
        if resolution_mode == "required" and not any(
            scenario.get("status") == "ok" for scenario in resolution_scenarios.values()
        ):
            errors = {name: scenario.get("error") for name, scenario in resolution_scenarios.items()}
            raise RuntimeError(f"all resolution scenarios failed for {scan_id}: {errors}")
    preferred_scenario_name = (
        "header_horizontal_collimation"
        if resolution_scenarios.get("header_horizontal_collimation", {}).get("status") == "ok"
        else "legacy_assumption"
    )
    resolution = resolution_scenarios[preferred_scenario_name]
    scenario_summaries = {
        name: summarize_resolution_scenario(scenario, pair)
        for name, scenario in resolution_scenarios.items()
    }
    if pair is not None:
        preferred_summary = scenario_summaries[preferred_scenario_name]
        pair["projected_coordinate_resolution_fwhm_rlu"] = preferred_summary.get(
            "projected_coordinate_resolution_fwhm_rlu"
        )
        pair["energy_resolution_sigma_mev"] = preferred_summary.get("energy_resolution_sigma_mev")
        pair["full_separation_over_projected_resolution_fwhm"] = preferred_summary.get(
            "full_separation_over_projected_q_fwhm"
        )
        pair["resolution_scenarios"] = scenario_summaries
    quality_flags, include, censored, delta_bic = build_quality(
        variants, selected_model, pair, resolution, profile
    )
    quality_flags.extend(str(flag) for flag in scan_configuration.get("quality_flags", []))
    if temperature_diagnostic["conflict"]:
        quality_flags.append("filename_temperature_label_conflicts_with_recorded_temperature")
    quality_flags = list(dict.fromkeys(quality_flags))
    pair_id = _stable_id("pair", scan_id, geometry.axis, _median(profile["e"]))
    canonical_rows = [
        [row.get(name, "") for name in ("H", "K", "L", "E", "Detector", "Monitor", "Temp")]
        for row in rows
    ]
    scan_data_sha256 = hashlib.sha256(
        json.dumps(canonical_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    scan_result = {
        "schema_version": "srmn2sb2.symmetric_pair_scan_fit.v1",
        "scan_id": scan_id,
        "pair_id": pair_id,
        "source": {
            "catalog_csv": str(source_csv),
            "catalog_csv_sha256": source_csv_sha256,
            "source_path": str(rows[0].get("path", "")),
            "source_csv_rows": [int(row["_source_csv_row"]) for row in rows],
            "scan_data_sha256": scan_data_sha256,
            "raw_scan_file_sha256": scan_configuration.get("source_sha256"),
        },
        "metadata": {
            "date": rows[0].get("date", ""),
            "title": rows[0].get("title", ""),
            "scan_variable": rows[0].get("scan_variable", ""),
            "orientation": rows[0].get("orient", ""),
            "lattice": rows[0].get("lattice", ""),
            "fixed_e": rows[0].get("fixed_e", ""),
            "fixed_ef_mev": fixed_ef_mev(rows),
            "temperature_min_k": float(np.min(profile["temperature"])),
            "temperature_median_k": float(np.median(profile["temperature"])),
            "temperature_max_k": float(np.max(profile["temperature"])),
            "filename_temperature_diagnostic": temperature_diagnostic,
        },
        "geometry": {
            "axis": geometry.axis,
            "coordinate_definition": geometry.coordinate_definition,
            "coordinate_is_absolute_rlu": True,
            "zone_center_coordinate_rlu": geometry.zone_center_coordinate,
            "zone_center_hkl": geometry.zone_center_hkl,
            "fixed_hkl": geometry.fixed_hkl,
            "scan_path_hkl_start": geometry.path_start_hkl,
            "scan_path_hkl_end": geometry.path_end_hkl,
        },
        "profile": {
            "n_points": len(profile["x"]),
            "energy_mev": _median(profile["e"]),
            "coordinate_min_rlu": float(profile["x"][0]),
            "coordinate_max_rlu": float(profile["x"][-1]),
            "coordinate_step_median_rlu": profile["step"],
            "normalization": profile["normalization"],
            "uncertainty": profile["uncertainty"],
            "sigma_floor": profile["sigma_floor"],
        },
        "model_selection": {
            "criterion": "minimum BIC = chi2 + k*ln(n), with supplied counting-statistics errors",
            "selected_model": selected_model,
            "pair_delta_bic_vs_best_nonpair": delta_bic,
        },
        "variants": variants,
        "pair": pair,
        "scan_configuration": scan_configuration,
        "resolution": resolution,
        "resolution_scenarios": resolution_scenarios,
        "resolution_scenario_summaries": scenario_summaries,
        "resolution_classification": "indeterminate_incomplete_configuration",
        "preferred_resolution_scenario": preferred_scenario_name,
        "quality_flags": quality_flags,
        "include_pair_in_position_fit": include,
        "resolved": None,
        "censored": censored,
        "branch_assignment": BRANCH_ASSIGNMENT,
    }

    locus_points: list[dict[str, Any]] = []
    if pair is not None:
        center_covariance = pair["center_covariance_rlu2"]
        center_covariance_counting = pair["center_covariance_counting_rlu2"]
        center_covariance_rescaled = pair["center_covariance_rescaled_rlu2"]
        for side, center_key, sigma_key, sigma_scaled_key in (
            ("left", "left_center_rlu", "left_center_sigma_rlu", "left_center_sigma_rescaled_rlu"),
            ("right", "right_center_rlu", "right_center_sigma_rlu", "right_center_sigma_rescaled_rlu"),
        ):
            coordinate = float(pair[center_key])
            hkl = _hkl_for_coordinate(geometry, coordinate)
            resolution_point = nearest_resolution_point(resolution, coordinate)
            energy_sigma = (
                float(resolution_point["energy_sigma_marginal_mev"])
                if resolution_point is not None else None
            )
            point = {
                "schema_version": "srmn2sb2.symmetric_pair_locus.v1",
                "point_id": _stable_id("locus", scan_id, geometry.axis, _median(profile["e"]), side),
                "pair_id": pair_id,
                "scan_id": scan_id,
                "source_path": str(rows[0].get("path", "")),
                "source_catalog_csv": str(source_csv),
                "source_catalog_sha256": source_csv_sha256,
                "date": rows[0].get("date", ""),
                "title": rows[0].get("title", ""),
                "scan_variable": rows[0].get("scan_variable", ""),
                "orientation": rows[0].get("orient", ""),
                "lattice": rows[0].get("lattice", ""),
                "fixed_e": rows[0].get("fixed_e", ""),
                "fixed_ef_mev": fixed_ef_mev(rows),
                "temperature_min_k": float(np.min(profile["temperature"])),
                "temperature_median_k": float(np.median(profile["temperature"])),
                "temperature_max_k": float(np.max(profile["temperature"])),
                "filename_temperature_label_k": temperature_diagnostic["filename_temperature_label_k"],
                "filename_temperature_conflict": temperature_diagnostic["conflict"],
                "axis": geometry.axis,
                "side": side,
                "energy_mev": _median(profile["e"]),
                "energy_sigma_mev": energy_sigma,
                "h": hkl[0], "k": hkl[1], "l": hkl[2],
                "coordinate_rlu": coordinate,
                "coordinate_offset_rlu": coordinate - geometry.zone_center_coordinate,
                "coordinate_sigma_rlu": float(pair[sigma_key]),
                "coordinate_sigma_counting_rlu": float(
                    pair[f"{side}_center_sigma_counting_rlu"]
                ),
                "coordinate_sigma_rescaled_rlu": float(pair[sigma_scaled_key]),
                "pair_coordinate_covariance_rlu2": center_covariance,
                "pair_coordinate_covariance_counting_rlu2": center_covariance_counting,
                "pair_coordinate_covariance_rescaled_rlu2": center_covariance_rescaled,
                "zone_center_hkl": geometry.zone_center_hkl,
                "scan_path_hkl_start": geometry.path_start_hkl,
                "scan_path_hkl_end": geometry.path_end_hkl,
                "coordinate_definition": geometry.coordinate_definition,
                "coordinate_is_absolute_rlu": True,
                "projected_coordinate_resolution_sigma_rlu": (
                    float(resolution_point["coordinate_sigma_marginal_rlu"])
                    if resolution_point is not None else None
                ),
                "projected_coordinate_resolution_fwhm_rlu": (
                    FWHM_FACTOR * float(resolution_point["coordinate_sigma_marginal_rlu"])
                    if resolution_point is not None else None
                ),
                "resolution_status": resolution.get("status"),
                "resolution_classification": "indeterminate_incomplete_configuration",
                "preferred_resolution_scenario": preferred_scenario_name,
                "resolution_scenario_summaries": scenario_summaries,
                "scan_configuration_status": scan_configuration.get("status"),
                "horizontal_collimation_arcmin": scan_configuration.get("horizontal_collimation_arcmin"),
                "horizontal_collimation_source": scan_configuration.get("horizontal_collimation_source"),
                "resolved": None,
                "censored": censored,
                "branch_assignment": BRANCH_ASSIGNMENT,
                "include": include,
                "quality_flags": quality_flags,
                "selected_profile_model": selected_model,
                "pair_delta_bic_vs_best_nonpair": delta_bic,
            }
            locus_points.append(point)
    return scan_result, locus_points


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def write_locus_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    scalar_fields = [
        "schema_version", "point_id", "pair_id", "scan_id", "source_path",
        "date", "title", "scan_variable", "orientation", "lattice", "fixed_e", "fixed_ef_mev",
        "temperature_min_k", "temperature_median_k", "temperature_max_k",
        "filename_temperature_label_k", "filename_temperature_conflict", "axis", "side",
        "energy_mev", "energy_sigma_mev", "h", "k", "l", "coordinate_rlu",
        "coordinate_offset_rlu", "coordinate_sigma_rlu", "coordinate_sigma_counting_rlu",
        "coordinate_sigma_rescaled_rlu",
        "coordinate_definition", "coordinate_is_absolute_rlu",
        "projected_coordinate_resolution_sigma_rlu", "projected_coordinate_resolution_fwhm_rlu",
        "resolution_status", "resolution_classification", "preferred_resolution_scenario",
        "scan_configuration_status", "horizontal_collimation_source",
        "resolved", "censored", "branch_assignment", "include",
        "selected_profile_model", "pair_delta_bic_vs_best_nonpair",
    ]
    nested_fields = [
        "zone_center_hkl", "scan_path_hkl_start", "scan_path_hkl_end",
        "pair_coordinate_covariance_rlu2", "pair_coordinate_covariance_counting_rlu2",
        "pair_coordinate_covariance_rescaled_rlu2", "horizontal_collimation_arcmin",
        "resolution_scenario_summaries", "quality_flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields + nested_fields)
        writer.writeheader()
        for record in records:
            row = {field: _jsonable(record.get(field)) for field in scalar_fields}
            row.update({field: json.dumps(_jsonable(record.get(field)), separators=(",", ":")) for field in nested_fields})
            writer.writerow(row)


def write_scan_summary_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "scan_id", "pair_id", "axis", "energy_mev", "zone_center_hkl", "selected_model",
        "pair_delta_bic_vs_best_nonpair", "left_center_rlu", "right_center_rlu",
        "half_separation_rlu", "half_separation_sigma_rlu", "common_center_offset_rlu",
        "projected_coordinate_resolution_fwhm_rlu", "energy_sigma_marginal_mev",
        "resolution_classification", "preferred_resolution_scenario",
        "filename_temperature_conflict", "resolved", "censored", "include", "quality_flags", "source_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            pair = record.get("pair") or {}
            resolution = record.get("resolution") or {}
            writer.writerow({
                "scan_id": record["scan_id"],
                "pair_id": record["pair_id"],
                "axis": record["geometry"]["axis"],
                "energy_mev": record["profile"]["energy_mev"],
                "zone_center_hkl": json.dumps(_jsonable(record["geometry"]["zone_center_hkl"]), separators=(",", ":")),
                "selected_model": record["model_selection"]["selected_model"],
                "pair_delta_bic_vs_best_nonpair": record["model_selection"]["pair_delta_bic_vs_best_nonpair"],
                "left_center_rlu": pair.get("left_center_rlu"),
                "right_center_rlu": pair.get("right_center_rlu"),
                "half_separation_rlu": pair.get("half_separation_rlu"),
                "half_separation_sigma_rlu": pair.get("half_separation_sigma_rlu"),
                "common_center_offset_rlu": pair.get("common_center_offset_rlu"),
                "projected_coordinate_resolution_fwhm_rlu": pair.get("projected_coordinate_resolution_fwhm_rlu"),
                "energy_sigma_marginal_mev": resolution.get("energy_sigma_marginal_median_mev"),
                "resolution_classification": record["resolution_classification"],
                "preferred_resolution_scenario": record["preferred_resolution_scenario"],
                "filename_temperature_conflict": record["metadata"]["filename_temperature_diagnostic"]["conflict"],
                "resolved": record["resolved"],
                "censored": record["censored"],
                "include": record["include_pair_in_position_fit"],
                "quality_flags": json.dumps(record["quality_flags"], separators=(",", ":")),
                "source_path": record["source"]["source_path"],
            })


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_scan_job(
    task: tuple[str, list[dict[str, str]], ScanGeometry, str, str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pickle-friendly worker boundary; preserves deterministic task ordering."""
    scan_id, rows, geometry, source_csv, source_csv_sha256, resolution_mode = task
    return fit_scan(
        scan_id,
        rows,
        geometry,
        source_csv=Path(source_csv),
        source_csv_sha256=source_csv_sha256,
        resolution_mode=resolution_mode,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature-max-k", type=float, default=30.0)
    parser.add_argument("--energy-min-mev", type=float, default=1.0)
    parser.add_argument("--min-points", type=int, default=11)
    parser.add_argument("--min-points-per-side", type=int, default=4)
    parser.add_argument("--resolution-mode", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--scan-id", action="append", default=[], help="fit only this scan id; repeatable")
    parser.add_argument("--max-scans", type=int, default=None, help="diagnostic limit after deterministic sorting")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    local_runs = (ROOT / "local_runs").resolve()
    if local_runs not in output_path.parents:
        raise ValueError(f"real-data output must be below ignored {local_runs}; got {output_path}")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_sha256 = _sha256_file(input_path)
    groups, input_fields = read_scan_groups(input_path)
    selection_audit: list[dict[str, Any]] = []
    eligible: list[tuple[str, list[dict[str, str]], ScanGeometry]] = []
    wanted = set(args.scan_id)
    for scan_id in sorted(groups):
        rows = groups[scan_id]
        audit, geometry = classify_scan(
            scan_id,
            rows,
            temperature_max_k=args.temperature_max_k,
            energy_min_mev=args.energy_min_mev,
            min_points=args.min_points,
            min_points_per_side=args.min_points_per_side,
        )
        if wanted and scan_id not in wanted:
            audit["eligible"] = False
            audit["exclusion_reasons"] = list(audit["exclusion_reasons"]) + ["not_requested_by_scan_id_filter"]
            geometry = None
        selection_audit.append(audit)
        if geometry is not None and audit["eligible"]:
            eligible.append((scan_id, rows, geometry))
    if args.max_scans is not None:
        eligible = eligible[: max(0, args.max_scans)]

    scan_results: list[dict[str, Any]] = []
    locus_points: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    tasks = [
        (scan_id, rows, geometry, str(input_path), source_sha256, args.resolution_mode)
        for scan_id, rows, geometry in eligible
    ]
    if workers == 1:
        result_iterator = map(fit_scan_job, tasks)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="srmn2sb2-fit")
        result_iterator = executor.map(fit_scan_job, tasks)
    try:
        for index, ((scan_id, _, _), (result, points)) in enumerate(
            zip(eligible, result_iterator), start=1
        ):
            print(f"[{index:03d}/{len(eligible):03d}] {scan_id}", flush=True)
            scan_results.append(result)
            locus_points.extend(points)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    write_jsonl(output_path / "selection_audit.jsonl", selection_audit)
    write_jsonl(output_path / "scan_fits.jsonl", scan_results)
    write_jsonl(output_path / "locus_points.jsonl", locus_points)
    write_locus_csv(output_path / "locus_points.csv", locus_points)
    write_scan_summary_csv(output_path / "scan_summary.csv", scan_results)

    selected_counts = Counter(result["model_selection"]["selected_model"] for result in scan_results)
    axis_counts = Counter(result["geometry"]["axis"] for result in scan_results)
    included_axis_counts = Counter(
        result["geometry"]["axis"] for result in scan_results if result["include_pair_in_position_fit"]
    )
    resolution_class_counts = Counter(result["resolution_classification"] for result in scan_results)
    configuration_status_counts = Counter(result["scan_configuration"]["status"] for result in scan_results)
    temperature_conflict_count = sum(
        bool(result["metadata"]["filename_temperature_diagnostic"]["conflict"])
        for result in scan_results
    )
    included_energies = {
        axis: sorted({
            float(result["profile"]["energy_mev"])
            for result in scan_results
            if result["geometry"]["axis"] == axis and result["include_pair_in_position_fit"]
        })
        for axis in ("H", "L")
    }
    manifest = {
        "schema_version": "srmn2sb2.symmetric_pair_extraction_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "git_head": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "input": str(input_path),
        "input_sha256": source_sha256,
        "input_fields": input_fields,
        "output": str(output_path),
        "selection": {
            "temperature_max_k": args.temperature_max_k,
            "energy_min_mev": args.energy_min_mev,
            "min_points": args.min_points,
            "min_points_per_side": args.min_points_per_side,
            "retain_repeats": True,
            "scan_id_filter": args.scan_id,
            "max_scans": args.max_scans,
        },
        "fit": {
            "models": [MODEL_BACKGROUND, MODEL_ONE, MODEL_PAIR],
            "background": "non-negative linear interpolation between fitted endpoint values",
            "pair": "independent left/right heights and centers; shared Gaussian sigma",
            "model_selection": "minimum BIC",
            "covariance": "consumer-facing covariance is inverse weighted-Jacobian information multiplied by max(1, reduced chi-square)",
            "also_recorded": "counting-only covariance and covariance multiplied directly by reduced chi-square",
            "branch_assignment": BRANCH_ASSIGNMENT,
        },
        "resolution_mode": args.resolution_mode,
        "resolution_policy": {
            "classification": "indeterminate_incomplete_configuration",
            "include_mask": "profile model-selection and fit-quality criteria only; no hard resolution exclusion",
            "scenarios": {
                "header_horizontal_collimation": "when ExptComment contains four collimations; remaining vertical/mosaic values are sensitivity assumptions",
                "legacy_assumption": list(LEGACY_HCOL_ARCMIN),
            },
            "resolved_field": None,
            "censored_field": False,
        },
        "counts": {
            "input_scan_groups": len(groups),
            "eligible_scan_groups": len(eligible),
            "fitted_by_axis": dict(axis_counts),
            "selected_models": dict(selected_counts),
            "locus_records": len(locus_points),
            "included_pairs_by_axis": dict(included_axis_counts),
            "included_locus_records": sum(bool(point["include"]) for point in locus_points),
            "resolution_classifications": dict(resolution_class_counts),
            "scan_configuration_statuses": dict(configuration_status_counts),
            "filename_temperature_conflicts": temperature_conflict_count,
        },
        "included_energies_mev": included_energies,
        "limitations": [
            "Peak profiles are not forward-convolved with the four-dimensional resolution function.",
            "No scan has a complete reviewed resolution configuration; resolved is null and scenario widths are sensitivity diagnostics only.",
            "ExptComment horizontal collimations are preserved when present, including conflicts between comment Ef and authoritative FixedE/data values.",
            "Gaussian peak-center covariance is a local Jacobian approximation and omits repeat-to-repeat systematics.",
            "No fitted pair is assigned to an acoustic or optical branch at this extraction stage.",
            "Resolution-scenario unresolved/off-center outcomes do not alter include; downstream analyses must report scenario sensitivity explicitly.",
            "Filename temperature labels are not trusted over recorded temperature channels; conflicts are flagged without changing measured-temperature selection.",
        ],
        "artifacts": {
            "selection_audit_jsonl": str(output_path / "selection_audit.jsonl"),
            "scan_fits_jsonl": str(output_path / "scan_fits.jsonl"),
            "scan_summary_csv": str(output_path / "scan_summary.csv"),
            "locus_points_jsonl": str(output_path / "locus_points.jsonl"),
            "locus_points_csv": str(output_path / "locus_points.csv"),
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_jsonable(manifest["counts"]), indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
