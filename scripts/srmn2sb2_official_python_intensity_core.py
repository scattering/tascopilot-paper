#!/usr/bin/env python3
"""Official-Python, resolution-convolved SrMn2Sb2 intensity objective.

This is a new scientific-evaluation boundary.  It deliberately does not import
the historical raw-Rust SrMn2Sb2 wrapper.  The pySpinW call is the official
post-PR110 high-level API with ``use_rust=False`` and all QMC points concatenated
into one request per spin domain.  The latter is required by the current Linux
Python backend, which creates its own process pool on every call.

The selected 20-scan pilot is inherited from the January analysis only as a
bounded direct-EPS test.  Its incomplete resolution configuration is recorded
as a sensitivity assumption, not promoted to reviewed instrument metadata.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import multiprocessing
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from lattice_calculator import Lattice
from rescalculator import TASResolution
from scipy.special import ndtri
from scipy.stats import qmc


A_ANGSTROM = 4.58233
C_ANGSTROM = 7.71463
Z_MN = 0.62123
SPIN_S = 1.5
INTRINSIC_FWHM_MEV = 0.35
# The 2N bosonic spectrum contains +/- partners.  Numerical Goldstone modes
# also carry a divergent 1/omega weight.  The selected data begin near 2 meV,
# so this threshold removes only the nonphysical duplicated/zero modes.
MIN_POSITIVE_MODE_ENERGY_MEV = 1.0e-3
SOBOL_SEED = 20260624
LEGACY_HCOL_ARCMIN = (25.0, 40.0, 40.0, 240.0)
DEFAULT_PYTHON_SCHEDULING_POLICY = "official_processpool"
PYTHON_SCHEDULING_POLICIES = (
    DEFAULT_PYTHON_SCHEDULING_POLICY,
    "outer_thread1",
)
PYTHON_SCHEDULING_ENVIRONMENT = "SRMN_PYSPINW_SCHEDULING"

# The outer-DREAM scheduling shim is deliberately pinned to the exact
# post-PR110 Python fallback reviewed on Stampede.  It changes only the
# executor selector used by this function; the pySpinW numerical kernel and
# its existing Q chunks remain untouched.  A source change requires a fresh
# audit rather than silently carrying the shim forward.
PINNED_PYSPINW_SPINWAVE_PARAMETERS = (
    "rotations", "magnitudes", "q_vectors", "couplings", "positions",
    "rlu_to_cart", "field", "rotating_frame", "save_sab",
    "save_wavefunctions",
)
PINNED_PYSPINW_SPINWAVE_SOURCE_SHA256 = (
    "8798f4e4fb01d60b2a558a2c0d04262c0e6202311f1d6da3ad6b1dcac46253be"
)
PINNED_PYSPINW_SPINWAVE_SOURCE_MARKERS = (
    "n_proc = max(int(np.floor(multiprocessing.cpu_count() / 4)), 1)",
    "Executor = get_Executor()",
    "with Executor() as executor:",
    "for q in _get_q_chunks(q_vectors, n_proc)",
)
_THREAD1_EXECUTOR = partial(ThreadPoolExecutor, max_workers=1)
_PINNED_OFFICIAL_EXECUTOR_SELECTOR: Any | None = None
_PATCHED_SPINWAVE_MODULE: Any | None = None
HEADER_HCOL_PATTERN = re.compile(
    r"(?<!\d)(\d{1,3})\s*[-/]\s*(\d{1,3})\s*[-/]\s*(\d{1,3})\s*[-/]\s*(\d{1,3})(?!\d)"
)

TARGETS = (
    ("Q002", 2.0), ("Q002", 4.0), ("Q002", 6.0), ("Q002", 8.0),
    ("Q002", 10.0), ("Q002", 12.0), ("Q112", 8.0),
    ("Q112", 12.0), ("Q112", 16.0), ("Q003", 5.0), ("Q003", 10.0),
)
L_TARGETS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

START_ZONE_SCALE = {
    "Q002_Ef14.7_HH": 0.006272,
    "Q002_Ef28_HH": 0.91157,
    "Q002_Ef28_L": 0.0025828,
    "Q003_Ef14.7_HH": 0.0028286,
    "Q112_Ef28_HH": 0.47278,
}
START_SCAN_SCALE = {
    "SMS_HH_Q002_e2ef14o7m_841224": 0.85974,
    "SMS_HH_Q002_e4ef14o7m_841225": 0.011584,
    "SMS_HH_Q002_e6ef14o7m_841226": 0.075607,
    "SMS_HH_Q002_e8ef14o7m_841227": 1.6245,
    "SMS_HH_Q002_e10ef14o7m_841228": 4.9567,
    "SMS_HH_Q002_e12_ef28841085": 1.2602,
    "SMS_HH_Q112_e8_ef28_841931": 1.1992,
    "SMS_HH_Q112_e12_ef28_841933": 2.6104,
    "SMS_HH_Q112_e16_ef28_841935": 0.46176,
    "SMS_HH_Q003_e5_841017": 0.0060455,
    "SMS_HH_Q003_e10_841019": 0.36484,
    "L_Q002_120K_ef28_e2_841953": 0.93828,
    "L_Q002_120K_ef28_e3_841955": 1.535,
    "L_Q002_120K_ef28_e4_841957": 0.013647,
    "L_Q002_120K_ef28_e5_841959": 0.0098532,
    "L_Q002_120K_ef28_e6_841961": 0.037395,
    "L_Q002_120K_ef28_e7_841962": 0.13705,
    "L_Q002_120K_ef28_e8_841963": 0.35065,
    "L_Q002_120K_ef28_e9_841964": 0.48746,
    "L_Q002_120K_ef28_e10_841965": 0.23231,
}
SELECTED_SCAN_FILENAMES = tuple(START_SCAN_SCALE)

SEED_PARAMETERS = {
    # Physical orientation of the exact acoustic label pair: large coefficient
    # is assigned to in-plane SJ1 and the smaller one to interlayer SJc.
    "all102": (10.1330288294171, 0.1519244980477527, 0.449970843036889),
    "center005": (6.9876109744411945, 0.1299802470982578, 0.7371469051365529),
}


def _get_thread1_executor():
    """Return the one-thread executor constructor used inside Bumps workers."""
    return _THREAD1_EXECUTOR


def validate_python_scheduling_policy(policy: str) -> str:
    if policy not in PYTHON_SCHEDULING_POLICIES:
        raise ValueError(
            f"unknown Python scheduling policy {policy!r}; use "
            + " or ".join(PYTHON_SCHEDULING_POLICIES)
        )
    return policy


def _is_official_executor_selector(selector: Any) -> bool:
    return (
        getattr(selector, "__name__", None) == "get_Executor"
        and getattr(selector, "__module__", None)
        == "pyspinw.windows_parallelisation"
    )


def _validate_pinned_python_spinwave(spinwave_module: Any) -> dict[str, Any]:
    """Fail closed unless the reviewed pySpinW Python scheduler is present."""
    global _PINNED_OFFICIAL_EXECUTOR_SELECTOR
    calculation = spinwave_module.spinwave_calculation
    parameters = tuple(inspect.signature(calculation).parameters)
    if parameters != PINNED_PYSPINW_SPINWAVE_PARAMETERS:
        raise RuntimeError(
            "official pySpinW Python fallback signature changed: "
            f"expected {PINNED_PYSPINW_SPINWAVE_PARAMETERS}, found {parameters}"
        )
    source = inspect.getsource(calculation)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != PINNED_PYSPINW_SPINWAVE_SOURCE_SHA256:
        raise RuntimeError(
            "official pySpinW Python fallback source changed: "
            f"expected {PINNED_PYSPINW_SPINWAVE_SOURCE_SHA256}, found {source_sha256}"
        )
    missing = [
        marker for marker in PINNED_PYSPINW_SPINWAVE_SOURCE_MARKERS
        if marker not in source
    ]
    if missing:
        raise RuntimeError(
            f"official pySpinW Python fallback scheduler markers changed: {missing}"
        )
    selector = spinwave_module.get_Executor
    if _is_official_executor_selector(selector):
        if _PINNED_OFFICIAL_EXECUTOR_SELECTOR is None:
            _PINNED_OFFICIAL_EXECUTOR_SELECTOR = selector
        elif selector is not _PINNED_OFFICIAL_EXECUTOR_SELECTOR:
            raise RuntimeError(
                "official pySpinW get_Executor identity changed within one fit process"
            )
    elif selector is _get_thread1_executor:
        if _PINNED_OFFICIAL_EXECUTOR_SELECTOR is None:
            raise RuntimeError(
                "thread1 selector is active without a captured pinned official selector"
            )
    else:
        raise RuntimeError(
            "pySpinW get_Executor was already replaced by an unreviewed selector: "
            f"{selector!r}"
        )
    cpu_count = multiprocessing.cpu_count()
    return {
        "spinwave_signature_parameters": list(parameters),
        "spinwave_source_sha256": source_sha256,
        "spinwave_source_markers": list(PINNED_PYSPINW_SPINWAVE_SOURCE_MARKERS),
        "host_cpu_count": cpu_count,
        "q_chunk_count": max(cpu_count // 4, 1),
    }


def python_scheduling_provenance(policy: str) -> dict[str, Any]:
    """Describe the exact pySpinW/Bumps process topology for one fit process."""
    policy = validate_python_scheduling_policy(policy)
    spinwave_module = importlib.import_module("pyspinw.calculations.spinwave")
    pinned = _validate_pinned_python_spinwave(spinwave_module)
    if policy == DEFAULT_PYTHON_SCHEDULING_POLICY:
        if not _is_official_executor_selector(spinwave_module.get_Executor):
            raise RuntimeError(
                "official_processpool provenance requested while thread1 executor remains active"
            )
        executor = "ProcessPoolExecutor"
        executor_max_workers: int | str = "CPython default host cpu_count"
        inner_child_processes_allowed = True
    else:
        if spinwave_module.get_Executor is not _get_thread1_executor:
            raise RuntimeError(
                "outer_thread1 provenance requested before its executor was activated"
            )
        executor = "ThreadPoolExecutor"
        executor_max_workers = 1
        inner_child_processes_allowed = False
    return {
        "policy": policy,
        "executor": executor,
        "executor_max_workers": executor_max_workers,
        "q_chunks_unchanged_from_official": True,
        "inner_child_processes_allowed": inner_child_processes_allowed,
        "patch_target": (
            None
            if policy == DEFAULT_PYTHON_SCHEDULING_POLICY
            else "pyspinw.calculations.spinwave.get_Executor"
        ),
        "patch_scope": "current fit process only; official source/install unchanged",
        **pinned,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dave_bt7_monitor_correction(ei: np.ndarray) -> np.ndarray:
    ei = np.asarray(ei, dtype=float)
    if np.any(np.isfinite(ei) & (ei < 5.0)):
        raise ValueError("DAVE BT7 monitor correction is undefined for Ei < 5 meV")
    correction = np.ones_like(ei)
    valid = np.isfinite(ei) & (ei >= 5.0) & (ei <= 34.0)
    e = ei[valid]
    correction[valid] = (
        6.1230 - 5.9630e-1 * e + 2.6244e-2 * e**2
        - 4.8430e-4 * e**3 + 2.8810e-6 * e**4
    )
    return correction


def fixed_ef(rows: pd.DataFrame) -> float:
    text = str(rows["fixed_e"].iloc[0])
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not numbers:
        raise ValueError(f"cannot parse fixed final energy from {text!r}")
    return float(numbers[-1])


def monitor_correct(
    rows: pd.DataFrame, ef: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    correction = dave_bt7_monitor_correction(ef + rows["E"].to_numpy(float))
    detector = rows["Detector"].to_numpy(float)
    monitor = rows["Monitor"].to_numpy(float)
    if not np.all(np.isfinite(detector)) or np.any(detector < 0.0):
        raise ValueError("detector counts must be finite and nonnegative")
    if not np.all(np.isfinite(monitor)) or np.any(monitor <= 0.0):
        raise ValueError("monitor counts must be finite and positive")
    y = detector / monitor * correction
    sigma = np.sqrt(np.maximum(detector, 1.0)) / monitor * correction
    finite = sigma[np.isfinite(sigma) & (sigma > 0)]
    floor = float(np.median(finite) * 0.5) if len(finite) else 1.0e-6
    floored = sigma < floor
    return y, np.maximum(sigma, floor), {
        "error_model": "Poisson sqrt(Detector)/Monitor after DAVE correction",
        "sigma_floor": floor,
        "sigma_floor_policy": "0.5 times the median finite positive Poisson sigma per scan",
        "sigma_floored_point_count": int(np.count_nonzero(floored)),
    }


def _read_header_comment(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#ExptComment"):
                return line[len("#ExptComment") :].strip()
            if not line.startswith("#"):
                break
    return ""


def horizontal_collimation(
    rows: pd.DataFrame, resolution_policy: str
) -> tuple[tuple[float, ...], str]:
    if resolution_policy == "legacy":
        return LEGACY_HCOL_ARCMIN, "legacy_local_analysis_assumption"
    if resolution_policy != "header_horizontal":
        raise ValueError("resolution policy must be legacy or header_horizontal")
    path = Path(str(rows["path"].iloc[0]))
    if not path.exists():
        raise FileNotFoundError(
            f"header_horizontal resolution requires the raw scan at {path}; "
            "stage/remap raw files explicitly or use the labeled legacy pilot"
        )
    match = HEADER_HCOL_PATTERN.search(_read_header_comment(path))
    if match:
        return tuple(float(value) for value in match.groups()), "raw_header_ExptComment_partial"
    raise ValueError(f"raw header for {path} has no explicit four-value horizontal collimation")


def _parse_numbers(value: Any, required: int, label: str) -> np.ndarray:
    numbers = [float(token) for token in str(value).replace(",", " ").split()]
    if len(numbers) < required:
        raise ValueError(f"{label} needs at least {required} numbers: {value!r}")
    return np.asarray(numbers[:required], dtype=float)


def _resolution_experiment(ef: float, hcol: Sequence[float]) -> dict[str, Any]:
    return {
        "efixed": float(ef), "infin": -1, "dir1": 1, "dir2": 1,
        "hcol": np.asarray(hcol, dtype=float),
        "vcol": np.asarray([120.0] * 4, dtype=float),
        "mono": {"tau": "pg(002)", "mosaic": 30.0},
        "ana": {"tau": "pg(002)", "mosaic": 30.0},
        "sample": {"mosaic": 30.0, "vmosaic": 30.0},
        "arms": [200.0, 200.0, 150.0, 150.0, 100.0],
        "horifoc": -1, "moncor": 1, "method": 0,
    }


def _sobol_normals(count: int) -> np.ndarray:
    if count <= 0 or count & (count - 1):
        raise ValueError("QMC sample count must be a positive power of two")
    sampler = qmc.Sobol(d=4, scramble=True, seed=SOBOL_SEED)
    uniform = sampler.random_base2(int(math.log2(count)))
    uniform = np.clip(uniform, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return ndtri(uniform)


def _covariance_from_precision(precision: np.ndarray) -> tuple[np.ndarray, float, bool]:
    matrix = 0.5 * (precision + precision.T)
    repaired = False
    try:
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive precision determinant")
        covariance = np.linalg.inv(matrix)
        np.linalg.cholesky(0.5 * (covariance + covariance.T))
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 1.0e-12, None)
        matrix = (vectors * values) @ vectors.T
        sign, logdet = np.linalg.slogdet(matrix)
        covariance = np.linalg.inv(matrix)
        repaired = True
    if sign <= 0:
        raise ValueError("resolution precision repair retained non-positive determinant")
    return 0.5 * (covariance + covariance.T), float(logdet), repaired


def resolution_qmc_samples(
    rows: pd.DataFrame, sample_count: int, resolution_policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows = rows.copy()
    ef = fixed_ef(rows)
    hcol, hcol_source = horizontal_collimation(rows, resolution_policy)
    lattice_values = _parse_numbers(rows["lattice"].iloc[0], 6, "lattice")
    orient = _parse_numbers(rows["orient"].iloc[0], 6, "orientation")
    npoint = len(rows)
    lattice = Lattice(
        a=np.full(npoint, lattice_values[0]), b=np.full(npoint, lattice_values[1]),
        c=np.full(npoint, lattice_values[2]), alpha=np.full(npoint, lattice_values[3]),
        beta=np.full(npoint, lattice_values[4]), gamma=np.full(npoint, lattice_values[5]),
        orient1=np.asarray([orient[:3]]), orient2=np.asarray([orient[3:6]]),
    )
    resolution = TASResolution(lattice, backend="numpy")
    h, k, l, e = (rows[name].to_numpy(float) for name in ("H", "K", "L", "E"))
    r0, precision = resolution.ResMatS(
        h, k, l, e, [_resolution_experiment(ef, hcol) for _ in range(npoint)]
    )
    r0, precision = np.asarray(r0).reshape(-1), np.asarray(precision)
    xvec, yvec, zvec = (np.asarray(value) for value in (lattice.x, lattice.y, lattice.z))
    normals = _sobol_normals(sample_count)
    all_hkl, all_e, all_factor, all_index, all_sigma = [], [], [], [], []
    repairs = 0
    for index in range(npoint):
        covariance, logdet, repaired = _covariance_from_precision(precision[:, :, index])
        repairs += int(repaired)
        chol = np.linalg.cholesky(covariance)
        offsets = normals @ chol.T
        q_indices = [0, 1, 3]
        sigma_qq = covariance[np.ix_(q_indices, q_indices)]
        sigma_eq = covariance[2, q_indices]
        conditional_weight = np.linalg.solve(sigma_qq, sigma_eq)
        conditional_variance = float(covariance[2, 2] - sigma_eq @ conditional_weight)
        if conditional_variance <= 0 or not math.isfinite(conditional_variance):
            raise ValueError("non-positive conditional energy variance")
        dq_x, dq_y, dq_z = offsets[:, 0], offsets[:, 1], offsets[:, 3]
        sample_h = h[index] + dq_x*xvec[0,index] + dq_y*yvec[0,index] + dq_z*zvec[0,index]
        sample_k = k[index] + dq_x*xvec[1,index] + dq_y*yvec[1,index] + dq_z*zvec[1,index]
        sample_l = l[index] + dq_x*xvec[2,index] + dq_y*yvec[2,index] + dq_z*zvec[2,index]
        sample_e = e[index] + offsets[:, q_indices] @ conditional_weight
        hkl = np.column_stack([sample_h, sample_k, sample_l])
        if not np.all(np.isfinite(hkl)) or not np.all(np.isfinite(sample_e)):
            raise ValueError("resolution sampling returned non-finite Q/E")
        volume = (2.0 * math.pi) ** 2 * math.exp(-0.5 * logdet)
        all_hkl.append(hkl)
        all_e.append(sample_e)
        all_factor.append(np.full(sample_count, float(r0[index]) * volume))
        all_index.append(np.full(sample_count, index, dtype=np.intp))
        all_sigma.append(np.full(sample_count, math.sqrt(conditional_variance)))
    return (
        np.vstack(all_hkl), np.concatenate(all_e), np.concatenate(all_factor),
        np.concatenate(all_index), np.concatenate(all_sigma),
        {
            "qmc_sample_count_per_point": sample_count,
            "sobol_seed": SOBOL_SEED,
            "fixed_ef_mev": ef,
            "horizontal_collimation_arcmin": list(hcol),
            "horizontal_collimation_source": hcol_source,
            "resolution_configuration_complete": False,
            "vertical_collimation_arcmin_assumed": [120.0] * 4,
            "mono_analyzer_mosaic_arcmin_assumed": 30.0,
            "sample_mosaic_arcmin_assumed": 30.0,
            "covariance_repairs": repairs,
            "energy_axis": "analytic conditional Gaussian (Rao-Blackwellised)",
        },
    )


def _select_hh_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    low = frame[(frame["Temp"] < 30.0) & (frame["Monitor"] > 0)].copy()
    hhl = low[
        (low["scan_variable"].astype(str).str.lower() == "qy")
        & (np.abs(low["H"] - low["K"]) < 0.05)
        & low["E"].between(1.0, 16.0)
    ].copy()
    result: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for zone, energy in TARGETS:
        candidate = hhl[np.abs(hhl["E"] - energy) < 0.05].copy()
        med_h = candidate.groupby("filename")["H"].transform("median")
        med_l = candidate.groupby("filename")["L"].transform("median")
        if zone == "Q002":
            candidate = candidate[(np.abs(med_h) < 0.15) & (np.abs(med_l - 2.0) < 0.12)]
        elif zone == "Q112":
            candidate = candidate[med_h.between(0.85, 1.15) & (np.abs(med_l - 2.0) < 0.12)]
        else:
            candidate = candidate[
                candidate["filename"].astype(str).str.contains("SMS_HH_Q003", regex=False)
                & (np.abs(med_h) < 0.2) & (np.abs(med_l - 3.0) < 0.12)
            ]
        if candidate.empty:
            raise ValueError(f"no selected scan for {zone} E={energy}")
        ranked = candidate.groupby("filename").agg(
            n=("filename", "size"), span=("H", lambda x: float(np.ptp(x)))
        ).sort_values(["n", "span"], ascending=False)
        filename = str(ranked.index[0])
        if filename not in seen:
            seen.add(filename)
            result.append((zone, candidate[candidate["filename"] == filename].sort_values("H")))
    return result


def _select_l_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    low = frame[(frame["Temp"] < 30.0) & (frame["Monitor"] > 0)].copy()
    scans = low[
        (low["scan_variable"].astype(str).str.lower() == "qz")
        & (np.abs(low["H"]) < 0.08) & (np.abs(low["K"]) < 0.08) & (low["E"] >= 1.0)
    ]
    result = []
    for energy in L_TARGETS:
        candidate = scans[np.abs(scans["E"] - energy) < 0.05]
        if candidate.empty:
            raise ValueError(f"no selected L scan for E={energy}")
        ranked = candidate.groupby("filename").agg(
            n=("filename", "size"), span=("L", lambda x: float(np.ptp(x)))
        ).sort_values(["span", "n"], ascending=False)
        filename = str(ranked.index[0])
        result.append(("Q002", candidate[candidate["filename"] == filename].sort_values("L")))
    return result


def _apply_reviewed_mask(rows: pd.DataFrame) -> pd.DataFrame:
    if str(rows["filename"].iloc[0]) != "SMS_HH_Q002_e2ef14o7m_841224":
        return rows
    bad = (np.abs(rows["H"] - 0.0625) < 0.0065) & (np.abs(rows["E"] - 2.0) < 0.05)
    return rows.loc[~bad].copy()


@dataclass(frozen=True)
class ScanGroup:
    filename: str
    zone: str
    zone_key: str
    scan_coordinate: np.ndarray
    scan_coordinate_label: str
    energy_transfer_mev: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    detector: np.ndarray
    monitor: np.ndarray
    sample_slice: slice
    sample_e: np.ndarray
    sample_factor: np.ndarray
    sample_to_point: np.ndarray
    sample_sigma_e: np.ndarray
    background: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScanPrediction:
    """One scan's signal and full prediction from the fitted objective."""

    filename: str
    signal: np.ndarray
    prediction: np.ndarray
    fixed_background: float


@dataclass(frozen=True)
class PreparedData:
    groups: tuple[ScanGroup, ...]
    all_sample_hkl: np.ndarray
    input_path: Path
    input_sha256: str
    backgrounds_path: Path
    backgrounds_sha256: str
    qmc_sample_count: int
    resolution_policy: str

    @property
    def npoints(self) -> int:
        return sum(len(group.y) for group in self.groups)


def _load_backgrounds(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    backgrounds: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("best"), dict):
            continue
        filename = str(row["filename"])
        if filename in backgrounds:
            raise ValueError(f"duplicate fitted background for {filename} in {path}")
        value = float(row["best"]["background"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"background for {filename} must be finite and nonnegative")
        backgrounds[filename] = value
    missing = set(SELECTED_SCAN_FILENAMES) - set(backgrounds)
    if missing:
        raise ValueError(f"missing fitted backgrounds for locked scans: {sorted(missing)}")
    return backgrounds


def prepare_data(
    input_path: Path, backgrounds_path: Path, qmc_sample_count: int,
    resolution_policy: str = "legacy",
) -> PreparedData:
    input_path, backgrounds_path = input_path.resolve(), backgrounds_path.resolve()
    frame = pd.read_csv(input_path)
    backgrounds = _load_backgrounds(backgrounds_path)
    selected = _select_hh_groups(frame) + _select_l_groups(frame)
    selected_filenames = tuple(str(rows["filename"].iloc[0]) for _, rows in selected)
    if selected_filenames != SELECTED_SCAN_FILENAMES:
        raise AssertionError(
            "selected scan manifest changed; expected "
            f"{SELECTED_SCAN_FILENAMES}, got {selected_filenames}"
        )
    all_hkl: list[np.ndarray] = []
    groups: list[ScanGroup] = []
    offset = 0
    for zone, raw_rows in selected:
        rows = _apply_reviewed_mask(raw_rows.copy())
        ef = fixed_ef(rows)
        y, sigma, error_metadata = monitor_correct(rows, ef)
        sample_hkl, sample_e, factor, to_point, sigma_e, metadata = resolution_qmc_samples(
            rows, qmc_sample_count, resolution_policy
        )
        filename = str(rows["filename"].iloc[0])
        if filename not in backgrounds:
            raise KeyError(f"no fixed peak-fit background for {filename}")
        scan_variable = str(rows["scan_variable"].iloc[0]).lower()
        geometry = "L" if scan_variable in {"qz", "l"} else "HH"
        coordinate_column = "L" if geometry == "L" else "H"
        coordinate_label = "L (r.l.u.)" if geometry == "L" else "H = K (r.l.u.)"
        zone_key = f"{zone}_Ef{ef:g}_{geometry}"
        groups.append(ScanGroup(
            filename=filename, zone=zone, zone_key=zone_key,
            scan_coordinate=rows[coordinate_column].to_numpy(float),
            scan_coordinate_label=coordinate_label,
            energy_transfer_mev=rows["E"].to_numpy(float), y=y, sigma=sigma,
            detector=rows["Detector"].to_numpy(float), monitor=rows["Monitor"].to_numpy(float),
            sample_slice=slice(offset, offset + len(sample_hkl)), sample_e=sample_e,
            sample_factor=factor, sample_to_point=to_point, sample_sigma_e=sigma_e,
            background=backgrounds[filename], metadata={**metadata, **error_metadata},
        ))
        offset += len(sample_hkl)
        all_hkl.append(sample_hkl)
    if len(groups) != 20:
        raise AssertionError(f"selected {len(groups)} groups; expected 20")
    return PreparedData(
        groups=tuple(groups), all_sample_hkl=np.vstack(all_hkl), input_path=input_path,
        input_sha256=sha256_file(input_path), backgrounds_path=backgrounds_path,
        backgrounds_sha256=sha256_file(backgrounds_path), qmc_sample_count=qmc_sample_count,
        resolution_policy=resolution_policy,
    )


def build_official_srmn2sb2_model(
    sj1_mev: float, j2_over_j1: float, sjc_mev: float, d_easy_plane_mev: float,
    spin_angle_deg: float = 90.0,
):
    """Build the post-PR110 representative-bond high-level Hamiltonian."""
    from pyspinw import Hamiltonian
    from pyspinw.exchange import HeisenbergExchange
    from pyspinw.interface import axis_anisotropies, generate_structure
    from pyspinw.symmetry.unitcell import UnitCell

    cell = UnitCell(A_ANGSTROM, A_ANGSTROM, C_ANGSTROM, gamma=120.0)
    angle = math.radians(spin_angle_deg)
    moment = [SPIN_S * math.cos(angle), SPIN_S * math.sin(angle), 0.0]
    structure = generate_structure(
        cell,
        positions=[[1/3, 2/3, Z_MN], [2/3, 1/3, 1-Z_MN]], positions_unit="lu",
        spins=[moment, [-value for value in moment]], spins_unit="xyz",
        magnitudes=[SPIN_S, SPIN_S], names=["Mn1", "Mn2"],
    )
    mn1, mn2 = structure.sites
    exchanges = []
    for offset in ((0, 0, 0), (0, 1, 0), (-1, 0, 0)):
        exchanges.append(HeisenbergExchange(mn1, mn2, j=2*sj1_mev/SPIN_S, cell_offset=offset, name="J1"))
    for site in (mn1, mn2):
        for offset in ((0, 1, 0), (1, 0, 0), (1, 1, 0)):
            exchanges.append(HeisenbergExchange(
                site, site, j=2*sj1_mev*j2_over_j1/SPIN_S, cell_offset=offset, name="J2"
            ))
    for offset in ((0, 0, 1), (-1, 0, 1), (0, 1, 1)):
        exchanges.append(HeisenbergExchange(mn1, mn2, j=2*sjc_mev/SPIN_S, cell_offset=offset, name="Jc"))
    # High-level adapter doubles onsite matrices before the kernel call.
    anisotropies = axis_anisotropies(structure, a=d_easy_plane_mev/2.0, axis=[0, 0, 1])
    return Hamiltonian(structure, exchanges, anisotropies)


def python_sperp(
    model: Any,
    q_rlu: np.ndarray,
    scheduling_policy: str = DEFAULT_PYTHON_SCHEDULING_POLICY,
) -> tuple[np.ndarray, np.ndarray]:
    """The only production intensity call: official high-level Python S_perp.

    Official commit 82c14f's public method is Path-only.  Its internal array
    adapter is used here so hundreds of thousands of QMC points remain exactly
    one batch.  Preflight pins this signature and source commit before use.
    """
    global _PATCHED_SPINWAVE_MODULE

    scheduling_policy = validate_python_scheduling_policy(scheduling_policy)
    if scheduling_policy == "outer_thread1":
        # This assignment is intentionally here, inside the importable
        # numerical call path.  Bumps cloudpickle workers therefore apply (or
        # inherit and revalidate) it before every official Python-backend call.
        # Do not patch multiprocessing.cpu_count: Bumps uses that same module
        # to choose its outer worker count.
        spinwave_module = importlib.import_module("pyspinw.calculations.spinwave")
        _validate_pinned_python_spinwave(spinwave_module)
        spinwave_module.get_Executor = _get_thread1_executor
        _PATCHED_SPINWAVE_MODULE = spinwave_module
    elif _PATCHED_SPINWAVE_MODULE is not None:
        # A dedicated fit process normally uses exactly one scheduling policy,
        # but restoring makes direct diagnostics and tests truthful if policies
        # are intentionally evaluated in sequence.
        spinwave_module = importlib.import_module("pyspinw.calculations.spinwave")
        if spinwave_module is not _PATCHED_SPINWAVE_MODULE:
            raise RuntimeError("pySpinW scheduler module identity changed before restore")
        _validate_pinned_python_spinwave(spinwave_module)
        if _PINNED_OFFICIAL_EXECUTOR_SELECTOR is None:
            raise RuntimeError("pinned official pySpinW executor selector was not captured")
        spinwave_module.get_Executor = _PINNED_OFFICIAL_EXECUTOR_SELECTOR
        _PATCHED_SPINWAVE_MODULE = None

    energies, intensities = model._energies_and_intensities(
        q_vectors=np.asarray(q_rlu, dtype=float), use_rust=False, use_rotating=False,
        intensity_unit="spin", components="Sperp",
    )
    energies = np.asarray(energies)
    intensities = np.asarray(intensities, dtype=float)
    if energies.ndim != 2 or intensities.shape != energies.shape:
        raise ValueError(f"unexpected mode shapes: E={energies.shape}, I={intensities.shape}")
    if np.max(np.abs(energies.imag)) > 1.0e-8:
        raise ValueError("Python backend returned appreciably complex mode energies")
    energies = energies.real
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(intensities)):
        raise ValueError("Python backend returned non-finite modes")
    return energies, intensities


def physical_positive_modes(
    energies: np.ndarray, intensities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove negative-energy partners and numerical Goldstone divergences.

    The returned arrays retain their original mode order and shape so every
    intensity remains paired with its own energy.  Excluded modes have zero
    energy and zero weight and therefore contribute exactly zero downstream.
    """

    energy = np.asarray(energies, dtype=float)
    intensity = np.asarray(intensities, dtype=float)
    if energy.shape != intensity.shape:
        raise ValueError(
            f"energy/intensity mode shapes differ: {energy.shape} != {intensity.shape}"
        )
    positive = energy > MIN_POSITIVE_MODE_ENERGY_MEV
    return np.where(positive, energy, 0.0), np.where(positive, intensity, 0.0)


def _mn2_form_factor_squared(q_rlu: np.ndarray) -> np.ndarray:
    xyz = UnitCellProxy().rlu_to_cartesian(q_rlu)
    q_magnitude = np.linalg.norm(xyz, axis=1)
    s2 = (q_magnitude / (4.0 * math.pi)) ** 2
    j0 = 0.422*np.exp(-17.684*s2) + 0.5948*np.exp(-6.005*s2) + 0.0043*np.exp(0.609*s2) - 0.0219
    return j0*j0


class UnitCellProxy:
    """Small deterministic RLU-to-|Q| helper for the hexagonal fit cell."""
    def __init__(self) -> None:
        from pyspinw.symmetry.unitcell import UnitCell
        cell = UnitCell(A_ANGSTROM, A_ANGSTROM, C_ANGSTROM, gamma=120.0)
        self.matrix = np.linalg.inv(cell._xyz).T * 2.0 * math.pi

    def rlu_to_cartesian(self, q_rlu: np.ndarray) -> np.ndarray:
        return np.asarray(q_rlu, dtype=float) @ self.matrix


def domain_angles(policy: str) -> tuple[float, ...]:
    if policy == "single_y":
        return (90.0,)
    if policy == "equal3":
        return (90.0, 210.0, 330.0)
    raise ValueError(f"unknown domain policy {policy!r}; use single_y or equal3")


class OfficialPythonIntensityObjective:
    """Raw-profile residual objective with ordinary external scale parameters."""
    def __init__(
        self,
        prepared: PreparedData,
        normalization_policy: str,
        domain_policy: str,
        python_scheduling_policy: str = DEFAULT_PYTHON_SCHEDULING_POLICY,
        python_scheduling_source: str = "direct_default",
    ):
        if normalization_policy not in {"per_scan", "shared_zone_config"}:
            raise ValueError("normalization policy must be per_scan or shared_zone_config")
        self.prepared = prepared
        self.normalization_policy = normalization_policy
        self.domain_policy = domain_policy
        self.python_scheduling_policy = validate_python_scheduling_policy(
            python_scheduling_policy
        )
        self.python_scheduling_source = python_scheduling_source
        self._domain_angles = domain_angles(domain_policy)
        self.scale_keys = tuple(sorted({
            group.filename if normalization_policy == "per_scan" else group.zone_key
            for group in prepared.groups
        }))
        self._ff2 = _mn2_form_factor_squared(prepared.all_sample_hkl)

    def scale_start(self, key: str) -> float:
        table = START_SCAN_SCALE if self.normalization_policy == "per_scan" else START_ZONE_SCALE
        if key not in table:
            raise KeyError(f"missing declared scale start for {key}")
        return float(table[key])

    def predict_groups(
        self, sj1_mev: float, j2_over_j1: float, sjc_mev: float,
        d_easy_plane_mev: float, scales: dict[str, float],
    ) -> tuple[ScanPrediction, ...]:
        """Evaluate the exact fitted model and return each scan prediction.

        This is the single prediction path used by both the likelihood and the
        Bumps model/data plot, including the same QMC resolution convolution,
        domain policy, Mn2+ form factor, normalization, and fixed background.
        """
        missing = set(self.scale_keys) - set(scales)
        if missing:
            raise KeyError(f"missing scale values: {sorted(missing)}")
        domain_local = []
        for angle in self._domain_angles:
            model = build_official_srmn2sb2_model(
                sj1_mev, j2_over_j1, sjc_mev, d_easy_plane_mev, angle
            )
            energy, intensity = python_sperp(
                model,
                self.prepared.all_sample_hkl,
                scheduling_policy=self.python_scheduling_policy,
            )
            sigma_intrinsic = INTRINSIC_FWHM_MEV / 2.354820045
            # Split before applying scan-specific conditional E means and widths.
            domain_local.append((energy, np.maximum(intensity, 0.0)))
        predictions = []
        for group in self.prepared.groups:
            energy = np.mean([value[0][group.sample_slice] for value in domain_local], axis=0)
            # Exchange/easy-plane energies are domain invariant; average intensity
            # only. Keep this assertion as a scientific domain-policy check.
            for value in domain_local[1:]:
                np.testing.assert_allclose(value[0][group.sample_slice], energy, atol=2e-8, rtol=2e-10)
            intensity = np.mean([value[1][group.sample_slice] for value in domain_local], axis=0)
            # Keep only physical positive-energy modes.  The Bogoliubov solver
            # returns +/- partners, and numerical Goldstone modes below this
            # floor have divergent S_perp.  Do not abs/sort intensities: mode
            # weights stay paired with their own energies.
            energy, intensity = physical_positive_modes(energy, intensity)
            sigma = np.sqrt(group.sample_sigma_e**2 + sigma_intrinsic**2)[:, None]
            local = (
                intensity * self._ff2[group.sample_slice, None]
                * np.exp(-0.5*((group.sample_e[:, None] - energy)/sigma)**2)
                / (sigma*math.sqrt(2.0*math.pi))
            ).sum(axis=1) * group.sample_factor
            count = np.bincount(group.sample_to_point, minlength=len(group.y))
            total = np.bincount(group.sample_to_point, weights=local, minlength=len(group.y))
            if np.any(count != self.prepared.qmc_sample_count):
                raise ValueError(f"incomplete QMC samples for {group.filename}")
            signal = total / count
            key = group.filename if self.normalization_policy == "per_scan" else group.zone_key
            prediction = scales[key] * signal + group.background
            if not np.all(np.isfinite(prediction)):
                raise ValueError(f"non-finite prediction for {group.filename}")
            predictions.append(ScanPrediction(
                filename=group.filename,
                signal=signal,
                prediction=prediction,
                fixed_background=group.background,
            ))
        return tuple(predictions)

    def residuals(
        self, sj1_mev: float, j2_over_j1: float, sjc_mev: float,
        d_easy_plane_mev: float, scales: dict[str, float],
    ) -> np.ndarray:
        predictions = self.predict_groups(
            sj1_mev, j2_over_j1, sjc_mev, d_easy_plane_mev, scales
        )
        residual_chunks = []
        for group, result in zip(self.prepared.groups, predictions):
            residual = (result.prediction - group.y) / group.sigma
            if not np.all(np.isfinite(residual)):
                raise ValueError(f"non-finite residual for {group.filename}")
            residual_chunks.append(residual)
        return np.concatenate(residual_chunks)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": {
                "api": "Hamiltonian._energies_and_intensities official array adapter",
                "public_wrapper_equivalence": (
                    "same official high-level implementation; private array adapter is pinned because "
                    "the 82c14f public wrapper requires a Path and would expand the QMC batch"
                ),
                "use_rust": False,
                "use_rotating": False,
                "intensity_unit": "spin",
                "components": "Sperp",
                "batching": "all QMC Q points concatenated into one call per spin domain",
                "python_scheduling": python_scheduling_provenance(
                    self.python_scheduling_policy
                ),
                "python_scheduling_selection_source": self.python_scheduling_source,
            },
            "normalization_policy": self.normalization_policy,
            "domain_policy": self.domain_policy,
            "domain_angles_deg": list(self._domain_angles),
            "domain_fractions_fitted": False,
            "input": str(self.prepared.input_path),
            "input_sha256": self.prepared.input_sha256,
            "backgrounds": str(self.prepared.backgrounds_path),
            "backgrounds_sha256": self.prepared.backgrounds_sha256,
            "selected_scan_count": len(self.prepared.groups),
            "selected_scan_filenames": [group.filename for group in self.prepared.groups],
            "data_point_count": self.prepared.npoints,
            "qmc_sample_count_per_point": self.prepared.qmc_sample_count,
            "positive_mode_floor_mev": MIN_POSITIVE_MODE_ENERGY_MEV,
            "total_qmc_q_count": len(self.prepared.all_sample_hkl),
            "resolution_policy": self.prepared.resolution_policy,
            "resolution_scenarios": [group.metadata for group in self.prepared.groups],
            "resolution_configuration_complete": False,
            "scientific_scope": "selected 20-scan direct-EPS pilot; not the all-scan final likelihood",
        }
