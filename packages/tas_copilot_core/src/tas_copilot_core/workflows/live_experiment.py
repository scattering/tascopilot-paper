"""Static "current experiment" summary page, rebuilt from the public NCNR archive.

``live_experiment_page`` mirrors the running experiment's NICE files into a
cache, classifies the scans with the dashboard's family rules, assembles
meshes and order-parameter series automatically, and writes one self-contained
HTML page plus ``status.json`` per instrument. It is meant to run from a timer
on a small shared host: every section is best-effort, and a failed build leaves
the previous page in place.
"""
from __future__ import annotations

import base64
import glob
import html as html_mod
import io
import json
import math
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tas_copilot_core.counting import rate_with_uncertainty
from tas_copilot_core.workflows.experiment_dashboard import _finalize_nice_header, _overnight_family, _parse_header_line, _parse_scan_file

MONITOR_REF = 4800.0
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
CONSOLE_EVENT_RE = re.compile(r"CRITICAL|ERROR|IMPORTANT|broken state|recovered from|Hardware communication|monitor rate of|SERIOUS", re.I)
CONSOLE_STAMP_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})>(.*)$")


# --------------------------------------------------------------------------
# scan reading (NICE 0.2 columns files) + optional NeXus H,K
# --------------------------------------------------------------------------
def read_nice_scan(path: Path) -> dict[str, Any] | None:
    header: dict[str, str] = {}
    columns: list[str] = []
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("##"):
                columns = stripped[2:].split()
                continue
            if stripped.startswith("#"):
                key, value = _parse_header_line(stripped)
                if key:
                    header[key] = value
                continue
            if columns:
                parts = stripped.split()
                if len(parts) == len(columns):
                    rows.append(parts)
    if not columns or not rows:
        return None
    header = _finalize_nice_header(header)
    data: dict[str, np.ndarray] = {}
    array = np.array(rows, dtype=object)
    for index, name in enumerate(columns):
        try:
            data[name] = array[:, index].astype(float)
        except ValueError:
            continue
    # NICE writes its own file number into the header; the file-name suffix is
    # a fallback (and misreads "order2001165.dat.bt7" as 2001165).
    number_match = re.search(r"(\d+)\.dat\.bt7$", Path(path).name)
    header_number = next((header[k] for k in ("trajectoryData.fileNum", "trajectoryData.instFileNum") if str(header.get(k, "")).strip().isdigit()), None)
    controls = [c.strip() for c in header.get("trajectory.controls", "").strip("[]").split(",") if c.strip()]
    trajectory_name = header.get("trajectory.name") or header.get("trajectoryData.trajName") or ""

    def _column(name: str) -> np.ndarray | None:
        for key, values in data.items():
            if key == name or key.split("(")[0] == name:
                return values
        return None

    # NICE lists its own bookkeeping nodes (trajectoryData.Kvar, ...) among the
    # controls; they mirror the scanned device and would make every Q scan
    # look coupled.
    real_controls = [c for c in controls if not c.startswith("trajectoryData.")]
    moving = [c for c in real_controls if _column(c) is not None and np.ptp(_column(c)) > 0.05]
    if not moving:
        moving = real_controls[:1] or controls[:1]
    variable = moving[0] if moving else None
    if len(moving) > 1:
        # a coupled scan (findPeak on A4 drives A3 with it): plot against the
        # axis the trajectory is named after, e.g. sampleTwoTheta -> A4.
        preferred = "A4" if "TwoTheta" in trajectory_name else ("A3" if "Theta" in trajectory_name else moving[0])
        variable = preferred if preferred in moving else moving[0]
    # Explicit NICE scan intent outranks control-list order for coupled scans.
    # Resolve only to an actually moving column; preserve old fallback otherwise.
    declared = str(header.get("trajectoryData.xAxis") or header.get("trajectory.xAxis") or "").strip().strip("\"'")
    aliases = {"a3": "A3", "sampletheta.softposition": "A3", "a4": "A4", "sampletwotheta.softposition": "A4"}
    canonical = aliases.get(declared.casefold(), declared)
    declared_match = next((axis for axis in moving if aliases.get(axis.casefold(), axis).casefold() == canonical.casefold()), None)
    if declared_match is not None:
        variable = declared_match
    stamp = header.get("date") or header.get("Date") or ""
    try:
        start = datetime.fromisoformat(stamp.strip())
    except ValueError:
        start = None

    def col(*names: str) -> np.ndarray | None:
        for name in names:
            for key in data:
                if key == name or key.split("(")[0] == name:
                    return data[key]
        return None

    return {
        "path": str(path),
        "name": Path(path).name,
        "number": int(header_number) if header_number is not None else (int(number_match.group(1)) if number_match else -1),
        "header": header,
        "variable": variable,
        "axis": _column(variable) if variable else None,
        "coupled": len(moving) > 1,
        "trajectory": trajectory_name,
        "start": start,
        "npts": len(rows),
        "A3": col("A3"),
        "A4": col("A4"),
        "det": col("Detector"),
        "mon": col("monitor"),
        "time": col("time"),
        "T": col("temp"),
        "Tset": col("temp.set1"),
        "ts": col("timeStamp"),
        "ei": col("ei"),
        "ef": col("ef"),
        "E": col("et"),
        "H": col("Q.H", "H"),
        "K": col("Q.K", "K"),
        "L": col("Q.L", "L"),
        "a3zero": float(np.nanmedian(col("sampleState.a3Zero"))) if col("sampleState.a3Zero") is not None else float("nan"),
        "point_start": col("start"),
        "point_stop": col("stop"),
        "planned_points": int(float(header["trajectory.scanLength"])) if str(header.get("trajectory.scanLength", "")).strip().replace(".", "", 1).isdigit() else None,
        "nice_estimated_seconds": float(header["trajectory.estimatedTime"]) if str(header.get("trajectory.estimatedTime", "")).strip() else None,
        "psd": np.column_stack([data[f"PSD{i + 1:02d}"] for i in range(48)]) if all(f"PSD{i + 1:02d}" in data for i in range(48)) else None,
        "anatrans": float(np.nanmedian(col("anaTrans"))) if col("anaTrans") is not None and np.isfinite(col("anaTrans")).any() else None,
        "analyzer_mode": str(header.get("analyzerMode", "")).strip(),
    }


def scan_intensity(rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    det = rec["det"]
    mon = rec["mon"] if rec["mon"] is not None else np.full_like(det, MONITOR_REF)
    # counts per MONITOR_REF monitor counts; the uncertainty is propagated by
    # tasreduction's Measurement (one standard deviation, Gaussian approximation)
    return rate_with_uncertainty(det, mon, target=MONITOR_REF)


def scan_axis_values(rec: dict[str, Any]) -> np.ndarray | None:
    """The scanned device's own column (Q.K for a K scan), never a stand-in:
    plotting a K scan against A3 under a 'Q.K' label is worse than no axis."""
    if rec.get("axis") is not None:
        return rec["axis"]
    variable = rec["variable"]
    if variable and rec.get(variable) is not None:
        return rec[variable]
    for name in ("A3", "A4"):
        values = rec.get(name)
        if values is not None and np.ptp(values) > 0.05:
            return values
    return None


def read_nexus_hk(path: Path) -> dict[str, Any] | None:
    try:
        import h5py  # optional dependency
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as handle:
            q = handle["entry/DAS_logs/Q"]
            sample = handle["entry/sample"]
            return {
                "H": q["H"][:], "K": q["K"][:], "L": q["L"][:],
                "refplane1": sample["refplane1"][0].tolist(), "refplane2": sample["refplane2"][0].tolist(),
                "a": float(sample["unit_cell_a"][0]), "b": float(sample["unit_cell_b"][0]), "c": float(sample["unit_cell_c"][0]),
            }
    except Exception:
        return None


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------
def gaussian(x: np.ndarray, amp: float, center: float, sigma: float, background: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background


def fit_peak(x: np.ndarray, y: np.ndarray, err: np.ndarray) -> dict[str, float] | None:
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, err = x[ok], y[ok], np.maximum(err[ok], 1.0)
    if len(x) < 6 or np.ptp(x) <= 0:
        return None
    b0 = float(np.median(np.sort(y)[: max(3, len(y) // 4)]))
    i = int(np.argmax(y))
    a0 = max(float(y[i] - b0), 1.0)
    step = float(np.median(np.abs(np.diff(x)))) or 1e-3
    above = x[y > b0 + a0 / 2]
    s0 = max((above.max() - above.min()) / 2.355, step) if len(above) > 1 else 2 * step
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[a0, x[i], s0, b0], sigma=err, absolute_sigma=True,
                               bounds=([0, x.min(), 0.2 * step, 0], [np.inf, x.max(), np.ptp(x), np.inf]), maxfev=20000)
    except Exception:
        return None
    perr = np.sqrt(np.abs(np.diag(pcov)))
    amp, center, sigma, background = popt
    area = amp * sigma * math.sqrt(2 * math.pi)
    area_gradient = np.array([sigma, 0.0, amp, 0.0]) * math.sqrt(2 * math.pi)
    area_variance = float(area_gradient @ pcov @ area_gradient)
    chi2 = float(np.sum(((y - gaussian(x, *popt)) / err) ** 2) / max(len(x) - 4, 1))
    return {"amp": float(amp), "damp": float(perr[0]), "center": float(center), "dcenter": float(perr[1]), "fwhm": float(2.3548 * sigma),
            "dfwhm": float(2.3548 * perr[2]), "background": float(background), "area": float(area), "darea": float(math.sqrt(max(0.0, area_variance))), "chi2": chi2}


MAX_COMPONENTS = 4


def find_components(x: np.ndarray, y: np.ndarray, err: np.ndarray, *, max_components: int = MAX_COMPONENTS) -> list[dict[str, float]]:
    """Peaks of one scan: local maxima of the lightly smoothed profile whose
    prominence is well above the counting noise and a fair fraction of the
    largest one, at least three points apart. Returns seeds (centre, sigma,
    amplitude) sorted by position; the strongest peak alone when nothing else
    stands out."""
    from scipy.signal import find_peaks, peak_widths

    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); err = np.asarray(err, dtype=float)
    order = np.argsort(x); x, y, err = x[order], y[order], err[order]
    step = float(np.median(np.abs(np.diff(x)))) or 1e-3
    smooth = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid") if len(y) >= 3 else y
    noise = float(np.nanmedian(err)) if np.isfinite(err).any() else 1.0
    peaks, props = find_peaks(smooth, prominence=4 * noise, distance=3)
    if len(peaks) == 0:
        i = int(np.argmax(smooth)); b0 = float(np.median(np.sort(y)[: max(3, len(y) // 4)]))
        return [{"center": float(x[i]), "sigma": 2 * step, "amp": max(float(y[i] - b0), 1.0)}]
    prominences = props["prominences"]
    keep = prominences >= 0.2 * prominences.max()
    peaks, prominences = peaks[keep], prominences[keep]
    widths = peak_widths(smooth, peaks, rel_height=0.5)[0] * step   # FWHM in x units
    strongest = np.argsort(prominences)[::-1][:max_components]
    seeds = [{"center": float(x[peaks[i]]), "sigma": max(float(widths[i]) / 2.3548, step), "amp": float(prominences[i])} for i in strongest]
    return sorted(seeds, key=lambda c: c["center"])


def _sum_of_gaussians(x: np.ndarray, background: float, *params: float) -> np.ndarray:
    total = np.full_like(np.asarray(x, dtype=float), float(background))
    for amp, center, sigma in zip(params[0::3], params[1::3], params[2::3]):
        total = total + gaussian(x, amp, center, sigma, 0.0)
    return total


def fit_components_free(x: np.ndarray, y: np.ndarray, err: np.ndarray, seeds: list[dict[str, float]], *, center_window: float | None = None) -> dict[str, Any] | None:
    """Sum of Gaussians plus a flat background, every parameter free within
    bounds around the seeds (centres within a window, sigma within a factor of
    two). Returns the components and the background, or None when the fit
    does not converge."""
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); err = np.where(np.asarray(err, dtype=float) > 0, err, 1.0)
    step = float(np.median(np.abs(np.diff(np.sort(x))))) or 1e-3
    b0 = float(np.median(np.sort(y)[: max(3, len(y) // 4)]))
    p0 = [max(b0, 0.0)]; lo = [0.0]; hi = [np.inf]
    for seed in seeds:
        window = center_window if center_window is not None else max(2 * step, 1.5 * 2.3548 * seed["sigma"])
        p0 += [max(seed["amp"], 1.0), seed["center"], seed["sigma"]]
        lo += [0.0, seed["center"] - window, 0.5 * seed["sigma"]]
        hi += [np.inf, seed["center"] + window, 2.0 * seed["sigma"]]
    try:
        popt, pcov = curve_fit(_sum_of_gaussians, x, y, p0=p0, sigma=err, absolute_sigma=True, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None
    perr = np.sqrt(np.abs(np.diag(pcov)))
    components = []
    for i in range(len(seeds)):
        amp, center, sigma = popt[1 + 3 * i: 4 + 3 * i]; damp, dcenter, dsigma = perr[1 + 3 * i: 4 + 3 * i]
        components.append({"amp": float(amp), "damp": float(damp), "center": float(center), "dcenter": float(dcenter), "sigma": float(sigma),
                           "fwhm": float(2.3548 * sigma), "dfwhm": float(2.3548 * dsigma), "area": float(amp * sigma * math.sqrt(2 * math.pi)),
                           "darea": float(math.sqrt(max(0.0, (sigma * damp) ** 2 + (amp * dsigma) ** 2) * 2 * math.pi))})
    chi2 = float(np.sum(((y - _sum_of_gaussians(x, *popt)) / err) ** 2) / max(len(x) - len(popt), 1))
    return {"background": float(popt[0]), "components": components, "chi2": chi2}


def fit_components_fixed_shape(x: np.ndarray, y: np.ndarray, err: np.ndarray, shapes: list[dict[str, float]]) -> dict[str, Any]:
    """Amplitudes of Gaussians with fixed centres and widths, plus a flat
    background: a weighted linear least-squares problem, so it always has an
    answer, and the area of a peak that has vanished comes out as zero within
    its error instead of a runaway free fit. Amplitudes are not constrained to
    be positive; a negative area within its error is noise, not a signal."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); err = np.where(np.asarray(err, dtype=float) > 0, err, 1.0)
    design = np.column_stack([gaussian(x, 1.0, c["center"], c["sigma"], 0.0) for c in shapes] + [np.ones_like(x)])
    weights = 1.0 / err
    coefficients, *_ = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)
    try:
        covariance = np.linalg.inv((design * weights[:, None]).T @ (design * weights[:, None]))
    except np.linalg.LinAlgError:
        covariance = np.full((len(coefficients), len(coefficients)), np.nan)
    errors = np.sqrt(np.abs(np.diag(covariance)))
    components = []
    for i, shape in enumerate(shapes):
        amp, damp = float(coefficients[i]), float(errors[i]); scale = shape["sigma"] * math.sqrt(2 * math.pi)
        components.append({"amp": amp, "damp": damp, "center": shape["center"], "dcenter": float("nan"), "sigma": shape["sigma"], "fwhm": 2.3548 * shape["sigma"],
                           "dfwhm": float("nan"), "area": amp * scale, "darea": damp * scale})
    residual = y - design @ coefficients
    return {"background": float(coefficients[-1]), "components": components, "chi2": float(np.sum((residual / err) ** 2) / max(len(x) - len(coefficients), 1))}


def edge_flank_mask(x: np.ndarray, y: np.ndarray, err: np.ndarray, *, min_run: int = 3) -> np.ndarray:
    """True for points that belong to the flank of a peak outside the scan
    window: a run of at least `min_run` points at either end that keeps
    rising towards the edge, ends well above the counting noise and above
    everything inside the window. Those points would otherwise dominate the
    fit of the peaks the scan was meant to measure."""
    order = np.argsort(x); y = np.asarray(y, dtype=float)[order]; err = np.asarray(err, dtype=float)[order]
    mask = np.zeros(len(y), dtype=bool)
    noise = float(np.nanmedian(err)) if np.isfinite(err).any() else 1.0
    for edge in ("left", "right"):
        yy = y if edge == "right" else y[::-1]
        k = 1
        while k < len(yy) and yy[-k] > yy[-k - 1] - err[-k - 1 if edge == "right" else k]:
            k += 1
        run = k  # points from the edge inward that rise towards it
        if run >= min_run and run < len(yy) - min_run:
            interior = yy[:-run]
            if yy[-1] - yy[-run] > 5 * noise and yy[-1] > float(np.nanmax(interior)):
                idx = np.arange(len(yy) - run, len(yy))
                mask[idx if edge == "right" else len(yy) - 1 - idx] = True
    unsorted = np.zeros(len(y), dtype=bool); unsorted[order] = mask
    return unsorted


def fit_series_components(scans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Fit every scan of a temperature series with the same components.
    The reference scan is the one with the strongest signal above background;
    its peaks (find_components) are fitted freely and define the component
    centres and widths. Every scan is then fitted with those shapes fixed
    (areas: the order parameters) and, where a component is still clearly
    present, refitted with centre and width free within bounds (centre drift
    and width). Returns the per-scan results and the reference description."""
    profiles = []
    masks: dict[int, np.ndarray] = {}
    for rec in scans:
        x = scan_axis_values(rec); y, err = scan_intensity(rec)
        if x is None or len(x) < 5:
            continue
        keep = ~edge_flank_mask(x, y, err)
        masks[rec["number"]] = keep
        x, y, err = x[keep], y[keep], err[keep]
        if len(x) < 5:
            continue
        b0 = float(np.median(np.sort(y)[: max(3, len(y) // 4)])); noise = float(np.nanmedian(err)) or 1.0
        profiles.append((float(np.nansum(np.clip(y - b0, 0, None)) / noise), rec, x, y, err))  # integrated signal above background, in noise units
    if not profiles:
        return None
    best = max(profiles, key=lambda t: t[0])
    _signal, ref, x, y, err = best
    seeds = find_components(x, y, err)
    reference = fit_components_free(x, y, err, seeds)
    if reference is None or not reference["components"]:
        return None
    shapes = [c for c in reference["components"] if c["amp"] > 5 * max(c["damp"], 1e-9)]
    if not shapes:
        return None
    step = float(np.median(np.abs(np.diff(np.sort(x))))) or 1e-3
    fitted = []
    edge_masked = 0
    for rec in scans:
        xs = scan_axis_values(rec); ys, es = scan_intensity(rec)
        if xs is None or len(xs) < 5:
            continue
        keep = masks.get(rec["number"], np.ones(len(xs), dtype=bool))
        edge_masked += int((~keep).sum())
        xs, ys, es = xs[keep], ys[keep], es[keep]
        if len(xs) < 5:
            continue
        fixed = fit_components_fixed_shape(xs, ys, es, shapes)
        free = fit_components_free(xs, ys, es, shapes, center_window=max(2 * step, 0.6 * max(c["fwhm"] for c in shapes)))
        components = []
        for i, (shape, fx) in enumerate(zip(shapes, fixed["components"])):
            component = {**fx, "label": f"{shape['center']:.2f}", "present": bool(fx["amp"] > 5 * max(fx["damp"], 1e-9)), "background": fixed["background"], "chi2": fixed["chi2"]}
            if free is not None and component["present"]:
                fr = free["components"][i]
                if fr["amp"] > 5 * max(fr["damp"], 1e-9):
                    component.update({"center": fr["center"], "dcenter": fr["dcenter"], "fwhm": fr["fwhm"], "dfwhm": fr["dfwhm"]})
            components.append(component)
        fitted.append({"rec": rec, "fit": components[int(np.argmax([c["amp"] for c in shapes]))], "components": components,
                       "T": float(np.nanmean(rec["T"])), "Tsd": float(np.nanstd(rec["T"])),
                       "Tset": float(np.nanmedian(rec["Tset"])) if rec.get("Tset") is not None else float("nan")})
    info = {"scan": ref["number"], "name": ref["name"], "T": float(np.nanmean(ref["T"])) if ref.get("T") is not None else float("nan"),
            "components": [{"label": f"{c['center']:.2f}", "center": c["center"], "fwhm": c["fwhm"], "area": c["area"]} for c in shapes],
            "primary": int(np.argmax([c["amp"] for c in shapes])), "chi2": reference["chi2"], "edge_points_excluded": edge_masked}
    return fitted, info


def filter_state(rec: dict[str, Any]) -> str:
    """The PG filter position NICE records (filterTrans IN/OUT), '' when absent."""
    return str(rec.get("header", {}).get("filterTrans", "")).strip().upper()


_HIERARCHY_RE = re.compile(r"loop(\d+):\s*([A-Za-z0-9_.]+)\s*=")


def split_two_axis_mesh(rec: dict[str, Any], *, min_points: int = 3) -> list[dict[str, Any]] | None:
    """A NICE mesh trajectory writes one file: an outer loop steps one angle,
    an inner loop sweeps the other. Split it into row records shaped like
    single scans so the mesh figures can draw it. The outer/inner order comes
    from NICE's `trajectoryData.hierarchy`; without it the angle with fewer
    distinct values is the outer loop. A coupled line scan, where both angles
    move at every point, is not a mesh. Each row keeps its own inner-angle
    values (NICE's range mode re-centres the sweep every row, so grids may be
    offset between rows); a row still being measured is padded with NaN to the
    full row length rather than dropped, and the figures show it as blank."""
    header = rec.get("header", {})
    a3, a4 = rec.get("A3"), rec.get("A4")
    if a3 is None or a4 is None or len(a3) < min_points:
        return None
    scanned = {c.strip() for c in str(header.get("trajectory.scanned") or header.get("trajectory.controls") or "").strip("[]").split(",") if c.strip()}
    if not {"A3", "A4"} <= scanned:
        return None
    loops = sorted(_HIERARCHY_RE.findall(str(header.get("trajectoryData.hierarchy", ""))), key=lambda t: int(t[0]))
    axes = [name for _, name in loops]
    if len(axes) >= 2 and {axes[0], axes[1]} == {"A3", "A4"}:
        outer, inner = axes[0], axes[1]
    else:
        n3, n4 = len(np.unique(np.round(a3, 3))), len(np.unique(np.round(a4, 3)))
        if max(n3, n4) < 2:
            return None
        outer, inner = ("A4", "A3") if n4 <= n3 else ("A3", "A4")
    out_values, in_values = rec[outer], rec[inner]
    changes = np.flatnonzero(np.abs(np.diff(out_values)) > 0.05)
    if len(out_values) > 4 and len(changes) >= max(2, len(out_values) // 3):
        return None  # the outer angle moves almost every point: a coupled line scan
    if np.ptp(in_values) <= 0.05:
        return None
    starts = [0, *(changes + 1)]
    ends = [*starts[1:], len(out_values)]
    full = max(b - a for a, b in zip(starts, ends))
    if full < min_points:
        return None
    longest = max(range(len(starts)), key=lambda i: ends[i] - starts[i])
    step = float(np.median(np.abs(np.diff(np.sort(in_values[starts[longest]:ends[longest]]))))) or 0.1
    array_keys = [k for k, v in rec.items() if isinstance(v, np.ndarray)]
    rows: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        n = b - a
        order = np.argsort(in_values[a:b], kind="stable")
        row = {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
        for k in array_keys:
            values = rec[k][a:b][order]
            row[k] = np.concatenate([values, np.full(full - n, np.nan)]) if n < full else values
        x = in_values[a:b][order]
        if n < full:  # the unmeasured remainder of a row in progress continues the grid
            x = np.concatenate([x, x[-1] + step * np.arange(1, full - n + 1)])
        row[inner] = x
        row["axis"] = x.copy()
        row[outer] = np.full(full, float(np.median(out_values[a:b])))
        row.update({"variable": inner, "coupled": False, "npts": int(n), "row": i, "complete": bool(n >= full)})
        rows.append(row)
    rows.sort(key=lambda r: float(r[outer][0]))
    return rows


def planned_loop_span(header: dict[str, Any], axis: str) -> tuple[float, float] | None:
    """The span a NICE trajectory loop will cover for `axis`, from the
    trajectory.config JSON: center/range (fixed centre only; range mode
    without a centre re-centres on the current position and has no fixed
    span) or start/stop."""
    try:
        config = json.loads(str(header.get("trajectory.config", "")))
    except (TypeError, ValueError):
        return None

    def walk(loops: Any) -> tuple[float, float] | None:
        for loop in loops or []:
            for entry in loop.get("vary", []) or []:
                if not (isinstance(entry, list) and len(entry) == 2 and entry[0] == axis and isinstance(entry[1], dict)):
                    continue
                spec = entry[1].get("range", entry[1])
                try:
                    if "start" in spec and "stop" in spec:
                        lo, hi = float(spec["start"]), float(spec["stop"])
                        return (min(lo, hi), max(lo, hi))
                    if "center" in spec and "range" in spec:
                        center, span = float(spec["center"]), float(spec["range"])
                        return (center - span / 2, center + span / 2)
                except (TypeError, ValueError):
                    return None
            found = walk(loop.get("loops"))
            if found:
                return found
        return None
    return walk(config.get("loops") if isinstance(config, dict) else None)


def mesh_guides(mesh: dict[str, Any], series_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Peak positions the temperature series found, as guide lines for a mesh:
    every component of a series scanned along the mesh's stepped angle whose
    centre lies within the mesh's planned or measured span. For a coupled
    theta-two-theta series the point where that scan crossed the component is
    marked too (its A3 moves at half the A4 rate)."""
    outer = mesh["other"]; inner = mesh["variable"]
    span = mesh.get("planned_second_axis") or (mesh["second_axis"][0], mesh["second_axis"][-1])
    guides: list[dict[str, Any]] = []
    for series in series_list:
        if series.get("variable") != outer or not series.get("reference"):
            continue
        for component in series["reference"]["components"]:
            value = float(component["center"])
            if not (span[0] - 0.5 <= value <= span[1] + 0.5):
                continue
            cross = None
            if series.get("other") == inner and np.isfinite(series.get("other_val", float("nan"))):
                slope = 0.5 if series.get("coupled") and outer == "A4" and inner == "A3" else (2.0 if series.get("coupled") else 0.0)
                cross = float(series["other_val"] + slope * (value - series["center"]))
            guides.append({"axis": outer, "value": value, "label": f"{series['label']} {outer} = {value:.2f}\u00b0", "cross": cross, "fwhm": float(component.get("fwhm", 0.0))})
    return guides


def detect_meshes(records: list[dict[str, Any]], *, min_rows: int = 2) -> list[dict[str, Any]]:
    """Consecutive scans from one trajectory with an identical scan-axis grid
    and a stepped second angle form a mesh. Rows are split into temperature
    groups so a row taken while cooling does not join a base-temperature mesh.
    A single NICE mesh file (outer loop over one angle, inner sweep of the
    other) is a mesh on its own; see split_two_axis_mesh."""
    groups: dict[tuple[str, str, tuple[float, ...]], list[dict[str, Any]]] = {}
    meshes: list[dict[str, Any]] = []
    for rec in records:
        rows = split_two_axis_mesh(rec)
        if rows is not None:
            if len(rows) >= min_rows:
                outer = "A4" if rows[0]["variable"] == "A3" else "A3"
                meshes.append({"trajectory": rec["trajectory"] or rec["name"].rstrip("0123456789.datbt7"), "variable": rows[0]["variable"], "other": outer, "rows": rows,
                               "second_axis": sorted({round(float(r[outer][0]), 2) for r in rows}), "filter": filter_state(rec), "single_file": True,
                               "planned_second_axis": planned_loop_span(rec.get("header", {}), outer)})
            continue
        x = scan_axis_values(rec)
        if x is None or rec["npts"] < 20 or rec["variable"] not in {"A3", "A4"}:
            continue
        other = "A4" if rec["variable"] == "A3" else "A3"
        if rec.get(other) is None or np.ptp(rec[other]) > 0.05 or rec.get("coupled"):
            continue
        # rows taken with a different filter state never join the same mesh
        # (a filter change alters higher-order contamination, so the maps differ)
        key = (rec["trajectory"] or rec["name"].rstrip("0123456789.datbt7"), rec["variable"], tuple(np.round(x, 3)), filter_state(rec))
        groups.setdefault(key, []).append(rec)
    for (trajectory, variable, _grid, filt), rows in groups.items():
        other = "A4" if variable == "A3" else "A3"
        rows.sort(key=lambda r: r["number"])
        # temperature groups
        remaining = list(rows)
        while remaining:
            seed = remaining.pop(0)
            seed_T = float(np.nanmean(seed["T"])) if seed["T"] is not None else float("nan")
            seed_sd = float(np.nanstd(seed["T"])) if seed["T"] is not None else 0.0
            group = [seed]
            if seed_sd <= 1.0:  # a row taken while the temperature was moving stands alone
                for rec in list(remaining):
                    t = float(np.nanmean(rec["T"])) if rec["T"] is not None else float("nan")
                    sd = float(np.nanstd(rec["T"])) if rec["T"] is not None else 0.0
                    if sd <= 1.0 and (not np.isfinite(seed_T) or not np.isfinite(t) or abs(t - seed_T) <= max(0.1, 3 * seed_sd)):
                        group.append(rec)
                        remaining.remove(rec)
            needed = max(min_rows, 3) if ".softPosition" in trajectory else min_rows  # findPeak scans that happen to share a grid
            # split into contiguous blocks of the second axis: the same trajectory re-run
            # around a different reflection weeks later is a different mesh
            group.sort(key=lambda r: float(np.median(r[other])))
            blocks: list[list[dict[str, Any]]] = []
            for rec in group:
                value = float(np.median(rec[other]))
                if blocks:
                    prev = [float(np.median(b[other])) for b in blocks[-1]]
                    typical = float(np.median(np.diff(sorted(set(prev))))) if len(set(prev)) > 1 else 0.5
                    if value - max(prev) > max(3 * typical, 1.0):
                        blocks.append([rec]); continue
                    blocks[-1].append(rec)
                else:
                    blocks.append([rec])
            for block in blocks:
                seconds = sorted({round(float(np.median(r[other])), 2) for r in block})
                if len(block) >= needed and len(seconds) >= needed:
                    meshes.append({"trajectory": trajectory, "variable": variable, "other": other, "rows": block, "second_axis": seconds, "filter": filt})
    meshes.sort(key=lambda m: max(r["number"] for r in m["rows"]), reverse=True)
    return meshes


def detect_order_parameter_series(records: list[dict[str, Any]], *, exclude: set[int], min_scans: int = 4, min_span_k: float = 10.0) -> list[dict[str, Any]]:
    """Scans with the same axis through the same place, repeated while the
    sample temperature changes, form an order-parameter series."""
    candidates = [r for r in records if r["number"] not in exclude and r["npts"] >= 8 and r["variable"] in {"A3", "A4"} and r["T"] is not None and scan_axis_values(r) is not None]
    clusters: list[dict[str, Any]] = []
    for rec in sorted(candidates, key=lambda r: r["number"]):
        x = scan_axis_values(rec)
        other = "A4" if rec["variable"] == "A3" else "A3"
        center = float(np.median(x))
        other_val = float(np.median(rec[other])) if rec.get(other) is not None else float("nan")
        placed = False
        for cluster in clusters:
            if cluster["variable"] == rec["variable"] and cluster["coupled"] == bool(rec.get("coupled")) and abs(cluster["center"] - center) <= 1.0 and (not np.isfinite(other_val) or abs(cluster["other_val"] - other_val) <= 1.0):
                cluster["scans"].append(rec)
                placed = True
                break
        if not placed:
            clusters.append({"variable": rec["variable"], "coupled": bool(rec.get("coupled")), "other": other, "center": center, "other_val": other_val, "scans": [rec]})
    series: list[dict[str, Any]] = []
    for cluster in clusters:
        scans = cluster["scans"]
        if len(scans) >= 2:  # drop partial scans (much narrower range than the rest)
            widths = np.array([float(np.ptp(scan_axis_values(s))) for s in scans])
            scans = [s for s, w in zip(scans, widths) if w >= 0.5 * float(np.median(widths))]
        if len(scans) < min_scans:
            continue
        temps = np.array([float(np.nanmean(s["T"])) for s in scans])
        if not np.isfinite(temps).any() or np.nanmax(temps) - np.nanmin(temps) < min_span_k or len({int(round(t)) for t in temps if np.isfinite(t)}) < min_scans:
            continue
        result = fit_series_components(scans)
        if result is None:
            continue
        fitted, reference = result
        if len(fitted) < min_scans:
            continue
        fitted.sort(key=lambda f: f["T"])
        prefix = _majority_stem([f["rec"]["name"] for f in fitted])
        axis_label = f"{cluster['variable']} (theta-2theta)" if cluster["coupled"] else cluster["variable"]
        series.append({**cluster, "scans": scans, "axis_label": axis_label, "fitted": fitted, "reference": reference,
                       "label": prefix or f"{axis_label} scans near {cluster['variable']} = {cluster['center']:.2f}, {cluster['other']} = {cluster['other_val']:.2f}"})
    # Longest series first (the peaks followed throughout the run), then the most
    # recent; the page shows up to MAX_ORDER_PARAMETER_SERIES of them.
    series.sort(key=lambda s: (len(s["fitted"]), max(f["rec"]["number"] for f in s["fitted"])), reverse=True)
    return series


MAX_ORDER_PARAMETER_SERIES = 6


def _majority_stem(names: list[str]) -> str:
    """The naming stem most scans share, e.g. 'Pbam_110' from
    'Pbam_110_100K_A3_570.dat.bt7'; alignment scans mixed into the series
    (fp_A3_524) are outvoted rather than erasing the prefix."""
    stems: dict[str, int] = {}
    for name in names:
        stem = re.sub(r"\.dat\.bt7$", "", name)
        stem = re.sub(r"_entry$", "", stem)            # NICE: <prefix>_<hash><number>_entry
        stem = re.sub(r"_[0-9a-f]{8,}$", "", stem)
        stem = re.sub(r"_\d+$", "", stem)
        stem = re.sub(r"_(A3|A4|baseT|base|T?\d+(?:\.\d+)?K\d*|\d+p\d+K)(?=_|$)", "", stem)
        stem = stem.strip("_")
        if stem:
            stems[stem] = stems.get(stem, 0) + 1
    if not stems:
        return ""
    best, count = max(stems.items(), key=lambda kv: kv[1])
    return best if count >= 0.5 * len(names) else ""


def _common_prefix(names: list[str]) -> str:
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix) and prefix:
            prefix = prefix[:-1]
    return re.sub(r"[_\-\d]+$", "", re.sub(r"\d+K.*$", "", prefix)).strip("_- ")


# --------------------------------------------------------------------------
# peaks the page itself found, placed in reciprocal space
# --------------------------------------------------------------------------
def nearest_fraction(value: float, *, max_denominator: int = 6, tolerance: float = 0.02) -> str | None:
    """'3/4' for 0.749, '-1/4' for -0.251, '1' for 0.995, '0' for 0.004; None
    when no fraction with a small denominator is within tolerance."""
    if not np.isfinite(value):
        return None
    for denominator in range(1, max_denominator + 1):
        numerator = round(value * denominator)
        if abs(value - numerator / denominator) <= tolerance:
            if denominator == 1:
                return str(int(numerator))
            return f"{int(numerator)}/{denominator}"
    return None


def hk_label(h: float, k: float, l: float = 0.0) -> str:
    parts = [nearest_fraction(h), nearest_fraction(k), nearest_fraction(l)]
    if all(parts):
        return "(" + " ".join(parts) + ")"  # type: ignore[arg-type]
    return f"({h:.3f}, {k:.3f}, {l:.2f})"


def peak_representatives(records: list[dict[str, Any]], series: list[dict[str, Any]], *, max_alignment: int = 10) -> list[dict[str, Any]]:
    """Scans whose fitted peak should be placed on the H,K maps: the strongest
    scan of each order-parameter series plus the most recent peaked
    alignment/findPeak scans. Each entry carries the fit."""
    chosen: dict[int, dict[str, Any]] = {}
    for s in series:
        best = max(s["fitted"], key=lambda f: f["fit"]["amp"])
        chosen[best["rec"]["number"]] = {"rec": best["rec"], "fit": best["fit"], "origin": f"order-parameter series {s['label']}"}
    # newest first, one representative per distinct place (A3, A4 within 0.5 deg)
    places: list[tuple[float, float]] = []
    for item in chosen.values():
        rec = item["rec"]
        places.append((float(np.nanmedian(rec["A3"])) if rec["A3"] is not None else 0.0, float(np.nanmedian(rec["A4"])) if rec["A4"] is not None else 0.0))
    # only scans taken with the current orientation calibration: NICE's H,K for
    # scans before a3Zero was set describe a different frame
    current_a3zero = next((r["a3zero"] for r in sorted(records, key=lambda r: r["number"], reverse=True) if np.isfinite(r.get("a3zero", float("nan")))), float("nan"))
    count = 0
    for rec in sorted(records, key=lambda r: r["number"], reverse=True):
        if count >= max_alignment:
            break
        if rec["number"] in chosen or rec["variable"] not in {"A3", "A4"} or rec["npts"] < 8 or rec.get("family") != "alignment_or_rocking_scan":
            continue
        if np.isfinite(current_a3zero) and (not np.isfinite(rec.get("a3zero", float("nan"))) or abs(rec["a3zero"] - current_a3zero) > 1e-3):
            continue
        x = scan_axis_values(rec)
        if x is None or rec["A3"] is None or rec["A4"] is None:
            continue
        place = (float(np.nanmedian(rec["A3"])), float(np.nanmedian(rec["A4"])))
        if any(abs(place[0] - p[0]) < 0.5 and abs(place[1] - p[1]) < 0.5 for p in places):
            continue
        y, err = scan_intensity(rec)
        fit = fit_peak(x, y, err)
        sharp = fit is not None and (rec["variable"] != "A3" or fit["fwhm"] < 1.5)  # a Bragg peak in A3 is mosaic-limited
        if fit and sharp and fit["amp"] > 5 * max(fit["damp"], 1e-9) and fit["amp"] > 5 * math.sqrt(max(fit["background"], 1.0)):
            chosen[rec["number"]] = {"rec": rec, "fit": fit, "origin": "findPeak / alignment scan"}
            places.append(place)
            count += 1
    return list(chosen.values())


def locate_peaks(representatives: list[dict[str, Any]], hk: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Interpolate NICE's per-point H,K to each fitted centre."""
    found: list[dict[str, Any]] = []
    for item in representatives:
        rec, fit = item["rec"], item["fit"]
        info = hk.get(rec["number"])
        x = scan_axis_values(rec)
        if not info or x is None or len(info["H"]) != len(x):
            continue
        order = np.argsort(x)
        try:
            h = float(np.interp(fit["center"], x[order], info["H"][order])); k = float(np.interp(fit["center"], x[order], info["K"][order])); l = float(np.interp(fit["center"], x[order], info["L"][order]))
        except Exception:
            continue
        found.append({"h": h, "k": k, "l": l, "label": hk_label(h, k, l), "scan": rec["number"], "name": rec["name"], "axis": rec["variable"],
                      "center": fit["center"], "fwhm": fit["fwhm"], "amp": fit["amp"], "T": float(np.nanmean(rec["T"])) if rec["T"] is not None else float("nan"), "origin": item["origin"]})
    # one entry per distinct position (0.02 r.l.u.), keep the strongest
    unique: list[dict[str, Any]] = []
    for peak in sorted(found, key=lambda p: -p["amp"]):
        if not any(abs(peak["h"] - u["h"]) < 0.02 and abs(peak["k"] - u["k"]) < 0.02 for u in unique):
            unique.append(peak)
    return sorted(unique, key=lambda p: p["scan"], reverse=True)


# --------------------------------------------------------------------------
# console + queue
# --------------------------------------------------------------------------
def console_events(text_path: Path | None, *, since: datetime | None) -> list[tuple[str, str]]:
    if not text_path or not Path(text_path).exists():
        return []
    events: list[tuple[str, str]] = []
    for line in Path(text_path).read_text(errors="replace").splitlines():
        match = CONSOLE_STAMP_RE.match(line)
        if not match:
            continue
        stamp, message = match.group(1), match.group(2).strip()
        if not message or "Error parsing html" in message or not CONSOLE_EVENT_RE.search(message):
            continue
        if since and stamp < since.strftime("%Y-%m-%dT%H:%M:%S"):
            continue
        events.append((stamp, message[:200]))
    # collapse repeats of the same message within the same minute
    collapsed: list[tuple[str, str]] = []
    for stamp, message in events:
        if collapsed and collapsed[-1][1] == message and collapsed[-1][0][:16] == stamp[:16]:
            continue
        collapsed.append((stamp, message))
    return collapsed[-60:]


def queue_status(queue_path: Path | None) -> dict[str, Any]:
    if not queue_path or not Path(queue_path).exists():
        return {}
    try:
        tree = json.loads(Path(queue_path).read_text(errors="replace"))
    except ValueError:
        return {}
    running: list[str] = []
    queued: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        status = node.get("status", {})
        state = status.get("state")
        command = str(status.get("commandStr", ""))
        if state == "RUNNING" and command:
            running.append(command)
        elif state == "QUEUED" and command and not node.get("children"):
            queued.append(command)
        for child in node.get("children", []) or []:
            walk(child)

    walk(tree)
    return {"running": running[-1] if running else "", "queued_count": len(queued), "next": queued[:4]}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _map_norm(kind: str, vmin: float, vmax: float):
    """Colour scale for a count map: log by default, linear on request."""
    from matplotlib.colors import LogNorm, Normalize
    return LogNorm(vmin=vmin, vmax=vmax) if kind == "log" else Normalize(vmin=0.0, vmax=vmax)


def _map_pair(draw) -> dict[str, str]:
    """Render one map twice (log and linear colour scale) for the page's toggle;
    draw(kind) returns a matplotlib figure."""
    return {"log": _fig_to_data_uri(draw("log")), "linear": _fig_to_data_uri(draw("linear"))}


def _fig_to_data_uri(fig: Any, dpi: int = 105) -> str:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _mesh_figures(mesh: dict[str, Any], hk: dict[int, dict[str, Any]], annotations: dict[str, Any], guides: list[dict[str, Any]] | None = None) -> tuple[list[str], dict[str, Any]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rows = mesh["rows"]
    x = scan_axis_values(rows[0])
    X = np.array([scan_axis_values(r) for r in rows])  # rows of a NICE mesh file may be offset from one another
    staggered = not np.allclose(X, X[0], atol=1e-6)
    Z = np.array([scan_intensity(r)[0] for r in rows])
    seconds = np.array([float(np.nanmedian(r[mesh["other"]])) for r in rows])
    T = np.concatenate([r["T"] for r in rows if r["T"] is not None]) if any(r["T"] is not None for r in rows) else np.array([np.nan])
    vmin = max(0.5, float(np.nanpercentile(Z, 5))); vmax = max(vmin * 2, float(np.nanmax(Z)))
    images: list[Any] = []
    step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.2
    ye = np.concatenate([[seconds[0] - step / 2], (seconds[:-1] + seconds[1:]) / 2, [seconds[-1] + step / 2]])
    def _edges(v: np.ndarray) -> np.ndarray:
        dv = float(np.median(np.diff(v))) or 0.1
        return np.concatenate([[v[0] - dv / 2], (v[:-1] + v[1:]) / 2, [v[-1] + dv / 2]])
    xe = _edges(x)
    if staggered:  # quadrilateral grid: cell boundaries between rows average the neighbours' edges
        row_edges = np.array([_edges(v) for v in X])
        XE = np.vstack([row_edges[:1], (row_edges[:-1] + row_edges[1:]) / 2, row_edges[-1:]])
        YE = np.repeat(ye[:, None], XE.shape[1], axis=1)
    else:
        XE, YE = xe, ye
    filt = f", PG filter {mesh['filter']}" if mesh.get("filter") else ""
    complete = sum(1 for r in rows if r.get("complete", True))
    if mesh.get("single_file"):
        scans_label = f"scan {rows[0]['number']}, {len(rows)} rows" + (f" ({complete} complete)" if complete < len(rows) else "")
    else:
        scans_label = f"scans {rows[0]['number']}-{rows[-1]['number']}, {len(rows)} rows"

    def draw_angles(kind: str):
        n_rows = len(rows)
        planned_rows = mesh.get("planned_second_axis")
        if planned_rows and step > 0:
            n_rows = max(n_rows, min(int(round((planned_rows[1] - planned_rows[0]) / step)) + 1, 3 * len(rows) + 10))
        fig, ax = plt.subplots(figsize=(11, 2.6 + 0.2 * n_rows))
        pm = ax.pcolormesh(XE, YE, np.ma.masked_invalid(np.maximum(Z, 0.5)), shading="flat", norm=_map_norm(kind, vmin, vmax), cmap="viridis")
        fig.colorbar(pm, ax=ax, label=f"counts per {MONITOR_REF:.0f} monitor ({kind})", pad=0.02)
        ax.set_yticks(seconds); ax.set_xlabel(f"{mesh['variable']} (deg)"); ax.set_ylabel(f"{mesh['other']} (deg)")
        planned = mesh.get("planned_second_axis")
        if planned and planned[0] <= ye[0] and planned[1] >= ye[-1] and planned[1] - planned[0] <= 3 * (ye[-1] - ye[0]) + 2 * step:
            ax.set_ylim(planned[0] - step / 2, planned[1] + step / 2)   # rows still to come are left blank
        for g in guides or []:
            ax.axhline(g["value"], color="#d62728", lw=0.9, ls="--", alpha=0.9)
            ax.text(float(np.nanmin(X)) + 0.02 * (float(np.nanmax(X)) - float(np.nanmin(X))), g["value"], g["label"], color="#d62728", fontsize=7.5, va="bottom")
            if g.get("cross") is not None and float(np.nanmin(X)) <= g["cross"] <= float(np.nanmax(X)):
                ax.plot(g["cross"], g["value"], "x", color="#d62728", ms=8, mew=1.4)
        ax.set_title(f"{mesh['trajectory']}: {scans_label}, T = {np.nanmean(T):.3f} K (sd {np.nanstd(T):.3f}){filt}", fontsize=10)
        return fig
    images.append(_map_pair(draw_angles))
    info: dict[str, Any] = {"hk": False}
    if all(r["number"] in hk for r in rows):
        H = np.array([hk[r["number"]]["H"] for r in rows]); K = np.array([hk[r["number"]]["K"] for r in rows])
        if H.shape == Z.shape:
            rad = np.hypot(H, K); phi = np.unwrap(np.arctan2(K, H), axis=1)
            r_c = np.median(rad, axis=1); dr = float(np.median(np.diff(r_c))) if len(r_c) > 1 else 0.01
            r_e = np.concatenate([[r_c[0] - dr / 2], (r_c[:-1] + r_c[1:]) / 2, [r_c[-1] + dr / 2]])
            phi_c = np.median(phi, axis=0); dphi = float(np.median(np.diff(phi_c))) or 1e-3
            phi_e = np.concatenate([[phi_c[0] - dphi / 2], (phi_c[:-1] + phi_c[1:]) / 2, [phi_c[-1] + dphi / 2]])
            RE, PE = np.meshgrid(r_e, phi_e, indexing="ij")
            xr = (float(H.min()) - 0.18, float(H.max()) + 0.18); yr = (float(K.min()) - 0.12, float(K.max()) + 0.12)  # room for marker labels
            meta = hk[rows[0]["number"]]
            r1 = [int(round(v)) for v in meta["refplane1"]]; r2 = [int(round(v)) for v in meta["refplane2"]]

            def draw_hk(kind: str):
                fig, ax = plt.subplots(figsize=(7.5, 7))
                pm = ax.pcolormesh(RE * np.cos(PE), RE * np.sin(PE), np.ma.masked_invalid(np.maximum(Z, 0.5)), shading="flat", norm=_map_norm(kind, vmin, vmax), cmap="viridis")
                fig.colorbar(pm, ax=ax, label=f"counts per {MONITOR_REF:.0f} monitor ({kind})", shrink=0.8)
                for h in range(-3, 4):
                    for k in range(-3, 4):
                        if xr[0] < h < xr[1] and yr[0] < k < yr[1]:
                            ax.plot(h, k, "s", ms=7, mfc="none", mec="red", mew=1.2)
                for marker in annotations.get("markers", []) or []:
                    try:
                        mh, mk = float(marker["h"]), float(marker["k"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if xr[0] < mh < xr[1] and yr[0] < mk < yr[1]:
                        ax.plot(mh, mk, marker.get("symbol", "+"), ms=11, mec=marker.get("color", "magenta"), mfc="none", mew=1.4)
                        ax.annotate(str(marker.get("label", "")), (mh, mk), textcoords="offset points", xytext=(7, 5 if marker.get("symbol", "+") == "+" else -11), fontsize=7.5, color=marker.get("color", "magenta"))
                ax.set_xlabel(f"H (r.l.u.) along refplane1 = [{r1[0]} {r1[1]} {r1[2]}]"); ax.set_ylabel(f"K (r.l.u.) along refplane2 = [{r2[0]} {r2[1]} {r2[2]}]")
                ax.set_title(f"{mesh['trajectory']} in reciprocal space (H,K from NICE; a = {meta['a']:.4f} A).\nRed squares: integer Bragg positions; green circles: peaks found by fitting; other markers: instrument scientist's annotations.", fontsize=9)
                ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal"); ax.grid(alpha=0.25)
                return fig
            images.append(_map_pair(draw_hk))
            info["hk"] = True
    # waterfall
    fig, ax = plt.subplots(figsize=(11, 2.0 + 0.6 * len(rows)))
    bkg = float(np.nanmedian(Z)); spread = float(np.nanstd(Z[Z < np.nanpercentile(Z, 98)])) if np.isfinite(Z).any() else 1.0
    off = max(3 * spread, 0.25 * (float(np.nanmax(Z)) - bkg) / max(len(rows) - 1, 1)) * 1.6
    xmin, xmax = float(X.min()), float(X.max())
    for i, r in enumerate(rows):
        y, err = scan_intensity(r); color = PALETTE[i % 10]; yy = y - bkg + i * off; xi = X[i]
        ax.fill_between(xi, yy - err, yy + err, color=color, alpha=0.18, lw=0); ax.plot(xi, yy, color=color, lw=0.9); ax.axhline(i * off, color=color, lw=0.5, ls=":", alpha=0.6)
        tag = f"#{r['number']} row {r['row'] + 1}" if "row" in r else f"#{r['number']}"
        ax.text(xmax + 0.01 * (xmax - xmin), i * off, f"{tag}  {mesh['other']} = {float(np.nanmedian(r[mesh['other']])):.2f}  T = {np.nanmean(r['T']) if r['T'] is not None else float('nan'):.3f} K", color=color, va="center", fontsize=8.5)
    ax.set_xlim(xmin, xmax + 0.3 * (xmax - xmin)); ax.set_xlabel(f"{mesh['variable']} (deg)"); ax.set_ylabel(f"counts per {MONITOR_REF:.0f} monitor, background ({bkg:.0f}) subtracted, rows offset by {off:.0f}", fontsize=8.5)
    ax.set_title("Each row as a line scan (band = +/- 1 sigma)", fontsize=10); ax.grid(alpha=0.25, axis="x")
    images.append(_fig_to_data_uri(fig))
    info.update({"rows": len(rows), "complete_rows": complete, "points": int(np.isfinite(Z).sum()), "guides": [g["label"] for g in guides or []], "T_mean": float(np.nanmean(T)), "T_sd": float(np.nanstd(T)), "max": float(np.nanmax(Z)), "background": bkg})
    return images, info


def _stack_grid(fitted: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scans of a series on one common axis grid (rows = scans sorted by T):
    scans with different windows or step sizes are interpolated onto the grid
    spanning the union of their ranges, NaN outside each scan's own window."""
    xs = [np.asarray(scan_axis_values(f["rec"]), dtype=float) for f in fitted]
    lo = min(float(np.nanmin(x)) for x in xs); hi = max(float(np.nanmax(x)) for x in xs)
    n = max(len(x) for x in xs)
    grid = np.linspace(lo, hi, n)
    Z = np.full((len(fitted), n), np.nan)
    for i, (f, x) in enumerate(zip(fitted, xs)):
        y = scan_intensity(f["rec"])[0]
        order = np.argsort(x)
        inside = (grid >= x.min()) & (grid <= x.max())
        Z[i, inside] = np.interp(grid[inside], x[order], np.asarray(y, dtype=float)[order])
    T = np.array([f["T"] for f in fitted])
    return grid, T, Z


def _edges(centers: np.ndarray) -> np.ndarray:
    d = np.diff(centers); d = np.where(d > 0, d, np.median(d[d > 0]) if (d > 0).any() else 1.0)
    return np.concatenate([[centers[0] - d[0] / 2], (centers[:-1] + centers[1:]) / 2, [centers[-1] + d[-1] / 2]])


def order_parameter_fits(annotations: dict[str, Any], warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Off-line order-parameter fits from the annotations file, keyed
    by series label. The page draws them; it never computes them. An entry
    needs a series label and a numeric Tc; a curve (T and I lists of equal
    length) is optional. Malformed entries are skipped with a warning."""
    out: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(annotations.get("order_parameter_fits", []) or []):
        try:
            label = str(entry["series"]).strip(); tc = float(entry["Tc"])
            if not label or not np.isfinite(tc):
                raise ValueError("series label and finite Tc required")
            fit = {k: v for k, v in entry.items()}
            fit["Tc"] = tc
            fit["Tc_err"] = float(entry.get("Tc_err", float("nan")))
            curve = entry.get("curve")
            if curve:
                Tc_, Ic_ = np.asarray(curve["T"], dtype=float), np.asarray(curve["I"], dtype=float)
                if Tc_.shape != Ic_.shape or Tc_.ndim != 1 or len(Tc_) < 2:
                    raise ValueError("curve.T and curve.I must be equal-length lists")
                fit["curve"] = {"T": Tc_, "I": Ic_}
            out[label] = fit
        except Exception as exc:
            warnings.append(f"order_parameter_fits[{i}] ignored: {exc}")
    return out


def _fit_description(fit: dict[str, Any]) -> str:
    tc = f"Tc = {fit['Tc']:.2f}"
    if np.isfinite(fit.get("Tc_err", float("nan"))):
        tc += f" ± {fit['Tc_err']:.2f}"
    tc += " K"
    return tc


def _draw_fit(ax: Any, fit: dict[str, Any] | None, *, color: str, label: str, text_prefix: str = "", text_y: float = 0.97) -> None:
    """Fitted curve (if supplied) and the Tc line with its value on the plot.
    `text_y` (axes fraction) lets several Tc labels stack on one panel."""
    if not fit:
        return
    if fit.get("curve"):
        ax.plot(fit["curve"]["T"], fit["curve"]["I"], "-", color=color, lw=1.4, alpha=0.9, label=label)
    ax.axvline(fit["Tc"], color=color, ls="--", lw=1.0)
    ax.annotate(text_prefix + _fit_description(fit), xy=(fit["Tc"], text_y), xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=9, color=color,
                xytext=(-4, 0), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85))


def _order_parameter_figures(series: dict[str, Any], fit: dict[str, Any] | None = None) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fitted = series["fitted"]; axis = series["variable"]
    T = np.array([f["T"] for f in fitted]); F = [f["fit"] for f in fitted]
    n_components = len(fitted[0].get("components", [])) or 1
    components = [[f["components"][i] if f.get("components") else f["fit"] for f in fitted] for i in range(n_components)]
    reference = series.get("reference") or {}
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    for i, comp in enumerate(components):
        tag = f"{axis} = {comp[0]['label']}\u00b0" if n_components > 1 else "integrated (Gaussian area)"
        axs[0].errorbar(T, [c["area"] for c in comp], yerr=[c["darea"] for c in comp], fmt="o", ms=4, color=PALETTE[i % 10], label=tag)
    if n_components == 1:
        axs[0].errorbar(T, [f["amp"] for f in F], yerr=[f["damp"] for f in F], fmt="s", ms=3.5, color=PALETTE[1], label="peak height above background")
    _draw_fit(axs[0], fit, color=PALETTE[3], label=f"off-line fit: {fit.get('model', 'order-parameter model')}" if fit else "")
    axs[0].axhline(0, color="0.6", lw=0.6)
    axs[0].set_ylabel(f"area (counts per {MONITOR_REF:.0f} mon.)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[0].set_title(f"{series['label']}: {len(fitted)} {series.get('axis_label', axis)} scans, {T.min():.1f}-{T.max():.1f} K" + (f"; {n_components} component{'s' if n_components > 1 else ''}, shapes from scan {reference.get('scan')}" if reference else ""), fontsize=9.5)
    for i, comp in enumerate(components):
        present = np.array([bool(c.get("present", True)) and np.isfinite(c["dcenter"]) for c in comp])
        if present.any():
            ref_center = reference["components"][i]["center"] if reference else comp[0]["center"]
            axs[1].errorbar(T[present], [c["center"] - ref_center for c, p in zip(comp, present) if p], yerr=[c["dcenter"] for c, p in zip(comp, present) if p], fmt="o", ms=4, color=PALETTE[i % 10], label=f"{axis} = {comp[0]['label']}\u00b0" if n_components > 1 else "")
            axs[2].errorbar(T[present], [c["fwhm"] for c, p in zip(comp, present) if p], yerr=[c["dfwhm"] for c, p in zip(comp, present) if p], fmt="o", ms=4, color=PALETTE[i % 10])
    axs[1].axhline(0, color="0.6", lw=0.6); axs[1].set_ylabel(f"{axis} centre shift (deg)"); axs[1].grid(alpha=0.3)
    if n_components > 1:
        axs[1].legend(fontsize=8)
    axs[2].set_ylabel(f"{axis} FWHM (deg)"); axs[2].grid(alpha=0.3)
    fwhm_max = max([c["fwhm"] for comp in components for c in comp if np.isfinite(c.get("dfwhm", np.nan))] or [1.0])
    axs[2].set_ylim(0, fwhm_max * 1.3); axs[2].set_xlabel("sample temperature (K, measured)")
    caption = ("Integrated intensity of each component (centres and widths fixed from the reference scan, amplitudes fitted), then centre shift and FWHM from a bounded free refit where the component is still clearly present, versus measured temperature."
               if n_components > 1 else "Integrated intensity and peak height (centre and width fixed from the reference scan), fitted centre shift and FWHM where the peak is clearly present, versus measured temperature.")
    if fit:
        caption += f" Dashed line and curve: the off-line fit, {_fit_description(fit)} ({fit.get('model', 'model not stated')}; {fit.get('fitter', 'fitter not stated')})."
    images = [(_fig_to_data_uri(fig), caption)]
    # T-vs-angle image; scans with different windows are interpolated onto one grid
    try:
        grid, Tg, Z = _stack_grid(fitted)
        zmin, zmax = max(0.5, float(np.nanpercentile(Z, 3))), float(np.nanmax(Z))

        def draw_stack(kind: str):
            fig, ax = plt.subplots(figsize=(7.5, 7))
            pm = ax.pcolormesh(_edges(grid), _edges(Tg), np.ma.masked_invalid(np.maximum(Z, 0.5)), shading="flat", norm=_map_norm(kind, zmin, zmax), cmap="viridis")
            fig.colorbar(pm, ax=ax, label=f"counts per {MONITOR_REF:.0f} monitor ({kind})")
            ax.set_xlabel(f"{axis} (deg)"); ax.set_ylabel("sample temperature (K)"); ax.set_title(f"{series['label']}: {axis} scans stacked by temperature", fontsize=10)
            return fig
        images.append((_map_pair(draw_stack), "Scans stacked by temperature; the button switches the colour scale between log and linear."))
    except Exception:
        pass
    # setpoint tracking
    tset = np.array([f["Tset"] for f in fitted])
    tset = np.where(tset < T.min() - 0.5, np.nan, tset)  # a setpoint below the reachable base is not a control error
    if np.isfinite(tset).sum() >= 3:
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        ok = np.isfinite(tset) & (tset > 0)
        ax.errorbar(tset[ok], T[ok] - tset[ok], yerr=[f["Tsd"] for f, o in zip(fitted, ok) if o], fmt="o", ms=4, color=PALETTE[1])
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xlabel("setpoint (K)"); ax.set_ylabel("measured - setpoint (K)"); ax.grid(alpha=0.3)
        ax.set_title("Temperature control: offset of the measured mean from the setpoint (bar = sd within scan)", fontsize=10)
        images.append((_fig_to_data_uri(fig), "Measured minus setpoint temperature at each step."))
    areas = np.array([f["area"] for f in F])
    present = [f for f in F if f.get("present", True) and np.isfinite(f.get("dcenter", np.nan))]
    info = {"n": len(fitted), "T_min": float(T.min()), "T_max": float(T.max()), "area_first": float(areas[0]), "area_last": float(areas[-1]),
            "center_drift": float(present[-1]["center"] - present[0]["center"]) if present else float("nan"),
            "fwhm_range": (float(min(f["fwhm"] for f in present)), float(max(f["fwhm"] for f in present))) if present else (float("nan"), float("nan")),
            "components": [c["label"] for c in reference.get("components", [])], "reference_scan": reference.get("scan"), "reference_T": reference.get("T"),
            "max_offset": float(np.nanmax(np.abs(T[np.isfinite(tset) & (tset > 0)] - tset[np.isfinite(tset) & (tset > 0)]))) if (np.isfinite(tset) & (tset > 0)).any() else float("nan"),
            "max_sd": float(max(f["Tsd"] for f in fitted))}
    return images, info


def _order_parameter_comparison_figures(series_list: list[dict[str, Any]], fits: dict[str, dict[str, Any]] | None = None) -> list[tuple[str, str]]:
    """Two or more order-parameter series on one temperature axis: the
    integrated intensities as lines (separate y-axes when there are two
    series, so each curve fills its panel), and the stacked colour maps side
    by side sharing the temperature axis. Off-line fits, when the
    annotations supply them, are drawn with their Tc."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out: list[tuple[str, str]] = []
    fits = fits or {}
    series_list = [s for s in series_list if s.get("fitted")]
    if len(series_list) < 2:
        return out
    fig, ax = plt.subplots(figsize=(9, 5.5))
    handles = []
    drawn_tc: list[float] = []
    k = 0
    for i, s in enumerate(series_list):
        T = np.array([f["T"] for f in s["fitted"]])
        n_components = len(s["fitted"][0].get("components", [])) or 1
        fit = fits.get(s["label"])
        for j in range(n_components):
            comp = [f["components"][j] if f.get("components") else f["fit"] for f in s["fitted"]]
            area = np.array([c["area"] for c in comp]); err = np.array([c["darea"] for c in comp])
            # normalise to the mean of the three strongest points so curves of very different intensity share one axis
            scale = float(np.mean(np.sort(area)[-3:])) if len(area) >= 3 else float(np.max(area))
            scale = scale if scale > 0 else 1.0
            tag = f"{s['label']}" + (f", {s['variable']} = {comp[0]['label']}\u00b0" if n_components > 1 else f" ({s.get('axis_label', s['variable'])} scans near {s['variable']} = {s['center']:.2f}\u00b0)") + f" [max {scale:.0f}]"
            joined = not (fit and n_components == 1)
            short = len(T) < 10  # a short diagnostic series is drawn lighter than the peaks followed throughout
            h = ax.errorbar(T, area / scale, yerr=err / scale, fmt="o", ms=4 if not short else 3, lw=1 if not short else 0.7, ls=("-" if not short else "--") if joined else "none", alpha=1.0 if not short else 0.6, color=PALETTE[k % 10], label=tag)
            handles.append(h); k += 1
            if fit and n_components == 1 and fit.get("curve"):
                (hc,) = ax.plot(fit["curve"]["T"], np.asarray(fit["curve"]["I"], dtype=float) / scale, "-", color=PALETTE[(k - 1) % 10], lw=1.4, label=f"{s['label']} off-line fit")
                handles.append(hc)
        if fit and not any(abs(fit["Tc"] - t) < 1e-6 for t in drawn_tc):  # a shared Tc is drawn once; per-peak Tc values stack
            per_peak = len({round(f["Tc"], 6) for f in fits.values()}) > 1
            _draw_fit(ax, {k2: v for k2, v in fit.items() if k2 != "curve"}, color=PALETTE[(k - 1) % 10] if per_peak else "0.25", label="",
                      text_prefix=f"{s['label']}: " if per_peak else "", text_y=0.97 - 0.075 * len(drawn_tc))
            drawn_tc.append(fit["Tc"])
    ax.axhline(0, color="0.6", lw=0.6); ax.set_ylim(-0.15, 1.35)
    ax.set_ylabel("normalised integrated intensity", fontsize=9)
    ax.set_xlabel("sample temperature (K, measured)"); ax.grid(alpha=0.3)
    ax.legend(handles=handles, labels=[h.get_label() for h in handles], fontsize=7.5, loc="lower left")
    ax.set_title("Order parameters compared: integrated intensity of every fitted component versus temperature, normalised", fontsize=10)
    fig.tight_layout()
    caption = "Every fitted component on one temperature axis, each normalised to the mean of its three strongest points (the bracketed number is that scale in counts per 4800 monitor), so a nuclear peak stays flat near 1 and a magnetic peak falls to 0. Series of fewer than ten scans are drawn dashed and lighter."
    if len(drawn_tc) > 1:
        caption += " Curves and dashed lines are the off-line fits from the annotations file, one Tc per peak."
    elif drawn_tc:
        caption += " Curves and the dashed Tc line are the off-line fit from the annotations file."
    else:
        caption += " Lines join the measured points in temperature order."
    out.append((_fig_to_data_uri(fig), caption))
    try:
        grids = [_stack_grid(s["fitted"]) for s in series_list]

        def draw_side(kind: str):
            fig, axs = plt.subplots(1, len(series_list), figsize=(5.2 * len(series_list), 6.5), sharey=True)
            for a, s, (grid, Tg, Z) in zip(np.atleast_1d(axs), series_list, grids):
                pm = a.pcolormesh(_edges(grid), _edges(Tg), np.ma.masked_invalid(np.maximum(Z, 0.5)), shading="flat", norm=_map_norm(kind, max(0.5, float(np.nanpercentile(Z, 3))), float(np.nanmax(Z))), cmap="viridis")
                fig.colorbar(pm, ax=a, label=f"counts per {MONITOR_REF:.0f} monitor ({kind})")
                a.set_xlabel(f"{s['variable']} (deg)"); a.set_title(f"{s['label']}: {s['variable']} \u2248 {s['center']:.2f}\u00b0, {s['other']} \u2248 {s['other_val']:.2f}\u00b0", fontsize=10)
            np.atleast_1d(axs)[0].set_ylabel("sample temperature (K)")
            fig.tight_layout()
            return fig
        out.append((_map_pair(draw_side), "The same series as colour maps side by side, sharing the temperature axis; the button switches the colour scale between log and linear."))
    except Exception:
        pass
    return out


def _activity_figure(records: list[dict[str, Any]], events: list[tuple[str, str]], tz: timezone, now: datetime) -> str:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    since = now - timedelta(hours=48)
    recent = [r for r in records if r["start"] and r["start"] >= since]
    if recent:
        starts = [r["start"] for r in recent]
        axs[0].hist(mdates.date2num(starts), bins=48, color=PALETTE[0])
    axs[0].set_ylabel("scans started / hour"); axs[0].set_title("Last 48 hours", fontsize=10); axs[0].grid(alpha=0.3)
    for r in recent:
        if r["T"] is not None and r["ts"] is not None:
            t = r["ts"] / 1000.0; ok = t > 1.0e9
            axs[1].plot([datetime.fromtimestamp(v, tz) for v in t[ok]], r["T"][ok], ".", ms=2, color=PALETTE[1])
            if r.get("Tset") is not None:
                axs[1].plot([datetime.fromtimestamp(v, tz) for v in t[ok]], r["Tset"][ok], "-", lw=0.6, color="0.4")
    axs[1].set_ylabel("temp (K): measured (points), setpoint (line)"); axs[1].grid(alpha=0.3)
    for stamp, message in events:
        if not re.search(r"CRITICAL|broken|recovered|SERIOUS|IMPORTANT: Server", message):
            continue
        try:
            when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz)
        except ValueError:
            continue
        if when >= since:
            for ax in axs:
                ax.axvline(when, color=PALETTE[3], lw=0.8, alpha=0.7)
    axs[1].set_xlim(since, now); fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def _gallery_figure(records: list[dict[str, Any]]) -> str:
    import matplotlib.pyplot as plt

    latest = sorted(records, key=lambda r: r["number"])[-8:][::-1]  # newest first, like every list on the page
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6))
    for ax, rec in zip(axs.ravel(), latest):
        x = scan_axis_values(rec); y, err = scan_intensity(rec)
        if x is None:
            x = np.arange(len(y))
        ax.errorbar(x, y, yerr=err, fmt="o-", ms=2.5, lw=0.8, color=PALETTE[0])
        ax.set_title(f"#{rec['number']} {rec['name'].replace('.dat.bt7', '')[:28]}\n{rec['start']:%m-%d %H:%M} T={np.nanmean(rec['T']) if rec['T'] is not None else float('nan'):.2f} K", fontsize=8)
        ax.set_xlabel(rec["variable"] or "point", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    for ax in axs.ravel()[len(latest):]:
        ax.axis("off")
    fig.suptitle(f"Latest scans, newest first (counts per {MONITOR_REF:.0f} monitor)", fontsize=10)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


# --------------------------------------------------------------------------
# scan browser (sortable table + linked plot with axis dropdowns)
# --------------------------------------------------------------------------
_BROWSER_X_COLUMNS = (("axis", None), ("H", "H"), ("K", "K"), ("L", "L"), ("E", "E (meV)"), ("A3", "A3 (deg)"), ("A4", "A4 (deg)"), ("T", "T (K)"), ("time", "time (s)"))


def _json_floats(values: np.ndarray | None, digits: int = 4) -> list[float | None] | None:
    if values is None:
        return None
    out: list[float | None] = []
    for v in np.asarray(values, dtype=float).tolist():
        out.append(None if not np.isfinite(v) else round(v, digits))
    return out


def _center(values: np.ndarray | None, digits: int = 3) -> float | None:
    if values is None or len(values) == 0 or not np.isfinite(values).any():
        return None
    return round(float(np.nanmedian(values)), digits)


def scan_browser_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One JSON-ready row per scan: the table columns (centres of H, K, L, E,
    A4, Ei, mean T) and the per-point series the plot can put on either axis.
    Counts per MONITOR_REF carry the tasreduction one-sigma error."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        try:
            rate, err = scan_intensity(rec)
        except Exception:
            rate, err = None, None
        det = rec.get("det"); time = rec.get("time")
        det_rate = (det / np.where(time > 0, time, np.nan)) if det is not None and time is not None else None
        series = {"axis": _json_floats(scan_axis_values(rec)), "rate": _json_floats(rate, 3), "err": _json_floats(err, 3),
                  "det": _json_floats(det, 1), "mon": _json_floats(rec.get("mon"), 1), "detrate": _json_floats(det_rate, 3)}
        for key, _label in _BROWSER_X_COLUMNS[1:]:
            series[key] = _json_floats(rec.get(key))
        rows.append({
            "number": rec["number"], "name": rec["name"], "start": rec["start"].strftime("%m-%d %H:%M") if rec.get("start") else "",
            "start_iso": rec["start"].isoformat(timespec="minutes") if rec.get("start") else "", "npts": rec["npts"],
            "family": rec.get("family", ""), "axis": rec.get("variable") or "", "coupled": bool(rec.get("coupled")),
            "H": _center(rec.get("H")), "K": _center(rec.get("K")), "L": _center(rec.get("L")), "E": _center(rec.get("E")),
            "A4": _center(rec.get("A4"), 2), "Ei": _center(rec.get("ei"), 2), "T": round(float(np.nanmean(rec["T"])), 3) if rec.get("T") is not None and np.isfinite(rec["T"]).any() else None,
            "series": series,
        })
    return rows


_BROWSER_COLUMNS = (("number", "#"), ("name", "file"), ("start", "start"), ("npts", "pts"), ("axis", "axis"), ("family", "family"),
                    ("H", "H"), ("K", "K"), ("L", "L"), ("E", "E"), ("A3", "A3"), ("A4", "A4"), ("T", "T"), ("time", "time"), ("mon", "monitor"), ("det", "detector"))
# columns drawn as a range bar (the scan's min..max within the global min..max of all scans)
_BROWSER_BAR_COLUMNS = ("H", "K", "L", "E", "A3", "A4", "T", "time", "mon", "det")

_BROWSER_JS = r"""
(function(){
  // Static port of the TAIPAN Scan Dashboard (Dash) used at ANSTO exp1337 and HFIR HB-1A:
  // sortable grid, range bars with min/max/step on hover, multi-select overlay plot with
  // free choice of x and y columns and a linear/log toggle, refresh when new scans land.
  const DATA = JSON.parse(document.getElementById('scan-browser-data').textContent);
  const byNumber = {}; DATA.forEach(r => byNumber[r.number] = r);
  const COLS = %COLS%;
  const BAR = %BAR%;
  const XCOLS = %XCOLS%;
  const MON = %MON%;
  const BUILT = %BUILT%;
  const PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
  const el = id => document.getElementById(id);
  const tbody = document.querySelector('#scan-browser-table tbody');
  const famSel = el('scan-browser-family'), textIn = el('scan-browser-text'), xSel = el('scan-browser-x'), ySel = el('scan-browser-y'), logChk = el('scan-browser-log'), count = el('scan-browser-count'), refreshChk = el('scan-browser-refresh');
  const newest = DATA.length ? Math.max(...DATA.map(r => r.number)) : null;
  let state = {sortKey: 'number', sortDir: -1, family: '', text: '', selected: newest === null ? [] : [newest], x: 'axis', y: 'rate', log: false, refresh: true};
  try { const saved = JSON.parse(localStorage.getItem('scan-browser-state') || 'null'); if (saved && typeof saved === 'object') state = Object.assign(state, saved); } catch (e) {}
  state.selected = (state.selected || []).filter(n => byNumber[n]); if (!state.selected.length && newest !== null) state.selected = [newest];
  function save(){ try { localStorage.setItem('scan-browser-state', JSON.stringify(state)); } catch (e) {} }

  // per-scan min / max / step of every series, and the global span per column (for the bars)
  function stats(vals){ const v = (vals || []).filter(x => x !== null && isFinite(x)); if (!v.length) return null; let mn = Infinity, mx = -Infinity; v.forEach(x => { if (x < mn) mn = x; if (x > mx) mx = x; }); const d = []; for (let i = 1; i < v.length; i++) d.push(Math.abs(v[i] - v[i-1])); d.sort((a, b) => a - b); return {min: mn, max: mx, step: d.length ? d[Math.floor(d.length / 2)] : 0}; }
  const GLOBAL = {};
  DATA.forEach(r => { r.stats = {}; BAR.forEach(c => { const st = stats(r.series[c]); r.stats[c] = st; if (st) { const g = GLOBAL[c] || (GLOBAL[c] = {min: Infinity, max: -Infinity}); g.min = Math.min(g.min, st.min); g.max = Math.max(g.max, st.max); } }); });
  function bar(r, c){
    const st = r.stats[c], g = GLOBAL[c]; if (!st || !g) return '';
    const span = g.max - g.min; const left = span > 0 ? 100 * (st.min - g.min) / span : 0; const width = span > 0 ? Math.max(1.5, 100 * (st.max - st.min) / span) : 100;
    const title = 'min ' + st.min.toPrecision(5) + '\nmax ' + st.max.toPrecision(5) + '\nstep ' + st.step.toPrecision(3) + '\n(all scans ' + g.min.toPrecision(5) + ' to ' + g.max.toPrecision(5) + ')';
    const label = st.max - st.min < 1e-9 ? st.min.toPrecision(4) : st.min.toPrecision(4) + ' – ' + st.max.toPrecision(4);
    return '<div class="bar" title="' + title.replace(/"/g, '&quot;') + '"><span class="fill" style="left:' + left.toFixed(1) + '%;width:' + width.toFixed(1) + '%"></span></div><span class="barlabel">' + label + '</span>';
  }
  const fams = [...new Set(DATA.map(r => r.family))].sort();
  fams.forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f; famSel.appendChild(o); });
  famSel.value = state.family; textIn.value = state.text; logChk.checked = !!state.log; refreshChk.checked = state.refresh !== false;
  function sortValue(r, key){ if (BAR.includes(key)) { const st = r.stats[key]; return st ? st.min : null; } return r[key]; }
  function visible(){
    const t = state.text.toLowerCase();
    let rows = DATA.filter(r => (!state.family || r.family === state.family) && (!t || (r.name + ' ' + r.family + ' ' + r.axis + ' ' + r.number).toLowerCase().includes(t)));
    rows.sort((a, b) => { const x = sortValue(a, state.sortKey), y = sortValue(b, state.sortKey); if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1; return (x < y ? -1 : 1) * state.sortDir; });
    return rows;
  }
  function cell(r, c){ if (BAR.includes(c)) return bar(r, c); const v = r[c]; if (v === null || v === undefined) return ''; return typeof v === 'number' ? String(v) : v; }
  function render(){
    const rows = visible();
    tbody.innerHTML = rows.map(r => { const i = state.selected.indexOf(r.number); const sel = i >= 0; return '<tr data-number="' + r.number + '"' + (sel ? ' class="selected" style="box-shadow: inset 5px 0 0 ' + PALETTE[i % 10] + '"' : '') + '>' + COLS.map(c => '<td class="c-' + c[0] + '">' + (c[0] === 'number' && sel ? '<span class="swatch" style="background:' + PALETTE[i % 10] + '"></span>' : '') + cell(r, c[0]) + '</td>').join('') + '</tr>'; }).join('');
    count.textContent = rows.length + ' of ' + DATA.length + ' scans; ' + state.selected.length + ' plotted';
    document.querySelectorAll('#scan-browser-table th').forEach(th => { th.classList.toggle('sorted', th.dataset.key === state.sortKey); th.querySelector('span').textContent = th.dataset.key === state.sortKey ? (state.sortDir > 0 ? ' ▲' : ' ▼') : ''; });
  }
  function axisOptions(){
    const sel = state.selected.map(n => byNumber[n]).filter(Boolean); const opts = [];
    XCOLS.forEach(c => { const key = c[0]; if (sel.some(r => (r.series[key] || []).some(v => v !== null))) opts.push([key, key === 'axis' ? (sel.length === 1 ? (sel[0].axis || 'scan axis') : 'scan axis') : c[1]]); });
    return opts;
  }
  const YOPTS = [['rate', 'counts per ' + MON + ' monitor'], ['det', 'detector counts'], ['detrate', 'detector counts / s'], ['mon', 'monitor counts']];
  function plot(){
    const sel = state.selected.map(n => byNumber[n]).filter(Boolean); if (!sel.length) return;
    const xo = axisOptions(); xSel.innerHTML = xo.map(o => '<option value="' + o[0] + '">' + o[1] + '</option>').join(''); if (!xo.some(o => o[0] === state.x)) state.x = 'axis'; xSel.value = state.x;
    const yo = YOPTS.concat(xo.filter(o => o[0] !== 'axis')); ySel.innerHTML = yo.map(o => '<option value="' + o[0] + '">' + o[1] + '</option>').join(''); if (!yo.some(o => o[0] === state.y)) state.y = 'rate'; ySel.value = state.y;
    const traces = sel.map((r, i) => {
      const x = r.series[state.x] || r.series.axis || r.series.det.map((_, j) => j + 1);
      const tr = {x: x, y: r.series[state.y], mode: 'lines+markers', type: 'scatter', marker: {size: 5, color: PALETTE[i % 10]}, line: {width: 1, color: PALETTE[i % 10]}, name: '#' + r.number + ' ' + r.name.replace('.dat.bt7', '') + (r.T === null ? '' : ' (' + r.T.toFixed(2) + ' K)')};
      if (state.y === 'rate' && r.series.err) tr.error_y = {type: 'data', array: r.series.err, visible: true, thickness: 1, color: PALETTE[i % 10]};
      return tr;
    });
    const xlabel = (xo.find(o => o[0] === state.x) || ['', state.x])[1], ylabel = (yo.find(o => o[0] === state.y) || ['', state.y])[1];
    const title = sel.length === 1 ? '#' + sel[0].number + ' ' + sel[0].name.replace('.dat.bt7', '') + '  (' + sel[0].family + ', ' + sel[0].start + ')' : sel.length + ' scans';
    Plotly.newPlot('scan-browser-plot', traces, {margin: {l: 60, r: 20, t: 40, b: 50}, height: 400, title: {text: title, font: {size: 13}}, xaxis: {title: xlabel, zeroline: false}, yaxis: {title: ylabel, type: state.log ? 'log' : 'linear', zeroline: false}, showlegend: true, legend: {orientation: 'h', y: -0.22, font: {size: 11}}}, {responsive: true, displaylogo: false});
  }
  tbody.addEventListener('click', e => { const tr = e.target.closest('tr[data-number]'); if (!tr) return; const n = Number(tr.dataset.number);
    if (e.metaKey || e.ctrlKey || e.shiftKey) { state.selected = state.selected.includes(n) ? state.selected.filter(x => x !== n) : state.selected.concat([n]); } else { state.selected = [n]; }
    save(); render(); plot(); });
  document.querySelectorAll('#scan-browser-table th').forEach(th => th.addEventListener('click', () => { const k = th.dataset.key; if (state.sortKey === k) state.sortDir = -state.sortDir; else { state.sortKey = k; state.sortDir = (k === 'number' || k === 'start') ? -1 : 1; } save(); render(); }));
  famSel.addEventListener('change', () => { state.family = famSel.value; save(); render(); });
  textIn.addEventListener('input', () => { state.text = textIn.value; save(); render(); });
  xSel.addEventListener('change', () => { state.x = xSel.value; save(); plot(); }); ySel.addEventListener('change', () => { state.y = ySel.value; save(); plot(); });
  logChk.addEventListener('change', () => { state.log = logChk.checked; save(); plot(); });
  refreshChk.addEventListener('change', () => { state.refresh = refreshChk.checked; save(); });
  el('scan-browser-clear').addEventListener('click', () => { state.selected = newest === null ? [] : [newest]; save(); render(); plot(); });
  render(); if (window.Plotly) plot(); else el('scan-browser-plot').textContent = 'Plot needs the Plotly script (network).';
  // Refresh: the page is rebuilt every 20 minutes; poll its status.json and reload when a
  // newer build exists. Selection, sort, filters and axes are restored from localStorage.
  if (BUILT) setInterval(() => { if (!state.refresh) return; fetch('status.json', {cache: 'no-store'}).then(r => r.ok ? r.json() : null).then(st => { if (st && st.built_at && st.built_at !== BUILT) location.reload(); }).catch(() => {}); }, 60000);
})();
"""


def _scan_browser_html(records: list[dict[str, Any]], *, built_at: str = "") -> str:
    payload = scan_browser_payload(records)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    heads = "".join(f'<th data-key="{k}">{_h(label)}<span></span></th>' for k, label in _BROWSER_COLUMNS)
    js = (_BROWSER_JS.replace("%COLS%", json.dumps([[k, l] for k, l in _BROWSER_COLUMNS]))
          .replace("%BAR%", json.dumps(list(_BROWSER_BAR_COLUMNS)))
          .replace("%XCOLS%", json.dumps([[k, l or ""] for k, l in _BROWSER_X_COLUMNS]))
          .replace("%MON%", f"{MONITOR_REF:.0f}")
          .replace("%BUILT%", json.dumps(built_at or "")))
    return (
        f"<h2>All scans</h2><p class='muted'>Click a column header to sort; click a row to plot it, Ctrl/Cmd-click to add scans to the plot. "
        "Bars show where each scan's range sits within the range of all scans; hover for min, max and step. "
        f"Counts per {MONITOR_REF:.0f} monitor carry one-standard-deviation error bars. The table reloads itself when a newer build appears.</p>"
        "<div id='scan-browser'>"
        "<div class='controls'><label>family <select id='scan-browser-family'><option value=''>all</option></select></label> "
        "<label>filter <input id='scan-browser-text' type='search' placeholder='file, family, axis, number'></label> "
        "<label>x <select id='scan-browser-x'></select></label> "
        "<label>y <select id='scan-browser-y'></select></label> "
        "<label><input id='scan-browser-log' type='checkbox'> log y</label> "
        "<button id='scan-browser-clear' type='button'>newest only</button> "
        "<label><input id='scan-browser-refresh' type='checkbox' checked> auto-refresh</label> "
        "<span id='scan-browser-count' class='muted'></span></div>"
        "<div id='scan-browser-plot'></div>"
        f"<div class='table-wrap'><table id='scan-browser-table'><thead><tr>{heads}</tr></thead><tbody></tbody></table></div>"
        "</div>"
        f"<script id='scan-browser-data' type='application/json'>{data_json}</script>"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        f"<script>{js}</script>"
    )


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def _h(text: Any) -> str:
    return html_mod.escape(str(text))


def _img(src: Any, caption: str) -> str:
    if isinstance(src, dict):  # log/linear pair: show log, keep the other for the toggle
        return (f'<figure class="maptoggle"><img src="{src["log"]}" data-other="{src["linear"]}" data-mode="log" alt="{_h(caption)}">'
                f'<div class="figctl"><button type="button" onclick="toggleMap(this)">linear colour scale</button></div><figcaption>{_h(caption)}</figcaption></figure>')
    return f'<figure><img src="{src}" alt="{_h(caption)}"><figcaption>{_h(caption)}</figcaption></figure>'


CSS = """
body{font-family:Helvetica,Arial,sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#1a1a1a;background:#fff;line-height:1.45}
h1{font-size:1.45em;border-bottom:2px solid #b0641e;padding-bottom:6px} h2{font-size:1.2em;margin-top:2em;color:#b0641e} h3{font-size:1.05em}
figure{margin:14px 0} img{max-width:100%;border:1px solid #ddd} figcaption{font-size:.85em;color:#555}
table{border-collapse:collapse;font-size:.85em;margin:10px 0} th,td{border:1px solid #ccc;padding:3px 7px;text-align:left} th{background:#f3efe9}
.note{background:#faf8f5;border-left:4px solid #b0641e;padding:8px 12px;margin:12px 0} .warn{background:#fff4f0;border-left:4px solid #c0392b;padding:8px 12px;margin:12px 0}
code{background:#f2f2f2;padding:1px 4px} details summary{cursor:pointer;color:#b0641e} .muted{color:#666;font-size:.9em}
.figctl{margin:4px 0} .figctl button{font-size:.85em;padding:2px 8px;cursor:pointer}
#scan-browser .controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0} #scan-browser .controls label{font-size:.9em}
#scan-browser .table-wrap{max-height:520px;overflow:auto;border:1px solid #ddd} #scan-browser-table{margin:0;width:100%} #scan-browser-table th{position:sticky;top:0;cursor:pointer;white-space:nowrap}
#scan-browser-table th.sorted{background:#e9dfd0} #scan-browser-table td{white-space:nowrap} #scan-browser-table tbody tr{cursor:pointer} #scan-browser-table tr.selected td{background:#fdf1e2} #scan-browser-table tbody tr:hover td{background:#f7f3ee}
#scan-browser-plot{min-height:400px;border:1px solid #ddd;margin:8px 0}
#scan-browser-table td .bar{position:relative;display:inline-block;width:64px;height:9px;background:#eee;border:1px solid #ccc;vertical-align:middle;margin-right:4px}
#scan-browser-table td .bar .fill{position:absolute;top:0;bottom:0;background:#b0641e}
#scan-browser-table td .barlabel{font-size:.85em;color:#444} #scan-browser-table td .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle} #scan-browser-table td.c-name{max-width:260px;overflow:hidden;text-overflow:ellipsis}
"""


# --------------------------------------------------------------------------
# PSD two-axis diffraction: channels to sample two-theta
# --------------------------------------------------------------------------
MAX_PSD_PATTERNS = 20


def psd_scan_records(records: list[dict[str, Any]], *, min_points: int = 10) -> list[dict[str, Any]]:
    """A4 scans with PSD channel columns in a PSD analyzer mode, newest first."""
    out = [r for r in records if r.get("psd") is not None and r.get("variable") == "A4" and r["npts"] >= min_points
           and str(r.get("analyzer_mode", "")).upper().startswith("PSD")]
    return sorted(out, key=lambda r: r["number"], reverse=True)


def _psd_scan_payload(rec: dict[str, Any]) -> dict[str, Any]:
    return {"name": rec["name"], "path": rec["path"], "format": "nice", "a4": rec["A4"], "monitor": rec["mon"], "time": rec["time"],
            "temperature": rec["T"], "psd": rec["psd"], "header": rec["header"], "npts": rec["npts"], "date": rec["header"].get("date"),
            "analyzer_detector_translation": rec.get("anatrans"), "detector_mode": rec.get("analyzer_mode"), "nominal_a4": None}


def resolve_psd_calibration_file(name: str, search_dirs: list[Path]) -> Path | None:
    from tas_copilot_core.psd_calibration import default_psd_calibration_registry_path

    candidate = Path(str(name)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in [*search_dirs, default_psd_calibration_registry_path().parent, Path.cwd()]:
        if (root / candidate).exists():
            return root / candidate
    return None


def psd_patterns(records: list[dict[str, Any]], config: dict[str, Any], warnings: list[str], *, search_dirs: list[Path] | None = None,
                 monitor_reference: float = MONITOR_REF) -> dict[str, Any] | None:
    """Reduce the PSD A4 scans with the configured calibration.  The sign and
    scale of the channel offsets come from a Bragg-peak check on the newest
    scan with enough points (``config['scale']`` / ``config['sign_convention']``
    override it); every pattern then uses the same mapping."""
    from tas_copilot_core import psd_reduction as pr

    scans = psd_scan_records(records)
    if not scans:
        return None
    calibration_name = config.get("file") or config.get("calibration")
    if not calibration_name:
        warnings.append("PSD scans present but no PSD calibration configured (annotations psd_calibration.file)")
        return None
    path = resolve_psd_calibration_file(str(calibration_name), search_dirs or [])
    if path is None:
        warnings.append(f"PSD calibration file not found: {calibration_name}")
        return None
    calibration = pr.load_psd_calibration(path)
    bin_width = float(config.get("bin_width") or pr.DEFAULT_BIN_WIDTH)
    masked = [int(c) for c in (config.get("mask") or [])]
    check: dict[str, Any] | None = None
    for rec in sorted(scans, key=lambda r: (r["npts"], r["number"]), reverse=True)[:3]:
        try:
            result = pr.check_calibration_against_scan(_psd_scan_payload(rec), calibration)
        except Exception as exc:  # the check is a validator, never a reason to lose the patterns
            warnings.append(f"PSD calibration check on #{rec['number']}: {exc}")
            continue
        if result.get("status") == "ok":
            check = result
            check["scan_number"] = rec["number"]
            break
    sign = config.get("sign_convention") or (check["sign_convention"] if check else "minus_relative_a4")
    scale = float(config["scale"]) if config.get("scale") is not None else (float(check["scale"]) if check else 1.0)
    if check is None:
        warnings.append("PSD calibration sign/scale check found no Bragg peak to follow; using the calibration's offsets unscaled")
    # the temperature series: the trajectory with the most PSD scans (ties go to the newest scan's); findPeak-type
    # A4 scans with the PSD stay in the table but not in the series
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in scans:
        groups.setdefault(rec.get("trajectory") or rec["name"], []).append(rec)
    series_key = max(groups, key=lambda k: (len(groups[k]), max(r["number"] for r in groups[k])))
    series_recs = sorted(groups[series_key], key=_scan_temperature)
    chosen = [scans[0]] + [r for r in series_recs if r is not scans[0]]
    patterns: list[dict[str, Any]] = []
    for rec in chosen[:MAX_PSD_PATTERNS]:
        try:
            pattern = pr.reduce_psd_scan(_psd_scan_payload(rec), calibration, sign_convention=sign, scale=scale, bin_width=bin_width,
                                         masked_channels=masked, monitor_reference=monitor_reference)
        except Exception as exc:
            warnings.append(f"PSD reduction of #{rec['number']}: {exc}")
            continue
        patterns.append({"rec": rec, "pattern": pattern, "T": _scan_temperature(rec)})
    if not patterns:
        return None
    by_number = {item["rec"]["number"]: item for item in patterns}
    series = [by_number[r["number"]] for r in series_recs if r["number"] in by_number]
    patterns.sort(key=lambda item: item["rec"]["number"], reverse=True)
    return {"calibration": calibration, "calibration_path": str(path), "check": check, "sign_convention": sign, "scale": scale, "bin_width": bin_width,
            "masked": masked, "patterns": patterns, "scan_count": len(scans), "series": series, "series_trajectory": series_key}


def _scan_temperature(rec: dict[str, Any]) -> float:
    values = rec.get("T")
    if values is None or not np.isfinite(values).any():
        return float("inf")
    return float(np.nanmean(values))


def psd_series_grid(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patterns of one trajectory on a common two-theta grid: (two_theta centres,
    temperatures, intensity matrix with NaN where a scan has no bin)."""
    bw = items[0]["pattern"].bin_width_degrees
    lo = min(item["pattern"].two_theta[0] for item in items); hi = max(item["pattern"].two_theta[-1] for item in items)
    x = lo + bw * np.arange(int(round((hi - lo) / bw)) + 1)        # bin centres; every scan's bins share the same phase
    Z = np.full((len(items), len(x)), np.nan)
    for i, item in enumerate(items):
        idx = np.round((np.asarray(item["pattern"].two_theta) - lo) / bw).astype(int)
        ok = (idx >= 0) & (idx < len(x))
        Z[i, idx[ok]] = np.asarray(item["pattern"].intensity)[ok]
    return x, np.array([item["T"] for item in items]), Z


def _psd_series_map_figures(items: list[dict[str, Any]], trajectory: str) -> tuple[dict[str, str], str | None, dict[str, Any]]:
    """Intensity against two-theta and temperature (log/linear pair), and the
    difference from the warmest scan when there are at least two."""
    import matplotlib.pyplot as plt

    x, T, Z = psd_series_grid(items)
    finite = np.isfinite(T)
    labels = [f"{item['T']:.2f} K  #{item['rec']['number']}" if np.isfinite(item["T"]) else f"#{item['rec']['number']}" for item in items]
    # one equal-height row per scan, ordered by temperature: scans at nearly the same temperature would
    # otherwise collapse into zero-height rows
    y = np.arange(len(items), dtype=float)
    ye = np.arange(len(items) + 1, dtype=float) - 0.5
    xe = np.concatenate([x - (x[1] - x[0]) / 2, [x[-1] + (x[1] - x[0]) / 2]]) if len(x) > 1 else np.array([x[0] - 0.05, x[0] + 0.05])
    XE, YE = np.meshgrid(xe, ye)
    vmax = float(np.nanpercentile(Z, 99.5)); vmin = max(float(np.nanpercentile(Z, 1)), 0.5)

    def draw(kind: str):
        fig, ax = plt.subplots(figsize=(12, 2.4 + 0.32 * len(items)))
        pm = ax.pcolormesh(XE, YE, np.ma.masked_invalid(np.maximum(Z, 0.5)), shading="flat", norm=_map_norm(kind, vmin, vmax), cmap="viridis")
        fig.colorbar(pm, ax=ax, label=f"counts per {MONITOR_REF:.0f} monitor ({kind})", pad=0.02)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("sample two-theta (deg)"); ax.set_ylabel("scans ordered by temperature")
        ax.set_title(f"{trajectory}: {len(items)} PSD scans, {np.nanmin(T):.2f} to {np.nanmax(T):.2f} K" if finite.any() else trajectory, fontsize=10)
        return fig

    images = _map_pair(draw)
    difference = None
    if len(items) >= 2 and finite.all():
        warm = int(np.argmax(T))
        D = Z - Z[warm][None, :]
        lim = float(np.nanpercentile(np.abs(D), 99)) or 1.0

        fig, ax = plt.subplots(figsize=(12, 2.4 + 0.32 * len(items)))
        pm = ax.pcolormesh(XE, YE, np.ma.masked_invalid(D), shading="flat", cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(pm, ax=ax, label=f"counts per {MONITOR_REF:.0f} monitor, minus the warmest scan", pad=0.02)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("sample two-theta (deg)"); ax.set_ylabel("scans ordered by temperature")
        ax.set_title(f"difference from the warmest scan, #{items[warm]['rec']['number']} at {T[warm]:.2f} K (red: more intensity than at {T[warm]:.2f} K)", fontsize=10)
        difference = _fig_to_data_uri(fig)
    info = {"scans": len(items), "T_min": float(np.nanmin(T)) if finite.any() else None, "T_max": float(np.nanmax(T)) if finite.any() else None,
            "two_theta_range": [float(x[0]), float(x[-1])], "warmest_scan": int(items[int(np.argmax(T))]["rec"]["number"]) if finite.any() else None}
    return images, difference, info


def _psd_pattern_figure(item: dict[str, Any]) -> str:
    from tas_copilot_core import psd_reduction as pr

    rec, pattern = item["rec"], item["pattern"]
    temperature = f"{np.nanmean(rec['T']):.2f} K" if rec["T"] is not None and np.isfinite(rec["T"]).any() else "T unknown"
    fig = pr.pattern_figure(pattern, title=f"#{rec['number']} {rec['name'].replace('.dat.bt7', '')}, {temperature}, {rec['npts']} points, {rec['start']:%Y-%m-%d %H:%M}")
    return _fig_to_data_uri(fig)


def _psd_stack_figure(items: list[dict[str, Any]]) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.2 + 0.55 * len(items)))
    top = max(max(i["pattern"].intensity) for i in items) or 1.0
    step = 0.6 * top
    ordered = sorted(items, key=lambda i: (i["T"], i["rec"]["number"]))          # coolest at the bottom
    for k, item in enumerate(ordered):
        rec, pattern = item["rec"], item["pattern"]
        temperature = f"{item['T']:.2f} K" if np.isfinite(item["T"]) else ""
        ax.errorbar(pattern.two_theta, np.asarray(pattern.intensity) + k * step, yerr=pattern.error, fmt="-", lw=0.8, elinewidth=0.4,
                    color=PALETTE[k % len(PALETTE)], label=f"#{rec['number']} {temperature} ({rec['npts']} pts)")
        ax.text(pattern.two_theta[0], k * step + 0.05 * top, f"#{rec['number']} {temperature}", fontsize=8, color=PALETTE[k % len(PALETTE)], va="bottom")
    ax.set_xlabel("sample two-theta (deg)")
    ax.set_ylabel(f"counts per {MONITOR_REF:.0f} monitor, per channel; offset {step:.0f} per scan")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("PSD patterns offset by temperature, coolest at the bottom", fontsize=10)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _csv_link(text: str, filename: str, label: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"<a download=\"{_h(filename)}\" href=\"data:text/csv;base64,{encoded}\">{_h(label)}</a>"


def psd_section_html(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from tas_copilot_core import psd_reduction as pr

    calibration = result["calibration"]
    check = result.get("check")
    items = result["patterns"]
    parts = [f"<h2>PSD diffraction patterns ({result['scan_count']} PSD A4 scans; the {len(items)} newest reduced)</h2>"]
    policy = f"two_theta = A4 {'-' if result['sign_convention'] == 'minus_relative_a4' else '+'} {result['scale']:.4f} &times; relative_a4[channel]"
    parts.append(f"<p>Each of the 48 detector channels is placed at its own sample two-theta, {policy}, divided by its relative efficiency, and the (point, channel) counts are binned in {result['bin_width']:g}&deg; steps. "
                 f"Intensities are counts per {MONITOR_REF:.0f} monitor for a unit-efficiency channel; error bars are counting statistics of the bin totals."
                 f"{' Masked channels (zero based): ' + ', '.join(str(c) for c in result['masked']) + '.' if result['masked'] else ''}</p>")
    parts.append(f"<div class='note'><b>Calibration:</b> <code>{_h(calibration.calibration_id)}</code> (status {_h(calibration.status)}), per-channel offsets and efficiencies from the FindPeak scan "
                 f"<code>{_h(calibration.source_file)}</code>{(' of ' + _h(calibration.source_date[:10])) if calibration.source_date else ''}; file <code>{_h(Path(result['calibration_path']).name)}</code>. "
                 + (f"Sign and scale of the offsets checked on scan #{check['scan_number']}: {len([p for p in check['peaks'] if p['status'] == 'ok'])} Bragg peaks followed across the channels give "
                    f"scale {check['scale']:.4f} &plusmn; {check['scale_error']:.4f} (spread between peaks {check['scale_spread']:.4f}), largest residual rms {check['residual_rms_max']:.3f}&deg;; "
                    f"fitted peak two-theta: {', '.join(f'{v:.2f}' for v in check['two_theta_peaks'])}&deg;." if check else "No Bragg-peak check succeeded; offsets used unscaled.")
                 + " Two-theta only: no Q axis is drawn from this calibration.</div>")
    newest = items[0]
    parts.append(_img(_psd_pattern_figure(newest), f"Newest PSD scan #{newest['rec']['number']} reduced to sample two-theta (blue, left axis, with error bars); grey, right axis: the plain sum of all channels against A4, i.e. what the NICE Detector column shows."))
    series = result.get("series") or []
    series_info: dict[str, Any] = {}
    if series:
        series_images, difference, series_info = _psd_series_map_figures(series, result.get("series_trajectory", ""))
        series_info["trajectory"] = result.get("series_trajectory", "")
        parts.append(f"<h3>Temperature series: {_h(result.get('series_trajectory', ''))}, {len(series)} scan{'s' if len(series) != 1 else ''}"
                     + (f" from {series_info['T_min']:.2f} to {series_info['T_max']:.2f} K" if series_info.get("T_min") is not None else "") + "</h3>")
        parts.append(_img(series_images, "Intensity against two-theta and temperature, one row per scan (log colour; a scan still being measured is blank where it has not reached)."))
        parts.append(_img(_psd_stack_figure(series), "The same scans offset vertically, coolest at the bottom."))
        if difference:
            parts.append(_img(difference, "Each scan minus the warmest one so far: red where the cooler scan has more intensity. Magnetic peaks that order on cooling show as red stripes; this becomes meaningful once a scan above the transition exists."))
    rows = []
    for item in items:
        rec, pattern = item["rec"], item["pattern"]
        temperature = f"{np.nanmean(rec['T']):.2f}" if rec["T"] is not None and np.isfinite(rec["T"]).any() else ""
        rows.append(f"<tr><td>#{rec['number']}</td><td>{_h(rec['name'].replace('.dat.bt7', ''))}</td><td>{rec['start']:%m-%d %H:%M}</td><td>{temperature}</td><td>{rec['npts']}</td>"
                    f"<td>{pattern.two_theta[0]:.2f} to {pattern.two_theta[-1]:.2f}</td><td>{len(pattern.two_theta)}</td><td>{max(pattern.intensity):.0f}</td>"
                    f"<td>{_csv_link(pr.pattern_csv(pattern), pr.scan_stem(rec['name']) + '_psd_pattern.csv', 'CSV')}</td></tr>")
    parts.append("<table><tr><th>scan</th><th>file</th><th>start</th><th>T (K)</th><th>points</th><th>two-theta range (deg)</th><th>bins</th><th>max I</th><th>reduced data</th></tr>" + "".join(rows) + "</table>")
    status = {"psd_patterns": len(items), "psd_scans": result["scan_count"], "psd_calibration": {"id": calibration.calibration_id, "status": calibration.status, "file": Path(result["calibration_path"]).name,
              "sign_convention": result["sign_convention"], "scale": result["scale"], "bin_width": result["bin_width"], "masked_channels": result["masked"]},
              "psd_check": ({k: check[k] for k in ("scan_number", "scale", "scale_error", "scale_spread", "residual_rms_max", "two_theta_peaks")} if check else None),
              "psd_series": series_info or None}
    return "\n".join(parts), status


def _cadence(rec: dict[str, Any]) -> tuple[float, float, int, float] | None:
    """(start-to-start cadence, counting time, points with stamps, last stop) from a scan's per-point stamps."""
    start = rec.get("point_start"); stop = rec.get("point_stop")
    if start is None or stop is None:
        return None
    start = np.asarray(start, dtype=float); stop = np.asarray(stop, dtype=float)
    ok = np.isfinite(start) & np.isfinite(stop)
    if ok.sum() < 3:
        return None
    return float(np.nanmedian(np.diff(start[ok]))), float(np.nanmedian(stop[ok] - start[ok])), int(ok.sum()), float(stop[ok][-1])


def scan_eta(rec: dict[str, Any], *, now: datetime | None = None, reference: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Projected duration and finish of a scan from per-point stamps.

    NICE's ``trajectory.estimatedTime`` counts only the preset counting time;
    the measured start-to-start cadence also contains the dead time between
    points (motor settling, detector readout, file writing: 5.7 s per point
    for the BT-7 PSD A4 scans of 2026-11-02 at 10 s counting).  Projection:
    finish = file start + last stop + (planned - done) x median cadence.  A
    scan with fewer than three stamped points borrows the cadence of
    ``reference`` (an earlier scan of the same trajectory) and says so."""
    began = rec.get("start"); planned = rec.get("planned_points")
    if began is None:
        return None
    own = _cadence(rec)
    borrowed = False
    if own is None:
        ref = _cadence(reference) if reference else None
        if ref is None:
            return None
        cadence, counting = ref[0], ref[1]
        start = rec.get("point_start"); stop = rec.get("point_stop")
        done = int(np.isfinite(np.asarray(stop, dtype=float)).sum()) if stop is not None else 0
        elapsed = float(np.nanmax(np.asarray(stop, dtype=float))) if done else 0.0
        borrowed = True
    else:
        cadence, counting, done, elapsed = own
    planned = int(planned) if planned else done
    remaining = max(planned - done, 0)
    projected_total = elapsed + remaining * cadence
    finish = began + timedelta(seconds=projected_total)
    out = {"points_done": done, "points_planned": planned, "cadence_seconds": round(cadence, 2), "counting_seconds": round(counting, 2),
           "dead_seconds": round(cadence - counting, 2), "elapsed_seconds": round(elapsed), "projected_total_seconds": round(projected_total),
           "projected_finish": finish.isoformat(timespec="minutes"), "nice_estimated_seconds": rec.get("nice_estimated_seconds"), "complete": remaining == 0,
           "cadence_from": (f"#{reference['number']}" if borrowed and reference else "this scan")}
    if now is not None and remaining:
        out["remaining_minutes"] = round(max((finish - now).total_seconds(), 0) / 60)
    return out


def build_page(*, instrument_label: str, records: list[dict[str, Any]], hk: dict[int, dict[str, Any]], meta: dict[str, Any], discovered: dict[str, Any],
               events: list[tuple[str, str]], queue: dict[str, Any], annotations: dict[str, Any], data_as_of: str, warnings: list[str],
               found: list[dict[str, Any]] | None = None, psd_calibration: dict[str, Any] | None = None, calibration_dirs: list[Path] | None = None) -> tuple[str, dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")

    records = sorted([r for r in records if r], key=lambda r: r["number"])
    tz = records[-1]["start"].tzinfo if records and records[-1]["start"] and records[-1]["start"].tzinfo else timezone.utc
    now = datetime.now(tz)
    status: dict[str, Any] = {"instrument": instrument_label, **{k: v for k, v in discovered.items() if k != "instrument"}, "scan_count": len(records), "built_at": now.isoformat(timespec="seconds"), "data_as_of": data_as_of, "warnings": list(warnings)}
    sections: list[str] = []

    title = meta.get("title") or records[-1]["header"].get("experiment.title", "") if records else ""
    participants = meta.get("participant_names") or []
    if isinstance(participants, str):
        participants = [participants]
    first = records[0]["start"] if records and records[0]["start"] else None
    last = records[-1]["start"] if records and records[-1]["start"] else None
    sample = records[-1]["header"].get("sample", "") if records else ""
    cell = " ".join(records[-1]["header"].get(k, "") for k in ("sampleState.A",)) if records else ""
    head = [f"<h1>{_h(instrument_label)}: {_h(title) or 'current experiment'}</h1>"]
    head.append(f"<p>{_h(', '.join(participants))}{' &middot; ' if participants else ''}experiment {_h(discovered.get('experiment_id', '?'))}, cycle {_h(discovered.get('cycle', '?'))}"
                f"{' &middot; sample ' + _h(sample) if sample else ''} &middot; {len(records)} scans"
                f"{f' from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}' if first and last else ''} (instrument local time).<br>"
                f"<span class='muted'>Data read from the NCNR archive at {_h(data_as_of)}; page built {now:%Y-%m-%d %H:%M}. Rebuilt automatically every 20 minutes.</span></p>")
    if queue:
        head.append(f"<div class='note'><b>Queue:</b> running <code>{_h(queue.get('running') or 'nothing')}</code>; {queue.get('queued_count', 0)} commands queued"
                    + (": " + "; ".join(f"<code>{_h(c[:90])}</code>" for c in queue.get("next", [])) if queue.get("next") else "") + "</div>")
    try:
        reference = next((r for r in reversed(records[:-1]) if r.get("trajectory") == records[-1].get("trajectory") and _cadence(r)), None) if records else None
        eta = scan_eta(records[-1], now=now, reference=reference) if records else None
        if eta:
            status["latest_scan_eta"] = {"scan": records[-1]["number"], **eta}
            nice = f"; NICE's own estimate was {eta['nice_estimated_seconds'] / 60:.0f} min" if eta.get("nice_estimated_seconds") else ""
            if eta["complete"]:
                head.append(f"<div class='note'><b>Latest scan #{records[-1]['number']}:</b> complete, {eta['points_done']} points in {eta['elapsed_seconds'] / 60:.0f} min: "
                            f"{eta['counting_seconds']:.1f} s counting plus {eta['dead_seconds']:.1f} s dead time per point, {eta['cadence_seconds']:.1f} s cadence{nice}.</div>")
            else:
                head.append(f"<div class='note'><b>Scan #{records[-1]['number']} in progress:</b> {eta['points_done']} of {eta['points_planned']} points after {eta['elapsed_seconds'] / 60:.0f} min; "
                            f"cadence {eta['cadence_seconds']:.1f} s per point measured on {eta['cadence_from']} ({eta['counting_seconds']:.1f} s counting + {eta['dead_seconds']:.1f} s dead time), "
                            f"projected total {eta['projected_total_seconds'] / 60:.0f} min, finish about {eta['projected_finish'][11:16]} instrument time"
                            f"{(', ' + str(eta['remaining_minutes']) + ' min from the last archive update') if eta.get('remaining_minutes') is not None else ''}{nice}.</div>")
    except Exception as exc:
        warnings.append(f"scan ETA: {exc}")
    if annotations.get("notes"):
        head.append(f"<div class='note'><b>Notes (instrument scientist):</b> {_h(annotations['notes'])}</div>")
    if warnings:
        head.append("<div class='warn'><b>Build warnings:</b><ul>" + "".join(f"<li>{_h(w)}</li>" for w in warnings) + "</ul></div>")
    sections.append("\n".join(head))

    # families
    families: dict[str, int] = {}
    for rec in records:
        families[rec.get("family", "unknown")] = families.get(rec.get("family", "unknown"), 0) + 1
    # activity
    try:
        sections.append("<h2>Recent activity</h2>" + _img(_activity_figure(records, events, tz, now), "Scans started per hour and the sample temperature over the last 48 hours; red lines mark console CRITICAL/server events."))
        recent_events = [(s, m) for s, m in events if s >= (now - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")]
        if recent_events:
            sections.append("<details open><summary>Console events, last 48 h (" + str(len(recent_events)) + ")</summary><table><tr><th>time</th><th>message</th></tr>" + "".join(f"<tr><td>{_h(s[11:16])} {_h(s[5:10])}</td><td>{_h(m)}</td></tr>" for s, m in reversed(recent_events[-40:])) + "</table></details>")
    except Exception as exc:
        warnings.append(f"activity section: {exc}")

    # PSD two-axis diffraction patterns (only with a configured calibration)
    try:
        psd_config = dict(psd_calibration or annotations.get("psd_calibration") or {})
        if psd_config:
            psd_result = psd_patterns(records, psd_config, warnings, search_dirs=calibration_dirs or [])
            if psd_result:
                section, psd_status = psd_section_html(psd_result)
                sections.append(section)
                status.update(psd_status)
        elif psd_scan_records(records):
            status["psd_scans"] = len(psd_scan_records(records))
            sections.append("<h2>PSD diffraction patterns</h2><p class='muted'>PSD A4 scans are present, but no PSD calibration is configured for this experiment (annotations <code>psd_calibration.file</code>), so the channels are not placed in two-theta here.</p>")
    except Exception as exc:
        warnings.append(f"PSD section: {exc}")

    # order parameters
    order_parameter_html = ""; series_newest = -1; mesh_html = ""; peaks_html = ""; series: list[dict[str, Any]] = []
    try:
        meshes = detect_meshes(records)
        mesh_numbers = {r["number"] for m in meshes for r in m["rows"]}
        if "component_series" in annotations:
            from .component_series_view import component_series_section
            section, component_status = component_series_section(annotations["component_series"],
                str(discovered.get("experiment_id", "")), records, warnings,
                cycle=str(discovered["cycle"]) if discovered.get("cycle") is not None else None)
            status["component_series"] = component_status
            sections.append(section)
        else:
            series = detect_order_parameter_series(records, exclude=mesh_numbers)
            status["order_parameter_series"] = len(series)
            if series:
                fits = order_parameter_fits(annotations, warnings)
                status["order_parameter_fits"] = sorted(label for label in fits if label in {s["label"] for s in series})
                shown = series[:MAX_ORDER_PARAMETER_SERIES]
                parts = ["<h2>Order parameters (temperature series detected automatically)</h2>"]
                if len(shown) >= 2:
                    parts.append(f"<h3>{len(shown)} peaks followed in temperature, compared</h3>")
                    parts.extend(_img(img, cap) for img, cap in _order_parameter_comparison_figures(shown, fits))
                for s in shown:
                    fit = fits.get(s["label"])
                    images, info = _order_parameter_figures(s, fit)
                    change = 100 * (1 - info["area_last"] / info["area_first"]) if info["area_first"] else float("nan")
                    parts.append(f"<h3>{_h(s['label'])}: {_h(s.get('axis_label', s['variable']))} scans through {_h(s['variable'])} &asymp; {s['center']:.2f}&deg;, {_h(s['other'])} &asymp; {s['other_val']:.2f}&deg;</h3>")
                    parts.append(f"<p>{info['n']} scans from {info['T_min']:.1f} K to {info['T_max']:.1f} K. Integrated intensity goes from {info['area_first']:.0f} to {info['area_last']:.0f} ({change:+.0f}% change); centre drifts {info['center_drift']:+.3f}&deg;; FWHM {info['fwhm_range'][0]:.3f}-{info['fwhm_range'][1]:.3f}&deg;. Temperature within a scan is stable to {info['max_sd']*1000:.0f} mK (worst scan); largest offset from setpoint {info['max_offset']:.2f} K. Gaussian + flat background fits, no constraint between temperatures.</p>")
                    if fit:
                        details = "; ".join(f"{_h(k.replace('_', ' '))}: {_h(fit[k])}" for k in ("model", "fitter", "data", "Tc_scale", "fitted_at", "reviewed_by", "note") if fit.get(k))
                        parts.append(f"<div class='note'><b>Order-parameter fit (off-line; not computed by this page):</b> {_h(_fit_description(fit))}. {details}.</div>")
                    parts.extend(_img(img, cap) for img, cap in images)
                order_parameter_html = "\n".join(parts)
                series_newest = max(f["rec"]["number"] for s in shown for f in s["fitted"])
    except Exception as exc:
        warnings.append(f"order-parameter section: {exc}")
        meshes = []

    # peaks found by fitting, drawn on the maps as green circles
    found = found or []
    map_annotations = dict(annotations)
    map_annotations["markers"] = list(annotations.get("markers", []) or []) + [
        {"h": p["h"], "k": p["k"], "label": p["label"], "symbol": "o", "color": "#2ca02c"} for p in found]
    status["found_peaks"] = [{k: v for k, v in p.items() if k in ("h", "k", "l", "label", "scan", "T", "amp", "fwhm", "axis")} for p in found]
    if found:
        rows_found = "".join(f"<tr><td>{_h(p['label'])}</td><td>{p['h']:.3f}, {p['k']:.3f}, {p['l']:.2f}</td><td>#{p['scan']} {_h(p['name'].replace('.dat.bt7', ''))}</td><td>{_h(p['axis'])} = {p['center']:.3f}&deg;, FWHM {p['fwhm']:.3f}&deg;</td><td>{p['amp']:.0f}</td><td>{p['T']:.2f}</td><td>{_h(p['origin'])}</td></tr>" for p in found)
        peaks_html = ("<h2>Peaks found</h2><p>Fitted centres of the order-parameter series and of the most recent findPeak scans, placed in reciprocal space with NICE's own H,K and labelled with the nearest simple fraction. These are measured positions, not assignments; green circles on the maps below.</p>"
                        f"<table><tr><th>nearest (H K L)</th><th>H, K, L</th><th>scan</th><th>centre</th><th>height / {MONITOR_REF:.0f} mon</th><th>T (K)</th><th>from</th></tr>{rows_found}</table>")

    # meshes
    try:
        status["meshes"] = len(meshes)
        if meshes:
            parts = ["<h2>Meshes (NICE mesh files, and meshes assembled from stepped scans)</h2>"]
            for mesh in meshes[:4]:
                guides = mesh_guides(mesh, series)
                images, info = _mesh_figures(mesh, hk, map_annotations, guides)
                rows = mesh["rows"]
                parts.append(f"<h3>{_h(mesh['trajectory'])}: {info['rows']} rows in {_h(mesh['other'])} ({mesh['second_axis'][0]:.2f} to {mesh['second_axis'][-1]:.2f}&deg;), {_h(mesh['variable'])} {min(float(np.nanmin(scan_axis_values(r))) for r in rows):.2f} to {max(float(np.nanmax(scan_axis_values(r))) for r in rows):.2f}&deg;, {min(r['start'] for r in rows):%m-%d %H:%M} to {max(r['start'] for r in rows):%m-%d %H:%M}{(', PG filter ' + mesh['filter']) if mesh.get('filter') else ''}</h3>")
                progress = f" One NICE mesh file; {info['complete_rows']} of {info['rows']} rows complete, {info['points']} points so far; rows not yet measured are blank." if mesh.get("single_file") and info["complete_rows"] < info["rows"] else (" One NICE mesh file." if mesh.get("single_file") else "")
                parts.append(f"<p>T = {info['T_mean']:.3f} K (sd {info['T_sd']:.3f}); background about {info['background']:.0f} and maximum {info['max']:.0f} counts per {MONITOR_REF:.0f} monitor.{progress}" + ("" if info["hk"] else " Reciprocal-space map not drawn: NeXus H,K not available for every row.") + "</p>")
                guide_text = (" Dashed red lines: peak positions found by the temperature series (" + ", ".join(info["guides"]) + "); crosses mark where those scans crossed each peak.") if info.get("guides") else ""
                caps = ["Mesh in instrument angles (log colour)." + guide_text, "The same mesh in reciprocal space using the H,K NICE wrote to the NeXus files.", "Each row as a line scan, background subtracted and offset."] if info["hk"] else ["Mesh in instrument angles (log colour)." + guide_text, "Each row as a line scan, background subtracted and offset."]
                parts.extend(_img(img, cap) for img, cap in zip(images, caps))
            mesh_html = "\n".join(parts)
    except Exception as exc:
        warnings.append(f"mesh section: {exc}")
    # the newer activity first: a mesh being measured now goes above the temperature series
    mesh_newest = max((r["number"] for m in meshes for r in m["rows"]), default=-1)
    ordered = [mesh_html, order_parameter_html, peaks_html] if mesh_newest > series_newest else [order_parameter_html, peaks_html, mesh_html]
    sections.extend(part for part in ordered if part)

    # latest scans
    try:
        sections.append("<h2>Latest scans</h2>" + _img(_gallery_figure(records), "The eight most recent scans, newest first."))
    except Exception as exc:
        warnings.append(f"gallery: {exc}")

    # scan browser (sortable table + linked plot)
    try:
        sections.append(_scan_browser_html(records, built_at=status["built_at"]))
    except Exception as exc:
        warnings.append(f"scan table: {exc}")

    sections.append("<h2>Method notes</h2><ul><li>Intensities are detector counts per monitor, rescaled to " + f"{MONITOR_REF:.0f}" + " monitor counts; errors are Poisson.</li>"
                    "<li>Meshes: consecutive scans of one trajectory with an identical grid and a stepped second angle, split by temperature. H,K come from NICE's own conversion in the NeXus files (refplane1/refplane2, unit cell, a3Zero at scan time).</li>"
                    "<li>Automatic temperature series, when no reviewed component view is configured: repeated scans of one axis through one place while temperature changed by more than 10 K. The scan with the strongest signal is the reference: its peaks (local maxima well above the counting noise) are fitted freely as Gaussians plus a flat background and define the components. Every scan is then fitted with those centres and widths fixed and only the amplitudes and background free (a linear least-squares problem, so a vanished peak gives an area of zero within its error), and refitted with centre and width free within bounds where a component is still clearly present, for the centre-shift and width panels. Reviewed component views instead display their supplied model, covariance and provenance.</li>"
                    "<li>PSD patterns, when a calibration is configured: every channel is placed at A4 minus (scale times its FindPeak offset) and divided by its relative efficiency; counts are binned in two-theta and the bin uncertainty is the counting error of the bin total (tasreduction). The sign and scale come from following Bragg peaks across the channels in the newest scan, and are shown with the calibration's status. No Q axis is drawn from a non-accepted calibration.</li>"
                    "<li>Scan families follow the tascopilot experiment-dashboard classifier. This page is read-only evidence for the instrument team, not a reduction.</li></ul>")
    status["families"] = families
    if warnings:
        status["warnings"] = list(warnings)
    page = f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='robots' content='noindex,nofollow'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{_h(instrument_label)} current experiment: {_h(title)[:80]}</title><style>{CSS}</style><script>function toggleMap(btn){{const img=btn.closest('figure').querySelector('img');const other=img.dataset.other;img.dataset.other=img.src;img.src=other;img.dataset.mode=img.dataset.mode==='log'?'linear':'log';btn.textContent=img.dataset.mode==='log'?'linear colour scale':'log colour scale';}}</script></head><body>" + "\n".join(sections) + "</body></html>"
    return page, status


# --------------------------------------------------------------------------
# workflow entry point
# --------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text()) or {}


def load_annotations(path: Path, experiment_id: str, warnings: list[str]) -> dict[str, Any]:
    """Hand-written notes/markers apply to ONE experiment: the file must name
    the experiment id it was written for, otherwise it is ignored (so notes for
    a finished experiment never appear on the next one)."""
    if not Path(path).exists():
        return {}
    try:
        annotations = _load_yaml(Path(path)) or {}
    except Exception as exc:
        warnings.append(f"annotations: {exc}")
        return {}
    declared = str(annotations.get("experiment_id", "")).strip()
    if not declared:
        warnings.append(f"annotations file {Path(path).name} ignored: it does not declare experiment_id")
        return {}
    if declared != str(experiment_id):
        warnings.append(f"annotations file {Path(path).name} is for experiment {declared}, current is {experiment_id}; ignored")
        return {}
    if "component_series" in annotations:
        from .component_series_view import load_component_series_payload
        annotations["component_series"] = load_component_series_payload(annotations["component_series"], Path(path), warnings)
    return annotations


def _load_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(Path(data_dir) / "*.dat.bt7"))):
        rec = read_nice_scan(Path(path))
        if not rec or rec["start"] is None:
            continue
        try:
            rec["family"] = _overnight_family(_parse_scan_file(Path(path), max_plot_points=40))
        except Exception:
            rec["family"] = "unclassified_scan_family"
        records.append(rec)
    return records


def _build_instrument(entry: dict[str, Any], config: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    from tas_copilot_core.ncnr_archive import NCNRArchive, reset_cache_if_experiment_changed

    name = str(entry.get("name", "bt7"))
    label = str(entry.get("label", name.upper()))
    html_root = Path(arguments.get("output_dir") or config.get("html_root", "html"))
    cache_root = Path(config.get("cache_root", "cache"))
    annotations_root = Path(config.get("annotations_root", "annotations"))
    out_dir = html_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    offline = arguments.get("offline_data_root")
    hk: dict[int, dict[str, Any]] = {}
    if offline:
        data_dir = Path(offline)
        discovered = {"instrument": name, "cycle": data_dir.parent.parent.name, "experiment_id": data_dir.parent.name, "latest_data_mtime": "offline"}
        console_txt = data_dir.parent / "console" / "console_history.txt"
        queue_path = data_dir.parent / "queue" / "queue"
        meta: dict[str, Any] = {}
        data_as_of = datetime.now().strftime("%Y-%m-%d %H:%M") + " (offline copy)"
    else:
        archive = NCNRArchive(config.get("archive_base", "https://ncnr.nist.gov/pub/ncnrdata"), metadata_api=config.get("metadata_api", "https://ncnr.nist.gov/ncnrdata/metadata/api/v1/experiments.php"))
        if entry.get("experiment_id"):
            cycles = archive.cycles(name)
            cycle = str(entry.get("cycle") or (cycles[0] if cycles else ""))
            discovered = {"instrument": name, "cycle": cycle, "experiment_id": str(entry["experiment_id"]), "latest_data_mtime": archive.newest_data_mtime(name, cycle, str(entry["experiment_id"]))}
        else:
            discovered = archive.discover_current_experiment(name)
        cache_dir = cache_root / name
        if reset_cache_if_experiment_changed(cache_dir, discovered):
            warnings.append("experiment changed; cache reset")
        sync = archive.sync_files(name, discovered["cycle"], discovered["experiment_id"], cache_dir, include=lambda n: n.endswith(".dat.bt7"))
        if sync["failed"]:
            warnings.append(f"{len(sync['failed'])} downloads failed")
        try:
            console = archive.sync_console(name, discovered["cycle"], discovered["experiment_id"], cache_dir)
            console_txt = console["text_path"]
        except Exception as exc:
            warnings.append(f"console log: {exc}")
            console_txt = cache_dir / "console_history.txt"
        queue_path = archive.fetch_queue(name, discovered["cycle"], discovered["experiment_id"], cache_dir)
        meta = archive.experiment_metadata(name, discovered["experiment_id"])
        data_dir = cache_dir / "data"
        data_as_of = discovered.get("latest_data_mtime", "")
    records = _load_records(data_dir)
    if not records:
        raise RuntimeError(f"no readable scans in {data_dir}")
    # NeXus twins only where H,K are needed: mesh rows and the scans whose fitted peaks go on the maps
    found: list[dict[str, Any]] = []
    try:
        meshes = detect_meshes(records)
        mesh_numbers = {r["number"] for m in meshes for r in m["rows"]}
        series = detect_order_parameter_series(records, exclude=mesh_numbers)
        representatives = peak_representatives(records, series)
        need = [r for m in meshes[:4] for r in m["rows"]] + [item["rec"] for item in representatives]
        wanted = sorted({r["name"].replace(".dat.bt7", ".nxs.bt7") for r in need})
        if wanted and not offline:
            archive.fetch_named(name, discovered["cycle"], discovered["experiment_id"], cache_root / name, wanted)
        for r in need:
            nxs = Path(data_dir) / r["name"].replace(".dat.bt7", ".nxs.bt7")
            if nxs.exists() and r["number"] not in hk:
                info = read_nexus_hk(nxs)
                if info and len(info["H"]) == r["npts"]:
                    hk[r["number"]] = info
        found = locate_peaks(representatives, hk)
    except Exception as exc:
        warnings.append(f"nexus H,K: {exc}")
    annotations = load_annotations(annotations_root / f"{name}.yaml", str(discovered.get("experiment_id", "")), warnings)
    events = console_events(Path(console_txt) if console_txt else None, since=None)
    queue = queue_status(Path(queue_path) if queue_path else None)
    psd_config = annotations.get("psd_calibration") or entry.get("psd_calibration")
    page, status = build_page(instrument_label=label, records=records, hk=hk, meta=meta, discovered=discovered, events=events, queue=queue, annotations=annotations, data_as_of=str(data_as_of), warnings=warnings, found=found,
                              psd_calibration=psd_config, calibration_dirs=[annotations_root, Path(config.get("psd_calibration_root", annotations_root))])
    tmp = out_dir / "index.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out_dir / "index.html")
    (out_dir / "status.json").write_text(json.dumps(status, indent=1, default=str))
    return {"instrument": name, "label": label, "status": status, "page": str(out_dir / "index.html")}


def live_experiment_page(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = dict(arguments or {})
    config: dict[str, Any] = {}
    if arguments.get("config"):
        config = _load_yaml(Path(arguments["config"]))
    instruments = config.get("instruments") or [{"name": arguments.get("instrument", "bt7"), "label": arguments.get("instrument", "bt7").upper()}]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in instruments:
        try:
            results.append(_build_instrument(entry, config, arguments))
        except Exception as exc:  # a failed build leaves the previous page in place
            errors.append(f"{entry.get('name')}: {exc}")
            html_root = Path(arguments.get("output_dir") or config.get("html_root", "html"))
            (html_root / str(entry.get("name", "bt7"))).mkdir(parents=True, exist_ok=True)
            (html_root / str(entry.get("name", "bt7")) / "status.json").write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()[-2000:], "failed_at": datetime.now().isoformat(timespec="seconds")}, indent=1))
    html_root = Path(arguments.get("output_dir") or config.get("html_root", "html"))
    html_root.mkdir(parents=True, exist_ok=True)
    items = "".join(f"<li><a href='{_h(r['instrument'])}/'>{_h(r['label'])}</a>: {_h(r['status'].get('experiment_id', ''))} &middot; {r['status'].get('scan_count', 0)} scans &middot; data as of {_h(r['status'].get('data_as_of', ''))} <span class='muted'>(<a href='{_h(r['instrument'])}/status.json'>status.json</a>)</span></li>" for r in results)
    (html_root / "index.html").write_text(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='robots' content='noindex,nofollow'><title>Current experiment summaries</title><style>{CSS}</style></head><body><h1>Current experiment summaries</h1><p>Static pages rebuilt from the public NCNR data archive for the experiment running on each instrument. Convenience view for the instrument team; not an archive and not a substitute for the data files.</p><ul>{items}</ul>" + ("<div class='warn'>" + "; ".join(_h(e) for e in errors) + "</div>" if errors else "") + "</body></html>", encoding="utf-8")
    if not (html_root / "robots.txt").exists():
        (html_root / "robots.txt").write_text("User-agent: *\nDisallow: /summary/\n")
    return {"status": "ok" if results and not errors else ("partial" if results else "error"), "results": results, "errors": errors, "html_root": str(html_root)}


__all__ = ["live_experiment_page", "read_nice_scan", "scan_eta", "psd_scan_records", "psd_series_grid", "psd_patterns", "psd_section_html", "resolve_psd_calibration_file", "detect_meshes", "detect_order_parameter_series", "fit_peak", "console_events", "queue_status", "build_page", "load_annotations", "order_parameter_fits", "nearest_fraction", "hk_label", "peak_representatives", "locate_peaks"]
