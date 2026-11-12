"""BT-7 PSD diffraction reduction: detector channels to a 2-theta pattern.

A BT-7 position-sensitive detector (PSD) scan in two-axis diffraction mode
records 48 channel counts at every A4 step.  A channel ``c`` sees scattering
angle ``A4 - scale * relative_a4[c]`` where ``relative_a4`` is the per-channel
offset measured in a FindPeak A4 scan through one Bragg peak (the A4 at which
the channel sees the peak, minus the scan's nominal A4; the convention of the
historical ``PSD_*.dat`` calibration files) and ``scale`` absorbs a change of
detector distance between the calibration and the data.  Each channel also
has a relative efficiency, the peak area it recorded in that FindPeak scan
divided by the mean area of the reference channels.

This module is deterministic: Gaussian fits per channel, a least-squares
check of sign and scale against the Bragg peaks of a data scan, and a binned
pattern whose uncertainties are counting statistics propagated by the
``tasreduction`` boundary (:mod:`tas_copilot_core.counting`).  It converts
channels to sample two-theta only; Q conversion stays behind the reviewed
calibration gate in :mod:`tas_copilot_core.psd_calibration`.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tas_copilot_contracts import PSDChannelCalibration, PSDChannelCalibrationRow, PSDDiffractionPattern, ProvenanceRecord

from tas_copilot_core.counting import rate_with_uncertainty

PSD_CHANNEL_COUNT = 48
DEFAULT_REFERENCE_CHANNELS = tuple(range(18, 28))
DEFAULT_BIN_WIDTH = 0.1
DEFAULT_MIN_EFFICIENCY = 0.25
SIGN_CONVENTIONS = {"minus_relative_a4": -1, "plus_relative_a4": 1}


# --------------------------------------------------------------------------
# scan files (ICE 0.4 ``#Columns`` with PSDC00..47, NICE 0.2 ``## pt`` with PSD01..48)
# --------------------------------------------------------------------------
def _float_or_nan(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")


def read_psd_scan(path: str | Path) -> dict[str, Any]:
    """Read a BT-7 scan with PSD channel columns in either file format.

    Returns ``a4``, ``monitor``, ``time``, ``temperature`` (arrays over points),
    ``psd`` (points x 48 counts), ``header``, ``format``, ``date``,
    ``nominal_a4`` (the scan centre the file declares, or None),
    ``analyzer_detector_translation`` (NICE ``anaTrans(cm)``, or None) and
    ``detector_mode``.  Raises ``ValueError`` when the file has no PSD columns.
    """
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        rest = handle.read()
    text = first + rest
    if first.startswith("#ICE") or "#Columns" in text:
        scan = _read_ice(text)
    elif first.startswith("# NICE") or "## pt" in text:
        scan = _read_nice(text)
    else:
        raise ValueError(f"{source.name}: not an ICE or NICE BT-7 scan file")
    scan["path"] = str(source)
    scan["name"] = source.name
    if scan["psd"] is None:
        raise ValueError(f"{source.name}: no PSD channel columns (PSDC00.. or PSD01..)")
    return scan


def _read_ice(text: str) -> dict[str, Any]:
    header: dict[str, str] = {}
    columns: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        if line.startswith("#Columns"):
            columns = line.split()[1:]
        elif line.startswith("#"):
            key, _, value = line[1:].partition(" ")
            header[key.strip()] = value.strip()
        elif columns and line.strip():
            parts = line.split()
            rows.append([_float_or_nan(p) for p in parts[: len(columns)]] + [float("nan")] * max(0, len(columns) - len(parts)))
    data = {name: np.array([r[i] for r in rows], dtype=float) for i, name in enumerate(columns)} if rows else {}
    psd_names = [f"PSDC{i:02d}" for i in range(PSD_CHANNEL_COUNT)]
    psd = np.column_stack([data[n] for n in psd_names]) if all(n in data for n in psd_names) else None
    nominal = None
    match = re.search(r"Range=A4=([-\d.]+)\s", header.get("ScanDescr", "") + " ")
    if match:
        nominal = float(match.group(1))
    return {
        "format": "ice",
        "header": header,
        "date": header.get("Date"),
        "a4": data.get("A4"),
        "monitor": data.get("Monitor"),
        "time": data.get("Time"),
        "temperature": data.get("Temp"),
        "psd": psd,
        "nominal_a4": nominal,
        "analyzer_detector_translation": None,
        "detector_mode": header.get("AnalyzerDetectorMode"),
        "experiment": header.get("ExptName"),
        "details": header.get("ExptDetails"),
        "npts": len(rows),
    }


def _read_nice(text: str) -> dict[str, Any]:
    header: dict[str, str] = {}
    columns: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        if line.startswith("## pt"):
            columns = line[2:].split()
        elif line.startswith("#"):
            key, _, value = line[1:].strip().partition(" ")
            if key:
                header[key] = value.strip()
        elif columns and line.strip():
            parts = line.split()
            rows.append([_float_or_nan(p) for p in parts[: len(columns)]] + [float("nan")] * max(0, len(columns) - len(parts)))
    data = {name: np.array([r[i] for r in rows], dtype=float) for i, name in enumerate(columns)} if rows else {}

    def col(*names: str) -> np.ndarray | None:
        for name in names:
            for key, values in data.items():
                if key == name or key.split("(")[0] == name:
                    return values
        return None

    psd_names = [f"PSD{i + 1:02d}" for i in range(PSD_CHANNEL_COUNT)]
    psd = np.column_stack([data[n] for n in psd_names]) if all(n in data for n in psd_names) else None
    anatrans = col("anaTrans")
    nominal = None
    match = re.search(r'"A4"\s*,\s*\{\s*"range"\s*:\s*\{\s*"center"\s*:\s*"([-\d.]+)"', header.get("trajectory.config", ""))
    if match:
        nominal = float(match.group(1))
    return {
        "format": "nice",
        "header": header,
        "date": header.get("date"),
        "a4": col("A4"),
        "monitor": col("monitor"),
        "time": col("time"),
        "temperature": col("temp"),
        "psd": psd,
        "nominal_a4": nominal,
        "analyzer_detector_translation": float(np.nanmedian(anatrans)) if anatrans is not None and np.isfinite(anatrans).any() else None,
        "detector_mode": header.get("analyzerMode"),
        "experiment": header.get("experiment.proposalId"),
        "details": header.get("experiment.title"),
        "npts": len(rows),
    }


# --------------------------------------------------------------------------
# calibration from a FindPeak A4 scan
# --------------------------------------------------------------------------
def _gaussian(x: np.ndarray, amp: float, center: float, sigma: float, background: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background


def fit_channel_peak(x: np.ndarray, y: np.ndarray, *, max_sigma: float | None = None) -> dict[str, float] | None:
    """Gaussian plus flat background on one channel's counts versus A4.
    Returns centre, its error, sigma, amplitude, background and area, or None."""
    from scipy.optimize import curve_fit

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.ptp(x) <= 0:
        return None
    err = np.sqrt(np.maximum(y, 1.0))
    step = float(np.median(np.abs(np.diff(np.sort(x))))) or 1e-3
    b0 = float(np.median(y))
    i = int(np.argmax(y))
    a0 = max(float(y[i] - b0), 1.0)
    upper_sigma = max_sigma or max(np.ptp(x) / 2, 2 * step)
    try:
        popt, pcov = curve_fit(_gaussian, x, y, p0=[a0, x[i], min(0.3, upper_sigma / 2), b0], sigma=err, absolute_sigma=True,
                               bounds=([0, x.min(), 0.5 * step, 0], [np.inf, x.max(), upper_sigma, np.inf]), maxfev=10000)
    except Exception:
        return None
    amp, center, sigma, background = (float(v) for v in popt)
    dcenter = float(math.sqrt(abs(pcov[1, 1]))) if np.isfinite(pcov[1, 1]) else float("nan")
    return {"amp": amp, "center": center, "dcenter": dcenter, "sigma": sigma, "background": background, "area": amp * sigma * math.sqrt(2 * math.pi)}


def derive_psd_calibration(path: str | Path, *, calibration_id: str | None = None, reference_channels: tuple[int, ...] = DEFAULT_REFERENCE_CHANNELS,
                           nominal_a4: float | None = None, analyzer_detector_translation: float | None = None, min_efficiency: float = DEFAULT_MIN_EFFICIENCY,
                           notes: list[str] | None = None) -> PSDChannelCalibration:
    """Per-channel efficiency and A4 offset from a FindPeak A4 scan through one
    Bragg peak (an ICE ``fpx`` file or a NICE A4 scan with PSD columns).

    ``relative_a4[c]`` is the fitted centre of channel ``c`` minus the scan's
    nominal A4 (``nominal_a4`` overrides the file's declared centre); the
    efficiency is the fitted area divided by the mean area of the reference
    channels.  Channels whose fit fails or whose efficiency is below
    ``min_efficiency`` are flagged masked.
    """
    scan = read_psd_scan(path)
    a4 = scan["a4"]
    if a4 is None:
        raise ValueError(f"{scan['name']}: no A4 column")
    nominal = nominal_a4 if nominal_a4 is not None else scan["nominal_a4"]
    if nominal is None:
        nominal = float(np.nanmean(a4))
    fits = [fit_channel_peak(a4, scan["psd"][:, c]) for c in range(PSD_CHANNEL_COUNT)]
    areas = np.array([f["area"] if f else float("nan") for f in fits])
    ref = [c for c in reference_channels if c < PSD_CHANNEL_COUNT and np.isfinite(areas[c])]
    if not ref:
        raise ValueError("no reference channel fitted; cannot normalise efficiencies")
    reference_area = float(np.mean(areas[ref]))
    rows: list[PSDChannelCalibrationRow] = []
    for c, fit in enumerate(fits):
        if fit is None:
            rows.append(PSDChannelCalibrationRow(channel=c, efficiency=0.0, relative_a4=0.0, masked=True))
            continue
        efficiency = fit["area"] / reference_area
        rows.append(PSDChannelCalibrationRow(channel=c, efficiency=efficiency, relative_a4=fit["center"] - nominal, center_a4=fit["center"],
                                             dcenter=fit["dcenter"], sigma=fit["sigma"], area=fit["area"], masked=bool(efficiency < min_efficiency)))
    stamp = (scan.get("date") or "").strip()
    ident = calibration_id or f"bt7-psd-{scan_stem(Path(path).name)}-{_date_token(stamp)}"
    pitch = float(np.nanmedian(np.diff([r.relative_a4 for r in rows if not r.masked]))) if sum(not r.masked for r in rows) > 2 else float("nan")
    return PSDChannelCalibration(
        calibration_id=ident,
        source_file=scan["name"],
        source_path=str(scan["path"]),
        source_date=stamp or None,
        source_format=scan["format"],
        method="gaussian_plus_background_per_channel; relative_a4 = centre - nominal_a4; efficiency = area / mean(area[reference_channels])",
        nominal_a4=float(nominal),
        reference_channels=list(ref),
        analyzer_detector_translation=analyzer_detector_translation,
        channel_pitch_degrees=pitch,
        channels=rows,
        status="proposed",
        notes=list(notes or []),
        provenance=[ProvenanceRecord(source_type="bt7_findpeak_scan", source_id=scan["name"], source_uri=str(scan["path"]), method="psd_reduction.derive_psd_calibration",
                                     metadata={"points": scan["npts"], "experiment": scan.get("experiment"), "details": scan.get("details"), "detector_mode": scan.get("detector_mode")})],
    )


def scan_stem(name: str) -> str:
    """``sample_A4_4.0567K_1461.dat.bt7`` -> ``sample_A4_4.0567K_1461``."""
    stem = Path(name).name
    for suffix in (".dat.bt7", ".nxs.bt7", ".bt7", ".dat"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _date_token(stamp: str) -> str:
    digits = re.sub(r"\D", "", stamp[:10])
    return digits or "undated"


def write_psd_calibration(calibration: PSDChannelCalibration, output_dir: str | Path, *, stem: str | None = None) -> dict[str, str]:
    """JSON (full record) plus the legacy ``#Efficiency A4 Ef`` text format that
    the historical BT-7 PSD tools and ``psd_calibration.discover_psd_dat_calibrations`` read."""
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or calibration.calibration_id
    json_path = out / f"{stem}.json"
    dat_path = out / f"{stem}.dat"
    json_path.write_text(json.dumps(calibration.model_dump(mode="json"), indent=1), encoding="utf-8")
    lines = [f"#BT7 PSD calibration based on {calibration.source_file}", f"#Created on: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S')}",
             f"#Calibration id: {calibration.calibration_id}; status {calibration.status}; tascopilot psd_reduction.derive_psd_calibration"]
    masked = [r.channel + 1 for r in calibration.channels if r.masked]
    if masked:
        lines.append("#Channels to be masked: " + ",".join(str(c) for c in masked))
    lines.append("#Efficiency  A4  Ef")
    lines.extend(f"{r.efficiency:.6g} {r.relative_a4:.6g} 0" for r in calibration.channels)
    dat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "dat": str(dat_path)}


def load_psd_calibration(path: str | Path) -> PSDChannelCalibration:
    """A calibration written by :func:`write_psd_calibration` (JSON) or a legacy
    ``PSD_*.dat`` file (efficiency, relative A4 per row; masked channels from its header)."""
    source = Path(path).expanduser()
    if source.suffix.lower() == ".json":
        return PSDChannelCalibration.model_validate(json.loads(source.read_text(encoding="utf-8")))
    from tas_copilot_core.psd_calibration import _parse_psd_dat

    parsed = _parse_psd_dat(source, read_contents=True)
    masked = set(parsed.get("masked_channels") or [])
    rows = [PSDChannelCalibrationRow(channel=row["channel"], efficiency=row["efficiency"], relative_a4=row["relative_a4"], masked=(row["channel"] + 1) in masked)
            for row in parsed.get("channels") or []]
    if not rows:
        raise ValueError(f"{source.name}: no calibration rows")
    return PSDChannelCalibration(calibration_id=f"legacy-{source.stem}", source_file=str(parsed.get("source_reference") or source.name), source_path=str(source),
                                 source_date=parsed.get("created_date"), source_format="legacy_dat", method="legacy PSD .dat file (efficiency, relative A4)",
                                 nominal_a4=None, reference_channels=[], analyzer_detector_translation=parsed.get("analyzer_detector_translation"),
                                 channel_pitch_degrees=float(np.median(np.diff([r.relative_a4 for r in rows]))) if len(rows) > 2 else float("nan"), channels=rows,
                                 status="candidate_unreviewed", notes=[], provenance=[ProvenanceRecord(source_type="psd_calibration_dat", source_id=source.name, source_uri=str(source), method="psd_reduction.load_psd_calibration")])


# --------------------------------------------------------------------------
# deterministic check of sign and scale on a data scan
# --------------------------------------------------------------------------
def check_calibration_against_scan(scan: dict[str, Any], calibration: PSDChannelCalibration, *, max_peaks: int = 6, window: float = 0.9,
                                   min_separation: float = 2.5, min_amplitude: float = 30.0, max_center_error: float = 0.08) -> dict[str, Any]:
    """Fit, for the strongest Bragg peaks seen by the central channels, the A4
    at which every channel sees the peak, and regress it on ``relative_a4``:
    the slope's sign gives the convention (positive slope means
    ``two_theta = A4 - scale * relative_a4``) and its magnitude the scale
    (detector distance change since the calibration).  Outlier channels are
    dropped iteratively (3 sigma, at least 0.05 deg) and reported."""
    a4 = scan["a4"]
    psd = scan["psd"]
    offsets = np.array([r.relative_a4 for r in calibration.channels])
    masked = np.array([r.masked for r in calibration.channels])
    if a4 is None or psd is None or len(a4) < 10:
        return {"status": "missing_input", "missing_fields": ["a4_scan_with_psd_columns"], "peaks": []}
    step = float(np.median(np.abs(np.diff(a4)))) or 0.1
    central = psd[:, 20:28].sum(axis=1)
    order = np.argsort(central)[::-1]
    seeds: list[int] = []
    for i in order:
        if central[i] <= 0:
            break
        if all(abs(a4[i] - a4[j]) > min_separation for j in seeds):
            seeds.append(int(i))
        if len(seeds) >= max_peaks:
            break
    peaks: list[dict[str, Any]] = []
    for seed in seeds:
        a4_seed = float(a4[seed])
        centers = np.full(PSD_CHANNEL_COUNT, np.nan)
        errors = np.full(PSD_CHANNEL_COUNT, np.nan)
        for c in range(PSD_CHANNEL_COUNT):
            if masked[c]:
                continue
            expected = a4_seed + offsets[c]
            sel = (a4 > expected - window) & (a4 < expected + window)
            if sel.sum() < 5:
                continue
            fit = fit_channel_peak(a4[sel], psd[sel, c], max_sigma=window)
            if fit and fit["amp"] >= min_amplitude and fit["dcenter"] <= max_center_error and (a4[sel].min() + step) < fit["center"] < (a4[sel].max() - step):
                centers[c] = fit["center"]
                errors[c] = max(fit["dcenter"], 1e-3)
        ok = np.isfinite(centers)
        if ok.sum() < 8:
            peaks.append({"a4_seed": a4_seed, "status": "too_few_channels", "channels_fitted": int(ok.sum())})
            continue
        excluded: list[int] = []
        residual = np.full(PSD_CHANNEL_COUNT, np.nan)
        coef = np.zeros(2)
        for _ in range(5):
            design = np.vstack([np.ones(ok.sum()), offsets[ok]]).T
            coef = np.linalg.lstsq(design, centers[ok], rcond=None)[0]
            residual[:] = np.nan
            residual[ok] = centers[ok] - design @ coef
            limit = max(3 * float(np.nanstd(residual[ok])), 0.05)
            bad = ok & (np.abs(residual) > limit)
            if not bad.any():
                break
            excluded.extend(int(c) for c in np.where(bad)[0])
            ok &= ~bad
        weights = 1.0 / errors[ok] ** 2
        design = np.vstack([np.ones(ok.sum()), offsets[ok]]).T
        cov = np.linalg.inv(design.T @ (design * weights[:, None]))
        coef = cov @ (design.T @ (weights * centers[ok]))
        peaks.append({"a4_seed": a4_seed, "status": "ok", "two_theta": float(coef[0]), "slope": float(coef[1]), "slope_error": float(math.sqrt(abs(cov[1, 1]))),
                      "residual_rms": float(np.nanstd(residual[ok])), "channels_fitted": int(ok.sum()), "excluded_channels": sorted(excluded),
                      "channel_centers": [None if not np.isfinite(v) else round(float(v), 4) for v in centers]})
    good = [p for p in peaks if p["status"] == "ok"]
    if not good:
        return {"status": "no_peak_fitted", "peaks": peaks, "warnings": ["no Bragg peak could be followed across the channels; sign and scale not determined"]}
    slopes = np.array([p["slope"] for p in good])
    slope_errors = np.array([p["slope_error"] for p in good])
    w = 1.0 / np.maximum(slope_errors, 1e-4) ** 2
    slope = float(np.sum(w * slopes) / np.sum(w))
    sign_convention = "minus_relative_a4" if slope > 0 else "plus_relative_a4"
    scale = abs(slope)
    return {
        "status": "ok",
        "sign_convention": sign_convention,
        "channel_to_two_theta_policy": ("two_theta = A4 - scale * relative_a4[channel]" if slope > 0 else "two_theta = A4 + scale * relative_a4[channel]"),
        "scale": scale,
        "scale_error": float(1.0 / math.sqrt(np.sum(w))),
        "scale_spread": float(np.std(np.abs(slopes))) if len(slopes) > 1 else 0.0,
        "residual_rms_max": float(max(p["residual_rms"] for p in good)),
        "peaks": peaks,
        "two_theta_peaks": [round(p["two_theta"], 3) for p in good],
        "scan": scan.get("name"),
        "calibration_id": calibration.calibration_id,
        "warnings": ([] if abs(scale - 1.0) < 0.05 else [f"scale {scale:.3f} differs from 1 by more than 5%: the detector distance differs from the calibration's; check anaTrans"]),
    }


# --------------------------------------------------------------------------
# reduction to I(two-theta)
# --------------------------------------------------------------------------
def reduce_psd_scan(scan: dict[str, Any], calibration: PSDChannelCalibration, *, sign_convention: str = "minus_relative_a4", scale: float = 1.0,
                    bin_width: float = DEFAULT_BIN_WIDTH, masked_channels: tuple[int, ...] | list[int] = (), monitor_reference: float | None = None,
                    min_efficiency: float = DEFAULT_MIN_EFFICIENCY, two_theta_range: tuple[float, float] | None = None) -> PSDDiffractionPattern:
    """Bin every (point, channel) count into sample two-theta.

    Bin intensity is total counts over total exposure, the exposure of one
    contribution being ``monitor * efficiency / monitor_reference``, so the
    result is counts per ``monitor_reference`` monitor counts for a unit
    efficiency channel; the uncertainty is the counting error of the bin
    total (floor of one count) propagated through tasreduction.  Masked
    channels (``masked_channels``, zero based, plus the calibration's own
    masks and any channel below ``min_efficiency``) contribute nothing.
    """
    if sign_convention not in SIGN_CONVENTIONS:
        raise ValueError(f"sign_convention must be one of {sorted(SIGN_CONVENTIONS)}")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    a4 = np.asarray(scan["a4"], dtype=float)
    psd = np.asarray(scan["psd"], dtype=float)
    monitor = np.asarray(scan["monitor"], dtype=float) if scan.get("monitor") is not None else np.ones_like(a4)
    if psd.shape[1] != len(calibration.channels):
        raise ValueError(f"scan has {psd.shape[1]} channels, calibration {len(calibration.channels)}")
    reference = float(monitor_reference) if monitor_reference else float(np.nanmedian(monitor))
    sign = SIGN_CONVENTIONS[sign_convention]
    offsets = np.array([r.relative_a4 for r in calibration.channels])
    efficiency = np.array([r.efficiency for r in calibration.channels])
    mask = np.array([r.masked for r in calibration.channels]) | ~np.isfinite(efficiency) | (efficiency < min_efficiency)
    for c in masked_channels:
        if 0 <= int(c) < len(mask):
            mask[int(c)] = True
    warnings: list[str] = []
    good_points = np.isfinite(a4) & np.isfinite(monitor) & (monitor > 0)
    if not good_points.all():
        warnings.append(f"{int((~good_points).sum())} points without A4 or monitor dropped")
    a4 = a4[good_points]; psd = psd[good_points]; monitor = monitor[good_points]
    two_theta = a4[:, None] + sign * scale * offsets[None, :]            # points x channels
    exposure = monitor[:, None] * efficiency[None, :] / reference
    counts = np.where(np.isfinite(psd), psd, 0.0)
    keep = (~mask)[None, :] & np.isfinite(psd)
    tt = two_theta[keep]; n = counts[keep]; w = exposure[keep]
    if two_theta_range is not None:
        inside = (tt >= two_theta_range[0]) & (tt <= two_theta_range[1])
        tt, n, w = tt[inside], n[inside], w[inside]
    if tt.size == 0:
        raise ValueError("no unmasked contributions to bin")
    lo = math.floor(tt.min() / bin_width) * bin_width
    hi = math.ceil(tt.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + 0.5 * bin_width, bin_width)
    index = np.clip(np.digitize(tt, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(index, weights=n, minlength=len(edges) - 1)
    expo = np.bincount(index, weights=w, minlength=len(edges) - 1)
    contributions = np.bincount(index, minlength=len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    filled = expo > 0
    intensity, error = rate_with_uncertainty(total[filled], expo[filled], target=1.0)
    # uncalibrated comparison trace: all channels summed, per point
    summed = counts.sum(axis=1)
    sum_intensity, sum_error = rate_with_uncertainty(summed, monitor, target=reference)
    return PSDDiffractionPattern(
        source_file=scan.get("name", ""),
        source_path=scan.get("path"),
        calibration_id=calibration.calibration_id,
        calibration_status=calibration.status,
        sign_convention=sign_convention,
        channel_to_two_theta_policy=f"two_theta = A4 {'-' if sign < 0 else '+'} {scale:.4f} * relative_a4[channel]",
        scale=float(scale),
        bin_width_degrees=float(bin_width),
        monitor_reference=reference,
        two_theta=[round(float(v), 5) for v in centers[filled]],
        intensity=[float(v) for v in intensity],
        error=[float(v) for v in error],
        counts=[float(v) for v in total[filled]],
        exposure=[float(v) for v in expo[filled]],
        contributions=[int(v) for v in contributions[filled]],
        a4=[float(v) for v in a4],
        a4_sum_intensity=[float(v) for v in sum_intensity],
        a4_sum_error=[float(v) for v in sum_error],
        masked_channels=[int(c) for c in np.where(mask)[0]],
        temperature_mean=float(np.nanmean(scan["temperature"])) if scan.get("temperature") is not None and np.isfinite(scan["temperature"]).any() else None,
        points=int(len(a4)),
        warnings=warnings,
        provenance=[ProvenanceRecord(source_type="bt7_psd_scan", source_id=scan.get("name"), source_uri=scan.get("path"), method="psd_reduction.reduce_psd_scan",
                                     metadata={"calibration_id": calibration.calibration_id, "calibration_source": calibration.source_file, "sign_convention": sign_convention,
                                               "scale": float(scale), "bin_width": float(bin_width), "monitor_reference": reference, "format": scan.get("format"),
                                               "analyzer_detector_translation": scan.get("analyzer_detector_translation")}),
                    *calibration.provenance],
    )


def pattern_csv(pattern: PSDDiffractionPattern) -> str:
    lines = [f"# {pattern.source_file}: PSD reduced to sample two-theta with calibration {pattern.calibration_id} ({pattern.calibration_status}); {pattern.channel_to_two_theta_policy}",
             f"# intensity = counts per {pattern.monitor_reference:.0f} monitor counts, efficiency corrected; error = counting statistics (one sigma); bin {pattern.bin_width_degrees} deg",
             "two_theta_deg,intensity,error,counts,exposure,contributions"]
    for row in zip(pattern.two_theta, pattern.intensity, pattern.error, pattern.counts, pattern.exposure, pattern.contributions):
        lines.append(f"{row[0]:.4f},{row[1]:.5g},{row[2]:.5g},{row[3]:.0f},{row[4]:.5g},{row[5]}")
    return "\n".join(lines) + "\n"


def pattern_figure(pattern: PSDDiffractionPattern, *, title: str | None = None, show_sum: bool = True) -> Any:
    """Matplotlib figure: I(two-theta) with error bars; the uncalibrated sum of
    all channels against A4 in grey on a second axis for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    handles = [ax.errorbar(pattern.two_theta, pattern.intensity, yerr=pattern.error, fmt="o-", ms=2.2, lw=0.7, elinewidth=0.6, color="#1f77b4",
                           label=f"PSD reduced to two-theta, {pattern.bin_width_degrees:g} deg bins (left axis)")]
    ax.set_xlabel("sample two-theta (deg)")
    ax.set_ylabel(f"counts per {pattern.monitor_reference:.0f} monitor, per channel (efficiency corrected)")
    ax.set_ylim(bottom=0)
    if show_sum and pattern.a4:
        twin = ax.twinx()
        handles.append(twin.plot(pattern.a4, pattern.a4_sum_intensity, color="#999999", lw=0.8, label="all 48 channels summed, against A4, uncalibrated (right axis)")[0])
        twin.set_ylabel(f"summed channels per {pattern.monitor_reference:.0f} monitor", color="#777777")
        twin.set_ylim(bottom=0)
        twin.tick_params(axis="y", colors="#777777")
    ax.set_title(title or f"{pattern.source_file}: calibration {pattern.calibration_id} ({pattern.calibration_status}), {pattern.channel_to_two_theta_policy}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def psd_diffraction_pattern_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    """CLI entry: reduce one scan with a calibration file; optional sign/scale
    check against the scan itself; writes CSV, JSON and PNG under output_dir."""
    data_file = arguments["data_file"]
    scan = read_psd_scan(data_file)
    calibration = load_psd_calibration(arguments["calibration"])
    check = None
    sign_convention = arguments.get("sign_convention")
    scale = arguments.get("scale")
    if arguments.get("check", True):
        check = check_calibration_against_scan(scan, calibration)
        if check.get("status") == "ok":
            sign_convention = sign_convention or check["sign_convention"]
            scale = scale if scale is not None else check["scale"]
    sign_convention = sign_convention or "minus_relative_a4"
    scale = float(scale) if scale is not None else 1.0
    pattern = reduce_psd_scan(scan, calibration, sign_convention=sign_convention, scale=scale, bin_width=float(arguments.get("bin_width") or DEFAULT_BIN_WIDTH),
                              masked_channels=tuple(int(c) for c in arguments.get("masked_channels") or ()), monitor_reference=arguments.get("monitor_reference"))
    payload: dict[str, Any] = {"status": "ok", "pattern": pattern.model_dump(mode="json"), "check": check,
                               "safe_conclusion": ("Channels were mapped to sample two-theta with a calibration of status "
                                                   f"'{calibration.status}'; Q conversion remains behind the reviewed PSD calibration gate.")}
    if arguments.get("output_dir"):
        out = Path(arguments["output_dir"]).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        stem = scan_stem(Path(data_file).name)
        (out / f"{stem}_psd_pattern.csv").write_text(pattern_csv(pattern), encoding="utf-8")
        (out / f"{stem}_psd_pattern.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        try:
            fig = pattern_figure(pattern)
            fig.savefig(out / f"{stem}_psd_pattern.png", dpi=110)
        except Exception as exc:  # matplotlib is an optional extra
            payload.setdefault("warnings", []).append(f"plot not written: {exc}")
        payload["artifacts"] = {"csv": str(out / f"{stem}_psd_pattern.csv"), "json": str(out / f"{stem}_psd_pattern.json"), "png": str(out / f"{stem}_psd_pattern.png")}
    return payload


def derive_psd_calibration_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    calibration = derive_psd_calibration(arguments["findpeak_file"], calibration_id=arguments.get("calibration_id"),
                                         nominal_a4=arguments.get("nominal_a4"), analyzer_detector_translation=arguments.get("analyzer_detector_translation"),
                                         notes=arguments.get("notes"))
    payload: dict[str, Any] = {"status": "ok", "calibration": calibration.model_dump(mode="json"),
                               "safe_conclusion": "Derived calibration has status 'proposed': it maps channels to sample two-theta only after a scan-based sign/scale check and does not unlock Q x temperature reconstruction."}
    if arguments.get("output_dir"):
        payload["artifacts"] = write_psd_calibration(calibration, arguments["output_dir"], stem=arguments.get("stem"))
    return payload


__all__ = ["read_psd_scan", "scan_stem", "fit_channel_peak", "derive_psd_calibration", "write_psd_calibration", "load_psd_calibration", "check_calibration_against_scan",
           "reduce_psd_scan", "pattern_csv", "pattern_figure", "psd_diffraction_pattern_workflow", "derive_psd_calibration_workflow", "PSD_CHANNEL_COUNT", "DEFAULT_BIN_WIDTH"]
