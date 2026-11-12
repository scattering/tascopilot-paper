from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from tas_copilot_contracts import AuditRecord, ProvenanceRecord, ToolCall, ToolResultStatus
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_runner import stable_hash
from tas_copilot_core.workflows.common import default_runner

WORKFLOW_NAME = "assess-figure-feature"


def assess_figure_feature(arguments: dict[str, Any], runner=None, audit_logger: JsonlAuditLogger | None = None) -> dict[str, Any]:
    data_root = arguments.get("data_root") or arguments.get("path")
    output_dir = Path(arguments.get("output_dir") or CoreConfig().root / "data" / "processed" / "feature_reality").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = audit_logger or JsonlAuditLogger(output_dir / "audit")
    runner = runner or default_runner()
    if not data_root:
        return {"workflow_name": WORKFLOW_NAME, "status": "missing_input", "errors": [{"code": "missing_data_root", "message": "data_root is required"}]}
    data_root = Path(data_root).expanduser()
    patterns = arguments.get("patterns") or ["*.bt7"]
    include_terms = [str(term).lower() for term in arguments.get("include_terms", ["LuFeMnO3_Tdep", "LuMnFeO3", "LuFe"])]
    q_features = [float(item) for item in arguments.get("q_features", [1.24])]
    temperature_features = [float(item) for item in arguments.get("temperature_features", [55.0, 130.0])]
    q_tolerance = float(arguments.get("q_tolerance", 0.04))
    sideband_width = float(arguments.get("sideband_width", q_tolerance))
    sideband_gap = float(arguments.get("sideband_gap", q_tolerance))
    temperature_tolerance = float(arguments.get("temperature_tolerance", 8.0))
    max_files = int(arguments.get("max_files", 300))

    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_root.glob(pattern)))
    files = [path for path in files if path.is_file() and (not include_terms or any(term in path.name.lower() for term in include_terms))][:max_files]

    line_scans = []
    psd_maps = []
    tool_errors = []
    audit_ids = []
    for path in files:
        result = runner.run(ToolCall(tool_namespace="tas.data_reader", tool_name="read_bt7_file", arguments={"path": str(path)}, caller_workflow=WORKFLOW_NAME))
        if result.audit_id:
            audit_ids.append(result.audit_id)
        if result.status != ToolResultStatus.success:
            tool_errors.append({"path": str(path), "status": result.status.value if hasattr(result.status, "value") else str(result.status)})
            continue
        output = result.output if isinstance(result.output, dict) else {}
        metadata = output.get("metadata") or {}
        line_scan = output.get("line_scan")
        if isinstance(line_scan, dict):
            line_scans.append(_line_scan_record(path, metadata, line_scan, q_features, q_tolerance, sideband_width, sideband_gap))
        psd_map = output.get("psd_map")
        if isinstance(psd_map, dict):
            psd_maps.append(_psd_record(path, metadata, psd_map))

    feature_checks = _feature_checks(line_scans, q_features, temperature_features, temperature_tolerance)
    artifacts = _write_artifacts(output_dir, line_scans, psd_maps, feature_checks)
    conclusion = _safe_conclusion(line_scans, psd_maps, feature_checks)
    payload = {
        "workflow_name": WORKFLOW_NAME,
        "status": "success" if line_scans or psd_maps else "missing_input",
        "data_root": str(data_root),
        "files_considered": len(files),
        "line_scan_count": len(line_scans),
        "psd_map_count": len(psd_maps),
        "line_scan_modes": sorted({scan.get("detector_mode") for scan in line_scans if scan.get("detector_mode")}),
        "psd_modes": sorted({scan.get("detector_mode") for scan in psd_maps if scan.get("detector_mode")}),
        "feature_checks": feature_checks,
        "tool_errors": tool_errors,
        "safe_conclusion": conclusion,
        "artifacts": artifacts,
        "audit_ids": audit_ids,
        "provenance": [ProvenanceRecord(source_type="data_directory", source_id=str(data_root), method="registry_line_scan_feature_triage").model_dump(mode="json")],
    }
    audit = AuditRecord(event_type="workflow", status=payload["status"], input_hash=stable_hash(arguments), output_hash=stable_hash(payload), details={"workflow_name": WORKFLOW_NAME, "tools_selected": ["tas.data_reader.read_bt7_file"]})
    audit_logger.write(audit)
    payload["workflow_audit_id"] = audit.audit_id
    return payload


def _line_scan_record(path: Path, metadata: dict[str, Any], line_scan: dict[str, Any], q_features: list[float], q_tolerance: float, sideband_width: float, sideband_gap: float) -> dict[str, Any]:
    q_axis = [float(item) for item in line_scan.get("q_axis") or []]
    intensity = [float(item) if item is not None else math.nan for item in line_scan.get("intensity") or []]
    finite = [item for item in intensity if item == item]
    med = median(finite) if finite else None
    points = [
        {"q": q, "intensity": intensity[index]}
        for index, q in enumerate(q_axis)
        if index < len(intensity) and intensity[index] == intensity[index]
    ]
    near = {}
    for q0 in q_features:
        vals = [intensity[index] for index, q in enumerate(q_axis) if index < len(intensity) and abs(q - q0) <= q_tolerance and intensity[index] == intensity[index]]
        sidebands = _sideband_values(q_axis, intensity, q0, sideband_width, sideband_gap)
        center_mean = sum(vals) / len(vals) if vals else None
        sideband_mean = sum(sidebands) / len(sidebands) if sidebands else None
        near[str(q0)] = {
            "mean_intensity": center_mean,
            "n_points": len(vals),
            "relative_to_scan_median": (center_mean / med) if center_mean is not None and med else None,
            "sideband_mean_intensity": sideband_mean,
            "sideband_n_points": len(sidebands),
            "center_minus_sideband": (center_mean - sideband_mean) if center_mean is not None and sideband_mean is not None else None,
            "center_over_sideband": (center_mean / sideband_mean) if center_mean is not None and sideband_mean else None,
            "background_subtracted": False,
        }
    return {
        "path": str(path),
        "file": path.name,
        "date": metadata.get("date_str"),
        "detector_mode": line_scan.get("detector_mode") or metadata.get("analyzer_detector_mode"),
        "temperature_K": line_scan.get("temperature_K"),
        "energy_transfer_meV": line_scan.get("energy_transfer_meV"),
        "q_min": min(q_axis) if q_axis else None,
        "q_max": max(q_axis) if q_axis else None,
        "point_count": len(q_axis),
        "intensity_source": line_scan.get("intensity_source"),
        "median_intensity": med,
        "background_subtracted": False,
        "background_model": "none",
        "feature_q_checks": near,
        "points": points,
    }


def _sideband_values(q_axis: list[float], intensity: list[float], q0: float, width: float, gap: float) -> list[float]:
    lower_min = q0 - gap - width
    lower_max = q0 - gap
    upper_min = q0 + gap
    upper_max = q0 + gap + width
    return [
        intensity[index]
        for index, q in enumerate(q_axis)
        if index < len(intensity)
        and intensity[index] == intensity[index]
        and ((lower_min <= q <= lower_max) or (upper_min <= q <= upper_max))
    ]


def _psd_record(path: Path, metadata: dict[str, Any], psd_map: dict[str, Any]) -> dict[str, Any]:
    two_theta = psd_map.get("two_theta_axis")
    temperature_axis = psd_map.get("temperature_axis") or []
    intensity = psd_map.get("intensity") or []
    return {
        "path": str(path),
        "file": path.name,
        "date": metadata.get("date_str"),
        "detector_mode": metadata.get("analyzer_detector_mode"),
        "status": psd_map.get("status"),
        "psd_channel_axis": psd_map.get("psd_channel_axis") or [],
        "two_theta_axis": two_theta,
        "temperature_axis": temperature_axis,
        "temperature_min": min(temperature_axis) if temperature_axis else None,
        "temperature_max": max(temperature_axis) if temperature_axis else None,
        "intensity": intensity,
        "intensity_source": psd_map.get("intensity_source"),
        "shape": [len(intensity), len(intensity[0]) if intensity else 0],
        "q_axis_present": psd_map.get("q_axis") is not None,
        "warnings": psd_map.get("warnings", []),
    }


def _feature_checks(line_scans: list[dict[str, Any]], q_features: list[float], temperature_features: list[float], temperature_tolerance: float) -> list[dict[str, Any]]:
    checks = []
    for t0 in temperature_features:
        nearby = [scan for scan in line_scans if scan.get("temperature_K") is not None and abs(float(scan["temperature_K"]) - t0) <= temperature_tolerance]
        for q0 in q_features:
            with_q = [scan for scan in nearby if scan["feature_q_checks"].get(str(q0), {}).get("n_points", 0) > 0]
            sideband_ratios = [
                scan["feature_q_checks"][str(q0)]["center_over_sideband"]
                for scan in with_q
                if scan["feature_q_checks"][str(q0)]["center_over_sideband"] is not None
            ]
            checks.append({
                "temperature_K": t0,
                "q": q0,
                "nearby_line_scan_count": len(nearby),
                "line_scans_with_q_points": len(with_q),
                "files": [scan["file"] for scan in with_q],
                "strongest_relative_to_scan_median": max([scan["feature_q_checks"][str(q0)]["relative_to_scan_median"] for scan in with_q if scan["feature_q_checks"][str(q0)]["relative_to_scan_median"] is not None], default=None),
                "strongest_center_over_sideband": max(sideband_ratios, default=None),
                "background_subtracted": False,
                "background_status": "not_subtracted",
            })
    return checks


def _write_artifacts(output_dir: Path, line_scans, psd_maps, feature_checks) -> dict[str, str]:
    line_json = output_dir / "line_scan_feature_summary.json"
    psd_json = output_dir / "psd_map_summary.json"
    checks_json = output_dir / "feature_checks.json"
    line_json.write_text(json.dumps(line_scans, indent=2, sort_keys=True), encoding="utf-8")
    psd_json.write_text(json.dumps(psd_maps, indent=2, sort_keys=True), encoding="utf-8")
    checks_json.write_text(json.dumps(feature_checks, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "line_scan_feature_summary.csv"
    if line_scans:
        keys = ["file", "date", "detector_mode", "temperature_K", "energy_transfer_meV", "q_min", "q_max", "point_count", "intensity_source", "median_intensity", "background_subtracted", "background_model"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for scan in line_scans:
                writer.writerow({key: scan.get(key) for key in keys})
    md_path = output_dir / "feature_reality_report.md"
    plot_paths = _write_matplotlib_plots(output_dir, line_scans, psd_maps, feature_checks)
    md_path.write_text(_markdown(line_scans, psd_maps, feature_checks, plot_paths), encoding="utf-8")
    return {
        "line_scan_json": str(line_json),
        "psd_json": str(psd_json),
        "feature_checks_json": str(checks_json),
        "line_scan_csv": str(csv_path),
        "report_markdown": str(md_path),
        **plot_paths,
    }


def _write_matplotlib_plots(output_dir: Path, line_scans, psd_maps, feature_checks) -> dict[str, str]:
    if not line_scans and not psd_maps:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting dependency.
        path = output_dir / "plot_error.txt"
        path.write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return {"plot_error": str(path)}

    paths: dict[str, str] = {}
    q_values = sorted({float(check["q"]) for check in feature_checks})
    temperature_marks = sorted({float(check["temperature_K"]) for check in feature_checks})
    by_energy: dict[float, list[dict[str, Any]]] = {}
    for scan in line_scans:
        energy = scan.get("energy_transfer_meV")
        if energy is None or not scan.get("points"):
            continue
        by_energy.setdefault(round(float(energy), 3), []).append(scan)

    for energy, scans in sorted(by_energy.items()):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for scan in sorted(scans, key=lambda item: float(item.get("temperature_K") or 0.0)):
            points = sorted(scan.get("points") or [], key=lambda item: item["q"])
            ax.plot(
                [point["q"] for point in points],
                [point["intensity"] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=_temperature_label(scan),
            )
        for index, q0 in enumerate(q_values):
            ax.axvline(q0, color="crimson", linestyle="--", linewidth=1, label=f"Q={q0:g}" if index == 0 else None)
        ax.set_title(f"Raw monitor-normalized Q scans, E={energy:g} meV")
        ax.set_xlabel("Q (1/Angstrom)")
        ax.set_ylabel("Detector / Monitor")
        if q_values:
            all_q = [point["q"] for scan in scans for point in scan.get("points", [])]
            if all_q:
                low = max(min(all_q), min(q_values) - 0.25)
                high = min(max(all_q), max(q_values) + 0.25)
                if low < high:
                    ax.set_xlim(low, high)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
        path = output_dir / f"raw_q_scans_E{_safe_token(energy)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"raw_q_scans_E{_safe_token(energy)}_png"] = str(path)

    for q0 in q_values:
        temp_path = _write_temperature_cut_plot(plt, output_dir, line_scans, q0, temperature_marks)
        if temp_path:
            paths[f"temperature_cut_Q{_safe_token(q0)}_png"] = str(temp_path)
        ratio_path = _write_sideband_ratio_plot(plt, output_dir, line_scans, q0, temperature_marks)
        if ratio_path:
            paths[f"sideband_ratio_Q{_safe_token(q0)}_png"] = str(ratio_path)
    for psd in psd_maps:
        psd_path = _write_psd_map_plot(plt, output_dir, psd)
        if psd_path:
            paths[f"psd_map_{_safe_filename_token(psd['file'])}_png"] = str(psd_path)
    return paths


def _write_psd_map_plot(plt, output_dir: Path, psd: dict[str, Any]) -> Path | None:
    intensity = psd.get("intensity") or []
    temperature_axis = psd.get("temperature_axis") or []
    if not intensity or not temperature_axis:
        return None
    x_axis = psd.get("psd_channel_axis") or list(range(1, len(intensity[0]) + 1))
    x_label = "PSD channel"
    two_theta = psd.get("two_theta_axis")
    title_suffix = ""
    if isinstance(two_theta, dict) and two_theta.get("type") == "constant":
        title_suffix = f", A4={float(two_theta['value']):g} deg"
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    extent = [min(x_axis), max(x_axis), min(temperature_axis), max(temperature_axis)]
    image = ax.imshow(intensity, aspect="auto", origin="lower", extent=extent, interpolation="nearest")
    ax.set_title(f"Elastic PSD temperature map\n{psd['file']}{title_suffix}", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Temperature (K)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(psd.get("intensity_source") or "Intensity")
    warnings = psd.get("warnings") or []
    if warnings:
        ax.text(0.01, 0.01, "No Q axis: PSD calibration required", transform=ax.transAxes, fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 3})
    path = output_dir / f"psd_map_{_safe_filename_token(psd['file'])}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_temperature_cut_plot(plt, output_dir: Path, line_scans: list[dict[str, Any]], q0: float, temperature_marks: list[float]) -> Path | None:
    series = _feature_series(line_scans, q0, "mean_intensity")
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for energy, points in series.items():
        ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=1.4, label=f"E={energy:g} meV")
    for temp in temperature_marks:
        ax.axvline(temp, color="0.35", linestyle="--", linewidth=1)
    ax.set_title(f"Temperature dependence near Q={q0:g}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean Detector / Monitor near Q")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / f"temperature_cut_Q{_safe_token(q0)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_sideband_ratio_plot(plt, output_dir: Path, line_scans: list[dict[str, Any]], q0: float, temperature_marks: list[float]) -> Path | None:
    series = _feature_series(line_scans, q0, "center_over_sideband")
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for energy, points in series.items():
        ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=1.4, label=f"E={energy:g} meV")
    for temp in temperature_marks:
        ax.axvline(temp, color="0.35", linestyle="--", linewidth=1)
    ax.axhline(1.0, color="0.2", linestyle=":", linewidth=1)
    ax.set_title(f"Center / sideband ratio near Q={q0:g}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Center / sideband intensity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / f"sideband_ratio_Q{_safe_token(q0)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _feature_series(line_scans: list[dict[str, Any]], q0: float, field: str) -> dict[float, list[tuple[float, float]]]:
    series: dict[float, list[tuple[float, float]]] = {}
    for scan in line_scans:
        temperature = scan.get("temperature_K")
        energy = scan.get("energy_transfer_meV")
        if temperature is None or energy is None:
            continue
        value = scan.get("feature_q_checks", {}).get(str(q0), {}).get(field)
        if value is None:
            continue
        series.setdefault(round(float(energy), 3), []).append((float(temperature), float(value)))
    return {energy: sorted(points) for energy, points in sorted(series.items()) if points}


def _temperature_label(scan: dict[str, Any]) -> str:
    temperature = scan.get("temperature_K")
    if temperature is None:
        return scan.get("file", "unknown")
    return f"{float(temperature):.0f} K"


def _safe_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _safe_filename_token(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _markdown(line_scans, psd_maps, feature_checks, plot_paths: dict[str, str] | None = None) -> str:
    lines = ["# Figure Feature Reality Triage", "", "## Measurement Classes", "", f"- Single-detector line scans: `{len(line_scans)}`", f"- PSD temperature maps: `{len(psd_maps)}`", "", "## Feature Checks", ""]
    for check in feature_checks:
        lines.append(
            f"- T~{check['temperature_K']} K, Q~{check['q']}: {check['line_scans_with_q_points']} line scans include nearby Q points; "
            f"strongest relative-to-median intensity {check['strongest_relative_to_scan_median']}; "
            f"strongest center/sideband ratio {check['strongest_center_over_sideband']}; background `{check['background_status']}`."
        )
        if check["files"]:
            lines.append(f"  Files: `{', '.join(check['files'][:8])}`")
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, path in sorted(plot_paths.items()):
            lines.append(f"- `{label}`: `{path}`")
            if path.endswith(".png"):
                lines.append(f"  ![{label}]({path})")
    lines.extend(["", "## Safe Interpretation", "", _safe_conclusion(line_scans, psd_maps, feature_checks)])
    return "\n".join(lines)


def _safe_conclusion(line_scans, psd_maps, feature_checks) -> str:
    if line_scans:
        return "The source set includes standard single-detector TAS line scans with explicit Q values, so red/yellow feature reality can be checked without PSD calibration by reviewing individual scans before interpolation or smoothing. These diagnostics use raw monitor-normalized intensities with no background subtraction; do not treat contour-map streaks as real until they survive raw-scan, sideband, neighboring-temperature, and background checks."
    if psd_maps:
        return "Only PSD/channel maps were found; feature checks in Q require reviewed PSD channel-to-A4/sample two-theta calibration. For elastic diffraction, Q from A4 is formulaic after that mapping is reviewed."
    return "No readable line scans or PSD maps were found."
