from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tas_copilot_contracts import AuditRecord, ProvenanceRecord
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.psd_calibration import discover_psd_dat_calibrations, psd_channel_q_axis
from tas_copilot_core.registry.tool_runner import stable_hash


WORKFLOW_NAME = "psd-mask-sensitivity"


def psd_mask_sensitivity(arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or "psd_mask_sensitivity").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = [Path(path).expanduser() for path in arguments.get("data_files", [])]
    calibration_dir = arguments.get("psd_calibration_dir") or arguments.get("calibration_dir")
    annotated_q = float(arguments.get("annotated_q", 1.24))
    if not data_files or not calibration_dir:
        return _missing(output_dir, ["data_files", "psd_calibration_dir"])
    missing = [str(path) for path in data_files if not path.exists()]
    if missing:
        return _missing(output_dir, ["data_files"], [f"Missing data file: {path}" for path in missing])
    discovered = discover_psd_dat_calibrations(calibration_dir, path_only=bool(arguments.get("path_only", False)))
    if discovered.get("status") != "ok":
        return _missing(output_dir, ["psd_calibration_dir"], [discovered.get("safe_conclusion", "Calibration discovery failed.")])
    calibration_names = arguments.get("calibration_files") or []
    records = discovered.get("records", [])
    if calibration_names:
        wanted = {Path(name).name for name in calibration_names}
        records = [record for record in records if record.get("filename") in wanted or Path(str(record.get("calibration_file"))).name in wanted]
    else:
        records = sorted(records, key=lambda record: str(record.get("filename")))[:2]
    if not records:
        return _missing(output_dir, ["calibration_files"], "No calibration DAT records matched the request.")

    summaries: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    for data_file in data_files:
        parsed = _parse_bt7_psd_file(data_file)
        if parsed["status"] != "ok":
            summaries.append(parsed)
            continue
        for calibration in records:
            result = _analyze_file_calibration(parsed, calibration, output_dir, annotated_q)
            summaries.extend(result["summaries"])
            plot_paths.extend(result["plots"])

    assessment = _probabilistic_assessment(summaries)
    payload = {
        "workflow_name": WORKFLOW_NAME,
        "status": "success" if summaries else "missing_input",
        "annotated_q": annotated_q,
        "data_files": [str(path) for path in data_files],
        "calibration_dir": str(Path(calibration_dir).expanduser()),
        "calibration_files": [record.get("filename") for record in records],
        "summaries": summaries,
        "artifact_assessment": assessment,
        "safe_conclusion": _safe_conclusion(assessment),
        "warnings": [
            "Q axes are computed from supplied PSD channel-to-A4 mappings; calibration acceptance is determined by reviewed registry records, not by this diagnostic workflow.",
            "Mask sensitivity is evidence for or against an artifact, not a definitive physics claim.",
        ],
        "provenance": [
            ProvenanceRecord(source_type="bt7_file", source_id=str(path), method=WORKFLOW_NAME).model_dump(mode="json")
            for path in data_files
        ]
        + [ProvenanceRecord(source_type="calibration_dir", source_id=str(Path(calibration_dir).expanduser()), method=WORKFLOW_NAME).model_dump(mode="json")],
    }
    payload["artifacts"] = _write_outputs(output_dir, payload, plot_paths)
    payload["audit_ids"] = _write_audit(output_dir, payload, arguments)
    return payload


def _parse_bt7_psd_file(path: Path) -> dict[str, Any]:
    cols: list[str] = []
    rows: list[list[str]] = []
    metadata: dict[str, Any] = {"path": str(path), "file": path.name}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#Filename"):
            metadata["native_filename"] = line.removeprefix("#Filename").strip()
        elif line.startswith("#Date"):
            metadata["date"] = line.removeprefix("#Date").strip()
        elif line.startswith("#AnalyzerDetectorMode"):
            metadata["analyzer_detector_mode"] = line.removeprefix("#AnalyzerDetectorMode").strip()
        elif line.startswith("#ScanDescr"):
            metadata["scan_description"] = line.removeprefix("#ScanDescr").strip()
        elif line.startswith("#Columns"):
            cols = line.split()[1:]
        elif cols and line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= len(cols):
                rows.append(parts[: len(cols)])
    if not rows:
        return {"status": "missing_input", "file": path.name, "missing_fields": ["data_rows"]}
    index = {column: idx for idx, column in enumerate(cols)}
    psd_columns = [column for column in cols if column.startswith("PSDC")]
    required = {"Temp", "A4", "Ef"}
    missing = sorted(required - set(index))
    if not psd_columns:
        missing.append("PSDC_columns")
    if missing:
        return {"status": "missing_input", "file": path.name, "missing_fields": missing}
    temperature = np.array([float(row[index["Temp"]]) for row in rows], dtype=float)
    a4 = np.array([float(row[index["A4"]]) for row in rows], dtype=float)
    ef = np.array([float(row[index["Ef"]]) for row in rows], dtype=float)
    intensity = np.array([[float(row[index[column]]) for column in psd_columns] for row in rows], dtype=float)
    order = np.argsort(temperature)
    return {
        "status": "ok",
        "metadata": metadata,
        "temperature": temperature[order],
        "a4": a4[order],
        "ef": ef[order],
        "intensity": intensity[order, :],
        "psd_channel_count": len(psd_columns),
    }


def _analyze_file_calibration(parsed: dict[str, Any], calibration: dict[str, Any], output_dir: Path, annotated_q: float) -> dict[str, Any]:
    rows = calibration.get("channels") or []
    if not rows:
        return {"summaries": [{"status": "missing_input", "file": parsed["metadata"]["file"], "calibration": calibration.get("filename"), "missing_fields": ["calibration_rows"]}], "plots": []}
    relative_a4 = np.array([float(row["relative_a4"]) for row in rows], dtype=float)
    efficiency = np.array([float(row["efficiency"]) for row in rows], dtype=float)
    q_axis = np.array(psd_channel_q_axis(float(parsed["a4"][0]), relative_a4.tolist(), energy_mev=float(parsed["ef"][0]))["q_axis"], dtype=float)
    corrected = parsed["intensity"] / np.where(efficiency == 0.0, np.nan, efficiency)[None, :]
    mask_sets = _mask_sets(calibration, efficiency)
    summaries: list[dict[str, Any]] = []
    images: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, one_based_mask in mask_sets.items():
        keep = np.ones(len(q_axis), dtype=bool)
        for channel in one_based_mask:
            if 1 <= channel <= len(keep):
                keep[channel - 1] = False
        q_keep = q_axis[keep]
        z_keep = corrected[:, keep]
        order = np.argsort(q_keep)
        q_sorted = q_keep[order]
        z_sorted = z_keep[:, order]
        stripe = _stripe_metric(q_sorted, z_sorted, annotated_q)
        horizontal = _horizontal_metric(parsed["temperature"], z_sorted)
        summaries.append(
            {
                "status": "success",
                "file": parsed["metadata"]["file"],
                "analyzer_detector_mode": parsed["metadata"].get("analyzer_detector_mode"),
                "calibration": calibration.get("filename"),
                "mask_label": label,
                "masked_one_based_channels": one_based_mask,
                "kept_channels": int(keep.sum()),
                "nearest_q_to_annotated": float(q_sorted[int(np.nanargmin(np.abs(q_sorted - annotated_q)))]),
                "q_stripe_metric": stripe,
                "horizontal_55_75K_metric": horizontal,
                "safe_use": False,
            }
        )
        images[label] = (q_sorted, parsed["temperature"], z_sorted)
    return {"summaries": summaries, "plots": _write_grid_plot(parsed["metadata"]["file"], calibration.get("filename"), images, output_dir, annotated_q)}


def _mask_sets(calibration: dict[str, Any], efficiency: np.ndarray) -> dict[str, list[int]]:
    masks: dict[str, list[int]] = {"none": []}
    if calibration.get("masked_channels"):
        masks["declared_calibration_mask"] = list(calibration["masked_channels"])
    masks["mask_ch48_one_based"] = [48]
    masks["mask_eff_lt_0p30"] = [int(idx + 1) for idx, value in enumerate(efficiency) if value < 0.30]
    masks["mask_eff_lt_0p50"] = [int(idx + 1) for idx, value in enumerate(efficiency) if value < 0.50]
    return {key: sorted(set(value)) for key, value in masks.items()}


def _stripe_metric(q_axis: np.ndarray, intensity: np.ndarray, annotated_q: float) -> float | None:
    if len(q_axis) < 3:
        return None
    nearest = int(np.nanargmin(np.abs(q_axis - annotated_q)))
    neighbors = [idx for idx in range(max(0, nearest - 2), min(len(q_axis), nearest + 3)) if idx != nearest]
    denominator = np.nanmedian([np.nanmean(intensity[:, idx]) for idx in neighbors]) if neighbors else np.nan
    if not denominator or not np.isfinite(denominator):
        return None
    return float(np.nanmean(intensity[:, nearest]) / denominator)


def _horizontal_metric(temperature: np.ndarray, intensity: np.ndarray) -> float | None:
    band = (temperature >= 55.0) & (temperature <= 75.0)
    reference = ((temperature >= 25.0) & (temperature < 45.0)) | ((temperature > 90.0) & (temperature <= 115.0))
    if not np.any(band) or not np.any(reference):
        return None
    denominator = np.nanmedian(intensity[reference, :])
    if not denominator or not np.isfinite(denominator):
        return None
    return float(np.nanmedian(intensity[band, :]) / denominator)


def _probabilistic_assessment(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for item in summaries:
        if item.get("status") != "success":
            continue
        by_key.setdefault((item["file"], item["calibration"]), {})[item["mask_label"]] = item
    for (file_name, calibration), masks in by_key.items():
        unmasked = masks.get("none")
        if not unmasked:
            continue
        before = unmasked.get("q_stripe_metric")
        if before is None:
            continue
        for mask_label, masked in masks.items():
            if mask_label == "none":
                continue
            after = masked.get("q_stripe_metric")
            if after is None:
                continue
            pairs.append({"file": file_name, "calibration": calibration, "mask_label": mask_label, "before": before, "after": after, "drop_factor": before / after if after else None})
    strong = [pair for pair in pairs if pair["before"] >= 3.0 and pair["after"] <= 1.2]
    probability = min(0.95, 0.55 + 0.1 * len(strong)) if strong else 0.35
    label = "probable_artifact" if probability >= 0.75 else "possible_artifact" if probability >= 0.5 else "insufficient_evidence"
    return {
        "conclusion": label,
        "artifact_probability": round(probability, 2),
        "evidence_count": len(strong),
        "mask_sensitivity_pairs": pairs,
        "probability_basis": "Heuristic probability from repeated disappearance of an annotated-Q stripe after applying the one-based channel-48 mask convention.",
    }


def _safe_conclusion(assessment: dict[str, Any]) -> str:
    probability = assessment.get("artifact_probability")
    conclusion = assessment.get("conclusion")
    if conclusion == "probable_artifact":
        return f"The annotated-Q feature is a probable detector-channel/calibration artifact (heuristic probability {probability:.2f}), because it repeatedly disappears after channel-mask sensitivity checks. This is not certainty."
    if conclusion == "possible_artifact":
        return f"The annotated-Q feature is a possible detector-channel/calibration artifact (heuristic probability {probability:.2f}); more review is required."
    return "Mask sensitivity did not provide enough evidence for a probability-level artifact assessment."


def _write_grid_plot(file_name: str, calibration_name: str | None, images: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_dir: Path, annotated_q: float) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    labels = [label for label in ["none", "mask_ch48_one_based", "mask_eff_lt_0p30", "mask_eff_lt_0p50"] if label in images]
    if not labels:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=140, sharey=True)
    axes_flat = axes.ravel()
    for ax, label in zip(axes_flat, labels):
        q_axis, temperature, intensity = images[label]
        image = ax.pcolormesh(
            q_axis,
            temperature,
            intensity,
            shading="auto",
            cmap="turbo",
            vmin=float(np.nanpercentile(intensity, 2)),
            vmax=float(np.nanpercentile(intensity, 98)),
        )
        ax.axvline(annotated_q, color="red", lw=1.1)
        ax.axhspan(55, 75, color="white", alpha=0.12)
        ax.set_xlim(1.10, 1.45)
        ax.set_ylim(0, 205)
        ax.set_title(label)
        ax.set_xlabel("Candidate Q axis (unaccepted)")
        fig.colorbar(image, ax=ax)
    for ax in axes_flat[len(labels) :]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Temperature (K)")
    axes_flat[2].set_ylabel("Temperature (K)")
    fig.suptitle(f"{file_name}\n{calibration_name} masked candidate Q axes (unaccepted)", fontsize=11)
    fig.tight_layout()
    path = output_dir / f"{_safe(file_name)}__{_safe(str(calibration_name))}_mask_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def _write_outputs(output_dir: Path, payload: dict[str, Any], plot_paths: list[str]) -> dict[str, Any]:
    json_path = output_dir / "psd_mask_sensitivity_report.json"
    md_path = output_dir / "psd_mask_sensitivity_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "plots": plot_paths}


def _write_audit(output_dir: Path, payload: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    audit = AuditRecord(
        event_type="workflow",
        status=payload["status"],
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": arguments}),
        output_hash=stable_hash(payload),
        details={"workflow_name": WORKFLOW_NAME, "artifact_probability": payload.get("artifact_assessment", {}).get("artifact_probability")},
    )
    JsonlAuditLogger(output_dir / "audit").write(audit)
    return [audit.audit_id]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PSD Mask Sensitivity Report",
        "",
        payload.get("safe_conclusion", ""),
        "",
        "## Artifact Assessment",
        "",
        f"- Conclusion: `{payload.get('artifact_assessment', {}).get('conclusion')}`",
        f"- Heuristic probability: `{payload.get('artifact_assessment', {}).get('artifact_probability')}`",
        f"- Basis: {payload.get('artifact_assessment', {}).get('probability_basis')}",
        "",
        "## Mask Sensitivity",
        "",
        "| File | Calibration | Mask | Stripe metric | Horizontal 55-75 K metric | Masked channels |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in payload.get("summaries", []):
        if item.get("status") != "success":
            continue
        horizontal = item.get("horizontal_55_75K_metric")
        lines.append(
            f"| `{item.get('file')}` | `{item.get('calibration')}` | `{item.get('mask_label')}` | "
            f"{_fmt(item.get('q_stripe_metric'))} | {_fmt(horizontal)} | `{item.get('masked_one_based_channels')}` |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3g}"


def _missing(output_dir: Path, fields: list[str], warnings: str | list[str] | None = None) -> dict[str, Any]:
    warning_list = [warnings] if isinstance(warnings, str) else warnings or ["Required inputs were missing."]
    payload = {"workflow_name": WORKFLOW_NAME, "status": "missing_input", "missing_fields": fields, "warnings": warning_list}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "psd_mask_sensitivity_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:140] or "item"
