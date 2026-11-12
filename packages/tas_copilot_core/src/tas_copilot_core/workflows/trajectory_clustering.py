from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.experiment_dashboard import _parse_scan_file


def cluster_order_parameter_trajectories(arguments: dict[str, Any]) -> dict[str, Any]:
    """Cluster temperature/intensity trajectories behind an explicit workflow boundary.

    This is an XTEC-inspired decision-support layer for TAS order-parameter
    triage. It compares normalized trajectory shapes and keeps the result
    probabilistic; it is not a direct import of XTEC and does not make a
    source-data or physics claim.
    """

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    id_column = str(arguments.get("id_column") or "trajectory_id")
    parameter_column = str(arguments.get("parameter_column") or "temperature")
    value_column = str(arguments.get("value_column") or "intensity")
    backend = str(arguments.get("backend") or "shape_rules")
    n_clusters = int(arguments.get("n_clusters") or 4)
    source_id: str
    source_method: str

    triage_path = Path(arguments["triage_json"]).expanduser() if arguments.get("triage_json") else None
    input_path = Path(arguments["input"]).expanduser() if arguments.get("input") else None
    if triage_path is not None:
        if not triage_path.exists():
            return _missing(output_dir, ["triage_json"], f"Order-parameter triage JSON does not exist: {triage_path}")
        value_column = str(arguments.get("observable") or value_column or "integrated_signal")
        trajectories, load_warnings = _load_triage_trajectories(triage_path, value_column=value_column)
        source_id = str(triage_path)
        source_method = "order_parameter_triage_json"
    elif input_path is not None:
        if not input_path.exists():
            return _missing(output_dir, ["input"], f"Trajectory input does not exist: {input_path}")
        trajectories, load_warnings = _load_trajectories(input_path, id_column=id_column, parameter_column=parameter_column, value_column=value_column)
        source_id = str(input_path)
        source_method = "trajectory_clustering_input"
    else:
        return _missing(output_dir, ["input_or_triage_json"], "A trajectory CSV/JSON file or order-parameter triage JSON is required.")
    valid = {key: sorted(points) for key, points in trajectories.items() if len({point[0] for point in points}) >= 3}
    rejected = [
        {"trajectory_id": key, "reason": "fewer than three parameter points"}
        for key, points in sorted(trajectories.items())
        if key not in valid
    ]
    if len(valid) < 2:
        report = {
            "workflow_name": "trajectory-cluster-order-parameters",
            "status": "missing_input",
            "missing_fields": ["at_least_two_trajectories_with_three_points"],
            "input": source_id,
            "input_type": source_method,
            "observable": value_column,
            "trajectory_count": len(trajectories),
            "valid_trajectory_count": len(valid),
            "rejected_trajectories": rejected,
            "warnings": load_warnings,
            "safe_conclusion": "Not enough valid trajectories were available for clustering.",
            "provenance": [_prov(source_id, source_method)],
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_trajectory_markdown, json_filename="trajectory_clustering_report.json", markdown_filename="trajectory_clustering_report.md")
        return report

    common_grid = _common_grid(list(valid.values()), max_points=int(arguments.get("max_grid_points") or 25))
    use_common_grid = backend in {"auto", "gmm"} and len(common_grid) >= 3
    if backend == "gmm" and not use_common_grid:
        report = {
            "workflow_name": "trajectory-cluster-order-parameters",
            "status": "missing_input",
            "missing_fields": ["common_parameter_grid"],
            "input": source_id,
            "input_type": source_method,
            "observable": value_column,
            "trajectory_count": len(trajectories),
            "valid_trajectory_count": len(valid),
            "rejected_trajectories": rejected,
            "warnings": [*load_warnings, "GMM clustering requires at least three shared parameter values across trajectories."],
            "safe_conclusion": "The trajectories do not overlap on at least three parameter values for GMM clustering.",
            "provenance": [_prov(source_id, source_method)],
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_trajectory_markdown, json_filename="trajectory_clustering_report.json", markdown_filename="trajectory_clustering_report.md")
        return report

    normalized = []
    for trajectory_id, points in sorted(valid.items()):
        if use_common_grid:
            grid = common_grid
            values = [_interpolate(points, value) for value in grid]
            if any(value is None for value in values):
                rejected.append({"trajectory_id": trajectory_id, "reason": "could not interpolate onto common grid"})
                continue
            numeric_values = [float(value) for value in values if value is not None]
        else:
            grid = [point[0] for point in points]
            numeric_values = [point[1] for point in points]
        norm = _normalize(numeric_values)
        label, confidence, reasons = _shape_label(grid, norm)
        normalized.append(
            {
                "trajectory_id": trajectory_id,
                "parameter_grid": grid,
                "normalized_values": norm,
                "shape_label": label,
                "shape_confidence": confidence,
                "shape_reasons": reasons,
            }
        )

    backend_status = {"backend": backend, "status": "shape_rules"}
    cluster_assignments = _shape_rule_assignments(normalized)
    if use_common_grid and backend in {"auto", "gmm"}:
        gmm_result = _try_gmm(normalized, n_clusters=n_clusters)
        if gmm_result["status"] == "success":
            cluster_assignments = gmm_result["assignments"]
            backend_status = {"backend": "sklearn_gaussian_mixture", "status": "success"}
        elif backend == "gmm":
            backend_status = {"backend": "sklearn_gaussian_mixture", "status": "unavailable", "reason": gmm_result["reason"]}
        else:
            backend_status = {"backend": "shape_rules", "status": "fallback", "reason": gmm_result["reason"]}
    elif backend == "auto":
        backend_status = {"backend": "shape_rules", "status": "fallback", "reason": "no common parameter grid across all trajectories"}

    clusters = _cluster_summaries(cluster_assignments, normalized)
    artifacts = _write_trajectory_artifacts(output_dir, normalized, clusters)
    report = {
        "workflow_name": "trajectory-cluster-order-parameters",
        "status": "ok" if normalized else "missing_input",
        "input": source_id,
        "input_type": source_method,
        "observable": value_column,
        "trajectory_count": len(trajectories),
        "valid_trajectory_count": len(normalized),
        "rejected_trajectories": rejected,
        "parameter_grid": common_grid if use_common_grid else "per_trajectory",
        "backend_status": backend_status,
        "clusters": clusters,
        "generated_artifacts": artifacts,
        "warnings": [
            *load_warnings,
            "This is an XTEC-inspired trajectory triage boundary, not a direct XTEC reciprocal-space volume analysis.",
            "Cluster labels are probability-style triage priors and require scan-level provenance, fits, and human review before physics conclusions.",
        ],
        "safe_conclusion": (
            f"Clustered {len(normalized)} normalized trajectories into {len(clusters)} trajectory-shape group(s). "
            "Use this to prioritize order-parameter review, not as a final fit or source-data claim."
        ),
        "provenance": [_prov(source_id, source_method)],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_trajectory_markdown, json_filename="trajectory_clustering_report.json", markdown_filename="trajectory_clustering_report.md")
    return report


def order_parameter_trajectory_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Join order-parameter triage and trajectory clustering into a review table."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    triage_path = Path(arguments["triage_json"]).expanduser() if arguments.get("triage_json") else None
    trajectory_path = Path(arguments["trajectory_json"]).expanduser() if arguments.get("trajectory_json") else None
    if triage_path is None or not triage_path.exists():
        return _missing(output_dir, ["triage_json"], "An order-parameter triage JSON file is required.")
    if trajectory_path is None or not trajectory_path.exists():
        return _missing(output_dir, ["trajectory_json"], "A trajectory clustering JSON file is required.")

    triage = json.loads(triage_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    labels = _trajectory_labels_from_report(trajectory)
    rows = []
    for candidate in triage.get("candidates") or []:
        candidate_id = str(candidate.get("candidate_id"))
        label = labels.get(candidate_id, {})
        rows.append(_combined_candidate_row(candidate, label, max_paths=int(arguments.get("max_paths_per_candidate") or 200)))
    rows.sort(key=lambda item: (item["combined_priority"], item.get("scan_count") or 0), reverse=True)
    top_k = int(arguments.get("top_k") or 25)
    top_rows = rows[:top_k]
    artifacts = _write_order_parameter_trajectory_artifacts(output_dir, top_rows, rows)
    report = {
        "workflow_name": "order-parameter-trajectory-report",
        "status": "ok" if rows else "missing_input",
        "triage_json": str(triage_path),
        "trajectory_json": str(trajectory_path),
        "candidate_count": len(rows),
        "reported_candidate_count": len(top_rows),
        "top_candidates": top_rows,
        "generated_artifacts": artifacts,
        "safe_conclusion": (
            f"Joined {len(rows)} order-parameter triage candidate(s) with trajectory-shape labels. "
            "Priorities are review aids, not fit results or physical conclusions."
            if rows
            else "No candidates were available to join."
        ),
        "warnings": [
            "Combined priorities are deterministic triage scores and must not replace peak fitting, background checks, repeatability checks, or human review.",
            "Trajectory labels can disagree with peak-splitting heuristics; disagreement is a diagnostic signal, not an error.",
            "Historical BT-7 metadata can be sparse, so representative paths and candidate families should be inspected before acting.",
        ],
        "provenance": [
            ProvenanceRecord(source_type="order_parameter_triage", source_id=str(triage_path), method="combined_order_parameter_trajectory_report").model_dump(mode="json"),
            ProvenanceRecord(source_type="trajectory_clustering", source_id=str(trajectory_path), method="combined_order_parameter_trajectory_report").model_dump(mode="json"),
        ],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_combined_markdown, json_filename="order_parameter_trajectory_report.json", markdown_filename="order_parameter_trajectory_report.md")
    return report


def plot_order_parameter_candidate_scans(arguments: dict[str, Any]) -> dict[str, Any]:
    """Plot raw normalized scan profiles for top combined order-parameter candidates."""

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(arguments["report_json"]).expanduser() if arguments.get("report_json") else None
    if report_path is None or not report_path.exists():
        return _missing(output_dir, ["report_json"], "A combined order-parameter trajectory report JSON file is required.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    top_k = int(arguments.get("top_k") or 5)
    max_scans = int(arguments.get("max_scans_per_candidate") or 12)
    allow_rebin = bool(arguments.get("allow_rebin"))
    mesh_grid_points = int(arguments.get("mesh_grid_points") or 120)
    candidates = (report.get("top_candidates") or [])[:top_k]
    plot_rows = []
    warnings = []
    for candidate in candidates:
        result = _plot_candidate_scan_family(candidate, output_dir=output_dir, max_scans=max_scans, allow_rebin=allow_rebin, mesh_grid_points=mesh_grid_points)
        plot_rows.append(result)
        if result.get("status") != "success":
            warnings.append(f"{candidate.get('candidate_id')}: {result.get('reason') or result.get('status')}")
    artifacts = _write_candidate_plot_index(output_dir, plot_rows)
    gallery = _write_plot_gallery(output_dir, plot_rows, experiment_id=str(arguments.get("experiment_id") or ""))
    artifacts.update(gallery)
    result = {
        "workflow_name": "plot-order-parameter-candidates",
        "status": "ok" if any(row.get("status") == "success" for row in plot_rows) else "missing_input",
        "report_json": str(report_path),
        "candidate_count": len(candidates),
        "plot_count": sum(1 for row in plot_rows if row.get("status") == "success"),
        "allow_rebin": allow_rebin,
        "mesh_grid_points": mesh_grid_points,
        "plots": plot_rows,
        "generated_artifacts": artifacts,
        "warnings": [
            *warnings,
            "Plots are raw normalized scan overlays for review. They are not peak fits, background subtraction, or transition claims.",
            "Mesh rebinning is disabled unless explicitly requested. When enabled, it is a review visualization assumption, not a reduction step.",
        ],
        "safe_conclusion": "Generated raw scan overlay plots for candidate review." if plot_rows else "No candidate plots were generated.",
        "provenance": [ProvenanceRecord(source_type="order_parameter_trajectory_report", source_id=str(report_path), method="plot_order_parameter_candidate_scans").model_dump(mode="json")],
    }
    result["artifacts"] = write_report_bundle(output_dir, result, markdown_renderer=_plot_markdown, json_filename="order_parameter_candidate_plots.json", markdown_filename="order_parameter_candidate_plots.md")
    return result


def _load_trajectories(path: Path, *, id_column: str, parameter_column: str, value_column: str) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    warnings: list[str] = []
    trajectories: dict[str, list[tuple[float, float]]] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("trajectories", payload) if isinstance(payload, dict) else payload
        for item in rows:
            trajectory_id = str(item.get(id_column) or item.get("trajectory_id") or item.get("id"))
            points = item.get("points") or []
            for point in points:
                parameter = point.get(parameter_column, point.get("temperature", point.get("parameter")))
                value = point.get(value_column, point.get("intensity", point.get("value")))
                _append_point(trajectories, trajectory_id, parameter, value, warnings)
        return trajectories, warnings

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trajectory_id = str(row.get(id_column) or row.get("trajectory_id") or row.get("id"))
            _append_point(trajectories, trajectory_id, row.get(parameter_column), row.get(value_column), warnings)
    return trajectories, warnings


def _plot_candidate_scan_family(candidate: dict[str, Any], *, output_dir: Path, max_scans: int, allow_rebin: bool = False, mesh_grid_points: int = 120) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "unknown_candidate")
    paths = [Path(path) for path in (candidate.get("representative_paths") or []) if path]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return {"candidate_id": candidate_id, "status": "missing_input", "reason": "no readable representative paths"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"candidate_id": candidate_id, "status": "unavailable", "reason": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}

    records = [_parse_scan_file(path, max_plot_points=2000) for path in paths]
    records = [record for record in records if (record.plot_data or {}).get("x") and (record.plot_data or {}).get("y")]
    records = _select_temperature_spread_records(records, max_records=max_scans)
    if not records:
        return {"candidate_id": candidate_id, "status": "missing_input", "reason": "representative scans had no plottable profile"}

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    rows = []
    for record in sorted(records, key=lambda item: float((item.temperature_range or {}).get("mean") or 0.0)):
        plot = record.plot_data or {}
        xs = [float(value) for value in plot.get("x") or []]
        ys = [float(value) for value in plot.get("y") or []]
        ys_norm = _normalize(ys) if ys else []
        temp = (record.temperature_range or {}).get("mean")
        label = f"{temp:.1f} K" if isinstance(temp, (int, float)) else Path(record.path).name
        ax.plot(xs[: len(ys_norm)], ys_norm, marker="o", markersize=2, linewidth=1, label=label)
        rows.append({"scan_id": record.scan_id, "path": record.path, "temperature": temp, "x_column": plot.get("x_column"), "y_column": plot.get("y_column"), "point_count": len(ys_norm)})
    ax.set_title(candidate_id)
    ax.set_xlabel(str((records[0].plot_data or {}).get("x_column") or "scan axis"))
    ax.set_ylabel("normalized intensity")
    if len(records) <= 12:
        ax.legend(fontsize=7, loc="best")
    safe_id = _safe_filename(candidate_id)
    png = output_dir / f"{safe_id}_raw_scan_overlay.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    csv_path = output_dir / f"{safe_id}_plotted_scans.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scan_id", "path", "temperature", "x_column", "y_column", "point_count"])
        writer.writeheader()
        writer.writerows(rows)
    mesh_result = _write_candidate_mesh_plot(output_dir / f"{safe_id}_temperature_mesh.png", records, allow_rebin=allow_rebin, mesh_grid_points=mesh_grid_points)
    shape_label = str(candidate.get("trajectory_shape_label") or "unclustered_or_sparse")
    priority = _safe_float(candidate.get("combined_priority")) or 0.0
    review_candidate = {**candidate, "scan_count": candidate.get("scan_count") or len(records)}
    return {
        "candidate_id": candidate_id,
        "status": "success",
        "plot_png": str(png),
        **mesh_result,
        "scan_table_csv": str(csv_path),
        "scan_count": len(records),
        "classification": candidate.get("classification"),
        "trajectory_shape_label": shape_label,
        "combined_priority": candidate.get("combined_priority"),
        "review_tier": candidate.get("review_tier") or _review_tier(review_candidate, shape_label, priority),
        "recommended_next_action": candidate.get("recommended_next_action"),
    }


def _write_candidate_plot_index(output_dir: Path, plot_rows: list[dict[str, Any]]) -> dict[str, str]:
    path = output_dir / "order_parameter_candidate_plot_index.csv"
    fieldnames = ["candidate_id", "status", "plot_png", "mesh_png", "mesh_status", "mesh_rebin_applied", "scan_table_csv", "scan_count", "classification", "trajectory_shape_label", "combined_priority", "review_tier", "recommended_next_action", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in plot_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return {"plot_index_csv": str(path)}


def _write_plot_gallery(output_dir: Path, plot_rows: list[dict[str, Any]], *, experiment_id: str) -> dict[str, str]:
    path = output_dir / "order_parameter_candidate_gallery.html"
    title = f"Order-Parameter Candidate Gallery {experiment_id}".strip()
    cards = []
    for row in plot_rows:
        candidate = row.get("candidate_id")
        png = row.get("plot_png")
        mesh = row.get("mesh_png")
        details = [
            f"status: {row.get('status')}",
            f"trajectory: {row.get('trajectory_shape_label')}",
            f"priority: {row.get('combined_priority')}",
            f"review tier: {row.get('review_tier')}",
            f"scans: {row.get('scan_count')}",
            f"mesh: {row.get('mesh_status')}",
            f"rebin: {row.get('mesh_rebin_applied')}",
            f"next: {row.get('recommended_next_action')}",
        ]
        image_html = f'<img src="{Path(png).name}" alt="{candidate} raw overlay">' if png else "<p>No raw overlay available.</p>"
        mesh_html = f'<img src="{Path(mesh).name}" alt="{candidate} temperature mesh">' if mesh else f"<p>Mesh: {row.get('mesh_status') or 'not available'}</p>"
        cards.append(
            "<section class=\"card\">"
            f"<h2>{candidate}</h2>"
            + image_html
            + mesh_html
            + "<ul>"
            + "".join(f"<li>{item}</li>" for item in details)
            + "</ul>"
            "</section>"
        )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f7f7f5; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8d8d2; border-radius: 8px; padding: 14px; }}
    img {{ width: 100%; height: auto; border: 1px solid #e4e4df; margin: 8px 0; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Raw normalized overlays and temperature meshes for one experiment-scoped candidate review. These are review artifacts, not fits or conclusions.</p>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
""".format(title=title, cards="\n".join(cards))
    path.write_text(html, encoding="utf-8")
    return {"gallery_html": str(path)}


def _select_temperature_spread_records(records: list[Any], *, max_records: int) -> list[Any]:
    if len(records) <= max_records:
        return records
    with_temp = [record for record in records if _safe_float((record.temperature_range or {}).get("mean")) is not None]
    if len(with_temp) < 2:
        return records[:max_records]
    ordered = sorted(with_temp, key=lambda record: float((record.temperature_range or {}).get("mean")))
    selected_indices = {
        round(index * (len(ordered) - 1) / max(1, max_records - 1))
        for index in range(max_records)
    }
    return [ordered[index] for index in sorted(selected_indices)]


def _safe_filename(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "candidate"


def _write_candidate_mesh_plot(path: Path, records: list[Any], *, allow_rebin: bool = False, mesh_grid_points: int = 120) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"mesh_status": "unavailable", "mesh_warning": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}
    records = sorted(records, key=lambda item: float((item.temperature_range or {}).get("mean") or 0.0))
    if len(records) < 3:
        return {"mesh_status": "missing_input", "mesh_warning": "At least three scans are required for a temperature mesh."}
    x_columns = {str((record.plot_data or {}).get("x_column")) for record in records}
    if len(x_columns) != 1:
        return {"mesh_status": "missing_input", "mesh_warning": "Scan axes are not compatible for a mesh plot."}
    lengths = {len((record.plot_data or {}).get("x") or []) for record in records}
    xs_by_record = [[float(value) for value in (record.plot_data or {}).get("x") or []] for record in records]
    temps = []
    z = []
    mesh_status = "success"
    mesh_rebin_applied = False
    mesh_warning = None
    if len(lengths) == 1 and _same_x_grid(xs_by_record):
        xs = xs_by_record[0]
    elif allow_rebin:
        lower = max(min(xs) for xs in xs_by_record if xs)
        upper = min(max(xs) for xs in xs_by_record if xs)
        if upper <= lower:
            return {"mesh_status": "missing_input", "mesh_warning": "Scan axes do not overlap enough for explicit rebinning."}
        grid_count = max(5, mesh_grid_points)
        step = (upper - lower) / float(grid_count - 1)
        xs = [lower + index * step for index in range(grid_count)]
        mesh_status = "success_rebinned"
        mesh_rebin_applied = True
        mesh_warning = "Mesh was explicitly rebinned onto a shared axis for visualization only."
    else:
        return {"mesh_status": "missing_input", "mesh_warning": "Scan point counts or x-grids differ; pass --allow-rebin to make a visual mesh with explicit interpolation."}
    for record in records:
        temp = (record.temperature_range or {}).get("mean")
        if not isinstance(temp, (int, float)):
            return {"mesh_status": "missing_input", "mesh_warning": "Temperature metadata is missing for at least one scan."}
        original_xs = [float(value) for value in (record.plot_data or {}).get("x") or []]
        ys = _normalize([float(value) for value in (record.plot_data or {}).get("y") or []])
        if mesh_rebin_applied:
            interpolated = [_interpolate(list(zip(original_xs, ys)), value) for value in xs]
            if any(value is None for value in interpolated):
                return {"mesh_status": "missing_input", "mesh_warning": "Could not interpolate at least one scan onto the shared mesh axis."}
            ys = [float(value) for value in interpolated if value is not None]
        temps.append(float(temp))
        z.append(ys)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mesh = ax.pcolormesh(xs, temps, z, shading="auto")
    fig.colorbar(mesh, ax=ax, label="normalized intensity")
    ax.set_xlabel(str((records[0].plot_data or {}).get("x_column") or "scan axis"))
    ax.set_ylabel("temperature (K)")
    ax.set_title(path.stem.replace("_temperature_mesh", ""))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    result = {"mesh_status": mesh_status, "mesh_png": str(path), "mesh_rebin_applied": mesh_rebin_applied}
    if mesh_warning:
        result["mesh_warning"] = mesh_warning
    return result


def _same_x_grid(xs_by_record: list[list[float]]) -> bool:
    if not xs_by_record:
        return False
    first = xs_by_record[0]
    return all(len(xs) == len(first) and all(abs(left - right) < 1e-9 for left, right in zip(first, xs)) for xs in xs_by_record[1:])


def _trajectory_labels_from_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for cluster in report.get("clusters") or []:
        cluster_id = str(cluster.get("cluster_id"))
        shape_labels = cluster.get("shape_labels") or []
        shape_label = str(shape_labels[0]) if shape_labels else cluster_id
        for trajectory_id in cluster.get("trajectory_ids") or []:
            labels[str(trajectory_id)] = {
                "cluster_id": cluster_id,
                "shape_label": shape_label,
                "shape_confidence": cluster.get("mean_shape_confidence"),
            }
    artifacts = report.get("generated_artifacts") or {}
    csv_path = artifacts.get("normalized_trajectories_csv")
    if csv_path and Path(csv_path).exists():
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                trajectory_id = str(row.get("trajectory_id") or "")
                if not trajectory_id:
                    continue
                labels.setdefault(
                    trajectory_id,
                    {
                        "cluster_id": row.get("shape_label"),
                        "shape_label": row.get("shape_label"),
                        "shape_confidence": _safe_float(row.get("shape_confidence")),
                    },
                )
    return labels


def _combined_candidate_row(candidate: dict[str, Any], label: dict[str, Any], *, max_paths: int = 200) -> dict[str, Any]:
    shape_label = str(label.get("shape_label") or "unclustered_or_sparse")
    shape_confidence = _safe_float(label.get("shape_confidence"))
    triage_probability = _safe_float(candidate.get("probability") or candidate.get("score")) or 0.0
    combined_priority = _combined_priority(triage_probability, shape_label, shape_confidence)
    per_scan = candidate.get("per_scan_peak_summary") or []
    paths = _temperature_representative_paths(per_scan, max_paths=max_paths)
    temp_range = candidate.get("temperature_range") or {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "classification": candidate.get("classification"),
        "triage_probability": round(triage_probability, 3),
        "trajectory_shape_label": shape_label,
        "trajectory_shape_confidence": round(shape_confidence, 3) if shape_confidence is not None else None,
        "combined_priority": round(combined_priority, 3),
        "review_tier": _review_tier(candidate, shape_label, combined_priority),
        "scan_count": candidate.get("scan_count"),
        "temperature_min": temp_range.get("min"),
        "temperature_max": temp_range.get("max"),
        "area_ratio": candidate.get("area_ratio"),
        "center_span": candidate.get("center_span"),
        "recommended_next_action": _recommended_combined_action(candidate, shape_label),
        "representative_paths": paths,
        "provenance_note": "candidate row joins order-parameter triage evidence with trajectory-clustering evidence",
    }


def _combined_priority(triage_probability: float, shape_label: str, shape_confidence: float | None) -> float:
    shape_weight = {
        "order_parameter_like_on_cooling": 0.9,
        "inverse_or_warming_enhanced": 0.72,
        "nonmonotonic_peak_or_nuisance_like": 0.5,
        "ambiguous_temperature_trajectory": 0.42,
        "nearly_flat_or_background_like": 0.18,
    }.get(shape_label, 0.25)
    confidence = shape_confidence if shape_confidence is not None else 0.35
    return min(0.98, 0.55 * triage_probability + 0.3 * shape_weight + 0.15 * confidence)


def _review_tier(candidate: dict[str, Any], shape_label: str, combined_priority: float) -> str:
    scan_count = int(candidate.get("scan_count") or 0)
    classification = str(candidate.get("classification") or "")
    if scan_count < 4:
        return "blocked_sparse"
    if "structural_splitting" in classification and shape_label == "order_parameter_like_on_cooling" and combined_priority >= 0.72:
        return "fit_ready_split_peak_review"
    if shape_label == "order_parameter_like_on_cooling" and combined_priority >= 0.68:
        return "fit_ready_intensity_review"
    if shape_label in {"nonmonotonic_peak_or_nuisance_like", "ambiguous_temperature_trajectory"}:
        return "diagnostic_review"
    if shape_label == "nearly_flat_or_background_like":
        return "low_priority_background_review"
    return "raw_scan_review"


def _temperature_representative_paths(per_scan: list[dict[str, Any]], *, max_paths: int) -> list[str]:
    rows = [row for row in per_scan if row.get("path")]
    if len(rows) <= max_paths:
        return [str(row["path"]) for row in rows]
    with_temp = [row for row in rows if _safe_float(row.get("temperature_mean")) is not None]
    if len(with_temp) < 2:
        return [str(row["path"]) for row in rows[:max_paths]]
    ordered = sorted(with_temp, key=lambda row: float(row["temperature_mean"]))
    selected_indices = {
        round(index * (len(ordered) - 1) / max(1, max_paths - 1))
        for index in range(max_paths)
    }
    return [str(ordered[index]["path"]) for index in sorted(selected_indices)]


def _recommended_combined_action(candidate: dict[str, Any], shape_label: str) -> str:
    classification = str(candidate.get("classification") or "")
    if "structural_splitting" in classification:
        if shape_label == "order_parameter_like_on_cooling":
            return "Plot raw scans, then run constrained split-peak fits with background and BIC checks."
        return "Inspect raw profiles first; trajectory shape does not simply confirm the split-peak heuristic."
    if shape_label == "order_parameter_like_on_cooling":
        return "Fit integrated intensity versus temperature after background and nuisance-peak checks."
    if shape_label == "inverse_or_warming_enhanced":
        return "Check sign convention, background subtraction, extinction, and whether the measured feature is a disappearing order parameter."
    if shape_label == "nonmonotonic_peak_or_nuisance_like":
        return "Check for nuisance peaks, sample-environment changes, scan drift, or multiple transitions before fitting."
    if shape_label == "nearly_flat_or_background_like":
        return "Deprioritize as an order-parameter fit unless raw scans show a specific peak-shape change."
    return "Plot raw scans and peak summaries before selecting an order-parameter or splitting model."


def _write_order_parameter_trajectory_artifacts(output_dir: Path, top_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, str]:
    top_csv = output_dir / "order_parameter_trajectory_top_candidates.csv"
    all_csv = output_dir / "order_parameter_trajectory_all_candidates.csv"
    fieldnames = [
        "candidate_id",
        "classification",
        "triage_probability",
        "trajectory_shape_label",
        "trajectory_shape_confidence",
        "combined_priority",
        "review_tier",
        "scan_count",
        "temperature_min",
        "temperature_max",
        "area_ratio",
        "center_span",
        "recommended_next_action",
        "representative_paths",
    ]
    for path, rows in ((top_csv, top_rows), (all_csv, all_rows)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload = dict(row)
                payload["representative_paths"] = ";".join(payload.get("representative_paths") or [])
                writer.writerow({key: payload.get(key) for key in fieldnames})
    return {"top_candidates_csv": str(top_csv), "all_candidates_csv": str(all_csv)}


def _load_triage_trajectories(path: Path, *, value_column: str) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    warnings: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectories: dict[str, list[tuple[float, float]]] = {}
    for candidate in payload.get("candidates") or []:
        trajectory_id = str(candidate.get("candidate_id") or candidate.get("group_id") or "unknown_candidate")
        rows = candidate.get("per_scan_peak_summary") or []
        for row in rows:
            temperature = row.get("temperature_mean")
            value = row.get(value_column)
            if value is None and value_column == "intensity":
                value = row.get("integrated_signal")
            _append_point(trajectories, trajectory_id, temperature, value, warnings)
    if not trajectories:
        warnings.append("No candidate per-scan peak summaries were found in the triage report.")
    return trajectories, warnings


def _append_point(trajectories: dict[str, list[tuple[float, float]]], trajectory_id: str, parameter: Any, value: Any, warnings: list[str]) -> None:
    try:
        x = float(parameter)
        y = float(value)
    except Exception:
        warnings.append(f"Skipped non-numeric point for trajectory {trajectory_id!r}.")
        return
    if not math.isfinite(x) or not math.isfinite(y):
        warnings.append(f"Skipped non-finite point for trajectory {trajectory_id!r}.")
        return
    trajectories.setdefault(trajectory_id, []).append((x, y))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _common_grid(all_points: list[list[tuple[float, float]]], *, max_points: int) -> list[float]:
    lower = max(min(point[0] for point in points) for points in all_points)
    upper = min(max(point[0] for point in points) for points in all_points)
    if upper <= lower:
        return []
    raw = sorted({point[0] for points in all_points for point in points if lower <= point[0] <= upper})
    if len(raw) <= max_points:
        return raw
    step = (upper - lower) / float(max_points - 1)
    return [lower + index * step for index in range(max_points)]


def _interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    points = sorted(points)
    if x < points[0][0] or x > points[-1][0]:
        return None
    for px, py in points:
        if abs(px - x) < 1e-12:
            return py
    for left, right in zip(points, points[1:]):
        if left[0] <= x <= right[0]:
            frac = (x - left[0]) / (right[0] - left[0])
            return left[1] + frac * (right[1] - left[1])
    return None


def _normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _shape_label(grid: list[float], values: list[float]) -> tuple[str, float, list[str]]:
    first = values[0]
    last = values[-1]
    middle = values[len(values) // 2]
    span = max(values) - min(values)
    delta = last - first
    reasons = [f"normalized span {span:.3f}", f"high-minus-low parameter endpoint delta {delta:.3f}"]
    if span < 0.2:
        return "nearly_flat_or_background_like", 0.55, reasons
    if delta <= -0.35:
        return "order_parameter_like_on_cooling", min(0.9, 0.55 + abs(delta) * 0.4), reasons
    if delta >= 0.35:
        return "inverse_or_warming_enhanced", min(0.85, 0.5 + abs(delta) * 0.35), reasons
    if middle > first + 0.25 and middle > last + 0.25:
        return "nonmonotonic_peak_or_nuisance_like", 0.65, reasons
    return "ambiguous_temperature_trajectory", 0.45, reasons


def _shape_rule_assignments(normalized: list[dict[str, Any]]) -> dict[str, str]:
    return {item["trajectory_id"]: item["shape_label"] for item in normalized}


def _try_gmm(normalized: list[dict[str, Any]], *, n_clusters: int) -> dict[str, Any]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        return {"status": "unavailable", "reason": f"sklearn unavailable: {type(exc).__name__}: {exc}"}
    if len(normalized) < 2:
        return {"status": "missing_input", "reason": "fewer than two trajectories"}
    n = min(max(1, n_clusters), len(normalized))
    vectors = [item["normalized_values"] for item in normalized]
    model = GaussianMixture(n_components=n, random_state=0)
    labels = model.fit_predict(vectors)
    return {"status": "success", "assignments": {item["trajectory_id"]: f"gmm_cluster_{int(label)}" for item, label in zip(normalized, labels)}}


def _cluster_summaries(assignments: dict[str, str], normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["trajectory_id"]: item for item in normalized}
    clusters: dict[str, list[dict[str, Any]]] = {}
    for trajectory_id, cluster_id in assignments.items():
        clusters.setdefault(cluster_id, []).append(by_id[trajectory_id])
    summaries = []
    for cluster_id, items in sorted(clusters.items()):
        labels = sorted({item["shape_label"] for item in items})
        summaries.append(
            {
                "cluster_id": cluster_id,
                "trajectory_count": len(items),
                "trajectory_ids": [item["trajectory_id"] for item in items],
                "shape_labels": labels,
                "mean_shape_confidence": round(sum(item["shape_confidence"] for item in items) / len(items), 3),
                "status": "proposed",
                "human_review_required": True,
            }
        )
    return summaries


def _write_trajectory_artifacts(output_dir: Path, normalized: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> dict[str, str]:
    trajectories_csv = output_dir / "normalized_trajectories.csv"
    with trajectories_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "parameter", "normalized_value", "shape_label", "shape_confidence"])
        writer.writeheader()
        for item in normalized:
            for parameter, value in zip(item["parameter_grid"], item["normalized_values"]):
                writer.writerow({"trajectory_id": item["trajectory_id"], "parameter": parameter, "normalized_value": value, "shape_label": item["shape_label"], "shape_confidence": item["shape_confidence"]})
    clusters_csv = output_dir / "trajectory_clusters.csv"
    with clusters_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "trajectory_count", "trajectory_ids", "shape_labels", "mean_shape_confidence", "status", "human_review_required"])
        writer.writeheader()
        for item in clusters:
            writer.writerow({**item, "trajectory_ids": ";".join(item["trajectory_ids"]), "shape_labels": ";".join(item["shape_labels"])})
    return {"normalized_trajectories_csv": str(trajectories_csv), "trajectory_clusters_csv": str(clusters_csv)}


def _missing(output_dir: Path, missing_fields: list[str], reason: str) -> dict[str, Any]:
    report = {
        "workflow_name": "trajectory-cluster-order-parameters",
        "status": "missing_input",
        "missing_fields": missing_fields,
        "safe_conclusion": reason,
        "warnings": ["Trajectory clustering is not attempted when required inputs are missing."],
        "provenance": [ProvenanceRecord(source_type="workflow", source_id="trajectory-cluster-order-parameters", method="missing_input").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_trajectory_markdown, json_filename="trajectory_clustering_report.json", markdown_filename="trajectory_clustering_report.md")
    return report


def _prov(source_id: str, method: str) -> dict[str, Any]:
    return ProvenanceRecord(source_type="trajectory_input", source_id=source_id, method=method).model_dump(mode="json")


def _trajectory_markdown(report: dict[str, Any]) -> str:
    lines = ["# Trajectory Clustering For Order-Parameter Triage", "", f"Status: {report.get('status')}", "", "## Safe Conclusion", str(report.get("safe_conclusion") or ""), ""]
    if report.get("backend_status"):
        lines.extend(["## Backend", ""])
        for key, value in report["backend_status"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## Clusters", ""])
    clusters = report.get("clusters") or []
    if clusters:
        lines.append("| cluster | trajectories | labels | confidence |")
        lines.append("| --- | --- | --- | --- |")
        for cluster in clusters:
            lines.append(f"| {cluster.get('cluster_id')} | {cluster.get('trajectory_count')} | {', '.join(cluster.get('shape_labels') or [])} | {cluster.get('mean_shape_confidence')} |")
    else:
        lines.append("No clusters generated.")
    lines.append("")
    if report.get("rejected_trajectories"):
        lines.extend(["## Rejected Trajectories", ""])
        for item in report["rejected_trajectories"][:20]:
            lines.append(f"- `{item.get('trajectory_id')}`: {item.get('reason')}")
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _combined_markdown(report: dict[str, Any]) -> str:
    lines = ["# Order-Parameter Trajectory Review Report", "", f"Status: {report.get('status')}", "", "## Safe Conclusion", str(report.get("safe_conclusion") or ""), ""]
    candidates = report.get("top_candidates") or []
    if candidates:
        lines.extend(["## Top Candidates", ""])
        lines.append("| candidate | triage | trajectory | priority | review tier | scans | next action |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in candidates[:25]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("candidate_id")),
                        str(item.get("classification")),
                        str(item.get("trajectory_shape_label")),
                        str(item.get("combined_priority")),
                        str(item.get("review_tier")),
                        str(item.get("scan_count")),
                        str(item.get("recommended_next_action")),
                    ]
                )
                + " |"
            )
        lines.append("")
    else:
        lines.extend(["## Top Candidates", "", "No joined candidates were available.", ""])
    if report.get("generated_artifacts"):
        lines.extend(["## Artifacts", ""])
        for key, value in report["generated_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _plot_markdown(report: dict[str, Any]) -> str:
    lines = ["# Order-Parameter Candidate Raw Scan Plots", "", f"Status: {report.get('status')}", "", "## Safe Conclusion", str(report.get("safe_conclusion") or ""), ""]
    plots = report.get("plots") or []
    if plots:
        lines.extend(["## Plots", ""])
        lines.append("| candidate | status | scans | trajectory | review tier | mesh | rebin | plot | mesh plot |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in plots:
            plot = item.get("plot_png") or ""
            mesh = item.get("mesh_png") or ""
            lines.append(
                f"| {item.get('candidate_id')} | {item.get('status')} | {item.get('scan_count', '')} | "
                f"{item.get('trajectory_shape_label', '')} | {item.get('review_tier', '')} | "
                f"{item.get('mesh_status', '')} | {item.get('mesh_rebin_applied', '')} | `{plot}` | `{mesh}` |"
            )
        lines.append("")
    if report.get("generated_artifacts"):
        lines.extend(["## Artifacts", ""])
        for key, value in report["generated_artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)
