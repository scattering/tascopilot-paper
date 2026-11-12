"""Semantic experiment reporting: group scans by scientific object, not time.

The overnight/morning surfaces are chronologies ("what happened last night").
These workflows answer "where does the experiment stand": alignment grouped by
reflection, order parameters as derived I(T) curves, and dispersions as
assembled E-vs-Q maps pooled across every inelastic scan so far.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.experiment_dashboard import (
    SCAN_PATTERNS,
    DashboardScanRecord,
    _ALIGNMENT_PLOT_FAMILIES,
    _find_scan_paths,
    _load_alignment_plot_context,
    _overnight_family,
    _parse_scan_file,
    _reflection_label,
    _safe_id,
    _write_overnight_series_plots,
)


def _load_records(arguments: dict[str, Any]) -> tuple[list[DashboardScanRecord], list[str]]:
    roots = arguments.get("data_roots") or ([arguments["data_root"]] if arguments.get("data_root") else [])
    records: list[DashboardScanRecord] = []
    seen_roots: list[str] = []
    for root in roots:
        data_root = Path(str(root)).expanduser()
        if not data_root.exists():
            continue
        seen_roots.append(str(data_root))
        paths = _find_scan_paths(
            data_root,
            patterns=tuple(arguments.get("patterns") or SCAN_PATTERNS),
            max_files=int(arguments.get("max_files", 2000)),
            post_year=int(arguments.get("post_year", 0) or 0),
        )
        records.extend(_parse_scan_file(path, max_plot_points=int(arguments.get("max_plot_points", 600))) for path in paths)
    return records, seen_roots


def _columns(record: DashboardScanRecord) -> dict[str, list[float]]:
    return ((record.plot_data or {}).get("all_columns_data")) or {}


def _span(values: list[Any]) -> float:
    numeric = [float(v) for v in values if v is not None]
    return (max(numeric) - min(numeric)) if numeric else 0.0


def _median(values: list[Any]) -> float | None:
    numeric = sorted(float(v) for v in values if v is not None)
    return numeric[len(numeric) // 2] if numeric else None


def _session_label(record: DashboardScanRecord) -> str:
    """Sessions must not be pooled: energy and orientation set the angles."""
    orient = str(((record.metadata or {}).get("raw_header") or {}).get("Orient") or "").strip()
    fixed = str((record.metadata or {}).get("fixed_energy") or "").strip()
    month = str(record.date or "")[:7]
    parts = [part for part in (month, f"[{orient}]" if orient else "", fixed) if part]
    return " ".join(parts) or "unlabeled session"


def _pool_inelastic_points(records: list[DashboardScanRecord]) -> list[dict[str, Any]]:
    """Pool (Q, E, I/M) points from every inelastic scan with HKLE columns."""
    points: list[dict[str, Any]] = []
    for record in records:
        if record.measurement_kind != "inelastic":
            continue
        cols = _columns(record)
        h, k, l, e = cols.get("H"), cols.get("K"), cols.get("L"), cols.get("E")
        detector, monitor = cols.get("Detector"), cols.get("Monitor")
        if not (h and k and l and e and detector):
            continue
        session = _session_label(record)
        times = cols.get("Time")
        n = min(len(h), len(k), len(l), len(e), len(detector), len(monitor or detector))
        for i in range(n):
            if None in (h[i], k[i], l[i], e[i], detector[i]):
                continue
            mon = float(monitor[i]) if monitor and monitor[i] else 1.0
            points.append(
                {
                    "scan_id": record.scan_id,
                    "date": record.date,
                    "session": session,
                    "H": float(h[i]),
                    "K": float(k[i]),
                    "L": float(l[i]),
                    "E": float(e[i]),
                    "intensity": float(detector[i]) / max(mon, 1e-12),
                    "count_time_s": float(times[i]) if times and i < len(times) and times[i] else None,
                }
            )
    return points


def _q_axis(points: list[dict[str, Any]]) -> tuple[str, str]:
    """In-plane coordinate: 'HH' when H tracks K, else whichever of H/K varies.

    Longitudinal runs along K are ordinary on BT7; assuming H always carries
    the in-plane variation collapsed them to a zero-span axis."""
    if not points:
        return ("H", "H (r.l.u.)")
    tracks = sum(1 for p in points if abs(p["H"] - p["K"]) < 0.02)
    if tracks / len(points) > 0.8:
        return ("HH", "(H H 0) coordinate")
    if _span([p["K"] for p in points]) > _span([p["H"] for p in points]):
        return ("K", "K (r.l.u.)")
    return ("H", "H (r.l.u.)")


def build_dispersion_map_panels(records: list[DashboardScanRecord], plots_dir: Path, *, min_points: int = 25) -> dict[str, Any]:
    """Assemble E-vs-Q maps per session (never pooling energies/orientations).

    Maps are driven by the Q-varying mapping scans; constant-Q energy scans
    are included only where they fall inside the mapped Q range. Rendering is
    a log-color scatter on the natural measurement grid. Fails soft without
    matplotlib."""
    points = _pool_inelastic_points(records)
    if not points:
        return {"status": "no_inelastic_points", "panels": []}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np
    except Exception:
        return {"status": "matplotlib_unavailable", "panels": [], "point_count": len(points)}
    plots_dir.mkdir(parents=True, exist_ok=True)
    axis_key, axis_label = _q_axis(points)
    for p in points:
        p["Qip"] = 0.5 * (p["H"] + p["K"]) if axis_key == "HH" else p[axis_key]
    scans: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in points:
        scans.setdefault((p["session"], p["scan_id"]), []).append(p)
    scan_roles: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in scans.items():
        qip_span, l_span = _span([p["Qip"] for p in group]), _span([p["L"] for p in group])
        role = "qip_scan" if qip_span > 0.05 and l_span <= 0.05 else "l_scan" if l_span > 0.05 and qip_span <= 0.05 else "e_scan" if qip_span <= 0.05 and l_span <= 0.05 else "diagonal"
        scan_roles[key] = {"role": role, "median_l": _median([p["L"] for p in group]), "median_qip": _median([p["Qip"] for p in group])}
    panels: list[dict[str, Any]] = []

    def render(group: list[dict[str, Any]], x_key: str, x_label: str, title: str, filename: str) -> None:
        if len(group) < min_points:
            return
        xs = np.array([p[x_key] for p in group])
        es = np.array([p["E"] for p in group])
        intensities = np.array([p["intensity"] for p in group])
        positives = intensities[intensities > 0]
        if positives.size == 0:
            return
        floor = float(positives.min()) / 10.0
        clipped = np.clip(intensities, floor, None)
        vmin = float(np.percentile(positives, 5))
        vmax = float(np.percentile(positives, 99.5))
        if vmax <= vmin:
            vmax = vmin * 10.0
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        scatter = ax.scatter(xs, es, c=clipped, s=34, cmap="turbo", norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(scatter, ax=ax, label="Detector / Monitor (log color)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Energy transfer E (meV)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.savefig(plots_dir / filename, dpi=140)
        plt.close(fig)
        scan_ids = sorted({p["scan_id"] for p in group})
        panels.append({"title": title, "png": f"plots/{filename}", "point_count": len(group), "scan_count": len(scan_ids), "example_scans": scan_ids[:8]})

    sessions = sorted({key[0] for key in scans})
    for session_index, session in enumerate(sessions, start=1):
        session_scans = {key: group for key, group in scans.items() if key[0] == session}
        tag = f"s{session_index}"
        # E vs in-plane at each fixed L, driven by the Q-mapping scans.
        by_l: dict[float, dict[str, list]] = {}
        for key, group in session_scans.items():
            meta = scan_roles[key]
            if meta["median_l"] is None:
                continue
            bucket = by_l.setdefault(round(meta["median_l"], 1), {"map": [], "escan": []})
            if meta["role"] == "qip_scan":
                bucket["map"].extend(group)
            elif meta["role"] == "e_scan":
                bucket["escan"].append(group)
        for l_value, bucket in sorted(by_l.items()):
            group = list(bucket["map"])
            if group:
                qip_lo = min(p["Qip"] for p in group) - 0.02
                qip_hi = max(p["Qip"] for p in group) + 0.02
                for escan in bucket["escan"]:
                    if escan and qip_lo <= escan[0]["Qip"] <= qip_hi:
                        group.extend(escan)
            elif len(bucket["escan"]) >= 3:
                group = [p for escan in bucket["escan"] for p in escan]
            if group and _span([p["Qip"] for p in group]) > 0.1:
                render(group, "Qip", axis_label, f"E vs {axis_key} at L={l_value:g} - {session}; {len(group)} points", f"dispersion_{tag}_{_safe_id(axis_key)}_L{_safe_id(f'{l_value:g}')}.png")
        # E vs L at each fixed in-plane value, driven by L-scans.
        by_qip: dict[float, list[dict[str, Any]]] = {}
        for key, group in session_scans.items():
            meta = scan_roles[key]
            if meta["role"] == "l_scan" and meta["median_qip"] is not None:
                by_qip.setdefault(round(meta["median_qip"], 1), []).extend(group)
        for q_value, group in sorted(by_qip.items()):
            if _span([p["L"] for p in group]) > 0.1:
                render(group, "L", "L (r.l.u.)", f"E vs L at {axis_key}={q_value:g} - {session}; {len(group)} points", f"dispersion_{tag}_L_{_safe_id(axis_key)}{_safe_id(f'{q_value:g}')}.png")
    return {
        "status": "ok" if panels else "no_extended_q_coverage",
        "panels": panels,
        "point_count": len(points),
        "in_plane_axis": axis_key,
        "sessions": sessions,
    }


def _write_dispersion_points_csv(records: list[DashboardScanRecord], path: Path) -> int:
    points = _pool_inelastic_points(records)
    if not points:
        return 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scan_id", "date", "H", "K", "L", "E", "intensity"])
        writer.writeheader()
        for p in points:
            writer.writerow({key: p[key] for key in writer.fieldnames})
    return len(points)


def dispersion_map_report(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the dispersion measured so far into E-vs-Q maps."""
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/dispersion_map_report")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, roots = _load_records(arguments)
    if not roots:
        report = {"workflow_name": "dispersion-map-report", "status": "missing_input", "missing_inputs": ["data_root"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_dispersion, json_filename="dispersion_map_report.json", markdown_filename="dispersion_map_report.md", html_filename="dispersion_map_report.html")
        return report
    maps = build_dispersion_map_panels(records, output_dir / "plots")
    point_count = _write_dispersion_points_csv(records, output_dir / "dispersion_points.csv")
    report = {
        "workflow_name": "dispersion-map-report",
        "status": "ok" if maps.get("panels") else maps.get("status", "no_inelastic_points"),
        "data_roots": roots,
        "scan_count": len(records),
        "inelastic_point_count": point_count,
        "in_plane_axis": maps.get("in_plane_axis"),
        "panels": maps.get("panels") or [],
        "warnings": [] if maps.get("panels") else [f"No dispersion panels produced: {maps.get('status')}"],
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=";".join(roots), method="dispersion_map_report").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_dispersion, json_filename="dispersion_map_report.json", markdown_filename="dispersion_map_report.md", html_filename="dispersion_map_report.html")
    return report


def _markdown_dispersion(report: dict[str, Any]) -> str:
    lines = [
        "# Dispersion Map Report",
        "",
        f"Status: {report.get('status')} | scans: {report.get('scan_count', 0)} | pooled inelastic points: {report.get('inelastic_point_count', 0)}",
        "",
    ]
    for panel in report.get("panels") or []:
        lines.append(f"## {panel['title']}")
        lines.append(f"![{panel['title']}]({panel['png']})")
        lines.append(f"- {panel['point_count']} points from {panel['scan_count']} scan(s), e.g. {', '.join(panel['example_scans'][:4])}")
        lines.append("")
    for warning in report.get("warnings") or []:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _order_parameter_section(records: list[DashboardScanRecord], plots_dir: Path) -> dict[str, Any]:
    """Derived I(T) curves per Q point from temperature-series scans."""
    series = [record for record in records if _overnight_family(record) == "temperature_dependence_or_order_parameter"]
    if not series:
        return {"status": "no_temperature_series", "curves": []}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {"status": "matplotlib_unavailable", "curves": []}
    plots_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[DashboardScanRecord]] = {}
    for record in series:
        cols = _columns(record)
        q_label = "Q=(" + " ".join(
            f"{_median(cols.get(axis) or []):.2f}" if _median(cols.get(axis) or []) is not None else "?" for axis in ("H", "K", "L")
        ) + ")"
        groups.setdefault(q_label, []).append(record)
    curves = []
    for q_label, members in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plotted = 0
        for record in members:
            cols = _columns(record)
            temps, det, mon = cols.get("Temp"), cols.get("Detector"), cols.get("Monitor")
            if not temps or not det:
                plot = record.plot_data or {}
                if plot.get("x_column") == "Temp":
                    temps, det, mon = plot.get("x"), plot.get("y"), None
            if not temps or not det:
                continue
            pairs = sorted(
                (float(t), float(d) / max(float(mon[i]) if mon and mon[i] else 1.0, 1e-12))
                for i, (t, d) in enumerate(zip(temps, det))
                if t is not None and d is not None
            )
            if len(pairs) < 3:
                continue
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker=".", ms=4, lw=1.0, label=str(record.scan_id)[:50])
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("I/monitor")
        ax.set_title(f"Order parameter candidate at {q_label}")
        ax.legend(fontsize="x-small")
        filename = f"order_parameter_{_safe_id(q_label)}.png"
        fig.savefig(plots_dir / filename, dpi=140)
        plt.close(fig)
        curves.append({"q_label": q_label, "png": f"plots/{filename}", "scan_count": len(members)})
    return {"status": "ok" if curves else "no_plottable_temperature_series", "curves": curves}


def _derive_alignment_state_jsons(
    roots: list[str],
    output_dir: Path,
    *,
    alignment_peak_report: Any,
    max_files: int,
    notes: list[str],
) -> list[str]:
    """Derive the alignment-state chain when no state JSON was supplied.

    Reflection grouping needs the alignment-state-machine's assignments; the
    deterministic chain (alignment-corpus-index -> alignment-state-machine)
    runs from the same data roots, so derive it fail-soft instead of
    reporting every scan unassigned. Software-change notes derive from the
    console logs alone; HKL assignments additionally need an alignment peak
    planning report (structure-dependent), which cannot be guessed here.
    """
    from tas_copilot_core.workflows.alignment_corpus_index import alignment_corpus_index
    from tas_copilot_core.workflows.alignment_state import alignment_state_machine

    derived: list[str] = []
    for index, root in enumerate(roots):
        session_dir = output_dir / "derived_alignment_state" / f"session_{index:02d}"
        try:
            index_report = alignment_corpus_index({"data_root": root, "output_dir": str(session_dir / "index"), "max_files": max_files})
            events_jsonl = (index_report.get("artifacts") or {}).get("events_jsonl")
            if not events_jsonl:
                notes.append(f"alignment-state derivation: no events index for {root} (status {index_report.get('status')}).")
                continue
            machine_args: dict[str, Any] = {"events_jsonl": events_jsonl, "output_dir": str(session_dir / "state")}
            if alignment_peak_report:
                machine_args["alignment_peak_report"] = alignment_peak_report
            machine = alignment_state_machine(machine_args)
            state_json = (machine.get("artifacts") or {}).get("json")
            if state_json:
                derived.append(str(state_json))
            else:
                notes.append(f"alignment-state derivation: state machine produced no JSON for {root} (status {machine.get('status')}).")
        except Exception as exc:  # noqa: BLE001 - derivation is best-effort; the report stays honest without it.
            notes.append(f"alignment-state derivation failed for {root}: {type(exc).__name__}: {exc}")
    return derived


_ASSIGNMENT_HELP = (
    "Reflection grouping needs HKL assignments from the alignment-state-machine's A4/d-spacing checks, "
    "which require an alignment peak planning report (structure-dependent). Supply alignment_state_json "
    "from a prior alignment-state-machine run, or alignment_peak_report (a suggest-alignment-peaks output) "
    "so assignments can be derived here."
)


def experiment_story_report(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Where the experiment stands, grouped semantically: alignment by
    reflection, order parameters as derived I(T), dispersions as E-vs-Q maps."""
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/experiment_story_report")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(arguments.get("sections") or ["alignment", "order_parameters", "dispersion"])
    records, roots = _load_records(arguments)
    if not roots:
        report = {"workflow_name": "experiment-story-report", "status": "missing_input", "missing_inputs": ["data_root"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_story, json_filename="experiment_story_report.json", markdown_filename="experiment_story_report.md", html_filename="experiment_story_report.html")
        return report
    plots_dir = output_dir / "plots"
    sections: dict[str, Any] = {}
    if "alignment" in wanted:
        state_jsons = arguments.get("alignment_state_json")
        alignment_peak_report = arguments.get("alignment_peak_report")
        derivation_notes: list[str] = []
        if state_jsons:
            assignments_source = "supplied"
        else:
            state_jsons = _derive_alignment_state_jsons(
                roots,
                output_dir,
                alignment_peak_report=alignment_peak_report,
                max_files=int(arguments.get("max_files") or 1000),
                notes=derivation_notes,
            )
            if not state_jsons:
                assignments_source = "none"
            elif alignment_peak_report:
                assignments_source = "derived"
            else:
                assignments_source = "derived_without_planning_report"
        alignment_context = _load_alignment_plot_context(state_jsons)
        alignment_records = [record for record in records if _overnight_family(record) in _ALIGNMENT_PLOT_FAMILIES]
        plot_entries = _write_overnight_series_plots(alignment_records, output_dir, alignment_context=alignment_context)
        assignments = (alignment_context or {}).get("assignments") or {}
        by_reflection: dict[str, int] = {}
        for record in alignment_records:
            label = _reflection_label(record, assignments) if assignments else "unassigned"
            by_reflection[label] = by_reflection.get(label, 0) + 1
        if not assignments and alignment_records:
            derivation_notes.append(_ASSIGNMENT_HELP)
        sections["alignment"] = {
            "status": "ok" if plot_entries else "no_alignment_scans",
            "scan_count": len(alignment_records),
            "scans_by_reflection": by_reflection,
            "assignments_source": assignments_source,
            "notes": derivation_notes,
            "plots": [entry for entries in plot_entries.values() for entry in entries],
            "software_changes": (alignment_context or {}).get("software_changes") or [],
        }
    if "order_parameters" in wanted:
        sections["order_parameters"] = _order_parameter_section(records, plots_dir)
    if "dispersion" in wanted:
        maps = build_dispersion_map_panels(records, plots_dir)
        sections["dispersion"] = {"status": maps.get("status"), "panels": maps.get("panels") or [], "point_count": maps.get("point_count", 0)}
    report = {
        "workflow_name": "experiment-story-report",
        "status": "ok",
        "data_roots": roots,
        "scan_count": len(records),
        "sections": sections,
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=";".join(roots), method="experiment_story_report").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_story, json_filename="experiment_story_report.json", markdown_filename="experiment_story_report.md", html_filename="experiment_story_report.html")
    return report


def _markdown_story(report: dict[str, Any]) -> str:
    lines = ["# Experiment Story Report", "", f"Status: {report.get('status')} | scans: {report.get('scan_count', 0)}", ""]
    sections = report.get("sections") or {}
    alignment = sections.get("alignment")
    if alignment:
        lines += ["## Alignment", ""]
        counts = alignment.get("scans_by_reflection") or {}
        if counts:
            lines.append("- Scans by reflection: " + ", ".join(f"{label}: {count}" for label, count in sorted(counts.items())))
        source = alignment.get("assignments_source")
        if source:
            lines.append(f"- Assignments source: {source}")
        for note in alignment.get("notes") or []:
            lines.append(f"- Note: {note}")
        for entry in alignment.get("plots") or []:
            lines.append(f"### {entry.get('label')}")
            lines.append(f"![{entry.get('label')}]({entry.get('png')})")
        for note in (alignment.get("software_changes") or [])[:12]:
            lines.append(f"- Software change (console log): `{note}`")
        lines.append("")
    order = sections.get("order_parameters")
    if order:
        lines += ["## Order Parameters (derived I vs T)", ""]
        if not order.get("curves"):
            lines.append(f"- {order.get('status')}")
        for curve in order.get("curves") or []:
            lines.append(f"### {curve['q_label']} ({curve['scan_count']} scan(s))")
            lines.append(f"![{curve['q_label']}]({curve['png']})")
        lines.append("")
    dispersion = sections.get("dispersion")
    if dispersion:
        lines += ["## Dispersion (assembled from all inelastic scans)", ""]
        if not dispersion.get("panels"):
            lines.append(f"- {dispersion.get('status')}")
        for panel in dispersion.get("panels") or []:
            lines.append(f"### {panel['title']}")
            lines.append(f"![{panel['title']}]({panel['png']})")
        lines.append("")
    return "\n".join(lines) + "\n"
