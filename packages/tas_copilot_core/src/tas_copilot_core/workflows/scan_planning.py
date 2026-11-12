"""Dispersion-mapping scan planning: propose the next scans, with steps.

Complements dispersion-map-report (what has been measured) with "what to
measure next": coverage-aware constant-E line scans whose steps come from the
resolution calculator (per-scan, through the registry), whose endpoints are
kinematics-checked (ExPlanner port), and whose alignment-scan companions use
step/range heuristics mined from real BT-7 FindPeak practice.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord, ToolCall
from tas_copilot_contracts.tools import ToolResultStatus
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.alignment_state import (
    _DEFAULT_A4_MIN_DEG,
    _EXPLANNER_BT7_A4_MAX_LIMITS_DEG,
    _EXPLANNER_BT7_EI_LIMITS_PG002_MEV,
)
from tas_copilot_core.workflows.common import default_runner
from tas_copilot_core.workflows.experiment_story import (
    _load_records,
    _median,
    _pool_inelastic_points,
    _q_axis,
    _session_label,
    _span,
)


def default_alignment_heuristics_path() -> Path:
    return Path(__file__).resolve().parents[5] / "configs" / "scan_plan_heuristics" / "bt7_alignment_scan_defaults.yaml"


def _load_alignment_heuristics(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser() if path_value else default_alignment_heuristics_path()
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _latest_session_context(records: list) -> dict[str, Any]:
    """Lattice, orientation, and fixed energy from the newest scan headers."""
    latest = None
    for record in records:
        if latest is None or str(record.date or "") > str(latest.date or ""):
            latest = record
    if latest is None:
        return {}
    raw = (latest.metadata or {}).get("raw_header") or {}
    lattice_parts = str(raw.get("Lattice") or "").split()
    orient_parts = str(raw.get("Orient") or "").split()
    fixed = str((latest.metadata or {}).get("fixed_energy") or "")
    ef = None
    for token in fixed.split():
        try:
            ef = float(token)
        except ValueError:
            continue
    context: dict[str, Any] = {"session": _session_label(latest)}
    if len(lattice_parts) == 6:
        context["lattice"] = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), (float(p) for p in lattice_parts)))
    if len(orient_parts) == 6:
        context["orient1"] = [float(p) for p in orient_parts[:3]]
        context["orient2"] = [float(p) for p in orient_parts[3:]]
    if ef:
        context["ef_mev"] = ef
    return context


def _reciprocal_modulus(lattice: dict[str, Any], hkl: list[float]) -> float | None:
    """|Q| in 1/Angstrom of an HKL vector, through the ubmatrix registry tool."""
    try:
        result = default_runner().run(
            ToolCall(
                tool_namespace="tas.ubmatrix",
                tool_name="calculate_reciprocal_angle",
                arguments={"lattice": lattice, "v1": hkl, "v2": hkl},
                caller_workflow="dispersion-scan-plan",
            )
        )
        if result.status == ToolResultStatus.success:
            return float(result.output["v1_modulus_inv_angstrom"])
    except Exception:
        return None
    return None


def _resolution_q_fwhm(context: dict[str, Any], hkl: list[float], energy_mev: float, *, collimation: str) -> dict[str, Any]:
    """Q/E FWHM at one (Q, E) point via the registered resolution tool."""
    lattice = dict(context.get("lattice") or {})
    lattice["orient1"] = [context.get("orient1") or [1, 0, 0]]
    lattice["orient2"] = [context.get("orient2") or [0, 0, 1]]
    try:
        result = default_runner().run(
            ToolCall(
                tool_namespace="tas.resolution",
                tool_name="calculate_resolution",
                arguments={
                    "lattice": lattice,
                    "q_point": {"h": hkl[0], "k": hkl[1], "l": hkl[2]},
                    "energy_transfer": {"value": energy_mev, "unit": "meV"},
                    "instrument_config": {
                        "fixed_energy": {"value": context.get("ef_mev") or 14.7},
                        "fixed_energy_mode": "fixed_final_energy",
                        "collimation": collimation,
                    },
                },
                caller_workflow="dispersion-scan-plan",
            )
        )
    except Exception as exc:
        return {"status": "resolution_tool_error", "error": str(exc)}
    if result.status != ToolResultStatus.success:
        return {"status": f"tool_{result.status.value}"}
    fwhm = ((result.output or {}).get("resolution") or {}).get("fwhm") or (result.output or {}).get("fwhm") or {}

    def value(axis: str) -> float | None:
        entry = fwhm.get(axis)
        if isinstance(entry, dict):
            return float(entry.get("value"))
        return float(entry) if entry is not None else None

    return {"status": "ok", "q_fwhm_inv_angstrom": value("axis_0"), "e_fwhm_mev": value("axis_3")}


def _check_reachability(context: dict[str, Any], q_moduli: list[float], energies: list[float]) -> dict[str, Any]:
    try:
        result = default_runner().run(
            ToolCall(
                tool_namespace="tas.psd_planner",
                tool_name="check_accessibility",
                arguments={
                    "ef_mev": float(context.get("ef_mev") or 14.7),
                    "mono_reflection": "PG(002)",
                    "ana_reflection": "PG(002)",
                    "ei_limits_pg002_mev": [float(v) for v in _EXPLANNER_BT7_EI_LIMITS_PG002_MEV],
                    "a4_max_limits_deg": [float(v) for v in _EXPLANNER_BT7_A4_MAX_LIMITS_DEG],
                    "a4_min_deg": float(_DEFAULT_A4_MIN_DEG),
                    "q_moduli_inv_ang": q_moduli,
                    "energy_transfers_mev": energies,
                },
                caller_workflow="dispersion-scan-plan",
            )
        )
    except Exception as exc:
        return {"status": "kinematics_tool_error", "error": str(exc)}
    if result.status != ToolResultStatus.success:
        return {"status": f"tool_{result.status.value}"}
    return {"status": "ok", "accessibility": (result.output or {}).get("accessibility") or {}}


def _coverage(points: list[dict[str, Any]], session: str) -> dict[str, Any]:
    """Measured (path, E-row) coverage for the planning session."""
    session_points = [p for p in points if p["session"] == session] or points
    axis_key, _ = _q_axis(session_points)
    for p in session_points:
        p["Qip"] = 0.5 * (p["H"] + p["K"]) if axis_key == "HH" else p["H"]
    rows: dict[float, dict[float, list[float]]] = {}
    for p in session_points:
        l_key = round(p["L"], 1)
        e_key = round(p["E"], 1)
        rows.setdefault(l_key, {}).setdefault(e_key, []).append(p["Qip"])
    return {"axis": axis_key, "rows": rows, "point_count": len(session_points)}


def _mapping_practice(points: list[dict[str, Any]], session: str) -> dict[str, Any]:
    """How this experiment actually mapped: scan-type census, observed steps,
    and per-point counting times, mined from the session's own scans."""
    import statistics

    session_points = [p for p in points if p["session"] == session] or points
    scans: dict[str, list[dict[str, Any]]] = {}
    for p in session_points:
        scans.setdefault(p["scan_id"], []).append(p)
    census = {"constant_E_q_scan": 0, "l_scan": 0, "constant_Q_e_scan": 0, "diagonal": 0}
    steps: list[float] = []
    npts: list[int] = []
    count_times: list[float] = []
    zone_qscan_counts: dict[float, int] = {}
    for group in scans.values():
        qip_span, l_span = _span([p["Qip"] for p in group]), _span([p["L"] for p in group])
        if qip_span > 0.05 and l_span <= 0.05:
            census["constant_E_q_scan"] += 1
            ordered = sorted(p["Qip"] for p in group)
            if len(ordered) > 2:
                steps.append(statistics.median(round(b - a, 4) for a, b in zip(ordered, ordered[1:])))
                npts.append(len(ordered))
            median_l = _median([p["L"] for p in group])
            if median_l is not None:
                zone_qscan_counts[round(median_l, 1)] = zone_qscan_counts.get(round(median_l, 1), 0) + 1
        elif l_span > 0.05 and qip_span <= 0.05:
            census["l_scan"] += 1
        elif qip_span <= 0.05 and l_span <= 0.05:
            census["constant_Q_e_scan"] += 1
        else:
            census["diagonal"] += 1
        times = [p.get("count_time_s") for p in group if p.get("count_time_s")]
        if times:
            count_times.append(statistics.median(times))
    return {
        "scan_type_census": census,
        "observed_step_rlu": round(statistics.median(steps), 4) if steps else None,
        "observed_npts": int(statistics.median(npts)) if npts else None,
        "observed_count_time_s": round(statistics.median(count_times), 1) if count_times else None,
        "zone_qscan_counts": zone_qscan_counts,
    }


def dispersion_scan_plan(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Propose the constant-E scans that extend the measured dispersion.

    Steps come from the resolution calculator (half the projected Q FWHM);
    endpoints are kinematics-checked; alignment-scan companions carry
    precedent-mined step/range heuristics. Planning evidence only.
    """
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/dispersion_scan_plan")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, roots = _load_records(arguments)
    if not roots:
        report = {"workflow_name": "dispersion-scan-plan", "status": "missing_input", "missing_inputs": ["data_root"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_plan, json_filename="dispersion_scan_plan.json", markdown_filename="dispersion_scan_plan.md", html_filename="dispersion_scan_plan.html")
        return report
    context = _latest_session_context(records)
    for key in ("lattice", "orient1", "ef_mev"):
        if arguments.get(key) is not None:
            context[key] = arguments[key]
    if arguments.get("orient2") is not None:
        context["orient2"] = arguments["orient2"]
    missing = [key for key in ("lattice", "orient1", "orient2", "ef_mev") if not context.get(key)]
    if missing:
        report = {"workflow_name": "dispersion-scan-plan", "status": "missing_input", "missing_inputs": missing, "note": "Could not derive lattice/orientation/fixed energy from the scan headers; supply them explicitly."}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_plan, json_filename="dispersion_scan_plan.json", markdown_filename="dispersion_scan_plan.md", html_filename="dispersion_scan_plan.html")
        return report

    points = _pool_inelastic_points(records)
    coverage = _coverage(points, str(context.get("session") or ""))
    practice = _mapping_practice(points, str(context.get("session") or ""))
    axis_key = coverage["axis"]
    # Numeric arc-minutes for the resolution tool; "open" approximated as 60'.
    collimation = str(arguments.get("collimation") or "60-80-80-120")
    e_min = float(arguments.get("e_min_mev", 2.0))
    e_max = float(arguments.get("e_max_mev", 30.0))
    e_step = float(arguments.get("e_row_step_mev", 2.0))
    q_lo = float(arguments.get("q_min_rlu", -0.5))
    q_hi = float(arguments.get("q_max_rlu", 0.5))
    warnings: list[str] = []

    # Zones to plan: ranked by how many constant-E mapping scans the
    # experimenters ran there (mapping intent), not raw point counts, so a
    # targeted side check (e.g. a gap check at high L) does not outrank the
    # primary zone. Only the primary zone is planned unless --l-values asks
    # for more.
    zone_ranks = practice.get("zone_qscan_counts") or {l: sum(len(v) for v in rows.items()) for l, rows in coverage["rows"].items()}
    ranked_zones = sorted(zone_ranks, key=lambda l_value: -zone_ranks[l_value])
    l_values = [float(v) for v in (arguments.get("l_values") or ranked_zones[:1] or [2.0])]
    unit_hkl = [1.0, 1.0, 0.0] if axis_key == "HH" else [1.0, 0.0, 0.0]
    modulus_per_rlu = _reciprocal_modulus(dict(context["lattice"]), unit_hkl)
    if modulus_per_rlu is None:
        warnings.append("tas.ubmatrix unavailable: Q steps reported in 1/Angstrom converted with a heuristic 0.025 r.l.u. step.")

    scans: list[dict[str, Any]] = []
    count_time_s = practice.get("observed_count_time_s")
    for l_value in l_values:
        covered_rows = coverage["rows"].get(round(l_value, 1), {})
        # Rows already measured with meaningful Q extent anywhere in the zone:
        # a proposed row must not interleave an existing deliberately-spaced
        # ladder (e.g. odd-E rows every 2 meV), so it must sit farther than
        # 0.75*e_step from every measured row.
        measured_rows = sorted(e for e, values in covered_rows.items() if _span(values) > 0.3 * (q_hi - q_lo))
        for e_row in [round(e_min + i * e_step, 1) for i in range(int((e_max - e_min) / e_step) + 1)]:
            nearest = min((abs(e_row - m) for m in measured_rows), default=None)
            if nearest is not None and nearest < 0.75 * e_step:
                continue  # row measured (or an existing ladder row is adjacent)
            reason = "extends_above_measured_rows" if measured_rows and e_row > max(measured_rows) else "fills_unmeasured_row"
            existing = covered_rows.get(e_row) or []
            q_center = 0.5 * (q_lo + q_hi)
            center_hkl = [q_center, q_center, l_value] if axis_key == "HH" else [q_center, 0.0, l_value]
            resolution = _resolution_q_fwhm(context, center_hkl, e_row, collimation=collimation)
            resolution_fwhm_rlu = (
                round(resolution["q_fwhm_inv_angstrom"] / modulus_per_rlu, 4)
                if resolution.get("status") == "ok" and resolution.get("q_fwhm_inv_angstrom") and modulus_per_rlu
                else None
            )
            if practice.get("observed_step_rlu"):
                # Follow the experiment's own demonstrated practice (observed
                # steps ran ~1.2x the resolution FWHM on SrMn2Sb2).
                step_rlu = practice["observed_step_rlu"]
                step_source = "observed_practice"
            elif resolution_fwhm_rlu:
                step_rlu = round(max(resolution_fwhm_rlu, 0.005), 4)
                step_source = "resolution_fwhm"
            else:
                step_rlu = 0.025
                step_source = "heuristic_default"
                if resolution.get("status") != "ok":
                    warnings.append(f"resolution unavailable at E={e_row}: {resolution.get('status')}")
            npts = int((q_hi - q_lo) / step_rlu) + 1
            scan = {
                "kind": "constant_E_q_scan",
                "path": f"{axis_key} at L={l_value:g}",
                "energy_mev": e_row,
                "reason": reason,
                "q_min_rlu": q_lo,
                "q_max_rlu": q_hi,
                "step_rlu": step_rlu,
                "step_source": step_source,
                "resolution_q_fwhm_rlu": resolution_fwhm_rlu,
                "npts": npts,
                "count_time_s_per_point": count_time_s,
                "estimated_minutes": round(npts * count_time_s / 60.0, 1) if count_time_s else None,
                "e_fwhm_mev": resolution.get("e_fwhm_mev"),
                "already_covered_qip": sorted(set(round(v, 2) for v in existing))[:10],
            }
            if modulus_per_rlu:
                endpoints = []
                for q_value in (q_lo, q_center, q_hi):
                    hkl = [q_value, q_value, l_value] if axis_key == "HH" else [q_value, 0.0, l_value]
                    modulus = _reciprocal_modulus(dict(context["lattice"]), hkl)
                    if modulus:
                        endpoints.append(modulus)
                if endpoints:
                    reach = _check_reachability(context, endpoints, [e_row])
                    if reach.get("status") == "ok":
                        points_out = (reach["accessibility"].get("points") or [])
                        scan["reachable"] = all(bool(p.get("accessible", p.get("reachable", True))) for p in points_out) if points_out else None
                    else:
                        scan["reachability_status"] = reach.get("status")
            scans.append(scan)
    heuristics = _load_alignment_heuristics(arguments.get("alignment_heuristics"))
    report = {
        "workflow_name": "dispersion-scan-plan",
        "status": "ok" if scans else "already_covered",
        "data_roots": roots,
        "planning_session": context.get("session"),
        "context": {key: context.get(key) for key in ("lattice", "orient1", "orient2", "ef_mev")},
        "collimation_assumed": collimation,
        "coverage_point_count": coverage["point_count"],
        "mapping_practice": practice,
        "planned_zones": l_values,
        "estimated_total_hours": round(sum(s.get("estimated_minutes") or 0 for s in scans) / 60.0, 1) if count_time_s else None,
        "proposed_scans": scans,
        "alignment_scan_heuristics": heuristics.get("alignment_scan_defaults") or {},
        "alignment_heuristics_status": heuristics.get("status"),
        "warnings": sorted(set(warnings)),
        "provenance": [ProvenanceRecord(source_type="data_root", source_id=";".join(roots), method="dispersion_scan_plan").model_dump(mode="json")],
    }
    csv_path = output_dir / "dispersion_scan_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "path", "energy_mev", "q_min_rlu", "q_max_rlu", "step_rlu", "step_source", "npts", "reachable"])
        writer.writeheader()
        for scan in scans:
            writer.writerow({key: scan.get(key) for key in writer.fieldnames})
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown_plan, json_filename="dispersion_scan_plan.json", markdown_filename="dispersion_scan_plan.md", html_filename="dispersion_scan_plan.html")
    report["artifacts"]["csv"] = str(csv_path)
    return report


def _markdown_plan(report: dict[str, Any]) -> str:
    lines = [
        "# Dispersion Scan Plan",
        "",
        f"Status: {report.get('status')} | session: {report.get('planning_session')} | measured points: {report.get('coverage_point_count', 0)}",
        "",
    ]
    if report.get("missing_inputs"):
        lines.append(f"- Missing inputs: {', '.join(report['missing_inputs'])}")
        return "\n".join(lines) + "\n"
    practice = report.get("mapping_practice") or {}
    if practice.get("scan_type_census"):
        census = practice["scan_type_census"]
        lines += [
            "## How this experiment actually mapped",
            "",
            f"- Scan types: {census.get('constant_E_q_scan', 0)} constant-E Q-scans, {census.get('l_scan', 0)} L-scans, {census.get('constant_Q_e_scan', 0)} constant-Q E-scans (constant-E cuts resolve steep branches; E-scans suit flat regions).",
            f"- Observed practice: step {practice.get('observed_step_rlu')} r.l.u., ~{practice.get('observed_npts')} pts/scan, {practice.get('observed_count_time_s')} s/point.",
            "",
        ]
    scans = report.get("proposed_scans") or []
    if scans:
        total = report.get("estimated_total_hours")
        lines += [
            f"## Proposed constant-E scans (zones: {', '.join(f'L={z:g}' for z in report.get('planned_zones') or [])}"
            + (f"; ~{total} h total at observed counting)" if total else ")"),
            "",
            "| path | E (meV) | reason | Q range (r.l.u.) | step | pts | ~min | step source | reachable |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for scan in scans:
            lines.append(
                f"| {scan['path']} | {scan['energy_mev']} | {scan.get('reason', '')} | {scan['q_min_rlu']} to {scan['q_max_rlu']} | {scan['step_rlu']} | {scan['npts']} | {scan.get('estimated_minutes', '')} | {scan['step_source']} | {scan.get('reachable', '?')} |"
            )
        lines.append("")
    heuristics = report.get("alignment_scan_heuristics") or {}
    if heuristics:
        lines += [
            f"## Alignment-scan defaults ({report.get('alignment_heuristics_status')})",
            "",
            "| device | step | span | pts | from |",
            "| --- | --- | --- | --- | --- |",
        ]
        for device, entry in heuristics.items():
            step = entry.get("step_deg") or entry.get("step_mm")
            span = entry.get("span_deg") or entry.get("span_mm")
            lines.append(f"| {device} | {step} | {span} | {entry.get('npts')} | {entry.get('sample_count')} real scans |")
        lines.append("")
    for warning in report.get("warnings") or []:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"
