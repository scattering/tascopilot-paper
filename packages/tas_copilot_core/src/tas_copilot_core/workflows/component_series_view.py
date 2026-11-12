"""Validated display of externally reviewed component intensities; never a fitter.

The experiment-specific model and interpretation live in annotation data. This
module preserves its preferred covariance and separates alternative decompositions
from independent observations. No network, instrument or analysis jobs are used.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#999933")
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "h")


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _stamp(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("snapshot_utc must include a timezone")
    return result.astimezone(timezone.utc)


def load_component_series_payload(entry: Any, annotation_path: Path, warnings: list[str]) -> dict[str, Any]:
    """Load inline data or one hash-pinned JSON sibling of the annotation YAML."""
    try:
        if not isinstance(entry, dict):
            raise ValueError("component_series must be a mapping")
        if "path" not in entry:
            return deepcopy(entry)
        expected = str(entry.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("component sidecar requires a lowercase SHA256")
        parent = Path(annotation_path).resolve().parent
        path = (parent / str(entry["path"])).resolve()
        if not path.is_relative_to(parent):
            raise ValueError("component sidecar must stay within the annotation directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("component sidecar SHA256 mismatch")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("component sidecar must contain an object")
        payload["annotation_source"] = {"file": str(entry["path"]), "sha256": expected}
        return payload
    except (OSError, ValueError, TypeError) as exc:
        message = f"reviewed component data unavailable: {exc}"
        warnings.append(message)
        return {"load_error": message}


def validate_component_series(payload: dict[str, Any], experiment_id: str, records: list[dict[str, Any]],
                              warnings: list[str], *, cycle: str | None = None) -> dict[str, Any] | None:
    """Validate identity, full covariance, normalization and raw-source freshness."""
    try:
        result = deepcopy(payload)
        if result.get("load_error"):
            raise ValueError(result["load_error"])
        if result.get("schema_version") != 1 or isinstance(result.get("schema_version"), bool):
            raise ValueError("component schema_version must be 1")
        if str(result.get("experiment_id", "")) != str(experiment_id):
            raise ValueError("component experiment_id does not match the current experiment")
        if result.get("cycle") is not None and cycle is not None and str(result["cycle"]) != str(cycle):
            raise ValueError("component cycle does not match the current experiment")
        snapshot = _stamp(result["snapshot_utc"])
        for key in ("title", "area_unit", "normalization_description"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                raise ValueError(f"component {key} is required")
        if not isinstance(result.get("provenance"), list) or not result["provenance"]:
            raise ValueError("component provenance is required")
        for key in ("caveats", "observations"):
            if not isinstance(result.get(key, []), list) or any(not isinstance(x, str) for x in result.get(key, [])):
                raise ValueError(f"component {key} must be a list of text")
        series = result["series"]
        ids = [item["id"] for item in series]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("component series IDs must be unique nonempty strings")
        if any(not isinstance(item.get("label"), str) or not item["label"].strip() for item in series):
            raise ValueError("every component needs a display label")
        points = result["points"]
        by_id = {}
        for point in points:
            identity = point["id"]
            if not isinstance(identity, str) or not identity or identity in by_id:
                raise ValueError("component point IDs must be unique nonempty strings")
            by_id[identity] = point
            if point["series_id"] not in ids or point["fit_choice"] not in {"preferred", "alternative"}:
                raise ValueError("point has an unknown component or fit choice")
            if type(point["scan"]) is not int or point["scan"] < 0:
                raise ValueError("point scan number must be a nonnegative integer")
            if Path(point["raw_file"]).name != point["raw_file"] or not point["raw_file"]:
                raise ValueError("raw_file must be an exact source basename")
            if not re.fullmatch(r"[0-9a-f]{64}", point["raw_sha256"]):
                raise ValueError("point source SHA256 is required")
            for key in ("temperature_k", "temperature_min_k", "temperature_max_k", "area", "ratio", "area_stderr", "ratio_stderr"):
                point[key] = _number(point[key], key, nonnegative=key.endswith("stderr"))
            if not 0 < point["temperature_min_k"] <= point["temperature_k"] <= point["temperature_max_k"]:
                raise ValueError("measured temperature must lie within its positive recorded range")
        preferred = {key: point for key, point in by_id.items() if point["fit_choice"] == "preferred"}
        if not preferred or {p["series_id"] for p in preferred.values()} != set(ids):
            raise ValueError("every component must contain preferred observations")
        preferred_sources = set()
        excluded_numbers = {item["scan"] for item in result.get("excluded_scans", [])}
        excluded_names = {item["raw_file"] for item in result.get("excluded_scans", []) if item.get("raw_file")}
        for point in preferred.values():
            key = (point["series_id"], point["scan"])
            if key in preferred_sources:
                raise ValueError("duplicate preferred observation for one component/source scan")
            preferred_sources.add(key)
            if point["scan"] in excluded_numbers or point["raw_file"] in excluded_names:
                raise ValueError("a preferred observation cannot also be an excluded scan")
        for identity in ids:
            references = {point["reference_id"] for point in points if point["series_id"] == identity}
            if len(references) != 1:
                raise ValueError("each component must use one fixed measured normalization reference")
        for point in points:
            if point["fit_choice"] == "alternative":
                corresponding = [p for p in preferred.values() if p["series_id"] == point["series_id"] and p["scan"] == point["scan"]]
                if len(corresponding) != 1 or any(corresponding[0][key] != point[key] for key in ("raw_file", "raw_sha256")):
                    raise ValueError("an alternative must match one preferred component/source scan")
            reference = preferred.get(point["reference_id"])
            if reference is None or reference["series_id"] != point["series_id"] or reference["area"] <= 0:
                raise ValueError("normalization reference must be a positive preferred point of the same component")
            if not math.isclose(point["ratio"], point["area"] / reference["area"], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("normalized value disagrees with its measured reference")
            if point["id"] == point["reference_id"] and (point["ratio"] != 1 or point["ratio_stderr"] > 1e-12):
                raise ValueError("a reference divided by itself must equal 1 with zero ratio variance")
        covariance = result["covariance"]
        order = covariance["point_order"]
        if len(order) != len(set(order)) or set(order) != set(preferred):
            raise ValueError("covariance point_order must contain every preferred point exactly once and no alternatives")
        matrices = {}
        for kind, error_key in (("area", "area_stderr"), ("ratio", "ratio_stderr")):
            matrix = np.asarray(covariance[kind], dtype=float)
            if matrix.shape != (len(order), len(order)) or not np.isfinite(matrix).all():
                raise ValueError(f"{kind} covariance must be a finite complete square matrix")
            if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
                raise ValueError(f"{kind} covariance must be symmetric")
            scale = max(1.0, float(np.max(np.abs(matrix))))
            if np.linalg.eigvalsh(matrix).min() < -1e-8 * scale:
                raise ValueError(f"{kind} covariance is not positive semidefinite")
            if not np.allclose(np.diag(matrix), [preferred[key][error_key] ** 2 for key in order], rtol=1e-6, atol=1e-9):
                raise ValueError(f"{kind} covariance diagonal disagrees with formal errors")
            matrices[kind] = matrix
        index = {key: i for i, key in enumerate(order)}
        jacobian = np.zeros_like(matrices["area"])
        for key, i in index.items():
            point = preferred[key]; reference = preferred[point["reference_id"]]
            jacobian[i, i] = 1 / reference["area"]
            jacobian[i, index[reference["id"]]] -= point["area"] / reference["area"] ** 2
        if not np.allclose(jacobian @ matrices["area"] @ jacobian.T, matrices["ratio"], rtol=1e-5, atol=1e-8):
            raise ValueError("ratio covariance does not preserve the shared-reference transformation")
        coverage = result.get("coverage", {})
        latest_scan = coverage.get("latest_scan")
        if type(latest_scan) is not int or latest_scan < max(p["scan"] for p in points):
            raise ValueError("coverage.latest_scan must include all represented scans")
        raw = {record.get("name"): record for record in records}
        checked, unavailable, freshness = set(), [], []
        source_hashes = {}
        source_numbers = {}
        for point in points:
            name = point["raw_file"]
            if name in source_hashes and source_hashes[name] != point["raw_sha256"]:
                raise ValueError("inconsistent source hash for one raw file")
            source_hashes[name] = point["raw_sha256"]
            if name in source_numbers and source_numbers[name] != point["scan"]:
                raise ValueError("inconsistent scan number for one raw source")
            source_numbers[name] = point["scan"]
        for name, expected in source_hashes.items():
            record = raw.get(name)
            if record:
                if record.get("number") is not None and record["number"] != source_numbers[name]:
                    raise ValueError(f"raw scan identity disagrees for {name}")
                npts = record.get("npts")
                if npts is not None:
                    for key in ("numPoints", "trajectory.length", "trajectory.scanLength"):
                        planned = record.get("header", {}).get(key)
                        if planned is not None and float(planned) > npts:
                            raise ValueError(f"reviewed source is incomplete: {name} has {npts} of {planned} points")
            path = Path(record["path"]) if record and record.get("path") else None
            if path is None or not path.is_file():
                unavailable.append(name)
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"reviewed source changed: {name}; refresh the reviewed analysis")
            checked.add(name)
        if unavailable:
            freshness.append(f"Raw source verification unavailable for {len(unavailable)} reviewed file(s); values remain a labeled offline snapshot.")
        newer = sorted({int(rec["number"]) for rec in records if int(rec.get("number", -1)) > latest_scan})
        if newer:
            freshness.append(f"Newer raw scans are available ({', '.join(map(str, newer))}); the reviewed component snapshot has not incorporated them.")
        for excluded in result.get("excluded_scans", []):
            record = raw.get(excluded.get("raw_file"))
            if record and excluded.get("raw_sha256") and record.get("path") and Path(record["path"]).is_file():
                if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != excluded["raw_sha256"]:
                    freshness.append(f"Excluded scan {excluded['scan']} has changed since the snapshot and may now be complete; no area has been inferred.")
            if record and excluded.get("npts") is not None and record.get("npts") != excluded["npts"]:
                freshness.append(f"Excluded scan {excluded['scan']} now contains {record.get('npts')} points; its snapshot exclusion needs review.")
        if any(rec.get("start") and rec["start"].astimezone(timezone.utc) > snapshot for rec in records):
            if not newer:
                freshness.append("Raw data acquired after the reviewed snapshot are present; the snapshot needs review.")
        warnings.extend(freshness)
        result["validation"] = {"valid": True, "preferred_points": len(preferred),
            "alternative_points": len(points) - len(preferred), "series_count": len(series),
            "verified_raw_files": len(checked), "unavailable_raw_files": unavailable,
            "freshness_warnings": freshness, "stale": bool(newer or any("changed since" in x or "needs review" in x for x in freshness))}
        return result
    except (KeyError, TypeError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        warnings.append(f"reviewed component data rejected: {exc}")
        return None


def component_series_figures(validated: dict[str, Any]) -> list[tuple[str, str]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    images = []
    for value, error, ylabel in (("ratio", "ratio_stderr", "Component area / its measured reference area"),
                                 ("area", "area_stderr", validated["area_unit"])):
        fig, ax = plt.subplots(figsize=(11, 6.4))
        has_alternatives = False
        for i, series in enumerate(validated["series"]):
            color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
            for choice in ("preferred", "alternative"):
                points = sorted([p for p in validated["points"] if p["series_id"] == series["id"] and p["fit_choice"] == choice],
                                key=lambda p: (p["temperature_k"], p["scan"]))
                if not points:
                    continue
                temperature = np.array([p["temperature_k"] for p in points])
                xerr = np.array([[p["temperature_k"] - p["temperature_min_k"] for p in points],
                                 [p["temperature_max_k"] - p["temperature_k"] for p in points]])
                ax.errorbar(temperature, [p[value] for p in points], yerr=[p[error] for p in points], xerr=xerr,
                    fmt=marker, linestyle="none", color=color, lw=.9, elinewidth=.8,
                    ms=5.5, capsize=2, markerfacecolor=color if choice == "preferred" else "none",
                    markeredgewidth=1.2, label=series["label"] if choice == "preferred" else "_nolegend_")
                has_alternatives |= choice == "alternative"
        if has_alternatives:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], marker="o", ls="", color="0.25", markerfacecolor="none"))
            labels.append("Alternative decomposition (same scan)")
            ax.legend(handles, labels, fontsize=9, ncol=2)
        else:
            ax.legend(fontsize=9, ncol=2)
        ax.axhline(1 if value == "ratio" else 0, color="0.65", lw=.7, ls="--")
        ax.set_xlabel("Measured sample temperature (K)")
        ax.set_ylabel(ylabel)
        ax.set_title(validated["title"] + (" — normalized areas" if value == "ratio" else " — absolute areas"), fontsize=12)
        ax.grid(alpha=.25)
        lower, upper = ax.get_ylim()
        minimum_interval = min(point[value] - point[error] for point in validated["points"])
        ax.set_ylim(0 if minimum_interval >= 0 else min(0, lower), max(0, upper))
        snapshot_label = _stamp(validated["snapshot_utc"]).strftime("%Y-%m-%d %H:%M UTC")
        fig.text(.5, .012, f"Snapshot {snapshot_label}  |  Formal 1σ; model uncertainty excluded; no Tc model", ha="center", fontsize=8, color="0.3")
        fig.tight_layout(rect=(0, .045, 1, 1))
        output = io.BytesIO(); fig.savefig(output, format="png", dpi=140); plt.close(fig)
        caption = ("All reviewed components share the same axes. Vertical bars: formal fit-only 1σ; horizontal bars: recorded temperature range. "
                   "Points are not joined or interpolated. Hollow markers are mutually exclusive same-scan decompositions, not additional observations. ")
        if value == "ratio":
            caption += "Each component uses its own measured reference; reference ratios are exactly 1, and shared-reference uncertainty is preserved in the supplied covariance. "
        caption += "No transition model is fitted or inferred by this page."
        images.append(("data:image/png;base64," + base64.b64encode(output.getvalue()).decode(), caption))
    return images


def component_series_section(payload: dict[str, Any], experiment_id: str, records: list[dict[str, Any]],
                             warnings: list[str], *, cycle: str | None = None) -> tuple[str, dict[str, Any]]:
    begin = len(warnings)
    data = validate_component_series(payload, experiment_id, records, warnings, cycle=cycle)
    escape = lambda value: html.escape(str(value), quote=True)
    if data is None:
        message = "; ".join(warnings[begin:])
        return ("<section><h2>Reviewed component intensities</h2><div class='note'>" + escape(message)
                + " Automatic single-peak fits are not substituted for this configured component view.</div></section>", {"valid": False, "reason": message})
    status = {**data["validation"], "snapshot_utc": data["snapshot_utc"], "coverage": data["coverage"]}
    parts = ["<section id='reviewed-component-series'><h2>" + escape(data["title"]) + "</h2>",
             "<p>Reviewed analysis snapshot: <b>" + escape(data["snapshot_utc"]) + "</b>. " + escape(data["normalization_description"]) + "</p>"]
    for observation in data.get("observations", []):
        parts.append("<div class='note'>" + escape(observation) + "</div>")
    for warning in data["validation"]["freshness_warnings"]:
        parts.append("<div class='note'><b>Snapshot coverage:</b> " + escape(warning) + "</div>")
    figures = component_series_figures(data)
    for i, (uri, caption) in enumerate(figures):
        if i == 1:
            parts.append("<details><summary>Absolute component areas on common axes</summary>")
        parts.append("<figure><img style='max-width:100%' src='" + uri + "' alt='" + escape(data["title"]) + "'><figcaption>" + escape(caption) + "</figcaption></figure>")
        if i == 1:
            parts.append("</details>")
    notes = list(data.get("caveats", []))
    notes.extend(f"Scan {entry['scan']}: {entry['reason']}" for entry in data.get("excluded_scans", []))
    if notes:
        parts.append("<ul>" + "".join("<li>" + escape(note) + "</li>" for note in notes) + "</ul>")
    parts.append("<details><summary>Component values, source files and model notes</summary><table><tr><th>Component</th><th>Scan</th><th>T (K)</th><th>Area ±1σ</th><th>Ratio ±1σ</th><th>Choice / note</th><th>Source</th></tr>")
    labels = {item["id"]: item["label"] for item in data["series"]}
    for point in data["points"]:
        parts.append("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in (
            labels[point["series_id"]], point["scan"], f"{point['temperature_k']:.3f}",
            f"{point['area']:.4g} ± {point['area_stderr']:.3g}", f"{point['ratio']:.4g} ± {point['ratio_stderr']:.3g}",
            point["fit_choice"] + (": " + point["note"] if point.get("note") else ""), point["raw_file"])) + "</tr>")
    parts.append("</table><p>Full preferred-point area/ratio covariance, source hashes and provenance are retained in the embedded reviewed-component-data JSON. Alternatives are excluded from that joint covariance.</p></details>")
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    parts.append("<script id='reviewed-component-data' type='application/json'>" + encoded + "</script></section>")
    return "\n".join(parts), status
