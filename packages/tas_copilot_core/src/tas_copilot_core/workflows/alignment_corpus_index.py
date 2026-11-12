from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tas_copilot_core.reporting import write_report_bundle


_RANGE_RE = re.compile(r"Range=([^:=\s]+)=([^:]+)")
_HEADER_KEYS = (
    "Filename",
    "Date",
    "Epoch",
    "ExptID",
    "ExptName",
    "ExptParticipants",
    "ScanType",
    "Scan",
    "ScanDescr",
    "Lattice",
    "Orient",
    "AnalyzerDetectorMode",
    "UBEnabled",
)
_ALIGNMENT_MOTORS = {
    "A3",
    "A4",
    "SmplLTilt",
    "SmplUTilt",
    "SmplLTrn",
    "SmplUTrn",
    "SmplHght",
    "SmplWdth",
    "SmplElev",
    "BkSltHght",
    "BkSltWdth",
    "ApertHori",
    "ApertVert",
    "FrontSlitWidth",
    "FrontSlitHeight",
    "BackSlitWidth",
    "BackSlitHeight",
}
_FINE_ALIGNMENT_MOTORS = {"SmplHght", "SmplWdth", "SmplElev", "SmplLTrn", "SmplUTrn", "BkSltHght", "BkSltWdth", "ApertHori", "ApertVert", "FrontSlitWidth", "FrontSlitHeight", "BackSlitWidth", "BackSlitHeight"}
_TILT_MOTORS = {"SmplLTilt", "SmplUTilt"}
# BT-7 data-column spellings for the harmonic (PG) filter translation and the
# velocity selector speed. Column-name knowledge stays in this reading layer;
# workflow scoring consumes only the typed event fields derived from them.
_FILTER_TRANSLATION_COLUMNS = ("Filtran", "FilTran", "filterTrans", "FilterTrans")
_VELOCITY_SELECTOR_COLUMNS = ("Velsel", "VelSel", "VelocitySelector")
_INCIDENT_ENERGY_COLUMNS = ("Ei", "EI")


def alignment_corpus_index(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_corpus_index")).expanduser()
    data_roots = _coerce_list(arguments.get("data_root") or arguments.get("data_roots"))
    if not data_roots:
        report = {
            "workflow_name": "alignment-corpus-index",
            "status": "missing_input",
            "missing_inputs": ["data_root"],
            "safe_next_actions": ["Provide one or more local data roots containing BT-7 .bt7 files and console logs."],
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_corpus_index.json", markdown_filename="alignment_corpus_index.md")
        return report

    max_files = int(arguments["max_files"]) if arguments.get("max_files") is not None else 5000
    include_logs = bool(arguments.get("include_logs", True))
    bt7_paths = _bounded_paths(data_roots, "*.bt7", max_files)
    log_paths = _bounded_paths(data_roots, "*.console.log", max_files) if include_logs else []

    scan_events = [_scan_event(path) for path in bt7_paths]
    scan_events = [event for event in scan_events if event is not None]
    log_events = []
    for path in log_paths:
        log_events.extend(_log_events(path, max_lines=int(arguments.get("max_log_lines") or 5000)))

    events = sorted(scan_events + log_events, key=lambda item: (str(item.get("source_path") or ""), int(item.get("line_number") or 0), str(item.get("event_id") or "")))
    summary = _summary(events, scan_events, log_events, bt7_paths, log_paths)
    report = {
        "workflow_name": "alignment-corpus-index",
        "status": "success",
        "data_roots": [str(Path(root).expanduser()) for root in data_roots],
        "summary": summary,
        "alignment_process_model": _alignment_process_model(),
        "heuristic_seed": _heuristic_seed(summary),
        "structure_factor_planning_link": _structure_factor_link(arguments),
        "events_sample": events[: int(arguments.get("sample_limit") or 50)],
        "warnings": [
            "Alignment logs are incomplete evidence: sample removal/re-mounting and rate-meter searches may be unlogged.",
            "A3 softposition resets and lattice updates can change the alignment frame; treat them as state-epoch boundaries until reviewed.",
            "Rapid/manual searches are coarse localization evidence and should be followed by recorded scans before accepting a reflection.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "alignment_events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_corpus_index.json", markdown_filename="alignment_corpus_index.md")
    report["artifacts"]["events_jsonl"] = str(events_path)
    return report


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _bounded_paths(roots: list[str], pattern: str, max_files: int) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root).expanduser()
        if root_path.is_file() and root_path.match(pattern):
            paths.append(root_path)
        elif root_path.exists():
            paths.extend(sorted(root_path.rglob(pattern)))
        if max_files and len(paths) >= max_files:
            return paths[:max_files]
    return paths


def _scan_event(path: Path) -> dict[str, Any] | None:
    header = _read_bt7_header(path)
    scan_descr = header.get("ScanDescr", "")
    scan = header.get("Scan", "")
    scan_type = header.get("ScanType", "")
    text = f"{scan_descr} {scan} {scan_type} {path.name}"
    if not _looks_alignment_related(text):
        return None
    ranges = _ranges(scan_descr)
    classification = _classify_scan(scan_descr, scan, scan_type, path.name)
    instrument_state = _bt7_instrument_state_columns(path)
    return {
        "event_id": f"scan:{path.name}",
        "event_type": "planned_scan",
        "source_type": "bt7_file",
        "source_path": str(path),
        "filename": path.name,
        "experiment_id": header.get("ExptID"),
        "date": header.get("Date"),
        "scan_type": scan_type,
        "scan": scan,
        "scan_descr": scan_descr,
        "scanned_devices": [item["device"] for item in ranges],
        "ranges": ranges,
        "alignment_stage": classification["alignment_stage"],
        "alignment_intent": classification["alignment_intent"],
        "automation_level": classification["automation_level"],
        "evidence_strength": classification["evidence_strength"],
        "state_epoch_effect": classification["state_epoch_effect"],
        "warnings": classification["warnings"],
        "harmonic_filter_translation": instrument_state["harmonic_filter_translation"],
        "velocity_selector_speed": instrument_state["velocity_selector_speed"],
        "incident_energy_mev": instrument_state["incident_energy_mev"],
        "a4_static_deg": instrument_state.get("a4_static_deg"),
        "a3_static_deg": instrument_state.get("a3_static_deg"),
        "metadata": {key: header.get(key) for key in sorted(_HEADER_KEYS) if header.get(key) is not None},
    }


def _read_bt7_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            stripped = line[1:].strip()
            for key in sorted(_HEADER_KEYS, key=len, reverse=True):
                if stripped.startswith(key):
                    header[key] = stripped[len(key) :].strip()
                    break
    return header


def _bt7_instrument_state_columns(path: Path) -> dict[str, Any]:
    """Extract per-scan Filtran/Velsel/Ei data-column values as typed fields.

    These are per-point data columns in BT-7 files; the unique value(s) across
    the scan are recorded (the state is expected constant within a scan).
    Returns harmonic_filter_translation IN|OUT|unknown and
    velocity_selector_speed / incident_energy_mev as float or None.
    """

    columns: list[str] = []
    observed: dict[str, list[str]] = {}
    watched: dict[str, tuple[str, ...]] = {
        "filter": _FILTER_TRANSLATION_COLUMNS,
        "selector": _VELOCITY_SELECTOR_COLUMNS,
        "ei": _INCIDENT_ENERGY_COLUMNS,
        "a4": ("A4",),
        "a3": ("A3",),
    }
    indices: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("#Columns"):
                    columns = stripped[len("#Columns") :].strip().split()
                    for role, names in watched.items():
                        for name in names:
                            if name in columns:
                                indices[role] = columns.index(name)
                                break
                    continue
                if not stripped or stripped.startswith("#") or not columns:
                    continue
                parts = stripped.split()
                for role, index in indices.items():
                    if index < len(parts):
                        value = parts[index]
                        bucket = observed.setdefault(role, [])
                        if value not in bucket:
                            bucket.append(value)
    except OSError:
        pass
    filter_values = {value.upper() for value in observed.get("filter", [])}
    harmonic_filter_translation = next(iter(filter_values)) if filter_values and filter_values <= {"IN", "OUT"} and len(filter_values) == 1 else "unknown"
    return {
        "harmonic_filter_translation": harmonic_filter_translation,
        "velocity_selector_speed": _first_float(observed.get("selector", [])),
        "incident_energy_mev": _first_float(observed.get("ei", [])),
        # Static motor positions: tilt/slit/rocking scans do not scan A4, but
        # the reflection they sit on is defined by the constant A4 (and A3)
        # in the data columns.
        "a4_static_deg": _static_float(observed.get("a4", [])),
        "a3_static_deg": _static_float(observed.get("a3", [])),
    }


def _static_float(values: list[str]) -> float | None:
    """The single value of a column that stays (nearly) constant, else None."""
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except ValueError:
            continue
    if not numeric:
        return None
    if max(numeric) - min(numeric) > 0.05:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _first_float(values: list[str]) -> float | None:
    for value in values:
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _ranges(scan_descr: str) -> list[dict[str, Any]]:
    ranges = []
    for match in _RANGE_RE.finditer(scan_descr):
        device = match.group(1).strip()
        values = match.group(2).strip()
        ranges.append({"device": device, "raw": values})
    return ranges


def _looks_alignment_related(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in ["findpeak", "find peak", "fpx", "range=a3", "range=a4", "smpllt", "smplut", "smplh", "smplw", "slt", "slit", "tilt"])


def _classify_scan(scan_descr: str, scan: str, scan_type: str, filename: str) -> dict[str, Any]:
    devices = {item["device"] for item in _ranges(scan_descr)}
    text = f"{scan_descr} {scan} {scan_type} {filename}".lower()
    warnings: list[str] = []
    if devices & _FINE_ALIGNMENT_MOTORS:
        return _classification("fine_alignment", "slit_translation_optimization", "supervised_autonomy_candidate", "strong_scan_data", "refines_sample_centering_or_beam_acceptance", warnings)
    if devices & _TILT_MOTORS:
        return _classification("second_reflection_search_or_plane_flattening", "tilt_axis_search_or_refinement", "suggest_or_supervised_only", "strong_scan_data", "updates_plane_confidence", warnings)
    if "range=a4" in text and "range=a3" in text:
        return _classification("first_or_second_reflection_localization", "theta_two_theta_or_bragg_coupled_scan", "supervised_autonomy_candidate", "strong_scan_data", "may_refine_peak_center_and_lattice", warnings)
    if "range=a3" in text:
        return _classification("first_or_second_reflection_localization", "rocking_scan", "supervised_autonomy_candidate", "strong_scan_data", "may_refine_peak_center", warnings)
    if "range=a4" in text:
        return _classification("gross_reflection_search_or_two_theta_localization", "sample_two_theta_scan", "suggest_or_supervised_only", "strong_scan_data", "may_refine_peak_center_and_lattice", warnings)
    return _classification("unknown_alignment_action", "alignment_related_scan", "suggest_only", "moderate_scan_metadata", "unknown", warnings)


def _classification(stage: str, intent: str, automation: str, evidence: str, epoch_effect: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "alignment_stage": stage,
        "alignment_intent": intent,
        "automation_level": automation,
        "evidence_strength": evidence,
        "state_epoch_effect": epoch_effect,
        "warnings": warnings,
    }


def _log_events(path: Path, *, max_lines: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, start=1):
            if max_lines and index > max_lines:
                break
            event = _log_event(path, index, line.rstrip())
            if event:
                events.append(event)
    return events


def _log_event(path: Path, line_number: int, line: str) -> dict[str, Any] | None:
    lower = line.lower()
    if "findpeak " in lower:
        return _base_log_event(path, line_number, line, "planned_scan_command", "recorded_peak_localization", "findpeak_command", "strong_log_command")
    if "scan descrtolist" in lower or "scan runscan" in lower:
        return _base_log_event(path, line_number, line, "planned_scan_command", "recorded_or_queued_scan", "scan_list_or_run_command", "moderate_log_command")
    if re.search(r">\s*move\s+", line, re.IGNORECASE):
        event = _base_log_event(path, line_number, line, "manual_search_move", "gross_reflection_search", "manual_or_rate_meter_search", "weak_inferred_search")
        event["warnings"].append("Move command may reflect rate-meter search or setup; intent is inferred unless followed by a nearby scan.")
        return event
    if "sample setlattice" in lower or "setorient" in lower:
        return _base_log_event(path, line_number, line, "software_state_update", "state_epoch_boundary", "lattice_or_orientation_update", "strong_log_command")
    if "drive to fit position" in lower or "lattice parameter" in lower:
        return _base_log_event(path, line_number, line, "fit_result_hint", "recorded_peak_localization", "fit_or_lattice_refinement_hint", "moderate_log_result")
    if "device [a3] will move with [a4]" in lower:
        return _base_log_event(path, line_number, line, "dependent_motor_hint", "theta_two_theta_or_bragg_coupled_scan", "a3_a4_coupled_motion", "strong_log_command")
    if "stopall" in lower or "abort" in lower:
        return _base_log_event(path, line_number, line, "possible_state_epoch_break", "unknown_alignment_action", "abort_or_stop", "moderate_log_command")
    if "creating file" in lower and ".bt7" in lower:
        return _base_log_event(path, line_number, line, "file_creation", "recorded_scan_output", "scan_file_written", "strong_log_record")
    return None


def _base_log_event(path: Path, line_number: int, line: str, event_type: str, stage: str, intent: str, evidence: str) -> dict[str, Any]:
    devices = sorted({device for device in _ALIGNMENT_MOTORS if re.search(rf"\b{re.escape(device)}\b", line, re.IGNORECASE)})
    return {
        "event_id": f"log:{path.name}:{line_number}",
        "event_type": event_type,
        "source_type": "console_log",
        "source_path": str(path),
        "line_number": line_number,
        "raw_line": line,
        "scanned_devices": devices,
        "alignment_stage": stage,
        "alignment_intent": intent,
        "automation_level": "suggest_only" if event_type == "manual_search_move" else "analysis_only",
        "evidence_strength": evidence,
        "state_epoch_effect": "starts_or_changes_epoch" if event_type in {"software_state_update", "possible_state_epoch_break"} else "observes_or_refines_current_epoch",
        "warnings": [],
    }


def _summary(events: list[dict[str, Any]], scan_events: list[dict[str, Any]], log_events: list[dict[str, Any]], bt7_paths: list[Path], log_paths: list[Path]) -> dict[str, Any]:
    stages = Counter(str(event.get("alignment_stage")) for event in events)
    intents = Counter(str(event.get("alignment_intent")) for event in events)
    event_types = Counter(str(event.get("event_type")) for event in events)
    devices = Counter(device for event in events for device in event.get("scanned_devices", []))
    by_experiment: dict[str, int] = defaultdict(int)
    for event in scan_events:
        by_experiment[str(event.get("experiment_id") or "unknown")] += 1
    return {
        "bt7_files_seen": len(bt7_paths),
        "console_logs_seen": len(log_paths),
        "alignment_scan_events": len(scan_events),
        "alignment_log_events": len(log_events),
        "alignment_event_count": len(events),
        "event_types": dict(event_types.most_common()),
        "alignment_stages": dict(stages.most_common()),
        "alignment_intents": dict(intents.most_common()),
        "devices": dict(devices.most_common()),
        "experiments_with_alignment_scans": dict(sorted(by_experiment.items())[:50]),
    }


def _alignment_process_model() -> dict[str, Any]:
    return {
        "representation": "temporal_evidence_graph_plus_alignment_state_machine",
        "not_a_pure_dag_reason": "sample orientation, software zeroes, and physical interventions can reset state; later evidence can revise earlier hypotheses.",
        "state_epochs": {
            "purpose": "Intervals where sample-instrument orientation is believed continuous.",
            "break_triggers": ["sample setlattice/setorient", "A3 softposition or hardware-zero reset", "StopAll/abort followed by new search", "long unobserved gap", "large unexplained motor jump", "manual sample manipulation note"],
        },
        "states": [
            "pre_alignment_planning",
            "gross_first_reflection_search",
            "first_reflection_localization",
            "first_reflection_accepted",
            "second_reflection_search",
            "second_reflection_localization",
            "gross_plane_established",
            "fine_alignment",
            "slit_translation_optimization",
            "alignment_validated",
            "alignment_invalid_or_reset_suspected",
        ],
        "event_types": ["planned_scan", "manual_search_move", "rapid_scan", "software_state_update", "fit_result_hint", "dependent_motor_hint", "possible_state_epoch_break", "file_creation"],
    }


def _heuristic_seed(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "heuristic_id": "scan_after_move_near_same_motor",
            "description": "A manual move followed by a scan on the same motor nearby is weak evidence of rate-meter search before recorded localization.",
            "needed_data": ["console_log move events", "subsequent scan file or findpeak event", "motor positions"],
            "status": "candidate_from_alignment_logs",
        },
        {
            "heuristic_id": "first_peak_to_second_tilt_axis",
            "description": "After an accepted first reflection, moving to calculated A3/A4 for a second reflection and scanning one tilt motor suggests rotation about the first scattering vector.",
            "needed_data": ["structure-factor/planned reflection table", "A3/A4 move", "SmplLTilt or SmplUTilt scan", "accepted first peak"],
            "status": "requires_state_machine_context",
        },
        {
            "heuristic_id": "fine_alignment_autonomy_candidate",
            "description": "Slit, sample height/width/elevation, and repeated rocking scans occur after gross alignment and are better candidates for supervised autonomy.",
            "observed_event_counts": {
                key: value for key, value in (summary.get("devices") or {}).items() if key in _FINE_ALIGNMENT_MOTORS
            },
            "status": "candidate_from_alignment_logs",
        },
    ]


def _structure_factor_link(arguments: dict[str, Any]) -> dict[str, Any]:
    report_path = arguments.get("alignment_peak_report")
    link = {
        "planning_workflow": "suggest-alignment-peaks",
        "role": "Use GSAS-II-backed structure factors and configured background materials to choose candidate first/second/third reflections before reading empirical alignment logs.",
        "status": "not_supplied",
        "safe_policy": "Peak suggestions are planning evidence; log-derived alignment events are observational evidence. Neither alone accepts an alignment.",
    }
    if report_path:
        path = Path(str(report_path)).expanduser()
        link["status"] = "loaded" if path.exists() else "missing_report"
        link["report_path"] = str(path)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                link["top_candidate_count"] = len(payload.get("alignment_peaks") or [])
                link["formula"] = payload.get("formula")
                link["scattering_plane"] = payload.get("scattering_plane")
            except Exception as exc:
                link["status"] = "unreadable_report"
                link["warning"] = str(exc)
    return link


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Alignment Corpus Index",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- BT-7 files seen: `{summary.get('bt7_files_seen', 0)}`",
        f"- Console logs seen: `{summary.get('console_logs_seen', 0)}`",
        f"- Alignment scan events: `{summary.get('alignment_scan_events', 0)}`",
        f"- Alignment log events: `{summary.get('alignment_log_events', 0)}`",
        "",
        "## Alignment Stages",
        "",
    ]
    for key, value in (summary.get("alignment_stages") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Devices", ""])
    for key, value in list((summary.get("devices") or {}).items())[:20]:
        lines.append(f"- `{key}`: {value}")
    model = report.get("alignment_process_model") or {}
    lines.extend(
        [
            "",
            "## Process Model",
            "",
            f"- Representation: `{model.get('representation')}`",
            f"- Why not a pure DAG: {model.get('not_a_pure_dag_reason')}",
            "",
            "## Structure-Factor Link",
            "",
            f"- Status: `{(report.get('structure_factor_planning_link') or {}).get('status')}`",
            f"- Role: {(report.get('structure_factor_planning_link') or {}).get('role')}",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in report.get("warnings") or []:
        lines.append(f"- {warning}")
    lines.extend(["", "## Next Actions", "", "- Review the JSONL event index, then build state-epoch heuristics against a few known alignment sessions before enabling any autonomous fine-alignment loop."])
    return "\n".join(lines)
