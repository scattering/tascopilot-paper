from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.corpus.pilot import corpus_search
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.structure_factors import suggest_alignment_peaks
from tas_copilot_core.workflows.catalog_tas_directory import _parse_scan_description
from tas_copilot_core.workflows.experiment_dashboard import build_experiment_dashboard


DEFAULT_REPLAY_REPORT = Path("local_runs/nice-bt7-replay-smoke/replay_report.json")
DEFAULT_PRECEDENT_INDEXES = (
    Path("local_runs/bt7_hydrated_index/bt7_hydrated_index_full_source_units.jsonl"),
    Path("local_runs/bt7_hydrated_index/bt7_hydrated_index_full.jsonl"),
)


def experiment_planning_brief(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compose read-only planning evidence into one experiment-planning brief.

    The workflow intentionally does not control an instrument. Replay reports are
    treated as simulation-only evidence that the proposed scan shape can be
    exercised through the NICE API pattern.
    """

    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    material = str(arguments.get("material") or arguments.get("formula") or "").strip()
    goal = str(arguments.get("goal") or arguments.get("question") or "").strip()
    if not material:
        report = {
            "workflow_name": "experiment-planning-brief",
            "status": "missing_input",
            "missing_fields": ["material"],
            "warnings": ["A material or formula is required."],
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_planning_brief_markdown, json_filename="experiment_planning_brief.json", markdown_filename="experiment_planning_brief.md")
        return report

    question = goal or f"Plan a TAS experiment for {material}."
    precedent_index = _resolve_path(arguments.get("precedent_index")) or _first_existing(DEFAULT_PRECEDENT_INDEXES)
    corpus_catalog = _resolve_path(arguments.get("corpus_catalog"))
    data_root = _resolve_path(arguments.get("data_root"))
    replay_report = _resolve_path(arguments.get("replay_report")) or (DEFAULT_REPLAY_REPORT if DEFAULT_REPLAY_REPORT.exists() else None)

    alignment = _alignment_channel(arguments, output_dir / "alignment_peaks", material)
    precedents = _precedent_channel(precedent_index, material, question, int(arguments.get("top_k") or 10))
    corpus = _corpus_channel(corpus_catalog, material, question, int(arguments.get("top_k") or 10))
    replay = _replay_channel(replay_report)
    dashboard = _dashboard_channel(data_root, output_dir / "dashboard", arguments)
    recommendations = _recommendations(material, question, alignment, precedents, replay)
    confidence = _confidence_summary(alignment, precedents, corpus, replay, dashboard)

    report = {
        "workflow_name": "experiment-planning-brief",
        "status": "ok",
        "material": material,
        "goal": question,
        "evidence_channels": {
            "alignment_peaks": alignment,
            "scan_precedents": precedents,
            "corpus": corpus,
            "replay": replay,
            "dashboard": dashboard,
        },
        "recommendations": recommendations,
        "confidence": confidence,
        "safe_conclusion": _safe_conclusion(material, confidence),
        "not_approved_operations": [
            "No live instrument control is authorized.",
            "Replay evidence is simulation-only and does not approve measurement execution.",
            "Planning recommendations require human instrument-scientist review.",
        ],
        "warnings": _warnings(alignment, precedents, corpus, replay),
        "provenance": [
            ProvenanceRecord(
                source_type="workflow",
                source_id="experiment-planning-brief",
                method="composed_read_only_planning_evidence",
                confidence=confidence["overall_probability_style_confidence"],
            ).model_dump(mode="json")
        ],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_planning_brief_markdown, json_filename="experiment_planning_brief.json", markdown_filename="experiment_planning_brief.md")
    return report


def _alignment_channel(arguments: dict[str, Any], output_dir: Path, material: str) -> dict[str, Any]:
    if not arguments.get("icsd_db"):
        return {
            "status": "not_supplied",
            "method": "suggest-alignment-peaks",
            "note": "No ICSD database was supplied; alignment peaks were not calculated.",
        }
    payload = {
        "formula": arguments.get("formula") or material,
        "icsd_db": arguments.get("icsd_db"),
        "scattering_plane": arguments.get("scattering_plane"),
        "space_group": arguments.get("space_group"),
        "space_group_number": arguments.get("space_group_number"),
        "space_group_setting": arguments.get("space_group_setting"),
        "energy_mev": float(arguments.get("energy_mev") or 14.7),
        "a4_max_deg": float(arguments.get("a4_max_deg") or 100.0),
        "top_k": int(arguments.get("alignment_top_k") or min(int(arguments.get("top_k") or 10), 20)),
        "candidate_limit": int(arguments.get("candidate_limit") or 25),
        "max_reflections": int(arguments.get("max_reflections") or 600),
        "max_background_reflections": int(arguments.get("max_background_reflections") or 300),
        "out_of_plane_max_deg": float(arguments.get("out_of_plane_max_deg") or 5.0),
        "background_overlap_tolerance_deg": float(arguments.get("background_overlap_tolerance_deg") or 0.5),
        "background_material": arguments.get("background_material"),
        "background_material_config": arguments.get("background_material_config"),
        "output_dir": str(output_dir),
    }
    try:
        result = suggest_alignment_peaks(payload)
    except Exception as exc:  # pragma: no cover - fail closed around optional local runtimes.
        return {
            "status": "error",
            "method": "suggest-alignment-peaks",
            "note": f"{type(exc).__name__}: {exc}",
            "artifacts": {},
        }
    peaks = result.get("alignment_peaks") or []
    return {
        "status": result.get("status"),
        "method": "suggest-alignment-peaks",
        "selected_structure": result.get("selected_structure"),
        "peak_count": len(peaks),
        "top_peaks": peaks[:8],
        "missing_inputs": result.get("missing_inputs") or [],
        "warnings": result.get("warnings") or [],
        "artifacts": result.get("artifacts") or {},
    }


def _precedent_channel(index_path: Path | None, material: str, question: str, top_k: int) -> dict[str, Any]:
    if not index_path or not index_path.exists():
        return {
            "status": "missing_input",
            "method": "hydrated_index_material_search",
            "missing_fields": ["precedent_index"],
            "note": "No hydrated BT-7 precedent index was available.",
        }
    aliases = _material_aliases(material)
    candidates = []
    for raw in index_path.open(encoding="utf-8"):
        line_lower = raw.lower()
        if not any(alias in line_lower for alias in aliases):
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates.append(_score_precedent_record(record, line_lower, aliases, question))
    candidates = [item for item in candidates if item["score"] > 0.05]
    candidates.sort(key=lambda item: (item["score"], item.get("scan_count") or 0, item.get("datafile_count") or 0), reverse=True)
    candidates = candidates[:top_k]
    return {
        "status": "used" if candidates else "searched_no_match",
        "method": "hydrated_index_material_search",
        "source": str(index_path),
        "aliases": aliases,
        "candidate_count": len(candidates),
        "top_candidates": candidates,
        "user_facing_top_patterns": [_sanitize_precedent_pattern(candidate, index) for index, candidate in enumerate(candidates[:top_k], start=1)],
        "user_facing_source": "internal precedent-pattern index",
        "privacy_boundary": "Do not expose prior user names, raw paths, or experiment identities in ordinary user-facing answers.",
    }


def _sanitize_precedent_pattern(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    recipes = []
    for recipe in candidate.get("scan_recipe_examples") or []:
        recipes.append(
            {
                "jtype": recipe.get("jtype"),
                "title_pattern": _generic_recipe_title(recipe.get("title")),
                "npts": recipe.get("npts"),
                "count_type": recipe.get("count_type"),
                "fixed_e": recipe.get("fixed_e"),
                "range_axes": recipe.get("range_axes") or [],
                "ranges": recipe.get("ranges") or [],
            }
        )
    return {
        "pattern_id": f"precedent_pattern_{index:02d}",
        "score": candidate.get("score"),
        "confidence_label": candidate.get("confidence_label"),
        "scan_count": candidate.get("scan_count"),
        "datafile_count": candidate.get("datafile_count"),
        "scan_family_priors": candidate.get("scan_family_priors") or [],
        "fixed_energy_values": candidate.get("fixed_energy_values") or [],
        "temperature_values": candidate.get("temperature_values") or [],
        "scan_recipe_patterns": recipes[:5],
        "privacy_note": "Summarized from private/local precedent metadata; raw experiment identity, paths, and prior user details are withheld from user-facing explanations.",
    }


def _generic_recipe_title(title: Any) -> str | None:
    if not title:
        return None
    text = str(title)
    text = re.sub(r"\d{5,}", "[scan-id]", text)
    text = re.sub(r"([A-Za-z]+)_?\d+p?\d*", r"\1_[offset]", text)
    return text[:80]


def _corpus_channel(corpus_catalog: Path | None, material: str, question: str, top_k: int) -> dict[str, Any]:
    if not corpus_catalog:
        return {"status": "not_supplied", "method": "corpus_search", "result_count": 0}
    if not corpus_catalog.exists():
        return {"status": "missing_input", "method": "corpus_search", "result_count": 0, "note": f"Missing corpus catalog: {corpus_catalog}"}
    query = f"{material} {question} TAS triple-axis alignment spin wave order parameter background spurion"
    try:
        result = corpus_search({"catalog": str(corpus_catalog), "query": query, "top_k": min(top_k, 10), "exclude_references": True})
    except Exception as exc:  # pragma: no cover - optional local corpus state.
        return {"status": "error", "method": "corpus_search", "result_count": 0, "note": f"{type(exc).__name__}: {exc}"}
    results = list(result.get("results") or [])
    filtered = []
    for item in results:
        blob = " ".join([str(item.get("title") or ""), str(item.get("snippet") or ""), " ".join(map(str, item.get("matched_terms") or []))]).lower()
        if any(term in blob for term in _material_aliases(material)) or any(term in blob for term in ("spin", "magnon", "dispersion", "triple-axis", "bt-7", "constant-q", "alignment")):
            filtered.append(item)
    return {
        "status": "used" if filtered else "searched_no_specific_support",
        "method": "corpus_search",
        "query": query,
        "result_count": len(filtered),
        "results": filtered[:5],
        "warnings": result.get("warnings") or [],
    }


def _replay_channel(report_path: Path | None) -> dict[str, Any]:
    if not report_path:
        return {
            "status": "not_supplied",
            "method": "nice_replay_report",
            "note": "No NICE replay report was supplied or found.",
        }
    if not report_path.exists():
        return {
            "status": "missing_input",
            "method": "nice_replay_report",
            "source": str(report_path),
            "missing_fields": ["replay_report"],
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "method": "nice_replay_report", "source": str(report_path), "note": f"{type(exc).__name__}: {exc}"}
    measurement = payload.get("measurement") or {}
    points = measurement.get("points") or []
    dag = payload.get("instrument_dag") or {}
    file_write = payload.get("file_write") or {}
    successes = sum(1 for point in points if point.get("move_status") == "success" and point.get("count_status") == "success")
    return {
        "status": "used" if payload.get("status") == "success" else str(payload.get("status") or "unknown"),
        "method": "nice_replay_report",
        "source": str(report_path),
        "simulation_only": True,
        "scan_path": payload.get("scan_path"),
        "scan_metadata": payload.get("scan_metadata") or {},
        "point_count": len(points),
        "move_and_count_success_count": successes,
        "created_file_count": len(file_write.get("created_files") or []),
        "created_file_previews_available": bool(file_write.get("created_previews")),
        "instrument_graph_summary": {
            "node_count": dag.get("node_count"),
            "control_input_count": dag.get("control_input_count"),
            "readback_only_count": dag.get("readback_only_count"),
            "detector_or_counter_node_count": dag.get("detector_or_counter_node_count"),
            "important_node_ids": sorted((dag.get("important_nodes") or {}).keys()),
        },
        "limit_overrides": payload.get("simulation_limit_overrides") or [],
        "warnings": payload.get("warnings") or [],
    }


def _dashboard_channel(data_root: Path | None, output_dir: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    if not data_root:
        return {"status": "not_supplied", "method": "experiment-dashboard"}
    if not data_root.exists():
        return {"status": "missing_input", "method": "experiment-dashboard", "source": str(data_root)}
    if not bool(arguments.get("run_dashboard")):
        return {
            "status": "available_not_run",
            "method": "experiment-dashboard",
            "source": str(data_root),
            "note": "Pass run_dashboard=true or CLI --run-dashboard to generate a dashboard bundle.",
        }
    try:
        result = build_experiment_dashboard(
            {
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "max_files": int(arguments.get("max_files") or 200),
                "experiment_config": arguments.get("experiment_config"),
            }
        )
    except Exception as exc:  # pragma: no cover - reader/local filesystem boundary.
        return {"status": "error", "method": "experiment-dashboard", "source": str(data_root), "note": f"{type(exc).__name__}: {exc}"}
    return {
        "status": result.get("status"),
        "method": "experiment-dashboard",
        "source": str(data_root),
        "scan_count": result.get("scan_count"),
        "artifacts": result.get("generated_artifacts") or result.get("artifacts") or {},
        "warnings": result.get("warnings") or [],
    }


def _recommendations(material: str, question: str, alignment: dict[str, Any], precedents: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    precedent_candidates = precedents.get("top_candidates") or []
    temperature_values = [item for candidate in precedent_candidates for item in candidate.get("temperature_values", [])]
    energy_values = [item for candidate in precedent_candidates for item in candidate.get("fixed_energy_values", [])]
    scan_patterns = Counter(pattern for candidate in precedent_candidates for pattern in candidate.get("scan_family_priors", []))
    peaks = alignment.get("top_peaks") or []
    alignment_recs = []
    for peak in peaks[:5]:
        alignment_recs.append(
            {
                "hkl": [peak.get("h"), peak.get("k"), peak.get("l")],
                "two_theta_deg": peak.get("two_theta_deg"),
                "q_inv_angstrom": peak.get("q_inv_angstrom"),
                "relative_structure_factor": peak.get("relative_structure_factor"),
                "cautions": (peak.get("background_warnings") or []) + ([peak["multiple_scattering_warning"]] if peak.get("multiple_scattering_warning") else []),
            }
        )
    scan_recs = _scan_recommendations(question, scan_patterns)
    return {
        "alignment_candidates": alignment_recs,
        "temperature_plan": _temperature_plan(material, temperature_values),
        "configuration_plan": _configuration_plan(energy_values),
        "scan_plan": scan_recs,
        "replay_plan": _replay_plan(replay, scan_recs),
        "precedent_explanation_policy": {
            "user_facing_source": precedents.get("user_facing_source"),
            "privacy_boundary": precedents.get("privacy_boundary"),
            "internal_candidate_count": precedents.get("candidate_count", 0),
            "user_facing_patterns": precedents.get("user_facing_top_patterns") or [],
        },
        "missing_inputs_to_request": [
            "sample orientation and scattering plane",
            "lattice/UB or chosen indexing convention",
            "sample environment and background materials",
            "monochromator/analyzer/filter/collimation state",
            "target energy and Q range",
        ],
    }


def _temperature_plan(material: str, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(alias in material.lower() for alias in ("lsmo", "la0.7sr0.3", "la.7sr.3")):
        return [
            {
                "recommendation": "Use a compact first pass near 100 K, 300 K, 340-350 K, and 360-380 K.",
                "probability_style_rationale": "The LSMO precedent index contains repeated measurements around 200 K, 250 K, 300 K, 345 K, and 365 K; these bracket ordered, near-transition, and above-transition regimes.",
                "evidence": values[:12],
            }
        ]
    return [
        {
            "recommendation": "Choose temperatures from the phase/transition question, then add repeated points near suspected transitions and background/control temperatures.",
            "probability_style_rationale": "No material-specific temperature rule is encoded for this material; use precedent temperatures only as weak priors.",
            "evidence": values[:12],
        }
    ]


def _configuration_plan(energy_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "recommendation": "Start with intensity-favoring/open or loose collimation for scouting; tighten after locating signal.",
            "probability_style_rationale": "Historic metadata rarely preserves a complete four-collimation state. Loose scouting has stronger precedent support than any exact collimation recipe.",
            "evidence": energy_values[:8],
        },
        {
            "recommendation": "Treat Ef=14.7 meV and 40/40-style collimation as labeled priors when relevant, not as automatic requirements.",
            "probability_style_rationale": "Ef=14.7 meV appears often in BT-7 precedents, but filters, analyzer state, and collimations still need explicit user confirmation.",
            "evidence": energy_values[:8],
        },
    ]


def _scan_recommendations(question: str, scan_patterns: Counter[str]) -> list[dict[str, Any]]:
    lower = question.lower()
    if any(term in lower for term in ("spin", "magnon", "dispersion", "stiffness", "inelastic")):
        return [
            {
                "recommendation": "Start with elastic alignment/Bragg scans at the selected zone center.",
                "probability": 0.82,
                "evidence": dict(scan_patterns),
            },
            {
                "recommendation": "Measure small-q constant-Q energy scans at several offsets from the zone center.",
                "probability": 0.78,
                "suggested_offsets_rlu": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                "evidence": dict(scan_patterns),
            },
            {
                "recommendation": "Add fixed-energy transverse and longitudinal Q scans after locating the modes.",
                "probability": 0.7,
                "suggested_energies_mev": [3, 5, 7, 10, 15],
                "evidence": dict(scan_patterns),
            },
        ]
    if any(term in lower for term in ("order", "temperature", "transition", "elastic", "diffraction", "psd")):
        return [
            {
                "recommendation": "Start with elastic scans through strong structural or magnetic peaks, then repeat a short temperature series with background side points.",
                "probability": 0.72,
                "evidence": dict(scan_patterns),
            },
            {
                "recommendation": "If using PSD data, replay and review detector-channel maps before converting to Q; keep channel/A4 provenance explicit.",
                "probability": 0.68,
                "evidence": dict(scan_patterns),
            },
        ]
    return [
        {
            "recommendation": "Start with alignment/Bragg checks, a short scout scan in the intended scan direction, and one background/control scan.",
            "probability": 0.55,
            "evidence": dict(scan_patterns),
        }
    ]


def _replay_plan(replay: dict[str, Any], scan_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if replay.get("status") != "used":
        return [
            {
                "recommendation": "Run a NICE simulation replay before turning this plan into an executable scan recipe.",
                "probability": 0.8,
                "reason": "No usable replay report was supplied.",
            }
        ]
    return [
        {
            "recommendation": "Use point-by-point replay for adaptive planning tests and trajectory replay for fixed scan recipes.",
            "probability": 0.86,
            "reason": f"Replay report shows {replay.get('move_and_count_success_count')} successful move/count points and {replay.get('created_file_count')} generated files.",
        },
        {
            "recommendation": "Before proposing live operations, replay the first alignment scan, one scout scan, and one background scan through the NICE API shape.",
            "probability": 0.82,
            "reason": "Replay validates command translation, readbacks, and file-writing without touching hardware.",
        },
    ]


def _score_precedent_record(record: dict[str, Any], line_lower: str, aliases: list[str], question: str) -> dict[str, Any]:
    title = str(record.get("title") or "")
    hydrated = record.get("hydrated_clue_summary") or {}
    top_values = hydrated.get("top_values") or {}
    scan_descr_values = hydrated.get("scan_descr_values") or {}
    path_clues = record.get("path_clues") or {}
    path_prefixes = list(path_clues.get("top_filename_prefixes") or [])[:6]
    scan_examples = _scan_description_examples(top_values, scan_descr_values)
    scan_recipe_examples = [_parse_scan_recipe(str(item.get("value") or "")) for item in scan_examples if item.get("value")]
    scan_variables = list(top_values.get("scan_variable") or [])[:6]
    detector_modes = []
    for subgroup in hydrated.get("reader_subgroups") or []:
        detector_modes.extend(subgroup.get("detector_modes") or [])
    family_text = " ".join(
        [
            title,
            " ".join(str(item.get("value", "")) for item in path_prefixes),
            " ".join(str(item.get("value", "")) for item in scan_examples),
            " ".join(str(item.get("value", "")) for item in scan_variables),
            " ".join(str(item.get("value", "")) for item in detector_modes),
        ]
    ).lower()
    question_lower = question.lower()
    score = 0.0
    evidence: list[dict[str, Any]] = []
    if any(alias in line_lower for alias in aliases):
        score += 0.35
        evidence.append({"component": "material_alias", "score": 0.35, "detail": "Material alias appears in hydrated metadata."})
    for term, amount in (("inelastic", 0.12), ("spin", 0.08), ("dispersion", 0.08), ("phonon", 0.05), ("elastic", 0.05), ("psd", 0.05), ("correlation", 0.05), ("temperature", 0.05)):
        if term in family_text or term in question_lower:
            score += amount
    if top_values.get("fixed_e"):
        score += 0.08
        evidence.append({"component": "fixed_energy", "score": 0.08, "detail": "Fixed-energy values are exposed."})
    if top_values.get("temp_mean") or top_values.get("temp_start"):
        score += 0.08
        evidence.append({"component": "temperature", "score": 0.08, "detail": "Temperature values are exposed."})
    if top_values.get("scan_variable") or scan_descr_values.get("scan_description"):
        score += 0.08
        evidence.append({"component": "scan_pattern", "score": 0.08, "detail": "Scan variable or scan description values are exposed."})
    asks_elastic_psd = "elastic" in question_lower and "psd" in question_lower
    recipe_nonzero_energy = any(_recipe_has_nonzero_energy(recipe) for recipe in scan_recipe_examples)
    recipe_psd_elastic = any(_recipe_has_axis(recipe, {"a4", "temp", "temperature"}) for recipe in scan_recipe_examples)
    looks_nonzero_energy = (
        recipe_nonzero_energy
        or bool(re.search(r"(^|[_\\W])(?:3|4|6|8|9|10|12|14)mev", family_text))
        or "inelastic" in family_text
        or any(str(item.get("value", "")).lower() == "e" for item in scan_variables)
    )
    looks_psd_elastic = any(term in family_text for term in ("psd", "psddiff", "psdflat", "correlations", "tdepwarm", "tdepcool", "29pt5")) and (
        recipe_psd_elastic or "psd" in family_text
    )
    if asks_elastic_psd and looks_nonzero_energy and not looks_psd_elastic:
        score -= 0.28
        evidence.append({"component": "elastic_psd_mismatch_penalty", "score": -0.28, "detail": "Elastic PSD planning question should not be driven by nonzero-energy line scans."})
    if asks_elastic_psd and looks_psd_elastic:
        score += 0.16
        evidence.append({"component": "elastic_psd_source_prior", "score": 0.16, "detail": "Candidate has PSD/correlation/temperature-map clues relevant to elastic PSD triage."})
    false_family = any(term in family_text for term in ("lasrmn2o7", "lasrmno4", "bilayer", "unknown experiment"))
    if false_family:
        score -= 0.22
        evidence.append({"component": "false_family_or_generic_penalty", "score": -0.22, "detail": "Possible false-family or generic match."})
    score = max(0.0, min(1.0, score))
    return {
        "candidate_id": record.get("experiment_id"),
        "title": title,
        "score": round(score, 3),
        "confidence_label": "high" if score >= 0.7 else "medium" if score >= 0.45 else "low",
        "start_date": record.get("start_date"),
        "datafile_count": record.get("datafile_count"),
        "scan_count": record.get("scan_count"),
        "path_prefixes": path_prefixes,
        "fixed_energy_values": list(top_values.get("fixed_e") or [])[:6],
        "temperature_values": (list(top_values.get("temp_mean") or []) + list(top_values.get("temp_start") or []))[:12],
        "scan_variables": scan_variables,
        "scan_examples": scan_examples,
        "scan_recipe_examples": scan_recipe_examples,
        "scan_family_priors": _scan_family_priors(family_text, scan_examples, scan_recipe_examples),
        "evidence": evidence,
        "privacy_label": "internal precedent pattern",
        "provenance": [{"source_type": "hydrated_bt7_index_record", "source_id": record.get("experiment_id"), "method": "experiment_planning_brief_score"}],
    }


def _scan_description_examples(top_values: dict[str, Any], scan_descr_values: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for source in (scan_descr_values.get("scan_description"), top_values.get("scan_description")):
        for item in list(source or []):
            if isinstance(item, dict) and item.get("value") and item not in examples:
                examples.append(item)
    return examples[:5]


def _scan_family_priors(line_lower: str, examples: list[dict[str, Any]], recipes: list[dict[str, Any]] | None = None) -> list[str]:
    text = line_lower + " " + " ".join(str(item.get("value", "")) for item in examples).lower()
    priors = []
    recipes = recipes or []
    recipe_axes = {str(axis).lower() for recipe in recipes for axis in recipe.get("range_axes", [])}
    recipe_jtypes = {str(recipe.get("jtype") or "").lower() for recipe in recipes if recipe.get("jtype")}
    if any(term in text for term in ("range=e=", "scan_variable\": \"e", "constantq", "constant q")):
        priors.append("constant_q_energy_scan")
    if any(term in text for term in ("range=q=", "trans", "long", "q0p", "qx", "qy")):
        priors.append("q_scan")
    if any(term in text for term in ("temp", "temperature", "tdep")):
        priors.append("temperature_series")
    if any(term in text for term in ("psd", "psddiff", "psdflat")):
        priors.append("psd_scan")
    if any(term in text for term in ("bragg", "elastic", "range=a3", "rock")):
        priors.append("elastic_alignment")
    if "e" in recipe_axes:
        priors.append("constant_q_energy_scan")
        if any(_recipe_has_nonzero_energy(recipe) for recipe in recipes):
            priors.append("nonzero_energy_scan")
    if recipe_axes.intersection({"qx", "qy", "qz", "q"}):
        priors.append("q_scan")
    if recipe_axes.intersection({"temp", "temperature"}) or "env" in recipe_jtypes:
        priors.append("temperature_series")
    if "a4" in recipe_axes or "angle" in recipe_jtypes:
        priors.append("angle_scan")
    if "vector" in recipe_jtypes:
        priors.append("vector_scan")
    return sorted(set(priors))


def _parse_scan_recipe(scan_description: str) -> dict[str, Any]:
    """Adapt the existing BT-7 ICE/NICE `#ScanDescr` parser for reports.

    The parser itself lives in the catalog workflow. This adapter preserves the
    raw recipe and converts parsed `Range.<axis>` fields into a structured list
    without trying to infer missing scientific configuration.
    """

    raw = str(scan_description or "").strip()
    recipe: dict[str, Any] = {
        "raw": raw,
        "fields": {},
        "ranges": [],
        "range_axes": [],
    }
    if not raw:
        return recipe

    parsed = _parse_scan_description(raw)
    for key, value in parsed.items():
        if key.startswith("Range."):
            axis = key.split(".", 1)[1].strip()
            values = str(value).strip()
            numbers = _numeric_values(values)
            range_record = {
                "axis": axis,
                "raw": values,
                "numeric_values": numbers,
            }
            if len(numbers) >= 1:
                range_record["start"] = numbers[0]
            if len(numbers) >= 2:
                range_record["stop"] = numbers[1]
            if len(numbers) >= 3 and "~" in values:
                range_record["first_point"] = numbers[:3]
            if len(numbers) >= 6 and "~" in values:
                range_record["last_point"] = numbers[3:6]
            recipe["ranges"].append(range_record)
            if axis:
                recipe["range_axes"].append(axis)
            continue
        normalized = _normalize_recipe_key(key)
        recipe["fields"][normalized] = str(value).strip()
        recipe[normalized] = _coerce_recipe_value(normalized, str(value).strip())
    recipe["range_axes"] = sorted(set(recipe["range_axes"]))
    return recipe


def _normalize_recipe_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    aliases = {
        "jtype": "jtype",
        "npts": "npts",
        "fixede": "fixed_e",
        "counttype": "count_type",
        "detectortype": "detector_type",
        "holdscan": "hold_scan",
        "holdpoint": "hold_point",
        "subid": "sub_id",
    }
    return aliases.get(normalized, normalized)


def _coerce_recipe_value(key: str, value: str) -> Any:
    if key == "npts":
        try:
            return int(float(value))
        except ValueError:
            return value
    if key in {"counts", "prefac", "fixed", "fixed_e", "hold_scan", "hold_point"}:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _numeric_values(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]


def _recipe_has_axis(recipe: dict[str, Any], axes: set[str]) -> bool:
    return any(str(axis).lower() in axes for axis in recipe.get("range_axes", []))


def _recipe_has_nonzero_energy(recipe: dict[str, Any]) -> bool:
    for range_record in recipe.get("ranges") or []:
        if str(range_record.get("axis") or "").lower() != "e":
            continue
        values = [float(value) for value in range_record.get("numeric_values") or []]
        if values and any(abs(value) > 0.05 for value in values[:2]):
            return True
    return False


def _confidence_summary(*channels: dict[str, Any]) -> dict[str, Any]:
    weights = {
        "alignment_peaks": 0.2,
        "scan_precedents": 0.32,
        "corpus": 0.12,
        "replay": 0.24,
        "dashboard": 0.12,
    }
    statuses = [channel.get("status") for channel in channels]
    score = 0.0
    for name, channel in zip(weights, channels):
        status = channel.get("status")
        if status in {"success", "used", "ok"}:
            score += weights[name]
        elif status in {"available_not_run", "searched_no_specific_support"}:
            score += weights[name] * 0.35
    return {
        "overall_probability_style_confidence": round(min(score, 0.95), 3),
        "channel_statuses": statuses,
        "interpretation": "decision-support probability, not certainty or approval",
    }


def _warnings(alignment: dict[str, Any], precedents: dict[str, Any], corpus: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    warnings = [
        "Do not treat this planning brief as an approved instrument operation.",
        "Indexing conventions are not converted automatically; preserve native/selected setting until reviewed.",
    ]
    for channel in (alignment, precedents, corpus, replay):
        warnings.extend(_format_warning(item) for item in channel.get("warnings") or [])
    if precedents.get("status") != "used":
        warnings.append("No strong material precedent channel was available.")
    if replay.get("status") != "used":
        warnings.append("Replay evidence was not available; run a NICE simulation replay before live planning review.")
    return list(dict.fromkeys(warnings))


def _format_warning(item: Any) -> str:
    if isinstance(item, dict):
        code = item.get("code")
        term = item.get("term")
        message = item.get("message") or item.get("note")
        parts = [str(value) for value in (code, term, message) if value]
        return ": ".join(parts) if parts else json.dumps(item, sort_keys=True)
    return str(item)


def _safe_conclusion(material: str, confidence: dict[str, Any]) -> str:
    return (
        f"For {material}, the co-pilot assembled a probability-style planning brief with "
        f"overall confidence {confidence['overall_probability_style_confidence']}. "
        "It is suitable for instrument-scientist review, not automatic execution."
    )


def _material_aliases(material: str) -> list[str]:
    compact = re.sub(r"[^a-z0-9]", "", material.lower())
    aliases = {compact}
    if "la07sr03mno3" in compact or "la7sr3mn10o30" in compact or "lsmo" in compact:
        aliases.update({"lsmo", "la07sr03mno3", "la7sr3mn10o30", "lasrmno3", "lasrmno"})
    if "lufe" in compact and "mn" in compact and "o3" in compact:
        aliases.update({"lufemno3", "lumnfeo3", "lufe75mn25o3", "lufemn"})
    if "lumnfeo3" in compact or "lufemno3" in compact:
        aliases.update({"lufemno3", "lumnfeo3", "lufemn"})
    return sorted(alias for alias in aliases if alias)


def _resolve_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _planning_brief_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Planning Brief",
        "",
        f"Status: {report.get('status')}",
        f"Material: `{report.get('material')}`",
        f"Goal: {report.get('goal')}",
        "",
        "## Safe Conclusion",
        report.get("safe_conclusion") or "",
        "",
        "## Evidence Channels",
    ]
    channels = report.get("evidence_channels") or {}
    for name, channel in channels.items():
        lines.append(f"- `{name}`: {channel.get('status')} via {channel.get('method')}")
    lines.extend(["", "## Header-Derived Scan Recipes"])
    precedent_candidates = ((channels.get("scan_precedents") or {}).get("top_candidates") or [])[:5]
    public_patterns = ((channels.get("scan_precedents") or {}).get("user_facing_top_patterns") or [])[:5]
    recipe_lines = 0
    for index, candidate in enumerate(precedent_candidates):
        recipes = candidate.get("scan_recipe_examples") or []
        if not recipes:
            continue
        public = public_patterns[index] if index < len(public_patterns) else {}
        lines.append(f"- {public.get('pattern_id') or 'precedent_pattern'} ({candidate.get('confidence_label')}, score {candidate.get('score')})")
        for recipe in recipes[:3]:
            ranges = "; ".join(_format_recipe_range(item) for item in recipe.get("ranges") or [])
            title = recipe.get("title") or recipe.get("fields", {}).get("title") or ""
            jtype = recipe.get("jtype") or recipe.get("fields", {}).get("jtype") or ""
            npts = recipe.get("npts") or recipe.get("fields", {}).get("npts") or ""
            counts = recipe.get("counts") or recipe.get("fields", {}).get("counts") or ""
            count_type = recipe.get("count_type") or recipe.get("fields", {}).get("count_type") or ""
            lines.append(
                f"  - title=`{title}`, jtype=`{jtype}`, npts=`{npts}`, counts=`{counts}`, count_type=`{count_type}`, ranges={ranges or 'none parsed'}"
            )
            recipe_lines += 1
    if not recipe_lines:
        lines.append("- No parsed header scan recipes were available from the precedent channel.")
    lines.extend(["", "## Precedent Privacy Boundary"])
    policy = ((report.get("recommendations") or {}).get("precedent_explanation_policy") or {})
    lines.append(f"- User-facing source: `{policy.get('user_facing_source') or 'none'}`")
    lines.append(f"- {policy.get('privacy_boundary') or 'Private precedent details are not available.'}")
    lines.extend(["", "## Alignment Candidates"])
    for item in (report.get("recommendations") or {}).get("alignment_candidates") or []:
        hkl = ",".join(str(v) for v in item.get("hkl") or [])
        cautions = "; ".join(item.get("cautions") or [])
        lines.append(f"- ({hkl}) A4/two-theta={item.get('two_theta_deg')} deg, Q={item.get('q_inv_angstrom')} 1/A, rel F2={item.get('relative_structure_factor')}" + (f" [{cautions}]" if cautions else ""))
    if not ((report.get("recommendations") or {}).get("alignment_candidates")):
        lines.append("- No alignment candidates were calculated.")
    lines.extend(["", "## Recommendations"])
    recs = report.get("recommendations") or {}
    for section in ("temperature_plan", "configuration_plan", "scan_plan", "replay_plan"):
        lines.extend(["", f"### {section.replace('_', ' ').title()}"])
        for item in recs.get(section) or []:
            lines.append(f"- {item.get('recommendation')}")
            rationale = item.get("probability_style_rationale") or item.get("rationale") or item.get("reason")
            if rationale:
                lines.append(f"  Rationale: {rationale}")
            if item.get("probability") is not None:
                lines.append(f"  Probability-style score: {item.get('probability')}")
    lines.extend(["", "## Missing Inputs To Ask For"])
    for item in recs.get("missing_inputs_to_request") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Confidence", "", f"- Overall probability-style confidence: `{(report.get('confidence') or {}).get('overall_probability_style_confidence')}`"])
    lines.extend(["", "## Warnings"])
    for item in report.get("warnings") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Not Approved Operations"])
    for item in report.get("not_approved_operations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _format_recipe_range(range_record: dict[str, Any]) -> str:
    axis = range_record.get("axis") or "unknown"
    if "first_point" in range_record and "last_point" in range_record:
        return f"{axis}: {range_record['first_point']} -> {range_record['last_point']}"
    if "start" in range_record and "stop" in range_record:
        return f"{axis}: {range_record['start']} -> {range_record['stop']}"
    return f"{axis}: {range_record.get('raw')}"
