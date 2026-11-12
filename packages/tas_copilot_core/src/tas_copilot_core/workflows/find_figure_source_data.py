from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import (
    AuditRecord,
    CandidateEvidence,
    CandidateMismatch,
    FigureAxisConstraint,
    FigurePanelConstraint,
    FigureSourceCandidate,
    FigureSourceQuery,
    FigureSourceReport,
    PSDMapReconstructionResult,
    ProvenanceRecord,
    RejectedCandidate,
    ToolCall,
    ToolResultStatus,
)
from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.registry.tool_inventory import build_tool_inventory
from tas_copilot_core.registry.tool_runner import stable_hash

from .catalog_tas_directory import catalog_tas_directory
from .common import default_runner


WORKFLOW_NAME = "find-figure-source-data"
DEFAULT_Q_TOLERANCE = 0.03


def find_figure_source_data(arguments: dict[str, Any], runner=None, audit_logger: JsonlAuditLogger | None = None) -> dict[str, Any]:
    output_dir = Path(arguments.get("output_dir") or CoreConfig().root / "data" / "processed" / "figure_source").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_logger = audit_logger or JsonlAuditLogger(output_dir / "audit")
    runner = runner or default_runner()

    query = load_figure_source_query(arguments)
    catalog_arg = arguments.get("catalog") or arguments.get("db_path")
    catalog_path = Path(catalog_arg).expanduser() if catalog_arg else None
    data_root = arguments.get("data_root")
    catalog_result: dict[str, Any] | None = None
    if data_root:
        if catalog_path is None:
            catalog_path = output_dir / "figure_source_catalog.sqlite"
        catalog_result = catalog_tas_directory({"path": data_root, "db_path": str(catalog_path), **arguments.get("catalog_args", {})})

    if catalog_path is None or not catalog_path.exists():
        report = _missing_catalog_report(query, catalog_path, output_dir, data_root, audit_logger, arguments)
        _write_reports(report, output_dir)
        return report.model_dump(mode="json")

    catalog = SQLiteDataCatalog(catalog_path)
    scans = catalog.list_scans()
    datafiles = catalog.list_datafiles()
    experiments = catalog.list_experiments()
    mappings = catalog.list_paper_experiment_links()
    warnings = panel_sample_warnings(query)
    interpretations = build_interpretations(query)
    candidates, rejected_candidates = rank_candidates(query, interpretations, scans, datafiles, experiments, mappings)

    if query.attempt_reconstruction or arguments.get("attempt_reconstruction"):
        for candidate in candidates[: int(query.max_reconstruction_candidates)]:
            candidate.reconstruction = attempt_psd_reconstruction(candidate, query, catalog, output_dir, runner)
            if candidate.reconstruction and candidate.reconstruction.status == ToolResultStatus.success.value:
                score = float(candidate.reconstruction.figure_comparison.get("reconstruction_similarity_score", 0.0))
                candidate.score_breakdown["reconstruction_similarity_score"] = score
                candidate.total_score = round(min(1.0, candidate.total_score + 0.1 * score), 4)
                candidate.confidence_label = _confidence(candidate.total_score)

    status = "success" if candidates else ToolResultStatus.missing_input.value
    safe_conclusion = _safe_conclusion(candidates, warnings, query)
    report = FigureSourceReport(
        status=status,
        query=query,
        panel_sample_warnings=warnings,
        interpretations=interpretations,
        candidates=candidates[: int(arguments.get("top_k") or 20)],
        rejected_candidates=rejected_candidates,
        safe_conclusion=safe_conclusion,
        next_actions=_next_actions(candidates, catalog_path, output_dir, data_root),
        artifacts={},
        provenance=[
            ProvenanceRecord(source_type="catalog", source_id=str(catalog_path), method="sqlite_candidate_search"),
            ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="deterministic_figure_source_search"),
        ],
    )
    audit_id = _write_workflow_audit(audit_logger, report.model_dump(mode="json"), arguments)
    report.audit_id = audit_id
    payload = report.model_dump(mode="json")
    payload["catalog_path"] = str(catalog_path)
    payload["catalog_result"] = catalog_result
    payload["tools_discovered"] = build_tool_inventory().as_dict()["summary"]
    payload["tools_used"] = ["tas.data_reader.read_bt7_file"] if query.attempt_reconstruction else []
    payload["audit_log_dir"] = str(output_dir / "audit")
    payload["decision_summary"] = _decision_summary(report)
    payload["artifacts"] = _write_reports(report, output_dir, extra_payload=payload)
    return payload


def load_figure_source_query(arguments: dict[str, Any]) -> FigureSourceQuery:
    raw: dict[str, Any] = {}
    query_file = arguments.get("query_file")
    if query_file:
        with Path(query_file).expanduser().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    raw.update(_cli_query_overrides(arguments))
    if arguments.get("figure_image"):
        raw["figure_image_path"] = str(arguments["figure_image"])
    if not raw.get("query_id"):
        raw["query_id"] = "figure-source-query"
    raw["dates"] = [_parse_date_string(item) for item in raw.get("dates", [])]
    raw.setdefault("alternate_panels", [])
    if raw.get("alternate_panel"):
        raw["alternate_panels"].append(
            {
                "panel_id": str(raw.pop("alternate_panel")),
                "label": None,
                "composition": {"x_mn": raw.pop("alternate_panel_xmn", None)},
            }
        )
    if isinstance(raw.get("q_range"), dict):
        raw["q_range"] = {"axis_name": "Q", "unit": "angstrom^-1", **raw["q_range"]}
    elif raw.get("q_min") is not None or raw.get("q_max") is not None:
        raw["q_range"] = {"axis_name": "Q", "min_value": raw.pop("q_min", None), "max_value": raw.pop("q_max", None), "unit": "angstrom^-1"}
    if isinstance(raw.get("temperature_range"), dict):
        raw["temperature_range"] = {"axis_name": "temperature", "unit": "K", **raw["temperature_range"]}
    elif raw.get("temperature_min") is not None or raw.get("temperature_max") is not None:
        raw["temperature_range"] = {
            "axis_name": "temperature",
            "min_value": raw.pop("temperature_min", None),
            "max_value": raw.pop("temperature_max", None),
            "unit": "K",
        }
    return FigureSourceQuery.model_validate(raw)


def _cli_query_overrides(arguments: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "instrument",
        "mode",
        "sample_type",
        "material_family",
        "sample",
        "target_panel",
        "target_panel_xmn",
        "alternate_panel",
        "alternate_panel_xmn",
        "top_k",
        "attempt_reconstruction",
        "max_reconstruction_candidates",
    }
    raw = {key: value for key, value in arguments.items() if key in keys and value is not None}
    if "mode" in raw:
        raw["measurement_mode"] = raw.pop("mode")
    if "sample" in raw:
        raw["user_sample_formula"] = raw.pop("sample")
    if "target_panel_xmn" in raw:
        raw["target_composition"] = {"x_mn": raw.pop("target_panel_xmn")}
    if "top_k" in raw:
        raw.pop("top_k")
    for source, target in [("date", "dates"), ("experimenter", "experimenters"), ("annotated_q", "annotated_q_values")]:
        if source in arguments and arguments[source] is not None:
            values = arguments[source] if isinstance(arguments[source], list) else [arguments[source]]
            raw[target] = values
    for source, target in [("q_min", "q_min"), ("q_max", "q_max"), ("temperature_min", "temperature_min"), ("temperature_max", "temperature_max")]:
        if source in arguments and arguments[source] is not None:
            raw[target] = arguments[source]
    return raw


def _parse_date_string(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    dotted = re.fullmatch(r"(\d{1,2})\.(\d{1,2})(\d{4})", text)
    if dotted:
        return f"{int(dotted.group(3)):04d}-{int(dotted.group(1)):02d}-{int(dotted.group(2)):02d}"
    dotted = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if dotted:
        return f"{int(dotted.group(3)):04d}-{int(dotted.group(1)):02d}-{int(dotted.group(2)):02d}"
    return text


def panel_sample_warnings(query: FigureSourceQuery) -> list[str]:
    warnings: list[str] = []
    sample_x = composition_xmn(query.user_sample_formula)
    target_x = _composition_value(query.target_composition)
    if sample_x is not None and target_x is not None and abs(sample_x - target_x) > 0.025:
        alternate = next((panel for panel in query.alternate_panels if abs(float(panel.composition.get("x_mn", -999)) - sample_x) <= 0.025), None)
        target = f"panel ({query.target_panel}), x_Mn={target_x:g}"
        sample = f"{query.user_sample_formula}, x_Mn={sample_x:g}"
        if alternate:
            warnings.append(
                f"Panel/sample mismatch: the user-stated sample {sample} is more consistent with panel ({alternate.panel_id}) than {target}."
            )
        else:
            warnings.append(f"Panel/sample mismatch: the user-stated sample {sample} does not match {target}.")
    return warnings


def build_interpretations(query: FigureSourceQuery) -> list[dict[str, Any]]:
    sample_x = composition_xmn(query.user_sample_formula)
    target_x = _composition_value(query.target_composition)
    interpretations = [
        {
            "interpretation_id": f"target_panel_{query.target_panel}_xmn_{target_x}",
            "panel_id": query.target_panel,
            "x_mn": target_x,
            "label": "target_panel_interpretation",
            "aliases": composition_aliases(
                query.material_family,
                target_x,
                query.user_sample_formula if sample_x is not None and target_x is not None and abs(sample_x - target_x) <= 0.025 else None,
            ),
        }
    ]
    for panel in query.alternate_panels:
        x_mn = _composition_value(panel.composition) or sample_x
        interpretations.append(
            {
                "interpretation_id": f"user_sample_panel_{panel.panel_id}_xmn_{x_mn}",
                "panel_id": panel.panel_id,
                "x_mn": x_mn,
                "label": "user_sample_interpretation",
                "aliases": composition_aliases(query.material_family, x_mn, query.user_sample_formula),
            }
        )
    if sample_x is not None and all(abs(float(item.get("x_mn") or -999) - sample_x) > 0.025 for item in interpretations):
        interpretations.append(
            {
                "interpretation_id": f"user_sample_xmn_{sample_x}",
                "panel_id": None,
                "x_mn": sample_x,
                "label": "user_sample_interpretation",
                "aliases": composition_aliases(query.material_family, sample_x, query.user_sample_formula),
            }
        )
    return interpretations


def composition_xmn(text: str | None) -> float | None:
    if not text:
        return None
    patterns = [
        r"Mn[_\{]*\s*([0-9]+(?:\.[0-9]+)?)",
        r"x[_\s-]*Mn[\"']?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"x\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def composition_aliases(material_family: str, x_mn: float | None, user_sample: str | None = None) -> list[str]:
    aliases = {material_family, "LuFeMnO3", "LuMnFeO3"}
    if user_sample:
        aliases.add(user_sample)
    if x_mn is not None:
        fe = max(0.0, 1.0 - float(x_mn))
        fe_percent = int(round(fe * 100))
        mn_percent = int(round(float(x_mn) * 100))
        fe_token = f"{fe:g}".replace(".", "p")
        mn_token = f"{x_mn:g}".replace(".", "p")
        aliases.update(
            {
                f"LuFe{fe:g}Mn{x_mn:g}O3",
                f"LuFe{fe_token}Mn{mn_token}O3",
                f"LuFe{fe_percent}Mn{mn_percent}O3",
                f"LuFe_{{{fe:g}}}Mn_{{{x_mn:g}}}O3",
                f"x_Mn = {x_mn:g}",
                f"xMn={x_mn:g}",
                f"x = {x_mn:g}",
            }
        )
    return sorted(alias for alias in aliases if alias)


def rank_candidates(query, interpretations, scans, datafiles, experiments, mappings) -> tuple[list[FigureSourceCandidate], list[RejectedCandidate]]:
    by_scan: dict[str, list[Any]] = {}
    for datafile in datafiles:
        if datafile.scan_id:
            by_scan.setdefault(datafile.scan_id, []).append(datafile)
    by_experiment = {experiment.experiment_id: experiment for experiment in experiments}
    mapping_by_experiment: dict[str, list[Any]] = {}
    for mapping in mappings:
        mapping_by_experiment.setdefault(mapping.experiment_id, []).append(mapping)

    candidates: list[FigureSourceCandidate] = []
    rejected: list[RejectedCandidate] = []
    for interpretation in interpretations:
        groups: dict[str, list[Any]] = {}
        for scan in scans:
            key = _candidate_group_key(scan)
            groups.setdefault(str(key), []).append(scan)
        if not groups and datafiles:
            for index, datafile in enumerate(datafiles):
                candidates.append(_score_file_candidate(query, interpretation, datafile, index))
        for index, group_scans in enumerate(groups.values()):
            experiment = by_experiment.get(group_scans[0].experiment_id or "")
            group_files = [item for scan in group_scans for item in by_scan.get(scan.scan_id, [])]
            candidate = _score_scan_group_candidate(
                query,
                interpretation,
                group_scans,
                group_files,
                experiment,
                mapping_by_experiment.get(group_scans[0].experiment_id or "", []),
                index,
            )
            rejected_candidate = _elastic_rejection(query, candidate, group_scans)
            if rejected_candidate:
                rejected.append(rejected_candidate)
            else:
                candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.total_score, reverse=True), rejected


def _candidate_group_key(scan) -> str:
    metadata = scan.metadata or {}
    filename = str(metadata.get("filename") or Path(str(metadata.get("path") or "")).name or scan.scan_id)
    stem = Path(filename).stem
    # BT-7 filenames often encode the scan number as a trailing integer. Remove it
    # so temperature-series or repeated scan families are ranked separately from
    # unrelated scans in the same experiment.
    family = re.sub(r"\d+$", "", stem).strip("_-")
    enrichment = metadata.get("bt7_enrichment") if isinstance(metadata.get("bt7_enrichment"), dict) else {}
    scan_description = str(enrichment.get("scan_description") or "")
    # Pair BT-7 temperature sweeps that are named as warming/cooling or open/closed
    # cycle variants. These are usually one figure-source series, not independent
    # scientific candidates.
    family = re.sub(r"(?i)(warm|cool)(oc)?$", "", family).strip("_-")
    if not family:
        family = str(enrichment.get("scan_variable") or scan.scan_type or scan.scan_id)
    if "Range=Temp" in scan_description:
        family = f"{family}:temperature-series"
    experiment = scan.experiment_id or metadata.get("native_experiment_id") or "unknown-experiment"
    return f"{experiment}:{family}"


def _score_scan_group_candidate(query, interpretation, scans, datafiles, experiment, mappings, index: int) -> FigureSourceCandidate:
    text = _blob({"scans": [scan.model_dump(mode="json") for scan in scans], "datafiles": [item.model_dump(mode="json") for item in datafiles], "experiment": experiment.model_dump(mode="json") if experiment else {}})
    evidence: list[CandidateEvidence] = []
    mismatches: list[CandidateMismatch] = []
    missing: set[str] = set()
    breakdown = {
        "date_score": _date_score(query.dates, text, scans, experiment, evidence, missing),
        "instrument_score": _term_score([query.instrument, "BT7", "NCNR BT-7"], text, "instrument", evidence, missing),
        "mode_score": _term_score([query.measurement_mode, "position sensitive detector", "detector array"], text, "PSD mode", evidence, missing),
        "sample_type_score": _term_score([query.sample_type or ""], text, "sample type", evidence, missing),
        "composition_score": _composition_score(interpretation, text, evidence, mismatches, missing),
        "experimenter_score": _term_score(query.experimenters, text, "experimenter", evidence, missing),
        "q_coverage_score": _range_score(query.q_range, _q_range(scans), "Q coverage", evidence, missing),
        "temperature_coverage_score": _range_score(query.temperature_range, _temperature_range(scans), "temperature coverage", evidence, missing),
        "psd_temperature_series_score": _psd_series_score(scans, datafiles, text, evidence, missing),
        "paper_mapping_score": _paper_mapping_score(mappings, evidence),
    }
    total = _weighted_total(breakdown)
    group = _scan_group(scans, datafiles, evidence, missing)
    return FigureSourceCandidate(
        candidate_id=f"candidate-{index + 1}-{interpretation['interpretation_id']}",
        candidate_type="scan_group" if len(scans) > 1 else "scan",
        interpretation=interpretation["interpretation_id"],
        scan_ids=[scan.scan_id for scan in scans],
        datafile_ids=[item.datafile_id for item in datafiles],
        experiment_id=scans[0].experiment_id,
        total_score=total,
        score_breakdown=breakdown,
        confidence_label=_confidence(total),
        evidence=evidence,
        mismatches=mismatches,
        missing_fields=sorted(missing),
        scan_group=group,
        provenance=[ProvenanceRecord(source_type="catalog", source_id=scans[0].scan_id, method="sqlite_candidate_scoring")],
    )


def _score_file_candidate(query, interpretation, datafile, index: int) -> FigureSourceCandidate:
    text = _blob(datafile.model_dump(mode="json"))
    evidence: list[CandidateEvidence] = []
    mismatches: list[CandidateMismatch] = []
    missing: set[str] = {"scan_linkage"}
    breakdown = {
        "date_score": _date_score(query.dates, text, [], None, evidence, missing),
        "instrument_score": _term_score([query.instrument, "BT7", "NCNR BT-7"], text, "instrument", evidence, missing),
        "mode_score": _term_score([query.measurement_mode, "position sensitive detector", "detector array"], text, "PSD mode", evidence, missing),
        "sample_type_score": _term_score([query.sample_type or ""], text, "sample type", evidence, missing),
        "composition_score": _composition_score(interpretation, text, evidence, mismatches, missing),
        "experimenter_score": _term_score(query.experimenters, text, "experimenter", evidence, missing),
        "q_coverage_score": 0.0,
        "temperature_coverage_score": 0.0,
        "psd_temperature_series_score": 0.0,
        "paper_mapping_score": 0.0,
    }
    total = _weighted_total(breakdown)
    return FigureSourceCandidate(
        candidate_id=f"candidate-file-{index + 1}-{interpretation['interpretation_id']}",
        candidate_type="file",
        interpretation=interpretation["interpretation_id"],
        datafile_ids=[datafile.datafile_id],
        experiment_id=datafile.experiment_id,
        total_score=total,
        score_breakdown=breakdown,
        confidence_label=_confidence(total),
        evidence=evidence,
        mismatches=mismatches,
        missing_fields=sorted(missing),
        provenance=[ProvenanceRecord(source_type="datafile", source_id=datafile.datafile_id, method="sqlite_candidate_scoring")],
    )


def _date_score(dates, text, scans, experiment, evidence, missing) -> float:
    if not dates:
        return 0.0
    variants = {variant for date in dates for variant in _date_variants(date)}
    if any(date in text for date in variants):
        evidence.append(CandidateEvidence(evidence_type="date", summary=f"Matched one of {dates}", score=1.0))
        return 1.0
    for scan in scans:
        for key in ["date", "start_date", "end_date"]:
            value = str(scan.metadata.get(key, ""))
            if any(date in value for date in variants):
                evidence.append(CandidateEvidence(evidence_type="date", summary=f"Matched scan {scan.scan_id} date", score=1.0, source_id=scan.scan_id))
                return 1.0
    if experiment and any(date in str(item or "") for date in variants for item in [experiment.start_date, experiment.end_date]):
        evidence.append(CandidateEvidence(evidence_type="date", summary="Matched experiment date", score=1.0, source_id=experiment.experiment_id))
        return 1.0
    missing.add("date")
    return 0.0


def _date_variants(date: str) -> list[str]:
    text = str(date)
    variants = [text]
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        variants.extend([f"{match.group(1)}_{match.group(2)}_{match.group(3)}", f"{match.group(1)} {match.group(2)} {match.group(3)}"])
    return variants


def _term_score(terms, text, label, evidence, missing) -> float:
    terms = [str(term) for term in terms if term]
    if not terms:
        return 0.0
    lower = text.lower()
    matches = [term for term in terms if term.lower() in lower]
    if matches:
        evidence.append(CandidateEvidence(evidence_type=label, summary=f"Matched {', '.join(matches)}", score=1.0))
        return 1.0
    missing.add(label.replace(" ", "_"))
    return 0.0


def _composition_score(interpretation, text, evidence, mismatches, missing) -> float:
    x_mn = interpretation.get("x_mn")
    other = composition_xmn(text)
    if x_mn is not None and other is not None and abs(float(x_mn) - other) > 0.025:
        mismatches.append(CandidateMismatch(mismatch_type="composition", summary=f"Candidate metadata suggests x_Mn={other:g}, interpretation expects x_Mn={float(x_mn):g}."))
        return 0.15
    matches = [alias for alias in interpretation["aliases"] if alias and alias.lower() in text.lower()]
    exact_matches = [
        alias
        for alias in matches
        if "x_mn" in alias.lower() or "xmn" in alias.lower() or re.search(r"LuFe[0-9.]+Mn[0-9.]+O3", alias)
    ]
    if exact_matches:
        evidence.append(CandidateEvidence(evidence_type="composition", summary=f"Matched {', '.join(exact_matches[:4])}", score=1.0))
        return 1.0
    if matches:
        evidence.append(CandidateEvidence(evidence_type="composition", summary=f"Matched broad material-family alias {', '.join(matches[:2])}", score=0.35))
        missing.add("specific_composition")
        return 0.35
    missing.add("composition")
    return 0.0


def _range_score(query_range: FigureAxisConstraint | None, observed: tuple[float, float] | None, label, evidence, missing) -> float:
    if query_range is None:
        return 0.0
    if observed is None:
        missing.add(label.lower().replace(" ", "_"))
        return 0.0
    qmin = query_range.min_value
    qmax = query_range.max_value
    if qmin is None or qmax is None:
        return 0.0
    overlap = max(0.0, min(float(qmax), observed[1]) - max(float(qmin), observed[0]))
    width = max(float(qmax) - float(qmin), 1e-9)
    score = min(1.0, overlap / width)
    if score > 0:
        evidence.append(CandidateEvidence(evidence_type=label, summary=f"Observed {observed[0]:g} to {observed[1]:g} overlaps requested {qmin:g} to {qmax:g}.", score=score))
    else:
        missing.add(label.lower().replace(" ", "_"))
    return round(score, 4)


def _psd_series_score(scans, datafiles, text, evidence, missing) -> float:
    has_psd = any(term in text.lower() for term in ["psd", "position sensitive detector", "detector array"])
    temperatures = _temperature_values(scans)
    if has_psd and len(set(temperatures)) >= 2:
        evidence.append(CandidateEvidence(evidence_type="psd_temperature_series", summary=f"PSD evidence with {len(set(temperatures))} temperatures.", score=1.0))
        return 1.0
    if has_psd and (len(scans) > 1 or len(datafiles) > 1):
        evidence.append(CandidateEvidence(evidence_type="psd_temperature_series", summary="PSD evidence with multiple scans/files but sparse temperature metadata.", score=0.5))
        missing.add("temperature_values")
        return 0.5
    missing.add("psd_temperature_series")
    return 0.0


def _paper_mapping_score(mappings, evidence) -> float:
    if not mappings:
        return 0.0
    best = max(float(mapping.confidence) for mapping in mappings)
    evidence.append(CandidateEvidence(evidence_type="paper_mapping", summary=f"Paper-experiment mapping confidence {best:g}.", score=best))
    return min(1.0, best)


def _weighted_total(breakdown: dict[str, float]) -> float:
    weights = {
        "date_score": 0.15,
        "instrument_score": 0.12,
        "mode_score": 0.12,
        "sample_type_score": 0.08,
        "composition_score": 0.18,
        "experimenter_score": 0.12,
        "q_coverage_score": 0.10,
        "temperature_coverage_score": 0.08,
        "psd_temperature_series_score": 0.05,
        "paper_mapping_score": 0.05,
    }
    denom = sum(weights.values())
    return round(sum(float(breakdown.get(key, 0.0)) * weight for key, weight in weights.items()) / denom, 4)


def _confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    if score > 0:
        return "low"
    return "insufficient"


def _elastic_rejection(query: FigureSourceQuery, candidate: FigureSourceCandidate, scans) -> RejectedCandidate | None:
    if not _requires_elastic(query):
        return None
    nonzero = [value for value in _energy_values(scans) if abs(value) > 1e-6]
    if not nonzero:
        return None
    return RejectedCandidate(
        candidate_id=candidate.candidate_id,
        rejection_reason=f"Rejected for elastic figure-source query because candidate has nonzero energy transfer values: {sorted(set(nonzero))}.",
        rejected_by="validator",
        evidence=[
            CandidateEvidence(
                evidence_type="energy_transfer",
                summary=f"Nonzero energy transfer values {sorted(set(nonzero))} conflict with required elastic E=0.",
                score=0.0,
            )
        ],
        provenance=candidate.provenance,
    )


def _requires_elastic(query: FigureSourceQuery) -> bool:
    if (query.measurement_kind or "").lower() == "elastic":
        return True
    if query.required_energy_transfer is not None and abs(float(query.required_energy_transfer)) <= 1e-6:
        return True
    return "elastic" in " ".join([query.measurement_mode or "", query.notes or ""]).lower()


def _energy_values(scans) -> list[float]:
    values: list[float] = []
    for scan in scans:
        if scan.energy_transfer and scan.energy_transfer.value:
            values.append(float(scan.energy_transfer.value.value))
        metadata = scan.metadata or {}
        transfer = metadata.get("energy_transfer")
        if isinstance(transfer, dict) and transfer.get("value") is not None:
            values.append(float(transfer["value"]))
        energy_range = metadata.get("energy_range")
        if isinstance(energy_range, dict) and energy_range.get("min") is not None and energy_range.get("max") is not None:
            values.extend([float(energy_range["min"]), float(energy_range["max"])])
    return values


def _q_range(scans) -> tuple[float, float] | None:
    values: list[float] = []
    for scan in scans:
        if scan.q_center:
            values.extend([float(scan.q_center.h), float(scan.q_center.k), float(scan.q_center.l)])
        metadata = scan.metadata or {}
        for key in ["q_range", "q_coverage"]:
            value = metadata.get(key)
            if isinstance(value, dict) and value.get("min") is not None and value.get("max") is not None:
                return float(value["min"]), float(value["max"])
        parsed = _parse_q_range_from_description(_scan_description(metadata))
        if parsed:
            return parsed
        axis = metadata.get("q_axis") or (metadata.get("psd_map") or {}).get("q_axis")
        if isinstance(axis, list) and axis:
            values.extend(float(item) for item in axis)
    return (min(values), max(values)) if values else None


def _temperature_range(scans) -> tuple[float, float] | None:
    values = _temperature_values(scans)
    return (min(values), max(values)) if values else None


def _temperature_values(scans) -> list[float]:
    values: list[float] = []
    for scan in scans:
        for value in scan.temperature_range:
            values.append(float(value.value))
        metadata = scan.metadata or {}
        for key in ["temperature", "temperature_k"]:
            if metadata.get(key) is not None:
                values.append(float(metadata[key]))
        for key in ["temperature_values", "temperature_axis"]:
            axis = metadata.get(key) or (metadata.get("psd_map") or {}).get(key)
            if isinstance(axis, list):
                values.extend(float(item) for item in axis)
        temp_range = metadata.get("temperature_range")
        if isinstance(temp_range, dict) and temp_range.get("min") is not None and temp_range.get("max") is not None:
            values.extend([float(temp_range["min"]), float(temp_range["max"])])
        parsed = _parse_temperature_range_from_description(_scan_description(metadata))
        if parsed:
            values.extend(parsed)
    return values


def _scan_description(metadata: dict[str, Any]) -> str:
    enrichment = metadata.get("bt7_enrichment")
    if isinstance(enrichment, dict):
        description = str(enrichment.get("scan_description") or "")
        if description:
            return description
    return str(metadata.get("scan_description") or "")


def _parse_q_range_from_description(description: str) -> tuple[float, float] | None:
    match = re.search(r"Range=Q=([+-]?[0-9.]+)~[+-]?[0-9.]+~[+-]?[0-9.]+\s+([+-]?[0-9.]+)~", description)
    if not match:
        return None
    values = [float(match.group(1)), float(match.group(2))]
    return min(values), max(values)


def _parse_temperature_range_from_description(description: str) -> tuple[float, float] | None:
    match = re.search(r"Range=Temp=([+-]?[0-9.]+)\s+([+-]?[0-9.]+)", description)
    if not match:
        return None
    values = [float(match.group(1)), float(match.group(2))]
    return min(values), max(values)


def _scan_group(scans, datafiles, evidence, missing) -> dict[str, Any]:
    temperatures = sorted(set(_temperature_values(scans)))
    q_range = _q_range(scans)
    return {
        "group_id": f"group-{scans[0].experiment_id or scans[0].scan_id}",
        "member_scan_ids": [scan.scan_id for scan in scans],
        "member_datafile_ids": [item.datafile_id for item in datafiles],
        "temperature_values": temperatures,
        "q_coverage": {"min": q_range[0], "max": q_range[1]} if q_range else {},
        "configuration_summary": {"instrument": scans[0].instrument, "scan_type": scans[0].scan_type},
        "missing_metadata": sorted(missing),
        "grouping_confidence": 0.8 if len(scans) > 1 and temperatures else 0.4,
        "evidence": [item.model_dump(mode="json") for item in evidence if item.evidence_type in {"psd_temperature_series", "Q coverage", "temperature coverage"}],
    }


def attempt_psd_reconstruction(candidate: FigureSourceCandidate, query: FigureSourceQuery, catalog: SQLiteDataCatalog, output_dir: Path, runner) -> PSDMapReconstructionResult:
    scans = [catalog.get_scan(scan_id) for scan_id in candidate.scan_ids]
    scans = [scan for scan in scans if scan is not None]
    datafiles = [item for scan_id in candidate.scan_ids for item in catalog.datafiles_for_scan(scan_id)]
    reconstruction_dir = output_dir / "reconstructions"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    path_only = [item.datafile_id for item in datafiles if (item.metadata or {}).get("metadata_mode") == "path_only" or (item.metadata or {}).get("hydration_status") in {"path_only", "not_attempted"}]
    if path_only:
        return PSDMapReconstructionResult(
            status=ToolResultStatus.missing_input.value,
            candidate_id=candidate.candidate_id,
            input_files=[item.path for item in datafiles],
            input_scans=[scan.scan_id for scan in scans],
            warnings=["Candidate files are path-only; hydrate/read contents before reconstruction."],
            missing_fields=["hydrated_data"],
            provenance=[ProvenanceRecord(source_type="candidate", source_id=candidate.candidate_id, method="path_only_guard")],
        )
    synthetic = _synthetic_psd_map(scans)
    if synthetic:
        result = _reconstruct_from_explicit_map(candidate, query, synthetic, reconstruction_dir)
        return result
    if not datafiles:
        return PSDMapReconstructionResult(
            status=ToolResultStatus.missing_input.value,
            candidate_id=candidate.candidate_id,
            input_scans=[scan.scan_id for scan in scans],
            missing_fields=["data_files"],
            warnings=["No source files are linked to this candidate."],
            provenance=[ProvenanceRecord(source_type="candidate", source_id=candidate.candidate_id, method="input_validation")],
        )
    read_results = [
        runner.run(ToolCall(tool_namespace="tas.data_reader", tool_name="read_bt7_file", arguments={"path": item.path}, caller_workflow=WORKFLOW_NAME))
        for item in datafiles
    ]
    if any(result.status == ToolResultStatus.unavailable for result in read_results):
        return PSDMapReconstructionResult(
            status=ToolResultStatus.unavailable.value,
            candidate_id=candidate.candidate_id,
            input_files=[item.path for item in datafiles],
            input_scans=[scan.scan_id for scan in scans],
            warnings=["The BT-7 reader plugin is unavailable or did not return reconstructable PSD arrays."],
            missing_fields=["reader_psd_arrays", "q_axis", "temperature_axis"],
            provenance=[ProvenanceRecord(source_type="tool", source_id="tas.data_reader.read_bt7_file", method="registry_reader_call")],
            audit_ids=[result.audit_id for result in read_results if result.audit_id],
        )
    partial = _partial_reader_psd_map(read_results)
    if partial:
        artifact_path = reconstruction_dir / f"partial_reconstruction_{_safe_filename(candidate.candidate_id)}.json"
        artifact_path.write_text(json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8")
        return PSDMapReconstructionResult(
            status=ToolResultStatus.missing_input.value,
            candidate_id=candidate.candidate_id,
            input_files=[item.path for item in datafiles],
            input_scans=[scan.scan_id for scan in scans],
            q_axis_source=None,
            temperature_source=partial.get("temperature_axis_source"),
            intensity_source=partial.get("intensity_source"),
            shape=partial.get("shape", []),
            output_artifact_paths=[str(artifact_path)],
            warnings=[
                "Reader exposed PSD channel/two-theta by temperature intensity data, but no calibrated q_axis.",
                "The source data were probably measured in A4/sample two-theta or PSD channel coordinates; no PSD channel-to-A4 calibration was guessed.",
                *partial.get("warnings", []),
            ],
            missing_fields=["q_axis"],
            provenance=[ProvenanceRecord(source_type="tool", source_id="tas.data_reader.read_bt7_file", method="registry_reader_call")],
            audit_ids=[result.audit_id for result in read_results if result.audit_id],
        )
    return PSDMapReconstructionResult(
        status=ToolResultStatus.missing_input.value,
        candidate_id=candidate.candidate_id,
        input_files=[item.path for item in datafiles],
        input_scans=[scan.scan_id for scan in scans],
        warnings=["Reader output did not include explicit q_axis, temperature_axis, and intensity arrays; no calibration was guessed."],
        missing_fields=["q_axis", "temperature_axis", "intensity"],
        provenance=[ProvenanceRecord(source_type="tool", source_id="tas.data_reader.read_bt7_file", method="registry_reader_call")],
        audit_ids=[result.audit_id for result in read_results if result.audit_id],
    )


def _partial_reader_psd_map(read_results) -> dict[str, Any] | None:
    maps = []
    for result in read_results:
        output = result.output if isinstance(result.output, dict) else {}
        psd_map = output.get("psd_map")
        if not isinstance(psd_map, dict):
            continue
        intensity = psd_map.get("intensity")
        temperature_axis = psd_map.get("temperature_axis")
        if intensity and temperature_axis:
            maps.append(
                {
                    "source_tool": f"{result.tool_namespace}.{result.tool_name}",
                    "temperature_axis": temperature_axis,
                    "temperature_axis_source": psd_map.get("temperature_axis_source"),
                    "two_theta_axis": psd_map.get("two_theta_axis"),
                    "two_theta_axis_source": psd_map.get("two_theta_axis_source"),
                    "psd_channel_axis": psd_map.get("psd_channel_axis"),
                    "psd_columns": psd_map.get("psd_columns"),
                    "intensity": intensity,
                    "intensity_source": psd_map.get("intensity_source"),
                    "q_axis": psd_map.get("q_axis"),
                    "q_axis_source": psd_map.get("q_axis_source"),
                    "shape": [len(intensity), len(intensity[0]) if intensity and isinstance(intensity[0], list) else 0],
                    "warnings": psd_map.get("warnings", []),
                }
            )
    if not maps:
        return None
    if len(maps) == 1:
        return maps[0]
    return {
        "source_tool": "tas.data_reader.read_bt7_file",
        "component_maps": maps,
        "temperature_axis_source": ",".join(sorted({str(item.get("temperature_axis_source")) for item in maps if item.get("temperature_axis_source")})),
        "intensity_source": ",".join(sorted({str(item.get("intensity_source")) for item in maps if item.get("intensity_source")})),
        "shape": [sum(item["shape"][0] for item in maps), max(item["shape"][1] for item in maps)],
        "warnings": ["Multiple reader PSD maps were retained as component channel/two-theta maps; no Q calibration was inferred."],
    }


def _synthetic_psd_map(scans) -> dict[str, Any] | None:
    for scan in scans:
        psd_map = (scan.metadata or {}).get("psd_map")
        if isinstance(psd_map, dict) and all(key in psd_map for key in ["q_axis", "temperature_axis", "intensity"]):
            return psd_map
    return None


def _reconstruct_from_explicit_map(candidate, query, psd_map, reconstruction_dir: Path) -> PSDMapReconstructionResult:
    q_axis = [float(item) for item in psd_map["q_axis"]]
    temperature_axis = [float(item) for item in psd_map["temperature_axis"]]
    intensity = psd_map["intensity"]
    comparison = _compare_reconstruction_to_query(query, q_axis, temperature_axis, intensity)
    artifact_path = reconstruction_dir / f"reconstruction_{_safe_filename(candidate.candidate_id)}.json"
    payload = {
        "candidate_id": candidate.candidate_id,
        "q_axis": q_axis,
        "temperature_axis": temperature_axis,
        "intensity": intensity,
        "figure_comparison": comparison,
        "warning": "Synthetic or explicitly cataloged PSD arrays were used; no reader calibration was inferred.",
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return PSDMapReconstructionResult(
        status=ToolResultStatus.success.value,
        candidate_id=candidate.candidate_id,
        input_scans=candidate.scan_ids,
        q_axis_source="explicit_catalog_metadata",
        temperature_source="explicit_catalog_metadata",
        intensity_source="explicit_catalog_metadata",
        q_range={"min": min(q_axis), "max": max(q_axis)},
        temperature_range={"min": min(temperature_axis), "max": max(temperature_axis)},
        shape=[len(temperature_axis), len(q_axis)],
        output_artifact_paths=[str(artifact_path)],
        warnings=["No PSD channel-to-A4 calibration was inferred; reconstruction used explicit q_axis metadata."],
        figure_comparison=comparison,
        provenance=[ProvenanceRecord(source_type="scan", source_id=",".join(candidate.scan_ids), method="explicit_psd_map_reconstruction")],
    )


def _compare_reconstruction_to_query(query, q_axis, temperature_axis, intensity) -> dict[str, Any]:
    matched: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    score_parts: list[float] = []
    if query.q_range:
        q_overlap = _numeric_overlap((min(q_axis), max(q_axis)), (float(query.q_range.min_value), float(query.q_range.max_value)))
        score_parts.append(q_overlap)
        (matched if q_overlap > 0 else mismatched).append("q_range")
    if query.temperature_range:
        t_overlap = _numeric_overlap((min(temperature_axis), max(temperature_axis)), (float(query.temperature_range.min_value), float(query.temperature_range.max_value)))
        score_parts.append(t_overlap)
        (matched if t_overlap > 0 else mismatched).append("temperature_range")
    for annotated_q in query.annotated_q_values:
        if any(abs(float(q) - float(annotated_q)) <= DEFAULT_Q_TOLERANCE for q in q_axis):
            matched.append(f"annotated_q_{annotated_q:g}")
            score_parts.append(1.0)
        else:
            missing.append(f"annotated_q_{annotated_q:g}")
            score_parts.append(0.0)
    return {
        "reconstruction_similarity_score": round(sum(score_parts) / len(score_parts), 4) if score_parts else 0.0,
        "matched_features": matched,
        "mismatched_features": mismatched,
        "missing_features": missing,
        "warnings": ["This is a constraint comparison, not image registration."],
    }


def _numeric_overlap(observed: tuple[float, float], requested: tuple[float, float]) -> float:
    overlap = max(0.0, min(observed[1], requested[1]) - max(observed[0], requested[0]))
    return min(1.0, overlap / max(requested[1] - requested[0], 1e-9))


def _missing_catalog_report(query, catalog_path, output_dir, data_root, audit_logger, arguments) -> FigureSourceReport:
    message = "A usable catalog or --data-root is required before figure-source candidate search can run."
    if data_root:
        message = f"Data-root cataloging did not create a usable catalog at {catalog_path}."
    report = FigureSourceReport(
        status=ToolResultStatus.missing_input.value,
        query=query,
        panel_sample_warnings=panel_sample_warnings(query),
        interpretations=build_interpretations(query),
        candidates=[],
        safe_conclusion="No source-data candidates were searched because the TAS data catalog is missing.",
        next_actions=[
            "Create a local TAS catalog from hydrated or path-only data before running this workflow.",
            f"Example: python -m tas_copilot_core.cli workflow find-figure-source-data --data-root [TAS_DATA_DIR] --catalog {catalog_path or '[CATALOG_DB]'} --output-dir {output_dir} --query-file [QUERY_YAML]",
        ],
        provenance=[ProvenanceRecord(source_type="workflow", source_id=WORKFLOW_NAME, method="input_validation")],
    )
    report.audit_id = _write_workflow_audit(audit_logger, report.model_dump(mode="json"), arguments)
    return report


def _write_reports(report: FigureSourceReport, output_dir: Path, extra_payload: dict[str, Any] | None = None) -> dict[str, str]:
    payload = extra_payload or report.model_dump(mode="json")
    candidates = [item.model_dump(mode="json") for item in report.candidates]
    rejected_candidates = [item.model_dump(mode="json") for item in report.rejected_candidates]
    query_path = output_dir / "figure_source_query.json"
    candidates_path = output_dir / "figure_source_candidates.json"
    rejected_path = output_dir / "rejected_candidates.json"
    report_json_path = output_dir / "figure_source_report.json"
    report_md_path = output_dir / "figure_source_report.md"
    table_csv_path = output_dir / "candidate_table.csv"
    table_md_path = output_dir / "candidate_table.md"
    review_template_path = output_dir / "figure_source_review_template.jsonl"
    artifacts = {
        "query_json": str(query_path),
        "candidates_json": str(candidates_path),
        "rejected_candidates_json": str(rejected_path),
        "report_json": str(report_json_path),
        "report_markdown": str(report_md_path),
        "candidate_table_csv": str(table_csv_path),
        "candidate_table_markdown": str(table_md_path),
        "figure_source_review_template_jsonl": str(review_template_path),
    }
    payload["artifacts"] = artifacts
    query_path.write_text(report.query.model_dump_json(indent=2), encoding="utf-8")
    candidates_path.write_text(json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected_candidates, indent=2, sort_keys=True), encoding="utf-8")
    _write_figure_source_review_template(report, review_template_path)
    report_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_md_path.write_text(_markdown_report(payload), encoding="utf-8")
    _write_candidate_tables(report.candidates, table_csv_path, table_md_path)
    return artifacts


def _write_figure_source_review_template(report: FigureSourceReport, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    query = report.query
    for index, candidate in enumerate(report.candidates, start=1):
        rows.append(
            {
                "paper_id": getattr(query, "paper_id", None),
                "figure_id": getattr(query, "figure_id", None),
                "panel_id": query.target_panel or "unknown-panel",
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "candidate_rank": index,
                "score": candidate.total_score,
                "confidence_label": candidate.confidence_label,
                "candidate_status": "candidate",
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence", "ambiguous", "not_source", "duplicate"],
                "evidence_channels_relied_on": ["evidence_channel_panel_label", "evidence_channel_material_identity", "evidence_channel_scan_family"],
                "source_workflow": WORKFLOW_NAME,
                "source_report": str(path.with_name("figure_source_report.json")),
                "review_notes": "",
                "human_review_required": True,
                "provenance": "find-figure-source-data review template",
            }
        )
    rejected_rank_start = len(rows)
    for index, rejected in enumerate(report.rejected_candidates, start=1):
        rows.append(
            {
                "paper_id": getattr(query, "paper_id", None),
                "figure_id": getattr(query, "figure_id", None),
                "panel_id": query.target_panel or "unknown-panel",
                "candidate_id": rejected.candidate_id,
                "candidate_type": "rejected_candidate",
                "candidate_rank": rejected_rank_start + index,
                "score": 0.0,
                "confidence_label": "rejected_by_workflow",
                "candidate_status": "rejected_by_workflow",
                "rejected_reason": rejected.rejection_reason,
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence", "ambiguous", "not_source", "duplicate"],
                "evidence_channels_relied_on": ["evidence_channel_scan_family"],
                "source_workflow": WORKFLOW_NAME,
                "source_report": str(path.with_name("figure_source_report.json")),
                "review_notes": "",
                "human_review_required": True,
                "provenance": "find-figure-source-data review template",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_candidate_tables(candidates: list[FigureSourceCandidate], csv_path: Path, md_path: Path) -> None:
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "type": candidate.candidate_type,
            "interpretation": candidate.interpretation,
            "score": candidate.total_score,
            "confidence": candidate.confidence_label,
            "scan_ids": ";".join(candidate.scan_ids),
            "datafile_ids": ";".join(candidate.datafile_ids),
            "missing_fields": ";".join(candidate.missing_fields),
        }
        for candidate in candidates
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "type", "interpretation", "score", "confidence", "scan_ids", "datafile_ids", "missing_fields"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Candidate | Type | Score | Confidence | Interpretation |", "| --- | --- | ---: | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['candidate_id']}` | {row['type']} | {row['score']} | {row['confidence']} | `{row['interpretation']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Figure Source Data Candidate Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Query: `{report.get('query', {}).get('query_id')}`",
        f"- Target panel: `({report.get('query', {}).get('target_panel')})`",
        f"- User sample: `{report.get('query', {}).get('user_sample_formula')}`",
        f"- Catalog: `{report.get('catalog_path')}`",
        f"- Workflow audit ID: `{report.get('audit_id')}`",
        "",
        "## Panel/Sample Mismatch",
        "",
    ]
    warnings = report.get("panel_sample_warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None detected.")
    decision = report.get("decision_summary") or {}
    lines.extend(["", "## Decision Summary", ""])
    if decision:
        lines.append(f"- Top candidate: `{decision.get('top_candidate_id')}`")
        lines.append(f"- Top candidate status: {decision.get('top_candidate_status')}")
        lines.append(f"- Rejected incompatible candidates: {decision.get('rejected_candidate_count')}")
        lines.append(f"- Calibration blocker: {decision.get('calibration_blocker')}")
        if decision.get("top_candidate_missing_fields"):
            lines.append(f"- Top candidate missing fields: `{', '.join(decision['top_candidate_missing_fields'])}`")
        if decision.get("top_candidate_scan_ids"):
            lines.append(f"- Top candidate scans: `{', '.join(decision['top_candidate_scan_ids'])}`")
    else:
        lines.append("- No decision summary available.")
    lines.extend(["", "## Candidate Ranking", ""])
    candidates = report.get("candidates", [])
    if candidates:
        lines.extend(["| Rank | Candidate | Score | Confidence | Type | Interpretation |", "| ---: | --- | ---: | --- | --- | --- |"])
        for rank, candidate in enumerate(candidates, start=1):
            lines.append(
                f"| {rank} | `{candidate['candidate_id']}` | {candidate['total_score']} | {candidate['confidence_label']} | {candidate['candidate_type']} | `{candidate['interpretation']}` |"
            )
    else:
        lines.append("- No candidates found.")
    rejected = report.get("rejected_candidates", [])
    lines.extend(["", "## Rejected Candidates", ""])
    if rejected:
        lines.extend(["| Candidate | Rejected by | Reason |", "| --- | --- | --- |"])
        for item in rejected[:20]:
            lines.append(f"| `{item.get('candidate_id')}` | `{item.get('rejected_by')}` | {item.get('rejection_reason')} |")
    else:
        lines.append("- No candidates rejected.")
    lines.extend(["", "## Top Candidate Details", ""])
    for candidate in candidates[:5]:
        lines.extend(
            [
                f"### `{candidate['candidate_id']}`",
                f"- Scans: `{', '.join(candidate.get('scan_ids', [])) or 'none'}`",
                f"- Files: `{', '.join(candidate.get('datafile_ids', [])) or 'none'}`",
                f"- Missing metadata: `{', '.join(candidate.get('missing_fields', [])) or 'none'}`",
                "- Evidence:",
            ]
        )
        evidence = candidate.get("evidence", [])
        lines.extend(f"  - {item['summary']} (score {item.get('score', 0)})" for item in evidence[:8]) if evidence else lines.append("  - None")
        mismatches = candidate.get("mismatches", [])
        if mismatches:
            lines.append("- Mismatches:")
            lines.extend(f"  - {item['summary']}" for item in mismatches)
        reconstruction = candidate.get("reconstruction")
        if reconstruction:
            lines.append(f"- Reconstruction status: `{reconstruction.get('status')}`")
            if reconstruction.get("missing_fields"):
                lines.append(f"- Reconstruction missing fields: `{', '.join(reconstruction['missing_fields'])}`")
    lines.extend(["", "## Safe Conclusion", "", report.get("safe_conclusion", "")])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in report.get("next_actions", []))
    return "\n".join(lines) + "\n"


def _write_workflow_audit(audit_logger: JsonlAuditLogger | None, report: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    if audit_logger is None:
        return None
    audit = AuditRecord(
        event_type="workflow",
        status=report["status"],
        input_hash=stable_hash({"workflow_name": WORKFLOW_NAME, "arguments": arguments}),
        output_hash=stable_hash(report),
        details={
            "workflow_name": WORKFLOW_NAME,
            "input_arguments": arguments,
            "candidate_count": len(report.get("candidates", [])),
            "provenance": report.get("provenance", []),
        },
    )
    audit_logger.write(audit)
    return audit.audit_id


def _safe_conclusion(candidates, warnings, query) -> str:
    prefix = " ".join(warnings) + " " if warnings else ""
    if not candidates:
        return prefix + "No candidate source data were found in the catalog. Do not make a figure-source claim without hydrated data and reconstruction."
    top = candidates[0]
    if top.reconstruction and top.reconstruction.status == ToolResultStatus.success.value:
        return (
            prefix
            + f"The system found candidate source data headed by {top.candidate_id}. Reconstruction succeeded for the candidate, but this remains a constraint comparison and requires human review before claiming it generated the figure panel."
        )
    return (
        prefix
        + f"The system found {len(candidates)} candidate source-data objects. Candidate {top.candidate_id} is the strongest match, but hydrated reconstruction is required before claiming it generated panel ({query.target_panel})."
    )


def _decision_summary(report: FigureSourceReport) -> dict[str, Any]:
    if not report.candidates:
        return {
            "top_candidate_id": None,
            "top_candidate_status": "No candidate source data were ranked.",
            "rejected_candidate_count": len(report.rejected_candidates),
            "calibration_blocker": "Q x temperature reconstruction remains blocked without accepted PSD channel-to-A4/sample two-theta calibration and a reviewed elastic Q-conversion adapter.",
        }
    top = report.candidates[0]
    status_parts = ["candidate source data, not a definitive panel-source claim"]
    if top.reconstruction:
        status_parts.append(f"reconstruction status: {top.reconstruction.status}")
    else:
        status_parts.append("no reconstruction attempted")
    if "q_coverage" in top.missing_fields:
        status_parts.append("Q coverage missing or uncalibrated")
    if "specific_composition" in top.missing_fields or "composition" in top.missing_fields:
        status_parts.append("composition unresolved from catalog metadata")
    return {
        "top_candidate_id": top.candidate_id,
        "top_candidate_status": "; ".join(status_parts),
        "top_candidate_score": top.total_score,
        "top_candidate_scan_ids": top.scan_ids,
        "top_candidate_datafile_ids": top.datafile_ids,
        "top_candidate_missing_fields": top.missing_fields,
        "rejected_candidate_count": len(report.rejected_candidates),
        "rejected_candidate_reasons": sorted({item.rejection_reason for item in report.rejected_candidates}),
        "calibration_blocker": "Detector-channel/angle data must not be called Q-space until an accepted PSD calibration and reviewed conversion adapter are available.",
    }


def _next_actions(candidates, catalog_path, output_dir, data_root) -> list[str]:
    actions = [
        f"Review `{output_dir / 'candidate_table.md'}` and hydrate source files for the highest-ranked path-only candidates.",
        f"Rerun with `--attempt-reconstruction --max-reconstruction-candidates 5` after Q-axis, temperature-axis, and PSD intensity metadata are available.",
    ]
    if not data_root:
        actions.append(
            f"When a real data root is available, run: python -m tas_copilot_core.cli workflow find-figure-source-data --data-root [TAS_DATA_DIR] --catalog {catalog_path} --output-dir {output_dir} --query-file examples/figure_queries/lu_femn_o3_bt7_psd_2013_panel_k.yaml --attempt-reconstruction"
        )
    if candidates:
        actions.append(f"Inspect top candidate scan IDs: {', '.join(candidates[0].scan_ids) or 'no scan IDs linked'}")
    return actions


def _composition_value(composition: dict[str, Any] | None) -> float | None:
    if not composition:
        return None
    value = composition.get("x_mn") or composition.get("xMn") or composition.get("x")
    return float(value) if value is not None else None


def _blob(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "candidate"
