from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_core.corpus.fixtures import compile_figure_query, corpus_fixture_report, load_corpus_fixtures, validate_source_support
from tas_copilot_core.material_identity import parse_material_text
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.source_triage import (
    classify_source_family,
    group_psd_sources,
    q_temperature_reconstruction_blocked,
    reconstruct_channel_temperature,
    triage_records,
)
from tas_copilot_core.workflows.figure_source_triage import _load_query, _load_records
from tas_copilot_core.workflows.catalog_tas_directory import catalog_tas_directory


DEFAULT_QUERY = "examples/figure_queries/lu_femn_o3_elastic_psd_triage.yaml"
DEFAULT_DATA_FIXTURE = "eval/figure_source_lu_femn_o3_triage.yaml"
DEFAULT_CORPUS_FIXTURE = "tests/fixtures/lu_femn_o3_paper_figure"


def source_triage_doctor(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    root = Path(arguments.get("root") or Path.cwd())
    checks = {
        "material_identity_module": _module_available("tas_copilot_core.material_identity"),
        "lu_femn_o3_family_grammar": (root / "configs/material_identity/family_patterns.yaml").exists(),
        "psd_source_family_classifier": _callable_available("tas_copilot_core.source_triage", "classify_source_family"),
        "psd_grouping": _callable_available("tas_copilot_core.source_triage", "group_psd_sources"),
        "psd_channel_temperature_reconstruction": _callable_available("tas_copilot_core.source_triage", "reconstruct_channel_temperature"),
        "detector_cut_diagnostic": _callable_available("tas_copilot_core.source_triage", "plot_psd_detector_cuts"),
        "figure_source_triage_workflow": _module_available("tas_copilot_core.workflows.figure_source_triage"),
        "example_query": (root / DEFAULT_QUERY).exists(),
        "regression_fixture": (root / DEFAULT_DATA_FIXTURE).exists(),
        "q_temperature_requires_reviewed_calibration": q_temperature_reconstruction_blocked("doctor").status == "unsupported",
        "live_control_disabled_by_policy": True,
    }
    missing = [name for name, ok in checks.items() if not ok]
    return {"workflow_name": "source-triage-doctor", "status": "ok" if not missing else "warning", "checks": checks, "missing": missing}


def run_lu_femn_o3_regression(arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = Path(arguments.get("suite") or "eval/figure_source_lu_femn_o3_known_answer.yaml")
    suite = _load_yaml(suite_path)
    data_catalog = suite.get("data_catalog") or DEFAULT_DATA_FIXTURE
    query_file = suite.get("query_file") or DEFAULT_QUERY
    corpus_fixture = suite.get("corpus_fixture") or DEFAULT_CORPUS_FIXTURE
    corpus_catalog = output_dir / "corpus_catalog"

    records = _load_records({"catalog": data_catalog})
    query = _load_query({"query_file": query_file})
    triage = triage_records(query, records, output_dir=output_dir / "triage")
    corpus_report = load_corpus_fixtures(corpus_fixture, corpus_catalog)

    checks = []
    checks.extend(_material_checks())
    checks.extend(_panel_checks(corpus_catalog))
    checks.extend(_classification_checks(records))
    checks.extend(_triage_checks(triage))
    checks.extend(_reconstruction_checks(records, output_dir / "reconstruction"))
    checks.extend(_corpus_checks(corpus_report))
    checks.append(_check("q_temperature_blocked", q_temperature_reconstruction_blocked("known-answer").status == "unsupported", {"missing_fields": q_temperature_reconstruction_blocked("known-answer").missing_fields}))
    checks.append(_check("live_control_not_invoked", True, {"policy": "no live-control workflow is part of this regression"}))

    failed = [item for item in checks if item["status"] != "pass"]
    report = {
        "workflow_name": "lu-femn-o3-regression",
        "status": "pass" if not failed else "fail",
        "suite": str(suite_path),
        "data_catalog": str(data_catalog),
        "corpus_fixture": str(corpus_fixture),
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
        "candidate_ranking": triage["source_triage_state"].get("candidates", []),
        "rejected_candidates": triage["source_triage_state"].get("rejected_candidates", []),
        "source_support_summary": corpus_fixture_report(corpus_catalog),
        "reconstruction_summary": [item for item in checks if item["name"].startswith("reconstruction") or item["name"].startswith("q_temperature")],
        "diagnostics_summary": {"status": "not_run", "reason": "diagnostic workflows are covered by unit tests and local-only real-data commands"},
        "provenance_summary": _provenance_summary(triage, corpus_report),
        "known_limitations": [
            "Synthetic fixtures do not contain raw BT-7 data.",
            "Q x temperature reconstruction remains blocked without reviewed PSD calibration.",
            "Candidate links remain proposed until human review or reconstruction evidence is accepted.",
        ],
    }
    (output_dir / "expected_vs_actual.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "candidate_ranking.json").write_text(json.dumps(report["candidate_ranking"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "rejected_candidates.json").write_text(json.dumps(report["rejected_candidates"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "source_support_summary.json").write_text(json.dumps(report["source_support_summary"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "reconstruction_summary.json").write_text(json.dumps(report["reconstruction_summary"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "diagnostics_summary.json").write_text(json.dumps(report["diagnostics_summary"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "provenance_summary.json").write_text(json.dumps(report["provenance_summary"], indent=2, sort_keys=True), encoding="utf-8")
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_regression_markdown,
        json_filename="lu_femn_o3_regression_report.json",
        markdown_filename="lu_femn_o3_regression_report.md",
    )
    return report


def link_figure_panel_to_source(arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_catalog = Path(arguments["corpus_catalog"]).expanduser()
    data_catalog = arguments["data_catalog"]
    figure_id = arguments["figure_id"]
    panel_id = arguments["panel_id"]
    if arguments.get("data_root"):
        catalog_tas_directory({"path": arguments["data_root"], "db_path": str(data_catalog), "max_files": arguments.get("max_files"), "max_records": arguments.get("max_records")})
    compiled = compile_figure_query(
        corpus_catalog,
        paper_id=arguments.get("paper_id"),
        figure_id=figure_id,
        panel_id=panel_id,
        output=output_dir / "compiled_panel_query.json",
        user_sample=arguments.get("user_sample"),
    )
    if compiled.get("status") != "ok":
        return {"workflow_name": "link-figure-panel-to-source", "status": "missing_input", "missing_fields": compiled.get("missing_fields", []), "warnings": compiled.get("warnings", [])}
    panel = compiled["panel"]
    query = _load_query({"query": compiled["query"]})
    records = _load_records({"catalog": data_catalog})
    triage = triage_records(query, records)
    candidates = []
    for candidate in triage["source_triage_state"].get("candidates", [])[: int(arguments.get("top_k", 5))]:
        candidates.append(
            {
                "panel_id": panel_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "score": candidate["total_score"],
                "confidence_label": candidate.get("confidence_label"),
                "status": "proposed",
                "evidence": candidate.get("evidence", []),
                "mismatches": candidate.get("mismatches", []),
                "missing_fields": candidate.get("missing_fields", []),
                "rejected_reason": None,
                "reconstruction_required": True,
                "human_review_required": True,
                "provenance": candidate.get("provenance", []),
                "audit_ids": candidate.get("audit_ids", []),
            }
        )
    support = validate_source_support(corpus_catalog)
    report = {
        "workflow_name": "link-figure-panel-to-source",
        "status": "ok",
        "paper_id": compiled.get("figure", {}).get("paper_id"),
        "figure_id": figure_id,
        "panel_id": panel_id,
        "panel": panel,
        "compiled_query": compiled["query"],
        "query_warnings": compiled.get("warnings", []),
        "candidate_links": candidates,
        "rejected_candidates": triage["source_triage_state"].get("rejected_candidates", []),
        "source_support_summary": support,
        "safe_conclusion": "Candidate links are proposed only; accepted source links require reconstruction/provenance or human review.",
    }
    (output_dir / "compiled_panel_query.json").write_text(json.dumps(compiled["query"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "figure_panel_source_candidates.json").write_text(json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "source_support_summary.json").write_text(json.dumps(support, indent=2, sort_keys=True), encoding="utf-8")
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_link_markdown, json_filename="figure_panel_source_linkage.json", markdown_filename="figure_panel_source_linkage.md")
    return report


def _material_checks() -> list[dict[str, Any]]:
    expected = {
        "LuFe75Mn25O3": 0.25,
        "LuFe60Mn40O3": 0.4,
        "LuFe50Mn50O3": 0.5,
        "LuFe0.6Mn0.4O3": 0.4,
    }
    checks = []
    for text, value in expected.items():
        parsed = parse_material_text(text)
        checks.append(_check(f"material_{text}", parsed.x_variables.get("x_Mn") == value, parsed.model_dump(mode="json")))
    for text in ["LuFeMnO3", "LuMnFeO3"]:
        parsed = parse_material_text(text)
        checks.append(_check(f"material_{text}_ambiguous", bool(parsed.ambiguity_reason) and not parsed.x_variables, parsed.model_dump(mode="json")))
    return checks


def _panel_checks(corpus_catalog: Path) -> list[dict[str, Any]]:
    panels = _load_jsonl(corpus_catalog / "figure_panels.jsonl")
    by_id = {item["panel_id"]: item for item in panels}
    checks = [
        _check("panel_i_xmn_025", by_id.get("i", {}).get("composition", {}).get("value") == 0.25, by_id.get("i", {})),
        _check("panel_j_xmn_04", by_id.get("j", {}).get("composition", {}).get("value") == 0.4, by_id.get("j", {})),
        _check("panel_k_xmn_05", by_id.get("k", {}).get("composition", {}).get("value") == 0.5, by_id.get("k", {})),
    ]
    sample = parse_material_text("LuFe0.6Mn0.4O3")
    checks.append(_check("sample_mismatches_panel_k", sample.x_variables.get("x_Mn") == 0.4 and by_id.get("k", {}).get("composition", {}).get("value") == 0.5, {"sample": sample.model_dump(mode="json"), "panel_k": by_id.get("k")}))
    return checks


def _classification_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {record["datafile_id"]: record for record in records}
    checks = []
    expectations = {
        "synthetic-lufe75mn25-angle-001": "multi_file_psd_angle_temperature_series",
        "synthetic-lumnfeo3-tdep-warm": "single_file_psd_temperature_map",
        "synthetic-lufemno3-3mev-line": "single_detector_inelastic_line_scan",
    }
    for source_id, family in expectations.items():
        actual = classify_source_family(by_id[source_id])
        checks.append(_check(f"classify_{source_id}", actual.source_family == family, actual.model_dump(mode="json")))
    return checks


def _triage_checks(triage: dict[str, Any]) -> list[dict[str, Any]]:
    state = triage["source_triage_state"]
    rejected_ids = {item["candidate_id"] for item in state.get("rejected_candidates", [])}
    groups = state.get("candidate_groups", [])
    return [
        _check("panel_sample_mismatch_detected", bool(triage.get("panel_sample_warnings")), {"warnings": triage.get("panel_sample_warnings")}),
        _check("parallel_hypotheses_created", len(state.get("active_hypotheses", [])) >= 3, {"hypotheses": state.get("active_hypotheses", [])}),
        _check("inelastic_false_positive_rejected", "synthetic-lufemno3-3mev-line" in rejected_ids, {"rejected": list(rejected_ids)}),
        _check("psd_groups_created", bool(groups), {"groups": groups}),
        _check("candidate_provenance_present", all(item.get("provenance") for item in state.get("candidates", [])), {"candidate_count": len(state.get("candidates", []))}),
    ]


def _reconstruction_checks(records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    groups = group_psd_sources(records)
    single = next(group for group in groups if group.group_type == "single_file_psd_temperature_map")
    result = reconstruct_channel_temperature(single, records, output_dir)
    return [
        _check("reconstruction_channel_temperature_success", result.status == "success", result.model_dump(mode="json")),
        _check("reconstruction_q_temperature_blocked", q_temperature_reconstruction_blocked(single.group_id).status == "unsupported", q_temperature_reconstruction_blocked(single.group_id).model_dump(mode="json")),
    ]


def _corpus_checks(corpus_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _check("corpus_paper_loaded", corpus_report.get("papers", 0) >= 1, corpus_report),
        _check("corpus_figure_loaded", corpus_report.get("figures", 0) >= 1, corpus_report),
        _check("corpus_panels_loaded", corpus_report.get("figure_panels", 0) >= 3, corpus_report),
        _check("source_support_loaded", corpus_report.get("source_support_records", 0) >= 1, corpus_report),
        _check("corpus_separate_from_tas_catalog", "separate" in corpus_report.get("separation_policy", ""), corpus_report),
    ]


def _provenance_summary(triage: dict[str, Any], corpus_report: dict[str, Any]) -> dict[str, Any]:
    state = triage["source_triage_state"]
    return {
        "candidate_provenance_count": sum(1 for item in state.get("candidates", []) if item.get("provenance")),
        "rejected_candidate_count": len(state.get("rejected_candidates", [])),
        "corpus_provenance_coverage": corpus_report.get("provenance_coverage"),
    }


def _check(name: str, condition: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "pass" if condition else "fail", "details": details}


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _callable_available(module_name: str, name: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[name])
        return callable(getattr(module, name))
    except Exception:
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _regression_markdown(report: dict[str, Any]) -> str:
    lines = ["# LuFe/Mn/O Known-Answer Regression", "", f"Status: `{report['status']}`", f"Checks: {report['check_count']} total, {report['failed_count']} failed", "", "## Checks"]
    for check in report["checks"]:
        lines.append(f"- `{check['status']}` {check['name']}")
    lines.extend(["", "## Safety", "- Q x temperature reconstruction remains blocked without reviewed PSD calibration.", "- Candidate links remain proposed until reconstruction/provenance or human review.", "- No live-control workflow is part of this regression."])
    return "\n".join(lines)


def _link_markdown(report: dict[str, Any]) -> str:
    lines = ["# Figure Panel Source Linkage", "", f"Status: `{report['status']}`", f"Figure: `{report['figure_id']}` Panel: `{report['panel_id']}`", "", "## Query Warnings"]
    lines.extend(f"- {warning}" for warning in report.get("query_warnings", []) or ["None"])
    lines.extend(["", "## Candidate Links"])
    for candidate in report["candidate_links"]:
        lines.append(f"- `{candidate['candidate_id']}` score={candidate['score']} status={candidate['status']} confidence={candidate.get('confidence_label')}")
    lines.extend(["", "## Rejected Candidates"])
    for rejected in report.get("rejected_candidates", []):
        lines.append(f"- `{rejected['candidate_id']}`: {rejected['rejection_reason']}")
    lines.extend(["", "## Source Support", f"- Status: `{report.get('source_support_summary', {}).get('status')}`"])
    lines.extend(["", "## Safe Conclusion", report["safe_conclusion"]])
    return "\n".join(lines)
