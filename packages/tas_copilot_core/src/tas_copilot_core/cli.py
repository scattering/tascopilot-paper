from __future__ import annotations

import argparse
import json
from pathlib import Path

from tas_copilot_core.corpus.paper_facts import (
    extract_paper_facts,
    paper_fact_review,
    paper_fact_table,
)
from tas_copilot_core.workflows.analyze_dataset import analyze_dataset
from tas_copilot_core.workflows.alignment_corpus_index import alignment_corpus_index
from tas_copilot_core.workflows.alignment_review_dashboard import build_alignment_review_dashboard
from tas_copilot_core.workflows.alignment_state import alignment_adaptive_nice_loop, alignment_fine_replay_plan, alignment_planning_brief, alignment_state_epochs, alignment_state_machine
from tas_copilot_core.workflows.bt7_corpus_candidate_pass import bt7_corpus_candidate_pass
from tas_copilot_core.workflows.add_scans import add_scans
from tas_copilot_core.workflows.apply_detailed_balance import apply_detailed_balance
from tas_copilot_core.workflows.calculate_angles import calculate_angles
from tas_copilot_core.workflows.calculate_ub import calculate_ub
from tas_copilot_core.workflows.check_resolution import check_resolution
from tas_copilot_core.workflows.check_geometry import check_geometry
from tas_copilot_core.workflows.check_spurions import check_spurions
from tas_copilot_core.workflows.classify_tas_format import classify_tas_format
from tas_copilot_core.workflows.compare_paper_to_data import compare_paper_to_data
from tas_copilot_core.workflows.composite import apply_corrections, combine_files, read_bt7_data
from tas_copilot_core.workflows.copilot_router import copilot_ask, copilot_intent_examples, run_copilot_route_quality_eval
from tas_copilot_core.workflows.extract_setup import extract_setup
from tas_copilot_core.workflows.experiment_dashboard import build_experiment_dashboard, dag_run_replay, draft_overnight_report, experiment_review, fit_order_parameter, inelastic_series_report, order_parameter_triage, overnight_scan_summary, survey_inelastic_spurion_evidence
from tas_copilot_core.workflows.experiment_planning_brief import experiment_planning_brief
from tas_copilot_core.workflows.experiment_workspace import experiment_workspace
from tas_copilot_core.workflows.eval_smoke import eval_smoke
from tas_copilot_core.workflows.find_experiments import find_experiments
from tas_copilot_core.workflows.find_figure_source_data import find_figure_source_data
from tas_copilot_core.workflows.figure_source_triage import figure_source_triage
from tas_copilot_core.workflows.live_experiment import live_experiment_page
from tas_copilot_core.workflows.lu_femn_o3_regression import link_figure_panel_to_source, run_lu_femn_o3_regression, source_triage_doctor
from tas_copilot_core.workflows.nice_bt7_replay import nice_bt7_replay
from tas_copilot_core.workflows.psd_mask_sensitivity import psd_mask_sensitivity
from tas_copilot_core.workflows.find_papers import find_papers
from tas_copilot_core.workflows.plan_spinwave import plan_spinwave
from tas_copilot_core.workflows.powder_order_parameter import powder_order_parameter
from tas_copilot_core.workflows.proposal_retrospective import ingest_proposal_evidence, proposal_experiment_retrospective, proposal_planning_brief, suggest_experiment_scans
from tas_copilot_core.workflows.reduce_scan_set import reduce_scan_set
from tas_copilot_core.workflows.retrieve_similar_scans import retrieve_similar_scans
from tas_copilot_core.workflows.run_spinwave_fixture import run_spinwave_fixture
from tas_copilot_core.workflows.tasai_adaptive_nice_sim import tasai_adaptive_nice_sim
from tas_copilot_core.workflows.read_tas_metadata import read_tas_metadata
from tas_copilot_core.workflows.summarize_psd_facility import summarize_psd_facility
from tas_copilot_core.workflows.scan_tas_directory import scan_tas_directory
from tas_copilot_core.workflows.read_bt7_file import read_bt7_file
from tas_copilot_core.workflows.group_bt7_scans import group_bt7_scans
from tas_copilot_core.workflows.inspect_bt7_directory import inspect_bt7_directory
from tas_copilot_core.workflows.catalog_bt7_directory import catalog_bt7_directory
from tas_copilot_core.workflows.catalog_tas_directory import catalog_tas_directory
from tas_copilot_core.workflows.catalog_report import catalog_report
from tas_copilot_core.workflows.catalog_tas_batch import catalog_tas_batch
from tas_copilot_core.workflows.scan_brief import scan_brief
from tas_copilot_core.workflows.experiment_brief import experiment_brief
from tas_copilot_core.workflows.feature_reality_check import assess_figure_feature
from tas_copilot_core.workflows.normalize_to_monitor import normalize_to_monitor
from tas_copilot_core.workflows.subtract_background import subtract_background
from tas_copilot_core.workflows.translate_ice_file import translate_ice_file
from tas_copilot_core.workflows.trajectory_clustering import cluster_order_parameter_trajectories, order_parameter_trajectory_report, plot_order_parameter_candidate_scans
from tas_copilot_core.catalogs.import_data_catalog import import_jsonl_catalog_dir
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report, markdown_quality_report
from tas_copilot_core.catalogs.completeness import eligible_scans_report, metadata_completeness_report
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.corpus.extractors import extract_document, extractor_bakeoff, extractor_quality_report
from tas_copilot_core.corpus.fixtures import compile_figure_query, corpus_fixture_report, load_corpus_fixtures, validate_source_support
from tas_copilot_core.corpus.pilot import agent_readiness, build_bt7_hydrated_index, build_paper_experiment_review_dashboard, corpus_pilot_report, corpus_search, extract_paper_constraints, figure_brief, ingest_extractor_run, ingest_paper_experiment_review_labels, ingest_pilot_corpus, link_paper_to_experiments, ontology_diagnostics, paper_brief, paper_candidate_shortlist, run_corpus_pilot_eval
from tas_copilot_core.corpus.evidence_graph_proposer import apply_evidence_graph_labels, run_evidence_graph_proposer
from tas_copilot_core.corpus.fact_eval_dashboard import build_fact_eval_dashboard
from tas_copilot_core.corpus.srmn2sb2_agentic_rag import build_srmn2sb2_agentic_rag_catalog, run_srmn2sb2_agentic_rag_loop, srmn2sb2_agentic_rag_report
from tas_copilot_core.llm_endpoints import apply_llm_endpoint
from tas_copilot_core.instrument_graph import instrument_graph_from_nice_report
from tas_copilot_core.ontology import (
    attach_external_mappings,
    export_alignment_json,
    external_alignment_diagnostics,
    format_alignment_report,
    load_external_mappings,
    load_external_vocabularies,
    load_ontology,
    mappings_for_concept,
    ontology_alignment_report,
    supernode_diagnostics,
    validate_ontology_alignment,
)
from tas_copilot_core.registry.tool_inventory import build_tool_inventory, doctor_report
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.review_labels import (
    backfill_review_labels,
    export_review_labels,
    ingest_review_labels,
    prepare_review_label_packet,
    review_label_summary,
    stale_review_label_check,
)
from tas_copilot_core.structure_factors import suggest_alignment_peaks
from tas_copilot_core.web_app import run_server
from tas_copilot_core.workflows.real_data_smoke import real_data_smoke
from tas_copilot_core.workflows.review_figure_source_candidate import review_figure_source_candidate
from tas_copilot_core.material_identity import compare_material_identity, material_identity_backend_status, material_identity_record
from tas_copilot_core.instrument_calibrations import instrument_calibrations_report, validate_instrument_calibrations_registry
from tas_copilot_core.psd_calibration import compare_psd_calibration_matches, discover_psd_dat_calibrations, match_psd_calibration, psd_calibration_report, validate_psd_calibration_registry
from tas_copilot_core.psd_reduction import derive_psd_calibration_workflow, psd_diffraction_pattern_workflow
from tas_copilot_core.source_triage import (
    check_psd_channel_artifact,
    classify_source_family,
    compare_psd_detector_cuts,
    discover_psd_calibration_candidates,
    group_psd_sources,
    hydrate_psd_records,
    plot_psd_detector_cuts,
    q_temperature_reconstruction_blocked,
    reconstruct_channel_temperature,
    select_detector_cut_records,
)


def _json(value) -> str:
    value = _to_jsonable(value)
    return json.dumps(value, indent=2, sort_keys=True)


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value

def _cmd_extract_paper_facts(args) -> None:
    report = extract_paper_facts(
        {
            "catalog": args.catalog,
            "output_dir": args.output_dir,
            "paper_id": args.paper_id,
            "provider": args.provider,
            "model": args.model,
            "model_output": args.model_output,
            "ollama_url": args.ollama_url,
            "vllm_url": args.vllm_url,
            "api_key": args.api_key,
            "max_chars_per_paper": args.max_chars_per_paper,
        }
    )
    print(_json(report))


def _cmd_paper_fact_review(args) -> None:
    report = paper_fact_review(
        {
            "catalog": args.catalog,
            "output_dir": args.output_dir,
            "paper_id": args.paper_id,
            "paper_ids_file": args.paper_ids_file,
        }
    )
    print(_json(report))


def _cmd_paper_fact_table(args) -> None:
    report = paper_fact_table(Path(args.catalog).expanduser())
    print(_json(report))


def _cmd_discover_fact_catalogs(args) -> None:
    catalogs = discover_fact_catalogs(Path(args.runs_root).expanduser())
    print(_json({"catalogs": catalogs, "count": len(catalogs)}))

def _parse_tool_versions(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            continue
        key, version = value.split("=", 1)
        if key.strip():
            parsed[key.strip()] = version.strip()
    return parsed


def _source_support_markdown(report: dict) -> str:
    return "\n".join(
        [
            "# Source Support Validation",
            "",
            f"Status: {report.get('status')}",
            f"Supported claims: {len(report.get('supported_claims', []))}",
            f"Unsupported claims: {len(report.get('unsupported_claims', []))}",
            f"Ambiguous claims: {len(report.get('ambiguous_claims', []))}",
            f"Missing source records: {', '.join(report.get('missing_source_records', [])) or 'none'}",
            "",
            "## Warnings",
            *[f"- {item}" for item in report.get("warnings", [])],
        ]
    ) + "\n"


def _front_door_menu() -> str:
    return "\n".join(
        [
            "TASCopilot: read-only/planning support for triple-axis neutron scattering.",
            "",
            "Start here:",
            '1. What can this do?        python -m tas_copilot_core.cli workflow copilot-ask --question "What can you do?" --output-dir local_runs/copilot_status',
            "2. See supported situations  python -m tas_copilot_core.cli workflow copilot-ask --list-intents",
            "3. Check installed tools     python -m tas_copilot_core.cli tools doctor",
            "4. Review an experiment      python -m tas_copilot_core.cli workflow copilot-ask --question \"Please do a full experiment review.\" --data-root [DATA_ROOT] --output-dir local_runs/experiment_review",
            "5. Review labels             python -m tas_copilot_core.cli review labels summary --store [REVIEW_OUTCOMES.sqlite]",
            "",
            "Safety boundary: no live instrument control, no approved operations, and no definitive claims without reviewed evidence.",
            "Docs: QUICKSTART.md and docs/html/index.html",
        ]
    )


def _copilot_intent_examples_text() -> str:
    lines = ["Bounded TASCopilot situations", ""]
    for index, item in enumerate(copilot_intent_examples(), start=1):
        lines.extend(
            [
                f"{index}. {item['situation']}",
                f"   Intent: {item['intent']}",
                f"   Ask: {item['example_question']}",
                f"   Runs: {item['runs']}",
                f"   Command: {item['command']}",
                "",
            ]
        )
    lines.append("All routes are read-only/planning routes. Missing inputs return prompts and examples instead of guessing.")
    return "\n".join(lines)


def _workflow_inventory() -> dict:
    examples = copilot_intent_examples()
    workflows = []
    for item in examples:
        command = str(item.get("command") or "")
        workflow = command.split(" workflow ", 1)[1].split(" ", 1)[0] if " workflow " in command else "copilot-ask"
        workflows.append(
            {
                "intent": item.get("intent"),
                "situation": item.get("situation"),
                "workflow": workflow,
                "example_question": item.get("example_question"),
                "command": command,
                "safety_boundary": "read_only_planning_no_live_control",
            }
        )
    workflows.extend(
        [
            {
                "intent": "review_label_summary",
                "situation": "I want to see what labels exist",
                "workflow": "review labels summary",
                "example_question": "What reviewed outcomes are available for calibration?",
                "command": "python -m tas_copilot_core.cli review labels summary --store [REVIEW_OUTCOMES.sqlite]",
                "safety_boundary": "review_state_only_no_scientific_acceptance",
            },
            {
                "intent": "review_label_backfill",
                "situation": "I want to migrate existing dashboard review templates",
                "workflow": "review labels backfill",
                "example_question": "Backfill local review templates into the unified store.",
                "command": "python -m tas_copilot_core.cli review labels backfill --root local_runs --store local_runs/review_outcomes.sqlite --output-dir local_runs/review_label_backfill",
                "safety_boundary": "local_label_store_only",
            },
            {
                "intent": "local_web_workbench",
                "situation": "I want a browser front door",
                "workflow": "serve",
                "example_question": "Open the local workbench.",
                "command": "python -m tas_copilot_core.cli serve --port 8765",
                "safety_boundary": "localhost_only_no_live_control",
            },
        ]
    )
    return {"workflow_count": len(workflows), "workflows": workflows}


def _workflow_inventory_text(inventory: dict) -> str:
    lines = ["TASCopilot workflow inventory", ""]
    for item in inventory.get("workflows", []):
        lines.append(f"- {item.get('situation')} (`{item.get('workflow')}`)")
        lines.append(f"  Intent: `{item.get('intent')}`")
        lines.append(f"  Try: {item.get('command')}")
    return "\n".join(lines)


def _ontology_decision_diagnostics(concepts: list) -> dict:
    decision = [concept for concept in concepts if getattr(concept, "family", None) == "decision_state"]
    sample_texts = []
    for concept in decision:
        sample_texts.extend(getattr(concept, "examples", []) or [])
        if getattr(concept, "label", None):
            sample_texts.append(concept.label)
    warnings = supernode_diagnostics(sample_texts, decision, max_fraction=0.6)
    external_warnings = external_alignment_diagnostics(decision)
    unsafe_retrieval = [
        {
            "concept_id": concept.id,
            "retrieval_policy": concept.retrieval_policy,
            "label": concept.label,
        }
        for concept in decision
        if concept.retrieval_policy not in {"no_expansion", "metadata_filter_only", "schema_alignment_only", "validation_only", "provenance_alignment_only"}
    ]
    return {
        "workflow_name": "ontology-diagnostics",
        "status": "ok" if not unsafe_retrieval else "warning",
        "decision_state_concept_count": len(decision),
        "supernode_warning_count": len(warnings),
        "supernode_warnings": warnings,
        "external_alignment_warning_count": len(external_warnings),
        "external_alignment_warnings": external_warnings,
        "unsafe_retrieval_policy_count": len(unsafe_retrieval),
        "unsafe_retrieval_policies": unsafe_retrieval,
        "safe_conclusion": "Decision-state ontology concepts are intended for labels, filters, and provenance; they should not become broad retrieval expansion nodes.",
    }


def _ontology_decision_diagnostics_text(report: dict) -> str:
    lines = [
        "TAS ontology diagnostics",
        "",
        f"Status: {report.get('status')}",
        f"Decision-state concepts: {report.get('decision_state_concept_count')}",
        f"Super-node warnings: {report.get('supernode_warning_count')}",
        f"Unsafe retrieval policies: {report.get('unsafe_retrieval_policy_count')}",
        "",
        str(report.get("safe_conclusion") or ""),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tas-copilot")
    sub = parser.add_subparsers(dest="command", required=False)
    examples = sub.add_parser("examples")
    examples.add_argument("--json", action="store_true", help="Emit machine-readable examples")
    inventory_cmd = sub.add_parser("inventory")
    inventory_cmd.add_argument("--json", action="store_true", help="Emit machine-readable workflow inventory")
    tools = sub.add_parser("tools")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    tools_doctor = tools_sub.add_parser("doctor")
    tools_doctor.add_argument("--runtime-profile", help="Check a selected runtime profile for example/placeholders before remote jobs")
    tools_list = tools_sub.add_parser("list")
    tools_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON inventory")
    mcp_serve = sub.add_parser("mcp-serve", help="Serve the bounded copilot intents as MCP tools over stdio (requires the mcp extra)")
    mcp_serve.add_argument("--local-runs-dir", default="local_runs")
    mcp_serve.add_argument("--surface", choices=["flat", "capabilities", "both"], default="flat", help="Tool vocabulary: flat = one tool per intent (default); capabilities = the ontology-routed capability facade; both = A/B side by side")
    chat_cmd = sub.add_parser("chat", help="Interactive chat shell over the bounded copilot router (optional local-model routing via Ollama)")
    chat_cmd.add_argument("--output-dir", default="local_runs/chat_sessions")
    chat_cmd.add_argument("--model", help="Ollama model for intent proposal (e.g. gemma3:27b); omit for deterministic routing")
    chat_cmd.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    chat_cmd.add_argument("--narrate", action="store_true", help="Append a model narration rendered from the answer envelope")
    chat_cmd.add_argument("--timeout", type=float, default=300.0)
    for field in ("corpus-catalog", "data-catalog", "data-root", "precedent-index", "paper-id", "figure-id", "fixture"):
        chat_cmd.add_argument(f"--{field}")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--local-runs-dir", default="local_runs")
    serve.add_argument("--docs-dir", default="docs/html")
    serve.add_argument("--app", choices=["workbench", "alignment"], default="workbench", help="workbench = full local UI; alignment = the standalone public-deployable alignment dashboard only")
    serve.add_argument("--root-path", default="", help="ASGI root path when served under a reverse-proxy subpath (alignment app only)")
    review = sub.add_parser("review")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_labels = review_sub.add_parser("labels")
    review_labels_sub = review_labels.add_subparsers(dest="review_labels_command", required=True)
    review_labels_summary = review_labels_sub.add_parser("summary")
    review_labels_summary.add_argument("--store", required=True)
    review_labels_summary.add_argument("--output-dir")
    review_labels_summary.add_argument("--sample-limit", type=int, default=10)
    review_labels_export = review_labels_sub.add_parser("export")
    review_labels_export.add_argument("--store", required=True)
    review_labels_export.add_argument("--output", required=True)
    review_labels_export.add_argument("--object-type")
    review_labels_export.add_argument("--object-id")
    review_labels_export.add_argument("--verdict")
    review_labels_export.add_argument("--source-workflow")
    review_labels_export.add_argument("--limit", type=int)
    review_labels_ingest = review_labels_sub.add_parser("ingest")
    review_labels_ingest.add_argument("--labels", required=True)
    review_labels_ingest.add_argument("--store", required=True)
    review_labels_ingest.add_argument("--output-dir")
    review_labels_backfill = review_labels_sub.add_parser("backfill")
    review_labels_backfill.add_argument("--root", default="local_runs")
    review_labels_backfill.add_argument("--store", required=True)
    review_labels_backfill.add_argument("--output-dir")
    review_labels_stale = review_labels_sub.add_parser("stale-check")
    review_labels_stale.add_argument("--store", required=True)
    review_labels_stale.add_argument("--output-dir")
    review_labels_stale.add_argument("--object-type")
    review_labels_stale.add_argument("--object-id")
    review_labels_stale.add_argument("--verdict")
    review_labels_stale.add_argument("--source-workflow")
    review_labels_stale.add_argument("--workflow-version")
    review_labels_stale.add_argument("--calibration-state-hash")
    review_labels_stale.add_argument("--dag-run-id")
    review_labels_stale.add_argument("--cache-key")
    review_labels_stale.add_argument("--tool-version", action="append", dest="tool_versions", help="tool=version; may be repeated")
    review_labels_stale.add_argument("--limit", type=int)
    review_labels_packet = review_labels_sub.add_parser("packet")
    review_labels_packet.add_argument("--store", required=True)
    review_labels_packet.add_argument("--output-dir", required=True)
    review_labels_packet.add_argument("--object-type")
    review_labels_packet.add_argument("--source-workflow")
    review_labels_packet.add_argument("--verdict", default="unreviewed")
    review_labels_packet.add_argument("--limit", type=int, default=50)
    ontology = sub.add_parser("ontology")
    ontology_sub = ontology.add_subparsers(dest="ontology_command", required=True)
    ontology_vocabularies = ontology_sub.add_parser("vocabularies")
    ontology_vocabularies_sub = ontology_vocabularies.add_subparsers(dest="ontology_vocabularies_command", required=True)
    ontology_vocabularies_list = ontology_vocabularies_sub.add_parser("list")
    ontology_vocabularies_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ontology_mappings = ontology_sub.add_parser("mappings")
    ontology_mappings_sub = ontology_mappings.add_subparsers(dest="ontology_mappings_command", required=True)
    ontology_mappings_sub.add_parser("validate")
    ontology_mappings_sub.add_parser("report")
    ontology_concept = ontology_sub.add_parser("concept")
    ontology_concept_sub = ontology_concept.add_subparsers(dest="ontology_concept_command", required=True)
    ontology_concept_show = ontology_concept_sub.add_parser("show")
    ontology_concept_show.add_argument("concept_id")
    ontology_export = ontology_sub.add_parser("export")
    ontology_diagnostics_cmd = ontology_sub.add_parser("diagnostics")
    ontology_diagnostics_cmd.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics")
    ontology_export.add_argument("--format", choices=["json"], default="json")
    corpus = sub.add_parser("corpus")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_load = corpus_sub.add_parser("load-fixtures")
    corpus_load.add_argument("--fixtures", required=True)
    corpus_load.add_argument("--catalog", required=True)
    corpus_report = corpus_sub.add_parser("report")
    corpus_report.add_argument("--catalog", required=True)
    corpus_report.add_argument("--output-dir")
    corpus_shortlist = corpus_sub.add_parser("paper-shortlist")
    corpus_shortlist.add_argument("--catalog", required=True)
    corpus_shortlist.add_argument("--output-dir", required=True)
    corpus_shortlist.add_argument("--quality-report")
    corpus_shortlist.add_argument("--top-k", type=int, default=10)
    corpus_constraints = corpus_sub.add_parser("extract-paper-constraints")
    corpus_constraints.add_argument("--catalog", required=True)
    corpus_constraints.add_argument("--output-dir", required=True)
    corpus_constraints.add_argument("--paper-id")
    corpus_facts = corpus_sub.add_parser("extract-paper-facts")
    corpus_facts.add_argument("--catalog", required=True)
    corpus_facts.add_argument("--output-dir", required=True)
    corpus_facts.add_argument("--paper-id")
    corpus_facts.add_argument("--provider", choices=["none", "json-file", "ollama", "vllm"], default="none")
    corpus_facts.add_argument("--model-output", help="Precomputed JSON model output for provider=json-file")
    corpus_facts.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    corpus_facts.add_argument("--vllm-url", help="Base URL for vLLM/OpenAI-compatible API")
    corpus_facts.add_argument("--api-key", help="API key for vLLM/OpenAI-compatible API (if required)")
    corpus_facts.add_argument("--model")
    corpus_facts.add_argument("--llm-endpoint", help="Named LLM endpoint profile from configs/llm_endpoints (e.g. nist_vllm, vista_ollama); explicit flags override profile values")
    corpus_facts.add_argument("--max-chars-per-paper", type=int, default=45000)
    corpus_fact_review = corpus_sub.add_parser("paper-fact-review")
    corpus_fact_review.add_argument("--catalog", required=True)
    corpus_fact_review.add_argument("--output-dir", required=True)
    corpus_fact_review.add_argument("--paper-id")
    corpus_fact_review.add_argument("--paper-ids-file")
    corpus_srmn_agentic = corpus_sub.add_parser("srmn2sb2-agentic-rag-catalog")
    corpus_srmn_agentic.add_argument("--catalog", required=True)
    corpus_srmn_agentic.add_argument("--output-dir", required=True)
    corpus_srmn_agentic.add_argument("--extractor-run-dir", help="Vista Docling extractor run directory containing outputs/docling/summary.json")
    corpus_srmn_agentic.add_argument("--source-markdown", action="append", dest="source_markdowns", help="Docling Markdown/text fixture; may be repeated")
    corpus_srmn_agentic.add_argument("--max-chars-per-chunk", type=int, default=3500)
    corpus_srmn_agentic_report = corpus_sub.add_parser("srmn2sb2-agentic-rag-report")
    corpus_srmn_agentic_report.add_argument("--catalog", required=True)
    corpus_srmn_agentic_report.add_argument("--output-dir", required=True)
    corpus_srmn_agentic_report.add_argument("--sr-case-note")
    corpus_srmn_agentic_report.add_argument("--sr-fit-report")
    corpus_srmn_agentic_loop = corpus_sub.add_parser("srmn2sb2-agentic-rag-loop")
    corpus_srmn_agentic_loop.add_argument("--catalog", required=True)
    corpus_srmn_agentic_loop.add_argument("--output-dir", required=True)
    corpus_srmn_agentic_loop.add_argument("--provider", choices=["none", "ollama", "vllm"], default="none")
    corpus_srmn_agentic_loop.add_argument("--model")
    corpus_srmn_agentic_loop.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    corpus_srmn_agentic_loop.add_argument("--vllm-url", help="Base URL for vLLM/OpenAI-compatible API")
    corpus_srmn_agentic_loop.add_argument("--api-key", help="API key for vLLM/OpenAI-compatible API (if required)")
    corpus_srmn_agentic_loop.add_argument("--llm-endpoint", help="Named LLM endpoint profile from configs/llm_endpoints (e.g. nist_vllm, vista_ollama); explicit flags override profile values")
    corpus_srmn_agentic_loop.add_argument("--top-k", type=int, default=6)
    corpus_srmn_agentic_loop.add_argument("--max-targets", type=int, default=4)
    corpus_srmn_agentic_loop.add_argument("--max-followups", type=int, default=1)
    corpus_srmn_agentic_loop.add_argument("--max-chars-per-target", type=int, default=18000)
    corpus_edge_propose = corpus_sub.add_parser("evidence-graph-propose")
    corpus_edge_propose.add_argument("--catalog", required=True)
    corpus_edge_propose.add_argument("--output-dir", required=True)
    corpus_edge_propose.add_argument("--provider", choices=["none", "ollama", "vllm"], default="none")
    corpus_edge_propose.add_argument("--model")
    corpus_edge_propose.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    corpus_edge_propose.add_argument("--vllm-url", help="Base URL for vLLM/OpenAI-compatible API")
    corpus_edge_propose.add_argument("--api-key", help="API key for vLLM/OpenAI-compatible API (if required)")
    corpus_edge_propose.add_argument("--case-target", help="Optional case-target id for ANALOG_MODEL_PRIOR_FOR edges, e.g. 'srmn2sb2-experiment-24858'")
    corpus_edge_propose.add_argument("--case-target-label", help="Human-readable case-target label")
    corpus_edge_propose.add_argument("--max-pairs", type=int, default=20)
    corpus_edge_propose.add_argument("--max-chunk-chars", type=int, default=4000)
    corpus_edge_propose.add_argument("--llm-endpoint", help="Named LLM endpoint profile from configs/llm_endpoints (e.g. nist_vllm, vista_ollama); explicit flags override profile values")
    corpus_edge_labels = corpus_sub.add_parser("evidence-graph-apply-labels")
    corpus_edge_labels.add_argument("--catalog", required=True)
    corpus_edge_labels.add_argument("--labels", required=True, help="Reviewed evidence_graph_review_template.jsonl with verdict/type_specific_status filled in")
    corpus_edge_labels.add_argument("--output-dir")
    corpus_edge_labels.add_argument("--reviewer", help="Fallback reviewer id when label rows omit reviewer_id")
    corpus_integrity = corpus_sub.add_parser("catalog-integrity", help="Flag papers whose chunk text is shared with other titles or does not contain their own title; --apply stamps facts and papers in place (ids kept)")
    corpus_integrity.add_argument("--catalog", required=True)
    corpus_integrity.add_argument("--apply", action="store_true")
    corpus_pdf_integrity = corpus_sub.add_parser("pdf-integrity", help="From a download manifest with sha256 values, list PDFs saved under several titles (bad downloads) and the DOIs to fetch again")
    corpus_pdf_integrity.add_argument("--pdf-root", required=True)
    corpus_pdf_integrity.add_argument("--manifest", help="Download manifest CSV (title, doi, downloaded_pdf_path) used to name titles and DOIs")
    corpus_pdf_integrity.add_argument("--first-page", action="store_true", help="Also check each PDF's first page against its title (needs pdftotext)")
    corpus_pdf_integrity.add_argument("--output", help="Write the report JSON here")
    corpus_revalidate = corpus_sub.add_parser("revalidate-paper-facts", help="Re-run evidence location, status, retyping, malformed-value and duplicate checks over an existing catalog with the current code; fact ids are kept")
    corpus_revalidate.add_argument("--catalog", required=True)
    corpus_revalidate.add_argument("--fact-file", default="paper_facts.jsonl")
    corpus_revalidate.add_argument("--in-place", action="store_true", help="Overwrite the fact file (a .pre_revalidate backup is written once)")
    corpus_revalidate.add_argument("--output-name", default="paper_facts.revalidated.jsonl", help="File written beside the fact file when not --in-place")
    corpus_score_labels = corpus_sub.add_parser("score-fact-labels", help="Tabulate a paper-fact catalog against human review labels (ReviewLabel JSONL); fact ids survive revalidation, so the same labels score a catalog before and after a change")
    corpus_score_labels.add_argument("--catalog", required=True)
    corpus_score_labels.add_argument("--labels", required=True, action="append", help="ReviewLabel JSONL (repeatable; later files win)")
    corpus_score_labels.add_argument("--fact-file", default="paper_facts.jsonl")
    corpus_score_labels.add_argument("--output-dir")
    corpus_score_labels.add_argument("--include-quarantined", action="store_true", help="Also score facts from papers quarantined by catalog-integrity")
    corpus_score_labels.add_argument("--judgments", help="paper_fact_judgments.jsonl from judge-paper-facts: adds judge-versus-human tables")
    corpus_judge = corpus_sub.add_parser("judge-paper-facts", help="Second-pass check of each fact against its located source passages with a local model (fixed-option answers, verdict derived by rule); never accepts a fact")
    corpus_judge.add_argument("--catalog", required=True)
    corpus_judge.add_argument("--fact-file", default="paper_facts.jsonl")
    corpus_judge.add_argument("--output-dir")
    corpus_judge.add_argument("--provider", choices=["ollama", "openai_compatible"], default="ollama")
    corpus_judge.add_argument("--model", required=True)
    corpus_judge.add_argument("--base-url", default="http://127.0.0.1:11434")
    corpus_judge.add_argument("--api-key")
    corpus_judge.add_argument("--fact-ids-from", help="Only judge these fact ids: a ReviewLabel JSONL or one id per line")
    corpus_judge.add_argument("--limit", type=int, default=0)
    corpus_judge.add_argument("--apply", action="store_true", help="Also write <fact-file stem>.judged.jsonl with the verdicts recorded as metadata and warnings")
    corpus_fact_eval = corpus_sub.add_parser("fact-eval-dashboard")
    corpus_fact_eval.add_argument("--catalog", required=True)
    corpus_fact_eval.add_argument("--output-dir", required=True)
    corpus_fact_eval.add_argument("--fact-file", dest="fact_file", default="paper_facts.jsonl", help="Fact file inside the catalog (e.g. paper_facts.judged.jsonl so second-pass judge flags show and sort first)")
    corpus_fact_eval.add_argument("--group-by-paper", dest="group_by_paper", action="store_true", help="One paper at a time: paper navigator, per-paper progress, the paper's chunk text at the top of each group")
    corpus_fact_eval.add_argument("--pdf-dir", dest="pdf_dir", help="Folder (or URL base) holding the corpus PDFs; with --group-by-paper each paper header links to <pdf-dir>/<relative path or pdf_name>. The folder is not read.")
    corpus_fact_eval.add_argument("--pdf-paths", dest="pdf_paths", help="JSONL of {paper_id, pdf_relpath} relative to --pdf-dir (default: <catalog>/paper_pdf_paths.jsonl if present)")
    corpus_fact_eval.add_argument("--only-second-looks", dest="only_second_looks", action="store_true", help="Write second_looks.html with only the cards whose earlier citation-defect label needs re-judging")
    corpus_fact_eval.add_argument("--only-ids", dest="only_ids", help="File of fact ids: write a page with only those cards")
    corpus_fact_eval.add_argument("--no-edges", dest="no_edges", action="store_true", help="Leave relation-edge cards off the page (facts only)")
    corpus_fact_eval.add_argument("--second-look-from", action="append", dest="second_look_from", help="Resume file(s) whose citation-defect labels predate the citation repair and are queued for a second look; labels in other resume files are final (default: all resume files)")
    corpus_fact_eval.add_argument("--page-name", dest="page_name", help="Output file name without .html (default fact_eval_dashboard, or second_looks with --only-second-looks); also names the server's draft file")
    corpus_fact_eval.add_argument("--prior-labels", action="append", dest="prior_labels", help="ReviewLabel JSONL shown on each card as a prior flag (repeatable, e.g. an initial AI pass)")
    corpus_fact_eval.add_argument("--resume-labels", action="append", dest="resume_labels", help="The reviewer's own earlier export: pre-fills the panel; citation-defect labels on facts whose quotes were since repaired are queued for a second look instead")
    corpus_fact_eval.add_argument("--storage-key", help="Browser storage key for drafts (default: the catalog path); reuse an earlier panel's key to pick up its drafts")
    corpus_review_dashboard = corpus_sub.add_parser("paper-link-review-dashboard")
    corpus_review_dashboard.add_argument("--review-template", required=True)
    corpus_review_dashboard.add_argument("--review-assist")
    corpus_review_dashboard.add_argument("--output-dir", required=True)
    corpus_review_dashboard.add_argument("--html-filename", default="paper_experiment_review_dashboard.html")
    corpus_review_dashboard.add_argument("--max-packet-scans", type=int, default=8, help="Maximum locally available candidate scans to plot in each generated review packet")
    corpus_review_dashboard.add_argument("--paper-base-url", default="/papers", help="Base URL for local paper opener links in generated review packets")
    corpus_review_dashboard.add_argument("--data-catalog", help="Local SQLite data catalog for resolving bare scan filenames to full paths when generating scan plots")
    corpus_review_dashboard.add_argument(
        "--review-scan-records",
        help="Optional JSONL sidecar with precomputed scan records for review packets; defaults to review_packet_scan_records.jsonl beside the template when present",
    )
    corpus_review_ingest = corpus_sub.add_parser("ingest-paper-link-reviews")
    corpus_review_ingest.add_argument("--labels", required=True)
    corpus_review_ingest.add_argument("--output-dir", required=True)
    corpus_review_ingest.add_argument("--min-each", type=int, default=10)
    corpus_ingest = corpus_sub.add_parser("ingest-pilot")
    corpus_ingest.add_argument("--manifest", required=True)
    corpus_ingest.add_argument("--catalog", required=True)
    corpus_ingest.add_argument("--output-dir", required=True)
    corpus_ingest_extractor = corpus_sub.add_parser("ingest-extractor-run")
    corpus_ingest_extractor.add_argument("--extractor-run-dir", required=True)
    corpus_ingest_extractor.add_argument("--catalog", required=True)
    corpus_ingest_extractor.add_argument("--output-dir", required=True)
    corpus_ingest_extractor.add_argument("--preferred-backend", choices=["docling", "grobid"], default="docling")
    corpus_ingest_extractor.add_argument("--max-chars-per-document", type=int, default=250000)
    corpus_ingest_extractor.add_argument("--append", action="store_true", help="merge into an existing catalog, replacing only the papers present in this extractor run (incremental update)")
    corpus_search_cmd = corpus_sub.add_parser("search")
    corpus_search_cmd.add_argument("--catalog", required=True)
    corpus_search_cmd.add_argument("--query", required=True)
    corpus_search_cmd.add_argument("--top-k", type=int, default=10)
    corpus_search_cmd.add_argument("--paper-id")
    corpus_search_cmd.add_argument("--document-type")
    corpus_search_cmd.add_argument("--chunk-type")
    corpus_search_cmd.add_argument("--figure-id")
    corpus_search_cmd.add_argument("--panel-id")
    corpus_search_cmd.add_argument("--tas-concept")
    corpus_search_cmd.add_argument("--year", type=int)
    corpus_search_cmd.add_argument("--instrument")
    corpus_search_cmd.add_argument("--material-family")
    corpus_search_cmd.add_argument("--requires-source-support", action="store_true")
    corpus_search_cmd.add_argument("--exclude-references", action="store_true")
    corpus_search_cmd.add_argument("--json", action="store_true")
    corpus_search_cmd.add_argument(
        "--use-dense",
        action="store_true",
        help="Enable dense vector search using a local sentence‑transformer model (hybrid with keyword scores)."
    )

    # corpus_fact_table_cmd = corpus_sub.add_parser(
    #     "paper-fact-table",
    #     help="Show per-paper extracted fact counts for a corpus catalog.",
    # )
    # corpus_fact_table_cmd.add_argument("--catalog", required=True)
    # corpus_fact_table_cmd.set_defaults(func=_cmd_paper_fact_table)

    # corpus_discover_fact_catalogs_cmd = corpus_sub.add_parser(
    #     "discover-fact-catalogs",
    #     help="Discover corpus catalogs that can be used for paper fact review.",
    # )
    # corpus_discover_fact_catalogs_cmd.add_argument(
    #     "--runs-root",
    #     default="local_runs",
    # )
    # corpus_discover_fact_catalogs_cmd.set_defaults(func=_cmd_discover_fact_catalogs)
    corpus_ontology = corpus_sub.add_parser("ontology-diagnostics")
    corpus_ontology.add_argument("--catalog", required=True)
    corpus_ontology.add_argument("--output-dir")
    corpus_agent = corpus_sub.add_parser("agent-readiness")
    corpus_agent.add_argument("--corpus-catalog", required=True)
    corpus_agent.add_argument("--data-catalog")
    corpus_agent.add_argument("--output-dir", required=True)
    corpus_compile = corpus_sub.add_parser("compile-figure-query")
    corpus_compile.add_argument("--corpus-catalog", required=True)
    corpus_compile.add_argument("--paper-id")
    corpus_compile.add_argument("--figure-id", required=True)
    corpus_compile.add_argument("--panel-id", required=True)
    corpus_compile.add_argument("--user-sample")
    corpus_compile.add_argument("--output")
    corpus_validate = corpus_sub.add_parser("validate-source-support")
    corpus_validate.add_argument("--catalog", required=True)
    corpus_validate.add_argument("--output-dir")
    corpus_validate.add_argument("--json", action="store_true")
    corpus_extract = corpus_sub.add_parser("extract-document")
    corpus_extract.add_argument("--path", required=True)
    corpus_extract.add_argument("--backend", default="pdftotext_baseline", choices=["auto", "pdftotext_baseline", "grobid", "docling"])
    corpus_extract.add_argument("--max-pages", type=int, default=8)
    corpus_extract.add_argument("--max-chars", type=int, default=20000)
    corpus_extract.add_argument("--grobid-url")
    corpus_extract.add_argument("--output-dir")
    corpus_extract.add_argument("--write-artifacts", action="store_true")
    corpus_extract.add_argument("--json", action="store_true")
    corpus_bakeoff = corpus_sub.add_parser("extractor-bakeoff")
    corpus_bakeoff.add_argument("--manifest", required=True)
    corpus_bakeoff.add_argument("--output-dir", required=True)
    corpus_bakeoff.add_argument("--limit", type=int, default=10)
    corpus_bakeoff.add_argument("--backend", action="append", dest="backends")
    corpus_bakeoff.add_argument("--max-pages", type=int, default=8)
    corpus_bakeoff.add_argument("--max-chars", type=int, default=20000)
    corpus_bakeoff.add_argument("--grobid-url")
    corpus_quality = corpus_sub.add_parser("extractor-quality")
    corpus_quality.add_argument("--extractor-run-dir", required=True)
    corpus_quality.add_argument("--output-dir", required=True)
    eval_cmd = sub.add_parser("eval")
    eval_sub = eval_cmd.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--suite", required=True)
    eval_run.add_argument("--output-dir", required=True)
    eval_run.add_argument("--json", default="{}", help="Optional JSON arguments for eval runner")
    material = sub.add_parser("material")
    material_sub = material.add_subparsers(dest="material_command", required=True)
    material_parse = material_sub.add_parser("parse")
    material_parse.add_argument("--text", required=True)
    material_parse.add_argument("--family", default="lu_femn_o3")
    material_parse.add_argument("--json", action="store_true")
    material_compare = material_sub.add_parser("compare")
    material_compare.add_argument("--left", required=True)
    material_compare.add_argument("--right", required=True)
    material_compare.add_argument("--family", default="lu_femn_o3")
    material_compare.add_argument("--json", action="store_true")
    material_sub.add_parser("doctor")
    instrument_calibrations = sub.add_parser("instrument-calibrations")
    instrument_calibrations_sub = instrument_calibrations.add_subparsers(dest="instrument_calibrations_command", required=True)
    instrument_calibrations_validate = instrument_calibrations_sub.add_parser("validate")
    instrument_calibrations_validate.add_argument("--registry")
    instrument_calibrations_list = instrument_calibrations_sub.add_parser("list")
    instrument_calibrations_list.add_argument("--registry")
    instrument_calibrations_list.add_argument("--json", action="store_true")
    psd_calibration = sub.add_parser("psd-calibration")
    psd_calibration_sub = psd_calibration.add_subparsers(dest="psd_calibration_command", required=True)
    psd_calibration_validate = psd_calibration_sub.add_parser("validate")
    psd_calibration_validate.add_argument("--registry")
    psd_calibration_report_cmd = psd_calibration_sub.add_parser("report")
    psd_calibration_report_cmd.add_argument("--registry")
    psd_calibration_report_cmd.add_argument("--json", action="store_true")
    psd_calibration_discover_dat = psd_calibration_sub.add_parser("discover-dat")
    psd_calibration_discover_dat.add_argument("--calibration-dir", required=True)
    psd_calibration_discover_dat.add_argument("--path-only", action="store_true", help="Use filename/path metadata only without hydrating calibration files")
    psd_calibration_match = psd_calibration_sub.add_parser("match")
    psd_calibration_match.add_argument("--calibration-dir", required=True)
    psd_calibration_match.add_argument("--data-file", required=True)
    psd_calibration_match.add_argument("--path-only", action="store_true", help="Use filename/path metadata only without hydrating calibration files")
    psd_calibration_compare = psd_calibration_sub.add_parser("compare")
    psd_calibration_compare.add_argument("--calibration-dir", required=True)
    psd_calibration_compare.add_argument("--data-file", required=True)
    psd_calibration_compare.add_argument("--top-k", type=int, default=5)
    psd_calibration_compare.add_argument("--output-dir")
    psd_calibration_compare.add_argument("--path-only", action="store_true", help="Use filename/path metadata only without hydrating calibration files")
    psd_calibration_derive = psd_calibration_sub.add_parser("derive-findpeak", help="Per-channel efficiency and A4 offset from a FindPeak A4 scan through one Bragg peak (ICE fpx file or NICE A4 scan with PSD columns)")
    psd_calibration_derive.add_argument("--findpeak-file", required=True)
    psd_calibration_derive.add_argument("--output-dir", help="Writes <stem>.json and the legacy <stem>.dat (efficiency, relative A4)")
    psd_calibration_derive.add_argument("--calibration-id")
    psd_calibration_derive.add_argument("--stem", help="Output file stem (default: the calibration id)")
    psd_calibration_derive.add_argument("--nominal-a4", type=float, help="Override the scan's declared centre; offsets are centre minus this")
    psd_calibration_derive.add_argument("--analyzer-detector-translation", type=float, help="AnaTrans (cm) of the FindPeak scan if known; recorded only")
    psd_calibration_derive.add_argument("--note", action="append", dest="notes")
    workflow = sub.add_parser("workflow")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_inspect = workflow_sub.add_parser("inspect-bt7-directory")
    workflow_inspect.add_argument("--path", required=True)
    workflow_inspect.add_argument("--json", default="{}", help="Optional JSON arguments for directory inspection")
    workflow_spinwave = workflow_sub.add_parser("run-spinwave-fixture")
    workflow_spinwave.add_argument("--json", default="{}", help="JSON SpinWaveCalculationInput arguments")
    workflow_geometry = workflow_sub.add_parser("check-geometry")
    workflow_geometry.add_argument("--scan-id", help="Scan id to resolve from the TAS data-file catalog")
    workflow_geometry.add_argument("--db", default=str(CoreConfig().data_catalog_db_path), help="SQLite TAS data-file catalog for --scan-id")
    workflow_geometry.add_argument("--json", default="{}", help="JSON Q/E/instrument/lattice geometry arguments")
    workflow_compare = workflow_sub.add_parser("compare-paper-to-data")
    workflow_compare.add_argument("--paper-id", required=True)
    workflow_compare.add_argument("--json", default="{}", help="Optional JSON arguments including paper, experiments, db_path, and threshold")
    workflow_reduce = workflow_sub.add_parser("reduce-scan-set")
    workflow_reduce.add_argument("--json", default="{}", help="JSON arguments including steps, input lineage, and derived output path")
    workflow_similar = workflow_sub.add_parser("retrieve-similar-scans")
    workflow_similar.add_argument("scan_id")
    workflow_similar.add_argument("--limit", type=int, default=5)
    workflow_similar.add_argument("--catalog", help="SQLite TAS data-file catalog")
    workflow_smoke = workflow_sub.add_parser("real-data-smoke")
    workflow_smoke.add_argument("--path", required=True)
    workflow_smoke.add_argument("--catalog", required=True)
    workflow_smoke.add_argument("--output-dir", required=True)
    workflow_smoke.add_argument("--json", default="{}", help="Optional JSON arguments including max_files and max_records")
    workflow_catalog_report = workflow_sub.add_parser("catalog-report")
    workflow_catalog_report.add_argument("--catalog", required=True)
    workflow_catalog_report.add_argument("--query", default="")
    workflow_catalog_report.add_argument("--limit", type=int, default=10)
    workflow_catalog_report.add_argument("--output-dir")
    workflow_catalog_report.add_argument("--json", default="{}", help="Optional JSON arguments for catalog reporting")
    workflow_scan_brief = workflow_sub.add_parser("scan-brief")
    workflow_scan_brief.add_argument("--scan-id", required=True)
    workflow_scan_brief.add_argument("--catalog", required=True)
    workflow_scan_brief.add_argument("--limit", type=int, default=5)
    workflow_scan_brief.add_argument("--output-dir")
    workflow_scan_brief.add_argument("--json", default="{}", help="Optional JSON arguments for scan briefing")
    workflow_experiment_brief = workflow_sub.add_parser("experiment-brief")
    workflow_experiment_brief.add_argument("--experiment-id", required=True)
    workflow_experiment_brief.add_argument("--catalog", required=True)
    workflow_experiment_brief.add_argument("--limit", type=int, default=10)
    workflow_experiment_brief.add_argument("--output-dir")
    workflow_experiment_brief.add_argument("--json", default="{}", help="Optional JSON arguments for experiment briefing")
    workflow_figure_source = workflow_sub.add_parser("find-figure-source-data")
    workflow_figure_source.add_argument("--catalog")
    workflow_figure_source.add_argument("--data-root")
    workflow_figure_source.add_argument("--output-dir", required=True)
    workflow_figure_source.add_argument("--query-file")
    workflow_figure_source.add_argument("--figure-image")
    workflow_figure_source.add_argument("--instrument")
    workflow_figure_source.add_argument("--mode")
    workflow_figure_source.add_argument("--sample-type")
    workflow_figure_source.add_argument("--material-family")
    workflow_figure_source.add_argument("--sample")
    workflow_figure_source.add_argument("--target-panel")
    workflow_figure_source.add_argument("--target-panel-xmn", type=float)
    workflow_figure_source.add_argument("--alternate-panel")
    workflow_figure_source.add_argument("--alternate-panel-xmn", type=float)
    workflow_figure_source.add_argument("--date", action="append")
    workflow_figure_source.add_argument("--experimenter", action="append")
    workflow_figure_source.add_argument("--q-min", type=float)
    workflow_figure_source.add_argument("--q-max", type=float)
    workflow_figure_source.add_argument("--temperature-min", type=float)
    workflow_figure_source.add_argument("--temperature-max", type=float)
    workflow_figure_source.add_argument("--annotated-q", type=float, action="append")
    workflow_figure_source.add_argument("--top-k", type=int, default=20)
    workflow_figure_source.add_argument("--attempt-reconstruction", action="store_true")
    workflow_figure_source.add_argument("--max-reconstruction-candidates", type=int, default=5)
    workflow_figure_source.add_argument("--json", default="{}", help="Optional JSON arguments for figure-source search")
    workflow_feature = workflow_sub.add_parser("assess-figure-feature")
    workflow_feature.add_argument("--data-root", required=True)
    workflow_feature.add_argument("--output-dir", required=True)
    workflow_feature.add_argument("--pattern", action="append", dest="patterns")
    workflow_feature.add_argument("--include-term", action="append", dest="include_terms")
    workflow_feature.add_argument("--q-feature", action="append", type=float, dest="q_features")
    workflow_feature.add_argument("--temperature-feature", action="append", type=float, dest="temperature_features")
    workflow_feature.add_argument("--q-tolerance", type=float)
    workflow_feature.add_argument("--sideband-width", type=float)
    workflow_feature.add_argument("--sideband-gap", type=float)
    workflow_feature.add_argument("--temperature-tolerance", type=float)
    workflow_feature.add_argument("--json", default="{}", help="Optional JSON arguments for feature reality triage")
    workflow_triage = workflow_sub.add_parser("figure-source-triage")
    workflow_triage.add_argument("--catalog")
    workflow_triage.add_argument("--query-file", required=True)
    workflow_triage.add_argument("--output-dir", required=True)
    workflow_triage.add_argument("--json", default="{}", help="Optional JSON arguments for source triage")
    workflow_sub.add_parser("source-triage-doctor")
    workflow_lu_regression = workflow_sub.add_parser("lu-femn-o3-regression")
    workflow_lu_regression.add_argument("--suite", default="eval/figure_source_lu_femn_o3_known_answer.yaml")
    workflow_lu_regression.add_argument("--output-dir", required=True)
    workflow_lu_regression.add_argument("--json", default="{}", help="Optional JSON arguments for LuFe/Mn/O regression")
    workflow_link_panel = workflow_sub.add_parser("link-figure-panel-to-source")
    workflow_link_panel.add_argument("--corpus-catalog", required=True)
    workflow_link_panel.add_argument("--data-catalog", required=True)
    workflow_link_panel.add_argument("--data-root")
    workflow_link_panel.add_argument("--paper-id")
    workflow_link_panel.add_argument("--figure-id", required=True)
    workflow_link_panel.add_argument("--panel-id", required=True)
    workflow_link_panel.add_argument("--user-sample")
    workflow_link_panel.add_argument("--output-dir", required=True)
    workflow_link_panel.add_argument("--query-file", default="examples/figure_queries/lu_femn_o3_elastic_psd_triage.yaml")
    workflow_link_panel.add_argument("--top-k", type=int, default=5)
    workflow_link_panel.add_argument("--json", default="{}", help="Optional JSON arguments for panel-source linkage")
    workflow_paper_brief = workflow_sub.add_parser("paper-brief")
    workflow_paper_brief.add_argument("--corpus-catalog", required=True)
    workflow_paper_brief.add_argument("--paper-id", required=True)
    workflow_paper_brief.add_argument("--output-dir", required=True)
    workflow_figure_brief = workflow_sub.add_parser("figure-brief")
    workflow_figure_brief.add_argument("--corpus-catalog", required=True)
    workflow_figure_brief.add_argument("--figure-id", required=True)
    workflow_figure_brief.add_argument("--output-dir", required=True)
    workflow_link_paper = workflow_sub.add_parser("link-paper-to-experiments")
    workflow_link_paper.add_argument("--corpus-catalog", required=True)
    workflow_link_paper.add_argument("--data-catalog")
    workflow_link_paper.add_argument("--paper-id", required=True)
    workflow_link_paper.add_argument("--output-dir", required=True)
    workflow_link_paper.add_argument("--top-k", type=int, default=20)
    workflow_link_paper.add_argument("--max-data-candidates", type=int, default=5000)
    workflow_link_paper.add_argument("--hydrate-candidate-clues", action="store_true", help="Read a bounded sample of local hydrated BT-7 files through the registered reader for weak linkage clues")
    workflow_link_paper.add_argument("--hydrate-candidate-limit", type=int, default=25)
    workflow_link_paper.add_argument("--max-hydrated-files-per-candidate", type=int, default=12)
    workflow_link_paper.add_argument("--hydrated-index", help="Optional local JSONL sidecar from build-bt7-hydrated-index")
    workflow_link_paper.add_argument("--review-label-store")
    workflow_proposal_ingest = workflow_sub.add_parser("proposal-evidence-ingest")
    workflow_proposal_ingest.add_argument("--proposal-document", action="append", dest="proposal_documents")
    workflow_proposal_ingest.add_argument("--actual-document", action="append", dest="actual_documents")
    workflow_proposal_ingest.add_argument("--output-dir", required=True)
    workflow_proposal_ingest.add_argument("--max-pages", type=int, default=12)
    workflow_proposal_ingest.add_argument("--max-chars", type=int, default=20000)
    workflow_proposal_ingest.add_argument("--json", default="{}", help="Optional JSON arguments for local-private proposal evidence ingestion")
    workflow_proposal_planning = workflow_sub.add_parser("proposal-planning-brief")
    workflow_proposal_planning.add_argument("--fixture")
    workflow_proposal_planning.add_argument("--evidence-dir")
    workflow_proposal_planning.add_argument("--proposal-evidence-dir")
    workflow_proposal_planning.add_argument("--output-dir", required=True)
    workflow_proposal_planning.add_argument("--proposal-id")
    workflow_proposal_planning.add_argument("--json", default="{}", help="Optional JSON arguments for proposal planning brief")
    workflow_proposal_retrospective = workflow_sub.add_parser("proposal-retrospective")
    workflow_proposal_retrospective.add_argument("--fixture")
    workflow_proposal_retrospective.add_argument("--evidence-dir")
    workflow_proposal_retrospective.add_argument("--proposal-evidence-dir")
    workflow_proposal_retrospective.add_argument("--actual-evidence-dir")
    workflow_proposal_retrospective.add_argument("--output-dir", required=True)
    workflow_proposal_retrospective.add_argument("--proposal-id")
    workflow_proposal_retrospective.add_argument("--experiment-id")
    workflow_proposal_retrospective.add_argument("--json", default="{}", help="Optional JSON arguments for proposal retrospective")
    workflow_suggest_scans = workflow_sub.add_parser("suggest-experiment-scans")
    workflow_suggest_scans.add_argument("--fixture")
    workflow_suggest_scans.add_argument("--evidence-dir")
    workflow_suggest_scans.add_argument("--proposal-evidence-dir")
    workflow_suggest_scans.add_argument("--actual-evidence-dir")
    workflow_suggest_scans.add_argument("--output-dir", required=True)
    workflow_suggest_scans.add_argument("--proposal-id")
    workflow_suggest_scans.add_argument("--json", default="{}", help="Optional JSON arguments for experiment scan suggestions")
    workflow_scan_plan = workflow_sub.add_parser("dispersion-scan-plan")
    workflow_scan_plan.add_argument("--data-root", action="append", dest="data_roots", required=True)
    workflow_scan_plan.add_argument("--output-dir", required=True)
    workflow_scan_plan.add_argument("--collimation", help="Numeric arc-minutes, e.g. '60-80-80-120' (default; 'open' approximated as 60)")
    workflow_scan_plan.add_argument("--e-min-mev", type=float, default=2.0)
    workflow_scan_plan.add_argument("--e-max-mev", type=float, default=30.0)
    workflow_scan_plan.add_argument("--json", default="{}", help="Optional JSON arguments for the dispersion scan plan")
    workflow_dispersion_map = workflow_sub.add_parser("dispersion-map-report")
    workflow_dispersion_map.add_argument("--data-root", action="append", dest="data_roots", required=True)
    workflow_dispersion_map.add_argument("--output-dir", required=True)
    workflow_dispersion_map.add_argument("--max-files", type=int, default=2000)
    workflow_dispersion_map.add_argument("--pattern", action="append", dest="patterns")
    workflow_dispersion_map.add_argument("--json", default="{}", help="Optional JSON arguments for the dispersion map report")
    workflow_story = workflow_sub.add_parser("experiment-story-report")
    workflow_story.add_argument("--data-root", action="append", dest="data_roots", required=True)
    workflow_story.add_argument("--output-dir", required=True)
    workflow_story.add_argument("--section", action="append", dest="sections", choices=["alignment", "order_parameters", "dispersion"], help="Limit to specific sections (repeatable); default is all")
    workflow_story.add_argument("--alignment-state-json", action="append", dest="alignment_state_json", help="alignment-state-machine JSON (repeatable, e.g. one per session) for reflection grouping and software-change notes; when omitted the chain is derived from the data roots")
    workflow_story.add_argument("--alignment-peak-report", help="suggest-alignment-peaks JSON used when deriving the alignment state, so HKL assignments (reflection grouping) can resolve")
    workflow_story.add_argument("--max-files", type=int, default=2000)
    workflow_story.add_argument("--pattern", action="append", dest="patterns")
    workflow_story.add_argument("--json", default="{}", help="Optional JSON arguments for the experiment story report")
    workflow_day_index = workflow_sub.add_parser("experiment-day-index")
    workflow_day_index.add_argument("--reports-root", required=True, help="Directory of per-day report bundles (YYYY-MM-DD subdirectories)")
    workflow_day_index.add_argument("--output-dir")
    workflow_day_index.add_argument("--title")
    workflow_day_index.add_argument("--json", default="{}", help="Optional JSON arguments for the day index")
    workflow_dashboard = workflow_sub.add_parser("experiment-dashboard")
    workflow_dashboard.add_argument("--data-root", required=True)
    workflow_dashboard.add_argument("--output-dir", required=True)
    workflow_dashboard.add_argument("--title")
    workflow_dashboard.add_argument("--alignment-state-json", help="Optional alignment-state-machine JSON whose fitted peak parameters are overlaid on the linked scan plots")
    workflow_dashboard.add_argument("--max-files", type=int, default=500, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_dashboard.add_argument("--post-year", type=int, default=0)
    workflow_dashboard.add_argument("--spurion-top-k", type=int, default=50)
    workflow_dashboard.add_argument("--ranking-mode", choices=["diversified", "score"], default="diversified")
    workflow_dashboard.add_argument("--pattern", action="append", dest="patterns")
    workflow_dashboard.add_argument("--overlay-material", action="append", dest="overlay_materials")
    workflow_dashboard.add_argument("--overlay-config")
    workflow_dashboard.add_argument("--experiment-config")
    workflow_dashboard.add_argument("--json", default="{}", help="Optional JSON arguments for dashboard generation")
    workflow_experiment_review = workflow_sub.add_parser("experiment-review")
    workflow_experiment_review.add_argument("--data-root", required=True)
    workflow_experiment_review.add_argument("--output-dir", required=True)
    workflow_experiment_review.add_argument("--title")
    workflow_experiment_review.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_experiment_review.add_argument("--post-year", type=int, default=0)
    workflow_experiment_review.add_argument("--top-k", type=int, default=8)
    workflow_experiment_review.add_argument("--plot-top-k", type=int, default=5)
    workflow_experiment_review.add_argument("--max-scans-per-candidate", type=int, default=12)
    workflow_experiment_review.add_argument("--allow-rebin", action="store_true", help="Explicitly interpolate scans onto a shared axis for order-parameter review meshes")
    workflow_experiment_review.add_argument("--mesh-grid-points", type=int, default=120)
    workflow_experiment_review.add_argument("--experiment-id")
    workflow_experiment_review.add_argument("--hours", type=float)
    workflow_experiment_review.add_argument("--since")
    workflow_experiment_review.add_argument("--pattern", action="append", dest="patterns")
    workflow_experiment_review.add_argument("--experiment-config")
    workflow_experiment_review.add_argument("--corpus-catalog")
    workflow_experiment_review.add_argument("--corpus-query")
    workflow_experiment_review.add_argument("--precedent-index")
    workflow_experiment_review.add_argument("--review-label-store")
    workflow_experiment_review.add_argument("--skip-spurion", action="store_true", help="Skip inelastic spurion triage in the composed experiment review")
    workflow_experiment_review.add_argument("--json", default="{}", help="Optional JSON arguments for composed experiment review")
    workflow_spurion_survey = workflow_sub.add_parser("survey-inelastic-spurion-evidence")
    workflow_spurion_survey.add_argument("--data-root", required=True)
    workflow_spurion_survey.add_argument("--output-dir", required=True)
    workflow_spurion_survey.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_spurion_survey.add_argument("--post-year", type=int, default=2010)
    workflow_spurion_survey.add_argument("--spurion-top-k", type=int, default=50)
    workflow_spurion_survey.add_argument("--ranking-mode", choices=["diversified", "score"], default="diversified")
    workflow_spurion_survey.add_argument("--pattern", action="append", dest="patterns")
    workflow_spurion_survey.add_argument("--experiment-config")
    workflow_spurion_survey.add_argument("--review-label-store")
    workflow_spurion_survey.add_argument("--json", default="{}", help="Optional JSON arguments for inelastic spurion survey")
    workflow_live = workflow_sub.add_parser("live-experiment-page", help="Rebuild the static current-experiment summary page(s) from the public NCNR archive (or an offline copy)")
    workflow_live.add_argument("--config", help="YAML config (instruments, html_root, cache_root, annotations_root); see deploy/tripleaxis_summary/config.example.yaml")
    workflow_live.add_argument("--offline-data-root", help="Build from a local .../<cycle>/<id>/data directory instead of fetching")
    workflow_live.add_argument("--output-dir", help="Override html_root")
    workflow_live.add_argument("--instrument", default="bt7")
    workflow_psd_pattern = workflow_sub.add_parser("psd-diffraction-pattern", help="Reduce one BT-7 PSD two-axis scan to intensity against sample two-theta with counting errors (CSV, JSON, PNG)")
    workflow_psd_pattern.add_argument("--data-file", required=True)
    workflow_psd_pattern.add_argument("--calibration", required=True, help="Calibration JSON from psd-calibration derive-findpeak, or a legacy PSD_*.dat file")
    workflow_psd_pattern.add_argument("--output-dir")
    workflow_psd_pattern.add_argument("--bin-width", type=float, default=0.1, help="Two-theta bin width in degrees")
    workflow_psd_pattern.add_argument("--sign-convention", choices=["minus_relative_a4", "plus_relative_a4"], help="Override the sign found by the scan check")
    workflow_psd_pattern.add_argument("--scale", type=float, help="Override the channel-offset scale found by the scan check")
    workflow_psd_pattern.add_argument("--mask-channel", action="append", type=int, dest="masked_channels", help="Zero-based channel to exclude (repeatable)")
    workflow_psd_pattern.add_argument("--monitor-reference", type=float, help="Intensities are counts per this many monitor counts (default: the scan's median monitor)")
    workflow_psd_pattern.add_argument("--no-check", action="store_true", help="Skip the Bragg-peak sign/scale check against the scan")
    workflow_overnight = workflow_sub.add_parser("overnight-scan-summary")
    workflow_overnight.add_argument("--alignment-state-json", help="Optional alignment-state-machine JSON: reflection assignments group the alignment plots and console-log software changes are noted beside them")
    workflow_overnight.add_argument("--data-root", required=True)
    workflow_overnight.add_argument("--output-dir", required=True)
    workflow_overnight.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_overnight.add_argument("--post-year", type=int, default=0)
    workflow_overnight.add_argument("--hours", type=float)
    workflow_overnight.add_argument("--since")
    workflow_overnight.add_argument("--pattern", action="append", dest="patterns")
    workflow_overnight.add_argument("--experiment-config")
    workflow_overnight.add_argument("--precedent-index")
    workflow_overnight.add_argument("--top-k", type=int, default=10)
    workflow_overnight.add_argument("--json", default="{}", help="Optional JSON arguments for overnight scan summary")
    workflow_morning_draft = workflow_sub.add_parser("draft-overnight-report")
    workflow_morning_draft.add_argument("--output-dir", required=True)
    workflow_morning_draft.add_argument("--summary-json")
    workflow_morning_draft.add_argument("--data-root")
    workflow_morning_draft.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect when building a summary; use 0 for no explicit cap")
    workflow_morning_draft.add_argument("--post-year", type=int, default=0)
    workflow_morning_draft.add_argument("--hours", type=float)
    workflow_morning_draft.add_argument("--since")
    workflow_morning_draft.add_argument("--pattern", action="append", dest="patterns")
    workflow_morning_draft.add_argument("--experiment-config")
    workflow_morning_draft.add_argument("--precedent-index")
    workflow_morning_draft.add_argument("--json", default="{}", help="Optional JSON arguments for morning report drafting")
    workflow_inelastic_report = workflow_sub.add_parser("inelastic-series-report")
    workflow_inelastic_report.add_argument("--data-root", required=True)
    workflow_inelastic_report.add_argument("--output-dir", required=True)
    workflow_inelastic_report.add_argument("--series-id")
    workflow_inelastic_report.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_inelastic_report.add_argument("--post-year", type=int, default=0)
    workflow_inelastic_report.add_argument("--hours", type=float)
    workflow_inelastic_report.add_argument("--since")
    workflow_inelastic_report.add_argument("--pattern", action="append", dest="patterns")
    workflow_inelastic_report.add_argument("--experiment-config")
    workflow_inelastic_report.add_argument("--json", default="{}", help="Optional JSON arguments for inelastic series report")
    workflow_order_parameter = workflow_sub.add_parser("order-parameter-triage")
    workflow_order_parameter.add_argument("--data-root", required=True)
    workflow_order_parameter.add_argument("--output-dir", required=True)
    workflow_order_parameter.add_argument("--max-files", type=int, default=1000, help="Maximum scan files to inspect; use 0 for no explicit cap")
    workflow_order_parameter.add_argument("--post-year", type=int, default=0)
    workflow_order_parameter.add_argument("--hours", type=float)
    workflow_order_parameter.add_argument("--since")
    workflow_order_parameter.add_argument("--pattern", action="append", dest="patterns")
    workflow_order_parameter.add_argument("--min-scans", type=int, default=4)
    workflow_order_parameter.add_argument("--top-k", type=int, default=20)
    workflow_order_parameter.add_argument("--review-label-store")
    workflow_order_parameter.add_argument("--json", default="{}", help="Optional JSON arguments for order-parameter triage")
    workflow_order_fit = workflow_sub.add_parser("fit-order-parameter")
    workflow_order_fit.add_argument("--triage-json", required=True)
    workflow_order_fit.add_argument("--candidate-id")
    workflow_order_fit.add_argument("--output-dir", required=True)
    workflow_order_fit.add_argument("--observable", choices=["auto", "splitting", "integrated_intensity"], default="auto")
    workflow_order_fit.add_argument("--fit-backend", choices=["core_preview", "panweb_global"], default="core_preview")
    workflow_order_fit.add_argument("--panweb-path")
    workflow_order_fit.add_argument("--model-type", default="split")
    workflow_order_fit.add_argument("--background-model", default="fitted_constant")
    workflow_order_fit.add_argument("--review-status", default="unreviewed", choices=["unreviewed", "human_reviewed"])
    workflow_order_fit.add_argument("--reviewed-by")
    workflow_order_fit.add_argument("--accept-fit", action="store_true")
    workflow_order_fit.add_argument("--op-function", default="power_law")
    workflow_order_fit.add_argument("--tanh-width", type=float, default=5.0)
    workflow_order_fit.add_argument("--use-tc-smearing", action="store_true")
    workflow_order_fit.add_argument("--fit-method", default="dream")
    workflow_order_fit.add_argument("--parallel", type=int, default=1)
    workflow_order_fit.add_argument("--fit-steps", type=int, default=800)
    workflow_order_fit.add_argument("--fit-burn", type=int, default=200)
    workflow_order_fit.add_argument("--initial-tc", type=float)
    workflow_order_fit.add_argument("--initial-beta", type=float)
    workflow_order_fit.add_argument("--inverse-transition", action="store_true")
    workflow_order_fit.add_argument("--compare-models", action="store_true", help="Fit the pan-web model grid (model_type x op_function) and rank variants by AIC/BIC (panweb_global backend only)")
    workflow_order_fit.add_argument("--model-grid", help="Comma-separated model_type:op_function pairs for --compare-models (default: full split/magnetic x power_law/brillouin/tanh grid)")
    workflow_order_fit.add_argument("--json", default="{}", help="Optional JSON arguments for order-parameter fit")
    workflow_dag_replay = workflow_sub.add_parser("dag-run-replay")
    workflow_dag_replay.add_argument("--dag-run-json", required=True)
    workflow_dag_replay.add_argument("--dag-run-id")
    workflow_dag_replay.add_argument("--output-dir", required=True)
    workflow_dag_replay.add_argument("--workflow-version")
    workflow_dag_replay.add_argument("--json", default="{}", help="Optional JSON arguments, including correction_overrides")
    workflow_trajectory_cluster = workflow_sub.add_parser("trajectory-cluster-order-parameters")
    workflow_trajectory_cluster.add_argument("--input", help="CSV or JSON trajectory table")
    workflow_trajectory_cluster.add_argument("--triage-json", help="Order-parameter triage JSON containing candidate per-scan summaries")
    workflow_trajectory_cluster.add_argument("--output-dir", required=True)
    workflow_trajectory_cluster.add_argument("--id-column", default="trajectory_id")
    workflow_trajectory_cluster.add_argument("--parameter-column", default="temperature")
    workflow_trajectory_cluster.add_argument("--value-column", default="intensity")
    workflow_trajectory_cluster.add_argument("--observable", default="integrated_signal", help="Per-scan field to cluster when --triage-json is used")
    workflow_trajectory_cluster.add_argument("--backend", choices=["shape_rules", "auto", "gmm"], default="shape_rules")
    workflow_trajectory_cluster.add_argument("--n-clusters", type=int, default=4)
    workflow_trajectory_cluster.add_argument("--max-grid-points", type=int, default=25)
    workflow_trajectory_cluster.add_argument("--json", default="{}", help="Optional JSON arguments for trajectory clustering")
    workflow_trajectory_report = workflow_sub.add_parser("order-parameter-trajectory-report")
    workflow_trajectory_report.add_argument("--triage-json", required=True)
    workflow_trajectory_report.add_argument("--trajectory-json", required=True)
    workflow_trajectory_report.add_argument("--output-dir", required=True)
    workflow_trajectory_report.add_argument("--top-k", type=int, default=25)
    workflow_trajectory_report.add_argument("--json", default="{}", help="Optional JSON arguments for combined order-parameter trajectory report")
    workflow_order_plots = workflow_sub.add_parser("plot-order-parameter-candidates")
    workflow_order_plots.add_argument("--report-json", required=True)
    workflow_order_plots.add_argument("--output-dir", required=True)
    workflow_order_plots.add_argument("--top-k", type=int, default=5)
    workflow_order_plots.add_argument("--max-scans-per-candidate", type=int, default=12)
    workflow_order_plots.add_argument("--experiment-id")
    workflow_order_plots.add_argument("--allow-rebin", action="store_true", help="Explicitly interpolate scans onto a shared axis for review mesh plots")
    workflow_order_plots.add_argument("--mesh-grid-points", type=int, default=120)
    workflow_order_plots.add_argument("--json", default="{}", help="Optional JSON arguments for candidate plotting")
    workflow_bt7_index = workflow_sub.add_parser("build-bt7-hydrated-index")
    workflow_bt7_index.add_argument("--data-catalog", required=True)
    workflow_bt7_index.add_argument("--output", required=True)
    workflow_bt7_index.add_argument("--output-dir")
    workflow_bt7_index.add_argument("--max-files", type=int)
    workflow_bt7_index.add_argument("--path-contains", help="Optional substring filter for local hydrated path triage")
    workflow_classify_psd = workflow_sub.add_parser("classify-psd-sources")
    workflow_classify_psd.add_argument("--catalog", required=True)
    workflow_classify_psd.add_argument("--output-dir")
    workflow_classify_psd.add_argument("--json", default="{}", help="Optional JSON records override")
    workflow_reconstruct_psd = workflow_sub.add_parser("reconstruct-psd-map")
    workflow_reconstruct_psd.add_argument("--catalog", required=True)
    workflow_reconstruct_psd.add_argument("--group-id", required=True)
    workflow_reconstruct_psd.add_argument("--level", default="channel_temperature")
    workflow_reconstruct_psd.add_argument("--output-dir", required=True)
    workflow_reconstruct_psd.add_argument("--json", default="{}", help="Optional JSON records override")
    workflow_detector_cuts = workflow_sub.add_parser("plot-psd-detector-cuts")
    workflow_detector_cuts.add_argument("--catalog", required=True)
    workflow_detector_cuts.add_argument("--scan-id")
    workflow_detector_cuts.add_argument("--group-id")
    workflow_detector_cuts.add_argument("--temperature-min", type=float)
    workflow_detector_cuts.add_argument("--temperature-max", type=float)
    workflow_detector_cuts.add_argument("--output-dir", required=True)
    workflow_detector_cuts.add_argument("--hydrate", action="store_true", help="Hydrate path-only/header-only records through the registered BT-7 reader before plotting")
    workflow_detector_cuts.add_argument("--json", default="{}", help="Optional JSON records override")
    workflow_compare_cuts = workflow_sub.add_parser("compare-psd-detector-cuts")
    workflow_compare_cuts.add_argument("--catalog", required=True)
    workflow_compare_cuts.add_argument("--left-scan-id", required=True)
    workflow_compare_cuts.add_argument("--right-scan-id", required=True)
    workflow_compare_cuts.add_argument("--temperature-min", type=float)
    workflow_compare_cuts.add_argument("--temperature-max", type=float)
    workflow_compare_cuts.add_argument("--temperature-tolerance", type=float, default=2.0)
    workflow_compare_cuts.add_argument("--output-dir", required=True)
    workflow_compare_cuts.add_argument("--hydrate", action="store_true", help="Hydrate selected records through the registered BT-7 reader before comparing")
    workflow_compare_cuts.add_argument("--json", default="{}", help="Optional JSON records override")
    workflow_channel_artifact = workflow_sub.add_parser("check-psd-channel-artifact")
    workflow_channel_artifact.add_argument("--catalog", required=True)
    workflow_channel_artifact.add_argument("--scan-id", required=True)
    workflow_channel_artifact.add_argument("--channel", action="append", type=float, dest="channels")
    workflow_channel_artifact.add_argument("--temperature-min", type=float)
    workflow_channel_artifact.add_argument("--temperature-max", type=float)
    workflow_channel_artifact.add_argument("--neighbor-width", type=int, default=2)
    workflow_channel_artifact.add_argument("--output-dir", required=True)
    workflow_channel_artifact.add_argument("--hydrate", action="store_true", help="Hydrate selected record through the registered BT-7 reader before checking channels")
    workflow_channel_artifact.add_argument("--json", default="{}", help="Optional JSON records override")
    workflow_copilot = workflow_sub.add_parser("copilot-ask")
    workflow_copilot.add_argument("--question")
    workflow_copilot.add_argument("--output-dir")
    workflow_copilot.add_argument("--list-intents", "--examples", action="store_true", help="List supported bounded-router situations and example questions")
    workflow_copilot.add_argument("--corpus-catalog")
    workflow_copilot.add_argument("--data-catalog")
    workflow_copilot.add_argument("--data-root")
    workflow_copilot.add_argument("--precedent-index")
    workflow_copilot.add_argument("--paper-id")
    workflow_copilot.add_argument("--figure-id")
    workflow_copilot.add_argument("--proposal-evidence-dir")
    workflow_copilot.add_argument("--actual-evidence-dir")
    workflow_copilot.add_argument("--fixture")
    workflow_copilot.add_argument("--experiment-config")
    workflow_copilot.add_argument("--max-files", type=int, default=1000)
    workflow_copilot.add_argument("--post-year", type=int, default=2011)
    workflow_copilot.add_argument("--hours", type=float)
    workflow_copilot.add_argument("--since")
    workflow_copilot.add_argument("--top-k", type=int, default=10)
    workflow_copilot.add_argument("--json", default="{}", help="Optional JSON arguments for bounded copilot routing")
    workflow_mask_sensitivity = workflow_sub.add_parser("psd-mask-sensitivity")
    workflow_mask_sensitivity.add_argument("--data-file", action="append", dest="data_files")
    workflow_mask_sensitivity.add_argument("--psd-calibration-dir", required=True)
    workflow_mask_sensitivity.add_argument("--calibration-file", action="append", dest="calibration_files")
    workflow_mask_sensitivity.add_argument("--annotated-q", type=float, default=1.24)
    workflow_mask_sensitivity.add_argument("--output-dir", required=True)
    workflow_mask_sensitivity.add_argument("--path-only", action="store_true")
    workflow_mask_sensitivity.add_argument("--json", default="{}", help="Optional JSON arguments for PSD mask sensitivity")
    workflow_review_candidate = workflow_sub.add_parser("review-figure-source-candidate")
    workflow_review_candidate.add_argument("--catalog", required=True)
    workflow_review_candidate.add_argument("--figure-report", required=True)
    workflow_review_candidate.add_argument("--candidate-id")
    workflow_review_candidate.add_argument("--temperature-min", type=float)
    workflow_review_candidate.add_argument("--temperature-max", type=float)
    workflow_review_candidate.add_argument("--temperature-tolerance", type=float, default=2.0)
    workflow_review_candidate.add_argument("--channel", action="append", type=float, dest="channels")
    workflow_review_candidate.add_argument("--calibration-registry")
    workflow_review_candidate.add_argument("--psd-calibration-dir")
    workflow_review_candidate.add_argument("--psd-calibration-file", action="append", dest="psd_calibration_files")
    workflow_review_candidate.add_argument("--psd-calibration-path-only", action="store_true")
    workflow_review_candidate.add_argument("--annotated-q", type=float)
    workflow_review_candidate.add_argument("--output-dir", required=True)
    workflow_review_candidate.add_argument("--hydrate", action="store_true", help="Hydrate candidate files through the registered BT-7 reader before diagnostics")
    workflow_review_candidate.add_argument("--json", default="{}", help="Optional JSON arguments for candidate review")
    workflow_instrument_graph = workflow_sub.add_parser("instrument-device-graph")
    workflow_instrument_graph.add_argument("--nice-report", required=True)
    workflow_instrument_graph.add_argument("--output-dir", required=True)
    workflow_instrument_graph.add_argument("--alias-config")
    workflow_instrument_graph.add_argument("--json", default="{}", help="Optional JSON arguments for instrument graph extraction")
    workflow_nice_replay = workflow_sub.add_parser("nice-bt7-replay")
    workflow_nice_replay.add_argument("--replay-plan", required=True)
    workflow_nice_replay.add_argument("--output-dir", required=True)
    workflow_nice_replay.add_argument("--recipe-id")
    workflow_nice_replay.add_argument("--executor", choices=["dry_run", "mock", "nice_remote", "nice_container", "nice_container_session"], default="dry_run")
    workflow_nice_replay.add_argument("--mode", choices=["point_by_point_adaptive", "trajectory_replay"], default="point_by_point_adaptive")
    workflow_nice_replay.add_argument("--host", default="127.0.0.1")
    workflow_nice_replay.add_argument("--port", type=int, default=1232)
    workflow_nice_replay.add_argument("--client-name", default="tas-copilot-nice-replay")
    workflow_nice_replay.add_argument("--file-prefix")
    workflow_nice_replay.add_argument("--preset-time", type=float)
    workflow_nice_replay.add_argument("--max-points", type=int, default=25)
    workflow_nice_replay.add_argument("--python-executable")
    workflow_nice_replay.add_argument("--container-image", default="nice-bt7-replay-smoke:latest")
    workflow_nice_replay.add_argument("--allow-live-host", action="store_true", help="Permit a non-local NICE host (a real instrument). Hardware limits stay with NICE.")
    workflow_nice_replay.add_argument("--nice-checkout", help="Path to external NICE checkout mounted into the container")
    workflow_nice_replay.add_argument("--docker-executable", default="docker")
    workflow_nice_replay.add_argument("--timeout-seconds", type=int)
    workflow_nice_replay.add_argument("--allow-unreviewed-node-hints", action="store_true", help="Allow explicitly local simulator execution using unreviewed NICE node hints")
    workflow_nice_replay.add_argument("--json", default="{}", help="Optional JSON arguments for NICE simulator replay")
    workflow_tasai_nice = workflow_sub.add_parser("tasai-adaptive-nice-sim")
    workflow_tasai_nice.add_argument("--output-dir", required=True)
    workflow_tasai_nice.add_argument("--runtime-profile", help="YAML runtime profile with TACC/NIST paths, account, partition, ControlPath, and container image locations")
    workflow_tasai_nice.add_argument("--candidate-points-file")
    workflow_tasai_nice.add_argument("--model-spec-file")
    workflow_tasai_nice.add_argument("--software-limits-file")
    workflow_tasai_nice.add_argument("--human-suggestions", help="JSONL file polled before each adaptive step; rows may suggest points, adjust count time, or stop")
    workflow_tasai_nice.add_argument("--policy-backend", default="fixture_tasai")
    workflow_tasai_nice.add_argument("--count-backend", default="fixture_spinw")
    workflow_tasai_nice.add_argument("--tasai-policy-adapter", help="Python module file exposing propose_next_point(payload) for external_tasai_adapter")
    workflow_tasai_nice.add_argument("--andie-policy-adapter", help="Python module file exposing propose_next_point(payload) for external_andie_adapter")
    workflow_tasai_nice.add_argument("--andie-container-image", default="tascopilot-andie:latest")
    workflow_tasai_nice.add_argument("--andie-container-workspace", help="Path to external ANDiE container wrapper workspace")
    workflow_tasai_nice.add_argument("--andie-source-root", help="Path to public/full ANDiE notebook checkout mounted read-only")
    workflow_tasai_nice.add_argument("--pyspinw-count-adapter", help="Python module file exposing simulate_counts(payload) for external_pyspinw_adapter")
    workflow_tasai_nice.add_argument("--tasai-path", help="External tasAI checkout for tasai_analytic count models")
    workflow_tasai_nice.add_argument("--analytic-model", default="ferromagnetic_magnon", help="Allowed tasAI analytic simulator function for tasai_analytic counts")
    workflow_tasai_nice.add_argument("--analytic-parameters", help="JSON object passed to the selected tasAI analytic model")
    workflow_tasai_nice.add_argument("--nice-executor", choices=["dry_run", "mock", "nice_remote", "nice_remote_session", "nice_container", "nice_container_session"], default="mock")
    workflow_tasai_nice.add_argument("--max-steps", type=int, default=5)
    workflow_tasai_nice.add_argument("--count-time", type=float, default=1.0)
    workflow_tasai_nice.add_argument("--host", default="127.0.0.1")
    workflow_tasai_nice.add_argument("--port", type=int, default=1232)
    workflow_tasai_nice.add_argument("--client-name", default="tas-copilot-tasai-adaptive")
    workflow_tasai_nice.add_argument("--container-image", default="nice-bt7-replay-smoke:latest")
    workflow_tasai_nice.add_argument("--allow-live-host", action="store_true", help="Permit a non-local NICE host (a real instrument). Hardware limits stay with NICE.")
    workflow_tasai_nice.add_argument("--python-executable", help="Interpreter with zeroc-ice + the NICE client for the nice_remote executor subprocess (e.g. the nice-client conda env)")
    workflow_tasai_nice.add_argument("--nice-checkout", help="Path to external NICE checkout mounted into the container")
    workflow_tasai_nice.add_argument("--docker-executable", default="docker")
    workflow_tasai_nice.add_argument("--timeout-seconds", type=int, help="Timeout for each NICE replay executor call")
    workflow_tasai_nice.add_argument("--suggestion-poll-seconds", type=float, default=0.0)
    workflow_tasai_nice.add_argument("--step-delay-seconds", type=float, default=0.0)
    workflow_tasai_nice.add_argument("--prepare-hpc-handoff", action="store_true", help="Write a TACC handoff payload instead of running unavailable HPC backends locally")
    workflow_tasai_nice.add_argument("--hpc-action", choices=["prepare", "submit", "poll", "fetch"], default="prepare", help="TACC handoff action scaffold; submit/poll/fetch write fail-closed reports")
    workflow_tasai_nice.add_argument("--hpc-system")
    workflow_tasai_nice.add_argument("--hpc-work-dir")
    workflow_tasai_nice.add_argument("--hpc-account", help="TACC allocation/account for generated SLURM scripts")
    workflow_tasai_nice.add_argument("--hpc-partition", help="TACC partition for generated SLURM scripts")
    workflow_tasai_nice.add_argument("--hpc-container-runtime", default="apptainer", help="Remote container runtime hint for generated TACC scripts")
    workflow_tasai_nice.add_argument("--andie-remote-image", help="Remote Apptainer/Singularity image path for ANDiE policy jobs")
    workflow_tasai_nice.add_argument("--andie-remote-workspace", help="Remote ANDiE container wrapper workspace path")
    workflow_tasai_nice.add_argument("--andie-remote-source-root", help="Remote full ANDiE source/notebook checkout path")
    workflow_tasai_nice.add_argument("--hpc-login", help="SSH login target for reviewed TACC submit/poll/fetch, e.g. user@stampede3.tacc.utexas.edu")
    workflow_tasai_nice.add_argument("--hpc-remote-handoff-dir", help="Expanded absolute remote directory for TACC handoff files")
    workflow_tasai_nice.add_argument("--hpc-remote-output-dir", help="Expanded absolute remote output directory to fetch")
    workflow_tasai_nice.add_argument("--hpc-job-id", help="SLURM job id for hpc-action=poll")
    workflow_tasai_nice.add_argument("--hpc-ssh-executable", default="ssh")
    workflow_tasai_nice.add_argument("--hpc-scp-executable", default="scp")
    workflow_tasai_nice.add_argument("--hpc-submit-command", help="Remote submit command, default: sbatch submit_andie_policy.slurm")
    workflow_tasai_nice.add_argument("--hpc-poll-command", help="Remote scheduler poll command override")
    workflow_tasai_nice.add_argument("--controlpath")
    workflow_tasai_nice.add_argument("--allow-unreviewed-node-hints", action="store_true")
    workflow_tasai_nice.add_argument("--json", default="{}", help="Optional JSON arguments for adaptive tasAI/NICE simulation")
    workflow_psd_calibration = workflow_sub.add_parser("discover-psd-calibrations")
    workflow_psd_calibration.add_argument("--data-root", required=True)
    workflow_psd_calibration.add_argument("--output-dir")
    workflow_psd_calibration.add_argument("--max-files", type=int, default=100000, help="Maximum files to inspect; use 0 for no limit")
    workflow_psd_calibration.add_argument("--json", default="{}", help="Optional JSON arguments for PSD calibration discovery")
    workflow_alignment_peaks = workflow_sub.add_parser("suggest-alignment-peaks")
    workflow_alignment_peaks.add_argument("--formula", required=True)
    workflow_alignment_peaks.add_argument("--icsd-db")
    workflow_alignment_peaks.add_argument("--scattering-plane")
    workflow_alignment_peaks.add_argument("--lattice", help="Experiment cell 'a b c alpha beta gamma' (e.g. from BT-7 headers) used to filter ICSD candidates to the matching polymorph/setting")
    workflow_alignment_peaks.add_argument("--space-group")
    workflow_alignment_peaks.add_argument("--space-group-number", type=int)
    workflow_alignment_peaks.add_argument("--space-group-setting", help="GSAS-II-readable setting symbol used after numeric ICSD filtering, e.g. 'P m -3 m' or 'R -3 c H'")
    workflow_alignment_peaks.add_argument("--gsasii-python")
    workflow_alignment_peaks.add_argument("--gsasii-path")
    workflow_alignment_peaks.add_argument("--gsasii-bin")
    workflow_alignment_peaks.add_argument("--gsasii-runtime")
    workflow_alignment_peaks.add_argument("--energy-mev", type=float, default=14.7)
    workflow_alignment_peaks.add_argument("--a4-max-deg", type=float, default=120.0)
    workflow_alignment_peaks.add_argument("--top-k", type=int, default=30)
    workflow_alignment_peaks.add_argument("--candidate-limit", type=int, default=50)
    workflow_alignment_peaks.add_argument("--max-reflections", type=int, default=5000)
    workflow_alignment_peaks.add_argument("--background-material-config")
    workflow_alignment_peaks.add_argument("--background-material", action="append")
    workflow_alignment_peaks.add_argument("--background-overlap-tolerance-deg", type=float, default=0.5)
    workflow_alignment_peaks.add_argument("--max-background-reflections", type=int, default=300)
    workflow_alignment_peaks.add_argument("--sample-thickness-mm", type=float)
    workflow_alignment_peaks.add_argument("--out-of-plane-low-q-deg", type=float)
    workflow_alignment_peaks.add_argument("--out-of-plane-high-q-deg", type=float)
    workflow_alignment_peaks.add_argument("--out-of-plane-max-deg", type=float, default=5.0, help="Hard projection cutoff from the scattering plane; reflections beyond this angle are filtered")
    workflow_alignment_peaks.add_argument("--output-dir", required=True)
    workflow_alignment_peaks.add_argument("--json", default="{}", help="Optional JSON arguments for alignment peak suggestions")
    workflow_alignment_index = workflow_sub.add_parser("alignment-corpus-index")
    workflow_alignment_index.add_argument("--data-root", action="append", dest="data_roots", required=True)
    workflow_alignment_index.add_argument("--output-dir", required=True)
    workflow_alignment_index.add_argument("--max-files", type=int, default=5000)
    workflow_alignment_index.add_argument("--max-log-lines", type=int, default=5000)
    workflow_alignment_index.add_argument("--sample-limit", type=int, default=50)
    workflow_alignment_index.add_argument("--alignment-peak-report", help="Optional suggest-alignment-peaks JSON report to link planning candidates to observed alignment events")
    workflow_alignment_index.add_argument("--no-logs", action="store_true", help="Index BT-7 scan files only; skip console logs")
    workflow_alignment_index.add_argument("--json", default="{}", help="Optional JSON arguments for alignment corpus indexing")
    workflow_alignment_epochs = workflow_sub.add_parser("alignment-state-epochs")
    workflow_alignment_epochs.add_argument("--events-jsonl", required=True)
    workflow_alignment_epochs.add_argument("--alignment-peak-report")
    workflow_alignment_epochs.add_argument("--output-dir", required=True)
    workflow_alignment_epochs.add_argument("--json", default="{}", help="Optional JSON arguments for alignment state epochs")
    workflow_alignment_machine = workflow_sub.add_parser("alignment-state-machine")
    workflow_alignment_machine.add_argument("--events-jsonl", required=True)
    workflow_alignment_machine.add_argument("--alignment-peak-report")
    workflow_alignment_machine.add_argument("--max-alignment-peaks", type=int, default=3)
    workflow_alignment_machine.add_argument("--peak-fit-backend", choices=["panweb_if_available", "panweb", "internal_preview"], default="internal_preview")
    workflow_alignment_machine.add_argument("--panweb-path")
    workflow_alignment_machine.add_argument("--panweb-use-dream", action="store_true")
    workflow_alignment_machine.add_argument("--panweb-parallel", type=int, default=1)
    workflow_alignment_machine.add_argument("--background-material-config")
    workflow_alignment_machine.add_argument("--background-material", action="append")
    workflow_alignment_machine.add_argument("--background-peak-table")
    workflow_alignment_machine.add_argument("--background-tolerance-deg", type=float, default=0.6)
    workflow_alignment_machine.add_argument(
        "--instrument-calibrations",
        help="Instrument-calibration registry YAML (for example configs/instrument_calibrations/registry.yaml); defaults to the checked-in registry when present",
    )
    workflow_alignment_machine.add_argument("--axis-policy", help="YAML/JSON axis policy: motors that must not be optimized in this run; recommendations and next checks touching them are marked blocked")
    workflow_alignment_machine.add_argument("--output-dir", required=True)
    workflow_alignment_machine.add_argument("--json", default="{}", help="Optional JSON arguments for alignment state machine")
    workflow_bt7_pass = workflow_sub.add_parser("bt7-corpus-candidate-pass")
    workflow_bt7_pass.add_argument("--data-root", action="append", dest="data_roots", required=True)
    workflow_bt7_pass.add_argument("--output-dir", required=True)
    workflow_bt7_pass.add_argument("--max-files", type=int, default=0, help="Maximum scan files per root; use 0 for no explicit cap")
    workflow_bt7_pass.add_argument("--post-year", type=int, default=2010)
    workflow_bt7_pass.add_argument("--max-log-lines", type=int, default=5000)
    workflow_bt7_pass.add_argument("--sample-limit", type=int, default=50)
    workflow_bt7_pass.add_argument("--spurion-top-k", type=int, default=100)
    workflow_bt7_pass.add_argument("--order-top-k", type=int, default=100)
    workflow_bt7_pass.add_argument("--min-scans", type=int, default=4)
    workflow_bt7_pass.add_argument("--review-label-store")
    workflow_bt7_pass.add_argument("--no-logs", action="store_true")
    workflow_bt7_pass.add_argument("--json", default="{}", help="Optional JSON arguments for full BT-7 corpus candidate pass")
    workflow_alignment_review_dashboard = workflow_sub.add_parser("alignment-review-dashboard")
    workflow_alignment_review_dashboard.add_argument("--review-template", required=True)
    workflow_alignment_review_dashboard.add_argument("--output-dir", required=True)
    workflow_alignment_review_dashboard.add_argument("--html-filename", default="alignment_review_dashboard.html")
    workflow_alignment_review_dashboard.add_argument("--max-dashboard-rows", type=int, default=5000, help="Maximum rows embedded in the HTML dashboard; use 0 for no explicit cap")
    workflow_alignment_review_dashboard.add_argument("--json", default="{}", help="Optional JSON arguments for alignment review dashboard")
    workflow_alignment_brief = workflow_sub.add_parser("alignment-planning-brief")
    workflow_alignment_brief.add_argument("--events-jsonl")
    workflow_alignment_brief.add_argument("--state-json")
    workflow_alignment_brief.add_argument("--alignment-peak-report")
    workflow_alignment_brief.add_argument("--ef-mev", type=float, help="Fixed final energy in meV; enables kinematic reachability annotation of candidate peaks via tas.psd_planner.check_accessibility")
    workflow_alignment_brief.add_argument("--output-dir", required=True)
    workflow_alignment_brief.add_argument("--json", default="{}", help="Optional JSON arguments for alignment planning brief")
    workflow_alignment_replay_plan = workflow_sub.add_parser("alignment-fine-replay-plan")
    workflow_alignment_replay_plan.add_argument("--state-json")
    workflow_alignment_replay_plan.add_argument("--brief-json")
    workflow_alignment_replay_plan.add_argument("--rocking-motor", default="A3")
    workflow_alignment_replay_plan.add_argument("--count-time-seconds", type=float, default=1.0)
    workflow_alignment_replay_plan.add_argument("--axis-policy", help="YAML/JSON axis policy: motors that must not be optimized in this run (see configs/alignment/)")
    workflow_alignment_replay_plan.add_argument("--output-dir", required=True)
    workflow_alignment_replay_plan.add_argument("--json", default="{}", help="Optional JSON arguments for simulator-only fine alignment replay planning")
    workflow_alignment_adaptive = workflow_sub.add_parser("alignment-adaptive-nice-loop")
    workflow_alignment_adaptive.add_argument("--events-jsonl")
    workflow_alignment_adaptive.add_argument("--state-json")
    workflow_alignment_adaptive.add_argument("--rocking-motor", default="A3")
    workflow_alignment_adaptive.add_argument("--count-time-seconds", type=float, default=1.0)
    workflow_alignment_adaptive.add_argument("--max-iterations", type=int, default=3)
    workflow_alignment_adaptive.add_argument("--max-points", type=int, default=25)
    workflow_alignment_adaptive.add_argument("--axis-policy", help="YAML/JSON axis policy: motors that must not be optimized in this run (see configs/alignment/)")
    workflow_alignment_adaptive.add_argument("--executor", choices=["dry_run", "mock", "nice_remote", "nice_container", "nice_container_session"], default="dry_run")
    workflow_alignment_adaptive.add_argument("--execute-dry-run", action="store_true", help="Run dry-run NICE replay recipes after preparing the adaptive loop")
    workflow_alignment_adaptive.add_argument("--allow-simulator-execution", action="store_true", help="Permit explicit local simulator executors other than dry_run")
    workflow_alignment_adaptive.add_argument("--host", default="127.0.0.1")
    workflow_alignment_adaptive.add_argument("--port", type=int, default=1232)
    workflow_alignment_adaptive.add_argument("--client-name", default="tas-copilot-alignment-adaptive")
    workflow_alignment_adaptive.add_argument("--container-image", default="nice-bt7-replay-smoke:latest")
    workflow_alignment_adaptive.add_argument("--nice-checkout")
    workflow_alignment_adaptive.add_argument("--docker-executable", default="docker")
    workflow_alignment_adaptive.add_argument("--timeout-seconds", type=int)
    workflow_alignment_adaptive.add_argument("--allow-unreviewed-node-hints", action="store_true")
    workflow_alignment_adaptive.add_argument("--output-dir", required=True)
    workflow_alignment_adaptive.add_argument("--json", default="{}", help="Optional JSON arguments for adaptive fine-alignment NICE loop planning")
    workflow_planning_brief = workflow_sub.add_parser("experiment-planning-brief")
    workflow_planning_brief.add_argument("--material", required=True)
    workflow_planning_brief.add_argument("--goal", default="")
    workflow_planning_brief.add_argument("--formula")
    workflow_planning_brief.add_argument("--scattering-plane")
    workflow_planning_brief.add_argument("--icsd-db")
    workflow_planning_brief.add_argument("--space-group")
    workflow_planning_brief.add_argument("--space-group-number", type=int)
    workflow_planning_brief.add_argument("--space-group-setting")
    workflow_planning_brief.add_argument("--energy-mev", type=float, default=14.7)
    workflow_planning_brief.add_argument("--a4-max-deg", type=float, default=100.0)
    workflow_planning_brief.add_argument("--out-of-plane-max-deg", type=float, default=5.0)
    workflow_planning_brief.add_argument("--background-overlap-tolerance-deg", type=float, default=0.5)
    workflow_planning_brief.add_argument("--max-background-reflections", type=int, default=300)
    workflow_planning_brief.add_argument("--background-material", action="append")
    workflow_planning_brief.add_argument("--background-material-config")
    workflow_planning_brief.add_argument("--precedent-index")
    workflow_planning_brief.add_argument("--corpus-catalog")
    workflow_planning_brief.add_argument("--data-root")
    workflow_planning_brief.add_argument("--replay-report")
    workflow_planning_brief.add_argument("--run-dashboard", action="store_true")
    workflow_planning_brief.add_argument("--experiment-config")
    workflow_planning_brief.add_argument("--psd-calibration-registry")
    workflow_planning_brief.add_argument("--psd-calibration-dir", action="append", dest="psd_calibration_dirs")
    workflow_planning_brief.add_argument("--max-files", type=int, default=200)
    workflow_planning_brief.add_argument("--top-k", type=int, default=10)
    workflow_planning_brief.add_argument("--output-dir", required=True)
    workflow_planning_brief.add_argument("--json", default="{}", help="Optional JSON arguments for experiment planning brief")
    workflow_workspace = workflow_sub.add_parser("experiment-workspace")
    workflow_workspace.add_argument("--material", required=True)
    workflow_workspace.add_argument("--goal", default="")
    workflow_workspace.add_argument("--data-root")
    workflow_workspace.add_argument("--formula")
    workflow_workspace.add_argument("--family", default="lu_femn_o3")
    workflow_workspace.add_argument("--scattering-plane")
    workflow_workspace.add_argument("--icsd-db")
    workflow_workspace.add_argument("--space-group")
    workflow_workspace.add_argument("--space-group-number", type=int)
    workflow_workspace.add_argument("--space-group-setting")
    workflow_workspace.add_argument("--energy-mev", type=float, default=14.7)
    workflow_workspace.add_argument("--a4-max-deg", type=float, default=100.0)
    workflow_workspace.add_argument("--background-material", action="append")
    workflow_workspace.add_argument("--background-material-config")
    workflow_workspace.add_argument("--background-branch", help="Optional preview background branch override such as no_background, same_scan_sideband, paired_background_scan, temperature_reference, or manual_user_selected")
    workflow_workspace.add_argument("--precedent-index")
    workflow_workspace.add_argument("--corpus-catalog")
    workflow_workspace.add_argument("--replay-report")
    workflow_workspace.add_argument("--experiment-config")
    workflow_workspace.add_argument("--psd-calibration-registry")
    workflow_workspace.add_argument("--psd-calibration-dir", action="append", dest="psd_calibration_dirs")
    workflow_workspace.add_argument("--max-files", type=int, default=200)
    workflow_workspace.add_argument("--top-k", type=int, default=10)
    workflow_workspace.add_argument("--output-dir", required=True)
    workflow_workspace.add_argument("--json", default="{}", help="Optional JSON arguments for experiment workspace")
    catalog = sub.add_parser("catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_completeness = catalog_sub.add_parser("completeness")
    catalog_completeness.add_argument("--catalog", required=True)
    catalog_completeness.add_argument("--json", action="store_true")
    catalog_eligible = catalog_sub.add_parser("eligible-scans")
    catalog_eligible.add_argument("--catalog", required=True)
    catalog_eligible.add_argument("--for", dest="purpose", default="geometry")
    catalog_eligible.add_argument("--limit", type=int)
    catalog_eligible.add_argument("--json", action="store_true")
    catalog_quality = catalog_sub.add_parser("quality")
    catalog_quality.add_argument("--catalog", required=True)
    catalog_quality.add_argument("--json", action="store_true")
    catalog_quality.add_argument("--output-dir")
    for name in ["find-papers", "find-experiments"]:
        command = sub.add_parser(name)
        command.add_argument("query")
        command.add_argument("--limit", type=int, default=10)
    check_resolution_cmd = sub.add_parser("check-resolution")
    check_resolution_cmd.add_argument("--json", default="{}", help="JSON arguments for the tool adapter")
    check_resolution_cmd.add_argument("--output-dir", help="Optional local directory for Plotly resolution visualization artifacts")
    for name in [
        "calculate-angles",
        "calculate-ub",
        "check-spurions",
        "plan-spinwave",
        "powder-order-parameter",
        "read-bt7-data",
        "read-bt7-file",
        "classify-tas-format",
        "read-tas-metadata",
        "scan-tas-directory",
        "group-bt7-scans",
        "translate-ice-file",
        "add-scans",
        "combine-files",
        "subtract-background",
        "apply-corrections",
        "normalize-to-monitor",
        "apply-detailed-balance",
        "summarize-psd-facility",
    ]:
        command = sub.add_parser(name)
        command.add_argument("--json", default="{}", help="JSON arguments for the tool adapter")
    inspect = sub.add_parser("inspect-bt7-directory")
    inspect.add_argument("--path", required=True)
    inspect.add_argument("--json", default="{}", help="Optional JSON arguments for directory inspection")
    sub.add_parser("extract-setup").add_argument("text")
    compare = sub.add_parser("compare-paper-to-data")
    compare.add_argument("paper_id")
    compare.add_argument("--json", default="{}", help="Optional JSON arguments including paper, experiments, and threshold")
    sub.add_parser("analyze-dataset").add_argument("identifier")
    similar = sub.add_parser("retrieve-similar-scans")
    similar.add_argument("scan_id")
    similar.add_argument("--limit", type=int, default=5)
    import_db = sub.add_parser("import-data-catalog")
    import_db.add_argument("--catalog-dir", default=str(CoreConfig().catalog_dir))
    import_db.add_argument("--db", default=str(CoreConfig().data_catalog_db_path))
    summary_db = sub.add_parser("data-catalog-summary")
    summary_db.add_argument("--db", default=str(CoreConfig().data_catalog_db_path))
    quality_db = sub.add_parser("data-catalog-quality")
    quality_db.add_argument("--db", default=str(CoreConfig().data_catalog_db_path))
    quality_db.add_argument("--output-dir")
    catalog_dates = sub.add_parser(
        "catalog-backfill-dates",
        help="Read measurement dates from scan-file headers into a path-only catalog",
    )
    catalog_dates.add_argument("--catalog-sqlite", required=True)
    catalog_dates.add_argument("--path-prefix", default="", help="Scope the run to paths starting with this prefix")
    catalog_dates.add_argument("--limit", type=int, default=0)
    catalog_dates.add_argument(
        "--archive-root",
        default="",
        help="Local archive root; makes the catalog portable across machines by resolving stored paths relative to it",
    )
    catalog_dates.add_argument("--path-marker", default="bt7", help="Segment marking the archive-relative portion of a path")

    catalog_dir = sub.add_parser("catalog-bt7-directory")
    catalog_dir.add_argument("--json", default="{}", help="JSON arguments including root_dirs and optional db_path")
    catalog_tas = sub.add_parser("catalog-tas-directory")
    catalog_tas.add_argument("--path", required=True)
    catalog_tas.add_argument("--json", default="{}", help="Optional JSON arguments including db_path, max_files, and max_records")
    catalog_batch = sub.add_parser("catalog-tas-batch")
    catalog_batch.add_argument("--manifest", required=True, help="YAML matrix with TAS directories to catalog and quality-check")
    eval_smoke_cmd = sub.add_parser("eval-smoke")
    eval_smoke_cmd.add_argument("--json", default="{}", help="Optional JSON arguments including eval_dir, catalog_dir, and limit")

    args = parser.parse_args(argv)
    if args.command is None:
        print(_front_door_menu())
        return 0
    if args.command == "examples":
        if args.json:
            print(_json({"examples": copilot_intent_examples()}))
        else:
            print(_copilot_intent_examples_text())
        return 0
    if args.command == "inventory":
        inventory = _workflow_inventory()
        print(_json(inventory) if args.json else _workflow_inventory_text(inventory))
        return 0
    if args.command == "tools":
        inventory = build_tool_inventory()
        if args.tools_command == "doctor":
            print(doctor_report(inventory, runtime_profile_path=getattr(args, "runtime_profile", None)))
            return 0
        if args.tools_command == "list":
            if args.json:
                print(_json(inventory.as_dict()))
            else:
                for tool in inventory.as_dict()["tools"]:
                    if tool["effective"]:
                        print(f"{tool['status']}\t{tool['namespace']}.{tool['name']}\t{tool['source']}")
            return 0
        parser.error("unknown tools command")
    if args.command == "chat":
        from tas_copilot_core.chat_shell import run_chat_repl

        session_defaults = {
            "corpus_catalog": args.corpus_catalog,
            "data_catalog": args.data_catalog,
            "data_root": args.data_root,
            "precedent_index": args.precedent_index,
            "paper_id": args.paper_id,
            "figure_id": args.figure_id,
            "fixture": args.fixture,
        }
        return run_chat_repl(
            args.output_dir,
            model=args.model,
            ollama_url=args.ollama_url,
            narrate=args.narrate,
            session_defaults=session_defaults,
            timeout=args.timeout,
        )
    if args.command == "mcp-serve":
        import asyncio

        from tas_copilot_core.mcp_gateway import run_stdio

        try:
            asyncio.run(run_stdio(args.local_runs_dir, surface=args.surface))
        except (RuntimeError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                return 0
            parser.exit(2, f"mcp-serve unavailable: {exc}\n")
        return 0
    if args.command == "serve":
        label = "alignment dashboard" if args.app == "alignment" else "local workbench"
        print(f"Starting TASCopilot {label} at http://{args.host}:{args.port}/")
        print("Local-only bind; no live-control endpoints are exposed. Public deployment goes through a reverse proxy.")
        try:
            run_server(host=args.host, port=args.port, local_runs_dir=args.local_runs_dir, docs_dir=args.docs_dir, app=args.app, root_path=args.root_path)
        except (RuntimeError, ValueError) as exc:
            parser.exit(2, f"serve unavailable: {exc}\n")
        return 0
    if args.command == "review":
        if args.review_command == "labels":
            if args.review_labels_command == "summary":
                print(_json(review_label_summary({"store": args.store, "output_dir": args.output_dir, "sample_limit": args.sample_limit})))
                return 0
            if args.review_labels_command == "export":
                print(
                    _json(
                        export_review_labels(
                            {
                                "store": args.store,
                                "output": args.output,
                                "object_type": args.object_type,
                                "object_id": args.object_id,
                                "verdict": args.verdict,
                                "source_workflow": args.source_workflow,
                                "limit": args.limit,
                            }
                        )
                    )
                )
                return 0
            if args.review_labels_command == "ingest":
                print(_json(ingest_review_labels({"labels": args.labels, "store": args.store, "output_dir": args.output_dir})))
                return 0
            if args.review_labels_command == "backfill":
                print(_json(backfill_review_labels({"root": args.root, "store": args.store, "output_dir": args.output_dir})))
                return 0
            if args.review_labels_command == "stale-check":
                print(
                    _json(
                        stale_review_label_check(
                            {
                                "store": args.store,
                                "output_dir": args.output_dir,
                                "object_type": args.object_type,
                                "object_id": args.object_id,
                                "verdict": args.verdict,
                                "source_workflow": args.source_workflow,
                                "workflow_version": args.workflow_version,
                                "calibration_state_hash": args.calibration_state_hash,
                                "dag_run_id": args.dag_run_id,
                                "cache_key": args.cache_key,
                                "tool_versions": _parse_tool_versions(args.tool_versions or []),
                                "limit": args.limit,
                            }
                        )
                    )
                )
                return 0
            if args.review_labels_command == "packet":
                print(
                    _json(
                        prepare_review_label_packet(
                            {
                                "store": args.store,
                                "output_dir": args.output_dir,
                                "object_type": args.object_type,
                                "source_workflow": args.source_workflow,
                                "verdict": args.verdict,
                                "limit": args.limit,
                            }
                        )
                    )
                )
                return 0
        parser.error("unknown review command")
    if args.command == "ontology":
        config = CoreConfig()
        vocabularies = load_external_vocabularies(config.external_vocabularies_path)
        concepts = load_ontology(config.ontology_path)
        mappings = load_external_mappings(config.external_mappings_path)
        if args.ontology_command == "vocabularies":
            if args.ontology_vocabularies_command == "list":
                if args.json:
                    print(_json({"external_vocabularies": vocabularies}))
                else:
                    for vocabulary in vocabularies:
                        print(f"{vocabulary.id}\t{vocabulary.label}\t{vocabulary.retrieval_policy}\t{vocabulary.import_policy}")
                return 0
        if args.ontology_command == "mappings":
            if args.ontology_mappings_command == "validate":
                report = validate_ontology_alignment(vocabularies, concepts, mappings).as_dict()
                print(_json(report))
                return 0 if report["status"] != "error" else 1
            if args.ontology_mappings_command == "report":
                print(format_alignment_report(ontology_alignment_report(vocabularies, concepts, mappings)))
                return 0
        if args.ontology_command == "concept":
            if args.ontology_concept_command == "show":
                aligned = attach_external_mappings(concepts, mappings)
                concept_id = args.concept_id.removeprefix("tas:")
                concept = next((item for item in aligned if item.id == concept_id), None)
                if concept is None:
                    parser.error(f"unknown concept id: {args.concept_id}")
                mapping_set = mappings_for_concept(concept_id, mappings)
                print(_json({"concept": concept, "external_alignment": mapping_set}))
                return 0
        if args.ontology_command == "export":
            print(_json(export_alignment_json(vocabularies, concepts, mappings)))
            return 0
        if args.ontology_command == "diagnostics":
            aligned = attach_external_mappings(concepts, mappings)
            report = _ontology_decision_diagnostics(aligned)
            print(_json(report) if args.json else _ontology_decision_diagnostics_text(report))
            return 0
        parser.error("unknown ontology command")
    if args.command == "corpus":
        if args.corpus_command == "load-fixtures":
            print(_json(load_corpus_fixtures(args.fixtures, args.catalog)))
            return 0
        if args.corpus_command == "report":
            print(_json(corpus_pilot_report(args.catalog, output_dir=args.output_dir)))
            return 0
        if args.corpus_command == "paper-shortlist":
            print(_json(paper_candidate_shortlist({"catalog": args.catalog, "output_dir": args.output_dir, "quality_report": args.quality_report, "top_k": args.top_k})))
            return 0
        if args.corpus_command == "extract-paper-constraints":
            print(_json(extract_paper_constraints({"catalog": args.catalog, "output_dir": args.output_dir, "paper_id": args.paper_id})))
            return 0
        if args.corpus_command == "extract-paper-facts":
            payload = {
                "catalog": args.catalog,
                "output_dir": args.output_dir,
                "paper_id": args.paper_id,
                "provider": args.provider,
                "model_output": args.model_output,
                "ollama_url": args.ollama_url,
                "vllm_url": args.vllm_url,
                "api_key": args.api_key,
                "model": args.model,
                "max_chars_per_paper": args.max_chars_per_paper,
            }
            try:
                payload = apply_llm_endpoint(payload, args.llm_endpoint)
            except ValueError as exc:
                print(_json({"status": "missing_input", "errors": [{"error": str(exc)}]}))
                return 0
            print(_json(extract_paper_facts(payload)))
            return 0
        if args.corpus_command == "paper-fact-review":
            print(
                _json(
                    paper_fact_review(
                        {
                            "catalog": args.catalog,
                            "output_dir": args.output_dir,
                            "paper_id": args.paper_id,
                            "paper_ids_file": args.paper_ids_file,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "srmn2sb2-agentic-rag-catalog":
            print(
                _json(
                    build_srmn2sb2_agentic_rag_catalog(
                        {
                            "catalog": args.catalog,
                            "output_dir": args.output_dir,
                            "extractor_run_dir": args.extractor_run_dir,
                            "source_markdowns": args.source_markdowns,
                            "max_chars_per_chunk": args.max_chars_per_chunk,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "srmn2sb2-agentic-rag-report":
            print(
                _json(
                    srmn2sb2_agentic_rag_report(
                        {
                            "catalog": args.catalog,
                            "output_dir": args.output_dir,
                            "sr_case_note": args.sr_case_note,
                            "sr_fit_report": args.sr_fit_report,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "srmn2sb2-agentic-rag-loop":
            payload = {
                "catalog": args.catalog,
                "output_dir": args.output_dir,
                "provider": args.provider,
                "model": args.model,
                "ollama_url": args.ollama_url,
                "vllm_url": args.vllm_url,
                "api_key": args.api_key,
                "top_k": args.top_k,
                "max_targets": args.max_targets,
                "max_followups": args.max_followups,
                "max_chars_per_target": args.max_chars_per_target,
            }
            try:
                payload = apply_llm_endpoint(payload, args.llm_endpoint)
            except ValueError as exc:
                print(_json({"status": "missing_input", "errors": [{"error": str(exc)}]}))
                return 0
            print(_json(run_srmn2sb2_agentic_rag_loop(payload)))
            return 0
        if args.corpus_command == "evidence-graph-propose":
            payload = {
                "catalog": args.catalog,
                "output_dir": args.output_dir,
                "provider": args.provider,
                "model": args.model,
                "ollama_url": args.ollama_url,
                "vllm_url": args.vllm_url,
                "api_key": args.api_key,
                "case_target": args.case_target,
                "case_target_label": args.case_target_label,
                "max_pairs": args.max_pairs,
                "max_chunk_chars": args.max_chunk_chars,
            }
            try:
                payload = apply_llm_endpoint(payload, args.llm_endpoint)
            except ValueError as exc:
                print(_json({"status": "missing_input", "errors": [{"error": str(exc)}]}))
                return 0
            print(_json(run_evidence_graph_proposer(payload)))
            return 0
        if args.corpus_command == "evidence-graph-apply-labels":
            print(
                _json(
                    apply_evidence_graph_labels(
                        {
                            "catalog": args.catalog,
                            "labels": args.labels,
                            "output_dir": args.output_dir,
                            "reviewer": args.reviewer,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "catalog-integrity":
            from tas_copilot_core.corpus.catalog_integrity import apply_catalog_quarantine, catalog_integrity_report

            report = catalog_integrity_report(args.catalog)
            if args.apply:
                print(_json(apply_catalog_quarantine(args.catalog, report=report)))
            else:
                print(_json({k: v for k, v in report.items() if k != "documents"} | {"content_mismatch": [report["documents"][p]["title"] for p in report["quarantine_paper_ids"]]}))
            return 0
        if args.corpus_command == "pdf-integrity":
            import subprocess

            from tas_copilot_core.corpus.catalog_integrity import pdf_integrity_report

            first_pages = None
            if args.first_page:
                first_pages = {}
                for pdf in Path(args.pdf_root).expanduser().rglob("*.pdf"):
                    try:
                        first_pages[pdf.name] = subprocess.run(["pdftotext", "-l", "1", str(pdf), "-"], capture_output=True, text=True, timeout=60).stdout
                    except Exception:
                        continue
            report = pdf_integrity_report(args.pdf_root, manifest_csv=args.manifest, first_page_text=first_pages)
            if args.output:
                Path(args.output).expanduser().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(_json({k: v for k, v in report.items() if k != "bad_downloads"}))
            return 0
        if args.corpus_command == "revalidate-paper-facts":
            from tas_copilot_core.corpus.paper_facts import revalidate_paper_facts

            print(_json(revalidate_paper_facts(args.catalog, in_place=args.in_place, fact_file=args.fact_file, output_name=args.output_name)))
            return 0
        if args.corpus_command == "judge-paper-facts":
            from tas_copilot_core.corpus.fact_judge import apply_judgments, judge_paper_facts

            report = judge_paper_facts({"catalog": args.catalog, "fact_file": args.fact_file, "output_dir": args.output_dir, "provider": args.provider,
                                        "model": args.model, "base_url": args.base_url, "api_key": args.api_key, "fact_ids_from": args.fact_ids_from, "limit": args.limit})
            if args.apply and report.get("status") == "ok":
                report["applied"] = apply_judgments(args.catalog, report["judgments"], fact_file=args.fact_file)
            print(_json(report))
            return 0
        if args.corpus_command == "score-fact-labels":
            from tas_copilot_core.corpus.fact_label_scoring import score_fact_labels

            print(_json(score_fact_labels({"catalog": args.catalog, "labels": args.labels, "fact_file": args.fact_file, "output_dir": args.output_dir, "include_quarantined": args.include_quarantined, "judgments": args.judgments})))
            return 0
        if args.corpus_command == "fact-eval-dashboard":
            print(_json(build_fact_eval_dashboard({"catalog": args.catalog, "output_dir": args.output_dir, "prior_labels": args.prior_labels or [], "resume_labels": args.resume_labels or [], "storage_key": args.storage_key, "fact_file": args.fact_file, "group_by_paper": args.group_by_paper, "pdf_dir": args.pdf_dir, "pdf_paths": args.pdf_paths, "only_second_looks": args.only_second_looks, "only_ids": args.only_ids, "no_edges": args.no_edges, "page_name": args.page_name, "second_look_from": args.second_look_from or []})))
            return 0
        if args.corpus_command == "paper-link-review-dashboard":
            print(
                _json(
                    build_paper_experiment_review_dashboard(
                        {
                            "review_template": args.review_template,
                            "review_assist": args.review_assist,
                            "output_dir": args.output_dir,
                            "html_filename": args.html_filename,
                            "max_packet_scans": args.max_packet_scans,
                            "paper_base_url": args.paper_base_url,
                            "review_scan_records": args.review_scan_records,
                            "data_catalog": args.data_catalog,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "ingest-paper-link-reviews":
            print(_json(ingest_paper_experiment_review_labels({"labels": args.labels, "output_dir": args.output_dir, "min_each": args.min_each})))
            return 0
        if args.corpus_command == "ingest-pilot":
            print(_json(ingest_pilot_corpus({"manifest": args.manifest, "catalog": args.catalog, "output_dir": args.output_dir})))
            return 0
        if args.corpus_command == "ingest-extractor-run":
            print(
                _json(
                    ingest_extractor_run(
                        {
                            "extractor_run_dir": args.extractor_run_dir,
                            "catalog": args.catalog,
                            "output_dir": args.output_dir,
                            "preferred_backend": args.preferred_backend,
                            "max_chars_per_document": args.max_chars_per_document,
                            "append": args.append,
                        }
                    )
                )
            )
            return 0
        if args.corpus_command == "search":
            result = corpus_search(
                {
                    "catalog": args.catalog,
                    "query": args.query,
                    "top_k": args.top_k,
                    "paper_id": args.paper_id,
                    "document_type": args.document_type,
                    "chunk_type": args.chunk_type,
                    "figure_id": args.figure_id,
                    "panel_id": args.panel_id,
                    "tas_concept": args.tas_concept,
                    "year": args.year,
                    "instrument": args.instrument,
                    "material_family": args.material_family,
                    "requires_source_support": args.requires_source_support,
                    "exclude_references": args.exclude_references,
                    "use_dense": args.use_dense,
                }
            )
            print(_json(result) if args.json else result)
            return 0
        if args.corpus_command == "ontology-diagnostics":
            print(_json(ontology_diagnostics({"catalog": args.catalog, "output_dir": args.output_dir})))
            return 0
        if args.corpus_command == "agent-readiness":
            print(_json(agent_readiness({"corpus_catalog": args.corpus_catalog, "data_catalog": args.data_catalog, "output_dir": args.output_dir})))
            return 0
        if args.corpus_command == "compile-figure-query":
            print(
                _json(
                    compile_figure_query(
                        args.corpus_catalog,
                        paper_id=args.paper_id,
                        figure_id=args.figure_id,
                        panel_id=args.panel_id,
                        output=args.output,
                        user_sample=args.user_sample,
                    )
                )
            )
            return 0
        if args.corpus_command == "validate-source-support":
            result = validate_source_support(args.catalog)
            if args.output_dir:
                result["artifacts"] = write_report_bundle(
                    args.output_dir,
                    result,
                    markdown_renderer=_source_support_markdown,
                    json_filename="source_support_validation.json",
                    markdown_filename="source_support_validation.md",
                )
            print(_json(result) if args.json else result)
            return 0
        if args.corpus_command == "extract-document":
            result = extract_document(
                {
                    "path": args.path,
                    "backend": args.backend,
                    "max_pages": args.max_pages,
                    "max_chars": args.max_chars,
                    "grobid_url": args.grobid_url,
                    "output_dir": args.output_dir,
                    "write_artifacts": args.write_artifacts,
                }
            )
            print(_json(result) if args.json else result)
            return 0
        if args.corpus_command == "extractor-bakeoff":
            result = extractor_bakeoff(
                {
                    "manifest": args.manifest,
                    "output_dir": args.output_dir,
                    "limit": args.limit,
                    "backends": args.backends,
                    "max_pages": args.max_pages,
                    "max_chars": args.max_chars,
                    "grobid_url": args.grobid_url,
                }
            )
            print(_json(result))
            return 0
        if args.corpus_command == "extractor-quality":
            print(_json(extractor_quality_report({"extractor_run_dir": args.extractor_run_dir, "output_dir": args.output_dir})))
            return 0
        parser.error("unknown corpus command")
    if args.command == "eval":
        if args.eval_command == "run":
            payload = json.loads(args.json)
            payload.update({"suite": args.suite, "output_dir": args.output_dir})
            suite_name = Path(args.suite).name
            if "copilot_router" in suite_name or "copilot_route" in suite_name:
                runner = run_copilot_route_quality_eval
            elif "corpus_pilot" in suite_name:
                runner = run_corpus_pilot_eval
            else:
                runner = run_lu_femn_o3_regression
            print(_json(runner(payload)))
            return 0
        parser.error("unknown eval command")
    if args.command == "material":
        if args.material_command == "parse":
            result = material_identity_record(args.text, family=args.family).model_dump(mode="json")
            print(_json(result) if args.json else result)
            return 0
        if args.material_command == "compare":
            result = compare_material_identity(args.left, args.right, family=args.family)
            print(_json(result) if args.json else result)
            return 0
        if args.material_command == "doctor":
            print(_json(material_identity_backend_status()))
            return 0
        parser.error("unknown material command")
    if args.command == "instrument-calibrations":
        if args.instrument_calibrations_command == "validate":
            result = validate_instrument_calibrations_registry(args.registry)
            print(_json(result))
            return 0 if result["status"] == "ok" else 1
        if args.instrument_calibrations_command == "list":
            result = instrument_calibrations_report(args.registry)
            if args.json:
                print(_json(result))
            else:
                print(f"Instrument calibration registry: {result['registry_path']}")
                print(f"Records: {result['record_count']} | accepted: {result['accepted_count']}")
                print(f"Status counts: {result['status_counts']}")
                for record in result["records"]:
                    print(f"- {record['id']} | kind={record['kind']} | instrument={record['instrument']} | status={record['status']} | reviewed_by={record['reviewed_by']}")
                print(result["safe_conclusion"])
            return 0 if result["status"] == "ok" else 1
        parser.error("unknown instrument-calibrations command")
    if args.command == "psd-calibration":
        if args.psd_calibration_command == "validate":
            result = validate_psd_calibration_registry(args.registry)
            print(_json(result))
            return 0 if result["status"] != "error" else 1
        if args.psd_calibration_command == "report":
            result = psd_calibration_report(args.registry)
            if args.json:
                print(_json(result))
            else:
                print(f"PSD calibration registry: {result['registry_path']}")
                print(f"Records: {result['record_count']} | accepted: {result['accepted_count']}")
                print(f"Status counts: {result['status_counts']}")
                print(result["safe_conclusion"])
            return 0
        if args.psd_calibration_command == "discover-dat":
            print(_json(discover_psd_dat_calibrations(args.calibration_dir, path_only=args.path_only)))
            return 0
        if args.psd_calibration_command == "match":
            print(_json(match_psd_calibration(args.calibration_dir, args.data_file, path_only=args.path_only)))
            return 0
        if args.psd_calibration_command == "compare":
            print(
                _json(
                    compare_psd_calibration_matches(
                        args.calibration_dir,
                        args.data_file,
                        top_k=args.top_k,
                        output_dir=args.output_dir,
                        path_only=args.path_only,
                    )
                )
            )
            return 0
        if args.psd_calibration_command == "derive-findpeak":
            print(_json(derive_psd_calibration_workflow({"findpeak_file": args.findpeak_file, "output_dir": args.output_dir, "calibration_id": args.calibration_id, "stem": args.stem,
                                                          "nominal_a4": args.nominal_a4, "analyzer_detector_translation": args.analyzer_detector_translation, "notes": args.notes})))
            return 0
        parser.error("unknown psd-calibration command")
    if args.command == "workflow":
        if args.workflow_command == "inspect-bt7-directory":
            payload = json.loads(args.json)
            payload["path"] = args.path
            result = inspect_bt7_directory(payload)
        elif args.workflow_command == "run-spinwave-fixture":
            result = run_spinwave_fixture(json.loads(args.json))
        elif args.workflow_command == "check-geometry":
            payload = json.loads(args.json)
            if args.scan_id:
                payload["scan_id"] = args.scan_id
                payload.setdefault("db_path", args.db)
            result = check_geometry(payload)
        elif args.workflow_command == "compare-paper-to-data":
            result = compare_paper_to_data(args.paper_id, json.loads(args.json))
        elif args.workflow_command == "reduce-scan-set":
            result = reduce_scan_set(json.loads(args.json))
        elif args.workflow_command == "retrieve-similar-scans":
            result = retrieve_similar_scans(args.scan_id, args.limit, catalog=args.catalog)
        elif args.workflow_command == "real-data-smoke":
            payload = json.loads(args.json)
            payload.update({"path": args.path, "catalog": args.catalog, "output_dir": args.output_dir})
            result = real_data_smoke(payload)
        elif args.workflow_command == "catalog-report":
            payload = json.loads(args.json)
            payload.update({"catalog": args.catalog, "query": args.query, "limit": args.limit})
            if args.output_dir:
                payload["output_dir"] = args.output_dir
            result = catalog_report(payload)
        elif args.workflow_command == "scan-brief":
            payload = json.loads(args.json)
            payload.update({"catalog": args.catalog, "scan_id": args.scan_id, "limit": args.limit})
            if args.output_dir:
                payload["output_dir"] = args.output_dir
            result = scan_brief(payload)
        elif args.workflow_command == "experiment-brief":
            payload = json.loads(args.json)
            payload.update({"catalog": args.catalog, "experiment_id": args.experiment_id, "limit": args.limit})
            if args.output_dir:
                payload["output_dir"] = args.output_dir
            result = experiment_brief(payload)
        elif args.workflow_command == "find-figure-source-data":
            payload = json.loads(args.json)
            for key in [
                "catalog",
                "data_root",
                "output_dir",
                "query_file",
                "figure_image",
                "instrument",
                "mode",
                "sample_type",
                "material_family",
                "sample",
                "target_panel",
                "target_panel_xmn",
                "alternate_panel",
                "alternate_panel_xmn",
                "date",
                "experimenter",
                "q_min",
                "q_max",
                "temperature_min",
                "temperature_max",
                "annotated_q",
                "top_k",
                "max_reconstruction_candidates",
            ]:
                value = getattr(args, key, None)
                if value is not None:
                    payload[key] = value
            if args.attempt_reconstruction:
                payload["attempt_reconstruction"] = True
            result = find_figure_source_data(payload)
        elif args.workflow_command == "assess-figure-feature":
            payload = json.loads(args.json)
            for key in ["data_root", "output_dir", "patterns", "include_terms", "q_features", "temperature_features", "q_tolerance", "sideband_width", "sideband_gap", "temperature_tolerance"]:
                value = getattr(args, key, None)
                if value is not None:
                    payload[key] = value
            result = assess_figure_feature(payload)
        elif args.workflow_command == "figure-source-triage":
            payload = json.loads(args.json)
            payload.update({"catalog": args.catalog, "query_file": args.query_file, "output_dir": args.output_dir})
            result = figure_source_triage(payload)
        elif args.workflow_command == "source-triage-doctor":
            result = source_triage_doctor({})
        elif args.workflow_command == "lu-femn-o3-regression":
            payload = json.loads(args.json)
            payload.update({"suite": args.suite, "output_dir": args.output_dir})
            result = run_lu_femn_o3_regression(payload)
        elif args.workflow_command == "link-figure-panel-to-source":
            payload = json.loads(args.json)
            payload.update(
                {
                    "corpus_catalog": args.corpus_catalog,
                    "data_catalog": args.data_catalog,
                    "data_root": args.data_root,
                    "precedent_index": args.precedent_index,
                    "paper_id": args.paper_id,
                    "figure_id": args.figure_id,
                    "panel_id": args.panel_id,
                    "user_sample": args.user_sample,
                    "output_dir": args.output_dir,
                    "query_file": args.query_file,
                    "top_k": args.top_k,
                }
            )
            result = link_figure_panel_to_source(payload)
        elif args.workflow_command == "paper-brief":
            result = paper_brief({"corpus_catalog": args.corpus_catalog, "paper_id": args.paper_id, "output_dir": args.output_dir})
        elif args.workflow_command == "figure-brief":
            result = figure_brief({"corpus_catalog": args.corpus_catalog, "figure_id": args.figure_id, "output_dir": args.output_dir})
        elif args.workflow_command == "link-paper-to-experiments":
            result = link_paper_to_experiments(
                {
                    "corpus_catalog": args.corpus_catalog,
                    "data_catalog": args.data_catalog,
                    "paper_id": args.paper_id,
                    "output_dir": args.output_dir,
                    "top_k": args.top_k,
                    "max_data_candidates": args.max_data_candidates,
                    "hydrate_candidate_clues": args.hydrate_candidate_clues,
                    "hydrate_candidate_limit": args.hydrate_candidate_limit,
                    "max_hydrated_files_per_candidate": args.max_hydrated_files_per_candidate,
                    "hydrated_index": args.hydrated_index,
                    "review_label_store": args.review_label_store,
                }
            )
        elif args.workflow_command == "proposal-evidence-ingest":
            payload = json.loads(args.json)
            payload.update(
                {
                    "proposal_documents": args.proposal_documents or [],
                    "actual_documents": args.actual_documents or [],
                    "output_dir": args.output_dir,
                    "max_pages": args.max_pages,
                    "max_chars": args.max_chars,
                }
            )
            result = ingest_proposal_evidence(payload)
        elif args.workflow_command == "proposal-planning-brief":
            payload = json.loads(args.json)
            payload.update(
                {
                    "fixture": args.fixture,
                    "evidence_dir": args.evidence_dir,
                    "proposal_evidence_dir": args.proposal_evidence_dir,
                    "output_dir": args.output_dir,
                    "proposal_id": args.proposal_id,
                }
            )
            result = proposal_planning_brief(payload)
        elif args.workflow_command == "proposal-retrospective":
            payload = json.loads(args.json)
            payload.update(
                {
                    "fixture": args.fixture,
                    "evidence_dir": args.evidence_dir,
                    "proposal_evidence_dir": args.proposal_evidence_dir,
                    "actual_evidence_dir": args.actual_evidence_dir,
                    "output_dir": args.output_dir,
                    "proposal_id": args.proposal_id,
                    "experiment_id": args.experiment_id,
                }
            )
            result = proposal_experiment_retrospective(payload)
        elif args.workflow_command == "suggest-experiment-scans":
            payload = json.loads(args.json)
            payload.update(
                {
                    "fixture": args.fixture,
                    "evidence_dir": args.evidence_dir,
                    "proposal_evidence_dir": args.proposal_evidence_dir,
                    "actual_evidence_dir": args.actual_evidence_dir,
                    "output_dir": args.output_dir,
                    "proposal_id": args.proposal_id,
                }
            )
            result = suggest_experiment_scans(payload)
        elif args.workflow_command == "dispersion-scan-plan":
            payload = json.loads(args.json)
            payload.update({"data_roots": args.data_roots, "output_dir": args.output_dir, "collimation": args.collimation, "e_min_mev": args.e_min_mev, "e_max_mev": args.e_max_mev})
            from tas_copilot_core.workflows.scan_planning import dispersion_scan_plan

            result = dispersion_scan_plan(payload)
        elif args.workflow_command == "dispersion-map-report":
            payload = json.loads(args.json)
            payload.update({"data_roots": args.data_roots, "output_dir": args.output_dir, "max_files": args.max_files, "patterns": args.patterns})
            from tas_copilot_core.workflows.experiment_story import dispersion_map_report

            result = dispersion_map_report(payload)
        elif args.workflow_command == "experiment-story-report":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_roots": args.data_roots,
                    "output_dir": args.output_dir,
                    "sections": args.sections,
                    "alignment_state_json": args.alignment_state_json,
                    "alignment_peak_report": args.alignment_peak_report,
                    "max_files": args.max_files,
                    "patterns": args.patterns,
                }
            )
            from tas_copilot_core.workflows.experiment_story import experiment_story_report

            result = experiment_story_report(payload)
        elif args.workflow_command == "experiment-day-index":
            payload = json.loads(args.json)
            payload.update({"reports_root": args.reports_root, "output_dir": args.output_dir or args.reports_root, "title": args.title})
            from tas_copilot_core.workflows.experiment_dashboard import experiment_day_index

            result = experiment_day_index(payload)
        elif args.workflow_command == "experiment-dashboard":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "output_dir": args.output_dir,
                    "title": args.title,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "spurion_top_k": args.spurion_top_k,
                    "ranking_mode": args.ranking_mode,
                    "patterns": args.patterns,
                    "overlay_materials": args.overlay_materials,
                    "overlay_config": args.overlay_config,
                    "experiment_config": args.experiment_config,
                    "alignment_state_json": args.alignment_state_json,
                }
            )
            result = build_experiment_dashboard(payload)
        elif args.workflow_command == "experiment-review":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "output_dir": args.output_dir,
                    "title": args.title,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "top_k": args.top_k,
                    "plot_top_k": args.plot_top_k,
                    "max_scans_per_candidate": args.max_scans_per_candidate,
                    "allow_rebin": args.allow_rebin,
                    "mesh_grid_points": args.mesh_grid_points,
                    "experiment_id": args.experiment_id,
                    "hours": args.hours,
                    "since": args.since,
                    "patterns": args.patterns,
                    "experiment_config": args.experiment_config,
                    "corpus_catalog": args.corpus_catalog,
                    "corpus_query": args.corpus_query,
                    "precedent_index": args.precedent_index,
                    "review_label_store": args.review_label_store,
                    "skip_spurion": args.skip_spurion,
                }
            )
            result = experiment_review(payload)
        elif args.workflow_command == "survey-inelastic-spurion-evidence":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "spurion_top_k": args.spurion_top_k,
                    "ranking_mode": args.ranking_mode,
                    "patterns": args.patterns,
                    "experiment_config": args.experiment_config,
                    "review_label_store": args.review_label_store,
                }
            )
            result = survey_inelastic_spurion_evidence(payload)
        elif args.workflow_command == "live-experiment-page":
            result = live_experiment_page({"config": args.config, "offline_data_root": args.offline_data_root, "output_dir": args.output_dir, "instrument": args.instrument})
        elif args.workflow_command == "psd-diffraction-pattern":
            result = psd_diffraction_pattern_workflow({"data_file": args.data_file, "calibration": args.calibration, "output_dir": args.output_dir, "bin_width": args.bin_width,
                                                       "sign_convention": args.sign_convention, "scale": args.scale, "masked_channels": args.masked_channels,
                                                       "monitor_reference": args.monitor_reference, "check": not args.no_check})
        elif args.workflow_command == "overnight-scan-summary":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "alignment_state_json": args.alignment_state_json,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "hours": args.hours,
                    "since": args.since,
                    "patterns": args.patterns,
                    "experiment_config": args.experiment_config,
                    "precedent_index": args.precedent_index,
                    "top_k": args.top_k,
                }
            )
            result = overnight_scan_summary(payload)
        elif args.workflow_command == "draft-overnight-report":
            payload = json.loads(args.json)
            payload.update(
                {
                    "output_dir": args.output_dir,
                    "summary_json": args.summary_json,
                    "data_root": args.data_root,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "hours": args.hours,
                    "since": args.since,
                    "patterns": args.patterns,
                    "experiment_config": args.experiment_config,
                    "precedent_index": args.precedent_index,
                }
            )
            result = draft_overnight_report(payload)
        elif args.workflow_command == "inelastic-series-report":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "output_dir": args.output_dir,
                    "series_id": args.series_id,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "hours": args.hours,
                    "since": args.since,
                    "patterns": args.patterns,
                    "experiment_config": args.experiment_config,
                }
            )
            result = inelastic_series_report(payload)
        elif args.workflow_command == "order-parameter-triage":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_root": args.data_root,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "hours": args.hours,
                    "since": args.since,
                    "patterns": args.patterns,
                    "min_scans": args.min_scans,
                    "top_k": args.top_k,
                    "review_label_store": args.review_label_store,
                }
            )
            result = order_parameter_triage(payload)
        elif args.workflow_command == "fit-order-parameter":
            payload = json.loads(args.json)
            payload.update(
                {
                    "triage_json": args.triage_json,
                    "candidate_id": args.candidate_id,
                    "output_dir": args.output_dir,
                    "observable": args.observable,
                    "fit_backend": args.fit_backend,
                    "panweb_path": args.panweb_path,
                    "model_type": args.model_type,
                    "background_model": args.background_model,
                    "review_status": args.review_status,
                    "reviewed_by": args.reviewed_by,
                    "accept_fit": args.accept_fit,
                    "op_function": args.op_function,
                    "tanh_width": args.tanh_width,
                    "use_tc_smearing": args.use_tc_smearing,
                    "fit_method": args.fit_method,
                    "parallel": args.parallel,
                    "fit_steps": args.fit_steps,
                    "fit_burn": args.fit_burn,
                    "initial_tc": args.initial_tc,
                    "initial_beta": args.initial_beta,
                    "compare_models": args.compare_models,
                    "model_grid": args.model_grid,
                    "normal_transition": not args.inverse_transition,
                }
            )
            result = fit_order_parameter(payload)
        elif args.workflow_command == "dag-run-replay":
            payload = json.loads(args.json)
            payload.update(
                {
                    "dag_run_json": args.dag_run_json,
                    "dag_run_id": args.dag_run_id,
                    "output_dir": args.output_dir,
                    "workflow_version": args.workflow_version,
                }
            )
            result = dag_run_replay(payload)
        elif args.workflow_command == "trajectory-cluster-order-parameters":
            payload = json.loads(args.json)
            payload.update(
                {
                    "input": args.input,
                    "triage_json": args.triage_json,
                    "output_dir": args.output_dir,
                    "id_column": args.id_column,
                    "parameter_column": args.parameter_column,
                    "value_column": args.value_column,
                    "observable": args.observable,
                    "backend": args.backend,
                    "n_clusters": args.n_clusters,
                    "max_grid_points": args.max_grid_points,
                }
            )
            result = cluster_order_parameter_trajectories(payload)
        elif args.workflow_command == "order-parameter-trajectory-report":
            payload = json.loads(args.json)
            payload.update(
                {
                    "triage_json": args.triage_json,
                    "trajectory_json": args.trajectory_json,
                    "output_dir": args.output_dir,
                    "top_k": args.top_k,
                }
            )
            result = order_parameter_trajectory_report(payload)
        elif args.workflow_command == "plot-order-parameter-candidates":
            payload = json.loads(args.json)
            payload.update(
                {
                    "report_json": args.report_json,
                    "output_dir": args.output_dir,
                    "top_k": args.top_k,
                    "max_scans_per_candidate": args.max_scans_per_candidate,
                    "experiment_id": args.experiment_id,
                    "allow_rebin": args.allow_rebin,
                    "mesh_grid_points": args.mesh_grid_points,
                }
            )
            result = plot_order_parameter_candidate_scans(payload)
        elif args.workflow_command == "build-bt7-hydrated-index":
            result = build_bt7_hydrated_index(
                {
                    "data_catalog": args.data_catalog,
                    "output": args.output,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "path_contains": args.path_contains,
                }
            )
        elif args.workflow_command == "classify-psd-sources":
            from tas_copilot_core.workflows.figure_source_triage import _load_records

            payload = json.loads(args.json)
            payload.setdefault("catalog", args.catalog)
            records = _load_records(payload)
            result = {"workflow_name": "classify-psd-sources", "classifications": [classify_source_family(record).model_dump(mode="json") for record in records]}
            if args.output_dir:
                result["artifacts"] = write_report_bundle(args.output_dir, result)
        elif args.workflow_command == "reconstruct-psd-map":
            from tas_copilot_core.workflows.figure_source_triage import _load_records

            payload = json.loads(args.json)
            payload.setdefault("catalog", args.catalog)
            records = _load_records(payload)
            groups = group_psd_sources(records)
            group = next((item for item in groups if item.group_id == args.group_id), None)
            if group is None:
                result = {"workflow_name": "reconstruct-psd-map", "status": "missing_input", "missing_fields": ["group_id"], "groups": [item.model_dump(mode="json") for item in groups]}
            elif args.level == "q_temperature":
                result = q_temperature_reconstruction_blocked(group.group_id, payload.get("calibration_registry")).model_dump(mode="json")
            else:
                result = reconstruct_channel_temperature(group, records, args.output_dir).model_dump(mode="json")
        elif args.workflow_command == "plot-psd-detector-cuts":
            from tas_copilot_core.workflows.figure_source_triage import _load_records

            payload = json.loads(args.json)
            payload.setdefault("catalog", args.catalog)
            records = _load_records(payload)
            hydration_results = []
            if args.hydrate:
                selected = select_detector_cut_records(records, scan_id=args.scan_id, group_id=args.group_id)
                records, hydration_results = hydrate_psd_records(selected, read_bt7_file)
            result = plot_psd_detector_cuts(
                records,
                args.output_dir,
                scan_id=args.scan_id,
                group_id=args.group_id,
                temperature_min=args.temperature_min,
                temperature_max=args.temperature_max,
            )
            if hydration_results:
                result["hydration_results"] = hydration_results
        elif args.workflow_command == "discover-psd-calibrations":
            payload = json.loads(args.json)
            result = discover_psd_calibration_candidates(
                payload.get("data_root", args.data_root),
                output_dir=payload.get("output_dir", args.output_dir),
                max_files=int(payload.get("max_files", args.max_files)),
            )
        elif args.workflow_command == "instrument-device-graph":
            payload = json.loads(args.json)
            payload.update({"nice_report": args.nice_report, "output_dir": args.output_dir, "alias_config": args.alias_config})
            result = instrument_graph_from_nice_report(payload)
        elif args.workflow_command == "nice-bt7-replay":
            payload = json.loads(args.json)
            payload.update(
                {
                    "replay_plan": args.replay_plan,
                    "output_dir": args.output_dir,
                    "recipe_id": args.recipe_id,
                    "executor": args.executor,
                    "mode": args.mode,
                    "host": args.host,
                    "port": args.port,
                    "client_name": args.client_name,
                    "file_prefix": args.file_prefix,
                    "preset_time": args.preset_time,
                    "max_points": args.max_points,
                    "python_executable": args.python_executable,
                    "container_image": args.container_image,
                    "nice_checkout": args.nice_checkout,
                    "docker_executable": args.docker_executable,
                    "timeout_seconds": args.timeout_seconds,
                    "allow_unreviewed_node_hints": args.allow_unreviewed_node_hints,
                }
            )
            payload["allow_live_host"] = bool(getattr(args, "allow_live_host", False))
            result = nice_bt7_replay(payload)
        elif args.workflow_command == "tasai-adaptive-nice-sim":
            payload = json.loads(args.json)
            payload.update(
                {
                    "output_dir": args.output_dir,
                    "runtime_profile": args.runtime_profile,
                    "candidate_points_file": args.candidate_points_file,
                    "model_spec_file": args.model_spec_file,
                    "software_limits_file": args.software_limits_file,
                    "human_suggestions": args.human_suggestions,
                    "policy_backend": args.policy_backend,
                    "count_backend": args.count_backend,
                    "tasai_policy_adapter": args.tasai_policy_adapter,
                    "andie_policy_adapter": args.andie_policy_adapter,
                    "andie_container_image": args.andie_container_image,
                    "andie_container_workspace": args.andie_container_workspace,
                    "andie_source_root": args.andie_source_root,
                    "pyspinw_count_adapter": args.pyspinw_count_adapter,
                    "tasai_path": args.tasai_path,
                    "analytic_model": args.analytic_model,
                    "analytic_parameters": json.loads(args.analytic_parameters) if args.analytic_parameters else None,
                    "nice_executor": args.nice_executor,
                    "max_steps": args.max_steps,
                    "count_time": args.count_time,
                    "host": args.host,
                    "port": args.port,
                    "client_name": args.client_name,
                    "container_image": args.container_image,
                    "nice_checkout": args.nice_checkout,
                    "docker_executable": args.docker_executable,
                    "timeout_seconds": args.timeout_seconds,
                    "suggestion_poll_seconds": args.suggestion_poll_seconds,
                    "step_delay_seconds": args.step_delay_seconds,
                    "prepare_hpc_handoff": args.prepare_hpc_handoff,
                    "hpc_action": args.hpc_action,
                    "hpc_system": args.hpc_system,
                    "hpc_work_dir": args.hpc_work_dir,
                    "hpc_account": args.hpc_account,
                    "hpc_partition": args.hpc_partition,
                    "hpc_container_runtime": args.hpc_container_runtime,
                    "andie_remote_image": args.andie_remote_image,
                    "andie_remote_workspace": args.andie_remote_workspace,
                    "andie_remote_source_root": args.andie_remote_source_root,
                    "hpc_login": args.hpc_login,
                    "hpc_remote_handoff_dir": args.hpc_remote_handoff_dir,
                    "hpc_remote_output_dir": args.hpc_remote_output_dir,
                    "hpc_job_id": args.hpc_job_id,
                    "hpc_ssh_executable": args.hpc_ssh_executable,
                    "hpc_scp_executable": args.hpc_scp_executable,
                    "hpc_submit_command": args.hpc_submit_command,
                    "hpc_poll_command": args.hpc_poll_command,
                    "controlpath": args.controlpath,
                    "allow_unreviewed_node_hints": args.allow_unreviewed_node_hints,
                }
            )
            payload["allow_live_host"] = bool(getattr(args, "allow_live_host", False))
            payload["python_executable"] = getattr(args, "python_executable", None)
            result = tasai_adaptive_nice_sim(payload)
        elif args.workflow_command == "suggest-alignment-peaks":
            payload = json.loads(args.json)
            payload.update(
                {
                    "formula": args.formula,
                    "icsd_db": args.icsd_db,
                    "scattering_plane": args.scattering_plane,
                    "lattice": args.lattice,
                    "space_group": args.space_group,
                    "space_group_number": args.space_group_number,
                    "space_group_setting": args.space_group_setting,
                    "gsasii_python": args.gsasii_python,
                    "gsasii_path": args.gsasii_path,
                    "gsasii_bin": args.gsasii_bin,
                    "gsasii_runtime": args.gsasii_runtime,
                    "energy_mev": args.energy_mev,
                    "a4_max_deg": args.a4_max_deg,
                    "top_k": args.top_k,
                    "candidate_limit": args.candidate_limit,
                    "max_reflections": args.max_reflections,
                    "background_material_config": args.background_material_config,
                    "background_material": args.background_material or payload.get("background_material"),
                    "background_overlap_tolerance_deg": args.background_overlap_tolerance_deg,
                    "max_background_reflections": args.max_background_reflections,
                    "sample_thickness_mm": args.sample_thickness_mm,
                    "out_of_plane_low_q_deg": args.out_of_plane_low_q_deg,
                    "out_of_plane_high_q_deg": args.out_of_plane_high_q_deg,
                    "out_of_plane_max_deg": args.out_of_plane_max_deg,
                    "output_dir": args.output_dir,
                }
            )
            result = suggest_alignment_peaks(payload)
        elif args.workflow_command == "alignment-corpus-index":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_roots": args.data_roots,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "max_log_lines": args.max_log_lines,
                    "sample_limit": args.sample_limit,
                    "alignment_peak_report": args.alignment_peak_report,
                    "include_logs": not args.no_logs,
                }
            )
            result = alignment_corpus_index(payload)
        elif args.workflow_command == "alignment-state-epochs":
            payload = json.loads(args.json)
            payload.update({"events_jsonl": args.events_jsonl, "alignment_peak_report": args.alignment_peak_report, "output_dir": args.output_dir})
            result = alignment_state_epochs(payload)
        elif args.workflow_command == "alignment-state-machine":
            payload = json.loads(args.json)
            payload.update(
                {
                    "events_jsonl": args.events_jsonl,
                    "alignment_peak_report": args.alignment_peak_report,
                    "max_alignment_peaks": args.max_alignment_peaks,
                    "peak_fit_backend": args.peak_fit_backend,
                    "panweb_path": args.panweb_path,
                    "panweb_use_dream": args.panweb_use_dream,
                    "panweb_parallel": args.panweb_parallel,
                    "background_material_config": args.background_material_config,
                    "background_material": args.background_material or payload.get("background_material"),
                    "background_peak_table": args.background_peak_table,
                    "background_tolerance_deg": args.background_tolerance_deg,
                    "instrument_calibrations": args.instrument_calibrations or payload.get("instrument_calibrations"),
                    "axis_policy": args.axis_policy or payload.get("axis_policy"),
                    "output_dir": args.output_dir,
                }
            )
            result = alignment_state_machine(payload)
        elif args.workflow_command == "bt7-corpus-candidate-pass":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_roots": args.data_roots,
                    "output_dir": args.output_dir,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "max_log_lines": args.max_log_lines,
                    "sample_limit": args.sample_limit,
                    "spurion_top_k": args.spurion_top_k,
                    "order_top_k": args.order_top_k,
                    "min_scans": args.min_scans,
                    "review_label_store": args.review_label_store,
                    "include_logs": not args.no_logs,
                }
            )
            result = bt7_corpus_candidate_pass(payload)
        elif args.workflow_command == "alignment-review-dashboard":
            payload = json.loads(args.json)
            payload.update(
                {
                    "review_template": args.review_template,
                    "output_dir": args.output_dir,
                    "html_filename": args.html_filename,
                    "max_dashboard_rows": args.max_dashboard_rows,
                }
            )
            result = build_alignment_review_dashboard(payload)
        elif args.workflow_command == "alignment-planning-brief":
            payload = json.loads(args.json)
            payload.update(
                {
                    "events_jsonl": args.events_jsonl,
                    "state_json": args.state_json,
                    "alignment_peak_report": args.alignment_peak_report,
                    "output_dir": args.output_dir,
                }
            )
            if args.ef_mev is not None:
                payload["ef_mev"] = args.ef_mev
            result = alignment_planning_brief(payload)
        elif args.workflow_command == "alignment-fine-replay-plan":
            payload = json.loads(args.json)
            payload.update(
                {
                    "state_json": args.state_json,
                    "brief_json": args.brief_json,
                    "rocking_motor": args.rocking_motor,
                    "count_time_seconds": args.count_time_seconds,
                    "axis_policy": args.axis_policy,
                    "output_dir": args.output_dir,
                }
            )
            result = alignment_fine_replay_plan(payload)
        elif args.workflow_command == "alignment-adaptive-nice-loop":
            payload = json.loads(args.json)
            payload.update(
                {
                    "events_jsonl": args.events_jsonl,
                    "state_json": args.state_json,
                    "rocking_motor": args.rocking_motor,
                    "count_time_seconds": args.count_time_seconds,
                    "max_iterations": args.max_iterations,
                    "max_points": args.max_points,
                    "axis_policy": args.axis_policy,
                    "executor": args.executor,
                    "execute_dry_run": args.execute_dry_run,
                    "allow_simulator_execution": args.allow_simulator_execution,
                    "host": args.host,
                    "port": args.port,
                    "client_name": args.client_name,
                    "container_image": args.container_image,
                    "nice_checkout": args.nice_checkout,
                    "docker_executable": args.docker_executable,
                    "timeout_seconds": args.timeout_seconds,
                    "allow_unreviewed_node_hints": args.allow_unreviewed_node_hints,
                    "output_dir": args.output_dir,
                }
            )
            result = alignment_adaptive_nice_loop(payload)
        elif args.workflow_command == "experiment-planning-brief":
            payload = json.loads(args.json)
            payload.update(
                {
                    "material": args.material,
                    "goal": args.goal,
                    "formula": args.formula,
                    "scattering_plane": args.scattering_plane,
                    "icsd_db": args.icsd_db,
                    "space_group": args.space_group,
                    "space_group_number": args.space_group_number,
                    "space_group_setting": args.space_group_setting,
                    "energy_mev": args.energy_mev,
                    "a4_max_deg": args.a4_max_deg,
                    "out_of_plane_max_deg": args.out_of_plane_max_deg,
                    "background_overlap_tolerance_deg": args.background_overlap_tolerance_deg,
                    "max_background_reflections": args.max_background_reflections,
                    "background_material": args.background_material or payload.get("background_material"),
                    "background_material_config": args.background_material_config,
                    "precedent_index": args.precedent_index,
                    "corpus_catalog": args.corpus_catalog,
                    "data_root": args.data_root,
                    "replay_report": args.replay_report,
                    "run_dashboard": args.run_dashboard,
                    "experiment_config": args.experiment_config,
                    "psd_calibration_registry": args.psd_calibration_registry,
                    "psd_calibration_dirs": args.psd_calibration_dirs or payload.get("psd_calibration_dirs", []),
                    "max_files": args.max_files,
                    "top_k": args.top_k,
                    "output_dir": args.output_dir,
                }
            )
            result = experiment_planning_brief(payload)
        elif args.workflow_command == "experiment-workspace":
            payload = json.loads(args.json)
            payload.update(
                {
                    "material": args.material,
                    "goal": args.goal,
                    "data_root": args.data_root,
                    "formula": args.formula,
                    "family": args.family,
                    "scattering_plane": args.scattering_plane,
                    "icsd_db": args.icsd_db,
                    "space_group": args.space_group,
                    "space_group_number": args.space_group_number,
                    "space_group_setting": args.space_group_setting,
                    "energy_mev": args.energy_mev,
                    "a4_max_deg": args.a4_max_deg,
                    "background_material": args.background_material or payload.get("background_material"),
                    "background_material_config": args.background_material_config,
                    "background_branch": args.background_branch or payload.get("background_branch"),
                    "precedent_index": args.precedent_index,
                    "corpus_catalog": args.corpus_catalog,
                    "replay_report": args.replay_report,
                    "experiment_config": args.experiment_config,
                    "psd_calibration_registry": args.psd_calibration_registry,
                    "psd_calibration_dirs": args.psd_calibration_dirs or payload.get("psd_calibration_dirs", []),
                    "max_files": args.max_files,
                    "top_k": args.top_k,
                    "output_dir": args.output_dir,
                }
            )
            result = experiment_workspace(payload)
        elif args.workflow_command == "psd-mask-sensitivity":
            payload = json.loads(args.json)
            payload.update(
                {
                    "data_files": args.data_files or payload.get("data_files", []),
                    "psd_calibration_dir": args.psd_calibration_dir,
                    "calibration_files": args.calibration_files or payload.get("calibration_files", []),
                    "annotated_q": args.annotated_q,
                    "output_dir": args.output_dir,
                    "path_only": args.path_only,
                }
            )
            result = psd_mask_sensitivity(payload)
        elif args.workflow_command == "review-figure-source-candidate":
            payload = json.loads(args.json)
            payload.update(
                {
                    "catalog": args.catalog,
                    "figure_report": args.figure_report,
                    "candidate_id": args.candidate_id,
                    "temperature_min": args.temperature_min,
                    "temperature_max": args.temperature_max,
                    "temperature_tolerance": args.temperature_tolerance,
                    "channels": args.channels or [],
                    "calibration_registry": args.calibration_registry,
                    "psd_calibration_dir": args.psd_calibration_dir,
                    "psd_calibration_files": args.psd_calibration_files or [],
                    "psd_calibration_path_only": args.psd_calibration_path_only,
                    "annotated_q": args.annotated_q,
                    "output_dir": args.output_dir,
                    "hydrate": args.hydrate,
                }
            )
            result = review_figure_source_candidate(payload)
        elif args.workflow_command == "compare-psd-detector-cuts":
            from tas_copilot_core.workflows.figure_source_triage import _load_records

            payload = json.loads(args.json)
            payload.setdefault("catalog", args.catalog)
            records = _load_records(payload)
            hydration_results = []
            if args.hydrate:
                selected = (
                    select_detector_cut_records(records, scan_id=args.left_scan_id, group_id=None)
                    + select_detector_cut_records(records, scan_id=args.right_scan_id, group_id=None)
                )
                records, hydration_results = hydrate_psd_records(selected, read_bt7_file)
            result = compare_psd_detector_cuts(
                records,
                args.output_dir,
                left_scan_id=args.left_scan_id,
                right_scan_id=args.right_scan_id,
                temperature_min=args.temperature_min,
                temperature_max=args.temperature_max,
                temperature_tolerance=args.temperature_tolerance,
            )
            if hydration_results:
                result["hydration_results"] = hydration_results
        elif args.workflow_command == "check-psd-channel-artifact":
            from tas_copilot_core.workflows.figure_source_triage import _load_records

            payload = json.loads(args.json)
            payload.setdefault("catalog", args.catalog)
            records = _load_records(payload)
            hydration_results = []
            if args.hydrate:
                selected = select_detector_cut_records(records, scan_id=args.scan_id, group_id=None)
                records, hydration_results = hydrate_psd_records(selected, read_bt7_file)
            result = check_psd_channel_artifact(
                records,
                args.output_dir,
                scan_id=args.scan_id,
                channels_of_interest=args.channels or [],
                temperature_min=args.temperature_min,
                temperature_max=args.temperature_max,
                neighbor_width=args.neighbor_width,
            )
            if hydration_results:
                result["hydration_results"] = hydration_results
        elif args.workflow_command == "copilot-ask":
            if args.list_intents or not args.question:
                print(_copilot_intent_examples_text())
                return 0
            if not args.output_dir:
                parser.error("workflow copilot-ask requires --output-dir unless --list-intents is used")
            payload = json.loads(args.json)
            payload.update(
                {
                    "question": args.question,
                    "output_dir": args.output_dir,
                    "corpus_catalog": args.corpus_catalog,
                    "data_catalog": args.data_catalog,
                    "data_root": args.data_root,
                    "precedent_index": args.precedent_index,
                    "paper_id": args.paper_id,
                    "figure_id": args.figure_id,
                    "proposal_evidence_dir": args.proposal_evidence_dir,
                    "actual_evidence_dir": args.actual_evidence_dir,
                    "fixture": args.fixture,
                    "experiment_config": args.experiment_config,
                    "max_files": args.max_files,
                    "post_year": args.post_year,
                    "hours": args.hours,
                    "since": args.since,
                    "top_k": args.top_k,
                }
            )
            result = copilot_ask(payload)
        else:
            parser.error("unknown workflow command")
    elif args.command == "catalog":
        if args.catalog_command == "completeness":
            result = metadata_completeness_report(args.catalog)
            if not args.json:
                result = _catalog_completeness_text(result)
        elif args.catalog_command == "eligible-scans":
            result = eligible_scans_report(args.catalog, purpose=args.purpose, limit=args.limit)
            if not args.json:
                result = _eligible_scans_text(result)
        elif args.catalog_command == "quality":
            result = data_catalog_quality_report(args.catalog)
            if args.output_dir:
                result["artifacts"] = write_report_bundle(
                    args.output_dir,
                    result,
                    markdown_renderer=markdown_quality_report,
                    json_filename="catalog_quality.json",
                    markdown_filename="catalog_quality.md",
                )
            if not args.json and not args.output_dir:
                result = markdown_quality_report(result)
        else:
            parser.error("unknown catalog command")
    elif args.command == "inspect-bt7-directory":
        payload = json.loads(args.json)
        payload["path"] = args.path
        result = inspect_bt7_directory(payload)
    elif args.command == "find-papers":
        result = find_papers(args.query, args.limit)
    elif args.command == "find-experiments":
        result = find_experiments(args.query, args.limit)
    elif args.command == "check-resolution":
        payload = json.loads(args.json)
        if args.output_dir:
            payload["output_dir"] = args.output_dir
        result = check_resolution(payload)
    elif args.command == "calculate-angles":
        result = calculate_angles(json.loads(args.json))
    elif args.command == "calculate-ub":
        result = calculate_ub(json.loads(args.json))
    elif args.command == "check-spurions":
        result = check_spurions(json.loads(args.json))
    elif args.command == "plan-spinwave":
        result = plan_spinwave(json.loads(args.json))
    elif args.command == "powder-order-parameter":
        result = powder_order_parameter(json.loads(args.json))
    elif args.command == "read-bt7-data":
        result = read_bt7_data(json.loads(args.json))
    elif args.command == "read-bt7-file":
        result = read_bt7_file(json.loads(args.json))
    elif args.command == "classify-tas-format":
        result = classify_tas_format(json.loads(args.json))
    elif args.command == "read-tas-metadata":
        result = read_tas_metadata(json.loads(args.json))
    elif args.command == "scan-tas-directory":
        result = scan_tas_directory(json.loads(args.json))
    elif args.command == "group-bt7-scans":
        result = group_bt7_scans(json.loads(args.json))
    elif args.command == "translate-ice-file":
        result = translate_ice_file(json.loads(args.json))
    elif args.command == "add-scans":
        result = add_scans(json.loads(args.json))
    elif args.command == "combine-files":
        result = combine_files(json.loads(args.json))
    elif args.command == "subtract-background":
        result = subtract_background(json.loads(args.json))
    elif args.command == "apply-corrections":
        result = apply_corrections(json.loads(args.json))
    elif args.command == "normalize-to-monitor":
        result = normalize_to_monitor(json.loads(args.json))
    elif args.command == "apply-detailed-balance":
        result = apply_detailed_balance(json.loads(args.json))
    elif args.command == "summarize-psd-facility":
        result = summarize_psd_facility(json.loads(args.json))
    elif args.command == "extract-setup":
        result = extract_setup(args.text)
    elif args.command == "compare-paper-to-data":
        result = compare_paper_to_data(args.paper_id, json.loads(args.json))
    elif args.command == "analyze-dataset":
        result = analyze_dataset(args.identifier)
    elif args.command == "retrieve-similar-scans":
        result = retrieve_similar_scans(args.scan_id, args.limit)
    elif args.command == "import-data-catalog":
        result = import_jsonl_catalog_dir(args.catalog_dir, args.db)
    elif args.command == "data-catalog-summary":
        result = SQLiteDataCatalog(args.db).summary()
    elif args.command == "data-catalog-quality":
        result = data_catalog_quality_report(args.db)
        if args.output_dir:
            result["artifacts"] = write_report_bundle(
                args.output_dir,
                result,
                markdown_renderer=markdown_quality_report,
                json_filename="catalog_quality.json",
                markdown_filename="catalog_quality.md",
            )
    elif args.command == "catalog-backfill-dates":
        from tas_copilot_core.workflows.catalog_date_backfill import backfill_catalog_dates

        print(_json(backfill_catalog_dates({
            "catalog_sqlite": args.catalog_sqlite,
            "path_prefix": args.path_prefix,
            "limit": args.limit,
            "archive_root": args.archive_root,
            "path_marker": args.path_marker,
        })))
    elif args.command == "catalog-bt7-directory":
        result = catalog_bt7_directory(json.loads(args.json))
    elif args.command == "catalog-tas-directory":
        payload = json.loads(args.json)
        payload["path"] = args.path
        result = catalog_tas_directory(payload)
    elif args.command == "catalog-tas-batch":
        result = catalog_tas_batch(args.manifest)
    elif args.command == "eval-smoke":
        result = eval_smoke(json.loads(args.json))
    else:
        parser.error("unknown command")
    if isinstance(result, dict):
        _attach_try_example(result, args)
    print(result if isinstance(result, str) else _json(result))
    return 0


def _attach_try_example(result: dict, args: argparse.Namespace) -> None:
    if result.get("status") != "missing_input":
        return
    if result.get("try") or result.get("try_this_command") or result.get("safe_next_command"):
        return
    command = getattr(args, "command", None)
    if command == "workflow" and getattr(args, "workflow_command", None):
        workflow = args.workflow_command
        result["try"] = f"python -m tas_copilot_core.cli workflow {workflow} --help"
        return
    if command:
        result["try"] = f"python -m tas_copilot_core.cli {command} --help"


def _catalog_completeness_text(report: dict) -> str:
    lines = [f"Catalog completeness: {report.get('status')}"]
    summary = report.get("summary", {})
    lines.append(f"Scans: {summary.get('scan_count', 0)}")
    lines.append(f"Geometry-eligible: {summary.get('eligible_for_geometry_check', 0)}")
    for scan in report.get("scans", [])[:20]:
        lines.append(f"- {scan['scan_id']}: score={scan['completeness_score']} missing={', '.join(scan['geometry_missing_fields'])}")
    return "\n".join(lines)


def _eligible_scans_text(report: dict) -> str:
    lines = [f"Eligible scans for {report.get('purpose')}: {report.get('status')}"]
    lines.append(f"Best candidate: {report.get('best_candidate')}")
    for scan_id in report.get("candidate_scan_ids", []):
        lines.append(f"- {scan_id}")
    if not report.get("candidate_scan_ids"):
        lines.append(f"Rejected: {len(report.get('rejected', []))}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
