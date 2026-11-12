from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

from tas_copilot_contracts import Document, Paper, ProvenanceRecord, SourceChunk
from tas_copilot_core.corpus.paper_facts import FACT_TYPES, ANSWER_STATUSES, normalize_paper_fact_response, _fact_value_text, _sanitize_agentic_fact_semantics
from tas_copilot_core.llm_router import ollama_chat, openai_compatible_chat
from tas_copilot_core.reporting import write_report_bundle


DEFAULT_CA_MN2SB2_SOURCES: list[str] = []
DEFAULT_SR_CASE_NOTE: str | None = None

AGENTIC_FACT_TARGETS = [
    {
        "target_id": "magnetic_structure",
        "fact_types": ["magnetic_structure", "propagation_vector", "spin_direction", "ordered_moment", "crystal_structure", "material", "sample"],
        "required_terms": ["magnetic structure", "ordered moment", "spin direction", "propagation vector", "space group"],
        "queries": ["magnetic structure ordered moment spin direction propagation vector"],
    },
    {
        "target_id": "hamiltonian",
        "fact_types": ["exchange_interaction", "source_data_clue", "measurement_kind", "instrument_configuration"],
        "required_terms": ["Hamiltonian", "Heisenberg", "J1", "J2", "exchange", "anisotropy"],
        "queries": ["Hamiltonian Heisenberg J1 J2 exchange anisotropy spin waves"],
    },
    {
        "target_id": "instrument_setup",
        "fact_types": ["instrument", "facility", "energy_setting", "collimation", "filter", "resolution"],
        "required_terms": ["BT-7", "BT-1", "incident energy", "final energy", "collimation", "filter", "resolution"],
        "queries": ["instrument BT-7 BT-1 incident energy final energy collimation filter resolution"],
    },
    {
        "target_id": "measurement_conditions",
        "fact_types": ["temperature_range", "magnetic_field", "pressure", "sample_environment", "data_reduction"],
        "required_terms": ["temperature", "magnetic field", "pressure", "sample environment", "Rietveld", "normalization"],
        "queries": ["temperature magnetic field pressure sample environment Rietveld normalization"],
    },
]

AGENTIC_FACT_TYPE_GUIDANCE = {
    "crystal_structure": "Crystallographic structure only: space group, lattice, unit cell, structural family, or atomic positions. Do not use for magnetic order.",
    "magnetic_structure": "Magnetic order pattern or magnetic ground state, such as Neel, stripe, ferromagnetic, antiferromagnetic, q=0, sublattice relationship, or ordered spin arrangement.",
    "propagation_vector": "Magnetic propagation vector or ordering wavevector only, such as k=(0,0,0) or q=(1/2,0,0).",
    "spin_direction": "Direction or plane of ordered moments, such as ab-plane, c-axis, along a, easy-plane, or easy-axis.",
    "ordered_moment": "Ordered magnetic moment size and units, such as 3.4 mu_B per Mn.",
    "exchange_interaction": "Hamiltonian exchange terms, exchange constants, bond labels, J1/J2/Jc/J3, and signs/magnitudes when stated. Hamiltonian anisotropy terms (single-ion D, easy-axis/easy-plane terms in meV) also go here, not under magnetic_anisotropy.",
    "magnetic_anisotropy": "Bulk anisotropy quantities in the MagMO sense: anisotropy constants K1/K2/Ku as energy densities, anisotropy field H_K, magnetocrystalline/shape/induced anisotropy statements. Not Hamiltonian D terms (those are exchange_interaction).",
    "magnetization": "Spontaneous or saturation magnetization or polarization (Ms, Js) in A/m, T, emu/g or mu_B per formula unit as a bulk quantity. Not the ordered moment per ion from diffraction (ordered_moment).",
    "hysteresis_property": "Hysteresis-loop quantities: coercivity (state the convention if given), remanence, energy product, loop squareness, susceptibility, demagnetizing factor.",
    "source_data_clue": "Measured reciprocal-space directions, data-source clues, or modeling clues that do not fit a narrower magnetic type.",
    "instrument_configuration": "Instrument setup details that are not purely neutron energy: choppers, monochromator/analyzer, detector setup, instrument mode.",
    "energy_setting": "Neutron energy settings only, such as incident energy Ei or final energy Ef. Do not use for sample temperature.",
    "temperature_range": "Sample temperature or temperature range at which measurements were made only. Do not use for neutron energy or for transition temperatures.",
    "transition_temperature": "Neel, Curie, superconducting or structural transition temperatures (TN, Tc, TS) stated by the paper, with the material they belong to. Not a measurement range.",
    "magnetic_field": "Externally applied magnetic field only. Do not use for magnetic moment, spin direction, exchange field, or magnetic order.",
    "pressure": "Externally applied pressure or pressure cell information only.",
    "sample_environment": "Cryostat, furnace, magnet, pressure cell, or other sample environment.",
    "data_reduction": "Rietveld refinement, background subtraction, normalization, form-factor correction, resolution treatment, or fitting method.",
    "resolution": "Instrumental energy or Q resolution, not model fit quality.",
    "filter": "Neutron filters such as PG, Be, velocity selector, or harmonic-suppression devices.",
    "collimation": "Angular collimation settings only.",
}


def build_srmn2sb2_agentic_rag_catalog(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a private document-side catalog for the SrMn2Sb2 analog-RAG case.

    SrMn2Sb2 scattering is unpublished in the current project context. This
    catalog therefore stores the provided CaMn2Sb2 bulk/neutron papers as analog
    evidence, not as direct SrMn2Sb2 publication evidence.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments.get("output_dir") or catalog_dir).expanduser()
    catalog_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor_run_dir = Path(str(arguments.get("extractor_run_dir") or "")).expanduser() if arguments.get("extractor_run_dir") else None
    source_markdowns = [Path(str(item)).expanduser() for item in (arguments.get("source_markdowns") or arguments.get("sources") or [])]
    max_chars_per_chunk = int(arguments.get("max_chars_per_chunk") or 3500)

    documents: list[dict[str, Any]] = []
    papers: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    docling_sources = _resolve_docling_sources(extractor_run_dir=extractor_run_dir, source_markdowns=source_markdowns)
    if not docling_sources:
        warnings.append(
            "No Docling Markdown sources were found. Run scripts/run_vista_srmn2sb2_agentic_rag.sh, "
            "or pass --extractor-run-dir pointing at a Vista Docling run."
        )

    for index, source_record in enumerate(docling_sources):
        source = Path(str(source_record["markdown_path"])).expanduser()
        paper_id, title, role = _source_identity(source, index=index)
        document_id = f"doc-{paper_id}"
        if not source.exists():
            warnings.append(f"Missing source: {source}")
            source_reports.append({"source": str(source), "status": "missing_input", "paper_id": paper_id})
            continue
        text, extraction = _read_docling_text(source, source_record=source_record)
        source_hash = _file_hash(source)
        text_hash = _text_hash(text)
        provenance_records = [
            ProvenanceRecord(
                source_type="vista_docling_markdown",
                source_id=str(source),
                method=extraction["method"],
                metadata={
                    "analog_role": role,
                    "private_local_only": True,
                    "original_pdf": source_record.get("pdf") or source_record.get("pdf_name"),
                    "extractor_run_dir": str(extractor_run_dir) if extractor_run_dir else None,
                },
            )
        ]
        documents.append(
            Document(
                document_id=document_id,
                document_type="paper",
                title=title,
                paper_id=paper_id,
                local_path=str(source),
                source_hash=source_hash,
                copyright_status="private_local_only",
                ingest_status="vista_docling_extracted",
                extraction_method=extraction["method"],
                parser_version=extraction.get("parser_version"),
                metadata={
                    "analog_role": role,
                    "target_case": "SrMn2Sb2 experiment 24858",
                    "is_direct_srmn2sb2_scattering_publication": False,
                    "docling_markdown_path": str(extraction.get("text_path") or ""),
                    "original_pdf": source_record.get("pdf") or source_record.get("pdf_name"),
                },
                provenance=provenance_records,
            ).model_dump(mode="json")
        )
        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                authors=[],
                year=_guess_year(text, source),
                doi=_guess_doi(text),
                metadata={
                    "analog_role": role,
                    "target_case": "SrMn2Sb2 experiment 24858",
                    "source_filename": source.name,
                    "private_local_only": True,
                },
                provenance=provenance_records,
            ).model_dump(mode="json")
        )
        for chunk_index, chunk in enumerate(_chunk_text(text, max_chars=max_chars_per_chunk)):
            chunk_id = f"chunk-{paper_id}-{chunk_index:03d}"
            chunk_hash = _text_hash(chunk)
            chunks.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    source_type="paper",
                    source_id=paper_id,
                    text=chunk,
                    document_id=document_id,
                    paper_id=paper_id,
                    chunk_type=_chunk_type_hint(chunk),
                    title=title,
                    char_start=None,
                    char_end=None,
                    token_count=len(chunk.split()),
                    source_hash=source_hash,
                    chunk_hash=chunk_hash,
                    extraction_method=extraction["method"],
                    parser_version=extraction.get("parser_version"),
                    metadata={
                        "analog_role": role,
                        "private_local_only": True,
                        "original_pdf": source_record.get("pdf") or source_record.get("pdf_name"),
                    },
                    provenance=provenance_records,
                ).model_dump(mode="json")
            )
        source_reports.append(
            {
                "source": str(source),
                "status": "ok",
                "paper_id": paper_id,
                "title": title,
                "analog_role": role,
                "text_chars": len(text),
                "text_hash": text_hash,
                "chunk_count": sum(1 for item in chunks if item.get("paper_id") == paper_id),
                "extraction": extraction,
            }
        )

    _write_jsonl(catalog_dir / "documents.jsonl", documents)
    _write_jsonl(catalog_dir / "papers.jsonl", papers)
    _write_jsonl(catalog_dir / "source_chunks.jsonl", chunks)
    manifest = {
        "case_id": "srmn2sb2_agentic_rag_ca_analog_v0",
        "target_material": "SrMn2Sb2",
        "target_experiment": "BT-7 experiment 24858",
        "direct_publication_status": "unpublished_scattering_case",
        "analog_material": "CaMn2Sb2",
        "sources": source_reports,
        "warnings": warnings,
        "policy": [
            "CaMn2Sb2 literature is analog evidence for SrMn2Sb2 model selection, not direct SrMn2Sb2 source evidence.",
            "Docling extraction and Gemma fact extraction should run on Vista for this case.",
            "Generated private Markdown and PaperFact records are local/Vista derived artifacts and must not be committed.",
            "LLM extraction should produce typed PaperFact records with cited chunk spans; missing fields are expected.",
        ],
    }
    _write_json(catalog_dir / "srmn2sb2_agentic_rag_manifest.json", manifest)
    report = {
        "status": "ok" if documents else "missing_input",
        "catalog": str(catalog_dir),
        "output_dir": str(output_dir),
        "document_count": len(documents),
        "paper_count": len(papers),
        "chunk_count": len(chunks),
        "manifest": manifest,
        "next_commands": _next_commands(catalog_dir, output_dir, papers),
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_catalog_markdown,
        json_filename="srmn2sb2_agentic_rag_catalog_report.json",
        markdown_filename="srmn2sb2_agentic_rag_catalog_report.md",
    )
    return report


def srmn2sb2_agentic_rag_report(arguments: dict[str, Any]) -> dict[str, Any]:
    """Assemble an analog-evidence report after bounded paper-fact extraction."""

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    sr_case_note_arg = arguments.get("sr_case_note") or DEFAULT_SR_CASE_NOTE
    sr_case_note = Path(str(sr_case_note_arg)).expanduser() if sr_case_note_arg else None
    sr_fit_report = Path(str(arguments.get("sr_fit_report") or "")).expanduser() if arguments.get("sr_fit_report") else None

    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    manifest = _load_json(catalog_dir / "srmn2sb2_agentic_rag_manifest.json")
    facts_by_type: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        facts_by_type.setdefault(str(fact.get("fact_type")), []).append(fact)

    sr_case_text = sr_case_note.read_text(encoding="utf-8") if sr_case_note and sr_case_note.exists() else ""
    sr_fit_text = sr_fit_report.read_text(encoding="utf-8") if sr_fit_report and sr_fit_report.exists() else ""
    answer = _build_case_answer(facts_by_type=facts_by_type, sr_case_text=sr_case_text, sr_fit_text=sr_fit_text)
    missing_inputs = []
    if not facts:
        missing_inputs.append("paper_facts.jsonl is absent; run Gemma extraction before treating this as an evidence report.")
    if sr_case_note and not sr_case_text:
        missing_inputs.append(f"SrMn2Sb2 case note missing: {sr_case_note}")
    if not sr_case_note:
        missing_inputs.append("SrMn2Sb2 case note not provided; pass --sr-case-note with a local review note if direct experiment context is needed.")
    if sr_fit_report and not sr_fit_text:
        missing_inputs.append(f"SrMn2Sb2 fit report missing: {sr_fit_report}")

    report = {
        "status": "needs_gemma_extraction" if not facts else "ok",
        "case_id": "srmn2sb2_agentic_rag_ca_analog_v0",
        "target_material": "SrMn2Sb2",
        "target_experiment": "BT-7 experiment 24858",
        "direct_publication_status": "unpublished_scattering_case",
        "analog_material": "CaMn2Sb2",
        "catalog": str(catalog_dir),
        "paper_count": len(papers),
        "chunk_count": len(chunks),
        "paper_fact_count": len(facts),
        "paper_fact_types": {key: len(value) for key, value in sorted(facts_by_type.items())},
        "answer": answer,
        "missing_inputs": missing_inputs,
        "manifest": manifest,
        "policy": [
            "Do not call the CaMn2Sb2 literature a SrMn2Sb2 paper.",
            "Use CaMn2Sb2 facts as analog priors for Hamiltonian and magnetic-structure choices.",
            "Use SrMn2Sb2 experiment 24858 scans and fits as the direct evidence for SrMn2Sb2.",
            "Keep all extracted private text and generated facts local-only.",
        ],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_agentic_report_markdown,
        json_filename="srmn2sb2_agentic_rag_report.json",
        markdown_filename="srmn2sb2_agentic_rag_report.md",
    )
    (output_dir / "srmn2sb2_agentic_rag_report.html").write_text(_agentic_report_html(report), encoding="utf-8")
    report["artifacts"]["html"] = str(output_dir / "srmn2sb2_agentic_rag_report.html")
    return report


def run_srmn2sb2_agentic_rag_loop(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a bounded retrieve -> extract -> gap-check loop for the SrMn2Sb2 case.

    This is intentionally not an open-ended agent. The controller owns the task
    list, retrieval rules, schema, merge policy, and stopping condition. The LLM
    only extracts typed facts from selected chunks for a named target.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = str(arguments.get("provider") or "none")
    model = str(arguments.get("model") or "")
    ollama_url = str(arguments.get("ollama_url") or "http://127.0.0.1:11434")
    vllm_url = str(arguments.get("vllm_url") or "")
    api_key = arguments.get("api_key")
    max_targets = int(arguments.get("max_targets") or len(AGENTIC_FACT_TARGETS))
    top_k = int(arguments.get("top_k") or 6)
    max_chars_per_target = int(arguments.get("max_chars_per_target") or 18000)
    max_followups = int(arguments.get("max_followups") or 1)

    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    existing_facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    if not chunks:
        return _agentic_loop_missing(output_dir, catalog_dir, "No source chunks found in catalog.")
    if provider == "ollama" and not model:
        return _agentic_loop_missing(output_dir, catalog_dir, "provider=ollama requires --model.")
    if provider == "vllm" and (not model or not vllm_url):
        return _agentic_loop_missing(output_dir, catalog_dir, "provider=vllm requires --model and --vllm-url (or --llm-endpoint).")

    all_new_facts: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    paper_by_id = {str(paper.get("paper_id")): paper for paper in papers if paper.get("paper_id")}
    paper_ids = sorted({str(chunk.get("paper_id")) for chunk in chunks if chunk.get("paper_id")})
    for target in AGENTIC_FACT_TARGETS[:max_targets]:
        for paper_id in paper_ids:
            target_report, facts, target_errors = _run_agentic_target(
                catalog_dir=catalog_dir,
                paper_id=paper_id,
                paper_title=paper_by_id.get(paper_id, {}).get("title"),
                chunks=[chunk for chunk in chunks if str(chunk.get("paper_id")) == paper_id],
                target=target,
                provider=provider,
                model=model,
                ollama_url=ollama_url,
                vllm_url=vllm_url,
                api_key=api_key,
                top_k=top_k,
                max_chars_per_target=max_chars_per_target,
                max_followups=max_followups,
            )
            target_reports.append(target_report)
            all_new_facts.extend(facts)
            errors.extend(target_errors)

    if all_new_facts:
        _write_jsonl(catalog_dir / "paper_facts.jsonl", _dedupe(existing_facts + all_new_facts, "fact_id"))

    final_facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    coverage = _agentic_coverage(final_facts)
    report = {
        "status": "ok" if provider != "none" and all_new_facts and not errors else ("partial" if all_new_facts else "needs_model_or_review"),
        "case_id": "srmn2sb2_agentic_rag_ca_analog_v0",
        "controller": "bounded_fact_gap_loop_v0",
        "catalog": str(catalog_dir),
        "provider": provider,
        "model": model or None,
        "target_count": len(target_reports),
        "new_fact_count": len(all_new_facts),
        "total_fact_count": len(final_facts),
        "coverage": coverage,
        "target_reports": target_reports,
        "errors": errors,
        "policy": [
            "The controller chooses retrieval targets deterministically.",
            "Gemma extracts typed candidate facts only from selected chunks.",
            "Schema/span validation remains mechanical; accepted graph edges require human review.",
            "CaMn2Sb2 facts are analog priors for SrMn2Sb2, not direct SrMn2Sb2 publications.",
        ],
    }
    _write_json(output_dir / "agentic_loop_targets.json", {"targets": target_reports})
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_agentic_loop_markdown,
        json_filename="srmn2sb2_agentic_rag_loop_report.json",
        markdown_filename="srmn2sb2_agentic_rag_loop_report.md",
    )
    (output_dir / "srmn2sb2_agentic_rag_loop_report.html").write_text(_agentic_loop_html(report), encoding="utf-8")
    report["artifacts"]["html"] = str(output_dir / "srmn2sb2_agentic_rag_loop_report.html")
    report["artifacts"]["targets"] = str(output_dir / "agentic_loop_targets.json")
    return report


def _run_agentic_target(
    *,
    catalog_dir: Path,
    paper_id: str,
    paper_title: str | None,
    chunks: list[dict[str, Any]],
    target: dict[str, Any],
    provider: str,
    model: str,
    ollama_url: str,
    vllm_url: str = "",
    api_key: Any = None,
    top_k: int,
    max_chars_per_target: int,
    max_followups: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_id = str(target["target_id"])
    selected = _select_target_chunks(chunks, target=target, top_k=top_k, max_chars=max_chars_per_target)
    target_report = {
        "target_id": target_id,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "required_fact_types": target.get("fact_types") or [],
        "queries": target.get("queries") or [],
        "selected_chunk_ids": [str(chunk.get("chunk_id")) for chunk in selected],
        "selected_chunks": [{"chunk_id": str(chunk.get("chunk_id")), "excerpt": str(chunk.get("text") or "")[:500]} for chunk in selected],
        "followup_count": 0,
        "status": "needs_model_or_review" if provider == "none" else "pending",
        "new_fact_count": 0,
        "coverage_before": {},
        "coverage_after": {},
    }
    if not selected:
        target_report["status"] = "missing_input"
        target_report["missing_reason"] = "No chunks matched this target."
        return target_report, [], []
    if provider == "none":
        return target_report, [], []

    packet = _agentic_fact_packet(paper_id, selected, target=target, title=paper_title)
    raw: dict[str, Any]
    errors: list[dict[str, Any]] = []
    try:
        raw = _call_agentic_model(packet, provider=provider, model=model, ollama_url=ollama_url, vllm_url=vllm_url, api_key=api_key)
    except Exception as exc:
        target_report["status"] = "error"
        errors.append({"paper_id": paper_id, "target_id": target_id, "error": f"{provider} target extraction failed: {exc}"})
        return target_report, [], errors
    for error in raw.get("errors") or []:
        errors.append({"paper_id": paper_id, "target_id": target_id, "error": str(error)})
    facts, validation_errors = normalize_paper_fact_response(raw, packet=packet, provider=f"agentic_{provider}", model=model)
    errors.extend({"paper_id": paper_id, "target_id": target_id, "error": item} for item in validation_errors)

    followup_rounds = 0
    missing_after_first = _target_missing_fact_types(facts, target)
    while missing_after_first and followup_rounds < max_followups:
        followup_rounds += 1
        followup_chunks = _select_followup_chunks(chunks, selected, target=target, missing_fact_types=missing_after_first, top_k=top_k, max_chars=max_chars_per_target)
        if not followup_chunks:
            break
        selected_ids = {str(chunk.get("chunk_id")) for chunk in selected}
        selected.extend(chunk for chunk in followup_chunks if str(chunk.get("chunk_id")) not in selected_ids)
        packet = _agentic_fact_packet(paper_id, followup_chunks, target=target, title=paper_title, missing_fact_types=missing_after_first)
        try:
            raw = _call_agentic_model(packet, provider=provider, model=model, ollama_url=ollama_url, vllm_url=vllm_url, api_key=api_key)
        except Exception as exc:
            errors.append({"paper_id": paper_id, "target_id": target_id, "error": f"{provider} follow-up extraction failed: {exc}"})
            break
        for error in raw.get("errors") or []:
            errors.append({"paper_id": paper_id, "target_id": target_id, "error": str(error)})
        followup_facts, validation_errors = normalize_paper_fact_response(raw, packet=packet, provider=f"agentic_{provider}", model=model)
        facts.extend(followup_facts)
        errors.extend({"paper_id": paper_id, "target_id": target_id, "error": item} for item in validation_errors)
        missing_after_first = _target_missing_fact_types(facts, target)

    facts = [_sanitize_agentic_fact_semantics(fact) for fact in _prune_target_facts(facts, target)]
    target_report.update(
        {
            "status": "ok" if facts else "no_facts",
            "selected_chunk_ids": [str(chunk.get("chunk_id")) for chunk in selected],
            "selected_chunks": [{"chunk_id": str(chunk.get("chunk_id")), "excerpt": str(chunk.get("text") or "")[:500]} for chunk in selected],
            "followup_count": followup_rounds,
            "new_fact_count": len(facts),
            "coverage_after": _target_coverage(facts, target),
            "missing_fact_types": _target_missing_fact_types(facts, target),
        }
    )
    return target_report, facts, errors


def _select_target_chunks(chunks: list[dict[str, Any]], *, target: dict[str, Any], top_k: int, max_chars: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for chunk in chunks:
        score = _target_chunk_score(chunk, target)
        if score > 0:
            scored.append((score, str(chunk.get("chunk_id") or ""), chunk))
    if not scored:
        scored = [(0.1, str(chunk.get("chunk_id") or ""), chunk) for chunk in chunks]
    selected: list[dict[str, Any]] = []
    used = 0
    for _score, _chunk_id, chunk in sorted(scored, key=lambda item: (-item[0], item[1])):
        text_len = len(str(chunk.get("text") or ""))
        if selected and used + text_len > max_chars:
            continue
        selected.append(chunk)
        used += text_len
        if len(selected) >= top_k or used >= max_chars:
            break
    return selected


def _select_followup_chunks(
    chunks: list[dict[str, Any]],
    already_selected: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    missing_fact_types: list[str],
    top_k: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    selected_ids = {str(chunk.get("chunk_id")) for chunk in already_selected}
    followup_target = dict(target)
    followup_target["required_terms"] = list(target.get("required_terms") or []) + missing_fact_types
    candidates = [chunk for chunk in chunks if str(chunk.get("chunk_id")) not in selected_ids]
    return _select_target_chunks(candidates, target=followup_target, top_k=top_k, max_chars=max_chars)


def _target_chunk_score(chunk: dict[str, Any], target: dict[str, Any]) -> float:
    text = _normalize_for_search(" ".join(str(chunk.get(key) or "") for key in ["chunk_type", "title", "section", "text"]))
    score = 0.0
    for term in target.get("required_terms") or []:
        normalized = _normalize_for_search(str(term))
        if normalized and normalized in text:
            score += 4.0 if len(normalized.split()) > 1 else 2.0
    for query in target.get("queries") or []:
        for term in _normalize_for_search(str(query)).split():
            if len(term) >= 3 and term in text:
                score += 0.7
    chunk_type = str(chunk.get("chunk_type") or "")
    if target["target_id"] == "hamiltonian" and chunk_type in {"results", "discussion", "unknown"}:
        score += 1.5
    if target["target_id"] == "instrument_setup" and chunk_type == "instrument_setup":
        score += 2.0
    if target["target_id"] == "magnetic_structure" and chunk_type in {"structure_parameters", "abstract"}:
        score += 2.0
    return score


def _agentic_fact_packet(
    paper_id: str,
    chunks: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    title: str | None,
    missing_fact_types: list[str] | None = None,
) -> dict[str, Any]:
    allowed_fact_types = [fact_type for fact_type in target.get("fact_types", []) if fact_type in FACT_TYPES]
    if missing_fact_types:
        allowed_fact_types = [fact_type for fact_type in missing_fact_types if fact_type in allowed_fact_types] or allowed_fact_types
    document_id = next((chunk.get("document_id") for chunk in chunks if chunk.get("document_id")), None)
    return {
        "paper_id": paper_id,
        "document_id": document_id,
        "title": title,
        "prompt_version": "srmn2sb2_agentic_fact_gap_loop_v0",
        "agentic_target": {
            "target_id": target.get("target_id"),
            "allowed_fact_types": allowed_fact_types,
            "required_terms": target.get("required_terms") or [],
            "missing_fact_types": missing_fact_types or [],
            "analog_policy": "CaMn2Sb2 facts are analog priors for unpublished SrMn2Sb2, not direct SrMn2Sb2 evidence.",
        },
        "instructions": [
            "Extract only facts relevant to this target.",
            "Use only the selected chunks below.",
            "Every supported fact must cite chunk_id and a short quote from that chunk.",
            "If a requested fact is missing, emit not_found or ambiguous rather than guessing.",
            "Do not fill a fact type with a related but different concept. A magnetic moment is not an applied magnetic_field, and a sample temperature is not an energy_setting.",
            "Return at most one best fact per allowed fact_type unless the chunks contain genuinely separate values for the same fact type.",
            "Do not create graph edges; this controller proposes edges later after validation.",
            "Allowed fact_type values for this target: " + ", ".join(allowed_fact_types),
            "Allowed answer_status values: " + ", ".join(sorted(ANSWER_STATUSES)),
        ],
        "fact_type_guidance": {fact_type: AGENTIC_FACT_TYPE_GUIDANCE.get(fact_type, "Use the literal meaning of this fact type.") for fact_type in allowed_fact_types},
        "output_schema": {
            "facts": [
                {
                    "fact_type": allowed_fact_types[0] if allowed_fact_types else "uncertainty_note",
                    "answer_status": "supported|not_found|ambiguous|unsupported",
                    "value": {"text": "short structured fact value"},
                    "confidence": 0.0,
                    "evidence": [{"chunk_id": "chunk-id", "quote": "short exact quote"}],
                    "warnings": [],
                }
            ]
        },
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_type": chunk.get("chunk_type"),
                "title": chunk.get("title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "section": chunk.get("section"),
                "text": str(chunk.get("text") or "")[:5000],
            }
            for chunk in chunks
        ],
    }


def _call_agentic_model(
    packet: dict[str, Any],
    *,
    provider: str,
    model: str,
    ollama_url: str,
    vllm_url: str = "",
    api_key: Any = None,
) -> dict[str, Any]:
    """Dispatch one bounded extraction call to the configured endpoint.

    The packet, schema, and validators are identical for every provider;
    only the transport differs, so Vista Gemma (Ollama) and the NIST
    OpenAI-compatible service are interchangeable per call.
    """

    if provider == "ollama":
        return _call_agentic_ollama(packet, base_url=ollama_url, model=model)
    if provider == "vllm":
        schema = _agentic_response_schema(packet)
        body = openai_compatible_chat(
            base_url=vllm_url,
            model=model,
            messages=_agentic_messages(packet),
            api_key=str(api_key) if api_key else None,
            response_format=schema,
            max_tokens=4096,
            timeout=300.0,
        )
        choices = body.get("choices") or []
        message = (choices[0].get("message") if choices and isinstance(choices[0], dict) else body.get("message")) or {}
        content = str(message.get("content") or "{}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {"facts": [], "errors": ["agentic target extraction returned malformed JSON"]}
        return data if isinstance(data, dict) else {"facts": [], "errors": ["agentic target extraction returned non-object JSON"]}
    raise ValueError(f"Unknown provider: {provider}")


def _agentic_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded paper-fact extractor inside TAS Co-pilot. "
                "The controller, not you, chooses retrieval targets. Return only valid JSON. "
                "Do not infer missing facts; use not_found or ambiguous. Cite chunk_id and short quotes. "
                "Do not put magnetic moment/order into magnetic_field or sample temperature into energy_setting."
            ),
        },
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]


def _agentic_response_schema(packet: dict[str, Any]) -> dict[str, Any]:
    allowed_fact_types = packet.get("agentic_target", {}).get("allowed_fact_types") or FACT_TYPES
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": min(10, max(1, len(allowed_fact_types) + 2)),
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_type": {"type": "string", "enum": allowed_fact_types},
                        "answer_status": {"type": "string", "enum": sorted(ANSWER_STATUSES)},
                        "value": {"type": "object"},
                        "confidence": {"type": "number"},
                        "evidence": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                                "required": ["chunk_id"],
                            },
                        },
                        "warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["fact_type", "answer_status", "value"],
                },
            },
            "errors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facts"],
    }


def _call_agentic_ollama(packet: dict[str, Any], *, base_url: str, model: str) -> dict[str, Any]:
    body = ollama_chat(
        base_url,
        model,
        _agentic_messages(packet),
        format_schema=_agentic_response_schema(packet),
        options={"temperature": 0, "num_predict": 4096},
        timeout=300.0,
    )
    content = str((body.get("message") or {}).get("content") or "{}")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"facts": [], "errors": ["agentic target extraction returned malformed JSON"]}
    return data if isinstance(data, dict) else {"facts": [], "errors": ["agentic target extraction returned non-object JSON"]}


def _target_missing_fact_types(facts: list[dict[str, Any]], target: dict[str, Any]) -> list[str]:
    supported = {str(fact.get("fact_type")) for fact in facts if fact.get("answer_status") == "supported"}
    return [fact_type for fact_type in target.get("fact_types", []) if fact_type not in supported]


def _prune_target_facts(facts: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = [str(fact_type) for fact_type in target.get("fact_types") or []]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        fact_type = str(fact.get("fact_type"))
        if fact_type in allowed:
            grouped.setdefault(fact_type, []).append(fact)
    pruned: list[dict[str, Any]] = []
    for fact_type in allowed:
        candidates = grouped.get(fact_type) or []
        if not candidates:
            continue
        candidates = sorted(candidates, key=_fact_rank)
        best = dict(candidates[0])
        dropped = len(candidates) - 1
        if dropped:
            warnings = list(best.get("warnings") or [])
            warnings.append(f"Pruned {dropped} lower-ranked duplicate candidate(s) for this target/fact_type.")
            best["warnings"] = warnings
        pruned.append(best)
    return pruned


def _fact_rank(fact: dict[str, Any]) -> tuple[int, float, int]:
    status_rank = {"supported": 0, "ambiguous": 1, "not_found": 2, "unsupported": 3}
    status = status_rank.get(str(fact.get("answer_status")), 9)
    try:
        confidence = float(fact.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence_count = len(fact.get("source_chunk_ids") or [])
    return (status, -confidence, -evidence_count)






def _target_coverage(facts: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, str]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        by_type.setdefault(str(fact.get("fact_type")), []).append(fact)
    coverage: dict[str, str] = {}
    for fact_type in target.get("fact_types", []):
        statuses = {str(fact.get("answer_status")) for fact in by_type.get(fact_type, [])}
        if "supported" in statuses:
            coverage[fact_type] = "supported"
        elif statuses:
            coverage[fact_type] = ",".join(sorted(statuses))
        else:
            coverage[fact_type] = "missing"
    return coverage


def _agentic_coverage(facts: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    for fact in facts:
        fact_type = str(fact.get("fact_type"))
        status = str(fact.get("answer_status") or "unknown")
        by_type.setdefault(fact_type, {})
        by_type[fact_type][status] = by_type[fact_type].get(status, 0) + 1
    target_coverage = {}
    for target in AGENTIC_FACT_TARGETS:
        target_coverage[target["target_id"]] = _target_coverage(facts, target)
    return {"fact_type_status_counts": by_type, "target_coverage": target_coverage}


def _agentic_loop_missing(output_dir: Path, catalog_dir: Path, reason: str) -> dict[str, Any]:
    report = {
        "status": "missing_input",
        "catalog": str(catalog_dir),
        "new_fact_count": 0,
        "total_fact_count": 0,
        "errors": [{"error": reason}],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_agentic_loop_markdown,
        json_filename="srmn2sb2_agentic_rag_loop_report.json",
        markdown_filename="srmn2sb2_agentic_rag_loop_report.md",
    )
    return report


def _agentic_loop_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SrMn2Sb2 Bounded Agentic-RAG Loop",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Controller: `{report.get('controller')}`",
        f"- Provider: `{report.get('provider')}`",
        f"- Model: `{report.get('model')}`",
        f"- New facts: {report.get('new_fact_count')}",
        f"- Total facts: {report.get('total_fact_count')}",
        "",
        "## Target Results",
        "",
    ]
    for target in report.get("target_reports") or []:
        lines.append(
            f"- `{target.get('paper_id')}` / `{target.get('target_id')}`: "
            f"{target.get('status')} facts={target.get('new_fact_count')} "
            f"missing={', '.join(target.get('missing_fact_types') or []) or 'none'}"
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item.get('paper_id')}` `{item.get('target_id')}`: {item.get('error')}" for item in report["errors"])
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- This is bounded agentic RAG: the controller chooses targets and retrieval; Gemma only extracts typed facts.",
            "- CaMn2Sb2 facts remain analog priors for SrMn2Sb2 until human review and direct experiment evidence support them.",
        ]
    )
    return "\n".join(lines) + "\n"


def _agentic_loop_html(report: dict[str, Any]) -> str:
    rows = []
    for target in report.get("target_reports") or []:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(target.get('paper_id')))}</code></td>"
            f"<td><code>{html.escape(str(target.get('target_id')))}</code></td>"
            f"<td>{html.escape(str(target.get('status')))}</td>"
            f"<td>{html.escape(str(target.get('new_fact_count')))}</td>"
            f"<td>{html.escape(', '.join(target.get('missing_fact_types') or []) or 'none')}</td>"
            "<td>"
            + "".join(
                f"<details><summary><code>{html.escape(str(chunk.get('chunk_id')))}</code></summary>"
                f"<blockquote>{html.escape(str(chunk.get('excerpt') or ''))}</blockquote></details>"
                for chunk in (target.get("selected_chunks") or [{"chunk_id": cid, "excerpt": ""} for cid in (target.get("selected_chunk_ids") or [])])
            )
            + "</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SrMn2Sb2 Bounded Agentic-RAG Loop</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.45}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{background:#f4f4f4}code{background:#f7f7f7;padding:1px 3px}</style></head><body>"
        "<h1>SrMn2Sb2 Bounded Agentic-RAG Loop</h1>"
        f"<p>Status: <code>{html.escape(str(report.get('status')))}</code>; "
        f"new facts: {html.escape(str(report.get('new_fact_count')))}; "
        f"total facts: {html.escape(str(report.get('total_fact_count')))}</p>"
        "<p>The controller selects targets and chunks. Gemma extracts typed candidate facts. Human review is still required.</p>"
        "<table><thead><tr><th>Paper</th><th>Target</th><th>Status</th><th>Facts</th><th>Missing</th><th>Chunks</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )


def _dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get(key):
            out[str(row[key])] = row
    return list(out.values())


def _normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+-]+", " ", text.lower())).strip()


def _source_identity(source: Path, *, index: int) -> tuple[str, str, str]:
    name = source.name.lower()
    if "bridge" in name or "poulton" in name:
        return (
            "paper-camn2sb2-bridges-powder",
            "Magnetic order in CaMn2Sb2 studied via powder neutron diffraction",
            "bulk_powder_neutron_magnetic_structure",
        )
    if "igor" in name or "inelastic" in name or "mcnally" in name:
        return (
            "paper-camn2sb2-mcnally-spinwaves",
            "CaMn2Sb2: Spin waves on a frustrated antiferromagnetic honeycomb lattice",
            "single_crystal_inelastic_spinwave_hamiltonian",
        )
    stem = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")
    return f"paper-camn2sb2-analog-{index:02d}-{stem[:32]}", source.stem.replace("_", " "), "ca_mn2sb2_analog_evidence"


def _resolve_docling_sources(*, extractor_run_dir: Path | None, source_markdowns: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if extractor_run_dir is not None:
        summary = extractor_run_dir / "outputs" / "docling" / "summary.json"
        if summary.exists():
            for item in json.loads(summary.read_text(encoding="utf-8")):
                if item.get("status") != "success":
                    continue
                markdown_path = item.get("markdown_path")
                if not markdown_path:
                    continue
                path = Path(str(markdown_path))
                if not path.exists():
                    candidate = extractor_run_dir / "outputs" / "docling" / "md" / Path(str(markdown_path)).name
                    path = candidate if candidate.exists() else path
                row = dict(item)
                row["markdown_path"] = str(path)
                rows.append(row)
        md_dir = extractor_run_dir / "outputs" / "docling" / "md"
        if md_dir.exists() and not rows:
            rows.extend({"markdown_path": str(path), "backend": "docling", "status": "success"} for path in sorted(md_dir.glob("*.md")))
    rows.extend({"markdown_path": str(path), "backend": "provided_markdown", "status": "success"} for path in source_markdowns)
    return rows


def _read_docling_text(source: Path, *, source_record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if source.suffix.lower() not in {".md", ".txt"}:
        raise RuntimeError(
            "SrMn2Sb2 agentic-RAG cataloging consumes Docling Markdown/text, not local PDFs. "
            "Run scripts/run_vista_srmn2sb2_agentic_rag.sh or pass --extractor-run-dir."
        )
    text = source.read_text(encoding="utf-8", errors="replace")
    backend = str(source_record.get("backend") or "docling")
    method = "vista_docling_markdown" if backend == "docling" else "provided_markdown_fixture"
    return text, {"method": method, "parser_version": source_record.get("parser_version") or "docling_vista_batch", "text_path": str(source)}


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    pages = [page.strip() for page in text.split("\f") if page.strip()]
    if not pages:
        pages = [text.strip()]
    chunks: list[str] = []
    for page in pages:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", page) if item.strip()]
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            while len(paragraph) > max_chars:
                chunks.append(paragraph[:max_chars])
                paragraph = paragraph[max_chars:]
            current = paragraph
        if current:
            chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def _chunk_type_hint(text: str) -> str:
    low = text.lower()
    if any(term in low for term in ["instrument", "neutron scattering", "measurements were", "incident energy", "collimation"]):
        return "instrument_setup"
    if any(term in low for term in ["heisenberg", "hamiltonian", "exchange", "spin wave", "spin-wave", "j1", "j2"]):
        return "results"
    if any(term in low for term in ["crystal structure", "magnetic structure", "space group", "lattice parameter", "ordered moment"]):
        return "structure_parameters"
    if any(term in low for term in ["fig.", "figure", "caption"]):
        return "figure_caption"
    if any(term in low for term in ["abstract", "we present", "we report"]):
        return "abstract"
    return "unknown"


def _build_case_answer(*, facts_by_type: dict[str, list[dict[str, Any]]], sr_case_text: str, sr_fit_text: str) -> dict[str, Any]:
    supported = {key: [fact for fact in values if fact.get("answer_status") == "supported"] for key, values in facts_by_type.items()}
    hamiltonian_facts = (
        supported.get("exchange_interaction", [])
        + supported.get("magnetic_structure", [])
        + supported.get("propagation_vector", [])
        + supported.get("spin_direction", [])
        + supported.get("ordered_moment", [])
        + supported.get("source_data_clue", [])
        + supported.get("measurement_kind", [])
    )
    direct_case_has_fit = "CaMn2Sb2-like" in sr_case_text or "J1" in sr_fit_text
    return {
        "short_answer": (
            "Use CaMn2Sb2 as analog literature evidence for the SrMn2Sb2 Hamiltonian and magnetic-structure prior; "
            "use BT-7 experiment 24858 and the pySpinW/resolution-convolved fits as the direct SrMn2Sb2 evidence."
        ),
        "direct_srmn2sb2_scattering_publication_found": False,
        "ca_mn2sb2_analog_evidence_available": bool(facts_by_type),
        "sr_mn2sb2_direct_experiment_evidence_available": bool(sr_case_text),
        "sr_mn2sb2_fit_context_available": direct_case_has_fit,
        "recommended_graph_edges": [
            {
                "edge_type": "ANALOG_MODEL_PRIOR_FOR",
                "source": "CaMn2Sb2 literature PaperFact records",
                "target": "SrMn2Sb2 experiment 24858 spin-wave modeling",
                "status": "proposed",
                "human_review_required": True,
            },
            {
                "edge_type": "DIRECT_EXPERIMENT_EVIDENCE_FOR",
                "source": "BT-7 experiment 24858 scans/fits",
                "target": "SrMn2Sb2 spin-wave branches",
                "status": "candidate",
                "human_review_required": True,
            },
        ],
        "hamiltonian_fact_count": len(hamiltonian_facts),
        "next_review_questions": [
            "Do the CaMn2Sb2 extracted facts correctly state the magnetic structure, spin direction, and exchange terms?",
            "Which CaMn2Sb2 Hamiltonian terms should be transferred as priors to SrMn2Sb2?",
            "Which SrMn2Sb2 scan families in experiment 24858 directly support or contradict those priors?",
            "Which facts are missing because the CaMn2Sb2 extraction packet did not include enough of the paper?",
        ],
    }


def _next_commands(catalog_dir: Path, output_dir: Path, papers: list[dict[str, Any]]) -> list[str]:
    commands = []
    for paper in papers:
        commands.append(
            "python -m tas_copilot_core.cli corpus extract-paper-facts "
            f"--catalog {catalog_dir} --paper-id {paper['paper_id']} --provider ollama "
            "--model gemma3:4b --output-dir "
            f"{output_dir / ('facts_' + paper['paper_id'])}"
        )
    commands.append(
        "python -m tas_copilot_core.cli corpus srmn2sb2-agentic-rag-report "
        f"--catalog {catalog_dir} --output-dir {output_dir / 'report'}"
    )
    return commands


def _catalog_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SrMn2Sb2 Agentic-RAG Catalog",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Catalog: `{report.get('catalog')}`",
        f"- Papers: {report.get('paper_count')}",
        f"- Chunks: {report.get('chunk_count')}",
        "",
        "## Policy",
        "",
        "- SrMn2Sb2 scattering is treated as unpublished for this case.",
        "- CaMn2Sb2 papers are analog evidence, not direct SrMn2Sb2 publication evidence.",
        "- Generated private text is local-only and must not be committed.",
        "",
        "## Sources",
        "",
    ]
    for source in report.get("manifest", {}).get("sources", []):
        lines.append(f"- `{source.get('paper_id')}` `{source.get('status')}`: {source.get('title')} ({source.get('chunk_count', 0)} chunks)")
    lines.extend(["", "## Next Commands", ""])
    for command in report.get("next_commands", []):
        lines.extend(["```bash", command, "```", ""])
    return "\n".join(lines) + "\n"


def _agentic_report_markdown(report: dict[str, Any]) -> str:
    answer = report.get("answer", {})
    lines = [
        "# SrMn2Sb2 Agentic-RAG Analog Evidence Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Target: {report.get('target_material')} / {report.get('target_experiment')}",
        f"- Analog material: {report.get('analog_material')}",
        f"- Paper facts: {report.get('paper_fact_count')}",
        "",
        "## Answer",
        "",
        str(answer.get("short_answer") or ""),
        "",
        "## Evidence State",
        "",
        f"- Direct SrMn2Sb2 scattering publication found: `{answer.get('direct_srmn2sb2_scattering_publication_found')}`",
        f"- CaMn2Sb2 analog facts available: `{answer.get('ca_mn2sb2_analog_evidence_available')}`",
        f"- SrMn2Sb2 direct experiment evidence available: `{answer.get('sr_mn2sb2_direct_experiment_evidence_available')}`",
        f"- SrMn2Sb2 fit context available: `{answer.get('sr_mn2sb2_fit_context_available')}`",
        "",
        "## Fact Types",
        "",
    ]
    if report.get("paper_fact_types"):
        lines.extend(f"- `{key}`: {value}" for key, value in report["paper_fact_types"].items())
    else:
        lines.append("- No extracted facts yet.")
    if report.get("missing_inputs"):
        lines.extend(["", "## Missing Inputs", ""])
        lines.extend(f"- {item}" for item in report["missing_inputs"])
    lines.extend(["", "## Review Questions", ""])
    lines.extend(f"- {item}" for item in answer.get("next_review_questions", []))
    return "\n".join(lines) + "\n"


def _agentic_report_html(report: dict[str, Any]) -> str:
    body = html.escape(_agentic_report_markdown(report))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SrMn2Sb2 Agentic-RAG Report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;line-height:1.45}"
        "pre{white-space:pre-wrap;background:#f6f7f8;padding:1rem;border:1px solid #ddd}</style>"
        "</head><body><pre>"
        f"{body}</pre></body></html>"
    )


def _guess_year(text: str, source: Path) -> int | None:
    if "mcnally" in source.name.lower() or "igor" in source.name.lower() or "may 25, 2015" in text.lower():
        return 2015
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None


def _guess_doi(text: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0).rstrip(".,;)") if match else None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
