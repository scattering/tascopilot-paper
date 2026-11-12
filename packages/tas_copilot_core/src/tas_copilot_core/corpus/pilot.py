from __future__ import annotations

import hashlib
import html
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import yaml

from tas_copilot_contracts import CorpusIngestRun, CorpusRetrievalRun, Document, EvidenceGraphEdge, EvidenceGraphNode, PaperExperimentLink, ProvenanceRecord, PublicationDataLagPrior, RetrievalResult, ReviewLabel, ReviewLabelBatch, SourceChunk, ToolCall
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.corpus.fixtures import _load_jsonl_dicts, corpus_fixture_report, load_corpus_fixtures, validate_source_support
from tas_copilot_core.llm_fencing import scan_for_injection
from tas_copilot_core.ontology import load_ontology, supernode_diagnostics
from tas_copilot_core.ontology.linker import match_concepts
from tas_copilot_core.retrieval import GraphRetriever
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.review_labels import label_calibrated_score, label_informed_priors, review_label_calibration_diagnostics
import math
from collections import Counter

def _document_frequencies(chunk_texts: list[str]) -> Counter:
    df: Counter = Counter()
    for text in chunk_texts:
        for term in set(_terms(text)):
            df[term] += 1
    return df

def _idf(term: str, df: Counter, n_docs: int) -> float:
    return math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0

CHUNK_FILES = ["source_chunks.jsonl", "textbook_sections.jsonl"]
BROAD_CONCEPT_LABELS = {"bt-7", "temperature", "sample", "experiment", "neutron scattering"}
DEFAULT_PRIVATE_PDF_PAGE_LIMIT = 8
DEFAULT_PRIVATE_TEXT_LIMIT = 20000

import os
import math
from tas_copilot_core.llm_router import openai_compatible_embeddings


def _rchat_embedding_config() -> tuple[str, str, str | None]:
    """Read the NIST rchat embedding endpoint config from the environment.
    Mirrors the pattern in configs/llm_endpoints (nist_vllm)."""
    base_url = os.environ.get("NIST_VLLM_BASE_URL", "")
    model = os.environ.get("NIST_VLLM_EMBEDDING_MODEL", "")
    api_key = os.environ.get("NIST_VLLM_API_KEY")
    return base_url, model, api_key


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


import os
import math 
from pathlib import Path
from tas_copilot_core.llm_router import openai_compatible_embeddings

# Cache directory for chunk embeddings (keyed by embedding model + chunk_hash).
_EMBED_CACHE_DIR = Path.home() / ".cache" / "tas_copilot" / "embeddings"


def _rchat_embedding_config() -> tuple[str, str, str | None]:
    """NIST rchat embedding endpoint config from the environment."""
    base_url = os.environ.get("NIST_VLLM_BASE_URL", "")
    model = os.environ.get("NIST_VLLM_EMBEDDING_MODEL", "")
    api_key = os.environ.get("NIST_VLLM_API_KEY")
    return base_url, model, api_key


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_cache_path(model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return _EMBED_CACHE_DIR / f"{safe_model}.json"


def _load_embed_cache(model: str) -> dict[str, list[float]]:
    path = _embed_cache_path(model)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_embed_cache(model: str, cache: dict[str, list[float]]) -> None:
    path = _embed_cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass  # cache is best-effort; never break search on cache write failure


def _dense_score_cached(
    query: str,
    texts: list[str],
    keys: list[str],
) -> list[float]:
    """Dense cosine similarity via rchat embeddings, cached per chunk key.

    `keys` are stable per-chunk identifiers (chunk_hash). Chunk embeddings are
    cached on disk keyed by (model, chunk_hash); only uncached chunks and the
    query are sent to rchat each call. Returns 0.0 for all chunks if the
    endpoint/model is unconfigured, so keyword+graph still carry the search.
    """
    if not texts:
        return []
    base_url, model, api_key = _rchat_embedding_config()
    if not base_url or not model:
        return [0.0] * len(texts)

    cache = _load_embed_cache(model)

    # 1) Embed the query (never cached — it changes every search)
    try:
        query_vec = openai_compatible_embeddings(base_url, model, [query], api_key=api_key)
    except Exception:
        return [0.0] * len(texts)
    if not query_vec:
        return [0.0] * len(texts)
    q = query_vec[0]

    # 2) Find which chunks need embedding (cache miss)
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        batch_size = 32
        for start in range(0, len(missing_idx), batch_size):
            batch_positions = missing_idx[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_positions]
            try:
                embs = openai_compatible_embeddings(base_url, model, batch_texts, api_key=api_key)
            except Exception:
                embs = []
            for pos, emb in zip(batch_positions, embs):
                cache[keys[pos]] = emb
        _save_embed_cache(model, cache)

    # 3) Cosine similarity from cache (fall back to 0.0 for any still-missing)
    return [
        _cosine(q, cache[keys[i]]) if keys[i] in cache else 0.0
        for i in range(len(texts))
    ]


def ingest_pilot_corpus(arguments: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(arguments["manifest"]).expanduser()
    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments.get("output_dir") or catalog_dir / "ingest_report").expanduser()
    catalog_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        return _missing("corpus-ingest-pilot", output_dir, ["manifest"], f"Manifest does not exist: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    corpus_id = manifest.get("corpus_id") or "pilot_corpus"
    documents = [_document_record(item, manifest_path) for item in manifest.get("documents", [])]
    JsonlCatalog(catalog_dir / "documents.jsonl", Document).save(documents)
    warnings = [warning for document in documents for warning in document.metadata.get("warnings", [])]
    fixture_reports = []
    for fixture in manifest.get("fixtures", []) or []:
        fixture_path = Path(fixture.get("path") or "").expanduser()
        if not fixture_path.is_absolute():
            cwd_path = Path.cwd() / fixture_path
            fixture_path = cwd_path if cwd_path.exists() else manifest_path.parent / fixture_path
        if fixture_path.exists():
            fixture_reports.append(load_corpus_fixtures(fixture_path, catalog_dir))
        else:
            warnings.append(f"Fixture path missing: {fixture_path}")
    local_chunks, local_warnings = _extract_manifest_document_chunks(documents, manifest)
    warnings.extend(local_warnings)
    if local_chunks:
        existing_chunks = _load_jsonl_dicts(catalog_dir / "source_chunks.jsonl")
        _write_jsonl(catalog_dir / "source_chunks.jsonl", _dedupe_records(existing_chunks + local_chunks, "chunk_id"))
    graph = build_evidence_graph(catalog_dir)
    ontology_links = link_corpus_ontology(catalog_dir)
    run = CorpusIngestRun(
        run_id=f"ingest-{_hash(str(manifest_path))}",
        corpus_id=corpus_id,
        manifest_path=str(manifest_path),
        status="ok",
        document_count=len(documents),
        chunk_count=corpus_fixture_report(catalog_dir).get("source_chunks", 0),
        warnings=warnings,
        provenance=[ProvenanceRecord(source_type="corpus_manifest", source_id=str(manifest_path), method="ingest_pilot_corpus")],
    )
    JsonlCatalog(catalog_dir / "ingest_runs.jsonl", CorpusIngestRun).save([run])
    report = {
        "workflow_name": "corpus-ingest-pilot",
        "status": "ok",
        "corpus_id": corpus_id,
        "manifest": str(manifest_path),
        "catalog_dir": str(catalog_dir),
        "documents": len(documents),
        "fixture_reports": fixture_reports,
        "graph": graph,
        "ontology_links": ontology_links,
        "warnings": warnings,
        "separation_policy": "document-side JSONL corpus catalog; TAS data-file SQLite catalogs remain separate",
        "provenance": [item.model_dump(mode="json") for item in run.provenance],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_ingest_markdown, json_filename="corpus_pilot_ingest_report.json", markdown_filename="corpus_pilot_ingest_report.md")
    return report


def ingest_extractor_run(arguments: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(arguments["extractor_run_dir"]).expanduser()
    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments.get("output_dir") or catalog_dir / "extractor_run_ingest_report").expanduser()
    preferred_backend = str(arguments.get("preferred_backend") or "docling")
    max_chars = int(arguments.get("max_chars_per_document") or 250000)
    append = bool(arguments.get("append"))  # merge into an existing catalog: replace only the papers in this run
    catalog_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        return _missing("corpus-ingest-extractor-run", output_dir, ["extractor_run_dir"], f"Extractor run directory does not exist: {run_dir}")
    docling_rows = _load_json(run_dir / "outputs" / "docling" / "summary.json")
    grobid_rows = _load_json(run_dir / "outputs" / "grobid" / "summary.json")
    skipped_sidecar_rows = sum(1 for row in [*docling_rows, *grobid_rows] if _is_macos_sidecar_row(row))
    docling_rows = [row for row in docling_rows if not _is_macos_sidecar_row(row)]
    grobid_rows = [row for row in grobid_rows if not _is_macos_sidecar_row(row)]
    if not docling_rows and not grobid_rows:
        return _missing("corpus-ingest-extractor-run", output_dir, ["extractor_summaries"], f"No GROBID or Docling summary.json files found under: {run_dir}")
    grobid_by_name = {str(row.get("pdf_name") or Path(str(row.get("pdf") or "")).name): row for row in grobid_rows}
    documents: list[Document] = []
    papers: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    warnings = [
        "Imported extractor outputs may contain private/copyrighted derived text; keep the resulting catalog under ignored local or Vista output paths.",
        "Chunks are evidence candidates only. Validate methods, captions, and source support before using them for scientific claims.",
    ]
    if skipped_sidecar_rows:
        warnings.append(f"Skipped {skipped_sidecar_rows} macOS AppleDouble sidecar extractor rows.")
    source_rows = docling_rows if preferred_backend == "docling" and docling_rows else grobid_rows
    for index, row in enumerate(source_rows, start=1):
        pdf_name = str(row.get("pdf_name") or Path(str(row.get("pdf") or f"document_{index}.pdf")).name)
        document_id = f"doc-{_hash(pdf_name)}"
        paper_id = document_id
        grobid_row = grobid_by_name.get(pdf_name, {})
        grobid_metadata = _grobid_bibliographic_metadata(run_dir, grobid_row)
        text_path, method, parser_version = _extractor_text_path(run_dir, row, preferred_backend)
        if text_path is None and preferred_backend == "docling":
            text_path, method, parser_version = _extractor_text_path(run_dir, grobid_row, "grobid")
        if text_path is None:
            warnings.append(f"{pdf_name}: no readable {preferred_backend} text artifact found; recorded document metadata only.")
            text = ""
            source_hash = None
            status = "missing_extracted_text"
        else:
            text = _read_extractor_text(text_path, preferred_backend)[:max_chars]
            source_hash = _file_hash(text_path)
            status = "extracted_text_available"
        title = grobid_metadata.get("title") or _title_from_pdf_name(pdf_name)
        document = Document(
            document_id=document_id,
            document_type="paper",
            title=title,
            paper_id=paper_id,
            local_path=str(row.get("pdf") or grobid_row.get("pdf") or pdf_name),
            source_hash=source_hash,
            copyright_status="private_local_only",
            ingest_status=status,
            extraction_method=method,
            parser_version=parser_version,
            metadata={
                "pdf_name": pdf_name,
                "extractor_run_dir": str(run_dir),
                "preferred_backend": preferred_backend,
                "docling_status": row.get("status") if preferred_backend == "docling" else None,
                "grobid_status": grobid_row.get("status"),
                "docling_bt7_terms": row.get("bt7_terms") if preferred_backend == "docling" else None,
                "grobid_bt7_terms": grobid_row.get("bt7_terms"),
                "doi": grobid_metadata.get("doi"),
                "year": grobid_metadata.get("year"),
                "year_source": grobid_metadata.get("year_source"),
                "authors": grobid_metadata.get("authors", []),
                "grobid_metadata_status": grobid_metadata.get("status"),
                "private_local_only": True,
                "warnings": [
                    "Publication year is bibliographic metadata, not data-acquisition evidence.",
                    "GROBID metadata is deterministic extractor output and remains proposed until reviewed.",
                ],
            },
            provenance=[ProvenanceRecord(source_type="extractor_run", source_id=str(run_dir), method="ingest_extractor_run")],
        )
        documents.append(document)
        papers.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": grobid_metadata.get("authors", []),
                "year": grobid_metadata.get("year"),
                "doi": grobid_metadata.get("doi"),
                "metadata": {
                    "pdf_name": pdf_name,
                    "document_id": document_id,
                    "source": "extractor_run_import",
                    "private_local_derived_text": True,
                    "year_source": grobid_metadata.get("year_source"),
                    "grobid_metadata_status": grobid_metadata.get("status"),
                },
                "provenance": [ProvenanceRecord(source_type="document", source_id=document_id, method="ingest_extractor_run").model_dump(mode="json")],
            }
        )
        if text:
            # Docling's markdown escapes "<" and ">" as HTML entities; a chunk that
            # reads "T = 12 K &gt; TN" shows the entity to reviewers and never matches
            # a model quote written with ">". Unescape once, on the way in.
            text = html.unescape(text)
            chunks.extend(_chunks_from_text(document, text, method=method or "extractor_run_import"))
    replaced_existing = 0
    if append:
        # Incremental update: keep every existing record whose paper is not in this run.
        new_ids = {paper["paper_id"] for paper in papers}
        existing_documents = [doc for doc in JsonlCatalog(catalog_dir / "documents.jsonl", Document).load() if str(doc.paper_id or doc.document_id) not in new_ids and str(doc.document_id) not in new_ids]
        existing_papers = [row for row in _load_jsonl_dicts(catalog_dir / "papers.jsonl") if str(row.get("paper_id")) not in new_ids]
        existing_chunks = [row for row in _load_jsonl_dicts(catalog_dir / "source_chunks.jsonl") if str(row.get("paper_id") or row.get("document_id")) not in new_ids and str(row.get("document_id")) not in new_ids]
        replaced_existing = len(_load_jsonl_dicts(catalog_dir / "papers.jsonl")) - len(existing_papers)
        documents = existing_documents + documents
        papers = existing_papers + papers
        chunks = existing_chunks + chunks
        warnings.append(f"Append mode: merged into the existing catalog; {replaced_existing} existing paper(s) replaced, {len(existing_papers)} kept. paper_facts.jsonl is untouched; run facts for the new paper ids.")
    JsonlCatalog(catalog_dir / "documents.jsonl", Document).save(documents)
    _write_jsonl(catalog_dir / "papers.jsonl", papers)
    if chunks:
        _write_jsonl(catalog_dir / "source_chunks.jsonl", _dedupe_records(chunks, "chunk_id"))
    graph = build_evidence_graph(catalog_dir)
    ontology_links = link_corpus_ontology(catalog_dir)
    report = {
        "workflow_name": "corpus-ingest-extractor-run",
        "status": "ok",
        "extractor_run_dir": str(run_dir),
        "catalog_dir": str(catalog_dir),
        "preferred_backend": preferred_backend,
        "append": append,
        "replaced_existing": replaced_existing,
        "documents": len(documents),
        "source_chunks": len(chunks),
        "docling_rows": len(docling_rows),
        "grobid_rows": len(grobid_rows),
        "skipped_sidecar_rows": skipped_sidecar_rows,
        "graph": graph,
        "ontology_links": {"status": ontology_links.get("status"), "link_count": ontology_links.get("link_count"), "unmapped_chunks": ontology_links.get("unmapped_chunks")},
        "warnings": warnings,
        "separation_policy": "document-side JSONL corpus catalog; TAS data-file SQLite catalogs remain separate",
        "provenance": [ProvenanceRecord(source_type="extractor_run", source_id=str(run_dir), method="ingest_extractor_run").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_extractor_ingest_markdown, json_filename="extractor_run_ingest_report.json", markdown_filename="extractor_run_ingest_report.md")
    return report


def corpus_pilot_report(catalog: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    catalog_dir = Path(catalog).expanduser()
    base = corpus_fixture_report(catalog_dir)
    documents = _load_jsonl_dicts(catalog_dir / "documents.jsonl")
    chunks = _source_chunks(catalog_dir)
    graph_nodes = _load_jsonl_dicts(catalog_dir / "evidence_graph_nodes.jsonl")
    graph_edges = _load_jsonl_dicts(catalog_dir / "evidence_graph_edges.jsonl")
    ontology_links = _load_jsonl_dicts(catalog_dir / "ontology_links.jsonl")
    chunk_types: dict[str, int] = {}
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type") or "unknown"
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    missing_local = [item for item in documents if item.get("ingest_status") == "missing_local_path"]
    report = {
        **base,
        "documents": len(documents),
        "document_types": _counts(item.get("document_type") or item.get("type") for item in documents),
        "post_2021_publications": len([item for item in documents if _document_year(item) and _document_year(item) > 2021]),
        "ncnr_timeline_policy": "For NCNR/BT-7 papers published after the Feb. 3, 2021 NCNR reactor incident, publication year must not be used as data-acquisition year evidence without explicit methods/provenance support.",
        "chunk_types": chunk_types,
        "claim_count": len(_load_jsonl_dicts(catalog_dir / "claims.jsonl")),
        "paper_constraints": len(_load_jsonl_dicts(catalog_dir / "paper_constraints.jsonl")),
        "tables": len(_load_jsonl_dicts(catalog_dir / "tables.jsonl")),
        "evidence_graph_nodes": len(graph_nodes),
        "evidence_graph_edges": len(graph_edges),
        "tas_concept_links": len(ontology_links),
        "unmapped_chunks": [chunk.get("chunk_id") for chunk in chunks if not any(link.get("chunk_id") == chunk.get("chunk_id") for link in ontology_links)],
        "supernode_warnings": ontology_diagnostics({"catalog": str(catalog_dir)}).get("warnings", []),
        "missing_local_files": [item.get("local_path") for item in missing_local],
        "extraction_warnings": [warning for item in documents for warning in item.get("metadata", {}).get("warnings", [])],
        "known_limitations": [
            "Pilot catalog uses synthetic fixtures or capped local-private text extraction only.",
            "No OCR, LLM extraction, or full corpus ingestion is performed.",
            "Private local PDF chunks are output artifacts and must not be committed.",
            "Evidence graph edges are deterministic fixture/catalog edges, not LLM-generated graph construction.",
        ],
    }
    if output_dir:
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = write_report_bundle(output, report, markdown_renderer=_report_markdown, json_filename="corpus_pilot_report.json", markdown_filename="corpus_pilot_report.md")
    return report


def paper_candidate_shortlist(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    top_k = int(arguments.get("top_k", 10))
    quality_report_path = arguments.get("quality_report")
    quality_by_pdf = _quality_by_pdf(Path(quality_report_path).expanduser()) if quality_report_path else {}
    documents = _load_jsonl_dicts(catalog_dir / "documents.jsonl")
    chunks = _source_chunks(catalog_dir)
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or id(chunk))
        for key in [chunk.get("document_id"), chunk.get("paper_id"), chunk.get("source_id")]:
            if key:
                bucket = chunks_by_document.setdefault(str(key), [])
                if not any(str(existing.get("chunk_id") or id(existing)) == chunk_id for existing in bucket):
                    bucket.append(chunk)
    candidates = []
    for document in documents:
        document_id = str(document.get("document_id") or document.get("paper_id"))
        paper_id = str(document.get("paper_id") or document_id)
        metadata = document.get("metadata") or {}
        pdf_name = str(metadata.get("pdf_name") or Path(str(document.get("local_path") or document_id)).name)
        doc_chunks = chunks_by_document.get(document_id, []) + [chunk for chunk in chunks_by_document.get(paper_id, []) if chunk not in chunks_by_document.get(document_id, [])]
        chunk_type_counts = _counts(chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type") for chunk in doc_chunks)
        text_summary = " ".join([chunk.get("title") or "" for chunk in doc_chunks])
        bt7_chunk_hits = sum(len(re.findall(r"\bbt-?7\b", (chunk.get("text") or "").lower())) for chunk in doc_chunks)
        quality = quality_by_pdf.get(pdf_name, {})
        score_breakdown = {
            "bt7_metadata": 3.0 if int(metadata.get("docling_bt7_terms") or metadata.get("grobid_bt7_terms") or quality.get("docling_bt7_terms") or quality.get("grobid_bt7_terms") or 0) > 0 else 0.0,
            "bt7_chunk_hits": min(2.0, 0.5 * bt7_chunk_hits),
            "instrument_setup_chunk": 1.5 if chunk_type_counts.get("instrument_setup", 0) else 0.0,
            "figure_caption_chunks": min(2.0, 0.4 * chunk_type_counts.get("figure_caption", 0)),
            "experimental_chunk": 1.0 if chunk_type_counts.get("experimental", 0) else 0.0,
            "data_reduction_chunk": 1.0 if chunk_type_counts.get("data_reduction", 0) else 0.0,
            "grobid_figure_structure": min(1.0, 0.1 * int(quality.get("grobid_figure_tags") or 0)),
            "docling_heading_structure": min(1.0, 0.05 * int(quality.get("docling_headings") or 0)),
        }
        total_score = round(sum(score_breakdown.values()), 3)
        reasons = []
        if score_breakdown["bt7_metadata"] or score_breakdown["bt7_chunk_hits"]:
            reasons.append("BT-7 terms found in extractor metadata or imported chunks")
        if chunk_type_counts.get("instrument_setup", 0):
            reasons.append("instrument/setup chunk exists")
        if chunk_type_counts.get("figure_caption", 0):
            reasons.append("figure-caption chunks exist")
        if chunk_type_counts.get("data_reduction", 0):
            reasons.append("data-reduction chunk exists")
        missing = []
        if not chunk_type_counts.get("instrument_setup", 0):
            missing.append("instrument_setup_chunk")
        if not chunk_type_counts.get("figure_caption", 0):
            missing.append("figure_caption_chunk")
        candidates.append(
            {
                "paper_id": paper_id,
                "document_id": document_id,
                "title": document.get("title"),
                "pdf_name": pdf_name,
                "total_score": total_score,
                "score_breakdown": score_breakdown,
                "chunk_type_counts": chunk_type_counts,
                "bt7_chunk_hits": bt7_chunk_hits,
                "quality_flags": quality.get("quality_flags", []),
                "manual_curation_reasons": reasons,
                "missing_for_curation": missing,
                "chunk_ids_by_type": {
                    chunk_type: [chunk.get("chunk_id") for chunk in doc_chunks if (chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type")) == chunk_type]
                    for chunk_type in sorted(chunk_type_counts)
                },
                "private_local_only": bool(metadata.get("private_local_only")),
                "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="paper_candidate_shortlist").model_dump(mode="json")],
            }
        )
    ranked = sorted(candidates, key=lambda item: item["total_score"], reverse=True)
    report = {
        "workflow_name": "corpus-paper-candidate-shortlist",
        "status": "ok",
        "catalog": str(catalog_dir),
        "quality_report": str(quality_report_path) if quality_report_path else None,
        "document_count": len(documents),
        "candidate_count": len(candidates),
        "top_k": top_k,
        "top_candidates": ranked[:top_k],
        "safe_conclusion": "This shortlist ranks papers for manual corpus curation. It is not source-data linkage and does not make scientific claims.",
        "warnings": [
            "Private/local derived corpus records may contain copyrighted text in the catalog; keep catalog and reports under ignored local output.",
            "Scores are deterministic triage heuristics for manual review priority, not probabilities of source-data correctness.",
        ],
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="paper_candidate_shortlist").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_shortlist_markdown, json_filename="paper_candidate_shortlist.json", markdown_filename="paper_candidate_shortlist.md")
    return report


def extract_paper_constraints(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    paper_id_filter = arguments.get("paper_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = _load_jsonl_dicts(catalog_dir / "papers.jsonl")
    documents = _load_jsonl_dicts(catalog_dir / "documents.jsonl")
    paper_titles = {
        str(item.get("paper_id") or item.get("document_id")): item.get("title")
        for item in [*documents, *papers]
        if item.get("paper_id") or item.get("document_id")
    }
    chunks = _source_chunks(catalog_dir)
    chunks_by_paper: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        for key in [chunk.get("paper_id"), chunk.get("document_id"), chunk.get("source_id")]:
            if key:
                bucket = chunks_by_paper.setdefault(str(key), [])
                chunk_id = str(chunk.get("chunk_id") or id(chunk))
                if not any(str(existing.get("chunk_id") or id(existing)) == chunk_id for existing in bucket):
                    bucket.append(chunk)
    target_ids = [str(paper_id_filter)] if paper_id_filter else sorted(chunks_by_paper)
    records = []
    warnings = [
        "Constraints are deterministic candidates extracted from local/private chunks. They are proposed evidence, not validated paper facts.",
        "Reports intentionally omit source text snippets; inspect source chunks manually before scientific use.",
    ]
    for paper_id in target_ids:
        paper_chunks = chunks_by_paper.get(paper_id, [])
        if not paper_chunks:
            continue
        records.append(_paper_constraint_record(catalog_dir, paper_id, paper_titles.get(paper_id), paper_chunks))
    existing = [item for item in _load_jsonl_dicts(catalog_dir / "paper_constraints.jsonl") if paper_id_filter and item.get("paper_id") != paper_id_filter]
    _write_jsonl(catalog_dir / "paper_constraints.jsonl", _dedupe_records(existing + records, "constraint_id"))
    report = {
        "workflow_name": "corpus-extract-paper-constraints",
        "status": "ok",
        "catalog": str(catalog_dir),
        "paper_id": paper_id_filter,
        "paper_count": len(records),
        "constraints": records,
        "warnings": warnings,
        "safe_conclusion": "Use these constraints to propose corpus/data linkage searches. Do not treat them as accepted claims without source-support validation or human review.",
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="extract_paper_constraints").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_constraints_markdown, json_filename="paper_constraints_report.json", markdown_filename="paper_constraints_report.md")
    return report


def corpus_search(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["catalog"]).expanduser()
    query = str(arguments.get("query") or "")
    top_k = int(arguments.get("top_k", 10))
    use_dense = bool(arguments.get("use_dense", True))
    chunks = _source_chunks(catalog_dir)
    documents = _load_jsonl_dicts(catalog_dir / "documents.jsonl")
    document_types = {
        str(item.get("document_id") or item.get("paper_id") or item.get("id")): str(item.get("document_type") or item.get("type") or "")
        for item in documents
    }
    ontology_links = _load_jsonl_dicts(catalog_dir / "ontology_links.jsonl")
    concepts_by_chunk: dict[str, set[str]] = {}
    for link in ontology_links:
        concepts_by_chunk.setdefault(str(link.get("chunk_id")), set()).add(str(link.get("concept_id")))
    source_support = _load_jsonl_dicts(catalog_dir / "source_support.jsonl")
    supported_sources = {
        str(item.get("source_object_id") or item.get("claim_id") or item.get("panel_id"))
        for item in source_support
        if str(item.get("status") or item.get("support_status") or "").lower() == "supported"
    }
    concepts = load_ontology(CoreConfig().ontology_path)
    concepts_by_id = {concept.id: concept for concept in concepts}
    query_concept_ids = {concept.id for concept in match_concepts(query, concepts)}
    query_terms = _terms(query)
    # Prepare texts for scoring
    chunk_texts = [
        " ".join([chunk.get("title") or "", chunk.get("text") or "", json.dumps(chunk.get("metadata") or {})])
        for chunk in chunks
    ]
    # Keyword-based scoring (IDF-weighted)
    doc_freq = _document_frequencies(chunk_texts)
    n_docs = max(1, len(chunk_texts))
    keyword_scores = []
    for idx, chunk in enumerate(chunks):
        text = chunk_texts[idx].lower()
        matched_terms = [t for t in query_terms if t in text]
        # IDF-weighted term score: rare, meaningful terms dominate
        term_score = sum(_idf(t, doc_freq, n_docs) for t in matched_terms)
        matched_concepts = [
            concept.id for concept in match_concepts(text, concepts)
            if concept.id in query_concept_ids
            or any(term.lower() in query.lower() for term in concept.keywords)
        ]
        weighted_concepts = [
            cid for cid in matched_concepts
            if _concept_retrieval_weight(concepts_by_id.get(cid)) > 0.0
        ]
        score = term_score + sum(
            _concept_retrieval_weight(concepts_by_id.get(cid)) for cid in weighted_concepts
        )
        keyword_scores.append(score if score > 0 else 0.0)
    # Dense scoring (if requested)
    if use_dense:
        chunk_keys = [
            str(chunk.get("chunk_hash") or chunk.get("chunk_id") or idx)
            for idx, chunk in enumerate(chunks)
        ]
        dense_scores = _dense_score_cached(query, chunk_texts, chunk_keys)
    else:
        dense_scores = [0.0] * len(chunks)

    # Normalize keyword scores to [0, 1] so IDF magnitude (unbounded) does not
    # swamp dense cosine similarity (0-1) when the two channels are combined.
    max_kw = max(keyword_scores) if keyword_scores else 0.0
    norm_keyword = [
        (score / max_kw if max_kw > 0 else 0.0) for score in keyword_scores
    ]

    # Combine normalized keyword scores with dense cosine (both now in [0, 1]).
    ALPHA = 0.6  # weight for keyword
    BETA = 0.4   # weight for dense
    combined_scores = [
        ALPHA * ks + BETA * ds for ks, ds in zip(norm_keyword, dense_scores)
    ]
    results: list[RetrievalResult] = []
    for idx, chunk in enumerate(chunks):
        if combined_scores[idx] == 0.0:
            continue
        # Re-scan for injection patterns at query time
        is_suspicious, matched_patterns = scan_for_injection(chunk_texts[idx])
        # Build snippet with highlighted terms
        snippet_raw = chunk.get("text") or ""
        highlighted = snippet_raw
        for term in query_terms:
            if term:
                highlighted = highlighted.replace(
                    term, f"**{term}**"
                ).replace(
                    term.title(), f"**{term.title()}**"
                ).replace(
                    term.upper(), f"**{term.upper()}**"
                )
        snippet = (highlighted[:300] + "...") if len(highlighted) > 300 else highlighted
        results.append(
            RetrievalResult(
                source_type="source_chunk",
                source_id=str(chunk.get("chunk_id")),
                title=chunk.get("title"),
                score=combined_scores[idx],
                matched_terms=[t for t in query_terms if t in chunk_texts[idx].lower()],
                matched_concepts=sorted({
                    concept.id
                    for concept in match_concepts(chunk_texts[idx], concepts)
                    if concept.id in query_concept_ids
                    or any(term.lower() in query.lower() for term in concept.keywords)
                }),
                snippet=snippet,
                structured_summary={
                    "paper_id": chunk.get("paper_id") or chunk.get("source_id"),
                    "document_id": chunk.get("document_id"),
                    "chunk_type": chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type"),
                    "figure_id": chunk.get("figure_id") or chunk.get("metadata", {}).get("figure_id"),
                    "panel_id": chunk.get("panel_id"),
                    "private_local_only": bool((chunk.get("metadata") or {}).get("private_local_only")),
                    "suspected_injection": is_suspicious,
                    # NEW: show corpus catalog path for traceability
                    "catalog_path": str(catalog_dir),
                },
                provenance=[ProvenanceRecord(**item) for item in chunk.get("provenance", [])],
                retrieval_method="hybrid_keyword_dense" if use_dense else "keyword_plus_tas_concept",
                confidence=min(1.0, combined_scores[idx] / max(1, len(query_terms))),
            )
        )
    # Typed-graph channel: node labels, underlying chunk text, review-status
    # weighting, and one-hop paths (same GraphRetriever as HybridRetriever).
    # Graph hits that resolve to an object the keyword channel already found
    # boost that result instead of duplicating it, mirroring HybridRetriever.
    by_key = {(result.source_type, result.source_id): result for result in results}
    # Use stopword-filtered terms for the graph channel so it does not match
    # paper titles purely on words like "and", "is", "its", "properties".
    graph_query = " ".join(sorted(_terms(query)))
    if graph_query.strip():
        graph_hits = [
            hit
            for hit in GraphRetriever(catalog_dir).search(graph_query, limit=max(top_k * 2, 10))
            if _passes_graph_filters(hit, arguments)
        ]
        # Normalize the graph branch like every other branch (and like
        # HybridRetriever): raw traversal scores are unbounded, while the
        # chunk scores above live in [0, 1] - unnormalized graph hits would
        # bury actual evidence chunks under graph metadata nodes.
        max_graph = max((hit.score for hit in graph_hits), default=0.0)
        for hit in graph_hits:
            normalized = (hit.score / max_graph) if max_graph > 0 else 0.0
            key = (hit.source_type, hit.source_id)
            if key in by_key:
                by_key[key].score += 0.3 * normalized
            else:
                hit.score = 0.3 * normalized
                by_key[key] = hit
    results = sorted(by_key.values(), key=lambda item: item.score, reverse=True)[:top_k]
    run = CorpusRetrievalRun(
        run_id=f"retrieval-{_hash(query)}",
        corpus_id=None,
        query=query,
        status="ok",
        result_count=len(results),
        provenance=[ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="corpus_search")],
    )
    return {
        "workflow_name": "corpus-search",
        "status": "ok",
        "query": query,
        "top_k": top_k,
        "results": [result.model_dump(mode="json") for result in results],
        "warnings": _broad_query_warnings(query) + (
            ["Dense vector search enabled via NIST rchat embeddinggemma-300m."]
            if use_dense and all(_rchat_embedding_config()[:2]) else []
        ),
    }


def ontology_diagnostics(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = arguments.get("output_dir")
    chunks = _source_chunks(catalog_dir)
    concepts = load_ontology(CoreConfig().ontology_path)
    warnings = supernode_diagnostics([chunk.get("text", "") for chunk in chunks], concepts, max_fraction=float(arguments.get("max_fraction", 0.6)))
    coverage = link_corpus_ontology(catalog_dir)
    broad = [item for item in coverage.get("links", []) if item.get("broad_warning")]
    report = {
        "workflow_name": "corpus-ontology-diagnostics",
        "status": "ok",
        "catalog_dir": str(catalog_dir),
        "chunk_count": len(chunks),
        "concept_link_count": coverage.get("link_count", 0),
        "warnings": warnings + [{"code": "broad_concept_link", **item} for item in broad],
        "unmapped_chunks": coverage.get("unmapped_chunks", []),
    }
    if output_dir:
        report["artifacts"] = write_report_bundle(Path(output_dir).expanduser(), report, markdown_renderer=_diagnostics_markdown, json_filename="ontology_diagnostics.json", markdown_filename="ontology_diagnostics.md")
    return report


def build_evidence_graph(catalog: str | Path) -> dict[str, Any]:
    catalog_dir = Path(catalog).expanduser()
    nodes: dict[str, EvidenceGraphNode] = {}
    edges: dict[str, EvidenceGraphEdge] = {}
    prov = [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="build_evidence_graph")]

    def add_node(node_id: str, node_type: str, label: str | None = None, source_object_id: str | None = None) -> None:
        nodes.setdefault(node_id, EvidenceGraphNode(node_id=node_id, node_type=node_type, label=label, source_object_id=source_object_id or node_id, provenance=prov))

    def add_edge(src: str, dst: str, edge_type: str, confidence: float = 0.8, status: str = "proposed", evidence: list[str] | None = None) -> None:
        edge_id = f"{edge_type.lower()}-{_hash(src + dst + edge_type)}"
        edges.setdefault(edge_id, EvidenceGraphEdge(edge_id=edge_id, source_node_id=src, target_node_id=dst, edge_type=edge_type, confidence=confidence, status=status, evidence=evidence or [], provenance=prov, created_by="corpus_pilot"))

    for paper in _load_jsonl_dicts(catalog_dir / "papers.jsonl"):
        paper_node = f"Paper:{paper['paper_id']}"
        add_node(paper_node, "Paper", paper.get("title"), paper["paper_id"])
    for chunk in _source_chunks(catalog_dir):
        chunk_node = f"SourceChunk:{chunk['chunk_id']}"
        add_node(chunk_node, "SourceChunk", chunk.get("title"), chunk["chunk_id"])
        paper_id = chunk.get("paper_id") or chunk.get("source_id")
        if paper_id:
            paper_node = f"Paper:{paper_id}"
            add_node(paper_node, "Paper", paper_id, paper_id)
            add_edge(paper_node, chunk_node, "HAS_CHUNK", 0.95, "accepted")
    for figure in _load_jsonl_dicts(catalog_dir / "figures.jsonl"):
        figure_node = f"Figure:{figure['figure_id']}"
        add_node(figure_node, "Figure", figure.get("label"), figure["figure_id"])
        add_edge(f"Paper:{figure['paper_id']}", figure_node, "HAS_FIGURE", 0.95, "accepted")
    for panel in _load_jsonl_dicts(catalog_dir / "figure_panels.jsonl"):
        panel_node = f"FigurePanel:{panel['figure_id']}:{panel['panel_id']}"
        add_node(panel_node, "FigurePanel", (panel.get("panel_label") or {}).get("label_text"), panel["panel_id"])
        add_edge(f"Figure:{panel['figure_id']}", panel_node, "HAS_PANEL", 0.95, "accepted")
    for support in _load_jsonl_dicts(catalog_dir / "source_support.jsonl"):
        add_node(f"Claim:{support['claim_id']}", "Claim", support.get("evidence_summary"), support["claim_id"])
        add_edge(f"SourceChunk:{support['source_object_id']}", f"Claim:{support['claim_id']}", "SUPPORTS_CLAIM", 0.9, support.get("status", "proposed"), [support.get("evidence_summary", "")])
    for mapping in _load_jsonl_dicts(catalog_dir / "mappings.jsonl"):
        add_node(f"Experiment:{mapping['experiment_id']}", "Experiment", mapping["experiment_id"], mapping["experiment_id"])
        add_edge(f"Paper:{mapping['paper_id']}", f"Experiment:{mapping['experiment_id']}", "CANDIDATE_MAPS_TO_EXPERIMENT", float(mapping.get("confidence", 0.5)), mapping.get("status", "proposed"), mapping.get("evidence", []))
    for link in _load_jsonl_dicts(catalog_dir / "panel_source_links.jsonl"):
        panel_node = f"FigurePanel:{link['figure_id']}:{link['panel_id']}"
        source_node = f"{link['source_object_type']}:{link['source_object_id']}"
        add_node(source_node, link["source_object_type"], link["source_object_id"], link["source_object_id"])
        if link.get("status") == "accepted":
            edge_type = "ACCEPTED_SOURCE_FOR_PANEL"
        elif link.get("status") == "rejected":
            edge_type = "REJECTED_AS_SOURCE_FOR_PANEL"
        else:
            edge_type = "CANDIDATE_SOURCE_FOR_PANEL"
        add_edge(panel_node, source_node, edge_type, float(link.get("confidence", 0.5)), link.get("status", "candidate"))
    JsonlCatalog(catalog_dir / "evidence_graph_nodes.jsonl", EvidenceGraphNode).save(nodes.values())
    JsonlCatalog(catalog_dir / "evidence_graph_edges.jsonl", EvidenceGraphEdge).save(edges.values())
    return {"status": "ok", "node_count": len(nodes), "edge_count": len(edges)}


def link_corpus_ontology(catalog: str | Path) -> dict[str, Any]:
    catalog_dir = Path(catalog).expanduser()
    concepts = load_ontology(CoreConfig().ontology_path)
    links: list[dict[str, Any]] = []
    for chunk in _source_chunks(catalog_dir):
        matches = match_concepts(" ".join([chunk.get("title") or "", chunk.get("text") or ""]), concepts)
        for concept in matches:
            label = concept.label.lower()
            links.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "concept_id": concept.id,
                    "score": 0.8 if not concept.avoid_if_matches_too_broadly else 0.2,
                    "match_method": "keyword",
                    "matched_terms": [term for term in concept.keywords if term.lower() in (chunk.get("text") or "").lower()],
                    "confidence": 0.8 if not concept.avoid_if_matches_too_broadly else 0.2,
                    "broad_warning": concept.avoid_if_matches_too_broadly or label in BROAD_CONCEPT_LABELS,
                    "provenance": [ProvenanceRecord(source_type="source_chunk", source_id=chunk["chunk_id"], method="tas_ontology_keyword_link").model_dump(mode="json")],
                }
            )
    _write_jsonl(catalog_dir / "ontology_links.jsonl", links)
    return {"status": "ok", "link_count": len(links), "links": links, "unmapped_chunks": [chunk["chunk_id"] for chunk in _source_chunks(catalog_dir) if not any(link["chunk_id"] == chunk["chunk_id"] for link in links)]}


def paper_brief(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["corpus_catalog"]).expanduser()
    paper_id = arguments["paper_id"]
    output_dir = Path(arguments["output_dir"]).expanduser()
    papers = _load_jsonl_dicts(catalog_dir / "papers.jsonl")
    paper = next((item for item in papers if item.get("paper_id") == paper_id), None)
    if paper is None:
        return _missing("paper-brief", output_dir, ["paper_id"], f"Paper not found: {paper_id}")
    chunks = [chunk for chunk in _source_chunks(catalog_dir) if (chunk.get("paper_id") or chunk.get("source_id")) == paper_id]
    figures = [item for item in _load_jsonl_dicts(catalog_dir / "figures.jsonl") if item.get("paper_id") == paper_id]
    panels = [panel for panel in _load_jsonl_dicts(catalog_dir / "figure_panels.jsonl") if any(panel.get("figure_id") == figure.get("figure_id") for figure in figures)]
    mappings = [item for item in _load_jsonl_dicts(catalog_dir / "mappings.jsonl") if item.get("paper_id") == paper_id]

    # Build content excerpts from actual chunks so the brief has real substance.
    chunk_previews = []
    for c in chunks[:6]:
        ctype = c.get("chunk_type") or (c.get("metadata") or {}).get("chunk_type") or "unknown"
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        if text:
            chunk_previews.append({"chunk_type": ctype, "text": text})

    report = {
        "workflow_name": "paper-brief",
        "status": "ok",
        "paper": paper,
        "paper_title": paper.get("title"),
        "paper_year": paper.get("year") or (paper.get("metadata") or {}).get("year"),
        "chunk_count": len(chunks),
        "chunks_by_type": _counts((chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type") or "unknown") for chunk in chunks),
        "chunk_previews": chunk_previews,
        "figures": figures,
        "panels": panels,
        "candidate_experiment_mappings": mappings,
        "source_support": validate_source_support(catalog_dir),
        "safe_next_actions": ["Review source chunks before using paper claims.", "Keep candidate experiment mappings proposed until human review."],
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="paper_brief").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_paper_brief_markdown,
        json_filename="paper_brief.json",
        markdown_filename="paper_brief.md",
    )
    return report


def figure_brief(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["corpus_catalog"]).expanduser()
    figure_id = arguments["figure_id"]
    output_dir = Path(arguments["output_dir"]).expanduser()
    figure = next((item for item in _load_jsonl_dicts(catalog_dir / "figures.jsonl") if item.get("figure_id") == figure_id), None)
    if figure is None:
        return _missing("figure-brief", output_dir, ["figure_id"], f"Figure not found: {figure_id}")
    panels = [item for item in _load_jsonl_dicts(catalog_dir / "figure_panels.jsonl") if item.get("figure_id") == figure_id]
    captions = [item for item in _load_jsonl_dicts(catalog_dir / "figure_captions.jsonl") if item.get("figure_id") == figure_id]
    links = [item for item in _load_jsonl_dicts(catalog_dir / "panel_source_links.jsonl") if item.get("figure_id") == figure_id]
    report = {
        "workflow_name": "figure-brief",
        "status": "ok",
        "figure": figure,
        "captions": captions,
        "panels": panels,
        "candidate_source_links": links,
        "source_support": validate_source_support(catalog_dir),
        "safe_next_actions": ["Treat candidate source links as proposed.", "Run source-triage and reconstruction diagnostics before source-data claims."],
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="figure_brief").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_figure_brief_markdown, json_filename="figure_brief.json", markdown_filename="figure_brief.md")
    return report


def link_paper_to_experiments(arguments: dict[str, Any]) -> dict[str, Any]:
    catalog_dir = Path(arguments["corpus_catalog"]).expanduser()
    paper_id = arguments["paper_id"]
    output_dir = Path(arguments["output_dir"]).expanduser()
    data_catalog = arguments.get("data_catalog")
    hydrated_index = arguments.get("hydrated_index")
    top_k = int(arguments.get("top_k") or 20)
    max_data_candidates = int(arguments.get("max_data_candidates") or 5000)
    hydrate_candidate_clues = bool(arguments.get("hydrate_candidate_clues"))
    hydrate_candidate_limit = int(arguments.get("hydrate_candidate_limit") or 25)
    max_hydrated_files_per_candidate = int(arguments.get("max_hydrated_files_per_candidate") or 12)
    mappings = [item for item in _load_jsonl_dicts(catalog_dir / "mappings.jsonl") if item.get("paper_id") == paper_id]
    paper = _paper_record(catalog_dir, paper_id)
    lag_prior = publication_data_lag_prior(paper_id, _document_year(paper or {}))
    scored_mappings = [_apply_lag_prior_to_mapping(item, lag_prior) for item in mappings]
    constraints = [item for item in _load_jsonl_dicts(catalog_dir / "paper_constraints.jsonl") if item.get("paper_id") == paper_id]
    data_catalog_links: list[dict[str, Any]] = []
    missing_inputs = []
    next_commands = []
    data_catalog_status = {"provided": bool(data_catalog), "path": str(data_catalog) if data_catalog else None, "status": "not_provided"}
    if data_catalog:
        data_catalog_links, data_catalog_status = _score_data_catalog_experiment_links(
            data_catalog,
            paper_id=paper_id,
            paper=paper,
            constraints=constraints,
            lag_prior=lag_prior,
            top_k=top_k,
            max_candidates=max_data_candidates,
            hydrate_candidate_clues=hydrate_candidate_clues,
            hydrate_candidate_limit=hydrate_candidate_limit,
            max_hydrated_files_per_candidate=max_hydrated_files_per_candidate,
            hydrated_index=hydrated_index,
        )
    if not scored_mappings and not data_catalog_links and not data_catalog:
        missing_inputs.append("data_catalog")
        next_commands.append(
            "python -m tas_copilot_core.cli workflow link-paper-to-experiments "
            f"--corpus-catalog {catalog_dir} --data-catalog [BT7_DATA_CATALOG] --paper-id {paper_id} --output-dir [OUTPUT_DIR]"
        )
    elif data_catalog and not data_catalog_links:
        missing_inputs.append("scored_data_catalog_candidates")
    review_label_store = arguments.get("review_label_store") or arguments.get("review_outcomes")
    review_label_diagnostics = review_label_calibration_diagnostics(
        review_label_store,
        object_type="paper_experiment_link",
        source_workflow="link-paper-to-experiments",
    )
    label_priors = label_informed_priors(
        review_label_store,
        object_type="paper_experiment_link",
        source_workflow="link-paper-to-experiments",
    )
    if review_label_store:
        label_priors = dict(label_priors)
        label_priors["consumer_behavior_change"] = "applied_to_unreviewed_paper_experiment_link_ranking_when_channels_are_behavior_ready"
        label_priors["interpretation"] = (
            "Exact reviewed labels override candidate rank. For unreviewed paper-experiment candidates, "
            "behavior-ready label priors calibrate ranking_score while probability_score remains the baseline."
        )
    calibrated_links = _apply_paper_link_review_calibration(
        [*scored_mappings, *data_catalog_links],
        paper_id=paper_id,
        review_label_store=review_label_store,
        label_priors=label_priors,
    )
    report = {
        "workflow_name": "link-paper-to-experiments",
        "status": "ok",
        "paper_id": paper_id,
        "data_catalog": data_catalog_status,
        "publication_data_lag_prior": lag_prior.model_dump(mode="json"),
        "extracted_constraints": constraints,
        "candidate_links": sorted(calibrated_links, key=_paper_link_rank_key, reverse=True)[:top_k],
        "review_label_diagnostics": review_label_diagnostics,
        "label_informed_priors": label_priors,
        "ranking_policy": (
            "Exact reviewed paper-experiment labels override candidate rank. "
            "Unreviewed candidates are ranked by label_calibrated_score when label priors are behavior-ready; "
            "probability_score is preserved as the uncalibrated baseline."
            if review_label_store
            else "Candidates are ranked by uncalibrated probability_score because no review label store was supplied."
        ),
        "missing_inputs": missing_inputs,
        "next_commands": next_commands,
        "safe_conclusion": "Paper-to-experiment links are proposed probability-weighted candidates, not accepted source-data claims. Every score is a ranking probability proxy with explicit components; publication-to-data lag is a weak prior, and catalog/paper/human-review evidence should override it.",
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="link_paper_to_experiments").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_paper_experiment_markdown, json_filename="paper_experiment_links.json", markdown_filename="paper_experiment_links.md")
    return report


def build_paper_experiment_review_dashboard(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a local browser dashboard for human review of paper-experiment links."""

    review_template = Path(arguments["review_template"]).expanduser()
    output_dir = Path(arguments.get("output_dir") or review_template.parent).expanduser()
    assist_path = Path(arguments["review_assist"]).expanduser() if arguments.get("review_assist") else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if not review_template.exists():
        return _missing("paper-experiment-review-dashboard", output_dir, ["review_template"], f"Review template does not exist: {review_template}")
    rows = _load_jsonl_records(review_template)
    assist_rows = _load_review_assist_items(assist_path)
    assist_by_key = {(str(item.get("paper_id")), int(item.get("candidate_rank") or 0)): item for item in assist_rows}
    precomputed_scans = _load_review_scan_records(
        Path(arguments["review_scan_records"]).expanduser()
        if arguments.get("review_scan_records")
        else review_template.parent / "review_packet_scan_records.jsonl"
    )
    records = []
    packet_dir = output_dir / "review_packets"
    max_packet_scans = int(arguments.get("max_packet_scans") or 8)
    paper_base_url = str(arguments.get("paper_base_url") or "/papers").rstrip("/")
    local_runs_root = (Path.cwd() / "local_runs").resolve()
    review_data_catalog = str(arguments.get("data_catalog") or "") or None
    for row in rows:
        rank = int(row.get("candidate_rank") or 0)
        item = assist_by_key.get((str(row.get("paper_id")), rank), {})
        source_report = Path(str(row.get("source_report") or ""))
        source_report_abs = source_report if source_report.is_absolute() else (Path.cwd() / source_report).resolve()
        source_report_md = source_report_abs.with_suffix(".md")
        source_payload = _load_json_object(source_report_abs)
        source_candidate = _paper_link_candidate_from_report(source_payload, str(row.get("candidate_experiment_id") or ""))
        candidate_metadata = source_candidate.get("metadata") if isinstance(source_candidate.get("metadata"), dict) else {}
        hydrated_summary = source_candidate.get("hydrated_clue_summary") or candidate_metadata.get("hydrated_clue_summary") or {}
        historic_summary = source_candidate.get("historic_clue_summary") or candidate_metadata.get("historic_clue_summary") or {}
        paper_constraints = _paper_constraints_for_review(source_payload)
        file_metadata_summary = _file_metadata_summary_for_review(hydrated_summary)
        packet = _write_paper_experiment_review_packet(
            packet_dir,
            row=row,
            candidate=source_candidate,
            source_payload=source_payload,
            paper_constraints=paper_constraints,
            file_metadata_summary=file_metadata_summary,
            max_scan_plots=max_packet_scans,
            paper_base_url=paper_base_url,
            local_runs_root=local_runs_root,
            data_catalog=review_data_catalog,
            precomputed_scan_records=precomputed_scans.get(
                _review_scan_record_key(
                    row.get("paper_id"),
                    row.get("candidate_experiment_id"),
                    row.get("candidate_rank"),
                )
            ),
        )
        records.append(
            {
                "review_id": f"{row.get('paper_id')}::rank-{rank}::{row.get('candidate_experiment_id')}",
                "paper_id": row.get("paper_id"),
                "paper_title": row.get("paper_title"),
                "candidate_experiment_id": row.get("candidate_experiment_id"),
                "candidate_rank": rank,
                "candidate_status": row.get("candidate_status"),
                "confidence_label": row.get("confidence_label"),
                "probability_score": row.get("probability_score"),
                "review_priority": item.get("review_priority") or "unprioritized",
                "human_review_status": row.get("human_review_status") or "unreviewed",
                "allowed_human_review_statuses": row.get("allowed_human_review_statuses") or [],
                "evidence_to_check": row.get("evidence_to_check") or [],
                "warnings": row.get("warnings") or [],
                "cautions": item.get("cautions") or [],
                "review_questions": item.get("review_questions") or [],
                "paper_constraints": paper_constraints,
                "score_breakdown": source_candidate.get("score_breakdown") or {},
                "candidate_metadata": _review_candidate_metadata_summary(source_candidate),
                "historic_clue_summary": historic_summary,
                "hydrated_clue_summary": hydrated_summary,
                "file_metadata_summary": file_metadata_summary,
                "review_packet": packet,
                "paper_open_url": _paper_open_url(row.get("paper_id"), paper_base_url=paper_base_url),
                "source_report_json": str(source_report_abs),
                "source_report_md": str(source_report_md),
                "source_report_json_url": _path_to_web_url(source_report_abs, local_runs_root) if source_report_abs.exists() else "",
                "source_report_md_url": _path_to_web_url(source_report_md, local_runs_root) if source_report_md.exists() else "",
                "reviewer": row.get("reviewer"),
                "review_notes": row.get("review_notes"),
                "reviewed_at": row.get("reviewed_at"),
                "provenance": row.get("provenance") or [],
            }
        )

    priority_counts = _count_values(records, "review_priority")
    status_counts = _count_values(records, "human_review_status")
    review_rows_csv = output_dir / "paper_experiment_review_rows.csv"
    review_summary_json = output_dir / "paper_experiment_review_summary.json"
    _write_paper_experiment_review_rows_csv(review_rows_csv, records)
    review_outcomes_sqlite = _write_paper_review_outcome_store(
        output_dir / "review_outcomes.sqlite",
        records,
        source_path=review_template,
        batch_method="paper_experiment_review_dashboard_dual_write",
    )
    html_path = output_dir / str(arguments.get("html_filename") or "paper_experiment_review_dashboard.html")
    payload = {
        "created_at": _utc_now(),
        "source_review_template": str(review_template),
        "source_review_assist": str(assist_path) if assist_path else None,
        "safe_conclusion": (
            "This page facilitates human review. It does not turn candidate links into accepted source-data "
            "attributions unless a human exports explicit labels."
        ),
        "records": records,
        "diagnostics": {
            "row_count": len(records),
            "priority_counts": priority_counts,
            "status_counts": status_counts,
            "rows_csv": str(review_rows_csv),
            "empty_row_warning": "No candidate rows were loaded; check that the review template JSONL is non-empty." if not records else None,
        },
    }
    review_summary_json.write_text(json.dumps(payload["diagnostics"], indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_paper_experiment_review_dashboard_html(payload), encoding="utf-8")
    report = {
        "workflow_name": "paper-experiment-review-dashboard",
        "status": "ok",
        "review_template": str(review_template),
        "review_assist": str(assist_path) if assist_path else None,
        "review_row_count": len(records),
        "priority_counts": priority_counts,
        "status_counts": status_counts,
        "html": str(html_path),
        "review_rows_csv": str(review_rows_csv),
        "review_summary_json": str(review_summary_json),
        "review_outcomes_sqlite": review_outcomes_sqlite,
        "safe_conclusion": payload["safe_conclusion"],
        "next_actions": [
            "Open the HTML dashboard in a browser.",
            "Set human review labels and notes.",
            "Export reviewed JSONL from the browser.",
            "Run corpus ingest-paper-link-reviews on the exported JSONL before calibrating ranking weights.",
        ],
        "provenance": [ProvenanceRecord(source_type="review_template", source_id=str(review_template), method="build_paper_experiment_review_dashboard").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_review_dashboard_markdown, json_filename="paper_experiment_review_dashboard_report.json", markdown_filename="paper_experiment_review_dashboard_report.md")
    report["artifacts"]["html"] = str(html_path)
    report["artifacts"]["review_rows_csv"] = str(review_rows_csv)
    report["artifacts"]["review_summary_json"] = str(review_summary_json)
    report["artifacts"]["review_outcomes_sqlite"] = review_outcomes_sqlite
    return report


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _paper_link_candidate_from_report(report: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for candidate in report.get("candidate_links", []) or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("experiment_id") or "") == experiment_id:
            return candidate
    return {}


def _paper_constraints_for_review(report: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = []
    for record in report.get("extracted_constraints", []) or []:
        if not isinstance(record, dict):
            continue
        constraints.append(
            {
                "instrument": record.get("instrument") or [],
                "detector_modes": record.get("detector_modes") or [],
                "measurement_kinds": record.get("measurement_kinds") or [],
                "sample_environment_terms": record.get("sample_environment_terms") or [],
                "formula_like_title_strings": record.get("formula_like_title_strings") or [],
                "chunk_type_counts": record.get("chunk_type_counts") or {},
                "evidence_count": len(record.get("constraint_evidence") or []),
                "warnings": record.get("warnings") or [],
            }
        )
    return constraints


def _review_candidate_metadata_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "candidate_source": candidate.get("candidate_source"),
        "mapping_method": candidate.get("mapping_method"),
        "experiment_date": candidate.get("experiment_date"),
        "experiment_year": candidate.get("experiment_year"),
        "data_years": candidate.get("data_years") or [],
        "proposal_ids": candidate.get("proposal_ids") or [],
        "scan_types": candidate.get("scan_types") or [],
        "scan_count": candidate.get("scan_count"),
        "datafile_count": candidate.get("datafile_count"),
        "missing_fields": candidate.get("missing_fields") or [],
    }


def _file_metadata_summary_for_review(hydrated_summary: Any) -> dict[str, Any]:
    if not isinstance(hydrated_summary, dict):
        return {}
    return {
        "parent_group_key": hydrated_summary.get("parent_group_key"),
        "attempted_file_count": hydrated_summary.get("attempted_file_count"),
        "missing_path_count": hydrated_summary.get("missing_path_count"),
        "planned_npts_values": hydrated_summary.get("planned_npts_values") or [],
        "actual_row_count_values": hydrated_summary.get("actual_row_count_values") or [],
        "line_scan_statuses": hydrated_summary.get("line_scan_statuses") or [],
        "column_count_estimate": hydrated_summary.get("column_count_estimate"),
        "column_names_sample": hydrated_summary.get("column_names_sample") or [],
        "clue_policy": hydrated_summary.get("clue_policy"),
        "npts_policy": hydrated_summary.get("npts_policy"),
    }


def _write_paper_experiment_review_packet(
    packet_dir: Path,
    *,
    row: dict[str, Any],
    candidate: dict[str, Any],
    source_payload: dict[str, Any],
    paper_constraints: list[dict[str, Any]],
    file_metadata_summary: dict[str, Any],
    max_scan_plots: int,
    paper_base_url: str,
    local_runs_root: Path | None = None,
    data_catalog: str | None = None,
    precomputed_scan_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a per-candidate evidence packet for human paper-link review."""

    review_id = f"{row.get('paper_id')}::rank-{int(row.get('candidate_rank') or 0)}::{row.get('candidate_experiment_id')}"
    packet_id = _safe_review_packet_id(review_id)
    packet_dir.mkdir(parents=True, exist_ok=True)
    html_path = packet_dir / f"{packet_id}.html"
    json_path = packet_dir / f"{packet_id}.json"
    datafile_paths = _candidate_datafile_paths_for_review(candidate, data_catalog=data_catalog)
    max_parse = 500
    if precomputed_scan_records is not None:
        all_scans = list(precomputed_scan_records)
        parse_warnings = [
            "scan plots loaded from review_packet_scan_records.jsonl sidecar; raw data files were not required",
        ]
    else:
        all_scans, parse_warnings = _parse_review_packet_scans(datafile_paths, max_scan_plots=max_parse)
    inline_limit = 30
    inline_scans = all_scans[:inline_limit]
    overflow_scans = all_scans[inline_limit:]
    scans_sidecar_url = ""
    if overflow_scans:
        sidecar_path = packet_dir / f"{packet_id}_scans.json"
        sidecar_path.write_text(json.dumps(all_scans, sort_keys=True), encoding="utf-8")
        scans_sidecar_url = _path_to_web_url(sidecar_path, local_runs_root) if local_runs_root else sidecar_path.name
    experiment_summary = _compute_experiment_summary(all_scans, output_dir=packet_dir.parent / "experiment_reviews", local_runs_root=local_runs_root)
    payload = {
        "packet_id": packet_id,
        "review_id": review_id,
        "paper_id": row.get("paper_id"),
        "paper_title": row.get("paper_title"),
        "paper_open_url": _paper_open_url(row.get("paper_id"), paper_base_url=paper_base_url),
        "candidate_experiment_id": row.get("candidate_experiment_id"),
        "candidate_rank": row.get("candidate_rank"),
        "probability_score": row.get("probability_score"),
        "paper_constraints": paper_constraints,
        "candidate_metadata": _review_candidate_metadata_summary(candidate),
        "score_breakdown": candidate.get("score_breakdown") or {},
        "evidence": candidate.get("evidence") or row.get("evidence_to_check") or [],
        "warnings": [*(candidate.get("warnings") or []), *(row.get("warnings") or [])],
        "file_metadata_summary": file_metadata_summary,
        "hydrated_clue_summary": candidate.get("hydrated_clue_summary") or ((candidate.get("metadata") or {}).get("hydrated_clue_summary") if isinstance(candidate.get("metadata"), dict) else {}) or {},
        "historic_clue_summary": candidate.get("historic_clue_summary") or ((candidate.get("metadata") or {}).get("historic_clue_summary") if isinstance(candidate.get("metadata"), dict) else {}) or {},
        "candidate_datafile_paths": datafile_paths,
        "parsed_scan_count": len(all_scans),
        "parse_warnings": parse_warnings,
        "scan_records": inline_scans,
        "scan_records_overflow_count": len(overflow_scans),
        "scans_sidecar_url": scans_sidecar_url,
        "experiment_summary": experiment_summary,
        "source_report_paper_id": source_payload.get("paper_id"),
        "safe_conclusion": "This packet is review evidence only. It does not promote a proposed paper-experiment link to an accepted claim.",
        "provenance": [
            ProvenanceRecord(source_type="paper_experiment_review_row", source_id=review_id, method="write_paper_experiment_review_packet").model_dump(mode="json")
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_paper_experiment_review_packet_html(payload), encoding="utf-8")
    return {
        "packet_id": packet_id,
        "html": str(html_path),
        "json": str(json_path),
        "html_url": _path_to_web_url(html_path, local_runs_root),
        "json_url": _path_to_web_url(json_path, local_runs_root),
        "parsed_scan_count": len(all_scans),
        "datafile_count": len(datafile_paths),
        "status": "ok" if all_scans else "metadata_only",
        "warnings": parse_warnings,
    }


def _load_review_scan_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    records: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        key = _review_scan_record_key(item.get("paper_id"), item.get("candidate_experiment_id"), item.get("candidate_rank"))
        scans = item.get("scan_records")
        if key and isinstance(scans, list):
            records[key] = [scan for scan in scans if isinstance(scan, dict)]
    return records


def _review_scan_record_key(paper_id: Any, candidate_experiment_id: Any, candidate_rank: Any) -> str:
    return f"{paper_id}::rank-{int(candidate_rank or 0)}::{candidate_experiment_id}"


def _candidate_datafile_paths_for_review(candidate: dict[str, Any], *, data_catalog: str | None = None) -> list[str]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    values: list[Any] = []
    for key in ("sample_datafile_paths", "file_paths", "paths"):
        raw = candidate.get(key) if candidate.get(key) is not None else metadata.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw:
            values.append(raw)
    for key in ("datafile_samples",):
        raw = candidate.get(key) if candidate.get(key) is not None else metadata.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("path"):
                    values.append(item["path"])
                elif isinstance(item, str):
                    values.append(item)
    hydrated = candidate.get("hydrated_clue_summary") or metadata.get("hydrated_clue_summary") or {}
    subgroup_prefixes: list[tuple[str, int]] = []
    if isinstance(hydrated, dict):
        for subgroup in hydrated.get("reader_subgroups") or []:
            if not isinstance(subgroup, dict):
                continue
            sample_files = subgroup.get("sample_files") or []
            for raw in sample_files:
                values.append(raw)
            file_count = int(subgroup.get("file_count") or 0)
            if file_count > len(sample_files) and len(sample_files) >= 2:
                import os
                prefix = os.path.commonprefix(sample_files).rstrip("_0123456789")
                if len(prefix) >= 4:
                    subgroup_prefixes.append((prefix, file_count))
    paths: list[str] = []
    seen_basenames: set[str] = set()
    seen_paths: set[str] = set()
    to_resolve: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        basename = Path(text).name if ("/" in text or "\\" in text) else text
        if basename in seen_basenames:
            continue
        seen_basenames.add(basename)
        if "/" in text or "\\" in text:
            local_path = Path(text)
            if local_path.exists():
                if text not in seen_paths:
                    seen_paths.add(text)
                    paths.append(text)
            else:
                to_resolve.append(basename)
        else:
            to_resolve.append(text)
    if to_resolve and data_catalog:
        for p in _resolve_bare_filenames_from_catalog(to_resolve, data_catalog):
            if p not in seen_paths:
                seen_paths.add(p)
                paths.append(p)
    elif to_resolve:
        paths.extend(to_resolve)
    if subgroup_prefixes and data_catalog:
        for prefix, expected_count in subgroup_prefixes:
            if len(paths) >= expected_count:
                continue
            for p in _resolve_prefix_from_catalog(prefix, data_catalog):
                if p not in seen_paths:
                    seen_paths.add(p)
                    paths.append(p)
    return paths


def _resolve_bare_filenames_from_catalog(filenames: list[str], data_catalog: str) -> list[str]:
    try:
        import sqlite3
        con = sqlite3.connect(data_catalog)
        resolved = []
        for fname in filenames:
            rows = con.execute("SELECT path FROM data_files WHERE path LIKE ?", (f"%{fname}",)).fetchall()
            if rows:
                resolved.append(rows[0][0])
        con.close()
        return resolved
    except Exception:
        return []


def _resolve_prefix_from_catalog(prefix: str, data_catalog: str) -> list[str]:
    try:
        import sqlite3
        con = sqlite3.connect(data_catalog)
        rows = con.execute("SELECT path FROM data_files WHERE path LIKE ? ORDER BY path", (f"%{prefix}%",)).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


_experiment_review_cache: dict[str, dict[str, Any]] = {}


def _compute_experiment_summary(scan_records: list[dict[str, Any]], *, output_dir: Path | None = None, local_runs_root: Path | None = None) -> dict[str, Any]:
    if not scan_records:
        return {"scan_count": 0, "scan_type_breakdown": {}, "temperature_coverage": {}, "order_parameter_candidates": [], "narrow_feature_scans": [], "experiment_review": {}}
    data_dirs = set()
    for sr in scan_records:
        p = sr.get("path") or ""
        if "/" in p:
            data_dirs.add(str(Path(p).parent))
    data_root = next(iter(data_dirs), None) if len(data_dirs) == 1 else (next(iter(data_dirs), None) if data_dirs else None)
    review_result = {}
    review_artifacts = {}
    if data_root:
        review_result, review_artifacts = _run_experiment_review_cached(data_root, output_dir=output_dir, local_runs_root=local_runs_root)
    type_counts: dict[str, int] = {}
    temps: list[float] = []
    narrow_scans = []
    for record in scan_records:
        kind = str(record.get("measurement_kind") or "unknown")
        type_counts[kind] = type_counts.get(kind, 0) + 1
        temp_range = record.get("temperature_range") or {}
        for key in ("min", "max"):
            val = temp_range.get(key)
            if val is not None:
                temps.append(float(val))
        if record.get("narrow_feature_flags"):
            narrow_scans.append({"scan_id": record.get("scan_id"), "flag_count": len(record["narrow_feature_flags"])})
    temp_coverage = {"min": min(temps), "max": max(temps), "width": max(temps) - min(temps)} if temps else {}
    return {
        "scan_count": len(scan_records),
        "scan_type_breakdown": type_counts,
        "temperature_coverage": temp_coverage,
        "order_parameter_candidates": review_result.get("order_parameter_candidates") or _run_order_parameter_triage_on_records(scan_records),
        "narrow_feature_scans": narrow_scans,
        "experiment_review": review_result,
        "experiment_review_artifacts": review_artifacts,
    }


def _run_experiment_review_cached(data_root: str, *, output_dir: Path | None = None, local_runs_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if data_root in _experiment_review_cache:
        cached = _experiment_review_cache[data_root]
        return cached, _experiment_review_artifact_urls(cached, local_runs_root)
    try:
        from tas_copilot_core.workflows.experiment_dashboard import experiment_review
    except Exception:
        return {}, {}
    review_output = Path(output_dir or Path.cwd() / "local_runs" / "experiment_reviews") / _safe_review_dir_name(data_root)
    review_output.mkdir(parents=True, exist_ok=True)
    try:
        result = experiment_review({"data_root": data_root, "output_dir": str(review_output), "max_files": 300, "skip_spurion": False})
    except Exception:
        result = {}
    summary = _experiment_review_summary(result, review_output=review_output)
    _experiment_review_cache[data_root] = summary
    return summary, _experiment_review_artifact_urls(result, local_runs_root)


def _safe_review_dir_name(data_root: str) -> str:
    parts = Path(data_root).parts
    relevant = [p for p in parts if p not in ("", "/", "home", "ylem", "local_runs", "bt7_data", "bt7", "data")]
    return "-".join(relevant[-3:]) or "review"


def _experiment_review_summary(result: dict[str, Any], *, review_output: Path | None = None) -> dict[str, Any]:
    if not result:
        return {}
    workflow_results = result.get("workflow_results") or {}
    spurion = workflow_results.get("spurion_survey") or {}
    order_param = workflow_results.get("order_parameter_triage") or {}
    overnight_series = _load_overnight_series_from_artifacts(review_output) if review_output else []
    return {
        "status": result.get("status"),
        "scan_count": result.get("scan_count"),
        "series_count": result.get("series_count"),
        "order_parameter_candidate_count": result.get("order_parameter_candidate_count"),
        "order_parameter_candidates": _extract_op_candidates(order_param),
        "spurion_candidate_count": result.get("spurion_candidate_count"),
        "spurion_candidates": _extract_spurion_candidates(spurion),
        "ranked_next_steps": result.get("ranked_next_steps") or [],
        "overnight_series": overnight_series,
    }


def _load_overnight_series_from_artifacts(review_output: Path) -> list[dict[str, Any]]:
    overnight_json = review_output / "overnight_summary" / "overnight_scan_summary.json"
    if not overnight_json.exists():
        return []
    try:
        data = json.loads(overnight_json.read_text(encoding="utf-8"))
        return _extract_overnight_series(data)
    except Exception:
        return []


def _extract_op_candidates(op_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = op_result.get("candidates") or []
    return [
        {k: c.get(k) for k in ("candidate_id", "classification", "score", "scan_count", "temperature_range", "score_reasons", "scan_variable")}
        for c in candidates[:10]
    ]


def _extract_spurion_candidates(spurion_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = spurion_result.get("candidates") or []
    return [
        {k: c.get(k) for k in ("scan_id", "candidate_type", "likelihood_score", "likelihood_label")}
        for c in candidates[:10]
    ]


def _extract_overnight_series(overnight_result: dict[str, Any]) -> list[dict[str, Any]]:
    series = overnight_result.get("series") or []
    return [
        {k: s.get(k) for k in ("series_id", "scan_count", "probable_report_type", "measurement_kinds", "scan_variables", "temperature_range", "energy_range")}
        for s in series[:20]
    ]


def _experiment_review_artifact_urls(result: dict[str, Any], local_runs_root: Path | None) -> dict[str, Any]:
    artifacts = result.get("generated_artifacts") or result.get("experiment_review_artifacts") or {}
    urls = {}
    for key, path_str in artifacts.items():
        if path_str and isinstance(path_str, str):
            p = Path(path_str)
            if p.exists():
                urls[key] = _path_to_web_url(p, local_runs_root)
    return urls


def _run_order_parameter_triage_on_records(scan_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from tas_copilot_core.workflows.experiment_dashboard import (
            DashboardScanRecord,
            _aggregate_single_point_records,
            _order_parameter_candidate_summary,
            _order_parameter_eligible_records,
            _order_parameter_groups,
        )
    except Exception:
        return []
    records = []
    for sr in scan_records:
        try:
            records.append(DashboardScanRecord(
                scan_id=str(sr.get("scan_id") or ""),
                path=str(sr.get("path") or ""),
                experiment_id=sr.get("experiment_id"),
                title=sr.get("title"),
                scan_type=sr.get("scan_type"),
                scan_variable=sr.get("scan_variable"),
                date=sr.get("date"),
                measurement_kind=sr.get("measurement_kind") or "unknown",
                detector_mode=sr.get("detector_mode"),
                row_count=sr.get("row_count") or 0,
                column_names=sr.get("column_names") or [],
                q_center=sr.get("q_center") or {},
                energy_range=sr.get("energy_range") or {},
                temperature_range=sr.get("temperature_range") or {},
                intensity_summary={},
                narrow_feature_flags=sr.get("narrow_feature_flags") or [],
                plot_data=sr.get("plot_data") or {},
                metadata=sr.get("metadata") or {},
            ))
        except Exception:
            continue
    synthetic = _aggregate_single_point_records(records)
    all_records = records + synthetic
    eligible = _order_parameter_eligible_records(all_records)
    if not eligible:
        return []
    regular = [r for r in eligible if not (r.metadata or {}).get("cross_file_series")]
    cross_file = [r for r in eligible if (r.metadata or {}).get("cross_file_series")]
    groups = _order_parameter_groups(regular, min_scans=3)
    candidates = [_order_parameter_candidate_summary(gid, items) for gid, items in groups.items()]
    from tas_copilot_core.workflows.experiment_dashboard import _cross_file_order_parameter_candidate
    for cf in cross_file:
        candidates.append(_cross_file_order_parameter_candidate(cf))
    candidates = [c for c in candidates if c.get("score", 0) > 0]
    candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    for c in candidates:
        c["per_scan_peak_summary"] = [
            {"temperature_mean": p.get("temperature_mean"), "integrated_signal": p.get("integrated_signal"), "peak_count": p.get("peak_count"), "scan_id": p.get("scan_id")}
            for p in (c.get("per_scan_peak_summary") or [])
        ]
        c.pop("provenance", None)
    return candidates[:10]


def _parse_review_packet_scans(paths: list[str], *, max_scan_plots: int) -> tuple[list[dict[str, Any]], list[str]]:
    parsed: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        from tas_copilot_core.workflows.experiment_dashboard import _parse_scan_file
    except Exception as exc:  # pragma: no cover - defensive optional workflow import.
        return [], [f"scan parser unavailable: {type(exc).__name__}: {exc}"]
    for raw in paths[: max(0, max_scan_plots)]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            warnings.append(f"not a local absolute path; metadata only: {raw}")
            continue
        if not path.exists():
            warnings.append(f"local data file missing; metadata only: {raw}")
            continue
        try:
            record = _parse_scan_file(path, max_plot_points=800).model_dump(mode="json")
        except Exception as exc:  # pragma: no cover - malformed private data should not break dashboard.
            warnings.append(f"could not parse {path.name}: {type(exc).__name__}: {exc}")
            continue
        parsed.append(
            {
                "scan_id": record.get("scan_id"),
                "path": record.get("path"),
                "experiment_id": record.get("experiment_id"),
                "title": record.get("title"),
                "date": record.get("date"),
                "scan_type": record.get("scan_type"),
                "scan_variable": record.get("scan_variable"),
                "measurement_kind": record.get("measurement_kind"),
                "detector_mode": record.get("detector_mode"),
                "row_count": record.get("row_count"),
                "column_names": record.get("column_names") or [],
                "q_center": record.get("q_center") or {},
                "energy_range": record.get("energy_range") or {},
                "temperature_range": record.get("temperature_range") or {},
                "metadata": record.get("metadata") or {},
                "plot_data": record.get("plot_data") or {},
                "narrow_feature_flags": record.get("narrow_feature_flags") or [],
            }
        )
    if len(paths) > max_scan_plots:
        warnings.append(f"packet plotted first {max_scan_plots} of {len(paths)} candidate files")
    return parsed, warnings


def _safe_review_packet_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-").lower()
    return (slug[:80] or "review-packet") + "-" + _hash(str(value))[:8]


def _paper_open_url(paper_id: Any, *, paper_base_url: str) -> str:
    if not paper_id:
        return ""
    return f"{paper_base_url}/{quote(str(paper_id), safe='')}"


def _path_to_web_url(path: Path, local_runs_root: Path | None) -> str:
    if local_runs_root:
        try:
            return "/local-runs/" + path.resolve().relative_to(local_runs_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_uri()


def ingest_paper_experiment_review_labels(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize exported human labels from the review dashboard."""

    labels_path = Path(arguments["labels"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not labels_path.exists():
        return _missing("paper-experiment-review-ingest", output_dir, ["labels"], f"Review labels file does not exist: {labels_path}")
    rows = _load_jsonl_records(labels_path)
    allowed = {"accepted", "rejected", "needs_more_evidence", "duplicate", "not_enough_information", "unreviewed"}
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        status = str(row.get("human_review_status") or "unreviewed")
        normalized = dict(row)
        normalized["human_review_status"] = status
        normalized["review_row_number"] = index
        if status not in allowed or not row.get("paper_id") or not row.get("candidate_experiment_id"):
            normalized["validation_error"] = "missing required IDs or invalid human_review_status"
            invalid_rows.append(normalized)
        elif status == "unreviewed":
            normalized["validation_warning"] = "unreviewed rows are preserved but do not count toward calibration labels"
            valid_rows.append(normalized)
        else:
            valid_rows.append(normalized)
    reviewed = [row for row in valid_rows if row.get("human_review_status") != "unreviewed"]
    accepted = [row for row in reviewed if row.get("human_review_status") == "accepted"]
    rejected = [row for row in reviewed if row.get("human_review_status") == "rejected"]
    min_each = int(arguments.get("min_each") or 10)
    labeled_path = output_dir / "paper_experiment_review_labels.validated.jsonl"
    _write_jsonl(labeled_path, valid_rows)
    invalid_path = output_dir / "paper_experiment_review_labels.invalid.jsonl"
    _write_jsonl(invalid_path, invalid_rows)
    review_outcomes_sqlite = _write_paper_review_outcome_store(
        output_dir / "review_outcomes.sqlite",
        valid_rows,
        source_path=labels_path,
        batch_method="paper_experiment_review_label_ingest_dual_write",
    )
    report = {
        "workflow_name": "paper-experiment-review-ingest",
        "status": "ok" if not invalid_rows else "partial",
        "labels": str(labels_path),
        "valid_row_count": len(valid_rows),
        "invalid_row_count": len(invalid_rows),
        "reviewed_label_count": len(reviewed),
        "status_counts": _count_values(valid_rows, "human_review_status"),
        "calibration_readiness": {
            "status": "ready_for_initial_calibration" if len(accepted) >= min_each and len(rejected) >= min_each else "needs_more_human_labels",
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "minimum_each_for_initial_calibration": min_each,
            "can_calibrate": len(accepted) >= min_each and len(rejected) >= min_each,
        },
        "generated_artifacts": {
            "validated_labels_jsonl": str(labeled_path),
            "invalid_labels_jsonl": str(invalid_path),
            "review_outcomes_sqlite": review_outcomes_sqlite,
        },
        "safe_conclusion": "Only exported human labels count as calibration input. Review-assist priorities and model scores are not labels.",
        "provenance": [ProvenanceRecord(source_type="review_labels", source_id=str(labels_path), method="ingest_paper_experiment_review_labels").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_review_label_ingest_markdown, json_filename="paper_experiment_review_label_ingest.json", markdown_filename="paper_experiment_review_label_ingest.md")
    return report


def build_bt7_hydrated_index(arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a local-only BT-7 hydrated metadata sidecar grouped by reader ExptID.

    The index is intentionally separate from the TAS SQLite catalog. It is derived
    from hydrated files through the registered BT-7 reader and remains proposed
    linkage evidence, not an accepted publication/data mapping.
    """

    data_catalog = Path(arguments["data_catalog"]).expanduser()
    output = Path(arguments["output"]).expanduser()
    output_dir = Path(arguments.get("output_dir") or output.parent / "bt7_hydrated_index_report").expanduser()
    max_files = arguments.get("max_files")
    max_files = int(max_files) if max_files not in (None, "", 0, "0") else None
    path_contains = str(arguments.get("path_contains") or "").strip() or None
    if not data_catalog.exists():
        return _missing("build-bt7-hydrated-index", output_dir, ["data_catalog"], f"Data catalog does not exist: {data_catalog}")

    rows = _bt7_datafile_rows(data_catalog, max_files=max_files, path_contains=path_contains)
    from tas_copilot_core.workflows.common import default_runner

    runner = default_runner()
    groups: dict[str, dict[str, Any]] = {}
    attempted = 0
    read_files = 0
    missing_files = 0
    read_errors: list[str] = []
    tool_status_counts: dict[str, int] = {}
    for row in rows:
        path = Path(str(row.get("path") or "")).expanduser()
        attempted += 1
        if not path.exists() or not path.is_file():
            missing_files += 1
            continue
        result = runner.run(
            ToolCall(
                tool_namespace="tas.data_reader",
                tool_name="read_bt7_file",
                arguments={"path": str(path)},
                caller_workflow="build-bt7-hydrated-index",
            )
        )
        status = str(getattr(result.status, "value", result.status))
        tool_status_counts[status] = tool_status_counts.get(status, 0) + 1
        if status != "success" or not isinstance(result.output, dict):
            read_errors.extend(f"{path.name}: {error.code}: {error.message}" for error in result.errors[:2])
            continue
        read_files += 1
        output_payload = result.output
        metadata = output_payload.get("metadata") if isinstance(output_payload.get("metadata"), dict) else {}
        _add_hydrated_index_file(groups, row, path, metadata, output_payload)

    records = []
    for group in groups.values():
        records.append(_hydrated_index_group_to_candidate(group, data_catalog=data_catalog, candidate_source="bt7_hydrated_index"))
        source_units = group.get("source_units") or {}
        for source_unit in source_units.values():
            records.append(_hydrated_index_group_to_candidate(source_unit, data_catalog=data_catalog, candidate_source="bt7_hydrated_source_unit"))
    records.sort(key=lambda item: (-int(item.get("datafile_count") or 0), str(item.get("title") or ""), str(item.get("experiment_id") or "")))
    _write_jsonl(output, records)
    report = {
        "workflow_name": "build-bt7-hydrated-index",
        "status": "ok",
        "data_catalog": str(data_catalog),
        "output": str(output),
        "path_contains": path_contains,
        "attempted_files": attempted,
        "read_files": read_files,
        "missing_files": missing_files,
        "group_count": len(groups),
        "source_unit_count": sum(len(group.get("source_units") or {}) for group in groups.values()),
        "candidate_count": len(records),
        "tool_status_counts": _top_counts(tool_status_counts, 8),
        "read_errors": read_errors[:20],
        "index_policy": "local hydrated BT-7 reader-derived sidecar; proposed linkage evidence only; does not mutate TAS data catalog",
        "provenance": [ProvenanceRecord(source_type="data_catalog", source_id=str(data_catalog), method="build_bt7_hydrated_index").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_bt7_hydrated_index_markdown, json_filename="bt7_hydrated_index_report.json", markdown_filename="bt7_hydrated_index_report.md")
    return report


def _score_data_catalog_experiment_links(
    data_catalog: str | Path,
    *,
    paper_id: str,
    paper: dict[str, Any] | None,
    constraints: list[dict[str, Any]],
    lag_prior: PublicationDataLagPrior,
    top_k: int,
    max_candidates: int,
    hydrate_candidate_clues: bool = False,
    hydrate_candidate_limit: int = 25,
    max_hydrated_files_per_candidate: int = 12,
    hydrated_index: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    db_path = Path(data_catalog).expanduser()
    status: dict[str, Any] = {"provided": True, "path": str(db_path), "status": "missing"}
    if not db_path.exists():
        status["reason"] = "data catalog path does not exist"
        return [], status
    try:
        candidates = _data_catalog_experiment_candidates(db_path, max_candidates=max_candidates)
    except sqlite3.Error as exc:
        status.update({"status": "error", "reason": str(exc)})
        return [], status
    if not candidates:
        status.update({"status": "empty", "candidate_count": 0})
        return [], status
    index_status = _load_hydrated_index_candidates(hydrated_index)
    if index_status["candidates"]:
        candidates.extend(index_status["candidates"])
    hydration_summary: dict[str, Any] = {"enabled": hydrate_candidate_clues, "attempted_candidates": 0, "read_files": 0}
    if hydrate_candidate_clues:
        from tas_copilot_core.workflows.common import default_runner

        runner = default_runner()
        preliminary = [_score_experiment_candidate(candidate, paper_id=paper_id, paper=paper, constraints=constraints, lag_prior=lag_prior, db_path=db_path) for candidate in candidates]
        hydrate_ids = {
            str(item.get("experiment_id"))
            for item in sorted(preliminary, key=lambda row: row["probability_score"], reverse=True)[: max(0, hydrate_candidate_limit)]
        }
        for candidate in candidates:
            if str(candidate.get("experiment_id")) in hydrate_ids and not candidate.get("hydrated_clue_summary"):
                summary = _hydrated_header_clue_summary(candidate, runner=runner, max_files=max_hydrated_files_per_candidate)
                candidate["hydrated_clue_summary"] = summary
                hydration_summary["attempted_candidates"] += 1
                hydration_summary["read_files"] += int(summary.get("read_file_count") or 0)
    scored = [_score_experiment_candidate(candidate, paper_id=paper_id, paper=paper, constraints=constraints, lag_prior=lag_prior, db_path=db_path) for candidate in candidates]
    ranked = sorted(scored, key=lambda item: item["probability_score"], reverse=True)
    _persist_candidate_links(db_path, [item for item in ranked[:top_k] if item.get("candidate_source") == "data_catalog_experiment"])
    status.update(
        {
            "status": "scored",
            "candidate_count": len(candidates),
            "sqlite_candidate_count": len(candidates) - len(index_status["candidates"]),
            "returned_top_k": min(top_k, len(ranked)),
            "returned_candidate_count": len(ranked),
            "scoring_policy": "deterministic probabilistic ranking; candidate links remain proposed and require human review",
            "hydrated_clue_enrichment": hydration_summary,
            "hydrated_index": {key: value for key, value in index_status.items() if key != "candidates"},
        }
    )
    return ranked, status


def _data_catalog_experiment_candidates(db_path: Path, *, max_candidates: int) -> list[dict[str, Any]]:
    query = """
        SELECT e.experiment_id,
               e.instrument,
               e.title,
               e.start_date,
               e.end_date,
               e.payload_json,
               (SELECT COUNT(*) FROM scans s WHERE s.experiment_id = e.experiment_id) AS scan_count,
               (SELECT COUNT(*) FROM data_files d WHERE d.experiment_id = e.experiment_id) AS datafile_count,
               (SELECT GROUP_CONCAT(DISTINCT scan_type) FROM scans s2 WHERE s2.experiment_id = e.experiment_id AND s2.scan_type IS NOT NULL) AS scan_types
        FROM experiments e
        ORDER BY e.experiment_id
        LIMIT ?
    """
    candidates = []
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (int(max_candidates),)).fetchall()
        for row in rows:
            experiment_id = str(row["experiment_id"])
            all_path_rows = connection.execute(
                """
                SELECT path
                FROM data_files
                WHERE experiment_id = ?
                ORDER BY datafile_id
                """,
                (experiment_id,),
            ).fetchall()
            all_paths = [str(item["path"] or "") for item in all_path_rows]
            samples = connection.execute(
                """
                SELECT path, instrument, native_format, native_scan_id, payload_json
                FROM data_files
                WHERE experiment_id = ?
                ORDER BY
                    CASE
                        WHEN lower(path) LIKE '%.bt7' THEN 0
                        WHEN lower(path) LIKE '%.dat' THEN 1
                        WHEN lower(path) LIKE '%.ng5' THEN 2
                        ELSE 3
                    END,
                    CASE
                        WHEN lower(path) LIKE '%/config/%' THEN 1
                        WHEN lower(path) LIKE '%console_history%' THEN 1
                        ELSE 0
                    END,
                    datafile_id
                LIMIT 12
                """,
                (experiment_id,),
            ).fetchall()
            scan_samples = connection.execute(
                """
                SELECT scan_id, scan_type, payload_json
                FROM scans
                WHERE experiment_id = ?
                ORDER BY scan_id
                LIMIT 12
                """,
                (experiment_id,),
            ).fetchall()
            datafile_samples = [dict(item) for item in samples]
            path_clues = _path_clue_summary(all_paths)
            candidates.append(
                {
                    "experiment_id": experiment_id,
                    "instrument": row["instrument"],
                    "title": row["title"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "experiment_payload": _safe_json(row["payload_json"]),
                    "scan_count": int(row["scan_count"] or 0),
                    "datafile_count": int(row["datafile_count"] or 0),
                    "scan_types": [item for item in str(row["scan_types"] or "").split(",") if item],
                    "datafile_samples": datafile_samples,
                    "scan_samples": [dict(item) for item in scan_samples],
                    "data_years": path_clues["data_years"],
                    "proposal_ids": path_clues["proposal_ids"],
                    "path_clues": path_clues,
                    "candidate_source": "data_catalog_experiment",
                    "persist_candidate_link": True,
                }
            )
    return candidates


def _bt7_datafile_rows(db_path: Path, *, max_files: int | None = None, path_contains: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT datafile_id, path, scan_id, experiment_id, instrument, native_format, native_scan_id, payload_json
        FROM data_files
        WHERE lower(path) LIKE '%.bt7'
    """
    parameters: list[Any] = []
    if path_contains:
        query += " AND path LIKE ?"
        parameters.append(f"%{path_contains}%")
    query += " ORDER BY path"
    if max_files is not None:
        query += " LIMIT ?"
        parameters.append(int(max_files))
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, tuple(parameters)).fetchall()]


def _add_hydrated_index_file(groups: dict[str, dict[str, Any]], row: dict[str, Any], path: Path, metadata: dict[str, Any], output: dict[str, Any]) -> None:
    expt_id = str(metadata.get("expt_id") or "").strip()
    expt_name = str(metadata.get("expt_name") or "").strip()
    if expt_id or expt_name:
        key = f"reader-expt|{expt_id or 'unknown'}|{expt_name or 'unknown'}"
    else:
        key = f"reader-unknown|{path.parent}"
    group = groups.setdefault(key, _new_hydrated_index_group(key, native_experiment_id=expt_id or None, title=expt_name or None))
    _record_hydrated_index_file(group, row, path, metadata, output)
    source_unit_key = _hydrated_source_unit_key(key, path, metadata)
    source_unit = group.setdefault("source_units", {}).setdefault(
        source_unit_key,
        _new_hydrated_index_group(
            source_unit_key,
            native_experiment_id=expt_id or None,
            title=_hydrated_source_unit_title(expt_name, path, metadata),
            parent_group_key=key,
        ),
    )
    _record_hydrated_index_file(source_unit, row, path, metadata, output)


def _new_hydrated_index_group(key: str, *, native_experiment_id: str | None, title: str | None, parent_group_key: str | None = None) -> dict[str, Any]:
    return {
        "group_key": key,
        "native_experiment_id": native_experiment_id,
        "title": title,
        "parent_group_key": parent_group_key,
        "file_count": 0,
        "datafile_ids": [],
        "scan_ids": [],
        "source_experiment_ids": [],
        "file_paths": [],
        "sample_files": [],
        "dates": [],
        "reader_fields": set(),
        "top_values": {field: {} for field in _BT7_READER_CLUE_FIELDS},
        "scan_descr_values": {field: {} for field in _SCAN_DESCR_CLUE_FIELDS},
        "column_names": {},
        "row_counts": {},
        "planned_npts": {},
        "psd_map_statuses": {},
        "line_scan_statuses": {},
        "subgroups": {},
        "source_units": {},
    }


def _record_hydrated_index_file(group: dict[str, Any], row: dict[str, Any], path: Path, metadata: dict[str, Any], output: dict[str, Any]) -> None:
    group["file_count"] += 1
    group["file_paths"].append(str(path))
    for source_key, target_key in [("datafile_id", "datafile_ids"), ("scan_id", "scan_ids"), ("experiment_id", "source_experiment_ids")]:
        value = row.get(source_key)
        if value and value not in group[target_key]:
            group[target_key].append(str(value))
    if len(group["sample_files"]) < 12:
        group["sample_files"].append(
            {
                "path": str(path),
                "instrument": "BT-7",
                "native_format": "ice",
                "native_scan_id": metadata.get("filename") or path.stem,
                "payload_json": json.dumps({"metadata": {"source_adapter": "bt7_hydrated_index", "reader_metadata": metadata}}, sort_keys=True),
            }
        )
    date = _reader_date(metadata)
    if date:
        group["dates"].append(date)
    for key in _BT7_READER_CLUE_FIELDS:
        value = metadata.get(key)
        if not _empty_reader_value(value):
            group["reader_fields"].add(key)
            bucket = group["top_values"][key]
            text = _short_json_value(value)
            bucket[text] = bucket.get(text, 0) + 1
    descr = _parse_scan_descr(str(metadata.get("scan_description") or ""))
    for key, value in descr.items():
        if key in group["scan_descr_values"] and value:
            bucket = group["scan_descr_values"][key]
            bucket[value] = bucket.get(value, 0) + 1
    for column in output.get("columns") or []:
        column_key = str(column).lower()
        group["column_names"][column_key] = group["column_names"].get(column_key, 0) + 1
    row_count = output.get("row_count")
    if row_count is not None:
        group["row_counts"][str(row_count)] = group["row_counts"].get(str(row_count), 0) + 1
    npts = metadata.get("npoints")
    if npts not in (None, ""):
        group["planned_npts"][str(npts)] = group["planned_npts"].get(str(npts), 0) + 1
    psd_map = output.get("psd_map") if isinstance(output.get("psd_map"), dict) else None
    psd_status = str(psd_map.get("status") if psd_map else "absent")
    group["psd_map_statuses"][psd_status] = group["psd_map_statuses"].get(psd_status, 0) + 1
    line_scan = output.get("line_scan") if isinstance(output.get("line_scan"), dict) else None
    line_status = str(line_scan.get("status") if line_scan else "absent")
    group["line_scan_statuses"][line_status] = group["line_scan_statuses"].get(line_status, 0) + 1
    _add_hydrated_subgroup(group["subgroups"], path, metadata, output)


def _hydrated_source_unit_key(parent_key: str, path: Path, metadata: dict[str, Any]) -> str:
    prefix = _filename_family_prefix(metadata.get("filename") or path.stem)
    scan_family = _bt7_scan_family(metadata, path)
    detector = _normalized_detector_mode(metadata.get("analyzer_detector_mode"))
    date_bucket = _reader_month_bucket(metadata) or "unknown_date"
    return "|".join(["reader-source-unit", parent_key, f"prefix={prefix}", f"family={scan_family}", f"detector={detector}", f"date={date_bucket}"])


def _hydrated_source_unit_title(expt_name: str, path: Path, metadata: dict[str, Any]) -> str:
    prefix = _filename_family_prefix(metadata.get("filename") or path.stem)
    scan_family = _bt7_scan_family(metadata, path)
    detector = _normalized_detector_mode(metadata.get("analyzer_detector_mode"))
    date_bucket = _reader_month_bucket(metadata) or "unknown date"
    base = expt_name or "unknown experiment"
    return f"{base} / {prefix} / {scan_family} / {detector} / {date_bucket}"


def _source_unit_metadata_from_key(key: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for part in key.split("|"):
        if "=" in part:
            field, value = part.split("=", 1)
            metadata[field] = value
    return metadata


def _filename_family_prefix(value: Any) -> str:
    stem = Path(str(value or "unknown")).stem.lower()
    stem = re.sub(r"\d+$", "", stem)
    stem = re.sub(r"(?:warm|cool)?oc$", "", stem)
    stem = stem.strip("_-.")
    if not stem:
        return "unknown_prefix"
    return re.sub(r"[^a-z0-9]+", "_", stem)[:80]


def _normalized_detector_mode(value: Any) -> str:
    text = str(value or "unknown").lower()
    if "psddiff" in text:
        return "psddiff"
    if "psdflat" in text:
        return "psdflat"
    if "singdet" in text:
        return "single_detector"
    if "diffdet" in text:
        return "diffraction_detector"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown_detector"


def _reader_month_bucket(metadata: dict[str, Any]) -> str | None:
    date = _reader_date(metadata)
    if not date:
        return None
    return date[:7]


def _bt7_scan_family(metadata: dict[str, Any], path: Path) -> str:
    detector = _normalized_detector_mode(metadata.get("analyzer_detector_mode"))
    scan_type = str(metadata.get("scan_type") or "").lower()
    scan_variable = str(metadata.get("scan_variable") or "").lower()
    scan_description = str(metadata.get("scan_description") or "").lower()
    filename = path.stem.lower()
    e_value = _float_or_none(metadata.get("e_center"))
    text = " ".join([scan_type, scan_variable, scan_description, filename])
    if "find peak" in scan_type or "alignment" in text or "fpx" in filename:
        return "alignment_or_find_peak"
    if "psd" in detector:
        if "env" in scan_type or scan_variable == "temp" or "tdep" in filename:
            return "single_file_psd_temperature_map"
        if "angle" in scan_type or scan_variable == "a4" or "correlations" in filename:
            return "multi_file_psd_angle_temperature_series"
        return "psd_scan"
    if e_value is not None and abs(e_value) > 1e-5:
        return "single_detector_inelastic_line_scan"
    if re.search(r"(?<!\\d)(?:3|6|9)mev", filename):
        return "single_detector_inelastic_line_scan"
    if "elastic" in text or (e_value is not None and abs(e_value) <= 1e-5):
        return "single_detector_elastic_line_scan"
    if "vector" in scan_type:
        return "vector_scan"
    if "angle" in scan_type or scan_variable == "a4":
        return "angle_scan"
    if "env" in scan_type or scan_variable == "temp":
        return "temperature_scan"
    return "unknown_or_sparse"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hydrated_index_group_to_candidate(group: dict[str, Any], *, data_catalog: Path, candidate_source: str) -> dict[str, Any]:
    dates = sorted(set(group.get("dates") or []))
    path_clues = _path_clue_summary(group.get("file_paths") or [])
    title = group.get("title") or Path(str((group.get("file_paths") or ["unknown"])[0])).parent.name
    candidate_id = f"{candidate_source.replace('_', '-')}-{_hash(group['group_key'])}"
    top_values = {field: _top_counts(values, 8) for field, values in group["top_values"].items() if values}
    scan_descr_values = {field: _top_counts(values, 8) for field, values in group["scan_descr_values"].items() if values}
    hydrated_summary = {
        "status": "ok",
        "attempted_file_count": int(group["file_count"]),
        "read_file_count": int(group["file_count"]),
        "missing_path_count": 0,
        "tool_status_counts": [{"value": "success", "count": int(group["file_count"])}],
        "read_errors": [],
        "reader_fields_seen": sorted(group["reader_fields"]),
        "top_values": top_values,
        "scan_descr_values": scan_descr_values,
        "column_names_sample": sorted(group["column_names"])[:80],
        "column_count_estimate": len(group["column_names"]),
        "actual_row_count_values": _top_counts(group["row_counts"], 8),
        "planned_npts_values": _top_counts(group["planned_npts"], 8),
        "npts_policy": "ScanDescr/Npoints are planned point counts; actual data coverage must use row_count from parsed data columns.",
        "psd_map_statuses": _top_counts(group["psd_map_statuses"], 8),
        "line_scan_statuses": _top_counts(group["line_scan_statuses"], 8),
        "reader_subgroups": _reader_subgroup_payloads(group["subgroups"]),
        "source_unit_count": len(group.get("source_units") or {}),
        "source_tool": "tas.data_reader.read_bt7_file",
        "clue_policy": "hydrated_bt7_index_reader_clues_only; untrusted weak linkage evidence requiring human review",
    }
    if group.get("parent_group_key"):
        hydrated_summary["parent_group_key"] = group.get("parent_group_key")
        hydrated_summary["source_unit"] = _source_unit_metadata_from_key(group["group_key"])
    scan_types = [item["value"] for item in _top_counts(group["top_values"].get("scan_type", {}), 12)]
    return {
        "experiment_id": candidate_id,
        "instrument": "BT-7",
        "title": title,
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "experiment_payload": {
            "metadata": {
                "candidate_source": candidate_source,
                "native_experiment_id": group.get("native_experiment_id"),
                "parent_group_key": group.get("parent_group_key"),
                "source_data_catalog": str(data_catalog),
                "source_experiment_ids": group.get("source_experiment_ids", [])[:20],
            }
        },
        "scan_count": len(group.get("scan_ids") or []),
        "datafile_count": int(group["file_count"]),
        "scan_types": scan_types,
        "datafile_samples": group.get("sample_files", []),
        "scan_samples": [],
        "data_years": path_clues["data_years"],
        "proposal_ids": path_clues["proposal_ids"],
        "path_clues": path_clues,
        "hydrated_clue_summary": hydrated_summary,
        "candidate_source": candidate_source,
        "persist_candidate_link": False,
    }


def _apply_paper_link_review_calibration(
    links: list[dict[str, Any]],
    *,
    paper_id: str,
    review_label_store: str | Path | None,
    label_priors: dict[str, Any],
) -> list[dict[str, Any]]:
    exact_labels = _paper_link_exact_review_labels(review_label_store, paper_id=paper_id)
    calibrated: list[dict[str, Any]] = []
    for link in links:
        item = dict(link)
        base_score = _paper_link_base_score(item)
        candidate_id = _paper_link_candidate_id(item)
        exact_label = exact_labels.get(candidate_id)
        item["uncalibrated_probability_score"] = base_score
        if exact_label is not None:
            _apply_exact_paper_link_label(item, exact_label, base_score=base_score)
        else:
            channels = _paper_link_calibration_channels(item)
            label_score = label_calibrated_score(
                base_score,
                evidence_channels=channels,
                priors=label_priors,
                require_behavior_ready=True,
            )
            item["review_label_status"] = "unreviewed"
            item["reviewed_label_verdict"] = None
            item["label_calibration_channels"] = channels
            item["label_calibrated_score"] = float(label_score.get("applied_score", base_score))
            item["ranking_score"] = item["label_calibrated_score"]
            item["label_calibration_status"] = label_score.get("calibration_status")
            item["label_calibration_applied"] = bool(label_score.get("applied"))
            item["label_calibration_applied_channels"] = label_score.get("applied_channels") or []
        calibrated.append(item)
    return calibrated


def _paper_link_exact_review_labels(review_label_store: str | Path | None, *, paper_id: str) -> dict[str, ReviewLabel]:
    if not review_label_store:
        return {}
    path = Path(review_label_store).expanduser()
    if not path.exists():
        return {}
    labels = [
        label
        for label in ReviewOutcomeSQLiteStore(path).list_labels(
            object_type="paper_experiment_link",
            object_id=paper_id,
            source_workflow="link-paper-to-experiments",
        )
        if label.verdict != "unreviewed" and label.object_secondary_id
    ]
    labels.sort(key=_review_label_recency_key)
    return {str(label.object_secondary_id): label for label in labels}


def _review_label_recency_key(label: ReviewLabel) -> tuple[str, str]:
    reviewed_at = label.reviewed_at.isoformat() if label.reviewed_at else ""
    return reviewed_at, label.label_id


def _apply_exact_paper_link_label(item: dict[str, Any], label: ReviewLabel, *, base_score: float) -> None:
    verdict = str(label.verdict)
    item["review_label_status"] = "reviewed"
    item["reviewed_label_verdict"] = verdict
    item["reviewed_label_claim_state"] = label.claim_state
    item["reviewed_label_id"] = label.label_id
    item["reviewed_by"] = label.reviewer_id
    item["reviewed_at"] = label.reviewed_at.isoformat() if label.reviewed_at else None
    item["label_calibration_channels"] = list(label.evidence_channels_relied_on or [])
    if verdict == "positive":
        item["label_calibrated_score"] = 1.0
        item["ranking_score"] = 1.0
        item["status"] = "accepted"
        item["confidence_label"] = "reviewed_positive"
        item["label_calibration_status"] = "exact_reviewed_positive"
    elif verdict == "negative":
        item["label_calibrated_score"] = 0.0
        item["ranking_score"] = 0.0
        item["status"] = "rejected"
        item["confidence_label"] = "reviewed_negative"
        item["label_calibration_status"] = "exact_reviewed_negative"
    else:
        item["label_calibrated_score"] = base_score
        item["ranking_score"] = base_score
        item["status"] = "candidate"
        item["confidence_label"] = "reviewed_inconclusive"
        item["label_calibration_status"] = "exact_reviewed_inconclusive"
    item["label_calibration_applied"] = verdict in {"positive", "negative"}
    item["label_calibration_applied_channels"] = []


def _paper_link_calibration_channels(link: dict[str, Any]) -> list[str]:
    metadata = link.get("metadata") if isinstance(link.get("metadata"), dict) else {}
    if link.get("hydrated_clue_summary") or metadata.get("hydrated_clue_summary"):
        return ["evidence_channel_hydrated_metadata"]
    return ["evidence_channel_paper_text"]


def _paper_link_base_score(link: dict[str, Any]) -> float:
    return float(link.get("probability_score", link.get("confidence", 0.0)) or 0.0)


def _paper_link_candidate_id(link: dict[str, Any]) -> str:
    return str(link.get("experiment_id") or link.get("candidate_experiment_id") or link.get("source_object_id") or "")


def _paper_link_rank_key(link: dict[str, Any]) -> tuple[int, float, float]:
    verdict = link.get("reviewed_label_verdict")
    if verdict == "positive":
        priority = 3
    elif verdict == "negative":
        priority = 0
    elif verdict == "inconclusive":
        priority = 1
    else:
        priority = 2
    return (
        priority,
        float(link.get("ranking_score", link.get("label_calibrated_score", link.get("probability_score", 0.0))) or 0.0),
        _paper_link_base_score(link),
    )


def _load_hydrated_index_candidates(hydrated_index: str | Path | None) -> dict[str, Any]:
    status: dict[str, Any] = {"provided": bool(hydrated_index), "path": str(hydrated_index) if hydrated_index else None, "status": "not_provided", "candidate_count": 0, "candidates": []}
    if not hydrated_index:
        return status
    path = Path(hydrated_index).expanduser()
    status["path"] = str(path)
    if not path.exists():
        status.update({"status": "missing", "reason": "hydrated index path does not exist"})
        return status
    try:
        rows = _load_jsonl_dicts(path)
    except Exception as exc:
        status.update({"status": "error", "reason": str(exc)})
        return status
    candidates = [row for row in rows if row.get("candidate_source") in {"bt7_hydrated_index", "bt7_hydrated_source_unit"}]
    status.update({"status": "loaded", "candidate_count": len(candidates), "candidates": candidates})
    return status


def _reader_date(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("date_str") or metadata.get("datetime") or metadata.get("epoch")
    if raw is None:
        return None
    text = str(raw)
    match = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", text)
    if match:
        return f"{int(match.group(3)):04d}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
    return None


def _score_experiment_candidate(
    candidate: dict[str, Any],
    *,
    paper_id: str,
    paper: dict[str, Any] | None,
    constraints: list[dict[str, Any]],
    lag_prior: PublicationDataLagPrior,
    db_path: Path,
) -> dict[str, Any]:
    haystack = _candidate_haystack(candidate)
    constraint = _merged_constraints(constraints)
    evidence: list[str] = []
    missing: list[str] = []
    warnings = [
        "Candidate link is proposed. Do not treat it as a publication-to-data attribution without human review or stronger provenance.",
        "Publication/data lag contributes only weak ranking evidence.",
    ]
    archive_penalty = _mixed_archive_penalty(candidate)
    if archive_penalty:
        warnings.append("Large or mixed directory group detected; broad archive-like groups are penalized until enriched into smaller experiment/proposal groups.")
    material_title_score = _material_or_title_score(paper, constraint, haystack, evidence)
    components = {
        "instrument_score": _term_match_score(constraint["instrument"], haystack, evidence, "instrument"),
        "detector_mode_score": _term_match_score(constraint["detector_modes"], haystack, evidence, "detector mode"),
        "measurement_kind_score": _term_match_score(constraint["measurement_kinds"], haystack, evidence, "measurement kind"),
        "material_title_score": material_title_score,
        "hydrated_header_score": _hydrated_header_score(candidate, paper, constraint, evidence),
        "sample_environment_score": _term_match_score(constraint["sample_environment_terms"], haystack, evidence, "sample environment"),
        "date_lag_score": _mapping_date_score(candidate, lag_prior),
        "catalog_coverage_score": min(1.0, 0.15 + 0.02 * int(candidate.get("scan_count") or 0) + 0.01 * int(candidate.get("datafile_count") or 0)),
        "group_specificity_score": _group_specificity_score(candidate),
        "mixed_archive_penalty": archive_penalty,
        "hydrated_subgroup_mixed_penalty": _hydrated_subgroup_mixed_penalty(candidate),
        "material_mismatch_penalty": _material_mismatch_penalty(constraint, material_title_score),
        "source_unit_mismatch_penalty": _source_unit_mismatch_penalty(candidate, constraint),
        "hydrated_parent_group_penalty": _hydrated_parent_group_penalty(candidate),
    }
    if components["date_lag_score"] is None:
        components["date_lag_score"] = 0.0
        missing.append("experiment_date_for_publication_lag_prior")
    for key, values in [
        ("instrument", constraint["instrument"]),
        ("detector_modes", constraint["detector_modes"]),
        ("measurement_kinds", constraint["measurement_kinds"]),
        ("formula_like_title_strings", constraint["formula_like_title_strings"]),
    ]:
        if not values:
            missing.append(f"paper_{key}")
    if not evidence:
        evidence.append("Catalog record is available but no strong paper/catalog term overlap was found.")
    weighted = (
        0.24 * components["instrument_score"]
        + 0.12 * components["detector_mode_score"]
        + 0.14 * components["measurement_kind_score"]
        + 0.20 * components["material_title_score"]
        + 0.06 * components["hydrated_header_score"]
        + 0.06 * components["sample_environment_score"]
        + 0.12 * components["date_lag_score"]
        + 0.04 * components["catalog_coverage_score"]
        + 0.04 * components["group_specificity_score"]
        - components["mixed_archive_penalty"]
        - components["hydrated_subgroup_mixed_penalty"]
        - components["material_mismatch_penalty"]
        - components["source_unit_mismatch_penalty"]
        - components["hydrated_parent_group_penalty"]
    )
    probability_score = round(max(0.03, min(0.95, weighted)), 3)
    confidence_label = "medium" if probability_score >= 0.55 else "low"
    if probability_score >= 0.78 and len(evidence) >= 3:
        confidence_label = "high"
    elif probability_score < 0.18:
        confidence_label = "insufficient"
    experiment_year = _mapping_year(candidate)
    link = PaperExperimentLink(
        paper_id=paper_id,
        experiment_id=str(candidate["experiment_id"]),
        confidence=probability_score,
        evidence=evidence[:12],
        mapping_method="probabilistic_catalog_constraint_linkage",
        experiment_year=experiment_year,
        experiment_date=candidate.get("start_date") or candidate.get("end_date"),
        metadata={
            "candidate_type": "experiment",
            "candidate_source": candidate.get("candidate_source", "data_catalog_experiment"),
            "confidence_label": confidence_label,
            "score_breakdown": components,
            "historic_clue_summary": candidate.get("path_clues", {}),
            "hydrated_clue_summary": candidate.get("hydrated_clue_summary", {}),
            "scan_count": candidate.get("scan_count", 0),
            "datafile_count": candidate.get("datafile_count", 0),
            "scan_types": candidate.get("scan_types", []),
            "data_years": candidate.get("data_years", []),
            "proposal_ids": candidate.get("proposal_ids", []),
            "sample_datafile_paths": [item.get("path") for item in candidate.get("datafile_samples", [])[:5] if item.get("path")],
            "missing_fields": missing,
            "warnings": warnings,
            "source_data_catalog": str(db_path),
            "human_review_required": True,
            "accepted_source_claim": False,
        },
        notes="Proposed probabilistic paper-to-experiment candidate. Not accepted without human review or stronger provenance.",
        provenance=[ProvenanceRecord(source_type="data_catalog", source_id=str(db_path), method="probabilistic_catalog_constraint_linkage")],
        status="candidate",
    )
    payload = link.model_dump(mode="json")
    payload["probability_score"] = probability_score
    payload["candidate_source"] = candidate.get("candidate_source", "data_catalog_experiment")
    payload["confidence_label"] = confidence_label
    payload["score_breakdown"] = components
    payload["missing_fields"] = missing
    payload["warnings"] = warnings
    payload["scan_count"] = candidate.get("scan_count", 0)
    payload["datafile_count"] = candidate.get("datafile_count", 0)
    payload["scan_types"] = candidate.get("scan_types", [])
    payload["data_years"] = candidate.get("data_years", [])
    payload["proposal_ids"] = candidate.get("proposal_ids", [])
    payload["historic_clue_summary"] = candidate.get("path_clues", {})
    payload["hydrated_clue_summary"] = candidate.get("hydrated_clue_summary", {})
    return payload


def _persist_candidate_links(db_path: Path, links: list[dict[str, Any]]) -> None:
    records = []
    for link in links:
        try:
            records.append(
                PaperExperimentLink(
                    paper_id=link["paper_id"],
                    experiment_id=link["experiment_id"],
                    confidence=float(link.get("confidence", link.get("probability_score", 0.0))),
                    evidence=list(link.get("evidence", [])),
                    mapping_method=str(link.get("mapping_method") or "probabilistic_catalog_constraint_linkage"),
                    experiment_year=link.get("experiment_year"),
                    experiment_date=link.get("experiment_date"),
                    metadata=dict(link.get("metadata") or {}),
                    notes=link.get("notes"),
                    provenance=[ProvenanceRecord(**item) for item in link.get("provenance", [])],
                    status=str(link.get("status") or "candidate"),
                )
            )
        except Exception:
            continue
    if records:
        SQLiteDataCatalog(db_path).upsert_paper_experiment_links(records)


def _candidate_haystack(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("experiment_id") or ""),
        str(candidate.get("instrument") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("start_date") or ""),
        str(candidate.get("end_date") or ""),
        json.dumps(candidate.get("experiment_payload") or {}, sort_keys=True),
        " ".join(candidate.get("scan_types") or []),
        json.dumps(candidate.get("path_clues") or {}, sort_keys=True),
        json.dumps(candidate.get("hydrated_clue_summary") or {}, sort_keys=True),
    ]
    for row in candidate.get("datafile_samples", []):
        parts.extend([str(row.get("path") or ""), str(row.get("instrument") or ""), str(row.get("native_format") or ""), str(row.get("native_scan_id") or "")])
        payload = _safe_json(row.get("payload_json"))
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        parts.append(json.dumps(metadata or {}, sort_keys=True))
    for row in candidate.get("scan_samples", []):
        parts.extend([str(row.get("scan_id") or ""), str(row.get("scan_type") or "")])
        payload = _safe_json(row.get("payload_json"))
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        parts.append(json.dumps(metadata or {}, sort_keys=True))
    return " ".join(parts).lower()


def _hydrated_header_clue_summary(candidate: dict[str, Any], *, runner, max_files: int = 12) -> dict[str, Any]:
    paths = [Path(str(row.get("path") or "")).expanduser() for row in candidate.get("datafile_samples", []) if row.get("path")]
    attempted = 0
    read_count = 0
    missing_paths = 0
    read_errors: list[str] = []
    tool_status_counts: dict[str, int] = {}
    values_by_field: dict[str, dict[str, int]] = {field: {} for field in _BT7_READER_CLUE_FIELDS}
    scan_descr_fields: dict[str, dict[str, int]] = {field: {} for field in _SCAN_DESCR_CLUE_FIELDS}
    column_names: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    planned_npts: dict[str, int] = {}
    psd_map_statuses: dict[str, int] = {}
    line_scan_statuses: dict[str, int] = {}
    subgroups: dict[str, dict[str, Any]] = {}
    for path in paths[: max(0, max_files)]:
        attempted += 1
        if not path.exists() or not path.is_file():
            missing_paths += 1
            continue
        result = runner.run(
            ToolCall(
                tool_namespace="tas.data_reader",
                tool_name="read_bt7_file",
                arguments={"path": str(path)},
                caller_workflow="link-paper-to-experiments",
            )
        )
        status = str(getattr(result.status, "value", result.status))
        tool_status_counts[status] = tool_status_counts.get(status, 0) + 1
        if status != "success" or not isinstance(result.output, dict):
            read_errors.extend(f"{path.name}: {error.code}: {error.message}" for error in result.errors[:2])
            continue
        output = result.output
        metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
        read_count += 1
        _add_hydrated_subgroup(subgroups, path, metadata, output)
        for key in _BT7_READER_CLUE_FIELDS:
            value = metadata.get(key)
            if not _empty_reader_value(value):
                string_value = _short_json_value(value)
                values_by_field[key][string_value] = values_by_field[key].get(string_value, 0) + 1
        row_count = output.get("row_count")
        if row_count is not None:
            row_counts[str(row_count)] = row_counts.get(str(row_count), 0) + 1
        npts = metadata.get("npoints")
        if npts not in (None, ""):
            planned_npts[str(npts)] = planned_npts.get(str(npts), 0) + 1
        psd_map = output.get("psd_map") if isinstance(output.get("psd_map"), dict) else None
        psd_status = str(psd_map.get("status") if psd_map else "absent")
        psd_map_statuses[psd_status] = psd_map_statuses.get(psd_status, 0) + 1
        line_scan = output.get("line_scan") if isinstance(output.get("line_scan"), dict) else None
        line_status = str(line_scan.get("status") if line_scan else "absent")
        line_scan_statuses[line_status] = line_scan_statuses.get(line_status, 0) + 1
        descr = _parse_scan_descr(str(metadata.get("scan_description") or ""))
        for key, value in descr.items():
            if key in scan_descr_fields and value:
                scan_descr_fields[key][value] = scan_descr_fields[key].get(value, 0) + 1
        for column in output.get("columns") or []:
            column_names[column.lower()] = column_names.get(column.lower(), 0) + 1
    return {
        "status": "ok" if read_count else "missing_input",
        "attempted_file_count": attempted,
        "read_file_count": read_count,
        "missing_path_count": missing_paths,
        "tool_status_counts": _top_counts(tool_status_counts, 10),
        "read_errors": read_errors[:5],
        "reader_fields_seen": sorted(field for field, values in values_by_field.items() if values),
        "top_values": {field: _top_counts(values, 8) for field, values in values_by_field.items() if values},
        "scan_descr_values": {field: _top_counts(values, 8) for field, values in scan_descr_fields.items() if values},
        "column_names_sample": sorted(column_names)[:80],
        "column_count_estimate": len(column_names),
        "actual_row_count_values": _top_counts(row_counts, 8),
        "planned_npts_values": _top_counts(planned_npts, 8),
        "npts_policy": "ScanDescr Npts is planned point count; actual data coverage must use row_count from parsed data columns.",
        "psd_map_statuses": _top_counts(psd_map_statuses, 8),
        "line_scan_statuses": _top_counts(line_scan_statuses, 8),
        "reader_subgroups": _reader_subgroup_payloads(subgroups),
        "source_tool": "tas.data_reader.read_bt7_file",
        "clue_policy": "hydrated_registry_bt7_reader_clues_only; untrusted weak linkage evidence requiring human review",
    }


def _add_hydrated_subgroup(subgroups: dict[str, dict[str, Any]], path: Path, metadata: dict[str, Any], output: dict[str, Any]) -> None:
    expt_id = str(metadata.get("expt_id") or "unknown")
    expt_name = str(metadata.get("expt_name") or "unknown")
    key = f"{expt_id}|{expt_name}"
    subgroup = subgroups.setdefault(
        key,
        {
            "subgroup_key": key,
            "expt_id": expt_id,
            "expt_name": expt_name,
            "file_count": 0,
            "sample_files": [],
            "participants": {},
            "detector_modes": {},
            "scan_types": {},
            "scan_variables": {},
            "format_types": {},
            "actual_row_counts": {},
            "planned_npts": {},
        },
    )
    subgroup["file_count"] += 1
    if len(subgroup["sample_files"]) < 5:
        subgroup["sample_files"].append(path.name)
    for source_key, target_key in [
        ("expt_participants", "participants"),
        ("analyzer_detector_mode", "detector_modes"),
        ("scan_type", "scan_types"),
        ("scan_variable", "scan_variables"),
        ("format_type", "format_types"),
    ]:
        value = metadata.get(source_key)
        if not _empty_reader_value(value):
            bucket = subgroup[target_key]
            text = _short_json_value(value)
            bucket[text] = bucket.get(text, 0) + 1
    row_count = output.get("row_count")
    if row_count is not None:
        bucket = subgroup["actual_row_counts"]
        bucket[str(row_count)] = bucket.get(str(row_count), 0) + 1
    npts = metadata.get("npoints")
    if npts not in (None, ""):
        bucket = subgroup["planned_npts"]
        bucket[str(npts)] = bucket.get(str(npts), 0) + 1


def _reader_subgroup_payloads(subgroups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for subgroup in subgroups.values():
        payloads.append(
            {
                "subgroup_key": subgroup["subgroup_key"],
                "expt_id": subgroup["expt_id"],
                "expt_name": subgroup["expt_name"],
                "file_count": subgroup["file_count"],
                "sample_files": subgroup["sample_files"],
                "participants": _top_counts(subgroup["participants"], 5),
                "detector_modes": _top_counts(subgroup["detector_modes"], 5),
                "scan_types": _top_counts(subgroup["scan_types"], 5),
                "scan_variables": _top_counts(subgroup["scan_variables"], 5),
                "format_types": _top_counts(subgroup["format_types"], 5),
                "actual_row_counts": _top_counts(subgroup["actual_row_counts"], 5),
                "planned_npts": _top_counts(subgroup["planned_npts"], 5),
            }
        )
    return sorted(payloads, key=lambda item: (-int(item["file_count"]), str(item["subgroup_key"])))[:12]


def _short_json_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)[:240]
    return str(value)[:240]


def _empty_reader_value(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return True
    return False


def _normalize_header_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _parse_scan_descr(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in str(value or "").split(":"):
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        normalized = _normalize_header_key(key)
        if normalized and raw.strip():
            fields[normalized] = " ".join(raw.strip().split())
    return fields


def _hydrated_header_score(candidate: dict[str, Any], paper: dict[str, Any] | None, constraint: dict[str, list[str]], evidence: list[str]) -> float:
    clues = candidate.get("hydrated_clue_summary") or {}
    if not clues or int(clues.get("read_file_count") or 0) == 0:
        return 0.0
    clue_text = json.dumps(clues, sort_keys=True).lower()
    score = 0.12
    matched_terms: list[str] = []
    for term in _terms(str((paper or {}).get("title") or "")):
        if len(term) >= 3 and term not in _GENERIC_TITLE_TERMS and term in clue_text:
            matched_terms.append(term)
    if matched_terms:
        evidence.append(f"Matched paper-title terms in hydrated BT-7 reader metadata: {', '.join(sorted(set(matched_terms))[:8])}.")
        score += min(0.35, len(set(matched_terms)) / 8)
    for label, values, weight in [
        ("detector mode", constraint.get("detector_modes", []), 0.22),
        ("measurement kind", constraint.get("measurement_kinds", []), 0.18),
        ("sample environment", constraint.get("sample_environment_terms", []), 0.12),
    ]:
        matched = [value for value in values if str(value).lower() in clue_text]
        if matched:
            evidence.append(f"Matched {label} in hydrated BT-7 reader metadata: {', '.join(sorted(matched))}.")
            score += weight
    authors = _paper_author_terms(paper)
    matched_authors = [author for author in authors if author in clue_text]
    if matched_authors:
        evidence.append(f"Matched author/experimenter surnames in hydrated BT-7 reader metadata: {', '.join(sorted(set(matched_authors))[:6])}.")
        score += min(0.22, len(set(matched_authors)) / 10)
    return min(1.0, score)


def _paper_author_terms(paper: dict[str, Any] | None) -> list[str]:
    metadata = (paper or {}).get("metadata") or {}
    raw_values = []
    for key in ["authors", "author", "creators"]:
        value = (paper or {}).get(key) or metadata.get(key)
        if isinstance(value, list):
            raw_values.extend(str(item) for item in value)
        elif value:
            raw_values.extend(re.split(r";|,|\band\b", str(value)))
    surnames = []
    for value in raw_values:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]+", value)]
        if tokens:
            surnames.append(tokens[-1])
    return sorted(set(token for token in surnames if len(token) >= 3))


_BT7_READER_CLUE_FIELDS = {
    "filename",
    "date_str",
    "expt_id",
    "expt_name",
    "expt_participants",
    "user",
    "comment",
    "scan_type",
    "scan_variable",
    "scan_description",
    "npoints",
    "h_center",
    "k_center",
    "l_center",
    "e_center",
    "lattice",
    "orient",
    "fixed_e",
    "fixed_e_value",
    "analyzer_detector_mode",
    "temp_start",
    "temp_mean",
    "temp_setpoint",
    "format_type",
}


_SCAN_DESCR_CLUE_FIELDS = {"subid", "jtype", "filename", "range", "title", "comment", "npts"}


def _merged_constraints(constraints: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged = {
        "instrument": [],
        "detector_modes": [],
        "measurement_kinds": [],
        "sample_environment_terms": [],
        "formula_like_title_strings": [],
    }
    for item in constraints:
        for key in merged:
            values = item.get(key) or []
            if isinstance(values, str):
                values = [values]
            merged[key].extend(str(value) for value in values if value)
    return {key: sorted(set(values)) for key, values in merged.items()}


def _term_match_score(values: list[str], haystack: str, evidence: list[str], label: str) -> float:
    if not values:
        return 0.0
    matched = [value for value in values if str(value).lower() in haystack]
    if matched:
        evidence.append(f"Matched {label}: {', '.join(sorted(matched))}.")
        return min(1.0, len(matched) / max(1, len(values)))
    return 0.0


def _material_or_title_score(paper: dict[str, Any] | None, constraint: dict[str, list[str]], haystack: str, evidence: list[str]) -> float:
    formula_matches = [value for value in constraint["formula_like_title_strings"] if str(value).lower() in haystack]
    if formula_matches:
        evidence.append(f"Matched formula-like title/catalog strings: {', '.join(sorted(formula_matches))}.")
        return 1.0
    title = str((paper or {}).get("title") or "")
    title_terms = [term for term in _terms(title) if len(term) >= 3 and term not in _GENERIC_TITLE_TERMS]
    matched = [term for term in title_terms if term in haystack]
    if matched:
        evidence.append(f"Matched paper-title terms in catalog metadata/path clues: {', '.join(sorted(set(matched))[:8])}.")
        return min(0.85, len(set(matched)) / max(4, len(set(title_terms))))
    return 0.0


def _group_specificity_score(candidate: dict[str, Any]) -> float:
    count = int(candidate.get("datafile_count") or 0)
    clues = candidate.get("path_clues") or {}
    years = clues.get("data_years") or []
    if count == 0:
        return 0.0
    if count <= 250 and len(years) <= 3:
        return 1.0
    if count <= 2500 and len(years) <= 6:
        return 0.65
    if count <= 10000:
        return 0.35
    return 0.05


def _mixed_archive_penalty(candidate: dict[str, Any]) -> float:
    count = int(candidate.get("datafile_count") or 0)
    clues = candidate.get("path_clues") or {}
    years = clues.get("data_years") or []
    tas_count = int(clues.get("tas_data_file_count") or 0)
    penalty = 0.0
    if count > 50000 or len(years) > 10:
        penalty += 0.22
    elif count > 10000 or len(years) > 6:
        penalty += 0.12
    if count > 0 and tas_count == 0:
        penalty += 0.08
    return min(0.3, penalty)


def _hydrated_subgroup_mixed_penalty(candidate: dict[str, Any]) -> float:
    clues = candidate.get("hydrated_clue_summary") or {}
    subgroups = clues.get("reader_subgroups") or []
    if not subgroups:
        return 0.0
    known = [item for item in subgroups if item.get("expt_id") != "unknown" or item.get("expt_name") != "unknown"]
    if len(known) <= 1:
        return 0.0
    total = sum(int(item.get("file_count") or 0) for item in known)
    top = max(int(item.get("file_count") or 0) for item in known)
    penalty = min(0.12, 0.035 * (len(known) - 1))
    if total and top / total < 0.6:
        penalty += 0.04
    return min(0.18, penalty)


def _material_mismatch_penalty(constraint: dict[str, list[str]], material_title_score: float) -> float:
    if constraint.get("formula_like_title_strings") and material_title_score < 0.5:
        return 0.12
    return 0.0


def _source_unit_mismatch_penalty(candidate: dict[str, Any], constraint: dict[str, list[str]]) -> float:
    source_unit = (candidate.get("hydrated_clue_summary") or {}).get("source_unit") or {}
    if not source_unit:
        return 0.0
    family = str(source_unit.get("family") or "")
    detector = str(source_unit.get("detector") or "")
    penalty = 0.0
    detector_modes = {str(item).lower() for item in constraint.get("detector_modes", [])}
    measurement_kinds = {str(item).lower() for item in constraint.get("measurement_kinds", [])}
    if "psd" in detector_modes and not detector.startswith("psd"):
        penalty += 0.18
    if {"elastic", "diffraction"} & measurement_kinds and "inelastic" in family:
        penalty += 0.25
    if "psd" in detector_modes and "psd" not in family and detector.startswith("psd"):
        penalty += 0.06
    return min(0.35, penalty)


def _hydrated_parent_group_penalty(candidate: dict[str, Any]) -> float:
    if candidate.get("candidate_source") != "bt7_hydrated_index":
        return 0.0
    source_unit_count = int((candidate.get("hydrated_clue_summary") or {}).get("source_unit_count") or 0)
    if source_unit_count > 1:
        return 0.08
    return 0.0


_GENERIC_TITLE_TERMS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "using",
    "study",
    "effect",
    "effects",
    "magnetic",
    "magnetism",
    "order",
    "spin",
    "scattering",
    "neutron",
    "temperature",
    "field",
    "phase",
    "diagram",
    "dynamics",
    "structure",
    "transition",
    "transitions",
    "source",
    "regression",
    "paper",
    "figure",
    "synthetic",
    "candidate",
    "data",
    "experiment",
    "experiments",
    "scan",
    "scans",
}


def _path_clue_summary(paths: list[str]) -> dict[str, Any]:
    directory_tokens: dict[str, int] = {}
    filename_prefixes: dict[str, int] = {}
    file_extensions: dict[str, int] = {}
    years = set()
    proposal_ids = set()
    tas_data_file_count = 0
    for path in paths:
        path_obj = Path(path)
        years.update(_years_from_path_parts(path_obj))
        proposal_ids.update(match.group(1) for match in re.finditer(r"/((?:1|2)\d{4})(?:/|$)", path))
        suffix = path_obj.suffix.lower()
        if suffix:
            file_extensions[suffix] = file_extensions.get(suffix, 0) + 1
        if suffix in _TAS_DATA_EXTENSIONS:
            tas_data_file_count += 1
            stem = path_obj.stem.lower()
            prefix = re.sub(r"\d+$", "", stem).strip("_-.")
            if prefix and prefix not in {"bt7", "scan", "data"}:
                filename_prefixes[prefix] = filename_prefixes.get(prefix, 0) + 1
        for part in path_obj.parts:
            token = part.lower()
            if token in _PATH_STOP_TOKENS or token.endswith(".bt7"):
                continue
            if re.fullmatch(r"\d+", token):
                continue
            for subtoken in re.split(r"[^a-zA-Z0-9]+", token):
                if len(subtoken) >= 3 and subtoken not in _PATH_STOP_TOKENS and not subtoken.isdigit():
                    directory_tokens[subtoken] = directory_tokens.get(subtoken, 0) + 1
    return {
        "data_years": sorted(year for year in years if 1900 <= year <= 2100),
        "proposal_ids": sorted(proposal_ids),
        "top_directory_tokens": _top_counts(directory_tokens, 20),
        "top_filename_prefixes": _top_counts(filename_prefixes, 20),
        "file_extensions": _top_counts(file_extensions, 10),
        "tas_data_file_count": tas_data_file_count,
        "clue_policy": "historic_path_filename_clues_only; weak evidence requiring hydration or human review",
    }


def _years_from_path_parts(path: Path) -> list[int]:
    years = set()
    for part in path.parts:
        token = part.strip()
        if re.fullmatch(r"(19|20)\d{2}", token):
            years.add(int(token))
        elif re.fullmatch(r"(19|20)\d{4}", token):
            years.add(int(token[:4]))
    return sorted(year for year in years if 1900 <= year <= 2100)


_TAS_DATA_EXTENSIONS = {".bt7", ".ng5", ".dat"}


_PATH_STOP_TOKENS = {
    "",
    "/",
    "users",
    "williamratcliff",
    "bt7",
    "from",
    "stampede",
    "data",
    "dat",
    "user",
    "user1",
    "user2",
    "user3",
    "guest",
    "guest1",
    "guest2",
    "guest3",
    "ncnr",
    "console",
    "history",
    "config",
    "logs",
    "log",
    "html",
    "txt",
    "com",
    "git",
    "local",
    "runs",
}


def _top_counts(counts: dict[str, int], limit: int) -> list[dict[str, Any]]:
    return [{"value": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _safe_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def publication_data_lag_prior(paper_id: str, publication_year: int | None) -> PublicationDataLagPrior:
    center = int(publication_year - 3) if publication_year else None
    start = int(publication_year - 6) if publication_year else None
    end = int(publication_year - 1) if publication_year else None
    cap_applied = bool(publication_year and publication_year > 2021)
    if cap_applied and end and end > 2021:
        end = 2021
    if cap_applied and center and center > 2021:
        center = 2021
    return PublicationDataLagPrior(
        prior_id=f"pub-data-lag-{_hash(paper_id + str(publication_year))}",
        paper_id=paper_id,
        publication_year=publication_year,
        prior_center_year=center,
        plausible_start_year=start,
        plausible_end_year=end,
        ncnr_shutdown_cap_date="2021-02-03" if cap_applied else None,
        ncnr_cap_applied=cap_applied,
        confidence=0.35,
        override_evidence_types=["methods_date", "proposal_id", "experiment_id", "scan_timestamp", "paper_mapping", "human_review"],
        notes=[
            "Anecdotal prior: data-to-publication lag is often around three years and varies by subfield.",
            "This is a probabilistic prior for ranking, not a causal claim and not a hard filter.",
            "Publication year must not be treated as data-acquisition year evidence.",
        ],
        provenance=[ProvenanceRecord(source_type="paper", source_id=paper_id, method="publication_data_lag_prior")],
    )


def _apply_lag_prior_to_mapping(mapping: dict[str, Any], lag_prior: PublicationDataLagPrior) -> dict[str, Any]:
    result = dict(mapping)
    prior_score = lag_prior.confidence
    date_score = _mapping_date_score(result, lag_prior)
    existing = float(result.get("confidence", 0.0) or 0.0)
    result["publication_data_lag_prior_score"] = prior_score
    if date_score is None:
        result["probability_score"] = round(min(1.0, 0.8 * existing + 0.2 * prior_score), 3)
    else:
        result["publication_data_lag_date_score"] = date_score
        result["probability_score"] = round(min(1.0, 0.65 * existing + 0.2 * prior_score + 0.15 * date_score), 3)
    result.setdefault("score_breakdown", {})
    result["score_breakdown"]["mapping_confidence"] = existing
    result["score_breakdown"]["publication_data_lag_prior"] = prior_score
    if date_score is not None:
        result["score_breakdown"]["publication_data_lag_date"] = date_score
    result.setdefault("warnings", [])
    if lag_prior.ncnr_cap_applied:
        result["warnings"].append("Post-2021 publication: NCNR/BT-7 acquisition date should be capped at Feb. 3, 2021 unless explicit evidence says otherwise.")
    if date_score == 0.0:
        result["warnings"].append("Experiment date evidence is outside the broad publication-to-data lag prior window; keep this as weak negative evidence unless stronger mapping evidence exists.")
    return result


def _mapping_date_score(mapping: dict[str, Any], lag_prior: PublicationDataLagPrior) -> float | None:
    years = [int(year) for year in mapping.get("data_years", []) if str(year).isdigit()]
    if years and lag_prior.plausible_start_year is not None and lag_prior.plausible_end_year is not None:
        scores = []
        for year in years:
            trial = dict(mapping)
            trial["data_year"] = year
            trial.pop("data_years", None)
            score = _mapping_date_score(trial, lag_prior)
            if score is not None:
                scores.append(score)
        return max(scores) if scores else None
    year = _mapping_year(mapping)
    if year is None or lag_prior.plausible_start_year is None or lag_prior.plausible_end_year is None:
        return None
    if lag_prior.plausible_start_year <= year <= lag_prior.plausible_end_year:
        if lag_prior.prior_center_year is not None and abs(year - lag_prior.prior_center_year) <= 1:
            return 1.0
        return 0.75
    if lag_prior.plausible_start_year - 1 <= year <= lag_prior.plausible_end_year + 1:
        return 0.35
    return 0.0


def _mapping_year(mapping: dict[str, Any]) -> int | None:
    for key in ["experiment_year", "data_year", "scan_year", "year"]:
        raw = mapping.get(key)
        if raw is not None:
            try:
                return int(str(raw)[:4])
            except (TypeError, ValueError):
                pass
    for key in ["experiment_date", "data_date", "scan_date", "start_date", "end_date"]:
        raw = mapping.get(key)
        if raw:
            match = re.search(r"(19|20)\d{2}", str(raw))
            if match:
                return int(match.group(0))
    return None


def _years_from_text(text: str) -> list[int]:
    years = {int(match.group(0)) for match in re.finditer(r"\b(19|20)\d{2}\b", text)}
    years.update(int(match.group(1)[:4]) for match in re.finditer(r"(?<!\d)((?:19|20)\d{4})(?!\d)", text))
    return sorted(year for year in years if 1900 <= year <= 2100)


def _paper_record(catalog_dir: Path, paper_id: str) -> dict[str, Any] | None:
    for path in [catalog_dir / "papers.jsonl", catalog_dir / "documents.jsonl"]:
        for item in _load_jsonl_dicts(path):
            if item.get("paper_id") == paper_id or item.get("document_id") == paper_id:
                return item
    return None


def run_corpus_pilot_eval(arguments: dict[str, Any]) -> dict[str, Any]:
    suite_path = Path(arguments["suite"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    catalog = arguments.get("catalog") or suite.get("catalog")
    if not catalog:
        fixture = suite.get("fixture_manifest", "tests/fixtures/corpus_pilot/pilot_manifest.synthetic.yaml")
        catalog = output_dir / "corpus_catalog"
        ingest_pilot_corpus({"manifest": fixture, "catalog": str(catalog), "output_dir": str(output_dir / "ingest")})
    checks = []
    catalog_report = corpus_pilot_report(catalog)
    for name, observed_key in [
        ("min_documents", "documents"),
        ("min_source_chunks", "source_chunks"),
        ("min_evidence_graph_edges", "evidence_graph_edges"),
        ("min_tas_concept_links", "tas_concept_links"),
        ("min_paper_constraints", "paper_constraints"),
    ]:
        if name in suite:
            observed = int(catalog_report.get(observed_key, 0))
            expected_min = int(suite.get(name, 0))
            checks.append({"name": name, "status": "pass" if observed >= expected_min else "fail", "expected_min": expected_min, "observed": observed})
    if suite.get("expected_post_2021_publications"):
        observed = int(catalog_report.get("post_2021_publications", 0))
        checks.append({"name": "post_2021_publications_flagged", "status": "pass" if observed > 0 else "fail", "observed": observed})
    for task in suite.get("tasks", []):
        search_args = {"catalog": str(catalog), "query": task.get("query", ""), "top_k": task.get("top_k", 5)}
        for key in ["paper_id", "document_type", "chunk_type", "figure_id", "panel_id", "tas_concept", "year", "instrument", "material_family", "requires_source_support", "exclude_references"]:
            if key in task:
                search_args[key] = task[key]
        result = corpus_search(search_args)
        result_ids = {item["source_id"] for item in result["results"]}
        expected = set(task.get("expected_source_ids", []))
        forbidden = set(task.get("forbidden_source_ids", []))
        provenance_ok = all(item.get("provenance") for item in result["results"] if item["source_id"] in expected)
        min_results_ok = len(result["results"]) >= int(task.get("minimum_result_count", 0))
        required_warning_terms = set(task.get("required_warning_terms", []))
        warning_terms = {str(item.get("term")) for item in result.get("warnings", [])}
        warnings_ok = required_warning_terms.issubset(warning_terms)
        required_chunk_types = set(task.get("required_chunk_types", []))
        observed_chunk_types = {
            str((item.get("structured_summary") or {}).get("chunk_type"))
            for item in result["results"]
            if (item.get("structured_summary") or {}).get("chunk_type")
        }
        chunk_types_ok = required_chunk_types.issubset(observed_chunk_types)
        forbidden_chunk_types = set(task.get("forbidden_chunk_types", []))
        forbidden_chunk_types_ok = not (forbidden_chunk_types & observed_chunk_types)
        forbidden_title_terms = {str(term).lower() for term in task.get("forbidden_title_terms", [])}
        observed_titles = [str(item.get("title") or "").lower() for item in result["results"]]
        forbidden_title_terms_ok = not any(term in title for term in forbidden_title_terms for title in observed_titles)
        minimum_private = int(task.get("minimum_private_local_results", 0) or 0)
        private_count = sum(1 for item in result["results"] if (item.get("structured_summary") or {}).get("private_local_only"))
        private_ok = private_count >= minimum_private
        required_methods = set(task.get("required_retrieval_methods", []))
        observed_methods = {str(item.get("retrieval_method")) for item in result["results"]}
        methods_ok = required_methods.issubset(observed_methods)
        status = (
            "pass"
            if expected.issubset(result_ids)
            and not (forbidden & result_ids)
            and provenance_ok
            and min_results_ok
            and warnings_ok
            and chunk_types_ok
            and forbidden_chunk_types_ok
            and forbidden_title_terms_ok
            and private_ok
            and methods_ok
            else "fail"
        )
        checks.append(
            {
                "name": task.get("name"),
                "status": status,
                "query": task.get("query"),
                "expected_source_ids": sorted(expected),
                "forbidden_source_ids": sorted(forbidden),
                "minimum_result_count": task.get("minimum_result_count"),
                "result_count": len(result["results"]),
                "required_warning_terms": sorted(required_warning_terms),
                "warning_terms": sorted(warning_terms),
                "required_chunk_types": sorted(required_chunk_types),
                "observed_chunk_types": sorted(observed_chunk_types),
                "forbidden_chunk_types": sorted(forbidden_chunk_types),
                "minimum_private_local_results": minimum_private,
                "private_local_result_count": private_count,
                "forbidden_title_terms": sorted(forbidden_title_terms),
                "required_retrieval_methods": sorted(required_methods),
                "observed_retrieval_methods": sorted(observed_methods),
                "result_ids": list(result_ids),
                "top_results": result["results"],
                "warnings": result.get("warnings", []),
            }
        )
    report = {
        "workflow_name": "corpus-pilot-eval",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "suite": str(suite_path),
        "catalog": str(catalog),
        "check_count": len(checks),
        "checks": checks,
        "catalog_report": catalog_report,
        "provenance_coverage": catalog_report.get("provenance_coverage"),
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_eval_markdown, json_filename="corpus_pilot_eval.json", markdown_filename="corpus_pilot_eval.md")
    return report


def agent_readiness(arguments: dict[str, Any]) -> dict[str, Any]:
    corpus_catalog = arguments.get("corpus_catalog")
    data_catalog = arguments.get("data_catalog")
    output_dir = Path(arguments["output_dir"]).expanduser()
    corpus = corpus_pilot_report(corpus_catalog) if corpus_catalog else {}
    checks = [
        {"name": "pilot_corpus_has_chunks", "status": "pass" if corpus.get("source_chunks", 0) > 0 else "fail"},
        {"name": "source_support_present", "status": "pass" if corpus.get("source_support_records", 0) > 0 else "fail"},
        {"name": "evidence_graph_present", "status": "pass" if corpus.get("evidence_graph_edges", 0) > 0 else "fail"},
        {"name": "data_catalog_separate", "status": "pass" if data_catalog and str(data_catalog) != str(corpus_catalog) else "warning"},
        {"name": "no_agent_layer_built", "status": "pass"},
    ]
    report = {
        "workflow_name": "corpus-agent-readiness",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "warning",
        "checks": checks,
        "safe_conclusion": "A bounded planner/router should wait until corpus evals, source support, false-positive tracking, and provenance-rich workflow outputs are stable.",
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_agent_markdown, json_filename="agent_readiness.json", markdown_filename="agent_readiness.md")
    return report


def _document_record(item: dict[str, Any], manifest_path: Path) -> Document:
    doc_id = item.get("document_id") or item.get("paper_id")
    local_path = item.get("local_path")
    resolved = Path(local_path).expanduser() if local_path and not str(local_path).startswith("[") else None
    warnings = []
    status = "path_recorded"
    source_hash = None
    extraction_method = "manifest_path_reference"
    if resolved and resolved.exists() and resolved.is_file() and resolved.suffix.lower() in {".txt", ".md"}:
        source_hash = _file_hash(resolved)
        status = "text_available"
        extraction_method = "safe_text_file_reference"
    elif resolved and resolved.exists() and resolved.is_file() and resolved.suffix.lower() == ".pdf":
        source_hash = _file_hash(resolved)
        status = "pdf_available"
        extraction_method = "pdftotext_optional_local"
    elif item.get("copyright_status") == "private_local_only":
        status = "missing_local_path"
        warnings.append("Private local document path is absent or placeholder; ingestion recorded metadata only.")
    metadata = {
        "doi": item.get("doi"),
        "year": item.get("year"),
        "expected_features": item.get("expected_features", []),
        "include_in_tests": item.get("include_in_tests", False),
        "notes": item.get("notes", []),
        "warnings": warnings,
    }
    if item.get("year") and int(item.get("year")) > 2021:
        metadata.setdefault("warnings", []).append("Publication is after the Feb. 3, 2021 NCNR reactor incident; do not infer NCNR data acquisition after that date without explicit evidence.")
    return Document(
        document_id=str(doc_id),
        document_type=str(item.get("type") or "paper"),
        title=item.get("title"),
        paper_id=item.get("paper_id"),
        local_path=local_path,
        source_hash=source_hash,
        copyright_status=item.get("copyright_status"),
        ingest_status=status,
        extraction_method=extraction_method,
        parser_version="corpus_pilot_v0",
        metadata=metadata,
        provenance=[ProvenanceRecord(source_type="corpus_manifest", source_id=str(manifest_path), method="document_manifest_load")],
    )


def _extract_manifest_document_chunks(documents: list[Document], manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    settings = manifest.get("extraction", {}) or {}
    max_pages = int(settings.get("max_pages", DEFAULT_PRIVATE_PDF_PAGE_LIMIT))
    max_chars = int(settings.get("max_chars_per_document", DEFAULT_PRIVATE_TEXT_LIMIT))
    chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    for document in documents:
        if document.ingest_status not in {"text_available", "pdf_available"} or not document.local_path:
            continue
        path = Path(document.local_path).expanduser()
        if path.suffix.lower() == ".pdf":
            text, warning = _extract_pdf_text(path, max_pages=max_pages, max_chars=max_chars)
            if warning:
                warnings.append(f"{document.document_id}: {warning}")
            if not text:
                continue
            method = "pdftotext_first_pages"
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            method = "safe_text_file_first_chars"
        else:
            continue
        chunks.extend(_chunks_from_text(document, text, method=method))
    return chunks, warnings


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_review_assist_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [item for item in payload.get("items", []) if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _write_paper_review_outcome_store(
    store_path: Path,
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    batch_method: str,
) -> str:
    source_hash = f"sha256:{_file_hash(source_path)}" if source_path.exists() else None
    labels = [_paper_review_row_to_label(row, source_path=source_path, source_hash=source_hash) for row in rows]
    batch = ReviewLabelBatch(
        batch_id=f"review-label-batch-{_hash(str(source_path) + batch_method)}",
        source_template_paths=[str(source_path)],
        labels=labels,
        provenance=[ProvenanceRecord(source_type="review_template", source_id=str(source_path), method=batch_method)],
    )
    ReviewOutcomeSQLiteStore(store_path).upsert_batch(batch)
    return str(store_path)


def _paper_review_row_to_label(row: dict[str, Any], *, source_path: Path, source_hash: str | None) -> ReviewLabel:
    status = str(row.get("human_review_status") or "unreviewed")
    paper_id = str(row.get("paper_id") or "unknown-paper")
    experiment_id = str(row.get("candidate_experiment_id") or "unknown-experiment")
    source_artifact_path = row.get("source_report") or row.get("source_report_json") or str(source_path)
    artifact_path = Path(str(source_artifact_path)).expanduser()
    artifact_hash = f"sha256:{_file_hash(artifact_path)}" if artifact_path.exists() else source_hash
    return ReviewLabel(
        label_id=f"review-label-{_hash('|'.join([str(source_path), paper_id, experiment_id, str(row.get('candidate_rank') or '')]))}",
        object_type="paper_experiment_link",
        object_id=paper_id,
        object_secondary_id=experiment_id,
        source_workflow="link-paper-to-experiments",
        source_artifact_path=str(source_artifact_path),
        source_artifact_hash=artifact_hash,
        verdict=_paper_review_status_to_verdict(status),
        type_specific_status=None if status == "unreviewed" else status,
        type_specific_allowed=[str(item) for item in row.get("allowed_human_review_statuses") or []],
        claim_state=_paper_review_status_to_claim_state(status),
        suggested_status=str(row.get("candidate_status") or "candidate"),
        # Preserve reviewer-selected channels from exported dashboard rows; the
        # paper-text + hydrated-metadata pair is only the initial default.
        evidence_channels_relied_on=[
            str(item)
            for item in (
                row.get("evidence_channels_relied_on")
                or row.get("evidence_channels")
                or ["evidence_channel_paper_text", "evidence_channel_hydrated_metadata"]
            )
        ],
        counter_evidence_observed=_paper_review_counter_evidence(row),
        recommendation=_paper_review_status_to_recommendation(status),
        uncertainty_class="probability_style_triage_prior",
        review_notes=str(row.get("review_notes") or ""),
        reviewer_id=row.get("reviewer"),
        reviewed_at=_parse_reviewed_at(row.get("reviewed_at")),
        legacy_provenance=row.get("provenance"),
        provenance=[ProvenanceRecord(source_type="review_template", source_id=str(source_path), method="paper_review_row_to_label")],
        metadata={
            "paper_title": row.get("paper_title"),
            "candidate_rank": row.get("candidate_rank"),
            "probability_score": row.get("probability_score"),
            "confidence_label": row.get("confidence_label"),
            "review_priority": row.get("review_priority"),
        },
    )


def _paper_review_status_to_verdict(status: str) -> str:
    if status == "accepted":
        return "positive"
    # A duplicate paper-link row is negative for this specific candidate because
    # it adds no independent linkage evidence; the duplicate target can still be
    # reviewed separately as positive, negative, or superseded.
    if status in {"rejected", "duplicate"}:
        return "negative"
    if status in {"needs_more_evidence", "not_enough_information"}:
        return "inconclusive"
    return "unreviewed"


def _paper_review_status_to_claim_state(status: str) -> str:
    if status == "accepted":
        return "accepted"
    # Keep duplicate as rejected for this row, not superseded globally. The
    # duplicate-of target remains a separate review label with its own verdict.
    if status in {"rejected", "duplicate"}:
        return "rejected"
    return "candidate"


def _paper_review_status_to_recommendation(status: str) -> str:
    if status in {"rejected", "duplicate"}:
        return "downweight"
    if status in {"needs_more_evidence", "not_enough_information"}:
        return "escalate"
    if status == "accepted":
        return "calibrate_prior"
    return "no_action"


def _paper_review_counter_evidence(row: dict[str, Any]) -> list[str]:
    warnings = " ".join(str(item).lower() for item in row.get("warnings") or [])
    evidence = []
    if "calibration" in warnings:
        evidence.append("counter_evidence_calibration_unreviewed")
    return evidence


def _parse_reviewed_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _write_paper_experiment_review_rows_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "review_id",
        "paper_id",
        "paper_title",
        "candidate_experiment_id",
        "candidate_rank",
        "candidate_status",
        "confidence_label",
        "probability_score",
        "review_priority",
        "human_review_status",
        "evidence_to_check",
        "warnings",
        "cautions",
        "review_questions",
        "review_packet",
        "source_report_json",
        "source_report_md",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: json.dumps(record.get(field), sort_keys=True) if isinstance(record.get(field), (list, dict)) else record.get(field)
                    for field in fields
                }
            )


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_macos_sidecar_row(row: dict[str, Any]) -> bool:
    pdf_name = str(row.get("pdf_name") or Path(str(row.get("pdf") or "")).name)
    return Path(pdf_name).name.startswith("._")


def _quality_by_pdf(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return {str(row.get("pdf_name")): row for row in rows if row.get("pdf_name")}


def _grobid_bibliographic_metadata(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    tei_path, _, parser_version = _extractor_text_path(run_dir, row, "grobid")
    if tei_path is None or not tei_path.exists():
        return {"status": "missing_tei"}
    try:
        root = ET.fromstring(tei_path.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError as exc:
        return {"status": "parse_error", "warning": str(exc)}
    namespaces = {"tei": "http://www.tei-c.org/ns/1.0"}
    title = _tei_text(root.find(".//tei:titleStmt/tei:title[@level='a']", namespaces)) or _tei_text(root.find(".//tei:titleStmt/tei:title", namespaces))
    source_bibl = root.find(".//tei:sourceDesc/tei:biblStruct[@status='extracted']", namespaces) or root.find(".//tei:sourceDesc/tei:biblStruct", namespaces)
    doi = None
    idno_parent = source_bibl if source_bibl is not None else root
    for idno in idno_parent.findall(".//tei:idno", namespaces):
        if str(idno.attrib.get("type", "")).lower() == "doi":
            doi = _tei_text(idno)
            break
    authors = []
    author_parent = source_bibl if source_bibl is not None else root
    for author in author_parent.findall(".//tei:analytic/tei:author", namespaces):
        name = " ".join(part for part in [_tei_text(author.find(".//tei:forename[@type='first']", namespaces)), _tei_text(author.find(".//tei:forename[@type='middle']", namespaces)), _tei_text(author.find(".//tei:surname", namespaces))] if part)
        if name:
            authors.append(name)
    years_with_source = _tei_year_candidates(root, namespaces, source_bibl)
    year = years_with_source[0][0] if years_with_source else None
    year_source = years_with_source[0][1] if years_with_source else None
    return {
        "status": "ok",
        "title": title,
        "doi": doi,
        "authors": authors,
        "year": year,
        "year_source": year_source,
        "parser_version": parser_version,
        "tei_path": str(tei_path),
    }


def _tei_text(element) -> str | None:
    if element is None:
        return None
    text = " ".join(" ".join(element.itertext()).split())
    return text or None


def _tei_year_candidates(root, namespaces: dict[str, str], source_bibl=None) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for date in root.findall(".//tei:publicationStmt/tei:date", namespaces):
        for raw in [date.attrib.get("when"), date.attrib.get("from"), date.attrib.get("to"), _tei_text(date)]:
            for year in _years_from_text(str(raw or "")):
                candidates.append((year, "grobid_publication_stmt_date"))
    bibl = source_bibl if source_bibl is not None else root.find(".//tei:sourceDesc/tei:biblStruct[@status='extracted']", namespaces)
    if bibl is not None:
        for date in bibl.findall(".//tei:date", namespaces):
            for raw in [date.attrib.get("when"), date.attrib.get("from"), date.attrib.get("to"), _tei_text(date)]:
                for year in _years_from_text(str(raw or "")):
                    candidates.append((year, "grobid_source_bibl_date"))
        notes = bibl.findall(".//tei:note", namespaces)
    else:
        notes = []
    for note in notes:
        for year in _years_from_text(_tei_text(note) or ""):
            candidates.append((year, f"grobid_source_bibl_note:{note.attrib.get('type', 'unknown')}"))
    if not candidates:
        file_desc_text = _tei_text(root.find(".//tei:fileDesc", namespaces)) or ""
        for year in _years_from_text(file_desc_text[:5000]):
            candidates.append((year, "grobid_file_desc_text_fallback"))
    seen = set()
    unique = []
    for item in candidates:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique


def _paper_constraint_record(catalog_dir: Path, paper_id: str, title: str | None, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    constraint_evidence = []
    chunk_ids_by_type: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_type = str(chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type") or "unknown")
        chunk_ids_by_type.setdefault(chunk_type, []).append(str(chunk.get("chunk_id")))
    text_by_chunk = {str(chunk.get("chunk_id")): " ".join([str(chunk.get("title") or ""), str(chunk.get("text") or "")]) for chunk in chunks}
    joined = "\n".join(text_by_chunk.values())
    lowered = joined.lower()

    instruments = _matched_terms(text_by_chunk, {"BT-7": [r"\bbt-?7\b"], "NCNR": [r"\bncnr\b"], "triple-axis": [r"\btriple[- ]axis\b"]})
    detector_modes = _matched_terms(text_by_chunk, {"PSD": [r"\bpsd\b", r"position sensitive detector"], "single_detector": [r"\bsingle[- ]detector\b"]})
    measurement_kinds = _matched_terms(text_by_chunk, {"elastic": [r"\belastic\b"], "inelastic": [r"\binelastic\b"], "diffraction": [r"\bdiffraction\b"], "scattering": [r"\bscattering\b"]})
    reduction_terms = _matched_terms(text_by_chunk, {"background": [r"\bbackground\b"], "normalization": [r"\bnormalization\b", r"\bnormalized\b"], "calibration": [r"\bcalibration\b", r"\bcalibrated\b"], "resolution": [r"\bresolution\b"]})
    sample_environment = _matched_terms(text_by_chunk, {"temperature": [r"\btemperature\b", r"\bkelvin\b", r"\bK\b"], "magnetic_field": [r"\bmagnetic field\b", r"\btesla\b"]})
    formula_candidates = _formula_like_title_candidates(title or "")
    ranges = {
        "temperature_k": sorted(set(re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*K\b", joined, flags=re.IGNORECASE))),
        "field_t": sorted(set(re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*T\b", joined, flags=re.IGNORECASE))),
    }

    for kind, matches in [
        ("instrument", instruments),
        ("detector_mode", detector_modes),
        ("measurement_kind", measurement_kinds),
        ("data_reduction", reduction_terms),
        ("sample_environment", sample_environment),
    ]:
        for value, chunk_ids in matches.items():
            constraint_evidence.append({"constraint_type": kind, "value": value, "source_chunk_ids": sorted(chunk_ids), "status": "proposed", "confidence": 0.65})
    for formula in formula_candidates:
        constraint_evidence.append({"constraint_type": "material_string", "value": formula, "source_chunk_ids": [], "status": "proposed", "confidence": 0.35, "warning": "Formula-like string was detected from the title only; this is not chemistry parsing."})

    return {
        "constraint_id": f"paper-constraints-{_hash(paper_id)}",
        "paper_id": paper_id,
        "title": title,
        "instrument": sorted(instruments),
        "detector_modes": sorted(detector_modes),
        "measurement_kinds": sorted(measurement_kinds),
        "data_reduction_terms": sorted(reduction_terms),
        "sample_environment_terms": sorted(sample_environment),
        "formula_like_title_strings": formula_candidates,
        "ranges": ranges,
        "chunk_type_counts": _counts(chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type") for chunk in chunks),
        "chunk_ids_by_type": chunk_ids_by_type,
        "constraint_evidence": constraint_evidence,
        "status": "proposed",
        "human_review_required": True,
        "warnings": [
            "No snippets are stored in this constraint record.",
            "Formula-like strings are not material identity assignments.",
            "Publication text is untrusted evidence until source-support validation or human review.",
        ],
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="extract_paper_constraints").model_dump(mode="json")],
    }


_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
}


def _formula_like_title_candidates(title: str) -> list[str]:
    candidates = set(re.findall(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?){2,}\b", title or ""))
    filtered = []
    for candidate in candidates:
        symbols = re.findall(r"([A-Z][a-z]?)(?:\d*(?:\.\d+)?)", candidate)
        if len(symbols) < 2:
            continue
        if all(symbol in _ELEMENT_SYMBOLS for symbol in symbols):
            filtered.append(candidate)
    return sorted(filtered)


def _matched_terms(text_by_chunk: dict[str, str], patterns: dict[str, list[str]]) -> dict[str, set[str]]:
    matches: dict[str, set[str]] = {}
    for chunk_id, text in text_by_chunk.items():
        for label, regexes in patterns.items():
            if any(re.search(regex, text, flags=re.IGNORECASE) for regex in regexes):
                matches.setdefault(label, set()).add(chunk_id)
    return matches


def _extractor_text_path(run_dir: Path, row: dict[str, Any], backend: str) -> tuple[Path | None, str | None, str | None]:
    if not row:
        return None, None, None
    if backend == "docling":
        raw_path = row.get("markdown_path")
        candidates = [Path(str(raw_path)).expanduser()] if raw_path else []
        if raw_path:
            candidates.append(run_dir / "outputs" / "docling" / "md" / Path(str(raw_path)).name)
        pdf_name = str(row.get("pdf_name") or "")
        if pdf_name:
            candidates.append(run_dir / "outputs" / "docling" / "md" / f"{Path(pdf_name).stem}.md")
        for candidate in candidates:
            if candidate.exists():
                return candidate, "docling_markdown_import", "docling_vista_batch"
    if backend == "grobid":
        raw_path = row.get("tei_path")
        candidates = [Path(str(raw_path)).expanduser()] if raw_path else []
        if raw_path:
            candidates.append(run_dir / "outputs" / "grobid" / "tei" / Path(str(raw_path)).name)
        pdf_name = str(row.get("pdf_name") or "")
        if pdf_name:
            candidates.append(run_dir / "outputs" / "grobid" / "tei" / f"{Path(pdf_name).stem}.tei.xml")
        for candidate in candidates:
            if candidate.exists():
                return candidate, "grobid_tei_import", "grobid_0.9.0_crf"
    return None, None, None


def _read_extractor_text(path: Path, backend: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if backend == "grobid" or path.suffix.lower() == ".xml":
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
    return text


def _title_from_pdf_name(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").strip() or pdf_name


def _extract_pdf_text(path: Path, *, max_pages: int, max_chars: int) -> tuple[str, str | None]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", "pdftotext unavailable; recorded PDF metadata only"
    try:
        result = subprocess.run(
            [pdftotext, "-enc", "UTF-8", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover - platform dependent subprocess failure
        return "", f"pdftotext failed: {exc}"
    if result.returncode != 0:
        return "", f"pdftotext returned {result.returncode}: {result.stderr[:200]}"
    return result.stdout[:max_chars], None


def _chunks_from_text(document: Document, text: str, *, method: str) -> list[dict[str, Any]]:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    source_hash = document.source_hash
    chunks: list[dict[str, Any]] = []
    base_provenance = [ProvenanceRecord(source_type="document", source_id=document.document_id, method=method).model_dump(mode="json")]
    title_text = document.title or cleaned[:200].strip()
    if title_text:
        chunks.append(_chunk_record(document, "title", title_text[:1000], 0, min(len(title_text), 1000), method, source_hash, base_provenance))
    abstract = _section_snippet(cleaned, ["abstract"], ["introduction", "keywords"], window=2500)
    if abstract:
        chunks.append(_chunk_record(document, "abstract", abstract, cleaned.lower().find(abstract[:40].lower()), None, method, source_hash, base_provenance))
    for chunk_type, terms in [
        ("instrument_setup", ["bt-7", "bt7", "triple-axis", "triple axis", "neutron scattering"]),
        ("experimental", ["experimental", "experiment", "instrument", "measurements"]),
        ("data_reduction", ["data reduction", "background", "resolution", "calibration"]),
    ]:
        snippet, start = _term_snippet(cleaned, terms, window=3000)
        if snippet and not any(item["chunk_type"] == chunk_type and item["text"] == snippet for item in chunks):
            chunks.append(_chunk_record(document, chunk_type, snippet, start, start + len(snippet), method, source_hash, base_provenance))
    for snippet, start, figure_label in _figure_caption_snippets(cleaned):
        record = _chunk_record(document, "figure_caption", snippet, start, start + len(snippet), method, source_hash, base_provenance)
        record.setdefault("metadata", {})["figure_label_guess"] = figure_label
        if not any(item["chunk_type"] == "figure_caption" and item["text"] == record["text"] for item in chunks):
            chunks.append(record)
    material = _material_composition_snippet(cleaned)
    if material:
        snippet, start, candidates = material
        record = _chunk_record(document, "material_composition", snippet, start, start + len(snippet), method, source_hash, base_provenance)
        record.setdefault("metadata", {}).update(
            {
                "formula_candidates": candidates,
                "derivation": "regex_material_structure_v0",
                "derivation_warning": "Formula candidates are regex guesses over untrusted extracted text; ambiguous shorthands stay ambiguous until reviewed.",
            }
        )
        chunks.append(record)
    structure_snippet, structure_start = _term_snippet(cleaned, _STRUCTURE_TERMS, window=2200)
    if structure_snippet:
        record = _chunk_record(document, "structure_parameters", structure_snippet, structure_start, structure_start + len(structure_snippet), method, source_hash, base_provenance)
        structure_metadata = record.setdefault("metadata", {})
        structure_metadata["structure_terms_found"] = [term for term in _STRUCTURE_TERMS if term in structure_snippet.lower()]
        structure_metadata["derivation"] = "regex_material_structure_v0"
        space_group = _SPACE_GROUP_RE.search(structure_snippet)
        if space_group:
            structure_metadata["space_group_guess"] = space_group.group(1)
        chunks.append(record)
    if len(chunks) == 1 and cleaned:
        chunks.append(_chunk_record(document, "unknown", cleaned[:2500], 0, min(len(cleaned), 2500), method, source_hash, base_provenance))
    return chunks


# Chemical-element symbols used to validate formula-like tokens.
_ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_FORMULA_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?|[xy])?){2,8}\b")
_FORMULA_PART_RE = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?|[xy])?")
_STRUCTURE_TERMS = [
    "space group",
    "lattice parameter",
    "lattice constant",
    "crystal structure",
    "unit cell",
    "rietveld",
    "hexagonal",
    "orthorhombic",
    "rhombohedral",
    "tetragonal",
    "monoclinic",
    "triclinic",
]
_SPACE_GROUP_RE = re.compile(r"space\s+group\s*(?:of|is|:)?\s*([A-Za-z0-9\-/_]{1,12})", re.IGNORECASE)
# Tokens that parse as valid element sequences but are well-known
# science-infrastructure acronyms appearing across many papers.
_FORMULA_DENYLIST = frozenset({"IN2P3"})


def _formula_candidates(text: str, *, max_candidates: int = 5) -> list[dict[str, Any]]:
    """Rank formula-like tokens whose parts are all valid element symbols.

    Tokens of fewer than three distinct elements must carry an explicit
    stoichiometric amount, which rejects ordinary capitalized words and most
    acronyms while keeping compositions such as LuFe0.5Mn0.5O3 or YBaCuO.
    """
    counts: dict[str, int] = {}
    first_position: dict[str, int] = {}
    for match in _FORMULA_TOKEN_RE.finditer(text):
        token = match.group(0)
        parts = _FORMULA_PART_RE.findall(token)
        if "".join(symbol + (amount or "") for symbol, amount in parts) != token:
            continue
        symbols = [symbol for symbol, _ in parts]
        if len(symbols) < 2 or any(symbol not in _ELEMENT_SYMBOLS for symbol in symbols):
            continue
        amounts = [amount for _, amount in parts if amount]
        has_amount = bool(amounts)
        # Two distinct elements minimum rejects Miller-index notation (HH0);
        # without amounts, three are needed to reject two-letter words.
        if len(set(symbols)) < (2 if has_amount else 3):
            continue
        # A stoichiometric coefficient of exactly zero is reciprocal-space
        # notation (HK0), and coefficients beyond 4 digits are grant or
        # report numbers (SC0002613), not compositions.
        if any(amount == "0" or len(amount) > 4 for amount in amounts):
            continue
        if token in _FORMULA_DENYLIST:
            continue
        # All-caps tokens without stoichiometric amounts are acronyms
        # (PHONON, INFN, INS), not compositions: real digit-free formulas
        # such as YBaCuO carry lowercase letters in two-letter symbols.
        if not has_amount and token.isupper():
            continue
        counts[token] = counts.get(token, 0) + 1
        first_position.setdefault(token, match.start())
    # The studied material almost always appears in the title/abstract region,
    # while institute/funding acronyms (IN2P3) cluster in acknowledgments, so
    # early first occurrence outweighs raw repetition.
    def _score(token: str) -> float:
        return counts[token] * (3.0 if first_position[token] < 3000 else 1.0)

    ranked = sorted(counts, key=lambda token: (-_score(token), first_position[token]))
    # Canonical parsing goes through the shared material-identity boundary
    # (pymatgen preferred, ASE fallback); the regex above only finds and
    # de-noises candidate tokens. The boundary accepts any chemically valid
    # string, so acronym rejection stays in the heuristics above.
    from tas_copilot_core.material_identity import parse_formula

    candidates = []
    for token in ranked[:max_candidates]:
        parsed = parse_formula(token)
        candidates.append(
            {
                "formula": token,
                "occurrences": counts[token],
                "first_char": first_position[token],
                "normalized_formula": parsed.normalized_formula,
                "reduced_formula": parsed.reduced_formula,
                "parser_backend": parsed.parser_backend,
                "parse_confidence": parsed.confidence,
                "parse_warnings": list(parsed.warnings),
            }
        )
    return candidates


def _material_composition_snippet(text: str, *, window: int = 1200) -> tuple[str, int, list[dict[str, Any]]] | None:
    candidates = _formula_candidates(text)
    if not candidates:
        return None
    anchor = candidates[0]["first_char"]
    start = max(0, anchor - window // 4)
    snippet = text[start : start + window].strip()
    if not snippet:
        return None
    return snippet, start, candidates


def _figure_caption_snippets(text: str, *, window: int = 1800, max_count: int = 4) -> list[tuple[str, int, str]]:
    snippets: list[tuple[str, int, str]] = []
    seen_starts: set[int] = set()
    pattern = re.compile(r"(?im)(?:^|\n)\s*((?:fig(?:ure)?\.?)\s*\d+[A-Za-z]?)\b")
    for match in pattern.finditer(text):
        start = max(0, match.start())
        if any(abs(start - existing) < 80 for existing in seen_starts):
            continue
        seen_starts.add(start)
        next_match = pattern.search(text, match.end())
        end = min(len(text), next_match.start() if next_match else start + window)
        snippet = text[start:end].strip()
        if snippet:
            snippets.append((snippet[:window], start, match.group(1).strip()))
        if len(snippets) >= max_count:
            break
    return snippets


def _chunk_record(
    document: Document,
    chunk_type: str,
    text: str,
    char_start: int | None,
    char_end: int | None,
    method: str,
    source_hash: str | None,
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_start = max(0, char_start or 0)
    normalized_end = char_end if char_end is not None else normalized_start + len(text)
    chunk_id = f"chunk-{_hash(document.document_id + chunk_type + str(normalized_start))}"

    # Scan for injection patterns
    is_suspicious, matched_patterns = scan_for_injection(text)

    return SourceChunk(
        chunk_id=chunk_id,
        source_type="document",
        source_id=document.document_id,
        document_id=document.document_id,
        paper_id=document.paper_id,
        chunk_type=chunk_type,
        title=f"{document.title or document.document_id} [{chunk_type}]",
        text=text.strip(),
        char_start=normalized_start,
        char_end=normalized_end,
        token_count=len(text.split()),
        source_hash=source_hash,
        chunk_hash=_hash(text),
        extraction_method=method,
        parser_version="corpus_pilot_pdftotext_v0",
        metadata={
            "private_local_only": document.copyright_status == "private_local_only",
            "document_year": document.metadata.get("year"),
            "doi": document.metadata.get("doi"),
            "ncnr_timeline_warning": "Publication year is not data-acquisition year evidence for post-2021 NCNR papers." if _document_year(document.model_dump(mode="json")) and _document_year(document.model_dump(mode="json")) > 2021 else None,
            "suspected_injection": is_suspicious,
            "injection_patterns": matched_patterns if is_suspicious else [],
        },
        provenance=[ProvenanceRecord(**item) for item in provenance],
    ).model_dump(mode="json")


def _document_year(document: dict[str, Any]) -> int | None:
    raw = document.get("year") or (document.get("metadata") or {}).get("year")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _section_snippet(text: str, starts: list[str], stops: list[str], *, window: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in starts if lowered.find(term) >= 0]
    if not positions:
        return ""
    start = min(positions)
    stop_candidates = [lowered.find(term, start + 20) for term in stops if lowered.find(term, start + 20) > start]
    end = min(stop_candidates) if stop_candidates else start + window
    return text[start : min(len(text), end, start + window)].strip()


def _term_snippet(text: str, terms: list[str], *, window: int) -> tuple[str, int]:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if lowered.find(term.lower()) >= 0]
    if not positions:
        return "", -1
    center = min(positions)
    start = max(0, center - window // 4)
    end = min(len(text), start + window)
    return text[start:end].strip(), start


def _source_chunks(catalog_dir: Path) -> list[dict[str, Any]]:
    chunks = _load_jsonl_dicts(catalog_dir / "source_chunks.jsonl")
    for section in _load_jsonl_dicts(catalog_dir / "textbook_sections.jsonl"):
        chunks.append(
            {
                "chunk_id": section.get("section_id"),
                "source_type": "textbook",
                "source_id": section.get("section_id"),
                "chunk_type": "textbook_explanation",
                "title": section.get("title"),
                "text": section.get("text", ""),
                "metadata": {"book_title": section.get("book_title"), "section_path": section.get("section_path")},
                "provenance": section.get("provenance", []),
            }
        )
    return chunks


def _passes_filters(
    chunk: dict[str, Any],
    args: dict[str, Any],
    document_types: dict[str, str] | None = None,
    concepts_by_chunk: dict[str, set[str]] | None = None,
    supported_sources: set[str] | None = None,
) -> bool:
    for key in ["paper_id", "figure_id", "panel_id", "chunk_type"]:
        value = args.get(key)
        if value and str(chunk.get(key) or chunk.get("metadata", {}).get(key)) != str(value):
            return False
    metadata = chunk.get("metadata", {}) or {}
    document_type = args.get("document_type")
    if document_type:
        document_id = str(chunk.get("document_id") or chunk.get("paper_id") or chunk.get("source_id") or "")
        observed_type = str(metadata.get("document_type") or (document_types or {}).get(document_id) or "")
        if observed_type != str(document_type):
            return False
    tas_concept = args.get("tas_concept")
    if tas_concept:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        if str(tas_concept) not in (concepts_by_chunk or {}).get(chunk_id, set()):
            return False
    if args.get("requires_source_support"):
        ids = {str(chunk.get("chunk_id") or ""), str(chunk.get("figure_id") or ""), str(chunk.get("panel_id") or "")}
        if not ids.intersection(supported_sources or set()):
            return False
    year = args.get("year")
    if year and str(chunk.get("year") or metadata.get("year") or metadata.get("document_year") or "") != str(year):
        return False
    for key in ["instrument", "material_family"]:
        value = args.get(key)
        haystack = " ".join([str(chunk.get(key) or ""), str(metadata.get(key) or ""), str(chunk.get("text") or "")]).lower()
        if value and str(value).lower() not in haystack:
            return False
    if args.get("exclude_references") and (chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type")) == "references":
        return False
    return True


def _passes_graph_filters(hit: RetrievalResult, args: dict[str, Any]) -> bool:
    haystack = json.dumps(hit.model_dump(mode="json"), sort_keys=True).lower()
    for key in ["paper_id", "figure_id", "panel_id", "instrument", "material_family"]:
        value = args.get(key)
        if value and str(value).lower() not in haystack:
            return False
    for key in ["document_type", "chunk_type", "tas_concept", "year"]:
        if args.get(key):
            return False
    if args.get("requires_source_support") and (hit.structured_summary or {}).get("status") != "supported":
        return False
    if args.get("exclude_references") and "references" in haystack:
        return False
    return True

_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "of", "to", "in", "on", "at", "by", "for", "with", "as",
    "its", "it", "this", "that", "these", "those", "what", "which", "who",
    "how", "why", "when", "where", "general", "properties", "property",
})

def _terms(text: str) -> set[str]:
    strip_chars = ".,;:()[]{}?!\"'"
    raw = {
        term.strip(strip_chars).lower()
        for term in text.replace("-", " ").replace("/", " ").split()
        if len(term.strip(strip_chars)) >= 2
    }
    return {term for term in raw if term not in _QUERY_STOPWORDS}


def _broad_query_warnings(query: str) -> list[dict[str, Any]]:
    lowered = query.lower()
    return [{"code": "broad_query_term", "term": term, "message": "Broad terms should not dominate retrieval; use as filters or low-weight evidence."} for term in BROAD_CONCEPT_LABELS if term in lowered]


def _concept_retrieval_weight(concept) -> float:
    if concept is None:
        return 0.0
    if getattr(concept, "avoid_if_matches_too_broadly", False):
        return 0.0
    if getattr(concept, "retrieval_policy", "") in {"metadata_filter_only", "schema_alignment_only", "validation_only", "provenance_alignment_only"}:
        return 0.0
    if getattr(concept, "retrieval_policy", "") == "low_weight_synonym_expansion":
        return 0.1
    return 0.25


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _dedupe_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        value = record.get(key)
        if value:
            deduped[str(value)] = record
        else:
            passthrough.append(record)
    return passthrough + list(deduped.values())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _missing(workflow: str, output_dir: Path, fields: list[str], warning: str) -> dict[str, Any]:
    payload = {"workflow_name": workflow, "status": "missing_input", "missing_fields": fields, "warnings": [warning]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{workflow.replace('-', '_')}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _ingest_markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Corpus Pilot Ingest", "", f"Status: `{report['status']}`", f"Corpus: `{report['corpus_id']}`", f"Documents: {report['documents']}", f"Graph: {report['graph']['node_count']} nodes / {report['graph']['edge_count']} edges", "", "## Warnings", *[f"- {item}" for item in report.get("warnings", []) or ["None"]]]) + "\n"


def _extractor_ingest_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Extractor Run Corpus Ingest",
            "",
            f"Status: `{report['status']}`",
            f"Extractor run: `{report['extractor_run_dir']}`",
            f"Preferred backend: `{report['preferred_backend']}`",
            f"Documents: {report['documents']}",
            f"Source chunks: {report['source_chunks']}",
            f"Graph: {report['graph']['node_count']} nodes / {report['graph']['edge_count']} edges",
            f"TAS concept links: {report['ontology_links']['link_count']}",
            "",
            "## Warnings",
            *[f"- {item}" for item in report.get("warnings", []) or ["None"]],
        ]
    ) + "\n"


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Corpus Pilot Report", "", f"Documents: {report.get('documents', 0)}", f"Source chunks: {report.get('source_chunks', 0)}", f"Figures: {report.get('figures', 0)}", f"Panels: {report.get('figure_panels', 0)}", f"Graph edges: {report.get('evidence_graph_edges', 0)}", f"TAS concept links: {report.get('tas_concept_links', 0)}", "", "## Limitations", *[f"- {item}" for item in report.get("known_limitations", [])]]) + "\n"


def _shortlist_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Candidate Shortlist",
        "",
        f"Status: `{report.get('status')}`",
        f"Documents: {report.get('document_count', 0)}",
        f"Top candidates: {len(report.get('top_candidates', []))}",
        "",
        "## Ranked Candidates",
    ]
    for index, candidate in enumerate(report.get("top_candidates", []), start=1):
        reasons = "; ".join(candidate.get("manual_curation_reasons", [])) or "metadata only"
        lines.extend(
            [
                f"{index}. `{candidate.get('paper_id')}` score={candidate.get('total_score')}: {candidate.get('title')}",
                f"   - PDF: `{candidate.get('pdf_name')}`",
                f"   - Chunk types: {candidate.get('chunk_type_counts')}",
                f"   - Reasons: {reasons}",
                f"   - Missing: {', '.join(candidate.get('missing_for_curation', [])) or 'none'}",
            ]
        )
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    lines.extend(["", report.get("safe_conclusion", "")])
    return "\n".join(lines) + "\n"


def _constraints_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Constraint Extraction",
        "",
        f"Status: `{report.get('status')}`",
        f"Papers: {report.get('paper_count', 0)}",
        "",
        "## Constraints",
    ]
    for record in report.get("constraints", []):
        lines.extend(
            [
                f"### `{record.get('paper_id')}`",
                str(record.get("title") or ""),
                f"- Instrument: {', '.join(record.get('instrument', [])) or 'none'}",
                f"- Detector modes: {', '.join(record.get('detector_modes', [])) or 'none'}",
                f"- Measurement kinds: {', '.join(record.get('measurement_kinds', [])) or 'none'}",
                f"- Data reduction terms: {', '.join(record.get('data_reduction_terms', [])) or 'none'}",
                f"- Sample environment terms: {', '.join(record.get('sample_environment_terms', [])) or 'none'}",
                f"- Formula-like title strings: {', '.join(record.get('formula_like_title_strings', [])) or 'none'}",
                f"- Evidence items: {len(record.get('constraint_evidence', []))}",
                "",
            ]
        )
    lines.extend(["## Warnings"])
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    lines.extend(["", report.get("safe_conclusion", "")])
    return "\n".join(lines) + "\n"


def _diagnostics_markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Corpus Ontology Diagnostics", "", f"Chunks: {report['chunk_count']}", f"Concept links: {report['concept_link_count']}", "", "## Warnings", *[f"- `{item.get('code', 'warning')}` {item}" for item in report.get("warnings", []) or []]]) + "\n"


def _paper_brief_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Brief",
        "",
        f"**Title:** {report.get('paper_title') or report['paper'].get('paper_id')}",
        f"**Paper ID:** `{report['paper']['paper_id']}`",
        f"**Year:** {report.get('paper_year') or 'unknown'}",
        f"**Chunks:** {report['chunk_count']}  |  Figures: {len(report['figures'])}  |  Panels: {len(report['panels'])}",
        "",
        "## Content Excerpts",
    ]
    previews = report.get("chunk_previews") or []
    if previews:
        for i, p in enumerate(previews, 1):
            lines.append(f"{i}. **[{p['chunk_type']}]** {p['text']}")
    else:
        lines.append("- No content excerpts available.")
    lines.extend([
        "",
        "## Safe Next Actions",
        *[f"- {item}" for item in report["safe_next_actions"]],
    ])
    return "\n".join(lines) + "\n"

def _figure_brief_markdown(report: dict[str, Any]) -> str:
    return "\n".join(["# Figure Brief", "", f"Figure: `{report['figure']['figure_id']}`", f"Panels: {len(report['panels'])}", f"Candidate links: {len(report['candidate_source_links'])}", "", "## Safe Next Actions", *[f"- {item}" for item in report["safe_next_actions"]]]) + "\n"


def _paper_experiment_markdown(report: dict[str, Any]) -> str:
    prior = report.get("publication_data_lag_prior", {})
    return "\n".join(
        [
            "# Paper To Experiment Links",
            "",
            f"Paper: `{report['paper_id']}`",
            f"Candidate links: {len(report['candidate_links'])}",
            f"Extracted constraint records: {len(report.get('extracted_constraints', []))}",
            "",
            "## Publication/Data Lag Prior",
            f"Publication year: {prior.get('publication_year')}",
            f"Typical lag: {prior.get('typical_lag_years')} years",
            f"Plausible year window: {prior.get('plausible_start_year')} to {prior.get('plausible_end_year')}",
            f"NCNR shutdown cap applied: {prior.get('ncnr_cap_applied')}",
            "",
            report["safe_conclusion"],
            "",
            "## Missing Inputs",
            *[f"- {item}" for item in report.get("missing_inputs", []) or ["none"]],
            "",
            "## Next Commands",
            *[f"- `{item}`" for item in report.get("next_commands", []) or ["none"]],
        ]
    ) + "\n"


def _review_dashboard_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Paper-Experiment Review Dashboard",
            "",
            f"Status: `{report.get('status')}`",
            f"Rows: `{report.get('review_row_count')}`",
            f"HTML: `{report.get('html')}`",
            f"Priority counts: `{report.get('priority_counts')}`",
            f"Status counts: `{report.get('status_counts')}`",
            f"Review rows CSV: `{report.get('review_rows_csv')}`",
            f"Review summary JSON: `{report.get('review_summary_json')}`",
            "",
            report.get("safe_conclusion", ""),
            "",
            "## Next Actions",
            *[f"- {item}" for item in report.get("next_actions", [])],
        ]
    ) + "\n"


def _review_label_ingest_markdown(report: dict[str, Any]) -> str:
    readiness = report.get("calibration_readiness") or {}
    return "\n".join(
        [
            "# Paper-Experiment Review Label Ingest",
            "",
            f"Status: `{report.get('status')}`",
            f"Valid rows: `{report.get('valid_row_count')}`",
            f"Invalid rows: `{report.get('invalid_row_count')}`",
            f"Reviewed labels: `{report.get('reviewed_label_count')}`",
            f"Status counts: `{report.get('status_counts')}`",
            "",
            "## Calibration Readiness",
            f"- Status: `{readiness.get('status')}`",
            f"- Accepted: `{readiness.get('accepted_count')}`",
            f"- Rejected: `{readiness.get('rejected_count')}`",
            f"- Can calibrate: `{readiness.get('can_calibrate')}`",
            "",
            report.get("safe_conclusion", ""),
        ]
    ) + "\n"


def _paper_experiment_review_packet_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload).replace("</script>", "<\\/script>")
    title = f"{payload.get('paper_id')} -> {payload.get('candidate_experiment_id')}"
    warning_rows = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in payload.get("parse_warnings", []) + payload.get("warnings", [])) or "<li>none</li>"
    paper_link = f'<a href="{html.escape(str(payload.get("paper_open_url")))}" target="_blank">Open paper from local cache/Vista</a>' if payload.get("paper_open_url") else ""
    overflow_count = payload.get("scan_records_overflow_count") or 0
    total_scans = len(payload.get("scan_records", [])) + overflow_count
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Link Review Packet</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#18212f; background:#f6f8fb; }}
header {{ padding:16px 26px; background:#fff; border-bottom:1px solid #d8dee9; }}
main {{ display:grid; grid-template-columns:340px minmax(0,1fr); gap:10px; padding:10px; height:calc(100vh - 100px); }}
section, aside, .panel {{ background:#fff; border:1px solid #d8dee9; border-radius:8px; padding:12px; overflow-y:auto; }}
h1 {{ margin:0 0 4px; font-size:20px; }}
h2 {{ margin:12px 0 6px; font-size:15px; }}
h3 {{ margin:10px 0 4px; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th, td {{ border-bottom:1px solid #d8dee9; padding:4px 6px; text-align:left; vertical-align:top; }}
th {{ background:#f0f3f8; position:sticky; top:0; z-index:2; }}
tbody tr {{ cursor:pointer; }}
tbody tr:hover, tbody tr.selected {{ background:#eff6ff; }}
code {{ font-size:11px; }}
.muted {{ color:#596579; font-size:11px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:5px; }}
.item {{ border:1px solid #d8dee9; background:#fbfcfe; border-radius:6px; padding:5px; font-size:12px; }}
.item b {{ display:block; font-size:11px; color:#596579; }}
.group-header {{ background:#f0f3f8; font-weight:600; cursor:pointer; user-select:none; }}
.group-header td {{ padding:6px; }}
.badge {{ display:inline-block; border:1px solid #d8dee9; border-radius:999px; padding:1px 6px; font-size:11px; }}
.badge.op {{ border-color:#bbf7d0; background:#f0fdf4; color:#166534; }}
.badge.narrow {{ border-color:#fecaca; background:#fff5f5; color:#991b1b; }}
.badge.info {{ border-color:#bfdbfe; background:#eff6ff; color:#1e40af; }}
.right-col {{ display:flex; flex-direction:column; gap:10px; }}
button {{ border:1px solid #174ea6; background:#174ea6; color:#fff; border-radius:6px; padding:5px 10px; cursor:pointer; font:inherit; font-size:12px; }}
button.secondary {{ background:#fff; color:#174ea6; }}
@media (max-width:980px) {{ main {{ grid-template-columns:1fr; height:auto; }} }}
</style>
</head>
<body>
<header>
<h1>{html.escape(str(title))}</h1>
<div class="grid">
  <div class="item"><b>Paper</b>{html.escape(str(payload.get("paper_id") or ""))}</div>
  <div class="item"><b>Candidate</b>{html.escape(str(payload.get("candidate_experiment_id") or ""))}</div>
  <div class="item"><b>Rank</b>{html.escape(str(payload.get("candidate_rank") or ""))}</div>
  <div class="item"><b>Score</b>{html.escape(str(payload.get("probability_score") or ""))}</div>
  <div class="item"><b>Scans</b>{total_scans} parsed</div>
  <div class="item"><b>Files</b>{len(payload.get("candidate_datafile_paths", []))}</div>
</div>
<p style="margin:6px 0 0">{paper_link}</p>
</header>
<main>
<aside>
  <h2>Scans</h2>
  <div id="scan-table-container" style="overflow-x:auto;">
    <table id="scan-table"><thead><tr><th style="width:28px;"><input type="checkbox" id="select-all-scans" title="Select all"></th><th>Scan</th><th>Variable</th><th>Kind</th><th>Rows</th></tr></thead><tbody id="scan-tbody"></tbody></table>
  </div>
  <div id="load-more-container" style="margin-top:8px;display:none;">
    <button id="load-all-btn" class="secondary">Load all {total_scans} scans</button>
    <span id="load-status" class="muted"></span>
  </div>
  <h2>Experiment Summary</h2>
  <div id="experiment-summary"></div>
  <h2>Warnings</h2>
  <ul style="font-size:12px;padding-left:18px;">{warning_rows}</ul>
</aside>
<div class="right-col">
<section>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:6px;">
    <label style="font-size:12px;">Scan <select id="scan-select" style="font-size:12px;"></select></label>
    <label style="font-size:12px;">X <select id="x-axis-select" style="min-width:120px;font-size:12px;"></select></label>
    <label style="font-size:12px;">Y <select id="y-axis-select" style="min-width:120px;font-size:12px;"></select></label>
    <button id="overlay-btn" class="secondary" style="font-size:12px;">Overlay selected</button>
  </div>
  <div id="plot" style="height:380px;"></div>
  <div id="scan-detail" class="muted"></div>
</section>
<div class="panel">
  <h2 style="margin-top:0;">Review Evidence</h2>
  <div id="metadata"></div>
</div>
</div>
</main>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
let scans = Array.isArray(payload.scan_records) ? [...payload.scan_records] : [];
const overflowCount = payload.scan_records_overflow_count || 0;
const sidecarUrl = payload.scans_sidecar_url || '';
function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function compact(v) {{
  if (Array.isArray(v)) return v.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(', ') || 'none';
  if (v && typeof v === 'object') return JSON.stringify(v);
  return String(v ?? 'none');
}}
function metadataGrid(obj) {{
  if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) return '<p class="muted">No structured metadata available.</p>';
  return `<div class="grid">${{Object.entries(obj).map(([k,v]) => `<div class="item"><b>${{esc(k)}}</b>${{esc(compact(v))}}</div>`).join('')}}</div>`;
}}

const X_PRIORITY = ['H','K','L','E','A4','A3','SmplUTilt','SmplLTilt'];
const ENV_COLS = new Set(['Temp','TemperatureSetpoint']);
const DETECTOR_GROUPS = [
  {{label:'Single Detector', test: c => ['Detector','SingleDet','Detector1'].includes(c)}},
  {{label:'PSD',             test: c => c === 'PSDet' || /^PSDC/.test(c)}},
  {{label:'Diffraction',     test: c => c === 'DiffDet'}},
  {{label:'Monitor',         test: c => c === 'Monitor'}},
];

function colVaries(col, d) {{ const v=(d[col]||[]).filter(x=>x!=null); return v.length>1 && Math.max(...v)-Math.min(...v)>1e-9; }}
function xAxisOrder(cols, sv, d) {{ return [...cols].sort((a,b)=>{{ const r=c=>{{ if(c===sv) return -1; const p=X_PRIORITY.indexOf(c); if(p>=0) return p; if(ENV_COLS.has(c)&&colVaries(c,d)) return 100; return colVaries(c,d)?101:200; }}; return r(a)-r(b); }}); }}
function buildYOptions(cols, def) {{
  const used=new Set(); let h='';
  for(const g of DETECTOR_GROUPS) {{ const gc=cols.filter(c=>g.test(c)); if(!gc.length) continue; h+=`<optgroup label="${{esc(g.label)}}">`; gc.forEach(c=>{{ h+=`<option value="${{esc(c)}}"${{c===def?' selected':''}}>${{esc(c)}}</option>`; used.add(c); }}); h+='</optgroup>'; }}
  const oth=cols.filter(c=>!used.has(c)); if(oth.length) {{ h+='<optgroup label="Other">'; oth.forEach(c=>{{ h+=`<option value="${{esc(c)}}"${{c===def?' selected':''}}>${{esc(c)}}</option>`; }}); h+='</optgroup>'; }}
  return h;
}}

function groupScans(scanList) {{
  const groups = {{}};
  for (const s of scanList) {{
    const key = (s.scan_variable||'unknown') + '|' + (s.measurement_kind||'unknown');
    if (!groups[key]) groups[key] = [];
    groups[key].push(s);
  }}
  return groups;
}}

function renderScanTable() {{
  const tbody = document.getElementById('scan-tbody');
  const groups = groupScans(scans);
  const sortedKeys = Object.keys(groups).sort();
  let html = '';
  for (const key of sortedKeys) {{
    const [variable, kind] = key.split('|');
    const items = groups[key];
    html += `<tr class="group-header" data-group="${{esc(key)}}"><td><input type="checkbox" class="group-check" data-group="${{esc(key)}}" title="Select group"></td><td colspan="4">${{esc(variable)}} / ${{esc(kind)}} (${{items.length}})</td></tr>`;
    for (const s of items) {{
      html += `<tr data-scan-id="${{esc(s.scan_id)}}" data-group="${{esc(key)}}"><td><input type="checkbox" class="scan-check" data-scan-id="${{esc(s.scan_id)}}"></td><td>${{esc(s.scan_id)}}</td><td>${{esc(s.scan_variable)}}</td><td>${{esc(s.measurement_kind)}}</td><td>${{s.row_count||0}}</td></tr>`;
    }}
  }}
  tbody.innerHTML = html;
  tbody.querySelectorAll('tr[data-scan-id]').forEach(row => {{
    const scanId = row.dataset.scanId;
    row.querySelectorAll('td:not(:first-child)').forEach(td => td.addEventListener('click', () => draw(scanId)));
  }});
  tbody.querySelectorAll('.group-header').forEach(hdr => {{
    hdr.querySelectorAll('td:not(:first-child)').forEach(td => td.addEventListener('click', () => {{
      const g = hdr.dataset.group;
      const rows = tbody.querySelectorAll(`tr[data-group="${{g}}"]:not(.group-header)`);
      const hidden = rows[0] && rows[0].style.display === 'none';
      rows.forEach(r => r.style.display = hidden ? '' : 'none');
    }}));
  }});
  tbody.querySelectorAll('.group-check').forEach(cb => cb.addEventListener('change', (e) => {{
    const g = e.target.dataset.group;
    tbody.querySelectorAll(`.scan-check[data-scan-id]`).forEach(sc => {{
      if (sc.closest('tr').dataset.group === g) sc.checked = e.target.checked;
    }});
  }}));
}}

function overlaySelected() {{
  const checked = [...document.querySelectorAll('.scan-check:checked')];
  const selectedScans = checked.map(cb => scans.find(s => s.scan_id === cb.dataset.scanId)).filter(Boolean);
  if (!selectedScans.length) return;
  const xCol = document.getElementById('x-axis-select').value;
  const yCol = document.getElementById('y-axis-select').value;
  const traces = selectedScans.map(s => {{
    const acd = (s.plot_data || {{}}).all_columns_data || {{}};
    const xData = acd[xCol] || (s.plot_data || {{}}).x || [];
    const yData = acd[yCol] || (s.plot_data || {{}}).y || [];
    return {{x: xData, y: yData, mode: 'lines+markers', type: 'scatter', name: s.scan_id}};
  }});
  Plotly.newPlot('plot', traces, {{
    title: `Overlay: ${{selectedScans.length}} scans`,
    xaxis: {{title: xCol || 'scan axis'}}, yaxis: {{title: yCol || 'intensity'}},
    margin: {{l:60, r:20, t:50, b:50}}
  }}, {{responsive: true}});
}}

let currentRecord = null;
function populateAxisSelects(record) {{
  const plot = record.plot_data || {{}};
  const acd = (plot.all_columns_data && Object.keys(plot.all_columns_data).length) ? plot.all_columns_data : null;
  const cols = acd ? Object.keys(acd) : (record.column_names || []);
  const xSel = document.getElementById('x-axis-select');
  const ySel = document.getElementById('y-axis-select');
  const px = xSel.value, py = ySel.value;
  const dx = plot.x_column || record.scan_variable || (cols[0] || '');
  const dy = plot.y_column || '';
  xSel.innerHTML = xAxisOrder(cols, record.scan_variable, acd||{{}}).map(c=>`<option value="${{esc(c)}}"${{c===(px||dx)?' selected':''}}>${{esc(c)}}</option>`).join('');
  ySel.innerHTML = buildYOptions(cols, py||dy);
  xSel.disabled = !acd; ySel.disabled = !acd;
}}
function redrawPlot(record) {{
  const plot = record.plot_data || {{}};
  const acd = (plot.all_columns_data && Object.keys(plot.all_columns_data).length) ? plot.all_columns_data : null;
  const xCol = document.getElementById('x-axis-select').value || plot.x_column || '';
  const yCol = document.getElementById('y-axis-select').value || plot.y_column || '';
  const xData = acd ? (acd[xCol]||[]) : (plot.x||[]);
  const yData = acd ? (acd[yCol]||[]) : (plot.y||[]);
  Plotly.newPlot('plot', [{{x:xData, y:yData, mode:'lines+markers', type:'scatter', name:record.scan_id}}],
    {{ title:record.title||record.scan_id, xaxis:{{title:xCol||'scan axis'}}, yaxis:{{title:yCol||'intensity'}}, margin:{{l:60,r:20,t:50,b:50}} }}, {{responsive:true}});
}}
function draw(scanId) {{
  const record = scans.find(s => s.scan_id === scanId) || scans[0];
  if (!record) {{ document.getElementById('plot').innerHTML = '<p class="muted">No scan data.</p>'; return; }}
  currentRecord = record;
  populateAxisSelects(record);
  redrawPlot(record);
  const select = document.getElementById('scan-select');
  if (select) select.value = record.scan_id;
  document.getElementById('scan-detail').innerHTML = `<b>${{esc(record.scan_id)}}</b> &mdash; ${{esc(record.title)}}<br><span class="muted">type=${{esc(record.scan_type)}} variable=${{esc(record.scan_variable)}} kind=${{esc(record.measurement_kind)}} rows=${{esc(record.row_count)}}</span><br><code>${{esc(record.path)}}</code><h3>Columns</h3><p class="muted">${{esc((record.column_names||[]).join(', '))}}</p><h3>Header Metadata</h3>${{metadataGrid(record.metadata||{{}})}}`;
  document.querySelectorAll('#scan-tbody tr[data-scan-id]').forEach(row => row.classList.toggle('selected', row.dataset.scanId === record.scan_id));
}}

function renderExperimentSummary() {{
  const summary = payload.experiment_summary || {{}};
  const review = summary.experiment_review || {{}};
  const el = document.getElementById('experiment-summary');
  let html = '';
  html += `<div class="grid"><div class="item"><b>Scans parsed</b>${{review.scan_count || summary.scan_count || 0}}</div>`;
  if (review.series_count) html += `<div class="item"><b>Series</b>${{review.series_count}}</div>`;
  const types = summary.scan_type_breakdown || {{}};
  for (const [k,v] of Object.entries(types)) html += `<div class="item"><b>${{esc(k)}}</b>${{v}}</div>`;
  const tc = summary.temperature_coverage || {{}};
  if (tc.min != null) html += `<div class="item"><b>Temp range</b>${{Number(tc.min).toFixed(1)}} – ${{Number(tc.max).toFixed(1)}} K</div>`;
  html += '</div>';

  const op = review.order_parameter_candidates || summary.order_parameter_candidates || [];
  if (op.length) {{
    html += `<h3>Order Parameter Candidates (${{op.length}})</h3>`;
    for (let oi = 0; oi < op.length; oi++) {{
      const c = op[oi];
      const isCrossFile = c.scan_variable === 'Temp' || (c.candidate_id || '').includes('cross_file');
      html += `<div style="margin:4px 0;font-size:12px;"><span class="badge op">${{esc(c.classification)}}</span> score=${{c.score}} scans=${{c.scan_count}}`;
      if (isCrossFile) html += ` <span class="badge info">cross-file series</span>`;
      if (c.scan_variable) html += ` axis=${{esc(c.scan_variable)}}`;
      if (c.temperature_range) html += ` T=${{Number(c.temperature_range.min||0).toFixed(1)}}–${{Number(c.temperature_range.max||0).toFixed(1)}}K`;
      html += `<br><span class="muted">${{esc((c.score_reasons||[]).join('; '))}}</span>`;
      const peaks = c.per_scan_peak_summary || [];
      if (peaks.length >= 2) html += `<div id="op-plot-${{oi}}" style="height:200px;margin:6px 0;"></div>`;
      html += '</div>';
    }}
  }}

  const spurions = review.spurion_candidates || [];
  if (spurions.length) {{
    html += `<h3>Spurion Candidates (${{spurions.length}})</h3>`;
    for (const s of spurions.slice(0, 5)) {{
      html += `<div style="font-size:12px;"><span class="badge narrow">${{esc(s.scan_id)}}</span> ${{esc(s.candidate_type)}} <span class="muted">${{esc(s.likelihood_label)}} (${{s.likelihood_score}})</span></div>`;
    }}
    if (spurions.length > 5) html += `<p class="muted">... and ${{spurions.length-5}} more</p>`;
  }}

  const series = review.overnight_series || [];
  if (series.length) {{
    html += `<h3>Scan Series (${{series.length}})</h3>`;
    for (const s of series.slice(0, 8)) {{
      const kinds = (s.measurement_kinds||[]).join('/');
      const vars = (s.scan_variables||[]).join('/');
      html += `<div style="font-size:12px;"><span class="badge info">${{esc(s.probable_report_type||'series')}}</span> ${{s.scan_count}} scans, ${{esc(vars)}} <span class="muted">(${{esc(kinds)}})</span></div>`;
    }}
    if (series.length > 8) html += `<p class="muted">... and ${{series.length-8}} more series</p>`;
  }}

  const nf = summary.narrow_feature_scans || [];
  if (nf.length) {{
    html += `<h3>Narrow Features (${{nf.length}} scans)</h3>`;
    for (const f of nf.slice(0,5)) html += `<div style="font-size:12px;"><span class="badge narrow">${{esc(f.scan_id)}}</span> ${{f.flag_count}} flag(s)</div>`;
    if (nf.length > 5) html += `<p class="muted">... and ${{nf.length-5}} more</p>`;
  }}

  const nextSteps = review.ranked_next_steps || [];
  if (nextSteps.length) {{
    html += '<h3>Next Steps</h3><ul style="font-size:12px;padding-left:16px;">';
    for (const step of nextSteps.slice(0, 5)) {{
      const label = typeof step === 'string' ? step : (step.description || step.label || JSON.stringify(step));
      html += `<li>${{esc(label)}}</li>`;
    }}
    html += '</ul>';
  }}

  const artifacts = summary.experiment_review_artifacts || {{}};
  const artifactLinks = Object.entries(artifacts).filter(([k,v]) => v && typeof v === 'string');
  if (artifactLinks.length) {{
    html += '<h3>Full Review Artifacts</h3><div style="font-size:12px;">';
    for (const [key, url] of artifactLinks) {{
      html += `<a href="${{esc(url)}}" target="_blank" style="display:inline-block;margin:2px 6px 2px 0;">${{esc(key.replace(/_/g,' '))}}</a>`;
    }}
    html += '</div>';
  }}

  el.innerHTML = html || '<p class="muted">No experiment summary available.</p>';
  for (let oi = 0; oi < op.length; oi++) {{
    const peaks = (op[oi].per_scan_peak_summary || []).filter(p => p.temperature_mean != null && p.integrated_signal != null);
    const plotEl = document.getElementById(`op-plot-${{oi}}`);
    if (plotEl && peaks.length >= 2) {{
      peaks.sort((a, b) => a.temperature_mean - b.temperature_mean);
      Plotly.newPlot(plotEl, [{{
        x: peaks.map(p => p.temperature_mean),
        y: peaks.map(p => p.integrated_signal),
        mode: 'lines+markers', type: 'scatter', name: 'Integrated signal'
      }}], {{
        xaxis: {{title: 'Temperature (K)'}},
        yaxis: {{title: 'Integrated signal'}},
        margin: {{l:50, r:10, t:30, b:40}},
        title: op[oi].candidate_id || 'Order parameter candidate'
      }}, {{responsive: true}});
    }}
  }}
}}

document.getElementById('x-axis-select').addEventListener('change', () => {{ if (currentRecord) redrawPlot(currentRecord); }});
document.getElementById('y-axis-select').addEventListener('change', () => {{ if (currentRecord) redrawPlot(currentRecord); }});
document.getElementById('overlay-btn').addEventListener('click', overlaySelected);
document.getElementById('select-all-scans').addEventListener('change', (e) => {{
  document.querySelectorAll('.scan-check').forEach(cb => cb.checked = e.target.checked);
  document.querySelectorAll('.group-check').forEach(cb => cb.checked = e.target.checked);
}});
document.getElementById('metadata').innerHTML =
  '<h3>Candidate</h3>' + metadataGrid(payload.candidate_metadata || {{}}) +
  '<h3>File/Header Summary</h3>' + metadataGrid(payload.file_metadata_summary || {{}}) +
  '<h3>Score Breakdown</h3>' + metadataGrid(payload.score_breakdown || {{}}) +
  '<h3>Paper Constraints</h3>' + metadataGrid({{constraint_count:(payload.paper_constraints || []).length, constraints:payload.paper_constraints || []}});
const select = document.getElementById('scan-select');
select.innerHTML = scans.map(s => `<option value="${{esc(s.scan_id)}}">${{esc(s.scan_id)}}</option>`).join('');
select.addEventListener('change', event => draw(event.target.value));

if (overflowCount > 0 && sidecarUrl) {{
  const container = document.getElementById('load-more-container');
  container.style.display = 'block';
  document.getElementById('load-all-btn').addEventListener('click', async () => {{
    const status = document.getElementById('load-status');
    status.textContent = 'Loading...';
    try {{
      const resp = await fetch(sidecarUrl);
      const allScans = await resp.json();
      scans = allScans;
      renderScanTable();
      select.innerHTML = scans.map(s => `<option value="${{esc(s.scan_id)}}">${{esc(s.scan_id)}}</option>`).join('');
      container.style.display = 'none';
      status.textContent = '';
    }} catch (err) {{
      status.textContent = 'Failed: ' + err.message;
    }}
  }});
}}

renderScanTable();
renderExperimentSummary();
draw((scans[0] || {{}}).scan_id);
</script>
</body>
</html>
"""


def _paper_experiment_review_dashboard_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload).replace("</script>", "<\\/script>")
    fallback_rows = "\n".join(_paper_experiment_review_fallback_row(record) for record in payload.get("records", []))
    initial_count = len(payload.get("records", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BT-7 Paper-to-Experiment Review</title>
<style>
:root {{ --bg:#f6f8fb; --card:#fff; --ink:#18212f; --muted:#596579; --line:#d8dee9; --accent:#174ea6; --ok:#166534; --bad:#991b1b; --warn:#92400e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
header {{ padding:22px 28px; background:#fff; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:24px; }}
p {{ margin:0 0 8px; line-height:1.45; }}
.toolbar {{ padding:10px 16px; background:var(--bg); border-bottom:1px solid var(--line); }}
.filter-row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:flex-end; margin-bottom:6px; }}
.filter-row label {{ font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:2px; }}
.filter-row input, .filter-row select {{ padding:5px 6px; font-size:12px; }}
.metric-inline {{ font-size:12px; color:var(--muted); margin-right:4px; }}
.metric-inline b {{ color:var(--ink); }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:10px; padding:10px; height:calc(100vh - 160px); }}
aside, section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; overflow-y:auto; }}
label {{ display:grid; gap:4px; font-size:13px; color:var(--muted); }}
input, select, textarea, button {{ font:inherit; }}
input, select, textarea {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; }}
textarea {{ min-height:90px; resize:vertical; }}
button {{ border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
button.secondary {{ background:#fff; color:var(--accent); }}
button.ghost {{ background:#fff; color:var(--ink); border-color:var(--line); }}
.metric-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin:12px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ border-bottom:1px solid var(--line); text-align:left; vertical-align:top; padding:7px; }}
th {{ background:#f0f3f8; }}
tr.selected {{ outline:2px solid var(--accent); outline-offset:-2px; }}
.badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:1px 7px; font-size:12px; background:#fff; margin:1px 2px 1px 0; }}
.badge.high_review_priority {{ color:var(--bad); border-color:#fecaca; background:#fff5f5; }}
.badge.medium_review_priority {{ color:var(--warn); border-color:#fed7aa; background:#fff7ed; }}
.badge.low_review_priority {{ color:var(--muted); }}
.badge.accepted {{ color:var(--ok); border-color:#bbf7d0; background:#f0fdf4; }}
.badge.rejected {{ color:var(--bad); border-color:#fecaca; background:#fff5f5; }}
.badge.needs_more_evidence {{ color:var(--warn); border-color:#fed7aa; background:#fff7ed; }}
.small {{ color:var(--muted); font-size:12px; }}
.detail-list {{ margin:8px 0; padding-left:18px; }}
.detail-list li {{ margin:4px 0; }}
.links a {{ display:inline-block; margin-right:8px; }}
.metadata-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:6px; margin:8px 0; }}
.metadata-item {{ border:1px solid var(--line); border-radius:6px; padding:6px; background:#fbfcfe; }}
.metadata-item b {{ display:block; color:var(--ink); }}
@media (max-width:980px) {{ main {{ grid-template-columns:1fr; height:auto; }} }}
</style>
</head>
<body>
<header>
<h1>BT-7 Paper-to-Experiment Review</h1>
<p>{html.escape(str(payload.get("safe_conclusion") or ""))}</p>
<p class="small">Source: <code>{html.escape(str(payload.get("source_review_template") or ""))}</code></p>
<p class="small">Rows loaded: <code>{initial_count}</code>. If the interactive table is empty, use the static fallback rows below or open the CSV summary next to this HTML file.</p>
</header>
<div class="toolbar">
  <div class="filter-row">
    <label>Filter <input id="textFilter" type="search" placeholder="paper, candidate, evidence"></label>
    <label>Priority <select id="priorityFilter"><option value="">all</option><option>high_review_priority</option><option>medium_review_priority</option><option>low_review_priority</option><option>unprioritized</option></select></label>
    <label>Status <select id="statusFilter"><option value="">all</option><option>unreviewed</option><option>accepted</option><option>rejected</option><option>needs_more_evidence</option><option>duplicate</option><option>not_enough_information</option></select></label>
    <label>Min score <input id="scoreFilter" type="number" min="0" max="1" step="0.01" placeholder="0.50" style="width:80px;"></label>
    <button id="showAll" class="secondary">Show all</button>
  </div>
  <div class="filter-row">
    <span class="metric-inline"><b id="visibleCount">{initial_count}</b> visible</span>
    <span class="metric-inline"><b id="reviewedCount">0</b> reviewed</span>
    <span class="metric-inline"><b id="acceptedCount">0</b> accepted</span>
    <span class="metric-inline"><b id="needsCount">0</b> needs evidence</span>
    <button id="exportJsonl">Export reviewed JSONL</button>
    <button id="exportAllJsonl" class="secondary">Export all JSONL</button>
    <button id="clearLocal" class="ghost">Clear edits</button>
  </div>
  <p id="renderError" class="small" style="display:none;color:var(--bad);"></p>
</div>
<main>
<section style="overflow-x:auto;overflow-y:auto;">
  <table id="reviewTable" style="min-width:900px;">
    <thead><tr>
      <th data-col="candidate_rank" style="cursor:pointer">Rank <span class="sort-ind"></span></th>
      <th data-col="paper_id" style="cursor:pointer">Paper <span class="sort-ind"></span></th>
      <th data-col="candidate_experiment_id" style="cursor:pointer">Candidate <span class="sort-ind"></span></th>
      <th data-col="probability_score" style="cursor:pointer">Score <span class="sort-ind"></span></th>
      <th data-col="review_priority" style="cursor:pointer">Priority <span class="sort-ind"></span></th>
      <th data-col="human_review_status" style="cursor:pointer">Status <span class="sort-ind"></span></th>
      <th>Evidence</th>
    </tr></thead>
    <tbody>{fallback_rows}</tbody>
  </table>
</section>
<aside>
  <h3>Review Selected Row</h3>
  <div id="detail" class="small">Click a row in the table to review it.</div>
</aside>
</main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const records = Array.isArray(payload.records) ? payload.records : [];
const storageKey = 'tas-paper-experiment-review::' + payload.source_review_template;
let edits = {{}};
try {{
  const rawEdits = localStorage.getItem(storageKey);
  edits = rawEdits ? (JSON.parse(rawEdits) || {{}}) : {{}};
}} catch (err) {{
  localStorage.removeItem(storageKey);
  document.getElementById('renderError').style.display = 'block';
  document.getElementById('renderError').textContent = 'Ignored corrupt saved browser edits for this local review page. Rows below are from the embedded review template.';
  edits = {{}};
}}
let selectedId = null;
let sortCol = null, sortDir = 1;
const PRIORITY_ORDER = {{'high_review_priority':0,'medium_review_priority':1,'low_review_priority':2,'unprioritized':3}};
const STATUS_ORDER = {{'accepted':0,'needs_more_evidence':1,'not_enough_information':2,'duplicate':3,'unreviewed':4,'rejected':5}};
function sortValue(r, col) {{
  const m = mergedRecord(r);
  if (col === 'review_priority') return PRIORITY_ORDER[r.review_priority] ?? 99;
  if (col === 'human_review_status') return STATUS_ORDER[m.human_review_status || 'unreviewed'] ?? 99;
  if (col === 'probability_score') return -(Number(r.probability_score || 0));
  if (col === 'candidate_rank') return Number(r.candidate_rank || 0);
  return String(r[col] || m[col] || '').toLowerCase();
}}
function updateSortIndicators() {{
  document.querySelectorAll('#reviewTable th[data-col]').forEach(th => {{
    const ind = th.querySelector('.sort-ind');
    if (!ind) return;
    ind.textContent = th.dataset.col === sortCol ? (sortDir === 1 ? ' ▲' : ' ▼') : '';
  }});
}}
function mergedRecord(r) {{ return {{...r, ...(edits[r.review_id] || {{}})}}; }}
function saveEdit(id, patch) {{ edits[id] = {{...(edits[id] || {{}}), ...patch}}; localStorage.setItem(storageKey, JSON.stringify(edits)); render(); selectRow(id); }}
function listText(value) {{ return Array.isArray(value) ? value.join(' ') : String(value || ''); }}
function rowText(r) {{ return [r.paper_id,r.paper_title,r.candidate_experiment_id,r.review_priority,listText(r.evidence_to_check),listText(r.warnings),listText(r.cautions)].join(' ').toLowerCase(); }}
function hasContent(value) {{ return value && (!(Array.isArray(value)) || value.length) && (typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length); }}
function compactValue(value) {{
  if (Array.isArray(value)) return value.map(v => typeof v === 'object' ? JSON.stringify(v) : String(v)).join(', ') || 'none';
  if (value && typeof value === 'object') return JSON.stringify(value);
  return String(value ?? 'none');
}}
function metadataGrid(obj) {{
  if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) return '<p class="small">No structured metadata summary available for this candidate.</p>';
  return `<div class="metadata-grid">${{Object.entries(obj).map(([k,v]) => `<div class="metadata-item"><b>${{escapeHtml(k)}}</b><span>${{escapeHtml(compactValue(v))}}</span></div>`).join('')}}</div>`;
}}
function listItems(value, empty='none') {{
  if (!Array.isArray(value) || !value.length) return `<li>${{escapeHtml(empty)}}</li>`;
  return value.map(e => `<li>${{escapeHtml(typeof e === 'object' ? JSON.stringify(e) : e)}}</li>`).join('');
}}
function scoreBreakdown(obj) {{
  if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) return '<p class="small">No score breakdown available.</p>';
  return `<ul class="detail-list">${{Object.entries(obj).sort().map(([k,v]) => `<li><code>${{escapeHtml(k)}}</code>: ${{escapeHtml(compactValue(v))}}</li>`).join('')}}</ul>`;
}}
function reviewPacketLinks(packet) {{
  if (!packet || typeof packet !== 'object') return '';
  const parts = [];
  if (packet.html_url) parts.push(`<a href="${{packet.html_url}}" target="_blank">Review packet with plots</a>`);
  if (packet.json_url) parts.push(`<a href="${{packet.json_url}}" target="_blank">Packet JSON</a>`);
  if (!parts.length) return '';
  const status = packet.status ? ` <span class="small">(${{escapeHtml(packet.status)}}, ${{escapeHtml(packet.parsed_scan_count || 0)}} parsed scans)</span>` : '';
  return `<div class="links">${{parts.join('')}}${{status}}</div>`;
}}
function paperOpenLink(url) {{
  return url ? `<div class="links"><a href="${{url}}" target="_blank">Open paper from local cache/Vista</a></div>` : '';
}}
function paperConstraintList(rows) {{
  if (!Array.isArray(rows) || !rows.length) return '<p class="small">No paper-constraint summary available.</p>';
  return rows.map((row, idx) => `<div class="metadata-item"><b>Constraint ${{idx + 1}}</b>
    <div>Instrument: ${{escapeHtml(compactValue(row.instrument))}}</div>
    <div>Detector: ${{escapeHtml(compactValue(row.detector_modes))}}</div>
    <div>Measurement: ${{escapeHtml(compactValue(row.measurement_kinds))}}</div>
    <div>Sample env: ${{escapeHtml(compactValue(row.sample_environment_terms))}}</div>
    <div>Formula/title strings: ${{escapeHtml(compactValue(row.formula_like_title_strings))}}</div>
    <div>Evidence count: ${{escapeHtml(compactValue(row.evidence_count))}}</div>
  </div>`).join('');
}}
function passes(r) {{
  const q = document.getElementById('textFilter').value.toLowerCase();
  const pr = document.getElementById('priorityFilter').value;
  const st = document.getElementById('statusFilter').value;
  const minScore = parseFloat(document.getElementById('scoreFilter').value || '0');
  const m = mergedRecord(r);
  return (!q || rowText(r).includes(q)) && (!pr || r.review_priority === pr) && (!st || (m.human_review_status || 'unreviewed') === st) && (Number(r.probability_score || 0) >= minScore);
}}
function escapeHtml(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function badge(text, cls='') {{ return `<span class="badge ${{cls}}">${{escapeHtml(text || '')}}</span>`; }}
function render() {{
  const tbody = document.querySelector('#reviewTable tbody');
  let visible = records.filter(passes);
  if (sortCol) {{
    visible = [...visible].sort((a, b) => {{
      const av = sortValue(a, sortCol), bv = sortValue(b, sortCol);
      return sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
    }});
  }}
  tbody.innerHTML = visible.map(r => {{
    const m = mergedRecord(r);
    const ev = (r.evidence_to_check || []).slice(0, 3).map(e => `<div>${{escapeHtml(e)}}</div>`).join('');
    return `<tr data-id="${{escapeHtml(r.review_id)}}" class="${{selectedId === r.review_id ? 'selected' : ''}}">
      <td>${{r.candidate_rank}}</td><td><b>${{escapeHtml(r.paper_id)}}</b><br>${{escapeHtml(r.paper_title)}}</td>
      <td><code>${{escapeHtml(r.candidate_experiment_id)}}</code><br><span class="small">${{escapeHtml(r.confidence_label)}} / ${{escapeHtml(r.candidate_status)}}</span></td>
      <td>${{Number(r.probability_score || 0).toFixed(3)}}</td><td>${{badge(r.review_priority, r.review_priority)}}</td>
      <td>${{badge(m.human_review_status || 'unreviewed', m.human_review_status || 'unreviewed')}}</td><td>${{ev}}</td></tr>`;
  }}).join('');
  tbody.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => selectRow(tr.dataset.id)));
  document.getElementById('visibleCount').textContent = visible.length;
  const merged = records.map(mergedRecord);
  document.getElementById('reviewedCount').textContent = merged.filter(r => (r.human_review_status || 'unreviewed') !== 'unreviewed').length;
  document.getElementById('acceptedCount').textContent = merged.filter(r => r.human_review_status === 'accepted').length;
  document.getElementById('needsCount').textContent = merged.filter(r => r.human_review_status === 'needs_more_evidence').length;
}}
function selectRow(id) {{
  selectedId = id;
  const base = records.find(r => r.review_id === id);
  if (!base) return;
  const r = mergedRecord(base);
  const allowedStatuses = (Array.isArray(base.allowed_human_review_statuses) && base.allowed_human_review_statuses.length) ? base.allowed_human_review_statuses : ['accepted','rejected','needs_more_evidence','duplicate','not_enough_information'];
  const statusOptions = ['unreviewed', ...allowedStatuses].map(s => `<option value="${{s}}" ${{(r.human_review_status||'unreviewed')===s?'selected':''}}>${{s}}</option>`).join('');
  const detail = document.getElementById('detail');
  detail.innerHTML = `<p><b>${{escapeHtml(base.paper_title)}}</b></p>
    <p>Paper: <code>${{escapeHtml(base.paper_id)}}</code><br>Candidate: <code>${{escapeHtml(base.candidate_experiment_id)}}</code><br>Score: <code>${{base.probability_score}}</code> ${{badge(base.review_priority, base.review_priority)}}</p>
    ${{paperOpenLink(base.paper_open_url)}}
    <div class="links">${{base.source_report_md_url ? `<a href="${{base.source_report_md_url}}" target="_blank">Markdown report</a>` : ''}}${{base.source_report_json_url ? `<a href="${{base.source_report_json_url}}" target="_blank">JSON report</a>` : ''}}</div>
    ${{reviewPacketLinks(base.review_packet)}}
    <p class="small">For a real paper-link decision, open the paper or paper brief separately, inspect the candidate scan dashboard/plots in the review packet where available, and use the metadata below as linkage evidence. This page stores the label; it is not itself the full scientific review packet.</p>
    <label>Status <select id="editStatus">${{statusOptions}}</select></label>
    <label>Reviewer <input id="editReviewer" value="${{escapeHtml(r.reviewer || '')}}" placeholder="name"></label>
    <label>Notes <textarea id="editNotes" placeholder="Why accept/reject? What evidence is missing?">${{escapeHtml(r.review_notes || '')}}</textarea></label>
    <button id="saveRow">Save row locally</button>
    <h4>Paper constraints</h4>${{paperConstraintList(base.paper_constraints)}}
    <h4>Candidate metadata</h4>${{metadataGrid(base.candidate_metadata)}}
    <h4>Hydrated file/header metadata</h4>${{metadataGrid(base.file_metadata_summary)}}
    <h4>Column names sample</h4><ul class="detail-list">${{listItems((base.file_metadata_summary || {{}}).column_names_sample, 'no column names available')}}</ul>
    <h4>Historic path/filename clues</h4>${{metadataGrid(base.historic_clue_summary)}}
    <h4>Score breakdown</h4>${{scoreBreakdown(base.score_breakdown)}}
    <h4>Evidence to check</h4><ul class="detail-list">${{(base.evidence_to_check || []).map(e => `<li>${{escapeHtml(e)}}</li>`).join('')}}</ul>
    <h4>Review questions</h4><ul class="detail-list">${{(base.review_questions || []).map(e => `<li>${{escapeHtml(e)}}</li>`).join('')}}</ul>
    <h4>Cautions / warnings</h4><ul class="detail-list">${{[...(base.cautions || []), ...(base.warnings || [])].map(e => `<li>${{escapeHtml(e)}}</li>`).join('')}}</ul>`;
  document.getElementById('saveRow').addEventListener('click', () => {{
    saveEdit(id, {{human_review_status: document.getElementById('editStatus').value, reviewer: document.getElementById('editReviewer').value || null, review_notes: document.getElementById('editNotes').value || null, reviewed_at: new Date().toISOString()}});
  }});
  render();
}}
function exportRows(reviewedOnly) {{
  const outRows = records.map(mergedRecord).filter(r => !reviewedOnly || (r.human_review_status || 'unreviewed') !== 'unreviewed').map(r => ({{
      paper_id:r.paper_id, paper_title:r.paper_title, candidate_experiment_id:r.candidate_experiment_id, candidate_rank:r.candidate_rank,
      probability_score:r.probability_score, review_priority:r.review_priority, human_review_status:r.human_review_status || 'unreviewed',
      reviewer:r.reviewer || null, reviewed_at:r.reviewed_at || null, review_notes:r.review_notes || null,
      source_report:r.source_report_json, review_packet:(r.review_packet || {{}}).html || null, paper_open_url:r.paper_open_url || null, provenance:r.provenance || []
  }}));
  const blob = new Blob([outRows.map(x => JSON.stringify(x)).join('\\n') + (outRows.length ? '\\n' : '')], {{type:'application/jsonl'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = reviewedOnly ? 'paper_experiment_reviewed_rows.jsonl' : 'paper_experiment_review_all_rows.jsonl'; a.click(); URL.revokeObjectURL(a.href);
}}
['textFilter','priorityFilter','statusFilter','scoreFilter'].forEach(id => document.getElementById(id).addEventListener('input', render));
document.querySelectorAll('#reviewTable th[data-col]').forEach(th => th.addEventListener('click', () => {{
  const col = th.dataset.col;
  if (sortCol === col) {{ sortDir = -sortDir; }} else {{ sortCol = col; sortDir = 1; }}
  updateSortIndicators();
  render();
}}));
document.getElementById('exportJsonl').addEventListener('click', () => exportRows(true));
document.getElementById('exportAllJsonl').addEventListener('click', () => exportRows(false));
document.getElementById('showAll').addEventListener('click', () => {{ document.getElementById('textFilter').value=''; document.getElementById('priorityFilter').value=''; document.getElementById('statusFilter').value=''; document.getElementById('scoreFilter').value=''; render(); }});
document.getElementById('clearLocal').addEventListener('click', () => {{ if (confirm('Clear local edits for this review page?')) {{ edits={{}}; localStorage.removeItem(storageKey); selectedId=null; render(); document.getElementById('detail').textContent='Select a row to review.'; }} }});
try {{
  render();
}} catch (err) {{
  document.getElementById('renderError').style.display = 'block';
  document.getElementById('renderError').textContent = 'Interactive rendering failed; showing static fallback rows from the embedded review template. Error: ' + err.message;
}}
</script>
</body>
</html>
"""


def _paper_experiment_review_fallback_row(record: dict[str, Any]) -> str:
    evidence = record.get("evidence_to_check") if isinstance(record.get("evidence_to_check"), list) else []
    evidence_html = "".join(f"<div>{html.escape(str(item))}</div>" for item in evidence[:3])
    return (
        f'<tr data-id="{html.escape(str(record.get("review_id") or ""))}">'
        f"<td>{html.escape(str(record.get('candidate_rank') or ''))}</td>"
        f"<td><b>{html.escape(str(record.get('paper_id') or ''))}</b><br>{html.escape(str(record.get('paper_title') or ''))}</td>"
        f"<td><code>{html.escape(str(record.get('candidate_experiment_id') or ''))}</code><br>"
        f"<span class=\"small\">{html.escape(str(record.get('confidence_label') or ''))} / {html.escape(str(record.get('candidate_status') or ''))}</span></td>"
        f"<td>{float(record.get('probability_score') or 0):.3f}</td>"
        f"<td><span class=\"badge {html.escape(str(record.get('review_priority') or ''))}\">{html.escape(str(record.get('review_priority') or ''))}</span></td>"
        f"<td><span class=\"badge {html.escape(str(record.get('human_review_status') or 'unreviewed'))}\">{html.escape(str(record.get('human_review_status') or 'unreviewed'))}</span></td>"
        f"<td>{evidence_html}</td>"
        "</tr>"
    )


def _bt7_hydrated_index_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BT-7 Hydrated Index",
            "",
            f"Status: `{report.get('status')}`",
            f"Data catalog: `{report.get('data_catalog')}`",
            f"Output: `{report.get('output')}`",
            f"Attempted files: {report.get('attempted_files', 0)}",
            f"Read files: {report.get('read_files', 0)}",
            f"Missing files: {report.get('missing_files', 0)}",
            f"Reader-derived groups: {report.get('group_count', 0)}",
            f"Reader-derived source units: {report.get('source_unit_count', 0)}",
            f"Candidate records: {report.get('candidate_count', report.get('group_count', 0))}",
            "",
            "## Policy",
            str(report.get("index_policy") or ""),
            "",
            "## Read Errors",
            *[f"- {item}" for item in report.get("read_errors", []) or ["none"]],
        ]
    ) + "\n"


def _eval_markdown(report: dict[str, Any]) -> str:
    lines = ["# Corpus Pilot Retrieval Eval", "", f"Status: `{report['status']}`", f"Checks: {report['check_count']}", ""]
    for check in report["checks"]:
        lines.append(f"- `{check['status']}` {check['name']}")
    return "\n".join(lines) + "\n"


def _agent_markdown(report: dict[str, Any]) -> str:
    lines = ["# Agent Readiness Checklist", "", f"Status: `{report['status']}`", ""]
    lines.extend(f"- `{item['status']}` {item['name']}" for item in report["checks"])
    lines.extend(["", report["safe_conclusion"]])
    return "\n".join(lines) + "\n"
