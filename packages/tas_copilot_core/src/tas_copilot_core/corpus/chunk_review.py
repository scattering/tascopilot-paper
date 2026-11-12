"""Chunk-quality review surface over corpus catalogs and extractor outputs.

Backs the workbench `/chunks` page: a paper table from a corpus catalog, the
chunks for a selected paper, the raw per-paper extractor markdown from a
bakeoff run, and chunk-extraction-quality review labels written as full
`ReviewLabel` records so the existing unified ingest path can consume them.

Extracted corpus text is untrusted evidence. This module only reads catalog
JSONL files and extractor markdown from local ignored directories; it never
executes extracted content and never runs extractors itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ReviewLabel

# Maps the dashboard's extraction-quality statuses onto the normalized
# ReviewVerdict axis; the specific status is preserved in type_specific_status.
CHUNK_REVIEW_STATUSES: dict[str, str] = {
    "acceptable": "positive",
    "minor_issues": "inconclusive",
    "missing_content": "negative",
    "wrong_sections": "negative",
    "garbled": "negative",
    "caption_table_failure": "negative",
    "unusable": "negative",
}

_CHUNK_FIELDS = [
    "chunk_id",
    "chunk_type",
    "section",
    "subsection",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "figure_id",
    "panel_id",
    "table_id",
    "token_count",
    "extraction_method",
    "text",
]


def discover_chunk_catalogs(runs_root: Path) -> list[dict[str, str]]:
    """List corpus catalog directories that contain papers and chunks."""
    catalogs: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in _candidate_roots(runs_root, "corpus_catalogs"):
        if not root.exists():
            continue
        for chunks_path in sorted(root.glob("*/source_chunks.jsonl")):
            catalog_dir = chunks_path.parent.resolve()
            if catalog_dir in seen or not (catalog_dir / "papers.jsonl").exists():
                continue
            seen.add(catalog_dir)
            catalogs.append(
                {
                    "path": str(catalog_dir),
                    "label": catalog_dir.name,
                    "modified": chunks_path.stat().st_mtime,
                }
            )
    catalogs.sort(key=lambda item: item["modified"], reverse=True)
    return [{"path": item["path"], "label": item["label"]} for item in catalogs]


def discover_extractor_runs(runs_root: Path) -> list[dict[str, Any]]:
    """List extractor bakeoff runs and the backends each one contains."""
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in _candidate_roots(runs_root, "vista_corpus_extractor_outputs"):
        if not root.exists():
            continue
        for summary in sorted(root.glob("*/outputs/*/summary.json")):
            run_dir = summary.parent.parent.parent.resolve()
            if run_dir in seen:
                continue
            backends = sorted(
                candidate.parent.name
                for candidate in run_dir.glob("outputs/*/summary.json")
                if candidate.parent.name in _TEXT_BACKENDS
            )
            seen.add(run_dir)
            runs.append({"path": str(run_dir), "label": run_dir.name, "backends": backends, "modified": summary.stat().st_mtime})
    runs.sort(key=lambda item: item["modified"], reverse=True)
    return [{"path": item["path"], "label": item["label"], "backends": item["backends"]} for item in runs]


def paper_table(catalog_dir: str | Path) -> dict[str, Any]:
    """Build the paper-selection table: identity fields plus chunk-type counts."""
    catalog = Path(catalog_dir).expanduser().resolve()
    papers = {str(row.get("paper_id") or ""): row for row in _read_jsonl(catalog / "papers.jsonl")}
    documents = {str(row.get("paper_id") or row.get("document_id") or ""): row for row in _read_jsonl(catalog / "documents.jsonl")}
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    material_guesses: dict[str, str] = {}
    for chunk in _read_jsonl(catalog / "source_chunks.jsonl"):
        paper_id = str(chunk.get("paper_id") or "")
        chunk_type = str(chunk.get("chunk_type") or "unknown")
        counts.setdefault(paper_id, {})
        counts[paper_id][chunk_type] = counts[paper_id].get(chunk_type, 0) + 1
        totals[paper_id] = totals.get(paper_id, 0) + 1
        if chunk_type == "material_composition" and paper_id not in material_guesses:
            candidates = (chunk.get("metadata") or {}).get("formula_candidates") or []
            if candidates:
                material_guesses[paper_id] = str(candidates[0].get("formula") or "")
    rows = []
    for paper_id in sorted((set(papers) | set(totals)) - {""}):
        paper = papers.get(paper_id, {})
        document = documents.get(paper_id, {})
        rows.append(
            {
                "paper_id": paper_id,
                "title": paper.get("title") or document.get("title"),
                "authors": list(paper.get("authors") or []),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "pdf_name": Path(str(document.get("local_path") or "")).name or None,
                "chunk_count": totals.get(paper_id, 0),
                "chunk_type_counts": counts.get(paper_id, {}),
                "material_guess": material_guesses.get(paper_id),
            }
        )
    rows.sort(key=lambda item: (-(item["chunk_count"]), str(item["title"] or "")))
    return {"catalog": str(catalog), "paper_count": len(rows), "papers": rows}


def paper_chunk_detail(catalog_dir: str | Path, paper_id: str) -> dict[str, Any]:
    """Return paper identity plus its chunks in document order."""
    catalog = Path(catalog_dir).expanduser().resolve()
    paper = next((row for row in _read_jsonl(catalog / "papers.jsonl") if str(row.get("paper_id")) == paper_id), None)
    document = next(
        (row for row in _read_jsonl(catalog / "documents.jsonl") if paper_id in {str(row.get("paper_id")), str(row.get("document_id"))}),
        None,
    )
    chunks = [
        {field: chunk.get(field) for field in _CHUNK_FIELDS}
        for chunk in _read_jsonl(catalog / "source_chunks.jsonl")
        if str(chunk.get("paper_id")) == paper_id
    ]
    chunks.sort(key=lambda item: (item.get("char_start") if isinstance(item.get("char_start"), (int, float)) else 0))
    abstract = next((chunk["text"] for chunk in chunks if chunk.get("chunk_type") == "abstract"), None)
    return {
        "catalog": str(catalog),
        "paper_id": paper_id,
        "title": (paper or {}).get("title") or (document or {}).get("title"),
        "authors": list((paper or {}).get("authors") or []),
        "year": (paper or {}).get("year"),
        "doi": (paper or {}).get("doi"),
        "pdf_name": Path(str((document or {}).get("local_path") or "")).name or None,
        "abstract": abstract,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


# Backends with per-paper readable artifacts; extractor_comparison and other
# summary-only directories carry no reviewable text.
_TEXT_BACKENDS = ("docling", "grobid")


def extractor_outputs(
    run_dir: str | Path,
    pdf_name: str,
    *,
    max_chars: int = 250000,
    derive_chunks: bool = False,
    paper: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve the per-paper readable text for every text backend in a bakeoff run.

    Returns one entry per backend so the review page can show outputs side by
    side. Docling yields markdown; GROBID yields TEI converted to plain text by
    the same reader the catalog ingest uses, so the comparison matches what a
    rebuilt catalog would actually contain. With ``derive_chunks``, each
    backend's text is also run through the same chunker the ingest uses, so
    chunk-level differences between extractors are directly comparable.
    """
    from .pilot import _chunks_from_text, _extractor_text_path, _read_extractor_text

    run = Path(run_dir).expanduser().resolve()
    stem = Path(pdf_name).stem
    paper = paper or {}
    outputs: dict[str, dict[str, Any]] = {}
    for backend in _TEXT_BACKENDS:
        summary_path = run / "outputs" / backend / "summary.json"
        if not summary_path.exists():
            outputs[backend] = {"status": "missing_summary", "summary_path": str(summary_path)}
            continue
        try:
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            outputs[backend] = {"status": "unreadable_summary", "summary_path": str(summary_path), "error": str(exc)}
            continue
        row = next(
            (
                item
                for item in rows
                if isinstance(item, dict) and Path(str(item.get("pdf_name") or "")).stem == stem
            ),
            None,
        )
        if row is None:
            outputs[backend] = {"status": "paper_not_in_run", "pdf_name": pdf_name}
            continue
        text_path, method, _ = _extractor_text_path(run, row, backend)
        if text_path is None:
            outputs[backend] = {"status": "artifact_not_synced_locally", "pdf_name": pdf_name}
            continue
        text = _read_extractor_text(text_path, backend)[:max_chars]
        output: dict[str, Any] = {
            "status": "ok",
            "backend": backend,
            "kind": "markdown" if backend == "docling" else "tei_text",
            "extraction_method": method,
            "path": str(text_path),
            "text": text,
        }
        if derive_chunks:
            output["chunks"] = _derive_preview_chunks(paper=paper, pdf_name=pdf_name, text=text, method=method or backend, chunker=_chunks_from_text)
        outputs[backend] = output
    return outputs


def _derive_preview_chunks(*, paper: dict[str, Any], pdf_name: str, text: str, method: str, chunker: Any) -> list[dict[str, Any]]:
    """Run the catalog's chunker over one backend's text for review preview.

    Preview chunks are derived on the fly and never written to the catalog;
    they exist so the review page can compare what each extractor's text would
    chunk into, using identical chunking logic.
    """
    from tas_copilot_contracts import Document

    paper_id = str(paper.get("paper_id") or f"preview-{Path(pdf_name).stem}")
    document = Document(
        document_id=paper_id,
        document_type="paper",
        title=paper.get("title"),
        paper_id=paper_id,
        local_path=pdf_name,
        copyright_status="private_local_only",
        ingest_status="extracted_text_available",
        extraction_method=method,
        metadata={"year": paper.get("year"), "doi": paper.get("doi")},
    )
    derived = chunker(document, text, method=method)
    derived.sort(key=lambda item: (item.get("char_start") if isinstance(item.get("char_start"), (int, float)) else 0))
    return [{field: chunk.get(field) for field in [*_CHUNK_FIELDS, "metadata"]} for chunk in derived]


def record_chunk_review_label(runs_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and append one chunk-extraction-quality label.

    Rows are full `ReviewLabel` records, so `ingest_review_labels` accepts the
    export file directly. Labels are calibration evidence about extraction
    quality, not acceptance of any scientific claim.
    """
    paper_id = str(payload.get("paper_id") or "").strip()
    status = str(payload.get("status") or "").strip()
    if not paper_id:
        return {"status": "invalid", "reason": "paper_id is required"}
    if status not in CHUNK_REVIEW_STATUSES:
        return {"status": "invalid", "reason": f"status must be one of {sorted(CHUNK_REVIEW_STATUSES)}"}
    catalog = str(payload.get("catalog") or "")
    chunk_id = str(payload.get("chunk_id") or "").strip() or None
    label_seed = "|".join([catalog, paper_id, chunk_id or "paper", str(payload.get("extractor_run") or ""), str(payload.get("backend") or "")])
    label = ReviewLabel(
        label_id=f"review-label-{hashlib.sha256(label_seed.encode('utf-8')).hexdigest()[:16]}",
        object_type="chunk_extraction_quality",
        object_id=paper_id,
        object_secondary_id=chunk_id,
        source_workflow="chunk-review-dashboard",
        source_artifact_path=catalog or None,
        verdict=CHUNK_REVIEW_STATUSES[status],
        type_specific_status=status,
        type_specific_allowed=sorted(CHUNK_REVIEW_STATUSES),
        claim_state="candidate",
        recommendation="calibrate_prior",
        uncertainty_class="unknown",
        review_notes=str(payload.get("comment") or ""),
        correction_status_snapshot={
            "reviewer": str(payload.get("reviewer") or ""),
            "extractor_run": str(payload.get("extractor_run") or ""),
            "backend": str(payload.get("backend") or ""),
            "best_backend": str(payload.get("best_backend") or "not_compared"),
            "labeled_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    labels_path = Path(runs_root) / "chunk_review" / "chunk_review_labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a", encoding="utf-8") as handle:
        handle.write(label.model_dump_json() + "\n")
    return {
        "status": "ok",
        "label_id": label.label_id,
        "labels_jsonl": str(labels_path),
        "verdict": label.verdict,
        "type_specific_status": status,
        "safe_conclusion": "Chunk-quality labels are extraction-calibration evidence; ingest them into the unified store via /labels.",
    }


def paper_chunk_labels(runs_root: Path, paper_id: str, catalog: str | None = None) -> dict[str, Any]:
    """Return previously saved chunk-quality labels for one paper.

    Reads the append-only ``chunk_review/chunk_review_labels.jsonl`` so the
    review page can show reviewers their earlier verdicts instead of a stale
    form. ``catalog`` narrows primary labels to records saved against that
    catalog, but same-paper labels from another catalog are returned separately
    as carryover labels. This keeps a git pull or regenerated catalog path from
    hiding review work while still making catalog mismatches explicit.
    """
    labels_path = Path(runs_root) / "chunk_review" / "chunk_review_labels.jsonl"
    rows = []
    carryover_rows = []
    for row in _read_jsonl(labels_path):
        if row.get("object_type") != "chunk_extraction_quality" or str(row.get("object_id")) != paper_id:
            continue
        label_catalog = str(row.get("source_artifact_path") or "")
        snapshot = row.get("correction_status_snapshot") or {}
        record = {
            "label_id": row.get("label_id"),
            "status": row.get("type_specific_status"),
            "verdict": row.get("verdict"),
            "comment": row.get("review_notes"),
            "reviewer": snapshot.get("reviewer"),
            "best_backend": snapshot.get("best_backend"),
            "extractor_run": snapshot.get("extractor_run"),
            "labeled_at": snapshot.get("labeled_at"),
            "catalog": label_catalog or None,
        }
        if catalog and label_catalog and not _same_catalog_path(label_catalog, catalog):
            record["catalog_mismatch"] = True
            carryover_rows.append(record)
        else:
            rows.append(record)
    rows.sort(key=lambda item: str(item.get("labeled_at") or ""), reverse=True)
    carryover_rows.sort(key=lambda item: str(item.get("labeled_at") or ""), reverse=True)
    return {
        "paper_id": paper_id,
        "label_count": len(rows),
        "labels": rows,
        "carryover_label_count": len(carryover_rows),
        "carryover_labels": carryover_rows,
        "labels_jsonl": str(labels_path),
    }


def discover_review_stores(runs_root: Path) -> list[dict[str, str]]:
    """List review-outcome sqlite stores (paper-experiment link labels)."""
    stores: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in _candidate_roots(runs_root, ""):
        if not root.exists():
            continue
        for pattern in ("review_outcomes.sqlite", "*/review_outcomes.sqlite", "review_outcomes/*.sqlite"):
            for path in sorted(root.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                stores.append({"path": str(resolved), "label": resolved.parent.name, "modified": resolved.stat().st_mtime})
    stores.sort(key=lambda item: item["modified"], reverse=True)
    return [{"path": item["path"], "label": item["label"]} for item in stores]


def paper_link_verdicts(store_path: str | Path) -> dict[str, dict[str, Any]]:
    """Aggregate paper-experiment-link review labels per paper from a store.

    Returns ``{paper_id: {"accepted": n, "rejected": n, "inconclusive": n,
    "total": n}}`` using the reviewer's type-specific status when present and
    the normalized verdict otherwise, so the chunk-review paper table can be
    filtered by link outcome.
    """
    from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore

    verdict_to_status = {"positive": "accepted", "negative": "rejected", "inconclusive": "inconclusive"}
    summary: dict[str, dict[str, Any]] = {}
    store = ReviewOutcomeSQLiteStore(store_path)
    for label in store.list_labels(object_type="paper_experiment_link"):
        status = label.type_specific_status or verdict_to_status.get(label.verdict) or label.verdict
        entry = summary.setdefault(label.object_id, {"total": 0})
        entry[status] = entry.get(status, 0) + 1
        entry["total"] += 1
    return summary


def _candidate_roots(runs_root: Path, subdir: str) -> list[Path]:
    roots = [Path(runs_root) / subdir, Path.cwd() / "local_runs" / subdir]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _same_catalog_path(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)
    except OSError:
        return str(Path(left).expanduser()) == str(Path(right).expanduser())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
