"""Catalog and PDF integrity checks for the paper corpus.

Two defects found on the 48-paper gold set (2026-09-19) motivate this module:

* Many distinct paper records held the SAME source text. In the Box corpus 26
  PDFs are one file, 21 are another, and so on: a download step saved one
  document (a landing page, or the wrong open-access PDF) under many titles.
  The catalog then carried, for example, the CMS/LHCb Bs -> mu mu paper under
  "Topological RPdBi half-Heusler semimetals". Facts extracted from such a
  document are faithful to the text and wrong for the paper.
* The same paper was sometimes ingested twice under two ids.

`catalog_integrity_report` groups papers by the source hash of their chunks,
checks each record's title against its own body text, checks the file name the
PDF was saved as (the download target) against that title, and labels every
paper `ok`, `duplicate_ingest` (same content, same title) or `content_mismatch`
(same content under different titles, a title absent from the body, or a file
saved under another paper's name: a wrong download with unique content, which
the hash check cannot see. 2026-10-16: eight such papers sat in the gold set).
`apply_catalog_quarantine` stamps the affected facts and papers in place,
keeping every fact id, so existing labels still resolve.

`pdf_integrity_report` does the same at the file level from a manifest with
sha256 values, listing the DOIs whose PDFs must be fetched again.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_STOP = {"the", "of", "in", "and", "a", "an", "on", "for", "to", "with", "by", "at", "from", "via", "its", "is", "are", "as", "new", "study", "system", "systems"}


_MATHML_DEBRIS = re.compile(r"mml\w*|xmlns\w*|http\w*|w3org\w*|mathvariant\w*|lttitlegt|ltmiggt")


def _tokens(text: str) -> set[str]:
    """Content words. Citation titles that carried MathML arrive as
    'Magnetic ground state of the Kitaev mmlmath xmlnsmmlhttpww...'; the
    debris is dropped so the remaining words can still identify the paper."""
    words = re.findall(r"[a-z0-9]+", _MATHML_DEBRIS.sub(" ", str(text or "").lower()))
    return {w for w in words if len(w) >= 3 and w not in _STOP}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def same_title(a: str, b: str, *, threshold: float = 0.8) -> bool:
    """Content-word containment: the smaller title's words are mostly present in
    the other. Tolerant of a trailing 'copy', a subtitle, or OCR debris; two
    unrelated titles share almost nothing."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ta == tb
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def _distinct_titles(titles: list[str]) -> list[str]:
    kept: list[str] = []
    for t in titles:
        if not any(same_title(t, k) for k in kept):
            kept.append(t)
    return kept


def title_body_overlap(title: str, body: str) -> float:
    """Fraction of the title's content words that occur in the body text."""
    t = _tokens(title)
    if not t:
        return 1.0
    b = _tokens(body)
    return len(t & b) / len(t)


def download_name_overlap(pdf_name: str, title: str) -> float | None:
    """Overlap between the words of the file the PDF was saved as (the download
    target, i.e. the paper the corpus meant to hold) and the title read from
    the file's own text. Near 1 for a correct download; near 0 when the
    downloader's title-search fallback saved some other paper under this name.
    None when either side has no content words."""
    stem = re.sub(r"\.pdf$", "", pdf_name or "", flags=re.IGNORECASE).replace("_", " ")
    n, t = _tokens(stem), _tokens(title)
    if not n or not t:
        return None
    return len(n & t) / min(len(n), len(t))


def catalog_integrity_report(catalog_dir: str | Path, *, min_title_overlap: float = 0.34, min_download_name_overlap: float = 0.3) -> dict[str, Any]:
    catalog = Path(catalog_dir).expanduser()
    papers = {str(p.get("paper_id")): p for p in _load_jsonl(catalog / "papers.jsonl") if p.get("paper_id")}
    chunks = _load_jsonl(catalog / "source_chunks.jsonl")
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hash_to_papers: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        pid = str(chunk.get("paper_id") or "")
        by_paper[pid].append(chunk)
        if chunk.get("source_hash"):
            hash_to_papers[str(chunk["source_hash"])].add(pid)
    documents: dict[str, dict[str, Any]] = {}
    for pid, paper in papers.items():
        title = str(paper.get("title") or "")
        pchunks = by_paper.get(pid, [])
        body = " ".join(str(c.get("text") or "") for c in pchunks if str(c.get("chunk_type") or "") != "title")
        overlap = title_body_overlap(title, body) if body else 0.0
        hashes = {str(c.get("source_hash")) for c in pchunks if c.get("source_hash")}
        sharing = {other for h in hashes for other in hash_to_papers.get(h, set())} - {pid}
        same_title_ids = {o for o in sharing if same_title(str(papers.get(o, {}).get("title", "")), title)}
        other_title = sharing - same_title_ids
        pdf_name = str((paper.get("metadata") or {}).get("pdf_name") or "")
        name_overlap = download_name_overlap(pdf_name, title)
        if not pchunks:
            status, reason = "no_chunks", "paper has no source chunks in this catalog"
        elif re.match(r"\s*supplementa(l|ry)\b", title, re.IGNORECASE) and not other_title:
            status, reason = "supplementary", "supplementary material, not a primary paper; excluded from evaluation but not a bad download"
        elif other_title:
            status = "content_mismatch"
            reason = f"source text is shared with {len(other_title)} paper(s) with different titles; title-body overlap {overlap:.2f}"
        elif overlap < min_title_overlap:
            status = "content_mismatch"
            reason = f"title words are mostly absent from the body (overlap {overlap:.2f})"
        elif name_overlap is not None and name_overlap < min_download_name_overlap:
            # A wrong download with unique content: nothing shares its hash and
            # its own title is in its own body, but the file was saved under
            # another paper's name, so the record is not the paper it claims.
            status = "content_mismatch"
            reason = f"file saved under a different paper's name ({pdf_name!r}; name-title overlap {name_overlap:.2f}): wrong download"
        elif same_title_ids:
            status, reason = "duplicate_ingest", f"same content and title as {sorted(same_title_ids)}"
        else:
            status, reason = "ok", ""
        documents[pid] = {
            "paper_id": pid,
            "title": title,
            "integrity_status": status,
            "reason": reason,
            "title_body_overlap": round(overlap, 3),
            "chunk_count": len(pchunks),
            "shared_content_with": sorted(sharing),
            "pdf_name": pdf_name or None,
            "download_name_overlap": None if name_overlap is None else round(name_overlap, 3),
            "doi": paper.get("doi"),
        }
    counts: dict[str, int] = defaultdict(int)
    for d in documents.values():
        counts[d["integrity_status"]] += 1
    report = {
        "status": "ok",
        "catalog": str(catalog),
        "paper_count": len(papers),
        "status_counts": dict(sorted(counts.items())),
        "quarantine_paper_ids": sorted(pid for pid, d in documents.items() if d["integrity_status"] in ("content_mismatch", "supplementary")),
        "content_mismatch_paper_ids": sorted(pid for pid, d in documents.items() if d["integrity_status"] == "content_mismatch"),
        "duplicate_ingest_paper_ids": sorted(pid for pid, d in documents.items() if d["integrity_status"] == "duplicate_ingest"),
        "documents": documents,
    }
    return report


def apply_catalog_quarantine(catalog_dir: str | Path, *, report: dict[str, Any] | None = None, fact_files: tuple[str, ...] = ("paper_facts.jsonl", "paper_facts.all.jsonl")) -> dict[str, Any]:
    """Stamp facts of content-mismatched papers in place (ids kept): warning,
    human_review_required, metadata.integrity_status; stamp papers.jsonl too."""
    catalog = Path(catalog_dir).expanduser()
    report = report or catalog_integrity_report(catalog)
    bad = set(report["quarantine_paper_ids"])
    dup = set(report["duplicate_ingest_paper_ids"])
    touched = 0
    for name in fact_files:
        path = catalog / name
        rows = _load_jsonl(path)
        if not rows:
            continue
        for fact in rows:
            pid = str(fact.get("paper_id"))
            meta = dict(fact.get("metadata") or {})
            warnings = [w for w in (fact.get("warnings") or []) if not str(w).startswith("Catalog integrity:")]
            meta.pop("integrity_status", None); meta.pop("integrity_reason", None)  # re-applying must not leave stale stamps
            status = (report["documents"].get(pid) or {}).get("integrity_status")
            if pid in bad:
                meta["integrity_status"] = status
                meta["integrity_reason"] = report["documents"][pid]["reason"]
                fact["human_review_required"] = True
                warnings.append(
                    "Catalog integrity: the cited document's text does not belong to this paper (content_mismatch); fact excluded from linkage and evaluation until the PDF is re-fetched."
                    if status == "content_mismatch" else "Catalog integrity: supplementary material, excluded from the evaluation set."
                )
                touched += 1
            elif pid in dup:
                meta["integrity_status"] = "duplicate_ingest"
            fact["warnings"] = warnings
            fact["metadata"] = meta
        _write_jsonl(path, rows)
    papers = _load_jsonl(catalog / "papers.jsonl")
    for paper in papers:
        pid = str(paper.get("paper_id"))
        if pid in report["documents"]:
            paper["integrity_status"] = report["documents"][pid]["integrity_status"]
    _write_jsonl(catalog / "papers.jsonl", papers)
    (catalog / "catalog_integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (catalog / "quarantined_paper_ids.txt").write_text("".join(pid + "\n" for pid in report["quarantine_paper_ids"]), encoding="utf-8")
    return {"status": "ok", "catalog": str(catalog), "facts_stamped": touched, "quarantined_papers": len(bad), "duplicate_ingest_papers": len(dup), "report": str(catalog / "catalog_integrity.json")}


def pdf_integrity_report(pdf_root: str | Path, *, manifest_csv: str | Path | None = None, first_page_text: dict[str, str] | None = None) -> dict[str, Any]:
    """File-level check of a PDF corpus. Groups PDFs by content hash: one file
    stored under several different names is a bad download (a landing page or
    the wrong open-access PDF fetched for many DOIs). If first-page text is
    supplied, a PDF whose first page does not contain its own title is flagged
    too. Titles come from the manifest when a row names the file, else from the
    file name. DOIs for re-fetching are taken from the manifest."""
    import hashlib

    root = Path(pdf_root).expanduser()
    rows = list(csv.DictReader(Path(manifest_csv).expanduser().open(encoding="utf-8"))) if manifest_csv else []
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = Path(str(row.get("downloaded_pdf_path") or "")).name
        if name:
            by_name[name] = row
    # Match on the title and, when present, the raw citation text: the title
    # column is what got garbled, the citation usually still reads.
    by_title_tokens = {frozenset(_tokens(str(r.get("title") or "") + " " + str(r.get("bt7_citation_raw") or ""))): r for r in rows}
    files = sorted(root.rglob("*.pdf"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for pdf in files:
        groups[hashlib.sha256(pdf.read_bytes()).hexdigest()].append(pdf)

    def title_for(pdf: Path) -> str:
        row = by_name.get(pdf.name)
        return str(row.get("title")) if row and row.get("title") else re.sub(r"[_-]+", " ", pdf.stem)

    def doi_for(pdf: Path) -> str | None:
        row = by_name.get(pdf.name)
        if row and row.get("doi"):
            return str(row["doi"])
        # Some downloads were named by DOI with '/' replaced: 10.1103_physrevb.106.214438
        m = re.match(r"^(10\.\d{4,9})[ _/](\S+)$", pdf.stem)
        if m:
            return f"{m.group(1)}/{m.group(2).replace('_', '/')}"
        toks = frozenset(_tokens(title_for(pdf)))
        if not toks:
            return None
        best = max(by_title_tokens.items(), key=lambda kv: len(kv[0] & toks) / len(toks), default=None)
        if best and len(best[0] & toks) / len(toks) >= 0.7:
            return str(best[1].get("doi") or "") or None
        return None

    bad: list[dict[str, Any]] = []
    flagged: set[Path] = set()
    for sha, members in groups.items():
        if len(members) > 1 and len(_distinct_titles([title_for(p) for p in members])) > 1:
            for p in members:
                bad.append({"pdf": str(p), "title": title_for(p), "doi": doi_for(p), "sha256": sha, "shared_by": len(members), "reason": "same file stored under different titles"})
                flagged.add(p)
    if first_page_text:
        for p in files:
            if p in flagged:
                continue
            text = first_page_text.get(p.name)
            if text is not None and title_body_overlap(title_for(p), text) < 0.25:
                bad.append({"pdf": str(p), "title": title_for(p), "doi": doi_for(p), "sha256": None, "shared_by": 1, "reason": "first page does not contain the title"})
    return {"status": "ok", "pdf_root": str(root), "manifest": str(manifest_csv) if manifest_csv else None, "pdf_count": len(files), "content_groups": sum(1 for m in groups.values() if len(m) > 1),
            "bad_download_count": len(bad), "bad_downloads": bad, "refetch_dois": sorted({b["doi"] for b in bad if b.get("doi")}), "unresolved_titles": sorted({b["title"] for b in bad if not b.get("doi")})}


__all__ = ["catalog_integrity_report", "apply_catalog_quarantine", "pdf_integrity_report", "title_body_overlap"]
