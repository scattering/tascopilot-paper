from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import DocumentExtractionResult, DocumentExtractorBackendReport, ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle


PDF_BACKENDS = {"pdftotext_baseline", "grobid", "docling", "auto"}
BASELINE_VERSION = "pdftotext_baseline_v0"


def extract_document(arguments: dict[str, Any]) -> dict[str, Any]:
    result = extract_document_file(
        arguments.get("path"),
        backend=str(arguments.get("backend") or "pdftotext_baseline"),
        max_pages=int(arguments.get("max_pages", 8)),
        max_chars=int(arguments.get("max_chars", 20000)),
        output_dir=arguments.get("output_dir"),
        write_artifacts=bool(arguments.get("write_artifacts", False)),
        grobid_url=arguments.get("grobid_url"),
    )
    return result.model_dump(mode="json")


def extract_document_file(
    path: str | Path | None,
    *,
    backend: str = "pdftotext_baseline",
    max_pages: int = 8,
    max_chars: int = 20000,
    output_dir: str | Path | None = None,
    write_artifacts: bool = False,
    grobid_url: str | None = None,
) -> DocumentExtractionResult:
    if not path:
        return _extraction_result("missing_input", "", backend, missing_fields=["path"], warnings=["Document path is required."])
    document_path = Path(path).expanduser()
    if not document_path.exists():
        return _extraction_result("missing_input", str(document_path), backend, missing_fields=["path"], warnings=[f"Document path does not exist: {document_path}"])
    if backend == "auto":
        backend = "docling" if _docling_available() and document_path.suffix.lower() == ".pdf" else "pdftotext_baseline"
    if backend not in PDF_BACKENDS:
        return _extraction_result("unsupported", str(document_path), backend, warnings=[f"Unsupported extractor backend: {backend}"])
    if backend == "grobid":
        return _extract_grobid(document_path, max_pages=max_pages, max_chars=max_chars, output_dir=output_dir, write_artifacts=write_artifacts, grobid_url=grobid_url)
    if backend == "docling":
        return _extract_docling(document_path, max_pages=max_pages, max_chars=max_chars, output_dir=output_dir, write_artifacts=write_artifacts)
    if document_path.suffix.lower() in {".txt", ".md"}:
        text = document_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return _success(document_path, backend, "safe_text_file_first_chars", text, max_pages, output_dir, write_artifacts)
    if document_path.suffix.lower() != ".pdf":
        return _extraction_result("unsupported", str(document_path), backend, warnings=[f"Unsupported document extension for baseline extractor: {document_path.suffix}"])
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return _extraction_result("unavailable", str(document_path), backend, warnings=["pdftotext is not available on PATH."])
    try:
        completed = subprocess.run(
            [pdftotext, "-enc", "UTF-8", "-layout", "-f", "1", "-l", str(max_pages), str(document_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as exc:  # pragma: no cover - platform dependent subprocess failure
        return _extraction_result("error", str(document_path), backend, warnings=[f"pdftotext failed: {exc}"])
    if completed.returncode != 0:
        return _extraction_result("error", str(document_path), backend, warnings=[f"pdftotext returned {completed.returncode}: {completed.stderr[:300]}"])
    return _success(document_path, backend, "pdftotext_first_pages", completed.stdout[:max_chars], max_pages, output_dir, write_artifacts)


def extractor_bakeoff(arguments: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(arguments["manifest"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    limit = int(arguments.get("limit", 10))
    backends = list(arguments.get("backends") or ["pdftotext_baseline", "grobid", "docling"])
    max_pages = int(arguments.get("max_pages", 8))
    max_chars = int(arguments.get("max_chars", 20000))
    grobid_url = arguments.get("grobid_url")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        report = {"workflow_name": "corpus-extractor-bakeoff", "status": "missing_input", "missing_fields": ["manifest"], "warnings": [f"Manifest does not exist: {manifest_path}"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_bakeoff_markdown, json_filename="extractor_bakeoff.json", markdown_filename="extractor_bakeoff.md")
        return report
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries = _manifest_document_paths(manifest, manifest_path)[:limit]
    reports: list[dict[str, Any]] = []
    for entry in entries:
        per_backend = []
        for backend in backends:
            result = extract_document_file(entry["path"], backend=backend, max_pages=max_pages, max_chars=max_chars, grobid_url=grobid_url)
            per_backend.append(
                DocumentExtractorBackendReport(
                    backend=backend,
                    status=result.status,
                    dependency_status="available" if result.status == "success" else ("unavailable" if result.status == "unavailable" else "not_applicable"),
                    extraction_quality=result.quality_metrics,
                    warnings=result.warnings,
                    provenance=result.provenance,
                ).model_dump(mode="json")
            )
        reports.append({"document_id": entry.get("document_id"), "path": entry["path"], "backends": per_backend})
    report = {
        "workflow_name": "corpus-extractor-bakeoff",
        "status": "ok",
        "manifest": str(manifest_path),
        "document_count": len(reports),
        "backends": backends,
        "reports": reports,
        "safe_conclusion": "pdftotext is a baseline extractor. GROBID/Docling are represented as optional boundaries and must be installed/configured before they can be compared on private PDFs.",
        "provenance": [ProvenanceRecord(source_type="corpus_manifest", source_id=str(manifest_path), method="extractor_bakeoff").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_bakeoff_markdown, json_filename="extractor_bakeoff.json", markdown_filename="extractor_bakeoff.md")
    return report


def extractor_quality_report(arguments: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(arguments["extractor_run_dir"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        report = {"workflow_name": "corpus-extractor-quality", "status": "missing_input", "missing_fields": ["extractor_run_dir"], "warnings": [f"Extractor run directory does not exist: {run_dir}"]}
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_quality_markdown, json_filename="extractor_quality_report.json", markdown_filename="extractor_quality_report.md")
        return report
    docling_rows = _load_json(run_dir / "outputs" / "docling" / "summary.json")
    grobid_rows = _load_json(run_dir / "outputs" / "grobid" / "summary.json")
    skipped_sidecar_rows = sum(1 for row in [*docling_rows, *grobid_rows] if _is_macos_sidecar_row(row))
    docling_rows = [row for row in docling_rows if not _is_macos_sidecar_row(row)]
    grobid_rows = [row for row in grobid_rows if not _is_macos_sidecar_row(row)]
    grobid_by_name = {str(row.get("pdf_name") or Path(str(row.get("pdf") or "")).name): row for row in grobid_rows}
    rows: list[dict[str, Any]] = []
    warnings = [
        "Quality report uses counts and structural diagnostics only; it does not write private article text into the report.",
        "Extractor quality is not source-support validation. Inspect high-value methods/caption chunks before scientific use.",
    ]
    if skipped_sidecar_rows:
        warnings.append(f"Skipped {skipped_sidecar_rows} macOS AppleDouble sidecar extractor rows.")
    for docling in docling_rows:
        pdf_name = str(docling.get("pdf_name") or Path(str(docling.get("pdf") or "")).name)
        grobid = grobid_by_name.get(pdf_name, {})
        markdown_path = _artifact_path(run_dir, docling, "markdown_path", "docling", "md")
        tei_path = _artifact_path(run_dir, grobid, "tei_path", "grobid", "tei")
        markdown_metrics = _text_quality_metrics(markdown_path.read_text(encoding="utf-8", errors="replace")) if markdown_path and markdown_path.exists() else {}
        tei_text = tei_path.read_text(encoding="utf-8", errors="replace") if tei_path and tei_path.exists() else ""
        tei_metrics = _tei_quality_metrics(tei_text) if tei_text else {}
        rows.append(
            {
                "pdf_name": pdf_name,
                "docling_status": docling.get("status"),
                "grobid_status": grobid.get("status"),
                "docling_markdown_chars": docling.get("markdown_chars") or markdown_metrics.get("char_count"),
                "docling_headings": markdown_metrics.get("markdown_heading_count", 0),
                "docling_figure_hits": markdown_metrics.get("figure_caption_hits", docling.get("figure_mentions", 0)),
                "docling_table_hits": markdown_metrics.get("table_mentions", docling.get("table_mentions", 0)),
                "docling_bt7_terms": markdown_metrics.get("bt7_terms", docling.get("bt7_terms", 0)),
                "docling_methods_terms": markdown_metrics.get("methods_terms", 0),
                "docling_references_heading": markdown_metrics.get("references_heading", False),
                "grobid_tei_bytes": grobid.get("tei_bytes") or len(tei_text.encode("utf-8")),
                "grobid_figure_tags": tei_metrics.get("figure_tags", grobid.get("figure_tags", 0)),
                "grobid_fig_desc_tags": tei_metrics.get("fig_desc_tags", 0),
                "grobid_bibl_tags": tei_metrics.get("bibl_tags", grobid.get("bibl_tags", 0)),
                "grobid_title_tags": tei_metrics.get("title_tags", grobid.get("title_tags", 0)),
                "grobid_abstract_present": tei_metrics.get("abstract_present", grobid.get("abstract_present", False)),
                "grobid_bt7_terms": tei_metrics.get("bt7_terms", grobid.get("bt7_terms", 0)),
                "grobid_methods_terms": tei_metrics.get("methods_terms", 0),
                "quality_flags": _quality_flags(docling, grobid, markdown_metrics, tei_metrics),
            }
        )
    summary = {
        "document_count": len(rows),
        "docling_success": sum(1 for row in rows if row.get("docling_status") == "success"),
        "grobid_success": sum(1 for row in rows if row.get("grobid_status") == "success"),
        "bt7_hit_documents": sum(1 for row in rows if int(row.get("docling_bt7_terms") or 0) > 0 or int(row.get("grobid_bt7_terms") or 0) > 0),
        "documents_with_docling_figure_hits": sum(1 for row in rows if int(row.get("docling_figure_hits") or 0) > 0),
        "documents_with_grobid_figure_tags": sum(1 for row in rows if int(row.get("grobid_figure_tags") or 0) > 0),
        "documents_with_methods_terms": sum(1 for row in rows if int(row.get("docling_methods_terms") or 0) > 0 or int(row.get("grobid_methods_terms") or 0) > 0),
    }
    bt7_rows = [row for row in rows if int(row.get("docling_bt7_terms") or 0) > 0 or int(row.get("grobid_bt7_terms") or 0) > 0]
    report = {
        "workflow_name": "corpus-extractor-quality",
        "status": "ok",
        "extractor_run_dir": str(run_dir),
        "summary": summary,
        "skipped_sidecar_rows": skipped_sidecar_rows,
        "bt7_hit_documents": bt7_rows,
        "rows": rows,
        "warnings": warnings,
        "safe_conclusion": "Use Docling Markdown for chunkable reading surfaces and GROBID TEI for article structure/citations; neither output is authoritative until retrieval and source-support evals pass.",
        "provenance": [ProvenanceRecord(source_type="extractor_run", source_id=str(run_dir), method="extractor_quality_report").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_quality_markdown, json_filename="extractor_quality_report.json", markdown_filename="extractor_quality_report.md")
    return report


def _success(
    path: Path,
    backend: str,
    method: str,
    text: str,
    max_pages: int,
    output_dir: str | Path | None,
    write_artifacts: bool,
) -> DocumentExtractionResult:
    chunk_candidates = _chunk_candidates(text)
    artifacts: list[str] = []
    if write_artifacts and output_dir:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / f"extraction_{_hash(str(path))}.json"
        summary_path.write_text(
            json.dumps({"path": str(path), "backend": backend, "chunk_candidates": chunk_candidates, "quality_metrics": _quality_metrics(text)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifacts.append(str(summary_path))
    warnings = []
    if backend == "pdftotext_baseline":
        warnings.append("pdftotext is a baseline text extractor; figure/caption structure may be incomplete compared with structured PDF parsers.")
    return DocumentExtractionResult(
        status="success",
        path=str(path),
        backend=backend,
        extraction_method=method,
        parser_version=BASELINE_VERSION,
        source_hash=_file_hash(path),
        text_char_count=len(text),
        page_count_requested=max_pages,
        chunk_candidates=chunk_candidates,
        quality_metrics=_quality_metrics(text),
        warnings=warnings,
        artifact_paths=artifacts,
        provenance=[ProvenanceRecord(source_type="document", source_id=str(path), method=method)],
    )


def _extract_docling(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
    output_dir: str | Path | None,
    write_artifacts: bool,
) -> DocumentExtractionResult:
    if path.suffix.lower() != ".pdf":
        return _extraction_result("unsupported", str(path), "docling", warnings=["Docling backend is only enabled for PDFs in the corpus pilot boundary."])
    if not _docling_available():
        return _extraction_result("unavailable", str(path), "docling", warnings=["Docling is not installed in the active environment; install/configure an optional extractor plugin before use."])
    with tempfile.TemporaryDirectory(prefix="tas_docling_") as temp_dir:
        text_path = Path(temp_dir) / "docling_text.md"
        script = """
from pathlib import Path
import sys
from docling.document_converter import DocumentConverter

path = sys.argv[1]
out = Path(sys.argv[2])
max_chars = int(sys.argv[3])
converted = DocumentConverter().convert(path)
document = converted.document
if hasattr(document, "export_to_markdown"):
    text = document.export_to_markdown()
elif hasattr(document, "export_to_text"):
    text = document.export_to_text()
else:
    text = str(document)
out.write_text(text[:max_chars], encoding="utf-8")
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, str(path), str(text_path), str(max_chars)],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return _extraction_result("error", str(path), "docling", warnings=["Docling extraction timed out in isolated subprocess."])
        except Exception as exc:  # pragma: no cover - optional dependency behavior varies by platform
            return _extraction_result("error", str(path), "docling", warnings=[f"Docling subprocess failed: {exc}"])
        if completed.returncode != 0:
            stderr = completed.stderr[-1200:].strip()
            return _extraction_result("error", str(path), "docling", warnings=[f"Docling subprocess returned {completed.returncode}: {stderr}"])
        if not text_path.exists():
            return _extraction_result("error", str(path), "docling", warnings=["Docling subprocess completed without writing extracted text."])
        text = text_path.read_text(encoding="utf-8", errors="replace")
    try:
        parser_version = _docling_version()
    except Exception:  # pragma: no cover - optional dependency behavior varies by version
        parser_version = "docling_subprocess"
    result = _success(path, "docling", "docling_document_converter_subprocess", text[:max_chars], max_pages, output_dir, write_artifacts)
    result.parser_version = parser_version
    result.warnings.append("Docling output is local derived text; validate figure captions, tables, and methods chunks before using as evidence.")
    return result


def _extract_grobid(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
    output_dir: str | Path | None,
    write_artifacts: bool,
    grobid_url: str | None,
) -> DocumentExtractionResult:
    if path.suffix.lower() != ".pdf":
        return _extraction_result("unsupported", str(path), "grobid", warnings=["GROBID backend accepts PDFs only."])
    endpoint = (grobid_url or os.environ.get("GROBID_URL") or "").rstrip("/")
    if not endpoint:
        return _extraction_result("unavailable", str(path), "grobid", warnings=["GROBID endpoint is not configured; set GROBID_URL or pass grobid_url before structured scholarly extraction."])
    boundary = f"----tascopilot-{_hash(str(path))}"
    body = _multipart_pdf_body(path, boundary)
    request = urllib.request.Request(
        f"{endpoint}/api/processFulltextDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/xml"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310 - endpoint is explicit local/user-configured extraction service
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return _extraction_result("unavailable", str(path), "grobid", warnings=[f"GROBID endpoint unavailable: {exc}"])
    except Exception as exc:  # pragma: no cover - service/network behavior
        return _extraction_result("error", str(path), "grobid", warnings=[f"GROBID extraction failed: {exc}"])
    text = _tei_to_text(payload)[:max_chars]
    result = _success(path, "grobid", "grobid_process_fulltext_document", text, max_pages, output_dir, write_artifacts)
    result.parser_version = "grobid_service"
    result.quality_metrics["tei_char_count"] = len(payload)
    result.warnings.append("GROBID TEI extraction requires local service configuration and corpus eval validation before use as authoritative evidence.")
    return result


def _docling_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("docling") is not None
    except Exception:  # pragma: no cover - defensive importlib fallback
        return False


def _docling_version() -> str:
    try:
        from importlib.metadata import version

        return f"docling_{version('docling')}"
    except Exception:  # pragma: no cover - optional dependency
        return "docling_unknown_version"


def _multipart_pdf_body(path: Path, boundary: str) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="input"; filename="{path.name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer


def _tei_to_text(payload: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", payload)
    without_entities = (
        without_tags.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )
    return re.sub(r"\s+", " ", without_entities).strip()


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _is_macos_sidecar_row(row: dict[str, Any]) -> bool:
    pdf_name = str(row.get("pdf_name") or Path(str(row.get("pdf") or "")).name)
    return Path(pdf_name).name.startswith("._")


def _artifact_path(run_dir: Path, row: dict[str, Any], key: str, backend: str, subdir: str) -> Path | None:
    if not row:
        return None
    raw = row.get(key)
    candidates = [Path(str(raw)).expanduser()] if raw else []
    if raw:
        candidates.append(run_dir / "outputs" / backend / subdir / Path(str(raw)).name)
    pdf_name = row.get("pdf_name")
    if pdf_name:
        suffix = ".md" if backend == "docling" else ".tei.xml"
        candidates.append(run_dir / "outputs" / backend / subdir / f"{Path(str(pdf_name)).stem}{suffix}")
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _text_quality_metrics(text: str) -> dict[str, Any]:
    base = _quality_metrics(text)
    lowered = text.lower()
    base.update(
        {
            "markdown_heading_count": len(re.findall(r"(?m)^#{1,6}\s+", text)),
            "table_mentions": len(re.findall(r"\btable\s+\d+", lowered)),
            "bt7_terms": len(re.findall(r"\bbt-?7\b", lowered)),
            "ncnr_terms": lowered.count("ncnr"),
            "methods_terms": sum(lowered.count(term) for term in ["method", "experimental", "experiment", "instrument", "measurement"]),
            "references_heading": bool(re.search(r"(?im)^#{0,6}\s*references\s*$", text)),
        }
    )
    return base


def _tei_quality_metrics(text: str) -> dict[str, Any]:
    lowered = text.lower()
    stripped = _tei_to_text(text).lower()
    return {
        "tei_char_count": len(text),
        "figure_tags": len(re.findall(r"<\s*figure\b", lowered)),
        "fig_desc_tags": len(re.findall(r"<\s*figdesc\b", lowered)),
        "bibl_tags": len(re.findall(r"<\s*biblstruct\b", lowered)),
        "title_tags": len(re.findall(r"<\s*title\b", lowered)),
        "abstract_present": bool(re.search(r"<\s*abstract\b", lowered)),
        "head_tags": len(re.findall(r"<\s*head\b", lowered)),
        "bt7_terms": len(re.findall(r"\bbt-?7\b", stripped)),
        "ncnr_terms": stripped.count("ncnr"),
        "methods_terms": sum(stripped.count(term) for term in ["method", "experimental", "experiment", "instrument", "measurement"]),
    }


def _quality_flags(docling: dict[str, Any], grobid: dict[str, Any], markdown_metrics: dict[str, Any], tei_metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if docling.get("status") != "success":
        flags.append("docling_not_success")
    if grobid.get("status") != "success":
        flags.append("grobid_not_success")
    if int(markdown_metrics.get("figure_caption_hits") or 0) == 0 and int(tei_metrics.get("figure_tags") or 0) > 0:
        flags.append("docling_no_figure_hits_but_grobid_has_figures")
    if int(markdown_metrics.get("methods_terms") or 0) == 0 and int(tei_metrics.get("methods_terms") or 0) > 0:
        flags.append("docling_no_methods_terms_but_grobid_has_methods_terms")
    if int(markdown_metrics.get("bt7_terms") or 0) != int(tei_metrics.get("bt7_terms") or 0):
        flags.append("bt7_term_count_disagrees")
    if int(markdown_metrics.get("markdown_heading_count") or 0) == 0:
        flags.append("docling_no_markdown_headings")
    return flags


def _extraction_result(
    status: str,
    path: str,
    backend: str,
    *,
    missing_fields: list[str] | None = None,
    warnings: list[str] | None = None,
) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        status=status,
        path=path,
        backend=backend,
        missing_fields=missing_fields or [],
        warnings=warnings or [],
        provenance=[ProvenanceRecord(source_type="document", source_id=path or "unknown", method=f"{backend}_extract_document")],
    )


def _manifest_document_paths(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in manifest.get("documents", []) or []:
        raw_path = item.get("local_path")
        if not raw_path or str(raw_path).startswith("["):
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            cwd_path = Path.cwd() / path
            path = cwd_path if cwd_path.exists() else manifest_path.parent / path
        entries.append({"document_id": str(item.get("document_id") or item.get("paper_id") or path.stem), "path": str(path)})
    for fixture in manifest.get("fixtures", []) or []:
        fixture_path = Path(str(fixture.get("path") or "")).expanduser()
        if not fixture_path.is_absolute():
            fixture_path = (manifest_path.parent / fixture_path).resolve()
        for child in sorted(fixture_path.rglob("*.txt")):
            entries.append({"document_id": child.stem, "path": str(child)})
        for child in sorted(fixture_path.rglob("*.md")):
            entries.append({"document_id": child.stem, "path": str(child)})
    return entries


def _chunk_candidates(text: str) -> list[dict[str, Any]]:
    candidates = []
    for chunk_type, terms in {
        "abstract": ["abstract"],
        "methods": ["methods", "experimental", "experiment"],
        "instrument_setup": ["BT-7", "BT7", "triple-axis", "triple axis"],
        "figure_caption": ["figure", "fig."],
        "data_reduction": ["background", "calibration", "normalization", "data reduction"],
    }.items():
        snippet, start = _term_snippet(text, terms)
        if snippet:
            candidates.append({"chunk_type": chunk_type, "char_start": start, "char_end": start + len(snippet), "snippet": snippet[:240]})
    return candidates


def _quality_metrics(text: str) -> dict[str, Any]:
    lowered = text.lower()
    caption_hits = len(re.findall(r"\b(?:fig(?:ure)?\.?)\s*\d+", lowered))
    return {
        "char_count": len(text),
        "line_count": len(text.splitlines()),
        "word_count": len(text.split()),
        "figure_caption_hits": caption_hits,
        "has_abstract": "abstract" in lowered,
        "has_methods_terms": any(term in lowered for term in ["experimental", "methods", "measurements"]),
        "has_bt7_terms": any(term in lowered for term in ["bt-7", "bt7"]),
        "reference_like_terms": lowered.count("references"),
    }


def _term_snippet(text: str, terms: list[str], *, window: int = 1400) -> tuple[str, int]:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if lowered.find(term.lower()) >= 0]
    if not positions:
        return "", -1
    center = min(positions)
    start = max(0, center - window // 5)
    end = min(len(text), start + window)
    return text[start:end].strip(), start


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _bakeoff_markdown(report: dict[str, Any]) -> str:
    lines = ["# Corpus Extractor Bakeoff", "", f"Status: `{report.get('status')}`", f"Documents: {report.get('document_count', 0)}", ""]
    for item in report.get("reports", []):
        lines.append(f"## {item.get('document_id')}")
        for backend in item.get("backends", []):
            metrics = backend.get("extraction_quality", {})
            lines.append(f"- `{backend.get('status')}` {backend.get('backend')}: {metrics.get('char_count', 0)} chars, {metrics.get('figure_caption_hits', 0)} figure hits")
    if report.get("safe_conclusion"):
        lines.extend(["", report["safe_conclusion"]])
    for warning in report.get("warnings", []):
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _quality_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Extractor Quality Report",
        "",
        f"Status: `{report.get('status')}`",
        f"Extractor run: `{report.get('extractor_run_dir')}`",
        f"Documents: {summary.get('document_count', 0)}",
        f"GROBID success: {summary.get('grobid_success', 0)}",
        f"Docling success: {summary.get('docling_success', 0)}",
        f"BT-7 hit documents: {summary.get('bt7_hit_documents', 0)}",
        f"Documents with Docling figure hits: {summary.get('documents_with_docling_figure_hits', 0)}",
        f"Documents with GROBID figure tags: {summary.get('documents_with_grobid_figure_tags', 0)}",
        "",
        "## BT-7 Hit Documents",
    ]
    for row in report.get("bt7_hit_documents", []):
        flags = ", ".join(row.get("quality_flags", [])) or "none"
        lines.append(
            f"- `{row.get('pdf_name')}`: docling BT-7={row.get('docling_bt7_terms')}, "
            f"grobid BT-7={row.get('grobid_bt7_terms')}, docling figures={row.get('docling_figure_hits')}, "
            f"grobid figures={row.get('grobid_figure_tags')}, flags={flags}"
        )
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    if report.get("safe_conclusion"):
        lines.extend(["", report["safe_conclusion"]])
    return "\n".join(lines) + "\n"
