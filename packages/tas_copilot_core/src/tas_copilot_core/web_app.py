from __future__ import annotations

import html
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tas_copilot_contracts import CopilotAskResponse
from tas_copilot_core.alignment_dashboard import alignment_dashboard_report, parse_cif_text, vector_tools_report
from tas_copilot_core.corpus.paper_facts import _paper_fact_review_html, build_paper_fact_review, discover_fact_catalogs, paper_fact_table
from tas_copilot_core.corpus.chunk_review import (
    discover_chunk_catalogs,
    discover_extractor_runs,
    discover_review_stores,
    extractor_outputs,
    paper_chunk_detail,
    paper_chunk_labels,
    paper_link_verdicts,
    paper_table,
    record_chunk_review_label,
)
from tas_copilot_core.registry.tool_inventory import build_tool_inventory, doctor_report
from tas_copilot_core.review_labels import ingest_review_labels, review_label_summary
from tas_copilot_core.workflows.copilot_router import copilot_ask, copilot_intent_examples

try:  # pragma: no cover - exercised in web-extra tests when installed.
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover - fail-closed optional dependency path.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    PlainTextResponse = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]


_RUN_RECORD_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
_RUN_RECORD_CACHE_SECONDS = 30.0


class WebAskRequest(BaseModel):
    question: str = ""
    # Optional explicit intent from a typed form (workspace/ask pages). Mirrors
    # the MCP gateway: the caller names the intent, the deterministic router
    # still applies the live-control block and required-input validation, and
    # an empty question defaults to the intent's canonical example question.
    intent: str | None = None
    output_dir: str | None = None
    data_root: str | None = None
    corpus_catalog: str | None = None
    data_catalog: str | None = None
    precedent_index: str | None = None
    paper_id: str | None = None
    figure_id: str | None = None
    fixture: str | None = None
    proposal_evidence_dir: str | None = None
    actual_evidence_dir: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    max_files: int = Field(default=1000, ge=0)


class WebAskResponse(BaseModel):
    result: CopilotAskResponse
    equivalent_cli: str
    # Generated artifact paths under local_runs, converted to /local-runs
    # links (HTML first) so pages can offer "view result" without re-deriving
    # paths client-side.
    artifact_links: list[dict[str, str]] = Field(default_factory=list)


class ReviewLabelSummaryRequest(BaseModel):
    store: str
    sample_limit: int = Field(default=10, ge=1, le=200)


class ReviewLabelIngestRequest(BaseModel):
    labels: str
    store: str
    output_dir: str | None = None


class AlignmentCifUploadRequest(BaseModel):
    filename: str
    content: str = Field(max_length=2_000_000)


class AlignmentReportRequest(BaseModel):
    cif_content: str = Field(max_length=2_000_000)
    orient1: list[float] = Field(default=[1, 0, 0], min_length=3, max_length=3)
    orient2: list[float] = Field(default=[0, 1, 0], min_length=3, max_length=3)
    energy_mev: float = Field(default=14.7, gt=0)
    a4_max_deg: float = Field(default=120.0, gt=0, le=180)
    top_k: int = Field(default=15, ge=1, le=100)
    environment_materials: list[str] = Field(default_factory=list)


class AlignmentVectorToolsRequest(BaseModel):
    cif_content: str = Field(max_length=2_000_000)
    orient1: list[float] = Field(default=[1, 0, 0], min_length=3, max_length=3)
    orient2: list[float] = Field(default=[0, 1, 0], min_length=3, max_length=3)
    v1: list[float] = Field(min_length=3, max_length=3)
    v2: list[float] | None = Field(default=None, min_length=3, max_length=3)


class ChunkReviewLabelRequest(BaseModel):
    paper_id: str
    status: str
    catalog: str | None = None
    chunk_id: str | None = None
    extractor_run: str | None = None
    backend: str | None = None
    best_backend: str = "not_compared"
    comment: str = ""
    reviewer: str = ""


def create_app(*, local_runs_dir: str | Path = "local_runs", docs_dir: str | Path = "docs") -> Any:
    """Create the local-only TASCopilot web shell.

    The web layer is intentionally thin: it renders discovery/review surfaces and
    calls the same read-only/planning workflow functions used by the CLI.
    """

    if FastAPI is None:
        raise RuntimeError("FastAPI web extra is not installed. Install tas-copilot-core[web] or install fastapi and uvicorn.")

    runs_root = Path(local_runs_dir).expanduser().resolve()
    docs_root = _resolve_docs_root(docs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="TASCopilot Local Workbench", docs_url="/api-docs", redoc_url="/api-redoc")
    app.mount("/local-runs", StaticFiles(directory=str(runs_root), html=True), name="local-runs")
    if docs_root.exists():
        app.mount("/docs", StaticFiles(directory=str(docs_root), html=True), name="docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page(
            "TASCopilot Workbench",
            _landing_body(),
        )

    @app.get("/ask", response_class=HTMLResponse)
    def ask_page() -> str:
        return _page("Ask TASCopilot", _ask_body())

    @app.post("/api/ask", response_model=WebAskResponse)
    async def ask_api(payload: WebAskRequest) -> WebAskResponse:
        question = payload.question.strip()
        intent = (payload.intent or "").strip() or None
        if intent:
            example = next((item for item in copilot_intent_examples() if item["intent"] == intent), None)
            if example is None:
                known = ", ".join(sorted(str(item["intent"]) for item in copilot_intent_examples()))
                raise HTTPException(status_code=400, detail=f"unknown intent: {intent}; known intents: {known}")
            if not question:
                question = str(example["example_question"])
        if not question:
            raise HTTPException(status_code=400, detail="question or intent is required")
        output_dir = _validated_output_dir(
            payload.output_dir,
            runs_root=runs_root,
            field_name="output_dir",
            default_prefix=f"web_{intent}" if intent else "web_copilot_ask",
        )
        request = {
            "question": question,
            "intent": intent,
            "output_dir": str(output_dir),
            "data_root": payload.data_root or None,
            "corpus_catalog": payload.corpus_catalog or None,
            "data_catalog": payload.data_catalog or None,
            "precedent_index": payload.precedent_index or None,
            "paper_id": payload.paper_id or None,
            "figure_id": payload.figure_id or None,
            "fixture": payload.fixture or None,
            "proposal_evidence_dir": payload.proposal_evidence_dir or None,
            "actual_evidence_dir": payload.actual_evidence_dir or None,
            "top_k": payload.top_k,
            "max_files": payload.max_files,
        }
        result = CopilotAskResponse.model_validate(copilot_ask(request))
        response = WebAskResponse(
            result=result,
            equivalent_cli=_copilot_cli(request),
            artifact_links=_artifact_links(result, runs_root),
        )
        _append_web_audit(
            runs_root,
            {
                "action": "ask",
                "question": question,
                "status": result.status,
                "intent": result.route.intent,
                "intent_requested": intent,
                "data_root": payload.data_root or None,
                "output_dir": str(output_dir),
                "equivalent_cli": response.equivalent_cli,
            },
        )
        return response

    @app.get("/api/intents")
    def intents_api() -> JSONResponse:
        return JSONResponse({"intents": copilot_intent_examples()})

    @app.get("/tools", response_class=HTMLResponse)
    def tools_page() -> str:
        report = doctor_report(build_tool_inventory())
        return _page("Tools And Capabilities", f"<pre>{html.escape(report)}</pre>")

    @app.get("/documentation", response_class=HTMLResponse)
    def documentation_page() -> str:
        return _page("Documentation", _documentation_body(docs_root))

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page() -> str:
        return _page("Local Runs", _runs_body(runs_root))

    @app.get("/api/runs")
    def runs_api() -> JSONResponse:
        return JSONResponse({"runs_root": str(runs_root), "runs": _run_records(runs_root)})

    @app.get("/papers/{paper_id}")
    def open_paper(paper_id: str, catalog: str | None = None, fetch: bool = True) -> Any:
        record = _resolve_paper_document(paper_id, runs_root=runs_root, catalog=catalog)
        if not record:
            raise HTTPException(status_code=404, detail=f"paper_id not found in local corpus catalogs: {paper_id}")
        cached = _ensure_paper_cached(record, runs_root=runs_root, fetch=fetch)
        if cached["status"] != "ok":
            raise HTTPException(status_code=404, detail=cached)
        path = Path(str(cached["path"]))
        media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
        _append_web_audit(
            runs_root,
            {
                "action": "open-paper",
                "paper_id": paper_id,
                "cache_status": cached.get("cache_status"),
                "source_path": record.get("local_path"),
                "served_path": str(path),
            },
        )
        return FileResponse(str(path), media_type=media_type, filename=path.name)

    @app.get("/api/papers/{paper_id}")
    def paper_info(paper_id: str, catalog: str | None = None, fetch: bool = False) -> JSONResponse:
        record = _resolve_paper_document(paper_id, runs_root=runs_root, catalog=catalog)
        if not record:
            raise HTTPException(status_code=404, detail=f"paper_id not found in local corpus catalogs: {paper_id}")
        cache = _ensure_paper_cached(record, runs_root=runs_root, fetch=fetch) if fetch else _paper_cache_probe(record, runs_root=runs_root)
        return JSONResponse({"paper": _public_paper_record(record), "cache": cache})

    @app.get("/labels", response_class=HTMLResponse)
    def labels_page(store: str | None = None) -> str:
        return _page("Review Labels", _labels_body(runs_root, selected_store=store))

    @app.get("/review", response_class=HTMLResponse)
    def review_page(store: str | None = None) -> str:
        return _page("Review Dashboard", _review_dashboard_body(runs_root, selected_store=store))

    @app.get("/labeling", response_class=HTMLResponse)
    def labeling_page(store: str | None = None) -> str:
        return _page("Review Dashboard", _review_dashboard_body(runs_root, selected_store=store))

    @app.get("/api/review-labels/stores")
    def label_stores_api() -> JSONResponse:
        return JSONResponse({"stores": _discover_review_stores(runs_root)})

    @app.post("/api/review-labels/summary")
    async def label_summary_api(payload: ReviewLabelSummaryRequest) -> JSONResponse:
        result = review_label_summary({"store": payload.store, "sample_limit": payload.sample_limit})
        _append_web_audit(runs_root, {"action": "review-labels-summary", "store": payload.store, "status": result.get("status")})
        return JSONResponse(result)

    @app.post("/api/review-labels/ingest")
    async def label_ingest_api(payload: ReviewLabelIngestRequest) -> JSONResponse:
        output_dir = _validated_output_dir(
            payload.output_dir,
            runs_root=runs_root,
            field_name="output_dir",
            default_prefix="review_label_ingest",
        )
        result = ingest_review_labels({"labels": payload.labels, "store": payload.store, "output_dir": str(output_dir)})
        _append_web_audit(
            runs_root,
            {
                "action": "review-labels-ingest",
                "labels": payload.labels,
                "store": payload.store,
                "status": result.get("status"),
                "ingested_label_count": result.get("ingested_label_count"),
                "output_dir": str(output_dir),
            },
        )
        return JSONResponse(result)

    @app.get("/chunks", response_class=HTMLResponse)
    def chunks_page() -> str:
        return _page("Chunk Review", _chunks_body())

    @app.get("/api/chunks/sources")
    def chunk_sources_api() -> JSONResponse:
        return JSONResponse(
            {
                "catalogs": discover_chunk_catalogs(runs_root),
                "extractor_runs": discover_extractor_runs(runs_root),
                "review_stores": discover_review_stores(runs_root),
            }
        )

    @app.get("/api/chunks/papers")
    def chunk_papers_api(catalog: str, link_store: str | None = None) -> JSONResponse:
        catalog_dir = _validated_chunk_source(catalog, runs_root=runs_root, required_file="source_chunks.jsonl")
        table = paper_table(catalog_dir)
        if link_store:
            known = {store["path"] for store in discover_review_stores(runs_root)}
            store_path = str(Path(link_store).expanduser().resolve())
            if store_path not in known:
                raise HTTPException(status_code=400, detail=f"unknown review store: {link_store}")
            verdicts = paper_link_verdicts(store_path)
            for row in table.get("papers") or []:
                row["link_verdicts"] = verdicts.get(row["paper_id"])
        return JSONResponse(table)

    @app.get("/api/chunks/labels")
    def chunk_labels_api(paper_id: str, catalog: str | None = None) -> JSONResponse:
        return JSONResponse(paper_chunk_labels(runs_root, paper_id, catalog))

    @app.get("/api/chunks/paper/{paper_id}")
    def chunk_paper_api(paper_id: str, catalog: str, run: str | None = None) -> JSONResponse:
        catalog_dir = _validated_chunk_source(catalog, runs_root=runs_root, required_file="source_chunks.jsonl")
        detail = paper_chunk_detail(catalog_dir, paper_id)
        if run and detail.get("pdf_name"):
            run_dir = _validated_chunk_source(run, runs_root=runs_root, required_glob="outputs/*/summary.json")
            detail["extractor_outputs"] = extractor_outputs(run_dir, str(detail["pdf_name"]), derive_chunks=True, paper=detail)
        elif run:
            detail["extractor_outputs"] = {"status": "no_pdf_name_for_paper"}
        _append_web_audit(runs_root, {"action": "chunk-review-open-paper", "paper_id": paper_id, "catalog": str(catalog_dir)})
        return JSONResponse(detail)

    @app.post("/api/chunks/labels")
    async def chunk_label_api(payload: ChunkReviewLabelRequest) -> JSONResponse:
        result = record_chunk_review_label(runs_root, payload.model_dump())
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        _append_web_audit(
            runs_root,
            {
                "action": "chunk-review-label",
                "paper_id": payload.paper_id,
                "label_status": payload.status,
                "label_id": result.get("label_id"),
                "labels_jsonl": result.get("labels_jsonl"),
            },
        )
        return JSONResponse(result)

    @app.get("/facts", response_class=HTMLResponse)
    def facts_page() -> str:
        return _page("Paper Fact Review", _facts_body())

    @app.get("/api/facts/catalogs")
    def fact_catalogs_api() -> JSONResponse:
        return JSONResponse({"catalogs": discover_fact_catalogs(runs_root)})

    @app.get("/api/facts/papers")
    def fact_papers_api(catalog: str) -> JSONResponse:
        catalog_dir = _validated_fact_catalog(catalog, runs_root=runs_root)
        return JSONResponse(paper_fact_table(catalog_dir))

    @app.get("/facts/review", response_class=HTMLResponse)
    def fact_review_page(catalog: str, paper_id: str | None = None) -> str:
        catalog_dir = _validated_fact_catalog(catalog, runs_root=runs_root)
        report = build_paper_fact_review(catalog_dir, paper_ids={paper_id} if paper_id else None)
        _append_web_audit(runs_root, {"action": "paper-fact-review-open", "catalog": str(catalog_dir), "paper_id": paper_id})
        return _paper_fact_review_html(report)

    @app.get("/experiments", response_class=HTMLResponse)
    def experiments_page() -> str:
        return _page("Experiment Reports", _experiments_body())

    @app.get("/workspace", response_class=HTMLResponse)
    def workspace_page() -> str:
        return _page("Experiment Workspace", _workspace_body())

    @app.get("/api/workspace/data-roots")
    def workspace_data_roots_api() -> JSONResponse:
        return JSONResponse({"data_roots": _recent_data_roots(runs_root)})

    @app.get("/api/experiments/reports")
    def experiment_reports_api() -> JSONResponse:
        return JSONResponse({"runs_root": str(runs_root), "reports": _discover_experiment_reports(runs_root)})

    _register_alignment_routes(app, runs_root)

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    return app


def create_alignment_app(*, local_runs_dir: str | Path = "local_runs", root_path: str = "") -> Any:
    """Standalone alignment dashboard for public deployment.

    Exposes ONLY the CIF-based alignment surface: the page, CIF upload, the
    structure-factor/zone report, vector tools, and a health check. It
    deliberately has no static mounts (no /local-runs exposure), no corpus,
    paper, chunk-review, label, or copilot-ask routes, and no paper fetcher:
    none of the private-corpus surfaces exist in this app. Bind it to
    localhost behind a reverse proxy; it has no authentication layer of its
    own.
    """
    if FastAPI is None:
        raise RuntimeError("FastAPI web extra is not installed. Install tas-copilot-core[web] or install fastapi and uvicorn.")
    runs_root = Path(local_runs_dir).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="TAS Alignment Dashboard", root_path=root_path, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def alignment_index() -> str:
        return _page("Alignment Dashboard", _alignment_body(), nav=False)

    _register_alignment_routes(app, runs_root)

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    return app


def _register_alignment_routes(app: Any, runs_root: Path) -> None:
    # CIF content stays in memory end to end: the browser holds the text and
    # resends it per request; nothing user-supplied is written to disk. The
    # audit log records only a content hash and parse status.
    def _cif_digest(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]

    @app.get("/alignment", response_class=HTMLResponse)
    def alignment_page() -> str:
        return _page("Alignment Dashboard", _alignment_body())

    @app.post("/api/alignment/upload-cif")
    async def alignment_upload_cif(payload: AlignmentCifUploadRequest) -> JSONResponse:
        structure = parse_cif_text(payload.content)
        _append_web_audit(
            runs_root,
            {
                "action": "alignment-parse-cif",
                "cif_sha256_12": _cif_digest(payload.content),
                "parse_status": structure.get("status"),
                "formula": structure.get("formula"),
            },
        )
        return JSONResponse({"structure": structure})

    @app.post("/api/alignment/report")
    async def alignment_report_api(payload: AlignmentReportRequest) -> JSONResponse:
        report = alignment_dashboard_report(
            {
                "cif_content": payload.cif_content,
                "orient1": payload.orient1,
                "orient2": payload.orient2,
                "energy_mev": payload.energy_mev,
                "a4_max_deg": payload.a4_max_deg,
                "top_k": payload.top_k,
                "environment_materials": payload.environment_materials,
            }
        )
        _append_web_audit(
            runs_root,
            {
                "action": "alignment-report",
                "cif_sha256_12": _cif_digest(payload.cif_content),
                "status": report.get("status"),
                "scattering_plane": report.get("scattering_plane"),
                "suggested_peak_count": len(report.get("suggested_peaks") or []),
            },
        )
        return JSONResponse(report)

    @app.post("/api/alignment/vector-tools")
    async def alignment_vector_tools_api(payload: AlignmentVectorToolsRequest) -> JSONResponse:
        report = vector_tools_report(
            {
                "cif_content": payload.cif_content,
                "orient1": payload.orient1,
                "orient2": payload.orient2,
                "v1": payload.v1,
                "v2": payload.v2,
            }
        )
        _append_web_audit(runs_root, {"action": "alignment-vector-tools", "cif_sha256_12": _cif_digest(payload.cif_content), "status": report.get("status"), "v1": payload.v1, "v2": payload.v2})
        return JSONResponse(report)


def _validated_chunk_source(value: str, *, runs_root: Path, required_file: str | None = None, required_glob: str | None = None) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"directory not found: {candidate}")
    if required_file and not (candidate / required_file).exists():
        raise HTTPException(status_code=400, detail=f"directory is missing {required_file}: {candidate}")
    if required_glob and not any(candidate.glob(required_glob)):
        raise HTTPException(status_code=400, detail=f"directory does not match an extractor run layout: {candidate}")
    return candidate


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    local_runs_dir: str | Path = "local_runs",
    docs_dir: str | Path = "docs",
    app: str = "workbench",
    root_path: str = "",
) -> None:
    _ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Unsupported host {host!r}. Use 127.0.0.1/localhost for local-only "
            "execution; this server has no network authentication layer. For "
            "public deployment, keep the bind local and put a reverse proxy in front."
        )
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("uvicorn is required for `serve`. Install tas-copilot-core[web] or install uvicorn.") from exc
    if app == "alignment":
        application = create_alignment_app(local_runs_dir=local_runs_dir, root_path=root_path)
    elif app == "workbench":
        application = create_app(local_runs_dir=local_runs_dir, docs_dir=docs_dir)
    else:
        raise ValueError(f"Unknown app {app!r}; expected 'workbench' or 'alignment'.")
    uvicorn.run(application, host=host, port=port)


def _resolve_docs_root(docs_dir: str | Path) -> Path:
    candidate = Path(docs_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    for parent in Path(__file__).resolve().parents:
        repo_candidate = (parent / candidate).resolve()
        if repo_candidate.exists():
            return repo_candidate
    return cwd_candidate


_EXPERIMENT_REPORT_KINDS = (
    ("day_index", "Day-by-day index", ("*/index.html",), "experiment_day_index.json"),
    ("story", "Experiment story (semantic)", ("*/experiment_story_report.html", "*/*/experiment_story_report.html"), None),
    ("dispersion_map", "Dispersion map report", ("*/dispersion_map_report.html", "*/*/dispersion_map_report.html"), None),
    ("scan_plan", "Dispersion scan plan", ("*/dispersion_scan_plan.html", "*/*/dispersion_scan_plan.html"), None),
    ("dashboard", "Experiment dashboard", ("*/dashboard.html", "*/*/dashboard.html", "*/*/*/dashboard.html"), "dashboard_manifest.json"),
)


def _discover_experiment_reports(runs_root: Path, *, max_per_kind: int = 40) -> list[dict[str, Any]]:
    """Experiment report artifacts under local_runs, newest first per kind."""
    reports: list[dict[str, Any]] = []
    for kind, label, patterns, required_sibling in _EXPERIMENT_REPORT_KINDS:
        found: list[tuple[float, Path]] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for path in runs_root.glob(pattern):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                if required_sibling and not (path.parent / required_sibling).exists():
                    continue
                seen.add(resolved)
                found.append((path.stat().st_mtime, path))
        found.sort(reverse=True)
        for mtime, path in found[:max_per_kind]:
            relative = path.relative_to(runs_root)
            reports.append(
                {
                    "kind": kind,
                    "kind_label": label,
                    "run": relative.parts[0],
                    "href": "/local-runs/" + "/".join(relative.parts),
                    "modified": mtime,
                }
            )
    return reports


def _experiments_body() -> str:
    return """
<h1>Experiment Reports</h1>
<p class="muted">Generated report surfaces found under <code>local_runs/</code>, newest first.
Day-by-day indexes link each day's morning report and dashboard; the story report groups scans semantically
(alignment by reflection, derived order parameters, session-separated dispersion maps).</p>
<div id="experiment-reports"></div>
<h2>Generate reports</h2>
<pre>tas-copilot workflow overnight-scan-summary --data-root [EXPT_DIR] --since [DAY]T00:00:00 --hours 24 --output-dir [OUT]/summary
tas-copilot workflow draft-overnight-report --summary-json [OUT]/summary/overnight_scan_summary.json --output-dir [OUT]/report
tas-copilot workflow experiment-dashboard --data-root [EXPT_DIR] --output-dir [OUT]/dashboard
tas-copilot workflow experiment-day-index --reports-root [DAYS_ROOT]
tas-copilot workflow experiment-story-report --data-root [EXPT_DIR] --output-dir [DAYS_ROOT]/story
tas-copilot workflow dispersion-scan-plan --data-root [EXPT_DIR] --output-dir [OUT]/scan_plan</pre>
<p class="muted">See <a href="/documentation">Experiment Reports And Scan Planning</a> for the full guide.</p>
<script>
async function loadReports() {
  const data = await (await fetch('/api/experiments/reports')).json();
  const groups = {};
  for (const report of data.reports) (groups[report.kind_label] = groups[report.kind_label] || []).push(report);
  const container = document.getElementById('experiment-reports');
  if (!data.reports.length) {
    container.innerHTML = '<p class="muted">No report artifacts found under local_runs/ yet; use the commands below.</p>';
    return;
  }
  container.innerHTML = Object.entries(groups).map(([label, items]) => `
    <h2>${label}</h2>
    <ul>${items.map(item => `<li><a href="${item.href}">${item.run}</a> <span class="muted">(${new Date(item.modified * 1000).toISOString().slice(0, 10)})</span></li>`).join('')}</ul>
  `).join('');
}
loadReports();
</script>
"""


def _workspace_body() -> str:
    return """
<h1>Experiment Workspace</h1>
<p class="muted">Point at an experiment data directory, then generate report surfaces with one click.
Every run goes through the same deterministic <code>copilot_ask</code> router as the CLI (no live control, planning evidence only)
and lands under <code>local_runs/</code>, where <a href="/experiments">Experiments</a> indexes it.
Long workflows can take minutes; each result shows the equivalent CLI command to run in a terminal instead.</p>
<label>Experiment data directory</label>
<input id="data_root" list="known-roots" placeholder="/path/to/experiment/data">
<datalist id="known-roots"></datalist>
<div id="actions" class="grid" style="margin-top:16px"></div>
<h2>Result</h2>
<p id="run-status" class="muted">No report generated yet.</p>
<ul id="artifact-links"></ul>
<pre id="run-detail" style="display:none"></pre>
<script>
let RUNNING = false;
async function init() {
  const [intentData, rootData] = await Promise.all([
    (await fetch('/api/intents')).json(),
    (await fetch('/api/workspace/data-roots')).json(),
  ]);
  const intents = intentData.intents.filter(item => (item.required_inputs || []).join(',') === 'data_root');
  document.getElementById('actions').innerHTML = intents.map(item => `
    <section class="card">
      <h2 style="font-size:1rem;margin:0 0 6px 0"><code>${item.intent}</code></h2>
      <p class="muted" style="margin:0 0 8px 0">${item.situation}</p>
      <button onclick="runIntent('${item.intent}')">Generate</button>
    </section>`).join('');
  document.getElementById('known-roots').innerHTML =
    rootData.data_roots.map(root => `<option value="${root}">`).join('');
  const saved = localStorage.getItem('workspace_data_root');
  if (saved) document.getElementById('data_root').value = saved;
}
async function runIntent(intent) {
  if (RUNNING) return;
  const dataRoot = document.getElementById('data_root').value.trim();
  if (!dataRoot) {
    document.getElementById('run-status').textContent = 'Set the experiment data directory first.';
    return;
  }
  localStorage.setItem('workspace_data_root', dataRoot);
  RUNNING = true;
  document.getElementById('run-status').textContent = `Running ${intent} on ${dataRoot} ... long workflows can take minutes.`;
  document.getElementById('artifact-links').innerHTML = '';
  document.getElementById('run-detail').style.display = 'none';
  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({intent: intent, data_root: dataRoot}),
    });
    const data = await response.json();
    if (!response.ok) {
      document.getElementById('run-status').textContent = 'Error: ' + (data.detail || response.status);
      return;
    }
    document.getElementById('run-status').textContent = `${intent}: ${data.result.status}. ${data.result.answer || ''}`;
    document.getElementById('artifact-links').innerHTML =
      (data.artifact_links || []).map(link => `<li><a href="${link.href}">${link.label || link.href}</a></li>`).join('');
    const detail = document.getElementById('run-detail');
    detail.textContent = 'Equivalent CLI:\\n' + data.equivalent_cli +
      (data.result.missing_inputs && data.result.missing_inputs.length ? '\\n\\nMissing inputs: ' + data.result.missing_inputs.join(', ') : '');
    detail.style.display = 'block';
  } finally {
    RUNNING = false;
  }
}
init();
</script>
"""


def _validated_fact_catalog(catalog: str, *, runs_root: Path) -> Path:
    known = {entry["path"] for entry in discover_fact_catalogs(runs_root)}
    catalog_dir = Path(catalog).expanduser().resolve()
    if str(catalog_dir) not in known or not (catalog_dir / "papers.jsonl").exists():
        raise HTTPException(status_code=400, detail=f"unknown fact catalog: {catalog}")
    return catalog_dir


def _facts_body() -> str:
    return """
<h1>Paper Fact Review</h1>
<p class="muted">Pick a catalog, open a paper's extracted facts, judge them against the cited chunk text, and export review labels.
Papers with 0 facts have not been extracted yet; the extraction command for the selected catalog is shown below the table.</p>
<label for="fact-catalog">Catalog</label>
<select id="fact-catalog"></select>
<p><a id="review-all" href="#">Open review page for the whole catalog</a></p>
<table id="fact-papers" style="width:100%;border-collapse:collapse"></table>
<h2>Extract facts for more papers</h2>
<pre id="extract-hint"></pre>
<p class="muted">Extraction needs a local model endpoint (Ollama or a NIST vLLM service); see docs/aditya_nist_vllm_gemma.md.
After reviewing, ingest exported labels with <code>tas-copilot review-labels ingest --labels paper_fact_review_labels.jsonl --store [REVIEW_OUTCOMES_SQLITE]</code>.</p>
<script>
async function loadCatalogs() {
  const data = await (await fetch('/api/facts/catalogs')).json();
  const select = document.getElementById('fact-catalog');
  select.innerHTML = data.catalogs.map(c => `<option value="${c.path}">${c.label} (${c.fact_count} facts)</option>`).join('');
  select.addEventListener('change', () => loadPapers(select.value));
  if (data.catalogs.length) loadPapers(data.catalogs[0].path);
}
async function loadPapers(catalog) {
  const data = await (await fetch('/api/facts/papers?catalog=' + encodeURIComponent(catalog))).json();
  document.getElementById('review-all').href = '/facts/review?catalog=' + encodeURIComponent(catalog);
  document.getElementById('extract-hint').textContent = data.extract_command_hint;
  const rows = data.papers.map(p => `<tr>
    <td style="border-bottom:1px solid #d7dee7;padding:6px">${p.title || p.paper_id}</td>
    <td style="border-bottom:1px solid #d7dee7;padding:6px"><code>${p.paper_id}</code></td>
    <td style="border-bottom:1px solid #d7dee7;padding:6px">${p.fact_count} (${p.supported_count} supported)</td>
    <td style="border-bottom:1px solid #d7dee7;padding:6px">${p.fact_count ? `<a href="/facts/review?catalog=${encodeURIComponent(catalog)}&paper_id=${encodeURIComponent(p.paper_id)}">review</a>` : '<span class="muted">not extracted</span>'}</td>
  </tr>`).join('');
  document.getElementById('fact-papers').innerHTML =
    '<tr><th style="text-align:left;padding:6px">Title</th><th style="text-align:left;padding:6px">Paper</th><th style="text-align:left;padding:6px">Facts</th><th style="text-align:left;padding:6px"></th></tr>' + rows;
}
loadCatalogs();
</script>
"""


def _page(title: str, body: str, *, nav: bool = True) -> str:
    nav_html = (
        """
    <nav>
      <span class="nav-group"><span class="nav-label">Operate</span>
        <a href="/">Home</a>
        <a href="/ask">Ask</a>
        <a href="/workspace">Workspace</a>
        <a href="/experiments">Experiments</a>
        <a href="/alignment">Alignment</a>
      </span>
      <span class="nav-group"><span class="nav-label">Review</span>
        <a href="/review">Review Dashboard</a>
        <a href="/chunks">Chunk Review</a>
        <a href="/facts">Paper Facts</a>
      </span>
      <span class="nav-group"><span class="nav-label">Reference</span>
        <a href="/runs">Local Runs</a>
        <a href="/tools">Tools</a>
        <a href="/documentation">Docs</a>
      </span>
    </nav>"""
        if nav
        else """
    <p style="margin: 4px 0 0 0; color: #52616f;">TAS alignment planning sandbox. Uploaded CIFs are processed in memory and never stored. Planning evidence only; not an approved instrument operation.</p>"""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - TASCopilot</title>
  <style>
    :root {{ --ink: #17212b; --muted: #52616f; --line: #d7dee7; --bg: #f7f9fb; --accent: #205493; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: white; line-height: 1.5; }}
    header {{ background: var(--bg); border-bottom: 1px solid var(--line); padding: 20px min(5vw, 44px); }}
    main {{ padding: 24px min(5vw, 44px); max-width: 1100px; }}
    nav a {{ margin-right: 16px; color: var(--accent); }}
    .nav-group {{ display: inline-block; margin-right: 28px; white-space: nowrap; }}
    .nav-label {{ color: var(--muted); font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: white; }}
    label {{ display: block; font-weight: 600; margin-top: 12px; }}
    input, textarea {{ width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px; font: inherit; }}
    button {{ margin-top: 12px; padding: 8px 12px; border: 1px solid var(--accent); border-radius: 6px; color: white; background: var(--accent); cursor: pointer; }}
    pre {{ overflow-x: auto; padding: 12px; border-radius: 6px; background: #17212b; color: #f7f9fb; }}
    code {{ background: #eef2f6; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>{nav_html}
  </header>
  <main>{body}</main>
</body>
</html>
"""


def _landing_body() -> str:
    cards = []
    for item in copilot_intent_examples():
        cards.append(
            "<section class=\"card\">"
            f"<h2>{html.escape(str(item['situation']))}</h2>"
            f"<p class=\"muted\">Intent: <code>{html.escape(str(item['intent']))}</code></p>"
            f"<p>{html.escape(str(item['example_question']))}</p>"
            f"<pre>{html.escape(str(item['command']))}</pre>"
            "</section>"
        )
    return (
        "<p>This is a local-only web shell over the existing CLI/workflow engine. It adds discovery and review surfaces, not new scientific behavior.</p>"
        "<p><strong>Safety boundary:</strong> no live control, no approved operations, no definitive claims without reviewed evidence.</p>"
        "<h2>Operate</h2>"
        "<div class=\"grid\">"
        "<section class=\"card\">"
        "<h2>Experiment Workspace</h2>"
        "<p>Point at an experiment data directory and generate dashboards, day summaries, story reports, dispersion maps, and scan plans - then open the results.</p>"
        "<p><a href=\"/workspace\">Open experiment workspace</a> or <a href=\"/experiments\">browse generated reports</a>.</p>"
        "<p class=\"muted\">Runs the same bounded copilot router as the CLI; each run shows its equivalent CLI command.</p>"
        "</section>"
        "<section class=\"card\">"
        "<h2>Alignment Dashboard</h2>"
        "<p>Upload a CIF, choose a scattering plane, and review suggested alignment peaks with the Brillouin-zone section and Al/Cu powder rings. F&sup2; comes from the configured GSAS-II runtime.</p>"
        "<p><a href=\"/alignment\">Open alignment dashboard</a>.</p>"
        "<p class=\"muted\">Planning evidence requiring instrument-scientist review; not approved operations.</p>"
        "</section>"
        "</div>"
        "<h2>Review</h2>"
        "<div class=\"grid\">"
        "<section class=\"card\">"
        "<h2>Review Dashboard</h2>"
        "<p>Review candidate paper links, figure-source candidates, order-parameter candidates, spurion candidates, alignment candidates, and adaptive decisions.</p>"
        "<p><a href=\"/review\">Open student review dashboard</a>, <a href=\"/labels\">open label-store summary</a>, or <a href=\"/runs\">find a recent run with a label store</a>.</p>"
        "<p class=\"muted\">Labels calibrate priors later; they do not accept scientific claims by themselves.</p>"
        "</section>"
        "<section class=\"card\">"
        "<h2>Chunk Review</h2>"
        "<p>Inspect a paper's extracted chunks next to the raw extractor markdown and the PDF, then record extraction-quality labels.</p>"
        "<p><a href=\"/chunks\">Open chunk review</a>. For extracted paper facts, use <a href=\"/facts\">paper fact review</a>.</p>"
        "<p class=\"muted\">Extracted text is untrusted evidence; chunk labels calibrate extraction quality only.</p>"
        "</section>"
        "</div>"
        "<h2>Reference</h2>"
        "<div class=\"grid\">"
        "<section class=\"card\">"
        "<h2>Documentation</h2>"
        "<p>Open the browser-readable docs, including the student day-one checklist and review-labeling guide.</p>"
        "<p><a href=\"/documentation\">Open documentation front door</a> or <a href=\"/documentation\">open static docs index</a>.</p>"
        "<p class=\"muted\">If this is missing, rerun <code>./scripts/bootstrap_student.sh</code> from the repo root and restart the server.</p>"
        "</section>"
        "</div>"
        "<p><a href=\"/ask\">Ask a bounded question</a> or choose a situation below.</p>"
        f"<div class=\"grid\">{''.join(cards)}</div>"
    )


def _ask_body() -> str:
    return """
<p>Questions are routed through the deterministic <code>copilot_ask</code> workflow. The web layer does not decide the route.
Picking an intent skips keyword routing and shows which inputs that intent needs; the router still validates them.</p>
<label>Intent</label>
<select id="intent"><option value="">auto (route from the question)</option></select>
<p id="intent-hint" class="muted"></p>
<label>Question</label>
<textarea id="question" rows="4">What can you do?</textarea>
<label>Output directory</label>
<input id="output_dir" value="local_runs/web_copilot_ask">
<label>Data root (optional)</label>
<input id="data_root" placeholder="/path/to/data">
<label>Corpus catalog (optional)</label>
<input id="corpus_catalog" placeholder="/path/to/corpus_catalog">
<label>Data catalog (optional)</label>
<input id="data_catalog" placeholder="/path/to/data_catalog.sqlite">
<label>Precedent index (optional)</label>
<input id="precedent_index" placeholder="/path/to/bt7_hydrated_index.jsonl">
<label>Paper ID (optional)</label>
<input id="paper_id">
<label>Figure ID (optional)</label>
<input id="figure_id">
<label>Fixture (optional)</label>
<input id="fixture" placeholder="tests/fixtures/...">
<button onclick="ask()">Ask</button>
<h2>Result</h2>
<ul id="artifact-links"></ul>
<pre id="result">No question submitted yet.</pre>
<script>
const FIELD_IDS = ["intent", "question", "output_dir", "data_root", "corpus_catalog", "data_catalog", "precedent_index", "paper_id", "figure_id", "fixture"];
let INTENTS = [];
async function loadIntents() {
  const data = await (await fetch("/api/intents")).json();
  INTENTS = data.intents;
  const select = document.getElementById("intent");
  select.innerHTML = '<option value="">auto (route from the question)</option>' +
    INTENTS.map(item => `<option value="${item.intent}">${item.intent}</option>`).join("");
  select.addEventListener("change", () => {
    const item = INTENTS.find(entry => entry.intent === select.value);
    const hint = document.getElementById("intent-hint");
    if (!item) { hint.textContent = ""; return; }
    hint.textContent = item.situation + " Required inputs: " + ((item.required_inputs || []).join(", ") || "none") + ".";
    document.getElementById("question").value = item.example_question;
  });
}
async function ask() {
  const payload = {};
  for (const id of FIELD_IDS) payload[id] = document.getElementById(id).value;
  document.getElementById("result").textContent = "Running... long workflows can take minutes; the equivalent CLI command in the result can be run in a terminal instead.";
  document.getElementById("artifact-links").innerHTML = "";
  const response = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await response.json();
  document.getElementById("artifact-links").innerHTML =
    (data.artifact_links || []).map(link => `<li><a href="${link.href}">${link.label || link.href}</a></li>`).join("");
  document.getElementById("result").textContent = JSON.stringify(data, null, 2);
}
loadIntents();
</script>
"""


def _runs_body(root: Path) -> str:
    runs = _run_records(root)[:80]
    if not runs:
        return f"<p>No reports found under <code>{html.escape(str(root))}</code>.</p>"
    rows = []
    for run in runs:
        link = run["href"] if run.get("href") else "#"
        labels_link = f"<a href=\"/labels?store={html.escape(run['review_store'])}\">Open labels</a>" if run.get("review_store") else ""
        rows.append(
            f"<tr><td><a href=\"{html.escape(link)}\">{html.escape(run['name'])}</a></td>"
            f"<td>{html.escape(run['kind'])}</td><td>{html.escape(run['modified'])}</td><td>{labels_link}</td><td><code>{html.escape(run['path'])}</code></td></tr>"
        )
    return (
        f"<p>Recent HTML/Markdown/JSON reports under <code>{html.escape(str(root))}</code>.</p>"
        "<table><thead><tr><th>Name</th><th>Kind</th><th>Modified</th><th>Review</th><th>Path</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _documentation_body(docs_root: Path) -> str:
    if not docs_root.exists():
        return (
            "<p>The documentation directory was not found.</p>"
            f"<p>Expected path: <code>{html.escape(str(docs_root))}</code></p>"
        )
    links = [
        (path.stem.replace("_", " ").title(), "/docs/" + path.name)
        for path in sorted(docs_root.iterdir())
        if path.is_file() and not path.is_symlink() and path.suffix in {".md", ".html"}
    ]
    rows = "".join(f"<li><a href=\"{href}\">{html.escape(label)}</a></li>" for label, href in links)
    return (
        "<p>These are the local browser-readable project docs served from the current checkout.</p>"
        f"<p class=\"muted\">Docs root: <code>{html.escape(str(docs_root))}</code></p>"
        f"<ul>{rows}</ul>"
    )


def _run_records(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    cache_key = str(root)
    cached = _RUN_RECORD_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached[0] < _RUN_RECORD_CACHE_SECONDS:
        return cached[1]
    records = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".html", ".md", ".json"}:
            continue
        stat = path.stat()
        href = f"/local-runs/{path.relative_to(root).as_posix()}"
        review_store = _nearest_review_store(path)
        records.append(
            {
                "name": path.name,
                "kind": path.suffix.lower().lstrip("."),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "path": str(path),
                "href": href,
                "review_store": str(review_store) if review_store else "",
            }
        )
    records.sort(key=lambda item: item["modified"], reverse=True)
    records = records[:300]
    _RUN_RECORD_CACHE[cache_key] = (now, records)
    return records


def _validated_output_dir(value: str | None, *, runs_root: Path, field_name: str, default_prefix: str) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        candidate = (runs_root / default_prefix / timestamp).resolve()
    try:
        candidate.relative_to(runs_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must live under {runs_root}") from exc
    return candidate


def _labels_body(root: Path, *, selected_store: str | None = None) -> str:
    stores = _discover_review_stores(root)
    options = "".join(f"<option value=\"{html.escape(item['path'])}\">{html.escape(item['label'])}</option>" for item in stores)
    selected = html.escape(selected_store or (stores[0]["path"] if stores else ""))
    return f"""
<p>Use this page for the unified review-label store. Review labels calibrate priors; they do not accept scientific claims by themselves.</p>
<p class="muted">Discovered stores under <code>{html.escape(str(root))}</code>: {len(stores)}</p>
<datalist id="review_stores">{options}</datalist>
<h2>Summary</h2>
<label>Review outcome store</label>
<input id="summary_store" list="review_stores" value="{selected}" placeholder="local_runs/.../review_outcomes.sqlite">
<button onclick="summary()">Summarize</button>
<pre id="summary_result">No store summarized yet.</pre>
<h2>Ingest reviewed dashboard export</h2>
<label>Reviewed rows JSONL</label>
<input id="labels" placeholder="/path/to/reviewed_rows.jsonl">
<label>Review outcome store</label>
<input id="ingest_store" list="review_stores" value="{selected}" placeholder="/path/to/review_outcomes.sqlite">
<label>Output directory</label>
<input id="ingest_output" value="local_runs/web_review_label_ingest">
<button onclick="ingest()">Ingest</button>
<pre id="ingest_result">No labels ingested yet.</pre>
<script>
async function summary() {{
  const payload = {{store: document.getElementById("summary_store").value}};
  const response = await fetch("/api/review-labels/summary", {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(payload)}});
  document.getElementById("summary_result").textContent = JSON.stringify(await response.json(), null, 2);
}}
async function ingest() {{
  const payload = {{labels: document.getElementById("labels").value, store: document.getElementById("ingest_store").value, output_dir: document.getElementById("ingest_output").value}};
  const response = await fetch("/api/review-labels/ingest", {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(payload)}});
  document.getElementById("ingest_result").textContent = JSON.stringify(await response.json(), null, 2);
}}
</script>
"""


def _chunks_body() -> str:
    return """
<style>
  main { max-width: none !important; }
  .chunk-layout { display: grid; grid-template-columns: minmax(320px, 38%) 1fr; gap: 16px; align-items: start; }
  .chunk-table-wrap { max-height: 75vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
  .chunk-table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .chunk-table-wrap th, .chunk-table-wrap td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
  .chunk-table-wrap tbody tr { cursor: pointer; }
  .chunk-table-wrap tbody tr:hover { background: #eef2f6; }
  .chunk-table-wrap tbody tr.selected { background: #dce8f7; }
  .tabbar button { margin: 0 8px 0 0; background: white; color: var(--accent); }
  .tabbar button.active { background: var(--accent); color: white; }
  .pane { border: 1px solid var(--line); border-radius: 8px; padding: 12px; max-height: 60vh; overflow: auto; background: white; }
  .chunk-item { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
  .chunk-item .meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
  .chunk-item.caption { border-left: 4px solid #b25f00; }
  .chunk-item.derived { border-left: 4px solid #205493; }
  .chunk-item pre { background: #f7f9fb; color: var(--ink); white-space: pre-wrap; margin: 0; padding: 6px; }
  #pdf_frame { width: 100%; height: 60vh; border: 1px solid var(--line); border-radius: 8px; }
  select { padding: 8px; border: 1px solid var(--line); border-radius: 6px; font: inherit; max-width: 100%; }
  .label-card textarea { min-height: 56px; }
</style>
<p>Select a corpus catalog and a paper, inspect its chunks against the extractor outputs (side by side) and the PDF, then record an extraction-quality label. Extracted text is untrusted evidence; labels calibrate extraction quality and do not accept scientific claims.</p>
<div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
  <div><label>Corpus catalog</label><select id="catalog"></select></div>
  <div><label>Extractor run</label><select id="run"></select></div>
  <div><label>Link-label store (optional)</label><select id="link_store"></select></div>
</div>
<div class="chunk-layout" style="margin-top: 16px;">
  <div>
    <input id="paper_filter" placeholder="Filter by title, author, or material" oninput="renderPapers()">
    <div class="chunk-table-wrap" style="margin-top: 8px;">
      <table>
        <thead><tr><th>Title</th><th>Year</th><th>Material</th><th>Chunks</th><th>Captions</th><th>Link labels</th></tr></thead>
        <tbody id="paper_rows"><tr><td colspan="6" class="muted">Choose a catalog to load papers.</td></tr></tbody>
      </table>
    </div>
  </div>
  <div>
    <h2 id="paper_title" style="margin-top: 0;">No paper selected</h2>
    <p id="paper_meta" class="muted"></p>
    <div class="tabbar">
      <button id="tab_chunks" class="active" onclick="showTab('chunks')">Chunks</button>
      <button id="tab_extractors" onclick="showTab('extractors')">Extractor Comparison</button>
      <button id="tab_pdf" onclick="showTab('pdf')">PDF</button>
    </div>
    <div id="pane_chunks" class="pane" style="margin-top: 10px;"><p class="muted">Chunks appear here.</p></div>
    <div id="pane_extractors" style="margin-top: 10px; display: none;">
      <p class="muted" style="margin: 0 0 8px 0;"><label style="display: inline; font-weight: 400;"><input type="checkbox" id="compare_raw" style="width: auto;" onchange="renderExtractorColumns()"> Show raw extracted text instead of derived chunks</label></p>
      <div id="extractor_columns" class="grid" style="grid-template-columns: 1fr 1fr; align-items: start;"></div>
    </div>
    <div id="pane_pdf" style="margin-top: 10px; display: none;"><iframe id="pdf_frame" title="paper pdf"></iframe></div>
    <section class="card label-card" style="margin-top: 14px;">
      <h2 style="margin-top: 0;">Extraction-Quality Label</h2>
      <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
        <div><label>Verdict</label><select id="label_status">
          <option value="acceptable">acceptable</option>
          <option value="minor_issues">minor issues</option>
          <option value="missing_content">missing content</option>
          <option value="wrong_sections">wrong sections</option>
          <option value="garbled">garbled</option>
          <option value="caption_table_failure">caption/table failure</option>
          <option value="unusable">unusable</option>
        </select></div>
        <div><label>Best extractor</label><select id="label_best_backend">
          <option value="not_compared">not compared</option>
          <option value="docling">docling</option>
          <option value="grobid">grobid</option>
          <option value="tie">tie</option>
          <option value="neither">neither</option>
        </select></div>
        <div><label>Reviewer</label><input id="label_reviewer" placeholder="your name"></div>
      </div>
      <label>Comments (free text, parsed later)</label>
      <textarea id="label_comment" placeholder="What is wrong or notable about this extraction?"></textarea>
      <button onclick="saveLabel()">Save label</button>
      <pre id="label_result" style="margin-top: 10px;">No label saved yet.</pre>
      <p class="muted">Labels append to <code>local_runs/chunk_review/chunk_review_labels.jsonl</code>; ingest them into the unified store from <a href="/labels">Review Labels</a>.</p>
    </section>
  </div>
</div>
<script>
let sources = {catalogs: [], extractor_runs: []};
let papers = [];
let selectedPaper = null;

function el(id) { return document.getElementById(id); }

function fillSelect(node, items, valueKey, labelKey, emptyLabel) {
  node.replaceChildren();
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    node.appendChild(option);
    return;
  }
  for (const item of items) {
    const option = document.createElement("option");
    option.value = valueKey ? item[valueKey] : item;
    option.textContent = labelKey ? item[labelKey] : item;
    node.appendChild(option);
  }
}

async function loadSources() {
  sources = await (await fetch("/api/chunks/sources")).json();
  fillSelect(el("catalog"), sources.catalogs, "path", "label", "no corpus catalogs found under local_runs/corpus_catalogs");
  fillSelect(el("run"), sources.extractor_runs, "path", "label", "no extractor runs found under local_runs/vista_corpus_extractor_outputs");
  const stores = [{path: "", label: "none"}].concat(sources.review_stores || []);
  fillSelect(el("link_store"), stores, "path", "label", "no review stores found under local_runs");
  el("catalog").onchange = loadPapers;
  el("link_store").onchange = loadPapers;
  el("run").onchange = () => { if (selectedPaper) openPaper(selectedPaper); };
  if (sources.catalogs.length) await loadPapers();
}

async function loadPapers() {
  const catalog = el("catalog").value;
  if (!catalog) return;
  const params = new URLSearchParams({catalog});
  if (el("link_store").value) params.set("link_store", el("link_store").value);
  const data = await (await fetch("/api/chunks/papers?" + params)).json();
  papers = data.papers || [];
  renderPapers();
}

function linkVerdictText(paper) {
  const verdicts = paper.link_verdicts;
  if (!verdicts) return "";
  return Object.entries(verdicts)
    .filter(([key, count]) => key !== "total" && count)
    .map(([key, count]) => count + " " + key)
    .join(", ");
}

function renderPapers() {
  const needle = el("paper_filter").value.toLowerCase();
  const body = el("paper_rows");
  body.replaceChildren();
  for (const paper of papers) {
    const linkText = linkVerdictText(paper);
    const haystack = (String(paper.title || "") + " " + (paper.authors || []).join(" ") + " " + String(paper.material_guess || "") + " " + linkText).toLowerCase();
    if (needle && !haystack.includes(needle)) continue;
    const row = document.createElement("tr");
    row.dataset.paperId = paper.paper_id;
    if (selectedPaper === paper.paper_id) row.classList.add("selected");
    const captions = (paper.chunk_type_counts || {}).caption || (paper.chunk_type_counts || {}).figure_caption || 0;
    for (const value of [paper.title || paper.paper_id, paper.year ?? "", paper.material_guess || "", paper.chunk_count, captions, linkText]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    row.onclick = () => openPaper(paper.paper_id);
    body.appendChild(row);
  }
  if (!body.children.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "muted";
    cell.textContent = "No papers match.";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

async function openPaper(paperId) {
  selectedPaper = paperId;
  renderPapers();
  const params = new URLSearchParams({catalog: el("catalog").value});
  if (el("run").value) params.set("run", el("run").value);
  const detail = await (await fetch("/api/chunks/paper/" + encodeURIComponent(paperId) + "?" + params)).json();
  el("paper_title").textContent = detail.title || paperId;
  el("paper_meta").textContent = [
    (detail.authors || []).slice(0, 6).join(", ") + ((detail.authors || []).length > 6 ? ", ..." : ""),
    detail.year ? "(" + detail.year + ")" : "",
    detail.doi ? "doi:" + detail.doi : "",
    detail.chunk_count + " chunks",
  ].filter(Boolean).join("  ");
  renderChunks(detail.chunks || []);
  extractorState = detail.extractor_outputs || {};
  renderExtractorColumns();
  el("pdf_frame").src = "/papers/" + encodeURIComponent(paperId) + "?catalog=" + encodeURIComponent(el("catalog").value);
  resetLabelForm();
  await loadExistingLabels(paperId);
}

function resetLabelForm() {
  // The verdict/comment fields describe one paper; never carry them over to
  // the next selection. The reviewer name is per-person and is kept.
  el("label_status").value = "acceptable";
  el("label_best_backend").value = "not_compared";
  el("label_comment").value = "";
  el("label_result").textContent = "No label saved yet for this paper.";
}

async function loadExistingLabels(paperId) {
  const params = new URLSearchParams({paper_id: paperId});
  if (el("catalog").value) params.set("catalog", el("catalog").value);
  const data = await (await fetch("/api/chunks/labels?" + params)).json();
  if (paperId !== selectedPaper) return;
  const labels = (data.labels && data.labels.length) ? data.labels : (data.carryover_labels || []);
  if (!labels.length) return;
  const latest = labels[0];
  if (latest.status) el("label_status").value = latest.status;
  if (latest.best_backend) el("label_best_backend").value = latest.best_backend;
  el("label_comment").value = latest.comment || "";
  if (!el("label_reviewer").value && latest.reviewer) el("label_reviewer").value = latest.reviewer;
  const lines = labels.map((label) => [
    label.labeled_at || "(no timestamp)",
    label.status,
    label.best_backend && label.best_backend !== "not_compared" ? "best: " + label.best_backend : "",
    label.reviewer ? "by " + label.reviewer : "",
    label.comment ? JSON.stringify(label.comment) : "",
    label.catalog_mismatch ? "[catalog path differs]" : "",
  ].filter(Boolean).join("  "));
  const prefix = data.labels && data.labels.length
    ? "Previous label(s) for this paper (newest first; form prefilled with the latest):"
    : "Previous label(s) for this paper from another catalog path (newest first; form prefilled with the latest):";
  el("label_result").textContent = prefix + "\\n" + lines.join("\\n");
}

let extractorState = {};

function renderExtractorColumns() {
  const outputs = extractorState;
  const showRaw = el("compare_raw").checked;
  const wrap = el("extractor_columns");
  wrap.replaceChildren();
  const backends = Object.keys(outputs).filter((key) => typeof outputs[key] === "object");
  if (!backends.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "Select an extractor run to compare backend outputs side by side.";
    wrap.appendChild(note);
    return;
  }
  for (const backend of backends) {
    const output = outputs[backend];
    const column = document.createElement("div");
    column.className = "pane";
    const heading = document.createElement("h3");
    heading.style.marginTop = "0";
    const chunkCount = (output.chunks || []).length;
    heading.textContent = backend + (output.kind ? " (" + output.kind + ")" : "") + (output.status === "ok" && !showRaw ? " - " + chunkCount + " chunks" : "");
    column.appendChild(heading);
    if (output.status !== "ok") {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "[" + output.status + "]";
      column.appendChild(note);
    } else if (showRaw) {
      const text = document.createElement("pre");
      text.style.background = "#f7f9fb";
      text.style.color = "var(--ink)";
      text.style.whiteSpace = "pre-wrap";
      text.textContent = output.text;
      column.appendChild(text);
    } else {
      for (const chunk of output.chunks || []) {
        column.appendChild(buildChunkItem(chunk));
      }
      if (!chunkCount) {
        const note = document.createElement("p");
        note.className = "muted";
        note.textContent = "The chunker produced no chunks from this backend's text.";
        column.appendChild(note);
      }
    }
    wrap.appendChild(column);
  }
}

function renderChunks(chunks) {
  const pane = el("pane_chunks");
  pane.replaceChildren();
  if (!chunks.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "No chunks recorded for this paper in the selected catalog.";
    pane.appendChild(note);
    return;
  }
  for (const chunk of chunks) {
    pane.appendChild(buildChunkItem(chunk));
  }
}

function buildChunkItem(chunk) {
  const item = document.createElement("div");
  const chunkType = String(chunk.chunk_type || "");
  let flavor = "";
  if (chunkType.includes("caption")) flavor = " caption";
  else if (chunkType === "material_composition" || chunkType === "structure_parameters") flavor = " derived";
  item.className = "chunk-item" + flavor;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = [
    chunk.chunk_type,
    chunk.section ? "section: " + chunk.section : "",
    chunk.figure_id ? "figure: " + chunk.figure_id : "",
    chunk.table_id ? "table: " + chunk.table_id : "",
    chunk.token_count != null ? chunk.token_count + " tokens" : "",
    chunk.chunk_id,
  ].filter(Boolean).join("  |  ");
  const text = document.createElement("pre");
  text.textContent = chunk.text || "";
  item.appendChild(meta);
  item.appendChild(text);
  return item;
}

function showTab(name) {
  for (const tab of ["chunks", "extractors", "pdf"]) {
    el("pane_" + tab).style.display = tab === name ? "" : "none";
    el("tab_" + tab).classList.toggle("active", tab === name);
  }
}

async function saveLabel() {
  if (!selectedPaper) {
    el("label_result").textContent = "Select a paper first.";
    return;
  }
  const payload = {
    paper_id: selectedPaper,
    status: el("label_status").value,
    catalog: el("catalog").value,
    extractor_run: el("run").value,
    best_backend: el("label_best_backend").value,
    comment: el("label_comment").value,
    reviewer: el("label_reviewer").value,
  };
  const response = await fetch("/api/chunks/labels", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  el("label_result").textContent = JSON.stringify(await response.json(), null, 2);
}

loadSources();
</script>
"""


def _alignment_body() -> str:
    return """
<style>
  main { max-width: none !important; }
  .align-layout { display: grid; grid-template-columns: minmax(330px, 36%) 1fr; gap: 16px; align-items: start; }
  .align-table-wrap { max-height: 46vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
  .align-table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .align-table-wrap th, .align-table-wrap td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: left; }
  .peak-item { border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
  .peak-item .meta { font-size: 0.8rem; color: var(--muted); }
  .hkl-inputs input { width: 56px; display: inline-block; margin-right: 4px; }
  #zone_plot { border: 1px solid var(--line); border-radius: 8px; background: white; width: 100%; }
  select, input[type=number] { padding: 6px; border: 1px solid var(--line); border-radius: 6px; font: inherit; }
</style>
<p>Upload a CIF, choose a scattering plane, and review suggested alignment peaks against the Brillouin-zone section and sample-environment powder rings. Neutron nuclear F&sup2; is calculated through the configured GSAS-II runtime; without it this page fails closed. All output is planning evidence requiring instrument-scientist review.</p>
<section class="card">
  <h2 style="margin-top: 0;">1. Structure (CIF)</h2>
  <input type="file" id="cif_file" accept=".cif" onchange="uploadCif()">
  <p id="structure_summary" class="muted">No CIF uploaded yet.</p>
</section>
<section class="card" style="margin-top: 12px;">
  <h2 style="margin-top: 0;">2. Plane, Energy, Environment</h2>
  <div class="grid" style="grid-template-columns: auto auto auto auto auto; align-items: end; gap: 18px;">
    <div class="hkl-inputs"><label>Orient 1 (H K L)</label>
      <input type="number" id="o1h" value="1" step="any"><input type="number" id="o1k" value="0" step="any"><input type="number" id="o1l" value="0" step="any"></div>
    <div class="hkl-inputs"><label>Orient 2 (H K L)</label>
      <input type="number" id="o2h" value="0" step="any"><input type="number" id="o2k" value="1" step="any"><input type="number" id="o2l" value="0" step="any"></div>
    <div><label>Ef (meV)</label><input type="number" id="energy_mev" value="14.7" step="any" style="width: 90px;"></div>
    <div><label>Environment</label>
      <label style="display: inline; font-weight: 400;"><input type="checkbox" id="env_aluminum" style="width: auto;" checked> Aluminum</label>
      <label style="display: inline; font-weight: 400; margin-left: 10px;"><input type="checkbox" id="env_copper" style="width: auto;"> Copper</label></div>
    <div><button onclick="calculate()" style="margin-top: 0;">Calculate</button></div>
  </div>
  <p id="calc_status" class="muted"></p>
</section>
<div class="align-layout" style="margin-top: 14px;">
  <div>
    <h2>Suggested Alignment Peaks</h2>
    <div id="suggested_peaks"><p class="muted">Run a calculation first.</p></div>
    <h2>All Reflections</h2>
    <div style="margin-bottom: 6px;">
      <label style="display: inline; font-weight: 400;">Plane family
        <select id="refl_family" onchange="renderReflections()">
          <option value="all">all (hkl)</option>
          <option value="hk0">(h k 0)</option>
          <option value="h0l">(h 0 l)</option>
          <option value="0kl">(0 k l)</option>
          <option value="hhl">(h h l)</option>
          <option value="h-hl">(h -h l)</option>
          <option value="00l">(0 0 l)</option>
        </select>
      </label>
      <span class="muted" style="margin-left: 10px;">F&sup2; scaled to strongest = 100. Click headers to sort.</span>
    </div>
    <div class="align-table-wrap">
      <table>
        <thead><tr>
          <th onclick="sortReflections('hkl')" style="cursor: pointer;">(H K L)</th>
          <th onclick="sortReflections('d_spacing')" style="cursor: pointer;">d (&Aring;)</th>
          <th onclick="sortReflections('q_inv_angstrom')" style="cursor: pointer;">|Q| (&Aring;&#8315;&sup1;)</th>
          <th onclick="sortReflections('two_theta_deg')" style="cursor: pointer;">2&theta;</th>
          <th onclick="sortReflections('multiplicity')" style="cursor: pointer;">mult</th>
          <th onclick="sortReflections('f2_scaled')" style="cursor: pointer;">F&sup2; (max=100)</th>
        </tr></thead>
        <tbody id="reflection_rows"></tbody>
      </table>
    </div>
    <section class="card" style="margin-top: 14px;">
      <h2 style="margin-top: 0;">Vector Tools</h2>
      <p class="muted" style="margin-top: 0;">Reciprocal-space angles and in-plane orthogonals from the icp-lattice-calculator metric; equivalents from GSAS-II. Uses the uploaded CIF and the plane above.</p>
      <div class="grid" style="grid-template-columns: auto auto auto; align-items: end; gap: 14px;">
        <div class="hkl-inputs"><label>Vector 1 (H K L)</label>
          <input type="number" id="vt1h" value="1" step="any"><input type="number" id="vt1k" value="0" step="any"><input type="number" id="vt1l" value="0" step="any"></div>
        <div class="hkl-inputs"><label>Vector 2 (optional)</label>
          <input type="number" id="vt2h" step="any"><input type="number" id="vt2k" step="any"><input type="number" id="vt2l" step="any"></div>
        <div><button onclick="vectorTools()" style="margin-top: 0;">Compute</button></div>
      </div>
      <div id="vector_tools_result"><p class="muted">Angle between vectors (any HKL, not plane-constrained), the in-plane orthogonal of vector 1, and symmetry equivalents appear here.</p></div>
    </section>
  </div>
  <div>
    <h2>Scattering Plane: Brillouin Zone, Peaks, Powder Rings</h2>
    <svg id="zone_plot" viewBox="0 0 700 700" xmlns="http://www.w3.org/2000/svg"></svg>
    <p class="muted">Scroll or drag to pan; pinch or ctrl+scroll to zoom; double-click to reset; hover any marker for its label. Marker size and color scale with relative F&sup2;; open circles are Friedel mates. Blue polygons are Brillouin-zone sections repeated at in-plane reciprocal-lattice points (central zone shaded). Purple arrows are the a*/b*/c* directions projected into the plane (dashed when out of plane). Dashed circles are powder rings: Aluminum gray, Copper orange. Axes are the orthonormal scattering-plane basis in &Aring;&#8315;&sup1;.</p>
  </div>
</div>
<script>
const SVGNS = "http://www.w3.org/2000/svg";
let cifText = null;

function el(id) { return document.getElementById(id); }

async function uploadCif() {
  const file = el("cif_file").files[0];
  if (!file) return;
  const content = await file.text();
  const response = await fetch("api/alignment/upload-cif", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({filename: file.name, content})});
  const data = await response.json();
  const s = data.structure || {};
  if (s.status !== "success") {
    el("structure_summary").textContent = "CIF parse failed: " + JSON.stringify(s.warnings || s.status);
    cifText = null;
    return;
  }
  cifText = content;
  el("structure_summary").textContent =
    s.formula + "  |  " + s.space_group + " (#" + s.space_group_number + ", " + s.crystal_system + ")  |  " +
    "a=" + s.lattice.a.toFixed(4) + " b=" + s.lattice.b.toFixed(4) + " c=" + s.lattice.c.toFixed(4) +
    "  alpha=" + s.lattice.alpha.toFixed(2) + " beta=" + s.lattice.beta.toFixed(2) + " gamma=" + s.lattice.gamma.toFixed(2) +
    "  |  " + s.asymmetric_site_count + " asymmetric sites";
}

function vec(prefix) { return ["h","k","l"].map((axis) => parseFloat(el(prefix + axis).value) || 0); }

async function calculate() {
  if (!cifText) { el("calc_status").textContent = "Upload a CIF first."; return; }
  const materials = [];
  if (el("env_aluminum").checked) materials.push("aluminum");
  if (el("env_copper").checked) materials.push("copper");
  el("calc_status").textContent = "Calculating through the GSAS-II runtime...";
  const payload = {cif_content: cifText, orient1: vec("o1"), orient2: vec("o2"), energy_mev: parseFloat(el("energy_mev").value) || 14.7, environment_materials: materials};
  const response = await fetch("api/alignment/report", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const report = await response.json();
  if (report.status !== "success") {
    el("calc_status").textContent = "Failed closed: " + (report.missing_inputs || []).join(", ") + "  " + (report.warnings || []).slice(-2).join("  ");
    return;
  }
  el("calc_status").textContent = report.reflection_count + " reflections (" + (report.unique_star_count ?? "?") + " symmetry stars) from GSAS-II  |  plane " + report.scattering_plane + "  |  " + (report.suggested_peaks || []).length + " suggested peaks";
  renderSuggested(report.suggested_peaks || []);
  prepareReflections(report.reflections || []);
  renderReflections();
  renderZone(report);
}

let reflectionRows = [];
let reflectionSort = {key: "f2_scaled", descending: true};

function prepareReflections(reflections) {
  let f2max = 0;
  for (const r of reflections) f2max = Math.max(f2max, r.relative_structure_factor);
  reflectionRows = reflections.map((r) => Object.assign({}, r, {
    f2_scaled: f2max > 0 ? 100 * r.relative_structure_factor / f2max : 0,
    hkl: [r.h, r.k, r.l],
  }));
}

const FAMILY_TESTS = {
  all: () => true,
  hk0: (r) => r.l === 0,
  h0l: (r) => r.k === 0,
  "0kl": (r) => r.h === 0,
  hhl: (r) => r.h === r.k,
  "h-hl": (r) => r.h === -r.k,
  "00l": (r) => r.h === 0 && r.k === 0,
};

function sortReflections(key) {
  if (reflectionSort.key === key) reflectionSort.descending = !reflectionSort.descending;
  else reflectionSort = {key, descending: key === "f2_scaled" || key === "multiplicity"};
  renderReflections();
}

function renderSuggested(peaks) {
  const wrap = el("suggested_peaks");
  wrap.replaceChildren();
  if (!peaks.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "No reflections survived the scattering-plane and A4 criteria.";
    wrap.appendChild(note);
    return;
  }
  for (const peak of peaks) {
    const item = document.createElement("div");
    item.className = "peak-item";
    const head = document.createElement("strong");
    head.textContent = "(" + peak.h + " " + peak.k + " " + peak.l + ")   score " + peak.score;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "|Q|=" + peak.q_inv_angstrom.toFixed(3) + "  d=" + peak.d_spacing.toFixed(3) + "  2theta=" + peak.two_theta_deg.toFixed(2) +
      "  F2=" + peak.relative_structure_factor.toFixed(3) + "  " + peak.scattering_plane_status +
      (peak.out_of_plane_angle_deg != null ? "  oop=" + peak.out_of_plane_angle_deg + " deg" : "");
    item.appendChild(head);
    item.appendChild(meta);
    for (const warning of (peak.background_warnings || [])) {
      const w = document.createElement("div");
      w.className = "meta";
      w.textContent = "background: " + warning;
      item.appendChild(w);
    }
    if (peak.multiple_scattering_warning) {
      const w = document.createElement("div");
      w.className = "meta";
      w.textContent = "multiple scattering: " + peak.multiple_scattering_warning;
      item.appendChild(w);
    }
    wrap.appendChild(item);
  }
}

function renderReflections() {
  const body = el("reflection_rows");
  body.replaceChildren();
  const family = FAMILY_TESTS[el("refl_family").value] || FAMILY_TESTS.all;
  const rows = reflectionRows.filter(family).slice();
  const key = reflectionSort.key;
  rows.sort((a, b) => {
    const left = key === "hkl" ? a.h * 10000 + a.k * 100 + a.l : a[key];
    const right = key === "hkl" ? b.h * 10000 + b.k * 100 + b.l : b[key];
    return reflectionSort.descending ? right - left : left - right;
  });
  for (const r of rows) {
    const row = document.createElement("tr");
    row.title = "(" + r.h + " " + r.k + " " + r.l + ")  d=" + r.d_spacing.toFixed(4) + " A  |Q|=" + r.q_inv_angstrom.toFixed(4) + "  F2(raw)=" + r.relative_structure_factor.toExponential(3);
    const f2text = r.f2_scaled >= 0.01 || r.f2_scaled === 0 ? r.f2_scaled.toFixed(2) : r.f2_scaled.toExponential(1);
    for (const value of ["(" + r.h + " " + r.k + " " + r.l + ")", r.d_spacing.toFixed(4), r.q_inv_angstrom.toFixed(4), r.two_theta_deg.toFixed(2), r.multiplicity, f2text]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "muted";
    cell.textContent = "No reflections in this plane family within the A4 limit.";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

async function vectorTools() {
  if (!cifText) { el("vector_tools_result").textContent = "Upload a CIF first."; return; }
  const v1 = ["h","k","l"].map((axis) => parseFloat(el("vt1" + axis).value) || 0);
  const v2raw = ["h","k","l"].map((axis) => el("vt2" + axis).value);
  const v2 = v2raw.some((value) => value !== "") ? v2raw.map((value) => parseFloat(value) || 0) : null;
  const payload = {cif_content: cifText, orient1: vec("o1"), orient2: vec("o2"), v1, v2};
  const response = await fetch("api/alignment/vector-tools", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const report = await response.json();
  const wrap = el("vector_tools_result");
  wrap.replaceChildren();
  function line(text, strong) {
    const p = document.createElement("p");
    p.style.margin = "6px 0";
    if (strong) { const b = document.createElement("strong"); b.textContent = text; p.appendChild(b); }
    else p.textContent = text;
    wrap.appendChild(p);
    return p;
  }
  if (report.status !== "success") { line("Failed: " + JSON.stringify(report.warnings || report.status)); return; }
  if (report.angle) {
    if (report.angle.angle_deg != null) {
      line("Angle (" + payload.v1.join(" ") + ") ^ (" + (v2 || []).join(" ") + ") = " + report.angle.angle_deg.toFixed(4) + " deg", true);
      line("|v1| = " + report.angle.v1_modulus_inv_angstrom.toFixed(4) + " 1/A,  |v2| = " + report.angle.v2_modulus_inv_angstrom.toFixed(4) + " 1/A  (general HKL, not plane-constrained)");
    } else line("Angle: " + (report.angle.note || report.angle.status));
  }
  const ortho = report.inplane_orthogonal || {};
  if (ortho.orthogonal_hkl) {
    const rational = ortho.orthogonal_hkl_rational ? "  =  (" + ortho.orthogonal_hkl_rational.join(" ") + ") direction" : "";
    line("In-plane orthogonal of (" + payload.v1.join(" ") + "): (" + ortho.orthogonal_hkl.map((value) => value.toFixed(4)).join(", ") + ")" + rational, true);
    line("Orthogonality check: " + ortho.orthogonality_check_deg.toFixed(3) + " deg; out-of-plane of input: " + ortho.v_out_of_plane_deg.toFixed(3) + " deg. Metric-correct: Euclidean HKL intuition fails for non-orthogonal cells.");
    for (const warning of ortho.warnings || []) line("note: " + warning);
  } else if (ortho.note) line("Orthogonal: " + ortho.note);
  const eq = report.equivalents || {};
  if (eq.status === "success") {
    for (const entry of eq.results || []) {
      const inPlane = entry.equivalents.filter((item) => item.in_plane);
      line("Equivalents of (" + entry.hkl.join(" ") + ")  [mult " + entry.multiplicity + (entry.systematically_absent ? ", systematically absent" : "") + "]:", true);
      line("in plane: " + (inPlane.length ? inPlane.map((item) => "(" + item.hkl.join(" ") + ")").join("  ") : "none"));
      const outPlane = entry.equivalents.filter((item) => !item.in_plane);
      if (outPlane.length) line("out of plane: " + outPlane.map((item) => "(" + item.hkl.join(" ") + ")").join("  "));
    }
  } else if (eq.note) line("Equivalents: " + eq.note);
}

const RING_COLORS = {aluminum: "#777777", copper: "#b87333"};
let zoneReport = null;
let view = {cx: 0, cy: 0, span: 6};

function renderZone(report) {
  zoneReport = report;
  let extent = 0.5;
  for (const p of report.peak_points || []) extent = Math.max(extent, Math.abs(p.x), Math.abs(p.y));
  const bz = report.bz_section || {x: [], y: []};
  for (const v of bz.x.concat(bz.y)) extent = Math.max(extent, Math.abs(v));
  for (const set of report.environment_rings || []) for (const ring of set.rings || []) extent = Math.max(extent, ring.q_inv_angstrom);
  view = {cx: 0, cy: 0, span: extent * 2.3};
  drawZone();
}

function drawZone() {
  const report = zoneReport;
  if (!report) return;
  const svg = el("zone_plot");
  svg.replaceChildren();
  const size = 700, half = size / 2;
  const scale = size / view.span;
  const X = (v) => half + (v - view.cx) * scale, Y = (v) => half - (v - view.cy) * scale;
  const xMin = view.cx - view.span / 2, xMax = view.cx + view.span / 2;
  const yMin = view.cy - view.span / 2, yMax = view.cy + view.span / 2;
  function add(tag, attrs, title, parent) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (title) {
      // data-tip drives the custom cursor tooltip (Safari does not reliably
      // show native SVG <title> tooltips); keep <title> for accessibility.
      node.setAttribute("data-tip", title);
      const t = document.createElementNS(SVGNS, "title");
      t.textContent = title;
      node.appendChild(t);
    }
    (parent || svg).appendChild(node);
    return node;
  }
  function addText(x, y, content, attrs) {
    const text = add("text", Object.assign({x, y, "font-size": "11", fill: "#17212b"}, attrs || {}));
    text.textContent = content;
    return text;
  }
  add("defs", {}).innerHTML = '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7a3294"/></marker>'
    + '<marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52616f"/></marker>';
  add("line", {x1: 0, y1: Y(0), x2: size, y2: Y(0), stroke: "#d7dee7"});
  add("line", {x1: X(0), y1: 0, x2: X(0), y2: size, stroke: "#d7dee7"});

  // Brillouin-zone sections tiled at in-plane reciprocal-lattice points.
  const bz = report.bz_section || {x: [], y: []};
  const tiling = (report.plane_lattice || {}).tiling;
  const centers = [];
  if (tiling) {
    const t1 = tiling.t1, t2 = tiling.t2;
    const det = t1.x * t2.y - t1.y * t2.x;
    const cornerIndices = [];
    for (const [wx, wy] of [[xMin, yMin], [xMin, yMax], [xMax, yMin], [xMax, yMax]]) {
      cornerIndices.push([(wx * t2.y - wy * t2.x) / det, (t1.x * wy - t1.y * wx) / det]);
    }
    const n1Min = Math.floor(Math.min(...cornerIndices.map((c) => c[0]))) - 1, n1Max = Math.ceil(Math.max(...cornerIndices.map((c) => c[0]))) + 1;
    const n2Min = Math.floor(Math.min(...cornerIndices.map((c) => c[1]))) - 1, n2Max = Math.ceil(Math.max(...cornerIndices.map((c) => c[1]))) + 1;
    const maxTiles = 700;
    for (let n1 = n1Min; n1 <= n1Max && centers.length < maxTiles; n1++) {
      for (let n2 = n2Min; n2 <= n2Max && centers.length < maxTiles; n2++) {
        const gx = n1 * t1.x + n2 * t2.x, gy = n1 * t1.y + n2 * t2.y;
        if (gx < xMin - view.span * 0.2 || gx > xMax + view.span * 0.2 || gy < yMin - view.span * 0.2 || gy > yMax + view.span * 0.2) continue;
        const hkl = t1.hkl.map((value, index) => n1 * value + n2 * t2.hkl[index]);
        centers.push({gx, gy, hkl});
      }
    }
  } else {
    centers.push({gx: 0, gy: 0, hkl: [0, 0, 0]});
  }
  if (bz.x.length >= 3) {
    for (const center of centers) {
      const central = Math.abs(center.gx) < 1e-9 && Math.abs(center.gy) < 1e-9;
      const hklLabel = "(" + center.hkl.map((value) => +value.toFixed(4)).join(" ") + ")";
      const path = bz.x.map((x, i) => (i ? "L" : "M") + X(x + center.gx) + "," + Y(bz.y[i] + center.gy)).join(" ") + " Z";
      add("path", {
        d: path,
        fill: central ? "#eef4fb" : "none",
        stroke: central ? "#205493" : "#9fb6d4",
        "stroke-width": central ? 1.6 : 0.8,
        "fill-opacity": central ? 0.7 : 0,
      }, central ? "First Brillouin zone section around (0 0 0)" : "Brillouin-zone section around " + hklLabel);
      add("circle", {cx: X(center.gx), cy: Y(center.gy), r: 3, fill: central ? "#205493" : "#9fb6d4"},
        hklLabel + " reciprocal-lattice point");
    }
  }

  for (const set of report.environment_rings || []) {
    const color = RING_COLORS[set.material_id] || "#999";
    for (const ring of set.rings || []) {
      add("circle", {cx: X(0), cy: Y(0), r: ring.q_inv_angstrom * scale, fill: "none", stroke: color, "stroke-dasharray": "6 5", "stroke-width": 1.4},
        set.label + " (" + ring.hkl.join(" ") + ")  |Q|=" + ring.q_inv_angstrom.toFixed(3) + "  2theta=" + ring.two_theta_deg.toFixed(2));
    }
  }

  // Reciprocal principal axes a*, b*, c* projected into the plane.
  for (const axis of (report.plane_lattice || {}).reciprocal_axes || []) {
    const inPlaneLength = Math.hypot(axis.x, axis.y);
    if (inPlaneLength < 1e-6) continue;
    const dashed = !axis.in_plane;
    add("line", {x1: X(0), y1: Y(0), x2: X(axis.x), y2: Y(axis.y), stroke: "#7a3294", "stroke-width": 1.8,
      "stroke-dasharray": dashed ? "4 4" : "none", "marker-end": "url(#arrow)"},
      axis.label + "  |" + axis.label + "|=" + axis.magnitude_inv_angstrom.toFixed(4) + " 1/A  out-of-plane " + axis.out_of_plane_deg + " deg" + (dashed ? " (projection shown)" : ""));
    addText(X(axis.x) + 6, Y(axis.y) - 4, axis.label + (dashed ? " (proj)" : ""), {fill: "#7a3294", "font-weight": "600"});
  }

  const points = report.peak_points || [];
  let f2max = 0;
  for (const p of points) f2max = Math.max(f2max, p.f2);
  for (const p of points) {
    const strength = f2max > 0 ? Math.sqrt(p.f2 / f2max) : 0.5;
    const radius = 4 + 9 * strength;
    const shade = Math.round(208 - 160 * strength);
    const color = "rgb(" + Math.round(shade * 0.35) + "," + Math.round(shade * 0.55) + "," + shade + ")";
    const attrs = {cx: X(p.x), cy: Y(p.y), r: radius};
    if (p.friedel_mate) { attrs.fill = "none"; attrs.stroke = color; attrs["stroke-width"] = 2; }
    else { attrs.fill = color; attrs["fill-opacity"] = 0.9; }
    add("circle", attrs, p.label + "  F2=" + p.f2.toFixed(3) + "  |Q|=" + p.q_inv_angstrom.toFixed(3) + "  " + p.plane_status);
    if (!p.friedel_mate) addText(X(p.x) + radius + 3, Y(p.y) + 4, p.label);
  }
  // Fixed orientation compass: always visible regardless of pan/zoom.
  // The orthonormal plane basis is screen-aligned by construction, so
  // orient1 always points along +x and orthogonalized orient2 along +y.
  const o1label = "orient1 = [" + vec("o1").map((value) => +value.toFixed(3)).join(" ") + "]";
  const o2label = "orient2 = [" + vec("o2").map((value) => +value.toFixed(3)).join(" ") + "] (orthogonalized)";
  const anchorX = 26, anchorY = 70;
  const boxWidth = Math.max(180, 12 + 8 + 6.5 * Math.max(o1label.length + 11, o2label.length + 2));
  add("rect", {x: 10, y: 14, width: boxWidth, height: 72, fill: "white", "fill-opacity": 0.9, stroke: "#d7dee7", rx: 6});
  add("line", {x1: anchorX, y1: anchorY, x2: anchorX + 56, y2: anchorY, stroke: "#52616f", "stroke-width": 2, "marker-end": "url(#arrow-gray)"}, o1label);
  add("line", {x1: anchorX, y1: anchorY, x2: anchorX, y2: anchorY - 36, stroke: "#52616f", "stroke-width": 2, "marker-end": "url(#arrow-gray)"}, o2label);
  addText(anchorX + 64, anchorY + 4, o1label, {fill: "#17212b", "font-weight": "600"});
  addText(anchorX + 10, anchorY - 38, o2label, {fill: "#17212b", "font-weight": "600"});
  addText(8, size - 10, "span " + view.span.toFixed(2) + " 1/A  |  scroll: pan, pinch or ctrl+scroll: zoom, drag: pan, double-click: reset", {fill: "#52616f"});
}

function initZoneInteractions() {
  const svg = el("zone_plot");
  let dragging = null;
  const tooltip = document.createElement("div");
  tooltip.style.cssText = "position: fixed; display: none; background: #17212b; color: #f7f9fb; padding: 4px 8px; border-radius: 5px; font-size: 12px; pointer-events: none; z-index: 50; max-width: 360px;";
  document.body.appendChild(tooltip);
  function hideTooltip() { tooltip.style.display = "none"; }
  svg.addEventListener("wheel", (event) => {
    if (!zoneReport) return;
    event.preventDefault();
    hideTooltip();
    const rect = svg.getBoundingClientRect();
    if (event.ctrlKey || event.metaKey) {
      // Pinch gesture (macOS sends ctrlKey) or ctrl/cmd + scroll: zoom around the cursor.
      const fx = (event.clientX - rect.left) / rect.width, fy = (event.clientY - rect.top) / rect.height;
      const wx = view.cx + (fx - 0.5) * view.span, wy = view.cy - (fy - 0.5) * view.span;
      const factor = Math.exp(event.deltaY * 0.01);
      view.span = Math.min(60, Math.max(0.3, view.span * factor));
      view.cx = wx - (fx - 0.5) * view.span;
      view.cy = wy + (fy - 0.5) * view.span;
    } else {
      // Two-finger scroll / wheel: translate the view (the page does not scroll).
      view.cx += event.deltaX * view.span / rect.width;
      view.cy -= event.deltaY * view.span / rect.height;
    }
    drawZone();
  }, {passive: false});
  svg.addEventListener("pointerdown", (event) => {
    dragging = {x: event.clientX, y: event.clientY, cx: view.cx, cy: view.cy};
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (dragging && zoneReport) {
      hideTooltip();
      const rect = svg.getBoundingClientRect();
      view.cx = dragging.cx - (event.clientX - dragging.x) * view.span / rect.width;
      view.cy = dragging.cy + (event.clientY - dragging.y) * view.span / rect.height;
      drawZone();
      return;
    }
    const target = event.target.closest ? event.target.closest("[data-tip]") : null;
    if (target) {
      tooltip.textContent = target.getAttribute("data-tip");
      tooltip.style.display = "block";
      tooltip.style.left = (event.clientX + 14) + "px";
      tooltip.style.top = (event.clientY + 14) + "px";
    } else {
      hideTooltip();
    }
  });
  svg.addEventListener("pointerup", () => { dragging = null; });
  svg.addEventListener("pointerleave", () => { dragging = null; hideTooltip(); });
  svg.addEventListener("dblclick", () => { if (zoneReport) renderZone(zoneReport); });
}
initZoneInteractions();
</script>
"""


def _review_dashboard_body(root: Path, *, selected_store: str | None = None) -> str:
    stores = _discover_review_stores(root)
    review_runs = [
        run
        for run in _run_records(root)
        if any(term in run.get("name", "").lower() for term in ["review", "dashboard", "packet"])
    ][:25]
    run_rows = "".join(_review_dashboard_run_row(run) for run in review_runs)
    store_options = "".join(f"<option value=\"{html.escape(item['path'])}\">{html.escape(item['label'])}</option>" for item in stores)
    selected = html.escape(selected_store or (stores[0]["path"] if stores else ""))
    return f"""
<p>This is the student-facing entry point for review and labeling work. It links to the task-specific dashboards where candidates are reviewed, and to the unified label-store summary/ingest page.</p>
<section class="card">
  <h2>Start Here</h2>
  <ol>
    <li>Open a task-specific review dashboard or packet from the recent runs below.</li>
    <li>For paper links, open the candidate's <strong>Review packet with plots</strong> and <strong>Open paper from local cache/Vista</strong> links.</li>
    <li>Set accepted/rejected/duplicate/needs-more-evidence fields in the task dashboard.</li>
    <li>Export reviewed JSONL.</li>
    <li>Use <a href="/labels">Review Labels</a> to ingest the export into the unified store.</li>
  </ol>
  <p><a href="/documentation">Open Review Labeling Dashboard docs</a></p>
</section>
<section class="card">
  <h2>Unified Label Store</h2>
  <p class="muted">This summarizes or ingests labels. The actual scientific review usually happens in a task-specific dashboard/packet.</p>
  <datalist id="review_stores">{store_options}</datalist>
  <label>Review outcome store</label>
  <input id="review_store" list="review_stores" value="{selected}" placeholder="local_runs/.../review_outcomes.sqlite">
  <button onclick="window.location.href='/labels?store=' + encodeURIComponent(document.getElementById('review_store').value)">Open label-store summary/ingest</button>
</section>
<section class="card">
  <h2>Recent Review Dashboards And Packets</h2>
  <p class="muted">Generated artifacts under <code>{html.escape(str(root))}</code>. If this list is empty, ask for or generate a review dashboard first.</p>
  <table><thead><tr><th>Name</th><th>Modified</th><th>Labels</th><th>Path</th></tr></thead><tbody>{run_rows or '<tr><td colspan="4">No review dashboards or packets found under local_runs yet.</td></tr>'}</tbody></table>
</section>
<section class="card">
  <h2>If This Page Looks Old</h2>
  <p>After pulling new code, rerun the bootstrap so the editable <code>tas-copilot</code> entry point points at the current checkout:</p>
  <pre>git pull
./scripts/bootstrap_student.sh
conda activate tascopilot-student
tas-copilot serve --host 127.0.0.1 --port 8765</pre>
</section>
"""


def _review_dashboard_run_row(run: dict[str, str]) -> str:
    store_link = f"<a href=\"/review?store={html.escape(run.get('review_store') or '')}\">Open store</a>" if run.get("review_store") else ""
    return (
        "<tr>"
        f"<td><a href=\"{html.escape(run.get('href') or '#')}\">{html.escape(run.get('name') or '')}</a></td>"
        f"<td>{html.escape(run.get('modified') or '')}</td>"
        f"<td>{store_link}</td>"
        f"<td><code>{html.escape(run.get('path') or '')}</code></td>"
        "</tr>"
    )


def _artifact_links(result: CopilotAskResponse, runs_root: Path) -> list[dict[str, str]]:
    """Convert generated artifact paths under local_runs into /local-runs links, HTML first."""

    links: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(label: str, value: Any) -> None:
        if isinstance(value, str):
            if "/" not in value:
                return
            path = Path(value)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                return
            try:
                relative = path.resolve().relative_to(runs_root)
            except ValueError:
                return
            href = "/local-runs/" + "/".join(relative.parts)
            if href not in seen:
                seen.add(href)
                links.append({"label": label or relative.parts[-1], "href": href})
        elif isinstance(value, dict):
            for key, item in value.items():
                _add(f"{label}.{key}" if label else str(key), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                _add(f"{label}[{index}]", item)

    _add("", dict(result.artifacts or {}))
    for call in result.routed_workflows:
        _add(call.workflow_name, dict(call.artifacts or {}))
    links.sort(key=lambda link: (not link["href"].endswith(".html"), link["label"]))
    return links


def _recent_data_roots(runs_root: Path, *, limit: int = 12, max_lines: int = 500) -> list[str]:
    """Distinct still-existing data roots from recent web audit events, newest first."""

    audit_file = runs_root.resolve().parent / "data" / "audit" / "web.jsonl"
    if not audit_file.exists():
        return []
    try:
        lines = audit_file.read_text(encoding="utf-8").splitlines()[-max_lines:]
    except OSError:
        return []
    roots: list[str] = []
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        root = event.get("data_root")
        if not isinstance(root, str) or not root or root in roots:
            continue
        if not Path(root).expanduser().is_dir():
            continue
        roots.append(root)
        if len(roots) >= limit:
            break
    return roots


def _copilot_cli(request: dict[str, Any]) -> str:
    parts = [
        "python -m tas_copilot_core.cli workflow copilot-ask",
        "--question",
        shlex.quote(str(request.get("question") or "")),
        "--output-dir",
        shlex.quote(str(request.get("output_dir") or "")),
    ]
    for key, flag in [
        ("data_root", "--data-root"),
        ("corpus_catalog", "--corpus-catalog"),
        ("data_catalog", "--data-catalog"),
        ("precedent_index", "--precedent-index"),
        ("paper_id", "--paper-id"),
        ("figure_id", "--figure-id"),
        ("fixture", "--fixture"),
        ("proposal_evidence_dir", "--proposal-evidence-dir"),
        ("actual_evidence_dir", "--actual-evidence-dir"),
    ]:
        if request.get(key):
            parts.extend([flag, shlex.quote(str(request[key]))])
    if request.get("intent"):
        # The CLI has no --intent flag; explicit intents ride the --json passthrough.
        parts.extend(["--json", shlex.quote(json.dumps({"intent": str(request["intent"])}))])
    return " ".join(parts)


def _nearest_review_store(path: Path) -> Path | None:
    current = path.parent
    for parent in [current, *current.parents]:
        candidate = parent / "review_outcomes.sqlite"
        if candidate.exists():
            return candidate
    return None


def _discover_review_stores(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    stores = []
    for path in sorted(root.rglob("review_outcomes.sqlite"), key=lambda item: item.stat().st_mtime, reverse=True)[:50]:
        stores.append(
            {
                "path": str(path),
                "label": f"{path.parent.name} ({datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')})",
            }
        )
    return stores


def _resolve_paper_document(paper_id: str, *, runs_root: Path, catalog: str | None = None) -> dict[str, Any] | None:
    if not paper_id or "/" in paper_id or "\\" in paper_id:
        return None
    catalog_paths = _paper_catalog_document_files(runs_root, catalog=catalog)
    matches: list[dict[str, Any]] = []
    for documents_path in catalog_paths:
        try:
            lines = documents_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if paper_id not in {str(row.get("paper_id") or ""), str(row.get("document_id") or "")}:
                continue
            local_path = str(row.get("local_path") or "")
            if not local_path:
                continue
            row = dict(row)
            row["_catalog_documents_path"] = str(documents_path)
            matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda item: (str(item.get("ingest_status") or ""), str(item.get("_catalog_documents_path") or "")), reverse=True)
    return matches[0]


def _paper_catalog_document_files(runs_root: Path, *, catalog: str | None = None) -> list[Path]:
    if catalog:
        candidate = Path(catalog).expanduser()
        if candidate.is_dir():
            candidate = candidate / "documents.jsonl"
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return [candidate] if candidate.exists() and candidate.name == "documents.jsonl" else []
    roots = [runs_root / "corpus_catalogs", Path.cwd() / "local_runs" / "corpus_catalogs"]
    paths: list[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/documents.jsonl"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return paths[:200]


def _ensure_paper_cached(record: dict[str, Any], *, runs_root: Path, fetch: bool) -> dict[str, Any]:
    source = str(record.get("local_path") or "")
    if not source:
        return {"status": "missing_input", "reason": "document record has no local_path"}
    cache_path = _paper_cache_path(record, runs_root=runs_root)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {"status": "ok", "cache_status": "hit", "path": str(cache_path), "source_path": source}
    source_path = Path(source).expanduser()
    if source_path.exists() and source_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, cache_path)
        return {"status": "ok", "cache_status": "copied_local", "path": str(cache_path), "source_path": source}
    if not fetch:
        return {"status": "missing", "cache_status": "miss", "source_path": source, "reason": "fetch disabled and local source is absent"}
    if not _is_allowed_remote_paper_path(source):
        return {"status": "blocked", "cache_status": "miss", "source_path": source, "reason": "source path is not an allowlisted remote paper path"}
    fetched = _fetch_remote_paper(source, cache_path)
    if fetched["status"] == "ok":
        fetched.update({"path": str(cache_path), "source_path": source})
    return fetched


def _paper_cache_probe(record: dict[str, Any], *, runs_root: Path) -> dict[str, Any]:
    source = str(record.get("local_path") or "")
    cache_path = _paper_cache_path(record, runs_root=runs_root)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {"status": "ok", "cache_status": "hit", "path": str(cache_path), "source_path": source}
    source_path = Path(source).expanduser()
    if source_path.exists() and source_path.is_file():
        return {"status": "available_local", "cache_status": "miss", "source_path": source}
    if _is_allowed_remote_paper_path(source):
        return {"status": "available_remote", "cache_status": "miss", "source_path": source, "remote_host": os.environ.get("TAS_COPILOT_PAPER_FETCH_HOST", "vista")}
    return {"status": "missing", "cache_status": "miss", "source_path": source}


def _paper_cache_path(record: dict[str, Any], *, runs_root: Path) -> Path:
    paper_id = str(record.get("paper_id") or record.get("document_id") or "paper")
    source_name = Path(str(record.get("local_path") or "paper.pdf")).name or "paper.pdf"
    safe_id = _safe_file_component(paper_id)
    safe_name = _safe_file_component(source_name)
    return (runs_root / "paper_cache" / safe_id / safe_name).resolve()


def _safe_file_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return cleaned.strip("._")[:180] or "item"


def _is_allowed_remote_paper_path(path: str) -> bool:
    normalized = str(path)
    return normalized.startswith(("/work/", "/work2/", "/scratch/", "/corral-repl/")) and normalized.lower().endswith(".pdf")


def _fetch_remote_paper(remote_path: str, cache_path: Path) -> dict[str, Any]:
    host = os.environ.get("TAS_COPILOT_PAPER_FETCH_HOST", "vista")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ssh_args = "ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPersist=600"
    if shutil.which("rsync"):
        command = ["rsync", "-a", "--protect-args", "-e", ssh_args, f"{host}:{remote_path}", str(cache_path)]
    elif shutil.which("scp"):
        command = ["scp", "-o", "BatchMode=yes", f"{host}:{remote_path}", str(cache_path)]
    else:
        return {"status": "unavailable", "cache_status": "miss", "reason": "neither rsync nor scp is available"}
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cache_status": "miss", "remote_host": host, "reason": "paper fetch timed out"}
    if completed.returncode != 0:
        return {
            "status": "fetch_failed",
            "cache_status": "miss",
            "remote_host": host,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-1000:],
        }
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return {"status": "fetch_failed", "cache_status": "miss", "remote_host": host, "reason": "fetch produced no cached file"}
    return {"status": "ok", "cache_status": "fetched_remote", "remote_host": host}


def _public_paper_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": record.get("paper_id"),
        "document_id": record.get("document_id"),
        "title": record.get("title"),
        "copyright_status": record.get("copyright_status"),
        "ingest_status": record.get("ingest_status"),
        "extraction_method": record.get("extraction_method"),
        "local_path": record.get("local_path"),
        "catalog_documents_path": record.get("_catalog_documents_path"),
    }


def _append_web_audit(root: Path, event: dict[str, Any]) -> None:
    audit_dir = root.resolve().parent / "data" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "tas-copilot-web",
        "runs_root": str(root),
        **event,
    }
    with (audit_dir / "web.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
