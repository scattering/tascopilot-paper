from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient

from tas_copilot_contracts import ReviewLabel
from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore
from tas_copilot_core.web_app import create_app, run_server


def test_web_app_front_door_and_intents(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "TASCopilot Workbench" in response.text
    assert "no live control" in response.text.lower()
    assert "Review Dashboard" in response.text
    assert 'href="/review"' in response.text
    assert "Documentation" in response.text
    assert 'href="/documentation"' in response.text
    assert client.get("/review").status_code == 200
    assert client.get("/labeling").status_code == 200
    assert client.get("/documentation").status_code == 200

    intents = client.get("/api/intents").json()
    assert any(item["intent"] == "experiment_review" for item in intents["intents"])
    assert client.get("/healthz").text == "ok"
    assert client.get("/api-docs").status_code == 200
    assert client.get("/docs").status_code == 404


def test_web_app_finds_repo_docs_when_started_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(local_runs_dir=tmp_path / "runs")
    client = TestClient(app)

    response = client.get("/docs/reviewer_testing.md")

    assert response.status_code == 200
    assert "Reviewer Testing" in response.text
    docs_front = client.get("/documentation")
    assert docs_front.status_code == 200
    assert "/docs/reviewer_testing.md" in docs_front.text


def test_web_app_ask_routes_to_copilot(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.post(
        "/api/ask",
        json={"question": "Could this narrow feature be a spurion?", "output_dir": str(tmp_path / "runs" / "ask")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "missing_input"
    assert payload["result"]["route"]["intent"] == "spurion_diagnostic_survey"
    assert "workflow copilot-ask" in payload["equivalent_cli"]
    assert "--question 'Could this narrow feature be a spurion?'" in payload["equivalent_cli"]


def test_web_app_ask_defaults_to_timestamped_local_run_dir(tmp_path):
    runs = tmp_path / "runs"
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.post("/api/ask", json={"question": "What can you do?"})

    assert response.status_code == 200
    output_dir = Path(response.json()["result"]["artifacts"]["json"]).parent
    assert output_dir.parent.name == "web_copilot_ask"
    assert output_dir.is_relative_to(runs)


def test_web_app_rejects_output_dir_outside_local_runs(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.post("/api/ask", json={"question": "What can you do?", "output_dir": str(tmp_path / "elsewhere")})

    assert response.status_code == 400
    assert "output_dir must live under" in response.json()["detail"]


def test_web_app_reports_and_label_endpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    report_dir = runs / "sample"
    report_dir.mkdir(parents=True)
    report = report_dir / "report.html"
    report.write_text("<h1>Report</h1>", encoding="utf-8")
    store_path = runs / "review_outcomes.sqlite"
    ReviewOutcomeSQLiteStore(store_path).upsert_labels(
        [ReviewLabel(object_type="paper_experiment_link", object_id="paper-1", source_workflow="link-paper-to-experiments", verdict="positive")]
    )
    labels_path = tmp_path / "reviewed.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "paper_id": "paper-2",
                "candidate_experiment_id": "exp-2",
                "human_review_status": "accepted",
                "reviewer": "tester",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    assert client.get("/runs").status_code == 200
    run_records = client.get("/api/runs").json()["runs"]
    assert any(item["href"].startswith("/local-runs/") for item in run_records)
    assert any(item.get("review_store") == str(store_path) for item in run_records)
    assert client.get("/tools").status_code == 200
    labels_page = client.get("/labels")
    assert labels_page.status_code == 200
    assert str(store_path) in labels_page.text
    stores = client.get("/api/review-labels/stores").json()["stores"]
    assert stores[0]["path"] == str(store_path)

    summary = client.post("/api/review-labels/summary", json={"store": str(store_path), "sample_limit": 5})
    assert summary.status_code == 200
    assert summary.json()["label_count"] == 1

    ingest = client.post(
        "/api/review-labels/ingest",
        json={"labels": str(labels_path), "store": str(store_path), "output_dir": str(runs / "label_ingest")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["ingested_label_count"] == 1
    assert (tmp_path / "data" / "audit" / "web.jsonl").exists()


def test_web_app_serves_paper_from_local_cache(tmp_path):
    runs = tmp_path / "runs"
    catalog = runs / "corpus_catalogs" / "cat"
    catalog.mkdir(parents=True)
    pdf = tmp_path / "private_paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nsynthetic private fixture\n")
    (catalog / "documents.jsonl").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "paper_id": "paper-1",
                "title": "Synthetic paper",
                "local_path": str(pdf),
                "copyright_status": "private_local_only",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    info = client.get("/api/papers/paper-1")
    assert info.status_code == 200
    assert info.json()["cache"]["status"] == "available_local"

    response = client.get("/papers/paper-1")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert (runs / "paper_cache" / "paper-1" / "private_paper.pdf").exists()

    cached = client.get("/api/papers/paper-1")
    assert cached.json()["cache"]["cache_status"] == "hit"


def _write_chunk_review_fixture(tmp_path):
    runs = tmp_path / "runs"
    catalog = runs / "corpus_catalogs" / "cat_docling"
    catalog.mkdir(parents=True)
    pdf = tmp_path / "synthetic_paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nsynthetic private fixture\n")
    (catalog / "papers.jsonl").write_text(
        json.dumps({"paper_id": "paper-1", "title": "Synthetic spin waves in FeO", "authors": ["A. Student", "B. Scientist"], "year": 2024, "doi": "10.0/synthetic"}) + "\n",
        encoding="utf-8",
    )
    (catalog / "documents.jsonl").write_text(
        json.dumps({"document_id": "paper-1", "paper_id": "paper-1", "title": "Synthetic spin waves in FeO", "local_path": str(pdf)}) + "\n",
        encoding="utf-8",
    )
    chunks = [
        {"chunk_id": "chunk-a", "paper_id": "paper-1", "chunk_type": "abstract", "char_start": 0, "char_end": 30, "text": "We measured synthetic spin waves.", "token_count": 6},
        {"chunk_id": "chunk-b", "paper_id": "paper-1", "chunk_type": "caption", "char_start": 31, "char_end": 60, "figure_id": "fig-2", "text": "FIG. 2. Synthetic dispersion.", "token_count": 5},
    ]
    (catalog / "source_chunks.jsonl").write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")
    run_dir = runs / "vista_corpus_extractor_outputs" / "run_2026"
    md_dir = run_dir / "outputs" / "docling" / "md"
    md_dir.mkdir(parents=True)
    (md_dir / "0001_synthetic_paper.md").write_text("# Synthetic spin waves in FeO\n\nFIG. 2. Synthetic dispersion.\n", encoding="utf-8")
    (run_dir / "outputs" / "docling" / "summary.json").write_text(
        json.dumps([{"index": 1, "pdf_name": "synthetic_paper.pdf", "backend": "docling", "markdown_path": "/work/remote/outputs/docling/md/0001_synthetic_paper.md"}]),
        encoding="utf-8",
    )
    tei_dir = run_dir / "outputs" / "grobid" / "tei"
    tei_dir.mkdir(parents=True)
    (tei_dir / "0001_synthetic_paper.tei.xml").write_text(
        "<TEI><teiHeader><title>Synthetic spin waves in FeO</title></teiHeader><text><body><p>GROBID body about synthetic dispersion.</p></body></text></TEI>",
        encoding="utf-8",
    )
    (run_dir / "outputs" / "grobid" / "summary.json").write_text(
        json.dumps([{"index": 1, "pdf_name": "synthetic_paper.pdf", "backend": "grobid", "tei_path": "/work/remote/outputs/grobid/tei/0001_synthetic_paper.tei.xml"}]),
        encoding="utf-8",
    )
    return runs, catalog, run_dir


def test_web_app_chunk_review_surface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs, catalog, run_dir = _write_chunk_review_fixture(tmp_path)
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    page = client.get("/chunks")
    assert page.status_code == 200
    assert "Extraction-Quality Label" in page.text

    sources = client.get("/api/chunks/sources").json()
    assert sources["catalogs"][0]["path"] == str(catalog)
    assert sources["extractor_runs"][0]["path"] == str(run_dir)
    assert sources["extractor_runs"][0]["backends"] == ["docling", "grobid"]

    table = client.get("/api/chunks/papers", params={"catalog": str(catalog)}).json()
    assert table["paper_count"] == 1
    row = table["papers"][0]
    assert row["title"] == "Synthetic spin waves in FeO"
    assert row["chunk_count"] == 2
    assert row["chunk_type_counts"]["caption"] == 1
    assert row["pdf_name"] == "synthetic_paper.pdf"

    detail = client.get(
        "/api/chunks/paper/paper-1",
        params={"catalog": str(catalog), "run": str(run_dir)},
    ).json()
    assert detail["abstract"] == "We measured synthetic spin waves."
    assert [chunk["chunk_id"] for chunk in detail["chunks"]] == ["chunk-a", "chunk-b"]
    outputs = detail["extractor_outputs"]
    assert outputs["docling"]["status"] == "ok"
    assert "Synthetic dispersion" in outputs["docling"]["text"]
    assert outputs["grobid"]["status"] == "ok"
    assert outputs["grobid"]["kind"] == "tei_text"
    assert "GROBID body" in outputs["grobid"]["text"]
    assert "<p>" not in outputs["grobid"]["text"]
    docling_chunk_types = {chunk["chunk_type"] for chunk in outputs["docling"]["chunks"]}
    assert "figure_caption" in docling_chunk_types
    assert outputs["grobid"]["chunks"], "grobid text should chunk through the same chunker"
    assert all(chunk.get("text") for chunk in outputs["grobid"]["chunks"])

    assert client.get("/api/chunks/papers", params={"catalog": str(tmp_path / "nope")}).status_code == 404


def test_web_app_chunk_review_labels_roundtrip_to_unified_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs, catalog, run_dir = _write_chunk_review_fixture(tmp_path)
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    saved = client.post(
        "/api/chunks/labels",
        json={
            "paper_id": "paper-1",
            "status": "caption_table_failure",
            "catalog": str(catalog),
            "extractor_run": str(run_dir),
            "best_backend": "grobid",
            "comment": "Figure 2 caption merged with body text.",
            "reviewer": "reviewer-1",
        },
    )
    assert saved.status_code == 200
    payload = saved.json()
    assert payload["verdict"] == "negative"
    labels_path = Path(payload["labels_jsonl"])
    assert labels_path.exists()
    row = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["object_type"] == "chunk_extraction_quality"
    assert row["type_specific_status"] == "caption_table_failure"
    assert row["review_notes"] == "Figure 2 caption merged with body text."
    assert row["correction_status_snapshot"]["reviewer"] == "reviewer-1"
    assert row["correction_status_snapshot"]["best_backend"] == "grobid"

    store_path = runs / "review_outcomes.sqlite"
    ingest = client.post(
        "/api/review-labels/ingest",
        json={"labels": str(labels_path), "store": str(store_path), "output_dir": str(runs / "chunk_label_ingest")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["ingested_label_count"] == 1
    assert ingest.json()["by_object_type"] == {"chunk_extraction_quality": 1}

    rejected = client.post("/api/chunks/labels", json={"paper_id": "paper-1", "status": "not_a_status"})
    assert rejected.status_code == 400


def test_web_app_chunk_review_prior_labels_and_link_verdicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs, catalog, run_dir = _write_chunk_review_fixture(tmp_path)
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    empty = client.get("/api/chunks/labels", params={"paper_id": "paper-1"}).json()
    assert empty["label_count"] == 0

    saved = client.post(
        "/api/chunks/labels",
        json={
            "paper_id": "paper-1",
            "status": "minor_issues",
            "catalog": str(catalog),
            "comment": "Abstract truncated mid-sentence.",
            "reviewer": "reviewer-1",
        },
    )
    assert saved.status_code == 200

    fetched = client.get("/api/chunks/labels", params={"paper_id": "paper-1", "catalog": str(catalog)}).json()
    assert fetched["label_count"] == 1
    label = fetched["labels"][0]
    assert label["status"] == "minor_issues"
    assert label["comment"] == "Abstract truncated mid-sentence."
    assert label["reviewer"] == "reviewer-1"
    assert label["labeled_at"]

    # labels recorded against a different catalog do not bleed in
    other_catalog = client.get("/api/chunks/labels", params={"paper_id": "paper-1", "catalog": str(tmp_path / "other")}).json()
    assert other_catalog["label_count"] == 0
    assert other_catalog["carryover_label_count"] == 1
    assert other_catalog["carryover_labels"][0]["catalog_mismatch"] is True
    assert other_catalog["carryover_labels"][0]["status"] == "minor_issues"
    other_paper = client.get("/api/chunks/labels", params={"paper_id": "paper-2"}).json()
    assert other_paper["label_count"] == 0

    # paper-experiment link verdicts join from a review-outcomes store
    from tas_copilot_contracts import ReviewLabel
    from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore

    store_dir = runs / "review_outcomes_full_corpus"
    store_dir.mkdir()
    store_path = store_dir / "review_outcomes.sqlite"
    ReviewOutcomeSQLiteStore(store_path).upsert_labels(
        [
            ReviewLabel(
                object_type="paper_experiment_link",
                object_id="paper-1",
                object_secondary_id="exp-1",
                source_workflow="link-paper-to-experiments",
                verdict="positive",
                type_specific_status="accepted",
                type_specific_allowed=["accepted", "rejected", "inconclusive"],
            ),
            ReviewLabel(
                object_type="paper_experiment_link",
                object_id="paper-1",
                object_secondary_id="exp-2",
                source_workflow="link-paper-to-experiments",
                verdict="negative",
                type_specific_status="rejected",
                type_specific_allowed=["accepted", "rejected", "inconclusive"],
            ),
        ]
    )
    sources = client.get("/api/chunks/sources").json()
    assert str(store_path.resolve()) in [store["path"] for store in sources["review_stores"]]

    table = client.get("/api/chunks/papers", params={"catalog": str(catalog), "link_store": str(store_path)}).json()
    verdicts = table["papers"][0]["link_verdicts"]
    assert verdicts == {"total": 2, "accepted": 1, "rejected": 1}

    unknown = client.get("/api/chunks/papers", params={"catalog": str(catalog), "link_store": str(tmp_path / "evil.sqlite")})
    assert unknown.status_code == 400


def test_web_app_validates_request_schema(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.post("/api/ask", json={"question": "What can you do?", "top_k": "abc"})

    assert response.status_code == 422


def test_web_app_rejects_nonlocal_bind():
    with pytest.raises(ValueError, match="local-only"):
        run_server(host="0.0.0.0")


def test_facts_page_lists_catalogs_and_serves_interactive_review(tmp_path):
    import json as _json

    from fastapi.testclient import TestClient

    from tas_copilot_core.web_app import create_app

    runs_root = tmp_path / "local_runs"
    catalog = runs_root / "corpus_catalogs" / "demo_catalog"
    catalog.mkdir(parents=True)
    (catalog / "papers.jsonl").write_text(
        _json.dumps({"paper_id": "doc-a", "title": "Paper A"}) + "\n" + _json.dumps({"paper_id": "doc-b", "title": "Paper B"}) + "\n",
        encoding="utf-8",
    )
    (catalog / "source_chunks.jsonl").write_text(
        _json.dumps({"chunk_id": "chunk-a-0", "chunk_type": "methods", "text": "Measured on BT-7 at 14.7 meV."}) + "\n",
        encoding="utf-8",
    )
    (catalog / "paper_facts.jsonl").write_text(
        _json.dumps(
            {
                "fact_id": "paper-fact-1",
                "paper_id": "doc-a",
                "fact_type": "instrument",
                "answer_status": "supported",
                "confidence": 0.9,
                "value": {"text": "BT-7"},
                "evidence": [{"chunk_id": "chunk-a-0", "quote": "Measured on BT-7"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(local_runs_dir=runs_root))

    assert 'href="/facts"' in client.get("/").text  # discoverable from nav
    catalogs = client.get("/api/facts/catalogs").json()["catalogs"]
    assert any(entry["path"] == str(catalog.resolve()) for entry in catalogs)

    table = client.get("/api/facts/papers", params={"catalog": str(catalog)}).json()
    by_id = {row["paper_id"]: row for row in table["papers"]}
    assert by_id["doc-a"]["fact_count"] == 1
    assert by_id["doc-b"]["fact_count"] == 0  # extraction gap is visible, not hidden
    assert "extract-paper-facts" in table["extract_command_hint"]

    review = client.get("/facts/review", params={"catalog": str(catalog), "paper_id": "doc-a"}).text
    assert "select class='verdict'" in review and "Export labels JSONL" in review
    assert "Measured on BT-7 at 14.7 meV." in review  # chunk content served

    assert client.get("/api/facts/papers", params={"catalog": "/etc"}).status_code == 400


def test_every_shipped_doc_is_listed_and_served(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    client = TestClient(create_app(local_runs_dir=tmp_path / "runs", docs_dir=repo / "docs"))
    page = client.get("/documentation")
    for path in (repo / "docs").glob("*.md"):
        href = "/docs/" + path.name
        assert href in page.text
        assert client.get(href).status_code == 200


def test_experiments_page_discovers_report_surfaces(tmp_path):
    import json as _json

    from fastapi.testclient import TestClient

    from tas_copilot_core.web_app import create_app

    runs_root = tmp_path / "local_runs"
    days = runs_root / "myexp_day_reports"
    (days / "2018-07-07/report").mkdir(parents=True)
    (days / "index.html").write_text("<html></html>", encoding="utf-8")
    (days / "experiment_day_index.json").write_text("{}", encoding="utf-8")
    story = days / "story"
    story.mkdir()
    (story / "experiment_story_report.html").write_text("<html></html>", encoding="utf-8")
    dash = runs_root / "myexp_day_reports/2018-07-07/dashboard"
    dash.mkdir(parents=True)
    (dash / "dashboard.html").write_text("<html></html>", encoding="utf-8")
    (dash / "dashboard_manifest.json").write_text("{}", encoding="utf-8")
    plan = runs_root / "scan_plan_run"
    plan.mkdir()
    (plan / "dispersion_scan_plan.html").write_text("<html></html>", encoding="utf-8")
    # An index.html WITHOUT the day-index marker must not appear.
    (runs_root / "random_run").mkdir()
    (runs_root / "random_run/index.html").write_text("<html></html>", encoding="utf-8")

    client = TestClient(create_app(local_runs_dir=runs_root))

    assert 'href="/experiments"' in client.get("/").text
    reports = client.get("/api/experiments/reports").json()["reports"]
    by_kind = {}
    for report in reports:
        by_kind.setdefault(report["kind"], []).append(report["href"])
    assert by_kind["day_index"] == ["/local-runs/myexp_day_reports/index.html"]
    assert by_kind["story"] == ["/local-runs/myexp_day_reports/story/experiment_story_report.html"]
    assert by_kind["dashboard"] == ["/local-runs/myexp_day_reports/2018-07-07/dashboard/dashboard.html"]
    assert by_kind["scan_plan"] == ["/local-runs/scan_plan_run/dispersion_scan_plan.html"]
    # The static mount actually serves the linked artifact.
    assert client.get("/local-runs/myexp_day_reports/index.html").status_code == 200


def test_web_app_nav_groups_by_audience(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    page = client.get("/").text

    for group in ("Operate", "Review", "Reference"):
        assert f'<span class="nav-label">{group}</span>' in page
    assert 'href="/workspace"' in page
    # Student review surfaces stay reachable from the grouped nav and landing.
    assert 'href="/review"' in page
    assert 'href="/chunks"' in page
    assert 'href="/facts"' in page


def test_web_app_ask_accepts_explicit_intent(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    # No question: the intent's canonical example question is substituted.
    response = client.post("/api/ask", json={"intent": "copilot_status"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["route"]["intent"] == "copilot_status"
    assert payload["result"]["status"] == "ok"
    # Explicit intents ride the CLI --json passthrough in the echoed command.
    assert "--json" in payload["equivalent_cli"]
    assert "copilot_status" in payload["equivalent_cli"]
    # Generated artifacts under local_runs come back as /local-runs links.
    assert payload["artifact_links"]
    for link in payload["artifact_links"]:
        assert link["href"].startswith("/local-runs/")
        assert client.get(link["href"]).status_code == 200


def test_web_app_ask_explicit_intent_reports_missing_inputs(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    response = client.post("/api/ask", json={"intent": "experiment_dashboard"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "missing_input"
    assert payload["result"]["route"]["intent"] == "experiment_dashboard"
    assert "data_root" in payload["result"]["missing_inputs"]


def test_web_app_ask_rejects_unknown_intent_and_empty_request(tmp_path):
    app = create_app(local_runs_dir=tmp_path / "runs", docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    unknown = client.post("/api/ask", json={"intent": "launch_the_beam"})
    assert unknown.status_code == 400
    assert "unknown intent" in unknown.json()["detail"]

    empty = client.post("/api/ask", json={})
    assert empty.status_code == 400
    assert "question or intent" in empty.json()["detail"]


def test_workspace_page_and_recent_data_roots(tmp_path):
    runs = tmp_path / "runs"
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    page = client.get("/workspace")
    assert page.status_code == 200
    assert "Experiment Workspace" in page.text
    assert "data_root" in page.text
    assert "/api/workspace/data-roots" in page.text

    # No audit history yet.
    assert client.get("/api/workspace/data-roots").json()["data_roots"] == []

    # An ask that names an existing data root surfaces it for reuse;
    # a vanished directory does not.
    existing = tmp_path / "expt"
    existing.mkdir()
    client.post("/api/ask", json={"question": "What can you do?", "data_root": str(existing)})
    client.post("/api/ask", json={"question": "What can you do?", "data_root": str(tmp_path / "gone")})

    roots = client.get("/api/workspace/data-roots").json()["data_roots"]
    assert roots == [str(existing)]
