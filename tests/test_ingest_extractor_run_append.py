"""Append mode of `corpus ingest-extractor-run`: a small extractor run merges into an existing
catalog, replacing only its own papers, so corpus updates never rebuild everything."""
import json
from pathlib import Path

from tas_copilot_core.corpus.pilot import ingest_extractor_run


def _run(tmp_path: Path, name: str, papers: dict[str, str]) -> Path:
    run = tmp_path / name
    md_dir = run / "outputs" / "docling" / "md"
    md_dir.mkdir(parents=True)
    rows = []
    for idx, (pdf_name, text) in enumerate(papers.items(), start=1):
        md = md_dir / f"{idx:03d}_{Path(pdf_name).stem}.md"
        md.write_text(text)
        rows.append({"index": idx, "pdf": f"/pdfs/{pdf_name}", "pdf_name": pdf_name, "backend": "docling", "markdown_path": str(md), "status": "success"})
    (run / "outputs" / "docling" / "summary.json").write_text(json.dumps(rows))
    return run


def _ids(catalog: Path) -> set[str]:
    return {json.loads(line)["paper_id"] for line in (catalog / "papers.jsonl").read_text().splitlines() if line.strip()}


def test_append_replaces_only_the_papers_in_the_new_run(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    first = _run(tmp_path, "run1", {"a.pdf": "# Paper A\n\nNeutron scattering on BT-7 at 5 K.\n" * 5, "b.pdf": "# Paper B\n\nMagnetic order below 20 K.\n" * 5})
    r1 = ingest_extractor_run({"extractor_run_dir": str(first), "catalog": str(catalog), "output_dir": str(tmp_path / "o1")})
    assert r1["status"] == "ok" and len(_ids(catalog)) == 2
    (catalog / "paper_facts.jsonl").write_text('{"fact_id": "f1", "paper_id": "x"}\n')
    chunks_before = (catalog / "source_chunks.jsonl").read_text().splitlines()

    second = _run(tmp_path, "run2", {"b.pdf": "# Paper B revised\n\nMagnetic order below 25 K.\n" * 5, "c.pdf": "# Paper C\n\nSpin waves measured on SPINS.\n" * 5})
    r2 = ingest_extractor_run({"extractor_run_dir": str(second), "catalog": str(catalog), "output_dir": str(tmp_path / "o2"), "append": True})
    assert r2["status"] == "ok" and r2["append"] is True and r2["replaced_existing"] == 1
    assert len(_ids(catalog)) == 3  # a kept, b replaced, c added
    assert (catalog / "paper_facts.jsonl").read_text() == '{"fact_id": "f1", "paper_id": "x"}\n'  # facts untouched
    chunks_after = [json.loads(l) for l in (catalog / "source_chunks.jsonl").read_text().splitlines() if l.strip()]
    assert any("25 K" in c.get("text", "") for c in chunks_after) and not any("20 K" in c.get("text", "") for c in chunks_after)
    assert len(chunks_after) >= len(chunks_before)
