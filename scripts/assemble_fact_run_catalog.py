"""Assemble a scoreable catalog from a per-paper extraction run.

A Horizon/Vista extraction run leaves one `paper_fact_extraction_report.json`
per paper under `<run>/facts/<paper_id>/`. This gathers those facts into
`paper_facts.jsonl` next to a copy of the source catalog's chunks and papers
(the same text the extraction saw), so the result can be revalidated, stamped
by `catalog-integrity`, scored against the human labels, and diffed against
the catalog it replaces. Fact ids are content-derived, so a fact whose type and
value did not change keeps its id and its labels.

    python scripts/assemble_fact_run_catalog.py --run local_runs/horizon_gold_v1_20261018 \
        --source-catalog local_runs/gold_eval_panel_48papers_20261002/catalog \
        --out local_runs/gold_v1_20261018/catalog
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory holding facts/<paper_id>/paper_fact_extraction_report.json")
    ap.add_argument("--source-catalog", required=True, help="catalog whose chunks and papers the run extracted from")
    ap.add_argument("--out", required=True, help="catalog directory to write")
    a = ap.parse_args()
    run, src, out = Path(a.run).expanduser(), Path(a.source_catalog).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    facts: list[dict] = []
    papers_seen: list[str] = []
    for report in sorted((run / "facts").glob("*/paper_fact_extraction_report.json")):
        payload = json.loads(report.read_text(encoding="utf-8"))
        rows = payload.get("facts") or payload.get("paper_facts") or []
        facts.extend(rows)
        papers_seen.append(report.parent.name)
    (out / "paper_facts.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
    for name in ("source_chunks.jsonl", "papers.jsonl", "paper_pdf_paths.jsonl", "evidence_graph_nodes.jsonl", "evidence_graph_edges.jsonl", "ontology_links.jsonl"):
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    (out / "ASSEMBLED_FROM.json").write_text(json.dumps({"run": str(run), "source_catalog": str(src), "papers": papers_seen, "fact_count": len(facts)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "catalog": str(out), "papers": len(papers_seen), "facts": len(facts)}))


if __name__ == "__main__":
    main()
