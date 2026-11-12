from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus.evidence_graph_proposer import run_evidence_graph_proposer
from tas_copilot_core.corpus.fact_eval_dashboard import EDGE_VERDICTS, FACT_VERDICTS, build_fact_eval_dashboard

from tests.test_evidence_graph_proposer import _catalog_with_facts


def test_dashboard_renders_fact_and_edge_cards(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog_with_facts(tmp_path)

    # Produce one relation edge so the edges section has a card.
    from tas_copilot_core.corpus import evidence_graph_proposer as proposer_module

    def fake_relation_model(packet, **_kwargs):
        return {
            "pair_id": packet["pair_id"],
            "relation": "consistent_with",
            "confidence": 0.8,
            "evidence_a": {"chunk_id": "chunk-a-001", "quote": "ordered moment of 3.46 muB per Mn"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Both papers report a similar Mn moment.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    run_evidence_graph_proposer({"catalog": str(catalog), "output_dir": str(tmp_path / "prop"), "provider": "ollama", "model": "gemma3:4b"})

    output_dir = tmp_path / "panel"
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(output_dir)})

    assert report["status"] == "ok"
    assert report["fact_count"] == 4  # all extracted facts appear, including non-supported ones
    assert report["relation_edge_count"] == 1
    html = (output_dir / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(html.split("<script id='panel-data' type='application/json'>")[1].split("</script>")[0].replace("<\\/", "</"))
    # Every fact and relation edge becomes a card datum.
    assert {fact["fact_id"] for fact in payload["facts"]} == {"paper-fact-aaa", "paper-fact-bbb", "paper-fact-ccc", "paper-fact-ddd"}
    assert payload["edges"][0]["edge_type"] == "CONSISTENT_WITH"
    # Chunk text is inline so quotes can be checked in context.
    assert any("ab plane" in evidence["chunk_text"] for fact in payload["facts"] for evidence in fact["evidence"])
    # The gold verdict vocabularies ship with the panel.
    assert payload["fact_verdicts"] == FACT_VERDICTS
    assert payload["edge_verdicts"] == EDGE_VERDICTS
    # Export logic maps verdicts onto ReviewLabel fields.
    assert "object_type: 'paper_fact'" in html
    assert "object_type: 'evidence_graph_edge'" in html
    assert "type_specific_status" in html


def test_dashboard_without_facts_is_missing_input(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "panel")})
    assert report["status"] == "missing_input"


def test_dashboard_without_edges_still_renders_facts(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "panel")})
    assert report["status"] == "ok"
    assert report["fact_count"] == 4
    assert report["relation_edge_count"] == 0

def test_dashboard_citation_checks_locate_leaked_quotes(tmp_path: Path) -> None:
    from tests.test_evidence_graph_proposer import _write_jsonl

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    _write_jsonl(
        catalog / "papers.jsonl",
        [
            {"paper_id": "paper-alpha", "title": "Alpha paper"},
            {"paper_id": "paper-beta", "title": "Beta paper"},
        ],
    )
    _write_jsonl(
        catalog / "source_chunks.jsonl",
        [
            {"chunk_id": "chunk-alpha-000", "paper_id": "paper-alpha", "text": "The transition occurs below 85 K in the susceptibility. A refined moment of 3.25 muB was found at 20 K."},
            {"chunk_id": "chunk-alpha-001", "paper_id": "paper-alpha", "text": "Samples were prepared from Sn flux and ground with an agate mortar."},
            {"chunk_id": "chunk-beta-000", "paper_id": "paper-beta", "text": "We will show that these excitations are well described by a Heisenberg model."},
        ],
    )
    _write_jsonl(
        catalog / "paper_facts.jsonl",
        [
            {
                "fact_id": "paper-fact-good",
                "paper_id": "paper-alpha",
                "fact_type": "sample",
                "answer_status": "supported",
                "value": {"text": "Sn flux growth"},
                "evidence": [{"chunk_id": "chunk-alpha-001", "quote": "prepared from Sn flux", "support_status": "supported"}],
            },
            {
                "fact_id": "paper-fact-mishomed",
                "paper_id": "paper-alpha",
                "fact_type": "magnetic_structure",
                "answer_status": "supported",
                "value": {"text": "transition below 85 K"},
                "evidence": [{"chunk_id": "chunk-alpha-001", "quote": "transition occurs below 85 K", "support_status": "proposed"}],
            },
            {
                "fact_id": "paper-fact-crosspaper",
                "paper_id": "paper-alpha",
                "fact_type": "spin_direction",
                "answer_status": "supported",
                "value": {"text": "Heisenberg excitations"},
                "evidence": [{"chunk_id": "chunk-alpha-000", "quote": "excitations are well described by a Heisenberg model", "support_status": "proposed"}],
            },
            {
                "fact_id": "paper-fact-stitched",
                "paper_id": "paper-alpha",
                "fact_type": "ordered_moment",
                "answer_status": "supported",
                "value": {"text": "3.25 muB below 85 K"},
                "evidence": [{"chunk_id": "chunk-alpha-001", "quote": "transition occurs below 85 K...refined moment of 3.25 muB was found", "support_status": "proposed"}],
            },
            {
                "fact_id": "paper-fact-fabricated",
                "paper_id": "paper-alpha",
                "fact_type": "propagation_vector",
                "answer_status": "supported",
                "value": {"text": "k = (0 0 0)"},
                "evidence": [{"chunk_id": "chunk-alpha-000", "quote": "propagation vector of one quarter along c", "support_status": "proposed"}],
            },
        ],
    )

    output_dir = tmp_path / "panel"
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(output_dir)})
    assert report["status"] == "ok"
    html = (output_dir / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(html.split("<script id='panel-data' type='application/json'>")[1].split("</script>")[0].replace("<\\/", "</"))

    cards = {card["fact_id"]: card for card in payload["facts"]}
    statuses = {fact_id: card["evidence"][0]["citation_check"]["status"] for fact_id, card in cards.items()}
    assert statuses["paper-fact-good"] == "in_cited_chunk"
    assert statuses["paper-fact-mishomed"] == "in_other_chunk"
    assert cards["paper-fact-mishomed"]["evidence"][0]["citation_check"]["found_in"] == ["chunk-alpha-000"]
    assert statuses["paper-fact-crosspaper"] == "in_other_paper"
    assert cards["paper-fact-crosspaper"]["evidence"][0]["citation_check"]["found_in"] == ["chunk-beta-000"]
    assert statuses["paper-fact-stitched"] == "stitched_fragments"
    assert statuses["paper-fact-fabricated"] == "not_found"

    # Flagged cards are listed before clean ones.
    flagged = [card["citation_flagged"] for card in payload["facts"]]
    assert flagged == sorted(flagged, reverse=True)
    assert payload["facts"][-1]["fact_id"] == "paper-fact-good"


def test_dashboard_shows_prior_labels_as_context(tmp_path: Path) -> None:
    import json

    from tas_copilot_core.corpus.fact_eval_dashboard import build_fact_eval_dashboard

    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": []}) + "\n")
    (catalog / "source_chunks.jsonl").write_text(""); (catalog / "evidence_graph_nodes.jsonl").write_text(""); (catalog / "evidence_graph_edges.jsonl").write_text("")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "A paper"}) + "\n")
    labels = tmp_path / "prior.jsonl"
    labels.write_text(json.dumps({"object_type": "paper_fact", "object_id": "paper-fact-1", "verdict": "negative", "type_specific_status": "wrong_field", "review_notes": "facility, not instrument", "reviewer_id": "prior-reviewer"}) + "\n")
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "prior_labels": [str(labels)]})
    assert report["status"] == "ok"
    assert report["prior_label_summary"] == {"prior-reviewer": {"wrong_field": 1}}
    html = (tmp_path / "out/fact_eval_dashboard.html").read_text()
    assert "facility, not instrument" in html and "priorBlock" in html and "prior-summary" in html


def test_dashboard_resumes_earlier_labels_and_queues_repaired_citations(tmp_path: Path) -> None:
    """A reviewer's earlier export pre-fills the panel. A citation-defect label
    on a fact whose quote has since been replaced by source text is not
    pre-filled: it is queued first for a second look."""
    import re
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    chunk_text = (
        "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer "
        "with two pyrolytic graphite filters after the sample position. The sample was cooled to 4 K."
    )
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]
    raw = {"facts": [
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "experiments were carried out on the BT-7 thermal triple-axis spectrometer"}]},
        {"fact_type": "filter", "answer_status": "supported", "value": {"text": "two PG filters"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "with two pyrolytic graphite graphite filters after the sample position"}]},
        {"fact_type": "temperature_range", "answer_status": "supported", "value": {"text": "4 K"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "The sample was cooled to 4 K."}]},
    ]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}, provider="test", model="mock")
    by_type = {fact["fact_type"]: fact["fact_id"] for fact in facts}
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    earlier = tmp_path / "earlier.jsonl"
    rows = [
        {"object_type": "paper_fact", "object_id": by_type["instrument"], "verdict": "positive", "type_specific_status": "correct", "review_notes": "fine"},
        {"object_type": "paper_fact", "object_id": by_type["filter"], "verdict": "negative", "type_specific_status": "ocr_or_chunking_problem", "review_notes": "duplicated word in quote"},
        {"object_type": "paper_fact", "object_id": "missing-0-paper-a-collimation", "verdict": "negative", "type_specific_status": "missing",
         "metadata": {"paper_id": "paper-a", "fact_type": "collimation", "expected_value": "open-80-80-120", "missing_fact": True}},
    ]
    earlier.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "resume_labels": [str(earlier)], "storage_key": "gold-set"})
    assert report["status"] == "ok"
    assert report["resumed_fact_labels"] == 1
    assert report["second_look_facts"] == 1
    assert report["resumed_missing_records"] == 1
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert payload["storage_key"] == "gold-set"
    assert payload["recheck_ids"] == [by_type["filter"]]
    assert payload["seed"]["facts"] == {by_type["instrument"]: {"verdict": "correct", "notes": "fine", "expected_value": ""}}
    assert payload["seed"]["missing"][0]["expected_value"] == "open-80-80-120"
    order = [card["fact_type"] for card in payload["facts"]]
    assert order == ["filter", "temperature_range", "instrument"]  # second look, unlabeled, already labeled
    second_look = payload["facts"][0]
    assert second_look["recheck"] is True and second_look["earlier_label"]["verdict"] == "ocr_or_chunking_problem"
    evidence = second_look["evidence"][0]
    assert evidence["quote_match"] == "snapped"
    assert chunk_text[evidence["char_start"]:evidence["char_end"]] == evidence["quote"]
    assert "graphite graphite" in evidence["quote_as_extracted"]


def test_dashboard_reads_judged_fact_file_and_sorts_judge_flags_first(tmp_path: Path) -> None:
    """Built from paper_facts.judged.jsonl, a card the second-pass judge flagged
    comes before an unflagged card, carries the judge warning, and has no
    pre-filled verdict: the judge routes review effort, it never labels."""
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    plain = {"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": [], "warnings": []}
    flagged = {"fact_id": "paper-fact-2", "paper_id": "p1", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "NCNR"}, "evidence": [],
               "warnings": ["Second-pass judge (mock): wrong_field (facility, not instrument)"], "human_review_required": True,
               "metadata": {"judge": {"model": "mock", "verdict": "wrong_field", "suggested_fact_type": "facility", "note": "facility, not instrument"}}}
    (catalog / "paper_facts.judged.jsonl").write_text(json.dumps(plain) + "\n" + json.dumps(flagged) + "\n")
    (catalog / "source_chunks.jsonl").write_text(""); (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "A paper"}) + "\n")
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "fact_file": "paper_facts.judged.jsonl"})
    assert report["status"] == "ok" and report["fact_file"] == "paper_facts.judged.jsonl" and report["judge_flagged_facts"] == 1
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert [c["fact_id"] for c in payload["facts"]] == ["paper-fact-2", "paper-fact-1"]
    assert payload["facts"][0]["judge_flagged"] is True and payload["facts"][0]["judge"]["suggested_fact_type"] == "facility"
    assert payload["seed"]["facts"] == {}
    assert "judge-flagged" in html_text


def test_dashboard_groups_cards_by_paper_with_pending_papers_first(tmp_path: Path) -> None:
    """--group-by-paper keeps each paper's cards together, orders papers by
    pending work, carries per-paper counts and the paper's chunk text, and
    leaves the ungrouped order untouched when the flag is off."""
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    facts = [
        {"fact_id": "paper-fact-a1", "paper_id": "pa", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-b1", "paper_id": "pb", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "SPINS"}, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-a2", "paper_id": "pa", "fact_type": "sample", "answer_status": "supported", "value": {"text": "crystal"}, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-b2", "paper_id": "pb", "fact_type": "sample", "answer_status": "supported", "value": {"text": "powder"}, "evidence": [], "warnings": []},
    ]
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (catalog / "source_chunks.jsonl").write_text(json.dumps({"chunk_id": "chunk-pb-0", "paper_id": "pb", "text": "Measured on SPINS."}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "pa", "title": "Paper A", "doi": "10.1000/a"}) + "\n" + json.dumps({"paper_id": "pb", "title": "Paper B", "metadata": {"pdf_name": "Paper B (2020).pdf"}}) + "\n")
    (catalog / "paper_pdf_paths.jsonl").write_text(json.dumps({"paper_id": "pb", "pdf_relpath": "2020/Paper B (2020).pdf"}) + "\n")
    earlier = tmp_path / "earlier.jsonl"  # paper A is finished; paper B has everything pending
    earlier.write_text("".join(json.dumps({"object_type": "paper_fact", "object_id": fid, "verdict": "positive", "type_specific_status": "correct"}) + "\n" for fid in ("paper-fact-a1", "paper-fact-a2")))

    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "grouped"), "resume_labels": [str(earlier)], "group_by_paper": True, "pdf_dir": "/Volumes/corpus/pdfs"})
    assert report["status"] == "ok" and report["group_by_paper"] is True and report["paper_count"] == 2
    html_text = (tmp_path / "grouped" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert [c["fact_id"] for c in payload["facts"]] == ["paper-fact-b1", "paper-fact-b2", "paper-fact-a1", "paper-fact-a2"]
    assert [g["paper_id"] for g in payload["paper_groups"]] == ["pb", "pa"]
    assert payload["paper_groups"][0] == {"paper_id": "pb", "title": "Paper B", "total": 2, "sidelined": 0, "second_look": 0, "labeled": 0, "pending": 2, "flagged": 0,
                                          "pdf_name": "Paper B (2020).pdf", "pdf_link": "file:///Volumes/corpus/pdfs/2020/Paper%20B%20%282020%29.pdf", "doi": ""}
    assert payload["paper_groups"][1]["pending"] == 0 and payload["paper_groups"][1]["labeled"] == 2
    assert payload["paper_groups"][1]["doi"] == "10.1000/a" and payload["paper_groups"][1]["pdf_link"] == ""  # no pdf_name: no file link, DOI only
    # a relative --pdf-dir links to a folder shipped beside the page (Box/zip hand-offs)
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "relative"), "resume_labels": [str(earlier)], "group_by_paper": True, "pdf_dir": "pdfs"})
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", (tmp_path / "relative" / "fact_eval_dashboard.html").read_text(encoding="utf-8"), re.S).group(1))
    assert payload["paper_groups"][0]["pdf_link"] == "pdfs/2020/Paper%20B%20%282020%29.pdf"
    assert payload["paper_chunks"] == {"pb": [{"chunk_id": "chunk-pb-0", "text": "Measured on SPINS."}]}
    assert "papernav" in html_text

    plain = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "plain"), "resume_labels": [str(earlier)]})
    assert plain["group_by_paper"] is False and plain["paper_count"] == 0
    plain_payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", (tmp_path / "plain" / "fact_eval_dashboard.html").read_text(), re.S).group(1))
    assert [c["fact_id"] for c in plain_payload["facts"]] == ["paper-fact-b1", "paper-fact-b2", "paper-fact-a1", "paper-fact-a2"]  # unlabeled first, by input order
    assert plain_payload["paper_groups"] == [] and plain_payload["paper_chunks"] == {}


def test_dashboard_finds_an_uncited_value_in_the_papers_text(tmp_path: Path) -> None:
    """An 'ambiguous' fact with no evidence whose value sits verbatim in the
    paper gets that sentence shown as found-by-search, not as a citation."""
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    text = "We co-aligned two HoBi crystals. We collected inelastic neutron scattering data with fixed incident energy Ei = 25 meV at T = 13 K with a total proton charge of 47 C. We used the high flux mode."
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "energy_setting", "answer_status": "ambiguous", "value": {"text": "Ei = 25 meV"}, "evidence": [], "warnings": []}) + "\n"
                                               + json.dumps({"fact_id": "paper-fact-2", "paper_id": "p1", "fact_type": "energy_setting", "answer_status": "not_found", "value": {"text": "Ef = 99 meV"}, "evidence": [], "warnings": []}) + "\n")
    (catalog / "source_chunks.jsonl").write_text(json.dumps({"chunk_id": "chunk-p1-0", "paper_id": "p1", "text": text}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "HoBi"}) + "\n")
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    assert report["status"] == "ok"
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    cards = {c["fact_id"]: c for c in payload["facts"]}
    hit = cards["paper-fact-1"]["value_hits"]
    assert len(hit) == 1 and hit[0]["chunk_id"] == "chunk-p1-0"
    assert hit[0]["passage"].startswith("We collected inelastic") and "Ei = 25 meV" in hit[0]["passage"]
    assert text[hit[0]["char_start"]:hit[0]["char_end"]] == "Ei = 25 meV"
    assert cards["paper-fact-2"]["value_hits"] == [{"chunk_id": None, "item": "Ef = 99 meV", "passage": "", "char_start": None, "char_end": None}]
    assert "Not a citation." in html_text


def test_dashboard_sidelines_edges_on_quarantined_facts(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    facts = [
        {"fact_id": "paper-fact-q", "paper_id": "p-bad", "fact_type": "ordered_moment", "answer_status": "supported", "value": {"text": "1 muB"}, "evidence": [], "warnings": [], "metadata": {"integrity_status": "content_mismatch"}},
        {"fact_id": "paper-fact-ok", "paper_id": "p-ok", "fact_type": "ordered_moment", "answer_status": "supported", "value": {"text": "2 muB"}, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-ok2", "paper_id": "p-ok2", "fact_type": "ordered_moment", "answer_status": "supported", "value": {"text": "2.1 muB"}, "evidence": [], "warnings": []},
    ]
    edges = [
        {"edge_id": "same_quantity_as-bad", "edge_type": "SAME_QUANTITY_AS", "source_node_id": "PaperFact:paper-fact-q", "target_node_id": "PaperFact:paper-fact-ok", "status": "proposed", "confidence": 1.0, "evidence": [], "provenance": []},
        {"edge_id": "consistent_with-good", "edge_type": "CONSISTENT_WITH", "source_node_id": "PaperFact:paper-fact-ok", "target_node_id": "PaperFact:paper-fact-ok2", "status": "proposed", "confidence": 0.8, "evidence": [], "provenance": []},
    ]
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (catalog / "source_chunks.jsonl").write_text(""); (catalog / "evidence_graph_nodes.jsonl").write_text("")
    (catalog / "evidence_graph_edges.jsonl").write_text("".join(json.dumps(e) + "\n" for e in edges))
    (catalog / "papers.jsonl").write_text("")
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    assert report["status"] == "ok"
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert [(e["edge_id"], e["sidelined"]) for e in payload["edges"]] == [("consistent_with-good", False), ("same_quantity_as-bad", True)]


def test_dashboard_looks_up_list_values_item_by_item(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    text = "Neutron powder diffraction was measured on BT1 at the NCNR. Pair distribution data came from NOMAD at the SNS. Transport used a Quantum Design PPMS."
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "ambiguous", "value": {"text": "BT1, NOMAD, HB-2A, Quantum Design Physical Property Measurement System (PPMS)"}, "evidence": [], "warnings": ["Supported fact has no cited evidence; downgraded to ambiguous."]}) + "\n")
    (catalog / "source_chunks.jsonl").write_text(json.dumps({"chunk_id": "chunk-p1-0", "paper_id": "p1", "text": text}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "Spin ladder"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    hits = payload["facts"][0]["value_hits"]
    assert [h["item"] for h in hits if h["chunk_id"]] == ["BT1", "NOMAD"]
    assert [h["item"] for h in hits if not h["chunk_id"]] == ["HB-2A", "Quantum Design Physical Property Measurement System (PPMS)"]
    assert "downgraded to" in html_text


def test_dashboard_shows_the_closest_passage_for_a_mangled_value(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    text = "The powder samples were studied with neutron diffraction on the HB-2A instrument at the  High  Flux  Isotope  Reactor  at  ORNL  using a monochromatic beam."
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "ambiguous", "value": {"text": "High Flux Is/Reactor"}, "evidence": [], "warnings": ["Supported fact has no cited evidence; downgraded to ambiguous."]}) + "\n")
    (catalog / "source_chunks.jsonl").write_text(json.dumps({"chunk_id": "chunk-p1-0", "paper_id": "p1", "text": text}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "Spin ladder"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    hit = payload["facts"][0]["value_hits"][0]
    assert hit["chunk_id"] == "chunk-p1-0" and hit["closest"] == "High Flux Isotope Reactor" and "HB-2A instrument" in hit["passage"]


def test_dashboard_sidelines_facts_whose_value_was_discarded(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    facts = [
        {"fact_id": "paper-fact-bad", "paper_id": "p1", "fact_type": "instrument", "answer_status": "ambiguous", "value": {}, "evidence": [],
         "warnings": ["Malformed value (letter-spaced text): model output artifact, not paper content; value cleared (original in metadata.discarded_value)."],
         "metadata": {"malformed_value": "letter-spaced text", "discarded_value": "B r u k e r  A P E X"}},
        {"fact_id": "paper-fact-ok", "paper_id": "p1", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": [], "warnings": []},
    ]
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (catalog / "source_chunks.jsonl").write_text(""); (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "T"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "group_by_paper": True})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert [c["fact_id"] for c in payload["facts"]] == ["paper-fact-ok", "paper-fact-bad"]  # discarded last
    assert payload["facts"][1]["malformed_value"] == "letter-spaced text" and payload["facts"][1]["discarded_value"] == "B r u k e r  A P E X"
    assert payload["paper_groups"][0]["total"] == 1 and payload["paper_groups"][0]["sidelined"] == 1
    assert "f.malformed_value" in html_text


def test_dashboard_matches_decorated_list_items_on_their_distinctive_word(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    chunks = [
        {"chunk_id": "c1", "paper_id": "p1", "text": "Single crystal neutron diffraction was conducted at the NCNR using the BT-7 triple-axis spectrometer. Data were also collected at ANSTO using the thermal triple-axis spectrometer Taipan with PG (002)."},
        {"chunk_id": "c2", "paper_id": "p1", "text": "The synchrotron x-ray powder diffraction patterns were collected on the BL01C2 beam line at the NSRRC in Taiwan."},
    ]
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "ambiguous",
        "value": {"text": "BT-7 triple-axis spectrometer, Taipan triple-axis (thermal), and BL01C2 beam line (synchrotron x-ray)"}, "evidence": [], "warnings": ["Supported fact has no cited evidence; downgraded to ambiguous."]}) + "\n")
    (catalog / "source_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "Co3TeO6"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    hits = payload["facts"][0]["value_hits"]
    # BT-7: whole item; Taipan: the word (the paper orders it differently); BL01C2: the item minus its parenthetical
    assert [(h["item"][:6], h["chunk_id"], h.get("matched_word")) for h in hits] == [("BT-7 t", "c1", None), ("Taipan", "c1", "Taipan"), ("BL01C2", "c2", None)]


def test_dashboard_shows_absence_cues_for_not_found_facts(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    facts = [
        {"fact_id": "paper-fact-nf-inst", "paper_id": "p1", "fact_type": "instrument", "answer_status": "not_found", "value": None, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-nf-press", "paper_id": "p1", "fact_type": "pressure", "answer_status": "not_found", "value": None, "evidence": [], "warnings": []},
        {"fact_id": "paper-fact-nf-other", "paper_id": "p1", "fact_type": "uncertainty_note", "answer_status": "not_found", "value": None, "evidence": [], "warnings": []},
    ]
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (catalog / "source_chunks.jsonl").write_text(json.dumps({"chunk_id": "c1", "paper_id": "p1", "text": "Transport was measured in a PPMS down to 2 K. The resistivity shows a switching transition. No pressure was applied."}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "T"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    cards = {c["fact_id"]: c for c in payload["facts"]}
    assert [c["term"] for c in cards["paper-fact-nf-inst"]["absence_cues"]] == ["PPMS"]
    assert cards["paper-fact-nf-press"]["absence_cues"] == []  # "No pressure was applied" has no value with a unit
    assert cards["paper-fact-nf-other"]["absence_cues"] is None  # no search defined for the type
    assert "Reported absent." in html_text


def test_second_looks_page_keeps_only_recheck_cards(tmp_path: Path) -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response
    import re

    chunk_text = "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer with two pyrolytic graphite filters after the sample position. The sample was cooled to 4 K."
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]
    raw = {"facts": [
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "confidence": 0.9, "evidence": [{"chunk_id": "chunk-a-000", "quote": "carried out on the BT-7 thermal triple-axis spectrometer"}]},
        {"fact_type": "filter", "answer_status": "supported", "value": {"text": "two PG filters"}, "confidence": 0.9, "evidence": [{"chunk_id": "chunk-a-000", "quote": "with two pyrolytic graphite graphite filters after the sample position"}]},
    ]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}, provider="test", model="mock")
    by_type = {f["fact_type"]: f["fact_id"] for f in facts}
    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    earlier = tmp_path / "earlier.jsonl"
    earlier.write_text(json.dumps({"object_type": "paper_fact", "object_id": by_type["instrument"], "verdict": "positive", "type_specific_status": "correct"}) + "\n"
                       + json.dumps({"object_type": "paper_fact", "object_id": by_type["filter"], "verdict": "negative", "type_specific_status": "ocr_or_chunking_problem", "review_notes": "doubled word"}) + "\n", encoding="utf-8")
    report = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "resume_labels": [str(earlier)], "only_second_looks": True, "group_by_paper": True})
    assert report["status"] == "ok" and report["only_second_looks"] is True and report["fact_count"] == 1
    html_text = (tmp_path / "out" / "second_looks.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    assert [c["fact_id"] for c in payload["facts"]] == [by_type["filter"]] and payload["facts"][0]["recheck"] is True
    assert payload["page_name"] == "second_looks" and payload["paper_groups"][0]["second_look"] == 1
    assert "Second looks only." in html_text


def test_a_newer_ocr_label_settles_a_second_look(tmp_path: Path) -> None:
    """The reviewer's second-look pass may conclude the paper text really is
    broken; that newer label must not re-queue the card."""
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response
    import re

    chunk_text = "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer with two pyrolytic graphite filters after the sample position."
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]
    raw = {"facts": [{"fact_type": "filter", "answer_status": "supported", "value": {"text": "two PG filters"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": "with two pyrolytic graphite graphite filters after the sample position"}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}, provider="test", model="mock")
    fid = facts[0]["fact_id"]
    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text(json.dumps(facts[0]) + "\n", encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text(json.dumps(chunks[0]) + "\n", encoding="utf-8")
    old = tmp_path / "old.jsonl"; new = tmp_path / "new.jsonl"
    old.write_text(json.dumps({"object_type": "paper_fact", "object_id": fid, "verdict": "negative", "type_specific_status": "ocr_or_chunking_problem", "review_notes": "doubled word"}) + "\n")
    new.write_text(json.dumps({"object_type": "paper_fact", "object_id": fid, "verdict": "negative", "type_specific_status": "ocr_or_chunking_problem", "review_notes": "still broken on second look"}) + "\n")
    r1 = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "a"), "resume_labels": [str(old)], "second_look_from": [str(old)]})
    assert r1["second_look_facts"] == 1
    r2 = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "b"), "resume_labels": [str(old), str(new)], "second_look_from": [str(old)]})
    assert r2["second_look_facts"] == 0 and r2["resumed_fact_labels"] == 1
    r3 = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "c"), "resume_labels": [str(old), str(new)]})
    assert r3["second_look_facts"] == 1  # without the option every resume file counts as old (previous behaviour)


def test_instrument_cards_carry_canonical_alias_chips(tmp_path: Path) -> None:
    import re

    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "paper-fact-1", "paper_id": "p1", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "the BT-7 thermal triple-axis spectrometer and SPINS"}, "evidence": [], "warnings": []}) + "\n"
                                               + json.dumps({"fact_id": "paper-fact-2", "paper_id": "p1", "fact_type": "facility", "answer_status": "supported", "value": {"text": "NIST Center for Neutron Research"}, "evidence": [], "warnings": []}) + "\n")
    (catalog / "source_chunks.jsonl").write_text(""); (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "p1", "title": "T"}) + "\n")
    build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "out")})
    html_text = (tmp_path / "out" / "fact_eval_dashboard.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"<script id='panel-data' type='application/json'>(.*?)</script>", html_text, re.S).group(1))
    cards = {c["fact_id"]: c for c in payload["facts"]}
    assert [c["id"] for c in cards["paper-fact-1"]["canonical"]] == ["bt7", "spins"]
    assert cards["paper-fact-2"]["canonical"][0]["id"] == "ncnr"


def test_no_edges_builds_a_facts_only_page(tmp_path):
    from tas_copilot_core.corpus.fact_eval_dashboard import build_fact_eval_dashboard

    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text(json.dumps({"fact_id": "f1", "paper_id": "pa", "fact_type": "material", "value": {"text": "MnF2"}, "answer_status": "supported", "evidence": []}) + "\n")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "pa", "title": "Paper A"}) + "\n")
    (catalog / "evidence_graph_edges.jsonl").write_text(json.dumps({"edge_id": "e1", "edge_type": "SAME_QUANTITY_AS", "source_node_id": "f1", "target_node_id": "f1", "status": "proposed"}) + "\n")
    with_edges = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "a"), "group_by_paper": True})
    without = build_fact_eval_dashboard({"catalog": str(catalog), "output_dir": str(tmp_path / "b"), "group_by_paper": True, "no_edges": True})
    assert with_edges["relation_edge_count"] >= 1 and without["relation_edge_count"] == 0
