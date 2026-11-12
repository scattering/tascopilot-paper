from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus.fact_label_scoring import score_fact_labels
from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response, revalidate_paper_facts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _label(fact_id: str, verdict: str, status: str) -> dict:
    return {"object_type": "paper_fact", "object_id": fact_id, "verdict": verdict, "type_specific_status": status}


def test_score_fact_labels_before_and_after_revalidation(tmp_path: Path) -> None:
    """The same human labels score a catalog before and after its citations are
    repaired: fact ids are stable, and a corrupted transcription that the
    reviewer marked as a citation defect becomes a sound citation."""
    chunk_text = (
        "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer "
        "with two pyrolytic graphite filters after the sample position. The sample was cooled to 4 K."
    )
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]
    packet = {"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}
    raw = {
        "facts": [
            {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "confidence": 0.9,
             "evidence": [{"chunk_id": "chunk-a-000", "quote": "experiments were carried out on the BT-7 thermal triple-axis spectrometer"}]},
            {"fact_type": "filter", "answer_status": "supported", "value": {"text": "two PG filters"}, "confidence": 0.9,
             "evidence": [{"chunk_id": "chunk-a-000", "quote": "with two pyrolytic graphite graphite filters after the sample position"}]},
            {"fact_type": "pressure", "answer_status": "not_found", "value": {}, "evidence": []},
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    by_type = {fact["fact_type"]: fact for fact in facts}
    # An older catalog: quotes stored as the model wrote them, no span fields.
    old_facts = json.loads(json.dumps(facts))
    for fact in old_facts:
        for ev in fact["evidence"]:
            ev["quote"] = ev.get("quote_as_extracted") or ev["quote"]
            for key in ("quote_as_extracted", "char_start", "char_end", "quote_match"):
                ev.pop(key, None)
            ev["warnings"] = []
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    _write_jsonl(catalog / "paper_facts.jsonl", old_facts)
    _write_jsonl(catalog / "source_chunks.jsonl", chunks)
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [
        _label(by_type["instrument"]["fact_id"], "positive", "correct"),
        _label(by_type["filter"]["fact_id"], "negative", "ocr_or_chunking_problem"),
        _label(by_type["pressure"]["fact_id"], "positive", "correct"),
        {"object_type": "paper_fact", "object_id": "missing-0-paper-a-collimation", "verdict": "negative", "type_specific_status": "missing", "metadata": {"missing_fact": True}},
        _label("paper-fact-not-in-catalog", "positive", "correct"),
    ])

    before = score_fact_labels({"catalog": str(catalog), "labels": [str(labels)], "output_dir": str(tmp_path / "before")})
    assert before["status"] == "ok"
    assert before["coverage"]["labeled_facts_scored"] == 3
    assert before["coverage"]["missing_fact_records"] == 1
    assert before["coverage"]["labels_for_unknown_fact_ids"] == 1
    assert before["by_extractor_status"]["supported"]["n"] == 2
    assert before["by_extractor_status"]["supported"]["correct_rate"] == 0.5
    assert before["citation_state"]["human_citation_defects_still_defective"] == 1
    assert (tmp_path / "before" / "fact_label_score.md").is_file()

    result = revalidate_paper_facts(catalog, output_name="paper_facts.revalidated.jsonl")
    assert result["status"] == "ok" and result["changes"]["id_mismatch"] == 0
    after = score_fact_labels({"catalog": str(catalog), "labels": [str(labels)], "fact_file": "paper_facts.revalidated.jsonl"})
    assert after["coverage"]["labeled_facts_scored"] == 3  # same ids, same labels
    assert after["citation_state"]["human_citation_defects_with_sound_citation_now"] == 1
    assert after["citation_state"]["human_citation_defects_still_defective"] == 0
    assert after["citation_state"]["quote_match_counts"] == {"verbatim": 1, "snapped": 1}


def test_score_fact_labels_reports_missing_input(tmp_path: Path) -> None:
    catalog = tmp_path / "empty"
    catalog.mkdir()
    assert score_fact_labels({"catalog": str(catalog), "labels": []})["status"] == "missing_input"
