from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus.fact_judge import apply_judgments, build_prompt, derive_verdict, judge_applies, judge_paper_facts, located_passages
from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response


def test_verdict_is_a_fixed_rule_over_the_answers() -> None:
    assert derive_verdict({"subject": "this_paper", "field": "correct", "stated": "stated"}) == "supported"
    assert derive_verdict({"subject": "other_material_or_prior_work", "field": "correct", "stated": "stated"}) == "not_supported"
    assert derive_verdict({"subject": "this_paper", "field": "correct", "stated": "not_stated"}) == "not_supported"
    assert derive_verdict({"subject": "this_paper", "field": "wrong", "stated": "stated"}) == "wrong_field"
    assert derive_verdict({"subject": "this_paper", "field": "correct", "stated": "partial_or_vague"}) == "too_vague"
    assert derive_verdict({"subject": "cannot_tell", "field": "correct", "stated": "stated"}) == "unclear"
    assert derive_verdict({}) == "unclear"


def test_judge_sees_only_located_passages_and_never_accepts(tmp_path: Path) -> None:
    chunk_text = (
        "Specific heat data measured on the isostructural compound UFe2Si2 were used to estimate the phonon contribution. "
        "Our URu2Si2 crystals were grown by the Czochralski method. Measurements were made at 1.5 K in a 10 T magnet."
    )
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]
    raw = {"facts": [
        {"fact_type": "material", "answer_status": "supported", "value": {"text": "UFe2Si2"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "Specific heat data measured on the isostructural compound UFe2Si2"}]},
        {"fact_type": "sample_preparation", "answer_status": "supported", "value": {"text": "Czochralski growth"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "crystals were grown by the Czochralski method"}]},
        {"fact_type": "magnetic_field", "answer_status": "supported", "value": {"text": "a magnet was used"}, "confidence": 0.9,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "this sentence is not in the paper at all"}]},
    ]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}, provider="test", model="mock")
    by_type = {f["fact_type"]: f for f in facts}
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    (catalog / "papers.jsonl").write_text(json.dumps({"paper_id": "paper-a", "title": "Hidden order in URu2Si2"}) + "\n", encoding="utf-8")

    # The unlocated quote yields no passage, so that fact is never sent to the model.
    assert located_passages(by_type["magnetic_field"], {"chunk-a-000": chunks[0]}) == []
    passages = located_passages(by_type["material"], {"chunk-a-000": chunks[0]})
    assert len(passages) == 1 and "«Specific heat data measured on the isostructural compound UFe2Si2»" in passages[0]["passage"]
    messages = build_prompt(by_type["material"], passages, paper_title="Hidden order in URu2Si2")
    user = json.loads(messages[1]["content"])
    assert user["extracted_value"] == "UFe2Si2" and user["fact_type"] == "material"
    assert "comparison" in user["questions"]["subject"]

    seen: list[str] = []

    def fake_chat(messages, *, provider, model, base_url, api_key):
        value = json.loads(messages[1]["content"])["extracted_value"]
        seen.append(value)
        if value == "UFe2Si2":
            return {"subject": "other_material_or_prior_work", "field": "correct", "suggested_fact_type": "none", "stated": "stated", "note": "isostructural reference compound"}, None, 0.1
        return {"subject": "this_paper", "field": "correct", "suggested_fact_type": "none", "stated": "stated", "note": "grown by Czochralski"}, None, 0.1

    report = judge_paper_facts({"catalog": str(catalog), "model": "mock-judge", "provider": "ollama"}, chat=fake_chat)
    assert report["status"] == "ok"
    assert sorted(seen) == ["Czochralski growth", "UFe2Si2"]
    assert report["counts"] == {"judged": 2, "skipped_no_located_evidence": 1, "skipped_not_selected": 0, "resumed": 0, "errors": 0}
    assert report["verdicts"] == {"not_supported": 1, "supported": 1, "no_located_evidence": 1}
    rows = {json.loads(l)["fact_id"]: json.loads(l) for l in (catalog / "paper_fact_judgments.jsonl").read_text().splitlines() if l.strip()}
    assert rows[by_type["material"]["fact_id"]]["verdict"] == "not_supported"

    # A second run resumes from the file instead of calling the model again.
    seen.clear()
    report2 = judge_paper_facts({"catalog": str(catalog), "model": "mock-judge", "provider": "ollama"}, chat=fake_chat)
    assert seen == [] and report2["counts"]["resumed"] == 3  # the no-evidence record is kept too

    # A failed call (empty content) is recorded as unclear with an error, and a later resume retries it.
    seen.clear()
    (catalog / "paper_fact_judgments.jsonl").write_text(
        json.dumps({"fact_id": by_type["material"]["fact_id"], "model": "mock-judge", "prompt_version": rows[by_type["material"]["fact_id"]]["prompt_version"],
                    "verdict": "unclear", "answers": None, "error": "empty content (done_reason='length', thinking=present)"}) + "\n",
        encoding="utf-8")
    report3 = judge_paper_facts({"catalog": str(catalog), "model": "mock-judge", "provider": "ollama"}, chat=fake_chat)
    assert sorted(seen) == ["Czochralski growth", "UFe2Si2"] and report3["counts"]["resumed"] == 0
    rows = {json.loads(l)["fact_id"]: json.loads(l) for l in (catalog / "paper_fact_judgments.jsonl").read_text().splitlines() if l.strip()}
    assert rows[by_type["material"]["fact_id"]]["verdict"] == "not_supported"  # the last row for a fact wins

    applied = apply_judgments(catalog, catalog / "paper_fact_judgments.jsonl")
    judged = {json.loads(l)["fact_id"]: json.loads(l) for l in (catalog / "paper_facts.judged.jsonl").read_text().splitlines() if l.strip()}
    flagged = judged[by_type["material"]["fact_id"]]
    assert flagged["answer_status"] == "supported"  # status untouched: the judge never accepts or rejects
    assert flagged["human_review_required"] is True
    assert any("Second-pass judge (mock-judge): not_supported" in w for w in flagged["warnings"])
    assert judged[by_type["sample_preparation"]["fact_id"]]["metadata"]["judge"]["verdict"] == "supported"
    assert applied["applied"] == 2


def test_judge_does_not_apply_to_not_found_or_empty_values(tmp_path: Path) -> None:
    """A not_found fact cites the sentence that shows the paper lacks the fact
    type; it has no value, so there is nothing for the judge to check. Such
    facts get `not_applicable` without a model call, and a verdict stored
    for one earlier is ignored when judgments are applied."""
    chunk = {"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": "The data reveal that long-range magnetic order is absent in the sample."}
    facts = [
        {"fact_id": "paper-fact-nf", "paper_id": "paper-a", "fact_type": "magnetic_structure", "answer_status": "not_found", "value": None,
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "long-range magnetic order is absent", "quote_match": "verbatim", "support_status": "supported", "char_start": 21, "char_end": 56}]},
        {"fact_id": "paper-fact-sup", "paper_id": "paper-a", "fact_type": "sample", "answer_status": "supported", "value": {"text": "the sample"},
         "evidence": [{"chunk_id": "chunk-a-000", "quote": "in the sample", "quote_match": "verbatim", "support_status": "supported", "char_start": 57, "char_end": 70}]},
    ]
    assert judge_applies(facts[0]) is False and judge_applies(facts[1]) is True
    catalog = tmp_path / "catalog"; catalog.mkdir()
    (catalog / "paper_facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    calls: list[str] = []

    def fake_chat(messages, *, provider, model, base_url, api_key):
        calls.append(json.loads(messages[1]["content"])["fact_type"])
        return {"subject": "this_paper", "field": "correct", "suggested_fact_type": "none", "stated": "partial_or_vague", "note": "vague"}, None, 0.1

    report = judge_paper_facts({"catalog": str(catalog), "model": "mock-judge", "provider": "ollama"}, chat=fake_chat)
    assert calls == ["sample"] and report["verdicts"] == {"not_applicable": 1, "too_vague": 1}
    # an old stored verdict on the not_found fact must not surface as a flag
    (catalog / "paper_fact_judgments.jsonl").write_text(
        json.dumps({"fact_id": "paper-fact-nf", "model": "mock-judge", "prompt_version": "x", "verdict": "not_supported", "answers": {}, "note": "value is null"}) + "\n" +
        json.dumps({"fact_id": "paper-fact-sup", "model": "mock-judge", "prompt_version": "x", "verdict": "too_vague", "answers": {}, "note": "vague"}) + "\n", encoding="utf-8")
    applied = apply_judgments(catalog, catalog / "paper_fact_judgments.jsonl")
    rows = {json.loads(l)["fact_id"]: json.loads(l) for l in (catalog / "paper_facts.judged.jsonl").read_text().splitlines() if l.strip()}
    assert applied["applied"] == 1 and "judge" not in rows["paper-fact-nf"].get("metadata", {}) and not rows["paper-fact-nf"].get("warnings")
    assert rows["paper-fact-sup"]["metadata"]["judge"]["verdict"] == "too_vague"
