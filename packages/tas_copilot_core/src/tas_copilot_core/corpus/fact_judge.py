"""Second-pass check of extracted paper facts against their located passages.

One fact at a time, a local model answers three narrow questions from fixed
options: is the value about this paper's own sample or measurement, is the
fact type right, and does the passage state the value. The verdict is derived
from those answers by code, not by the model. The judge sees only the located
source passages (quotes the pipeline verified as source text) with a little
surrounding context, never the whole paper, and it never changes a fact's
status: its verdict is recorded beside the fact and keeps the fact in human
review. Human review labels remain the only route to acceptance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from tas_copilot_core.corpus.fact_label_scoring import _is_quarantined
from tas_copilot_core.corpus.paper_facts import FACT_TYPES, _normalize_alnum
from tas_copilot_core.llm_router import ollama_chat, openai_compatible_chat

PROMPT_VERSION = "paper_fact_judge_v1"  # v1 (2026-10-17): definitions aligned with the reviewers (processing temperatures count; k from a magnetic space group)
_SOUND_QUOTE_MATCHES = {"verbatim", "snapped", "split_fragment"}
_CONTEXT_CHARS = 300
_MAX_PASSAGES = 4

SUBJECT_OPTIONS = ["this_paper", "other_material_or_prior_work", "cannot_tell"]
FIELD_OPTIONS = ["correct", "wrong"]
STATED_OPTIONS = ["stated", "partial_or_vague", "not_stated"]
VERDICTS = ["supported", "wrong_field", "too_vague", "not_supported", "unclear", "no_located_evidence", "not_applicable"]


def _value_text(fact: dict[str, Any]) -> str:
    value = fact.get("value")
    text = value.get("text") if isinstance(value, dict) else value
    return str(text or "").strip()


def judge_applies(fact: dict[str, Any]) -> bool:
    """The judge asks whether a VALUE is stated by the passages; a fact the
    extractor reported as not_found, or one with no value, has nothing to
    judge (the question for a human there is whether the paper lacks the fact
    type, which needs the whole paper, not the cited passage)."""
    return bool(_value_text(fact)) and str(fact.get("answer_status")) != "not_found"

FACT_TYPE_DEFINITIONS: dict[str, str] = {
    "instrument": "a named beamline, spectrometer or apparatus used in this paper's own measurements (BT-7, HB-1, a SQUID magnetometer); not a facility, not software",
    "facility": "the laboratory or neutron/x-ray source (NCNR, HFIR, ORNL, SNS), not the instrument",
    "sample": "the specific specimen measured in this paper beyond its form and formula: mass, dimensions, orientation, mosaic, mount, number of pieces",
    "sample_form": "the physical form of the measured sample: single crystal, powder, polycrystal, thin film, co-aligned crystals",
    "material": "the chemical compound, or substitution series, that is the subject of this paper's measurements, not a comparison or reference compound; without abbreviations",
    "material_alias": "an abbreviation the paper defines for a material, stated as 'ABBREVIATION = full name', cited to the defining sentence",
    "composition": "the stoichiometry or doping level of a measured sample: one studied x or y value with its formula, or a range the paper states as a range",
    "crystal_structure": "the crystallographic structure of the measured material: space group, structure type, lattice parameters",
    "magnetic_structure": "the magnetic ordering of the measured material: type of order, moment arrangement, ordering wavevector description",
    "propagation_vector": "the magnetic ordering wavevector k (or q) itself, as a vector; a stated magnetic space group that keeps the full nuclear lattice (Fd'd'2, Pn'ma, ...) states k = 0; scan positions such as Q = (2+xi, 2-xi, 0) are not propagation vectors",
    "spin_direction": "the direction of ordered moments in the magnetic structure",
    "ordered_moment": "the size of the ordered magnetic moment, with units",
    "exchange_interaction": "exchange constants or couplings (J values) stated or fitted",
    "magnetic_anisotropy": "anisotropy terms, constants, fields or easy axes/planes",
    "magnetization": "magnetization values or curves reported",
    "hysteresis_property": "coercive field, remanence or hysteresis loop properties",
    "sample_preparation": "how the sample was made: growth method, flux, annealing, synthesis conditions, cutting or alignment",
    "measurement_kind": "the kind of measurement performed in this paper (inelastic neutron scattering, powder diffraction, susceptibility)",
    "detector_mode": "detector configuration or mode (single detector, PSD, analyzer mode)",
    "instrument_configuration": "the instrument setup used: fixed energies, monochromator/analyzer choices, modes, scattering plane",
    "monochromator": "the monochromator crystal or setting",
    "analyzer": "the analyzer crystal or setting",
    "collimation": "beam collimations used",
    "filter": "beam filters used (pyrolytic graphite, beryllium)",
    "energy_setting": "incident or final neutron energy, or energy transfer settings used",
    "temperature_range": "temperatures at which this paper's measurements were made or its sample was processed (annealing, cooling), with units; a value that gives only one end of a stated range is partial",
    "transition_temperature": "a transition temperature of the measured material (Neel, Curie, superconducting, structural) with its value",
    "magnetic_field": "an applied magnetic field used in the measurements, with magnitude and units",
    "pressure": "an applied pressure used in the measurements, with magnitude and units",
    "sample_environment": "cryostat, furnace, magnet or pressure cell used",
    "data_reduction": "how data were corrected, normalized, fitted or reduced; software used for that",
    "calibration": "calibration procedures or standards",
    "background": "background measurement or subtraction procedure",
    "normalization": "how intensities were normalized (monitor, vanadium, absolute units)",
    "resolution": "instrumental resolution values or how resolution was treated",
    "figure_caption": "the caption text of a figure in this paper",
    "figure_panel_mapping": "which panel of a figure shows which data",
    "source_data_clue": "a clue linking a figure or result to specific scans, files or measurement positions",
    "uncertainty_note": "the paper's convention for error bars or uncertainties",
}


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "enum": SUBJECT_OPTIONS},
            "field": {"type": "string", "enum": FIELD_OPTIONS},
            "suggested_fact_type": {"type": "string", "enum": FACT_TYPES + ["none"]},
            "stated": {"type": "string", "enum": STATED_OPTIONS},
            "note": {"type": "string"},
        },
        "required": ["subject", "field", "suggested_fact_type", "stated", "note"],
    }


def derive_verdict(answers: dict[str, Any]) -> str:
    """Fixed mapping from the three answers to a verdict (no model judgment here)."""
    subject = answers.get("subject")
    stated = answers.get("stated")
    field = answers.get("field")
    if subject == "other_material_or_prior_work" or stated == "not_stated":
        return "not_supported"
    if field == "wrong":
        return "wrong_field"
    if stated == "partial_or_vague":
        return "too_vague"
    if subject == "cannot_tell" or field not in FIELD_OPTIONS or stated != "stated":
        return "unclear"
    return "supported"


def located_passages(fact: dict[str, Any], chunk_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The verified source passages for a fact, each with a little context.

    Only quotes the pipeline located as source text count; the quote is marked
    with « » inside its context so the judge knows what was cited."""
    passages: list[dict[str, Any]] = []
    for ev in fact.get("evidence") or []:
        if ev.get("quote_match") not in _SOUND_QUOTE_MATCHES or str(ev.get("support_status")) == "proposed":
            continue
        chunk = chunk_by_id.get(str(ev.get("chunk_id") or ""))
        text = str((chunk or {}).get("text") or "")
        quote = str(ev.get("quote") or "")
        start, end = ev.get("char_start"), ev.get("char_end")
        if chunk and isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            before = text[max(0, start - _CONTEXT_CHARS):start]
            after = text[end:end + _CONTEXT_CHARS]
            passage = " ".join((before + "«" + text[start:end] + "»" + after).split())
        elif quote:
            passage = "«" + " ".join(quote.split()) + "»"
        else:
            continue
        passages.append({"chunk_id": ev.get("chunk_id"), "passage": passage})
        if len(passages) >= _MAX_PASSAGES:
            break
    return passages


def build_prompt(fact: dict[str, Any], passages: list[dict[str, Any]], *, paper_title: str | None) -> list[dict[str, str]]:
    fact_type = str(fact.get("fact_type") or "")
    value = fact.get("value") or {}
    value_text = value.get("text") if isinstance(value, dict) else str(value)
    system = (
        "You check one extracted fact from one scientific paper. You are given the fact type, the extracted value, "
        "and the exact passages of the paper that the extraction cites; the cited words are marked with « ». "
        "Judge only from these passages. Do not use outside knowledge and do not guess what the rest of the paper says. "
        "Answer the three questions by choosing from the fixed options and return only JSON."
    )
    user = {
        "paper_title": paper_title or "",
        "fact_type": fact_type,
        "fact_type_definition": FACT_TYPE_DEFINITIONS.get(fact_type, ""),
        "extracted_value": value_text,
        "extracted_subject": (value.get("subject") if isinstance(value, dict) else None) or "",
        "passages": passages,
        "questions": {
            "subject": "Is the extracted value about THIS paper's own sample, material or measurement (this_paper), or about a comparison compound, a reference material, prior work by others, or a cited paper (other_material_or_prior_work)? Use cannot_tell only if the passages do not show which.",
            "field": "Given the fact_type definition, is fact_type the right field for this value (correct) or should it be a different field (wrong)? If wrong, give suggested_fact_type from the allowed list, else 'none'.",
            "stated": "Do the passages state the value as given, including its numbers, units and named entities (stated)? Is only part of it there, or is it vaguer than a usable fact, such as a bare threshold or a name with no value (partial_or_vague)? Or is the value not in the passages at all, including values assembled from unrelated sentences (not_stated)?",
        },
        "allowed_fact_types": FACT_TYPES,
        "note_instruction": "In 'note', give the decisive reason in at most 25 words.",
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _call(messages: list[dict[str, str]], *, provider: str, model: str, base_url: str, api_key: str | None) -> tuple[dict[str, Any] | None, str | None, float]:
    schema = _response_schema()
    started = time.monotonic()
    if provider == "ollama":
        body = ollama_chat(base_url, model, messages, format_schema=schema, options={"temperature": 0, "num_predict": 600}, think=False, timeout=180.0)
        content = str((body.get("message") or {}).get("content") or "")
        if not content.strip():
            reason = body.get("done_reason")
            thinking = (body.get("message") or {}).get("thinking")
            return None, f"empty content (done_reason={reason!r}, thinking={'present' if thinking else 'absent'})", round(time.monotonic() - started, 3)
    elif provider == "openai_compatible":
        body = openai_compatible_chat(base_url=base_url, model=model, messages=messages, api_key=api_key, response_format={"type": "json_object"})
        content = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    else:
        raise ValueError(f"unknown provider {provider!r}")
    latency = round(time.monotonic() - started, 3)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, content[:500], latency
    return (parsed if isinstance(parsed, dict) else None), (None if isinstance(parsed, dict) else content[:500]), latency


def judge_paper_facts(payload: dict[str, Any], *, chat: Callable[..., tuple[dict[str, Any] | None, str | None, float]] | None = None) -> dict[str, Any]:
    catalog = Path(str(payload["catalog"])).expanduser()
    fact_file = str(payload.get("fact_file") or "paper_facts.jsonl")
    output_dir = Path(str(payload.get("output_dir") or catalog)).expanduser()
    provider = str(payload.get("provider") or "ollama")
    model = str(payload.get("model") or "")
    base_url = str(payload.get("base_url") or "http://127.0.0.1:11434")
    api_key = payload.get("api_key")
    limit = int(payload.get("limit") or 0)
    only_ids: set[str] | None = None
    if payload.get("fact_ids_from"):
        only_ids = set()
        for line in Path(str(payload["fact_ids_from"])).expanduser().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("object_type", "paper_fact") == "paper_fact" and row.get("object_id"):
                    only_ids.add(str(row["object_id"]))
            else:
                only_ids.add(line)
    if not model:
        return {"status": "missing_input", "errors": ["--model is required"]}
    facts = _load_jsonl(catalog / fact_file)
    chunks = _load_jsonl(catalog / "source_chunks.jsonl")
    papers = _load_jsonl(catalog / "papers.jsonl")
    if not facts:
        return {"status": "missing_input", "errors": [f"{fact_file} is absent or empty in {catalog}"]}
    chunk_by_id = {str(c.get("chunk_id")): c for c in chunks}
    title_by_paper = {str(p.get("paper_id")): str(p.get("title") or "") for p in papers}
    call = chat or _call
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "paper_fact_judgments.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if out_path.is_file() and payload.get("resume", True):
        for row in _load_jsonl(out_path):
            if row.get("prompt_version") != PROMPT_VERSION or row.get("model") != model:
                continue
            fact_id = str(row.get("fact_id"))
            if row.get("verdict") == "unclear" and row.get("answers") is None:
                done.pop(fact_id, None)  # a failed call is retried, not resumed
                continue
            done[fact_id] = row
    counts: dict[str, int] = {"judged": 0, "skipped_no_located_evidence": 0, "skipped_not_selected": 0, "resumed": len(done), "errors": 0}
    verdicts: dict[str, int] = {}
    rows: list[dict[str, Any]] = list(done.values())
    with out_path.open("a", encoding="utf-8") as handle:
        for fact in facts:
            fact_id = str(fact.get("fact_id"))
            if only_ids is not None and fact_id not in only_ids:
                counts["skipped_not_selected"] += 1
                continue
            if _is_quarantined(fact):  # same rule as score-fact-labels; plain within-paper duplicates are judged
                counts["skipped_not_selected"] += 1
                continue
            if fact_id in done:
                continue
            if limit and counts["judged"] >= limit:
                break
            passages = located_passages(fact, chunk_by_id) if judge_applies(fact) else []
            record: dict[str, Any] = {
                "fact_id": fact_id, "paper_id": fact.get("paper_id"), "fact_type": fact.get("fact_type"),
                "answer_status": fact.get("answer_status"), "model": model, "provider": provider,
                "prompt_version": PROMPT_VERSION, "passages_used": [p["chunk_id"] for p in passages],
            }
            if not judge_applies(fact):
                record.update({"verdict": "not_applicable", "answers": None})
                counts["skipped_not_applicable"] = counts.get("skipped_not_applicable", 0) + 1
            elif not passages:
                record.update({"verdict": "no_located_evidence", "answers": None})
                counts["skipped_no_located_evidence"] += 1
            else:
                messages = build_prompt(fact, passages, paper_title=title_by_paper.get(str(fact.get("paper_id"))))
                try:
                    answers, error, latency = call(messages, provider=provider, model=model, base_url=base_url, api_key=api_key)
                except Exception as exc:  # the job keeps going; the fact is recorded as unjudged
                    answers, error, latency = None, f"{type(exc).__name__}: {exc}"[:300], 0.0
                record["latency_s"] = latency
                if answers is None:
                    record.update({"verdict": "unclear", "answers": None, "error": error})
                    counts["errors"] += 1
                else:
                    record.update({
                        "verdict": derive_verdict(answers), "answers": answers,
                        "suggested_fact_type": answers.get("suggested_fact_type") if answers.get("field") == "wrong" else None,
                        "note": str(answers.get("note") or "")[:300],
                    })
                counts["judged"] += 1
            verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(record)
    report = {
        "status": "ok", "workflow_name": "judge-paper-facts", "catalog": str(catalog), "fact_file": fact_file,
        "model": model, "provider": provider, "prompt_version": PROMPT_VERSION, "counts": counts,
        "verdicts": verdicts, "judgments": str(out_path),
        "policy": [
            "The judge sees only located source passages with short context, never the whole paper.",
            "Answers come from fixed options; the verdict is derived by a fixed rule.",
            "A judge verdict never accepts a fact; it keeps the fact in human review.",
        ],
    }
    (output_dir / "paper_fact_judge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def apply_judgments(catalog_dir: str | Path, judgments_path: str | Path, *, fact_file: str = "paper_facts.jsonl", output_name: str = "paper_facts.judged.jsonl") -> dict[str, Any]:
    """Record judge verdicts on a copy of the fact file: metadata.judge plus a
    warning, and human_review_required for anything but 'supported'. Status
    and ids are untouched."""
    catalog = Path(catalog_dir).expanduser()
    facts = _load_jsonl(catalog / fact_file)
    judged = {str(r.get("fact_id")): r for r in _load_jsonl(Path(judgments_path).expanduser())}
    applied = 0
    for fact in facts:
        row = judged.get(str(fact.get("fact_id")))
        if not row or row.get("verdict") in (None, "no_located_evidence", "not_applicable"):
            continue
        if not judge_applies(fact):  # a verdict stored before this rule existed (not_found / empty value)
            continue
        meta = dict(fact.get("metadata") or {})
        meta["judge"] = {k: row.get(k) for k in ("model", "prompt_version", "verdict", "answers", "suggested_fact_type", "note")}
        fact["metadata"] = meta
        if row["verdict"] != "supported":
            fact["human_review_required"] = True
            fact["warnings"] = list(fact.get("warnings") or []) + [
                f"Second-pass judge ({row.get('model')}): {row['verdict']}"
                + (f", suggested field {row['suggested_fact_type']}" if row.get("suggested_fact_type") else "")
                + (f": {row['note']}" if row.get("note") else "")
            ]
        applied += 1
    target = catalog / output_name
    target.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
    return {"status": "ok", "applied": applied, "output": str(target)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
