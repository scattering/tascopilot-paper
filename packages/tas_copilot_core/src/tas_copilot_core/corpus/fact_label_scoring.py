"""Score a paper-fact catalog against human review labels.

The labels (ReviewLabel JSONL, object_type paper_fact) are the regression
fixture: fact ids survive revalidation, so the same labels score the catalog
before and after any change to the extractor's post-processing. Nothing here
judges a fact; it only tabulates the machine's fields against the human
verdicts. Labels and catalogs hold private paper text and stay in ignored
local directories.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tas_copilot_core.corpus.paper_facts import _locate_quote, _normalize_alnum

CITATION_DEFECT_STATUS = "ocr_or_chunking_problem"
_SOUND_QUOTE_MATCHES = {"verbatim", "snapped", "split_fragment"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_quarantined(fact: dict[str, Any]) -> bool:
    meta = fact.get("metadata") or {}
    return bool(meta.get("integrity_status")) or any("Catalog integrity" in str(w) for w in fact.get("warnings") or [])


def _machine_flagged(fact: dict[str, Any]) -> bool:
    """The catalog's own citation flag: some evidence could not be verified."""
    if any(str(ev.get("support_status")) == "proposed" for ev in fact.get("evidence") or []):
        return True
    return any("no evidence quote could be located" in str(w) for w in fact.get("warnings") or [])


def _citation_state(fact: dict[str, Any], norm_chunk_texts: dict[str, str], paper_by_chunk: dict[str, str]) -> str:
    """sound: every quote is source text at its cited chunk; none: no quotes;
    defective: at least one quote is not."""
    quotes = [ev for ev in fact.get("evidence") or [] if str(ev.get("quote") or "").strip()]
    if not quotes:
        return "none"
    for ev in quotes:
        match = ev.get("quote_match")
        if match is not None:
            ok = match in _SOUND_QUOTE_MATCHES and str(ev.get("support_status")) != "proposed"
        else:  # catalog written before quotes were snapped to the source
            located = _locate_quote(
                str(ev.get("quote")), cited_chunk_id=str(ev.get("chunk_id") or ""), fact_paper_id=str(fact.get("paper_id") or ""),
                norm_chunk_texts=norm_chunk_texts, paper_by_chunk=paper_by_chunk,
            )
            ok = located["status"] == "in_cited_chunk" and str(ev.get("support_status")) != "proposed"
        if not ok:
            return "defective"
    return "sound"


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score_fact_labels(payload: dict[str, Any]) -> dict[str, Any]:
    catalog = Path(str(payload["catalog"])).expanduser()
    fact_file = str(payload.get("fact_file") or "paper_facts.jsonl")
    label_paths = [Path(str(p)).expanduser() for p in payload.get("labels") or []]
    include_quarantined = bool(payload.get("include_quarantined"))
    facts = {str(f.get("fact_id")): f for f in _load_jsonl(catalog / fact_file)}
    if not facts:
        return {"status": "missing_input", "errors": [f"{fact_file} is absent or empty in {catalog}"]}
    labels: dict[str, dict[str, Any]] = {}
    missing_records = 0
    unknown_ids = 0
    for path in label_paths:
        for row in _load_jsonl(path):
            if row.get("object_type") != "paper_fact":
                continue
            if str(row.get("type_specific_status")) == "missing" or (row.get("metadata") or {}).get("missing_fact"):
                missing_records += 1
                continue
            object_id = str(row.get("object_id"))
            if object_id not in facts:
                unknown_ids += 1
                continue
            labels[object_id] = row  # later rows win
    if not labels:
        return {"status": "missing_input", "errors": ["No paper_fact labels matched this catalog."]}
    chunks = _load_jsonl(catalog / "source_chunks.jsonl")
    norm_chunk_texts = {str(c.get("chunk_id")): _normalize_alnum(str(c.get("text") or "")) for c in chunks}
    paper_by_chunk = {str(c.get("chunk_id")): str(c.get("paper_id") or "") for c in chunks}

    scored = [(facts[fid], row) for fid, row in labels.items() if include_quarantined or not _is_quarantined(facts[fid])]
    excluded = len(labels) - len(scored)
    by_status: dict[str, Counter] = defaultdict(Counter)
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_state: dict[str, Counter] = defaultdict(Counter)
    quote_match: Counter = Counter()
    flag = Counter()
    for fact, row in scored:
        human = str(row.get("type_specific_status"))
        positive = str(row.get("verdict")) == "positive"
        by_status[str(fact.get("answer_status"))][human] += 1
        by_type[str(fact.get("fact_type"))]["n"] += 1
        by_type[str(fact.get("fact_type"))]["correct"] += int(positive)
        state = _citation_state(fact, norm_chunk_texts, paper_by_chunk)
        by_state[state][human] += 1
        for ev in fact.get("evidence") or []:
            if str(ev.get("quote") or "").strip():
                quote_match[str(ev.get("quote_match") or "not_recorded")] += 1
        flagged = _machine_flagged(fact)
        flag["flagged"] += int(flagged)
        flag["flagged_negative"] += int(flagged and not positive)
        flag["unflagged"] += int(not flagged)
        flag["unflagged_negative"] += int(not flagged and not positive)
        flag["citation_defects"] += int(human == CITATION_DEFECT_STATUS)
        flag["citation_defects_flagged"] += int(flagged and human == CITATION_DEFECT_STATUS)

    def status_row(name: str) -> dict[str, Any]:
        counts = by_status.get(name, Counter())
        total = sum(counts.values())
        return {"n": total, "human_status": dict(counts), "correct_rate": _rate(counts.get("correct", 0), total)}

    defects = by_state.get("defective", Counter())
    sound = by_state.get("sound", Counter())
    human_defects = sum(c.get(CITATION_DEFECT_STATUS, 0) for c in by_state.values())
    report = {
        "status": "ok",
        "workflow_name": "score-fact-labels",
        "catalog": str(catalog),
        "fact_file": fact_file,
        "labels": [str(p) for p in label_paths],
        "coverage": {
            "catalog_facts": len(facts),
            "labeled_facts_scored": len(scored),
            "labeled_facts_excluded_quarantined": excluded,
            "labeled_papers": len({str(f.get("paper_id")) for f, _ in scored}),
            "missing_fact_records": missing_records,
            "labels_for_unknown_fact_ids": unknown_ids,
        },
        "by_extractor_status": {name: status_row(name) for name in sorted(by_status)},
        "citation_state": {
            "by_state": {state: dict(counts) for state, counts in sorted(by_state.items())},
            "human_citation_defects": human_defects,
            "human_citation_defects_with_sound_citation_now": sound.get(CITATION_DEFECT_STATUS, 0),
            "human_citation_defects_still_defective": defects.get(CITATION_DEFECT_STATUS, 0),
            "human_correct_with_defective_citation": defects.get("correct", 0),
            "quote_match_counts": dict(quote_match),
        },
        "machine_flag": {
            **dict(flag),
            "precision_vs_human_negative": _rate(flag["flagged_negative"], flag["flagged"]),
            "negative_rate_unflagged": _rate(flag["unflagged_negative"], flag["unflagged"]),
            "recall_of_citation_defects": _rate(flag["citation_defects_flagged"], flag["citation_defects"]),
        },
        "by_fact_type": {
            name: {"n": c["n"], "correct_rate": _rate(c["correct"], c["n"])}
            for name, c in sorted(by_type.items(), key=lambda kv: -kv[1]["n"])
        },
        "safe_conclusion": (
            "Tabulation of machine fields against human verdicts on the labeled facts only. "
            "A sound citation means the stored quote is source text at its cited chunk; it does not mean the quote supports the value."
        ),
    }
    judgments_path = payload.get("judgments")
    if judgments_path:
        judged = {str(r.get("fact_id")): r for r in _load_jsonl(Path(str(judgments_path)).expanduser())}
        value_errors = {"wrong_field", "too_vague", "not_supported"}
        confusion: dict[str, Counter] = defaultdict(Counter)
        jc = Counter()
        for fact, row in scored:
            judgment = judged.get(str(fact.get("fact_id")))
            if not judgment:
                jc["unjudged"] += 1
                continue
            human = str(row.get("type_specific_status"))
            verdict = str(judgment.get("verdict"))
            confusion[human][verdict] += 1
            flagged = verdict in value_errors
            jc["judged"] += 1
            if human in value_errors:
                jc["human_value_errors"] += 1
                jc["human_value_errors_flagged"] += int(flagged)
                jc["human_value_errors_same_status"] += int(verdict == human)
            if human == "correct":
                jc["human_correct"] += 1
                jc["human_correct_flagged"] += int(flagged)
            if human == CITATION_DEFECT_STATUS:
                jc["human_citation_defects_flagged"] += int(flagged)
        report["judge"] = {
            "judgments": str(judgments_path),
            **dict(jc),
            "recall_of_value_errors": _rate(jc["human_value_errors_flagged"], jc["human_value_errors"]),
            "same_status_on_value_errors": _rate(jc["human_value_errors_same_status"], jc["human_value_errors"]),
            "false_flag_rate_on_correct": _rate(jc["human_correct_flagged"], jc["human_correct"]),
            "confusion_human_vs_judge": {h: dict(c) for h, c in sorted(confusion.items())},
        }
    output_dir = payload.get("output_dir")
    if output_dir:
        out = Path(str(output_dir)).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "fact_label_score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Paper-fact label score", "",
            f"Catalog `{catalog}` ({fact_file}); {len(scored)} labeled facts on {report['coverage']['labeled_papers']} papers"
            f" ({excluded} quarantined excluded, {missing_records} missing-fact records).", "",
            "| Extractor status | n | correct | citation defect | other |", "|---|---|---|---|---|",
        ]
        for name, row in report["by_extractor_status"].items():
            hs = row["human_status"]
            other = row["n"] - hs.get("correct", 0) - hs.get(CITATION_DEFECT_STATUS, 0)
            lines.append(f"| {name} | {row['n']} | {hs.get('correct', 0)} | {hs.get(CITATION_DEFECT_STATUS, 0)} | {other} |")
        cs, mf = report["citation_state"], report["machine_flag"]
        lines += [
            "", f"Human citation defects: {cs['human_citation_defects']}; with a sound citation in this catalog: "
            f"{cs['human_citation_defects_with_sound_citation_now']}; still defective: {cs['human_citation_defects_still_defective']}.",
            f"Machine flag: {mf['flagged']} flagged, {mf['flagged_negative']} human-negative; "
            f"catches {mf['citation_defects_flagged']} of {mf['citation_defects']} citation defects.", "",
            report["safe_conclusion"], "",
        ]
        if report.get("judge"):
            j = report["judge"]
            lines += ["## Second-pass judge", "",
                      f"Judged {j.get('judged', 0)} labeled facts ({j.get('unjudged', 0)} unjudged). Flags {j.get('human_value_errors_flagged', 0)} of "
                      f"{j.get('human_value_errors', 0)} human value errors (same status {j.get('human_value_errors_same_status', 0)}); "
                      f"flags {j.get('human_correct_flagged', 0)} of {j.get('human_correct', 0)} human-correct facts.", "",
                      "| human | " + " | ".join(sorted({v for c in j["confusion_human_vs_judge"].values() for v in c})) + " |"]
            cols = sorted({v for c in j["confusion_human_vs_judge"].values() for v in c})
            lines.append("|---|" + "---|" * len(cols))
            for h, c in j["confusion_human_vs_judge"].items():
                lines.append(f"| {h} | " + " | ".join(str(c.get(v, 0)) for v in cols) + " |")
            lines.append("")
        (out / "fact_label_score.md").write_text("\n".join(lines), encoding="utf-8")
        report["generated_artifacts"] = {"json": str(out / "fact_label_score.json"), "markdown": str(out / "fact_label_score.md")}
    return report
