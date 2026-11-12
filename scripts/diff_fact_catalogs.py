"""Diff two paper-fact catalogs of the same papers and carry human labels across.

Fact ids include the fact's position in the model output, so a re-extraction
gives new ids even for identical facts. Facts are matched instead on
(paper_id, fact_type, normalized value): an exact value match carries the
label as is; a containment match (one value inside the other, both at least
six characters) carries it marked `carried: approx`; a fact of the same type
whose located citation overlaps the old fact's citation in the same chunk is
the same fact restated ("INS experiment" vs "inelastic neutron scattering")
and carries it marked `carried: evidence`; approx and evidence matches are
listed for a second look. Everything else is new (needs a label) or dropped
(its label is reported so recall loss is visible). The carried labels file is ReviewLabel JSONL keyed
to the NEW catalog's fact ids, for `fact-eval-dashboard --resume-labels` and
`score-fact-labels --labels`.

    python scripts/diff_fact_catalogs.py --old local_runs/gold_eval_panel_48papers_20261002/catalog \
        --new local_runs/gold_v1_20261018/catalog --labels local_runs/gold_labels_merged_20261017.jsonl \
        --out local_runs/gold_v1_20261018/diff
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "tas_copilot_core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "tas_copilot_contracts" / "src"))
from tas_copilot_core.instrument_aliases import load_registry  # noqa: E402

_ALIASES = load_registry()


def canonical_ids(fact: dict) -> set[str]:
    """Instrument/facility facts compare by the curated alias registry, so
    "BT-7" and "the BT-7 thermal triple-axis spectrometer" are one fact."""
    text = value_text(fact)
    if fact.get("fact_type") == "instrument":
        return {m.id for m in _ALIASES.instruments_in(text)}
    if fact.get("fact_type") == "facility":
        return {m.id for m in _ALIASES.facilities_in(text)}
    return set()


def norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def value_text(fact: dict) -> str:
    value = fact.get("value")
    return str(value.get("text") if isinstance(value, dict) else value or "")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spans(fact: dict) -> list[tuple[str, int, int]]:
    out = []
    for e in fact.get("evidence") or []:
        if e.get("chunk_id") and isinstance(e.get("char_start"), int) and isinstance(e.get("char_end"), int):
            out.append((str(e["chunk_id"]), int(e["char_start"]), int(e["char_end"])))
    return out


def evidence_overlaps(a: dict, b: dict) -> bool:
    for ca, sa, ea in spans(a):
        for cb, sb, eb in spans(b):
            if ca == cb and min(ea, eb) - max(sa, sb) > 0:
                return True
    return False


def quarantined(fact: dict) -> bool:
    meta = fact.get("metadata") or {}
    return bool(meta.get("integrity_status")) or bool(meta.get("malformed_value"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--labels", required=True, help="ReviewLabel JSONL keyed to the OLD catalog's fact ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old = [f for f in load(Path(a.old) / "paper_facts.jsonl") if not quarantined(f)]
    new = [f for f in load(Path(a.new) / "paper_facts.jsonl") if not quarantined(f)]
    labels = {r["object_id"]: r for r in load(Path(a.labels)) if r.get("object_type", "paper_fact") == "paper_fact" and not (r.get("metadata") or {}).get("missing_fact")}
    new_papers = {f["paper_id"] for f in new}
    old = [f for f in old if f["paper_id"] in new_papers]  # compare only papers the new run covered
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    by_key_new: dict[tuple, list[dict]] = collections.defaultdict(list)
    for f in new:
        by_key_new[(f["paper_id"], f["fact_type"])].append(f)
    carried: list[dict] = []
    matched_new: set[str] = set()
    rows: list[dict] = []
    for f in old:
        cands = by_key_new.get((f["paper_id"], f["fact_type"]), [])
        v = norm(value_text(f))
        exact = [c for c in cands if c["fact_id"] not in matched_new and norm(value_text(c)) == v and v]
        approx = [c for c in cands if c["fact_id"] not in matched_new and v and len(v) >= 6 and (v in norm(value_text(c)) or norm(value_text(c)) in v)]
        ids_old = canonical_ids(f)
        by_alias = [c for c in cands if c["fact_id"] not in matched_new and ids_old and ids_old == canonical_ids(c)]
        by_evidence = [c for c in cands if c["fact_id"] not in matched_new and evidence_overlaps(f, c)]
        match, kind = (exact[0], "exact") if exact else ((approx[0], "approx") if approx else ((by_alias[0], "alias") if by_alias else ((by_evidence[0], "evidence") if by_evidence else (None, "dropped"))))
        label = labels.get(f["fact_id"])
        if match is not None:
            matched_new.add(match["fact_id"])
            if label:
                r = dict(label)
                r["object_id"] = match["fact_id"]
                r.setdefault("metadata", {})
                r["metadata"] = dict(r["metadata"], carried=kind, carried_from=f["fact_id"], old_value=value_text(f), new_value=value_text(match))
                carried.append(r)
        rows.append({"paper_id": f["paper_id"], "fact_type": f["fact_type"], "old_fact_id": f["fact_id"], "old_value": value_text(f), "old_status": f.get("answer_status"),
                     "label": (label or {}).get("type_specific_status"), "match": kind, "new_fact_id": match["fact_id"] if match else None, "new_value": value_text(match) if match else None})
    new_only = [f for f in new if f["fact_id"] not in matched_new]
    for f in new_only:
        rows.append({"paper_id": f["paper_id"], "fact_type": f["fact_type"], "old_fact_id": None, "old_value": None, "old_status": None, "label": None, "match": "new", "new_fact_id": f["fact_id"], "new_value": value_text(f), "new_status": f.get("answer_status")})

    (out / "carried_labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in carried), encoding="utf-8")
    (out / "fact_diff.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    kinds = collections.Counter(r["match"] for r in rows)
    dropped_labels = collections.Counter(r["label"] for r in rows if r["match"] == "dropped" and r["label"])
    sup_old = sum(1 for f in old if f.get("answer_status") == "supported"); sup_new = sum(1 for f in new if f.get("answer_status") == "supported")
    def located(fs): return sum(1 for f in fs for e in f.get("evidence") or [] if e.get("quote_match") in ("verbatim", "snapped", "split_fragment"))
    def unlocated(fs): return sum(1 for f in fs for e in f.get("evidence") or [] if e.get("quote") and e.get("quote_match") in (None, "unverified"))
    def uncited(fs): return sum(1 for f in fs if "downgraded" in " ".join(f.get("warnings") or []))
    summary = {
        "papers": len(new_papers), "old_facts": len(old), "new_facts": len(new), "old_supported": sup_old, "new_supported": sup_new,
        "old_quotes_located": located(old), "old_quotes_unlocated": unlocated(old), "new_quotes_located": located(new), "new_quotes_unlocated": unlocated(new),
        "old_uncited_downgraded": uncited(old), "new_uncited_downgraded": uncited(new),
        "matches": dict(kinds), "labels_carried": len(carried), "labels_carried_exact": sum(1 for r in carried if r["metadata"]["carried"] == "exact"),
        "dropped_old_facts_by_label": dict(dropped_labels), "new_facts_needing_labels": kinds.get("new", 0) + sum(1 for r in carried if r["metadata"]["carried"] != "exact"),
    }
    (out / "diff_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = ["# Fact catalog diff", "", f"Papers compared: {summary['papers']}", "",
          "| | old | new |", "|---|---|---|",
          f"| facts (non-quarantined) | {summary['old_facts']} | {summary['new_facts']} |",
          f"| supported | {sup_old} | {sup_new} |",
          f"| quotes located / unlocated | {summary['old_quotes_located']} / {summary['old_quotes_unlocated']} | {summary['new_quotes_located']} / {summary['new_quotes_unlocated']} |",
          f"| supported but uncited (downgraded) | {summary['old_uncited_downgraded']} | {summary['new_uncited_downgraded']} |", "",
          f"Old facts matched in new: exact {kinds.get('exact', 0)}, approximate {kinds.get('approx', 0)}, by instrument/facility alias {kinds.get('alias', 0)}, by citation overlap {kinds.get('evidence', 0)}, dropped {kinds.get('dropped', 0)}; new facts without an old counterpart: {kinds.get('new', 0)}.",
          f"Labels carried to new ids: {len(carried)} ({summary['labels_carried_exact']} exact). Dropped old facts by human label: {dict(dropped_labels)}.",
          f"Facts needing a label in the new catalog: {summary['new_facts_needing_labels']} (new + approximate and citation-overlap matches)."]
    (out / "diff_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
