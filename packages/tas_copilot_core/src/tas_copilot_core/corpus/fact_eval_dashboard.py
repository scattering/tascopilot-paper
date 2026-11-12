from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote
from typing import Any

from tas_copilot_core.corpus.evidence_graph_proposer import RELATION_EDGE_TYPES
from tas_copilot_core.corpus.paper_facts import FACT_TYPES, _alnum_offsets, _compact_alignment, _locate_quote, _normalize_alnum
from tas_copilot_core.instrument_aliases import load_registry as _load_aliases

FACT_VERDICTS = [
    "correct",
    "wrong_field",
    "too_vague",
    "not_supported",
    "ocr_or_chunking_problem",
]
EDGE_VERDICTS = ["accepted", "rejected", "inconclusive"]
_MAX_CHUNK_CHARS = 6000

def build_fact_eval_dashboard(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generate the static gold-eval labeling panel from a catalog.

    The panel is a single local HTML file: extracted facts and proposed
    relation edges render as labeling cards with the cited quotes and full
    chunk text inline, verdicts follow the gold-eval vocabulary, and the
    export button downloads ReviewLabel-compatible JSONL that feeds both
    `review labels ingest` and `corpus evidence-graph-apply-labels`.
    Nothing leaves the browser; drafts persist in localStorage.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not catalog_dir.exists():
        return {"status": "missing_input", "errors": [{"error": f"Corpus catalog does not exist: {catalog_dir}"}]}

    # --fact-file lets the panel be built from paper_facts.judged.jsonl, so the
    # second-pass judge's flags (carried as warnings and metadata.judge) show up
    # and flagged cards come first; the judge never pre-fills a verdict.
    fact_file = str(arguments.get("fact_file") or "paper_facts.jsonl")
    facts = _load_jsonl(catalog_dir / fact_file)
    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    # Prior labels (for example an initial AI flag pass, or a previous reviewer's
    # export) are shown on each card as context, never as the card's own verdict:
    # the reviewer still labels every card, and the export carries only their labels.
    prior_by_fact: dict[str, list[dict[str, Any]]] = {}
    for label_path in [Path(str(p)).expanduser() for p in (arguments.get("prior_labels") or [])]:
        for label in _load_jsonl(label_path):
            if str(label.get("object_type")) != "paper_fact" or not label.get("object_id"):
                continue
            prior_by_fact.setdefault(str(label["object_id"]), []).append(
                {
                    "reviewer": label.get("reviewer_id") or label.get("reviewer_role") or label_path.stem,
                    "verdict": label.get("verdict"),
                    "status": label.get("type_specific_status"),
                    "notes": label.get("review_notes") or "",
                    "reviewed_at": label.get("reviewed_at"),
                }
            )
    # Resume labels are the reviewer's own earlier export. They pre-fill the
    # panel so work continues where it stopped, on any machine or browser. A
    # fact the reviewer marked as a citation defect, whose citation the pipeline
    # has since repaired, is NOT pre-filled: it is queued for a second look with
    # the earlier label shown beside it.
    resume: dict[str, dict[str, Any]] = {"facts": {}, "edges": {}}
    resume_missing: list[dict[str, Any]] = []
    # Which resume files predate the citation repair: only THEIR citation-defect
    # labels are queued for a second look. A later file (the reviewer's own
    # second-look pass) is final, so re-labeling a card ocr_or_chunking_problem
    # does not send it back to the queue. Default: every resume file (old behaviour).
    second_look_files = {str(Path(str(p)).expanduser()) for p in (arguments.get("second_look_from") or [])}
    predates_repair: set[str] = set()
    for label_path in [Path(str(p)).expanduser() for p in (arguments.get("resume_labels") or [])]:
        is_old = not second_look_files or str(label_path) in second_look_files
        for label in _load_jsonl(label_path):
            meta = label.get("metadata") or {}
            status = str(label.get("type_specific_status") or "")
            if not label.get("object_id") or not status:
                continue
            record = {"verdict": status, "notes": label.get("review_notes") or "", "expected_value": meta.get("expected_value") or ""}
            if str(label.get("object_type")) == "evidence_graph_edge":
                resume["edges"][str(label["object_id"])] = record
            elif status == "missing" or meta.get("missing_fact"):
                key = (str(meta.get("paper_id") or ""), str(meta.get("fact_type") or label.get("object_secondary_id") or ""), str(meta.get("expected_value") or ""))
                if any((m["paper_id"], m["fact_type"], m["expected_value"]) == key for m in resume_missing):
                    continue  # the same record exported twice (two resume files)
                resume_missing.append({
                    "paper_id": meta.get("paper_id") or "", "fact_type": meta.get("fact_type") or label.get("object_secondary_id") or "",
                    "expected_value": meta.get("expected_value") or "", "chunk_hint": meta.get("chunk_hint") or "", "notes": label.get("review_notes") or "",
                })
            elif str(label.get("object_type")) == "paper_fact":
                resume["facts"][str(label["object_id"])] = record
                if is_old:
                    predates_repair.add(str(label["object_id"]))
                else:
                    predates_repair.discard(str(label["object_id"]))  # a newer label settles it
    nodes = _load_jsonl(catalog_dir / "evidence_graph_nodes.jsonl")
    # --no-edges: a facts-only page (e.g. a v1 sample whose catalog still carries v0 edges)
    edges = [] if arguments.get("no_edges") else _load_jsonl(catalog_dir / "evidence_graph_edges.jsonl")
    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    if not facts:
        return {"status": "missing_input", "errors": [{"error": f"{fact_file} is absent or empty; run bounded fact extraction first."}]}

    chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks if chunk.get("chunk_id")}
    label_by_node = {str(node.get("node_id")): str(node.get("label") or node.get("node_id")) for node in nodes if node.get("node_id")}
    title_by_paper = {str(paper.get("paper_id")): str(paper.get("title") or paper.get("paper_id")) for paper in papers if paper.get("paper_id")}
    norm_chunk_texts = {cid: _normalize_alnum(str(chunk.get("text") or "")) for cid, chunk in chunk_by_id.items()}
    paper_by_chunk = {cid: str(chunk.get("paper_id") or "") for cid, chunk in chunk_by_id.items()}

    fact_cards = [
        _fact_card(
            fact,
            chunk_by_id=chunk_by_id,
            title_by_paper=title_by_paper,
            norm_chunk_texts=norm_chunk_texts,
            paper_by_chunk=paper_by_chunk,
        )
        for fact in facts
        if fact.get("fact_id")
    ]
    recheck_ids: list[str] = []
    for card in fact_cards:
        fact_id = str(card.get("fact_id"))
        card["prior_labels"] = prior_by_fact.get(fact_id, [])
        earlier = resume["facts"].get(fact_id)
        if earlier and earlier["verdict"] == "ocr_or_chunking_problem" and card.get("citation_repaired") and fact_id in predates_repair:
            card["recheck"] = True
            card["earlier_label"] = resume["facts"].pop(fact_id)
            recheck_ids.append(fact_id)
        card["resumed"] = fact_id in resume["facts"]
    # Machine-flagged citations surface first so review effort lands where the
    # pipeline already knows something is wrong.
    # Quarantined documents and duplicate copies go last and are not counted
    # toward labelling progress; they stay visible so nothing is hidden.
    # With resume labels the order is: second looks, unlabeled, already labeled.
    def _order(card: dict[str, Any]) -> tuple[int, int]:
        if card.get("integrity_status") == "content_mismatch" or card.get("duplicate_of") or card.get("malformed_value"):
            return (3, 0)
        stage = 0 if card.get("recheck") else (2 if card.get("resumed") else 1)
        return (stage, 0 if card.get("citation_flagged") or card.get("judge_flagged") else 1)

    fact_cards.sort(key=_order)
    # --only-second-looks: a page with nothing but the cards whose earlier
    # citation-defect label needs re-judging, for a reviewer who has only
    # those to do. Same storage key as the full panel, so labels made here show
    # up there and the export/post carries everything from both pages.
    only_second_looks = bool(arguments.get("only_second_looks"))
    if only_second_looks:
        # a second look on a quarantined or duplicate paper is not reviewable; leave it off this page
        fact_cards = [c for c in fact_cards if c.get("recheck") and not (c.get("integrity_status") == "content_mismatch" or c.get("duplicate_of") or c.get("malformed_value"))]
    # --only-ids: a page restricted to listed fact ids (a random sample for an
    # unbiased accuracy estimate, or the facts a diff marked as changed).
    if arguments.get("only_ids"):
        wanted = {line.strip() for line in Path(str(arguments["only_ids"])).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()}
        fact_cards = [c for c in fact_cards if str(c.get("fact_id")) in wanted]
    # --group-by-paper keeps one paper's cards together so the reviewer reads
    # one paper at a time. Papers with pending work come first; inside a paper
    # the order is the same as above (second looks, flagged, unlabeled, labeled,
    # sidelined). The paper's chunk texts ride along once so a fact with no
    # evidence can be checked against the whole paper without leaving the group.
    group_by_paper = bool(arguments.get("group_by_paper"))
    paper_groups: list[dict[str, Any]] = []
    paper_chunks: dict[str, list[dict[str, Any]]] = {}
    if group_by_paper:
        by_paper: dict[str, list[dict[str, Any]]] = {}
        for card in fact_cards:
            by_paper.setdefault(str(card.get("paper_id") or ""), []).append(card)
        def _sidelined(card: dict[str, Any]) -> bool:
            return card.get("integrity_status") == "content_mismatch" or bool(card.get("duplicate_of")) or bool(card.get("malformed_value"))
        pdf_dir = str(arguments.get("pdf_dir") or "").rstrip("/")
        # Optional sidecar (paper_pdf_paths.jsonl, or --pdf-paths): paper_id -> path
        # relative to --pdf-dir, for corpora laid out in subfolders (e.g. by year).
        pdf_relpath: dict[str, str] = {}
        sidecar = Path(str(arguments.get("pdf_paths"))).expanduser() if arguments.get("pdf_paths") else catalog_dir / "paper_pdf_paths.jsonl"
        for row in _load_jsonl(sidecar):
            if row.get("paper_id") and row.get("pdf_relpath"):
                pdf_relpath[str(row["paper_id"])] = str(row["pdf_relpath"])
        meta_by_paper = {str(p.get("paper_id")): (p.get("metadata") or {}) for p in papers if p.get("paper_id")}
        doi_by_paper = {str(p.get("paper_id")): str(p.get("doi") or "") for p in papers if p.get("paper_id")}
        for paper_id, cards in by_paper.items():
            pdf_name = str(meta_by_paper.get(paper_id, {}).get("pdf_name") or "")
            pdf_link = ""
            rel = pdf_relpath.get(paper_id) or pdf_name
            if pdf_dir and rel:
                # a URL is used as is; an absolute path becomes a file:// URL; a relative
                # path (e.g. "pdfs") stays relative, for a folder shipped next to the page
                base = pdf_dir if "://" in pdf_dir or not pdf_dir.startswith("/") else "file://" + quote(pdf_dir)
                pdf_link = base + "/" + quote(rel)
            live = [c for c in cards if not _sidelined(c)]
            second = sum(1 for c in live if c.get("recheck"))
            labeled = sum(1 for c in live if c.get("resumed"))
            paper_groups.append({
                "paper_id": paper_id,
                "title": title_by_paper.get(paper_id, paper_id),
                "total": len(live), "sidelined": len(cards) - len(live),
                "second_look": second, "labeled": labeled, "pending": len(live) - labeled,
                "flagged": sum(1 for c in live if not c.get("resumed") and (c.get("citation_flagged") or c.get("judge_flagged"))),
                "pdf_name": pdf_name, "pdf_link": pdf_link,
                "doi": doi_by_paper.get(paper_id, "") if "/" in doi_by_paper.get(paper_id, "") and "ToBeAssigned" not in doi_by_paper.get(paper_id, "") else "",
            })
        # papers with the most pending work first; finished papers after; quarantined-only last
        paper_groups.sort(key=lambda g: (g["total"] == 0, -(g["second_look"] + g["pending"]), g["title"].lower()))
        order_index = {g["paper_id"]: i for i, g in enumerate(paper_groups)}
        fact_cards.sort(key=lambda c: (order_index.get(str(c.get("paper_id") or ""), len(order_index)), _order(c)))
        for chunk in chunks:
            pid = str(chunk.get("paper_id") or "")
            if pid in order_index and chunk.get("chunk_id"):
                paper_chunks.setdefault(pid, []).append({"chunk_id": str(chunk["chunk_id"]), "text": str(chunk.get("text") or "")})
    relation_types = set(RELATION_EDGE_TYPES.values())
    sidelined_fact_ids = {
        str(fact.get("fact_id"))
        for fact in facts
        if (fact.get("metadata") or {}).get("integrity_status") or (fact.get("metadata") or {}).get("duplicate_of")
    }
    edge_cards = []
    for edge in edges:
        if str(edge.get("edge_type")) not in relation_types or not edge.get("edge_id"):
            continue
        card = _edge_card(edge, label_by_node=label_by_node)
        endpoints = [str(edge.get(k) or "").removeprefix("PaperFact:") for k in ("source_node_id", "target_node_id")]
        card["sidelined"] = any(fid in sidelined_fact_ids for fid in endpoints)
        edge_cards.append(card)
    # edges on quarantined or duplicate facts go last and are not counted
    edge_cards.sort(key=lambda c: bool(c.get("sidelined")))

    payload = {
        "catalog": str(catalog_dir),
        "fact_verdicts": FACT_VERDICTS,
        "edge_verdicts": EDGE_VERDICTS,
        "fact_types": FACT_TYPES,
        "paper_ids": sorted(title_by_paper) or sorted({str(fact.get("paper_id")) for fact in facts if fact.get("paper_id")}),
        "facts": fact_cards,
        "edges": edge_cards,
        "prior_label_summary": _prior_label_summary(fact_cards),
        "storage_key": str(arguments.get("storage_key") or catalog_dir),
        "seed": {"facts": resume["facts"], "edges": resume["edges"], "missing": resume_missing},
        "recheck_ids": recheck_ids,
        "group_by_paper": group_by_paper,
        "paper_groups": paper_groups,
        "paper_chunks": paper_chunks,
        "only_second_looks": only_second_looks,
        "page_name": str(arguments.get("page_name") or ("second_looks" if only_second_looks else "fact_eval_dashboard")),
    }
    html_path = output_dir / (str(arguments.get("page_name") or ("second_looks" if only_second_looks else "fact_eval_dashboard")) + ".html")
    html_path.write_text(_dashboard_html(payload), encoding="utf-8")

    report = {
        "status": "ok",
        "catalog": str(catalog_dir),
        "fact_file": fact_file,
        "only_second_looks": only_second_looks,
        "fact_count": len(fact_cards),
        "group_by_paper": group_by_paper,
        "paper_count": len(paper_groups),
        "judge_flagged_facts": sum(1 for c in fact_cards if c.get("judge_flagged")),
        "relation_edge_count": len(edge_cards),
        "resumed_fact_labels": len(resume["facts"]),
        "second_look_facts": len(recheck_ids),
        "resumed_missing_records": len(resume_missing),
        "prior_label_summary": _prior_label_summary(fact_cards),
        "artifacts": {"html": str(html_path)},
        "next_steps": [
            "Open fact_eval_dashboard.html locally and label every card.",
            "Add 'missing' records for facts the loop should have extracted but did not.",
            "Export the labels JSONL, then run review labels ingest and corpus evidence-graph-apply-labels.",
            "Do not commit the exported gold JSONL when the catalog holds private paper text.",
        ],
    }
    (output_dir / "fact_eval_dashboard_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["report"] = str(output_dir / "fact_eval_dashboard_report.json")
    return report


def _prior_label_summary(fact_cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_reviewer: dict[str, dict[str, int]] = {}
    for card in fact_cards:
        for label in card.get("prior_labels") or []:
            counts = by_reviewer.setdefault(str(label.get("reviewer")), {})
            key = str(label.get("status") or label.get("verdict") or "unlabelled")
            counts[key] = counts.get(key, 0) + 1
    return {reviewer: dict(sorted(counts.items())) for reviewer, counts in by_reviewer.items()}


def _fact_card(
    fact: dict[str, Any],
    *,
    chunk_by_id: dict[str, dict[str, Any]],
    title_by_paper: dict[str, str],
    norm_chunk_texts: dict[str, str] | None = None,
    paper_by_chunk: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence = []
    citation_flagged = False
    citation_repaired = False
    for item in fact.get("evidence") or []:
        chunk_id = str(item.get("chunk_id") or "")
        chunk = chunk_by_id.get(chunk_id, {})
        citation_check = _locate_quote(
            str(item.get("quote") or ""),
            cited_chunk_id=chunk_id,
            fact_paper_id=str(fact.get("paper_id") or ""),
            norm_chunk_texts=norm_chunk_texts or {},
            paper_by_chunk=paper_by_chunk or {},
        )
        # near_verbatim_in_chunk is informational, not a defect: the quote is the
        # chunk's own text with stray characters. Flagging it sorted correct
        # facts to the top of the review queue as if their evidence were missing.
        if citation_check["status"] not in ("in_cited_chunk", "no_quote", "near_verbatim_in_chunk"):
            citation_flagged = True
        item_warnings = [str(w) for w in item.get("warnings") or []]
        if item.get("quote_as_extracted") or item.get("quote_match") == "split_fragment" or any("re-homed" in w for w in item_warnings):
            citation_repaired = True
        evidence.append(
            {
                "chunk_id": chunk_id,
                "quote": item.get("quote"),
                "quote_as_extracted": item.get("quote_as_extracted"),
                "quote_match": item.get("quote_match"),
                "char_start": item.get("char_start"),
                "char_end": item.get("char_end"),
                "support_status": item.get("support_status"),
                "warnings": item_warnings,
                "citation_check": citation_check,
                "chunk_text": str(chunk.get("text") or "")[:_MAX_CHUNK_CHARS],
            }
        )
    value = fact.get("value")
    value_text = value.get("text") if isinstance(value, dict) else ("" if value is None else str(value))
    judge = (fact.get("metadata") or {}).get("judge") or None
    return {
        "fact_id": fact.get("fact_id"),
        "paper_id": fact.get("paper_id"),
        "paper_title": title_by_paper.get(str(fact.get("paper_id")), str(fact.get("paper_id"))),
        "fact_type": fact.get("fact_type"),
        "answer_status": fact.get("answer_status"),
        "value_text": value_text,
        "value_subject": (fact.get("value") or {}).get("subject") if isinstance(fact.get("value"), dict) else None,
        # curated aliases: "BT-7 thermal triple-axis spectrometer" == BT-7; shown as a chip, never used as the label
        "canonical": _canonical_chips(str(fact.get("fact_type") or ""), value_text),
        "confidence": fact.get("confidence"),
        "model": fact.get("model"),
        "warnings": fact.get("warnings") or [],
        "citation_flagged": citation_flagged,
        # The stored quotes are source text that the pipeline located or moved;
        # they differ from what the model wrote.
        "citation_repaired": not citation_flagged
        and (citation_repaired or any("duplicate evidence" in str(w) for w in fact.get("warnings") or [])),
        "judge": judge,
        "judge_flagged": str((judge or {}).get("verdict")) in {"not_supported", "wrong_field", "too_vague"},
        "value_hits": _value_hits(fact, value_text, evidence, chunk_by_id=chunk_by_id, norm_chunk_texts=norm_chunk_texts or {}, paper_by_chunk=paper_by_chunk or {}),
        "absence_cues": _absence_cues(str(fact.get("fact_type") or ""), str(fact.get("paper_id") or ""), chunk_by_id=chunk_by_id, paper_by_chunk=paper_by_chunk or {}) if fact.get("answer_status") == "not_found" else None,
        "integrity_status": (fact.get("metadata") or {}).get("integrity_status"),
        "duplicate_of": (fact.get("metadata") or {}).get("duplicate_of"),
        # The normalizer discards values that are model output, not paper
        # content (letter-spaced text, repetition loops, serialized fields).
        # Nothing is left to label; the card is sidelined with the discarded text shown.
        "malformed_value": (fact.get("metadata") or {}).get("malformed_value"),
        "discarded_value": str((fact.get("metadata") or {}).get("discarded_value") or "")[:300],
        "evidence": evidence,
    }


# For a fact the extractor reported absent: what the paper would say if it had
# one. The hits (or their absence) are shown so "confirmed absent" is a judgment
# on the text, not on trust. Deliberately broad; a hit is a cue, not a claim.
_ABSENCE_CUES = {
    "instrument": r"\b(bt-?\d|hb-?\d[a-z]?|ng-?\d|spins|macs|sequoia|arcs|hyspec|cncs|wand|nomad|taipan|merlin|in\d{1,2}|d\d{1,2}|spectrometer|diffractometer|beam ?line|ppms|mpms|squid|magnetometer|dilatometer)\b",
    "facility": r"\b(ncnr|nist|hfir|sns|ornl|ill|isis|j-?parc|ansto|psi|sinq|frm ?ii|mlz|lansce|reactor|spallation|synchrotron)\b",
    "sample_environment": r"\b(cryostat|cryomagnet|dilution|refrigerator|furnace|pressure cell|clamp cell|closed[- ]cycle|displex|orange|magnet\b)",
    "magnetic_field": r"\b\d+(\.\d+)?\s*(t|tesla|oe|koe)\b|magnetic field",
    "pressure": r"\b\d+(\.\d+)?\s*(gpa|kbar|mpa|bar)\b|under pressure|hydrostatic",
    "energy_setting": r"\b(e_?[if]|incident energy|final energy|wavelength|λ)\s*=|\b\d+(\.\d+)?\s*mev\b",
    "data_reduction": r"\b(fullprof|gsas|dave|mantid|reductus|spinw|jana|rietveld|refine[dm]|fit(ted)? (with|using))\b",
    "sample_preparation": r"\b(grown|growth|synthesi[sz]ed|annealed|flux method|bridgman|floating[- ]zone|czochralski|sintered|calcined|solid[- ]state reaction)\b",
    "sample_form": r"\b(single[- ]crystal|single crystals|powder|polycrystal|polycrystalline|thin film|co-?aligned)\b",
    "material_alias": r"\(([A-Z]{2,6})\)|hereafter|abbreviated|referred to as",
    "composition": r"\bx\s*=\s*\d|\by\s*=\s*\d|\bx\s*(<|≤|<=)|doping|substitut",
    "collimation": r"\b\d{2,3}['′]?\s*[-–]\s*\d{2,3}['′]?\s*[-–]\s*\d{2,3}|collimat",
    "filter": r"\b(pg|pyrolytic graphite|beryllium|be|sapphire) filter|\bfilter(s)?\b",
    "monochromator": r"monochromator|\bpg\s*\(?0?02\)?|heusler|si\s*\(1 ?1 ?1\)",
    "analyzer": r"analy[sz]er",
    "transition_temperature": r"\bt_?[ncs]\b|n[ée]el|curie|orders? (below|at)|transition (temperature|at)",
    "ordered_moment": r"\d+(\.\d+)?\s*(μ|mu|µ)_?b\b|ordered moment",
    "propagation_vector": r"propagation vector|\bk\s*=\s*\(|wave ?vector\s*(k|q)\b|magnetic space group|shubnikov",
}


def _absence_cues(fact_type: str, paper_id: str, *, chunk_by_id: dict[str, dict[str, Any]], paper_by_chunk: dict[str, str], limit: int = 3) -> list[dict[str, Any]] | None:
    """Sentences of the paper that mention terms of this fact type; None when
    no search is defined for the type (the card then says so)."""
    pattern = _ABSENCE_CUES.get(fact_type)
    if not pattern:
        return None
    cue = re.compile(pattern, re.IGNORECASE)
    out: list[dict[str, Any]] = []
    for chunk_id, chunk in chunk_by_id.items():
        if paper_by_chunk.get(chunk_id) != paper_id:
            continue
        text = str(chunk.get("text") or "")
        for m in cue.finditer(text):
            prev_dot = text.rfind(". ", max(0, m.start() - 220), m.start())
            lo = prev_dot + 2 if prev_dot >= 0 else max(0, m.start() - 160)
            hi_dot = text.find(". ", m.end(), m.end() + 220)
            hi = hi_dot + 1 if hi_dot >= 0 else min(len(text), m.end() + 160)
            passage = " ".join(text[lo:hi].split())
            if any(o["passage"] == passage for o in out):
                continue
            out.append({"chunk_id": chunk_id, "term": m.group(0), "passage": passage})
            if len(out) >= limit:
                return out
    return out


_GENERIC_WORDS = {
    "triple", "axis", "spectrometer", "diffractometer", "beamline", "beam", "line", "instrument", "thermal", "cold",
    "neutron", "synchrotron", "ray", "powder", "single", "crystal", "system", "measurement", "physical", "property",
    "quantum", "design", "high", "flux", "source", "reactor", "national", "laboratory", "center", "centre", "facility",
    "time", "flight", "chopper", "magnet", "cryostat", "sample", "temperature", "field", "energy", "data",
}


def _canonical_chips(fact_type: str, value_text: str) -> list[dict[str, str]]:
    if fact_type not in ("instrument", "facility") or not value_text:
        return []
    reg = _load_aliases()
    hits = reg.instruments_in(value_text) if fact_type == "instrument" else reg.facilities_in(value_text)
    return [{"id": h.id, "label": h.label, "kind": h.instrument_kind or "", "facility": h.facility or ""} for h in hits]


def _value_hits(
    fact: dict[str, Any],
    value_text: str,
    evidence: list[dict[str, Any]],
    *,
    chunk_by_id: dict[str, dict[str, Any]],
    norm_chunk_texts: dict[str, str],
    paper_by_chunk: dict[str, str],
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Where the extracted value's own text occurs in this paper, for a fact
    that cites nothing located. The extractor hedged ("ambiguous", no
    evidence) on facts whose value sits verbatim in the methods paragraph;
    showing that sentence lets the reviewer judge without opening the paper.
    A search hit is not a citation: it is labeled as found, not cited."""
    if any(e.get("quote_match") in ("verbatim", "snapped", "split_fragment") for e in evidence):
        return []
    # A list value ("BT1, NOMAD, HB-2A, BT-7") is looked up item by item; a
    # single value as a whole. Items shorter than three characters are noise.
    raw = str(value_text or "")
    items = [part.strip() for part in re.split(r",|;|\band\b|/", raw) if part.strip()] if len(raw) > 24 else [raw]
    needles = [(item, _normalize_alnum(item)) for item in items]
    needles = [(item, n) for item, n in needles if len(n) >= (6 if len(needles) == 1 else 3)]
    if not needles:
        return []
    paper_id = str(fact.get("paper_id") or "")
    hits: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, int]] = set()
    for item, needle in needles:
        found = False
        matched_word = None
        # The model decorates names ("Taipan triple-axis (thermal)") and the
        # paper orders words differently ("thermal triple-axis spectrometer
        # Taipan"), so after the whole item, try it without parentheticals and
        # then its most distinctive word (a proper name or an id, not a generic).
        stripped = _normalize_alnum(re.sub(r"\([^)]*\)", " ", item))
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", re.sub(r"\([^)]*\)", " ", item))
                 if not all(part.lower() in _GENERIC_WORDS for part in w.split("-") if part)]
        words.sort(key=lambda w: (-sum(ch.isdigit() for ch in w), -len(w)))
        candidates = [(needle, None)]
        if stripped and stripped != needle and len(stripped) >= 3:
            candidates.append((stripped, None))
        candidates += [(_normalize_alnum(w), w) for w in words[:2] if len(_normalize_alnum(w)) >= 4]
        for cand, label in candidates:
            if any(paper_by_chunk.get(cid) == paper_id and cand in norm for cid, norm in norm_chunk_texts.items()):
                needle, matched_word = cand, label
                break
        for chunk_id, norm in norm_chunk_texts.items():
            if paper_by_chunk.get(chunk_id) != paper_id or needle not in norm:
                continue
            text = str((chunk_by_id.get(chunk_id) or {}).get("text") or "")
            _, offsets = _alnum_offsets(text)
            at = norm.find(needle)
            start, end = offsets[at], offsets[at + len(needle) - 1] + 1
            # one sentence either side, bounded
            prev_dot = text.rfind(". ", max(0, start - 300), start)
            lo = prev_dot + 2 if prev_dot >= 0 else max(0, start - 200)
            hi_dot = text.find(". ", end, end + 300)
            hi = hi_dot + 1 if hi_dot >= 0 else min(len(text), end + 200)
            key = (chunk_id, lo)
            if key not in seen_spans:
                seen_spans.add(key)
                hit = {"chunk_id": chunk_id, "item": item, "passage": " ".join(text[lo:hi].split()), "char_start": start, "char_end": end}
                if matched_word:
                    hit["matched_word"] = matched_word
                hits.append(hit)
            found = True
            break
        if not found and len(needle) >= 8:
            # The model may have mangled the value ("High Flux Is/Reactor" for
            # "High Flux Isotope Reactor"); show the closest passage so the
            # reviewer sees what the paper says without opening it.
            best = None
            for chunk_id, norm in norm_chunk_texts.items():
                if paper_by_chunk.get(chunk_id) != paper_id:
                    continue
                aligned = _compact_alignment(needle, norm)
                if aligned and aligned[2] >= 0.75 and (best is None or aligned[2] > best[1][2]):
                    best = (chunk_id, aligned)
            if best is not None:
                chunk_id, (a, b, coverage) = best
                text = str((chunk_by_id.get(chunk_id) or {}).get("text") or "")
                _, offsets = _alnum_offsets(text)
                start, end = offsets[a], offsets[b - 1] + 1
                prev_dot = text.rfind(". ", max(0, start - 300), start)
                lo = prev_dot + 2 if prev_dot >= 0 else max(0, start - 200)
                hi_dot = text.find(". ", end, end + 300)
                hi = hi_dot + 1 if hi_dot >= 0 else min(len(text), end + 200)
                hits.append({"chunk_id": chunk_id, "item": item, "passage": " ".join(text[lo:hi].split()), "char_start": start, "char_end": end,
                             "closest": " ".join(text[start:end].split()), "coverage": round(coverage, 2)})
                found = True
        if not found:
            hits.append({"chunk_id": None, "item": item, "passage": "", "char_start": None, "char_end": None})
        if len([h for h in hits if h["chunk_id"]]) >= max(limit, len(needles)):
            break
    return hits


def _edge_card(edge: dict[str, Any], *, label_by_node: dict[str, str]) -> dict[str, Any]:
    provenance_meta = {}
    for record in edge.get("provenance") or []:
        if isinstance(record, dict) and record.get("metadata"):
            provenance_meta = record["metadata"]
            break
    return {
        "edge_id": edge.get("edge_id"),
        "edge_type": edge.get("edge_type"),
        "status": edge.get("status"),
        "confidence": edge.get("confidence"),
        "source_label": label_by_node.get(str(edge.get("source_node_id")), str(edge.get("source_node_id"))),
        "target_label": label_by_node.get(str(edge.get("target_node_id")), str(edge.get("target_node_id"))),
        "evidence": edge.get("evidence") or [],
        "model": provenance_meta.get("model"),
        "verification_warnings": provenance_meta.get("warnings") or [],
    }


def _dashboard_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # Keep embedded JSON safe inside a <script> block.
    data_json = data_json.replace("</", "<\\/")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Agentic-RAG Gold Eval Panel</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f7f9fb;color:#17212b;line-height:1.45}"
        "header{position:sticky;top:0;background:#17212b;color:#f7f9fb;padding:12px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:5}"
        "header .grow{flex:1}"
        "main{max-width:1100px;margin:0 auto;padding:24px}"
        "body.grouped main{max-width:none;margin:0;padding:24px 24px 24px 340px}"
        "#papernav{display:none;position:fixed;left:0;top:58px;bottom:0;width:300px;overflow:auto;background:#f2f4f7;border-right:1px solid #d7dee7;padding:12px;font-size:.85rem}"
        "body.grouped #papernav{display:block}"
        "#papernav h3{margin:.2rem 0 .6rem;font-size:.9rem}"
        "#papernav .pn{display:block;padding:5px 8px;margin:2px 0;border-radius:6px;color:#17212b;text-decoration:none;cursor:pointer;line-height:1.25}"
        "#papernav .pn:hover{background:#e3e9f0}#papernav .pn.current{background:#205493;color:#fff}"
        "#papernav .pn .cnt{float:right;font-variant-numeric:tabular-nums;opacity:.85}"
        "#papernav .pn.done{opacity:.6}"
        ".paper-head{position:sticky;top:58px;z-index:4;background:#fbfcfd;border:1px solid #d7dee7;border-left:6px solid #205493;border-radius:8px;padding:10px 14px;margin:22px 0 10px}"
        ".paper-head h3{margin:0 0 4px;font-size:1.05rem}"
        ".paper-head .meta{margin:0}"
        ".paper-head a.pdf{font-weight:700;color:#205493;background:#e3ecf7;border-radius:6px;padding:2px 8px;text-decoration:none}"
        ".paper-head .meta{overflow-wrap:anywhere}"
        ".fact{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin:.35rem 0}"
        ".fact .ft{background:#17212b;color:#fff;border-radius:6px;padding:2px 9px;font-family:ui-monospace,Menlo,monospace;font-size:.85rem}"
        ".fact .value{font-size:1.15rem;font-weight:600}"
        ".question{margin:.4rem 0 .2rem;color:#17212b}"
        ".sect{margin:.7rem 0 .2rem;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:#52616f;font-weight:700}"
        "blockquote.unlocated{border-left-color:#b00020;background:#fbf0f1}"
        "blockquote.found{border-left-color:#1b7f3b;background:#f0f8f2}"
        "details.machine{margin:.5rem 0;font-size:.88rem}details.machine summary{color:#52616f;cursor:pointer}details.machine ul{margin:.3rem 0 0;padding-left:1.2rem}details.machine li{margin:.25rem 0}"
        "code.garbled{background:#fbf0f1;color:#7a1f28}"
        ".legend{background:#fff;border:1px solid #d7dee7;border-radius:8px;padding:10px 14px;margin:10px 0;font-size:.9rem}"
        ".legend dt{font-family:ui-monospace,Menlo,monospace;font-weight:700;margin-top:.35rem}.legend dd{margin:0 0 .2rem 0}"
        ".paper-head .nav{float:right}.paper-head .nav button{padding:4px 10px;font-size:.85rem;margin-left:6px}"
        ".paper-text{margin:0 0 10px;padding:0 14px}"
        "body.grouped details pre{max-height:none}"  # the page scrolls, not a box inside it
        "#focus-toggle{margin-left:8px}"
        "section h2{border-bottom:2px solid #d7dee7;padding-bottom:6px}"
        ".card{background:#fff;border:1px solid #d7dee7;border-radius:8px;padding:14px 16px;margin:14px 0;box-shadow:0 1px 2px rgba(23,33,43,.06)}"
        ".card.judged{border-left:6px solid #2e8540}"
        ".card .meta{color:#52616f;font-size:.85rem;margin-bottom:6px}"
        ".card .value{font-size:1.05rem;margin:.3rem 0}"
        "code{background:#eef2f6;border-radius:4px;padding:.05rem .3rem}"
        "blockquote{margin:.4rem 0;padding:.3rem .6rem;border-left:3px solid #205493;background:#f2f6fa}"
        "details{margin:.3rem 0}details pre{white-space:pre-wrap;background:#f2f4f7;padding:8px;border-radius:6px;max-height:260px;overflow:auto;font-size:.82rem}"
        ".verdicts label{margin-right:12px;white-space:nowrap}"
        ".fields{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}"
        ".fields input[type=text]{flex:1;min-width:220px;padding:6px 8px;border:1px solid #d7dee7;border-radius:6px}"
        "button{background:#205493;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer;font-size:.95rem}"
        "button.secondary{background:#52616f}"
        ".warn{color:#b0680c;font-size:.85rem}"
        ".bad{color:#b00020;font-size:.85rem;font-weight:600;margin:.2rem 0}"
        ".card.flagged{border-left:6px solid #b00020}"
        ".card.judge-flagged{border-left:6px solid #7a3fbf}"
        ".pill{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.78rem;background:#eef2f6;margin-left:6px}"
        ".pill.proposed{background:#fdf3e0}.pill.accepted{background:#e2f2e5}"
        ".card.quarantined{opacity:.75;border-left:6px solid #6c7a89}"
        ".card.recheck{border-left:6px solid #8a5a00;background:#fffaf0}"
        ".recheck-note{color:#6b4500;font-size:.88rem;font-weight:600;margin:.25rem 0}"
        ".src{font-size:.78rem;color:#1b7f3b;font-weight:600}"
        "mark{background:#ffe9a8;padding:0}"
        ".prior{margin:.3rem 0;padding:.35rem .6rem;border-left:4px solid #6c7a89;background:#f4f6f8;font-size:.85rem}"
        ".prior .st-correct{color:#1b7f3b;font-weight:600}.prior .st-neg{color:#b00020;font-weight:600}.prior .st-inc{color:#8a6d00;font-weight:600}"
        "#missing .card{border-style:dashed}"
        "select,textarea{padding:6px 8px;border:1px solid #d7dee7;border-radius:6px}"
        "</style></head><body>"
        "<header><strong>Agentic-RAG Gold Eval Panel</strong>"
        "<span class='grow' id='progress'></span>"
        "<button class='secondary' id='add-missing'>Add missing fact</button>"
        "<button id='export'>Export labels JSONL</button></header>"
        "<aside id='papernav'></aside>"
        "<main>"
        "<p>Label every card. Cards with a red citation check are listed first: the pipeline already located each quote across every chunk in the catalog, so a mis-cited, stitched, or unfound quote is called out for you. Read the chunk text before judging: the quote can look right while the chunk shows it was taken out of context. "
        "Drafts save automatically in this browser. Exported labels are ReviewLabel-compatible JSONL for "
        "<code>review labels ingest</code> and <code>corpus evidence-graph-apply-labels</code>. "
        "Nothing on this page leaves your machine.</p>"
        "<details class='legend'><summary><strong>What the labels mean</strong> (one fact = one value the extraction model pulled from one paper)</summary><dl>"
        "<dt>correct</dt><dd>The value is right, it belongs under this fact type, and the paper's passage states it about this paper's own work.</dd>"
        "<dt>wrong_field</dt><dd>The value is real and stated, but it belongs under a different fact type (say, a facility filed as an instrument). Put the right type in the notes.</dd>"
        "<dt>too_vague</dt><dd>The passage is right but the value is incomplete or too loose to use: a bare threshold (\"below 1.3 K\"), a name without a value, a range the paper does not give.</dd>"
        "<dt>not_supported</dt><dd>The paper's text does not state this value, or it is about another material, a comparison, or someone else's prior work.</dd>"
        "<dt>ocr_or_chunking_problem</dt><dd>The paper text itself is broken here (garbled characters, a caption cut in half, two papers mixed), so the fact cannot be judged from it. Not for a garbled <em>model copy</em> of a good passage: those are already replaced by the paper's text.</dd>"
        "</dl><p class='meta'>Each card asks one question about one value and shows the paper's own text under it. Anything under \"Automatic notes\" is machine context: the extraction model's garbled copy of a quote, or a second automatic read that agrees or disagrees. Neither is a label; you are.</p></details>"
        "<p class='meta' id='resume-summary'></p>"
        "<p class='meta' id='prior-summary'></p>"
        "<section id='facts'><h2>Extracted facts</h2></section>"
        "<section id='missing'><h2>Missing facts (gold recall records)</h2><p class='meta'>Facts the papers state but the loop never extracted. These are what make recall measurable.</p></section>"
        "<section id='edges'><h2>Proposed relation edges</h2></section>"
        "</main>"
        f"<script id='panel-data' type='application/json'>{data_json}</script>"
        "<script>\n" + _PANEL_JS + "\n</script>"
        "</body></html>"
    )


_PANEL_JS = r"""
const DATA = JSON.parse(document.getElementById('panel-data').textContent);
const STORE_KEY = 'tas-gold-eval:' + (DATA.storage_key || DATA.catalog);
// Start from the reviewer's earlier export (if the panel was built with one),
// then lay this browser's saved drafts over it: the newer work wins.
const SEED = DATA.seed || {facts: {}, edges: {}, missing: []};
let state = {facts: Object.assign({}, SEED.facts), edges: Object.assign({}, SEED.edges), missing: (SEED.missing || []).slice()};
try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
  if (saved) {
    Object.assign(state.facts, saved.facts || {});
    Object.assign(state.edges, saved.edges || {});
    if ((saved.missing || []).length) state.missing = saved.missing;
  }
} catch (e) {}
// Facts queued for a second look start unjudged, whatever was saved before.
for (const id of (DATA.recheck_ids || [])) { const r = state.facts[id]; if (r && !r.rechecked) delete state.facts[id]; }
const RECHECK = new Set(DATA.recheck_ids || []);

function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) {} updateProgress(); postLabels(); }
const TITLE_BY_PAPER = Object.fromEntries((DATA.paper_groups || []).map(g => [g.paper_id, g.title]));
function paperTitle(p) { const t = TITLE_BY_PAPER[p]; return t ? (t.length > 70 ? t.slice(0, 68) + '…' : t) : p; }
function esc(s) { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }

function verdictRow(kind, id, options) {
  const chosen = (state[kind][id] || {}).verdict || '';
  return '<div class="verdicts">' + options.map(v =>
    `<label><input type="radio" name="${kind}:${id}" value="${v}" ${v === chosen ? 'checked' : ''}> ${v}</label>`
  ).join('') + '</div>';
}

function fieldRow(kind, id, withExpected) {
  const rec = state[kind][id] || {};
  const hint = withExpected ? 'Why (and the right value or field, when not correct)' : 'Why';
  return `<div class="fields"><input type="text" data-kind="${kind}" data-id="${id}" data-field="notes" placeholder="${hint}" value="${esc(rec.notes || '')}"></div>`;
}

function priorBlock(labels) {
  if (!labels || !labels.length) return '';
  return labels.map(l => {
    const st = String(l.status || l.verdict || '');
    const cls = st === 'correct' ? 'st-correct' : (st === 'too_vague' || l.verdict === 'inconclusive') ? 'st-inc' : 'st-neg';
    return `<div class="prior">Prior flag · <strong>${esc(l.reviewer)}</strong>: <span class="${cls}">${esc(st)}</span>${l.notes ? ' — ' + esc(l.notes) : ''}</div>`;
  }).join('');
}

function renderPriorSummary() {
  const el = document.getElementById('prior-summary'); if (!el) return;
  const s = DATA.prior_label_summary || {}; const names = Object.keys(s);
  if (!names.length) { el.textContent = ''; return; }
  el.innerHTML = 'Prior flags shown on cards (context only, not your label): ' + names.map(n => `<strong>${esc(n)}</strong> — ` + Object.entries(s[n]).map(([k, v]) => `${esc(k)} ${v}`).join(', ')).join('; ');
}

const GROUPED = !!DATA.group_by_paper;
let focusPaper = null;      // grouped mode: the one paper shown, or null for all
try { if (GROUPED) focusPaper = localStorage.getItem(STORE_KEY + ':focus') || null; } catch (e) {}
if (GROUPED && focusPaper && !(DATA.paper_groups || []).some(g => g.paper_id === focusPaper)) focusPaper = null;

function paperProgress(g) {
  const done = DATA.facts.filter(f => f.paper_id === g.paper_id && countsTowardProgress(f) && (state.facts[f.fact_id] || {}).verdict).length;
  return {done, total: g.total};
}

function renderPaperNav() {
  const nav = document.getElementById('papernav'); if (!nav || !GROUPED) return;
  const groups = DATA.paper_groups || [];
  const allDone = groups.reduce((a, g) => a + paperProgress(g).done, 0), allTotal = groups.reduce((a, g) => a + g.total, 0);
  nav.innerHTML = `<h3>Papers (${groups.length}) · ${allDone}/${allTotal} labeled</h3>` +
    `<a class="pn ${focusPaper === null ? 'current' : ''}" data-paper="*">All papers on one page</a>` +
    groups.map(g => {
      const p = paperProgress(g);
      const flags = (g.second_look ? ` · ${g.second_look} second look` : '') + (g.flagged ? ` · ${g.flagged} flagged` : '');
      return `<a class="pn ${focusPaper === g.paper_id ? 'current' : ''} ${p.done >= p.total ? 'done' : ''}" data-paper="${esc(g.paper_id)}" title="${esc(g.title)}${esc(flags)}">` +
        `<span class="cnt">${p.done}/${p.total}</span>${esc(g.title.length > 60 ? g.title.slice(0, 58) + '…' : g.title)}</a>`;
    }).join('');
}

function setFocus(paperId) {
  focusPaper = paperId === '*' ? null : paperId;
  try { if (focusPaper) localStorage.setItem(STORE_KEY + ':focus', focusPaper); else localStorage.removeItem(STORE_KEY + ':focus'); } catch (e) {}
  renderFacts(); renderPaperNav(); window.scrollTo(0, 0);
}

function paperHeader(g, idx) {
  const p = paperProgress(g);
  const groups = DATA.paper_groups || [];
  const prev = idx > 0 ? groups[idx - 1].paper_id : null, next = idx + 1 < groups.length ? groups[idx + 1].paper_id : null;
  const chunks = (DATA.paper_chunks || {})[g.paper_id] || [];
  const nav = focusPaper ? `<span class="nav">` +
      (prev ? `<button class="secondary" data-paper="${esc(prev)}">← previous paper</button>` : '') +
      (next ? `<button data-paper="${esc(next)}">next paper →</button>` : '') + `</span>` : '';
  const links = (g.pdf_link ? ` · <a class="pdf" href="${esc(g.pdf_link)}" target="_blank" rel="noopener">open PDF</a>` : '') +
    (g.pdf_name ? ` · <span class="meta" title="file name in the corpus folder">${esc(g.pdf_name)}</span>` : '') +
    (g.doi ? ` · <a href="https://doi.org/${esc(g.doi)}" target="_blank" rel="noopener">doi:${esc(g.doi)}</a>` : '');
  return `<div class="paper-head" id="paper-${esc(g.paper_id)}">${nav}<h3>${esc(g.title)}</h3>` +
    `<p class="meta"><code>${esc(g.paper_id)}</code>${links}</p>` +
    `<p class="meta"><span class="pp" data-paper="${esc(g.paper_id)}">${p.done}/${p.total} labeled</span>` +
    (g.second_look ? ` · <strong>${g.second_look} second look</strong>` : '') + (g.flagged ? ` · ${g.flagged} flagged` : '') +
    (g.sidelined ? ` · ${g.sidelined} sidelined` : '') + `</p>` +
    `</div>` +
    (chunks.length ? `<details class="paper-text"><summary>paper text (${chunks.length} chunks) — for facts with no evidence, or to check context</summary>` +
      chunks.map(c => `<p class="meta"><code>${esc(c.chunk_id)}</code></p><pre>${esc(c.text)}</pre>`).join('') + `</details>` : '');
}

function renderFacts() {
  const sec = document.getElementById('facts');
  sec.innerHTML = '<h2>Extracted facts</h2>';
  let lastPaper = null;
  const groups = DATA.paper_groups || [];
  for (const f of DATA.facts) {
    if (GROUPED) {
      if (focusPaper && f.paper_id !== focusPaper) continue;
      if (f.paper_id !== lastPaper) {
        const idx = groups.findIndex(g => g.paper_id === f.paper_id);
        if (idx >= 0) sec.insertAdjacentHTML('beforeend', paperHeader(groups[idx], idx));
        lastPaper = f.paper_id;
      }
    }
    const card = document.createElement('div');
    const sidelined = f.integrity_status === 'content_mismatch' || !!f.duplicate_of || !!f.malformed_value;
    card.className = 'card' + ((state.facts[f.fact_id] || {}).verdict ? ' judged' : '') + (f.citation_flagged ? ' flagged' : '') + (f.judge_flagged ? ' judge-flagged' : '') + (sidelined ? ' quarantined' : '') + (f.recheck ? ' recheck' : '');
    card.id = 'fact-' + f.fact_id;
    const sidelineNote =
      f.integrity_status === 'content_mismatch' ? `<div class="bad">Not counted: this document's text belongs to a different paper (bad PDF download). Label only if you want to record the mismatch.</div>` :
      f.duplicate_of ? `<div class="meta">Not counted: duplicate of <code>${esc(f.duplicate_of)}</code> (same type and value); its evidence was merged there.</div>` :
      f.malformed_value ? `<div class="bad">Not counted: the extracted value was model output, not paper content (${esc(f.malformed_value)}), and was discarded. Nothing to label.` + (f.discarded_value ? ` It read: <code class="garbled">${esc(f.discarded_value)}${f.discarded_value.length >= 300 ? '…' : ''}</code>` : '') + `</div>` : '';

    // --- the paper's own text that the fact rests on ---
    let passages = '', machine = [], problems = [];
    for (const e of f.evidence) {
      const located = e.quote_match === 'verbatim' || e.quote_match === 'snapped' || e.quote_match === 'split_fragment';
      const cc = e.citation_check || {};
      if (located) {
        passages += `<blockquote>${esc(e.quote || '(no quote)')} <span class="meta">— <code>${esc(e.chunk_id)}</code></span></blockquote>`;
        if (e.quote_as_extracted && e.quote_match !== 'verbatim')
          machine.push(`<li><strong>Extraction model's copy of this quote was garbled</strong> and has been replaced above by the paper's own text. It wrote: <code class="garbled">${esc(e.quote_as_extracted)}</code> — judge the paper's text, not this copy.</li>`);
      } else if (!(e.quote || '').trim() && !e.chunk_text) {
        machine.push(`<li>The extraction also emitted an empty citation record (no quote, chunk id <code>${esc(e.chunk_id || '')}</code>); it is ignored.</li>`);
      } else {
        passages += `<blockquote class="unlocated">${esc(e.quote || '(no quote)')} <span class="meta">— cited <code>${esc(e.chunk_id)}</code></span></blockquote>`;
        problems.push(`<div class="bad">This quote could not be found in the paper's text${cc.status === 'near_verbatim_in_chunk' ? ' as one passage' : ''}${(cc.found_in || []).length ? ' (similar text in ' + cc.found_in.map(c => `<code>${esc(c)}</code>`).join(', ') + ')' : ''}. If the paper does not say this, the fact is <code>not_supported</code>; if the paper text itself is garbled here, use <code>ocr_or_chunking_problem</code>.</div>`);
      }
      for (const w of (e.warnings || [])) {
        if (/^Quote replaced|re-homed|split into|Extraction schema text/.test(w)) continue;  // already explained above
        if (/not found|fabricated|requires review|cannot be machine-verified/.test(w)) problems.push(`<div class="bad">${esc(w)}</div>`);
        else machine.push(`<li>${esc(w)}</li>`);
      }
      if (e.chunk_text) {
        const a = e.char_start, b = e.char_end;
        const body = (Number.isInteger(a) && Number.isInteger(b) && a < b && b <= e.chunk_text.length)
          ? esc(e.chunk_text.slice(0, a)) + '<mark>' + esc(e.chunk_text.slice(a, b)) + '</mark>' + esc(e.chunk_text.slice(b))
          : esc(e.chunk_text);
        passages += `<details><summary>surrounding text in the paper (${esc(e.chunk_id)})</summary><pre>${body}</pre></details>`;
      }
    }
    const hits = (f.value_hits || []).filter(h => h.chunk_id), misses = (f.value_hits || []).filter(h => !h.chunk_id);
    const downgraded = f.warnings.some(w => /downgraded to ambiguous/.test(w));
    if (hits.length) {
      for (const h of hits) {
        passages += `<blockquote class="found">${esc(h.passage)} <span class="meta">— <code>${esc(h.chunk_id)}</code>${h.item && h.item !== f.value_text ? ' · for <code>' + esc(h.item) + '</code>' : ''}${h.matched_word ? ' (matched on <code>' + esc(h.matched_word) + '</code>)' : ''}${h.closest ? ' · <strong>closest match</strong>: the paper says <code>' + esc(h.closest) + '</code>, the extracted value differs' : ''}</span></blockquote>`;
      }
      if (misses.length) passages += `<div class="warn">Not found in the paper's text: ${misses.map(m => '<code>' + esc(m.item) + '</code>').join(', ')}.</div>`;
      if (!f.evidence.length) passages += `<div class="meta"><strong>Not a citation.</strong> ` + (downgraded
        ? `The extractor found this fact but cited no passage, so it was downgraded to <code>ambiguous</code>; the sentences above are where the value's words occur in this paper.`
        : `The extraction reported <code>${esc(f.answer_status)}</code> and cited nothing, so these are simply the sentences of this paper that contain the value's words.`) + ` A generic value will pull in unrelated sentences.</div>`;
    }
    if (!f.evidence.length && !hits.length) {
      passages = f.answer_status === 'supported' || downgraded
        ? `<div class="bad">The extractor found this fact but cited no passage${downgraded ? ' (downgraded to ambiguous for that reason)' : ''}, and its value's words were not found in the paper's text. Check the paper text at the top of this section; if the value is not there, it is <code>not_supported</code>.</div>`
        : f.answer_status === 'not_found'
          ? `<div class="meta"><strong>Reported absent.</strong> The extractor says this paper states no <code>${esc(f.fact_type)}</code>. ` +
            (f.absence_cues === null || f.absence_cues === undefined
              ? `No automatic search is defined for this type; check the paper text at the top of this section.`
              : f.absence_cues.length
                ? `A search of the paper for <code>${esc(f.fact_type)}</code>-like terms found these sentences; if one of them states the fact, the extractor missed it (<code>not_supported</code>, and add a missing record):`
                : `A search of the paper for <code>${esc(f.fact_type)}</code>-like terms found nothing, which supports <code>correct</code> (confirmed absent).`) + `</div>` +
            (f.absence_cues || []).map(c => `<blockquote class="found">${esc(c.passage)} <span class="meta">— <code>${esc(c.chunk_id)}</code> · term <code>${esc(c.term)}</code></span></blockquote>`).join('')
          : `<div class="meta">The extraction reported <code>${esc(f.answer_status)}</code> for this fact type and cited nothing. Judge whether the paper truly lacks it (see the paper text at the top of this paper's section).</div>`;
    }
    // --- automatic checks on the fact itself ---
    for (const w of f.warnings) {
      if (/^Second-pass judge/.test(w)) continue;  // rendered from f.judge below
      machine.push(`<li>${esc(w)}</li>`);
    }
    if (f.judge && f.judge.verdict && f.judge.verdict !== 'no_located_evidence') {
      const v = f.judge.verdict;
      const what = v === 'supported' ? 'found no problem' : `would mark it <code>${esc(v)}</code>` + (f.judge.suggested_fact_type && f.judge.suggested_fact_type !== 'none' ? ` (suggests the field <code>${esc(f.judge.suggested_fact_type)}</code>)` : '');
      machine.push(`<li><strong>Second automatic read</strong> (${esc(f.judge.model || 'model')} re-read only the passage above, after extraction) ${what}${f.judge.note ? ': ' + esc(f.judge.note).replace(/\.$/, '') : ''}. A machine opinion, right about two times in three; not a label.</li>`);
    }
    const extractedBy = f.model ? `extracted by ${esc(f.model)}` : 'extracted';
    const question = f.evidence.length
      ? `Does the passage below, from this paper, state <strong>${esc(f.value_text)}</strong> as a <code>${esc(f.fact_type)}</code> fact about this paper's own work?`
      : f.answer_status === 'not_found' && !f.value_text
        ? `Is it true that this paper states no <code>${esc(f.fact_type)}</code>? (<code>correct</code> = confirmed absent; <code>not_supported</code> = the paper does state one.)`
        : `Does this paper state a <code>${esc(f.fact_type)}</code> fact${f.value_text ? ' with the value <strong>' + esc(f.value_text) + '</strong>' : ''}?`;
    const recheckBox = f.recheck ? (() => { const el = f.earlier_label || {}; return `<div class="recheck-note">Second look. You labeled this <code>${esc(el.verdict)}</code> before; since then the garbled quote was replaced by the paper's own text (below). Please label it again on that text.</div>` + (el.notes ? `<div class="prior">Your earlier note: ${esc(el.notes)}</div>` : ''); })() : '';

    card.innerHTML =
      `<div class="meta">${esc(f.paper_title)} · <code>${esc(f.fact_id)}</code> · ${extractedBy}</div>` +
      `<div class="fact"><span class="ft">${esc(f.fact_type)}</span><span class="value">${esc(f.value_text) || (f.answer_status === 'not_found' ? '<em>reported absent from this paper</em>' : '<em>(empty value)</em>')}</span>${f.value_subject ? '<span class="pill">about ' + esc(f.value_subject) + '</span>' : ''}${(f.canonical || []).map(c => '<span class="pill" title="curated alias registry">= ' + esc(c.label) + (c.kind ? ' · ' + esc(c.kind) : '') + (c.facility ? ' · ' + esc(c.facility.toUpperCase()) : '') + '</span>').join('')}</div>` +
      sidelineNote + recheckBox +
      `<div class="question">${question}</div>` +
      `<div class="sect">Passage from the paper</div>` + passages + problems.join('') +
      (machine.length ? `<details class="machine"><summary>Automatic notes (${machine.length}) — context only, never a label</summary><ul>${machine.join('')}</ul></details>` : '') +
      priorBlock(f.prior_labels) +
      `<div class="sect">Your label</div>` + verdictRow('facts', f.fact_id, DATA.fact_verdicts) + fieldRow('facts', f.fact_id, true);
    sec.appendChild(card);
  }
}

function renderEdges() {
  const sec = document.getElementById('edges');
  if (!DATA.edges.length) { sec.insertAdjacentHTML('beforeend', '<p class="meta">No proposed relation edges in this catalog. Run corpus evidence-graph-propose first.</p>'); return; }
  for (const e of DATA.edges) {
    const card = document.createElement('div');
    card.className = 'card' + ((state.edges[e.edge_id] || {}).verdict ? ' judged' : '') + (e.sidelined ? ' quarantined' : '');
    card.id = 'edge-' + e.edge_id;
    card.innerHTML =
      `<div class="meta"><code>${esc(e.edge_id)}</code> · conf ${esc(e.confidence)}${e.model ? ' · ' + esc(e.model) : ''}<span class="pill ${esc(e.status)}">${esc(e.status)}</span></div>` +
      (e.sidelined ? `<div class="bad">Not counted: at least one of the facts this edge joins belongs to a quarantined (wrong download) or duplicate paper. Nothing to review.</div>` : '') +
      `<div class="value">${esc(e.source_label)} —[<strong>${esc(e.edge_type)}</strong>]→ ${esc(e.target_label)}</div>` +
      e.evidence.map(x => `<blockquote>${esc(x)}</blockquote>`).join('') +
      (e.verification_warnings.length ? `<div class="warn">${e.verification_warnings.map(esc).join('<br>')}</div>` : '') +
      verdictRow('edges', e.edge_id, DATA.edge_verdicts) + fieldRow('edges', e.edge_id, false);
    sec.appendChild(card);
  }
}

function renderMissing() {
  const sec = document.getElementById('missing');
  sec.querySelectorAll('.card').forEach(el => el.remove());
  state.missing.forEach((m, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML =
      `<div class="fields">` +
      `<select data-missing="${i}" data-field="paper_id">` + DATA.paper_ids.map(p => `<option value="${esc(p)}" ${p === m.paper_id ? 'selected' : ''}>${esc(paperTitle(p))}</option>`).join('') + `</select>` +
      `<select data-missing="${i}" data-field="fact_type">` + DATA.fact_types.map(t => `<option ${t === m.fact_type ? 'selected' : ''}>${esc(t)}</option>`).join('') + `</select>` +
      `</div><div class="fields">` +
      `<input type="text" data-missing="${i}" data-field="expected_value" placeholder="What the paper states" value="${esc(m.expected_value || '')}">` +
      `<input type="text" data-missing="${i}" data-field="chunk_hint" placeholder="Where (chunk id / section hint)" value="${esc(m.chunk_hint || '')}">` +
      `<input type="text" data-missing="${i}" data-field="notes" placeholder="Notes" value="${esc(m.notes || '')}">` +
      `<button class="secondary" data-remove="${i}">Remove</button></div>`;
    sec.appendChild(card);
  });
}

function countsTowardProgress(f) { return !(f.integrity_status === 'content_mismatch' || f.duplicate_of || f.malformed_value); }
function updateProgress() {
  const counted = DATA.facts.filter(countsTowardProgress);
  const factDone = counted.filter(f => (state.facts[f.fact_id] || {}).verdict).length;
  const sidelined = DATA.facts.length - counted.length;
  const liveEdges = DATA.edges.filter(e => !e.sidelined);
  const edgeDone = liveEdges.filter(e => (state.edges[e.edge_id] || {}).verdict).length;
  document.getElementById('progress').textContent =
    `facts ${factDone}/${counted.length}${sidelined ? ' (+' + sidelined + ' sidelined)' : ''} · edges ${edgeDone}/${liveEdges.length}${DATA.edges.length - liveEdges.length ? ' (+' + (DATA.edges.length - liveEdges.length) + ' sidelined)' : ''} · missing ${state.missing.length}` + (postState ? ' · ' + postState : '');
  if (GROUPED) {
    for (const g of (DATA.paper_groups || [])) {
      const p = paperProgress(g);
      document.querySelectorAll(`.pp[data-paper="${CSS.escape(g.paper_id)}"]`).forEach(el => el.textContent = `${p.done}/${p.total} labeled`);
    }
    renderPaperNav();
  }
}

document.addEventListener('change', ev => {
  const t = ev.target;
  if (t.matches('input[type=radio]')) {
    const [kind, id] = t.name.split(/:(.+)/);
    state[kind][id] = state[kind][id] || {};
    state[kind][id].verdict = t.value;
    if (kind === 'facts' && FACT_BY_ID[id]) { const f = FACT_BY_ID[id]; state[kind][id].fact_type = f.fact_type; state[kind][id].paper_id = f.paper_id; state[kind][id].value_text = f.value_text; }
    if (kind === 'facts' && RECHECK.has(id)) state[kind][id].rechecked = true;
    document.getElementById((kind === 'facts' ? 'fact-' : 'edge-') + id).classList.add('judged');
    save();
  } else if (t.dataset.kind) {
    const rec = state[t.dataset.kind][t.dataset.id] = state[t.dataset.kind][t.dataset.id] || {};
    rec[t.dataset.field] = t.value; save();
  } else if (t.dataset.missing !== undefined) {
    state.missing[Number(t.dataset.missing)][t.dataset.field] = t.value; save();
  }
});
document.addEventListener('click', ev => {
  if (ev.target.dataset.remove !== undefined) { state.missing.splice(Number(ev.target.dataset.remove), 1); save(); renderMissing(); }
  const pn = ev.target.closest('[data-paper]');
  if (pn && GROUPED && pn.dataset.paper) { ev.preventDefault(); setFocus(pn.dataset.paper); }
});
document.getElementById('add-missing').addEventListener('click', () => {
  state.missing.push({paper_id: (GROUPED && focusPaper) || DATA.paper_ids[0] || '', fact_type: DATA.fact_types[0] || '', expected_value: '', chunk_hint: '', notes: ''});
  save(); renderMissing();
});

const NEGATIVE = new Set(['wrong_field', 'too_vague', 'not_supported', 'ocr_or_chunking_problem']);
const FACT_BY_ID = Object.fromEntries((DATA.facts || []).map(f => [f.fact_id, f]));
function exportRows() {
  const rows = [];
  // Every labeled fact the browser holds (this page's cards, labels carried from
  // other pages with the same storage key, pre-filled labels): a page that shows a
  // subset must not export a subset, or an earlier draft is silently lost.
  for (const [fact_id, rec] of Object.entries(state.facts)) {
    if (!rec || !rec.verdict) continue;
    const f = FACT_BY_ID[fact_id] || {fact_id, fact_type: rec.fact_type || null, paper_id: rec.paper_id || null, value_text: rec.value_text || null, model: null};
    rows.push({
      object_type: 'paper_fact', object_id: fact_id, object_secondary_id: f.fact_type,
      source_workflow: 'fact-eval-dashboard',
      verdict: rec.verdict === 'correct' ? 'positive' : (NEGATIVE.has(rec.verdict) ? 'negative' : 'inconclusive'),
      type_specific_status: rec.verdict, type_specific_allowed: DATA.fact_verdicts,
      review_notes: rec.notes || '',
      metadata: {paper_id: f.paper_id, fact_type: f.fact_type, extracted_value: f.value_text,
                 expected_value: rec.expected_value || null, model: f.model, gold_eval: true, on_page: !!FACT_BY_ID[fact_id]},
    });
  }
  for (const e of DATA.edges) {
    const rec = state.edges[e.edge_id];
    if (!rec || !rec.verdict) continue;
    rows.push({
      object_type: 'evidence_graph_edge', object_id: e.edge_id, object_secondary_id: e.edge_type,
      source_workflow: 'fact-eval-dashboard',
      verdict: rec.verdict === 'accepted' ? 'positive' : (rec.verdict === 'rejected' ? 'negative' : 'inconclusive'),
      type_specific_status: rec.verdict, type_specific_allowed: DATA.edge_verdicts,
      review_notes: rec.notes || '',
      metadata: {edge_type: e.edge_type, confidence: e.confidence, model: e.model, gold_eval: true},
    });
  }
  state.missing.forEach((m, i) => {
    if (!m.expected_value) return;
    rows.push({
      object_type: 'paper_fact', object_id: `missing-${i}-${m.paper_id}-${m.fact_type}`, object_secondary_id: m.fact_type,
      source_workflow: 'fact-eval-dashboard', verdict: 'negative',
      type_specific_status: 'missing', type_specific_allowed: DATA.fact_verdicts.concat(['missing']),
      review_notes: m.notes || '',
      metadata: {paper_id: m.paper_id, fact_type: m.fact_type, expected_value: m.expected_value,
                 chunk_hint: m.chunk_hint || null, gold_eval: true, missing_fact: true},
    });
  });
  return rows;
}
// Served over http (scripts/serve_fact_panel.py), every change is also posted to
// the server, which keeps the newest full label set on disk: no export needed
// for the labels to reach the pipeline. Export stays as a backup.
let postTimer = null, postState = '';
function postLabels() {
  if (!/^https?:$/.test(location.protocol)) return;
  clearTimeout(postTimer);
  postTimer = setTimeout(async () => {
    const rows = exportRows();
    try {
      const r = await fetch('/labels', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rows, page: DATA.page_name || 'fact_eval_dashboard'})});
      postState = r.ok ? `saved to server ${new Date().toLocaleTimeString()}` : 'server save failed';
    } catch (e) { postState = 'server save failed'; }
    updateProgress();
  }, 800);
}
document.getElementById('export').addEventListener('click', () => {
  const rows = exportRows();
  if (!rows.length) { alert('Nothing labeled yet.'); return; }
  const blob = new Blob([rows.map(r => JSON.stringify(r)).join('\n') + '\n'], {type: 'application/jsonl'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (DATA.page_name || 'fact_eval_dashboard') + '_labels.jsonl';
  a.click();
  URL.revokeObjectURL(a.href);
});

(function renderResumeSummary() {
  const el = document.getElementById('resume-summary'); if (!el) return;
  const n = Object.keys(SEED.facts || {}).length, r = (DATA.recheck_ids || []).length;
  if (!n && !r) return;
  el.innerHTML = `<strong>Resumed:</strong> ${n} of your earlier labels are filled in and listed last. ` +
    (r ? `<strong>${r}</strong> facts you had marked as citation problems are listed first for a second look, because their quotes are now the exact source text. ` : '') +
    'Unlabeled facts follow. Export gives one file with everything.';
})();
renderPriorSummary();
if (DATA.only_second_looks) {
  const intro0 = document.querySelector('main > p');
  if (intro0) intro0.innerHTML = '<strong>Second looks only.</strong> Each card here carried a citation-defect label from the first pass (the quote was garbled or mis-cited); the citation has since been replaced by the paper\'s own text. Judge the fact again on that text: is the value right for this field and this paper? Labels made here appear in the full panel too.';
}
if (GROUPED) {
  document.body.classList.add('grouped');
  const intro = document.querySelector('main > p');
  if (intro && !DATA.only_second_looks) intro.innerHTML = 'One paper at a time. Pick a paper on the left (papers with the most unlabeled facts come first); inside a paper, second looks and flagged cards come first, then unlabeled, then your earlier labels. ' +
    'The paper text at the top of each group is there for facts with no evidence and for checking context. Drafts save automatically in this browser; <strong>Export</strong> gives one file with everything, whichever papers you did. Nothing on this page leaves your machine.';
  // open on the first paper with pending work unless the reviewer chose one before
  if (!focusPaper) { const g = (DATA.paper_groups || []).find(g => paperProgress(g).done < g.total); if (g) focusPaper = g.paper_id; }
  renderPaperNav();
}
renderFacts(); renderEdges(); renderMissing(); updateProgress();
"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
