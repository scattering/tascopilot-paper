from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tas_copilot_contracts import EvidenceGraphEdge, EvidenceGraphNode, ProvenanceRecord
from tas_copilot_core.llm_router import ollama_chat, openai_compatible_chat
from tas_copilot_core.reporting import write_report_bundle


PROMPT_VERSION = "evidence_graph_relation_judgment_v0"
CREATED_BY = "evidence_graph_proposer"

# Curated edge taxonomy. The proposer may not invent edge types: this is the
# boundary between bounded candidate-edge proposal and an unconstrained
# LLM-generated knowledge graph, which project policy forbids.
DETERMINISTIC_EDGE_TYPES = {
    "REPORTS_FACT",
    "FACT_SUPPORTED_BY",
    "ANALOG_MODEL_PRIOR_FOR",
}
RELATION_JUDGMENTS = {
    "consistent_with",
    "conflicts_with",
    "same_quantity_different_conditions",
    "unrelated",
    "ambiguous",
}
RELATION_EDGE_TYPES = {
    "consistent_with": "CONSISTENT_WITH",
    "conflicts_with": "CONFLICTS_WITH",
    "same_quantity_different_conditions": "SAME_QUANTITY_AS",
}

# Fact types that may seed ANALOG_MODEL_PRIOR_FOR edges toward a named case
# target (mirrors recommended_graph_edges in the SrMn2Sb2 agentic-RAG report).
ANALOG_PRIOR_FACT_TYPES = {
    "magnetic_structure",
    "propagation_vector",
    "spin_direction",
    "ordered_moment",
    "exchange_interaction",
    "magnetic_anisotropy",
    "crystal_structure",
}

# Fact types worth cross-paper comparison in stage B. Pairing is deterministic:
# same fact type, different papers, both answer_status=supported. Only
# quantities a material HAS are compared; measurement conditions (the
# temperature or incident energy a scan was taken at) are not properties of
# anything and produced "same quantity, different conditions" edges between
# unrelated measurements (2026-10-17: a 5 K scan in one paper paired with a
# Mössbauer series in another).
_MATERIAL_IDENTITY_FACT_TYPES = {"material", "composition", "sample"}
PAIRABLE_FACT_TYPES = set(ANALOG_PRIOR_FACT_TYPES)
# A paper whose material facts name more compounds than this is a review or a
# comparison study; its fact-derived keys would make it share a subject with
# every paper that mentions any of them. Such a paper pairs on its title only.
_MAX_MATERIAL_KEYS_FROM_FACTS = 4


def run_evidence_graph_proposer(arguments: dict[str, Any]) -> dict[str, Any]:
    """Propose reviewable evidence-graph edges from validated paper facts.

    This is the bounded counterpart of GraphRAG-style graph building: the
    controller owns node/edge types, pairing, merge policy, and stopping
    conditions. Stage A deterministically projects already-validated PaperFact
    records into typed nodes and status=proposed edges. Stage B optionally asks
    a local model to judge cross-paper fact pairs against a fixed relation
    enum, with cited chunk quotes. Nothing is accepted here; every proposed
    edge flows to human review through the unified review-label store.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = str(arguments.get("provider") or "none")
    model = str(arguments.get("model") or "")
    ollama_url = str(arguments.get("ollama_url") or "http://127.0.0.1:11434")
    vllm_url = str(arguments.get("vllm_url") or "")
    api_key = arguments.get("api_key")
    case_target = str(arguments.get("case_target") or "")
    case_target_label = str(arguments.get("case_target_label") or case_target)
    max_pairs = int(arguments.get("max_pairs") or 20)
    max_chunk_chars = int(arguments.get("max_chunk_chars") or 4000)

    if not catalog_dir.exists():
        return _missing_report(output_dir, catalog_dir, f"Corpus catalog does not exist: {catalog_dir}")
    facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    if not facts:
        return _missing_report(output_dir, catalog_dir, "paper_facts.jsonl is absent or empty; run bounded fact extraction first.")
    if provider == "ollama" and not model:
        return _missing_report(output_dir, catalog_dir, "provider=ollama requires --model.")
    if provider == "vllm" and (not model or not vllm_url):
        return _missing_report(output_dir, catalog_dir, "provider=vllm requires --model and --vllm-url.")

    chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks if chunk.get("chunk_id")}
    title_by_paper = {str(paper.get("paper_id")): paper.get("title") for paper in papers if paper.get("paper_id")}
    supported = [fact for fact in facts if fact.get("answer_status") == "supported" and fact.get("fact_id")]

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    # Stage A: deterministic projection of validated facts into typed
    # nodes/edges. No model involved; provenance and warnings are preserved.
    stage_a = _project_fact_edges(
        supported,
        nodes=nodes,
        edges=edges,
        catalog_dir=catalog_dir,
        title_by_paper=title_by_paper,
        case_target=case_target,
        case_target_label=case_target_label,
    )

    # Stage B: bounded cross-paper relation judgments over deterministic pairs.
    #
    # Each judgment is a model call, so the loop checkpoints after every pair:
    # a run killed at the scheduler wall must not throw away the inference it
    # already paid for. The checkpoint doubles as the resume set - re-running
    # with the same output_dir replays completed pairs from disk for free.
    paper_groups_arg = arguments.get("paper_groups") or {}
    paper_groups = {str(k): {str(v) for v in (vals or [])} for k, vals in dict(paper_groups_arg).items()}
    community_grouping: dict[str, Any] | None = None
    community_csv = arguments.get("community_assignments_csv")
    if community_csv and not paper_groups:
        # Structural grouping from the crystal-communities released assignments.
        # Exact formula matching alone grouped 5 of 46 papers on the BT-7 set;
        # community membership is the broader subject signal.
        from tas_copilot_core.corpus.crystal_community_groups import build_paper_groups, load_community_index

        index = load_community_index(community_csv, layout_csv=arguments.get("community_layout_csv"))
        paper_groups, community_grouping = build_paper_groups(
            supported,
            titles=title_by_paper,
            index=index,
            mode=str(arguments.get("community_group_mode") or "community"),
            max_centroid_distance=arguments.get("max_centroid_distance"),
        )
    pairs, pair_selection = _cross_paper_pairs(
        supported, max_pairs=max_pairs, paper_groups=paper_groups, titles=title_by_paper
    )
    pair_reports: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "pair_judgments.checkpoint.jsonl"
    completed_pairs: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pair_id"):
                completed_pairs[str(row["pair_id"])] = row

    def _checkpoint(report: dict[str, Any], edge: dict[str, Any] | None) -> None:
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pair_report": report, "edge": edge, "pair_id": report.get("pair_id")}, sort_keys=True) + "\n")

    for pair in pairs:
        resumed = completed_pairs.get(str(pair["pair_id"]))
        if resumed is not None:
            restored_edge = resumed.get("edge")
            if restored_edge and restored_edge.get("edge_id"):
                edges.setdefault(restored_edge["edge_id"], restored_edge)
            pair_reports.append(resumed.get("pair_report") or {"pair_id": pair["pair_id"], "status": "resumed"})
            continue
        pair_report = {
            "pair_id": pair["pair_id"],
            "fact_type": pair["fact_type"],
            "paper_a": pair["fact_a"].get("paper_id"),
            "paper_b": pair["fact_b"].get("paper_id"),
            "status": "needs_model_or_review" if provider == "none" else "pending",
            "relation": None,
            "edge_id": None,
        }
        if provider == "none":
            pair_reports.append(pair_report)
            continue
        packet = _relation_packet(pair, chunk_by_id=chunk_by_id, max_chunk_chars=max_chunk_chars)
        try:
            raw = _call_relation_model(
                packet,
                provider=provider,
                model=model,
                ollama_url=ollama_url,
                vllm_url=vllm_url,
                api_key=api_key,
            )
        except Exception as exc:
            pair_report["status"] = "error"
            errors.append({"pair_id": pair["pair_id"], "error": f"{provider} relation judgment failed: {exc}"})
            pair_reports.append(pair_report)
            _checkpoint(pair_report, None)
            continue
        judgment, judgment_errors = _validate_relation_judgment(raw, pair=pair, chunk_by_id=chunk_by_id)
        # An empty relation is the model declining to answer, not a verdict of
        # "unrelated" - 27 of 60 pairs came back this way on the 48-paper run.
        # Retry once naming the allowed values rather than coercing a judgment
        # the model never made.
        if judgment is None and not str(raw.get("relation") or "").strip():
            try:
                retry_raw = _call_relation_model(
                    {**packet, "relation_enum_reminder": sorted(RELATION_JUDGMENTS)},
                    provider=provider,
                    model=model,
                    ollama_url=ollama_url,
                    vllm_url=vllm_url,
                    api_key=api_key,
                )
            except Exception as exc:
                errors.append({"pair_id": pair["pair_id"], "error": f"{provider} relation retry failed: {exc}"})
            else:
                retry_judgment, retry_errors = _validate_relation_judgment(retry_raw, pair=pair, chunk_by_id=chunk_by_id)
                if retry_judgment is not None:
                    judgment = retry_judgment
                    judgment_errors = []
                    pair_report["warnings"] = (pair_report.get("warnings") or []) + [
                        "empty_relation_on_first_attempt: recovered by one retry naming the allowed relations."
                    ]
                else:
                    judgment_errors = judgment_errors + [f"empty_relation_after_retry: {retry_errors}"]
        errors.extend({"pair_id": pair["pair_id"], "error": item} for item in judgment_errors)
        if judgment is None:
            pair_report["status"] = "invalid_model_output"
            pair_reports.append(pair_report)
            _checkpoint(pair_report, None)
            continue
        pair_report["relation"] = judgment["relation"]
        pair_report["confidence"] = judgment.get("confidence")
        # Merge, never overwrite: the retry marker set above is provenance about
        # how this verdict was obtained and must survive.
        pair_report["warnings"] = (pair_report.get("warnings") or []) + [str(w) for w in (judgment.get("warnings") or [])]
        edge_type = RELATION_EDGE_TYPES.get(judgment["relation"])
        if edge_type is None:
            pair_report["status"] = "no_edge"
            pair_reports.append(pair_report)
            _checkpoint(pair_report, None)
            continue
        edge = _relation_edge(pair, judgment, edge_type=edge_type, model=model, provider=provider)
        edges.setdefault(edge["edge_id"], edge)
        pair_report["status"] = "edge_proposed"
        pair_report["edge_id"] = edge["edge_id"]
        pair_reports.append(pair_report)
        _checkpoint(pair_report, edge)

    proposed_edges = list(edges.values())
    proposed_nodes = list(nodes.values())
    merge_summary = _merge_into_catalog(catalog_dir, nodes=proposed_nodes, edges=proposed_edges)
    review_template = _review_template_rows(proposed_edges, nodes=nodes, catalog_dir=catalog_dir)
    _write_jsonl(output_dir / "evidence_graph_review_template.jsonl", review_template)
    _write_jsonl(output_dir / "proposed_evidence_graph_nodes.jsonl", proposed_nodes)
    _write_jsonl(output_dir / "proposed_evidence_graph_edges.jsonl", proposed_edges)

    if provider == "none":
        status = "needs_model_or_review" if pairs else "ok"
    else:
        status = "ok" if not errors else "partial"
    report = {
        "status": status,
        "controller": "bounded_edge_proposer_v0",
        "prompt_version": PROMPT_VERSION,
        "catalog": str(catalog_dir),
        "provider": provider,
        "model": model or None,
        "supported_fact_count": len(supported),
        "stage_a": stage_a,
        "pair_count": len(pairs),
        "pair_selection": pair_selection,
        "community_grouping": community_grouping,
        "pair_reports": pair_reports,
        "proposed_node_count": len(proposed_nodes),
        "proposed_edge_count": len(proposed_edges),
        "edge_type_counts": _count_by(proposed_edges, "edge_type"),
        "merge_summary": merge_summary,
        "errors": errors,
        "policy": [
            "Edge types come from a curated taxonomy; the model may not invent node or edge types.",
            "Every proposed edge is status=proposed and requires human review before it is treated as accepted evidence.",
            "Relation judgments must cite chunk quotes; unverifiable quotes are downgraded, not trusted.",
            "This proposer never deletes or overwrites reviewed edges; it only adds proposed candidates.",
        ],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_proposer_markdown,
        json_filename="evidence_graph_proposer_report.json",
        markdown_filename="evidence_graph_proposer_report.md",
    )
    (output_dir / "evidence_graph_proposer_report.html").write_text(_proposer_html(report), encoding="utf-8")
    report["artifacts"]["html"] = str(output_dir / "evidence_graph_proposer_report.html")
    report["artifacts"]["review_template"] = str(output_dir / "evidence_graph_review_template.jsonl")
    return report


def _project_fact_edges(
    supported: list[dict[str, Any]],
    *,
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    catalog_dir: Path,
    title_by_paper: dict[str, Any],
    case_target: str,
    case_target_label: str,
) -> dict[str, Any]:
    prov = [
        ProvenanceRecord(
            source_type="paper_facts",
            source_id=str(catalog_dir / "paper_facts.jsonl"),
            method="deterministic_fact_projection",
        )
    ]
    counts = {"REPORTS_FACT": 0, "FACT_SUPPORTED_BY": 0, "ANALOG_MODEL_PRIOR_FOR": 0}
    case_node_id = None
    if case_target:
        case_node_id = f"CaseTarget:{_slug(case_target)}"
        _add_node(nodes, node_id=case_node_id, node_type="CaseTarget", label=case_target_label, source_object_id=case_target, provenance=prov)

    for fact in supported:
        fact_id = str(fact.get("fact_id"))
        paper_id = str(fact.get("paper_id") or "")
        fact_type = str(fact.get("fact_type") or "")
        paper_node = f"Paper:{paper_id}"
        fact_node = f"PaperFact:{fact_id}"
        _add_node(nodes, node_id=paper_node, node_type="Paper", label=str(title_by_paper.get(paper_id) or paper_id), source_object_id=paper_id, provenance=prov)
        _add_node(
            nodes,
            node_id=fact_node,
            node_type="PaperFact",
            label=f"{fact_type}: {_fact_value_text(fact)}"[:200],
            source_object_id=fact_id,
            provenance=prov,
            metadata={"fact_type": fact_type, "paper_id": paper_id},
        )
        confidence = _coerce_confidence(fact.get("confidence"), default=0.6)
        _add_edge(
            edges,
            source_node_id=paper_node,
            target_node_id=fact_node,
            edge_type="REPORTS_FACT",
            confidence=confidence,
            evidence=[str(chunk_id) for chunk_id in fact.get("source_chunk_ids") or []],
            provenance=prov,
        )
        counts["REPORTS_FACT"] += 1
        for evidence in fact.get("evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunk_node = f"SourceChunk:{chunk_id}"
            _add_node(nodes, node_id=chunk_node, node_type="SourceChunk", label=chunk_id, source_object_id=chunk_id, provenance=prov)
            supported_evidence = str(evidence.get("support_status") or "proposed") == "supported"
            quote = str(evidence.get("quote") or "")
            _add_edge(
                edges,
                source_node_id=fact_node,
                target_node_id=chunk_node,
                edge_type="FACT_SUPPORTED_BY",
                confidence=0.9 if supported_evidence else 0.5,
                evidence=[quote] if quote else [],
                provenance=prov,
            )
            counts["FACT_SUPPORTED_BY"] += 1
        if case_node_id and fact_type in ANALOG_PRIOR_FACT_TYPES:
            _add_edge(
                edges,
                source_node_id=fact_node,
                target_node_id=case_node_id,
                edge_type="ANALOG_MODEL_PRIOR_FOR",
                confidence=0.5,
                evidence=[f"analog prior from {paper_id} ({fact_type})"],
                provenance=prov,
            )
            counts["ANALOG_MODEL_PRIOR_FOR"] += 1
    return {"edge_counts": counts, "case_target_node": case_node_id}


def _material_keys_by_paper(
    facts: list[dict[str, Any]],
    titles: dict[str, Any] | None = None,
) -> dict[str, set[str]]:
    """Normalized material identifiers each paper talks about.

    Used to keep relation pairing inside a subject. Chemical formulas arrive
    with inconsistent spacing, subscript artifacts, and case, so compare on
    alphanumerics only.

    Keys come from BOTH material-type facts and formula-like tokens in the paper
    title. Titles matter: a paper whose material fact was never extracted - and
    extraction returns nothing at all for some papers - would otherwise be
    unpairable, silently dropping it from stage B.
    """
    keys: dict[str, set[str]] = {}
    from_facts: dict[str, set[str]] = {}
    for fact in facts:
        if str(fact.get("fact_type") or "") not in _MATERIAL_IDENTITY_FACT_TYPES:
            continue
        value = fact.get("value")
        text = value.get("text") if isinstance(value, dict) else value
        normalized = re.sub(r"[^a-z0-9]", "", str(text or "").lower())
        if len(normalized) >= 4:
            from_facts.setdefault(str(fact.get("paper_id")), set()).add(normalized)
    for paper_id, fact_keys in from_facts.items():
        if len(fact_keys) <= _MAX_MATERIAL_KEYS_FROM_FACTS:
            keys.setdefault(paper_id, set()).update(fact_keys)
    for paper_id, title in (titles or {}).items():
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\.\-]*", str(title or "")):
            # Formula-like: contains a digit and a letter (CaMn2Sb2, Ho2Ti2O7).
            if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
                normalized = re.sub(r"[^a-z0-9]", "", token.lower())
                if len(normalized) >= 4:
                    keys.setdefault(str(paper_id), set()).add(normalized)
    return keys


def _papers_share_subject(
    paper_a: str,
    paper_b: str,
    *,
    material_keys: dict[str, set[str]],
    paper_groups: dict[str, set[str]],
) -> bool:
    groups_a, groups_b = paper_groups.get(paper_a), paper_groups.get(paper_b)
    if groups_a and groups_b:
        return bool(groups_a & groups_b)
    keys_a, keys_b = material_keys.get(paper_a), material_keys.get(paper_b)
    if not keys_a or not keys_b:
        return False
    if keys_a & keys_b:
        return True
    # One formula containing the other catches "CaMn2Sb2" vs "CaMn2Sb2 single
    # crystals" without matching unrelated compounds.
    return any(a in b or b in a for a in keys_a for b in keys_b)


def _cross_paper_pairs(
    supported: list[dict[str, Any]],
    *,
    max_pairs: int,
    paper_groups: dict[str, set[str]] | None = None,
    titles: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair same-type facts across papers that share a subject.

    Pairing every same-type fact across an entire heterogeneous corpus asks the
    model to relate quantities that have nothing to do with each other: on the
    48-paper BT-7 set, 33 of 60 such pairs were correctly judged "unrelated" and
    not one labelable edge came out of the run. Two papers must therefore share
    a material (or an explicitly supplied grouping, e.g. papers linked to the
    same experiment) before their facts are compared.
    """
    material_keys = _material_keys_by_paper(supported, titles)
    groups = paper_groups or {}
    papers_with_facts = {
        str(fact.get("paper_id"))
        for fact in supported
        if str(fact.get("fact_type") or "") in PAIRABLE_FACT_TYPES
    }
    by_type: dict[str, list[dict[str, Any]]] = {}
    for fact in supported:
        fact_type = str(fact.get("fact_type") or "")
        if fact_type in PAIRABLE_FACT_TYPES:
            by_type.setdefault(fact_type, []).append(fact)
    pairs: list[dict[str, Any]] = []
    considered = 0
    skipped_same_paper = 0
    skipped_different_subject = 0
    for fact_type in sorted(by_type):
        facts = sorted(by_type[fact_type], key=lambda item: str(item.get("fact_id")))
        for i, fact_a in enumerate(facts):
            for fact_b in facts[i + 1 :]:
                considered += 1
                paper_a, paper_b = str(fact_a.get("paper_id")), str(fact_b.get("paper_id"))
                if paper_a == paper_b:
                    skipped_same_paper += 1
                    continue
                if not _papers_share_subject(paper_a, paper_b, material_keys=material_keys, paper_groups=groups):
                    skipped_different_subject += 1
                    continue
                pair_id = f"pair-{_hash(str(fact_a.get('fact_id')) + str(fact_b.get('fact_id')))}"
                pairs.append({"pair_id": pair_id, "fact_type": fact_type, "fact_a": fact_a, "fact_b": fact_b})
    selection = {
        "candidate_pairs_considered": considered,
        "skipped_same_paper": skipped_same_paper,
        "skipped_different_subject": skipped_different_subject,
        "eligible_pairs": len(pairs),
        "pairs_selected": min(len(pairs), max_pairs),
        "subject_scoping": "explicit_paper_groups" if groups else "shared_material_fact",
        # Count only papers that actually contribute pairable facts: material_keys
        # is built from every catalog title too, so its raw length overstates
        # coverage and would be read as "papers eligible for pairing".
        "papers_with_pairable_facts": len(papers_with_facts),
        "papers_with_subject_key": len([paper for paper in papers_with_facts if material_keys.get(paper)]),
    }
    if len(pairs) > max_pairs:
        selection["note"] = f"{len(pairs) - max_pairs} eligible pairs were dropped by max_pairs={max_pairs}."
    return pairs[:max_pairs], selection


def _relation_packet(pair: dict[str, Any], *, chunk_by_id: dict[str, dict[str, Any]], max_chunk_chars: int) -> dict[str, Any]:
    def fact_side(fact: dict[str, Any]) -> dict[str, Any]:
        cited = [str(chunk_id) for chunk_id in fact.get("source_chunk_ids") or []]
        return {
            "fact_id": fact.get("fact_id"),
            "paper_id": fact.get("paper_id"),
            "fact_type": fact.get("fact_type"),
            "value": fact.get("value"),
            "evidence": [
                {"chunk_id": item.get("chunk_id"), "quote": item.get("quote")}
                for item in fact.get("evidence") or []
            ],
            "chunks": [
                {"chunk_id": chunk_id, "text": str(chunk_by_id.get(chunk_id, {}).get("text") or "")[:max_chunk_chars]}
                for chunk_id in cited
                if chunk_id in chunk_by_id
            ],
        }

    return {
        "pair_id": pair["pair_id"],
        "fact_type": pair["fact_type"],
        "prompt_version": PROMPT_VERSION,
        "instructions": [
            "Two papers report a fact of the same type. Judge how the two values relate.",
            "Use only the provided values, quotes, and chunks. Do not use outside knowledge of these materials.",
            "Allowed relation values: " + ", ".join(sorted(RELATION_JUDGMENTS)) + ".",
            "consistent_with: the two values agree within stated precision or wording.",
            "conflicts_with: the two values disagree about the same quantity under comparable conditions.",
            "same_quantity_different_conditions: same physical quantity, but conditions (temperature, field, composition) differ enough that agreement cannot be judged.",
            "unrelated: the two facts do not describe the same physical quantity.",
            "ambiguous: the provided evidence is insufficient to judge.",
            "Cite one short exact quote from each side's chunks supporting your judgment.",
            "Copy each quote character-for-character from the chunk text field, including OCR artifacts, odd spacing, and symbols. Do not normalize, translate symbols, or paraphrase.",
            "The chunk_id you cite must be the chunk whose text field actually contains your quote.",
        ],
        "output_schema": {
            "pair_id": pair["pair_id"],
            "relation": "consistent_with|conflicts_with|same_quantity_different_conditions|unrelated|ambiguous",
            "confidence": 0.0,
            "evidence_a": {"chunk_id": "chunk-id", "quote": "short exact quote"},
            "evidence_b": {"chunk_id": "chunk-id", "quote": "short exact quote"},
            "rationale": "one sentence",
        },
        "fact_a": fact_side(pair["fact_a"]),
        "fact_b": fact_side(pair["fact_b"]),
    }


def _relation_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pair_id": {"type": "string"},
            "relation": {"type": "string", "enum": sorted(RELATION_JUDGMENTS)},
            "confidence": {"type": "number"},
            "evidence_a": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["chunk_id"],
            },
            "evidence_b": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["chunk_id"],
            },
            "rationale": {"type": "string"},
        },
        "required": ["pair_id", "relation"],
    }


_RELATION_SYSTEM_PROMPT = (
    "You judge relations between two typed scientific paper facts for TAS Co-pilot. "
    "Return only JSON matching the schema. Use only the provided chunks and quotes. "
    "Do not invent relations outside the allowed enum. Prefer ambiguous over guessing. "
    "Quotes must be copied character-for-character from chunk text, preserving OCR "
    "artifacts, spacing, and symbols; paraphrased quotes are treated as unverified."
)


def _call_relation_model(
    packet: dict[str, Any],
    *,
    provider: str,
    model: str,
    ollama_url: str,
    vllm_url: str,
    api_key: Any,
) -> dict[str, Any]:
    schema = _relation_response_schema()
    messages = [
        {"role": "system", "content": _RELATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    if provider == "ollama":
        body = ollama_chat(
            ollama_url,
            model,
            messages,
            format_schema=schema,
            options={"temperature": 0, "num_predict": 1024},
            timeout=300.0,
        )
        content = str((body.get("message") or {}).get("content") or "{}")
    elif provider == "vllm":
        body = openai_compatible_chat(
            base_url=vllm_url,
            model=model,
            messages=messages,
            api_key=str(api_key) if api_key else None,
            response_format=schema,
            max_tokens=1024,
            timeout=300.0,
        )
        choices = body.get("choices") or []
        message = (choices[0].get("message") if choices and isinstance(choices[0], dict) else body.get("message")) or {}
        content = str(message.get("content") or "{}")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return {"errors": [f"relation judgment returned malformed JSON: {exc}"]}
    return data if isinstance(data, dict) else {"errors": ["relation judgment returned non-object JSON"]}


def _validate_relation_judgment(
    raw: dict[str, Any],
    *,
    pair: dict[str, Any],
    chunk_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = [str(item) for item in raw.get("errors") or []]
    relation = str(raw.get("relation") or "")
    if relation not in RELATION_JUDGMENTS:
        errors.append(f"relation {relation!r} is not in the allowed enum")
        return None, errors
    if str(raw.get("pair_id") or "") not in {"", pair["pair_id"]}:
        errors.append("model returned a judgment for a different pair_id")
        return None, errors
    warnings: list[str] = []
    cited: dict[str, dict[str, Any]] = {}
    confidence_cap = 1.0
    for side, fact in (("evidence_a", pair["fact_a"]), ("evidence_b", pair["fact_b"])):
        evidence = raw.get(side)
        if not isinstance(evidence, dict) or not evidence.get("chunk_id"):
            warnings.append(f"{side} missing; relation kept but requires review")
            confidence_cap = min(confidence_cap, 0.3)
            continue
        chunk_id = str(evidence.get("chunk_id"))
        quote = str(evidence.get("quote") or "").strip()
        tier, corrected_chunk_id, tier_warnings = _verify_relation_quote(
            quote,
            chunk_id=chunk_id,
            fact=fact,
            chunk_by_id=chunk_by_id,
        )
        warnings.extend(f"{side} {item}" for item in tier_warnings)
        confidence_cap = min(confidence_cap, _QUOTE_TIER_CAPS[tier])
        cited[side] = {"chunk_id": corrected_chunk_id, "quote": quote or None, "quote_verification": tier}
    judgment = {
        "relation": relation,
        "confidence": min(_coerce_confidence(raw.get("confidence"), default=0.3), confidence_cap),
        "rationale": str(raw.get("rationale") or "")[:500],
        "evidence": cited,
        "warnings": warnings,
    }
    return judgment, errors


# Confidence ceilings per quote-verification tier. Verified provenance keeps
# the model's confidence; real-but-misattributed or approximate quotes cap at
# 0.5; unverifiable quotes cap at 0.3. Review decides in every case.
_QUOTE_TIER_CAPS = {
    "verbatim_in_cited_chunk": 1.0,
    "relocated_within_fact_chunks": 1.0,
    "echoed_extraction_quote": 1.0,
    "found_elsewhere_in_paper": 0.5,
    "approximate_in_cited_chunk": 0.5,
    "unverified": 0.3,
}


def _verify_relation_quote(
    quote: str,
    *,
    chunk_id: str,
    fact: dict[str, Any],
    chunk_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str, list[str]]:
    """Grade a cited quote against the catalog: exact provenance, mechanical
    relocation, echo of the fact's original extraction quote, same-paper
    misattribution, approximate wording, or unverified.

    Returns (tier, corrected_chunk_id, warnings). Relocation only happens
    within the fact's own cited chunks, so citations never migrate across
    facts or papers silently.
    """

    fact_chunk_ids = [str(item) for item in fact.get("source_chunk_ids") or []]
    warnings: list[str] = []
    if chunk_id not in fact_chunk_ids:
        warnings.append(f"cites chunk {chunk_id} outside the fact's cited chunks")
    if not quote:
        warnings.append("has no quote; relation kept but requires review")
        return "unverified", chunk_id, warnings

    quote_key = _alnum_text(quote)
    cited_chunk = chunk_by_id.get(chunk_id)
    if cited_chunk is not None and quote_key and quote_key in _alnum_text(str(cited_chunk.get("text") or "")):
        return "verbatim_in_cited_chunk", chunk_id, warnings

    # Relocation first: verifying against actual chunk text is stronger
    # provenance than matching the extraction quote.
    for other_id in fact_chunk_ids:
        if other_id == chunk_id:
            continue
        other = chunk_by_id.get(other_id)
        if other is not None and quote_key and quote_key in _alnum_text(str(other.get("text") or "")):
            warnings.append(f"quote found in {other_id}, not the cited {chunk_id}; citation corrected")
            return "relocated_within_fact_chunks", other_id, warnings

    # The model may echo the extraction quote the fact was validated with;
    # that quote was already chunk-tied by normalize_paper_fact_response.
    for item in fact.get("evidence") or []:
        extraction_quote = _alnum_text(str(item.get("quote") or ""))
        if extraction_quote and quote_key and (quote_key in extraction_quote or extraction_quote in quote_key):
            warnings.append(f"quote echoes the fact's extraction quote from chunk {item.get('chunk_id')}")
            return "echoed_extraction_quote", str(item.get("chunk_id") or chunk_id), warnings

    paper_id = str(fact.get("paper_id") or "")
    for other_id, other in chunk_by_id.items():
        if str(other.get("paper_id") or "") != paper_id or other_id == chunk_id:
            continue
        if quote_key and quote_key in _alnum_text(str(other.get("text") or "")):
            warnings.append(f"quote found elsewhere in the paper ({other_id}), outside the fact's cited chunks")
            return "found_elsewhere_in_paper", other_id, warnings

    # 0.8 token coverage admits OCR symbol substitutions (e.g. "±" -> "7")
    # while fabricated values stay below the line; capped at 0.5 either way.
    if cited_chunk is not None and _token_coverage(quote, str(cited_chunk.get("text") or "")) >= 0.8:
        warnings.append(f"quote is approximate, not verbatim, in chunk {chunk_id}")
        return "approximate_in_cited_chunk", chunk_id, warnings

    warnings.append(f"quote not found in chunk {chunk_id} or the fact's paper")
    return "unverified", chunk_id, warnings


def _alnum_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _token_coverage(quote: str, text: str) -> float:
    quote_tokens = [token for token in _normalize_text("".join(c if c.isalnum() else " " for c in quote)).split() if len(token) >= 2]
    if len(quote_tokens) < 4:
        return 0.0
    text_tokens = set(_normalize_text("".join(c if c.isalnum() else " " for c in text)).split())
    return sum(1 for token in quote_tokens if token in text_tokens) / len(quote_tokens)


def _relation_edge(
    pair: dict[str, Any],
    judgment: dict[str, Any],
    *,
    edge_type: str,
    model: str,
    provider: str,
) -> dict[str, Any]:
    fact_a = pair["fact_a"]
    fact_b = pair["fact_b"]
    source_node_id = f"PaperFact:{fact_a.get('fact_id')}"
    target_node_id = f"PaperFact:{fact_b.get('fact_id')}"
    evidence = [judgment.get("rationale") or ""]
    for side in ("evidence_a", "evidence_b"):
        item = (judgment.get("evidence") or {}).get(side) or {}
        if item.get("quote"):
            evidence.append(f"{item.get('chunk_id')}: {item.get('quote')}")
    prov = [
        ProvenanceRecord(
            source_type="relation_judgment",
            source_id=pair["pair_id"],
            method=f"bounded_relation_judgment_{provider}",
            confidence=judgment.get("confidence"),
            metadata={"model": model, "prompt_version": PROMPT_VERSION, "warnings": judgment.get("warnings") or []},
        )
    ]
    return _edge_dict(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence=float(judgment.get("confidence") or 0.0),
        evidence=[item for item in evidence if item],
        provenance=prov,
    )


def _add_node(
    nodes: dict[str, dict[str, Any]],
    *,
    node_id: str,
    node_type: str,
    label: str,
    source_object_id: str,
    provenance: list[ProvenanceRecord],
    metadata: dict[str, Any] | None = None,
) -> None:
    if node_id in nodes:
        return
    nodes[node_id] = EvidenceGraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        source_object_id=source_object_id,
        metadata=metadata or {},
        provenance=provenance,
    ).model_dump(mode="json")


def _add_edge(
    edges: dict[str, dict[str, Any]],
    *,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    confidence: float,
    evidence: list[str],
    provenance: list[ProvenanceRecord],
) -> None:
    edge = _edge_dict(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence=confidence,
        evidence=evidence,
        provenance=provenance,
    )
    edges.setdefault(edge["edge_id"], edge)


def _edge_dict(
    *,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    confidence: float,
    evidence: list[str],
    provenance: list[ProvenanceRecord],
) -> dict[str, Any]:
    # Same edge-id convention as corpus.pilot.build_evidence_graph so merged
    # graphs stay dedupable.
    edge_id = f"{edge_type.lower()}-{_hash(source_node_id + target_node_id + edge_type)}"
    return EvidenceGraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence=max(0.0, min(1.0, confidence)),
        status="proposed",
        evidence=evidence,
        provenance=provenance,
        created_by=CREATED_BY,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump(mode="json")


def _merge_into_catalog(catalog_dir: Path, *, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Append proposed nodes/edges to the catalog graph without touching
    existing records; reviewed edges always win over re-proposals."""

    existing_nodes = _load_jsonl(catalog_dir / "evidence_graph_nodes.jsonl")
    existing_edges = _load_jsonl(catalog_dir / "evidence_graph_edges.jsonl")
    node_ids = {str(node.get("node_id")) for node in existing_nodes}
    edge_ids = {str(edge.get("edge_id")) for edge in existing_edges}
    new_nodes = [node for node in nodes if str(node.get("node_id")) not in node_ids]
    new_edges = [edge for edge in edges if str(edge.get("edge_id")) not in edge_ids]
    _write_jsonl(catalog_dir / "evidence_graph_nodes.jsonl", existing_nodes + new_nodes)
    _write_jsonl(catalog_dir / "evidence_graph_edges.jsonl", existing_edges + new_edges)
    return {
        "existing_node_count": len(existing_nodes),
        "existing_edge_count": len(existing_edges),
        "new_node_count": len(new_nodes),
        "new_edge_count": len(new_edges),
        "skipped_existing_edge_count": len(edges) - len(new_edges),
    }


def _review_template_rows(edges: list[dict[str, Any]], *, nodes: dict[str, dict[str, Any]], catalog_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for edge in edges:
        source = nodes.get(str(edge.get("source_node_id")), {})
        target = nodes.get(str(edge.get("target_node_id")), {})
        rows.append(
            {
                "object_type": "evidence_graph_edge",
                "object_id": edge.get("edge_id"),
                "object_secondary_id": edge.get("edge_type"),
                "source_workflow": "evidence-graph-propose",
                "source_artifact_path": str(catalog_dir / "evidence_graph_edges.jsonl"),
                "verdict": "unreviewed",
                "claim_state": "proposed",
                "suggested_status": "proposed",
                "type_specific_status": None,
                "type_specific_allowed": ["accepted", "rejected", "inconclusive"],
                "review_notes": "",
                "metadata": {
                    "edge_type": edge.get("edge_type"),
                    "source_node": source.get("label") or edge.get("source_node_id"),
                    "target_node": target.get("label") or edge.get("target_node_id"),
                    "confidence": edge.get("confidence"),
                    "evidence": edge.get("evidence") or [],
                    "created_by": edge.get("created_by"),
                },
            }
        )
    return rows


def apply_evidence_graph_labels(arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply reviewed edge labels back onto the catalog evidence graph.

    This closes the propose -> review -> retrieve loop: accepted edges gain
    full weight in the GraphRetriever, rejected edges sink, and inconclusive
    edges stay proposed with a review note. Labels always record who reviewed
    and when; prior status is preserved in edge provenance-style metadata so
    the change is auditable.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    labels_path = Path(arguments["labels"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser() if arguments.get("output_dir") else None
    reviewer = str(arguments.get("reviewer") or "") or None
    edges_path = catalog_dir / "evidence_graph_edges.jsonl"
    if not edges_path.exists():
        return {"status": "missing_input", "errors": [{"error": f"No evidence graph at {edges_path}"}], "applied_count": 0}
    if not labels_path.exists():
        return {"status": "missing_input", "errors": [{"error": f"Labels file does not exist: {labels_path}"}], "applied_count": 0}

    verdict_to_status = {"accepted": "accepted", "rejected": "rejected", "inconclusive": "proposed"}
    labels_by_edge: dict[str, dict[str, Any]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for row in _load_jsonl(labels_path):
        if str(row.get("object_type")) != "evidence_graph_edge":
            skipped_rows.append({"object_id": row.get("object_id"), "reason": "not an evidence_graph_edge label"})
            continue
        verdict = str(row.get("type_specific_status") or row.get("verdict") or "").lower()
        if verdict in {"", "unreviewed"}:
            skipped_rows.append({"object_id": row.get("object_id"), "reason": "unreviewed row"})
            continue
        if verdict not in verdict_to_status:
            skipped_rows.append({"object_id": row.get("object_id"), "reason": f"unknown verdict {verdict!r}"})
            continue
        labels_by_edge[str(row.get("object_id"))] = {"verdict": verdict, "row": row}

    edges = _load_jsonl(edges_path)
    applied: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for edge in edges:
        label = labels_by_edge.pop(str(edge.get("edge_id")), None)
        if label is None:
            continue
        verdict = label["verdict"]
        row = label["row"]
        prior_status = str(edge.get("status") or "proposed")
        new_status = verdict_to_status[verdict]
        edge["status"] = new_status
        edge["reviewed_by"] = str(row.get("reviewer_id") or "") or reviewer
        edge["reviewed_at"] = str(row.get("reviewed_at") or "") or now
        if verdict == "rejected":
            edge["rejection_reason"] = str(row.get("review_notes") or "") or "rejected in review"
        elif verdict == "inconclusive":
            evidence = list(edge.get("evidence") or [])
            note = str(row.get("review_notes") or "") or "review inconclusive; stays proposed"
            evidence.append(f"review note: {note}")
            edge["evidence"] = evidence
        applied.append(
            {
                "edge_id": edge.get("edge_id"),
                "edge_type": edge.get("edge_type"),
                "verdict": verdict,
                "prior_status": prior_status,
                "new_status": new_status,
                "reviewed_by": edge.get("reviewed_by"),
            }
        )
    unmatched = [{"object_id": edge_id, "reason": "no matching edge in catalog"} for edge_id in labels_by_edge]
    _write_jsonl(edges_path, edges)

    report = {
        "status": "ok" if applied and not unmatched else ("partial" if applied else "missing_input"),
        "catalog": str(catalog_dir),
        "labels": str(labels_path),
        "applied_count": len(applied),
        "applied": applied,
        "by_verdict": _count_by(applied, "verdict"),
        "skipped_rows": skipped_rows + unmatched,
        "safe_conclusion": (
            "Applied labels change retrieval weighting only. Accepted edges are reviewed evidence, "
            "not calibrated probabilities, until enough labels exist for calibration."
        ),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = write_report_bundle(
            output_dir,
            report,
            markdown_renderer=_apply_labels_markdown,
            json_filename="evidence_graph_apply_labels_report.json",
            markdown_filename="evidence_graph_apply_labels_report.md",
        )
    return report


def _apply_labels_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Graph Label Application",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Catalog: `{report.get('catalog')}`",
        f"- Applied: {report.get('applied_count')}",
        "",
        "## Applied Labels",
        "",
    ]
    for item in report.get("applied") or []:
        lines.append(
            f"- `{item.get('edge_id')}` `{item.get('edge_type')}`: "
            f"{item.get('prior_status')} -> {item.get('new_status')} ({item.get('verdict')}, by {item.get('reviewed_by')})"
        )
    if report.get("skipped_rows"):
        lines.extend(["", "## Skipped Rows", ""])
        lines.extend(f"- `{item.get('object_id')}`: {item.get('reason')}" for item in report["skipped_rows"])
    return "\n".join(lines) + "\n"


def _proposer_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Graph Candidate Edge Proposer",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Controller: `{report.get('controller')}`",
        f"- Provider: `{report.get('provider')}`",
        f"- Model: `{report.get('model')}`",
        f"- Supported facts: {report.get('supported_fact_count')}",
        f"- Proposed nodes: {report.get('proposed_node_count')}",
        f"- Proposed edges: {report.get('proposed_edge_count')}",
        "",
        "## Edge Types",
        "",
    ]
    for edge_type, count in sorted((report.get("edge_type_counts") or {}).items()):
        lines.append(f"- `{edge_type}`: {count}")
    lines.extend(["", "## Cross-Paper Pairs", ""])
    for pair in report.get("pair_reports") or []:
        lines.append(
            f"- `{pair.get('pair_id')}` `{pair.get('fact_type')}` "
            f"({pair.get('paper_a')} vs {pair.get('paper_b')}): {pair.get('status')}"
            + (f" relation={pair.get('relation')}" if pair.get("relation") else "")
        )
    if not (report.get("pair_reports") or []):
        lines.append("- No cross-paper pairs were available.")
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item.get('pair_id')}`: {item.get('error')}" for item in report["errors"])
    lines.extend(["", "## Policy", ""])
    lines.extend(f"- {item}" for item in report.get("policy") or [])
    return "\n".join(lines) + "\n"


def _proposer_html(report: dict[str, Any]) -> str:
    rows = []
    for pair in report.get("pair_reports") or []:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(pair.get('pair_id')))}</code></td>"
            f"<td><code>{html.escape(str(pair.get('fact_type')))}</code></td>"
            f"<td>{html.escape(str(pair.get('paper_a')))}<br>{html.escape(str(pair.get('paper_b')))}</td>"
            f"<td>{html.escape(str(pair.get('status')))}</td>"
            f"<td>{html.escape(str(pair.get('relation') or ''))}</td>"
            f"<td><code>{html.escape(str(pair.get('edge_id') or ''))}</code></td>"
            "</tr>"
        )
    edge_counts = "".join(
        f"<li><code>{html.escape(str(edge_type))}</code>: {count}</li>"
        for edge_type, count in sorted((report.get("edge_type_counts") or {}).items())
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Evidence-Graph Candidate Edge Proposer</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.45}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{background:#f4f4f4}code{background:#f7f7f7;padding:1px 3px}</style></head><body>"
        "<h1>Evidence-Graph Candidate Edge Proposer</h1>"
        f"<p>Status: <code>{html.escape(str(report.get('status')))}</code>; "
        f"proposed edges: {html.escape(str(report.get('proposed_edge_count')))}; "
        f"proposed nodes: {html.escape(str(report.get('proposed_node_count')))}</p>"
        "<p>All edges are status=proposed and require human review before acceptance. "
        "The relation taxonomy is curated; this is not an open-ended LLM knowledge graph.</p>"
        f"<h2>Edge types</h2><ul>{edge_counts}</ul>"
        "<h2>Cross-paper pairs</h2>"
        "<table><thead><tr><th>Pair</th><th>Fact type</th><th>Papers</th><th>Status</th><th>Relation</th><th>Edge</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )


def _missing_report(output_dir: Path, catalog_dir: Path, reason: str) -> dict[str, Any]:
    report = {
        "status": "missing_input",
        "catalog": str(catalog_dir),
        "proposed_node_count": 0,
        "proposed_edge_count": 0,
        "pair_reports": [],
        "errors": [{"error": reason}],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_proposer_markdown,
        json_filename="evidence_graph_proposer_report.json",
        markdown_filename="evidence_graph_proposer_report.md",
    )
    return report


def _fact_value_text(fact: dict[str, Any]) -> str:
    value = fact.get("value")
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if item is not None)
    return "" if value is None else str(value)


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
    return counts


def _coerce_confidence(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _slug(text: str) -> str:
    return "-".join("".join(char if char.isalnum() else " " for char in text.lower()).split())[:64]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
