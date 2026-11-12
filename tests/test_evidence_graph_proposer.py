from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus import evidence_graph_proposer as proposer_module
from tas_copilot_core.corpus.evidence_graph_proposer import apply_evidence_graph_labels, run_evidence_graph_proposer
from tas_copilot_core.retrieval import GraphRetriever


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _catalog_with_facts(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    _write_jsonl(
        catalog / "papers.jsonl",
        [
            {"paper_id": "paper-camn2sb2-bridges-powder", "title": "CaMn2Sb2 powder neutron diffraction"},
            {"paper_id": "paper-camn2sb2-mcnally-spinwaves", "title": "CaMn2Sb2 spin waves"},
        ],
    )
    _write_jsonl(
        catalog / "source_chunks.jsonl",
        [
            {
                "chunk_id": "chunk-a-001",
                "paper_id": "paper-camn2sb2-bridges-powder",
                "text": "The Mn moments lie in the ab plane with an ordered moment of 3.46 muB per Mn.",
            },
            {
                "chunk_id": "chunk-b-001",
                "paper_id": "paper-camn2sb2-mcnally-spinwaves",
                "text": "Spin wave fits give an ordered moment of about 3.4 muB with moments in the ab plane.",
            },
        ],
    )
    _write_jsonl(
        catalog / "paper_facts.jsonl",
        [
            {
                "fact_id": "paper-fact-aaa",
                "paper_id": "paper-camn2sb2-bridges-powder",
                "fact_type": "ordered_moment",
                "answer_status": "supported",
                "value": {"text": "3.46 muB per Mn"},
                "confidence": 0.9,
                "source_chunk_ids": ["chunk-a-001"],
                "evidence": [{"chunk_id": "chunk-a-001", "quote": "ordered moment of 3.46 muB per Mn", "support_status": "supported"}],
            },
            {
                "fact_id": "paper-fact-bbb",
                "paper_id": "paper-camn2sb2-mcnally-spinwaves",
                "fact_type": "ordered_moment",
                "answer_status": "supported",
                "value": {"text": "about 3.4 muB"},
                "confidence": 0.8,
                "source_chunk_ids": ["chunk-b-001"],
                "evidence": [{"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB", "support_status": "supported"}],
            },
            {
                "fact_id": "paper-fact-ccc",
                "paper_id": "paper-camn2sb2-mcnally-spinwaves",
                "fact_type": "exchange_interaction",
                "answer_status": "supported",
                "value": {"text": "J1 and J2 Heisenberg exchange"},
                "confidence": 0.85,
                "source_chunk_ids": ["chunk-b-001"],
                "evidence": [{"chunk_id": "chunk-b-001", "quote": "Spin wave fits", "support_status": "supported"}],
            },
            {
                "fact_id": "paper-fact-ddd",
                "paper_id": "paper-camn2sb2-bridges-powder",
                "fact_type": "temperature_range",
                "answer_status": "ambiguous",
                "value": {"text": "unclear"},
                "source_chunk_ids": [],
                "evidence": [],
            },
        ],
    )
    return catalog


def test_provider_none_projects_deterministic_edges_only(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)

    report = run_evidence_graph_proposer(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "out"),
            "provider": "none",
            "case_target": "srmn2sb2-experiment-24858",
            "case_target_label": "SrMn2Sb2 BT-7 experiment 24858",
        }
    )

    assert report["status"] == "needs_model_or_review"
    # Only supported facts project: 3 REPORTS_FACT, 3 FACT_SUPPORTED_BY,
    # 3 ANALOG_MODEL_PRIOR_FOR (ordered_moment x2 + exchange_interaction).
    assert report["stage_a"]["edge_counts"] == {
        "REPORTS_FACT": 3,
        "FACT_SUPPORTED_BY": 3,
        "ANALOG_MODEL_PRIOR_FOR": 3,
    }
    # The ambiguous fact must not appear in the graph.
    nodes = _jsonl(catalog / "evidence_graph_nodes.jsonl")
    assert not any("paper-fact-ddd" in str(node.get("node_id")) for node in nodes)
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    assert edges and all(edge["status"] == "proposed" for edge in edges)
    assert all(edge["created_by"] == "evidence_graph_proposer" for edge in edges)
    assert {edge["edge_type"] for edge in edges} <= proposer_module.DETERMINISTIC_EDGE_TYPES
    # The cross-paper pair is listed but unjudged without a model.
    assert report["pair_count"] == 1
    assert report["pair_reports"][0]["status"] == "needs_model_or_review"


def test_relation_judgment_creates_reviewable_edge_and_review_template(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog_with_facts(tmp_path)

    def fake_relation_model(packet, **_kwargs):
        return {
            "pair_id": packet["pair_id"],
            "relation": "consistent_with",
            "confidence": 0.8,
            "evidence_a": {"chunk_id": "chunk-a-001", "quote": "ordered moment of 3.46 muB per Mn"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Both papers report an ordered Mn moment near 3.4 muB.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    output_dir = tmp_path / "out"
    report = run_evidence_graph_proposer(
        {
            "catalog": str(catalog),
            "output_dir": str(output_dir),
            "provider": "ollama",
            "model": "gemma3:4b",
        }
    )

    assert report["status"] == "ok"
    assert report["pair_reports"][0]["status"] == "edge_proposed"
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    consistent = [edge for edge in edges if edge["edge_type"] == "CONSISTENT_WITH"]
    assert len(consistent) == 1
    assert consistent[0]["status"] == "proposed"
    assert consistent[0]["confidence"] == 0.8
    assert any("3.46 muB" in item for item in consistent[0]["evidence"])
    # Review template rows are ReviewLabel-shaped and cover every proposed edge.
    template = _jsonl(output_dir / "evidence_graph_review_template.jsonl")
    assert len(template) == report["proposed_edge_count"]
    assert all(row["object_type"] == "evidence_graph_edge" for row in template)
    assert all(row["verdict"] == "unreviewed" for row in template)
    assert all(row["type_specific_allowed"] == ["accepted", "rejected", "inconclusive"] for row in template)


def test_unverifiable_quote_downgrades_confidence(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog_with_facts(tmp_path)

    def fake_relation_model(packet, **_kwargs):
        return {
            "pair_id": packet["pair_id"],
            "relation": "conflicts_with",
            "confidence": 0.95,
            "evidence_a": {"chunk_id": "chunk-a-001", "quote": "this quote does not exist in the chunk"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Claimed conflict.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    report = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "ollama", "model": "gemma3:4b"}
    )

    pair = report["pair_reports"][0]
    assert pair["status"] == "edge_proposed"
    assert pair["warnings"]
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    conflict = next(edge for edge in edges if edge["edge_type"] == "CONFLICTS_WITH")
    # Mechanical downgrade: unverifiable citation caps confidence at 0.3.
    assert conflict["confidence"] <= 0.3


def test_invalid_relation_is_rejected_not_invented(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog_with_facts(tmp_path)

    def fake_relation_model(packet, **_kwargs):
        return {"pair_id": packet["pair_id"], "relation": "CAUSES", "confidence": 1.0}

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    report = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "ollama", "model": "gemma3:4b"}
    )

    assert report["status"] == "partial"
    assert report["pair_reports"][0]["status"] == "invalid_model_output"
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    assert all(edge["edge_type"] in proposer_module.DETERMINISTIC_EDGE_TYPES for edge in edges)


def test_rerun_is_idempotent_and_preserves_existing_graph(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)
    _write_jsonl(
        catalog / "evidence_graph_nodes.jsonl",
        [{"node_id": "Paper:paper-camn2sb2-bridges-powder", "node_type": "Paper", "label": "existing", "provenance": []}],
    )
    _write_jsonl(
        catalog / "evidence_graph_edges.jsonl",
        [
            {
                "edge_id": "has_chunk-existing",
                "source_node_id": "Paper:paper-camn2sb2-bridges-powder",
                "target_node_id": "SourceChunk:chunk-a-001",
                "edge_type": "HAS_CHUNK",
                "confidence": 0.95,
                "status": "accepted",
                "evidence": [],
                "provenance": [],
                "created_by": "corpus_pilot",
            }
        ],
    )

    first = run_evidence_graph_proposer({"catalog": str(catalog), "output_dir": str(tmp_path / "out1"), "provider": "none"})
    second = run_evidence_graph_proposer({"catalog": str(catalog), "output_dir": str(tmp_path / "out2"), "provider": "none"})

    assert first["merge_summary"]["new_edge_count"] > 0
    assert second["merge_summary"]["new_edge_count"] == 0
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    # The pre-existing accepted pilot edge is untouched.
    existing = next(edge for edge in edges if edge["edge_id"] == "has_chunk-existing")
    assert existing["status"] == "accepted"
    # Existing node labels are not overwritten by re-proposals.
    nodes = _jsonl(catalog / "evidence_graph_nodes.jsonl")
    paper_node = next(node for node in nodes if node["node_id"] == "Paper:paper-camn2sb2-bridges-powder")
    assert paper_node["label"] == "existing"


def test_proposed_edges_are_served_by_graph_retriever_with_downweighted_status(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)
    run_evidence_graph_proposer(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "out"),
            "provider": "none",
            "case_target": "srmn2sb2-experiment-24858",
            "case_target_label": "SrMn2Sb2 BT-7 experiment 24858",
        }
    )

    hits = GraphRetriever(catalog).search("ordered moment ab plane", limit=10)
    assert hits
    assert all(hit.retrieval_method == "typed_graph_traversal" for hit in hits)
    # Proposed edges surface with raised uncertainty, so review state is visible.
    assert all(hit.uncertainty >= 0.35 for hit in hits if (hit.structured_summary or {}).get("status") == "proposed")


def test_quote_verification_tiers_match_observed_failure_modes(tmp_path: Path) -> None:
    """Tiers are modeled on the failure modes from the first Vista run."""

    chunk_by_id = {
        "chunk-x-1": {"chunk_id": "chunk-x-1", "paper_id": "paper-x", "text": "The ordered moment is 3.25 7 0.06 m B at 20 K in CaMn 2 Sb 2 ."},
        "chunk-x-2": {"chunk_id": "chunk-x-2", "paper_id": "paper-x", "text": "Fits give an ordered moment between 2.8-3.4 µ B /Mn."},
        "chunk-x-3": {"chunk_id": "chunk-x-3", "paper_id": "paper-x", "text": "The spin wave excitations are well described by a Heisenberg model."},
    }
    fact = {
        "fact_id": "f-x",
        "paper_id": "paper-x",
        "source_chunk_ids": ["chunk-x-1", "chunk-x-2"],
        "evidence": [{"chunk_id": "chunk-x-2", "quote": "ordered moment between 2.8-3.4 µ B /Mn"}],
    }
    verify = proposer_module._verify_relation_quote

    # Spacing-only differences verify fully via alphanumeric normalization.
    tier, chunk_id, _ = verify("ordered moment is 3.25 7 0.06 m B", chunk_id="chunk-x-1", fact=fact, chunk_by_id=chunk_by_id)
    assert (tier, chunk_id) == ("verbatim_in_cited_chunk", "chunk-x-1")
    # OCR symbol substitution ("±" where the chunk has "7") is approximate:
    # real quote, imperfect fidelity, capped at 0.5.
    tier, chunk_id, _ = verify("ordered moment is 3.25±0.06 mB", chunk_id="chunk-x-1", fact=fact, chunk_by_id=chunk_by_id)
    assert (tier, chunk_id) == ("approximate_in_cited_chunk", "chunk-x-1")
    assert proposer_module._QUOTE_TIER_CAPS[tier] == 0.5
    # Quote real but cited against the wrong chunk: relocated, citation fixed.
    tier, chunk_id, _ = verify("moment between 2.8-3.4 µ B /Mn", chunk_id="chunk-x-1", fact=fact, chunk_by_id=chunk_by_id)
    assert (tier, chunk_id) == ("relocated_within_fact_chunks", "chunk-x-2")
    # Quote exists in the paper but outside the fact's cited chunks: capped 0.5.
    tier, chunk_id, _ = verify("spin wave excitations are well described", chunk_id="chunk-x-1", fact=fact, chunk_by_id=chunk_by_id)
    assert (tier, chunk_id) == ("found_elsewhere_in_paper", "chunk-x-3")
    assert proposer_module._QUOTE_TIER_CAPS[tier] == 0.5
    # Fabricated value ("3.38" never appears): unverified, capped 0.3.
    tier, _, _ = verify("a moment of 3.38 mB at 20K measured on BT7", chunk_id="chunk-x-1", fact=fact, chunk_by_id=chunk_by_id)
    assert tier == "unverified"
    assert proposer_module._QUOTE_TIER_CAPS[tier] == 0.3


def test_relocated_quote_keeps_model_confidence(tmp_path: Path, monkeypatch) -> None:
    catalog = _catalog_with_facts(tmp_path)

    def fake_relation_model(packet, **_kwargs):
        return {
            "pair_id": packet["pair_id"],
            "relation": "consistent_with",
            "confidence": 0.9,
            # evidence_a is a real quote but cites the wrong side's chunk.
            "evidence_a": {"chunk_id": "chunk-b-001", "quote": "ordered moment of 3.46 muB per Mn"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Same moment.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    report = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "ollama", "model": "gemma3:4b"}
    )

    pair = report["pair_reports"][0]
    assert pair["status"] == "edge_proposed"
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    edge = next(e for e in edges if e["edge_type"] == "CONSISTENT_WITH")
    # evidence_a is relocated to chunk-a-001 (real quote, wrong citation);
    # evidence_b is verbatim in its cited chunk. No cap applies.
    assert edge["confidence"] == 0.9
    assert any("citation corrected" in w for w in pair["warnings"])


def test_apply_labels_closes_the_review_loop(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)
    output_dir = tmp_path / "out"
    run_evidence_graph_proposer(
        {
            "catalog": str(catalog),
            "output_dir": str(output_dir),
            "provider": "none",
            "case_target": "srmn2sb2-experiment-24858",
        }
    )
    template = _jsonl(output_dir / "evidence_graph_review_template.jsonl")
    assert len(template) >= 3
    # A reviewer exports only the rows they actually judged.
    template[0]["type_specific_status"] = "accepted"
    template[0]["reviewer_id"] = "william"
    template[1]["type_specific_status"] = "rejected"
    template[1]["review_notes"] = "not supported by the cited chunk"
    template[2]["type_specific_status"] = "inconclusive"
    reviewed = template[:3]
    labels_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(labels_path, reviewed)

    report = apply_evidence_graph_labels(
        {"catalog": str(catalog), "labels": str(labels_path), "output_dir": str(tmp_path / "apply"), "reviewer": "fallback"}
    )

    assert report["applied_count"] == len(reviewed)
    assert report["status"] == "ok"
    edges = {edge["edge_id"]: edge for edge in _jsonl(catalog / "evidence_graph_edges.jsonl")}
    accepted = edges[template[0]["object_id"]]
    assert accepted["status"] == "accepted"
    assert accepted["reviewed_by"] == "william"
    assert accepted["reviewed_at"]
    rejected = edges[template[1]["object_id"]]
    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"] == "not supported by the cited chunk"
    assert rejected["reviewed_by"] == "fallback"
    inconclusive = edges[template[2]["object_id"]]
    assert inconclusive["status"] == "proposed"
    assert any("review note" in item for item in inconclusive["evidence"])

    # The retriever now weights the accepted edge above rejected ones.
    hits = GraphRetriever(catalog).search("ordered moment ab plane", limit=20)
    status_by_uncertainty = {
        (hit.structured_summary or {}).get("status"): hit.uncertainty for hit in hits
    }
    if "accepted" in status_by_uncertainty and "rejected" in status_by_uncertainty:
        assert status_by_uncertainty["accepted"] < status_by_uncertainty["rejected"]


def test_apply_labels_skips_unreviewed_and_unknown_rows(tmp_path: Path) -> None:
    catalog = _catalog_with_facts(tmp_path)
    run_evidence_graph_proposer({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "none"})
    labels_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        labels_path,
        [
            {"object_type": "evidence_graph_edge", "object_id": "nonexistent-edge", "type_specific_status": "accepted"},
            {"object_type": "evidence_graph_edge", "object_id": "whatever", "verdict": "unreviewed"},
            {"object_type": "paper_link", "object_id": "not-an-edge", "verdict": "accepted"},
        ],
    )

    report = apply_evidence_graph_labels({"catalog": str(catalog), "labels": str(labels_path)})

    assert report["applied_count"] == 0
    assert len(report["skipped_rows"]) == 3
    edges = _jsonl(catalog / "evidence_graph_edges.jsonl")
    assert all(edge["status"] == "proposed" for edge in edges)


def test_proposed_edges_surface_through_corpus_search(tmp_path: Path) -> None:
    """The router-facing corpus_search consults the typed graph channel."""

    from tas_copilot_core.corpus.pilot import corpus_search

    catalog = _catalog_with_facts(tmp_path)
    run_evidence_graph_proposer(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "out"),
            "provider": "none",
            "case_target": "srmn2sb2-experiment-24858",
            "case_target_label": "SrMn2Sb2 BT-7 experiment 24858",
        }
    )

    report = corpus_search({"catalog": str(catalog), "query": "ordered moment ab plane", "top_k": 12})

    assert report["status"] == "ok"
    methods = {result["retrieval_method"] for result in report["results"]}
    assert "typed_graph_traversal" in methods
    graph_results = [r for r in report["results"] if r["retrieval_method"] == "typed_graph_traversal"]
    # Proposed edges surface with visible review status and raised uncertainty.
    assert any((r.get("structured_summary") or {}).get("status") == "proposed" for r in graph_results)
    assert all(r["uncertainty"] >= 0.35 for r in graph_results if (r.get("structured_summary") or {}).get("status") == "proposed")


def test_missing_facts_is_a_missing_input_not_an_error(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    report = run_evidence_graph_proposer({"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "none"})
    assert report["status"] == "missing_input"
    assert report["proposed_edge_count"] == 0


def test_proposer_checkpoints_each_pair_and_resumes_from_disk(tmp_path: Path, monkeypatch) -> None:
    """A wall-killed proposer must not discard inference it already paid for.

    Every pair judgment is a model call. Before this, all 60 were held in memory
    and written only after the loop, so hitting the scheduler wall lost the whole
    run. The checkpoint is also the resume set.
    """
    from tas_copilot_core.corpus import evidence_graph_proposer as proposer_module

    catalog = _catalog_with_facts(tmp_path)
    calls: list[str] = []

    def fake_relation_model(packet, **_kwargs):
        calls.append(packet["pair_id"])
        return {
            "pair_id": packet["pair_id"],
            "relation": "consistent_with",
            "confidence": 0.8,
            "evidence_a": {"chunk_id": "chunk-a-001", "quote": "ordered moment of 3.46 muB per Mn"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Both papers report a similar Mn moment.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", fake_relation_model)
    output_dir = tmp_path / "prop"
    first = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(output_dir), "provider": "ollama", "model": "gemma4:26b"}
    )
    assert first["status"] in {"ok", "partial"}
    checkpoint = output_dir / "pair_judgments.checkpoint.jsonl"
    assert checkpoint.exists()
    rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "each judged pair must be persisted as it completes"
    assert all(row.get("pair_id") for row in rows)
    first_call_count = len(calls)
    assert first_call_count > 0

    # Re-running against the same output_dir replays from the checkpoint and
    # spends no further model calls.
    calls.clear()
    second = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(output_dir), "provider": "ollama", "model": "gemma4:26b"}
    )
    assert second["status"] in {"ok", "partial"}
    assert calls == [], "resumed pairs must not re-invoke the model"
    # The restored edges still reach the catalog.
    edge_types = {json.loads(line)["edge_type"] for line in (catalog / "evidence_graph_edges.jsonl").read_text().splitlines() if line.strip()}
    assert "CONSISTENT_WITH" in edge_types


def test_pairing_stays_within_a_subject(tmp_path: Path) -> None:
    """Facts from papers on different materials must not be paired.

    On the 48-paper BT-7 set, pairing every same-type fact across the corpus
    produced 33 correctly-"unrelated" judgments and zero labelable edges: the
    model was being asked to relate a moment in one compound to a moment in
    another. Papers must share a subject first.
    """
    from tas_copilot_core.corpus.evidence_graph_proposer import _cross_paper_pairs

    def fact(fid, paper, ftype, text):
        return {"fact_id": fid, "paper_id": paper, "fact_type": ftype, "value": {"text": text}, "answer_status": "supported"}

    supported = [
        # Two papers on the same compound, plus one on something else.
        fact("f1", "paper-a", "material", "CaMn2Sb2"),
        fact("f2", "paper-a", "ordered_moment", "3.46 muB per Mn"),
        fact("f3", "paper-b", "material", "CaMn2Sb2 single crystals"),
        fact("f4", "paper-b", "ordered_moment", "about 3.4 muB"),
        fact("f5", "paper-c", "material", "Ho2Ti2O7"),
        fact("f6", "paper-c", "ordered_moment", "10 muB per Ho"),
    ]
    pairs, selection = _cross_paper_pairs(supported, max_pairs=50)
    paired = {(min(p["fact_a"]["paper_id"], p["fact_b"]["paper_id"]), max(p["fact_a"]["paper_id"], p["fact_b"]["paper_id"])) for p in pairs}
    assert ("paper-a", "paper-b") in paired, "same-material papers must pair"
    assert ("paper-a", "paper-c") not in paired, "different materials must not pair"
    assert ("paper-b", "paper-c") not in paired
    assert selection["skipped_different_subject"] > 0
    assert selection["subject_scoping"] == "shared_material_fact"

    # An explicit grouping (e.g. papers linked to the same experiment) overrides
    # material matching, so unrelated-looking formulas can still be compared.
    grouped, selection2 = _cross_paper_pairs(
        supported, max_pairs=50, paper_groups={"paper-a": {"expt-1"}, "paper-c": {"expt-1"}}
    )
    grouped_pairs = {(min(p["fact_a"]["paper_id"], p["fact_b"]["paper_id"]), max(p["fact_a"]["paper_id"], p["fact_b"]["paper_id"])) for p in grouped}
    assert ("paper-a", "paper-c") in grouped_pairs
    assert selection2["subject_scoping"] == "explicit_paper_groups"


def test_empty_relation_is_retried_not_coerced(tmp_path: Path, monkeypatch) -> None:
    """An empty relation means the model declined, not that it judged 'unrelated'."""
    from tas_copilot_core.corpus import evidence_graph_proposer as proposer_module

    catalog = _catalog_with_facts(tmp_path)
    attempts: list[bool] = []

    def flaky_relation_model(packet, **_kwargs):
        reminded = "relation_enum_reminder" in packet
        attempts.append(reminded)
        if not reminded:
            return {"pair_id": packet["pair_id"], "relation": "", "confidence": 0.5}
        return {
            "pair_id": packet["pair_id"],
            "relation": "consistent_with",
            "confidence": 0.8,
            "evidence_a": {"chunk_id": "chunk-a-001", "quote": "ordered moment of 3.46 muB per Mn"},
            "evidence_b": {"chunk_id": "chunk-b-001", "quote": "ordered moment of about 3.4 muB"},
            "rationale": "Both report a similar Mn moment.",
        }

    monkeypatch.setattr(proposer_module, "_call_relation_model", flaky_relation_model)
    report = run_evidence_graph_proposer(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "prop"), "provider": "ollama", "model": "gemma4:26b"}
    )
    assert any(attempts), "first attempt must happen"
    assert True in attempts, "a retry naming the enum must be issued"
    judged = [p for p in report["pair_reports"] if p.get("relation") == "consistent_with"]
    assert judged, "the retry's verdict must be used"
    assert any("empty_relation_on_first_attempt" in w for p in judged for w in (p.get("warnings") or []))


def test_pairing_ignores_measurement_conditions_and_review_paper_material_lists() -> None:
    """Measurement conditions are not comparable quantities, and a review that
    names many compounds must not share a subject with every paper that
    studies one of them."""
    from tas_copilot_core.corpus.evidence_graph_proposer import PAIRABLE_FACT_TYPES, _cross_paper_pairs, _material_keys_by_paper

    assert "temperature_range" not in PAIRABLE_FACT_TYPES and "energy_setting" not in PAIRABLE_FACT_TYPES
    assert "ordered_moment" in PAIRABLE_FACT_TYPES

    def fact(fid, pid, ftype, text):
        return {"fact_id": fid, "paper_id": pid, "fact_type": ftype, "answer_status": "supported", "value": {"text": text}, "evidence": []}

    review = [fact(f"r{i}", "p-review", "material", m) for i, m in enumerate(["FeSe", "FeTe", "LaFeAsO", "BaFe2As2", "Fe1.01Se", "KFe2Se2"])]
    lafeaso = [fact("l1", "p-lafeaso", "material", "LaFeAsO"), fact("l2", "p-lafeaso", "ordered_moment", "0.36 muB")]
    other = [fact("o1", "p-fese", "material", "FeSe"), fact("o2", "p-fese", "ordered_moment", "none")]
    review_moment = [fact("r-m", "p-review", "ordered_moment", "0.3 muB")]
    titles = {"p-review": "A review of Fe-chalcogenide superconductors", "p-lafeaso": "Magnetic interactions in LaFeAsO", "p-fese": "Spin fluctuations in FeSe"}
    keys = _material_keys_by_paper(review + lafeaso + other, titles)
    assert "p-review" not in keys  # six compounds from facts, no formula in the title: pairs with nothing
    assert keys["p-lafeaso"] == {"lafeaso"} and keys["p-fese"] == {"fese"}

    pairs, _ = _cross_paper_pairs(review + review_moment + lafeaso + other, max_pairs=50, titles=titles)
    assert pairs == []  # LaFeAsO and FeSe share no subject; the review is scoped out
    same = [fact("s1", "p-fese2", "material", "FeSe"), fact("s2", "p-fese2", "ordered_moment", "0.1 muB")]
    titles["p-fese2"] = "Pressure dependence of magnetism in FeSe"
    pairs, _ = _cross_paper_pairs(other + same, max_pairs=50, titles=titles)
    assert [(p["fact_a"]["paper_id"], p["fact_b"]["paper_id"]) for p in pairs] == [("p-fese", "p-fese2")]
