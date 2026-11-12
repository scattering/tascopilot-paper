from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord, RetrievalResult
from tas_copilot_core.retrieval.keyword import tokenize


@dataclass
class _GraphHit:
    edge: dict[str, Any]
    source: dict[str, Any]
    target: dict[str, Any]
    matched_terms: set[str] = field(default_factory=set)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class GraphRetriever:
    """Deterministic retrieval over typed evidence graph nodes and edges.

    The evidence graph is an interpretation layer over reviewed catalog records,
    not an LLM-generated graph. Retrieval seeds on query matches in node labels,
    source chunk text/metadata, and edge evidence, then expands one hop through
    explicit edges while preserving edge status, confidence, and provenance.
    """

    def __init__(self, catalog_dir: str | Path) -> None:
        self.catalog_dir = Path(catalog_dir)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._source_chunks: dict[str, dict[str, Any]] = {}
        self._load_graph()

    def _load_graph(self) -> None:
        """Load graph records if present; invalid catalogs degrade to no hits."""
        try:
            self._nodes = {
                str(node.get("node_id")): node
                for node in _load_jsonl(self.catalog_dir / "evidence_graph_nodes.jsonl")
                if node.get("node_id")
            }
            self._edges = _load_jsonl(self.catalog_dir / "evidence_graph_edges.jsonl")
            self._source_chunks = {
                str(chunk.get("chunk_id") or chunk.get("section_id")): chunk
                for chunk in [
                    *_load_jsonl(self.catalog_dir / "source_chunks.jsonl"),
                    *_textbook_sections_as_chunks(self.catalog_dir / "textbook_sections.jsonl"),
                ]
                if chunk.get("chunk_id") or chunk.get("section_id")
            }
        except Exception:
            self._nodes = {}
            self._edges = []
            self._source_chunks = {}

    def search(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        """Return graph edges and one-hop paths relevant to the query."""
        if not self._edges:
            return []

        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        seed_nodes: dict[str, tuple[set[str], float, list[str]]] = {}
        direct_edges: dict[str, tuple[set[str], float, list[str]]] = {}

        for node_id, node in self._nodes.items():
            matched, weight = _weighted_match(query_terms, self._node_text(node), field_weight=1.0)
            if matched:
                seed_nodes[node_id] = (matched, weight, ["node_match"])

        for edge in self._edges:
            matched, weight = _weighted_match(query_terms, self._edge_text(edge), field_weight=1.25)
            if matched:
                direct_edges[str(edge.get("edge_id"))] = (matched, weight, ["edge_match"])

        hits: dict[str, _GraphHit] = {}
        for edge in self._edges:
            edge_id = str(edge.get("edge_id") or "")
            source_id = str(edge.get("source_node_id") or "")
            target_id = str(edge.get("target_node_id") or "")
            source = self._nodes.get(source_id, {"node_id": source_id, "node_type": _node_type(source_id), "label": source_id})
            target = self._nodes.get(target_id, {"node_id": target_id, "node_type": _node_type(target_id), "label": target_id})

            matched_terms: set[str] = set()
            score = 0.0
            reasons: list[str] = []

            if edge_id in direct_edges:
                matched, weight, direct_reasons = direct_edges[edge_id]
                matched_terms.update(matched)
                score += weight
                reasons.extend(direct_reasons)

            for node_id, side in [(source_id, "source_node_match"), (target_id, "target_node_match")]:
                if node_id in seed_nodes:
                    matched, weight, _ = seed_nodes[node_id]
                    matched_terms.update(matched)
                    score += 0.7 * weight
                    reasons.append(side)

            if not matched_terms:
                continue

            confidence = _confidence(edge)
            score += confidence * _status_weight(str(edge.get("status") or "proposed"))
            hit = hits.setdefault(edge_id, _GraphHit(edge=edge, source=source, target=target))
            hit.matched_terms.update(matched_terms)
            hit.score += score
            hit.reasons.extend(reason for reason in reasons if reason not in hit.reasons)

        results: list[RetrievalResult] = []
        for hit in hits.values():
            edge = hit.edge
            source = hit.source
            target = hit.target
            source_label = source.get("label") or source.get("source_object_id") or source.get("node_id")
            target_label = target.get("label") or target.get("source_object_id") or target.get("node_id")
            snippet = f"{source_label} -[{edge.get('edge_type')}]-> {target_label}"
            provenance = [ProvenanceRecord(**item) for item in edge.get("provenance", [])]
            if not provenance:
                provenance = [
                    ProvenanceRecord(
                        source_type="evidence_graph_edge",
                        source_id=str(edge.get("edge_id")),
                        method="typed_graph_traversal",
                    )
                ]
            source_object_id = target.get("source_object_id") or _node_object_id(str(target.get("node_id") or ""))
            summary = {
                "edge_type": edge.get("edge_type"),
                "status": edge.get("status"),
                "confidence": _confidence(edge),
                "source_node": _node_summary(source),
                "target_node": _node_summary(target),
                "path": [
                    str(source.get("node_id") or ""),
                    str(edge.get("edge_type") or ""),
                    str(target.get("node_id") or ""),
                ],
                "evidence": edge.get("evidence", []),
                "match_reasons": sorted(set(hit.reasons)),
                "source_object_id": source_object_id,
                "source_object_type": target.get("node_type") or _node_type(str(target.get("node_id") or "")),
            }
            chunk_type = _chunk_type_for_node(target, self._source_chunks) or _chunk_type_for_node(source, self._source_chunks)
            if chunk_type:
                summary["chunk_type"] = chunk_type
            result_source_type, result_source_id = _result_identity(edge, source, target)
            confidence = _confidence(edge)
            result_uncertainty = max(0.0, min(1.0, 1.0 - confidence))
            if str(edge.get("status") or "").lower() in {"candidate", "proposed"}:
                result_uncertainty = max(result_uncertainty, 0.35)
            if str(edge.get("status") or "").lower() == "rejected":
                result_uncertainty = max(result_uncertainty, 0.75)
            results.append(
                RetrievalResult(
                    source_type=result_source_type,
                    source_id=result_source_id,
                    title=str(edge.get("edge_type") or ""),
                    score=hit.score,
                    matched_terms=sorted(hit.matched_terms),
                    snippet=snippet[:300],
                    structured_summary=summary,
                    provenance=provenance,
                    retrieval_method="typed_graph_traversal",
                    confidence=confidence,
                    uncertainty=result_uncertainty,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _node_text(self, node: dict[str, Any]) -> str:
        parts = [
            str(node.get("node_id") or ""),
            str(node.get("node_type") or ""),
            str(node.get("label") or ""),
            str(node.get("source_object_id") or ""),
            json.dumps(node.get("metadata") or {}, sort_keys=True),
        ]
        source_object_id = str(node.get("source_object_id") or _node_object_id(str(node.get("node_id") or "")))
        chunk = self._source_chunks.get(source_object_id)
        if chunk:
            parts.extend(
                [
                    str(chunk.get("title") or ""),
                    str(chunk.get("chunk_type") or ""),
                    str(chunk.get("text") or ""),
                    json.dumps(chunk.get("metadata") or {}, sort_keys=True),
                ]
            )
        return " ".join(parts)

    def _edge_text(self, edge: dict[str, Any]) -> str:
        return " ".join(
            [
                str(edge.get("edge_id") or ""),
                str(edge.get("edge_type") or ""),
                str(edge.get("source_node_id") or ""),
                str(edge.get("target_node_id") or ""),
                " ".join(str(item) for item in edge.get("evidence", [])),
                str(edge.get("status") or ""),
            ]
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _textbook_sections_as_chunks(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": section.get("section_id"),
            "source_type": "textbook",
            "source_id": section.get("section_id"),
            "chunk_type": "textbook_explanation",
            "title": section.get("title"),
            "text": section.get("text", ""),
            "metadata": {
                "book_title": section.get("book_title"),
                "section_path": section.get("section_path"),
            },
            "provenance": section.get("provenance", []),
        }
        for section in _load_jsonl(path)
    ]


def _weighted_match(query_terms: set[str], text: str, *, field_weight: float) -> tuple[set[str], float]:
    text_terms = _graph_tokens(text)
    matched = query_terms & text_terms
    if not matched:
        return set(), 0.0
    return matched, field_weight * (len(matched) / max(len(query_terms), 1))


def _graph_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in tokenize(text):
        tokens.add(token)
        tokens.update(part for part in re.split(r"[^a-z0-9]+", token.lower()) if part)
    return tokens


def _confidence(edge: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(edge.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _status_weight(status: str) -> float:
    return {
        "accepted": 1.0,
        "reviewed": 0.9,
        "supported": 0.9,
        "candidate": 0.65,
        "proposed": 0.5,
        "rejected": 0.15,
        "deprecated": 0.05,
    }.get(status.lower(), 0.5)


def _node_type(node_id: str) -> str:
    return node_id.split(":", 1)[0] if ":" in node_id else "Unknown"


def _node_object_id(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type") or _node_type(str(node.get("node_id") or "")),
        "label": node.get("label"),
        "source_object_id": node.get("source_object_id"),
    }


def _chunk_type_for_node(node: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> str | None:
    source_object_id = str(node.get("source_object_id") or _node_object_id(str(node.get("node_id") or "")))
    chunk = chunks.get(source_object_id)
    if not chunk:
        return None
    return str(chunk.get("chunk_type") or (chunk.get("metadata") or {}).get("chunk_type") or "") or None


def _result_identity(edge: dict[str, Any], source: dict[str, Any], target: dict[str, Any]) -> tuple[str, str]:
    edge_type = str(edge.get("edge_type") or "").lower()
    status = str(edge.get("status") or "").lower()
    if edge_type.endswith("_source_for_panel") or edge_type in {
        "candidate_source_for_panel",
        "accepted_source_for_panel",
        "rejected_as_source_for_panel",
    }:
        return "evidence_graph_edge", str(edge.get("edge_id"))
    if status in {"candidate", "proposed", "rejected"} and edge_type != "has_chunk":
        return "evidence_graph_edge", str(edge.get("edge_id"))
    node_type = str(target.get("node_type") or _node_type(str(target.get("node_id") or "")))
    object_id = str(target.get("source_object_id") or _node_object_id(str(target.get("node_id") or "")))
    source_type = {
        "SourceChunk": "source_chunk",
        "Paper": "paper",
        "TextbookSection": "textbook",
        "Experiment": "experiment",
        "Scan": "scan",
        "DataFile": "datafile",
    }.get(node_type, "evidence_graph_node")
    return source_type, object_id
