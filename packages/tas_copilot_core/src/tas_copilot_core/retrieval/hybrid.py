from __future__ import annotations
from tas_copilot_contracts import RetrievalResult, TASConcept
from tas_copilot_core.ontology.linker import match_concepts


def _normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Min-max normalize scores to [0, 1] so retrievers on different scales
    (unbounded TF-IDF vs. cosine in [0,1]) can be fused fairly."""
    if not results:
        return results
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for r in results:
        r.score = (r.score - lo) / span if span > 0 else 1.0
    return results


class HybridRetriever:
    def __init__(
        self,
        keyword_retriever,
        vector_retriever=None,
        graph_retriever=None,
        concepts: list[TASConcept] | None = None,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        graph_weight: float = 0.3,
    ) -> None:
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.concepts = concepts or []
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def search(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        by_key: dict[tuple[str, str], RetrievalResult] = {}

        # --- Keyword branch (normalized, then weighted) ---
        kw = _normalize_scores(self.keyword_retriever.search(query, limit=limit * 2))
        for result in kw:
            result.score *= self.keyword_weight
            by_key[(result.source_type, result.source_id)] = result

        # --- Vector branch (normalized, then weighted) ---
        if self.vector_retriever:
            vec = _normalize_scores(self.vector_retriever.search(query, limit=limit * 2))
            for result in vec:
                key = (result.source_type, result.source_id)
                weighted = self.vector_weight * result.score
                if key in by_key:
                    by_key[key].score += weighted
                else:
                    result.score = weighted
                    by_key[key] = result

        # --- Graph branch (normalized, then weighted) ---
        if self.graph_retriever:
            gr = _normalize_scores(self.graph_retriever.search(query, limit=limit * 2))
            for result in gr:
                key = (result.source_type, result.source_id)
                weighted = self.graph_weight * result.score
                if key in by_key:
                    by_key[key].score += weighted
                else:
                    result.score = weighted
                    by_key[key] = result

        # --- Concept boost (ontology match) ---
        query_concepts = match_concepts(query, self.concepts)
        concept_ids = sorted({c.id for c in query_concepts})
        for result in by_key.values():
            result.matched_concepts = concept_ids
            result.score += 0.2 * len(concept_ids)
            result.retrieval_method = (
                "hybrid_normalized_graph" if self.graph_retriever else "hybrid_normalized"
            )

        return sorted(by_key.values(), key=lambda r: r.score, reverse=True)[:limit]