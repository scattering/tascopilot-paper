from __future__ import annotations
import math
import os
from collections.abc import Iterable
from tas_copilot_contracts import ProvenanceRecord, RetrievalResult, SourceChunk
from tas_copilot_core.llm_router import openai_compatible_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorRetriever:
    """Dense semantic retriever backed by the NIST vLLM/rchat embeddings endpoint.

    Reads endpoint config from environment variables (matching llm_endpoints.py):
      - NIST_VLLM_BASE_URL
      - NIST_VLLM_EMBEDDING_MODEL   (separate from the chat model)
      - NIST_VLLM_API_KEY           (optional)

    Embeddings for the corpus are computed once at construction; the query is
    embedded per search() call.
    """

    def __init__(
        self,
        chunks: Iterable[SourceChunk],
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.chunks = list(chunks)
        self.base_url = base_url or os.environ.get("NIST_VLLM_BASE_URL", "")
        self.model = model or os.environ.get("NIST_VLLM_EMBEDDING_MODEL", "")
        self.api_key = api_key or os.environ.get("NIST_VLLM_API_KEY")
        self.batch_size = batch_size
        self._embeddings: list[list[float]] = []
        self._enabled = bool(self.base_url and self.model and self.chunks)

        if self._enabled:
            texts = [c.text for c in self.chunks]
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                self._embeddings.extend(
                    openai_compatible_embeddings(
                        self.base_url, self.model, batch, api_key=self.api_key
                    )
                )

    def search(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        # Gracefully no-op if not configured — keyword search still works.
        if not self._enabled or not self._embeddings:
            return []
        q_vec = openai_compatible_embeddings(
            self.base_url, self.model, [query], api_key=self.api_key
        )
        if not q_vec:
            return []
        q = q_vec[0]

        scored = [
            (_cosine(q, emb), chunk)
            for emb, chunk in zip(self._embeddings, self.chunks)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievalResult] = []
        for score, chunk in scored[:limit]:
            if score <= 0.0:
                continue
            results.append(
                RetrievalResult(
                    source_type=chunk.source_type,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    score=float(score),  # cosine in [-1, 1], practically [0, 1]
                    matched_terms=[],
                    snippet=chunk.text[:300],
                    retrieval_method="vector_rchat_embeddings",
                    provenance=chunk.provenance
                    or [ProvenanceRecord(source_type=chunk.source_type, source_id=chunk.source_id, method="vector")],
                )
            )
        return results