from __future__ import annotations
import math
import re
from collections import Counter
from collections.abc import Iterable
from tas_copilot_contracts import ProvenanceRecord, RetrievalResult, SourceChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_.+-]+")

# Common English stopwords + low-signal query words
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "of", "to", "in", "on", "at", "by", "for", "with", "as",
    "its", "it", "this", "that", "these", "those", "what", "which", "who",
    "how", "why", "when", "where", "general", "properties", "property",
})


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) >= 2]


class KeywordRetriever:
    def __init__(self, chunks: Iterable[SourceChunk]) -> None:
        self.chunks = list(chunks)
        self._doc_freq = self._compute_doc_freq()

    def _compute_doc_freq(self) -> Counter:
        df: Counter = Counter()
        for chunk in self.chunks:
            for term in set(content_terms(chunk.text)):
                df[term] += 1
        return df

    def _idf(self, term: str) -> float:
        n = len(self.chunks) or 1
        # Smoothed IDF: rare terms (like a chemical formula) score much higher
        return math.log((n + 1) / (self._doc_freq.get(term, 0) + 1)) + 1.0

    def search(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        query_terms = set(content_terms(query))
        if not query_terms:
            return []  # no meaningful query terms -> return nothing, not junk

        results = []
        for chunk in self.chunks:
            text_counts = Counter(content_terms(chunk.text))
            matched = sorted(query_terms & set(text_counts))
            if not matched:
                continue
            # TF-IDF style score: rewards rare, meaningful matches
            score = sum(
                (1 + math.log(text_counts[term])) * self._idf(term)
                for term in matched
            )
            results.append(
                RetrievalResult(
                    source_type=chunk.source_type,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    score=score,
                    matched_terms=matched,
                    snippet=chunk.text[:300],
                    retrieval_method="keyword_tfidf",
                    provenance=chunk.provenance
                    or [ProvenanceRecord(source_type=chunk.source_type, source_id=chunk.source_id, method="keyword")],
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)[:limit]