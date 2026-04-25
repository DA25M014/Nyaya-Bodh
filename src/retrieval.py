"""Retrieval orchestration: query -> embedding -> top-k chunks."""
from __future__ import annotations

from src.embeddings import embed_query
from src.vector_store import RetrievedChunk, VectorStore


def retrieve(
    store: VectorStore,
    query: str,
    top_k: int,
    min_score: float = 0.55,
) -> list[RetrievedChunk]:
    if not query.strip():
        return []
    query_vector = embed_query(query)
    results = store.search(query_vector, top_k)
    filtered = [r for r in results if r.score >= min_score]
    # Always return at least the top hit so the answer panel has something to cite,
    # even if all scores are below the threshold.
    if not filtered and results:
        filtered = [results[0]]
    return filtered
