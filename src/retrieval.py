"""Retrieval orchestration: query -> embedding -> top-k chunks."""
from __future__ import annotations

from src.embeddings import embed_query
from src.vector_store import RetrievedChunk, VectorStore


def retrieve(store: VectorStore, query: str, top_k: int) -> list[RetrievedChunk]:
    if not query.strip():
        return []
    query_vector = embed_query(query)
    return store.search(query_vector, top_k)
