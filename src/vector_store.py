"""FAISS-backed vector store. Built once during setup, queried at request time."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config import CONFIG
from src.embeddings import embed_texts


@dataclass
class RetrievedChunk:
    chunk_id: str
    section_id: str
    title: str
    text: str
    score: float


class VectorStore:
    """Inner-product index over L2-normalised embeddings (so scores are cosine)."""

    def __init__(self, index: faiss.Index, metadata: pd.DataFrame):
        self._index = index
        self._metadata = metadata.reset_index(drop=True)

    @classmethod
    def build(cls, chunks_df: pd.DataFrame) -> "VectorStore":
        if chunks_df.empty:
            raise ValueError("chunks_df is empty; nothing to index.")

        embeddings = embed_texts(chunks_df["text"].tolist())
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, chunks_df)

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        meta_path = index_path.with_suffix(".meta.parquet")
        self._metadata.to_parquet(meta_path, index=False)
        manifest = {
            "dim": self._index.d,
            "ntotal": self._index.ntotal,
            "metadata_columns": list(self._metadata.columns),
        }
        index_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, index_path: Path) -> "VectorStore":
        index = faiss.read_index(str(index_path))
        metadata = pd.read_parquet(index_path.with_suffix(".meta.parquet"))
        return cls(index, metadata)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, indices = self._index.search(query_vector.astype(np.float32), top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self._metadata.iloc[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    section_id=str(row["section_id"]),
                    title=str(row.get("title", "")),
                    text=str(row["text"]),
                    score=float(score),
                )
            )
        return results


def build_index_from_artifacts() -> VectorStore:
    chunks_df = pd.read_parquet(CONFIG.chunks_parquet_path)
    store = VectorStore.build(chunks_df)
    store.save(CONFIG.faiss_index_path)
    return store


def load_index() -> VectorStore:
    return VectorStore.load(CONFIG.faiss_index_path)
