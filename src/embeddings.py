"""Embeddings client backed by sentence-transformers.

Default model: paraphrase-multilingual-MiniLM-L12-v2 (470 MB, 384 dims,
supports Hindi + English + ~50 other languages, runs on CPU).

Embeddings are L2-normalised so cosine similarity equals inner product.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from config import CONFIG


_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL

    from sentence_transformers import SentenceTransformer

    cache_dir = str(CONFIG.st_cache_dir) if CONFIG.st_cache_dir else None
    print(f"[embeddings] Loading sentence-transformers model: {CONFIG.st_model_name}")
    _ST_MODEL = SentenceTransformer(CONFIG.st_model_name, cache_folder=cache_dir)
    print("[embeddings] Model loaded.")
    return _ST_MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, embedding_dim()), dtype=np.float32)

    model = _get_st_model()
    vectors = model.encode(
        list(texts),
        batch_size=32,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    arr = vectors.astype(np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    model = _get_st_model()
    return int(model.get_sentence_embedding_dimension())
