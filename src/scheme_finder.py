"""Match a free-text query to relevant government schemes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG
from src.embeddings import embed_query, embed_texts


@dataclass
class SchemeMatch:
    scheme_id: str
    name: str
    ministry: str
    description: str
    eligibility: str
    url: str
    score: float


_CACHE: tuple[pd.DataFrame, np.ndarray] | None = None


def _load() -> tuple[pd.DataFrame, np.ndarray]:
    global _CACHE
    if _CACHE is None:
        df = pd.read_parquet(CONFIG.schemes_parquet_path).fillna("")
        text = (
            df["name"].astype(str)
            + ". "
            + df["description"].astype(str)
            + ". Eligibility: "
            + df["eligibility"].astype(str)
        )
        embeddings = embed_texts(text.tolist())
        _CACHE = (df, embeddings)
    return _CACHE


def find_schemes(query: str, top_k: int = 3, min_score: float = 0.25) -> list[SchemeMatch]:
    df, embeddings = _load()
    query_vec = embed_query(query)
    scores = embeddings @ query_vec
    order = np.argsort(-scores)[:top_k]

    matches: list[SchemeMatch] = []
    for idx in order:
        score = float(scores[int(idx)])
        if score < min_score:
            continue
        row = df.iloc[int(idx)]
        matches.append(
            SchemeMatch(
                scheme_id=str(row["scheme_id"]),
                name=str(row["name"]),
                ministry=str(row["ministry"]),
                description=str(row["description"]),
                eligibility=str(row["eligibility"]),
                url=str(row["url"]),
                score=score,
            )
        )
    return matches
