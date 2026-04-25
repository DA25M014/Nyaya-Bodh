"""Token-aware text chunker. Wrapped as a Spark UDF in `ingestion.py`.

Uses tiktoken when available; falls back to a character-based heuristic
so the pipeline still works when the tiktoken BPE file cannot be downloaded.
"""
from __future__ import annotations

from typing import Iterable

_ENCODER = None
_TIKTOKEN_OK: bool | None = None
_CHARS_PER_TOKEN = 4


def _get_encoder():
    global _ENCODER, _TIKTOKEN_OK
    if _TIKTOKEN_OK is False:
        return None
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_OK = True
        return _ENCODER
    except Exception:
        _TIKTOKEN_OK = False
        return None


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    if not text:
        return []
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk_tokens")

    encoder = _get_encoder()
    if encoder is not None:
        token_ids = encoder.encode(text)
        if not token_ids:
            return []
        step = chunk_tokens - overlap
        chunks: list[str] = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + chunk_tokens]
            if not window:
                break
            chunks.append(encoder.decode(window))
            if start + chunk_tokens >= len(token_ids):
                break
        return chunks

    char_window = chunk_tokens * _CHARS_PER_TOKEN
    char_step = (chunk_tokens - overlap) * _CHARS_PER_TOKEN
    chunks_chars: list[str] = []
    for start in range(0, len(text), char_step):
        window = text[start : start + char_window]
        if not window:
            break
        chunks_chars.append(window)
        if start + char_window >= len(text):
            break
    return chunks_chars


def chunk_records(
    records: Iterable[tuple[str, str]],
    chunk_tokens: int,
    overlap: int,
) -> list[dict]:
    """Turn (section_id, text) pairs into one row per chunk."""
    out: list[dict] = []
    for section_id, text in records:
        for idx, chunk in enumerate(chunk_text(text, chunk_tokens, overlap)):
            out.append(
                {
                    "section_id": section_id,
                    "chunk_id": f"{section_id}::chunk_{idx:04d}",
                    "chunk_index": idx,
                    "text": chunk,
                }
            )
    return out
