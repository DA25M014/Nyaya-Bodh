"""Chat LLM client backed by llama-cpp-python (GGUF models).

The model is loaded once and cached as a module-level singleton because
loading a 24B Q4 quant takes 30 to 60 seconds and 14 to 16 GB of RAM.
"""
from __future__ import annotations

import re

from config import CONFIG


_LLAMA = None


def _get_llama():
    global _LLAMA
    if _LLAMA is not None:
        return _LLAMA

    if CONFIG.llama_model_path is None:
        raise RuntimeError(
            "LLAMA_MODEL_PATH is not set. Add it to .env, e.g. "
            "LLAMA_MODEL_PATH=C:/models/sarvam-m.Q4_K_M.gguf"
        )
    if not CONFIG.llama_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {CONFIG.llama_model_path}. "
            "Check the path in .env."
        )

    from llama_cpp import Llama

    print(f"[llm] Loading GGUF model: {CONFIG.llama_model_path}")
    print(
        f"[llm] n_ctx={CONFIG.llama_n_ctx}  n_threads={CONFIG.llama_n_threads}  "
        f"n_gpu_layers={CONFIG.llama_n_gpu_layers}"
    )
    print("[llm] First load takes 30 to 60 seconds for a 24B Q4 model.")

    _LLAMA = Llama(
        model_path=str(CONFIG.llama_model_path),
        n_ctx=CONFIG.llama_n_ctx,
        n_threads=CONFIG.llama_n_threads,
        n_gpu_layers=CONFIG.llama_n_gpu_layers,
        verbose=False,
    )
    print("[llm] Model loaded.")
    return _LLAMA




_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Strip reasoning preambles from sarvam-m output, conservatively.

    Only strips when there are clear reasoning markers in the early text.
    Never strips so aggressively that the actual answer disappears.
    """
    text = text.strip()

    # Strategy 1: explicit <think>...</think> tags
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        return text[idx + len("</think>"):].strip()

    # Strategy 2: detect reasoning preamble by telltale phrases at the start.
    reasoning_starters = [
        "okay,", "okay ", "let me", "i need to", "first,", "first ",
        "looking at", "the user", "i should", "i'll", "i will",
        "to answer", "based on the context", "checking the",
    ]
    first_chunk = text[:200].lower()
    has_reasoning_start = any(starter in first_chunk for starter in reasoning_starters)

    if not has_reasoning_start:
        return text

    # Reasoning detected. Find boundary: paragraph break followed by an answer marker.
    answer_patterns = [
        r"\n\s*\n\s*\[BNS",
        r"\n\s*\n\s*[\u0900-\u097F]",
        r"\n\s*\n\s*(?:Answer|Response|उत्तर|जवाब):",
        r"\n\s*\n\s*According to",
        r"\n\s*\n\s*Under (?:Section|BNS)",
    ]

    earliest = None
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()

    if earliest is not None and earliest > 50:
        return text[earliest:].strip()

    # Couldn't confidently locate the boundary - return as-is rather than risk
    # eating the answer.
    return text

def chat(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    llm = _get_llama()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=CONFIG.llama_max_tokens,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    return _strip_reasoning(content)


def warmup() -> None:
    """Pre-load the model. Call once at app startup to avoid cold-start latency."""
    _get_llama()
