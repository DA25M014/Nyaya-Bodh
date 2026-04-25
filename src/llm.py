"""Chat LLM client backed by llama-cpp-python (GGUF models).

The model is loaded once and cached as a module-level singleton because
loading a 24B Q4 quant takes 30 to 60 seconds and 14 to 16 GB of RAM.
"""
from __future__ import annotations

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
    return content.strip()


def warmup() -> None:
    """Pre-load the model. Call once at app startup to avoid cold-start latency."""
    _get_llama()
