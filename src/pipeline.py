"""Top-level orchestration: question -> answer + citations + schemes."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import mlflow

from config import CONFIG
from src.llm import chat
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval import retrieve
from src.scheme_finder import SchemeMatch, find_schemes
from src.vector_store import RetrievedChunk, VectorStore, load_index


@dataclass
class QAResponse:
    question: str
    answer: str
    citations: list[RetrievedChunk]
    schemes: list[SchemeMatch]
    latency_ms: int


class NyayaBodhPipeline:
    def __init__(self, store: VectorStore | None = None):
        self._store = store or load_index()
        try:
            mlflow.set_experiment(CONFIG.mlflow_experiment_name)
            self._mlflow_enabled = True
        except Exception:
            self._mlflow_enabled = False

    def answer(self, question: str, include_schemes: bool = True) -> QAResponse:
        start = time.perf_counter()

        retrieved = retrieve(self._store, question, CONFIG.retrieval_top_k)
        user_prompt = build_user_prompt(question, retrieved)
        answer_text = chat(SYSTEM_PROMPT, user_prompt)

        schemes: list[SchemeMatch] = []
        if include_schemes:
            try:
                schemes = find_schemes(question, top_k=3)
            except FileNotFoundError:
                schemes = []

        latency_ms = int((time.perf_counter() - start) * 1000)

        response = QAResponse(
            question=question,
            answer=answer_text,
            citations=retrieved,
            schemes=schemes,
            latency_ms=latency_ms,
        )
        self._log_to_mlflow(response)
        return response

    def _log_to_mlflow(self, response: QAResponse) -> None:
        if not self._mlflow_enabled:
            return
        try:
            with mlflow.start_run(nested=False):
                mlflow.log_params(
                    {
                        "chat_model": CONFIG.chat_model_id,
                        "embed_model": CONFIG.st_model_name,
                        "top_k": CONFIG.retrieval_top_k,
                        "n_ctx": CONFIG.llama_n_ctx,
                    }
                )
                mlflow.log_metrics(
                    {
                        "latency_ms": response.latency_ms,
                        "n_citations": len(response.citations),
                        "n_schemes": len(response.schemes),
                    }
                )
                mlflow.log_dict(
                    {
                        "question": response.question,
                        "answer": response.answer,
                        "citations": [asdict(c) for c in response.citations],
                        "schemes": [asdict(s) for s in response.schemes],
                    },
                    "trace.json",
                )
        except Exception:
            pass
