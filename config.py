"""Central configuration loaded from environment variables.

The project runs fully locally:
- LLM: sarvam-m (or any GGUF) via llama-cpp-python.
- Embeddings: sentence-transformers (multilingual MiniLM by default).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_optional_int(key: str) -> int | None:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return None
    return int(raw)


@dataclass(frozen=True)
class Config:
    chunk_tokens: int
    chunk_overlap: int
    retrieval_top_k: int
    artifacts_dir: Path
    mlflow_experiment_name: str

    llama_model_path: Path | None
    llama_n_ctx: int
    llama_n_threads: int | None
    llama_n_gpu_layers: int
    llama_max_tokens: int

    st_model_name: str
    st_cache_dir: Path | None

    @property
    def faiss_index_path(self) -> Path:
        return self.artifacts_dir / "faiss.index"

    @property
    def chunks_parquet_path(self) -> Path:
        return self.artifacts_dir / "bns_chunks.parquet"

    @property
    def schemes_parquet_path(self) -> Path:
        return self.artifacts_dir / "schemes.parquet"

    @property
    def ipc_bns_parquet_path(self) -> Path:
        return self.artifacts_dir / "ipc_bns_map.parquet"

    @property
    def chat_model_id(self) -> str:
        if self.llama_model_path is None:
            return "unknown"
        return self.llama_model_path.name


def load_config() -> Config:
    artifacts_dir = Path(_env("ARTIFACTS_DIR", "./artifacts")).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    llama_model_path_raw = _env("LLAMA_MODEL_PATH", "")
    llama_model_path = Path(llama_model_path_raw).resolve() if llama_model_path_raw else None

    st_cache_dir_raw = _env("ST_CACHE_DIR", "")
    st_cache_dir = Path(st_cache_dir_raw).resolve() if st_cache_dir_raw else None

    return Config(
        chunk_tokens=_env_int("CHUNK_TOKENS", 512),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 64),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
        artifacts_dir=artifacts_dir,
        mlflow_experiment_name=_env("MLFLOW_EXPERIMENT_NAME", "nyaya-bodh"),
        llama_model_path=llama_model_path,
        llama_n_ctx=_env_int("LLAMA_N_CTX", 4096),
        llama_n_threads=_env_optional_int("LLAMA_N_THREADS"),
        llama_n_gpu_layers=_env_int("LLAMA_N_GPU_LAYERS", 0),
        llama_max_tokens=_env_int("LLAMA_MAX_TOKENS", 512),
        st_model_name=_env("ST_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"),
        st_cache_dir=st_cache_dir,
    )


CONFIG = load_config()
