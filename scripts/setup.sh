#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "Missing .env. Copy .env.example to .env and set LLAMA_MODEL_PATH." >&2
  exit 1
fi

echo "[1/2] Ingesting BNS sections, schemes, and IPC mapping..."
python -m src.ingestion

echo "[2/2] Building FAISS index from chunks..."
python -c "from src.vector_store import build_index_from_artifacts; build_index_from_artifacts(); print('Index built.')"

echo "Setup complete. Run: python -m app.app"
