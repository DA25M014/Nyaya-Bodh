#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f artifacts/faiss.index ]; then
  echo "FAISS index not found. Running setup first..."
  bash scripts/setup.sh
fi

python -m app.app
