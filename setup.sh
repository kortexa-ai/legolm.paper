#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but was not found on PATH" >&2
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "[setup] syncing .venv from uv.lock with Python ${PYTHON_VERSION}"
if ! uv sync --frozen --python "${PYTHON_VERSION}"; then
  echo "[setup] existing .venv looked cursed, rebuilding it"
  rm -rf "$ROOT_DIR/.venv"
  uv sync --frozen --python "${PYTHON_VERSION}"
fi

echo "[setup] bootstrapping ~/.cache/autoresearch"
"$ROOT_DIR/.venv/bin/python" -m src.bootstrap_setup "$@"

echo "[setup] done"
