#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

if [[ $# -eq 0 ]]; then
  set -- --suite all
fi

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  ./setup.sh
fi

exec uv run paper-reproduce "$@"
