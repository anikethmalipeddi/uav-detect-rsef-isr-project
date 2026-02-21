#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-poster}"
if [[ "${MODE}" != "poster" && "${MODE}" != "full" ]]; then
  echo "Usage: bash scripts/run_demo.sh [poster|full]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v streamlit >/dev/null 2>&1; then
  echo "streamlit not found. Activate demo environment first."
  exit 1
fi

export UAV_DEMO_MODE="${MODE}"
echo "Launching demo in mode: ${MODE}"
exec streamlit run demo_app/app.py --server.headless false

