#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY_CMD="python3"
if ! command -v "${PY_CMD}" >/dev/null 2>&1; then
  PY_CMD="python"
fi

if ! command -v "${PY_CMD}" >/dev/null 2>&1; then
  echo "Python not found. Install Python/Conda and rerun."
  exit 1
fi

exec "${PY_CMD}" tools/verify_demo_cross_platform.py

