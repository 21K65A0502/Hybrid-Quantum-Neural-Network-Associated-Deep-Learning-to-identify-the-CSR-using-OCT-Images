#!/usr/bin/env bash
# HQNNDL v3.0  —  Launch Script (Linux / macOS)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON=""

echo ""
echo "══════════════════════════════════════════════════"
echo "  HQNNDL v3.0  |  Quantum CSR Detection"
echo "══════════════════════════════════════════════════"

for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PYTHON="$cmd"; echo "  Python : $($PYTHON --version)"; break
        fi
    fi
done

[[ -z "$PYTHON" ]] && { echo "  [ERROR] Python 3.9+ required."; exit 1; }

if [[ ! -d "$VENV_DIR" ]]; then
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"; PYTHON="python"
echo "  Checking dependencies..."
pip install -q -r "${PROJECT_DIR}/requirements.txt"

mkdir -p "${PROJECT_DIR}/uploads"
mkdir -p "${PROJECT_DIR}/results/reports"
mkdir -p "${PROJECT_DIR}/results/heatmaps"

echo ""; echo "  Starting server..."; echo "  Open: http://localhost:5050"
echo "══════════════════════════════════════════════════"; echo ""

cd "$BACKEND_DIR" && exec "$PYTHON" app.py
