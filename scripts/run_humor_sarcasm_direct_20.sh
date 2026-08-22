#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEC_PATH="runs/humor_lls_sarcasm_direct_20/spec.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY must be set for GPT-4o, Claude, and Gemini." >&2
    exit 1
fi

# Keep large model and dataset downloads on the persistent RunPod volume.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

"${PYTHON_BIN}" scripts/prepare_airiskdilemmas.py \
    --output data/scenarios/airiskdilemmas.json

echo "Planned call counts:"
"${PYTHON_BIN}" scripts/run.py "${SPEC_PATH}" --estimate-calls

if [[ "${ESTIMATE_ONLY:-0}" == "1" ]]; then
    echo "ESTIMATE_ONLY=1; collection was not started."
    exit 0
fi

"${PYTHON_BIN}" scripts/run.py "${SPEC_PATH}"
