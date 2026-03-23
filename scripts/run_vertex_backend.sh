#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ROOT="${HOME}/.local/share/uv/python"
PYTHON_BIN="$(find "${PYTHON_ROOT}" -maxdepth 4 -path '*cpython-3.10*/bin/python3.10' | head -n 1)"
VENV_DIR="${REPO_ROOT}/.venv"
APP_CONFIG="/Applications/MineContext.app/Contents/Resources/backend/config/config.yaml"
CONTEXT_PATH_DEFAULT="${HOME}/Library/Application Support/MineContext/Data"

PROJECT_ID="${PROJECT_ID:-}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"
CONTEXT_PATH="${CONTEXT_PATH:-${CONTEXT_PATH_DEFAULT}}"
VERTEX_OPENAI_BASE_URL="https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${VERTEX_LOCATION}/endpoints/openapi"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.10 from uv was not found under ${PYTHON_ROOT}." >&2
  echo "Run: uv python install 3.10" >&2
  exit 1
fi

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Set PROJECT_ID before running this script." >&2
  echo "Example: PROJECT_ID=my-gcp-project ${BASH_SOURCE[0]}" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required for ADC-backed Vertex auth." >&2
  exit 1
fi

if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "ADC is not configured. Run: gcloud auth application-default login" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -U pip >/dev/null
python -m pip install -e "${REPO_ROOT}"

cat <<EOF
Backend starting with:
  PROJECT_ID=${PROJECT_ID}
  VERTEX_LOCATION=${VERTEX_LOCATION}
  CONTEXT_PATH=${CONTEXT_PATH}

MineContext Custom settings:
  VLM Base URL: ${VERTEX_OPENAI_BASE_URL}
  VLM Model:    gemini-2.5-flash
  Emb Base URL: ${VERTEX_OPENAI_BASE_URL}
  Emb Model:    gemini-embedding-001
  API Keys:     leave blank; backend will use ADC refresh
EOF

export CONTEXT_PATH

exec opencontext start --config "${APP_CONFIG}" --host 127.0.0.1 --port 1733
