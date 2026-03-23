#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/minoo/src/MineContext-vertex"
VENV_BIN="${REPO_ROOT}/.venv/bin"

if [[ ! -x "${VENV_BIN}/python" ]]; then
  echo "Patched MineContext venv is missing at ${VENV_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VENV_BIN}/python" -m opencontext.cli "$@"
