#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the Python environment inside the devcontainer.
# Intended to be run manually, or triggered from postCreateCommand.

repo_root="$(pwd)"
if command -v git >/dev/null 2>&1; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv is not available in the dev container. Check devcontainer features." >&2
  exit 1
fi

if [[ -z "${MAX_JOBS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    MAX_JOBS="$(nproc)"
  else
    MAX_JOBS=4
  fi
fi
export MAX_JOBS

echo "[bootstrap] MAX_JOBS=$MAX_JOBS"

# If .venv exists but is broken, optionally reset it.
if [[ -d .venv && ! -x .venv/bin/python ]]; then
  if [[ "${ASYMDSD_RESET_VENV:-}" == "1" ]]; then
    echo "[bootstrap] Existing .venv looks invalid; ASYMDSD_RESET_VENV=1 so removing it"
    rm -rf .venv
  else
    echo "[bootstrap] Existing .venv looks invalid (missing .venv/bin/python)." >&2
    echo "[bootstrap] Remove .venv manually, or set ASYMDSD_RESET_VENV=1 to auto-reset." >&2
    exit 1
  fi
fi

if [[ ! -d .venv ]]; then
  echo "[bootstrap] Creating virtualenv (.venv)"
  uv venv
fi

venv_python="$repo_root/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "[bootstrap] Expected venv python at $venv_python but it was not found" >&2
  exit 1
fi

marker_file=".venv/.asymdsd_bootstrap_done"
if [[ -f "$marker_file" ]]; then
  echo "[bootstrap] Already bootstrapped (marker present); skipping installs"
  exit 0
fi

echo "[bootstrap] Installing Python dependencies"
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
uv pip install -e .

touch "$marker_file"
echo "[bootstrap] Done"