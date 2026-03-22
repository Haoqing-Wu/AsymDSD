#!/usr/bin/env bash
set -euo pipefail

# Keep devcontainer startup fast by default.
# Set ASYMDSD_BOOTSTRAP=1 (in your local env before container creation)
# to run the full bootstrap automatically.

# Ensure the required Python is available.
uv python install 3.11

if [[ "${ASYMDSD_BOOTSTRAP:-}" != "1" ]]; then
  echo "[postCreate] ASYMDSD_BOOTSTRAP not set to 1; skipping bootstrap"
  exit 0
fi

exec bash .devcontainer/bootstrap.sh
