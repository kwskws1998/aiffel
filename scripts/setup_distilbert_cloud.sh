#!/usr/bin/env bash
# Backward-compatible cloud entrypoint; the implementation now uses Conda.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/scripts/setup_distilbert_conda_cloud.sh" "$@"
