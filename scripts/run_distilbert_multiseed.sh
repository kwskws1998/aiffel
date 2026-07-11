#!/usr/bin/env bash
# Multi-seed orchestration for the DistilBERT condition runner.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/run_distilbert_multiseed.sh" \
    "" \
    "Defaults:" \
    "  SEEDS=42,123,2025" \
    "  CONDITIONS=main" \
    "  ET_MODEL_TYPE=et2" \
    "  RESUME=1" \
    "" \
    "Examples:" \
    "  REQUIRE_CUDA=1 bash scripts/run_distilbert_multiseed.sh" \
    "  SEEDS=42,43,44 CONDITIONS=core bash scripts/run_distilbert_multiseed.sh" \
    "  ET_MODEL_TYPE=emotion-trt CONDITIONS=trt bash scripts/run_distilbert_multiseed.sh"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${SEEDS:-42,123,2025}"
CONDITIONS="${CONDITIONS:-main}"
ET_MODEL_TYPE="${ET_MODEL_TYPE:-et2}"
ET_MODEL_ID="${ET_MODEL_ID:-}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAFE_ET_TYPE="${ET_MODEL_TYPE//[^A-Za-z0-9_-]/_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Preds/distilbert_multiseed_${SAFE_ET_TYPE}_${RUN_TAG}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi

read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
if [[ ${#SEED_VALUES[@]} -lt 2 ]]; then
  echo "SEEDS must contain at least two integer seeds." >&2
  exit 2
fi

SEEN_SEEDS=" "
for seed in "${SEED_VALUES[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
  if [[ "$SEEN_SEEDS" == *" $seed "* ]]; then
    echo "Duplicate seed: $seed" >&2
    exit 2
  fi
  SEEN_SEEDS+="$seed "
done

echo "DistilBERT multi-seed run"
echo "  seeds: ${SEED_VALUES[*]}"
echo "  conditions: $CONDITIONS"
echo "  ET model: $ET_MODEL_TYPE"
echo "  output: $OUTPUT_ROOT"

for seed in "${SEED_VALUES[@]}"; do
  echo "Starting downstream seed $seed"
  SEED="$seed" \
  CONDITIONS="$CONDITIONS" \
  ET_MODEL_TYPE="$ET_MODEL_TYPE" \
  ET_MODEL_ID="$ET_MODEL_ID" \
  ET2_CHECKPOINT="$ET2_CHECKPOINT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RUN_TAG="$RUN_TAG" \
  SKIP_COMPLETED="$RESUME" \
  DRY_RUN="$DRY_RUN" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_distilbert_experiments.sh
done

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" scripts/summarize_distilbert_multiseed.py \
    --output-root "$OUTPUT_ROOT" \
    --expected-seeds "${SEED_VALUES[*]}"
fi

echo "Completed DistilBERT multi-seed matrix: $OUTPUT_ROOT"
