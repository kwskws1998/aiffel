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
    "  LOSS=mse" \
    "  METRIC_FOR_BEST_MODEL=ccc_mean" \
    "  HETERO_MSE_WEIGHT=0.1 HETERO_CCC_WEIGHT=0.1" \
    "  HETERO_LOGVAR_MIN=-5.0 HETERO_LOGVAR_MAX=3.0" \
    "  ET_MODEL_TYPE=et2" \
    "  RESUME=1" \
    "" \
    "Examples:" \
    "  REQUIRE_CUDA=1 bash scripts/run_distilbert_multiseed.sh" \
    "  SEEDS=42,43,44 CONDITIONS=core bash scripts/run_distilbert_multiseed.sh" \
    "  LOSS=hetero+ccc CONDITIONS=main bash scripts/run_distilbert_multiseed.sh" \
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
LOSS="${LOSS:-mse}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-ccc_mean}"
HETERO_MSE_WEIGHT="${HETERO_MSE_WEIGHT:-0.1}"
HETERO_CCC_WEIGHT="${HETERO_CCC_WEIGHT:-0.1}"
HETERO_LOGVAR_MIN="${HETERO_LOGVAR_MIN:--5.0}"
HETERO_LOGVAR_MAX="${HETERO_LOGVAR_MAX:-3.0}"
ET_MODEL_TYPE="${ET_MODEL_TYPE:-et2}"
ET_MODEL_ID="${ET_MODEL_ID:-}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAFE_ET_TYPE="${ET_MODEL_TYPE//[^A-Za-z0-9_-]/_}"
SAFE_LOSS="${LOSS//[^A-Za-z0-9_+-]/_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Preds/distilbert_multiseed_${SAFE_ET_TYPE}_${SAFE_LOSS}_${RUN_TAG}}"

case "$LOSS" in
  mse|ccc|robust|mse+ccc|robust+ccc|hetero|hetero+ccc)
    ;;
  *)
    echo "LOSS must be mse, ccc, robust, mse+ccc, robust+ccc, hetero, or hetero+ccc." >&2
    exit 2
    ;;
esac

case "$METRIC_FOR_BEST_MODEL" in
  loss|mse_mean|ccc_mean|pearson_corr_mean|gaussian_nll_mean)
    ;;
  *)
    echo "METRIC_FOR_BEST_MODEL must be loss, mse_mean, ccc_mean, pearson_corr_mean, or gaussian_nll_mean." >&2
    exit 2
    ;;
esac

if [[ "$METRIC_FOR_BEST_MODEL" == "gaussian_nll_mean" && "$LOSS" != "hetero" && "$LOSS" != "hetero+ccc" ]]; then
  echo "METRIC_FOR_BEST_MODEL=gaussian_nll_mean requires LOSS=hetero or LOSS=hetero+ccc." >&2
  exit 2
fi

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
echo "  loss: $LOSS"
echo "  best-model metric: $METRIC_FOR_BEST_MODEL"
echo "  ET model: $ET_MODEL_TYPE"
echo "  output: $OUTPUT_ROOT"

for seed in "${SEED_VALUES[@]}"; do
  echo "Starting downstream seed $seed"
  SEED="$seed" \
  CONDITIONS="$CONDITIONS" \
  LOSS="$LOSS" \
  METRIC_FOR_BEST_MODEL="$METRIC_FOR_BEST_MODEL" \
  HETERO_MSE_WEIGHT="$HETERO_MSE_WEIGHT" \
  HETERO_CCC_WEIGHT="$HETERO_CCC_WEIGHT" \
  HETERO_LOGVAR_MIN="$HETERO_LOGVAR_MIN" \
  HETERO_LOGVAR_MAX="$HETERO_LOGVAR_MAX" \
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
