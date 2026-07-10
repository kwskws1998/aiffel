#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_ROOT="${SMOKE_ROOT:-artifacts/gaze_strategy_smoke}"
ASSET_ROOT="$SMOKE_ROOT/assets"
MODEL_DIR="$ASSET_ROOT/tiny_encoder"
DATA_DIR="$ASSET_ROOT/data"

if [[ -z "$SMOKE_ROOT" || "$SMOKE_ROOT" == "/" ]]; then
  echo "Refusing unsafe SMOKE_ROOT: '$SMOKE_ROOT'" >&2
  exit 2
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled

"$PYTHON_BIN" scripts/create_gaze_smoke_assets.py --output-root "$ASSET_ROOT"

rm -rf "$SMOKE_ROOT/results/cls_attention_bias"
rm -f "$SMOKE_ROOT/cls_attention_bias.log"

run_condition() {
  local name="$1"
  shift
  local result_dir="$SMOKE_ROOT/results/$name"
  local log_path="$SMOKE_ROOT/${name}.log"
  local sentinel="$result_dir/.smoke_complete"
  rm -rf "$result_dir"
  rm -f "$log_path"
  mkdir -p "$result_dir"

  "$PYTHON_BIN" train_model.py xlmroberta-base mse \
    --checkpoint-override "$MODEL_DIR" \
    --et-model-type heuristic \
    --features-used 0,1,0,1,0 \
    --data-dir "$DATA_DIR" \
    --preds-dir "$result_dir" \
    --maxlen 32 \
    --batch-size 2 \
    --train-epochs 1 \
    --max-steps 1 \
    --save-strategy no \
    --no-save-final-model \
    --report-to none \
    --gaze-hidden-size 8 \
    --gaze-num-heads 2 \
    --gaze-num-layers 1 \
    --gaze-alignment-dim 8 \
    --gaze-alignment-max-tokens 32 \
    --gaze-fusion-dropout 0.0 \
    "$@" 2>&1 | tee "$log_path"

  test -s "$result_dir/training_parameters.json"
  test -s "$result_dir/predictions_fold1.csv"
  test -s "$result_dir/predictions_fold2.csv"
  test -s "$result_dir/overall_metrics.json"
  printf '%s\n' "$name" > "$sentinel"
  test "$(<"$sentinel")" = "$name"
}

run_condition conditioned_pooling --gaze-fusion conditioned-pooling
run_condition postencoder_cls_attention_bias --gaze-fusion postencoder-cls-attention-bias
run_condition cross_attention --gaze-fusion cross-attention
run_condition auxiliary_only --gaze-aux-weight 0.1
run_condition alignment_only --gaze-alignment-weight 0.05

echo "All gaze strategy smoke runs passed. Results: $SMOKE_ROOT/results"
