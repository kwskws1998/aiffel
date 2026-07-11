#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_ROOT="${SMOKE_ROOT:-artifacts/gaze_strategy_smoke}"
MODEL_NAME="${MODEL_NAME:-xlmroberta-base}"
ASSET_ROOT="$SMOKE_ROOT/assets"
DATA_DIR="$ASSET_ROOT/data"

if [[ -z "$SMOKE_ROOT" || "$SMOKE_ROOT" == "/" ]]; then
  echo "Refusing unsafe SMOKE_ROOT: '$SMOKE_ROOT'" >&2
  exit 2
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled

if [[ "$MODEL_NAME" == "all" ]]; then
  for backbone in distilbert xlmroberta-base xlmroberta-large; do
    MODEL_NAME="$backbone" SMOKE_ROOT="$SMOKE_ROOT" bash "$0"
  done
  echo "All configured conditions passed for all three model selectors."
  exit 0
fi

"$PYTHON_BIN" scripts/create_gaze_smoke_assets.py --output-root "$ASSET_ROOT"

case "$MODEL_NAME" in
  distilbert)
    MODEL_DIR="$ASSET_ROOT/tiny_distilbert"
    ;;
  xlmroberta-base)
    MODEL_DIR="$ASSET_ROOT/tiny_encoder"
    ;;
  xlmroberta-large)
    MODEL_DIR="$ASSET_ROOT/tiny_xlm_roberta_large"
    ;;
  *)
    echo "Unsupported MODEL_NAME: $MODEL_NAME" >&2
    exit 2
    ;;
esac

run_condition() {
  local name="$1"
  shift
  local result_dir="$SMOKE_ROOT/results/$MODEL_NAME/$name"
  local log_path="$SMOKE_ROOT/logs/$MODEL_NAME/${name}.log"
  local sentinel="$result_dir/.smoke_complete"
  rm -rf "$result_dir"
  rm -f "$log_path"
  mkdir -p "$result_dir"
  mkdir -p "$(dirname "$log_path")"

  "$PYTHON_BIN" train_model.py "$MODEL_NAME" mse \
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
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["model"]==sys.argv[2]' \
    "$result_dir/training_parameters.json" "$MODEL_NAME"
  if [[ "$name" == "postfix_concat" ]]; then
    "$PYTHON_BIN" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); assert p["gaze_fusion"]=="postfix-concat"' \
      "$result_dir/training_parameters.json"
  elif [[ "$name" == "prefix_concat_legacy" ]]; then
    "$PYTHON_BIN" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); assert p["gaze_fusion"]=="prefix-concat"' \
      "$result_dir/training_parameters.json"
  fi
  printf '%s\n' "$name" > "$sentinel"
  test "$(<"$sentinel")" = "$name"
}

run_condition baseline
run_condition gaze_add --gaze-fusion add --gaze-add-scale 0.05
run_condition gaze_summary --gaze-fusion summary
run_condition conditioned_pooling --gaze-fusion conditioned-pooling
run_condition postencoder_cls_attention_bias --gaze-fusion postencoder-cls-attention-bias
run_condition cross_attention --gaze-fusion cross-attention
run_condition auxiliary_only --gaze-aux-weight 0.1
run_condition alignment_only --gaze-alignment-weight 0.05
run_condition postfix_concat --gaze-fusion postfix-concat
run_condition prefix_concat_legacy --gaze-fusion prefix-concat

echo "Baseline and all configured gaze-condition smoke runs passed for $MODEL_NAME. Results: $SMOKE_ROOT/results/$MODEL_NAME"
