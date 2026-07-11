#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_ROOT="${SMOKE_ROOT:-artifacts/postfix_concat_backbone_smoke}"
ASSET_ROOT="$SMOKE_ROOT/assets"
DATA_DIR="$ASSET_ROOT/data"

if [[ -z "$SMOKE_ROOT" || "$SMOKE_ROOT" == "/" ]]; then
  echo "Refusing unsafe SMOKE_ROOT: '$SMOKE_ROOT'" >&2
  exit 2
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled

"$PYTHON_BIN" scripts/create_gaze_smoke_assets.py --output-root "$ASSET_ROOT"

run_backbone() {
  local model_name="$1"
  local checkpoint="$2"
  local result_dir="$SMOKE_ROOT/results/$model_name"
  local log_path="$SMOKE_ROOT/${model_name}.log"
  rm -rf "$result_dir"
  rm -f "$log_path"
  mkdir -p "$result_dir"

  "$PYTHON_BIN" train_model.py "$model_name" mse \
    --checkpoint-override "$checkpoint" \
    --gaze-fusion postfix-concat \
    --et-model-type heuristic \
    --features-used 0,1,0,1,0 \
    --data-dir "$DATA_DIR" \
    --preds-dir "$result_dir" \
    --maxlen 24 \
    --batch-size 2 \
    --train-epochs 1 \
    --max-steps 1 \
    --save-strategy no \
    --no-save-final-model \
    --report-to none \
    --fp-dropout 0.0,0.0 \
    2>&1 | tee "$log_path"

  test -s "$result_dir/training_parameters.json"
  test -s "$result_dir/predictions_fold1.csv"
  test -s "$result_dir/predictions_fold2.csv"
  test -s "$result_dir/overall_metrics.json"
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["model"]==sys.argv[2]; assert p["gaze_fusion"]=="postfix-concat"' \
    "$result_dir/training_parameters.json" "$model_name"
}

run_backbone distilbert "$ASSET_ROOT/tiny_distilbert"
run_backbone xlmroberta-base "$ASSET_ROOT/tiny_encoder"
run_backbone xlmroberta-large "$ASSET_ROOT/tiny_xlm_roberta_large"

echo "Postfix concat passed both folds for all three model selectors."
