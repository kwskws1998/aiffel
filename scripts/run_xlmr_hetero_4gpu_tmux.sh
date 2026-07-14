#!/usr/bin/env bash
# Launch XLM-R-large heteroscedastic VA training as two seeds by two folds in four tmux panes.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/run_xlmr_hetero_4gpu_tmux.sh" \
    "" \
    "Experiment fixed by default:" \
    "  model=xlmroberta-large" \
    "  loss=hetero+ccc" \
    "  gaze=ET2 postfix-concat, TRT only" \
    "  data=./data_no_iemocap" \
    "  seeds=42,43; folds=1,2; GPUs=0,1,2,3" \
    "  checkpoint selection=full-fold ccc_mean (maximize)" \
    "" \
    "Useful overrides:" \
    "  RUN_TAG=... SESSION_NAME=... PYTHON_BIN=..." \
    "  DATA_DIR=... ET2_CHECKPOINT=... LOG_ROOT=..." \
    "  BATCH_SIZE=16 TRAIN_EPOCHS=10 LEARNING_RATE=6e-6" \
    "  HETERO_MSE_WEIGHT=0.1 HETERO_CCC_WEIGHT=0.1" \
    "  MIN_FREE_GB=60 MIN_GPU_FREE_MIB=28000" \
    "  REPORT_TO=none ATTACH=1 DRY_RUN=0" \
    "" \
    "Examples:" \
    "  bash scripts/run_xlmr_hetero_4gpu_tmux.sh" \
    "  RUN_TAG=main_hetero_ccc bash scripts/run_xlmr_hetero_4gpu_tmux.sh" \
    "  DRY_RUN=1 bash scripts/run_xlmr_hetero_4gpu_tmux.sh"
}

fail() {
  printf '[error] %s\n' "$1" >&2
  exit 2
}

resolve_python() {
  local requested_python="${PYTHON_BIN:-}"

  if [[ -n "$requested_python" ]]; then
    if [[ "$requested_python" == */* ]]; then
      printf '%s\n' "$requested_python"
    else
      command -v "$requested_python"
    fi
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    printf '%s\n' "$CONDA_PREFIX/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    printf '%s\n' "$VIRTUAL_ENV/bin/python"
  else
    command -v python
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || fail "$name must be a positive integer."
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a nonnegative integer."
}

cleanup_partial_session() {
  local status=$?
  if [[ "$status" -ne 0 && "${SESSION_CREATED:-0}" == "1" ]]; then
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  fi
  return "$status"
}

checkpoint_exists() {
  local checkpoint_base="$1"
  [[ -f "$checkpoint_base" \
    || -f "${checkpoint_base}.safetensors" \
    || -f "${checkpoint_base}.pt" \
    || -f "${checkpoint_base}.bin" ]]
}

serialize_command() {
  local output=""
  printf -v output '%q ' "$@"
  printf '%s' "$output"
}

build_pane_command() {
  local gpu="$1"
  local seed="$2"
  local fold="$3"
  local run_id="$4"
  local serialized
  local -a command=(
    env
    PYTHONUNBUFFERED=1
    TOKENIZERS_PARALLELISM=false
    "CUDA_VISIBLE_DEVICES=$gpu"
    "$PYTHON_BIN"
    "$REPO_ROOT/train_model.py"
    xlmroberta-large
    hetero+ccc
    --fold "$fold"
    --run-id "$run_id"
    --preds-dir "$REPO_ROOT/Preds/$run_id"
    --data-dir "$DATA_DIR"
    --et-model-type et2
    --et2-checkpoint "$ET2_CHECKPOINT"
    --gaze-transform raw
    --gaze-fusion postfix-concat
    --features-used 0,0,0,1,0
    --fp-dropout 0.1,0.3
    --batch-size "$BATCH_SIZE"
    --train-epochs "$TRAIN_EPOCHS"
    --maxlen "$MAXLEN"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --hetero-mse-weight "$HETERO_MSE_WEIGHT"
    --hetero-ccc-weight "$HETERO_CCC_WEIGHT"
    --hetero-logvar-min "$HETERO_LOGVAR_MIN"
    --hetero-logvar-max "$HETERO_LOGVAR_MAX"
    --metric-for-best-model ccc_mean
    --save-strategy epoch
    --save-total-limit 1
    --load-best-model-at-end
    --save-final-model
    --optim adamw_torch
    --gradient-accumulation-steps 1
    --bf16
    --gradient-checkpointing
    --seed "$seed"
    --report-to "$REPORT_TO"
  )

  serialized="$(serialize_command "${command[@]}")"
  printf 'cd %q && printf "[start] gpu=%s seed=%s fold=%s run=%s\\n" && %s; status=$?; printf "\\n[finished] gpu=%s seed=%s fold=%s exit_code=%%s\\n" "$status"; exec /bin/bash -l' \
    "$REPO_ROOT" "$gpu" "$seed" "$fold" "$run_id" "$serialized" "$gpu" "$seed" "$fold"
}

configure_pane() {
  local pane_id="$1"
  local title="$2"
  local log_file="$3"
  local pane_command="$4"
  local pipe_command

  printf -v pipe_command 'cat >> %q' "$log_file"
  tmux select-pane -t "$pane_id" -T "$title"
  tmux pipe-pane -o -t "$pane_id" "$pipe_command"
  tmux send-keys -t "$pane_id" -l "$pane_command"
  tmux send-keys -t "$pane_id" Enter
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG//[^A-Za-z0-9_-]/_}"
SESSION_NAME="${SESSION_NAME:-xlmr_hetero_ccc_$RUN_TAG}"
SESSION_NAME="${SESSION_NAME//[^A-Za-z0-9_-]/_}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data_no_iemocap}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-$REPO_ROOT/checkpoints/et_predictor2_seed123}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/$SESSION_NAME}"
PYTHON_BIN="$(resolve_python)"
SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-43}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
MAXLEN="${MAXLEN:-200}"
LEARNING_RATE="${LEARNING_RATE:-6e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
HETERO_MSE_WEIGHT="${HETERO_MSE_WEIGHT:-0.1}"
HETERO_CCC_WEIGHT="${HETERO_CCC_WEIGHT:-0.1}"
HETERO_LOGVAR_MIN="${HETERO_LOGVAR_MIN:--5}"
HETERO_LOGVAR_MAX="${HETERO_LOGVAR_MAX:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-28000}"
REPORT_TO="${REPORT_TO:-none}"
ATTACH="${ATTACH:-1}"
DRY_RUN="${DRY_RUN:-0}"

RUN_A="xlmr_large_postfix_et2_trt_no_iemocap_hetero_ccc_s${SEED_A}_bs${BATCH_SIZE}_${RUN_TAG}"
RUN_B="xlmr_large_postfix_et2_trt_no_iemocap_hetero_ccc_s${SEED_B}_bs${BATCH_SIZE}_${RUN_TAG}"

[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
[[ "$SEED_A" =~ ^[0-9]+$ ]] || fail "SEED_A must be an integer."
[[ "$SEED_B" =~ ^[0-9]+$ ]] || fail "SEED_B must be an integer."
[[ "$SEED_A" != "$SEED_B" ]] || fail "SEED_A and SEED_B must differ."
require_positive_integer "BATCH_SIZE" "$BATCH_SIZE"
require_positive_integer "TRAIN_EPOCHS" "$TRAIN_EPOCHS"
require_positive_integer "MAXLEN" "$MAXLEN"
require_nonnegative_integer "MIN_FREE_GB" "$MIN_FREE_GB"
require_nonnegative_integer "MIN_GPU_FREE_MIB" "$MIN_GPU_FREE_MIB"
[[ "$ATTACH" == "0" || "$ATTACH" == "1" ]] || fail "ATTACH must be 0 or 1."
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1."

"$PYTHON_BIN" - \
  "$LEARNING_RATE" \
  "$WEIGHT_DECAY" \
  "$WARMUP_RATIO" \
  "$HETERO_MSE_WEIGHT" \
  "$HETERO_CCC_WEIGHT" \
  "$HETERO_LOGVAR_MIN" \
  "$HETERO_LOGVAR_MAX" <<'PY'
import math
import sys

names = (
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "WARMUP_RATIO",
    "HETERO_MSE_WEIGHT",
    "HETERO_CCC_WEIGHT",
    "HETERO_LOGVAR_MIN",
    "HETERO_LOGVAR_MAX",
)
values = []
for name, raw_value in zip(names, sys.argv[1:]):
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SystemExit(f"[error] {name} must be numeric: {raw_value}") from error
    if not math.isfinite(value):
        raise SystemExit(f"[error] {name} must be finite: {raw_value}")
    values.append(value)

(
    learning_rate,
    weight_decay,
    warmup_ratio,
    mse_weight,
    ccc_weight,
    logvar_min,
    logvar_max,
) = values
if learning_rate <= 0:
    raise SystemExit("[error] LEARNING_RATE must be positive.")
if weight_decay < 0:
    raise SystemExit("[error] WEIGHT_DECAY must be nonnegative.")
if not 0 <= warmup_ratio <= 1:
    raise SystemExit("[error] WARMUP_RATIO must be between 0 and 1.")
if mse_weight < 0 or ccc_weight < 0:
    raise SystemExit("[error] Heteroscedastic auxiliary weights must be nonnegative.")
if logvar_min >= logvar_max:
    raise SystemExit("[error] HETERO_LOGVAR_MIN must be smaller than HETERO_LOGVAR_MAX.")
PY

COMMAND_0="$(build_pane_command 0 "$SEED_A" 1 "$RUN_A")"
COMMAND_1="$(build_pane_command 1 "$SEED_A" 2 "$RUN_A")"
COMMAND_2="$(build_pane_command 2 "$SEED_B" 1 "$RUN_B")"
COMMAND_3="$(build_pane_command 3 "$SEED_B" 2 "$RUN_B")"

printf '%s\n' \
  "XLM-R-large heteroscedastic four-GPU run" \
  "  session: $SESSION_NAME" \
  "  python: $PYTHON_BIN" \
  "  data: $DATA_DIR" \
  "  ET2 checkpoint: $ET2_CHECKPOINT" \
  "  seed $SEED_A output: Preds/$RUN_A" \
  "  seed $SEED_B output: Preds/$RUN_B" \
  "  logs: $LOG_ROOT" \
  "  loss: hetero+ccc" \
  "  best checkpoint: ccc_mean (maximize)"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '\nGPU 0 / seed %s / fold 1:\n%s\n' "$SEED_A" "$COMMAND_0"
  printf '\nGPU 1 / seed %s / fold 2:\n%s\n' "$SEED_A" "$COMMAND_1"
  printf '\nGPU 2 / seed %s / fold 1:\n%s\n' "$SEED_B" "$COMMAND_2"
  printf '\nGPU 3 / seed %s / fold 2:\n%s\n' "$SEED_B" "$COMMAND_3"
  exit 0
fi

command -v tmux >/dev/null 2>&1 || fail "tmux is not installed."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable."
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Unset CUDA_VISIBLE_DEVICES before launching; this script maps physical GPUs 0-3."
if [[ -n "${VIRTUAL_ENV:-}" && -n "${CONDA_PREFIX:-}" && "$VIRTUAL_ENV" != "$CONDA_PREFIX" ]]; then
  fail "Both virtualenv and Conda are active. Deactivate the virtualenv and keep only the Conda environment."
fi
[[ -f "$DATA_DIR/full_dataset_fold1.csv" ]] || fail "Missing $DATA_DIR/full_dataset_fold1.csv"
[[ -f "$DATA_DIR/full_dataset_fold2.csv" ]] || fail "Missing $DATA_DIR/full_dataset_fold2.csv"
checkpoint_exists "$ET2_CHECKPOINT" || fail "Missing ET2 checkpoint: $ET2_CHECKPOINT(.safetensors/.pt/.bin)"
tmux has-session -t "$SESSION_NAME" 2>/dev/null && fail "tmux session already exists: $SESSION_NAME"
[[ ! -e "$LOG_ROOT" ]] || fail "Log directory already exists: $LOG_ROOT"

for run_id in "$RUN_A" "$RUN_B"; do
  [[ ! -e "$REPO_ROOT/Preds/$run_id" ]] || fail "Prediction output already exists: Preds/$run_id"
  [[ ! -e "$REPO_ROOT/Output Directory/$run_id" ]] || fail "Checkpoint output already exists: Output Directory/$run_id"
  [[ ! -e "$REPO_ROOT/model/$run_id" ]] || fail "Final model output already exists: model/$run_id"
done

GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || fail "Could not determine CUDA GPU count: $GPU_COUNT"
(( GPU_COUNT >= 4 )) || fail "Four CUDA GPUs are required; detected $GPU_COUNT."

GPU_FREE_MIB_VALUES=()
while IFS= read -r free_mib; do
  GPU_FREE_MIB_VALUES+=("${free_mib//[[:space:]]/}")
done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( ${#GPU_FREE_MIB_VALUES[@]} >= 4 )) || fail "nvidia-smi reported fewer than four GPUs."
for gpu in 0 1 2 3; do
  free_mib="${GPU_FREE_MIB_VALUES[$gpu]}"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || fail "Could not parse free memory for GPU $gpu: $free_mib"
  (( free_mib >= MIN_GPU_FREE_MIB )) || fail "GPU $gpu has ${free_mib} MiB free; ${MIN_GPU_FREE_MIB} MiB is required."
done

AVAILABLE_KB="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
REQUIRED_KB="$((MIN_FREE_GB * 1024 * 1024))"
(( AVAILABLE_KB >= REQUIRED_KB )) || fail "At least ${MIN_FREE_GB} GiB free is required in the repository filesystem."

"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path

import accelerate
import numpy as np
import pandas as pd
import robust_loss_pytorch
import scipy
import sklearn
import torch
import transformers

data_dir = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable in the selected Python environment.")
if torch.cuda.device_count() < 4:
    raise SystemExit(
        f"[error] Four CUDA GPUs are required; detected {torch.cuda.device_count()}."
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("[error] The selected CUDA device does not support bfloat16.")

required_columns = {"index", "text", "dataset_of_origin", "valence", "arousal"}
for fold in (1, 2):
    path = data_dir / f"full_dataset_fold{fold}.csv"
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise SystemExit(f"[error] {path} is missing columns: {missing_columns}")
    if frame.empty:
        raise SystemExit(f"[error] {path} is empty.")
    origins = frame["dataset_of_origin"].astype(str)
    if origins.str.contains("IEMOCAP", case=False, regex=False).any():
        raise SystemExit(f"[error] IEMOCAP rows remain in {path}.")
    labels = frame[["valence", "arousal"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(labels.to_numpy(dtype=np.float64)).all():
        raise SystemExit(f"[error] {path} contains invalid VA labels.")
    print(f"data_fold{fold}_rows={len(frame)}")

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
for gpu in range(4):
    print(f"gpu{gpu}={torch.cuda.get_device_name(gpu)}")
print("python_cuda_data_preflight=ok")
PY

mkdir -p "$LOG_ROOT"
printf '%s\n' \
  "session=$SESSION_NAME" \
  "run_tag=$RUN_TAG" \
  "run_seed_${SEED_A}=$RUN_A" \
  "run_seed_${SEED_B}=$RUN_B" \
  "git_commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)" \
  "python=$PYTHON_BIN" \
  "data_dir=$DATA_DIR" \
  "et2_checkpoint=$ET2_CHECKPOINT" \
  "loss=hetero+ccc" \
  "metric_for_best_model=ccc_mean" \
  "hetero_mse_weight=$HETERO_MSE_WEIGHT" \
  "hetero_ccc_weight=$HETERO_CCC_WEIGHT" \
  "hetero_logvar_min=$HETERO_LOGVAR_MIN" \
  "hetero_logvar_max=$HETERO_LOGVAR_MAX" \
  "command_gpu0=$COMMAND_0" \
  "command_gpu1=$COMMAND_1" \
  "command_gpu2=$COMMAND_2" \
  "command_gpu3=$COMMAND_3" \
  > "$LOG_ROOT/launch_manifest.txt"

SESSION_CREATED=0
trap cleanup_partial_session EXIT
tmux new-session -d -s "$SESSION_NAME" -n train /bin/bash
SESSION_CREATED=1
PANE_0="$(tmux display-message -p -t "$SESSION_NAME:train.0" '#{pane_id}')"
PANE_1="$(tmux split-window -h -P -F '#{pane_id}' -t "$PANE_0" /bin/bash)"
PANE_2="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_0" /bin/bash)"
PANE_3="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_1" /bin/bash)"
tmux select-layout -t "$SESSION_NAME:train" tiled >/dev/null
tmux set-window-option -t "$SESSION_NAME:train" pane-border-status top >/dev/null
tmux set-window-option \
  -t "$SESSION_NAME:train" \
  pane-border-format ' #{pane_index}: #{pane_title} ' \
  >/dev/null

configure_pane "$PANE_0" "GPU0 seed${SEED_A} fold1" "$LOG_ROOT/gpu0_seed${SEED_A}_fold1.log" "$COMMAND_0"
configure_pane "$PANE_1" "GPU1 seed${SEED_A} fold2" "$LOG_ROOT/gpu1_seed${SEED_A}_fold2.log" "$COMMAND_1"
configure_pane "$PANE_2" "GPU2 seed${SEED_B} fold1" "$LOG_ROOT/gpu2_seed${SEED_B}_fold1.log" "$COMMAND_2"
configure_pane "$PANE_3" "GPU3 seed${SEED_B} fold2" "$LOG_ROOT/gpu3_seed${SEED_B}_fold2.log" "$COMMAND_3"
SESSION_CREATED=0
trap - EXIT

printf 'Started tmux session: %s\n' "$SESSION_NAME"
printf 'Attach: tmux attach -t %s\n' "$SESSION_NAME"
printf 'Detach: Ctrl-b d\n'
printf 'Reattach: tmux attach -t %s\n' "$SESSION_NAME"

if [[ "$ATTACH" == "1" ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION_NAME"
  else
    tmux attach-session -t "$SESSION_NAME"
  fi
fi
