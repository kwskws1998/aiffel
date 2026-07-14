#!/usr/bin/env bash
# Launch four XLM-R-large heteroscedastic VA workers under one tmux-pane supervisor.

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
    "  PRELOAD_MODELS=1 HF_HUB_DOWNLOAD_TIMEOUT=600" \
    "  HF_HUB_DISABLE_XET=1" \
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

prepare_worker_command() {
  local gpu="$1"
  local seed="$2"
  local fold="$3"
  local run_id="$4"
  WORKER_COMMAND=(
    env
    PYTHONUNBUFFERED=1
    TOKENIZERS_PARALLELISM=false
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
    "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
    "HF_HUB_ETAG_TIMEOUT=$HF_HUB_ETAG_TIMEOUT"
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
}

serialize_worker_command() {
  prepare_worker_command "$@"
  serialize_command "${WORKER_COMMAND[@]}"
}

prefix_lines() {
  local label="$1"
  local line

  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '[%s] %s\n' "$label" "$line"
  done
}

run_worker() {
  local gpu="$1"
  local seed="$2"
  local train_fold="$3"
  local test_fold="$4"
  local run_id="$5"
  local log_file="$6"
  local status_file="$7"
  local label="GPU${gpu} seed${seed} trainF${train_fold} testF${test_fold}"
  local command_status
  local logger_status=0
  local stage_status
  local -a pipeline_status

  prepare_worker_command "$gpu" "$seed" "$train_fold" "$run_id"
  : > "$log_file"
  printf '[%s] [start] run=%s\n' "$label" "$run_id" | tee -a "$log_file"

  set +e
  "${WORKER_COMMAND[@]}" 2>&1 \
    | tee -a "$log_file" \
    | tr '\r' '\n' \
    | prefix_lines "$label"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e

  command_status="${pipeline_status[0]}"
  for stage_status in "${pipeline_status[@]:1}"; do
    if [[ "$stage_status" -ne 0 ]]; then
      logger_status=125
    fi
  done

  printf '[%s] [finished] python_exit_code=%s logger_exit_code=%s\n' \
    "$label" "$command_status" "$logger_status" \
    | tee -a "$log_file"
  {
    printf 'python_exit_code\tlogger_exit_code\n'
    printf '%s\t%s\n' "$command_status" "$logger_status"
  } > "${status_file}.tmp"
  mv "${status_file}.tmp" "$status_file"

  if [[ "$command_status" -ne 0 ]]; then
    return "$command_status"
  fi
  return "$logger_status"
}

verify_run_outputs() {
  local run_id="$1"
  local filename
  local missing=0
  local -a expected_files=(
    training_parameters.json
    training_parameters_fold1.json
    training_parameters_fold2.json
    predictions_fold1.csv
    predictions_fold2.csv
    fold1_metrics.csv
    fold2_metrics.csv
    all_predictions.csv
    overall_metrics.csv
    overall_metrics.json
    dataset_metrics.csv
    uncertainty_calibration.csv
    uncertainty_risk_coverage.csv
  )

  for filename in "${expected_files[@]}"; do
    if [[ ! -s "$REPO_ROOT/Preds/$run_id/$filename" ]]; then
      printf '[supervisor] missing result: Preds/%s/%s\n' "$run_id" "$filename"
      missing=1
    fi
  done
  return "$missing"
}

preload_models() {
  printf '[supervisor] preloading XLM-R-large and RoBERTa-base once before workers start\n'
  env \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET" \
    "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT" \
    "HF_HUB_ETAG_TIMEOUT=$HF_HUB_ETAG_TIMEOUT" \
    "$PYTHON_BIN" - <<'PY'
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

def load_with_cache_repair(loader, model_name, **kwargs):
    try:
        return loader.from_pretrained(model_name, **kwargs)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"cache load failed for {model_name}; retrying one forced download: {error}",
            flush=True,
        )
        return loader.from_pretrained(model_name, force_download=True, **kwargs)


xlmr_tokenizer = load_with_cache_repair(AutoTokenizer, "xlm-roberta-large")
xlmr_model = load_with_cache_repair(AutoModel, "xlm-roberta-large")
roberta_tokenizer = load_with_cache_repair(
    RobertaTokenizer,
    "roberta-base",
    add_prefix_space=True,
)
roberta_model = load_with_cache_repair(RobertaModel, "roberta-base")
del xlmr_tokenizer, xlmr_model, roberta_tokenizer, roberta_model

xlmr_tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large",
    local_files_only=True,
)
xlmr_model = AutoModel.from_pretrained("xlm-roberta-large", local_files_only=True)
roberta_tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base", add_prefix_space=True, local_files_only=True
)
roberta_model = RobertaModel.from_pretrained("roberta-base", local_files_only=True)
del xlmr_tokenizer, xlmr_model, roberta_tokenizer, roberta_model
print("model_cache_preload_and_local_validation=ok")
PY
}

run_supervisor() {
  local overall_status=0
  local index
  local wait_code
  local -a pids
  local -a wait_codes
  local -a labels=(
    "GPU0 seed${SEED_A} trainF1 testF2"
    "GPU1 seed${SEED_A} trainF2 testF1"
    "GPU2 seed${SEED_B} trainF1 testF2"
    "GPU3 seed${SEED_B} trainF2 testF1"
  )

  printf '[supervisor] session=%s\n' "$SESSION_NAME"
  printf '[supervisor] combined_log=%s/combined.log\n' "$LOG_ROOT"
  if [[ "$PRELOAD_MODELS" == "1" ]]; then
    if ! preload_models; then
      printf '[supervisor] model preload failed; no training workers were started\n'
      return 1
    fi
  fi

  run_worker 0 "$SEED_A" 1 2 "$RUN_A" \
    "$LOG_ROOT/gpu0_seed${SEED_A}_fold1.log" \
    "$LOG_ROOT/gpu0_seed${SEED_A}_fold1.status.tsv" &
  pids[0]=$!
  run_worker 1 "$SEED_A" 2 1 "$RUN_A" \
    "$LOG_ROOT/gpu1_seed${SEED_A}_fold2.log" \
    "$LOG_ROOT/gpu1_seed${SEED_A}_fold2.status.tsv" &
  pids[1]=$!
  run_worker 2 "$SEED_B" 1 2 "$RUN_B" \
    "$LOG_ROOT/gpu2_seed${SEED_B}_fold1.log" \
    "$LOG_ROOT/gpu2_seed${SEED_B}_fold1.status.tsv" &
  pids[2]=$!
  run_worker 3 "$SEED_B" 2 1 "$RUN_B" \
    "$LOG_ROOT/gpu3_seed${SEED_B}_fold2.log" \
    "$LOG_ROOT/gpu3_seed${SEED_B}_fold2.status.tsv" &
  pids[3]=$!

  set +e
  for index in 0 1 2 3; do
    wait "${pids[$index]}"
    wait_code=$?
    wait_codes[$index]="$wait_code"
    if [[ "$wait_code" -ne 0 ]]; then
      overall_status=1
    fi
  done

  {
    printf 'worker\twait_exit_code\n'
    for index in 0 1 2 3; do
      printf '%s\t%s\n' "${labels[$index]}" "${wait_codes[$index]}"
    done
  } > "$LOG_ROOT/exit_codes.tsv.tmp"
  mv "$LOG_ROOT/exit_codes.tsv.tmp" "$LOG_ROOT/exit_codes.tsv"

  if [[ "$overall_status" -eq 0 ]]; then
    verify_run_outputs "$RUN_A" || overall_status=1
    verify_run_outputs "$RUN_B" || overall_status=1
  fi
  if [[ "$overall_status" -eq 0 ]]; then
    printf '[supervisor] all four workers and both OOF reports completed successfully\n'
  else
    printf '[supervisor] failure detected; inspect %s/exit_codes.tsv and worker logs\n' \
      "$LOG_ROOT"
  fi
  return "$overall_status"
}

run_supervisor_entrypoint() {
  local status

  set +e
  run_supervisor
  status=$?
  set -e
  printf '%s\n' "$status" > "$LOG_ROOT/supervisor.exit_code"
  printf '[supervisor] exit_code=%s; pane will remain open\n' "$status"
  exec /bin/bash -l
}

build_supervisor_command() {
  local variable_name
  local -a command=(
    env
    VA_GAZE_SUPERVISOR_MODE=1
    "RUN_TAG=$RUN_TAG"
    "SESSION_NAME=$SESSION_NAME"
    "DATA_DIR=$DATA_DIR"
    "ET2_CHECKPOINT=$ET2_CHECKPOINT"
    "LOG_ROOT=$LOG_ROOT"
    "PYTHON_BIN=$PYTHON_BIN"
    "SEED_A=$SEED_A"
    "SEED_B=$SEED_B"
    "BATCH_SIZE=$BATCH_SIZE"
    "TRAIN_EPOCHS=$TRAIN_EPOCHS"
    "MAXLEN=$MAXLEN"
    "LEARNING_RATE=$LEARNING_RATE"
    "WEIGHT_DECAY=$WEIGHT_DECAY"
    "WARMUP_RATIO=$WARMUP_RATIO"
    "HETERO_MSE_WEIGHT=$HETERO_MSE_WEIGHT"
    "HETERO_CCC_WEIGHT=$HETERO_CCC_WEIGHT"
    "HETERO_LOGVAR_MIN=$HETERO_LOGVAR_MIN"
    "HETERO_LOGVAR_MAX=$HETERO_LOGVAR_MAX"
    "PRELOAD_MODELS=$PRELOAD_MODELS"
    "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
    "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
    "HF_HUB_ETAG_TIMEOUT=$HF_HUB_ETAG_TIMEOUT"
    "REPORT_TO=$REPORT_TO"
    ATTACH=0
    DRY_RUN=0
  )

  for variable_name in HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE; do
    if [[ -n "${!variable_name:-}" ]]; then
      command+=("$variable_name=${!variable_name}")
    fi
  done
  command+=("${BASH:-/bin/bash}" "$REPO_ROOT/scripts/run_xlmr_hetero_4gpu_tmux.sh")
  serialize_command "${command[@]}"
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
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
REPORT_TO="${REPORT_TO:-none}"
ATTACH="${ATTACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
SUPERVISOR_MODE="${VA_GAZE_SUPERVISOR_MODE:-0}"

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
require_positive_integer "HF_HUB_DOWNLOAD_TIMEOUT" "$HF_HUB_DOWNLOAD_TIMEOUT"
require_positive_integer "HF_HUB_ETAG_TIMEOUT" "$HF_HUB_ETAG_TIMEOUT"
[[ "$PRELOAD_MODELS" == "0" || "$PRELOAD_MODELS" == "1" ]] || fail "PRELOAD_MODELS must be 0 or 1."
[[ "$HF_HUB_DISABLE_XET" == "0" || "$HF_HUB_DISABLE_XET" == "1" ]] || fail "HF_HUB_DISABLE_XET must be 0 or 1."
[[ "$ATTACH" == "0" || "$ATTACH" == "1" ]] || fail "ATTACH must be 0 or 1."
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1."
[[ "$SUPERVISOR_MODE" == "0" || "$SUPERVISOR_MODE" == "1" ]] || fail "VA_GAZE_SUPERVISOR_MODE must be 0 or 1."

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

if [[ "$SUPERVISOR_MODE" == "1" ]]; then
  [[ -d "$LOG_ROOT" ]] || fail "Supervisor log directory is missing: $LOG_ROOT"
  run_supervisor_entrypoint
fi

COMMAND_0="$(serialize_worker_command 0 "$SEED_A" 1 "$RUN_A")"
COMMAND_1="$(serialize_worker_command 1 "$SEED_A" 2 "$RUN_A")"
COMMAND_2="$(serialize_worker_command 2 "$SEED_B" 1 "$RUN_B")"
COMMAND_3="$(serialize_worker_command 3 "$SEED_B" 2 "$RUN_B")"
SUPERVISOR_COMMAND="$(build_supervisor_command)"

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
  "  tmux layout: one pane with four prefixed workers" \
  "  model preload: $PRELOAD_MODELS" \
  "  best checkpoint: ccc_mean (maximize)"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '\nGPU 0 / seed %s / fold 1:\n%s\n' "$SEED_A" "$COMMAND_0"
  printf '\nGPU 1 / seed %s / fold 2:\n%s\n' "$SEED_A" "$COMMAND_1"
  printf '\nGPU 2 / seed %s / fold 1:\n%s\n' "$SEED_B" "$COMMAND_2"
  printf '\nGPU 3 / seed %s / fold 2:\n%s\n' "$SEED_B" "$COMMAND_3"
  printf '\nSingle-pane supervisor command:\n%s\n' "$SUPERVISOR_COMMAND"
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

sys.path.insert(0, str(Path.cwd() / "src"))

import accelerate
import numpy as np
import pandas as pd
import robust_loss_pytorch
import scipy
import sklearn
import torch
import transformers

from va_gaze.models.et2_wrapper import FixationsPredictor_2


class BFloat16Et2Probe:
    def __call__(self, input_ids, attention_mask, predict_mask):
        batch_size, sequence_length = input_ids.shape
        return torch.ones(
            (batch_size, sequence_length, 5),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )


et2_probe = object.__new__(FixationsPredictor_2)
et2_probe.model = BFloat16Et2Probe()
probe_input_ids = torch.ones((1, 8), dtype=torch.long)
probe_predictions = et2_probe._sliding_window_predict(
    probe_input_ids,
    torch.ones_like(probe_input_ids),
)
if probe_predictions.dtype != np.float32:
    raise SystemExit(
        f"[error] ET2 BF16 NumPy boundary returned {probe_predictions.dtype}, expected float32."
    )
print("et2_bfloat16_numpy_preflight=ok")

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
  "preload_models=$PRELOAD_MODELS" \
  "hf_hub_disable_xet=$HF_HUB_DISABLE_XET" \
  "hf_hub_download_timeout=$HF_HUB_DOWNLOAD_TIMEOUT" \
  "hf_hub_etag_timeout=$HF_HUB_ETAG_TIMEOUT" \
  "command_gpu0=$COMMAND_0" \
  "command_gpu1=$COMMAND_1" \
  "command_gpu2=$COMMAND_2" \
  "command_gpu3=$COMMAND_3" \
  "supervisor_command=$SUPERVISOR_COMMAND" \
  > "$LOG_ROOT/launch_manifest.txt"

SESSION_CREATED=0
trap cleanup_partial_session EXIT
tmux new-session -d -s "$SESSION_NAME" -n train /bin/bash
SESSION_CREATED=1
PANE="$(tmux display-message -p -t "$SESSION_NAME:train.0" '#{pane_id}')"
printf -v PIPE_COMMAND 'cat >> %q' "$LOG_ROOT/combined.log"
tmux select-pane -t "$PANE" -T "4-GPU heteroscedastic supervisor"
tmux pipe-pane -o -t "$PANE" "$PIPE_COMMAND"
tmux send-keys -t "$PANE" -l "$SUPERVISOR_COMMAND"
tmux send-keys -t "$PANE" Enter
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
