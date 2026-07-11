#!/usr/bin/env bash
# Reproducible DistilBERT VA experiment runner.
# Groups: CONDITIONS=core, CONDITIONS=new, CONDITIONS=trt, CONDITIONS=all, or a list.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/run_distilbert_experiments.sh" \
    "" \
    "Condition groups:" \
    "  CONDITIONS=core   baseline, postfix FFD+TRT, GazeAdd FFD+TRT" \
    "  CONDITIONS=main   recommended ET2 matrix without legacy prefix" \
    "  CONDITIONS=new    baseline, postfix, summary, three post-encoder methods, two objectives" \
    "  CONDITIONS=trt    baseline and all non-legacy gaze strategies using TRT only" \
    "  CONDITIONS=all    every condition, including all-feature postfix and legacy prefix" \
    "  CONDITIONS=a,b    custom comma- or space-separated condition names" \
    "" \
    "Conditions:" \
    "  baseline postfix_all postfix_ffd_trt postfix_trt gaze_add_ffd_trt gaze_add_trt" \
    "  gaze_summary_ffd_trt conditioned_pooling cls_attention_bias" \
    "  cross_attention auxiliary_only alignment_only prefix_legacy" \
    "" \
    "Main variables:" \
    "  LOSS=mse BATCH_SIZE=8 MAXLEN=200 TRAIN_EPOCHS=10 SEED=42" \
    "  DATA_DIR=data ET_MODEL_TYPE=et2 ET2_CHECKPOINT=./checkpoints/et_predictor2_seed123" \
    "  ET_MODEL_TYPE=emotion-trt ET_MODEL_ID=skboy/emotion_trt_roberta_lr2e5_preval10" \
    "  OUTPUT_ROOT=Preds/distilbert_matrix_<timestamp>" \
    "  SAVE_STRATEGY=epoch SAVE_FINAL_MODEL=1 SKIP_COMPLETED=0 REQUIRE_CUDA=0 DRY_RUN=0"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  DEFAULT_PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x .venv/bin/python ]]; then
  DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
LOSS="${LOSS:-mse}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAXLEN="${MAXLEN:-200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-6e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-data}"
ET_MODEL_TYPE="${ET_MODEL_TYPE:-et2}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
ET_MODEL_ID="${ET_MODEL_ID:-}"
FP_DROPOUT="${FP_DROPOUT:-0.1,0.3}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
CONDITIONS="${CONDITIONS:-core}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Preds/distilbert_matrix_${RUN_TAG}}"
DISTILBERT_CHECKPOINT="${DISTILBERT_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

if [[ "$SKIP_COMPLETED" != "0" && "$SKIP_COMPLETED" != "1" ]]; then
  echo "SKIP_COMPLETED must be 0 or 1." >&2
  exit 2
fi

CORE_CONDITIONS=(baseline postfix_ffd_trt gaze_add_ffd_trt)
MAIN_CONDITIONS=(baseline postfix_all postfix_ffd_trt gaze_add_ffd_trt gaze_summary_ffd_trt conditioned_pooling cls_attention_bias cross_attention auxiliary_only alignment_only)
NEW_CONDITIONS=(baseline postfix_ffd_trt gaze_summary_ffd_trt conditioned_pooling cls_attention_bias cross_attention auxiliary_only alignment_only)
TRT_CONDITIONS=(baseline postfix_trt gaze_add_trt gaze_summary_trt conditioned_pooling_trt cls_attention_bias_trt cross_attention_trt auxiliary_only_trt alignment_only_trt)
ALL_CONDITIONS=(baseline postfix_all postfix_ffd_trt postfix_trt gaze_add_ffd_trt gaze_add_trt gaze_summary_ffd_trt gaze_summary_trt conditioned_pooling conditioned_pooling_trt cls_attention_bias cls_attention_bias_trt cross_attention cross_attention_trt auxiliary_only auxiliary_only_trt alignment_only alignment_only_trt prefix_legacy)

case "$CONDITIONS" in
  core)
    SELECTED_CONDITIONS=("${CORE_CONDITIONS[@]}")
    ;;
  main)
    SELECTED_CONDITIONS=("${MAIN_CONDITIONS[@]}")
    ;;
  new)
    SELECTED_CONDITIONS=("${NEW_CONDITIONS[@]}")
    ;;
  trt)
    SELECTED_CONDITIONS=("${TRT_CONDITIONS[@]}")
    ;;
  all)
    SELECTED_CONDITIONS=("${ALL_CONDITIONS[@]}")
    ;;
  *)
    read -r -a SELECTED_CONDITIONS <<< "${CONDITIONS//,/ }"
    ;;
esac

if [[ ${#SELECTED_CONDITIONS[@]} -eq 0 ]]; then
  echo "No experiment conditions selected." >&2
  exit 2
fi

for condition in "${SELECTED_CONDITIONS[@]}"; do
  if [[ "$condition" == postfix_* || "$condition" == prefix_legacy ]]; then
    if (( MAXLEN > 255 )); then
      echo "MAXLEN must be <= 255 when a concat condition is selected." >&2
      exit 2
    fi
  fi
done

case "$ET_MODEL_TYPE" in
  emotion_trt|emotion_trt_roberta|emotion-trt-roberta)
    ET_MODEL_TYPE="emotion-trt"
    ;;
  emotion_et)
    ET_MODEL_TYPE="emotion-et"
    ;;
esac

if [[ "$ET_MODEL_TYPE" != "et2" && "$ET_MODEL_TYPE" != "emotion-et" && "$ET_MODEL_TYPE" != "emotion-trt" && "$ET_MODEL_TYPE" != "heuristic" ]]; then
  echo "ET_MODEL_TYPE must be et2, emotion-et, emotion-trt, or heuristic." >&2
  exit 2
fi

if [[ "$ET_MODEL_TYPE" == "emotion-trt" && -z "$ET_MODEL_ID" ]]; then
  ET_MODEL_ID="skboy/emotion_trt_roberta_lr2e5_preval10"
elif [[ "$ET_MODEL_TYPE" == "emotion-et" && -z "$ET_MODEL_ID" ]]; then
  ET_MODEL_ID="skboy/emotion_et_model"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
  fi
  if [[ ! -s "$DATA_DIR/full_dataset_fold1.csv" || ! -s "$DATA_DIR/full_dataset_fold2.csv" ]]; then
    echo "Prepared fold files are missing under $DATA_DIR. Run scripts/setup_distilbert_env.sh first." >&2
    exit 2
  fi
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required but unavailable"; print("gpu", torch.cuda.get_device_name(0)); print("torch_cuda", torch.version.cuda)'
  fi
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

SAVE_ARGS=(--save-strategy "$SAVE_STRATEGY" --save-total-limit "$SAVE_TOTAL_LIMIT")
if [[ "$SAVE_FINAL_MODEL" == "0" ]]; then
  SAVE_ARGS+=(--no-save-final-model)
fi

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --batch-size "$BATCH_SIZE"
  --maxlen "$MAXLEN"
  --train-epochs "$TRAIN_EPOCHS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-ratio "$WARMUP_RATIO"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --seed "$SEED"
  --optim adamw_torch
  --report-to none
  "${SAVE_ARGS[@]}"
)
if [[ -n "$DISTILBERT_CHECKPOINT" ]]; then
  COMMON_ARGS+=(--checkpoint-override "$DISTILBERT_CHECKPOINT")
fi

ET_ARGS=(
  --et-model-type "$ET_MODEL_TYPE"
  --fp-dropout "$FP_DROPOUT"
)
if [[ "$ET_MODEL_TYPE" == "et2" ]]; then
  ET_ARGS+=(--et2-checkpoint "$ET2_CHECKPOINT")
elif [[ "$ET_MODEL_TYPE" == "emotion-et" || "$ET_MODEL_TYPE" == "emotion-trt" ]]; then
  ET_ARGS+=(--et-model-id "$ET_MODEL_ID")
fi

require_et_source() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  if [[ "$ET_MODEL_TYPE" != "et2" ]]; then
    return
  fi
  if [[ -f "$ET2_CHECKPOINT" || -f "$ET2_CHECKPOINT.safetensors" || -f "$ET2_CHECKPOINT.pt" || -f "$ET2_CHECKPOINT.bin" ]]; then
    return
  fi
  echo "ET2 checkpoint not found: $ET2_CHECKPOINT[.safetensors/.pt/.bin]" >&2
  exit 2
}

run_condition() {
  local condition="$1"
  shift
  local result_dir="$OUTPUT_ROOT/${condition}_seed${SEED}"
  local log_path="$OUTPUT_ROOT/logs/${condition}_seed${SEED}.log"
  local command=(
    "$PYTHON_BIN" train_model.py distilbert "$LOSS"
    --preds-dir "$result_dir"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi

  if [[ -e "$result_dir" ]]; then
    if [[ "$SKIP_COMPLETED" == "1" && -s "$result_dir/training_parameters.json" && -s "$result_dir/predictions_fold1.csv" && -s "$result_dir/predictions_fold2.csv" && -s "$result_dir/overall_metrics.json" ]]; then
      echo "Skipping completed result: $result_dir"
      return
    fi
    echo "Refusing to overwrite existing result directory: $result_dir" >&2
    exit 2
  fi
  mkdir -p "$result_dir" "$(dirname "$log_path")"
  "${command[@]}" 2>&1 | tee "$log_path"
  test -s "$result_dir/training_parameters.json"
  test -s "$result_dir/predictions_fold1.csv"
  test -s "$result_dir/predictions_fold2.csv"
  test -s "$result_dir/overall_metrics.json"
}

for condition in "${SELECTED_CONDITIONS[@]}"; do
  case "$condition" in
    baseline)
      run_condition "$condition"
      ;;
    postfix_all)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion postfix-concat --features-used 1,1,1,1,1
      ;;
    postfix_ffd_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion postfix-concat --features-used 0,1,0,1,0
      ;;
    postfix_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion postfix-concat --features-used 0,0,0,1,0
      ;;
    gaze_add_ffd_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion add --features-used 0,1,0,1,0 --gaze-add-scale 0.05
      ;;
    gaze_add_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion add --features-used 0,0,0,1,0 --gaze-add-scale 0.05
      ;;
    gaze_summary_ffd_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion summary --features-used 0,1,0,1,0
      ;;
    gaze_summary_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion summary --features-used 0,0,0,1,0
      ;;
    conditioned_pooling)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion conditioned-pooling --features-used 0,1,0,1,0
      ;;
    conditioned_pooling_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion conditioned-pooling --features-used 0,0,0,1,0
      ;;
    cls_attention_bias)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion postencoder-cls-attention-bias --features-used 0,1,0,1,0
      ;;
    cls_attention_bias_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion postencoder-cls-attention-bias --features-used 0,0,0,1,0
      ;;
    cross_attention)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion cross-attention --features-used 0,1,0,1,0
      ;;
    cross_attention_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion cross-attention --features-used 0,0,0,1,0
      ;;
    auxiliary_only)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --features-used 0,1,0,1,0 --gaze-aux-weight 0.1
      ;;
    auxiliary_only_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --features-used 0,0,0,1,0 --gaze-aux-weight 0.1
      ;;
    alignment_only)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --features-used 0,1,0,1,0 --gaze-alignment-weight 0.05
      ;;
    alignment_only_trt)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --features-used 0,0,0,1,0 --gaze-alignment-weight 0.05
      ;;
    prefix_legacy)
      require_et_source
      run_condition "$condition" "${ET_ARGS[@]}" --gaze-fusion prefix-concat --features-used 0,1,0,1,0
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      exit 2
      ;;
  esac
done

echo "Completed DistilBERT conditions: ${SELECTED_CONDITIONS[*]}"
echo "Results: $OUTPUT_ROOT"
