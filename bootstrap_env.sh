#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python/python3 not found"
  exit 1
fi

INSTALL_DEPS="${INSTALL_DEPS:-1}"
SKIP_ET1="${SKIP_ET1:-1}"
FORCE_DATA="${FORCE_DATA:-0}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
DATA_DIR="${DATA_DIR:-./data}"
DATA_SEED="${DATA_SEED:-42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-deps)
      INSTALL_DEPS="0"
      shift
      ;;
    --with-et1)
      SKIP_ET1="0"
      shift
      ;;
    --force-data)
      FORCE_DATA="1"
      shift
      ;;
    --et2-checkpoint)
      ET2_CHECKPOINT="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --data-seed)
      DATA_SEED="$2"
      shift 2
      ;;
    *)
      echo "[error] unknown option: $1"
      echo "usage: bash bootstrap_env.sh [--no-deps] [--with-et1] [--force-data] [--et2-checkpoint <path>] [--data-dir <dir>] [--data-seed <int>]"
      exit 1
      ;;
  esac
done

echo "[1/3] Environment dependencies"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -U pip setuptools wheel
  "$PYTHON_BIN" -m pip install -r requirements.txt
else
  echo "  - skipped (--no-deps)"
fi

echo "[2/3] ET models (ET2 auto-download if missing)"
ET_ARGS=(--skip-install --et2-checkpoint "$ET2_CHECKPOINT")
if [[ "$SKIP_ET1" == "1" ]]; then
  ET_ARGS+=(--skip-et1)
fi
"$PYTHON_BIN" setup_et_models.py "${ET_ARGS[@]}"

echo "[3/3] English dataset"
DATA_ARGS=(--output-dir "$DATA_DIR" --seed "$DATA_SEED")
if [[ "$FORCE_DATA" == "1" ]]; then
  DATA_ARGS+=(--force)
fi
"$PYTHON_BIN" prepare_english_data.py "${DATA_ARGS[@]}"

echo
echo "Done."
echo "Next:"
echo "python train_model.py xlmroberta-large mse --use-gaze-concat --et2-checkpoint $ET2_CHECKPOINT --features-used 1,1,1,1,1 --fp-dropout 0.1,0.3 --batch-size 8 --maxlen 200 --optim adamw_torch"
