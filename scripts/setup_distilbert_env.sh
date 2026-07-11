#!/usr/bin/env bash
# DistilBERT experiment environment bootstrap for this repository.
# Required for Drive data: DATA_ZIP_FILE_ID or DATA_ZIP_URL.
# Example: DATA_ZIP_FILE_ID=... DATA_ZIP_SHA256=... bash scripts/setup_distilbert_env.sh

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/setup_distilbert_env.sh" \
    "" \
    "Environment variables:" \
    "  PYTHON_CMD=python3       Python used to create the venv" \
    "  VENV_DIR=.venv          Virtual environment directory" \
    "  DATA_ZIP_FILE_ID=...    Authorized Google Drive zip file id" \
    "  DATA_ZIP_URL=...        Authorized Google Drive share URL" \
    "  DATA_ZIP_SHA256=...     Optional expected archive checksum" \
    "  DATA_DIR=./data         Fold-data output directory" \
    "  ET2_CHECKPOINT=...      ET2 checkpoint base path" \
    "  SKIP_DEPS=1             Skip dependency installation" \
    "  FORCE_DATA=1            Rebuild prepared fold files" \
    "  PRELOAD_MODELS=0        Skip DistilBERT/RoBERTa cache preloading"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_CMD" >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

if [[ "$VENV_DIR" = /* ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
else
  PYTHON_BIN="$REPO_ROOT/$VENV_DIR/bin/python"
fi
export PYTHON_BIN
export WITH_ET1="${WITH_ET1:-0}"
export ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
export DATA_DIR="${DATA_DIR:-./data}"

bash install.sh

if [[ "$PRELOAD_MODELS" == "1" ]]; then
  "$PYTHON_BIN" -c 'from transformers import AutoModel,AutoTokenizer,RobertaModel,RobertaTokenizer; AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"); AutoModel.from_pretrained("distilbert-base-multilingual-cased"); RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True); RobertaModel.from_pretrained("roberta-base")'
fi

"$PYTHON_BIN" -c 'import torch,transformers,gdown; print("torch", torch.__version__); print("transformers", transformers.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_devices", torch.cuda.device_count())'

echo "Environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run core experiments with: bash scripts/run_distilbert_experiments.sh"
echo "Run the full condition matrix with: CONDITIONS=all bash scripts/run_distilbert_experiments.sh"
