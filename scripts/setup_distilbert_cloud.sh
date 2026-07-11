#!/usr/bin/env bash
# Cloud Linux GPU bootstrap that preserves the CUDA-enabled PyTorch from the image.
# Example: DATA_ZIP_FILE_ID=... bash scripts/setup_distilbert_cloud.sh

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/setup_distilbert_cloud.sh" \
    "" \
    "Run this inside the cloned repository on a CUDA-ready cloud image." \
    "The script keeps the image-provided torch and installs all other requirements." \
    "" \
    "Environment variables:" \
    "  PYTHON_CMD=python        CUDA-enabled base Python" \
    "  VENV_DIR=.venv          Venv created with --system-site-packages" \
    "  DATA_ZIP_FILE_ID=...    Authorized Google Drive zip id" \
    "  DATA_ZIP_URL=...        Authorized Google Drive share URL" \
    "  DATA_ZIP_SHA256=...     Optional expected checksum" \
    "  DATA_DIR=./data         Prepared fold directory" \
    "  ET2_CHECKPOINT=...      ET2 checkpoint base path" \
    "  PRELOAD_MODELS=0        Skip model cache preloading"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_CMD" >&2
  exit 2
fi

"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "The base cloud Python does not see CUDA"; print("base_torch", torch.__version__); print("base_cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0))'

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
fi

if [[ "$VENV_DIR" = /* ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
else
  PYTHON_BIN="$REPO_ROOT/$VENV_DIR/bin/python"
fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "The venv does not inherit CUDA torch; choose a new VENV_DIR"'
"$PYTHON_BIN" -m pip install -U pip setuptools wheel

TEMP_REQUIREMENTS="$(mktemp /tmp/va_gaze_requirements_no_torch.XXXXXX)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
sed '/^torch==/d' requirements.txt > "$TEMP_REQUIREMENTS"
"$PYTHON_BIN" -m pip install -r "$TEMP_REQUIREMENTS"

export PYTHON_BIN
export SKIP_DEPS=1
export WITH_ET1="${WITH_ET1:-0}"
export ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
export DATA_DIR="${DATA_DIR:-./data}"

bash install.sh

if [[ "$PRELOAD_MODELS" == "1" ]]; then
  "$PYTHON_BIN" -c 'from transformers import AutoModel,AutoTokenizer,RobertaModel,RobertaTokenizer; AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"); AutoModel.from_pretrained("distilbert-base-multilingual-cased"); RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True); RobertaModel.from_pretrained("roberta-base")'
fi

"$PYTHON_BIN" -c 'import torch,transformers,gdown; assert torch.cuda.is_available(); print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("transformers", transformers.__version__); print("gpu", torch.cuda.get_device_name(0))'

echo "Cloud GPU environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Start training with: REQUIRE_CUDA=1 bash scripts/run_distilbert_experiments.sh"
